"""
feature_matching.py — FGPL-style camera pose estimation pipeline

Stages:
  1. Load per-view pixel lines, back-project to unit sphere
  2. Load 3D wireframe from room_geometry.pkl
  3. Extract principal directions (2D and 3D) via icosphere voting
  4. Enumerate 24 rotation candidates, score, pick best R
  5. Classify lines by principal direction, compute intersections
  6. Generate translation candidate grid
  7. Coarse XDF cost search over (R, t) grid
  8. ICP refinement via PnL_solver
  9. Save camera_pose.json
"""

import json
import pickle
import re
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import open3d as o3d

# Allow importing from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from PnL_solver import refine_pose_icp

# ============================================================
# CONFIGURATION
# ============================================================
POINT_CLOUD_NAME = "tmb_office1"
ROOM_NAME        = "TMB_office1"
CROP_W = CROP_H  = 1024
CROP_FOV_DEG     = 60.0
MIN_LINE_LEN_PX  = 30.0   # discard very short pixel-space lines


# ============================================================
# STAGE 1 — Per-view pixel lines → unit sphere
# ============================================================

def _focal(w=CROP_W, fov_deg=CROP_FOV_DEG):
    return 0.5 * w / np.tan(np.radians(fov_deg / 2.0))


def pixel_to_ray(u, v, W=CROP_W, H=CROP_H, fov_deg=CROP_FOV_DEG):
    """Map pixel (u, v) to unit camera-frame ray."""
    f  = _focal(W, fov_deg)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    ray = np.array([(u - cx) / f,
                    -(v - cy) / f,   # image Y↓ → camera Y↑
                    1.0])
    return ray / np.linalg.norm(ray)


def parse_view_angles(filename):
    """Return (yaw_rad, pitch_rad) parsed from filename like '00_yaw0.0_pitch-45.0.jpg'."""
    m = re.search(r'yaw(-?\d+(?:\.\d+)?)_pitch(-?\d+(?:\.\d+)?)', filename)
    return np.radians(float(m.group(1))), np.radians(float(m.group(2)))


def make_view_rotation(yaw_rad, pitch_rad):
    """
    Build rotation matrix: camera frame → pano sphere frame.

    Matches pano_processing_virtual_camerasV2.py exactly:
        phi_rad = -pitch_rad   (note sign flip)
        Rx = rotation around X by phi_rad
        Ry = rotation around Y by yaw_rad
        R  = Ry @ Rx
        world_ray = R @ camera_ray
    Sphere convention: X=right, Y=up, Z=forward (at yaw=0, pitch=0).
    """
    phi_rad   = -pitch_rad           # sign flip to match renderer
    theta_rad = yaw_rad

    Rx = np.array([[1,  0,              0],
                   [0,  np.cos(phi_rad), -np.sin(phi_rad)],
                   [0,  np.sin(phi_rad),  np.cos(phi_rad)]])

    Ry = np.array([[ np.cos(theta_rad), 0, np.sin(theta_rad)],
                   [ 0,                 1, 0],
                   [-np.sin(theta_rad), 0, np.cos(theta_rad)]])

    return Ry @ Rx


def sphere_line_to_9vec(s, e):
    """Two unit sphere points → [normal(3), start(3), end(3)] or None if degenerate."""
    normal = np.cross(s, e)
    n = np.linalg.norm(normal)
    if n < 1e-6:
        return None
    return np.concatenate([normal / n, s, e])


def load_edge_2d(json_path, min_len_px=MIN_LINE_LEN_PX):
    """
    Load extracted_2d_lines.json and back-project all line segments to
    the unit sphere. Returns (N, 9) ndarray [normal, start, end].
    """
    with open(json_path) as f:
        data = json.load(f)

    vecs = []
    for filename, lines in data.items():
        yaw, pitch = parse_view_angles(filename)
        R = make_view_rotation(yaw, pitch)
        for pt1, pt2 in lines:
            u1, v1 = pt1
            u2, v2 = pt2
            if np.hypot(u2 - u1, v2 - v1) < min_len_px:
                continue
            s = R @ pixel_to_ray(u1, v1)
            e = R @ pixel_to_ray(u2, v2)
            vec = sphere_line_to_9vec(s, e)
            if vec is not None:
                vecs.append(vec)

    return np.array(vecs)   # (N, 9)


# ============================================================
# STAGE 2 — Load 3D wireframe
# ============================================================

def load_3d_lines(pkl_path):
    """
    Load wireframe_segments from room_geometry.pkl.
    Returns (starts, ends, dirs, lengths) all as ndarrays.
    """
    with open(pkl_path, 'rb') as f:
        geom = pickle.load(f)

    segs    = geom['wireframe_segments']
    starts  = np.array([s['start'] for s in segs], dtype=float)
    ends    = np.array([s['end']   for s in segs], dtype=float)
    raw_dirs = ends - starts
    lengths  = np.linalg.norm(raw_dirs, axis=1)

    valid   = lengths > 1e-6
    starts, ends = starts[valid], ends[valid]
    dirs    = raw_dirs[valid] / lengths[valid, None]
    lengths = lengths[valid]
    return starts, ends, dirs, lengths


# ============================================================
# STAGE 3 — Principal direction extraction (icosphere voting)
# ============================================================

def _build_icosphere(subdivisions=5):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh = mesh.subdivide_loop(number_of_iterations=subdivisions)
    pts  = np.asarray(mesh.vertices)
    # Keep upper hemisphere only (mirrors FGPL)
    return pts[pts[:, 1] >= 0]   # Y-up pano frame; keep Y >= 0


def _suppress_vote(remaining, sphere_pts, direction, suppress_deg=30):
    cos_thresh = np.cos(np.radians(suppress_deg))
    remaining[np.abs(sphere_pts @ direction) > cos_thresh] = 0.0


def extract_principal_3d(dirs_sparse, sphere_pts, suppress_deg=30):
    """Vote on icosphere to find 3 dominant 3D line directions."""
    votes     = np.abs(sphere_pts @ dirs_sparse.T).max(axis=1)
    remaining = votes.copy()
    principal = []
    for _ in range(3):
        idx = int(np.argmax(remaining))
        d   = sphere_pts[idx]
        principal.append(d)
        _suppress_vote(remaining, sphere_pts, d, suppress_deg)
    return np.array(principal)   # (3, 3)


def extract_principal_2d(edge_2d, sphere_pts, perp_thres=0.05, suppress_deg=30):
    """
    Vote on icosphere to find 3 dominant 2D line directions (vanishing points).

    A sphere point p is 'on' a great circle with normal n if |p·n| < perp_thres
    (i.e., p is nearly perpendicular to n). Counts how many line great circles
    pass through each sphere point — matches FGPL's extract_principal_2d logic.
    """
    normals   = edge_2d[:, :3]
    # (Q, N) — True where sphere_pt lies on/near the great circle
    votes     = (np.abs(sphere_pts @ normals.T) < perp_thres).sum(axis=1).astype(float)
    remaining = votes.copy()
    principal = []
    for _ in range(3):
        idx = int(np.argmax(remaining))
        d   = sphere_pts[idx]
        principal.append(d)
        _suppress_vote(remaining, sphere_pts, d, suppress_deg)
    return np.array(principal)   # (3, 3)


# ============================================================
# STAGE 4 — 24 rotation candidates
# ============================================================

def enumerate_rotation_candidates(principal_2d, principal_3d):
    """Enumerate up to 48 R matrices aligning principal_2d to principal_3d."""
    candidates = []
    sign_combos = [(s0, s1, s2)
                   for s0 in [1, -1]
                   for s1 in [1, -1]
                   for s2 in [1, -1]]
    for perm in permutations([0, 1, 2]):
        P = principal_2d[list(perm), :]
        for signs in sign_combos:
            A = np.diag(signs) @ P
            H = principal_3d.T @ A
            U, _, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            R = Vt.T @ np.diag([1, 1, d]) @ U.T
            candidates.append(R)
    return candidates


def score_rotation(R, edge_2d, principal_3d, inlier_thres=0.15):
    """Count 2D great-circle normals aligned with any rotated 3D principal dir."""
    p3d_in_2d = (R @ principal_3d.T).T   # (3, 3): 3D principals in sphere frame
    normals   = edge_2d[:, :3]            # (N, 3)
    score = 0
    for d in p3d_in_2d:
        score += int(np.sum(np.abs(normals @ d) > 1.0 - inlier_thres))
    return score


# ============================================================
# STAGE 5 — Line classification & intersections
# ============================================================

def _sph_dist(a, b):
    return np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))


def _on_arc(d, s, e, thres=0.12):
    return abs(_sph_dist(d, s) + _sph_dist(d, e) - _sph_dist(s, e)) < thres


def compute_2d_intersections(edge_2d, principal_2d, inlier_thres=0.5, arc_thres=0.12):
    """
    Classify 2D lines by principal direction then find great-circle intersections.
    Returns list of 3 ndarrays (M_k, 3).
    """
    normals = edge_2d[:, :3]
    starts  = edge_2d[:, 3:6]
    ends    = edge_2d[:, 6:9]

    # Classify: line i belongs to group j if |normal_i . principal_j| > inlier_thres
    groups = []
    for d in principal_2d:
        groups.append(np.where(np.abs(normals @ d) > inlier_thres)[0])

    inter = []
    for (gi, gj) in [(0, 1), (1, 2), (2, 0)]:
        pts = []
        for a in groups[gi]:
            for b in groups[gj]:
                cand = np.cross(normals[a], normals[b])
                cn   = np.linalg.norm(cand)
                if cn < 1e-6:
                    continue
                for sign in [1, -1]:
                    d = sign * cand / cn
                    if _on_arc(d, starts[a], ends[a], arc_thres) and \
                       _on_arc(d, starts[b], ends[b], arc_thres):
                        pts.append(d)
        inter.append(np.array(pts) if pts else np.zeros((0, 3)))
    return inter


def _closest_point_3d(p1, d1, p2, d2, max_gap=0.35):
    """Midpoint of closest approach between two 3D lines, or None if too far apart."""
    w = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e    = d1 @ w, d2 @ w
    denom   = a * c - b * b
    if abs(denom) < 1e-8:
        return None
    sc = (b * e - c * d) / denom
    tc = (a * e - b * d) / denom
    pt1 = p1 + sc * d1
    pt2 = p2 + tc * d2
    return (pt1 + pt2) / 2 if np.linalg.norm(pt1 - pt2) < max_gap else None


def compute_3d_intersections(starts, ends, dirs, principal_3d,
                             inlier_thres=0.1, seg_thres=0.25):
    """
    Classify 3D lines by principal direction then find closest-point intersections.
    Returns list of 3 ndarrays (N_k, 3).
    """
    lengths = np.linalg.norm(ends - starts, axis=1)
    groups  = []
    for d in principal_3d:
        groups.append(np.where(np.abs(dirs @ d) > 1.0 - inlier_thres)[0])

    inter = []
    for (gi, gj) in [(0, 1), (1, 2), (2, 0)]:
        pts = []
        for a in groups[gi]:
            for b in groups[gj]:
                pt = _closest_point_3d(starts[a], dirs[a], starts[b], dirs[b])
                if pt is None:
                    continue
                # check pt is near both segments (extended by seg_thres)
                if (np.linalg.norm(pt - starts[a]) <= lengths[a] + seg_thres and
                        np.linalg.norm(pt - starts[b]) <= lengths[b] + seg_thres):
                    pts.append(pt)
        inter.append(np.array(pts) if pts else np.zeros((0, 3)))
    return inter


# ============================================================
# STAGE 6 — Translation candidate grid
# ============================================================

def generate_translation_candidates(starts, ends, spacing=0.5):
    """
    Regular XY grid across the room bounding box at estimated head height.
    Room uses Z-up: Z range determines height.
    """
    pts     = np.vstack([starts, ends])
    mins    = pts.min(0)
    maxs    = pts.max(0)
    xs      = np.arange(mins[0], maxs[0], spacing)
    ys      = np.arange(mins[1], maxs[1], spacing)
    z_cam   = mins[2] + 0.45 * (maxs[2] - mins[2])   # ~45% of height range
    return np.array([[x, y, z_cam] for x in xs for y in ys])


# ============================================================
# STAGE 7 — Coarse XDF pose search
# ============================================================

def _build_query_sphere(subdivisions=1):
    """Low-res icosphere for XDF integration (~42 pts at level 1)."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh = mesh.subdivide_loop(number_of_iterations=subdivisions)
    return np.asarray(mesh.vertices)


def xdf_cost(R, t, edge_2d, starts_3d, ends_3d, dirs_3d,
             inter_2d, inter_3d, query_pts, point_gamma=0.2):
    """
    Scalar cost (lower = better) comparing 2D and 3D distance functions
    over a low-res query sphere.
    """
    # --- Project 3D lines to camera sphere ---
    starts_cam = (starts_3d - t) @ R.T
    ends_cam   = (ends_3d   - t) @ R.T
    ns_start   = np.linalg.norm(starts_cam, axis=1, keepdims=True)
    ns_end     = np.linalg.norm(ends_cam,   axis=1, keepdims=True)
    valid      = (ns_start[:, 0] > 1e-6) & (ns_end[:, 0] > 1e-6)

    if valid.sum() == 0:
        return np.inf

    s_sph = starts_cam[valid] / ns_start[valid]
    e_sph = ends_cam[valid]   / ns_end[valid]
    normals_3d_cam = np.cross(s_sph, e_sph)
    nn = np.linalg.norm(normals_3d_cam, axis=1, keepdims=True)
    good = nn[:, 0] > 1e-6
    normals_3d_cam[good] /= nn[good]
    normals_3d_cam = normals_3d_cam[good]

    # --- Line distance functions ---
    normals_2d = edge_2d[:, :3]
    # dist(q, GC) = |π/2 - arccos(|q·n|)|
    cos2d = np.clip(np.abs(query_pts @ normals_2d.T), 0.0, 1.0)
    dist_2d = np.abs(np.pi / 2 - np.arccos(cos2d)).min(axis=1)  # (Q,)

    if normals_3d_cam.shape[0] == 0:
        return np.inf
    cos3d = np.clip(np.abs(query_pts @ normals_3d_cam.T), 0.0, 1.0)
    dist_3d = np.abs(np.pi / 2 - np.arccos(cos3d)).min(axis=1)  # (Q,)

    cost = float(np.abs(dist_2d - dist_3d).mean())

    # --- Intersection point distance functions ---
    for k in range(3):
        i2d = inter_2d[k]
        i3d = inter_3d[k]
        if len(i2d) == 0 or len(i3d) == 0:
            continue
        i3d_cam = (i3d - t) @ R.T
        ni = np.linalg.norm(i3d_cam, axis=1, keepdims=True)
        valid_i = ni[:, 0] > 1e-6
        if not valid_i.any():
            continue
        i3d_sph = i3d_cam[valid_i] / ni[valid_i]

        cos_2d = np.clip(query_pts @ i2d.T, -1.0, 1.0)
        cos_3d = np.clip(query_pts @ i3d_sph.T, -1.0, 1.0)
        pd2d = np.arccos(cos_2d).min(axis=1)
        pd3d = np.arccos(cos_3d).min(axis=1)
        cost += float((np.abs(pd2d - pd3d) ** point_gamma).mean())

    return cost


# ============================================================
# MAIN
# ============================================================

def main():
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent

    pkl_path  = project_root / "data" / "debug_renderer" / POINT_CLOUD_NAME / "room_geometry.pkl"
    json_path = project_root / "data" / "pano" / "2d_feature_extracted" / ROOM_NAME / "extracted_2d_lines.json"

    print("[1] Loading 2D lines and back-projecting to unit sphere...")
    edge_2d = load_edge_2d(json_path)
    print(f"    edge_2d shape: {edge_2d.shape}")

    print("[2] Loading 3D wireframe...")
    starts, ends, dirs, lengths = load_3d_lines(pkl_path)
    print(f"    wireframe segments: {len(starts)}")

    # Length filters (relative to room extent)
    pts   = np.vstack([starts, ends])
    extent = np.linalg.norm(pts.max(0) - pts.min(0))
    sparse_mask = lengths >= 0.10 * extent
    dense_mask  = lengths >= 0.20

    starts_s, ends_s = starts[sparse_mask], ends[sparse_mask]
    dirs_s           = dirs[sparse_mask]
    starts_d, ends_d = starts[dense_mask], ends[dense_mask]
    dirs_d, lengths_d = dirs[dense_mask], lengths[dense_mask]
    print(f"    sparse (>10% extent): {sparse_mask.sum()}  "
          f"dense (>0.2m): {dense_mask.sum()}")

    print("[3] Extracting principal directions...")
    sphere_pts_hi = _build_icosphere(subdivisions=5)
    principal_3d  = extract_principal_3d(dirs_s, sphere_pts_hi)
    principal_2d  = extract_principal_2d(edge_2d, sphere_pts_hi)
    print(f"    principal_3d:\n{principal_3d}")
    print(f"    principal_2d:\n{principal_2d}")

    print("[4] Enumerating rotation candidates...")
    candidates = enumerate_rotation_candidates(principal_2d, principal_3d)
    best_R  = max(candidates,
                  key=lambda R: score_rotation(R, edge_2d, principal_3d))
    best_score = score_rotation(best_R, edge_2d, principal_3d)
    print(f"    best rotation score: {best_score}  det={np.linalg.det(best_R):.4f}")

    print("[5] Computing intersections...")
    inter_2d = compute_2d_intersections(edge_2d, principal_2d)
    inter_3d = compute_3d_intersections(starts_d, ends_d, dirs_d, principal_3d)
    for k in range(3):
        print(f"    inter_2d[{k}]: {len(inter_2d[k])}  "
              f"inter_3d[{k}]: {len(inter_3d[k])}")

    print("[6] Generating translation candidates...")
    t_candidates = generate_translation_candidates(starts, ends, spacing=0.5)
    print(f"    {len(t_candidates)} translation candidates")

    print("[7] Coarse XDF search...")
    query_pts = _build_query_sphere(subdivisions=1)
    best_cost = np.inf
    best_t    = t_candidates[0]

    for t_cand in t_candidates:
        cost = xdf_cost(best_R, t_cand, edge_2d,
                        starts_d, ends_d, dirs_d,
                        inter_2d, inter_3d, query_pts)
        if cost < best_cost:
            best_cost = cost
            best_t    = t_cand.copy()

    print(f"    best XDF cost: {best_cost:.4f}  t={best_t}")

    print("[8] ICP refinement...")
    final_R, final_t = refine_pose_icp(
        best_R, best_t, inter_2d, inter_3d,
        n_iters=100, lr=0.05, nn_dist_thres=0.5)
    print(f"    refined t={final_t}")
    print(f"    det(R)={np.linalg.det(final_R):.4f}")

    print("[9] Saving camera_pose.json...")
    output_dir = project_root / "data" / "pose_estimates" / ROOM_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "rotation":          final_R.tolist(),
        "translation":       final_t.tolist(),
        "principal_3d":      principal_3d.tolist(),
        "principal_2d":      principal_2d.tolist(),
        "n_inter_matched":   [int(len(m)) for m in inter_2d],
        "xdf_cost_coarse":   float(best_cost),
    }
    out_path = output_dir / "camera_pose.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"    Saved to {out_path}")

    # --- Sanity checks ---
    print("\n--- Sanity checks ---")
    dots = [float(principal_2d[i] @ principal_2d[j])
            for i in range(3) for j in range(3) if i != j]
    print(f"principal_2d orthogonality (all should be ~0): {[f'{d:.3f}' for d in dots]}")
    dots3 = [float(principal_3d[i] @ principal_3d[j])
             for i in range(3) for j in range(3) if i != j]
    print(f"principal_3d orthogonality:                    {[f'{d:.3f}' for d in dots3]}")
    print(f"n_inter_matched:    {result['n_inter_matched']} (want >5 each)")
    tx, ty, tz = final_t
    mins = pts.min(0)
    maxs = pts.max(0)
    inside = (mins[0] <= tx <= maxs[0] and
              mins[1] <= ty <= maxs[1] and
              mins[2] <= tz <= maxs[2])
    print(f"translation in room bbox: {inside}  t=({tx:.2f}, {ty:.2f}, {tz:.2f})")
    print(f"det(R)={np.linalg.det(final_R):.4f}  (want +1.0)")


if __name__ == "__main__":
    main()
