"""
feature_matchingV2.py — FGPL-faithful camera pose estimation pipeline

Mirrors the panoramic-localization FGPL pipeline using PyTorch tensors throughout,
canonical frame precomputation, proper XDF inlier-counting cost, and FGPL-style
intersection matching.

Stages:
  1. Load 3D wireframe from room_geometry.pkl
  2. Load per-view pixel lines, back-project to unit sphere
  3. Extract principal directions (3D and 2D) via icosphere voting
  4. Build 24 rotation candidates (6 perm × 4 sign, determinant-preserving)
  5. Compute line intersections (2D on sphere, 3D in world space)
  6. Generate translation candidate grid with chamfer filtering
  7. XDF coarse search via canonical precomputation + inlier counting
  8. Full ICP refinement (R + t) via PnL_solver.refine_pose_full
  9. Save camera_pose.json
"""

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PnL_solver import refine_pose_full

# ============================================================
# CONFIGURATION
# ============================================================
POINT_CLOUD_NAME = "tmb_office1"
ROOM_NAME        = "TMB_office1"
CROP_W = CROP_H  = 1024
CROP_FOV_DEG     = 60.0
MIN_LINE_LEN_PX  = 30.0

# Algorithm constants
VOTE_SPHERE_LEVEL  = 5      # icosphere for voting (~2562 upper-hemisphere pts)
QUERY_SPHERE_LEVEL = 3      # icosphere for XDF queries
INLIER_THRES_2D    = 0.05   # line classification threshold (2D: |n·p| < thres)
INLIER_THRES_3D    = 0.05   # line classification threshold (3D: |d·p| > 1-thres)
INTERSECT_THRES_2D = 0.1    # arc membership tolerance
INTERSECT_THRES_3D = 0.2    # closest-point distance threshold
XDF_INLIER_THRES   = 0.1    # inlier counting threshold
POINT_GAMMA        = 0.2    # PDF power
CHAMFER_MIN_DIST   = 0.3    # translation filter
TRANS_SPACING      = 0.5    # translation grid spacing


# ============================================================
# ICOSPHERE GENERATION (replaces Open3D dependency)
# Ported from panoramic-localization/edge_utils.py:1235-1288
# ============================================================

def icosahedron2sphere(level):
    """Generate icosphere vertices by recursive subdivision of an icosahedron."""
    a = 2 / (1 + np.sqrt(5))
    M = np.array([
        0, a, -1, a, 1, 0, -a, 1, 0,
        0, a, 1, -a, 1, 0, a, 1, 0,
        0, a, 1, 0, -a, 1, -1, 0, a,
        0, a, 1, 1, 0, a, 0, -a, 1,
        0, a, -1, 0, -a, -1, 1, 0, -a,
        0, a, -1, -1, 0, -a, 0, -a, -1,
        0, -a, 1, a, -1, 0, -a, -1, 0,
        0, -a, -1, -a, -1, 0, a, -1, 0,
        -a, 1, 0, -1, 0, a, -1, 0, -a,
        -a, -1, 0, -1, 0, -a, -1, 0, a,
        a, 1, 0, 1, 0, -a, 1, 0, a,
        a, -1, 0, 1, 0, a, 1, 0, -a,
        0, a, 1, -1, 0, a, -a, 1, 0,
        0, a, 1, a, 1, 0, 1, 0, a,
        0, a, -1, -a, 1, 0, -1, 0, -a,
        0, a, -1, 1, 0, -a, a, 1, 0,
        0, -a, -1, -1, 0, -a, -a, -1, 0,
        0, -a, -1, a, -1, 0, 1, 0, -a,
        0, -a, 1, -a, -1, 0, -1, 0, a,
        0, -a, 1, 1, 0, a, a, -1, 0])

    coor = M.T.reshape(3, 60, order='F').T
    coor, idx = np.unique(coor, return_inverse=True, axis=0)
    tri = idx.reshape(3, 20, order='F').T
    coor = list(coor / np.linalg.norm(coor, axis=1, keepdims=True))

    for _ in range(level):
        triN = []
        for t in range(len(tri)):
            n = len(coor)
            coor.append((coor[tri[t, 0]] + coor[tri[t, 1]]) / 2)
            coor.append((coor[tri[t, 1]] + coor[tri[t, 2]]) / 2)
            coor.append((coor[tri[t, 2]] + coor[tri[t, 0]]) / 2)
            triN.append([n, tri[t, 0], n+2])
            triN.append([n, tri[t, 1], n+1])
            triN.append([n+1, tri[t, 2], n+2])
            triN.append([n, n+1, n+2])
        tri = np.array(triN)
        coor, idx = np.unique(coor, return_inverse=True, axis=0)
        tri = idx[tri]
        coor = list(coor / np.sqrt(np.sum(coor * coor, 1, keepdims=True)))

    return np.array(coor)


def generate_sphere_pts(level, device='cpu'):
    """Generate icosphere points as a torch tensor."""
    pts = icosahedron2sphere(level)
    return torch.from_numpy(pts).float().to(device)


# ============================================================
# STAGE 1 — Load 3D wireframe
# ============================================================

def load_3d_lines(pkl_path, device='cpu'):
    """
    Load wireframe_segments from room_geometry.pkl.
    Returns starts, ends, dirs, lengths as torch tensors.
    """
    with open(pkl_path, 'rb') as f:
        geom = pickle.load(f)

    segs = geom['wireframe_segments']
    starts = torch.tensor([s['start'] for s in segs], dtype=torch.float32, device=device)
    ends = torch.tensor([s['end'] for s in segs], dtype=torch.float32, device=device)
    raw_dirs = ends - starts
    lengths = raw_dirs.norm(dim=1)

    valid = lengths > 1e-6
    starts, ends = starts[valid], ends[valid]
    dirs = raw_dirs[valid] / lengths[valid, None]
    lengths = lengths[valid]
    return starts, ends, dirs, lengths


# ============================================================
# STAGE 2 — Per-view pixel lines → unit sphere
# ============================================================

def _focal(w=CROP_W, fov_deg=CROP_FOV_DEG):
    return 0.5 * w / np.tan(np.radians(fov_deg / 2.0))


def pixel_to_ray(u, v, W=CROP_W, H=CROP_H, fov_deg=CROP_FOV_DEG):
    """Map pixel (u, v) to unit camera-frame ray."""
    f = _focal(W, fov_deg)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    ray = np.array([(u - cx) / f, -(v - cy) / f, 1.0])
    return ray / np.linalg.norm(ray)


def parse_view_angles(filename):
    """Return (yaw_rad, pitch_rad) from filename like '00_yaw0.0_pitch-45.0.jpg'."""
    m = re.search(r'yaw(-?\d+(?:\.\d+)?)_pitch(-?\d+(?:\.\d+)?)', filename)
    return np.radians(float(m.group(1))), np.radians(float(m.group(2)))


def make_view_rotation(yaw_rad, pitch_rad):
    """Camera-to-pano-sphere rotation. Matches pano_processing_virtual_camerasV2."""
    phi = -pitch_rad
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi),  np.cos(phi)]])
    Ry = np.array([[ np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                   [ 0,               1, 0],
                   [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])
    return Ry @ Rx


def load_edge_2d(json_path, min_len_px=MIN_LINE_LEN_PX, device='cpu'):
    """
    Load extracted_2d_lines.json and back-project to unit sphere.
    Returns (N, 9) torch tensor [normal(3), start(3), end(3)].
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
            normal = np.cross(s, e)
            n = np.linalg.norm(normal)
            if n < 1e-6:
                continue
            vecs.append(np.concatenate([normal / n, s, e]))

    return torch.tensor(np.array(vecs), dtype=torch.float32, device=device)


# ============================================================
# STAGE 3 — Principal direction extraction (FGPL-faithful)
# ============================================================

def extract_principal_3d(dirs_sparse, sphere_pts):
    """
    Vote on icosphere to find 3 dominant 3D line directions.
    Uses argmax-bincount voting (FGPL style).
    """
    vote_dirs = dirs_sparse.clone()
    principal = []
    for _ in range(3):
        if vote_dirs.shape[0] == 0:
            break
        votes = torch.abs(vote_dirs[:, :3] @ sphere_pts.T).argmax(-1)
        vote_counts = votes.bincount(minlength=sphere_pts.shape[0])
        max_idx = vote_counts.argmax()
        principal.append(sphere_pts[max_idx])
        # Suppress inliers
        outlier = (torch.abs(vote_dirs[:, :3] @ sphere_pts[max_idx:max_idx+1].T) < 0.95).squeeze(-1)
        vote_dirs = vote_dirs[outlier]

    principal = torch.stack(principal, dim=0)
    if torch.det(principal) < 0:
        principal[-1] *= -1
    return principal  # (3, 3)


def extract_principal_2d(edge_2d, sphere_pts):
    """
    Vote on icosphere for 3 vanishing points via great-circle membership.
    FGPL-faithful: searches combinations for orthogonal triple.
    """
    vote_lines = edge_2d.clone()
    tot_directions = []
    max_search_iter = 20

    for _ in range(max_search_iter):
        if vote_lines.shape[0] == 0:
            break
        membership = torch.abs(vote_lines[:, :3] @ sphere_pts.T) < 0.05
        vote_counts = membership.any(dim=0).long()  # wrong — need count
        # Actually: count how many lines each sphere point lies on
        vote_counts = torch.where(
            torch.abs(vote_lines[:, :3] @ sphere_pts.T) < 0.05
        )[1].bincount(minlength=sphere_pts.shape[0])
        max_idx = vote_counts.argmax()
        tot_directions.append(sphere_pts[max_idx])
        # Suppress lines passing through this direction
        outlier = (torch.abs(vote_lines[:, :3] @ sphere_pts[max_idx:max_idx+1].T) > 0.05).squeeze(-1)
        vote_lines = vote_lines[outlier]

    if len(tot_directions) < 3:
        # Fallback: pad with cross products
        while len(tot_directions) < 3:
            if len(tot_directions) >= 2:
                cross = torch.cross(tot_directions[0], tot_directions[1])
                cross = cross / cross.norm().clamp(min=1e-6)
                tot_directions.append(cross)
            else:
                tot_directions.append(torch.tensor([1, 0, 0], dtype=torch.float32,
                                                    device=edge_2d.device))

    tot_directions = torch.stack(tot_directions, dim=0)

    # Search combinations for orthogonal triple
    n = tot_directions.shape[0]
    combs = torch.combinations(torch.arange(n, device=edge_2d.device), r=3)
    comb_dirs = tot_directions[combs]  # (C, 3, 3)
    comb_dots = torch.stack([
        (comb_dirs[:, i % 3] * comb_dirs[:, (i+1) % 3]).sum(-1).abs()
        for i in range(3)
    ], dim=-1)  # (C, 3)
    valid = (comb_dots < 0.1).sum(-1) == 3

    if valid.sum() > 0:
        idx = torch.where(valid)[0][0]
        principal_2d = comb_dirs[idx]
    elif (comb_dots < 0.15).any():
        # At least one perpendicular pair — get 2 and cross for third
        where = torch.where(comb_dots < 0.15)
        vec_0 = comb_dirs[where[0][0], where[1][0]]
        vec_1 = comb_dirs[where[0][0], (where[1][0] + 1) % 3]
        third = torch.cross(vec_0, vec_1)
        third = third / third.norm().clamp(min=1e-6)
        principal_2d = torch.stack([vec_0, vec_1, third])
    else:
        # Worst case: top 2 + cross
        two = tot_directions[:2]
        third = torch.cross(two[0], two[1])
        third = third / third.norm().clamp(min=1e-6)
        principal_2d = torch.cat([two, third.unsqueeze(0)])

    if torch.det(principal_2d) < 0:
        principal_2d[-1] *= -1
    return principal_2d  # (3, 3)


# ============================================================
# STAGE 4 — 24 rotation candidates (FGPL-faithful)
# ============================================================

def build_rotation_candidates(principal_2d, principal_3d):
    """
    Build 24 rotation candidates: 6 permutations × 4 determinant-preserving sign flips.
    Returns (24, 3, 3) tensor and associated permutation/sign info.
    """
    device = principal_2d.device
    perms_list = list(torch.combinations(torch.arange(3), r=3, with_replacement=False))
    # Actually use itertools for proper permutations
    from itertools import permutations as iter_perms
    perm_indices = list(iter_perms(range(3)))  # 6 permutations
    perms = torch.tensor(perm_indices, device=device, dtype=torch.long)

    # Build sign masks (4 per permutation, determinant-preserving)
    bin_mask = torch.ones(len(perm_indices) * 4, 3, 1, device=device)
    for perm_idx in range(len(perm_indices)):
        for idx in range(4):
            bin_mask[perm_idx * 4 + idx, 0, 0] = (-1) ** (idx // 2)
            bin_mask[perm_idx * 4 + idx, 1, 0] = (-1) ** (idx % 2)
            bin_mask[perm_idx * 4 + idx, 2, 0] = (-1) ** (idx // 2 + idx % 2)
            if perm_idx in [1, 2, 5]:
                bin_mask[perm_idx * 4 + idx, 2, 0] *= -1

    # Expand permutations to match bin_mask (4 copies each)
    perms_expanded = perms.repeat_interleave(4, dim=0)  # (24, 3)

    # Canonical target: identity (since we'll work in canonical frame)
    canonical = torch.eye(3, device=device).unsqueeze(0)  # (1, 3, 3)

    # Build rotations via SVD Procrustes
    N = perms_expanded.shape[0]  # 24
    pts_2d = principal_2d[perms_expanded]  # (24, 3, 3)
    H = canonical.permute(0, 2, 1) @ (bin_mask * pts_2d)  # (24, 3, 3)
    U, S, V = torch.svd(H)
    U_t = U.transpose(1, 2)
    d = torch.sign(torch.det(V @ U_t))
    diag = torch.eye(3, device=device).unsqueeze(0).repeat(N, 1, 1)
    diag[:, 2, 2] = d
    rotations = V @ diag @ U_t  # (24, 3, 3)

    return rotations, perms_expanded


def score_rotation(R, edge_2d, principal_3d, inlier_thres=INLIER_THRES_3D):
    """Score a rotation by counting 2D lines aligned with rotated 3D principals."""
    normals = edge_2d[:, :3]  # (N, 3)
    # In canonical frame: rotation maps 2D→canonical, principal_3d→canonical(identity)
    # So check: |normal @ R.T @ eye| → |normal @ R.T|
    inner = torch.abs(normals @ R.T)  # (N, 3)
    # Line is classified if its min inner product < threshold
    classified = (inner.min(-1).values < inlier_thres).sum()
    return classified.item()


# ============================================================
# STAGE 5 — Line classification & intersections (FGPL-faithful)
# ============================================================

def split_2d(edge_2d, principal_2d, inlier_thres=INLIER_THRES_2D):
    """Classify 2D lines by principal direction. Returns (N, 3) bool mask."""
    inner = torch.abs(edge_2d[:, :3] @ principal_2d.T)  # (N, 3)
    min_mask = inner.argmin(-1, keepdim=True) == torch.arange(3, device=edge_2d.device).unsqueeze(0)
    return (inner < inlier_thres) & min_mask


def split_3d(dirs, principal_3d, inlier_thres=INLIER_THRES_3D):
    """Classify 3D lines by principal direction. Returns (N, 3) bool mask."""
    inner = torch.abs(dirs @ principal_3d.T)
    return inner > 1 - inlier_thres


def intersections_2d(edge_2d, principal_2d, inlier_thres=INLIER_THRES_2D,
                     intersect_thres=INTERSECT_THRES_2D):
    """
    Find 2D line intersections on sphere via cross-product + arc membership.
    Returns (inter_pts: list of 3 tensors, inter_idx: list of 3 tensors).
    """
    pi = torch.acos(torch.zeros(1, device=edge_2d.device)).item() * 2
    mask = split_2d(edge_2d, principal_2d, inlier_thres)
    edge_p = [edge_2d[mask[:, i]] for i in range(3)]
    edge_num = [ep.shape[0] for ep in edge_p]

    full_range = torch.arange(edge_2d.shape[0], device=edge_2d.device)
    ids_p = [full_range[mask[:, i]] for i in range(3)]

    # Arc lengths
    arc_len = [torch.arccos((ep[:, 3:6] * ep[:, 6:]).sum(-1).clamp(-1, 1))
               for ep in edge_p]

    total_inter = []
    total_idx = []

    for i in range(3):
        j = (i + 1) % 3
        ni, nj = edge_num[i], edge_num[j]

        if ni == 0 or nj == 0:
            total_inter.append(torch.zeros(0, 3, device=edge_2d.device))
            total_idx.append(torch.zeros(0, 2, device=edge_2d.device, dtype=torch.long))
            continue

        # Cross product of normals → intersection candidates
        n_rep0 = edge_p[i][:, :3].repeat_interleave(nj, dim=0)
        n_rep1 = edge_p[j][:, :3].repeat(ni, 1)
        cand = torch.cross(n_rep0, n_rep1, dim=-1).reshape(ni, nj, 3)
        cand_norm = cand.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cand = cand / cand_norm

        # Build id grids
        id0 = ids_p[i].unsqueeze(1).expand(ni, nj)
        id1 = ids_p[j].unsqueeze(0).expand(ni, nj)

        # Arc membership test for candidate (up) and antipodal (down)
        s0 = edge_p[i][:, 3:6].unsqueeze(1).expand(ni, nj, 3)
        e0 = edge_p[i][:, 6:].unsqueeze(1).expand(ni, nj, 3)
        s1 = edge_p[j][:, 3:6].unsqueeze(0).expand(ni, nj, 3)
        e1 = edge_p[j][:, 6:].unsqueeze(0).expand(ni, nj, 3)
        al0 = arc_len[i].unsqueeze(1).expand(ni, nj)
        al1 = arc_len[j].unsqueeze(0).expand(ni, nj)

        # UP candidate
        ds0 = torch.arccos((cand * s0).sum(-1).clamp(-1, 1))
        de0 = torch.arccos((cand * e0).sum(-1).clamp(-1, 1))
        ds1 = torch.arccos((cand * s1).sum(-1).clamp(-1, 1))
        de1 = torch.arccos((cand * e1).sum(-1).clamp(-1, 1))
        up_on_0 = (ds0 + de0 - al0) < intersect_thres
        up_on_1 = (ds1 + de1 - al1) < intersect_thres
        up_valid = up_on_0 & up_on_1

        # DOWN candidate (antipodal)
        dn_ds0 = pi - ds0
        dn_de0 = pi - de0
        dn_ds1 = pi - ds1
        dn_de1 = pi - de1
        dn_on_0 = (dn_ds0 + dn_de0 - al0) < intersect_thres
        dn_on_1 = (dn_ds1 + dn_de1 - al1) < intersect_thres
        dn_valid = dn_on_0 & dn_on_1

        up_pts = cand[up_valid]
        dn_pts = -cand[dn_valid]
        up_ids = torch.stack([id0[up_valid], id1[up_valid]], dim=-1)
        dn_ids = torch.stack([id0[dn_valid], id1[dn_valid]], dim=-1)

        pts = torch.cat([up_pts, dn_pts], dim=0) if (up_pts.shape[0] + dn_pts.shape[0]) > 0 \
            else torch.zeros(0, 3, device=edge_2d.device)
        ids = torch.cat([up_ids, dn_ids], dim=0) if (up_ids.shape[0] + dn_ids.shape[0]) > 0 \
            else torch.zeros(0, 2, device=edge_2d.device, dtype=torch.long)

        total_inter.append(pts)
        total_idx.append(ids)

    return total_inter, total_idx


def intersections_3d(dirs, starts, ends, principal_3d,
                     inlier_thres=INLIER_THRES_3D, intersect_thres=INTERSECT_THRES_3D):
    """
    Find 3D line intersections via closest-point formula + parametric membership.
    Returns (inter_pts: list of 3 tensors, inter_idx: list of 3 tensors).
    """
    mask = split_3d(dirs, principal_3d, inlier_thres)
    starts_p = [starts[mask[:, i]] for i in range(3)]
    ends_p = [ends[mask[:, i]] for i in range(3)]

    full_range = torch.arange(dirs.shape[0], device=dirs.device)
    ids_p = [full_range[mask[:, i]] for i in range(3)]

    total_inter = []
    total_idx = []

    for i in range(3):
        j = (i + 1) % 3
        s0, e0, id0 = starts_p[i], ends_p[i], ids_p[i]
        s1, e1, id1 = starts_p[j], ends_p[j], ids_p[j]
        n0, n1 = s0.shape[0], s1.shape[0]

        if n0 == 0 or n1 == 0:
            total_inter.append(torch.zeros(0, 3, device=dirs.device))
            total_idx.append(torch.zeros(0, 2, device=dirs.device, dtype=torch.long))
            continue

        # Expand all pairs
        idx0, idx1 = torch.meshgrid(torch.arange(n0, device=dirs.device),
                                     torch.arange(n1, device=dirs.device), indexing='ij')
        idx0, idx1 = idx0.reshape(-1), idx1.reshape(-1)

        s0_e = s0[idx0]
        e0_e = e0[idx0]
        s1_e = s1[idx1]
        e1_e = e1[idx1]
        id0_e = id0[idx0]
        id1_e = id1[idx1]

        d0 = e0_e - s0_e
        d1 = e1_e - s1_e
        dc = torch.cross(d0, d1, dim=-1)
        dc_norm2 = (dc * dc).sum(-1)

        # Skip near-parallel pairs
        valid_par = dc_norm2 > 1e-8
        if not valid_par.any():
            total_inter.append(torch.zeros(0, 3, device=dirs.device))
            total_idx.append(torch.zeros(0, 2, device=dirs.device, dtype=torch.long))
            continue

        sd = s1_e - s0_e
        sd1 = torch.cross(sd, d1, dim=-1)
        sd0 = torch.cross(sd, d0, dim=-1)
        u = (sd1 * dc).sum(-1) / dc_norm2
        v = (sd0 * dc).sum(-1) / dc_norm2

        d0_len = d0.norm(dim=-1)
        d1_len = d1.norm(dim=-1)

        # Check parametric bounds with threshold
        on_0 = (u > -intersect_thres / d0_len.clamp(min=1e-6)) & \
               (u < 1 + intersect_thres / d0_len.clamp(min=1e-6))
        on_1 = (v > -intersect_thres / d1_len.clamp(min=1e-6)) & \
               (v < 1 + intersect_thres / d1_len.clamp(min=1e-6))
        on_both = on_0 & on_1 & valid_par

        pt0 = s0_e[on_both] + u[on_both].unsqueeze(-1) * d0[on_both]
        pt1 = s1_e[on_both] + v[on_both].unsqueeze(-1) * d1[on_both]
        close = (pt0 - pt1).norm(dim=-1) < intersect_thres

        inter_pts = pt0[close]
        inter_ids = torch.stack([id0_e[on_both][close], id1_e[on_both][close]], dim=-1)

        total_inter.append(inter_pts if inter_pts.shape[0] > 0
                           else torch.zeros(0, 3, device=dirs.device))
        total_idx.append(inter_ids if inter_ids.shape[0] > 0
                         else torch.zeros(0, 2, device=dirs.device, dtype=torch.long))

    return total_inter, total_idx


def _build_intersection_masks(inter_list, device):
    """Build FGPL-style masks: inter[k] is from groups k and (k+1)%3,
    so mask column k and (k+1)%3 are True (inverted from FGPL convention)."""
    masks = []
    for k in range(3):
        n = inter_list[k].shape[0]
        m = torch.zeros(n, 3, dtype=torch.bool, device=device)
        m[:, k] = True
        m[:, (k + 1) % 3] = True
        masks.append(m)
    return torch.cat(masks, dim=0)


# ============================================================
# STAGE 6 — Translation candidate grid
# ============================================================

def generate_translation_candidates(starts, ends, spacing=TRANS_SPACING):
    """
    3D grid within wireframe bounding box, filtered by chamfer distance.
    """
    device = starts.device
    pts = torch.cat([starts, ends], dim=0)
    mins = pts.min(0).values
    maxs = pts.max(0).values

    xs = torch.arange(mins[0].item(), maxs[0].item(), spacing, device=device)
    ys = torch.arange(mins[1].item(), maxs[1].item(), spacing, device=device)
    zs = torch.arange(mins[2].item(), maxs[2].item(), spacing, device=device)

    if len(xs) == 0: xs = torch.tensor([mins[0].item()], device=device)
    if len(ys) == 0: ys = torch.tensor([mins[1].item()], device=device)
    if len(zs) == 0: zs = torch.tensor([mins[2].item()], device=device)

    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
    trans = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)

    # Chamfer filter: keep points > CHAMFER_MIN_DIST from wireframe midpoints
    midpoints = (starts + ends) / 2
    if midpoints.shape[0] > 500:
        sample_idx = torch.randperm(midpoints.shape[0], device=device)[:500]
        midpoints = midpoints[sample_idx]

    # Process in chunks to avoid OOM
    chunk = 200
    keep = torch.ones(trans.shape[0], dtype=torch.bool, device=device)
    for i in range(0, trans.shape[0], chunk):
        batch = trans[i:i+chunk]
        chamfer = (batch.unsqueeze(1) - midpoints.unsqueeze(0)).norm(dim=-1).min(dim=1).values
        keep[i:i+chunk] = chamfer > CHAMFER_MIN_DIST

    return trans[keep]


# ============================================================
# STAGE 7 — XDF coarse search (canonical precomputation)
# ============================================================

def _distance_func_2d(query_pts, edge_2d, mask=None, rot_mtx=None):
    """
    Spherical arc distance from query points to 2D line segments.
    Handles endpoint fallback (FGPL-faithful).

    Args:
        query_pts: (N_q, 3)
        edge_2d:   (N_2D, 9) [normal, start, end]
        mask:      (N_2D, K) bool or None
        rot_mtx:   (3, 3) rotation to apply to lines, or None

    Returns: (N_q,) or (N_q, K) distances
    """
    normals = edge_2d[:, :3]
    starts = edge_2d[:, 3:6]
    ends = edge_2d[:, 6:]

    if rot_mtx is not None:
        normals = (normals @ rot_mtx.T)
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        starts = (starts @ rot_mtx.T)
        starts = starts / starts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ends = (ends @ rot_mtx.T)
        ends = ends / ends.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    cos_se = (starts * ends).sum(-1).unsqueeze(0)  # (1, N_2D)
    cos_qs = query_pts @ starts.T  # (N_q, N_2D)
    cos_qe = query_pts @ ends.T

    normal_acute = np.pi / 2 - torch.arccos(torch.abs(query_pts @ normals.T).clamp(-1, 1))
    theta_s = torch.arccos(cos_qs.clamp(-1, 1))
    theta_e = torch.arccos(cos_qe.clamp(-1, 1))

    sign_s = cos_qs - cos_se * cos_qe > 0
    sign_e = cos_qe - cos_se * cos_qs > 0
    on_arc = sign_s & sign_e

    sphere_dist = on_arc * normal_acute + ~on_arc * torch.minimum(theta_s, theta_e)

    if mask is None:
        return sphere_dist.min(-1).values
    else:
        MAX = np.pi
        dist = sphere_dist.unsqueeze(-1) * mask.unsqueeze(0)
        dist += ~mask.unsqueeze(0) * MAX
        return dist.min(1).values  # (N_q, K)


def _distance_func_3d(query_pts, starts, ends, trans, rot, mask=None):
    """
    Spherical arc distance from query points to 3D line segments projected to sphere.

    Args:
        query_pts: (N_q, 3)
        starts:    (N_3D, 3) canonical-frame line starts
        ends:      (N_3D, 3)
        trans:     (3,) canonical-frame translation
        rot:       (3, 3) rotation (identity in canonical frame)
        mask:      (N_3D, K) bool or None

    Returns: (N_q,) or (N_q, K)
    """
    s_cam = (starts - trans.unsqueeze(0)) @ rot.T
    e_cam = (ends - trans.unsqueeze(0)) @ rot.T
    s_sph = s_cam / s_cam.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    e_sph = e_cam / e_cam.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    normals = torch.cross(s_sph, e_sph, dim=-1)
    n_norm = normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normals = normals / n_norm

    cos_se = (s_sph * e_sph).sum(-1).unsqueeze(0)
    cos_qs = query_pts @ s_sph.T
    cos_qe = query_pts @ e_sph.T

    normal_acute = np.pi / 2 - torch.arccos(torch.abs(query_pts @ normals.T).clamp(-1, 1))
    theta_s = torch.arccos(cos_qs.clamp(-1, 1))
    theta_e = torch.arccos(cos_qe.clamp(-1, 1))

    sign_s = cos_qs - cos_se * cos_qe > 0
    sign_e = cos_qe - cos_se * cos_qs > 0
    on_arc = sign_s & sign_e

    sphere_dist = on_arc * normal_acute + ~on_arc * torch.minimum(theta_s, theta_e)

    if mask is None:
        return sphere_dist.min(-1).values
    else:
        MAX = np.pi
        dist = sphere_dist.unsqueeze(-1) * mask.unsqueeze(0)
        dist += ~mask.unsqueeze(0) * MAX
        return dist.min(1).values


def _distance_func_point(query_pts, kpts, mask=None):
    """Spherical distance from query points to keypoints (intersections)."""
    cos_theta = (query_pts @ kpts.T).clamp(-1, 1)
    sphere_dist = torch.arccos(cos_theta)  # (N_q, N_kpts)

    if mask is None:
        return sphere_dist.min(-1).values
    else:
        MAX = np.pi
        dist = sphere_dist.unsqueeze(-1) * mask.unsqueeze(0)
        dist += ~mask.unsqueeze(0) * MAX
        return dist.min(1).values


def xdf_coarse_search(rotations, perms_expanded, principal_2d, principal_3d,
                      edge_2d, starts, ends, dirs,
                      inter_2d_list, inter_3d_list,
                      inter_2d_idx_list, inter_3d_idx_list,
                      trans_candidates, query_pts, top_k=5):
    """
    Canonical precomputation + inlier-counting XDF search.

    Returns: list of (R, t, rot_idx, inter_2d_concat, inter_2d_mask, inter_2d_idx_concat)
             for top-K poses.
    """
    device = edge_2d.device
    N_r = rotations.shape[0]
    N_t = trans_candidates.shape[0]
    N_q = query_pts.shape[0]

    # --- Canonical frame: rotate 3D so principal_3d → identity ---
    canonical_rot = principal_3d  # (3,3) — maps canonical → world
    can_starts = starts @ canonical_rot.T
    can_ends = ends @ canonical_rot.T
    can_dirs = dirs @ canonical_rot.T
    can_trans = trans_candidates @ canonical_rot.T

    # 3D line mask in canonical frame
    can_principal = canonical_rot @ canonical_rot.T  # = identity
    mask_3d = split_3d(can_dirs, torch.eye(3, device=device), INLIER_THRES_3D)

    # 3D intersections in canonical frame + mask
    can_inter_3d = torch.cat(inter_3d_list, dim=0) @ canonical_rot.T
    mask_3d_inter = _build_intersection_masks(inter_3d_list, device)

    # --- Precompute LDF-3D: (N_t, N_q, 3) ---
    print("    Precomputing LDF-3D...")
    eye = torch.eye(3, device=device)
    chunk_t = 200
    ldf_3d_chunks = []
    for i in range(0, N_t, chunk_t):
        batch_trans = can_trans[i:i+chunk_t]
        batch_ldf = []
        for ti in range(batch_trans.shape[0]):
            d = _distance_func_3d(query_pts, can_starts, can_ends, batch_trans[ti], eye, mask_3d)
            batch_ldf.append(d)
        ldf_3d_chunks.append(torch.stack(batch_ldf))
    ldf_3d = torch.cat(ldf_3d_chunks, dim=0)  # (N_t, N_q, 3)

    # --- Precompute PDF-3D: (N_t, N_q, 3) ---
    print("    Precomputing PDF-3D...")
    pdf_3d_chunks = []
    for i in range(0, N_t, chunk_t):
        batch_trans = can_trans[i:i+chunk_t]
        batch_pdf = []
        for ti in range(batch_trans.shape[0]):
            # Project 3D intersections to sphere
            i3d_cam = (can_inter_3d - batch_trans[ti].unsqueeze(0)) @ eye.T
            i3d_sph = i3d_cam / i3d_cam.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            d = _distance_func_point(query_pts, i3d_sph, mask_3d_inter)
            batch_pdf.append(d)
        pdf_3d_chunks.append(torch.stack(batch_pdf))
    pdf_3d = torch.cat(pdf_3d_chunks, dim=0)  # (N_t, N_q, 3)

    # --- Precompute LDF-2D: (N_r, N_q, 3) per rotation ---
    print("    Precomputing LDF-2D...")
    # For each rotation, classify 2D lines and compute distance
    ldf_2d_list = []
    canonical_mask_2d_list = []
    for ri in range(N_r):
        R_can = rotations[ri]  # maps 2D→canonical
        # Classify: apply rotation to normals, check against canonical axes (identity)
        rot_inner = torch.abs(edge_2d[:, :3] @ R_can.T)  # (N_2D, 3)
        min_mask = rot_inner.argmin(-1, keepdim=True) == torch.arange(3, device=device).unsqueeze(0)
        can_mask = (rot_inner < INLIER_THRES_2D) & min_mask
        canonical_mask_2d_list.append(can_mask)
        d = _distance_func_2d(query_pts, edge_2d, can_mask, rot_mtx=R_can)
        ldf_2d_list.append(d)
    ldf_2d = torch.stack(ldf_2d_list)  # (N_r, N_q, 3)

    # --- Precompute PDF-2D: (N_r, N_q, 3) ---
    print("    Precomputing PDF-2D...")
    # Build 2D intersections per rotation (permuted)
    # Raw intersections are computed once; permute group assignment per rotation
    raw_inter_2d = torch.cat(inter_2d_list, dim=0)

    pdf_2d_list = []
    full_inter_2d_per_rot = []
    full_mask_2d_per_rot = []
    full_idx_2d_per_rot = []

    for ri in range(N_r):
        perm = perms_expanded[ri].tolist()
        # Remap intersection groups according to permutation
        inter_perm_order = [_intersections_idx(perm[k % 3], perm[(k+1) % 3]) for k in range(3)]

        inter_pts = torch.cat([inter_2d_list[p] for p in inter_perm_order], dim=0)
        inter_ids = torch.cat([inter_2d_idx_list[p] for p in inter_perm_order], dim=0)

        # Build mask for permuted intersections
        masks = []
        for k, p in enumerate(inter_perm_order):
            n = inter_2d_list[p].shape[0]
            m = torch.zeros(n, 3, dtype=torch.bool, device=device)
            m[:, k] = True
            m[:, (k+1) % 3] = True
            masks.append(m)
        inter_mask = torch.cat(masks, dim=0)

        full_inter_2d_per_rot.append(inter_pts)
        full_mask_2d_per_rot.append(inter_mask)
        full_idx_2d_per_rot.append(inter_ids)

        # Apply rotation to intersection points
        R_can = rotations[ri]
        rot_pts = inter_pts @ R_can.T
        rot_pts = rot_pts / rot_pts.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        d = _distance_func_point(query_pts, rot_pts, inter_mask)
        pdf_2d_list.append(d)

    pdf_2d = torch.stack(pdf_2d_list)  # (N_r, N_q, 3)

    # --- Combine and compute cost ---
    print("    Computing XDF cost matrix...")
    dist_3d = torch.cat([ldf_3d, pdf_3d ** POINT_GAMMA], dim=-1)  # (N_t, N_q, 6)
    dist_2d = torch.cat([ldf_2d, pdf_2d ** POINT_GAMMA], dim=-1)  # (N_r, N_q, 6)

    # Inlier counting: cost[t,r] = -sum over (q,k) of (|d2d - d3d| < thres)
    # Process in chunks to save memory
    cost = torch.zeros(N_t, N_r, device=device)
    for i in range(0, N_t, chunk_t):
        d3d_batch = dist_3d[i:i+chunk_t]  # (B, N_q, 6)
        # (B, N_r, N_q, 6)
        diff = torch.abs(dist_2d.unsqueeze(0) - d3d_batch.unsqueeze(1))
        cost[i:i+chunk_t] = -(diff < XDF_INLIER_THRES).sum(-1).sum(-1)  # (B, N_r)

    # Find top-K
    flat_idx = cost.flatten().argsort()[:top_k]
    t_idx = flat_idx // N_r
    r_idx = flat_idx % N_r

    results = []
    for k in range(top_k):
        ti, ri = t_idx[k].item(), r_idx[k].item()
        # Un-canonicalize rotation
        R_world = rotations[ri] @ canonical_rot
        t_world = trans_candidates[ti]
        results.append({
            'R': R_world,
            't': t_world,
            'rot_idx': ri,
            'cost': cost[ti, ri].item(),
            'inter_2d': full_inter_2d_per_rot[ri],
            'inter_2d_mask': full_mask_2d_per_rot[ri],
            'inter_2d_idx': full_idx_2d_per_rot[ri],
        })

    return results


def _intersections_idx(idx0, idx1):
    """Map (direction_i, direction_j) → intersection group index 0/1/2."""
    r0, r1 = idx0 % 3, idx1 % 3
    if (r0, r1) in [(0, 1), (1, 0)]:
        return 0
    elif (r0, r1) in [(1, 2), (2, 1)]:
        return 1
    else:
        return 2


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device('cpu')
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    pkl_path = project_root / "data" / "debug_renderer" / POINT_CLOUD_NAME / "room_geometry.pkl"
    json_path = project_root / "data" / "pano" / "2d_feature_extracted" / ROOM_NAME / "extracted_2d_lines.json"

    print("[1] Loading 3D wireframe...")
    starts, ends, dirs, lengths = load_3d_lines(pkl_path, device)
    print(f"    wireframe segments: {starts.shape[0]}")

    # Length filters
    pts = torch.cat([starts, ends], dim=0)
    extent = (pts.max(0).values - pts.min(0).values).norm().item()
    sparse_mask = lengths >= 0.10 * extent
    dense_mask = lengths >= 0.20

    starts_s, dirs_s = starts[sparse_mask], dirs[sparse_mask]
    starts_d, ends_d = starts[dense_mask], ends[dense_mask]
    dirs_d, lengths_d = dirs[dense_mask], lengths[dense_mask]
    print(f"    sparse (>10% extent): {sparse_mask.sum().item()}  "
          f"dense (>0.2m): {dense_mask.sum().item()}")

    print("[2] Loading 2D lines and back-projecting to unit sphere...")
    edge_2d = load_edge_2d(json_path, device=device)
    print(f"    edge_2d shape: {edge_2d.shape}")

    print("[3] Extracting principal directions...")
    vote_sphere = generate_sphere_pts(VOTE_SPHERE_LEVEL, device)
    vote_sphere_upper = vote_sphere[vote_sphere[:, 1] >= 0]  # Y-up upper hemisphere
    principal_3d = extract_principal_3d(dirs_s, vote_sphere_upper)
    principal_2d = extract_principal_2d(edge_2d, vote_sphere_upper)
    print(f"    principal_3d:\n{principal_3d}")
    print(f"    principal_2d:\n{principal_2d}")

    print("[4] Building 24 rotation candidates...")
    rotations, perms_expanded = build_rotation_candidates(principal_2d, principal_3d)
    # Score and report best
    scores = [score_rotation(rotations[i], edge_2d, principal_3d) for i in range(rotations.shape[0])]
    best_ri = int(np.argmax(scores))
    print(f"    best rotation score: {scores[best_ri]}  idx={best_ri}")

    print("[5] Computing intersections...")
    inter_2d_list, inter_2d_idx = intersections_2d(edge_2d, principal_2d)
    inter_3d_list, inter_3d_idx = intersections_3d(dirs_d, starts_d, ends_d, principal_3d)
    for k in range(3):
        print(f"    inter_2d[{k}]: {inter_2d_list[k].shape[0]}  "
              f"inter_3d[{k}]: {inter_3d_list[k].shape[0]}")

    print("[6] Generating translation candidates...")
    trans_candidates = generate_translation_candidates(starts, ends)
    print(f"    {trans_candidates.shape[0]} translation candidates")

    print("[7] XDF coarse search (canonical precomputation)...")
    query_pts = generate_sphere_pts(QUERY_SPHERE_LEVEL, device)
    top_poses = xdf_coarse_search(
        rotations, perms_expanded, principal_2d, principal_3d,
        edge_2d, starts_d, ends_d, dirs_d,
        inter_2d_list, inter_3d_list,
        inter_2d_idx, inter_3d_idx,
        trans_candidates, query_pts, top_k=1)

    best = top_poses[0]
    best_R = best['R'].numpy()
    best_t = best['t'].numpy()
    print(f"    best XDF cost: {best['cost']:.4f}  t={best_t}")

    print("[8] Full ICP refinement...")
    # Concatenate intersections for refinement
    all_inter_2d = best['inter_2d'].numpy()
    all_inter_2d_mask = best['inter_2d_mask'].numpy()
    all_inter_2d_idx = best['inter_2d_idx'].numpy()
    all_inter_3d = torch.cat(inter_3d_list, dim=0).numpy()
    all_inter_3d_mask = _build_intersection_masks(inter_3d_list, device).numpy()
    all_inter_3d_idx = torch.cat(inter_3d_idx, dim=0).numpy()

    final_R, final_t, matched_pairs = refine_pose_full(
        best_R, best_t,
        all_inter_2d, all_inter_3d,
        inter_2d_mask=all_inter_2d_mask,
        inter_3d_mask=all_inter_3d_mask,
        line_dirs_2d=edge_2d[:, :3].numpy(),
        line_dirs_3d=dirs_d.numpy(),
        inter_2d_idx=all_inter_2d_idx,
        inter_3d_idx=all_inter_3d_idx,
        n_iters_t=100, n_iters_r=50, lr=0.1,
        nn_dist_thres=0.5, rot_inlier_thres=0.2)

    print(f"    refined t={final_t}")
    print(f"    det(R)={np.linalg.det(final_R):.4f}")

    print("[9] Saving camera_pose.json...")
    output_dir = project_root / "data" / "pose_estimates" / ROOM_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build per-group intersection arrays for visualization
    inter_2d_groups = []
    inter_3d_groups = []
    offset_2d = 0
    for k in range(3):
        n2d = inter_2d_list[k].shape[0]
        inter_2d_groups.append(best['inter_2d'][offset_2d:offset_2d+n2d].numpy().tolist()
                               if n2d > 0 else [])
        # Note: permutation may reorder groups — use the permuted data
        offset_2d += n2d
    # If permutation reordered, we need to handle that. For simplicity, store all_inter_2d split by mask
    inter_2d_groups = [[], [], []]
    for idx_pt in range(all_inter_2d.shape[0]):
        for k in range(3):
            if all_inter_2d_mask[idx_pt, k] and all_inter_2d_mask[idx_pt, (k+1) % 3]:
                inter_2d_groups[k].append(all_inter_2d[idx_pt].tolist())
                break

    inter_3d_groups = [[], [], []]
    for idx_pt in range(all_inter_3d.shape[0]):
        for k in range(3):
            if all_inter_3d_mask[idx_pt, k] and all_inter_3d_mask[idx_pt, (k+1) % 3]:
                inter_3d_groups[k].append(all_inter_3d[idx_pt].tolist())
                break

    result = {
        "rotation":          final_R.tolist(),
        "translation":       final_t.tolist(),
        "principal_3d":      principal_3d.numpy().tolist(),
        "principal_2d":      principal_2d.numpy().tolist(),
        "n_inter_matched":   [len(mp) for mp in matched_pairs],
        "xdf_cost_coarse":   float(best['cost']),
        "inter_2d":          inter_2d_groups,
        "inter_3d":          inter_3d_groups,
        "matched_pairs":     [mp.tolist() for mp in matched_pairs],
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
    print(f"n_inter_matched: {result['n_inter_matched']} (want >5 each)")
    tx, ty, tz = final_t
    all_pts = torch.cat([starts, ends], dim=0).numpy()
    mins = all_pts.min(0)
    maxs = all_pts.max(0)
    inside = (mins[0] <= tx <= maxs[0] and
              mins[1] <= ty <= maxs[1] and
              mins[2] <= tz <= maxs[2])
    print(f"translation in room bbox: {inside}  t=({tx:.2f}, {ty:.2f}, {tz:.2f})")
    print(f"det(R)={np.linalg.det(final_R):.4f}  (want +1.0)")


if __name__ == "__main__":
    main()
