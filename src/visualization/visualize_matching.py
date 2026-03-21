"""
visualize_matching.py — Human-inspection overlay for pose estimation results.

For each of the 22 perspective crops, produces a 2-panel PNG:
  Left  : photo + 2D lines (green, thick) + projected 3D wireframe (cyan, thin)
  Right : photo + matched intersection pairs (colored by group, with connecting lines)

Falls back to old 2-panel layout (2D-only left, 3D-only right) if camera_pose.json
lacks the V2 intersection keys.

Run after feature_matching.py or feature_matchingV2.py.

Output: data/pose_estimates/<ROOM_NAME>/vis/<view_name>.png
"""

import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# CONFIGURATION — match feature_matching.py
# ============================================================
POINT_CLOUD_NAME = "tmb_office1"
ROOM_NAME        = "TMB_office1"
CROP_W = CROP_H  = 1024
CROP_FOV_DEG     = 60.0

# Drawing
COLOR_2D  = (0,   220,  0)    # green  — extracted 2D lines
COLOR_3D  = (255, 200,  0)    # cyan   — projected 3D wireframe
GROUP_COLORS = [
    (60,  60,  255),   # red   (BGR) — group 0
    (255, 100, 60),    # blue  (BGR) — group 1
    (0,   230, 230),   # yellow (BGR) — group 2
]


# ============================================================
# GEOMETRY HELPERS  (mirror feature_matching.py conventions)
# ============================================================

def focal():
    return 0.5 * CROP_W / np.tan(np.radians(CROP_FOV_DEG / 2.0))


def make_view_rotation(yaw_rad, pitch_rad):
    """Camera-to-pano-sphere rotation. Matches pano_processing_virtual_camerasV2."""
    phi_rad   = -pitch_rad          # sign flip matches renderer
    theta_rad = yaw_rad
    Rx = np.array([[1, 0,               0              ],
                   [0, np.cos(phi_rad), -np.sin(phi_rad)],
                   [0, np.sin(phi_rad),  np.cos(phi_rad)]])
    Ry = np.array([[ np.cos(theta_rad), 0, np.sin(theta_rad)],
                   [ 0,                 1, 0                 ],
                   [-np.sin(theta_rad), 0, np.cos(theta_rad)]])
    return Ry @ Rx


def parse_view_angles(filename):
    m = re.search(r'yaw(-?\d+(?:\.\d+)?)_pitch(-?\d+(?:\.\d+)?)', filename)
    return np.radians(float(m.group(1))), np.radians(float(m.group(2)))


def clip_to_front(P1, P2, R_pano, t, R_crop, eps=0.05):
    """
    Transform both endpoints to crop-camera space and clip the segment
    so both endpoints have z > eps.
    Returns (P1_cam, P2_cam) both with z > eps, or None if fully behind.
    """
    def to_cam(P):
        return R_crop.T @ (R_pano @ (P - t))

    c1 = to_cam(P1)
    c2 = to_cam(P2)

    if c1[2] < eps and c2[2] < eps:
        return None

    if c1[2] < eps:
        alpha = (eps - c1[2]) / (c2[2] - c1[2])
        c1 = c1 + alpha * (c2 - c1)

    if c2[2] < eps:
        alpha = (eps - c2[2]) / (c1[2] - c2[2])
        c2 = c2 + alpha * (c1 - c2)

    return c1, c2


def cam_to_pixel(P_cam):
    """Camera-space 3D point → (u, v) pixel. Assumes z > 0."""
    f  = focal()
    cx = (CROP_W - 1) / 2.0
    cy = (CROP_H - 1) / 2.0
    u = f * (P_cam[0] / P_cam[2]) + cx
    v = -f * (P_cam[1] / P_cam[2]) + cy
    return int(round(u)), int(round(v))


def pixel_in_bounds(u, v, margin=0):
    return (-margin <= u <= CROP_W + margin and
            -margin <= v <= CROP_H + margin)


def sphere_to_crop_pixel(sphere_pt, R_crop):
    """Project a unit-sphere point to crop pixel coords. Returns (u,v) or None."""
    P_crop = R_crop.T @ sphere_pt
    if P_crop[2] <= 1e-4:
        return None
    f  = focal()
    cx = (CROP_W - 1) / 2.0
    cy = (CROP_H - 1) / 2.0
    u = f * (P_crop[0] / P_crop[2]) + cx
    v = -f * (P_crop[1] / P_crop[2]) + cy
    return int(round(u)), int(round(v))


def world_to_crop_pixel(P_world, R_pano, t, R_crop):
    """Project a 3D world point to crop pixel coords. Returns (u,v) or None."""
    P_pano = R_pano @ (P_world - t)
    P_crop = R_crop.T @ P_pano
    if P_crop[2] <= 1e-4:
        return None
    f  = focal()
    cx = (CROP_W - 1) / 2.0
    cy = (CROP_H - 1) / 2.0
    u = f * (P_crop[0] / P_crop[2]) + cx
    v = -f * (P_crop[1] / P_crop[2]) + cy
    return int(round(u)), int(round(v))


# ============================================================
# DRAW HELPERS
# ============================================================

def draw_2d_lines(img, lines, color=COLOR_2D, thickness=2):
    """Draw a list of [[u1,v1],[u2,v2]] pixel segments."""
    for (p1, p2) in lines:
        pt1 = (int(round(p1[0])), int(round(p1[1])))
        pt2 = (int(round(p2[0])), int(round(p2[1])))
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def draw_3d_wireframe(img, wireframe_segs, R_pano, t, R_crop, color=COLOR_3D, thickness=1):
    """Project and draw 3D wireframe segments."""
    for seg in wireframe_segs:
        P1 = np.asarray(seg['start'], dtype=float)
        P2 = np.asarray(seg['end'],   dtype=float)

        clipped = clip_to_front(P1, P2, R_pano, t, R_crop)
        if clipped is None:
            continue
        c1, c2 = clipped

        u1, v1 = cam_to_pixel(c1)
        u2, v2 = cam_to_pixel(c2)

        if not (pixel_in_bounds(u1, v1, margin=50) or
                pixel_in_bounds(u2, v2, margin=50)):
            continue

        u1c = int(np.clip(u1, 0, CROP_W - 1))
        v1c = int(np.clip(v1, 0, CROP_H - 1))
        u2c = int(np.clip(u2, 0, CROP_W - 1))
        v2c = int(np.clip(v2, 0, CROP_H - 1))

        cv2.line(img, (u1c, v1c), (u2c, v2c), color, thickness, cv2.LINE_AA)


def draw_intersection_panel(img, R_pano, t, R_crop, inter_2d, inter_3d, matched_pairs):
    """
    Draw matched intersection pairs on the image.
    2D intersections → open circles, 3D intersections → filled circles.
    Matched pairs connected by thin lines. Colored by group.
    Returns match count string.
    """
    match_counts = []
    margin = 20

    for k in range(3):
        color = GROUP_COLORS[k]
        i2d_group = inter_2d[k] if k < len(inter_2d) else []
        i3d_group = inter_3d[k] if k < len(inter_3d) else []
        pairs = matched_pairs[k] if k < len(matched_pairs) else []
        match_counts.append(len(pairs))

        # Draw 2D intersection points (open circles)
        i2d_pixels = {}
        for pi, pt in enumerate(i2d_group):
            pt = np.asarray(pt, dtype=float)
            uv = sphere_to_crop_pixel(pt, R_crop)
            if uv is not None and pixel_in_bounds(uv[0], uv[1], margin):
                i2d_pixels[pi] = uv
                cv2.circle(img, uv, 6, color, 2, cv2.LINE_AA)

        # Draw 3D intersection points (filled circles)
        i3d_pixels = {}
        for pi, pt in enumerate(i3d_group):
            pt = np.asarray(pt, dtype=float)
            uv = world_to_crop_pixel(pt, R_pano, t, R_crop)
            if uv is not None and pixel_in_bounds(uv[0], uv[1], margin):
                i3d_pixels[pi] = uv
                cv2.circle(img, uv, 4, color, -1, cv2.LINE_AA)

        # Draw connecting lines for matched pairs
        for pair in pairs:
            i2_idx, i3_idx = int(pair[0]), int(pair[1])
            if i2_idx in i2d_pixels and i3_idx in i3d_pixels:
                cv2.line(img, i2d_pixels[i2_idx], i3d_pixels[i3_idx],
                         color, 1, cv2.LINE_AA)

    return "/".join(str(c) for c in match_counts)


def add_legend(img, label, color):
    cv2.rectangle(img, (8, 8), (22, 22), color, -1)
    cv2.putText(img, label, (28, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def add_legend_multi(img, items, y_start=8):
    """Draw multiple legend items vertically."""
    for i, (label, color, filled) in enumerate(items):
        y = y_start + i * 22
        if filled:
            cv2.circle(img, (15, y + 7), 5, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (15, y + 7), 5, color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (28, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================
# MAIN
# ============================================================

def main():
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # --- Paths ---
    pose_path   = project_root / "data" / "pose_estimates" / ROOM_NAME / "camera_pose.json"
    json_path   = project_root / "data" / "pano" / "2d_feature_extracted" / ROOM_NAME / "extracted_2d_lines.json"
    pkl_path    = project_root / "data" / "debug_renderer" / POINT_CLOUD_NAME / "room_geometry.pkl"
    crops_dir   = project_root / "data" / "pano" / "virtual_camera_processed" / ROOM_NAME
    output_dir  = project_root / "data" / "pose_estimates" / ROOM_NAME / "vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pose ---
    print("Loading camera_pose.json...")
    with open(pose_path) as f:
        pose = json.load(f)
    R_pano = np.array(pose['rotation'])
    t      = np.array(pose['translation'])
    print(f"  R det={np.linalg.det(R_pano):.4f}  t={t}")

    # Check for V2 intersection data
    has_v2_data = all(key in pose for key in ('inter_2d', 'inter_3d', 'matched_pairs'))
    if has_v2_data:
        inter_2d = pose['inter_2d']       # list of 3 arrays
        inter_3d = pose['inter_3d']       # list of 3 arrays
        matched_pairs = pose['matched_pairs']  # list of 3 arrays
        print(f"  V2 data found: inter_2d groups={[len(g) for g in inter_2d]}, "
              f"matched_pairs={[len(g) for g in matched_pairs]}")
    else:
        print("  No V2 intersection data — using legacy 2-panel layout")

    # --- Load 2D lines ---
    print("Loading extracted_2d_lines.json...")
    with open(json_path) as f:
        lines_2d = json.load(f)
    print(f"  {len(lines_2d)} views")

    # --- Load 3D wireframe ---
    print("Loading room_geometry.pkl...")
    with open(pkl_path, 'rb') as f:
        geom = pickle.load(f)
    wireframe_segs = geom['wireframe_segments']
    print(f"  {len(wireframe_segs)} wireframe segments")

    # --- Per-view visualization ---
    n_saved = 0
    for filename, seg_list in sorted(lines_2d.items()):
        yaw_rad, pitch_rad = parse_view_angles(filename)
        R_crop = make_view_rotation(yaw_rad, pitch_rad)

        # Load original crop (fall back to blank if missing)
        crop_path = crops_dir / filename
        if crop_path.exists():
            base = cv2.imread(str(crop_path))
            if base is None:
                base = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
        else:
            base = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)

        if has_v2_data:
            # --- V2 layout: [line overlay | matched intersections] ---

            # Left panel: 2D lines (green thick) + 3D wireframe (cyan thin) on same image
            left = base.copy()
            draw_2d_lines(left, seg_list, COLOR_2D, thickness=2)
            draw_3d_wireframe(left, wireframe_segs, R_pano, t, R_crop, COLOR_3D, thickness=1)
            add_legend(left, f"2D(green) + 3D(cyan)", (200, 200, 200))

            # Right panel: matched intersection pairs
            right = base.copy()
            match_str = draw_intersection_panel(
                right, R_pano, t, R_crop, inter_2d, inter_3d, matched_pairs)
            add_legend_multi(right, [
                ("2D inter (open)", GROUP_COLORS[0], False),
                ("3D inter (filled)", GROUP_COLORS[0], True),
                ("grp0/1/2", (200, 200, 200), False),
            ])

        else:
            # --- Legacy layout: [2D lines | 3D wireframe] ---
            left = base.copy()
            draw_2d_lines(left, seg_list, COLOR_2D, thickness=2)
            add_legend(left, f"2D lines ({len(seg_list)})", COLOR_2D)

            right = base.copy()
            draw_3d_wireframe(right, wireframe_segs, R_pano, t, R_crop, COLOR_3D, thickness=2)
            add_legend(right, f"3D projected ({len(wireframe_segs)} segs)", COLOR_3D)
            match_str = ""

        # --- Compose side-by-side ---
        divider = np.full((CROP_H, 4, 3), 200, dtype=np.uint8)
        panel   = np.hstack([left, divider, right])

        # Title bar
        title_h  = 36
        title    = np.zeros((title_h, panel.shape[1], 3), dtype=np.uint8)
        stem     = filename.replace('.jpg', '').replace('.png', '')
        title_text = stem
        if match_str:
            title_text += f"  matches: {match_str}"
        cv2.putText(title, title_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
        out_img = np.vstack([title, panel])

        out_path = output_dir / filename.replace('.jpg', '.png').replace('.jpeg', '.png')
        cv2.imwrite(str(out_path), out_img)
        n_saved += 1

    print(f"\nSaved {n_saved} comparison images -> {output_dir}")
    if has_v2_data:
        print("Left panel  = 2D lines (green) + projected 3D wireframe (cyan)")
        print("Right panel = matched intersection pairs (colored by group)")
    else:
        print("Left panel  = extracted 2D lines (green)")
        print("Right panel = projected 3D wireframe segments (cyan)")


if __name__ == "__main__":
    main()
