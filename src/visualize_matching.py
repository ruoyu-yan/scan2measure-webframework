"""
visualize_matching.py — Human-inspection overlay for pose estimation results.

For each of the 22 perspective crops, produces a side-by-side PNG:
  Left  : original photo crop + extracted 2D lines (green)
  Right : original photo crop + projected 3D wireframe segments (cyan)

Run after feature_matching.py.

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
LINE_THICKNESS = 2


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


def project_point(P_world, R_pano, t, R_crop):
    """
    Project a 3D world point into a perspective crop.

    R_pano : (3,3) world-to-pano rotation   (from camera_pose.json)
    t      : (3,)  camera position in world (from camera_pose.json)
    R_crop : (3,3) crop orientation matrix  (from make_view_rotation)

    Returns (u, v, depth) in crop pixel space. depth may be <= 0 (behind camera).
    """
    P_pano  = R_pano @ (P_world - t)       # world → panorama frame
    P_crop  = R_crop.T @ P_pano            # pano  → crop camera frame

    f  = focal()
    cx = (CROP_W - 1) / 2.0
    cy = (CROP_H - 1) / 2.0

    depth = P_crop[2]
    u = f * (P_crop[0] / depth) + cx if depth > 1e-4 else None
    v = -f * (P_crop[1] / depth) + cy if depth > 1e-4 else None
    return u, v, depth


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
        return None          # fully behind camera

    if c1[2] < eps:          # c1 behind, c2 in front → clip c1
        alpha = (eps - c1[2]) / (c2[2] - c1[2])
        c1 = c1 + alpha * (c2 - c1)

    if c2[2] < eps:          # c2 behind, c1 in front → clip c2
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


# ============================================================
# DRAW HELPERS
# ============================================================

def draw_2d_lines(img, lines):
    """Draw a list of [[u1,v1],[u2,v2]] pixel segments in green."""
    for (p1, p2) in lines:
        pt1 = (int(round(p1[0])), int(round(p1[1])))
        pt2 = (int(round(p2[0])), int(round(p2[1])))
        cv2.line(img, pt1, pt2, COLOR_2D, LINE_THICKNESS, cv2.LINE_AA)


def draw_3d_wireframe(img, wireframe_segs, R_pano, t, R_crop):
    """Project and draw 3D wireframe segments as cyan lines."""
    for seg in wireframe_segs:
        P1 = np.asarray(seg['start'], dtype=float)
        P2 = np.asarray(seg['end'],   dtype=float)

        clipped = clip_to_front(P1, P2, R_pano, t, R_crop)
        if clipped is None:
            continue
        c1, c2 = clipped

        u1, v1 = cam_to_pixel(c1)
        u2, v2 = cam_to_pixel(c2)

        # Skip if both projected pixels are far outside the image
        if not (pixel_in_bounds(u1, v1, margin=50) or
                pixel_in_bounds(u2, v2, margin=50)):
            continue

        # Clamp to image bounds for drawing
        u1c = int(np.clip(u1, 0, CROP_W - 1))
        v1c = int(np.clip(v1, 0, CROP_H - 1))
        u2c = int(np.clip(u2, 0, CROP_W - 1))
        v2c = int(np.clip(v2, 0, CROP_H - 1))

        cv2.line(img, (u1c, v1c), (u2c, v2c), COLOR_3D, LINE_THICKNESS, cv2.LINE_AA)


# ============================================================
# LEGEND
# ============================================================

def add_legend(img, label, color):
    cv2.rectangle(img, (8, 8), (22, 22), color, -1)
    cv2.putText(img, label, (28, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


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

        # --- Left panel: photo + 2D lines ---
        left = base.copy()
        draw_2d_lines(left, seg_list)
        add_legend(left, f"2D lines ({len(seg_list)})", COLOR_2D)

        # --- Right panel: photo + projected 3D wireframe ---
        right = base.copy()
        draw_3d_wireframe(right, wireframe_segs, R_pano, t, R_crop)
        add_legend(right, f"3D projected ({len(wireframe_segs)} segs)", COLOR_3D)

        # --- Compose side-by-side ---
        divider = np.full((CROP_H, 4, 3), 200, dtype=np.uint8)   # light grey divider
        panel   = np.hstack([left, divider, right])

        # Title bar
        title_h  = 36
        title    = np.zeros((title_h, panel.shape[1], 3), dtype=np.uint8)
        stem     = filename.replace('.jpg', '').replace('.png', '')
        cv2.putText(title, stem, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
        out_img = np.vstack([title, panel])

        out_path = output_dir / filename.replace('.jpg', '.png').replace('.jpeg', '.png')
        cv2.imwrite(str(out_path), out_img)
        n_saved += 1

    print(f"\nSaved {n_saved} comparison images → {output_dir}")
    print("Left panel  = extracted 2D lines (green)")
    print("Right panel = projected 3D wireframe segments (cyan) from estimated pose")


if __name__ == "__main__":
    main()
