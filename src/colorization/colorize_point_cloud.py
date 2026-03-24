"""Point cloud colorization using equirectangular panoramic images.

Colors a raw point cloud by projecting it onto multiple panoramic images
with known camera poses, using depth-buffer occlusion handling and
inverse-distance-weighted multi-panorama blending.

Usage:
    python src/colorization/colorize_point_cloud.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

_SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_ROOT / "colorization"))
from projection import project_points_to_pano
from visibility import compute_visibility_depth_buffer
from color_sampling import sample_colors_bilinear, blend_colors_idw

# ── Config ──────────────────────────────────────────────────────────────────
POINT_CLOUD_NAME = "tmb_office_one_corridor_bigger_noRGB"
PANO_NAMES = ["TMB_office1", "TMB_corridor_south1", "TMB_corridor_south2"]

# Depth buffer visibility
DEPTH_BUFFER_W = 2048
DEPTH_BUFFER_H = 1024
DEPTH_MARGIN = 0.05  # meters

# Blending
IDW_POWER = 2.0

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
PC_PATH = ROOT / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
PANO_DIR = ROOT / "data" / "pano" / "raw"
POSE_PATH = (ROOT / "data" / "pose_estimates" / "multiroom"
             / "local_filter_results.json")
OUTPUT_DIR = ROOT / "data" / "textured_point_cloud" / POINT_CLOUD_NAME


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # ================================================================
    # PHASE A: Setup
    # ================================================================
    print("=" * 60)
    print("PHASE A: Setup")
    print("=" * 60)

    # ── A1: Load point cloud ─────────────────────────────────────────
    print("\n[A1] Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(PC_PATH))
    points = np.asarray(pcd.points)
    N = points.shape[0]
    print(f"    {N:,} points loaded from {PC_PATH.name}")

    # ── A2: Load camera poses ────────────────────────────────────────
    print("\n[A2] Loading camera poses from local_filter_results.json...")
    with open(POSE_PATH) as f:
        pose_data = json.load(f)

    poses = {}
    for name in PANO_NAMES:
        entry = pose_data[name]
        R = np.array(entry['R'])
        t = np.array(entry['t'])
        poses[name] = (R, t)
        print(f"    {name}: t=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")

    # ── A3: Load panorama images ─────────────────────────────────────
    print("\n[A3] Loading panorama images...")
    pano_images = {}
    for name in PANO_NAMES:
        img_path = PANO_DIR / f"{name}.jpg"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pano_images[name] = img
        print(f"    {name}: {img.shape[1]}x{img.shape[0]}")

    img_h, img_w = next(iter(pano_images.values())).shape[:2]

    # ================================================================
    # PHASE B: Per-panorama projection + visibility + sampling
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE B: Per-panorama processing")
    print("=" * 60)

    all_colors = []
    all_depths = []
    all_indices = []

    for pano_name in PANO_NAMES:
        print(f"\n--- {pano_name} ---")
        R, t = poses[pano_name]
        image = pano_images[pano_name]
        t_pano = time.time()

        # ── B1: Project all points ───────────────────────────────────
        u, v, depths, valid_idx = project_points_to_pano(
            points, R, t, img_w, img_h)
        print(f"    [B1] Projected: {len(valid_idx):,} valid points "
              f"(min_depth > 0.1m)")

        # ── B2: Depth-buffer visibility ──────────────────────────────
        vis_mask = compute_visibility_depth_buffer(
            u, v, depths, img_w, img_h,
            buffer_w=DEPTH_BUFFER_W, buffer_h=DEPTH_BUFFER_H,
            depth_margin=DEPTH_MARGIN)
        n_visible = vis_mask.sum()
        print(f"    [B2] Visible: {n_visible:,} / {len(valid_idx):,} "
              f"({100 * n_visible / max(len(valid_idx), 1):.1f}%)")

        # Apply visibility mask
        u_vis = u[vis_mask]
        v_vis = v[vis_mask]
        depths_vis = depths[vis_mask]
        indices_vis = valid_idx[vis_mask]

        # ── B3: Bilinear color sampling ──────────────────────────────
        colors = sample_colors_bilinear(image, u_vis, v_vis)
        print(f"    [B3] Sampled {colors.shape[0]:,} colors  "
              f"({time.time() - t_pano:.1f}s)")

        all_colors.append(colors)
        all_depths.append(depths_vis)
        all_indices.append(indices_vis)

    # ================================================================
    # PHASE C: Blending + output
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE C: Blending + output")
    print("=" * 60)

    # ── C1: Inverse-distance-weighted blending ───────────────────────
    print(f"\n[C1] Blending colors across panoramas "
          f"(IDW, power={IDW_POWER:.1f})...")
    final_colors, colored_mask = blend_colors_idw(
        all_colors, all_depths, all_indices, N, power=IDW_POWER)
    n_colored = colored_mask.sum()
    print(f"    Colored: {n_colored:,} / {N:,} ({100 * n_colored / N:.1f}%)")

    # Count per-point panorama coverage
    coverage = np.zeros(N, dtype=np.int32)
    for indices in all_indices:
        coverage[indices] += 1
    for k in range(1, len(PANO_NAMES) + 1):
        nk = (coverage == k).sum()
        if nk > 0:
            print(f"    Points seen by {k} pano(s): {nk:,}")

    # ── C2: Uncolored points ─────────────────────────────────────────
    n_uncolored = N - n_colored
    if n_uncolored > 0:
        print(f"\n[C2] {n_uncolored:,} uncolored points left as black")

    # ── C3: Save colored PLY ─────────────────────────────────────────
    print("\n[C3] Saving colored point cloud...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}_textured.ply"

    pcd.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.io.write_point_cloud(str(out_path), pcd)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {out_path}  ({size_mb:.1f} MB)")

    # ── C4: Summary ──────────────────────────────────────────────────
    dt = time.time() - t_start
    print(f"\nTotal time: {dt:.1f}s")


if __name__ == "__main__":
    main()
