"""Downsample a PLY point cloud for Electron 3D preview.

Usage:
    python downsample_for_preview.py --config <path_to_json>

Config keys:
    input_path  -- full path to input PLY
    output_path -- full path to output downsampled PLY
"""

import shutil

import numpy as np
import open3d as o3d
from pathlib import Path

from config_loader import load_config


def main():
    cfg = load_config()

    input_path = Path(cfg["input_path"])
    output_path = Path(cfg["output_path"])

    print(f"Loading point cloud from {input_path}")
    pcd = o3d.io.read_point_cloud(str(input_path))
    n_original = len(pcd.points)
    print(f"Original point count: {n_original}")

    target_points = 200_000

    if n_original <= target_points:
        print(f"Point count ({n_original}) already <= {target_points}, copying as-is.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(input_path), str(output_path))
        print(f"Copied point cloud to {output_path}")
        return

    # Compute voxel size to reach ~200K points
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.max(bbox.get_extent())
    voxel_size = extent / (target_points ** (1 / 3))
    print(f"Bounding box extent: {extent:.3f} m, voxel size: {voxel_size:.4f} m")

    pcd_down = pcd.voxel_down_sample(voxel_size)
    n_down = len(pcd_down.points)
    print(f"Downsampled point count: {n_down}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd_down)
    print(f"Saved downsampled point cloud to {output_path}")


if __name__ == "__main__":
    main()
