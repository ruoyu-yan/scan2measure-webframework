"""
point_cloud_geometry_baker_V4.py — Python wrapper for the 3DLineDetection C++ binary.

Steps:
  1. Convert .ply point cloud to plain-text XYZ
  2. Run 3DLineDetection/build/src/LineFromPointCloud
  3. Parse lines.obj → wireframe_segments
  4. Save room_geometry.pkl consumed by feature_matching.py and visualize_matching.py
"""

import os
import pickle
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d

# ============================================================
# CONFIGURATION
# ============================================================
POINT_CLOUD_NAME = "tmb_office_one_corridor_dense"   # input PLY name (without .ply) and output folder name
KNN              = 20              # k neighbours passed to 3DLineDetection

# ============================================================
# PATHS
# ============================================================
project_root = Path(__file__).resolve().parent.parent
ply_path     = project_root / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
out_dir      = project_root / "data" / "debug_renderer" / POINT_CLOUD_NAME
binary       = project_root / "3DLineDetection" / "build" / "src" / "LineFromPointCloud"
tmp_xyz      = out_dir / "tmp_points.txt"
obj_prefix   = out_dir / f"{POINT_CLOUD_NAME}_"


def parse_lines_obj(obj_path):
    """Parse an OBJ file containing vertices and line segments."""
    verts = []
    segments = []
    with open(obj_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'l':
                i, j = int(parts[1]) - 1, int(parts[2]) - 1  # OBJ is 1-indexed
                segments.append({'start': verts[i], 'end': verts[j]})
    return segments


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — PLY → XYZ
    print(f"Reading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    print(f"  {len(pts):,} points — writing {tmp_xyz.name} ...")
    np.savetxt(str(tmp_xyz), pts, fmt="%.6f")

    # Step 2 — Run binary
    print(f"Running 3DLineDetection binary ...")
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/home/ruoyu/miniconda3/lib:' + env.get('LD_LIBRARY_PATH', '')
    subprocess.run(
        [str(binary), str(tmp_xyz), str(obj_prefix)],
        env=env, check=True
    )

    # Step 3 — Parse lines.obj
    lines_obj = out_dir / f"{POINT_CLOUD_NAME}_lines.obj"
    print(f"Parsing {lines_obj.name} ...")
    wireframe_segments = parse_lines_obj(lines_obj)
    print(f"  {len(wireframe_segments)} line segments found")

    # Step 4 — Save PKL + cleanup
    pkl_path = out_dir / "room_geometry.pkl"
    bake_data = {
        'wireframe_segments': wireframe_segments,
        'detected_planes': []
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(bake_data, f)

    tmp_xyz.unlink()
    print(f"\nOutputs:")
    print(f"  {pkl_path}")
    print(f"  {lines_obj}")
    print(f"  {out_dir / f'{POINT_CLOUD_NAME}_planes.obj'}")
    print(f"  tmp_points.txt removed")


if __name__ == "__main__":
    main()
