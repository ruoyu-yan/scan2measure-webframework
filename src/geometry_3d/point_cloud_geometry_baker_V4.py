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
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from config_loader import load_config, progress

# ============================================================
# CONFIGURATION
# ============================================================
POINT_CLOUD_NAME = "tmb_office_one_corridor_bigger_noRGB"   # input PLY name (without .ply) and output folder name
KNN              = 20              # k neighbours passed to 3DLineDetection

# ============================================================
# PATHS
# ============================================================
project_root = Path(__file__).resolve().parent.parent.parent
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
    cfg = load_config()
    pc_name = cfg.get("point_cloud_name", POINT_CLOUD_NAME)
    knn = cfg.get("knn", KNN)
    _ply_path = Path(cfg["point_cloud_path"]) if cfg.get("point_cloud_path") else project_root / "data" / "raw_point_cloud" / f"{pc_name}.ply"
    _out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else project_root / "data" / "debug_renderer" / pc_name
    _out_dir.mkdir(parents=True, exist_ok=True)
    _tmp_xyz = _out_dir / "tmp_points.txt"
    _obj_prefix = _out_dir / f"{pc_name}_"

    # Step 1 — PLY → XYZ
    progress(1, 4, "Converting PLY to XYZ")
    print(f"Reading point cloud: {_ply_path}")
    pcd = o3d.io.read_point_cloud(str(_ply_path))
    pts = np.asarray(pcd.points)
    print(f"  {len(pts):,} points — writing {_tmp_xyz.name} ...")
    np.savetxt(str(_tmp_xyz), pts, fmt="%.6f")

    # Step 2 — Run binary
    progress(2, 4, "Running 3DLineDetection binary")
    print(f"Running 3DLineDetection binary ...")
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/home/ruoyu/miniconda3/lib:' + env.get('LD_LIBRARY_PATH', '')
    subprocess.run(
        [str(binary), str(_tmp_xyz), str(_obj_prefix)],
        env=env, check=True
    )

    # Step 3 — Parse lines.obj
    progress(3, 4, "Parsing line segments from OBJ")
    lines_obj = _out_dir / f"{pc_name}_lines.obj"
    print(f"Parsing {lines_obj.name} ...")
    wireframe_segments = parse_lines_obj(lines_obj)
    print(f"  {len(wireframe_segments)} line segments found")

    # Step 4 — Save PKL + cleanup
    progress(4, 4, "Saving room_geometry.pkl")
    pkl_path = _out_dir / "room_geometry.pkl"
    bake_data = {
        'wireframe_segments': wireframe_segments,
        'detected_planes': []
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(bake_data, f)

    _tmp_xyz.unlink()
    print(f"\nOutputs:")
    print(f"  {pkl_path}")
    print(f"  {lines_obj}")
    print(f"  {_out_dir / f'{pc_name}_planes.obj'}")
    print(f"  tmp_points.txt removed")


if __name__ == "__main__":
    main()
