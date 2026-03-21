"""
ply_to_xyz.py — Convert a binary/ASCII PLY point cloud to plain-text XYZ.

Usage:
    python src/ply_to_xyz.py <input.ply> <output.txt>

Output format (one point per line):
    x y z
"""

import sys
from pathlib import Path
import open3d as o3d
import numpy as np


def main():
    if len(sys.argv) != 3:
        print("Usage: python ply_to_xyz.py <input.ply> <output.txt>")
        sys.exit(1)

    ply_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    print(f"Loading {ply_path} ...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    print(f"  {len(pts):,} points")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out_path} ...")
    np.savetxt(str(out_path), pts, fmt="%.6f")
    print("Done.")


if __name__ == "__main__":
    main()
