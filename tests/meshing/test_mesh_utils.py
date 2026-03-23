import numpy as np
import open3d as o3d
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "meshing"))
from mesh_utils import compute_tile_grid, extract_tile_points


def test_compute_tile_grid_single_tile():
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([5.0, 5.0, 3.0])
    tiles = compute_tile_grid(bbox_min, bbox_max, tile_size=6.0, overlap=1.0)
    assert len(tiles) == 1
    core_min, core_max, ext_min, ext_max = tiles[0]
    assert core_min[0] <= 0.0 and core_max[0] >= 5.0
    assert core_min[1] <= 0.0 and core_max[1] >= 5.0


def test_compute_tile_grid_multiple_tiles():
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([12.0, 12.0, 3.0])
    tiles = compute_tile_grid(bbox_min, bbox_max, tile_size=6.0, overlap=1.0)
    assert len(tiles) == 4


def test_compute_tile_grid_returns_core_and_extended_bounds():
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([12.0, 12.0, 3.0])
    tiles = compute_tile_grid(bbox_min, bbox_max, tile_size=6.0, overlap=1.0)
    for core_min, core_max, ext_min, ext_max in tiles:
        assert np.all(ext_min[:2] <= core_min[:2])
        assert np.all(ext_max[:2] >= core_max[:2])
        assert ext_min[2] == bbox_min[2]
        assert ext_max[2] == bbox_max[2]


def test_extract_tile_points():
    points = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 1.0], [10.0, 10.0, 1.0], [15.0, 15.0, 1.0]])
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    ext_min = np.array([0.0, 0.0, 0.0])
    ext_max = np.array([6.0, 6.0, 3.0])
    tile_pcd = extract_tile_points(pcd, ext_min, ext_max)
    assert len(tile_pcd.points) == 2
    assert tile_pcd.has_colors()
