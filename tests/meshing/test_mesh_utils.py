import numpy as np
import open3d as o3d
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "meshing"))
from mesh_utils import compute_tile_grid, extract_tile_points, trim_to_ownership_region, merge_tile_meshes, uv_unwrap_mesh, bake_texture_atlas, dilate_texture


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


def test_trim_to_ownership_region():
    vertices = np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0],
        [7.0, 0.0, 0.0], [9.0, 0.0, 0.0], [8.0, 2.0, 0.0],
    ], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    core_min = np.array([0.0, 0.0, -1.0])
    core_max = np.array([6.0, 6.0, 1.0])
    trimmed = trim_to_ownership_region(mesh, core_min, core_max)
    assert len(trimmed.triangles) == 1
    verts = np.asarray(trimmed.vertices)
    tris = np.asarray(trimmed.triangles)
    centroid = verts[tris[0]].mean(axis=0)
    assert centroid[0] < 6.0


def test_merge_tile_meshes():
    mesh_a = o3d.geometry.TriangleMesh()
    mesh_a.vertices = o3d.utility.Vector3dVector(np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64))
    mesh_a.triangles = o3d.utility.Vector3iVector(np.array([[0,1,2]]))
    mesh_b = o3d.geometry.TriangleMesh()
    mesh_b.vertices = o3d.utility.Vector3dVector(np.array([[5,0,0],[6,0,0],[5,1,0]], dtype=np.float64))
    mesh_b.triangles = o3d.utility.Vector3iVector(np.array([[0,1,2]]))
    merged = merge_tile_meshes([mesh_a, mesh_b])
    assert len(merged.vertices) == 6
    assert len(merged.triangles) == 2


def test_merge_tile_meshes_empty_list():
    merged = merge_tile_meshes([])
    assert len(merged.vertices) == 0
    assert len(merged.triangles) == 0


def test_uv_unwrap_mesh():
    mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vmapping, new_faces, uv_coords = uv_unwrap_mesh(vertices, faces, atlas_resolution=512)
    assert uv_coords.ndim == 2 and uv_coords.shape[1] == 2
    assert np.all(uv_coords >= 0.0) and np.all(uv_coords <= 1.0)
    assert len(vmapping) == len(uv_coords)
    assert new_faces.shape[1] == 3


def test_bake_texture_atlas():
    vertices = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
    faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
    uv_coords = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float64)
    source_points = np.array([[0.25,0.25,0],[0.75,0.25,0],[0.25,0.75,0],[0.75,0.75,0]])
    source_colors = np.full((4, 3), [1.0, 0.0, 0.0])
    atlas = bake_texture_atlas(vertices, faces, uv_coords, source_points, source_colors, atlas_resolution=64, knn=4)
    assert atlas.shape == (64, 64, 3)
    assert atlas.dtype == np.uint8
    nonzero = atlas[atlas.sum(axis=2) > 0]
    if len(nonzero) > 0:
        avg_color = nonzero.mean(axis=0)
        assert avg_color[0] > 200


def test_dilate_texture():
    atlas = np.zeros((8, 8, 3), dtype=np.uint8)
    atlas[4, 4] = [255, 0, 0]
    dilated = dilate_texture(atlas, iterations=2)
    assert np.array_equal(dilated[4, 4], [255, 0, 0])
    assert dilated[3, 4].sum() > 0 or dilated[5, 4].sum() > 0
