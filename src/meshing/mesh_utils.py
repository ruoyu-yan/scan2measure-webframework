"""Library helpers for mesh reconstruction from colored point clouds.

Provides normal estimation, Poisson reconstruction, density-based cleanup,
vertex color transfer, spatial tiling, UV unwrapping, and texture atlas baking.
Includes parallel execution support for tile processing and texture baking.
"""

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import open3d as o3d
import xatlas
from scipy.spatial import cKDTree

# Use 'spawn' context to avoid fork+thread deadlocks with Open3D
_MP_CONTEXT = multiprocessing.get_context('spawn')


def estimate_normals(pcd, knn=30, radius=0.1):
    """Estimate and orient normals for a point cloud.

    Args:
        pcd: Open3D PointCloud with points.
        knn: Number of nearest neighbors for normal estimation.
        radius: Search radius for hybrid KNN.

    Returns:
        The same PointCloud with normals estimated and consistently oriented.
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=knn
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=knn)
    return pcd


def poisson_reconstruct(pcd, depth=9, width=0, scale=1.1, linear_fit=False):
    """Run Poisson surface reconstruction.

    Args:
        pcd: Open3D PointCloud with normals.
        depth: Octree depth — higher = more detail but slower.
            depth=9 is a good balance for TLS data (~8M points).
        width: Target width of the finest octree cells (0 = use depth).
        scale: Ratio between the diameter of the cube used for reconstruction
            and the diameter of the samples' bounding cube.
        linear_fit: Use linear interpolation for iso-surface extraction.

    Returns:
        (mesh, densities): TriangleMesh and per-vertex density array.
    """
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )
    densities = np.asarray(densities)
    return mesh, densities


def remove_low_density(mesh, densities, quantile=0.01):
    """Remove vertices with density below a quantile threshold.

    Poisson reconstruction extrapolates beyond the actual data, creating
    artifacts in unscanned regions. Density-based trimming removes these.

    Args:
        mesh: Open3D TriangleMesh from Poisson reconstruction.
        densities: Per-vertex density array from Poisson reconstruction.
        quantile: Density quantile threshold — vertices below this are removed.

    Returns:
        Trimmed TriangleMesh.
    """
    threshold = np.quantile(densities, quantile)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh


def transfer_vertex_colors(mesh, source_pcd):
    """Transfer colors from a point cloud to mesh vertices via batch KD-tree.

    Args:
        mesh: Open3D TriangleMesh (vertices will receive colors).
        source_pcd: Open3D PointCloud with colors.

    Returns:
        The mesh with vertex colors set.
    """
    source_points = np.asarray(source_pcd.points)
    source_colors = np.asarray(source_pcd.colors)
    mesh_vertices = np.asarray(mesh.vertices)
    tree = cKDTree(source_points)
    _, idx = tree.query(mesh_vertices, k=1)
    colors = source_colors[idx]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


# ── Parallel tile processing ────────────────────────────────────────────────

def _process_single_tile(tile_idx, core_min, core_max, tile_points, tile_colors,
                         tile_ply_path, normal_knn, normal_radius, poisson_depth,
                         density_trim_quantile):
    """Worker: reconstruct one tile in a subprocess. Returns picklable tuple."""
    t = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tile_points)
    if tile_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(tile_colors)

    pcd = estimate_normals(pcd, knn=normal_knn, radius=normal_radius)
    mesh, densities = poisson_reconstruct(pcd, depth=poisson_depth)
    mesh = remove_low_density(mesh, densities, quantile=density_trim_quantile)
    del densities

    if len(mesh.triangles) == 0:
        return (tile_idx, None, 0, 0, time.time() - t)

    mesh = trim_to_ownership_region(mesh, core_min, core_max)

    if len(mesh.triangles) == 0:
        del pcd
        return (tile_idx, None, 0, 0, time.time() - t)

    if pcd.has_colors():
        mesh = transfer_vertex_colors(mesh, pcd)
    del pcd

    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    o3d.io.write_triangle_mesh(tile_ply_path, mesh)
    del mesh
    return (tile_idx, tile_ply_path, n_verts, n_tris, time.time() - t)


def process_tiles_parallel(pcd_ds, tiles, tiles_dir, normal_knn=50,
                           normal_radius=0.15, poisson_depth=9,
                           density_trim_quantile=0.06, min_tile_points=1000,
                           max_workers=None):
    """Process all tiles in parallel using ProcessPoolExecutor.

    Pre-extracts each tile's points as numpy arrays in the main process,
    then dispatches independent worker processes for Poisson reconstruction.

    Returns:
        (tile_ply_paths, n_skipped, tile_results)
        tile_results: list of (tile_idx, ply_path_or_None, n_verts, n_tris, elapsed_s)
    """
    n_tiles = len(tiles)
    if max_workers is None:
        max_workers = min(n_tiles, max(1, os.cpu_count() // 8))

    # Pre-extract tile numpy arrays in main process
    points_arr = np.asarray(pcd_ds.points)
    colors_arr = np.asarray(pcd_ds.colors) if pcd_ds.has_colors() else None

    tile_jobs = []
    n_skipped = 0
    for tile_idx, (core_min, core_max, ext_min, ext_max) in enumerate(tiles):
        mask = np.all((points_arr >= ext_min) & (points_arr <= ext_max), axis=1)
        tile_points = points_arr[mask]
        n_tile_pts = len(tile_points)

        if n_tile_pts < min_tile_points:
            print(f"       Tile {tile_idx+1}/{n_tiles}: "
                  f"skipped ({n_tile_pts} pts < {min_tile_points})")
            n_skipped += 1
            continue

        tile_colors = colors_arr[mask] if colors_arr is not None else None
        tile_ply_path = str(Path(tiles_dir) / f"tile_{tile_idx}.ply")

        tile_jobs.append((tile_idx, core_min, core_max, tile_points, tile_colors,
                          tile_ply_path, normal_knn, normal_radius, poisson_depth,
                          density_trim_quantile))
        print(f"       Tile {tile_idx+1}/{n_tiles}: {n_tile_pts:,} pts -> dispatched")

    if not tile_jobs:
        return [], n_skipped, []

    actual_workers = min(max_workers, len(tile_jobs))
    print(f"       Processing {len(tile_jobs)} tiles with {actual_workers} workers ...")

    tile_results = []
    with ProcessPoolExecutor(max_workers=actual_workers,
                             mp_context=_MP_CONTEXT) as executor:
        futures = {
            executor.submit(_process_single_tile, *args): args[0]
            for args in tile_jobs
        }
        for future in as_completed(futures):
            result = future.result()
            tile_results.append(result)
            tidx, ply_path, n_verts, n_tris, elapsed = result
            if ply_path is not None:
                print(f"       Tile {tidx+1}/{n_tiles}: "
                      f"{n_verts:,}v / {n_tris:,}t in {elapsed:.1f}s")
            else:
                n_skipped += 1
                print(f"       Tile {tidx+1}/{n_tiles}: "
                      f"skipped (0 triangles) in {elapsed:.1f}s")

    # Sort by tile index for deterministic merge order
    tile_results.sort(key=lambda x: x[0])
    tile_ply_paths = [Path(r[1]) for r in tile_results if r[1] is not None]

    return tile_ply_paths, n_skipped, tile_results


# ── Mesh statistics & tiling ────────────────────────────────────────────────

def compute_mesh_stats(mesh):
    """Compute basic mesh statistics.

    Returns:
        dict with n_vertices, n_triangles, bbox_min, bbox_max, bbox_extent.
    """
    vertices = np.asarray(mesh.vertices)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return {
        "n_vertices": len(mesh.vertices),
        "n_triangles": len(mesh.triangles),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_extent": (bbox_max - bbox_min).tolist(),
    }


def compute_tile_grid(bbox_min, bbox_max, tile_size=6.0, overlap=1.0):
    """Divide XY bounding box into a grid of tiles with overlap."""
    tiles = []
    x_min, y_min, z_min = bbox_min
    x_max, y_max, z_max = bbox_max
    x_edges = np.arange(x_min, x_max, tile_size)
    y_edges = np.arange(y_min, y_max, tile_size)
    for x_start in x_edges:
        for y_start in y_edges:
            core_min = np.array([x_start, y_start, z_min])
            core_max = np.array([min(x_start + tile_size, x_max),
                                 min(y_start + tile_size, y_max), z_max])
            ext_min = np.array([x_start - overlap, y_start - overlap, z_min])
            ext_max = np.array([min(x_start + tile_size + overlap, x_max + overlap),
                                min(y_start + tile_size + overlap, y_max + overlap), z_max])
            tiles.append((core_min, core_max, ext_min, ext_max))
    return tiles


def extract_tile_points(pcd, ext_min, ext_max):
    """Extract points within the extended tile bounds."""
    points = np.asarray(pcd.points)
    mask = np.all((points >= ext_min) & (points <= ext_max), axis=1)
    tile_pcd = o3d.geometry.PointCloud()
    tile_pcd.points = o3d.utility.Vector3dVector(points[mask])
    if pcd.has_colors():
        tile_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        tile_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    return tile_pcd


def trim_to_ownership_region(mesh, core_min, core_max):
    """Remove triangles whose centroids fall outside the core tile bounds."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    centroids = vertices[triangles].mean(axis=1)
    mask = (
        (centroids[:, 0] >= core_min[0]) & (centroids[:, 0] <= core_max[0]) &
        (centroids[:, 1] >= core_min[1]) & (centroids[:, 1] <= core_max[1])
    )
    kept_triangles = triangles[mask]
    trimmed = o3d.geometry.TriangleMesh()
    trimmed.vertices = o3d.utility.Vector3dVector(vertices)
    trimmed.triangles = o3d.utility.Vector3iVector(kept_triangles)
    if mesh.has_vertex_colors():
        trimmed.vertex_colors = mesh.vertex_colors
    trimmed.remove_unreferenced_vertices()
    return trimmed


def merge_tile_meshes(meshes):
    """Concatenate multiple tile meshes into a single mesh."""
    if not meshes:
        return o3d.geometry.TriangleMesh()
    all_vertices = []
    all_triangles = []
    all_colors = []
    vertex_offset = 0
    for mesh in meshes:
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        all_vertices.append(verts)
        all_triangles.append(tris + vertex_offset)
        if mesh.has_vertex_colors():
            all_colors.append(np.asarray(mesh.vertex_colors))
        vertex_offset += len(verts)
    merged = o3d.geometry.TriangleMesh()
    merged.vertices = o3d.utility.Vector3dVector(np.vstack(all_vertices))
    merged.triangles = o3d.utility.Vector3iVector(np.vstack(all_triangles))
    if all_colors and len(all_colors) == len(meshes):
        merged.vertex_colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    merged.remove_degenerate_triangles()
    merged.compute_vertex_normals()
    return merged


# ── UV unwrapping & texture baking ──────────────────────────────────────────

def uv_unwrap_mesh(vertices, faces, atlas_resolution=4096):
    """Compute UV coordinates for a mesh using xatlas.

    Args:
        vertices: (N, 3) float64 array of vertex positions.
        faces: (F, 3) int array of triangle indices.
        atlas_resolution: Target resolution hint for the atlas packer.

    Returns:
        (vmapping, new_faces, uv_coords): vmapping maps new vertices to original,
        new_faces are the remapped triangle indices, uv_coords are (M, 2) in [0, 1].
    """
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32))
    pack_options = xatlas.PackOptions()
    pack_options.resolution = atlas_resolution
    atlas.generate(xatlas.ChartOptions(), pack_options)
    vmapping, new_faces, uv_coords = atlas[0]
    uv_coords = uv_coords.astype(np.float64)
    return vmapping, new_faces, uv_coords


# Module-level state for bake workers (set by initializer in spawn context)
_bake_face_uvs = None
_bake_face_colors = None


def _bake_worker_init(face_uvs, face_colors):
    """Initializer for bake worker processes — sets module globals."""
    global _bake_face_uvs, _bake_face_colors
    _bake_face_uvs = face_uvs
    _bake_face_colors = face_colors


def _bake_face_chunk(face_start, face_end, atlas_resolution):
    """Worker: bake a chunk of faces by interpolating vertex colors."""
    face_uvs = _bake_face_uvs
    face_colors = _bake_face_colors

    res = atlas_resolution
    atlas = np.zeros((res, res, 3), dtype=np.uint8)

    for fi in range(face_start, face_end):
        uvs = face_uvs[fi]       # (3, 2)
        cols = face_colors[fi]   # (3, 3) RGB float [0, 1]
        pix = (uvs * (res - 1)).astype(np.int32)
        u_min = max(pix[:, 0].min(), 0)
        u_max = min(pix[:, 0].max(), res - 1)
        v_min = max(pix[:, 1].min(), 0)
        v_max = min(pix[:, 1].max(), res - 1)
        if u_min > u_max or v_min > v_max:
            continue
        us = np.arange(u_min, u_max + 1)
        vs = np.arange(v_min, v_max + 1)
        grid_u, grid_v = np.meshgrid(us, vs)
        pixels = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)
        if len(pixels) == 0:
            continue
        p = pixels.astype(np.float64)
        v0 = pix[1].astype(np.float64) - pix[0].astype(np.float64)
        v1 = pix[2].astype(np.float64) - pix[0].astype(np.float64)
        v2 = p - pix[0].astype(np.float64)
        dot00 = v0 @ v0
        dot01 = v0 @ v1
        dot11 = v1 @ v1
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:
            continue
        dot02 = v2 @ v0
        dot12 = v2 @ v1
        u_b = (dot11 * dot02 - dot01 * dot12) / denom
        v_b = (dot00 * dot12 - dot01 * dot02) / denom
        inside = (u_b >= -0.01) & (v_b >= -0.01) & (u_b + v_b <= 1.01)
        if not inside.any():
            continue
        ins_pix = pixels[inside]
        ub = u_b[inside]
        vb = v_b[inside]
        wb = 1.0 - ub - vb
        interp = wb[:, None] * cols[0] + ub[:, None] * cols[1] + vb[:, None] * cols[2]
        colors_uint8 = np.clip(interp * 255, 0, 255).astype(np.uint8)
        atlas[ins_pix[:, 1], ins_pix[:, 0]] = colors_uint8
    return atlas


def bake_texture_atlas(faces, uv_coords, vertex_colors,
                       atlas_resolution=4096, max_workers=1):
    """Bake a texture atlas by interpolating vertex colors via barycentrics.

    For each texel covered by a triangle, computes barycentric weights in UV
    space and blends the three vertex colors accordingly.  This avoids KNN
    queries against the point cloud and eliminates cross-surface color bleeding.

    Args:
        faces: (F, 3) int array of triangle indices.
        uv_coords: (N, 2) float64 UV coordinates in [0, 1] per vertex.
        vertex_colors: (N, 3) float64 RGB colors in [0, 1] per vertex.
        atlas_resolution: Width and height of the output atlas in pixels.
        max_workers: Number of parallel workers (1 = sequential).

    Returns:
        atlas: (atlas_resolution, atlas_resolution, 3) uint8 RGB image.
    """
    res = atlas_resolution
    face_uvs = uv_coords[faces]           # (F, 3, 2)
    face_colors = vertex_colors[faces]     # (F, 3, 3)
    n_faces = len(faces)

    # ── Parallel path ────────────────────────────────────────────────────────
    if max_workers > 1:
        chunk_size = max(1, n_faces // max_workers)
        face_chunks = []
        for i in range(0, n_faces, chunk_size):
            face_chunks.append((i, min(i + chunk_size, n_faces)))

        actual_workers = min(max_workers, len(face_chunks))
        atlas = np.zeros((res, res, 3), dtype=np.uint8)
        with ProcessPoolExecutor(
            max_workers=actual_workers,
            mp_context=_MP_CONTEXT,
            initializer=_bake_worker_init,
            initargs=(face_uvs, face_colors),
        ) as executor:
            futures = [
                executor.submit(_bake_face_chunk, start, end, res)
                for start, end in face_chunks
            ]
            for future in futures:
                partial = future.result()
                mask = partial.sum(axis=2) > 0
                atlas[mask] = partial[mask]
        return atlas

    # ── Sequential path ─────────────────────────────────────────────────────
    atlas = np.zeros((res, res, 3), dtype=np.uint8)
    for fi in range(n_faces):
        uvs = face_uvs[fi]
        cols = face_colors[fi]
        pix = (uvs * (res - 1)).astype(np.int32)
        u_min = max(pix[:, 0].min(), 0)
        u_max = min(pix[:, 0].max(), res - 1)
        v_min = max(pix[:, 1].min(), 0)
        v_max = min(pix[:, 1].max(), res - 1)
        if u_min > u_max or v_min > v_max:
            continue
        us = np.arange(u_min, u_max + 1)
        vs = np.arange(v_min, v_max + 1)
        grid_u, grid_v = np.meshgrid(us, vs)
        pixels = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)
        if len(pixels) == 0:
            continue
        p = pixels.astype(np.float64)
        v0 = pix[1].astype(np.float64) - pix[0].astype(np.float64)
        v1 = pix[2].astype(np.float64) - pix[0].astype(np.float64)
        v2 = p - pix[0].astype(np.float64)
        dot00 = v0 @ v0
        dot01 = v0 @ v1
        dot11 = v1 @ v1
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:
            continue
        dot02 = v2 @ v0
        dot12 = v2 @ v1
        u_b = (dot11 * dot02 - dot01 * dot12) / denom
        v_b = (dot00 * dot12 - dot01 * dot02) / denom
        inside = (u_b >= -0.01) & (v_b >= -0.01) & (u_b + v_b <= 1.01)
        if not inside.any():
            continue
        ins_pix = pixels[inside]
        ub = u_b[inside]
        vb = v_b[inside]
        wb = 1.0 - ub - vb
        interp = wb[:, None] * cols[0] + ub[:, None] * cols[1] + vb[:, None] * cols[2]
        colors_uint8 = np.clip(interp * 255, 0, 255).astype(np.uint8)
        atlas[ins_pix[:, 1], ins_pix[:, 0]] = colors_uint8
    return atlas


def dilate_texture(atlas, iterations=8):
    """Fill empty (black) texels by averaging neighboring filled texels.

    Iteratively expands filled regions into adjacent empty pixels, which
    eliminates black seams at UV chart boundaries when mipmapping is applied.

    Args:
        atlas: (H, W, 3) uint8 RGB texture atlas.
        iterations: Number of dilation passes.

    Returns:
        Dilated atlas with the same shape and dtype as the input.
    """
    result = atlas.copy()
    empty_mask = result.sum(axis=2) == 0
    for _ in range(iterations):
        if not empty_mask.any():
            break
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode='edge')
        padded_mask = np.pad(~empty_mask, ((1, 1), (1, 1)), mode='constant', constant_values=False)
        h, w = result.shape[:2]
        neighbor_sum = np.zeros_like(result, dtype=np.float64)
        neighbor_count = np.zeros((h, w), dtype=np.float64)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = 1 + dy, 1 + dx
                neighbor_sum += padded[ny:ny+h, nx:nx+w].astype(np.float64) * padded_mask[ny:ny+h, nx:nx+w][:, :, None]
                neighbor_count += padded_mask[ny:ny+h, nx:nx+w].astype(np.float64)
        fillable = empty_mask & (neighbor_count > 0)
        if not fillable.any():
            break
        avg = neighbor_sum[fillable] / neighbor_count[fillable][:, None]
        result[fillable] = np.clip(avg, 0, 255).astype(np.uint8)
        empty_mask[fillable] = False
    return result
