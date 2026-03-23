"""Library helpers for mesh reconstruction from colored point clouds.

Provides normal estimation, Poisson reconstruction, density-based cleanup,
vertex color transfer, and quadric decimation.
"""

import numpy as np
import open3d as o3d
import xatlas
from scipy.spatial import cKDTree


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
    w = max(atlas.width, 1)
    h = max(atlas.height, 1)
    uv_coords[:, 0] /= w
    uv_coords[:, 1] /= h
    return vmapping, new_faces, uv_coords


def bake_texture_atlas(vertices, faces, uv_coords, source_points, source_colors,
                       atlas_resolution=4096, knn=8):
    """Bake a texture atlas by projecting source point cloud colors onto UV space.

    For each texel covered by a triangle, interpolates 3D position via barycentric
    coordinates, then queries the nearest source points via KD-tree and blends their
    colors using inverse-distance weighting.

    Args:
        vertices: (N, 3) float64 mesh vertex positions.
        faces: (F, 3) int array of triangle indices.
        uv_coords: (N, 2) float64 UV coordinates in [0, 1] per vertex.
        source_points: (P, 3) float64 colored point cloud positions.
        source_colors: (P, 3) float64 RGB colors in [0, 1].
        atlas_resolution: Width and height of the output atlas in pixels.
        knn: Number of nearest source points used for IDW color blending.

    Returns:
        atlas: (atlas_resolution, atlas_resolution, 3) uint8 RGB image.
    """
    res = atlas_resolution
    atlas = np.zeros((res, res, 3), dtype=np.uint8)
    tree = cKDTree(source_points)
    face_uvs = uv_coords[faces]
    face_verts = vertices[faces]
    n_faces = len(faces)
    batch_size = 10000
    for batch_start in range(0, n_faces, batch_size):
        batch_end = min(batch_start + batch_size, n_faces)
        all_pixels_u = []
        all_pixels_v = []
        all_pos_3d = []
        for fi in range(batch_start, batch_end):
            uvs = face_uvs[fi]
            verts = face_verts[fi]
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
            pos = (wb[:, None] * verts[0] + ub[:, None] * verts[1] + vb[:, None] * verts[2])
            all_pixels_u.append(ins_pix[:, 0])
            all_pixels_v.append(ins_pix[:, 1])
            all_pos_3d.append(pos)
        if not all_pos_3d:
            continue
        batch_u = np.concatenate(all_pixels_u)
        batch_v = np.concatenate(all_pixels_v)
        batch_pos = np.concatenate(all_pos_3d)
        dists, idxs = tree.query(batch_pos, k=knn)
        weights = 1.0 / np.maximum(dists, 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        colors = np.zeros((len(batch_pos), 3))
        for k_i in range(knn):
            colors += weights[:, k_i:k_i+1] * source_colors[idxs[:, k_i]]
        colors_uint8 = np.clip(colors * 255, 0, 255).astype(np.uint8)
        atlas[batch_v, batch_u] = colors_uint8
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
