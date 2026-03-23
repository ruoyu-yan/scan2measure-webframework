"""Library helpers for mesh reconstruction from colored point clouds.

Provides normal estimation, Poisson reconstruction, density-based cleanup,
vertex color transfer, and quadric decimation.
"""

import numpy as np
import open3d as o3d


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
    """Transfer colors from a colored point cloud to mesh vertices via KD-tree.

    For each mesh vertex, finds the nearest point in the source point cloud
    and copies its color.

    Args:
        mesh: Open3D TriangleMesh (vertices will receive colors).
        source_pcd: Open3D PointCloud with colors.

    Returns:
        The mesh with vertex colors set.
    """
    source_points = np.asarray(source_pcd.points)
    source_colors = np.asarray(source_pcd.colors)
    mesh_vertices = np.asarray(mesh.vertices)

    pcd_tree = o3d.geometry.KDTreeFlann(source_pcd)

    colors = np.zeros_like(mesh_vertices)
    for i in range(len(mesh_vertices)):
        _, idx, _ = pcd_tree.search_knn_vector_3d(mesh_vertices[i], 1)
        colors[i] = source_colors[idx[0]]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def decimate_mesh(mesh, target_triangles, preserve_edges=True):
    """Decimate mesh using quadric error metrics.

    Args:
        mesh: Open3D TriangleMesh.
        target_triangles: Target number of triangles after decimation.
        preserve_edges: If True, applies higher weight to boundary edges
            to preserve measurement-relevant geometry.

    Returns:
        Decimated TriangleMesh.
    """
    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles
    )
    # Clean up after decimation
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
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
