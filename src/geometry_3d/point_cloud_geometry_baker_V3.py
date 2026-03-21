import numpy as np
import open3d as o3d
import json
import pickle
import random
from pathlib import Path

from scipy.spatial import ConvexHull, cKDTree
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import ndimage
from matplotlib.path import Path as MplPath
import alphashape
from shapely.geometry import Polygon, MultiPolygon

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# ==========================================
# 2. PHASE 1: GLOBAL PLANE DETECTION & CONSOLIDATION
# ==========================================

def detect_planes_global(pcd, normal_variance_deg=30.0, coplanarity_deg=80.0,
                         outlier_ratio=0.60, min_plane_edge=0.3, min_num_points=100, knn=50):
    """
    Step 1.1: Replace sequential RANSAC with Open3D's region-growing detect_planar_patches.
    """
    print("\n[Phase 1.1] Running global planar patch detection...")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=normal_variance_deg,
        coplanarity_deg=coplanarity_deg,
        outlier_ratio=outlier_ratio,
        min_plane_edge_length=min_plane_edge,
        min_num_points=min_num_points,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    )

    print(f"  Detected {len(oboxes)} raw planar patches")
    return oboxes


def extract_patch_inliers(pcd, oboxes):
    """
    Step 1.2: Extract inlier points for each OrientedBoundingBox patch.
    Re-fit plane equation via SVD for accuracy.
    """
    print("\n[Phase 1.2] Extracting patch inliers...")

    all_points = np.asarray(pcd.points)
    patches = []

    for i, obb in enumerate(oboxes):
        indices = obb.get_point_indices_within_bounding_box(pcd.points)
        if len(indices) < 4:
            continue

        pts = all_points[indices]

        # Normal from OBB rotation matrix (third column = patch normal)
        R_obb = np.asarray(obb.R)
        obb_normal = R_obb[:, 2]

        # Re-fit plane equation via SVD on inlier points
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        svd_normal = Vt[2]

        # Ensure SVD normal aligns with OBB normal direction
        if np.dot(svd_normal, obb_normal) < 0:
            svd_normal = -svd_normal

        # Plane equation: ax + by + cz + d = 0
        d = -np.dot(svd_normal, centroid)
        plane_model = np.array([svd_normal[0], svd_normal[1], svd_normal[2], d])

        patches.append({
            'normal': svd_normal,
            'center': centroid,
            'points_3d': pts,
            'plane_model': plane_model,
        })

    print(f"  Extracted {len(patches)} patches with valid inlier sets")
    return patches


def cluster_dominant_directions(patches, angle_threshold_deg=10.0, manhattan_snap_deg=15.0):
    """
    Step 1.3: Cluster plane normals into dominant directions using agglomerative clustering.
    Snap to nearest axis if within manhattan_snap_deg (Manhattan World assumption).
    """
    print("\n[Phase 1.3] Clustering dominant directions...")

    normals = []
    for p in patches:
        n = p['normal'].copy()
        # Canonicalize: for vertical-ish normals (|z| > 0.5), flip so z >= 0
        # For horizontal normals, flip so the dominant horizontal component is positive
        if abs(n[2]) > 0.5:
            if n[2] < 0:
                n = -n
        else:
            h_idx = 0 if abs(n[0]) >= abs(n[1]) else 1
            if n[h_idx] < 0:
                n = -n
        normals.append(n)

    normals = np.array(normals)
    n_patches = len(normals)

    if n_patches <= 1:
        labels = np.zeros(n_patches, dtype=int)
        dominant_dirs = [normals[0] if n_patches == 1 else np.array([0, 0, 1])]
        return labels, dominant_dirs

    # Pairwise angular distance: arccos(|n1 . n2|)
    dot_matrix = np.abs(normals @ normals.T)
    dot_matrix = np.clip(dot_matrix, 0.0, 1.0)
    angle_matrix = np.arccos(dot_matrix)
    np.fill_diagonal(angle_matrix, 0.0)

    condensed = squareform(angle_matrix, checks=False)

    Z = linkage(condensed, method='average')
    threshold_rad = np.deg2rad(angle_threshold_deg)
    labels = fcluster(Z, t=threshold_rad, criterion='distance') - 1  # 0-indexed

    # Compute dominant direction for each cluster
    unique_labels = np.unique(labels)
    dominant_dirs = []

    axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]

    for lbl in unique_labels:
        mask = labels == lbl
        cluster_normals = normals[mask]

        # Weight by point count
        weights = np.array([len(patches[i]['points_3d']) for i in range(n_patches)])[mask]
        weights = weights.astype(float)
        weights = weights / weights.sum()

        mean_normal = np.average(cluster_normals, axis=0, weights=weights)
        mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)

        # Manhattan World snapping
        snap_threshold_rad = np.deg2rad(manhattan_snap_deg)
        best_axis = None
        best_angle = float('inf')

        for ax in axes:
            angle = np.arccos(np.clip(abs(np.dot(mean_normal, ax)), 0.0, 1.0))
            if angle < best_angle:
                best_angle = angle
                best_axis = ax

        if best_angle < snap_threshold_rad:
            if np.dot(mean_normal, best_axis) < 0:
                snapped = -best_axis.copy()
            else:
                snapped = best_axis.copy()
            print(f"  Cluster {lbl}: {mask.sum()} patches, snapped "
                  f"[{mean_normal[0]:.3f}, {mean_normal[1]:.3f}, {mean_normal[2]:.3f}] -> "
                  f"[{snapped[0]:.1f}, {snapped[1]:.1f}, {snapped[2]:.1f}]")
            dominant_dirs.append(snapped.astype(float))
        else:
            print(f"  Cluster {lbl}: {mask.sum()} patches, kept "
                  f"[{mean_normal[0]:.3f}, {mean_normal[1]:.3f}, {mean_normal[2]:.3f}] "
                  f"(no axis within {manhattan_snap_deg}deg)")
            dominant_dirs.append(mean_normal)

    print(f"  Found {len(dominant_dirs)} dominant directions")
    return labels, dominant_dirs


def merge_coplanar_patches(patches, direction_labels, dominant_directions,
                           coplanar_merge_distance=0.08):
    """
    Step 1.4: Within each direction cluster, merge patches on the same physical plane.

    ============================================================
    CRITICAL TUNING PARAMETER: coplanar_merge_distance (meters)
    Controls whether two parallel patches at different offsets
    merge into one plane or stay separate.
    - Too large: cabinet face merges into wall (loses furniture)
    - Too small: wall stays fragmented (over-segmentation)
    - Default 0.08m is conservative.
    ============================================================
    """
    print(f"\n[Phase 1.4] Merging coplanar patches "
          f"(coplanar_merge_distance={coplanar_merge_distance}m)...")

    consolidated = []
    unique_labels = np.unique(direction_labels)

    for lbl in unique_labels:
        dir_normal = np.array(dominant_directions[lbl])
        mask = direction_labels == lbl
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        # Project each patch centroid onto the dominant normal to get offset
        offsets = []
        for idx in indices:
            offset = np.dot(patches[idx]['center'], dir_normal)
            offsets.append((offset, idx))

        offsets.sort(key=lambda x: x[0])

        # Merge patches with offsets within coplanar_merge_distance
        groups = []
        current_group = [offsets[0]]

        for k in range(1, len(offsets)):
            if offsets[k][0] - current_group[-1][0] < coplanar_merge_distance:
                current_group.append(offsets[k])
            else:
                groups.append(current_group)
                current_group = [offsets[k]]
        groups.append(current_group)

        for group in groups:
            group_indices = [g[1] for g in group]
            group_offsets = [g[0] for g in group]

            all_pts = np.concatenate(
                [patches[idx]['points_3d'] for idx in group_indices], axis=0)

            mean_offset = np.mean(group_offsets)
            d = -mean_offset
            plane_model = np.array([dir_normal[0], dir_normal[1], dir_normal[2], d])

            centroid = np.mean(all_pts, axis=0)

            consolidated.append({
                'normal': dir_normal.copy(),
                'center': centroid,
                'points_3d': all_pts,
                'plane_model': plane_model,
                'merged_from': group_indices,
            })

            if len(group_indices) > 1:
                print(f"  Merged {len(group_indices)} patches "
                      f"(offsets: {[f'{o:.3f}' for o in group_offsets]}) -> {len(all_pts)} pts")
            else:
                print(f"  Kept single patch (offset: {group_offsets[0]:.3f}) -> {len(all_pts)} pts")

    print(f"  Consolidated to {len(consolidated)} planes (from {len(patches)} raw patches)")
    return consolidated


def _compute_rotation_to_z(normal):
    """Compute rotation matrix to align a normal vector with the Z-axis (Rodrigues formula)."""
    normal_normalized = normal / (np.linalg.norm(normal) + 1e-8)
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = np.dot(normal_normalized, z_axis)

    if abs(dot) > 0.9999:
        if dot > 0:
            return np.eye(3)
        else:
            return np.diag([1.0, 1.0, -1.0])

    v = np.cross(normal_normalized, z_axis)
    s = np.linalg.norm(v)
    c = dot
    vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-8))
    return R


def extract_plane_boundaries(consolidated_planes, alpha=2.0, simplify_tol=0.05,
                              min_hull_area=0.5):
    """
    Step 1.5: Compute 2D alpha-shape boundaries for each consolidated plane.
    Reuses V2 logic: Rodrigues rotation to 2D, ConvexHull area filter, alpha shape.
    """
    print(f"\n[Phase 1.5] Extracting plane boundaries (min_hull_area={min_hull_area} m^2)...")

    valid_planes = []

    for i, plane in enumerate(consolidated_planes):
        pts = plane['points_3d']
        normal = plane['normal']

        R = _compute_rotation_to_z(normal)
        rotated_points = pts @ R.T
        points_2d = rotated_points[:, :2]

        try:
            hull = ConvexHull(points_2d)
            hull_area = hull.volume  # 2D: volume = area
        except Exception:
            continue

        if hull_area < min_hull_area:
            print(f"  Plane {i}: {len(pts)} pts, Area={hull_area:.2f} m^2 - REJECTED (too small)")
            continue

        try:
            alpha_shape = alphashape.alphashape(points_2d, alpha)

            if isinstance(alpha_shape, MultiPolygon):
                alpha_shape = max(alpha_shape.geoms, key=lambda p: p.area)
            elif not isinstance(alpha_shape, Polygon):
                from shapely.geometry import MultiPoint
                alpha_shape = MultiPoint(points_2d).convex_hull
                if not isinstance(alpha_shape, Polygon):
                    print(f"  Plane {i}: Could not generate valid polygon, skipping")
                    continue

            simplified_polygon = alpha_shape.simplify(simplify_tol, preserve_topology=True)

            if not isinstance(simplified_polygon, Polygon) or simplified_polygon.is_empty:
                simplified_polygon = alpha_shape

        except Exception as e:
            print(f"  Plane {i}: Alpha shape failed: {e}, skipping")
            continue

        plane['shapely_2d_polygon'] = simplified_polygon
        plane['rotation_matrix'] = R
        plane['points_2d'] = points_2d
        plane['z_level'] = np.mean(rotated_points[:, 2])

        print(f"  Plane {i}: {len(pts)} pts, Area={hull_area:.2f} m^2 - ACCEPTED")
        valid_planes.append(plane)

    print(f"  {len(valid_planes)} planes with valid boundaries")
    return valid_planes


def detect_interior_holes(planes, hole_area_min=0.3, cell_size=0.05, point_radius=0.08):
    """
    Step 1.6: Detect internal holes (doorways, windows) within consolidated wall planes.
    Rasterizes the 2D polygon, finds connected empty regions, subtracts them as holes.
    """
    print(f"\n[Phase 1.6] Detecting interior holes (min area={hole_area_min} m^2)...")

    total_holes = 0

    for i, plane in enumerate(planes):
        polygon = plane['shapely_2d_polygon']
        points_2d = plane['points_2d']

        if len(points_2d) < 10:
            continue

        minx, miny, maxx, maxy = polygon.bounds

        nx = max(1, int(np.ceil((maxx - minx) / cell_size)))
        ny = max(1, int(np.ceil((maxy - miny) / cell_size)))

        if nx * ny > 1_000_000 or nx * ny < 4:
            continue

        # Build 2D KD-tree on projected points
        tree_2d = cKDTree(points_2d)

        # Create grid of cell centers
        grid_x = np.linspace(minx + cell_size / 2, maxx - cell_size / 2, nx)
        grid_y = np.linspace(miny + cell_size / 2, maxy - cell_size / 2, ny)
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_centers = np.column_stack([gx.ravel(), gy.ravel()])

        # Point-in-polygon test using matplotlib Path (fast vectorized)
        exterior_coords = np.array(polygon.exterior.coords)
        mpl_path = MplPath(exterior_coords)
        inside_mask = mpl_path.contains_points(grid_centers).reshape(ny, nx)

        # Check which cells have nearby point support
        distances, _ = tree_2d.query(grid_centers, k=1)
        has_points = (distances < point_radius).reshape(ny, nx)

        # Empty cells = inside polygon but no point support
        empty_cells = inside_mask & ~has_points

        # Connected components of empty cells
        labeled, num_features = ndimage.label(empty_cells)

        holes_found = 0
        hole_polygons = []

        for label_id in range(1, num_features + 1):
            component_mask = labeled == label_id
            component_area = np.sum(component_mask) * cell_size * cell_size

            if component_area < hole_area_min:
                continue

            # Get coordinates of empty cells in this component
            ys, xs = np.where(component_mask)
            hole_coords = np.column_stack([grid_x[xs], grid_y[ys]])

            if len(hole_coords) < 3:
                continue

            try:
                hull = ConvexHull(hole_coords)
                hull_vertices = hole_coords[hull.vertices]
                hole_poly = Polygon(hull_vertices)

                if hole_poly.is_valid and hole_poly.area >= hole_area_min:
                    hole_polygons.append(hole_poly)
                    holes_found += 1
            except Exception:
                continue

        if hole_polygons:
            result = polygon
            for hole_poly in hole_polygons:
                result = result.difference(hole_poly)

            if isinstance(result, Polygon) and not result.is_empty:
                plane['shapely_2d_polygon'] = result
                total_holes += holes_found
                print(f"  Plane {i}: Found {holes_found} hole(s)")
            elif isinstance(result, MultiPolygon):
                largest = max(result.geoms, key=lambda p: p.area)
                plane['shapely_2d_polygon'] = largest
                total_holes += holes_found
                print(f"  Plane {i}: Found {holes_found} hole(s), kept largest fragment")

    print(f"  Total holes detected: {total_holes}")
    return planes


# ==========================================
# 3. PHASE 2: GLOBAL TOPOLOGICAL WIREFRAME ASSEMBLY
# ==========================================

def _polygon_to_3d_boundary(plane_data):
    """Project 2D polygon boundary back to 3D world coordinates."""
    polygon = plane_data['shapely_2d_polygon']
    R = plane_data['rotation_matrix']
    centroid = plane_data['center']

    coords_2d = np.array(polygon.exterior.coords)
    centroid_rotated = centroid @ R.T
    z_level = centroid_rotated[2]
    coords_3d_rotated = np.column_stack([coords_2d, np.full(len(coords_2d), z_level)])
    coords_3d_world = coords_3d_rotated @ R
    return coords_3d_world


def compute_intersection_lines(planes, parallel_threshold_deg=10.0):
    """
    Step 2.1: Compute theoretical intersection lines for all non-parallel plane pairs.
    """
    print(f"\n[Phase 2.1] Computing intersection lines...")

    cos_threshold = np.cos(np.deg2rad(parallel_threshold_deg))
    intersection_lines = []

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            n1 = planes[i]['plane_model'][:3]
            n2 = planes[j]['plane_model'][:3]

            dot_product = abs(np.dot(n1, n2))
            if dot_product > cos_threshold:
                continue

            line_dir = np.cross(n1, n2)
            line_dir_norm = np.linalg.norm(line_dir)
            if line_dir_norm < 1e-8:
                continue
            line_dir = line_dir / line_dir_norm

            d1, d2 = planes[i]['plane_model'][3], planes[j]['plane_model'][3]
            fix_axis = np.argmax(np.abs(line_dir))
            axes = [k for k in range(3) if k != fix_axis]

            A = np.array([[n1[axes[0]], n1[axes[1]]],
                          [n2[axes[0]], n2[axes[1]]]])
            b_vec = np.array([-d1, -d2])

            if abs(np.linalg.det(A)) < 1e-8:
                continue

            solution = np.linalg.solve(A, b_vec)
            point_on_line = np.zeros(3)
            point_on_line[axes[0]] = solution[0]
            point_on_line[axes[1]] = solution[1]

            intersection_lines.append({
                'line_point': point_on_line,
                'line_dir': line_dir,
                'plane_i': i,
                'plane_j': j,
            })

    print(f"  Found {len(intersection_lines)} intersection lines from {len(planes)} planes")
    return intersection_lines


def clip_lines_to_boundaries(intersection_lines, planes, extension=0.05,
                              min_segment_length=0.1):
    """
    Step 2.2: Clip intersection lines to plane boundary extents with extension for gap closing.
    """
    print(f"\n[Phase 2.2] Clipping to boundaries (extension={extension}m)...")

    wireframe_segments = []

    for line_data in intersection_lines:
        pi = line_data['plane_i']
        pj = line_data['plane_j']
        line_point = line_data['line_point']
        line_dir = line_data['line_dir']

        boundary_i = _polygon_to_3d_boundary(planes[pi])
        boundary_j = _polygon_to_3d_boundary(planes[pj])

        # Project boundaries onto intersection line
        t_i = np.dot(boundary_i - line_point, line_dir)
        t_j = np.dot(boundary_j - line_point, line_dir)

        t_i_min, t_i_max = np.min(t_i) - extension, np.max(t_i) + extension
        t_j_min, t_j_max = np.min(t_j) - extension, np.max(t_j) + extension

        # Overlap interval
        t_overlap_min = max(t_i_min, t_j_min)
        t_overlap_max = min(t_i_max, t_j_max)

        if t_overlap_max - t_overlap_min < min_segment_length:
            continue

        start_point = line_point + t_overlap_min * line_dir
        end_point = line_point + t_overlap_max * line_dir

        wireframe_segments.append({
            'start': start_point,
            'end': end_point,
            'plane_i': pi,
            'plane_j': pj,
        })

    print(f"  Created {len(wireframe_segments)} wireframe segments")
    return wireframe_segments


class _UnionFind:
    """Union-Find data structure for vertex merging."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def merge_vertices(wireframe_segments, merge_distance=0.10):
    """
    Step 2.3: Merge nearby wireframe segment endpoints into shared vertices.
    Uses KD-tree + Union-Find for efficient grouping.
    """
    print(f"\n[Phase 2.3] Merging vertices (merge_distance={merge_distance}m)...")

    if not wireframe_segments:
        return wireframe_segments

    # Collect all endpoints
    endpoints = []
    for seg in wireframe_segments:
        endpoints.append(np.array(seg['start']))
        endpoints.append(np.array(seg['end']))
    endpoints = np.array(endpoints)

    n = len(endpoints)

    # KD-tree for nearby pairs
    tree = cKDTree(endpoints)
    pairs = tree.query_pairs(merge_distance)

    # Union-Find grouping
    uf = _UnionFind(n)
    for a, b in pairs:
        uf.union(a, b)

    # Compute centroid for each group
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    merged_positions = np.copy(endpoints)
    merge_count = 0
    for root, members in groups.items():
        if len(members) > 1:
            centroid = np.mean(endpoints[members], axis=0)
            for m in members:
                merged_positions[m] = centroid
            merge_count += 1

    # Update segments with merged positions
    for k, seg in enumerate(wireframe_segments):
        seg['start'] = merged_positions[2 * k]
        seg['end'] = merged_positions[2 * k + 1]

    print(f"  Merged {merge_count} vertex groups from {n} endpoints")
    return wireframe_segments


def select_edges_blp(wireframe_segments, planes, points, lambda_complexity=0.1):
    """
    Step 2.4 (MANDATORY): Binary LP to select optimal edge subset with topological guarantees.

    Objective: maximize coverage - lambda_complexity * num_edges
    Constraints:
    - No dangling edges (vertex degree 0 or >= 2)
    - At most one edge per plane pair
    """
    print(f"\n[Phase 2.4] Running BLP edge selection (lambda={lambda_complexity})...")

    if not wireframe_segments:
        return wireframe_segments

    n_edges = len(wireframe_segments)

    # Build unique vertex set (after merging, endpoints may be identical)
    all_endpoints = []
    for seg in wireframe_segments:
        all_endpoints.append(tuple(np.round(seg['start'], decimals=6)))
        all_endpoints.append(tuple(np.round(seg['end'], decimals=6)))

    unique_verts = list(set(all_endpoints))
    vert_to_idx = {v: i for i, v in enumerate(unique_verts)}
    n_verts = len(unique_verts)

    # Edge-vertex incidence
    edge_verts = []
    for seg in wireframe_segments:
        v_s = vert_to_idx[tuple(np.round(seg['start'], decimals=6))]
        v_e = vert_to_idx[tuple(np.round(seg['end'], decimals=6))]
        edge_verts.append((v_s, v_e))

    # Coverage metric: segment length
    coverage = np.array([
        np.linalg.norm(np.array(seg['end']) - np.array(seg['start']))
        for seg in wireframe_segments
    ])

    # Variables: x_0..x_{n_edges-1} (edge selection), z_0..z_{n_verts-1} (vertex active)
    n_vars = n_edges + n_verts

    # Objective: minimize -coverage_i * x_i + lambda * x_i
    c = np.zeros(n_vars)
    c[:n_edges] = -coverage + lambda_complexity

    # Build constraint rows: all as A @ x <= b
    A_rows = []
    b_rows = []

    # Vertex-edge incidence map
    vert_edges = {v: [] for v in range(n_verts)}
    for edge_idx, (vs, ve) in enumerate(edge_verts):
        vert_edges[vs].append(edge_idx)
        vert_edges[ve].append(edge_idx)

    # Constraint 1: x_i - z_v <= 0 for each edge-vertex pair
    # (if edge is selected, its vertices must be active)
    for edge_idx, (vs, ve) in enumerate(edge_verts):
        row = np.zeros(n_vars)
        row[edge_idx] = 1.0
        row[n_edges + vs] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)

        row = np.zeros(n_vars)
        row[edge_idx] = 1.0
        row[n_edges + ve] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)

    # Constraint 2: 2*z_v - sum(x_i for edges at v) <= 0
    # (if vertex is active, at least 2 edges must be incident)
    # Only enforce at vertices with >= 2 candidate edges (potential corners).
    # Leaf vertices (1 candidate edge) are allowed degree 0 or 1.
    for v in range(n_verts):
        if len(vert_edges[v]) < 2:
            continue
        row = np.zeros(n_vars)
        row[n_edges + v] = 2.0
        for edge_idx in vert_edges[v]:
            row[edge_idx] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)

    # Constraint 3: Plane-pair uniqueness (at most one edge per plane pair)
    plane_pair_edges = {}
    for edge_idx, seg in enumerate(wireframe_segments):
        pair = (min(seg['plane_i'], seg['plane_j']), max(seg['plane_i'], seg['plane_j']))
        if pair not in plane_pair_edges:
            plane_pair_edges[pair] = []
        plane_pair_edges[pair].append(edge_idx)

    for pair, edge_indices in plane_pair_edges.items():
        if len(edge_indices) > 1:
            row = np.zeros(n_vars)
            for edge_idx in edge_indices:
                row[edge_idx] = 1.0
            A_rows.append(row)
            b_rows.append(1.0)

    A_ub = np.array(A_rows) if A_rows else np.zeros((1, n_vars))
    b_ub = np.array(b_rows) if b_rows else np.zeros(1)

    # All variables are binary
    integrality = np.ones(n_vars)
    bounds = Bounds(lb=0, ub=1)

    constraints = LinearConstraint(A_ub, ub=b_ub)

    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
        )

        if result.success:
            x_solution = np.round(result.x[:n_edges]).astype(int)
            selected = [wireframe_segments[i] for i in range(n_edges) if x_solution[i] == 1]
            print(f"  BLP selected {len(selected)}/{n_edges} edges "
                  f"(objective: {result.fun:.3f})")
            return selected
        else:
            print(f"  BLP solver failed ({result.message}), keeping all edges")
            return wireframe_segments

    except Exception as e:
        print(f"  BLP solver error: {e}, keeping all edges")
        return wireframe_segments


# ==========================================
# 4. ORCHESTRATOR
# ==========================================

def extract_3d_wireframe_v3(points, output_dir,
                            # Phase 1 params
                            normal_variance_deg=30.0, coplanarity_deg=80.0,
                            outlier_ratio=0.60, min_plane_edge=0.3,
                            min_num_points=100, knn=50,
                            angle_threshold_deg=10.0, manhattan_snap_deg=15.0,
                            coplanar_merge_distance=0.08,
                            alpha=2.0, simplify_tol=0.05, min_hull_area=0.5,
                            hole_area_min=0.3, hole_cell_size=0.05,
                            hole_point_radius=0.08,
                            # Phase 2 params
                            parallel_threshold_deg=10.0, boundary_extension=0.05,
                            min_segment_length=0.1, vertex_merge_distance=0.10,
                            lambda_complexity=0.1):
    """
    Drop-in replacement for extract_3d_wireframe().
    Same output signature: (wireframe_segments, detected_planes)
    Same pickle-compatible data format.
    """
    print(f"\n{'=' * 60}")
    print(f"[V3 Wireframe Extraction] Global Detection + Topological Assembly")
    print(f"{'=' * 60}")

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # --- Phase 1: Global Plane Detection & Consolidation ---
    oboxes = detect_planes_global(
        pcd, normal_variance_deg, coplanarity_deg,
        outlier_ratio, min_plane_edge, min_num_points, knn)

    patches = extract_patch_inliers(pcd, oboxes)

    direction_labels, dominant_directions = cluster_dominant_directions(
        patches, angle_threshold_deg, manhattan_snap_deg)

    consolidated = merge_coplanar_patches(
        patches, direction_labels, dominant_directions, coplanar_merge_distance)

    detected_planes = extract_plane_boundaries(
        consolidated, alpha, simplify_tol, min_hull_area)

    detected_planes = detect_interior_holes(
        detected_planes, hole_area_min, hole_cell_size, hole_point_radius)

    # Debug visualization
    print("\n  [Debug] Saving plane visualization...")
    debug_colored_pcds = []

    # Collect all indices that belong to a detected plane
    assigned_indices = set()
    for plane in detected_planes:
        # Round-trip through the KD-tree to find which rows of `points` each plane owns
        plane_pts = plane['points_3d']
        tree = cKDTree(points)
        dists, idxs = tree.query(plane_pts, k=1)
        for d, idx in zip(dists, idxs):
            if d < 1e-6:
                assigned_indices.add(idx)

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(plane_pts)
        cluster_pcd.paint_uniform_color([random.random(), random.random(), random.random()])
        debug_colored_pcds.append(cluster_pcd)

    # Add unassigned points in gray so the full room is always visible
    unassigned_mask = np.ones(len(points), dtype=bool)
    for idx in assigned_indices:
        unassigned_mask[idx] = False
    unassigned_pts = points[unassigned_mask]

    if len(unassigned_pts) > 0:
        unassigned_pcd = o3d.geometry.PointCloud()
        unassigned_pcd.points = o3d.utility.Vector3dVector(unassigned_pts)
        unassigned_pcd.paint_uniform_color([0.6, 0.6, 0.6])
        debug_colored_pcds.append(unassigned_pcd)
        print(f"  [Debug] {len(unassigned_pts)} unassigned points added in gray")

    if debug_colored_pcds:
        combined_pcd = o3d.geometry.PointCloud()
        for p in debug_colored_pcds:
            combined_pcd += p
        debug_ply_path = output_dir / "debug_ransac_planes.ply"
        o3d.io.write_point_cloud(str(debug_ply_path), combined_pcd)
        print(f"  [Debug] Saved to: {debug_ply_path}")

    # --- Phase 2: Global Topological Wireframe Assembly ---
    intersection_lines = compute_intersection_lines(
        detected_planes, parallel_threshold_deg)

    wireframe_segments = clip_lines_to_boundaries(
        intersection_lines, detected_planes, boundary_extension, min_segment_length)

    wireframe_segments = merge_vertices(wireframe_segments, vertex_merge_distance)

    wireframe_segments = select_edges_blp(
        wireframe_segments, detected_planes, points, lambda_complexity)

    print(f"\n{'=' * 60}")
    print(f"  Final: {len(detected_planes)} planes, {len(wireframe_segments)} wireframe segments")
    print(f"{'=' * 60}")

    return wireframe_segments, detected_planes


# ==========================================
# 5. MAIN BAKING SCRIPT
# ==========================================

def main():
    print("--- 3D Point Cloud Geometry Baker V3 ---")

    point_cloud_name = "tmb_office1_subsampled"

    pcd_path = project_root / "data" / "raw_point_cloud" / f"{point_cloud_name}.ply"
    meta_path = project_root / "data" / "density_image" / point_cloud_name / "metadata.json"

    # Setup output directory
    output_base_dir = project_root / "data" / "debug_renderer" / point_cloud_name
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Metadata & Alignment...")
    with open(meta_path) as f:
        meta = json.load(f)

    rotation_matrix = np.array(meta['rotation_matrix'])

    print(f"Loading Point Cloud: {pcd_path.name}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))

    # Apply global alignment so the baked coordinates match the room rotation
    print("Applying global alignment rotation...")
    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    points = np.asarray(pcd.points)

    # ============================================================
    # CRITICAL TUNING PARAMETER: coplanar_merge_distance
    # Controls wall-vs-cabinet separation. See Step 1.4.
    # Default 0.08m is conservative. Increase if walls still split.
    # ============================================================
    wireframe_segments, detected_planes = extract_3d_wireframe_v3(
        points,
        output_dir=output_base_dir,
        coplanar_merge_distance=0.08,
        min_hull_area=0.90,
    )

    # Package data for pickling
    bake_data = {
        'wireframe_segments': wireframe_segments,
        'detected_planes': detected_planes
    }

    # Save to disk
    output_pkl = output_base_dir / "room_geometry.pkl"
    print(f"\nBaking extracted geometry to: {output_pkl}")

    with open(output_pkl, 'wb') as f:
        pickle.dump(bake_data, f)

    print("[Success] Geometry successfully baked and ready for the lightweight renderer!")


if __name__ == "__main__":
    main()
