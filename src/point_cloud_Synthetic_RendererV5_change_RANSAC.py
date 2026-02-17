import numpy as np
import open3d as o3d
import cv2
import json
import sys
import random
from pathlib import Path

from scipy.spatial import ConvexHull
import alphashape
from shapely.geometry import Polygon, MultiPolygon

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# Standard "Virtual Camera" Intrinsics (Matches your pano slicer)
IMG_W, IMG_H = 1024, 1024
FOV = 60  # Degrees
FOCAL_LENGTH = 0.5 * IMG_W / np.tan(0.5 * np.radians(FOV))
CX, CY = (IMG_W - 1) / 2.0, (IMG_H - 1) / 2.0

# Views to render: (yaw_offset, pitch_offset) in degrees
# 8 standard horizontal views at 45-degree intervals
VIEWS_TO_RENDER = [
    (0, 0),    # Front
    (45, 0),   # Front-Right
    (90, 0),   # Right
    (135, 0),  # Back-Right
    (180, 0),  # Back
    (225, 0),  # Back-Left
    (270, 0),  # Left
    (315, 0),  # Front-Left
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def find_floor_z(points, bin_size=0.05):
    """Finds the floor Z-level using histogram analysis."""
    z_values = points[:, 2]
    z_min, z_max = np.min(z_values), np.max(z_values)
    bins = np.arange(z_min, z_max, bin_size)
    hist, edges = np.histogram(z_values, bins=bins)
    
    # Floor is likely in the bottom 30%
    lower_threshold_idx = int(len(hist) * 0.30)
    if lower_threshold_idx == 0: lower_threshold_idx = len(hist)
    
    peak_idx = np.argmax(hist[:lower_threshold_idx])
    floor_z = edges[peak_idx] + (bin_size / 2.0)
    return floor_z

def filter_short_edges(binary_image, min_size):
    """
    Removes small connected components from a binary edge image.
    
    Args:
        binary_image: Input binary image (white edges on black background)
        min_size: Minimum area (in pixels) for a component to be retained
    
    Returns:
        Filtered binary image with small components removed
    """
    # Find all connected components with statistics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    
    # Create output image (start with black)
    output = np.zeros_like(binary_image)
    
    # Iterate through components (skip label 0 which is background)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        
        # Keep only components with area >= min_size
        if area >= min_size:
            output[labels == label_id] = 255
    
    return output


def filter_jagged_edges(binary_image, min_length=60, epsilon=3.0, max_density=0.10, max_wobble=0.15):
    """
    Enhanced filter that separates structural lines from 'squiggle' noise 
    using angular analysis.
    """
    output = np.zeros_like(binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, closed=False)
        if perimeter < min_length:
            continue

        # --- METRIC 1: Raw Vertex Density (The Roughness Check) ---
        # We use a very strict approximation just to count the "real" directional changes
        # epsilon=1.0 ensures we capture the zigs and zags of the noise
        raw_approx = cv2.approxPolyDP(contour, epsilon=1.0, closed=False)
        raw_density = len(raw_approx) / perimeter
        
        # --- METRIC 2: Directional "Wobble" (The Squiggle Check) ---
        # Calculate the sum of absolute angular changes along the path
        # Smooth curves have low accumulated angle per pixel.
        # Squiggly noise has high accumulated angle per pixel.
        total_angle_change = 0
        if len(raw_approx) > 2:
            pts = raw_approx[:, 0, :]
            # Vectors between consecutive points
            vecs = pts[1:] - pts[:-1]
            # Normalize vectors
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            # Avoid division by zero
            valid_vecs = vecs / (norms + 1e-6)
            
            # Dot product of consecutive vectors (v1 . v2)
            # Clip to [-1, 1] to avoid float errors in arccos
            dots = np.sum(valid_vecs[:-1] * valid_vecs[1:], axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            
            # Angles in radians
            angles = np.arccos(dots)
            total_angle_change = np.sum(angles)

        # "Wobble" = average turning angle per unit length
        # Normal straight lines/smooth curves are < 0.1
        # Chaotic squiggles are often > 0.2
        wobble_metric = total_angle_change / perimeter

        # --- DECISION LOGIC ---
        is_structure = True
        
        # Filter 1: Is it too rough? (Too many vertices for its length)
        if raw_density > max_density:
            is_structure = False
            
        # Filter 2: Is it too squiggly? (Direction changes too chaotically)
        if wobble_metric > max_wobble:
            is_structure = False

        if is_structure:
            # Draw the simplified version (clean look)
            final_approx = cv2.approxPolyDP(contour, epsilon, closed=False)
            cv2.drawContours(output, [final_approx], -1, 255, thickness=1)
            
    return output


def extract_3d_wireframe(points, output_dir, distance_threshold=0.05, ransac_n=3, num_iterations=1000, 
                         min_remaining_points=2000, dbscan_eps=0.20, dbscan_min_samples=30, 
                         min_hull_area=0.05, parallel_threshold=0.9, alpha_shape_alpha=2.0):
    """
    Extracts a 3D wireframe using Hybrid RANSAC + DBSCAN approach.
    
    This method detects planes based on physical connectivity and surface area rather
    than just mathematical consensus. It splits mathematically coplanar but spatially
    disconnected objects (like a coffee table and cupboard at the same height) into
    separate plane clusters. Each valid cluster is represented by an Alpha Shape
    (Concave Hull) polygon that precisely defines its 2D boundary.
    
    Args:
        points: (N, 3) numpy array of point cloud
        output_dir: Path object where debug PLY will be saved
        distance_threshold: RANSAC distance threshold for plane fitting
        ransac_n: Number of points to sample for RANSAC
        num_iterations: Number of RANSAC iterations
        min_remaining_points: Stop processing when fewer points remain (default: 2000)
        dbscan_eps: DBSCAN epsilon - max distance between points in same cluster (meters)
        dbscan_min_samples: DBSCAN minimum samples per cluster
        min_hull_area: Minimum convex hull area (sq meters) to accept a cluster
        parallel_threshold: Dot product threshold to consider planes parallel
        alpha_shape_alpha: Alpha parameter for concave hull generation (higher = tighter fit)
    
    Returns:
        tuple: (wireframe_segments, detected_planes)
            wireframe_segments: List of dicts, each with keys:
                'start': (3,) ndarray - segment start point
                'end':   (3,) ndarray - segment end point
                'plane_i': int - index of first parent plane in detected_planes
                'plane_j': int - index of second parent plane in detected_planes
            detected_planes: List of plane data dicts (used for visibility pre-filtering)
    """
    print(f"\n[Wireframe Extraction] Starting Hybrid RANSAC + DBSCAN plane segmentation...")
    
    # Create Open3D point cloud for segmentation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Store detected planes: list of (plane_eq, inlier_points)
    detected_planes = []
    remaining_pcd = pcd
    
    # --- VISUALIZATION SETUP ---
    debug_colored_pcds = []
    # ---------------------------
    
    iteration = 0
    total_clusters_accepted = 0
    
    # Dynamic while loop - continue processing until points are exhausted
    while len(remaining_pcd.points) > min_remaining_points:
        iteration += 1
        print(f"\n  [Iteration {iteration}] Remaining points: {len(remaining_pcd.points)}")
        
        # RANSAC plane segmentation - find the dominant mathematical plane
        plane_model, inlier_indices = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        # If RANSAC finds too few inliers, we've likely exhausted meaningful planes
        if len(inlier_indices) < dbscan_min_samples:
            print(f"  [Info] RANSAC found only {len(inlier_indices)} inliers, stopping")
            break
        
        # Extract inlier points for DBSCAN clustering
        inlier_pcd = remaining_pcd.select_by_index(inlier_indices)
        inlier_points = np.asarray(inlier_pcd.points)
        
        normal = np.array(plane_model[:3])
        print(f"  RANSAC Plane: {len(inlier_indices)} inliers, Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
        
        # --- DBSCAN CLUSTERING (Open3D C++ backend) ---
        # Split the mathematical plane into physically distinct clusters
        labels = np.array(inlier_pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_samples, print_progress=True))
        cluster_labels = labels
        
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label
        
        print(f"  DBSCAN found {len(unique_labels)} potential clusters (excluding noise)")
        
        # --- VALIDATE EACH CLUSTER BY SURFACE AREA ---
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_points = inlier_points[cluster_mask]
            
            if len(cluster_points) < 4:  # Need at least 4 points for ConvexHull in 2D
                continue
            
            # Calculate physical surface area using Convex Hull projection
            # Step 1: Rotate cluster so plane normal aligns with Z-axis
            normal_normalized = normal / (np.linalg.norm(normal) + 1e-8)
            
            # Find rotation to align normal with Z-axis [0, 0, 1]
            z_axis = np.array([0, 0, 1])
            
            # Handle case where normal is already aligned with Z (or opposite)
            dot = np.dot(normal_normalized, z_axis)
            if abs(dot) > 0.9999:
                # Normal is already (anti-)parallel to Z, use identity or flip
                if dot > 0:
                    R = np.eye(3)
                else:
                    R = np.diag([1, 1, -1])
            else:
                # Rodrigues' rotation formula
                v = np.cross(normal_normalized, z_axis)
                s = np.linalg.norm(v)
                c = dot
                vx = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-8))
            
            # Step 2: Apply rotation and project to 2D (drop Z)
            rotated_points = cluster_points @ R.T
            points_2d = rotated_points[:, :2]  # Drop Z coordinate
            
            # Step 3: Calculate Convex Hull area
            try:
                hull = ConvexHull(points_2d)
                hull_area = hull.volume  # In 2D, 'volume' is actually area
            except Exception:
                # ConvexHull can fail for degenerate cases
                continue
            
            # Filter out noise - only accept clusters with sufficient area
            if hull_area < min_hull_area:
                print(f"    Cluster {label}: {len(cluster_points)} pts, Area={hull_area:.4f} m² - REJECTED (too small)")
                continue
            
            print(f"    Cluster {label}: {len(cluster_points)} pts, Area={hull_area:.4f} m² - ACCEPTED")
            
            # --- ALPHA SHAPE (CONCAVE HULL) GENERATION ---
            # Generate tight boundary polygon for this cluster using alpha shapes
            # This handles concave geometries (L-shaped rooms, etc.) that convex hulls would close off
            try:
                alpha_shape = alphashape.alphashape(points_2d, alpha_shape_alpha)
                
                # Ensure result is a Polygon (handle MultiPolygon by taking largest component)
                if isinstance(alpha_shape, MultiPolygon):
                    # Take the polygon with the largest area
                    alpha_shape = max(alpha_shape.geoms, key=lambda p: p.area)
                elif not isinstance(alpha_shape, Polygon):
                    # Fallback: if alphashape returns something else (e.g., LineString, Point),
                    # fall back to convex hull
                    from shapely.geometry import MultiPoint
                    alpha_shape = MultiPoint(points_2d).convex_hull
                    if not isinstance(alpha_shape, Polygon):
                        print(f"      [Warning] Could not generate valid polygon for cluster {label}, skipping")
                        continue
                
                # Simplify to smooth out jagged artifacts from pixel noise
                simplified_polygon = alpha_shape.simplify(0.05, preserve_topology=True)
                
                # Ensure simplification didn't degenerate the polygon
                if not isinstance(simplified_polygon, Polygon) or simplified_polygon.is_empty:
                    simplified_polygon = alpha_shape  # Fall back to unsimplified
                    
            except Exception as e:
                print(f"      [Warning] Alpha shape generation failed for cluster {label}: {e}")
                continue
            # -----------------------------------------------
            
            # --- VISUALIZATION: Generate unique random color for each accepted cluster ---
            cluster_color = [random.random(), random.random(), random.random()]
            
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
            cluster_pcd.paint_uniform_color(cluster_color)
            debug_colored_pcds.append(cluster_pcd)
            # -----------------------------------------------------------------------------
            
            # Store rich geometric data for this plane cluster
            # Recompute plane equation (d value) for this specific cluster
            cluster_centroid = np.mean(cluster_points, axis=0)
            d = -np.dot(normal, cluster_centroid)
            cluster_plane_eq = np.array([normal[0], normal[1], normal[2], d])
            
            # Store as dictionary with all transformation data needed for future intersection mapping
            plane_data = {
                'plane_model': cluster_plane_eq,           # [a, b, c, d] plane equation
                'points_3d': cluster_points,                # Original 3D points
                'shapely_2d_polygon': simplified_polygon,   # Alpha shape boundary in 2D local coords
                'rotation_matrix': R,                       # Rotation to flatten plane to XY
                'centroid': cluster_centroid,               # 3D centroid of the cluster
                'normal': normal_normalized,                # Unit normal vector
            }
            
            detected_planes.append(plane_data)
            total_clusters_accepted += 1
        
        # Remove ALL original RANSAC inliers from remaining cloud
        # This ensures the loop progresses to the next surface
        remaining_pcd = remaining_pcd.select_by_index(inlier_indices, invert=True)
    
    print(f"\n  Detected {total_clusters_accepted} physically distinct planar clusters")

    # --- SAVE DEBUG POINT CLOUD ---
    print("  [Debug] Saving RANSAC+DBSCAN visualization...")
    # Paint remaining noise points Grey
    remaining_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    debug_colored_pcds.append(remaining_pcd)
    
    # Combine all clouds
    combined_pcd = o3d.geometry.PointCloud()
    for p in debug_colored_pcds:
        combined_pcd += p
        
    debug_ply_path = output_dir / "debug_ransac_planes.ply"
    o3d.io.write_point_cloud(str(debug_ply_path), combined_pcd)
    print(f"  [Debug] Saved to: {debug_ply_path}")
    # ------------------------------
    
    # --- PHASE 3: TOPOLOGY-AWARE PROJECTION-CLIPPING INTERSECTION ---
    # Only create wireframes where physical surfaces actually touch
    wireframe_segments = []
    adjacency_threshold = 0.15  # 15 cm maximum distance for adjacency
    min_segment_length = 0.1   # Minimum segment length to accept
    
    print(f"\n[Wireframe Extraction] Computing topology-aware plane intersections...")
    print(f"  Adjacency threshold: {adjacency_threshold} m")
    
    # Helper function to reconstruct 3D boundary from 2D polygon
    def polygon_to_3d_boundary(plane_data):
        """
        Transform 2D polygon vertices back to 3D space using stored transformation data.
        
        The 2D polygon was created by:
        1. Rotating cluster_points by R.T (so normal aligns with Z)
        2. Taking only X, Y coordinates (dropping Z)
        
        To reverse this:
        1. Add back the Z coordinate (use Z from the rotated centroid)
        2. Apply inverse rotation R (since R @ R.T = I)
        """
        polygon = plane_data['shapely_2d_polygon']
        R = plane_data['rotation_matrix']
        centroid = plane_data['centroid']
        normal = plane_data['normal']
        
        # Get 2D boundary coordinates from polygon exterior
        coords_2d = np.array(polygon.exterior.coords)
        
        # The original points were transformed as: rotated = original @ R.T
        # Then 2D was taken as: coords_2d = rotated[:, :2]
        # The Z value in rotated space is constant (the plane is flat after rotation)
        
        # Find the Z level in rotated space by rotating the centroid
        centroid_rotated = centroid @ R.T
        z_level = centroid_rotated[2]
        
        # Reconstruct 3D points in rotated coordinate system
        coords_3d_rotated = np.column_stack([coords_2d, np.full(len(coords_2d), z_level)])
        
        # Apply inverse rotation: original = rotated @ R (since R is orthogonal, R.T.T = R)
        coords_3d_world = coords_3d_rotated @ R
        
        return coords_3d_world
    
    # Helper function to compute minimum distance between two point sets
    def min_boundary_distance(boundary1, boundary2):
        """
        Compute minimum Euclidean distance between two sets of boundary points.
        Uses vectorized operations for efficiency.
        """
        # Compute pairwise distances using broadcasting
        # boundary1: (N, 3), boundary2: (M, 3)
        # diff: (N, M, 3)
        diff = boundary1[:, np.newaxis, :] - boundary2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return np.min(distances)
    
    # Helper function to project points onto a line and get t-parameter values
    def project_to_line(points, line_point, line_direction):
        """Project 3D points onto a line, returning t-parameter values."""
        vecs = points - line_point
        t_values = np.dot(vecs, line_direction)
        return t_values
    
    pairs_checked = 0
    pairs_adjacent = 0
    
    for i in range(len(detected_planes)):
        for j in range(i + 1, len(detected_planes)):
            pairs_checked += 1
            
            plane1_data = detected_planes[i]
            plane2_data = detected_planes[j]
            
            # Get plane equations and normals
            plane1_eq = plane1_data['plane_model']
            plane2_eq = plane2_data['plane_model']
            n1 = plane1_eq[:3]
            n2 = plane2_eq[:3]
            
            # Check if planes are parallel (skip if so)
            dot_product = abs(np.dot(n1, n2))
            if dot_product > parallel_threshold:
                continue
            
            # --- ADJACENCY CHECK ---
            # Reconstruct 3D boundaries from stored 2D polygons
            boundary1_3d = polygon_to_3d_boundary(plane1_data)
            boundary2_3d = polygon_to_3d_boundary(plane2_data)
            
            # Calculate minimum distance between boundaries
            min_dist = min_boundary_distance(boundary1_3d, boundary2_3d)
            
            # Skip if planes are not physically adjacent
            if min_dist > adjacency_threshold:
                continue
            
            pairs_adjacent += 1
            
            # --- INTERSECTION LINE CALCULATION ---
            # Compute intersection line direction using cross product of normals
            line_dir = np.cross(n1, n2)
            line_dir_norm = np.linalg.norm(line_dir)
            if line_dir_norm < 1e-8:
                continue
            line_dir = line_dir / line_dir_norm
            
            # Find a point on the intersection line by solving the plane equations
            d1, d2 = plane1_eq[3], plane2_eq[3]
            
            # Choose the axis with smallest component in line_dir to fix at 0
            fix_axis = np.argmin(np.abs(line_dir))
            axes = [k for k in range(3) if k != fix_axis]
            
            A = np.array([[n1[axes[0]], n1[axes[1]]],
                          [n2[axes[0]], n2[axes[1]]]])
            b = np.array([-d1, -d2])
            
            if abs(np.linalg.det(A)) < 1e-8:
                continue
            
            solution = np.linalg.solve(A, b)
            
            point_on_line = np.zeros(3)
            point_on_line[axes[0]] = solution[0]
            point_on_line[axes[1]] = solution[1]
            
            # --- PROJECTION-CLIPPING ---
            # Project 3D boundary vertices onto the intersection line to find valid intervals
            t1_values = project_to_line(boundary1_3d, point_on_line, line_dir)
            t2_values = project_to_line(boundary2_3d, point_on_line, line_dir)
            
            # Get intervals for each plane's boundary
            t1_min, t1_max = np.min(t1_values), np.max(t1_values)
            t2_min, t2_max = np.min(t2_values), np.max(t2_values)
            
            # Find overlapping interval
            t_overlap_min = max(t1_min, t2_min)
            t_overlap_max = min(t1_max, t2_max)
            
            # Skip if no overlap or negligible overlap
            if t_overlap_max <= t_overlap_min:
                continue
            
            segment_length = t_overlap_max - t_overlap_min
            if segment_length < min_segment_length:
                continue
            
            # Create the wireframe segment from the overlapping portion
            start_point = point_on_line + t_overlap_min * line_dir
            end_point = point_on_line + t_overlap_max * line_dir
            
            wireframe_segments.append({
                'start': start_point,
                'end': end_point,
                'plane_i': i,
                'plane_j': j,
            })
    
    print(f"  Pairs checked: {pairs_checked}, Adjacent pairs: {pairs_adjacent}")
    print(f"  Found {len(wireframe_segments)} wireframe segments")
    return wireframe_segments, detected_planes

def render_synthetic_wireframe(points, camera_pose_global, rotation_deg, view_yaw=0, view_pitch=0, wireframe_segments=None, detected_planes=None):
    """
    Renders a synthetic depth-based wireframe from the point cloud.
    
    Args:
        points: (N, 3) numpy array of point cloud
        camera_pose_global: [x, y, z] in meters
        rotation_deg: Global alignment rotation (Room orientation)
        view_yaw: Relative yaw of the virtual camera (0 = Front, 90 = Right, etc.)
        view_pitch: Relative pitch of the virtual camera (positive = look up, negative = look down)
        wireframe_segments: List of segment dicts for Channel B (from extract_3d_wireframe)
        detected_planes: List of plane data dicts (from extract_3d_wireframe)
    """
    print(f"  > Rendering View (Yaw={view_yaw} deg, Pitch={view_pitch} deg)...")
    
    # 1. Translate Points to Camera Center
    # Vector from Camera -> Point
    vecs = points - camera_pose_global
    
    # 2. Rotate Points to Align with Camera Orientation
    # Rotation order: Global Yaw -> View Yaw -> View Pitch
    # We rotate the world by the negative of these angles to bring it into camera frame
    
    # Step 2a: Apply Yaw rotation (around Z-axis)
    total_yaw_rad = np.radians(-(rotation_deg + view_yaw))
    c_yaw, s_yaw = np.cos(total_yaw_rad), np.sin(total_yaw_rad)
    
    # Rotation Matrix (around Z-axis for Yaw)
    R_z = np.array([
        [c_yaw, -s_yaw, 0],
        [s_yaw,  c_yaw, 0],
        [0,      0,     1]
    ])
    
    local_vecs = vecs @ R_z.T
    
    # 3. Swizzle Coordinates to Standard Pinhole Frame
    # LGT/RoomFormer World: Z is Up, Y is Forward (usually), X is Right
    # Camera Pinhole:       X is Right, Y is Down, Z is Forward
    
    # Map:
    # World X -> Cam X
    # World Y -> Cam Z (Depth)
    # World Z -> Cam -Y (Up becomes -Down)
    
    points_cam = np.zeros_like(local_vecs)
    points_cam[:, 0] = local_vecs[:, 0]  # X -> X
    points_cam[:, 1] = -local_vecs[:, 2] # Z -> -Y
    points_cam[:, 2] = local_vecs[:, 1]  # Y -> Z (Forward)
    
    # Step 2b: Apply Pitch rotation (around Camera's X-axis) AFTER swizzle
    # Pitch rotates around X-axis in camera frame
    pitch_rad = np.radians(-view_pitch)  # Negative because we rotate the world
    c_pitch, s_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
    
    # Rotation Matrix (around X-axis for Pitch)
    R_x = np.array([
        [1,       0,        0],
        [0, c_pitch, -s_pitch],
        [0, s_pitch,  c_pitch]
    ])
    
    points_cam = points_cam @ R_x.T
    
    # 4. Filter Points Behind Camera (Z > near_clip)
    mask = points_cam[:, 2] > 0.1
    points_cam = points_cam[mask]
    
    if len(points_cam) == 0:
        print("    [Warning] No points visible in this view!")
        empty = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        return empty, empty, empty, empty, empty, empty

    # 5. Project to 2D (u, v)
    u = (points_cam[:, 0] * FOCAL_LENGTH / points_cam[:, 2]) + CX
    v = (points_cam[:, 1] * FOCAL_LENGTH / points_cam[:, 2]) + CY
    depths = points_cam[:, 2]
    
    # 6. Filter Points Outside Image Bounds
    valid_mask = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    u = u[valid_mask].astype(int)
    v = v[valid_mask].astype(int)
    depths = depths[valid_mask]
    
    # 7. Z-Buffer (Depth Map Generation)
    # Initialize with Infinity
    depth_map = np.full((IMG_H, IMG_W), np.inf, dtype=np.float32)
    
    # Sorting ensures closest points overwrite distant ones (simple painter's algo)
    # Note: For massive clouds, we might simply loop, but sorting is vectorized-friendly
    sort_idx = np.argsort(depths)[::-1] # Furthest first
    u_sorted = u[sort_idx]
    v_sorted = v[sort_idx]
    d_sorted = depths[sort_idx]
    
    # Assign depths
    depth_map[v_sorted, u_sorted] = d_sorted
    
    # 8. Post-Processing (Sparsity Filling)
    # Replace Inf with 0 for image processing
    depth_valid = depth_map.copy()
    depth_valid[depth_valid == np.inf] = 0
    
    # Dilate to fill gaps between points (The "Splat")
    kernel = np.ones((3,3), np.uint8)
    depth_filled = cv2.dilate(depth_valid, kernel, iterations=2)
    
    # 8b. Plane Visibility Pre-Filtering
    # For each detected plane, project a random subsample of its points into the
    # current camera view and compare against the depth_filled buffer.  A plane is
    # "visible" only if >= VIS_MIN_RATIO of sampled points pass the Z-test.
    VIS_SAMPLE_N   = 200   # max points to sample per plane
    VIS_DEPTH_TOL  = 0.05  # metres – same tolerance used for per-pixel Z-test
    VIS_MIN_RATIO  = 0.05  # 5% of samples must be visible
    
    plane_visible = {}  # dict mapping plane index -> bool
    
    if detected_planes is not None:
        total_yaw_rad_vis = np.radians(-(rotation_deg + view_yaw))
        pitch_rad_vis = np.radians(-view_pitch)
        c_yaw_v, s_yaw_v = np.cos(total_yaw_rad_vis), np.sin(total_yaw_rad_vis)
        c_pitch_v, s_pitch_v = np.cos(pitch_rad_vis), np.sin(pitch_rad_vis)
        
        R_z_vis = np.array([[c_yaw_v, -s_yaw_v, 0],
                            [s_yaw_v,  c_yaw_v, 0],
                            [0,        0,       1]])
        R_x_vis = np.array([[1,         0,          0],
                            [0, c_pitch_v, -s_pitch_v],
                            [0, s_pitch_v,  c_pitch_v]])
        
        for pi, pdata in enumerate(detected_planes):
            pts = pdata['points_3d']
            # Random subsample
            n_sample = min(VIS_SAMPLE_N, len(pts))
            indices = np.random.choice(len(pts), n_sample, replace=False)
            sample_pts = pts[indices]
            
            # Vectorised transform: translate -> yaw -> swizzle -> pitch
            vecs = sample_pts - camera_pose_global
            local = vecs @ R_z_vis.T
            cam = np.empty_like(local)
            cam[:, 0] = local[:, 0]
            cam[:, 1] = -local[:, 2]
            cam[:, 2] = local[:, 1]
            cam = cam @ R_x_vis.T
            
            # Keep only points in front of camera
            front = cam[:, 2] > 0.1
            if not np.any(front):
                plane_visible[pi] = False
                continue
            
            cam_f = cam[front]
            u_p = (cam_f[:, 0] * FOCAL_LENGTH / cam_f[:, 2] + CX).astype(int)
            v_p = (cam_f[:, 1] * FOCAL_LENGTH / cam_f[:, 2] + CY).astype(int)
            depths_p = cam_f[:, 2]
            
            # Bounds mask
            bm = (u_p >= 0) & (u_p < IMG_W) & (v_p >= 0) & (v_p < IMG_H)
            if not np.any(bm):
                plane_visible[pi] = False
                continue
            
            u_b, v_b, d_b = u_p[bm], v_p[bm], depths_p[bm]
            buf_d = depth_filled[v_b, u_b]
            
            # Pass Z-test: point depth <= buffer depth + tolerance, or buffer is empty (0)
            passed = (buf_d == 0) | (d_b <= buf_d + VIS_DEPTH_TOL)
            ratio = np.sum(passed) / n_sample  # relative to total sampled, not just in-bounds
            plane_visible[pi] = (ratio >= VIS_MIN_RATIO)
        
        vis_count = sum(1 for v in plane_visible.values() if v)
        print(f"    Plane visibility: {vis_count}/{len(detected_planes)} planes visible")
    
    # Normalize for visualization/edge detection
    # Map range [0.1m, 10m] to [0, 255]
    depth_vis = cv2.normalize(depth_filled, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 9. Channel A Denoising: Bilateral Filter
    # Smooth planar noise while preserving depth discontinuities
    depth_denoised = cv2.bilateralFilter(depth_vis, d=20, sigmaColor=100, sigmaSpace=75)
    
    # 10. Channel A Edge Detection (The Wireframe)
    # Use Canny to find depth discontinuities
    edges = cv2.Canny(depth_denoised, 50, 150)
    
    # 10b. Channel A Geometric Filtering: Remove jagged/high-tortuosity edges
    # This removes squiggly noise while preserving smooth structural lines
    edges_geometry_filtered = filter_jagged_edges(edges, min_length=60, epsilon=3.0, max_density=0.1)

    # 11. Channel B: Geometry-based Wireframe via Plane Intersection
    # Project 3D wireframe segments onto the 2D image plane
    normal_edges = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    
    if wireframe_segments is not None and len(wireframe_segments) > 0:
        # Define transformation functions (same as point cloud transformation above)
        def transform_point_to_camera(point_3d, cam_pose, total_yaw_rad, pitch_rad):
            """Transform a 3D world point to camera coordinates."""
            # Translate to camera center
            vec = point_3d - cam_pose
            
            # Apply Yaw rotation (around Z-axis)
            c_yaw, s_yaw = np.cos(total_yaw_rad), np.sin(total_yaw_rad)
            R_z = np.array([
                [c_yaw, -s_yaw, 0],
                [s_yaw,  c_yaw, 0],
                [0,      0,     1]
            ])
            local_vec = R_z @ vec
            
            # Swizzle coordinates to camera frame
            # World X -> Cam X, World Y -> Cam Z, World Z -> Cam -Y
            point_cam = np.array([
                local_vec[0],   # X -> X
                -local_vec[2],  # Z -> -Y
                local_vec[1]    # Y -> Z (Forward)
            ])
            
            # Apply Pitch rotation (around X-axis)
            c_pitch, s_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
            R_x = np.array([
                [1,       0,        0],
                [0, c_pitch, -s_pitch],
                [0, s_pitch,  c_pitch]
            ])
            point_cam = R_x @ point_cam
            
            return point_cam
        
        def project_to_2d(point_cam):
            """Project a camera-space point to 2D image coordinates."""
            if point_cam[2] <= 0.1:  # Behind camera
                return None
            u = (point_cam[0] * FOCAL_LENGTH / point_cam[2]) + CX
            v = (point_cam[1] * FOCAL_LENGTH / point_cam[2]) + CY
            return np.array([u, v])
        
        # Precompute rotation parameters
        total_yaw_rad_seg = np.radians(-(rotation_deg + view_yaw))
        pitch_rad_seg = np.radians(-view_pitch)
        
        segments_skipped_by_plane = 0
        
        # Process each wireframe segment
        for seg in wireframe_segments:
            start_3d = seg['start']
            end_3d   = seg['end']
            pi_a     = seg['plane_i']
            pi_b     = seg['plane_j']
            
            # Pre-filter: skip segment if NEITHER parent plane is visible
            if plane_visible and not plane_visible.get(pi_a, True) and not plane_visible.get(pi_b, True):
                segments_skipped_by_plane += 1
                continue
            
            # Transform endpoints to camera space
            start_cam = transform_point_to_camera(start_3d, camera_pose_global, total_yaw_rad_seg, pitch_rad_seg)
            end_cam = transform_point_to_camera(end_3d, camera_pose_global, total_yaw_rad_seg, pitch_rad_seg)
            
            # Skip segments entirely behind camera
            if start_cam[2] <= 0.1 and end_cam[2] <= 0.1:
                continue
            
            # Clip segment to near plane if partially behind camera
            near_clip = 0.1
            if start_cam[2] <= near_clip or end_cam[2] <= near_clip:
                # Parametric line: P(t) = start + t * (end - start), t in [0, 1]
                # Find t where Z = near_clip
                direction = end_cam - start_cam
                if abs(direction[2]) > 1e-8:
                    t_clip = (near_clip - start_cam[2]) / direction[2]
                    clipped_point = start_cam + t_clip * direction
                    
                    if start_cam[2] <= near_clip:
                        start_cam = clipped_point
                    else:
                        end_cam = clipped_point
            
            # Project to 2D
            start_2d = project_to_2d(start_cam)
            end_2d = project_to_2d(end_cam)
            
            if start_2d is None or end_2d is None:
                continue
            
            # Check if line is within image bounds (at least partially)
            # Use Cohen-Sutherland-like clipping logic
            u1, v1 = int(np.clip(start_2d[0], 0, IMG_W - 1)), int(np.clip(start_2d[1], 0, IMG_H - 1))
            u2, v2 = int(np.clip(end_2d[0], 0, IMG_W - 1)), int(np.clip(end_2d[1], 0, IMG_H - 1))
            
            # Skip lines completely outside the image
            if (start_2d[0] < 0 and end_2d[0] < 0) or (start_2d[0] >= IMG_W and end_2d[0] >= IMG_W):
                continue
            if (start_2d[1] < 0 and end_2d[1] < 0) or (start_2d[1] >= IMG_H and end_2d[1] >= IMG_H):
                continue
            
            # Z-Test: Discretize the 3D segment and only draw visible portions
            SAMPLE_STEP = 0.02   # meters between test points
            DEPTH_TOL   = 0.05   # occlusion tolerance in meters
            
            seg_vec_3d = end_3d - start_3d
            seg_length = np.linalg.norm(seg_vec_3d)
            num_samples = max(int(np.ceil(seg_length / SAMPLE_STEP)), 2)
            
            prev_px = None       # previous visible pixel (for drawing sub-segments)
            prev_visible = False
            
            for si in range(num_samples + 1):
                t = si / num_samples  # parametric position [0, 1]
                sample_3d = start_3d + t * seg_vec_3d
                
                # Transform sample to camera space
                sample_cam = transform_point_to_camera(
                    sample_3d, camera_pose_global,
                    total_yaw_rad_seg, pitch_rad_seg
                )
                
                # Skip if behind camera
                if sample_cam[2] <= 0.1:
                    prev_visible = False
                    prev_px = None
                    continue
                
                # Project to 2D
                sample_2d = project_to_2d(sample_cam)
                if sample_2d is None:
                    prev_visible = False
                    prev_px = None
                    continue
                
                su = int(round(sample_2d[0]))
                sv = int(round(sample_2d[1]))
                
                # Bounds check
                if su < 0 or su >= IMG_W or sv < 0 or sv >= IMG_H:
                    prev_visible = False
                    prev_px = None
                    continue
                
                # Depth test against the filled depth buffer
                buf_depth = depth_filled[sv, su]
                sample_depth = sample_cam[2]
                
                # buf_depth == 0 means no depth data (empty region); treat as visible
                is_visible = (buf_depth == 0) or (sample_depth <= buf_depth + DEPTH_TOL)
                
                cur_px = (su, sv)
                
                if is_visible:
                    if prev_visible and prev_px is not None:
                        cv2.line(normal_edges, prev_px, cur_px, 255, thickness=1)
                    prev_px = cur_px
                    prev_visible = True
                else:
                    prev_visible = False
                    prev_px = None
        
        print(f"    Segments skipped by plane pre-filter: {segments_skipped_by_plane}/{len(wireframe_segments)}")
    
    # 12. Geometric Filtering: Remove short/noisy edge segments from both channels
    # Filter Channel A (geometry-filtered depth edges) with min_size of 60 pixels
    edges_filtered = filter_short_edges(edges_geometry_filtered, min_size=60)
    
    # Filter Channel B (normal edges) with min_size of 60 pixels
    normal_edges_filtered = filter_short_edges(normal_edges, min_size=60)
    
    # 13. Combined Wireframe: Bitwise OR of filtered Channel A and Channel B
    combined_wireframe = cv2.bitwise_or(edges_filtered, normal_edges_filtered)
    
    return edges, depth_vis, normal_edges, edges_filtered, normal_edges_filtered, combined_wireframe

# ==========================================
# 3. MAIN
# ==========================================
def main():
    print("--- Testing Synthetic Point Cloud Renderer ---")
    
    # 1. Paths
    point_cloud_name = "tmb_office_one_corridor"
    pcd_path = project_root / "data" / "raw_point_cloud" / f"{point_cloud_name}.ply"
    json_path = project_root / "data" / "reconstructed_floorplans_RoomFormer" / point_cloud_name / "global_alignment.json"
    # Assuming metadata is here based on generate_density_image structure
    meta_path = project_root / "data" / "density_image" / point_cloud_name / "metadata.json"
    
    target_rooms = ["TMB_office1", "TMB_corridor_south2", "TMB_corridor_south1"]

    # Define the base output folder (e.g. .../debug_renderer/tmb_office_one_corridor)
    output_base_dir = project_root / "data" / "debug_renderer" / point_cloud_name
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Data
    print("Loading Metadata & Alignment...")
    with open(meta_path) as f: meta = json.load(f)
    with open(json_path) as f: align = json.load(f)

    # Extract the cumulative rotation and translation saved by generate_density_image.py
    rotation_matrix = np.array(meta['rotation_matrix'])  # 3x3
    translation = np.array(meta['translation'])           # 3-vector

    print(f"Loading Point Cloud: {pcd_path.name}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))

    # Apply the same alignment transforms used during density map generation
    # so the point cloud is in the exact same coordinate system.
    pcd.rotate(rotation_matrix, center=(0, 0, 0))

    points = np.asarray(pcd.points)
    
    print(f"Point Cloud Bounds (after alignment):")
    print(f"  Min: {pcd.get_min_bound()}")
    print(f"  Max: {pcd.get_max_bound()}")
    
    # Precompute shared metadata values (constant across rooms)
    img_width = meta['image_width']
    min_coords = [c / 1000.0 for c in meta['min_coords']]
    max_dim = meta['max_dim'] / 1000.0
    offset = meta.get('offset', [0, 0])
    offset = [o / 1000.0 for o in offset]  # Convert offset to meters too
    scale_factor = max_dim / img_width
    
    # 3. Extract 3D Wireframe from Point Cloud (Plane Intersections) — once for the whole cloud
    wireframe_segments, detected_planes = extract_3d_wireframe(
        points,
        output_dir=output_base_dir,
        distance_threshold=0.03,
        ransac_n=3,
        num_iterations=1000,
        min_remaining_points=2000,
        dbscan_eps=0.20,
        dbscan_min_samples=30,
        min_hull_area=0.90,  # in square meters
        parallel_threshold=0.9
    )

    # 4. Iterate over each target room
    for target_room in target_rooms:
        print(f"\n{'='*60}")
        print(f"Processing Room: {target_room}")
        print(f"{'='*60}")
        
        # Create per-room output directories
        room_output_dir = output_base_dir / target_room
        output_dir_a = room_output_dir / "Channel_A"
        output_dir_a_filtered = room_output_dir / "Channel_A_Filtered"
        output_dir_b = room_output_dir / "Channel_B"
        output_dir_b_filtered = room_output_dir / "Channel_B_Filtered"
        output_dir_combined = room_output_dir / "Combined"
        output_dir_a.mkdir(parents=True, exist_ok=True)
        output_dir_a_filtered.mkdir(parents=True, exist_ok=True)
        output_dir_b.mkdir(parents=True, exist_ok=True)
        output_dir_b_filtered.mkdir(parents=True, exist_ok=True)
        output_dir_combined.mkdir(parents=True, exist_ok=True)
        
        # Extract Room Info
        room_info = next(
            (r for r in align['alignment_results'] if r['room_name'] == target_room),
            None
        )
        if room_info is None:
            print(f"  [WARNING] Room '{target_room}' not found in alignment results, skipping.")
            continue
        
        # Construct Metric Pose using Absolute Coordinate Transformation
        pose_px = room_info['camera_pose_global']
        
        cam_x = min_coords[0] + (pose_px[0] * scale_factor)
        cam_y = min_coords[1] + max_dim - (pose_px[1] * scale_factor)
        
        pose_world_xy = np.array([cam_x, cam_y])
        
        print(f"\n[DEBUG] Pose Calculation (Absolute Coordinate Transformation):")
        print(f"  Pixel: {pose_px}, Image Width: {img_width}")
        print(f"  min_coords: {min_coords}, max_dim: {max_dim}, offset: {offset}")
        print(f"  scale_factor: {scale_factor:.6f} m/px")
        print(f"  Absolute World X: {cam_x:.3f} m")
        print(f"  Absolute World Y: {cam_y:.3f} m")
        
        # Z from Local Floor Detection + 1.6m
        local_radius = 2.0
        distances_xy = np.sqrt((points[:, 0] - cam_x)**2 + (points[:, 1] - cam_y)**2)
        local_mask = distances_xy <= local_radius
        local_points = points[local_mask]
        
        if len(local_points) > 100:
            floor_z = find_floor_z(local_points)
            print(f"  Local floor detection: {len(local_points)} points within {local_radius}m radius")
        else:
            floor_z = find_floor_z(points)
            print(f"  WARNING: Only {len(local_points)} local points, using global floor detection")
        
        cam_z = floor_z + 1.6
        
        # Yaw from Alignment
        yaw_deg = room_info['transformation']['rotation_deg']
        
        final_pose = np.array([pose_world_xy[0], pose_world_xy[1], cam_z])
        
        print(f"  Floor Z: {floor_z:.3f} m, Camera Z: {cam_z:.3f} m")
        print(f"\nComputed Camera Pose: {final_pose}")
        print(f"Room Orientation: {yaw_deg} deg")
        
        # 5. Render all views for this room
        print(f"\nRendering {len(VIEWS_TO_RENDER)} Synthetic Views for '{target_room}'...")
        
        for view_yaw, view_pitch in VIEWS_TO_RENDER:
            edges, depth_map, normal_edges, edges_filtered, normal_edges_filtered, combined_wireframe = render_synthetic_wireframe(
                points, final_pose, yaw_deg, 
                view_yaw=view_yaw, view_pitch=view_pitch,
                wireframe_segments=wireframe_segments,
                detected_planes=detected_planes
            )
            
            # Save Channel A outputs (depth-based wireframe)
            edge_path = output_dir_a / f"synthetic_wireframe_yaw{view_yaw}_pitch{view_pitch}.png"
            depth_path = output_dir_a / f"synthetic_depth_yaw{view_yaw}_pitch{view_pitch}.png"
            
            cv2.imwrite(str(edge_path), edges)
            cv2.imwrite(str(depth_path), depth_map)
            
            # Save Channel A Filtered outputs (short edges removed)
            edge_filtered_path = output_dir_a_filtered / f"synthetic_wireframe_filtered_yaw{view_yaw}_pitch{view_pitch}.png"
            cv2.imwrite(str(edge_filtered_path), edges_filtered)
            
            # Save Channel B outputs (normal-based crease edges)
            normal_edge_path = output_dir_b / f"synthetic_normal_edges_yaw{view_yaw}_pitch{view_pitch}.png"
            cv2.imwrite(str(normal_edge_path), normal_edges)
            
            # Save Channel B Filtered outputs (short edges removed)
            normal_edge_filtered_path = output_dir_b_filtered / f"synthetic_normal_edges_filtered_yaw{view_yaw}_pitch{view_pitch}.png"
            cv2.imwrite(str(normal_edge_filtered_path), normal_edges_filtered)
            
            # Save Combined wireframe (filtered A | filtered B)
            combined_path = output_dir_combined / f"synthetic_combined_yaw{view_yaw}_pitch{view_pitch}.png"
            cv2.imwrite(str(combined_path), combined_wireframe)
            
            print(f"    Saved: {edge_path.name}, {edge_filtered_path.name}, {normal_edge_path.name}, {normal_edge_filtered_path.name}, {combined_path.name}")
        
        print(f"\n[Success] Room '{target_room}' — {len(VIEWS_TO_RENDER)} views saved to: {room_output_dir}")
    
    print(f"\n{'='*60}")
    print(f"[Done] All {len(target_rooms)} rooms processed.")
    print(f"  Base output: {output_base_dir}")
    print(f"  debug_ransac_planes.ply saved in: {output_base_dir}")
    for room in target_rooms:
        print(f"  {room}/: Channel_A, Channel_A_Filtered, Channel_B, Channel_B_Filtered, Combined")

if __name__ == "__main__":
    main()