import numpy as np
import open3d as o3d
import cv2
import json
import sys
from pathlib import Path

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


def filter_jagged_edges(binary_image, min_length=60, epsilon=5.0, max_density=0.07, max_tortuosity=3.0):
    """
    Removes high-tortuosity (jagged/squiggly) contours from a binary edge image.
    
    This filter discards contours that are either too short, have excessive
    "vertex density" (many vertices relative to their length), or have high
    "tortuosity" (path length much greater than straight-line distance),
    which indicates jagged noise rather than smooth structural edges.
    
    Args:
        binary_image: Input binary image (white edges on black background)
        min_length: Minimum perimeter length (in pixels) for a contour to be retained
        epsilon: Approximation accuracy parameter for cv2.approxPolyDP.
                 Larger values = more aggressive simplification (ignores small jitters).
        max_density: Maximum allowed vertex density (num_vertices / perimeter).
                     Contours with higher density are considered "jagged" and removed.
                     Typical values: 0.05-0.10 (lower = stricter filtering)
        max_tortuosity: Maximum allowed tortuosity ratio (perimeter / chord_length).
                        High values indicate tight squiggles or closed loops.
    
    Returns:
        Filtered binary image with jagged contours removed
    """
    # Create output image (start with black)
    output = np.zeros_like(binary_image)
    
    # Find all contours in the binary image
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    
    for contour in contours:
        # Calculate the perimeter (arc length) of the contour
        perimeter = cv2.arcLength(contour, closed=False)
        
        # Skip contours shorter than min_length
        if perimeter < min_length:
            continue
        
        # Compute polygonal approximation
        # epsilon controls how much the contour is simplified (larger = more aggressive)
        approx = cv2.approxPolyDP(contour, epsilon, closed=False)
        
        # Calculate vertex density: number of vertices per unit length
        num_vertices = len(approx)
        vertex_density = num_vertices / perimeter if perimeter > 0 else float('inf')
        
        # Calculate Tortuosity: ratio of path length to straight-line (chord) distance
        # High tortuosity indicates squiggly paths or closed loops (noise blobs)
        first_point = contour[0][0]  # Shape is (N, 1, 2), so [0][0] gets (x, y)
        last_point = contour[-1][0]
        chord_length = np.linalg.norm(last_point - first_point)
        tortuosity = perimeter / (chord_length + 1e-5)
        
        # Dual-Filter Logic:
        # For very long lines (perimeter > 200), relax tortuosity to allow U-shapes/corners
        # but still enforce strict density check
        if perimeter > 200:
            # Long contours: allow higher tortuosity (U-shapes, corners)
            # but require low vertex density (must be structurally simple)
            keep_contour = (vertex_density <= max_density)
        else:
            # Short/medium contours: require BOTH low density AND low tortuosity
            # This aggressively removes noise blobs and tight squiggles
            keep_contour = (vertex_density <= max_density) and (tortuosity < max_tortuosity)
        
        if keep_contour:
            # Draw the contour on the output image
            cv2.drawContours(output, [contour], -1, 255, thickness=1)
    
    return output


def extract_3d_wireframe(points, num_planes=6, distance_threshold=0.05, ransac_n=3, num_iterations=1000, min_inliers=500, parallel_threshold=0.9):
    """
    Extracts a 3D wireframe by finding plane intersections using RANSAC.
    
    Args:
        points: (N, 3) numpy array of point cloud
        num_planes: Number of dominant planes to detect (e.g., 6 for walls/floor/ceiling)
        distance_threshold: RANSAC distance threshold for inlier classification
        ransac_n: Number of points to sample for RANSAC
        num_iterations: Number of RANSAC iterations
        min_inliers: Minimum number of inliers for a valid plane
        parallel_threshold: Dot product threshold to consider planes parallel (skip intersection)
    
    Returns:
        List of 3D line segments as tuples: [(start_point, end_point), ...]
        Each point is a numpy array of shape (3,)
    """
    print(f"\n[Wireframe Extraction] Starting RANSAC plane segmentation...")
    
    # Create Open3D point cloud for segmentation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Store detected planes: list of (plane_eq, inlier_points)
    # plane_eq = [a, b, c, d] where ax + by + cz + d = 0
    detected_planes = []
    remaining_pcd = pcd
    
    for i in range(num_planes):
        if len(remaining_pcd.points) < min_inliers:
            print(f"  [Info] Stopping early: only {len(remaining_pcd.points)} points remaining")
            break
        
        # RANSAC plane segmentation
        plane_model, inlier_indices = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        if len(inlier_indices) < min_inliers:
            print(f"  [Info] Plane {i+1}: Only {len(inlier_indices)} inliers, skipping")
            break
        
        # Extract inlier points
        inlier_pcd = remaining_pcd.select_by_index(inlier_indices)
        inlier_points = np.asarray(inlier_pcd.points)
        
        # Store plane equation and inliers
        detected_planes.append((np.array(plane_model), inlier_points))
        
        # Get normal vector for display
        normal = plane_model[:3]
        print(f"  Plane {i+1}: {len(inlier_indices)} inliers, Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
        
        # Remove inliers from remaining cloud
        remaining_pcd = remaining_pcd.select_by_index(inlier_indices, invert=True)
    
    print(f"  Detected {len(detected_planes)} planes")
    
    # Find plane-plane intersections
    wireframe_segments = []
    
    print(f"\n[Wireframe Extraction] Computing plane intersections...")
    
    for i in range(len(detected_planes)):
        for j in range(i + 1, len(detected_planes)):
            plane1_eq, plane1_pts = detected_planes[i]
            plane2_eq, plane2_pts = detected_planes[j]
            
            # Get normal vectors
            n1 = plane1_eq[:3]
            n2 = plane2_eq[:3]
            
            # Check if planes are parallel
            dot_product = abs(np.dot(n1, n2))
            if dot_product > parallel_threshold:
                # Planes are nearly parallel, skip
                continue
            
            # Compute intersection line direction (cross product of normals)
            line_dir = np.cross(n1, n2)
            line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-8)  # Normalize
            
            # Find a point on the intersection line
            # Solve the system: n1 · p = -d1, n2 · p = -d2
            # We fix one coordinate and solve for the other two
            d1, d2 = plane1_eq[3], plane2_eq[3]
            
            # Find which axis to fix (use the one with smallest line_dir component)
            fix_axis = np.argmin(np.abs(line_dir))
            
            # Build 2x2 system for the other two axes
            axes = [k for k in range(3) if k != fix_axis]
            A = np.array([[n1[axes[0]], n1[axes[1]]],
                          [n2[axes[0]], n2[axes[1]]]])
            b = np.array([-d1 - n1[fix_axis] * 0, -d2 - n2[fix_axis] * 0])  # Assuming fixed axis = 0
            
            # Check if system is solvable
            if abs(np.linalg.det(A)) < 1e-8:
                continue
            
            solution = np.linalg.solve(A, b)
            
            # Construct point on line
            point_on_line = np.zeros(3)
            point_on_line[fix_axis] = 0
            point_on_line[axes[0]] = solution[0]
            point_on_line[axes[1]] = solution[1]
            
            # Project inlier points from both planes onto the intersection line
            # to find the parameter intervals covered by each plane
            def project_to_line(points, line_point, line_direction):
                """Project points onto a line and return parameter values."""
                vecs = points - line_point
                t_values = np.dot(vecs, line_direction)
                return t_values
            
            t1_values = project_to_line(plane1_pts, point_on_line, line_dir)
            t2_values = project_to_line(plane2_pts, point_on_line, line_dir)
            
            # Find intervals
            t1_min, t1_max = np.min(t1_values), np.max(t1_values)
            t2_min, t2_max = np.min(t2_values), np.max(t2_values)
            
            # Find intersection of intervals
            t_overlap_min = max(t1_min, t2_min)
            t_overlap_max = min(t1_max, t2_max)
            
            # Check if there's a valid overlap
            if t_overlap_max <= t_overlap_min:
                continue
            
            # Skip very short segments (noise)
            segment_length = t_overlap_max - t_overlap_min
            if segment_length < 0.1:  # Minimum 10cm segment
                continue
            
            # Generate 3D line segment endpoints
            start_point = point_on_line + t_overlap_min * line_dir
            end_point = point_on_line + t_overlap_max * line_dir
            
            wireframe_segments.append((start_point, end_point))
    
    print(f"  Found {len(wireframe_segments)} wireframe segments")
    
    return wireframe_segments

def render_synthetic_wireframe(points, camera_pose_global, rotation_deg, view_yaw=0, view_pitch=0, wireframe_segments=None):
    """
    Renders a synthetic depth-based wireframe from the point cloud.
    
    Args:
        points: (N, 3) numpy array of point cloud
        camera_pose_global: [x, y, z] in meters
        rotation_deg: Global alignment rotation (Room orientation)
        view_yaw: Relative yaw of the virtual camera (0 = Front, 90 = Right, etc.)
        view_pitch: Relative pitch of the virtual camera (positive = look up, negative = look down)
        wireframe_segments: List of 3D line segments for Channel B (from extract_3d_wireframe)
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
        
        # Process each wireframe segment
        for start_3d, end_3d in wireframe_segments:
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
            
            # Draw the line segment
            cv2.line(normal_edges, (u1, v1), (u2, v2), 255, thickness=1)
    
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
    pcd_path = project_root / "data" / "raw_point_cloud" / "Area_3_study_no_RGB.ply"
    json_path = project_root / "data" / "reconstructed_floorplans_RoomFormer" / "Area_3_study_no_RGB" / "global_alignment.json"
    # Assuming metadata is here based on generate_density_image structure
    meta_path = project_root / "data" / "density_image" / "Area_3_study_no_RGB" / "metadata.json"
    
    target_room = "Area3_study"

    # Extract point cloud name from pcd_path stem for output directory structure
    point_cloud_name = pcd_path.stem
    output_dir_a = project_root / "data" / "debug_renderer" / point_cloud_name / "Channel_A"
    output_dir_a_filtered = project_root / "data" / "debug_renderer" / point_cloud_name / "Channel_A_Filtered"
    output_dir_b = project_root / "data" / "debug_renderer" / point_cloud_name / "Channel_B"
    output_dir_b_filtered = project_root / "data" / "debug_renderer" / point_cloud_name / "Channel_B_Filtered"
    output_dir_combined = project_root / "data" / "debug_renderer" / point_cloud_name / "Combined"
    output_dir_a.mkdir(parents=True, exist_ok=True)
    output_dir_a_filtered.mkdir(parents=True, exist_ok=True)
    output_dir_b.mkdir(parents=True, exist_ok=True)
    output_dir_b_filtered.mkdir(parents=True, exist_ok=True)
    output_dir_combined.mkdir(parents=True, exist_ok=True)

    # 2. Load Data
    print(f"Loading Point Cloud: {pcd_path.name}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)
    
    print(f"Point Cloud Bounds:")
    print(f"  Min: {pcd.get_min_bound()}")
    print(f"  Max: {pcd.get_max_bound()}")
    
    print("Loading Metadata & Alignment...")
    with open(meta_path) as f: meta = json.load(f)
    with open(json_path) as f: align = json.load(f)
    
    # Extract Room Info
    
    room_info = next(r for r in align['alignment_results'] if r['room_name'] == target_room)
    
    # 3. Construct Metric Pose
    # Use relative position in density image to interpolate Point Cloud Bounds
    # This bypasses global coordinate mismatches
    
    pose_px = room_info['camera_pose_global']
    img_width = meta['image_width']
    
    # Calculate relative position (0.0 to 1.0)
    rel_u = pose_px[0] / img_width
    rel_v = pose_px[1] / img_width  # Assuming square image
    
    # Get Point Cloud Bounds
    pcd_min = np.array(pcd.get_min_bound())
    pcd_max = np.array(pcd.get_max_bound())
    pcd_range = pcd_max - pcd_min
    
    # Interpolate Pose
    # X: Min + rel_u * Range
    cam_x = pcd_min[0] + rel_u * pcd_range[0]
    
    # Y: Max - rel_v * Range (Image V goes down, World Y goes up)
    cam_y = pcd_max[1] - rel_v * pcd_range[1]
    
    pose_world_xy = np.array([cam_x, cam_y])
    
    print(f"\n[DEBUG] Pose Calculation (Relative to PC Bounds):")
    print(f"  Pixel: {pose_px}, Image Width: {img_width}")
    print(f"  Relative: ({rel_u:.3f}, {rel_v:.3f})")
    print(f"  PC Bounds X: [{pcd_min[0]:.3f}, {pcd_max[0]:.3f}]")
    print(f"  PC Bounds Y: [{pcd_min[1]:.3f}, {pcd_max[1]:.3f}]")
    print(f"  Calculated Pose: {pose_world_xy}")
    
    # Z from Floor Detection + 1.6m
    floor_z = find_floor_z(points)
    cam_z = floor_z + 1.6
    
    # Yaw from Alignment
    yaw_deg = room_info['transformation']['rotation_deg']
    
    final_pose = np.array([pose_world_xy[0], pose_world_xy[1], cam_z])
    
    print(f"\nComputed Camera Pose: {final_pose}")
    print(f"Room Orientation: {yaw_deg} deg")
    
    # 4. Extract 3D Wireframe from Point Cloud (Plane Intersections)
    wireframe_segments = extract_3d_wireframe(
        points,
        num_planes=6,
        distance_threshold=0.05,
        ransac_n=3,
        num_iterations=1000,
        min_inliers=500,
        parallel_threshold=0.9
    )

    # 5. Render all views!
    print(f"\nRendering {len(VIEWS_TO_RENDER)} Synthetic Views...")
    
    for view_yaw, view_pitch in VIEWS_TO_RENDER:
        edges, depth_map, normal_edges, edges_filtered, normal_edges_filtered, combined_wireframe = render_synthetic_wireframe(
            points, final_pose, yaw_deg, 
            view_yaw=view_yaw, view_pitch=view_pitch,
            wireframe_segments=wireframe_segments
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
    
    print(f"\n[Success] All {len(VIEWS_TO_RENDER)} views saved to:")
    print(f"  Channel A: {output_dir_a}")
    print(f"  Channel A Filtered: {output_dir_a_filtered}")
    print(f"  Channel B: {output_dir_b}")
    print(f"  Channel B Filtered: {output_dir_b_filtered}")
    print(f"  Combined: {output_dir_combined}")

if __name__ == "__main__":
    main()