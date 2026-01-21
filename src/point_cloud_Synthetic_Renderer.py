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

def pixels_to_world(pixel_coords, metadata):
    """Converts Density Image Pixels to World Meters."""
    u, v = pixel_coords
    min_coords = np.array(metadata['min_coords'])
    offset = np.array(metadata['offset'])
    max_dim = metadata['max_dim']
    width = metadata['image_width']
    
    # Debug prints for coordinate transform
    print(f"\n[DEBUG] pixels_to_world():")
    print(f"  Input pixel_coords: u={u}, v={v}")
    print(f"  metadata['min_coords']: {metadata['min_coords']}")
    print(f"  metadata['max_dim']: {metadata['max_dim']}")
    print(f"  metadata['image_width']: {metadata['image_width']}")
    print(f"  metadata['offset']: {metadata['offset']}")
    
    scale_factor = width - 1
    print(f"  Calculated scale_factor (width-1): {scale_factor}")
    
    world_x = (u / scale_factor * max_dim) - offset[0] + min_coords[0]
    world_y = (v / scale_factor * max_dim) - offset[1] + min_coords[1]
    
    print(f"  Calculated world_x: {world_x}")
    print(f"  Calculated world_y: {world_y}")
    
    return np.array([world_x, world_y])

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

def render_synthetic_wireframe(points, camera_pose_global, rotation_deg, view_yaw=0, view_pitch=0):
    """
    Renders a synthetic depth-based wireframe from the point cloud.
    
    Args:
        points: (N, 3) numpy array of point cloud
        camera_pose_global: [x, y, z] in meters
        rotation_deg: Global alignment rotation (Room orientation)
        view_yaw: Relative yaw of the virtual camera (0 = Front, 90 = Right, etc.)
        view_pitch: Relative pitch of the virtual camera (positive = look up, negative = look down)
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
    
    # 11. Channel B: Crease Edge Extraction via Surface Normals
    # Compute gradients in X and Y directions using Sobel
    grad_x = cv2.Sobel(depth_denoised, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_denoised, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute surface normal components from depth gradients
    # For a depth map Z(x,y), surface normal N = (-dZ/dx, -dZ/dy, 1) normalized
    # We use a scale factor to balance the Z component
    z_scale = 1.0  # Adjustable: higher = flatter normals, lower = steeper
    
    # Compute normal components
    nx = -grad_x
    ny = -grad_y
    nz = np.ones_like(grad_x) * z_scale * 255  # Scale to match gradient magnitude
    
    # Normalize the normal vectors
    magnitude = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8  # Avoid division by zero
    nx_norm = nx / magnitude
    ny_norm = ny / magnitude
    nz_norm = nz / magnitude
    
    # Create a visual normal map (RGB: X, Y, Z mapped to 0-255)
    # Standard normal map encoding: (N + 1) / 2 * 255
    normal_map = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    normal_map[:, :, 0] = ((nx_norm + 1) / 2 * 255).astype(np.uint8)  # R = X
    normal_map[:, :, 1] = ((ny_norm + 1) / 2 * 255).astype(np.uint8)  # G = Y
    normal_map[:, :, 2] = ((nz_norm + 1) / 2 * 255).astype(np.uint8)  # B = Z
    
    # Convert normal map to grayscale for edge detection
    # Use orientation angle (atan2 of X and Y components) to detect orientation changes
    normal_angle = np.arctan2(ny_norm, nx_norm)  # Range: [-pi, pi]
    normal_angle_vis = cv2.normalize(normal_angle, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Run Canny on the normal angle map to detect orientation discontinuities
    # These correspond to geometric creases like wall-ceiling seams
    normal_edges = cv2.Canny(normal_angle_vis, 50, 150)
    
    # 12. Geometric Filtering: Remove short/noisy edge segments from both channels
    # Filter Channel A (depth edges) with min_size of 60 pixels
    edges_filtered = filter_short_edges(edges, min_size=60)
    
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
    pcd_path = project_root / "data" / "raw_point_cloud" / "lab_new_cleaned_no_roof.ply"
    json_path = project_root / "data" / "reconstructed_floorplans_RoomFormer" / "lab_new_cleaned_no_roof" / "global_alignment.json"
    # Assuming metadata is here based on generate_density_image structure
    meta_path = project_root / "data" / "density_image" / "lab_new_cleaned_no_roof" / "metadata.json"
    
    target_room = "Setup13"

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
    # X, Y from Global Pixels -> Meters
    pose_px = room_info['camera_pose_global']
    pose_world_xy = pixels_to_world(pose_px, meta)
    
    # ==========================================
    # Unit Scaling Check (mm vs m mismatch)
    # ==========================================
    pcd_min = np.array(pcd.get_min_bound())
    pcd_max = np.array(pcd.get_max_bound())
    pcd_centroid = (pcd_min + pcd_max) / 2.0
    
    # Compare magnitude of pose_world_xy to point cloud centroid (XY only)
    pose_magnitude = np.linalg.norm(pose_world_xy)
    centroid_magnitude = np.linalg.norm(pcd_centroid[:2])  # XY only
    
    print(f"\n[DEBUG] Unit Scaling Check:")
    print(f"  pose_world_xy magnitude: {pose_magnitude:.2f}")
    print(f"  pcd centroid XY magnitude: {centroid_magnitude:.2f}")
    
    if centroid_magnitude > 0:
        ratio = pose_magnitude / centroid_magnitude
        print(f"  Ratio (pose/centroid): {ratio:.2f}")
        
        if ratio > 500:
            print("  Unit mismatch detected (mm vs m). Scaling camera pose by 0.001.")
            pose_world_xy = pose_world_xy / 1000.0
            
            # Check Y-axis orientation after scaling
            # If pose Y is negative but cloud Y is positive, flip Y
            if pose_world_xy[1] < 0 and pcd_min[1] > 0:
                print("  Inverting Y-axis to match point cloud.")
                pose_world_xy[1] = -pose_world_xy[1]
            elif pose_world_xy[1] < pcd_min[1] or pose_world_xy[1] > pcd_max[1]:
                # Y is out of bounds - try inverting
                inverted_y = -pose_world_xy[1]
                if pcd_min[1] <= inverted_y <= pcd_max[1]:
                    print("  Inverting Y-axis to match point cloud (Y was out of bounds).")
                    pose_world_xy[1] = inverted_y
    
    print(f"  Final pose_world_xy after corrections: {pose_world_xy}")
    
    # Z from Floor Detection + 1.6m
    floor_z = find_floor_z(points)
    cam_z = floor_z + 1.6
    
    # Yaw from Alignment
    yaw_deg = room_info['transformation']['rotation_deg']
    
    final_pose = np.array([pose_world_xy[0], pose_world_xy[1], cam_z])
    
    print(f"\nComputed Camera Pose: {final_pose}")
    print(f"Room Orientation: {yaw_deg} deg")

    # 4. Render all views!
    print(f"\nRendering {len(VIEWS_TO_RENDER)} Synthetic Views...")
    
    for view_yaw, view_pitch in VIEWS_TO_RENDER:
        edges, depth_map, normal_edges, edges_filtered, normal_edges_filtered, combined_wireframe = render_synthetic_wireframe(
            points, final_pose, yaw_deg, 
            view_yaw=view_yaw, view_pitch=view_pitch
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