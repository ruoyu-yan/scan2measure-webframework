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

def render_synthetic_wireframe(points, camera_pose_global, rotation_deg, view_yaw=0):
    """
    Renders a synthetic depth-based wireframe from the point cloud.
    
    Args:
        points: (N, 3) numpy array of point cloud
        camera_pose_global: [x, y, z] in meters
        rotation_deg: Global alignment rotation (Room orientation)
        view_yaw: Relative yaw of the virtual camera (0 = Front, 90 = Right, etc.)
    """
    print(f"  > Rendering View (Yaw={view_yaw} deg)...")
    
    # 1. Translate Points to Camera Center
    # Vector from Camera -> Point
    vecs = points - camera_pose_global
    
    # 2. Rotate Points to Align with Camera Orientation
    # We need to undo the Global Rotation AND apply the specific View Rotation.
    # Total rotation to "un-rotate" the world to the camera's front:
    # We essentially rotate the world by -(Global_Yaw + View_Yaw)
    
    total_yaw_rad = np.radians(-(rotation_deg + view_yaw))
    c, s = np.cos(total_yaw_rad), np.sin(total_yaw_rad)
    
    # Rotation Matrix (around Z-axis)
    R_z = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
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
    
    # 4. Filter Points Behind Camera (Z > near_clip)
    mask = points_cam[:, 2] > 0.1
    points_cam = points_cam[mask]
    
    if len(points_cam) == 0:
        print("    [Warning] No points visible in this view!")
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8), np.zeros((IMG_H, IMG_W), dtype=np.uint8)

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
    
    # 9. Edge Detection (The Wireframe)
    # Use Canny to find depth discontinuities
    edges = cv2.Canny(depth_vis, 50, 150)
    
    return edges, depth_vis

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
    
    output_dir = project_root / "data" / "debug_renderer"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    target_room = "Area3_study"
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

    # 4. Render!
    # Let's render the Front view (yaw=0 relative to camera)
    print("\nRendering Synthetic Views...")
    edges, depth_map = render_synthetic_wireframe(points, final_pose, yaw_deg, view_yaw=0)
    
    # 5. Save Results
    edge_path = output_dir / "synthetic_wireframe_front.png"
    depth_path = output_dir / "synthetic_depth_front.png"
    
    cv2.imwrite(str(edge_path), edges)
    cv2.imwrite(str(depth_path), depth_map)
    
    print(f"\n[Success] Output saved to {output_dir}")
    print(f"  - {edge_path.name}")
    print(f"  - {depth_path.name}")

if __name__ == "__main__":
    main()