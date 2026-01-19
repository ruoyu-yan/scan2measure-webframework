import sys
import json
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path

# ==========================================
# 1. PATH SETUP
# ==========================================
# Resolve paths relative to this script file
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # .../scan2measure-webframework

# Default Directories
input_pcd_dir = project_root / 'data' / 'raw_point_cloud'
pano_dir = project_root / 'data' / 'pano' / 'raw'
alignment_dir = project_root / 'data' / 'reconstructed_floorplans_RoomFormer'
output_dir = project_root / 'data' / 'textured_point_cloud'

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================

def find_floor_z(pcd, bin_size=0.05):
    """
    Analyzes the Z-histogram of the point cloud to find the floor height.
    Assumes the floor is the lowest significant geometric mode.
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("Point cloud is empty.")

    z_values = points[:, 2]
    
    # Create Histogram
    z_min, z_max = np.min(z_values), np.max(z_values)
    bins = np.arange(z_min, z_max, bin_size)
    hist, edges = np.histogram(z_values, bins=bins)
    
    # Look for the peak in the lowest 30% of the Z-range (floor assumption)
    lower_threshold_idx = int(len(hist) * 0.30)
    
    # Safety check if cloud is very flat
    if lower_threshold_idx == 0: 
        lower_threshold_idx = len(hist)

    peak_idx = np.argmax(hist[:lower_threshold_idx])
    
    # The floor Z is the center of that bin
    floor_z = edges[peak_idx] + (bin_size / 2.0)
    
    print(f"  > Vertical Alignment: Floor Z detected at {floor_z:.3f} (Peak count: {hist[peak_idx]})")
    return floor_z

def project_texture(pcd, pano_img, cam_pose_global, rotation_deg):
    """
    Projects 3D points to Equirectangular UV coordinates and samples colors.
    """
    points = np.asarray(pcd.points)
    cx, cy, cz = cam_pose_global
    
    # 1. Translate Points to Camera Local Frame (Global -> Local Translation)
    # Vector from Camera -> Point
    vecs = points - np.array([cx, cy, cz])
    
    # 2. Undo Global Rotation (Global -> Local Rotation)
    # The room was rotated by 'rotation_deg' to fit the map.
    # We must rotate the vectors by -rotation_deg to match the original Pano view.
    theta_rad = np.radians(-rotation_deg) 
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_inv = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    # Rotate the vectors
    local_vecs = vecs @ R_inv.T
    
    # 3. Cartesian -> Spherical (UV)
    # Standard Math: X=Right, Y=Forward, Z=Up
    x, y, z = local_vecs[:, 0], local_vecs[:, 1], local_vecs[:, 2]
    
    # Longitude (Theta): -pi to pi. 
    # atan2(x, y) assumes Y is forward (0 deg).
    lon = np.arctan2(x, y) 
    
    # Latitude (Phi): -pi/2 to pi/2
    hypot = np.hypot(x, y)
    lat = np.arctan2(z, hypot)
    
    # 4. Spherical -> UV Image Coordinates [0, 1]
    # U: Maps -pi to 0, pi to 1
    u = (lon / (2 * np.pi)) + 0.5
    
    # V: Maps pi/2 (top) to 0, -pi/2 (bottom) to 1
    v = 0.5 - (lat / np.pi) 
    
    # 5. Sample Image Pixels
    h_img, w_img = pano_img.shape[:2]
    
    u_px = (u * w_img).astype(int) % w_img
    v_px = (v * h_img).astype(int)
    
    # Clamp V to be safe
    v_px = np.clip(v_px, 0, h_img - 1)
    
    # Sample Colors (OpenCV is BGR, we need RGB for Open3D)
    colors = pano_img[v_px, u_px][:, ::-1]
    
    # Normalize to 0.0 - 1.0 for Open3D
    return colors.astype(np.float64) / 255.0

# ==========================================
# 3. MAIN
# ==========================================
def main():
    # --- USER CONFIGURATION ---
    # The Point Cloud file (Single Room Subset)
    pcd_filename = "Area_3_study_no_RGB.ply"
    
    # The Folder containing 'global_alignment.json' (usually matches PCD stem)
    # Note: As per your prompt, the json is in a folder named after the PCD
    alignment_folder_name = "Area_3_study_no_RGB"
    
    # The Panorama Image
    pano_filename = "Area3_study.jpg"
    
    # The specific Room Name Key inside global_alignment.json to look for
    target_room_name = "Area3_study"
    
    # LGT-Net Assumed Height (Offset from detected floor)
    LGT_CAMERA_HEIGHT = 1.6 
    # ---------------------------

    print(f"--- Texture Mapping (Single Room) ---")
    
    # 1. Path Construction
    pcd_path = input_pcd_dir / pcd_filename
    pano_path = pano_dir / pano_filename
    json_path = alignment_dir / alignment_folder_name / "global_alignment.json"

    # Validation
    if not pcd_path.exists(): sys.exit(f"Error: Point cloud not found at {pcd_path}")
    if not pano_path.exists(): sys.exit(f"Error: Panorama not found at {pano_path}")
    if not json_path.exists(): sys.exit(f"Error: Alignment JSON not found at {json_path}")

    # 2. Load Data
    print(f"Loading Point Cloud: {pcd_path.name}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    
    print(f"Loading Alignment Data...")
    with open(json_path, 'r') as f:
        alignment_data = json.load(f)

    # Find the specific room in the JSON
    room_info = None
    for r in alignment_data["alignment_results"]:
        if r["room_name"] == target_room_name:
            room_info = r
            break
    
    if room_info is None:
        sys.exit(f"Error: Room '{target_room_name}' not found in global_alignment.json")

    # 3. Calculate Vertical Pose (Solve Z)
    # We ignore the Z in the JSON (if any) and measure it from the cloud.
    floor_z = find_floor_z(pcd)
    cam_z_global = floor_z + LGT_CAMERA_HEIGHT
    
    # 4. Construct Full 3D Pose
    # JSON has [x, y], we append our calculated z
    pose_2d = room_info["camera_pose_global"]
    final_pose_3d = np.array([pose_2d[0], pose_2d[1], cam_z_global])
    
    rotation_deg = room_info["transformation"]["rotation_deg"]
    
    print(f"  > Camera Pose (Global): [X={final_pose_3d[0]:.2f}, Y={final_pose_3d[1]:.2f}, Z={final_pose_3d[2]:.2f}]")
    print(f"  > Rotation Correction: {rotation_deg:.2f} degrees")

    # 5. Load Panorama
    print(f"Loading Panorama: {pano_path.name}")
    pano_img = cv2.imread(str(pano_path))
    if pano_img is None:
        sys.exit("Error: Failed to read image file.")

    # 6. Project Texture
    # (No filtering step here, assuming input is already the room subset)
    print("Projecting texture onto point cloud...")
    colors = project_texture(pcd, pano_img, final_pose_3d, rotation_deg)
    
    # 7. Save Result
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    output_filename = f"{pcd_path.stem}_textured.ply"
    output_path = output_dir / output_filename
    
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Done! Textured point cloud saved to:\n{output_path}")

if __name__ == "__main__":
    main()