import os
import cv2
import numpy as np
import sys
from tqdm import tqdm

# ==========================================
# 1. GLOBAL CONSTANTS (Core Algorithm Settings)
# ==========================================
OUTPUT_SIZE = (1024, 1024)  
FOV = 60                    

# The 14-Photo Layout Definition: (Yaw, Pitch) in degrees
VIEWS = [
    # --- Middle Ring (8 photos) ---
    (0, 0), (45, 0), (90, 0), (135, 0), (180, 0), (225, 0), (270, 0), (315, 0),
    
    # --- Top Ring (4 photos, looking UP) ---
    (0, 60), (90, 60), (180, 60), (270, 60),
    
    # --- Bottom Ring (2 photos, looking DOWN) ---
    (0, -90), (90, -90)
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_perspective_map(fov, theta, phi, height, width):
    """
    Generates the X and Y mapping matrices for cv2.remap to convert 
    Equirectangular to Rectilinear.
    """
    f = 0.5 * width / np.tan(0.5 * fov * np.pi / 180.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    # Create a grid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert to normalized camera coordinates
    x_c = (x - cx) / f
    y_c = -(y - cy) / f 
    z_c = np.ones_like(x_c)
    
    # Stack to form 3D vectors (x, y, z)
    xyz = np.stack((x_c, y_c, z_c), axis=-1)
    
    # Normalize vectors
    norm = np.linalg.norm(xyz, axis=2, keepdims=True)
    xyz = xyz / norm

    # Rotation Matrices
    theta_rad = np.radians(theta)
    phi_rad = np.radians(-phi) 

    # Rotation around X-axis (Pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi_rad), -np.sin(phi_rad)],
        [0, np.sin(phi_rad), np.cos(phi_rad)]
    ])
    
    # Rotation around Y-axis (Yaw)
    Ry = np.array([
        [np.cos(theta_rad), 0, np.sin(theta_rad)],
        [0, 1, 0],
        [-np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])

    # Combined Rotation: Yaw then Pitch
    R = Ry @ Rx
    
    # Rotate the rays
    h, w, _ = xyz.shape
    xyz_flat = xyz.reshape(-1, 3)
    xyz_rotated = xyz_flat @ R.T
    xyz_final = xyz_rotated.reshape(h, w, 3)
    
    # Convert 3D (x, y, z) back to Spherical (lon, lat)
    x_f = xyz_final[:, :, 0]
    y_f = xyz_final[:, :, 1]
    z_f = xyz_final[:, :, 2]

    # Calculate longitude (theta) and latitude (phi)
    lon = np.arctan2(x_f, z_f)
    lat = np.arcsin(y_f)

    return lon, lat

def process_panorama_views(input_path, output_folder):
    """
    Loads a single panorama and generates the 14 virtual camera views.
    """
    # 1. Load Image
    img = cv2.imread(input_path)
    if img is None:
        print(f"[Error] Could not read image: {input_path}")
        return

    h_pano, w_pano = img.shape[:2]
    print(f"Generating {len(VIEWS)} virtual camera views...")
    
    # 2. Process Views
    for idx, (yaw, pitch) in enumerate(tqdm(VIEWS, desc="Processing")):
        
        # Core Mapping Logic
        lon, lat = get_perspective_map(FOV, yaw, pitch, OUTPUT_SIZE[0], OUTPUT_SIZE[1])

        # Map lon/lat to image u,v
        u = (lon / (2 * np.pi) + 0.5) * w_pano
        v = (-lat / np.pi + 0.5) * h_pano 

        map_x = u.astype(np.float32)
        map_y = v.astype(np.float32)

        # Remap
        rect_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        # Save
        save_name = f"{idx:02d}_yaw{yaw}_pitch{pitch}.jpg"
        cv2.imwrite(os.path.join(output_folder, save_name), rect_img)

    print(f"\n[Success] Processed images saved to: {output_folder}")

# ==========================================
# 3. MAIN FUNCTION
# ==========================================
def main():
    # --- USER CONFIGURATION ---
    target_image_name = "corridor2.jpg"
    
    # --- DIRECTORY SETUP ---
    # Matches LGT-Net logic: navigate relative to the script file
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) # Go up from 'src' to project root

    # Input: .../data/pano/input
    input_dir = os.path.join(project_root, "data", "pano", "input")
    input_image_path = os.path.join(input_dir, target_image_name)
    
    # Output: .../data/pano/virtual_camera_processed/{image_name}
    base_output_dir = os.path.join(project_root, "data", "pano", "virtual_camera_processed")
    image_stem = os.path.splitext(target_image_name)[0]
    output_folder = os.path.join(base_output_dir, image_stem)

    # --- VALIDATION & SETUP ---
    if not os.path.exists(input_image_path):
        print(f"[Error] Input file not found: {input_image_path}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    else:
        print(f"Output directory exists: {output_folder}")

    print(f"Processing: {input_image_path}")
    print(f"Output to:  {output_folder}")

    # --- EXECUTE PROCESSING ---
    process_panorama_views(input_image_path, output_folder)

if __name__ == "__main__":
    main()