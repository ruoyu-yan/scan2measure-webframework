import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

# ================= CONFIGURATION =================
# Use absolute paths or paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Go up one level from src

INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "pano", "raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "pano", "processed")

# Output Image Settings
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
# =================================================

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
    # FIX: We negate y_c because in images, Y grows downwards, 
    # but in 3D space (and latitude), Y grows upwards.
    x_c = (x - cx) / f
    y_c = -(y - cy) / f  # <--- This fixes the Upside Down issue
    z_c = np.ones_like(x_c)
    
    # Stack to form 3D vectors (x, y, z)
    xyz = np.stack((x_c, y_c, z_c), axis=-1)
    
    # Normalize vectors
    norm = np.linalg.norm(xyz, axis=2, keepdims=True)
    xyz = xyz / norm

    # Rotation Matrices
    theta_rad = np.radians(theta)
    # FIX: Negate pitch because standard rotation matrices rotate the axis, not the vector.
    # To "look up" (positive pitch), we effectively rotate the world down.
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

def process_panoramas():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return
        
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    input_files = []
    for ext in image_extensions:
        input_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not input_files:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(input_files)} panoramas to process.")

    for file_path in input_files:
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Create subfolder in output
        save_folder = os.path.join(OUTPUT_DIR, name_no_ext)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        print(f"\nProcessing: {filename} -> {save_folder}")
        
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error reading {file_path}, skipping.")
            continue

        h_pano, w_pano = img.shape[:2]

        for idx, (yaw, pitch) in enumerate(tqdm(VIEWS, desc="Generating Views")):
            
            lon, lat = get_perspective_map(FOV, yaw, pitch, OUTPUT_SIZE[0], OUTPUT_SIZE[1])

            # Map lon/lat to image u,v
            u = (lon / (2 * np.pi) + 0.5) * w_pano
            v = (-lat / np.pi + 0.5) * h_pano 

            map_x = u.astype(np.float32)
            map_y = v.astype(np.float32)

            rect_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

            save_name = f"{idx:02d}_yaw{yaw}_pitch{pitch}.jpg"
            cv2.imwrite(os.path.join(save_folder, save_name), rect_img)

    print("\n--- Processing Complete! ---")

if __name__ == "__main__":
    process_panoramas()