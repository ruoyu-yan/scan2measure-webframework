import os
import sys
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path

# -------------------------------------------------------------------------
# 0. PATH SETUP
# -------------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # .../scan2measure-webframework

# Input Directory: .../data/point_cloud
input_dir = project_root / "data" / "point_cloud"

# Output Directory: .../data/density_image
output_dir = project_root / "data" / "density_image"
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 1. ALIGNMENT LOGIC (Right Side Up)
# -------------------------------------------------------------------------
def align_to_floor(pcd):
    print("  > Detecting dominant plane (floor)...")
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model

    # 1. Rotate plane normal [a, b, c] to align with Z+ [0, 0, 1]
    normal = np.array([a, b, c])
    target = np.array([0, 0, 1])
    
    v = np.cross(normal, target)
    c_ang = np.dot(normal, target)
    s = np.linalg.norm(v)
    
    if s < 1e-6:
        R = np.eye(3)
    else:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c_ang) / (s ** 2))

    pcd.rotate(R, center=(0, 0, 0))

    # 2. Upside Down Check
    points = np.asarray(pcd.points)
    floor_z = np.median(points[inliers, 2])
    room_z = np.median(np.delete(points, inliers, axis=0)[:, 2])

    if room_z < floor_z:
        print("    Detected ceiling as plane (room is below). Flipping 180 degrees...")
        R_flip = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        pcd.rotate(R_flip, center=(0, 0, 0))
    
    return pcd

# -------------------------------------------------------------------------
# 2. DENSITY MAP GENERATION (FIXED PROPORTIONS)
# -------------------------------------------------------------------------
def generate_density(point_cloud, width=256, height=256):
    """
    Modified implementation to maintain correct aspect ratio.
    """
    # Structured3D utils flips axes before projecting. We keep this consistency.
    ps = point_cloud.copy() * -1
    ps[:, 0] *= -1
    ps[:, 1] *= -1

    image_res = np.array((width, height))

    # 1. Calculate physical bounds
    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)
    physical_dims = max_coords - min_coords

    # 2. Add 10% padding relative to the dimensions
    padding = 0.1 * physical_dims
    max_coords = max_coords + padding
    min_coords = min_coords - padding

    # --- FIX STARTS HERE ---
    
    # Recalculate padded dimensions (only needed for X and Y)
    padded_dims = max_coords[:2] - min_coords[:2]

    # Find the single largest dimension (length or width)
    max_dim = np.max(padded_dims)

    # Calculate offset to center the smaller dimension within the square canvas
    # If X is smaller than Y, shift X over so it's centered.
    offset = (max_dim - padded_dims) / 2.0

    # Project to 2D coordinates using UNIFORM scaling
    # Note: We divide both X and Y by the SAME `max_dim` scalar.
    coordinates = (ps[:, :2] - min_coords[None, :2] + offset[None, :]) / max_dim * (image_res[None] - 1)
    
    # --- FIX ENDS HERE ---

    coordinates = np.round(coordinates)
    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                                image_res - 1)

    density = np.zeros((height, width), dtype=np.float32)

    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    unique_coordinates = unique_coordinates.astype(np.int32)

    # Fill density map (note indices: y, x)
    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
    
    # Normalize
    if np.max(density) > 0:
        density = density / np.max(density)

    return density

# -------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -------------------------------------------------------------------------
def main():
    # --- CONFIGURATION ---
    FILENAME = "Area_3_4_rooms.ply"
    VIZ = False
    # ---------------------

    input_path = input_dir / FILENAME
    
    if not input_path.exists():
        print(f"Error: File not found at: {input_path}")
        sys.exit(1)

    print(f"--- Processing {FILENAME} ---")

    # 1. Load
    pcd = o3d.io.read_point_cloud(str(input_path))
    if pcd.is_empty(): sys.exit(1)
    
    # 2. Align
    pcd = align_to_floor(pcd)
    if VIZ: o3d.visualization.draw_geometries([pcd])

    # 3. Units & Quantization
    points = np.asarray(pcd.points)
    extents = np.max(points, axis=0) - np.min(points, axis=0)
    if np.max(extents) < 500: points *= 1000.0

    print("  > Applying quantization...")
    points[:,:2] = np.round(points[:,:2] / 10) * 10.
    points[:,2] = np.round(points[:,2] / 100) * 100.
    unique_coords = np.unique(points, axis=0)
    
    # 4. Generate Density Image
    print("  > Generating proportional density map...")
    density_map = generate_density(unique_coords)
    
    density_img_vis = (density_map * 255).astype(np.uint8)
    
    # 5. Save
    out_name = input_path.stem + ".png"
    save_path = output_dir / out_name
    cv2.imwrite(str(save_path), density_img_vis)
    print(f"Done! Saved to: {save_path.name}")

if __name__ == "__main__":
    main()