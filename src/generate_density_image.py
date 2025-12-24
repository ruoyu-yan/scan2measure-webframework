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
    
    # RANSAC to find the largest plane (assumed to be floor or ceiling)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model

    # 1. Rotate plane normal [a, b, c] to align with Z+ [0, 0, 1]
    normal = np.array([a, b, c])
    target = np.array([0, 0, 1])
    
    # Formula to get rotation matrix between two vectors
    v = np.cross(normal, target)
    c_ang = np.dot(normal, target)
    s = np.linalg.norm(v)
    
    # If already aligned, skip
    if s < 1e-6:
        R = np.eye(3)
    else:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c_ang) / (s ** 2))

    pcd.rotate(R, center=(0, 0, 0))

    # 2. Upside Down Check
    # Check if the majority of points are ABOVE (floor) or BELOW (ceiling) the plane.
    points = np.asarray(pcd.points)
    floor_z = np.median(points[inliers, 2])
    # Get median Z of the rest of the room
    room_z = np.median(np.delete(points, inliers, axis=0)[:, 2])

    if room_z < floor_z:
        print("    Detected ceiling as plane (room is below). Flipping 180 degrees...")
        # Flip around X axis to invert Z
        R_flip = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        pcd.rotate(R_flip, center=(0, 0, 0))
    
    return pcd

# -------------------------------------------------------------------------
# 2. DENSITY MAP GENERATION (Matching stru3d_utils.py)
# -------------------------------------------------------------------------
def generate_density(point_cloud, width=256, height=256):
    """
    Exact implementation from RoomFormer's data preprocessing
    """
    # structured3d utils flips z before projecting
    ps = point_cloud * -1
    ps[:,0] *= -1
    ps[:,1] *= -1

    image_res = np.array((width, height))

    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)
    max_m_min = max_coords - min_coords

    # Add 10% padding
    max_coords = max_coords + 0.1 * max_m_min
    min_coords = min_coords - 0.1 * max_m_min

    # Project to 2D image coordinates
    coordinates = np.round(
            (ps[:, :2] - min_coords[None, :2]) / (max_coords[None,:2] - min_coords[None, :2]) * image_res[None])
    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                                image_res - 1)

    density = np.zeros((height, width), dtype=np.float32)

    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)

    unique_coordinates = unique_coordinates.astype(np.int32)

    # Fill density map
    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
    
    # Normalize
    if np.max(density) > 0:
        density = density / np.max(density)

    return density

# -------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -------------------------------------------------------------------------
def main():
    # --- CONFIGURATION: CHANGE THESE VALUES ---
    FILENAME = "Area_3_multiple_rooms_cleaned.ply"  # Change this to your file name
    VIZ = True                              # Set to False to disable visualization
    # ------------------------------------------

    input_path = input_dir / FILENAME
    
    if not input_path.exists():
        print(f"Error: File not found at:")
        print(f"  {input_path}")
        print("Please check the FILENAME variable in the script.")
        sys.exit(1)

    print(f"--- Processing {FILENAME} ---")
    print(f"Source: {input_dir}")
    print(f"Target: {output_dir}")

    # 1. Load
    pcd = o3d.io.read_point_cloud(str(input_path))
    if pcd.is_empty():
        print("Error: Point cloud is empty or format not supported.")
        sys.exit(1)
    
    # 2. Align (Right Side Up)
    pcd = align_to_floor(pcd)
    
    if VIZ:
        print("Displaying aligned point cloud (Close window to continue)...")
        o3d.visualization.draw_geometries([pcd])

    # 3. Convert to Numpy & Handle Units
    points = np.asarray(pcd.points)
    
    # Heuristic: If coordinate extents are small (e.g., < 500), it's likely Meters.
    extents = np.max(points, axis=0) - np.min(points, axis=0)
    if np.max(extents) < 500: 
        print("  > Detected unit: Meters (likely). Converting to Millimeters...")
        points *= 1000.0
    else:
        print("  > Detected unit: Millimeters (likely). Keeping scale.")

    # 4. Quantization (Crucial Step for RoomFormer)
    print("  > Applying quantization...")
    points[:,:2] = np.round(points[:,:2] / 10) * 10.   # 1cm horiz resolution
    points[:,2] = np.round(points[:,2] / 100) * 100.   # 10cm vert resolution
    unique_coords = np.unique(points, axis=0)
    
    # 5. Generate Density Image
    print("  > Generating density map...")
    density_map = generate_density(unique_coords)
    
    # Convert to 8-bit image
    density_img_vis = (density_map * 255).astype(np.uint8)
    
    # 6. Save
    out_name = input_path.stem + ".png"
    save_path = output_dir / out_name
    
    cv2.imwrite(str(save_path), density_img_vis)
    print(f"Done! Saved density map to:\n  {save_path}")

if __name__ == "__main__":
    main()