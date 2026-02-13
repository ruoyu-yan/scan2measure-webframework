import os
import sys
import numpy as np
import open3d as o3d
import cv2
import json
from pathlib import Path

# -------------------------------------------------------------------------
# 0. PATH SETUP
# -------------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # .../scan2measure-webframework

# Input Directory: .../data/raw_point_cloud
input_dir = project_root / "data" / "raw_point_cloud"

# Output Directory: .../data/density_image
output_dir = project_root / "data" / "density_image"
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 1. VERTICAL ALIGNMENT (Floor Flat on XY Plane)
# -------------------------------------------------------------------------
def align_to_floor(pcd):
    print("  > Detecting dominant plane (floor)...")
    
    # RANSAC to find the largest plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model

    # Rotate plane normal to align with Z+
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
    R_total = R.copy()

    # Upside Down Check
    points = np.asarray(pcd.points)
    floor_z = np.median(points[inliers, 2])
    room_z = np.median(np.delete(points, inliers, axis=0)[:, 2])

    if room_z < floor_z:
        print("    Detected ceiling as plane. Flipping 180 degrees...")
        R_flip = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        pcd.rotate(R_flip, center=(0, 0, 0))
        R_total = R_flip @ R_total
    
    return pcd, R_total

# -------------------------------------------------------------------------
# 2. HORIZONTAL ALIGNMENT (Walls Parallel to X/Y Axes)
# -------------------------------------------------------------------------
def align_to_axes(pcd):
    print("  > Aligning walls to X/Y axes...")

    # Estimate Normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)
    
    # Filter for Vertical Walls (Z component near 0)
    is_wall = np.abs(normals[:, 2]) < 0.2
    wall_normals = normals[is_wall]

    if len(wall_normals) == 0:
        print("    Warning: No vertical walls detected. Skipping horizontal alignment.")
        return pcd, np.eye(3)

    # Calculate angles in 2D
    angles = np.arctan2(wall_normals[:, 1], wall_normals[:, 0])
    
    # Histogram analysis to find dominant direction
    hist, edges = np.histogram(angles, bins=360, range=(-np.pi, np.pi))
    peak_idx = np.argmax(hist)
    dominant_angle = (edges[peak_idx] + edges[peak_idx+1]) / 2.0
    
    print(f"    Dominant wall angle detected: {np.degrees(dominant_angle):.2f} degrees")

    # Rotate to align dominant angle with X-axis
    R_yaw = pcd.get_rotation_matrix_from_xyz((0, 0, -dominant_angle))
    pcd.rotate(R_yaw, center=(0,0,0))
    
    return pcd, R_yaw

# -------------------------------------------------------------------------
# 3. DENSITY MAP GENERATION (Proportional)
# -------------------------------------------------------------------------
def generate_density(point_cloud, width=256, height=256):
    # Structured3D utils flips axes before projecting. We keep this consistency.
    ps = point_cloud.copy() * -1
    ps[:, 0] *= -1
    ps[:, 1] *= -1

    image_res = np.array((width, height))

    # Calculate physical bounds and padding
    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)
    physical_dims = max_coords - min_coords
    
    padding = 0.1 * physical_dims
    max_coords = max_coords + padding
    min_coords = min_coords - padding

    # Uniform Scaling Logic
    padded_dims = max_coords[:2] - min_coords[:2]
    max_dim = np.max(padded_dims)
    offset = (max_dim - padded_dims) / 2.0

    # Project
    coordinates = (ps[:, :2] - min_coords[None, :2] + offset[None, :]) / max_dim * (image_res[None] - 1)
    coordinates = np.round(coordinates)
    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)), image_res - 1)

    # Fill Density
    density = np.zeros((height, width), dtype=np.float32)
    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    unique_coordinates = unique_coordinates.astype(np.int32)
    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
    
    if np.max(density) > 0:
        density = density / np.max(density)

    # The effective translation that maps a rotated point to pixel space is:
    #   pixel = (p[:2] - min_coords[:2] + offset) / max_dim * (image_res - 1)
    # So the shift from the rotated-cloud origin is  -min_coords[:2] + offset
    # We store the full 3-component translation used to shift the cloud.
    translation = np.zeros(3)
    translation[:2] = -min_coords[:2] + offset
    translation[2] = -min_coords[2]

    metadata = {
        "min_coords": min_coords.tolist(),
        "max_dim": float(max_dim),
        "offset": offset.tolist(),
        "image_width": int(width),
        "image_height": int(height),
        "translation": translation.tolist(),
    }

    return density, metadata

# -------------------------------------------------------------------------
# 4. MAIN EXECUTION
# -------------------------------------------------------------------------
def main():
    # --- CONFIGURATION ---
    FILENAME = "tmb_office_one_corridor_dense.ply"
    # ---------------------

    input_path = input_dir / FILENAME
    
    if not input_path.exists():
        print(f"Error: File not found at: {input_path}")
        sys.exit(1)

    print(f"--- Processing {FILENAME} ---")

    # 1. Load
    pcd = o3d.io.read_point_cloud(str(input_path))
    if pcd.is_empty(): sys.exit(1)
    
    # 2. Align (Z and XY) — track cumulative rotation
    R_global = np.eye(3)

    pcd, R_floor = align_to_floor(pcd)
    R_global = R_floor @ R_global

    pcd, R_yaw = align_to_axes(pcd)
    R_global = R_yaw @ R_global

    # 3. Units & Quantization
    points = np.asarray(pcd.points)
    extents = np.max(points, axis=0) - np.min(points, axis=0)
    if np.max(extents) < 500: points *= 1000.0

    # Initialize unique_coords with raw points (Default)
    unique_coords = points 

    # --- QUANTIZATION (Comment out to disable) ---
    print("  > Applying quantization...")
    points[:,:2] = np.round(points[:,:2] / 10) * 10.
    points[:,2] = np.round(points[:,2] / 100) * 100.
    unique_coords = np.unique(points, axis=0)
    # ---------------------------------------------
    
    # 4. Generate Density Image
    print("  > Generating proportional density map...")
    density_map, metadata = generate_density(unique_coords)

    # Embed the accumulated rotation matrix so downstream tools can
    # replicate the exact coordinate system from the raw point cloud.
    metadata["rotation_matrix"] = R_global.tolist()
    
    density_img_vis = (density_map * 255).astype(np.uint8)
    
    # 5. Save into a per-pointcloud folder
    out_folder = output_dir / input_path.stem
    out_folder.mkdir(parents=True, exist_ok=True)

    image_path = out_folder / (input_path.stem + ".png")
    metadata_path = out_folder / "metadata.json"

    cv2.imwrite(str(image_path), density_img_vis)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done! Saved to: {out_folder}")

if __name__ == "__main__":
    main()