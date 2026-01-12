import os
import sys
import json
import numpy as np
import open3d as o3d
import matplotlib.path as mplPath
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path

# -------------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

pcd_dir = project_root / "data" / "raw_point_cloud"
density_dir = project_root / "data" / "density_image"
recon_dir = project_root / "data" / "reconstructed_floorplans_RoomFormer"
output_dir = project_root / "data" / "segmented_point_cloud"
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# ALIGNMENT
# -------------------------------------------------------------------------
def align_to_floor(pcd):
    print("  > Re-applying vertical alignment...")
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model

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

    points = np.asarray(pcd.points)
    floor_z = np.median(points[inliers, 2])
    room_z = np.median(np.delete(points, inliers, axis=0)[:, 2])

    if room_z < floor_z:
        print("    Detected ceiling. Flipping 180 degrees...")
        R_flip = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        pcd.rotate(R_flip, center=(0, 0, 0))
    
    return pcd

def align_to_axes(pcd):
    print("  > Re-applying horizontal alignment...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)
    is_wall = np.abs(normals[:, 2]) < 0.2
    wall_normals = normals[is_wall]

    if len(wall_normals) == 0:
        return pcd

    angles = np.arctan2(wall_normals[:, 1], wall_normals[:, 0])
    hist, edges = np.histogram(angles, bins=360, range=(-np.pi, np.pi))
    peak_idx = np.argmax(hist)
    dominant_angle = (edges[peak_idx] + edges[peak_idx+1]) / 2.0
    
    print(f"    Dominant angle: {np.degrees(dominant_angle):.2f} deg")
    R_yaw = pcd.get_rotation_matrix_from_xyz((0, 0, -dominant_angle))
    pcd.rotate(R_yaw, center=(0,0,0))
    
    return pcd

# -------------------------------------------------------------------------
# INVERSE PROJECTION
# -------------------------------------------------------------------------
def pixels_to_coords(pixel_poly, metadata):
    min_coords = np.array(metadata['min_coords'])
    offset = np.array(metadata['offset'])
    max_dim = metadata['max_dim']
    width = metadata['image_width']
    
    scale_factor = (width - 1)
    min_xy = min_coords[:2]
    off_xy = offset[:2]
    
    world_poly = []
    for (u, v) in pixel_poly:
        pixel_vec = np.array([u, v])
        world_xy = (pixel_vec / scale_factor * max_dim) - off_xy + min_xy
        world_poly.append(world_xy)
        
    return np.array(world_poly)

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    FILENAME = "Area_3_selected_rooms_no_RGB.ply" 
    
    input_pcd_path = pcd_dir / FILENAME
    stem_name = input_pcd_path.stem
    
    metadata_path = density_dir / stem_name / "metadata.json"
    predictions_path = recon_dir / stem_name / "predictions.json"

    if not input_pcd_path.exists(): sys.exit(f"Error: {input_pcd_path} missing")
    if not metadata_path.exists(): sys.exit(f"Error: {metadata_path} missing")
    if not predictions_path.exists(): sys.exit(f"Error: {predictions_path} missing")

    print(f"--- Debugging Segmentation for {FILENAME} ---")

    # 1. Load Data
    pcd = o3d.io.read_point_cloud(str(input_pcd_path))
    with open(metadata_path, 'r') as f: metadata = json.load(f)
    with open(predictions_path, 'r') as f: predictions = json.load(f)

    # 2. Re-Align (Required so the geometry matches the room orientation)
    pcd = align_to_floor(pcd)
    pcd = align_to_axes(pcd)
    
    # 3. PREPARE POINTS (Decoupled Logic)
    # We get a reference to the points in the PCD
    points_ref = np.asarray(pcd.points)
    
    # We create a COPY for the math check. 
    # This allows us to scale the copy to Millimeters while keeping 'pcd' in Meters.
    points_check = points_ref.copy() 

    # --- SCALING LOGIC (Apply only to the copy) ---
    extents = np.max(points_check, axis=0) - np.min(points_check, axis=0)
    
    if np.max(extents) < 500:
        print(f"  > Detected Meter scale (Max: {np.max(extents):.2f}). Scaling check-copy to Millimeters...")
        points_check *= 1000.0
    else:
        print(f"  > Detected Millimeter scale. No scaling needed.")
    # ----------------------------------------------

    # Use the scaled copy for XY checks
    points_xy_check = points_check[:, :2] 
    
    # Debugging Bounds
    p_min = np.min(points_check, axis=0)
    p_max = np.max(points_check, axis=0)
    print(f"\n[DEBUG] Check-Cloud Bounds (Millimeters):")
    print(f"  Min: {p_min[:2]}")
    print(f"  Max: {p_max[:2]}")
    
    # 4. COLORING
    # We apply colors to an array that matches the length of the original PCD
    pcd_colors = np.full(points_ref.shape, 0.7) 
    cmap = mcolors.ListedColormap(cm.get_cmap('tab20').colors)

    print(f"\n[DEBUG] Checking Intersection:")
    total_colored = 0

    for i, room_pixel_poly in enumerate(predictions):
        # The room polygons are ALREADY in Millimeters (from metadata)
        room_world_poly = pixels_to_coords(room_pixel_poly, metadata)
        
        path = mplPath.Path(room_world_poly)
        
        # We check if the Millimeter-scaled points are inside the Millimeter polygon
        mask = path.contains_points(points_xy_check)
        count = np.sum(mask)
        
        if count > 0:
            print(f"    > Room {i}: Matched {count} points")
            color = cmap(i % 20)[:3] 
            pcd_colors[mask] = color
            total_colored += count

    # 5. Save
    # We assign the colors to the ORIGINAL pcd object. 
    # Since we never modified 'points_ref' or 'pcd.points', the geometry is still in METERS.
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    
    out_name = f"{stem_name}_segmented.ply"
    out_path = output_dir / out_name
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"\nDone. Saved to {out_path}. Total points colored: {total_colored}")

if __name__ == "__main__":
    main()