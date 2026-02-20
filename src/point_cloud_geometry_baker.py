import numpy as np
import open3d as o3d
import json
import pickle
import random
from pathlib import Path

from scipy.spatial import ConvexHull
import alphashape
from shapely.geometry import Polygon, MultiPolygon

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# ==========================================
# 2. CORE EXTRACTION FUNCTION
# ==========================================
def extract_3d_wireframe(points, output_dir, distance_threshold=0.05, ransac_n=3, num_iterations=1000, 
                         min_remaining_points=2000, dbscan_eps=0.20, dbscan_min_samples=30, 
                         min_hull_area=0.05, parallel_threshold=0.9, alpha_shape_alpha=2.0):
    """
    Extracts a 3D wireframe using Hybrid RANSAC + DBSCAN approach.
    """
    print(f"\n[Wireframe Extraction] Starting Hybrid RANSAC + DBSCAN plane segmentation...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    detected_planes = []
    remaining_pcd = pcd
    debug_colored_pcds = []
    
    iteration = 0
    total_clusters_accepted = 0
    
    while len(remaining_pcd.points) > min_remaining_points:
        iteration += 1
        print(f"\n  [Iteration {iteration}] Remaining points: {len(remaining_pcd.points)}")
        
        plane_model, inlier_indices = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        if len(inlier_indices) < dbscan_min_samples:
            print(f"  [Info] RANSAC found only {len(inlier_indices)} inliers, stopping")
            break
        
        inlier_pcd = remaining_pcd.select_by_index(inlier_indices)
        inlier_points = np.asarray(inlier_pcd.points)
        normal = np.array(plane_model[:3])
        
        print(f"  RANSAC Plane: {len(inlier_indices)} inliers, Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
        
        labels = np.array(inlier_pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_samples, print_progress=True))
        cluster_labels = labels
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)
        
        print(f"  DBSCAN found {len(unique_labels)} potential clusters (excluding noise)")
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_points = inlier_points[cluster_mask]
            
            if len(cluster_points) < 4:
                continue
            
            normal_normalized = normal / (np.linalg.norm(normal) + 1e-8)
            z_axis = np.array([0, 0, 1])
            dot = np.dot(normal_normalized, z_axis)
            
            if abs(dot) > 0.9999:
                if dot > 0:
                    R = np.eye(3)
                else:
                    R = np.diag([1, 1, -1])
            else:
                v = np.cross(normal_normalized, z_axis)
                s = np.linalg.norm(v)
                c = dot
                vx = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-8))
            
            rotated_points = cluster_points @ R.T
            points_2d = rotated_points[:, :2]
            
            try:
                hull = ConvexHull(points_2d)
                hull_area = hull.volume 
            except Exception:
                continue
            
            if hull_area < min_hull_area:
                print(f"    Cluster {label}: {len(cluster_points)} pts, Area={hull_area:.4f} m² - REJECTED (too small)")
                continue
            
            print(f"    Cluster {label}: {len(cluster_points)} pts, Area={hull_area:.4f} m² - ACCEPTED")
            
            try:
                alpha_shape = alphashape.alphashape(points_2d, alpha_shape_alpha)
                
                if isinstance(alpha_shape, MultiPolygon):
                    alpha_shape = max(alpha_shape.geoms, key=lambda p: p.area)
                elif not isinstance(alpha_shape, Polygon):
                    from shapely.geometry import MultiPoint
                    alpha_shape = MultiPoint(points_2d).convex_hull
                    if not isinstance(alpha_shape, Polygon):
                        print(f"      [Warning] Could not generate valid polygon for cluster {label}, skipping")
                        continue
                
                simplified_polygon = alpha_shape.simplify(0.05, preserve_topology=True)
                
                if not isinstance(simplified_polygon, Polygon) or simplified_polygon.is_empty:
                    simplified_polygon = alpha_shape 
                    
            except Exception as e:
                print(f"      [Warning] Alpha shape generation failed for cluster {label}: {e}")
                continue
            
            cluster_color = [random.random(), random.random(), random.random()]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
            cluster_pcd.paint_uniform_color(cluster_color)
            debug_colored_pcds.append(cluster_pcd)
            
            cluster_centroid = np.mean(cluster_points, axis=0)
            d = -np.dot(normal, cluster_centroid)
            cluster_plane_eq = np.array([normal[0], normal[1], normal[2], d])
            
            plane_data = {
                'plane_model': cluster_plane_eq,           
                'points_3d': cluster_points,                
                'shapely_2d_polygon': simplified_polygon,   
                'rotation_matrix': R,                       
                'centroid': cluster_centroid,               
                'normal': normal_normalized,                
            }
            
            detected_planes.append(plane_data)
            total_clusters_accepted += 1
        
        remaining_pcd = remaining_pcd.select_by_index(inlier_indices, invert=True)
    
    print(f"\n  Detected {total_clusters_accepted} physically distinct planar clusters")

    print("  [Debug] Saving RANSAC+DBSCAN visualization...")
    remaining_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    debug_colored_pcds.append(remaining_pcd)
    
    combined_pcd = o3d.geometry.PointCloud()
    for p in debug_colored_pcds:
        combined_pcd += p
        
    debug_ply_path = output_dir / "debug_ransac_planes.ply"
    o3d.io.write_point_cloud(str(debug_ply_path), combined_pcd)
    print(f"  [Debug] Saved to: {debug_ply_path}")
    
    wireframe_segments = []
    adjacency_threshold = 0.15  
    min_segment_length = 0.1   
    
    print(f"\n[Wireframe Extraction] Computing topology-aware plane intersections...")
    
    def polygon_to_3d_boundary(plane_data):
        polygon = plane_data['shapely_2d_polygon']
        R = plane_data['rotation_matrix']
        centroid = plane_data['centroid']
        
        coords_2d = np.array(polygon.exterior.coords)
        centroid_rotated = centroid @ R.T
        z_level = centroid_rotated[2]
        coords_3d_rotated = np.column_stack([coords_2d, np.full(len(coords_2d), z_level)])
        coords_3d_world = coords_3d_rotated @ R
        return coords_3d_world
    
    def min_boundary_distance(boundary1, boundary2):
        diff = boundary1[:, np.newaxis, :] - boundary2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return np.min(distances)
    
    def project_to_line(points, line_point, line_direction):
        vecs = points - line_point
        t_values = np.dot(vecs, line_direction)
        return t_values
    
    pairs_checked = 0
    pairs_adjacent = 0
    
    for i in range(len(detected_planes)):
        for j in range(i + 1, len(detected_planes)):
            pairs_checked += 1
            
            plane1_data = detected_planes[i]
            plane2_data = detected_planes[j]
            
            n1 = plane1_data['plane_model'][:3]
            n2 = plane2_data['plane_model'][:3]
            
            dot_product = abs(np.dot(n1, n2))
            if dot_product > parallel_threshold:
                continue
            
            boundary1_3d = polygon_to_3d_boundary(plane1_data)
            boundary2_3d = polygon_to_3d_boundary(plane2_data)
            
            min_dist = min_boundary_distance(boundary1_3d, boundary2_3d)
            if min_dist > adjacency_threshold:
                continue
            
            pairs_adjacent += 1
            
            line_dir = np.cross(n1, n2)
            line_dir_norm = np.linalg.norm(line_dir)
            if line_dir_norm < 1e-8:
                continue
            line_dir = line_dir / line_dir_norm
            
            d1, d2 = plane1_data['plane_model'][3], plane2_data['plane_model'][3]
            fix_axis = np.argmin(np.abs(line_dir))
            axes = [k for k in range(3) if k != fix_axis]
            
            A = np.array([[n1[axes[0]], n1[axes[1]]],
                          [n2[axes[0]], n2[axes[1]]]])
            b = np.array([-d1, -d2])
            
            if abs(np.linalg.det(A)) < 1e-8:
                continue
            
            solution = np.linalg.solve(A, b)
            point_on_line = np.zeros(3)
            point_on_line[axes[0]] = solution[0]
            point_on_line[axes[1]] = solution[1]
            
            t1_values = project_to_line(boundary1_3d, point_on_line, line_dir)
            t2_values = project_to_line(boundary2_3d, point_on_line, line_dir)
            
            t1_min, t1_max = np.min(t1_values), np.max(t1_values)
            t2_min, t2_max = np.min(t2_values), np.max(t2_values)
            
            t_overlap_min = max(t1_min, t2_min)
            t_overlap_max = min(t1_max, t2_max)
            
            if t_overlap_max <= t_overlap_min:
                continue
            
            segment_length = t_overlap_max - t_overlap_min
            if segment_length < min_segment_length:
                continue
            
            start_point = point_on_line + t_overlap_min * line_dir
            end_point = point_on_line + t_overlap_max * line_dir
            
            wireframe_segments.append({
                'start': start_point,
                'end': end_point,
                'plane_i': i,
                'plane_j': j,
            })
    
    print(f"  Pairs checked: {pairs_checked}, Adjacent pairs: {pairs_adjacent}")
    print(f"  Found {len(wireframe_segments)} wireframe segments")
    return wireframe_segments, detected_planes

# ==========================================
# 3. MAIN BAKING SCRIPT
# ==========================================
def main():
    print("--- 3D Point Cloud Geometry Baker ---")
    
    point_cloud_name = "tmb_office1_subsampled"
    
    pcd_path = project_root / "data" / "raw_point_cloud" / f"{point_cloud_name}.ply"
    meta_path = project_root / "data" / "density_image" / point_cloud_name / "metadata.json"
    
    # Setup output directory
    output_base_dir = project_root / "data" / "debug_renderer" / point_cloud_name
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Metadata & Alignment...")
    with open(meta_path) as f: 
        meta = json.load(f)

    rotation_matrix = np.array(meta['rotation_matrix'])

    print(f"Loading Point Cloud: {pcd_path.name}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))

    # Apply global alignment so the baked coordinates match the room rotation
    print("Applying global alignment rotation...")
    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    points = np.asarray(pcd.points)

    # Extract geometry
    wireframe_segments, detected_planes = extract_3d_wireframe(
        points,
        output_dir=output_base_dir,
        distance_threshold=0.03,
        ransac_n=3,
        num_iterations=1000,
        min_remaining_points=2000,
        dbscan_eps=0.20,
        dbscan_min_samples=30,
        min_hull_area=0.90, 
        parallel_threshold=0.9
    )

    # Package data for pickling
    bake_data = {
        'wireframe_segments': wireframe_segments,
        'detected_planes': detected_planes
    }

    # Save to disk
    output_pkl = output_base_dir / "room_geometry.pkl"
    print(f"\nBaking extracted geometry to: {output_pkl}")
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(bake_data, f)

    print("[Success] Geometry successfully baked and ready for the lightweight renderer!")

if __name__ == "__main__":
    main()