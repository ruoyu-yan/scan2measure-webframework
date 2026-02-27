import numpy as np
import open3d as o3d
import cv2
import json
import sys
import pickle
from pathlib import Path

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# Standard "Virtual Camera" Intrinsics
IMG_W, IMG_H = 1024, 1024
FOV = 60  
FOCAL_LENGTH = 0.5 * IMG_W / np.tan(0.5 * np.radians(FOV))
CX, CY = (IMG_W - 1) / 2.0, (IMG_H - 1) / 2.0

# Unified 22-View Spherical Tiling (60-Degree FOV)
VIEWS_TO_RENDER = [
    # --- Equator ---
    (0.0, 0.0), (45.0, 0.0), (90.0, 0.0), (135.0, 0.0), 
    (180.0, 0.0), (225.0, 0.0), (270.0, 0.0), (315.0, 0.0),
    # --- Upper Ring ---
    (0.0, 45.0), (60.0, 45.0), (120.0, 45.0), 
    (180.0, 45.0), (240.0, 45.0), (300.0, 45.0),
    # --- Lower Ring ---
    (0.0, -45.0), (60.0, -45.0), (120.0, -45.0), 
    (180.0, -45.0), (240.0, -45.0), (300.0, -45.0),
    # --- Poles ---
    (0.0, 90.0),   
    (0.0, -90.0)   
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def find_floor_z(points, bin_size=0.05):
    z_values = points[:, 2]
    z_min, z_max = np.min(z_values), np.max(z_values)
    bins = np.arange(z_min, z_max, bin_size)
    hist, edges = np.histogram(z_values, bins=bins)
    
    lower_threshold_idx = int(len(hist) * 0.30)
    if lower_threshold_idx == 0: lower_threshold_idx = len(hist)
    
    peak_idx = np.argmax(hist[:lower_threshold_idx])
    floor_z = edges[peak_idx] + (bin_size / 2.0)
    return floor_z

def filter_short_edges(binary_image, min_size):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    output = np.zeros_like(binary_image)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_size:
            output[labels == label_id] = 255
    return output

def filter_jagged_edges(binary_image, min_length=60, epsilon=3.0, max_density=0.10, max_wobble=0.15):
    output = np.zeros_like(binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, closed=False)
        if perimeter < min_length:
            continue

        raw_approx = cv2.approxPolyDP(contour, epsilon=1.0, closed=False)
        raw_density = len(raw_approx) / perimeter
        
        total_angle_change = 0
        if len(raw_approx) > 2:
            pts = raw_approx[:, 0, :]
            vecs = pts[1:] - pts[:-1]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            valid_vecs = vecs / (norms + 1e-6)
            dots = np.sum(valid_vecs[:-1] * valid_vecs[1:], axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)
            total_angle_change = np.sum(angles)

        wobble_metric = total_angle_change / perimeter

        is_structure = True
        if raw_density > max_density or wobble_metric > max_wobble:
            is_structure = False

        if is_structure:
            final_approx = cv2.approxPolyDP(contour, epsilon, closed=False)
            cv2.drawContours(output, [final_approx], -1, 255, thickness=1)
            
    return output

# ==========================================
# 3. RENDERING ENGINE
# ==========================================

def render_synthetic_wireframe(points, camera_pose_global, rotation_deg, view_yaw=0, view_pitch=0, wireframe_segments=None, detected_planes=None):
    print(f"  > Rendering View (Yaw={view_yaw} deg, Pitch={view_pitch} deg)...")
    
    # Track visible 3D lines for PnL solver
    visible_3d_lines = []
    
    vecs = points - camera_pose_global
    
    total_yaw_rad = np.radians(-(rotation_deg + view_yaw))
    c_yaw, s_yaw = np.cos(total_yaw_rad), np.sin(total_yaw_rad)
    
    R_z = np.array([
        [c_yaw, -s_yaw, 0],
        [s_yaw,  c_yaw, 0],
        [0,      0,     1]
    ])
    
    local_vecs = vecs @ R_z.T
    
    points_cam = np.zeros_like(local_vecs)
    points_cam[:, 0] = local_vecs[:, 0]  
    points_cam[:, 1] = -local_vecs[:, 2] 
    points_cam[:, 2] = local_vecs[:, 1]  
    
    pitch_rad = np.radians(-view_pitch) 
    c_pitch, s_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
    
    R_x = np.array([
        [1,       0,        0],
        [0, c_pitch, -s_pitch],
        [0, s_pitch,  c_pitch]
    ])
    
    points_cam = points_cam @ R_x.T
    
    mask = points_cam[:, 2] > 0.1
    points_cam = points_cam[mask]
    
    if len(points_cam) == 0:
        empty = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        return empty, empty, empty, empty, empty, empty

    u = (points_cam[:, 0] * FOCAL_LENGTH / points_cam[:, 2]) + CX
    v = (points_cam[:, 1] * FOCAL_LENGTH / points_cam[:, 2]) + CY
    depths = points_cam[:, 2]
    
    valid_mask = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    u = u[valid_mask].astype(int)
    v = v[valid_mask].astype(int)
    depths = depths[valid_mask]
    
    depth_map = np.full((IMG_H, IMG_W), np.inf, dtype=np.float32)
    
    sort_idx = np.argsort(depths)[::-1] 
    u_sorted = u[sort_idx]
    v_sorted = v[sort_idx]
    d_sorted = depths[sort_idx]
    
    depth_map[v_sorted, u_sorted] = d_sorted
    
    depth_valid = depth_map.copy()
    depth_valid[depth_valid == np.inf] = 0
    
    kernel = np.ones((3,3), np.uint8)
    depth_filled = cv2.dilate(depth_valid, kernel, iterations=2)
    
    # Plane Visibility Test
    VIS_SAMPLE_N   = 200   
    VIS_DEPTH_TOL  = 0.05  
    VIS_MIN_RATIO  = 0.05  
    plane_visible = {}  
    
    if detected_planes is not None:
        total_yaw_rad_vis = np.radians(-(rotation_deg + view_yaw))
        pitch_rad_vis = np.radians(-view_pitch)
        c_yaw_v, s_yaw_v = np.cos(total_yaw_rad_vis), np.sin(total_yaw_rad_vis)
        c_pitch_v, s_pitch_v = np.cos(pitch_rad_vis), np.sin(pitch_rad_vis)
        
        R_z_vis = np.array([[c_yaw_v, -s_yaw_v, 0],
                            [s_yaw_v,  c_yaw_v, 0],
                            [0,        0,       1]])
        R_x_vis = np.array([[1,         0,          0],
                            [0, c_pitch_v, -s_pitch_v],
                            [0, s_pitch_v,  c_pitch_v]])
        
        for pi, pdata in enumerate(detected_planes):
            pts = pdata['points_3d']
            n_sample = min(VIS_SAMPLE_N, len(pts))
            indices = np.random.choice(len(pts), n_sample, replace=False)
            sample_pts = pts[indices]
            
            vecs = sample_pts - camera_pose_global
            local = vecs @ R_z_vis.T
            cam = np.empty_like(local)
            cam[:, 0] = local[:, 0]
            cam[:, 1] = -local[:, 2]
            cam[:, 2] = local[:, 1]
            cam = cam @ R_x_vis.T
            
            front = cam[:, 2] > 0.1
            if not np.any(front):
                plane_visible[pi] = False
                continue
            
            cam_f = cam[front]
            u_p = (cam_f[:, 0] * FOCAL_LENGTH / cam_f[:, 2] + CX).astype(int)
            v_p = (cam_f[:, 1] * FOCAL_LENGTH / cam_f[:, 2] + CY).astype(int)
            depths_p = cam_f[:, 2]
            
            bm = (u_p >= 0) & (u_p < IMG_W) & (v_p >= 0) & (v_p < IMG_H)
            if not np.any(bm):
                plane_visible[pi] = False
                continue
            
            u_b, v_b, d_b = u_p[bm], v_p[bm], depths_p[bm]
            buf_d = depth_filled[v_b, u_b]
            
            passed = (buf_d == 0) | (d_b <= buf_d + VIS_DEPTH_TOL)
            ratio = np.sum(passed) / n_sample  
            plane_visible[pi] = (ratio >= VIS_MIN_RATIO)
    
    depth_vis = cv2.normalize(depth_filled, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_denoised = cv2.bilateralFilter(depth_vis, d=20, sigmaColor=100, sigmaSpace=75)
    
    # Channel A: Depth Edges
    edges = cv2.Canny(depth_denoised, 50, 150)
    edges_geometry_filtered = filter_jagged_edges(edges, min_length=60, epsilon=3.0, max_density=0.1)

    # Channel B: Plane Intersections (Loaded from Pickle)
    normal_edges = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    
    if wireframe_segments is not None and len(wireframe_segments) > 0:
        def transform_point_to_camera(point_3d, cam_pose, total_yaw_rad, pitch_rad):
            vec = point_3d - cam_pose
            c_yaw, s_yaw = np.cos(total_yaw_rad), np.sin(total_yaw_rad)
            R_z = np.array([
                [c_yaw, -s_yaw, 0],
                [s_yaw,  c_yaw, 0],
                [0,      0,     1]
            ])
            local_vec = R_z @ vec
            
            point_cam = np.array([
                local_vec[0],   
                -local_vec[2],  
                local_vec[1]    
            ])
            
            c_pitch, s_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
            R_x = np.array([
                [1,       0,        0],
                [0, c_pitch, -s_pitch],
                [0, s_pitch,  c_pitch]
            ])
            point_cam = R_x @ point_cam
            return point_cam
        
        def project_to_2d(point_cam):
            if point_cam[2] <= 0.1:  
                return None
            u = (point_cam[0] * FOCAL_LENGTH / point_cam[2]) + CX
            v = (point_cam[1] * FOCAL_LENGTH / point_cam[2]) + CY
            return np.array([u, v])
        
        total_yaw_rad_seg = np.radians(-(rotation_deg + view_yaw))
        pitch_rad_seg = np.radians(-view_pitch)
        
        for seg in wireframe_segments:
            start_3d = seg['start']
            end_3d   = seg['end']
            pi_a     = seg['plane_i']
            pi_b     = seg['plane_j']
            
            if plane_visible and not plane_visible.get(pi_a, True) and not plane_visible.get(pi_b, True):
                continue
            
            start_cam = transform_point_to_camera(start_3d, camera_pose_global, total_yaw_rad_seg, pitch_rad_seg)
            end_cam = transform_point_to_camera(end_3d, camera_pose_global, total_yaw_rad_seg, pitch_rad_seg)
            
            if start_cam[2] <= 0.1 and end_cam[2] <= 0.1:
                continue
            
            near_clip = 0.1
            if start_cam[2] <= near_clip or end_cam[2] <= near_clip:
                direction = end_cam - start_cam
                if abs(direction[2]) > 1e-8:
                    t_clip = (near_clip - start_cam[2]) / direction[2]
                    clipped_point = start_cam + t_clip * direction
                    if start_cam[2] <= near_clip: start_cam = clipped_point
                    else: end_cam = clipped_point
            
            start_2d = project_to_2d(start_cam)
            end_2d = project_to_2d(end_cam)
            
            if start_2d is None or end_2d is None: continue
            
            if (start_2d[0] < 0 and end_2d[0] < 0) or (start_2d[0] >= IMG_W and end_2d[0] >= IMG_W): continue
            if (start_2d[1] < 0 and end_2d[1] < 0) or (start_2d[1] >= IMG_H and end_2d[1] >= IMG_H): continue
            
            SAMPLE_STEP = 0.02   
            DEPTH_TOL   = 0.05   
            
            seg_vec_3d = end_3d - start_3d
            seg_length = np.linalg.norm(seg_vec_3d)
            num_samples = max(int(np.ceil(seg_length / SAMPLE_STEP)), 2)
            
            prev_px = None       
            prev_visible = False
            segment_has_visible_parts = False
            
            for si in range(num_samples + 1):
                t = si / num_samples  
                sample_3d = start_3d + t * seg_vec_3d
                
                sample_cam = transform_point_to_camera(sample_3d, camera_pose_global, total_yaw_rad_seg, pitch_rad_seg)
                if sample_cam[2] <= 0.1:
                    prev_visible = False
                    prev_px = None
                    continue
                
                sample_2d = project_to_2d(sample_cam)
                if sample_2d is None:
                    prev_visible = False
                    prev_px = None
                    continue
                
                su, sv = int(round(sample_2d[0])), int(round(sample_2d[1]))
                if su < 0 or su >= IMG_W or sv < 0 or sv >= IMG_H:
                    prev_visible = False
                    prev_px = None
                    continue
                
                buf_depth = depth_filled[sv, su]
                sample_depth = sample_cam[2]
                
                is_visible = (buf_depth == 0) or (sample_depth <= buf_depth + DEPTH_TOL)
                cur_px = (su, sv)
                
                if is_visible:
                    segment_has_visible_parts = True
                    if prev_visible and prev_px is not None:
                        cv2.line(normal_edges, prev_px, cur_px, 255, thickness=1)
                    prev_px = cur_px
                    prev_visible = True
                else:
                    prev_visible = False
                    prev_px = None
            
            # Record visible segment for PnL solver
            if segment_has_visible_parts:
                visible_3d_lines.append({
                    "start": start_3d.tolist(),
                    "end": end_3d.tolist()
                })
        
    edges_filtered = filter_short_edges(edges_geometry_filtered, min_size=60)
    normal_edges_filtered = filter_short_edges(normal_edges, min_size=60)
    combined_wireframe = cv2.bitwise_or(edges_filtered, normal_edges_filtered)
    
    return edges, depth_vis, normal_edges, edges_filtered, normal_edges_filtered, combined_wireframe, visible_3d_lines

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    print("--- Lightweight Synthetic Point Cloud Renderer ---")
    
    point_cloud_name = "tmb_office1_subsampled"
    target_rooms = ["TMB_office1"]
    
    # Paths
    pcd_path = project_root / "data" / "raw_point_cloud" / f"{point_cloud_name}.ply"
    json_path = project_root / "data" / "reconstructed_floorplans_RoomFormer" / point_cloud_name / "global_alignment.json"
    meta_path = project_root / "data" / "density_image" / point_cloud_name / "metadata.json"
    
    output_base_dir = project_root / "data" / "debug_renderer" / point_cloud_name
    pkl_path = output_base_dir / "room_geometry.pkl"
    
    # 1. Load Metadata & Alignment
    print("Loading Metadata & Alignment...")
    with open(meta_path) as f: meta = json.load(f)
    with open(json_path) as f: align = json.load(f)

    rotation_matrix = np.array(meta['rotation_matrix'])
    img_width = meta['image_width']
    min_coords = [c / 1000.0 for c in meta['min_coords']]
    max_dim = meta['max_dim'] / 1000.0
    offset = [o / 1000.0 for o in meta.get('offset', [0, 0])]
    scale_factor = max_dim / img_width

    # 2. Load Point Cloud
    print(f"Loading Point Cloud: {pcd_path.name}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    points = np.asarray(pcd.points)
    
    # 3. Load Baked Geometry (The Magic Step)
    if not pkl_path.exists():
        print(f"[Error] Geometry file not found at {pkl_path}. Please run point_cloud_geometry_baker.py first.")
        sys.exit(1)
        
    print(f"Loading Baked Geometry from: {pkl_path.name}")
    with open(pkl_path, 'rb') as f:
        bake_data = pickle.load(f)
        
    wireframe_segments = bake_data['wireframe_segments']
    detected_planes = bake_data['detected_planes']
    print(f"  Loaded {len(wireframe_segments)} segments and {len(detected_planes)} planes.")

    # 4. Render Target Rooms
    for target_room in target_rooms:
        print(f"\nProcessing Room: {target_room}")
        
        room_output_dir = output_base_dir / target_room
        output_dir_a = room_output_dir / "Channel_A"
        output_dir_a_filtered = room_output_dir / "Channel_A_Filtered"
        output_dir_b = room_output_dir / "Channel_B"
        output_dir_b_filtered = room_output_dir / "Channel_B_Filtered"
        output_dir_combined = room_output_dir / "Combined"
        
        for d in [output_dir_a, output_dir_a_filtered, output_dir_b, output_dir_b_filtered, output_dir_combined]:
            d.mkdir(parents=True, exist_ok=True)
            
        room_info = next((r for r in align['alignment_results'] if r['room_name'] == target_room), None)
        if room_info is None:
            continue
            
        pose_px = room_info['camera_pose_global']
        cam_x = min_coords[0] - offset[0] + (pose_px[0] * scale_factor)
        cam_y = min_coords[1] - offset[1] + (pose_px[1] * scale_factor)
        
        local_radius = 2.0
        distances_xy = np.sqrt((points[:, 0] - cam_x)**2 + (points[:, 1] - cam_y)**2)
        local_points = points[distances_xy <= local_radius]
        floor_z = find_floor_z(local_points) if len(local_points) > 100 else find_floor_z(points)
        
        cam_z = floor_z + 1.6
        yaw_deg = room_info['transformation']['rotation_deg']
        final_pose = np.array([cam_x, cam_y, cam_z])
        
        print(f"  Computed Camera Pose: {final_pose}")
        
        # Dictionary to store all visible 3D lines for this room
        all_visible_3d_lines = {}
        
        for view_yaw, view_pitch in VIEWS_TO_RENDER:
            edges, depth_map, normal_edges, edges_filtered, normal_edges_filtered, combined_wireframe, visible_3d_lines = render_synthetic_wireframe(
                points, final_pose, yaw_deg, 
                view_yaw=view_yaw, view_pitch=view_pitch,
                wireframe_segments=wireframe_segments,
                detected_planes=detected_planes
            )
            
            cv2.imwrite(str(output_dir_a / f"synthetic_wireframe_yaw{view_yaw}_pitch{view_pitch}.png"), edges)
            cv2.imwrite(str(output_dir_a / f"synthetic_depth_yaw{view_yaw}_pitch{view_pitch}.png"), depth_map)
            cv2.imwrite(str(output_dir_a_filtered / f"synthetic_wireframe_filtered_yaw{view_yaw}_pitch{view_pitch}.png"), edges_filtered)
            cv2.imwrite(str(output_dir_b / f"synthetic_normal_edges_yaw{view_yaw}_pitch{view_pitch}.png"), normal_edges)
            cv2.imwrite(str(output_dir_b_filtered / f"synthetic_normal_edges_filtered_yaw{view_yaw}_pitch{view_pitch}.png"), normal_edges_filtered)
            cv2.imwrite(str(output_dir_combined / f"synthetic_combined_yaw{view_yaw}_pitch{view_pitch}.png"), combined_wireframe)
            
            # Store visible 3D line coordinates for PnL solver
            view_key = f"yaw{view_yaw}_pitch{view_pitch}"
            all_visible_3d_lines[view_key] = visible_3d_lines
        
        # Save all visible 3D lines to a single JSON file
        json_output_path = room_output_dir / "visible_3d_lines.json"
        with open(json_output_path, "w") as f:
            json.dump(all_visible_3d_lines, f, indent=4)
            
    print(f"\n[Done] Ready for the optimizer loop!")

if __name__ == "__main__":
    main()