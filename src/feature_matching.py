import json
import math
import numpy as np
from pathlib import Path

# ==========================================
# SETUP & CONFIGURATION
# ==========================================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent

def load_data(target_room):
    """Loads the 2D JSON and prepares the path for the 3D JSONs."""
    dir_2d = PROJECT_ROOT / "data" / "pano" / "2d_feature_extracted" / target_room
    file_2d = dir_2d / "extracted_2d_lines.json"
    
    # We map the room name to the subsampled cloud name convention used in the renderer
    cloud_name = f"{target_room.lower()}_subsampled"
    dir_3d = PROJECT_ROOT / "data" / "debug_renderer" / cloud_name / target_room / "Channel_B_3D_Lines"
    
    if not file_2d.exists():
        print(f"[Error] 2D lines file not found at {file_2d}")
        return None, None
        
    with open(file_2d, 'r') as f:
        data_2d = json.load(f)
        
    return data_2d, dir_3d

# ==========================================
# STEP 1: JUNCTION DETECTION
# ==========================================
def detect_junctions(lines, is_3d=False, tolerance=None):
    """
    Groups intersecting lines into junctions.
    tolerance: distance threshold to consider endpoints as touching.
    """
    if tolerance is None:
        tolerance = 0.05 if is_3d else 15.0 # 5cm for 3D, 15px for 2D
        
    junctions = []
    
    for i, line in enumerate(lines):
        if is_3d:
            p1, p2 = np.array(line['start']), np.array(line['end'])
        else:
            p1, p2 = np.array(line[0]), np.array(line[1])
            
        endpoints = [p1, p2]
        
        for ep in endpoints:
            matched_junction = None
            # Check if this endpoint is near an existing junction center
            for junc in junctions:
                dist = np.linalg.norm(junc['center'] - ep)
                if dist < tolerance:
                    matched_junction = junc
                    break
            
            if matched_junction:
                if i not in matched_junction['line_indices']:
                    matched_junction['line_indices'].append(i)
                    matched_junction['lines'].append(line)
                    # Update center to be the average of connected endpoints
                    matched_junction['center'] = (matched_junction['center'] + ep) / 2.0
            else:
                # Create a new junction
                junctions.append({
                    'center': ep,
                    'line_indices': [i],
                    'lines': [line]
                })
                
    # Filter out "junctions" that only consist of 1 line (dead ends)
    valid_junctions = [j for j in junctions if len(j['line_indices']) > 1]
    return valid_junctions

# ==========================================
# STEP 2: RELATIONAL ENCODING (MANHATTAN WORLD VP)
# ==========================================
def encode_3d_junction(junction):
    """Classifies 3D lines into X, Y, Z axes based on world coordinates."""
    encoded_lines = []
    
    for line in junction['lines']:
        p1, p2 = np.array(line['start']), np.array(line['end'])
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-5: continue
        
        vec_norm = vec / length
        
        # Manhattan classification based on global axes
        if abs(vec_norm[2]) > 0.85:
            axis = 'Z_Vertical'
        elif abs(vec_norm[0]) > 0.85:
            axis = 'X_Horizontal'
        elif abs(vec_norm[1]) > 0.85:
            axis = 'Y_Horizontal'
        else:
            axis = 'Unknown'
            
        encoded_lines.append({'axis': axis, 'length': length, 'raw_line': line})
        
    return encoded_lines

def encode_2d_junction(junction):
    """
    Classifies 2D lines by vanishing points.
    In standard gravity-aligned panoramas, vertical world lines map to vertical 2D lines.
    Horizontal world lines map to lines converging at horizontal vanishing points.
    """
    encoded_lines = []
    
    for line in junction['lines']:
        p1, p2 = np.array(line[0]), np.array(line[1])
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-5: continue
        
        dx, dy = vec[0], vec[1]
        angle = abs(math.degrees(math.atan2(dy, dx)))
        
        # Manhattan VP logic for rectilinear slices:
        # Lines pointing near 90 degrees converge to the Zenith/Nadir VP (Verticals)
        # All other lines converge to horizontal VPs along the horizon
        if 75 < angle < 105 or 255 < angle < 285:
            axis = 'Z_Vertical'
        else:
            axis = 'XY_Horizontal' # Combines X and Y since 2D doesn't easily distinguish depth vs width without ray casting
            
        encoded_lines.append({'axis': axis, 'length': length, 'raw_line': line})
        
    return encoded_lines

# ==========================================
# STEP 3: DESCRIPTOR MATCHING
# ==========================================
def calculate_length_ratio(encoded_lines):
    """Calculates the ratio of the longest line to the shortest line in the junction."""
    lengths = [line['length'] for line in encoded_lines]
    if not lengths or min(lengths) == 0: return 0
    return max(lengths) / min(lengths)

def match_junctions(juncs_2d, juncs_3d, ratio_tolerance=0.3):
    """Matches 2D and 3D junctions based on topological signatures and relative length ratios."""
    matches = []
    
    for idx_2d, j2d in enumerate(juncs_2d):
        enc_2d = encode_2d_junction(j2d)
        axes_2d = sorted([line['axis'] for line in enc_2d])
        ratio_2d = calculate_length_ratio(enc_2d)
        
        for idx_3d, j3d in enumerate(juncs_3d):
            enc_3d = encode_3d_junction(j3d)
            
            # Map specific 3D horizontal axes to the generic 2D horizontal VP for comparison
            axes_3d_mapped = sorted(['XY_Horizontal' if 'Horizontal' in line['axis'] else line['axis'] for line in enc_3d])
            ratio_3d = calculate_length_ratio(enc_3d)
            
            # 1. Topological Signature Check (e.g., does it connect a Vertical to a Horizontal?)
            if axes_2d == axes_3d_mapped:
                
                # 2. Relative Length Ratio Check
                if abs(ratio_2d - ratio_3d) < ratio_tolerance:
                    matches.append({
                        '2d_junction_idx': idx_2d,
                        '3d_junction_idx': idx_3d,
                        'signature': axes_2d,
                        'ratio_diff': abs(ratio_2d - ratio_3d)
                    })
                    
    return matches

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main():
    target_room = "TMB_office1"
    print(f"--- Starting Relational Feature Matcher for {target_room} ---")
    
    data_2d, dir_3d = load_data(target_room)
    if not data_2d: return
    
    total_matches = 0
    
    # Process each 2D panorama view
    for view_filename, lines_2d in data_2d.items():
        # Reconstruct the expected 3D JSON filename based on the 2D filename (e.g., 00_yaw0.0_pitch0.0.jpg)
        parts = view_filename.replace('.jpg', '').split('_')
        if len(parts) >= 3:
            yaw_part = parts[1]
            pitch_part = parts[2]
            file_3d = dir_3d / f"visible_3d_lines_{yaw_part}_{pitch_part}.json"
            
            if file_3d.exists():
                with open(file_3d, 'r') as f:
                    lines_3d = json.load(f)
                    
                # 1. Detect Junctions
                juncs_2d = detect_junctions(lines_2d, is_3d=False)
                juncs_3d = detect_junctions(lines_3d, is_3d=True)
                
                # 2 & 3. Encode and Match
                view_matches = match_junctions(juncs_2d, juncs_3d)
                
                if view_matches:
                    print(f"[{view_filename}] Found {len(view_matches)} topological matches.")
                    total_matches += len(view_matches)
                    
    print(f"\n[Success] Matching complete. Total structural correspondences found: {total_matches}")

if __name__ == "__main__":
    main()