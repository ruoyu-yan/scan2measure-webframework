import json
from pathlib import Path

def main():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    # Target room variable for user selection
    target_room = "TMB_office1"
    
    # Matching point cloud sub-folder name
    point_cloud_name = "tmb_office1_subsampled"

    # ==========================================
    # 2. PATH RESOLUTION (Relative to src folder)
    # ==========================================
    # Resolves to: .../scan2measure-webframework/src
    current_script_dir = Path(__file__).resolve().parent
    
    # Resolves to: .../scan2measure-webframework
    project_root = current_script_dir.parent

    # Construct the paths to the JSON files
    path_3d_lines = project_root / "data" / "debug_renderer" / point_cloud_name / target_room / "visible_3d_lines.json"
    path_2d_lines = project_root / "data" / "pano" / "2d_feature_extracted" / target_room / "extracted_2d_lines.json"

    # ==========================================
    # 3. LOAD & PRINT DATA
    # ==========================================
    
    # --- Load 3D Features (Channel B) ---
    print(f"[{target_room}] Fetching 3D visible lines...")
    if path_3d_lines.exists():
        with open(path_3d_lines, 'r') as f:
            visible_3d_lines = json.load(f)
        print(f"  -> Successfully loaded {path_3d_lines.name}")
        
        # Print the content (Truncated for console readability)
        json_str_3d = json.dumps(visible_3d_lines, indent=4)
        if len(json_str_3d) > 1500:
            print(f"{json_str_3d[:1500]}\n... [TRUNCATED - {len(visible_3d_lines.keys()) if isinstance(visible_3d_lines, dict) else len(visible_3d_lines)} entries total]\n")
        else:
            print(f"{json_str_3d}\n")
    else:
        print(f"  -> [ERROR] Could not find 3D lines at: {path_3d_lines}\n")

    # --- Load 2D Features (Panorama) ---
    print(f"[{target_room}] Fetching 2D optical lines...")
    if path_2d_lines.exists():
        with open(path_2d_lines, 'r') as f:
            extracted_2d_lines = json.load(f)
        print(f"  -> Successfully loaded {path_2d_lines.name}")
        
        # Print the content (Truncated for console readability)
        json_str_2d = json.dumps(extracted_2d_lines, indent=4)
        if len(json_str_2d) > 1500:
            print(f"{json_str_2d[:1500]}\n... [TRUNCATED - {len(extracted_2d_lines.keys()) if isinstance(extracted_2d_lines, dict) else len(extracted_2d_lines)} entries total]\n")
        else:
            print(f"{json_str_2d}\n")
    else:
        print(f"  -> [ERROR] Could not find 2D lines at: {path_2d_lines}\n")

if __name__ == "__main__":
    main()