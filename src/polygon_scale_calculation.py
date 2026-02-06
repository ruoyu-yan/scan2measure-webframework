import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# ==========================================
# 1. PATH SETUP
# ==========================================
# Assuming script is in 'src' and data is in 'data'
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
ROOMFORMER_BASE = project_root / 'data' / 'reconstructed_floorplans_RoomFormer'
LGT_NET_BASE = project_root / 'data' / 'pano' / 'LGT_Net_processed'

# ==========================================
# 2. LOADING UTILITIES
# ==========================================
def load_rf_global_vertices(rf_path):
    """Loads RF polygons and returns the vertices of the unified Global Shell."""
    with open(rf_path, 'r') as f:
        data = json.load(f)
    
    polys = [Polygon(p) for p in data if len(p) >= 3]
    # Merge all sub-rooms into one "Red Polygon" map
    global_shell = unary_union(polys)
    
    vertices = []
    if isinstance(global_shell, Polygon):
        vertices.extend(global_shell.exterior.coords)
    elif isinstance(global_shell, MultiPolygon):
        for geom in global_shell.geoms:
            vertices.extend(geom.exterior.coords)
    
    # Deduplicate points to create a clean corner cloud
    unique_vertices = np.unique(np.array(vertices), axis=0)
    return unique_vertices

def load_lgt_corners(folder_name):
    """Loads LGT corners from the processed folder."""
    folder_path = LGT_NET_BASE / folder_name
    # Search for layout files
    layout_files = list(folder_path.glob("*_layout.json"))
    if not layout_files:
        print(f"Warning: No layout file found for {folder_name}")
        return None
    
    with open(layout_files[0], 'r') as f:
        data = json.load(f)
    return np.array(data["layout_corners"])

# ==========================================
# 3. DISTANCE HISTOGRAM LOGIC
# ==========================================
def get_all_pairwise_distances(pts):
    """Calculates distances between every possible pair of corners."""
    n = len(pts)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            # Filter out tiny distances (noise/overlap points)
            if d > 0.1: 
                dists.append(d)
    return np.array(dists)

def calculate_global_scale(rf_folder, lgt_folders):
    rf_path = ROOMFORMER_BASE / rf_folder / "predictions.json"
    
    print(f"Processing RoomFormer Map: {rf_folder}...")
    rf_vertices = load_rf_global_vertices(rf_path)
    rf_dists = get_all_pairwise_distances(rf_vertices)
    
    all_lgt_dists = []
    print(f"Processing LGT Rooms: {lgt_folders}...")
    for folder in lgt_folders:
        corners = load_lgt_corners(folder)
        if corners is not None:
            all_lgt_dists.extend(get_all_pairwise_distances(corners))
    
    lgt_dists = np.array(all_lgt_dists)
    
    # GENERATE CANDIDATE SCALES (RF_pixels / LGT_meters)
    # This is a cross-product of all distances to find recurring ratios
    candidate_scales = []
    
    # We use a subset of distances to keep the computation efficient 
    # (Significant features > 1.0m)
    lgt_filtered = lgt_dists[lgt_dists > 1.0]
    rf_filtered = rf_dists[rf_dists > 10.0]
    
    for d_lgt in lgt_filtered:
        scales = rf_filtered / d_lgt
        # Only keep scales in a plausible range (e.g., 2 to 30 pixels/meter)
        candidate_scales.extend(scales[(scales > 2) & (scales < 30)])
    
    candidate_scales = np.array(candidate_scales)
    
    # FIND THE PEAK (Histogram Mode)
    # Use fine bins to find the exact consensus scale
    bins = np.arange(2.0, 30.0, 0.1)
    hist, bin_edges = np.histogram(candidate_scales, bins=bins)
    
    peak_idx = np.argmax(hist)
    best_scale = (bin_edges[peak_idx] + bin_edges[peak_idx+1]) / 2
    
    print(f"\n--- Scale Calculation Complete ---")
    print(f"Consensus Scale: {best_scale:.4f} pixels/meter")
    print(f"Peak Confidence: {hist[peak_idx]} votes")
    
    # Optional: Visualization of the voting peak
    plt.figure(figsize=(10, 5))
    plt.bar(bin_edges[:-1], hist, width=0.1, color='blue', alpha=0.7)
    plt.axvline(best_scale, color='red', linestyle='--', label=f'Peak Scale: {best_scale:.2f}')
    plt.title(f"Scale Consensus Histogram ({rf_folder})")
    plt.xlabel("Scale (Pixels / Meter)")
    plt.ylabel("Frequency of Geometric Matches")
    plt.legend()
    plt.savefig(project_root / 'data' / 'scale_histogram.png')
    print(f"Histogram saved to data/scale_histogram.png")
    
    return best_scale

if __name__ == "__main__":
    rf_folder_name = "tmb_office_corridors_subsampled"
    lgt_folders = ["TMB_office1", "TMB_corridor_south2", "TMB_corridor_south1", "TMB_hall1"]
    
    try:
        calculate_global_scale(rf_folder_name, lgt_folders)
    except Exception as e:
        print(f"Error: {e}")