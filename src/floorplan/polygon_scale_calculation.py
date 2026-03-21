import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.signal import find_peaks

# ==========================================
# 1. PATH SETUP
# ==========================================
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
    
    return np.unique(np.array(vertices), axis=0)

def load_lgt_corners(folder_name):
    """Loads LGT corners from the processed folder."""
    folder_path = LGT_NET_BASE / folder_name
    layout_files = list(folder_path.glob("*_layout.json"))
    if not layout_files:
        return None
    
    with open(layout_files[0], 'r') as f:
        data = json.load(f)
    return np.array(data["layout_corners"])

# ==========================================
# 3. DISTANCE CALCULATIONS
# ==========================================
def get_all_pairwise_distances(pts, min_d=1.0):
    """Calculates distances between every possible pair of corners."""
    n = len(pts)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > min_d: 
                dists.append(d)
    return np.array(dists)

# ==========================================
# 4. MAIN SCALE LOGIC
# ==========================================
def calculate_global_scale_ranked(rf_folder, lgt_folders):
    rf_path = ROOMFORMER_BASE / rf_folder / "predictions.json"
    
    print(f"--- Automated Scale Ranking ---")
    rf_vertices = load_rf_global_vertices(rf_path)
    # Filter for structural walls (longer than 15 pixels)
    rf_dists = get_all_pairwise_distances(rf_vertices, min_d=15.0)
    
    all_lgt_dists = []
    for folder in lgt_folders:
        corners = load_lgt_corners(folder)
        if corners is not None:
            # Filter for structural walls (longer than 1.5 meters)
            all_lgt_dists.extend(get_all_pairwise_distances(corners, min_d=1.5))
    
    lgt_dists = np.array(all_lgt_dists)
    
    # GENERATE CANDIDATE SCALES
    candidate_scales = []
    for d_lgt in lgt_dists:
        scales = rf_dists / d_lgt
        # platoon of plausible scales
        candidate_scales.extend(scales[(scales > 5.0) & (scales < 15.0)])
    
    candidate_scales = np.array(candidate_scales)
    
    # HISTOGRAM PEAK DETECTION
    bin_width = 0.05
    bins = np.arange(5.0, 15.0, bin_width)
    hist, bin_edges = np.histogram(candidate_scales, bins=bins)
    
    # Use scipy to find local maxima (peaks)
    peaks, properties = find_peaks(hist, distance=5, height=np.mean(hist))
    peak_heights = properties['peak_heights']
    
    # Sort by height to find Top 5
    top_indices = peaks[np.argsort(peak_heights)[-5:][::-1]]
    
    results = []
    print(f"\nTop 5 Consensus Scale Peaks:")
    for i, idx in enumerate(top_indices):
        scale_val = (bin_edges[idx] + bin_edges[idx+1]) / 2
        confidence = hist[idx]
        results.append((scale_val, confidence))
        print(f"  Rank {i+1}: {scale_val:.4f} pixels/meter (Votes: {confidence})")
    
    # VISUALIZATION
    plt.figure(figsize=(12, 6))
    plt.plot(bin_edges[:-1], hist, color='black', alpha=0.3)
    plt.fill_between(bin_edges[:-1], hist, color='blue', alpha=0.1)
    
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for i, (val, conf) in enumerate(results):
        plt.axvline(val, color=colors[i], linestyle='--', label=f'Rank {i+1}: {val:.2f}')
        plt.text(val, max(hist)*0.9, f" {i+1}", color=colors[i], fontweight='bold')

    plt.title(f"Top 5 Scale Candidates for {rf_folder}")
    plt.xlabel("Scale (Pixels / Meter)")
    plt.ylabel("Geometric Vote Count")
    plt.legend()
    plt.savefig(project_root / 'data' / 'top_5_scale_histogram.png')
    
    return results

if __name__ == "__main__":
    rf_folder_name = "tmb_office_corridors_subsampled"
    lgt_folders = ["TMB_office1", "TMB_corridor_south2", "TMB_corridor_south1", "TMB_hall1"]
    
    calculate_global_scale_ranked(rf_folder_name, lgt_folders)