import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from scipy.optimize import minimize

# ==========================================
# 1. PATH SETUP & CONFIGURATION
# ==========================================
# Logic matches RoomFormer_inference.py: Navigate relative to this script file
current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent  # Go up one level from 'src' to 'scan2measure-webframework'

# Default Base Directories (relative to project root)
ROOMFORMER_BASE = project_root / 'data' / 'reconstructed_floorplans_RoomFormer'
LGT_NET_BASE = project_root / 'data' / 'pano' / 'LGT_Net_processed'

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_roomformer_poly(path):
    """Loads the largest polygon from RoomFormer predictions.json."""
    if not path.exists():
        raise FileNotFoundError(f"RoomFormer file not found: {path}")

    with open(path, 'r') as f:
        data = json.load(f)
    
    # RoomFormer saves a list of polygons. We pick the largest one by area.
    # Format: [ [x, y], [x, y], ... ]
    try:
        polys = [np.array(p) for p in data]
    except Exception as e:
         raise ValueError(f"Could not parse polygons from {path}: {e}")

    if not polys:
        raise ValueError(f"No polygons found in RoomFormer file: {path}")
        
    # Simple area approx (shoelace formula) to find the main room
    def poly_area(p):
        return 0.5 * np.abs(np.dot(p[:,0], np.roll(p[:,1],1)) - np.dot(p[:,1], np.roll(p[:,0],1)))
    
    main_poly = max(polys, key=poly_area)
    return main_poly

def load_lgt_poly(folder_path):
    """
    Loads the LGT-Net layout json. 
    Searches for any file ending in '_layout.json' inside the specific folder.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"LGT-Net folder not found: {folder_path}")

    # Find the specific layout file (e.g., Area3_study_layout.json)
    layout_files = list(folder_path.glob("*_layout.json"))
    
    if not layout_files:
        raise FileNotFoundError(f"No '*_layout.json' file found in {folder_path}")
    
    target_file = layout_files[0] # Take the first one found
    print(f"Found LGT Layout file: {target_file.name}")

    with open(target_file, 'r') as f:
        data = json.load(f)
    
    # LGT-Net Format: "layout_corners": [[x, z], ...]
    # The LGT inference script already handled the coordinate flip, so we load as [x, y]
    poly = np.array(data["layout_corners"])
    return poly

# ==========================================
# 3. ALIGNMENT LOGIC
# ==========================================
def get_centroid(poly):
    return np.mean(poly, axis=0)

def get_scale(poly):
    # Use perimeter as a scale proxy
    return np.sum(np.linalg.norm(poly - np.roll(poly, 1, axis=0), axis=1))


def get_bounding_box(poly):
    mins = np.min(poly, axis=0)
    maxs = np.max(poly, axis=0)
    return mins, maxs


def bounding_box_center(poly):
    mins, maxs = get_bounding_box(poly)
    return (mins + maxs) / 2.0


def bounding_box_diagonal(poly):
    mins, maxs = get_bounding_box(poly)
    return float(np.linalg.norm(maxs - mins))


def _rotation_matrix(angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def _point_to_segment_distance(points, seg_a, seg_b):
    # points: (N, 2), seg_a/seg_b: (2,)
    ab = seg_b - seg_a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return np.linalg.norm(points - seg_a, axis=1)
    ap = points - seg_a
    t = (ap @ ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = seg_a + np.outer(t, ab)
    return np.linalg.norm(points - proj, axis=1)


def mean_distance_points_to_polygon(points, polygon):
    # Mean distance from each point to the nearest point on the polygon boundary (polyline segments)
    if polygon.shape[0] < 2:
        return float('inf')
    min_dists = np.full(points.shape[0], np.inf)
    for i in range(polygon.shape[0]):
        a = polygon[i]
        b = polygon[(i + 1) % polygon.shape[0]]
        d = _point_to_segment_distance(points, a, b)
        min_dists = np.minimum(min_dists, d)
    return float(np.mean(min_dists))

def align_polygons(target_poly, source_poly):
    """
    Aligns source_poly (LGT) to target_poly (RoomFormer).
    Returns: transformed_source_poly, (scale, rotation_deg, translation)
    """
    # Global alignment strategy:
    # - Scale by bounding box diagonal ratio
    # - Initialize translation by aligning bounding box centers
    # - For each 90-degree rotation, optimize translation (dx, dy) via minimize()
    #   using mean distance from transformed source vertices to the target polygon boundary.

    target_center = bounding_box_center(target_poly)
    source_center = bounding_box_center(source_poly)

    target_diag = bounding_box_diagonal(target_poly)
    source_diag = bounding_box_diagonal(source_poly)
    if source_diag <= 1e-12:
        raise ValueError("Source polygon bounding box diagonal is zero; cannot compute scale.")

    scale_factor = target_diag / source_diag
    centered_source = (source_poly - source_center) * scale_factor

    best_error = float('inf')
    best_angle = 0
    best_translation = None
    best_poly = None

    rotations = [0, 90, 180, 270]
    for angle in rotations:
        R = _rotation_matrix(angle)

        def objective(x):
            dx, dy = float(x[0]), float(x[1])
            transformed = centered_source @ R.T + (target_center + np.array([dx, dy]))
            return mean_distance_points_to_polygon(transformed, target_poly)

        res = minimize(
            objective,
            x0=np.array([0.0, 0.0]),
            method="Powell",
            options={"maxiter": 300, "xtol": 1e-4, "ftol": 1e-4},
        )

        err = float(res.fun)
        if err < best_error:
            best_error = err
            best_angle = angle
            dx_opt, dy_opt = float(res.x[0]), float(res.x[1])
            best_translation = target_center + np.array([dx_opt, dy_opt])
            best_poly = centered_source @ R.T + best_translation

    return best_poly, (scale_factor, best_angle, best_translation)

# ==========================================
# 4. MAIN FUNCTION
# ==========================================
def main():
    # --- USER CONFIGURATION: FOLDER NAMES ---
    # Change these variables to switch between datasets
    rf_folder_name = "Area_3_selected_rooms_no_RGB"
    lgt_folder_name = "Area3_study"
    
    # --- PATH CONSTRUCTION ---
    rf_path = ROOMFORMER_BASE / rf_folder_name / "predictions.json"
    lgt_folder_path = LGT_NET_BASE / lgt_folder_name
    
    print(f"--- Polygon Alignment Script ---")
    print(f"Project Root: {project_root}")
    print(f"Loading RoomFormer: {rf_path}")
    print(f"Loading LGT-Net from folder: {lgt_folder_path}")

    try:
        rf_poly = load_roomformer_poly(rf_path)
        lgt_poly = load_lgt_poly(lgt_folder_path)
    except Exception as e:
        print(f"\n[Error] Failed to load data: {e}")
        return

    # --- Perform Alignment ---
    print("\nAligning...")
    aligned_lgt, params = align_polygons(rf_poly, lgt_poly)
    scale, angle, trans = params
    
    print(f"Alignment Found:")
    print(f"  Scale Factor (Meters -> Pixels): {scale:.2f}")
    print(f"  Rotation: {angle} degrees")
    print(f"  Target Center (Translation): {trans}")

    # --- Plotting ---
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1, 2],
        height_ratios=[1, 1],
        wspace=0.05,
        hspace=0.05,
    )

    ax_lgt = fig.add_subplot(gs[0, 0])
    ax_rf = fig.add_subplot(gs[1, 0])
    ax_align = fig.add_subplot(gs[:, 1])

    def _style_axes(ax):
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

    # Top-left: Raw Input LGT-Net polygon
    ax_lgt.set_title("Raw Input LGT-Net Polygon")
    ax_lgt.plot(
        np.append(lgt_poly[:, 0], lgt_poly[0, 0]),
        np.append(lgt_poly[:, 1], lgt_poly[0, 1]),
        'b-',
        linewidth=2,
    )

    # Bottom-left: Raw Input RoomFormer polygon
    ax_rf.set_title("Raw Input RoomFormer Polygon")
    ax_rf.plot(
        np.append(rf_poly[:, 0], rf_poly[0, 0]),
        np.append(rf_poly[:, 1], rf_poly[0, 1]),
        'r-',
        linewidth=2,
    )

    # Right: Align Result (spans both rows)
    ax_align.set_title(f"Align Result (Rot: {angle}°)")

    # RoomFormer (Static)
    ax_align.plot(
        np.append(rf_poly[:, 0], rf_poly[0, 0]),
        np.append(rf_poly[:, 1], rf_poly[0, 1]),
        'r-',
        linewidth=2,
        label='Target (RoomFormer)',
    )

    # LGT-Net (Transformed)
    ax_align.plot(
        np.append(aligned_lgt[:, 0], aligned_lgt[0, 0]),
        np.append(aligned_lgt[:, 1], aligned_lgt[0, 1]),
        'b--',
        linewidth=2,
        label='Aligned LGT-Net',
    )
    
    # Calculate Camera Position
    # LGT origin (0,0) is camera. Transform (0,0) using the same parameters.
    lgt_origin = np.array([[0, 0]])
    lgt_center = bounding_box_center(lgt_poly)
    lgt_origin_centered = (lgt_origin - lgt_center) * scale

    R = _rotation_matrix(angle)
    final_cam_pos = lgt_origin_centered @ R.T + trans

    ax_align.plot(
        final_cam_pos[0, 0],
        final_cam_pos[0, 1],
        'ko',
        markersize=8,
        label="Camera Position",
    )

    # Style all axes (equal aspect, no ticks/labels)
    for ax in (ax_lgt, ax_rf, ax_align):
        _style_axes(ax)

    # Invert Y axis for Image Coordinates
    for ax in (ax_lgt, ax_rf, ax_align):
        ax.invert_yaxis()

    # Legend outside plot area, anchored to bottom-right of the figure
    handles, labels = ax_align.get_legend_handles_labels()
    fig.subplots_adjust(right=0.82)
    fig.legend(
        handles,
        labels,
        loc='lower right',
        bbox_to_anchor=(0.99, 0.01),
        bbox_transform=fig.transFigure,
    )

    plt.show()

if __name__ == "__main__":
    main()