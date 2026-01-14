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
    """Loads all polygons from RoomFormer predictions.json."""
    if not path.exists():
        raise FileNotFoundError(f"RoomFormer file not found: {path}")

    with open(path, 'r') as f:
        data = json.load(f)
    
    # RoomFormer saves a list of polygons.
    # Format: [ [ [x, y], [x, y], ... ], [ [x, y], ... ], ... ]
    try:
        polys = [np.array(p) for p in data]
    except Exception as e:
         raise ValueError(f"Could not parse polygons from {path}: {e}")

    if not polys:
        raise ValueError(f"No polygons found in RoomFormer file: {path}")

    return polys

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


def load_lgt_rooms(lgt_folders):
    rooms = []
    for name in lgt_folders:
        folder_path = LGT_NET_BASE / name
        poly = load_lgt_poly(folder_path)
        rooms.append({"name": name, "poly": poly})
    return rooms

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

def align_polygons(target_poly, source_poly, force_scale=None):
    """
    Aligns source_poly (LGT) to one target_poly (RoomFormer).
    Returns: transformed_source_poly, (scale, rotation_deg, translation), error
    """
    # Global alignment strategy:
    # - Scale by bounding box diagonal ratio
    # - Initialize translation by aligning bounding box centers
    # - For each 90-degree rotation, optimize translation (dx, dy) via minimize()
    #   using symmetric Chamfer distance between polygon boundaries.

    target_center = bounding_box_center(target_poly)
    source_center = bounding_box_center(source_poly)

    target_diag = bounding_box_diagonal(target_poly)
    source_diag = bounding_box_diagonal(source_poly)
    if source_diag <= 1e-12:
        raise ValueError("Source polygon bounding box diagonal is zero; cannot compute scale.")

    if force_scale is None:
        scale_factor = target_diag / source_diag
    else:
        scale_factor = float(force_scale)
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
            d_st = mean_distance_points_to_polygon(transformed, target_poly)
            d_ts = mean_distance_points_to_polygon(target_poly, transformed)
            return 0.5 * (d_st + d_ts)

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

    return best_poly, (scale_factor, best_angle, best_translation), best_error

# ==========================================
# 4. MAIN FUNCTION
# ==========================================
def main():
    # --- USER CONFIGURATION: FOLDER NAMES ---
    # Change these variables to switch between datasets
    rf_folder_name = "Area_3_selected_rooms_no_RGB"
    lgt_folders = ["Area3_study", "Area_3_office2", "Area3_small_office"]
    
    # --- PATH CONSTRUCTION ---
    rf_path = ROOMFORMER_BASE / rf_folder_name / "predictions.json"
    
    print(f"--- Polygon Alignment Script ---")
    print(f"Project Root: {project_root}")
    print(f"Loading RoomFormer: {rf_path}")
    print(f"Loading LGT-Net folders: {lgt_folders}")

    try:
        rf_polys = load_roomformer_poly(rf_path)
        lgt_rooms = load_lgt_rooms(lgt_folders)
    except Exception as e:
        print(f"\n[Error] Failed to load data: {e}")
        return

    # --- Global Scale Consensus ---
    # Vote for pixels-per-meter using bbox diagonal ratios across all (RF, LGT) pairs.
    scale_votes = []
    for rf_poly in rf_polys:
        rf_diag = bounding_box_diagonal(rf_poly)
        if rf_diag <= 1e-12:
            continue
        for room in lgt_rooms:
            lgt_diag = bounding_box_diagonal(room["poly"])
            if lgt_diag <= 1e-12:
                continue
            scale_votes.append(rf_diag / lgt_diag)

    if not scale_votes:
        print("\n[Error] No valid scale votes could be computed.")
        return

    hist, bin_edges = np.histogram(scale_votes, bins=60)
    peak_bin = int(np.argmax(hist))
    GLOBAL_PIXEL_TO_METER_RATIO = float((bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0)
    print(f"\nGlobal Scale Consensus (Pixels/Meter): {GLOBAL_PIXEL_TO_METER_RATIO:.4f}")

    # --- Matching with global scale consensus ---
    # For each LGT room, find best RF polygon match under a 15% scale-consistency constraint.
    matches = []
    print("\nMatching each LGT room to RoomFormer polygons...")

    for room in lgt_rooms:
        room_name = room["name"]
        lgt_poly = room["poly"]
        lgt_diag = bounding_box_diagonal(lgt_poly)
        if lgt_diag <= 1e-12:
            print(f"  - Skipping {room_name}: invalid LGT diagonal")
            continue

        best_error = float('inf')
        best_idx = None
        best_aligned = None
        best_params = None

        for idx, rf_poly in enumerate(rf_polys):
            rf_diag = bounding_box_diagonal(rf_poly)
            if rf_diag <= 1e-12:
                continue

            ratio = rf_diag / lgt_diag
            if abs(ratio - GLOBAL_PIXEL_TO_METER_RATIO) / GLOBAL_PIXEL_TO_METER_RATIO > 0.15:
                continue

            try:
                aligned, params, err = align_polygons(
                    rf_poly,
                    lgt_poly,
                    force_scale=GLOBAL_PIXEL_TO_METER_RATIO,
                )
            except Exception:
                continue

            if err < best_error:
                best_error = err
                best_idx = idx
                best_aligned = aligned
                best_params = params

        if best_idx is None:
            print(f"  - No match for {room_name} (scale-constrained)")
            continue

        scale, angle, trans = best_params
        matches.append(
            {
                "name": room_name,
                "rf_idx": best_idx,
                "aligned": best_aligned,
                "error": best_error,
                "scale": scale,
                "angle": angle,
                "trans": trans,
            }
        )
        print(f"  - {room_name}: RF[{best_idx}] rot={angle} err={best_error:.4f}")

    if not matches:
        print("\n[Error] No rooms matched under the global-scale constraint.")
        return

    # --- Plotting ---
    n_rooms = len(lgt_rooms)
    fig_height = max(2.5 * n_rooms, 6)
    fig = plt.figure(figsize=(14, fig_height))
    gs = fig.add_gridspec(
        nrows=n_rooms,
        ncols=2,
        width_ratios=[1, 2.2],
        wspace=0.05,
        hspace=0.15,
    )

    ax_lgts = [fig.add_subplot(gs[i, 0]) for i in range(n_rooms)]
    ax_map = fig.add_subplot(gs[:, 1])

    def _style_axes(ax):
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

    # Left column: raw input LGT-Net rooms
    for ax, room in zip(ax_lgts, lgt_rooms):
        poly = room["poly"]
        ax.set_title(room["name"])
        ax.plot(
            np.append(poly[:, 0], poly[0, 0]),
            np.append(poly[:, 1], poly[0, 1]),
            'b-',
            linewidth=2,
        )

    # Right column: RoomFormer map + matched aligned overlays
    ax_map.set_title("Align Result (Global Scale Consensus)")

    # RoomFormer background map
    for i, poly in enumerate(rf_polys):
        ax_map.plot(
            np.append(poly[:, 0], poly[0, 0]),
            np.append(poly[:, 1], poly[0, 1]),
            color='gray',
            linewidth=1.0,
            alpha=0.6,
            label='RoomFormer Map' if i == 0 else '_nolegend_',
        )

    # Overlay matched aligned LGT rooms
    for i, m in enumerate(matches):
        aligned = m["aligned"]
        ax_map.plot(
            np.append(aligned[:, 0], aligned[0, 0]),
            np.append(aligned[:, 1], aligned[0, 1]),
            'b--',
            linewidth=2,
            label='Aligned LGT Rooms' if i == 0 else '_nolegend_',
        )

        center = bounding_box_center(aligned)
        ax_map.text(
            center[0],
            center[1],
            m["name"],
            color='blue',
            fontsize=10,
            ha='center',
            va='center',
        )

    # Style all axes (equal aspect, no ticks/labels)
    for ax in (*ax_lgts, ax_map):
        _style_axes(ax)

    # Invert Y axis for Image Coordinates
    for ax in (*ax_lgts, ax_map):
        ax.invert_yaxis()

    # Legend outside plot area, anchored to bottom-right of the figure
    handles, labels = ax_map.get_legend_handles_labels()
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