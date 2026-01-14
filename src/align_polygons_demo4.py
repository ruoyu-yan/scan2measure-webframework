import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from scipy.optimize import minimize, linear_sum_assignment

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


def bounding_box_width(poly):
    """Return the minimum dimension of the bounding box (width)."""
    mins, maxs = get_bounding_box(poly)
    dims = maxs - mins
    return float(np.min(dims))


def polygon_area(poly):
    """Calculate polygon area using the Shoelace formula."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


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

def align_polygons(target_poly, source_poly, force_scale=None, asymmetric_mode=False):
    """
    Aligns source_poly (LGT) to one target_poly (RoomFormer).
    If asymmetric_mode is True, error is computed only as Source->Target distance,
    allowing shorter LGT shapes to fit inside longer RoomFormer slots.
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
            if asymmetric_mode:
                return d_st
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
    rf_folder_name = "Area_3_selected_rooms_subset_big"
    lgt_folders = ["Area3_study", "Area_3_office2", "Area3_small_office", "Area3_corridor_short"]
    
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

    # --- Global Linear Assignment (Hungarian) with width-based scale hypothesis testing ---
    # 1) Generate candidate scales from RF_width / LGT_width for all valid pairs
    rf_widths = np.array([bounding_box_width(p) for p in rf_polys], dtype=float)
    lgt_widths = np.array([bounding_box_width(r["poly"]) for r in lgt_rooms], dtype=float)
    rf_areas = np.array([polygon_area(p) for p in rf_polys], dtype=float)

    valid_rf = rf_widths > 1e-12
    valid_lgt = lgt_widths > 1e-12
    if not np.any(valid_rf) or not np.any(valid_lgt):
        print("\n[Error] Invalid widths; cannot generate scale candidates.")
        return

    ratio_matrix = rf_widths[None, :] / lgt_widths[:, None]
    candidate_scales = ratio_matrix[np.isfinite(ratio_matrix)].ravel().tolist()

    if not candidate_scales:
        print("\n[Error] No scale candidates could be generated.")
        return

    # Deduplicate to keep runtime reasonable
    candidate_scales = np.unique(np.round(np.array(candidate_scales, dtype=float), 4))

    tol = 0.15  # 15% tolerance
    BIG_COST = 1e9
    COVERAGE_BONUS_WEIGHT = 0.1

    best_scale = None
    best_valid_count = -1
    best_total_cost = float('inf')

    print(f"\nTesting {len(candidate_scales)} candidate scales (width-based)...")

    # Precompute full width ratio matrix for gating
    full_ratio = rf_widths[None, :] / lgt_widths[:, None]

    for s in candidate_scales:
        if not np.isfinite(s) or s <= 0:
            continue

        cost = np.full((len(lgt_rooms), len(rf_polys)), BIG_COST, dtype=float)
        gate = np.isfinite(full_ratio) & (np.abs(full_ratio - s) / s <= tol)

        for i_room, room in enumerate(lgt_rooms):
            lgt_poly = room["poly"]
            for j_rf, rf_poly in enumerate(rf_polys):
                if not gate[i_room, j_rf]:
                    continue
                try:
                    aligned, _, err = align_polygons(
                        rf_poly, lgt_poly, force_scale=s, asymmetric_mode=True
                    )
                except Exception:
                    continue
                # Coverage bonus: reward matches that fill more of the target room
                lgt_area = polygon_area(aligned)
                rf_area = rf_areas[j_rf]
                if rf_area > 1e-12:
                    coverage_bonus = COVERAGE_BONUS_WEIGHT * (lgt_area / rf_area)
                else:
                    coverage_bonus = 0.0
                cost[i_room, j_rf] = err - coverage_bonus

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_costs = cost[row_ind, col_ind]
        valid_mask = assigned_costs < BIG_COST / 2
        valid_count = int(np.sum(valid_mask))
        total_cost = float(np.sum(assigned_costs[valid_mask]))

        if (valid_count > best_valid_count) or (
            valid_count == best_valid_count and total_cost < best_total_cost
        ):
            best_valid_count = valid_count
            best_total_cost = total_cost
            best_scale = float(s)

    if best_scale is None or best_valid_count <= 0:
        print("\n[Error] No valid assignment found under any candidate scale.")
        return

    print(
        f"\nBest Scale: {best_scale:.4f} | Valid Matches: {best_valid_count} | Total Cost: {best_total_cost:.4f}"
    )

    # 2) Run final assignment under the best scale and build final matches
    final_cost = np.full((len(lgt_rooms), len(rf_polys)), BIG_COST, dtype=float)
    final_gate = np.isfinite(full_ratio) & (np.abs(full_ratio - best_scale) / best_scale <= tol)
    for i_room, room in enumerate(lgt_rooms):
        lgt_poly = room["poly"]
        for j_rf, rf_poly in enumerate(rf_polys):
            if not final_gate[i_room, j_rf]:
                continue
            try:
                aligned, _, err = align_polygons(
                    rf_poly, lgt_poly, force_scale=best_scale, asymmetric_mode=True
                )
            except Exception:
                continue
            lgt_area = polygon_area(aligned)
            rf_area = rf_areas[j_rf]
            if rf_area > 1e-12:
                coverage_bonus = COVERAGE_BONUS_WEIGHT * (lgt_area / rf_area)
            else:
                coverage_bonus = 0.0
            final_cost[i_room, j_rf] = err - coverage_bonus

    row_ind, col_ind = linear_sum_assignment(final_cost)
    matches = []
    for i_room, j_rf in zip(row_ind, col_ind):
        if final_cost[i_room, j_rf] >= BIG_COST / 2:
            continue
        room = lgt_rooms[i_room]
        rf_poly = rf_polys[j_rf]
        aligned, params, err = align_polygons(
            rf_poly, room["poly"], force_scale=best_scale, asymmetric_mode=True
        )
        scale, angle, trans = params
        matches.append(
            {
                "name": room["name"],
                "rf_idx": int(j_rf),
                "aligned": aligned,
                "error": float(err),
                "scale": float(scale),
                "angle": int(angle),
                "trans": trans,
            }
        )
        print(f"  - {room['name']} -> RF[{j_rf}] rot={angle}° err={err:.4f}")

    if not matches:
        print("\n[Error] Final assignment produced no finite matches.")
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
    ax_map.set_title("Align Result (Width-Based Asymmetric Assignment)")

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