"""
SAM3 Footprint Comparison — Three-Way Fusion + Morphological + LGT-Net
=======================================================================
Implements three parameter-free floor/ceiling fusion variants (A, B, C),
a morphological baseline (Approach 2), and compares against LGT-Net
on 6 test panoramas.

Approach A: Per-column optimistic (min of floor, mirrored ceiling)
Approach B: Dual XZ radial maximum (project both, take furthest per azimuth)
Approach C: Ceiling-primary with floor expansion
Approach 2: Morphological Cleanup on Union Mask (unchanged)
All:        Manhattan Regularization post-processing

Usage:
    conda run -n sam3 python src/experiments/SAM3_footprint_comparison.py
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_fill_holes
from scipy.ndimage import label as ndimage_label
from shapely.geometry import LinearRing, Polygon, box
import torch

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
script_dir = Path(__file__).resolve().parent          # src/experiments/
src_dir = script_dir.parent                            # src/
project_root = src_dir.parent                          # scan2measure-webframework/
sam3_repo_path = project_root / "sam3"

# Add sam3 repo to sys.path FIRST (so spp's module-level import succeeds)
if str(sam3_repo_path) not in sys.path:
    sys.path.insert(0, str(sam3_repo_path))

# Add src/experiments/ so we can import the sibling module
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import SAM3_pano_processing as spp

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PANO_STEMS = ["TMB_office1", "TMB_corridor_south1", "TMB_corridor_south2",
              "TMB_hall1", "TMB_hall2", "TMB_hall3"]

INPUT_DIR = project_root / "data" / "sam3_pano_processing"
OUTPUT_DIR = project_root / "data" / "sam3_pano_processing" / "footprint_comparison"
LGT_NET_DIR = project_root / "data" / "pano" / "LGT_Net_processed"

MORPH_DISK_RADIUS = 40        # Approach 2: morphological closing kernel radius
MANHATTAN_BIN_DEG = 2         # Regularization: angle histogram bin size
MANHATTAN_SNAP_DEG = 15       # Regularization: max snap angle
DEPTH_CLIP_MAX = 8.0          # Regularization: corridor depth cap (meters)
BOUNDARY_RESOLUTION = 256     # Resampled boundary points (match existing)
SIMPLIFY_TOLERANCE = 0.05     # Shapely polygon simplification (match existing)
CONFIDENCE_THRESHOLD = 0.5
COLUMN_DILATE_RADIUS = 15


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
def load_model():
    """Load SAM3 model with correct path setup."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("Loading SAM3 model...")
    weights_path = sam3_repo_path / "weights" / "sam3.pt"
    if weights_path.exists():
        print(f"  Using local weights: {weights_path}")
        model = build_sam3_image_model(checkpoint_path=str(weights_path))
    else:
        print("  Downloading weights from HuggingFace...")
        model = build_sam3_image_model(load_from_HF=True)
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)
    print(f"  Model loaded (confidence_threshold={CONFIDENCE_THRESHOLD}).")
    return processor


# ---------------------------------------------------------
# CEILING BOUNDARY EXTRACTION
# ---------------------------------------------------------
def extract_ceiling_boundary(masks_binary):
    """
    Extract lower boundary of the largest ceiling mask.
    For each column, finds the bottommost ceiling pixel (highest row index
    where ceiling is present). Returns boundary_row array of shape (width,).
    """
    if masks_binary.shape[0] == 0:
        return None

    areas = masks_binary.sum(axis=(1, 2))
    largest_idx = np.argmax(areas)
    mask = masks_binary[largest_idx]

    h, w = mask.shape
    boundary = np.full(w, -1, dtype=np.float64)

    for col in range(w):
        col_data = mask[:, col]
        ceil_rows = np.where(col_data > 0)[0]
        if len(ceil_rows) > 0:
            boundary[col] = ceil_rows[-1]  # bottommost ceiling pixel

    boundary = spp._interpolate_gaps(boundary)
    return boundary


# ---------------------------------------------------------
# SHARED: COLUMN MASK DETECTION
# ---------------------------------------------------------
def _detect_column_cols(column_masks, img_h, img_w):
    """Return boolean array marking columns occluded by architectural columns."""
    cols = np.zeros(img_w, dtype=bool)
    if column_masks is not None and column_masks.shape[0] > 0:
        combined = np.any(column_masks, axis=0).astype(np.uint8)
        coverage = combined.sum(axis=0) / img_h
        cols = coverage > 0.1
    return cols


# ---------------------------------------------------------
# SHARED: HEIGHT RATIO ESTIMATION
# ---------------------------------------------------------
def _estimate_height_ratio(floor_bnd, ceil_bnd, img_h):
    """Estimate ceiling_height / floor_height from boundary positions.

    At unobstructed columns both boundaries see the same wall at distance d:
        tan(angle_floor) = camera_h / d
        tan(angle_ceil)  = ceil_h   / d
    So the ratio  ceil_h / camera_h = tan(angle_ceil) / tan(angle_floor).

    Returns the median ratio across all valid columns (robust to obstacle outliers).
    """
    angle_f = (floor_bnd / img_h - 0.5) * np.pi   # radians below equator
    angle_c = (0.5 - ceil_bnd / img_h) * np.pi     # radians above equator

    # Only use columns where both angles are positive (valid, non-degenerate)
    valid = (angle_f > 0.01) & (angle_c > 0.01)
    if valid.sum() < 10:
        return 1.0  # fallback: assume symmetric

    ratios = np.tan(angle_c[valid]) / np.tan(angle_f[valid])
    ratio = float(np.median(ratios))
    print(f"    Height ratio (ceil/floor): {ratio:.2f}")
    return ratio


def _ceiling_to_floor_row(ceil_bnd, img_h, height_ratio):
    """Convert ceiling boundary rows to equivalent floor boundary rows.

    For each ceiling row, compute the wall distance implied by the ceiling,
    then compute what floor row that same wall distance would produce.
    This corrects for asymmetric camera placement (camera closer to floor).
    """
    angle_c = (0.5 - ceil_bnd / img_h) * np.pi     # radians above equator
    # angle_f = arctan(tan(angle_c) / height_ratio)
    angle_f = np.arctan(np.tan(angle_c) / height_ratio)
    floor_equiv = img_h * (0.5 + angle_f / np.pi)
    return floor_equiv


# ---------------------------------------------------------
# APPROACH A: PER-COLUMN OPTIMISTIC
# ---------------------------------------------------------
def approach_A_optimistic(floor_bnd, ceil_bnd, column_masks, img_h, img_w,
                          height_ratio):
    """Per-column min(floor, height-corrected ceiling). Returns fused boundary."""
    ceiling_as_floor = _ceiling_to_floor_row(ceil_bnd, img_h, height_ratio)
    fused = np.minimum(floor_bnd, ceiling_as_floor)
    col_mask = _detect_column_cols(column_masks, img_h, img_w)
    fused[col_mask] = -1
    fused = spp._interpolate_gaps(fused)
    return fused


# ---------------------------------------------------------
# APPROACH B: DUAL XZ RADIAL MAXIMUM
# ---------------------------------------------------------
def approach_B_dual_xz(floor_bnd, ceil_bnd, column_masks, img_h, img_w,
                        height_ratio):
    """Project both boundaries to XZ at correct heights, take radial max per azimuth.
    Returns (raw_polygon_xz, floor_xz_points, ceiling_xz_points)."""
    col_mask = _detect_column_cols(column_masks, img_h, img_w)

    # Median-filter both boundaries for noise reduction (wrap mode for equirect)
    floor_filt = median_filter(floor_bnd, size=15, mode='wrap')
    ceil_filt = median_filter(ceil_bnd, size=15, mode='wrap')

    # Mark column-occluded columns as NaN
    floor_filt = floor_filt.astype(np.float64)
    ceil_filt = ceil_filt.astype(np.float64)
    floor_filt[col_mask] = np.nan
    ceil_filt[col_mask] = np.nan

    # Resample to BOUNDARY_RESOLUTION equidistant columns
    src_w = len(floor_bnd)
    dst_cols = np.linspace(0, src_w - 1, BOUNDARY_RESOLUTION)
    floor_resampled = np.interp(dst_cols, np.arange(src_w), floor_filt)
    ceil_resampled = np.interp(dst_cols, np.arange(src_w), ceil_filt)

    # Propagate NaN: if any source column in the neighborhood was NaN
    for i in range(BOUNDARY_RESOLUTION):
        src_idx = int(round(dst_cols[i]))
        if col_mask[min(src_idx, src_w - 1)]:
            floor_resampled[i] = np.nan
            ceil_resampled[i] = np.nan

    # Project floor boundary to XZ (plan_y=1.0 = floor)
    f_valid = ~np.isnan(floor_resampled)
    f_lon, f_lat = spp._pixel2lonlat(dst_cols[f_valid], floor_resampled[f_valid],
                                      img_w, img_h)
    x_f, z_f = spp._lonlat2xyz(f_lon, f_lat, plan_y=1.0)
    z_f = -z_f
    floor_xz = np.column_stack([x_f, z_f])

    # Project ceiling boundary to XZ at correct height
    plan_y_ceil = -height_ratio  # negative because ceiling is above camera
    c_valid = ~np.isnan(ceil_resampled)
    c_lon, c_lat = spp._pixel2lonlat(dst_cols[c_valid], ceil_resampled[c_valid],
                                      img_w, img_h)
    x_c, z_c = spp._lonlat2xyz(c_lon, c_lat, plan_y=plan_y_ceil)
    z_c = -z_c
    ceil_xz = np.column_stack([x_c, z_c])

    # Compute azimuth and radial distance for both point sets
    f_azimuth = np.arctan2(x_f, z_f)
    f_radius = np.sqrt(x_f**2 + z_f**2)
    c_azimuth = np.arctan2(x_c, z_c)
    c_radius = np.sqrt(x_c**2 + z_c**2)

    # Bin into BOUNDARY_RESOLUTION azimuth bins over [-pi, pi)
    n_bins = BOUNDARY_RESOLUTION
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    selected_x = np.full(n_bins, np.nan)
    selected_z = np.full(n_bins, np.nan)

    # Digitize both sets
    f_bin = np.digitize(f_azimuth, bin_edges) - 1
    f_bin = np.clip(f_bin, 0, n_bins - 1)
    c_bin = np.digitize(c_azimuth, bin_edges) - 1
    c_bin = np.clip(c_bin, 0, n_bins - 1)

    # Per bin: collect best (largest radius) from each set
    for b in range(n_bins):
        f_in_bin = np.where(f_bin == b)[0]
        c_in_bin = np.where(c_bin == b)[0]

        f_best_r, f_best_x, f_best_z = -1, np.nan, np.nan
        c_best_r, c_best_x, c_best_z = -1, np.nan, np.nan

        if len(f_in_bin) > 0:
            best_f_idx = f_in_bin[np.argmax(f_radius[f_in_bin])]
            f_best_r = f_radius[best_f_idx]
            f_best_x = x_f[best_f_idx]
            f_best_z = z_f[best_f_idx]

        if len(c_in_bin) > 0:
            best_c_idx = c_in_bin[np.argmax(c_radius[c_in_bin])]
            c_best_r = c_radius[best_c_idx]
            c_best_x = x_c[best_c_idx]
            c_best_z = z_c[best_c_idx]

        # Take the one with larger radial distance
        if f_best_r >= 0 and c_best_r >= 0:
            if f_best_r >= c_best_r:
                selected_x[b], selected_z[b] = f_best_x, f_best_z
            else:
                selected_x[b], selected_z[b] = c_best_x, c_best_z
        elif f_best_r >= 0:
            selected_x[b], selected_z[b] = f_best_x, f_best_z
        elif c_best_r >= 0:
            selected_x[b], selected_z[b] = c_best_x, c_best_z

    # Interpolate empty bins from neighbors
    valid = ~np.isnan(selected_x)
    n_valid = valid.sum()
    if n_valid < 3:
        return None, floor_xz, ceil_xz

    if n_valid < n_bins:
        valid_idx = np.where(valid)[0]
        invalid_idx = np.where(~valid)[0]
        selected_x[invalid_idx] = np.interp(invalid_idx, valid_idx, selected_x[valid_idx],
                                            period=n_bins)
        selected_z[invalid_idx] = np.interp(invalid_idx, valid_idx, selected_z[valid_idx],
                                            period=n_bins)

    # Form polygon from selected points in azimuth order
    polygon_pts = np.column_stack([selected_x, selected_z])

    # Simplify
    try:
        ring = LinearRing(polygon_pts)
        simplified = ring.simplify(SIMPLIFY_TOLERANCE)
        raw_polygon = np.array(simplified.coords)[:-1]
    except Exception:
        raw_polygon = polygon_pts

    return raw_polygon, floor_xz, ceil_xz


# ---------------------------------------------------------
# APPROACH C: CEILING-PRIMARY WITH FLOOR EXPANSION
# ---------------------------------------------------------
def approach_C_ceiling_primary(floor_bnd, ceil_bnd, column_masks, img_h, img_w,
                               height_ratio):
    """Ceiling-primary with floor expansion. Returns (fused_boundary, used_floor_mask)."""
    ceiling_as_floor = _ceiling_to_floor_row(ceil_bnd, img_h, height_ratio)
    fused = np.copy(ceiling_as_floor)
    used_floor = floor_bnd < ceiling_as_floor
    fused[used_floor] = floor_bnd[used_floor]
    col_mask = _detect_column_cols(column_masks, img_h, img_w)
    fused[col_mask] = -1
    used_floor[col_mask] = False
    fused = spp._interpolate_gaps(fused)
    return fused, used_floor


# ---------------------------------------------------------
# APPROACH 2: MORPHOLOGICAL CLEANUP ON UNION MASK
# ---------------------------------------------------------
def _disk_kernel(radius):
    """Create a circular structuring element."""
    diameter = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2 <= radius ** 2).astype(np.uint8)


def approach2_morphological(floor_masks, ceiling_masks, column_masks, img_h, img_w):
    """
    Union floor | ceiling | column(dilated), morphological closing, hole-fill,
    keep largest component, extract upper boundary.

    Returns: (boundary, union_mask, cleaned_mask) for debug visualization.
    """
    # Largest floor mask
    floor_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if floor_masks.shape[0] > 0:
        areas = floor_masks.sum(axis=(1, 2))
        floor_mask = floor_masks[np.argmax(areas)]

    # Largest ceiling mask
    ceiling_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if ceiling_masks.shape[0] > 0:
        areas = ceiling_masks.sum(axis=(1, 2))
        ceiling_mask = ceiling_masks[np.argmax(areas)]

    # Union of all column masks, then dilate
    column_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if column_masks is not None and column_masks.shape[0] > 0:
        column_mask = np.any(column_masks, axis=0).astype(np.uint8)
        if column_mask.any():
            dilate_kernel = _disk_kernel(COLUMN_DILATE_RADIUS)
            column_mask = cv2.dilate(column_mask, dilate_kernel)

    # Union
    union = np.clip(
        floor_mask.astype(np.int16) + ceiling_mask.astype(np.int16)
        + column_mask.astype(np.int16), 0, 1
    ).astype(np.uint8)

    # Morphological closing
    close_kernel = _disk_kernel(MORPH_DISK_RADIUS)
    closed = cv2.morphologyEx(union, cv2.MORPH_CLOSE, close_kernel)

    # Fill holes
    filled = binary_fill_holes(closed).astype(np.uint8)

    # Keep largest connected component
    labeled, n_components = ndimage_label(filled)
    if n_components > 0:
        component_sizes = np.bincount(labeled.ravel())[1:]  # skip background
        largest_label = int(np.argmax(component_sizes)) + 1
        cleaned = (labeled == largest_label).astype(np.uint8)
    else:
        cleaned = filled

    # Extract upper boundary (topmost pixel per column)
    boundary = np.full(img_w, -1, dtype=np.float64)
    for col in range(img_w):
        col_data = cleaned[:, col]
        rows = np.where(col_data > 0)[0]
        if len(rows) > 0:
            boundary[col] = rows[0]

    boundary = spp._interpolate_gaps(boundary)
    return boundary, union, cleaned


# ---------------------------------------------------------
# MANHATTAN REGULARIZATION
# ---------------------------------------------------------
def manhattan_regularize(polygon_xz):
    """
    Manhattan regularization for an XZ polygon:
    1. Pre-simplify aggressively to reduce noisy corners
    2. Detect dominant directions from length-weighted edge angle histogram
    3. Force-snap ALL edges to the nearest principal direction
    4. Merge consecutive collinear edges (same direction) into single edges
    5. Reconstruct corners by intersecting consecutive non-collinear edges
    6. Validate with Shapely; fall back to raw polygon if invalid
    7. Depth-clip against ±DEPTH_CLIP_MAX bounding box
    """
    if polygon_xz is None or len(polygon_xz) < 3:
        return polygon_xz

    # Pre-simplify: reduce fine detail before angle analysis.
    # The input from boundary_to_polygon uses 0.05m tolerance — too fine.
    # Use 0.3m to get a cleaner skeleton for Manhattan snapping.
    try:
        ring = LinearRing(polygon_xz)
        simplified = ring.simplify(0.3)
        pts = np.array(simplified.coords)[:-1]
        if len(pts) < 3:
            pts = polygon_xz
    except Exception:
        pts = polygon_xz

    n = len(pts)

    # Edge vectors and lengths
    edges = np.diff(np.vstack([pts, pts[0]]), axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles_mod = angles % np.pi  # unsigned direction → [0, π)

    # Length-weighted angle histogram
    bin_size = np.radians(MANHATTAN_BIN_DEG)
    n_bins = int(np.pi / bin_size)
    hist = np.zeros(n_bins)
    for i in range(n):
        if lengths[i] < 1e-6:
            continue
        bin_idx = int(angles_mod[i] / bin_size) % n_bins
        hist[bin_idx] += lengths[i]

    primary_bin = int(np.argmax(hist))
    primary_angle = (primary_bin + 0.5) * bin_size
    secondary_angle = (primary_angle + np.pi / 2) % np.pi
    principal_angles = [primary_angle, secondary_angle]

    # Force-snap EVERY edge to the nearest principal direction.
    # In a Manhattan building all walls are axis-aligned — forcing the snap
    # lets the subsequent merge step collapse collinear runs effectively.
    edge_snaps = []  # (snap_id, unit_direction, midpoint)
    for i in range(n):
        mid = (pts[i] + pts[(i + 1) % n]) / 2.0
        if lengths[i] < 1e-6:
            # Degenerate edge: assign to primary
            unit = np.array([np.cos(primary_angle), np.sin(primary_angle)])
            edge_snaps.append((0, unit, mid))
            continue

        # Find nearest principal direction (no threshold — always snap)
        best_id = 0
        best_diff = np.pi
        for idx, pa in enumerate(principal_angles):
            diff = min(abs(angles_mod[i] - pa),
                       np.pi - abs(angles_mod[i] - pa))
            if diff < best_diff:
                best_diff = diff
                best_id = idx

        pa = principal_angles[best_id]
        unit = np.array([np.cos(pa), np.sin(pa)])
        if np.dot(edges[i], unit) < 0:
            unit = -unit
        edge_snaps.append((best_id, unit, mid))

    # Merge consecutive edges with the same snap direction.
    # Use length-weighted centroid as the representative point.
    merged = []
    i = 0
    while i < n:
        sid, unit, mid = edge_snaps[i]
        total_len = lengths[i]
        weighted_mid = mid * lengths[i]
        j = i + 1
        while j < i + n:
            nj = j % n
            nsid, _, nmid = edge_snaps[nj]
            if nsid != sid:
                break
            total_len += lengths[nj]
            weighted_mid += nmid * lengths[nj]
            j += 1
        rep = weighted_mid / total_len if total_len > 1e-12 else mid
        merged.append((sid, unit, rep))
        consumed = j - i
        i += consumed

    m = len(merged)
    if m < 3:
        return polygon_xz

    # Reconstruct corners by intersecting consecutive non-collinear edges
    new_corners = []
    for i in range(m):
        _, d_i, p_i = merged[i]
        _, d_j, p_j = merged[(i + 1) % m]

        A = np.column_stack([d_i, -d_j])
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        b = p_j - p_i

        if abs(det) < 1e-10:
            new_corners.append((p_i + p_j) / 2.0)
        else:
            t = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
            new_corners.append(p_i + t * d_i)

    new_polygon = np.array(new_corners)

    # Validate
    try:
        poly = Polygon(new_polygon)
        if not poly.is_valid or poly.area < 0.1:
            print("    Manhattan: invalid polygon, falling back to raw")
            new_polygon = polygon_xz
        else:
            poly = poly.buffer(0)
            if poly.geom_type == 'Polygon':
                new_polygon = np.array(poly.exterior.coords)[:-1]
    except Exception:
        print("    Manhattan: exception, falling back to raw")
        new_polygon = polygon_xz

    # Depth clipping (corridors)
    try:
        poly = Polygon(new_polygon)
        clip_box = box(-DEPTH_CLIP_MAX, -DEPTH_CLIP_MAX,
                       DEPTH_CLIP_MAX, DEPTH_CLIP_MAX)
        clipped = poly.intersection(clip_box)
        if clipped.geom_type == 'Polygon' and clipped.area > 0.1:
            new_polygon = np.array(clipped.exterior.coords)[:-1]
    except Exception:
        pass

    return new_polygon


# ---------------------------------------------------------
# LGT-NET BASELINE
# ---------------------------------------------------------
def load_lgt_baseline(stem):
    """Load LGT-Net layout_corners [[x, z], ...] from JSON."""
    json_path = LGT_NET_DIR / stem / f"{stem}_layout.json"
    if not json_path.exists():
        print(f"  WARNING: LGT-Net layout not found: {json_path}")
        return None
    with open(json_path) as f:
        data = json.load(f)
    return np.array(data["layout_corners"])


def xz_polygon_to_pixel_boundary(polygon_xz, img_w, img_h, plan_y=1.0):
    """
    Reverse-project XZ polygon corners to equirectangular pixel coordinates.
    Densely interpolates along edges for a smooth overlay curve.

    Returns (px_array, py_array) or (None, None) on failure.
    """
    if polygon_xz is None or len(polygon_xz) < 3:
        return None, None

    # Densely interpolate along polygon edges
    dense = []
    n = len(polygon_xz)
    for i in range(n):
        p0 = polygon_xz[i]
        p1 = polygon_xz[(i + 1) % n]
        n_pts = max(int(np.linalg.norm(p1 - p0) * 50), 10)
        for t in np.linspace(0, 1, n_pts, endpoint=False):
            dense.append(p0 + t * (p1 - p0))
    dense = np.array(dense)

    x = dense[:, 0]
    z_lgt = dense[:, 1]

    # Undo z-flip (LGT-Net convention)
    z_3d = -z_lgt

    # 3D point on floor plane at y = plan_y
    r = np.sqrt(x ** 2 + plan_y ** 2 + z_3d ** 2)
    x_n, y_n, z_n = x / r, plan_y / r, z_3d / r

    lon = np.arctan2(x_n, z_n)
    lat = np.arcsin(np.clip(y_n, -1, 1))

    px = (lon / (2 * np.pi) + 0.5) * img_w - 0.5
    py = (lat / np.pi + 0.5) * img_h - 0.5
    return px, py


# ---------------------------------------------------------
# BOUNDARY → POLYGON PIPELINE
# ---------------------------------------------------------
def boundary_to_polygon_pipeline(boundary, img_h, img_w):
    """smooth_and_resample → boundary_to_polygon → manhattan_regularize."""
    if boundary is None:
        return None, None

    resampled, dst_cols = spp.smooth_and_resample(
        boundary, img_h, resolution=BOUNDARY_RESOLUTION)
    raw_poly = spp.boundary_to_polygon(resampled, dst_cols, img_w, img_h)
    reg_poly = manhattan_regularize(raw_poly)
    return raw_poly, reg_poly


# ---------------------------------------------------------
# VISUALIZATION — per-panorama detail figure
# ---------------------------------------------------------
def _overlay_mask(base, mask, color, alpha=0.4):
    """Blend a single binary mask onto an RGB image."""
    out = base.copy()
    colored = np.zeros_like(base)
    colored[mask > 0] = color
    out = np.where(mask[..., None] > 0,
                   (alpha * colored + (1 - alpha) * out).astype(np.uint8), out)
    return out


def save_detail_figure(stem, img_np,
                       floor_masks, ceiling_masks, column_masks,
                       floor_bnd, ceil_bnd,
                       aA_fused, aA_raw_poly, aA_reg_poly,
                       aB_raw_poly, aB_reg_poly, aB_floor_xz, aB_ceil_xz,
                       aC_fused, aC_used_floor, aC_raw_poly, aC_reg_poly,
                       a2_reg_poly, lgt_poly, out_dir):
    """5 rows × 2 columns per-panorama debug figure."""
    fig, axes = plt.subplots(5, 2, figsize=(20, 30))
    h, w = img_np.shape[:2]
    cols_arr = np.arange(w)

    # ---- Row 0, Col 0: SAM3 masks overlay ----
    overlay = img_np.copy()
    if floor_masks.shape[0] > 0:
        overlay = _overlay_mask(overlay,
                                floor_masks[np.argmax(floor_masks.sum(axis=(1, 2)))],
                                [255, 80, 80])
    if ceiling_masks.shape[0] > 0:
        overlay = _overlay_mask(overlay,
                                ceiling_masks[np.argmax(ceiling_masks.sum(axis=(1, 2)))],
                                [180, 80, 220])
    if column_masks is not None and column_masks.shape[0] > 0:
        for i in range(column_masks.shape[0]):
            overlay = _overlay_mask(overlay, column_masks[i], [80, 80, 255])
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title("SAM3 Masks (red=floor, purple=ceiling, blue=column)")
    axes[0, 0].axis("off")

    # ---- Row 0, Col 1: Raw boundaries on pano ----
    axes[0, 1].imshow(img_np)
    if floor_bnd is not None:
        axes[0, 1].plot(cols_arr, floor_bnd, 'r-', lw=0.8, alpha=0.7, label="Floor raw")
    if ceil_bnd is not None:
        axes[0, 1].plot(cols_arr, ceil_bnd, 'b-', lw=0.8, alpha=0.7, label="Ceiling raw")
        ceiling_mirrored = h - ceil_bnd
        axes[0, 1].plot(cols_arr, ceiling_mirrored, 'b--', lw=0.8, alpha=0.5,
                        label="Ceiling mirrored")
    axes[0, 1].legend(loc="lower right", fontsize=8)
    axes[0, 1].set_title("Floor (red) + Ceiling (blue) + Mirrored (blue dashed)")
    axes[0, 1].axis("off")

    # ---- Row 1, Col 0: Approach A fused boundary on pano ----
    axes[1, 0].imshow(img_np)
    if aA_fused is not None:
        axes[1, 0].plot(cols_arr, aA_fused, 'g-', lw=1.5, alpha=0.9, label="A fused")
    axes[1, 0].legend(loc="lower right", fontsize=8)
    axes[1, 0].set_title("Approach A: Fused boundary (green)")
    axes[1, 0].axis("off")

    # ---- Row 1, Col 1: Approach A XZ polygons ----
    ax = axes[1, 1]
    if aA_raw_poly is not None:
        c = np.vstack([aA_raw_poly, aA_raw_poly[0]])
        ax.plot(c[:, 0], c[:, 1], 'g--', lw=1, alpha=0.6, label="Raw")
    if aA_reg_poly is not None:
        c = np.vstack([aA_reg_poly, aA_reg_poly[0]])
        ax.plot(c[:, 0], c[:, 1], 'g-', lw=2, label="Regularized")
    ax.plot(0, 0, 'k+', ms=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    nA = len(aA_reg_poly) if aA_reg_poly is not None else 0
    ax.set_title(f"Approach A XZ ({nA} corners)")

    # ---- Row 2, Col 0: Approach B XZ scatter ----
    ax = axes[2, 0]
    if aB_floor_xz is not None:
        ax.scatter(aB_floor_xz[:, 0], aB_floor_xz[:, 1], c='red', s=2, alpha=0.4,
                   label="Floor XZ")
    if aB_ceil_xz is not None:
        ax.scatter(aB_ceil_xz[:, 0], aB_ceil_xz[:, 1], c='blue', s=2, alpha=0.4,
                   label="Ceiling XZ")
    if aB_raw_poly is not None:
        c = np.vstack([aB_raw_poly, aB_raw_poly[0]])
        ax.plot(c[:, 0], c[:, 1], 'c-', lw=1.5, alpha=0.8, label="Merged poly")
    ax.plot(0, 0, 'k+', ms=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Approach B: Floor (red) + Ceiling (blue) XZ points")

    # ---- Row 2, Col 1: Approach B XZ polygons ----
    ax = axes[2, 1]
    if aB_raw_poly is not None:
        c = np.vstack([aB_raw_poly, aB_raw_poly[0]])
        ax.plot(c[:, 0], c[:, 1], 'c--', lw=1, alpha=0.6, label="Raw")
    if aB_reg_poly is not None:
        c = np.vstack([aB_reg_poly, aB_reg_poly[0]])
        ax.plot(c[:, 0], c[:, 1], 'c-', lw=2, label="Regularized")
    ax.plot(0, 0, 'k+', ms=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    nB = len(aB_reg_poly) if aB_reg_poly is not None else 0
    ax.set_title(f"Approach B XZ ({nB} corners)")

    # ---- Row 3, Col 0: Approach C fused boundary colored per source ----
    axes[3, 0].imshow(img_np)
    if aC_fused is not None and aC_used_floor is not None:
        floor_cols = np.where(aC_used_floor)[0]
        ceil_cols = np.where(~aC_used_floor)[0]
        if len(floor_cols) > 0:
            axes[3, 0].scatter(floor_cols, aC_fused[floor_cols], c='green', s=1,
                               alpha=0.7, label="Floor-used")
        if len(ceil_cols) > 0:
            axes[3, 0].scatter(ceil_cols, aC_fused[ceil_cols], c='orange', s=1,
                               alpha=0.7, label="Ceiling-used")
    axes[3, 0].legend(loc="lower right", fontsize=8)
    axes[3, 0].set_title("Approach C: Source (green=floor, orange=ceiling)")
    axes[3, 0].axis("off")

    # ---- Row 3, Col 1: Approach C XZ polygons ----
    ax = axes[3, 1]
    if aC_raw_poly is not None:
        c = np.vstack([aC_raw_poly, aC_raw_poly[0]])
        ax.plot(c[:, 0], c[:, 1], color='orange', ls='--', lw=1, alpha=0.6, label="Raw")
    if aC_reg_poly is not None:
        c = np.vstack([aC_reg_poly, aC_reg_poly[0]])
        ax.plot(c[:, 0], c[:, 1], color='orange', ls='-', lw=2, label="Regularized")
    ax.plot(0, 0, 'k+', ms=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    nC = len(aC_reg_poly) if aC_reg_poly is not None else 0
    ax.set_title(f"Approach C XZ ({nC} corners)")

    # ---- Row 4, Col 0: All 5 polygon overlays on pano ----
    axes[4, 0].imshow(img_np)
    poly_styles = [
        (aA_reg_poly, 'green', '-', "A"),
        (aB_reg_poly, 'cyan', '-', "B"),
        (aC_reg_poly, 'orange', '-', "C"),
        (a2_reg_poly, 'blue', '--', "Morph"),
        (lgt_poly, 'red', '--', "LGT"),
    ]
    for poly, color, ls, name in poly_styles:
        if poly is not None and len(poly) >= 3:
            px, py = xz_polygon_to_pixel_boundary(poly, w, h)
            if px is not None:
                axes[4, 0].plot(px, py, color=color, ls=ls, lw=1.5, label=name)
    axes[4, 0].legend(loc="lower right", fontsize=8)
    axes[4, 0].set_title("All Polygons on Pano")
    axes[4, 0].axis("off")

    # ---- Row 4, Col 1: All 5 XZ polygons + metrics ----
    ax = axes[4, 1]
    metrics = []
    for poly, color, ls, name in poly_styles:
        if poly is not None and len(poly) >= 3:
            c = np.vstack([poly, poly[0]])
            ax.plot(c[:, 0], c[:, 1], color=color, ls=ls, lw=2, label=name)
            try:
                area = Polygon(poly).area
                perim = LinearRing(poly).length
                metrics.append(f"{name}: {len(poly)}c, A={area:.1f}m², P={perim:.1f}m")
            except Exception:
                metrics.append(f"{name}: {len(poly)}c")
    ax.plot(0, 0, 'k+', ms=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("All XZ Polygons")
    if metrics:
        ax.text(0.02, 0.02, "\n".join(metrics), transform=ax.transAxes, fontsize=8,
                va='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f"{stem} — Detail Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = out_dir / f"{stem}_detail.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved detail: {save_path}")


# ---------------------------------------------------------
# VISUALIZATION — cross-panorama summary figure
# ---------------------------------------------------------
def save_summary_figure(all_results):
    """6 rows (one per pano) × 2 columns summary figure with 5 methods."""
    n = len(all_results)
    fig, axes = plt.subplots(n, 2, figsize=(18, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    poly_keys_styles = [
        ("aA_poly", 'green', '-', "A"),
        ("aB_poly", 'cyan', '-', "B"),
        ("aC_poly", 'orange', '-', "C"),
        ("a2_poly", 'blue', '--', "Morph"),
        ("lgt_poly", 'red', '--', "LGT"),
    ]

    for row, res in enumerate(all_results):
        stem = res["stem"]
        img_np = res["img_np"]
        h, w = img_np.shape[:2]

        # Col 0: pano + all 5 polygon overlays
        ax = axes[row, 0]
        ax.imshow(img_np)
        for key, color, ls, name in poly_keys_styles:
            poly = res.get(key)
            if poly is not None and len(poly) >= 3:
                px, py = xz_polygon_to_pixel_boundary(poly, w, h)
                if px is not None:
                    ax.plot(px, py, color=color, ls=ls, lw=1.5, label=name)
        ax.set_title(stem, fontsize=10, fontweight="bold")
        ax.axis("off")
        if row == 0:
            ax.legend(loc="lower right", fontsize=7)

        # Col 1: XZ polygon comparison
        ax = axes[row, 1]
        annotations = []
        for key, color, ls, name in poly_keys_styles:
            poly = res.get(key)
            if poly is not None and len(poly) >= 3:
                c = np.vstack([poly, poly[0]])
                ax.plot(c[:, 0], c[:, 1], color=color, ls=ls, lw=2, label=name)
                try:
                    a = Polygon(poly).area
                    annotations.append(f"{name}: {len(poly)}c, {a:.1f}m²")
                except Exception:
                    annotations.append(f"{name}: {len(poly)}c")
        ax.plot(0, 0, 'k+', ms=8)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_title(f"{stem} — XZ", fontsize=10)
        if row == 0:
            ax.legend(fontsize=7)
        if annotations:
            ax.text(0.02, 0.02, "\n".join(annotations), transform=ax.transAxes,
                    fontsize=7, va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle("SAM3 Footprint Comparison Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = OUTPUT_DIR / "comparison_summary.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved summary: {save_path}")


# ---------------------------------------------------------
# PROCESS ONE PANORAMA
# ---------------------------------------------------------
def process_pano(processor, stem):
    """Run all approaches on one panorama, save JSON + detail figure."""
    img_path = INPUT_DIR / f"{stem}.jpg"
    if not img_path.exists():
        print(f"  WARNING: Image not found: {img_path}")
        return None

    print(f"\n{'=' * 60}")
    print(f"Processing: {stem}")
    print(f"{'=' * 60}")

    image = Image.open(str(img_path)).convert("RGB")
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    pano_out = OUTPUT_DIR / stem
    pano_out.mkdir(parents=True, exist_ok=True)

    # Delete stale approach1_layout.json if present
    stale = pano_out / "approach1_layout.json"
    if stale.exists():
        stale.unlink()
        print(f"  Removed stale: {stale.name}")

    # --- SAM3 segmentation (shared across approaches) ---
    print("  Segmenting: floor...")
    floor_masks, _ = spp.segment(processor, image, "floor")
    print(f"    {floor_masks.shape[0]} mask(s)")

    print("  Segmenting: ceiling...")
    ceiling_masks, _ = spp.segment(processor, image, "ceiling")
    print(f"    {ceiling_masks.shape[0]} mask(s)")

    print("  Segmenting: column...")
    column_masks, _ = spp.segment(processor, image, "column")
    print(f"    {column_masks.shape[0]} mask(s)")

    # --- Extract shared boundaries ---
    floor_bnd = spp.extract_floor_boundary(floor_masks)
    ceil_bnd = extract_ceiling_boundary(ceiling_masks)

    # --- Estimate height ratio (ceiling_dist / floor_dist from camera) ---
    height_ratio = 1.0
    if floor_bnd is not None and ceil_bnd is not None:
        height_ratio = _estimate_height_ratio(floor_bnd, ceil_bnd, h)

    # --- Approach A: Per-column optimistic ---
    print("\n  --- Approach A: Per-column optimistic ---")
    aA_fused = None
    aA_raw_poly = aA_reg_poly = None

    if floor_bnd is not None and ceil_bnd is not None:
        aA_fused = approach_A_optimistic(floor_bnd, ceil_bnd, column_masks, h, w,
                                         height_ratio)
    elif floor_bnd is not None:
        print("    No ceiling mask — using floor boundary only")
        aA_fused = floor_bnd.copy()

    if aA_fused is not None:
        aA_raw_poly, aA_reg_poly = boundary_to_polygon_pipeline(aA_fused, h, w)
        if aA_reg_poly is not None:
            spp.save_layout_json(pano_out / "approach_A_layout.json",
                                 aA_reg_poly, "sam3_approach_A_optimistic")
            print(f"    Polygon: {len(aA_reg_poly)} corners")
    else:
        print("    WARNING: No floor mask, skipping Approach A")

    # --- Approach B: Dual XZ radial maximum ---
    print("\n  --- Approach B: Dual XZ radial maximum ---")
    aB_raw_poly = aB_reg_poly = None
    aB_floor_xz = aB_ceil_xz = None

    if floor_bnd is not None and ceil_bnd is not None:
        aB_raw_poly, aB_floor_xz, aB_ceil_xz = \
            approach_B_dual_xz(floor_bnd, ceil_bnd, column_masks, h, w,
                               height_ratio)
        if aB_raw_poly is not None:
            aB_reg_poly = manhattan_regularize(aB_raw_poly)
            if aB_reg_poly is not None:
                spp.save_layout_json(pano_out / "approach_B_layout.json",
                                     aB_reg_poly, "sam3_approach_B_dual_xz")
                print(f"    Polygon: {len(aB_reg_poly)} corners")
        else:
            print("    WARNING: Not enough valid bins for Approach B")
    elif floor_bnd is not None:
        print("    No ceiling mask — falling back to floor-only pipeline")
        aB_raw_poly, aB_reg_poly = boundary_to_polygon_pipeline(floor_bnd, h, w)
        if aB_reg_poly is not None:
            spp.save_layout_json(pano_out / "approach_B_layout.json",
                                 aB_reg_poly, "sam3_approach_B_floor_only_fallback")
            print(f"    Polygon: {len(aB_reg_poly)} corners (floor-only fallback)")
    else:
        print("    WARNING: No floor mask, skipping Approach B")

    # --- Approach C: Ceiling-primary with floor expansion ---
    print("\n  --- Approach C: Ceiling-primary ---")
    aC_fused = None
    aC_used_floor = None
    aC_raw_poly = aC_reg_poly = None

    if floor_bnd is not None and ceil_bnd is not None:
        aC_fused, aC_used_floor = \
            approach_C_ceiling_primary(floor_bnd, ceil_bnd, column_masks, h, w,
                                      height_ratio)
    elif floor_bnd is not None:
        print("    No ceiling mask — using floor boundary only")
        aC_fused = floor_bnd.copy()
        aC_used_floor = np.ones(w, dtype=bool)

    if aC_fused is not None:
        aC_raw_poly, aC_reg_poly = boundary_to_polygon_pipeline(aC_fused, h, w)
        if aC_reg_poly is not None:
            spp.save_layout_json(pano_out / "approach_C_layout.json",
                                 aC_reg_poly, "sam3_approach_C_ceiling_primary")
            print(f"    Polygon: {len(aC_reg_poly)} corners")
    else:
        print("    WARNING: No floor mask, skipping Approach C")

    # --- Approach 2: Morphological Cleanup (unchanged) ---
    print("\n  --- Approach 2: Morphological Cleanup ---")
    a2_bnd, a2_union, a2_cleaned = \
        approach2_morphological(floor_masks, ceiling_masks, column_masks, h, w)
    a2_raw_poly = a2_reg_poly = None

    if a2_bnd is not None:
        a2_raw_poly, a2_reg_poly = boundary_to_polygon_pipeline(a2_bnd, h, w)
        if a2_reg_poly is not None:
            spp.save_layout_json(pano_out / "approach2_layout.json",
                                 a2_reg_poly, "sam3_approach2_morphological_cleanup")
            print(f"    Polygon: {len(a2_reg_poly)} corners")
    else:
        print("    WARNING: No boundary, skipping Approach 2")

    # --- LGT-Net Baseline ---
    print("\n  --- LGT-Net Baseline ---")
    lgt_poly = load_lgt_baseline(stem)
    if lgt_poly is not None:
        print(f"    Loaded: {len(lgt_poly)} corners")

    # --- Detail figure ---
    print("\n  Generating detail figure...")
    save_detail_figure(
        stem, img_np,
        floor_masks, ceiling_masks, column_masks,
        floor_bnd, ceil_bnd,
        aA_fused, aA_raw_poly, aA_reg_poly,
        aB_raw_poly, aB_reg_poly, aB_floor_xz, aB_ceil_xz,
        aC_fused, aC_used_floor, aC_raw_poly, aC_reg_poly,
        a2_reg_poly, lgt_poly, pano_out,
    )

    return {
        "stem": stem,
        "img_np": img_np,
        "floor_bnd": floor_bnd,
        "ceil_bnd": ceil_bnd,
        "aA_fused": aA_fused,
        "aA_poly": aA_reg_poly,
        "aB_poly": aB_reg_poly,
        "aB_floor_xz": aB_floor_xz,
        "aB_ceil_xz": aB_ceil_xz,
        "aC_fused": aC_fused,
        "aC_poly": aC_reg_poly,
        "aC_used_floor": aC_used_floor,
        "a2_boundary": a2_bnd,
        "a2_poly": a2_reg_poly,
        "lgt_poly": lgt_poly,
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("SAM3 Footprint Comparison")
    print(f"  Input:  {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Panos:  {PANO_STEMS}")

    processor = load_model()

    all_results = []
    for stem in PANO_STEMS:
        result = process_pano(processor, stem)
        if result is not None:
            all_results.append(result)

    if all_results:
        print("\nGenerating summary figure...")
        save_summary_figure(all_results)

    print(f"\nDone. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
