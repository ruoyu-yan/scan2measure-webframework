"""
SAM3 Pano Footprint Extraction — Room Polygons via Floor/Ceiling Fusion
========================================================================
Extracts Manhattan-regularized room footprint polygons from equirectangular
panoramas using SAM3 floor + ceiling boundary fusion with height correction.

The fusion exploits the physical invariant that obstacles can only shrink the
visible room extent: per-column, the boundary showing the wall further from
the camera is selected (height-corrected optimistic fusion).

Usage:
    # Process all panos in default directory:
    conda run -n sam3 python src/experiments/SAM3_pano_footprint_extraction.py

    # Process a single pano:
    conda run -n sam3 python src/experiments/SAM3_pano_footprint_extraction.py data/sam3_pano_processing/TMB_office1.jpg

    # Process a directory of panos:
    conda run -n sam3 python src/experiments/SAM3_pano_footprint_extraction.py data/sam3_pano_processing/
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LinearRing, Polygon, box

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
script_dir = Path(__file__).resolve().parent          # src/experiments/
src_dir = script_dir.parent                            # src/
project_root = src_dir.parent                          # scan2measure-webframework/
sam3_repo_path = project_root / "sam3"

if str(sam3_repo_path) not in sys.path:
    sys.path.insert(0, str(sam3_repo_path))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

os.environ["HF_TOKEN"] = "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"

import SAM3_pano_processing as spp

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_DIR = project_root / "data" / "sam3_pano_processing"
OUTPUT_DIR = project_root / "data" / "sam3_pano_processing"

MANHATTAN_BIN_DEG = 2
DEPTH_CLIP_MAX = 8.0
BOUNDARY_RESOLUTION = 256
SIMPLIFY_TOLERANCE = 0.05
CONFIDENCE_THRESHOLD = 0.5


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
    """Extract lower boundary of the largest ceiling mask.
    Returns boundary_row array of shape (width,), or None."""
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
            boundary[col] = ceil_rows[-1]

    boundary = spp._interpolate_gaps(boundary)
    return boundary


# ---------------------------------------------------------
# HEIGHT RATIO ESTIMATION
# ---------------------------------------------------------
def _estimate_height_ratio(floor_bnd, ceil_bnd, img_h):
    """Estimate ceiling_height / floor_height from boundary positions.

    At unobstructed columns both boundaries see the same wall at distance d:
        tan(angle_floor) = camera_h / d
        tan(angle_ceil)  = ceil_h   / d
    So the ratio  ceil_h / camera_h = tan(angle_ceil) / tan(angle_floor).

    Returns the median ratio (robust to obstacle outliers).
    """
    angle_f = (floor_bnd / img_h - 0.5) * np.pi
    angle_c = (0.5 - ceil_bnd / img_h) * np.pi

    valid = (angle_f > 0.01) & (angle_c > 0.01)
    if valid.sum() < 10:
        return 1.0

    ratios = np.tan(angle_c[valid]) / np.tan(angle_f[valid])
    ratio = float(np.median(ratios))
    print(f"    Height ratio (ceil/floor): {ratio:.2f}")
    return ratio


def _ceiling_to_floor_row(ceil_bnd, img_h, height_ratio):
    """Convert ceiling boundary rows to equivalent floor boundary rows,
    correcting for asymmetric camera placement."""
    angle_c = (0.5 - ceil_bnd / img_h) * np.pi
    angle_f = np.arctan(np.tan(angle_c) / height_ratio)
    return img_h * (0.5 + angle_f / np.pi)


# ---------------------------------------------------------
# COLUMN MASK DETECTION
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
# FLOOR/CEILING FUSION (APPROACH A)
# ---------------------------------------------------------
def fuse_boundaries(floor_bnd, ceil_bnd, column_masks, img_h, img_w,
                    height_ratio):
    """Per-column min(floor, height-corrected ceiling). Returns fused boundary."""
    ceiling_as_floor = _ceiling_to_floor_row(ceil_bnd, img_h, height_ratio)
    fused = np.minimum(floor_bnd, ceiling_as_floor)
    col_mask = _detect_column_cols(column_masks, img_h, img_w)
    fused[col_mask] = -1
    fused = spp._interpolate_gaps(fused)
    return fused


# ---------------------------------------------------------
# MANHATTAN REGULARIZATION
# ---------------------------------------------------------
def manhattan_regularize(polygon_xz):
    """Manhattan regularization: snap all edges to two principal directions,
    merge collinear runs, reconstruct corners, validate, depth-clip."""
    if polygon_xz is None or len(polygon_xz) < 3:
        return polygon_xz

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
    angles_mod = angles % np.pi

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

    # Force-snap every edge to nearest principal direction
    edge_snaps = []
    for i in range(n):
        mid = (pts[i] + pts[(i + 1) % n]) / 2.0
        if lengths[i] < 1e-6:
            unit = np.array([np.cos(primary_angle), np.sin(primary_angle)])
            edge_snaps.append((0, unit, mid))
            continue

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

    # Merge consecutive collinear edges
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
        i += j - i

    m = len(merged)
    if m < 3:
        return polygon_xz

    # Reconstruct corners by intersecting consecutive edges
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

    # Depth clipping
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
# XZ → PIXEL REVERSE PROJECTION (for debug viz)
# ---------------------------------------------------------
def xz_polygon_to_pixel_boundary(polygon_xz, img_w, img_h, plan_y=1.0):
    """Reverse-project XZ polygon corners to equirectangular pixel coordinates.
    Returns (px_array, py_array) or (None, None)."""
    if polygon_xz is None or len(polygon_xz) < 3:
        return None, None

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
    z_3d = -dense[:, 1]

    r = np.sqrt(x ** 2 + plan_y ** 2 + z_3d ** 2)
    x_n, y_n, z_n = x / r, plan_y / r, z_3d / r

    lon = np.arctan2(x_n, z_n)
    lat = np.arcsin(np.clip(y_n, -1, 1))

    px = (lon / (2 * np.pi) + 0.5) * img_w - 0.5
    py = (lat / np.pi + 0.5) * img_h - 0.5
    return px, py


# ---------------------------------------------------------
# DEBUG FIGURE
# ---------------------------------------------------------
def save_debug_figure(stem, img_np, floor_bnd, ceil_bnd, fused_bnd,
                      raw_poly, reg_poly, out_dir):
    """1×2 debug figure: boundaries on pano + XZ polygon."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    h, w = img_np.shape[:2]
    cols_arr = np.arange(w)

    # Left: pano with boundaries
    axes[0].imshow(img_np)
    if floor_bnd is not None:
        axes[0].plot(cols_arr, floor_bnd, 'r-', lw=0.6, alpha=0.6, label="Floor")
    if ceil_bnd is not None:
        axes[0].plot(cols_arr, ceil_bnd, 'b-', lw=0.6, alpha=0.6, label="Ceiling")
    if fused_bnd is not None:
        axes[0].plot(cols_arr, fused_bnd, 'g-', lw=1.5, alpha=0.9, label="Fused")
    if reg_poly is not None:
        px, py = xz_polygon_to_pixel_boundary(reg_poly, w, h)
        if px is not None:
            axes[0].plot(px, py, 'lime', ls='--', lw=1, alpha=0.7, label="Polygon")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title(f"{stem} — Boundaries")
    axes[0].axis("off")

    # Right: XZ polygon
    ax = axes[1]
    if raw_poly is not None:
        c = np.vstack([raw_poly, raw_poly[0]])
        ax.plot(c[:, 0], c[:, 1], 'g--', lw=1, alpha=0.5, label="Raw")
    if reg_poly is not None:
        c = np.vstack([reg_poly, reg_poly[0]])
        ax.plot(c[:, 0], c[:, 1], 'g-', lw=2, label="Regularized")
        try:
            area = Polygon(reg_poly).area
            ax.text(0.02, 0.02, f"{len(reg_poly)} corners, {area:.1f} m²",
                    transform=ax.transAxes, fontsize=9, va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        except Exception:
            pass
    ax.plot(0, 0, 'k+', ms=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{stem} — XZ Polygon")

    plt.tight_layout()
    save_path = out_dir / "debug.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved debug: {save_path}")


# ---------------------------------------------------------
# PROCESS ONE PANORAMA
# ---------------------------------------------------------
def process_pano(processor, img_path):
    """Segment, fuse, project, regularize, save layout JSON + debug figure."""
    stem = img_path.stem
    image = Image.open(str(img_path)).convert("RGB")
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    pano_out = OUTPUT_DIR / stem
    pano_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Processing: {stem}")
    print(f"{'=' * 60}")

    # --- SAM3 segmentation ---
    print("  Segmenting: floor...")
    floor_masks, _ = spp.segment(processor, image, "floor")
    print(f"    {floor_masks.shape[0]} mask(s)")

    print("  Segmenting: ceiling...")
    ceiling_masks, _ = spp.segment(processor, image, "ceiling")
    print(f"    {ceiling_masks.shape[0]} mask(s)")

    print("  Segmenting: column...")
    column_masks, _ = spp.segment(processor, image, "column")
    print(f"    {column_masks.shape[0]} mask(s)")

    # --- Extract boundaries ---
    floor_bnd = spp.extract_floor_boundary(floor_masks)
    ceil_bnd = extract_ceiling_boundary(ceiling_masks)

    if floor_bnd is None:
        print("  WARNING: No floor mask — cannot extract polygon")
        return

    # --- Fuse ---
    fused_bnd = None
    if ceil_bnd is not None:
        height_ratio = _estimate_height_ratio(floor_bnd, ceil_bnd, h)
        fused_bnd = fuse_boundaries(floor_bnd, ceil_bnd, column_masks, h, w,
                                    height_ratio)
    else:
        print("  No ceiling mask — using floor boundary only")
        fused_bnd = floor_bnd.copy()

    # --- Project to polygon ---
    raw_poly, reg_poly = boundary_to_polygon_pipeline(fused_bnd, h, w)

    if reg_poly is not None:
        spp.save_layout_json(pano_out / "layout.json", reg_poly,
                             "sam3_floor_ceiling_fusion")
        print(f"  Polygon: {len(reg_poly)} corners")
    else:
        print("  WARNING: Failed to produce polygon")

    # --- Debug figure ---
    save_debug_figure(stem, img_np, floor_bnd, ceil_bnd, fused_bnd,
                      raw_poly, reg_poly, pano_out)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) >= 2:
        target = Path(sys.argv[1])
        if not target.is_absolute():
            target = project_root / target
    else:
        target = INPUT_DIR

    # Find JPG files
    if target.is_file() and target.suffix.lower() in (".jpg", ".jpeg", ".png"):
        jpg_files = [target]
    elif target.is_dir():
        jpg_files = sorted([
            p for p in target.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg")
            and "Zone.Identifier" not in p.name
        ])
    else:
        print(f"Not found: {target}")
        sys.exit(1)

    if not jpg_files:
        print(f"No JPG files found in {target}")
        sys.exit(1)

    print(f"SAM3 Pano Footprint Extraction")
    print(f"  Input:  {target}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Panos:  {len(jpg_files)}")

    processor = load_model()

    for img_path in jpg_files:
        process_pano(processor, img_path)

    print(f"\nDone. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
