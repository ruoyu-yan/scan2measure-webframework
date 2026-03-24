"""
SAM3 Pano Processing — Room Polygons from Panoramic Images
==========================================================
Uses SAM3 text-prompted segmentation on equirectangular panoramic images
to extract floor/wall boundaries, then projects them to XZ polygons
via equirectangular deprojection.

Usage:
    # Process all panos in default directory:
    conda run -n sam3 python src/SAM3_pano_processing.py

    # Process a single pano:
    conda run -n sam3 python src/SAM3_pano_processing.py data/sam3_pano_processing/TMB_office1.jpg

    # Process a directory of panos:
    conda run -n sam3 python src/SAM3_pano_processing.py data/sam3_pano_processing/
"""

import sys
import os
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from shapely.geometry import LinearRing

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sam3_repo_path = project_root / "sam3"

if str(sam3_repo_path) not in sys.path:
    sys.path.insert(0, str(sam3_repo_path))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
OUTPUT_DIR = project_root / "data" / "sam3_pano_processing"
INPUT_DIR = project_root / "data" / "sam3_pano_processing"
CONFIDENCE_THRESHOLD = 0.5
FLOOR_PROMPTS = ["floor"]
WALL_PROMPTS = ["wall"]
MEDIAN_FILTER_WINDOW = 15
BOUNDARY_RESOLUTION = 256
SIMPLIFY_TOLERANCE = 0.05
CAMERA_HEIGHT = 1.4


# ---------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------

def load_model():
    """Load SAM3 model with confidence threshold for natural images."""
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


def segment(processor, image, prompt):
    """Run SAM3 text-prompted segmentation. Returns (masks, scores)."""
    state = processor.set_image(image)
    output = processor.set_text_prompt(prompt=prompt, state=state)

    masks = output["masks"]
    scores = output["scores"]

    if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
        h, w = np.array(image).shape[:2]
        return np.zeros((0, h, w), dtype=np.uint8), np.array([])

    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = np.array(masks)

    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)

    masks_binary = (masks_np > 0.5).astype(np.uint8)

    if scores is not None:
        scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
    else:
        scores_np = np.ones(masks_binary.shape[0])

    return masks_binary, scores_np


# ---------------------------------------------------------
# BOUNDARY EXTRACTION
# ---------------------------------------------------------

def _interpolate_gaps(boundary):
    """Linearly interpolate columns where boundary == -1."""
    valid = boundary >= 0
    if not np.any(valid):
        return None
    if np.all(valid):
        return boundary

    indices = np.arange(len(boundary))
    boundary[~valid] = np.interp(indices[~valid], indices[valid], boundary[valid])
    return boundary


def extract_floor_boundary(masks_binary):
    """
    Extract upper boundary of the largest floor mask.
    Returns boundary_row array of shape (width,) — one row index per column.
    Returns None if no masks.
    """
    if masks_binary.shape[0] == 0:
        return None

    # Use largest mask by pixel count
    areas = masks_binary.sum(axis=(1, 2))
    largest_idx = np.argmax(areas)
    mask = masks_binary[largest_idx]

    h, w = mask.shape
    boundary = np.full(w, -1, dtype=np.float64)

    # For each column, find topmost floor pixel (scan top-to-bottom)
    for col in range(w):
        col_data = mask[:, col]
        floor_rows = np.where(col_data > 0)[0]
        if len(floor_rows) > 0:
            boundary[col] = floor_rows[0]  # topmost = upper edge

    boundary = _interpolate_gaps(boundary)
    return boundary


def extract_wall_boundary(masks_binary):
    """
    Extract lower boundary of combined wall masks.
    Returns boundary_row array of shape (width,) — one row index per column.
    Returns None if no masks.
    """
    if masks_binary.shape[0] == 0:
        return None

    # Combine all wall masks with logical OR
    combined = np.any(masks_binary, axis=0).astype(np.uint8)

    h, w = combined.shape
    boundary = np.full(w, -1, dtype=np.float64)

    # For each column, find bottommost wall pixel (scan bottom-to-top)
    for col in range(w):
        col_data = combined[:, col]
        wall_rows = np.where(col_data > 0)[0]
        if len(wall_rows) > 0:
            boundary[col] = wall_rows[-1]  # bottommost = lower edge

    boundary = _interpolate_gaps(boundary)
    return boundary


def smooth_and_resample(boundary, img_h, resolution=BOUNDARY_RESOLUTION,
                        median_window=MEDIAN_FILTER_WINDOW):
    """
    Smooth boundary with median filter, clamp to below equator,
    resample to `resolution` equidistant longitude steps.
    """
    # Median filter to smooth jagged edges
    smoothed = median_filter(boundary, size=median_window, mode='wrap')

    # Clamp to below equator (row must be > h/2)
    equator_row = img_h / 2.0
    smoothed = np.clip(smoothed, equator_row + 1, img_h - 1)

    # Resample to `resolution` equidistant columns
    w = len(smoothed)
    src_cols = np.arange(w)
    dst_cols = np.linspace(0, w - 1, resolution)
    resampled = np.interp(dst_cols, src_cols, smoothed)

    return resampled, dst_cols


# ---------------------------------------------------------
# GEOMETRIC PROJECTION (boundary → polygon)
# ---------------------------------------------------------

def _pixel2lonlat(px, py, w, h):
    """
    Convert pixel coordinates to longitude/latitude.
    Reimplements LGT-Net's pixel2uv -> uv2lonlat chain.
    """
    u = (px + 0.5) / w
    v = (py + 0.5) / h
    lon = (u - 0.5) * 2 * np.pi    # [-pi, pi]
    lat = (v - 0.5) * np.pi         # [-pi/2, pi/2]
    return lon, lat


def _lonlat2xyz(lon, lat, plan_y=1.0):
    """
    Convert longitude/latitude to XZ projected onto floor plane at y=plan_y.
    Reimplements LGT-Net's lonlat2xyz with plan_y projection.
    """
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    # Project onto y=plan_y plane
    scale = plan_y / y
    x = x * scale
    z = z * scale

    return x, z


def boundary_to_polygon(boundary_rows, dst_cols, img_w, img_h):
    """
    Convert boundary pixel rows to 2D XZ polygon.

    boundary_rows: array of shape (256,) — row indices at resampled columns
    dst_cols: array of shape (256,) — corresponding column pixel positions
    img_w, img_h: original image dimensions

    Returns: simplified polygon as np.array of shape (N, 2) — [[x, z], ...]
    """
    # Step 1: pixel -> lonlat
    lon, lat = _pixel2lonlat(dst_cols, boundary_rows, img_w, img_h)

    # Step 2: lonlat -> XZ with plan_y=1
    x, z = _lonlat2xyz(lon, lat, plan_y=1.0)

    # Step 3: Z-flip to match LGT-Net's save_simple_layout_json convention
    z = z * -1

    # Step 4: Form polygon and simplify
    points = np.column_stack([x, z])
    ring = LinearRing(points)
    simplified = ring.simplify(SIMPLIFY_TOLERANCE)

    # Extract coordinates (LinearRing repeats first point — drop the last)
    coords = np.array(simplified.coords)[:-1]

    return coords


# ---------------------------------------------------------
# JSON OUTPUT
# ---------------------------------------------------------

def save_layout_json(filepath, corners, source):
    """
    Save polygon in LGT-Net layout_corners format.
    corners: np.array of shape (N, 2) — [[x, z], ...]
    source: "sam3_floor" or "sam3_wall"
    """
    data = {
        "layout_corners": corners.tolist(),
        "source": source
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  Saved layout: {filepath}")


# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------

def make_mask_overlay(img_np, masks_binary, alpha=0.5):
    """Create colored mask overlay with contours."""
    overlay = img_np.copy()
    N = masks_binary.shape[0]
    for i in range(N):
        hue = int(180 * i / max(N, 1))
        hsv = np.uint8([[[hue, 200, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        color = (int(bgr[2]), int(bgr[1]), int(bgr[0]))

        mask = masks_binary[i]
        colored = np.zeros_like(img_np)
        colored[mask > 0] = color
        overlay = np.where(
            mask[..., None] > 0,
            (alpha * colored + (1 - alpha) * overlay).astype(np.uint8),
            overlay,
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, (color[2], color[1], color[0]), 2)
        overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay


def save_comparison(stem, img_np, results):
    """
    Save 6-panel comparison figure:
    1. Original pano
    2. Floor mask overlay + boundary (red)
    3. Wall mask overlay + boundary (blue)
    4. Both boundaries overlaid on pano
    5. Floor XZ polygon
    6. Wall XZ polygon
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    h, w = img_np.shape[:2]

    # Panel 1: Original pano
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original Pano")
    axes[0, 0].axis("off")

    # Panel 2: Floor mask + boundary
    if "floor" in results:
        r = results["floor"]
        overlay = make_mask_overlay(img_np, r["masks"])
        axes[0, 1].imshow(overlay)
        axes[0, 1].plot(np.arange(w), r["boundary"], 'r-', linewidth=1, alpha=0.8)
        axes[0, 1].set_title(f"Floor Mask ({r['masks'].shape[0]} masks)")
    else:
        axes[0, 1].imshow(img_np)
        axes[0, 1].set_title("Floor: no mask")
    axes[0, 1].axis("off")

    # Panel 3: Wall mask + boundary
    if "wall" in results:
        r = results["wall"]
        overlay = make_mask_overlay(img_np, r["masks"])
        axes[0, 2].imshow(overlay)
        axes[0, 2].plot(np.arange(w), r["boundary"], 'b-', linewidth=1, alpha=0.8)
        axes[0, 2].set_title(f"Wall Mask ({r['masks'].shape[0]} masks)")
    else:
        axes[0, 2].imshow(img_np)
        axes[0, 2].set_title("Wall: no mask")
    axes[0, 2].axis("off")

    # Panel 4: Both boundaries overlaid
    axes[1, 0].imshow(img_np)
    if "floor" in results:
        axes[1, 0].plot(np.arange(w), results["floor"]["boundary"], 'r-',
                        linewidth=1.5, label="Floor boundary")
    if "wall" in results:
        axes[1, 0].plot(np.arange(w), results["wall"]["boundary"], 'b-',
                        linewidth=1.5, label="Wall boundary")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].set_title("Boundary Comparison")
    axes[1, 0].axis("off")

    # Panel 5: Floor XZ polygon
    if "floor" in results:
        poly = results["floor"]["polygon"]
        closed = np.vstack([poly, poly[0]])
        axes[1, 1].plot(closed[:, 0], closed[:, 1], 'r-o', markersize=4)
        axes[1, 1].plot(0, 0, 'k+', markersize=10)  # camera at origin
        axes[1, 1].set_aspect('equal')
        axes[1, 1].set_title(f"Floor Polygon ({len(poly)} corners)")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].set_title("Floor: no polygon")

    # Panel 6: Wall XZ polygon
    if "wall" in results:
        poly = results["wall"]["polygon"]
        closed = np.vstack([poly, poly[0]])
        axes[1, 2].plot(closed[:, 0], closed[:, 1], 'b-o', markersize=4)
        axes[1, 2].plot(0, 0, 'k+', markersize=10)  # camera at origin
        axes[1, 2].set_aspect('equal')
        axes[1, 2].set_title(f"Wall Polygon ({len(poly)} corners)")
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].set_title("Wall: no polygon")

    plt.suptitle(stem, fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"{stem}_comparison.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison: {save_path}")


# ---------------------------------------------------------
# PROCESS PANO
# ---------------------------------------------------------

def process_pano(processor, img_path):
    """Process a single panoramic image: segment, extract boundary, project to polygon."""
    stem = img_path.stem
    print(f"\nProcessing: {stem}")

    image = Image.open(str(img_path)).convert("RGB")
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    results = {}

    # --- Approach A: Floor ---
    for prompt in FLOOR_PROMPTS:
        print(f"  Floor prompt: '{prompt}'")
        masks, scores = segment(processor, image, prompt)
        print(f"    Masks: {masks.shape[0]}")

        boundary = extract_floor_boundary(masks)
        if boundary is not None:
            resampled, dst_cols = smooth_and_resample(boundary, h)
            polygon = boundary_to_polygon(resampled, dst_cols, w, h)
            json_path = OUTPUT_DIR / f"{stem}_sam3_floor_layout.json"
            save_layout_json(json_path, polygon, "sam3_floor")
            results["floor"] = {
                "masks": masks, "scores": scores, "boundary": boundary,
                "resampled": resampled, "polygon": polygon
            }
        else:
            print(f"    WARNING: No floor mask detected, skipping floor approach")

    # --- Approach B: Walls ---
    for prompt in WALL_PROMPTS:
        print(f"  Wall prompt: '{prompt}'")
        masks, scores = segment(processor, image, prompt)
        print(f"    Masks: {masks.shape[0]}")

        boundary = extract_wall_boundary(masks)
        if boundary is not None:
            resampled, dst_cols = smooth_and_resample(boundary, h)
            polygon = boundary_to_polygon(resampled, dst_cols, w, h)
            json_path = OUTPUT_DIR / f"{stem}_sam3_wall_layout.json"
            save_layout_json(json_path, polygon, "sam3_wall")
            results["wall"] = {
                "masks": masks, "scores": scores, "boundary": boundary,
                "resampled": resampled, "polygon": polygon
            }
        else:
            print(f"    WARNING: No wall mask detected, skipping wall approach")

    # Visualization
    save_comparison(stem, img_np, results)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Accept optional CLI arg (single image or directory, default to INPUT_DIR)
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

    print(f"Found {len(jpg_files)} pano(s):")
    for f in jpg_files:
        print(f"  {f.name}")

    # Load model once
    processor = load_model()

    # Process each pano
    for img_path in jpg_files:
        process_pano(processor, img_path)

    print(f"\nDone. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
