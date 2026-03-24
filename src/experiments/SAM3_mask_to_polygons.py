"""
SAM3 Mask-to-Polygon Extraction
================================
Reads binary mask PNGs from SAM3 room segmentation and extracts room
polygons via a robust filtering pipeline:

1. Discard full-image junk masks (>70% of total image pixels)
2. Among remaining, identify building-outline masks (>85% of occupied area)
3. Remaining masks = individual room polygons
4. Uncovered occupied area → extra "remaining" polygon(s) for full tiling

Outputs a JSON file compatible with the polygon matching pipeline.

Usage:
    python src/experiments/SAM3_mask_to_polygons.py <map_name>

Examples:
    python src/experiments/SAM3_mask_to_polygons.py tmb_office_one_corridor
    python src/experiments/SAM3_mask_to_polygons.py tmb_office_corridor_bigger
    python src/experiments/SAM3_mask_to_polygons.py tmb_office_one_corridor_dense
"""

import json
import sys
import numpy as np
import cv2
from pathlib import Path

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Polygon approximation tolerance as fraction of contour perimeter.
APPROX_EPS_FRAC = 0.002
# Minimum contour area in pixels to keep (filters noise)
MIN_CONTOUR_AREA_PX = 50
# Mask covers >70% of total image pixels → full-image junk
FULL_IMAGE_RATIO = 0.70
# Mask covers >85% of occupied (non-black) area → building outline
BUILDING_OUTLINE_RATIO = 0.85
# Density image threshold to determine occupied pixels
DENSITY_THRESHOLD = 10


# ---------------------------------------------------------
# PIXEL-TO-WORLD CONVERSION
# ---------------------------------------------------------
def pixels_to_world_meters(pixel_poly, metadata):
    """Convert pixel coordinates to world coordinates in meters."""
    min_coords = np.array(metadata["min_coords"])
    offset = np.array(metadata["offset"])
    max_dim = metadata["max_dim"]
    width = metadata["image_width"]

    scale_factor = width - 1
    min_xy = min_coords[:2]
    off_xy = offset[:2]

    world_poly = []
    for u, v in pixel_poly:
        pixel_vec = np.array([u, v], dtype=float)
        world_xy_mm = (pixel_vec / scale_factor * max_dim) - off_xy + min_xy
        world_poly.append(world_xy_mm / 1000.0)
    return np.array(world_poly)


# ---------------------------------------------------------
# CONTOUR EXTRACTION
# ---------------------------------------------------------
def extract_polygon_from_mask(mask_img):
    """Extract the largest polygon from a binary mask image.

    Returns polygon vertices as (N, 2) array in pixel coordinates,
    or None if no valid contour found.
    """
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA_PX]
    if not valid:
        return None

    largest = max(valid, key=cv2.contourArea)

    perimeter = cv2.arcLength(largest, closed=True)
    eps = APPROX_EPS_FRAC * perimeter
    approx = cv2.approxPolyDP(largest, eps, closed=True)

    poly = approx.reshape(-1, 2).astype(float)
    return poly


# ---------------------------------------------------------
# MASK CLASSIFICATION
# ---------------------------------------------------------
def classify_masks(mask_images, density_img):
    """Classify masks into full-image junk, building outline, or room masks.

    Args:
        mask_images: list of (mask_index, binary_mask_uint8) tuples
        density_img: grayscale density image

    Returns:
        room_masks: list of (mask_index, binary_mask) for room-level masks
        building_mask: combined binary mask of the building footprint (uint8)
    """
    h, w = density_img.shape[:2]
    total_pixels = h * w

    # Occupied area = non-black pixels in density image
    occupied = (density_img > DENSITY_THRESHOLD).astype(np.uint8)
    occupied_pixels = int(np.sum(occupied))

    print(f"\n  Image: {w}x{h} = {total_pixels} px, occupied: {occupied_pixels} px "
          f"({100 * occupied_pixels / total_pixels:.1f}%)")

    # Step 1: compute white pixel counts for each mask
    mask_info = []
    for idx, mask in mask_images:
        white_pixels = int(np.sum(mask > 0))
        image_ratio = white_pixels / total_pixels
        occupied_ratio = white_pixels / occupied_pixels if occupied_pixels > 0 else 0
        mask_info.append({
            "idx": idx,
            "mask": mask,
            "white_pixels": white_pixels,
            "image_ratio": image_ratio,
            "occupied_ratio": occupied_ratio,
        })
        print(f"  Mask {idx:02d}: {white_pixels} px, "
              f"image_ratio={image_ratio:.2f}, occupied_ratio={occupied_ratio:.2f}")

    # Step 2: discard full-image junk (covers >70% of entire image)
    kept = []
    for info in mask_info:
        if info["image_ratio"] > FULL_IMAGE_RATIO:
            print(f"  Mask {info['idx']:02d}: DISCARDED (full-image junk, "
                  f"covers {info['image_ratio']:.0%} of image)")
        else:
            kept.append(info)

    # Step 3: among remaining, identify building outline (covers >85% of occupied area)
    building_outlines = []
    room_masks = []
    for info in kept:
        if info["occupied_ratio"] > BUILDING_OUTLINE_RATIO:
            print(f"  Mask {info['idx']:02d}: BUILDING OUTLINE "
                  f"(covers {info['occupied_ratio']:.0%} of occupied area)")
            building_outlines.append(info)
        else:
            print(f"  Mask {info['idx']:02d}: ROOM MASK")
            room_masks.append(info)

    # Build the building footprint mask:
    # - If we have explicit building outlines, use their union
    # - Otherwise, use the union of all room masks
    if building_outlines:
        building_mask = np.zeros((h, w), dtype=np.uint8)
        for info in building_outlines:
            building_mask = cv2.bitwise_or(building_mask, info["mask"])
    else:
        building_mask = np.zeros((h, w), dtype=np.uint8)
        for info in room_masks:
            building_mask = cv2.bitwise_or(building_mask, info["mask"])

    return [(info["idx"], info["mask"]) for info in room_masks], building_mask


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python src/experiments/SAM3_mask_to_polygons.py <map_name>")
        print("\nExamples:")
        print("  python src/experiments/SAM3_mask_to_polygons.py tmb_office_one_corridor")
        print("  python src/experiments/SAM3_mask_to_polygons.py tmb_office_corridor_bigger")
        sys.exit(1)

    map_name = sys.argv[1]
    mask_dir = _PROJECT_ROOT / "data" / "sam3_room_segmentation" / map_name
    density_dir = _PROJECT_ROOT / "data" / "density_image" / map_name
    output_path = mask_dir / f"{map_name}_polygons.json"

    # Load density image metadata
    metadata_path = density_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: metadata not found at {metadata_path}")
        sys.exit(1)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load density image for occupied area computation
    density_path = density_dir / f"{map_name}.png"
    density_img = cv2.imread(str(density_path), cv2.IMREAD_GRAYSCALE)
    if density_img is None:
        print(f"ERROR: could not read density image at {density_path}")
        sys.exit(1)

    h, w = density_img.shape[:2]
    print(f"Map: {map_name}")

    # Load all mask files
    mask_files = sorted(mask_dir.glob(f"{map_name}_mask_*.png"))
    if not mask_files:
        print(f"ERROR: no mask files found in {mask_dir}")
        sys.exit(1)
    print(f"Found {len(mask_files)} mask files")

    # Read all masks
    mask_images = []
    for mf in mask_files:
        idx = int(mf.stem.split("_")[-1])
        img = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            mask_images.append((idx, img))

    # Classify masks
    room_masks, building_mask = classify_masks(mask_images, density_img)

    if not room_masks:
        print("\nERROR: no room masks identified after filtering.")
        sys.exit(1)

    # Save classified masks to disk
    cv2.imwrite(str(mask_dir / f"{map_name}_building_outline.png"), building_mask)
    print(f"\n  Saved: {map_name}_building_outline.png")
    for idx, mask in room_masks:
        fname = f"{map_name}_room_{idx:02d}.png"
        cv2.imwrite(str(mask_dir / fname), mask)
        print(f"  Saved: {fname}")

    # Extract room polygons
    rooms = []
    room_union = np.zeros((h, w), dtype=np.uint8)
    print(f"\nExtracting room polygons:")
    for idx, mask in room_masks:
        room_union = cv2.bitwise_or(room_union, mask)
        poly_px = extract_polygon_from_mask(mask)
        if poly_px is None:
            print(f"  Mask {idx:02d}: no valid contour, skipping")
            continue

        poly_world = pixels_to_world_meters(poly_px, metadata)
        area_px = cv2.contourArea(poly_px.astype(np.float32).reshape(-1, 1, 2))
        print(f"  Mask {idx:02d}: {len(poly_px)} vertices, area={area_px:.0f} px")

        rooms.append({
            "mask_index": idx,
            "mask_file": f"{map_name}_mask_{idx:02d}.png",
            "label": f"room_{idx:02d}",
            "vertices_pixel": poly_px.tolist(),
            "vertices_world_meters": poly_world.tolist(),
            "n_vertices": len(poly_px),
            "area_pixels": float(area_px),
        })

    # Compute remaining area: building footprint minus room union
    remaining = cv2.bitwise_and(building_mask, cv2.bitwise_not(room_union))
    remaining_pixels = int(np.sum(remaining > 0))
    print(f"\nRemaining uncovered area: {remaining_pixels} px")

    cv2.imwrite(str(mask_dir / f"{map_name}_remaining.png"), remaining)
    print(f"  Saved: {map_name}_remaining.png")

    if remaining_pixels > MIN_CONTOUR_AREA_PX:
        # Extract polygon(s) from the remaining area
        contours, _ = cv2.findContours(remaining, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA_PX]

        for ci, contour in enumerate(valid_contours):
            perimeter = cv2.arcLength(contour, closed=True)
            eps = APPROX_EPS_FRAC * perimeter
            approx = cv2.approxPolyDP(contour, eps, closed=True)
            poly_px = approx.reshape(-1, 2).astype(float)
            poly_world = pixels_to_world_meters(poly_px, metadata)
            area_px = cv2.contourArea(poly_px.astype(np.float32).reshape(-1, 1, 2))
            print(f"  Remaining polygon {ci}: {len(poly_px)} vertices, area={area_px:.0f} px")

            rooms.append({
                "mask_index": -1,
                "mask_file": None,
                "label": f"remaining_{ci:02d}",
                "vertices_pixel": poly_px.tolist(),
                "vertices_world_meters": poly_world.tolist(),
                "n_vertices": len(poly_px),
                "area_pixels": float(area_px),
            })

    if not rooms:
        print("\nERROR: no valid polygons extracted.")
        sys.exit(1)

    # Save JSON output
    output = {
        "map_name": map_name,
        "source": "sam3_room_segmentation",
        "coordinate_system": "world_meters",
        "metadata_used": str(metadata_path),
        "rooms": rooms,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved {len(rooms)} room polygons to {output_path}")


if __name__ == "__main__":
    main()
