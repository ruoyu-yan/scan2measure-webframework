# SAM3 Panoramic Image Processing — Design Spec

## Goal

Use SAM3 to extract room footprint polygons from panoramic images, replacing LGT-Net's depth+ratio prediction. SAM3 handles segmentation (floor or wall detection), and LGT-Net's equirectangular→XYZ deprojection math is reused for correct relative polygon scaling.

## Context

- SAM3 has already been validated for room segmentation on density images (`SAM3_room_segmentation.py`)
- LGT-Net currently produces room polygons from panos via `depth[256]` + `ratio` (scalar) → `depth2xyz()` → `corners2boundaries()`
- The polygons feed into `align_polygons_demo5.py` which matches them to density-image polygons via Hungarian algorithm
- The critical requirement: relative polygon sizes must be correct (larger rooms → bigger polygons) for the alignment to work

## Pipeline Overview

A single script `SAM3_pano_processing.py` in `src/` that:

1. Loads each pano from `data/sam3_pano_processing/`
2. Runs SAM3 twice per pano — floor prompt and wall prompt
3. Extracts floor-wall boundary from each mask type
4. Resamples boundary to 256 equidistant longitude steps
5. Projects boundary to XZ polygon using LGT-Net's conversion math
6. Saves output as `layout_corners` JSON (compatible with `align_polygons_demo5.py`)
7. Generates 6-panel comparison visualization

## Approach A: Floor Mask

### SAM3 Segmentation
- Text prompts to try: `"floor"`, `"ground"`, `"floor surface"`
- Confidence threshold: start with 0.5 (natural images, unlike dark density images that needed 0.1)
- No preprocessing needed for initial test
- **Risk: equirectangular distortion.** SAM3 is trained on perspective images. Equirectangular panos have severe distortion near the poles (the floor is a warped band at the bottom). The equator region (where the floor-wall boundary typically lies) has the least distortion, which is favorable. If SAM3 fails on equirectangular images, a fallback plan is to decompose the pano into perspective crops (as `pano_processing_virtual_camerasV2.py` already does), run SAM3 on each crop, and stitch masks back. This is out of scope for the initial script but noted as Plan B

### Boundary Extraction
- For each column `px` in the mask, scan top-to-bottom, find the first row where floor mask = 1
- This yields the **upper edge** of the floor region = floor-wall junction
- If a column has no floor pixels (furniture occlusion), linearly interpolate from nearest valid neighboring columns
- If SAM3 returns multiple floor masks (e.g. adjacent rooms visible through doorways), use only the largest mask by pixel count
- If SAM3 returns no floor masks at all, log a warning and skip this pano

## Approach B: Wall Masks

### SAM3 Segmentation
- Text prompts to try: `"wall"`, `"room wall"`, `"interior wall"`
- SAM3 may return multiple wall masks (one per wall segment) — combine with logical OR

### Boundary Extraction
- For each column `px` in the combined wall mask, scan bottom-to-top, find the first row where wall mask = 1
- This yields the **lower edge** of the wall region = floor-wall junction
- If a column has no wall pixels (doorways, windows), linearly interpolate from nearest valid neighbors
- If SAM3 returns no wall masks at all, log a warning and skip this pano for this approach

## Shared Post-Processing

### Boundary Smoothing
- Apply 1D median filter (window ~15px) to smooth jagged mask edges
- Clamp boundary to bottom half of image only (v > 0.5, below the equator line). If any boundary points are at or above the equator, clamp them to v = 0.5 + 1 pixel
- Resample from full image width down to 256 equidistant longitude steps

### Geometric Projection (reuses LGT-Net math from `LGT-Net/utils/conversion.py`)

1. **Construct pixel pairs**: For each of the 256 resampled columns, create `(col_px, row_px)` pairs from the boundary. These are fed as a `(256, 2)` array to the conversion functions
2. **Pixel → Longitude/Latitude**: `pixel2lonlat(pixel_pairs, w, h)` → `(lon, lat)` where `lat ∈ (0, π/2]` for floor boundary
3. **Longitude/Latitude → XYZ on floor plane**: `lonlat2xyz(lonlat, plan_y=1)` — projects onto Y=1 plane, giving `depth = 1/tan(lat)` per point
4. **XZ extraction + Z-flip**: `polygon_2d = xyz[:, [0, 2]]` then `polygon_2d[:, 1] *= -1` — the Z-negation matches `LGT-Net_inference_demo2.py`'s `save_simple_layout_json` which applies `z * -1` to align with the coordinate convention used by `align_polygons_demo5.py`
5. **Polygon simplification**: Douglas-Peucker via `shapely.simplify()` on a `LinearRing` constructed from the 256 points. The output is an implicitly-closed polygon (first point NOT repeated) matching LGT-Net's `layout_corners` convention

### Why Relative Scaling Is Correct

The projection `depth = 1/tan(lat)` is purely geometric. A larger room pushes the floor-wall boundary closer to the equator (smaller latitude), producing larger depth values and thus a larger polygon. No learned ratio is needed — the angular information in the equirectangular image encodes relative room size directly. All polygons use `plan_y=1` (normalized camera-to-floor = 1 unit, unitless). This is acceptable because `align_polygons_demo5.py` computes its own relative scale factor from bounding-box-diagonal ratios — it does not depend on absolute metric units.

## Output Format

### Per-pano JSON (one per approach)
```json
{
  "layout_corners": [[x1, z1], [x2, z2], ...],
  "source": "sam3_floor" or "sam3_wall"
}
```
Saved to `data/sam3_pano_processing/{pano_stem}_sam3_floor_layout.json` and `{pano_stem}_sam3_wall_layout.json`.

For downstream integration with `align_polygons_demo5.py`, the JSONs can be copied/symlinked into `data/pano/LGT_Net_processed/{room_name}/` where `load_lgt_poly()` expects them. The `load_lgt_poly()` function reads the `layout_corners` key, which this output provides in the same format. No code changes to `align_polygons_demo5.py` are needed.

**Which approach to use downstream**: This script is an experiment to compare floor vs wall segmentation. After visual inspection of the comparison figures, the user will decide which approach produces better polygons and use those JSONs for the alignment step.

### Comparison Visualization

6-panel figure per pano saved to `data/sam3_pano_processing/{pano_stem}_comparison.png`:

| Panel | Content |
|-------|---------|
| 1 | Original equirectangular pano |
| 2 | Floor mask overlay + extracted boundary line (red) |
| 3 | Wall mask overlay + extracted boundary line (blue) |
| 4 | Both boundary curves overlaid on pano (red=floor, blue=wall) |
| 5 | Projected XZ polygon from floor approach (camera at origin) |
| 6 | Projected XZ polygon from wall approach (camera at origin) |

## Dependencies

- SAM3 model (already available at `sam3/weights/sam3.pt`)
- LGT-Net's `utils/conversion.py` — imported for `pixel2lonlat`, `lonlat2xyz` (or the math is reimplemented as standalone NumPy functions to avoid LGT-Net model dependencies)
- `shapely` for polygon simplification
- `scipy.ndimage.median_filter` for boundary smoothing
- `matplotlib` for comparison figures
- Conda environment: `sam3`

## Configuration

- `CAMERA_HEIGHT = 1.4` — fixed camera height in meters (for reference/metadata, not used in relative scaling since `plan_y=1`)
- `CONFIDENCE_THRESHOLD = 0.5` — SAM3 confidence (may need tuning)
- `MEDIAN_FILTER_WINDOW = 15` — boundary smoothing window
- `BOUNDARY_RESOLUTION = 256` — number of equidistant longitude samples
- `SIMPLIFY_TOLERANCE = 0.05` — Douglas-Peucker tolerance in camera-height units (0.05 × 1.4m ≈ 7cm at camera height 1.4m)

## Scope Exclusions

- This script does NOT run `align_polygons_demo5.py` — that is a separate downstream step
- No LGT-Net model inference — we only reuse the coordinate conversion math
- No density image processing — that remains handled by `SAM3_room_segmentation.py`
