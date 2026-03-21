# Three-Way Fusion Comparison for SAM3 Floor/Ceiling Boundaries

## Problem

The current Approach 1 fusion in `SAM3_footprint_comparison.py` uses rolling min/max with a 51-column window to detect floor/ceiling boundary corruption. This fails for obstacles wider than the window (e.g., cabinets in TMB_office1): the rolling minimum sees only corrupted columns in the interior, so corruption goes undetected and the floor's indentation passes through to the fused polygon, producing an L-shape instead of the true rectangle.

## Goal

Replace the single fusion strategy with three variants (A, B, C) that each produce a polygon, enabling visual and metric comparison across 6 test panoramas. The downstream consumer (`align_polygons_demo5.py`) rescales polygons, so **correct shape and proportions** matter more than absolute metric accuracy.

The old `approach1_layout.json` is removed — replaced by `approach_A/B/C_layout.json`. Stale `approach1_layout.json` files from previous runs will be deleted at the start of each panorama's processing.

## Physical Invariant

Obstacles (furniture, light fixtures, columns) can only **shrink** the visible room extent from either floor or ceiling perspective. Neither boundary can show the wall further away than it truly is. Any fusion strategy can exploit this one-directional property.

## Fusion Variants

### Approach A — Per-column optimistic

Per pixel column, take whichever boundary shows the wall further from the camera:

```python
ceiling_mirrored = img_h - ceiling_boundary_raw
fused = np.minimum(floor_boundary_raw, ceiling_mirrored)
```

Lower row = wall further away. `np.minimum` selects the more optimistic (further wall) estimate. Column-occluded columns (from SAM3 "column" mask) are marked as gaps and interpolated through.

**Fallback**: if only one boundary is available (no ceiling mask), use the floor boundary alone (same as existing behavior).

- Parameters: none
- Strengths: trivially correct by the one-directional corruption property; handles obstacles of any width
- Weaknesses: vulnerable to mask noise that erroneously jumps toward the equator (mitigated by downstream median filter in `smooth_and_resample`)

### Approach B — Dual XZ radial maximum

Project each boundary to XZ space independently at its own height, avoiding the pixel-space mirroring approximation. This variant has a fundamentally different pipeline — it bypasses `smooth_and_resample` and `boundary_to_polygon`, working directly in XZ space. A new function `approach_B_dual_xz()` handles the full path from raw boundaries to XZ polygon.

**Pre-processing**: apply median filter (window=15, wrap mode) to both raw boundaries for noise reduction, but do NOT apply the equator clamping from `smooth_and_resample` (ceiling rows are above the equator). Column-occluded columns are excluded from both boundaries before projection; the polygon naturally bridges those gaps.

**Steps**:

1. Median-filter both raw boundaries (noise reduction only, no clamping)
2. Resample both to `BOUNDARY_RESOLUTION` (256) equidistant columns
3. Floor boundary → XZ via `_pixel2lonlat` + `_lonlat2xyz(plan_y=1.0)`, then z-flip: `z_f = -z_f`
4. Ceiling boundary → XZ via `_pixel2lonlat` + `_lonlat2xyz(plan_y=-1.0)`, then z-flip: `z_c = -z_c`
5. Compute azimuth per point: `azimuth = atan2(x, z)` (matches existing `lon` convention)
6. Bin both point sets into `BOUNDARY_RESOLUTION` azimuth bins over `[-pi, pi)`
7. Per bin: if both have points, take the one with larger `sqrt(x² + z²)`; if only one has a point, use that; if neither, interpolate from neighbors
8. Form polygon from selected points in azimuth order
9. Simplify with `SIMPLIFY_TOLERANCE` (0.05m), then Manhattan-regularize

**Fallback**: if fewer than 3 valid bins remain after selection, skip Approach B for this panorama.

- Parameters: none beyond existing `BOUNDARY_RESOLUTION` and `SIMPLIFY_TOLERANCE`
- Strengths: no mirroring approximation; each boundary projects at its correct height; natural XZ-space operation
- Weaknesses: more complex implementation; azimuth binning can create artifacts at bin boundaries; different pipeline path than A/C makes direct boundary-overlay comparison harder

### Approach C — Ceiling-primary with floor expansion

Use the ceiling boundary as the baseline. Only adopt the floor value where it shows a **larger** room:

```python
ceiling_mirrored = img_h - ceiling_boundary_raw
fused = np.copy(ceiling_mirrored)
floor_shows_larger = floor_boundary_raw < ceiling_mirrored
fused[floor_shows_larger] = floor_boundary_raw[floor_shows_larger]
```

Column-occluded columns interpolated through as in Approach A.

Mathematically equivalent to Approach A (`min` of two values), but implemented with explicit ceiling-first logic. The debug visualization shows which columns used ceiling vs floor via a per-column boolean source array `used_floor`.

**Fallback**: same as Approach A — if no ceiling mask, use floor boundary alone.

- Parameters: none
- Strengths: same correctness guarantee as A; ceiling-dominant framing matches the intuition that ceiling is more reliable; debug visualization shows per-column source attribution
- Weaknesses: same vulnerability to mask noise as A

## Shared Components (unchanged)

- **SAM3 segmentation**: 3 passes per panorama (floor, ceiling, column) — shared across all variants
- **Column mask handling**: columns detected via SAM3 "column" prompt; covered columns marked as gaps and linearly interpolated (used by A, B, and C)
- **Manhattan regularization**: pre-simplify at 0.3m, force-snap all edges to two principal directions, merge collinear edges, reconstruct corners, Shapely validation, ±8m depth clipping
- **Approach 2 (morphological)**: union mask + closing + hole-fill pipeline — kept as-is for comparison
- **LGT-Net baseline**: loaded from `data/pano/LGT_Net_processed/<stem>/<stem>_layout.json`

## Implementation Notes

- `approach1_fuse_boundaries` is **removed** entirely, replaced by three new functions: `approach_A_optimistic`, `approach_B_dual_xz`, `approach_C_ceiling_primary`
- `process_pano` gains structural changes: A and C follow the existing pipeline (`fuse → smooth_and_resample → boundary_to_polygon → manhattan_regularize`), but B has its own pipeline path that goes directly from raw boundaries to an XZ polygon
- `save_detail_figure` is **rewritten** (different parameters, 5 rows instead of 4, different plot logic per row)
- `save_summary_figure` is updated from 3 methods to 5 methods; Approach B's polygon is reverse-projected to pixel space for the panorama overlay using `xz_polygon_to_pixel_boundary`
- The Approach 2 mask debug view (union vs cleaned) from the old Row 0 Col 1 is dropped — Approach 2 is unchanged and only appears in the Row 4 comparison

## Pipeline per Panorama

```
SAM3 segment("floor") ──┐
SAM3 segment("ceiling") ─┤── shared masks
SAM3 segment("column") ──┘
         │
    ┌────┴─────────────────────┐
    │                          │
    ▼                          ▼
extract_floor_boundary    extract_ceiling_boundary
    │                          │
    ├──────────┬───────────────┤
    │          │               │
    ▼          ▼               ▼
 Approach A  Approach B    Approach C
 (pixel min) (XZ radial)  (ceil-primary)
    │          │               │
    ▼          │               ▼
 smooth &     │            smooth &
 resample     │            resample
    │          │               │
    ▼          │               ▼
 boundary_    │            boundary_
 to_polygon   │            to_polygon
    │          │               │
    ▼          ▼               ▼
 manhattan_regularize (shared)
    │          │               │
    ▼          ▼               ▼
 JSON out    JSON out       JSON out
```

## Output Structure

```
data/sam3_pano_processing/footprint_comparison/
    <stem>/
        approach_A_layout.json      # Per-column optimistic
        approach_B_layout.json      # Dual XZ radial maximum
        approach_C_layout.json      # Ceiling-primary + floor expansion
        approach2_layout.json       # Morphological (unchanged)
        <stem>_detail.png           # 5-row debug figure
    comparison_summary.png          # 6-pano summary
```

## Visualization

### Colors (consistent across all figures)

| Method | Color | Line style |
|--------|-------|------------|
| Approach A | green | solid |
| Approach B | cyan | solid |
| Approach C | orange | solid |
| Approach 2 (morph) | blue | dashed |
| LGT-Net | red | dashed |

### Detail figure (`<stem>_detail.png`) — 5 rows x 2 columns

| Row | Left | Right |
|-----|------|-------|
| 0 | SAM3 masks overlay (floor=red, ceiling=purple, column=blue) | Floor raw (red) + ceiling raw (blue) + ceiling mirrored (blue dashed) boundaries on pano |
| 1 | Approach A: fused boundary (green) on pano | Approach A: XZ polygon (dashed=raw, solid=regularized) |
| 2 | Approach B: floor XZ points (red) + ceiling XZ points (blue) + merged polygon (cyan) | Approach B: final XZ polygon (dashed=raw, solid=regularized) |
| 3 | Approach C: fused boundary on pano, colored per source (orange=ceiling-used, green=floor-used) | Approach C: XZ polygon (dashed=raw, solid=regularized) |
| 4 | All polygons reverse-projected on pano (colors per table above, thin lines) | All XZ polygons + area/corner metrics text box |

### Summary figure (`comparison_summary.png`) — 6 rows x 2 columns

| Column | Content |
|--------|---------|
| Left | Panorama with all 5 polygon overlays (reverse-projected, colors per table above) |
| Right | XZ polygon comparison (A, B, C, Morph, LGT) with area/corner annotations |

## Configuration

No new hyperparameters. The existing constants remain for Manhattan regularization and morphological cleanup. The three fusion variants are parameter-free.

## Verification

1. Script runs without error on all 6 panoramas
2. `approach_A_layout.json`, `approach_B_layout.json`, `approach_C_layout.json` exist for each pano with valid `layout_corners`
3. TMB_office1: all three fusion variants should produce a roughly rectangular polygon (no L-shape from cabinet indentation)
4. Corridor polygons remain elongated rectangles
5. Detail PNGs show meaningful differences between the variants (especially B vs A/C in XZ space)
6. A and C should produce near-identical polygons (mathematical equivalence), confirming the implementation is correct
