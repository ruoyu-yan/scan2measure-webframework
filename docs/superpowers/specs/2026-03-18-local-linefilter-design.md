# Design: Local 3D Line Filtering for Multi-Room Pose Estimation

## Problem

The multi-room pose estimation pipeline (`multiroom_pose_estimation.py`) fails on TMB_office1 (~2.3m error) and TMB_corridor_south1 (converges to wrong room entirely). The single-room pipeline (`pose_estimation_pipeline.py`) produces correct results for office1 when given only office geometry. The root cause: distant geometry from other rooms creates competing features in the XDF cost landscape, causing false local minima during coarse search and ICP refinement.

## Solution

Spatially filter the existing combined 3D line map so each panorama only sees local geometry during pose estimation. Uses Voronoi assignment (nearest panorama) with an overlap margin to partition 3D lines. All poses remain in the shared coordinate frame — no segmentation of raw point clouds, no re-running upstream pipeline stages, no coordinate stitching.

## Context

- **Single-room pipeline** (`pose_estimation_pipeline.py`): Works well for office1 with a dedicated point cloud. Produces t=[-4.67, 11.96].
- **Multi-room pipeline** (`multiroom_pose_estimation.py`): Office1 converges to t=[-3.08, 9.95] (~2.3m off). Corridor_south1 baseline converges to office location (16m off).
- **Approach B** (polygon translation filter from `experiment_polygon_prior.py`): Fixed corridor_south1 wrong-room problem (16m→0.82m) but did not improve office1 (same result). Approach B only filters translation candidates; the 3D distance functions still include all geometry.
- **Alignment** (`global_alignment.json` from `align_polygons_demo5.py`): Correctly identifies which panorama is in which room. Provides approximate panorama positions in density image pixel coordinates.

## Architecture

### File

| File | Action | Purpose |
|------|--------|---------|
| `src/experiment_local_linefilter.py` | **Create** | Standalone experiment: local 3D line filtering per panorama |

No modifications to existing library modules.

### Data Dependencies

- `data/debug_renderer/tmb_office_one_corridor_dense/3d_line_map.pkl` — combined 3D line map
- `data/reconstructed_floorplans_RoomFormer/tmb_office_one_corridor_dense/global_alignment.json` — panorama-to-room assignments + positions
- `data/density_image/tmb_office_one_corridor_dense/metadata.json` — coordinate transform for pixel→3D conversion
- `data/pano/2d_feature_extracted/<pano>_v2/fgpl_features.json` — per-panorama 2D features (unchanged)

### Panorama Position Derivation

Convert `camera_pose_global` (pixel coordinates in density image) to raw 3D world coordinates using the established transform:

```python
def pixels_to_raw_3d(pixel_coord, metadata):
    R = np.array(metadata['rotation_matrix'])
    min_coords = np.array(metadata['min_coords'])
    offset = np.array(metadata['offset'])
    max_dim = metadata['max_dim']
    scale = metadata['image_width'] - 1
    aligned_mm = np.array(pixel_coord) / scale * max_dim - offset[:2] + min_coords[:2]
    raw_3d = R.T @ np.array([aligned_mm[0], aligned_mm[1], 0.0]) / 1000.0
    return raw_3d[:2]
```

Approximate ground truth positions (from global_alignment):
- TMB_office1: [-4.20, 10.71]
- TMB_corridor_south1: [1.82, -5.48]
- TMB_corridor_south2: [1.07, 4.77]

## 3D Line Filtering

### Voronoi + Margin

For each 3D element (line segment or intersection point), compute its position (midpoint for lines, XYZ for intersections), then:

1. **Voronoi assignment**: Assign to the nearest panorama based on XY distance
2. **Overlap margin**: For each panorama, also include elements assigned to other panoramas whose position is within `OVERLAP_MARGIN` meters of this panorama

### What Gets Filtered

| Array | Shape | Filter by | Used for |
|-------|-------|-----------|----------|
| `dense_starts/ends/dirs` | (N_dense, 3) | Line midpoint XY | LDF-3D precomputation |
| `starts/ends` (sparse) | (N_sparse, 3) | Line midpoint XY | Translation grid generation |
| `inter_3d` | (N_inter, 3) | Intersection XY | ICP refinement |
| `inter_3d_mask` | (N_inter, 3) | Same rows as inter_3d | ICP refinement |
| `inter_3d_idx` | (N_inter, 2) | Same rows as inter_3d | ICP line direction lookup |

### What Stays Unchanged

- **2D features**: Per-panorama, already independent
- **`principal_3d`**: Global property of the combined line map. Used as-is in `precompute_xdf_3d` (canonical rotation) and `build_rotation_candidates`. Not filtered or recomputed per panorama.
- **Full `dense_dirs`**: Passed to `refine_pose` for line direction lookup via `inter_3d_idx`. When intersections are filtered (rows removed), the remaining `inter_3d_idx` values still contain indices into the original dense line array. Since `dense_dirs` is passed unfiltered, these lookups remain valid. The filtered dense lines used for LDF precomputation are a separate variable — `dense_dirs` for refine_pose must always be the original full array.

### Parameters

- `OVERLAP_MARGIN = 2.0` meters — buffer beyond Voronoi boundary
- All algorithm parameters identical to `multiroom_pose_estimation.py`: TOP_K=10, NUM_TRANS=1700, XDF_INLIER_THRES=0.1, POINT_GAMMA=0.2, CHAMFER_MIN_DIST=0.3, etc. `NUM_TRANS=1700` is kept unchanged — `generate_translation_grid` adapts to the input data range, so filtered sparse lines will naturally produce fewer candidates covering only the local area.

## Script Flow

```
Phase A: One-time setup
  A1. Load 3D line map from 3d_line_map.pkl
  A2. Load panorama positions from global_alignment.json → convert to 3D world coords
  A3. Generate icosphere query points
  A4. Compute Voronoi assignment for all dense lines, sparse lines, and intersections

For each PANO_NAME:
  B1. Filter dense lines, sparse lines, and intersections (Voronoi + margin)
  B2. Generate translation grid from filtered sparse lines
  B3. Precompute LDF-3D + PDF-3D for filtered dense lines + translations
  B4. Load 2D features (unchanged)
  B5. Compute 2D intersections, build 24 rotation candidates, rearrange
  B6. XDF coarse search (from precomputed)
  B7. ICP refinement on top-K (pass full dense_dirs for direction lookup)
  B8. Compute metrics (n_tight, avg_dist)
  B9. Generate side_by_side.png (using full sparse lines for context)

Print comparison table + save results JSON
```

Key difference from `multiroom_pose_estimation.py`: 3D precomputation (B2-B3) moves inside the per-panorama loop. Trade-off: 3x precomputation instead of 1x shared, but each is on a smaller line set, so total time should be comparable.

## Output

```
data/pose_estimates/multiroom/local_filter_results.json
data/pose_estimates/multiroom/local_filter_vis/<pano>/side_by_side.png
```

Results JSON includes per-panorama: n_tight, avg_dist, n_translations, n_dense_lines, n_sparse_lines, n_intersections, translation vector.

## Verification

```bash
conda run -n scan_env python src/experiment_local_linefilter.py
```

1. **Office1 should improve**: Currently 2.3m from single-room result. With local filtering, should approach t≈[-4.67, 11.96].
2. **Corridor_south2 should stay good**: Currently 0.50m from derived GT.
3. **Corridor_south1 should be at least as good as B_only**: Currently 0.82m from derived GT.
4. Print comparison table: local_filter vs multiroom baseline (loaded from `polygon_ablation_results.json` if it exists, otherwise print local_filter results standalone).
5. Visual inspection of side_by_side.png for all 3 panoramas.

## Risks

- **Voronoi boundary cuts through geometry**: A wall shared between office and corridor might be split. The 2m overlap margin mitigates this — shared walls are typically within 2m of both panorama positions.
- **Too few lines in a segment**: If a panorama's Voronoi cell contains very few sparse lines, the translation grid may be too sparse. Unlikely given room sizes, but the script should log line counts for diagnosis.
- **Alignment position inaccuracy**: Panorama positions from `global_alignment.json` only need to be roughly correct (within ~3m) for Voronoi assignment to work. The alignment correctly identifies rooms, so positions are adequate for this purpose.
