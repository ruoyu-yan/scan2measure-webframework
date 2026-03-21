# ROADMAP — scan2measure-webframework

> **Goal**: Color an RGB-less TLS point cloud of an indoor environment using panoramic images taken in the same space, then build a measurable virtual tour from the result.

> **Test case**: TMB office floor

---

## Phase 1: Room-Level Pano-to-Room Assignment (Jigsaw Puzzle Solver) 🟡

**Status**: 🟡 Core pipeline implemented, accuracy limited by upstream polygon quality.

Determine which panorama was taken inside which physical room by matching single-room footprints (from panos) against a multi-room floorplan (from point cloud).

### 1.1 Point Cloud Floorplan Extraction ✅

- `generate_density_image.py` — RANSAC floor plane detection, Manhattan axis alignment, ceiling detection, 2D density projection. Produces PNG + rotation metadata JSON.
- `RoomFormer_inference.py` — ResNet50+Transformer model on density image (not directly on point cloud). Outputs `predictions.json` with room polygon vertices in pixel coordinates.
- `map_RoomFormer_results_to_3d.py` — Inverse-projects pixel polygons to world coordinates using density image metadata. Segments point cloud by room assignment.

### 1.2 Panorama Room Footprint Extraction ✅

- `LGT-Net_inference_demo2.py` — Vanishing point detection, Manhattan frame alignment, LGT-Net inference. Outputs `layout_corners` in meters (floor polygon of the room visible in the panorama).

### 1.3 Scale Estimation and Polygon Matching ✅

- `polygon_scale_calculation.py` — Consensus pixel-to-meter scale via histogram voting over pairwise corner distances between RoomFormer (pixels) and LGT-Net (meters) polygons.
- `align_polygons_demo5.py` — Hungarian algorithm matching: tests candidate scales, 4 discrete rotations (0/90/180/270 deg), brute-force vertex-to-vertex snapping. Outputs `global_alignment.json` with per-pair transformation parameters and match errors.

### 1.4 Improve Footprint Quality with SAM3 🟡

Both LGT-Net and RoomFormer footprints have limited accuracy. Investigate using Segment Anything Model 3 for extracting room boundaries from:
- Panoramic images (replacing or augmenting LGT-Net footprints)
- Density images (replacing or augmenting RoomFormer polygons)

**Current state**: Panoramic footprint extraction is production-ready. SAM3 floor/ceiling fusion with height correction produces clean Manhattan-regularized polygons on all 6 test panoramas. Density image approaches still pending evaluation.

#### Panoramic image approaches (replacing/augmenting LGT-Net) ✅

**Production script**: `SAM3_pano_footprint_extraction.py` — the validated approach. Per panorama:
1. SAM3 text-prompted segmentation for floor, ceiling, and column masks
2. Floor/ceiling boundary extraction (topmost floor pixel, bottommost ceiling pixel per column)
3. Height ratio estimation from boundary geometry (median `tan(angle_ceil)/tan(angle_floor)` — compensates for camera not being at room midpoint; typical ratio ~1.7)
4. Per-column optimistic fusion: `min(floor_row, height_corrected_ceiling_row)` — selects whichever boundary shows the wall further from camera, exploiting the physical invariant that obstacles can only shrink visible room extent
5. Column-occluded columns (from SAM3 "column" mask) marked as gaps and interpolated
6. Smooth, resample, equirectangular deprojection to XZ polygon, Manhattan regularization
7. Output: `<stem>/layout.json` + `<stem>/debug.png` in `data/sam3_pano_processing/`

Key insight: naive ceiling mirroring (`h - ceil_bnd`) assumes camera is at room midpoint, which it never is (~1m from floor, ~1.7m from ceiling). Without height correction, the fusion silently ignores the ceiling boundary entirely. The height ratio is estimated directly from the boundary data (no configuration needed).

**Results on 6 test panoramas**: TMB_office1 produces a clean 4-corner rectangle (LGT-Net gives 12 corners with artifacts). Corridors produce correct elongated rectangles. All polygons are Manhattan-regularized.

**Comparison experiment**: `SAM3_footprint_comparison.py` — tested three fusion variants:
- Approach A (per-column optimistic): selected as production method
- Approach B (dual XZ radial maximum): projects floor/ceiling independently to XZ, takes radial max per azimuth bin. Slightly different results but more complex pipeline
- Approach C (ceiling-primary with floor expansion): mathematically equivalent to A, confirms correctness
- Also compared against morphological cleanup (Approach 2) and LGT-Net baseline

**Earlier exploration scripts**:
- `SAM3_pano_processing.py` — Initial floor/wall boundary extraction pipeline (predecessor to footprint_extraction)
- `SAM3_pano_raw_segmentation.py` — Multi-prompt raw segmentation comparison

#### Density image approaches (replacing/augmenting RoomFormer) 🟡

- `SAM3_room_segmentation.py` — CLAHE+invert preprocessing + "floor plan" text prompt. Tested on 3 datasets (03251, 03264, tmb_office_corridor_hall). Outputs per-mask binary PNGs + 3-panel comparison figures. Results in `data/sam3_room_segmentation/`.
- `SAM3_room_extraction_test.py` — Systematic 3-approach experiment:
  - Approach 1: Target walls ("bright line", "white line", "wall") → post-process to room regions
  - Approach 2: Target room areas directly ("gray area", "room", "rectangle", "floor plan")
  - Approach 3: Two-pass hybrid (footprint + walls → rooms)
  - Tests 3 preprocessing modes (raw, clahe, inverted) on 4 datasets. Results in `data/sam3_room_extraction/`.

All SAM3 scripts require `conda run -n sam3`.

**Open questions**:
- ~~Can SAM3 reliably segment room floor boundaries from equirectangular panoramas?~~ → ✅ Yes. Floor/ceiling fusion with height correction produces clean polygons on all 6 test panos.
- ~~Does SAM3 on density images outperform RoomFormer's learned polygon detection?~~ → Multiple approaches tested; results need quantitative comparison against RoomFormer baselines
- ~~What prompting strategy works best?~~ → For panoramas: "floor" + "ceiling" + "column" prompts with fusion. For density images: "floor plan" on CLAHE-inverted images appears most promising.

**Next**: Integrate SAM3 panoramic footprints into the polygon matching pipeline (`align_polygons_demo5.py`) as an alternative/complement to LGT-Net. Evaluate density image approaches quantitatively against RoomFormer.

---

## Phase 2: Camera Pose Estimation (Panoramic Localization) 🟡

**Status**: 🟡 Single-room and multi-room pose estimation implemented and validated. Some improvements pending.

Estimate the exact 6-DoF camera pose (position + orientation) for each panorama within the point cloud coordinate frame.

### 2.1 3D Feature Extraction ✅

- `point_cloud_geometry_baker_V4.py` — Thin wrapper around the `3DLineDetection` C++ binary. PLY → XYZ → `LineFromPointCloud` → parse `lines.obj` → `room_geometry.pkl`. ~100x faster than the legacy V3 plane detection approach.
- `cluster_3d_lines.py` — Orchestrator that loads `room_geometry.pkl` and runs FGPL-faithful 3D line clustering via `line_clustering_3d.py`: principal direction voting (icosphere) → line classification → 3D intersection finding → writes `3d_line_map.pkl` (dense/sparse line tensors, principal_3d, pre-computed 3D intersections) + colored OBJ debug files.
- `line_clustering_3d.py` — Library module: `vote_principal_directions()`, `classify_lines_3d()`, `find_intersections_3d()`, `build_intersection_masks()`. Mirrors FGPL `map_utils.py` + `edge_utils.py` algorithms.

### 2.2 2D Feature Extraction ✅ (cleanup pending)

- `image_feature_extractionV2.py` — Orchestrator calling three library modules:
  - `pano_line_detector.py` — 26-view HorizonNet-style decomposition, LSD detection, exact pinhole back-projection to sphere, 3-pass colinear segment merging
  - `line_analysis.py` — Icosphere voting for 3 vanishing points, line classification into principal groups, great-circle arc intersection finding
  - `sphere_geometry.py` — Icosphere generation, equirectangular/sphere coordinate conversions
- Outputs `fgpl_features.json` (sphere-projected lines + principal directions + intersections)

**Next**: Clean up the 2D feature extraction pipeline — simplify orchestration, reduce redundancy, improve robustness for diverse panorama inputs (corridors, rooms with poster-heavy walls, etc.).

### 2.3 Single-Room Pose Estimation ✅

**Canonical implementation**: `pose_estimation_pipeline.py` — modular 10-stage FGPL-faithful pipeline.

Orchestrator that imports from four library modules:
- `xdf_distance.py` — LDF/PDF sphere distance functions, 2D intersection finding with line-pair index tracking
- `pose_search.py` — 24 rotation candidates (SVD Procrustes), adaptive quantile translation grid (~1700 pts), canonical-frame XDF coarse search with `single_pose_compute` optimization
- `pose_refine.py` — Two-phase sphere ICP (Phase 1: translation via grouped mutual-NN, Phase 2: rotation via YPR with line direction alignment)
- `visualize_pose.py` — Side-by-side equirectangular overlay rendering

Consumes pre-computed features from upstream:
- **3D**: `3d_line_map.pkl` (from `cluster_3d_lines.py`) — dense/sparse line tensors, principal_3d, pre-computed 3D intersections
- **2D**: `fgpl_features.json` (from `image_feature_extractionV2.py`) — sphere-projected lines, principal_2d

Key advantages over the original FGPL (documented in `FGPL_reimplementation_comparison.md`):
- 642 XDF query points (vs 12 in original) for stronger discrimination
- Top-K=10 with rotation diversity (vs single candidate) to avoid premature rotation commitment
- Post-refinement quality selection by `(-n_tight, avg_dist)` ranking
- `single_pose_compute` approximation matching FGPL paper's Theorem 1

**Archived**: `feature_matchingV2.py` (monolithic 9-stage, 40K) moved to `Archive/`. It consumed the older V1 input format (`room_geometry.pkl` + `extracted_2d_lines.json`), used exact per-rotation LDF-2D (no `single_pose_compute`), uniform translation grid, and `top_k=1` with no multi-candidate refinement. The core math (LDF, PDF, XDF cost, intersection computation, ICP) was identical but the orchestration was less advanced.

**Validation**: `data/pose_estimates/TMB_office1/vis/side_by_side.png` shows precise alignment of 2D (green) and projected 3D (cyan) line features. The native FGPL result (`side_by_side_native.png`) shows visibly worse alignment.

### 2.4 Pose Estimation Visualization Improvement 🟡

`pose_estimation_pipeline.py` now generates three visualizations:
- **Side-by-side**: 2D lines (green) + projected 3D wireframe (cyan) overlay — shows feature matching quality
- **Reprojection**: Depth-coded point cloud projected onto panorama — shows pose correctness directly
- **Top-down**: Camera position + orientation arrow on density image with RoomFormer room polygons

Still missing:
- **Point cloud overlay**: Textured sphere at camera position inside 3D point cloud
- **Before/after comparison**: Side-by-side of estimated vs ground-truth pose (if available)

### 2.5 Multi-Room Pose Estimation ✅

`multiroom_pose_estimation.py` — orchestrator that estimates camera poses for multiple panoramas against a shared multi-room 3D wireframe map. Zero imports from `panoramic-localization/`.

**Architecture**: Split `xdf_coarse_search` in `pose_search.py` into:
- `precompute_xdf_3d()` — one-time 3D canonical frame + LDF/PDF precomputation (~10s)
- `xdf_coarse_search_from_precomputed()` — per-panorama 2D computation + cost + top-K (~1s each)

**Baseline results** (test case: `tmb_office_one_corridor_dense` map, 3 panoramas):
- TMB_office1: n_tight=283, avg_dist=0.039 — degraded by cross-room geometry
- TMB_corridor_south2: n_tight=185, avg_dist=0.024
- TMB_corridor_south1: n_tight=70, avg_dist=0.121 — converged to wrong room

### 2.6 Consolidate Pose Estimation Codebase ✅

Resolved: `pose_estimation_pipeline.py` (modular) is the canonical single-room implementation. `multiroom_pose_estimation.py` is the canonical multi-room implementation. `feature_matchingV2.py` (monolithic) archived to `Archive/`. The `PnL_solver.py` module is now unused by the active pipeline (superseded by `pose_refine.py`) but kept for reference.

### 2.7 Jigsaw-Informed Pose Estimation Improvements 🟡

#### Approach B: Jigsaw-guided translation prior ✅ (experiment)

`experiment_polygon_prior.py` — filters translation candidates using RoomFormer polygon boundaries from `global_alignment.json`. Fixed corridor_south1 wrong-room problem (16m→0.82m) but did not improve office1 (same result as baseline). Only filters translation candidates; 3D distance functions still include all geometry.

#### Approach D: Local 3D line filtering ✅ (experiment, validated)

`experiment_local_linefilter.py` — Voronoi-based spatial filtering of the combined 3D line map. Each panorama only sees local geometry (nearest-panorama assignment + 2.0m overlap margin) during LDF/PDF precomputation and ICP refinement. Moves 3D precomputation inside the per-panorama loop.

**Results** (dramatic improvement over baseline):
- TMB_office1: n_tight=408, avg_dist=0.029, t=[-4.68, 11.97] — matches single-room target [-4.67, 11.96] to sub-cm
- TMB_corridor_south1: n_tight=73, avg_dist=0.018, t=[1.87, -3.63]
- TMB_corridor_south2: n_tight=114, avg_dist=0.022, t=[0.80, 5.18]

Generates full visualization suite per panorama: `side_by_side.png`, `reprojection.png`, `topdown.png`.

**Next**: Integrate local line filtering into the production `multiroom_pose_estimation.py` pipeline.

#### Approach A: Line-length weighting in XDF cost ⬜

Structural lines (walls, ceiling, floor) are long great-circle arcs on the sphere. Poster/clutter lines are short, clustered segments. Currently every classified line contributes equally to LDF and every intersection contributes equally to PDF.

**Idea**: Weight each line's contribution to the distance function by its arc length. A 40° wall line gets 8x the influence of a 5° poster fragment. This naturally suppresses poster noise without needing a perfect classification threshold.

- Touches: `xdf_distance.py` (LDF computation), possibly `find_intersections_2d_indexed` (weight intersections by parent line lengths)
- Risk: low — additive change to existing cost, doesn't break working cases
- Effort: moderate — need to thread weights through the precompute pipeline

#### Approach C: Robust line pre-filtering (spatial density) ⬜

Detect and remove poster-like line clusters before matching. Structural lines are spatially distributed across the panorama; poster lines cluster in a small angular region.

- Touches: new preprocessing step before `find_intersections_2d_indexed`
- Risk: medium — might accidentally remove legitimate dense structural features
- Effort: moderate — new spatial analysis code

---

## Phase 3: Point Cloud Coloring ✅

**Status**: ✅ Implemented, validated against ground truth. Modular pipeline in `src/colorization/`.

With camera poses from Phase 2 (specifically `experiment_local_linefilter.py`'s Voronoi-filtered poses), project panoramic RGB pixels onto point cloud vertices.

### 3.1 Equirectangular Projection + Occlusion ✅

`colorize_point_cloud.py` — orchestrator. `projection.py` + `visibility.py` — libraries.

For each panorama with a known pose (R, t):
1. Transform each point to camera frame: `p_cam = (p_world - t) @ R.T`
2. Project to equirectangular coordinates matching `sphere_geometry.sphere_to_equirect` convention
3. **Depth-buffer occlusion**: Rasterize all points to a low-res depth buffer (2048×1024), keep only points within 5cm of the frontmost depth per pixel. O(N) pure numpy, handles furniture/wall occlusion.
4. Sample panorama color via bilinear interpolation with horizontal wraparound at the equirectangular seam

Design references: sub-pixel interpolation from [color_cloud_from_image](https://github.com/tu-darmstadt-ros-pkg/color_cloud_from_image), IDW blending from [pointcloud_painter](https://github.com/UTNuclearRoboticsPublic/pointcloud_painter). Neither tool was usable directly (no equirectangular support, ROS-only).

### 3.2 Multi-Pano Blending ✅

`color_sampling.py` — library.

When a point is visible from multiple panoramas:
- Inverse-distance-weighted (IDW) blending: `weight = 1/depth²`
- Closer panoramas contribute more, producing smooth transitions at Voronoi boundaries
- No hard seams between rooms

**Test case results** (tmb_office_one_corridor_dense, 8.8M points × 3 panos):
- 94% coverage (490K uncolored — ceiling/under-furniture)
- 1.1M points blended from 2 panoramas (overlap zones)
- 13K points seen by all 3 panoramas
- Total colorization time: ~20 seconds

### 3.3 Coverage and Quality Assessment ✅

`evaluate_colorization.py` — evaluation script.

Compares colorized PLY against original scanner RGB (Leica BLK360 G1 — same camera, same position, ground-truth comparison).

**Metrics computed**:
- Per-point RGB L2 distance: mean, median, 90th/95th percentile
- CIEDE2000 Delta-E (perceptual): mean, median, percentiles + threshold breakdown (<1 imperceptible, <2.3 barely noticeable, <5 noticeable, <10 obvious)
- Per-channel signed bias (detects systematic white balance / exposure shifts)
- Error heatmap PLY (green=low error, red=high, blue=uncolored) for visual inspection in CloudCompare

**Baseline results** (tmb_office_one_corridor_dense):
- Median RGB L2: 0.025 (very low — half the points are near-exact matches)
- 77.7% of points have Delta-E < 5 (noticeable but acceptable)
- 46.7% below Delta-E 2.3 (barely noticeable to human eye)
- Small uniform darkening bias (~-0.013 per channel), likely JPEG compression artifact

---

## Phase 4: Meshing ⬜

**Status**: ⬜ Not implemented. No meshing code exists in the repository.

Convert the colored point cloud into a watertight triangle mesh suitable for real-time rendering and measurement.

### 4.1 Surface Reconstruction ⬜

Candidate approaches:
- **Poisson surface reconstruction** (Open3D `create_from_point_cloud_poisson`) — requires oriented normals, produces watertight mesh, good for smooth indoor surfaces
- **Ball-pivoting algorithm** — better for preserving sharp edges (walls, furniture)
- **Alpha shapes** — already partially used in `point_cloud_geometry_baker_V3.py` for boundary extraction

### 4.2 Texture Baking ⬜

Transfer per-vertex colors to a UV-mapped texture atlas:
- Generate UV parameterization for the mesh
- Bake vertex colors (or direct panorama projections) into texture images
- Produce glTF/GLB format for web consumption

### 4.3 Mesh Optimization ⬜

- Decimation to reduce triangle count for web rendering
- Normal smoothing for visual quality
- Hole filling for scanning gaps

---

## Phase 5: Virtual Tour with Measurement ⬜

**Status**: ⬜ Not implemented. No web framework or viewer code exists (despite the repository name).

Build an interactive web-based virtual tour from the textured mesh, with real-world measurement capability.

### 5.1 3D Viewer Setup ⬜

Technology candidates:
- **Potree** — purpose-built for large point clouds, supports measurements natively
- **Three.js** — general-purpose WebGL, requires more custom code but maximum flexibility
- **3D Tiles / CesiumJS** — good for very large scenes with level-of-detail

### 5.2 Virtual Tour Navigation ⬜

- Hotspot-based navigation between panorama viewpoints
- Smooth camera transitions between rooms
- Minimap showing current position on floorplan (reuse RoomFormer output)
- Optional: free-fly camera mode through the 3D model

### 5.3 Measurement Tool ⬜

Since the point cloud is from a TLS (terrestrial laser scanner), the geometry is metrically accurate:
- Point-to-point distance measurement (click two points, display distance in meters)
- Area measurement (select polygon on surface)
- Height measurement (vertical distance between two horizontal planes)
- Export measurements as CSV/JSON

### 5.4 Deployment ⬜

- Static site hosting (the viewer is client-side)
- Data pipeline: mesh + textures -> 3D Tiles or Potree octree format
- Progressive loading for large scenes

---

## Phase Summary

| Phase | Description | Status | Key Scripts |
|-------|------------|--------|-------------|
| 1 | Room-level pano assignment (jigsaw puzzle) | 🟡 Core implemented, accuracy limited | `align_polygons_demo5.py`, `LGT-Net_inference_demo2.py`, `RoomFormer_inference.py` |
| 1.4 | SAM3 for better footprints | 🟡 Pano extraction production-ready, density image pending | `SAM3_pano_footprint_extraction.py`, `SAM3_footprint_comparison.py`, `SAM3_room_extraction_test.py` |
| 2.1 | 3D feature extraction | ✅ Implemented | `point_cloud_geometry_baker_V4.py`, `cluster_3d_lines.py`, `line_clustering_3d.py` |
| 2.2 | 2D feature extraction | ✅ Implemented | `image_feature_extractionV2.py`, `pano_line_detector.py`, `line_analysis.py` |
| 2.3 | Single-room pose estimation | ✅ Implemented, validated | `pose_estimation_pipeline.py` + library modules |
| 2.4 | Pose visualization | 🟡 Mostly done | `visualize_pose.py` (side-by-side, reprojection, topdown) |
| 2.5 | Multi-room pose estimation | ✅ Implemented | `multiroom_pose_estimation.py` |
| 2.6 | Consolidate pose codebase | ✅ Done | — |
| 2.7 | Jigsaw-informed improvements | 🟡 B+D done, A+C not started | `experiment_polygon_prior.py`, `experiment_local_linefilter.py` |
| 3 | Point cloud coloring | ✅ Implemented, validated | `colorize_point_cloud.py`, `evaluate_colorization.py` + library modules |
| 4 | Meshing | ⬜ Not started | — |
| 5 | Virtual tour + measurement | ⬜ Not started | — |

---

## Dependencies Between Phases

```
Phase 1 (room assignment) ──┐
                             ├──> Phase 2.5 (multi-room pose) ──> Phase 2.7 (jigsaw improvements) ──> Phase 3 (coloring) ✅ ──> Phase 4 (meshing) ──> Phase 5 (virtual tour)
Phase 2.1-2.3 (single-room) ┘
```

Phase 1.4 (SAM3 footprints) can proceed independently at any time.
Phase 5.3 (measurement) depends on Phase 4 (mesh) or can work directly on the point cloud.

---

## Next Steps (as of 2026-03-21)

1. ~~**Evaluate SAM3 footprint quality** (Phase 1.4)~~ — ✅ Panoramic footprint extraction is production-ready. `SAM3_pano_footprint_extraction.py` produces Manhattan-regularized polygons via floor/ceiling fusion with height correction. Validated on 6 test panoramas. Three fusion variants compared; per-column optimistic (Approach A) selected.

2. **Integrate SAM3 footprints into polygon matching** (Phase 1.3 → 1.4) — Feed SAM3 `layout.json` polygons from `SAM3_pano_footprint_extraction.py` into `align_polygons_demo5.py` as an alternative/complement to LGT-Net footprints. Compare matching accuracy.

3. **Update the production multi-room pose estimation pipeline** (Phase 2.7 → 2.5) — Integrate the validated local 3D line filtering (Voronoi-based, from `experiment_local_linefilter.py`) into `multiroom_pose_estimation.py`. The experiment proved that filtering 3D lines per-panorama eliminates cross-room false minima (office1: 2.3m error → sub-cm).

4. ~~**Color the point cloud** (Phase 3)~~ — ✅ Done. Colorization pipeline implemented with depth-buffer occlusion, bilinear sampling, IDW multi-pano blending. Validated against scanner ground truth (median Delta-E 2.47, 77.7% below 5).

5. **Start meshing** (Phase 4) — Convert the colored point cloud into a triangle mesh (Poisson or ball-pivoting), bake textures, export as glTF for web rendering.

6. **Start developing the virtual tour** (Phase 5) — Build the interactive web-based viewer with measurement capability. Evaluate technology candidates (Potree, Three.js, 3D Tiles).
