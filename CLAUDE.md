# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**scan2measure-webframework** is a multi-stage 3D reconstruction pipeline for indoor building measurement. It takes raw TLS point clouds and panoramic images as input, estimates camera poses, colors the point cloud, and produces a UV-textured GLB mesh for a virtual tour with measurement capability. The end goal is a Unity-based desktop app orchestrating the full pipeline.

## Source Directory Structure

```
src/
├── preprocessing/          # Raw data → intermediate representations
│   ├── generate_density_image.py
│   ├── ply_to_xyz.py
│   └── map_RoomFormer_results_to_3d.py
│
├── segmentation/           # Neural network inference for room/object detection
│   ├── RoomFormer_inference.py
│   └── LGT-Net_inference_demo2.py
│
├── geometry_3d/            # 3D line detection, plane fitting, clustering
│   ├── point_cloud_geometry_baker_V3.py
│   ├── point_cloud_geometry_baker_V4.py
│   ├── cluster_3d_lines.py
│   └── line_clustering_3d.py
│
├── features_2d/            # 2D feature extraction from panoramas
│   ├── image_feature_extraction.py       # V3 path (LSD on perspective crops)
│   ├── image_feature_extractionV2.py     # V4 path (FGPL sphere-based)
│   ├── pano_line_detector.py
│   ├── line_analysis.py
│   ├── pano_processing_virtual_camerasV2.py
│   └── lightweight_synthetic_renderer_V2.py
│
├── pose_estimation/        # Camera localization pipeline
│   ├── pose_estimation_pipeline.py
│   ├── multiroom_pose_estimation.py
│   ├── pose_search.py
│   ├── pose_refine.py
│   └── xdf_distance.py
│
├── floorplan/              # Scale estimation & polygon alignment
│   ├── align_polygons_demo6.py          # SAM3-to-SAM3 jigsaw (enumerate + optimize)
│   ├── align_polygons_demo5.py          # Legacy: Hungarian matching (RoomFormer ↔ LGT-Net)
│   ├── polygon_scale_calculation_v2.py  # Three scale methods (edge, Procrustes, area)
│   ├── polygon_scale_calculation.py     # Legacy: histogram consensus scale
│   └── SAM3_mask_to_polygons.py         # Density image masks → world-meter polygons
│
├── visualization/          # All rendering & debug visualization
│   ├── visualize_pose.py
│   └── visualize_matching.py
│
├── colorization/          # Point cloud coloring from panoramic images
│   ├── colorize_point_cloud.py    # Orchestrator: multi-pano colorization pipeline
│   ├── evaluate_colorization.py   # Ground-truth evaluation (RGB L2 + CIEDE2000 Delta-E)
│   ├── projection.py              # Equirectangular projection math
│   ├── visibility.py              # Depth-buffer occlusion test
│   └── color_sampling.py          # Bilinear sampling + multi-pano IDW blending
│
├── meshing/               # Point cloud → textured GLB mesh (parallel, quality-tiered)
│   ├── mesh_reconstruction.py     # Orchestrator: 17-stage pipeline with quality tiers (preview/balanced/high)
│   ├── mesh_utils.py              # Library: normals, Poisson, tiling, trimming, UV unwrap, texture baking + parallel workers
│   └── export_gltf.py             # GLB export with PBR texture + metric metadata injection
│
├── utils/                  # Shared math & geometry primitives
│   └── sphere_geometry.py
│
├── experiments/            # One-off experiments (not part of main pipeline)
│   ├── experiment_polygon_prior.py
│   ├── experiment_local_linefilter.py
│   ├── SAM3_inference.py
│   ├── SAM3_pano_processing.py
│   ├── SAM3_pano_footprint_extraction.py
│   ├── SAM3_footprint_comparison.py
│   ├── SAM3_pano_raw_segmentation.py
│   ├── SAM3_room_extraction_test.py
│   └── SAM3_room_segmentation.py
│
└── legacy/                 # Superseded code kept for reference
    ├── PnL_solver.py
    ├── pano_processing_virtual_cameras.py
    ├── test_FGPL_3d_feature.py
    ├── test_FGPL_pose_estimation.py
    ├── test_fgpl_multiroom.py
    └── texture_mapping.py
```

Cross-subfolder imports use `sys.path.insert(0, str(_SRC_ROOT / "<subfolder>"))` where `_SRC_ROOT = Path(__file__).resolve().parent.parent` resolves to `src/`.

## Two Feature Extraction Approaches

The pipeline supports two independent approaches for extracting geometric features from point clouds and panoramic images. Both feed into the same downstream pose estimation and scale calculation stages.

### In-house pipeline (V3 path)

Custom feature extraction developed for this project.

- **3D features**: `geometry_3d/point_cloud_geometry_baker_V3.py` — two-phase plane detection (Open3D region growing → alpha-shape boundaries) + topological wireframe (plane intersections → Binary LP). ~900 lines, slow but produces clean planar decompositions
- **3D rendering**: `features_2d/lightweight_synthetic_renderer_V2.py` — renders wireframe into 22 synthetic perspective views for visual comparison
- **2D features**: `features_2d/image_feature_extraction.py` — CLAHE contrast enhancement → anisotropic diffusion → LSD line detection on 22 virtual perspective crops (from `features_2d/pano_processing_virtual_camerasV2.py`)
- **Output**: `extracted_2d_lines.json` (pinhole-projected line segments) + `visible_3d_lines.json` (per-view 3D wireframe)

### FGPL pipeline (V4 path)

Feature extraction following the algorithms from the FGPL panoramic localization paper (CVPR 2024). Reimplemented independently with no imports from `panoramic-localization/`.

- **3D features**: `geometry_3d/point_cloud_geometry_baker_V4.py` — thin wrapper around `3DLineDetection` C++ binary for ~100x speedup. PLY → XYZ → `LineFromPointCloud` → parse lines.obj → `room_geometry.pkl`
- **2D features**: `features_2d/image_feature_extractionV2.py` — orchestrator that calls three library modules:
  - `utils/sphere_geometry.py` — icosphere generation, equirectangular ↔ sphere coordinate conversions, panoramic rasterization
  - `features_2d/pano_line_detector.py` — 26-view panorama decomposition (HorizonNet tangent-plane projection), LSD detection, sphere back-projection, 3-pass colinear segment merging. Applies 90° Z-rotation to align HorizonNet coordinates with equirectangular rendering frame
  - `features_2d/line_analysis.py` — icosphere voting for vanishing points, line classification into 3 principal groups, great-circle arc intersection finding
- **Output**: `fgpl_features.json` (sphere-projected lines + principal directions + intersections) + `edge_overlay.png` + `grouped_lines.png` in `data/pano/2d_feature_extracted/<room>_v2/`

### Which to use

The FGPL path (V4) is preferred for pose estimation via `pose_estimation/pose_estimation_pipeline.py` — the 3D and 2D features are in compatible sphere-based representations and the intersection-based matching is more robust. The in-house path (V3) is useful when planar decomposition or synthetic rendering is needed.

## Running Pipeline Stages

All scripts are run individually from the repo root. Each uses `Path(__file__).resolve()` internally to find data paths, so working directory does not matter.

**Shared stages (both paths):**
```bash
python src/preprocessing/generate_density_image.py            # Stage 1: Point cloud → density image
python src/segmentation/RoomFormer_inference.py                # Stage 2: Density image → 2D floorplan polygons
```

**FGPL path (V4) — preferred (single-room):**
```bash
python src/geometry_3d/point_cloud_geometry_baker_V4.py        # Stage 3a: Point cloud → 3D wireframe (3DLineDetection)
python src/geometry_3d/cluster_3d_lines.py                     # Stage 3b: Wireframe → clustered 3D line map (principal dirs + intersections)
python src/features_2d/image_feature_extractionV2.py           # Stage 4: Panoramic image → sphere-based 2D line features
python src/pose_estimation/pose_estimation_pipeline.py         # Stage 5: FGPL-faithful camera pose estimation
python src/floorplan/polygon_scale_calculation.py              # Stage 6: Compute pixel-to-meter scale
```

**FGPL path (V4) — multi-room:**
```bash
python src/geometry_3d/point_cloud_geometry_baker_V4.py        # Stage 3a: Combined point cloud → 3D wireframe
python src/geometry_3d/cluster_3d_lines.py                     # Stage 3b: Wireframe → clustered 3D line map
python src/features_2d/image_feature_extractionV2.py           # Stage 4: Per-panorama 2D feature extraction (run for each pano)
python src/pose_estimation/multiroom_pose_estimation.py        # Stage 5: Multi-pano pose estimation (3D precompute once, 2D per-pano)
python src/experiments/experiment_local_linefilter.py          # Stage 5b (optional): Voronoi-filtered pose re-estimation (saves R+t)
python src/colorization/colorize_point_cloud.py                # Stage 6: Color point cloud from panoramas using poses
conda run -n scan_env python src/meshing/mesh_reconstruction.py  # Stage 7: Colored PLY → textured GLB mesh (quality-tiered: ~6/11/20 min)
```

**In-house path (V3):**
```bash
python src/geometry_3d/point_cloud_geometry_baker_V3.py        # Stage 3: Point cloud → 3D wireframe (plane detection + Binary LP)
python src/features_2d/lightweight_synthetic_renderer_V2.py    # Stage 4a: 3D wireframe → synthetic views
python src/features_2d/pano_processing_virtual_camerasV2.py    # Stage 4b: Panorama → 22 perspective crops
python src/features_2d/image_feature_extraction.py             # Stage 4c: Perspective crops → 2D line features
python src/pose_estimation/pose_estimation_pipeline.py         # Stage 5: FGPL-faithful camera pose estimation
python src/floorplan/polygon_scale_calculation.py              # Stage 6: Compute pixel-to-meter scale
```

Stage 5: `pose_estimation/pose_estimation_pipeline.py` — canonical FGPL-faithful pose estimation. Modular 10-stage pipeline importing `xdf_distance.py`, `pose_search.py`, `pose_refine.py`, `visualize_pose.py`. Loads `3d_line_map.pkl` + `fgpl_features.json`, outputs `camera_pose.json` (V2 keys).
- Archived: `feature_matchingV2.py` (monolithic, older V1 input format) → `Archive/`
- Archived: `feature_matching.py` (V1 legacy) → `Archive/`

Supporting scripts:
```bash
python src/visualization/visualize_matching.py                 # Generate 2-panel overlay PNGs for pose inspection (supports V1 + V2)
python src/floorplan/align_polygons_demo6.py                   # SAM3-to-SAM3 jigsaw matching (enumerate + optimize)
python src/floorplan/align_polygons_demo5.py                   # Legacy: Hungarian algorithm polygon matching
python src/preprocessing/map_RoomFormer_results_to_3d.py       # Project 2D floorplan polygons back to 3D point cloud
python src/segmentation/LGT-Net_inference_demo2.py             # Panoramic layout prediction via LGT-Net
python src/preprocessing/ply_to_xyz.py                         # Convert PLY to plain-text XYZ (preprocessing for 3DLineDetection)
```

SAM3 experiment scripts (require `conda run -n sam3`):
```bash
python src/experiments/SAM3_inference.py                       # Basic text-prompted segmentation demo
python src/experiments/SAM3_room_segmentation.py               # Room footprints from density images via SAM3
python src/experiments/SAM3_room_extraction_test.py            # 3-approach experiment: walls, rooms, hybrid
python src/experiments/SAM3_pano_processing.py                 # Room polygons from panoramic images via SAM3
python src/experiments/SAM3_pano_footprint_extraction.py       # Production: floor/ceiling fusion → Manhattan polygon per pano
python src/experiments/SAM3_footprint_comparison.py             # Three-way fusion comparison (A/B/C) + morphological + LGT-Net
python src/experiments/SAM3_pano_raw_segmentation.py           # Raw multi-prompt segmentation comparison on panos
```

Experiment scripts:
```bash
python src/experiments/experiment_polygon_prior.py             # Jigsaw-guided translation filtering (Approach B)
python src/experiments/experiment_local_linefilter.py          # Voronoi-based local 3D line filtering (Approach D)
```

Legacy test scripts (validate against FGPL native code):
```bash
python src/legacy/test_FGPL_3d_feature.py                     # Test FGPL native 3D line clustering & intersection on OBJ data
python src/legacy/test_FGPL_pose_estimation.py                 # Single-room pose estimation using FGPL native XDF + ICP
python src/legacy/test_fgpl_multiroom.py                       # Multi-room baseline using FGPL native code
```

There is no build step, test runner, or package install — dependencies (open3d, torch, shapely, scipy, opencv, matplotlib, pylsd-nova) must be installed manually. External models and libraries live as subdirectories at the repo root. SAM3 scripts require a separate conda environment (`conda run -n sam3`).

## Architecture

### Data Flow

```
=== Shared stages ===

data/raw_point_cloud/*.ply
  → preprocessing/generate_density_image.py
  → data/density_image/<room>/  (PNG + rotation metadata JSON)
  → segmentation/RoomFormer_inference.py
  → data/reconstructed_floorplans_RoomFormer/<room>/  (polygon JSON + images)

=== FGPL path (V4) — 3D features ===

data/raw_point_cloud/*.ply
  → geometry_3d/point_cloud_geometry_baker_V4.py  (calls 3DLineDetection C++ binary)
  → room_geometry.pkl  (wireframe segments: dirs, starts, ends)
  → geometry_3d/cluster_3d_lines.py  (principal direction voting + classification + intersection)
  → 3d_line_map.pkl  (dense/sparse lines, principal_3d, 3D intersections)
  → clustered_lines.obj + intersections.obj  (debug visualization)

=== FGPL path (V4) — 2D features ===

data/pano/raw/<room>.jpg
  → features_2d/image_feature_extractionV2.py
    (calls features_2d/pano_line_detector.py → features_2d/line_analysis.py → utils/sphere_geometry.py)
  → data/pano/2d_feature_extracted/<room>_v2/
    (fgpl_features.json + edge_overlay.png + grouped_lines.png)

=== In-house path (V3) — 3D features ===

data/raw_point_cloud/*.ply
  → geometry_3d/point_cloud_geometry_baker_V3.py  (plane detection + Binary LP wireframe)
  → room_geometry.pkl
  → features_2d/lightweight_synthetic_renderer_V2.py
  → data/debug_renderer/<room>/  (synthetic PNGs + visible_3d_lines.json)

=== In-house path (V3) — 2D features ===

data/pano/<room>/  (panoramic images)
  → features_2d/pano_processing_virtual_camerasV2.py  (22 perspective crops)
  → features_2d/image_feature_extraction.py  (CLAHE + anisotropic diffusion + LSD)
  → data/pano/2d_feature_extracted/<room>/  (extracted_2d_lines.json + debug images)

=== Single-room pose estimation ===

pose_estimation/pose_estimation_pipeline.py + pose_search.py + pose_refine.py + xdf_distance.py
  (loads 3d_line_map.pkl + fgpl_features.json → camera_pose.json)
  → visualization/visualize_pose.py  (side_by_side.png + reprojection.png + topdown.png)
  → data/pose_estimates/<room>/

=== Multi-room pose estimation ===

pose_estimation/multiroom_pose_estimation.py  (shared 3D precompute once, per-pano 2D + matching)
  (loads 3d_line_map.pkl + fgpl_features.json per pano → camera_pose.json per pano)
  → data/pose_estimates/<map_name>/<pano>/

=== Point cloud colorization ===

data/raw_point_cloud/*.ply + data/pano/raw/*.jpg + local_filter_results.json (R+t per pano)
  → colorization/colorize_point_cloud.py
    (calls colorization/projection.py → visibility.py → color_sampling.py)
  → data/textured_point_cloud/<map>_textured.ply

=== Meshing ===

data/textured_point_cloud/<map>_textured.ply
  → meshing/mesh_reconstruction.py  (17-stage parallel pipeline, quality-tiered)
    Quality tiers (set QUALITY_TIER in config):
      "preview"  — Poisson depth 7, 15mm voxel, 2048 atlas, 250K tris (~6 min)
      "balanced" — Poisson depth 8, 10mm voxel, 4096 atlas, 500K tris (~11 min) [default]
      "high"     — Poisson depth 9,  5mm voxel, 4096 atlas, 500K tris (~20 min)
    Stage 1-2: Load + voxel downsample
    Stage 3-9: Tiled Poisson (6x6m tiles, 1m overlap) — parallel via ProcessPoolExecutor (spawn)
    Stage 10-11: Merge → save full-res vertex-colored PLY
    Stage 12: Decimate for textured GLB
    Stage 13-16: Reload full cloud → xatlas UV unwrap → bake atlas (KNN=4, parallel) → dilate
    Stage 17: Export GLB with metric metadata
  → data/mesh/<map>/
    <name>.glb                    (~15 MB, UV-textured)
    <name>_vertex_colored.ply     (full-res, vertex-colored)
    <name>_metadata.json          (pipeline parameters + stats + quality_tier)

=== Legacy visualization ===

visualization/visualize_matching.py  (loads camera_pose.json → 2-panel debug PNGs, supports V1 + V2 formats)

=== Scale estimation ===

floorplan/align_polygons_demo6.py  (SAM3 jigsaw: enumerate assignments + optimize scale/rotation/translation)
  → data/sam3_room_segmentation/<map>/  (demo6_alignment.json + demo6_alignment.png)
```

### Key Scripts

| Script | Core Logic |
|--------|-----------|
| `preprocessing/generate_density_image.py` | RANSAC floor plane detection, Manhattan World axis alignment, ceiling detection, 2D density projection |
| `segmentation/RoomFormer_inference.py` | ResNet50+Transformer (RoomFormer model), area filtering via Shapely, polygon probability threshold filtering |
| `geometry_3d/point_cloud_geometry_baker_V4.py` | Python wrapper for 3DLineDetection C++ binary: PLY → XYZ → `LineFromPointCloud` → parse lines.obj → room_geometry.pkl |
| `geometry_3d/cluster_3d_lines.py` | Orchestrator: loads room_geometry.pkl → `line_clustering_3d` for principal direction voting + classification + 3D intersection → writes `3d_line_map.pkl` + colored OBJ debug files |
| `geometry_3d/line_clustering_3d.py` | Library: 3D line principal direction voting (icosphere), `classify_lines_3d`, `find_intersections_3d`, `build_intersection_masks`. Mirrors FGPL map_utils + edge_utils algorithms |
| `geometry_3d/point_cloud_geometry_baker_V3.py` | Legacy: Two-phase plane detection (Open3D region growing → alpha-shape boundaries) + topological wireframe (plane intersections → Binary LP). ~900 lines |
| `features_2d/lightweight_synthetic_renderer_V2.py` | 22-view spherical tiling (60° FOV, 1024x1024), z-buffer depth rendering, 3-channel output (wireframe / edges / regions) |
| `features_2d/image_feature_extraction.py` | In-house 2D: CLAHE → anisotropic diffusion → LSD line detection on 22 perspective crops |
| `features_2d/image_feature_extractionV2.py` | FGPL 2D: orchestrator calling `pano_line_detector` → `line_analysis` → `sphere_geometry` for sphere-based line extraction, vanishing points, and intersections |
| `utils/sphere_geometry.py` | Library: icosphere generation, equirectangular ↔ sphere projection, panoramic point/line rasterization |
| `features_2d/pano_line_detector.py` | Library: 26-view HorizonNet-style panorama decomposition, LSD detection, tangent-plane back-projection to sphere, 3-pass segment merging |
| `features_2d/line_analysis.py` | Library: icosphere voting for 3 vanishing points, line classification into principal groups, great-circle arc intersection finding |
| `pose_estimation/pose_estimation_pipeline.py` | **Canonical**. Modular 10-stage FGPL-faithful pipeline: loads pre-computed `3d_line_map.pkl` + `fgpl_features.json` → 2D intersections → 24 rotation candidates → rearrange intersections → adaptive quantile translation grid → `single_pose_compute` XDF search → top-K=10 ICP refinement → quality-ranked selection → camera_pose.json (V2 keys) |
| `pose_estimation/xdf_distance.py` | Library: LDF/PDF sphere distance functions, 3D line classification, 2D intersection finding with line-pair index tracking |
| `pose_estimation/pose_search.py` | Library: 24 rotation candidates (SVD Procrustes), adaptive quantile translation grid, canonical-frame XDF coarse search with `single_pose_compute`, rotation diversity selection |
| `pose_estimation/pose_refine.py` | Library: two-phase sphere ICP refinement (Phase 1: translation via grouped mutual-NN + global fallback, Phase 2: rotation via YPR line direction alignment). Safety drift checks |
| `visualization/visualize_pose.py` | Library: `render_side_by_side` (2D lines + projected 3D wireframe overlay), `render_reprojection` (depth-coded point cloud on panorama), `render_topdown` (camera on density image with room polygons) |
| `pose_estimation/multiroom_pose_estimation.py` | Multi-pano orchestrator: `precompute_xdf_3d()` runs once (~10s), `xdf_coarse_search_from_precomputed()` + ICP per panorama (~1s each). Zero imports from `panoramic-localization/` |
| `experiments/experiment_polygon_prior.py` | Approach B: filters translation candidates by RoomFormer polygon containment from `global_alignment.json`. Fixed corridor wrong-room problem but limited general improvement |
| `experiments/experiment_local_linefilter.py` | Approach D: Voronoi-based spatial filtering of 3D line map per panorama (nearest-pano assignment + 2.0m overlap margin). Dramatic accuracy improvement — sub-cm on validated rooms. Saves R+t to `local_filter_results.json` |
| `colorization/colorize_point_cloud.py` | **Orchestrator**. Loads PLY + panoramas + poses from `local_filter_results.json`, projects points via equirectangular math, depth-buffer occlusion, bilinear color sampling, IDW multi-pano blending → colored PLY. ~20s for 8.8M points × 3 panos |
| `colorization/projection.py` | Library: `world_to_camera`, `camera_to_equirect`, `project_points_to_pano`. Equirectangular projection matching `sphere_geometry.sphere_to_equirect` convention |
| `colorization/visibility.py` | Library: `compute_visibility_depth_buffer`. Rasterizes to low-res depth buffer, marks points visible if within margin of frontmost depth. O(N) numpy |
| `colorization/color_sampling.py` | Library: `sample_colors_bilinear` (scipy `map_coordinates` with horizontal wraparound padding), `blend_colors_idw` (inverse-distance-weighted accumulation across panoramas) |
| `colorization/evaluate_colorization.py` | Ground-truth evaluation: compares colorized PLY against original scanner RGB. RGB L2 distance, CIEDE2000 Delta-E (perceptual), per-channel bias, error heatmap PLY output (green→red gradient, blue=uncolored) |
| `meshing/mesh_reconstruction.py` | **Orchestrator**. 17-stage parallel pipeline with quality tiers (preview/balanced/high). Tile processing parallelized via `ProcessPoolExecutor` (spawn context). Texture baking also parallel. `QUALITY_TIER` config selects Poisson depth (7/8/9), voxel size, atlas resolution, and target triangles. Balanced tier: ~11 min on 32-core system. Timing breakdown printed at end |
| `meshing/mesh_utils.py` | Library: `estimate_normals`, `poisson_reconstruct`, `remove_low_density`, `transfer_vertex_colors`, `_process_single_tile` (parallel worker), `process_tiles_parallel`, `compute_tile_grid`, `extract_tile_points`, `trim_to_ownership_region`, `merge_tile_meshes`, `uv_unwrap_mesh`, `_bake_face_chunk` (parallel worker), `bake_texture_atlas` (supports `max_workers`), `dilate_texture` |
| `meshing/export_gltf.py` | `export_textured_glb` (trimesh PBR material + `_inject_gltf_metadata` for metric unit/scale in asset.extras), `export_vertex_color_ply` (Open3D PLY writer) |
| `legacy/PnL_solver.py` | Legacy ICP module (unused by active pipeline, kept for reference). Contains `refine_pose_icp()` (V1) and `refine_pose_full()` (V2) |
| `floorplan/align_polygons_demo6.py` | **Canonical**. SAM3-to-SAM3 jigsaw matching. Stage 1: enumerate all pano→room assignments, optimize shared scale + per-pano rotation/translation via `differential_evolution` (true IoU scoring). Stage 2: OBB long-axis alignment, dense translation grid, non-overlap penalty, largest pano placed first. Outputs `demo6_alignment.json` + visualization |
| `floorplan/polygon_scale_calculation_v2.py` | Three consensus-scale methods: A=edge distances, B=Procrustes, C=area ratio. Used by demo6 legacy modes (`--compare`) |
| `floorplan/align_polygons_demo5.py` | Legacy: Hungarian matching (RoomFormer 2D ↔ LGT-Net 3D) |
| `floorplan/polygon_scale_calculation.py` | Legacy: histogram consensus scale |
| `experiments/SAM3_mask_to_polygons.py` | Converts SAM3 density image masks to world-meter polygons via `pixels_to_world_meters()`. Outputs `*_polygons.json` with `vertices_world_meters` per room |
| `visualization/visualize_matching.py` | 22 side-by-side PNGs per panorama. V2 layout: left = 2D lines (green) + projected 3D wireframe (cyan) overlay, right = matched intersection pairs colored by group. Falls back to V1 layout (separate 2D/3D panels) when V2 keys absent |
| `preprocessing/map_RoomFormer_results_to_3d.py` | Inverse-projects RoomFormer 2D polygons to world coordinates, segments point cloud by room |
| `features_2d/pano_processing_virtual_camerasV2.py` | Extracts 22 perspective crops from equirectangular panorama (matches renderer's spherical tiling) |
| `segmentation/LGT-Net_inference_demo2.py` | Panoramic layout prediction: Manhattan frame detection, vanishing point alignment, depth + layout boundary output |
| `experiments/SAM3_inference.py` | Basic SAM3 demo: text-prompted segmentation, returns masks + bounding boxes + confidence |
| `experiments/SAM3_room_segmentation.py` | SAM3 on CLAHE-inverted density images with "floor plan" prompt → room footprint masks |
| `experiments/SAM3_room_extraction_test.py` | 3-approach experiment: walls-first, room-areas-direct, hybrid two-pass. Tests SAM3 for density image room extraction |
| `experiments/SAM3_pano_processing.py` | SAM3 on equirectangular panoramas → floor/wall boundary masks → XZ polygon projection via equirectangular deprojection |
| `experiments/SAM3_pano_footprint_extraction.py` | **Production SAM3 footprint script**. Floor/ceiling fusion with height correction → Manhattan-regularized XZ polygon. Outputs `<stem>/layout.json` + `<stem>/debug.png` per pano |
| `experiments/SAM3_footprint_comparison.py` | Three-way fusion comparison (A=optimistic, B=dual-XZ, C=ceiling-primary) + morphological + LGT-Net. Height-corrected ceiling projection. 5-row detail + summary figures |
| `experiments/SAM3_pano_raw_segmentation.py` | Multi-prompt raw segmentation comparison on panoramas (distinct colors per component) |
| `preprocessing/ply_to_xyz.py` | Utility: convert PLY → plain-text XYZ (one point per line, used by 3DLineDetection) |
| `legacy/test_FGPL_3d_feature.py` | Test: converts OBJ → FGPL TXT, calls native `generate_line_map_single()` for principal direction voting + 3D intersection |
| `legacy/test_FGPL_pose_estimation.py` | Test: feeds our 3D line map + 2D features into FGPL's native XDF coarse search + Sphere ICP. Validates against native baseline |
| `legacy/test_fgpl_multiroom.py` | Test: multi-room pose estimation using FGPL native code. Baseline results for comparison with own implementation |

### `pose_estimation/pose_estimation_pipeline.py` — FGPL-Faithful Pose Estimation

Modular 10-stage pipeline using PyTorch tensors throughout:

1. **Load 3D features** — `3d_line_map.pkl` → dense/sparse lines, principal_3d, pre-computed 3D intersections
2. **Load 2D features** — `fgpl_features.json` → sphere-projected lines `(N, 9)`, principal_2d
3. **2D intersections** — `xdf_distance.find_intersections_2d_indexed()`: cross-product + arc membership, with line-pair index tracking
4. **24 rotation candidates** — `pose_search.build_rotation_candidates()`: 6 permutations × 4 det-preserving sign flips, SVD Procrustes
5. **Rearrange intersections** — `pose_search.rearrange_intersections_for_rotations()`: remap 2D intersection groups per rotation permutation
6. **Translation grid** — `pose_search.generate_translation_grid()`: adaptive quantile-based (~1700 pts), chamfer-filtered
7. **XDF coarse search** — `pose_search.xdf_coarse_search()`: canonical frame precomputation, `single_pose_compute` LDF/PDF approximation, inlier counting `|d_2d - d_3d| < 0.1`, rotation-diverse top-K=10
8. **ICP refinement** — `pose_refine.refine_pose()` on 10 candidates: Phase 1 = translation (100 iters), Phase 2 = rotation via YPR (50 iters). Quality ranking by `(-n_tight, avg_dist)`
9. **Output** — `camera_pose.json` (V2 keys)
10. **Visualization** — `visualize_pose.render_side_by_side()` → `side_by_side.png`

Key configuration constants (top of file):
- `VOTE_SPHERE_LEVEL = 5`, `QUERY_SPHERE_LEVEL = 3`
- `XDF_INLIER_THRES = 0.1`, `POINT_GAMMA = 0.2`
- `CHAMFER_MIN_DIST = 0.3`, `NUM_TRANS = 1700`, `TOP_K = 10`

### `camera_pose.json` Output Format

Common keys (V1 + V2):
- `rotation` — `(3,3)` world-to-camera rotation matrix
- `translation` — `(3,)` camera position in world frame
- `principal_3d`, `principal_2d` — `(3,3)` principal direction matrices
- `n_inter_matched` — number of matched intersection pairs
- `xdf_cost_coarse` — best coarse XDF cost

V2-only keys (added by `pose_estimation/pose_estimation_pipeline.py`):
- `inter_2d` — list of 3 arrays, each `(M_k, 3)` — 2D intersection sphere points per group
- `inter_3d` — list of 3 arrays, each `(N_k, 3)` — 3D intersection world points per group
- `matched_pairs` — list of 3 arrays, each `(P_k, 2)` — mutual-NN index pairs `[i_2d, i_3d]` from ICP

`visualization/visualize_matching.py` detects V2 keys and switches layout accordingly.

### `colorization/colorize_point_cloud.py` — Point Cloud Colorization

Colors a raw PLY point cloud using equirectangular panoramic images and camera poses from `local_filter_results.json` (produced by `experiment_local_linefilter.py`).

**Three-phase pipeline:**
1. **Phase A** — Load point cloud, camera poses (R+t), and panorama images
2. **Phase B** — Per-panorama: equirectangular projection → depth-buffer occlusion → bilinear color sampling
3. **Phase C** — IDW blending across panoramas → save colored PLY

**Library modules:**
- `projection.py` — Equirectangular projection (`world_to_camera` + `camera_to_equirect`). Convention: `theta=atan2(norm(xy),z)`, `phi=atan2(y,x)+pi`, matching `sphere_geometry.sphere_to_equirect`
- `visibility.py` — Depth-buffer occlusion test. Quantizes projections to low-res buffer (2048×1024), keeps points within 5cm of frontmost depth per pixel. O(N) pure numpy
- `color_sampling.py` — Bilinear sampling via `scipy.ndimage.map_coordinates` with horizontal padding for equirectangular wraparound. IDW blending weights by `1/depth²` across panoramas

**Key configuration constants:**
- `DEPTH_BUFFER_W/H = 2048/1024` — occlusion buffer resolution
- `DEPTH_MARGIN = 0.05` — meters, co-planar surface tolerance
- `IDW_POWER = 2.0` — inverse-distance blending exponent

**Input:** `data/raw_point_cloud/<name>.ply` + `data/pano/raw/*.jpg` + `data/pose_estimates/multiroom/local_filter_results.json`
**Output:** `data/textured_point_cloud/<name>_textured.ply`

### `geometry_3d/point_cloud_geometry_baker_V3.py` — Legacy Core Algorithm

Superseded by V4 for ~100x speedup, but still available. Key tunable parameters:

- `coplanar_merge_distance = 0.08m` — separates walls from cabinets; increase if walls merge incorrectly
- `min_hull_area = 0.90 m²` — minimum plane area to keep; increase to filter small planes
- `manhattan_snap_deg = 15°` — snaps normals to ±X/Y/Z axes; reduce for non-Manhattan spaces
- `alpha = 2.0` — alpha-shape boundary smoothness; reduce for tighter boundaries
- `lambda_complexity = 0.1` — Binary LP trade-off: higher = fewer edges, sparser wireframe

### Hardcoded Room Name

Most scripts use name constants at the top:
- **Single-room scripts**: `POINT_CLOUD_NAME` (directory under `data/debug_renderer/` or `data/raw_point_cloud/`) and `ROOM_NAME` (directory under `data/pano/`). Currently set to `tmb_office1` / `TMB_office1`.
- **Multi-room scripts** (`pose_estimation/multiroom_pose_estimation.py`, experiments): `POINT_CLOUD_NAME` (combined map, e.g. `tmb_office_one_corridor_dense`) and `PANO_NAMES` (list of panorama names to process).

Update these constants in each script to process a different room or dataset.

### Data Serialization Conventions

- **JSON**: polygon coordinates, 2D/3D line features, rotation metadata, camera_pose.json
- **PKL**: complex geometry objects (`room_geometry.pkl`, `3d_line_map.pkl`)
- **PNG**: density images, synthetic renders, debug visualizations, pose overlays
- **OBJ + MTL**: 3DLineDetection output, clustered line debug visualization (color-coded by principal group)

### Key Data Directories

```
data/
├── raw_point_cloud/                      # Input PLY files
├── density_image/<room>/                 # Stage 1 output: PNG + rotation metadata JSON
├── reconstructed_floorplans_RoomFormer/  # Stage 2 output: polygon JSON + images
├── debug_renderer/<map>/                 # 3D features: room_geometry.pkl, 3d_line_map.pkl, OBJ debug
├── debug_3dlinedetection/                # Raw 3DLineDetection C++ output
├── pano/
│   ├── raw/                              # Input panoramic JPGs
│   ├── 2d_feature_extracted/<room>_v2/   # FGPL 2D features: fgpl_features.json + debug PNGs
│   ├── virtual_camera_processed/         # V3 path: 22 perspective crops
│   └── LGT_Net_processed/               # LGT-Net layout predictions
├── pose_estimates/<map>/<pano>/          # camera_pose.json + visualization PNGs
├── segmented_point_cloud/                # Room-assigned point clouds
├── sam3_output/                          # Basic SAM3 inference results
├── sam3_room_segmentation/               # SAM3 room footprints from density images
├── sam3_room_extraction/                 # SAM3 3-approach room extraction results
├── sam3_pano_processing/                 # SAM3 room polygons from panoramas
│   ├── <stem>/                           #   Per-pano: layout.json + debug.png (from footprint_extraction)
│   └── footprint_comparison/             #   Three-way fusion comparison outputs
├── textured_point_cloud/                 # Colorized point cloud PLY output from colorize_point_cloud.py
└── mesh/<map>/                           # Meshing output: .glb (textured), _vertex_colored.ply (full-res), _metadata.json
```

## Unity Virtual Tour

**Unity project location**: `E:\OneDrive\File\Uni\KIT\WS25-26\Master Thesis\unity\VirtualTour\` (Windows/E: drive — Unity Hub cannot create projects on `\\wsl.localhost` paths). Scripts mirrored to `unity/Assets/Scripts/` in this repo.

**Build output**: `E:\OneDrive\File\Uni\KIT\WS25-26\Master Thesis\unity\VirtualTour\Build\VirtualTour.exe`

**Launch command**:
```
VirtualTour.exe --glb <path.glb> --camera-pose <camera_pose.json> [--minimap <img>] [--metadata <json>]
```

### Coordinate Conversion Chain

Point Cloud (Z-up) → GLB (no transforms, trimesh exports as-is) → glTFast (negates X for right-to-left hand conversion) → `Euler(-90,0,0)` rotation → Unity world coordinates.

**Mapping**: `Unity = (-pc_x, pc_z, -pc_y)` where `pc` = point cloud frame.

For camera spawn from `camera_pose.json`: `spawnX = -t[0]`, `spawnZ = -t[1]`, `spawnY = floorY + 1.6`.

### Triangle Winding Fix

glTFast's X-negation flips triangle winding order, which inverts normals. This causes: backface culling holes, CharacterController falling through mesh, wall-to-wall measurement reading 0.000m. **Fix**: reverse triangle indices `[a,b,c] → [a,c,b]` + `RecalculateNormals()` in `GLBLoader.cs` after import, before generating MeshColliders.

### Build Requirements

**Shader Preloader**: Shaders loaded at runtime via `Shader.Find()` get stripped from builds. `ShaderPreloader.cs` (attached to a scene GameObject) holds serialized references to force inclusion:
- `Shader Graphs/glTF-pbrMetallicRoughness` — from Packages > glTFast > Runtime > Shader
- `Shader Graphs/glTF-unlit` — same location
- `Custom/OverlayUnlit` — from Assets/Shaders (measurement line/marker overlay)
- `TextMeshPro/Distance Field Overlay` — from Assets/TextMesh Pro/Shaders (measurement text overlay)

**Build cache lock**: If build fails with "Failed to delete AsyncPluginsFromLinker" or "BurstOutput" (access denied), delete `Library/Bee/artifacts/WinPlayerBuildProgram/` and/or `Temp/BurstOutput/` and retry. Often caused by OneDrive sync or a previous VirtualTour.exe still running.

### Key Scripts

| Script | Purpose |
|--------|---------|
| `AppBootstrap.cs` | CLI arg parsing (`--glb`, `--minimap`, `--metadata`, `--camera-pose`) |
| `GLBLoader.cs` | `File.ReadAllBytes` + `LoadGltfBinary` (UNC-safe), triangle winding fix, MeshCollider generation + yield, camera spawn |
| `FirstPersonController.cs` | WASD walk + F to toggle fly/noclip mode (Q/E up/down). No SnapToGroundHeight (caused oscillation with CharacterController) |
| `ControlsHelpPanel.cs` | HUD showing current mode (Walk/Fly) and key bindings |
| `MeasurementManager.cs` | Tool state machine, raycast input, `IMeasurementTool` interface |
| `PointToPointTool.cs` | Two-click Euclidean distance measurement |
| `WallToWallTool.cs` | Single-click perpendicular wall distance (raycasts along +wallNormal) |
| `HeightTool.cs` | Single-click floor-to-ceiling height |
| `MeasurementRenderer.cs` | Yellow line + red markers + distance label, all overlay-rendered |
| `ShaderPreloader.cs` | Holds shader references to prevent build stripping |
| `OverlayUnlit.shader` | Custom shader with `ZTest Always` for measurement visuals |
| `ToolbarUI.cs` | Top toolbar (1/2/3 shortcuts) + bottom status bar |
| `MinimapController.cs` | Density image overlay with player dot |

## Electron App Development

**Dev launch** (from WSL, renders via WSLg):
```bash
node app/scripts/dev.js
```

`app/scripts/dev.js` replaces the `concurrently`-based `npm run electron:dev` script because `cmd.exe` (used by npm/concurrently on Windows) cannot handle UNC paths (`\\wsl.localhost\...`) as CWD. The script starts Vite and Electron directly via Node.js `spawn()` with explicit `cwd`, bypassing the shell entirely.

**Unity launcher** (`app/src/main/unity-launcher.ts`): Detects WSL via `/proc/sys/fs/binfmt_misc/WSLInterop`, converts file paths from Linux to Windows format using `wslpath -w` before passing to Unity.exe. Auto-discovers `camera_pose.json` from `data/pose_estimates/multiroom/` (prefers rooms over corridors).

**Pipeline engine** (`app/src/main/pipeline-engine.ts`): Spawns Python scripts via `conda run -n <env>`. Currently assumes `conda` is available in PATH (works in WSL where conda is installed).

### External Subdirectories

- `RoomFormer/` — floorplan polygon detection model
- `LGT-Net/` — panoramic layout prediction model
- `PointNeXt/` — point cloud deep learning backbone
- `Open3D-ML/` — 3D geometric processing utilities
- `sam3/` — Segment Anything Model 3
- `3DLineDetection/` — C++ library for fast 3D line segment detection from point clouds (arXiv:1901.02532). Used by `geometry_3d/point_cloud_geometry_baker_V4.py`. Build with CMake in `3DLineDetection/build/`
- `RandLA-Net/` — Point cloud semantic segmentation model (NeurIPS 2019). Models only; not actively used in the pipeline
- `panoramic-localization/` — Research library implementing PICCOLO, CPO, LDL, FGPL localization algorithms. FGPL math reimplemented independently in `pose_estimation/pose_search.py`, `pose_estimation/pose_refine.py`, `pose_estimation/xdf_distance.py`, `geometry_3d/line_clustering_3d.py`. Legacy test scripts (`legacy/test_FGPL_*.py`) call native code for baseline comparison. See `panoramic-localization/CLAUDE.md` for detailed architecture
- `Archive/` — Deprecated script versions: `feature_matchingV2.py` (monolithic pose estimation), `feature_matching.py` (V1 legacy), `compare_filtering_approaches.py`, old renderers, old geometry bakers, old alignment demos
