# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**scan2measure-webframework** is a multi-stage 3D reconstruction pipeline for indoor building floorplans and room layouts. It takes raw point clouds and panoramic images as input and produces calibrated 2D/3D room geometry, camera poses, and scale estimates.

## Two Feature Extraction Approaches

The pipeline supports two independent approaches for extracting geometric features from point clouds and panoramic images. Both feed into the same downstream pose estimation and scale calculation stages.

### In-house pipeline (V3 path)

Custom feature extraction developed for this project.

- **3D features**: `point_cloud_geometry_baker_V3.py` — two-phase plane detection (Open3D region growing → alpha-shape boundaries) + topological wireframe (plane intersections → Binary LP). ~900 lines, slow but produces clean planar decompositions
- **3D rendering**: `lightweight_synthetic_renderer_V2.py` — renders wireframe into 22 synthetic perspective views for visual comparison
- **2D features**: `image_feature_extraction.py` — CLAHE contrast enhancement → anisotropic diffusion → LSD line detection on 22 virtual perspective crops (from `pano_processing_virtual_camerasV2.py`)
- **Output**: `extracted_2d_lines.json` (pinhole-projected line segments) + `visible_3d_lines.json` (per-view 3D wireframe)

### FGPL pipeline (V4 path)

Feature extraction following the algorithms from the FGPL panoramic localization paper (CVPR 2024). Reimplemented independently with no imports from `panoramic-localization/`.

- **3D features**: `point_cloud_geometry_baker_V4.py` — thin wrapper around `3DLineDetection` C++ binary for ~100x speedup. PLY → XYZ → `LineFromPointCloud` → parse lines.obj → `room_geometry.pkl`
- **2D features**: `image_feature_extractionV2.py` — orchestrator that calls three library modules:
  - `sphere_geometry.py` — icosphere generation, equirectangular ↔ sphere coordinate conversions, panoramic rasterization
  - `pano_line_detector.py` — 26-view panorama decomposition (HorizonNet tangent-plane projection), LSD detection, sphere back-projection, 3-pass colinear segment merging. Applies 90° Z-rotation to align HorizonNet coordinates with equirectangular rendering frame
  - `line_analysis.py` — icosphere voting for vanishing points, line classification into 3 principal groups, great-circle arc intersection finding
- **Output**: `fgpl_features.json` (sphere-projected lines + principal directions + intersections) + `edge_overlay.png` + `grouped_lines.png` in `data/pano/2d_feature_extracted/<room>_v2/`

### Which to use

The FGPL path (V4) is preferred for pose estimation via `feature_matchingV2.py` — the 3D and 2D features are in compatible sphere-based representations and the intersection-based matching is more robust. The in-house path (V3) is useful when planar decomposition or synthetic rendering is needed.

## Running Pipeline Stages

All scripts are run individually from the repo root or from `src/`. Each uses `Path(__file__).resolve()` internally to find data paths, so working directory does not matter.

**Shared stages (both paths):**
```bash
python src/generate_density_image.py            # Stage 1: Point cloud → density image
python src/RoomFormer_inference.py               # Stage 2: Density image → 2D floorplan polygons
```

**FGPL path (V4) — preferred:**
```bash
python src/point_cloud_geometry_baker_V4.py      # Stage 3: Point cloud → 3D wireframe (3DLineDetection)
python src/image_feature_extractionV2.py         # Stage 4: Panoramic image → sphere-based 2D line features
python src/feature_matchingV2.py                 # Stage 5: FGPL-faithful camera pose estimation
python src/polygon_scale_calculation.py          # Stage 6: Compute pixel-to-meter scale
```

**In-house path (V3):**
```bash
python src/point_cloud_geometry_baker_V3.py      # Stage 3: Point cloud → 3D wireframe (plane detection + Binary LP)
python src/lightweight_synthetic_renderer_V2.py  # Stage 4a: 3D wireframe → synthetic views
python src/pano_processing_virtual_camerasV2.py  # Stage 4b: Panorama → 22 perspective crops
python src/image_feature_extraction.py           # Stage 4c: Perspective crops → 2D line features
python src/feature_matchingV2.py                 # Stage 5: Camera pose estimation
python src/polygon_scale_calculation.py          # Stage 6: Compute pixel-to-meter scale
```

Stage 5 alternatives:
- `feature_matchingV2.py` — **preferred**. FGPL-faithful pipeline with canonical frame precomputation, XDF inlier-counting cost, vectorized intersections, and full R+t ICP refinement via `PnL_solver.refine_pose_full()`. Outputs V2 keys (`inter_2d`, `inter_3d`, `matched_pairs`) in `camera_pose.json`.
- `feature_matching.py` — V1 (legacy). Simplified FGPL with translation-only ICP refinement via `PnL_solver.refine_pose_icp()`.

Supporting scripts:
```bash
python src/visualize_matching.py                 # Generate 2-panel overlay PNGs for pose inspection (supports V1 + V2)
python src/align_polygons_demo5.py               # Hungarian algorithm polygon matching
python src/map_RoomFormer_results_to_3d.py       # Project 2D floorplan polygons back to 3D point cloud
python src/LGT-Net_inference_demo2.py            # Panoramic layout prediction via LGT-Net
python src/SAM3_inference.py                     # Segment Anything Model 3 inference demo
python src/ply_to_xyz.py                         # Convert PLY to plain-text XYZ (preprocessing for 3DLineDetection)
python src/test_FGPL_3d_feature.py               # Test FGPL native 3D line clustering & intersection on OBJ data
```

There is no build step, test runner, or package install — dependencies (open3d, torch, shapely, scipy, opencv, matplotlib, pylsd-nova) must be installed manually. External models and libraries live as subdirectories at the repo root.

## Architecture

### Data Flow

```
=== Shared stages ===

data/raw_point_cloud/*.ply
  → generate_density_image.py
  → data/density_image/<room>/  (PNG + rotation metadata JSON)
  → RoomFormer_inference.py
  → data/reconstructed_floorplans_RoomFormer/<room>/  (polygon JSON + images)

=== FGPL path (V4) — 3D features ===

data/raw_point_cloud/*.ply
  → point_cloud_geometry_baker_V4.py  (calls 3DLineDetection C++ binary)
  → room_geometry.pkl  (wireframe segments: dirs, starts, ends)

=== FGPL path (V4) — 2D features ===

data/pano/raw/<room>.jpg
  → image_feature_extractionV2.py
    (calls pano_line_detector.py → line_analysis.py → sphere_geometry.py)
  → data/pano/2d_feature_extracted/<room>_v2/
    (fgpl_features.json + edge_overlay.png + grouped_lines.png)

=== In-house path (V3) — 3D features ===

data/raw_point_cloud/*.ply
  → point_cloud_geometry_baker_V3.py  (plane detection + Binary LP wireframe)
  → room_geometry.pkl
  → lightweight_synthetic_renderer_V2.py
  → data/debug_renderer/<room>/  (synthetic PNGs + visible_3d_lines.json)

=== In-house path (V3) — 2D features ===

data/pano/<room>/  (panoramic images)
  → pano_processing_virtual_camerasV2.py  (22 perspective crops)
  → image_feature_extraction.py  (CLAHE + anisotropic diffusion + LSD)
  → data/pano/2d_feature_extracted/<room>/  (extracted_2d_lines.json + debug images)

=== Pose estimation (both paths) ===

feature_matchingV2.py + PnL_solver.py  (loads room_geometry.pkl + 2D features → camera_pose.json)
visualize_matching.py  (loads camera_pose.json → 2-panel debug PNGs in data/pose_estimates/<room>/vis/)
align_polygons_demo5.py + polygon_scale_calculation.py  (align floorplans, output scale)
```

### Key Scripts

| Script | Core Logic |
|--------|-----------|
| `generate_density_image.py` | RANSAC floor plane detection, Manhattan World axis alignment, ceiling detection, 2D density projection |
| `RoomFormer_inference.py` | ResNet50+Transformer (RoomFormer model), area filtering via Shapely, polygon probability threshold filtering |
| `point_cloud_geometry_baker_V4.py` | Python wrapper for 3DLineDetection C++ binary: PLY → XYZ → `LineFromPointCloud` → parse lines.obj → room_geometry.pkl |
| `point_cloud_geometry_baker_V3.py` | Legacy: Two-phase plane detection (Open3D region growing → alpha-shape boundaries) + topological wireframe (plane intersections → Binary LP). ~900 lines |
| `lightweight_synthetic_renderer_V2.py` | 22-view spherical tiling (60° FOV, 1024x1024), z-buffer depth rendering, 3-channel output (wireframe / edges / regions) |
| `image_feature_extraction.py` | In-house 2D: CLAHE → anisotropic diffusion → LSD line detection on 22 perspective crops |
| `image_feature_extractionV2.py` | FGPL 2D: orchestrator calling `pano_line_detector` → `line_analysis` → `sphere_geometry` for sphere-based line extraction, vanishing points, and intersections |
| `sphere_geometry.py` | Library: icosphere generation, equirectangular ↔ sphere projection, panoramic point/line rasterization |
| `pano_line_detector.py` | Library: 26-view HorizonNet-style panorama decomposition, LSD detection, tangent-plane back-projection to sphere, 3-pass segment merging |
| `line_analysis.py` | Library: icosphere voting for 3 vanishing points, line classification into principal groups, great-circle arc intersection finding |
| `feature_matchingV2.py` | **Preferred**. 9-stage FGPL-faithful pipeline: load 3D wireframe → sphere back-project 2D lines → principal direction voting (icosphere) → 24 rotation candidates → vectorized 2D/3D intersections → translation grid with chamfer filtering → canonical-frame XDF inlier-counting cost → full R+t ICP refinement → camera_pose.json with V2 keys |
| `feature_matching.py` | V1 legacy. 8-stage simplified FGPL: load lines → sphere back-project → principal direction voting → 24 rotation candidates → translation grid → XDF cost → translation-only ICP refinement → camera_pose.json |
| `PnL_solver.py` | ICP-style pose refinement module (imported by feature_matching scripts, not run standalone). Contains `refine_pose_icp()` (V1: translation-only) and `refine_pose_full()` (V2: two-phase R+t with grouped mutual-NN matching) |
| `align_polygons_demo5.py` | Hungarian algorithm polygon matching (RoomFormer 2D ↔ LGT-Net 3D) |
| `polygon_scale_calculation.py` | Consensus scale estimation via histogram peak detection over pairwise polygon corner distances |
| `visualize_matching.py` | 22 side-by-side PNGs per panorama. V2 layout: left = 2D lines (green) + projected 3D wireframe (cyan) overlay, right = matched intersection pairs colored by group. Falls back to V1 layout (separate 2D/3D panels) when V2 keys absent |
| `map_RoomFormer_results_to_3d.py` | Inverse-projects RoomFormer 2D polygons to world coordinates, segments point cloud by room |
| `pano_processing_virtual_camerasV2.py` | Extracts 22 perspective crops from equirectangular panorama (matches renderer's spherical tiling) |
| `LGT-Net_inference_demo2.py` | Panoramic layout prediction: Manhattan frame detection, vanishing point alignment, depth + layout boundary output |
| `SAM3_inference.py` | Segment Anything Model 3: text-prompted segmentation, returns masks + bounding boxes + confidence |
| `ply_to_xyz.py` | Utility: convert PLY → plain-text XYZ (one point per line, used by 3DLineDetection) |
| `test_FGPL_3d_feature.py` | Test script: converts OBJ → FGPL TXT, calls native `generate_line_map_single()` for principal direction voting + 3D intersection, prints stats + 3D visualization |

### `feature_matchingV2.py` — FGPL-Faithful Pose Estimation

9-stage pipeline using PyTorch tensors throughout:

1. **Load 3D wireframe** — `room_geometry.pkl` → dirs, lengths, filter by min length
2. **Load 2D lines** — `extracted_2d_lines.json` → pinhole inverse → sphere back-projection → `(N, 9)` tensor `[normal, start, end]`
3. **Principal directions** — icosphere voting (level 5, ~2562 pts): argmax-bincount for 3D, great-circle membership for 2D. Ensures `det > 0`
4. **24 rotation candidates** — 6 permutations × 4 det-preserving sign flips, SVD Procrustes alignment
5. **Line intersections** — 2D: cross-product of great-circle normals + arc membership. 3D: closest-point formula + parametric test. 3 groups each
6. **Translation candidates** — uniform 3D grid in wireframe bbox, chamfer-filtered (~1700 pts)
7. **XDF coarse search** — canonical frame precomputation: LDF + PDF distance functions, inlier counting `|d_2d - d_3d| < 0.1`, shape `(N_t, N_r)` → top-K poses
8. **Full ICP refinement** — `PnL_solver.refine_pose_full()`: Phase 1 = translation (100 iters), Phase 2 = rotation via YPR (50 iters)
9. **Output** — `camera_pose.json`

Key configuration constants (top of file):
- `VOTE_SPHERE_LEVEL = 5`, `QUERY_SPHERE_LEVEL = 3`
- `XDF_INLIER_THRES = 0.1`, `POINT_GAMMA = 0.2`
- `CHAMFER_MIN_DIST = 0.3`, `NUM_TRANS = 1700`

### `camera_pose.json` Output Format

Common keys (V1 + V2):
- `rotation` — `(3,3)` world-to-camera rotation matrix
- `translation` — `(3,)` camera position in world frame
- `principal_3d`, `principal_2d` — `(3,3)` principal direction matrices
- `n_inter_matched` — number of matched intersection pairs
- `xdf_cost_coarse` — best coarse XDF cost

V2-only keys (added by `feature_matchingV2.py`):
- `inter_2d` — list of 3 arrays, each `(M_k, 3)` — 2D intersection sphere points per group
- `inter_3d` — list of 3 arrays, each `(N_k, 3)` — 3D intersection world points per group
- `matched_pairs` — list of 3 arrays, each `(P_k, 2)` — mutual-NN index pairs `[i_2d, i_3d]` from ICP

`visualize_matching.py` detects V2 keys and switches layout accordingly.

### `point_cloud_geometry_baker_V3.py` — Legacy Core Algorithm

Superseded by V4 for ~100x speedup, but still available. Key tunable parameters:

- `coplanar_merge_distance = 0.08m` — separates walls from cabinets; increase if walls merge incorrectly
- `min_hull_area = 0.90 m²` — minimum plane area to keep; increase to filter small planes
- `manhattan_snap_deg = 15°` — snaps normals to ±X/Y/Z axes; reduce for non-Manhattan spaces
- `alpha = 2.0` — alpha-shape boundary smoothness; reduce for tighter boundaries
- `lambda_complexity = 0.1` — Binary LP trade-off: higher = fewer edges, sparser wireframe

### Hardcoded Room Name

Most scripts use two name constants at the top: `POINT_CLOUD_NAME` (matches directory under `data/debug_renderer/` or `data/raw_point_cloud/`) and `ROOM_NAME` (matches directory under `data/pano/`). Currently set to `tmb_office1` / `TMB_office1` in V2 scripts and `tmb_office1_subsampled` in some older scripts. Update these constants in each script to process a different room.

### Data Serialization Conventions

- **JSON**: polygon coordinates, 2D/3D line features, rotation metadata, camera_pose.json
- **PKL**: complex geometry objects (`room_geometry.pkl`)
- **PNG**: density images, synthetic renders, debug visualizations
- **OBJ**: 3DLineDetection output (vertices + line segments)

### External Subdirectories

- `RoomFormer/` — floorplan polygon detection model
- `LGT-Net/` — panoramic layout prediction model
- `PointNeXt/` — point cloud deep learning backbone
- `Open3D-ML/` — 3D geometric processing utilities
- `sam3/` — Segment Anything Model 3
- `3DLineDetection/` — C++ library for fast 3D line segment detection from point clouds (arXiv:1901.02532). Used by `point_cloud_geometry_baker_V4.py`. Build with CMake in `3DLineDetection/build/`
- `panoramic-localization/` — Research library implementing PICCOLO, CPO, LDL, FGPL localization algorithms. FGPL math (line intersections, sphere ICP, canonical frame precomputation) adopted by `feature_matchingV2.py` and `PnL_solver.py`. See `panoramic-localization/CLAUDE.md` for detailed architecture
- `Archive/` — Deprecated script versions (V1/V2 renderers, old geometry bakers, old alignment demos)
