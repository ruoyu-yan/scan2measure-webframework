# CLAUDE.md

## Project Overview

**scan2measure-webframework** — multi-stage 3D reconstruction pipeline for indoor building measurement. Takes TLS point clouds + panoramic images, estimates camera poses, colors the point cloud, and produces a UV-textured GLB mesh. Orchestrated by an Electron desktop app with a Unity virtual tour viewer.

## Quick Start

No build step or test runner. Dependencies (open3d, torch, shapely, scipy, opencv, matplotlib, pylsd-nova) installed manually. SAM3 scripts require `conda run -n sam3`. Meshing scripts require `conda run -n scan_env`.

**Production multi-room pipeline (FGPL/V4 path):**
```bash
python src/preprocessing/generate_density_image.py
python src/segmentation/RoomFormer_inference.py
python src/geometry_3d/point_cloud_geometry_baker_V4.py
python src/geometry_3d/cluster_3d_lines.py
python src/features_2d/image_feature_extractionV2.py           # per panorama
python src/pose_estimation/multiroom_pose_estimation.py
python src/colorization/colorize_point_cloud.py
conda run -n scan_env python src/meshing/test_mvs_texturing.py
```

**Electron dev:** `node app/scripts/dev.js` (WSLg)
**Unity build:** Windows-only at `E:\OneDrive\...\unity\VirtualTour\Build\VirtualTour.exe`

## Source Layout

```
src/
├── preprocessing/          # generate_density_image, ply_to_xyz, map_RoomFormer_results_to_3d
├── segmentation/           # RoomFormer_inference, LGT-Net_inference_demo2
├── geometry_3d/            # point_cloud_geometry_baker_V4 (preferred), V3 (legacy), cluster_3d_lines, line_clustering_3d
├── features_2d/            # image_feature_extractionV2 (preferred), V1 (legacy), pano_line_detector, line_analysis, pano_processing_virtual_camerasV2, lightweight_synthetic_renderer_V2
├── pose_estimation/        # multiroom_pose_estimation (prod), pose_estimation_pipeline (single-room), pose_search, pose_refine, xdf_distance
├── floorplan/              # align_polygons_demo6 (preferred), demo5 (legacy), polygon_scale_calculation_v2 (preferred), v1 (legacy), explore_approach1/2 (experimental)
├── visualization/          # visualize_pose, visualize_matching
├── colorization/           # colorize_point_cloud, projection, visibility, color_sampling, evaluate_colorization
├── meshing/                # test_mvs_texturing (preferred), mesh_pipeline (production variant), mesh_reconstruction (legacy vertex-color), mesh_from_images (legacy), mesh_utils, cubemap_utils, face_visibility, export_gltf, debug_projection, test_poissonrecon, test_cgal_advancing_front
├── utils/                  # config_loader, sphere_geometry, downsample_for_preview
├── experiments/            # SAM3_* (8 scripts), experiment_local_linefilter, experiment_polygon_prior, SAM3_mask_to_polygons
└── 00 legacy/              # PnL_solver, pano_processing_virtual_cameras, test_FGPL_*, texture_mapping (empty)
app/                        # Electron desktop app (React + TypeScript + Three.js)
unity/Assets/Scripts/       # Unity virtual tour (15 C# scripts + 1 shader)
tests/                      # Unit tests (meshing module)
```

Cross-subfolder imports use `sys.path.insert(0, str(_SRC_ROOT / "<subfolder>"))` where `_SRC_ROOT` resolves to `src/`.

## Two Feature Extraction Paths

Both feed into the same pose estimation pipeline.

**FGPL path (V4) — preferred:** `point_cloud_geometry_baker_V4.py` (3DLineDetection C++ wrapper, ~100x faster) + `image_feature_extractionV2.py` (sphere-based line extraction via `pano_line_detector` + `line_analysis` + `sphere_geometry`). Outputs `3d_line_map.pkl` + `fgpl_features.json`.

**In-house path (V3) — legacy:** `point_cloud_geometry_baker_V3.py` (plane detection + Binary LP, ~900 lines) + `image_feature_extraction.py` (CLAHE + LSD on 22 perspective crops). Useful when planar decomposition or synthetic rendering is needed.

## Redundant Implementations (Intentionally Kept)

| Current (preferred) | Legacy (kept) | Why kept |
|---------------------|---------------|----------|
| `geometry_3d/point_cloud_geometry_baker_V4.py` | `V3.py` | V3 produces planar decompositions for visualization |
| `features_2d/image_feature_extractionV2.py` | `image_feature_extraction.py` | Feeds V3 path |
| `floorplan/align_polygons_demo6.py` (SAM3-to-SAM3) | `demo5.py` (RoomFormer-to-LGT-Net) | Different input sources |
| `floorplan/polygon_scale_calculation_v2.py` (3 methods) | `polygon_scale_calculation.py` (histogram) | v1 has zero active callers, could be archived |
| `meshing/test_mvs_texturing.py` (texrecon) | `mesh_reconstruction.py` (vertex-color bake) | Fallback when mvs-texturing unavailable |
| `meshing/test_mvs_texturing.py` | `mesh_from_images.py` (Open3D albedo) | Superseded, no active callers |
| `floorplan/explore_approach1_scale_sweep.py` | `explore_approach2_enumerate.py` | Both experimental, consolidated into demo6 |

Additionally, `mesh_pipeline.py` is a production variant of `test_mvs_texturing.py` with PoissonRecon CLI integration and per-face visibility labeling.

## External Subdirectories

| Directory | Purpose |
|-----------|---------|
| `3DLineDetection/` | C++ 3D line segment detection (CMake build in `build/`) |
| `mvs-texturing/` | C++ mesh texturing CLI (`build/apps/texrecon/texrecon`). Needs `libpng-dev libjpeg-dev libtiff-dev libtbb-dev`. MVE dependency needs `-std=c++14` patch |
| `PoissonRecon/` | Screened Poisson surface reconstruction CLI |
| `cgal/` | CGAL geometry library (for advancing front meshing experiments) |
| `RoomFormer/` | Floorplan polygon detection model |
| `LGT-Net/` | Panoramic layout prediction model |
| `sam3/` | Segment Anything Model 3 |
| `panoramic-localization/` | FGPL/PICCOLO/CPO/LDL research library (math reimplemented in `src/`, native code used only by `00 legacy/test_FGPL_*.py` for baseline comparison) |
| `Open3D-ML/`, `PointNeXt/`, `RandLA-Net/` | 3D ML models (not actively used in pipeline) |
| `Archive/` | Deprecated script versions |

## Key Conventions

- **Room names**: Scripts use `POINT_CLOUD_NAME` and `ROOM_NAME`/`PANO_NAMES` constants at the top. Update these to process different datasets.
- **Config**: All pipeline scripts accept `--config <json>` and emit `[PROGRESS]` stdout protocol for Electron integration. Shared via `utils/config_loader.py`.
- **Serialization**: JSON (coordinates, features, poses), PKL (complex geometry), PNG (images, debug viz), OBJ (line visualizations).
- **camera_pose.json**: V2 keys include `rotation` (3x3), `translation` (3-vec), `principal_3d/2d`, `inter_2d/3d`, `matched_pairs`.
- **Coordinate chain** (Unity): Point Cloud (Z-up) → GLB (as-is) → glTFast (-X) → Euler(-90,0,0) → Unity. Mapping: `Unity = (-pc_x, pc_z, -pc_y)`.

## Electron App

13 pipeline stages (0-12) defined in `app/src/shared/constants.ts`. Stages 5 and 10 are confirmation gates. Pipeline spawns Python via `conda run`. Unity launched as subprocess with WSL path translation.

Key files: `app/src/main/index.ts` (IPC + `local-file://` protocol), `pipeline-engine.ts`, `project-store.ts`, `unity-launcher.ts`, `app/src/renderer/pages/PipelinePage.tsx`, `hooks/usePipeline.ts`.

## Unity Virtual Tour

15 C# scripts + `OverlayUnlit.shader`. GLTFast runtime loading with triangle winding fix in `GLBLoader.cs`. WASD + fly mode. Measurement tools (point-to-point, wall-to-wall, height). Minimap overlay. `ShaderPreloader.cs` prevents build stripping.
