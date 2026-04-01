# CLAUDE.md

## Project Overview

**scan2measure-webframework** — multi-stage 3D reconstruction pipeline for indoor building measurement. Takes TLS point clouds + panoramic images, estimates camera poses, colors the point cloud, and produces a UV-textured GLB mesh. Orchestrated by an Electron desktop app with a Unity virtual tour viewer.

## Quick Start

No build step or test runner. Dependencies (open3d, torch, shapely, scipy, opencv, matplotlib, pylsd-nova) installed manually. SAM3 scripts require `conda run -n sam3`. Meshing scripts require `conda run -n scan_env`.

**Production pipeline (FGPL/V4 path):**
```bash
python src/preprocessing/generate_density_image.py
python src/segmentation/RoomFormer_inference.py
python src/geometry_3d/point_cloud_geometry_baker_V4.py
python src/geometry_3d/cluster_3d_lines.py
python src/features_2d/image_feature_extractionV2.py
python src/pose_estimation/multiroom_pose_estimation.py
python src/colorization/colorize_point_cloud.py
conda run -n scan_env python src/meshing/test_mvs_texturing.py
```

**Electron dev:** `node app/scripts/dev.js` (WSLg)
**Unity build:** Windows-only at `E:\OneDrive\...\unity\VirtualTour\Build\VirtualTour.exe`

## Source Layout

```
src/
├── preprocessing/       # generate_density_image, ply_to_xyz, map_RoomFormer_results_to_3d
├── segmentation/        # RoomFormer_inference, LGT-Net_inference_demo2
├── geometry_3d/         # point_cloud_geometry_baker_V4, cluster_3d_lines, line_clustering_3d
├── features_2d/         # image_feature_extractionV2, pano_line_detector, line_analysis, pano_processing_virtual_camerasV2, lightweight_synthetic_renderer_V2
├── pose_estimation/     # multiroom_pose_estimation, pose_estimation_pipeline, pose_search, pose_refine, xdf_distance
├── floorplan/           # align_polygons_demo6, polygon_scale_calculation_v2
├── colorization/        # colorize_point_cloud, projection, visibility, color_sampling, evaluate_colorization
├── meshing/             # test_mvs_texturing, mesh_pipeline, mesh_utils, face_visibility, export_gltf, cubemap_utils, debug_projection
├── visualization/       # visualize_pose, visualize_matching
├── utils/               # config_loader, sphere_geometry, downsample_for_preview
└── experiments/         # experiment_local_linefilter, experiment_polygon_prior
app/                     # Electron desktop app (React + TypeScript + Three.js)
unity/Assets/Scripts/    # Unity virtual tour (C# + OverlayUnlit.shader)
archive/                 # Deprecated scripts, experiments, and repos — do not use
```

Cross-subfolder imports use `sys.path.insert(0, str(_SRC_ROOT / "<subfolder>"))`.

## External Dependencies (Compiled)

| Directory | Purpose |
|-----------|---------|
| `3DLineDetection/` | C++ 3D line segment detection (CMake build in `build/`) |
| `mvs-texturing/` | C++ mesh texturing CLI (`build/apps/texrecon/texrecon`). Needs `libpng-dev libjpeg-dev libtiff-dev libtbb-dev`. MVE needs `-std=c++14` patch |
| `PoissonRecon/` | Screened Poisson surface reconstruction CLI |
| `RoomFormer/` | Floorplan polygon detection model |
| `LGT-Net/` | Panoramic layout prediction model |
| `sam3/` | Segment Anything Model 3 |

## Key Conventions

- **Room names**: Scripts use `POINT_CLOUD_NAME` and `ROOM_NAME`/`PANO_NAMES` constants at the top. Update these per dataset.
- **Config**: All pipeline scripts accept `--config <json>` and emit `[PROGRESS]` stdout protocol for Electron. Shared via `utils/config_loader.py`.
- **Serialization**: JSON (coordinates, features, poses), PKL (complex geometry), PNG (images, debug viz), OBJ (line visualizations).
- **camera_pose.json V2 keys**: `rotation` (3x3), `translation` (3-vec), `principal_3d/2d`, `inter_2d/3d`, `matched_pairs`.
- **Coordinate chain**: Point Cloud (Z-up) → GLB (as-is, 1 unit = 1 meter) → glTFast (-X) → Euler(-90,0,0) → Unity. Mapping: `Unity = (-pc_x, pc_z, -pc_y)`.

## Electron App

13 pipeline stages (0-12) in `app/src/shared/constants.ts`. Stages 5 and 10 are confirmation gates. Pipeline spawns Python via `conda run`. Unity launched as subprocess with WSL path translation.

Key files: `app/src/main/index.ts` (IPC + `local-file://` protocol), `pipeline-engine.ts`, `project-store.ts`, `unity-launcher.ts`, `app/src/renderer/pages/PipelinePage.tsx`, `hooks/usePipeline.ts`.

## Unity Virtual Tour

GLTFast runtime loading with triangle winding fix in `GLBLoader.cs`. WASD + fly mode. Measurement tools (point-to-point, wall-to-wall, height). Minimap overlay. `ShaderPreloader.cs` prevents build stripping.

## Thesis Manuscript

Structure spec: `docs/superpowers/specs/2026-04-01-thesis-manuscript-structure-design.md`. KIT Word template (serifen), English, ~40 pages. Template at `E:\OneDrive\File\Uni\KIT\WS25-26\Master Thesis\manuscript\KIT-Vorlage\Dokumentvorlage_A4_serifen_2021-06.docx`.
