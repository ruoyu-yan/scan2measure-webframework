# ROADMAP — scan2measure-webframework

> **Goal**: Color an RGB-less TLS point cloud of an indoor environment using panoramic images taken in the same space, then build a measurable virtual tour from the result.

> **Test case**: TMB office floor

---

## Phase 1: Room-Level Pano-to-Room Assignment (Jigsaw Puzzle Solver) ✅

**Status**: ✅ SAM3-to-SAM3 jigsaw matching implemented and validated. Correct room assignments with 6.5% confidence margin over next-best.

Determine which panorama was taken inside which physical room by matching single-room footprints (from panos) against a multi-room floorplan (from point cloud).

### 1.1 Point Cloud Floorplan Extraction ✅

- `generate_density_image.py` — RANSAC floor plane detection, Manhattan axis alignment, ceiling detection, 2D density projection. Produces PNG + rotation metadata JSON.
- `RoomFormer_inference.py` — ResNet50+Transformer model on density image (not directly on point cloud). Outputs `predictions.json` with room polygon vertices in pixel coordinates.
- `map_RoomFormer_results_to_3d.py` — Inverse-projects pixel polygons to world coordinates using density image metadata. Segments point cloud by room assignment.

### 1.2 Panorama Room Footprint Extraction ✅

- `LGT-Net_inference_demo2.py` — Vanishing point detection, Manhattan frame alignment, LGT-Net inference. Outputs `layout_corners` in meters (floor polygon of the room visible in the panorama).

### 1.3 Scale Estimation and Polygon Matching ✅

**Legacy (RoomFormer ↔ LGT-Net)**:
- `polygon_scale_calculation.py` — Consensus pixel-to-meter scale via histogram voting over pairwise corner distances between RoomFormer (pixels) and LGT-Net (meters) polygons.
- `align_polygons_demo5.py` — Hungarian algorithm matching: tests candidate scales, 4 discrete rotations (0/90/180/270 deg), brute-force vertex-to-vertex snapping. Outputs `global_alignment.json` with per-pair transformation parameters and match errors.

**Current (SAM3 ↔ SAM3)**:
- `align_polygons_demo6.py` — **Canonical jigsaw matching**. Matches SAM3 pano footprints to SAM3 density-image room polygons. Two-stage pipeline:
  - **Stage 1**: Enumerate all M^N pano→room assignments (e.g., 8 for 3 panos × 2 rooms). For each, optimize a shared scale + per-pano (rotation, translation) using `scipy.optimize.differential_evolution` with true IoU scoring (intersection/union prevents scale collapse). Pick the assignment with highest total IoU.
  - **Stage 2**: Refine placement with OBB long-axis alignment (aligns pano shape to room shape), dense translation grid, non-overlap penalty for multiple panos in the same room, largest pano placed first (strongest shape constraint).
  - Outputs `demo6_alignment.json` (room assignment, scale, per-pano rotation/position) + `demo6_alignment.png`
- `polygon_scale_calculation_v2.py` — Three consensus-scale methods (edge distances, Procrustes, area ratio). Used by demo6 legacy modes.
- `SAM3_mask_to_polygons.py` — Converts density image SAM3 masks to world-meter polygon coordinates.

**Key findings** from scale exploration:
- Blind statistical scale methods (edge ratios, Procrustes, area) give unreliable results (0.13 to 1.63) — too few polygons for histogram consensus.
- The enumerate+optimize approach reliably finds scale ≈ 1.2 and correct room assignments (validated against ground truth poses from Phase 2).
- True IoU scoring is essential — containment-only scoring causes scale collapse (optimizer shrinks polygons to trivially fit).
- Coordinate convention: SAM3 pano `layout_corners` [x, z_flipped] map directly to density image [map_x, map_y] (both use "backward" y-direction). No additional Z-flip needed.

### 1.4 Improve Footprint Quality with SAM3 ✅

Both LGT-Net and RoomFormer footprints have limited accuracy. SAM3 produces better footprints from both panoramic images and density images.

**Current state**: Both panoramic and density-image footprint extraction are production-ready. SAM3 footprints integrated into the polygon matching pipeline via `align_polygons_demo6.py`.

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

~~**Next**: Integrate SAM3 panoramic footprints into the polygon matching pipeline~~ → ✅ Done. `align_polygons_demo6.py` implements SAM3-to-SAM3 jigsaw matching with enumerate+optimize approach. Correct room assignments and scale ≈ 1.2 validated against ground truth poses.

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

## Phase 4: Meshing 🔴

**Status**: 🔴 Current Poisson-based approach produces geometrically correct meshes but visually unsatisfactory textured GLBs. The vertex-colored PLY looks correct, but the textured GLB has persistent color artifacts. A new meshing technique needs to be developed.

Convert the colored point cloud into a UV-textured triangle mesh for real-time rendering and measurement.

### 4.1 Surface Reconstruction (Poisson — current, problematic) 🔴

**Approach**: Tiled Screened Poisson reconstruction (Open3D). Selected over Ball Pivoting (unreliable with 7-85 mm density variation) and Gaussian Splatting (not geometrically accurate).

**Pipeline** (`mesh_reconstruction.py` — 17-stage orchestrator):
1. Load colored PLY → 5 mm voxel downsample (5.4M → 4.2M points)
2. Spatial chunking: 6×6 m XY tiles, 1 m overlap → 8 tiles
3. Per-tile: normals → Poisson depth 9 → density trim → ownership trim → vertex color transfer → save to disk
4. Merge tiles → 3.6M triangles (full-res PLY saved for CloudCompare)
5. Decimate to 500K triangles for textured GLB
6. xatlas UV unwrap → bake 4096×4096 texture atlas (vertex color interpolation) → dilate → export GLB

**Known issues**:
- **Vertex-colored PLY looks correct**, but the **textured GLB has color artifacts** that make the scene look unrealistic. The texture baking step (converting per-vertex colors into a UV atlas) loses fidelity. Two bugs were fixed (double UV normalization, KNN cross-surface bleeding) but the visual quality remains unsatisfactory.
- The 7.3× decimation (3.6M → 500K triangles) before UV unwrapping may be too aggressive, degrading vertex color quality that feeds into the atlas bake.
- Poisson reconstruction inherently smooths geometry and can create phantom surfaces in unscanned regions even after density trimming.

**What works**: The geometry (shape, dimensions, metric accuracy) is correct. The vertex-colored PLY is visually faithful to the source point cloud. The issue is specifically in producing a good-looking textured GLB.

**Scripts**: `src/meshing/mesh_reconstruction.py` (orchestrator), `src/meshing/mesh_utils.py` (library), `src/meshing/export_gltf.py` (GLB export)

### 4.2 New Meshing Technique Needed ⬜

The current Poisson + UV atlas approach is fundamentally limited for producing visually high-quality textured meshes from colored point clouds. A new technique should be explored that better preserves the realistic appearance of the source point cloud in the final GLB.

**Possible directions**:
- Alternative surface reconstruction methods (TSDF, neural implicit surfaces, alpha shapes)
- Direct panoramic image projection onto mesh (bypass point cloud colors entirely — project panorama textures onto the mesh using known camera poses)
- Gaussian splatting or point-based rendering (skip triangle meshes for visual quality, keep mesh only for measurement)
- Higher-fidelity texture baking (GPU-based, higher atlas resolution, normal-aware sampling)

**Design spec**: `docs/superpowers/specs/2026-03-23-balanced-mesh-reconstruction-design.md`

---

## Phase 5: End-to-End Desktop App 🟡

**Status**: 🟡 Code implemented (2026-03-25). Electron → Unity "Tour Only" integration tested (2026-03-30). Pipeline stage execution pending.

Two-app desktop system: **Electron** (pipeline orchestrator + 3D preview) + **Unity** (virtual tour + measurement). Designed 2026-03-24, implemented 2026-03-25.

**Design spec**: `docs/superpowers/specs/2026-03-24-desktop-app-design.md`

**Architecture**: Electron app (React + TypeScript + Three.js) handles file management, pipeline orchestration with animated stage visualization, 3D preview, and project management. Unity app (C#, GLTFast) provides first-person navigation with collision, measurement tools, and minimap. Electron launches Unity .exe as subprocess, passing GLB path via CLI.

**Three entry points**:
1. **Full Pipeline** — uncolored PLY + panoramas → all pipeline stages with animation → confirmation gate → colorize → mesh → virtual tour
2. **Mesh Only** — colored PLY → preview → select quality tier → mesh → virtual tour
3. **Tour Only** — existing GLB → launch Unity directly

### 5.1 Python Script Modifications ✅

Add `--config <json>` CLI support and `[PROGRESS]` stdout protocol to all 11 pipeline scripts. Shared `config_loader.py` utility. Required before Electron integration.

**Plan**: `docs/superpowers/plans/2026-03-24-python-script-modifications.md` (14 tasks)

**Completed**: `src/utils/config_loader.py` created. All 11 pipeline scripts import `load_config()` and `progress()`. Scripts fall back to hardcoded defaults when `--config` is not provided, maintaining backward compatibility.

### 5.2 Electron App ✅ (GUI tested)

Pipeline hub with React UI: home screen (3 entry cards + recent projects), pipeline view (sidebar + animated canvas), confirmation gate (draggable polygon correction + re-run), Three.js 3D preview (OBJ/PLY/GLB), project management (JSON store, per-project directories).

**Plan**: `docs/superpowers/plans/2026-03-24-electron-app.md` (37 tasks)

**Completed** (2026-03-25): 31 source files under `app/`. TypeScript compiles clean (both renderer and main process configs, zero errors). npm dependencies installed (558 packages).

**Key files**:
- `app/src/main/index.ts` — Electron main process, IPC handlers, file dialogs
- `app/src/main/pipeline-engine.ts` — Conda subprocess spawner, `[PROGRESS]` parser, process group kill
- `app/src/main/project-store.ts` — JSON-based project CRUD
- `app/src/main/unity-launcher.ts` — Spawns Unity .exe with CLI args, WSL path translation, auto-discovers camera pose
- `app/src/renderer/pages/PipelinePage.tsx` — Pipeline orchestration UI
- `app/src/renderer/components/ConfirmationGate.tsx` — Draggable polygon correction
- `app/src/renderer/components/ThreeViewer.tsx` — Three.js OBJ/PLY/GLB viewer
- `app/src/shared/constants.ts` — Stage definitions, conda envs, script paths
- `app/scripts/dev.js` — Dev server launcher (bypasses cmd.exe UNC path limitations)

**Code-reviewed and fixed**: IPC channel allowlist (security), config JSON generation, React hooks ordering, CSP for file:// fetch, Three.js memory disposal, subprocess process-group kill.

**Dev environment** (2026-03-30): Electron runs from WSL via WSLg. `app/scripts/dev.js` starts Vite + Electron without `concurrently` (cmd.exe cannot handle UNC paths as CWD). Launch: `node app/scripts/dev.js` from repo root.

**Remaining**: Pipeline stage execution test (Electron → conda → Python), stage icons (`app/assets/icons/`), Vite production build.

### 5.3 Unity Virtual Tour ✅ (built and tested)

First-person viewer: GLB runtime loading (GLTFast), WASD + mouse look with CharacterController collision, fly/noclip mode, top toolbar measurement tools (point-to-point, wall-to-wall, height), minimap with density image + player position, mesh info panel.

**Plan**: `docs/superpowers/plans/2026-03-24-unity-virtual-tour.md` (16 tasks)

**Completed** (2026-03-27): 15 C# scripts + 1 custom shader under `unity/Assets/Scripts/` and `unity/Assets/Shaders/`. Unity project built and tested as standalone Windows .exe.

**Key files**:
- `AppBootstrap.cs` — CLI arg parsing (`--glb`, `--minimap`, `--metadata`, `--camera-pose`)
- `GLBLoader.cs` — GLTFast runtime import + triangle winding fix + MeshCollider generation + camera spawn from `camera_pose.json`
- `FirstPersonController.cs` — WASD walk + fly/noclip mode (F toggle, Q/E up/down, camera-relative movement, no collision in fly mode)
- `ControlsHelpPanel.cs` — HUD panel showing current mode (Walk/Fly) and key bindings
- `MeasurementManager.cs` — Tool state machine with `IMeasurementTool` interface
- `PointToPointTool.cs`, `WallToWallTool.cs`, `HeightTool.cs` — Three measurement tools
- `MeasurementRenderer.cs` — Yellow line + red markers + distance labels, overlay-rendered (always visible through geometry)
- `ShaderPreloader.cs` — Holds serialized shader references to prevent build stripping
- `OverlayUnlit.shader` — Custom shader with `ZTest Always` for measurement visuals
- `ToolbarUI.cs` — Top toolbar with keyboard shortcuts (1/2/3/Delete)
- `MinimapController.cs` — Density image overlay with player dot
- `MeshInfoPanel.cs` — Mesh metadata overlay

**Issues fixed** (2026-03-27):
- **Mesh holes + physics fall-through**: glTFast X-negation flips triangle winding → reversed indices in `GLBLoader.cs` after import
- **Camera spawn offset**: Coordinate mapping empirically confirmed as `Unity = (-pc_x, pc_z, -pc_y)`, camera pose uses `spawnX = -t[0]`, `spawnZ = -t[1]`
- **Wall-to-wall 0.000m**: After winding fix, normals point inward → changed raycast to `+wallNormal` direction
- **Measurement visuals occluded by mesh noise**: Created `Custom/OverlayUnlit` shader with `ZTest Always` + TMP `Distance Field Overlay` shader for text
- **Pink mesh in builds**: Shaders loaded via `Shader.Find()` get stripped → `ShaderPreloader.cs` with serialized references
- **Build failure (access denied)**: OneDrive file locking → delete `Library/Bee/artifacts/WinPlayerBuildProgram/` or `Temp/BurstOutput/` and retry

**Issues fixed** (2026-03-30):
- **GLB loading from UNC paths**: `gltf.Load($"file://{path}")` fails on `\\wsl.localhost\...` paths → changed to `File.ReadAllBytes()` + `gltf.LoadGltfBinary(data)` (works with any path format)
- **Walk mode mesh shaking**: `SnapToGroundHeight()` set `transform.position` directly, conflicting with `CharacterController.Move()` every frame → removed SnapToGroundHeight, rely on CharacterController's built-in ground handling
- **Collider cooking race**: Player spawned before MeshCollider was ready → added `await Task.Yield()` after collider generation

### 5.4 Unity Editor Setup + Build ✅

Unity project created in Unity 2022.3 on Windows. GLTFast 6.9.0 + TextMeshPro packages installed. Scene hierarchy built, all scripts attached, Windows .exe build working.

**Unity project path**: `E:\OneDrive\File\Uni\KIT\WS25-26\Master Thesis\unity\VirtualTour\`
**Build output**: `E:\...\unity\VirtualTour\Build\VirtualTour.exe`

**Launch**: `VirtualTour.exe --glb <path.glb> --camera-pose <camera_pose.json>`

### 5.5 Execution Order

1. ~~Python Script Modifications (smallest, unblocks Electron integration)~~ ✅
2. ~~Electron App + Unity App (can be built in parallel after step 1)~~ ✅ Code complete
3. ~~Unity Editor setup + build~~ ✅ Built and tested (2026-03-27)
4. ~~Integration test (Electron launches Unity with GLB)~~ ✅ Tour Only flow tested (2026-03-30)
5. **Next**: Pipeline stage execution test (Electron runs Python scripts via conda)

---

## Phase Summary

| Phase | Description | Status | Key Scripts |
|-------|------------|--------|-------------|
| 1 | Room-level pano assignment (jigsaw puzzle) | ✅ SAM3-to-SAM3 matching validated | `align_polygons_demo6.py`, `SAM3_pano_footprint_extraction.py`, `SAM3_room_segmentation.py` |
| 1.4 | SAM3 for better footprints | ✅ Both pano and density image extraction production-ready | `SAM3_pano_footprint_extraction.py`, `SAM3_mask_to_polygons.py`, `SAM3_room_segmentation.py` |
| 2.1 | 3D feature extraction | ✅ Implemented | `point_cloud_geometry_baker_V4.py`, `cluster_3d_lines.py`, `line_clustering_3d.py` |
| 2.2 | 2D feature extraction | ✅ Implemented | `image_feature_extractionV2.py`, `pano_line_detector.py`, `line_analysis.py` |
| 2.3 | Single-room pose estimation | ✅ Implemented, validated | `pose_estimation_pipeline.py` + library modules |
| 2.4 | Pose visualization | 🟡 Mostly done | `visualize_pose.py` (side-by-side, reprojection, topdown) |
| 2.5 | Multi-room pose estimation | ✅ Implemented | `multiroom_pose_estimation.py` |
| 2.6 | Consolidate pose codebase | ✅ Done | — |
| 2.7 | Jigsaw-informed improvements | 🟡 B+D done, A+C not started | `experiment_polygon_prior.py`, `experiment_local_linefilter.py` |
| 3 | Point cloud coloring | ✅ Implemented, validated | `colorize_point_cloud.py`, `evaluate_colorization.py` + library modules |
| 4 | Meshing | 🔴 Geometry OK, textured GLB visually poor | `mesh_reconstruction.py`, `mesh_utils.py`, `export_gltf.py` |
| 4.2 | New meshing technique | ⬜ Current approach insufficient, needs rethink | — |
| 5.1 | Python script modifications (config + progress) | ✅ All 11 scripts wired | `src/utils/config_loader.py` |
| 5.2 | Electron desktop app | ✅ GUI tested, Tour Only flow working | `app/src/` |
| 5.3 | Unity virtual tour | ✅ Built and tested (15 scripts + 1 shader) | `unity/Assets/Scripts/`, `unity/Assets/Shaders/` |
| 5.4 | Unity Editor setup + build | ✅ Windows .exe working (2026-03-27) | `VirtualTour/Build/VirtualTour.exe` |
| 5.5 | Integration test (Tour Only) | ✅ Electron → Unity with GLB + camera pose (2026-03-30) | `unity-launcher.ts`, `dev.js` |

---

## Dependencies Between Phases

```
Phase 1 (room assignment) ──┐
                             ├──> Phase 2.5 (multi-room pose) ──> Phase 2.7 (jigsaw improvements) ──> Phase 3 (coloring) ✅ ──> Phase 4 (meshing) ✅ ──> Phase 5 (Unity app)
Phase 2.1-2.3 (single-room) ┘
```

Phase 1.4 (SAM3 footprints) can proceed independently at any time.
Phase 5.3 (measurement) depends on Phase 4 (mesh) or can work directly on the point cloud.

---

## Next Steps (as of 2026-03-25)

1. ~~**Evaluate SAM3 footprint quality** (Phase 1.4)~~ — ✅ Done.

2. ~~**Integrate SAM3 footprints into polygon matching** (Phase 1.3 → 1.4)~~ — ✅ Done. `align_polygons_demo6.py` implements enumerate+optimize jigsaw matching. Correct room assignments and scale ≈ 1.2 validated against ground truth poses.

3. **Wire demo6 output into multiroom_pose_estimation.py** (Phase 1 → 2.5) — `multiroom_pose_estimation.py` currently reads `global_alignment.json` from the RoomFormer path. Update it to optionally read `demo6_alignment.json` from `data/sam3_room_segmentation/<map>/` for Voronoi-based local line filtering.

4. **Update the production multi-room pose estimation pipeline** (Phase 2.7 → 2.5) — Integrate the validated local 3D line filtering (Voronoi-based, from `experiment_local_linefilter.py`) into `multiroom_pose_estimation.py`. The experiment proved that filtering 3D lines per-panorama eliminates cross-room false minima (office1: 2.3m error → sub-cm).

5. ~~**Color the point cloud** (Phase 3)~~ — ✅ Done.

6. **Develop new meshing approach** (Phase 4.2) — Current Poisson + UV atlas pipeline produces geometrically correct meshes but visually poor textured GLBs. The vertex-colored PLY looks good but the GLB does not. Need to explore alternative techniques (direct panoramic projection, TSDF, higher-fidelity baking, or non-mesh rendering).

8. ~~**Build end-to-end desktop app** (Phase 5)~~ — ✅ Code complete (2026-03-25). All three sub-plans implemented:
   - ~~Python script modifications (14 tasks)~~ ✅ `config_loader.py` + all 11 scripts wired
   - ~~Electron app (37 tasks)~~ ✅ 31 files, TypeScript compiles clean, code-reviewed and fixed
   - ~~Unity virtual tour (16 tasks)~~ ✅ 13 C# scripts (1,326 LOC), code-reviewed and fixed

9. ~~**Unity Editor setup** (Phase 5.4)~~ — ✅ Done (2026-03-27). Unity project created, all scripts attached, Windows .exe built and tested with `--glb` and `--camera-pose` args.

10. ~~**Integration test (Tour Only)**~~ — ✅ Done (2026-03-30). Electron launches via `node app/scripts/dev.js` (WSLg), Tour Only entry selects GLB, Electron auto-discovers camera pose from `data/pose_estimates/multiroom/`, launches Unity .exe with `--glb` and `--camera-pose` (WSL→Windows path translation via `wslpath -w`). Player spawns inside the office room correctly.

11. **Integration test (Pipeline)** — Test Electron → Python pipeline execution. `pipeline-engine.ts` calls `conda run` which needs to work from the WSL Electron process. Verify `[PROGRESS]` parsing and stage completion flow.
