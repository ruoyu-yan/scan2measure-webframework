# ROADMAP — scan2measure-webframework

> **Goal**: Color an RGB-less TLS point cloud of an indoor environment using panoramic images taken in the same space, then build a measurable virtual tour from the result.

> **Test case**: TMB office floor

---

## Phase 1: Room-Level Pano-to-Room Assignment (Jigsaw Puzzle Solver)

**Status**: Core pipeline implemented, accuracy limited by upstream polygon quality.

Determine which panorama was taken inside which physical room by matching single-room footprints (from panos) against a multi-room floorplan (from point cloud).

### 1.1 Point Cloud Floorplan Extraction (implemented)

- `generate_density_image.py` — RANSAC floor plane detection, Manhattan axis alignment, ceiling detection, 2D density projection. Produces PNG + rotation metadata JSON.
- `RoomFormer_inference.py` — ResNet50+Transformer model on density image (not directly on point cloud). Outputs `predictions.json` with room polygon vertices in pixel coordinates.
- `map_RoomFormer_results_to_3d.py` — Inverse-projects pixel polygons to world coordinates using density image metadata. Segments point cloud by room assignment.

### 1.2 Panorama Room Footprint Extraction (implemented)

- `LGT-Net_inference_demo2.py` — Vanishing point detection, Manhattan frame alignment, LGT-Net inference. Outputs `layout_corners` in meters (floor polygon of the room visible in the panorama).

### 1.3 Scale Estimation and Polygon Matching (implemented)

- `polygon_scale_calculation.py` — Consensus pixel-to-meter scale via histogram voting over pairwise corner distances between RoomFormer (pixels) and LGT-Net (meters) polygons.
- `align_polygons_demo5.py` — Hungarian algorithm matching: tests candidate scales, 4 discrete rotations (0/90/180/270 deg), brute-force vertex-to-vertex snapping. Outputs `global_alignment.json` with per-pair transformation parameters and match errors.

### 1.4 Improve Footprint Quality with SAM3 (planned)

Both LGT-Net and RoomFormer footprints have limited accuracy. Investigate using Segment Anything Model 3 for extracting room boundaries from:
- Panoramic images (replacing or augmenting LGT-Net footprints)
- Density images (replacing or augmenting RoomFormer polygons)

**Current state**: `SAM3_inference.py` is a disconnected demo (text-prompted segmentation on a tennis player image). Needs integration with the main pipeline.

**Open questions**:
- Can SAM3 reliably segment room floor boundaries from equirectangular panoramas?
- Does SAM3 on density images outperform RoomFormer's learned polygon detection?
- What prompting strategy works best (text prompts, point prompts, box prompts)?

---

## Phase 2: Camera Pose Estimation (Panoramic Localization)

**Status**: Single-room and multi-room pose estimation implemented and validated.

Estimate the exact 6-DoF camera pose (position + orientation) for each panorama within the point cloud coordinate frame.

### 2.1 3D Feature Extraction (implemented)

- `point_cloud_geometry_baker_V4.py` — Thin wrapper around the `3DLineDetection` C++ binary. PLY -> XYZ -> `LineFromPointCloud` -> parse `lines.obj` -> `room_geometry.pkl`. ~100x faster than the legacy V3 plane detection approach.

### 2.2 2D Feature Extraction (implemented, cleanup next)

- `image_feature_extractionV2.py` — Orchestrator calling three library modules:
  - `pano_line_detector.py` — 26-view HorizonNet-style decomposition, LSD detection, exact pinhole back-projection to sphere, 3-pass colinear segment merging
  - `line_analysis.py` — Icosphere voting for 3 vanishing points, line classification into principal groups, great-circle arc intersection finding
  - `sphere_geometry.py` — Icosphere generation, equirectangular/sphere coordinate conversions
- Outputs `fgpl_features.json` (sphere-projected lines + principal directions + intersections)

**Next**: Clean up the 2D feature extraction pipeline — simplify orchestration, reduce redundancy, improve robustness for diverse panorama inputs (corridors, rooms with poster-heavy walls, etc.).

### 2.3 Single-Room Pose Estimation (implemented)

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

### 2.4 Pose Estimation Visualization Improvement (mostly done)

`pose_estimation_pipeline.py` now generates three visualizations:
- **Side-by-side**: 2D lines (green) + projected 3D wireframe (cyan) overlay — shows feature matching quality
- **Reprojection**: Depth-coded point cloud projected onto panorama — shows pose correctness directly
- **Top-down**: Camera position + orientation arrow on density image with RoomFormer room polygons

Still missing:
- **Point cloud overlay**: Textured sphere at camera position inside 3D point cloud
- **Before/after comparison**: Side-by-side of estimated vs ground-truth pose (if available)

### 2.5 Multi-Room Pose Estimation (implemented)

`multiroom_pose_estimation.py` — orchestrator that estimates camera poses for multiple panoramas against a shared multi-room 3D wireframe map. Zero imports from `panoramic-localization/`.

**Architecture**: Split `xdf_coarse_search` in `pose_search.py` into:
- `precompute_xdf_3d()` — one-time 3D canonical frame + LDF/PDF precomputation (~10s)
- `xdf_coarse_search_from_precomputed()` — per-panorama 2D computation + cost + top-K (~1s each)

**Results** (test case: `tmb_office_one_corridor_dense` map, 3 panoramas):
- TMB_office1: excellent (n_tight=283, avg_dist=0.039)
- TMB_corridor_south2: excellent (n_tight=185, avg_dist=0.024)
- TMB_corridor_south1: poor (n_tight=70, avg_dist=0.121) — known difficult case, motivates 2.7 improvements

### 2.6 Consolidate Pose Estimation Codebase (done)

Resolved: `pose_estimation_pipeline.py` (modular) is the canonical single-room implementation. `multiroom_pose_estimation.py` is the canonical multi-room implementation. `feature_matchingV2.py` (monolithic) archived to `Archive/`. The `PnL_solver.py` module is now unused by the active pipeline (superseded by `pose_refine.py`) but kept for reference.

### 2.7 Jigsaw-Informed Pose Estimation Improvements (next after 2D cleanup)

Three approaches to leverage jigsaw puzzle room assignments and improve pose accuracy, especially for difficult panoramas (e.g., corridor_south1):

#### Approach A: Line-length weighting in XDF cost (recommended first)

Structural lines (walls, ceiling, floor) are long great-circle arcs on the sphere. Poster/clutter lines are short, clustered segments. Currently every classified line contributes equally to LDF and every intersection contributes equally to PDF.

**Idea**: Weight each line's contribution to the distance function by its arc length. A 40° wall line gets 8x the influence of a 5° poster fragment. This naturally suppresses poster noise without needing a perfect classification threshold.

- Touches: `xdf_distance.py` (LDF computation), possibly `find_intersections_2d_indexed` (weight intersections by parent line lengths)
- Risk: low — additive change to existing cost, doesn't break working cases
- Effort: moderate — need to thread weights through the precompute pipeline

#### Approach B: Jigsaw-guided translation prior

The jigsaw puzzle already tells us which RoomFormer polygon the camera belongs to. Use that polygon's bounding box (+ margin) to **prune translation candidates** before XDF coarse search.

**Idea**: Instead of searching ~1300 translation candidates across the entire multi-room map, restrict to the ~200-400 that fall within the matched room's footprint. Fewer candidates = less chance of a false match in a geometrically similar region.

- Touches: `multiroom_pose_estimation.py` (translation filtering), needs polygon data from jigsaw output
- Risk: medium — if jigsaw assignment is wrong, correct pose is excluded entirely
- Effort: low — just a bounding-box mask on `trans_candidates`

#### Approach C: Robust line pre-filtering (spatial density)

Detect and remove poster-like line clusters before matching. Structural lines are spatially distributed across the panorama; poster lines cluster in a small angular region.

**Idea**: Compute local density of line midpoints on the sphere. Lines in high-density clusters (>N lines within radius r) get removed or down-weighted.

- Touches: new preprocessing step before `find_intersections_2d_indexed`
- Risk: medium — might accidentally remove legitimate dense structural features (e.g., window grids, door frames)
- Effort: moderate — new spatial analysis code

**Recommended order**: A first (most principled, universally applicable), then B (leverages existing jigsaw data), then C as fallback if A+B aren't sufficient.

---

## Phase 3: Point Cloud Coloring

**Status**: Not implemented. `texture_mapping.py` exists as an empty placeholder (0 bytes).

With camera poses estimated (Phase 2), project panoramic RGB pixels onto point cloud vertices.

### 3.1 Single-Pano Projection (planned)

For each panorama with a known pose (R, t):
1. Transform each point cloud vertex to the camera frame: `p_cam = R @ (p_world - t)`
2. Project to equirectangular coordinates: `(theta, phi) = (atan2(x, z), asin(y/r))`
3. Sample the panorama pixel at `(theta, phi)` -> RGB color
4. Handle occlusion: only color points that are visible from the camera (not behind walls)

**Key challenges**:
- **Occlusion handling**: Need depth testing or ray casting to avoid coloring occluded points with wrong colors
- **Grazing angles**: Points viewed at steep angles produce stretched/unreliable colors — need angle-of-incidence weighting
- **Distance weighting**: Closer points get more reliable color than distant ones

### 3.2 Multi-Pano Blending (planned)

When a point is visible from multiple panoramas:
- Weight contributions by viewing angle quality (prefer frontal views)
- Weight by distance (prefer closer cameras)
- Blend RGB values (weighted average or best-view selection)
- Handle exposure/white-balance differences between panoramas

### 3.3 Coverage and Quality Assessment (planned)

- Identify uncolored regions (points not visible from any panorama)
- Visualize coverage map on the point cloud
- Flag areas needing additional panorama captures

---

## Phase 4: Meshing

**Status**: Not implemented. No meshing code exists in the repository.

Convert the colored point cloud into a watertight triangle mesh suitable for real-time rendering and measurement.

### 4.1 Surface Reconstruction (planned)

Candidate approaches:
- **Poisson surface reconstruction** (Open3D `create_from_point_cloud_poisson`) — requires oriented normals, produces watertight mesh, good for smooth indoor surfaces
- **Ball-pivoting algorithm** — better for preserving sharp edges (walls, furniture)
- **Alpha shapes** — already partially used in `point_cloud_geometry_baker_V3.py` for boundary extraction

### 4.2 Texture Baking (planned)

Transfer per-vertex colors to a UV-mapped texture atlas:
- Generate UV parameterization for the mesh
- Bake vertex colors (or direct panorama projections) into texture images
- Produce glTF/GLB format for web consumption

### 4.3 Mesh Optimization (planned)

- Decimation to reduce triangle count for web rendering
- Normal smoothing for visual quality
- Hole filling for scanning gaps

---

## Phase 5: Virtual Tour with Measurement

**Status**: Not implemented. No web framework or viewer code exists (despite the repository name).

Build an interactive web-based virtual tour from the textured mesh, with real-world measurement capability.

### 5.1 3D Viewer Setup (planned)

Technology candidates:
- **Potree** — purpose-built for large point clouds, supports measurements natively
- **Three.js** — general-purpose WebGL, requires more custom code but maximum flexibility
- **3D Tiles / CesiumJS** — good for very large scenes with level-of-detail

### 5.2 Virtual Tour Navigation (planned)

- Hotspot-based navigation between panorama viewpoints
- Smooth camera transitions between rooms
- Minimap showing current position on floorplan (reuse RoomFormer output)
- Optional: free-fly camera mode through the 3D model

### 5.3 Measurement Tool (planned)

Since the point cloud is from a TLS (terrestrial laser scanner), the geometry is metrically accurate:
- Point-to-point distance measurement (click two points, display distance in meters)
- Area measurement (select polygon on surface)
- Height measurement (vertical distance between two horizontal planes)
- Export measurements as CSV/JSON

### 5.4 Deployment (planned)

- Static site hosting (the viewer is client-side)
- Data pipeline: mesh + textures -> 3D Tiles or Potree octree format
- Progressive loading for large scenes

---

## Phase Summary

| Phase | Description | Status | Key Scripts |
|-------|------------|--------|-------------|
| 1 | Room-level pano assignment (jigsaw puzzle) | Core implemented, accuracy limited | `align_polygons_demo5.py`, `LGT-Net_inference_demo2.py`, `RoomFormer_inference.py` |
| 1.4 | SAM3 for better footprints | Planned (demo only) | `SAM3_inference.py` (disconnected) |
| 2.1 | 3D feature extraction | Implemented | `point_cloud_geometry_baker_V4.py` |
| 2.2 | 2D feature extraction | Implemented, **cleanup next** | `image_feature_extractionV2.py`, `pano_line_detector.py`, `line_analysis.py` |
| 2.3 | Single-room pose estimation | Implemented, validated | `pose_estimation_pipeline.py` + library modules |
| 2.4 | Pose visualization | Mostly done | `visualize_pose.py` (side-by-side, reprojection, topdown) |
| 2.5 | Multi-room pose estimation | **Implemented** | `multiroom_pose_estimation.py` |
| 2.7 | Jigsaw-informed improvements | **Next after 2.2 cleanup** | Approaches A (line-length weight), B (translation prior), C (density filter) |
| 3 | Point cloud coloring | Not started | `texture_mapping.py` (empty) |
| 4 | Meshing | Not started | — |
| 5 | Virtual tour + measurement | Not started | — |

---

## Dependencies Between Phases

```
Phase 1 (room assignment) ──┐
                             ├──> Phase 2.5 (multi-room pose) ──> Phase 2.7 (jigsaw improvements) ──> Phase 3 (coloring) ──> Phase 4 (meshing) ──> Phase 5 (virtual tour)
Phase 2.1-2.3 (single-room) ┘

Current focus:
  2.2 cleanup ──> 2.7A (line-length weighting) ──> 2.7B (jigsaw translation prior) ──> 2.7C (density filter)
```

Phase 2.4 (visualization improvement) can proceed independently at any time.
Phase 1.4 (SAM3 footprints) can proceed independently at any time.
Phase 5.3 (measurement) depends on Phase 4 (mesh) or can work directly on the point cloud.
