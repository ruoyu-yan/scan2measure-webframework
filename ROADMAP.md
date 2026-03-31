# ROADMAP — scan2measure-webframework

> **Goal**: Color an RGB-less TLS point cloud of an indoor environment using panoramic images taken in the same space, then build a measurable virtual tour from the result.

> **Test case**: TMB office floor

---

## Phase 1: Room-Level Pano-to-Room Assignment (Jigsaw Puzzle Solver) ✅

**Status**: Complete. SAM3-to-SAM3 jigsaw matching implemented and validated. Correct room assignments with 6.5% confidence margin over next-best.

### 1.1 Point Cloud Floorplan Extraction ✅

- `generate_density_image.py` — RANSAC floor plane, Manhattan alignment, ceiling detection, 2D density projection.
- `RoomFormer_inference.py` — ResNet50+Transformer on density image. Outputs `predictions.json` with room polygon vertices.
- `map_RoomFormer_results_to_3d.py` — Inverse-projects pixel polygons to world coordinates.

### 1.2 Panorama Room Footprint Extraction ✅

- `LGT-Net_inference_demo2.py` — Vanishing point detection, Manhattan alignment, LGT-Net inference. Outputs `layout_corners` in meters.

### 1.3 Scale Estimation and Polygon Matching ✅

**Current (SAM3 ↔ SAM3):**
- `align_polygons_demo6.py` — Canonical jigsaw matching. Stage 1: enumerate all M^N pano→room assignments, optimize shared scale + per-pano rotation/translation via `differential_evolution` with true IoU scoring. Stage 2: OBB long-axis alignment refinement. Outputs `demo6_alignment.json`.
- `polygon_scale_calculation_v2.py` — Three consensus-scale methods (edge, Procrustes, area). Used by demo6 legacy modes.
- `SAM3_mask_to_polygons.py` — Converts density image SAM3 masks to world-meter polygon coordinates.

**Legacy (RoomFormer ↔ LGT-Net):**
- `polygon_scale_calculation.py` — Histogram consensus scale (no active callers).
- `align_polygons_demo5.py` — Hungarian algorithm matching with brute-force vertex snapping.

**Key finding**: True IoU scoring is essential — containment-only scoring causes scale collapse. Enumerate+optimize reliably finds scale ≈ 1.2 and correct room assignments.

### 1.4 SAM3 Footprints ✅

Both panoramic and density-image footprint extraction are production-ready.

**Production script**: `SAM3_pano_footprint_extraction.py` — floor/ceiling fusion with height correction → Manhattan-regularized XZ polygon per panorama. Key insight: naive ceiling mirroring assumes camera at room midpoint; height ratio correction (~1.7) is estimated from boundary data.

**Density image**: `SAM3_room_segmentation.py` — CLAHE+invert + "floor plan" prompt. `SAM3_room_extraction_test.py` — 3-approach experiment (walls-first, room-areas-direct, hybrid).

**Comparison**: `SAM3_footprint_comparison.py` — three fusion variants tested, per-column optimistic (Approach A) selected as production method.

---

## Phase 2: Camera Pose Estimation (Panoramic Localization) ✅

**Status**: Complete. Single-room and multi-room pose estimation implemented and validated. Voronoi-based local line filtering achieves sub-cm accuracy.

### 2.1 3D Feature Extraction ✅

- `point_cloud_geometry_baker_V4.py` — Wrapper around 3DLineDetection C++ binary. PLY → XYZ → `LineFromPointCloud` → `room_geometry.pkl`. ~100x faster than V3.
- `cluster_3d_lines.py` + `line_clustering_3d.py` — FGPL-faithful principal direction voting + classification + 3D intersection → `3d_line_map.pkl`.

### 2.2 2D Feature Extraction ✅

- `image_feature_extractionV2.py` — Orchestrator calling `pano_line_detector.py` (26-view HorizonNet decomposition + LSD + sphere back-projection + segment merging), `line_analysis.py` (vanishing point voting + classification + intersections), `sphere_geometry.py` (coordinates).
- Outputs `fgpl_features.json` (sphere-projected lines + principal directions + intersections).

### 2.3 Single-Room Pose Estimation ✅

`pose_estimation_pipeline.py` — Modular 10-stage FGPL-faithful pipeline. 24 rotation candidates (SVD Procrustes), adaptive quantile translation grid (~1700 pts), `single_pose_compute` XDF search, top-K=10 ICP refinement, quality-ranked selection.

Key improvements over original FGPL: 642 XDF query points (vs 12), rotation diversity in top-K, post-refinement quality selection.

### 2.4 Multi-Room Pose Estimation ✅

`multiroom_pose_estimation.py` — Shared 3D precompute once (~10s), per-panorama 2D + matching (~1s each). Zero imports from `panoramic-localization/`.

### 2.5 Voronoi-Based Local Line Filtering ✅

`experiment_local_linefilter.py` — Each panorama only sees local geometry (nearest-pano assignment + 2.0m overlap margin). Dramatic improvement over baseline: TMB_office1 matches single-room target to sub-cm. Saves R+t to `local_filter_results.json`, consumed by colorization and meshing.

### 2.6 Unimplemented Improvement Ideas ⬜

- **Approach A**: Line-length weighting in XDF cost — weight each line's contribution by arc length to suppress poster noise.
- **Approach C**: Robust line pre-filtering — remove poster-like line clusters via spatial density analysis before intersection finding.

---

## Phase 3: Point Cloud Coloring ✅

**Status**: Complete. Modular pipeline in `src/colorization/`, validated against ground truth.

`colorize_point_cloud.py` orchestrates: equirectangular projection → depth-buffer occlusion → bilinear color sampling → IDW multi-pano blending.

**Test results** (tmb_office_one_corridor_dense, 8.8M pts × 3 panos): 94% coverage, ~20s total. Median RGB L2: 0.025. 77.7% of points have Delta-E < 5. 46.7% below Delta-E 2.3.

`evaluate_colorization.py` — Ground-truth evaluation (RGB L2, CIEDE2000 Delta-E, per-channel bias, error heatmap PLY).

---

## Phase 4: Meshing 🟡

**Status**: mvs-texturing integrated and texture quality confirmed good. Geometry pipeline needs tiled Poisson and double-sided face fix.

### 4.1 Current: mvs-texturing Pipeline 🟡

`test_mvs_texturing.py` — Python wrapper: uncolored PLY → voxel downsample → Poisson reconstruction → export PLY → cubemap conversion (`cubemap_utils.py`) + .cam files → `texrecon` CLI (MRF view selection + seam leveling) → OBJ → GLB via trimesh.

`mesh_pipeline.py` — Production variant with PoissonRecon CLI integration and per-face visibility labeling (`face_visibility.py`).

**What works**: texrecon produces good texture quality (mean brightness 160 vs 106, saturation 20.8 vs 4.6 compared to Open3D approach). Confirmed good in Unity.

**Remaining:**
- [ ] **Tiled Poisson reconstruction** — Current whole-cloud Poisson at depth 7 is too coarse (~23cm/cell). Need to use existing tiled parallel pipeline from `mesh_utils.py` (6x6m tiles, 1m overlap) so depth 8 gives ~4.7cm/cell.
- [ ] **Wall single-side visibility** — Poisson produces single-sided faces. Unity backface culling makes walls only visible from one direction. Fix: double-sided material in GLB export or reverse triangle winding.

### 4.2 Legacy: Vertex-Color Pipeline

`mesh_reconstruction.py` — 17-stage pipeline (Poisson + xatlas UV unwrap + KNN texture bake). Geometry is correct but textured GLB has color artifacts from atlas baking. Kept as fallback.

`mesh_from_images.py` — Open3D `project_images_to_albedo()` approach. Superseded by mvs-texturing.

### 4.3 Experimental

- `test_poissonrecon.py` — Simple PoissonRecon CLI test wrapper.
- `test_cgal_advancing_front.py` — CGAL advancing front reconstruction experiment.

---

## Phase 5: End-to-End Desktop App 🟡

**Status**: Code implemented. Electron → Unity "Tour Only" integration tested. Pipeline stages 0-9 execute successfully. Stages 10-12 (meshing + done) pending.

### 5.1 Python Script Modifications ✅

All 11 pipeline scripts accept `--config <json>` and emit `[PROGRESS]` stdout protocol. Shared `config_loader.py` utility.

### 5.2 Electron App 🟡

Pipeline hub with React UI: home screen (3 entry cards + recent projects), pipeline view (sidebar + filmstrip + per-stage visualization), confirmation gates, Three.js 3D preview, project management.

**Tested**: Stages 0-9 (density image through colorization) run successfully via `conda run`. Per-stage visualization components working (8 new React components: Filmstrip, ImageViewer, ImageGallery, PolygonViewer, ConfirmMatching, ObjViewer, PlyViewer, RunningOverlay).

**Remaining:**
- [ ] Debug ConfirmMatching visualization (coordinate transforms, marker drag)
- [ ] Test stages 10-12 (meshing + done) end-to-end
- [ ] Verify confirm_colorization gate passes quality tier to meshing stage
- [ ] Stage icons, Vite production build
- [ ] Polygon matching performance (~3.5 min for 3 panos due to `differential_evolution`)

### 5.3 Unity Virtual Tour ✅

First-person viewer: GLB runtime loading (GLTFast), WASD + fly/noclip mode, measurement tools (point-to-point, wall-to-wall, height), minimap, mesh info panel. 15 C# scripts + 1 custom shader. Windows .exe build working.

### 5.4 Integration ✅ (partial)

- Tour Only flow: Electron → Unity with GLB + camera pose. Tested 2026-03-30.
- Full pipeline: Stages 0-9 pass. Stages 10-12 untested.

---

## Phase Summary

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | Room-level pano assignment (SAM3-to-SAM3 jigsaw matching) |
| 2 | ✅ | Camera pose estimation (FGPL-faithful, single + multi-room, Voronoi filtering) |
| 3 | ✅ | Point cloud coloring (equirectangular projection + IDW blending) |
| 4 | 🟡 | Meshing (mvs-texturing integrated, needs tiled Poisson + double-sided faces) |
| 5 | 🟡 | Desktop app (Electron stages 0-9 tested, Unity tour working, stages 10-12 pending) |

## Next Steps

1. **Finish mvs-texturing pipeline** (Phase 4) — tiled Poisson reconstruction + double-sided wall fix.
2. **Test pipeline stages 10-12** (Phase 5) — meshing + done stages end-to-end via Electron.
3. **Debug ConfirmMatching visualization** (Phase 5) — coordinate transforms from `demo6_alignment.json` to canvas pixels.
4. **Polygon matching performance** — `differential_evolution` takes ~3.5 min for 3 panos. Consider progress output or faster optimizer.

## Dependencies Between Phases

```
Phase 1 (room assignment) ──┐
                             ├──> Phase 2 (multi-room pose) ──> Phase 3 (coloring) ✅ ──> Phase 4 (meshing) ──> Phase 5 (desktop app)
Phase 2.1-2.3 (single-room) ┘
```
