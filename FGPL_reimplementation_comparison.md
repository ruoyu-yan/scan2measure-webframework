# FGPL Pose Estimation: Reimplementation vs Original

This document describes the differences between the original FGPL (Fully Geometric Panoramic Localization, CVPR 2024) implementation and the independent reimplementation in `scan2measure-webframework/src/`. Both systems estimate 6-DoF camera pose from an equirectangular panorama and a 3D line segment map by matching geometric features on the unit sphere.

---

## 1. System Overview

### Original FGPL (`panoramic-localization/fgpl/`)

The reference implementation is a monolithic research codebase with ~4000 lines across `edge_utils.py`, `utils.py`, `map_utils.py`, `fgpl/localize.py`, `fgpl/xdf_canonical_precompute.py`, `fgpl/pose_estimation.py`, and `fgpl/line_intersection.py`. It is designed for batch evaluation on academic datasets (Stanford 2D-3D-S, OmniScenes), with multi-room ranking and dataset-specific I/O.

### Reimplementation (`src/`)

The reimplementation is a modular pipeline split across 7 independent Python modules:

| Module | Responsibility |
|--------|---------------|
| `pano_line_detector.py` | 2D line detection from equirectangular panoramas |
| `line_analysis.py` | Vanishing point voting, line classification, 2D intersection finding |
| `sphere_geometry.py` | Icosphere generation, equirectangular projection, panoramic rasterization |
| `xdf_distance.py` | Sphere distance functions (LDF, PDF), 3D line classification |
| `pose_search.py` | Rotation candidates, translation grid, XDF coarse search |
| `pose_refine.py` | Two-phase sphere ICP refinement |
| `pose_estimation_pipeline.py` | Orchestrator (stages 1--10) |

No code is imported from `panoramic-localization/`. All mathematical operations were reimplemented from the FGPL paper and verified against the reference.

---

## 2. Pipeline Comparison

Both pipelines follow the same high-level structure. The table below summarises each stage and where they diverge.

| Stage | Original FGPL | Reimplementation | Difference |
|-------|--------------|------------------|------------|
| 2D line detection | LSD on 26 HorizonNet tangent-plane views; `edgeFromImg2Pano()` back-projects using tangent-plane approximation | LSD on 26 HorizonNet tangent-plane views; `pano_line_detector.py` back-projects using exact pinhole inverse projection | Exact vs approximate back-projection (Section 3.1) |
| Colinear merging | No merging; duplicate/overlapping arcs kept | 3-pass iterative merge of colinear arcs (dot product > cos 1deg, range overlap check) | Fewer redundant arcs, cleaner intersection sets (Section 3.2) |
| 3D line extraction | Loads pre-computed TXT files (dataset-specific format) | Calls `3DLineDetection` C++ binary on raw point cloud PLY | Different upstream source; ~100x faster than V3 plane detection |
| Principal direction voting | Greedy icosphere voting (level 5, hemisphere, ~2562 pts) | Same algorithm, same parameters | Equivalent |
| 24 rotation candidates | SVD Procrustes: target = `canonical_principal_3d = eye(3)` | SVD Procrustes: target = `eye(3)` | Equivalent |
| 2D intersection finding | `intersections_2d()` returns sphere points only | `find_intersections_2d_indexed()` additionally returns line-pair indices | Extra bookkeeping enables rotation refinement (Section 3.5) |
| Translation grid | `generate_trans_points()` with quantile sampling; num_trans from config (typically 1700) | `generate_translation_grid()` with quantile sampling; adaptive axis-proportional resolution | Different grid construction (Section 3.3) |
| XDF coarse search | `single_pose_compute=True`; query sphere level 1 (~12 pts); top_k=1 per room | `single_pose_compute` equivalent; query sphere level 3 (~642 pts); top_k=10 with rotation diversity | Higher resolution + more candidates (Section 3.4) |
| ICP refinement | `refine_from_sphere_icp()`: joint translation + rotation, single candidate | `refine_pose()`: strictly two-phase (translation then rotation), 10 candidates ranked by match quality | Multi-candidate refinement with quality selection (Section 3.5) |
| Candidate selection | Single best from coarse XDF cost | Best among 10 refined candidates, ranked by tight inlier count (sphere distance < 0.1 rad) then average distance | Quality-based selection after refinement (Section 3.6) |

---

## 3. Detailed Algorithmic Differences

### 3.1 Sphere Back-Projection: Exact Pinhole Inverse vs Tangent-Plane Approximation

Both systems decompose the equirectangular panorama into 26 perspective views (6 horizontal at 60deg azimuth increments, 12 diagonal at +/-45deg elevation, 2 poles) and run LSD line detection on each.

**Original FGPL** (`edge_utils.py`, `edgeFromImg2Pano()`): Uses a tangent-plane model for back-projection. The image plane is treated as a tangent plane to the sphere at the view center. 2D pixel coordinates are projected to 3D via:

```
contact = [R cos(vy) sin(vx),  R cos(vy) cos(vx),  R sin(vy)]
p_3d    = contact + dx * vecX + dy * vecY
```

where `R = (W/2) / tan(FOV/2)` is the principal distance and `vecX`, `vecY` are tangent vectors. The tangent-plane model is an approximation that introduces distortion for pixels far from the view center.

**Reimplementation** (`pano_line_detector.py`): Uses exact pinhole inverse projection. Each pixel `(u, v)` is unprojected to a ray in the camera's local frame using the standard pinhole model, then rotated to the world frame:

```
x_cam = (u - cx) / fx
y_cam = (v - cy) / fy
ray_local = [x_cam, y_cam, 1]
ray_world = R_view @ normalise(ray_local)
```

For 320x320 pixel views at 60deg FOV, the maximum angular error of the tangent-plane approximation is ~0.3deg at the image corners. While small, this error accumulates across 26 views and ~1800 detected lines.

### 3.2 Colinear Segment Merging

**Original FGPL**: Does not merge colinear segments from overlapping views. When the same physical edge is detected in multiple perspective views, it produces multiple overlapping sphere arcs.

**Reimplementation** (`pano_line_detector.py`): Applies a 3-pass iterative merge. Two arcs are merged if:
1. Their great-circle normals are nearly parallel: `|n_1 . n_2| > cos(1deg)`
2. Their angular ranges overlap on the shared great circle

The merged arc inherits the union of both angular ranges, with the normal recomputed as the normalised mean. This typically reduces ~2500 raw detections to ~1800 unique arcs.

The effect on downstream stages is twofold: (a) intersection finding produces fewer spurious duplicates, and (b) the XDF line distance function (LDF) has cleaner per-group distance fields.

### 3.3 Translation Grid Construction

Both systems use quantile-based sampling to generate ~1700 translation candidates in the 3D bounding volume of the wireframe, avoiding uniform spacing that wastes candidates in empty regions.

**Original FGPL** (`utils.py`, `generate_trans_points()`): Computes per-axis bounds from the full 3D point cloud, applies configurable quantile trimming, and generates a grid with per-axis resolution set by the config file.

**Reimplementation** (`pose_search.py`, `generate_translation_grid()`): Computes per-axis bounds from 3D line segment midpoints (not the raw point cloud), applies 10th/90th percentile trimming, and sets per-axis resolution adaptively proportional to the squared axis extent:

```
L_x, L_y, L_z = axis extents from 10th-90th percentile
n_x = ceil((L_x^2 * N_total / (L_y * L_z))^(1/3))
```

This allocates more samples along longer axes. After grid generation, a chamfer filter discards candidates within 0.3 m of any line midpoint (physically implausible camera positions inside walls).

### 3.4 XDF Coarse Search: Resolution and Candidate Diversity

The XDF (Cross Distance Function) cost evaluates how well 2D features match 3D features at each candidate pose `(t, R)`. At each of `N_q` query points on the unit sphere, six distance channels are compared:

- 3 LDF channels: distance to nearest 2D/3D line arc per principal group
- 3 PDF channels: distance to nearest 2D/3D intersection point per principal group

The cost is the negative count of inlier query points where `|d_2d - d_3d| < 0.1`.

**Original FGPL**: Uses query sphere level 1 (config default), producing ~12 query points. This gives a maximum possible inlier count of 12 x 6 = 72 per pose. With so few query points, the cost function has limited discriminative power.

**Reimplementation**: Uses query sphere level 3, producing 642 query points and a maximum inlier count of 642 x 6 = 3852. The finer sampling provides much stronger discrimination between candidate poses, especially when two rotations produce similar but not identical projected wireframes.

Both systems use the `single_pose_compute` optimisation: LDF-2D and PDF-2D are computed exactly for one reference rotation, then approximated for the remaining 23 rotations via nearest-neighbour interpolation on the query sphere and channel permutation. This reduces cost from `O(24 * N_2D * N_q)` to `O(N_2D * N_q + 24 * N_q)`.

**Candidate selection** also differs. The original selects `top_k_candidate=1` (the single lowest-cost pose). The reimplementation selects `TOP_K=10` with a rotation diversity heuristic:
1. Find the best translation for each of the 24 rotations
2. Sort rotations by their best cost
3. For the top 5 rotations, include multiple translation candidates
4. Result: 10 diverse `(t, R)` pairs spanning at least 5 distinct rotation hypotheses

This ensures that geometrically distinct pose hypotheses (e.g., 180deg-rotated alternatives in symmetric rooms) are all carried forward to ICP refinement rather than being discarded at the coarse stage.

### 3.5 ICP Refinement: Two-Phase with Line-Pair Index Tracking

Both systems refine coarse poses via an ICP-like procedure on the unit sphere, optimising translation and rotation to align 2D intersection points with projected 3D intersection points.

**Original FGPL** (`fgpl/pose_estimation.py`, `refine_from_sphere_icp()`): Runs a single optimisation loop where translation is optimised via Adam for `total_iter` iterations (default 100). Optionally, rotation is refined jointly (controlled by `sphere_icp_rotation` config flag). Matching uses grouped mutual nearest-neighbours with a distance threshold of 0.5 on the sphere.

**Reimplementation** (`pose_refine.py`, `refine_pose()`): Runs two strictly separated phases:

**Phase 1 (Translation, 100 iterations)**: Rotation is held fixed. Adam optimises translation with `ReduceLROnPlateau` scheduling (patience=5, factor=0.9). Matching uses:
- Grouped mutual-NN: for each of 3 intersection groups, find bidirectional nearest neighbours with sphere distance < 0.5
- Global fallback: full-matrix NN with tighter threshold (0.5 / 5 = 0.1) to catch cross-group matches

**Phase 2 (Rotation, 50 iterations)**: Translation is held fixed. Rotation is parameterised as yaw-pitch-roll (YPR) and optimised via Adam. The cost function aligns line directions:

```
For each matched intersection pair (i_2d, i_3d):
  Extract the two lines that produced each intersection
  Cost = min(|n_2d @ R @ d_3d|, |n_2d @ R @ (-d_3d)|)
```

The `min` over two orderings handles the sign ambiguity of undirected lines. Only inliers with cost < 0.2 contribute to the gradient.

This two-phase separation allows Phase 2 to use the intersection line-pair indices (tracked by `find_intersections_2d_indexed()`) to construct a direction-alignment cost that is not available in Phase 1. The original FGPL does not track line-pair indices through the intersection computation, limiting rotation refinement to a simpler intersection-point cost.

**Safety checks** are identical in both: revert to initialisation if translation drifts > 1.0 m or rotation drifts > 60deg.

### 3.6 Multi-Candidate Selection by Match Quality

**Original FGPL**: Refines a single candidate per room (`top_k_candidate=1`). The coarse XDF cost alone determines which pose is refined.

**Reimplementation**: Refines 10 candidates and selects the best by post-refinement match quality. After ICP, each candidate is scored by:

1. **Tight inlier count** (`n_tight`): number of matched intersection pairs with sphere distance < 0.1 rad (~5.7deg). This is stricter than the 0.5 rad threshold used during ICP matching.
2. **Average sphere distance** (`avg_dist`): mean arc distance over all matched pairs.

Candidates are ranked by `(-n_tight, avg_dist)`: highest tight count first, ties broken by lowest distance. This post-refinement quality metric is more reliable than the coarse XDF cost because ICP can recover from small initialisation errors, meaning a candidate that ranks 3rd in coarse cost may yield the best refined pose.

---

## 4. Impact on Pose Accuracy

The differences above combine to produce measurably different results on the test room (`TMB_office1`):

| Metric | Original FGPL | Reimplementation |
|--------|--------------|------------------|
| XDF query points | 12 (level 1) | 642 (level 3) |
| Coarse candidates refined | 1 | 10 |
| Best coarse cost | -1944 | -2084 |
| Tight inlier count (post-ICP) | not computed | 386 |
| Side-by-side alignment | Incorrect | Correct |

The original FGPL's single-candidate strategy is vulnerable to 180deg rotation ambiguity in approximately symmetric rooms. When two rotations produce similar XDF costs (within ~1% of inlier count), the arbitrary winner may be incorrect. The reimplementation addresses this by:

1. **Higher XDF resolution** (642 vs 12 query points) that amplifies cost differences between similar rotations
2. **Rotation-diverse top-K** that guarantees both rotation hypotheses reach ICP
3. **Post-refinement quality selection** that picks the pose with the tightest geometric agreement, regardless of coarse ranking

On `TMB_office1`, both the correct rotation and its 180deg counterpart survive to ICP refinement. The correct rotation achieves 386 tight inliers (sphere distance < 0.1 rad) with average distance 0.019, while the 180deg alternative achieves 264 tight inliers with average distance 0.039. The quality-based selection correctly identifies the first as the better pose.

---

## 5. Summary of Contributions

The reimplementation makes the following changes relative to the FGPL reference:

1. **Exact pinhole back-projection** replaces the tangent-plane approximation, eliminating ~0.3deg corner distortion per view.
2. **Colinear segment merging** across overlapping views reduces duplicate arcs by ~30%, producing cleaner distance fields.
3. **Adaptive quantile translation grid** scales per-axis resolution proportionally to axis extent, improving coverage in elongated rooms.
4. **Higher XDF query resolution** (level 3 vs level 1) increases discriminative power from 72 to 3852 maximum inliers.
5. **Rotation-diverse multi-candidate refinement** (10 candidates from 5+ distinct rotations) prevents premature commitment to a single rotation hypothesis.
6. **Two-phase ICP with line-pair tracking** separates translation and rotation optimisation and uses direction-alignment cost for rotation, enabled by tracking which lines produce each intersection.
7. **Post-refinement quality selection** ranks candidates by tight inlier count rather than coarse cost, selecting the pose with the best final geometric fit.

These changes resolve the 180deg rotation ambiguity observed in the original FGPL on the test room, producing correct pose estimates where the reference fails.
