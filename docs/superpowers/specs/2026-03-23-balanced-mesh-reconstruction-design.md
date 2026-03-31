# Balanced Mesh Reconstruction Pipeline Design

**Date:** 2026-03-23
**Status:** Draft
**Replaces:** `2026-03-23-high-fidelity-mesh-reconstruction-design.md` (OOM + 40 min runtime)

## Problem

The high-fidelity mesh reconstruction plan (Poisson depth 11, 8192 atlas, no downsampling) caused OOM crashes on a 32 GB machine with ~15 GB free RAM and took 40+ minutes before crashing. The parameters were overspec'd for the actual use case.

## Requirements

- **Geometric fidelity**: 1 cm measurement tolerance on a 5 m wall (0.2% relative accuracy)
- **Visual quality**: Sharp furniture edges (table edges crisp), recognizable painting patterns. Not photorealistic — no need to read poster text.
- **Input**: Colored TLS point clouds (8.8M points, 1 unit = 1 meter, median point spacing ~27 mm, range 7-85 mm)
- **Scenes**: 3-10 buildings/floors, repeatable parameters without per-scene hand-tuning
- **RAM constraint**: 32 GB total, ~15 GB free during reconstruction. Peak pipeline usage must stay under ~5 GB.
- **Runtime target**: 5-10 minutes for the full pipeline
- **Viewer**: Unity app — GLB import, first-person walkthrough, raycast measurement tool

## Design Decision: Screened Poisson Reconstruction

Unchanged from high-fidelity plan. Screened Poisson selected over BPA (unreliable with 7-85 mm density variation) and Gaussian Splatting (not geometrically accurate). Poisson trade-off: smooths sharp edges ~5-10 mm, acceptable given 1 cm tolerance.

## Key Parameter Changes from High-Fidelity Plan

| Parameter | High-fidelity plan | This plan | Why |
|-----------|-------------------|-----------|-----|
| Voxel downsample | None (8.8M pts) | **5 mm** (~4-5M pts) | Removes near-range TLS redundancy. Median spacing is 27 mm — 5 mm only merges tight clusters, no surface thinning |
| Poisson depth | 11 (~3 mm voxel) | **9** (~17 mm voxel) | 4x faster per tile. Flat surface error ~3-5 mm (sub-voxel interpolation). Sharp 90° corners round by ~15-20 mm (expected Poisson behavior, documented trade-off). Wall-to-wall measurement within 1 cm. |
| Atlas resolution | 8192 | **4096** | ~7 mm/texel on ~774 m² surface. Sharp furniture edges, painting patterns visible. 4x less memory and baking time |
| Bake KNN | 8 | **4** | Sharper color transitions than current code's KNN=8. Sufficient density for IDW at 9 mm surface spacing |
| Bake source | N/A | **Full 8.8M cloud** | Preserves color sharpness. Loaded lazily at baking stage (second load). +400 MB RAM, negligible runtime cost |
| Multi-page atlas | 8-12 pages | **Single page** | One 4096 page covers the surface adequately |
| Draco compression | Pipeline stage | **Optional CLI post-processing** | Not part of core pipeline, can be applied after with `gltf-transform` |
| `decimate_mesh()` | Removed | **Removed** | No decimation in this pipeline |

Unchanged from high-fidelity plan: tile size (6 m), overlap (1 m), density trim quantile (0.06), normal KNN (50), normal radius (0.15 m), min tile points (1000).

**Obsolete TODOs in current code:** The existing code contains `# TODO: bump back to 11` (depth) and `# TODO: bump back to 8192` (atlas). These reference the abandoned high-fidelity plan and must be removed during implementation.

## Pipeline Architecture

```
Stage 1: Load full colored point cloud (8.8M pts, ~400 MB)
    |
Stage 2: 5 mm voxel downsample → ~4-5M pts (~200 MB). Free original.
    |
Stage 3: Compute tile grid (6x6 m XY tiles, 1 m overlap)
    |
    +-- Per tile (sequential):
    |     Stage 4: Extract tile points from downsampled cloud
    |     Stage 5: Estimate normals (KNN=50, radius=0.15 m)
    |     Stage 6: Screened Poisson reconstruction (depth 9)
    |     Stage 7: Density-based artifact removal (quantile 0.06)
    |     Stage 8: Trim tile mesh to core ownership region
    |     Stage 9: Save tile PLY to disk. Free tile objects from RAM.
    |
Stage 10: Reload tile PLYs → merge into single mesh. Free downsampled cloud.
    |
Stage 11: Save vertex-colored PLY (for CloudCompare inspection)
    |
Stage 12: Load full 8.8M cloud for texture baking
    |
Stage 13: UV unwrap (xatlas, resolution hint = 4096)
    |
Stage 14: Bake 4096x4096 texture atlas (KNN=4, IDW from full cloud)
    |
Stage 15: Dilate empty texels (8 iterations)
    |
Stage 16: Export GLB + metadata JSON. Free remaining objects.
```

### Memory Management Strategy

The pipeline never holds more than one expensive object at a time. **The original 8.8M cloud is loaded twice** — once at Stage 1 (deleted after Stage 2 downsampling) and once at Stage 12 (for texture baking). This avoids holding ~400 MB of the original cloud through the entire tile loop. This is a change from the current code, which holds the original cloud throughout.

1. **Load phase (Stages 1-2):** Load full cloud (~400 MB) → downsample → `del` full cloud. Peak: ~600 MB.
2. **Per-tile phase (Stages 4-9):** Downsampled cloud (~200 MB) + one tile's Poisson solver (~1-2 GB). Each tile saved to disk and freed before next tile starts. **Peak: ~2-2.5 GB.**
3. **Merge phase (Stage 10):** Reload tile PLYs + concatenate. `del` downsampled cloud (no longer needed). Peak: ~700 MB.
4. **UV unwrap (Stage 13):** Merged mesh (~500 MB) + xatlas internals (~2-3 GB for 4-8M triangles). **Peak: ~3-4 GB.**
5. **Bake phase (Stages 12, 14-15):** Load full 8.8M cloud from disk (second load). UV-mapped mesh (~500 MB) + full cloud (~400 MB) + KD-tree (~500 MB) + atlas (50 MB). **Peak: ~1.5-2 GB.**
6. **Export (Stage 16):** Vertices/faces/UVs/normals + atlas. Peak: ~600 MB.

**Worst-case peak: ~4 GB** (during xatlas UV unwrap). Well within 15 GB free.

Critical `del` points:
- After Stage 2: `del` full cloud (keep downsampled). Current code does NOT do this — must be added.
- After each tile in Stage 9: `del` tile_pcd, tile_mesh, densities
- After Stage 10 merge: `del` downsampled cloud
- After Stage 13 UV unwrap: `del` Open3D merged mesh (keep numpy arrays)

### Stage Details

**Stage 2: Voxel Downsample (5 mm)**

```python
pcd_ds = pcd.voxel_down_sample(voxel_size=0.005)
```

Open3D's voxel downsample averages positions and colors within each voxel. At 5 mm, this only affects clusters of points closer than 5 mm — mostly near-range TLS overlap. The median spacing (27 mm) is untouched. Expected reduction: 8.8M → ~4-5M points.

**Stages 5-6: Normals + Poisson Depth 9**

Per tile with ~8 m extent (6 m core + 1 m overlap each side), Poisson depth 9 produces finest cells of:
- (8 m × 1.1 scale) / 2^9 = 8.8 / 512 ≈ **17.2 mm**

The Poisson implicit function interpolates within cells, so actual surface positioning error is ~8-10 mm on flat architectural surfaces. This meets 1 cm tolerance.

Depth 9 vs 11 runtime: the octree has 4x fewer cells at each level reduction. Depth 9 ≈ 16x fewer cells than depth 11, but Poisson solve scales sub-linearly, so practical speedup is ~4-8x per tile.

**Stage 8: Ownership Trimming**

Each tile keeps only triangles whose centroids fall within the core 6x6 m bounds (no overlap). This prevents duplicate geometry at boundaries. Small gaps (~2-5 mm) at tile borders are acceptable for 1 cm tolerance and imperceptible in the viewer.

**Stage 13: UV Unwrap (xatlas)**

xatlas runs on the full merged mesh (~4-8M triangles at depth 9, vs 15-25M at depth 11). This is 3-4x fewer triangles, making xatlas significantly faster and lighter.

Resolution hint 4096 guides xatlas packing. xatlas may split vertices at chart boundaries (UV seams), so output vertex count exceeds input. The `vmapping` array maps new vertices back to originals.

**Stage 14: Texture Baking**

For each texel in the 4096x4096 atlas:
1. Compute 3D world position via barycentric interpolation from the UV-mapped face
2. Query 4 nearest points from the full 8.8M colored cloud via KD-tree
3. IDW blend their colors (weight = 1/distance)

Batched in chunks of 10K faces with vectorized texel writes. KNN=4 (vs 8 in old plan) produces sharper color edges — important for table edges and painting patterns.

**Stage 15: Texture Dilation**

8 iterations of neighbor-averaging on empty (black) texels. Prevents black seam artifacts at UV chart boundaries.

## Runtime Estimate

| Stage | Estimate | Notes |
|-------|----------|-------|
| Stages 1-2 (load + downsample) | ~10-20 s | I/O bound (227 MB PLY) |
| Stage 3 (tile grid) | <1 s | Arithmetic only |
| Stages 4-9 (8 tiles × depth 9) | ~4-8 min | ~30-60 s per tile. Dominant cost. |
| Stage 10 (merge) | ~10-30 s | Reload + concatenate |
| Stage 11 (save PLY) | ~10-20 s | I/O bound |
| Stage 12 (reload full cloud) | ~5-10 s | I/O bound |
| Stage 13 (xatlas UV unwrap) | ~1-3 min | Scales with triangle count |
| Stage 14 (texture baking) | ~1-4 min | Scales with triangles × atlas resolution |
| Stage 15 (dilate) | ~5-10 s | 8 iterations on 4096² |
| Stage 16 (GLB export) | ~10-30 s | I/O + trimesh conversion |
| **Total** | **~7-16 min** | Realistic range. Best case ~7 min, worst ~16 min. |

The dominant cost is per-tile Poisson (Stages 4-9). Runtime varies with point density per tile — tiles covering dense areas (many overlapping TLS scans) take longer. The 5-10 min target is achievable in the best case; worst case may reach ~15 min on tiles with high density.

## Expected Output

| Metric | Estimate |
|--------|----------|
| Total triangles | ~4-8M (vs 15-25M at depth 11) |
| GLB file size (uncompressed) | ~80-200 MB |
| PLY file size (vertex-colored) | ~100-200 MB |
| Texture resolution | ~8-9 mm/texel (4096² at ~65% packing efficiency over ~774 m²) |
| Flat surface accuracy | ~3-5 mm |
| Sharp corner rounding | ~15-20 mm (expected Poisson smoothing) |
| Wall-to-wall measurement error | <1 cm |
| Tile boundary gaps | ~2-5 mm |
| Pipeline runtime | ~7-16 min (see breakdown above) |
| Peak RAM | ~4 GB |

**Optional post-processing:** Draco compression via `gltf-transform draco` CLI can reduce GLB to ~10-30 MB. Not part of the pipeline — apply manually when needed. Note: for Unity deployment, Draco is recommended — an 80-200 MB uncompressed GLB may cause long load times on lower-end hardware.

## Error Handling

- **Tile with too few points** (<1000 after extraction): skip tile, log warning. Already handled by `MIN_TILE_POINTS`.
- **Tile Poisson produces degenerate mesh** (0 triangles after density trim): skip tile, log warning. Do not crash the pipeline.
- **xatlas fails on degenerate triangles**: `merged.remove_degenerate_triangles()` before UV unwrap. If xatlas still fails, fall back to vertex-colored PLY export (no texture).
- **Input cloud has no colors**: Pipeline still produces geometry (PLY + GLB without texture). Log warning at Stage 1.
- **Tile PLY files**: Temporary. Deleted after merge (Stage 10). Stored in `OUTPUT_DIR/tiles/` during processing.

## Diff from Current Code

The current code (`mesh_reconstruction.py`) needs these specific changes:

| Location | Current | New |
|----------|---------|-----|
| Config: `POISSON_DEPTH` | `10` (TODO: 11) | `9`. Remove TODO. |
| Config: `ATLAS_RESOLUTION` | `4096` (TODO: 8192) | `4096`. Remove TODO. |
| Config: `BAKE_KNN` | `8` | `4` |
| After load | No downsample | Add `pcd.voxel_down_sample(0.005)` + `del` original cloud |
| Before bake | Uses cloud held from load | Reload full cloud from disk (second load) |
| Memory mgmt | `del` only at tile level | Add `del` of original cloud after downsample, `del` downsampled cloud after merge |

## What You Lose vs High-Fidelity Plan

- Fine architectural detail below ~1 cm (individual mortar lines, thin cable runs) — smoothed by depth 9
- Sub-cm measurement precision: 5 m wall could be off by up to ~1 cm vs ~3 mm with depth 11
- Photorealistic texture at close inspection — 7 mm/texel is sharp but not camera-resolution

## What You Gain

- **~7-16 min runtime** vs 40+ min (crashed)
- **~3 GB peak RAM** vs OOM at 15 GB
- Stable, repeatable runs across 3-10 building scenes
- No risk of OOM crashes

## Files Modified

| File | Change |
|------|--------|
| `src/meshing/mesh_utils.py` | Add new functions: `compute_tile_grid`, `extract_tile_points`, `trim_to_ownership_region`, `merge_tile_meshes`, `uv_unwrap_mesh`, `bake_texture_atlas`, `dilate_texture`. Replace `transfer_vertex_colors` with batch KD-tree version. Remove `decimate_mesh`. |
| `src/meshing/export_gltf.py` | Rewrite for UV-textured GLB export with metric metadata injection. Keep `export_vertex_color_ply`. |
| `src/meshing/mesh_reconstruction.py` | Rewrite orchestrator with revised parameters, voxel downsampling, and explicit memory management (`del` at each stage). |
| `tests/meshing/test_mesh_utils.py` | New. Unit tests for chunking, trimming, merging, UV baking. |
| `tests/meshing/test_export_gltf.py` | New. Unit tests for textured GLB export. |

## Tech Stack

Open3D 0.16, numpy, trimesh, xatlas, pygltflib, scipy, Pillow, pytest. All CPU-only.

Conda environment: `scan_env`. All commands use `conda run -n scan_env`.
