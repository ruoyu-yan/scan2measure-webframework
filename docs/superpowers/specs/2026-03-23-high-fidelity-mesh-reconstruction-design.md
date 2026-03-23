# High-Fidelity Mesh Reconstruction Pipeline Design

**Date:** 2026-03-23
**Status:** Draft
**Replaces:** Current web-optimized meshing pipeline (`src/meshing/`)

## Problem

The current meshing pipeline was designed for lightweight web hosting. Every parameter trades quality for web performance: 1.5 cm voxel downsampling, Poisson depth 9, decimation to 2M triangles, vertex colors. The resulting mesh is too coarse for a virtual tour with sub-centimeter measurement capability.

## Requirements

- **Geometric fidelity**: Sub-centimeter measurement accuracy on architectural elements and fixed furniture (desks, cabinets, shelves)
- **Visual quality**: Smooth, textured appearance suitable for a walkthrough virtual tour
- **Input**: Colored TLS point clouds (8.8M points, 1 unit = 1 meter, median point spacing ~27 mm, range 7-85 mm)
- **Scenes**: 3-10 buildings/floors, repeatable parameters without per-scene hand-tuning
- **RAM constraint**: 32 GB total, ~12 GB free during reconstruction
- **GPU**: Available but not leveraged — Open3D Poisson, xatlas, and KD-tree baking are all CPU-only. GPU may be useful for the Unity viewer (rendering) but not the reconstruction pipeline
- **Viewer**: Unity app (separate project) — GLB import, first-person walkthrough, raycast measurement tool

## Design Decision: Screened Poisson Reconstruction

Screened Poisson was selected over Ball Pivoting (BPA) and 3D Gaussian Splatting:

- **vs. BPA**: The point cloud's 7-85 mm density variation makes BPA unreliable without per-scene radius tuning. BPA also leaves holes in sparse areas, breaking both immersion and measurement. Poisson interpolates across sparse regions gracefully.
- **vs. Gaussian Splatting**: Splat positions are optimized for visual quality, not geometric fidelity. Sub-cm measurement accuracy cannot be guaranteed. Tooling is less mature.
- **Poisson trade-off**: Smooths sharp edges (~5-10 mm rounding on corners). Acceptable given the sub-cm tolerance and the benefit of complete, gap-free surfaces for measurement.

## Pipeline Architecture

```
Stage 1: Load colored point cloud (no downsampling)
    |
Stage 2: Spatial chunking (6x6 m XY tiles, 1 m overlap)
    |
    +-- Per tile (sequential, ~3-4 GB RAM each):
    |     Stage 3: Estimate normals (KNN=50, radius=0.15 m)
    |     Stage 4: Screened Poisson reconstruction (depth 11)
    |     Stage 5: Density-based artifact removal (quantile 0.06)
    |     Stage 6: Trim tile to core region
    |
Stage 7: Merge tile meshes (concatenate ownership-trimmed tiles)
    |
Stage 8: UV texture atlas baking from colored point cloud
    |
Stage 9: Export GLB with embedded textures + metric metadata
```

### Stage 1: Load Point Cloud

Load the full colored PLY. No voxel downsampling — all 8.8M points are preserved. Verify metric scale via bounding box extent check (same as current pipeline).

### Stage 2: Spatial Chunking

Divide the point cloud's XY bounding box into a grid of 6 x 6 m tiles with 1.0 m overlap on all sides. For the primary dataset (11.4 x 21.8 m), this produces a 2 x 4 grid = 8 tiles, each containing ~1-1.5M points.

**Why XY-only**: The scene is 4.4 m tall (single story). Vertical splitting would cut walls, making seam stitching much harder.

**Parameters**:
- `TILE_SIZE = 6.0` m
- `OVERLAP = 1.0` m — enough Poisson context at tile edges to prevent boundary artifacts
- Tiles are processed sequentially to keep peak RAM at ~4-5 GB

### Stage 3: Normal Estimation (per tile)

```python
KNN = 50          # was 30 — more neighbors for smoother normals at depth 11
RADIUS = 0.15     # meters, unchanged — covers sparse areas (85 mm spacing)
```

Uses `estimate_normals()` with hybrid KDTree search, then `orient_normals_consistent_tangent_plane(k=50)`.

### Stage 4: Screened Poisson Reconstruction (per tile)

```python
POISSON_DEPTH = 11    # was 9
SCALE = 1.1           # unchanged
LINEAR_FIT = False    # unchanged
```

On a 6 m tile with 1 m overlap on each side, the Poisson reconstruction domain is (6.0 + 2×1.0) × 1.1 = 8.8 m. At depth 11: finest cell = 8.8 / 2^11 = **4.3 mm**. This comfortably meets the sub-cm measurement target.

Expected output per tile: ~2-4M triangles. RAM: ~3-4 GB peak.

### Stage 5: Density-Based Artifact Removal (per tile)

```python
DENSITY_TRIM_QUANTILE = 0.06    # was 0.03 — more aggressive at higher depth
```

Removes lowest 6% density vertices. At depth 11, Poisson extrapolates further into unscanned regions, so a higher trim quantile is needed.

### Stage 6: Trim Tile to Ownership Region

Each tile is clipped to its **ownership region** — the area closer to this tile's center than to any neighboring tile's center (Voronoi-style partitioning along the XY tile grid). In practice this is the midplane between adjacent tile centers, which falls at the core tile boundary (no overlap in kept geometry).

**Implementation**: for each tile, define ownership bounds = core tile bounds (without overlap). Remove all triangles whose centroids fall outside these bounds, then `remove_unreferenced_vertices()`.

This means:
- No two tiles contribute geometry to the same region — no doubled faces or T-junctions
- Each tile's Poisson had 1 m of context beyond the ownership boundary (the overlap), so the kept geometry near the boundary is well-supported by the reconstruction
- Small gaps (~2-5 mm, roughly one Poisson cell width) may exist at tile boundaries where adjacent meshes don't share vertices

**Gap acceptability**: A 2-5 mm gap is invisible in a virtual tour and does not affect sub-cm measurement. If someone measures across a tile boundary, the gap contributes < 5 mm error — within tolerance.

### Stage 7: Merge Tile Meshes

Concatenate all ownership-trimmed tile meshes into a single mesh. No vertex welding needed — the ownership partitioning eliminates overlapping geometry.

After merging:
- `remove_degenerate_triangles()`
- `compute_vertex_normals()` on the full merged mesh for consistent lighting across tile boundaries

**Minimum tile point threshold**: tiles with fewer than 1,000 points are skipped (open space, no meaningful geometry to reconstruct).

Expected output: ~15-25M triangles total.

### Stage 8: UV Texture Atlas from Colored Point Cloud

1. **UV unwrap** — `xatlas-python` automatically parameterizes the merged mesh into UV charts, packed into atlas pages

2. **Bake textures** — for each texel in the atlas:
   - Compute 3D world position on the mesh surface from the UV mapping
   - Query K=8 nearest points in the colored point cloud (KD-tree)
   - Inverse-distance-weighted color average
   - Produces smooth color interpolation between point cloud samples
   - In sparse regions (85 mm spacing), texels will interpolate over fewer unique source points, producing softer/blurrier color — acceptable since these are typically ceiling/upper-wall areas less visually scrutinized

3. **Fill empty texels** — dilate from neighboring texels to fill regions with no nearby points (standard texture padding for seam prevention)

**Texture resolution**: 4096 x 4096 per atlas page. For a corridor with ~500-800 m² surface area, achieving ~2-3 mm/texel requires approximately **8-12 atlas pages** (accounting for ~65% xatlas packing efficiency). Each page is ~1-3 MB as JPEG. Stored as embedded JPEG inside GLB.

**RAM during UV baking**: The merged mesh (15-25M triangles) requires ~400-650 MB in memory. xatlas UV unwrapping uses ~3-5x working space = ~1.5-3 GB. The KD-tree over 8.8M source points adds ~300 MB. Atlas pages in memory add ~500 MB. **Total peak: ~4-6 GB** — fits within 12 GB free.

**New dependency**: `xatlas-python`

### Stage 9: Export

**GLB (primary output)**:
- Mesh geometry + UV coordinates + vertex normals
- Texture atlas pages as embedded JPEG
- Metric metadata in `asset.extras`: `{"unit": "meter", "scale": 1.0, ...}`
- Uses existing `export_gltf.py` Open3D → trimesh → GLB path (trimesh supports textured meshes natively)
- **With Draco mesh compression** (supported by trimesh via `glTF-Transform` or `pygltflib`): geometry compresses ~10-20x, reducing ~500-700 MB raw geometry to ~30-70 MB. Plus ~10-30 MB JPEG textures. **Expected GLB size: ~50-100 MB with Draco**, ~550-750 MB without
- Draco is widely supported (Unity, Three.js, Blender all decompress natively)

**PLY (secondary output)**:
- Full-resolution mesh with vertex colors (for CloudCompare inspection)
- Same as current `_full.ply` output

**JSON sidecar** (`<name>_metadata.json`):
```json
{
  "source_point_cloud": "tmb_office_one_corridor_dense_noRGB_textured.ply",
  "n_input_points": 8805624,
  "n_tiles": 8,
  "tile_size_m": 6.0,
  "poisson_depth": 11,
  "n_vertices": 10234567,
  "n_triangles": 19876543,
  "bbox_min_m": [0.0, 0.0, 0.0],
  "bbox_max_m": [11.39, 21.81, 4.41],
  "atlas_pages": 10,
  "atlas_resolution": 4096,
  "glb_size_mb": 75.2,
  "draco_compressed": true,
  "unit": "meter",
  "reconstruction_time_s": 1200.0
}
```

## File Structure

Three files, same as current — rewritten in place:

| File | Changes |
|------|---------|
| `mesh_reconstruction.py` | New orchestrator: chunking loop, merge, texture bake, no decimation |
| `mesh_utils.py` | Updated: new chunking functions, merge/stitch, UV atlas baking. Remove `decimate_mesh()`, update `transfer_vertex_colors()` as fallback only |
| `export_gltf.py` | Updated: pass UV coords + texture to trimesh, embed JPEG atlas |

## Parameters Summary

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `VOXEL_SIZE` | 0.015 m | Removed | Preserve full point cloud |
| `POISSON_DEPTH` | 9 | 11 | ~4.3 mm finest cell per tile |
| `DENSITY_TRIM_QUANTILE` | 0.03 | 0.06 | More aggressive for depth 11 |
| `NORMAL_KNN` | 30 | 50 | Smoother normals for higher depth |
| `NORMAL_RADIUS` | 0.15 m | 0.15 m | Unchanged |
| `TARGET_TRIANGLES` | 2,000,000 | Removed | No decimation |
| `TILE_SIZE` | N/A | 6.0 m | RAM-safe chunking |
| `OVERLAP` | N/A | 1.0 m | Tile boundary context |
| `MIN_TILE_POINTS` | N/A | 1,000 | Skip near-empty tiles |
| `ATLAS_RESOLUTION` | N/A | 8192 x 8192 | ~3 mm/texel |
| `BAKE_KNN` | N/A | 8 | IDW color interpolation |

## Expected Output

- **Triangles**: ~15-25M
- **Vertices**: ~8-12M
- **GLB size**: ~50-100 MB with Draco compression, ~550-750 MB uncompressed
- **PLY size**: ~400-600 MB
- **Peak RAM**: ~4-5 GB during per-tile Poisson, ~4-6 GB during UV baking (both within 12 GB free)
- **Reconstruction time**: ~15-25 min total (8 tiles, ~2-3 min each)
- **Measurement accuracy**: < 5 mm on flat surfaces, ~5-10 mm on sharp edges

## Dependencies

Existing:
- `open3d`
- `numpy`
- `trimesh`

New:
- `xatlas-python` — UV unwrapping and atlas packing
- `pygltflib` or `gltf-transform` — Draco mesh compression for GLB export

## Viewer (Separate Project)

Unity-based virtual tour app:
- GLB import (native Unity support)
- First-person walkthrough controller
- Raycast-based measurement tool (click two points → metric distance)
- Deploys to desktop/mobile

The viewer is out of scope for this reconstruction pipeline spec. The pipeline's contract is: produce a metric-accurate GLB with textures that any glTF-compatible viewer can consume.
