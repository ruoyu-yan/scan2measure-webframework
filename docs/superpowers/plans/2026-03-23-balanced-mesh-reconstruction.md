# Balanced Mesh Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the existing meshing pipeline to use balanced parameters (depth 9, 4096 atlas, 5 mm downsample, KNN=4) with explicit memory management to prevent OOM and achieve ~7-16 min runtime.

**Architecture:** The pipeline already exists and works. This plan modifies parameters and memory flow in `mesh_reconstruction.py`, updates one default in `mesh_utils.py`, and adds a test for voxel downsampling. No new files are created. No functions are added or removed.

**Tech Stack:** Open3D 0.16, numpy, trimesh, xatlas, pygltflib, scipy, Pillow, pytest

**Spec:** `docs/superpowers/specs/2026-03-23-balanced-mesh-reconstruction-design.md`

**Conda env:** `scan_env` — all commands use `conda run -n scan_env`

---

## File Structure

| File | Change |
|------|--------|
| `src/meshing/mesh_reconstruction.py` | **Modify.** Update config constants, add voxel downsample stage, restructure memory management (two-load pattern for original cloud), add degenerate triangle removal before xatlas. |
| `src/meshing/mesh_utils.py` | **Modify.** Update `bake_texture_atlas` default `knn` from 8 to 4. |
| `tests/meshing/test_mesh_utils.py` | **Modify.** Add test for voxel downsample behavior (colors preserved). |

---

### Task 1: Update Config Constants in mesh_reconstruction.py

**Files:**
- Modify: `src/meshing/mesh_reconstruction.py:63-74`

- [ ] **Step 1: Update constants and remove stale TODOs**

Replace the config block at lines 63-74 with:

```python
# ── Config ───────────────────────────────────────────────────────────────────

POINT_CLOUD_NAME = "tmb_office_one_corridor_dense_noRGB_textured"
POISSON_DEPTH = 9
DENSITY_TRIM_QUANTILE = 0.06
NORMAL_KNN = 50
NORMAL_RADIUS = 0.15
TILE_SIZE = 6.0
OVERLAP = 1.0
MIN_TILE_POINTS = 1000
ATLAS_RESOLUTION = 4096
BAKE_KNN = 4
VOXEL_SIZE = 0.005  # 5 mm downsample
```

- [ ] **Step 2: Verify no import errors**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -c "
import sys; sys.path.insert(0, 'src/meshing')
from mesh_reconstruction import *
print('Imports OK')
"
```

Expected: `Imports OK`

- [ ] **Step 3: Commit**

```bash
cd /home/ruoyu/scan2measure-webframework && git add src/meshing/mesh_reconstruction.py && git commit -m "feat(meshing): update config to balanced parameters (depth 9, KNN 4, 5mm downsample)"
```

---

### Task 2: Add Voxel Downsample Stage and Memory Management

This is the core change. The orchestrator needs:
1. Voxel downsample after load, then `del` original cloud
2. Use downsampled cloud for tiling/Poisson
3. `del` downsampled cloud after merge
4. Reload original cloud for texture baking (second load)

**Files:**
- Modify: `src/meshing/mesh_reconstruction.py:83-278`

- [ ] **Step 1: Write failing test for voxel downsample preserving colors**

Append to `tests/meshing/test_mesh_utils.py`:

```python
def test_voxel_downsample_preserves_colors():
    """Open3D voxel downsample keeps color information."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.001, 0.001, 0.001],  # within 5mm of first point
        [1.0, 1.0, 1.0],
    ])
    colors = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    ds = pcd.voxel_down_sample(voxel_size=0.005)
    assert ds.has_colors()
    assert len(ds.points) <= 3  # first two may merge
    assert len(ds.points) >= 2  # at least two distinct voxels
```

- [ ] **Step 2: Run test to verify it passes**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_mesh_utils.py::test_voxel_downsample_preserves_colors -v 2>&1
```

Expected: PASS (this tests Open3D behavior, not our code — verifying our assumption)

- [ ] **Step 3: Rewrite the `main()` function in mesh_reconstruction.py**

Replace the entire `main()` function (lines 83-277) with:

```python
def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles_dir = OUTPUT_DIR / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    # ── Stage 1: Load point cloud ────────────────────────────────────────────
    print(f"[1/16] Loading point cloud: {INPUT_PATH.name}")
    pcd = o3d.io.read_point_cloud(str(INPUT_PATH))
    n_input_points = len(pcd.points)
    has_colors = pcd.has_colors()
    print(f"       {n_input_points:,} points, colors={has_colors}")

    points_arr = np.asarray(pcd.points)
    bbox_min_pcd = points_arr.min(axis=0)
    bbox_max_pcd = points_arr.max(axis=0)
    extent = bbox_max_pcd - bbox_min_pcd
    print(f"       Extent: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")

    # ── Stage 2: Voxel downsample ────────────────────────────────────────────
    print(f"[2/16] Voxel downsample ({VOXEL_SIZE*1000:.0f} mm) ...")
    t = time.time()
    pcd_ds = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    n_ds_points = len(pcd_ds.points)
    del pcd  # Free original cloud (~400 MB)
    gc.collect()
    print(f"       {n_input_points:,} → {n_ds_points:,} points "
          f"({100*n_ds_points/n_input_points:.0f}%) in {time.time()-t:.1f}s")

    # ── Stage 3: Compute tile grid ───────────────────────────────────────────
    ds_points = np.asarray(pcd_ds.points)
    bbox_min_ds = ds_points.min(axis=0)
    bbox_max_ds = ds_points.max(axis=0)
    print(f"[3/16] Computing tile grid (tile={TILE_SIZE}m, overlap={OVERLAP}m)")
    tiles = compute_tile_grid(bbox_min_ds, bbox_max_ds,
                              tile_size=TILE_SIZE, overlap=OVERLAP)
    n_tiles = len(tiles)
    print(f"       {n_tiles} tiles")

    # ── Stages 4-9: Per-tile reconstruction ──────────────────────────────────
    print(f"[4-9] Running per-tile Poisson reconstruction ...")
    tile_ply_paths = []
    n_skipped = 0

    for tile_idx, (core_min, core_max, ext_min, ext_max) in enumerate(tiles):
        print(f"       Tile {tile_idx+1}/{n_tiles} ...", end=" ", flush=True)

        # Stage 4: Extract tile points from downsampled cloud
        tile_pcd = extract_tile_points(pcd_ds, ext_min, ext_max)
        n_tile_pts = len(tile_pcd.points)

        if n_tile_pts < MIN_TILE_POINTS:
            print(f"skipped ({n_tile_pts} pts < {MIN_TILE_POINTS})")
            n_skipped += 1
            continue

        print(f"{n_tile_pts:,} pts", end=" ", flush=True)

        # Stage 5: Estimate normals
        tile_pcd = estimate_normals(tile_pcd, knn=NORMAL_KNN, radius=NORMAL_RADIUS)

        # Stage 6: Poisson reconstruction
        tile_mesh, densities = poisson_reconstruct(tile_pcd, depth=POISSON_DEPTH)

        # Stage 7: Density trim
        tile_mesh = remove_low_density(tile_mesh, densities,
                                       quantile=DENSITY_TRIM_QUANTILE)
        del densities

        n_tris_after_density = len(tile_mesh.triangles)
        if n_tris_after_density == 0:
            print("skipped (0 triangles after density trim)")
            del tile_pcd, tile_mesh
            n_skipped += 1
            continue

        # Stage 8: Ownership trim
        tile_mesh = trim_to_ownership_region(tile_mesh, core_min, core_max)

        # Transfer vertex colors (for PLY inspection output)
        tile_mesh = transfer_vertex_colors(tile_mesh, tile_pcd)
        del tile_pcd

        n_verts = len(tile_mesh.vertices)
        n_tris = len(tile_mesh.triangles)
        print(f"-> {n_verts:,}v / {n_tris:,}t")

        # Stage 9: Save tile mesh to disk, free memory
        tile_ply = tiles_dir / f"tile_{tile_idx}.ply"
        o3d.io.write_triangle_mesh(str(tile_ply), tile_mesh)
        tile_ply_paths.append(tile_ply)
        del tile_mesh
        gc.collect()

    print(f"       Tiles processed: {len(tile_ply_paths)}, skipped: {n_skipped}")

    # Free downsampled cloud — no longer needed after tiling
    del pcd_ds
    gc.collect()

    # ── Stage 10: Merge tile meshes ──────────────────────────────────────────
    print(f"[10/16] Merging {len(tile_ply_paths)} tile meshes ...")
    t = time.time()
    tile_meshes = []
    for tile_ply in tile_ply_paths:
        tile_meshes.append(o3d.io.read_triangle_mesh(str(tile_ply)))
    merged_mesh = merge_tile_meshes(tile_meshes)
    del tile_meshes
    gc.collect()
    n_merged_verts = len(merged_mesh.vertices)
    n_merged_tris = len(merged_mesh.triangles)
    print(f"       Merged: {n_merged_verts:,} vertices, {n_merged_tris:,} triangles "
          f"in {time.time()-t:.1f}s")

    # Clean up temp tile files
    for tile_ply in tile_ply_paths:
        tile_ply.unlink(missing_ok=True)
    tiles_dir.rmdir()  # remove empty tiles dir

    # ── Stage 11: Save vertex-colored PLY ────────────────────────────────────
    ply_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}_vertex_colored.ply"
    print(f"[11/16] Saving vertex-colored PLY: {ply_path.name}")
    export_vertex_color_ply(merged_mesh, ply_path)

    # ── Stage 12: Reload original cloud for texture baking ───────────────────
    print(f"[12/16] Reloading original point cloud for texture baking ...")
    pcd_full = o3d.io.read_point_cloud(str(INPUT_PATH))
    source_points = np.asarray(pcd_full.points)
    source_colors = np.asarray(pcd_full.colors) if pcd_full.has_colors() else None

    if source_colors is None:
        print("       WARNING: No colors in point cloud. Skipping texture baking.")
        print("       Vertex-colored PLY was already saved. No GLB produced.")
        del pcd_full
        return

    # ── Stage 13: UV unwrap ──────────────────────────────────────────────────
    print(f"[13/16] UV unwrapping with xatlas (resolution={ATLAS_RESOLUTION}) ...")
    t = time.time()
    vertices = np.asarray(merged_mesh.vertices)
    normals_arr = np.asarray(merged_mesh.vertex_normals)
    faces = np.asarray(merged_mesh.triangles)

    # Note: merge_tile_meshes() already called remove_degenerate_triangles()

    try:
        vmapping, new_faces, uv_coords = uv_unwrap_mesh(
            vertices, faces, atlas_resolution=ATLAS_RESOLUTION
        )
    except Exception as e:
        print(f"       ERROR: xatlas failed: {e}")
        print("       Vertex-colored PLY was already saved. No GLB produced.")
        del merged_mesh, pcd_full
        return

    new_vertices = vertices[vmapping]
    new_normals = normals_arr[vmapping]
    print(f"       {len(vmapping):,} vertices, {len(new_faces):,} faces "
          f"in {time.time()-t:.1f}s")

    # Free Open3D mesh — we have numpy arrays now
    del merged_mesh
    gc.collect()

    # ── Stage 14: Bake texture atlas ─────────────────────────────────────────
    print(f"[14/16] Baking texture atlas ({ATLAS_RESOLUTION}x{ATLAS_RESOLUTION}, "
          f"knn={BAKE_KNN}) ...")
    t = time.time()
    atlas = bake_texture_atlas(
        new_vertices, new_faces, uv_coords,
        source_points, source_colors,
        atlas_resolution=ATLAS_RESOLUTION,
        knn=BAKE_KNN,
    )
    del pcd_full, source_points, source_colors
    gc.collect()
    print(f"       Done in {time.time()-t:.1f}s")

    # ── Stage 15: Dilate empty texels ────────────────────────────────────────
    print(f"[15/16] Dilating empty texels ...")
    atlas = dilate_texture(atlas, iterations=8)

    # ── Stage 16: Export GLB ─────────────────────────────────────────────────
    glb_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}.glb"
    print(f"[16/16] Exporting GLB: {glb_path.name}")
    t = time.time()
    export_textured_glb(
        new_vertices, new_faces, uv_coords, new_normals,
        [atlas], glb_path, mesh_name="tls_mesh"
    )
    glb_size_mb = glb_path.stat().st_size / (1024 * 1024)
    print(f"       GLB size: {glb_size_mb:.1f} MB in {time.time()-t:.1f}s")

    # ── Metadata + summary ───────────────────────────────────────────────────
    total_time = time.time() - t_start
    bbox_min_m = new_vertices.min(axis=0).tolist()
    bbox_max_m = new_vertices.max(axis=0).tolist()

    metadata = {
        "source_point_cloud": str(INPUT_PATH),
        "n_input_points": n_input_points,
        "n_downsampled_points": int(n_ds_points),
        "voxel_size_mm": VOXEL_SIZE * 1000,
        "n_tiles": n_tiles,
        "n_tiles_skipped": n_skipped,
        "tile_size_m": TILE_SIZE,
        "poisson_depth": POISSON_DEPTH,
        "n_vertices": int(len(new_vertices)),
        "n_triangles": int(len(new_faces)),
        "bbox_min_m": bbox_min_m,
        "bbox_max_m": bbox_max_m,
        "atlas_resolution": ATLAS_RESOLUTION,
        "bake_knn": BAKE_KNN,
        "glb_size_mb": round(glb_size_mb, 2),
        "draco_compressed": False,
        "unit": "meter",
        "reconstruction_time_s": round(total_time, 1),
    }

    json_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}_metadata.json"
    with open(str(json_path), "w") as f:
        json.dump(metadata, f, indent=2)

    bbox_extent = [bbox_max_m[i] - bbox_min_m[i] for i in range(3)]
    print(f"\n{'='*60}")
    print(f"Mesh reconstruction complete in {total_time:.1f}s")
    print(f"  Input:       {n_input_points:,} points → {n_ds_points:,} after downsample")
    print(f"  Tiles:       {n_tiles} total, {n_skipped} skipped")
    print(f"  Output:      {len(new_vertices):,} vertices, {len(new_faces):,} triangles")
    print(f"  Extent:      {bbox_extent[0]:.2f} x {bbox_extent[1]:.2f} x {bbox_extent[2]:.2f} m")
    print(f"  GLB file:    {glb_path}")
    print(f"  GLB size:    {glb_size_mb:.1f} MB")
    print(f"  PLY file:    {ply_path}")
    print(f"  Atlas:       {ATLAS_RESOLUTION}x{ATLAS_RESOLUTION} px")
    print(f"  Metadata:    {json_path}")
    print(f"  Unit:        1 unit = 1 meter (no rescaling applied)")
    print(f"{'='*60}")
```

Also update the imports at the top of the file:

1. Add `import gc` to module-level imports (lines 37-41). Replace with:

```python
import gc
import json
import sys
import time
from pathlib import Path
```

2. Remove `compute_mesh_stats` from the `mesh_utils` import block (lines 47-60), since the new `main()` does not use it. The import block becomes:

```python
from mesh_utils import (
    estimate_normals,
    poisson_reconstruct,
    remove_low_density,
    transfer_vertex_colors,
    compute_tile_grid,
    extract_tile_points,
    trim_to_ownership_region,
    merge_tile_meshes,
    uv_unwrap_mesh,
    bake_texture_atlas,
    dilate_texture,
)
```

- [ ] **Step 4: Verify no import errors**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -c "
import sys; sys.path.insert(0, 'src/meshing')
from mesh_reconstruction import *
print('Imports OK')
"
```

Expected: `Imports OK`

- [ ] **Step 5: Run existing tests to verify nothing breaks**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/ -v 2>&1
```

Expected: All existing tests PASS (they test mesh_utils and export_gltf, not the orchestrator)

- [ ] **Step 6: Commit**

```bash
cd /home/ruoyu/scan2measure-webframework && git add src/meshing/mesh_reconstruction.py tests/meshing/test_mesh_utils.py && git commit -m "feat(meshing): add voxel downsample + two-load memory management for balanced pipeline"
```

---

### Task 3: Update bake_texture_atlas Default KNN

**Files:**
- Modify: `src/meshing/mesh_utils.py:216-217`

- [ ] **Step 1: Update the default knn parameter**

In `src/meshing/mesh_utils.py`, line 217, change the function signature:

From:
```python
def bake_texture_atlas(vertices, faces, uv_coords, source_points, source_colors,
                       atlas_resolution=4096, knn=8):
```

To:
```python
def bake_texture_atlas(vertices, faces, uv_coords, source_points, source_colors,
                       atlas_resolution=4096, knn=4):
```

- [ ] **Step 2: Run existing bake test to verify it still passes**

The existing test explicitly passes `knn=4`, so changing the default doesn't affect it:

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_mesh_utils.py::test_bake_texture_atlas -v 2>&1
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
cd /home/ruoyu/scan2measure-webframework && git add src/meshing/mesh_utils.py && git commit -m "feat(meshing): change bake_texture_atlas default knn from 8 to 4 for sharper textures"
```

---

### Task 4: Update Docstring in mesh_reconstruction.py

**Files:**
- Modify: `src/meshing/mesh_reconstruction.py:1-35`

- [ ] **Step 1: Replace the module docstring**

Replace lines 1-35 with:

```python
"""Balanced mesh reconstruction from colored TLS point clouds.

Converts a colored PLY point cloud into a UV-textured GLB mesh with
metric scale preservation. Targets ~7-16 min runtime and ~4 GB peak RAM.

Pipeline:
1.  Load colored point cloud
2.  5 mm voxel downsample, free original cloud
3.  Compute tile grid (6x6 m XY tiles, 1 m overlap)
4-9. Per-tile: extract → normals → Poisson depth 9 → density trim
     → ownership trim → save to disk, free tile from RAM
10. Reload tiles → merge
11. Save vertex-colored PLY (CloudCompare inspection)
12. Reload original cloud for texture baking
13. UV unwrap with xatlas (4096 resolution)
14. Bake texture atlas (KNN=4 IDW from full cloud)
15. Dilate empty texels
16. Export GLB with metric metadata + JSON sidecar

The TLS point cloud has 1 unit = 1 meter. No rescaling is applied.
Design spec: docs/superpowers/specs/2026-03-23-balanced-mesh-reconstruction-design.md

Usage:
    conda run -n scan_env python src/meshing/mesh_reconstruction.py
"""
```

- [ ] **Step 2: Commit**

```bash
cd /home/ruoyu/scan2measure-webframework && git add src/meshing/mesh_reconstruction.py && git commit -m "docs(meshing): update docstring to reflect balanced pipeline parameters"
```

---

### Task 5: Integration Test on Actual Data

**Files:**
- Run: `src/meshing/mesh_reconstruction.py`

- [ ] **Step 1: Run the full pipeline**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 src/meshing/mesh_reconstruction.py 2>&1
```

Monitor for:
- Stage 2 downsample: 8.8M → ~4-5M points
- Each tile completes in ~30-90s
- No OOM — peak RAM should stay under ~4 GB
- Merge, UV unwrap, and baking complete without errors
- Total runtime: ~7-16 min

- [ ] **Step 2: Verify output files exist**

```bash
ls -lh /home/ruoyu/scan2measure-webframework/data/mesh/tmb_office_one_corridor_dense/
```

Expected:
- `tmb_office_one_corridor_dense_noRGB_textured.glb` (~80-200 MB)
- `tmb_office_one_corridor_dense_noRGB_textured_vertex_colored.ply`
- `tmb_office_one_corridor_dense_noRGB_textured_metadata.json`
- No leftover `tiles/` directory

- [ ] **Step 3: Validate metadata JSON**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -c "
import json
meta = json.load(open('data/mesh/tmb_office_one_corridor_dense/tmb_office_one_corridor_dense_noRGB_textured_metadata.json'))
print(json.dumps(meta, indent=2))
assert meta['unit'] == 'meter'
assert meta['poisson_depth'] == 9
assert meta['bake_knn'] == 4
assert meta['voxel_size_mm'] == 5.0
assert meta['n_triangles'] > 100_000
print('Metadata validation OK')
"
```

- [ ] **Step 4: Verify mesh scale preservation**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -c "
import json
meta = json.load(open('data/mesh/tmb_office_one_corridor_dense/tmb_office_one_corridor_dense_noRGB_textured_metadata.json'))
bbox_min = meta['bbox_min_m']
bbox_max = meta['bbox_max_m']
extent = [bbox_max[i] - bbox_min[i] for i in range(3)]
print(f'Mesh extent: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m')
# Should be close to the original ~11.39 x 21.81 x 4.41 m
assert 10.0 < extent[0] < 13.0, f'X extent {extent[0]} out of range'
assert 20.0 < extent[1] < 23.0, f'Y extent {extent[1]} out of range'
assert 3.5 < extent[2] < 5.5, f'Z extent {extent[2]} out of range'
print('Scale validation OK')
"
```

- [ ] **Step 5: Commit**

```bash
cd /home/ruoyu/scan2measure-webframework && git add -A && git commit -m "feat(meshing): verified balanced mesh reconstruction pipeline on actual data"
```
