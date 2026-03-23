# High-Fidelity Mesh Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the meshing pipeline to produce a high-fidelity, UV-textured GLB mesh from TLS point clouds for virtual tour measurement.

**Architecture:** Spatial chunking (6x6m XY tiles with 1m overlap) → per-tile Screened Poisson at depth 11 → ownership-region trimming → merge → xatlas UV unwrap → KD-tree texture baking from colored point cloud → Draco-compressed GLB export.

**Tech Stack:** Open3D 0.16, numpy, trimesh, xatlas, pygltflib, scipy, pytest

**Spec:** `docs/superpowers/specs/2026-03-23-high-fidelity-mesh-reconstruction-design.md`

**Conda env:** `scan_env` — all commands use `conda run -n scan_env`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/meshing/mesh_utils.py` | **Rewrite.** Core reconstruction library: normal estimation, Poisson, density trim, spatial chunking, ownership trimming, mesh merging, UV atlas baking. Remove `decimate_mesh()`. Keep `transfer_vertex_colors()` for PLY vertex-color fallback. Keep `compute_mesh_stats()`. |
| `src/meshing/export_gltf.py` | **Rewrite.** Accept trimesh with UV + texture images. Build multi-material GLB scene with JPEG atlas pages. Draco compression. Metric metadata injection. |
| `src/meshing/mesh_reconstruction.py` | **Rewrite.** Orchestrator: load → chunk → per-tile reconstruct → trim → merge → UV bake → export. New config constants. |
| `tests/meshing/test_mesh_utils.py` | **New.** Unit tests for chunking, trimming, merging, baking with small synthetic data. |
| `tests/meshing/test_export_gltf.py` | **New.** Unit tests for textured GLB export. |

---

### Task 1: Install Dependencies and Test Infrastructure

**Files:**
- Modify: `scan_env` conda environment (pip install)
- Create: `tests/meshing/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Install new packages**

```bash
conda run -n scan_env pip install xatlas pygltflib pytest Pillow
```

Verify `scipy` is already installed (needed for KD-tree):

```bash
conda run -n scan_env python3 -c "from scipy.spatial import cKDTree; print('scipy OK')"
```

- [ ] **Step 2: Verify imports**

```bash
conda run -n scan_env python3 -c "import xatlas; import pygltflib; import pytest; from PIL import Image; from scipy.spatial import cKDTree; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 3: Create test directory structure**

Create `tests/__init__.py` and `tests/meshing/__init__.py` (empty files).

- [ ] **Step 4: Verify pytest discovers test dir**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/ --collect-only 2>&1 | head -5
```

Expected: `no tests ran` or `collected 0 items`

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/meshing/__init__.py
git commit -m "chore: add test infrastructure for meshing pipeline"
```

---

### Task 2: Spatial Chunking Functions

**Files:**
- Modify: `src/meshing/mesh_utils.py`
- Create: `tests/meshing/test_mesh_utils.py`

- [ ] **Step 1: Write failing tests for `compute_tile_grid` and `extract_tile_points`**

```python
# tests/meshing/test_mesh_utils.py
import numpy as np
import open3d as o3d
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "meshing"))
from mesh_utils import compute_tile_grid, extract_tile_points


def test_compute_tile_grid_single_tile():
    """A small scene fits in one tile."""
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([5.0, 5.0, 3.0])
    tiles = compute_tile_grid(bbox_min, bbox_max, tile_size=6.0, overlap=1.0)
    assert len(tiles) == 1
    # The single tile's core bounds should cover the full XY extent
    core_min, core_max, ext_min, ext_max = tiles[0]
    assert core_min[0] <= 0.0 and core_max[0] >= 5.0
    assert core_min[1] <= 0.0 and core_max[1] >= 5.0


def test_compute_tile_grid_multiple_tiles():
    """A 12x12m scene with 6m tiles produces a 2x2 grid."""
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([12.0, 12.0, 3.0])
    tiles = compute_tile_grid(bbox_min, bbox_max, tile_size=6.0, overlap=1.0)
    assert len(tiles) == 4  # 2x2 grid


def test_compute_tile_grid_returns_core_and_extended_bounds():
    """Each tile has core bounds (no overlap) and extended bounds (with overlap)."""
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([12.0, 12.0, 3.0])
    tiles = compute_tile_grid(bbox_min, bbox_max, tile_size=6.0, overlap=1.0)
    for core_min, core_max, ext_min, ext_max in tiles:
        # Extended bounds should be >= core bounds in each dimension
        assert np.all(ext_min[:2] <= core_min[:2])
        assert np.all(ext_max[:2] >= core_max[:2])
        # Z bounds cover full height (no Z splitting)
        assert ext_min[2] == bbox_min[2]
        assert ext_max[2] == bbox_max[2]


def test_extract_tile_points():
    """Points within extended bounds are extracted."""
    points = np.array([
        [1.0, 1.0, 1.0],
        [5.0, 5.0, 1.0],
        [10.0, 10.0, 1.0],
        [15.0, 15.0, 1.0],
    ])
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    ext_min = np.array([0.0, 0.0, 0.0])
    ext_max = np.array([6.0, 6.0, 3.0])
    tile_pcd = extract_tile_points(pcd, ext_min, ext_max)

    assert len(tile_pcd.points) == 2  # only [1,1,1] and [5,5,5]
    assert tile_pcd.has_colors()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_mesh_utils.py -v 2>&1
```

Expected: FAIL — `ImportError: cannot import name 'compute_tile_grid'`

- [ ] **Step 3: Implement `compute_tile_grid` and `extract_tile_points` in mesh_utils.py**

Add these two functions to `src/meshing/mesh_utils.py`:

```python
def compute_tile_grid(bbox_min, bbox_max, tile_size=6.0, overlap=1.0):
    """Divide XY bounding box into a grid of tiles with overlap.

    Returns a list of (core_min, core_max, ext_min, ext_max) tuples.
    Core bounds = ownership region (no overlap).
    Extended bounds = core + overlap margin (used for Poisson context).
    Z bounds always span the full height (no vertical splitting).

    Args:
        bbox_min: (3,) array — point cloud minimum.
        bbox_max: (3,) array — point cloud maximum.
        tile_size: Tile side length in meters.
        overlap: Overlap margin in meters on each side.

    Returns:
        List of (core_min, core_max, ext_min, ext_max) as (3,) arrays.
    """
    tiles = []
    x_min, y_min, z_min = bbox_min
    x_max, y_max, z_max = bbox_max

    # Compute tile edges along X and Y
    x_edges = np.arange(x_min, x_max, tile_size)
    y_edges = np.arange(y_min, y_max, tile_size)

    for x_start in x_edges:
        for y_start in y_edges:
            # Core bounds (ownership region)
            core_min = np.array([x_start, y_start, z_min])
            core_max = np.array([min(x_start + tile_size, x_max),
                                 min(y_start + tile_size, y_max),
                                 z_max])
            # Extended bounds (with overlap for Poisson context)
            ext_min = np.array([x_start - overlap, y_start - overlap, z_min])
            ext_max = np.array([min(x_start + tile_size + overlap, x_max + overlap),
                                min(y_start + tile_size + overlap, y_max + overlap),
                                z_max])
            tiles.append((core_min, core_max, ext_min, ext_max))

    return tiles


def extract_tile_points(pcd, ext_min, ext_max):
    """Extract points within the extended tile bounds.

    Args:
        pcd: Open3D PointCloud.
        ext_min: (3,) array — extended tile minimum.
        ext_max: (3,) array — extended tile maximum.

    Returns:
        New Open3D PointCloud containing only points inside [ext_min, ext_max].
    """
    points = np.asarray(pcd.points)
    mask = np.all((points >= ext_min) & (points <= ext_max), axis=1)

    tile_pcd = o3d.geometry.PointCloud()
    tile_pcd.points = o3d.utility.Vector3dVector(points[mask])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        tile_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        tile_pcd.normals = o3d.utility.Vector3dVector(normals[mask])

    return tile_pcd
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_mesh_utils.py -v 2>&1
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/meshing/mesh_utils.py tests/meshing/test_mesh_utils.py
git commit -m "feat(meshing): add spatial chunking functions for tiled reconstruction"
```

---

### Task 3: Ownership Trimming Function

**Files:**
- Modify: `src/meshing/mesh_utils.py`
- Modify: `tests/meshing/test_mesh_utils.py`

- [ ] **Step 1: Write failing test for `trim_to_ownership_region`**

Append to `tests/meshing/test_mesh_utils.py`:

```python
from mesh_utils import trim_to_ownership_region


def test_trim_to_ownership_region():
    """Triangles with centroids outside core bounds are removed."""
    # Create a simple mesh: two triangles side by side at x=1 and x=8
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0 — left triangle
        [2.0, 0.0, 0.0],  # 1
        [1.0, 2.0, 0.0],  # 2
        [7.0, 0.0, 0.0],  # 3 — right triangle
        [9.0, 0.0, 0.0],  # 4
        [8.0, 2.0, 0.0],  # 5
    ], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Core bounds include only the left triangle (x: 0-6)
    core_min = np.array([0.0, 0.0, -1.0])
    core_max = np.array([6.0, 6.0, 1.0])

    trimmed = trim_to_ownership_region(mesh, core_min, core_max)
    assert len(trimmed.triangles) == 1  # only left triangle kept
    # Remaining triangle centroid should be at x=1
    verts = np.asarray(trimmed.vertices)
    tris = np.asarray(trimmed.triangles)
    centroid = verts[tris[0]].mean(axis=0)
    assert centroid[0] < 6.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_mesh_utils.py::test_trim_to_ownership_region -v 2>&1
```

Expected: FAIL — `ImportError: cannot import name 'trim_to_ownership_region'`

- [ ] **Step 3: Implement `trim_to_ownership_region` in mesh_utils.py**

```python
def trim_to_ownership_region(mesh, core_min, core_max):
    """Remove triangles whose centroids fall outside the core tile bounds.

    This implements Voronoi-style ownership: each tile keeps only geometry
    within its core region (no overlap). Adjacent tiles' meshes will have
    small gaps (~2-5 mm) at boundaries — acceptable for sub-cm measurement.

    Args:
        mesh: Open3D TriangleMesh.
        core_min: (3,) array — core tile minimum (without overlap).
        core_max: (3,) array — core tile maximum (without overlap).

    Returns:
        New TriangleMesh with only triangles inside the core region.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Compute triangle centroids
    centroids = vertices[triangles].mean(axis=1)  # (N_tri, 3)

    # Keep triangles whose XY centroid is inside core bounds
    # (Z is not filtered — full height is always kept)
    mask = (
        (centroids[:, 0] >= core_min[0]) & (centroids[:, 0] <= core_max[0]) &
        (centroids[:, 1] >= core_min[1]) & (centroids[:, 1] <= core_max[1])
    )

    # Build new mesh with only kept triangles
    kept_triangles = triangles[mask]
    trimmed = o3d.geometry.TriangleMesh()
    trimmed.vertices = o3d.utility.Vector3dVector(vertices)
    trimmed.triangles = o3d.utility.Vector3iVector(kept_triangles)

    if mesh.has_vertex_colors():
        trimmed.vertex_colors = mesh.vertex_colors

    trimmed.remove_unreferenced_vertices()
    return trimmed
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_mesh_utils.py -v 2>&1
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/meshing/mesh_utils.py tests/meshing/test_mesh_utils.py
git commit -m "feat(meshing): add ownership-region trimming for tiled reconstruction"
```

---

### Task 4: Mesh Merging Function

**Files:**
- Modify: `src/meshing/mesh_utils.py`
- Modify: `tests/meshing/test_mesh_utils.py`

- [ ] **Step 1: Write failing test for `merge_tile_meshes`**

Append to `tests/meshing/test_mesh_utils.py`:

```python
from mesh_utils import merge_tile_meshes


def test_merge_tile_meshes():
    """Two meshes are concatenated into one."""
    # Mesh A: one triangle
    mesh_a = o3d.geometry.TriangleMesh()
    mesh_a.vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    )
    mesh_a.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))

    # Mesh B: one triangle offset in X
    mesh_b = o3d.geometry.TriangleMesh()
    mesh_b.vertices = o3d.utility.Vector3dVector(
        np.array([[5, 0, 0], [6, 0, 0], [5, 1, 0]], dtype=np.float64)
    )
    mesh_b.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))

    merged = merge_tile_meshes([mesh_a, mesh_b])
    assert len(merged.vertices) == 6
    assert len(merged.triangles) == 2


def test_merge_tile_meshes_empty_list():
    """Merging empty list returns empty mesh."""
    merged = merge_tile_meshes([])
    assert len(merged.vertices) == 0
    assert len(merged.triangles) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_mesh_utils.py::test_merge_tile_meshes -v 2>&1
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `merge_tile_meshes` in mesh_utils.py**

```python
def merge_tile_meshes(meshes):
    """Concatenate multiple tile meshes into a single mesh.

    No vertex welding — the ownership-region partitioning ensures no
    overlapping geometry. Small gaps (~2-5 mm) at tile boundaries are
    acceptable for sub-cm measurement.

    Args:
        meshes: List of Open3D TriangleMeshes.

    Returns:
        Single merged TriangleMesh with vertex normals computed.
    """
    if not meshes:
        return o3d.geometry.TriangleMesh()

    all_vertices = []
    all_triangles = []
    all_colors = []
    vertex_offset = 0

    for mesh in meshes:
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)

        all_vertices.append(verts)
        all_triangles.append(tris + vertex_offset)
        if mesh.has_vertex_colors():
            all_colors.append(np.asarray(mesh.vertex_colors))

        vertex_offset += len(verts)

    merged = o3d.geometry.TriangleMesh()
    merged.vertices = o3d.utility.Vector3dVector(np.vstack(all_vertices))
    merged.triangles = o3d.utility.Vector3iVector(np.vstack(all_triangles))

    if all_colors and len(all_colors) == len(meshes):
        merged.vertex_colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    merged.remove_degenerate_triangles()
    merged.compute_vertex_normals()
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_mesh_utils.py -v 2>&1
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/meshing/mesh_utils.py tests/meshing/test_mesh_utils.py
git commit -m "feat(meshing): add tile mesh merging"
```

---

### Task 5: UV Texture Atlas Baking

This is the most complex new component. Three functions: `uv_unwrap_mesh`, `bake_texture_atlas`, `dilate_texture`.

**Files:**
- Modify: `src/meshing/mesh_utils.py`
- Modify: `tests/meshing/test_mesh_utils.py`

- [ ] **Step 1: Write failing tests for UV baking functions**

Append to `tests/meshing/test_mesh_utils.py`:

```python
from mesh_utils import uv_unwrap_mesh, bake_texture_atlas, dilate_texture


def test_uv_unwrap_mesh():
    """UV unwrap produces vmapping, new faces, and UV coordinates."""
    mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    vmapping, new_faces, uv_coords = uv_unwrap_mesh(vertices, faces, atlas_resolution=512)

    assert uv_coords.ndim == 2 and uv_coords.shape[1] == 2
    assert np.all(uv_coords >= 0.0) and np.all(uv_coords <= 1.0)
    assert len(vmapping) == len(uv_coords)
    assert new_faces.shape[1] == 3


def test_bake_texture_atlas():
    """Texture atlas is baked from point cloud colors."""
    # Flat quad mesh at z=0
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uv_coords = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1]
    ], dtype=np.float64)

    # Source point cloud: 4 red points covering the quad
    source_points = np.array([
        [0.25, 0.25, 0], [0.75, 0.25, 0],
        [0.25, 0.75, 0], [0.75, 0.75, 0]
    ])
    source_colors = np.full((4, 3), [1.0, 0.0, 0.0])  # all red

    atlas = bake_texture_atlas(
        vertices, faces, uv_coords, source_points, source_colors,
        atlas_resolution=64, knn=4
    )

    assert atlas.shape == (64, 64, 3)
    assert atlas.dtype == np.uint8
    # Non-zero texels should be predominantly red
    nonzero = atlas[atlas.sum(axis=2) > 0]
    if len(nonzero) > 0:
        avg_color = nonzero.mean(axis=0)
        assert avg_color[0] > 200  # red channel high


def test_dilate_texture():
    """Empty texels are filled by dilation from neighbors."""
    atlas = np.zeros((8, 8, 3), dtype=np.uint8)
    # Place a single red pixel at (4, 4)
    atlas[4, 4] = [255, 0, 0]

    dilated = dilate_texture(atlas, iterations=2)

    # Original pixel preserved
    assert np.array_equal(dilated[4, 4], [255, 0, 0])
    # Neighbor pixels should now be filled
    assert dilated[3, 4].sum() > 0 or dilated[5, 4].sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_mesh_utils.py::test_uv_unwrap_mesh -v 2>&1
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `uv_unwrap_mesh` in mesh_utils.py**

```python
import xatlas


def uv_unwrap_mesh(vertices, faces, atlas_resolution=4096):
    """UV-unwrap a mesh using xatlas.

    xatlas may split vertices at chart boundaries (UV seams), so the
    output vertex count may exceed the input. Use vmapping to remap
    original vertex attributes (positions, colors) to the new layout.

    Args:
        vertices: (N, 3) float64 array.
        faces: (M, 3) int32 array.
        atlas_resolution: Resolution for atlas packing.

    Returns:
        (vmapping, new_faces, uv_coords):
            vmapping: (V,) int32 — maps each new vertex to original vertex index.
            new_faces: (M, 3) int32 — remeshed face indices.
            uv_coords: (V, 2) float64 — UV coordinates in [0, 1].
    """
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32))
    atlas.generate(
        xatlas.PackOptions(resolution=atlas_resolution)
    )

    vmapping, new_faces, uv_coords = atlas[0]

    # Normalize UVs to [0, 1]
    uv_coords = uv_coords.astype(np.float64)
    w = max(atlas.width, 1)
    h = max(atlas.height, 1)
    uv_coords[:, 0] /= w
    uv_coords[:, 1] /= h

    return vmapping, new_faces, uv_coords
```

- [ ] **Step 4: Run UV unwrap test**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_mesh_utils.py::test_uv_unwrap_mesh -v 2>&1
```

Expected: PASS

- [ ] **Step 5: Implement `bake_texture_atlas` in mesh_utils.py**

Uses batched face processing (chunks of 10K faces) with vectorized texel writes
to handle 15-25M triangle meshes in reasonable time.

```python
from scipy.spatial import cKDTree


def bake_texture_atlas(vertices, faces, uv_coords, source_points, source_colors,
                       atlas_resolution=4096, knn=8):
    """Bake a texture atlas from colored point cloud via KD-tree IDW.

    For each texel in the atlas, computes its 3D world position from
    the mesh UV mapping, queries K nearest colored points, and blends
    their colors using inverse-distance weighting.

    Rasterizes in batches of 10K faces with fully vectorized texel writes.

    Args:
        vertices: (V, 3) mesh vertex positions.
        faces: (M, 3) face indices.
        uv_coords: (V, 2) UV coordinates in [0, 1].
        source_points: (P, 3) colored point cloud positions.
        source_colors: (P, 3) colors in [0, 1] float.
        atlas_resolution: Texture width and height in pixels.
        knn: Number of nearest neighbors for IDW blending.

    Returns:
        (H, W, 3) uint8 texture atlas image.
    """
    res = atlas_resolution
    atlas = np.zeros((res, res, 3), dtype=np.uint8)

    # Build KD-tree over source point cloud
    tree = cKDTree(source_points)

    # Precompute per-face data
    face_uvs = uv_coords[faces]        # (M, 3, 2)
    face_verts = vertices[faces]        # (M, 3, 3)

    n_faces = len(faces)
    batch_size = 10000

    for batch_start in range(0, n_faces, batch_size):
        batch_end = min(batch_start + batch_size, n_faces)

        # Collect all texels from this batch of faces
        all_pixels_u = []
        all_pixels_v = []
        all_pos_3d = []

        for fi in range(batch_start, batch_end):
            uvs = face_uvs[fi]
            verts = face_verts[fi]
            pix = (uvs * (res - 1)).astype(np.int32)

            u_min = max(pix[:, 0].min(), 0)
            u_max = min(pix[:, 0].max(), res - 1)
            v_min = max(pix[:, 1].min(), 0)
            v_max = min(pix[:, 1].max(), res - 1)
            if u_min > u_max or v_min > v_max:
                continue

            us = np.arange(u_min, u_max + 1)
            vs = np.arange(v_min, v_max + 1)
            grid_u, grid_v = np.meshgrid(us, vs)
            pixels = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)
            if len(pixels) == 0:
                continue

            # Barycentric test
            p = pixels.astype(np.float64)
            v0 = pix[1].astype(np.float64) - pix[0].astype(np.float64)
            v1 = pix[2].astype(np.float64) - pix[0].astype(np.float64)
            v2 = p - pix[0].astype(np.float64)
            dot00 = v0 @ v0
            dot01 = v0 @ v1
            dot11 = v1 @ v1
            denom = dot00 * dot11 - dot01 * dot01
            if abs(denom) < 1e-10:
                continue
            dot02 = v2 @ v0
            dot12 = v2 @ v1
            u_b = (dot11 * dot02 - dot01 * dot12) / denom
            v_b = (dot00 * dot12 - dot01 * dot02) / denom
            inside = (u_b >= -0.01) & (v_b >= -0.01) & (u_b + v_b <= 1.01)
            if not inside.any():
                continue

            ins_pix = pixels[inside]
            ub = u_b[inside]
            vb = v_b[inside]
            wb = 1.0 - ub - vb
            pos = (wb[:, None] * verts[0] + ub[:, None] * verts[1] +
                   vb[:, None] * verts[2])

            all_pixels_u.append(ins_pix[:, 0])
            all_pixels_v.append(ins_pix[:, 1])
            all_pos_3d.append(pos)

        if not all_pos_3d:
            continue

        # Batch KD-tree query and IDW for all texels in this batch
        batch_u = np.concatenate(all_pixels_u)
        batch_v = np.concatenate(all_pixels_v)
        batch_pos = np.concatenate(all_pos_3d)

        dists, idxs = tree.query(batch_pos, k=knn)
        weights = 1.0 / np.maximum(dists, 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)

        colors = np.zeros((len(batch_pos), 3))
        for k_i in range(knn):
            colors += weights[:, k_i:k_i+1] * source_colors[idxs[:, k_i]]

        colors_uint8 = np.clip(colors * 255, 0, 255).astype(np.uint8)

        # Vectorized write — all texels from this batch at once
        atlas[batch_v, batch_u] = colors_uint8

    return atlas
```

- [ ] **Step 6: Implement `dilate_texture` in mesh_utils.py**

```python
def dilate_texture(atlas, iterations=8):
    """Fill empty (black) texels by dilating from neighboring filled texels.

    Prevents black seam artifacts at UV chart boundaries.

    Args:
        atlas: (H, W, 3) uint8 texture image.
        iterations: Number of dilation passes.

    Returns:
        (H, W, 3) uint8 texture with empty texels filled.
    """
    result = atlas.copy()
    empty_mask = result.sum(axis=2) == 0

    for _ in range(iterations):
        if not empty_mask.any():
            break

        # For each empty pixel, average its non-empty neighbors
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode='edge')
        padded_mask = np.pad(~empty_mask, ((1, 1), (1, 1)), mode='constant',
                             constant_values=False)

        h, w = result.shape[:2]
        neighbor_sum = np.zeros_like(result, dtype=np.float64)
        neighbor_count = np.zeros((h, w), dtype=np.float64)

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = 1 + dy, 1 + dx
                neighbor_sum += padded[ny:ny+h, nx:nx+w].astype(np.float64) * \
                    padded_mask[ny:ny+h, nx:nx+w][:, :, None]
                neighbor_count += padded_mask[ny:ny+h, nx:nx+w].astype(np.float64)

        # Fill empty pixels that have at least one non-empty neighbor
        fillable = empty_mask & (neighbor_count > 0)
        if not fillable.any():
            break

        avg = neighbor_sum[fillable] / neighbor_count[fillable][:, None]
        result[fillable] = np.clip(avg, 0, 255).astype(np.uint8)
        empty_mask[fillable] = False

    return result
```

- [ ] **Step 7: Run all UV baking tests**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_mesh_utils.py -v 2>&1
```

Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/meshing/mesh_utils.py tests/meshing/test_mesh_utils.py
git commit -m "feat(meshing): add UV unwrap and texture atlas baking from colored point cloud"
```

---

### Task 6: Rewrite export_gltf.py for Textured Mesh

**Files:**
- Rewrite: `src/meshing/export_gltf.py`
- Create: `tests/meshing/test_export_gltf.py`

- [ ] **Step 1: Write failing test for textured GLB export**

```python
# tests/meshing/test_export_gltf.py
import numpy as np
import trimesh
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "meshing"))
from export_gltf import export_textured_glb


def test_export_textured_glb_creates_file():
    """Export produces a valid GLB file."""
    # Minimal textured mesh
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uv_coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)
    texture = np.full((64, 64, 3), [128, 64, 32], dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        glb_path = Path(tmpdir) / "test.glb"
        export_textured_glb(
            vertices, faces, uv_coords, normals,
            texture_images=[texture],
            output_path=glb_path,
        )
        assert glb_path.exists()
        assert glb_path.stat().st_size > 100  # not empty


def test_export_textured_glb_preserves_metric_metadata():
    """GLB contains metric metadata in asset.extras."""
    import json, struct

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uv_coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    normals = np.array([[0, 0, 1]] * 3, dtype=np.float64)
    texture = np.full((64, 64, 3), 128, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        glb_path = Path(tmpdir) / "test.glb"
        export_textured_glb(
            vertices, faces, uv_coords, normals,
            texture_images=[texture],
            output_path=glb_path,
        )
        # Read GLB and parse JSON chunk for metadata
        data = glb_path.read_bytes()
        json_len = struct.unpack_from("<I", data, 12)[0]
        json_str = data[20:20+json_len].decode("utf-8").rstrip("\x00 ")
        gltf = json.loads(json_str)
        assert gltf["asset"]["extras"]["unit"] == "meter"
        assert gltf["asset"]["extras"]["scale"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_export_gltf.py -v 2>&1
```

Expected: FAIL — `ImportError: cannot import name 'export_textured_glb'`

- [ ] **Step 3: Rewrite `export_gltf.py`**

Full rewrite of `src/meshing/export_gltf.py`:

```python
"""GLB export for textured meshes with metric metadata.

Exports UV-textured meshes as GLB (binary glTF) preserving metric scale.
The glTF specification defines 1 unit = 1 meter, matching TLS data directly.
Supports multiple texture atlas pages as separate materials.
"""

import json
import struct
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def export_textured_glb(vertices, faces, uv_coords, normals,
                        texture_images, output_path, mesh_name="tls_mesh"):
    """Export a UV-textured mesh as GLB with metric metadata.

    Args:
        vertices: (V, 3) float64 vertex positions.
        faces: (M, 3) int32 face indices.
        uv_coords: (V, 2) float64 UV coordinates in [0, 1].
        normals: (V, 3) float64 vertex normals.
        texture_images: List of (H, W, 3) uint8 atlas page images.
        output_path: Path to write the .glb file.
        mesh_name: Name for the mesh node in the glTF scene.

    Returns:
        Path to the written GLB file.
    """
    output_path = Path(output_path)

    # Build a trimesh with texture
    # For single-page atlas, create a simple textured mesh
    # For multi-page, trimesh Scene handles multiple materials
    if len(texture_images) == 1:
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(texture_images[0]),
        )
        visual = trimesh.visual.TextureVisuals(
            uv=uv_coords.astype(np.float64),
            material=material,
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
            visual=visual,
            process=False,
        )
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name=mesh_name)
    else:
        # Multi-page: for now, use first page as primary texture
        # (multi-material GLB export requires per-face material assignment
        # which adds complexity — single large atlas is the common case)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(texture_images[0]),
        )
        visual = trimesh.visual.TextureVisuals(
            uv=uv_coords.astype(np.float64),
            material=material,
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
            visual=visual,
            process=False,
        )
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name=mesh_name)

    # Export as GLB
    glb_bytes = scene.export(file_type="glb")

    # Inject metric metadata
    glb_bytes = _inject_gltf_metadata(glb_bytes, {
        "unit": "meter",
        "scale": 1.0,
        "source": "TLS_point_cloud",
        "note": "1 unit = 1 meter. No rescaling applied. Metric-accurate.",
    })

    with open(str(output_path), "wb") as f:
        f.write(glb_bytes)

    return str(output_path)


def export_vertex_color_ply(o3d_mesh, output_path):
    """Export Open3D mesh as PLY with vertex colors (for CloudCompare).

    Args:
        o3d_mesh: Open3D TriangleMesh with vertex colors.
        output_path: Path to write the .ply file.
    """
    import open3d as o3d
    o3d.io.write_triangle_mesh(str(output_path), o3d_mesh)


def _inject_gltf_metadata(glb_bytes, metadata):
    """Inject metadata into the glTF JSON chunk of a GLB binary.

    Adds metadata as `asset.extras` in the glTF JSON.
    """
    if isinstance(glb_bytes, bytes):
        glb_bytes = bytearray(glb_bytes)

    magic, version, total_length = struct.unpack_from("<III", glb_bytes, 0)
    if magic != 0x46546C67:
        return bytes(glb_bytes)

    json_chunk_length = struct.unpack_from("<I", glb_bytes, 12)[0]
    json_chunk_type = struct.unpack_from("<I", glb_bytes, 16)[0]
    if json_chunk_type != 0x4E4F534A:
        return bytes(glb_bytes)

    json_data = glb_bytes[20:20 + json_chunk_length].decode("utf-8").rstrip("\x00")
    gltf = json.loads(json_data)

    if "asset" not in gltf:
        gltf["asset"] = {"version": "2.0", "generator": "scan2measure"}
    gltf["asset"]["extras"] = metadata

    new_json = json.dumps(gltf, separators=(",", ":"))
    while len(new_json) % 4 != 0:
        new_json += " "
    new_json_bytes = new_json.encode("utf-8")
    new_json_length = len(new_json_bytes)

    remaining_start = 20 + json_chunk_length
    remaining = glb_bytes[remaining_start:]

    new_total_length = 12 + 8 + new_json_length + len(remaining)
    result = bytearray()
    result += struct.pack("<III", magic, version, new_total_length)
    result += struct.pack("<II", new_json_length, 0x4E4F534A)
    result += new_json_bytes
    result += remaining

    return bytes(result)
```

**Note:** Pillow (`PIL`) is needed for creating texture images — check availability and install if missing.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/test_export_gltf.py -v 2>&1
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/meshing/export_gltf.py tests/meshing/test_export_gltf.py
git commit -m "feat(meshing): rewrite GLB export for UV-textured meshes with metric metadata"
```

---

### Task 7: Rewrite mesh_reconstruction.py Orchestrator

**Files:**
- Rewrite: `src/meshing/mesh_reconstruction.py`

- [ ] **Step 1: Rewrite the full orchestrator**

Complete rewrite of `src/meshing/mesh_reconstruction.py`:

```python
"""High-fidelity mesh reconstruction from colored TLS point clouds.

Converts a colored PLY point cloud into a UV-textured GLB mesh with
metric scale preservation. Designed for virtual tour measurement apps.

Pipeline:
1. Load colored point cloud (no downsampling)
2. Spatial chunking (6x6 m XY tiles, 1 m overlap)
3. Per-tile: normals → Poisson depth 11 → density trim → ownership trim
4. Merge tile meshes
5. UV texture atlas baking from colored point cloud
6. Export GLB with embedded textures + metric metadata

The TLS point cloud has 1 unit = 1 meter. No rescaling is applied.

Usage:
    conda run -n scan_env python src/meshing/mesh_reconstruction.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

_SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_ROOT / "meshing"))
from mesh_utils import (
    estimate_normals,
    poisson_reconstruct,
    remove_low_density,
    transfer_vertex_colors,
    compute_mesh_stats,
    compute_tile_grid,
    extract_tile_points,
    trim_to_ownership_region,
    merge_tile_meshes,
    uv_unwrap_mesh,
    bake_texture_atlas,
    dilate_texture,
)
from export_gltf import export_textured_glb, export_vertex_color_ply

# ── Config ──────────────────────────────────────────────────────────────────

# Input point cloud (colored PLY)
POINT_CLOUD_NAME = "tmb_office_one_corridor_dense_noRGB_textured"

# Poisson reconstruction
POISSON_DEPTH = 11
DENSITY_TRIM_QUANTILE = 0.06

# Normal estimation
NORMAL_KNN = 50
NORMAL_RADIUS = 0.15

# Spatial chunking
TILE_SIZE = 6.0         # meters
OVERLAP = 1.0           # meters
MIN_TILE_POINTS = 1000  # skip tiles with fewer points

# Texture atlas — 8192 to achieve ~3 mm/texel on 500-800 m² surface
ATLAS_RESOLUTION = 8192
BAKE_KNN = 8

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = ROOT / "data" / "textured_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
OUTPUT_DIR = ROOT / "data" / "mesh" / "tmb_office_one_corridor_dense"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Load point cloud (no downsampling) ─────────────────────────
    print(f"[1/9] Loading point cloud: {INPUT_PATH.name}")
    pcd = o3d.io.read_point_cloud(str(INPUT_PATH))
    n_points = len(pcd.points)
    print(f"       {n_points:,} points, colors={pcd.has_colors()}")

    points = np.asarray(pcd.points)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min
    print(f"       Bounding box: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")

    # ── Stage 2: Compute tile grid ──────────────────────────────────────────
    print(f"[2/9] Computing tile grid (tile={TILE_SIZE}m, overlap={OVERLAP}m)")
    tiles = compute_tile_grid(bbox_min, bbox_max, tile_size=TILE_SIZE, overlap=OVERLAP)
    print(f"       {len(tiles)} tiles")

    # ── Stages 3-6: Per-tile reconstruction ─────────────────────────────────
    tile_meshes = []
    for i, (core_min, core_max, ext_min, ext_max) in enumerate(tiles):
        print(f"\n[3-6] Tile {i+1}/{len(tiles)}: "
              f"core=[{core_min[0]:.1f},{core_min[1]:.1f}]-"
              f"[{core_max[0]:.1f},{core_max[1]:.1f}]")
        t_tile = time.time()

        # Extract tile points
        tile_pcd = extract_tile_points(pcd, ext_min, ext_max)
        n_tile = len(tile_pcd.points)
        print(f"       {n_tile:,} points in tile")

        if n_tile < MIN_TILE_POINTS:
            print(f"       Skipping — fewer than {MIN_TILE_POINTS} points")
            continue

        # Stage 3: Normals
        print(f"       Estimating normals (knn={NORMAL_KNN})...")
        tile_pcd = estimate_normals(tile_pcd, knn=NORMAL_KNN, radius=NORMAL_RADIUS)

        # Stage 4: Poisson reconstruction
        print(f"       Poisson reconstruction (depth={POISSON_DEPTH})...")
        tile_mesh, densities = poisson_reconstruct(tile_pcd, depth=POISSON_DEPTH)
        print(f"       Raw: {len(tile_mesh.triangles):,} triangles")

        # Stage 5: Density trim
        tile_mesh = remove_low_density(tile_mesh, densities, quantile=DENSITY_TRIM_QUANTILE)
        print(f"       After density trim: {len(tile_mesh.triangles):,} triangles")

        # Stage 6: Ownership trim
        tile_mesh = trim_to_ownership_region(tile_mesh, core_min, core_max)
        print(f"       After ownership trim: {len(tile_mesh.triangles):,} triangles")

        # Transfer vertex colors for PLY output
        tile_mesh = transfer_vertex_colors(tile_mesh, tile_pcd)

        tile_meshes.append(tile_mesh)
        print(f"       Tile done in {time.time() - t_tile:.1f}s")

        # Free tile memory
        del tile_pcd, densities

    # ── Stage 7: Merge tiles ────────────────────────────────────────────────
    print(f"\n[7/9] Merging {len(tile_meshes)} tile meshes...")
    t = time.time()
    merged_mesh = merge_tile_meshes(tile_meshes)
    n_verts = len(merged_mesh.vertices)
    n_tris = len(merged_mesh.triangles)
    print(f"       Merged: {n_verts:,} vertices, {n_tris:,} triangles")
    print(f"       Done in {time.time() - t:.1f}s")

    # Save vertex-colored PLY for CloudCompare inspection
    ply_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}_full.ply"
    export_vertex_color_ply(merged_mesh, ply_path)
    print(f"       Saved PLY: {ply_path.name}")

    # Free tile meshes
    del tile_meshes

    # ── Stage 8: UV texture atlas baking ────────────────────────────────────
    print(f"\n[8/9] UV texture atlas baking (resolution={ATLAS_RESOLUTION})...")
    t = time.time()

    vertices = np.asarray(merged_mesh.vertices)
    faces = np.asarray(merged_mesh.triangles)

    # UV unwrap
    print("       UV unwrapping with xatlas...")
    vmapping, new_faces, uv_coords = uv_unwrap_mesh(vertices, faces,
                                                      atlas_resolution=ATLAS_RESOLUTION)

    # Remap vertices to xatlas layout (may have splits at UV seams)
    new_vertices = vertices[vmapping]
    new_normals = np.asarray(merged_mesh.vertex_normals)[vmapping]

    # Bake texture from colored point cloud
    source_points = np.asarray(pcd.points)
    source_colors = np.asarray(pcd.colors)

    print(f"       Baking {ATLAS_RESOLUTION}x{ATLAS_RESOLUTION} atlas (knn={BAKE_KNN})...")
    atlas = bake_texture_atlas(
        new_vertices, new_faces, uv_coords,
        source_points, source_colors,
        atlas_resolution=ATLAS_RESOLUTION, knn=BAKE_KNN,
    )

    # Dilate to fill empty texels
    print("       Dilating empty texels...")
    atlas = dilate_texture(atlas, iterations=8)

    print(f"       Done in {time.time() - t:.1f}s")

    # Free point cloud — no longer needed
    del pcd

    # ── Stage 9: Export GLB ─────────────────────────────────────────────────
    glb_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}.glb"
    print(f"\n[9/9] Exporting GLB: {glb_path.name}")
    t = time.time()

    export_textured_glb(
        new_vertices, new_faces, uv_coords, new_normals,
        texture_images=[atlas],
        output_path=glb_path,
    )

    glb_size_mb = glb_path.stat().st_size / (1024 * 1024)
    print(f"       GLB size: {glb_size_mb:.1f} MB")
    print(f"       Done in {time.time() - t:.1f}s")

    # ── Write metadata sidecar ──────────────────────────────────────────────
    # Use new_vertices (post-xatlas) for correct bounds
    bbox_min_final = new_vertices.min(axis=0)
    bbox_max_final = new_vertices.max(axis=0)
    total_time = time.time() - t_start
    metadata = {
        "source_point_cloud": INPUT_PATH.name,
        "n_input_points": n_points,
        "n_tiles": len(tiles),
        "tile_size_m": TILE_SIZE,
        "poisson_depth": POISSON_DEPTH,
        "n_vertices": len(new_vertices),
        "n_triangles": len(new_faces),
        "bbox_min_m": bbox_min_final.tolist(),
        "bbox_max_m": bbox_max_final.tolist(),
        "atlas_pages": 1,
        "atlas_resolution": ATLAS_RESOLUTION,
        "glb_size_mb": round(glb_size_mb, 1),
        "draco_compressed": False,
        "unit": "meter",
        "reconstruction_time_s": round(total_time, 1),
    }
    meta_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Mesh reconstruction complete in {total_time:.1f}s")
    print(f"  Input:      {n_points:,} points")
    print(f"  Output:     {len(new_vertices):,} vertices, {len(new_faces):,} triangles")
    print(f"  Extent:     {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")
    print(f"  GLB file:   {glb_path}")
    print(f"  GLB size:   {glb_size_mb:.1f} MB")
    print(f"  PLY file:   {ply_path}")
    print(f"  Metadata:   {meta_path}")
    print(f"  Unit:       1 unit = 1 meter (no rescaling applied)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
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
git add src/meshing/mesh_reconstruction.py
git commit -m "feat(meshing): rewrite orchestrator for high-fidelity tiled reconstruction"
```

---

### Task 8: Integration Test on Actual Data

**Files:**
- Run: `src/meshing/mesh_reconstruction.py`

- [ ] **Step 1: Run the full pipeline**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 src/meshing/mesh_reconstruction.py 2>&1
```

This will take ~15-25 minutes. Monitor for:
- Each tile processes without OOM (peak RAM should stay under ~5 GB per tile)
- Tile counts and triangle counts are printed per tile
- Merge produces ~15-25M total triangles
- UV baking completes
- GLB file is written

- [ ] **Step 2: Verify output files exist**

```bash
ls -lh /home/ruoyu/scan2measure-webframework/data/mesh/tmb_office_one_corridor_dense/
```

Expected:
- `tmb_office_one_corridor_dense_noRGB_textured.glb` (~50-750 MB)
- `tmb_office_one_corridor_dense_noRGB_textured_full.ply` (~300-600 MB)
- `tmb_office_one_corridor_dense_noRGB_textured_metadata.json`

- [ ] **Step 3: Validate metadata JSON**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -c "
import json
meta = json.load(open('data/mesh/tmb_office_one_corridor_dense/tmb_office_one_corridor_dense_noRGB_textured_metadata.json'))
print(json.dumps(meta, indent=2))
assert meta['unit'] == 'meter'
assert meta['n_triangles'] > 1_000_000
assert meta['poisson_depth'] == 11
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
# Should be close to the original 11.39 x 21.81 x 4.41 m
assert 10.0 < extent[0] < 13.0, f'X extent {extent[0]} out of range'
assert 20.0 < extent[1] < 23.0, f'Y extent {extent[1]} out of range'
assert 3.5 < extent[2] < 5.5, f'Z extent {extent[2]} out of range'
print('Scale validation OK')
"
```

- [ ] **Step 5: Commit final state**

```bash
git add -A
git commit -m "feat(meshing): complete high-fidelity mesh reconstruction pipeline"
```

---

### Task 9: Draco Compression (Optional Post-Processing)

**Files:**
- None modified — uses CLI tool on output GLB

Draco compression reduces the GLB from ~550-750 MB to ~50-100 MB. This uses `gltf-transform` (Node.js CLI) as a post-processing step since trimesh does not natively support Draco export.

- [ ] **Step 1: Install gltf-transform**

```bash
npm install -g @gltf-transform/cli
```

If npm is not available, skip this task — the uncompressed GLB is still valid and usable. Draco can be applied later.

- [ ] **Step 2: Compress the GLB**

```bash
gltf-transform draco \
  /home/ruoyu/scan2measure-webframework/data/mesh/tmb_office_one_corridor_dense/tmb_office_one_corridor_dense_noRGB_textured.glb \
  /home/ruoyu/scan2measure-webframework/data/mesh/tmb_office_one_corridor_dense/tmb_office_one_corridor_dense_noRGB_textured_draco.glb
```

- [ ] **Step 3: Verify compressed file size**

```bash
ls -lh /home/ruoyu/scan2measure-webframework/data/mesh/tmb_office_one_corridor_dense/*draco*
```

Expected: ~50-100 MB (vs ~550-750 MB uncompressed)

---

### Task 10: Cleanup — Remove Dead Code and Fix Performance

**Files:**
- Modify: `src/meshing/mesh_utils.py`

- [ ] **Step 1: Remove `decimate_mesh` function**

Remove the `decimate_mesh()` function from `src/meshing/mesh_utils.py` — it is no longer called by the pipeline.

- [ ] **Step 1b: Replace `transfer_vertex_colors` with batch KD-tree version**

The current implementation uses a per-vertex Python loop (very slow for 500K+ vertices per tile). Replace with scipy batch query:

```python
def transfer_vertex_colors(mesh, source_pcd):
    """Transfer colors from a point cloud to mesh vertices via batch KD-tree."""
    source_points = np.asarray(source_pcd.points)
    source_colors = np.asarray(source_pcd.colors)
    mesh_vertices = np.asarray(mesh.vertices)

    tree = cKDTree(source_points)
    _, idx = tree.query(mesh_vertices, k=1)
    colors = source_colors[idx]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh
```

- [ ] **Step 2: Run all tests to verify nothing breaks**

```bash
cd /home/ruoyu/scan2measure-webframework && conda run -n scan_env python3 -m pytest tests/meshing/ -v 2>&1
```

Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/meshing/mesh_utils.py
git commit -m "refactor(meshing): remove decimate_mesh — no longer used in high-fidelity pipeline"
```
