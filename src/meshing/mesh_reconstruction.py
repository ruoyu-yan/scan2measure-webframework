"""Mesh reconstruction orchestrator for colored TLS point clouds.

Converts a colored PLY point cloud into a web-ready GLB mesh with metric
scale preservation using a tiled Poisson reconstruction pipeline.

Pipeline:
1.  Load colored point cloud (no downsampling), print extent
2.  Compute tile grid, print tile count
3.  Per-tile loop (sequential):
    - Extract tile points
    - Skip if fewer than MIN_TILE_POINTS
    - Estimate normals
    - Poisson reconstruction
    - Density trim
    - Ownership trim
    - Transfer vertex colors (for PLY output)
    - Append to tile_meshes list
    - Delete tile_pcd, densities to free memory
4.  Merge tile meshes, print counts
5.  Save vertex-colored PLY via export_vertex_color_ply
6.  UV unwrap with xatlas (ATLAS_RESOLUTION=8192)
7.  Remap vertices: new_vertices = vertices[vmapping], new_normals = normals[vmapping]
8.  Bake texture atlas from colored point cloud colors
9.  Dilate empty texels
10. Export GLB via export_textured_glb
11. Write JSON metadata sidecar
12. Print summary

The TLS point cloud has 1 unit = 1 meter. No rescaling is applied at any step.
The glTF spec uses meters as the default unit, so the exported mesh preserves
metric accuracy for downstream measurement tools.

Usage:
    python src/meshing/mesh_reconstruction.py
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

# ── Config ───────────────────────────────────────────────────────────────────

POINT_CLOUD_NAME = "tmb_office_one_corridor_dense_noRGB_textured"
POISSON_DEPTH = 11
DENSITY_TRIM_QUANTILE = 0.06
NORMAL_KNN = 50
NORMAL_RADIUS = 0.15
TILE_SIZE = 6.0
OVERLAP = 1.0
MIN_TILE_POINTS = 1000
ATLAS_RESOLUTION = 8192
BAKE_KNN = 8

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = ROOT / "data" / "textured_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
OUTPUT_DIR = ROOT / "data" / "mesh" / "tmb_office_one_corridor_dense"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Load point cloud ────────────────────────────────────────────
    print(f"[1] Loading point cloud: {INPUT_PATH.name}")
    pcd = o3d.io.read_point_cloud(str(INPUT_PATH))
    n_input_points = len(pcd.points)
    has_colors = pcd.has_colors()
    print(f"    {n_input_points:,} points, colors={has_colors}")

    points_arr = np.asarray(pcd.points)
    bbox_min_pcd = points_arr.min(axis=0)
    bbox_max_pcd = points_arr.max(axis=0)
    extent = bbox_max_pcd - bbox_min_pcd
    print(f"    Extent: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")

    # ── Stage 2: Compute tile grid ───────────────────────────────────────────
    print(f"[2] Computing tile grid (tile_size={TILE_SIZE}m, overlap={OVERLAP}m)")
    tiles = compute_tile_grid(bbox_min_pcd, bbox_max_pcd,
                              tile_size=TILE_SIZE, overlap=OVERLAP)
    n_tiles = len(tiles)
    print(f"    {n_tiles} tiles")

    # ── Stage 3: Per-tile reconstruction ─────────────────────────────────────
    print(f"[3] Running per-tile Poisson reconstruction ...")
    tile_meshes = []
    n_skipped = 0

    for tile_idx, (core_min, core_max, ext_min, ext_max) in enumerate(tiles):
        print(f"    Tile {tile_idx + 1}/{n_tiles} ...", end=" ", flush=True)

        # Extract points within extended tile bounds
        tile_pcd = extract_tile_points(pcd, ext_min, ext_max)
        n_tile_pts = len(tile_pcd.points)

        if n_tile_pts < MIN_TILE_POINTS:
            print(f"skipped ({n_tile_pts} pts < {MIN_TILE_POINTS})")
            n_skipped += 1
            continue

        print(f"{n_tile_pts:,} pts", end=" ", flush=True)

        # Estimate normals
        tile_pcd = estimate_normals(tile_pcd, knn=NORMAL_KNN, radius=NORMAL_RADIUS)

        # Poisson reconstruction
        tile_mesh, densities = poisson_reconstruct(tile_pcd, depth=POISSON_DEPTH)

        # Density trim
        tile_mesh = remove_low_density(tile_mesh, densities,
                                       quantile=DENSITY_TRIM_QUANTILE)

        # Ownership trim — keep only triangles whose centroids fall in the core
        tile_mesh = trim_to_ownership_region(tile_mesh, core_min, core_max)

        # Transfer vertex colors (from tile point cloud)
        tile_mesh = transfer_vertex_colors(tile_mesh, tile_pcd)

        n_verts = len(tile_mesh.vertices)
        n_tris = len(tile_mesh.triangles)
        print(f"-> {n_verts:,}v / {n_tris:,}t")

        tile_meshes.append(tile_mesh)

        # Free tile data
        del tile_pcd, densities

    print(f"    Tiles processed: {len(tile_meshes)}, skipped: {n_skipped}")

    # ── Stage 4: Merge tile meshes ───────────────────────────────────────────
    print(f"[4] Merging {len(tile_meshes)} tile meshes ...")
    merged_mesh = merge_tile_meshes(tile_meshes)
    n_merged_verts = len(merged_mesh.vertices)
    n_merged_tris = len(merged_mesh.triangles)
    print(f"    Merged: {n_merged_verts:,} vertices, {n_merged_tris:,} triangles")

    # ── Stage 5: Save vertex-colored PLY ─────────────────────────────────────
    ply_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}_vertex_colored.ply"
    print(f"[5] Saving vertex-colored PLY: {ply_path.name}")
    export_vertex_color_ply(merged_mesh, ply_path)

    # ── Stage 6: UV unwrap with xatlas ───────────────────────────────────────
    print(f"[6] UV unwrapping with xatlas (resolution hint={ATLAS_RESOLUTION}) ...")
    t = time.time()
    vertices = np.asarray(merged_mesh.vertices)
    normals = np.asarray(merged_mesh.vertex_normals)
    faces = np.asarray(merged_mesh.triangles)

    vmapping, new_faces, uv_coords = uv_unwrap_mesh(
        vertices, faces, atlas_resolution=ATLAS_RESOLUTION
    )
    print(f"    Done in {time.time() - t:.1f}s — "
          f"{len(vmapping):,} new vertices, {len(new_faces):,} faces")

    # ── Stage 7: Remap vertices and normals via vmapping ─────────────────────
    print(f"[7] Remapping vertices and normals via vmapping ...")
    new_vertices = vertices[vmapping]
    new_normals = normals[vmapping]

    # ── Stage 8: Bake texture atlas ──────────────────────────────────────────
    print(f"[8] Baking texture atlas ({ATLAS_RESOLUTION}x{ATLAS_RESOLUTION}) ...")
    t = time.time()
    source_points = np.asarray(pcd.points)
    source_colors = np.asarray(pcd.colors)

    atlas = bake_texture_atlas(
        new_vertices, new_faces, uv_coords,
        source_points, source_colors,
        atlas_resolution=ATLAS_RESOLUTION,
        knn=BAKE_KNN,
    )
    print(f"    Done in {time.time() - t:.1f}s")

    # ── Stage 9: Dilate empty texels ─────────────────────────────────────────
    print(f"[9] Dilating empty texels ...")
    atlas = dilate_texture(atlas, iterations=8)

    # ── Stage 10: Export GLB ─────────────────────────────────────────────────
    glb_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}.glb"
    print(f"[10] Exporting GLB: {glb_path.name}")
    t = time.time()
    export_textured_glb(
        new_vertices, new_faces, uv_coords, new_normals,
        [atlas], glb_path, mesh_name="tls_mesh"
    )
    glb_size_mb = glb_path.stat().st_size / (1024 * 1024)
    print(f"     GLB size: {glb_size_mb:.1f} MB")
    print(f"     Done in {time.time() - t:.1f}s")

    # ── Stage 11: Write JSON metadata sidecar ────────────────────────────────
    print(f"[11] Writing JSON metadata sidecar ...")
    total_time = time.time() - t_start

    # Bounding box from new_vertices (post-xatlas), NOT from merged_mesh
    bbox_min_m = new_vertices.min(axis=0).tolist()
    bbox_max_m = new_vertices.max(axis=0).tolist()

    metadata = {
        "source_point_cloud": str(INPUT_PATH),
        "n_input_points": n_input_points,
        "n_tiles": n_tiles,
        "tile_size_m": TILE_SIZE,
        "poisson_depth": POISSON_DEPTH,
        "n_vertices": int(len(new_vertices)),
        "n_triangles": int(len(new_faces)),
        "bbox_min_m": bbox_min_m,
        "bbox_max_m": bbox_max_m,
        "atlas_pages": 1,
        "atlas_resolution": ATLAS_RESOLUTION,
        "glb_size_mb": round(glb_size_mb, 2),
        "draco_compressed": False,
        "unit": "meter",
        "reconstruction_time_s": round(total_time, 1),
    }

    json_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}_metadata.json"
    with open(str(json_path), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"     Metadata saved: {json_path.name}")

    # ── Stage 12: Print summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Mesh reconstruction complete in {total_time:.1f}s")
    print(f"  Input:       {n_input_points:,} points")
    print(f"  Tiles:       {n_tiles} total, {n_skipped} skipped")
    print(f"  Output:      {len(new_vertices):,} vertices, {len(new_faces):,} triangles")
    bbox_extent = [bbox_max_m[i] - bbox_min_m[i] for i in range(3)]
    print(f"  Extent:      {bbox_extent[0]:.2f} x {bbox_extent[1]:.2f} x {bbox_extent[2]:.2f} m")
    print(f"  GLB file:    {glb_path}")
    print(f"  GLB size:    {glb_size_mb:.1f} MB")
    print(f"  Atlas:       {ATLAS_RESOLUTION}x{ATLAS_RESOLUTION} px")
    print(f"  Unit:        1 unit = 1 meter (no rescaling applied)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
