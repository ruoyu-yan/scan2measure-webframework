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

import gc
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
GLB_TARGET_TRIANGLES = 500_000  # decimate for UV/bake speed (full-res kept in PLY)

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = ROOT / "data" / "textured_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
OUTPUT_DIR = ROOT / "data" / "mesh" / "tmb_office_one_corridor_dense"


# ── Main ─────────────────────────────────────────────────────────────────────

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
    if not has_colors:
        print("       WARNING: Point cloud has no colors. Texture baking will be skipped.")

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

    # ── Stage 11: Save full-resolution vertex-colored PLY ───────────────────
    ply_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}_vertex_colored.ply"
    print(f"[11/17] Saving full-res vertex-colored PLY: {ply_path.name}")
    export_vertex_color_ply(merged_mesh, ply_path)
    n_full_tris = len(merged_mesh.triangles)

    # ── Stage 12: Decimate for UV/bake (full-res preserved in PLY) ───────────
    if n_full_tris > GLB_TARGET_TRIANGLES:
        print(f"[12/17] Decimating {n_full_tris:,} → {GLB_TARGET_TRIANGLES:,} triangles "
              f"for textured GLB ...")
        t = time.time()
        merged_mesh = merged_mesh.simplify_quadric_decimation(
            target_number_of_triangles=GLB_TARGET_TRIANGLES
        )
        merged_mesh.compute_vertex_normals()
        n_dec_verts = len(merged_mesh.vertices)
        n_dec_tris = len(merged_mesh.triangles)
        print(f"       {n_dec_verts:,} vertices, {n_dec_tris:,} triangles "
              f"in {time.time()-t:.1f}s")
    else:
        print(f"[12/17] No decimation needed ({n_full_tris:,} triangles ≤ target)")
        n_dec_tris = n_full_tris

    # ── Stage 13: Reload original cloud for texture baking ───────────────────
    print(f"[13/17] Reloading original point cloud for texture baking ...")
    pcd_full = o3d.io.read_point_cloud(str(INPUT_PATH))
    source_points = np.asarray(pcd_full.points)
    source_colors = np.asarray(pcd_full.colors) if pcd_full.has_colors() else None

    if source_colors is None:
        print("       WARNING: No colors in point cloud. Skipping texture baking.")
        print("       Vertex-colored PLY was already saved. No GLB produced.")
        del pcd_full
        return

    # ── Stage 14: UV unwrap ──────────────────────────────────────────────────
    print(f"[14/17] UV unwrapping with xatlas (resolution={ATLAS_RESOLUTION}) ...")
    t = time.time()
    vertices = np.asarray(merged_mesh.vertices)
    normals_arr = np.asarray(merged_mesh.vertex_normals)
    faces = np.asarray(merged_mesh.triangles)

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

    # ── Stage 15: Bake texture atlas ─────────────────────────────────────────
    print(f"[15/17] Baking texture atlas ({ATLAS_RESOLUTION}x{ATLAS_RESOLUTION}, "
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

    # ── Stage 16: Dilate empty texels ────────────────────────────────────────
    print(f"[16/17] Dilating empty texels ...")
    atlas = dilate_texture(atlas, iterations=8)

    # ── Stage 17: Export GLB ─────────────────────────────────────────────────
    glb_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}.glb"
    print(f"[17/17] Exporting GLB: {glb_path.name}")
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
        "n_full_res_triangles": int(n_full_tris),
        "glb_target_triangles": GLB_TARGET_TRIANGLES,
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


if __name__ == "__main__":
    main()
