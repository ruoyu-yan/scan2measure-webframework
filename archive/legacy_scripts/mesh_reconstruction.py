"""Balanced mesh reconstruction from colored TLS point clouds.

Converts a colored PLY point cloud into a UV-textured GLB mesh with
metric scale preservation.

Pipeline:
1.  Load colored point cloud
2.  5 mm voxel downsample, free original cloud
3.  Compute tile grid (6x6 m XY tiles, 1 m overlap)
4-9. Per-tile (parallel): extract -> normals -> Poisson depth 9 -> density trim
     -> ownership trim -> save to disk, free tile from RAM
10. Reload tiles -> merge
11. Save vertex-colored PLY (CloudCompare inspection)
12. Decimate for UV/bake (full-res preserved in PLY)
13. Reload original cloud for texture baking
14. UV unwrap with xatlas (4096 resolution)
15. Bake texture atlas (KNN=4 IDW from full cloud, parallel)
16. Dilate empty texels
17. Export GLB with metric metadata + JSON sidecar

The TLS point cloud has 1 unit = 1 meter. No rescaling is applied.
Design spec: docs/superpowers/specs/2026-03-23-balanced-mesh-reconstruction-design.md

Usage:
    conda run -n scan_env python src/meshing/mesh_reconstruction.py
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

_SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_ROOT / "meshing"))
sys.path.insert(0, str(_SRC_ROOT / "utils"))
from config_loader import load_config, progress
from mesh_utils import (
    compute_tile_grid,
    process_tiles_parallel,
    merge_tile_meshes,
    uv_unwrap_mesh,
    bake_texture_atlas,
    dilate_texture,
)
from export_gltf import export_textured_glb, export_vertex_color_ply

# -- Config -------------------------------------------------------------------

POINT_CLOUD_NAME = "tmb_office_one_corridor_bigger_noRGB_textured"

# Quality tier: "preview" (~2-3 min), "balanced" (~5-8 min), "high" (~15-20 min)
QUALITY_TIER = "balanced"

_QUALITY_PRESETS = {
    "preview": {    # Fast check — ~68 mm octree cells, coarser texture
        "poisson_depth": 7,
        "voxel_size": 0.015,
        "normal_knn": 15,
        "atlas_resolution": 2048,
        "glb_target_triangles": 250_000,
    },
    "balanced": {   # Good quality — ~34 mm octree cells, full texture
        "poisson_depth": 8,
        "voxel_size": 0.010,
        "normal_knn": 20,
        "atlas_resolution": 4096,
        "glb_target_triangles": 500_000,
    },
    "high": {       # Maximum detail — ~17 mm octree cells, full texture
        "poisson_depth": 9,
        "voxel_size": 0.005,
        "normal_knn": 30,
        "atlas_resolution": 4096,
        "glb_target_triangles": 500_000,
    },
}

_preset = _QUALITY_PRESETS[QUALITY_TIER]
POISSON_DEPTH = _preset["poisson_depth"]
VOXEL_SIZE = _preset["voxel_size"]
NORMAL_KNN = _preset["normal_knn"]
ATLAS_RESOLUTION = _preset["atlas_resolution"]
GLB_TARGET_TRIANGLES = _preset["glb_target_triangles"]

DENSITY_TRIM_QUANTILE = 0.06
NORMAL_RADIUS = 0.15
TILE_SIZE = 6.0
OVERLAP = 1.0
MIN_TILE_POINTS = 1000

MAX_TILE_WORKERS = min(6, max(1, os.cpu_count() // 8))
MAX_BAKE_WORKERS = min(4, max(1, os.cpu_count() // 8))

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = ROOT / "data" / "textured_point_cloud" / "tmb_office_one_corridor_bigger_noRGB" / f"{POINT_CLOUD_NAME}.ply"
OUTPUT_DIR = ROOT / "data" / "mesh" / "tmb_office_one_corridor_bigger_noRGB"


# -- Main ---------------------------------------------------------------------

def main():
    cfg = load_config()
    pc_name = cfg.get("point_cloud_name", POINT_CLOUD_NAME)
    quality_tier = cfg.get("quality_tier", QUALITY_TIER)
    preset = _QUALITY_PRESETS[quality_tier]
    poisson_depth = preset["poisson_depth"]
    voxel_size = preset["voxel_size"]
    normal_knn = preset["normal_knn"]
    atlas_resolution = preset["atlas_resolution"]
    glb_target_triangles = preset["glb_target_triangles"]
    input_path = Path(cfg["input_path"]) if cfg.get("input_path") else INPUT_PATH
    out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else OUTPUT_DIR

    t_start = time.time()
    stage_times = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    print(f"Quality tier: {quality_tier} (depth={poisson_depth}, "
          f"voxel={voxel_size*1000:.0f}mm, atlas={atlas_resolution})")

    # -- Stage 1: Load point cloud --------------------------------------------
    progress(1, 17, "Loading point cloud")
    t_stage = time.time()
    print(f"[1/17] Loading point cloud: {input_path.name}")
    pcd = o3d.io.read_point_cloud(str(input_path))
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
    stage_times['1_load'] = time.time() - t_stage

    # -- Stage 2: Voxel downsample --------------------------------------------
    t_stage = time.time()
    progress(2, 17, "Voxel downsample")
    print(f"[2/17] Voxel downsample ({voxel_size*1000:.0f} mm) ...")
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    n_ds_points = len(pcd_ds.points)
    del pcd  # Free original cloud (~400 MB)
    gc.collect()
    print(f"       {n_input_points:,} -> {n_ds_points:,} points "
          f"({100*n_ds_points/n_input_points:.0f}%)")
    stage_times['2_downsample'] = time.time() - t_stage

    # -- Stage 3: Compute tile grid -------------------------------------------
    t_stage = time.time()
    progress(3, 17, "Computing tile grid")
    ds_points = np.asarray(pcd_ds.points)
    bbox_min_ds = ds_points.min(axis=0)
    bbox_max_ds = ds_points.max(axis=0)
    print(f"[3/17] Computing tile grid (tile={TILE_SIZE}m, overlap={OVERLAP}m)")
    tiles = compute_tile_grid(bbox_min_ds, bbox_max_ds,
                              tile_size=TILE_SIZE, overlap=OVERLAP)
    n_tiles = len(tiles)
    print(f"       {n_tiles} tiles")
    stage_times['3_tile_grid'] = time.time() - t_stage

    # -- Stages 4-9: Per-tile reconstruction (parallel) -----------------------
    t_stage = time.time()
    progress(4, 17, "Per-tile Poisson reconstruction (parallel)")
    print(f"[4-9] Running per-tile Poisson reconstruction "
          f"({MAX_TILE_WORKERS} workers) ...")
    tile_ply_paths, n_skipped, tile_results = process_tiles_parallel(
        pcd_ds, tiles, tiles_dir,
        normal_knn=normal_knn,
        normal_radius=NORMAL_RADIUS,
        poisson_depth=poisson_depth,
        density_trim_quantile=DENSITY_TRIM_QUANTILE,
        min_tile_points=MIN_TILE_POINTS,
        max_workers=MAX_TILE_WORKERS,
    )
    print(f"       Tiles processed: {len(tile_ply_paths)}, skipped: {n_skipped}")
    stage_times['4-9_tiles'] = time.time() - t_stage

    # Free downsampled cloud -- no longer needed after tiling
    del pcd_ds
    gc.collect()

    # -- Stage 10: Merge tile meshes ------------------------------------------
    t_stage = time.time()
    progress(10, 17, "Merging tile meshes")
    print(f"[10/17] Merging {len(tile_ply_paths)} tile meshes ...")
    tile_meshes = []
    for tile_ply in tile_ply_paths:
        m = o3d.io.read_triangle_mesh(str(tile_ply))
        if len(m.triangles) > 0:
            tile_meshes.append(m)
    merged_mesh = merge_tile_meshes(tile_meshes)
    del tile_meshes
    gc.collect()
    n_merged_verts = len(merged_mesh.vertices)
    n_merged_tris = len(merged_mesh.triangles)
    print(f"       Merged: {n_merged_verts:,} vertices, {n_merged_tris:,} triangles")
    stage_times['10_merge'] = time.time() - t_stage

    # Clean up temp tile files
    for tile_ply in tile_ply_paths:
        tile_ply.unlink(missing_ok=True)
    if tiles_dir.exists():
        try:
            tiles_dir.rmdir()
        except OSError:
            pass

    # -- Stage 11: Save full-resolution vertex-colored PLY --------------------
    t_stage = time.time()
    progress(11, 17, "Saving vertex-colored PLY")
    ply_path = out_dir / f"{pc_name}_vertex_colored.ply"
    print(f"[11/17] Saving full-res vertex-colored PLY: {ply_path.name}")
    export_vertex_color_ply(merged_mesh, ply_path)
    n_full_tris = len(merged_mesh.triangles)
    stage_times['11_save_ply'] = time.time() - t_stage

    # -- Stage 12: Decimate for UV/bake (full-res preserved in PLY) -----------
    t_stage = time.time()
    progress(12, 17, "Decimating for textured GLB")
    if n_full_tris > glb_target_triangles:
        print(f"[12/17] Decimating {n_full_tris:,} -> {glb_target_triangles:,} triangles "
              f"for textured GLB ...")
        merged_mesh = merged_mesh.simplify_quadric_decimation(
            target_number_of_triangles=glb_target_triangles
        )
        merged_mesh.compute_vertex_normals()
        n_dec_verts = len(merged_mesh.vertices)
        n_dec_tris = len(merged_mesh.triangles)
        print(f"       {n_dec_verts:,} vertices, {n_dec_tris:,} triangles")
    else:
        print(f"[12/17] No decimation needed ({n_full_tris:,} triangles <= target)")
        n_dec_tris = n_full_tris
    stage_times['12_decimate'] = time.time() - t_stage

    # -- Stage 13: Extract vertex colors for texture baking -------------------
    t_stage = time.time()
    progress(13, 17, "Extracting vertex colors")
    print(f"[13/17] Extracting vertex colors from decimated mesh ...")
    if not merged_mesh.has_vertex_colors():
        print("       WARNING: Mesh has no vertex colors. Skipping texture baking.")
        print("       Vertex-colored PLY was already saved. No GLB produced.")
        return

    vertices = np.asarray(merged_mesh.vertices)
    normals_arr = np.asarray(merged_mesh.vertex_normals)
    vertex_colors_arr = np.asarray(merged_mesh.vertex_colors)
    faces = np.asarray(merged_mesh.triangles)
    stage_times['13_extract_colors'] = time.time() - t_stage

    # -- Stage 14: UV unwrap --------------------------------------------------
    t_stage = time.time()
    progress(14, 17, "UV unwrapping with xatlas")
    print(f"[14/17] UV unwrapping with xatlas (resolution={atlas_resolution}) ...")

    try:
        vmapping, new_faces, uv_coords = uv_unwrap_mesh(
            vertices, faces, atlas_resolution=atlas_resolution
        )
    except Exception as e:
        print(f"       ERROR: xatlas failed: {e}")
        print("       Vertex-colored PLY was already saved. No GLB produced.")
        del merged_mesh
        return

    new_vertices = vertices[vmapping]
    new_normals = normals_arr[vmapping]
    new_colors = vertex_colors_arr[vmapping]
    print(f"       {len(vmapping):,} vertices, {len(new_faces):,} faces")
    stage_times['14_uv_unwrap'] = time.time() - t_stage

    # Free Open3D mesh -- we have numpy arrays now
    del merged_mesh
    gc.collect()

    # -- Stage 15: Bake texture atlas (parallel) ------------------------------
    t_stage = time.time()
    progress(15, 17, "Baking texture atlas (parallel)")
    print(f"[15/17] Baking texture atlas ({atlas_resolution}x{atlas_resolution}, "
          f"{MAX_BAKE_WORKERS} workers) ...")
    atlas = bake_texture_atlas(
        new_faces, uv_coords, new_colors,
        atlas_resolution=atlas_resolution,
        max_workers=MAX_BAKE_WORKERS,
    )
    gc.collect()
    stage_times['15_bake'] = time.time() - t_stage
    print(f"       Done in {stage_times['15_bake']:.1f}s")

    # -- Stage 16: Dilate empty texels ----------------------------------------
    t_stage = time.time()
    progress(16, 17, "Dilating empty texels")
    print(f"[16/17] Dilating empty texels ...")
    atlas = dilate_texture(atlas, iterations=8)
    stage_times['16_dilate'] = time.time() - t_stage

    # -- Stage 17: Export GLB -------------------------------------------------
    t_stage = time.time()
    progress(17, 17, "Exporting GLB with metadata")
    glb_path = out_dir / f"{pc_name}.glb"
    print(f"[17/17] Exporting GLB: {glb_path.name}")
    export_textured_glb(
        new_vertices, new_faces, uv_coords, new_normals,
        [atlas], glb_path, mesh_name="tls_mesh"
    )
    glb_size_mb = glb_path.stat().st_size / (1024 * 1024)
    print(f"       GLB size: {glb_size_mb:.1f} MB")
    stage_times['17_export'] = time.time() - t_stage

    # -- Metadata + summary ---------------------------------------------------
    total_time = time.time() - t_start
    bbox_min_m = new_vertices.min(axis=0).tolist()
    bbox_max_m = new_vertices.max(axis=0).tolist()

    metadata = {
        "source_point_cloud": str(input_path),
        "n_input_points": n_input_points,
        "n_downsampled_points": int(n_ds_points),
        "voxel_size_mm": voxel_size * 1000,
        "n_tiles": n_tiles,
        "n_tiles_skipped": n_skipped,
        "tile_size_m": TILE_SIZE,
        "poisson_depth": poisson_depth,
        "n_full_res_triangles": int(n_full_tris),
        "glb_target_triangles": glb_target_triangles,
        "n_vertices": int(len(new_vertices)),
        "n_triangles": int(len(new_faces)),
        "bbox_min_m": bbox_min_m,
        "bbox_max_m": bbox_max_m,
        "atlas_resolution": atlas_resolution,
        "bake_method": "vertex_color_interpolation",
        "glb_size_mb": round(glb_size_mb, 2),
        "draco_compressed": False,
        "unit": "meter",
        "quality_tier": quality_tier,
        "reconstruction_time_s": round(total_time, 1),
        "max_tile_workers": MAX_TILE_WORKERS,
        "max_bake_workers": MAX_BAKE_WORKERS,
    }

    json_path = out_dir / f"{pc_name}_metadata.json"
    with open(str(json_path), "w") as f:
        json.dump(metadata, f, indent=2)

    bbox_extent = [bbox_max_m[i] - bbox_min_m[i] for i in range(3)]
    print(f"\n{'='*60}")
    print(f"Mesh reconstruction complete in {total_time:.1f}s")
    print(f"  Input:       {n_input_points:,} points -> {n_ds_points:,} after downsample")
    print(f"  Tiles:       {n_tiles} total, {n_skipped} skipped")
    print(f"  Workers:     {MAX_TILE_WORKERS} tile, {MAX_BAKE_WORKERS} bake")
    print(f"  Output:      {len(new_vertices):,} vertices, {len(new_faces):,} triangles")
    print(f"  Extent:      {bbox_extent[0]:.2f} x {bbox_extent[1]:.2f} x {bbox_extent[2]:.2f} m")
    print(f"  GLB file:    {glb_path}")
    print(f"  GLB size:    {glb_size_mb:.1f} MB")
    print(f"  PLY file:    {ply_path}")
    print(f"  Atlas:       {atlas_resolution}x{atlas_resolution} px")
    print(f"  Metadata:    {json_path}")
    print(f"  Unit:        1 unit = 1 meter (no rescaling applied)")
    print(f"\n  Timing breakdown:")
    for name, elapsed in stage_times.items():
        print(f"    {name:25s} {elapsed:7.1f}s")
    print(f"    {'TOTAL':25s} {total_time:7.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
