"""Image-projection mesh reconstruction from uncolored TLS point clouds.

Takes an uncolored PLY point cloud, equirectangular panoramic images, and
estimated camera poses, then produces a UV-textured GLB mesh by projecting
panorama pixels directly onto the mesh surface.

Uses Open3D v0.19's project_images_to_albedo() with Embree raycasting for
visibility and softmax multi-view blending. Panoramas are converted to
cubemap faces (6 pinhole images each) since the API only supports pinhole
camera models.

Pipeline:
1.   Load config + validate inputs
2.   Load point cloud + voxel downsample
3.   Compute tile grid (6x6 m XY tiles, 1 m overlap)
4-9. Per-tile Poisson reconstruction (parallel)
10.  Merge tile meshes
11.  Decimate for textured GLB
12.  Convert to tensor mesh + manifold repair + UV atlas
13.  Load poses + cubemap conversion + project images + export GLB

The TLS point cloud has 1 unit = 1 meter. No rescaling is applied.

Usage:
    conda run -n scan_env python src/meshing/mesh_from_images.py
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import cv2
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
)
from cubemap_utils import equirect_to_cubemap_faces, build_cubemap_cameras
from export_gltf import _inject_gltf_metadata

# -- Config -------------------------------------------------------------------

POINT_CLOUD_NAME = "tmb_office_one_corridor_bigger_noRGB"
PANO_NAMES = ["TMB_office1", "TMB_corridor_south1", "TMB_corridor_south2"]
QUALITY_TIER = "balanced"

_QUALITY_PRESETS = {
    "preview": {
        "poisson_depth": 7,
        "voxel_size": 0.015,
        "normal_knn": 15,
        "glb_target_triangles": 250_000,
        "tex_size": 2048,
        "cubemap_face_size": 512,
    },
    "balanced": {
        "poisson_depth": 8,
        "voxel_size": 0.010,
        "normal_knn": 20,
        "glb_target_triangles": 500_000,
        "tex_size": 4096,
        "cubemap_face_size": 1024,
    },
    "high": {
        "poisson_depth": 9,
        "voxel_size": 0.005,
        "normal_knn": 30,
        "glb_target_triangles": 500_000,
        "tex_size": 4096,
        "cubemap_face_size": 1024,
    },
}

DENSITY_TRIM_QUANTILE = 0.06
NORMAL_RADIUS = 0.15
TILE_SIZE = 6.0
OVERLAP = 1.0
MIN_TILE_POINTS = 1000

MAX_TILE_WORKERS = min(6, max(1, os.cpu_count() // 8))

ROOT = Path(__file__).resolve().parent.parent.parent
PC_PATH = ROOT / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
PANO_DIR = ROOT / "data" / "pano" / "raw"
POSE_PATH = ROOT / "data" / "pose_estimates" / "multiroom" / "local_filter_results.json"
OUTPUT_DIR = ROOT / "data" / "mesh" / POINT_CLOUD_NAME

N_STAGES = 13


# -- Main ---------------------------------------------------------------------

def main():
    cfg = load_config()
    pc_name = cfg.get("point_cloud_name", POINT_CLOUD_NAME)
    pano_names = cfg.get("pano_names", PANO_NAMES)
    quality_tier = cfg.get("quality_tier", QUALITY_TIER)

    preset = _QUALITY_PRESETS[quality_tier]
    poisson_depth = preset["poisson_depth"]
    voxel_size = preset["voxel_size"]
    normal_knn = preset["normal_knn"]
    glb_target_triangles = preset["glb_target_triangles"]
    tex_size = preset["tex_size"]
    cubemap_face_size = preset["cubemap_face_size"]

    pc_path = Path(cfg["pc_path"]) if cfg.get("pc_path") else PC_PATH
    pano_dir = Path(cfg["pano_dir"]) if cfg.get("pano_dir") else PANO_DIR
    pose_path = Path(cfg["pose_path"]) if cfg.get("pose_path") else POSE_PATH
    out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else OUTPUT_DIR

    t_start = time.time()
    stage_times = {}

    # -- Stage 1: Validate inputs ---------------------------------------------
    progress(1, N_STAGES, "Validating inputs")
    t_stage = time.time()
    print(f"[1/{N_STAGES}] Validating inputs ...")
    print(f"  Quality tier: {quality_tier} (depth={poisson_depth}, "
          f"voxel={voxel_size*1000:.0f}mm, tex={tex_size}, "
          f"cubemap={cubemap_face_size})")

    assert pc_path.exists(), f"Point cloud not found: {pc_path}"
    assert pose_path.exists(), f"Pose file not found: {pose_path}"
    for name in pano_names:
        pano_file = pano_dir / f"{name}.jpg"
        assert pano_file.exists(), f"Panorama not found: {pano_file}"
    print(f"  Point cloud: {pc_path.name}")
    print(f"  Panoramas:   {len(pano_names)} ({', '.join(pano_names)})")
    print(f"  Poses:       {pose_path.name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    stage_times['01_validate'] = time.time() - t_stage

    # -- Stage 2: Load point cloud + voxel downsample -------------------------
    progress(2, N_STAGES, "Loading and downsampling point cloud")
    t_stage = time.time()
    print(f"[2/{N_STAGES}] Loading point cloud: {pc_path.name}")
    pcd = o3d.io.read_point_cloud(str(pc_path))
    n_input_points = len(pcd.points)
    points_arr = np.asarray(pcd.points)
    bbox_min_pcd = points_arr.min(axis=0)
    bbox_max_pcd = points_arr.max(axis=0)
    extent = bbox_max_pcd - bbox_min_pcd
    print(f"  {n_input_points:,} points, extent: "
          f"{extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")

    print(f"  Voxel downsample ({voxel_size*1000:.0f} mm) ...")
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    n_ds_points = len(pcd_ds.points)
    del pcd
    gc.collect()
    print(f"  {n_input_points:,} -> {n_ds_points:,} points "
          f"({100*n_ds_points/n_input_points:.0f}%)")
    stage_times['02_load_downsample'] = time.time() - t_stage

    # -- Stage 3: Compute tile grid -------------------------------------------
    progress(3, N_STAGES, "Computing tile grid")
    t_stage = time.time()
    ds_points = np.asarray(pcd_ds.points)
    bbox_min_ds = ds_points.min(axis=0)
    bbox_max_ds = ds_points.max(axis=0)
    print(f"[3/{N_STAGES}] Computing tile grid (tile={TILE_SIZE}m, "
          f"overlap={OVERLAP}m)")
    tiles = compute_tile_grid(bbox_min_ds, bbox_max_ds,
                              tile_size=TILE_SIZE, overlap=OVERLAP)
    n_tiles = len(tiles)
    print(f"  {n_tiles} tiles")
    stage_times['03_tile_grid'] = time.time() - t_stage

    # -- Stages 4-9: Per-tile Poisson reconstruction (parallel) ---------------
    progress(4, N_STAGES, "Per-tile Poisson reconstruction")
    t_stage = time.time()
    print(f"[4-9/{N_STAGES}] Per-tile Poisson reconstruction "
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
    print(f"  Tiles processed: {len(tile_ply_paths)}, skipped: {n_skipped}")
    stage_times['04-09_tiles'] = time.time() - t_stage

    del pcd_ds
    gc.collect()

    # -- Stage 10: Merge tile meshes ------------------------------------------
    progress(10, N_STAGES, "Merging tile meshes")
    t_stage = time.time()
    print(f"[10/{N_STAGES}] Merging {len(tile_ply_paths)} tile meshes ...")
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
    print(f"  Merged: {n_merged_verts:,} vertices, {n_merged_tris:,} triangles")
    stage_times['10_merge'] = time.time() - t_stage

    # Clean up temp tile files
    for tile_ply in tile_ply_paths:
        tile_ply.unlink(missing_ok=True)
    if tiles_dir.exists():
        try:
            tiles_dir.rmdir()
        except OSError:
            pass

    # -- Stage 11: Decimate ---------------------------------------------------
    progress(11, N_STAGES, "Decimating mesh")
    t_stage = time.time()
    n_full_tris = n_merged_tris
    if n_merged_tris > glb_target_triangles:
        print(f"[11/{N_STAGES}] Decimating {n_merged_tris:,} -> "
              f"{glb_target_triangles:,} triangles ...")
        merged_mesh = merged_mesh.simplify_quadric_decimation(
            target_number_of_triangles=glb_target_triangles
        )
        merged_mesh.compute_vertex_normals()
        n_dec_tris = len(merged_mesh.triangles)
        print(f"  {len(merged_mesh.vertices):,} vertices, {n_dec_tris:,} triangles")
    else:
        print(f"[11/{N_STAGES}] No decimation needed "
              f"({n_merged_tris:,} <= {glb_target_triangles:,})")
        n_dec_tris = n_merged_tris
    stage_times['11_decimate'] = time.time() - t_stage

    # -- Stage 12: Tensor mesh + manifold repair + UV atlas -------------------
    progress(12, N_STAGES, "Computing UV atlas")
    t_stage = time.time()
    print(f"[12/{N_STAGES}] Converting to tensor mesh + UV atlas "
          f"(tex_size={tex_size}) ...")

    # Manifold repair on legacy mesh (compute_uvatlas requires manifold input)
    n_before = len(merged_mesh.triangles)
    merged_mesh.remove_degenerate_triangles()
    merged_mesh.remove_duplicated_triangles()
    merged_mesh.remove_duplicated_vertices()
    merged_mesh.remove_non_manifold_edges()
    # Iteratively remove non-manifold vertices until clean
    for _ in range(5):
        non_manifold = merged_mesh.get_non_manifold_vertices()
        if len(non_manifold) == 0:
            break
        merged_mesh.remove_vertices_by_index(non_manifold)
    merged_mesh.remove_unreferenced_vertices()
    n_after = len(merged_mesh.triangles)
    if n_before != n_after:
        print(f"  Manifold repair: {n_before:,} -> {n_after:,} triangles "
              f"(removed {n_before - n_after})")
    is_manifold = merged_mesh.is_edge_manifold() and merged_mesh.is_vertex_manifold()
    print(f"  Edge-manifold: {merged_mesh.is_edge_manifold()}, "
          f"Vertex-manifold: {merged_mesh.is_vertex_manifold()}")

    # Convert legacy mesh to tensor mesh
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(merged_mesh)
    del merged_mesh
    gc.collect()

    # Compute UV atlas — try Open3D's built-in first, fall back to xatlas
    print(f"  Computing UV atlas ...")
    try:
        t_mesh.compute_uvatlas(size=tex_size, gutter=2.0, max_stretch=0.167,
                               parallel_partitions=1, nthreads=0)
    except RuntimeError as e:
        if "Non-manifold" in str(e):
            print(f"  compute_uvatlas failed ({e}), falling back to xatlas ...")
            from mesh_utils import uv_unwrap_mesh
            verts_np = t_mesh.vertex.positions.numpy()
            tris_np = t_mesh.triangle.indices.numpy()
            vmapping, new_faces, uv_coords = uv_unwrap_mesh(
                verts_np, tris_np, atlas_resolution=tex_size
            )
            # Build per-triangle-corner UVs: (F, 3, 2)
            tri_uvs = uv_coords[new_faces]  # (F, 3, 2)
            # Rebuild tensor mesh with remapped vertices
            new_verts = verts_np[vmapping]
            t_mesh = o3d.t.geometry.TriangleMesh()
            t_mesh.vertex.positions = o3d.core.Tensor(new_verts, dtype=o3d.core.float32)
            t_mesh.triangle.indices = o3d.core.Tensor(new_faces, dtype=o3d.core.int64)
            t_mesh.triangle.texture_uvs = o3d.core.Tensor(
                tri_uvs.astype(np.float32).reshape(-1, 2),
                dtype=o3d.core.float32
            ).reshape((-1, 3, 2))
        else:
            raise
    print(f"  UV atlas computed")

    # Recompute normals (UV atlas may split vertices, leaving new ones without normals)
    t_mesh.compute_vertex_normals()
    print(f"  Vertex normals recomputed")
    stage_times['12_uv_atlas'] = time.time() - t_stage

    # -- Stage 13: Image projection + GLB export ------------------------------
    progress(13, N_STAGES, "Projecting panoramas onto mesh")
    t_stage = time.time()
    print(f"[13/{N_STAGES}] Loading poses and projecting panoramas ...")

    # Load poses
    with open(str(pose_path)) as f:
        pose_data = json.load(f)

    all_images = []
    all_intrinsics = []
    all_extrinsics = []

    for pano_name in pano_names:
        entry = pose_data[pano_name]
        R_pano = np.array(entry["R"], dtype=np.float64)
        t_pano = np.array(entry["t"], dtype=np.float64)

        # Load panorama and convert to cubemap faces
        pano_path = pano_dir / f"{pano_name}.jpg"
        pano_bgr = cv2.imread(str(pano_path))
        pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
        del pano_bgr

        faces_list = equirect_to_cubemap_faces(pano_rgb, face_size=cubemap_face_size)
        del pano_rgb

        intrinsics, extrinsics = build_cubemap_cameras(
            R_pano, t_pano, face_size=cubemap_face_size
        )

        for face_img, K, ext in zip(faces_list, intrinsics, extrinsics):
            # Convert BGR cubemap face to RGB (cv2.remap preserves channel order,
            # but we loaded as BGR and converted to RGB before equirect_to_cubemap)
            all_images.append(o3d.t.geometry.Image(face_img))
            all_intrinsics.append(o3d.core.Tensor(K))
            all_extrinsics.append(o3d.core.Tensor(ext))

        print(f"  {pano_name}: 6 cubemap faces ({cubemap_face_size}x"
              f"{cubemap_face_size})")

    n_proj_images = len(all_images)
    print(f"  Projecting {n_proj_images} images onto mesh "
          f"(tex_size={tex_size}) ...")

    # Project images to albedo texture
    t_mesh.project_images_to_albedo(
        images=all_images,
        intrinsic_matrices=all_intrinsics,
        extrinsic_matrices=all_extrinsics,
        tex_size=tex_size,
        update_material=True,
    )
    del all_images, all_intrinsics, all_extrinsics
    gc.collect()
    print(f"  Projection complete")

    # Exposure correction: project_images_to_albedo() produces dim output due
    # to foreshortening weighting. Scale brightness to match direct panoramic
    # sampling (target: p75 of non-zero texels ≈ 140).
    albedo_key = "albedo"
    if albedo_key in t_mesh.material.texture_maps:
        albedo_img = t_mesh.material.texture_maps[albedo_key]
        albedo_np = albedo_img.numpy().astype(np.float32) if hasattr(albedo_img, 'numpy') else np.asarray(albedo_img).astype(np.float32)
        mask = albedo_np.sum(axis=2) > 0
        if mask.any():
            nz_brightness = albedo_np[mask].mean(axis=1)
            p75 = np.percentile(nz_brightness, 75)
            if p75 > 0:
                scale = 140.0 / p75
                albedo_np[mask] = np.clip(albedo_np[mask] * scale, 0, 255)
                print(f"  Exposure correction: {scale:.2f}x (p75 {p75:.0f} -> 140)")
        t_mesh.material.texture_maps[albedo_key] = o3d.t.geometry.Image(
            albedo_np.astype(np.uint8)
        )

    # Export GLB via Open3D's native writer
    glb_path = out_dir / f"{pc_name}.glb"
    print(f"  Exporting GLB: {glb_path.name}")
    o3d.t.io.write_triangle_mesh(str(glb_path), t_mesh)

    # Fix material and inject metadata: Open3D sets emissiveFactor=[1,1,1]
    # which makes the mesh glow white in viewers
    import struct as _struct
    with open(str(glb_path), "rb") as f:
        glb_bytes = bytearray(f.read())
    magic = _struct.unpack_from("<I", glb_bytes, 0)[0]
    if magic == 0x46546C67:
        json_len = _struct.unpack_from("<I", glb_bytes, 12)[0]
        gltf = json.loads(glb_bytes[20:20+json_len].decode("utf-8").rstrip("\x00"))
        # Fix emissive and set proper PBR
        for mat in gltf.get("materials", []):
            mat["emissiveFactor"] = [0.0, 0.0, 0.0]
            if "pbrMetallicRoughness" in mat:
                mat["pbrMetallicRoughness"]["metallicFactor"] = 0.0
                mat["pbrMetallicRoughness"]["roughnessFactor"] = 1.0
        # Inject metadata
        if "asset" not in gltf:
            gltf["asset"] = {"version": "2.0"}
        gltf["asset"]["extras"] = {
            "unit": "meter",
            "scale": 1.0,
            "source": "TLS_point_cloud",
            "texture_method": "image_projection",
            "note": "1 unit = 1 meter. No rescaling applied. Metric-accurate.",
        }
        new_json = json.dumps(gltf, separators=(",", ":"))
        while len(new_json) % 4 != 0:
            new_json += " "
        new_json_bytes = new_json.encode("utf-8")
        remaining = glb_bytes[20 + json_len:]
        new_total = 12 + 8 + len(new_json_bytes) + len(remaining)
        result = bytearray()
        result += _struct.pack("<III", magic, 2, new_total)
        result += _struct.pack("<II", len(new_json_bytes), 0x4E4F534A)
        result += new_json_bytes
        result += remaining
        glb_bytes = bytes(result)
    with open(str(glb_path), "wb") as f:
        f.write(glb_bytes)

    glb_size_mb = glb_path.stat().st_size / (1024 * 1024)
    print(f"  GLB size: {glb_size_mb:.1f} MB")
    stage_times['13_project_export'] = time.time() - t_stage

    # Extract final counts from tensor mesh
    n_final_verts = int(t_mesh.vertex.positions.shape[0])
    n_final_tris = int(t_mesh.triangle.indices.shape[0])
    vertices_np = t_mesh.vertex.positions.numpy()
    bbox_min_m = vertices_np.min(axis=0).tolist()
    bbox_max_m = vertices_np.max(axis=0).tolist()
    del t_mesh
    gc.collect()

    # -- Metadata + summary ---------------------------------------------------
    total_time = time.time() - t_start

    metadata = {
        "source_point_cloud": str(pc_path),
        "n_input_points": n_input_points,
        "n_downsampled_points": int(n_ds_points),
        "voxel_size_mm": voxel_size * 1000,
        "n_tiles": n_tiles,
        "n_tiles_skipped": n_skipped,
        "tile_size_m": TILE_SIZE,
        "poisson_depth": poisson_depth,
        "n_full_res_triangles": int(n_full_tris),
        "glb_target_triangles": glb_target_triangles,
        "n_vertices": n_final_verts,
        "n_triangles": n_final_tris,
        "bbox_min_m": bbox_min_m,
        "bbox_max_m": bbox_max_m,
        "tex_size": tex_size,
        "cubemap_face_size": cubemap_face_size,
        "texture_method": "image_projection",
        "n_panoramas": len(pano_names),
        "pano_names": pano_names,
        "n_projection_images": n_proj_images,
        "uv_method": "compute_uvatlas",
        "glb_size_mb": round(glb_size_mb, 2),
        "unit": "meter",
        "quality_tier": quality_tier,
        "reconstruction_time_s": round(total_time, 1),
        "max_tile_workers": MAX_TILE_WORKERS,
        "open3d_version": o3d.__version__,
    }

    json_path = out_dir / f"{pc_name}_metadata.json"
    with open(str(json_path), "w") as f:
        json.dump(metadata, f, indent=2)

    bbox_extent = [bbox_max_m[i] - bbox_min_m[i] for i in range(3)]
    print(f"\n{'='*60}")
    print(f"Image-projection mesh reconstruction complete in {total_time:.1f}s")
    print(f"  Input:       {n_input_points:,} points -> {n_ds_points:,} downsampled")
    print(f"  Tiles:       {n_tiles} total, {n_skipped} skipped")
    print(f"  Geometry:    {n_full_tris:,} full-res -> {n_final_tris:,} final triangles")
    print(f"  Extent:      {bbox_extent[0]:.2f} x {bbox_extent[1]:.2f} x "
          f"{bbox_extent[2]:.2f} m")
    print(f"  Texture:     {tex_size}x{tex_size} from {len(pano_names)} panoramas "
          f"({n_proj_images} cubemap images)")
    print(f"  GLB:         {glb_path} ({glb_size_mb:.1f} MB)")
    print(f"  Metadata:    {json_path}")
    print(f"  Unit:        1 unit = 1 meter (no rescaling applied)")
    print(f"\n  Timing breakdown:")
    for name, elapsed in stage_times.items():
        print(f"    {name:25s} {elapsed:7.1f}s")
    print(f"    {'TOTAL':25s} {total_time:7.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
