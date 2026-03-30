"""Test wrapper for mvs-texturing: mesh a colorless PLY and texture with panos.

Pipeline:
  1. Load uncolored PLY, voxel downsample
  2. Tiled parallel Poisson reconstruction (6x6m tiles, 1m overlap)
  3. Merge tiles + decimate to target triangle count
  4. Export mesh as PLY for texrecon
  5. Load poses, convert panos to cubemap faces
  6. Write .cam files (MVE format) per cubemap face
  7. Run texrecon CLI
  8. Convert OBJ output to GLB via trimesh
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

# ── Path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SRC_ROOT.parent

sys.path.insert(0, str(_SRC_ROOT / "meshing"))
from cubemap_utils import FACE_ROTATIONS, equirect_to_cubemap_faces
from mesh_utils import (compute_tile_grid, merge_tile_meshes,
                        process_tiles_parallel)

# ── Config ──────────────────────────────────────────────────────────────────
TEXRECON_BIN = _PROJECT_ROOT / "mvs-texturing" / "build" / "apps" / "texrecon" / "texrecon"
POINT_CLOUD_NAME = "tmb_office_one_corridor_bigger_noRGB"
PANO_NAMES = ["TMB_corridor_south1", "TMB_corridor_south2", "TMB_office1"]
POSE_JSON = _PROJECT_ROOT / "data" / "pose_estimates" / "multiroom" / "local_filter_results.json"
PANO_DIR = _PROJECT_ROOT / "data" / "pano" / "raw"
PLY_PATH = _PROJECT_ROOT / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "mesh" / POINT_CLOUD_NAME

# Reconstruction params (balanced tier)
VOXEL_SIZE = 0.010          # 10mm voxel downsample
POISSON_DEPTH = 8
DENSITY_QUANTILE = 0.06
TILE_SIZE = 6.0             # meters
TILE_OVERLAP = 1.0          # meters
TARGET_TRIS = 500_000
CUBEMAP_FACE_SIZE = 1024


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Stage 1-3: Tiled mesh reconstruction ───────────────────────────────────

def reconstruct_mesh():
    """Load PLY, downsample, tiled Poisson reconstruct, merge, decimate."""
    log("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(PLY_PATH))
    log(f"  {len(pcd.points)} points")

    log(f"Voxel downsampling ({VOXEL_SIZE}m)...")
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    log(f"  {len(pcd.points)} points after downsample")

    # Compute tile grid
    bbox_min = np.asarray(pcd.get_min_bound())
    bbox_max = np.asarray(pcd.get_max_bound())
    tiles = compute_tile_grid(bbox_min, bbox_max,
                              tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
    log(f"Tile grid: {len(tiles)} tiles ({TILE_SIZE}m, {TILE_OVERLAP}m overlap)")

    # Parallel tiled Poisson reconstruction
    tiles_dir = OUTPUT_DIR / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    log(f"Tiled Poisson reconstruction (depth={POISSON_DEPTH})...")
    tile_ply_paths, n_skipped, tile_results = process_tiles_parallel(
        pcd, tiles, str(tiles_dir),
        normal_knn=50, normal_radius=0.15,
        poisson_depth=POISSON_DEPTH,
        density_trim_quantile=DENSITY_QUANTILE,
    )
    del pcd

    total_verts = sum(r[2] for r in tile_results if r[1] is not None)
    total_tris = sum(r[3] for r in tile_results if r[1] is not None)
    log(f"  {len(tile_ply_paths)} tiles produced, {n_skipped} skipped")
    log(f"  Total: {total_verts:,} vertices, {total_tris:,} triangles")

    # Merge tiles
    log("Merging tile meshes...")
    tile_meshes = [o3d.io.read_triangle_mesh(str(p)) for p in tile_ply_paths]
    merged = merge_tile_meshes(tile_meshes)
    del tile_meshes
    log(f"  Merged: {len(merged.vertices):,} vertices, {len(merged.triangles):,} triangles")

    # Decimate
    n_tris = len(merged.triangles)
    if n_tris > TARGET_TRIS:
        log(f"Decimating {n_tris:,} → {TARGET_TRIS:,} triangles...")
        merged = merged.simplify_quadric_decimation(
            target_number_of_triangles=TARGET_TRIS)
        log(f"  {len(merged.vertices):,} vertices, {len(merged.triangles):,} triangles")

    # Recompute normals after merge+decimate
    merged.compute_vertex_normals()
    return merged


# ── Stage 4: Export mesh PLY ────────────────────────────────────────────────

def export_mesh_ply(mesh, out_path):
    """Export mesh as PLY (vertices + faces only, no colors)."""
    log(f"Exporting mesh to {out_path.name}")
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=False)


# ── Stage 5-6: Prepare scene folder for texrecon ───────────────────────────

def write_cam_file(cam_path, R_wc, t_wc, focal_normalized):
    """Write a single .cam file in MVE format.

    Args:
        cam_path: output .cam file path
        R_wc: (3,3) world-to-camera rotation
        t_wc: (3,) world-to-camera translation (NOT camera center)
        focal_normalized: focal length normalized by max(width, height)
    """
    # Line 1: tx ty tz R00 R01 R02 R10 R11 R12 R20 R21 R22
    line1_parts = list(t_wc) + list(R_wc.flatten())
    line1 = " ".join(f"{v:.12f}" for v in line1_parts)
    # Line 2: f d0 d1 paspect ppx ppy
    line2 = f"{focal_normalized:.12f} 0 0 1 0.5 0.5"

    with open(cam_path, "w") as f:
        f.write(line1 + "\n")
        f.write(line2 + "\n")


def prepare_scene_folder(scene_dir):
    """Convert panos to cubemap faces and write .cam files.

    Returns:
        Number of view images created.
    """
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True)

    # Load poses
    with open(POSE_JSON) as f:
        poses = json.load(f)

    # Focal length for 90-deg FOV cubemap face, normalized by image size
    # f_pixels = face_size / 2, f_normalized = f_pixels / face_size = 0.5
    focal_normalized = 0.5

    view_idx = 0
    for pano_name in PANO_NAMES:
        if pano_name not in poses:
            log(f"  WARNING: {pano_name} not in pose JSON, skipping")
            continue

        pose = poses[pano_name]
        R_pano = np.array(pose["R"], dtype=np.float64)   # world-to-camera rotation
        t_pano = np.array(pose["t"], dtype=np.float64)    # camera center in world frame

        # Load panorama
        pano_path = PANO_DIR / f"{pano_name}.jpg"
        log(f"  Loading {pano_path.name}...")
        pano_img = cv2.imread(str(pano_path))
        if pano_img is None:
            log(f"  ERROR: cannot read {pano_path}")
            continue

        # Convert to cubemap faces
        faces = equirect_to_cubemap_faces(pano_img, face_size=CUBEMAP_FACE_SIZE)

        for face_idx, (R_face_to_cam, face_img) in enumerate(zip(FACE_ROTATIONS, faces)):
            # Combined rotation: world → pano-cam → face-local
            R_ext = R_face_to_cam.T @ R_pano          # world-to-face rotation
            t_ext = -R_ext @ t_pano                     # world-to-face translation

            # Write image + cam file with matching names
            view_name = f"view_{view_idx:04d}"
            img_path = scene_dir / f"{view_name}.jpg"
            cam_path = scene_dir / f"{view_name}.cam"

            cv2.imwrite(str(img_path), face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_cam_file(cam_path, R_ext, t_ext, focal_normalized)

            view_idx += 1

    log(f"  Created {view_idx} view images + cam files")
    return view_idx


# ── Stage 7: Run texrecon ──────────────────────────────────────────────────

def run_texrecon(scene_dir, mesh_ply, out_prefix):
    """Run the texrecon CLI."""
    cmd = [
        str(TEXRECON_BIN),
        str(scene_dir),
        str(mesh_ply),
        str(out_prefix),
        "--skip_geometric_visibility_test",
        "--keep_unseen_faces",
        "--data_term=area",
    ]
    log(f"Running texrecon...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            log(f"  [texrecon] {line}")
    if result.returncode != 0:
        log(f"  texrecon FAILED (exit code {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                log(f"  [stderr] {line}")
        return False
    return True


# ── Stage 8: Convert OBJ to GLB ────────────────────────────────────────────

def convert_obj_to_glb(obj_path, glb_path):
    """Convert textured OBJ to GLB using trimesh."""
    log(f"Converting {obj_path.name} → {glb_path.name}")
    import trimesh
    scene = trimesh.load(str(obj_path), process=False)
    scene.export(str(glb_path), file_type="glb")
    log(f"  GLB written: {glb_path} ({glb_path.stat().st_size / 1024 / 1024:.1f} MB)")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1-3: Tiled Poisson reconstruction
    mesh = reconstruct_mesh()

    # Stage 4: Export as PLY
    mesh_ply = OUTPUT_DIR / "texrecon_input_mesh.ply"
    export_mesh_ply(mesh, mesh_ply)
    del mesh

    # Stage 5-6: Prepare scene folder (cubemap faces + cam files)
    scene_dir = OUTPUT_DIR / "texrecon_scene"
    log("Preparing scene folder (cubemap faces + .cam files)...")
    n_views = prepare_scene_folder(scene_dir)
    if n_views == 0:
        log("ERROR: No views created, aborting")
        return

    # Stage 7: Run texrecon
    out_prefix = OUTPUT_DIR / "texrecon_output"
    success = run_texrecon(scene_dir, mesh_ply, out_prefix)
    if not success:
        log("texrecon failed, aborting")
        return

    # Stage 8: Convert to GLB
    obj_path = OUTPUT_DIR / "texrecon_output.obj"
    if obj_path.exists():
        glb_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}_texrecon.glb"
        convert_obj_to_glb(obj_path, glb_path)
    else:
        log(f"WARNING: {obj_path} not found")
        for p in OUTPUT_DIR.glob("texrecon_output*"):
            log(f"  Found: {p.name}")

    elapsed = time.time() - t_start
    log(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
