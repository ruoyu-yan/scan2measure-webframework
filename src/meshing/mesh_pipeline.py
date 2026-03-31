"""Production meshing pipeline: PoissonRecon geometry + texrecon UV texturing.

Pipeline:
  1. Load PLY point cloud, voxel downsample
  2. Estimate + orient normals
  3. Export PLY with normals for PoissonRecon
  4. Run PoissonRecon CLI (octree depth controlled by quality tier)
  5. Run SurfaceTrimmer CLI (density-based trimming)
  6. Load trimmed mesh, decimate to target triangle count, recompute normals
  7. Export decimated mesh PLY for texrecon
  8. Prepare texrecon scene (panos -> cubemap faces + .cam files)
  9. Compute per-face visibility via depth-buffer rendering + write labeling
  10. Run texrecon CLI with --labeling_file (atlas packing + seam leveling)
  11. Convert OBJ to GLB (double-sided material) | fallback: untextured GLB

External binaries required:
  - PoissonRecon/Bin/Linux/PoissonRecon
  - PoissonRecon/Bin/Linux/SurfaceTrimmer
  - mvs-texturing/build/apps/texrecon/texrecon
"""

import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

# -- Path setup ----------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SRC_ROOT.parent

sys.path.insert(0, str(_SRC_ROOT / "meshing"))
sys.path.insert(0, str(_SRC_ROOT / "utils"))

from config_loader import load_config, progress
from cubemap_utils import FACE_ROTATIONS, equirect_to_cubemap_faces
from face_visibility import compute_and_write_labeling

# -- Binary paths --------------------------------------------------------------
POISSONRECON_BIN = _PROJECT_ROOT / "PoissonRecon" / "Bin" / "Linux" / "PoissonRecon"
SURFACETRIMMER_BIN = _PROJECT_ROOT / "PoissonRecon" / "Bin" / "Linux" / "SurfaceTrimmer"
TEXRECON_BIN = _PROJECT_ROOT / "mvs-texturing" / "build" / "apps" / "texrecon" / "texrecon"
SUBPROCESS_TIMEOUT = 600

# -- Hardcoded defaults (used when --config is not provided) -------------------
_DEFAULT_PC_NAME = "tmb_office_one_corridor_bigger_noRGB"
_DEFAULT_PANO_NAMES = ["TMB_corridor_south1", "TMB_corridor_south2", "TMB_office1"]

# -- Quality presets -----------------------------------------------------------
_QUALITY_PRESETS = {
    "preview": {
        "voxel_size": 0.010,
        "normal_radius": 0.15,
        "normal_max_nn": 50,
        "normal_orient_k": 30,
        "poisson_depth": 8,
        "poisson_point_weight": 2,
        "trim_value": 7,
        "trim_aratio": 0.001,
        "target_triangles": 250_000,
        "cubemap_face_size": 512,
    },
    "balanced": {
        "voxel_size": 0.010,
        "normal_radius": 0.15,
        "normal_max_nn": 50,
        "normal_orient_k": 30,
        "poisson_depth": 10,
        "poisson_point_weight": 2,
        "trim_value": 7,
        "trim_aratio": 0.001,
        "target_triangles": 500_000,
        "cubemap_face_size": 1024,
    },
    "high": {
        "voxel_size": 0.010,
        "normal_radius": 0.15,
        "normal_max_nn": 50,
        "normal_orient_k": 30,
        "poisson_depth": 11,
        "poisson_point_weight": 2,
        "trim_value": 7,
        "trim_aratio": 0.001,
        "target_triangles": 1_000_000,
        "cubemap_face_size": 2048,
    },
}

TOTAL_STEPS = 11


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -- Step 1: Load + downsample ------------------------------------------------

def load_and_downsample(ply_path, voxel_size):
    log(f"Loading point cloud: {ply_path.name}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_raw = len(pcd.points)
    log(f"  {n_raw:,} points")

    log(f"Voxel downsampling ({voxel_size * 1000:.0f}mm)...")
    pcd = pcd.voxel_down_sample(voxel_size)
    log(f"  {len(pcd.points):,} points after downsample")
    return pcd, n_raw


# -- Step 2: Estimate normals -------------------------------------------------

def estimate_normals(pcd, radius, max_nn, orient_k):
    log(f"Estimating normals (radius={radius}, max_nn={max_nn})...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    log(f"Orienting normals (tangent plane, k={orient_k})...")
    pcd.orient_normals_consistent_tangent_plane(k=orient_k)
    return pcd


# -- Step 3: Export PLY with normals -------------------------------------------

def export_ply_with_normals(pcd, out_path):
    log(f"Exporting PLY with normals: {out_path.name}")
    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)
    log(f"  Written: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


# -- Step 4: Run PoissonRecon -------------------------------------------------

def run_poissonrecon(input_ply, output_ply, depth, point_weight):
    cmd = [
        str(POISSONRECON_BIN),
        "--in", str(input_ply),
        "--out", str(output_ply),
        "--depth", str(depth),
        "--density",
        "--pointWeight", str(point_weight),
    ]
    log(f"Running PoissonRecon (depth={depth}, pointWeight={point_weight})...")

    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=SUBPROCESS_TIMEOUT)

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            log(f"  [PoissonRecon] {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            log(f"  [PoissonRecon stderr] {line}")

    if result.returncode != 0:
        log(f"  PoissonRecon FAILED (exit code {result.returncode})")
        return None

    # PoissonRecon may auto-append .ply to the output path
    if output_ply.exists():
        log(f"  Output: {output_ply.name} ({output_ply.stat().st_size / 1024 / 1024:.1f} MB)")
        return output_ply

    alt_path = output_ply.with_suffix(output_ply.suffix + ".ply")
    if alt_path.exists():
        alt_path.rename(output_ply)
        log(f"  Output (renamed): {output_ply.name} ({output_ply.stat().st_size / 1024 / 1024:.1f} MB)")
        return output_ply

    log(f"  WARNING: Expected output not found at {output_ply}")
    return None


# -- Step 5: Run SurfaceTrimmer -----------------------------------------------

def run_surfacetrimmer(input_ply, output_ply, trim_value, trim_aratio):
    cmd = [
        str(SURFACETRIMMER_BIN),
        "--in", str(input_ply),
        "--out", str(output_ply),
        "--trim", str(trim_value),
        "--aRatio", str(trim_aratio),
    ]
    log(f"Running SurfaceTrimmer (trim={trim_value}, aRatio={trim_aratio})...")

    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=SUBPROCESS_TIMEOUT)

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            log(f"  [SurfaceTrimmer] {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            log(f"  [SurfaceTrimmer stderr] {line}")

    if result.returncode != 0:
        log(f"  SurfaceTrimmer FAILED (exit code {result.returncode})")
        return None

    if output_ply.exists():
        log(f"  Output: {output_ply.name} ({output_ply.stat().st_size / 1024 / 1024:.1f} MB)")
        return output_ply

    log(f"  WARNING: Expected output not found at {output_ply}")
    return None


# -- Step 6: Load trimmed mesh, decimate, recompute normals --------------------

def load_and_decimate(trimmed_ply, target_triangles):
    log(f"Loading trimmed mesh: {trimmed_ply.name}")
    mesh = o3d.io.read_triangle_mesh(str(trimmed_ply))
    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    log(f"  {n_verts:,} vertices, {n_tris:,} triangles")

    if n_tris > target_triangles:
        log(f"Decimating {n_tris:,} -> {target_triangles:,} triangles...")
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles
        )
        log(f"  {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    log("Recomputing vertex normals...")
    mesh.compute_vertex_normals()
    return mesh


# -- Step 7: Export mesh PLY for texrecon --------------------------------------

def export_mesh_for_texrecon(mesh, out_path):
    log(f"Exporting mesh for texrecon: {out_path.name}")
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=False)
    log(f"  {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")


# -- Step 8: Prepare texrecon scene folder -------------------------------------

def write_cam_file(cam_path, R_wc, t_wc, focal_normalized):
    """Write a single .cam file in MVE format.

    Args:
        cam_path: output .cam file path
        R_wc: (3,3) world-to-camera rotation
        t_wc: (3,) world-to-camera translation (NOT camera center)
        focal_normalized: focal length normalized by max(width, height)
    """
    line1_parts = list(t_wc) + list(R_wc.flatten())
    line1 = " ".join(f"{v:.12f}" for v in line1_parts)
    line2 = f"{focal_normalized:.12f} 0 0 1 0.5 0.5"

    with open(cam_path, "w") as f:
        f.write(line1 + "\n")
        f.write(line2 + "\n")


def prepare_scene_folder(scene_dir, pano_names, pano_dir, pose_json_path,
                         cubemap_face_size):
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True)

    with open(pose_json_path) as f:
        poses = json.load(f)

    focal_normalized = 0.5  # 90-deg FOV cubemap: f_pixels/face_size = 0.5

    view_idx = 0
    for pano_name in pano_names:
        if pano_name not in poses:
            log(f"  WARNING: {pano_name} not in pose JSON, skipping")
            continue

        pose = poses[pano_name]
        R_pano = np.array(pose["R"], dtype=np.float64)
        t_pano = np.array(pose["t"], dtype=np.float64)

        pano_path = Path(pano_dir) / f"{pano_name}.jpg"
        log(f"  Loading {pano_path.name}...")
        pano_img = cv2.imread(str(pano_path))
        if pano_img is None:
            log(f"  ERROR: cannot read {pano_path}")
            continue

        faces = equirect_to_cubemap_faces(pano_img, face_size=cubemap_face_size)

        for face_idx, (R_face_to_cam, face_img) in enumerate(
            zip(FACE_ROTATIONS, faces)
        ):
            R_ext = R_face_to_cam.T @ R_pano
            t_ext = -R_ext @ t_pano

            view_name = f"view_{view_idx:04d}"
            img_path = scene_dir / f"{view_name}.jpg"
            cam_path = scene_dir / f"{view_name}.cam"

            cv2.imwrite(str(img_path), face_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_cam_file(cam_path, R_ext, t_ext, focal_normalized)

            view_idx += 1

    log(f"  Created {view_idx} view images + cam files")
    return view_idx


# -- Step 9: Run texrecon -----------------------------------------------------

def run_texrecon(scene_dir, mesh_ply, out_prefix, labeling_path=None):
    cmd = [
        str(TEXRECON_BIN),
        str(scene_dir),
        str(mesh_ply),
        str(out_prefix),
        "--keep_unseen_faces",
    ]
    if labeling_path and labeling_path.exists():
        cmd.append(f"--labeling_file={labeling_path}")
        cmd.append("--skip_local_seam_leveling")
        log("Running texrecon with pre-computed labeling...")
    else:
        cmd.extend(["--skip_geometric_visibility_test",
                     "--data_term=area",
                     "--outlier_removal=gauss_clamping"])
        log("Running texrecon (fallback: no labeling)...")

    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=SUBPROCESS_TIMEOUT)

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


# -- Post-texrecon: dilate texture atlases to prevent bleeding ----------------

def dilate_texture_atlases(output_dir):
    """Fill magenta/black padding in texture atlases with nearest patch colors.

    texrecon fills gaps between texture patches with bright magenta (255,0,255).
    GPU bilinear filtering samples across patch borders, picking up this magenta
    and creating purple/green artifacts on the mesh. This replaces every padding
    pixel with the color of its nearest valid (non-padding) pixel.
    """
    from pathlib import Path
    from scipy import ndimage

    atlas_paths = sorted(Path(output_dir).glob("texrecon_output_material*_map_Kd.png"))
    if not atlas_paths:
        return

    log(f"Dilating {len(atlas_paths)} texture atlas(es) to prevent bleeding...")
    for atlas_path in atlas_paths:
        img = cv2.imread(str(atlas_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Detect padding: magenta tint (R and B both much higher than G) or pure black
        r = img[:, :, 2].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        b = img[:, :, 0].astype(np.float32)
        is_magenta = ((r + b) / 2 - g) > 75
        is_black = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
        padding_mask = is_magenta | is_black

        n_padding = padding_mask.sum()
        if n_padding == 0:
            continue

        # For each padding pixel, find the nearest non-padding pixel
        # and copy its color (nearest-neighbor fill)
        indices = ndimage.distance_transform_edt(
            padding_mask, return_distances=False, return_indices=True
        )
        result = img[indices[0], indices[1]]

        cv2.imwrite(str(atlas_path), result)
        log(f"  {atlas_path.name}: filled {n_padding:,} padding pixels")


# -- Step 11: Convert to GLB --------------------------------------------------

def convert_to_glb(source_path, glb_path, textured=True):
    import trimesh

    log(f"Converting {source_path.name} -> {glb_path.name}")

    if textured:
        scene = trimesh.load(str(source_path), process=False)
        # Set double-sided on all geometry materials
        geoms = scene.geometry.values() if hasattr(scene, "geometry") else [scene]
        for geom in geoms:
            if hasattr(geom, "visual") and hasattr(geom.visual, "material"):
                geom.visual.material.doubleSided = True
        scene.export(str(glb_path), file_type="glb")
    else:
        mesh = trimesh.load(str(source_path), process=False)
        material = trimesh.visual.material.PBRMaterial(doubleSided=True)
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
        mesh.export(str(glb_path), file_type="glb")

    size_mb = glb_path.stat().st_size / 1024 / 1024
    log(f"  GLB written: {glb_path.name} ({size_mb:.1f} MB)")


# -- Main orchestrator ---------------------------------------------------------

def main():
    t_start = time.time()
    cfg = load_config()

    # Resolve parameters from config with hardcoded defaults
    pc_name = cfg.get("point_cloud_name", _DEFAULT_PC_NAME)
    ply_path = Path(cfg.get("point_cloud_path",
                            _PROJECT_ROOT / "data" / "raw_point_cloud" / f"{pc_name}.ply"))
    pano_names = cfg.get("pano_names", _DEFAULT_PANO_NAMES)
    pano_dir = Path(cfg.get("panorama_dir",
                            _PROJECT_ROOT / "data" / "pano" / "raw"))
    pose_json = Path(cfg.get("pose_json_path",
                             _PROJECT_ROOT / "data" / "pose_estimates" / "multiroom" / "local_filter_results.json"))
    output_dir = Path(cfg.get("output_dir",
                              _PROJECT_ROOT / "data" / "mesh" / pc_name))
    quality_tier = cfg.get("quality_tier", "balanced")

    # Resolve quality preset
    params = _QUALITY_PRESETS.get(quality_tier, _QUALITY_PRESETS["balanced"])
    log(f"Quality tier: {quality_tier}")
    log(f"  poisson_depth={params['poisson_depth']}, "
        f"target_tris={params['target_triangles']:,}, "
        f"cubemap={params['cubemap_face_size']}px")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify binaries
    for name, bin_path in [("PoissonRecon", POISSONRECON_BIN),
                           ("SurfaceTrimmer", SURFACETRIMMER_BIN),
                           ("texrecon", TEXRECON_BIN)]:
        if not bin_path.exists():
            log(f"ERROR: {name} binary not found at {bin_path}")
            sys.exit(1)
    if not ply_path.exists():
        log(f"ERROR: Input point cloud not found at {ply_path}")
        sys.exit(1)

    # -- Step 1 --
    progress(1, TOTAL_STEPS, "Loading and downsampling point cloud")
    pcd, n_raw = load_and_downsample(ply_path, params["voxel_size"])
    n_downsampled = len(pcd.points)

    # -- Step 2 --
    progress(2, TOTAL_STEPS, "Estimating normals")
    pcd = estimate_normals(pcd, params["normal_radius"],
                           params["normal_max_nn"], params["normal_orient_k"])

    # -- Step 3 --
    progress(3, TOTAL_STEPS, "Exporting for PoissonRecon")
    pr_input_ply = output_dir / "poissonrecon_input.ply"
    export_ply_with_normals(pcd, pr_input_ply)
    del pcd
    gc.collect()

    # -- Step 4 --
    progress(4, TOTAL_STEPS, "Running PoissonRecon")
    pr_raw_ply = output_dir / "poissonrecon_raw.ply"
    result_ply = run_poissonrecon(pr_input_ply, pr_raw_ply,
                                 params["poisson_depth"],
                                 params["poisson_point_weight"])
    if result_ply is None:
        log("PoissonRecon failed, aborting")
        sys.exit(1)

    # -- Step 5 --
    progress(5, TOTAL_STEPS, "Running SurfaceTrimmer")
    pr_trimmed_ply = output_dir / "poissonrecon_trimmed.ply"
    trimmed_ply = run_surfacetrimmer(result_ply, pr_trimmed_ply,
                                    params["trim_value"],
                                    params["trim_aratio"])
    if trimmed_ply is None:
        log("SurfaceTrimmer failed, aborting")
        sys.exit(1)

    # -- Step 6 --
    progress(6, TOTAL_STEPS, "Decimating mesh")
    mesh = load_and_decimate(trimmed_ply, params["target_triangles"])
    decimated_verts = len(mesh.vertices)
    decimated_tris = len(mesh.triangles)

    # -- Step 7 --
    progress(7, TOTAL_STEPS, "Exporting mesh for texrecon")
    texrecon_mesh_ply = output_dir / "texrecon_input_mesh.ply"
    export_mesh_for_texrecon(mesh, texrecon_mesh_ply)
    del mesh
    gc.collect()

    # -- Step 8 --
    progress(8, TOTAL_STEPS, "Preparing cubemap scene")
    scene_dir = output_dir / "texrecon_scene"
    log("Preparing scene folder (cubemap faces + .cam files)...")
    n_views = prepare_scene_folder(scene_dir, pano_names, pano_dir,
                                   pose_json, params["cubemap_face_size"])
    if n_views == 0:
        log("ERROR: No views created, aborting")
        sys.exit(1)

    # -- Step 9 --
    progress(9, TOTAL_STEPS, "Computing per-face visibility")
    labeling_path = output_dir / "face_visibility_labeling.vec"
    vis_stats = compute_and_write_labeling(
        texrecon_mesh_ply, pano_names, pose_json,
        params["cubemap_face_size"], labeling_path
    )

    # -- Step 10 --
    progress(10, TOTAL_STEPS, "Running texrecon")
    out_prefix = output_dir / "texrecon_output"
    texrecon_ok = run_texrecon(scene_dir, texrecon_mesh_ply, out_prefix,
                               labeling_path)

    # -- Step 11 --
    progress(11, TOTAL_STEPS, "Converting to GLB")
    textured_glb = output_dir / f"{pc_name}_textured.glb"
    untextured_glb = output_dir / f"{pc_name}_untextured.glb"

    obj_path = output_dir / "texrecon_output.obj"
    if texrecon_ok and obj_path.exists():
        dilate_texture_atlases(output_dir)
        convert_to_glb(obj_path, textured_glb, textured=True)
        final_glb = textured_glb
    else:
        log("texrecon failed or OBJ not found, falling back to untextured GLB")
        convert_to_glb(texrecon_mesh_ply, untextured_glb, textured=False)
        final_glb = untextured_glb

    # -- Metadata --
    elapsed = time.time() - t_start
    metadata = {
        "source_point_cloud": str(ply_path),
        "n_input_points": n_raw,
        "n_downsampled_points": n_downsampled,
        "quality_tier": quality_tier,
        "poisson_depth": params["poisson_depth"],
        "target_triangles": params["target_triangles"],
        "decimated_verts": decimated_verts,
        "decimated_faces": decimated_tris,
        "n_panos": len(pano_names),
        "n_views": n_views,
        "cubemap_face_size": params["cubemap_face_size"],
        "texrecon_success": texrecon_ok,
        "glb_path": str(final_glb),
        "glb_size_mb": round(final_glb.stat().st_size / 1024 / 1024, 1),
        "reconstruction_time_s": round(elapsed, 1),
        "unit": "meter",
    }
    meta_path = output_dir / f"{pc_name}_mesh_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"Metadata written: {meta_path.name}")

    log(f"Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    log(f"Output: {final_glb}")


if __name__ == "__main__":
    main()
