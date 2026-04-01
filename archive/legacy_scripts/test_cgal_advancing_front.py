"""Test wrapper for CGAL Advancing Front Surface Reconstruction.

Pipeline:
  1. Load PLY point cloud, voxel downsample (5mm)
  2. Estimate normals (for potential future use; CGAL AFSR ignores them)
  3. Export downsampled point cloud as PLY (with normals)
  4. Run CGAL advancing_front binary via subprocess
  5. Load output mesh, convert to GLB with double-sided material
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

# -- Path setup ----------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SRC_ROOT.parent

# -- Config --------------------------------------------------------------------
POINT_CLOUD_NAME = "tmb_office_one_corridor_bigger_noRGB"
PLY_PATH = _PROJECT_ROOT / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "mesh" / POINT_CLOUD_NAME

CGAL_BIN = _PROJECT_ROOT / "cgal" / "build" / "cgal_advancing_front"

# Reconstruction params
VOXEL_SIZE = 0.010          # 10mm voxel downsample (5mm OOMs on orient_normals)
NORMAL_RADIUS = 0.15        # meters, radius for normal estimation
NORMAL_MAX_NN = 50          # max neighbours for normal estimation
NORMAL_ORIENT_K = 30        # k for consistent tangent plane orientation
PERIMETER_BOUND = 0.0       # 0 = no limit; set e.g. 0.3 to reject large triangles


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -- Stage 1: Load + downsample ------------------------------------------------

def load_and_downsample():
    """Load PLY point cloud and voxel downsample."""
    log(f"Loading {PLY_PATH.name}...")
    pcd = o3d.io.read_point_cloud(str(PLY_PATH))
    log(f"  {len(pcd.points):,} points")

    log(f"Voxel downsampling at {VOXEL_SIZE * 1000:.0f}mm...")
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    log(f"  {len(pcd.points):,} points after downsample")
    return pcd


# -- Stage 2: Estimate normals -------------------------------------------------

def estimate_normals(pcd):
    """Estimate and orient normals for the point cloud."""
    log(f"Estimating normals (radius={NORMAL_RADIUS}m, max_nn={NORMAL_MAX_NN})...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=NORMAL_MAX_NN,
        )
    )

    log(f"Orienting normals (consistent tangent plane, k={NORMAL_ORIENT_K})...")
    pcd.orient_normals_consistent_tangent_plane(k=NORMAL_ORIENT_K)
    log(f"  Normals estimated for {len(pcd.normals)} points")
    return pcd


# -- Stage 3: Export PLY with normals -------------------------------------------

def export_ply_with_normals(pcd, out_path):
    """Save point cloud as PLY (binary LE) with normals."""
    log(f"Exporting {out_path.name} ({len(pcd.points):,} points with normals)...")
    o3d.io.write_point_cloud(str(out_path), pcd,
                             write_ascii=False,
                             compressed=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log(f"  Written: {size_mb:.1f} MB")


# -- Stage 4: Run CGAL binary --------------------------------------------------

def run_cgal_advancing_front(input_ply, output_ply):
    """Run the CGAL advancing front binary via subprocess."""
    if not CGAL_BIN.exists():
        log(f"ERROR: CGAL binary not found at {CGAL_BIN}")
        log("  Build it with: cd cgal/build && cmake . && make")
        return False

    cmd = [str(CGAL_BIN), str(input_ply), str(output_ply)]
    if PERIMETER_BOUND > 0:
        cmd.append(str(PERIMETER_BOUND))

    log(f"Running CGAL advancing front...")
    log(f"  Command: {' '.join(cmd)}")
    # NOTE: For 5M+ points, Delaunay tetrahedralization can take significant
    # memory (~10x the point cloud size). If the system runs out of memory,
    # increase VOXEL_SIZE to reduce the point count.
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            log(f"  [cgal] {line}")
    if result.returncode != 0:
        log(f"  CGAL FAILED (exit code {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                log(f"  [stderr] {line}")
        return False
    return True


# -- Stage 5: Convert to GLB ---------------------------------------------------

def convert_to_glb(ply_path, glb_path):
    """Load PLY mesh with trimesh and export as GLB with double-sided material."""
    log(f"Converting {ply_path.name} to GLB...")
    import trimesh

    mesh = trimesh.load(str(ply_path), process=False)

    # Set double-sided material so both face orientations are visible
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "material"):
        mesh.visual.material.doubleSided = True
    else:
        material = trimesh.visual.material.PBRMaterial(doubleSided=True)
        mesh.visual = trimesh.visual.TextureVisuals(material=material)

    mesh.export(str(glb_path), file_type="glb")
    size_mb = glb_path.stat().st_size / (1024 * 1024)
    log(f"  GLB written: {glb_path.name} ({size_mb:.1f} MB)")
    log(f"  {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")


# -- Main ----------------------------------------------------------------------

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define output paths
    cgal_input_ply = OUTPUT_DIR / "cgal_input.ply"
    cgal_output_ply = OUTPUT_DIR / "cgal_advancing_front_output.ply"
    glb_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}_cgal.glb"

    # Stage 1: Load + downsample
    pcd = load_and_downsample()

    # Stage 2: Estimate normals
    pcd = estimate_normals(pcd)

    # Stage 3: Export PLY with normals
    export_ply_with_normals(pcd, cgal_input_ply)
    del pcd

    # Stage 4: Run CGAL advancing front
    success = run_cgal_advancing_front(cgal_input_ply, cgal_output_ply)
    if not success:
        log("CGAL advancing front failed, aborting")
        return

    # Stage 5: Convert to GLB
    if cgal_output_ply.exists():
        convert_to_glb(cgal_output_ply, glb_path)
    else:
        log(f"ERROR: {cgal_output_ply.name} not found")

    elapsed = time.time() - t_start
    log(f"Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
