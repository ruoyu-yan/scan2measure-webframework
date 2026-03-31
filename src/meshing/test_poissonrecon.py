"""Test wrapper for PoissonRecon: colorless PLY -> Poisson surface -> trimmed mesh -> GLB.

Pipeline:
  1. Load PLY, voxel downsample (5mm)
  2. Estimate normals (KDTree hybrid search)
  3. Export PLY with normals for PoissonRecon
  4. Run PoissonRecon CLI (depth 10, density annotation)
  5. Run SurfaceTrimmer CLI (trim 7, aRatio 0.001)
  6. Convert trimmed PLY to double-sided GLB for Unity
"""

import subprocess
import sys
import time
from pathlib import Path

import open3d as o3d

# -- Path setup ----------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SRC_ROOT.parent

# -- Config --------------------------------------------------------------------
POINT_CLOUD_NAME = "tmb_office_one_corridor_bigger_noRGB"
PLY_PATH = _PROJECT_ROOT / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "mesh" / POINT_CLOUD_NAME

POISSONRECON_BIN = _PROJECT_ROOT / "PoissonRecon" / "Bin" / "Linux" / "PoissonRecon"
SURFACETRIMMER_BIN = _PROJECT_ROOT / "PoissonRecon" / "Bin" / "Linux" / "SurfaceTrimmer"

# Reconstruction params
VOXEL_SIZE = 0.010          # 10mm voxel downsample (5mm OOMs on orient_normals)
NORMAL_RADIUS = 0.15        # meters, search radius for normal estimation
NORMAL_MAX_NN = 50          # max neighbors for normal estimation
NORMAL_ORIENT_K = 30        # k for consistent tangent plane orientation
POISSON_DEPTH = 10          # octree depth for PoissonRecon
POISSON_POINT_WEIGHT = 2    # interpolation weight
TRIM_VALUE = 7              # density trim threshold for SurfaceTrimmer
TRIM_ARATIO = 0.001         # area ratio to preserve small islands
SUBPROCESS_TIMEOUT = 600    # seconds


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -- Stage 1: Load + downsample ------------------------------------------------

def load_and_downsample():
    """Load PLY and voxel downsample."""
    log(f"Loading point cloud: {PLY_PATH.name}")
    pcd = o3d.io.read_point_cloud(str(PLY_PATH))
    log(f"  {len(pcd.points)} points")

    log(f"Voxel downsampling ({VOXEL_SIZE * 1000:.0f}mm)...")
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    log(f"  {len(pcd.points)} points after downsample")
    return pcd


# -- Stage 2: Estimate normals -------------------------------------------------

def estimate_normals(pcd):
    """Estimate and orient normals on the point cloud."""
    log(f"Estimating normals (radius={NORMAL_RADIUS}, max_nn={NORMAL_MAX_NN})...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN
        )
    )
    log(f"Orienting normals (tangent plane, k={NORMAL_ORIENT_K})...")
    pcd.orient_normals_consistent_tangent_plane(k=NORMAL_ORIENT_K)
    log(f"  Normals estimated: {pcd.has_normals()}")
    return pcd


# -- Stage 3: Export PLY with normals ------------------------------------------

def export_ply_with_normals(pcd, out_path):
    """Save point cloud with normals as PLY for PoissonRecon."""
    log(f"Exporting PLY with normals: {out_path.name}")
    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)
    log(f"  Written: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


# -- Stage 4: Run PoissonRecon -------------------------------------------------

def run_poissonrecon(input_ply, output_ply):
    """Run the PoissonRecon CLI binary.

    Note: PoissonRecon may auto-append .ply to --out if not present,
    so we handle both cases.
    """
    cmd = [
        str(POISSONRECON_BIN),
        "--in", str(input_ply),
        "--out", str(output_ply),
        "--depth", str(POISSON_DEPTH),
        "--density",
        "--pointWeight", str(POISSON_POINT_WEIGHT),
    ]
    log(f"Running PoissonRecon (depth={POISSON_DEPTH}, pointWeight={POISSON_POINT_WEIGHT})...")
    log(f"  cmd: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT
    )

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            log(f"  [PoissonRecon] {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            log(f"  [PoissonRecon stderr] {line}")

    if result.returncode != 0:
        log(f"  PoissonRecon FAILED (exit code {result.returncode})")
        return None

    # Check for the output file -- PoissonRecon may or may not append .ply
    if output_ply.exists():
        log(f"  Output: {output_ply.name} ({output_ply.stat().st_size / 1024 / 1024:.1f} MB)")
        return output_ply

    # Try with .ply appended (some versions auto-append)
    alt_path = output_ply.with_suffix(output_ply.suffix + ".ply")
    if alt_path.exists():
        log(f"  Output (auto-appended): {alt_path.name}")
        alt_path.rename(output_ply)
        log(f"  Renamed to: {output_ply.name} ({output_ply.stat().st_size / 1024 / 1024:.1f} MB)")
        return output_ply

    log(f"  WARNING: Expected output not found at {output_ply}")
    for p in output_ply.parent.glob("poissonrecon_raw*"):
        log(f"  Found: {p.name}")
    return None


# -- Stage 5: Run SurfaceTrimmer -----------------------------------------------

def run_surfacetrimmer(input_ply, output_ply):
    """Run the SurfaceTrimmer CLI binary."""
    cmd = [
        str(SURFACETRIMMER_BIN),
        "--in", str(input_ply),
        "--out", str(output_ply),
        "--trim", str(TRIM_VALUE),
        "--aRatio", str(TRIM_ARATIO),
    ]
    log(f"Running SurfaceTrimmer (trim={TRIM_VALUE}, aRatio={TRIM_ARATIO})...")
    log(f"  cmd: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT
    )

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
    for p in output_ply.parent.glob("poissonrecon_trimmed*"):
        log(f"  Found: {p.name}")
    return None


# -- Stage 6: Convert to GLB ---------------------------------------------------

def convert_to_glb(trimmed_ply, glb_path):
    """Load trimmed PLY with trimesh and export as GLB with double-sided material."""
    log(f"Converting {trimmed_ply.name} -> {glb_path.name}")
    import trimesh

    mesh = trimesh.load(str(trimmed_ply), process=False)

    # Set material to double-sided to fix wall visibility in Unity
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "material"):
        mesh.visual.material.doubleSided = True
    else:
        material = trimesh.visual.material.PBRMaterial(doubleSided=True)
        mesh.visual = trimesh.visual.TextureVisuals(material=material)

    mesh.export(str(glb_path), file_type="glb")
    log(f"  GLB written: {glb_path} ({glb_path.stat().st_size / 1024 / 1024:.1f} MB)")


# -- Main ----------------------------------------------------------------------

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify binaries exist
    if not POISSONRECON_BIN.exists():
        log(f"ERROR: PoissonRecon binary not found at {POISSONRECON_BIN}")
        sys.exit(1)
    if not SURFACETRIMMER_BIN.exists():
        log(f"ERROR: SurfaceTrimmer binary not found at {SURFACETRIMMER_BIN}")
        sys.exit(1)
    if not PLY_PATH.exists():
        log(f"ERROR: Input point cloud not found at {PLY_PATH}")
        sys.exit(1)

    # Stage 1: Load + downsample
    pcd = load_and_downsample()

    # Stage 2: Estimate normals
    pcd = estimate_normals(pcd)

    # Stage 3: Export PLY with normals
    pr_input_ply = OUTPUT_DIR / "poissonrecon_input.ply"
    export_ply_with_normals(pcd, pr_input_ply)
    del pcd

    # Stage 4: Run PoissonRecon
    pr_raw_ply = OUTPUT_DIR / "poissonrecon_raw.ply"
    result_ply = run_poissonrecon(pr_input_ply, pr_raw_ply)
    if result_ply is None:
        log("PoissonRecon failed, aborting")
        sys.exit(1)

    # Stage 5: Run SurfaceTrimmer
    pr_trimmed_ply = OUTPUT_DIR / "poissonrecon_trimmed.ply"
    trimmed_ply = run_surfacetrimmer(result_ply, pr_trimmed_ply)
    if trimmed_ply is None:
        log("SurfaceTrimmer failed, aborting")
        sys.exit(1)

    # Stage 6: Convert to GLB
    glb_path = OUTPUT_DIR / f"{POINT_CLOUD_NAME}_poissonrecon.glb"
    convert_to_glb(trimmed_ply, glb_path)

    elapsed = time.time() - t_start
    log(f"Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
