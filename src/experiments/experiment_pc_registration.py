"""Point cloud registration for coarse panorama localization.

Backprojects DAP depth maps to 3D point clouds, then registers each
against the TLS point cloud using FPFH + RANSAC global registration.

Requires: conda env 'scan_env' (Python 3.8, open3d 0.19.0)
Run:  conda run -n scan_env python src/experiments/experiment_pc_registration.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# ── Config ──────────────────────────────────────────────────────────────────

POINT_CLOUD_NAME = "tmb_scan_subsampled_subsampled_no_RGB"
PANO_NAMES = [
    "TMB_corridor_north1", "TMB_corridor_north2",
    "TMB_corridor_north3", "TMB_corridor_north4",
    "TMB_corridor_south1", "TMB_corridor_south2",
    "TMB_hall1", "TMB_office1",
]

# Ground truth positions (approximate, from user's marked image)
GROUND_TRUTH = {
    "TMB_office1":         [-5.0, 12.0],
    "TMB_hall1":           [-1.0, 12.0],
    "TMB_corridor_north1": [ 9.0, 17.0],
    "TMB_corridor_north2": [ 9.0, 12.0],
    "TMB_corridor_north3": [ 7.0,  3.0],
    "TMB_corridor_north4": [ 8.0, -4.0],
    "TMB_corridor_south1": [ 2.0, -5.0],
    "TMB_corridor_south2": [-3.0,  7.0],
}

# Registration parameters
VOXEL_SIZE_TLS = 0.1        # meters — downsample TLS for registration
VOXEL_SIZE_PANO = 0.1       # meters — downsample backprojected cloud
FPFH_RADIUS = 0.5           # meters — FPFH feature radius (5x voxel size)
FPFH_MAX_NN = 100
RANSAC_DISTANCE = 0.3       # meters — max correspondence distance for RANSAC
RANSAC_N = 3                # points per RANSAC sample
RANSAC_ITERS = 4000000      # RANSAC iterations
ICP_DISTANCE = 0.2          # meters — ICP refinement threshold

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
PC_PATH = ROOT / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
DAP_DIR = ROOT / "data" / "pano" / "dap_depth"
OUTPUT_DIR = ROOT / "data" / "experiments" / "pc_registration"


# ── Step 1: Backproject depth to 3D ─────────────────────────────────────────

DAP_DEPTH_SCALE = 100.0  # DAP outputs normalized depth; multiply by 100 to get meters


def backproject_depth_to_pointcloud(depth, H, W):
    """Backproject equirectangular depth map to 3D point cloud.

    Uses DAP's convention: theta = (1 - u) * 2*pi, phi = v * pi.
    DAP depth is normalized — multiply by DAP_DEPTH_SCALE to get meters.

    Returns (N, 3) float64 array of 3D points.
    """
    # Scale to meters
    depth_m = depth * DAP_DEPTH_SCALE

    v_idx, u_idx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    u_norm = (u_idx + 0.5) / W
    v_norm = (v_idx + 0.5) / H

    # Spherical coordinates (DAP convention)
    theta = (1 - u_norm) * 2 * np.pi   # azimuth
    phi = v_norm * np.pi                # elevation (0=top/zenith, pi=bottom/nadir)

    # DAP: x=sin(phi)*cos(theta), y=sin(phi)*sin(theta), z=cos(phi)
    # phi=0 → z=1 (up), phi=pi → z=-1 (down). This IS Z-up.
    x = depth_m * np.sin(phi) * np.cos(theta)
    y = depth_m * np.sin(phi) * np.sin(theta)
    z = depth_m * np.cos(phi)

    # Mask out invalid depth and poles
    valid = depth_m > 0.1  # at least 10cm
    pole_margin = int(H * 0.1)
    valid[:pole_margin, :] = False
    valid[-pole_margin:, :] = False

    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    return points


# ── Step 2: Prepare point clouds ────────────────────────────────────────────

def prepare_tls_cloud(pc_path, voxel_size):
    """Load and prepare TLS point cloud for registration."""
    print("  Loading TLS point cloud...")
    pcd = o3d.io.read_point_cloud(str(pc_path))
    pts = np.asarray(pcd.points)
    print(f"  {len(pts)} points")

    # Filter to wall-height band (skip floor and ceiling — they're flat/featureless)
    floor_z = np.percentile(pts[:, 2], 5)
    ceiling_z = np.percentile(pts[:, 2], 95)
    wall_band = 0.3  # exclude 0.3m near floor and ceiling
    mask = (pts[:, 2] > floor_z + wall_band) & (pts[:, 2] < ceiling_z - wall_band)
    pcd_walls = o3d.geometry.PointCloud()
    pcd_walls.points = o3d.utility.Vector3dVector(pts[mask])
    print(f"  {mask.sum()} points in wall band [z={floor_z+wall_band:.2f}, {ceiling_z-wall_band:.2f}]")

    pcd_down = pcd_walls.voxel_down_sample(voxel_size)
    print(f"  {len(pcd_down.points)} points after {voxel_size}m downsample")

    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return pcd_down, floor_z, ceiling_z


def prepare_pano_cloud(depth, voxel_size, floor_z, ceiling_z):
    """Backproject depth and prepare for registration."""
    H, W = depth.shape
    pts = backproject_depth_to_pointcloud(depth, H, W)
    print(f"  Backprojected {len(pts)} points")

    # Filter to wall-height band (same as TLS)
    wall_band = 0.3
    # The backprojected cloud is centered at origin (camera position)
    # We can't filter by absolute Z yet — need to filter by relative height
    # Camera is at some height above floor. In the TLS, camera_z ≈ floor_z + 1.5
    # In the backprojected cloud, camera is at origin, so floor is at z ≈ -1.5
    # Filter: keep points roughly at wall height relative to camera
    # wall band in camera coords: floor+wall_band-cam_z to ceiling-wall_band-cam_z
    cam_height = 1.5  # approximate
    z_min_rel = -(cam_height - wall_band)        # ~-1.2
    z_max_rel = (ceiling_z - floor_z) - cam_height - wall_band  # ~1.5
    mask = (pts[:, 2] > z_min_rel) & (pts[:, 2] < z_max_rel)
    pts = pts[mask]
    print(f"  {len(pts)} points in wall band [z_rel={z_min_rel:.2f}, {z_max_rel:.2f}]")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"  {len(pcd_down.points)} points after {voxel_size}m downsample")

    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return pcd_down


# ── Step 3: Registration ────────────────────────────────────────────────────

def compute_fpfh(pcd, voxel_size):
    """Compute FPFH features for a point cloud."""
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_RADIUS, max_nn=FPFH_MAX_NN),
    )


def register_global(source, target, source_fpfh, target_fpfh):
    """Global registration using FPFH + RANSAC."""
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=RANSAC_DISTANCE,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=RANSAC_N,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(RANSAC_DISTANCE),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(RANSAC_ITERS, 0.999),
    )
    return result


def register_fgr(source, target, source_fpfh, target_fpfh):
    """Fast Global Registration (backup if RANSAC is slow)."""
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=RANSAC_DISTANCE,
        ),
    )
    return result


def refine_icp(source, target, initial_transform):
    """ICP refinement after global registration."""
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        ICP_DISTANCE,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result


# ── Visualization ───────────────────────────────────────────────────────────

PANO_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


def render_topdown(tls_pts, positions, pano_names, ground_truth, output_path):
    """Top-down with estimated positions AND ground truth."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    n = len(tls_pts)
    if n > 200000:
        idx = np.random.choice(n, 200000, replace=False)
        pts_sub = tls_pts[idx]
    else:
        pts_sub = tls_pts

    ax.scatter(pts_sub[:, 0], pts_sub[:, 1], s=0.01, c="gray", alpha=0.3)

    for i, (name, pos) in enumerate(zip(pano_names, positions)):
        color = PANO_COLORS[i % len(PANO_COLORS)]

        # Ground truth (X marker)
        if name in ground_truth:
            gt = ground_truth[name]
            ax.scatter(gt[0], gt[1], s=300, c=color, marker="x",
                       linewidths=3, zorder=9, alpha=0.5)

        # Estimated position (circle)
        if pos is not None:
            ax.scatter(pos[0], pos[1], s=200, c=color, edgecolors="white",
                       linewidths=2, zorder=10)
            ax.annotate(name.replace("TMB_", ""), (pos[0], pos[1]),
                        fontsize=7, fontweight="bold", color=color,
                        ha="center", va="bottom",
                        xytext=(0, 12), textcoords="offset points")

            # Draw line from estimated to ground truth
            if name in ground_truth:
                gt = ground_truth[name]
                ax.plot([pos[0], gt[0]], [pos[1], gt[1]], c=color,
                        linewidth=1, linestyle="--", alpha=0.5)

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("PC Registration — Estimated (circles) vs Ground Truth (X)")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved top-down view to {output_path.name}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare TLS cloud
    print("=== Step 1: Prepare TLS cloud ===")
    tls_down, floor_z, ceiling_z = prepare_tls_cloud(PC_PATH, VOXEL_SIZE_TLS)
    print("  Computing FPFH features for TLS...")
    tls_fpfh = compute_fpfh(tls_down, VOXEL_SIZE_TLS)
    print(f"  TLS FPFH: {tls_fpfh.data.shape}")

    # Load full cloud for visualization
    pcd_full = o3d.io.read_point_cloud(str(PC_PATH))
    tls_pts = np.asarray(pcd_full.points)

    # Step 2: Register each panorama
    print("\n=== Step 2: Register panoramas ===")
    results = {}
    positions = []
    names = []

    for pano_name in PANO_NAMES:
        dap_path = DAP_DIR / f"{pano_name}.npy"
        if not dap_path.exists():
            print(f"\n  SKIP {pano_name}: no DAP depth")
            positions.append(None)
            names.append(pano_name)
            continue

        print(f"\n  --- {pano_name} ---")
        depth = np.load(str(dap_path))

        # Backproject
        pano_down = prepare_pano_cloud(depth, VOXEL_SIZE_PANO, floor_z, ceiling_z)
        if len(pano_down.points) < 100:
            print(f"  Too few points after filtering, skipping")
            positions.append(None)
            names.append(pano_name)
            continue

        # FPFH features
        print("  Computing FPFH features...")
        pano_fpfh = compute_fpfh(pano_down, VOXEL_SIZE_PANO)

        # Try both RANSAC and FGR, keep better result
        t_reg = time.time()

        print("  Running RANSAC global registration...")
        result_ransac = register_global(pano_down, tls_down, pano_fpfh, tls_fpfh)
        print(f"    RANSAC fitness={result_ransac.fitness:.3f}, "
              f"inlier_rmse={result_ransac.inlier_rmse:.3f}")

        print("  Running FGR global registration...")
        result_fgr = register_fgr(pano_down, tls_down, pano_fpfh, tls_fpfh)
        print(f"    FGR fitness={result_fgr.fitness:.3f}, "
              f"inlier_rmse={result_fgr.inlier_rmse:.3f}")

        # Pick the one with higher fitness
        if result_ransac.fitness >= result_fgr.fitness:
            result = result_ransac
            method = "RANSAC"
        else:
            result = result_fgr
            method = "FGR"

        # ICP refinement
        print(f"  Refining with ICP (from {method})...")
        result_icp = refine_icp(pano_down, tls_down, result.transformation)
        print(f"    ICP fitness={result_icp.fitness:.3f}, "
              f"inlier_rmse={result_icp.inlier_rmse:.3f}")

        # Extract camera position (the translation component)
        T = result_icp.transformation
        # The transformation maps source (pano cloud, centered at origin) to target (TLS)
        # So the camera position in TLS frame = T[:3, 3] (the translation)
        cam_pos = T[:3, 3].tolist()
        elapsed = time.time() - t_reg

        # Compute error vs ground truth
        gt = GROUND_TRUTH.get(pano_name)
        if gt:
            err = np.sqrt((cam_pos[0] - gt[0])**2 + (cam_pos[1] - gt[1])**2)
            err_str = f", error={err:.1f}m"
        else:
            err = None
            err_str = ""

        print(f"  Position: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}] "
              f"({method}+ICP, {elapsed:.1f}s{err_str})")

        results[pano_name] = {
            "position": cam_pos,
            "fitness": float(result_icp.fitness),
            "inlier_rmse": float(result_icp.inlier_rmse),
            "method": method,
            "error_2d": float(err) if err else None,
        }
        positions.append(cam_pos)
        names.append(pano_name)

    # Step 3: Output
    print("\n=== Step 3: Save results ===")
    total_time = time.time() - t_start
    output = {
        "metadata": {
            "pipeline": "pc-registration-v1",
            "point_cloud": POINT_CLOUD_NAME,
            "voxel_size_tls": VOXEL_SIZE_TLS,
            "voxel_size_pano": VOXEL_SIZE_PANO,
            "total_time_seconds": round(total_time, 1),
        },
        "positions": results,
    }
    json_path = OUTPUT_DIR / "coarse_positions.json"
    with open(str(json_path), "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved positions to {json_path.name}")

    # Visualization with ground truth comparison
    render_topdown(tls_pts, positions, names, GROUND_TRUTH,
                   OUTPUT_DIR / "topdown_result.png")

    # Summary
    print("\n=== Summary ===")
    errors = []
    for name in PANO_NAMES:
        if name in results and results[name]["error_2d"] is not None:
            err = results[name]["error_2d"]
            errors.append(err)
            status = "OK" if err < 2.0 else "MISS"
            print(f"  {name}: {err:.1f}m [{status}]")
    if errors:
        print(f"\n  Mean error: {np.mean(errors):.1f}m")
        print(f"  Median error: {np.median(errors):.1f}m")
        print(f"  Within 2m: {sum(1 for e in errors if e < 2.0)}/{len(errors)}")

    print(f"\nDone in {total_time:.1f}s")


if __name__ == "__main__":
    main()
