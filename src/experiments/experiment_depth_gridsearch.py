"""Depth grid search for coarse panorama localization.

Meshes an uncolored TLS point cloud, renders equirectangular depth at a
grid of candidate positions, compares against DAP-estimated depth from
each panorama, and picks the best-matching position per panorama.

Requires: conda env 'scan_env' (Python 3.8, open3d 0.19.0)
Run:  conda run -n scan_env python src/experiments/experiment_depth_gridsearch.py
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
from scipy.spatial import cKDTree

# ── Config ──────────────────────────────────────────────────────────────────

POINT_CLOUD_NAME = "tmb_scan_subsampled_subsampled_no_RGB"
PANO_NAMES = [
    "TMB_corridor_north1", "TMB_corridor_north2",
    "TMB_corridor_north3", "TMB_corridor_north4",
    "TMB_corridor_south1", "TMB_corridor_south2",
    "TMB_hall1", "TMB_office1",
]

GRID_SPACING = 0.5          # meters
CAMERA_HEIGHT = 1.5         # meters above detected floor
VOXEL_SIZE = 0.05           # meters — coarser for speed (still ~1M points)
RENDER_H, RENDER_W = 512, 1024
POLE_MASK_RATIO = 0.1       # exclude top/bottom 10% of rows
MIN_VALID_PIXELS = 1000     # skip candidates with too few valid pixels

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
PC_PATH = ROOT / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
DAP_DIR = ROOT / "data" / "pano" / "dap_depth"
OUTPUT_DIR = ROOT / "data" / "experiments" / "depth_gridsearch"


# ── Step 1: Load point cloud ─────────────────────────────────────────────────

def load_point_cloud(pc_path):
    """Load and downsample the point cloud."""
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(pc_path))
    pts = np.asarray(pcd.points)
    print(f"  {len(pts)} points")

    print(f"  Voxel downsampling to {VOXEL_SIZE}m...")
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    pts = np.asarray(pcd.points).astype(np.float32)
    print(f"  {len(pts)} points after downsample")
    return pts


# ── Step 2: Candidate grid ──────────────────────────────────────────────────

def build_candidate_grid(pts):
    """Generate candidate camera positions on a 2D grid above the floor."""
    # Detect floor and compute camera height
    floor_z = np.percentile(pts[:, 2], 5)
    cam_z = floor_z + CAMERA_HEIGHT
    print(f"  Floor Z = {floor_z:.2f}, camera Z = {cam_z:.2f}")

    # 2D grid over XY bounding box
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    xs = np.arange(x_min + GRID_SPACING / 2, x_max, GRID_SPACING)
    ys = np.arange(y_min + GRID_SPACING / 2, y_max, GRID_SPACING)
    grid_xy = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    print(f"  Raw grid: {len(grid_xy)} candidates")

    # Filter: keep candidates inside the building footprint using 2D occupancy
    bin_size = GRID_SPACING
    x_bins = np.arange(x_min, x_max + bin_size, bin_size)
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)
    occupancy, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_bins, y_bins])

    cx = np.digitize(grid_xy[:, 0], x_bins) - 1
    cy = np.digitize(grid_xy[:, 1], y_bins) - 1
    cx = np.clip(cx, 0, occupancy.shape[0] - 1)
    cy = np.clip(cy, 0, occupancy.shape[1] - 1)
    cell_counts = occupancy[cx, cy]
    valid = cell_counts > 50

    candidates = np.zeros((valid.sum(), 3), dtype=np.float32)
    candidates[:, 0] = grid_xy[valid, 0]
    candidates[:, 1] = grid_xy[valid, 1]
    candidates[:, 2] = cam_z
    print(f"  Filtered grid: {len(candidates)} candidates")
    return candidates


# ── Step 3: Point cloud depth rendering ──────────────────────────────────────

def render_depth_from_pc(pts, position, H, W):
    """Render equirectangular depth by projecting points onto a z-buffer.

    Projects ALL points (already downsampled) from the candidate position.
    Uses vectorized sort + index assignment for z-buffer.

    Returns (H, W) float32 depth (0 where no point hit).
    """
    rel = pts - position
    dist = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2 + rel[:, 2]**2)

    # Skip points too close
    valid = dist > 0.1
    rel = rel[valid]
    dist = dist[valid]

    # Spherical coordinates (Z-up)
    azimuth = np.arctan2(rel[:, 1], rel[:, 0])
    elevation = np.arcsin(np.clip(rel[:, 2] / dist, -1, 1))

    # Map to pixel coordinates
    u = (((azimuth / (2 * np.pi)) % 1.0) * W).astype(np.int32)
    v = ((0.5 - elevation / np.pi) * H).astype(np.int32)
    np.clip(u, 0, W - 1, out=u)
    np.clip(v, 0, H - 1, out=v)

    # Z-buffer: sort farthest-first, write to array (nearest overwrites)
    order = np.argsort(-dist)
    depth = np.zeros((H, W), dtype=np.float32)
    depth[v[order], u[order]] = dist[order]

    return depth


# ── Step 4: Depth comparison ────────────────────────────────────────────────

NUM_AZIMUTH_SHIFTS = 18  # test every 20 degrees


def compare_depths(rendered, estimated):
    """Compare depths with azimuth rotation search.

    Tests NUM_AZIMUTH_SHIFTS horizontal shifts of the rendered depth,
    using properly normalized Pearson correlation of log-depth.

    Returns (best_score, best_absrel, best_shift_columns).
    """
    H, W = rendered.shape
    pole_margin = int(H * POLE_MASK_RATIO)
    shift_step = W // NUM_AZIMUTH_SHIFTS

    # Precompute estimated log-depth (doesn't change across shifts)
    e_mask_base = (estimated > 1e-6)
    e_mask_base[:pole_margin, :] = False
    e_mask_base[-pole_margin:, :] = False
    log_e = np.zeros_like(estimated)
    log_e[e_mask_base] = np.log(estimated[e_mask_base])

    best_score = -999.0
    best_absrel = 999.0
    best_shift = 0

    for i in range(NUM_AZIMUTH_SHIFTS):
        shift = i * shift_step
        shifted = np.roll(rendered, shift, axis=1)

        # Combined mask
        mask = e_mask_base & (shifted > 0.01)
        n_valid = mask.sum()
        if n_valid < MIN_VALID_PIXELS:
            continue

        log_r = np.log(shifted[mask])
        le = log_e[mask]

        # Pearson correlation
        r_mean, e_mean = log_r.mean(), le.mean()
        r_std, e_std = log_r.std(), le.std()
        if r_std < 1e-8 or e_std < 1e-8:
            continue
        score = float(np.mean((log_r - r_mean) * (le - e_mean)) / (r_std * e_std))

        if score > best_score:
            best_score = score
            best_shift = shift
            # AbsRel at this shift
            scale = np.median(shifted[mask]) / np.median(estimated[mask])
            aligned = estimated[mask] * scale
            best_absrel = float(np.mean(np.abs(shifted[mask] - aligned) / shifted[mask]))

    if best_score < -10:
        return float("nan"), float("nan"), 0
    return best_score, best_absrel, best_shift


# ── Step 6: Visualization ───────────────────────────────────────────────────

PANO_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


def render_topdown(pts, positions, pano_names, output_path):
    """Top-down point cloud view with estimated camera positions."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Subsample points for plotting
    n = len(pts)
    if n > 200000:
        idx = np.random.choice(n, 200000, replace=False)
        pts_sub = pts[idx]
    else:
        pts_sub = pts

    ax.scatter(pts_sub[:, 0], pts_sub[:, 1], s=0.01, c="gray", alpha=0.3)

    for i, (name, pos) in enumerate(zip(pano_names, positions)):
        if pos is None:
            continue
        color = PANO_COLORS[i % len(PANO_COLORS)]
        ax.scatter(pos[0], pos[1], s=200, c=color, edgecolors="white",
                   linewidths=2, zorder=10)
        ax.annotate(name, (pos[0], pos[1]), fontsize=7, fontweight="bold",
                    color=color, ha="center", va="bottom",
                    xytext=(0, 12), textcoords="offset points")

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Depth Grid Search — Coarse Camera Positions")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved top-down view to {output_path.name}")


def render_depth_comparison(dap_depth, rendered_depth, score, pano_name, output_path):
    """Side-by-side: DAP depth (left), best rendered depth (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    rendered_vis = rendered_depth.copy()
    rendered_vis[~np.isfinite(rendered_vis)] = 0

    axes[0].imshow(np.log1p(dap_depth), cmap="Spectral")
    axes[0].set_title(f"DAP depth — {pano_name}")
    axes[0].axis("off")

    axes[1].imshow(np.log1p(rendered_vis), cmap="Spectral")
    axes[1].set_title(f"Best rendered depth (score={score:.3f})")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load point cloud
    print("=== Step 1: Load point cloud ===")
    pts = load_point_cloud(PC_PATH)

    # Step 2: Candidate grid
    print("\n=== Step 2: Candidate grid ===")
    candidates = build_candidate_grid(pts)

    # Step 4-5: Compare and select per panorama
    print("\n=== Step 4-5: Compare depths ===")
    results = {}
    best_positions = []
    best_names = []

    for pano_name in PANO_NAMES:
        dap_path = DAP_DIR / f"{pano_name}.npy"
        if not dap_path.exists():
            print(f"  SKIP {pano_name}: no DAP depth at {dap_path}")
            best_positions.append(None)
            best_names.append(pano_name)
            continue

        dap_depth = np.load(str(dap_path))
        print(f"\n  {pano_name}: comparing against {len(candidates)} candidates...")

        best_score = -999.0
        best_absrel = 999.0
        best_idx = -1
        best_rendered = None
        best_shift = 0

        t_pano = time.time()
        for i, pos in enumerate(candidates):
            rendered = render_depth_from_pc(pts, pos, RENDER_H, RENDER_W)
            score, absrel, shift = compare_depths(rendered, dap_depth)

            if not np.isnan(score) and score > best_score:
                best_score = score
                best_absrel = absrel
                best_idx = i
                best_shift = shift
                best_rendered = rendered.copy()

        elapsed = time.time() - t_pano
        if best_idx >= 0:
            pos = candidates[best_idx].tolist()
            azimuth_deg = best_shift / RENDER_W * 360.0
            print(
                f"  {pano_name}: best position = [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
                f"score = {best_score:.3f}, absrel = {best_absrel:.3f}, "
                f"azimuth_shift = {azimuth_deg:.0f}deg ({elapsed:.1f}s)"
            )
            results[pano_name] = {
                "position": pos,
                "score": best_score,
                "absrel": best_absrel,
                "azimuth_shift_deg": azimuth_deg,
            }
            best_positions.append(pos)
            best_names.append(pano_name)

            # Per-pano debug viz — show the ALIGNED rendered depth
            aligned_rendered = np.roll(best_rendered, best_shift, axis=1)
            render_depth_comparison(
                dap_depth, aligned_rendered, best_score, pano_name,
                OUTPUT_DIR / f"depth_comparison_{pano_name}.png",
            )
        else:
            print(f"  {pano_name}: no valid candidate found")
            best_positions.append(None)
            best_names.append(pano_name)

    # Step 6: Output
    print("\n=== Step 6: Save results ===")
    total_time = time.time() - t_start
    output = {
        "metadata": {
            "pipeline": "depth-gridsearch-v1",
            "point_cloud": POINT_CLOUD_NAME,
            "grid_spacing": GRID_SPACING,
            "camera_height": CAMERA_HEIGHT,
            "n_candidates_tested": len(candidates),
            "total_time_seconds": round(total_time, 1),
        },
        "positions": results,
    }
    json_path = OUTPUT_DIR / "coarse_positions.json"
    with open(str(json_path), "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved positions to {json_path.name}")

    # Top-down visualization
    render_topdown(pts, best_positions, best_names, OUTPUT_DIR / "topdown_result.png")

    print(f"\nDone in {total_time:.1f}s")


if __name__ == "__main__":
    main()
