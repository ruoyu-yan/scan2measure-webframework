# Depth Grid Search Experiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two experiment scripts that find coarse camera positions of panoramas inside an uncolored TLS point cloud by comparing monocular depth estimates against depth rendered from a meshed point cloud.

**Architecture:** Script 1 (dap_env) runs DAP depth estimation on panoramas and saves `.npy` files. Script 2 (scan_env) meshes the point cloud, raycasts equirectangular depth at a grid of candidates, compares against DAP depth via log-depth correlation, and outputs positions + top-down visualization.

**Tech Stack:** Open3D 0.19.0 (meshing, raycasting), DAP (panoramic depth estimation, PyTorch 2.7.1), matplotlib (visualization), numpy/scipy.

**Spec:** `docs/superpowers/specs/2026-04-04-depth-gridsearch-design.md`

**Key data facts (verified):**
- Point cloud: 12.3M points, X[-7.2, 10.8], Y[-8.9, 21.4], Z[-1.2, 1.95], Z-up, no normals, no colors
- Floor Z ≈ -1.09 (5th percentile), ceiling Z ≈ 1.91 (95th percentile)
- Panoramas: 8192×4096 JPGs in `data/pano/raw/`
- DAP output: float32 (512, 1024), values in [0.008, 0.057] range (normalized, NOT metric)
- DAP must run from `/tmp/DAP` working directory (DINOv3 uses relative paths)
- DAP weights cached at `~/.cache/huggingface/hub/models--Insta360-Research--DAP-weights/snapshots/558e9ac84efbcb46dc8c47b32c73b333d95f4f0d/model.pth`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/experiments/experiment_dap_depth.py` (create) | DAP depth estimation for panoramas → .npy files |
| `src/experiments/experiment_depth_gridsearch.py` (create) | Mesh + grid search + comparison + visualization |

No existing files are modified. No tests — these are standalone experiment scripts (matching project convention for `src/experiments/`).

---

### Task 1: Script 1 — DAP depth estimation

**Files:**
- Create: `src/experiments/experiment_dap_depth.py`

- [ ] **Step 1: Write the script**

```python
"""DAP panoramic depth estimation for depth grid search experiment.

Runs the DAP (Depth Any Panoramas) foundation model on each panorama
and saves metric depth maps as .npy files for downstream comparison.

Requires: conda env 'dap_env' (Python 3.12, PyTorch 2.7.1)
Run:  conda run -n dap_env --cwd /tmp/DAP python \
        /home/ruoyu/scan2measure-webframework/src/experiments/experiment_dap_depth.py
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib

# DAP must be imported from its repo root (DINOv3 uses relative hubconf.py)
DAP_ROOT = "/tmp/DAP"
sys.path.insert(0, DAP_ROOT)
os.chdir(DAP_ROOT)
from networks.models import make  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────

PANO_NAMES = [
    "TMB_corridor_north1", "TMB_corridor_north2",
    "TMB_corridor_north3", "TMB_corridor_north4",
    "TMB_corridor_south1", "TMB_corridor_south2",
    "TMB_hall1", "TMB_office1",
]

PROJECT_ROOT = Path("/home/ruoyu/scan2measure-webframework")
PANO_DIR = PROJECT_ROOT / "data" / "pano" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "pano" / "dap_depth"

RENDER_H, RENDER_W = 512, 1024

DAP_MODEL_CONFIG = {
    "model": {
        "name": "dap",
        "args": {
            "midas_model_type": "vitl",
            "fine_tune_type": "hypersim",
            "min_depth": 0.01,
            "max_depth": 1.0,
            "train_decoder": True,
        },
    }
}

DAP_WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Insta360-Research--DAP-weights"
    "/snapshots/558e9ac84efbcb46dc8c47b32c73b333d95f4f0d/model.pth"
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_dap_model():
    """Load DAP model from cached HuggingFace weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(DAP_WEIGHTS, map_location=device)
    model = make(DAP_MODEL_CONFIG["model"])
    if any(k.startswith("module") for k in state.keys()):
        model = nn.DataParallel(model)
    model = model.to(device)
    m_state = model.state_dict()
    model.load_state_dict(
        {k: v for k, v in state.items() if k in m_state}, strict=False
    )
    model.eval()
    return model, device


def estimate_depth(model, device, img_rgb):
    """Run DAP on a single RGB image. Returns float32 (H, W) depth."""
    img = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(tensor)
        pred = outputs["pred_depth"][0].detach().cpu().squeeze().numpy()
    return pred.astype(np.float32)


def colorize_depth(depth, cmap="Spectral"):
    """Convert depth to colorized uint8 RGB for visualization."""
    d = depth.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    colored = matplotlib.colormaps[cmap](d)[..., :3]
    return (colored * 255).astype(np.uint8)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading DAP model...")
    model, device = load_dap_model()
    print(f"DAP model loaded on {device}")

    for pano_name in PANO_NAMES:
        t0 = time.time()
        img_path = PANO_DIR / f"{pano_name}.jpg"
        if not img_path.exists():
            print(f"SKIP {pano_name}: {img_path} not found")
            continue

        # Read and resize
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (RENDER_W, RENDER_H))

        # Estimate depth
        depth = estimate_depth(model, device, img_rgb)

        # Save .npy
        npy_path = OUTPUT_DIR / f"{pano_name}.npy"
        np.save(str(npy_path), depth)

        # Save colorized visualization
        vis = colorize_depth(depth)
        vis_path = OUTPUT_DIR / f"{pano_name}_vis.png"
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        elapsed = time.time() - t0
        print(
            f"{pano_name}: depth range [{depth.min():.4f}, {depth.max():.4f}], "
            f"saved to {npy_path.name} ({elapsed:.1f}s)"
        )

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run:
```bash
conda run -n dap_env --cwd /tmp/DAP python /home/ruoyu/scan2measure-webframework/src/experiments/experiment_dap_depth.py
```

Expected: 8 `.npy` files + 8 `_vis.png` files in `data/pano/dap_depth/`. Each takes ~2-3s on GPU. Check the `_vis.png` files visually — they should show clear room structure (walls closer = brighter or darker depending on colormap).

- [ ] **Step 3: Commit**

```bash
cd /home/ruoyu/scan2measure-webframework
git add src/experiments/experiment_dap_depth.py
git commit -m "feat: add DAP depth estimation experiment script"
```

---

### Task 2: Script 2 — Mesh, grid search, compare, visualize

**Files:**
- Create: `src/experiments/experiment_depth_gridsearch.py`

- [ ] **Step 1: Write the script**

```python
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

import cv2
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

GRID_SPACING = 0.5          # meters
CAMERA_HEIGHT = 1.5         # meters above detected floor
POISSON_DEPTH = 9
VOXEL_SIZE = 0.02           # meters for downsampling before meshing
RENDER_H, RENDER_W = 512, 1024
POLE_MASK_RATIO = 0.1       # exclude top/bottom 10% of rows
MIN_VALID_PIXELS = 1000     # skip candidates with too few valid pixels

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
PC_PATH = ROOT / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
DAP_DIR = ROOT / "data" / "pano" / "dap_depth"
OUTPUT_DIR = ROOT / "data" / "experiments" / "depth_gridsearch"
MESH_CACHE = OUTPUT_DIR / "mesh_cache.ply"


# ── Step 1: Mesh ────────────────────────────────────────────────────────────

def get_or_create_mesh(pc_path, cache_path):
    """Load cached mesh or create via Poisson reconstruction."""
    if cache_path.exists():
        print(f"Loading cached mesh from {cache_path.name}...")
        return o3d.io.read_triangle_mesh(str(cache_path))

    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(pc_path))
    print(f"  {len(pcd.points)} points")

    print(f"  Voxel downsampling to {VOXEL_SIZE}m...")
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    print(f"  {len(pcd.points)} points after downsample")

    print("  Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    print(f"  Poisson reconstruction (depth={POISSON_DEPTH})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH
    )

    # Trim low-density vertices
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(cache_path), mesh)
    print(f"  Cached to {cache_path.name}")
    return mesh


# ── Step 2: Candidate grid ──────────────────────────────────────────────────

def build_candidate_grid(pc_path, mesh):
    """Generate candidate camera positions on a 2D grid above the floor."""
    pcd = o3d.io.read_point_cloud(str(pc_path))
    pts = np.asarray(pcd.points)

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

    # Filter: raycast downward to check for floor hit
    scene = o3d.t.geometry.RaycastingScene()
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(t_mesh)

    # Cast rays straight down from above ceiling
    ceiling_z = np.percentile(pts[:, 2], 95) + 1.0
    origins = np.zeros((len(grid_xy), 3), dtype=np.float32)
    origins[:, 0] = grid_xy[:, 0]
    origins[:, 1] = grid_xy[:, 1]
    origins[:, 2] = ceiling_z
    directions = np.zeros_like(origins)
    directions[:, 2] = -1.0  # straight down

    rays = np.concatenate([origins, directions], axis=1)
    result = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = result["t_hit"].numpy()

    # Keep candidates where a floor was hit at reasonable distance
    floor_hit_z = ceiling_z - t_hit
    valid = np.isfinite(t_hit) & (floor_hit_z < cam_z) & (floor_hit_z > floor_z - 0.5)

    candidates = np.zeros((valid.sum(), 3), dtype=np.float32)
    candidates[:, 0] = grid_xy[valid, 0]
    candidates[:, 1] = grid_xy[valid, 1]
    candidates[:, 2] = cam_z
    print(f"  Filtered grid: {len(candidates)} candidates")
    return candidates, scene


# ── Step 3: Equirectangular raycasting ───────────────────────────────────────

def make_equirect_ray_directions(H, W):
    """Generate unit ray directions for equirectangular projection.

    Returns (H, W, 3) float32. Convention: Z-up (matches TLS point cloud).
    Azimuth 0 = +X direction, increases counterclockwise.
    Elevation 0 = horizon, +pi/2 = up (Z+), -pi/2 = down (Z-).
    """
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # Elevation: top row = +pi/2 (up), bottom = -pi/2 (down)
    elevation = (0.5 - (v + 0.5) / H) * np.pi
    # Azimuth: left to right = 0 to 2*pi
    azimuth = (u + 0.5) / W * 2.0 * np.pi

    dx = np.cos(elevation) * np.cos(azimuth)
    dy = np.cos(elevation) * np.sin(azimuth)
    dz = np.sin(elevation)
    return np.stack([dx, dy, dz], axis=-1).astype(np.float32)


def raycast_depth(scene, position, ray_dirs):
    """Render equirectangular depth from a single position.

    Returns (H, W) float32 depth (inf where no hit).
    """
    H, W, _ = ray_dirs.shape
    origins = np.broadcast_to(position, ray_dirs.shape).copy()
    rays = np.concatenate([origins, ray_dirs], axis=-1).reshape(-1, 6)
    result = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
    depth = result["t_hit"].numpy().reshape(H, W)
    return depth


# ── Step 4: Depth comparison ────────────────────────────────────────────────

def compare_depths(rendered, estimated):
    """Compute log-depth Pearson correlation between rendered and estimated.

    Both are (H, W) float32. Returns (score, absrel) or (nan, nan) if
    insufficient valid pixels.
    """
    H = rendered.shape[0]
    pole_margin = int(H * POLE_MASK_RATIO)

    # Mask: valid depth, exclude poles
    mask = np.ones_like(rendered, dtype=bool)
    mask[:pole_margin, :] = False
    mask[-pole_margin:, :] = False
    mask &= np.isfinite(rendered) & (rendered > 0.01)
    mask &= (estimated > 1e-6)

    if mask.sum() < MIN_VALID_PIXELS:
        return float("nan"), float("nan")

    log_r = np.log(rendered[mask])
    log_e = np.log(estimated[mask])

    # Pearson correlation of log-depth (scale-invariant)
    log_r_n = (log_r - log_r.mean()) / (log_r.std() + 1e-8)
    log_e_n = (log_e - log_e.mean()) / (log_e.std() + 1e-8)
    score = np.mean(log_r_n * log_e_n)

    # AbsRel after median scale alignment
    scale = np.median(rendered[mask]) / np.median(estimated[mask])
    aligned = estimated[mask] * scale
    absrel = np.mean(np.abs(rendered[mask] - aligned) / rendered[mask])

    return float(score), float(absrel)


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

    vmin = min(np.nanmin(dap_depth[dap_depth > 0]),
               np.nanmin(rendered_depth[np.isfinite(rendered_depth) & (rendered_depth > 0)]))
    vmax = max(np.nanmax(dap_depth), np.nanmax(rendered_depth[np.isfinite(rendered_depth)]))
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

    # Step 1: Mesh
    print("=== Step 1: Mesh ===")
    mesh = get_or_create_mesh(PC_PATH, MESH_CACHE)

    # Step 2: Candidate grid
    print("\n=== Step 2: Candidate grid ===")
    candidates, scene = build_candidate_grid(PC_PATH, mesh)

    # Step 3: Precompute ray directions
    print("\n=== Step 3: Precompute ray directions ===")
    ray_dirs = make_equirect_ray_directions(RENDER_H, RENDER_W)
    print(f"  Ray directions shape: {ray_dirs.shape}")

    # Load point cloud for top-down viz
    pcd = o3d.io.read_point_cloud(str(PC_PATH))
    pts = np.asarray(pcd.points)

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

        t_pano = time.time()
        for i, pos in enumerate(candidates):
            rendered = raycast_depth(scene, pos, ray_dirs)
            score, absrel = compare_depths(rendered, dap_depth)

            if not np.isnan(score) and score > best_score:
                best_score = score
                best_absrel = absrel
                best_idx = i
                best_rendered = rendered.copy()

        elapsed = time.time() - t_pano
        if best_idx >= 0:
            pos = candidates[best_idx].tolist()
            print(
                f"  {pano_name}: best position = [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
                f"score = {best_score:.3f}, absrel = {best_absrel:.3f} ({elapsed:.1f}s)"
            )
            results[pano_name] = {
                "position": pos,
                "score": best_score,
                "absrel": best_absrel,
            }
            best_positions.append(pos)
            best_names.append(pano_name)

            # Per-pano debug viz
            render_depth_comparison(
                dap_depth, best_rendered, best_score, pano_name,
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
```

- [ ] **Step 2: Run DAP depth estimation first (if not already done)**

```bash
conda run -n dap_env --cwd /tmp/DAP python /home/ruoyu/scan2measure-webframework/src/experiments/experiment_dap_depth.py
```

Verify: `ls data/pano/dap_depth/*.npy` shows 8 files.

- [ ] **Step 3: Run the grid search script**

```bash
cd /home/ruoyu/scan2measure-webframework
conda run -n scan_env python src/experiments/experiment_depth_gridsearch.py
```

Expected output:
- First run: Poisson meshing takes 1-5 minutes, then grid search ~30-60s
- Subsequent runs: mesh loaded from cache, grid search only
- `data/experiments/depth_gridsearch/coarse_positions.json` — positions for each panorama
- `data/experiments/depth_gridsearch/topdown_result.png` — top-down view with camera dots
- `data/experiments/depth_gridsearch/depth_comparison_*.png` — per-pano debug images

Check: Open `topdown_result.png` — camera dots should be inside the building footprint in distinct rooms/corridors.

- [ ] **Step 4: Evaluate and iterate if needed**

If results look wrong:
1. Check `depth_comparison_*.png` — does DAP depth structurally resemble rendered depth?
2. If all scores are very low (<0.1), the equirectangular convention may be flipped. Try negating the azimuth direction in `make_equirect_ray_directions` (change `azimuth = (u + 0.5) / W * 2.0 * np.pi` to `azimuth = -(u + 0.5) / W * 2.0 * np.pi`).
3. If positions cluster together, try reducing `GRID_SPACING` to 0.3m for finer resolution.
4. If candidates are too few, relax the floor-hit filter thresholds.

- [ ] **Step 5: Commit**

```bash
cd /home/ruoyu/scan2measure-webframework
git add src/experiments/experiment_depth_gridsearch.py
git commit -m "feat: add depth grid search experiment for coarse localization"
```

---

### Task 3: Run end-to-end and verify

- [ ] **Step 1: Run Script 1 (DAP depth)**

```bash
conda run -n dap_env --cwd /tmp/DAP python /home/ruoyu/scan2measure-webframework/src/experiments/experiment_dap_depth.py
```

- [ ] **Step 2: Visually check DAP outputs**

Open any `data/pano/dap_depth/*_vis.png` — verify room structure is visible (walls should appear as depth gradients, not noise).

- [ ] **Step 3: Run Script 2 (grid search)**

```bash
cd /home/ruoyu/scan2measure-webframework
conda run -n scan_env python src/experiments/experiment_depth_gridsearch.py
```

- [ ] **Step 4: Check results**

```bash
cat data/experiments/depth_gridsearch/coarse_positions.json
```

Verify: each panorama has a position with score > 0 (ideally > 0.3). Scores near 0 or negative suggest the approach needs debugging.

- [ ] **Step 5: Open top-down visualization**

View `data/experiments/depth_gridsearch/topdown_result.png`. Camera positions should be spatially distributed across the building, not clustered in one spot.

- [ ] **Step 6: Commit data outputs (optional)**

Only if results look good:
```bash
cd /home/ruoyu/scan2measure-webframework
git add data/experiments/depth_gridsearch/coarse_positions.json
git add data/experiments/depth_gridsearch/topdown_result.png
git commit -m "results: depth grid search coarse localization — initial run"
```
