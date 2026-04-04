# Depth Grid Search for Coarse Panorama Localization

**Date:** 2026-04-04
**Status:** Approved
**Goal:** Replace SAM3 + polygon matching with a faster depth-based approach for finding approximate camera positions of panoramas within an uncolored TLS point cloud.

## Motivation

The current coarse localization pipeline (SAM3 room segmentation + polygon matching) takes ~350 seconds for 3 panoramas in 3 rooms. The polygon matching uses differential evolution over all possible room assignments, which is slow and brittle. This experiment tests a simpler approach: render depth panoramas from the point cloud at a grid of candidate positions, compare against monocular depth estimated from the real panoramas, and pick the best match.

## Architecture

Two scripts, two conda environments:

```
Script 1 (dap_env):   panoramas → DAP model → depth .npy files
Script 2 (scan_env):  point cloud → mesh → candidate grid → raycast depth
                      → compare vs DAP depth → best position per pano
                      → top-down visualization
```

## Script 1: `experiment_dap_depth.py`

**Conda env:** `dap_env` (Python 3.12, PyTorch 2.7.1)
**Run:** `conda run -n dap_env --cwd /tmp/DAP python /home/ruoyu/scan2measure-webframework/src/experiments/experiment_dap_depth.py`

**Input:** Panorama JPGs from `data/pano/raw/`

**Output:**
- `data/pano/dap_depth/{PANO_NAME}.npy` — float32 (512, 1024) depth map
- `data/pano/dap_depth/{PANO_NAME}_vis.png` — colorized depth for sanity checking

**Logic:**
1. Load DAP model from HuggingFace cache (`~/.cache/huggingface/hub/models--Insta360-Research--DAP-weights/`)
2. DAP repo path: `/tmp/DAP` (cloned from `github.com/Insta360-Research-Team/DAP`)
3. For each panorama: read JPG → resize to 512x1024 → normalize to [0,1] → forward pass → save raw prediction as `.npy` + colorized PNG
4. Model config: `midas_model_type=vitl`, `fine_tune_type=hypersim`, `min_depth=0.01`, `max_depth=1.0`

**Constants:**
```python
PANO_NAMES = [
    "TMB_corridor_north1", "TMB_corridor_north2",
    "TMB_corridor_north3", "TMB_corridor_north4",
    "TMB_corridor_south1", "TMB_corridor_south2",
    "TMB_hall1", "TMB_office1",
]
```

## Script 2: `experiment_depth_gridsearch.py`

**Conda env:** `scan_env` (Python 3.8, open3d 0.19.0)
**Run:** `conda run -n scan_env python src/experiments/experiment_depth_gridsearch.py`

**Input:**
- Point cloud: `data/raw_point_cloud/tmb_scan_subsampled_subsampled_no_RGB.ply`
- DAP depth maps: `data/pano/dap_depth/{PANO_NAME}.npy`

**Output:**
- `data/experiments/depth_gridsearch/coarse_positions.json`
- `data/experiments/depth_gridsearch/topdown_result.png`
- `data/experiments/depth_gridsearch/depth_comparison_{PANO_NAME}.png` (per-pano debug viz)

### Step-by-step logic

**Step 1 — Mesh the point cloud (one-time, cached):**
- Voxel downsample to 2cm
- Estimate normals (KDTree hybrid, radius=0.1, max_nn=30)
- Orient normals consistently
- Poisson reconstruction at depth=9
- Trim low-density vertices (bottom 5th percentile)
- Cache mesh to `data/experiments/depth_gridsearch/mesh_cache.ply`
- Skip if cache exists

**Step 2 — Build candidate grid:**
- Get point cloud bounding box (XY extent)
- Generate 2D grid at 0.5m spacing
- Set Z = floor_z + CAMERA_HEIGHT (1.5m)
- Floor Z detected as the 5th percentile of point cloud Z values
- Filter candidates inside walls: raycast downward from each candidate, keep only those with a floor hit within reasonable distance (0.5-3.0m)
- Expected: ~200-400 candidates for 80m² apartment

**Step 3 — Raycast equirectangular depth:**
- Build `o3d.t.geometry.RaycastingScene` from mesh
- For each candidate position, generate equirectangular ray directions at 512x1024 resolution:
  - `theta = (v + 0.5) / H * pi` (elevation, 0=top to pi=bottom)
  - `phi = (u + 0.5) / W * 2 * pi` (azimuth)
  - `dx = sin(theta) * sin(phi)`, `dy = cos(theta)`, `dz = sin(theta) * cos(phi)`
  - (Adjust to match point cloud coordinate convention — Z-up)
- `scene.cast_rays()` returns `t_hit` as depth per pixel
- Store rendered depth maps as float32 numpy arrays

**Step 4 — Compare depths:**
- For each (panorama, candidate) pair:
  - Load DAP depth (512x1024) and rendered depth (512x1024)
  - Create valid mask: `t_hit < inf` AND `dap_depth > 0` AND exclude top/bottom 10% rows (ceiling/floor poles are unreliable)
  - If valid pixel count < 1000, skip (candidate is outside the scan)
  - Compute Pearson correlation of log-depth:
    ```python
    log_r = log(rendered[mask])
    log_e = log(dap[mask])
    log_r = (log_r - mean) / std
    log_e = (log_e - mean) / std
    score = corrcoef(log_r, log_e)
    ```
  - Also compute AbsRel after median scale alignment as secondary metric

**Step 5 — Select best positions:**
- Per panorama: rank candidates by correlation score, pick the best
- No multi-pano constraint enforcement in this experiment (keep it simple — just evaluate per-pano independently)
- Report score and position for each panorama

**Step 6 — Visualize:**
- **Top-down result**: Project point cloud to XY plane (density image style), overlay colored dots at estimated positions, label with pano names. Use matplotlib, save as PNG.
- **Per-pano debug**: Side-by-side showing DAP depth (left), best-matching rendered depth (middle), and correlation heatmap across all candidates on the floorplan (right).

### Constants

```python
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
```

### Output JSON format

```json
{
  "metadata": {
    "pipeline": "depth-gridsearch-v1",
    "point_cloud": "tmb_scan_subsampled_subsampled_no_RGB",
    "grid_spacing": 0.5,
    "camera_height": 1.5,
    "n_candidates_tested": 312,
    "total_time_seconds": 45.2
  },
  "positions": {
    "TMB_office1": {
      "position": [2.3, 4.1, 1.5],
      "score": 0.87,
      "absrel": 0.12
    },
    "TMB_corridor_north1": {
      "position": [6.7, 1.2, 1.5],
      "score": 0.72,
      "absrel": 0.18
    }
  }
}
```

## Key risks

1. **DAP depth quality on real TLS panoramas** — DAP was trained on internet panoramas, not TLS-captured ones. If depth estimation is poor, correlation scores will be low everywhere. The debug visualization will reveal this.

2. **Coordinate convention mismatch** — The equirectangular ray directions must match the panorama's orientation convention (which way is "forward"). May need to rotate the rendered depth horizontally to align. The correlation metric is rotation-invariant if we search over horizontal shifts (just a 1D circular shift of the rendered depth).

3. **Furniture/clutter** — Both the point cloud and panorama contain furniture, so this should be consistent. Unlike color-based methods, depth is less affected by lighting changes.

4. **Scale ambiguity** — DAP claims metric depth but output range is [0, 1] with max_depth=1.0 in config. This likely means depth is normalized. Using log-correlation handles this gracefully.

## Success criteria

- At least 6/8 panoramas correctly assigned to the right room (position within 2m of ground truth)
- Total runtime under 60 seconds (excluding one-time mesh caching)
- Clear visual confirmation in the top-down plot
