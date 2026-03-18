# Phase 2.4 — Pose Estimation Visualization Improvement

## Goal

Add two new visualizations to validate camera pose correctness beyond the existing feature-matching overlay (`side_by_side.png`). Both are integrated into `visualize_pose.py` as library functions and callable standalone.

## Approach

Option C: library functions in `visualize_pose.py` + standalone `__main__` entry point.

## Visualizations

### 1. Reprojection check (`reprojection.png`)

Project point cloud points onto the panorama using the estimated camera pose, color-coded by depth.

**Function**: `render_reprojection(pano_img, points_world, R, t, resolution=(1024, 2048))`

**Returns**: `(H, W, 3)` uint8 RGB image

**Algorithm**:
1. Transform to camera frame: `p_cam = (pts - t) @ R.T`
2. Filter points closer than 0.1m
3. Equirectangular projection matching `sphere_geometry.sphere_to_equirect`:
   - `theta = atan2(norm(xy), z)` — polar angle from z-axis
   - `phi = atan2(y, x) + pi` — azimuthal angle
   - `u = (1 - phi/(2*pi)) * (W-1)`
   - `v = (theta/pi) * (H-1)`
4. Color by depth (turbo colormap, 2nd–98th percentile range)
5. Render far-to-near so closer points draw on top
6. Composite onto resized panorama using OpenCV (no matplotlib)

**Caller responsibility**: subsample point cloud to ~50K before passing in.

### 2. Top-down view (`topdown.png`)

Camera position and orientation plotted on the density image with RoomFormer polygon overlay.

**Function**: `render_topdown(density_img, density_metadata, camera_t, camera_R, room_polygons=None, resolution=800)`

**Returns**: `(H, W, 3)` uint8 RGB image

**Algorithm**:
1. World-to-pixel transform from density metadata (must match `generate_density_image.py`):
   - Rotate world point by `rotation_matrix`
   - Conditionally scale to mm: if `max(abs(min_coords)) > 500`, multiply by 1000
   - Apply axis flip: `[+x, +y, -z]` (z-axis negation from density pipeline)
   - Project to pixel: `px = (x - min_coords[0] + offset[0]) / max_dim * (img_w - 1)`, same for py
2. Map camera position (`camera_t`) to density image pixel using the above transform
3. Derive forward direction: camera forward = +x axis in camera frame (panorama center maps to +x per `sphere_to_equirect`). In world frame: `forward_world = R.T @ [1, 0, 0]`. Transform to density image space using the same rotate/flip pipeline, take 2D projection for the arrow direction
4. Draw camera as colored dot + direction arrow
5. If `room_polygons` provided, overlay as green outlines (vertices already in density image pixel coords)
6. Resize to output resolution, dark background

## File changes

### `visualize_pose.py`

**Add**:
- `render_reprojection()` function
- `render_topdown()` function
- `__main__` block: loads camera_pose.json, point cloud, panorama, density image, metadata, RoomFormer polygons. Generates all three PNGs (`side_by_side.png`, `reprojection.png`, `topdown.png`) to `data/pose_estimates/<ROOM_NAME>/vis/`.

**Existing**: `render_side_by_side()` unchanged.

**Note**: `__main__` generates `reprojection.png` and `topdown.png` only. `side_by_side.png` is omitted because it requires 2D/3D feature data (`fgpl_features.json`, `3d_line_map.pkl`) that adds significant loading complexity. The pipeline generates all three.

### `pose_estimation_pipeline.py`

**Stage [10] additions** (after existing `render_side_by_side` call):
1. Load point cloud PLY, subsample to 50K, call `render_reprojection`, save `reprojection.png`
2. Load density image PNG + metadata.json + RoomFormer predictions.json, call `render_topdown`, save `topdown.png`

**New path constants** at top of file:
- `POINT_CLOUD_PATH` — `data/raw_point_cloud/{POINT_CLOUD_NAME}.ply`
- `DENSITY_IMG_PATH` — `data/density_image/{POINT_CLOUD_NAME}/{POINT_CLOUD_NAME}.png`
- `DENSITY_META_PATH` — `data/density_image/{POINT_CLOUD_NAME}/metadata.json`
- `ROOMFORMER_PATH` — `data/reconstructed_floorplans_RoomFormer/{POINT_CLOUD_NAME}/predictions.json`

### Cleanup

Delete after implementation validated:
- `src/proto_viz_reprojection.py`
- `src/proto_viz_topdown.py`
- `src/proto_viz_3d_overlay.py`

## Data dependencies

| Input | Source | Used by |
|-------|--------|---------|
| `camera_pose.json` | `pose_estimation_pipeline.py` | Both |
| `tmb_office1.ply` | Raw scan data | Reprojection |
| `TMB_office1.jpg` | Raw panorama | Reprojection, side-by-side |
| `tmb_office1.png` + `metadata.json` | `generate_density_image.py` | Top-down |
| `predictions.json` | `RoomFormer_inference.py` | Top-down (optional) |

## Output

All PNGs written to `data/pose_estimates/<ROOM_NAME>/vis/`:
- `side_by_side.png` — existing feature alignment overlay
- `reprojection.png` — depth-colored point cloud on panorama
- `topdown.png` — camera on floorplan
