"""Pose estimation visualization — side-by-side 2D features vs projected 3D wireframe.

Uses sphere_geometry for all projections. No imports from panoramic-localization/.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_ROOT / "utils"))
from sphere_geometry import render_sphere_lines


# ============================================================
# Side-by-side (2D features | projected 3D wireframe)
# ============================================================

def render_side_by_side(pano_img, edge_2d, starts_3d, ends_3d, R, t,
                        resolution=(512, 1024)):
    """Left: panorama + 2D lines (green). Right: panorama + projected 3D wireframe (cyan).

    Uses equirectangular projection throughout (not perspective crops).

    Args:
        pano_img: (H, W, 3) uint8 RGB
        edge_2d: (N, 9) tensor — 2D line segments
        starts_3d: (M, 3) ndarray — 3D wireframe start points
        ends_3d:   (M, 3) ndarray — 3D wireframe end points
        R: (3, 3) ndarray — rotation
        t: (3,) ndarray — translation
        resolution: (H, W)

    Returns:
        (H, 2*W + margin, 3) uint8 RGB
    """
    H, W = resolution
    pano = cv2.resize(pano_img, (W, H))

    # Left: 2D lines in green
    left = pano.copy()
    if edge_2d is not None and edge_2d.shape[0] > 0:
        green = torch.tensor([[0, 0.85, 0]]).expand(edge_2d.shape[0], -1)
        line_img = render_sphere_lines(edge_2d, resolution=(H, W), rgb=green)
        mask = line_img.sum(axis=-1) > 0
        left[mask] = line_img[mask]

    # Right: 3D wireframe in cyan
    right = pano.copy()
    if starts_3d.shape[0] > 0:
        # Project 3D lines to sphere and render
        s_cam = (starts_3d - t[np.newaxis, :]) @ R.T
        e_cam = (ends_3d - t[np.newaxis, :]) @ R.T
        s_sph = s_cam / np.maximum(np.linalg.norm(s_cam, axis=-1, keepdims=True), 1e-6)
        e_sph = e_cam / np.maximum(np.linalg.norm(e_cam, axis=-1, keepdims=True), 1e-6)

        normals = np.cross(s_sph, e_sph)
        n_norm = np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
        normals = normals / n_norm

        edges_3d_sphere = torch.tensor(
            np.concatenate([normals, s_sph, e_sph], axis=-1),
            dtype=torch.float32)
        cyan = torch.tensor([[0, 0.85, 0.85]]).expand(edges_3d_sphere.shape[0], -1)
        wire_img = render_sphere_lines(edges_3d_sphere, resolution=(H, W), rgb=cyan)
        mask = wire_img.sum(axis=-1) > 0
        right[mask] = wire_img[mask]

    # Compose with divider
    margin = 4
    divider = np.full((H, margin, 3), 200, dtype=np.uint8)
    panel = np.hstack([left, divider, right])

    # Labels
    cv2.putText(panel, "2D lines (green)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "3D wireframe (cyan)", (W + margin + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)

    return panel


# ============================================================
# Reprojection check (point cloud depth overlay on panorama)
# ============================================================

def render_reprojection(pano_img, points_world, R, t, resolution=(1024, 2048)):
    """Project point cloud onto panorama, color-coded by depth.

    Uses sphere_to_equirect convention: theta = atan2(norm(xy), z),
    phi = atan2(y, x) + pi.  Camera frame: x=forward, y=left, z=up.

    Args:
        pano_img: (H, W, 3) uint8 RGB panorama
        points_world: (N, 3) ndarray — 3D points in world frame (pre-subsampled)
        R: (3, 3) ndarray — rotation matrix
        t: (3,) ndarray — camera translation in world frame
        resolution: (H, W) output image size

    Returns:
        (H, W, 3) uint8 RGB image
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    H, W = resolution
    pano = cv2.resize(pano_img, (W, H))

    # Transform to camera frame (same convention as render_side_by_side line 54)
    p_cam = (points_world - t[np.newaxis, :]) @ R.T

    # Filter points too close to camera
    dists = np.linalg.norm(p_cam, axis=1)
    valid = dists > 0.1
    p_cam = p_cam[valid]
    dists = dists[valid]

    if p_cam.shape[0] == 0:
        return pano

    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]

    # Equirectangular projection (matching sphere_geometry.sphere_to_equirect)
    norm_xy = np.sqrt(x**2 + y**2)
    theta = np.arctan2(norm_xy, z)        # polar from z-axis: 0..pi
    phi = np.arctan2(y, x) + np.pi        # azimuthal: 0..2pi

    u = (1.0 - phi / (2 * np.pi)) * W
    v = (theta / np.pi) * H

    # Depth normalization (2nd-98th percentile)
    d_lo, d_hi = np.percentile(dists, [2, 98])
    d_norm = np.clip((dists - d_lo) / (d_hi - d_lo + 1e-6), 0, 1)

    # Render with matplotlib for nice scatter + colorbar
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), dpi=102)
    ax.imshow(pano)
    order = np.argsort(-dists)
    ax.scatter(u[order], v[order], c=d_norm[order], cmap="turbo",
               s=1.5, alpha=0.6, edgecolors="none")
    cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label(f"Depth ({d_lo:.1f}m - {d_hi:.1f}m)", fontsize=10)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title("Point Cloud Reprojection on Panorama (depth-coded)", fontsize=14)
    ax.axis("off")

    # Render figure to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img


# ============================================================
# Top-down view (camera on density image + floorplan)
# ============================================================

def _world_to_density_pixel(point_world, density_meta):
    """Map a 3D world point to density image pixel coordinates.

    Replicates generate_density_image.py transform:
    rotate → mm scale → z-flip → pixel projection.
    """
    rot = np.array(density_meta["rotation_matrix"])
    min_coords = np.array(density_meta["min_coords"])
    max_dim = density_meta["max_dim"]
    offset = np.array(density_meta["offset"])
    img_w = density_meta["image_width"]
    img_h = density_meta["image_height"]

    p = rot @ point_world

    # Conditional mm scaling (generate_density_image: if max(extents) < 500: *= 1000)
    if np.max(np.abs(min_coords)) > 500:
        p = p * 1000.0

    # Axis flip: [+x, +y, -z]
    p[2] = -p[2]

    px = (p[0] - min_coords[0] + offset[0]) / max_dim * (img_w - 1)
    py = (p[1] - min_coords[1] + offset[1]) / max_dim * (img_h - 1)
    return px, py


def render_topdown(density_img, density_meta, camera_t, camera_R,
                   room_polygons=None, resolution=800):
    """Camera position + heading on density image with optional polygon overlay.

    Args:
        density_img: (H, W) uint8 grayscale density image
        density_meta: dict from metadata.json
        camera_t: (3,) ndarray — camera position in world frame
        camera_R: (3, 3) ndarray — rotation matrix
        room_polygons: list of (K, 2) arrays — polygon vertices in density pixel coords
        resolution: output image height (width scaled proportionally)

    Returns:
        (H, W, 3) uint8 RGB image
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img_w = density_meta["image_width"]
    img_h = density_meta["image_height"]

    # Camera pixel position in density image coords
    cx, cy = _world_to_density_pixel(camera_t, density_meta)

    # Camera forward direction: +x axis in camera frame → world frame
    forward_world = camera_R.T @ np.array([1.0, 0.0, 0.0])
    rot = np.array(density_meta["rotation_matrix"])
    min_coords = np.array(density_meta["min_coords"])
    max_dim = density_meta["max_dim"]
    fwd_rot = rot @ forward_world
    if np.max(np.abs(min_coords)) > 500:
        fwd_rot = fwd_rot * 1000.0
    fwd_2d = np.array([fwd_rot[0], fwd_rot[1]])
    fwd_len = np.linalg.norm(fwd_2d)
    if fwd_len > 1e-6:
        fwd_2d = fwd_2d / fwd_len
    arrow_scale = 20.0
    dx = fwd_2d[0] / max_dim * (img_w - 1) * arrow_scale
    dy = fwd_2d[1] / max_dim * (img_h - 1) * arrow_scale

    # Render with matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
    ax.imshow(density_img, cmap="gray", origin="upper")

    # RoomFormer polygons
    if room_polygons is not None:
        for poly in room_polygons:
            poly_arr = np.array(poly)
            poly_closed = np.vstack([poly_arr, poly_arr[0:1]])
            ax.plot(poly_closed[:, 0], poly_closed[:, 1], "-", color="lime",
                    linewidth=2, alpha=0.8)
            ax.fill(poly_arr[:, 0], poly_arr[:, 1], alpha=0.1, color="lime")

    # Camera position
    ax.plot(cx, cy, "o", color="red", markersize=12, markeredgecolor="white",
            markeredgewidth=2, zorder=10)

    # Camera heading arrow
    ax.annotate("", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5),
                zorder=11)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="gray", edgecolor="gray", label="Density map"),
        plt.Line2D([0], [0], color="lime", lw=2, label="RoomFormer polygon"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=10, label="Camera position"),
        plt.Line2D([0], [0], color="yellow", lw=2, label="Camera forward"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.8)

    ax.set_title("Top-Down View: Camera on Density Map + Floorplan", fontsize=13)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis("off")

    # Render figure to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img


# ============================================================
# Composite top-down view (all cameras on one image)
# ============================================================

# Distinct colors per camera (up to 8)
_CAMERA_COLORS = [
    "#e63946",  # red
    "#2a9d8f",  # teal
    "#e9c46a",  # gold
    "#264653",  # dark blue
    "#f4a261",  # orange
    "#6a4c93",  # purple
    "#1d3557",  # navy
    "#457b9d",  # steel blue
]


def render_topdown_composite(density_img, density_meta, cameras, room_polygons=None,
                             resolution=800):
    """All camera positions + headings overlaid on a single density image.

    Args:
        density_img: (H, W) uint8 grayscale density image
        density_meta: dict from metadata.json
        cameras: list of dicts with keys 'name', 't' (3,), 'R' (3,3)
        room_polygons: list of (K, 2) arrays — polygon vertices in density pixel coords
        resolution: output image height

    Returns:
        (H, W, 3) uint8 RGB image
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img_w = density_meta["image_width"]
    img_h = density_meta["image_height"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
    ax.imshow(density_img, cmap="gray", origin="upper")

    # RoomFormer polygons
    if room_polygons is not None:
        for poly in room_polygons:
            poly_arr = np.array(poly)
            poly_closed = np.vstack([poly_arr, poly_arr[0:1]])
            ax.plot(poly_closed[:, 0], poly_closed[:, 1], "-", color="lime",
                    linewidth=2, alpha=0.8)
            ax.fill(poly_arr[:, 0], poly_arr[:, 1], alpha=0.1, color="lime")

    legend_elements = [
        mpatches.Patch(facecolor="gray", edgecolor="gray", label="Density map"),
        plt.Line2D([0], [0], color="lime", lw=2, label="Room polygon"),
    ]

    for i, cam in enumerate(cameras):
        color = _CAMERA_COLORS[i % len(_CAMERA_COLORS)]
        camera_t = np.asarray(cam['t'])
        camera_R = np.asarray(cam['R'])

        cx, cy = _world_to_density_pixel(camera_t, density_meta)

        # Camera forward direction
        forward_world = camera_R.T @ np.array([1.0, 0.0, 0.0])
        rot = np.array(density_meta["rotation_matrix"])
        min_coords = np.array(density_meta["min_coords"])
        max_dim = density_meta["max_dim"]
        fwd_rot = rot @ forward_world
        if np.max(np.abs(min_coords)) > 500:
            fwd_rot = fwd_rot * 1000.0
        fwd_2d = np.array([fwd_rot[0], fwd_rot[1]])
        fwd_len = np.linalg.norm(fwd_2d)
        if fwd_len > 1e-6:
            fwd_2d = fwd_2d / fwd_len
        arrow_scale = 20.0
        dx = fwd_2d[0] / max_dim * (img_w - 1) * arrow_scale
        dy = fwd_2d[1] / max_dim * (img_h - 1) * arrow_scale

        ax.plot(cx, cy, "o", color=color, markersize=12,
                markeredgecolor="white", markeredgewidth=2, zorder=10)
        ax.annotate("", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5),
                    zorder=11)
        # Label near the dot
        ax.text(cx + 8, cy - 8, cam['name'], fontsize=8, color=color,
                fontweight="bold", zorder=12,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        legend_elements.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markersize=8, label=cam['name'])
        )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.8)
    ax.set_title("All Camera Poses on Density Map", fontsize=13)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis("off")

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img


# ============================================================
# Standalone entry point
# ============================================================

if __name__ == "__main__":
    import json
    import open3d as o3d

    POINT_CLOUD_NAME = "tmb_office1"
    ROOM_NAME = "TMB_office1"

    ROOT = Path(__file__).resolve().parent.parent
    POSE_PATH = ROOT / "data" / "pose_estimates" / ROOM_NAME / "camera_pose.json"
    PANO_PATH = ROOT / "data" / "pano" / "raw" / f"{ROOM_NAME}.jpg"
    PC_PATH = ROOT / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
    DENSITY_IMG_PATH = ROOT / "data" / "density_image" / POINT_CLOUD_NAME / f"{POINT_CLOUD_NAME}.png"
    DENSITY_META_PATH = ROOT / "data" / "density_image" / POINT_CLOUD_NAME / "metadata.json"
    ROOMFORMER_PATH = ROOT / "data" / "reconstructed_floorplans_RoomFormer" / POINT_CLOUD_NAME / "predictions.json"
    VIS_DIR = ROOT / "data" / "pose_estimates" / ROOM_NAME / "vis"
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load pose
    print("Loading camera_pose.json...")
    with open(POSE_PATH) as f:
        pose = json.load(f)
    R = np.array(pose["rotation"])
    t = np.array(pose["translation"])
    print(f"  t = {t}")

    # --- Reprojection ---
    print("Generating reprojection.png...")
    pano_img = cv2.imread(str(PANO_PATH))
    pano_rgb = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)

    pcd = o3d.io.read_point_cloud(str(PC_PATH))
    pts = np.asarray(pcd.points)
    rng = np.random.default_rng(42)
    if pts.shape[0] > 50000:
        pts = pts[rng.choice(pts.shape[0], 50000, replace=False)]

    reproj = render_reprojection(pano_rgb, pts, R, t)
    cv2.imwrite(str(VIS_DIR / "reprojection.png"),
                cv2.cvtColor(reproj, cv2.COLOR_RGB2BGR))
    print(f"  Saved {VIS_DIR / 'reprojection.png'}")

    # --- Top-down ---
    print("Generating topdown.png...")
    density_img_raw = cv2.imread(str(DENSITY_IMG_PATH), cv2.IMREAD_GRAYSCALE)
    with open(DENSITY_META_PATH) as f:
        density_meta = json.load(f)

    room_polygons = None
    if ROOMFORMER_PATH.exists():
        with open(ROOMFORMER_PATH) as f:
            room_polygons = json.load(f)

    topdown = render_topdown(density_img_raw, density_meta, t, R,
                             room_polygons=room_polygons)
    cv2.imwrite(str(VIS_DIR / "topdown.png"),
                cv2.cvtColor(topdown, cv2.COLOR_RGB2BGR))
    print(f"  Saved {VIS_DIR / 'topdown.png'}")

    print("Done.")
