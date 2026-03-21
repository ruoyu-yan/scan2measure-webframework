"""Equirectangular projection utilities for point cloud colorization.

Projects 3D world points onto equirectangular panoramic images.

Coordinate convention matches sphere_geometry.sphere_to_equirect:
    theta = atan2(norm(xy), z)   — polar from z-axis, [0, pi]
    phi   = atan2(y, x) + pi    — azimuthal, [0, 2*pi]
    u     = (1 - phi/(2*pi)) * W
    v     = (theta/pi) * H
"""

import numpy as np


def world_to_camera(points_world, R, t):
    """Transform world-frame points to camera frame.

    Args:
        points_world: (N, 3) world coordinates
        R: (3, 3) world-to-camera rotation matrix
        t: (3,) camera position in world frame

    Returns:
        (N, 3) camera-frame coordinates
    """
    return (points_world - t[np.newaxis, :]) @ R.T


def camera_to_equirect(p_cam, img_w, img_h):
    """Project camera-frame points to equirectangular pixel coordinates.

    Args:
        p_cam: (N, 3) camera-frame points
        img_w: panorama width in pixels
        img_h: panorama height in pixels

    Returns:
        u: (N,) float pixel x-coordinates [0, W]
        v: (N,) float pixel y-coordinates [0, H]
        depths: (N,) Euclidean distances from camera
    """
    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    depths = np.linalg.norm(p_cam, axis=1)

    norm_xy = np.sqrt(x**2 + y**2)
    theta = np.arctan2(norm_xy, z)          # polar: 0..pi
    phi = np.arctan2(y, x) + np.pi          # azimuthal: 0..2pi

    u = (1.0 - phi / (2 * np.pi)) * img_w
    v = (theta / np.pi) * img_h

    return u, v, depths


def project_points_to_pano(points_world, R, t, img_w, img_h, min_depth=0.1):
    """Project world points to equirectangular coordinates with depth filtering.

    Args:
        points_world: (N, 3) world coordinates
        R: (3, 3) rotation matrix
        t: (3,) camera translation
        img_w, img_h: panorama dimensions
        min_depth: discard points closer than this (meters)

    Returns:
        u: (M,) pixel x-coordinates (valid points only)
        v: (M,) pixel y-coordinates
        depths: (M,) distances from camera
        valid_indices: (M,) indices into original points_world
    """
    p_cam = world_to_camera(points_world, R, t)
    depths = np.linalg.norm(p_cam, axis=1)

    valid_mask = depths > min_depth
    valid_indices = np.where(valid_mask)[0]

    p_cam = p_cam[valid_mask]
    depths = depths[valid_mask]

    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    norm_xy = np.sqrt(x**2 + y**2)
    theta = np.arctan2(norm_xy, z)
    phi = np.arctan2(y, x) + np.pi

    u = (1.0 - phi / (2 * np.pi)) * img_w
    v = (theta / np.pi) * img_h

    return u, v, depths, valid_indices
