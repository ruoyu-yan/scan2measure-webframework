"""Depth-buffer occlusion for equirectangular panoramas.

Determines which 3D points are visible from a camera position by building
a low-resolution depth buffer and keeping only frontmost points per pixel.
"""

import numpy as np


def compute_visibility_depth_buffer(u, v, depths, img_w, img_h,
                                     buffer_w=2048, buffer_h=1024,
                                     depth_margin=0.05):
    """Depth-buffer visibility test for equirectangular projection.

    Rasterizes points to a low-resolution depth buffer, then marks a point
    as visible if its depth is within depth_margin of the minimum depth at
    that pixel.

    Args:
        u: (N,) float pixel x-coordinates in panorama space
        v: (N,) float pixel y-coordinates in panorama space
        depths: (N,) Euclidean distances from camera
        img_w: full panorama width
        img_h: full panorama height
        buffer_w: depth buffer width (lower = faster, coarser)
        buffer_h: depth buffer height
        depth_margin: tolerance in meters (5cm default — keeps co-planar surfaces)

    Returns:
        visible_mask: (N,) boolean mask
    """
    # Quantize to buffer resolution
    bu = np.clip((u * buffer_w / img_w).astype(np.int32), 0, buffer_w - 1)
    bv = np.clip((v * buffer_h / img_h).astype(np.int32), 0, buffer_h - 1)

    # Build min-depth buffer
    depth_buf = np.full((buffer_h, buffer_w), np.inf, dtype=np.float64)
    np.minimum.at(depth_buf, (bv, bu), depths)

    # Point is visible if within margin of the frontmost depth at its pixel
    visible_mask = depths <= depth_buf[bv, bu] + depth_margin

    return visible_mask
