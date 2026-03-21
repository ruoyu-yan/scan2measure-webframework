"""Color sampling and multi-panorama blending for point cloud colorization.

Provides bilinear sampling from equirectangular images with horizontal
wraparound, and inverse-distance-weighted blending across panoramas.
"""

import numpy as np
from scipy.ndimage import map_coordinates


def sample_colors_bilinear(image, u, v):
    """Sample colors from equirectangular panorama with bilinear interpolation.

    Handles horizontal wraparound at the equirectangular seam by padding
    the image. Vertical edges (poles) are clamped.

    Args:
        image: (H, W, 3) uint8 RGB panorama
        u: (N,) float pixel x-coordinates
        v: (N,) float pixel y-coordinates

    Returns:
        colors: (N, 3) float64 RGB in [0, 1]
    """
    H, W = image.shape[:2]
    img_float = image.astype(np.float64) / 255.0

    # Pad 1 column on each side for horizontal wraparound
    padded = np.concatenate(
        [img_float[:, -1:, :], img_float, img_float[:, :1, :]], axis=1)
    u_padded = u + 1.0

    v_clamped = np.clip(v, 0, H - 1)

    coords = [v_clamped, u_padded]
    colors = np.stack([
        map_coordinates(padded[:, :, c], coords, order=1, mode='nearest')
        for c in range(3)
    ], axis=-1)

    return colors


def blend_colors_idw(per_pano_colors, per_pano_depths, per_pano_indices,
                     num_points, power=2.0):
    """Inverse-distance-weighted blending of colors from multiple panoramas.

    For each point, blends colors from all panoramas where it is visible,
    weighted by 1/depth^power. Closer panoramas contribute more, producing
    smooth transitions in overlap regions.

    Args:
        per_pano_colors: list of (M_k, 3) color arrays (one per panorama)
        per_pano_depths: list of (M_k,) depth arrays
        per_pano_indices: list of (M_k,) index arrays into the full point cloud
        num_points: total number of points
        power: exponent for inverse-distance weighting (2.0 = quadratic falloff)

    Returns:
        final_colors: (N, 3) float64 RGB in [0, 1]
        colored_mask: (N,) bool — True for points colored by at least one pano
    """
    color_accum = np.zeros((num_points, 3), dtype=np.float64)
    weight_accum = np.zeros(num_points, dtype=np.float64)

    for colors, depths, indices in zip(
            per_pano_colors, per_pano_depths, per_pano_indices):
        if len(indices) == 0:
            continue
        w = 1.0 / (depths ** power)
        color_accum[indices] += colors * w[:, np.newaxis]
        weight_accum[indices] += w

    colored_mask = weight_accum > 0
    final_colors = np.zeros((num_points, 3), dtype=np.float64)
    final_colors[colored_mask] = (
        color_accum[colored_mask] / weight_accum[colored_mask, np.newaxis]
    )

    return final_colors, colored_mask
