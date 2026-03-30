"""Equirectangular panorama to cubemap face conversion.

Converts 360-degree equirectangular panoramic images into 6 pinhole cubemap
face images for use with Open3D's project_images_to_albedo(), which only
supports pinhole camera models.

Coordinate convention matches projection.py:
    Camera frame: Z = up, equirectangular mapping via
        theta = arctan2(norm(xy), z)   — polar from z-axis, [0, pi]
        phi   = arctan2(y, x) + pi    — azimuthal, [0, 2*pi]
        u     = (1 - phi/(2*pi)) * W
        v     = (theta/pi) * H

Face-local frame (standard pinhole): Z = forward, X = right, Y = down.
"""

import numpy as np
import cv2


# ── Face rotation matrices ──────────────────────────────────────────────────
# Each R_face_to_cam maps face-local (Z=forward, X=right, Y=down) to
# panorama camera frame (Z=up). Columns are the face-local basis vectors
# expressed in camera-frame coordinates.

FACE_ROTATIONS = [
    # +X: looking along camera +X (center of equirect)
    np.array([[0, 0, 1],
              [-1, 0, 0],
              [0, -1, 0]], dtype=np.float64),
    # -X: looking along camera -X (left/right edges of equirect)
    np.array([[0, 0, -1],
              [1, 0, 0],
              [0, -1, 0]], dtype=np.float64),
    # +Y: looking along camera +Y
    np.array([[1, 0, 0],
              [0, 0, 1],
              [0, -1, 0]], dtype=np.float64),
    # -Y: looking along camera -Y
    np.array([[-1, 0, 0],
              [0, 0, -1],
              [0, -1, 0]], dtype=np.float64),
    # +Z: looking up (zenith, top of equirect)
    np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]], dtype=np.float64),
    # -Z: looking down (nadir, bottom of equirect)
    np.array([[1, 0, 0],
              [0, -1, 0],
              [0, 0, -1]], dtype=np.float64),
]

FACE_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


# ── Remap table construction ────────────────────────────────────────────────

def _build_remap_tables(face_size, R_face_to_cam, pano_w, pano_h):
    """Precompute remap tables from cubemap face pixels to equirectangular.

    For each pixel (px, py) in the face image, computes the corresponding
    (u, v) coordinates in the equirectangular panorama.

    Args:
        face_size: side length of the square cubemap face
        R_face_to_cam: (3, 3) rotation from face-local to camera frame
        pano_w: panorama width in pixels
        pano_h: panorama height in pixels

    Returns:
        map_x: (face_size, face_size) float32 — pano x-coordinates
        map_y: (face_size, face_size) float32 — pano y-coordinates
    """
    f = face_size / 2.0                     # focal length for 90-degree FOV
    cx = (face_size - 1) / 2.0
    cy = (face_size - 1) / 2.0

    px, py = np.meshgrid(np.arange(face_size, dtype=np.float64),
                         np.arange(face_size, dtype=np.float64))

    # Ray directions in face-local frame (Z=forward, X=right, Y=down)
    dir_x = (px - cx) / f
    dir_y = (py - cy) / f
    dir_z = np.ones_like(dir_x)

    norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
    dir_x /= norm
    dir_y /= norm
    dir_z /= norm

    # Transform to camera frame: row-vector @ R^T = column-vector convention R @ col
    dirs_local = np.stack([dir_x, dir_y, dir_z], axis=-1)       # (H, W, 3)
    dirs_cam = dirs_local @ R_face_to_cam.T                      # (H, W, 3)

    # Equirectangular projection (matches projection.py convention)
    x = dirs_cam[..., 0]
    y = dirs_cam[..., 1]
    z = dirs_cam[..., 2]

    theta = np.arctan2(np.sqrt(x**2 + y**2), z)                 # [0, pi]
    phi = np.arctan2(y, x) + np.pi                               # [0, 2*pi]

    u = (1.0 - phi / (2 * np.pi)) * pano_w
    v = (theta / np.pi) * pano_h

    # Wrap horizontal, clamp vertical
    map_x = (u % pano_w).astype(np.float32)
    map_y = np.clip(v, 0, pano_h - 1).astype(np.float32)

    return map_x, map_y


# ── Public API ───────────────────────────────────────────────────────────────

def equirect_to_cubemap_faces(pano_image, face_size=1024):
    """Convert an equirectangular panorama to 6 cubemap face images.

    Args:
        pano_image: (H, W, 3) uint8 BGR or RGB panoramic image
        face_size: output face resolution (square)

    Returns:
        faces: list of 6 (face_size, face_size, 3) uint8 images
    """
    pano_h, pano_w = pano_image.shape[:2]
    faces = []
    for R in FACE_ROTATIONS:
        map_x, map_y = _build_remap_tables(face_size, R, pano_w, pano_h)
        face = cv2.remap(pano_image, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        faces.append(face)
    return faces


def build_cubemap_cameras(pano_R, pano_t, face_size):
    """Build intrinsic and extrinsic matrices for 6 cubemap faces.

    Args:
        pano_R: (3, 3) world-to-camera rotation (from pose JSON)
        pano_t: (3,) camera position in world frame (meters)
        face_size: cubemap face resolution (for intrinsic computation)

    Returns:
        intrinsics: list of 6 (3, 3) float64 intrinsic matrices (all identical)
        extrinsics: list of 6 (4, 4) float64 extrinsic matrices
    """
    # Intrinsic: 90-degree FOV pinhole
    f = face_size / 2.0
    cx = (face_size - 1) / 2.0
    cy = (face_size - 1) / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)

    intrinsics = []
    extrinsics = []

    for R_face_to_cam in FACE_ROTATIONS:
        # World → face-local camera frame:
        # 1. World → pano-cam: R_pano @ (p - t)
        # 2. Pano-cam → face-local: R_face_to_cam^T @ p_cam
        # Combined rotation: R_face_to_cam^T @ R_pano
        R_ext = R_face_to_cam.T @ pano_R
        t_ext = -R_ext @ pano_t

        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = R_ext
        ext[:3, 3] = t_ext

        intrinsics.append(K.copy())
        extrinsics.append(ext)

    return intrinsics, extrinsics
