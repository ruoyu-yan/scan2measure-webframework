"""Panoramic LSD line detection with sphere back-projection.

Reimplements the HorizonNet-based line extraction pipeline from panoramic-localization/edge_utils.py.
No PICCOLO rotation is applied — coordinates match the raw equirectangular frame.
"""

import numpy as np
import cv2
from scipy.ndimage import map_coordinates
from pylsd import lsd


# ---------------------------------------------------------------------------
# Coordinate helpers (HorizonNet-specific math, kept private)
# ---------------------------------------------------------------------------

def _xyz_to_uv(xyz, plane_id=1):
    """3D sphere point → planar UV on one of 3 canonical planes.

    plane_id 1 = XY, 2 = YZ, 3 = ZX (cyclic permutation of axes).
    Returns (N, 2) array with u ∈ [-π, π], v ∈ [-π/2, π/2].
    """
    ID1 = (int(plane_id) - 1 + 0) % 3
    ID2 = (int(plane_id) - 1 + 1) % 3
    ID3 = (int(plane_id) - 1 + 2) % 3
    normXY = np.sqrt(xyz[:, [ID1]] ** 2 + xyz[:, [ID2]] ** 2)
    normXY[normXY < 1e-6] = 1e-6
    normXYZ = np.sqrt(xyz[:, [ID1]] ** 2 + xyz[:, [ID2]] ** 2 + xyz[:, [ID3]] ** 2)
    v = np.arcsin(xyz[:, [ID3]] / normXYZ)
    u = np.arcsin(xyz[:, [ID1]] / normXY)
    valid = (xyz[:, [ID2]] < 0) & (u >= 0)
    u[valid] = np.pi - u[valid]
    valid = (xyz[:, [ID2]] < 0) & (u <= 0)
    u[valid] = -np.pi - u[valid]
    uv = np.hstack([u, v])
    uv[np.isnan(uv[:, 0]), 0] = 0
    return uv


def _uv_to_xyz(uv, plane_id=1):
    """Planar UV → 3D sphere point (inverse of _xyz_to_uv)."""
    ID1 = (int(plane_id) - 1 + 0) % 3
    ID2 = (int(plane_id) - 1 + 1) % 3
    ID3 = (int(plane_id) - 1 + 2) % 3
    xyz = np.zeros((uv.shape[0], 3))
    xyz[:, ID1] = np.cos(uv[:, 1]) * np.sin(uv[:, 0])
    xyz[:, ID2] = np.cos(uv[:, 1]) * np.cos(uv[:, 0])
    xyz[:, ID3] = np.sin(uv[:, 1])
    return xyz


def _compute_uv(n, u_vals, plane_id):
    """Given great-circle normal n and u-values, compute v on the great circle."""
    if plane_id == 2:
        n = np.array([n[1], n[2], n[0]])
    elif plane_id == 3:
        n = np.array([n[2], n[0], n[1]])
    bc = n[0] * np.sin(u_vals) + n[1] * np.cos(u_vals)
    bs = n[2]
    return np.arctan(-bc / (bs + 1e-9))


# ---------------------------------------------------------------------------
# Range overlap helpers for segment merging
# ---------------------------------------------------------------------------

def _ranges_overlap(r1, r2):
    """Test if two angular ranges (in [0,1]) overlap, handling wraparound."""
    if r1[1] < r1[0]:
        r1a, r1b = [r1[0], 1], [0, r1[1]]
    else:
        r1a, r1b = list(r1), [0, 0]
    if r2[1] < r2[0]:
        r2a, r2b = [r2[0], 1], [0, r2[1]]
    else:
        r2a, r2b = list(r2), [0, 0]
    if max(r1a[0], r2a[0]) < min(r1a[1], r2a[1]):
        return True
    if max(r1b[0], r2b[0]) < min(r1b[1], r2b[1]):
        return True
    return False


def _in_range(pt, r):
    """Test if a point falls within an angular range (in [0,1]), handling wraparound."""
    if r[1] > r[0]:
        return pt >= r[0] and pt <= r[1]
    else:
        return (pt >= r[0] and pt <= 1) or (pt >= 0 and pt <= r[1])


# ---------------------------------------------------------------------------
# Image warping and perspective projection
# ---------------------------------------------------------------------------

def _bilinear_warp(img, Px, Py):
    """Bilinear interpolation warp via scipy map_coordinates."""
    minX = max(1.0, np.floor(Px.min()) - 1)
    minY = max(1.0, np.floor(Py.min()) - 1)
    maxX = min(img.shape[1], np.ceil(Px.max()) + 1)
    maxY = min(img.shape[0], np.ceil(Py.max()) + 1)

    img = img[int(round(minY - 1)):int(round(maxY)),
              int(round(minX - 1)):int(round(maxX))]

    out_shape = Px.shape
    coordinates = [
        (Py - minY).reshape(-1),
        (Px - minX).reshape(-1),
    ]
    warped = np.stack([
        map_coordinates(img[..., c], coordinates, order=1).reshape(out_shape)
        for c in range(img.shape[-1])
    ], axis=-1)
    return warped


def _perspective_crop(pano, cx, cy, size, fov):
    """Tangent-plane perspective projection for one view.

    Args:
        pano: (H, W, 3) equirectangular panorama
        cx, cy: view center direction in radians
        size: output square image size
        fov: field of view in radians

    Returns:
        (size, size, 3) perspective crop
    """
    sphereH, sphereW = pano.shape[:2]
    TX, TY = np.meshgrid(range(1, size + 1), range(1, size + 1))
    TX = TX.reshape(-1, 1, order='F')
    TY = TY.reshape(-1, 1, order='F')
    TX = TX - 0.5 - size / 2
    TY = TY - 0.5 - size / 2
    r = size / 2 / np.tan(fov / 2)

    R = np.sqrt(TY ** 2 + r ** 2)
    ANGy = np.arctan(-TY / r)
    ANGy = ANGy + cy

    X = np.sin(ANGy) * R
    Y = -np.cos(ANGy) * R
    Z = TX

    INDn = np.nonzero(np.abs(ANGy) > np.pi / 2)

    ANGx = np.arctan(Z / -Y)
    RZY = np.sqrt(Z ** 2 + Y ** 2)
    ANGy = np.arctan(X / RZY)

    ANGx[INDn] = ANGx[INDn] + np.pi
    ANGx = ANGx + cx

    INDy = np.nonzero(ANGy < -np.pi / 2)
    ANGy[INDy] = -np.pi - ANGy[INDy]
    ANGx[INDy] = ANGx[INDy] + np.pi

    for _ in range(3):
        ANGx[np.nonzero(ANGx <= -np.pi)] += 2 * np.pi
        ANGx[np.nonzero(ANGx > np.pi)] -= 2 * np.pi

    Px = (ANGx + np.pi) / (2 * np.pi) * sphereW + 0.5
    Py = (-ANGy + np.pi / 2) / np.pi * sphereH + 0.5

    INDxx = np.nonzero(Px < 1)
    Px[INDxx] = Px[INDxx] + sphereW
    pano_ext = np.concatenate([pano, pano[:, :2]], axis=1)

    Px = Px.reshape(size, size, order='F')
    Py = Py.reshape(size, size, order='F')

    return _bilinear_warp(pano_ext, Px, Py)


def _decompose_pano(img, fov, x, y, view_size):
    """Cut equirectangular panorama into perspective views."""
    if not isinstance(fov, np.ndarray):
        fov = fov * np.ones_like(x)
    return [
        {
            'img': _perspective_crop(img.copy(), xi, yi, view_size, fi),
            'vx': xi,
            'vy': yi,
            'fov': fi,
            'sz': view_size,
        }
        for xi, yi, fi in zip(x, y, fov)
    ]


# ---------------------------------------------------------------------------
# LSD line detection
# ---------------------------------------------------------------------------

def _detect_lsd(img):
    """LSD line detection via pylsd. Returns (edge_map, edge_list)."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = img.astype(np.uint8)

    lines = lsd(gray.astype(np.float64), quant=0.7)
    if lines is None:
        return np.zeros_like(gray), np.array([])

    edge_map = np.zeros_like(gray)
    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        width = lines[i, 4]
        cv2.line(edge_map, pt1, pt2, 255, int(np.ceil(width / 2)))

    edge_list = np.concatenate([lines, np.ones_like(lines[:, :2])], axis=1)
    return edge_map, edge_list


# ---------------------------------------------------------------------------
# Back-projection: perspective crop lines → sphere arcs
# ---------------------------------------------------------------------------

def _back_project_to_sphere(edge_data):
    """Back-project perspective crop line endpoints to unit sphere.

    Args:
        edge_data: dict with 'edgeLst' (N,7), 'vx', 'vy', 'fov', 'img' (H,W) edge map

    Returns:
        (N, 10) array: [normal(3), coord1(3), coord2(3), score(1)]
    """
    edge_list = edge_data['edgeLst']
    if len(edge_list) == 0:
        return np.array([])

    vx = edge_data['vx']
    vy = edge_data['vy']
    fov = edge_data['fov']
    imH, imW = edge_data['img'].shape[:2]

    R = (imW / 2) / np.tan(fov / 2)

    x0 = R * np.cos(vy) * np.sin(vx)
    y0 = R * np.cos(vy) * np.cos(vx)
    z0 = R * np.sin(vy)

    vecposX = np.array([np.cos(vx), -np.sin(vx), 0])
    vecposY = np.cross(np.array([x0, y0, z0]), vecposX)
    vecposY = vecposY / np.sqrt(vecposY @ vecposY.T)
    vecposX = vecposX.reshape(1, -1)
    vecposY = vecposY.reshape(1, -1)

    Xc = (imW - 1) / 2
    Yc = (imH - 1) / 2

    vecx1 = edge_list[:, [0]] - Xc
    vecy1 = edge_list[:, [1]] - Yc
    vecx2 = edge_list[:, [2]] - Xc
    vecy2 = edge_list[:, [3]] - Yc

    vec1 = np.tile(vecx1, [1, 3]) * vecposX + np.tile(vecy1, [1, 3]) * vecposY
    vec2 = np.tile(vecx2, [1, 3]) * vecposX + np.tile(vecy2, [1, 3]) * vecposY

    coord1 = np.array([[x0, y0, z0]]) + vec1
    coord2 = np.array([[x0, y0, z0]]) + vec2

    normal = np.cross(coord1, coord2, axis=1)
    normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)

    return np.hstack([normal, coord1, coord2, edge_list[:, [-1]]])


# ---------------------------------------------------------------------------
# Segment merging (conservative 3-pass)
# ---------------------------------------------------------------------------

def _merge_segments(all_edges):
    """Merge colinear line segments from multiple views.

    Args:
        all_edges: list of dicts, each with 'panoLst' (N, 10) from _back_project_to_sphere

    Returns:
        lines: (N, 8) [nx, ny, nz, planeID, umin, umax, arc_length, score]
        coordN_lines: (N, 9) [nx, ny, nz, start(3), end(3)]
    """
    arc_list = []
    for edge in all_edges:
        pano_lst = edge['panoLst']
        if len(pano_lst) == 0:
            continue
        arc_list.append(pano_lst)
    arc_list = np.vstack(arc_list)

    num_line = len(arc_list)
    ori_lines = np.zeros((num_line, 8))
    ori_coordN = np.zeros((num_line, 9))

    areaXY = np.abs(arc_list[:, 2])
    areaYZ = np.abs(arc_list[:, 0])
    areaZX = np.abs(arc_list[:, 1])
    plane_ids = np.argmax(np.stack([areaXY, areaYZ, areaZX], -1), 1) + 1

    for i in range(num_line):
        ori_lines[i, :3] = arc_list[i, :3]
        ori_lines[i, 3] = plane_ids[i]
        c1 = arc_list[i, 3:6]
        c2 = arc_list[i, 6:9]
        uv = _xyz_to_uv(np.stack([c1, c2]), plane_ids[i])
        umax = uv[:, 0].max() + np.pi
        umin = uv[:, 0].min() + np.pi
        if umax - umin > np.pi:
            ori_lines[i, 4:6] = np.array([umax, umin]) / 2 / np.pi
        else:
            ori_lines[i, 4:6] = np.array([umin, umax]) / 2 / np.pi
        ori_lines[i, 6] = np.arccos(np.clip(
            np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2)), -1, 1))
        ori_lines[i, 7] = arc_list[i, 9]

        ori_coordN[i, :3] = arc_list[i, :3]
        ori_coordN[i, 3:6] = c1 / np.linalg.norm(c1)
        ori_coordN[i, 6:9] = c2 / np.linalg.norm(c2)

    lines = ori_lines.copy()
    coordN = ori_coordN.copy()

    for _ in range(3):
        num_line = len(lines)
        valid = np.ones(num_line, bool)
        for i in range(num_line):
            if not valid[i]:
                continue
            dot_prod = (lines[:, :3] * lines[[i], :3]).sum(1)
            cand = np.logical_and(np.abs(dot_prod) > np.cos(np.pi / 180), valid)
            cand[i] = False
            for j in np.nonzero(cand)[0]:
                r1 = lines[i, 4:6]
                r2 = lines[j, 4:6]
                if not _ranges_overlap(r1, r2):
                    continue

                I = np.argmax(np.abs(lines[i, :3]))
                if lines[i, I] * lines[j, I] > 0:
                    nc = lines[i, :3] * lines[i, 6] + lines[j, :3] * lines[j, 6]
                else:
                    nc = lines[i, :3] * lines[i, 6] - lines[j, :3] * lines[j, 6]
                nc = nc / np.linalg.norm(nc)

                nrmin = r2[0] if _in_range(r1[0], r2) else r1[0]
                nrmax = r2[1] if _in_range(r1[1], r2) else r1[1]

                u = np.array([[nrmin], [nrmax]]) * 2 * np.pi - np.pi
                v = _compute_uv(nc, u, lines[i, 3])
                xyz = _uv_to_xyz(np.hstack([u, v]), lines[i, 3])
                l = np.arccos(np.clip(np.dot(xyz[0], xyz[1]), -1, 1))
                scr = (lines[i, 6] * lines[i, 7] + lines[j, 6] * lines[j, 7]) / (lines[i, 6] + lines[j, 6])

                lines[i] = [*nc, lines[i, 3], nrmin, nrmax, l, scr]
                coordN[i] = [*nc, *xyz[0], *xyz[1]]
                valid[j] = False

        lines = lines[valid]
        coordN = coordN[valid]

    return lines, coordN


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _filter_by_length(coordN_lines, threshold=0.3):
    """Keep lines with arc length > threshold."""
    length = np.arccos(np.clip((coordN_lines[:, 3:6] * coordN_lines[:, 6:]).sum(-1), -1, 1))
    mask = length > threshold
    return coordN_lines[mask], mask


def _filter_top_k(coordN_lines, k):
    """Keep top-K lines by arc length."""
    length = np.arccos(np.clip((coordN_lines[:, 3:6] * coordN_lines[:, 6:]).sum(-1), -1, 1))
    idx = np.argsort(length)[::-1][:k]
    return coordN_lines[idx], idx


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def detect_pano_lines(img, view_size=320, length_thres=None, length_topk=None):
    """Full pipeline: decompose panorama → LSD → back-project → merge → filter.

    Args:
        img: (H, W, 3) uint8 equirectangular panorama
        view_size: perspective crop size in pixels
        length_thres: minimum arc length threshold (radians), or None
        length_topk: keep only top-K longest lines, or None

    Returns:
        (N, 9) numpy array: [normal(3), start(3), end(3)] — great-circle arcs on unit sphere
    """
    fov = np.pi / 3
    xh = np.arange(-np.pi, np.pi * 5 / 6, np.pi / 6)
    yh = np.zeros(xh.shape[0])
    xp = np.array([-3, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2]) / 3 * np.pi
    yp = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]) / 4 * np.pi
    x = np.concatenate([xh, xp, [0, 0]])
    y = np.concatenate([yh, yp, [np.pi / 2, -np.pi / 2]])

    scenes = _decompose_pano(img, fov, x, y, view_size)

    edges = []
    for scene in scenes:
        edge_map, edge_list = _detect_lsd(scene['img'])
        edge_data = {
            'img': edge_map,
            'edgeLst': edge_list,
            'vx': scene['vx'],
            'vy': scene['vy'],
            'fov': scene['fov'],
        }
        edge_data['panoLst'] = _back_project_to_sphere(edge_data)
        edges.append(edge_data)

    _, coordN_lines = _merge_segments(edges)

    if length_thres is not None:
        coordN_lines, _ = _filter_by_length(coordN_lines, length_thres)

    if length_topk is not None:
        coordN_lines, _ = _filter_top_k(coordN_lines, length_topk)

    # Coordinate frame alignment: HorizonNet back-projection uses a convention
    # where vx=0 points along +Y, but sphere_to_equirect (cloud2idx) expects
    # vx=0 along +X.  A 90° Z-rotation maps (x,y,z) → (y,-x,z) bridging the
    # two frames.  (This is the same rotation FGPL applies; it is NOT specific
    # to the PICCOLO coordinate system.)
    rot = np.array([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]], dtype=np.float64)
    coordN_lines[:, :3] = coordN_lines[:, :3] @ rot
    coordN_lines[:, 3:6] = coordN_lines[:, 3:6] @ rot
    coordN_lines[:, 6:9] = coordN_lines[:, 6:9] @ rot

    return coordN_lines
