"""Sphere point generation, coordinate conversions, panoramic rendering."""

import numpy as np
import torch


def icosahedron_to_sphere(level):
    """Icosahedron subdivision → (N, 3) unit sphere points + triangle indices."""
    a = 2 / (1 + np.sqrt(5))
    M = np.array([
        0, a, -1, a, 1, 0, -a, 1, 0,
        0, a, 1, -a, 1, 0, a, 1, 0,
        0, a, 1, 0, -a, 1, -1, 0, a,
        0, a, 1, 1, 0, a, 0, -a, 1,
        0, a, -1, 0, -a, -1, 1, 0, -a,
        0, a, -1, -1, 0, -a, 0, -a, -1,
        0, -a, 1, a, -1, 0, -a, -1, 0,
        0, -a, -1, -a, -1, 0, a, -1, 0,
        -a, 1, 0, -1, 0, a, -1, 0, -a,
        -a, -1, 0, -1, 0, -a, -1, 0, a,
        a, 1, 0, 1, 0, -a, 1, 0, a,
        a, -1, 0, 1, 0, a, 1, 0, -a,
        0, a, 1, -1, 0, a, -a, 1, 0,
        0, a, 1, a, 1, 0, 1, 0, a,
        0, a, -1, -a, 1, 0, -1, 0, -a,
        0, a, -1, 1, 0, -a, a, 1, 0,
        0, -a, -1, -1, 0, -a, -a, -1, 0,
        0, -a, -1, a, -1, 0, 1, 0, -a,
        0, -a, 1, -a, -1, 0, -1, 0, a,
        0, -a, 1, 1, 0, a, a, -1, 0])

    coor = M.T.reshape(3, 60, order='F').T
    coor, idx = np.unique(coor, return_inverse=True, axis=0)
    tri = idx.reshape(3, 20, order='F').T

    coor = list(coor / np.linalg.norm(coor, axis=1, keepdims=True))

    for _ in range(level):
        triN = []
        for t in range(len(tri)):
            n = len(coor)
            coor.append((coor[tri[t, 0]] + coor[tri[t, 1]]) / 2)
            coor.append((coor[tri[t, 1]] + coor[tri[t, 2]]) / 2)
            coor.append((coor[tri[t, 2]] + coor[tri[t, 0]]) / 2)
            triN.append([n, tri[t, 0], n + 2])
            triN.append([n, tri[t, 1], n + 1])
            triN.append([n + 1, tri[t, 2], n + 2])
            triN.append([n, n + 1, n + 2])
        tri = np.array(triN)
        coor, idx = np.unique(coor, return_inverse=True, axis=0)
        tri = idx[tri]
        coor = list(coor / np.sqrt(np.sum(coor * coor, 1, keepdims=True)))

    return np.array(coor), np.array(tri)


def generate_sphere_points(level, device='cpu'):
    """Generate unit sphere points via icosahedron subdivision → torch tensor."""
    sphere_pts, _ = icosahedron_to_sphere(level)
    return torch.from_numpy(sphere_pts).float().to(device)


def sphere_to_equirect(xyz):
    """(N, 3) 3D points → (N, 2) normalized UV in [-1, 1] for equirectangular projection."""
    theta = torch.atan2(torch.norm(xyz[:, :2], dim=-1), xyz[:, 2] + 1e-6).unsqueeze(1)
    phi = torch.atan2(xyz[:, 1:2], xyz[:, 0:1] + 1e-6) + np.pi
    coord = torch.stack([
        1.0 - phi[:, 0] / (2 * np.pi),
        theta[:, 0] / np.pi,
    ], dim=-1)
    return 2 * coord - 1


def equirect_to_sphere(uv):
    """(N, 2) UV in [-1, 1] → (N, 3) unit sphere points."""
    s = (uv + 1.0) / 2.0
    phi = (1.0 - s[:, 0]) * (2 * np.pi) - np.pi
    theta = np.pi * s[:, 1]
    xyz = torch.zeros(s.shape[0], 3, device=uv.device)
    xyz[:, 0] = torch.sin(theta) * torch.cos(phi)
    xyz[:, 1] = torch.sin(theta) * torch.sin(phi)
    xyz[:, 2] = torch.cos(theta)
    return xyz


def render_points_to_pano(xyz, rgb, resolution=(512, 1024), bg_white=False, pad=0):
    """Rasterize 3D points to equirectangular image.

    Args:
        xyz: (N, 3) torch tensor, points on unit sphere or in 3D
        rgb: (N, 3) torch tensor in [0, 1]
        resolution: (H, W) output image size
        bg_white: white background if True, black if False
        pad: pixel padding radius (0 = single pixel, 1 = 3x3 block)

    Returns:
        (H, W, 3) uint8 numpy array
    """
    H, W = resolution
    with torch.no_grad():
        dist = torch.norm(xyz, dim=-1)
        order = torch.argsort(dist, descending=True)
        xyz_s = xyz[order]
        rgb_s = rgb[order]

        coord = sphere_to_equirect(xyz_s)
        coord = (coord + 1.0) / 2.0
        coord[:, 0] *= (W - 1)
        coord[:, 1] *= (H - 1)
        coord = torch.flip(coord, [-1]).long()
        rc = tuple(coord.t())

        bg_val = 1.0 if bg_white else 0.0
        image = torch.full([H, W, 3], bg_val, dtype=torch.float, device=xyz.device)

        if pad >= 1:
            temp = torch.ones_like(rc[0])
            neighbors = [
                (rc[0], torch.clamp(rc[1] - temp, min=0)),
                (rc[0], torch.clamp(rc[1] + temp, max=W - 1)),
                (torch.clamp(rc[0] - temp, min=0), torch.clamp(rc[1] - temp, min=0)),
                (torch.clamp(rc[0] - temp, min=0), rc[1]),
                (torch.clamp(rc[0] - temp, min=0), torch.clamp(rc[1] + temp, max=W - 1)),
                (torch.clamp(rc[0] + temp, max=H - 1), torch.clamp(rc[1] - temp, min=0)),
                (torch.clamp(rc[0] + temp, max=H - 1), rc[1]),
                (torch.clamp(rc[0] + temp, max=H - 1), torch.clamp(rc[1] + temp, max=W - 1)),
            ]
            for r, c in neighbors:
                image.index_put_((r, c), rgb_s, accumulate=False)
        image.index_put_(rc, rgb_s, accumulate=False)

        return (image * 255).cpu().numpy().astype(np.uint8)


def render_sphere_lines(edge_2d, resolution=(512, 1024), rgb=None):
    """Render great-circle arcs to equirectangular image (1-pixel thin lines).

    Args:
        edge_2d: (N, 9) tensor [normal(3), start(3), end(3)]
        resolution: (H, W) output image size
        rgb: (N, 3) tensor of line colors in [0, 1], or None for white

    Returns:
        (H, W, 3) uint8 numpy array
    """
    starts = edge_2d[:, 3:6]
    ends = edge_2d[:, 6:]
    dirs = ends - starts

    line_steps = max(resolution[0], resolution[1])
    t = torch.linspace(0, 1, line_steps, device=edge_2d.device).reshape(1, -1, 1)
    pts = (dirs.unsqueeze(1) * t + starts.unsqueeze(1)).reshape(-1, 3)

    if rgb is not None:
        colors = rgb.repeat_interleave(line_steps, dim=0)
    else:
        colors = torch.ones_like(pts)

    return render_points_to_pano(pts, colors, resolution, pad=0)
