"""Sphere distance functions for XDF pose estimation.

Provides Line Distance Function (LDF) and Point Distance Function (PDF)
computations on the unit sphere, used by the XDF coarse search to score
rotation + translation candidates.

Extracted from feature_matchingV2.py — no imports from panoramic-localization/.
"""

import sys
from pathlib import Path

import numpy as np
import torch

_SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_ROOT / "features_2d"))
from line_analysis import classify_lines


# ============================================================
# 3D line classification
# ============================================================

def classify_3d(dirs, principal_3d, inlier_thres=0.05):
    """Classify 3D line directions by principal direction.

    Args:
        dirs: (N, 3) unit direction vectors
        principal_3d: (3, 3) principal direction matrix

    Returns:
        (N, 3) bool mask — True where |dir · principal| > 1 - inlier_thres
    """
    inner = torch.abs(dirs @ principal_3d.T)
    return inner > 1 - inlier_thres


# ============================================================
# Sphere distance: query points → 2D line arcs
# ============================================================

def distance_to_2d_arcs(query_pts, edge_2d, mask=None, rot_mtx=None):
    """Spherical arc distance from query points to 2D line segments.

    Handles endpoint fallback (FGPL-faithful): if the closest point on the
    great circle is outside the arc, falls back to the nearer endpoint.

    Args:
        query_pts: (N_q, 3) unit sphere points
        edge_2d:   (N_2D, 9) [normal(3), start(3), end(3)]
        mask:      (N_2D, K) bool — per-group membership, or None
        rot_mtx:   (3, 3) rotation to apply to lines, or None

    Returns:
        (N_q,) if mask is None, else (N_q, K) — min distance per group
    """
    normals = edge_2d[:, :3]
    starts = edge_2d[:, 3:6]
    ends = edge_2d[:, 6:]

    if rot_mtx is not None:
        normals = normals @ rot_mtx.T
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        starts = starts @ rot_mtx.T
        starts = starts / starts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ends = ends @ rot_mtx.T
        ends = ends / ends.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    cos_se = (starts * ends).sum(-1).unsqueeze(0)       # (1, N_2D)
    cos_qs = query_pts @ starts.T                        # (N_q, N_2D)
    cos_qe = query_pts @ ends.T

    normal_acute = np.pi / 2 - torch.arccos(
        torch.abs(query_pts @ normals.T).clamp(-1, 1))
    theta_s = torch.arccos(cos_qs.clamp(-1, 1))
    theta_e = torch.arccos(cos_qe.clamp(-1, 1))

    sign_s = cos_qs - cos_se * cos_qe > 0
    sign_e = cos_qe - cos_se * cos_qs > 0
    on_arc = sign_s & sign_e

    sphere_dist = on_arc * normal_acute + ~on_arc * torch.minimum(theta_s, theta_e)

    if mask is None:
        return sphere_dist.min(-1).values
    else:
        MAX = np.pi
        dist = sphere_dist.unsqueeze(-1) * mask.unsqueeze(0)
        dist += ~mask.unsqueeze(0) * MAX
        return dist.min(1).values  # (N_q, K)


# ============================================================
# Sphere distance: query points → projected 3D lines
# ============================================================

def distance_to_3d_lines(query_pts, starts, ends, translation, rotation, mask=None):
    """Spherical arc distance from query points to 3D line segments projected to sphere.

    Args:
        query_pts:   (N_q, 3) unit sphere points
        starts:      (N_3D, 3) line start points (in the coordinate frame of rotation)
        ends:        (N_3D, 3) line end points
        translation: (3,) camera position
        rotation:    (3, 3) world-to-camera rotation
        mask:        (N_3D, K) bool — per-group membership, or None

    Returns:
        (N_q,) if mask is None, else (N_q, K)
    """
    s_cam = (starts - translation.unsqueeze(0)) @ rotation.T
    e_cam = (ends - translation.unsqueeze(0)) @ rotation.T
    s_sph = s_cam / s_cam.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    e_sph = e_cam / e_cam.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    normals = torch.cross(s_sph, e_sph, dim=-1)
    n_norm = normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normals = normals / n_norm

    cos_se = (s_sph * e_sph).sum(-1).unsqueeze(0)
    cos_qs = query_pts @ s_sph.T
    cos_qe = query_pts @ e_sph.T

    normal_acute = np.pi / 2 - torch.arccos(
        torch.abs(query_pts @ normals.T).clamp(-1, 1))
    theta_s = torch.arccos(cos_qs.clamp(-1, 1))
    theta_e = torch.arccos(cos_qe.clamp(-1, 1))

    sign_s = cos_qs - cos_se * cos_qe > 0
    sign_e = cos_qe - cos_se * cos_qs > 0
    on_arc = sign_s & sign_e

    sphere_dist = on_arc * normal_acute + ~on_arc * torch.minimum(theta_s, theta_e)

    if mask is None:
        return sphere_dist.min(-1).values
    else:
        MAX = np.pi
        dist = sphere_dist.unsqueeze(-1) * mask.unsqueeze(0)
        dist += ~mask.unsqueeze(0) * MAX
        return dist.min(1).values


# ============================================================
# Sphere distance: query points → keypoints (intersections)
# ============================================================

def distance_to_sphere_points(query_pts, keypoints, mask=None):
    """Great-circle distance from query points to intersection keypoints.

    Args:
        query_pts: (N_q, 3) unit sphere points
        keypoints: (N_kpt, 3) intersection points on unit sphere
        mask:      (N_kpt, K) bool — per-group membership, or None

    Returns:
        (N_q,) if mask is None, else (N_q, K)
    """
    cos_theta = (query_pts @ keypoints.T).clamp(-1, 1)
    sphere_dist = torch.arccos(cos_theta)  # (N_q, N_kpts)

    if mask is None:
        return sphere_dist.min(-1).values
    else:
        MAX = np.pi
        dist = sphere_dist.unsqueeze(-1) * mask.unsqueeze(0)
        dist += ~mask.unsqueeze(0) * MAX
        return dist.min(1).values


# ============================================================
# Intersection mask builder
# ============================================================

def build_intersection_masks(inter_list, device=None):
    """Build FGPL-style group masks from per-group intersection lists.

    inter_list[k] contains intersections from groups k and (k+1)%3,
    so mask columns k and (k+1)%3 are True for those points.

    Args:
        inter_list: list of 3 tensors, each (M_k, 3)
        device: torch device (inferred from tensors if None)

    Returns:
        (sum(M_k), 3) bool tensor
    """
    if device is None:
        device = inter_list[0].device if inter_list[0].shape[0] > 0 else 'cpu'
    masks = []
    for k in range(3):
        n = inter_list[k].shape[0]
        m = torch.zeros(n, 3, dtype=torch.bool, device=device)
        m[:, k] = True
        m[:, (k + 1) % 3] = True
        masks.append(m)
    return torch.cat(masks, dim=0)


# ============================================================
# 2D intersections with line-pair indices
# ============================================================

def find_intersections_2d_indexed(edge_2d, principal_2d,
                                  inlier_thres=0.3, intersect_thres=0.1):
    """Find 2D line intersections on sphere, tracking which lines produced each.

    Like line_analysis.find_intersections_2d() but also returns (M_k, 2)
    line-pair index tensors per group. Needed for ICP rotation refinement,
    which must look up line normals/directions for matched intersections.

    Args:
        edge_2d: (N, 9) tensor [normal(3), start(3), end(3)]
        principal_2d: (3, 3) vanishing direction matrix
        inlier_thres: threshold for classify_lines
        intersect_thres: arc membership tolerance

    Returns:
        (inter_pts, inter_idx):
            inter_pts: list of 3 (M_k, 3) tensors — intersection sphere points
            inter_idx: list of 3 (M_k, 2) long tensors — global line indices
    """
    device = edge_2d.device
    pi = torch.acos(torch.zeros(1, device=device)).item() * 2

    mask = classify_lines(edge_2d, principal_2d, inlier_thres)
    edge_p = [edge_2d[mask[:, i]] for i in range(3)]
    edge_num = [ep.shape[0] for ep in edge_p]

    full_range = torch.arange(edge_2d.shape[0], device=device)
    ids_p = [full_range[mask[:, i]] for i in range(3)]

    arc_len = [torch.arccos((ep[:, 3:6] * ep[:, 6:]).sum(-1).clamp(-1, 1))
               for ep in edge_p]

    total_inter = []
    total_idx = []

    for i in range(3):
        j = (i + 1) % 3
        ni, nj = edge_num[i], edge_num[j]

        if ni == 0 or nj == 0:
            total_inter.append(torch.zeros(0, 3, device=device))
            total_idx.append(torch.zeros(0, 2, device=device, dtype=torch.long))
            continue

        # Cross product of normals → intersection candidates
        n_rep0 = edge_p[i][:, :3].repeat_interleave(nj, dim=0)
        n_rep1 = edge_p[j][:, :3].repeat(ni, 1)
        cand = torch.cross(n_rep0, n_rep1, dim=-1).reshape(ni, nj, 3)
        cand_norm = cand.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cand = cand / cand_norm

        # Build id grids
        id0 = ids_p[i].unsqueeze(1).expand(ni, nj)
        id1 = ids_p[j].unsqueeze(0).expand(ni, nj)

        # Arc membership test
        s0 = edge_p[i][:, 3:6].unsqueeze(1).expand(ni, nj, 3)
        e0 = edge_p[i][:, 6:].unsqueeze(1).expand(ni, nj, 3)
        s1 = edge_p[j][:, 3:6].unsqueeze(0).expand(ni, nj, 3)
        e1 = edge_p[j][:, 6:].unsqueeze(0).expand(ni, nj, 3)
        al0 = arc_len[i].unsqueeze(1).expand(ni, nj)
        al1 = arc_len[j].unsqueeze(0).expand(ni, nj)

        # UP candidate
        ds0 = torch.arccos((cand * s0).sum(-1).clamp(-1, 1))
        de0 = torch.arccos((cand * e0).sum(-1).clamp(-1, 1))
        ds1 = torch.arccos((cand * s1).sum(-1).clamp(-1, 1))
        de1 = torch.arccos((cand * e1).sum(-1).clamp(-1, 1))
        up_on_0 = (ds0 + de0 - al0) < intersect_thres
        up_on_1 = (ds1 + de1 - al1) < intersect_thres
        up_valid = up_on_0 & up_on_1

        # DOWN candidate (antipodal)
        dn_ds0 = pi - ds0
        dn_de0 = pi - de0
        dn_ds1 = pi - ds1
        dn_de1 = pi - de1
        dn_on_0 = (dn_ds0 + dn_de0 - al0) < intersect_thres
        dn_on_1 = (dn_ds1 + dn_de1 - al1) < intersect_thres
        dn_valid = dn_on_0 & dn_on_1

        up_pts = cand[up_valid]
        dn_pts = -cand[dn_valid]
        up_ids = torch.stack([id0[up_valid], id1[up_valid]], dim=-1)
        dn_ids = torch.stack([id0[dn_valid], id1[dn_valid]], dim=-1)

        pts = torch.cat([up_pts, dn_pts], dim=0) if (up_pts.shape[0] + dn_pts.shape[0]) > 0 \
            else torch.zeros(0, 3, device=device)
        ids = torch.cat([up_ids, dn_ids], dim=0) if (up_ids.shape[0] + dn_ids.shape[0]) > 0 \
            else torch.zeros(0, 2, device=device, dtype=torch.long)

        total_inter.append(pts)
        total_idx.append(ids)

    return total_inter, total_idx
