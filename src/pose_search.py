"""Rotation candidate generation and XDF coarse pose search.

Builds 24 rotation candidates from principal directions, generates a
chamfer-filtered translation grid, and performs canonical-frame XDF
inlier-counting search over (rotation, translation) pairs.

Extracted from feature_matchingV2.py — no imports from panoramic-localization/.
"""

import sys
from itertools import permutations as iter_perms
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sphere_geometry import generate_sphere_points
from xdf_distance import (
    classify_3d,
    distance_to_2d_arcs,
    distance_to_3d_lines,
    distance_to_sphere_points,
    build_intersection_masks,
)
from line_analysis import classify_lines


# ============================================================
# YPR ↔ rotation matrix (differentiable)
# ============================================================

def rot_from_ypr(ypr):
    """(3,) tensor → (3,3) rotation matrix. RZ(yaw) @ RY(pitch) @ RX(roll). Differentiable."""
    yaw, pitch, roll = ypr
    y = yaw.unsqueeze(0)
    p = pitch.unsqueeze(0)
    r = roll.unsqueeze(0)
    z0 = torch.zeros(1, device=ypr.device)
    o1 = torch.ones(1, device=ypr.device)

    RX = torch.stack([
        torch.stack([o1, z0, z0]),
        torch.stack([z0, torch.cos(r), -torch.sin(r)]),
        torch.stack([z0, torch.sin(r),  torch.cos(r)])]).reshape(3, 3)
    RY = torch.stack([
        torch.stack([torch.cos(p), z0, torch.sin(p)]),
        torch.stack([z0, o1, z0]),
        torch.stack([-torch.sin(p), z0, torch.cos(p)])]).reshape(3, 3)
    RZ = torch.stack([
        torch.stack([torch.cos(y), -torch.sin(y), z0]),
        torch.stack([torch.sin(y),  torch.cos(y), z0]),
        torch.stack([z0, z0, o1])]).reshape(3, 3)
    return RZ @ RY @ RX


def ypr_from_rot(R):
    """(3,3) tensor → (3,) yaw-pitch-roll tensor."""
    yaw   = torch.atan2(R[1, 0], R[0, 0] + 1e-6)
    pitch = torch.arcsin(-R[2, 0].clamp(-1, 1))
    roll  = torch.atan2(R[2, 1], R[2, 2])
    return torch.tensor([yaw.item(), pitch.item(), roll.item()],
                        device=R.device, dtype=R.dtype)


# ============================================================
# 24 rotation candidates
# ============================================================

def build_rotation_candidates(principal_2d, principal_3d):
    """Build 24 rotation candidates: 6 permutations × 4 det-preserving sign flips.

    Args:
        principal_2d: (3, 3) 2D vanishing directions
        principal_3d: (3, 3) 3D principal directions

    Returns:
        rotations: (24, 3, 3) tensor
        perms_expanded: (24, 3) long tensor — permutation indices
    """
    device = principal_2d.device
    perm_indices = list(iter_perms(range(3)))  # 6 permutations
    perms = torch.tensor(perm_indices, device=device, dtype=torch.long)

    bin_mask = torch.ones(len(perm_indices) * 4, 3, 1, device=device)
    for perm_idx in range(len(perm_indices)):
        for idx in range(4):
            bin_mask[perm_idx * 4 + idx, 0, 0] = (-1) ** (idx // 2)
            bin_mask[perm_idx * 4 + idx, 1, 0] = (-1) ** (idx % 2)
            bin_mask[perm_idx * 4 + idx, 2, 0] = (-1) ** (idx // 2 + idx % 2)
            if perm_idx in [1, 2, 5]:
                bin_mask[perm_idx * 4 + idx, 2, 0] *= -1

    perms_expanded = perms.repeat_interleave(4, dim=0)  # (24, 3)

    # Target = eye(3): in canonical frame, 3D principal directions ARE identity
    # Native: canonical_principal_3d = torch.eye(3), H = I^T @ (bin_mask * P_2d[perm])
    N = perms_expanded.shape[0]
    pts_2d = principal_2d[perms_expanded]  # (24, 3, 3)
    H = bin_mask * pts_2d  # equivalent to eye(3)^T @ (bin_mask * P_2d[perm])
    U, S, V = torch.svd(H)
    U_t = U.transpose(1, 2)
    d = torch.sign(torch.linalg.det(V @ U_t))
    diag = torch.eye(3, device=device).unsqueeze(0).repeat(N, 1, 1)
    diag[:, 2, 2] = d
    rotations = V @ diag @ U_t  # (24, 3, 3)

    return rotations, perms_expanded


# ============================================================
# Translation grid
# ============================================================

def generate_translation_grid(starts, ends, num_trans=1700, chamfer_min_dist=0.3,
                               spacing=None):
    """Quantile-based 3D grid matching FGPL's generate_trans_points().

    Uses line midpoints to build an adaptive grid where denser regions of the
    point cloud get more samples. Grid dimensions scale with each axis's
    extent squared, and positions are placed at quantile values.

    Args:
        starts: (N, 3) line start points
        ends:   (N, 3) line end points
        num_trans: target number of translation candidates (~1700)
        chamfer_min_dist: min distance from midpoints to keep
        spacing: ignored (kept for API compatibility)

    Returns:
        (N_t, 3) translation candidates
    """
    device = starts.device
    midpoints = (starts + ends) / 2

    # Adaptive grid dimensions (matches FGPL adaptive_trans_num)
    q90 = torch.quantile(midpoints, 0.90, dim=0)
    q10 = torch.quantile(midpoints, 0.10, dim=0)
    lengths = (q90 - q10).clamp(min=1e-3)

    # Grid dims proportional to length^2, product ≈ num_trans
    # n_x = ceil((Lx^2 * N / (Ly * Lz))^(1/3)), etc.
    import math
    nx = math.ceil((lengths[0]**2 * num_trans / (lengths[1] * lengths[2])) ** (1/3))
    ny = math.ceil((lengths[1]**2 * num_trans / (lengths[0] * lengths[2])) ** (1/3))
    nz = math.ceil((lengths[2]**2 * num_trans / (lengths[0] * lengths[1])) ** (1/3))

    # Force odd (for centering, matching FGPL)
    if nx % 2 == 0: nx -= 1
    if ny % 2 == 0: ny -= 1
    if nz % 2 == 0: nz -= 1
    nx, ny, nz = max(nx, 1), max(ny, 1), max(nz, 1)

    # Quantile sampling along each axis
    def quantile_pts(vals, n):
        split = (torch.arange(n, device=device, dtype=torch.float32) + 1) / (n + 1)
        if split[0] > 0.1:
            return torch.quantile(vals, split)
        else:
            split = torch.linspace(0.1, 0.9, n, device=device)
            return torch.quantile(vals, split)

    x_pts = quantile_pts(midpoints[:, 0], nx)
    y_pts = quantile_pts(midpoints[:, 1], ny)
    z_pts = quantile_pts(midpoints[:, 2], nz)

    gx, gy, gz = torch.meshgrid(x_pts, y_pts, z_pts, indexing='ij')
    trans = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)

    # Chamfer filter: keep points away from wireframe midpoints
    # Match native FGPL: sample n_lines // 100 reference points (very sparse)
    n_sample = max(1, midpoints.shape[0] // 100)
    sample_idx = torch.randperm(midpoints.shape[0], device=device)[:n_sample]
    sample_mid = midpoints[sample_idx]

    chunk = 200
    keep = torch.ones(trans.shape[0], dtype=torch.bool, device=device)
    for i in range(0, trans.shape[0], chunk):
        batch = trans[i:i+chunk]
        chamfer = (batch.unsqueeze(1) - sample_mid.unsqueeze(0)).norm(dim=-1).min(dim=1).values
        keep[i:i+chunk] = chamfer > chamfer_min_dist

    return trans[keep]


# ============================================================
# Intersection group index mapping
# ============================================================

def intersections_idx(idx0, idx1):
    """Map (direction_i, direction_j) → intersection group index 0/1/2."""
    r0, r1 = idx0 % 3, idx1 % 3
    if (r0, r1) in [(0, 1), (1, 0)]:
        return 0
    elif (r0, r1) in [(1, 2), (2, 1)]:
        return 1
    else:
        return 2


# ============================================================
# Rearrange 2D intersections for 24 rotation permutations
# ============================================================

def rearrange_intersections_for_rotations(raw_inter_2d, raw_inter_2d_idx, perms):
    """Rearrange 2D intersections per rotation permutation.

    For each of the 24 rotations, the permutation reorders which principal
    direction maps to which canonical axis. Intersection groups must be
    remapped accordingly.

    Args:
        raw_inter_2d: list of 3 (M_k, 3) tensors — raw 2D intersection points
        raw_inter_2d_idx: list of 3 (M_k, 2) long tensors — line-pair indices
        perms: (24, 3) long tensor — permutation indices

    Returns:
        (full_inter_2d, full_inter_2d_mask, full_inter_2d_idx):
            lists of 24 tensors each
    """
    device = perms.device
    N_r = perms.shape[0]

    full_inter_2d = []
    full_inter_2d_mask = []
    full_inter_2d_idx = []

    for ri in range(N_r):
        perm = perms[ri].tolist()
        inter_perm_order = [intersections_idx(perm[k % 3], perm[(k + 1) % 3])
                            for k in range(3)]

        inter_pts = torch.cat([raw_inter_2d[p] for p in inter_perm_order], dim=0)
        inter_ids = torch.cat([raw_inter_2d_idx[p] for p in inter_perm_order], dim=0)

        masks = []
        for k, p in enumerate(inter_perm_order):
            n = raw_inter_2d[p].shape[0]
            m = torch.zeros(n, 3, dtype=torch.bool, device=device)
            m[:, k] = True
            m[:, (k + 1) % 3] = True
            masks.append(m)
        inter_mask = torch.cat(masks, dim=0)

        full_inter_2d.append(inter_pts)
        full_inter_2d_mask.append(inter_mask)
        full_inter_2d_idx.append(inter_ids)

    return full_inter_2d, full_inter_2d_mask, full_inter_2d_idx


# ============================================================
# XDF 3D precomputation (one-time, shared across panoramas)
# ============================================================

def precompute_xdf_3d(
    principal_3d, starts, ends, dirs,
    inter_3d, inter_3d_mask,
    trans_candidates, query_pts,
    inlier_thres_3d=0.05, chunk_size=200,
):
    """One-time 3D canonical frame precomputation for XDF coarse search.

    Transforms 3D geometry into the canonical frame and precomputes
    LDF-3D and PDF-3D for all translation candidates. This is the
    expensive step (~20s) that can be shared across multiple panoramas.

    Args:
        principal_3d: (3, 3) 3D principal directions
        starts, ends, dirs: (N_3D, 3) 3D line geometry
        inter_3d: (N_3D_inter, 3) concatenated 3D intersection points
        inter_3d_mask: (N_3D_inter, 3) bool group mask
        trans_candidates: (N_t, 3) translation candidates
        query_pts: (N_q, 3) icosphere query points
        inlier_thres_3d: 3D line classification threshold
        chunk_size: translation batch size for memory efficiency

    Returns:
        dict with keys:
            'canonical_rot':  (3, 3) — the canonical rotation (= principal_3d)
            'ldf_3d':         (N_t, N_q, 3) — line distance function
            'pdf_3d':         (N_t, N_q, 3) — point distance function
            'mask_3d':        (N_3D, 3) bool — 3D line classification
    """
    device = starts.device
    N_t = trans_candidates.shape[0]

    # Canonical frame: rotate 3D so principal_3d → identity
    canonical_rot = principal_3d  # (3,3)
    can_starts = starts @ canonical_rot.T
    can_ends = ends @ canonical_rot.T
    can_dirs = dirs @ canonical_rot.T
    can_trans = trans_candidates @ canonical_rot.T

    eye = torch.eye(3, device=device)
    mask_3d = classify_3d(can_dirs, eye, inlier_thres_3d)

    can_inter_3d = inter_3d @ canonical_rot.T

    # LDF-3D: (N_t, N_q, 3)
    print("    Precomputing LDF-3D...")
    ldf_3d_chunks = []
    for i in range(0, N_t, chunk_size):
        batch_trans = can_trans[i:i+chunk_size]
        batch_ldf = []
        for ti in range(batch_trans.shape[0]):
            d = distance_to_3d_lines(query_pts, can_starts, can_ends,
                                     batch_trans[ti], eye, mask_3d)
            batch_ldf.append(d)
        ldf_3d_chunks.append(torch.stack(batch_ldf))
    ldf_3d = torch.cat(ldf_3d_chunks, dim=0)

    # PDF-3D: (N_t, N_q, 3)
    print("    Precomputing PDF-3D...")
    pdf_3d_chunks = []
    for i in range(0, N_t, chunk_size):
        batch_trans = can_trans[i:i+chunk_size]
        batch_pdf = []
        for ti in range(batch_trans.shape[0]):
            i3d_cam = (can_inter_3d - batch_trans[ti].unsqueeze(0)) @ eye.T
            i3d_sph = i3d_cam / i3d_cam.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            d = distance_to_sphere_points(query_pts, i3d_sph, inter_3d_mask)
            batch_pdf.append(d)
        pdf_3d_chunks.append(torch.stack(batch_pdf))
    pdf_3d = torch.cat(pdf_3d_chunks, dim=0)

    return {
        'canonical_rot': canonical_rot,
        'ldf_3d': ldf_3d,
        'pdf_3d': pdf_3d,
        'mask_3d': mask_3d,
    }


# ============================================================
# XDF 2D computation + cost (per-panorama)
# ============================================================

def xdf_coarse_search_from_precomputed(
    precomputed_3d,
    rotations, perms, edge_2d,
    inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot,
    trans_candidates, query_pts,
    top_k=5, xdf_inlier_thres=0.1, point_gamma=0.2,
    inlier_thres_2d=0.05, chunk_size=200,
):
    """Per-panorama 2D computation + cost matrix + top-K selection.

    Takes precomputed 3D distance functions and per-panorama 2D features,
    computes LDF-2D/PDF-2D via single_pose_compute, builds the cost
    matrix, and returns top-K poses with rotation diversity.

    Args:
        precomputed_3d: dict from precompute_xdf_3d()
        rotations: (N_r, 3, 3) rotation candidates
        perms: (N_r, 3) permutation indices
        edge_2d: (N_2D, 9) 2D line segments
        inter_2d_per_rot: list of N_r tensors — 2D intersections per rotation
        inter_2d_mask_per_rot: list of N_r tensors — group masks
        inter_2d_idx_per_rot: list of N_r tensors — line-pair indices
        trans_candidates: (N_t, 3) translation candidates
        query_pts: (N_q, 3) icosphere query points
        top_k: number of top poses to return
        xdf_inlier_thres: inlier counting threshold
        point_gamma: PDF power exponent
        inlier_thres_2d: 2D line classification threshold
        chunk_size: translation batch size for memory efficiency

    Returns:
        (top_poses, cost_matrix)
    """
    device = edge_2d.device
    N_r = rotations.shape[0]
    N_t = trans_candidates.shape[0]
    N_q = query_pts.shape[0]

    canonical_rot = precomputed_3d['canonical_rot']
    ldf_3d = precomputed_3d['ldf_3d']
    pdf_3d = precomputed_3d['pdf_3d']

    # --- Precompute LDF-2D: (N_r, N_q, 3) ---
    # single_pose_compute: compute for rotation 0, interpolate for others
    # (matching native FGPL config: single_pose_compute=True)
    print("    Precomputing LDF-2D (single_pose_compute)...")

    # Step 1: exact LDF-2D for rotation 0 (no rotation applied to lines)
    R0 = rotations[0]
    rot_inner_0 = torch.abs(edge_2d[:, :3] @ R0)
    min_mask_0 = rot_inner_0.argmin(-1, keepdim=True) == torch.arange(3, device=device).unsqueeze(0)
    can_mask_0 = (rot_inner_0 < inlier_thres_2d) & min_mask_0
    ldf_base = distance_to_2d_arcs(query_pts, edge_2d, can_mask_0)  # (N_q, 3) — no rot_mtx

    # Step 2: for each rotation, rotate query points and NN-interpolate
    # rot_query_pts[ri] = query_pts @ R_can[ri].T
    rot_query = query_pts.unsqueeze(0) @ rotations.permute(0, 2, 1)  # (N_r, N_q, 3)
    # Find nearest original query point for each rotated query point
    # cos_sim = rot_query @ query_pts.T → (N_r, N_q, N_q)
    nn_cos = torch.bmm(rot_query, query_pts.T.unsqueeze(0).expand(N_r, -1, -1))
    nn_idx = nn_cos.argmax(-1)  # (N_r, N_q) — nearest original query point

    # Step 3: permute channels and gather
    # ldf_base[:, perms] selects columns by permutation → (N_q, N_r, 3)
    ldf_permuted = ldf_base[:, perms]  # (N_q, N_r, 3)
    ldf_permuted = ldf_permuted.permute(1, 0, 2)  # (N_r, N_q, 3)
    # Gather: for each (ri, qi), look up distance at nn_idx[ri, qi]
    ldf_2d = torch.gather(ldf_permuted, 1,
                           nn_idx.unsqueeze(-1).expand(-1, -1, 3))  # (N_r, N_q, 3)

    # --- Precompute PDF-2D: (N_r, N_q, 3) ---
    # single_pose_compute: compute for rotation 0's intersections, interpolate
    print("    Precomputing PDF-2D (single_pose_compute)...")
    # Step 1: exact PDF-2D for rotation 0's intersections (no rotation)
    pdf_base = distance_to_sphere_points(
        query_pts, inter_2d_per_rot[0], inter_2d_mask_per_rot[0])  # (N_q, 3)

    # Step 2: permute channels and NN-interpolate (same nn_idx as LDF)
    pdf_permuted = pdf_base[:, perms]  # (N_q, N_r, 3)
    pdf_permuted = pdf_permuted.permute(1, 0, 2)  # (N_r, N_q, 3)
    pdf_2d = torch.gather(pdf_permuted, 1,
                           nn_idx.unsqueeze(-1).expand(-1, -1, 3))  # (N_r, N_q, 3)

    # --- Combine and compute cost ---
    print("    Computing XDF cost matrix...")
    dist_3d = torch.cat([ldf_3d, pdf_3d ** point_gamma], dim=-1)  # (N_t, N_q, 6)
    dist_2d = torch.cat([ldf_2d, pdf_2d ** point_gamma], dim=-1)  # (N_r, N_q, 6)

    cost = torch.zeros(N_t, N_r, device=device)
    for i in range(0, N_t, chunk_size):
        d3d_batch = dist_3d[i:i+chunk_size]
        diff = torch.abs(dist_2d.unsqueeze(0) - d3d_batch.unsqueeze(1))
        cost[i:i+chunk_size] = -(diff < xdf_inlier_thres).sum(-1).sum(-1)

    # Find top-K with rotation diversity: for the top rotations, try
    # multiple translations per rotation so ICP has better starting points.
    best_per_rot_cost, best_per_rot_ti = cost.min(dim=0)  # (N_r,)
    rot_order = best_per_rot_cost.argsort()

    n_top_rot = min(5, N_r)  # Top 5 rotations
    n_trans_per_rot = max(1, top_k // n_top_rot)  # Translations per rotation

    selected = []
    for ri in rot_order[:n_top_rot].tolist():
        # Top translations for this rotation
        ti_order = cost[:, ri].argsort()
        for j in range(min(n_trans_per_rot, ti_order.shape[0])):
            ti = ti_order[j].item()
            if (ti, ri) not in selected:
                selected.append((ti, ri))
            if len(selected) >= top_k:
                break
        if len(selected) >= top_k:
            break

    results = []
    for ti, ri in selected:
        R_world = rotations[ri] @ canonical_rot
        t_world = trans_candidates[ti]
        results.append({
            'R': R_world,
            't': t_world,
            'rot_idx': ri,
            'cost': cost[ti, ri].item(),
            'inter_2d': inter_2d_per_rot[ri],
            'inter_2d_mask': inter_2d_mask_per_rot[ri],
            'inter_2d_idx': inter_2d_idx_per_rot[ri],
        })

    return results, cost


# ============================================================
# XDF coarse search (convenience wrapper)
# ============================================================

def xdf_coarse_search(
    rotations, perms, principal_3d,
    edge_2d, starts, ends, dirs,
    inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot,
    inter_3d, inter_3d_mask,
    trans_candidates, query_pts,
    top_k=5, xdf_inlier_thres=0.1, point_gamma=0.2,
    inlier_thres_2d=0.05, inlier_thres_3d=0.05, chunk_size=200
):
    """Canonical frame precomputation + inlier-counting XDF search.

    Convenience wrapper that calls precompute_xdf_3d() then
    xdf_coarse_search_from_precomputed(). For multi-room use,
    call these two functions separately to share 3D precomputation.

    Args:
        rotations: (N_r, 3, 3) rotation candidates
        perms: (N_r, 3) permutation indices
        principal_3d: (3, 3) 3D principal directions
        edge_2d: (N_2D, 9) 2D line segments
        starts, ends, dirs: (N_3D, 3) 3D line geometry
        inter_2d_per_rot: list of N_r tensors — 2D intersections per rotation
        inter_2d_mask_per_rot: list of N_r tensors — group masks
        inter_2d_idx_per_rot: list of N_r tensors — line-pair indices
        inter_3d: (N_3D_inter, 3) concatenated 3D intersection points
        inter_3d_mask: (N_3D_inter, 3) bool group mask
        trans_candidates: (N_t, 3) translation candidates
        query_pts: (N_q, 3) icosphere query points
        top_k: number of top poses to return
        xdf_inlier_thres: inlier counting threshold
        point_gamma: PDF power exponent
        inlier_thres_2d: 2D line classification threshold
        inlier_thres_3d: 3D line classification threshold
        chunk_size: translation batch size for memory efficiency

    Returns:
        list of top_k dicts with keys: R, t, cost, rot_idx,
        inter_2d, inter_2d_mask, inter_2d_idx
    """
    precomputed_3d = precompute_xdf_3d(
        principal_3d, starts, ends, dirs, inter_3d, inter_3d_mask,
        trans_candidates, query_pts,
        inlier_thres_3d=inlier_thres_3d, chunk_size=chunk_size)
    return xdf_coarse_search_from_precomputed(
        precomputed_3d, rotations, perms, edge_2d,
        inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot,
        trans_candidates, query_pts,
        top_k=top_k, xdf_inlier_thres=xdf_inlier_thres,
        point_gamma=point_gamma, inlier_thres_2d=inlier_thres_2d,
        chunk_size=chunk_size)
