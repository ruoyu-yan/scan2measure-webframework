"""Sphere ICP pose refinement.

Two-phase refinement on the unit sphere:
  Phase 1: Optimize translation via grouped mutual-NN matching (Adam).
  Phase 2: Optimize rotation via YPR parameterization on matched line directions.

Clean reimplementation of PnL_solver.refine_pose_full() — existing
PnL_solver.py is not modified. No imports from panoramic-localization/.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_search import rot_from_ypr, ypr_from_rot


def refine_pose(
    R_init, t_init,
    inter_2d, inter_3d,
    inter_2d_mask, inter_3d_mask,
    line_normals_2d=None, line_dirs_3d=None,
    inter_2d_idx=None, inter_3d_idx=None,
    n_iters_t=100, n_iters_r=50, lr=0.1,
    nn_dist_thres=0.5, rot_inlier_thres=0.2,
    max_t_drift=1.0, max_r_drift_deg=60.0
):
    """Two-phase ICP on unit sphere.

    Phase 1 (translation): Adam optimizer, grouped mutual-NN matching.
    Phase 2 (rotation): YPR parameterization, line direction alignment cost.
    Safety: reverts if drift > thresholds.

    Args:
        R_init:          (3, 3) ndarray — world-to-camera rotation
        t_init:          (3,)  ndarray — camera position in world frame
        inter_2d:        (M, 3) ndarray — all groups concatenated
        inter_3d:        (N, 3) ndarray — all groups concatenated
        inter_2d_mask:   (M, 3) bool ndarray — group membership
        inter_3d_mask:   (N, 3) bool ndarray — group membership
        line_normals_2d: (L2, 3) ndarray — 2D line normals (for rotation phase)
        line_dirs_3d:    (L3, 3) ndarray — 3D line directions
        inter_2d_idx:    (M, 2) int ndarray — line indices for each 2D intersection
        inter_3d_idx:    (N, 2) int ndarray — line indices for each 3D intersection
        n_iters_t:       translation optimization iterations
        n_iters_r:       rotation optimization iterations
        lr:              Adam learning rate
        nn_dist_thres:   max sphere distance for mutual NN matching
        rot_inlier_thres: max cost for rotation inlier
        max_t_drift:     safety threshold for translation drift
        max_r_drift_deg: safety threshold for rotation drift (degrees)

    Returns:
        R:             (3, 3) ndarray
        t:             (3,) ndarray
        matched_pairs: list of 3 arrays, each (P_k, 2) int
    """
    R_t = torch.tensor(R_init.copy(), dtype=torch.float32)
    t_param = torch.tensor(t_init.copy(), dtype=torch.float32, requires_grad=True)

    pts_2d = torch.tensor(inter_2d.copy(), dtype=torch.float32)
    pts_3d = torch.tensor(inter_3d.copy(), dtype=torch.float32)

    has_masks = inter_2d_mask is not None and inter_3d_mask is not None
    if has_masks:
        mask_2d = torch.tensor(inter_2d_mask, dtype=torch.bool)
        mask_3d = torch.tensor(inter_3d_mask, dtype=torch.bool)
        inv_mask_2d = ~mask_2d
        inv_mask_3d = ~mask_3d

    range_2d = torch.arange(pts_2d.shape[0])
    range_3d = torch.arange(pts_3d.shape[0])

    optimizer = optim.Adam([t_param], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.9)

    track_2d_idx = None
    track_3d_idx = None

    # --- Phase 1: translation ---
    ypr = ypr_from_rot(R_t)
    for _ in range(n_iters_t):
        optimizer.zero_grad()
        R_cur = rot_from_ypr(ypr)

        i3d_cam = (pts_3d - t_param) @ R_cur.T
        i3d_sph = i3d_cam / i3d_cam.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        with torch.no_grad():
            match_2d_list = []
            match_3d_list = []

            if has_masks:
                for k in [2, 0, 1]:
                    m2_idx = range_2d[inv_mask_2d[:, k]]
                    m3_idx = range_3d[inv_mask_3d[:, k]]
                    if m2_idx.shape[0] == 0 or m3_idx.shape[0] == 0:
                        continue
                    m2_range = torch.arange(m2_idx.shape[0])
                    dm = (pts_2d[inv_mask_2d[:, k]].unsqueeze(1) -
                          i3d_sph[inv_mask_3d[:, k]].unsqueeze(0)).norm(dim=-1)
                    nn_2to3 = dm.argmin(-1)
                    nn_3to2 = dm.argmin(0)
                    valid = (nn_3to2[nn_2to3] == m2_range) & (dm.min(-1).values < nn_dist_thres)
                    match_2d_list.append(m2_idx[valid])
                    match_3d_list.append(m3_idx[nn_2to3[valid]])

            # Fallback: full distance NN with tighter threshold
            full_dm = (pts_2d.unsqueeze(1) - i3d_sph.unsqueeze(0)).norm(dim=-1)
            nn_2to3_f = full_dm.argmin(-1)
            nn_3to2_f = full_dm.argmin(0)
            valid_f = (nn_3to2_f[nn_2to3_f] == range_2d) & (full_dm.min(-1).values < nn_dist_thres / 5)
            match_2d_list.append(range_2d[valid_f])
            match_3d_list.append(nn_2to3_f[valid_f])

            all_m2 = torch.cat(match_2d_list)
            all_m3 = torch.cat(match_3d_list)
            if all_m2.shape[0] == 0:
                break

        cost = (pts_2d[all_m2] - i3d_sph[all_m3]).abs().sum(-1).mean()
        cost.backward()
        optimizer.step()
        scheduler.step(cost.item())

        if has_masks and len(match_2d_list) > 1:
            track_2d_idx = torch.cat(match_2d_list[:-1])
            track_3d_idx = torch.cat(match_3d_list[:-1])
        elif all_m2.shape[0] > 0:
            track_2d_idx = all_m2
            track_3d_idx = all_m3

    # --- Phase 2: rotation (optional) ---
    can_refine_rot = (line_normals_2d is not None and line_dirs_3d is not None and
                      inter_2d_idx is not None and inter_3d_idx is not None and
                      track_2d_idx is not None and track_3d_idx is not None and
                      track_2d_idx.shape[0] > 0)

    if can_refine_rot:
        ypr_param = ypr_from_rot(rot_from_ypr(ypr).detach()).requires_grad_(True)
        opt_r = optim.Adam([ypr_param], lr=lr)
        sch_r = optim.lr_scheduler.ReduceLROnPlateau(opt_r, patience=5, factor=0.9)

        dirs_2d_t = torch.tensor(line_normals_2d, dtype=torch.float32)
        dirs_3d_t = torch.tensor(line_dirs_3d, dtype=torch.float32)
        idx_2d_t = torch.tensor(inter_2d_idx, dtype=torch.long)
        idx_3d_t = torch.tensor(inter_3d_idx, dtype=torch.long)

        li2d_0 = dirs_2d_t[idx_2d_t[track_2d_idx, 0]]
        li2d_1 = dirs_2d_t[idx_2d_t[track_2d_idx, 1]]
        li3d_0 = dirs_3d_t[idx_3d_t[track_3d_idx, 0]]
        li3d_1 = dirs_3d_t[idx_3d_t[track_3d_idx, 1]]

        match_dirs_2d = torch.cat([li2d_0, li2d_1], dim=0)
        match_dirs_3d = torch.cat([li3d_0, li3d_1], dim=0)
        match_dirs_3d_inv = torch.cat([li3d_1, li3d_0], dim=0)
        match_dirs_3d_pair = torch.stack([match_dirs_3d, match_dirs_3d_inv], dim=-1)

        for _ in range(n_iters_r):
            opt_r.zero_grad()
            R_cur = rot_from_ypr(ypr_param)
            rot_prod = match_dirs_2d @ R_cur
            cost_r = torch.abs(rot_prod.unsqueeze(-1) * match_dirs_3d_pair).sum(1).min(-1).values
            cost_r = cost_r[cost_r < rot_inlier_thres].mean() \
                if (cost_r < rot_inlier_thres).any() else cost_r.mean()
            cost_r.backward()
            opt_r.step()
            sch_r.step(cost_r.item())

        R_refined = rot_from_ypr(ypr_param).detach()
    else:
        R_refined = rot_from_ypr(ypr).detach()

    t_refined = t_param.detach()

    # --- Safety check ---
    t_delta = (t_refined - torch.tensor(t_init, dtype=torch.float32)).norm().item()
    R_delta_trace = (R_refined @ torch.tensor(R_init, dtype=torch.float32).T).trace()
    R_delta_trace = max(-1.0, min(3.0, R_delta_trace.item()))
    r_angle = np.degrees(np.abs(np.arccos((R_delta_trace - 1) / 2)))

    if t_delta > max_t_drift or r_angle > max_r_drift_deg:
        R_out = R_init.copy()
        t_out = t_init.copy()
    else:
        R_out = R_refined.numpy()
        t_out = t_refined.numpy()

    # --- Build matched_pairs per group ---
    matched_pairs = [np.zeros((0, 2), dtype=int) for _ in range(3)]
    if track_2d_idx is not None and track_3d_idx is not None and has_masks:
        for k in range(3):
            pairs = []
            for i in range(track_2d_idx.shape[0]):
                i2 = track_2d_idx[i].item()
                i3 = track_3d_idx[i].item()
                if not mask_2d[i2, k] and not mask_3d[i3, k]:
                    pairs.append([i2, i3])
            if pairs:
                matched_pairs[k] = np.array(pairs, dtype=int)

    return R_out, t_out, matched_pairs
