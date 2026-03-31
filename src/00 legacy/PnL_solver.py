"""
PnL_solver.py — ICP-style pose refinement utility for feature_matching.py

Contains:
  refine_pose_icp   — V1: optimize translation only (rotation fixed)
  refine_pose_full  — V2: optimize translation then rotation (FGPL-style)
"""

import numpy as np
import torch
import torch.optim as optim


# ============================================================
# YPR ↔ rotation matrix  (ported from panoramic-localization/utils.py)
# ============================================================

def _rot_from_ypr(ypr):
    """(3,) tensor → (3,3) rotation matrix. Differentiable."""
    yaw, pitch, roll = ypr
    y, p, r = yaw.unsqueeze(0), pitch.unsqueeze(0), roll.unsqueeze(0)
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


def _ypr_from_rot(R):
    """(3,3) tensor → (3,) ypr tensor."""
    yaw   = torch.atan2(R[1, 0], R[0, 0] + 1e-6)
    pitch = torch.arcsin(-R[2, 0].clamp(-1, 1))
    roll  = torch.atan2(R[2, 1], R[2, 2])
    return torch.tensor([yaw.item(), pitch.item(), roll.item()],
                        device=R.device, dtype=R.dtype)


def refine_pose_icp(R_init, t_init, inter_2d, inter_3d,
                    n_iters=100, lr=0.1, nn_dist_thres=0.5):
    """
    Refine camera translation via gradient descent on sphere ICP.

    R is kept fixed (already solved analytically from principal directions).
    Each pair in inter_2d[k] / inter_3d[k] are intersections from principal
    direction pair k (k in 0,1,2).

    Args:
        R_init:        (3,3) ndarray — world-to-camera rotation
        t_init:        (3,)  ndarray — camera position in world frame
        inter_2d:      list of 3 ndarrays, each (M_k, 3) unit sphere points
        inter_3d:      list of 3 ndarrays, each (N_k, 3) world-space 3D points
        n_iters:       number of gradient steps
        lr:            Adam learning rate
        nn_dist_thres: max L1 distance on sphere for mutual NN matching

    Returns:
        R (3,3 ndarray), t (3, ndarray)
    """
    t = torch.tensor(t_init.copy(), dtype=torch.float32, requires_grad=True)
    R = torch.tensor(R_init.copy(), dtype=torch.float32)
    optimizer = optim.Adam([t], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.9)

    i2d_tensors = [torch.tensor(x.copy(), dtype=torch.float32) for x in inter_2d]
    i3d_tensors = [torch.tensor(x.copy(), dtype=torch.float32) for x in inter_3d]

    for _ in range(n_iters):
        optimizer.zero_grad()
        total_loss = torch.zeros(1)

        for k in range(3):
            i2d = i2d_tensors[k]
            i3d = i3d_tensors[k]
            if i2d.shape[0] == 0 or i3d.shape[0] == 0:
                continue

            # Project 3D intersections into camera sphere
            i3d_cam = (i3d - t) @ R.T
            norms = i3d_cam.norm(dim=1, keepdim=True).clamp(min=1e-6)
            i3d_sph = i3d_cam / norms  # (N, 3)

            # L2 distance matrix on sphere
            dists = torch.cdist(i2d, i3d_sph)  # (M, N)

            # Mutual nearest-neighbour matching
            nn_2to3 = dists.argmin(dim=1)   # (M,)
            nn_3to2 = dists.argmin(dim=0)   # (N,)

            m_idx = torch.arange(i2d.shape[0])
            mutual = (nn_3to2[nn_2to3] == m_idx) & \
                     (dists[m_idx, nn_2to3] < nn_dist_thres)

            if mutual.sum() == 0:
                continue

            mi = m_idx[mutual]
            ni = nn_2to3[mutual]
            loss = (i2d[mi] - i3d_sph[ni]).abs().mean()
            total_loss = total_loss + loss

        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

    return R.numpy(), t.detach().numpy()


# ============================================================
# V2: Full refinement (translation + rotation)
# ============================================================

def refine_pose_full(R_init, t_init, inter_2d, inter_3d,
                     inter_2d_mask=None, inter_3d_mask=None,
                     line_dirs_2d=None, line_dirs_3d=None,
                     inter_2d_idx=None, inter_3d_idx=None,
                     n_iters_t=100, n_iters_r=50, lr=0.1,
                     nn_dist_thres=0.5, rot_inlier_thres=0.2):
    """
    FGPL-style refinement: Phase 1 optimizes translation, Phase 2 optimizes rotation.

    Uses grouped mutual-NN matching (by intersection mask) plus fallback full-distance NN.

    Args:
        R_init:        (3,3) ndarray — world-to-camera rotation
        t_init:        (3,)  ndarray — camera position in world frame
        inter_2d:      (M, 3) ndarray — 2D intersection sphere points (all groups concatenated)
        inter_3d:      (N, 3) ndarray — 3D intersection world points (all groups concatenated)
        inter_2d_mask: (M, 3) bool ndarray — group membership for 2D intersections
        inter_3d_mask: (N, 3) bool ndarray — group membership for 3D intersections
        line_dirs_2d:  (L2, 3) ndarray — 2D line normals (for rotation refinement)
        line_dirs_3d:  (L3, 3) ndarray — 3D line directions (for rotation refinement)
        inter_2d_idx:  (M, 2) int ndarray — line indices producing each 2D intersection
        inter_3d_idx:  (N, 2) int ndarray — line indices producing each 3D intersection
        n_iters_t:     translation optimization iterations
        n_iters_r:     rotation optimization iterations
        lr:            Adam learning rate
        nn_dist_thres: max sphere distance for mutual NN matching
        rot_inlier_thres: max cost for rotation inlier

    Returns:
        R (3,3 ndarray), t (3, ndarray), matched_pairs list of 3 arrays each (P_k, 2)
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
    ypr = _ypr_from_rot(R_t)
    for _ in range(n_iters_t):
        optimizer.zero_grad()
        R_cur = _rot_from_ypr(ypr)

        i3d_cam = (pts_3d - t_param) @ R_cur.T
        i3d_sph = i3d_cam / i3d_cam.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        with torch.no_grad():
            match_2d_list = []
            match_3d_list = []

            if has_masks:
                # Grouped mutual-NN matching (mask_idx k → intersections NOT from direction k)
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

        # Save grouped matches for rotation refinement
        if has_masks and len(match_2d_list) > 1:
            track_2d_idx = torch.cat(match_2d_list[:-1])
            track_3d_idx = torch.cat(match_3d_list[:-1])
        elif all_m2.shape[0] > 0:
            track_2d_idx = all_m2
            track_3d_idx = all_m3

    # --- Phase 2: rotation (optional) ---
    can_refine_rot = (line_dirs_2d is not None and line_dirs_3d is not None and
                      inter_2d_idx is not None and inter_3d_idx is not None and
                      track_2d_idx is not None and track_3d_idx is not None and
                      track_2d_idx.shape[0] > 0)

    if can_refine_rot:
        ypr_param = _ypr_from_rot(_rot_from_ypr(ypr).detach()).requires_grad_(True)
        opt_r = optim.Adam([ypr_param], lr=lr)
        sch_r = optim.lr_scheduler.ReduceLROnPlateau(opt_r, patience=5, factor=0.9)

        dirs_2d_t = torch.tensor(line_dirs_2d, dtype=torch.float32)
        dirs_3d_t = torch.tensor(line_dirs_3d, dtype=torch.float32)
        idx_2d_t = torch.tensor(inter_2d_idx, dtype=torch.long)
        idx_3d_t = torch.tensor(inter_3d_idx, dtype=torch.long)

        # Extract matched line directions from intersection indices
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
            R_cur = _rot_from_ypr(ypr_param)
            rot_prod = match_dirs_2d @ R_cur
            cost_r = torch.abs(rot_prod.unsqueeze(-1) * match_dirs_3d_pair).sum(1).min(-1).values
            cost_r = cost_r[cost_r < rot_inlier_thres].mean() if (cost_r < rot_inlier_thres).any() else cost_r.mean()
            cost_r.backward()
            opt_r.step()
            sch_r.step(cost_r.item())

        R_refined = _rot_from_ypr(ypr_param).detach()
    else:
        R_refined = _rot_from_ypr(ypr).detach()

    t_refined = t_param.detach()

    # --- Safety check ---
    t_delta = (t_refined - torch.tensor(t_init, dtype=torch.float32)).norm().item()
    R_delta_trace = (R_refined @ torch.tensor(R_init, dtype=torch.float32).T).trace()
    R_delta_trace = max(-1.0, min(3.0, R_delta_trace.item()))
    r_angle = np.degrees(np.abs(np.arccos((R_delta_trace - 1) / 2)))

    if t_delta > 1.0 or r_angle > 60:
        R_out = R_init.copy()
        t_out = t_init.copy()
    else:
        R_out = R_refined.numpy()
        t_out = t_refined.numpy()

    # --- Build matched_pairs per group ---
    matched_pairs = [np.zeros((0, 2), dtype=int) for _ in range(3)]
    if track_2d_idx is not None and track_3d_idx is not None and has_masks:
        for k in range(3):
            # Group k contains intersections from directions (k) and (k+1) → mask column k is 0
            pairs = []
            for i in range(track_2d_idx.shape[0]):
                i2 = track_2d_idx[i].item()
                i3 = track_3d_idx[i].item()
                if not mask_2d[i2, k] and not mask_3d[i3, k]:
                    pairs.append([i2, i3])
            if pairs:
                matched_pairs[k] = np.array(pairs, dtype=int)

    return R_out, t_out, matched_pairs
