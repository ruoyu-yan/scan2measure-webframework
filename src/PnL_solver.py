"""
PnL_solver.py — ICP-style pose refinement utility for feature_matching.py

Optimizes camera translation via gradient descent on sphere, keeping rotation fixed.
"""

import numpy as np
import torch
import torch.optim as optim


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
