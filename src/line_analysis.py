"""Vanishing point detection, line classification, intersection finding on unit sphere."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from sphere_geometry import generate_sphere_points


def extract_vanishing_points(edge_lines, vote_sphere_pts=None):
    """Icosphere voting → (3, 3) vanishing point matrix.

    Args:
        edge_lines: (N, 9) tensor [normal(3), start(3), end(3)]
        vote_sphere_pts: precomputed sphere points, or None to generate level-5

    Returns:
        (3, 3) tensor — three orthogonal vanishing directions
    """
    device = edge_lines.device
    vote_lines = edge_lines.clone().detach()

    if vote_sphere_pts is None:
        vote_sphere_pts = generate_sphere_points(5, device=device)
        vote_sphere_pts = vote_sphere_pts[:vote_sphere_pts.shape[0] // 2]

    tot_directions = []
    for _ in range(20):
        if len(vote_lines) == 0:
            break
        votes = torch.where(
            torch.abs(vote_lines[:, :3] @ vote_sphere_pts.t()) < 0.05
        )[1].bincount(minlength=vote_sphere_pts.shape[0])
        max_idx = votes.argmax()
        tot_directions.append(vote_sphere_pts[max_idx])
        outlier = (torch.abs(vote_lines[:, :3] @ vote_sphere_pts[max_idx:max_idx + 1].t()) > 0.05).squeeze()
        vote_lines = vote_lines[outlier]

    tot_directions = torch.stack(tot_directions, dim=0)

    combs = torch.combinations(torch.arange(tot_directions.shape[0]), r=3)
    comb_dirs = tot_directions[combs]
    comb_dots = torch.stack([
        (comb_dirs[:, i % 3] * comb_dirs[:, (i + 1) % 3]).sum(-1).abs()
        for i in range(3)
    ], dim=-1)
    valid = (comb_dots < 0.1).sum(-1) == 3

    if valid.sum() == 0:
        if (comb_dots < 0.15).sum() != 0:
            idx = torch.where(comb_dots < 0.15)
            v0 = comb_dirs[idx[0][0], idx[1][0]]
            v1 = comb_dirs[idx[0][0], (idx[1][0] + 1) % 3]
            v2 = torch.cross(v0, v1).unsqueeze(0)
            principal_2d = torch.cat([v0.unsqueeze(0), v1.unsqueeze(0), v2])
        else:
            v01 = tot_directions[:2]
            v2 = torch.cross(v01[0], v01[1]).unsqueeze(0)
            principal_2d = torch.cat([v01, v2])
    else:
        best = torch.where(valid)[0][0]
        principal_2d = comb_dirs[best]

    if torch.det(principal_2d) < 0:
        principal_2d[-1] *= -1

    return principal_2d


def classify_lines(edge_2d, principal_2d, inlier_thres=0.05):
    """Classify lines by which vanishing direction their normal is most perpendicular to.

    Args:
        edge_2d: (N, 9) tensor
        principal_2d: (3, 3) tensor
        inlier_thres: max dot-product to count as inlier

    Returns:
        (N, 3) bool mask — which group each line belongs to
    """
    inner = torch.abs(edge_2d[:, :3] @ principal_2d.t())  # (N, 3)
    min_mask = inner.argmin(-1, keepdim=True) == torch.arange(
        principal_2d.shape[0], device=edge_2d.device
    ).unsqueeze(0).repeat(inner.shape[0], 1)
    return (inner < inlier_thres) & min_mask


def find_intersections_2d(edge_2d, principal_2d, inlier_thres=0.3, intersect_thres=0.1):
    """Find sphere intersections of great-circle arcs from different principal groups.

    Args:
        edge_2d: (N, 9) tensor
        principal_2d: (3, 3) tensor
        inlier_thres: threshold for line classification
        intersect_thres: arc membership tolerance

    Returns:
        list of 3 tensors, each (M_k, 3) — intersection points on sphere per group pair
    """
    pi = torch.acos(torch.zeros(1, device=edge_2d.device)).item() * 2

    mask = classify_lines(edge_2d, principal_2d, inlier_thres)
    groups = [edge_2d[mask[:, i]] for i in range(3)]
    counts = [g.shape[0] for g in groups]

    arc_len = [None] * 3
    for i in range(3):
        if counts[i] > 0:
            arc_len[i] = torch.acos(
                (groups[i][:, 3:6] * groups[i][:, 6:]).sum(dim=1).clamp(-1, 1)
            )

    # Cross-product intersection candidates
    cands = [None] * 3
    for i in range(3):
        j = (i + 1) % 3
        ni, nj = counts[i], counts[j]
        if ni > 0 and nj > 0:
            n_i = groups[i][:, :3].repeat_interleave(nj, dim=0).reshape(ni, nj, 3)
            n_j = groups[j][:, :3].repeat(ni, 1).reshape(ni, nj, 3)
            c = torch.cross(n_i, n_j, dim=-1)
            c = c / (torch.norm(c, dim=-1, keepdim=True) + 1e-10)
            cands[i] = c

    results = []
    for i in range(3):
        j = (i + 1) % 3
        ni, nj = counts[i], counts[j]

        if ni > 0 and nj > 0 and cands[i] is not None:
            s0 = groups[i][:, 3:6].repeat_interleave(nj, dim=0).reshape(ni, nj, 3)
            e0 = groups[i][:, 6:].repeat_interleave(nj, dim=0).reshape(ni, nj, 3)
            s1 = groups[j][:, 3:6].repeat(ni, 1).reshape(ni, nj, 3)
            e1 = groups[j][:, 6:].repeat(ni, 1).reshape(ni, nj, 3)

            ds0 = torch.acos((cands[i] * s0).sum(-1).clamp(-1, 1))
            de0 = torch.acos((cands[i] * e0).sum(-1).clamp(-1, 1))
            ds1 = torch.acos((cands[i] * s1).sum(-1).clamp(-1, 1))
            de1 = torch.acos((cands[i] * e1).sum(-1).clamp(-1, 1))

            al_i = arc_len[i].repeat_interleave(nj).reshape(ni, nj)
            al_j = arc_len[j]

            # Check candidate d ("up")
            up_0 = (ds0 + de0 - al_i) < intersect_thres
            up_1 = (ds1 + de1 - al_j) < intersect_thres
            up_valid = up_0 & up_1
            up_pts = cands[i][up_valid]

            # Check antipodal -d ("down")
            dn_0 = ((pi - ds0) + (pi - de0) - al_i) < intersect_thres
            dn_1 = ((pi - ds1) + (pi - de1) - al_j) < intersect_thres
            dn_valid = dn_0 & dn_1
            dn_pts = -cands[i][dn_valid]

            inter = torch.cat([up_pts, dn_pts], dim=0)
            results.append(inter if inter.shape[0] > 0 else torch.zeros((0, 3), device=edge_2d.device))
        else:
            results.append(torch.zeros((0, 3), device=edge_2d.device))

    return results
