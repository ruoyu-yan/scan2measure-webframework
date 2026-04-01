"""
Test FGPL pose estimation (Steps 2 & 3) using our own extracted features.

Feeds our 3D line map (3d_line_map.pkl) and 2D features (fgpl_features.json)
into FGPL's native XDF coarse search + Sphere ICP refinement code.
"""

import os
import sys
import json
import time
import pickle
from pathlib import Path

# Force CPU — FGPL's internal code hardcodes cuda:0 in XDFCanonicalPrecompute,
# and torch.det triggers nvrtc compilation failures on some GPU architectures.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
POINT_CLOUD_NAME = "tmb_office1"
ROOM_NAME = "TMB_office1"
ROOM_KEY = "test_room"

MAP_3D_PATH = PROJECT_ROOT / "data" / "debug_renderer" / POINT_CLOUD_NAME / "3d_line_map.pkl"
FEATURES_2D_PATH = PROJECT_ROOT / "data" / "pano" / "2d_feature_extracted" / f"{ROOM_NAME}_v2" / "fgpl_features.json"
CONFIG_PATH = PROJECT_ROOT / "panoramic-localization" / "config" / "omniscenes_fgpl.ini"
OUTPUT_DIR = PROJECT_ROOT / "data" / "pose_estimates" / ROOM_NAME

# Add panoramic-localization to import path
PANO_LOC_DIR = PROJECT_ROOT / "panoramic-localization"
sys.path.insert(0, str(PANO_LOC_DIR))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Import FGPL modules (after sys.path setup)
    from parse_utils import parse_ini
    from dict_utils import get_init_dict_fgpl
    from utils import generate_trans_points
    from fgpl.xdf_canonical_precompute import XDFCanonicalPrecompute
    from fgpl.pose_estimation import xdf_cost, refine_from_sphere_icp
    from fgpl.line_intersection import intersections_2d, intersections_idx

    cfg = parse_ini(str(CONFIG_PATH))

    # Algorithm configs (from localize_single.py)
    top_k_candidate = getattr(cfg, "top_k_candidate", 1)
    inlier_thres_2d = getattr(cfg, "inlier_thres_2d", 0.5)
    intersect_thres_2d = getattr(cfg, "intersect_thres_2d", 0.1)
    point_gamma = getattr(cfg, "point_gamma", 0.2)

    # Optimization configs
    opt_dict = {
        "optimizer": getattr(cfg, "optimizer", "Adam"),
        "total_iter": getattr(cfg, "total_iter", 100),
        "step_size": getattr(cfg, "step_size", 0.1),
        "decay_patience": getattr(cfg, "decay_patience", 5),
        "decay_factor": getattr(cfg, "decay_factor", 0.9),
        "nn_dist_thres": getattr(cfg, "nn_dist_thres", 0.5),
        "match_thres": getattr(cfg, "match_thres", 0),
    }

    # ── Step 1: Load 3D features ──────────────────────────────────────
    print("STEP 1: Loading 3D features")
    t0 = time.time()
    with open(MAP_3D_PATH, "rb") as f:
        map_3d = pickle.load(f)

    dirs = map_3d["dirs"].to(device)
    starts = map_3d["starts"].to(device)
    ends = map_3d["ends"].to(device)
    dense_dirs = map_3d["dense_dirs"].to(device)
    dense_starts = map_3d["dense_starts"].to(device)
    dense_ends = map_3d["dense_ends"].to(device)
    principal_3d = map_3d["principal_3d"].to(device)
    inter_3d = map_3d["inter_3d"].to(device)
    inter_3d_idx = map_3d["inter_3d_idx"].to(device)
    inter_3d_mask = map_3d["inter_3d_mask"].to(device)

    print(f"  Sparse lines: {dirs.shape[0]}, Dense lines: {dense_dirs.shape[0]}")
    print(f"  3D intersections: {inter_3d.shape[0]}")
    print(f"  Loaded in {time.time() - t0:.3f}s\n")

    # ── Step 2: Generate translation candidates ───────────────────────
    print("STEP 2: Generating translation candidates")
    t0 = time.time()
    xyz = (starts + ends) / 2  # midpoints of sparse lines
    init_dict = get_init_dict_fgpl(cfg)
    trans_tensors = generate_trans_points(xyz, init_dict, device=device)

    # Chamfer filter: keep points far from point cloud (map_utils.py:188-191)
    sample_xyz = xyz[torch.randperm(xyz.shape[0])[:max(1, xyz.shape[0] // 100)]]
    sample_cmf = (trans_tensors.unsqueeze(1) - sample_xyz.unsqueeze(0)).norm(dim=-1).min(dim=-1).values
    trans_tensors = trans_tensors[sample_cmf > 0.3]
    print(f"  Translation candidates: {trans_tensors.shape[0]}")
    print(f"  Generated in {time.time() - t0:.3f}s\n")

    # ── Step 3: Build map_dict ────────────────────────────────────────
    map_dict = {
        ROOM_KEY: {
            "dirs": dirs,
            "starts": starts,
            "ends": ends,
            "dense_dirs": dense_dirs,
            "dense_starts": dense_starts,
            "dense_ends": dense_ends,
            "principal_3d": principal_3d,
            "inter_3d": inter_3d,
            "inter_3d_idx": inter_3d_idx,
            "inter_3d_mask": inter_3d_mask,
            "trans_tensors": trans_tensors,
        }
    }

    # ── Step 4: Load 2D features ──────────────────────────────────────
    print("STEP 3: Loading 2D features")
    t0 = time.time()
    with open(FEATURES_2D_PATH) as f:
        feat_2d = json.load(f)

    edge_lines = torch.tensor(feat_2d["lines"], dtype=torch.float32, device=device)
    principal_2d = torch.tensor(feat_2d["principal_2d"], dtype=torch.float32, device=device)
    print(f"  2D lines: {edge_lines.shape}")
    print(f"  Principal 2D det: {torch.det(principal_2d.cpu()).item():.4f}")
    print(f"  Loaded in {time.time() - t0:.3f}s\n")

    # ── Step 5: Build 24 rotation candidates ──────────────────────────
    # Matches localize_single.py:69-81
    from itertools import permutations as iter_perms
    num_principal = 3
    perms = list(iter_perms(range(num_principal)))
    perms = torch.tensor(perms, device=device, dtype=torch.long)
    bin_mask = torch.ones([len(perms) * 4, 3, 1], device=device)

    for perm_idx, perm in enumerate(perms):
        for idx in range(4):
            bin_mask[perm_idx * 4 + idx, 0, 0] = (-1) ** (idx // 2)
            bin_mask[perm_idx * 4 + idx, 1, 0] = (-1) ** (idx % 2)
            bin_mask[perm_idx * 4 + idx, 2, 0] = (-1) ** (idx // 2 + idx % 2)
            if perm_idx in [1, 2, 5]:
                bin_mask[perm_idx * 4 + idx, 2, 0] *= -1

    perms = torch.repeat_interleave(
        perms,
        repeats=torch.tensor([4] * len(perms), dtype=torch.long, device=device),
        dim=0,
    )
    print(f"STEP 4: 24 rotation candidates built (perms: {perms.shape})\n")

    # ── Step 6: Compute 2D intersections via FGPL ─────────────────────
    print("STEP 5: Computing 2D intersections via FGPL")
    t0 = time.time()
    raw_inter_2d, raw_inter_2d_idx = intersections_2d(
        edge_lines, principal_2d,
        inlier_thres=inlier_thres_2d,
        intersect_thres=intersect_thres_2d,
        return_idx=True,
    )
    for i, group in enumerate(raw_inter_2d):
        print(f"  Group {i}∩{(i+1)%3}: {group.shape[0]} intersections")
    print(f"  Computed in {time.time() - t0:.3f}s\n")

    # ── Step 7: Rearrange intersections for 24 rotations ──────────────
    # Matches localize_single.py:160-178
    print("STEP 6: Rearranging intersections for 24 rotations")
    t0 = time.time()
    full_inter_2d = []
    full_inter_2d_mask = []
    full_inter_2d_idx = []

    for perm_idx in range(perms.shape[0]):
        principal_perm = perms[perm_idx].tolist()
        intersection_perm = [
            intersections_idx(principal_perm[i % 3], principal_perm[(i + 1) % 3])
            for i in range(3)
        ]

        inter_2d_rot = torch.cat([raw_inter_2d[p_idx] for p_idx in intersection_perm], dim=0)
        full_inter_2d.append(inter_2d_rot)

        inter_2d_rot_idx = torch.cat([raw_inter_2d_idx[p_idx] for p_idx in intersection_perm], dim=0)
        full_inter_2d_idx.append(inter_2d_rot_idx)

        mask_2d_rot = []
        for k, p_idx in enumerate(intersection_perm):
            mask_temp = torch.zeros_like(raw_inter_2d[p_idx])
            mask_temp[:, k] = 1
            mask_temp[:, (k + 1) % 3] = 1
            mask_2d_rot.append(mask_temp.bool())
        mask_2d_rot = torch.cat(mask_2d_rot, dim=0)
        full_inter_2d_mask.append(mask_2d_rot)

    print(f"  Rearranged in {time.time() - t0:.3f}s\n")

    # ── Step 8: Precompute distance functions ─────────────────────────
    print("STEP 7: Precomputing XDF distance functions")
    t0 = time.time()
    curr_map_room_list = [ROOM_KEY]

    xdf_precompute = XDFCanonicalPrecompute(cfg, str(OUTPUT_DIR), map_dict)

    # 3D LDF + PDF
    precomputed_dist_3d, precomputed_mask_3d = xdf_precompute.generate_ldf_3d()
    precomputed_point_dist_3d = xdf_precompute.generate_pdf_3d()
    print(f"  3D LDF shape: {precomputed_dist_3d[ROOM_KEY].shape}")
    print(f"  3D PDF shape: {precomputed_point_dist_3d[ROOM_KEY].shape}")

    # 2D LDF — pass full edge_lines (no sparse/dense split in our features)
    precomputed_rot, precomputed_mask_2d, precomputed_dist_2d = \
        xdf_precompute.generate_ldf_2d(principal_2d, edge_lines, perms, bin_mask, single_pose_compute=True)
    print(f"  2D LDF shape: {precomputed_dist_2d[ROOM_KEY].shape}")

    # 2D PDF
    total_full_inter_2d = torch.stack(full_inter_2d, dim=0)
    total_full_inter_2d_mask = torch.stack(full_inter_2d_mask, dim=0)
    precomputed_point_dist_2d = xdf_precompute.generate_pdf_2d(
        total_full_inter_2d, total_full_inter_2d_mask, precomputed_rot, single_pose_compute=True,
    )
    print(f"  2D PDF shape: {precomputed_point_dist_2d[ROOM_KEY].shape}")
    print(f"  Precomputed in {time.time() - t0:.3f}s\n")

    # ── Step 9: XDF coarse search ─────────────────────────────────────
    print("STEP 8: XDF coarse pose search")
    t0 = time.time()

    room = ROOM_KEY
    estim_rot = precomputed_rot[room].to(device)
    batch_mask_2d = precomputed_mask_2d[room].to(device)
    batch_mask_3d = precomputed_mask_3d[room].to(device)
    N_r = batch_mask_2d.shape[0]
    batch_mask_3d = batch_mask_3d[None, ...].repeat(N_r, 1, 1)

    cost_mtx = xdf_cost(
        map_dict[room]["trans_tensors"],
        estim_rot,
        precomputed_dist_3d=precomputed_dist_3d[room].to(device),
        precomputed_dist_2d=precomputed_dist_2d[room].to(device),
        precomputed_point_dist_3d=precomputed_point_dist_3d[room].to(device),
        precomputed_point_dist_2d=precomputed_point_dist_2d[room].to(device),
        point_gamma=point_gamma,
    )

    # Extract top-K poses
    min_inds = cost_mtx.flatten().argsort()[:top_k_candidate]
    trimmed_trans = map_dict[room]["trans_tensors"][min_inds // len(estim_rot)]
    trimmed_rot = estim_rot[min_inds % len(estim_rot)]

    # Un-canonicalize rotations (localize_single.py:277)
    trimmed_rot = trimmed_rot @ xdf_precompute.canonical_rot_3d[room].unsqueeze(0)

    best_coarse_cost = cost_mtx.flatten()[min_inds[0]].item()
    print(f"  Cost matrix shape: {cost_mtx.shape}")
    print(f"  Best coarse cost: {best_coarse_cost:.4f}")
    print(f"  Best translation idx: {(min_inds[0] // len(estim_rot)).item()}, rotation idx: {(min_inds[0] % len(estim_rot)).item()}")
    print(f"  Searched in {time.time() - t0:.3f}s\n")

    # ── Step 10: Prepare refinement inputs ────────────────────────────
    # Select 2D intersections for top-K rotation indices
    inter_2d_sel = [full_inter_2d[i] for i in min_inds % len(estim_rot)]
    inter_2d_mask_sel = [full_inter_2d_mask[i] for i in min_inds % len(estim_rot)]
    inter_2d_idx_sel = [full_inter_2d_idx[i] for i in min_inds % len(estim_rot)]

    # 3D intersections (same for all candidates in single-room mode)
    cand_inter_3d = [map_dict[room]["inter_3d"] for _ in range(top_k_candidate)]
    cand_inter_3d_mask = [map_dict[room]["inter_3d_mask"] for _ in range(top_k_candidate)]
    cand_inter_3d_idx = [map_dict[room]["inter_3d_idx"] for _ in range(top_k_candidate)]

    # Line dict for rotation refinement
    line_dict = {
        "dense_dirs": [map_dict[room]["dense_dirs"] for _ in range(top_k_candidate)],
        "edge_lines": edge_lines,
        "inter_2d_idx": inter_2d_idx_sel,
        "inter_3d_idx": cand_inter_3d_idx,
    }

    # ── Step 11: Sphere ICP refinement ────────────────────────────────
    print("STEP 9: Sphere ICP refinement")
    t0 = time.time()
    refined_trans, refined_rot, best_idx = refine_from_sphere_icp(
        trimmed_trans, trimmed_rot,
        inter_2d_sel, cand_inter_3d, opt_dict,
        inter_2d_mask_sel, cand_inter_3d_mask,
        line_dict=line_dict,
    )
    print(f"  Refined in {time.time() - t0:.3f}s\n")

    # ── Step 12: Output ───────────────────────────────────────────────
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    trans_np = refined_trans.squeeze().cpu().numpy()
    rot_np = refined_rot.cpu().numpy()
    det = np.linalg.det(rot_np)

    print(f"\nTranslation: [{trans_np[0]:.4f}, {trans_np[1]:.4f}, {trans_np[2]:.4f}]")
    print(f"\nRotation matrix:")
    for row in rot_np:
        print(f"  [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")
    print(f"\ndet(R) = {det:.6f}")
    print(f"Coarse XDF cost: {best_coarse_cost:.4f}")

    # Save result
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "camera_pose_fgpl_native.json"
    result = {
        "rotation": rot_np.tolist(),
        "translation": trans_np.tolist(),
        "xdf_cost_coarse": best_coarse_cost,
        "principal_3d": principal_3d.cpu().numpy().tolist(),
        "principal_2d": principal_2d.cpu().numpy().tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {output_path}")

    # Compare with existing camera_pose.json
    existing_path = OUTPUT_DIR / "camera_pose.json"
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        ex_trans = np.array(existing["translation"])
        ex_rot = np.array(existing["rotation"])

        trans_diff = np.linalg.norm(trans_np - ex_trans)
        # Rotation angle difference
        R_diff = rot_np @ ex_rot.T
        trace = np.clip(np.trace(R_diff), -1.0, 3.0)
        rot_angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))

        print(f"\nComparison with existing camera_pose.json:")
        print(f"  Translation diff: {trans_diff:.4f} m")
        print(f"  Rotation diff:    {rot_angle:.2f}°")


if __name__ == "__main__":
    main()
