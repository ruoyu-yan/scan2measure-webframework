"""
Test FGPL native multi-room pose estimation.

Runs pose estimation for multiple panoramas against a shared multi-room 3D map
(tmb_office_one_corridor_dense). 3D precomputation happens once; 2D per-pano.
Outputs camera_pose_fgpl_native.json + side_by_side.png for each panorama.
"""

import os
import sys
import json
import time
import pickle
from pathlib import Path
from itertools import permutations as iter_perms

# Force CPU — avoids nvrtc compilation failures (matches test_FGPL_pose_estimation.py)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import torch

# ── Paths & Constants ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
POINT_CLOUD_NAME = "tmb_office_one_corridor_dense"
PANO_NAMES = ["TMB_office1", "TMB_corridor_south1", "TMB_corridor_south2"]
ROOM_KEY = "multiroom"
INLIER_THRES_2D = 0.05  # classification threshold for PDF intersections

MAP_3D_PATH = PROJECT_ROOT / "data" / "debug_renderer" / POINT_CLOUD_NAME / "3d_line_map.pkl"
CONFIG_PATH = PROJECT_ROOT / "panoramic-localization" / "config" / "omniscenes_fgpl.ini"
OUTPUT_BASE = PROJECT_ROOT / "data" / "pose_estimates" / "fgpl_native_multiroom"

# Add panoramic-localization and src to import path
PANO_LOC_DIR = PROJECT_ROOT / "panoramic-localization"
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PANO_LOC_DIR))
sys.path.insert(0, str(SRC_DIR))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Panoramas: {PANO_NAMES}")
    print(f"3D map: {POINT_CLOUD_NAME}\n")

    # Import FGPL modules (after sys.path setup)
    from parse_utils import parse_ini
    from dict_utils import get_init_dict_fgpl
    from utils import generate_trans_points
    from fgpl.xdf_canonical_precompute import XDFCanonicalPrecompute
    from fgpl.pose_estimation import xdf_cost, refine_from_sphere_icp
    from fgpl.line_intersection import intersections_2d, intersections_idx
    from visualize_pose import render_side_by_side

    cfg = parse_ini(str(CONFIG_PATH))

    # Algorithm configs
    top_k_candidate = getattr(cfg, "top_k_candidate", 1)
    inlier_thres_2d = INLIER_THRES_2D
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

    # ================================================================
    # PHASE A: One-time 3D setup
    # ================================================================
    print("=" * 60)
    print("PHASE A: One-time 3D precomputation")
    print("=" * 60)

    # ── A1: Load 3D features ──────────────────────────────────────────
    print("\nA1: Loading 3D features")
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
    print(f"  Loaded in {time.time() - t0:.3f}s")

    # ── A2: Generate translation candidates ───────────────────────────
    print("\nA2: Generating translation candidates")
    t0 = time.time()
    xyz = (starts + ends) / 2
    init_dict = get_init_dict_fgpl(cfg)
    trans_tensors = generate_trans_points(xyz, init_dict, device=device)

    # Chamfer filter
    sample_xyz = xyz[torch.randperm(xyz.shape[0])[:max(1, xyz.shape[0] // 100)]]
    sample_cmf = (trans_tensors.unsqueeze(1) - sample_xyz.unsqueeze(0)).norm(dim=-1).min(dim=-1).values
    trans_tensors = trans_tensors[sample_cmf > 0.3]
    print(f"  Translation candidates: {trans_tensors.shape[0]}")
    print(f"  Generated in {time.time() - t0:.3f}s")

    # ── A3: Build map_dict ────────────────────────────────────────────
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

    # ── A4: Create XDFCanonicalPrecompute and precompute 3D ───────────
    print("\nA4: Precomputing 3D distance functions")
    t0 = time.time()
    xdf_precompute = XDFCanonicalPrecompute(cfg, str(OUTPUT_BASE), map_dict)

    precomputed_dist_3d, precomputed_mask_3d = xdf_precompute.generate_ldf_3d()
    precomputed_point_dist_3d = xdf_precompute.generate_pdf_3d()
    print(f"  3D LDF shape: {precomputed_dist_3d[ROOM_KEY].shape}")
    print(f"  3D PDF shape: {precomputed_point_dist_3d[ROOM_KEY].shape}")
    print(f"  Precomputed in {time.time() - t0:.3f}s")

    # ── A5: Build 24 rotation candidates (shared structure) ───────────
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
    print(f"\n  24 rotation candidates built (perms: {perms.shape})")
    print(f"\nPhase A total: {time.time() - t0:.1f}s\n")

    # ================================================================
    # PHASE B: Per-panorama pose estimation
    # ================================================================
    for pano_idx, pano_name in enumerate(PANO_NAMES):
        print("=" * 60)
        print(f"PHASE B [{pano_idx+1}/{len(PANO_NAMES)}]: {pano_name}")
        print("=" * 60)

        pano_output_dir = OUTPUT_BASE / pano_name
        vis_dir = pano_output_dir / "vis"
        features_2d_path = (
            PROJECT_ROOT / "data" / "pano" / "2d_feature_extracted"
            / f"{pano_name}_v2" / "fgpl_features.json"
        )
        pano_img_path = PROJECT_ROOT / "data" / "pano" / "raw" / f"{pano_name}.jpg"

        # ── B1: Load 2D features ──────────────────────────────────────
        print(f"\nB1: Loading 2D features from {features_2d_path.name}")
        t0 = time.time()
        with open(features_2d_path) as f:
            feat_2d = json.load(f)

        edge_lines = torch.tensor(feat_2d["lines"], dtype=torch.float32, device=device)
        principal_2d = torch.tensor(feat_2d["principal_2d"], dtype=torch.float32, device=device)
        print(f"  2D lines: {edge_lines.shape}")
        print(f"  Principal 2D det: {torch.det(principal_2d.cpu()).item():.4f}")
        print(f"  Loaded in {time.time() - t0:.3f}s")

        # Classification stats
        from line_analysis import classify_lines
        mask = classify_lines(edge_lines, principal_2d, inlier_thres=inlier_thres_2d)
        n_classified = mask.any(dim=1).sum().item()
        print(f"  Classified: {n_classified}/{edge_lines.shape[0]} ({100*n_classified/edge_lines.shape[0]:.1f}%)")

        # ── B2: Compute 2D intersections ──────────────────────────────
        print("\nB2: Computing 2D intersections")
        t0 = time.time()
        raw_inter_2d, raw_inter_2d_idx = intersections_2d(
            edge_lines, principal_2d,
            inlier_thres=inlier_thres_2d,
            intersect_thres=intersect_thres_2d,
            return_idx=True,
        )
        for i, group in enumerate(raw_inter_2d):
            print(f"  Group {i}∩{(i+1)%3}: {group.shape[0]} intersections")
        print(f"  Computed in {time.time() - t0:.3f}s")

        # ── B3: Rearrange intersections for 24 rotations ──────────────
        print("\nB3: Rearranging intersections for 24 rotations")
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

        print(f"  Rearranged in {time.time() - t0:.3f}s")

        # ── B4: Precompute 2D distance functions ──────────────────────
        print("\nB4: Precomputing 2D distance functions")
        t0 = time.time()

        # 2D LDF
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
        print(f"  Precomputed in {time.time() - t0:.3f}s")

        # ── B5: XDF coarse search ─────────────────────────────────────
        print("\nB5: XDF coarse pose search")
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

        # Un-canonicalize rotations
        trimmed_rot = trimmed_rot @ xdf_precompute.canonical_rot_3d[room].unsqueeze(0)

        best_coarse_cost = cost_mtx.flatten()[min_inds[0]].item()
        print(f"  Cost matrix shape: {cost_mtx.shape}")
        print(f"  Best coarse cost: {best_coarse_cost:.4f}")
        print(f"  Best trans idx: {(min_inds[0] // len(estim_rot)).item()}, rot idx: {(min_inds[0] % len(estim_rot)).item()}")
        print(f"  Searched in {time.time() - t0:.3f}s")

        # ── B6: Prepare refinement inputs ─────────────────────────────
        inter_2d_sel = [full_inter_2d[i] for i in min_inds % len(estim_rot)]
        inter_2d_mask_sel = [full_inter_2d_mask[i] for i in min_inds % len(estim_rot)]
        inter_2d_idx_sel = [full_inter_2d_idx[i] for i in min_inds % len(estim_rot)]

        cand_inter_3d = [map_dict[room]["inter_3d"] for _ in range(top_k_candidate)]
        cand_inter_3d_mask = [map_dict[room]["inter_3d_mask"] for _ in range(top_k_candidate)]
        cand_inter_3d_idx = [map_dict[room]["inter_3d_idx"] for _ in range(top_k_candidate)]

        line_dict = {
            "dense_dirs": [map_dict[room]["dense_dirs"] for _ in range(top_k_candidate)],
            "edge_lines": edge_lines,
            "inter_2d_idx": inter_2d_idx_sel,
            "inter_3d_idx": cand_inter_3d_idx,
        }

        # ── B7: Sphere ICP refinement ─────────────────────────────────
        print("\nB6: Sphere ICP refinement")
        t0 = time.time()
        refined_trans, refined_rot, best_idx = refine_from_sphere_icp(
            trimmed_trans, trimmed_rot,
            inter_2d_sel, cand_inter_3d, opt_dict,
            inter_2d_mask_sel, cand_inter_3d_mask,
            line_dict=line_dict,
        )
        print(f"  Refined in {time.time() - t0:.3f}s")

        # ── B8: Output ────────────────────────────────────────────────
        trans_np = refined_trans.squeeze().cpu().numpy()
        rot_np = refined_rot.cpu().numpy()
        det = np.linalg.det(rot_np)

        print(f"\n  Translation: [{trans_np[0]:.4f}, {trans_np[1]:.4f}, {trans_np[2]:.4f}]")
        print(f"  det(R) = {det:.6f}")
        print(f"  Coarse XDF cost: {best_coarse_cost:.4f}")

        # Save camera_pose_fgpl_native.json
        pano_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = pano_output_dir / "camera_pose_fgpl_native.json"
        result = {
            "rotation": rot_np.tolist(),
            "translation": trans_np.tolist(),
            "xdf_cost_coarse": best_coarse_cost,
            "principal_3d": principal_3d.cpu().numpy().tolist(),
            "principal_2d": principal_2d.cpu().numpy().tolist(),
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved pose: {output_path}")

        # ── B9: Visualization ─────────────────────────────────────────
        print(f"\nB7: Generating side_by_side.png")
        pano_img = cv2.imread(str(pano_img_path))
        if pano_img is not None:
            pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)
            starts_np = starts.cpu().numpy()
            ends_np = ends.cpu().numpy()

            panel = render_side_by_side(
                pano_img, edge_lines, starts_np, ends_np, rot_np, trans_np,
            )

            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_path = vis_dir / "side_by_side.png"
            cv2.imwrite(str(vis_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {vis_path}")
        else:
            print(f"  WARNING: Could not load panorama from {pano_img_path}")

        print()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for pano_name in PANO_NAMES:
        pose_path = OUTPUT_BASE / pano_name / "camera_pose_fgpl_native.json"
        vis_path = OUTPUT_BASE / pano_name / "vis" / "side_by_side.png"
        print(f"  {pano_name}:")
        print(f"    pose: {'OK' if pose_path.exists() else 'MISSING'}")
        print(f"    vis:  {'OK' if vis_path.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
