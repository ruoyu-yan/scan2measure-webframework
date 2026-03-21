"""Multi-room pose estimation orchestrator.

Estimates camera poses for multiple panoramas against a shared multi-room
3D wireframe map. The expensive 3D precomputation (LDF-3D, PDF-3D) runs
once; only 2D features + matching run per panorama.

Zero imports from panoramic-localization/ — uses only own modules.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_ROOT / "utils"))
sys.path.insert(0, str(_SRC_ROOT / "visualization"))
sys.path.insert(0, str(_SRC_ROOT / "features_2d"))
from sphere_geometry import generate_sphere_points
from xdf_distance import find_intersections_2d_indexed
from pose_search import (
    build_rotation_candidates,
    generate_translation_grid,
    rearrange_intersections_for_rotations,
    precompute_xdf_3d,
    xdf_coarse_search_from_precomputed,
)
from pose_refine import refine_pose
from visualize_pose import render_side_by_side
from line_analysis import classify_lines

# ── Config ──────────────────────────────────────────────────────────────────
POINT_CLOUD_NAME = "tmb_office_one_corridor_dense"
PANO_NAMES = ["TMB_office1", "TMB_corridor_south1", "TMB_corridor_south2"]
DEVICE = "cpu"

# Algorithm constants
QUERY_SPHERE_LEVEL = 3
XDF_INLIER_THRES = 0.1
POINT_GAMMA = 0.2
NUM_TRANS = 1700
CHAMFER_MIN_DIST = 0.3
TOP_K = 10
INLIER_THRES_2D = 0.05
INLIER_THRES_3D = 0.05
INTERSECT_THRES_2D = 0.1
CLASSIFY_THRES_2D = 0.5

# Paths
ROOT = Path(__file__).resolve().parent.parent
PKL_3D_PATH = ROOT / "data" / "debug_renderer" / POINT_CLOUD_NAME / "3d_line_map.pkl"
OUTPUT_BASE = ROOT / "data" / "pose_estimates" / "multiroom"


def main():
    device = torch.device(DEVICE)
    t_start = time.time()

    # ================================================================
    # PHASE A: One-time 3D setup
    # ================================================================
    print("=" * 60)
    print("PHASE A: One-time 3D precomputation")
    print("=" * 60)

    # ── A1: Load 3D map ───────────────────────────────────────────────
    print("\n[A1] Loading 3D wireframe from 3d_line_map.pkl...")
    with open(PKL_3D_PATH, 'rb') as f:
        line_map = pickle.load(f)

    # Dense lines for XDF matching
    dense_starts = line_map['dense_starts'].to(device).float()
    dense_ends = line_map['dense_ends'].to(device).float()
    dense_dirs = line_map['dense_dirs'].to(device).float()
    principal_3d = line_map['principal_3d'].to(device).float()

    # Pre-computed 3D intersections
    inter_3d_all = line_map['inter_3d'].to(device).float()
    inter_3d_idx_all = line_map['inter_3d_idx'].to(device).long()
    inter_3d_mask_all = line_map['inter_3d_mask'].to(device).bool()

    # Split 3D intersections into 3 groups (for camera_pose.json output)
    inter_3d_list = []
    for k in range(3):
        group_mask = inter_3d_mask_all[:, k] & inter_3d_mask_all[:, (k + 1) % 3]
        inter_3d_list.append(inter_3d_all[group_mask])

    # Sparse lines for translation grid
    starts_sparse = line_map['starts'].to(device).float()
    ends_sparse = line_map['ends'].to(device).float()

    print(f"    dense segments: {dense_starts.shape[0]}, "
          f"sparse segments: {starts_sparse.shape[0]}")
    print(f"    3D intersections: {inter_3d_all.shape[0]}")
    print(f"    principal_3d det: {torch.linalg.det(principal_3d):.4f}")

    # ── A2: Generate translation grid ─────────────────────────────────
    print("\n[A2] Generating translation grid (quantile-based)...")
    trans_candidates = generate_translation_grid(
        starts_sparse, ends_sparse,
        num_trans=NUM_TRANS, chamfer_min_dist=CHAMFER_MIN_DIST)
    print(f"    {trans_candidates.shape[0]} translation candidates")

    # ── A3: Generate query points ─────────────────────────────────────
    print("\n[A3] Generating icosphere query points...")
    query_pts = generate_sphere_points(QUERY_SPHERE_LEVEL, device=device)
    print(f"    {query_pts.shape[0]} query points (level {QUERY_SPHERE_LEVEL})")

    # ── A4: Precompute 3D distance functions ──────────────────────────
    print("\n[A4] Precomputing 3D distance functions (LDF-3D + PDF-3D)...")
    t0 = time.time()
    precomputed_3d = precompute_xdf_3d(
        principal_3d, dense_starts, dense_ends, dense_dirs,
        inter_3d_all, inter_3d_mask_all,
        trans_candidates, query_pts,
        inlier_thres_3d=INLIER_THRES_3D)
    dt_3d = time.time() - t0
    print(f"    LDF-3D: {precomputed_3d['ldf_3d'].shape}")
    print(f"    PDF-3D: {precomputed_3d['pdf_3d'].shape}")
    print(f"    Phase A total: {dt_3d:.1f}s")

    # ================================================================
    # PHASE B: Per-panorama pose estimation
    # ================================================================
    for pano_idx, pano_name in enumerate(PANO_NAMES):
        print("\n" + "=" * 60)
        print(f"PHASE B [{pano_idx+1}/{len(PANO_NAMES)}]: {pano_name}")
        print("=" * 60)

        t_pano = time.time()

        features_2d_path = (
            ROOT / "data" / "pano" / "2d_feature_extracted"
            / f"{pano_name}_v2" / "fgpl_features.json"
        )
        pano_img_path = ROOT / "data" / "pano" / "raw" / f"{pano_name}.jpg"
        pano_output_dir = OUTPUT_BASE / pano_name
        vis_dir = pano_output_dir / "vis"

        # ── B1: Load 2D features ──────────────────────────────────────
        print(f"\n[B1] Loading 2D features from {features_2d_path.name}...")
        with open(features_2d_path) as f:
            feat_2d = json.load(f)

        edge_2d = torch.tensor(feat_2d['lines'], dtype=torch.float32, device=device)
        principal_2d = torch.tensor(feat_2d['principal_2d'], dtype=torch.float32, device=device)
        print(f"    edge_2d: {edge_2d.shape}, "
              f"principal_2d det: {torch.linalg.det(principal_2d):.4f}")

        # ── B2: Classification stats ──────────────────────────────────
        mask = classify_lines(edge_2d, principal_2d, inlier_thres=INLIER_THRES_2D)
        n_classified = mask.any(dim=1).sum().item()
        print(f"    Classified: {n_classified}/{edge_2d.shape[0]} "
              f"({100*n_classified/edge_2d.shape[0]:.1f}%)")

        # ── B3: 2D intersections ──────────────────────────────────────
        print("\n[B3] Computing 2D intersections with line-pair indices...")
        inter_2d_list_2d, inter_2d_idx_list = find_intersections_2d_indexed(
            edge_2d, principal_2d,
            inlier_thres=CLASSIFY_THRES_2D, intersect_thres=INTERSECT_THRES_2D)
        for k in range(3):
            print(f"    inter_2d[{k}]: {inter_2d_list_2d[k].shape[0]}")

        # ── B4: 24 rotation candidates ────────────────────────────────
        print("\n[B4] Building 24 rotation candidates...")
        rotations, perms_expanded = build_rotation_candidates(principal_2d, principal_3d)
        print(f"    rotations: {rotations.shape}")

        # ── B5: Rearrange intersections ───────────────────────────────
        print("\n[B5] Rearranging 2D intersections for rotation permutations...")
        inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot = \
            rearrange_intersections_for_rotations(
                inter_2d_list_2d, inter_2d_idx_list, perms_expanded)

        # ── B6: XDF coarse search (from precomputed 3D) ──────────────
        print("\n[B6] XDF coarse search (2D only — 3D cached)...")
        t0 = time.time()
        top_poses, _cost_matrix = xdf_coarse_search_from_precomputed(
            precomputed_3d,
            rotations, perms_expanded, edge_2d,
            inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot,
            trans_candidates, query_pts,
            top_k=TOP_K, xdf_inlier_thres=XDF_INLIER_THRES,
            point_gamma=POINT_GAMMA,
            inlier_thres_2d=INLIER_THRES_2D)

        for i, p in enumerate(top_poses):
            print(f"    top-{i}: cost={p['cost']:.1f}  rot_idx={p['rot_idx']}  "
                  f"t={p['t'].numpy()}")
        print(f"    XDF search: {time.time() - t0:.1f}s")

        # ── B7: ICP refinement on top-K ───────────────────────────────
        print("\n[B7] ICP refinement on top-K candidates...")
        candidates = []

        i3d = inter_3d_all.numpy()
        i3d_mask = inter_3d_mask_all.numpy()
        i3d_idx = inter_3d_idx_all.numpy()

        for i, pose in enumerate(top_poses):
            R_init = pose['R'].numpy()
            t_init = pose['t'].numpy()
            i2d = pose['inter_2d'].numpy()
            i2d_mask = pose['inter_2d_mask'].numpy()
            i2d_idx = pose['inter_2d_idx'].numpy()

            R_ref, t_ref, matched = refine_pose(
                R_init, t_init, i2d, i3d, i2d_mask, i3d_mask,
                line_normals_2d=edge_2d[:, :3].numpy(),
                line_dirs_3d=dense_dirs.numpy(),
                inter_2d_idx=i2d_idx, inter_3d_idx=i3d_idx,
                n_iters_t=100, n_iters_r=50, lr=0.1)

            n_matched = sum(len(mp) for mp in matched)

            # Compute match quality: count tight inliers (sphere dist < 0.1)
            n_tight = 0
            avg_dist = 0.0
            if n_matched > 0:
                total_dist = 0.0
                for k in range(3):
                    if len(matched[k]) > 0:
                        pairs = matched[k]
                        pts_2d_k = i2d[pairs[:, 0]]
                        pts_3d_w = i3d[pairs[:, 1]]
                        pts_3d_cam = (pts_3d_w - t_ref[np.newaxis, :]) @ R_ref.T
                        pts_3d_sph = pts_3d_cam / np.maximum(
                            np.linalg.norm(pts_3d_cam, axis=-1, keepdims=True), 1e-6)
                        cos_sim = np.clip((pts_2d_k * pts_3d_sph).sum(-1), -1, 1)
                        dists = np.arccos(cos_sim)
                        total_dist += dists.sum()
                        n_tight += (dists < 0.1).sum()
                avg_dist = total_dist / n_matched

            print(f"    candidate {i}: n_matched={n_matched}  n_tight={n_tight}  "
                  f"avg_dist={avg_dist:.4f}  t={t_ref}")

            candidates.append({
                'R': R_ref, 't': t_ref, 'matched': matched,
                'inter_2d': i2d, 'inter_2d_mask': i2d_mask,
                'cost': pose['cost'],
                'n_matched': n_matched,
                'n_tight': int(n_tight),
                'avg_dist': avg_dist,
            })

        # Select best: highest tight inlier count, then lowest avg_dist
        candidates.sort(key=lambda c: (-c['n_tight'], c['avg_dist']))
        best = candidates[0]
        final_R = best['R']
        final_t = best['t']
        matched_pairs = best['matched']

        print(f"\n    BEST: n_matched={best['n_matched']}  "
              f"avg_dist={best['avg_dist']:.4f}  t={final_t}")
        print(f"    det(R) = {np.linalg.det(final_R):.6f}")

        # ── B8: Save camera_pose.json ─────────────────────────────────
        print("\n[B8] Saving camera_pose.json...")
        pano_output_dir.mkdir(parents=True, exist_ok=True)

        # Split 2D intersections by group from mask
        i2d_all = best['inter_2d']
        i2d_mask_best = best['inter_2d_mask']
        inter_2d_groups = [[], [], []]
        for idx_pt in range(i2d_all.shape[0]):
            for k in range(3):
                if i2d_mask_best[idx_pt, k] and i2d_mask_best[idx_pt, (k + 1) % 3]:
                    inter_2d_groups[k].append(i2d_all[idx_pt].tolist())
                    break

        inter_3d_groups = [g.numpy().tolist() for g in inter_3d_list]

        result = {
            "rotation":        final_R.tolist(),
            "translation":     final_t.tolist(),
            "principal_3d":    principal_3d.cpu().numpy().tolist(),
            "principal_2d":    principal_2d.cpu().numpy().tolist(),
            "n_inter_matched": [len(mp) for mp in matched_pairs],
            "xdf_cost_coarse": float(best['cost']),
            "inter_2d":        inter_2d_groups,
            "inter_3d":        inter_3d_groups,
            "matched_pairs":   [mp.tolist() for mp in matched_pairs],
        }

        out_path = pano_output_dir / "camera_pose.json"
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"    Saved to {out_path}")

        # ── B9: Visualization ─────────────────────────────────────────
        print("\n[B9] Generating side_by_side.png...")
        pano_img = cv2.imread(str(pano_img_path))
        if pano_img is not None:
            pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)

            starts_np = starts_sparse.cpu().numpy()
            ends_np = ends_sparse.cpu().numpy()

            sbs = render_side_by_side(
                pano_img, edge_2d, starts_np, ends_np,
                final_R, final_t, resolution=(512, 1024))

            vis_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_dir / "side_by_side.png"),
                        cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))
            print(f"    Saved {vis_dir / 'side_by_side.png'}")
        else:
            print(f"    WARNING: could not load panorama from {pano_img_path}")

        dt_pano = time.time() - t_pano
        print(f"\n    Pano {pano_name} total: {dt_pano:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────
    dt_total = time.time() - t_start
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  3D precompute: {dt_3d:.1f}s")
    for pano_name in PANO_NAMES:
        pose_path = OUTPUT_BASE / pano_name / "camera_pose.json"
        vis_path = OUTPUT_BASE / pano_name / "vis" / "side_by_side.png"
        print(f"  {pano_name}:")
        print(f"    pose: {'OK' if pose_path.exists() else 'MISSING'}")
        print(f"    vis:  {'OK' if vis_path.exists() else 'MISSING'}")
    print(f"  Total: {dt_total:.1f}s")


if __name__ == "__main__":
    main()
