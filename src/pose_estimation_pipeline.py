"""FGPL Pose Estimation Pipeline — orchestrator.

Loads pre-computed 3D wireframe features and 2D sphere-based line features,
runs XDF coarse search + sphere ICP refinement, saves camera_pose.json (V2
format) and side-by-side visualization PNG.

Pattern follows image_feature_extractionV2.py: constants at top, single main().
No imports from panoramic-localization/.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sphere_geometry import generate_sphere_points
from xdf_distance import find_intersections_2d_indexed
from pose_search import (
    build_rotation_candidates,
    generate_translation_grid,
    rearrange_intersections_for_rotations,
    xdf_coarse_search,
)
from pose_refine import refine_pose
from visualize_pose import render_side_by_side, render_reprojection, render_topdown

# ── Config ──────────────────────────────────────────────────────────────────
POINT_CLOUD_NAME = "tmb_office1"
ROOM_NAME = "TMB_office1"
DEVICE = "cpu"  # Change to "cuda:0" if CUDA works

# Algorithm constants
QUERY_SPHERE_LEVEL = 3
VOTE_SPHERE_LEVEL = 5
XDF_INLIER_THRES = 0.1
POINT_GAMMA = 0.2
NUM_TRANS = 1700
CHAMFER_MIN_DIST = 0.3
TOP_K = 10
INLIER_THRES_2D = 0.05
INLIER_THRES_3D = 0.05
INTERSECT_THRES_2D = 0.1
CLASSIFY_THRES_2D = 0.5  # line classification threshold

# Paths
ROOT = Path(__file__).resolve().parent.parent
PKL_3D_PATH = ROOT / "data" / "debug_renderer" / POINT_CLOUD_NAME / "3d_line_map.pkl"
FEATURES_2D_PATH = ROOT / "data" / "pano" / "2d_feature_extracted" / f"{ROOM_NAME}_v2" / "fgpl_features.json"
PANO_PATH = ROOT / "data" / "pano" / "raw" / f"{ROOM_NAME}.jpg"
OUTPUT_DIR = ROOT / "data" / "pose_estimates" / ROOM_NAME
POINT_CLOUD_PATH = ROOT / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
DENSITY_IMG_PATH = ROOT / "data" / "density_image" / POINT_CLOUD_NAME / f"{POINT_CLOUD_NAME}.png"
DENSITY_META_PATH = ROOT / "data" / "density_image" / POINT_CLOUD_NAME / "metadata.json"
ROOMFORMER_PATH = ROOT / "data" / "reconstructed_floorplans_RoomFormer" / POINT_CLOUD_NAME / "predictions.json"


def main():
    device = torch.device(DEVICE)
    t0 = time.time()

    # ============================================================
    # [1] Load 3D features
    # ============================================================
    print("[1] Loading 3D wireframe from 3d_line_map.pkl...")
    with open(PKL_3D_PATH, 'rb') as f:
        line_map = pickle.load(f)

    # Use dense lines for intersections and matching
    starts = line_map['dense_starts'].to(device).float()
    ends = line_map['dense_ends'].to(device).float()
    dirs = line_map['dense_dirs'].to(device).float()
    principal_3d = line_map['principal_3d'].to(device).float()

    # Pre-computed 3D intersections
    inter_3d_all = line_map['inter_3d'].to(device).float()
    inter_3d_idx_all = line_map['inter_3d_idx'].to(device).long()
    inter_3d_mask_all = line_map['inter_3d_mask'].to(device).bool()

    # Split 3D intersections into 3 groups from mask
    inter_3d_list = []
    inter_3d_idx_list = []
    for k in range(3):
        group_mask = inter_3d_mask_all[:, k] & inter_3d_mask_all[:, (k + 1) % 3]
        inter_3d_list.append(inter_3d_all[group_mask])
        inter_3d_idx_list.append(inter_3d_idx_all[group_mask])

    # Sparse lines for translation grid (matching native FGPL)
    starts_sparse = line_map['starts'].to(device).float()
    ends_sparse = line_map['ends'].to(device).float()

    print(f"    dense segments: {starts.shape[0]}, "
          f"3D intersections: {[g.shape[0] for g in inter_3d_list]}")
    print(f"    principal_3d det: {torch.linalg.det(principal_3d):.4f}")
    dt1 = time.time() - t0

    # ============================================================
    # [2] Load 2D features
    # ============================================================
    print("[2] Loading 2D features from fgpl_features.json...")
    with open(FEATURES_2D_PATH) as f:
        feat_2d = json.load(f)

    edge_2d = torch.tensor(feat_2d['lines'], dtype=torch.float32, device=device)
    principal_2d = torch.tensor(feat_2d['principal_2d'], dtype=torch.float32, device=device)
    print(f"    edge_2d: {edge_2d.shape}, principal_2d det: {torch.linalg.det(principal_2d):.4f}")
    dt2 = time.time() - t0

    # ============================================================
    # [3] Compute 2D intersections WITH indices
    # ============================================================
    print("[3] Computing 2D intersections with line-pair indices...")
    inter_2d_list, inter_2d_idx_list = find_intersections_2d_indexed(
        edge_2d, principal_2d,
        inlier_thres=CLASSIFY_THRES_2D, intersect_thres=INTERSECT_THRES_2D)
    for k in range(3):
        print(f"    inter_2d[{k}]: {inter_2d_list[k].shape[0]}")
    dt3 = time.time() - t0

    # ============================================================
    # [4] Build 24 rotation candidates
    # ============================================================
    print("[4] Building 24 rotation candidates...")
    rotations, perms_expanded = build_rotation_candidates(principal_2d, principal_3d)
    print(f"    rotations: {rotations.shape}")
    dt4 = time.time() - t0

    # ============================================================
    # [5] Rearrange 2D intersections for 24 rotations
    # ============================================================
    print("[5] Rearranging 2D intersections for rotation permutations...")
    inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot = \
        rearrange_intersections_for_rotations(
            inter_2d_list, inter_2d_idx_list, perms_expanded)
    dt5 = time.time() - t0

    # ============================================================
    # [6] Generate translation grid
    # ============================================================
    print("[6] Generating translation grid (quantile-based)...")
    # Use sparse line midpoints as reference (matching native FGPL)
    trans_candidates = generate_translation_grid(
        starts_sparse, ends_sparse,
        num_trans=NUM_TRANS, chamfer_min_dist=CHAMFER_MIN_DIST)
    print(f"    {trans_candidates.shape[0]} translation candidates")
    dt6 = time.time() - t0

    # ============================================================
    # [7] XDF coarse search
    # ============================================================
    print("[7] XDF coarse search (canonical precomputation)...")
    query_pts = generate_sphere_points(QUERY_SPHERE_LEVEL, device=device)
    top_poses, _cost_matrix = xdf_coarse_search(
        rotations, perms_expanded, principal_3d,
        edge_2d, starts, ends, dirs,
        inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot,
        inter_3d_all, inter_3d_mask_all,
        trans_candidates, query_pts,
        top_k=TOP_K, xdf_inlier_thres=XDF_INLIER_THRES,
        point_gamma=POINT_GAMMA,
        inlier_thres_2d=INLIER_THRES_2D, inlier_thres_3d=INLIER_THRES_3D)

    for i, p in enumerate(top_poses):
        print(f"    top-{i}: cost={p['cost']:.1f}  rot_idx={p['rot_idx']}  t={p['t'].numpy()}")
    dt7 = time.time() - t0

    # ============================================================
    # [8] ICP refinement on top-K, pick best
    # ============================================================
    print("[8] ICP refinement on top-K candidates...")
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
            line_dirs_3d=dirs.numpy(),
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
              f"avg_dist={avg_dist:.4f}  t={t_ref}  rot_idx={pose['rot_idx']}")

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
    best_result = candidates[0]
    final_R = best_result['R']
    final_t = best_result['t']
    matched_pairs = best_result['matched']
    best_n_matched = best_result['n_matched']
    dt8 = time.time() - t0

    print(f"    BEST: n_matched={best_n_matched}  avg_dist={best_result['avg_dist']:.4f}  t={final_t}")
    print(f"    det(R) = {np.linalg.det(final_R):.6f}")

    # ============================================================
    # [9] Save camera_pose.json
    # ============================================================
    print("[9] Saving camera_pose.json...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Split 2D intersections by group from mask
    i2d_all = best_result['inter_2d']
    i2d_mask = best_result['inter_2d_mask']
    inter_2d_groups = [[], [], []]
    for idx_pt in range(i2d_all.shape[0]):
        for k in range(3):
            if i2d_mask[idx_pt, k] and i2d_mask[idx_pt, (k + 1) % 3]:
                inter_2d_groups[k].append(i2d_all[idx_pt].tolist())
                break

    inter_3d_groups = [g.numpy().tolist() for g in inter_3d_list]

    result = {
        "rotation":        final_R.tolist(),
        "translation":     final_t.tolist(),
        "principal_3d":    principal_3d.cpu().numpy().tolist(),
        "principal_2d":    principal_2d.cpu().numpy().tolist(),
        "n_inter_matched": [len(mp) for mp in matched_pairs],
        "xdf_cost_coarse": float(best_result['cost']),
        "inter_2d":        inter_2d_groups,
        "inter_3d":        inter_3d_groups,
        "matched_pairs":   [mp.tolist() for mp in matched_pairs],
    }

    out_path = OUTPUT_DIR / "camera_pose.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"    Saved to {out_path}")
    dt9 = time.time() - t0

    # ============================================================
    # [10] Visualizations
    # ============================================================
    print("[10] Generating visualizations...")
    vis_dir = OUTPUT_DIR / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    pano_img = cv2.imread(str(PANO_PATH))
    if pano_img is not None:
        pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)

        # Side-by-side: our pipeline pose
        sbs = render_side_by_side(
            pano_img, edge_2d, starts.cpu().numpy(), ends.cpu().numpy(),
            final_R, final_t, resolution=(512, 1024))
        cv2.imwrite(str(vis_dir / "side_by_side.png"),
                    cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))
        print(f"    Saved side_by_side.png")

        # Reprojection: depth-coded point cloud on panorama
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(POINT_CLOUD_PATH))
        pts_viz = np.asarray(pcd.points)
        rng = np.random.default_rng(42)
        if pts_viz.shape[0] > 50000:
            pts_viz = pts_viz[rng.choice(pts_viz.shape[0], 50000, replace=False)]
        reproj = render_reprojection(pano_img, pts_viz, final_R, final_t)
        cv2.imwrite(str(vis_dir / "reprojection.png"),
                    cv2.cvtColor(reproj, cv2.COLOR_RGB2BGR))
        print(f"    Saved reprojection.png")

    else:
        print(f"    WARNING: could not load panorama from {PANO_PATH}")

    # Top-down: camera on density image (doesn't need panorama)
    density_img_raw = cv2.imread(str(DENSITY_IMG_PATH), cv2.IMREAD_GRAYSCALE)
    if density_img_raw is not None:
        with open(DENSITY_META_PATH) as f:
            density_meta = json.load(f)
        room_polys = None
        if ROOMFORMER_PATH.exists():
            with open(ROOMFORMER_PATH) as f:
                room_polys = json.load(f)
        topdown = render_topdown(density_img_raw, density_meta, final_t, final_R,
                                 room_polygons=room_polys)
        cv2.imwrite(str(vis_dir / "topdown.png"),
                    cv2.cvtColor(topdown, cv2.COLOR_RGB2BGR))
        print(f"    Saved topdown.png")
    else:
        print(f"    WARNING: could not load density image from {DENSITY_IMG_PATH}")

    dt10 = time.time() - t0

    print("\n--- Results ---")
    print(f"det(R) = {np.linalg.det(final_R):.6f}")
    print(f"translation = ({final_t[0]:.3f}, {final_t[1]:.3f}, {final_t[2]:.3f})")
    print(f"n_inter_matched = {[len(mp) for mp in matched_pairs]}")

    # Timing
    print(f"\n--- Timing ---")
    print(f"  Load 3D:        {dt1:.1f}s")
    print(f"  Load 2D:        {dt2 - dt1:.1f}s")
    print(f"  2D intersect:   {dt3 - dt2:.1f}s")
    print(f"  Rot candidates: {dt4 - dt3:.1f}s")
    print(f"  Rearrange:      {dt5 - dt4:.1f}s")
    print(f"  Trans grid:     {dt6 - dt5:.1f}s")
    print(f"  XDF search:     {dt7 - dt6:.1f}s")
    print(f"  ICP refine:     {dt8 - dt7:.1f}s")
    print(f"  Save JSON:      {dt9 - dt8:.1f}s")
    print(f"  Visualize:      {dt10 - dt9:.1f}s")
    print(f"  Total:          {dt10:.1f}s")


if __name__ == "__main__":
    main()
