"""Multi-room pose estimation orchestrator.

Estimates camera poses for multiple panoramas against a shared multi-room
3D wireframe map. Supports two modes:

  USE_LOCAL_FILTERING = False (original):
    3D precomputation (LDF-3D, PDF-3D) runs once globally; only 2D features
    + matching run per panorama.

  USE_LOCAL_FILTERING = True (Voronoi-based):
    Each panorama sees only local 3D geometry (nearest-panorama Voronoi
    assignment + overlap margin). 3D precomputation and translation grid
    are generated per-panorama from filtered lines. Eliminates cross-room
    false minima.

Zero imports from panoramic-localization/ -- uses only own modules.
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
from config_loader import load_config, progress
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
from visualize_pose import render_side_by_side, render_reprojection, render_topdown
from line_analysis import classify_lines

# -- Config ------------------------------------------------------------------
POINT_CLOUD_NAME = "tmb_office_one_corridor_bigger_noRGB"
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

# Local line filtering
USE_LOCAL_FILTERING = True
OVERLAP_MARGIN = 2.0  # meters, buffer beyond Voronoi boundary

# Paths
ROOT = _SRC_ROOT.parent
PKL_3D_PATH = ROOT / "data" / "debug_renderer" / POINT_CLOUD_NAME / "3d_line_map.pkl"
ALIGNMENT_PATH = ROOT / "data" / "sam3_room_segmentation" / POINT_CLOUD_NAME / "demo6_alignment.json"
METADATA_PATH = ROOT / "data" / "density_image" / POINT_CLOUD_NAME / "metadata.json"
PC_PATH = ROOT / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
DENSITY_IMG_PATH = ROOT / "data" / "density_image" / POINT_CLOUD_NAME / f"{POINT_CLOUD_NAME}.png"
ROOMFORMER_PATH = ROOT / "data" / "reconstructed_floorplans_RoomFormer" / POINT_CLOUD_NAME / "predictions.json"
OUTPUT_BASE = ROOT / "data" / "pose_estimates" / "multiroom"


# -- Local filtering helpers -------------------------------------------------

def pixel_to_raw_3d(pixel_coord, metadata):
    """Convert single pixel coordinate to raw 3D world XY (meters)."""
    R = np.array(metadata['rotation_matrix'])
    min_coords = np.array(metadata['min_coords'])
    offset = np.array(metadata['offset'])
    max_dim = metadata['max_dim']
    scale = metadata['image_width'] - 1
    aligned_mm = np.array(pixel_coord) / scale * max_dim - offset[:2] + min_coords[:2]
    raw_3d = R.T @ np.array([aligned_mm[0], aligned_mm[1], 0.0]) / 1000.0
    return raw_3d[:2]


def aligned_meters_to_raw_3d(xy_aligned, metadata):
    """Convert XY in Manhattan-aligned meters to raw 3D world XY (meters).

    SAM3_mask_to_polygons outputs in aligned frame (no R.T applied).
    The 3D line map is in raw frame. This applies the inverse rotation.
    """
    R = np.array(metadata['rotation_matrix'])
    pos_3d = R.T @ np.array([xy_aligned[0], xy_aligned[1], 0.0])
    return pos_3d[:2]


def load_panorama_positions(alignment_path, metadata_path, pano_name_list):
    """Load panorama 3D XY positions from demo6_alignment.json.

    Returns dict: pano_name -> np.array([x, y]) in raw 3D meters.
    """
    with open(alignment_path) as f:
        alignment = json.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)

    positions = {}
    for match in alignment['matches']:
        name = match['pano_name']
        if name in pano_name_list:
            cam_aligned = match['camera_position']
            xy = aligned_meters_to_raw_3d(cam_aligned, metadata)
            positions[name] = xy
            print(f"    {name}: room={match['room_label']}  "
                  f"aligned=({cam_aligned[0]:.2f}, {cam_aligned[1]:.2f})  "
                  f"raw_3d=({xy[0]:.2f}, {xy[1]:.2f})")
    return positions


def compute_voronoi_assignment(points_xy, pano_positions, pano_name_list):
    """Assign each point to nearest panorama. Returns (N,) int array of pano indices."""
    pano_xys = np.array([pano_positions[name] for name in pano_name_list])  # (P, 2)
    dists = np.linalg.norm(points_xy[:, None, :] - pano_xys[None, :, :], axis=2)  # (N, P)
    return np.argmin(dists, axis=1)


def get_local_mask(points_xy, pano_idx, pano_positions, voronoi_labels, margin, pano_name_list):
    """Get boolean mask for points belonging to this panorama (Voronoi + margin).

    Includes: Voronoi-assigned points + any point within margin of this panorama.
    """
    voronoi_mask = voronoi_labels == pano_idx
    pano_xy = pano_positions[pano_name_list[pano_idx]]
    dist_to_pano = np.linalg.norm(points_xy - pano_xy[None, :], axis=1)
    margin_mask = dist_to_pano <= margin
    return voronoi_mask | margin_mask


# -- Main --------------------------------------------------------------------

def main():
    cfg = load_config()

    pc_name = cfg.get("point_cloud_name", POINT_CLOUD_NAME)
    pano_names = cfg.get("pano_names", PANO_NAMES)
    # Important: Electron app MUST pass "use_local_filtering": true
    use_local = cfg.get("use_local_filtering", USE_LOCAL_FILTERING)

    pkl_3d_path = Path(cfg["pkl_3d_path"]) if cfg.get("pkl_3d_path") else ROOT / "data" / "debug_renderer" / pc_name / "3d_line_map.pkl"
    alignment_path = Path(cfg["alignment_path"]) if cfg.get("alignment_path") else ROOT / "data" / "sam3_room_segmentation" / pc_name / "demo6_alignment.json"
    metadata_path = Path(cfg["metadata_path"]) if cfg.get("metadata_path") else ROOT / "data" / "density_image" / pc_name / "metadata.json"
    pc_path = Path(cfg["point_cloud_path"]) if cfg.get("point_cloud_path") else ROOT / "data" / "raw_point_cloud" / f"{pc_name}.ply"
    density_img_path = Path(cfg["density_image_path"]) if cfg.get("density_image_path") else ROOT / "data" / "density_image" / pc_name / f"{pc_name}.png"
    features_2d_base = Path(cfg["features_2d_dir"]) if cfg.get("features_2d_dir") else ROOT / "data" / "pano" / "2d_feature_extracted"
    pano_dir = Path(cfg["pano_dir"]) if cfg.get("pano_dir") else ROOT / "data" / "pano" / "raw"
    output_base = Path(cfg["output_dir"]) if cfg.get("output_dir") else OUTPUT_BASE

    device = torch.device(DEVICE)
    t_start = time.time()
    total_steps = 1 + len(pano_names)

    # ================================================================
    # PHASE A: One-time 3D setup
    # ================================================================
    progress(1, total_steps, "3D setup and precomputation")
    print("=" * 60)
    print("PHASE A: One-time 3D setup")
    print("=" * 60)
    if use_local:
        print("  Mode: LOCAL FILTERING (Voronoi + margin)")
    else:
        print("  Mode: GLOBAL (original)")

    # -- A1: Load 3D map ---------------------------------------------------
    print("\n[A1] Loading 3D wireframe from 3d_line_map.pkl...")
    with open(pkl_3d_path, 'rb') as f:
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

    # Keep full dense_dirs for refine_pose (inter_3d_idx indexes into original array)
    dense_dirs_full_np = dense_dirs.numpy()

    print(f"    dense segments: {dense_starts.shape[0]}, "
          f"sparse segments: {starts_sparse.shape[0]}")
    print(f"    3D intersections: {inter_3d_all.shape[0]}")
    print(f"    principal_3d det: {torch.linalg.det(principal_3d):.4f}")

    # -- A2: Generate query points -----------------------------------------
    print("\n[A2] Generating icosphere query points...")
    query_pts = generate_sphere_points(QUERY_SPHERE_LEVEL, device=device)
    print(f"    {query_pts.shape[0]} query points (level {QUERY_SPHERE_LEVEL})")

    # -- A3: Local filtering setup OR global precomputation ----------------
    if use_local:
        print("\n[A3] Loading panorama positions for Voronoi filtering...")
        pano_positions = load_panorama_positions(alignment_path, metadata_path, pano_names)

        print("\n[A4] Computing Voronoi assignments...")
        dense_mid_xy = ((dense_starts + dense_ends) / 2)[:, :2].numpy()
        sparse_mid_xy = ((starts_sparse + ends_sparse) / 2)[:, :2].numpy()
        inter_xy = inter_3d_all[:, :2].numpy()

        voronoi_dense = compute_voronoi_assignment(dense_mid_xy, pano_positions, pano_names)
        voronoi_sparse = compute_voronoi_assignment(sparse_mid_xy, pano_positions, pano_names)
        voronoi_inter = compute_voronoi_assignment(inter_xy, pano_positions, pano_names)

        for pi, name in enumerate(pano_names):
            nd = (voronoi_dense == pi).sum()
            ns = (voronoi_sparse == pi).sum()
            ni = (voronoi_inter == pi).sum()
            print(f"    {name}: dense={nd}, sparse={ns}, inter={ni}")

        # Load visualization data
        print("\n[A5] Loading visualization data (point cloud, density image, polygons)...")
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(pc_path))
        pc_pts = np.asarray(pcd.points)
        rng = np.random.default_rng(42)
        if pc_pts.shape[0] > 50000:
            pc_pts = pc_pts[rng.choice(pc_pts.shape[0], 50000, replace=False)]
        print(f"    Point cloud: {pc_pts.shape[0]} points (subsampled)")

        density_img = cv2.imread(str(density_img_path), cv2.IMREAD_GRAYSCALE)
        with open(metadata_path) as f:
            density_meta = json.load(f)

        room_polygons = None
        roomformer_path = ROOT / "data" / "reconstructed_floorplans_RoomFormer" / pc_name / "predictions.json"
        if roomformer_path.exists():
            with open(roomformer_path) as f:
                room_polygons = json.load(f)
        print(f"    Density image: {density_img.shape}")
        print(f"    RoomFormer polygons: {len(room_polygons) if room_polygons else 0}")

        # Shared variables for global mode (not used)
        precomputed_3d = None
        trans_candidates = None
        dt_3d = 0.0
    else:
        # Global mode: precompute once for all panoramas
        print("\n[A3] Generating translation grid (quantile-based)...")
        trans_candidates = generate_translation_grid(
            starts_sparse, ends_sparse,
            num_trans=NUM_TRANS, chamfer_min_dist=CHAMFER_MIN_DIST)
        print(f"    {trans_candidates.shape[0]} translation candidates")

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
    all_results = {}

    for pano_idx, pano_name in enumerate(pano_names):
        progress(2 + pano_idx, total_steps, f"Pose estimation: {pano_name}")
        print("\n" + "=" * 60)
        print(f"PHASE B [{pano_idx+1}/{len(pano_names)}]: {pano_name}")
        print("=" * 60)

        t_pano = time.time()

        features_2d_path = features_2d_base / f"{pano_name}_v2" / "fgpl_features.json"
        pano_img_path = pano_dir / f"{pano_name}.jpg"
        pano_output_dir = output_base / pano_name
        vis_dir = pano_output_dir / "vis"

        # -- Per-panorama local filtering ----------------------------------
        if use_local:
            print(f"\n[B0] Filtering 3D lines for {pano_name}...")
            dense_mask = get_local_mask(dense_mid_xy, pano_idx, pano_positions,
                                        voronoi_dense, OVERLAP_MARGIN, pano_names)
            sparse_mask = get_local_mask(sparse_mid_xy, pano_idx, pano_positions,
                                         voronoi_sparse, OVERLAP_MARGIN, pano_names)
            inter_mask = get_local_mask(inter_xy, pano_idx, pano_positions,
                                        voronoi_inter, OVERLAP_MARGIN, pano_names)

            local_dense_starts = dense_starts[dense_mask]
            local_dense_ends = dense_ends[dense_mask]
            local_dense_dirs = dense_dirs[dense_mask]
            local_starts = starts_sparse[sparse_mask]
            local_ends = ends_sparse[sparse_mask]
            local_inter_3d = inter_3d_all[inter_mask]
            local_inter_3d_mask = inter_3d_mask_all[inter_mask]
            local_inter_3d_idx = inter_3d_idx_all[inter_mask]

            n_dense = local_dense_starts.shape[0]
            n_sparse = local_starts.shape[0]
            n_inter = local_inter_3d.shape[0]
            print(f"    Filtered: dense={n_dense}, sparse={n_sparse}, inter={n_inter}")

            print("\n[B0b] Generating translation grid from filtered sparse lines...")
            pano_trans_candidates = generate_translation_grid(
                local_starts, local_ends,
                num_trans=NUM_TRANS, chamfer_min_dist=CHAMFER_MIN_DIST)
            n_trans = pano_trans_candidates.shape[0]
            print(f"    {n_trans} translation candidates")

            print("\n[B0c] Precomputing 3D distance functions (LDF-3D + PDF-3D)...")
            t0 = time.time()
            pano_precomputed_3d = precompute_xdf_3d(
                principal_3d, local_dense_starts, local_dense_ends, local_dense_dirs,
                local_inter_3d, local_inter_3d_mask,
                pano_trans_candidates, query_pts,
                inlier_thres_3d=INLIER_THRES_3D)
            dt_precomp = time.time() - t0
            print(f"    LDF-3D: {pano_precomputed_3d['ldf_3d'].shape}")
            print(f"    PDF-3D: {pano_precomputed_3d['pdf_3d'].shape}")
            print(f"    Precompute: {dt_precomp:.1f}s")

            # Use per-panorama precomputed data for coarse search and refinement
            cur_precomputed_3d = pano_precomputed_3d
            cur_trans_candidates = pano_trans_candidates
            cur_inter_3d = local_inter_3d
            cur_inter_3d_mask = local_inter_3d_mask
            cur_inter_3d_idx = local_inter_3d_idx
        else:
            cur_precomputed_3d = precomputed_3d
            cur_trans_candidates = trans_candidates
            cur_inter_3d = inter_3d_all
            cur_inter_3d_mask = inter_3d_mask_all
            cur_inter_3d_idx = inter_3d_idx_all

        # -- B1: Load 2D features ------------------------------------------
        print(f"\n[B1] Loading 2D features from {features_2d_path.name}...")
        with open(features_2d_path) as f:
            feat_2d = json.load(f)

        edge_2d = torch.tensor(feat_2d['lines'], dtype=torch.float32, device=device)
        principal_2d = torch.tensor(feat_2d['principal_2d'], dtype=torch.float32, device=device)
        print(f"    edge_2d: {edge_2d.shape}, "
              f"principal_2d det: {torch.linalg.det(principal_2d):.4f}")

        # -- B2: Classification stats --------------------------------------
        mask = classify_lines(edge_2d, principal_2d, inlier_thres=INLIER_THRES_2D)
        n_classified = mask.any(dim=1).sum().item()
        print(f"    Classified: {n_classified}/{edge_2d.shape[0]} "
              f"({100*n_classified/edge_2d.shape[0]:.1f}%)")

        # -- B3: 2D intersections ------------------------------------------
        print("\n[B3] Computing 2D intersections with line-pair indices...")
        inter_2d_list_2d, inter_2d_idx_list = find_intersections_2d_indexed(
            edge_2d, principal_2d,
            inlier_thres=CLASSIFY_THRES_2D, intersect_thres=INTERSECT_THRES_2D)
        for k in range(3):
            print(f"    inter_2d[{k}]: {inter_2d_list_2d[k].shape[0]}")

        # -- B4: 24 rotation candidates ------------------------------------
        print("\n[B4] Building 24 rotation candidates...")
        rotations, perms_expanded = build_rotation_candidates(principal_2d, principal_3d)
        print(f"    rotations: {rotations.shape}")

        # -- B5: Rearrange intersections -----------------------------------
        print("\n[B5] Rearranging 2D intersections for rotation permutations...")
        inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot = \
            rearrange_intersections_for_rotations(
                inter_2d_list_2d, inter_2d_idx_list, perms_expanded)

        # -- B6: XDF coarse search -----------------------------------------
        if use_local:
            print("\n[B6] XDF coarse search (per-panorama 3D)...")
        else:
            print("\n[B6] XDF coarse search (2D only -- 3D cached)...")
        t0 = time.time()
        top_poses, _cost_matrix = xdf_coarse_search_from_precomputed(
            cur_precomputed_3d,
            rotations, perms_expanded, edge_2d,
            inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot,
            cur_trans_candidates, query_pts,
            top_k=TOP_K, xdf_inlier_thres=XDF_INLIER_THRES,
            point_gamma=POINT_GAMMA,
            inlier_thres_2d=INLIER_THRES_2D)

        for i, p in enumerate(top_poses):
            print(f"    top-{i}: cost={p['cost']:.1f}  rot_idx={p['rot_idx']}  "
                  f"t={p['t'].numpy()}")
        print(f"    XDF search: {time.time() - t0:.1f}s")

        # -- B7: ICP refinement on top-K -----------------------------------
        print("\n[B7] ICP refinement on top-K candidates...")
        candidates = []

        i3d = cur_inter_3d.numpy()
        i3d_mask = cur_inter_3d_mask.numpy()
        i3d_idx = cur_inter_3d_idx.numpy()

        for i, pose in enumerate(top_poses):
            R_init = pose['R'].numpy()
            t_init = pose['t'].numpy()
            i2d = pose['inter_2d'].numpy()
            i2d_mask = pose['inter_2d_mask'].numpy()
            i2d_idx = pose['inter_2d_idx'].numpy()

            R_ref, t_ref, matched = refine_pose(
                R_init, t_init, i2d, i3d, i2d_mask, i3d_mask,
                line_normals_2d=edge_2d[:, :3].numpy(),
                line_dirs_3d=dense_dirs_full_np,
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
                'avg_dist': float(avg_dist),
            })

        # Select best: highest tight inlier count, then lowest avg_dist
        candidates.sort(key=lambda c: (-c['n_tight'], c['avg_dist']))
        best = candidates[0]
        final_R = best['R']
        final_t = best['t']
        matched_pairs = best['matched']

        print(f"\n    BEST: n_matched={best['n_matched']}  n_tight={best['n_tight']}  "
              f"avg_dist={best['avg_dist']:.4f}  t={final_t}")
        print(f"    det(R) = {np.linalg.det(final_R):.6f}")

        # Collect results for local_filter_results.json
        result_entry = {
            'n_tight': best['n_tight'],
            'avg_dist': best['avg_dist'],
            'n_matched': best['n_matched'],
            't': final_t.tolist(),
            'R': final_R.tolist(),
        }
        if use_local:
            result_entry['n_translations'] = n_trans
            result_entry['n_dense_lines'] = n_dense
            result_entry['n_sparse_lines'] = n_sparse
            result_entry['n_intersections'] = n_inter
        all_results[pano_name] = result_entry

        # -- B8: Save camera_pose.json -------------------------------------
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

        # -- B9: Visualization ---------------------------------------------
        print("\n[B9] Generating visualizations...")
        pano_img = cv2.imread(str(pano_img_path))
        if pano_img is not None:
            pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)

            # side_by_side.png -- use FULL sparse lines for context
            starts_np = starts_sparse.cpu().numpy()
            ends_np = ends_sparse.cpu().numpy()

            sbs = render_side_by_side(
                pano_img, edge_2d, starts_np, ends_np,
                final_R, final_t, resolution=(512, 1024))

            vis_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_dir / "side_by_side.png"),
                        cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))
            print(f"    Saved {vis_dir / 'side_by_side.png'}")

            if use_local:
                # reprojection.png -- point cloud depth overlay
                reproj = render_reprojection(pano_img, pc_pts, final_R, final_t)
                cv2.imwrite(str(vis_dir / "reprojection.png"),
                            cv2.cvtColor(reproj, cv2.COLOR_RGB2BGR))
                print(f"    Saved {vis_dir / 'reprojection.png'}")

                # topdown.png -- camera on density image + floorplan
                topdown = render_topdown(density_img, density_meta, final_t, final_R,
                                         room_polygons=room_polygons)
                cv2.imwrite(str(vis_dir / "topdown.png"),
                            cv2.cvtColor(topdown, cv2.COLOR_RGB2BGR))
                print(f"    Saved {vis_dir / 'topdown.png'}")
        else:
            print(f"    WARNING: could not load panorama from {pano_img_path}")

        dt_pano = time.time() - t_pano
        print(f"\n    Pano {pano_name} total: {dt_pano:.1f}s")

    # ================================================================
    # Output: local_filter_results.json
    # ================================================================
    out_json = output_base / "local_filter_results.json"
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_json}")

    # -- Summary -----------------------------------------------------------
    dt_total = time.time() - t_start
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Mode: {'LOCAL FILTERING' if use_local else 'GLOBAL'}")
    if not use_local:
        print(f"  3D precompute: {dt_3d:.1f}s")
    for pano_name in pano_names:
        r = all_results[pano_name]
        pose_path = output_base / pano_name / "camera_pose.json"
        vis_path = output_base / pano_name / "vis" / "side_by_side.png"
        print(f"  {pano_name}:")
        print(f"    n_tight={r['n_tight']}  avg_dist={r['avg_dist']:.4f}  "
              f"t={r['t']}")
        print(f"    pose: {'OK' if pose_path.exists() else 'MISSING'}")
        print(f"    vis:  {'OK' if vis_path.exists() else 'MISSING'}")
    print(f"  Total: {dt_total:.1f}s")


if __name__ == "__main__":
    main()
