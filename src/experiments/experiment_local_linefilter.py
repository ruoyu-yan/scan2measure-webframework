"""Local 3D line filtering experiment.

Spatially filters the combined 3D line map per panorama using Voronoi
assignment (nearest panorama) + overlap margin, so each panorama only
sees local geometry during pose estimation.

No modifications to existing library modules.
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
    precompute_xdf_3d,
    xdf_coarse_search_from_precomputed,
)
from pose_refine import refine_pose
from visualize_pose import render_side_by_side, render_reprojection, render_topdown
from line_analysis import classify_lines

# ── Config ──────────────────────────────────────────────────────────────────
POINT_CLOUD_NAME = "tmb_office_one_corridor_dense"
PANO_NAMES = ["TMB_office1", "TMB_corridor_south1", "TMB_corridor_south2"]
DEVICE = "cpu"

# Algorithm constants (same as multiroom_pose_estimation.py)
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

# Local line filter
OVERLAP_MARGIN = 2.0  # meters, buffer beyond Voronoi boundary

# Paths
ROOT = Path(__file__).resolve().parent.parent
PKL_3D_PATH = ROOT / "data" / "debug_renderer" / POINT_CLOUD_NAME / "3d_line_map.pkl"
ALIGNMENT_PATH = ROOT / "data" / "reconstructed_floorplans_RoomFormer" / POINT_CLOUD_NAME / "global_alignment.json"
METADATA_PATH = ROOT / "data" / "density_image" / POINT_CLOUD_NAME / "metadata.json"
PC_PATH = ROOT / "data" / "raw_point_cloud" / f"{POINT_CLOUD_NAME}.ply"
DENSITY_IMG_PATH = ROOT / "data" / "density_image" / POINT_CLOUD_NAME / f"{POINT_CLOUD_NAME}.png"
ROOMFORMER_PATH = ROOT / "data" / "reconstructed_floorplans_RoomFormer" / POINT_CLOUD_NAME / "predictions.json"
OUTPUT_BASE = ROOT / "data" / "pose_estimates" / "multiroom"


# ── Helpers ─────────────────────────────────────────────────────────────────

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


def load_panorama_positions():
    """Load panorama 3D XY positions from global_alignment.json.

    Returns dict: pano_name -> np.array([x, y]) in meters.
    """
    with open(ALIGNMENT_PATH) as f:
        alignment = json.load(f)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    positions = {}
    for entry in alignment['alignment_results']:
        name = entry['room_name']
        if name in PANO_NAMES:
            xy = pixel_to_raw_3d(entry['camera_pose_global'], metadata)
            positions[name] = xy
            print(f"    {name}: XY=({xy[0]:.2f}, {xy[1]:.2f})")
    return positions


def compute_voronoi_assignment(points_xy, pano_positions):
    """Assign each point to nearest panorama. Returns (N,) int array of pano indices."""
    pano_xys = np.array([pano_positions[name] for name in PANO_NAMES])  # (P, 2)
    dists = np.linalg.norm(points_xy[:, None, :] - pano_xys[None, :, :], axis=2)  # (N, P)
    return np.argmin(dists, axis=1)


def get_local_mask(points_xy, pano_idx, pano_positions, voronoi_labels, margin):
    """Get boolean mask for points belonging to this panorama (Voronoi + margin).

    Includes: Voronoi-assigned points + any point within margin of this panorama.
    """
    voronoi_mask = voronoi_labels == pano_idx
    pano_xy = pano_positions[PANO_NAMES[pano_idx]]
    dist_to_pano = np.linalg.norm(points_xy - pano_xy[None, :], axis=1)
    margin_mask = dist_to_pano <= margin
    return voronoi_mask | margin_mask


def compute_metrics(candidates_list):
    """Select best candidate and return metrics dict."""
    candidates_list.sort(key=lambda c: (-c['n_tight'], c['avg_dist']))
    best = candidates_list[0]
    return {
        'n_tight': best['n_tight'],
        'avg_dist': best['avg_dist'],
        'n_matched': best['n_matched'],
        't': best['t'].tolist(),
        'R': best['R'].tolist(),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device(DEVICE)
    t_start = time.time()

    # ================================================================
    # PHASE A: One-time setup
    # ================================================================
    print("=" * 60)
    print("PHASE A: One-time setup")
    print("=" * 60)

    # ── A1: Load 3D map ───────────────────────────────────────────────
    print("\n[A1] Loading 3D wireframe from 3d_line_map.pkl...")
    with open(PKL_3D_PATH, 'rb') as f:
        line_map = pickle.load(f)

    dense_starts = line_map['dense_starts'].to(device).float()
    dense_ends = line_map['dense_ends'].to(device).float()
    dense_dirs = line_map['dense_dirs'].to(device).float()
    principal_3d = line_map['principal_3d'].to(device).float()

    inter_3d_all = line_map['inter_3d'].to(device).float()
    inter_3d_idx_all = line_map['inter_3d_idx'].to(device).long()
    inter_3d_mask_all = line_map['inter_3d_mask'].to(device).bool()

    starts_sparse = line_map['starts'].to(device).float()
    ends_sparse = line_map['ends'].to(device).float()

    # Keep full dense_dirs for refine_pose (inter_3d_idx indexes into original array)
    dense_dirs_full_np = dense_dirs.numpy()

    print(f"    dense segments: {dense_starts.shape[0]}, "
          f"sparse segments: {starts_sparse.shape[0]}")

    # ── A2: Load panorama positions ───────────────────────────────────
    print("\n[A2] Loading panorama positions from global_alignment.json...")
    pano_positions = load_panorama_positions()

    # ── A3: Generate query points ─────────────────────────────────────
    print("\n[A3] Generating icosphere query points...")
    query_pts = generate_sphere_points(QUERY_SPHERE_LEVEL, device=device)
    print(f"    {query_pts.shape[0]} query points (level {QUERY_SPHERE_LEVEL})")

    # ── A4: Compute Voronoi assignments ───────────────────────────────
    print("\n[A4] Computing Voronoi assignments...")
    dense_mid_xy = ((dense_starts + dense_ends) / 2)[:, :2].numpy()
    sparse_mid_xy = ((starts_sparse + ends_sparse) / 2)[:, :2].numpy()
    inter_xy = inter_3d_all[:, :2].numpy()

    voronoi_dense = compute_voronoi_assignment(dense_mid_xy, pano_positions)
    voronoi_sparse = compute_voronoi_assignment(sparse_mid_xy, pano_positions)
    voronoi_inter = compute_voronoi_assignment(inter_xy, pano_positions)

    for pi, name in enumerate(PANO_NAMES):
        nd = (voronoi_dense == pi).sum()
        ns = (voronoi_sparse == pi).sum()
        ni = (voronoi_inter == pi).sum()
        print(f"    {name}: dense={nd}, sparse={ns}, inter={ni}")

    # ── A5: Load visualization data ───────────────────────────────────
    print("\n[A5] Loading visualization data (point cloud, density image, polygons)...")
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(PC_PATH))
    pc_pts = np.asarray(pcd.points)
    rng = np.random.default_rng(42)
    if pc_pts.shape[0] > 50000:
        pc_pts = pc_pts[rng.choice(pc_pts.shape[0], 50000, replace=False)]
    print(f"    Point cloud: {pc_pts.shape[0]} points (subsampled)")

    density_img = cv2.imread(str(DENSITY_IMG_PATH), cv2.IMREAD_GRAYSCALE)
    with open(METADATA_PATH) as f:
        density_meta = json.load(f)

    room_polygons = None
    if ROOMFORMER_PATH.exists():
        with open(ROOMFORMER_PATH) as f:
            room_polygons = json.load(f)
    print(f"    Density image: {density_img.shape}")
    print(f"    RoomFormer polygons: {len(room_polygons) if room_polygons else 0}")

    # ================================================================
    # PHASE B: Per-panorama pose estimation
    # ================================================================
    all_results = {}

    for pano_idx, pano_name in enumerate(PANO_NAMES):
        print("\n" + "=" * 60)
        print(f"PANO {pano_idx+1}/{len(PANO_NAMES)}: {pano_name}")
        print("=" * 60)

        t_pano = time.time()

        features_2d_path = (
            ROOT / "data" / "pano" / "2d_feature_extracted"
            / f"{pano_name}_v2" / "fgpl_features.json"
        )
        pano_img_path = ROOT / "data" / "pano" / "raw" / f"{pano_name}.jpg"

        # ── B1: Filter lines using Voronoi + margin ────────────────────
        print(f"\n[B1] Filtering 3D lines for {pano_name}...")
        dense_mask = get_local_mask(dense_mid_xy, pano_idx, pano_positions, voronoi_dense, OVERLAP_MARGIN)
        sparse_mask = get_local_mask(sparse_mid_xy, pano_idx, pano_positions, voronoi_sparse, OVERLAP_MARGIN)
        inter_mask = get_local_mask(inter_xy, pano_idx, pano_positions, voronoi_inter, OVERLAP_MARGIN)

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

        # ── B2: Generate translation grid from filtered sparse lines ───
        print("\n[B2] Generating translation grid from filtered sparse lines...")
        trans_candidates = generate_translation_grid(
            local_starts, local_ends,
            num_trans=NUM_TRANS, chamfer_min_dist=CHAMFER_MIN_DIST)
        n_trans = trans_candidates.shape[0]
        print(f"    {n_trans} translation candidates")

        # ── B3: Precompute LDF-3D + PDF-3D for filtered lines ──────────
        print("\n[B3] Precomputing 3D distance functions (LDF-3D + PDF-3D)...")
        t0 = time.time()
        precomputed_3d = precompute_xdf_3d(
            principal_3d, local_dense_starts, local_dense_ends, local_dense_dirs,
            local_inter_3d, local_inter_3d_mask,
            trans_candidates, query_pts,
            inlier_thres_3d=INLIER_THRES_3D)
        dt_3d = time.time() - t0
        print(f"    LDF-3D: {precomputed_3d['ldf_3d'].shape}")
        print(f"    PDF-3D: {precomputed_3d['pdf_3d'].shape}")
        print(f"    Precompute: {dt_3d:.1f}s")

        # ── B4: Load 2D features ──────────────────────────────────────
        print(f"\n[B4] Loading 2D features...")
        with open(features_2d_path) as f:
            feat_2d = json.load(f)

        edge_2d = torch.tensor(feat_2d['lines'], dtype=torch.float32, device=device)
        principal_2d = torch.tensor(feat_2d['principal_2d'], dtype=torch.float32, device=device)
        print(f"    edge_2d: {edge_2d.shape}")

        # ── B5: 2D intersections + rotations + rearrange ──────────────
        print("\n[B5] Computing 2D intersections...")
        inter_2d_list_2d, inter_2d_idx_list = find_intersections_2d_indexed(
            edge_2d, principal_2d,
            inlier_thres=CLASSIFY_THRES_2D, intersect_thres=INTERSECT_THRES_2D)

        print("    Building 24 rotation candidates...")
        rotations, perms_expanded = build_rotation_candidates(principal_2d, principal_3d)

        print("    Rearranging 2D intersections for rotations...")
        inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot = \
            rearrange_intersections_for_rotations(
                inter_2d_list_2d, inter_2d_idx_list, perms_expanded)

        # ── B6: XDF coarse search ──────────────────────────────────────
        print("\n[B6] XDF coarse search...")
        t0 = time.time()
        top_poses, _cost_matrix = xdf_coarse_search_from_precomputed(
            precomputed_3d,
            rotations, perms_expanded, edge_2d,
            inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot,
            trans_candidates, query_pts,
            top_k=TOP_K, xdf_inlier_thres=XDF_INLIER_THRES,
            point_gamma=POINT_GAMMA,
            inlier_thres_2d=INLIER_THRES_2D)
        print(f"    XDF search: {time.time() - t0:.1f}s")

        for i, p in enumerate(top_poses):
            print(f"    top-{i}: cost={p['cost']:.1f}  rot_idx={p['rot_idx']}  "
                  f"t={p['t'].numpy()}")

        # ── B7: ICP refinement on top-K ────────────────────────────────
        print("\n[B7] ICP refinement on top-K candidates...")
        candidates = []
        # Use FILTERED inter_3d for matching, but FULL dense_dirs for direction lookup
        i3d = local_inter_3d.numpy()
        i3d_mask = local_inter_3d_mask.numpy()
        i3d_idx = local_inter_3d_idx.numpy()

        for i, pose in enumerate(top_poses):
            R_init = pose['R'].numpy()
            t_init = pose['t'].numpy()
            i2d = pose['inter_2d'].numpy()
            i2d_mask = pose['inter_2d_mask'].numpy()
            i2d_idx = pose['inter_2d_idx'].numpy()

            R_ref, t_ref, matched = refine_pose(
                R_init, t_init, i2d, i3d, i2d_mask, i3d_mask,
                line_normals_2d=edge_2d[:, :3].numpy(),
                line_dirs_3d=dense_dirs_full_np,  # FULL dense_dirs, not filtered
                inter_2d_idx=i2d_idx, inter_3d_idx=i3d_idx,
                n_iters_t=100, n_iters_r=50, lr=0.1)

            n_matched = sum(len(mp) for mp in matched)

            # ── B8: Compute metrics ───────────────────────────────────
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

            print(f"    candidate {i}: n_tight={n_tight}  avg_dist={avg_dist:.4f}")

            candidates.append({
                'R': R_ref, 't': t_ref, 'matched': matched,
                'inter_2d': i2d, 'inter_2d_mask': i2d_mask,
                'cost': pose['cost'],
                'n_matched': n_matched,
                'n_tight': int(n_tight),
                'avg_dist': float(avg_dist),
            })

        # Select best
        candidates.sort(key=lambda c: (-c['n_tight'], c['avg_dist']))
        best = candidates[0]
        final_R = best['R']
        final_t = best['t']

        print(f"\n    BEST: n_tight={best['n_tight']}  "
              f"avg_dist={best['avg_dist']:.4f}  t={final_t}")

        all_results[pano_name] = {
            'n_tight': best['n_tight'],
            'avg_dist': best['avg_dist'],
            'n_matched': best['n_matched'],
            'n_translations': n_trans,
            'n_dense_lines': n_dense,
            'n_sparse_lines': n_sparse,
            'n_intersections': n_inter,
            't': final_t.tolist(),
        }

        # ── B9: Visualization ─────────────────────────────────────────
        print("\n[B9] Generating visualizations...")
        pano_img = cv2.imread(str(pano_img_path))
        if pano_img is not None:
            pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)
            vis_dir = OUTPUT_BASE / "local_filter_vis" / pano_name
            vis_dir.mkdir(parents=True, exist_ok=True)

            # side_by_side.png — use FULL sparse lines for context
            starts_np = starts_sparse.cpu().numpy()
            ends_np = ends_sparse.cpu().numpy()
            sbs = render_side_by_side(
                pano_img, edge_2d, starts_np, ends_np,
                final_R, final_t, resolution=(512, 1024))
            cv2.imwrite(str(vis_dir / "side_by_side.png"),
                        cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))
            print(f"    Saved {vis_dir / 'side_by_side.png'}")

            # reprojection.png — point cloud depth overlay
            reproj = render_reprojection(pano_img, pc_pts, final_R, final_t)
            cv2.imwrite(str(vis_dir / "reprojection.png"),
                        cv2.cvtColor(reproj, cv2.COLOR_RGB2BGR))
            print(f"    Saved {vis_dir / 'reprojection.png'}")

            # topdown.png — camera on density image + floorplan
            topdown = render_topdown(density_img, density_meta, final_t, final_R,
                                     room_polygons=room_polygons)
            cv2.imwrite(str(vis_dir / "topdown.png"),
                        cv2.cvtColor(topdown, cv2.COLOR_RGB2BGR))
            print(f"    Saved {vis_dir / 'topdown.png'}")
        else:
            print(f"    WARNING: could not load panorama from {pano_img_path}")

        dt_pano = time.time() - t_pano
        print(f"\n    Pano total: {dt_pano:.1f}s")

    # ================================================================
    # Summary + output
    # ================================================================
    dt_total = time.time() - t_start

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)

    # Load polygon ablation results for comparison if available
    polygon_results_path = OUTPUT_BASE / "polygon_ablation_results.json"
    prev_results = None
    if polygon_results_path.exists():
        with open(polygon_results_path) as f:
            prev_results = json.load(f)
        print("(Including baseline/B_only from polygon_ablation_results.json)")

    header = f"{'Config':<14}"
    for pano_name in PANO_NAMES:
        short = pano_name.replace("TMB_", "")
        header += f" | {short:>24s}"
    print(header)

    subhdr = f"{'':14}"
    for _ in PANO_NAMES:
        subhdr += f" | {'tight  avg_d    Nt    Nd':>24s}"
    print(subhdr)
    print("-" * len(header))

    # Print previous results if available
    if prev_results:
        for config_name in ['baseline', 'B_only']:
            if config_name in prev_results:
                row = f"{config_name:<14}"
                for pano_name in PANO_NAMES:
                    r = prev_results[config_name][pano_name]
                    cell = f"{r['n_tight']:>5d}  {r['avg_dist']:.3f}  {r['n_translations']:>4d}     -"
                    row += f" | {cell:>24s}"
                print(row)

    # Print local filter results
    row = f"{'local_filter':<14}"
    for pano_name in PANO_NAMES:
        r = all_results[pano_name]
        cell = f"{r['n_tight']:>5d}  {r['avg_dist']:.3f}  {r['n_translations']:>4d}  {r['n_dense_lines']:>4d}"
        row += f" | {cell:>24s}"
    print(row)

    print(f"\nTotal time: {dt_total:.1f}s")

    # Save JSON results
    out_json = OUTPUT_BASE / "local_filter_results.json"
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
