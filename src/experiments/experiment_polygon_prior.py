"""Polygon-only translation prior experiment (B-only).

Tests whether restricting translation candidates to the RoomFormer polygon
assigned to each panorama (+ margin) improves pose estimation.

Configs:
  baseline     — all translation candidates (identical to multiroom_pose_estimation.py)
  B_only       — translations filtered by polygon containment

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
from matplotlib.path import Path as MplPath

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
from visualize_pose import render_side_by_side
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

# Polygon prior
POLYGON_MARGIN_M = 1.5

# Experiment configurations
CONFIGS = [
    {"name": "baseline", "polygon_filter": False},
    {"name": "B_only",   "polygon_filter": True},
]

# Paths
ROOT = Path(__file__).resolve().parent.parent
PKL_3D_PATH = ROOT / "data" / "debug_renderer" / POINT_CLOUD_NAME / "3d_line_map.pkl"
ALIGNMENT_PATH = ROOT / "data" / "reconstructed_floorplans_RoomFormer" / POINT_CLOUD_NAME / "global_alignment.json"
PREDICTIONS_PATH = ROOT / "data" / "reconstructed_floorplans_RoomFormer" / POINT_CLOUD_NAME / "predictions.json"
METADATA_PATH = ROOT / "data" / "density_image" / POINT_CLOUD_NAME / "metadata.json"
OUTPUT_BASE = ROOT / "data" / "pose_estimates" / "multiroom"


# ── Polygon helpers ─────────────────────────────────────────────────────────

def pixels_to_raw_3d(pixel_poly, metadata):
    """Convert pixel polygon coordinates to raw 3D world XY (meters)."""
    R = np.array(metadata['rotation_matrix'])
    min_coords = np.array(metadata['min_coords'])
    offset = np.array(metadata['offset'])
    max_dim = metadata['max_dim']
    scale = metadata['image_width'] - 1
    result = []
    for (u, v) in pixel_poly:
        aligned_mm = np.array([u, v]) / scale * max_dim - offset[:2] + min_coords[:2]
        raw_3d = R.T @ np.array([aligned_mm[0], aligned_mm[1], 0.0]) / 1000.0
        result.append(raw_3d[:2])
    return np.array(result)


def expand_polygon(poly_xy, margin):
    """Expand polygon outward from centroid by margin (meters)."""
    centroid = poly_xy.mean(axis=0)
    directions = poly_xy - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    expanded = poly_xy + directions / norms * margin
    return expanded


def load_polygon_data():
    """Load polygon assignments and transform to 3D world coordinates.

    Returns dict: pano_name -> expanded polygon XY (meters) as MplPath.
    """
    with open(ALIGNMENT_PATH) as f:
        alignment = json.load(f)
    with open(PREDICTIONS_PATH) as f:
        predictions = json.load(f)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    # Build pano_name -> rf_idx mapping
    pano_to_rf = {}
    for entry in alignment['alignment_results']:
        pano_to_rf[entry['room_name']] = entry['rf_idx']

    # Transform each polygon to 3D and expand
    polygon_paths = {}
    for pano_name in PANO_NAMES:
        rf_idx = pano_to_rf[pano_name]
        pixel_poly = np.array(predictions[rf_idx])
        poly_3d_xy = pixels_to_raw_3d(pixel_poly, metadata)
        expanded = expand_polygon(poly_3d_xy, POLYGON_MARGIN_M)
        polygon_paths[pano_name] = MplPath(expanded)
        print(f"    {pano_name}: rf_idx={rf_idx}, "
              f"polygon vertices={len(pixel_poly)}, "
              f"centroid=({poly_3d_xy.mean(0)[0]:.2f}, {poly_3d_xy.mean(0)[1]:.2f})")

    return polygon_paths


def filter_translations_by_polygon(trans_candidates, polygon_path):
    """Return boolean mask of translations inside the polygon (XY only)."""
    xy = trans_candidates[:, :2].numpy()
    mask = polygon_path.contains_points(xy)
    return torch.tensor(mask, dtype=torch.bool)


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
    # PHASE A: One-time 3D setup
    # ================================================================
    print("=" * 60)
    print("PHASE A: One-time 3D precomputation")
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

    print(f"    dense segments: {dense_starts.shape[0]}, "
          f"sparse segments: {starts_sparse.shape[0]}")

    # ── A2: Generate translation grid ─────────────────────────────────
    print("\n[A2] Generating translation grid...")
    trans_candidates = generate_translation_grid(
        starts_sparse, ends_sparse,
        num_trans=NUM_TRANS, chamfer_min_dist=CHAMFER_MIN_DIST)
    n_total = trans_candidates.shape[0]
    print(f"    {n_total} translation candidates")

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
    print(f"    Precompute: {dt_3d:.1f}s")

    # ── A5: Load polygon data ─────────────────────────────────────────
    print("\n[A5] Loading polygon data for translation filtering...")
    polygon_paths = load_polygon_data()

    # ── A6: Pre-compute polygon masks ─────────────────────────────────
    print("\n[A6] Computing polygon containment masks...")
    polygon_masks = {}
    for pano_name in PANO_NAMES:
        mask = filter_translations_by_polygon(trans_candidates, polygon_paths[pano_name])
        polygon_masks[pano_name] = mask
        print(f"    {pano_name}: {mask.sum().item()}/{n_total} inside polygon")

    # ================================================================
    # PHASE B: Per-config, per-panorama pose estimation
    # ================================================================
    all_results = {}

    for config in CONFIGS:
        config_name = config['name']
        use_polygon = config['polygon_filter']
        all_results[config_name] = {}

        print("\n" + "#" * 60)
        print(f"CONFIG: {config_name} (polygon_filter={use_polygon})")
        print("#" * 60)

        for pano_idx, pano_name in enumerate(PANO_NAMES):
            print("\n" + "=" * 60)
            print(f"[{config_name}] PANO {pano_idx+1}/{len(PANO_NAMES)}: {pano_name}")
            print("=" * 60)

            t_pano = time.time()

            features_2d_path = (
                ROOT / "data" / "pano" / "2d_feature_extracted"
                / f"{pano_name}_v2" / "fgpl_features.json"
            )
            pano_img_path = ROOT / "data" / "pano" / "raw" / f"{pano_name}.jpg"

            # ── B1: Load 2D features ──────────────────────────────────
            print(f"\n[B1] Loading 2D features...")
            with open(features_2d_path) as f:
                feat_2d = json.load(f)

            edge_2d = torch.tensor(feat_2d['lines'], dtype=torch.float32, device=device)
            principal_2d = torch.tensor(feat_2d['principal_2d'], dtype=torch.float32, device=device)
            print(f"    edge_2d: {edge_2d.shape}")

            # ── B2: 2D intersections ──────────────────────────────────
            print("\n[B2] Computing 2D intersections...")
            inter_2d_list_2d, inter_2d_idx_list = find_intersections_2d_indexed(
                edge_2d, principal_2d,
                inlier_thres=CLASSIFY_THRES_2D, intersect_thres=INTERSECT_THRES_2D)

            # ── B3: 24 rotation candidates ────────────────────────────
            print("\n[B3] Building 24 rotation candidates...")
            rotations, perms_expanded = build_rotation_candidates(principal_2d, principal_3d)

            # ── B4: Rearrange intersections ───────────────────────────
            print("\n[B4] Rearranging 2D intersections for rotations...")
            inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot = \
                rearrange_intersections_for_rotations(
                    inter_2d_list_2d, inter_2d_idx_list, perms_expanded)

            # ── B5: Prepare translation candidates ────────────────────
            if use_polygon:
                keep_mask = polygon_masks[pano_name]
                tc = trans_candidates[keep_mask]
                # Slice precomputed 3D arrays
                pc3d = {
                    'canonical_rot': precomputed_3d['canonical_rot'],
                    'ldf_3d': precomputed_3d['ldf_3d'][keep_mask],
                    'pdf_3d': precomputed_3d['pdf_3d'][keep_mask],
                    'mask_3d': precomputed_3d['mask_3d'],
                }
                n_tc = tc.shape[0]
                print(f"\n[B5] Polygon filter: {n_tc}/{n_total} translations kept")
            else:
                tc = trans_candidates
                pc3d = precomputed_3d
                n_tc = n_total
                print(f"\n[B5] No filter: {n_tc} translations")

            # ── B6: XDF coarse search ─────────────────────────────────
            print("\n[B6] XDF coarse search...")
            t0 = time.time()
            top_poses, _cost_matrix = xdf_coarse_search_from_precomputed(
                pc3d,
                rotations, perms_expanded, edge_2d,
                inter_2d_per_rot, inter_2d_mask_per_rot, inter_2d_idx_per_rot,
                tc, query_pts,
                top_k=TOP_K, xdf_inlier_thres=XDF_INLIER_THRES,
                point_gamma=POINT_GAMMA,
                inlier_thres_2d=INLIER_THRES_2D)
            print(f"    XDF search: {time.time() - t0:.1f}s")

            for i, p in enumerate(top_poses):
                print(f"    top-{i}: cost={p['cost']:.1f}  rot_idx={p['rot_idx']}  "
                      f"t={p['t'].numpy()}")

            # ── B7: ICP refinement on top-K ───────────────────────────
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

                # ── B8: Compute metrics ───────────────────────────────
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

            all_results[config_name][pano_name] = {
                'n_tight': best['n_tight'],
                'avg_dist': best['avg_dist'],
                'n_matched': best['n_matched'],
                'n_translations': n_tc,
                't': final_t.tolist(),
            }

            # ── B9: Visualization ─────────────────────────────────────
            print("\n[B9] Generating side_by_side.png...")
            pano_img = cv2.imread(str(pano_img_path))
            if pano_img is not None:
                pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)
                starts_np = starts_sparse.cpu().numpy()
                ends_np = ends_sparse.cpu().numpy()

                sbs = render_side_by_side(
                    pano_img, edge_2d, starts_np, ends_np,
                    final_R, final_t, resolution=(512, 1024))

                vis_dir = OUTPUT_BASE / "polygon_ablation_vis" / config_name / pano_name
                vis_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vis_dir / "side_by_side.png"),
                            cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))
                print(f"    Saved {vis_dir / 'side_by_side.png'}")
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
    header = f"{'Config':<12}"
    for pano_name in PANO_NAMES:
        short = pano_name.replace("TMB_", "")
        header += f" | {short:>20s}"
    print(header)

    subhdr = f"{'':12}"
    for _ in PANO_NAMES:
        subhdr += f" | {'tight  avg_d    Nt':>20s}"
    print(subhdr)
    print("-" * len(header))

    for config in CONFIGS:
        cn = config['name']
        row = f"{cn:<12}"
        for pano_name in PANO_NAMES:
            r = all_results[cn][pano_name]
            cell = f"{r['n_tight']:>5d}  {r['avg_dist']:.3f}  {r['n_translations']:>4d}"
            row += f" | {cell:>20s}"
        print(row)

    print(f"\nTotal time: {dt_total:.1f}s")

    # Save JSON results
    out_json = OUTPUT_BASE / "polygon_ablation_results.json"
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
