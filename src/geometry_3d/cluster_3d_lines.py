"""Orchestrator: load V4 wireframe → cluster by principal direction → intersect → save.

Reads room_geometry.pkl from point_cloud_geometry_baker_V4.py output,
runs the FGPL-faithful clustering and intersection pipeline via
line_clustering_3d.py, and writes:
  - 3d_line_map.pkl    (serialized map_dict for downstream pose estimation)
  - clustered_lines.obj + .mtl   (lines colored by principal group)
  - intersections.obj + .mtl     (octahedra at intersection points)
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from config_loader import load_config, progress

from line_clustering_3d import (
    vote_principal_directions,
    classify_lines_3d,
    find_intersections_3d,
    build_intersection_masks,
    write_colored_lines_obj,
    write_intersections_obj,
)

# ── Configuration (matching FGPL defaults) ────────────────────────────
POINT_CLOUD_NAME    = "tmb_office_one_corridor_bigger_noRGB"
SPARSE_LENGTH_FACTOR = 0.10      # fraction of extent for sparse set
DENSE_LENGTH_THRES  = 0.2        # meters, fixed threshold for dense set
INLIER_THRES        = 0.1        # classification cosine threshold
INTERSECT_THRES     = 0.2        # intersection distance threshold

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "debug_renderer" / POINT_CLOUD_NAME
PKL_IN       = DATA_DIR / "room_geometry.pkl"


def main():
    cfg = load_config()
    pc_name = cfg.get("point_cloud_name", POINT_CLOUD_NAME)
    pkl_in = Path(cfg["input_pkl"]) if cfg.get("input_pkl") else PROJECT_ROOT / "data" / "debug_renderer" / pc_name / "room_geometry.pkl"
    data_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else PROJECT_ROOT / "data" / "debug_renderer" / pc_name

    start_time = time.time()

    # ── 1. Load room_geometry.pkl ─────────────────────────────────────
    progress(1, 4, "Loading room geometry")
    print(f"Loading {pkl_in} ...")
    with open(pkl_in, 'rb') as f:
        bake = pickle.load(f)
    segments = bake['wireframe_segments']
    print(f"  {len(segments)} line segments")

    all_starts = torch.tensor([s['start'] for s in segments], dtype=torch.float32)
    all_ends   = torch.tensor([s['end']   for s in segments], dtype=torch.float32)
    all_dirs   = all_ends - all_starts
    all_lengths = torch.norm(all_dirs, dim=1)
    all_dirs   = all_dirs / all_lengths.unsqueeze(-1).clamp(min=1e-8)

    # ── 2. Compute extent and split sparse / dense ────────────────────
    midpoints = (all_starts + all_ends) / 2.0
    extent = (midpoints.max(0).values - midpoints.min(0).values).mean().item()
    sparse_thres = extent * SPARSE_LENGTH_FACTOR
    print(f"  Extent: {extent:.3f} m → sparse threshold: {sparse_thres:.3f} m")

    sparse_mask = all_lengths >= sparse_thres
    dense_mask  = all_lengths >= DENSE_LENGTH_THRES

    sparse_dirs   = all_dirs[sparse_mask]
    sparse_starts = all_starts[sparse_mask]
    sparse_ends   = all_ends[sparse_mask]

    dense_dirs   = all_dirs[dense_mask]
    dense_starts = all_starts[dense_mask]
    dense_ends   = all_ends[dense_mask]

    print(f"  Sparse lines (>= {sparse_thres:.3f} m): {sparse_dirs.shape[0]}")
    print(f"  Dense lines  (>= {DENSE_LENGTH_THRES} m): {dense_dirs.shape[0]}")

    # ── 3. Vote principal directions (on sparse set) ──────────────────
    progress(2, 4, "Voting principal directions and classifying lines")
    print("\nVoting principal directions ...")
    principal_3d = vote_principal_directions(sparse_dirs)

    # ── 4. Classify lines ─────────────────────────────────────────────
    edge_mask_sparse = classify_lines_3d(sparse_dirs, principal_3d, INLIER_THRES)
    edge_mask_dense  = classify_lines_3d(dense_dirs, principal_3d, INLIER_THRES)

    # ── 5. Find intersections (on dense set) ──────────────────────────
    progress(3, 4, "Finding 3D intersections")
    print("Finding intersections ...")
    inter_pts, inter_idx = find_intersections_3d(
        dense_dirs, dense_starts, dense_ends, principal_3d,
        inlier_thres=INLIER_THRES, intersect_thres=INTERSECT_THRES,
    )

    # ── 6. Build masks ────────────────────────────────────────────────
    inter_3d_mask = build_intersection_masks(inter_pts, device=dense_dirs.device)
    inter_3d      = torch.cat([p for p in inter_pts], dim=0)
    inter_3d_idx  = torch.cat([p for p in inter_idx], dim=0)

    # ── 7. Save 3d_line_map.pkl ───────────────────────────────────────
    map_dict = {
        'dirs':             sparse_dirs,
        'starts':           sparse_starts,
        'ends':             sparse_ends,
        'dense_dirs':       dense_dirs,
        'dense_starts':     dense_starts,
        'dense_ends':       dense_ends,
        'principal_3d':     principal_3d,
        'inter_3d':         inter_3d,
        'inter_3d_idx':     inter_3d_idx,
        'inter_3d_mask':    inter_3d_mask,
        'edge_mask_sparse': edge_mask_sparse,
        'edge_mask_dense':  edge_mask_dense,
    }
    progress(4, 4, "Saving 3d_line_map.pkl")
    pkl_out = data_dir / "3d_line_map.pkl"
    with open(pkl_out, 'wb') as f:
        pickle.dump(map_dict, f)
    print(f"\nSaved {pkl_out}")

    # ── 8. Write clustered_lines.obj ──────────────────────────────────
    obj_lines = data_dir / "clustered_lines.obj"
    write_colored_lines_obj(obj_lines, dense_starts, dense_ends,
                            edge_mask_dense, principal_3d)
    print(f"Saved {obj_lines}")

    # ── 9. Write intersections.obj ────────────────────────────────────
    obj_inter = data_dir / "intersections.obj"
    write_intersections_obj(obj_inter, inter_pts)
    print(f"Saved {obj_inter}")

    # ── 10. Print statistics ──────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("3D LINE CLUSTERING & INTERSECTION RESULTS")
    print("=" * 60)

    print(f"\nPrincipal 3D directions:")
    for i in range(3):
        d = principal_3d[i].numpy()
        print(f"  Direction {i}: [{d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}]")
    det = torch.det(principal_3d).item()
    print(f"  Determinant: {det:.4f} ({'right-handed' if det > 0 else 'left-handed'})")

    print(f"\nSparse lines (for voting): {sparse_dirs.shape[0]}")
    for i in range(3):
        count = edge_mask_sparse[:, i].sum().item()
        print(f"  Group {i}: {count} lines "
              f"({100 * count / sparse_dirs.shape[0]:.1f}%)")
    unclass_s = (~edge_mask_sparse.any(dim=1)).sum().item()
    print(f"  Unclassified: {unclass_s} lines "
          f"({100 * unclass_s / sparse_dirs.shape[0]:.1f}%)")

    print(f"\nDense lines (for intersections): {dense_dirs.shape[0]}")
    for i in range(3):
        count = edge_mask_dense[:, i].sum().item()
        print(f"  Group {i}: {count} lines "
              f"({100 * count / dense_dirs.shape[0]:.1f}%)")
    unclass_d = (~edge_mask_dense.any(dim=1)).sum().item()
    print(f"  Unclassified: {unclass_d} lines "
          f"({100 * unclass_d / dense_dirs.shape[0]:.1f}%)")

    print(f"\n3D intersections (total): {inter_3d.shape[0]}")
    for i in range(3):
        j = (i + 1) % 3
        pair = inter_3d_mask[:, i] & inter_3d_mask[:, j]
        count = pair.sum().item()
        print(f"  Group {i} ∩ Group {j}: {count} intersections")

    print(f"\nFinished in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
