"""
Test FGPL 3D line clustering and intersection using native panoramic-localization code.

Converts OBJ 3D lines → FGPL TXT format, then calls generate_line_map_single()
to run principal direction voting + pairwise intersection computation.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OBJ_PATH = PROJECT_ROOT / "data" / "debug_3dlinedetection" / "tmb_office1_lines.obj"
TXT_PATH = PROJECT_ROOT / "data" / "debug_3dlinedetection" / "tmb_office1_lines.txt"
CONFIG_PATH = PROJECT_ROOT / "panoramic-localization" / "config" / "omniscenes_fgpl.ini"

# Add panoramic-localization to import path
PANO_LOC_DIR = PROJECT_ROOT / "panoramic-localization"
sys.path.insert(0, str(PANO_LOC_DIR))


# ── Step 1: Convert OBJ → FGPL TXT format ─────────────────────────────
def obj_to_fgpl_txt(obj_path, txt_path):
    """Convert OBJ (v/l lines) to FGPL format: 'x y z line_id' per row."""
    verts = []
    segments = []
    with open(obj_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'l':
                i, j = int(parts[1]) - 1, int(parts[2]) - 1
                segments.append((i, j))

    with open(txt_path, 'w') as f:
        for line_id, (i, j) in enumerate(segments):
            f.write(f"{verts[i][0]} {verts[i][1]} {verts[i][2]} {line_id}\n")
            f.write(f"{verts[j][0]} {verts[j][1]} {verts[j][2]} {line_id}\n")

    print(f"Converted {len(segments)} line segments ({len(verts)} vertices) → {txt_path.name}")
    return len(segments)


# ── Step 2: Run FGPL pipeline ──────────────────────────────────────────
def main():
    # Convert OBJ → TXT
    num_segments = obj_to_fgpl_txt(OBJ_PATH, TXT_PATH)

    # Import FGPL modules (after sys.path setup)
    from parse_utils import parse_ini
    from map_utils import generate_line_map_single
    from edge_utils import split_3d

    # Parse FGPL config
    cfg = parse_ini(str(CONFIG_PATH))

    # Run generate_line_map_single — does principal direction voting + 3D intersections
    print("\nRunning FGPL generate_line_map_single()...")
    map_dict, topk_ratio, sparse_topk_ratio = generate_line_map_single(
        cfg, str(TXT_PATH), ['test_room']
    )
    room = map_dict['test_room']

    # ── Extract results ────────────────────────────────────────────────
    principal_3d = room['principal_3d']
    inter_3d = room['inter_3d']
    inter_3d_idx = room['inter_3d_idx']
    inter_3d_mask = room['inter_3d_mask']
    dirs = room['dirs']
    starts = room['starts']
    ends = room['ends']
    dense_dirs = room['dense_dirs']
    dense_starts = room['dense_starts']
    dense_ends = room['dense_ends']

    # Classify lines by principal direction (for stats and visualization)
    edge_mask_sparse = split_3d(dirs, principal_3d, inlier_thres=0.1)
    edge_mask_dense = split_3d(dense_dirs, principal_3d, inlier_thres=0.1)

    # ── Print statistics ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FGPL 3D LINE CLUSTERING & INTERSECTION RESULTS")
    print("=" * 60)

    print(f"\nPrincipal 3D directions:")
    for i in range(3):
        d = principal_3d[i].cpu().numpy()
        print(f"  Direction {i}: [{d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}]")
    det = torch.det(principal_3d).item()
    print(f"  Determinant: {det:.4f} ({'right-handed' if det > 0 else 'left-handed'})")

    print(f"\nSparse lines (for voting): {dirs.shape[0]}")
    for i in range(3):
        count = edge_mask_sparse[:, i].sum().item()
        print(f"  Group {i}: {count} lines ({100*count/dirs.shape[0]:.1f}%)")
    unclassified_sparse = (~edge_mask_sparse.any(dim=1)).sum().item()
    print(f"  Unclassified: {unclassified_sparse} lines ({100*unclassified_sparse/dirs.shape[0]:.1f}%)")

    print(f"\nDense lines (for intersections): {dense_dirs.shape[0]}")
    for i in range(3):
        count = edge_mask_dense[:, i].sum().item()
        print(f"  Group {i}: {count} lines ({100*count/dense_dirs.shape[0]:.1f}%)")
    unclassified_dense = (~edge_mask_dense.any(dim=1)).sum().item()
    print(f"  Unclassified: {unclassified_dense} lines ({100*unclassified_dense/dense_dirs.shape[0]:.1f}%)")

    # Intersection stats per group pair
    print(f"\n3D intersections (total): {inter_3d.shape[0]}")
    # Decode group pairs from inter_3d_mask
    for i in range(3):
        j = (i + 1) % 3
        pair_mask = inter_3d_mask[:, i] & inter_3d_mask[:, j]
        count = pair_mask.sum().item()
        print(f"  Group {i} ∩ Group {j}: {count} intersections")

    print(f"\n  topk_ratio: {topk_ratio:.4f}")
    print(f"  sparse_topk_ratio: {sparse_topk_ratio:.4f}")

    # ── Visualization ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 7))
    colors_group = ['red', 'green', 'blue']
    colors_inter = ['orange', 'cyan', 'magenta']

    # Left panel: Dense lines colored by principal group
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('3D Lines by Principal Direction Group')
    s_np = dense_starts.cpu().numpy()
    e_np = dense_ends.cpu().numpy()
    mask_np = edge_mask_dense.cpu().numpy()

    # Draw unclassified lines first (gray, thin)
    unclass = ~mask_np.any(axis=1)
    for idx in np.where(unclass)[0]:
        ax1.plot([s_np[idx, 0], e_np[idx, 0]],
                 [s_np[idx, 1], e_np[idx, 1]],
                 [s_np[idx, 2], e_np[idx, 2]],
                 color='lightgray', linewidth=0.3, alpha=0.4)
    # Draw classified lines
    for g in range(3):
        for idx in np.where(mask_np[:, g])[0]:
            ax1.plot([s_np[idx, 0], e_np[idx, 0]],
                     [s_np[idx, 1], e_np[idx, 1]],
                     [s_np[idx, 2], e_np[idx, 2]],
                     color=colors_group[g], linewidth=0.8, alpha=0.7)
    # Legend
    for g in range(3):
        d = principal_3d[g].cpu().numpy()
        ax1.plot([], [], color=colors_group[g], linewidth=2,
                 label=f'Dir {g}: [{d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f}]')
    ax1.plot([], [], color='lightgray', linewidth=1, label='Unclassified')
    ax1.legend(fontsize=7)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

    # Right panel: Intersection points colored by group pair
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('3D Intersection Points by Group Pair')
    inter_np = inter_3d.cpu().numpy()
    mask_i_np = inter_3d_mask.cpu().numpy()

    for i in range(3):
        j = (i + 1) % 3
        pair = mask_i_np[:, i] & mask_i_np[:, j]
        pts = inter_np[pair]
        if len(pts) > 0:
            ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        s=4, alpha=0.5, color=colors_inter[i],
                        label=f'Group {i}∩{j} ({len(pts)} pts)')
    ax2.legend(fontsize=8)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

    plt.tight_layout()
    out_path = PROJECT_ROOT / "data" / "debug_3dlinedetection" / "fgpl_3d_clustering_test.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
