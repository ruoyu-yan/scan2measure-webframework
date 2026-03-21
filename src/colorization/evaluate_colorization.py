"""Evaluate colorization accuracy against ground-truth point cloud RGB.

Compares per-point colors between the original RGB point cloud (from the
Leica BLK360 G1 scanner camera) and our panorama-based colorization.
Same camera, same position — so this is a true ground-truth comparison.

Usage:
    python src/colorization/evaluate_colorization.py
"""

import time
from pathlib import Path

import numpy as np
import open3d as o3d

# ── Config ──────────────────────────────────────────────────────────────────
ORIGINAL_PLY = "tmb_office_one_corridor_dense"
TEXTURED_PLY = "tmb_office_one_corridor_dense_noRGB_textured"

ROOT = Path(__file__).resolve().parent.parent.parent
ORIGINAL_PATH = ROOT / "data" / "raw_point_cloud" / f"{ORIGINAL_PLY}.ply"
TEXTURED_PATH = ROOT / "data" / "textured_point_cloud" / f"{TEXTURED_PLY}.ply"
OUTPUT_DIR = ROOT / "data" / "textured_point_cloud"


def main():
    t_start = time.time()

    # ── [1] Load both point clouds ───────────────────────────────────
    print("=" * 60)
    print("[1] Loading point clouds")
    print("=" * 60)

    pcd_orig = o3d.io.read_point_cloud(str(ORIGINAL_PATH))
    pcd_text = o3d.io.read_point_cloud(str(TEXTURED_PATH))

    N_orig = len(pcd_orig.points)
    N_text = len(pcd_text.points)
    print(f"    Original:  {N_orig:,} points")
    print(f"    Textured:  {N_text:,} points")

    assert N_orig == N_text, (
        f"Point count mismatch: {N_orig} vs {N_text}")

    colors_orig = np.asarray(pcd_orig.colors)   # (N, 3) float64 [0, 1]
    colors_text = np.asarray(pcd_text.colors)    # (N, 3) float64 [0, 1]
    N = N_orig

    # ── [2] Identify colored vs uncolored ────────────────────────────
    print(f"\n[2] Coverage")
    colored_mask = np.any(colors_text > 0, axis=1)
    n_colored = colored_mask.sum()
    n_uncolored = N - n_colored
    print(f"    Colored:   {n_colored:,} / {N:,} ({100 * n_colored / N:.1f}%)")
    print(f"    Uncolored: {n_uncolored:,} ({100 * n_uncolored / N:.1f}%)")

    # Work only with colored points
    orig_c = colors_orig[colored_mask]
    text_c = colors_text[colored_mask]

    # ── [3] RGB L2 distance ──────────────────────────────────────────
    print(f"\n[3] RGB L2 distance (range [0, {np.sqrt(3):.2f}])")
    diff = orig_c - text_c
    rgb_l2 = np.linalg.norm(diff, axis=1)

    p50, p90, p95 = np.percentile(rgb_l2, [50, 90, 95])
    print(f"    Mean:   {rgb_l2.mean():.4f}")
    print(f"    Median: {p50:.4f}")
    print(f"    90th:   {p90:.4f}")
    print(f"    95th:   {p95:.4f}")
    print(f"    Max:    {rgb_l2.max():.4f}")

    # ── [4] CIEDE2000 Delta-E ────────────────────────────────────────
    print(f"\n[4] CIEDE2000 Delta-E (perceptual)")
    try:
        from skimage.color import rgb2lab, deltaE_ciede2000

        # rgb2lab expects (M, N, 3) image shape — reshape to (1, M, 3)
        lab_orig = rgb2lab(orig_c[np.newaxis, :, :]).squeeze(0)
        lab_text = rgb2lab(text_c[np.newaxis, :, :]).squeeze(0)

        delta_e = deltaE_ciede2000(lab_orig, lab_text)

        p50, p90, p95 = np.percentile(delta_e, [50, 90, 95])
        print(f"    Mean:   {delta_e.mean():.2f}")
        print(f"    Median: {p50:.2f}")
        print(f"    90th:   {p90:.2f}")
        print(f"    95th:   {p95:.2f}")
        print(f"    Max:    {delta_e.max():.2f}")
        print(f"    ---")
        print(f"    < 1   (imperceptible):   "
              f"{(delta_e < 1).sum():,} ({100 * (delta_e < 1).mean():.1f}%)")
        print(f"    < 2.3 (barely noticeable):"
              f" {(delta_e < 2.3).sum():,} ({100 * (delta_e < 2.3).mean():.1f}%)")
        print(f"    < 5   (noticeable):       "
              f"{(delta_e < 5).sum():,} ({100 * (delta_e < 5).mean():.1f}%)")
        print(f"    < 10  (obvious):          "
              f"{(delta_e < 10).sum():,} ({100 * (delta_e < 10).mean():.1f}%)")
    except ImportError:
        print("    SKIPPED: scikit-image not installed "
              "(pip install scikit-image)")

    # ── [5] Per-channel bias ─────────────────────────────────────────
    print(f"\n[5] Per-channel signed bias (textured - original)")
    bias = diff.mean(axis=0)
    print(f"    R: {bias[0]:+.4f}")
    print(f"    G: {bias[1]:+.4f}")
    print(f"    B: {bias[2]:+.4f}")
    mag = np.linalg.norm(bias)
    if mag > 0.02:
        print(f"    NOTE: systematic color shift detected (magnitude {mag:.4f})")
    else:
        print(f"    No significant systematic shift (magnitude {mag:.4f})")

    # ── [6] Save error heatmap PLY ───────────────────────────────────
    print(f"\n[6] Saving error heatmap PLY")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm

    # Normalize error to [0, 1] using 95th percentile as max
    err_full = np.zeros(N, dtype=np.float64)
    err_full[colored_mask] = rgb_l2
    p95_full = np.percentile(rgb_l2, 95)
    err_norm = np.clip(err_full / max(p95_full, 1e-6), 0, 1)

    # Apply colormap: green (low error) → yellow → red (high error)
    cmap = cm.get_cmap('RdYlGn_r')
    heatmap_colors = cmap(err_norm)[:, :3]

    # Uncolored points in blue
    heatmap_colors[~colored_mask] = [0.2, 0.2, 1.0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{TEXTURED_PLY}_error_heatmap.ply"
    pcd_heat = o3d.geometry.PointCloud()
    pcd_heat.points = pcd_orig.points
    pcd_heat.colors = o3d.utility.Vector3dVector(heatmap_colors)
    o3d.io.write_point_cloud(str(out_path), pcd_heat)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"    Legend: green=low error, red=high error, blue=uncolored")

    # ── [7] Summary ──────────────────────────────────────────────────
    dt = time.time() - t_start
    print(f"\nTotal time: {dt:.1f}s")


if __name__ == "__main__":
    main()
