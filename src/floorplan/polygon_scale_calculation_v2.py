"""
polygon_scale_calculation_v2.py
Three consensus-scale methods for matching SAM3 pano polygons to SAM3 density-image polygons.

Methods:
    A — Edge-Only Distances (consecutive edge length ratios)
    B — Procrustes Shape Matching (RMS spread ratios)
    C — Area Ratio (sqrt of area ratios)

All methods share the signature:
    method(map_polys, pano_polys) -> (best_scale, [(scale, votes), ...])
where map_polys / pano_polys are lists of (N,2) numpy arrays.
"""

import numpy as np
from shapely.geometry import Polygon
from scipy.signal import find_peaks


# ==========================================
# SHARED UTILITIES
# ==========================================

def _safe_area(coords):
    """Polygon area via Shapely.  Returns 0 for degenerate inputs."""
    if len(coords) < 3:
        return 0.0
    return Polygon(coords).area


def _consecutive_edge_lengths(poly, min_length=0.0):
    """Edge lengths between consecutive vertices of a polygon ring.

    Parameters
    ----------
    poly : (N, 2) ndarray
        Ordered polygon vertices.  The ring is closed automatically.
    min_length : float
        Edges shorter than this are discarded (avoids near-zero denominators).

    Returns
    -------
    list[float]
    """
    n = len(poly)
    if n < 2:
        return []
    lengths = []
    for i in range(n):
        d = np.linalg.norm(poly[(i + 1) % n] - poly[i])
        if d > min_length:
            lengths.append(d)
    return lengths


def _histogram_peak_detection(candidates, n_bins=200):
    """Adaptive histogram + scipy peak detection on a 1-D array of scale candidates.

    Returns
    -------
    (best_scale, ranked_peaks)
        best_scale : float  — scale value at the tallest peak
        ranked_peaks : list[(scale, votes)]  — top-5 peaks sorted by vote count
    """
    candidates = np.asarray(candidates, dtype=float)
    if len(candidates) == 0:
        return (1.0, [])

    lo, hi = candidates.min(), candidates.max()
    if hi - lo < 1e-12:
        return (float(np.median(candidates)), [(float(np.median(candidates)), len(candidates))])

    bins = np.linspace(lo, hi, n_bins + 1)
    hist, bin_edges = np.histogram(candidates, bins=bins)

    peaks, properties = find_peaks(hist, distance=3, height=max(np.mean(hist), 1))
    if len(peaks) == 0:
        # Fallback: just take the tallest bin
        idx = int(np.argmax(hist))
        best = float((bin_edges[idx] + bin_edges[idx + 1]) / 2)
        return (best, [(best, int(hist[idx]))])

    peak_heights = properties['peak_heights']
    top_indices = peaks[np.argsort(peak_heights)[::-1]][:5]

    ranked = []
    for idx in top_indices:
        s = float((bin_edges[idx] + bin_edges[idx + 1]) / 2)
        ranked.append((s, int(hist[idx])))

    best_scale = ranked[0][0]
    return (best_scale, ranked)


# ==========================================
# METHOD A — Edge-Only Distances
# ==========================================

def method_a_edge_distances(map_polys, pano_polys, min_edge=0.3):
    """Scale from ratios of consecutive edge lengths.

    For every pair of (map_edge, pano_edge) the ratio map_edge / pano_edge
    is a candidate scale.  Histogram peak detection picks the consensus.
    """
    map_edges = []
    for poly in map_polys:
        map_edges.extend(_consecutive_edge_lengths(np.asarray(poly), min_length=min_edge))

    pano_edges = []
    for poly in pano_polys:
        pano_edges.extend(_consecutive_edge_lengths(np.asarray(poly), min_length=min_edge))

    if not map_edges or not pano_edges:
        return (1.0, [])

    candidates = []
    for me in map_edges:
        for pe in pano_edges:
            candidates.append(me / pe)

    return _histogram_peak_detection(candidates)


# ==========================================
# METHOD B — Procrustes Shape Matching
# ==========================================

def method_b_procrustes(map_polys, pano_polys):
    """Scale from RMS-spread ratios of centred polygons.

    For every (map, pano) pair, centre both shapes and compute
    scale = RMS_spread_map / RMS_spread_pano.
    """
    def _rms_spread(poly):
        poly = np.asarray(poly, dtype=float)
        centroid = poly.mean(axis=0)
        dists = np.linalg.norm(poly - centroid, axis=1)
        return float(np.sqrt(np.mean(dists ** 2)))

    candidates = []
    for mp in map_polys:
        rms_m = _rms_spread(mp)
        if rms_m < 1e-12:
            continue
        for pp in pano_polys:
            rms_p = _rms_spread(pp)
            if rms_p < 1e-12:
                continue
            candidates.append(rms_m / rms_p)

    if not candidates:
        return (1.0, [])

    return _histogram_peak_detection(candidates)


# ==========================================
# METHOD C — Area Ratio
# ==========================================

def method_c_area_ratio(map_polys, pano_polys):
    """Scale from sqrt(area_map / area_pano) for every polygon pair."""
    candidates = []
    for mp in map_polys:
        a_m = _safe_area(mp)
        if a_m < 1e-12:
            continue
        for pp in pano_polys:
            a_p = _safe_area(pp)
            if a_p < 1e-12:
                continue
            candidates.append(np.sqrt(a_m / a_p))

    if not candidates:
        return (1.0, [])

    return _histogram_peak_detection(candidates)


# ==========================================
# SMOKE TEST
# ==========================================

if __name__ == "__main__":
    # A 10×5 rectangle and a 5×2.5 rectangle → expected scale ≈ 2.0
    map_rect = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=float)
    pano_rect = np.array([[0, 0], [5, 0], [5, 2.5], [0, 2.5]], dtype=float)

    map_polys = [map_rect]
    pano_polys = [pano_rect]

    print("=== Method A: Edge-Only Distances ===")
    best_a, peaks_a = method_a_edge_distances(map_polys, pano_polys, min_edge=0.0)
    print(f"  Best scale: {best_a:.4f}")
    for s, v in peaks_a:
        print(f"    scale={s:.4f}  votes={v}")

    print("\n=== Method B: Procrustes Shape Matching ===")
    best_b, peaks_b = method_b_procrustes(map_polys, pano_polys)
    print(f"  Best scale: {best_b:.4f}")
    for s, v in peaks_b:
        print(f"    scale={s:.4f}  votes={v}")

    print("\n=== Method C: Area Ratio ===")
    best_c, peaks_c = method_c_area_ratio(map_polys, pano_polys)
    print(f"  Best scale: {best_c:.4f}")
    for s, v in peaks_c:
        print(f"    scale={s:.4f}  votes={v}")

    # Verify all are close to 2.0
    tol = 0.15
    for name, val in [("A", best_a), ("B", best_b), ("C", best_c)]:
        assert abs(val - 2.0) < tol, f"Method {name} failed: got {val:.4f}, expected ~2.0"

    print("\n✓ All methods returned scale ≈ 2.0 — smoke test passed.")
