# Scale Detection V2 — Three-Method Comparison

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three scale detection methods for SAM3 pano-to-density polygon matching, and a comparison mode in demo6 to evaluate them side by side.

**Architecture:** A new `polygon_scale_calculation_v2.py` module exposes three independent scale detection functions (edge-distances, Procrustes, area-ratio) sharing the same signature. `align_polygons_demo6.py` gains a `--compare` flag that runs all three and produces a 3-panel comparison figure.

**Tech Stack:** numpy, scipy (find_peaks, svd), shapely, matplotlib

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/floorplan/polygon_scale_calculation_v2.py` | Create | Three scale detection methods + shared histogram utility |
| `src/floorplan/align_polygons_demo6.py` | Modify | Add `--compare` mode, refactor to use v2 scale methods |

---

### Task 1: Create `polygon_scale_calculation_v2.py` — shared utilities

**Files:**
- Create: `src/floorplan/polygon_scale_calculation_v2.py`

- [ ] **Step 1: Create the file with histogram utility and edge extraction**

```python
"""
Scale Detection V2 — Three methods for finding the scale ratio between
SAM3 pano polygons and SAM3 density-image polygons.

All methods share the same interface:
    method(map_polys, pano_polys) -> (best_scale, [(scale, votes), ...])

Usage:
    from polygon_scale_calculation_v2 import method_a_edge_distances
    scale, candidates = method_a_edge_distances(map_polys, pano_polys)
"""

import numpy as np
from scipy.signal import find_peaks
from shapely.geometry import Polygon as ShapelyPolygon


# ---------------------------------------------------------
# SHARED UTILITIES
# ---------------------------------------------------------
def _safe_area(coords):
    """Compute polygon area via Shapely."""
    poly = ShapelyPolygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly.area


def _consecutive_edge_lengths(poly, min_length=0.0):
    """Extract lengths of consecutive edges (wall segments) from a polygon.

    Args:
        poly: (N, 2) array of polygon vertices
        min_length: minimum edge length to keep

    Returns:
        1D array of edge lengths
    """
    edges = np.diff(np.vstack([poly, poly[0]]), axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    return lengths[lengths > min_length]


def _histogram_peak_detection(candidates, n_bins=200):
    """Find consensus scale from a list of candidate scale values.

    Uses adaptive range based on data percentiles and scipy peak detection.

    Returns:
        (best_scale, [(scale, votes), ...]) — up to 5 top peaks
    """
    if len(candidates) == 0:
        return 1.0, [(1.0, 0)]

    candidates = np.array(candidates)

    # Adaptive range from percentiles
    lo = np.percentile(candidates, 5)
    hi = np.percentile(candidates, 95)
    margin = (hi - lo) * 0.2
    lo = max(0.01, lo - margin)
    hi = hi + margin

    bin_width = (hi - lo) / n_bins
    if bin_width <= 0:
        median = float(np.median(candidates))
        return median, [(median, len(candidates))]

    bins = np.arange(lo, hi, bin_width)
    hist, bin_edges = np.histogram(candidates, bins=bins)

    peaks, properties = find_peaks(hist, distance=5, height=max(1, np.mean(hist)))
    if len(peaks) == 0:
        best_idx = np.argmax(hist)
        best_scale = (bin_edges[best_idx] + bin_edges[best_idx + 1]) / 2
        return best_scale, [(best_scale, int(hist[best_idx]))]

    peak_heights = properties["peak_heights"]
    top_indices = peaks[np.argsort(peak_heights)[-5:][::-1]]

    results = []
    for idx in top_indices:
        scale_val = (bin_edges[idx] + bin_edges[idx + 1]) / 2
        results.append((float(scale_val), int(hist[idx])))

    return results[0][0], results
```

- [ ] **Step 2: Verify file parses without errors**

Run: `python -c "import sys; sys.path.insert(0,'src/floorplan'); import polygon_scale_calculation_v2; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add src/floorplan/polygon_scale_calculation_v2.py
git commit -m "feat: add polygon_scale_calculation_v2 with shared utilities"
```

---

### Task 2: Method A — Edge-Only Distances

**Files:**
- Modify: `src/floorplan/polygon_scale_calculation_v2.py`

- [ ] **Step 1: Add method_a_edge_distances**

Append to `polygon_scale_calculation_v2.py`:

```python
# ---------------------------------------------------------
# METHOD A: Edge-Only Distances
# ---------------------------------------------------------
def method_a_edge_distances(map_polys, pano_polys, min_edge=0.3):
    """Scale detection via consecutive edge length ratios.

    Computes wall segment lengths from both polygon sets,
    forms all ratios map_edge / pano_edge, and finds the
    histogram consensus peak.

    Args:
        map_polys: list of (N, 2) arrays — density image room polygons
        pano_polys: list of (M, 2) arrays — pano footprint polygons
        min_edge: minimum edge length to include (filters noise)

    Returns:
        (best_scale, [(scale, votes), ...])
    """
    map_edges = []
    for poly in map_polys:
        map_edges.extend(_consecutive_edge_lengths(poly, min_length=min_edge))
    map_edges = np.array(map_edges)

    pano_edges = []
    for poly in pano_polys:
        pano_edges.extend(_consecutive_edge_lengths(poly, min_length=min_edge))
    pano_edges = np.array(pano_edges)

    if len(map_edges) == 0 or len(pano_edges) == 0:
        return 1.0, [(1.0, 0)]

    candidates = []
    for d_pano in pano_edges:
        ratios = map_edges / d_pano
        candidates.extend(ratios)

    return _histogram_peak_detection(candidates)
```

- [ ] **Step 2: Quick smoke test**

Run: `python -c "
import numpy as np, sys; sys.path.insert(0,'src/floorplan')
from polygon_scale_calculation_v2 import method_a_edge_distances
m = [np.array([[0,0],[10,0],[10,5],[0,5]])]
p = [np.array([[0,0],[5,0],[5,2.5],[0,2.5]])]
s, c = method_a_edge_distances(m, p)
print(f'scale={s:.2f}, expected ~2.0')
"`

Expected: scale ≈ 2.0

- [ ] **Step 3: Commit**

```bash
git add src/floorplan/polygon_scale_calculation_v2.py
git commit -m "feat: add Method A (edge-only distances) to scale v2"
```

---

### Task 3: Method B — Procrustes Shape Matching

**Files:**
- Modify: `src/floorplan/polygon_scale_calculation_v2.py`

- [ ] **Step 1: Add method_b_procrustes**

Append to `polygon_scale_calculation_v2.py`:

```python
# ---------------------------------------------------------
# METHOD B: Procrustes Shape Matching
# ---------------------------------------------------------
def _procrustes_scale(poly_a, poly_b):
    """Compute the scale factor that best maps poly_b onto poly_a.

    Centers both polygons, computes RMS spread from centroid,
    returns scale = spread_a / spread_b.
    """
    ca = poly_a - poly_a.mean(axis=0)
    cb = poly_b - poly_b.mean(axis=0)

    spread_a = np.sqrt(np.mean(np.sum(ca ** 2, axis=1)))
    spread_b = np.sqrt(np.mean(np.sum(cb ** 2, axis=1)))

    if spread_b < 1e-8:
        return None
    return spread_a / spread_b


def method_b_procrustes(map_polys, pano_polys):
    """Scale detection via Procrustes spread ratios.

    For every (pano, room) pair, computes the scale as the ratio
    of RMS spreads from centroid. Finds consensus via histogram.

    Args:
        map_polys: list of (N, 2) arrays
        pano_polys: list of (M, 2) arrays

    Returns:
        (best_scale, [(scale, votes), ...])
    """
    candidates = []
    for pano_poly in pano_polys:
        for map_poly in map_polys:
            scale = _procrustes_scale(map_poly, pano_poly)
            if scale is not None:
                candidates.append(scale)

    if not candidates:
        return 1.0, [(1.0, 0)]

    return _histogram_peak_detection(candidates)
```

- [ ] **Step 2: Quick smoke test**

Run: `python -c "
import numpy as np, sys; sys.path.insert(0,'src/floorplan')
from polygon_scale_calculation_v2 import method_b_procrustes
m = [np.array([[0,0],[10,0],[10,5],[0,5]])]
p = [np.array([[0,0],[5,0],[5,2.5],[0,2.5]])]
s, c = method_b_procrustes(m, p)
print(f'scale={s:.2f}, expected ~2.0')
"`

Expected: scale ≈ 2.0

- [ ] **Step 3: Commit**

```bash
git add src/floorplan/polygon_scale_calculation_v2.py
git commit -m "feat: add Method B (Procrustes) to scale v2"
```

---

### Task 4: Method C — Area Ratio

**Files:**
- Modify: `src/floorplan/polygon_scale_calculation_v2.py`

- [ ] **Step 1: Add method_c_area_ratio**

Append to `polygon_scale_calculation_v2.py`:

```python
# ---------------------------------------------------------
# METHOD C: Area Ratio
# ---------------------------------------------------------
def method_c_area_ratio(map_polys, pano_polys):
    """Scale detection via sqrt(room_area / pano_area) ratios.

    For every (pano, room) pair, computes the linear scale from
    the area ratio. Finds consensus via histogram.

    Args:
        map_polys: list of (N, 2) arrays
        pano_polys: list of (M, 2) arrays

    Returns:
        (best_scale, [(scale, votes), ...])
    """
    candidates = []
    for pano_poly in pano_polys:
        pano_area = _safe_area(pano_poly)
        if pano_area < 1e-8:
            continue
        for map_poly in map_polys:
            map_area = _safe_area(map_poly)
            if map_area < 1e-8:
                continue
            scale = np.sqrt(map_area / pano_area)
            candidates.append(scale)

    if not candidates:
        return 1.0, [(1.0, 0)]

    return _histogram_peak_detection(candidates)
```

- [ ] **Step 2: Quick smoke test**

Run: `python -c "
import numpy as np, sys; sys.path.insert(0,'src/floorplan')
from polygon_scale_calculation_v2 import method_c_area_ratio
m = [np.array([[0,0],[10,0],[10,5],[0,5]])]
p = [np.array([[0,0],[5,0],[5,2.5],[0,2.5]])]
s, c = method_c_area_ratio(m, p)
print(f'scale={s:.2f}, expected ~2.0')
"`

Expected: scale ≈ 2.0

- [ ] **Step 3: Commit**

```bash
git add src/floorplan/polygon_scale_calculation_v2.py
git commit -m "feat: add Method C (area ratio) to scale v2"
```

---

### Task 5: Add `--compare` mode to `align_polygons_demo6.py`

**Files:**
- Modify: `src/floorplan/align_polygons_demo6.py`

- [ ] **Step 1: Refactor demo6 to use v2 scale methods**

Replace the `find_consensus_scale` function and related imports in `align_polygons_demo6.py`. Remove the old inline scale detection code. Add import:

```python
import sys
sys.path.insert(0, str(_SRC_DIR / "floorplan"))
from polygon_scale_calculation_v2 import (
    method_a_edge_distances,
    method_b_procrustes,
    method_c_area_ratio,
)
```

Remove the old `find_consensus_scale`, `get_all_pairwise_distances` functions, and `from scipy.signal import find_peaks`.

- [ ] **Step 2: Add a `run_matching` helper**

Extract the matching + visualization logic into a reusable function so it can be called once per method:

```python
def run_matching(map_polys, map_labels, panos, scale, method_name):
    """Run matching at a given scale. Returns matches list."""
    matches = []
    for pano in panos:
        poly_scaled = pano["poly"] * scale
        best = {"score": -1.0, "room_idx": -1}
        for room_idx, room_poly in enumerate(map_polys):
            result = find_best_placement(poly_scaled, room_poly)
            if result["score"] > best["score"]:
                best = {**result, "room_idx": room_idx}
        matches.append({"name": pano["name"], **best})
    return matches
```

- [ ] **Step 3: Add comparison visualization function**

```python
def plot_comparison(map_polys, map_labels, all_results, output_dir, map_name):
    """Generate 1x3 panel figure comparing all three methods."""
    fig, axes = plt.subplots(1, 3, figsize=(30, 14))

    cmap = plt.cm.tab10

    for ax_idx, (method_name, scale, matches) in enumerate(all_results):
        ax = axes[ax_idx]
        ax.set_title(f"{method_name}\nscale={scale:.4f}", fontsize=12)

        # Map rooms
        for i, poly in enumerate(map_polys):
            closed = np.vstack([poly, poly[0]])
            ax.plot(closed[:, 0], closed[:, 1], "k-", linewidth=2.5)
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.08, color="gray")
            centroid = np.mean(poly, axis=0)
            ax.text(centroid[0], centroid[1], map_labels[i], fontsize=8,
                    ha="center", va="center", color="gray", fontweight="bold")

        # Fitted pieces
        for mi, match in enumerate(matches):
            color = cmap(mi % 10)
            poly = match["transformed"]
            closed = np.vstack([poly, poly[0]])
            ax.plot(closed[:, 0], closed[:, 1], "--", color=color, linewidth=2,
                    label=f"{match['name']} (s={match['score']:.2f})")
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.15, color=color)

            T = match["translation"]
            ax.plot(T[0], T[1], "o", color=color, markersize=8)

        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.invert_yaxis()

    plt.suptitle(f"Scale Method Comparison — {map_name}", fontsize=16)
    plt.tight_layout()
    fig_path = output_dir / "demo6_scale_comparison.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    return fig_path
```

- [ ] **Step 4: Update main() to support --compare flag**

Detect `--compare` in argv. If present:
1. Run all three methods to get scales
2. Run matching for each scale via `run_matching`
3. Call `plot_comparison` for the 3-panel figure
4. Save `demo6_scale_methods.json` with all three results
5. Print summary table

If `--compare` is NOT present, default to Method A and run the normal single-method flow.

- [ ] **Step 5: Test the comparison mode**

Run: `conda run -n scan_env python src/floorplan/align_polygons_demo6.py --compare tmb_office_corridor_bigger TMB_corridor_south1 TMB_corridor_south2 TMB_office1`

Expected: Prints all three scales, generates `demo6_scale_comparison.png` with 3 panels.

- [ ] **Step 6: Commit**

```bash
git add src/floorplan/align_polygons_demo6.py
git commit -m "feat: add --compare mode with three scale methods to demo6"
```

---

### Task 6: Run comparison and evaluate

- [ ] **Step 1: Run the full comparison**

Run: `conda run -n scan_env python src/floorplan/align_polygons_demo6.py --compare tmb_office_corridor_bigger TMB_corridor_south1 TMB_corridor_south2 TMB_office1`

- [ ] **Step 2: Show comparison figure to user for evaluation**

Read and display `demo6_scale_comparison.png`. Report which method gives the best visual fit.
