"""
Polygon Alignment Demo 6 — Jigsaw Puzzle Matching
==================================================
Fits SAM3 panoramic footprints (pieces) into SAM3 density-image room
polygons (slots).

Pipeline:
1. Scale detection via v2 methods (edge distances, Procrustes, area ratio).
2. Scale the pano polygons to match the map coordinate system.
3. Grid search over (rotation, translation) to maximize containment.

Key design choices:
- Multiple pieces CAN map to the same room (overlapping corridor scans)
- Automatic scale discovery via histogram consensus (no hardcoded scale)
- 4 discrete rotations × dense translation grid search
- Fit score = intersection_area / piece_area × size_compatibility
- No Hungarian matching — each piece independently finds its best slot

Usage:
    python src/floorplan/align_polygons_demo6.py <map_name> [pano1 pano2 ...]
    python src/floorplan/align_polygons_demo6.py --compare <map_name> [pano1 pano2 ...]

Examples:
    conda run -n scan_env python src/floorplan/align_polygons_demo6.py tmb_office_corridor_bigger TMB_corridor_south1 TMB_corridor_south2 TMB_office1
    conda run -n scan_env python src/floorplan/align_polygons_demo6.py --compare tmb_office_corridor_bigger TMB_corridor_south1 TMB_corridor_south2 TMB_office1
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon as ShapelyPolygon

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
_SCRIPT_FILE = Path(__file__).resolve()
_SRC_DIR = _SCRIPT_FILE.parent.parent
_PROJECT_ROOT = _SRC_DIR.parent

sys.path.insert(0, str(_SRC_DIR / "floorplan"))
from polygon_scale_calculation_v2 import (
    method_a_edge_distances,
    method_b_procrustes,
    method_c_area_ratio,
)

# SAM3 panoramic footprints (meters, camera at origin in XZ plane)
SAM3_PANO_BASE = _PROJECT_ROOT / "data" / "sam3_pano_processing"

# ---------------------------------------------------------
# MATCHING PARAMETERS
# ---------------------------------------------------------
ROTATIONS_DEG = [0, 90, 180, 270]
GRID_STEPS = 30       # translation grid resolution per axis


# ---------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------
def rotation_matrix_2d(angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def safe_shapely(coords):
    """Create a valid Shapely polygon from (N,2) coordinates."""
    poly = ShapelyPolygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def fit_score(piece_coords, room_coords):
    """Jigsaw fit score: containment × size compatibility."""
    piece = safe_shapely(piece_coords)
    room = safe_shapely(room_coords)
    if piece.area < 1e-8:
        return 0.0
    try:
        containment = piece.intersection(room).area / piece.area
        size_compat = min(1.0, room.area / piece.area)
        return containment * size_compat
    except Exception:
        return 0.0


# ---------------------------------------------------------
# JIGSAW MATCHING
# ---------------------------------------------------------
def find_best_placement(piece_poly, room_poly):
    """Grid search over (rotation, translation) to maximize containment.

    Returns dict: score, rot_deg, translation, transformed
    """
    room_min = np.min(room_poly, axis=0)
    room_max = np.max(room_poly, axis=0)

    tx_grid = np.linspace(room_min[0], room_max[0], GRID_STEPS)
    ty_grid = np.linspace(room_min[1], room_max[1], GRID_STEPS)

    best = {"score": -1.0}

    for rot in ROTATIONS_DEG:
        R = rotation_matrix_2d(rot)
        rotated = piece_poly @ R.T

        for tx in tx_grid:
            for ty in ty_grid:
                T = np.array([tx, ty])
                translated = rotated + T
                score = fit_score(translated, room_poly)

                if score > best["score"]:
                    best = {
                        "score": score,
                        "rot_deg": rot,
                        "translation": T.copy(),
                        "transformed": translated.copy(),
                    }

    return best


# ---------------------------------------------------------
# MATCHING HELPER
# ---------------------------------------------------------
def run_matching(map_polys, map_labels, panos, scale):
    """Run jigsaw matching for all panos at a given scale.

    Parameters
    ----------
    map_polys : list of (N,2) ndarray
    map_labels : list of str
    panos : list of dict with keys 'name', 'poly'
    scale : float

    Returns
    -------
    list of dict — one per pano with keys: name, score, rot_deg,
        translation, transformed, room_idx
    """
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


# ---------------------------------------------------------
# COMPARISON VISUALIZATION
# ---------------------------------------------------------
def plot_comparison(map_polys, map_labels, all_results, output_dir, map_name):
    """Generate 1x3 panel figure comparing all three methods.

    Parameters
    ----------
    map_polys : list of (N,2) ndarray
    map_labels : list of str
    all_results : list of (method_name, scale, matches)
    output_dir : Path
    map_name : str
    """
    n_methods = len(all_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(8 * n_methods, 14))
    if n_methods == 1:
        axes = [axes]

    cmap = plt.cm.tab10

    for ax_idx, (method_name, scale, matches) in enumerate(all_results):
        ax = axes[ax_idx]
        ax.set_title(f"{method_name}\nscale={scale:.4f}", fontsize=13, fontweight="bold")

        # Map rooms (black outlines)
        for i, poly in enumerate(map_polys):
            closed = np.vstack([poly, poly[0]])
            ax.plot(closed[:, 0], closed[:, 1], "k-", linewidth=2.5)
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.08, color="gray")
            centroid = np.mean(poly, axis=0)
            ax.text(centroid[0], centroid[1], map_labels[i], fontsize=7,
                    ha="center", va="center", color="gray", fontweight="bold")

        # Fitted pano pieces (colored dashed)
        for mi, match in enumerate(matches):
            color = cmap(mi % 10)
            poly = match["transformed"]
            closed = np.vstack([poly, poly[0]])
            ax.plot(closed[:, 0], closed[:, 1], "--", color=color, linewidth=2,
                    label=f"{match['name']} ({match['score']:.2f})")
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.12, color=color)

        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.invert_yaxis()

    fig.suptitle(f"Scale Method Comparison — {map_name}", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = output_dir / "demo6_scale_comparison.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison figure: {fig_path}")
    return fig_path


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    argv = list(sys.argv[1:])

    # Detect --compare flag
    compare_mode = "--compare" in argv
    if compare_mode:
        argv.remove("--compare")

    if len(argv) < 2:
        print("Usage: python src/floorplan/align_polygons_demo6.py [--compare] <map_name> [pano1 pano2 ...]")
        print("\nExamples:")
        print("  conda run -n scan_env python src/floorplan/align_polygons_demo6.py "
              "tmb_office_corridor_bigger TMB_corridor_south1 TMB_corridor_south2 TMB_office1")
        print("  conda run -n scan_env python src/floorplan/align_polygons_demo6.py "
              "--compare tmb_office_corridor_bigger TMB_corridor_south1 TMB_corridor_south2 TMB_office1")
        sys.exit(1)

    map_name = argv[0]
    pano_names = argv[1:]

    # Resolve paths
    sam3_map_json = (
        _PROJECT_ROOT / "data" / "sam3_room_segmentation"
        / map_name / f"{map_name}_polygons.json"
    )
    output_dir = _PROJECT_ROOT / "data" / "sam3_room_segmentation" / map_name

    print("=" * 60)
    print("Polygon Alignment Demo 6 — Jigsaw Puzzle Matching")
    if compare_mode:
        print("  MODE: --compare (all three scale methods)")
    print("=" * 60)

    # Load map room polygons
    print(f"\nLoading map: {sam3_map_json.name}")
    with open(sam3_map_json) as f:
        map_data = json.load(f)
    map_polys = []
    map_labels = []
    for r in map_data["rooms"]:
        poly = np.array(r["vertices_world_meters"])
        label = r.get("label", f"room_{r['mask_index']:02d}")
        area = safe_shapely(poly).area
        map_polys.append(poly)
        map_labels.append(label)
        print(f"  {label}: {poly.shape[0]} vertices, area={area:.2f} m²")

    # Load pano footprints
    print(f"\nLoading {len(pano_names)} pano footprints:")
    panos = []
    for name in pano_names:
        layout_path = SAM3_PANO_BASE / name / "layout.json"
        with open(layout_path) as f:
            data = json.load(f)
        poly = np.array(data["layout_corners"])
        area = safe_shapely(poly).area
        panos.append({"name": name, "poly": poly})
        print(f"  {name}: {poly.shape[0]} vertices, area={area:.2f} m²")

    pano_polys_raw = [p["poly"] for p in panos]

    if compare_mode:
        # ---- Compare mode: run all three methods ----
        methods = [
            ("Method A: Edge Distances", method_a_edge_distances),
            ("Method B: Procrustes", method_b_procrustes),
            ("Method C: Area Ratio", method_c_area_ratio),
        ]

        # Compute scales
        print(f"\n{'='*60}")
        print("Scale Detection — All Methods")
        print(f"{'='*60}")
        scale_results = []
        for method_name, method_fn in methods:
            scale, candidates = method_fn(map_polys, pano_polys_raw)
            scale_results.append((method_name, scale, candidates))

        # Print summary table
        print(f"\n  {'Method':<30s} {'Scale':>10s}  {'Top votes':>10s}")
        print(f"  {'-'*30} {'-'*10}  {'-'*10}")
        for method_name, scale, candidates in scale_results:
            top_votes = candidates[0][1] if candidates else 0
            print(f"  {method_name:<30s} {scale:>10.4f}  {top_votes:>10d}")

        # Run matching for each method
        print(f"\nMatching (grid={GRID_STEPS}x{GRID_STEPS}, "
              f"rotations={ROTATIONS_DEG})...")
        all_results = []
        for method_name, scale, candidates in scale_results:
            print(f"\n  --- {method_name} (scale={scale:.4f}) ---")
            matches = run_matching(map_polys, map_labels, panos, scale)
            for m in matches:
                print(f"    {m['name']} -> {map_labels[m['room_idx']]} "
                      f"rot={m['rot_deg']}deg score={m['score']:.3f}")
            all_results.append((method_name, scale, matches))

        # Comparison visualization
        plot_comparison(map_polys, map_labels, all_results, output_dir, map_name)

        # Save JSON summary
        json_data = {
            "metadata": {
                "pipeline": "SAM3-to-SAM3 jigsaw (demo6) — scale comparison",
                "map_name": map_name,
                "pano_names": pano_names,
                "grid_steps": GRID_STEPS,
            },
            "methods": [],
        }
        for method_name, scale, matches in all_results:
            method_entry = {
                "method": method_name,
                "scale": float(scale),
                "matches": [
                    {
                        "pano_name": m["name"],
                        "room_idx": m["room_idx"],
                        "room_label": map_labels[m["room_idx"]],
                        "score": float(m["score"]),
                        "rotation_deg": int(m["rot_deg"]),
                        "camera_position": m["translation"].tolist(),
                    }
                    for m in matches
                ],
            }
            json_data["methods"].append(method_entry)

        json_path = output_dir / "demo6_scale_methods.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        print(f"Results: {json_path}")

    else:
        # ---- Normal mode: default to Method A ----
        print(f"\nScale detection (Method A — Edge Distances):")
        scale, scale_candidates = method_a_edge_distances(map_polys, pano_polys_raw)
        print(f"\n  Using scale: {scale:.4f}")
        if scale_candidates:
            print(f"  Top scale candidates:")
            for i, (s, v) in enumerate(scale_candidates):
                print(f"    Rank {i + 1}: {s:.4f} (votes: {v})")

        # Apply scale to pano polygons
        for pano in panos:
            pano["poly_scaled"] = pano["poly"] * scale
            area = safe_shapely(pano["poly_scaled"]).area
            print(f"  {pano['name']} scaled area: {area:.2f} m²")

        # --- Match each pano independently (allows many-to-one) ---
        print(f"\nMatching (grid={GRID_STEPS}x{GRID_STEPS}, "
              f"rotations={ROTATIONS_DEG})...")

        matches = run_matching(map_polys, map_labels, panos, scale)
        for m in matches:
            print(f"  {m['name']} -> {map_labels[m['room_idx']]} "
                  f"rot={m['rot_deg']}deg score={m['score']:.3f} "
                  f"T=[{m['translation'][0]:.2f}, {m['translation'][1]:.2f}]")

        # --- Visualization ---
        fig, ax = plt.subplots(figsize=(10, 14))
        ax.set_title(f"SAM3 Jigsaw Matching — {map_name} (scale={scale:.3f})", fontsize=14)

        # Map rooms (thick black outline + light gray fill)
        for i, poly in enumerate(map_polys):
            closed = np.vstack([poly, poly[0]])
            ax.plot(closed[:, 0], closed[:, 1], "k-", linewidth=2.5,
                    label=map_labels[i])
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.08, color="gray")
            centroid = np.mean(poly, axis=0)
            ax.text(centroid[0], centroid[1], map_labels[i], fontsize=8,
                    ha="center", va="center", color="gray", fontweight="bold")

        # Fitted pieces (colored dashed outline + light fill)
        cmap = plt.cm.tab10
        for mi, match in enumerate(matches):
            color = cmap(mi % 10)
            poly = match["transformed"]
            closed = np.vstack([poly, poly[0]])
            ax.plot(closed[:, 0], closed[:, 1], "--", color=color, linewidth=2,
                    label=f"{match['name']} (score={match['score']:.2f})")
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.15, color=color)

            # Camera position marker
            T = match["translation"]
            ax.plot(T[0], T[1], "o", color=color, markersize=8)
            center = np.mean(poly, axis=0)
            ax.annotate(match["name"], xy=(T[0], T[1]),
                        xytext=(center[0], center[1]),
                        fontsize=8, color=color, ha="center",
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.invert_yaxis()

        fig_path = output_dir / "demo6_alignment.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nVisualization: {fig_path}")

        # --- Save JSON ---
        results = {
            "metadata": {
                "pipeline": "SAM3-to-SAM3 jigsaw (demo6)",
                "map_name": map_name,
                "pano_names": pano_names,
                "grid_steps": GRID_STEPS,
                "scale": float(scale),
                "scale_method": "Method A: Edge Distances",
                "scale_candidates": [
                    {"scale": float(s), "votes": int(v)} for s, v in scale_candidates
                ],
            },
            "matches": [
                {
                    "pano_name": m["name"],
                    "room_idx": m["room_idx"],
                    "room_label": map_labels[m["room_idx"]],
                    "score": float(m["score"]),
                    "rotation_deg": int(m["rot_deg"]),
                    "camera_position": m["translation"].tolist(),
                }
                for m in matches
            ],
        }

        json_path = output_dir / "demo6_alignment.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results: {json_path}")


if __name__ == "__main__":
    main()
