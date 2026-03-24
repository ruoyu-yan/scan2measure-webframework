"""
Polygon Alignment Demo 6 — Jigsaw Puzzle Matching
==================================================
Fits SAM3 panoramic footprints (pieces) into SAM3 density-image room
polygons (slots).

Default mode — Enumerate + Optimize:
1. Enumerate all possible pano→room assignments (M^N combos).
2. For each assignment, optimize (shared_scale, per-pano rotation,
   per-pano translation) using differential_evolution with true IoU.
3. Pick the assignment with highest total IoU.
4. Refine placement: OBB long-axis alignment, dense translation grid,
   non-overlap penalty, largest pano placed first.

Legacy modes (--compare, --anchor) kept for backwards compatibility.

Usage:
    python src/floorplan/align_polygons_demo6.py <map_name> [pano1 pano2 ...]
    python src/floorplan/align_polygons_demo6.py --compare <map_name> [pano1 ...]
    python src/floorplan/align_polygons_demo6.py --anchor <map_name> [pano1 ...]

Examples:
    conda run -n scan_env python src/floorplan/align_polygons_demo6.py tmb_office_corridor_bigger TMB_corridor_south1 TMB_corridor_south2 TMB_office1
"""

import json
import sys
import time
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid
from scipy.optimize import differential_evolution

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
# PARAMETERS
# ---------------------------------------------------------
# Legacy modes
ROTATIONS_DEG = [0, 90, 180, 270]
GRID_STEPS = 30

# Enumerate+Optimize mode
SCALE_BOUNDS = (0.2, 2.5)
ROTATION_BOUNDS = (0.0, 2.0 * np.pi)
TRANSLATION_MARGIN = 5.0
DE_MAXITER = 300
DE_POPSIZE = 25


# ---------------------------------------------------------
# GEOMETRY — shared
# ---------------------------------------------------------
def safe_shapely(coords):
    """Create a valid Shapely polygon from (N,2) coordinates."""
    poly = ShapelyPolygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def rotation_matrix_2d(angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def compute_aspect_ratio(poly):
    """Aspect ratio from oriented bounding box. Returns AR >= 1."""
    pts = poly.astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    w, h = rect[1]
    if w < 1e-8 or h < 1e-8:
        return 1.0
    return max(w, h) / min(w, h)


# ---------------------------------------------------------
# ENUMERATE + OPTIMIZE — core functions
# ---------------------------------------------------------
def transform_polygon(vertices, scale, angle_rad, tx, ty):
    """Apply scale, rotation (radians), and translation."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (vertices * scale) @ R.T + np.array([tx, ty])


def compute_true_iou(verts_a, verts_b):
    """IoU = intersection / union. Prevents scale collapse."""
    try:
        pa = ShapelyPolygon(verts_a)
        pb = ShapelyPolygon(verts_b)
        if not pa.is_valid:
            pa = make_valid(pa)
        if not pb.is_valid:
            pb = make_valid(pb)
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return inter / union if union > 1e-6 else 0.0
    except Exception:
        return 0.0


def compute_containment(verts_piece, verts_room):
    """Containment = intersection / piece_area."""
    try:
        piece = ShapelyPolygon(verts_piece)
        room = ShapelyPolygon(verts_room)
        if not piece.is_valid:
            piece = make_valid(piece)
        if not room.is_valid:
            room = make_valid(room)
        pa = piece.area
        return piece.intersection(room).area / pa if pa > 1e-6 else 0.0
    except Exception:
        return 0.0


def enumerate_assignments(pano_names, room_labels):
    """All possible pano→room assignments (M^N combos)."""
    return [
        dict(zip(pano_names, combo))
        for combo in itertools.product(room_labels, repeat=len(pano_names))
    ]


def optimize_assignment(assignment, pano_names, pano_polys, room_polys):
    """Find optimal (scale, per-pano rotation+translation) via diff. evolution.

    Parameters
    ----------
    assignment : dict  pano_name -> room_label
    pano_names : list[str]
    pano_polys : dict  name -> (N,2) ndarray
    room_polys : dict  label -> (N,2) ndarray

    Returns
    -------
    dict with keys: assignment, scale, total_score, per_pano
    """
    bounds = [SCALE_BOUNDS]
    for name in pano_names:
        rv = room_polys[assignment[name]]
        rmin, rmax = np.min(rv, axis=0), np.max(rv, axis=0)
        bounds.append(ROTATION_BOUNDS)
        bounds.append((rmin[0] - TRANSLATION_MARGIN,
                        rmax[0] + TRANSLATION_MARGIN))
        bounds.append((rmin[1] - TRANSLATION_MARGIN,
                        rmax[1] + TRANSLATION_MARGIN))

    def objective(params):
        s = params[0]
        total = 0.0
        for i, name in enumerate(pano_names):
            a, tx, ty = params[1+i*3], params[2+i*3], params[3+i*3]
            xf = transform_polygon(pano_polys[name], s, a, tx, ty)
            total += compute_true_iou(xf, room_polys[assignment[name]])
        return -total

    res = differential_evolution(
        objective, bounds=bounds, seed=42,
        maxiter=DE_MAXITER, tol=1e-7, polish=True,
        workers=1, popsize=DE_POPSIZE,
    )

    scale = res.x[0]
    per_pano = {}
    for i, name in enumerate(pano_names):
        a, tx, ty = res.x[1+i*3], res.x[2+i*3], res.x[3+i*3]
        xf = transform_polygon(pano_polys[name], scale, a, tx, ty)
        per_pano[name] = {
            "room": assignment[name],
            "angle_rad": a,
            "angle_deg": float(np.degrees(a) % 360),
            "tx": float(tx), "ty": float(ty),
            "iou": compute_true_iou(xf, room_polys[assignment[name]]),
            "containment": compute_containment(xf, room_polys[assignment[name]]),
            "transformed": xf,
        }

    return {
        "assignment": assignment,
        "scale": float(scale),
        "total_score": float(-res.fun),
        "per_pano": per_pano,
    }


# ---------------------------------------------------------
# STAGE 2 — OBB-aligned refinement with non-overlap penalty
# ---------------------------------------------------------
def obb_long_axis_angle(verts):
    """Angle (radians) of the OBB long axis."""
    pts = np.array(verts, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    (_, _), (w, h), angle_deg = rect
    angle_rad = np.radians(angle_deg)
    if w < h:
        angle_rad += np.pi / 2
    return angle_rad % np.pi


def refine_all_placements(pano_names, pano_polys, room_polys,
                          assignment, scale):
    """OBB-aligned placement with non-overlap penalty, largest first.

    Returns dict: name -> {angle, tx, ty, iou, containment, transformed, ...}
    """
    room_groups = {}
    for name in pano_names:
        room_groups.setdefault(assignment[name], []).append(name)

    results = {}

    for room_label, names_in_room in room_groups.items():
        room_verts = room_polys[room_label]
        room_long = obb_long_axis_angle(room_verts)
        rmin, rmax = np.min(room_verts, axis=0), np.max(room_verts, axis=0)

        # Largest pano first — strongest shape constraint
        names_in_room = sorted(
            names_in_room,
            key=lambda n: ShapelyPolygon(pano_polys[n]).area,
            reverse=True,
        )

        already_placed = []

        for name in names_in_room:
            pv = pano_polys[name]
            pano_long = obb_long_axis_angle(pv)

            # Candidate rotations: align long axes + coarse sweep
            base = room_long - pano_long
            angles = []
            for off in [0, np.pi]:
                for d in range(-20, 21, 4):
                    angles.append(base + off + np.radians(d))
            for d in range(0, 360, 15):
                angles.append(np.radians(d))

            n_tx = min(50, max(20, int((rmax[0]-rmin[0]) * 3)))
            n_ty = min(70, max(20, int((rmax[1]-rmin[1]) * 3)))
            tx_vals = np.linspace(rmin[0], rmax[0], n_tx)
            ty_vals = np.linspace(rmin[1], rmax[1], n_ty)

            best = {"score": -1e9}

            for angle in angles:
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c, -s], [s, c]])
                sr = (pv * scale) @ R.T
                sr_c = sr - np.mean(sr, axis=0)

                for tx in tx_vals:
                    for ty in ty_vals:
                        xf = sr_c + np.array([tx, ty])
                        cont = compute_containment(xf, room_verts)
                        if cont < 0.3:
                            continue
                        iou = compute_true_iou(xf, room_verts)

                        overlap = 0.0
                        if already_placed:
                            try:
                                p = ShapelyPolygon(xf)
                                if not p.is_valid:
                                    p = make_valid(p)
                                pa = p.area
                                for prev in already_placed:
                                    overlap += p.intersection(prev).area / pa if pa > 1e-6 else 0
                            except Exception:
                                pass

                        score = iou + cont - 1.5 * overlap
                        if score > best["score"]:
                            # Camera was at origin in pano coords;
                            # after center+translate it lands here:
                            cam = np.array([tx, ty]) - np.mean(sr, axis=0)
                            best = {
                                "score": score, "iou": iou,
                                "containment": cont, "overlap": overlap,
                                "angle_rad": angle,
                                "angle_deg": float(np.degrees(angle) % 360),
                                "tx": float(tx), "ty": float(ty),
                                "camera_position": [float(cam[0]), float(cam[1])],
                                "transformed": xf.copy(),
                            }

            results[name] = best
            try:
                pp = ShapelyPolygon(best["transformed"])
                if not pp.is_valid:
                    pp = make_valid(pp)
                already_placed.append(pp)
            except Exception:
                pass

    return results


# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------
PANO_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
               "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]


def plot_alignment(ax, map_polys, map_labels, matches_dict, pano_names, title):
    """Draw rooms + fitted pano polygons on an axes."""
    for i, (label, verts) in enumerate(zip(map_labels, map_polys)):
        closed = np.vstack([verts, verts[0]])
        ax.plot(closed[:, 0], closed[:, 1], "k-", linewidth=2.5)
        ax.fill(verts[:, 0], verts[:, 1], alpha=0.08, color="gray")
        cx, cy = np.mean(verts, axis=0)
        ax.text(cx, cy, label, fontsize=8, ha="center", va="center",
                color="gray", fontweight="bold")

    for i, name in enumerate(pano_names):
        m = matches_dict[name]
        color = PANO_COLORS[i % len(PANO_COLORS)]
        poly = m["transformed"]
        closed = np.vstack([poly, poly[0]])
        short = name.replace("TMB_", "")
        ax.plot(closed[:, 0], closed[:, 1], "--", color=color, linewidth=2,
                label=f"{short} -> {m['room']} (IoU={m['iou']:.2f})")
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.15, color=color)
        cx, cy = np.mean(poly, axis=0)
        ax.text(cx, cy, short, fontsize=7, ha="center", va="center",
                color=color, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)


# ---------------------------------------------------------
# LEGACY — kept for --compare / --anchor
# ---------------------------------------------------------
def fit_score(piece_coords, room_coords):
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


def find_best_placement(piece_poly, room_poly):
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
                    best = {"score": score, "rot_deg": rot,
                            "translation": T.copy(), "transformed": translated.copy()}
    return best


def run_matching(map_polys, map_labels, panos, scale):
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
# MAIN
# ---------------------------------------------------------
def main():
    argv = list(sys.argv[1:])

    compare_mode = "--compare" in argv
    if compare_mode:
        argv.remove("--compare")
    anchor_mode = "--anchor" in argv
    if anchor_mode:
        argv.remove("--anchor")

    if len(argv) < 2:
        print("Usage: python src/floorplan/align_polygons_demo6.py "
              "[--compare|--anchor] <map_name> [pano1 pano2 ...]")
        print("\nExamples:")
        print("  conda run -n scan_env python src/floorplan/align_polygons_demo6.py "
              "tmb_office_corridor_bigger TMB_corridor_south1 TMB_corridor_south2 TMB_office1")
        sys.exit(1)

    map_name = argv[0]
    pano_names = argv[1:]

    sam3_map_json = (
        _PROJECT_ROOT / "data" / "sam3_room_segmentation"
        / map_name / f"{map_name}_polygons.json"
    )
    output_dir = _PROJECT_ROOT / "data" / "sam3_room_segmentation" / map_name

    print("=" * 60)
    print("Polygon Alignment Demo 6 — Jigsaw Puzzle Matching")
    if compare_mode:
        print("  MODE: --compare (legacy)")
    elif anchor_mode:
        print("  MODE: --anchor (legacy)")
    else:
        print("  MODE: enumerate + optimize (default)")
    print("=" * 60)

    # ---- Load map room polygons ----
    print(f"\nLoading map: {sam3_map_json.name}")
    with open(sam3_map_json) as f:
        map_data = json.load(f)
    map_polys = []
    map_labels = []
    map_polys_dict = {}
    for r in map_data["rooms"]:
        poly = np.array(r["vertices_world_meters"])
        label = r.get("label", f"room_{r['mask_index']:02d}")
        area = safe_shapely(poly).area
        map_polys.append(poly)
        map_labels.append(label)
        map_polys_dict[label] = poly
        print(f"  {label}: {poly.shape[0]} vertices, area={area:.2f} m²")

    # ---- Load pano footprints ----
    # layout_corners are [x, z_flipped] from SAM3 extraction.
    # z_flipped maps directly to map_y (both are "backward" direction).
    print(f"\nLoading {len(pano_names)} pano footprints:")
    panos = []
    pano_polys_dict = {}
    for name in pano_names:
        layout_path = SAM3_PANO_BASE / name / "layout.json"
        with open(layout_path) as f:
            data = json.load(f)
        poly = np.array(data["layout_corners"])
        area = safe_shapely(poly).area
        panos.append({"name": name, "poly": poly})
        pano_polys_dict[name] = poly
        print(f"  {name}: {poly.shape[0]} vertices, area={area:.2f} m²")

    pano_polys_raw = [p["poly"] for p in panos]

    # ==============================================================
    # DEFAULT MODE: Enumerate + Optimize
    # ==============================================================
    if not compare_mode and not anchor_mode:
        t0 = time.time()

        # ---- Stage 1: Enumerate all assignments, optimize each ----
        assignments = enumerate_assignments(pano_names, map_labels)
        n_assign = len(assignments)
        print(f"\n{'='*60}")
        print(f"Stage 1: Enumerate {n_assign} assignments + optimize")
        print(f"{'='*60}")

        all_results = []
        for i, assignment in enumerate(assignments):
            mapping = ", ".join(f"{k.replace('TMB_','')}->{v}"
                                for k, v in assignment.items())
            t_s = time.time()
            result = optimize_assignment(
                assignment, pano_names, pano_polys_dict, map_polys_dict
            )
            dt = time.time() - t_s
            all_results.append(result)
            per_iou = [result["per_pano"][n]["iou"] for n in pano_names]
            print(f"  [{i+1}/{n_assign}] {mapping}  "
                  f"score={result['total_score']:.3f}  "
                  f"scale={result['scale']:.3f}  "
                  f"IoU=[{', '.join(f'{v:.3f}' for v in per_iou)}]  "
                  f"({dt:.1f}s)")

        all_results.sort(key=lambda r: r["total_score"], reverse=True)

        best = all_results[0]
        margin = 0.0
        if len(all_results) >= 2:
            margin = (best["total_score"] - all_results[1]["total_score"])

        print(f"\n  Best: {best['assignment']}  "
              f"score={best['total_score']:.4f}  scale={best['scale']:.4f}")
        if margin > 0:
            pct = margin / best["total_score"] * 100
            print(f"  Margin over 2nd: {margin:.4f} ({pct:.1f}%)")

        # ---- Stage 2: Refine placement ----
        print(f"\n{'='*60}")
        print("Stage 2: OBB-aligned refinement (largest first, non-overlap)")
        print(f"{'='*60}")

        refined = refine_all_placements(
            pano_names, pano_polys_dict, map_polys_dict,
            best["assignment"], best["scale"],
        )

        refined_total = 0.0
        for name in pano_names:
            r = refined[name]
            r["room"] = best["assignment"][name]
            refined_total += r["iou"]
            print(f"  {name} -> {r['room']}:  "
                  f"rot={r['angle_deg']:.1f}deg  "
                  f"t=({r['tx']:.2f}, {r['ty']:.2f})  "
                  f"IoU={r['iou']:.3f}  cont={r['containment']:.3f}")

        total_time = time.time() - t0
        print(f"\n  Refined total IoU: {refined_total:.4f}")
        print(f"  Total time: {total_time:.1f}s")

        # ---- Visualization ----
        fig, ax = plt.subplots(figsize=(10, 14))
        plot_alignment(
            ax, map_polys, map_labels, refined, pano_names,
            f"SAM3 Jigsaw — {map_name}\n"
            f"scale={best['scale']:.3f}, total IoU={refined_total:.3f}",
        )
        fig_path = output_dir / "demo6_alignment.png"
        plt.tight_layout()
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nVisualization: {fig_path}")

        # ---- Save JSON ----
        results_json = {
            "metadata": {
                "pipeline": "SAM3-to-SAM3 jigsaw (demo6) — enumerate+optimize",
                "map_name": map_name,
                "pano_names": pano_names,
                "scale": best["scale"],
                "scale_method": "differential_evolution (true IoU)",
                "total_iou_stage1": best["total_score"],
                "total_iou_stage2": refined_total,
                "confidence_margin": float(margin),
                "n_assignments_tested": n_assign,
            },
            "matches": [
                {
                    "pano_name": name,
                    "room_idx": map_labels.index(refined[name]["room"]),
                    "room_label": refined[name]["room"],
                    "score": refined[name]["iou"],
                    "rotation_deg": refined[name]["angle_deg"],
                    "camera_position": refined[name]["camera_position"],
                }
                for name in pano_names
            ],
        }

        json_path = output_dir / "demo6_alignment.json"
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=4)
        print(f"Results: {json_path}")

    # ==============================================================
    # LEGACY: --compare
    # ==============================================================
    elif compare_mode:
        methods = [
            ("Method A: Edge Distances", method_a_edge_distances),
            ("Method B: Procrustes", method_b_procrustes),
            ("Method C: Area Ratio", method_c_area_ratio),
        ]
        print(f"\n{'='*60}")
        print("Scale Detection — All Methods (legacy)")
        print(f"{'='*60}")
        all_results = []
        for method_name, method_fn in methods:
            scale, candidates = method_fn(map_polys, pano_polys_raw)
            print(f"  {method_name}: scale={scale:.4f}")
            matches = run_matching(map_polys, map_labels, panos, scale)
            for m in matches:
                print(f"    {m['name']} -> {map_labels[m['room_idx']]} "
                      f"rot={m['rot_deg']}deg score={m['score']:.3f}")
            all_results.append((method_name, scale, matches))

        # Comparison plot
        n = len(all_results)
        fig, axes = plt.subplots(1, n, figsize=(8*n, 14))
        if n == 1:
            axes = [axes]
        cmap = plt.cm.tab10
        for ax_idx, (mn, sc, matches) in enumerate(all_results):
            ax = axes[ax_idx]
            ax.set_title(f"{mn}\nscale={sc:.4f}", fontsize=13, fontweight="bold")
            for i, poly in enumerate(map_polys):
                closed = np.vstack([poly, poly[0]])
                ax.plot(closed[:,0], closed[:,1], "k-", lw=2.5)
                ax.fill(poly[:,0], poly[:,1], alpha=0.08, color="gray")
            for mi, match in enumerate(matches):
                color = cmap(mi % 10)
                poly = match["transformed"]
                closed = np.vstack([poly, poly[0]])
                ax.plot(closed[:,0], closed[:,1], "--", color=color, lw=2,
                        label=f"{match['name']} ({match['score']:.2f})")
                ax.fill(poly[:,0], poly[:,1], alpha=0.12, color=color)
            ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
            ax.invert_yaxis()
        fig.suptitle(f"Scale Comparison — {map_name}", fontsize=15, fontweight="bold")
        plt.tight_layout()
        fig_path = output_dir / "demo6_scale_comparison.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nComparison: {fig_path}")

    # ==============================================================
    # LEGACY: --anchor
    # ==============================================================
    elif anchor_mode:
        print(f"\n{'='*60}")
        print("AR Anchor-Based Matching (legacy)")
        print(f"{'='*60}")
        room_ars = [compute_aspect_ratio(p) for p in map_polys]
        pano_ars = [compute_aspect_ratio(p["poly"]) for p in panos]
        room_areas = [safe_shapely(p).area for p in map_polys]
        pano_areas = [safe_shapely(p["poly"]).area for p in panos]

        for i, (l, ar) in enumerate(zip(map_labels, room_ars)):
            print(f"  {l}: AR={ar:.3f}, area={room_areas[i]:.2f}")
        for i, (p, ar) in enumerate(zip(panos, pano_ars)):
            print(f"  {p['name']}: AR={ar:.3f}, area={pano_areas[i]:.2f}")

        # Find anchor (highest AR contrast)
        pano_best_room = []
        pano_contrasts = []
        for pi, par in enumerate(pano_ars):
            diffs = sorted([(abs(par - rar), ri) for ri, rar in enumerate(room_ars)])
            best_diff, best_ri = diffs[0]
            second_diff = diffs[1][0] if len(diffs) > 1 else best_diff
            contrast = second_diff / best_diff if best_diff > 1e-8 else 1e6
            pano_contrasts.append(contrast)
            pano_best_room.append(best_ri)

        anchor_idx = max(range(len(panos)), key=lambda i: pano_contrasts[i])
        anchor_room = pano_best_room[anchor_idx]
        scale = np.sqrt(room_areas[anchor_room] / pano_areas[anchor_idx])
        print(f"\n  Anchor: {panos[anchor_idx]['name']} -> "
              f"{map_labels[anchor_room]}, scale={scale:.4f}")

        matches = run_matching(map_polys, map_labels, panos, scale)
        for m in matches:
            print(f"  {m['name']} -> {map_labels[m['room_idx']]} "
                  f"rot={m['rot_deg']}deg score={m['score']:.3f}")

        fig, ax = plt.subplots(figsize=(10, 14))
        ax.set_title(f"AR Anchor — {map_name} (scale={scale:.3f})", fontsize=14)
        for i, poly in enumerate(map_polys):
            closed = np.vstack([poly, poly[0]])
            ax.plot(closed[:,0], closed[:,1], "k-", lw=2.5, label=map_labels[i])
            ax.fill(poly[:,0], poly[:,1], alpha=0.08, color="gray")
        cmap = plt.cm.tab10
        for mi, match in enumerate(matches):
            color = cmap(mi % 10)
            poly = match["transformed"]
            closed = np.vstack([poly, poly[0]])
            ax.plot(closed[:,0], closed[:,1], "--", color=color, lw=2,
                    label=f"{match['name']} ({match['score']:.2f})")
            ax.fill(poly[:,0], poly[:,1], alpha=0.15, color=color)
        ax.set_aspect("equal"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
        ax.invert_yaxis()
        fig_path = output_dir / "demo6_alignment.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nVisualization: {fig_path}")


if __name__ == "__main__":
    main()
