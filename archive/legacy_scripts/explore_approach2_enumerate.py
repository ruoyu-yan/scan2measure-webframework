#!/usr/bin/env python3
"""
Approach 2: Enumerate Assignments + Per-Assignment Optimization

Match SAM3 panoramic footprint polygons to SAM3 density-image room polygons.
Instead of sweeping scale, enumerate all pano->room assignments and optimize
(shared_scale, per-pano rotation, per-pano translation) for each assignment.

With 3 panos and 2 rooms: 2^3 = 8 possible assignments.

Ground truth: TMB_office1 -> room_01, TMB_corridor_south1 -> room_03,
              TMB_corridor_south2 -> room_03

Scoring: true IoU = intersection / union per pano, summed.
This prevents scale-collapse since shrinking panos hurts the union denominator.
"""

import json
import itertools
import time
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from shapely.validation import make_valid
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# ── Paths ───────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent

ROOM_POLY_JSON = (
    _REPO / "data" / "sam3_room_segmentation"
    / "tmb_office_corridor_bigger" / "tmb_office_corridor_bigger_polygons.json"
)

PANO_NAMES = ["TMB_office1", "TMB_corridor_south1", "TMB_corridor_south2"]
PANO_LAYOUT_DIR = _REPO / "data" / "sam3_pano_processing"

GT_POSES_JSON = (
    _REPO / "data" / "pose_estimates" / "multiroom" / "local_filter_results.json"
)

OUT_PNG = (
    _REPO / "data" / "sam3_room_segmentation"
    / "tmb_office_corridor_bigger" / "approach2_enumerate.png"
)

# ── Optimization bounds ─────────────────────────────────────────────────────
SCALE_BOUNDS = (0.2, 2.5)
ROTATION_BOUNDS = (0.0, 2.0 * np.pi)
TRANSLATION_MARGIN = 5.0  # meters beyond room bbox


# ── Data loading ────────────────────────────────────────────────────────────

def load_room_polygons():
    """Load room polygons from SAM3 room segmentation JSON."""
    with open(ROOM_POLY_JSON) as f:
        data = json.load(f)
    rooms = {}
    for room in data["rooms"]:
        label = room["label"]
        verts = np.array(room["vertices_world_meters"])
        rooms[label] = verts
    return rooms


def load_pano_footprints():
    """Load pano footprint polygons from SAM3 pano processing.

    Pano coordinate system: +x = right, -z = forward.
    We use (x, -z) as our 2D coordinates.
    """
    panos = {}
    for name in PANO_NAMES:
        layout_path = PANO_LAYOUT_DIR / name / "layout.json"
        with open(layout_path) as f:
            data = json.load(f)
        corners = np.array(data["layout_corners"])
        # corners are [x, z_flipped] from SAM3 (z already flipped in extraction)
        # This maps directly to [map_x, map_y] since map_y = -world_y = z_flipped
        panos[name] = corners
    return panos


def load_ground_truth():
    """Load ground truth poses and compute expected map positions."""
    with open(GT_POSES_JSON) as f:
        poses = json.load(f)

    gt_positions = {}
    for name in PANO_NAMES:
        t = np.array(poses[name]["t"])
        # World coords -> density image coords: (x, -y)
        gt_positions[name] = np.array([t[0], -t[1]])

    gt_assignment = {
        "TMB_office1": "room_01",
        "TMB_corridor_south1": "room_03",
        "TMB_corridor_south2": "room_03",
    }

    return gt_positions, gt_assignment


# ── Geometry helpers ────────────────────────────────────────────────────────

def transform_polygon(vertices, scale, angle, tx, ty):
    """Apply scale, rotation, and translation to polygon vertices."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([[cos_a, -sin_a],
                   [sin_a,  cos_a]])
    scaled = vertices * scale
    rotated = scaled @ R.T
    translated = rotated + np.array([tx, ty])
    return translated


def compute_true_iou(pano_verts, room_verts):
    """Compute true IoU = intersection_area / union_area.

    This penalizes both under-coverage (pano too small) and
    over-reach (pano extends beyond room).
    """
    try:
        pano_poly = Polygon(pano_verts)
        room_poly = Polygon(room_verts)
        if not pano_poly.is_valid:
            pano_poly = make_valid(pano_poly)
        if not room_poly.is_valid:
            room_poly = make_valid(room_poly)
        intersection = pano_poly.intersection(room_poly).area
        union = pano_poly.union(room_poly).area
        if union < 1e-6:
            return 0.0
        return intersection / union
    except Exception:
        return 0.0


def compute_containment(pano_verts, room_verts):
    """Compute containment = intersection_area / pano_area."""
    try:
        pano_poly = Polygon(pano_verts)
        room_poly = Polygon(room_verts)
        if not pano_poly.is_valid:
            pano_poly = make_valid(pano_poly)
        if not room_poly.is_valid:
            room_poly = make_valid(room_poly)
        pano_area = pano_poly.area
        if pano_area < 1e-6:
            return 0.0
        intersection = pano_poly.intersection(room_poly).area
        return intersection / pano_area
    except Exception:
        return 0.0


def obb_long_axis_angle(verts):
    """Return the angle (radians) of the long axis of the oriented bounding box."""
    import cv2
    pts = np.array(verts, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle_deg = rect
    # cv2 angle is in [-90, 0); convert to the long axis angle
    angle_rad = np.radians(angle_deg)
    if w < h:
        angle_rad += np.pi / 2
    return angle_rad % np.pi  # [0, pi)


def refine_all_placements(pano_names, pano_footprints, room_polygons,
                          assignment, scale):
    """Refine placement of ALL panos jointly, with non-overlap penalty.

    Strategy:
    1. OBB long-axis alignment to get candidate rotations
    2. Dense translation grid within each room
    3. Score = sum(containment) - overlap_penalty between panos in same room
    4. Greedy sequential placement: place panos one at a time,
       penalizing overlap with already-placed panos

    Returns dict: name -> {angle, tx, ty, iou, containment, transformed}
    """
    # Group panos by room
    room_groups = {}
    for name in pano_names:
        room_label = assignment[name]
        room_groups.setdefault(room_label, []).append(name)

    results = {}

    for room_label, names_in_room in room_groups.items():
        room_verts = room_polygons[room_label]
        room_long_angle = obb_long_axis_angle(room_verts)
        room_xmin, room_ymin = np.min(room_verts, axis=0)
        room_xmax, room_ymax = np.max(room_verts, axis=0)

        already_placed = []  # list of Shapely polygons already placed

        # Place largest (most distinctive) panos first — they have
        # stronger shape constraints
        names_in_room = sorted(
            names_in_room,
            key=lambda n: Polygon(pano_footprints[n]).area,
            reverse=True,
        )
        print(f"    Placement order for {room_label}: {names_in_room}")

        for name in names_in_room:
            pano_verts = pano_footprints[name]
            pano_long_angle = obb_long_axis_angle(pano_verts)

            # Candidate rotations: align long axes
            base_angle = room_long_angle - pano_long_angle
            candidate_angles = []
            for offset in [0, np.pi]:
                center = base_angle + offset
                for delta_deg in range(-20, 21, 4):
                    candidate_angles.append(center + np.radians(delta_deg))
            # Coarse sweep for safety
            for deg in range(0, 360, 15):
                candidate_angles.append(np.radians(deg))

            # Translation grid — dense enough to catch shape features
            n_tx = max(20, int((room_xmax - room_xmin) * 3))
            n_ty = max(20, int((room_ymax - room_ymin) * 3))
            n_tx = min(n_tx, 50)
            n_ty = min(n_ty, 70)  # more steps along long axis
            tx_vals = np.linspace(room_xmin, room_xmax, n_tx)
            ty_vals = np.linspace(room_ymin, room_ymax, n_ty)

            best = {"score": -1e9}

            for angle in candidate_angles:
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                scaled_rotated = (pano_verts * scale) @ R.T
                sr_center = np.mean(scaled_rotated, axis=0)
                centered = scaled_rotated - sr_center

                for tx in tx_vals:
                    for ty in ty_vals:
                        translated = centered + np.array([tx, ty])
                        cont = compute_containment(translated, room_verts)
                        if cont < 0.3:
                            continue  # skip clearly bad placements

                        iou = compute_true_iou(translated, room_verts)

                        # Overlap penalty with already-placed panos
                        overlap_penalty = 0.0
                        if already_placed:
                            try:
                                this_poly = Polygon(translated)
                                if not this_poly.is_valid:
                                    this_poly = make_valid(this_poly)
                                this_area = this_poly.area
                                for prev_poly in already_placed:
                                    inter = this_poly.intersection(prev_poly).area
                                    if this_area > 1e-6:
                                        overlap_penalty += inter / this_area
                            except Exception:
                                pass

                        score = iou + cont - 1.5 * overlap_penalty

                        if score > best["score"]:
                            best = {
                                "score": score,
                                "iou": iou,
                                "containment": cont,
                                "overlap_penalty": overlap_penalty,
                                "angle": angle,
                                "tx": tx,
                                "ty": ty,
                                "transformed": translated.copy(),
                            }

            results[name] = best

            # Register this pano for overlap penalty
            try:
                placed_poly = Polygon(best["transformed"])
                if not placed_poly.is_valid:
                    placed_poly = make_valid(placed_poly)
                already_placed.append(placed_poly)
            except Exception:
                pass

    return results


# ── Assignment enumeration ──────────────────────────────────────────────────

def enumerate_assignments(pano_names, room_labels):
    """Generate all possible pano->room assignments."""
    assignments = []
    for combo in itertools.product(room_labels, repeat=len(pano_names)):
        assignment = dict(zip(pano_names, combo))
        assignments.append(assignment)
    return assignments


# ── Optimization for a single assignment ────────────────────────────────────

def optimize_assignment(assignment, pano_footprints, room_polygons):
    """Find optimal (scale, rotation[], translation[]) for a given assignment.

    Scoring uses true IoU (intersection/union) which prevents scale collapse.
    """
    # Build bounds
    bounds = [SCALE_BOUNDS]  # shared scale
    for name in PANO_NAMES:
        room_label = assignment[name]
        room_verts = room_polygons[room_label]
        room_xmin = np.min(room_verts[:, 0])
        room_xmax = np.max(room_verts[:, 0])
        room_ymin = np.min(room_verts[:, 1])
        room_ymax = np.max(room_verts[:, 1])

        bounds.append(ROTATION_BOUNDS)  # rotation
        bounds.append((room_xmin - TRANSLATION_MARGIN,
                        room_xmax + TRANSLATION_MARGIN))  # tx
        bounds.append((room_ymin - TRANSLATION_MARGIN,
                        room_ymax + TRANSLATION_MARGIN))  # ty

    def objective(params):
        scale = params[0]
        total_score = 0.0
        for i, name in enumerate(PANO_NAMES):
            angle = params[1 + i * 3]
            tx = params[2 + i * 3]
            ty = params[3 + i * 3]

            pano_verts = pano_footprints[name]
            room_label = assignment[name]
            room_verts = room_polygons[room_label]

            transformed = transform_polygon(pano_verts, scale, angle, tx, ty)
            iou = compute_true_iou(transformed, room_verts)
            total_score += iou

        return -total_score  # minimize negative

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=300,
        tol=1e-7,
        polish=True,
        workers=1,
        popsize=25,
    )

    # Extract results
    best_scale = result.x[0]
    best_score = -result.fun
    per_pano = {}
    for i, name in enumerate(PANO_NAMES):
        angle = result.x[1 + i * 3]
        tx = result.x[2 + i * 3]
        ty = result.x[3 + i * 3]

        pano_verts = pano_footprints[name]
        room_label = assignment[name]
        room_verts = room_polygons[room_label]
        transformed = transform_polygon(pano_verts, best_scale, angle, tx, ty)

        iou = compute_true_iou(transformed, room_verts)
        containment = compute_containment(transformed, room_verts)

        per_pano[name] = {
            "room": room_label,
            "angle_deg": np.degrees(angle) % 360,
            "tx": tx,
            "ty": ty,
            "iou": iou,
            "containment": containment,
            "transformed_verts": transformed,
        }

    return {
        "assignment": assignment,
        "scale": best_scale,
        "total_score": best_score,
        "per_pano": per_pano,
        "opt_result": result,
    }


# ── Visualization ───────────────────────────────────────────────────────────

def _draw_panel(ax, res, room_polygons, gt_positions, title, colors_pano):
    """Draw one panel: rooms + transformed pano polygons + GT markers."""
    colors_room = {"room_01": "#bdc3c7", "room_03": "#95a5a6"}

    for label, verts in room_polygons.items():
        poly = MplPolygon(verts, closed=True, fill=True,
                           facecolor=colors_room[label], edgecolor="black",
                           linewidth=1.5, alpha=0.3, label=f"Room: {label}")
        ax.add_patch(poly)
        cx, cy = np.mean(verts, axis=0)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="black")

    for name in PANO_NAMES:
        info = res["per_pano"][name]
        verts = info["transformed_verts"]
        poly = MplPolygon(verts, closed=True, fill=True,
                           facecolor=colors_pano[name], edgecolor=colors_pano[name],
                           linewidth=2, alpha=0.4,
                           label=f"{name.replace('TMB_','')} -> {info['room']} "
                                 f"(IoU={info['iou']:.3f})")
        ax.add_patch(poly)
        cx, cy = np.mean(verts, axis=0)
        short_name = name.replace("TMB_", "")
        ax.text(cx, cy, short_name, ha="center", va="center",
                fontsize=7, color=colors_pano[name], fontweight="bold")

    # Ground truth camera positions (stars)
    if gt_positions:
        for name, pos in gt_positions.items():
            ax.plot(pos[0], pos[1], '*', color=colors_pano[name],
                    markersize=14, markeredgecolor='black', markeredgewidth=0.8,
                    zorder=10)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.autoscale()


def visualize_comparison(stage1_result, stage2_result, room_polygons,
                         gt_positions, gt_assignment):
    """2-panel figure: Stage 1 (diff. evolution) vs Stage 2 (OBB-refined)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

    colors_pano = {
        "TMB_office1": "#e74c3c",
        "TMB_corridor_south1": "#3498db",
        "TMB_corridor_south2": "#2ecc71",
    }

    _draw_panel(ax1, stage1_result, room_polygons, gt_positions,
                f"Stage 1: diff_evolution\n"
                f"Total IoU={stage1_result['total_score']:.3f}, "
                f"scale={stage1_result['scale']:.4f}",
                colors_pano)

    _draw_panel(ax2, stage2_result, room_polygons, gt_positions,
                f"Stage 2: OBB-aligned refinement\n"
                f"Total IoU={stage2_result['total_score']:.3f}, "
                f"scale={stage2_result['scale']:.4f}",
                colors_pano)

    fig.suptitle("Approach 2: Assignment Enumeration + Placement Refinement\n"
                 "(Stars = ground truth camera positions)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {OUT_PNG}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("APPROACH 2: Enumerate Assignments + Per-Assignment Optimization")
    print("  Scoring: true IoU (intersection / union) per pano")
    print("=" * 70)

    # Load data
    print("\n--- Loading data ---")
    room_polygons = load_room_polygons()
    pano_footprints = load_pano_footprints()
    gt_positions, gt_assignment = load_ground_truth()

    for label, verts in room_polygons.items():
        poly = Polygon(verts)
        print(f"  Room {label}: {len(verts)} vertices, "
              f"area={poly.area:.2f} m^2, "
              f"bounds=({np.min(verts[:,0]):.1f},{np.min(verts[:,1]):.1f})-"
              f"({np.max(verts[:,0]):.1f},{np.max(verts[:,1]):.1f})")

    for name, verts in pano_footprints.items():
        poly = Polygon(verts)
        print(f"  Pano {name}: {len(verts)} vertices, "
              f"area={poly.area:.2f} m^2 (camera-local)")

    print(f"\n  Ground truth assignment: {gt_assignment}")
    for name, pos in gt_positions.items():
        print(f"  GT position {name}: ({pos[0]:.2f}, {pos[1]:.2f})")

    # Quick area analysis to predict expected scale
    print("\n--- Area analysis (helps predict scale) ---")
    for name in PANO_NAMES:
        pano_area = Polygon(pano_footprints[name]).area
        for label in sorted(room_polygons.keys()):
            room_area = Polygon(room_polygons[label]).area
            # If pano is scaled by s, pano_area_scaled = pano_area * s^2
            # Perfect overlap: pano_area * s^2 = room_area
            # s = sqrt(room_area / pano_area)
            s_est = np.sqrt(room_area / pano_area)
            print(f"  {name} -> {label}: "
                  f"sqrt(room_area/pano_area) = sqrt({room_area:.1f}/{pano_area:.1f}) "
                  f"= {s_est:.3f}")

    # Enumerate all assignments
    room_labels = sorted(room_polygons.keys())
    assignments = enumerate_assignments(PANO_NAMES, room_labels)
    n_assignments = len(assignments)
    print(f"\n--- Enumerating {n_assignments} possible assignments ---")
    for i, a in enumerate(assignments):
        mapping = ", ".join(f"{k}->{v}" for k, v in a.items())
        print(f"  Assignment {i}: {mapping}")

    # Optimize each assignment
    print(f"\n--- Optimizing each assignment (true IoU scoring) ---")
    all_results = []
    t0 = time.time()

    for i, assignment in enumerate(assignments):
        mapping = ", ".join(f"{k}->{v}" for k, v in assignment.items())
        t_start = time.time()
        result = optimize_assignment(assignment, pano_footprints, room_polygons)
        t_elapsed = time.time() - t_start

        all_results.append(result)

        per_iou = [result["per_pano"][n]["iou"] for n in PANO_NAMES]
        per_str = " | ".join(f"{v:.3f}" for v in per_iou)
        print(f"  [{i+1}/{n_assignments}] {mapping}")
        print(f"           score={result['total_score']:.4f}  "
              f"scale={result['scale']:.4f}  "
              f"per-pano IoU=[{per_str}]  ({t_elapsed:.1f}s)")

    total_time = time.time() - t0
    print(f"\n  Total optimization time: {total_time:.1f}s")

    # Rank by total score (descending)
    all_results.sort(key=lambda r: r["total_score"], reverse=True)

    # Report top 3
    print("\n" + "=" * 70)
    print("TOP 3 ASSIGNMENTS")
    print("=" * 70)

    for rank, res in enumerate(all_results[:3]):
        assignment = res["assignment"]
        matches_gt = all(
            assignment[name] == gt_assignment[name] for name in PANO_NAMES
        )

        print(f"\n--- Rank #{rank + 1} "
              f"{'[MATCHES GROUND TRUTH]' if matches_gt else ''} ---")
        print(f"  Total IoU score: {res['total_score']:.6f}")
        print(f"  Shared scale: {res['scale']:.6f}")
        print(f"  Assignment:")
        for name in PANO_NAMES:
            info = res["per_pano"][name]
            print(f"    {name} -> {info['room']}  "
                  f"angle={info['angle_deg']:.1f} deg  "
                  f"t=({info['tx']:.3f}, {info['ty']:.3f})  "
                  f"IoU={info['iou']:.4f}  "
                  f"containment={info['containment']:.4f}")

    # Confidence margin
    if len(all_results) >= 2:
        margin = all_results[0]["total_score"] - all_results[1]["total_score"]
        pct = (margin / all_results[0]["total_score"] * 100
               if all_results[0]["total_score"] > 0 else 0)
        print(f"\n--- Confidence ---")
        print(f"  Best score:   {all_results[0]['total_score']:.6f}")
        print(f"  2nd best:     {all_results[1]['total_score']:.6f}")
        print(f"  Margin:       {margin:.6f} ({pct:.1f}% of best)")

    # Check ground truth match
    best = all_results[0]
    best_matches_gt = all(
        best["assignment"][name] == gt_assignment[name] for name in PANO_NAMES
    )
    print(f"\n--- Ground truth check ---")
    print(f"  Best assignment matches ground truth: {best_matches_gt}")
    if not best_matches_gt:
        for rank, res in enumerate(all_results):
            if all(res["assignment"][name] == gt_assignment[name]
                   for name in PANO_NAMES):
                print(f"  Ground truth assignment is at rank #{rank + 1} "
                      f"with score {res['total_score']:.6f}")
                break

    # Full ranking table
    print(f"\n--- Full ranking ---")
    print(f"  {'Rank':>4}  {'Score':>8}  {'Scale':>7}  "
          f"{'Per-pano IoU':<30}  {'Assignment':<55}  {'GT?'}")
    for rank, res in enumerate(all_results):
        assignment = res["assignment"]
        matches_gt = all(
            assignment[name] == gt_assignment[name] for name in PANO_NAMES
        )
        mapping = ", ".join(f"{k.replace('TMB_','')}->{v}"
                            for k, v in assignment.items())
        per_iou = [res["per_pano"][n]["iou"] for n in PANO_NAMES]
        iou_str = ", ".join(f"{v:.3f}" for v in per_iou)
        gt_mark = " <-- GT" if matches_gt else ""
        print(f"  {rank+1:>4}  {res['total_score']:>8.4f}  "
              f"{res['scale']:>7.4f}  [{iou_str:<28}]  "
              f"{mapping:<55}{gt_mark}")

    # ================================================================
    # STAGE 2: Refine placement for the best assignment
    # ================================================================
    print(f"\n{'='*70}")
    print("STAGE 2: Refine placement (OBB long-axis alignment + dense grid)")
    print(f"{'='*70}")

    best_result = all_results[0]
    best_scale = best_result["scale"]
    best_assignment = best_result["assignment"]
    print(f"  Using assignment: {best_assignment}")
    print(f"  Using scale: {best_scale:.4f}")

    placements = refine_all_placements(
        PANO_NAMES, pano_footprints, room_polygons,
        best_assignment, best_scale
    )

    refined_per_pano = {}
    refined_total_iou = 0.0
    for name in PANO_NAMES:
        p = placements[name]
        room_label = best_assignment[name]
        refined_per_pano[name] = {
            "room": room_label,
            "angle_deg": np.degrees(p["angle"]) % 360,
            "tx": p["tx"],
            "ty": p["ty"],
            "iou": p["iou"],
            "containment": p["containment"],
            "transformed_verts": p["transformed"],
        }
        refined_total_iou += p["iou"]
        print(f"\n  {name} in {room_label}:")
        print(f"    angle={np.degrees(p['angle']) % 360:.1f}deg  "
              f"t=({p['tx']:.2f}, {p['ty']:.2f})  "
              f"IoU={p['iou']:.4f}  containment={p['containment']:.4f}  "
              f"overlap_penalty={p['overlap_penalty']:.4f}")

    print(f"\n  Refined total IoU: {refined_total_iou:.4f} "
          f"(was {best_result['total_score']:.4f})")

    # Build refined result for visualization
    refined_result = {
        "assignment": best_assignment,
        "scale": best_scale,
        "total_score": refined_total_iou,
        "per_pano": refined_per_pano,
    }

    # Visualize: Stage 1 best vs Stage 2 refined
    print("\n--- Generating visualization ---")
    visualize_comparison(best_result, refined_result, room_polygons,
                         gt_positions, gt_assignment)

    print("\nDone.")


if __name__ == "__main__":
    main()
