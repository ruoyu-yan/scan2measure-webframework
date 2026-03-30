#!/usr/bin/env python3
"""
Approach 1: Improved Scale Sweep for polygon alignment.

Strategy:
1. Determine room assignments using aspect-ratio similarity
   (corridor panos -> corridor room, office pano -> office room)
2. Sweep global scale + per-pano rotation to maximize sum of IoUs
   given the fixed room assignments
3. Use translation grid to find best placement within assigned room

Two-phase sweep:
  Phase 1 (coarse): scale 0.3-2.0 in 50 steps, per-pano rotation 0-360 in 5deg steps
  Phase 2 (fine): +/-0.15 around best scale in 60 steps, +/-20deg in 1deg steps
"""

import json
import time
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
BASE = Path("/home/ruoyu/scan2measure-webframework")

ROOM_POLY_PATH = (
    BASE / "data/sam3_room_segmentation/tmb_office_corridor_bigger"
    / "tmb_office_corridor_bigger_polygons.json"
)

PANO_NAMES = ["TMB_corridor_south1", "TMB_corridor_south2", "TMB_office1"]

PANO_LAYOUT_PATHS = {
    name: BASE / f"data/sam3_pano_processing/{name}/layout.json"
    for name in PANO_NAMES
}

POSE_PATH = BASE / "data/pose_estimates/multiroom/local_filter_results.json"

OUT_DIR = BASE / "data/sam3_room_segmentation/tmb_office_corridor_bigger"
OUT_PNG = OUT_DIR / "approach1_improved_sweep.png"

GT_ASSIGNMENT = {
    "TMB_office1": "room_01",
    "TMB_corridor_south1": "room_03",
    "TMB_corridor_south2": "room_03",
}


# ── data loading ───────────────────────────────────────────────────────

def load_room_polygons():
    with open(ROOM_POLY_PATH) as f:
        data = json.load(f)
    rooms = {}
    for r in data["rooms"]:
        verts = np.array(r["vertices_world_meters"])
        poly = make_valid(Polygon(verts))
        rooms[r["label"]] = poly
    return rooms


def load_pano_footprint(name):
    with open(PANO_LAYOUT_PATHS[name]) as f:
        data = json.load(f)
    corners = np.array(data["layout_corners"])
    corners[:, 1] = -corners[:, 1]
    return corners


def load_gt_poses():
    with open(POSE_PATH) as f:
        data = json.load(f)
    poses = {}
    for name in PANO_NAMES:
        t = np.array(data[name]["t"])
        R_mat = np.array(data[name]["R"])
        yaw = np.degrees(np.arctan2(R_mat[1][0], R_mat[0][0]))
        poses[name] = {"x": t[0], "y": t[1], "yaw": yaw}
    return poses


# ── geometry helpers ───────────────────────────────────────────────────

def transform_polygon(pts, scale, angle_deg):
    """Scale and rotate (N,2) polygon around its centroid. Returns centered."""
    c = pts.mean(axis=0)
    centered = pts - c
    scaled = centered * scale
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return scaled @ R.T


def make_poly(pts):
    p = Polygon(pts)
    if not p.is_valid:
        p = make_valid(p)
    return p


def get_aspect_ratio(poly):
    """Aspect ratio from minimum rotated rectangle."""
    rect = poly.minimum_rotated_rectangle
    coords = np.array(rect.exterior.coords)
    sides = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    if min(sides) < 1e-6:
        return 999.0
    return max(sides) / min(sides)


def compute_iou(poly_a, poly_b):
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0
    try:
        inter = poly_a.intersection(poly_b).area
    except Exception:
        return 0.0
    union = poly_a.area + poly_b.area - inter
    if union < 1e-12:
        return 0.0
    return inter / union


def compute_containment(pano_poly, room_poly):
    if pano_poly.is_empty or room_poly.is_empty:
        return 0.0
    try:
        inter = pano_poly.intersection(room_poly).area
    except Exception:
        return 0.0
    pa = pano_poly.area
    if pa < 1e-12:
        return 0.0
    return inter / pa


# ── translation grid ──────────────────────────────────────────────────

def build_translation_grid(room_poly, step=1.0):
    """Dense grid of interior points for translation candidates."""
    bounds = room_poly.bounds
    xs = np.arange(bounds[0], bounds[2] + step, step)
    ys = np.arange(bounds[1], bounds[3] + step, step)
    points = [(room_poly.centroid.x, room_poly.centroid.y)]
    for x in xs:
        for y in ys:
            if room_poly.contains(Point(x, y)):
                points.append((x, y))
    return np.array(points)


# ── room assignment via aspect ratio ──────────────────────────────────

def assign_rooms_by_aspect(pano_corners, room_polys):
    """Assign each pano to the room with most similar aspect ratio."""
    room_aspects = {}
    for rlabel, rpoly in room_polys.items():
        room_aspects[rlabel] = get_aspect_ratio(rpoly)

    assignments = {}
    for pname, corners in pano_corners.items():
        poly = Polygon(corners)
        pano_ar = get_aspect_ratio(poly)
        best_sim = -1
        best_room = None
        for rlabel, rar in room_aspects.items():
            sim = min(pano_ar, rar) / max(pano_ar, rar)
            if sim > best_sim:
                best_sim = sim
                best_room = rlabel
        assignments[pname] = best_room
        print(f"  {pname}: aspect={pano_ar:.2f} -> {best_room} "
              f"(room_aspect={room_aspects[best_room]:.2f}, sim={best_sim:.3f})")
    return assignments


# ── IoU-based sweep with fixed assignments ────────────────────────────

def best_iou_in_room(transformed_pts, room_poly, translations):
    """Find best IoU by placing pano centroid at each translation point."""
    best = 0.0
    best_tx = 0.0
    best_ty = 0.0
    for tx, ty in translations:
        shifted = transformed_pts + np.array([tx, ty])
        pano_poly = make_poly(shifted)
        iou = compute_iou(pano_poly, room_poly)
        if iou > best:
            best = iou
            best_tx = tx
            best_ty = ty
    return best, best_tx, best_ty


def score_scale(scale, pano_corners, assignments, room_polys, room_trans,
                angles):
    """For a given scale, optimize per-pano rotation and translation.

    Returns (total_iou, per_pano_info).
    """
    total = 0.0
    info = {}
    for pname, corners in pano_corners.items():
        rlabel = assignments[pname]
        rpoly = room_polys[rlabel]
        trans = room_trans[rlabel]
        best_iou = -1.0
        best_angle = 0.0
        best_tx = 0.0
        best_ty = 0.0
        for a in angles:
            transformed = transform_polygon(corners, scale, a)
            iou, tx, ty = best_iou_in_room(transformed, rpoly, trans)
            if iou > best_iou:
                best_iou = iou
                best_angle = a
                best_tx = tx
                best_ty = ty
        total += best_iou
        info[pname] = {
            "iou": best_iou, "angle": best_angle,
            "tx": best_tx, "ty": best_ty, "room": rlabel,
        }
    return total, info


def coarse_sweep(pano_corners, assignments, room_polys, room_trans):
    """Phase 1: coarse sweep over scale, per-pano rotation."""
    scales = np.linspace(0.3, 2.0, 50)
    angles = np.arange(0, 360, 5)
    print(f"Phase 1 (coarse): {len(scales)} scales x "
          f"{len(angles)} rotations per pano")

    best_total = -1.0
    best_scale = None
    best_info = None
    t0 = time.time()

    for i, s in enumerate(scales):
        total, info = score_scale(
            s, pano_corners, assignments, room_polys, room_trans, angles)
        if total > best_total:
            best_total = total
            best_scale = s
            best_info = info
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = (len(scales) - i - 1) * elapsed / (i + 1)
            print(f"  [{i+1}/{len(scales)}]  best: s={best_scale:.4f} "
                  f"total_IoU={best_total:.4f}  (ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"Phase 1 done in {elapsed:.1f}s")
    print(f"  Best scale: {best_scale:.4f}, total_IoU={best_total:.4f}")
    for pname, pi in best_info.items():
        print(f"    {pname} -> {pi['room']}: "
              f"angle={pi['angle']:.0f}deg, IoU={pi['iou']:.4f}")
    return best_scale, best_total, best_info


def fine_sweep(pano_corners, assignments, room_polys, room_trans,
               coarse_scale, coarse_info):
    """Phase 2: fine refinement around coarse peak."""
    s_lo = max(0.05, coarse_scale - 0.15)
    s_hi = coarse_scale + 0.15
    scales = np.linspace(s_lo, s_hi, 60)

    # Per-pano fine angles around coarse best
    pano_angles = {}
    for pname, pi in coarse_info.items():
        ca = pi["angle"]
        pano_angles[pname] = np.arange(ca - 20, ca + 21, 1.0) % 360

    print(f"\nPhase 2 (fine): {len(scales)} scales, "
          f"per-pano +/-20deg in 1deg steps")

    best_total = -1.0
    best_scale = None
    best_info = None
    t0 = time.time()

    for s in scales:
        total = 0.0
        info = {}
        for pname, corners in pano_corners.items():
            rlabel = assignments[pname]
            rpoly = room_polys[rlabel]
            trans = room_trans[rlabel]
            best_iou = -1.0
            best_angle = 0.0
            best_tx = 0.0
            best_ty = 0.0
            for a in pano_angles[pname]:
                transformed = transform_polygon(corners, s, a)
                iou, tx, ty = best_iou_in_room(transformed, rpoly, trans)
                if iou > best_iou:
                    best_iou = iou
                    best_angle = a
                    best_tx = tx
                    best_ty = ty
            total += best_iou
            info[pname] = {
                "iou": best_iou, "angle": best_angle,
                "tx": best_tx, "ty": best_ty, "room": rlabel,
            }
        if total > best_total:
            best_total = total
            best_scale = s
            best_info = info

    elapsed = time.time() - t0
    print(f"Phase 2 done in {elapsed:.1f}s")
    print(f"  Best scale: {best_scale:.4f}, total_IoU={best_total:.4f}")
    for pname, pi in best_info.items():
        print(f"    {pname} -> {pi['room']}: "
              f"angle={pi['angle']:.0f}deg, IoU={pi['iou']:.4f}")
    return best_scale, best_total, best_info


# ── visualization ──────────────────────────────────────────────────────

def visualize(room_polys, pano_corners, best_scale, best_info,
              gt_poses, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))

    colors_room = {"room_01": "#4488CC", "room_03": "#CC8844"}
    colors_pano = {
        "TMB_office1": "#FF4444",
        "TMB_corridor_south1": "#44CC44",
        "TMB_corridor_south2": "#AA44FF",
    }

    # Left: aligned result
    ax = axes[0]
    ax.set_title(f"Approach 1: scale={best_scale:.4f}", fontsize=12)

    for rlabel, rpoly in room_polys.items():
        xs, ys = rpoly.exterior.xy
        ax.fill(xs, ys, alpha=0.12, color=colors_room.get(rlabel, "gray"))
        ax.plot(xs, ys, color=colors_room.get(rlabel, "gray"),
                linewidth=2, label=f"room: {rlabel} "
                f"(area={rpoly.area:.1f}m2)")

    for pname in PANO_NAMES:
        corners = pano_corners[pname]
        pi = best_info[pname]
        transformed = transform_polygon(corners, best_scale, pi["angle"])
        shifted = transformed + np.array([pi["tx"], pi["ty"]])
        poly = make_poly(shifted)
        xs, ys = poly.exterior.xy
        c = colors_pano[pname]
        ax.fill(xs, ys, alpha=0.3, color=c)
        ax.plot(xs, ys, color=c, linewidth=2, linestyle="--",
                label=f"{pname}->{pi['room']} "
                f"(a={pi['angle']:.0f}deg, IoU={pi['iou']:.3f})")
        # Mark centroid
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.plot(cx, cy, "x", color=c, markersize=8, markeredgewidth=2)

    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")

    # Right: ground truth
    ax2 = axes[1]
    ax2.set_title("Ground truth poses", fontsize=12)

    for rlabel, rpoly in room_polys.items():
        xs, ys = rpoly.exterior.xy
        ax2.fill(xs, ys, alpha=0.12, color=colors_room.get(rlabel, "gray"))
        ax2.plot(xs, ys, color=colors_room.get(rlabel, "gray"),
                 linewidth=2, label=f"room: {rlabel}")

    for pname in PANO_NAMES:
        info = gt_poses[pname]
        c = colors_pano[pname]
        ax2.plot(info["x"], info["y"], "o", color=c, markersize=10,
                 label=f"{pname} ({info['x']:.1f},{info['y']:.1f}) "
                 f"yaw={info['yaw']:.0f}deg")
        # Draw heading arrow
        yaw_rad = np.radians(info["yaw"])
        dx = np.cos(yaw_rad) * 1.0
        dy = np.sin(yaw_rad) * 1.0
        ax2.arrow(info["x"], info["y"], dx, dy,
                  head_width=0.15, head_length=0.1, fc=c, ec=c, alpha=0.5)

    ax2.set_aspect("equal")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("world X (m)")
    ax2.set_ylabel("world Y (m)")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"\nVisualization saved to {out_path}")


# ── main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Approach 1: Improved Scale Sweep")
    print("  Step 1: Room assignment via aspect-ratio similarity")
    print("  Step 2: Global scale + per-pano rotation sweep (IoU)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    room_polys = load_room_polygons()
    for rlabel, rpoly in room_polys.items():
        ar = get_aspect_ratio(rpoly)
        b = rpoly.bounds
        print(f"  Room {rlabel}: area={rpoly.area:.2f} m^2, "
              f"aspect={ar:.2f}, bbox={b[2]-b[0]:.1f}x{b[3]-b[1]:.1f}m")

    pano_corners = {}
    for name in PANO_NAMES:
        corners = load_pano_footprint(name)
        pano_corners[name] = corners
        poly = Polygon(corners)
        ar = get_aspect_ratio(poly)
        print(f"  Pano {name}: {len(corners)} corners, "
              f"area={poly.area:.2f} m^2, aspect={ar:.2f}")

    gt_poses = load_gt_poses()
    for name, info in gt_poses.items():
        print(f"  GT {name}: ({info['x']:.3f}, {info['y']:.3f}), "
              f"yaw={info['yaw']:.1f}deg")

    # Step 1: Room assignment
    print("\nStep 1: Room assignment via aspect-ratio similarity")
    assignments = assign_rooms_by_aspect(pano_corners, room_polys)

    all_correct = True
    for pname, assigned in assignments.items():
        gt = GT_ASSIGNMENT[pname]
        ok = "CORRECT" if assigned == gt else "WRONG"
        if assigned != gt:
            all_correct = False
        print(f"  {pname}: {assigned} (GT: {gt}) {ok}")
    if all_correct:
        print("  -> All assignments correct!")
    else:
        print("  -> WARNING: Some assignments wrong!")

    # Build translation grids
    print("\nBuilding translation grids (0.5m step)...")
    room_trans = {}
    for rlabel, rpoly in room_polys.items():
        pts = build_translation_grid(rpoly, step=0.5)
        room_trans[rlabel] = pts
        print(f"  Room {rlabel}: {len(pts)} candidates")

    # Step 2: Scale + rotation sweep
    print("\n" + "=" * 70)
    print("Step 2: Scale + per-pano rotation sweep (IoU objective)")
    cs, ct, ci = coarse_sweep(
        pano_corners, assignments, room_polys, room_trans)

    print("=" * 70)
    fs, ft, fi = fine_sweep(
        pano_corners, assignments, room_polys, room_trans, cs, ci)

    # Report
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Optimal scale: {fs:.4f}")
    print(f"Total IoU:     {ft:.4f}")
    print()

    for pname in PANO_NAMES:
        pi = fi[pname]
        gt_room = GT_ASSIGNMENT[pname]
        match = "CORRECT" if pi["room"] == gt_room else "WRONG"
        print(f"  {pname}:")
        print(f"    Room: {pi['room']} (GT: {gt_room}) -> {match}")
        print(f"    Rotation: {pi['angle']:.0f}deg  "
              f"(GT yaw: {gt_poses[pname]['yaw']:.1f}deg)")
        print(f"    IoU: {pi['iou']:.4f}")
        print(f"    Position: ({pi['tx']:.2f}, {pi['ty']:.2f})")

        # Also compute containment at final position
        corners = pano_corners[pname]
        transformed = transform_polygon(corners, fs, pi["angle"])
        shifted = transformed + np.array([pi["tx"], pi["ty"]])
        pano_poly = make_poly(shifted)
        cont = compute_containment(pano_poly, room_polys[pi["room"]])
        print(f"    Containment: {cont:.4f}")

    print(f"\nComparison with previous approach 3:")
    print(f"  Previous: scale=0.65, max single IoU=0.66 (4 discrete rotations)")
    print(f"  This:     scale={fs:.4f}, total_IoU={ft:.4f}")

    avg_iou = ft / len(PANO_NAMES)
    print(f"  Average per-pano IoU: {avg_iou:.4f}")

    visualize(room_polys, pano_corners, fs, fi, gt_poses, OUT_PNG)


if __name__ == "__main__":
    main()
