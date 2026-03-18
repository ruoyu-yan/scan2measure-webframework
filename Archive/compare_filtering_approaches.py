"""Compare three approaches to cleaning 2D feature extraction.

Approach 1: Skip zenith/nadir views + post-merge elevation mask
Approach 2: Pre-merge per-view elevation filtering (segments filtered before merge)
Approach 3: Post-merge elevation mask only (all 26 views, filter after merge)

Outputs to data/pano/2d_feature_extracted/{ROOM_NAME}_v2/approach{1,2,3}/
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch

from pano_line_detector import (
    _decompose_pano, _detect_lsd, _back_project_to_sphere, _merge_segments,
)
from line_analysis import extract_vanishing_points, classify_lines, find_intersections_2d
from sphere_geometry import render_sphere_lines

# ── Config ──────────────────────────────────────────────────────────────────
ROOM_NAME = "TMB_corridor_south1"
PANO_RESOLUTION = (512, 1024)
ELEV_MASK_DEGS = [15, 30, 45]  # sweep of thresholds to compare

ROOT = Path(__file__).resolve().parent.parent
PANO_PATH = ROOT / "data" / "pano" / "raw" / f"{ROOM_NAME}.jpg"
BASE_OUT = ROOT / "data" / "pano" / "2d_feature_extracted" / f"{ROOM_NAME}_v2"

# 90-degree Z rotation (HorizonNet → equirectangular frame)
ROT = np.array([[0, -1, 0],
                [1,  0, 0],
                [0,  0, 1]], dtype=np.float64)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_view_params():
    """Return the standard 26-view parameters."""
    fov = np.pi / 3
    xh = np.arange(-np.pi, np.pi * 5 / 6, np.pi / 6)
    yh = np.zeros(xh.shape[0])
    xp = np.array([-3, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2]) / 3 * np.pi
    yp = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]) / 4 * np.pi
    x = np.concatenate([xh, xp, [0, 0]])
    y = np.concatenate([yh, yp, [np.pi / 2, -np.pi / 2]])
    return fov, x, y


def _midpoint_elevation_deg(coordN_lines):
    """Compute the sphere elevation (degrees) of each line's midpoint.

    Works on raw back-projected coordinates (before or after ROT —
    the rotation preserves the z component).
    """
    starts = coordN_lines[:, 3:6]
    ends = coordN_lines[:, 6:9]
    starts = starts / np.linalg.norm(starts, axis=1, keepdims=True)
    ends = ends / np.linalg.norm(ends, axis=1, keepdims=True)
    midpoints = starts + ends
    midpoints = midpoints / np.linalg.norm(midpoints, axis=1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(midpoints[:, 2], -1, 1)))


def _elevation_mask(coordN_lines, elev_mask_deg):
    """Keep lines whose midpoint is NOT within elev_mask_deg of either pole."""
    elev = _midpoint_elevation_deg(coordN_lines)
    keep = np.abs(elev) < (90 - elev_mask_deg)
    return coordN_lines[keep], keep


def _detect_and_backproject(scenes):
    """Run LSD + back-project for each perspective scene."""
    edges = []
    for scene in scenes:
        edge_map, edge_list = _detect_lsd(scene['img'])
        edge_data = {
            'img': edge_map,
            'edgeLst': edge_list,
            'vx': scene['vx'],
            'vy': scene['vy'],
            'fov': scene['fov'],
        }
        edge_data['panoLst'] = _back_project_to_sphere(edge_data)
        edges.append(edge_data)
    return edges


def _apply_rotation(coordN_lines):
    """Apply HorizonNet → equirectangular 90-deg Z rotation."""
    out = coordN_lines.copy()
    out[:, :3] = out[:, :3] @ ROT
    out[:, 3:6] = out[:, 3:6] @ ROT
    out[:, 6:9] = out[:, 6:9] @ ROT
    return out


def _render_and_save(coordN_lines, img, out_dir, H, W, label):
    """VP extraction, classification, rendering, JSON export."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = torch.from_numpy(coordN_lines).float()

    principal_2d = extract_vanishing_points(lines)
    det_val = torch.det(principal_2d).item()
    mask = classify_lines(lines, principal_2d)
    inters = find_intersections_2d(lines, principal_2d)
    inter_counts = [g.shape[0] for g in inters]

    # edge_overlay.png
    edge_img = render_sphere_lines(lines, resolution=(H, W))
    cv2.imwrite(str(out_dir / "edge_overlay.png"),
                cv2.cvtColor(edge_img, cv2.COLOR_RGB2BGR))

    # grouped_lines.png
    colors = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float)
    pano_base = img.copy()
    for i in range(3):
        group_lines = lines[mask[:, i]]
        if group_lines.shape[0] == 0:
            continue
        rgb = colors[i].unsqueeze(0).expand(group_lines.shape[0], -1)
        group_img = render_sphere_lines(group_lines, resolution=(H, W), rgb=rgb)
        line_mask = group_img.sum(axis=-1) > 0
        pano_base[line_mask] = group_img[line_mask]
    cv2.imwrite(str(out_dir / "grouped_lines.png"),
                cv2.cvtColor(pano_base, cv2.COLOR_RGB2BGR))

    # fgpl_features.json
    features = {
        "n_lines": int(lines.shape[0]),
        "line_shape": list(lines.shape),
        "principal_2d": principal_2d.tolist(),
        "principal_2d_det": det_val,
        "intersections_per_group": inter_counts,
        "total_intersections": sum(inter_counts),
        "lines": lines.tolist(),
        "inter_2d": [g.tolist() for g in inters],
    }
    with open(out_dir / "fgpl_features.json", "w") as f:
        json.dump(features, f)

    classified = mask.any(dim=1).sum().item()
    print(f"  [{label}] {lines.shape[0]} lines, {classified} classified, "
          f"intersections: {inter_counts} (total: {sum(inter_counts)}), "
          f"det={det_val:.4f}")


# ── Approaches ──────────────────────────────────────────────────────────────

def approach_1(img, H, W, elev_deg):
    """Skip zenith/nadir views + post-merge elevation mask."""
    print(f"\n=== Approach 1 @ {elev_deg}°: Skip views + post-merge elevation mask ===")
    fov, x, y = _get_view_params()

    # Drop last 2 entries (zenith=index 24, nadir=index 25) → 24 views
    scenes = _decompose_pano(img, fov, x[:24], y[:24], 320)
    print(f"  Using {len(scenes)} views (skipped zenith + nadir)")

    edges = _detect_and_backproject(scenes)
    _, coordN = _merge_segments(edges)
    coordN = _apply_rotation(coordN)

    n_before = len(coordN)
    coordN, _ = _elevation_mask(coordN, elev_deg)
    print(f"  Elevation mask ({elev_deg}°): {n_before} → {len(coordN)} "
          f"(removed {n_before - len(coordN)})")

    _render_and_save(coordN, img, BASE_OUT / f"approach1_{elev_deg}deg", H, W,
                     f"A1@{elev_deg}")


def approach_2(img, H, W, elev_deg):
    """Pre-merge per-view elevation filtering."""
    print(f"\n=== Approach 2 @ {elev_deg}°: Pre-merge per-view filtering ===")
    fov, x, y = _get_view_params()

    scenes = _decompose_pano(img, fov, x, y, 320)
    print(f"  Using all {len(scenes)} views")

    elev_threshold = 90 - elev_deg
    n_total_raw = 0
    n_removed = 0

    edges = []
    for scene in scenes:
        edge_map, edge_list = _detect_lsd(scene['img'])
        edge_data = {
            'img': edge_map,
            'edgeLst': edge_list,
            'vx': scene['vx'],
            'vy': scene['vy'],
            'fov': scene['fov'],
        }
        pano_lst = _back_project_to_sphere(edge_data)

        # Filter segments by midpoint elevation BEFORE merging
        if len(pano_lst) > 0:
            n_total_raw += len(pano_lst)
            elev = _midpoint_elevation_deg(pano_lst)
            keep = np.abs(elev) < elev_threshold
            n_removed += (~keep).sum()
            pano_lst = pano_lst[keep]

        edge_data['panoLst'] = pano_lst
        edges.append(edge_data)

    print(f"  Pre-merge filtering ({elev_deg}°): {n_total_raw} raw segments → "
          f"{n_total_raw - n_removed} kept (removed {n_removed})")

    _, coordN = _merge_segments(edges)
    coordN = _apply_rotation(coordN)

    _render_and_save(coordN, img, BASE_OUT / f"approach2_{elev_deg}deg", H, W,
                     f"A2@{elev_deg}")


def approach_3(img, H, W, elev_deg):
    """Post-merge elevation mask only (all 26 views)."""
    print(f"\n=== Approach 3 @ {elev_deg}°: Post-merge elevation mask only ===")
    fov, x, y = _get_view_params()

    scenes = _decompose_pano(img, fov, x, y, 320)
    print(f"  Using all {len(scenes)} views")

    edges = _detect_and_backproject(scenes)
    _, coordN = _merge_segments(edges)
    coordN = _apply_rotation(coordN)

    n_before = len(coordN)
    coordN, _ = _elevation_mask(coordN, elev_deg)
    print(f"  Elevation mask ({elev_deg}°): {n_before} → {len(coordN)} "
          f"(removed {n_before - len(coordN)})")

    _render_and_save(coordN, img, BASE_OUT / f"approach3_{elev_deg}deg", H, W,
                     f"A3@{elev_deg}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    img = cv2.imread(str(PANO_PATH))
    assert img is not None, f"Cannot read {PANO_PATH}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (PANO_RESOLUTION[1], PANO_RESOLUTION[0]))
    H, W = PANO_RESOLUTION
    print(f"Loaded panorama: {img.shape}")

    for elev_deg in ELEV_MASK_DEGS:
        approach_1(img.copy(), H, W, elev_deg)
        approach_2(img.copy(), H, W, elev_deg)
        approach_3(img.copy(), H, W, elev_deg)

    print(f"\nResults saved to {BASE_OUT}/approach{{1,2,3}}_{{15,30,45}}deg/")


if __name__ == "__main__":
    main()
