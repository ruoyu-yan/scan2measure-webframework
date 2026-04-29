"""FGPL Feature Extraction Pipeline V2 — independent reimplementation.

Extracts 2D line features from a panoramic image using sphere-based LSD detection,
vanishing point voting, and great-circle intersection finding. Produces the same
output schema as the panoramic-localization FGPL pipeline but without the PICCOLO
90-degree Z-axis rotation.

Outputs to data/pano/2d_feature_extracted/{ROOM_NAME}_v2/:
    - fgpl_features.json  (lines, principal directions, intersections)
    - edge_overlay.png    (all lines white on black)
    - grouped_lines.png   (classified lines in R/G/B over panorama)
"""

import json
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_ROOT / "utils"))
from config_loader import load_config, progress

import cv2
import numpy as np
import torch

from pano_line_detector import detect_pano_lines
from line_analysis import extract_vanishing_points, classify_lines, find_intersections_2d
from sphere_geometry import render_sphere_lines

# ── Config ──────────────────────────────────────────────────────────────────
ROOM_NAME = "TMB_corridor_south1"
PANO_RESOLUTION = (512, 1024)
ELEV_MASK_DEG = 30  # discard lines whose midpoint is within this many degrees of either pole

ROOT = Path(__file__).resolve().parent.parent
PANO_PATH = ROOT / "data" / "pano" / "raw" / f"{ROOM_NAME}.jpg"
OUT_DIR = ROOT / "data" / "pano" / "2d_feature_extracted" / f"{ROOM_NAME}_v2"


def main():
    cfg = load_config()
    room_name = cfg.get("room_name", ROOM_NAME)
    pano_res = tuple(cfg["pano_resolution"]) if cfg.get("pano_resolution") else PANO_RESOLUTION
    pano_path = Path(cfg["pano_path"]) if cfg.get("pano_path") else ROOT / "data" / "pano" / "raw" / f"{room_name}.jpg"
    out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else ROOT / "data" / "pano" / "2d_feature_extracted" / f"{room_name}_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load panorama
    progress(1, 4, f"Loading panorama {room_name}")
    img = cv2.imread(str(pano_path))
    assert img is not None, f"Cannot read {pano_path}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (pano_res[1], pano_res[0]))
    print(f"Loaded panorama: {img.shape}")

    # 2. Detect lines + elevation mask (discard near-pole lines)
    progress(2, 4, "Detecting lines on sphere")
    lines_np = detect_pano_lines(img)
    n_raw = lines_np.shape[0]
    starts = lines_np[:, 3:6]
    ends = lines_np[:, 6:9]
    starts_n = starts / np.linalg.norm(starts, axis=1, keepdims=True)
    ends_n = ends / np.linalg.norm(ends, axis=1, keepdims=True)
    midpoints = starts_n + ends_n
    midpoints = midpoints / np.linalg.norm(midpoints, axis=1, keepdims=True)
    elev_deg = np.degrees(np.arcsin(np.clip(midpoints[:, 2], -1, 1)))
    keep = np.abs(elev_deg) < (90 - ELEV_MASK_DEG)
    lines_np = lines_np[keep]
    lines = torch.from_numpy(lines_np).float()
    print(f"Detected {n_raw} lines, kept {lines.shape[0]} after "
          f"elevation mask ({ELEV_MASK_DEG}° from poles)")

    # 3. Extract vanishing points
    progress(3, 4, "Analyzing vanishing points and intersections")
    principal_2d = extract_vanishing_points(lines)
    det_val = torch.det(principal_2d).item()
    print(f"Principal directions (det={det_val:.6f}):\n{principal_2d}")

    # 4. Classify lines and find intersections
    mask = classify_lines(lines, principal_2d)
    inters = find_intersections_2d(lines, principal_2d)
    inter_counts = [g.shape[0] for g in inters]
    print(f"Intersections per group: {inter_counts}, total: {sum(inter_counts)}")

    # 5. Render visualizations
    progress(4, 4, "Saving features")
    H, W = pano_res

    # edge_overlay.png — white lines on black
    edge_img = render_sphere_lines(lines, resolution=(H, W))
    cv2.imwrite(str(out_dir / "edge_overlay.png"),
                cv2.cvtColor(edge_img, cv2.COLOR_RGB2BGR))
    print(f"Saved edge_overlay.png")

    # grouped_lines.png — R/G/B groups composited over panorama
    colors = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float)
    pano_base = img.copy()
    for i in range(3):
        group_lines = lines[mask[:, i]]
        if group_lines.shape[0] == 0:
            continue
        rgb = colors[i].unsqueeze(0).expand(group_lines.shape[0], -1)
        group_img = render_sphere_lines(group_lines, resolution=(H, W), rgb=rgb)
        # Composite: overwrite panorama where group has non-black pixels
        line_mask = group_img.sum(axis=-1) > 0
        pano_base[line_mask] = group_img[line_mask]
    cv2.imwrite(str(out_dir / "grouped_lines.png"),
                cv2.cvtColor(pano_base, cv2.COLOR_RGB2BGR))
    print(f"Saved grouped_lines.png")

    # 6. Save JSON
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
    print(f"Saved fgpl_features.json ({sum(inter_counts)} intersections)")


if __name__ == "__main__":
    main()
