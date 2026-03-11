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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch

from pano_line_detector import detect_pano_lines
from line_analysis import extract_vanishing_points, classify_lines, find_intersections_2d
from sphere_geometry import render_sphere_lines

# ── Config ──────────────────────────────────────────────────────────────────
ROOM_NAME = "TMB_office1"
PANO_RESOLUTION = (512, 1024)

ROOT = Path(__file__).resolve().parent.parent
PANO_PATH = ROOT / "data" / "pano" / "raw" / f"{ROOM_NAME}.jpg"
OUT_DIR = ROOT / "data" / "pano" / "2d_feature_extracted" / f"{ROOM_NAME}_v2"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load panorama
    img = cv2.imread(str(PANO_PATH))
    assert img is not None, f"Cannot read {PANO_PATH}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (PANO_RESOLUTION[1], PANO_RESOLUTION[0]))
    print(f"Loaded panorama: {img.shape}")

    # 2. Detect lines
    lines_np = detect_pano_lines(img)
    lines = torch.from_numpy(lines_np).float()
    print(f"Detected {lines.shape[0]} lines")

    # 3. Extract vanishing points
    principal_2d = extract_vanishing_points(lines)
    det_val = torch.det(principal_2d).item()
    print(f"Principal directions (det={det_val:.6f}):\n{principal_2d}")

    # 4. Classify lines and find intersections
    mask = classify_lines(lines, principal_2d)
    inters = find_intersections_2d(lines, principal_2d)
    inter_counts = [g.shape[0] for g in inters]
    print(f"Intersections per group: {inter_counts}, total: {sum(inter_counts)}")

    # 5. Render visualizations
    H, W = PANO_RESOLUTION

    # edge_overlay.png — white lines on black
    edge_img = render_sphere_lines(lines, resolution=(H, W))
    cv2.imwrite(str(OUT_DIR / "edge_overlay.png"),
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
    cv2.imwrite(str(OUT_DIR / "grouped_lines.png"),
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
    with open(OUT_DIR / "fgpl_features.json", "w") as f:
        json.dump(features, f)
    print(f"Saved fgpl_features.json ({sum(inter_counts)} intersections)")


if __name__ == "__main__":
    main()
