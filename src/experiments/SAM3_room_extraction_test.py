"""
SAM3 Room Footprint Extraction — Experiment Script
====================================================
Tests 3 approaches for extracting room footprints from density images
using Meta's SAM3 text-prompted segmentation.

Approach 1: Target walls (bright lines) → post-process to room regions
Approach 2: Target room areas directly
Approach 3: Two-pass hybrid (footprint + walls → rooms)

Usage:
    conda run -n sam3 python src/SAM3_room_extraction_test.py
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sam3_repo_path = project_root / "sam3"

if str(sam3_repo_path) not in sys.path:
    sys.path.insert(0, str(sam3_repo_path))

os.environ["HF_TOKEN"] = "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"

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
TEST_DIR = project_root / "data" / "density_image" / "test_sam3"
OUTPUT_DIR = project_root / "data" / "sam3_room_extraction"

APPROACH1_PROMPTS = ["bright line", "white line", "wall"]
APPROACH2_PROMPTS = ["gray area", "room", "rectangle", "floor plan"]
APPROACH3_PASS1_PROMPTS = ["building", "structure"]
APPROACH3_PASS2_PROMPTS = ["bright line"]

MIN_ROOM_AREA = 50  # minimum pixels for a valid room region
CONFIDENCE_THRESHOLD = 0.1  # lowered from default 0.5 for dark density images

# Image preprocessing modes to test
PREPROCESS_MODES = ["raw", "clahe", "inverted"]


# ---------------------------------------------------------
# SAM3 HELPERS
# ---------------------------------------------------------

def load_model():
    """Load SAM3 model once with lowered confidence threshold."""
    print("Loading SAM3 model...")
    weights_path = sam3_repo_path / "weights" / "sam3.pt"
    if weights_path.exists():
        print(f"  Using local weights: {weights_path}")
        model = build_sam3_image_model(checkpoint_path=str(weights_path))
    else:
        print("  Downloading weights from HuggingFace...")
        model = build_sam3_image_model(load_from_HF=True)
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)
    print(f"  Model loaded (confidence_threshold={CONFIDENCE_THRESHOLD}).")
    return processor


def preprocess_image(img_np, mode):
    """
    Preprocess density image to improve SAM3 detection.
    Returns RGB numpy array.
    """
    if mode == "raw":
        return img_np

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    if mode == "inverted":
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        inverted = 255 - enhanced
        return cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)


def run_sam3(processor, image, prompt):
    """Run SAM3 text-prompted segmentation. Returns (masks_binary, scores)."""
    state = processor.set_image(image)
    output = processor.set_text_prompt(prompt=prompt, state=state)

    masks = output["masks"]
    scores = output["scores"]

    if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
        h, w = np.array(image).shape[:2]
        return np.zeros((0, h, w), dtype=np.uint8), np.array([])

    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = np.array(masks)

    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)

    masks_binary = (masks_np > 0.5).astype(np.uint8)

    if scores is not None:
        if isinstance(scores, torch.Tensor):
            scores_np = scores.cpu().numpy()
        else:
            scores_np = np.array(scores)
    else:
        scores_np = np.ones(masks_binary.shape[0])

    return masks_binary, scores_np


# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------

def get_distinct_color(index, total):
    """Generate a distinct RGB color using HSV spacing."""
    hue = int(180 * index / max(total, 1))
    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def make_overlay(img_np, masks_binary, alpha=0.5):
    """Create colored mask overlay on image with contours."""
    overlay = img_np.copy()
    N = masks_binary.shape[0]
    for i in range(N):
        color = get_distinct_color(i, N)
        mask = masks_binary[i]
        colored = np.zeros_like(img_np)
        colored[mask > 0] = color
        overlay = np.where(
            mask[..., None] > 0,
            (alpha * colored + (1 - alpha) * overlay).astype(np.uint8),
            overlay,
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, (color[2], color[1], color[0]), 2)
        overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay


# ---------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------

def postprocess_walls_to_rooms(wall_mask, img_shape):
    """Wall mask → room regions via flood-fill."""
    h, w = img_shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    walls_dilated = cv2.dilate(wall_mask, kernel, iterations=2)
    inverted = (1 - walls_dilated).astype(np.uint8)

    flood = inverted.copy()
    flood_pad = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flood[seed[1], seed[0]] == 1:
            cv2.floodFill(flood, flood_pad, seed, 0)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flood)
    room_masks = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_ROOM_AREA:
            room_masks.append((labels == i).astype(np.uint8))

    if room_masks:
        return np.stack(room_masks)
    return np.zeros((0, h, w), dtype=np.uint8)


def postprocess_hybrid(footprint_mask, wall_mask):
    """Subtract walls from footprint → room regions."""
    h, w = footprint_mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls_dilated = cv2.dilate(wall_mask, kernel, iterations=1)

    rooms_raw = footprint_mask.copy()
    rooms_raw[walls_dilated > 0] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rooms_raw)
    room_masks = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_ROOM_AREA:
            room_masks.append((labels == i).astype(np.uint8))

    if room_masks:
        return np.stack(room_masks)
    return np.zeros((0, h, w), dtype=np.uint8)


# ---------------------------------------------------------
# PER-APPROACH RUNNERS
# ---------------------------------------------------------

def run_approach1(processor, image, img_np, out_dir):
    """Approach 1: Target walls → post-process to rooms."""
    results = {}

    for prompt in APPROACH1_PROMPTS:
        tag = prompt.replace(" ", "_")
        print(f"  Prompt: '{prompt}'")

        masks, scores = run_sam3(processor, image, prompt)
        n_masks = masks.shape[0]
        print(f"    Masks: {n_masks}, scores: {scores.tolist()}")

        n_rooms = 0
        if n_masks > 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            overlay = make_overlay(img_np, masks)
            Image.fromarray(overlay).save(str(out_dir / f"{tag}_overlay.png"))

            combined_walls = np.any(masks > 0, axis=0).astype(np.uint8)
            room_masks = postprocess_walls_to_rooms(combined_walls, img_np.shape)
            n_rooms = room_masks.shape[0]
            print(f"    Post-processed rooms: {n_rooms}")

            if n_rooms > 0:
                rooms_overlay = make_overlay(img_np, room_masks, alpha=0.4)
                Image.fromarray(rooms_overlay).save(str(out_dir / f"{tag}_rooms.png"))

        results[prompt] = {"n_masks": n_masks, "n_rooms": n_rooms, "scores": scores.tolist()}

    return results


def run_approach2(processor, image, img_np, out_dir):
    """Approach 2: Target room areas directly."""
    results = {}

    for prompt in APPROACH2_PROMPTS:
        tag = prompt.replace(" ", "_")
        print(f"  Prompt: '{prompt}'")

        masks, scores = run_sam3(processor, image, prompt)
        n_masks = masks.shape[0]
        print(f"    Masks: {n_masks}, scores: {scores.tolist()}")

        if n_masks > 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            overlay = make_overlay(img_np, masks)
            Image.fromarray(overlay).save(str(out_dir / f"{tag}_overlay.png"))

        results[prompt] = {"n_masks": n_masks, "scores": scores.tolist()}

    return results


def run_approach3(processor, image, img_np, out_dir):
    """Approach 3: Two-pass hybrid (footprint + walls → rooms)."""
    results = {}

    for p1 in APPROACH3_PASS1_PROMPTS:
        for p2 in APPROACH3_PASS2_PROMPTS:
            tag = f"{p1}+{p2}".replace(" ", "_")
            print(f"  Pass1: '{p1}' + Pass2: '{p2}'")

            masks1, scores1 = run_sam3(processor, image, p1)
            print(f"    Pass 1 masks: {masks1.shape[0]}")

            if masks1.shape[0] == 0:
                print(f"    No footprint, skipping.")
                results[tag] = {"n_fp": 0, "n_walls": 0, "n_rooms": 0}
                continue

            footprint = np.any(masks1 > 0, axis=0).astype(np.uint8)

            masks2, scores2 = run_sam3(processor, image, p2)
            print(f"    Pass 2 masks: {masks2.shape[0]}")

            if masks2.shape[0] == 0:
                room_masks = footprint[np.newaxis, ...]
            else:
                walls = np.any(masks2 > 0, axis=0).astype(np.uint8)
                room_masks = postprocess_hybrid(footprint, walls)

            n_rooms = room_masks.shape[0]
            print(f"    Hybrid rooms: {n_rooms}")

            out_dir.mkdir(parents=True, exist_ok=True)
            # Save footprint overlay
            fp_overlay = make_overlay(img_np, masks1)
            Image.fromarray(fp_overlay).save(str(out_dir / f"{tag}_footprint.png"))

            if n_rooms > 0:
                rooms_overlay = make_overlay(img_np, room_masks, alpha=0.4)
                Image.fromarray(rooms_overlay).save(str(out_dir / f"{tag}_rooms.png"))

            results[tag] = {
                "n_fp": masks1.shape[0],
                "n_walls": masks2.shape[0],
                "n_rooms": n_rooms,
            }

    return results


# ---------------------------------------------------------
# COMPARISON FIGURE
# ---------------------------------------------------------

def create_comparison(img_np, stem, a1_dir, a2_dir, a3_dir, a1_res, a2_res, a3_res, save_path):
    """Create a summary comparison figure. Only called when there are results."""
    a1_items = []
    for prompt in APPROACH1_PROMPTS:
        tag = prompt.replace(" ", "_")
        a1_items.append((prompt, a1_dir / f"{tag}_overlay.png", a1_dir / f"{tag}_rooms.png", a1_res.get(prompt, {})))

    a2_items = []
    for prompt in APPROACH2_PROMPTS:
        tag = prompt.replace(" ", "_")
        a2_items.append((prompt, a2_dir / f"{tag}_overlay.png", a2_res.get(prompt, {})))

    a3_items = []
    for p1 in APPROACH3_PASS1_PROMPTS:
        for p2 in APPROACH3_PASS2_PROMPTS:
            tag = f"{p1}+{p2}".replace(" ", "_")
            a3_items.append((f"{p1}+{p2}", a3_dir / f"{tag}_footprint.png", a3_dir / f"{tag}_rooms.png", a3_res.get(tag, {})))

    max_cols = 1 + max(len(a1_items) * 2, len(a2_items), len(a3_items) * 2)
    fig, axes = plt.subplots(3, max_cols, figsize=(4 * max_cols, 12))

    for row in axes:
        for ax in row:
            ax.axis("off")

    # Row 0: Approach 1
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original", fontsize=10)
    col = 1
    for prompt, overlay_p, rooms_p, res in a1_items:
        nm, nr = res.get("n_masks", 0), res.get("n_rooms", 0)
        if overlay_p.exists():
            axes[0, col].imshow(np.array(Image.open(str(overlay_p))))
        axes[0, col].set_title(f"A1 '{prompt}'\n{nm} masks", fontsize=9)
        col += 1
        if rooms_p.exists():
            axes[0, col].imshow(np.array(Image.open(str(rooms_p))))
        axes[0, col].set_title(f"→ {nr} rooms", fontsize=9)
        col += 1

    # Row 1: Approach 2
    axes[1, 0].imshow(img_np)
    axes[1, 0].set_title("Original", fontsize=10)
    col = 1
    for prompt, overlay_p, res in a2_items:
        nm = res.get("n_masks", 0)
        if overlay_p.exists():
            axes[1, col].imshow(np.array(Image.open(str(overlay_p))))
        axes[1, col].set_title(f"A2 '{prompt}'\n{nm} masks", fontsize=9)
        col += 1

    # Row 2: Approach 3
    axes[2, 0].imshow(img_np)
    axes[2, 0].set_title("Original", fontsize=10)
    col = 1
    for combo, fp_p, rooms_p, res in a3_items:
        nf, nr = res.get("n_fp", 0), res.get("n_rooms", 0)
        if fp_p.exists():
            axes[2, col].imshow(np.array(Image.open(str(fp_p))))
        axes[2, col].set_title(f"A3 '{combo}'\n{nf} fp masks", fontsize=9)
        col += 1
        if rooms_p.exists():
            axes[2, col].imshow(np.array(Image.open(str(rooms_p))))
        axes[2, col].set_title(f"→ {nr} rooms", fontsize=9)
        col += 1

    fig.text(0.01, 0.83, "A1: Walls", fontsize=12, fontweight="bold", va="center", rotation=90)
    fig.text(0.01, 0.50, "A2: Rooms", fontsize=12, fontweight="bold", va="center", rotation=90)
    fig.text(0.01, 0.17, "A3: Hybrid", fontsize=12, fontweight="bold", va="center", rotation=90)

    fig.suptitle(f"SAM3 Room Extraction — {stem}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    test_images = sorted([
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() == ".png" and "Zone.Identifier" not in p.name
    ])

    print(f"Test images ({len(test_images)}):")
    for p in test_images:
        print(f"  {p.name}")

    processor = load_model()

    for img_path in test_images:
        stem = img_path.stem
        raw_image = Image.open(str(img_path)).convert("RGB")
        raw_np = np.array(raw_image)

        for preproc in PREPROCESS_MODES:
            label = f"{stem}/{preproc}"
            print(f"\n{'='*60}")
            print(f"IMAGE: {label}")
            print(f"{'='*60}")

            proc_np = preprocess_image(raw_np, preproc)
            proc_image = Image.fromarray(proc_np)

            img_out = OUTPUT_DIR / stem / preproc
            a1_dir = img_out / "approach1_walls"
            a2_dir = img_out / "approach2_rooms"
            a3_dir = img_out / "approach3_hybrid"

            print(f"\n--- Approach 1: Wall targeting ---")
            a1_res = run_approach1(processor, proc_image, proc_np, a1_dir)

            print(f"\n--- Approach 2: Room area targeting ---")
            a2_res = run_approach2(processor, proc_image, proc_np, a2_dir)

            print(f"\n--- Approach 3: Hybrid ---")
            a3_res = run_approach3(processor, proc_image, proc_np, a3_dir)

            # Check if anything was detected at all
            any_detected = (
                any(r["n_masks"] > 0 for r in a1_res.values())
                or any(r["n_masks"] > 0 for r in a2_res.values())
                or any(r.get("n_fp", 0) > 0 for r in a3_res.values())
            )

            if any_detected:
                # Save preprocessed input and comparison only when there are results
                img_out.mkdir(parents=True, exist_ok=True)
                Image.fromarray(proc_np).save(str(img_out / "preprocessed.png"))
                create_comparison(
                    proc_np, label, a1_dir, a2_dir, a3_dir,
                    a1_res, a2_res, a3_res,
                    img_out / "comparison.png",
                )
            else:
                print(f"\n  No masks detected for any prompt — skipping output.")

            # Summary
            print(f"\n{'='*40}")
            print(f"SUMMARY: {label}")
            print(f"{'='*40}")
            print("Approach 1 (Walls):")
            for p, r in a1_res.items():
                print(f"  '{p}': {r['n_masks']} masks → {r['n_rooms']} rooms")
            print("Approach 2 (Rooms):")
            for p, r in a2_res.items():
                print(f"  '{p}': {r['n_masks']} masks")
            print("Approach 3 (Hybrid):")
            for combo, r in a3_res.items():
                print(f"  '{combo}': {r['n_fp']} fp + {r['n_walls']} walls → {r['n_rooms']} rooms")

    print(f"\n{'='*60}")
    print(f"All results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
