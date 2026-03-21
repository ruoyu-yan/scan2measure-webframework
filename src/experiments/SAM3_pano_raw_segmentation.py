"""
SAM3 Raw Segmentation on Panoramic Images
==========================================
Runs SAM3 separately for each text prompt on pano images and saves
a combined comparison with distinct colors per component.

Usage:
    conda run -n sam3 python src/SAM3_pano_raw_segmentation.py
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
INPUT_DIR = project_root / "data" / "sam3_pano_processing"
OUTPUT_DIR = INPUT_DIR / "raw_segmentation"
CONFIDENCE_THRESHOLD = 0.5

# Each prompt is run separately; fixed color per component (RGB)
PROMPTS = {
    "floor":   (230, 25, 75),    # red
    "wall":    (60, 180, 75),    # green
    "ceiling": (180, 120, 255),  # purple
    "door":    (255, 225, 25),   # yellow
    "column":  (0, 130, 200),    # blue
}


def load_model():
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


def segment(processor, image, prompt):
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
        scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
    else:
        scores_np = np.ones(masks_binary.shape[0])

    return masks_binary, scores_np


def make_combined_overlay(img_np, prompt_results, alpha=0.45):
    """Overlay all masks with fixed color per prompt. Later prompts paint over earlier ones."""
    overlay = img_np.copy()

    for prompt, (masks, scores, color) in prompt_results.items():
        if masks.shape[0] == 0:
            continue
        # Combine all masks for this prompt
        combined = np.any(masks, axis=0).astype(np.uint8)

        colored = np.zeros_like(img_np)
        colored[combined > 0] = color
        overlay = np.where(
            combined[..., None] > 0,
            (alpha * colored + (1 - alpha) * overlay).astype(np.uint8),
            overlay,
        )
        # Draw contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, (color[2], color[1], color[0]), 2)
        overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return overlay


def process_image(processor, img_path):
    stem = img_path.stem
    print(f"\nProcessing: {stem}")

    image = Image.open(str(img_path)).convert("RGB")
    img_np = np.array(image)

    # Run each prompt separately
    prompt_results = {}
    for prompt, color in PROMPTS.items():
        masks, scores = segment(processor, image, prompt)
        prompt_results[prompt] = (masks, scores, color)
        n = masks.shape[0]
        if n > 0:
            score_str = ", ".join(f"{s:.2f}" for s in scores)
            print(f"  '{prompt}': {n} mask(s), scores=[{score_str}]")
        else:
            print(f"  '{prompt}': no masks")

    # Combined overlay
    overlay = make_combined_overlay(img_np, prompt_results)

    # Save 2-panel comparison with legend
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Pano")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    # Build legend
    legend_patches = []
    for prompt, color in PROMPTS.items():
        n = prompt_results[prompt][0].shape[0]
        rgb_norm = tuple(c / 255.0 for c in color)
        legend_patches.append(mpatches.Patch(color=rgb_norm, label=f"{prompt} ({n})"))
    axes[1].legend(handles=legend_patches, loc="lower right", fontsize=10)
    axes[1].set_title("SAM3 per-component segmentation")
    axes[1].axis("off")

    plt.suptitle(stem, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / f"{stem}_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / f'{stem}_comparison.png'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jpg_files = sorted([
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg")
        and "Zone.Identifier" not in p.name
    ])

    if not jpg_files:
        print(f"No JPG files found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(jpg_files)} pano(s)")
    print(f"Prompts: {list(PROMPTS.keys())}")
    print(f"Output: {OUTPUT_DIR}")

    processor = load_model()

    for img_path in jpg_files:
        process_image(processor, img_path)

    print(f"\nDone. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
