"""
SAM3 Room Segmentation from Density Images
============================================
Uses Meta's SAM3 with text prompt "floor plan" on CLAHE-inverted density
images to segment room footprints.

Usage:
    # Process a single density image:
    conda run -n sam3 python src/SAM3_room_segmentation.py data/density_image/test_sam3/03264.png

    # Process all images in a directory:
    conda run -n sam3 python src/SAM3_room_segmentation.py data/density_image/test_sam3/

    # Process a density image subfolder (with metadata.json):
    conda run -n sam3 python src/SAM3_room_segmentation.py data/density_image/Area_3_office2
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
OUTPUT_DIR = project_root / "data" / "sam3_room_segmentation"
TEXT_PROMPT = "floor plan"
CONFIDENCE_THRESHOLD = 0.1


# ---------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------

def load_model():
    """Load SAM3 model with lowered confidence threshold."""
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


def preprocess(img_np):
    """CLAHE + invert: density image → floor-plan-like image."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    inverted = 255 - enhanced
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)


def segment(processor, image):
    """Run SAM3 text-prompted segmentation. Returns (masks, scores)."""
    state = processor.set_image(image)
    output = processor.set_text_prompt(prompt=TEXT_PROMPT, state=state)

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


def make_overlay(img_np, masks_binary, alpha=0.5):
    """Create colored mask overlay with contours."""
    overlay = img_np.copy()
    N = masks_binary.shape[0]
    for i in range(N):
        hue = int(180 * i / max(N, 1))
        hsv = np.uint8([[[hue, 200, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        color = (int(bgr[2]), int(bgr[1]), int(bgr[0]))

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


def save_results(stem, raw_np, proc_np, masks, scores, out_dir):
    """Save overlay and 3-panel comparison figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    N = masks.shape[0]

    # Overlay on preprocessed image
    overlay = make_overlay(proc_np, masks)
    Image.fromarray(overlay).save(str(out_dir / f"{stem}_overlay.png"))

    # 3-panel figure: original | preprocessed | segmentation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(raw_np)
    axes[0].set_title("Original Density Image")
    axes[0].axis("off")

    axes[1].imshow(proc_np)
    axes[1].set_title("CLAHE + Inverted")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    score_str = ", ".join(f"{s:.2f}" for s in scores)
    axes[2].set_title(f"SAM3 \"{TEXT_PROMPT}\": {N} masks\nscores: [{score_str}]")
    axes[2].axis("off")

    plt.suptitle(stem, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_dir / f"{stem}_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save individual binary masks
    for i in range(N):
        Image.fromarray(masks[i] * 255).save(str(out_dir / f"{stem}_mask_{i:02d}.png"))


def process_image(processor, img_path, out_dir):
    """Process a single density image."""
    stem = img_path.stem
    print(f"\nProcessing: {stem}")

    raw_image = Image.open(str(img_path)).convert("RGB")
    raw_np = np.array(raw_image)

    proc_np = preprocess(raw_np)
    proc_image = Image.fromarray(proc_np)

    masks, scores = segment(processor, proc_image)
    N = masks.shape[0]

    if N == 0:
        print(f"  No masks detected.")
        return

    print(f"  Masks: {N}")
    for i, s in enumerate(scores):
        print(f"    Mask {i}: score={s:.4f}")

    save_results(stem, raw_np, proc_np, masks, scores, out_dir)
    print(f"  Results saved to: {out_dir}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/SAM3_room_segmentation.py <image_or_directory>")
        print("\nExamples:")
        print("  python src/SAM3_room_segmentation.py data/density_image/test_sam3/03264.png")
        print("  python src/SAM3_room_segmentation.py data/density_image/test_sam3/")
        print("  python src/SAM3_room_segmentation.py data/density_image/Area_3_office2")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.is_absolute():
        target = project_root / target

    processor = load_model()

    if target.is_file() and target.suffix.lower() == ".png":
        # Single image
        process_image(processor, target, OUTPUT_DIR)

    elif target.is_dir():
        # Directory — find all PNG files (skip Zone.Identifier files)
        png_files = sorted([
            p for p in target.iterdir()
            if p.suffix.lower() == ".png" and "Zone.Identifier" not in p.name
        ])
        # Also check for a single PNG named after the directory (subfolder format)
        if not png_files:
            candidate = target / f"{target.name}.png"
            if candidate.exists():
                png_files = [candidate]

        if not png_files:
            print(f"No PNG files found in {target}")
            sys.exit(1)

        print(f"Found {len(png_files)} image(s)")
        for img_path in png_files:
            process_image(processor, img_path, OUTPUT_DIR)
    else:
        print(f"Not found: {target}")
        sys.exit(1)

    print(f"\nDone. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
