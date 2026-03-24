"""
SAM3 Inference — Meta's Segment Anything Model 3
=================================================
Text-prompted segmentation using Meta's official SAM3 library.
Segments objects from an image and saves mask overlays.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# RELATIVE PATH SETUP
# ---------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sam3_repo_path = project_root / "sam3"

if str(sam3_repo_path) not in sys.path:
    sys.path.insert(0, str(sam3_repo_path))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
image_path = project_root / "tennis_player.jpg"
output_dir = project_root / "data" / "sam3_output"
output_dir.mkdir(parents=True, exist_ok=True)

TEXT_PROMPT = "tennis player"

# ---------------------------------------------------------
# STEP 1: Load model
# ---------------------------------------------------------
print("Step 1: Loading SAM3 model...")
weights_path = sam3_repo_path / "weights" / "sam3.pt"
if weights_path.exists():
    print(f"  Using local weights: {weights_path}")
    model = build_sam3_image_model(checkpoint_path=str(weights_path))
else:
    print("  Downloading weights from HuggingFace...")
    model = build_sam3_image_model(load_from_HF=True)

processor = Sam3Processor(model)
print("  Model loaded successfully.")

# ---------------------------------------------------------
# STEP 2: Load image
# ---------------------------------------------------------
print(f"Step 2: Loading image: {image_path}")
image = Image.open(str(image_path)).convert("RGB")
img_w, img_h = image.size
print(f"  Image size: {img_w} x {img_h}")
inference_state = processor.set_image(image)

# ---------------------------------------------------------
# STEP 3: Run text-prompted segmentation
# ---------------------------------------------------------
print(f'Step 3: Running segmentation with prompt: "{TEXT_PROMPT}"')
output = processor.set_text_prompt(prompt=TEXT_PROMPT, state=inference_state)

masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

# ---------------------------------------------------------
# STEP 4: Process and save results
# ---------------------------------------------------------
print("\nStep 4: Processing results...")

if masks is None or len(masks) == 0:
    print("No objects were found by the model.")
    sys.exit(0)

# Convert masks to numpy: shape (N, 1, H, W) or (N, H, W) -> (N, H, W)
if isinstance(masks, torch.Tensor):
    masks_np = masks.cpu().numpy()
else:
    masks_np = np.array(masks)

if masks_np.ndim == 4:
    masks_np = masks_np.squeeze(1)

masks_binary = (masks_np > 0.5).astype(np.uint8)

print(f"  Objects found: {masks_binary.shape[0]}")
print(f"  Mask shape: {masks_binary.shape}")
if scores is not None:
    if isinstance(scores, torch.Tensor):
        scores_np = scores.cpu().numpy()
    else:
        scores_np = np.array(scores)
    print(f"  Confidence scores: {scores_np}")

# --- Save individual masks ---
img_np = np.array(image)  # RGB

for i in range(masks_binary.shape[0]):
    mask = masks_binary[i]
    mask_path = output_dir / f"mask_{i:02d}.png"
    Image.fromarray(mask * 255).save(str(mask_path))
    print(f"  Saved mask {i}: {mask_path}")

# --- Save overlay visualization ---
overlay = img_np.copy()
N = masks_binary.shape[0]
for i in range(N):
    mask = masks_binary[i]
    # Generate distinct color per mask
    hue = int(180 * i / max(N, 1))
    # Use a semi-transparent colored overlay
    color = np.array([
        int(255 * ((hue * 67) % 255) / 255),
        int(255 * ((hue * 137 + 80) % 255) / 255),
        int(255 * ((hue * 37 + 160) % 255) / 255),
    ], dtype=np.uint8)
    colored_mask = np.zeros_like(img_np)
    colored_mask[mask > 0] = color
    overlay = np.where(
        mask[..., None] > 0,
        (0.5 * overlay + 0.5 * colored_mask).astype(np.uint8),
        overlay,
    )
    # Draw contour
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color.tolist(), 2)

overlay_path = output_dir / "segmentation_overlay.png"
Image.fromarray(overlay).save(str(overlay_path))
print(f"\n  Saved overlay: {overlay_path}")

# --- Save masked cutout (original pixels, transparent background) ---
if N > 0:
    # Combine all masks into one
    combined_mask = np.any(masks_binary > 0, axis=0).astype(np.uint8)
    rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    rgba[..., :3] = img_np
    rgba[..., 3] = combined_mask * 255
    cutout_path = output_dir / "segmentation_cutout.png"
    Image.fromarray(rgba).save(str(cutout_path))
    print(f"  Saved cutout (transparent bg): {cutout_path}")

print(f"\nDone. All outputs saved to: {output_dir}")
