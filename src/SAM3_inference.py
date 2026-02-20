import sys
import os
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------
# 1. RELATIVE PATH SETUP
# ---------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sam3_repo_path = project_root / "sam3"

if str(sam3_repo_path) not in sys.path:
    sys.path.append(str(sam3_repo_path))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def main():
    # ---------------------------------------------------------
    # 2. DEFINE FILE PATHS
    # ---------------------------------------------------------
    image_dir = project_root / "data" / "density_image" / "tmb_office_one_corridor_dense"
    image_path = image_dir / "tmb_office_one_corridor_dense.png"
    weights_path = sam3_repo_path / "weights" / "sam3.pt"
    output_path = image_dir / "tmb_office_one_corridor_dense_segmented.png"

    # ---------------------------------------------------------
    # 3. RUN MODEL
    # ---------------------------------------------------------
    print(f"Loading SAM 3 model from: {weights_path}")
    model = build_sam3_image_model(checkpoint_path=str(weights_path))
    processor = Sam3Processor(model)

    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert("RGB")
    inference_state = processor.set_image(image)

    # ---------------------------------------------------------
    # 4. DESCRIPTIVE PROMPTS FOR DENSITY MAPS
    # ---------------------------------------------------------
    # Since the image is abstract, we try a few visual descriptions
    prompts_to_try = [
        "bright white lines and boundaries representing walls",
        "thick white outlines of the floorplan",
        "bright illuminated lines on the dark background"
    ]

    masks = None
    successful_prompt = ""

    for prompt in prompts_to_try:
        print(f"Trying prompt: '{prompt}'...")
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        
        current_masks = output["masks"]
        
        # Check if the model returned any mask data
        if current_masks is not None and len(current_masks) > 0:
            # Convert tensor to numpy to check for actual True values
            if isinstance(current_masks, torch.Tensor):
                mask_np = current_masks.detach().cpu().numpy()
            else:
                mask_np = current_masks
                
            if np.any(mask_np):
                print(f"  -> Success! Found structures using prompt: '{prompt}'")
                masks = mask_np
                successful_prompt = prompt
                break
        
        print("  -> No structures found. Trying next prompt...")

    if masks is None or not np.any(masks):
        print("\nFailed to detect walls with any of the text prompts.")
        return

    # ---------------------------------------------------------
    # 5. PROCESS & SAVE IMAGE
    # ---------------------------------------------------------
    print("Processing masks and generating visualization...")
    img_np = np.array(image)
    
    # Collapse dimensions to get a single 2D boolean mask
    if masks.ndim > 2:
        combined_mask = np.any(masks, axis=tuple(range(masks.ndim - 2)))
    else:
        combined_mask = masks

    # Create green overlay
    overlay = np.zeros_like(img_np, dtype=np.uint8)
    overlay[:] = [0, 255, 0]

    # Blend original image with the green mask
    alpha = 0.5 
    segmented_img = img_np.copy()
    segmented_img[combined_mask] = cv2.addWeighted(
        img_np[combined_mask], 1 - alpha, 
        overlay[combined_mask], alpha, 0
    )

    cv2.imwrite(str(output_path), cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
    print(f"Success! Segmented image saved to: {output_path}")

if __name__ == "__main__":
    main()