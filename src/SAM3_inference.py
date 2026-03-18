import sys
import os
from pathlib import Path
import torch
from PIL import Image

# ---------------------------------------------------------
# RELATIVE PATH SETUP
# ---------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sam3_repo_path = project_root / "sam3"

if str(sam3_repo_path) not in sys.path:
    sys.path.append(str(sam3_repo_path))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Define paths
image_dir = project_root / "data" / "test" 
image_path = image_dir / "tennis_player.jpg"
weights_path = sam3_repo_path / "weights" / "sam3.pt"

# ---------------------------------------------------------
# 4 SIMPLE STEPS FROM THE GUIDE
# ---------------------------------------------------------

print("Step 1: Loading model...")
# Load the model
model = build_sam3_image_model(checkpoint_path=str(weights_path))
processor = Sam3Processor(model)

print("Step 2: Loading image...")
# Load an image 
image = Image.open(str(image_path))
inference_state = processor.set_image(image)

print("Step 3: Prompting model...")
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="segment the tennis player")

print("Step 4: Extracting results...")
# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# Print the raw results so you can inspect what the model returns
print("\n--- RAW OUTPUT ---")
if masks is not None and len(masks) > 0:
    print(f"Number of objects found: {len(masks)}")
    print(f"Mask tensor shape: {masks.shape}")
    print(f"Confidence scores: {scores}")
else:
    print("No objects were found by the model.")