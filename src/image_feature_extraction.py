import cv2
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

# Define directories relative to this script
# Script location: .../scan2measure-webframework/src/
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent

# Target Input Folder (from your specific request)
INPUT_FOLDER_NAME = "BIM_Lab_elevator_room0"
INPUT_DIR = PROJECT_ROOT / "data" / "pano" / "virtual_camera_processed" / INPUT_FOLDER_NAME

# Output Directory
OUTPUT_DIR_ROOT = PROJECT_ROOT / "data" / "pano" / "2d_feature_extracted"
OUTPUT_DIR = OUTPUT_DIR_ROOT / INPUT_FOLDER_NAME

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "step1_cartoon").mkdir(exist_ok=True)
(OUTPUT_DIR / "step2_wireframe").mkdir(exist_ok=True)
(OUTPUT_DIR / "step2_overlay").mkdir(exist_ok=True)

# Algorithm Parameters
# Step 1: Anisotropic Diffusion (The "Cartoonizer")
# alpha: Integration constant (speed of diffusion).
# K: Gradient sensitivity. Lower = preserves more edges, Higher = smooths more.
# iterations: How strong the effect is.
DIFFUSION_ALPHA = 0.1
DIFFUSION_K = 15
DIFFUSION_ITERS = 20

# Step 2: Line Segment Detector (LSD)
# Scale=0.8 downsamples slightly to reduce high-freq noise before detection
LSD_SCALE = 0.8 

def main():
    # 1. Validation
    if not INPUT_DIR.exists():
        print(f"[Error] Input directory not found: {INPUT_DIR}")
        return

    # Get list of images
    image_files = sorted(list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png")))
    if not image_files:
        print(f"[Error] No images found in {INPUT_DIR}")
        return

    print(f"--- 2D Feature Extraction Pipeline ---")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Processing {len(image_files)} images...")

    # Initialize LSD (Line Segment Detector)
    # STANDARD mode is generally best for geometric features
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, scale=LSD_SCALE)

    # ... (inside the loop)
    for img_path in tqdm(image_files, desc="Extracting Features"):
        
        # FIX 1: Load as COLOR (required for anisotropicDiffusion)
        img_color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_color is None:
            continue

        # ==========================================
        # STEP 1: Anisotropic Diffusion (Pre-processing)
        # ==========================================
        try:
            # Now passing a 3-channel color image
            diffused_color = cv2.ximgproc.anisotropicDiffusion(
                src=img_color,
                alpha=DIFFUSION_ALPHA,
                K=DIFFUSION_K,
                niters=DIFFUSION_ITERS
            )
        except AttributeError:
            print("\n[Error] 'cv2.ximgproc' not found.")
            sys.exit(1)

        # FIX 2: Convert to Grayscale for LSD (LSD requires single channel)
        diffused_gray = cv2.cvtColor(diffused_color, cv2.COLOR_BGR2GRAY)

        # ==========================================
        # STEP 2: LSD Feature Extraction
        # ==========================================
        # Detect lines in the grayscale version
        lines, width, prec, nfa = lsd.detect(diffused_gray)

        # Create output visualizations
        
        # A. Wireframe Map (Synthetic Style)
        wireframe_map = np.zeros_like(diffused_gray)
        
        # B. Overlay (Debug) - Draw on the color cartoon image
        overlay_map = diffused_color.copy()

        if lines is not None:
            lsd.drawSegments(wireframe_map, lines)
            
            for line in lines:
                x1, y1, x2, y2 = map(int, line[0])
                cv2.line(overlay_map, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ==========================================
        # SAVE RESULTS
        # ==========================================
        fname = img_path.name
        
        # Save Step 1: The "Cartoon" Image (Color looks better for inspection)
        cv2.imwrite(str(OUTPUT_DIR / "step1_cartoon" / fname), diffused_color)
        
        # Save Step 2: The "Wireframe" 
        cv2.imwrite(str(OUTPUT_DIR / "step2_wireframe" / fname), wireframe_map)
        
        # Save Step 2: Debug Overlay
        cv2.imwrite(str(OUTPUT_DIR / "step2_overlay" / fname), overlay_map)
        
    print(f"\n[Success] Processing complete.")
    print(f"  - Cartoonized images: {OUTPUT_DIR / 'step1_cartoon'}")
    print(f"  - Wireframes (Black/White): {OUTPUT_DIR / 'step2_wireframe'}")

if __name__ == "__main__":
    main()