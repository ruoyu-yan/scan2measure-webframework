import cv2
import numpy as np
import os
import sys
import json
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
INPUT_FOLDER_NAME = "TMB_office1"
INPUT_DIR = PROJECT_ROOT / "data" / "pano" / "virtual_camera_processed" / INPUT_FOLDER_NAME

# Output Directory
OUTPUT_DIR_ROOT = PROJECT_ROOT / "data" / "pano" / "2d_feature_extracted"
OUTPUT_DIR = OUTPUT_DIR_ROOT / INPUT_FOLDER_NAME

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "step0_contrast_boost").mkdir(exist_ok=True)
(OUTPUT_DIR / "step1_cartoon").mkdir(exist_ok=True)
(OUTPUT_DIR / "step2_wireframe").mkdir(exist_ok=True)
(OUTPUT_DIR / "step3_overlay").mkdir(exist_ok=True)

# Algorithm Parameters
# Step 1: Anisotropic Diffusion (The "Cartoonizer")
# alpha: Integration constant (speed of diffusion).
# K: Gradient sensitivity. Lower = preserves more edges, Higher = smooths more.
# iterations: How strong the effect is.
DIFFUSION_ALPHA = 0.1
DIFFUSION_K = 30
DIFFUSION_ITERS = 30

# Step 2: Line Segment Detector (LSD)
# Scale=0.8 downsamples slightly to reduce high-freq noise before detection
LSD_SCALE = 0.8
MIN_LINE_LENGTH = 30

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

    # Dictionary to store extracted line coordinates for all images
    all_extracted_lines = {}

    # ... (inside the loop)
    for img_path in tqdm(image_files, desc="Extracting Features"):
        
        # FIX 1: Load as COLOR (required for anisotropicDiffusion)
        img_color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_color is None:
            continue

        # ==========================================
        # STEP 0: CLAHE Contrast Enhancement
        # ==========================================
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to the L (Lightness) channel
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Merge enhanced L with original A and B channels
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        
        # Convert back to BGR
        contrast_boosted = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Save contrast-boosted image
        stem = img_path.stem
        cv2.imwrite(str(OUTPUT_DIR / "step0_contrast_boost" / f"{stem}_contrast_boost.jpg"), contrast_boosted)

        # ==========================================
        # STEP 1: Anisotropic Diffusion (Pre-processing)
        # ==========================================
        try:
            # Now passing the contrast-boosted image
            diffused_color = cv2.ximgproc.anisotropicDiffusion(
                src=contrast_boosted,
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

        # ==========================================
        # Border Artifact Filter
        # ==========================================
        h, w = diffused_gray.shape[:2]
        border_margin = 5
        
        if lines is not None:
            filtered_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Reject Left Edge artifact
                if x1 < border_margin and x2 < border_margin:
                    continue
                # Reject Right Edge artifact
                if x1 > w - border_margin and x2 > w - border_margin:
                    continue
                # Reject Top Edge artifact
                if y1 < border_margin and y2 < border_margin:
                    continue
                # Reject Bottom Edge artifact
                if y1 > h - border_margin and y2 > h - border_margin:
                    continue
                
                # Reject short lines (noise filter)
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length < MIN_LINE_LENGTH:
                    continue
                
                filtered_lines.append(line)
            
            lines = np.array(filtered_lines) if filtered_lines else None

        # Extract line coordinates for JSON export
        image_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                image_lines.append([[float(x1), float(y1)], [float(x2), float(y2)]])
        all_extracted_lines[img_path.name] = image_lines

        # Create output visualizations
        
        # A. Wireframe Map (Synthetic Style)
        wireframe_map = np.zeros_like(diffused_gray)
        
        # B. Overlay (Debug) - Draw on the color cartoon image
        overlay_map = diffused_color.copy()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line[0])
                cv2.line(wireframe_map, (x1, y1), (x2, y2), 255, 1)
            
            for line in lines:
                x1, y1, x2, y2 = map(int, line[0])
                cv2.line(overlay_map, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ==========================================
        # SAVE RESULTS
        # ==========================================
        
        # Save Step 1: The "Cartoon" Image (Color looks better for inspection)
        cv2.imwrite(str(OUTPUT_DIR / "step1_cartoon" / f"{stem}_cartonized.jpg"), diffused_color)
        
        # Save Step 2: The "Wireframe" 
        cv2.imwrite(str(OUTPUT_DIR / "step2_wireframe" / f"{stem}_wireframe.jpg"), wireframe_map)
        
        # Save Step 3: Debug Overlay
        cv2.imwrite(str(OUTPUT_DIR / "step3_overlay" / f"{stem}_overlay.jpg"), overlay_map)

    # Save extracted line coordinates to JSON
    json_output_path = OUTPUT_DIR / "extracted_2d_lines.json"
    with open(json_output_path, "w") as f:
        json.dump(all_extracted_lines, f, indent=4)

    print(f"\n[Success] Processing complete.")
    print(f"  - Cartoonized images: {OUTPUT_DIR / 'step1_cartoon'}")
    print(f"  - Wireframes (Black/White): {OUTPUT_DIR / 'step2_wireframe'}")
    print(f"  - Line coordinates JSON: {json_output_path}")

if __name__ == "__main__":
    main()