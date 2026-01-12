import argparse
import os
import sys
import torch
import numpy as np
import cv2
import json
from PIL import Image
from pathlib import Path

# New import for Area filtering
from shapely.geometry import Polygon

# -------------------------------------------------------------------------
# 0. PATH SETUP
# -------------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent # scan2measure-webframework/
roomformer_root = project_root / 'RoomFormer'

if str(roomformer_root) not in sys.path:
    sys.path.append(str(roomformer_root))

try:
    from models import build_model
    # Import the official visualization utils to mirror engine.py style
    from util.plot_utils import plot_room_map, plot_floorplan_with_regions
except ImportError:
    print(f"Error: Could not find RoomFormer modules at {roomformer_root}")
    sys.exit(1)

# -------------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser('RoomFormer Custom Inference')
    
    # --- PATHS ---
    default_input = project_root / 'data' / 'density_image'
    default_output = project_root / 'data' / 'reconstructed_floorplans_RoomFormer'
    default_ckpt = roomformer_root / 'checkpoints' / 'roomformer_stru3d.pth'

    parser.add_argument('--input_dir', type=str, default=str(default_input))
    parser.add_argument('--output_dir', type=str, default=str(default_output))
    parser.add_argument('--checkpoint', type=str, default=str(default_ckpt))

    # --- MODEL PARAMS ---
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=800, type=int)
    parser.add_argument('--num_polys', default=20, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--query_pos_type', default='sine', type=str)
    parser.add_argument('--with_poly_refine', default=True, action='store_true')
    parser.add_argument('--masked_attn', default=False, action='store_true')
    parser.add_argument('--semantic_classes', default=-1, type=int)
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- TRAINING ARGS ---
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--position_embedding_scale', default=6.28318530718, type=float)
    
    # --- MATCHING/LOSS ARGS ---
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_coords', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--room_cls_loss_coef', default=0.2, type=float)
    parser.add_argument('--coords_loss_coef', default=5, type=float)
    parser.add_argument('--raster_loss_coef', default=1, type=float)

    # --- INFERENCE PARAMS ---
    parser.add_argument('--prob_threshold', default=0.5, type=float) 

    return parser.parse_args()

# -------------------------------------------------------------------------
# 2. PREPROCESSING
# -------------------------------------------------------------------------
def preprocess_image(image_path):
    img_pil = Image.open(image_path)
    img_np = np.array(img_pil)

    if len(img_np.shape) == 3:
        img_np = np.array(img_pil.convert('L')) 
    
    h, w = img_np.shape
    img_tensor = torch.as_tensor(np.expand_dims(img_np, 0)).float() / 255.0
    
    return img_tensor, (h, w), np.array(img_pil.convert("RGB"))

# -------------------------------------------------------------------------
# 3. VISUALIZATION (USING OFFICIAL UTILS)
# -------------------------------------------------------------------------
def draw_results(img_rgb, outputs, threshold=0.5):
    h, w, _ = img_rgb.shape
    
    # Get Logits & Coords
    pred_logits = outputs['pred_logits'] # [1, 20, 40]
    pred_coords = outputs['pred_coords'] # [1, 20, 40, 2]

    # Convert to probabilities
    probs = pred_logits.sigmoid().detach().cpu().numpy() 
    coords = pred_coords.detach().cpu().numpy()          

    # Process first batch item
    batch_probs = probs[0]   
    batch_coords = coords[0] 

    # Collect valid room polygons
    room_polys = []

    for room_idx in range(batch_probs.shape[0]):
        room_scores = batch_probs[room_idx]      
        room_points = batch_coords[room_idx]     
        
        # --- FILTER 1: Score Threshold ---
        valid_mask = room_scores > threshold
        valid_points = room_points[valid_mask]   
        
        # --- FILTER 2: Minimum Corners ---
        if len(valid_points) >= 4:
            
            # --- FILTER 3: Area Check ---
            # Scale to 255 to check area (consistent with engine.py logic)
            check_poly = valid_points * 255.0
            
            if Polygon(check_poly).area >= 100:
                
                # Scale to ACTUAL image size for final output
                scaled_poly = valid_points.copy()
                scaled_poly[:, 0] *= w
                scaled_poly[:, 1] *= h
                
                # Append as integer numpy array (required by utils)
                room_polys.append(scaled_poly.astype(np.int32))

    print(f"  > Found {len(room_polys)} valid rooms.")

    # --- GENERATE IMAGES USING OFFICIAL UTILS ---
    
    # 1. Floorplan Map (Binary/Colored Regions)
    # scale=1000 is what engine.py uses, though for non-semantic it might just be black/white or random colors
    floorplan_img = plot_floorplan_with_regions(room_polys, scale=1000) 

    # 2. Room Map (Overlay on Density Image)
    # Initialize empty map (H, W, 3)
    pred_room_map = np.zeros([h, w, 3])
    
    for room_poly in room_polys:
        # This util draws the green lines and blue corners
        pred_room_map = plot_room_map(room_poly, pred_room_map)

    # Overlay: clip(prediction + original_density)
    # Note: img_rgb is already 0-255 uint8. 
    room_map_img = np.clip(pred_room_map + img_rgb, 0, 255).astype(np.uint8)

    return room_map_img, floorplan_img, room_polys

# -------------------------------------------------------------------------
# 4. MAIN
# -------------------------------------------------------------------------
def main():
    args = get_args()
    device = torch.device(args.device)

    # Hard-coded single image inference (edit this filename as needed)
    target_image_name = "Area_3_selected_rooms.png"
    
    print(f"--- RoomFormer Inference ---")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    
    model = build_model(args, train=False)
    model.to(device)
    model.eval()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    print(f"Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(target_image_name).stem

    # New layout: data/density_image/<base_name>/<base_name>.png
    img_path = input_dir / base_name / target_image_name
    if not img_path.exists():
        # Backward-compatible fallback (old layout): data/density_image/<base_name>.png
        fallback_path = input_dir / target_image_name
        if fallback_path.exists():
            print(f"Warning: Using legacy input path: {fallback_path}")
            img_path = fallback_path
        else:
            print(f"Error: Target image not found: {img_path}")
            return

    print(f"Processing 1 image: {img_path.name}")

    img_tensor, (h, w), img_vis = preprocess_image(img_path)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    room_map_img, floorplan_img, room_polys = draw_results(
        img_vis, outputs, threshold=args.prob_threshold
    )

    per_image_output_dir = output_dir / base_name
    per_image_output_dir.mkdir(parents=True, exist_ok=True)

    path_room_map = per_image_output_dir / f"{base_name}_pred_room_map.png"
    path_floorplan = per_image_output_dir / f"{base_name}_pred_floorplan.png"
    path_predictions = per_image_output_dir / "predictions.json"

    cv2.imwrite(str(path_room_map), cv2.cvtColor(room_map_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(path_floorplan), cv2.cvtColor(floorplan_img, cv2.COLOR_RGB2BGR))

    with open(path_predictions, "w", encoding="utf-8") as f:
        json.dump([poly.tolist() for poly in room_polys], f, indent=2)

    print(f"Saved: {base_name}")

    print("Done!")

if __name__ == '__main__':
    main()