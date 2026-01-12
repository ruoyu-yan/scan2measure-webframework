import sys
import os
import argparse
import numpy as np
import torch
import cv2
import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. PATH SETUP
# ==========================================
current_script_dir = os.path.dirname(os.path.abspath(__file__))
lgt_net_root = os.path.abspath(os.path.join(current_script_dir, "..", "LGT-Net"))

if not os.path.exists(lgt_net_root):
    raise FileNotFoundError(f"Could not find LGT-Net at: {lgt_net_root}")

sys.path.append(lgt_net_root)

# ==========================================
# 2. IMPORTS
# ==========================================
from config.defaults import get_config
from models.build import build_model
from utils.logger import get_logger
from utils.misc import tensor2np_d, tensor2np
from utils.boundary import corners2boundaries
from utils.writer import xyz2json
from postprocessing.post_process import post_process
from preprocessing.pano_lsd_align import panoEdgeDetection, rotatePanorama
from utils.conversion import depth2xyz
from evaluation.accuracy import show_grad
from loss import GradLoss
from visualization.boundary import draw_boundaries
from visualization.floorplan import draw_floorplan, draw_iou_floorplan

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def preprocess_manhattan(img, vp_cache_path=None):
    # CRITICAL: This function expects 0-255 range image (float or uint8 is fine, but range must be 0-255)
    
    if vp_cache_path and os.path.exists(vp_cache_path):
        with open(vp_cache_path) as f:
            vp = [[float(v) for v in line.rstrip().split(' ')] for line in f.readlines()]
            vp = np.array(vp)
    else:
        # qError=0.7, refineIter=3 are defaults from LGT-Net
        _, vp, _, _, _, _, _ = panoEdgeDetection(img, qError=0.7, refineIter=3)
    
    # rotatePanorama returns a float64 image in range 0-255
    i_img = rotatePanorama(img, vp[2::-1])

    if vp_cache_path is not None:
        with open(vp_cache_path, 'w') as f:
            for i in range(3):
                f.write('%.6f %.6f %.6f\n' % (vp[i, 0], vp[i, 1], vp[i, 2]))

    return i_img, vp

def show_depth_normal_grad(dt):
    grad_conv = GradLoss().to(dt['depth'].device).grad_conv
    dt_grad_img = show_grad(dt['depth'][0], grad_conv, 50)
    dt_grad_img = cv2.resize(dt_grad_img, (1024, 60), interpolation=cv2.INTER_NEAREST)
    return dt_grad_img

def show_alpha_floorplan(dt_xyz, side_l=512, border_color=None):
    if border_color is None:
        border_color = [1, 0, 0, 1]
    fill_color = [0.2, 0.2, 0.2, 0.2]
    dt_floorplan = draw_floorplan(xz=dt_xyz[..., ::2], fill_color=fill_color,
                                  border_color=border_color, side_l=side_l, show=False, center_color=[1, 0, 0, 1])
    dt_floorplan = Image.fromarray((dt_floorplan * 255).astype(np.uint8), mode='RGBA')
    back = np.zeros([side_l, side_l, len(fill_color)], dtype=float)
    back[..., :] = [0.8, 0.8, 0.8, 1]
    back = Image.fromarray((back * 255).astype(np.uint8), mode='RGBA')
    iou_floorplan = Image.alpha_composite(back, dt_floorplan).convert("RGB")
    dt_floorplan = np.array(iou_floorplan) / 255.0
    return dt_floorplan

def visualize_2d_complex(img, dt, show_depth=True, show_floorplan=True, save_path=None):
    # Input 'img' MUST be 0-1 float
    dt_np = tensor2np_d(dt)
    dt_depth = dt_np['depth'][0]
    dt_xyz = depth2xyz(np.abs(dt_depth))
    dt_ratio = dt_np['ratio'][0][0]
    
    # 1. Draw Green Lines
    dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt_xyz, step=None, visible=False, length=img.shape[1])
    vis_img = draw_boundaries(img, boundary_list=dt_boundaries, boundary_color=[0, 1, 0])

    # 2. Draw Red Lines
    if 'processed_xyz' in dt:
        dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt['processed_xyz'][0], step=None, visible=False,
                                           length=img.shape[1])
        vis_img = draw_boundaries(vis_img, boundary_list=dt_boundaries, boundary_color=[1, 0, 0])

    # 3. Add Bottom Strip
    if show_depth:
        dt_grad_img = show_depth_normal_grad(dt)
        grad_h = dt_grad_img.shape[0]
        vis_merge = [
            vis_img[0:-grad_h, :, :],
            dt_grad_img,
        ]
        vis_img = np.concatenate(vis_merge, axis=0)

    # 4. Add Floorplan
    if show_floorplan:
        if 'processed_xyz' in dt:
            floorplan = draw_iou_floorplan(dt['processed_xyz'][0][..., ::2], dt_xyz[..., ::2],
                                           dt_board_color=[1, 0, 0, 1], gt_board_color=[0, 1, 0, 1])
        else:
            floorplan = show_alpha_floorplan(dt_xyz, border_color=[0, 1, 0, 1])

        vis_img = np.concatenate([vis_img, floorplan[:, 60:-60, :]], axis=1)

    # 5. Save (Convert 0-1 back to 0-255 uint8)
    if save_path:
        result = Image.fromarray((vis_img * 255).astype(np.uint8))
        result.save(save_path)
        print(f"Saved Visualization: {save_path}")
    
    return vis_img

def save_results(output_folder, name, img, dt, output_xyz):
    # img passed here must be 0-1 float
    
    # 1. Save JSON
    json_path = os.path.join(output_folder, f"{name}_pred.json")
    ratio = tensor2np(dt['ratio'][0])[0]
    json_data = xyz2json(output_xyz, ratio)
    with open(json_path, 'w') as f:
        import json
        f.write(json.dumps(json_data, indent=4) + '\n')
    print(f"Saved JSON: {json_path}")

    # 2. Save Visualization
    pred_path = os.path.join(output_folder, f"{name}_pred.jpg")
    visualize_2d_complex(img, dt, show_depth=True, show_floorplan=True, save_path=pred_path)

    # 3. Save Aligned Image (Clean)
    aligned_path = os.path.join(output_folder, f"{name}_aligned.jpg")
    img_bgr = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite(aligned_path, (img_bgr * 255).astype(np.uint8))

# ==========================================
# 4. MAIN INFERENCE LOGIC
# ==========================================
def main():
    # --- USER CONFIGURATION ---
    target_image_name = "corridor2.jpg" 
    
    # --- DIRECTORY SETUP ---
    base_data_dir = os.path.abspath(os.path.join(current_script_dir, "..", "data", "pano"))
    input_dir = os.path.join(base_data_dir, "input")
    input_image_path = os.path.join(input_dir, target_image_name)
    
    if not os.path.exists(input_image_path):
        print(f"[Error] Input file not found: {input_image_path}")
        return

    image_stem = os.path.splitext(target_image_name)[0]
    output_dir = os.path.join(base_data_dir, "output", image_stem)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing: {input_image_path}")
    print(f"Output to:  {output_dir}")

    # --- LGT-Net Configuration ---
    config_file = os.path.join(lgt_net_root, "src", "config", "zind.yaml")

    class MockArgs:
        def __contains__(self, key):
            return hasattr(self, key)
            
        cfg = config_file
        mode = 'test'
        debug = False
        hidden_bar = False
        bs = 1 
        save_eval = False
        val_name = None
        post_processing = 'manhattan'
        need_cpe = False
        need_f1 = False
        need_rmse = False
        force_cube = False
        wall_num = None

    config = get_config(MockArgs())

    config.defrost()
    config.CKPT.ROOT = os.path.join(lgt_net_root, "checkpoints")
    decoder = config.MODEL.ARGS[0]['decoder_name']
    output_name = config.MODEL.ARGS[0]['output_name']
    config.CKPT.DIR = os.path.join(config.CKPT.ROOT, f"{decoder}_{output_name}_Net", config.TAG)
    config.freeze()

    logger = get_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": print("WARNING: Using CPU.")
    
    config.defrost()
    config.TRAIN.DEVICE = device
    config.freeze()

    model, _, _, _ = build_model(config, logger)
    model.eval()

    # --- Run Inference ---
    # 1. Load Image as UINT8 (0-255) to satisfy LSD detector requirements
    pil_img = Image.open(input_image_path).resize((1024, 512), Image.Resampling.BICUBIC)
    img_uint8 = np.array(pil_img)[..., :3] 

    # 2. Preprocess (Pass 0-255 image)
    vp_path = os.path.join(output_dir, f"{image_stem}_vp.txt")
    img_aligned_255, vp = preprocess_manhattan(img_uint8, vp_cache_path=vp_path)
    # Result img_aligned_255 is float64 0-255

    # 3. Normalize to 0-1 float32 (For Model & Visualization)
    img_float = (img_aligned_255 / 255.0).astype(np.float32)

    # 4. Prepare Tensor
    img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)[None]).to(device)

    # 5. Inference
    with torch.no_grad():
        dt = model(img_tensor)

    # 6. Post-Process
    dt['processed_xyz'] = post_process(tensor2np(dt['depth']), type_name='manhattan')
    output_xyz = dt['processed_xyz'][0]

    # 7. Save (Pass 0-1 float image)
    save_results(output_dir, image_stem, img_float, dt, output_xyz)

    print("\nInference Complete.")

if __name__ == '__main__':
    main()