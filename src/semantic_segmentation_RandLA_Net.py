import open3d as o3d
import open3d.ml.torch as ml3d
import open3d.ml as _ml3d
import numpy as np
import os
import sys
import torch

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Updated paths
MODEL_CONFIG = os.path.join(PROJECT_ROOT, "RandLA-Net/models/randlanet_s3dis/config.yml")
MODEL_WEIGHTS = os.path.join(PROJECT_ROOT, "RandLA-Net/models/randlanet_s3dis/weights.pth")
INPUT_FILE = os.path.join(PROJECT_ROOT, "data/point_cloud/Area_3.ply") 
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/outputs/lab_classified.ply")

# S3DIS Color Map
COLOR_MAP = {
    0: [0, 1, 0], 1: [0, 0, 1], 2: [0.8, 0.8, 0.8], 3: [1, 1, 0],
    4: [1, 0, 1], 5: [0, 1, 1], 6: [1, 0, 0], 7: [0.5, 0.5, 0],
    8: [0.5, 0, 0.5], 9: [0, 0, 1], 10: [0.6, 0.6, 1], 11: [0.3, 0.3, 0.3], 
    12: [0.2, 0.2, 0.2]
}

def main():
    print(f"Loading point cloud: {INPUT_FILE}")
    pcd = o3d.io.read_point_cloud(INPUT_FILE)
    if pcd.is_empty():
        sys.exit("Error: Cloud is empty.")

    # --- CRITICAL FIX: PREPROCESSING ---
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # 1. CENTER THE DATA
    # The model expects points near (0,0,0). We save the center to move it back later.
    center = np.mean(points, axis=0)
    points_centered = points - center
    
    # 2. COLOR CHECK
    # S3DIS weights usually expect colors in [0, 1]
    # If your ply has 0-255, we might need to divide by 255. 
    # Open3D usually handles this auto-magically, but we ensure it here.
    if colors.max() > 1.1:
        print("Normalizing colors from 0-255 to 0-1...")
        colors = colors / 255.0

    print("Initializing RandLA-Net...")
    cfg = _ml3d.utils.Config.load_from_file(MODEL_CONFIG)
    model = ml3d.models.RandLANet(**cfg.model)
    pipeline = ml3d.pipelines.SemanticSegmentation(model, device="gpu", **cfg.pipeline)
    pipeline.load_ckpt(MODEL_WEIGHTS)

    # Prepare data dictionary
    data = {
        'point': points_centered.astype(np.float32), # Use CENTERED points
        'feat': colors.astype(np.float32), 
        'label': np.zeros((len(points),), dtype=np.int32)
    }

    print("Running Inference (on centered data)...")
    results = pipeline.run_inference(data)
    pred_labels = results['predict_labels']

    # --- VISUALIZATION ---
    print("Coloring output...")
    out_colors = np.zeros((len(pred_labels), 3))
    for i, label in enumerate(pred_labels):
        out_colors[i] = COLOR_MAP.get(label, [0, 0, 0])

    out_pcd = o3d.geometry.PointCloud()
    # SAVE WITH ORIGINAL POSITIONS (add center back if you want, 
    # but usually we just use the original 'points' array so it matches your other files)
    out_pcd.points = o3d.utility.Vector3dVector(points) 
    out_pcd.colors = o3d.utility.Vector3dVector(out_colors)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    o3d.io.write_point_cloud(OUTPUT_FILE, out_pcd)
    print(f"SUCCESS! Check {OUTPUT_FILE}")

if __name__ == "__main__":
    main()