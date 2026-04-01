import os
import numpy as np
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

# --- 1. CONFIGURATION ---
cfg_file = "Open3D-ML/ml3d/configs/randlanet_s3dis.yml"
ply_path = "data/point_cloud/Area_3.ply"
save_path = "data/point_cloud/Area_3_segmented_v2.ply"
ckpt_path = "./logs/randlanet_s3dis_202201071330utc.pth"

# Helper: S3DIS Color Map (R, G, B)
def get_s3dis_colors():
    return np.array([
        [0.0, 1.0, 0.0],  # 0: Ceiling
        [0.0, 0.0, 1.0],  # 1: Floor
        [1.0, 0.0, 0.0],  # 2: Wall
        [1.0, 1.0, 0.0],  # 3: Beam
        [0.0, 1.0, 1.0],  # 4: Column
        [1.0, 0.0, 1.0],  # 5: Window
        [0.5, 0.5, 1.0],  # 6: Door
        [0.8, 0.8, 0.4],  # 7: Table
        [0.6, 0.4, 0.8],  # 8: Chair
        [1.0, 0.6, 0.0],  # 9: Sofa
        [0.3, 0.2, 0.2],  # 10: Bookcase
        [0.2, 0.2, 0.2],  # 11: Board
        [0.2, 1.0, 0.2],  # 12: Clutter
    ])

# --- 2. INITIALIZE PIPELINE ---
print("1. Loading Configuration...")
cfg = _ml3d.utils.Config.load_from_file(cfg_file)
model = ml3d.models.RandLANet(**cfg.model)

# --- BYPASS DATASET CHECKS ---
dummy_path = "/tmp/dummy_s3dis"
if not os.path.exists(dummy_path):
    os.makedirs(dummy_path)
cfg.dataset['dataset_path'] = dummy_path
cfg.dataset['use_cache'] = False
dataset = ml3d.datasets.S3DIS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)

pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

# Load Weights
if not os.path.exists(ckpt_path):
    print("   Downloading weights...")
    os.system(f"wget https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_s3dis_202201071330utc.pth -O {ckpt_path}")

print(f"2. Loading Weights: {ckpt_path}")
pipeline.load_ckpt(ckpt_path=ckpt_path)

# --- 3. LOAD DATA ---
if not os.path.exists(ply_path):
    print(f"ERROR: File not found at {ply_path}")
    exit()

print(f"3. Loading Point Cloud: {ply_path}")
pcd = o3d.io.read_point_cloud(ply_path)

print("   Downsampling to 4cm grid (Required for S3DIS)...")
pcd = pcd.voxel_down_sample(voxel_size=0.04)

# --- CRITICAL FIX: XY-ONLY CENTERING ---
# We must center X and Y, but preserve Z so the AI knows height.
print("   Centering X and Y coordinates (keeping Z absolute)...")
center = pcd.get_center()
# Create a translation vector that only moves X and Y
translation = np.array([-center[0], -center[1], 0.0])
pcd.translate(translation)
print(f"   New Center: {pcd.get_center()}")
# ---------------------------------------

xyz = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

data = {
    'point': xyz,
    'feat': colors, # RGB Only
    'label': np.zeros((len(xyz),), dtype=np.int32)
}

# --- 4. RUN INFERENCE ---
print("4. Running Inference (This takes ~7 minutes)...")
result = pipeline.run_inference(data)
pred_labels = result['predict_labels']

# --- 5. COLORIZE & SAVE ---
print("5. Applying colors to result...")
color_map = get_s3dis_colors()
safe_labels = np.clip(pred_labels, 0, len(color_map) - 1)
pcd.colors = o3d.utility.Vector3dVector(color_map[safe_labels])

print(f"6. Saving segmented cloud to: {save_path}")
o3d.io.write_point_cloud(save_path, pcd)

print("\nDONE! You can now drag the file into CloudCompare to view it.")