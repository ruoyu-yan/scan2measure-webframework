import os
import numpy as np
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

# --- 1. CONFIGURATION ---
cfg_file = "Open3D-ML/ml3d/configs/randlanet_s3dis.yml"
ply_path = "data/point_cloud/Area_3_office1.ply"
save_path = "data/point_cloud/Area_3_office1_segmented_v3.ply"
ckpt_path = "./logs/randlanet_s3dis_202201071330utc.pth"

# Helper: S3DIS Color Map (R, G, B)
def get_s3dis_colors():
    return np.array([
        [0.0, 1.0, 0.0],  # 0: Ceiling (Green)
        [0.0, 0.0, 1.0],  # 1: Floor (Blue)
        [1.0, 0.0, 0.0],  # 2: Wall (Red)
        [1.0, 1.0, 0.0],  # 3: Beam (Yellow)
        [0.0, 1.0, 1.0],  # 4: Column (Cyan)
        [1.0, 0.0, 1.0],  # 5: Window (Magenta)
        [0.5, 0.5, 1.0],  # 6: Door
        [0.8, 0.8, 0.4],  # 7: Table
        [0.6, 0.4, 0.8],  # 8: Chair
        [1.0, 0.6, 0.0],  # 9: Sofa (Orange)
        [0.3, 0.2, 0.2],  # 10: Bookcase
        [0.2, 0.2, 0.2],  # 11: Board (Dark Grey)
        [0.2, 1.0, 0.2],  # 12: Clutter (Light Green)
    ])

# --- 2. INITIALIZE PIPELINE ---
print("1. Loading Configuration...")
cfg = _ml3d.utils.Config.load_from_file(cfg_file)
model = ml3d.models.RandLANet(**cfg.model)

# Bypass dataset checks
dummy_path = "/tmp/dummy_s3dis"
if not os.path.exists(dummy_path):
    os.makedirs(dummy_path)
cfg.dataset['dataset_path'] = dummy_path
cfg.dataset['use_cache'] = False
dataset = ml3d.datasets.S3DIS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)

pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

if not os.path.exists(ckpt_path):
    print("   Downloading weights...")
    os.system(f"wget https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_s3dis_202201071330utc.pth -O {ckpt_path}")
print(f"2. Loading Weights: {ckpt_path}")
pipeline.load_ckpt(ckpt_path=ckpt_path)

# --- 3. LOAD DATA & FIX COORDINATES ---
if not os.path.exists(ply_path):
    print(f"ERROR: File not found at {ply_path}")
    exit()

print(f"3. Loading Point Cloud: {ply_path}")
pcd = o3d.io.read_point_cloud(ply_path)

print("   Downsampling to 4cm grid...")
pcd = pcd.voxel_down_sample(voxel_size=0.04)

# --- CORRECTED CENTERING (XY ONLY) ---
print("   Centering X and Y (Preserving Z height)...")
center = pcd.get_center()
# We subtract the center from X and Y, but subtract 0 from Z
# This keeps the Floor at Z=0 and Ceiling at Z=3
pcd.translate([-center[0], -center[1], 0.0]) 

# Sanity Check for Z-values
min_z = np.asarray(pcd.points)[:, 2].min()
print(f"   Debug: Floor Height (Min Z) is now {min_z:.2f}m (Should be near 0.0)")
# -------------------------------------

xyz = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

data = {
    'point': xyz,
    'feat': colors,
    'label': np.zeros((len(xyz),), dtype=np.int32)
}

# --- 4. INFERENCE ---
print("4. Running Inference...")
result = pipeline.run_inference(data)
pred_labels = result['predict_labels']

# --- 5. SAVE ---
print("5. Saving result...")
color_map = get_s3dis_colors()
safe_labels = np.clip(pred_labels, 0, len(color_map) - 1)
pcd.colors = o3d.utility.Vector3dVector(color_map[safe_labels])

o3d.io.write_point_cloud(save_path, pcd)
print("\nDONE! Drag the file to CloudCompare.")