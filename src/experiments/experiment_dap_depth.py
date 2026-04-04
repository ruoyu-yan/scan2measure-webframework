"""DAP panoramic depth estimation for depth grid search experiment.

Runs the DAP (Depth Any Panoramas) foundation model on each panorama
and saves metric depth maps as .npy files for downstream comparison.

Requires: conda env 'dap_env' (Python 3.12, PyTorch 2.7.1)
Run:  conda run -n dap_env --cwd /tmp/DAP python \
        /home/ruoyu/scan2measure-webframework/src/experiments/experiment_dap_depth.py
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib

# DAP must be imported from its repo root (DINOv3 uses relative hubconf.py)
DAP_ROOT = "/tmp/DAP"
sys.path.insert(0, DAP_ROOT)
os.chdir(DAP_ROOT)
from networks.models import make  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────

PANO_NAMES = [
    "TMB_corridor_north1", "TMB_corridor_north2",
    "TMB_corridor_north3", "TMB_corridor_north4",
    "TMB_corridor_south1", "TMB_corridor_south2",
    "TMB_hall1", "TMB_office1",
]

PROJECT_ROOT = Path("/home/ruoyu/scan2measure-webframework")
PANO_DIR = PROJECT_ROOT / "data" / "pano" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "pano" / "dap_depth"

RENDER_H, RENDER_W = 512, 1024

DAP_MODEL_CONFIG = {
    "model": {
        "name": "dap",
        "args": {
            "midas_model_type": "vitl",
            "fine_tune_type": "hypersim",
            "min_depth": 0.01,
            "max_depth": 1.0,
            "train_decoder": True,
        },
    }
}

DAP_WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Insta360-Research--DAP-weights"
    "/snapshots/558e9ac84efbcb46dc8c47b32c73b333d95f4f0d/model.pth"
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_dap_model():
    """Load DAP model from cached HuggingFace weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(DAP_WEIGHTS, map_location=device)
    model = make(DAP_MODEL_CONFIG["model"])
    if any(k.startswith("module") for k in state.keys()):
        model = nn.DataParallel(model)
    model = model.to(device)
    m_state = model.state_dict()
    model.load_state_dict(
        {k: v for k, v in state.items() if k in m_state}, strict=False
    )
    model.eval()
    return model, device


def estimate_depth(model, device, img_rgb):
    """Run DAP on a single RGB image. Returns float32 (H, W) depth."""
    img = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(tensor)
        pred = outputs["pred_depth"][0].detach().cpu().squeeze().numpy()
    return pred.astype(np.float32)


def colorize_depth(depth, cmap="Spectral"):
    """Convert depth to colorized uint8 RGB for visualization."""
    d = depth.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    colored = matplotlib.colormaps[cmap](d)[..., :3]
    return (colored * 255).astype(np.uint8)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading DAP model...")
    model, device = load_dap_model()
    print(f"DAP model loaded on {device}")

    for pano_name in PANO_NAMES:
        t0 = time.time()
        img_path = PANO_DIR / f"{pano_name}.jpg"
        if not img_path.exists():
            print(f"SKIP {pano_name}: {img_path} not found")
            continue

        # Read and resize
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (RENDER_W, RENDER_H))

        # Estimate depth
        depth = estimate_depth(model, device, img_rgb)

        # Save .npy
        npy_path = OUTPUT_DIR / f"{pano_name}.npy"
        np.save(str(npy_path), depth)

        # Save colorized visualization
        vis = colorize_depth(depth)
        vis_path = OUTPUT_DIR / f"{pano_name}_vis.png"
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        elapsed = time.time() - t0
        print(
            f"{pano_name}: depth range [{depth.min():.4f}, {depth.max():.4f}], "
            f"saved to {npy_path.name} ({elapsed:.1f}s)"
        )

    print("Done.")


if __name__ == "__main__":
    main()
