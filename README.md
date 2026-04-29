# scan2measure-webframework

Multi-stage 3D reconstruction pipeline for indoor building measurement. Takes a TLS point cloud (no RGB) plus panoramic images, estimates camera poses, colorizes the point cloud, reconstructs a UV-textured GLB mesh, and serves it through a measurable Unity virtual tour. Pipeline orchestration is done from an Electron desktop app.

This is the implementation accompanying the master's thesis *scan2measure* (KIT, 2026).

---

## Two ways to use it

| | What you do | What you get |
|---|---|---|
| **Tour Only** | Download the release zip, extract, double-click `scan2measure.bat` | Walk through a pre-built sample tour and use the in-app measurement tools |
| **Full Pipeline** | Clone the repo with submodules, run `setup-native.sh`, then `scan2measure.bat` | Run the full reconstruction pipeline on your own scans (or on the included sample data) |

Both paths use the same Electron app. The Full Pipeline adds the reconstruction stages on top.

---

## Tour Only quickstart

1. Download `scan2measure-v0.1.0.zip` from the [latest release](https://github.com/ruoyu-yan/scan2measure-webframework/releases).
2. Extract anywhere on a Windows 10/11 machine that has WSL2 (Ubuntu 22.04) installed.
3. Double-click `scan2measure.bat`.
4. In the app, click **Tour Only** and load the included sample GLB at `sample-data/tmb_office_one_corridor_bigger_noRGB_textured.glb`.

---

## Full Pipeline setup

### Prerequisites you need to install yourself

The setup script does not install these. Install them first, in any order:

| Required | Why | Hint |
|---|---|---|
| **Windows 10/11 + WSL2 (Ubuntu 22.04)** | The Electron app and Python pipeline run inside WSL; Unity runs on the Windows host | `wsl --install -d Ubuntu-22.04` |
| **Miniconda or Anaconda** (inside WSL) | Two conda environments (`scan_env`, `sam3`) are auto-created by the app on first launch | https://docs.conda.io/projects/miniconda/en/latest/ |
| **Node.js 20+ and npm** (inside WSL) | Builds the Electron app | https://nodejs.org/ |
| **NVIDIA GPU + CUDA driver** | Required by SAM3 inference | NVIDIA driver on Windows host; `nvidia-smi` should work in WSL |
| **HuggingFace account, license accepted for `facebook/sam3`** | Model is gated (manual approval, may take a few hours) | https://huggingface.co/facebook/sam3 → Request access; then `pip install huggingface_hub && huggingface-cli login` |

### Setup steps

```bash
# 1. Clone the repo with all four submodules (sam3, 3DLineDetection, PoissonRecon, mvs-texturing)
git clone --recursive https://github.com/ruoyu-yan/scan2measure-webframework.git
cd scan2measure-webframework

# 2. Apply patches + apt deps + cmake-build the three C++ tools
bash scripts/setup-native.sh

# 3. Build the Electron app (one-off)
cd app
npm install
npm run electron:build:linux
cd ..

# 4. Authenticate with HuggingFace so SAM3 weights can auto-download
huggingface-cli login  # paste a read token from huggingface.co/settings/tokens

# 5. Launch the app
./launch.sh             # from inside WSL
# or, from a Windows Explorer:
# double-click scan2measure.bat
```

The first time you click **Full Pipeline**, the app will prompt you to create the `scan_env` and `sam3` conda environments from `scan_env.yml` / `sam3_env.yml` automatically. The first inference run will then download the SAM3 weights (~3.5 GB) from HuggingFace.

### Sample data

The release zip includes a small sample dataset taken from the TMB office floor (the case study used in the thesis):

| File | Use |
|---|---|
| `sample-data/tmb_office_one_corridor_bigger_noRGB_textured.glb` | Pre-built mesh — load this in **Tour Only** to see the final result |
| `sample-data/tmb_office_one_corridor_dense_noRGB.ply` | Raw uncolored point cloud — use as input to **Full Pipeline** |
| `sample-data/panos/TMB_corridor_south1.jpg`, `TMB_corridor_south2.jpg`, `TMB_office1.jpg` | Three panoramic images covering one corridor and one room |
| `sample-data/camera_pose.json` | Tour spawn pose for the sample GLB |

---

## Dependencies

### External submodules (auto-cloned by `git clone --recursive`)

| Path | Upstream | Pinned at |
|---|---|---|
| `sam3/` | `facebookresearch/sam3` | `f0399e7` |
| `3DLineDetection/` | `xiaohulugo/3DLineDetection` (+ patch) | `f7a0976` |
| `PoissonRecon/` | `mkazhdan/PoissonRecon` v18.75 | `cd6dc7d` |
| `mvs-texturing/` | `nmoehrle/mvs-texturing` (+ patch) | `f337429` |

The two `+ patch` repos have small required modifications (OpenCV 4.x rename, argv-driven `main`, MVE `-std=c++14`) which are stored in `scripts/patches/` and applied by `setup-native.sh`.

### Conda environments (auto-created from inside the app)

| Env | Python | Used by |
|---|---|---|
| `scan_env` | 3.8 | Most of the pipeline (open3d, torch 1.12.1+cu116, opencv, shapely, trimesh, pylsd-nova, ...) |
| `sam3` | 3.12 | SAM3 inference (timm, huggingface_hub, ...) |

Definitions: `scan_env.yml`, `sam3_env.yml` in the repo root.

### System packages installed by `setup-native.sh`

`build-essential cmake git pkg-config libpng-dev libjpeg-dev libtiff-dev libtbb-dev libopencv-dev`

---

## Source layout

See `CLAUDE.md` for a one-page cheat sheet of the source tree, pipeline stages, and key conventions.

---

## Troubleshooting

- **`scan2measure.bat` exits immediately** — open `%TEMP%\scan2measure_err.log` and check the WSL error. The most common cause is that WSL is not installed or `wsl.exe` is not on `PATH`.
- **App opens but Unity tour fails to start** — `unity-launcher.ts` looks for `unity/Build/VirtualTour.exe` next to the app. Confirm the file is there (the release zip ships it; the git checkout does not).
- **Full Pipeline conda env setup hangs / fails** — the env-creation logs are visible in the app. Nine times out of ten this is the PyTorch CUDA wheel mismatch — make sure your NVIDIA driver supports CUDA 11.6.
- **`huggingface_hub.utils._errors.GatedRepoError`** — you have not been approved for `facebook/sam3` yet. Visit the model page and request access; approval is manual.
- **`cmake` fails on `mvs-texturing`** — the patch in `scripts/patches/mvs-texturing.patch` must be applied first. `setup-native.sh` does this automatically. If you ran cmake by hand, run the script instead.

---

## License

See [LICENSE](./LICENSE).
