# Python Script Modifications Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add --config CLI support and [PROGRESS] progress reporting to all pipeline scripts so the Electron app can orchestrate them dynamically.

**Architecture:** A shared config_loader utility handles argparse + JSON loading with fallback to hardcoded defaults. Each script imports it and replaces top-level constants with config values. Progress reporting uses a simple print-based protocol.

**Tech Stack:** Python, argparse, json

---

## Task 1: Create `src/utils/config_loader.py`

**File:** `/home/ruoyu/scan2measure-webframework/src/utils/config_loader.py`

This module provides two things every pipeline script needs: (1) parse `--config <path>` from argv, (2) load the JSON and return a dict, (3) a helper to emit `[PROGRESS]` lines.

- [ ] 1a. Create `src/utils/config_loader.py` with the following content:

```python
"""Shared config loading and progress reporting for pipeline scripts.

Usage in any pipeline script:
    from config_loader import load_config, progress

    cfg = load_config()  # returns {} if --config not provided
    name = cfg.get("point_cloud_name", "default_name")

    progress(1, 10, "Loading point cloud")
"""

import argparse
import json
import sys


def load_config(extra_args=None):
    """Parse --config <path> from sys.argv and return the JSON dict.

    If --config is not provided, returns an empty dict so callers can
    fall back to hardcoded defaults via cfg.get(key, default).

    Parameters
    ----------
    extra_args : list[str], optional
        Additional argparse arguments to parse (e.g. positional args
        that the script already expects). Not used by default.

    Returns
    -------
    dict
        Parsed config values, or {} if no --config flag was given.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    known, _remaining = parser.parse_known_args()

    if known.config is None:
        return {}

    with open(known.config, "r", encoding="utf-8") as f:
        return json.load(f)


def progress(current, total, message=""):
    """Print a structured progress line for the Electron app to parse.

    Format: [PROGRESS] <current> <total> <message>

    Parameters
    ----------
    current : int
        Current step number (1-based).
    total : int
        Total number of steps.
    message : str
        Human-readable description of the current step.
    """
    print(f"[PROGRESS] {current} {total} {message}", flush=True)
```

- [ ] 1b. Verify the module imports cleanly:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "import sys; sys.path.insert(0, 'src/utils'); from config_loader import load_config, progress; cfg = load_config(); print('OK, cfg =', cfg)"
```

- [ ] 1c. Verify `--config` flag with a test JSON:

```bash
cd /home/ruoyu/scan2measure-webframework && echo '{"point_cloud_name": "test_pc"}' > /tmp/test_cfg.json && python -c "
import sys; sys.argv = ['test', '--config', '/tmp/test_cfg.json']
sys.path.insert(0, 'src/utils')
from config_loader import load_config, progress
cfg = load_config()
assert cfg['point_cloud_name'] == 'test_pc', f'Got {cfg}'
progress(1, 5, 'Loading')
print('PASS')
"
```

Expected stdout includes `[PROGRESS] 1 5 Loading` and `PASS`.

---

## Task 2: Modify `generate_density_image.py` (Stage 1)

**File:** `/home/ruoyu/scan2measure-webframework/src/preprocessing/generate_density_image.py`

**Current config pattern:** `FILENAME` hardcoded inside `main()` (line 161), `input_dir` and `output_dir` at module level (lines 16-20).

**Config keys:**
- `point_cloud_path` -- full path to input PLY (replaces `input_dir / FILENAME`)
- `output_dir` -- output directory (replaces module-level `output_dir / stem`)
- `point_cloud_name` -- stem name for output files (replaces `input_path.stem`)

- [ ] 2a. Add config_loader import and config loading at the top of `main()`:

After the existing imports (line 7), add:

```python
from config_loader import load_config, progress
```

At the start of `main()` (currently line 159), replace the `FILENAME` block:

```python
def main():
    cfg = load_config()

    # --- CONFIGURATION (overridable via --config) ---
    if cfg.get("point_cloud_path"):
        input_path = Path(cfg["point_cloud_path"])
    else:
        FILENAME = "tmb_office_one_corridor_bigger_noRGB.ply"
        input_path = input_dir / FILENAME

    out_folder = Path(cfg["output_dir"]) if cfg.get("output_dir") else output_dir / input_path.stem
    pc_name = cfg.get("point_cloud_name", input_path.stem)
    # ------------------------------------------------
```

- [ ] 2b. Update the output path references (lines 211-218) to use `out_folder` and `pc_name`:

```python
    out_folder.mkdir(parents=True, exist_ok=True)

    image_path = out_folder / (pc_name + ".png")
    metadata_path = out_folder / "metadata.json"
```

- [ ] 2c. Add progress calls. This script has 4 logical steps (load, align, density, save):

```python
    progress(1, 4, "Loading point cloud")
    # ... existing load code ...
    progress(2, 4, "Aligning to floor and axes")
    # ... existing alignment code ...
    progress(3, 4, "Generating density image")
    # ... existing density code ...
    progress(4, 4, "Saving output")
    # ... existing save code ...
```

- [ ] 2d. Verify standalone mode (no --config) still works:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "
import ast, sys
with open('src/preprocessing/generate_density_image.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')
"
```

---

## Task 3: Modify `SAM3_room_segmentation.py` (Stage 2a)

**File:** `/home/ruoyu/scan2measure-webframework/src/experiments/SAM3_room_segmentation.py`

**Current config pattern:** `OUTPUT_DIR` at module level (line 46). Input comes from `sys.argv[1]` positional arg.

**Config keys:**
- `input_path` -- path to density image PNG or directory (replaces `sys.argv[1]`)
- `output_dir` -- output directory (replaces `OUTPUT_DIR`)

- [ ] 3a. Add `sys.path.insert(0, str(project_root / "src" / "utils"))` after line 31, then `from config_loader import load_config, progress`.

- [ ] 3b. Modify `main()` (line 200) to load config and fall back to sys.argv:

```python
def main():
    cfg = load_config()

    # Override output directory if config provides one
    out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else OUTPUT_DIR

    if cfg.get("input_path"):
        target = Path(cfg["input_path"])
    elif len(sys.argv) >= 2:
        target = Path(sys.argv[1])
        if not target.is_absolute():
            target = project_root / target
    else:
        print("Usage: ...")
        sys.exit(1)

    processor = load_model()
    # ... rest unchanged, but replace OUTPUT_DIR with out_dir ...
```

- [ ] 3c. No meaningful progress milestones (model load + single inference), so skip `[PROGRESS]` for this script. The script is fast and atomic.

- [ ] 3d. Syntax-check:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "
import ast
with open('src/experiments/SAM3_room_segmentation.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')
"
```

---

## Task 4: Modify `SAM3_mask_to_polygons.py` (Stage 2b)

**File:** `/home/ruoyu/scan2measure-webframework/src/floorplan/SAM3_mask_to_polygons.py`

**Current config pattern:** `map_name` from `sys.argv[1]` (line 186). Paths derived from map_name.

**Config keys:**
- `map_name` -- map name (replaces `sys.argv[1]`)
- `mask_dir` -- directory with mask PNGs (replaces derived path)
- `density_dir` -- directory with density image + metadata (replaces derived path)
- `output_path` -- output JSON path (replaces derived path)

- [ ] 4a. Add `sys.path.insert(0, str(_PROJECT_ROOT / "src" / "utils"))` after line 33, then `from config_loader import load_config, progress`.

- [ ] 4b. Modify `main()` to load config:

```python
def main():
    cfg = load_config()

    if cfg.get("map_name"):
        map_name = cfg["map_name"]
    elif len(sys.argv) >= 2:
        map_name = sys.argv[1]
    else:
        print("Usage: ...")
        sys.exit(1)

    mask_dir = Path(cfg["mask_dir"]) if cfg.get("mask_dir") else _PROJECT_ROOT / "data" / "sam3_room_segmentation" / map_name
    density_dir = Path(cfg["density_dir"]) if cfg.get("density_dir") else _PROJECT_ROOT / "data" / "density_image" / map_name
    output_path = Path(cfg["output_path"]) if cfg.get("output_path") else mask_dir / f"{map_name}_polygons.json"
    # ... rest uses these local variables instead of re-deriving ...
```

- [ ] 4c. No progress milestones (fast script). Syntax-check:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "
import ast
with open('src/floorplan/SAM3_mask_to_polygons.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')
"
```

---

## Task 5: Modify `SAM3_pano_footprint_extraction.py` (Stage 3)

**File:** `/home/ruoyu/scan2measure-webframework/src/experiments/SAM3_pano_footprint_extraction.py`

**Current config pattern:** `INPUT_DIR` and `OUTPUT_DIR` at module level (lines 54-55). Input also from `sys.argv[1]`.

**Config keys:**
- `input_path` -- path to pano JPG or directory (replaces `sys.argv[1]` / `INPUT_DIR`)
- `output_dir` -- output directory (replaces `OUTPUT_DIR`)
- `pano_names` -- list of pano names to process (optional, if only specific panos needed)

- [ ] 5a. Add `sys.path.insert(0, str(project_root / "src" / "utils"))` after line 43, then `from config_loader import load_config, progress`.

- [ ] 5b. Modify `main()` (line 475) to load config:

```python
def main():
    cfg = load_config()

    out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("input_path"):
        target = Path(cfg["input_path"])
    elif len(sys.argv) >= 2:
        target = Path(sys.argv[1])
        if not target.is_absolute():
            target = project_root / target
    else:
        target = INPUT_DIR

    # ... existing file discovery logic, but use out_dir instead of OUTPUT_DIR ...
```

- [ ] 5c. Add per-pano progress in the processing loop (line 509):

```python
    for i, img_path in enumerate(jpg_files):
        progress(i + 1, len(jpg_files), f"Processing {img_path.stem}")
        process_pano(processor, img_path)
```

Note: `process_pano` uses the module-level `OUTPUT_DIR`. Either pass `out_dir` as a parameter or reassign the module global. Passing as parameter is cleaner -- add an `output_dir=None` parameter to `process_pano()` and use it when provided.

- [ ] 5d. Syntax-check:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "
import ast
with open('src/experiments/SAM3_pano_footprint_extraction.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')
"
```

---

## Task 6: Modify `align_polygons_demo6.py` (Stage 4)

**File:** `/home/ruoyu/scan2measure-webframework/src/floorplan/align_polygons_demo6.py`

**Current config pattern:** `map_name` and `pano_names` from `sys.argv` positional args (lines 412-430). Paths derived from map_name. Legacy `--compare` / `--anchor` flags.

**Config keys:**
- `map_name` -- map name (replaces `argv[0]`)
- `pano_names` -- list of pano names (replaces `argv[1:]`)
- `sam3_map_json` -- path to SAM3 room polygons JSON (replaces derived path)
- `sam3_pano_base` -- base directory for pano footprints (replaces `SAM3_PANO_BASE`)
- `output_dir` -- output directory (replaces derived path)

- [ ] 6a. Add `sys.path.insert(0, str(_SRC_DIR / "utils"))` after line 47, then `from config_loader import load_config, progress`.

- [ ] 6b. Modify `main()` to load config before the existing argv parsing:

```python
def main():
    cfg = load_config()

    if cfg.get("map_name") and cfg.get("pano_names"):
        map_name = cfg["map_name"]
        pano_names = cfg["pano_names"]
        compare_mode = False
        anchor_mode = False
    else:
        # Existing argv parsing (lines 412-430)
        argv = list(sys.argv[1:])
        compare_mode = "--compare" in argv
        if compare_mode:
            argv.remove("--compare")
        anchor_mode = "--anchor" in argv
        if anchor_mode:
            argv.remove("--anchor")
        if len(argv) < 2:
            print("Usage: ...")
            sys.exit(1)
        map_name = argv[0]
        pano_names = argv[1:]

    sam3_map_json = Path(cfg["sam3_map_json"]) if cfg.get("sam3_map_json") else (
        _PROJECT_ROOT / "data" / "sam3_room_segmentation" / map_name / f"{map_name}_polygons.json"
    )
    sam3_pano_base = Path(cfg["sam3_pano_base"]) if cfg.get("sam3_pano_base") else SAM3_PANO_BASE
    output_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else _PROJECT_ROOT / "data" / "sam3_room_segmentation" / map_name
```

- [ ] 6c. No per-step progress (single optimization run). Syntax-check:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "
import ast
with open('src/floorplan/align_polygons_demo6.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')
"
```

---

## Task 7: Modify `point_cloud_geometry_baker_V4.py` (Stage 5a-i)

**File:** `/home/ruoyu/scan2measure-webframework/src/geometry_3d/point_cloud_geometry_baker_V4.py`

**Current config pattern:** `POINT_CLOUD_NAME` (line 22), `KNN` (line 23), derived paths (lines 28-33).

**Config keys:**
- `point_cloud_name` -- PLY stem name (replaces `POINT_CLOUD_NAME`)
- `point_cloud_path` -- full path to PLY (replaces derived `ply_path`)
- `output_dir` -- output directory (replaces derived `out_dir`)
- `knn` -- k neighbours (replaces `KNN`)

- [ ] 7a. Add import after line 5 (`from pathlib import Path`):

```python
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from config_loader import load_config, progress
```

- [ ] 7b. Add config loading at the start of `main()`:

```python
def main():
    cfg = load_config()
    pc_name = cfg.get("point_cloud_name", POINT_CLOUD_NAME)
    knn = cfg.get("knn", KNN)
    _ply_path = Path(cfg["point_cloud_path"]) if cfg.get("point_cloud_path") else project_root / "data" / "raw_point_cloud" / f"{pc_name}.ply"
    _out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else project_root / "data" / "debug_renderer" / pc_name
    _out_dir.mkdir(parents=True, exist_ok=True)
```

Then replace all references to `ply_path`, `out_dir`, `POINT_CLOUD_NAME`, `KNN` inside `main()` with the local variables `_ply_path`, `_out_dir`, `pc_name`, `knn`.

- [ ] 7c. Add progress calls (4 steps: PLY-to-XYZ, run binary, parse OBJ, save PKL):

```python
    progress(1, 4, "Converting PLY to XYZ")
    # ... Step 1 ...
    progress(2, 4, "Running 3DLineDetection binary")
    # ... Step 2 ...
    progress(3, 4, "Parsing line segments from OBJ")
    # ... Step 3 ...
    progress(4, 4, "Saving room_geometry.pkl")
    # ... Step 4 ...
```

- [ ] 7d. Syntax-check:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "
import ast
with open('src/geometry_3d/point_cloud_geometry_baker_V4.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')
"
```

---

## Task 8: Modify `cluster_3d_lines.py` (Stage 5a-ii)

**File:** `/home/ruoyu/scan2measure-webframework/src/geometry_3d/cluster_3d_lines.py`

**Current config pattern:** `POINT_CLOUD_NAME` (line 28), derived paths (lines 35-37).

**Config keys:**
- `point_cloud_name` -- name (replaces `POINT_CLOUD_NAME`)
- `input_pkl` -- path to room_geometry.pkl (replaces derived `PKL_IN`)
- `output_dir` -- output directory (replaces derived `DATA_DIR`)

- [ ] 8a. Add import after line 13 (`import torch`):

```python
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from config_loader import load_config, progress
```

- [ ] 8b. Modify `main()` to load config:

```python
def main():
    cfg = load_config()
    pc_name = cfg.get("point_cloud_name", POINT_CLOUD_NAME)
    pkl_in = Path(cfg["input_pkl"]) if cfg.get("input_pkl") else PROJECT_ROOT / "data" / "debug_renderer" / pc_name / "room_geometry.pkl"
    data_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else PROJECT_ROOT / "data" / "debug_renderer" / pc_name
```

Then replace all references to `PKL_IN`, `DATA_DIR`, `POINT_CLOUD_NAME` inside `main()` with local variables.

- [ ] 8c. Add progress calls (4 steps: load, vote+classify, intersect, save):

```python
    progress(1, 4, "Loading room geometry")
    # ... load pkl ...
    progress(2, 4, "Voting principal directions and classifying lines")
    # ... vote + classify ...
    progress(3, 4, "Finding 3D intersections")
    # ... intersect ...
    progress(4, 4, "Saving 3d_line_map.pkl")
    # ... save ...
```

- [ ] 8d. Syntax-check:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "
import ast
with open('src/geometry_3d/cluster_3d_lines.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')
"
```

---

## Task 9: Modify `image_feature_extractionV2.py` (Stage 5b)

**File:** `/home/ruoyu/scan2measure-webframework/src/features_2d/image_feature_extractionV2.py`

**Current config pattern:** `ROOM_NAME` (line 30), `PANO_RESOLUTION` (line 31), derived `PANO_PATH` and `OUT_DIR` (lines 35-36).

**Config keys:**
- `room_name` -- pano name (replaces `ROOM_NAME`)
- `pano_path` -- full path to panorama JPG (replaces derived `PANO_PATH`)
- `output_dir` -- output directory (replaces derived `OUT_DIR`)
- `pano_resolution` -- `[h, w]` (replaces `PANO_RESOLUTION`)

- [ ] 9a. Add import after line 16 (`from pathlib import Path`):

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from config_loader import load_config, progress
```

- [ ] 9b. Modify `main()` to load config:

```python
def main():
    cfg = load_config()
    room_name = cfg.get("room_name", ROOM_NAME)
    pano_res = tuple(cfg["pano_resolution"]) if cfg.get("pano_resolution") else PANO_RESOLUTION
    pano_path = Path(cfg["pano_path"]) if cfg.get("pano_path") else ROOT / "data" / "pano" / "raw" / f"{room_name}.jpg"
    out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else ROOT / "data" / "pano" / "2d_feature_extracted" / f"{room_name}_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
```

Then replace `PANO_PATH`, `OUT_DIR`, `ROOM_NAME`, `PANO_RESOLUTION` with local variables inside `main()`.

- [ ] 9c. Add progress calls (4 steps: load, detect lines, analyze, save):

```python
    progress(1, 4, f"Loading panorama {room_name}")
    # ... load + resize ...
    progress(2, 4, "Detecting lines on sphere")
    # ... detect_pano_lines + elevation mask ...
    progress(3, 4, "Analyzing vanishing points and intersections")
    # ... VP + classify + intersections ...
    progress(4, 4, "Saving features")
    # ... save JSON + images ...
```

- [ ] 9d. Syntax-check:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "
import ast
with open('src/features_2d/image_feature_extractionV2.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')
"
```

---

## Task 10: Modify `multiroom_pose_estimation.py` (Stage 6)

**File:** `/home/ruoyu/scan2measure-webframework/src/pose_estimation/multiroom_pose_estimation.py`

**Current config pattern:** `POINT_CLOUD_NAME` (line 47), `PANO_NAMES` (line 48), `USE_LOCAL_FILTERING` (line 64), many derived paths (lines 68-76).

**Config keys:**
- `point_cloud_name` -- map name (replaces `POINT_CLOUD_NAME`)
- `pano_names` -- list of pano names (replaces `PANO_NAMES`)
- `use_local_filtering` -- boolean (replaces `USE_LOCAL_FILTERING`)
- `pkl_3d_path` -- path to 3d_line_map.pkl (replaces derived)
- `alignment_path` -- path to demo6_alignment.json (replaces derived)
- `metadata_path` -- path to density image metadata.json (replaces derived)
- `point_cloud_path` -- path to PLY (replaces derived)
- `density_image_path` -- path to density PNG (replaces derived)
- `features_2d_dir` -- base dir for 2D features (replaces derived per-pano path)
- `pano_dir` -- directory with raw pano JPGs (replaces derived)
- `output_dir` -- output base directory (replaces `OUTPUT_BASE`)

**Important:** When called from the Electron app, the config JSON MUST include `"use_local_filtering": true` to ensure Voronoi-based local line filtering is active. The hardcoded default may differ.

- [ ] 10a. The `sys.path` for `utils` is already set up (line 30). Add import:

```python
from config_loader import load_config, progress
```

- [ ] 10b. Modify `main()` to load config at the top, before the existing logic:

```python
def main():
    cfg = load_config()

    pc_name = cfg.get("point_cloud_name", POINT_CLOUD_NAME)
    pano_names = cfg.get("pano_names", PANO_NAMES)
    use_local = cfg.get("use_local_filtering", USE_LOCAL_FILTERING)

    pkl_3d_path = Path(cfg["pkl_3d_path"]) if cfg.get("pkl_3d_path") else ROOT / "data" / "debug_renderer" / pc_name / "3d_line_map.pkl"
    alignment_path = Path(cfg["alignment_path"]) if cfg.get("alignment_path") else ROOT / "data" / "sam3_room_segmentation" / pc_name / "demo6_alignment.json"
    metadata_path = Path(cfg["metadata_path"]) if cfg.get("metadata_path") else ROOT / "data" / "density_image" / pc_name / "metadata.json"
    pc_path = Path(cfg["point_cloud_path"]) if cfg.get("point_cloud_path") else ROOT / "data" / "raw_point_cloud" / f"{pc_name}.ply"
    density_img_path = Path(cfg["density_image_path"]) if cfg.get("density_image_path") else ROOT / "data" / "density_image" / pc_name / f"{pc_name}.png"
    features_2d_base = Path(cfg["features_2d_dir"]) if cfg.get("features_2d_dir") else ROOT / "data" / "pano" / "2d_feature_extracted"
    pano_dir = Path(cfg["pano_dir"]) if cfg.get("pano_dir") else ROOT / "data" / "pano" / "raw"
    output_base = Path(cfg["output_dir"]) if cfg.get("output_dir") else OUTPUT_BASE
```

Then replace all references to the module-level constants (`POINT_CLOUD_NAME`, `PANO_NAMES`, `USE_LOCAL_FILTERING`, `PKL_3D_PATH`, `ALIGNMENT_PATH`, `METADATA_PATH`, `PC_PATH`, `DENSITY_IMG_PATH`, `OUTPUT_BASE`) inside `main()` with the local variables.

Also update the per-pano feature path (line 278-279):
```python
        features_2d_path = features_2d_base / f"{pano_name}_v2" / "fgpl_features.json"
        pano_img_path = pano_dir / f"{pano_name}.jpg"
        pano_output_dir = output_base / pano_name
```

- [ ] 10c. Add per-pano progress. Total steps = 1 (setup) + len(pano_names):

```python
    total_steps = 1 + len(pano_names)
    progress(1, total_steps, "3D setup and precomputation")
    # ... Phase A ...

    for pano_idx, pano_name in enumerate(pano_names):
        progress(2 + pano_idx, total_steps, f"Pose estimation: {pano_name}")
        # ... Phase B per-pano ...
```

- [ ] 10d. Also update the `load_panorama_positions()` function and `compute_voronoi_assignment()` which reference module globals `PANO_NAMES`, `ALIGNMENT_PATH`, `METADATA_PATH`. The cleanest approach: pass these as parameters. Modify signatures:

```python
def load_panorama_positions(alignment_path, metadata_path, pano_names):
    ...

def compute_voronoi_assignment(points_xy, pano_positions, pano_names):
    pano_xys = np.array([pano_positions[name] for name in pano_names])
    ...
```

And update call sites in `main()` accordingly.

- [ ] 10e. Syntax-check:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "
import ast
with open('src/pose_estimation/multiroom_pose_estimation.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')
"
```

---

## Task 11: Modify `colorize_point_cloud.py` (Stage 7)

**File:** `/home/ruoyu/scan2measure-webframework/src/colorization/colorize_point_cloud.py`

**Current config pattern:** `POINT_CLOUD_NAME` (line 27), `PANO_NAMES` (line 28), derived paths (lines 39-44).

**Config keys:**
- `point_cloud_name` -- name (replaces `POINT_CLOUD_NAME`)
- `pano_names` -- list (replaces `PANO_NAMES`)
- `point_cloud_path` -- full path to PLY (replaces derived `PC_PATH`)
- `pano_dir` -- directory with raw pano JPGs (replaces `PANO_DIR`)
- `pose_path` -- path to local_filter_results.json (replaces derived `POSE_PATH`)
- `output_dir` -- output directory (replaces derived `OUTPUT_DIR`)

- [ ] 11a. Add import after line 14 (`from pathlib import Path`):

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from config_loader import load_config, progress
```

- [ ] 11b. Modify `main()` to load config:

```python
def main():
    cfg = load_config()
    t_start = time.time()

    pc_name = cfg.get("point_cloud_name", POINT_CLOUD_NAME)
    pano_names = cfg.get("pano_names", PANO_NAMES)
    pc_path = Path(cfg["point_cloud_path"]) if cfg.get("point_cloud_path") else PC_PATH
    pano_dir = Path(cfg["pano_dir"]) if cfg.get("pano_dir") else PANO_DIR
    pose_path = Path(cfg["pose_path"]) if cfg.get("pose_path") else POSE_PATH
    out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else OUTPUT_DIR
```

Then replace all references to the module-level constants inside `main()`.

- [ ] 11c. Add per-pano progress. Total steps = 1 (load) + len(pano_names) (per-pano) + 1 (blend+save):

```python
    total_steps = 2 + len(pano_names)
    progress(1, total_steps, "Loading point cloud, poses, and panoramas")
    # ... Phase A ...

    for pi, pano_name in enumerate(pano_names):
        progress(2 + pi, total_steps, f"Processing {pano_name}")
        # ... Phase B per-pano ...

    progress(total_steps, total_steps, "Blending and saving colored point cloud")
    # ... Phase C ...
```

- [ ] 11d. Syntax-check:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "
import ast
with open('src/colorization/colorize_point_cloud.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')
"
```

---

## Task 12: Modify `mesh_reconstruction.py` (Stage 8)

**File:** `/home/ruoyu/scan2measure-webframework/src/meshing/mesh_reconstruction.py`

**Current config pattern:** `POINT_CLOUD_NAME` (line 52), `QUALITY_TIER` (line 55), derived `INPUT_PATH` and `OUTPUT_DIR` (lines 98-100). Quality presets dict at module level (lines 57-79).

**Config keys:**
- `point_cloud_name` -- name (replaces `POINT_CLOUD_NAME`)
- `quality_tier` -- "preview" | "balanced" | "high" (replaces `QUALITY_TIER`)
- `input_path` -- full path to colored PLY (replaces derived `INPUT_PATH`)
- `output_dir` -- output directory (replaces derived `OUTPUT_DIR`)

- [ ] 12a. Add import after line 8 (`from pathlib import Path`):

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from config_loader import load_config, progress
```

- [ ] 12b. Modify `main()` to load config:

```python
def main():
    cfg = load_config()
    t_start = time.time()
    stage_times = {}

    pc_name = cfg.get("point_cloud_name", POINT_CLOUD_NAME)
    quality_tier = cfg.get("quality_tier", QUALITY_TIER)
    preset = _QUALITY_PRESETS[quality_tier]
    poisson_depth = preset["poisson_depth"]
    voxel_size = preset["voxel_size"]
    normal_knn = preset["normal_knn"]
    atlas_resolution = preset["atlas_resolution"]
    glb_target_triangles = preset["glb_target_triangles"]

    input_path = Path(cfg["input_path"]) if cfg.get("input_path") else INPUT_PATH
    out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)
```

Then replace all references to module-level `POISSON_DEPTH`, `VOXEL_SIZE`, `NORMAL_KNN`, `ATLAS_RESOLUTION`, `GLB_TARGET_TRIANGLES`, `INPUT_PATH`, `OUTPUT_DIR`, `POINT_CLOUD_NAME`, `QUALITY_TIER` inside `main()` with local variables.

- [ ] 12c. Add progress calls aligned to the existing 17 stages. The script already numbers its stages, so map directly:

```python
    progress(1, 17, "Loading point cloud")
    # ... stage 1 ...
    progress(2, 17, "Voxel downsample")
    # ... stage 2 ...
    progress(3, 17, "Computing tile grid")
    # ... stage 3 ...
    progress(4, 17, "Per-tile Poisson reconstruction (parallel)")
    # ... stages 4-9 ...
    progress(10, 17, "Merging tile meshes")
    # ... stage 10 ...
    progress(11, 17, "Saving vertex-colored PLY")
    # ... stage 11 ...
    progress(12, 17, "Decimating for textured GLB")
    # ... stage 12 ...
    progress(13, 17, "Reloading point cloud for texture baking")
    # ... stage 13 ...
    progress(14, 17, "UV unwrapping with xatlas")
    # ... stage 14 ...
    progress(15, 17, "Baking texture atlas (parallel)")
    # ... stage 15 ...
    progress(16, 17, "Dilating empty texels")
    # ... stage 16 ...
    progress(17, 17, "Exporting GLB with metadata")
    # ... stage 17 ...
```

- [ ] 12d. Syntax-check:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "
import ast
with open('src/meshing/mesh_reconstruction.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')
"
```

---

## Task 13: Integration test with a mock config JSON

Verify the full contract works end-to-end by testing one script (generate_density_image.py) with a config JSON.

- [ ] 13a. Create a test config JSON at `/tmp/test_density_config.json`:

```json
{
    "point_cloud_path": "/home/ruoyu/scan2measure-webframework/data/raw_point_cloud/tmb_office_one_corridor_bigger_noRGB.ply",
    "output_dir": "/tmp/test_density_output",
    "point_cloud_name": "test_density"
}
```

- [ ] 13b. Run the script with `--config` and verify it uses the overridden paths:

```bash
cd /home/ruoyu/scan2measure-webframework && python src/preprocessing/generate_density_image.py --config /tmp/test_density_config.json 2>&1 | head -20
```

Expected: output goes to `/tmp/test_density_output/`, file named `test_density.png`, and `[PROGRESS]` lines appear in stdout.

- [ ] 13c. Run the same script without `--config` and verify it falls back to defaults:

```bash
cd /home/ruoyu/scan2measure-webframework && python src/preprocessing/generate_density_image.py 2>&1 | head -20
```

Expected: uses the hardcoded `FILENAME` and default `output_dir`.

- [ ] 13d. Verify `[PROGRESS]` protocol parsing works:

```bash
cd /home/ruoyu/scan2measure-webframework && python src/preprocessing/generate_density_image.py --config /tmp/test_density_config.json 2>&1 | grep '^\[PROGRESS\]'
```

Expected: 4 lines matching `[PROGRESS] 1 4 ...` through `[PROGRESS] 4 4 ...`.

---

## Task 14: Create `src/utils/downsample_for_preview.py`

**File:** `/home/ruoyu/scan2measure-webframework/src/utils/downsample_for_preview.py`

A lightweight script for the Electron 3D preview. Voxel-downsamples a PLY point cloud to approximately 200K points using Open3D.

**Config keys:**
- `input_path` -- full path to input PLY
- `output_path` -- full path to output downsampled PLY

- [ ] 14a. Create `src/utils/downsample_for_preview.py` with the following content:

```python
"""Downsample a PLY point cloud for Electron 3D preview.

Usage:
    python downsample_for_preview.py --config <path_to_json>

Config keys:
    input_path  -- full path to input PLY
    output_path -- full path to output downsampled PLY
"""

import numpy as np
import open3d as o3d
from pathlib import Path

from config_loader import load_config


def main():
    cfg = load_config()

    input_path = Path(cfg["input_path"])
    output_path = Path(cfg["output_path"])

    print(f"Loading point cloud from {input_path}")
    pcd = o3d.io.read_point_cloud(str(input_path))
    n_original = len(pcd.points)
    print(f"Original point count: {n_original}")

    # Compute voxel size to reach ~200K points
    target_points = 200_000
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.max(bbox.get_extent())
    voxel_size = extent / (target_points ** (1 / 3))
    print(f"Bounding box extent: {extent:.3f} m, voxel size: {voxel_size:.4f} m")

    pcd_down = pcd.voxel_down_sample(voxel_size)
    n_down = len(pcd_down.points)
    print(f"Downsampled point count: {n_down}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd_down)
    print(f"Saved downsampled point cloud to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] 14b. Syntax-check:

```bash
cd /home/ruoyu/scan2measure-webframework && python -c "
import ast
with open('src/utils/downsample_for_preview.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')
"
```

- [ ] 14c. Test with a real PLY:

```bash
cd /home/ruoyu/scan2measure-webframework && echo '{"input_path": "data/raw_point_cloud/tmb_office_one_corridor_bigger_noRGB.ply", "output_path": "/tmp/preview_downsampled.ply"}' > /tmp/test_downsample_cfg.json && python src/utils/downsample_for_preview.py --config /tmp/test_downsample_cfg.json
```

Expected: prints original count, computed voxel size, downsampled count (~200K), and saves to `/tmp/preview_downsampled.ply`.

---

## Summary: Config key reference per script

| Script | Config keys |
|--------|------------|
| `generate_density_image.py` | `point_cloud_path`, `output_dir`, `point_cloud_name` |
| `SAM3_room_segmentation.py` | `input_path`, `output_dir` |
| `SAM3_mask_to_polygons.py` | `map_name`, `mask_dir`, `density_dir`, `output_path` |
| `SAM3_pano_footprint_extraction.py` | `input_path`, `output_dir`, `pano_names` |
| `align_polygons_demo6.py` | `map_name`, `pano_names`, `sam3_map_json`, `sam3_pano_base`, `output_dir` |
| `point_cloud_geometry_baker_V4.py` | `point_cloud_name`, `point_cloud_path`, `output_dir`, `knn` |
| `cluster_3d_lines.py` | `point_cloud_name`, `input_pkl`, `output_dir` |
| `image_feature_extractionV2.py` | `room_name`, `pano_path`, `output_dir`, `pano_resolution` |
| `multiroom_pose_estimation.py` | `point_cloud_name`, `pano_names`, `use_local_filtering`, `pkl_3d_path`, `alignment_path`, `metadata_path`, `point_cloud_path`, `density_image_path`, `features_2d_dir`, `pano_dir`, `output_dir` |
| `colorize_point_cloud.py` | `point_cloud_name`, `pano_names`, `point_cloud_path`, `pano_dir`, `pose_path`, `output_dir` |
| `mesh_reconstruction.py` | `point_cloud_name`, `quality_tier`, `input_path`, `output_dir` |
| `downsample_for_preview.py` | `input_path`, `output_path` |

## Summary: Progress reporting per script

| Script | Has `[PROGRESS]` | Steps | Milestone type |
|--------|:-:|-------|----------------|
| `generate_density_image.py` | Yes | 4 | Per-phase (load, align, density, save) |
| `SAM3_room_segmentation.py` | No | -- | Too fast / atomic |
| `SAM3_mask_to_polygons.py` | No | -- | Too fast / atomic |
| `SAM3_pano_footprint_extraction.py` | Yes | N panos | Per-pano |
| `align_polygons_demo6.py` | No | -- | Single optimization run |
| `point_cloud_geometry_baker_V4.py` | Yes | 4 | Per-step (convert, run, parse, save) |
| `cluster_3d_lines.py` | Yes | 4 | Per-step (load, classify, intersect, save) |
| `image_feature_extractionV2.py` | Yes | 4 | Per-step (load, detect, analyze, save) |
| `multiroom_pose_estimation.py` | Yes | 1 + N panos | Per-pano |
| `colorize_point_cloud.py` | Yes | 2 + N panos | Per-pano |
| `mesh_reconstruction.py` | Yes | 17 | Per-stage (existing 17-stage numbering) |
| `downsample_for_preview.py` | No | -- | Too fast / atomic |
