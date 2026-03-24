# scan2measure Desktop App — Design Spec

**Date:** 2026-03-24
**Status:** Draft

---

## 1. Overview

A desktop application for the scan2measure pipeline that orchestrates point cloud processing, colorization, meshing, and interactive virtual tour with measurement. Two-app architecture: an Electron app (pipeline hub + 3D preview) and a Unity app (virtual tour + measurement).

**Target user:** Power user (researcher/developer). Assumes Python + conda environment already installed.

**Platform:** Windows desktop (primary).

**Key dependency:** The `3DLineDetection` C++ binary (`3DLineDetection/build/src/LineFromPointCloud`) must be pre-built. The Electron app locates it relative to the project root.

**Pipeline context:** SAM3 replaces both RoomFormer (density image room segmentation) and LGT-Net (panorama layout prediction) in this app. RoomFormer/LGT-Net remain in the codebase for research comparison but are not used by the app pipeline.

---

## 2. Architecture

### 2.1 Two-App System

| Component | Technology | Role |
|-----------|-----------|------|
| **Pipeline Hub** | Electron (React + TypeScript) | File management, pipeline orchestration, stage animation, 3D preview |
| **Virtual Tour** | Unity (C#) | First-person navigation, collision, measurement tools |

The Electron app launches the Unity .exe as a subprocess, passing the GLB file path as a command-line argument. They run as separate windows.

### 2.2 Electron App Internals

- **UI layer:** React + CSS for all screens (home, pipeline, confirmation)
- **3D preview panel:** Three.js — reusable component that loads OBJ wireframes, downsampled PLY point clouds, or GLB meshes depending on the pipeline stage. For large PLY files (>1M points), the preview uses a voxel-downsampled version (~200K points) generated as a preprocessing step to keep the renderer responsive.
- **Pipeline engine:** Node.js `child_process.spawn()` runs Python scripts. SAM3 stages (2, 3) use `conda run -n sam3`, all other stages use `conda run -n scan_env`. The active conda env per stage is defined in a stage configuration table (see Section 5).
- **Project store:** JSON file on disk tracking project metadata, input paths, output paths, and status

### 2.3 Unity App Internals

- **GLB loading:** GLTFast package for runtime import
- **Navigation:** CharacterController with WASD + mouse look, mesh collider, gravity, ground snap at ~1.6m
- **Measurement:** Physics.Raycast for surface point selection
- **Minimap:** Secondary orthographic camera rendering density image with player position + direction arrow. If no density image is available (Tour Only or Mesh Only entry), the minimap is hidden.
- **Launch interface:** Accepts GLB path and optional density image path as command-line arguments

---

## 3. Three Entry Points

The home screen presents three paths:

### 3.1 Full Pipeline

**Input:** Uncolored PLY + panoramic images (JPG)

Runs all pipeline stages with animated visualization, pauses at the confirmation gate for pose verification, then completes colorization and meshing.

### 3.2 Mesh Only

**Input:** Already-colored PLY

Previews the point cloud in the 3D viewer (downsampled for preview), user selects a quality tier via the same dropdown as the full pipeline (preview / balanced / high), meshes it, previews the mesh, then launches virtual tour. No density image is generated, so the Unity minimap is hidden.

### 3.3 Tour Only

**Input:** Existing GLB mesh file

Launches Unity directly with the GLB. No Python processing needed. Minimap hidden unless a density image is co-located with the GLB.

---

## 4. Electron UI Flow

### 4.1 Home Screen

Full-screen layout with:
- Three entry point cards (Full Pipeline, Mesh Only, Tour Only) centered on the page
- Recent projects list below, showing project name, status indicator (completed/in-progress), and date
- Clicking a recent project re-opens it at its last state

### 4.2 Pipeline View

Left sidebar + main canvas layout:
- **Left sidebar:** Vertical stage list with progress indicators per stage (checkmark = done, filled circle = active, empty circle = pending, exclamation = confirmation needed). Back arrow to return to home.
- **Main canvas:** Renders the animated visualization for the current stage. Stage header bar shows stage name, description, and elapsed time.

### 4.3 File Upload

After selecting "Full Pipeline" or "Mesh Only," a file selection dialog appears:
- Full Pipeline: select PLY file + one or more JPG panoramas
- Mesh Only: select a single colored PLY file
- Files are validated (PLY format check, image readability)

### 4.4 Error Handling

When a pipeline stage fails (non-zero exit code):
- The stage is marked as failed (red indicator) in the sidebar
- The main canvas shows the last 30 lines of stderr in a scrollable log panel
- Two buttons: **Retry** (re-runs the failed stage) and **Back to Home** (abandons the pipeline, project saved at last successful stage)
- The project remains in the project list at its last completed stage and can be resumed later

---

## 5. Pipeline Animation Stages

Each stage runs a Python script via subprocess. When the script completes, its output artifacts are loaded into the canvas for visualization before the next stage begins.

| # | Stage | Python Script(s) | Conda Env | View | Animation |
|---|-------|-----------------|-----------|------|-----------|
| 1 | Density image generation | `src/preprocessing/generate_density_image.py` | scan_env | 2D | Density image appears on canvas |
| 2a | SAM3 room segmentation | `src/experiments/SAM3_room_segmentation.py` | sam3 | 2D | Room boundary masks overlay on density image |
| 2b | Mask to polygons | `src/experiments/SAM3_mask_to_polygons.py` | scan_env | 2D | Room boundary polygons derived from masks |
| 3 | Pano footprint extraction | `src/experiments/SAM3_pano_footprint_extraction.py` (per pano) | sam3 | 2D | Pano room polygons shown, one per panorama |
| 4 | Polygon matching (jigsaw) | `src/floorplan/align_polygons_demo6.py` | scan_env | 2D | Pano polygons animate into position on density image |
| 5a | 3D line detection | `src/geometry_3d/point_cloud_geometry_baker_V4.py` + `src/geometry_3d/cluster_3d_lines.py` | scan_env | 3D | Wireframe OBJ rendered in Three.js with orbit controls |
| 5b | 2D line detection | `src/features_2d/image_feature_extractionV2.py` (per pano) | scan_env | 2D | Detected lines overlaid on panorama image |
| 6 | Pose estimation | `src/pose_estimation/multiroom_pose_estimation.py` | scan_env | 2D | Camera icons placed on density image + 2D/3D feature overlay on pano |
| **!** | **Confirmation gate** | — | — | **2D** | **User verifies camera positions (see Section 6)** |
| 7 | Colorization | `src/colorization/colorize_point_cloud.py` | scan_env | Progress | Progress bar + coverage percentage |
| 8 | Meshing | `src/meshing/mesh_reconstruction.py` | scan_env | Progress | Quality tier dropdown + progress bar |
| ✓ | Done | — | — | 3D | Preview mesh in Three.js + "Launch Virtual Tour" button |

**Note on Stage 6:** `multiroom_pose_estimation.py` runs with `USE_LOCAL_FILTERING = True`, which integrates Voronoi-based local 3D line filtering. Its output artifact is `local_filter_results.json` (containing R and t per panorama), which is consumed by `colorize_point_cloud.py` in Stage 7.

### 5.1 Meshing Quality Tier

Before the meshing stage (in both Full Pipeline and Mesh Only paths), a dropdown lets the user select:
- **Preview** (~2-3 min) — fast, lower detail
- **Balanced** (~5-8 min) — good quality, recommended default
- **High** (~15-20 min) — maximum detail

The selection maps to the `QUALITY_TIER` parameter in `mesh_reconstruction.py`, which controls voxel size, Poisson depth, atlas resolution, and triangle count.

### 5.2 Script Interface Contract

All Python scripts currently use hardcoded constants (`POINT_CLOUD_NAME`, `PANO_NAMES`, etc.) at the top of each file. Before integration with the Electron app, each script must be modified to accept a **config JSON file** as a command-line argument:

```bash
conda run -n scan_env python src/preprocessing/generate_density_image.py --config /path/to/stage_config.json
```

The config JSON contains all parameters the script needs (input paths, output paths, names, quality tier, etc.). Each script falls back to its current hardcoded defaults if `--config` is not provided, preserving standalone usage.

Example config JSON for `generate_density_image.py`:
```json
{
  "point_cloud_path": "/path/to/input.ply",
  "output_dir": "/path/to/project/density_image/",
  "point_cloud_name": "my_scan"
}
```

The Electron app generates a stage-specific config JSON per stage and passes it to the subprocess.

### 5.3 Progress Reporting Protocol

Python scripts report progress to stdout using a structured prefix that the Electron app parses:

```
[PROGRESS] 45 100 Processing tile 3 of 8
```

Format: `[PROGRESS] <current> <total> <message>`

Lines without the `[PROGRESS]` prefix are treated as log output and displayed in a collapsible log panel. The pipeline engine parses prefixed lines to update the progress bar and stage status.

Existing `print()` statements in scripts continue to work — they appear in the log panel. The `[PROGRESS]` prefix is added incrementally to scripts that have meaningful progress milestones (e.g., per-tile in meshing, per-pano in colorization).

---

## 6. Confirmation Gate

The pipeline pauses after pose estimation. The user must verify that camera positions are correct before colorization proceeds.

### 6.1 Verification View

The main canvas shows the density image with:
- Camera position icons for each panorama
- Each icon labeled with the panorama name
- Clicking an icon shows a thumbnail preview of that panorama

### 6.2 User Actions

- **Confirm & Continue:** Camera positions are accepted. Pipeline proceeds to colorization.
- **Manual Correction:** User drags pano polygon outlines to corrected positions on the density image.
  - Each polygon is labeled with its pano name and shows a thumbnail preview of the associated panorama, so the user knows which room footprint they are repositioning
  - Clicking a polygon shows a larger preview of the associated panorama
  - After repositioning, user clicks "Re-run with corrections"
  - The dragged polygon positions are saved as a corrected alignment JSON with the same schema as `demo6_alignment.json`:
    ```json
    {
      "matches": [
        {
          "pano_name": "TMB_office1",
          "room_label": "room_0",
          "camera_position": [x, y],
          "angle_deg": 45.0,
          "scale": 1.2
        }
      ],
      "scale": 1.2
    }
    ```
  - `multiroom_pose_estimation.py` re-runs using the corrected alignment as input
  - Updated camera positions are shown — the user can confirm or correct again

### 6.3 Correction Loop

The confirm/correct cycle can repeat until the user is satisfied. Each correction re-runs only the pose estimation stage (not the full pipeline).

---

## 7. Unity Virtual Tour

### 7.1 Navigation

- First-person camera controller: WASD movement + mouse look
- Collision detection via mesh collider (prevents walking through walls)
- Gravity with ground snapping (camera height ~1.6m above floor)
- Minimap overlay in corner: density image with player position dot and viewing direction arrow. Hidden when no density image is available.

### 7.2 Measurement Tools (v1)

Top toolbar with icon buttons for each tool:

| Tool | Interaction | Output |
|------|------------|--------|
| **Point-to-point** | Click two surface points | Dashed line + distance label in meters |
| **Wall-to-wall** | Click a wall surface, snaps to nearest plane, measures perpendicular distance to opposite wall | Distance label |
| **Height** | Click floor or ceiling, measures vertical distance to opposite surface | Distance label |

- Active tool highlighted in toolbar
- ESC cancels current measurement
- Clear button removes all measurements
- Status bar at bottom shows active tool name and instructions

### 7.3 Measurement Tools (v2 — future)

- Area measurement: click polygon vertices on a surface, compute area
- Export all measurements as CSV/JSON

### 7.4 Launch Interface

Unity app accepts command-line arguments:
- `--glb <path>` — required, path to the GLB mesh file
- `--minimap <path>` — optional, path to density image PNG for minimap
- `--metadata <path>` — optional, path to mesh metadata JSON

---

## 8. Project Management

### 8.1 Project Store

A JSON file (`projects.json`) in the app's data directory stores project metadata:

```json
{
  "projects": [
    {
      "id": "uuid",
      "name": "TMB Office Corridor",
      "created": "2026-03-24T10:00:00Z",
      "type": "full_pipeline",
      "status": "completed",
      "inputs": {
        "point_cloud": "/path/to/tmb_office.ply",
        "panoramas": ["/path/to/TMB_office1.jpg", "/path/to/TMB_corridor_south1.jpg"]
      },
      "outputs": {
        "density_image": "/path/to/density.png",
        "colored_ply": "/path/to/colored.ply",
        "mesh_glb": "/path/to/mesh.glb",
        "mesh_metadata": "/path/to/metadata.json"
      },
      "last_completed_stage": "done",
      "quality_tier": "balanced"
    }
  ]
}
```

### 8.2 Per-Project Output Directory

Each project gets its own output directory to prevent collisions between projects:

```
data/projects/<project-id>/
├── density_image/          # Stage 1 output
├── sam3_segmentation/      # Stage 2 output
├── sam3_footprints/        # Stage 3 output
├── alignment/              # Stage 4 output (demo6_alignment.json)
├── line_detection/         # Stage 5a output (room_geometry.pkl, 3d_line_map.pkl, OBJ)
├── feature_extraction/     # Stage 5b output (fgpl_features.json per pano)
├── pose_estimation/        # Stage 6 output (local_filter_results.json)
├── textured_point_cloud/   # Stage 7 output (colored PLY)
└── mesh/                   # Stage 8 output (GLB, full-res PLY, metadata)
```

The stage config JSON (Section 5.2) points each script's output to the project-specific directory.

### 8.3 Project Operations

- **Create:** User selects entry point + input files → project created with pending status
- **Re-open:** Clicking a recent project opens it at its last completed stage. If completed, user can launch virtual tour or re-run meshing with different quality.
- **Stale path handling:** On project open, check if input/output files still exist. Show warning if files are missing.

---

## 9. Data Flow

```
User selects files in Electron
    ↓
Electron generates stage config JSON with project-specific paths
    ↓
Electron spawns Python scripts via Node.js child_process
    (conda run -n <env> python <script> --config <config.json>)
    ↓
Python scripts read config, write outputs to project directory
    ↓
Electron reads output artifacts (images, OBJ, PLY, JSON)
    and renders them in the canvas via Three.js or <img>
    ↓
On "Launch Virtual Tour":
    Electron spawns Unity .exe with --glb <path> [--minimap <path>]
    ↓
Unity loads GLB, user navigates and measures
```

### 9.1 Python Script Communication

Each Python script is spawned as a subprocess. The Electron app:
- Generates a stage config JSON and passes it via `--config` argument
- Activates the correct conda environment per stage (see Section 5 table)
- Captures stdout in real-time, parsing `[PROGRESS]` lines for progress bar updates
- Detects completion via process exit code (0 = success, non-zero = error)
- On error: shows the last 30 lines of stderr, offers Retry or Back to Home (see Section 4.4)

### 9.2 Confirmation Gate Data Flow

When the user manually corrects polygon positions:
1. Electron writes the corrected positions to a JSON file (schema in Section 6.2)
2. Electron spawns `multiroom_pose_estimation.py` with a config JSON pointing to the corrected alignment
3. Pose estimation re-runs using the corrected alignment as input
4. Updated camera positions are displayed back in the Electron canvas

---

## 10. Out of Scope (v1)

- Bundled Python/conda installer (user manages their own environment)
- WebGL browser version
- Multi-user collaboration or sharing
- Advanced error recovery beyond retry/abandon (e.g., partial stage re-runs, parameter tuning mid-pipeline)
- Area measurement tool
- Measurement export (CSV/JSON)
- Draco mesh compression
- Cross-platform builds (macOS, Linux)
