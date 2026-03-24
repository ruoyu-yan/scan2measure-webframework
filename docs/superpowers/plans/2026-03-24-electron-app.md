# Electron App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Electron desktop app that orchestrates the scan2measure Python pipeline with animated stage visualization, 3D preview, and project management.

**Architecture:** Electron main process handles subprocess management and file I/O. React renderer provides the UI with React Router for page navigation. Three.js component provides reusable 3D preview. Pipeline engine manages conda subprocess lifecycle with structured progress parsing.

**Tech Stack:** Electron, React, TypeScript, Three.js, React Router, Node.js child_process

---

## File Structure

All new files live under `app/` at the repo root (`/home/ruoyu/scan2measure-webframework/app/`).

```
app/
├── package.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── electron-builder.yml
├── index.html
├── src/
│   ├── main/                          # Electron main process
│   │   ├── index.ts                   # Main entry: BrowserWindow, IPC handlers
│   │   ├── preload.ts                 # Context bridge exposing safe IPC API
│   │   ├── pipeline-engine.ts         # Subprocess spawner + progress parser
│   │   ├── project-store.ts           # projects.json CRUD
│   │   └── unity-launcher.ts          # Spawn Unity .exe with CLI args
│   ├── renderer/                      # React renderer
│   │   ├── App.tsx                    # Root component with React Router
│   │   ├── main.tsx                   # Renderer entry (ReactDOM.createRoot)
│   │   ├── pages/
│   │   │   ├── HomePage.tsx           # Three entry point cards + recent projects
│   │   │   ├── PipelinePage.tsx       # Left sidebar + main canvas layout
│   │   │   └── TourOnlyPage.tsx       # GLB file selection + direct launch
│   │   ├── components/
│   │   │   ├── EntryCard.tsx          # Clickable card for an entry point
│   │   │   ├── RecentProjects.tsx     # Recent projects list
│   │   │   ├── FileUploadDialog.tsx   # PLY + panorama file selection + validation
│   │   │   ├── StageSidebar.tsx       # Vertical stage list with progress indicators
│   │   │   ├── StageCanvas.tsx        # Main canvas router (2D / 3D / progress)
│   │   │   ├── Canvas2D.tsx           # Density image + polygon overlays
│   │   │   ├── ThreeViewer.tsx        # Three.js component for OBJ/PLY/GLB preview
│   │   │   ├── ProgressPanel.tsx      # Progress bar + coverage/elapsed display
│   │   │   ├── ConfirmationGate.tsx   # Camera icons + draggable polygons
│   │   │   ├── ErrorPanel.tsx         # Stderr log + Retry/Back buttons
│   │   │   └── QualityTierSelect.tsx  # Meshing quality dropdown
│   │   ├── hooks/
│   │   │   ├── usePipeline.ts         # Pipeline state management hook
│   │   │   └── useProject.ts          # Project CRUD hook
│   │   ├── types/
│   │   │   ├── pipeline.ts            # Stage, StageStatus, PipelineState types
│   │   │   └── project.ts             # Project, ProjectInput, ProjectOutput types
│   │   └── styles/
│   │       ├── global.css             # Base reset + CSS variables
│   │       ├── home.css               # Home page layout
│   │       ├── pipeline.css           # Pipeline view layout
│   │       └── components.css         # Shared component styles
│   └── shared/
│       └── constants.ts               # Stage definitions, conda envs, script paths
└── assets/
    └── icons/                         # Stage icons (SVG)
```

---

## Part 1: Project Scaffolding

### Task 1.1: Initialize npm project and install dependencies

**Files:**
- Create: `app/package.json`

- [ ] **Step 1: Create `app/` directory**

```bash
mkdir -p /home/ruoyu/scan2measure-webframework/app
```

- [ ] **Step 2: Write `package.json`**

Create `/home/ruoyu/scan2measure-webframework/app/package.json`:

```json
{
  "name": "scan2measure",
  "version": "0.1.0",
  "description": "scan2measure desktop pipeline hub",
  "main": "dist/main/index.js",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "electron:dev": "concurrently \"vite\" \"wait-on http://localhost:5173 && electron .\"",
    "electron:build": "tsc && vite build && electron-builder",
    "typecheck": "tsc --noEmit",
    "lint": "eslint src/"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.0",
    "three": "^0.168.0",
    "@react-three/fiber": "^8.17.0",
    "@react-three/drei": "^9.112.0",
    "uuid": "^10.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.5",
    "@types/react-dom": "^18.3.0",
    "@types/three": "^0.168.0",
    "@types/uuid": "^10.0.0",
    "@vitejs/plugin-react": "^4.3.1",
    "concurrently": "^9.0.1",
    "electron": "^32.0.1",
    "electron-builder": "^25.0.5",
    "typescript": "^5.5.4",
    "vite": "^5.4.2",
    "vite-plugin-electron": "^0.28.7",
    "vite-plugin-electron-renderer": "^0.14.5",
    "wait-on": "^7.2.0"
  }
}
```

- [ ] **Step 3: Install dependencies**

```bash
cd /home/ruoyu/scan2measure-webframework/app && npm install
```

- [ ] **Step 4: Verify `node_modules/` exists and key packages installed**

```bash
ls /home/ruoyu/scan2measure-webframework/app/node_modules/electron/package.json
ls /home/ruoyu/scan2measure-webframework/app/node_modules/react/package.json
ls /home/ruoyu/scan2measure-webframework/app/node_modules/three/package.json
```

Expected: all three paths exist.

---

### Task 1.2: TypeScript configuration

**Files:**
- Create: `app/tsconfig.json`
- Create: `app/tsconfig.node.json`

- [ ] **Step 1: Write `tsconfig.json` (renderer)**

Create `/home/ruoyu/scan2measure-webframework/app/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "baseUrl": ".",
    "paths": {
      "@shared/*": ["src/shared/*"],
      "@renderer/*": ["src/renderer/*"]
    }
  },
  "include": ["src/renderer/**/*", "src/shared/**/*"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

- [ ] **Step 2: Write `tsconfig.node.json` (main process)**

Create `/home/ruoyu/scan2measure-webframework/app/tsconfig.node.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": false,
    "outDir": "dist/main",
    "declaration": true,
    "strict": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true
  },
  "include": ["src/main/**/*", "src/shared/**/*"]
}
```

- [ ] **Step 3: Run typecheck (expect no source files yet, should pass vacuously)**

```bash
cd /home/ruoyu/scan2measure-webframework/app && npx tsc --noEmit 2>&1 | head -5
```

---

### Task 1.3: Vite configuration with Electron plugin

**Files:**
- Create: `app/vite.config.ts`

- [ ] **Step 1: Write `vite.config.ts`**

Create `/home/ruoyu/scan2measure-webframework/app/vite.config.ts`:

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import electron from "vite-plugin-electron";
import renderer from "vite-plugin-electron-renderer";
import path from "node:path";

export default defineConfig({
  plugins: [
    react(),
    electron([
      {
        entry: "src/main/index.ts",
        vite: {
          build: {
            outDir: "dist/main",
            rollupOptions: {
              external: ["electron"],
            },
          },
        },
      },
      {
        entry: "src/main/preload.ts",
        onstart(args) {
          args.reload();
        },
        vite: {
          build: {
            outDir: "dist/preload",
          },
        },
      },
    ]),
    renderer(),
  ],
  resolve: {
    alias: {
      "@shared": path.resolve(__dirname, "src/shared"),
      "@renderer": path.resolve(__dirname, "src/renderer"),
    },
  },
});
```

---

### Task 1.4: HTML entry point and renderer bootstrap

**Files:**
- Create: `app/index.html`
- Create: `app/src/renderer/main.tsx`

- [ ] **Step 1: Write `index.html`**

Create `/home/ruoyu/scan2measure-webframework/app/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta
      http-equiv="Content-Security-Policy"
      content="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: file:;"
    />
    <title>scan2measure</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/renderer/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 2: Write renderer entry `main.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/main.tsx`:

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter } from "react-router-dom";
import App from "./App";
import "./styles/global.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </React.StrictMode>
);
```

---

### Task 1.5: Electron main process entry

**Files:**
- Create: `app/src/main/index.ts`
- Create: `app/src/main/preload.ts`

- [ ] **Step 1: Write main process entry `index.ts`**

Create `/home/ruoyu/scan2measure-webframework/app/src/main/index.ts`:

```typescript
import { app, BrowserWindow, ipcMain, dialog } from "electron";
import path from "node:path";

let mainWindow: BrowserWindow | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    webPreferences: {
      preload: path.join(__dirname, "../preload/preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
    title: "scan2measure",
  });

  // In dev, load Vite dev server; in prod, load built files
  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, "../renderer/index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  app.quit();
});

// ── IPC: File dialogs ────────────────────────────────────────────────────

ipcMain.handle("dialog:openPLY", async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    title: "Select PLY point cloud",
    filters: [{ name: "PLY Files", extensions: ["ply"] }],
    properties: ["openFile"],
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle("dialog:openPanoramas", async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    title: "Select panoramic images",
    filters: [{ name: "Images", extensions: ["jpg", "jpeg", "png"] }],
    properties: ["openFile", "multiSelections"],
  });
  return result.canceled ? null : result.filePaths;
});

ipcMain.handle("dialog:openGLB", async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    title: "Select GLB mesh",
    filters: [{ name: "GLB Files", extensions: ["glb"] }],
    properties: ["openFile"],
  });
  return result.canceled ? null : result.filePaths[0];
});
```

- [ ] **Step 2: Write preload script `preload.ts`**

Create `/home/ruoyu/scan2measure-webframework/app/src/main/preload.ts`:

```typescript
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electronAPI", {
  // File dialogs
  openPLY: () => ipcRenderer.invoke("dialog:openPLY"),
  openPanoramas: () => ipcRenderer.invoke("dialog:openPanoramas"),
  openGLB: () => ipcRenderer.invoke("dialog:openGLB"),

  // Pipeline control
  startPipeline: (projectId: string, stageIndex: number) =>
    ipcRenderer.invoke("pipeline:start", projectId, stageIndex),
  cancelPipeline: () => ipcRenderer.invoke("pipeline:cancel"),
  retryStage: (projectId: string, stageIndex: number) =>
    ipcRenderer.invoke("pipeline:retry", projectId, stageIndex),

  // Pipeline events (main -> renderer)
  onProgress: (callback: (data: { current: number; total: number; message: string }) => void) =>
    ipcRenderer.on("pipeline:progress", (_event, data) => callback(data)),
  onStageComplete: (callback: (stageIndex: number) => void) =>
    ipcRenderer.on("pipeline:stage-complete", (_event, stageIndex) => callback(stageIndex)),
  onStageError: (callback: (data: { stageIndex: number; stderr: string }) => void) =>
    ipcRenderer.on("pipeline:stage-error", (_event, data) => callback(data)),
  onLog: (callback: (line: string) => void) =>
    ipcRenderer.on("pipeline:log", (_event, line) => callback(line)),

  // Project CRUD
  getProjects: () => ipcRenderer.invoke("project:list"),
  createProject: (data: unknown) => ipcRenderer.invoke("project:create", data),
  updateProject: (id: string, data: unknown) =>
    ipcRenderer.invoke("project:update", id, data),
  deleteProject: (id: string) => ipcRenderer.invoke("project:delete", id),

  // Unity launcher
  launchUnity: (glbPath: string, minimapPath?: string, metadataPath?: string) =>
    ipcRenderer.invoke("unity:launch", glbPath, minimapPath, metadataPath),

  // Minimap finder
  findMinimapPng: (glbPath: string) =>
    ipcRenderer.invoke("find-minimap-png", glbPath),

  // Remove listeners
  removeAllListeners: (channel: string) =>
    ipcRenderer.removeAllListeners(channel),
});
```

- [ ] **Step 3: Verify typecheck passes on main process files**

```bash
cd /home/ruoyu/scan2measure-webframework/app && npx tsc --noEmit -p tsconfig.node.json 2>&1 | head -10
```

---

### Task 1.6: Global styles and CSS variables

**Files:**
- Create: `app/src/renderer/styles/global.css`

- [ ] **Step 1: Write `global.css`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/styles/global.css`:

```css
:root {
  /* Colors */
  --bg-primary: #1a1a2e;
  --bg-secondary: #16213e;
  --bg-card: #0f3460;
  --bg-sidebar: #1a1a2e;
  --text-primary: #e6e6e6;
  --text-secondary: #a0a0b0;
  --accent: #00b4d8;
  --accent-hover: #48cae4;
  --success: #4caf50;
  --error: #f44336;
  --warning: #ff9800;
  --pending: #607d8b;

  /* Spacing */
  --sidebar-width: 280px;
  --header-height: 48px;

  /* Typography */
  --font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  --font-mono: "Cascadia Code", "Fira Code", monospace;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body, #root {
  height: 100%;
  width: 100%;
  overflow: hidden;
}

body {
  font-family: var(--font-family);
  background-color: var(--bg-primary);
  color: var(--text-primary);
  -webkit-font-smoothing: antialiased;
}

button {
  cursor: pointer;
  font-family: inherit;
}

a {
  color: var(--accent);
  text-decoration: none;
}
```

---

## Part 2: Shared Types and Constants

### Task 2.1: Pipeline type definitions

**Files:**
- Create: `app/src/renderer/types/pipeline.ts`

- [ ] **Step 1: Write `pipeline.ts`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/types/pipeline.ts`:

```typescript
export type StageStatus =
  | "pending"
  | "active"
  | "complete"
  | "error"
  | "confirmation";

export type QualityTier = "preview" | "balanced" | "high";

export interface StageProgress {
  current: number;
  total: number;
  message: string;
}

export interface StageDefinition {
  index: number;
  id: string;
  name: string;
  description: string;
  scriptPath: string; // relative to project root
  condaEnv: "scan_env" | "sam3";
  viewType: "2d" | "3d" | "progress" | "confirmation";
  /** For stages that run per-panorama, set true */
  perPano: boolean;
}

export interface PipelineState {
  stages: StageStatus[];
  currentStage: number;
  progress: StageProgress | null;
  logLines: string[];
  stderrTail: string;
  elapsedMs: number;
  qualityTier: QualityTier;
}
```

---

### Task 2.2: Project type definitions

**Files:**
- Create: `app/src/renderer/types/project.ts`

- [ ] **Step 1: Write `project.ts`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/types/project.ts`:

```typescript
export type ProjectType = "full_pipeline" | "mesh_only" | "tour_only";
export type ProjectStatus = "pending" | "in_progress" | "completed" | "error";

export interface ProjectInputs {
  pointCloud?: string;
  panoramas?: string[];
  glbFile?: string;
}

export interface ProjectOutputs {
  densityImage?: string;
  coloredPly?: string;
  meshGlb?: string;
  meshMetadata?: string;
}

export interface Project {
  id: string;
  name: string;
  created: string;
  type: ProjectType;
  status: ProjectStatus;
  inputs: ProjectInputs;
  outputs: ProjectOutputs;
  lastCompletedStage: number;
  qualityTier: string;
}

export interface ProjectStore {
  projects: Project[];
}
```

---

### Task 2.3: Stage definitions and constants

**Files:**
- Create: `app/src/shared/constants.ts`

- [ ] **Step 1: Write `constants.ts`**

This maps the spec's pipeline stage table (Section 5) to a typed array used by the pipeline engine and UI sidebar.

Create `/home/ruoyu/scan2measure-webframework/app/src/shared/constants.ts`:

```typescript
export interface StageConfig {
  index: number;
  id: string;
  name: string;
  description: string;
  scriptPaths: string[];
  condaEnv: "scan_env" | "sam3";
  viewType: "2d" | "3d" | "progress" | "confirmation";
  perPano: boolean;
}

/**
 * Full pipeline stage definitions.
 * Script paths are relative to the project root (scan2measure-webframework/).
 * Stages 5a and 5b run sequentially within a single "Line Detection" stage group.
 */
export const FULL_PIPELINE_STAGES: StageConfig[] = [
  {
    index: 0,
    id: "density_image",
    name: "Density Image",
    description: "Generating 2D density projection from point cloud",
    scriptPaths: ["src/preprocessing/generate_density_image.py"],
    condaEnv: "scan_env",
    viewType: "2d",
    perPano: false,
  },
  {
    index: 1,
    id: "sam3_segmentation",
    name: "Room Segmentation",
    description: "SAM3 room boundary detection on density image",
    scriptPaths: ["src/experiments/SAM3_room_segmentation.py"],
    condaEnv: "sam3",
    viewType: "2d",
    perPano: false,
  },
  {
    index: 2,
    id: "sam3_polygons",
    name: "Mask to Polygons",
    description: "Converting SAM3 masks to world-meter polygons",
    scriptPaths: ["src/floorplan/SAM3_mask_to_polygons.py"],
    condaEnv: "scan_env",
    viewType: "2d",
    perPano: false,
  },
  {
    index: 3,
    id: "pano_footprints",
    name: "Pano Footprints",
    description: "Extracting room polygons from panoramic images",
    scriptPaths: ["src/experiments/SAM3_pano_footprint_extraction.py"],
    condaEnv: "sam3",
    viewType: "2d",
    perPano: true,
  },
  {
    index: 4,
    id: "polygon_matching",
    name: "Polygon Matching",
    description: "Fitting pano footprints into density-image room slots",
    scriptPaths: ["src/floorplan/align_polygons_demo6.py"],
    condaEnv: "scan_env",
    viewType: "2d",
    perPano: false,
  },
  {
    index: 5,
    id: "line_detection_3d",
    name: "3D Line Detection",
    description: "Detecting 3D line segments and clustering by principal direction",
    scriptPaths: [
      "src/geometry_3d/point_cloud_geometry_baker_V4.py",
      "src/geometry_3d/cluster_3d_lines.py",
    ],
    condaEnv: "scan_env",
    viewType: "3d",
    perPano: false,
  },
  {
    index: 6,
    id: "line_detection_2d",
    name: "2D Feature Extraction",
    description: "Extracting sphere-based line features from panoramas",
    scriptPaths: ["src/features_2d/image_feature_extractionV2.py"],
    condaEnv: "scan_env",
    viewType: "2d",
    perPano: true,
  },
  {
    index: 7,
    id: "pose_estimation",
    name: "Pose Estimation",
    description: "Multi-room camera localization with Voronoi filtering",
    scriptPaths: ["src/pose_estimation/multiroom_pose_estimation.py"],
    condaEnv: "scan_env",
    viewType: "2d",
    perPano: false,
  },
  {
    index: 8,
    id: "confirmation",
    name: "Confirm Poses",
    description: "Verify camera positions before colorization",
    scriptPaths: [],
    condaEnv: "scan_env",
    viewType: "confirmation",
    perPano: false,
  },
  {
    index: 9,
    id: "colorization",
    name: "Colorization",
    description: "Coloring point cloud from panoramic images",
    scriptPaths: ["src/colorization/colorize_point_cloud.py"],
    condaEnv: "scan_env",
    viewType: "progress",
    perPano: false,
  },
  {
    index: 10,
    id: "meshing",
    name: "Meshing",
    description: "Generating UV-textured GLB mesh",
    scriptPaths: ["src/meshing/mesh_reconstruction.py"],
    condaEnv: "scan_env",
    viewType: "progress",
    perPano: false,
  },
  {
    index: 11,
    id: "done",
    name: "Complete",
    description: "Pipeline complete -- preview mesh or launch virtual tour",
    scriptPaths: [],
    condaEnv: "scan_env",
    viewType: "3d",
    perPano: false,
  },
];

/** Stages for Mesh Only entry point (colored PLY in, GLB out). */
export const MESH_ONLY_STAGES: StageConfig[] = [
  {
    index: 0,
    id: "meshing",
    name: "Meshing",
    description: "Generating UV-textured GLB mesh from colored point cloud",
    scriptPaths: ["src/meshing/mesh_reconstruction.py"],
    condaEnv: "scan_env",
    viewType: "progress",
    perPano: false,
  },
  {
    index: 1,
    id: "done",
    name: "Complete",
    description: "Mesh ready -- preview or launch virtual tour",
    scriptPaths: [],
    condaEnv: "scan_env",
    viewType: "3d",
    perPano: false,
  },
];

/** Per-project output subdirectory names, indexed by stage id. */
export const STAGE_OUTPUT_DIRS: Record<string, string> = {
  density_image: "density_image",
  sam3_segmentation: "sam3_segmentation",
  sam3_polygons: "sam3_polygons",
  pano_footprints: "sam3_footprints",
  polygon_matching: "alignment",
  line_detection_3d: "line_detection",
  line_detection_2d: "feature_extraction",
  pose_estimation: "pose_estimation",
  colorization: "textured_point_cloud",
  meshing: "mesh",
};

/** Default projects data directory (relative to repo root). */
export const PROJECTS_DATA_DIR = "data/projects";

/** Default projects.json location (relative to repo root). */
export const PROJECTS_JSON = "data/projects/projects.json";
```

---

## Part 3: Pipeline Engine (Main Process)

### Task 3.1: Pipeline engine -- subprocess spawner and progress parser

**Files:**
- Create: `app/src/main/pipeline-engine.ts`

- [ ] **Step 1: Write `pipeline-engine.ts`**

Create `/home/ruoyu/scan2measure-webframework/app/src/main/pipeline-engine.ts`:

```typescript
import { ChildProcess, spawn } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import { BrowserWindow } from "electron";

/** Parsed progress line from Python stdout. */
interface ProgressUpdate {
  current: number;
  total: number;
  message: string;
}

/**
 * Parse a stdout line for the [PROGRESS] protocol.
 * Format: [PROGRESS] <current> <total> <message>
 * Returns null if the line does not match.
 */
function parseProgressLine(line: string): ProgressUpdate | null {
  const match = line.match(/^\[PROGRESS\]\s+(\d+)\s+(\d+)\s+(.*)$/);
  if (!match) return null;
  return {
    current: parseInt(match[1], 10),
    total: parseInt(match[2], 10),
    message: match[3].trim(),
  };
}

export interface StageRunConfig {
  scriptPath: string;       // absolute path to Python script
  condaEnv: string;         // conda environment name
  configJsonPath: string;   // absolute path to stage config JSON
  projectRoot: string;      // absolute path to repo root
}

export class PipelineEngine {
  private activeProcess: ChildProcess | null = null;
  private stderrBuffer: string[] = [];
  private window: BrowserWindow;

  constructor(window: BrowserWindow) {
    this.window = window;
  }

  /**
   * Write a stage-specific config JSON to disk.
   * Returns the absolute path to the written file.
   */
  writeStageConfig(outputDir: string, stageId: string, config: Record<string, unknown>): string {
    const configPath = path.join(outputDir, `${stageId}_config.json`);
    fs.mkdirSync(path.dirname(configPath), { recursive: true });
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
    return configPath;
  }

  /**
   * Run a single Python script via conda as a subprocess.
   * Returns a promise that resolves on exit code 0 and rejects on non-zero.
   */
  runStage(config: StageRunConfig): Promise<void> {
    return new Promise((resolve, reject) => {
      this.stderrBuffer = [];

      const args = [
        "run", "--no-banner", "-n", config.condaEnv,
        "python", config.scriptPath,
        "--config", config.configJsonPath,
      ];

      const proc = spawn("conda", args, {
        cwd: config.projectRoot,
        env: { ...process.env },
        stdio: ["ignore", "pipe", "pipe"],
      });

      this.activeProcess = proc;

      // stdout: parse [PROGRESS] lines, forward others as log
      let stdoutRemainder = "";
      proc.stdout!.on("data", (chunk: Buffer) => {
        const text = stdoutRemainder + chunk.toString();
        const lines = text.split("\n");
        stdoutRemainder = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          const progress = parseProgressLine(line);
          if (progress) {
            this.window.webContents.send("pipeline:progress", progress);
          } else {
            this.window.webContents.send("pipeline:log", line);
          }
        }
      });

      // stderr: collect last 30 lines
      let stderrRemainder = "";
      proc.stderr!.on("data", (chunk: Buffer) => {
        const text = stderrRemainder + chunk.toString();
        const lines = text.split("\n");
        stderrRemainder = lines.pop() || "";

        for (const line of lines) {
          this.stderrBuffer.push(line);
          if (this.stderrBuffer.length > 30) {
            this.stderrBuffer.shift();
          }
        }
      });

      proc.on("close", (code) => {
        this.activeProcess = null;
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(this.stderrBuffer.join("\n")));
        }
      });

      proc.on("error", (err) => {
        this.activeProcess = null;
        reject(err);
      });
    });
  }

  /** Cancel the currently running subprocess. */
  cancel(): void {
    if (this.activeProcess) {
      this.activeProcess.kill("SIGTERM");
      this.activeProcess = null;
    }
  }

  /** Get the last 30 stderr lines from the most recent run. */
  getStderrTail(): string {
    return this.stderrBuffer.join("\n");
  }
}
```

- [ ] **Step 2: Verify typecheck**

```bash
cd /home/ruoyu/scan2measure-webframework/app && npx tsc --noEmit -p tsconfig.node.json 2>&1 | head -10
```

---

### Task 3.2: Project store -- JSON CRUD

**Files:**
- Create: `app/src/main/project-store.ts`

- [ ] **Step 1: Write `project-store.ts`**

Create `/home/ruoyu/scan2measure-webframework/app/src/main/project-store.ts`:

```typescript
import fs from "node:fs";
import path from "node:path";
import { v4 as uuidv4 } from "uuid";

export interface ProjectRecord {
  id: string;
  name: string;
  created: string;
  type: "full_pipeline" | "mesh_only" | "tour_only";
  status: "pending" | "in_progress" | "completed" | "error";
  inputs: {
    pointCloud?: string;
    panoramas?: string[];
    glbFile?: string;
  };
  outputs: {
    densityImage?: string;
    coloredPly?: string;
    meshGlb?: string;
    meshMetadata?: string;
  };
  lastCompletedStage: number;
  qualityTier: string;
}

interface StoreData {
  projects: ProjectRecord[];
}

export class ProjectStore {
  private filePath: string;
  private data: StoreData;

  constructor(projectRoot: string) {
    this.filePath = path.join(projectRoot, "data", "projects", "projects.json");
    this.data = this.load();
  }

  private load(): StoreData {
    try {
      const raw = fs.readFileSync(this.filePath, "utf-8");
      return JSON.parse(raw) as StoreData;
    } catch {
      return { projects: [] };
    }
  }

  private save(): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(this.data, null, 2));
  }

  list(): ProjectRecord[] {
    return this.data.projects;
  }

  get(id: string): ProjectRecord | undefined {
    return this.data.projects.find((p) => p.id === id);
  }

  create(partial: Omit<ProjectRecord, "id" | "created" | "status" | "outputs" | "lastCompletedStage">): ProjectRecord {
    const project: ProjectRecord = {
      ...partial,
      id: uuidv4(),
      created: new Date().toISOString(),
      status: "pending",
      outputs: {},
      lastCompletedStage: -1,
    };
    this.data.projects.unshift(project);
    this.save();
    return project;
  }

  update(id: string, patch: Partial<ProjectRecord>): ProjectRecord | null {
    const idx = this.data.projects.findIndex((p) => p.id === id);
    if (idx === -1) return null;
    this.data.projects[idx] = { ...this.data.projects[idx], ...patch };
    this.save();
    return this.data.projects[idx];
  }

  delete(id: string): boolean {
    const before = this.data.projects.length;
    this.data.projects = this.data.projects.filter((p) => p.id !== id);
    if (this.data.projects.length < before) {
      this.save();
      return true;
    }
    return false;
  }

  /** Create the per-project output directory and return its absolute path. */
  ensureProjectDir(projectRoot: string, projectId: string): string {
    const dir = path.join(projectRoot, "data", "projects", projectId);
    fs.mkdirSync(dir, { recursive: true });
    return dir;
  }

  /** Check that all input/output file paths still exist on disk. */
  validatePaths(project: ProjectRecord): string[] {
    const missing: string[] = [];
    const check = (p: string | undefined) => {
      if (p && !fs.existsSync(p)) missing.push(p);
    };
    check(project.inputs.pointCloud);
    project.inputs.panoramas?.forEach(check);
    check(project.inputs.glbFile);
    check(project.outputs.densityImage);
    check(project.outputs.coloredPly);
    check(project.outputs.meshGlb);
    return missing;
  }
}
```

---

### Task 3.3: Unity launcher

**Files:**
- Create: `app/src/main/unity-launcher.ts`

- [ ] **Step 1: Write `unity-launcher.ts`**

Create `/home/ruoyu/scan2measure-webframework/app/src/main/unity-launcher.ts`:

```typescript
import { spawn } from "node:child_process";
import path from "node:path";
import fs from "node:fs";

/**
 * Locate the Unity executable.
 * Searches for VirtualTour.exe in the expected build directory
 * relative to the project root: unity/Build/VirtualTour.exe
 */
function findUnityExe(projectRoot: string): string | null {
  const candidates = [
    path.join(projectRoot, "unity", "Build", "VirtualTour.exe"),
    path.join(projectRoot, "unity", "build", "VirtualTour.exe"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

/**
 * Launch Unity virtual tour with the given GLB file.
 * Returns true if the process was spawned, false if the exe was not found.
 */
export function launchUnity(
  projectRoot: string,
  glbPath: string,
  minimapPath?: string,
  metadataPath?: string
): boolean {
  const exe = findUnityExe(projectRoot);
  if (!exe) return false;

  const args = ["--glb", glbPath];
  if (minimapPath && fs.existsSync(minimapPath)) {
    args.push("--minimap", minimapPath);
  }
  if (metadataPath && fs.existsSync(metadataPath)) {
    args.push("--metadata", metadataPath);
  }

  const proc = spawn(exe, args, {
    cwd: path.dirname(exe),
    detached: true,
    stdio: "ignore",
  });

  // Detach so Unity runs independently of Electron
  proc.unref();
  return true;
}
```

---

### Task 3.4: Wire IPC handlers for pipeline, projects, and Unity in main process

**Files:**
- Modify: `app/src/main/index.ts`

- [ ] **Step 1: Add imports and instantiate engine + store at the top of `index.ts`**

After the existing imports in `/home/ruoyu/scan2measure-webframework/app/src/main/index.ts`, add:

```typescript
import { PipelineEngine } from "./pipeline-engine";
import { ProjectStore } from "./project-store";
import { launchUnity } from "./unity-launcher";

// Project root is two levels up from app/dist/main/
const PROJECT_ROOT = path.resolve(__dirname, "../../..");

let pipelineEngine: PipelineEngine | null = null;
let projectStore: ProjectStore;
```

- [ ] **Step 2: Initialize store and engine after window creation**

After the `mainWindow.on("closed", ...)` block, add:

```typescript
  projectStore = new ProjectStore(PROJECT_ROOT);
  pipelineEngine = new PipelineEngine(mainWindow);
```

- [ ] **Step 3: Add project CRUD IPC handlers**

After the file dialog handlers, add:

```typescript
// ── IPC: Project CRUD ─────────────────────────────────────────────────────

ipcMain.handle("project:list", () => projectStore.list());

ipcMain.handle("project:create", (_event, data) => projectStore.create(data));

ipcMain.handle("project:update", (_event, id: string, patch) =>
  projectStore.update(id, patch)
);

ipcMain.handle("project:delete", (_event, id: string) =>
  projectStore.delete(id)
);
```

- [ ] **Step 4: Add pipeline IPC handlers**

```typescript
// ── IPC: Pipeline control ─────────────────────────────────────────────────

ipcMain.handle(
  "pipeline:start",
  async (_event, projectId: string, stageIndex: number) => {
    if (!pipelineEngine) return { error: "No window" };
    const project = projectStore.get(projectId);
    if (!project) return { error: "Project not found" };

    // Stage config JSON is generated per-stage by the renderer
    // and passed through project:update before pipeline:start.
    // The pipeline engine is invoked by the renderer's usePipeline hook
    // which calls individual stage runs sequentially.
    return { ok: true, projectId, stageIndex };
  }
);

ipcMain.handle("pipeline:cancel", () => {
  pipelineEngine?.cancel();
  return { ok: true };
});

ipcMain.handle(
  "pipeline:retry",
  async (_event, projectId: string, stageIndex: number) => {
    // Same as start -- the renderer hook manages retry logic
    return { ok: true, projectId, stageIndex };
  }
);

// ── IPC: Run a single stage subprocess ───────────────────────────────────

ipcMain.handle(
  "pipeline:run-stage",
  async (
    _event,
    scriptPath: string,
    condaEnv: string,
    configJsonPath: string
  ) => {
    if (!pipelineEngine) return { error: "No window" };
    try {
      await pipelineEngine.runStage({
        scriptPath: path.resolve(PROJECT_ROOT, scriptPath),
        condaEnv,
        configJsonPath,
        projectRoot: PROJECT_ROOT,
      });
      return { ok: true };
    } catch (err) {
      return {
        error: true,
        stderr: pipelineEngine.getStderrTail(),
        message: (err as Error).message,
      };
    }
  }
);
```

- [ ] **Step 5: Add Unity launcher IPC handler**

```typescript
// ── IPC: Unity launcher ──────────────────────────────────────────────────

ipcMain.handle(
  "unity:launch",
  (_event, glbPath: string, minimapPath?: string, metadataPath?: string) => {
    const launched = launchUnity(PROJECT_ROOT, glbPath, minimapPath, metadataPath);
    return { ok: launched, error: launched ? null : "Unity executable not found" };
  }
);
```

- [ ] **Step 6: Add minimap finder and PLY downsample IPC handlers**

```typescript
// ── IPC: Find minimap PNG near a GLB file ────────────────────────────

ipcMain.handle("find-minimap-png", (_event, glbFilePath: string) => {
  const dir = path.dirname(glbFilePath);
  const parent = path.dirname(dir);

  // Search candidate directories for .png files
  const candidateDirs = [
    dir,
    path.join(parent, "density_image"),
    path.join(dir, "density_image"),
  ];

  for (const d of candidateDirs) {
    try {
      const files = fs.readdirSync(d);
      const png = files.find((f) => f.toLowerCase().endsWith(".png"));
      if (png) {
        return { found: true, path: path.join(d, png) };
      }
    } catch {
      // Directory doesn't exist, skip
    }
  }
  return { found: false };
});

// ── IPC: PLY downsample for preview ──────────────────────────────────

ipcMain.handle(
  "downsample-ply",
  async (_event, inputPath: string, outputPath: string) => {
    const configPath = path.join(path.dirname(outputPath), "downsample_config.json");
    fs.mkdirSync(path.dirname(configPath), { recursive: true });
    fs.writeFileSync(
      configPath,
      JSON.stringify({ input_path: inputPath, output_path: outputPath }, null, 2)
    );
    try {
      await new Promise<void>((resolve, reject) => {
        const proc = spawn(
          "conda",
          [
            "run", "--no-banner", "-n", "scan_env",
            "python", path.resolve(PROJECT_ROOT, "src/utils/downsample_for_preview.py"),
            "--config", configPath,
          ],
          { cwd: PROJECT_ROOT, stdio: ["ignore", "pipe", "pipe"] }
        );
        proc.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`exit ${code}`))));
        proc.on("error", reject);
      });
      return { ok: true, outputPath };
    } catch (err) {
      return { error: true, message: (err as Error).message };
    }
  }
);
```

- [ ] **Step 7: Update preload to expose `runStage` and `downsamplePly`**

In `/home/ruoyu/scan2measure-webframework/app/src/main/preload.ts`, add to the `electronAPI` object:

```typescript
  runStage: (scriptPath: string, condaEnv: string, configJsonPath: string) =>
    ipcRenderer.invoke("pipeline:run-stage", scriptPath, condaEnv, configJsonPath),

  downsamplePly: (inputPath: string, outputPath: string) =>
    ipcRenderer.invoke("downsample-ply", inputPath, outputPath),
```

- [ ] **Step 8: Verify typecheck on all main process files**

```bash
cd /home/ruoyu/scan2measure-webframework/app && npx tsc --noEmit -p tsconfig.node.json 2>&1 | head -10
```

---

## Part 4: React App Shell and Routing

### Task 4.1: App component with React Router

**Files:**
- Create: `app/src/renderer/App.tsx`

- [ ] **Step 1: Write `App.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/App.tsx`:

```tsx
import { Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import PipelinePage from "./pages/PipelinePage";
import TourOnlyPage from "./pages/TourOnlyPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/pipeline/:projectId" element={<PipelinePage />} />
      <Route path="/tour-only" element={<TourOnlyPage />} />
    </Routes>
  );
}
```

---

## Part 5: Home Screen

### Task 5.1: EntryCard component

**Files:**
- Create: `app/src/renderer/components/EntryCard.tsx`

- [ ] **Step 1: Write `EntryCard.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/components/EntryCard.tsx`:

```tsx
import React from "react";

interface EntryCardProps {
  title: string;
  description: string;
  icon: string;
  onClick: () => void;
}

export default function EntryCard({ title, description, icon, onClick }: EntryCardProps) {
  return (
    <button className="entry-card" onClick={onClick}>
      <div className="entry-card__icon">{icon}</div>
      <h3 className="entry-card__title">{title}</h3>
      <p className="entry-card__desc">{description}</p>
    </button>
  );
}
```

---

### Task 5.2: RecentProjects component

**Files:**
- Create: `app/src/renderer/components/RecentProjects.tsx`

- [ ] **Step 1: Write `RecentProjects.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/components/RecentProjects.tsx`:

```tsx
import React from "react";
import { useNavigate } from "react-router-dom";
import type { Project } from "../types/project";

interface RecentProjectsProps {
  projects: Project[];
}

const STATUS_LABELS: Record<string, string> = {
  pending: "Pending",
  in_progress: "In Progress",
  completed: "Completed",
  error: "Error",
};

export default function RecentProjects({ projects }: RecentProjectsProps) {
  const navigate = useNavigate();

  if (projects.length === 0) {
    return (
      <div className="recent-projects recent-projects--empty">
        <p>No recent projects. Start by selecting an entry point above.</p>
      </div>
    );
  }

  return (
    <div className="recent-projects">
      <h2 className="recent-projects__title">Recent Projects</h2>
      <ul className="recent-projects__list">
        {projects.slice(0, 10).map((project) => (
          <li
            key={project.id}
            className="recent-projects__item"
            onClick={() => navigate(`/pipeline/${project.id}`)}
          >
            <span className="recent-projects__name">{project.name}</span>
            <span className={`recent-projects__status recent-projects__status--${project.status}`}>
              {STATUS_LABELS[project.status] || project.status}
            </span>
            <span className="recent-projects__date">
              {new Date(project.created).toLocaleDateString()}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

---

### Task 5.3: FileUploadDialog component

**Files:**
- Create: `app/src/renderer/components/FileUploadDialog.tsx`

- [ ] **Step 1: Write `FileUploadDialog.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/components/FileUploadDialog.tsx`:

```tsx
import React, { useState } from "react";

declare global {
  interface Window {
    electronAPI: {
      openPLY: () => Promise<string | null>;
      openPanoramas: () => Promise<string[] | null>;
      openGLB: () => Promise<string | null>;
      [key: string]: unknown;
    };
  }
}

interface FileUploadDialogProps {
  mode: "full_pipeline" | "mesh_only";
  onSubmit: (data: { plyPath: string; panoramas?: string[]; projectName: string }) => void;
  onCancel: () => void;
}

export default function FileUploadDialog({ mode, onSubmit, onCancel }: FileUploadDialogProps) {
  const [plyPath, setPlyPath] = useState<string | null>(null);
  const [panoramas, setPanoramas] = useState<string[]>([]);
  const [projectName, setProjectName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleSelectPLY = async () => {
    const path = await window.electronAPI.openPLY();
    if (path) {
      setPlyPath(path);
      setError(null);
      // Auto-generate project name from filename
      if (!projectName) {
        const baseName = path.split(/[/\\]/).pop()?.replace(".ply", "") || "";
        setProjectName(baseName);
      }
    }
  };

  const handleSelectPanoramas = async () => {
    const paths = await window.electronAPI.openPanoramas();
    if (paths && paths.length > 0) {
      setPanoramas(paths);
      setError(null);
    }
  };

  const handleSubmit = () => {
    if (!plyPath) {
      setError("Please select a PLY file.");
      return;
    }
    if (mode === "full_pipeline" && panoramas.length === 0) {
      setError("Please select at least one panoramic image.");
      return;
    }
    if (!projectName.trim()) {
      setError("Please enter a project name.");
      return;
    }
    onSubmit({
      plyPath,
      panoramas: mode === "full_pipeline" ? panoramas : undefined,
      projectName: projectName.trim(),
    });
  };

  return (
    <div className="file-upload-overlay">
      <div className="file-upload-dialog">
        <h2>{mode === "full_pipeline" ? "Full Pipeline" : "Mesh Only"}</h2>

        <div className="file-upload-field">
          <label>Project Name</label>
          <input
            type="text"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            placeholder="Enter project name"
          />
        </div>

        <div className="file-upload-field">
          <label>Point Cloud (PLY)</label>
          <button className="file-upload-btn" onClick={handleSelectPLY}>
            {plyPath ? plyPath.split(/[/\\]/).pop() : "Select PLY file..."}
          </button>
        </div>

        {mode === "full_pipeline" && (
          <div className="file-upload-field">
            <label>Panoramic Images</label>
            <button className="file-upload-btn" onClick={handleSelectPanoramas}>
              {panoramas.length > 0
                ? `${panoramas.length} image(s) selected`
                : "Select panoramas..."}
            </button>
            {panoramas.length > 0 && (
              <ul className="file-upload-filelist">
                {panoramas.map((p) => (
                  <li key={p}>{p.split(/[/\\]/).pop()}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {error && <p className="file-upload-error">{error}</p>}

        <div className="file-upload-actions">
          <button className="btn btn--secondary" onClick={onCancel}>
            Cancel
          </button>
          <button className="btn btn--primary" onClick={handleSubmit}>
            Start
          </button>
        </div>
      </div>
    </div>
  );
}
```

---

### Task 5.4: HomePage

**Files:**
- Create: `app/src/renderer/pages/HomePage.tsx`
- Create: `app/src/renderer/styles/home.css`

- [ ] **Step 1: Write `HomePage.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/pages/HomePage.tsx`:

```tsx
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import EntryCard from "../components/EntryCard";
import RecentProjects from "../components/RecentProjects";
import FileUploadDialog from "../components/FileUploadDialog";
import type { Project } from "../types/project";
import "../styles/home.css";

type DialogMode = "full_pipeline" | "mesh_only" | null;

export default function HomePage() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[]>([]);
  const [dialogMode, setDialogMode] = useState<DialogMode>(null);

  useEffect(() => {
    window.electronAPI.getProjects().then((list: Project[]) => setProjects(list));
  }, []);

  const handleFullPipeline = () => setDialogMode("full_pipeline");
  const handleMeshOnly = () => setDialogMode("mesh_only");
  const handleTourOnly = () => navigate("/tour-only");

  const handleFileSubmit = async (data: {
    plyPath: string;
    panoramas?: string[];
    projectName: string;
  }) => {
    const project = await window.electronAPI.createProject({
      name: data.projectName,
      type: dialogMode as string,
      inputs: {
        pointCloud: data.plyPath,
        panoramas: data.panoramas,
      },
      qualityTier: "balanced",
    });
    setDialogMode(null);
    navigate(`/pipeline/${(project as Project).id}`);
  };

  return (
    <div className="home">
      <header className="home__header">
        <h1 className="home__title">scan2measure</h1>
        <p className="home__subtitle">
          Point cloud processing, colorization, and virtual tour pipeline
        </p>
      </header>

      <section className="home__cards">
        <EntryCard
          title="Full Pipeline"
          description="Uncolored PLY + panoramic images. Runs all stages with pose verification."
          icon="[F]"
          onClick={handleFullPipeline}
        />
        <EntryCard
          title="Mesh Only"
          description="Already-colored PLY. Meshes and previews, then launches virtual tour."
          icon="[M]"
          onClick={handleMeshOnly}
        />
        <EntryCard
          title="Tour Only"
          description="Existing GLB mesh. Launches Unity virtual tour directly."
          icon="[T]"
          onClick={handleTourOnly}
        />
      </section>

      <RecentProjects projects={projects} />

      {dialogMode && (
        <FileUploadDialog
          mode={dialogMode}
          onSubmit={handleFileSubmit}
          onCancel={() => setDialogMode(null)}
        />
      )}
    </div>
  );
}
```

- [ ] **Step 2: Write `home.css`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/styles/home.css`:

```css
.home {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 48px 32px;
  height: 100%;
  overflow-y: auto;
}

.home__header {
  text-align: center;
  margin-bottom: 48px;
}

.home__title {
  font-size: 2.4rem;
  font-weight: 700;
  color: var(--text-primary);
}

.home__subtitle {
  margin-top: 8px;
  font-size: 1rem;
  color: var(--text-secondary);
}

/* Entry point cards */
.home__cards {
  display: flex;
  gap: 24px;
  margin-bottom: 48px;
}

.entry-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 32px 24px;
  width: 240px;
  background: var(--bg-card);
  border: 2px solid transparent;
  border-radius: 12px;
  color: var(--text-primary);
  transition: border-color 0.2s, transform 0.15s;
}

.entry-card:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
}

.entry-card__icon {
  font-size: 2rem;
  margin-bottom: 12px;
  color: var(--accent);
  font-family: var(--font-mono);
}

.entry-card__title {
  font-size: 1.1rem;
  margin-bottom: 8px;
}

.entry-card__desc {
  font-size: 0.85rem;
  color: var(--text-secondary);
  text-align: center;
  line-height: 1.4;
}

/* Recent projects */
.recent-projects {
  width: 100%;
  max-width: 700px;
}

.recent-projects--empty {
  text-align: center;
  color: var(--text-secondary);
}

.recent-projects__title {
  font-size: 1.2rem;
  margin-bottom: 12px;
  border-bottom: 1px solid var(--bg-card);
  padding-bottom: 8px;
}

.recent-projects__list {
  list-style: none;
}

.recent-projects__item {
  display: flex;
  align-items: center;
  padding: 10px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s;
}

.recent-projects__item:hover {
  background: var(--bg-secondary);
}

.recent-projects__name {
  flex: 1;
  font-weight: 500;
}

.recent-projects__status {
  font-size: 0.8rem;
  padding: 2px 10px;
  border-radius: 10px;
  margin-right: 16px;
}

.recent-projects__status--completed { background: var(--success); color: #fff; }
.recent-projects__status--in_progress { background: var(--accent); color: #fff; }
.recent-projects__status--error { background: var(--error); color: #fff; }
.recent-projects__status--pending { background: var(--pending); color: #fff; }

.recent-projects__date {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

/* File upload dialog */
.file-upload-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.file-upload-dialog {
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 32px;
  width: 460px;
  max-height: 80vh;
  overflow-y: auto;
}

.file-upload-dialog h2 {
  margin-bottom: 20px;
}

.file-upload-field {
  margin-bottom: 16px;
}

.file-upload-field label {
  display: block;
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin-bottom: 6px;
}

.file-upload-field input[type="text"] {
  width: 100%;
  padding: 8px 12px;
  background: var(--bg-primary);
  border: 1px solid var(--bg-card);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 0.95rem;
}

.file-upload-btn {
  width: 100%;
  padding: 10px;
  background: var(--bg-card);
  border: 1px dashed var(--accent);
  border-radius: 6px;
  color: var(--text-secondary);
  font-size: 0.9rem;
  text-align: left;
}

.file-upload-btn:hover {
  background: var(--bg-primary);
}

.file-upload-filelist {
  list-style: none;
  margin-top: 6px;
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.file-upload-filelist li {
  padding: 2px 0;
}

.file-upload-error {
  color: var(--error);
  font-size: 0.85rem;
  margin-bottom: 12px;
}

.file-upload-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 20px;
}

/* Shared button styles */
.btn {
  padding: 8px 20px;
  border-radius: 6px;
  border: none;
  font-size: 0.9rem;
  font-weight: 500;
}

.btn--primary {
  background: var(--accent);
  color: #fff;
}

.btn--primary:hover {
  background: var(--accent-hover);
}

.btn--secondary {
  background: var(--bg-card);
  color: var(--text-primary);
}

.btn--secondary:hover {
  background: var(--bg-primary);
}
```

---

### Task 5.5: TourOnlyPage

**Files:**
- Create: `app/src/renderer/pages/TourOnlyPage.tsx`

- [ ] **Step 1: Write `TourOnlyPage.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/pages/TourOnlyPage.tsx`:

```tsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

/**
 * Search for a minimap PNG near the GLB file.
 * Checks: same directory, parent directory, sibling density_image/ directory.
 */
function findMinimapPng(glbFilePath: string): string | null {
  const dir = glbFilePath.replace(/[/\\][^/\\]+$/, "");
  const parent = dir.replace(/[/\\][^/\\]+$/, "");

  // Candidate locations for a density image PNG
  const candidates = [
    // Same directory: any .png file
    ...findPngFiles(dir),
    // Parent density_image/ directory
    ...findPngFiles(`${parent}/density_image`),
    // Sibling density_image/ directory
    ...findPngFiles(`${dir}/density_image`),
  ];

  return candidates.length > 0 ? candidates[0] : null;
}

/** Return all .png file paths in a directory (non-recursive). */
function findPngFiles(dirPath: string): string[] {
  try {
    // Use the electronAPI to check -- but since we are in renderer,
    // we rely on a simple file:// probe. In practice, the main process
    // should expose a helper. For now, we build candidate paths.
    return [];
  } catch {
    return [];
  }
}

export default function TourOnlyPage() {
  const navigate = useNavigate();
  const [glbPath, setGlbPath] = useState<string | null>(null);
  const [minimapPath, setMinimapPath] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSelectGLB = async () => {
    const path = await window.electronAPI.openGLB();
    if (path) {
      setGlbPath(path);
      setError(null);
      // Check for minimap PNG near the GLB
      const result = await window.electronAPI.findMinimapPng(path);
      const typed = result as { found: boolean; path?: string };
      setMinimapPath(typed.found && typed.path ? typed.path : null);
    }
  };

  const handleLaunch = async () => {
    if (!glbPath) return;
    setLaunching(true);
    const result = await window.electronAPI.launchUnity(
      glbPath,
      minimapPath || undefined
    );
    if (!(result as { ok: boolean }).ok) {
      setError("Unity executable not found. Check that unity/Build/VirtualTour.exe exists.");
    }
    setLaunching(false);
  };

  return (
    <div className="home" style={{ justifyContent: "center" }}>
      <h2>Tour Only</h2>
      <p style={{ color: "var(--text-secondary)", margin: "12px 0 24px" }}>
        Select an existing GLB mesh to launch the Unity virtual tour.
      </p>

      <button className="file-upload-btn" style={{ maxWidth: 400 }} onClick={handleSelectGLB}>
        {glbPath ? glbPath.split(/[/\\]/).pop() : "Select GLB file..."}
      </button>

      {minimapPath && (
        <p style={{ color: "var(--success)", fontSize: "0.8rem", marginTop: 8 }}>
          Minimap found: {minimapPath.split(/[/\\]/).pop()}
        </p>
      )}

      {error && <p className="file-upload-error" style={{ marginTop: 12 }}>{error}</p>}

      <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
        <button className="btn btn--secondary" onClick={() => navigate("/")}>
          Back
        </button>
        <button
          className="btn btn--primary"
          disabled={!glbPath || launching}
          onClick={handleLaunch}
        >
          {launching ? "Launching..." : "Launch Virtual Tour"}
        </button>
      </div>
    </div>
  );
}
```

---

## Part 6: Pipeline View -- Sidebar and Canvas

### Task 6.1: StageSidebar component

**Files:**
- Create: `app/src/renderer/components/StageSidebar.tsx`

- [ ] **Step 1: Write `StageSidebar.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/components/StageSidebar.tsx`:

```tsx
import React from "react";
import type { StageStatus } from "../types/pipeline";

interface StageSidebarProps {
  stages: Array<{ name: string; status: StageStatus }>;
  currentStage: number;
  onBack: () => void;
}

const STATUS_ICONS: Record<StageStatus, string> = {
  complete: "\u2713",       // checkmark
  active: "\u25CF",         // filled circle
  pending: "\u25CB",        // empty circle
  error: "!",               // exclamation
  confirmation: "?",        // question mark
};

export default function StageSidebar({ stages, currentStage, onBack }: StageSidebarProps) {
  return (
    <aside className="stage-sidebar">
      <button className="stage-sidebar__back" onClick={onBack}>
        &larr; Home
      </button>
      <ul className="stage-sidebar__list">
        {stages.map((stage, i) => (
          <li
            key={i}
            className={`stage-sidebar__item stage-sidebar__item--${stage.status} ${
              i === currentStage ? "stage-sidebar__item--current" : ""
            }`}
          >
            <span className="stage-sidebar__icon">{STATUS_ICONS[stage.status]}</span>
            <span className="stage-sidebar__name">{stage.name}</span>
          </li>
        ))}
      </ul>
    </aside>
  );
}
```

---

### Task 6.2: StageCanvas component (router for 2D/3D/progress/confirmation views)

**Files:**
- Create: `app/src/renderer/components/StageCanvas.tsx`

- [ ] **Step 1: Write `StageCanvas.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/components/StageCanvas.tsx`:

```tsx
import React from "react";
import Canvas2D from "./Canvas2D";
import ThreeViewer from "./ThreeViewer";
import ProgressPanel from "./ProgressPanel";
import ConfirmationGate from "./ConfirmationGate";
import ErrorPanel from "./ErrorPanel";
import type { StageProgress, StageStatus } from "../types/pipeline";

interface StageCanvasProps {
  viewType: "2d" | "3d" | "progress" | "confirmation";
  stageStatus: StageStatus;
  stageName: string;
  stageDescription: string;
  elapsedMs: number;
  progress: StageProgress | null;
  logLines: string[];
  stderrTail: string;
  /** Absolute paths to output artifacts for the current stage */
  artifacts: {
    imagePaths?: string[];
    objPath?: string;
    plyPath?: string;
    glbPath?: string;
    densityImagePath?: string;
    polygonsJsonPath?: string;
    alignmentJsonPath?: string;
    cameraPositions?: Array<{ name: string; x: number; y: number }>;
  };
  onConfirm?: () => void;
  onCorrect?: (correctedAlignment: unknown) => void;
  onRetry?: () => void;
  onBack?: () => void;
}

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const remainder = s % 60;
  return `${m}:${String(remainder).padStart(2, "0")}`;
}

export default function StageCanvas(props: StageCanvasProps) {
  const { viewType, stageStatus, stageName, stageDescription, elapsedMs } = props;

  return (
    <main className="stage-canvas">
      <header className="stage-canvas__header">
        <div>
          <h2 className="stage-canvas__name">{stageName}</h2>
          <p className="stage-canvas__desc">{stageDescription}</p>
        </div>
        <span className="stage-canvas__elapsed">{formatElapsed(elapsedMs)}</span>
      </header>

      <div className="stage-canvas__content">
        {stageStatus === "error" ? (
          <ErrorPanel
            stderr={props.stderrTail}
            onRetry={props.onRetry}
            onBack={props.onBack}
          />
        ) : viewType === "2d" ? (
          <Canvas2D
            imagePaths={props.artifacts.imagePaths}
            densityImagePath={props.artifacts.densityImagePath}
            polygonsJsonPath={props.artifacts.polygonsJsonPath}
          />
        ) : viewType === "3d" ? (
          <ThreeViewer
            objPath={props.artifacts.objPath}
            plyPath={props.artifacts.plyPath}
            glbPath={props.artifacts.glbPath}
          />
        ) : viewType === "progress" ? (
          <ProgressPanel
            progress={props.progress}
            logLines={props.logLines}
          />
        ) : viewType === "confirmation" ? (
          <ConfirmationGate
            densityImagePath={props.artifacts.densityImagePath}
            cameraPositions={props.artifacts.cameraPositions}
            alignmentJsonPath={props.artifacts.alignmentJsonPath}
            onConfirm={props.onConfirm}
            onCorrect={props.onCorrect}
          />
        ) : null}
      </div>
    </main>
  );
}
```

---

### Task 6.3: Pipeline page styles

**Files:**
- Create: `app/src/renderer/styles/pipeline.css`

- [ ] **Step 1: Write `pipeline.css`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/styles/pipeline.css`:

```css
.pipeline-page {
  display: flex;
  height: 100%;
}

/* Sidebar */
.stage-sidebar {
  width: var(--sidebar-width);
  background: var(--bg-sidebar);
  border-right: 1px solid var(--bg-card);
  display: flex;
  flex-direction: column;
  padding: 16px 0;
  overflow-y: auto;
  flex-shrink: 0;
}

.stage-sidebar__back {
  padding: 8px 16px;
  margin: 0 12px 16px;
  background: none;
  border: 1px solid var(--bg-card);
  border-radius: 6px;
  color: var(--text-secondary);
  font-size: 0.85rem;
  text-align: left;
}

.stage-sidebar__back:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.stage-sidebar__list {
  list-style: none;
}

.stage-sidebar__item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 20px;
  font-size: 0.9rem;
  color: var(--text-secondary);
  border-left: 3px solid transparent;
}

.stage-sidebar__item--current {
  border-left-color: var(--accent);
  color: var(--text-primary);
  background: var(--bg-secondary);
}

.stage-sidebar__item--complete .stage-sidebar__icon {
  color: var(--success);
}

.stage-sidebar__item--active .stage-sidebar__icon {
  color: var(--accent);
}

.stage-sidebar__item--error .stage-sidebar__icon {
  color: var(--error);
}

.stage-sidebar__item--confirmation .stage-sidebar__icon {
  color: var(--warning);
}

.stage-sidebar__icon {
  width: 18px;
  text-align: center;
  font-weight: 700;
}

/* Canvas */
.stage-canvas {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.stage-canvas__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 24px;
  border-bottom: 1px solid var(--bg-card);
  background: var(--bg-secondary);
}

.stage-canvas__name {
  font-size: 1.1rem;
  font-weight: 600;
}

.stage-canvas__desc {
  font-size: 0.8rem;
  color: var(--text-secondary);
  margin-top: 2px;
}

.stage-canvas__elapsed {
  font-family: var(--font-mono);
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.stage-canvas__content {
  flex: 1;
  overflow: auto;
  padding: 16px;
}
```

---

## Part 7: Stage Visualization Components

### Task 7.1: Canvas2D component (density image + polygon overlays)

**Files:**
- Create: `app/src/renderer/components/Canvas2D.tsx`

- [ ] **Step 1: Write `Canvas2D.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/components/Canvas2D.tsx`:

```tsx
import React, { useRef, useEffect } from "react";

interface Canvas2DProps {
  imagePaths?: string[];
  densityImagePath?: string;
  polygonsJsonPath?: string;
}

export default function Canvas2D({ imagePaths, densityImagePath }: Canvas2DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Load density image as the base layer
    const imgSrc = densityImagePath || (imagePaths && imagePaths[0]);
    if (!imgSrc) {
      ctx.fillStyle = "#1a1a2e";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#a0a0b0";
      ctx.font = "16px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for stage output...", canvas.width / 2, canvas.height / 2);
      return;
    }

    const img = new Image();
    // file:// protocol for local images in Electron
    img.src = `file://${imgSrc}`;
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
    };
    img.onerror = () => {
      ctx.fillStyle = "#1a1a2e";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#f44336";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`Failed to load: ${imgSrc}`, canvas.width / 2, canvas.height / 2);
    };
  }, [imagePaths, densityImagePath]);

  return (
    <div className="canvas2d-container">
      <canvas ref={canvasRef} width={800} height={600} />
    </div>
  );
}
```

---

### Task 7.2: ThreeViewer component (OBJ/PLY/GLB preview)

**Files:**
- Create: `app/src/renderer/components/ThreeViewer.tsx`

- [ ] **Step 1: Write `ThreeViewer.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/components/ThreeViewer.tsx`:

```tsx
import React, { useRef, useEffect, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

interface ThreeViewerProps {
  objPath?: string;
  plyPath?: string;
  glbPath?: string;
}

export default function ThreeViewer({ objPath, plyPath, glbPath }: ThreeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [effectivePlyPath, setEffectivePlyPath] = useState<string | undefined>(plyPath);

  // Downsample PLY for preview before loading
  useEffect(() => {
    if (!plyPath) {
      setEffectivePlyPath(undefined);
      return;
    }
    const downsampledPath = plyPath.replace(/\.ply$/, "_preview.ply");
    window.electronAPI
      .downsamplePly(plyPath, downsampledPath)
      .then((result: { ok?: boolean; outputPath?: string }) => {
        setEffectivePlyPath(result.ok && result.outputPath ? result.outputPath : plyPath);
      })
      .catch(() => {
        // Fall back to original PLY if downsampling fails
        setEffectivePlyPath(plyPath);
      });
  }, [plyPath]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(5, 5, 5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(10, 10, 10);
    scene.add(directional);

    // Grid helper
    const grid = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
    scene.add(grid);

    // Load model based on provided path (use downsampled PLY if available)
    const filePath = glbPath || objPath || effectivePlyPath;
    if (filePath) {
      setLoading(true);
      setError(null);
      const fileUrl = `file://${filePath}`;

      if (glbPath) {
        new GLTFLoader().load(
          fileUrl,
          (gltf) => {
            scene.add(gltf.scene);
            fitCameraToObject(camera, controls, gltf.scene);
            setLoading(false);
          },
          undefined,
          (err) => { setError(`Failed to load GLB: ${err}`); setLoading(false); }
        );
      } else if (objPath) {
        new OBJLoader().load(
          fileUrl,
          (obj) => {
            // Wireframe material for line detection output
            obj.traverse((child) => {
              if (child instanceof THREE.Mesh) {
                child.material = new THREE.MeshBasicMaterial({
                  color: 0x00b4d8,
                  wireframe: true,
                });
              }
            });
            scene.add(obj);
            fitCameraToObject(camera, controls, obj);
            setLoading(false);
          },
          undefined,
          (err) => { setError(`Failed to load OBJ: ${err}`); setLoading(false); }
        );
      } else if (effectivePlyPath) {
        new PLYLoader().load(
          fileUrl,
          (geometry) => {
            if (geometry.hasAttribute("color")) {
              const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true });
              scene.add(new THREE.Points(geometry, material));
            } else {
              const material = new THREE.PointsMaterial({ size: 0.01, color: 0x00b4d8 });
              scene.add(new THREE.Points(geometry, material));
            }
            fitCameraToObject(camera, controls, new THREE.Mesh(geometry));
            setLoading(false);
          },
          undefined,
          (err) => { setError(`Failed to load PLY: ${err}`); setLoading(false); }
        );
      }
    }

    // Animation loop
    let animId: number;
    function animate() {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    // Handle resize
    const handleResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
      controls.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [objPath, effectivePlyPath, glbPath]);

  return (
    <div ref={containerRef} className="three-viewer" style={{ width: "100%", height: "100%" }}>
      {loading && (
        <div className="three-viewer__overlay">Loading 3D model...</div>
      )}
      {error && (
        <div className="three-viewer__overlay three-viewer__overlay--error">{error}</div>
      )}
    </div>
  );
}

/** Fit camera to center on a loaded object. */
function fitCameraToObject(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  object: THREE.Object3D
) {
  const box = new THREE.Box3().setFromObject(object);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();

  controls.target.copy(center);
  camera.position.copy(center);
  camera.position.x += size * 0.8;
  camera.position.y += size * 0.5;
  camera.position.z += size * 0.8;
  camera.near = size / 100;
  camera.far = size * 100;
  camera.updateProjectionMatrix();
  controls.update();
}
```

---

### Task 7.3: ProgressPanel component

**Files:**
- Create: `app/src/renderer/components/ProgressPanel.tsx`

- [ ] **Step 1: Write `ProgressPanel.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/components/ProgressPanel.tsx`:

```tsx
import React, { useRef, useEffect } from "react";
import type { StageProgress } from "../types/pipeline";

interface ProgressPanelProps {
  progress: StageProgress | null;
  logLines: string[];
}

export default function ProgressPanel({ progress, logLines }: ProgressPanelProps) {
  const logRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log to bottom
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  const pct = progress && progress.total > 0
    ? Math.round((progress.current / progress.total) * 100)
    : 0;

  return (
    <div className="progress-panel">
      {progress && (
        <div className="progress-panel__bar-container">
          <div className="progress-panel__bar">
            <div className="progress-panel__fill" style={{ width: `${pct}%` }} />
          </div>
          <span className="progress-panel__pct">{pct}%</span>
        </div>
      )}

      {progress && (
        <p className="progress-panel__message">
          {progress.message} ({progress.current}/{progress.total})
        </p>
      )}

      <div className="progress-panel__log" ref={logRef}>
        {logLines.map((line, i) => (
          <div key={i} className="progress-panel__log-line">{line}</div>
        ))}
      </div>
    </div>
  );
}
```

---

### Task 7.4: ErrorPanel component

**Files:**
- Create: `app/src/renderer/components/ErrorPanel.tsx`

- [ ] **Step 1: Write `ErrorPanel.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/components/ErrorPanel.tsx`:

```tsx
import React from "react";

interface ErrorPanelProps {
  stderr: string;
  onRetry?: () => void;
  onBack?: () => void;
}

export default function ErrorPanel({ stderr, onRetry, onBack }: ErrorPanelProps) {
  return (
    <div className="error-panel">
      <h3 className="error-panel__title">Stage Failed</h3>
      <p className="error-panel__subtitle">Last 30 lines of stderr:</p>
      <pre className="error-panel__log">{stderr || "(no stderr output)"}</pre>
      <div className="error-panel__actions">
        {onRetry && (
          <button className="btn btn--primary" onClick={onRetry}>
            Retry
          </button>
        )}
        {onBack && (
          <button className="btn btn--secondary" onClick={onBack}>
            Back to Home
          </button>
        )}
      </div>
    </div>
  );
}
```

---

### Task 7.5: QualityTierSelect component

**Files:**
- Create: `app/src/renderer/components/QualityTierSelect.tsx`

- [ ] **Step 1: Write `QualityTierSelect.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/components/QualityTierSelect.tsx`:

```tsx
import React from "react";
import type { QualityTier } from "../types/pipeline";

interface QualityTierSelectProps {
  value: QualityTier;
  onChange: (tier: QualityTier) => void;
  disabled?: boolean;
}

const TIER_INFO: Record<QualityTier, { label: string; time: string }> = {
  preview: { label: "Preview", time: "~2-3 min" },
  balanced: { label: "Balanced", time: "~5-8 min" },
  high: { label: "High", time: "~15-20 min" },
};

export default function QualityTierSelect({ value, onChange, disabled }: QualityTierSelectProps) {
  return (
    <div className="quality-tier-select">
      <label className="quality-tier-select__label">Mesh Quality</label>
      <select
        className="quality-tier-select__select"
        value={value}
        onChange={(e) => onChange(e.target.value as QualityTier)}
        disabled={disabled}
      >
        {(Object.keys(TIER_INFO) as QualityTier[]).map((tier) => (
          <option key={tier} value={tier}>
            {TIER_INFO[tier].label} ({TIER_INFO[tier].time})
          </option>
        ))}
      </select>
    </div>
  );
}
```

---

### Task 7.6: Component styles

**Files:**
- Create: `app/src/renderer/styles/components.css`

- [ ] **Step 1: Write `components.css`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/styles/components.css`:

```css
/* Canvas 2D */
.canvas2d-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
}

.canvas2d-container canvas {
  max-width: 100%;
  max-height: 100%;
  border-radius: 4px;
}

/* Three.js viewer */
.three-viewer {
  position: relative;
}

.three-viewer__overlay {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: var(--text-secondary);
  font-size: 1rem;
  background: rgba(0, 0, 0, 0.6);
  padding: 12px 24px;
  border-radius: 8px;
}

.three-viewer__overlay--error {
  color: var(--error);
}

/* Progress panel */
.progress-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: 100%;
}

.progress-panel__bar-container {
  display: flex;
  align-items: center;
  gap: 12px;
}

.progress-panel__bar {
  flex: 1;
  height: 12px;
  background: var(--bg-card);
  border-radius: 6px;
  overflow: hidden;
}

.progress-panel__fill {
  height: 100%;
  background: var(--accent);
  border-radius: 6px;
  transition: width 0.3s ease;
}

.progress-panel__pct {
  font-family: var(--font-mono);
  font-size: 0.9rem;
  min-width: 40px;
}

.progress-panel__message {
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.progress-panel__log {
  flex: 1;
  background: var(--bg-primary);
  border: 1px solid var(--bg-card);
  border-radius: 6px;
  padding: 12px;
  overflow-y: auto;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  line-height: 1.5;
  color: var(--text-secondary);
}

.progress-panel__log-line {
  white-space: pre-wrap;
  word-break: break-all;
}

/* Error panel */
.error-panel {
  max-width: 700px;
  margin: 32px auto;
}

.error-panel__title {
  color: var(--error);
  font-size: 1.2rem;
  margin-bottom: 8px;
}

.error-panel__subtitle {
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin-bottom: 12px;
}

.error-panel__log {
  background: var(--bg-primary);
  border: 1px solid var(--error);
  border-radius: 6px;
  padding: 16px;
  max-height: 300px;
  overflow-y: auto;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  line-height: 1.5;
  color: var(--text-secondary);
  white-space: pre-wrap;
}

.error-panel__actions {
  display: flex;
  gap: 12px;
  margin-top: 20px;
}

/* Quality tier select */
.quality-tier-select {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.quality-tier-select__label {
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.quality-tier-select__select {
  padding: 6px 12px;
  background: var(--bg-card);
  border: 1px solid var(--accent);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 0.9rem;
}

/* Confirmation gate */
.confirmation-gate {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.confirmation-gate__canvas {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
}

.confirmation-gate__canvas canvas {
  max-width: 100%;
  max-height: 100%;
}

.confirmation-gate__actions {
  display: flex;
  justify-content: center;
  gap: 16px;
  padding: 16px;
  border-top: 1px solid var(--bg-card);
}
```

---

## Part 8: Confirmation Gate

### Task 8.1: ConfirmationGate component

**Files:**
- Create: `app/src/renderer/components/ConfirmationGate.tsx`

- [ ] **Step 1: Write `ConfirmationGate.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/components/ConfirmationGate.tsx`:

```tsx
import React, { useRef, useEffect, useState, useCallback } from "react";
import fs from "node:fs";

/** A single match entry from demo6_alignment.json */
interface AlignmentMatch {
  pano_name: string;
  room_label: string;
  camera_position: [number, number];
  angle_deg: number;
  scale: number;
  transformed: number[][];  // polygon vertices [[x,y], ...]
}

/** Top-level alignment JSON structure */
interface AlignmentData {
  matches: AlignmentMatch[];
  scale: number;
}

/** Internal state per pano: original data + drag offset */
interface PanoEntry {
  match: AlignmentMatch;
  offsetX: number;
  offsetY: number;
}

interface ConfirmationGateProps {
  densityImagePath?: string;
  cameraPositions?: Array<{ name: string; x: number; y: number }>;
  alignmentJsonPath?: string;
  panoramaDir?: string;
  onConfirm?: () => void;
  onCorrect?: (correctedAlignment: unknown) => void;
}

export default function ConfirmationGate({
  densityImagePath,
  alignmentJsonPath,
  panoramaDir,
  onConfirm,
  onCorrect,
}: ConfirmationGateProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [entries, setEntries] = useState<PanoEntry[]>([]);
  const [alignmentData, setAlignmentData] = useState<AlignmentData | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [selectedPano, setSelectedPano] = useState<number | null>(null);
  const [thumbnailSrc, setThumbnailSrc] = useState<string | null>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number }>({ w: 800, h: 600 });

  // Load alignment JSON
  useEffect(() => {
    if (!alignmentJsonPath) return;
    fetch(`file://${alignmentJsonPath}`)
      .then((r) => r.json())
      .then((data: AlignmentData) => {
        setAlignmentData(data);
        setEntries(
          data.matches.map((m) => ({ match: m, offsetX: 0, offsetY: 0 }))
        );
      })
      .catch(() => {});
  }, [alignmentJsonPath]);

  // Load density image to get natural size
  useEffect(() => {
    if (!densityImagePath) return;
    const img = new Image();
    img.src = `file://${densityImagePath}`;
    img.onload = () => setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
  }, [densityImagePath]);

  // Load pano thumbnail on click
  useEffect(() => {
    if (selectedPano === null || !panoramaDir) {
      setThumbnailSrc(null);
      return;
    }
    const panoName = entries[selectedPano]?.match.pano_name;
    if (!panoName) return;
    // Try common extensions
    for (const ext of [".jpg", ".jpeg", ".png"]) {
      const tryPath = `${panoramaDir}/${panoName}${ext}`;
      try {
        // In Electron renderer, use file:// protocol
        setThumbnailSrc(`file://${tryPath}`);
        return;
      } catch {
        continue;
      }
    }
  }, [selectedPano, entries, panoramaDir]);

  // Colors for polygon outlines (cycle through palette)
  const COLORS = ["#00b4d8", "#ff6b6b", "#51cf66", "#ffd43b", "#cc5de8", "#ff922b"];

  // SVG coordinate helpers
  const getSVGCoords = (e: React.MouseEvent<SVGSVGElement>): { x: number; y: number } => {
    const svg = svgRef.current!;
    const rect = svg.getBoundingClientRect();
    const scaleX = imgSize.w / rect.width;
    const scaleY = imgSize.h / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  // Build polygon points string with offset applied
  const polygonPoints = (entry: PanoEntry): string =>
    entry.match.transformed
      .map(([x, y]) => `${x + entry.offsetX},${y + entry.offsetY}`)
      .join(" ");

  // Compute centroid for label placement
  const centroid = (entry: PanoEntry): { cx: number; cy: number } => {
    const verts = entry.match.transformed;
    const n = verts.length || 1;
    const cx = verts.reduce((s, [x]) => s + x, 0) / n + entry.offsetX;
    const cy = verts.reduce((s, [, y]) => s + y, 0) / n + entry.offsetY;
    return { cx, cy };
  };

  // Check if click is inside a polygon (ray casting)
  const pointInPolygon = (px: number, py: number, entry: PanoEntry): boolean => {
    const verts = entry.match.transformed.map(([x, y]) => [
      x + entry.offsetX,
      y + entry.offsetY,
    ]);
    let inside = false;
    for (let i = 0, j = verts.length - 1; i < verts.length; j = i++) {
      const xi = verts[i][0], yi = verts[i][1];
      const xj = verts[j][0], yj = verts[j][1];
      if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
        inside = !inside;
      }
    }
    return inside;
  };

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    const { x, y } = getSVGCoords(e);
    // Check polygons in reverse order (topmost first)
    for (let i = entries.length - 1; i >= 0; i--) {
      if (pointInPolygon(x, y, entries[i])) {
        setDragging(i);
        setDragStart({ x, y });
        return;
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (dragging === null || !dragStart) return;
    const { x, y } = getSVGCoords(e);
    const dx = x - dragStart.x;
    const dy = y - dragStart.y;
    setEntries((prev) => {
      const updated = [...prev];
      updated[dragging] = {
        ...updated[dragging],
        offsetX: updated[dragging].offsetX + dx,
        offsetY: updated[dragging].offsetY + dy,
      };
      return updated;
    });
    setDragStart({ x, y });
  };

  const handleMouseUp = () => {
    setDragging(null);
    setDragStart(null);
  };

  const handlePolygonClick = (idx: number) => {
    if (dragging !== null) return; // Don't toggle on drag end
    setSelectedPano((prev) => (prev === idx ? null : idx));
  };

  const handleCorrect = () => {
    if (!onCorrect || !alignmentData) return;
    // Preserve ALL original fields, only update camera_position based on drag offset
    const corrected: AlignmentData = {
      matches: entries.map((entry) => ({
        pano_name: entry.match.pano_name,
        room_label: entry.match.room_label,
        camera_position: [
          entry.match.camera_position[0] + entry.offsetX,
          entry.match.camera_position[1] + entry.offsetY,
        ] as [number, number],
        angle_deg: entry.match.angle_deg,
        scale: entry.match.scale,
      })),
      scale: alignmentData.scale,
    };
    onCorrect(corrected);
  };

  return (
    <div className="confirmation-gate">
      <div className="confirmation-gate__canvas">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${imgSize.w} ${imgSize.h}`}
          preserveAspectRatio="xMidYMid meet"
          style={{ width: "100%", maxHeight: "100%", cursor: dragging !== null ? "grabbing" : "default" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {/* Density image background */}
          {densityImagePath && (
            <image
              href={`file://${densityImagePath}`}
              x={0} y={0}
              width={imgSize.w} height={imgSize.h}
            />
          )}

          {/* Polygon outlines for each pano */}
          {entries.map((entry, idx) => {
            const color = COLORS[idx % COLORS.length];
            const { cx, cy } = centroid(entry);
            return (
              <g key={entry.match.pano_name} onClick={() => handlePolygonClick(idx)}>
                <polygon
                  points={polygonPoints(entry)}
                  fill={selectedPano === idx ? `${color}33` : `${color}1a`}
                  stroke={color}
                  strokeWidth={2}
                  style={{ cursor: "grab" }}
                />
                {/* Camera position marker */}
                <circle
                  cx={entry.match.camera_position[0] + entry.offsetX}
                  cy={entry.match.camera_position[1] + entry.offsetY}
                  r={5}
                  fill={color}
                />
                {/* Pano name label at centroid */}
                <text
                  x={cx} y={cy}
                  fill="#fff"
                  fontSize={12}
                  fontWeight="bold"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{ pointerEvents: "none", textShadow: "1px 1px 2px #000" }}
                >
                  {entry.match.pano_name}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Pano thumbnail overlay on click */}
        {selectedPano !== null && thumbnailSrc && (
          <div className="confirmation-gate__thumbnail">
            <img
              src={thumbnailSrc}
              alt={entries[selectedPano]?.match.pano_name}
              style={{ maxWidth: 320, maxHeight: 180, borderRadius: 6, border: "2px solid var(--accent)" }}
              onError={() => setThumbnailSrc(null)}
            />
            <p style={{ color: "var(--text-secondary)", fontSize: "0.8rem", marginTop: 4 }}>
              {entries[selectedPano]?.match.pano_name}
            </p>
          </div>
        )}
      </div>
      <div className="confirmation-gate__actions">
        <button className="btn btn--primary" onClick={onConfirm}>
          Confirm &amp; Continue
        </button>
        <button className="btn btn--secondary" onClick={handleCorrect}>
          Re-run with Corrections
        </button>
      </div>
    </div>
  );
}
```

---

## Part 9: Pipeline State Management Hook

### Task 9.1: usePipeline hook

**Files:**
- Create: `app/src/renderer/hooks/usePipeline.ts`

- [ ] **Step 1: Write `usePipeline.ts`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/hooks/usePipeline.ts`:

```typescript
import { useState, useEffect, useCallback, useRef } from "react";
import type { StageStatus, StageProgress, PipelineState, QualityTier } from "../types/pipeline";
import type { StageConfig } from "../../shared/constants";

interface UsePipelineOptions {
  stages: StageConfig[];
  projectId: string;
  startStage?: number;
}

export function usePipeline({ stages, projectId, startStage = 0 }: UsePipelineOptions) {
  const [state, setState] = useState<PipelineState>(() => ({
    stages: stages.map((_, i) =>
      i < startStage ? "complete" : "pending"
    ) as StageStatus[],
    currentStage: startStage,
    progress: null,
    logLines: [],
    stderrTail: "",
    elapsedMs: 0,
    qualityTier: "balanced" as QualityTier,
  }));

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(Date.now());

  // Start elapsed timer when a stage is active
  useEffect(() => {
    const isActive = state.stages[state.currentStage] === "active";
    if (isActive) {
      startTimeRef.current = Date.now();
      timerRef.current = setInterval(() => {
        setState((prev) => ({
          ...prev,
          elapsedMs: Date.now() - startTimeRef.current,
        }));
      }, 1000);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [state.currentStage, state.stages]);

  // Listen for IPC events from main process
  useEffect(() => {
    const api = window.electronAPI;

    api.onProgress((data: { current: number; total: number; message: string }) => {
      setState((prev) => ({ ...prev, progress: data }));
    });

    api.onLog((line: string) => {
      setState((prev) => ({
        ...prev,
        logLines: [...prev.logLines.slice(-500), line],
      }));
    });

    api.onStageComplete((stageIndex: number) => {
      setState((prev) => {
        const updated = [...prev.stages];
        updated[stageIndex] = "complete";
        const next = stageIndex + 1;
        if (next < stages.length) {
          updated[next] = stages[next].viewType === "confirmation" ? "confirmation" : "pending";
        }
        return {
          ...prev,
          stages: updated,
          currentStage: next < stages.length ? next : stageIndex,
          progress: null,
          logLines: [],
          elapsedMs: 0,
        };
      });
    });

    api.onStageError((data: { stageIndex: number; stderr: string }) => {
      setState((prev) => {
        const updated = [...prev.stages];
        updated[data.stageIndex] = "error";
        return {
          ...prev,
          stages: updated,
          stderrTail: data.stderr,
        };
      });
    });

    return () => {
      api.removeAllListeners("pipeline:progress");
      api.removeAllListeners("pipeline:log");
      api.removeAllListeners("pipeline:stage-complete");
      api.removeAllListeners("pipeline:stage-error");
    };
  }, [stages]);

  const runCurrentStage = useCallback(async () => {
    const stage = stages[state.currentStage];
    if (!stage || stage.scriptPaths.length === 0) return;

    setState((prev) => {
      const updated = [...prev.stages];
      updated[state.currentStage] = "active";
      return { ...prev, stages: updated, progress: null, logLines: [], stderrTail: "" };
    });

    // Run each script in the stage sequentially
    for (const scriptPath of stage.scriptPaths) {
      const configJsonPath = ""; // Will be set by the pipeline page based on project paths
      const result = await window.electronAPI.runStage(scriptPath, stage.condaEnv, configJsonPath);
      const typed = result as { ok?: boolean; error?: boolean; stderr?: string };
      if (typed.error) {
        setState((prev) => {
          const updated = [...prev.stages];
          updated[state.currentStage] = "error";
          return { ...prev, stages: updated, stderrTail: typed.stderr || "" };
        });
        return;
      }
    }

    // All scripts in stage succeeded
    setState((prev) => {
      const updated = [...prev.stages];
      updated[state.currentStage] = "complete";
      const next = state.currentStage + 1;
      return {
        ...prev,
        stages: updated,
        currentStage: next < stages.length ? next : state.currentStage,
        progress: null,
        logLines: [],
        elapsedMs: 0,
      };
    });
  }, [state.currentStage, stages]);

  const retry = useCallback(() => {
    setState((prev) => {
      const updated = [...prev.stages];
      updated[prev.currentStage] = "pending";
      return { ...prev, stages: updated, stderrTail: "" };
    });
  }, []);

  const setQualityTier = useCallback((tier: QualityTier) => {
    setState((prev) => ({ ...prev, qualityTier: tier }));
  }, []);

  const confirm = useCallback(() => {
    setState((prev) => {
      const updated = [...prev.stages];
      updated[prev.currentStage] = "complete";
      const next = prev.currentStage + 1;
      return {
        ...prev,
        stages: updated,
        currentStage: next < stages.length ? next : prev.currentStage,
      };
    });
  }, [stages]);

  return {
    state,
    runCurrentStage,
    retry,
    confirm,
    setQualityTier,
  };
}
```

---

### Task 9.2: useProject hook

**Files:**
- Create: `app/src/renderer/hooks/useProject.ts`

- [ ] **Step 1: Write `useProject.ts`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/hooks/useProject.ts`:

```typescript
import { useState, useEffect, useCallback } from "react";
import type { Project } from "../types/project";

export function useProject(projectId: string | undefined) {
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!projectId) {
      setLoading(false);
      return;
    }
    window.electronAPI
      .getProjects()
      .then((projects: Project[]) => {
        const found = projects.find((p) => p.id === projectId) || null;
        setProject(found);
        setLoading(false);
      });
  }, [projectId]);

  const updateProject = useCallback(
    async (patch: Partial<Project>) => {
      if (!projectId) return;
      const updated = await window.electronAPI.updateProject(projectId, patch);
      setProject(updated as Project);
    },
    [projectId]
  );

  return { project, loading, updateProject };
}
```

---

## Part 10: Pipeline Page (Integration)

### Task 10.1: PipelinePage

**Files:**
- Create: `app/src/renderer/pages/PipelinePage.tsx`

- [ ] **Step 1: Write `PipelinePage.tsx`**

Create `/home/ruoyu/scan2measure-webframework/app/src/renderer/pages/PipelinePage.tsx`:

```tsx
import React, { useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import StageSidebar from "../components/StageSidebar";
import StageCanvas from "../components/StageCanvas";
import QualityTierSelect from "../components/QualityTierSelect";
import { useProject } from "../hooks/useProject";
import { usePipeline } from "../hooks/usePipeline";
import { FULL_PIPELINE_STAGES, MESH_ONLY_STAGES } from "../../shared/constants";
import "../styles/pipeline.css";

export default function PipelinePage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const { project, loading, updateProject } = useProject(projectId);

  // Determine which stage list to use based on project type
  const stages =
    project?.type === "mesh_only" ? MESH_ONLY_STAGES : FULL_PIPELINE_STAGES;

  const { state, runCurrentStage, retry, confirm, setQualityTier } = usePipeline({
    stages,
    projectId: projectId || "",
    startStage: project?.lastCompletedStage !== undefined
      ? project.lastCompletedStage + 1
      : 0,
  });

  // Update project status when pipeline state changes
  useEffect(() => {
    if (!project) return;
    const allDone = state.stages.every((s) => s === "complete");
    const hasError = state.stages.some((s) => s === "error");
    const newStatus = allDone ? "completed" : hasError ? "error" : "in_progress";
    if (project.status !== newStatus) {
      updateProject({ status: newStatus, lastCompletedStage: state.currentStage - 1 });
    }
  }, [state.stages, project, updateProject, state.currentStage]);

  if (loading) {
    return <div className="pipeline-page" style={{ padding: 32 }}>Loading project...</div>;
  }

  if (!project) {
    return (
      <div className="pipeline-page" style={{ padding: 32 }}>
        <p>Project not found.</p>
        <button className="btn btn--secondary" onClick={() => navigate("/")}>
          Back to Home
        </button>
      </div>
    );
  }

  const currentStageConfig = stages[state.currentStage];
  const currentStatus = state.stages[state.currentStage];

  // Auto-run the current stage when it transitions to pending (and is not confirmation)
  useEffect(() => {
    if (
      currentStatus === "pending" &&
      currentStageConfig &&
      currentStageConfig.viewType !== "confirmation" &&
      currentStageConfig.scriptPaths.length > 0
    ) {
      runCurrentStage();
    }
  }, [currentStatus, currentStageConfig, runCurrentStage]);

  // Build sidebar stage data
  const sidebarStages = stages.map((s, i) => ({
    name: s.name,
    status: state.stages[i],
  }));

  // Determine artifacts for current stage based on project paths
  const artifacts = {
    densityImagePath: project.outputs.densityImage,
    glbPath: project.outputs.meshGlb,
    // Additional artifact paths would be resolved from the project output directory
    // based on the current stage id and STAGE_OUTPUT_DIRS mapping
  };

  // Show quality tier selector before meshing stage
  const showQualitySelect =
    currentStageConfig?.id === "meshing" && currentStatus !== "active";

  return (
    <div className="pipeline-page">
      <StageSidebar
        stages={sidebarStages}
        currentStage={state.currentStage}
        onBack={() => navigate("/")}
      />

      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        {showQualitySelect && (
          <div style={{ padding: "12px 24px", borderBottom: "1px solid var(--bg-card)" }}>
            <QualityTierSelect
              value={state.qualityTier}
              onChange={setQualityTier}
              disabled={currentStatus === "active"}
            />
          </div>
        )}

        <StageCanvas
          viewType={currentStageConfig?.viewType || "progress"}
          stageStatus={currentStatus}
          stageName={currentStageConfig?.name || ""}
          stageDescription={currentStageConfig?.description || ""}
          elapsedMs={state.elapsedMs}
          progress={state.progress}
          logLines={state.logLines}
          stderrTail={state.stderrTail}
          artifacts={artifacts}
          onConfirm={confirm}
          onRetry={retry}
          onBack={() => navigate("/")}
        />

        {/* Launch Tour button on final stage */}
        {currentStageConfig?.id === "done" && currentStatus === "complete" && project.outputs.meshGlb && (
          <div style={{ padding: 16, textAlign: "center", borderTop: "1px solid var(--bg-card)" }}>
            <button
              className="btn btn--primary"
              onClick={() =>
                window.electronAPI.launchUnity(
                  project.outputs.meshGlb!,
                  project.outputs.densityImage
                )
              }
            >
              Launch Virtual Tour
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
```

---

## Part 11: Electron Builder Configuration

### Task 11.1: electron-builder.yml

**Files:**
- Create: `app/electron-builder.yml`

- [ ] **Step 1: Write `electron-builder.yml`**

Create `/home/ruoyu/scan2measure-webframework/app/electron-builder.yml`:

```yaml
appId: com.scan2measure.app
productName: scan2measure
directories:
  output: release
  buildResources: assets
files:
  - dist/**/*
  - node_modules/**/*
  - package.json
win:
  target: dir
  icon: assets/icons/icon.ico
```

---

## Part 12: Smoke Test and Verification

### Task 12.1: Typecheck all files

- [ ] **Step 1: Run full typecheck**

```bash
cd /home/ruoyu/scan2measure-webframework/app && npx tsc --noEmit 2>&1 | head -20
```

Fix any type errors reported until `npx tsc --noEmit` exits cleanly.

---

### Task 12.2: Build check

- [ ] **Step 1: Run Vite build to verify bundling works**

```bash
cd /home/ruoyu/scan2measure-webframework/app && npx vite build 2>&1 | tail -10
```

Expected: build completes without errors, output in `dist/`.

---

### Task 12.3: Dev launch smoke test

- [ ] **Step 1: Start dev server and Electron (manual verification)**

```bash
cd /home/ruoyu/scan2measure-webframework/app && npm run electron:dev
```

Expected behavior:
- Electron window opens at 1400x900
- Home screen displays with three entry cards (Full Pipeline, Mesh Only, Tour Only)
- No console errors in DevTools
- Clicking "Full Pipeline" opens the file upload dialog

---

### Task 12.4: Verify pipeline engine progress parsing

- [ ] **Step 1: Create a minimal test for `parseProgressLine`**

Create `/home/ruoyu/scan2measure-webframework/app/src/main/__tests__/pipeline-engine.test.ts`:

```typescript
// Manual test: run with ts-node or node after tsc compilation
// This verifies the progress parsing regex works correctly.

function parseProgressLine(line: string) {
  const match = line.match(/^\[PROGRESS\]\s+(\d+)\s+(\d+)\s+(.*)$/);
  if (!match) return null;
  return {
    current: parseInt(match[1], 10),
    total: parseInt(match[2], 10),
    message: match[3].trim(),
  };
}

// Test cases
const tests = [
  {
    input: "[PROGRESS] 45 100 Processing tile 3 of 8",
    expected: { current: 45, total: 100, message: "Processing tile 3 of 8" },
  },
  {
    input: "[PROGRESS] 0 10 Starting...",
    expected: { current: 0, total: 10, message: "Starting..." },
  },
  {
    input: "Some regular log output",
    expected: null,
  },
  {
    input: "[INFO] Not a progress line",
    expected: null,
  },
];

let passed = 0;
for (const t of tests) {
  const result = parseProgressLine(t.input);
  const ok = JSON.stringify(result) === JSON.stringify(t.expected);
  console.log(ok ? "PASS" : "FAIL", t.input);
  if (ok) passed++;
}
console.log(`${passed}/${tests.length} tests passed`);
```

Run:

```bash
cd /home/ruoyu/scan2measure-webframework/app && npx ts-node src/main/__tests__/pipeline-engine.test.ts
```

Expected: `4/4 tests passed`

---

## Summary

| Part | Description | Tasks | Key Files |
|------|-------------|-------|-----------|
| 1 | Scaffolding | 1.1-1.6 | `package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `main.tsx`, `index.ts`, `preload.ts`, `global.css` |
| 2 | Types + constants | 2.1-2.3 | `pipeline.ts`, `project.ts`, `constants.ts` |
| 3 | Pipeline engine | 3.1-3.4 | `pipeline-engine.ts`, `project-store.ts`, `unity-launcher.ts`, IPC wiring in `index.ts` |
| 4 | App shell | 4.1 | `App.tsx` |
| 5 | Home screen | 5.1-5.5 | `EntryCard.tsx`, `RecentProjects.tsx`, `FileUploadDialog.tsx`, `HomePage.tsx`, `TourOnlyPage.tsx`, `home.css` |
| 6 | Pipeline view | 6.1-6.3 | `StageSidebar.tsx`, `StageCanvas.tsx`, `pipeline.css` |
| 7 | Visualization | 7.1-7.6 | `Canvas2D.tsx`, `ThreeViewer.tsx`, `ProgressPanel.tsx`, `ErrorPanel.tsx`, `QualityTierSelect.tsx`, `components.css` |
| 8 | Confirmation | 8.1 | `ConfirmationGate.tsx` |
| 9 | State hooks | 9.1-9.2 | `usePipeline.ts`, `useProject.ts` |
| 10 | Integration | 10.1 | `PipelinePage.tsx` |
| 11 | Packaging | 11.1 | `electron-builder.yml` |
| 12 | Verification | 12.1-12.4 | Typecheck, build, dev launch, unit test |
