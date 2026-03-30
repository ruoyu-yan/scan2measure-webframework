import { app, BrowserWindow, ipcMain, dialog } from "electron";
import path from "node:path";
import fs from "node:fs";
import { spawn } from "node:child_process";
import { PipelineEngine } from "./pipeline-engine";
import { ProjectStore } from "./project-store";
import { launchUnity } from "./unity-launcher";
import { initLogger, logInfo, logError, closeLogger, getLogPath } from "./logger";

// Project root is two levels up from app/dist/main/
const PROJECT_ROOT = path.resolve(__dirname, "../../..");

let mainWindow: BrowserWindow | null = null;
let pipelineEngine: PipelineEngine | null = null;
let projectStore: ProjectStore;

initLogger(PROJECT_ROOT);
logInfo("app", `Project root: ${PROJECT_ROOT}`);

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

  projectStore = new ProjectStore(PROJECT_ROOT);
  pipelineEngine = new PipelineEngine(mainWindow);
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  logInfo("app", "All windows closed, quitting");
  closeLogger();
  app.quit();
});

// -- IPC: File dialogs --------------------------------------------------------

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

// -- IPC: Project CRUD --------------------------------------------------------

ipcMain.handle("project:list", () => projectStore.list());

ipcMain.handle("project:create", (_event, data) => projectStore.create(data));

ipcMain.handle("project:update", (_event, id: string, patch) =>
  projectStore.update(id, patch)
);

ipcMain.handle("project:delete", (_event, id: string) =>
  projectStore.delete(id)
);

// -- IPC: Pipeline control ----------------------------------------------------

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
  async (_event, _projectId: string, _stageIndex: number) => {
    // Same as start -- the renderer hook manages retry logic
    return { ok: true, projectId: _projectId, stageIndex: _stageIndex };
  }
);

// -- IPC: Run a single stage subprocess ---------------------------------------

ipcMain.handle(
  "pipeline:run-stage",
  async (
    _event,
    scriptPath: string,
    condaEnv: string,
    configJsonPath: string
  ) => {
    if (!pipelineEngine) return { error: "No window" };
    logInfo("ipc", `pipeline:run-stage script=${scriptPath} env=${condaEnv} config=${configJsonPath}`);
    try {
      await pipelineEngine.runStage({
        scriptPath: path.resolve(PROJECT_ROOT, scriptPath),
        condaEnv,
        configJsonPath,
        projectRoot: PROJECT_ROOT,
      });
      return { ok: true };
    } catch (err) {
      logError("ipc", `pipeline:run-stage failed: ${(err as Error).message}`);
      return {
        error: true,
        stderr: pipelineEngine.getStderrTail(),
        message: (err as Error).message,
      };
    }
  }
);

// -- IPC: Unity launcher ------------------------------------------------------

ipcMain.handle(
  "unity:launch",
  (_event, glbPath: string, minimapPath?: string, metadataPath?: string) => {
    logInfo("ipc", `unity:launch glb=${glbPath}`);
    const launched = launchUnity(PROJECT_ROOT, glbPath, minimapPath, metadataPath);
    if (!launched) logError("ipc", "Unity executable not found");
    return { ok: launched, error: launched ? null : "Unity executable not found" };
  }
);

// -- IPC: Find minimap PNG near a GLB file ------------------------------------

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

// -- IPC: PLY downsample for preview ------------------------------------------

ipcMain.handle("log:path", () => getLogPath());

ipcMain.handle(
  "downsample-ply",
  async (_event, inputPath: string, outputPath: string) => {
    logInfo("ipc", `downsample-ply input=${inputPath} output=${outputPath}`);
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
