import { app, BrowserWindow, ipcMain, dialog, protocol, net } from "electron";
import path from "node:path";
import fs from "node:fs";
import { spawn } from "node:child_process";
import { PipelineEngine } from "./pipeline-engine";
import { ProjectStore } from "./project-store";
import { EnvironmentManager } from "./environment-manager";
import { launchUnity } from "./unity-launcher";
import { initLogger, logInfo, logError, closeLogger, getLogPath } from "./logger";

/**
 * Resolve the monorepo root directory.
 * - Packaged (AppImage): derive from process.env.APPIMAGE path
 * - Packaged (other):    fall back to SCAN2MEASURE_ROOT env var
 * - Development:         walk up from __dirname (app/dist/main/)
 */
function resolveProjectRoot(): string {
  if (app.isPackaged) {
    const appImagePath = process.env.APPIMAGE;
    if (appImagePath) {
      // e.g. <repo>/app/release/scan2measure.AppImage → 2 dirs up = repo root
      return path.resolve(path.dirname(appImagePath), "../..");
    }
    if (process.env.SCAN2MEASURE_ROOT) {
      return process.env.SCAN2MEASURE_ROOT;
    }
    return path.resolve(path.dirname(process.execPath), "../..");
  }
  // Dev mode: __dirname = app/dist/main/ → ../../.. = repo root
  return path.resolve(__dirname, "../../..");
}

const PROJECT_ROOT = resolveProjectRoot();

let mainWindow: BrowserWindow | null = null;
let pipelineEngine: PipelineEngine | null = null;
let envManager: EnvironmentManager | null = null;
let projectStore: ProjectStore;

initLogger(PROJECT_ROOT);
logInfo("app", `Project root: ${PROJECT_ROOT}`);
logInfo("app", `app.isPackaged: ${app.isPackaged}`);
if (process.env.APPIMAGE) {
  logInfo("app", `APPIMAGE: ${process.env.APPIMAGE}`);
}
if (!fs.existsSync(path.join(PROJECT_ROOT, "src"))) {
  logError("app", `PROJECT_ROOT validation failed: ${path.join(PROJECT_ROOT, "src")} does not exist`);
}

// Register custom protocol for serving local files to the renderer
protocol.registerSchemesAsPrivileged([
  { scheme: "local-file", privileges: { standard: true, supportFetchAPI: true, stream: true } },
]);

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
    mainWindow.loadFile(path.join(__dirname, "../index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  projectStore = new ProjectStore(PROJECT_ROOT);
  pipelineEngine = new PipelineEngine(mainWindow);
  envManager = new EnvironmentManager(mainWindow, PROJECT_ROOT);
}

app.whenReady().then(() => {
  // Handle local-file:// protocol to serve local files to renderer
  protocol.handle("local-file", async (request) => {
    // Standard-scheme URL parsing may absorb the leading path component as a host
    // (e.g. local-file:///home/... → host="home", pathname="/...").
    // Reconstruct the absolute path from host + pathname to preserve the leading /.
    const parsed = new URL(request.url);
    const filePath = decodeURIComponent(
      (parsed.host ? "/" + parsed.host : "") + parsed.pathname
    );
    logInfo("protocol", `local-file request: ${request.url} -> ${filePath}`);
    try {
      return await net.fetch("file://" + filePath);
    } catch (err) {
      logError("protocol", `local-file fetch failed for "${filePath}": ${(err as Error).message}`);
      return new Response(`Failed to load local file: ${filePath}`, {
        status: 404,
        headers: { "Content-Type": "text/plain" },
      });
    }
  });
  createWindow();
});

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

// -- IPC: Environment management -----------------------------------------------

ipcMain.handle("environment:check", async () => {
  if (!envManager) return { error: true, message: "No window" };
  try {
    const status = await envManager.check();
    return { ok: true, status };
  } catch (err) {
    logError("ipc", `environment:check failed: ${(err as Error).message}`);
    return { error: true, message: (err as Error).message };
  }
});

ipcMain.handle("environment:setup", async (_event, envName: "scan_env" | "sam3") => {
  if (!envManager) return { error: true, message: "No window" };
  logInfo("ipc", `environment:setup starting for ${envName}`);
  try {
    await envManager.setup(envName);
    return { ok: true };
  } catch (err) {
    logError("ipc", `environment:setup failed for ${envName}: ${(err as Error).message}`);
    return { error: true, message: (err as Error).message };
  }
});

ipcMain.handle("environment:cancel", () => {
  envManager?.cancel();
  return { ok: true };
});

// -- IPC: Write stage config JSON and return its path -------------------------

ipcMain.handle(
  "pipeline:write-config",
  (_event, projectId: string, stageId: string, overrides: Record<string, unknown> = {}) => {
    const project = projectStore.get(projectId);
    if (!project) return { error: "Project not found" };

    const projDir = projectStore.ensureProjectDir(PROJECT_ROOT, projectId);
    const pcPath = project.inputs.pointCloud || "";
    const pcName = pcPath ? path.basename(pcPath, path.extname(pcPath)) : "";
    const panoPaths = project.inputs.panoramas || [];
    // Derive pano_names (bare stems) from full paths: /data/pano/raw/TMB_office1.jpg -> "TMB_office1"
    const panoNames = panoPaths.map((p: string) => path.basename(p, path.extname(p)));

    // Derive inter-stage paths from point cloud name
    const densityImageDir = pcName
      ? path.join(PROJECT_ROOT, "data", "density_image", pcName)
      : "";
    const sam3SegDir = path.join(PROJECT_ROOT, "data", "sam3_room_segmentation");
    const poseEstimatesDir = pcName
      ? path.join(PROJECT_ROOT, "data", "pose_estimates", "multiroom")
      : "";
    const texturedPlyDir = pcName
      ? path.join(PROJECT_ROOT, "data", "textured_point_cloud", pcName)
      : "";

    // Stage-specific input_path mapping
    const stageInputPath: Record<string, string> = {
      density_image: pcPath,                    // point cloud -> density image
      sam3_segmentation: densityImageDir,        // density image dir -> SAM3 masks
      sam3_polygons: densityImageDir,             // density image dir -> polygons
      pano_footprints: "",                        // uses pano_name override per invocation
      polygon_matching: densityImageDir,          // density image + pano footprints
      line_detection_3d: pcPath,                  // point cloud -> 3D lines
      line_detection_2d: "",                      // uses pano_name override per invocation
      pose_estimation: "",                        // uses precomputed data
      colorization: pcPath,                       // point cloud + panos
      meshing: pcPath,                              // raw point cloud -> PoissonRecon + texrecon -> GLB
    };

    // Stage-specific output_dir mapping (default: mesh dir)
    const defaultOutputDir = pcName ? path.join(PROJECT_ROOT, "data", "mesh", pcName) : "";
    const debugRendererDir = pcName
      ? path.join(PROJECT_ROOT, "data", "debug_renderer", pcName)
      : "";
    const panoProcessingDir = path.join(PROJECT_ROOT, "data", "sam3_pano_processing");
    const stageOutputDir: Record<string, string> = {
      density_image: densityImageDir,                  // data/density_image/<pcName>
      sam3_segmentation: sam3SegDir,                    // data/sam3_room_segmentation
      sam3_polygons: pcName ? path.join(sam3SegDir, pcName) : "",  // data/sam3_room_segmentation/<pcName>
      pano_footprints: panoProcessingDir,               // data/sam3_pano_processing
      polygon_matching: pcName ? path.join(sam3SegDir, pcName) : "",  // data/sam3_room_segmentation/<pcName>
      line_detection_3d: debugRendererDir,              // data/debug_renderer/<pcName>
      line_detection_2d: path.join(PROJECT_ROOT, "data", "pano", "2d_feature_extracted"),
      pose_estimation: poseEstimatesDir,                // data/pose_estimates/multiroom
      colorization: texturedPlyDir,                     // data/textured_point_cloud/<pcName>
      // meshing: uses defaultOutputDir (data/mesh/<pcName>) — correct as-is
    };

    // Per-pano stageIds look like "pano_footprints_<panoName>" — extract the
    // base stage ID so the output_dir lookup still works.
    const baseStageId = Object.keys(stageOutputDir).find((key) =>
      stageId === key || stageId.startsWith(key + "_")
    ) || stageId;

    const config: Record<string, unknown> = {
      point_cloud_path: pcPath,
      point_cloud_name: pcName,
      map_name: pcName,
      panorama_paths: panoPaths,
      pano_names: panoNames,
      project_dir: projDir,
      quality_tier: project.qualityTier || "balanced",
      // Inter-stage derived paths
      density_image_dir: densityImageDir,
      input_path: stageInputPath[baseStageId] || "",
      sam3_segmentation_dir: sam3SegDir,
      pose_estimates_dir: poseEstimatesDir,
      panorama_dir: path.join(PROJECT_ROOT, "data", "pano", "raw"),
      pose_json_path: path.join(poseEstimatesDir, "local_filter_results.json"),
      output_dir: stageOutputDir[baseStageId] || defaultOutputDir,
      ...overrides,
    };

    // Per-pano overrides: add script-expected aliases
    if (config.pano_name) {
      config.room_name = config.room_name || config.pano_name;
      // Set pano_path from input_path if the script expects it
      if (config.input_path) {
        config.pano_path = config.pano_path || config.input_path;
      }
    }

    const configPath = path.join(projDir, `${stageId}_config.json`);
    fs.mkdirSync(path.dirname(configPath), { recursive: true });
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
    logInfo("ipc", `pipeline:write-config ${stageId} -> ${configPath}`);
    return { ok: true, configPath };
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

// -- IPC: Artifact resolution -------------------------------------------------

ipcMain.handle(
  "artifacts:resolve",
  (_event, projectId: string, stageId: string) => {
    const project = projectStore.get(projectId);
    if (!project) return { stageId, artifacts: {} };

    const pcPath = project.inputs.pointCloud || "";
    const pcName = pcPath ? path.basename(pcPath, path.extname(pcPath)) : "";
    const panoPaths = project.inputs.panoramas || [];
    const panoNames = panoPaths.map((p: string) =>
      path.basename(p, path.extname(p))
    );

    const artifacts: Record<string, unknown> = {};

    const existsOrNull = (p: string): string | null =>
      fs.existsSync(p) ? p : null;

    const filterExisting = (paths: string[]): string[] =>
      paths.filter((p) => fs.existsSync(p));

    switch (stageId) {
      case "density_image": {
        const imgPath = path.join(
          PROJECT_ROOT, "data", "density_image", pcName, `${pcName}.png`
        );
        const existing = existsOrNull(imgPath);
        if (existing) artifacts.images = [existing];
        break;
      }
      case "sam3_segmentation": {
        // Show the overlay image: <pcName>_overlay.png
        const overlayImg = path.join(
          PROJECT_ROOT, "data", "sam3_room_segmentation", pcName, `${pcName}_overlay.png`
        );
        const existing = existsOrNull(overlayImg);
        if (existing) artifacts.images = [existing];
        break;
      }
      case "sam3_polygons": {
        // Show density image with polygon outlines overlaid
        const densityImg = path.join(
          PROJECT_ROOT, "data", "density_image", pcName, `${pcName}.png`
        );
        const polygonOverlayImg = path.join(
          PROJECT_ROOT, "data", "sam3_room_segmentation", pcName, `${pcName}_polygon_overlay.png`
        );
        const polygonsJson = path.join(
          PROJECT_ROOT, "data", "sam3_room_segmentation", pcName, `${pcName}_polygons.json`
        );
        if (fs.existsSync(densityImg)) {
          artifacts.densityImage = densityImg;
        }
        // Use polygon overlay for filmstrip thumbnail (shows polygons on density image)
        // Falls back to raw density image if overlay not yet generated
        if (fs.existsSync(polygonOverlayImg)) {
          artifacts.images = [polygonOverlayImg];
        } else if (fs.existsSync(densityImg)) {
          artifacts.images = [densityImg];
        }
        if (fs.existsSync(polygonsJson)) artifacts.polygonsJson = polygonsJson;
        break;
      }
      case "pano_footprints": {
        const imgs = filterExisting(
          panoNames.map((pn: string) =>
            path.join(PROJECT_ROOT, "data", "sam3_pano_processing", pn, "debug.png")
          )
        );
        if (imgs.length > 0) artifacts.images = imgs;
        break;
      }
      case "polygon_matching": {
        const alignPng = path.join(
          PROJECT_ROOT, "data", "sam3_room_segmentation", pcName, "demo6_alignment.png"
        );
        const alignJson = path.join(
          PROJECT_ROOT, "data", "sam3_room_segmentation", pcName, "demo6_alignment.json"
        );
        const existingPng = existsOrNull(alignPng);
        if (existingPng) artifacts.images = [existingPng];
        const existingJson = existsOrNull(alignJson);
        if (existingJson) artifacts.alignmentJson = existingJson;
        break;
      }
      case "confirm_matching": {
        const densityImg = path.join(
          PROJECT_ROOT, "data", "density_image", pcName, `${pcName}.png`
        );
        const alignJson = path.join(
          PROJECT_ROOT, "data", "sam3_room_segmentation", pcName, "demo6_alignment.json"
        );
        const polyJson = path.join(
          PROJECT_ROOT, "data", "sam3_room_segmentation", pcName, `${pcName}_polygons.json`
        );
        if (fs.existsSync(densityImg)) artifacts.densityImage = densityImg;
        if (fs.existsSync(alignJson)) artifacts.alignmentJson = alignJson;
        if (fs.existsSync(polyJson)) artifacts.polygonsJson = polyJson;
        const panoThumbnails: Record<string, string> = {};
        for (const pn of panoNames) {
          const thumbPath = path.join(PROJECT_ROOT, "data", "pano", "raw", `${pn}.jpg`);
          if (fs.existsSync(thumbPath)) panoThumbnails[pn] = thumbPath;
        }
        if (Object.keys(panoThumbnails).length > 0) {
          artifacts.panoThumbnails = panoThumbnails;
        }
        break;
      }
      case "line_detection_3d": {
        // Prefer clustered OBJ (direction-colored via MTL) over raw lines OBJ
        const clusteredObj = path.join(
          PROJECT_ROOT, "data", "debug_renderer", pcName, "clustered_lines.obj"
        );
        const rawObj = path.join(
          PROJECT_ROOT, "data", "debug_renderer", pcName, `${pcName}_lines.obj`
        );
        if (fs.existsSync(clusteredObj)) artifacts.objPath = clusteredObj;
        else if (fs.existsSync(rawObj)) artifacts.objPath = rawObj;
        break;
      }
      case "line_detection_2d": {
        const imgs = filterExisting(
          panoNames.map((pn: string) =>
            path.join(
              PROJECT_ROOT, "data", "pano", "2d_feature_extracted",
              `${pn}_v2`, "grouped_lines.png"
            )
          )
        );
        if (imgs.length > 0) artifacts.images = imgs;
        break;
      }
      case "pose_estimation": {
        // Prefer composite (all cameras on one image) over per-pano gallery
        const composite = path.join(
          PROJECT_ROOT, "data", "pose_estimates", "multiroom", "composite_topdown.png"
        );
        if (fs.existsSync(composite)) {
          artifacts.images = [composite];
        } else {
          const imgs = filterExisting(
            panoNames.map((pn: string) =>
              path.join(
                PROJECT_ROOT, "data", "pose_estimates", "multiroom", pn, "vis", "topdown.png"
              )
            )
          );
          if (imgs.length > 0) artifacts.images = imgs;
        }
        break;
      }
      case "colorization":
      case "confirm_colorization": {
        const texturedPly = path.join(
          PROJECT_ROOT, "data", "textured_point_cloud", pcName, `${pcName}_textured.ply`
        );
        logInfo("ipc", `artifacts:resolve ${stageId} plyPath candidate: ${texturedPly} (exists=${fs.existsSync(texturedPly)})`);
        if (fs.existsSync(texturedPly)) artifacts.plyPath = texturedPly;
        break;
      }
      case "meshing":
      case "done": {
        // Prefer _textured GLB (mesh_pipeline.py), then legacy _texrecon / base name
        const meshDir = path.join(PROJECT_ROOT, "data", "mesh", pcName);
        const texturedGlb = path.join(meshDir, `${pcName}_textured.glb`);
        const texreconGlb = path.join(meshDir, `${pcName}_texrecon.glb`);
        const baseGlb = path.join(meshDir, `${pcName}.glb`);
        if (fs.existsSync(texturedGlb)) {
          artifacts.glbPath = texturedGlb;
        } else if (fs.existsSync(texreconGlb)) {
          artifacts.glbPath = texreconGlb;
        } else if (fs.existsSync(baseGlb)) {
          artifacts.glbPath = baseGlb;
        } else {
          // Search for any GLB in the mesh directory
          try {
            const files = fs.readdirSync(meshDir).filter((f: string) => f.endsWith(".glb"));
            if (files.length > 0) artifacts.glbPath = path.join(meshDir, files[0]);
          } catch { /* meshDir may not exist yet */ }
        }
        if (artifacts.glbPath) {
          logInfo("ipc", `artifacts:resolve ${stageId} glbPath: ${artifacts.glbPath}`);
        }
        break;
      }
      default:
        break;
    }

    logInfo("ipc", `artifacts:resolve stageId=${stageId} keys=${Object.keys(artifacts).join(",")}`);
    return { stageId, artifacts };
  }
);

// -- IPC: Read image as base64 data URI ---------------------------------------

ipcMain.handle("artifacts:read-image", (_event, filePath: string) => {
  try {
    const data = fs.readFileSync(filePath);
    const ext = path.extname(filePath).toLowerCase();
    const mime =
      ext === ".png"
        ? "image/png"
        : ext === ".jpg" || ext === ".jpeg"
          ? "image/jpeg"
          : "application/octet-stream";
    return `data:${mime};base64,${data.toString("base64")}`;
  } catch (err) {
    logError("ipc", `artifacts:read-image failed for ${filePath}: ${(err as Error).message}`);
    return null;
  }
});

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
            "run", "-n", "scan_env",
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
