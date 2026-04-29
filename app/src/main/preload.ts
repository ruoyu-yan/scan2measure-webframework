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

  // Write stage config JSON (returns { ok, configPath } or { error })
  writeConfig: (projectId: string, stageId: string, overrides?: Record<string, unknown>) =>
    ipcRenderer.invoke("pipeline:write-config", projectId, stageId, overrides || {}),

  // Run a single stage subprocess
  runStage: (scriptPath: string, condaEnv: string, configJsonPath: string) =>
    ipcRenderer.invoke("pipeline:run-stage", scriptPath, condaEnv, configJsonPath),

  // PLY downsample for preview
  downsamplePly: (inputPath: string, outputPath: string) =>
    ipcRenderer.invoke("downsample-ply", inputPath, outputPath),

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

  // Artifact resolution
  resolveArtifacts: (projectId: string, stageId: string) =>
    ipcRenderer.invoke("artifacts:resolve", projectId, stageId),

  // Read image as base64 data URI
  readImage: (filePath: string) =>
    ipcRenderer.invoke("artifacts:read-image", filePath),

  // Environment management
  checkEnvironment: () => ipcRenderer.invoke("environment:check"),
  setupEnvironment: (envName: string) =>
    ipcRenderer.invoke("environment:setup", envName),
  onSetupProgress: (callback: (data: { env: string; phase: string; message: string }) => void) =>
    ipcRenderer.on("environment:setup-progress", (_event, data) => callback(data)),
  onSetupLog: (callback: (line: string) => void) =>
    ipcRenderer.on("environment:setup-log", (_event, line) => callback(line)),
  cancelEnvironmentSetup: () => ipcRenderer.invoke("environment:cancel"),

  // Remove listeners (restricted to pipeline channels only)
  removeAllListeners: (channel: string) => {
    const ALLOWED_CHANNELS = [
      "pipeline:progress",
      "pipeline:log",
      "pipeline:stage-complete",
      "pipeline:stage-error",
      "environment:setup-progress",
      "environment:setup-log",
    ];
    if (ALLOWED_CHANNELS.includes(channel)) {
      ipcRenderer.removeAllListeners(channel);
    }
  },
});
