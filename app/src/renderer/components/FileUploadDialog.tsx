import { useState } from "react";

declare global {
  interface Window {
    electronAPI: {
      openPLY: () => Promise<string | null>;
      openPanoramas: () => Promise<string[] | null>;
      openGLB: () => Promise<string | null>;
      startPipeline: (projectId: string, stageIndex: number) => Promise<unknown>;
      cancelPipeline: () => Promise<unknown>;
      retryStage: (projectId: string, stageIndex: number) => Promise<unknown>;
      writeConfig: (projectId: string, stageId: string, overrides?: Record<string, unknown>) => Promise<unknown>;
      runStage: (scriptPath: string, condaEnv: string, configJsonPath: string) => Promise<unknown>;
      downsamplePly: (inputPath: string, outputPath: string) => Promise<unknown>;
      onProgress: (callback: (data: { current: number; total: number; message: string }) => void) => void;
      onStageComplete: (callback: (stageIndex: number) => void) => void;
      onStageError: (callback: (data: { stageIndex: number; stderr: string }) => void) => void;
      onLog: (callback: (line: string) => void) => void;
      getProjects: () => Promise<unknown[]>;
      createProject: (data: unknown) => Promise<unknown>;
      updateProject: (id: string, data: unknown) => Promise<unknown>;
      deleteProject: (id: string) => Promise<unknown>;
      launchUnity: (glbPath: string, minimapPath?: string, metadataPath?: string) => Promise<unknown>;
      findMinimapPng: (glbPath: string) => Promise<unknown>;
      resolveArtifacts: (projectId: string, stageId: string) => Promise<unknown>;
      readImage: (filePath: string) => Promise<string>;
      removeAllListeners: (channel: string) => void;
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
