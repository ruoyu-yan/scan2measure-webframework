import { useState } from "react";
import { useNavigate } from "react-router-dom";

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
