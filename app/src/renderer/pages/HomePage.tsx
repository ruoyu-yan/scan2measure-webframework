import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import EntryCard from "../components/EntryCard";
import RecentProjects from "../components/RecentProjects";
import FileUploadDialog from "../components/FileUploadDialog";
import EnvironmentSetup from "../components/EnvironmentSetup";
import { useEnvironment } from "../hooks/useEnvironment";
import type { Project } from "../types/project";
import "../styles/home.css";

export default function HomePage() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[]>([]);
  const [showDialog, setShowDialog] = useState(false);
  const { state: envState, checkEnvironment: retryCheck, setupEnvironments, cancelSetup } = useEnvironment();

  const envReady = envState.checkState === "ready";

  useEffect(() => {
    window.electronAPI.getProjects().then((list) => setProjects(list as Project[]));
  }, []);

  const handleFullPipeline = () => {
    if (!envReady) return;
    setShowDialog(true);
  };
  const handleTourOnly = () => navigate("/tour-only");

  const handleFileSubmit = async (data: {
    plyPath: string;
    panoramas?: string[];
    projectName: string;
  }) => {
    const project = await window.electronAPI.createProject({
      name: data.projectName,
      type: "full_pipeline",
      inputs: {
        pointCloud: data.plyPath,
        panoramas: data.panoramas,
      },
      qualityTier: "balanced",
    });
    setShowDialog(false);
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

      {envState.checkState === "checking" && (
        <div className="env-setup">
          <div className="env-setup__spinner" />
          <p className="env-setup__desc">Checking conda environments...</p>
        </div>
      )}

      {!envReady && envState.checkState !== "checking" && (
        <EnvironmentSetup
          state={envState}
          onSetup={setupEnvironments}
          onCancel={cancelSetup}
          onRetryCheck={retryCheck}
        />
      )}

      <section className="home__cards">
        <EntryCard
          title="Full Pipeline"
          description={envReady
            ? "Uncolored PLY + panoramic images. Runs all stages with pose verification."
            : "Environment setup required before running the pipeline."
          }
          icon="[F]"
          onClick={handleFullPipeline}
          disabled={!envReady}
        />
        <EntryCard
          title="Tour Only"
          description="Existing GLB mesh. Launches Unity virtual tour directly."
          icon="[T]"
          onClick={handleTourOnly}
        />
      </section>

      <RecentProjects projects={projects} />

      {showDialog && (
        <FileUploadDialog
          onSubmit={handleFileSubmit}
          onCancel={() => setShowDialog(false)}
        />
      )}
    </div>
  );
}
