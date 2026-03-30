import { useEffect, useState } from "react";
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
    window.electronAPI.getProjects().then((list) => setProjects(list as Project[]));
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
