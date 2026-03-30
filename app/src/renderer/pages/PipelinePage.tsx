import { useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import StageSidebar from "../components/StageSidebar";
import StageCanvas from "../components/StageCanvas";
import QualityTierSelect from "../components/QualityTierSelect";
import { useProject } from "../hooks/useProject";
import { usePipeline } from "../hooks/usePipeline";
import { FULL_PIPELINE_STAGES, MESH_ONLY_STAGES } from "../../shared/constants";
import type { StageStatus } from "../types/pipeline";
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

  const currentStageConfig = stages[state.currentStage];
  const currentStatus: StageStatus = state.stages[state.currentStage] ?? "pending";

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

  // Auto-run the current stage when it transitions to pending (and is not confirmation)
  useEffect(() => {
    if (!project) return;
    if (
      currentStatus === "pending" &&
      currentStageConfig &&
      currentStageConfig.viewType !== "confirmation" &&
      currentStageConfig.scriptPaths.length > 0
    ) {
      runCurrentStage();
    }
  }, [currentStatus, currentStageConfig, runCurrentStage, project]);

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
  const isMeshingStage = currentStageConfig?.id === "meshing";
  const isStageActive = currentStatus === "active";
  const showQualitySelect = isMeshingStage && !isStageActive;

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
              disabled={isStageActive}
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
