import { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import StageSidebar from "../components/StageSidebar";
import StageCanvas from "../components/StageCanvas";
import type { ResolvedArtifacts } from "../components/StageCanvas";
import Filmstrip from "../components/Filmstrip";
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

  // -- Artifact resolution state --
  const [resolvedMap, setResolvedMap] = useState<Map<string, ResolvedArtifacts>>(new Map());
  const [filmstripThumbs, setFilmstripThumbs] = useState<Map<string, string>>(new Map());
  const [filmstripSelectedIndex, setFilmstripSelectedIndex] = useState<number | null>(null);

  // Track which stages we have already resolved to avoid redundant calls
  const resolvedStagesRef = useRef<Set<string>>(new Set());

  // Resolve artifacts for a given stage
  const resolveStage = useCallback(async (stageId: string) => {
    if (!projectId) return;
    try {
      const result = await window.electronAPI.resolveArtifacts(projectId, stageId) as {
        stageId: string;
        artifacts: ResolvedArtifacts;
      };
      if (result.artifacts && Object.keys(result.artifacts).length > 0) {
        setResolvedMap((prev) => {
          const next = new Map(prev);
          next.set(stageId, result.artifacts);
          return next;
        });

        // Load thumbnail for filmstrip (first image, or placeholder for 3D stages)
        const imgs = result.artifacts.images;
        if (imgs && imgs.length > 0) {
          try {
            const thumb = await window.electronAPI.readImage(imgs[0]);
            if (thumb) {
              setFilmstripThumbs((prev) => {
                const next = new Map(prev);
                next.set(stageId, thumb);
                return next;
              });
            }
          } catch {
            // thumbnail load failed, skip
          }
        } else if (result.artifacts.objPath || result.artifacts.plyPath || result.artifacts.glbPath) {
          // 3D stages (OBJ/PLY/GLB) have no raster thumbnail -- use a sentinel
          // value so the filmstrip knows the stage has artifacts
          setFilmstripThumbs((prev) => {
            const next = new Map(prev);
            next.set(stageId, "__3d__");
            return next;
          });
        }
      }
    } catch {
      // resolve failed, skip silently
    }
  }, [projectId]);

  // Resolve artifacts when a stage completes, or when a confirmation stage becomes current
  useEffect(() => {
    state.stages.forEach((status, i) => {
      if (status === "complete") {
        const sid = stages[i].id;
        if (!resolvedStagesRef.current.has(sid)) {
          resolvedStagesRef.current.add(sid);
          resolveStage(sid);
        }
      }
    });
    // Also resolve for the current stage if it has no scripts to run
    // (confirmation gates and the "done" stage)
    if (
      currentStageConfig &&
      currentStatus === "pending" &&
      (currentStageConfig.viewType === "confirmation" || currentStageConfig.scriptPaths.length === 0) &&
      !resolvedStagesRef.current.has(currentStageConfig.id)
    ) {
      resolvedStagesRef.current.add(currentStageConfig.id);
      resolveStage(currentStageConfig.id);
    }
  }, [state.stages, stages, resolveStage, currentStageConfig, currentStatus]);

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

  // Determine the stage to display: either filmstrip selection or current stage
  const displayStageIndex = filmstripSelectedIndex !== null
    ? filmstripSelectedIndex
    : state.currentStage;
  const displayStageConfig = stages[displayStageIndex];
  const displayStatus: StageStatus = filmstripSelectedIndex !== null
    ? (state.stages[filmstripSelectedIndex] ?? "pending")
    : currentStatus;

  // Determine artifacts for displayed stage based on project paths
  const artifacts = {
    densityImagePath: project.outputs.densityImage,
    glbPath: project.outputs.meshGlb,
  };

  // Provide resolved artifacts for completed stages, confirmation gates, and no-script stages (done)
  const displayResolvedArtifacts =
    displayStageConfig && (
      displayStatus === "complete" ||
      displayStageConfig.viewType === "confirmation" ||
      displayStageConfig.scriptPaths.length === 0
    )
      ? resolvedMap.get(displayStageConfig.id)
      : undefined;

  // When current stage is running, find the previous completed stage's artifacts
  // so StageCanvas can show them instead of an empty screen
  let prevStageId: string | undefined;
  let prevResolvedArtifacts: ResolvedArtifacts | undefined;
  if (displayStatus === "active" && displayStageIndex > 0) {
    for (let i = displayStageIndex - 1; i >= 0; i--) {
      const sid = stages[i].id;
      const ra = resolvedMap.get(sid);
      if (ra && Object.keys(ra).length > 0) {
        prevStageId = sid;
        prevResolvedArtifacts = ra;
        break;
      }
    }
  }

  // Build filmstrip stage data
  const filmstripStages = stages.map((s, i) => ({
    id: s.id,
    name: s.name,
    status: state.stages[i] as string,
  }));

  return (
    <div className="pipeline-page">
      <StageSidebar
        stages={sidebarStages}
        currentStage={state.currentStage}
        onBack={() => navigate("/")}
      />

      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>

        <Filmstrip
          stages={filmstripStages}
          completedArtifacts={filmstripThumbs}
          onSelect={(idx) => {
            // Toggle: clicking already-selected goes back to current stage
            setFilmstripSelectedIndex((prev) => (prev === idx ? null : idx));
          }}
          selectedIndex={filmstripSelectedIndex}
        />

        <StageCanvas
          viewType={displayStageConfig?.viewType || "progress"}
          stageId={displayStageConfig?.id || ""}
          stageStatus={displayStatus}
          stageName={displayStageConfig?.name || ""}
          stageDescription={displayStageConfig?.description || ""}
          elapsedMs={filmstripSelectedIndex !== null ? 0 : state.elapsedMs}
          progress={filmstripSelectedIndex !== null ? null : state.progress}
          logLines={filmstripSelectedIndex !== null ? [] : state.logLines}
          stderrTail={filmstripSelectedIndex !== null ? "" : state.stderrTail}
          artifacts={artifacts}
          resolvedArtifacts={displayResolvedArtifacts}
          prevStageId={prevStageId}
          prevResolvedArtifacts={prevResolvedArtifacts}
          qualityTier={state.qualityTier}
          onQualityTierChange={setQualityTier}
          onConfirm={confirm}
          onRetry={retry}
          onBack={() => navigate("/")}
          onLaunchTour={async (glbPath) => {
            // Find minimap PNG near GLB, then launch Unity
            let minimapPath: string | undefined;
            try {
              const result = await window.electronAPI.findMinimapPng(glbPath) as { found: boolean; path?: string };
              if (result.found && result.path) minimapPath = result.path;
            } catch { /* no minimap, that's ok */ }
            window.electronAPI.launchUnity(glbPath, minimapPath);
          }}
        />
      </div>
    </div>
  );
}
