import Canvas2D from "./Canvas2D";
import ThreeViewer from "./ThreeViewer";
import ConfirmationGate from "./ConfirmationGate";
import ErrorPanel from "./ErrorPanel";
import ImageViewer from "./ImageViewer";
import ImageGallery from "./ImageGallery";
import ConfirmMatching from "./ConfirmMatching";
import ObjViewer from "./ObjViewer";
import PolygonViewer from "./PolygonViewer";
import PlyViewer from "./PlyViewer";
import ProgressPanel from "./ProgressPanel";
import type { StageProgress, StageStatus } from "../types/pipeline";

/** Resolved artifact data for the current stage (from artifacts:resolve IPC). */
export interface ResolvedArtifacts {
  images?: string[];
  objPath?: string;
  plyPath?: string;
  glbPath?: string;
  alignmentJson?: string;
  densityImage?: string;
  polygonsJson?: string;
  panoThumbnails?: Record<string, string>;
}

interface StageCanvasProps {
  viewType: "2d" | "3d" | "progress" | "confirmation";
  stageId: string;
  stageStatus: StageStatus;
  stageName: string;
  stageDescription: string;
  elapsedMs: number;
  progress: StageProgress | null;
  logLines: string[];
  stderrTail: string;
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
  resolvedArtifacts?: ResolvedArtifacts;
  /** Previous completed stage's id + artifacts, shown while current stage runs */
  prevStageId?: string;
  prevResolvedArtifacts?: ResolvedArtifacts;
  /** Quality tier for meshing (shown on confirm_colorization gate) */
  qualityTier?: string;
  onQualityTierChange?: (tier: "preview" | "balanced" | "high") => void;
  onConfirm?: () => void;
  onCorrect?: (correctedAlignment: unknown) => void;
  onRetry?: () => void;
  onBack?: () => void;
  onLaunchTour?: (glbPath: string) => void;
}

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const remainder = s % 60;
  return `${m}:${String(remainder).padStart(2, "0")}`;
}

/** Stages that show a single image or first image from a list. */
const SINGLE_IMAGE_STAGES = new Set([
  "density_image",
  "sam3_segmentation",
  "polygon_matching",
]);

/** Stages that show per-pano image galleries. */
const GALLERY_STAGES = new Set([
  "pano_footprints",
  "line_detection_2d",
]);

/** Render visualization for a given stageId + resolvedArtifacts (reused for both current and previous stage). */
function renderStageVisualization(stageId: string, ra: ResolvedArtifacts): JSX.Element | null {
  // Single-image stages
  if (SINGLE_IMAGE_STAGES.has(stageId) && ra.images && ra.images.length > 0) {
    return <ImageViewer imagePath={ra.images[0]} />;
  }

  // Gallery stages
  if (GALLERY_STAGES.has(stageId) && ra.images && ra.images.length > 0) {
    const galleryItems = ra.images.map((p) => ({
      path: p,
      label: extractLabelFromPath(p),
    }));
    return <ImageGallery images={galleryItems} />;
  }

  // Polygon viewer
  if (stageId === "sam3_polygons" && ra.densityImage && ra.polygonsJson) {
    return <PolygonViewer densityImagePath={ra.densityImage} polygonsJsonPath={ra.polygonsJson} />;
  }

  // 3D line detection: OBJ viewer
  if (stageId === "line_detection_3d" && ra.objPath) {
    return <ObjViewer objPath={ra.objPath} />;
  }

  // Colorization: colored point cloud PLY viewer
  // (confirm_colorization is handled separately with quality tier + confirm overlay)
  if (stageId === "colorization" && ra.plyPath) {
    return <PlyViewer plyPath={ra.plyPath} />;
  }

  // Pose estimation: topdown images
  if (stageId === "pose_estimation" && ra.images && ra.images.length > 0) {
    if (ra.images.length === 1) {
      return <ImageViewer imagePath={ra.images[0]} />;
    }
    const galleryItems = ra.images.map((p) => ({
      path: p,
      label: extractLabelFromPath(p),
    }));
    return <ImageGallery images={galleryItems} />;
  }

  return null;
}

function renderContent(props: StageCanvasProps) {
  const {
    stageId,
    stageStatus,
    viewType,
    resolvedArtifacts,
    prevStageId,
    prevResolvedArtifacts,
  } = props;

  // Error panel always takes priority
  if (stageStatus === "error") {
    return (
      <ErrorPanel
        stderr={props.stderrTail}
        onRetry={props.onRetry}
        onBack={props.onBack}
      />
    );
  }

  const ra = resolvedArtifacts;

  // -- Completed stage: show its own visualization --
  if (ra) {
    const viz = renderStageVisualization(stageId, ra);
    if (viz) return viz;
  }

  // Confirm matching stage (special: needs onConfirm/onCorrect callbacks)
  if (stageId === "confirm_matching") {
    if (ra?.densityImage && ra?.alignmentJson) {
      return (
        <ConfirmMatching
          densityImagePath={ra.densityImage}
          alignmentJsonPath={ra.alignmentJson}
          polygonsJsonPath={ra.polygonsJson}
          panoThumbnails={ra.panoThumbnails || {}}
          onConfirm={props.onConfirm || (() => {})}
          onCorrect={props.onCorrect}
        />
      );
    }
    return (
      <ConfirmationGate
        densityImagePath={props.artifacts.densityImagePath}
        cameraPositions={props.artifacts.cameraPositions}
        alignmentJsonPath={props.artifacts.alignmentJsonPath}
        onConfirm={props.onConfirm}
        onCorrect={props.onCorrect}
      />
    );
  }

  // Confirm colorization gate: PLY viewer + quality tier + confirm button
  if (stageId === "confirm_colorization") {
    return (
      <div style={{ position: "relative", width: "100%", height: "100%" }}>
        {ra?.plyPath ? (
          <PlyViewer plyPath={ra.plyPath} />
        ) : (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#8888aa" }}>
            Loading colored point cloud...
          </div>
        )}
        <div
          style={{
            position: "absolute",
            bottom: 24,
            left: "50%",
            transform: "translateX(-50%)",
            background: "rgba(30, 41, 59, 0.95)",
            borderRadius: 12,
            padding: "16px 28px",
            display: "flex",
            alignItems: "center",
            gap: 20,
            boxShadow: "0 4px 24px rgba(0,0,0,0.5)",
            zIndex: 20,
          }}
        >
          <div style={{ color: "#94a3b8", fontSize: 13 }}>
            Inspect the colored point cloud, then select mesh quality:
          </div>
          <select
            value={props.qualityTier || "balanced"}
            onChange={(e) => props.onQualityTierChange?.(e.target.value as "preview" | "balanced" | "high")}
            style={{
              padding: "6px 12px",
              borderRadius: 6,
              border: "1px solid #475569",
              background: "#1e293b",
              color: "#e2e8f0",
              fontSize: 13,
            }}
          >
            <option value="preview">Preview (~2-3 min)</option>
            <option value="balanced">Balanced (~5-8 min)</option>
            <option value="high">High (~15-20 min)</option>
          </select>
          <button
            className="btn btn--primary"
            onClick={props.onConfirm}
            style={{ padding: "8px 24px" }}
          >
            Proceed to Meshing
          </button>
        </div>
      </div>
    );
  }

  // -- Active stage --
  if (stageStatus === "active") {
    // Progress-type stages (meshing, colorization): show centered progress panel
    if (viewType === "progress") {
      return (
        <div style={{ position: "relative", width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div style={{ width: "100%", maxWidth: 640, padding: "0 24px" }}>
            <div style={{ textAlign: "center", marginBottom: 24 }}>
              <div className="running-spinner" style={{ width: 48, height: 48, margin: "0 auto 16px", borderWidth: 4 }} />
              <div style={{ color: "#e2e8f0", fontSize: 18, fontWeight: 600 }}>{props.stageName}</div>
              <div style={{ color: "#60a5fa", fontSize: 24, fontWeight: 600, marginTop: 4 }}>
                {formatElapsed(props.elapsedMs)}
              </div>
            </div>
            <ProgressPanel progress={props.progress} logLines={props.logLines} />
          </div>
        </div>
      );
    }
    // Other viewTypes: show previous stage's visualization with running overlay
    if (prevStageId && prevResolvedArtifacts) {
      const prevViz = renderStageVisualization(prevStageId, prevResolvedArtifacts);
      if (prevViz) {
        return (
          <div style={{ position: "relative", width: "100%", height: "100%" }}>
            <div style={{ width: "100%", height: "100%", opacity: 0.6 }}>
              {prevViz}
            </div>
            <RunningOverlay
              elapsedMs={props.elapsedMs}
              stageName={props.stageName}
            />
          </div>
        );
      }
    }
    // No previous artifacts — show a simple spinner
    return (
      <div style={{ position: "relative", width: "100%", height: "100%" }}>
        <RunningOverlay elapsedMs={props.elapsedMs} stageName={props.stageName} />
      </div>
    );
  }

  // Done stage: 3D GLB viewer + Launch Virtual Tour button
  if (stageId === "done") {
    const glbPath = ra?.glbPath || props.artifacts.glbPath;
    return (
      <div style={{ position: "relative", width: "100%", height: "100%" }}>
        {glbPath ? (
          <ThreeViewer glbPath={glbPath} />
        ) : (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#8888aa" }}>
            No mesh found. Run the meshing stage first.
          </div>
        )}
        {glbPath && (
          <div
            style={{
              position: "absolute",
              bottom: 24,
              left: "50%",
              transform: "translateX(-50%)",
              background: "rgba(30, 41, 59, 0.95)",
              borderRadius: 12,
              padding: "16px 28px",
              display: "flex",
              alignItems: "center",
              gap: 20,
              boxShadow: "0 4px 24px rgba(0,0,0,0.5)",
              zIndex: 20,
            }}
          >
            <div style={{ color: "#94a3b8", fontSize: 13 }}>
              Pipeline complete — mesh is ready.
            </div>
            <button
              className="btn btn--primary"
              onClick={() => props.onLaunchTour?.(glbPath)}
              style={{
                padding: "10px 28px",
                fontSize: 14,
                fontWeight: 600,
                background: "#e94560",
                border: "none",
                borderRadius: 8,
                color: "#fff",
                cursor: "pointer",
              }}
            >
              Launch Virtual Tour
            </button>
          </div>
        )}
      </div>
    );
  }

  // -- Fallback to viewType-based routing --

  if (viewType === "2d") {
    return (
      <Canvas2D
        imagePaths={props.artifacts.imagePaths}
        densityImagePath={props.artifacts.densityImagePath}
      />
    );
  }

  if (viewType === "3d") {
    return (
      <ThreeViewer
        objPath={props.artifacts.objPath}
        plyPath={props.artifacts.plyPath}
        glbPath={props.artifacts.glbPath}
      />
    );
  }

  if (viewType === "confirmation") {
    return (
      <ConfirmationGate
        densityImagePath={props.artifacts.densityImagePath}
        cameraPositions={props.artifacts.cameraPositions}
        alignmentJsonPath={props.artifacts.alignmentJsonPath}
        onConfirm={props.onConfirm}
        onCorrect={props.onCorrect}
      />
    );
  }

  return null;
}

/** Floating badge shown over the previous stage's visualization while a stage runs. */
function RunningOverlay({ elapsedMs, stageName }: { elapsedMs: number; stageName: string }) {
  return (
    <div
      style={{
        position: "absolute",
        top: 16,
        right: 16,
        background: "rgba(30, 41, 59, 0.92)",
        borderRadius: 10,
        padding: "12px 20px",
        display: "flex",
        alignItems: "center",
        gap: 12,
        boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
        zIndex: 10,
      }}
    >
      <div className="running-spinner" />
      <div>
        <div style={{ color: "#94a3b8", fontSize: 12, marginBottom: 2 }}>Running</div>
        <div style={{ color: "#e2e8f0", fontSize: 14, fontWeight: 600 }}>{stageName}</div>
      </div>
      <div style={{ color: "#60a5fa", fontSize: 16, fontWeight: 600, marginLeft: 8 }}>
        {formatElapsed(elapsedMs)}
      </div>
    </div>
  );
}

/** Extract a human-readable label from a file path. */
function extractLabelFromPath(filePath: string): string {
  const parts = filePath.split(/[/\\]/);
  const fileName = parts[parts.length - 1];
  if (fileName === "debug.png" || fileName === "grouped_lines.png" || fileName === "topdown.png") {
    for (let i = parts.length - 2; i >= 0; i--) {
      const dir = parts[i];
      if (dir === "vis" || dir === "2d_feature_extracted" || dir === "sam3_pano_processing") continue;
      return dir.replace(/_v2$/, "");
    }
  }
  return fileName;
}

export default function StageCanvas(props: StageCanvasProps) {
  const { stageName, stageDescription, elapsedMs } = props;

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
        {renderContent(props)}
      </div>
    </main>
  );
}
