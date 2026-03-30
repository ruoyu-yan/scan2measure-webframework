import Canvas2D from "./Canvas2D";
import ThreeViewer from "./ThreeViewer";
import ProgressPanel from "./ProgressPanel";
import ConfirmationGate from "./ConfirmationGate";
import ErrorPanel from "./ErrorPanel";
import type { StageProgress, StageStatus } from "../types/pipeline";

interface StageCanvasProps {
  viewType: "2d" | "3d" | "progress" | "confirmation";
  stageStatus: StageStatus;
  stageName: string;
  stageDescription: string;
  elapsedMs: number;
  progress: StageProgress | null;
  logLines: string[];
  stderrTail: string;
  /** Absolute paths to output artifacts for the current stage */
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
  onConfirm?: () => void;
  onCorrect?: (correctedAlignment: unknown) => void;
  onRetry?: () => void;
  onBack?: () => void;
}

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const remainder = s % 60;
  return `${m}:${String(remainder).padStart(2, "0")}`;
}

export default function StageCanvas(props: StageCanvasProps) {
  const { viewType, stageStatus, stageName, stageDescription, elapsedMs } = props;

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
        {stageStatus === "error" ? (
          <ErrorPanel
            stderr={props.stderrTail}
            onRetry={props.onRetry}
            onBack={props.onBack}
          />
        ) : viewType === "2d" ? (
          <Canvas2D
            imagePaths={props.artifacts.imagePaths}
            densityImagePath={props.artifacts.densityImagePath}
          />
        ) : viewType === "3d" ? (
          <ThreeViewer
            objPath={props.artifacts.objPath}
            plyPath={props.artifacts.plyPath}
            glbPath={props.artifacts.glbPath}
          />
        ) : viewType === "progress" ? (
          <ProgressPanel
            progress={props.progress}
            logLines={props.logLines}
          />
        ) : viewType === "confirmation" ? (
          <ConfirmationGate
            densityImagePath={props.artifacts.densityImagePath}
            cameraPositions={props.artifacts.cameraPositions}
            alignmentJsonPath={props.artifacts.alignmentJsonPath}
            onConfirm={props.onConfirm}
            onCorrect={props.onCorrect}
          />
        ) : null}
      </div>
    </main>
  );
}
