import { useRef, useEffect } from "react";
import type { StageProgress } from "../types/pipeline";

interface ProgressPanelProps {
  progress: StageProgress | null;
  logLines: string[];
}

export default function ProgressPanel({ progress, logLines }: ProgressPanelProps) {
  const logRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log to bottom
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  const pct = progress && progress.total > 0
    ? Math.round((progress.current / progress.total) * 100)
    : 0;

  return (
    <div className="progress-panel">
      {progress && (
        <div className="progress-panel__bar-container">
          <div className="progress-panel__bar">
            <div className="progress-panel__fill" style={{ width: `${pct}%` }} />
          </div>
          <span className="progress-panel__pct">{pct}%</span>
        </div>
      )}

      {progress && (
        <p className="progress-panel__message">
          {progress.message} ({progress.current}/{progress.total})
        </p>
      )}

      <div className="progress-panel__log" ref={logRef}>
        {logLines.map((line, i) => (
          <div key={i} className="progress-panel__log-line">{line}</div>
        ))}
      </div>
    </div>
  );
}
