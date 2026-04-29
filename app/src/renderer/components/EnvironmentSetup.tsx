import { useRef, useEffect } from "react";
import type { EnvironmentState } from "../hooks/useEnvironment";

interface EnvironmentSetupProps {
  state: EnvironmentState;
  onSetup: () => void;
  onCancel: () => void;
  onRetryCheck: () => void;
}

function formatPhase(phase: string): string {
  switch (phase) {
    case "creating":
      return "Creating conda environment...";
    case "installing-pytorch":
      return "Installing PyTorch (CUDA)...";
    case "installing-sam3":
      return "Installing SAM3 (editable)...";
    case "done":
      return "Complete";
    case "error":
      return "Failed";
    default:
      return phase;
  }
}

export default function EnvironmentSetup({ state, onSetup, onCancel, onRetryCheck }: EnvironmentSetupProps) {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [state.setupLogLines]);

  // No conda installed
  if (state.checkState === "no-conda") {
    return (
      <div className="env-setup">
        <h2 className="env-setup__title">Conda Not Found</h2>
        <p className="env-setup__desc">
          scan2measure requires conda (Miniconda, Anaconda, or Miniforge) installed
          at ~/miniconda3, ~/anaconda3, or ~/miniforge3.
        </p>
        <p className="env-setup__desc">
          Install Miniconda, then restart the app.
        </p>
        <button className="btn btn--primary" onClick={onRetryCheck}>
          Retry
        </button>
      </div>
    );
  }

  // Check error
  if (state.checkState === "error") {
    return (
      <div className="env-setup">
        <h2 className="env-setup__title">Environment Check Failed</h2>
        <p className="env-setup__desc">{state.setupError}</p>
        <button className="btn btn--primary" onClick={onRetryCheck}>
          Retry
        </button>
      </div>
    );
  }

  // Setup in progress
  if (state.setupActive) {
    return (
      <div className="env-setup env-setup--active">
        <h2 className="env-setup__title">Setting Up Environments</h2>
        <div className="env-setup__status-row">
          <span className="env-setup__env-name">{state.currentEnv}</span>
          <span className="env-setup__phase">{formatPhase(state.setupPhase)}</span>
        </div>
        {state.setupPhase === "error" && (
          <div className="env-setup__error">{state.setupError}</div>
        )}
        <div className="env-setup__log" ref={logRef}>
          {state.setupLogLines.map((line, i) => (
            <div key={i} className="env-setup__log-line">
              {line}
            </div>
          ))}
        </div>
        <div className="env-setup__actions">
          {state.setupPhase === "error" ? (
            <button className="btn btn--primary" onClick={onSetup}>
              Retry
            </button>
          ) : (
            <button className="btn btn--secondary" onClick={onCancel}>
              Cancel
            </button>
          )}
        </div>
      </div>
    );
  }

  // Missing environments (pre-setup)
  return (
    <div className="env-setup">
      <h2 className="env-setup__title">Environment Setup Required</h2>
      <p className="env-setup__desc">
        The following conda environments need to be created before running the pipeline:
      </p>
      <ul className="env-setup__list">
        {!state.scanEnvExists && (
          <li className="env-setup__list-item env-setup__list-item--missing">
            scan_env &mdash; Python 3.8 + PyTorch CUDA 11.6
          </li>
        )}
        {!state.sam3Exists && (
          <li className="env-setup__list-item env-setup__list-item--missing">
            sam3 &mdash; Python 3.12 + PyTorch CUDA 12.6 + SAM3
          </li>
        )}
        {state.scanEnvExists && (
          <li className="env-setup__list-item env-setup__list-item--ok">
            scan_env &mdash; ready
          </li>
        )}
        {state.sam3Exists && (
          <li className="env-setup__list-item env-setup__list-item--ok">
            sam3 &mdash; ready
          </li>
        )}
      </ul>
      <p className="env-setup__desc env-setup__desc--warn">
        This may take 10-20 minutes depending on network speed.
      </p>
      <button className="btn btn--primary" onClick={onSetup}>
        Install Environments
      </button>
    </div>
  );
}
