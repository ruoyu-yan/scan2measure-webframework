interface ErrorPanelProps {
  stderr: string;
  onRetry?: () => void;
  onBack?: () => void;
}

export default function ErrorPanel({ stderr, onRetry, onBack }: ErrorPanelProps) {
  return (
    <div className="error-panel">
      <h3 className="error-panel__title">Stage Failed</h3>
      <p className="error-panel__subtitle">Last 30 lines of stderr:</p>
      <pre className="error-panel__log">{stderr || "(no stderr output)"}</pre>
      <div className="error-panel__actions">
        {onRetry && (
          <button className="btn btn--primary" onClick={onRetry}>
            Retry
          </button>
        )}
        {onBack && (
          <button className="btn btn--secondary" onClick={onBack}>
            Back to Home
          </button>
        )}
      </div>
    </div>
  );
}
