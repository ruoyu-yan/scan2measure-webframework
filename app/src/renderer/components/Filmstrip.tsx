interface FilmstripProps {
  stages: Array<{ id: string; name: string; status: string }>;
  completedArtifacts: Map<string, string>;
  onSelect: (stageIndex: number) => void;
  selectedIndex: number | null;
}

export default function Filmstrip({
  stages,
  completedArtifacts,
  onSelect,
  selectedIndex,
}: FilmstripProps) {
  // Only show stages that have completed and have a thumbnail
  const visibleStages = stages
    .map((s, i) => ({ ...s, index: i }))
    .filter((s) => s.status === "complete" && completedArtifacts.has(s.id));

  if (visibleStages.length === 0) return null;

  return (
    <div
      style={{
        display: "flex",
        gap: 8,
        padding: "8px 16px",
        background: "var(--bg-primary)",
        borderBottom: "1px solid var(--bg-card)",
        overflowX: "auto",
        overflowY: "hidden",
        flexShrink: 0,
      }}
    >
      {visibleStages.map((stage) => {
        const thumbUri = completedArtifacts.get(stage.id);
        const isSelected = selectedIndex === stage.index;
        return (
          <div
            key={stage.id}
            onClick={() => onSelect(stage.index)}
            style={{
              flexShrink: 0,
              width: 100,
              cursor: "pointer",
              borderRadius: 6,
              border: isSelected
                ? "2px solid var(--accent)"
                : "2px solid transparent",
              background: "var(--bg-card)",
              overflow: "hidden",
              transition: "border-color 0.15s",
            }}
          >
            <div
              style={{
                width: "100%",
                height: 64,
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                background: "var(--bg-secondary)",
              }}
            >
              {thumbUri ? (
                <img
                  src={thumbUri}
                  alt={stage.name}
                  style={{
                    maxWidth: "100%",
                    maxHeight: "100%",
                    objectFit: "contain",
                  }}
                />
              ) : (
                <span
                  style={{
                    color: "var(--text-secondary)",
                    fontSize: "0.65rem",
                  }}
                >
                  No preview
                </span>
              )}
            </div>
            <div
              style={{
                padding: "3px 6px",
                fontSize: "0.7rem",
                color: isSelected
                  ? "var(--text-primary)"
                  : "var(--text-secondary)",
                textAlign: "center",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {stage.name}
            </div>
          </div>
        );
      })}
    </div>
  );
}
