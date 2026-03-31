interface FilmstripProps {
  stages: Array<{ id: string; name: string; status: string }>;
  completedArtifacts: Map<string, string>;
  onSelect: (stageIndex: number) => void;
  selectedIndex: number | null;
}

/** Inline SVG icon for 3D stages (wireframe cube). */
function ThreeDIcon() {
  return (
    <svg
      width="32"
      height="32"
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Cube wireframe */}
      <path
        d="M16 4 L28 10 L28 22 L16 28 L4 22 L4 10 Z"
        stroke="#60a5fa"
        strokeWidth="1.5"
        fill="none"
        opacity="0.7"
      />
      <path
        d="M16 4 L16 16 M28 10 L16 16 M4 10 L16 16"
        stroke="#60a5fa"
        strokeWidth="1.5"
        opacity="0.5"
      />
      <path
        d="M16 16 L16 28"
        stroke="#60a5fa"
        strokeWidth="1.5"
        opacity="0.5"
      />
    </svg>
  );
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
        const is3D = thumbUri === "__3d__";
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
                background: is3D ? "#0a0a1a" : "var(--bg-secondary)",
              }}
            >
              {is3D ? (
                <ThreeDIcon />
              ) : thumbUri ? (
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
