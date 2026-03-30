import type { StageStatus } from "../types/pipeline";

interface StageSidebarProps {
  stages: Array<{ name: string; status: StageStatus }>;
  currentStage: number;
  onBack: () => void;
}

const STATUS_ICONS: Record<StageStatus, string> = {
  complete: "\u2713",       // checkmark
  active: "\u25CF",         // filled circle
  pending: "\u25CB",        // empty circle
  error: "!",               // exclamation
  confirmation: "?",        // question mark
};

export default function StageSidebar({ stages, currentStage, onBack }: StageSidebarProps) {
  return (
    <aside className="stage-sidebar">
      <button className="stage-sidebar__back" onClick={onBack}>
        &larr; Home
      </button>
      <ul className="stage-sidebar__list">
        {stages.map((stage, i) => (
          <li
            key={i}
            className={`stage-sidebar__item stage-sidebar__item--${stage.status} ${
              i === currentStage ? "stage-sidebar__item--current" : ""
            }`}
          >
            <span className="stage-sidebar__icon">{STATUS_ICONS[stage.status]}</span>
            <span className="stage-sidebar__name">{stage.name}</span>
          </li>
        ))}
      </ul>
    </aside>
  );
}
