import { useNavigate } from "react-router-dom";
import type { Project } from "../types/project";

interface RecentProjectsProps {
  projects: Project[];
}

const STATUS_LABELS: Record<string, string> = {
  pending: "Pending",
  in_progress: "In Progress",
  completed: "Completed",
  error: "Error",
};

export default function RecentProjects({ projects }: RecentProjectsProps) {
  const navigate = useNavigate();

  if (projects.length === 0) {
    return (
      <div className="recent-projects recent-projects--empty">
        <p>No recent projects. Start by selecting an entry point above.</p>
      </div>
    );
  }

  return (
    <div className="recent-projects">
      <h2 className="recent-projects__title">Recent Projects</h2>
      <ul className="recent-projects__list">
        {projects.slice(0, 10).map((project) => (
          <li
            key={project.id}
            className="recent-projects__item"
            onClick={() => navigate(`/pipeline/${project.id}`)}
          >
            <span className="recent-projects__name">{project.name}</span>
            <span className={`recent-projects__status recent-projects__status--${project.status}`}>
              {STATUS_LABELS[project.status] || project.status}
            </span>
            <span className="recent-projects__date">
              {new Date(project.created).toLocaleDateString()}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
