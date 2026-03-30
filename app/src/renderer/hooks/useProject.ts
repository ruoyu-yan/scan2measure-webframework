import { useState, useEffect, useCallback } from "react";
import type { Project } from "../types/project";

export function useProject(projectId: string | undefined) {
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!projectId) {
      setLoading(false);
      return;
    }
    window.electronAPI
      .getProjects()
      .then((projects) => {
        const typed = projects as Project[];
        const found = typed.find((p) => p.id === projectId) || null;
        setProject(found);
        setLoading(false);
      });
  }, [projectId]);

  const updateProject = useCallback(
    async (patch: Partial<Project>) => {
      if (!projectId) return;
      const updated = await window.electronAPI.updateProject(projectId, patch);
      setProject(updated as Project);
    },
    [projectId]
  );

  return { project, loading, updateProject };
}
