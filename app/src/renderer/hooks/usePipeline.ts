import { useState, useEffect, useCallback, useRef } from "react";
import type { StageStatus, PipelineState, QualityTier } from "../types/pipeline";
import type { StageConfig } from "../../shared/constants";

interface UsePipelineOptions {
  stages: StageConfig[];
  projectId: string;
  startStage?: number;
}

export function usePipeline({ stages, projectId, startStage = 0 }: UsePipelineOptions) {
  const [state, setState] = useState<PipelineState>(() => ({
    stages: stages.map((_, i) =>
      i < startStage ? "complete" : "pending"
    ) as StageStatus[],
    currentStage: startStage,
    progress: null,
    logLines: [],
    stderrTail: "",
    elapsedMs: 0,
    qualityTier: "balanced" as QualityTier,
  }));

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(Date.now());
  const isRunningRef = useRef(false);

  // Start elapsed timer when a stage is active
  useEffect(() => {
    const isActive = state.stages[state.currentStage] === "active";
    if (isActive) {
      startTimeRef.current = Date.now();
      timerRef.current = setInterval(() => {
        setState((prev) => ({
          ...prev,
          elapsedMs: Date.now() - startTimeRef.current,
        }));
      }, 1000);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [state.currentStage, state.stages]);

  // Listen for IPC events from main process (progress + log only)
  useEffect(() => {
    const api = window.electronAPI;

    api.onProgress((data: { current: number; total: number; message: string }) => {
      setState((prev) => ({ ...prev, progress: data }));
    });

    api.onLog((line: string) => {
      setState((prev) => ({
        ...prev,
        logLines: [...prev.logLines.slice(-500), line],
      }));
    });

    return () => {
      api.removeAllListeners("pipeline:progress");
      api.removeAllListeners("pipeline:log");
    };
  }, []);

  const runCurrentStage = useCallback(async () => {
    // Mutex: prevent concurrent invocations
    if (isRunningRef.current) return;
    isRunningRef.current = true;

    try {
      const stageIndex = state.currentStage;
      const stage = stages[stageIndex];
      if (!stage || stage.scriptPaths.length === 0) return;

      setState((prev) => {
        const updated = [...prev.stages];
        updated[prev.currentStage] = "active";
        return { ...prev, stages: updated, progress: null, logLines: [], stderrTail: "" };
      });

      // For perPano stages, run each script once per panorama
      if (stage.perPano) {
        // Get pano list from project record
        const projectData = await window.electronAPI.getProjects() as Array<{
          id: string;
          inputs: { panoramas?: string[] };
        }>;
        const project = projectData.find((p) => p.id === projectId);
        const panoPaths = project?.inputs?.panoramas || [];

        for (let pi = 0; pi < panoPaths.length; pi++) {
          const panoPath = panoPaths[pi];
          const panoName = panoPath.replace(/^.*[\\/]/, "").replace(/\.[^.]+$/, "");

          // Write per-pano config
          const panoConfigResult = await window.electronAPI.writeConfig(
            projectId,
            `${stage.id}_${panoName}`,
            { quality_tier: state.qualityTier, pano_name: panoName, input_path: panoPath }
          ) as { ok?: boolean; configPath?: string; error?: string };

          if (panoConfigResult.error || !panoConfigResult.configPath) {
            setState((prev) => {
              const updated = [...prev.stages];
              updated[prev.currentStage] = "error";
              return { ...prev, stages: updated, stderrTail: panoConfigResult.error || "Failed to write pano config" };
            });
            return;
          }

          for (const scriptPath of stage.scriptPaths) {
            const result = await window.electronAPI.runStage(scriptPath, stage.condaEnv, panoConfigResult.configPath);
            const typed = result as { ok?: boolean; error?: boolean; stderr?: string };
            if (typed.error) {
              setState((prev) => {
                const updated = [...prev.stages];
                updated[prev.currentStage] = "error";
                return { ...prev, stages: updated, stderrTail: typed.stderr || "" };
              });
              return;
            }
          }
        }
      } else {
        // Non-perPano: single invocation
        const configResult = await window.electronAPI.writeConfig(
          projectId, stage.id, { quality_tier: state.qualityTier }
        ) as { ok?: boolean; configPath?: string; error?: string };

        if (configResult.error || !configResult.configPath) {
          setState((prev) => {
            const updated = [...prev.stages];
            updated[prev.currentStage] = "error";
            return { ...prev, stages: updated, stderrTail: configResult.error || "Failed to write config" };
          });
          return;
        }

        for (const scriptPath of stage.scriptPaths) {
          const result = await window.electronAPI.runStage(scriptPath, stage.condaEnv, configResult.configPath);
          const typed = result as { ok?: boolean; error?: boolean; stderr?: string };
          if (typed.error) {
            setState((prev) => {
              const updated = [...prev.stages];
              updated[prev.currentStage] = "error";
              return { ...prev, stages: updated, stderrTail: typed.stderr || "" };
            });
            return;
          }
        }
      }

      // All scripts succeeded — advance to next stage
      setState((prev) => {
        const updated = [...prev.stages];
        updated[prev.currentStage] = "complete";
        const next = prev.currentStage + 1;
        return {
          ...prev,
          stages: updated,
          currentStage: next < stages.length ? next : prev.currentStage,
          progress: null,
          logLines: [],
          elapsedMs: 0,
        };
      });
    } finally {
      isRunningRef.current = false;
    }
  }, [state.currentStage, stages, projectId, state.qualityTier]);

  const retry = useCallback(() => {
    setState((prev) => {
      const updated = [...prev.stages];
      updated[prev.currentStage] = "pending";
      return { ...prev, stages: updated, stderrTail: "" };
    });
  }, []);

  const setQualityTier = useCallback((tier: QualityTier) => {
    setState((prev) => ({ ...prev, qualityTier: tier }));
  }, []);

  const confirm = useCallback(() => {
    setState((prev) => {
      const updated = [...prev.stages];
      updated[prev.currentStage] = "complete";
      const next = prev.currentStage + 1;
      return {
        ...prev,
        stages: updated,
        currentStage: next < stages.length ? next : prev.currentStage,
      };
    });
  }, [stages]);

  return {
    state,
    runCurrentStage,
    retry,
    confirm,
    setQualityTier,
  };
}
