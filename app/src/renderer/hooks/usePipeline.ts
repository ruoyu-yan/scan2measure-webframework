import { useState, useEffect, useCallback, useRef } from "react";
import type { StageStatus, PipelineState, QualityTier } from "../types/pipeline";
import type { StageConfig } from "../../shared/constants";

interface UsePipelineOptions {
  stages: StageConfig[];
  projectId: string;
  projectDir?: string;
  startStage?: number;
}

export function usePipeline({ stages, projectId: _projectId, projectDir = "", startStage = 0 }: UsePipelineOptions) {
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

  // Listen for IPC events from main process
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

    api.onStageComplete((stageIndex: number) => {
      setState((prev) => {
        const updated = [...prev.stages];
        updated[stageIndex] = "complete";
        const next = stageIndex + 1;
        if (next < stages.length) {
          updated[next] = stages[next].viewType === "confirmation" ? "confirmation" : "pending";
        }
        return {
          ...prev,
          stages: updated,
          currentStage: next < stages.length ? next : stageIndex,
          progress: null,
          logLines: [],
          elapsedMs: 0,
        };
      });
    });

    api.onStageError((data: { stageIndex: number; stderr: string }) => {
      setState((prev) => {
        const updated = [...prev.stages];
        updated[data.stageIndex] = "error";
        return {
          ...prev,
          stages: updated,
          stderrTail: data.stderr,
        };
      });
    });

    return () => {
      api.removeAllListeners("pipeline:progress");
      api.removeAllListeners("pipeline:log");
      api.removeAllListeners("pipeline:stage-complete");
      api.removeAllListeners("pipeline:stage-error");
    };
  }, [stages]);

  const runCurrentStage = useCallback(async () => {
    const stage = stages[state.currentStage];
    if (!stage || stage.scriptPaths.length === 0) return;

    setState((prev) => {
      const updated = [...prev.stages];
      updated[state.currentStage] = "active";
      return { ...prev, stages: updated, progress: null, logLines: [], stderrTail: "" };
    });

    // Run each script in the stage sequentially
    // Write a per-stage config JSON so Python scripts receive dynamic parameters
    const configJson: Record<string, unknown> = {
      project_dir: projectDir,
      quality_tier: state.qualityTier,
    };
    const configJsonPath = projectDir
      ? `${projectDir}/${stage.id}_config.json`
      : "";
    if (configJsonPath) {
      await window.electronAPI.runStage(
        "__write_config__", "scan_env",
        JSON.stringify({ path: configJsonPath, data: configJson })
      ).catch(() => { /* config write is best-effort */ });
    }
    for (const scriptPath of stage.scriptPaths) {
      const result = await window.electronAPI.runStage(scriptPath, stage.condaEnv, configJsonPath);
      const typed = result as { ok?: boolean; error?: boolean; stderr?: string };
      if (typed.error) {
        setState((prev) => {
          const updated = [...prev.stages];
          updated[state.currentStage] = "error";
          return { ...prev, stages: updated, stderrTail: typed.stderr || "" };
        });
        return;
      }
    }

    // All scripts in stage succeeded
    setState((prev) => {
      const updated = [...prev.stages];
      updated[state.currentStage] = "complete";
      const next = state.currentStage + 1;
      return {
        ...prev,
        stages: updated,
        currentStage: next < stages.length ? next : state.currentStage,
        progress: null,
        logLines: [],
        elapsedMs: 0,
      };
    });
  }, [state.currentStage, stages]);

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
