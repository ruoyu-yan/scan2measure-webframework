import { useState, useEffect, useCallback } from "react";

export type EnvCheckState = "checking" | "ready" | "missing" | "no-conda" | "error";
export type SetupPhase = "idle" | "creating" | "installing-pytorch" | "installing-sam3" | "done" | "error";

export interface EnvironmentState {
  checkState: EnvCheckState;
  condaFound: boolean;
  scanEnvExists: boolean;
  sam3Exists: boolean;
  setupActive: boolean;
  currentEnv: string | null;
  setupPhase: SetupPhase;
  setupLogLines: string[];
  setupError: string | null;
}

interface CheckResult {
  ok?: boolean;
  error?: boolean;
  status?: {
    condaFound: boolean;
    condaPath: string | null;
    environments: Record<string, boolean>;
  };
  message?: string;
}

interface SetupResult {
  ok?: boolean;
  error?: boolean;
  message?: string;
}

export function useEnvironment() {
  const [state, setState] = useState<EnvironmentState>({
    checkState: "checking",
    condaFound: false,
    scanEnvExists: false,
    sam3Exists: false,
    setupActive: false,
    currentEnv: null,
    setupPhase: "idle",
    setupLogLines: [],
    setupError: null,
  });

  const checkEnvironment = useCallback(async () => {
    setState((prev) => ({ ...prev, checkState: "checking" }));
    try {
      const result = (await window.electronAPI.checkEnvironment()) as CheckResult;

      if (result.error || !result.status) {
        setState((prev) => ({
          ...prev,
          checkState: "error",
          setupError: result.message || "Failed to check environments",
        }));
        return;
      }

      const { condaFound, environments } = result.status;
      if (!condaFound) {
        setState((prev) => ({ ...prev, checkState: "no-conda", condaFound: false }));
        return;
      }

      const allReady = environments.scan_env && environments.sam3;
      setState((prev) => ({
        ...prev,
        checkState: allReady ? "ready" : "missing",
        condaFound: true,
        scanEnvExists: environments.scan_env,
        sam3Exists: environments.sam3,
      }));
    } catch {
      setState((prev) => ({
        ...prev,
        checkState: "error",
        setupError: "Unexpected error checking environments",
      }));
    }
  }, []);

  // Check on mount
  useEffect(() => {
    checkEnvironment();
  }, [checkEnvironment]);

  // Listen for setup IPC events
  useEffect(() => {
    const api = window.electronAPI;

    api.onSetupProgress((data) => {
      setState((prev) => ({
        ...prev,
        currentEnv: data.env,
        setupPhase: data.phase as SetupPhase,
        setupError: data.phase === "error" ? data.message : prev.setupError,
      }));
    });

    api.onSetupLog((line) => {
      setState((prev) => ({
        ...prev,
        setupLogLines: [...prev.setupLogLines.slice(-500), line],
      }));
    });

    return () => {
      api.removeAllListeners("environment:setup-progress");
      api.removeAllListeners("environment:setup-log");
    };
  }, []);

  const setupEnvironments = useCallback(async () => {
    setState((prev) => ({
      ...prev,
      setupActive: true,
      setupLogLines: [],
      setupError: null,
    }));

    // Re-check to get fresh status (avoids stale closure on retry)
    const freshResult = (await window.electronAPI.checkEnvironment()) as CheckResult;
    const envsToSetup: string[] = [];
    if (freshResult.ok && freshResult.status) {
      if (!freshResult.status.environments.scan_env) envsToSetup.push("scan_env");
      if (!freshResult.status.environments.sam3) envsToSetup.push("sam3");
    }

    if (envsToSetup.length === 0) {
      setState((prev) => ({ ...prev, setupActive: false }));
      await checkEnvironment();
      return;
    }

    for (const envName of envsToSetup) {
      setState((prev) => ({
        ...prev,
        currentEnv: envName,
        setupPhase: "creating",
        setupLogLines: [],
      }));

      const result = (await window.electronAPI.setupEnvironment(envName)) as SetupResult;

      if (result.error) {
        setState((prev) => ({
          ...prev,
          setupActive: false,
          setupPhase: "error",
          setupError: result.message || `Failed to set up ${envName}`,
        }));
        return;
      }
    }

    setState((prev) => ({
      ...prev,
      setupActive: false,
      setupPhase: "idle",
      currentEnv: null,
    }));
    await checkEnvironment();
  }, [checkEnvironment]);

  const cancelSetup = useCallback(() => {
    window.electronAPI.cancelEnvironmentSetup();
    setState((prev) => ({
      ...prev,
      setupActive: false,
      setupPhase: "idle",
      currentEnv: null,
    }));
  }, []);

  return { state, checkEnvironment, setupEnvironments, cancelSetup };
}
