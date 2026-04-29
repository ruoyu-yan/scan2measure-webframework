import { ChildProcess, spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { BrowserWindow } from "electron";
import { logInfo, logError } from "./logger";

export type EnvName = "scan_env" | "sam3";

export interface EnvironmentStatus {
  condaFound: boolean;
  condaPath: string | null;
  environments: Record<EnvName, boolean>;
}

export interface SetupProgress {
  env: EnvName;
  phase: string;
  message: string;
}

// Same conda init fallback chain used by PipelineEngine (pipeline-engine.ts:66)
const CONDA_INIT =
  'source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || ' +
  'source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || ' +
  'source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null';

const CONDA_PATHS = [
  "miniconda3/etc/profile.d/conda.sh",
  "anaconda3/etc/profile.d/conda.sh",
  "miniforge3/etc/profile.d/conda.sh",
];

/** Post-create steps per environment. PyTorch needs --extra-index-url which YAML pip sections don't support. */
const POST_CREATE_STEPS: Record<EnvName, { phase: string; cmd: string }[]> = {
  scan_env: [
    {
      phase: "installing-pytorch",
      cmd: `pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116`,
    },
  ],
  sam3: [
    {
      phase: "installing-pytorch",
      cmd: `pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126`,
    },
    {
      phase: "installing-sam3",
      cmd: `pip install -e ./sam3`,
    },
  ],
};

export class EnvironmentManager {
  private window: BrowserWindow;
  private projectRoot: string;
  private activeProcess: ChildProcess | null = null;

  constructor(window: BrowserWindow, projectRoot: string) {
    this.window = window;
    this.projectRoot = projectRoot;
  }

  /** Check conda availability and whether both environments exist. */
  async check(): Promise<EnvironmentStatus> {
    const result: EnvironmentStatus = {
      condaFound: false,
      condaPath: null,
      environments: { scan_env: false, sam3: false },
    };

    // Detect which conda.sh exists
    const home = os.homedir();
    for (const rel of CONDA_PATHS) {
      const full = path.join(home, rel);
      if (fs.existsSync(full)) {
        result.condaPath = full;
        break;
      }
    }

    // Verify conda is reachable
    try {
      await this.exec(`${CONDA_INIT} && conda --version`);
      result.condaFound = true;
    } catch {
      logError("env", "conda not found via fallback chain");
      return result;
    }

    // List environments
    try {
      const output = await this.exec(`${CONDA_INIT} && conda env list --json`);
      const parsed = JSON.parse(output);
      const envPaths: string[] = parsed.envs || [];
      for (const envPath of envPaths) {
        const name = path.basename(envPath);
        if (name === "scan_env") result.environments.scan_env = true;
        if (name === "sam3") result.environments.sam3 = true;
      }
    } catch (err) {
      logError("env", `Failed to list conda envs: ${(err as Error).message}`);
    }

    logInfo("env", `check: conda=${result.condaFound}, scan_env=${result.environments.scan_env}, sam3=${result.environments.sam3}`);
    return result;
  }

  /** Create a missing environment from its YAML file, then run post-create steps. */
  async setup(envName: EnvName): Promise<void> {
    const ymlFile = envName === "scan_env" ? "scan_env.yml" : "sam3_env.yml";
    const ymlPath = path.join(this.projectRoot, ymlFile);

    if (!fs.existsSync(ymlPath)) {
      throw new Error(`Environment YAML not found: ${ymlPath}`);
    }

    // Phase 1: conda env create
    this.sendProgress(envName, "creating", `Creating ${envName} from ${ymlFile}...`);
    logInfo("env", `Creating environment ${envName} from ${ymlPath}`);

    try {
      await this.execStreamed(
        `${CONDA_INIT} && conda env create -f '${ymlPath}' --yes`,
        envName,
      );
    } catch (err) {
      this.sendProgress(envName, "error", (err as Error).message);
      throw err;
    }

    // Phase 2+: post-create steps (PyTorch install, sam3 editable install)
    const steps = POST_CREATE_STEPS[envName] || [];
    for (const step of steps) {
      this.sendProgress(envName, step.phase, `Running: ${step.cmd}`);
      logInfo("env", `Post-create step for ${envName}: ${step.phase}`);

      try {
        await this.execStreamed(
          `${CONDA_INIT} && conda run -n '${envName}' ${step.cmd}`,
          envName,
        );
      } catch (err) {
        this.sendProgress(envName, "error", (err as Error).message);
        throw err;
      }
    }

    this.sendProgress(envName, "done", `${envName} setup complete`);
    logInfo("env", `Environment ${envName} setup complete`);
  }

  /** Cancel any active setup process. */
  cancel(): void {
    if (this.activeProcess?.pid) {
      logInfo("env", `Cancelling setup process ${this.activeProcess.pid}`);
      try {
        process.kill(-this.activeProcess.pid, "SIGTERM");
      } catch {
        // Process may have already exited
      }
      this.activeProcess = null;
    }
  }

  /** Run a shell command and return stdout. Rejects on non-zero exit. */
  private exec(cmd: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const proc = spawn("bash", ["-c", cmd], {
        stdio: ["ignore", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";
      proc.stdout!.on("data", (chunk) => { stdout += chunk.toString(); });
      proc.stderr!.on("data", (chunk) => { stderr += chunk.toString(); });

      proc.on("close", (code) => {
        if (code === 0) resolve(stdout.trim());
        else reject(new Error(stderr.trim() || `Exit code ${code}`));
      });

      proc.on("error", (err) => reject(err));
    });
  }

  /** Run a shell command, streaming stdout/stderr to the renderer via IPC. Rejects on non-zero exit. */
  private execStreamed(cmd: string, envName: EnvName): Promise<void> {
    return new Promise((resolve, reject) => {
      const proc = spawn("bash", ["-c", cmd], {
        cwd: this.projectRoot,
        env: { ...process.env },
        stdio: ["ignore", "pipe", "pipe"],
        detached: true,
      });

      this.activeProcess = proc;
      const stderrLines: string[] = [];

      const streamLine = (line: string) => {
        if (this.window && !this.window.isDestroyed()) {
          this.window.webContents.send("environment:setup-log", line);
        }
      };

      let stdoutRemainder = "";
      proc.stdout!.on("data", (chunk: Buffer) => {
        const text = stdoutRemainder + chunk.toString();
        const lines = text.split("\n");
        stdoutRemainder = lines.pop() || "";
        for (const line of lines) {
          streamLine(line);
        }
      });

      let stderrRemainder = "";
      proc.stderr!.on("data", (chunk: Buffer) => {
        const text = stderrRemainder + chunk.toString();
        const lines = text.split("\n");
        stderrRemainder = lines.pop() || "";
        for (const line of lines) {
          streamLine(line);
          stderrLines.push(line);
          if (stderrLines.length > 30) stderrLines.shift();
        }
      });

      proc.on("close", (code) => {
        this.activeProcess = null;
        // Flush remainders
        if (stdoutRemainder) streamLine(stdoutRemainder);
        if (stderrRemainder) {
          streamLine(stderrRemainder);
          stderrLines.push(stderrRemainder);
        }

        if (code === 0) {
          resolve();
        } else {
          const tail = stderrLines.slice(-30).join("\n");
          reject(new Error(tail || `Exit code ${code}`));
        }
      });

      proc.on("error", (err) => {
        this.activeProcess = null;
        reject(err);
      });
    });
  }

  private sendProgress(env: EnvName, phase: string, message: string): void {
    if (this.window && !this.window.isDestroyed()) {
      this.window.webContents.send("environment:setup-progress", { env, phase, message });
    }
  }
}
