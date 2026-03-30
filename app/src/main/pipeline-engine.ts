import { ChildProcess, spawn } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import { BrowserWindow } from "electron";
import { logInfo, logError, logRaw } from "./logger";

/** Parsed progress line from Python stdout. */
interface ProgressUpdate {
  current: number;
  total: number;
  message: string;
}

/**
 * Parse a stdout line for the [PROGRESS] protocol.
 * Format: [PROGRESS] <current> <total> <message>
 * Returns null if the line does not match.
 */
function parseProgressLine(line: string): ProgressUpdate | null {
  const match = line.match(/^\[PROGRESS\]\s+(\d+)\s+(\d+)\s+(.*)$/);
  if (!match) return null;
  return {
    current: parseInt(match[1], 10),
    total: parseInt(match[2], 10),
    message: match[3].trim(),
  };
}

export interface StageRunConfig {
  scriptPath: string;       // absolute path to Python script
  condaEnv: string;         // conda environment name
  configJsonPath: string;   // absolute path to stage config JSON
  projectRoot: string;      // absolute path to repo root
}

export class PipelineEngine {
  private activeProcess: ChildProcess | null = null;
  private stderrBuffer: string[] = [];
  private window: BrowserWindow;

  constructor(window: BrowserWindow) {
    this.window = window;
  }

  /**
   * Write a stage-specific config JSON to disk.
   * Returns the absolute path to the written file.
   */
  writeStageConfig(outputDir: string, stageId: string, config: Record<string, unknown>): string {
    const configPath = path.join(outputDir, `${stageId}_config.json`);
    fs.mkdirSync(path.dirname(configPath), { recursive: true });
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
    return configPath;
  }

  /**
   * Run a single Python script via conda as a subprocess.
   * Returns a promise that resolves on exit code 0 and rejects on non-zero.
   */
  runStage(config: StageRunConfig): Promise<void> {
    return new Promise((resolve, reject) => {
      this.stderrBuffer = [];

      const args = [
        "run", "--no-banner", "-n", config.condaEnv,
        "python", config.scriptPath,
        "--config", config.configJsonPath,
      ];

      logInfo("pipeline", `Starting stage: conda ${args.join(" ")}`);
      logInfo("pipeline", `  cwd: ${config.projectRoot}`);

      const proc = spawn("conda", args, {
        cwd: config.projectRoot,
        env: { ...process.env },
        stdio: ["ignore", "pipe", "pipe"],
        detached: true,
      });

      this.activeProcess = proc;

      // stdout: parse [PROGRESS] lines, forward others as log
      let stdoutRemainder = "";
      proc.stdout!.on("data", (chunk: Buffer) => {
        const text = stdoutRemainder + chunk.toString();
        const lines = text.split("\n");
        stdoutRemainder = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          logRaw("stdout", line);
          const progress = parseProgressLine(line);
          if (progress) {
            this.window.webContents.send("pipeline:progress", progress);
          } else {
            this.window.webContents.send("pipeline:log", line);
          }
        }
      });

      // stderr: collect last 30 lines
      let stderrRemainder = "";
      proc.stderr!.on("data", (chunk: Buffer) => {
        const text = stderrRemainder + chunk.toString();
        const lines = text.split("\n");
        stderrRemainder = lines.pop() || "";

        for (const line of lines) {
          logRaw("stderr", line);
          this.stderrBuffer.push(line);
          if (this.stderrBuffer.length > 30) {
            this.stderrBuffer.shift();
          }
        }
      });

      proc.on("close", (code) => {
        this.activeProcess = null;
        if (code === 0) {
          logInfo("pipeline", `Stage exited with code 0 (success)`);
          resolve();
        } else {
          logError("pipeline", `Stage exited with code ${code}`);
          logError("pipeline", `stderr tail:\n${this.stderrBuffer.join("\n")}`);
          reject(new Error(this.stderrBuffer.join("\n")));
        }
      });

      proc.on("error", (err) => {
        this.activeProcess = null;
        logError("pipeline", `Spawn error: ${err.message}`);
        reject(err);
      });
    });
  }

  /** Cancel the currently running subprocess and its entire process group. */
  cancel(): void {
    if (this.activeProcess && this.activeProcess.pid) {
      try {
        process.kill(-this.activeProcess.pid, "SIGTERM");
      } catch {
        this.activeProcess.kill("SIGTERM");
      }
      this.activeProcess = null;
    }
  }

  /** Get the last 30 stderr lines from the most recent run. */
  getStderrTail(): string {
    return this.stderrBuffer.join("\n");
  }
}
