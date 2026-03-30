import fs from "node:fs";
import path from "node:path";

/**
 * Simple file logger for the Electron main process.
 * Writes timestamped lines to data/logs/app.log (rotated per session).
 */

let logStream: fs.WriteStream | null = null;
let logFilePath = "";

/**
 * Initialize the logger. Creates the log directory and opens a write stream.
 * Call once at app startup with the project root path.
 */
export function initLogger(projectRoot: string): void {
  const logDir = path.join(projectRoot, "data", "logs");
  fs.mkdirSync(logDir, { recursive: true });

  // One log file per session, timestamped
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  logFilePath = path.join(logDir, `app_${ts}.log`);

  logStream = fs.createWriteStream(logFilePath, { flags: "a" });
  logStream.write(`=== scan2measure session started ${new Date().toISOString()} ===\n`);

  // Also keep a symlink/copy as "latest.log" for easy access
  const latestPath = path.join(logDir, "latest.log");
  try {
    if (fs.existsSync(latestPath)) fs.unlinkSync(latestPath);
    fs.symlinkSync(logFilePath, latestPath);
  } catch {
    // Symlinks may fail on some systems; ignore
  }
}

function write(level: string, tag: string, message: string): void {
  const ts = new Date().toISOString();
  const line = `${ts} [${level}] [${tag}] ${message}\n`;
  logStream?.write(line);
}

export function logInfo(tag: string, message: string): void {
  write("INFO", tag, message);
}

export function logError(tag: string, message: string): void {
  write("ERROR", tag, message);
}

export function logWarn(tag: string, message: string): void {
  write("WARN", tag, message);
}

/**
 * Write a raw line (for subprocess output).
 * Prefixed with timestamp and tag but no level.
 */
export function logRaw(tag: string, line: string): void {
  const ts = new Date().toISOString();
  logStream?.write(`${ts} [${tag}] ${line}\n`);
}

/** Flush and close the log stream. */
export function closeLogger(): void {
  if (logStream) {
    logStream.write(`=== session ended ${new Date().toISOString()} ===\n`);
    logStream.end();
    logStream = null;
  }
}

/** Get the current log file path (for display in UI or error messages). */
export function getLogPath(): string {
  return logFilePath;
}
