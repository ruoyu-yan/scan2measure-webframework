import { spawn, execSync } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";
import { logInfo, logError } from "./logger";

const IS_WSL = os.platform() === "linux" && fs.existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");

/**
 * Convert a Linux path to a Windows path when running under WSL.
 * Unity.exe is a Windows binary, so all file arguments must be Windows paths.
 */
function toWindowsPath(linuxPath: string): string {
  if (!IS_WSL) return linuxPath;
  return execSync(`wslpath -w "${linuxPath}"`, { encoding: "utf-8" }).trim();
}

/**
 * Locate the Unity executable.
 * Looks for VirtualTour.exe under <projectRoot>/unity/Build/.
 */
function findUnityExe(projectRoot: string): string | null {
  const candidates = [
    path.join(projectRoot, "unity", "Build", "VirtualTour.exe"),
    path.join(projectRoot, "unity", "build", "VirtualTour.exe"),
  ];

  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

/**
 * Search for a camera_pose.json near a GLB file.
 * Checks the GLB directory, its parent, and known project data directories.
 */
function findCameraPose(projectRoot: string, glbPath: string): string | null {
  const glbDir = path.dirname(glbPath);
  const parent = path.dirname(glbDir);

  const candidates = [
    path.join(glbDir, "camera_pose.json"),
    path.join(parent, "camera_pose.json"),
  ];

  // Search in data/pose_estimates/ — prefer rooms over corridors
  const poseEstimatesDir = path.join(projectRoot, "data", "pose_estimates", "multiroom");
  if (fs.existsSync(poseEstimatesDir)) {
    try {
      const subs = fs.readdirSync(poseEstimatesDir).sort((a, b) => {
        // Prefer rooms/offices over corridors (rooms have more solid mesh floors)
        const aIsCorridor = /corridor/i.test(a);
        const bIsCorridor = /corridor/i.test(b);
        if (aIsCorridor !== bIsCorridor) return aIsCorridor ? 1 : -1;
        return a.localeCompare(b);
      });
      for (const sub of subs) {
        candidates.push(path.join(poseEstimatesDir, sub, "camera_pose.json"));
      }
    } catch { /* ignore */ }
  }

  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

/**
 * Launch Unity virtual tour with the given GLB file.
 * Returns true if the process was spawned, false if the exe was not found.
 */
export function launchUnity(
  projectRoot: string,
  glbPath: string,
  minimapPath?: string,
  metadataPath?: string
): boolean {
  const exe = findUnityExe(projectRoot);
  if (!exe) {
    logError("unity", "Unity executable not found");
    return false;
  }

  logInfo("unity", `Found exe: ${exe}`);

  // Unity.exe is a Windows binary — convert all paths to Windows format
  const args = ["--glb", toWindowsPath(glbPath)];
  if (minimapPath && fs.existsSync(minimapPath)) {
    args.push("--minimap", toWindowsPath(minimapPath));
  }
  if (metadataPath && fs.existsSync(metadataPath)) {
    args.push("--metadata", toWindowsPath(metadataPath));
  }

  // Auto-discover camera pose for spawn positioning
  const cameraPose = findCameraPose(projectRoot, glbPath);
  if (cameraPose) {
    args.push("--camera-pose", toWindowsPath(cameraPose));
  }

  logInfo("unity", `Launching: ${exe} ${args.join(" ")}`);

  const proc = spawn(exe, args, {
    cwd: path.dirname(exe),
    detached: true,
    stdio: "ignore",
  });

  // Detach so Unity runs independently of Electron
  proc.unref();
  return true;
}
