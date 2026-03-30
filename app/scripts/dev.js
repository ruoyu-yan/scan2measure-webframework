const { spawn } = require("child_process");
const path = require("path");
const http = require("http");

const appDir = path.resolve(__dirname, "..");

// Start Vite directly via Node (no cmd.exe shell)
const vite = spawn(
  process.execPath,
  [path.join(appDir, "node_modules/vite/bin/vite.js")],
  { cwd: appDir, stdio: "inherit" }
);

// Poll until Vite is serving, then launch Electron
const interval = setInterval(() => {
  http
    .get("http://localhost:5173", (res) => {
      res.resume();
      clearInterval(interval);

      const electronPath = require("electron");
      process.env.VITE_DEV_SERVER_URL = "http://localhost:5173";

      const electron = spawn(electronPath, [appDir], {
        cwd: appDir,
        stdio: "inherit",
        env: process.env,
      });

      electron.on("close", () => {
        vite.kill();
        process.exit();
      });
    })
    .on("error", () => {}); // Vite not ready yet
}, 500);

process.on("SIGINT", () => {
  vite.kill();
  process.exit();
});
