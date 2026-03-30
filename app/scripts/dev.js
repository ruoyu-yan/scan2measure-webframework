const { spawn } = require("child_process");
const path = require("path");

const appDir = path.resolve(__dirname, "..");

// Start Vite directly via Node (no cmd.exe shell).
// vite-plugin-electron automatically compiles main/preload and launches Electron.
const vite = spawn(
  process.execPath,
  [path.join(appDir, "node_modules/vite/bin/vite.js")],
  { cwd: appDir, stdio: "inherit", env: process.env }
);

vite.on("close", (code) => process.exit(code ?? 0));

process.on("SIGINT", () => {
  vite.kill();
  process.exit();
});
