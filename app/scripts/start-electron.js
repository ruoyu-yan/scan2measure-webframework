const { spawn } = require("child_process");
const path = require("path");

const electronPath = require("electron");
const appDir = path.resolve(__dirname, "..");

process.env.VITE_DEV_SERVER_URL = "http://localhost:5173";

const proc = spawn(electronPath, [appDir], {
  stdio: "inherit",
  env: process.env,
});

proc.on("close", (code) => process.exit(code ?? 0));
