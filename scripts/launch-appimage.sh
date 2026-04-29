#!/bin/bash -l
# launch-appimage.sh - launch the scan2measure AppImage with WSLg env defaults.
#
# Called from scan2measure.bat (Windows double-click) and launch.sh (WSL).
# Centralizing env handling here avoids cmd.exe quoting hell - cmd just runs
# this script verbatim through wsl.exe.

# WSLg env vars - keep current value if set, otherwise fall back to defaults.
# This handles the case where wsl.exe is invoked from cmd.exe and the WSL
# session lost DISPLAY (Electron crashes with "Missing X server or $DISPLAY").
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
APPIMAGE="$REPO_ROOT/app/release/scan2measure.AppImage"

if [[ ! -x "$APPIMAGE" ]]; then
    echo "ERROR: AppImage not found or not executable at $APPIMAGE" >&2
    exit 1
fi

exec "$APPIMAGE" "$@"
