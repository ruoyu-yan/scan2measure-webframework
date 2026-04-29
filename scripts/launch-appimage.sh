#!/bin/bash -l
# launch-appimage.sh - launch the scan2measure AppImage with WSLg env defaults.
#
# Called from scan2measure.bat (Windows double-click) and launch.sh (WSL).
# Centralizing env handling here avoids cmd.exe quoting hell - cmd just runs
# this script verbatim through wsl.exe.

# WSLg env vars - keep current value if set, otherwise fall back to defaults.
# Handles the case where wsl.exe is invoked from cmd.exe and the session
# loses DISPLAY (Electron then crashes with "Missing X server or $DISPLAY").
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
# Strip trailing slash so paths don't end up like /run/user/1000//wayland-0
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR%/}"
export XDG_RUNTIME_DIR

# Sanity-check that a display server is actually reachable. WSLgd
# (the WSLg daemon) occasionally fails at WSL boot and leaves dangling
# Wayland / X11 socket symlinks. In that state Electron crashes
# immediately at startup with the same "Missing X server" error.
WAYLAND_OK=false
X11_OK=false
[[ -S "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" ]] && WAYLAND_OK=true
[[ -S "/tmp/.X11-unix/X${DISPLAY#:}" ]] && X11_OK=true

if ! $WAYLAND_OK && ! $X11_OK; then
    cat >&2 <<EOF
ERROR: No working WSLg display server.

Wayland and X11 sockets are both missing or broken. This usually means
WSLgd (the WSLg daemon) failed to start when WSL booted.

To fix:
  1. From a Windows PowerShell (or cmd.exe), run:
       wsl --shutdown
  2. Wait ~5 seconds.
  3. Double-click scan2measure.bat again. WSL will cold-start and
     WSLg should come up cleanly.

Diagnostic:
  Expected Wayland socket: $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY
  Expected X11 socket:     /tmp/.X11-unix/X${DISPLAY#:}
  /mnt/wslg/stderr.log (last 3 lines):
$(tail -3 /mnt/wslg/stderr.log 2>/dev/null | sed 's/^/    /')
EOF
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
APPIMAGE="$REPO_ROOT/app/release/scan2measure.AppImage"

if [[ ! -x "$APPIMAGE" ]]; then
    echo "ERROR: AppImage not found or not executable at $APPIMAGE" >&2
    exit 1
fi

exec "$APPIMAGE" "$@"
