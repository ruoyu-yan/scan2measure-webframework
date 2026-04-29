#!/bin/bash -l
# scan2measure launcher (WSL/Linux)
# Delegates to scripts/launch-appimage.sh which handles WSLg env defaults.
REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec "$REPO_ROOT/scripts/launch-appimage.sh" "$@"
