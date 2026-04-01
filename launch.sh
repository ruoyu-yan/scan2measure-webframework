#!/bin/bash -l
# scan2measure launcher
# -l (login shell) ensures conda is initialized via .bash_profile/.bashrc
REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec "$REPO_ROOT/app/release/scan2measure.AppImage"
