#!/usr/bin/env bash
# setup-native.sh - one-shot setup for the Full Pipeline.
#
# Initializes git submodules, applies required source patches, installs
# system build deps (apt), and CMake-builds the three C++ tools used by the
# pipeline: 3DLineDetection, PoissonRecon, mvs-texturing.
#
# SAM3 model weights are NOT downloaded here - the inference scripts auto-fetch
# them from HuggingFace facebook/sam3 on first run (you may need to run
# `huggingface-cli login` once first if the model is gated).
#
# Run from the repository root:
#   bash scripts/setup-native.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\033[1;36m[setup-native]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup-native]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[setup-native]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. Submodules ---------------------------------------------------------
log "Initializing git submodules..."
git submodule update --init --recursive

# ---- 2. Apply patches ------------------------------------------------------
apply_patch_if_clean() {
    local repo="$1" patch="$2"
    if [[ ! -d "$repo" ]]; then
        die "submodule directory $repo missing - did 'git submodule update' fail?"
    fi
    if [[ ! -f "$patch" ]]; then
        die "patch file $patch missing"
    fi
    if ( cd "$repo" && git diff --quiet ); then
        log "Applying $patch -> $repo"
        ( cd "$repo" && git apply "$REPO_ROOT/$patch" )
    else
        warn "$repo working tree is dirty; skipping patch (assume already applied)"
    fi
}

apply_patch_if_clean "3DLineDetection" "scripts/patches/3dlinedetection.patch"
apply_patch_if_clean "mvs-texturing"   "scripts/patches/mvs-texturing.patch"

# ---- 3. System packages ----------------------------------------------------
APT_PKGS=(
    build-essential cmake git pkg-config
    libpng-dev libjpeg-dev libtiff-dev libtbb-dev
    libopencv-dev
)

if command -v apt-get >/dev/null 2>&1; then
    log "Installing apt packages (sudo): ${APT_PKGS[*]}"
    sudo apt-get update
    sudo apt-get install -y "${APT_PKGS[@]}"
else
    warn "apt-get not found - skipping system package install. Ensure you have:"
    warn "  ${APT_PKGS[*]}"
fi

# ---- 4. CMake builds -------------------------------------------------------
JOBS="$(nproc 2>/dev/null || echo 4)"

build_cmake_project() {
    local dir="$1"
    log "Building $dir (-j$JOBS)"
    cmake -S "$dir" -B "$dir/build"
    cmake --build "$dir/build" -j"$JOBS"
}

build_cmake_project "3DLineDetection"
build_cmake_project "mvs-texturing"

# PoissonRecon ships a Makefile, not CMakeLists.txt - build via make.
log "Building PoissonRecon (-j$JOBS)"
make -C PoissonRecon -j"$JOBS"

# ---- 5. Final checks -------------------------------------------------------
log "Verifying built artifacts..."
declare -a EXPECTED=(
    "3DLineDetection/build/src/LineFromPointCloud"
    "mvs-texturing/build/apps/texrecon/texrecon"
    "PoissonRecon/Bin/Linux/PoissonRecon"
    "PoissonRecon/Bin/Linux/SurfaceTrimmer"
)
missing=0
for f in "${EXPECTED[@]}"; do
    if [[ -x "$f" ]]; then
        log "  OK   $f"
    else
        warn "  MISSING $f"
        missing=$((missing + 1))
    fi
done

if (( missing > 0 )); then
    die "$missing expected binary/binaries are missing. Check build logs above."
fi

log "Native setup complete."
log "Next steps:"
log "  1. Open the Electron app: bash launch.sh   (or scan2measure.bat from Windows)"
log "  2. The app will prompt to create the conda environments on first launch."
log "  3. SAM3 weights auto-download from HuggingFace on first inference run."
log "     If 'facebook/sam3' is gated, run: huggingface-cli login"
