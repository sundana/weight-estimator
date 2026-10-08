#!/usr/bin/env bash
# Phase E provisioning: isolated environment for the ROMBRL D4RL/Tokamak reproduction.
#
# ROMBRL (Chen et al., ICML 2026) runs on offline D4RL MuJoCo + Tokamak tasks and its
# dependency stack (D4RL, dm_control, legacy gym) conflicts with this repo's
# `gymnasium` 1.2.2 environment. This script therefore provisions a *separate* conda
# (or venv) environment and vendors the official ROMBRL repository; it never touches
# the main `distractor-gym` environment.
#
# Usage (inside WSL2 / Linux):
#     bash benchmark/rombri/provision.sh
#     ROMBRI_DRY_RUN=1 bash benchmark/rombri/provision.sh      # plan only
#     ROMBRI_TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash benchmark/rombri/provision.sh
#
# Outputs:
#     benchmark/rombri/readiness.json    machine-readable go/no-go report
#     replicate/rombri/vendor/ROMBRL/    vendored official code
#
# Exit codes: 0 = ready, 1 = provisioning/verification failed (go/no-go blocker).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENDOR_DIR="${ROMBRI_VENDOR_DIR:-$REPO_ROOT/replicate/rombri/vendor}"
ROMBRI_REPO="${ROMBRI_REPO:-https://github.com/Agentic-Intelligence-Lab/ROMBRL.git}"
REQUIREMENTS="$SCRIPT_DIR/requirements-rombri.txt"
READINESS="${ROMBRI_READINESS:-$SCRIPT_DIR/readiness.json}"
ENV_NAME="${ROMBRI_ENV_NAME:-rombri}"
PYTHON_VERSION="${ROMBRI_PYTHON:-3.10}"
TORCH_INDEX="${ROMBRI_TORCH_INDEX:-}"
DRY_RUN="${ROMBRI_DRY_RUN:-0}"
D4RL_REPO="${ROMBRI_D4RL_REPO:-https://github.com/Farama-Foundation/d4rl.git}"

log()  { printf '[provision] %s\n' "$*"; }
warn() { printf '[provision][warn] %s\n' "$*" >&2; }
die()  { printf '[provision][error] %s\n' "$*" >&2; exit 1; }

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[provision][dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

require_linux() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    die "This provisioning script targets Linux/WSL2 (uname=$(uname -s)). On Windows run it via WSL: pwsh benchmark/rombri/provision.ps1"
  fi
}

detect_pm() {
  if command -v mamba >/dev/null 2>&1; then
    echo "mamba"
  elif command -v conda >/dev/null 2>&1; then
    echo "conda"
  else
    echo ""
  fi
}

install_system_deps() {
  log "Checking system packages needed by mujoco-py/D4RL (best effort) ..."
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get not found; ensure gcc/patchelf and OpenGL headers are available."
    return 0
  fi
  local sudo_cmd=""
  if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      sudo_cmd="sudo"
    else
      warn "Not root and sudo unavailable; skipping system package install."
      return 0
    fi
  fi
  run $sudo_cmd apt-get update -y
  run $sudo_cmd apt-get install -y build-essential patchelf \
    libosmesa6-dev libgl1-mesa-glx libglew-dev libglfw3 \
    || warn "Some system packages failed to install; mujoco-py may still work if shipped wheels are available."
}

create_env() {
  local pm="$1"
  if [[ -n "$pm" ]]; then
    log "Creating $pm environment '$ENV_NAME' (python=$PYTHON_VERSION) ..."
    if [[ "$DRY_RUN" == "1" ]]; then
      run "$pm" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
      return 0
    fi
    if ! "$pm" env list | grep -qE "(^|/)${ENV_NAME}( |$)"; then
      run "$pm" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
    fi
    # shellcheck disable=SC1091
    eval "$("$pm" shell.bash hook)"
    run "$pm" activate "$ENV_NAME"
  else
    warn "conda/mamba not found; falling back to python venv (mujoco-py support may be limited)."
    local venv="$REPO_ROOT/.venv-rombri"
    if [[ "$DRY_RUN" == "1" ]]; then
      run python3 -m venv "$venv"
      return 0
    fi
    run python3 -m venv "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
  fi
  run python -m pip install --upgrade pip setuptools wheel
}

install_torch() {
  log "Installing torch ..."
  if [[ -n "$TORCH_INDEX" ]]; then
    run python -m pip install "torch>=2.1" --index-url "$TORCH_INDEX"
  else
    run python -m pip install "torch>=2.1"
  fi
}

install_python_deps() {
  log "Installing core requirements ($REQUIREMENTS) ..."
  run python -m pip install -r "$REQUIREMENTS"
}

install_d4rl() {
  log "Installing D4RL from $D4RL_REPO (legacy MuJoCo stack) ..."
  if ! run python -m pip install "git+${D4RL_REPO}@master"; then
    warn "D4RL install from git failed. The reproduction is BLOCKED until D4RL imports."
    return 1
  fi
  return 0
}

install_dm_control() {
  log "Installing dm_control (Tokamak/DMC tasks) ..."
  run python -m pip install dm-control || warn "dm_control install failed; DMC tasks may be unavailable."
}

vendor_rombri() {
  log "Vendoring official ROMBRL into $VENDOR_DIR/ROMBRL ..."
  run mkdir -p "$VENDOR_DIR"
  if [[ -d "$VENDOR_DIR/ROMBRL/.git" ]]; then
    log "ROMBRL already vendored; leaving as-is."
  else
    run git clone --depth 1 "$ROMBRI_REPO" "$VENDOR_DIR/ROMBRL" || {
      warn "Could not clone ROMBRL (offline?); vendoring skipped."
      return 1
    }
  fi
  # Install the vendored ROMBRL's own requirements when present.
  local req
  for req in "$VENDOR_DIR/ROMBRL/D4RL/requirements.txt" "$VENDOR_DIR/ROMBRL/requirements.txt"; do
    if [[ -f "$req" ]]; then
      log "Installing vendored ROMBRL requirements: $req"
      run python -m pip install -r "$req" || warn "vendored requirements install failed: $req"
      break
    fi
  done
  return 0
}

verify() {
  log "Verifying environment and writing $READINESS ..."
  run python "$SCRIPT_DIR/verify_env.py" --out "$READINESS" --vendor "$VENDOR_DIR/ROMBRL"
}

main() {
  require_linux
  log "Repo root: $REPO_ROOT"
  log "Dry run:   $DRY_RUN"

  local pm
  pm="$(detect_pm)"
  log "Package manager: ${pm:-<none>}"

  install_system_deps
  create_env "$pm"
  install_torch
  install_python_deps
  install_dm_control
  install_d4rl || true
  vendor_rombri || true

  if [[ "$DRY_RUN" == "1" ]]; then
    log "Dry-run complete: no changes made. Re-run without ROMBRI_DRY_RUN=1 to provision."
    exit 0
  fi

  if verify; then
    log "READY: ROMBRL reproduction environment provisioned."
    exit 0
  fi
  warn "NOT READY: see $READINESS for the missing components (go/no-go blocker)."
  exit 1
}

main "$@"
