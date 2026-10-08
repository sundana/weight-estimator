# Phase E — ROMBRL reproduction (provisioning)

Provisioning for the WP3 comparison baseline **ROMBRL** (Chen, Xu, Venugopal & Schneider,
ICML 2026; arXiv:2505.13709), the first formal coupled-convergence result in MBRL
(Stackelberg game + two-timescale learning rates). It is the empirical baseline behind
the novelty boundary recorded in `paper/coval/notes/coval_scope.md`.

## Why a separate environment

ROMBRL runs on offline **D4RL MuJoCo** + **Tokamak** tasks and depends on the legacy
MuJoCo stack (`D4RL`, `dm_control`, legacy `gym`). That stack conflicts with this repo's
`gymnasium` 1.2.2 environment (same reason `mbrl-lib` is not installed here; see
`AGENTS.md`). Phase E therefore never touches the main `distractor-gym` environment.

## Hard requirements (go/no-go)

- **Linux or WSL2.** Native Windows cannot install D4RL / mujoco-py. This repo's host is
  win32, so use the PowerShell wrapper, which delegates into WSL2.
- `conda` or `mamba` (recommended) with a writable env; `mamba` preferred.
- A CUDA-capable GPU + driver for GPU training (CPU works but is very slow).
- Network access for PyPI + the ROMBRL/D4RL git remotes.

## Usage

Inside Linux/WSL2:

```bash
bash benchmark/rombri/provision.sh
ROMBRI_DRY_RUN=1 bash benchmark/rombri/provision.sh          # show plan, install nothing
ROMBRI_TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
  bash benchmark/rombri/provision.sh                          # CUDA 12.1 torch wheel
```

From Windows (delegates to WSL2):

```powershell
pwsh benchmark/rombri/provision.ps1
pwsh benchmark/rombri/provision.ps1 -DryRun
pwsh benchmark/rombri/provision.ps1 -TorchIndex https://download.pytorch.org/whl/cu121
```

## What the script does

1. Detects Linux, the package manager (`mamba`/`conda`, else `venv` fallback).
2. Creates an isolated env named `rombri` (python 3.10 by default).
3. Installs torch, then `requirements-rombri.txt`, then `dm-control`.
4. Installs D4RL from the Farama fork (`https://github.com/Farama-Foundation/d4rl.git`).
5. Vendors the official ROMBRL repo to `replicate/rombri/vendor/ROMBRL` and installs its
   own `D4RL/requirements.txt` when present.
6. Runs `verify_env.py`, writing `benchmark/rombri/readiness.json`.

Exit code `0` = ready; `1` = blocker (inspect `readiness.json`). `readiness.json` is the
automated go/no-go signal: if D4RL does not import, **Phase E is blocked** and Phases A–D
(the analytic + neural COVAL stability work) proceed independently.

## Overrides

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `ROMBRI_ENV_NAME` | `rombri` | conda/venv env name |
| `ROMBRI_PYTHON` | `3.10` | env Python version |
| `ROMBRI_TORCH_INDEX` | unset | custom torch wheel index (CUDA build) |
| `ROMBRI_REPO` | official ROMBRL git URL | vendor source |
| `ROMBRI_D4RL_REPO` | Farama `d4rl` git URL | D4RL source |
| `ROMBRI_VENDOR_DIR` | `replicate/rombri/vendor` | vendor destination |
| `ROMBRI_READINESS` | `benchmark/rombri/readiness.json` | report path |
| `ROMBRI_DRY_RUN` | `0` | `1` prints commands only |

## After provisioning

The reproduction driver and config (`experiments/exp3_rombri_baseline.py`,
`configs/exp3_rombri_baseline.yaml`) are the WP3 entry points; they run the vendored
checkout inside the `rombri` env on 12 noisy D4RL MuJoCo tasks + 3 Tokamak tasks (3 seeds)
and report mean score, performance drop, and `rliable` CIs. Until those land, this folder
is a documented, self-verifying scaffold.

## Status

Scaffold. `readiness.json` is not committed (generated at provision time). No reproduction
numbers exist yet; the ROMBRL ordering comparison used by EXP 2.1 is implemented
separately in `src/distractor_gym/deep/coupled.py` (no D4RL required).
