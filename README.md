# Distractor-Gym: Weight-Estimator Theory & Mismatch Diagnostic

When and why does value-aware model learning fail? A diagnostic suite for objective
mismatch in model-based RL, with a theory of the bias-variance of the weight estimator
`w`.

See the WP1-WP4 plan in `C:\Users\USER\Documents\Obsidian Vault\Research Plan` for the
full research programme (value-aware loss stabilization and scalability for MBRL).

## Layout

```
src/distractor_gym/     core package: envs, losses, diagnostics
src/distractor_gym/deep/      torch model-learning stack (WP1 deep diagnostics)
src/distractor_gym/profiling/ per-sample VJP hardware profiling (WP1 Exp 1.2)
configs/                regime + baseline configuration files
experiments/            experiment entry points (Exp 1-3 + 1b + 4 + 5)
paper/                  per-WP papers (wp1/ wp2/ wp3/ wp4/) + shared/ assets
```

## Setup

The package is **not** pip-installed by default. Either install it editable:

```powershell
pip install -e .
```

or set the import path once per shell (no install needed):

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Tests require only `numpy` and `pytest`.

## Run

Tests:

```powershell
$env:PYTHONPATH = "$PWD\src"; python -m pytest tests -q
```

Experiments (`experiments/exp1_alignment.py`, `exp2_decomposition.py`,
`exp3_benchmark.py`):

```powershell
python -m experiments.exp1_alignment --config configs/tabular_default.yaml
```

Configs in `configs/` feed the experiments (regime knobs, loss family, seeds); keep
the knob names in sync between `configs/*.yaml` and `src/distractor_gym/core.py`
(`RegimeConfig`). Deep-baseline runs (MBPO/VaGraM, then DreamerV3/TD-MPC2) are
configured in `configs/phase1_mbpo_vagram.yaml` and require the optional `bench`
dependencies.

## Status

Phase 1a/1b/1c. The tabular Distractor-Gym suite, exact policy-gradient machinery,
weighted loss families, and the diagnostics API are implemented and tested. The
distractor family (`linear`/`nonlinear`/`stochastic`) is functional, coverage is drawn
i.i.d. (`uniform`/`goal`), and the transition model can be the uncapacitated empirical
table or a **reduced-rank feature-budget** model (`model_kind: feature`,
`capacity`/`model_noise`). Experiments write a `manifest.json` (git SHA + config +
versions) next to their results.

The revised tabular results are in `runs/` (10 seeds each): `exp1`/`exp2` (uncapacitated
ground-truth), `exp1_capacity`/`exp2_capacity` (capacity-limited + stochastic
distractors), and `exp1b`/`exp1b_capacity` (crossover). Corrected theory and results are
written up in `paper/wp1/notes/archive/draft_v2.md` and `paper/wp1/notes/theory.md`. The continuous neural
experiment (Exp 4) needs `torch` (not installed) and is withdrawn pending a coordinate
scaling fix. The deep-baseline benchmark (MBPO/VaGraM/DreamerV3/TD-MPC2, Phase 1c/2/3)
needs `mbrl-lib` + `dm_control` provisioning and remains planned work
(`NotImplementedError`).

### WP1 (complete)

The repo is migrated to the WP1-WP4 plan. The tabular suite is retained as the
ground-truth lab (H1.2); `src/distractor_gym/deep/` holds the deep Distractor-Gym
(MuJoCo + analytic), the weighted model losses, and Exp 1.1 policy-gradient alignment;
`src/distractor_gym/profiling/` holds the Exp 1.2 per-sample VJP profiling harness.
`experiments/report.py` generates `paper/wp1/tables/*.tex` from the committed `runs/`.
Results are summarised in `paper/wp1/notes/wp1_report.md`. Optional extras:
`pip install -e ".[mujoco,stats]"` for `gymnasium[mujoco]` + `rliable`. Profiling runs on
the local RTX 5060 Ti (H1.3 exact VJP fails, stale caching passes); the plan's A100/4090
numbers are reported as relative overhead ratios.
