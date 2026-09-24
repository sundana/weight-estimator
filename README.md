# Distractor-Gym: Weight-Estimator Theory & Mismatch Diagnostic

When and why does value-aware model learning fail? A diagnostic suite for objective
mismatch in model-based RL, with a theory of the bias-variance of the weight estimator
`w`.

See `RESEARCH_PLAN.md` for the full research plan.

## Layout

```
RESEARCH_PLAN.md        research plan and paper outline
src/distractor_gym/     core package: envs, losses, diagnostics
configs/                regime + baseline configuration files
experiments/            experiment entry points (Exp 1-3)
paper/                  notes and drafts
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
written up in `paper/draft_v2.md` and `paper/notes/theory.md`. The continuous neural
experiment (Exp 4) needs `torch` (not installed) and is withdrawn pending a coordinate
scaling fix. The deep-baseline benchmark (MBPO/VaGraM/DreamerV3/TD-MPC2, Phase 1c/2/3)
needs `mbrl-lib` + `dm_control` provisioning and remains planned work
(`NotImplementedError`).
