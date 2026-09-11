# Phase 0 Implementation: Repo Scaffold, Diagnostics API, Theory Skeleton

Phase 0 (RESEARCH_PLAN.md Sec. 8, est. 1–2 wk) lays the foundation on which all
later phases build: the repository structure, the public diagnostics API, and the
skeleton of the theory write-up. It does **not** include the environments,
experiments, or empirical validation — those are Phase 1a/1b/1c.

This document records what was actually implemented and where each piece lives, so
the deliverable of Phase 0 is auditable and reproducible.

---

## 1. Overview

Phase 0 delivers three things:

1. **Repo scaffold** — the package layout, build configuration, and entry points
   that every later phase plugs into.
2. **Diagnostics API** — the public functions that quantify objective mismatch and
   weight-estimator quality, as the interface between experiments and the theory.
3. **Theory skeleton** — the statements of Lemma 1 and Theorems 1–3, written up in
   `paper/notes/theory.md` so the empirical phases have an object to validate.

The scaffolding deliberately carries no heavy dependencies: Phase 0–1b need only
`numpy` + `pytest` (the `torch`/`mbrl-lib`/`dm_control` stack is gated behind
optional extras and deferred to the deep-baseline phases).

---

## 2. Repo scaffold

```
RESEARCH_PLAN.md              research plan, phases, theory program, timeline
AGENTS.md                     developer conventions and run instructions
README.md                     layout, setup, run commands, status
pyproject.toml                build system + dependency extras
src/distractor_gym/           the package (envs, losses, diagnostics)
configs/                      regime + baseline YAML configs
experiments/                  experiment entry points (Exp 1–4)
tests/                        pytest suite
paper/                        notes + drafts
replicate/                    Lambert et al. reproduction artifacts
runs/                         experiment output
```

Key decisions in the scaffold:

- **`src/` layout** (`[tool.setuptools.packages.find] where = ["src"]`): the package
  is *not* pip-installed by default. Set `PYTHONPATH=$PWD/src` (PowerShell
  `$env:PYTHONPATH="$PWD\src"`) or `pip install -e .`.
- **Minimal core dependencies** (`pyproject.toml`): `numpy`, `scipy`, `pyyaml`,
  `gymnasium`. Optional extras keep the heavy stack out of the base install:
  - `dev` — `pytest`, `ruff`, `mypy`
  - `deep` — `torch` (used by the continuous neural-model experiments, Exp 4)
  - `bench` — `mbrl-lib` (deep baselines, Phase 1c/2; provisioning is an open item)
- **Configs decoupled from code**: experiments read `configs/*.yaml` and keep the
  knob names in sync with `RegimeConfig` in `src/distractor_gym/core.py`.
- **`runs/`** holds experiment outputs (`results.json`, figures) by experiment.

---

## 3. Core foundation (`src/distractor_gym/core.py`, `losses.py`)

These modules are the minimal substrate the diagnostics API operates on.

### `core.py` — regime configuration and distractor dynamics

- `RegimeConfig` — the dataclass of swept knobs for the phase diagram
  (RESEARCH_PLAN.md Sec. 3): control dims `d_c`, distractor dims `d_d`,
  `distractor_class`, `reward_sparsity`, `goal_radius`, `transition_noise`,
  `model_capacity`, `seed`.
- `DistractorClass` — the three distractor dynamics families ordered by failure
  pressure: `LINEAR` (coupled oscillators), `NONLINEAR` (tanh/logistic), `STOCHASTIC`
  (noisy random walk). Implemented as `DistractorDynamics.step`, which advances the
  reward-irrelevant block `s_d` and leaves the control block `s_c` untouched.
- By construction distractors have **zero reward relevance**, so
  `grad_{s_d} V = 0` (RESEARCH_PLAN.md Sec. 3) — the property that later drives the
  value-aware failure regimes.

### `losses.py` — weighted model-learning loss family

- `LossFamily` — the objectives compared across experiments: `MLE` (`w == 1`),
  `VAML1` (`|V(s') - V(s)|`), `VAGRAM` (`||grad_s V(s')||`), `LAMBERT`
  (exponential tilt), `DECISION_ALIGNED` (external weight function, e.g. Bellman
  residual).
- `weight(family, V_s, V_sp, grad_V_sp, weight_fn, tau)` — computes the per-transition
  weight `w(s, a, s')` from the *estimated* value quantities; this `w_hat` is the
  random variable whose bias/variance the theory studies (RESEARCH_PLAN.md Sec. 2.1).
- `weighted_loss` — `L_w(theta) = E_D[ w * ell(p_theta) ]` with Laplace-clipped NLL.

---

## 4. Diagnostics API (`src/distractor_gym/diagnostics.py`)

The public API is re-exported from `distractor_gym/__init__.py`. Every function is
pure `numpy`, smoke-tested in `tests/test_diagnostics.py`.

### `gradient_alignment(g_true, g_model) -> float`

Cosine similarity between the true and the model-induced policy gradient
(RESEARCH_PLAN.md Sec. 4):

```
cos(g_true, g_model) = (g_true . g_model) / (||g_true|| ||g_model||)
```

Values near 1 mean the model-based update tracks the true improvement direction.
Zero-norm guard returns 0.0.

### `decompose_td_error(...) -> DecompositionResult`

Empirical test of the first-order factorization (RESEARCH_PLAN.md Sec. 2.2):

```
|delta_TD| ~ ||grad V(s')|| * eps_model * |cos phi|
```

with `delta_td = |V(s') - V(s_hat')|`, `eps_model = ||s_hat' - s'||`, and `cos_phi`
the alignment of the error vector with `grad V`. Reports:
- `r2`, `slope` — quality of the least-squares fit of `pred = grad_norm * eps *
  |cos_phi|` against `delta_td`;
- `mean_cos_phi` — the |cos phi| distribution;
- `curvature_residual` — mean |delta_td - pred|, the unexplained second-order term
  (`(1/2) ||H_V||_2 ||e||^2` from Lemma 1).

This is the object that separates "value-gradient mis-specification" (small
`cos_phi` / large curvature residual) from "value-flatness" (small `||grad V||`).

### `weight_estimator_stats(w, w_oracle, weighted_losses) -> WeightStats`

Bias/variance/ESS/SNR of the weight estimator (RESEARCH_PLAN.md Sec. 2.3, Sec. 5
Part B):

- `bias = mean(w - w_oracle)` — estimated vs oracle (true-V) weights; isolates
  Theorem-1 `Bias^2`.
- `variance = var(w)` — weight-induced variance.
- `effective_sample_size = (sum w)^2 / sum w^2` — Kish ESS; shrinks as `var(w)`
  grows (Theorem 2).
- `signal_to_noise` — the Theorem-1 crossover statistic (below).

### `weight_signal_to_noise(weighted_losses, w) -> float`

The crossover statistic `SNR_w` (RESEARCH_PLAN.md Sec. 2.3):

```
SNR_w = (E[w * ell])^2 / Var(w * ell)
```

Below a threshold this predicts MLE (`w == 1`) has strictly lower risk; computed
per regime to check against the Theorem-1 crossover.

### Dataclasses

- `DecompositionResult` — `r2`, `slope`, `mean_cos_phi`, `curvature_residual`, `n`.
- `WeightStats` — `bias`, `variance`, `effective_sample_size`, `signal_to_noise`.

Both are lightweight value objects returned by the API functions.

---

## 5. Theory skeleton (`paper/notes/theory.md`, `outline.md`)

Phase 0 establishes the *statements*; the proofs/validation notes are filled in
during Phase 1a/1b. Skeleton contents:

- **Lemma 1 (delta_TD decomposition)** — Taylor expansion of the model-induced TD
  error with the curvature (second-order, `H_V`) remainder bound.
- **Theorem 1 (prediction-risk dominance of MLE)** — ratio-estimator argument:
  value-aware weighting adds a variance penalty `sigma_w^2 E[V^2]/n` and a bias
  `Cov(w, V)^2`, so it cannot improve conditional-mean (Bellman) prediction.
- **Theorem 2 (effective sample size)** — `E[ESS] ~ n / (1 + sigma_w^2)`, the
  variance inflation of Theorem 1 expressed as a sample loss.
- **Theorem 3 (decision-risk crossover)** — the signal (`grad V * Cov(w, Y)`) vs
  penalty (`||grad V||^2 sigma_w^2 E[||Y||^2]/n`) criterion behind `SNR_w`.

`paper/notes/outline.md` maps these to the paper sections (RESEARCH_PLAN.md Sec. 7).
The skeleton is intentionally provisional: `theory.md` records the empirical checks
that later phases add next to each statement.

---

## 6. Verification and status

Phase 0 is verified by the unit-level tests on the foundation and diagnostics:

```powershell
$env:PYTHONPATH = "$PWD\src"; python -m pytest tests -q
```

Targeted at this phase: `tests/test_diagnostics.py` and `tests/test_losses.py`
(parallel/orthogonal alignment, perfect-decomposition fit, SNR degenerate/formula,
weight-family semantics, ESS). `ruff` is configured in `pyproject.toml`
(line-length 100, py310).

**Status:** Phase 0 complete. It hands off to Phase 1a the exact-machinery skeleton
(`core.py`, `losses.py`, `diagnostics.py`) and the theory statements that Phase 1a/1b
validate in the tabular suite (see `README.md` Status and `paper/notes/outline.md`).