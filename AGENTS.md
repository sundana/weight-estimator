# AGENTS.md

Research scaffold, migrated to the WP1-WP4 plan in
`C:\Users\USER\Documents\Obsidian Vault\Research Plan` (value-aware loss stabilization
and scalability for MBRL). Current status: **WP1 complete** — the tabular ground-truth
suite, weighted losses, exact policy-gradient machinery, the deep Distractor-Gym
(MuJoCo + torch), the deep Exp 1.1 gradient-alignment diagnostic, and the Exp 1.2 VJP
profiling harness are implemented and committed (`paper/wp1/notes/wp1_report.md`). WP2-WP4
(coupled-stability theory, scaling/latent/goal-conditioned, synthesis) are planned work.

## Run / verify

- Package is **not** pip-installed. Set the import path first (PowerShell):
  `$env:PYTHONPATH="$PWD\src"` — or `pip install -e .`
- Tests (only need numpy + pytest):
  `$env:PYTHONPATH="$PWD\src"; python -m pytest tests -q`
- `ruff` is configured in `pyproject.toml` but not installed in this environment.

## Structure

- `src/distractor_gym/` — package: `core.py` (RegimeConfig + distractor dynamics),
  `tabular.py`/`continuous.py` (env suites), `agents.py` (exact tabular policy
  gradient, model fitting, rollouts), `losses.py` (weighted loss family),
  `diagnostics.py` (public diagnostics API, smoke-tested)
- `src/distractor_gym/deep/` — torch model-learning stack (nets, VJP, offline training)
  for the WP1 deep diagnostics; imports torch lazily
- `src/distractor_gym/profiling/` — per-sample VJP hardware profiling harness (WP1 Exp 1.2)
- `experiments/` — tabular (exp1/1b/2/3/5), deep (`exp1_deep_alignment`), and profiling
  (`exp1_profiling`) entry points; `configs/*.yaml` feed them (keep knobs in sync);
  `experiments/report.py` writes `paper/wp1/tables/*.tex` from committed `runs/`
- `paper/` — one folder per work-package paper: `wp1/` (ICML/ICLR diagnostics+profiling,
  `main.tex`, `notes/`, `tables/`), `wp2/` (COVAL theory+algorithm), `wp3/`
  (scaling/latent/goal-conditioned), `wp4/` (dissertation); index in `paper/README.md`,
  shared `paper/shared/{macros.tex,references.bib}`
- `paper/wp1/notes/theory.md` states Lemma 1, Theorems 1-3; `paper/wp1/notes/wp1_report.md`
  records WP1 results (H1.1/H1.2/H1.3); `paper/wp1/notes/outline.md` — outline + open items

## Conventions

- Functions throwing `NotImplementedError` are **planned work** (per the WP1-WP4 plan),
  not bugs — do not implement them unless the corresponding phase is in scope.
- Tabular experiments are ground-truth labs: verify exact quantities against
  finite-difference checks before trusting them (`tests/test_agents.py`).
- Optional extras in `pyproject.toml`: `deep` (torch), `mujoco` (`gymnasium[mujoco]`),
  `stats` (`rliable`), `bench` (mbrl-lib — pins old `gym`, provisioning open).
- WP1 profiling runs on the local RTX 5060 Ti (16GB); the plan's A100/4090 absolute
  numbers are reported as relative overhead ratios, not reproduced exactly.
- `exp4_continuous` needs torch; the WP3 deep benchmark needs `mbrl-lib` + `dm_control`,
  which are not installed.
- Docstrings state the math/API contract; no inline comments in source.

## Commits

Conventional Commits: `<type>(<scope>): <summary>`

- Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`
- Scope: package area — `env`, `losses`, `diagnostics`, `configs`, `experiments`,
  `paper`, `tests`, `plan`
- Summary: imperative mood, lowercase, ≤ 72 chars, no trailing period; body (if any)
  explains the *why* (e.g., the research result a change was driven by)
- Stub/planned-work markers in code are never committed as fixes

Example: `feat(diagnostics): add decompose_td_error fit with curvature residual`