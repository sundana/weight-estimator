# AGENTS.md

Research scaffold for the paper *"When and why does value-aware model learning fail?"*
Full plan: `RESEARCH_PLAN.md`. Current status: Phase 0 — most code is intentionally stubbed.

## Run / verify

- Package is **not** pip-installed. Set the import path first (PowerShell):
  `$env:PYTHONPATH="$PWD\src"` — or `pip install -e .`
- Tests (only need numpy + pytest):
  `$env:PYTHONPATH="$PWD\src"; python -m pytest tests -q`
- `ruff` is configured in `pyproject.toml` but not installed in this environment.

## Structure

- `src/distractor_gym/` — package: `core.py` (RegimeConfig + distractor dynamics),
  `tabular.py`/`continuous.py` (env suites), `losses.py` (weighted loss family),
  `diagnostics.py` (public diagnostics API, smoke-tested)
- `experiments/` — Exp 1–3 entry points; `configs/*.yaml` feed them (keep knobs in sync)
- `paper/notes/outline.md` — paper outline + open items

## Conventions

- Functions throwing `NotImplementedError` are **planned work** (per `RESEARCH_PLAN.md`),
  not bugs — do not implement them unless the corresponding phase is in scope.
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