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

## Status

Phase 0 (scaffold). Tabular and continuous Distractor-Gym suites, weighted loss families,
and the diagnostics API are stubbed and documented in `src/distractor_gym/`.