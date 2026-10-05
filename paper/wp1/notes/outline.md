# Paper Outline

Now aligned to the WP1-WP4 plan in
`C:\Users\USER\Documents\Obsidian Vault\Research Plan` (value-aware loss stabilization
and scalability for MBRL). WP1 status and results: `paper/wp1/notes/wp1_report.md`.
Generated tables: `paper/wp1/tables/` (via `experiments/report.py`).

Historical: `paper/wp1/notes/archive/draft_v1.md` (superseded) and the corrected tabular
revision `paper/wp1/notes/archive/draft_v2.md` (the pre-WP1 diagnostic-paper narrative,
retained for reference).

1. Introduction — objective mismatch, value-aware learning, the missing diagnostic.
2. Background & Related Work — Lambert; VAML/IterVAML; VaGraM; CVAML; MOBILE; decision-aware taxonomy.
3. Theory — Lemma 1 (factorization), Theorem 1 (crossover), Theorem 2 (ESS).
4. Distractor-Gym — design, regime knobs, phase diagram.
5. Gradient-Alignment Analysis (Exp 1).
6. delta_TD Decomposition & Weight-Estimator Ablation (Exp 2).
7. Benchmark — Phase 1 (MBPO/VaGraM) + Phase 2 (DreamerV3/TD-MPC2).
8. Discussion & Conclusion.

## Theory write-up

Full statements and proofs: `paper/wp1/notes/theory.md`.

## Empirical results (Phase 1a/1b, tabular suite — revised)

- Lemma 1 (uncapacitated `runs/exp2`): `|delta_TD| ~ ||grad V|| * eps * |cos phi|`
  holds with R^2 = 0.98 (dense), 0.76-0.81 (sparse); invariant to `d_d` because
  deterministic distractors are fit exactly and `grad_{s_d} V = 0`.
- Lemma 1 (capacity-limited `runs/exp2_capacity`): `mean |cos phi|` falls 0.92 -> 0.46
  and curvature residual rises with `d_d`; R^2 destabilizes (can go negative).
- Theorem 1 (uncapacitated `runs/exp1b`, 48 regimes): estimated-weight Bellman risk
  `>=` MLE in 100% of regimes; corrected penalty is `sigma_w^2 Var(V)/n`.
- Theorem 1 scope (`runs/exp1b_capacity`, 12 regimes): dominance fails under a capacity
  limit (est `>=` MLE in 50% VaGraM / 33% VAML-1).
- Decision crossover: under capacity-limited + stochastic distractors, MLE alignment
  collapses with `d_d` (`cos` 0.86 -> -0.08) and VAML-1/Lambert restore it; VaGraM
  (gradient-only) is unstable (Exp 1, `runs/exp1_capacity`).
- `SNR_w` remains a positive-but-noisy predictor (Pearson ~0.25-0.34).
- Deep policy-gradient alignment (Exp 1.1, `runs/exp1_deep_alignment`, 10 seeds, 6
  families): seed noise dominates (medium `d_d=0` MLE `+0.16 +/- 0.65`); the model is
  underfit at one step (`h11_quality.tex`) and the horizon sweep (`h11_horizon.tex`)
  shows medium `d_d=0` is already misaligned (`+0.12`) at horizon 2. Paired within-seed
  gains (`h11_paired.tex`) are `~0` for VaGraM and negative for VAML-1/Lambert, so H1.1
  is not supported at the gradient level. Clip is non-binding (`h11_clip.tex`); the
  identical-loss and ceiling arms confirm determinism (`h11_control.tex`). Details in
  `wp1_report.md`.

## Phase 1c results (continuous Distractor-Gym, exp4)

**Withdrawn in the tabular revision.** Requires `torch`; the earlier projection loss was
computed in mismatched raw vs standardized coordinates and must be fixed before re-running.


## Open items

- Sharper decision-aware crossover statistic (replace SNR_w proxy).
- Threshold `tau` calibration for Lambert weights.
- Provisioning for the full MBPO/VaGraM benchmark: `mbrl-lib` + `dm_control`/`mujoco`
  are not installed; mbrl-lib pins old `gym` that conflicts with `gymnasium` 1.2.2.
- Phase 2: instrument DreamerV3/TD-MPC2.
- Distractor dynamics classes beyond the logistic map.