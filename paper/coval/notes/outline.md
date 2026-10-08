# Paper Outline (bundled WP1+WP2 -> `paper/coval/`)

Now aligned to the WP1-WP4 plan in
`C:\Users\USER\Documents\Obsidian Vault\Research Plan` (value-aware loss stabilization
and scalability for MBRL). Part I status and results: `paper/coval/notes/wp1_report.md`.
Generated tables: `paper/coval/tables/` (via `experiments/report.py`).

**Role:** the bundled paper has Part I = the diagnostic suite and negative result (WP1)
and Part II = COVAL theory + stabilized algorithm (WP2). Part I does not claim a new
failure-mode taxonomy (CVAML/VaGraM/ROMI already isolate mechanism-specific failure
modes); it asks whether value-aware weighting can robustly beat MLE once those are
controlled (answer: no) and identifies the weight-estimator bias-variance as the binding
constraint. Part II turns that diagnosis into an algorithm. The bundle is not submittable
until Part II has results. See `wp1_report.md` and `coval_scope.md`.

Historical: `paper/coval/notes/archive/draft_v1.md` (superseded) and the corrected tabular
revision `paper/coval/notes/archive/draft_v2.md` (the pre-WP1 diagnostic-paper narrative,
retained for reference).

1. Introduction — objective mismatch, value-aware learning, prior failure-mode diagnoses,
   the negative result, and the move to stabilization.
2. Background & Related Work — Lambert; VAML/IterVAML; VaGraM; CVAML; ROMI; MOBILE;
   decision-aware taxonomy.
3. Theory (Part I) — Lemma 1 (factorization), Theorem 1 (prediction risk), Theorem 2 (ESS),
   Theorem 3 (decision-risk crossover).
4. Distractor-Gym — design, regime knobs, phase diagram.
5. Part I experiments — gradient alignment (Exp 1), delta_TD decomposition & weight-estimator
   ablation (Exp 2), deep alignment (Exp 1.1), VJP profiling (Exp 1.2).
6. **Part II (COVAL)** — coupled model/critic stability theory, the stabilized algorithm,
   EXP 2.1-2.5. *Scaffold; results pending.*
7. Discussion & Conclusion.

## Theory write-up

Part I statements and proofs: `paper/coval/notes/theory.md`. Part II (COVAL) scope:
`paper/coval/notes/coval_scope.md`.

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
  families): after fixing the `fit_dynamics` input/output normalization bug the null is
  recovered (medium `d_d=0` MLE `+0.92 +/- 0.10`, random `+0.74 +/- 0.37`; horizon-2
  `+0.95`/`+0.99`). MLE degrades `~15-25%` to `d_d=50`; paired within-seed gains
  (`h11_paired.tex`) are within seed noise (calibrated `+0.14`/`+0.15` on random high
  `d_d`, VAML-1 `+0.17` at random `d_d=10`; everything else non-positive), so H1.1 is not
  supported at the gradient level. Random high-`d_d` remains flat-return-limited
  (`h11_quality.tex`). Clip is non-binding (`h11_clip.tex`); the identical-loss and
  ceiling arms confirm determinism (`h11_control.tex`). Details in `wp1_report.md`.

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