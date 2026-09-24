# Paper Outline (RESEARCH_PLAN.md Sec. 7)

Draft v1: `paper/draft_v1.md` (superseded). Corrected tabular revision: `paper/draft_v2.md`
(all numbers from committed `runs/`, 10 seeds).

1. Introduction — objective mismatch, value-aware learning, the missing diagnostic.
2. Background & Related Work — Lambert; VAML/IterVAML; VaGraM; CVAML; MOBILE; decision-aware taxonomy.
3. Theory — Lemma 1 (factorization), Theorem 1 (crossover), Theorem 2 (ESS).
4. Distractor-Gym — design, regime knobs, phase diagram.
5. Gradient-Alignment Analysis (Exp 1).
6. delta_TD Decomposition & Weight-Estimator Ablation (Exp 2).
7. Benchmark — Phase 1 (MBPO/VaGraM) + Phase 2 (DreamerV3/TD-MPC2).
8. Discussion & Conclusion.

## Theory write-up

Full statements and proofs: `paper/notes/theory.md`.

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