# Paper Outline (RESEARCH_PLAN.md Sec. 7)

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

## Empirical results (Phase 1a/1b, tabular suite)

- Lemma 1: `|delta_TD| ~ ||grad V|| * eps * |cos phi|` holds with R^2 = 0.99 (dense),
  0.83 (sparse reward); curvature residual is the gap (exp2 part A).
- Theorem 1: MLE dominates the Bellman/prediction risk; oracle weights beat estimated
  weights, isolating weight-estimator noise (exp1b, exp2 part B).
- Decision crossover: value-aware weighting restores policy-gradient alignment under
  goal coverage + sparse reward where MLE misguides the policy (cos ~ 0 or negative);
  value-aware collapses where SNR_w / ESS collapse (uniform coverage, dense reward).
- `SNR_w` is a positive-but-noisy predictor of the alignment delta.

## Open items

- Sharper decision-aware crossover statistic (replace SNR_w proxy).
- Threshold `tau` calibration for Lambert weights.
- Phase 1c: continuous Distractor-Gym + MBPO/VaGraM benchmark.
- Distractor dynamics classes beyond the logistic map.