# Paper Outline (RESEARCH_PLAN.md Sec. 7)

Draft v1 written: `paper/draft_v1.md` (Phase 1d).

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

## Phase 1c results (continuous Distractor-Gym, exp4)

- Pendulum-v1 + chaotic/white-noise distractor dims, small MLP dynamics model,
  MLE-MSE vs VaGraM projection loss `(grad r . (s_hat' - s'))^2`.
- Pure projection loss (lam=1) is degenerate and fails to train a usable model —
  reproducing VaGraM's documented divergence of naive value-aware losses.
- With a bounded combination (lam=0.5) the model is stable and matches MLE on global
  MSE, while improving the decision-relevant metrics when unpredictable stochastic
  distractors consume capacity: value error 0.214 vs 0.306 (d_d=8) and 0.524 vs 0.674
  (d_d=16); directional alignment 0.971/0.942 vs 0.952/0.905. Benefit grows with
  distractor count and requires sufficient model capacity.

## Open items

- Sharper decision-aware crossover statistic (replace SNR_w proxy).
- Threshold `tau` calibration for Lambert weights.
- Provisioning for the full MBPO/VaGraM benchmark: `mbrl-lib` + `dm_control`/`mujoco`
  are not installed; mbrl-lib pins old `gym` that conflicts with `gymnasium` 1.2.2.
- Phase 2: instrument DreamerV3/TD-MPC2.
- Distractor dynamics classes beyond the logistic map.