# Paper Outline (RESEARCH_PLAN.md Sec. 7)

1. Introduction — objective mismatch, value-aware learning, the missing diagnostic.
2. Background & Related Work — Lambert; VAML/IterVAML; VaGraM; CVAML; MOBILE; decision-aware taxonomy.
3. Theory — Lemma 1 (factorization), Theorem 1 (crossover), Theorem 2 (ESS).
4. Distractor-Gym — design, regime knobs, phase diagram.
5. Gradient-Alignment Analysis (Exp 1).
6. delta_TD Decomposition & Weight-Estimator Ablation (Exp 2).
7. Benchmark — Phase 1 (MBPO/VaGraM) + Phase 2 (DreamerV3/TD-MPC2).
8. Discussion & Conclusion.

## Open items

- Exact continuous base task (default: DMC-style reach + chaotic distractor dims).
- Threshold `tau` calibration for SNR_w across regimes.