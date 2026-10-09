# COVAL scope (Part II of the bundled paper)

Migrated from the retired `paper/wp2/README.md`. The WP1+WP2 papers are now bundled as
`paper/coval/main.tex`: Part I = objective-mismatch diagnostics (WP1), Part II = COVAL
theory + stabilized algorithm (WP2).

- **Venue target:** ICML 2026 (bundled with Part I; supersedes WP2's earlier
  NeurIPS/ICLR target).
- **Scope:** non-asymptotic two-timescale ODE tracking-error analysis and Lyapunov drift
  under constant step sizes for the coupled model/critic system; the stabilized losses
  (detached target-critic, batch self-normalization, weight clipping, curriculum
  annealing); experiments EXP 2.1-2.5 (learning-rate ratio, Polyak target, normalization,
  clipping, spectral normalization) and Lyapunov/Jacobian logging.
- **Motivation:** Part I shows value-aware weighting does not robustly beat MLE and that
  the binding constraint is the bias-variance of the estimated weight `w_hat`
  (`paper/coval/notes/theory.md`, Thm 1-2); COVAL bounds that estimator noise.
- **Novelty boundary vs ROMBRL:** ROMBRL (Chen et al., ICML 2026; `chen2026rombri`)
  already gives the first formal coupled-convergence result in MBRL (Stackelberg game +
  two-timescale, `eta_model >> eta_policy`, model follower on the fast timescale) and
  documents the equal-rate collapse (score `~3.3`). COVAL does **not** claim the first
  MBRL coupled-convergence theory; it targets the coupling induced by the value-aware
  weight estimator (`w_hat = w(V_hat, grad V_hat)`), a non-asymptotic constant-step-size
  bound tied to `sigma_w^2`, and deliberately places the *critic* (weight provider) on the
  faster timescale so the model's weighted objective is near-stationary.
- **Status:** partial implementation. The stabilization primitives and the experimental
  harness now exist, but the theory proofs and the full-grid results do not:
  - Phase A (tabular coupled lab): `src/distractor_gym/coupled.py`,
    `experiments/exp2_coupled_lab.py`, `configs/exp2_coupled_lab.yaml` — stability
    boundary `kappa` per weight family and `kappa*(sigma_w)`; smoke run in
    `runs/exp2_coupled_lab/`; table `paper/coval/tables/h20_coupled.tex`.
  - Phase B (EXP 2.1-2.5 ablations): `experiments/exp2_coval_stability.py`,
    `configs/exp2_coval_stability.yaml` — LR ratio, Polyak target, normalization,
    clipping, spectral/Lipschitz; smoke run in `runs/exp2_coval_stability/`; table
    `paper/coval/tables/h21_stability.tex`.
  - Phase C (deep online loop): `src/distractor_gym/deep/coupled_loop.py`,
    `experiments/exp2_coval_loop.py`, `configs/exp2_coval_loop.yaml` — smoke run in
    `runs/exp2_coval_loop/` (return-level harness; noisy at smoke scale).
  - Still open: the non-asymptotic tracking-error/Lyapunov proofs, the full-grid deep
    runs (more iterations/seeds) needed for the return-level H1.1 and COVAL-vs-MLE
    claims, and the `\input` of the tables into `main.tex` Part II.

