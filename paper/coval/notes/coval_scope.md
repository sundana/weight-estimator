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
- **Status:** scaffold only. Sections `sec:coval`, `sec:coval-theory`, `sec:coval-alg`,
  `sec:coval-exp` in `main.tex` are placeholders; no theory proofs, no algorithm
  implementation, no `runs/` artifacts, no tables. `NotImplementedError` stubs in
  `src/` mark the stabilization primitives.
