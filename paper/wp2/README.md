# WP2 — Coupled-Stability Theory & Stabilized Algorithm (COVAL)

- **Venue target:** NeurIPS/ICLR (theory + algorithm).
- **Main document:** `main.tex` (stub; not yet written).
- **Scope:** non-asymptotic two-timescale ODE tracking-error analysis and Lyapunov
  drift under constant step sizes for the coupled model/critic system; the stabilized
  losses (detached target-critic, batch self-normalization, weight clipping, curriculum
  annealing); experiments EXP 2.1-2.5 (learning-rate ratio, Polyak target, normalization,
  clipping, spectral normalization) and the Lyapunov/Jacobian logging.
- **Notes / tables:** `notes/`, `tables/` (empty; populated when WP2 starts).
- **Build:** `latexmk -pdf main.tex` from this directory.
