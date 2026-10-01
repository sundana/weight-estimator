# WP3 — Scaling, Latent Dynamics & Goal-Conditioned Generalization

- **Venue target:** ICML/TMLR (empirical scalability paper).
- **Main document:** `main.tex` (stub; not yet written).
- **Scope:** standard MuJoCo/DMC and sparse/distractor suites; goal-conditioned RL on
  Meta-World and OGBench with HER/UVFA weight aggregation and gradient-conflict handling;
  latent value-aware extension with stop-gradient (TD-MPC2-style); baselines MBPO, VaGraM,
  Calibrated VAML, TEMPO, TD-MPC2; `rliable` evaluation (IQM, bootstrap CIs, performance
  profiles).
- **Notes / tables:** `notes/`, `tables/` (empty; populated when WP3 starts).
- **Build:** `latexmk -pdf main.tex` from this directory.
