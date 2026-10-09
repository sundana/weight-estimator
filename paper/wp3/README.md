# WP3 — Scaling, Latent Dynamics & Goal-Conditioned Generalization

- **Venue target:** ICML/TMLR (empirical scalability paper).
- **Main document:** `main.tex` (stub; not yet written).
- **Scope:** standard MuJoCo/DMC and sparse/distractor suites; goal-conditioned RL on
  Meta-World and OGBench with HER/UVFA weight aggregation and gradient-conflict handling;
  latent value-aware extension with stop-gradient (TD-MPC2-style); baselines MBPO, VaGraM,
  Calibrated VAML, TEMPO, TD-MPC2, MR-CRL; `rliable` evaluation (IQM, bootstrap CIs,
  performance profiles).
- **Attribution note (multi-goal interference):** the multi-goal gradient-interference
  diagnosis and the UVFA weight-aggregation remedy are contributions of this work, **not**
  of MR-CRL (Ali et al., RLBrew @ RLC 2025; `ali2025mrcrl`). MR-CRL integrates predictive world-model representations
  into the CRL critic and attributes its partial failures to state-encoder overestimation;
  it neither diagnoses multi-goal gradient interference nor formulates UVFA weights (its
  UVFA citation is background only).
- **Notes / tables:** `notes/`, `tables/` (empty; populated when WP3 starts).
- **Build:** `latexmk -pdf main.tex` from this directory.
