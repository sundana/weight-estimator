"""Reproduction of Lambert et al. 2020 Fig 3 objective mismatch.

Modules: config (hyper-parameters), envs (CartPole environments), model
(probabilistic dynamics network), datasets (grid/expert/on-policy protocols),
planner (CEM + model-based evaluation), experiment (Pearson correlation run).
The CLI endpoint lives in the parent directory: reproduce_objective_mismatch.py.
"""