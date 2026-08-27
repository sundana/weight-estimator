"""Experiment 2: decomposition and weight-estimator ablation.

Part A: empirical test of ``|delta_TD| ~ ||grad V|| * eps_model * |cos phi|`` in the
tabular suite (R^2, slope, curvature residual).

Part B: ablation separating the weight estimator from policy effects — oracle vs
estimated weights, fixed model / varying weights, fixed weights / varying policy
learner, and the Theorem-1 crossover check via ESS and SNR_w.
"""

from __future__ import annotations


def run_part_a(config_path: str, out_dir: str) -> dict:
    raise NotImplementedError


def run_part_b(config_path: str, out_dir: str) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("run via `python -m experiments.exp2_decomposition --config ...`")