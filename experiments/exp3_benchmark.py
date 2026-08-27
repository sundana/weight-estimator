"""Experiment 3: benchmark suite.

Modernizes Lambert et al. (2020) diagnostics for current baselines: reproduces the
one-step likelihood vs return correlation and adds the gradient-alignment,
``delta_TD``-decomposition and weight-estimator diagnostics.

Phase 1: MBPO (MLE), MBPO+VaGraM, native VaGraM.
Phase 2: DreamerV3, TD-MPC2 (instrumented).
"""

from __future__ import annotations


def run_phase1(config_path: str, out_dir: str) -> dict:
    raise NotImplementedError


def run_phase2(config_path: str, out_dir: str) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("run via `python -m experiments.exp3_benchmark --config ...`")