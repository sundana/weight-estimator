"""Experiment 1: gradient alignment.

For each loss family (MLE, value-weighted, decision-aligned) and regime, learn the
model, then compare ``g_true`` (true-dynamics policy gradient) with ``g_model``
(model-induced policy gradient) via cosine similarity over training.
"""

from __future__ import annotations

from distractor_gym.losses import LossFamily


def run(family: LossFamily, config_path: str, out_dir: str) -> dict:
    """Return per-step alignment curves and model-loss attribution."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("run via `python -m experiments.exp1_alignment --config ...`")