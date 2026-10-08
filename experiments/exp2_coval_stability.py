"""Experiment 2 (COVAL / Part II): coupled model/critic stability ablations, EXP 2.1-2.5.

Runs the stabilized-algorithm experiments of ``paper/coval`` Part II: learning-rate ratio
``alpha_model / alpha_critic``, target-network Polyak ``tau``, weight normalization,
weight clipping, and critic spectral normalization, with Lyapunov/Jacobian logging.

Planned work per ``paper/coval/notes/coval_scope.md``; the entry points raise
``NotImplementedError`` until Part II starts.
"""

from __future__ import annotations

import argparse

from .common import load_config


def run_exp21_lr_ratio(cfg: dict, out_dir: str) -> dict:
    raise NotImplementedError


def run_exp22_polyak(cfg: dict, out_dir: str) -> dict:
    raise NotImplementedError


def run_exp23_normalization(cfg: dict, out_dir: str) -> dict:
    raise NotImplementedError


def run_exp24_clipping(cfg: dict, out_dir: str) -> dict:
    raise NotImplementedError


def run_exp25_spectral(cfg: dict, out_dir: str) -> dict:
    raise NotImplementedError


def run(cfg: dict, out_dir: str) -> dict:
    raise NotImplementedError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="runs/exp2_coval_stability")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
