"""Redraw the objective-mismatch figures from saved experiment points.

Regenerates the Fig-3-style LL-vs-reward scatter grid and the rho summary bar
chart from the .npz points (and .json metric ordering) written by
reproduce_objective_mismatch.py -- no model retraining is needed.

Usage:
    python replicate/lambert/plot_objective_mismatch.py [--points PATH] [--output DIR]
"""

import argparse
from pathlib import Path

from objective_mismatch.plotting import make_ll_vs_reward_figure, make_rho_summary_figure
from objective_mismatch.report import load_results

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the objective-mismatch figures from saved points.")
    parser.add_argument("--points", type=Path,
                        default=DEFAULT_OUTPUT / "objective_mismatch_points.npz",
                        help="saved points file (default: outputs/objective_mismatch_points.npz)")
    parser.add_argument("--metrics", type=Path,
                        default=DEFAULT_OUTPUT / "objective_mismatch_metrics.json",
                        help="saved metric summary (default: outputs/objective_mismatch_metrics.json)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="directory for the regenerated figures")
    args = parser.parse_args()

    results = load_results(args.points, metrics_path=args.metrics)
    make_ll_vs_reward_figure(results, args.output / "objective_mismatch_ll_vs_reward.png")
    make_rho_summary_figure(results, args.output / "objective_mismatch_rho_summary.png")
    print(f"figures written to {args.output}")


if __name__ == "__main__":
    main()