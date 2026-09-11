"""Persist the objective-mismatch reproduction as metrics and clear figures.

Writes per-model (LL, reward) points to .npz, per-dataset summary metrics to
.json, and renders a Fig-3-style scatter grid plus a rho summary bar chart."""

import json
from pathlib import Path

import numpy as np

from .plotting import make_ll_vs_reward_figure, make_rho_summary_figure


def _metrics(results):
    return {
        name: {
            "rho": float(res["rho"]),
            "mean_ll": float(res["mean_ll"]),
            "mean_reward": float(res["mean_reward"]),
            "M": int(res["M"]),
            "num_models": int(res["ll"].size),
        }
        for name, res in results.items()
    }


def save_results(results, output_dir):
    """Write objective_mismatch_metrics.json, _points.npz and both PNG figures
    into output_dir. Returns the list of written paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = {}
    for name, res in results.items():
        points[f"{name}_ll"] = res["ll"]
        points[f"{name}_reward"] = res["reward"]
    points_path = output_dir / "objective_mismatch_points.npz"
    np.savez(points_path, **points)

    metrics_path = output_dir / "objective_mismatch_metrics.json"
    metrics_path.write_text(json.dumps(_metrics(results), indent=2) + "\n")

    scatter_path = output_dir / "objective_mismatch_ll_vs_reward.png"
    make_ll_vs_reward_figure(results, scatter_path)

    rho_path = output_dir / "objective_mismatch_rho_summary.png"
    make_rho_summary_figure(results, rho_path)

    return [metrics_path, points_path, scatter_path, rho_path]


def load_results(points_path, metrics_path=None):
    """Rebuild the results dict from saved points (and, optionally, the saved
    metric summary that fixes the dataset order and reported rho)."""
    data = np.load(points_path)
    names = sorted({key.rsplit("_", 1)[0] for key in data.files})
    if metrics_path is not None:
        names = sorted(json.loads(Path(metrics_path).read_text()).keys())
    results = {}
    for name in names:
        ll = data[f"{name}_ll"]
        reward = data[f"{name}_reward"]
        rho = float(np.corrcoef(ll, reward)[0, 1]) if ll.size >= 2 else float("nan")
        results[name] = {
            "ll": ll,
            "reward": reward,
            "rho": rho,
            "mean_ll": float(np.mean(ll)),
            "mean_reward": float(np.mean(reward)),
            "M": int(ll.size),
        }
    return results