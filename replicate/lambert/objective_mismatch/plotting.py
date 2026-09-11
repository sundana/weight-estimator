"""Matplotlib figures for the objective-mismatch reproduction (paper Fig 3)."""

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_ORDER = ["expert", "on-policy", "grid"]
DATASET_COLORS = {"grid": "#1f77b4", "expert": "#2ca02c", "on-policy": "#d62728"}
DATASET_LABELS = {"grid": "Grid", "expert": "Expert", "on-policy": "On-policy"}


def _regression_line(x, y):
    """Best-fit line over LL x reward, or None if degenerate."""
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 2)
    return xs, slope * xs + intercept


def _ordered(results):
    """Results ordered expert -> on-policy -> grid (paper Fig 3 layout)."""
    names = [n for n in PAPER_ORDER if n in results] + [
        n for n in results if n not in PAPER_ORDER
    ]
    return [(n, results[n]) for n in names]


def make_ll_vs_reward_figure(results, output_path):
    """Fig-3-style scatter of validation LL vs mean episode reward, one panel
    per dataset, with the Pearson rho and best-fit line annotated."""
    ordered = _ordered(results)
    fig, axes = plt.subplots(1, len(ordered), figsize=(4.4 * len(ordered), 4.0),
                             sharey=True, constrained_layout=True)
    if len(ordered) == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, ordered):
        x, y = res["ll"], res["reward"]
        ax.scatter(x, y, s=16, alpha=0.55, linewidths=0,
                   color=DATASET_COLORS.get(name, "#1f77b4"), rasterized=True)
        fit = _regression_line(x, y)
        if fit is not None:
            xs, ys = fit
            ax.plot(xs, ys, color="black", lw=1.2, ls="--", alpha=0.7)
        rho = res["rho"]
        label = DATASET_LABELS.get(name, name)
        ax.set_title(f"{label}  ($\\rho={rho:+.2f}$)", fontsize=12)
        ax.set_xlabel("validation log-likelihood")
        ax.grid(alpha=0.3)
        ax.text(0.97, 0.06, f"$M={res['M']}$\nmean reward = {res['mean_reward']:.1f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.7", "alpha": 0.9})
    axes[0].set_ylabel("mean episode reward (10 trials)")
    fig.suptitle("Objective mismatch: validation LL vs episode reward per dataset",
                 fontsize=13, y=1.02)
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_rho_summary_figure(results, output_path):
    """Bar chart of Pearson rho(LL, reward) per dataset, the headline metric."""
    ordered = _ordered(results)
    names = [DATASET_LABELS.get(n, n) for n, _ in ordered]
    rhos = [res["rho"] for _, res in ordered]
    fig, ax = plt.subplots(figsize=(max(3.4, 1.1 * len(names)), 3.8))
    bars = ax.bar(names, rhos, width=0.55,
                  color=[DATASET_COLORS.get(n, "#1f77b4") for n, _ in ordered])
    ax.axhline(0.0, color="0.4", lw=1)
    for bar, rho in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width() / 2, rho,
                f"{rho:+.3f}", ha="center",
                va="bottom" if rho >= 0 else "top", fontsize=10)
    ax.set_ylabel("Pearson $\\rho$(LL, reward)")
    ax.set_title("Objective mismatch metric per dataset", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path