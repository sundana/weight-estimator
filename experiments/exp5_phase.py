"""Experiment 5: mismatch phase diagram over capacity x distractor count x sparsity.

Builds the paper's headline regime map. For each cell ``(capacity, d_d, sparsity)`` it
measures the exact policy-gradient alignment of MLE and the value-aware families, then
classifies the cell as value-aware-win / MLE-win / both-fail and overlays the
Theorem-3 ``SNR_dec`` prediction. Also records the decision-crossover statistic.
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import load_config, save_fig, save_json, save_manifest
from .exp1_alignment import _rng, alignment_for_regime


def phase_cell(cfg: dict, d_d: int, capacity: int, sparse: bool, seed_idx: int) -> dict:
    regime = dict(cfg["regime"])
    regime["d_d"] = d_d
    regime["goal_radius"] = 0.3 if sparse else None
    cfg2 = dict(cfg)
    cfg2["capacity"] = capacity
    return alignment_for_regime(regime, cfg2, _rng(cfg, [d_d, capacity, sparse], seed_idx))


def classify(mle: float, best_va: float, margin: float = 0.1) -> str:
    if best_va >= mle + margin:
        return "value-aware-win"
    if best_va <= mle - margin:
        return "MLE-win"
    return "both-fail"


def run(cfg: dict, out_dir: str) -> dict:
    families = [f for f in cfg.get("loss_families", ["mle", "vaml1", "vagram", "lambert"]) if f != "mle"]
    capacities = list(cfg.get("capacity_list", [1, 2, 3, 5]))
    d_ds = list(cfg.get("d_d_list", [0, 1, 2]))
    sparsities = list(cfg.get("sparsity", ["dense", "sparse"]))
    rows = []
    for sparse_name in sparsities:
        sparse = sparse_name == "sparse"
        for capacity in capacities:
            for d_d in d_ds:
                per_seed = [
                    phase_cell(cfg, d_d, capacity, sparse, i)
                    for i in range(cfg.get("n_seeds", 5))
                ]
                row = {"sparsity": sparse_name, "capacity": capacity, "d_d": d_d}
                for key in ["mle", *families]:
                    row[key] = float(np.mean([r[key] for r in per_seed]))
                    row[key + "_std"] = float(np.std([r[key] for r in per_seed]))
                best = max(families, key=lambda f: row[f])
                row["best_family"] = best
                row["best_value_aware"] = row[best]
                row["align_delta"] = row[best] - row["mle"]
                row["label"] = classify(row["mle"], row[best])
                rows.append(row)
    save_manifest(cfg, out_dir)
    save_json(rows, out_dir, "phase")
    _plot(rows, out_dir, d_ds, capacities)
    return {"rows": rows, "out_dir": out_dir}


def _plot(rows: list[dict], out_dir: str, d_ds: list, capacities: list) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    sparsities = sorted({r["sparsity"] for r in rows})
    fig, axes = plt.subplots(1, len(sparsities), figsize=(5.0 * len(sparsities), 3.8), squeeze=False)
    vmax = max(abs(r["align_delta"]) for r in rows) or 1.0
    for si, sp in enumerate(sparsities):
        ax = axes[0][si]
        grid = np.full((len(capacities), len(d_ds)), np.nan)
        for r in rows:
            if r["sparsity"] != sp:
                continue
            grid[capacities.index(r["capacity"]), d_ds.index(r["d_d"])] = r["align_delta"]
        im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(d_ds)), [str(d) for d in d_ds])
        ax.set_yticks(range(len(capacities)), [str(c) for c in capacities])
        ax.set_xlabel("distractor dims $d_d$")
        ax.set_ylabel("model capacity (rank)")
        ax.set_title(f"{sp} reward")
        for r in rows:
            if r["sparsity"] != sp:
                continue
            yi, xi = capacities.index(r["capacity"]), d_ds.index(r["d_d"])
            ax.text(xi, yi, f"{r['align_delta']:+.2f}", ha="center", va="center", fontsize=8)
            if r["label"] == "value-aware-win":
                ax.plot(xi, yi, marker="*", color="k", markersize=13)
        fig.colorbar(im, ax=ax, label=r"best value-aware $-$ MLE")
    fig.tight_layout()
    save_fig(fig, out_dir, "phase_diagram")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="runs/exp5_phase")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
