"""Experiment 1: gradient alignment (RESEARCH_PLAN.md Sec. 4).

For each loss family (MLE, value-weighted, decision-aligned) and regime, learn the
model from a transition batch, then compare ``g_true`` (policy gradient under true
dynamics) with ``g_model`` (policy gradient under the fitted model) via cosine
similarity.

The target policy is deliberately off-center (suboptimal) so ``g_true`` is
informative. ``coverage`` controls the data distribution: ``uniform`` spreads
samples over the state space, ``goal`` concentrates them where the value gradient
is high.
"""

from __future__ import annotations

import argparse

import numpy as np

from distractor_gym.agents import (
    centered_policy,
    collect_transitions,
    fit_empirical_model,
    induced_value_error,
    policy_gradient,
)
from distractor_gym.diagnostics import gradient_alignment
from distractor_gym.losses import LossFamily, weight

from .common import coverage_behavior, iter_sweep, load_config, make_env, save_fig, save_json


def alignment_for_regime(regime: dict, cfg: dict, rng: np.random.Generator) -> dict:
    env = make_env(regime, seed=regime.get("seed"))
    gamma = cfg.get("gamma", 0.99)
    alpha = cfg.get("alpha", 0.1)
    tau = cfg.get("tau", 0.5)
    n_data = cfg.get("n_data", 8000)
    target_gain = cfg.get("target_gain", 2.0)
    target_center = cfg.get("target_center", 0.0)
    families = [LossFamily(f) for f in cfg.get("loss_families", ["mle"])]

    policy = centered_policy(env, center=target_center, gain=target_gain)
    V_true = env.evaluate_v(env.transition, policy.probs(), gamma)
    g_true = policy_gradient(env, env.transition, policy, gamma)
    g_true_flat = g_true.ravel()

    behavior = coverage_behavior(env, cfg.get("coverage", "uniform"), gain=cfg.get("coverage_gain", 1.5))
    data = collect_transitions(env, n_data, behavior, rng)

    P_mle = fit_empirical_model(env, data, None, alpha)
    V_hat = env.evaluate_v(P_mle, policy.probs(), gamma)
    grad_hat = env.value_grad(V_hat)

    results = {}
    for fam in families:
        if fam is LossFamily.MLE:
            w = None
        else:
            w = weight(
                fam,
                V_s=V_hat[data[:, 0]],
                V_sp=V_hat[data[:, 2]],
                grad_V_sp=grad_hat[data[:, 2]],
                tau=tau,
            )
        P_w = fit_empirical_model(env, data, w, alpha)
        g_model = policy_gradient(env, P_w, policy, gamma)
        results[fam.value] = float(gradient_alignment(g_true_flat, g_model.ravel()))
    results["risk_mle"] = float(induced_value_error(env, P_mle, V_true, data))
    return results


def run(cfg: dict, out_dir: str) -> dict:
    rows = []
    for regime, extra in iter_sweep(cfg):
        cfg2 = dict(cfg)
        cfg2.update(extra)
        per_seed = [alignment_for_regime(regime, cfg2, _rng(cfg, key_for(regime, extra), i)) for i in range(cfg.get("n_seeds", 1))]
        row = {"regime": regime, "coverage": cfg2.get("coverage", "uniform")}
        keys = [k for k in per_seed[0]]
        for k in keys:
            vals = [r[k] for r in per_seed]
            row[k] = float(np.mean(vals))
            row[k + "_std"] = float(np.std(vals))
        rows.append(row)
    save_json(rows, out_dir, "results")
    _plot(rows, out_dir)
    return {"rows": rows, "out_dir": out_dir}


def key_for(regime: dict, extra: dict) -> list:
    return sorted(regime.items()) + sorted(extra.items())


def _rng(cfg: dict, key: list, seed_idx: int) -> np.random.Generator:
    import hashlib

    payload = f"{cfg.get('seed', 0)}|{seed_idx}|{key}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.md5(payload).digest()[:8], "big"))


def _plot(rows: list[dict], out_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fams = [k for k in rows[0] if k in LossFamily._value2member_map_]
    coverages = sorted({r["coverage"] for r in rows})
    radii = sorted({r["regime"].get("goal_radius") for r in rows}, key=lambda r: (r is None, r))
    fig, axes = plt.subplots(len(coverages), len(radii), figsize=(4.5 * len(radii), 3.6 * len(coverages)), squeeze=False)
    for ci, cov in enumerate(coverages):
        for ri, rad in enumerate(radii):
            ax = axes[ci][ri]
            sub = [r for r in rows if r["coverage"] == cov and r["regime"].get("goal_radius") == rad]
            x = [r["regime"]["d_d"] for r in sub]
            for fam in fams:
                y = [r[fam] for r in sub]
                err = [r[fam + "_std"] for r in sub]
                ax.errorbar(x, y, yerr=err, marker="o", label=fam)
            ax.set_xlabel("distractor dims d_d")
            ax.set_ylabel("cos(g_true, g_model)")
            ax.set_title(f"coverage={cov}, radius={rad}")
            ax.axhline(0.0, color="k", lw=0.5)
            ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, "alignment")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="runs/exp1")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())