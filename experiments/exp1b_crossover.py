"""Experiment 1b: crossover validation (RESEARCH_PLAN.md Sec. 2.3, 5).

Tests the two halves of the weight-estimator theory:

1. Prediction (Bellman) risk ``R_b(P_hat) = E[(E_{P*}[V] - E_{P_hat}[V])^2]``:
   Theorem 1 predicts weighting never helps the conditional-mean value prediction
   (variance penalty ``sigma_w^2 E[V^2] / n``), so ``Delta_b = R_b(MLE) - R_b(w) <= 0``.

2. Decision (policy-gradient alignment) risk ``Delta_a = cos(g_true, g_w) -
   cos(g_true, g_mle)``: value-aware weighting should help when the weight signal is
   informative (high ``SNR_w``) and hurt when weights are noisy/degenerate.
"""

from __future__ import annotations

import argparse

import numpy as np

from distractor_gym.agents import (
    centered_policy,
    policy_gradient,
    sample_transitions,
)
from distractor_gym.diagnostics import gradient_alignment, weight_signal_to_noise
from distractor_gym.losses import LossFamily, weight

from .common import (
    coverage_behavior,
    coverage_state_probs,
    fit_model,
    iter_sweep,
    load_config,
    make_env,
    save_fig,
    save_json,
    save_manifest,
)


def bellman_risk(env, P_hat, V, data) -> float:
    E_Pstar = np.tensordot(env.transition, V, axes=([2], [0]))
    E_hat = np.tensordot(P_hat, V, axes=([2], [0]))
    s, a = data[:, 0], data[:, 1]
    return float(np.mean((E_Pstar[s, a] - E_hat[s, a]) ** 2))


def regime_row(regime: dict, cfg: dict, rng: np.random.Generator) -> dict:
    env = make_env(regime, seed=regime.get("seed"))
    gamma = cfg.get("gamma", 0.99)
    tau = cfg.get("tau", 0.5)
    n_data = cfg.get("n_data", 8000)
    target_center = cfg.get("target_center", 0.0)
    families = [LossFamily(f) for f in cfg.get("loss_families", ["vagram"])]

    policy = centered_policy(env, center=target_center, gain=cfg.get("target_gain", 2.0))
    V_true = env.evaluate_v(env.transition, policy.probs(), gamma)
    grad_true = env.value_grad(V_true)
    g_true = policy_gradient(env, env.transition, policy, gamma).ravel()
    behavior = coverage_behavior(env, cfg.get("coverage", "goal"), gain=cfg.get("coverage_gain", 1.5))
    state_probs = coverage_state_probs(env, cfg.get("coverage", "goal"))
    data = sample_transitions(env, n_data, behavior, rng, state_probs=state_probs)

    P_mle = fit_model(env, data, None, cfg)
    r_b_mle = bellman_risk(env, P_mle, V_true, data)
    align_mle = float(gradient_alignment(g_true, policy_gradient(env, P_mle, policy, gamma).ravel()))
    nll = -np.log(P_mle[data[:, 0], data[:, 1], data[:, 2]])

    V_hat = env.evaluate_v(P_mle, policy.probs(), gamma)
    grad_hat = env.value_grad(V_hat)
    delta_mle = np.abs(V_true[data[:, 2]] - np.tensordot(P_mle, V_true, axes=([2], [0]))[data[:, 0], data[:, 1]])

    row = {"r_b_mle": r_b_mle, "align_mle": align_mle}
    for fam in families:
        w_est = weight(fam, V_s=V_hat[data[:, 0]], V_sp=V_hat[data[:, 2]], grad_V_sp=grad_hat[data[:, 2]], tau=tau)
        w_ora = weight(fam, V_s=V_true[data[:, 0]], V_sp=V_true[data[:, 2]], grad_V_sp=grad_true[data[:, 2]], tau=tau)
        P_est = fit_model(env, data, w_est, cfg)
        P_ora = fit_model(env, data, w_ora, cfg)
        g_est = policy_gradient(env, P_est, policy, gamma).ravel()
        g_ora = policy_gradient(env, P_ora, policy, gamma).ravel()
        ess = float(np.sum(w_est) ** 2 / max(np.sum(w_est**2), 1e-12))
        row[f"r_b_{fam.value}_est"] = bellman_risk(env, P_est, V_true, data)
        row[f"r_b_{fam.value}_ora"] = bellman_risk(env, P_ora, V_true, data)
        row[f"align_{fam.value}_est"] = float(gradient_alignment(g_true, g_est))
        row[f"align_{fam.value}_ora"] = float(gradient_alignment(g_true, g_ora))
        row[f"snr_{fam.value}"] = weight_signal_to_noise(w_est * nll, w_est)
        row[f"ess_{fam.value}"] = ess
        row[f"varw_{fam.value}"] = float(np.var(w_est))
        row[f"corr_w_vmse_{fam.value}"] = float(np.corrcoef(w_est, delta_mle)[0, 1])
    return row


def run(cfg: dict, out_dir: str) -> dict:
    rows = []
    for regime, extra in iter_sweep(cfg):
        cfg2 = dict(cfg)
        cfg2.update(extra)
        per_seed = [regime_row(regime, cfg2, _rng(cfg, regime, i)) for i in range(cfg.get("n_seeds", 1))]
        row = {"regime": regime, "coverage": cfg2.get("coverage", "goal"), **extra}
        keys = [k for k in per_seed[0]]
        for k in keys:
            vals = [r[k] for r in per_seed]
            row[k] = float(np.mean(vals))
            row[k + "_std"] = float(np.std(vals))
        rows.append(row)
    save_manifest(cfg, out_dir)
    save_json(rows, out_dir, "crossover")
    _plot(rows, out_dir)
    return {"rows": rows, "out_dir": out_dir}


def _rng(cfg: dict, regime: dict, seed_idx: int) -> np.random.Generator:
    import hashlib

    payload = f"{cfg.get('seed', 0)}|{seed_idx}|{sorted(regime.items())}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.md5(payload).digest()[:8], "big"))


def _plot(rows: list[dict], out_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fams = [k.split("_")[1] for k in rows[0] if k.startswith("snr_")]
    fig, axes = plt.subplots(2, len(fams), figsize=(5 * len(fams), 6.6), squeeze=False)
    for fi, fam in enumerate(fams):
        x = [r[f"snr_{fam}"] for r in rows]
        ax = axes[0][fi]
        y_est = [r[f"r_b_{fam}_est"] - r["r_b_mle"] for r in rows]
        y_ora = [r[f"r_b_{fam}_ora"] - r["r_b_mle"] for r in rows]
        ax.scatter(x, y_est, label="estimated weights", marker="o")
        ax.scatter(x, y_ora, label="oracle weights", marker="x")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("SNR_w")
        ax.set_ylabel("R_b(value-aware) - R_b(MLE)")
        ax.set_title(f"{fam}: Bellman risk (predict.)")
        ax.legend(fontsize=8)
        ax = axes[1][fi]
        y_est = [r[f"align_{fam}_est"] - r["align_mle"] for r in rows]
        y_ora = [r[f"align_{fam}_ora"] - r["align_mle"] for r in rows]
        ax.scatter(x, y_est, label="estimated weights", marker="o")
        ax.scatter(x, y_ora, label="oracle weights", marker="x")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("SNR_w")
        ax.set_ylabel("cos(g_w) - cos(g_mle)")
        ax.set_title(f"{fam}: policy-gradient alignment (decision)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, "crossover")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="runs/exp1b")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())