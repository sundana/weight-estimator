"""Experiment 2: decomposition and weight-estimator ablation (RESEARCH_PLAN.md Sec. 5).

Part A tests ``|delta_TD| ~ ||grad V|| * eps_model * |cos phi|`` per transition,
reporting R^2, slope, mean alignment and the curvature residual.

Part B separates the weight estimator from policy effects: oracle weights (true
``V``/``grad V``) vs estimated weights (data ``V_hat``/``grad V_hat``) are compared
on the policy-free value-aware risk and on policy-gradient alignment, together with
the weight-estimator diagnostics (bias, variance, ESS, SNR_w).
"""

from __future__ import annotations

import argparse

import numpy as np

from distractor_gym.agents import (
    centered_policy,
    collect_transitions,
    fit_empirical_model,
    induced_value_error,
    mode_prediction,
    policy_gradient,
)
from distractor_gym.diagnostics import (
    decompose_td_error,
    gradient_alignment,
    weight_estimator_stats,
)
from distractor_gym.losses import LossFamily, weight

from .common import coverage_behavior, iter_sweep, load_config, make_env, save_fig, save_json


def _per_transition_tensors(env, V_true, grad_true, data, s_hat):
    s, a, sp = data[:, 0], data[:, 1], data[:, 2]
    delta = np.abs(V_true[sp] - V_true[s_hat])
    grad_norm = np.linalg.norm(grad_true[sp], axis=1)
    err_vec = np.array([env.coordinates(sh) - env.coordinates(spp) for spp, sh in zip(sp, s_hat)])
    eps = np.linalg.norm(err_vec, axis=1)
    cos_phi = np.zeros(len(data))
    nz = eps > 0
    cos_phi[nz] = np.einsum("ij,ij->i", grad_true[sp][nz], err_vec[nz]) / (grad_norm[nz] * eps[nz])
    return delta, grad_norm, eps, cos_phi


def part_a_regime(regime: dict, cfg: dict, rng: np.random.Generator) -> dict:
    env = make_env(regime, seed=regime.get("seed"))
    gamma = cfg.get("gamma", 0.99)
    alpha = cfg.get("alpha", 0.1)
    n_data = cfg.get("n_data", 8000)
    target_center = cfg.get("target_center", 0.0)

    policy = centered_policy(env, center=target_center, gain=cfg.get("target_gain", 2.0))
    V_true = env.evaluate_v(env.transition, policy.probs(), gamma)
    grad_true = env.value_grad(V_true)
    behavior = coverage_behavior(env, cfg.get("coverage", "goal"), gain=cfg.get("coverage_gain", 1.5))
    data = collect_transitions(env, n_data, behavior, rng)

    P_mle = fit_empirical_model(env, data, None, alpha)
    s_hat = mode_prediction(P_mle)[data[:, 0], data[:, 1]]
    delta, grad_norm, eps, cos_phi = _per_transition_tensors(env, V_true, grad_true, data, s_hat)
    res = decompose_td_error(delta, grad_norm, eps, cos_phi)
    return {
        "r2": res.r2,
        "slope": res.slope,
        "mean_cos_phi": res.mean_cos_phi,
        "curvature_residual": res.curvature_residual,
        "n": res.n,
    }


def part_b_regime(regime: dict, cfg: dict, rng: np.random.Generator) -> dict:
    env = make_env(regime, seed=regime.get("seed"))
    gamma = cfg.get("gamma", 0.99)
    alpha = cfg.get("alpha", 0.1)
    tau = cfg.get("tau", 0.5)
    n_data = cfg.get("n_data", 8000)
    target_center = cfg.get("target_center", 0.0)
    families = [LossFamily(f) for f in cfg.get("loss_families", ["vagram", "vaml1"])]

    policy = centered_policy(env, center=target_center, gain=cfg.get("target_gain", 2.0))
    V_true = env.evaluate_v(env.transition, policy.probs(), gamma)
    grad_true = env.value_grad(V_true)
    g_true = policy_gradient(env, env.transition, policy, gamma).ravel()
    behavior = coverage_behavior(env, cfg.get("coverage", "goal"), gain=cfg.get("coverage_gain", 1.5))
    data = collect_transitions(env, n_data, behavior, rng)

    P_mle = fit_empirical_model(env, data, None, alpha)
    V_hat = env.evaluate_v(P_mle, policy.probs(), gamma)
    grad_hat = env.value_grad(V_hat)

    nll = -np.log(P_mle[data[:, 0], data[:, 1], data[:, 2]])
    out = {}
    for fam in families:
        w_est = weight(fam, V_s=V_hat[data[:, 0]], V_sp=V_hat[data[:, 2]], grad_V_sp=grad_hat[data[:, 2]], tau=tau)
        w_ora = weight(fam, V_s=V_true[data[:, 0]], V_sp=V_true[data[:, 2]], grad_V_sp=grad_true[data[:, 2]], tau=tau)
        P_est = fit_empirical_model(env, data, w_est, alpha)
        P_ora = fit_empirical_model(env, data, w_ora, alpha)
        g_est = policy_gradient(env, P_est, policy, gamma).ravel()
        g_ora = policy_gradient(env, P_ora, policy, gamma).ravel()
        stats = weight_estimator_stats(w_est, w_ora, w_est * nll)
        out[fam.value] = {
            "risk_est": float(induced_value_error(env, P_est, V_true, data)),
            "risk_ora": float(induced_value_error(env, P_ora, V_true, data)),
            "align_est": float(gradient_alignment(g_true, g_est)),
            "align_ora": float(gradient_alignment(g_true, g_ora)),
            "bias": stats.bias,
            "variance": stats.variance,
            "ess": stats.effective_sample_size,
            "snr": stats.signal_to_noise,
        }
    out["risk_mle"] = float(induced_value_error(env, P_mle, V_true, data))
    out["align_mle"] = float(gradient_alignment(g_true, policy_gradient(env, P_mle, policy, gamma).ravel()))
    return out


def run(cfg: dict, out_dir: str, part: str) -> dict:
    fn = {"a": part_a_regime, "b": part_b_regime}[part]
    rows = []
    for regime, extra in iter_sweep(cfg):
        cfg2 = dict(cfg)
        cfg2.update(extra)
        per_seed = [fn(regime, cfg2, _rng(cfg, regime, i)) for i in range(cfg.get("n_seeds", 1))]
        row = {"regime": regime, "coverage": cfg2.get("coverage", "goal")}
        for k in per_seed[0]:
            vals = [_deep_get(r, k) for r in per_seed]
            row[k] = _mean_deep(vals)
        rows.append(row)
    save_json(rows, out_dir, f"part_{part}")
    if part == "a":
        _plot_a(rows, out_dir)
    return {"rows": rows, "out_dir": out_dir}


def _deep_get(d: dict, key: str):
    if isinstance(d.get(key), dict):
        return d[key]
    return d[key]


def _mean_deep(vals: list) -> object:
    if isinstance(vals[0], dict):
        return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    return float(np.mean(vals))


def _rng(cfg: dict, regime: dict, seed_idx: int) -> np.random.Generator:
    import hashlib

    payload = f"{cfg.get('seed', 0)}|{seed_idx}|{sorted(regime.items())}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.md5(payload).digest()[:8], "big"))


def _plot_a(rows: list[dict], out_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    radii = sorted({r["regime"].get("goal_radius") for r in rows}, key=lambda r: (r is None, r))
    fig, axes = plt.subplots(1, len(radii), figsize=(4.5 * len(radii), 3.4), squeeze=False)
    for ri, rad in enumerate(radii):
        ax = axes[0][ri]
        sub = [r for r in rows if r["regime"].get("goal_radius") == rad]
        x = [r["regime"]["d_d"] for r in sub]
        for metric in ("r2", "mean_cos_phi"):
            ax.plot(x, [r[metric] for r in sub], marker="o", label=metric)
        ax.plot(x, [r["curvature_residual"] / (r["curvature_residual"] + 1e-9) for r in sub], marker="s", label="curvature share")
        ax.set_xlabel("distractor dims d_d")
        ax.set_ylabel("decomposition fit")
        ax.set_title(f"radius={rad}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, "decomposition")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="runs/exp2")
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    for part in ("a", "b") if args.part == "both" else [args.part]:
        run(cfg, args.out_dir, part)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())