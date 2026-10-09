"""Phase A: tabular coupled model/critic lab (Part II of ``paper/coval``).

Locates the stability boundary ``kappa`` of the coupled two-timescale system in
``distractor_gym.coupled`` as a function of the learning-rate ratio ``alpha_model /
alpha_critic`` (Part A1), and traces how that boundary moves with the weight-estimator
noise ``sigma_w`` (Part A2). Uses the exact finite-difference Jacobian of the update map
and the empirical tracking error / Lyapunov series.

Config: ``configs/exp2_coupled_lab.yaml``. Output: ``runs/exp2_coupled_lab/``.
"""

from __future__ import annotations

import argparse

import numpy as np

from distractor_gym.coupled import CoupledConfig, TabularCoupledSystem

from .common import (
    load_config,
    make_env,
    safe_mean,
    safe_std,
    save_fig,
    save_json,
    save_manifest,
    uniform_transition_data,
)


def make_data(env, cfg: dict, seed: int) -> np.ndarray:
    """i.i.d. uniform-coverage transition batch for the coupled lab."""
    return uniform_transition_data(env, int(cfg.get("n_data", 4000)), 10_000 * cfg.get("seed", 0) + seed)


def run_condition(
    cfg: dict,
    regime: dict,
    *,
    seed: int,
    weight_kind: str,
    alpha_model: float,
    weight_noise: float,
) -> dict:
    """Run one coupled system and return its stability and tracking diagnostics."""
    env = make_env(regime, seed=cfg.get("seed", 0) + seed)
    data = make_data(env, cfg, seed)
    cc = CoupledConfig(
        gamma=float(cfg.get("gamma", 0.99)),
        alpha_model=float(alpha_model),
        alpha_critic=float(cfg["alpha_critic"]),
        polyak_tau=float(cfg.get("polyak_tau", 1.0)),
        weight_kind=weight_kind,
        weight_noise=float(weight_noise),
        self_norm=bool(cfg.get("self_norm", False)),
        bandwidth=float(cfg.get("bandwidth", 0.5)),
        n_steps=int(cfg.get("n_steps", 400)),
        seed=seed,
    )
    system = TabularCoupledSystem(env, data, cc)
    spectral_radius = system.spectral_radius()
    out = system.rollout()
    error = system.tracking_error(out["states"], k=5)
    finite_error = error[np.isfinite(error)]
    return {
        "spectral_radius": spectral_radius,
        "diverged": float(out["diverged"]),
        "tracking_error_final": safe_mean(finite_error[-20:]) if finite_error.size else float("nan"),
        "tracking_error_max": float(np.max(finite_error)) if finite_error.size else float("nan"),
        "weight_var_mean": safe_mean(out["weight_var"]),
        "g_theta_norm_final": safe_mean(out["g_theta_norm"][-20:]),
    }


def _aggregate(cfg: dict, regime: dict, rows_spec: list[dict]) -> list[dict]:
    """Average each condition over seeds and attach the learning-rate ratio."""
    rows = []
    for spec in rows_spec:
        per_seed = [run_condition(cfg, regime, seed=s, **spec) for s in range(int(cfg.get("n_seeds", 1)))]
        row = dict(spec)
        row["ratio"] = float(spec["alpha_model"]) / float(cfg["alpha_critic"])
        for key in per_seed[0]:
            vals = [r[key] for r in per_seed]
            row[key] = safe_mean(vals)
            row[key + "_std"] = safe_std(vals)
        rows.append(row)
    return rows


def find_kappa(
    rows: list[dict], predicate, key: str = "alpha_model", tol: float = 1e-3
) -> float | None:
    """Smallest step/ratio with a spectral radius above ``1 + tol`` (linear instability).

    The tolerance excludes the benign marginal (unit) modes that the value-aware coupling
    introduces under self-normalization; those are neutral, not divergent.
    """
    selected = sorted((r for r in rows if predicate(r)), key=lambda r: r[key])
    for row in selected:
        if row["spectral_radius"] > 1.0 + tol or row["diverged"] >= 0.5:
            return float(row[key])
    return None


def find_kappa_empirical(
    rows: list[dict], base_te: dict[float, float], factor: float, key: str = "alpha_model"
) -> float | None:
    """Smallest step whose empirical tracking error inflates past ``factor`` vs noise-free.

    Treats a reference (noise-free) divergence, an own divergence, or a tracking-error
    inflation above ``factor`` as the operational stability boundary ``kappa``.
    """
    for row in sorted(rows, key=lambda r: r[key]):
        base = base_te.get(float(row[key]), float("nan"))
        inflated = (
            np.isfinite(row["tracking_error_final"])
            and np.isfinite(base)
            and base > 0.0
            and row["tracking_error_final"] > factor * base
        )
        if row["diverged"] >= 0.5 or not np.isfinite(base) or not np.isfinite(
            row["tracking_error_final"]
        ) or inflated:
            return float(row[key])
    return None


def run(cfg: dict, out_dir: str) -> dict:
    regime = cfg["regime"]
    alpha_levels = [float(a) for a in cfg["alpha_model_levels"]]

    part_a_spec = [
        {"weight_kind": kind, "alpha_model": a, "weight_noise": 0.0}
        for kind in cfg["weight_kinds"]
        for a in alpha_levels
    ]
    rows_a = _aggregate(cfg, regime, part_a_spec)

    noise_kind = cfg.get("noise_weight_kind", "vaml1")
    part_b_spec = [
        {"weight_kind": noise_kind, "alpha_model": a, "weight_noise": float(sigma)}
        for sigma in cfg["weight_noise_levels"]
        for a in alpha_levels
    ]
    rows_b = _aggregate(cfg, regime, part_b_spec)

    kappa_a = {
        kind: find_kappa(
            rows_a, lambda r, k=kind: r["weight_kind"] == k, tol=float(cfg.get("stability_tol", 1e-3))
        )
        for kind in cfg["weight_kinds"]
    }
    base_te = {
        float(r["alpha_model"]): r["tracking_error_final"]
        for r in rows_a
        if r["weight_kind"] == noise_kind
    }
    factor = float(cfg.get("noise_inflation_factor", 1.25))
    kappa_b = {}
    for sigma in cfg["weight_noise_levels"]:
        selected = sorted(
            [r for r in rows_b if r["weight_noise"] == float(sigma)], key=lambda r: r["alpha_model"]
        )
        kappa_b[float(sigma)] = find_kappa_empirical(selected, base_te, factor)

    save_manifest(cfg, out_dir)
    save_json(
        {
            "part_a": rows_a,
            "part_b": rows_b,
            "kappa_a": kappa_a,
            "kappa_b": {str(k): v for k, v in kappa_b.items()},
        },
        out_dir,
        "coupled_lab",
    )
    _plot(rows_a, rows_b, kappa_b, cfg, out_dir)
    return {"rows_a": rows_a, "rows_b": rows_b, "kappa_a": kappa_a, "kappa_b": kappa_b}


def _plot(rows_a: list[dict], rows_b: list[dict], kappa_b: dict, cfg: dict, out_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    for kind in cfg["weight_kinds"]:
        sel = sorted((r for r in rows_a if r["weight_kind"] == kind), key=lambda r: r["alpha_model"])
        ax0.plot(
            [r["alpha_model"] for r in sel],
            [r["spectral_radius"] for r in sel],
            marker="o",
            label=kind,
        )
    ax0.axhline(1.0, color="k", lw=0.8, ls="--")
    ax0.set_xscale("log")
    ax0.set_xlabel(r"$\alpha_{model}$")
    ax0.set_ylabel("spectral radius of update Jacobian")
    ax0.set_title("A1: linear stability")
    ax0.legend(fontsize=8)

    sigmas = sorted({r["weight_noise"] for r in rows_b})
    y = [np.nan if kappa_b.get(sigma) is None else kappa_b[sigma] for sigma in sigmas]
    ax1.plot(sigmas, y, marker="s", color="C3")
    ax1.set_xlabel(r"weight-estimator noise $\sigma_w$")
    ax1.set_ylabel(r"$\kappa$ ($\alpha_{model}$ boundary)")
    ax1.set_title("A2: boundary vs weight-estimator noise")
    fig.tight_layout()
    save_fig(fig, out_dir, "coupled_lab")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="runs/exp2_coupled_lab")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
