"""Experiment 2 (COVAL / Part II): coupled model/critic stability ablations, EXP 2.1-2.5.

Runs the stabilized-algorithm experiments of ``paper/coval`` Part II on the tabular
coupled lab (``distractor_gym.coupled``): the learning-rate ratio
``alpha_model / alpha_critic``, target-network Polyak ``tau``, weight normalization,
weight clipping, and critic spectral (Lipschitz) normalization, with Jacobian and
tracking-error logging. Each factor is swept around a fixed baseline operating point
while the others are held at their baseline values.

Config: ``configs/exp2_coval_stability.yaml``. Output: ``runs/exp2_coval_stability/``.
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


def _baseline_overrides(cfg: dict) -> dict:
    """CoupledConfig overrides held fixed across a one-factor-at-a-time sweep."""
    base = cfg.get("baseline", {})
    return {
        "weight_kind": base.get("weight_kind", "vaml1"),
        "polyak_tau": float(base.get("polyak_tau", 1.0)),
        "normalization": base.get("normalization", "none"),
        "clip_median_ratio": base.get("clip_median_ratio", None),
        "spectral_norm": bool(base.get("spectral_norm", False)),
        "lip_target": base.get("lip_target", None),
    }


def _point(cfg: dict, regime: dict, seed: int, alpha_model: float, **over) -> dict:
    """Run one coupled system and return its stability/tracking diagnostics."""
    env = make_env(regime, seed=int(cfg.get("seed", 0)) + seed)
    data = uniform_transition_data(
        env, int(cfg.get("n_data", 1000)), 10_000 * int(cfg.get("seed", 0)) + seed
    )
    fields = dict(
        gamma=float(cfg.get("gamma", 0.99)),
        alpha_model=float(alpha_model),
        alpha_critic=float(cfg["alpha_critic"]),
        self_norm=bool(cfg.get("self_norm", False)),
        n_steps=int(cfg.get("n_steps", 300)),
        bandwidth=float(cfg.get("bandwidth", 0.5)),
        seed=seed,
    )
    fields.update(over)
    system = TabularCoupledSystem(env, data, CoupledConfig(**fields))
    spectral_radius = system.spectral_radius()
    out = system.rollout()
    error = system.tracking_error(out["states"], k=5)
    finite = error[np.isfinite(error)]
    g = out["g_theta_norm"]
    g_finite = g[np.isfinite(g)]
    lip = out["critic_lip"]
    lip_finite = lip[np.isfinite(lip)]
    return {
        "spectral_radius": spectral_radius,
        "diverged": float(out["diverged"]),
        "tracking_error_final": safe_mean(finite[-20:]) if finite.size else float("nan"),
        "g_theta_norm_final": safe_mean(g_finite[-20:]),
        "g_theta_norm_std": safe_std(g_finite[-20:]),
        "weight_var_mean": safe_mean(out["weight_var"]),
        "critic_lip_final": safe_mean(lip_finite[-20:]),
    }


def _aggregate(cfg: dict, regime: dict, specs: list[dict]) -> list[dict]:
    """Average each one-factor condition over seeds and attach the learning-rate ratio."""
    rows = []
    for spec in specs:
        over = {k: v for k, v in spec.items() if k != "alpha_model"}
        alpha_model = float(spec["alpha_model"])
        per_seed = [
            _point(cfg, regime, seed, alpha_model, **over)
            for seed in range(int(cfg.get("n_seeds", 1)))
        ]
        row = dict(over)
        row["alpha_model"] = alpha_model
        row["ratio"] = alpha_model / float(cfg["alpha_critic"])
        for key in per_seed[0]:
            vals = [r[key] for r in per_seed]
            row[key] = safe_mean(vals)
            row[key + "_std"] = safe_std(vals)
        rows.append(row)
    return rows


def find_kappa(rows: list[dict], tol: float) -> float | None:
    """Smallest learning-rate ratio with a spectral radius above ``1 + tol`` (instability)."""
    for row in sorted(rows, key=lambda r: r["ratio"]):
        if row["spectral_radius"] > 1.0 + tol or row["diverged"] >= 0.5:
            return float(row["ratio"])
    return None


def _alpha_model_at_baseline(cfg: dict) -> float:
    ratio = float(cfg.get("baseline", {}).get("alpha_ratio", 5.0))
    return ratio * float(cfg["alpha_critic"])


def run_exp21_lr_ratio(cfg: dict, out_dir: str) -> dict:
    """EXP 2.1: stability boundary over ``alpha_model / alpha_critic``."""
    base = _baseline_overrides(cfg)
    specs = [
        dict(alpha_model=float(ratio) * float(cfg["alpha_critic"]), **base)
        for ratio in cfg["exp21_lr_ratio"]
    ]
    rows = _aggregate(cfg, cfg["regime"], specs)
    return {"rows": rows, "kappa_ratio": find_kappa(rows, float(cfg.get("stability_tol", 1e-3)))}


def run_exp22_polyak(cfg: dict, out_dir: str) -> dict:
    """EXP 2.2: target-network Polyak ``tau`` against gradient oscillation."""
    base = _baseline_overrides(cfg)
    alpha_model = _alpha_model_at_baseline(cfg)
    specs = []
    for tau in cfg["exp22_polyak_tau"]:
        spec = dict(alpha_model=alpha_model, **base)
        spec["polyak_tau"] = float(tau)
        specs.append(spec)
    return {"rows": _aggregate(cfg, cfg["regime"], specs)}


def run_exp23_normalization(cfg: dict, out_dir: str) -> dict:
    """EXP 2.3: weight normalization (none / min-max / batch self-normalization)."""
    base = _baseline_overrides(cfg)
    alpha_model = _alpha_model_at_baseline(cfg)
    specs = []
    for mode in cfg["exp23_normalization"]:
        spec = dict(alpha_model=alpha_model, **base)
        spec["normalization"] = mode
        specs.append(spec)
    return {"rows": _aggregate(cfg, cfg["regime"], specs)}


def run_exp24_clipping(cfg: dict, out_dir: str) -> dict:
    """EXP 2.4: weight clipping at a multiple of the batch-median weight."""
    base = _baseline_overrides(cfg)
    alpha_model = _alpha_model_at_baseline(cfg)
    specs = []
    for ratio in cfg["exp24_clip_median_ratio"]:
        spec = dict(alpha_model=alpha_model, **base)
        spec["clip_median_ratio"] = ratio
        specs.append(spec)
    return {"rows": _aggregate(cfg, cfg["regime"], specs)}


def run_exp25_spectral(cfg: dict, out_dir: str) -> dict:
    """EXP 2.5: critic spectral (Lipschitz) normalization on vs off."""
    base = _baseline_overrides(cfg)
    alpha_model = _alpha_model_at_baseline(cfg)
    specs = []
    for on in cfg["exp25_spectral"]:
        spec = dict(alpha_model=alpha_model, **base)
        spec["spectral_norm"] = bool(on)
        specs.append(spec)
    return {"rows": _aggregate(cfg, cfg["regime"], specs)}


def run(cfg: dict, out_dir: str) -> dict:
    results = {
        "exp21": run_exp21_lr_ratio(cfg, out_dir),
        "exp22": run_exp22_polyak(cfg, out_dir),
        "exp23": run_exp23_normalization(cfg, out_dir),
        "exp24": run_exp24_clipping(cfg, out_dir),
        "exp25": run_exp25_spectral(cfg, out_dir),
    }
    save_manifest(cfg, out_dir)
    save_json(results, out_dir, "stability")
    _plot(results, cfg, out_dir)
    return results


def _plot(results: dict, cfg: dict, out_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0][0]
    rows = sorted(results["exp21"]["rows"], key=lambda r: r["ratio"])
    ax.plot([r["ratio"] for r in rows], [r["spectral_radius"] for r in rows], marker="o")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha_{model}/\alpha_{critic}$")
    ax.set_ylabel("spectral radius")
    ax.set_title("EXP 2.1: LR ratio")

    ax = axes[0][1]
    rows = sorted(results["exp22"]["rows"], key=lambda r: r["polyak_tau"], reverse=True)
    ax.plot([r["polyak_tau"] for r in rows], [r["tracking_error_final"] for r in rows], marker="s")
    ax.set_xscale("log")
    ax.set_xlabel(r"Polyak $\tau$")
    ax.set_ylabel("tracking error")
    ax.set_title("EXP 2.2: target network")

    ax = axes[0][2]
    rows = results["exp23"]["rows"]
    ax.bar([str(r["normalization"]) for r in rows], [r["weight_var_mean"] for r in rows])
    ax.set_ylabel(r"Var($\bar{w}$)")
    ax.set_title("EXP 2.3: normalization")

    ax = axes[1][0]
    rows = results["exp24"]["rows"]
    labels = ["none" if r["clip_median_ratio"] is None else f"{r['clip_median_ratio']:g}x" for r in rows]
    ax.bar(labels, [r["diverged"] for r in rows])
    ax.set_ylabel("divergence rate")
    ax.set_title("EXP 2.4: clipping")

    ax = axes[1][1]
    rows = results["exp25"]["rows"]
    labels = ["on" if r["spectral_norm"] else "off" for r in rows]
    ax.bar(labels, [r["critic_lip_final"] for r in rows])
    ax.set_ylabel(r"$||V_\phi||_{Lip}$")
    ax.set_title("EXP 2.5: spectral norm")

    ax = axes[1][2]
    ax.axis("off")
    kappa = results["exp21"]["kappa_ratio"]
    ax.text(0.0, 0.5, f"kappa (ratio) = {kappa}", fontsize=12)
    fig.tight_layout()
    save_fig(fig, out_dir, "stability")


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
