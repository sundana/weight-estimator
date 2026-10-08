"""Experiment 1.1 (deep): policy-gradient cosine alignment (Part I of paper/coval).

Measures objective mismatch between the true environment and a learned model at the
level of the policy gradient. ``g_true`` is the exact rollout gradient of a fixed
policy pi_phi through the real MuJoCo ``Distractor-Gym`` dynamics, differentiated
per integrator step via ``mujoco.mjd_transitionFD`` (see
``distractor_gym.deep.mujoco_diff``); ``g_model`` is the same gradient with the
learned ensemble dynamics substituted. The reported statistic is
``cos(g_true, g_model)`` per model-loss family (MLE, VaGraM, TD-error, calibrated)
and per offline dataset (random, medium-replay SAC).

Scope: the gradient-cosine diagnostic. The H1.1 policy-return claim (>=40% degradation
of MLE) is deferred to the full model-based training loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from distractor_gym.deep.mujoco_diff import (
    MujocoDistractorEnv,
    MujocoDistractorGym,
    rollout_return,
)
from distractor_gym.deep.sac import collect_random, train_sac
from distractor_gym.deep.trainer import fit_dynamics, fit_state_value, model_one_step_mse
from distractor_gym.deep.vjp import state_value_grad_norm
from distractor_gym.losses import LossFamily

from .common import load_config, save_fig, save_json, save_manifest


def _buffer_data(buf) -> dict:
    return buf.arrays()


def _random_data(env: MujocoDistractorEnv, n: int, seed: int, scripted_prob: float = 0.5) -> dict:
    """i.i.d. one-step transitions under a mixture of random and pole-stabilizing actions."""
    gen = torch.Generator().manual_seed(seed + 7)
    s = env.sample_states(n, generator=gen)
    nq = env.physics.nq
    qvel = s[:, nq : env.d_c]
    th = s[:, 1 : nq].sum(dim=-1)
    vth = qvel[:, 1:].sum(dim=-1)
    a_script = torch.clamp(-2.0 * th - 1.0 * vth, -1.0, 1.0)
    a_rand = torch.rand(n, generator=gen, dtype=env.dtype) * 2.0 - 1.0
    mask = (torch.rand(n, generator=gen) < scripted_prob).to(env.dtype)
    a = (mask * a_script + (1.0 - mask) * a_rand).unsqueeze(-1)
    with torch.no_grad():
        s2 = env.step(s, a, noise=env.sigma_dist > 0.0)
        r = env.reward(s, a)
    return {
        "obs": s.numpy().astype(np.float32),
        "act": a.numpy().astype(np.float32),
        "next_obs": s2.numpy().astype(np.float32),
        "rew": r.numpy().astype(np.float32),
        "done": np.zeros(n, dtype=np.float32),
        "terminal": np.zeros(n, dtype=np.float32),
    }


def _torch_stats(fd):
    x = (torch.as_tensor(fd.x_scaler.mean), torch.as_tensor(fd.x_scaler.std))
    y = (torch.as_tensor(fd.y_scaler.mean), torch.as_tensor(fd.y_scaler.std))
    return x, y


def policy_gradient(env, actor, s0, horizon, gamma, fd=None, clip_sigma=6.0) -> torch.Tensor:
    """Flattened ``grad_phi J`` under the true dynamics (``fd=None``) or a fitted model."""
    params = [p for p in actor.parameters() if p.requires_grad]
    model, x_stats, y_stats = None, None, None
    if fd is not None:
        model = fd.model
        x_stats, y_stats = _torch_stats(fd)
    value = rollout_return(
        env,
        actor,
        s0,
        horizon,
        gamma,
        model=model,
        x_stats=x_stats,
        y_stats=y_stats,
        clip_sigma=clip_sigma,
    )
    if not value.requires_grad:
        return torch.cat([torch.zeros_like(p).reshape(-1) for p in params])
    grads = torch.autograd.grad(value, params, allow_unused=True)
    return torch.cat(
        [(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)]
    )


def gradient_cosine(g_true: torch.Tensor, g_model: torch.Tensor) -> float:
    denom = float(g_true.norm() * g_model.norm())
    return float(torch.dot(g_true, g_model) / denom) if denom > 0.0 else 0.0


def _prepare(cfg: dict, d_d: int, sigma: float, seed: int) -> dict:
    """Train the shared SAC agent/replays and the rollout start states for one regime."""
    torch.manual_seed(cfg.get("seed", 0) + seed)
    env = MujocoDistractorEnv(
        d_d=d_d,
        sigma_dist=sigma,
        seed=seed,
        base_task=cfg.get("base_task", "inverted_double_pendulum"),
        eps=cfg.get("mujoco_eps", 1e-6),
        centered=cfg.get("mujoco_centered", True),
    )
    gym_env = MujocoDistractorGym(
        d_d=d_d,
        sigma_dist=sigma,
        seed=seed,
        base_task=cfg.get("base_task", "inverted_double_pendulum"),
        horizon=cfg.get("data_horizon", cfg.get("horizon", 8)),
    )
    agent, medium = train_sac(
        gym_env,
        n_steps=cfg.get("sac_steps", 1500),
        seed=seed,
        start_steps=cfg.get("sac_start_steps", 300),
        update_after=cfg.get("sac_update_after", 300),
        update_every=cfg.get("sac_update_every", 50),
        batch_size=cfg.get("sac_batch_size", 128),
        hidden=cfg.get("hidden", 64),
        n_layers=cfg.get("n_layers", 2),
    )
    random_buf = collect_random(gym_env, n_steps=cfg.get("random_steps", 2000), seed=seed)
    gen = torch.Generator().manual_seed(seed + 999)
    s0 = env.sample_states(cfg.get("n_starts", 64), generator=gen)
    return {
        "env": env,
        "actor": agent.actor,
        "agent": agent,
        "s0": s0,
        "random": random_buf,
        "datasets": {
            "random": _random_data(env, cfg.get("random_steps", 2000), seed),
            "medium": _buffer_data(medium),
        },
    }


def _row(
    seed,
    d_d,
    sigma,
    dataset,
    family,
    horizon,
    arm,
    clip_sigma,
    g_true,
    g_model,
    one_step_mse,
):
    true_norm = float(g_true.norm())
    model_norm = float(g_model.norm())
    return {
        "seed": seed,
        "d_d": d_d,
        "sigma_dist": sigma,
        "dataset": dataset,
        "family": family,
        "horizon": horizon,
        "arm": arm,
        "clip_sigma": clip_sigma,
        "cos": gradient_cosine(g_true, g_model),
        "g_true_norm": true_norm,
        "g_model_norm": model_norm,
        "grad_ratio": model_norm / true_norm if true_norm > 0.0 else None,
        "one_step_mse": one_step_mse,
    }


def _value_fns(cfg: dict, data: dict, gamma: float, seed: int):
    """Return ``(value_fn, grad_norm_fn)`` for the fitted TD(0) state-value head."""
    sv = fit_state_value(
        data,
        hidden=cfg.get("hidden", 64),
        n_layers=cfg.get("n_layers", 2),
        epochs=cfg.get("critic_epochs", 30),
        batch_size=cfg.get("batch_size", 256),
        gamma=gamma,
        seed=seed,
    )
    return (lambda o: sv(o)), (lambda o: state_value_grad_norm(sv, o))  # noqa: E731


def regime_rows(
    cfg: dict,
    d_d: int,
    sigma: float,
    seed: int,
    force_mle: bool = False,
    arm: str = "default",
    horizons: list[int] | None = None,
    clip_sigma: float | None = None,
) -> list[dict]:
    horizons = horizons or [cfg.get("horizon", 8)]
    clip_sigma = cfg.get("clip_sigma", 6.0) if clip_sigma is None else clip_sigma
    gamma = cfg.get("gamma", 0.99)
    prep = _prepare(cfg, d_d, sigma, seed)
    env, actor, s0 = prep["env"], prep["actor"], prep["s0"]
    g_true = {h: policy_gradient(env, actor, s0, h, gamma) for h in horizons}
    if arm == "ceiling":
        return [
            _row(
                seed,
                d_d,
                sigma,
                "true_model",
                "true_dynamics",
                h,
                arm,
                clip_sigma,
                g_true[h],
                g_true[h],
                None,
            )
            for h in horizons
        ]
    families = [
        LossFamily(f)
        for f in cfg.get("loss_families", ["mle", "vagram", "td_error", "calibrated"])
    ]
    rows = []
    for name, data in prep["datasets"].items():
        vfn, gfn = _value_fns(cfg, data, gamma, seed)
        for fam in families:
            fit_fam = LossFamily.MLE if force_mle else fam
            fd = fit_dynamics(
                data,
                fit_fam,
                value_fn=vfn,
                grad_norm_fn=gfn,
                n_models=cfg.get("n_models", 3),
                hidden=cfg.get("hidden", 64),
                n_layers=cfg.get("n_layers", 2),
                epochs=cfg.get("epochs", 30),
                batch_size=cfg.get("batch_size", 256),
                seed=seed,
            )
            mse = model_one_step_mse(fd, data)
            for h in horizons:
                g_model = policy_gradient(env, actor, s0, h, gamma, fd=fd, clip_sigma=clip_sigma)
                rows.append(
                    _row(
                        seed,
                        d_d,
                        sigma,
                        name,
                        fam.value,
                        h,
                        arm,
                        clip_sigma,
                        g_true[h],
                        g_model,
                        mse,
                    )
                )
    return rows


def run(cfg: dict, out_dir: str) -> dict:
    seeds = list(range(cfg.get("n_seeds", 2)))
    d_ds = cfg.get("d_d_list", [0, 10, 50])
    sigmas = cfg.get("sigma_dist_list", [0.0, 0.3])
    save_manifest(cfg, out_dir)

    rows = []
    if cfg.get("run_main", True):
        for d_d in d_ds:
            for sigma in sigmas:
                for seed in seeds:
                    print(f"[exp1_deep] d_d={d_d} sigma={sigma} seed={seed}", flush=True)
                    rows.extend(regime_rows(cfg, d_d, sigma, seed, arm="default"))
        save_json(rows, out_dir, "alignment")

    control = []
    if cfg.get("control_identical_loss", False):
        ctrl_dd = cfg.get("control_d_d", 0)
        for sigma in sigmas:
            for seed in seeds:
                print(f"[exp1_deep] identical-loss d_d={ctrl_dd} sigma={sigma} seed={seed}", flush=True)
                control.extend(regime_rows(cfg, ctrl_dd, sigma, seed, force_mle=True, arm="identical"))
        save_json(control, out_dir, "alignment_control")

    horizon = []
    if cfg.get("horizon_list"):
        for d_d in cfg.get("horizon_d_d_list", d_ds):
            for sigma in cfg.get("horizon_sigma_list", sigmas):
                for seed in seeds:
                    print(f"[exp1_deep] horizon d_d={d_d} sigma={sigma} seed={seed}", flush=True)
                    horizon.extend(
                        regime_rows(cfg, d_d, sigma, seed, arm="horizon", horizons=cfg["horizon_list"])
                    )
        save_json(horizon, out_dir, "alignment_horizon")
        _plot_horizon(horizon, out_dir)

    clip = []
    if cfg.get("clip_sigma_list"):
        for clip_sigma in cfg["clip_sigma_list"]:
            for seed in seeds[: cfg.get("clip_n_seeds", 5)]:
                print(f"[exp1_deep] clip={clip_sigma} seed={seed}", flush=True)
                clip.extend(
                    regime_rows(
                        cfg,
                        cfg.get("clip_d_d", 50),
                        cfg.get("clip_sigma_dist", 0.0),
                        seed,
                        arm="clip",
                        horizons=cfg.get("clip_horizons", [4, 8]),
                        clip_sigma=clip_sigma,
                    )
                )
        save_json(clip, out_dir, "alignment_clip")
        _plot_clip(clip, out_dir)

    ceiling = []
    if cfg.get("ceiling_control", False):
        for sigma in cfg.get("ceiling_sigma_list", [0.0]):
            for seed in seeds:
                print(f"[exp1_deep] ceiling sigma={sigma} seed={seed}", flush=True)
                ceiling.extend(
                    regime_rows(cfg, 0, sigma, seed, arm="ceiling", horizons=[2, 8])
                )
        save_json(ceiling, out_dir, "alignment_ceiling")
        ok = all(abs(r["cos"] - 1.0) < 1e-6 for r in ceiling)
        print(f"[exp1_deep] ceiling cos==1 check: {ok}", flush=True)

    if rows:
        _plot(rows, out_dir)
    return {
        "rows": rows,
        "control": control,
        "horizon": horizon,
        "clip": clip,
        "ceiling": ceiling,
        "out_dir": out_dir,
    }


def _mean_std(xs: list[float]) -> tuple[float, float]:
    """Population mean and standard deviation of a non-empty sample."""
    n = len(xs)
    mean = sum(xs) / n
    std = (sum((x - mean) ** 2 for x in xs) / n) ** 0.5
    return mean, std


def _read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _panel(ax, sub: list[dict], xkey: str, families: list[str]) -> None:
    """Draw mean +/- std error bars over seeds plus faint per-seed trajectories."""
    seeds = sorted({r["seed"] for r in sub})
    for fam in families:
        xs = sorted({r[xkey] for r in sub if r["family"] == fam})
        if not xs:
            continue
        means, stds = [], []
        for x in xs:
            vals = [r["cos"] for r in sub if r["family"] == fam and r[xkey] == x]
            mean, std = _mean_std(vals)
            means.append(mean)
            stds.append(std)
        for seed in seeds:
            pts = sorted(
                [r for r in sub if r["family"] == fam and r["seed"] == seed],
                key=lambda r: r[xkey],
            )
            ax.plot(
                [r[xkey] for r in pts],
                [r["cos"] for r in pts],
                color="0.75",
                lw=0.6,
                alpha=0.4,
                zorder=0,
            )
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=2, label=fam, zorder=2)
    ax.axhline(0.0, color="k", lw=0.5)


def _plot(rows: list[dict], out_dir: str) -> None:
    """Main alignment grid: mean cos vs d_d per family, panels dataset x sigma_dist."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    datasets = sorted({r["dataset"] for r in rows})
    families = sorted({r["family"] for r in rows})
    sigmas = sorted({r["sigma_dist"] for r in rows})
    fig, axes = plt.subplots(
        len(datasets),
        len(sigmas),
        figsize=(4.5 * len(sigmas), 3.6 * len(datasets)),
        squeeze=False,
    )
    for di, ds in enumerate(datasets):
        for si, sig in enumerate(sigmas):
            ax = axes[di][si]
            sub = [r for r in rows if r["dataset"] == ds and r["sigma_dist"] == sig]
            _panel(ax, sub, "d_d", families)
            ax.set_xlabel("distractor dims $d_d$")
            ax.set_ylabel(r"$\cos(g_{\mathrm{true}}, g_{\mathrm{model}})$")
            ax.set_title(f"{ds}, sigma={sig}")
            ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, out_dir, "deep_alignment")


def _plot_horizon(rows: list[dict], out_dir: str) -> None:
    """Alignment vs rollout horizon at sigma_dist=0, panels dataset x d_d."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = [r for r in rows if abs(r["sigma_dist"]) < 1e-9]
    datasets = sorted({r["dataset"] for r in rows})
    d_ds = sorted({r["d_d"] for r in rows})
    families = sorted({r["family"] for r in rows})
    fig, axes = plt.subplots(
        len(datasets),
        len(d_ds),
        figsize=(4.5 * len(d_ds), 3.6 * len(datasets)),
        squeeze=False,
    )
    for di, ds in enumerate(datasets):
        for xi, d in enumerate(d_ds):
            ax = axes[di][xi]
            sub = [r for r in rows if r["dataset"] == ds and r["d_d"] == d]
            _panel(ax, sub, "horizon", families)
            ax.set_xticks(sorted({r["horizon"] for r in sub}))
            ax.set_xlabel("rollout horizon")
            ax.set_ylabel(r"$\cos(g_{\mathrm{true}}, g_{\mathrm{model}})$")
            ax.set_title(f"{ds}, $d_d$={d}")
            ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, out_dir, "deep_alignment_horizon")


def _plot_clip(rows: list[dict], out_dir: str) -> None:
    """Alignment vs prediction clip, panels dataset x horizon at d_d=50, sigma_dist=0."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    datasets = sorted({r["dataset"] for r in rows})
    horizons = sorted({r["horizon"] for r in rows})
    families = sorted({r["family"] for r in rows})
    fig, axes = plt.subplots(
        len(datasets),
        len(horizons),
        figsize=(4.5 * len(horizons), 3.6 * len(datasets)),
        squeeze=False,
    )
    for di, ds in enumerate(datasets):
        for hi, h in enumerate(horizons):
            ax = axes[di][hi]
            sub = [r for r in rows if r["dataset"] == ds and r["horizon"] == h]
            _panel(ax, sub, "clip_sigma", families)
            ax.set_xticks(sorted({r["clip_sigma"] for r in sub}))
            ax.set_xlabel(r"prediction clip $\pm k \sigma$")
            ax.set_ylabel(r"$\cos(g_{\mathrm{true}}, g_{\mathrm{model}})$")
            ax.set_title(f"{ds}, horizon={h}")
            ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, out_dir, "deep_alignment_clip")


def replot(out_dir: str) -> None:
    """Regenerate the three alignment figures from committed JSON without re-running."""
    base = Path(out_dir)
    _plot(_read_json(base / "alignment.json"), out_dir)
    _plot_horizon(_read_json(base / "alignment_horizon.json"), out_dir)
    _plot_clip(_read_json(base / "alignment_clip.json"), out_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="runs/exp1_deep_alignment")
    ap.add_argument(
        "--replot",
        action="store_true",
        help="regenerate figures from committed JSON without re-running the experiment",
    )
    args = ap.parse_args(argv)
    if args.replot:
        replot(args.out_dir)
        return 0
    cfg = load_config(args.config)
    run(cfg, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
