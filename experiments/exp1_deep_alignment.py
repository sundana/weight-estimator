"""Experiment 1.1 (deep): policy-gradient cosine alignment (WP1).

Measures objective mismatch between the true environment and a learned model at the
level of the policy gradient. Using the differentiable analytic Distractor-Gym,
``g_true`` is the exact rollout gradient of a fixed policy pi_phi under the true
dynamics, and ``g_model`` is the same gradient with the learned ensemble dynamics
substituted. The reported statistic is ``cos(g_true, g_model)`` per model-loss family
(MLE, VaGraM, TD-error, calibrated) and per offline dataset (random, medium-replay SAC).

Scope: the gradient-cosine diagnostic. The H1.1 policy-return claim (>=40% degradation
of MLE) is deferred to the full model-based training loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from distractor_gym.deep.analytic import AnalyticDistractorEnv, AnalyticDistractorGym, rollout_return
from distractor_gym.deep.sac import collect_random, train_sac
from distractor_gym.deep.trainer import fit_dynamics, fit_state_value
from distractor_gym.deep.vjp import state_value_grad_norm
from distractor_gym.losses import LossFamily

from .common import load_config, save_fig, save_json, save_manifest


def _buffer_data(buf) -> dict:
    return buf.arrays()


def _random_data(env: AnalyticDistractorEnv, n: int, seed: int, scripted_prob: float = 0.5) -> dict:
    """i.i.d. one-step transitions under a mixture of random and goal-directed actions."""
    gen = torch.Generator().manual_seed(seed + 7)
    s = env.sample_states(n, generator=gen)
    x, v = s[:, 0], s[:, 1]
    a_script = torch.clamp(3.0 * (env.goal - x) - 1.0 * v, -1.0, 1.0)
    a_rand = torch.rand(n, generator=gen, dtype=torch.float32) * 2.0 - 1.0
    mask = (torch.rand(n, generator=gen) < scripted_prob).float()
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
    }


def _torch_stats(fd):
    x = (torch.as_tensor(fd.x_scaler.mean), torch.as_tensor(fd.x_scaler.std))
    y = (torch.as_tensor(fd.y_scaler.mean), torch.as_tensor(fd.y_scaler.std))
    return x, y


def policy_gradient(env, actor, s0, horizon, gamma, fd=None) -> torch.Tensor:
    """Flattened ``grad_phi J`` under the true dynamics (``fd=None``) or a fitted model."""
    params = [p for p in actor.parameters() if p.requires_grad]
    model, x_stats, y_stats = None, None, None
    if fd is not None:
        model = fd.model
        x_stats, y_stats = _torch_stats(fd)
    value = rollout_return(
        env, actor, s0, horizon, gamma, model=model, x_stats=x_stats, y_stats=y_stats
    )
    grads = torch.autograd.grad(value, params, allow_unused=True)
    return torch.cat(
        [(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)]
    )


def gradient_cosine(g_true: torch.Tensor, g_model: torch.Tensor) -> float:
    denom = float(g_true.norm() * g_model.norm())
    return float(torch.dot(g_true, g_model) / denom) if denom > 0.0 else 0.0


def regime_rows(cfg: dict, d_d: int, sigma: float, seed: int) -> list[dict]:
    torch.manual_seed(cfg.get("seed", 0) + seed)
    env = AnalyticDistractorEnv(d_d=d_d, sigma_dist=sigma, seed=seed)
    gym_env = AnalyticDistractorGym(d_d=d_d, sigma_dist=sigma, seed=seed, horizon=cfg.get("horizon", 8))
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
    actor = agent.actor

    gen = torch.Generator().manual_seed(seed + 999)
    s0 = env.sample_states(cfg.get("n_starts", 64), generator=gen)
    gamma = cfg.get("gamma", 0.99)
    horizon = cfg.get("horizon", 8)
    g_true = policy_gradient(env, actor, s0, horizon, gamma)

    datasets = {"random": _random_data(env, cfg.get("random_steps", 2000), seed), "medium": _buffer_data(medium)}
    families = [LossFamily(f) for f in cfg.get("loss_families", ["mle", "vagram", "td_error", "calibrated"])]
    rows = []
    for name, data in datasets.items():
        sv = fit_state_value(
            data,
            hidden=cfg.get("hidden", 64),
            n_layers=cfg.get("n_layers", 2),
            epochs=cfg.get("critic_epochs", 30),
            batch_size=cfg.get("batch_size", 256),
            gamma=gamma,
            seed=seed,
        )
        vfn = lambda o: sv(o)  # noqa: E731
        gfn = lambda o: state_value_grad_norm(sv, o)  # noqa: E731
        for fam in families:
            fd = fit_dynamics(
                data,
                fam,
                value_fn=vfn,
                grad_norm_fn=gfn,
                n_models=cfg.get("n_models", 3),
                hidden=cfg.get("hidden", 64),
                n_layers=cfg.get("n_layers", 2),
                epochs=cfg.get("epochs", 30),
                batch_size=cfg.get("batch_size", 256),
                seed=seed,
            )
            g_model = policy_gradient(env, actor, s0, horizon, gamma, fd=fd)
            rows.append(
                {
                    "d_d": d_d,
                    "sigma_dist": sigma,
                    "dataset": name,
                    "family": fam.value,
                    "cos": gradient_cosine(g_true, g_model),
                    "g_true_norm": float(g_true.norm()),
                    "g_model_norm": float(g_model.norm()),
                }
            )
    return rows


def run(cfg: dict, out_dir: str) -> dict:
    rows = []
    for d_d in cfg.get("d_d_list", [0, 10, 50]):
        for sigma in cfg.get("sigma_dist_list", [0.0, 0.3]):
            for seed in range(cfg.get("n_seeds", 2)):
                print(f"[exp1_deep] d_d={d_d} sigma={sigma} seed={seed}", flush=True)
                rows.extend(regime_rows(cfg, d_d, sigma, seed))
    save_manifest(cfg, out_dir)
    save_json(rows, out_dir, "alignment")
    _plot(rows, out_dir)
    return {"rows": rows, "out_dir": out_dir}


def _plot(rows: list[dict], out_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    datasets = sorted({r["dataset"] for r in rows})
    families = sorted({r["family"] for r in rows})
    sigmas = sorted({r["sigma_dist"] for r in rows})
    fig, axes = plt.subplots(len(datasets), len(sigmas), figsize=(4.5 * len(sigmas), 3.6 * len(datasets)), squeeze=False)
    for di, ds in enumerate(datasets):
        for si, sig in enumerate(sigmas):
            ax = axes[di][si]
            sub = [r for r in rows if r["dataset"] == ds and r["sigma_dist"] == sig]
            xs = sorted({r["d_d"] for r in sub})
            for fam in families:
                y = [np.mean([r["cos"] for r in sub if r["d_d"] == d and r["family"] == fam]) for d in xs]
                ax.plot(xs, y, marker="o", label=fam)
            ax.axhline(0.0, color="k", lw=0.5)
            ax.set_xlabel("distractor dims $d_d$")
            ax.set_ylabel(r"$\cos(g_{true}, g_{model})$")
            ax.set_title(f"{ds}, sigma={sig}")
            ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, out_dir, "deep_alignment")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="runs/exp1_deep_alignment")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
