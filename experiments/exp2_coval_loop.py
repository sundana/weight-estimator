"""Phase C: deep online coupled MBRL loop (Part II of ``paper/coval``).

Runs ``distractor_gym.deep.coupled_loop.DeepCoupledLoop`` across distractor regimes and
loss families on the differentiable MuJoCo Distractor-Gym. Reports the policy return
over training (the return-level H1.1 test) plus the weight-estimator variance ``sigma_w^2``
and effective sample size, comparing COVAL against MLE and the unstabilized VaGraM.

Config: ``configs/exp2_coval_loop.yaml``. Output: ``runs/exp2_coval_loop/``.
"""

from __future__ import annotations

import argparse

import numpy as np

from distractor_gym.deep.coupled_loop import CoupledLoopConfig, DeepCoupledLoop
from distractor_gym.deep.mujoco_diff import MujocoDistractorEnv, MujocoDistractorGym

from .common import load_config, save_fig, save_json, save_manifest


def _loop_config(cfg: dict, family: str, seed: int) -> CoupledLoopConfig:
    """Build the loop config for one family, applying its stabilization overrides."""
    base = dict(
        gamma=float(cfg.get("gamma", 0.99)),
        horizon=int(cfg.get("horizon", 5)),
        iterations=int(cfg.get("iterations", 8)),
        model_updates=int(cfg.get("model_updates", 150)),
        agent_updates=int(cfg.get("agent_updates", 150)),
        batch_size=int(cfg.get("batch_size", 256)),
        model_lr=float(cfg.get("model_lr", 1e-3)),
        agent_lr=float(cfg.get("agent_lr", 3e-4)),
        n_models=int(cfg.get("n_models", 3)),
        hidden=int(cfg.get("hidden", 64)),
        n_layers=int(cfg.get("n_layers", 2)),
        clip_sigma=float(cfg.get("clip_sigma", 6.0)),
        init_steps=int(cfg.get("init_steps", 800)),
        collect_steps=int(cfg.get("collect_steps", 250)),
        rollouts_per_iter=int(cfg.get("rollouts_per_iter", 2)),
        rollout_batch=int(cfg.get("rollout_batch", 256)),
        eval_every=int(cfg.get("eval_every", 2)),
        eval_episodes=int(cfg.get("eval_episodes", 3)),
        seed=seed,
        family=family,
    )
    if family == "coval":
        base.update(cfg.get("coval", {}))
    elif family == "vagram":
        base.update(cfg.get("vagram", {}))
    return CoupledLoopConfig(**base)


def _run_one(cfg: dict, d_d: int, sigma: float, family: str, seed: int) -> list[dict]:
    base_task = cfg.get("base_task", "inverted_double_pendulum")
    gym_env = MujocoDistractorGym(
        d_d=d_d,
        sigma_dist=sigma,
        seed=seed,
        base_task=base_task,
        horizon=int(cfg.get("data_horizon", 100)),
        reward_mode=cfg.get("reward_mode", "dense"),
        goal_sigma=float(cfg.get("goal_sigma", 0.25)),
    )
    torch_env = MujocoDistractorEnv(
        d_d=d_d,
        sigma_dist=sigma,
        seed=seed,
        base_task=base_task,
        eps=float(cfg.get("mujoco_eps", 1e-6)),
        centered=bool(cfg.get("mujoco_centered", True)),
        reward_mode=cfg.get("reward_mode", "dense"),
        goal_sigma=float(cfg.get("goal_sigma", 0.25)),
    )
    try:
        loop = DeepCoupledLoop(gym_env, torch_env, _loop_config(cfg, family, seed))
        history = loop.run()
    finally:
        gym_env.close()
    return [{"d_d": d_d, "sigma_dist": sigma, "family": family, "seed": seed, **h} for h in history]


def run(cfg: dict, out_dir: str) -> dict:
    save_manifest(cfg, out_dir)
    rows: list[dict] = []
    for d_d in cfg.get("d_d_list", [0]):
        for sigma in cfg.get("sigma_dist_list", [0.0]):
            for family in cfg.get("families", ["mle"]):
                for seed in range(int(cfg.get("n_seeds", 1))):
                    print(f"[exp2_coval_loop] d_d={d_d} sigma={sigma} family={family} seed={seed}", flush=True)
                    rows.extend(_run_one(cfg, d_d, sigma, family, seed))
                    save_json(rows, out_dir, "coupled_loop")
    _plot(rows, out_dir)
    return {"rows": rows, "out_dir": out_dir}


def _plot(rows: list[dict], out_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    d_ds = sorted({r["d_d"] for r in rows})
    sigmas = sorted({r["sigma_dist"] for r in rows})
    families = sorted({r["family"] for r in rows})
    fig, axes = plt.subplots(
        len(d_ds), len(sigmas), figsize=(4.5 * len(sigmas), 3.4 * len(d_ds)), squeeze=False
    )
    for di, d_d in enumerate(d_ds):
        for si, sigma in enumerate(sigmas):
            ax = axes[di][si]
            for family in families:
                sub = [r for r in rows if r["d_d"] == d_d and r["sigma_dist"] == sigma and r["family"] == family]
                if not sub:
                    continue
                its = sorted({r["iteration"] for r in sub})
                means, stds = [], []
                for it in its:
                    vals = [r["return"] for r in sub if r["iteration"] == it]
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
                ax.errorbar(its, means, yerr=stds, marker="o", capsize=2, label=family)
            ax.set_xlabel("iteration")
            ax.set_ylabel("evaluation return")
            ax.set_title(f"$d_d$={d_d}, sigma={sigma}")
            ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, "coupled_loop")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="runs/exp2_coval_loop")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
