"""Experiment 4: continuous Distractor-Gym model-learning diagnostics.

Transfers the Phase 1a/1b tabular findings to a continuous neural setting:
Pendulum-v1 with appended chaotic distractor dims and a small MLP dynamics model.
The model is trained with either the MLE/MSE loss or the VaGraM projection loss
``(grad r(s') . (s_hat' - s'))^2``, which ignores prediction error along the
value-flat distractor dims.

Metrics per (distractor count, capacity) regime:

- global MSE (held-out): MLE should be at or below VaGraM (Theorem 1 analogue).
- value error ``E[|r(s') - r(s_hat')|]``: VaGraM should win when distractors consume
  capacity (its native regime).
- Lemma-1 transfer: ``R^2`` of ``|r(s') - r(s_hat')| ~ ||grad r(s')|| * eps * |cos phi|``.
- directional alignment: mean ``cos(grad r(s'), grad r(s_hat'))``.
"""

from __future__ import annotations

import argparse

import numpy as np

from distractor_gym.continuous import make_distractor_gym
from distractor_gym.core import DistractorClass, RegimeConfig
from distractor_gym.diagnostics import decompose_td_error

from .common import load_config, save_fig, save_json

TORCH = None


def _torch():
    global TORCH
    if TORCH is None:
        import torch

        TORCH = torch
    return TORCH


def pendulum_reward(obs_c: np.ndarray, action: np.ndarray) -> np.ndarray:
    cos, sin, omega = obs_c[:, 0], obs_c[:, 1], obs_c[:, 2]
    theta = np.arctan2(sin, cos)
    return -(theta**2 + 0.1 * omega**2 + 0.001 * action[:, 0] ** 2)


def pendulum_grad_r(obs_c: np.ndarray) -> np.ndarray:
    cos, sin, omega = obs_c[:, 0], obs_c[:, 1], obs_c[:, 2]
    theta = np.arctan2(sin, cos)
    g = np.stack([2.0 * theta * sin, -2.0 * theta * cos, -0.2 * omega], axis=1)
    return g


def collect_data(env, n_steps: int, rng: np.random.Generator) -> dict:
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    s, a, sp = [], [], []
    for _ in range(n_steps):
        act = rng.uniform(-2.0, 2.0, size=env.action_space.shape).astype(np.float32)
        obs2, _, term, trunc, _ = env.step(act)
        s.append(obs)
        a.append(act)
        sp.append(obs2)
        obs = obs2
        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    return {
        "s": np.asarray(s, dtype=np.float32),
        "a": np.asarray(a, dtype=np.float32),
        "sp": np.asarray(sp, dtype=np.float32),
    }


def standardize(train, val):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, (val - mean) / std


def make_model(in_dim, out_dim, hidden, seed):
    torch = _torch()
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, out_dim),
    )


def train_model(cfg: dict, data: dict, loss_kind: str, seed: int) -> np.ndarray:
    torch = _torch()
    d_c = data["s"].shape[1] - cfg.get("d_d", 0)
    d_d = cfg.get("d_d", 0)
    in_dim = data["s"].shape[1] + data["a"].shape[1]
    out_dim = data["s"].shape[1]
    X = np.concatenate([data["s"], data["a"]], axis=1)
    Y = data["sp"]
    split = int(len(X) * (1.0 - cfg.get("val_frac", 0.2)))
    Xtr, Xva = standardize(X[:split], X[split:])
    Ytr, Yva = standardize(Y[:split], Y[split:])

    grad_proj_tr = None
    if loss_kind == "vagram":
        g = pendulum_grad_r(data["sp"][:split])
        grad_proj_tr = np.zeros((len(g), d_c + d_d), dtype=np.float32)
        grad_proj_tr[:, :d_c] = g
        gn = np.linalg.norm(grad_proj_tr, axis=1, keepdims=True) + 1e-6
        grad_proj_tr = torch.from_numpy(grad_proj_tr / gn)

    model = make_model(in_dim, out_dim, cfg.get("hidden", 64), seed)
    if torch.cuda.is_available():
        model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    bs = cfg.get("batch_size", 512)
    epochs = cfg.get("epochs", 200)
    lam = cfg.get("lam", 0.0)
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb = torch.from_numpy(Xtr[idx]).to(next(model.parameters()).device)
            yb = torch.from_numpy(Ytr[idx]).to(next(model.parameters()).device)
            pred = model(xb)
            mse = ((pred - yb) ** 2).mean()
            if loss_kind == "vagram":
                w = grad_proj_tr[idx].to(next(model.parameters()).device)
                proj = (w * (pred - yb)).sum(dim=1)
                loss = (1.0 - lam) * mse + lam * (proj**2).mean()
            else:
                loss = mse
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(Xva).to(next(model.parameters()).device)).cpu().numpy()
    return preds * np.std(Y[split:], axis=0, keepdims=True) + np.mean(Y[split:], axis=0, keepdims=True), Y[split:]


def evaluate(cfg: dict, pred: np.ndarray, target: np.ndarray) -> dict:
    d_c = target.shape[1] - cfg.get("d_d", 0)
    d = target.shape[1]
    sp_c = target[:, :d_c]
    sp_hat_c = pred[:, :d_c]
    r_true = pendulum_reward(sp_c, np.zeros((len(target), 1)))
    r_pred = pendulum_reward(sp_hat_c, np.zeros((len(target), 1)))
    grad_true = np.zeros((len(target), d), dtype=np.float32)
    grad_true[:, :d_c] = pendulum_grad_r(sp_c)
    grad_pred = np.zeros((len(target), d), dtype=np.float32)
    grad_pred[:, :d_c] = pendulum_grad_r(sp_hat_c)
    err_vec = pred - target

    global_mse = float(np.mean((pred - target) ** 2))
    value_error = float(np.mean(np.abs(r_true - r_pred)))
    delta = np.abs(r_true - r_pred)
    grad_norm = np.linalg.norm(grad_true, axis=1)
    eps = np.linalg.norm(err_vec, axis=1)
    cos_phi = np.zeros(len(target))
    nz = eps > 0
    cos_phi[nz] = np.einsum("ij,ij->i", grad_true[nz], err_vec[nz]) / (grad_norm[nz] * eps[nz])
    dec = decompose_td_error(delta, grad_norm, eps, cos_phi)
    num = np.einsum("ij,ij->i", grad_true, grad_pred)
    den = np.linalg.norm(grad_true, axis=1) * np.linalg.norm(grad_pred, axis=1)
    valid = den > 1e-8
    dir_align = float(np.mean(num[valid] / den[valid])) if valid.any() else float("nan")
    return {
        "global_mse": global_mse,
        "value_error": value_error,
        "r2": dec.r2,
        "curvature_residual": dec.curvature_residual,
        "dir_alignment": dir_align,
        "n": int(len(target)),
    }


def run(cfg: dict, out_dir: str) -> dict:
    rows = []
    for d_d in cfg.get("d_d_list", [0, 4]):
        for hidden in cfg.get("hidden_list", [32]):
            for loss_kind in cfg.get("loss_kinds", ["mle", "vagram"]):
                per_seed = []
                for i in range(cfg.get("n_seeds", 2)):
                    env_cfg = RegimeConfig(
                        d_d=d_d,
                        distractor_class=DistractorClass(cfg.get("distractor_class", "nonlinear")),
                        seed=cfg.get("seed", 0) + i,
                    )
                    env = make_distractor_gym(env_cfg, base_task=cfg.get("base_task", "pendulum"))
                    rng = np.random.default_rng(1000 * (i + 1) + d_d * 10 + hidden)
                    data = collect_data(env, cfg.get("n_data", 20000), rng)
                    row_cfg = dict(cfg, d_d=d_d, hidden=hidden)
                    print(f"[exp4] d_d={d_d} hidden={hidden} loss={loss_kind} seed={i}", flush=True)
                    pred, target = train_model(row_cfg, data, loss_kind, seed=i)
                    per_seed.append(evaluate(row_cfg, pred, target))
                row = {"d_d": d_d, "hidden": hidden, "loss": loss_kind}
                for k in per_seed[0]:
                    row[k] = float(np.mean([r[k] for r in per_seed]))
                rows.append(row)
                save_json(rows, out_dir, "results")
    _plot(rows, out_dir)
    return {"rows": rows, "out_dir": out_dir}


def _plot(rows: list[dict], out_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    hides = sorted({r["hidden"] for r in rows})
    fig, axes = plt.subplots(1, len(hides), figsize=(5.2 * len(hides), 3.8), squeeze=False)
    for hi, hidden in enumerate(hides):
        ax = axes[0][hi]
        sub = [r for r in rows if r["hidden"] == hidden]
        x = sorted({r["d_d"] for r in sub})
        for metric in ("global_mse", "value_error"):
            ax.plot(x, [next(r[metric] for r in sub if r["d_d"] == d and r["loss"] == "mle") for d in x], "o-", label=f"{metric} MLE")
            ax.plot(x, [next(r[metric] for r in sub if r["d_d"] == d and r["loss"] == "vagram") for d in x], "s--", label=f"{metric} VaGraM")
        ax.set_xlabel("distractor dims d_d")
        ax.set_title(f"hidden={hidden}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, out_dir, "model_comparison")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="runs/exp4")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())