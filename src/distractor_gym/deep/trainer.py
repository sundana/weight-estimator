"""Offline weighted dynamics fitting for the WP1 deep diagnostics.

Fits a ``GaussianEnsemble`` next-state model on an offline replay under one of the WP1
estimators. Weights are recomputed every update from a supplied value function and its
per-sample state gradient, so the scheme mirrors ``deep.losses.model_weights`` while
keeping the critic free of gradients from the model loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..losses import LossFamily
from .losses import self_normalize
from .nets import GaussianEnsemble, StateValue


@dataclass
class Standardizer:
    """Per-dimension affine normalization ``(x - mean) / std``."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        return cls(x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True) + 1e-6)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class FittedDynamics:
    """Trained ensemble plus its normalization, with raw-space prediction helpers."""

    def __init__(
        self,
        model: GaussianEnsemble,
        x_scaler: Standardizer,
        y_scaler: Standardizer,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.device = torch.device(device)

    def _normalize_inputs(self, s: np.ndarray, a: np.ndarray) -> torch.Tensor:
        x = np.concatenate([s, a], axis=-1).astype(np.float32)
        return torch.as_tensor(self.x_scaler.transform(x), device=self.device)

    @torch.no_grad()
    def predict_mean(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Ensemble-mean next state in raw coordinates, shape ``(batch, state_dim)``."""
        mean, _ = self.model(self._normalize_inputs(s, a))
        raw = self.y_scaler.inverse(mean.mean(dim=1).cpu().numpy())
        return raw

    @torch.no_grad()
    def epistemic_std(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Per-sample epistemic std (member spread), shape ``(batch,)``."""
        mean, _ = self.model(self._normalize_inputs(s, a))
        raw = self.y_scaler.inverse(mean.cpu().numpy())
        return raw.std(axis=1).mean(axis=-1)

    def predict_torch(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Differentiable ensemble-mean next state in raw coordinates, shape ``(batch, d)``."""
        x = torch.cat([s, a], dim=-1)
        mean = torch.as_tensor(self.x_scaler.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.x_scaler.std, dtype=x.dtype, device=x.device)
        y_mean = torch.as_tensor(self.y_scaler.mean, dtype=x.dtype, device=x.device)
        y_std = torch.as_tensor(self.y_scaler.std, dtype=x.dtype, device=x.device)
        pred_n, _ = self.model((x - mean) / std)
        return pred_n.mean(dim=1) * y_std + y_mean


def fit_dynamics(
    data: dict,
    family: LossFamily = LossFamily.MLE,
    *,
    value_fn=None,
    grad_norm_fn=None,
    n_models: int = 5,
    hidden: int = 256,
    n_layers: int = 2,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    seed: int = 0,
    self_norm: bool = True,
    w_max: float | None = None,
    eps_reg: float = 1e-8,
    log_every: int = 0,
) -> FittedDynamics:
    """Fit a weighted next-state ensemble on ``data`` (dict of obs/act/next_obs arrays).

    ``value_fn(obs) -> (batch,)`` and ``grad_norm_fn(obs) -> (batch,)`` supply the raw
    value and ``||grad_s V||`` used by the non-MLE estimators; both are evaluated under
    ``no_grad`` so the model loss does not back-propagate into the critic. Following the
    WP1 weight definitions, the value and its gradient are evaluated at the *next*
    state ``s'`` (``VAML1``/``VAGRAM``/``LAMBERT``) and the TD-error weight compares
    ``V(s')`` against ``V(s_hat')``; only the model error ``||s_hat' - s'||`` and the
    epistemic spread are functions of the current fit.
    """
    torch.manual_seed(seed)
    obs = np.asarray(data["obs"], dtype=np.float32)
    act = np.asarray(data["act"], dtype=np.float32)
    next_obs = np.asarray(data["next_obs"], dtype=np.float32)
    x = np.concatenate([obs, act], axis=-1)
    x_scaler = Standardizer.fit(x)
    y_scaler = Standardizer.fit(next_obs)
    device = torch.device(device)

    model = GaussianEnsemble(
        in_dim=x.shape[1], state_dim=next_obs.shape[1], n_models=n_models, hidden=hidden, n_layers=n_layers
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(next_obs, dtype=torch.float32)
    n = len(xt)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = xt[idx].to(device)
            yb = yt[idx].to(device)
            obs_b = torch.as_tensor(obs[idx.numpy()], device=device)
            mean_n, _ = model(xb)
            pred_raw = mean_n * torch.as_tensor(y_scaler.std, device=device) + torch.as_tensor(
                y_scaler.mean, device=device
            )
            with torch.no_grad():
                sig = pred_raw.std(dim=1).mean(dim=-1)
                mean_pred = pred_raw.mean(dim=1)
                eps_model = (mean_pred - yb).norm(dim=-1)
                v_s = value_fn(obs_b) if family == LossFamily.VAML1 else None
                v_sp = (
                    value_fn(yb)
                    if family
                    in (
                        LossFamily.VAML1,
                        LossFamily.TD_ERROR,
                        LossFamily.CALIBRATED,
                        LossFamily.LAMBERT,
                    )
                    else None
                )
                v_hat = (
                    value_fn(mean_pred)
                    if family in (LossFamily.TD_ERROR, LossFamily.CALIBRATED)
                    else None
                )
                grad_norm = (
                    grad_norm_fn(yb)
                    if grad_norm_fn is not None
                    and family in (LossFamily.VAGRAM, LossFamily.CALIBRATED)
                    else None
                )
                delta_td = (
                    (v_sp - v_hat).abs() if v_sp is not None and v_hat is not None else None
                )
                weights = _weights(
                    family, grad_norm, eps_model, sig, delta_td, v_s, v_sp, eps_reg
                )
                if w_max is not None:
                    weights = torch.clamp(weights, max=w_max)
            per_member = ((pred_raw - yb.unsqueeze(1)) ** 2).sum(dim=-1)
            w = self_normalize(weights, eps_reg) if self_norm else weights
            loss = (w.unsqueeze(1) * per_member).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[fit_dynamics] epoch {epoch + 1}/{epochs} loss={float(loss):.4f}", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return FittedDynamics(model, x_scaler, y_scaler, device=device)


def model_one_step_mse(fd: FittedDynamics, data: dict) -> float:
    """Scale-free one-step error: mean ``((s_hat' - s') / y_std)^2`` over batch and dims."""
    pred = fd.predict_mean(data["obs"], data["act"])
    target = np.asarray(data["next_obs"], dtype=np.float64)
    return float(np.mean(((pred - target) / fd.y_scaler.std) ** 2))


def _weights(family, grad_norm, eps_model, sig, delta_td, v_s, v_sp, eps_reg, tau=1.0):
    if family == LossFamily.MLE:
        return torch.ones_like(eps_model)
    if family == LossFamily.VAML1:
        return (v_sp - v_s).abs()
    if family == LossFamily.VAGRAM:
        return grad_norm
    if family == LossFamily.TD_ERROR:
        return delta_td
    if family == LossFamily.CALIBRATED:
        return grad_norm * eps_model / (sig + eps_reg)
    if family == LossFamily.LAMBERT:
        return torch.exp((v_sp - v_sp.max()) / tau)
    raise ValueError(f"unknown loss family: {family}")


def fit_state_value(
    data: dict,
    *,
    hidden: int = 256,
    n_layers: int = 2,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    gamma: float = 0.99,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> StateValue:
    """Fit a state-value baseline by TD(0) on an offline ``data`` dict.

    Requires ``obs``, ``next_obs``, ``rew`` and (optionally) ``done``. Used to supply
    the raw value and ``||grad_s V||`` for the value-aware model weights.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    obs = torch.as_tensor(np.asarray(data["obs"], dtype=np.float32), device=device)
    next_obs = torch.as_tensor(np.asarray(data["next_obs"], dtype=np.float32), device=device)
    rew = torch.as_tensor(np.asarray(data["rew"], dtype=np.float32), device=device)
    terminal = data.get("terminal", data.get("done", np.zeros(len(obs))))
    done = torch.as_tensor(np.asarray(terminal, dtype=np.float32), device=device)
    net = StateValue(obs.shape[1], hidden=hidden, n_layers=n_layers).to(device)
    target = StateValue(obs.shape[1], hidden=hidden, n_layers=n_layers).to(device)
    target.load_state_dict(net.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(obs)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            with torch.no_grad():
                bootstrap = rew[idx] + gamma * (1.0 - done[idx]) * target(next_obs[idx])
            loss = ((net(obs[idx]) - bootstrap) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.data.mul_(0.99).add_(0.01 * p.data)
    net.eval()
    return net


def load_replay(path: str | Path) -> dict:
    """Load a saved replay ``.npz`` into a plain dict of arrays for ``fit_dynamics``."""
    data = np.load(path)
    return {k: data[k] for k in ("obs", "act", "next_obs")}
