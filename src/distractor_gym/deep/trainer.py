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
from .nets import GaussianEnsemble


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
    ``no_grad`` so the model loss does not back-propagate into the critic.
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
                grad_norm = grad_norm_fn(obs_b) if grad_norm_fn is not None else None
                if family == LossFamily.TD_ERROR or family == LossFamily.CALIBRATED:
                    v_sp = value_fn(obs_b)
                    v_hat = value_fn(mean_pred)
                    delta_td = (v_sp - v_hat).abs()
                else:
                    delta_td = None
                weights = _weights(family, grad_norm, eps_model, sig, delta_td, eps_reg)
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
    return FittedDynamics(model, x_scaler, y_scaler, device=device)


def _weights(family, grad_norm, eps_model, sig, delta_td, eps_reg):
    if family == LossFamily.MLE:
        return torch.ones_like(eps_model)
    if family == LossFamily.VAGRAM:
        return grad_norm
    if family == LossFamily.TD_ERROR:
        return delta_td
    if family == LossFamily.CALIBRATED:
        return grad_norm * eps_model / (sig + eps_reg)
    raise ValueError(f"unknown loss family: {family}")


def load_replay(path: str | Path) -> dict:
    """Load a saved replay ``.npz`` into a plain dict of arrays for ``fit_dynamics``."""
    data = np.load(path)
    return {k: data[k] for k in ("obs", "act", "next_obs")}
