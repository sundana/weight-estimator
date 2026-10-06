"""Weighted model-learning losses for the WP1 deep diagnostics (torch).

Mirrors the numpy family in ``distractor_gym.losses`` and implements the four WP1 Exp
1.1 weight estimators: MLE (``w = 1``), VaGraM (``||grad_s V||``), TD-error
(``|delta_TD|``), and the proposed calibrated estimator
``||grad_s V|| * eps_model / (sigma_epistemic + eps_reg)``. Also provides batch
self-normalization and clipping, the WP2 stabilization primitives.
"""

from __future__ import annotations

import torch

from ..losses import LossFamily


def self_normalize(w: torch.Tensor, eps_reg: float = 1e-8) -> torch.Tensor:
    """Batch self-normalization ``w / (mean(w) + eps_reg)`` (``E[w_bar] ~ 1``)."""
    return w / (w.mean() + eps_reg)


def clip_weights(
    w: torch.Tensor, w_min: float = 0.0, w_max: float | None = None
) -> torch.Tensor:
    """Clip weights into ``[w_min, w_max]``; ``w_max=None`` leaves the upper end free."""
    w = torch.clamp(w, min=w_min)
    if w_max is not None:
        w = torch.clamp(w, max=w_max)
    return w


def model_weights(
    family: LossFamily,
    *,
    grad_norm: torch.Tensor | None = None,
    eps_model: torch.Tensor | None = None,
    sigma_epistemic: torch.Tensor | None = None,
    delta_td: torch.Tensor | None = None,
    V_s: torch.Tensor | None = None,
    V_sp: torch.Tensor | None = None,
    tau: float = 1.0,
    eps_reg: float = 1e-8,
) -> torch.Tensor:
    """Per-transition weight for the WP1 estimators (all inputs are ``(batch,)`` tensors)."""
    if family == LossFamily.MLE:
        ref = grad_norm if grad_norm is not None else delta_td
        if ref is None:
            raise ValueError("MLE requires a reference tensor to infer batch size")
        return torch.ones_like(ref)
    if family == LossFamily.VAML1:
        if V_s is None or V_sp is None:
            raise ValueError("VAML1 requires V_s and V_sp")
        return (V_sp - V_s).abs()
    if family == LossFamily.VAGRAM:
        if grad_norm is None:
            raise ValueError("VAGRAM requires grad_norm")
        return grad_norm
    if family == LossFamily.TD_ERROR:
        if delta_td is None:
            raise ValueError("TD_ERROR requires delta_td")
        return delta_td.abs()
    if family == LossFamily.CALIBRATED:
        if grad_norm is None or eps_model is None or sigma_epistemic is None:
            raise ValueError("CALIBRATED requires grad_norm, eps_model and sigma_epistemic")
        return grad_norm * eps_model.detach() / (sigma_epistemic.detach() + eps_reg)
    if family == LossFamily.LAMBERT:
        if V_sp is None:
            raise ValueError("LAMBERT requires V_sp")
        return torch.exp((V_sp - V_sp.max()) / tau)
    raise ValueError(f"unknown loss family: {family}")


def state_sq_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample squared L2 error ``||pred - target||_2^2``, shape ``(batch,)``."""
    return ((pred - target) ** 2).sum(dim=-1)


def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    normalize: bool = True,
    eps_reg: float = 1e-8,
) -> torch.Tensor:
    """Weighted squared-error loss ``mean(w_bar * ||s' - f(s,a)||_2^2)``."""
    per = state_sq_error(pred, target)
    w = self_normalize(weights, eps_reg) if normalize else weights
    return (w * per).mean()


def weighted_gaussian_nll(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    normalize: bool = True,
    eps_reg: float = 1e-8,
) -> torch.Tensor:
    """Weighted diagonal-Gaussian NLL summed over state dims and averaged over the batch."""
    per = 0.5 * ((target - mean) ** 2 / log_var.exp() + log_var).sum(dim=-1)
    w = self_normalize(weights, eps_reg) if normalize else weights
    return (w * per).mean()
