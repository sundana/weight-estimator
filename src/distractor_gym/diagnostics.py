"""Diagnostics API.

Quantifies objective mismatch and weight-estimator quality:

- ``gradient_alignment``: cosine similarity between the true and the model-induced
  policy gradient (RESEARCH_PLAN.md Sec. 4).
- ``decompose_td_error``: empirical test of ``|delta_TD| ~ ||grad V|| * eps_model``
  (RESEARCH_PLAN.md Sec. 2.2, Sec. 5).
- ``weight_estimator_stats`` / ``weight_signal_to_noise``: bias/variance/ESS/SNR of
  the weight estimator and the Theorem-1 crossover statistic (Sec. 2.3, Sec. 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gradient_alignment(g_true: np.ndarray, g_model: np.ndarray) -> float:
    """Cosine similarity between the true and model-induced policy gradients.

    ``g_true`` is ``grad_theta J(pi)`` under true dynamics; ``g_model`` is the policy
    gradient computed inside the model. Values near 1 mean the model-based update
    tracks the true improvement direction.
    """
    denom = np.linalg.norm(g_true) * np.linalg.norm(g_model)
    if denom == 0.0:
        return 0.0
    return float(np.dot(g_true, g_model) / denom)


@dataclass
class DecompositionResult:
    """Fit of ``|delta_TD| ~ ||grad V|| * eps_model * |cos phi|`` over transitions."""

    r2: float
    slope: float
    mean_cos_phi: float
    curvature_residual: float
    n: int


def decompose_td_error(
    delta_td: np.ndarray, grad_V_norm: np.ndarray, eps_model: np.ndarray, cos_phi: np.ndarray
) -> DecompositionResult:
    """Regress the factorization and report the unexplained (curvature) residual.

    ``delta_td = |V(s') - V(s_hat')|``, ``grad_V_norm = ||grad V(s')||``,
    ``eps_model = ||s_hat' - s'||``, ``cos_phi`` the alignment of the error vector
    with ``grad V``.
    """
    pred = grad_V_norm * eps_model * np.abs(cos_phi)
    n = len(delta_td)
    if n == 0:
        raise ValueError("empty input")
    slope = float(np.dot(delta_td, pred) / max(np.dot(pred, pred), 1e-12))
    ss_res = float(np.sum((delta_td - slope * pred) ** 2))
    ss_tot = float(np.sum((delta_td - np.mean(delta_td)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    curvature_residual = float(np.mean(np.abs(delta_td - pred)))
    return DecompositionResult(
        r2=r2,
        slope=slope,
        mean_cos_phi=float(np.mean(np.abs(cos_phi))),
        curvature_residual=curvature_residual,
        n=n,
    )


@dataclass
class WeightStats:
    """Bias/variance diagnostics of the weight estimator ``w_hat``."""

    bias: float
    variance: float
    effective_sample_size: float
    signal_to_noise: float


def weight_estimator_stats(w: np.ndarray, w_oracle: np.ndarray, weighted_losses: np.ndarray) -> WeightStats:
    """Report bias, variance, effective sample size, and SNR of the weight estimator.

    ``w`` is the estimated weight batch, ``w_oracle`` the oracle (true ``V``/``grad V``)
    weight batch, ``weighted_losses`` the per-sample weighted losses ``w_hat * ell``.
    """
    if len(w) == 0:
        raise ValueError("empty input")
    bias = float(np.mean(w - w_oracle))
    variance = float(np.var(w))
    ess = float(np.sum(w) ** 2 / max(np.sum(w**2), 1e-12))
    snr = weight_signal_to_noise(weighted_losses, w)
    return WeightStats(bias=bias, variance=variance, effective_sample_size=ess, signal_to_noise=snr)


def weight_signal_to_noise(weighted_losses: np.ndarray, w: np.ndarray) -> float:
    """Theorem-1 crossover statistic ``SNR_w``.

    ``SNR_w = (E[w * ell])^2 / Var(w * ell)``. Below a threshold this predicts the MLE
    model (``w == 1``) has strictly lower risk; see RESEARCH_PLAN.md Sec. 2.3.
    """
    if len(weighted_losses) == 0:
        raise ValueError("empty input")
    mean = float(np.mean(weighted_losses))
    var = float(np.var(weighted_losses))
    if var == 0.0:
        return float("inf")
    return mean**2 / var