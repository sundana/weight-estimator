"""Weighted model-learning loss family.

``L_w(theta) = E_D[ w(s, a, s') * ell(p_theta(s' | s, a)) ]`` with the weight
estimator ``w_hat = w(V_hat, grad V_hat)`` as the object of study
(RESEARCH_PLAN.md Sec. 2).
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class LossFamily(str, Enum):
    """Model-learning objectives compared across experiments."""

    MLE = "mle"
    VAML1 = "vaml1"
    VAGRAM = "vagram"
    TD_ERROR = "td_error"
    CALIBRATED = "calibrated"
    LAMBERT = "lambert"
    DECISION_ALIGNED = "decision_aligned"


def weight(
    family: LossFamily,
    *,
    V_s: np.ndarray | None = None,
    V_sp: np.ndarray | None = None,
    grad_V_sp: np.ndarray | None = None,
    delta_td: np.ndarray | None = None,
    eps_model: np.ndarray | None = None,
    sigma_epistemic: np.ndarray | None = None,
    eps_reg: float = 1e-8,
    weight_fn: np.ndarray | None = None,
    tau: float = 1.0,
) -> np.ndarray:
    """Compute the per-transition weight ``w(s, a, s')``.

    Inputs are batches over transitions; at least one value input is required so the
    batch size can be inferred. Required inputs per family:

    - MLE: none (returns all-ones of the inferred batch size).
    - VAML1: ``V_s`` and ``V_sp``; returns ``|V(s') - V(s)|``.
    - VAGRAM: ``grad_V_sp``; returns ``||grad_s V(s')||_2`` (WP1 ``w1``).
    - TD_ERROR: ``delta_td = |V(s') - V(s_hat')|`` (WP1 ``w2``).
    - CALIBRATED: ``grad_V_sp``, ``eps_model = ||s_hat' - s'||`` and
      ``sigma_epistemic``; returns the WP1 proposed estimator
      ``w_prop = ||grad_s V(s')||_2 * eps_model / (sigma_epistemic + eps_reg)``.
    - LAMBERT: ``V_sp``; exponential tilt toward high-value outcomes,
      ``exp((V(s') - max V(s')) / tau)``.
    - DECISION_ALIGNED: ``weight_fn`` (e.g. per-sample Bellman residual).

    Returns:
        Batch of non-negative weights.
    """
    n = _infer_batch_size(
        V_s, V_sp, grad_V_sp, delta_td, eps_model, sigma_epistemic, weight_fn
    )
    if family == LossFamily.MLE:
        return np.ones(n)
    if family == LossFamily.VAML1:
        if V_s is None or V_sp is None:
            raise ValueError("VAML1 requires V_s and V_sp")
        return np.abs(V_sp - V_s)
    if family == LossFamily.VAGRAM:
        if grad_V_sp is None:
            raise ValueError("VAGRAM requires grad_V_sp")
        return np.linalg.norm(grad_V_sp, axis=-1)
    if family == LossFamily.TD_ERROR:
        if delta_td is None:
            raise ValueError("TD_ERROR requires delta_td")
        return np.abs(np.asarray(delta_td, dtype=float))
    if family == LossFamily.CALIBRATED:
        if grad_V_sp is None or eps_model is None or sigma_epistemic is None:
            raise ValueError("CALIBRATED requires grad_V_sp, eps_model and sigma_epistemic")
        grad_norm = np.linalg.norm(grad_V_sp, axis=-1)
        return grad_norm * np.asarray(eps_model, dtype=float) / (
            np.asarray(sigma_epistemic, dtype=float) + eps_reg
        )
    if family == LossFamily.LAMBERT:
        if V_sp is None:
            raise ValueError("LAMBERT requires V_sp")
        return np.exp((V_sp - np.max(V_sp)) / tau)
    if family == LossFamily.DECISION_ALIGNED:
        if weight_fn is None:
            raise ValueError("decision-aligned weighting requires weight_fn")
        return np.asarray(weight_fn, dtype=float)
    raise ValueError(f"unknown loss family: {family}")


def self_normalize(w: np.ndarray, eps_reg: float = 1e-8) -> np.ndarray:
    """Batch self-normalization ``w_bar_i = w_i / (mean_j w_j + eps_reg)``.

    Keeps the expected model-loss gradient scale near one (``E[w_bar] ~ 1``) so the
    stabilized objective does not shift the effective optimizer step size (WP2).
    """
    w = np.asarray(w, dtype=float)
    return w / (w.mean() + eps_reg)


def clip_weights(
    w: np.ndarray, w_min: float = 0.0, w_max: float | None = None
) -> np.ndarray:
    """Clip weights into ``[w_min, w_max]``; ``w_max=None`` leaves the upper end free.

    Bounds the empirical Lipschitz constant of the weighted loss and suppresses
    gradient spikes from critic singularities (WP2 weight clipping).
    """
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, w_min)
    if w_max is not None:
        w = np.minimum(w, w_max)
    return w


def _infer_batch_size(*arrays) -> int:
    for arr in arrays:
        if arr is not None:
            return np.asarray(arr).shape[0]
    raise ValueError("at least one input array is required to infer batch size")


def weighted_loss(
    model, batch: np.ndarray, weights: np.ndarray, eps: float = 1e-12
) -> float:
    """Weighted negative log-likelihood over a transition batch.

    ``model`` maps ``(s, a)`` to a probability vector over ``s'``
    (shape ``(n, n_states)``); ``batch`` is an ``(n, 3)`` array of ``(s, a, s')``.
    """
    probs = model(batch[:, 0], batch[:, 1])
    p = probs[np.arange(len(batch)), batch[:, 2]]
    return float(np.mean(weights * -np.log(np.clip(p, eps, 1.0))))