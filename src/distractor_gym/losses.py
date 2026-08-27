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
    LAMBERT = "lambert"
    DECISION_ALIGNED = "decision_aligned"


def weight(
    family: LossFamily,
    *,
    V_s: np.ndarray | None = None,
    V_sp: np.ndarray | None = None,
    grad_V_sp: np.ndarray | None = None,
    weight_fn: np.ndarray | None = None,
    tau: float = 1.0,
) -> np.ndarray:
    """Compute the per-transition weight ``w(s, a, s')``.

    Inputs are batches over transitions; at least one value input is required so the
    batch size can be inferred. Required inputs per family:

    - MLE: none (returns all-ones of the inferred batch size).
    - VAML1: ``V_s`` and ``V_sp``; returns ``|V(s') - V(s)|``.
    - VAGRAM: ``grad_V_sp``; returns the L2 norm of the value gradient at ``s'``.
    - LAMBERT: ``V_sp``; exponential tilt toward high-value outcomes,
      ``exp((V(s') - max V(s')) / tau)``.
    - DECISION_ALIGNED: ``weight_fn`` (e.g. per-sample Bellman residual).

    Returns:
        Batch of non-negative weights.
    """
    n = _infer_batch_size(V_s, V_sp, grad_V_sp, weight_fn)
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
    if family == LossFamily.LAMBERT:
        if V_sp is None:
            raise ValueError("LAMBERT requires V_sp")
        return np.exp((V_sp - np.max(V_sp)) / tau)
    if family == LossFamily.DECISION_ALIGNED:
        if weight_fn is None:
            raise ValueError("decision-aligned weighting requires weight_fn")
        return np.asarray(weight_fn, dtype=float)
    raise ValueError(f"unknown loss family: {family}")


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