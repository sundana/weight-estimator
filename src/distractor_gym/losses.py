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
    s: np.ndarray,
    s_prime: np.ndarray,
    V: np.ndarray,
    grad_V: np.ndarray,
    weight_fn=None,
) -> np.ndarray:
    """Compute the per-transition weight ``w(s, a, s')``.

    Args:
        family: loss family selecting the weighting rule.
        s: batch of current states.
        s_prime: batch of true next states.
        V: batch of value estimates ``V(s')``.
        grad_V: batch of value gradients ``grad_s V(s')``.
        weight_fn: optional decision-aligned weight (e.g. model-Bellman inconsistency).

    Returns:
        Batch of non-negative weights; MLE returns all-ones.
    """
    if family == LossFamily.MLE:
        return np.ones(s.shape[0])
    if family == LossFamily.VAML1:
        raise NotImplementedError
    if family == LossFamily.VAGRAM:
        raise NotImplementedError
    if family == LossFamily.LAMBERT:
        raise NotImplementedError
    if family == LossFamily.DECISION_ALIGNED:
        if weight_fn is None:
            raise ValueError("decision-aligned weighting requires weight_fn")
        raise NotImplementedError
    raise ValueError(f"unknown loss family: {family}")


def weighted_loss(family: LossFamily, model, batch: dict) -> float:
    """Evaluate the weighted model-learning loss for a model and transition batch."""
    raise NotImplementedError