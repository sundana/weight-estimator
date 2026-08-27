"""Tabular Distractor-Gym suite.

Ground-truth lab: exact ``g_true``, ``delta_TD``, ``eps_model`` and ``grad V`` are
computable, enabling exact validation of the weight-estimator theory and the
``|delta_TD| ~ ||grad V|| * eps_model`` decomposition (RESEARCH_PLAN.md Sec. 2, 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import RegimeConfig


@dataclass
class Grid:
    """Equispaced 1D grid of ``n`` points in ``[low, high]``."""

    low: float
    high: float
    n: int

    @property
    def points(self) -> np.ndarray:
        return np.linspace(self.low, self.high, self.n)

    @property
    def dx(self) -> float:
        return (self.high - self.low) / (self.n - 1)


class TabularDistractorEnv:
    """Discretized MDP with control grid ``s_c`` and distractor grid ``s_d``.

    Reward depends only on ``s_c``; the value function is flat along ``s_d``.
    """

    def __init__(self, config: RegimeConfig, grid_c: Grid | None = None, grid_d: Grid | None = None) -> None:
        self.config = config
        self.grid_c = grid_c if grid_c is not None else Grid(-3.0, 3.0, 21)
        self.grid_d = grid_d if grid_d is not None else Grid(-3.0, 3.0, 7)
        self.rng = np.random.default_rng(config.seed)

    def reward(self, s_c: np.ndarray) -> float:
        """Reward as a function of the control block only."""
        raise NotImplementedError

    def transition_probs(self) -> np.ndarray:
        """Exact transition tensor ``P[(s_c, s_d, a)] -> (s_c', s_d')``."""
        raise NotImplementedError

    def value_iteration(self, gamma: float = 0.99) -> np.ndarray:
        """Exact value function ``V(s_c, s_d)`` on the product grid."""
        raise NotImplementedError

    def true_value_grad(self, V: np.ndarray) -> np.ndarray:
        """Finite-difference gradient ``grad_s V`` on the grid."""
        raise NotImplementedError

    def td_error(self, V: np.ndarray, s: np.ndarray, s_hat: np.ndarray) -> float:
        """Model-induced bootstrap-target error ``|V(s') - V(s_hat')|``."""
        raise NotImplementedError

    def transition_error(self, s: np.ndarray, s_hat: np.ndarray) -> float:
        """One-step model prediction error ``eps_model = ||s_hat' - s'||``."""
        return float(np.linalg.norm(s_hat - s))

    def policy_gradient_true(self, theta: np.ndarray) -> np.ndarray:
        """Exact policy gradient ``g_true = grad_theta J(pi)`` under true dynamics."""
        raise NotImplementedError