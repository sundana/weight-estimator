"""Regime configuration and distractor dynamics for Distractor-Gym.

State is ``s = (s_c, s_d)`` with ``d_c`` control-relevant and ``d_d`` distractor
dimensions. Distractors have complex dynamics and zero reward relevance, so
``grad_{s_d} V = 0`` by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class DistractorClass(str, Enum):
    """Distractor dynamics family, ordered by failure pressure."""

    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    STOCHASTIC = "stochastic"


@dataclass
class RegimeConfig:
    """Knobs swept across the mismatch phase diagram (WP1 Distractor-Gym)."""

    d_c: int = 2
    d_d: int = 8
    distractor_class: DistractorClass = DistractorClass.NONLINEAR
    reward_sparsity: float = 1.0
    goal_radius: float | None = None
    transition_noise: float = 0.0
    sigma_dist: float = 0.0
    model_capacity: float = 1.0
    seed: int = 0


class DistractorDynamics:
    """Action-coupled dynamics for the reward-irrelevant state block ``s_d``.

    Implements the WP1 Distractor-Gym form ``s_d' = tanh(A s_d + B a + xi)`` for the
    nonlinear class, with ``xi ~ N(0, sigma_dist^2 I)``. The linear class drops the
    ``tanh`` and the stochastic class is an action-coupled random walk. The couplings
    ``A`` (``d_d x d_d``) and ``B`` (``d_d x action_dim``) are fixed by the regime seed,
    while the noise stream is re-seeded on every environment reset for replay.
    """

    def __init__(
        self,
        config: RegimeConfig,
        action_dim: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.action_dim = int(action_dim)
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        d = config.d_d
        coupling = np.random.default_rng(config.seed + 101)
        if d > 0:
            self._matrix = coupling.normal(size=(d, d)) / np.sqrt(d)
            self._action_matrix = (
                coupling.normal(size=(d, self.action_dim)) / np.sqrt(max(self.action_dim, 1))
                if self.action_dim > 0
                else np.zeros((d, 0))
            )
        else:
            self._matrix = np.zeros((0, 0))
            self._action_matrix = np.zeros((0, self.action_dim))

    def reseed(self, seed: int) -> None:
        """Re-seed the distractor noise stream (couplings stay fixed)."""
        self.rng = np.random.default_rng(seed)

    def _drive(self, action: np.ndarray | None) -> np.ndarray:
        if self.action_dim == 0 or action is None:
            return np.zeros(self.config.d_d)
        a = np.asarray(action, dtype=float).reshape(-1)
        return self._action_matrix @ a

    def _noise(self, scale: float) -> np.ndarray:
        if scale <= 0.0:
            return np.zeros(self.config.d_d)
        return self.rng.normal(0.0, scale, size=self.config.d_d)

    def step(self, s_d: np.ndarray, action: np.ndarray | None = None) -> np.ndarray:
        """Advance ``s_d`` by one step according to the configured dynamics class."""
        if self.config.d_d == 0:
            return np.zeros(0)
        s_d = np.asarray(s_d, dtype=float).reshape(-1)
        drive = self._drive(action)
        sigma = float(self.config.sigma_dist)
        cls = self.config.distractor_class
        if cls == DistractorClass.NONLINEAR:
            return np.tanh(s_d @ self._matrix.T + drive + self._noise(sigma))
        if cls == DistractorClass.LINEAR:
            return s_d @ self._matrix.T + drive + self._noise(sigma)
        if cls == DistractorClass.STOCHASTIC:
            return s_d + drive + self._noise(sigma if sigma > 0.0 else 1.0)
        raise ValueError(f"unknown distractor class: {cls}")