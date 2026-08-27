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
    """Knobs swept across the mismatch phase diagram (RESEARCH_PLAN.md Sec. 3)."""

    d_c: int = 2
    d_d: int = 8
    distractor_class: DistractorClass = DistractorClass.NONLINEAR
    reward_sparsity: float = 1.0
    goal_radius: float | None = None
    transition_noise: float = 0.0
    model_capacity: float = 1.0
    seed: int = 0


class DistractorDynamics:
    """Dynamics for the reward-irrelevant state block ``s_d``."""

    def __init__(self, config: RegimeConfig, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        self._matrix = self.rng.normal(size=(config.d_d, config.d_d)) / np.sqrt(config.d_d)

    def step(self, s_d: np.ndarray) -> np.ndarray:
        """Advance ``s_d`` by one step according to the configured dynamics class."""
        if self.config.distractor_class == DistractorClass.LINEAR:
            return s_d @ self._matrix.T
        if self.config.distractor_class == DistractorClass.NONLINEAR:
            return np.tanh(s_d @ self._matrix.T) * 1.5
        if self.config.distractor_class == DistractorClass.STOCHASTIC:
            return self.rng.normal(loc=s_d, scale=1.0)
        raise ValueError(f"unknown distractor class: {self.config.distractor_class}")