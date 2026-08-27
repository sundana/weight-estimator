"""Continuous Distractor-Gym suite.

Gymnasium wrapper that appends distractor dimensions with zero reward relevance to a
base control task (Pendulum-v1 by default). Reward depends only on the control
block, so the reward/value gradient is zero along the distractor dims. Used for the
neural model-learning diagnostics (exp4) and the deep-baseline benchmark (MBPO,
VaGraM, DreamerV3 / TD-MPC2).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .core import DistractorDynamics, RegimeConfig

_BASE_TASKS = {"pendulum": "Pendulum-v1", "cartpole": "CartPole-v1"}


class DistractorGym(gym.Env):
    """Base control task with an appended distractor block ``s_d``.

    The observation is ``concat(obs_c, s_d)`` with ``obs_c`` from the base task and
    ``s_d`` evolving under ``DistractorDynamics``. Rewards are untouched, so the
    value function is flat in ``s_d`` by construction.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: RegimeConfig, base_task: gym.Env) -> None:
        super().__init__()
        self.config = config
        self.base_task = base_task
        self.dynamics = DistractorDynamics(config)
        self.d_c = int(np.prod(base_task.observation_space.shape))
        self.d_d = int(config.d_d)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.d_c + self.d_d,), dtype=np.float32
        )
        self.action_space = base_task.action_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_task.step(action)
        self._s_d = self.dynamics.step(self._s_d)
        return self._obs(obs), float(reward), bool(terminated), bool(truncated), info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.base_task.reset(seed=seed, options=options)
        self._s_d = np.zeros(self.d_d, dtype=np.float32)
        return self._obs(obs), info

    def _obs(self, obs_c: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(obs_c, dtype=np.float32), self._s_d]).astype(np.float32)


def make_distractor_gym(
    config: RegimeConfig,
    base_task: str = "pendulum",
) -> DistractorGym:
    """Build a ``DistractorGym`` wrapping a named base task."""
    if base_task not in _BASE_TASKS:
        raise ValueError(f"unknown base task: {base_task}; use {sorted(_BASE_TASKS)}")
    return DistractorGym(config, gym.make(_BASE_TASKS[base_task]))