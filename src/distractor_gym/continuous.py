"""Continuous Distractor-Gym suite.

Gymnasium wrapper that appends distractor dimensions with zero reward relevance to a
DMC-style reach task. Used for the deep-baseline benchmark (MBPO, VaGraM, then
DreamerV3 / TD-MPC2).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .core import DistractorDynamics, RegimeConfig


class DistractorGym(gym.Env):
    """Base task with appended distractor block ``s_d`` and reward on ``s_c`` only."""

    metadata = {"render_modes": []}

    def __init__(self, config: RegimeConfig, base_task: gym.Env) -> None:
        super().__init__()
        self.config = config
        self.base_task = base_task
        self.dynamics = DistractorDynamics(config)
        d_c = np.prod(base_task.observation_space.shape, dtype=int)
        d_d = config.d_d
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(d_c) + d_d,), dtype=np.float32
        )
        self.action_space = base_task.action_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_task.step(action)
        s_d = self.dynamics.step(self._s_d)
        self._s_d = s_d
        return np.concatenate([np.asarray(obs, dtype=np.float32), s_d]), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_task.reset(seed=seed, options=options)
        self._s_d = np.zeros(self.config.d_d, dtype=np.float32)
        return np.concatenate([np.asarray(obs, dtype=np.float32), self._s_d]), info