"""Continuous Distractor-Gym suite over Gymnasium/MuJoCo base tasks.

Wraps a base control task (``InvertedPendulum-v4``, ``HalfCheetah-v4``,
``Walker2d-v4``, plus ``Pendulum-v1``/``CartPole-v1`` for the tabular-era
experiments) and appends a ``d_d``-dimensional distractor block ``s_d`` with zero
reward relevance. The observation is ``concat(obs_c, s_d)`` and rewards are taken
verbatim from the base task, so the value function is flat in ``s_d``
(``grad_{s_d} V = 0``) by construction.

The distractor block follows the WP1 form
``s_d' = tanh(A s_d + B a + xi)`` with ``xi ~ N(0, sigma_dist^2 I)`` (see
``core.DistractorDynamics``). The dynamics are numpy; the torch model stack that
consumes this suite lives under ``distractor_gym.deep`` so the base package does not
require torch.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .core import DistractorDynamics, RegimeConfig

_BASE_TASKS = {
    "inverted_pendulum": "InvertedPendulum-v4",
    "halfcheetah": "HalfCheetah-v4",
    "walker2d": "Walker2d-v4",
    "pendulum": "Pendulum-v1",
    "cartpole": "CartPole-v1",
}


class DistractorGym(gym.Env):
    """Base control task with an appended, action-coupled distractor block ``s_d``.

    The reward is the base task's reward untouched, so it is invariant to ``s_d`` and
    to the number of distractor dimensions. The distractor noise stream is re-seeded on
    every ``reset`` so an episode is reproducible from its reset seed.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: RegimeConfig, base_task: gym.Env) -> None:
        super().__init__()
        self.config = config
        self.base_task = base_task
        self.d_c = int(np.prod(base_task.observation_space.shape))
        self.d_d = int(config.d_d)
        self.action_dim = int(np.prod(base_task.action_space.shape) or 1)
        self.dynamics = DistractorDynamics(config, action_dim=self.action_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.d_c + self.d_d,), dtype=np.float32
        )
        self.action_space = base_task.action_space
        self._s_d = np.zeros(self.d_d, dtype=np.float64)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_task.step(action)
        self._s_d = self.dynamics.step(self._s_d, np.asarray(action, dtype=float).reshape(-1))
        return self._obs(obs), float(reward), bool(terminated), bool(truncated), info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.base_task.reset(seed=seed, options=options)
        self._s_d = np.zeros(self.d_d, dtype=np.float64)
        self.dynamics.reseed(int(seed) if seed is not None else self.config.seed)
        return self._obs(obs), info

    def _obs(self, obs_c: np.ndarray) -> np.ndarray:
        obs_c = np.asarray(obs_c, dtype=np.float32).reshape(-1)
        return np.concatenate([obs_c, self._s_d.astype(np.float32)]).astype(np.float32)


def make_distractor_gym(
    config: RegimeConfig,
    base_task: str = "pendulum",
) -> DistractorGym:
    """Build a ``DistractorGym`` wrapping a named base task."""
    if base_task not in _BASE_TASKS:
        raise ValueError(f"unknown base task: {base_task}; use {sorted(_BASE_TASKS)}")
    return DistractorGym(config, gym.make(_BASE_TASKS[base_task]))
