"""Differentiable analytic Distractor-Gym for the WP1 deep gradient diagnostic.

MuJoCo dynamics are not differentiable, so the ``g_true`` of Exp 1.1 cannot be
obtained by back-propagating through the real environment. This module supplies an
analytic, differentiable stand-in with the same structure as the MuJoCo suite: a
2D control block ``(x, v)`` (damped double integrator stabilizing a goal) plus a
``d_d``-dimensional action-coupled distractor block ``s_d' = tanh(A s_d + B a + xi)``
with zero reward relevance. The true dynamics are exact here, so ``g_true`` is the
exact analytic rollout gradient and ``g_model`` is the ensemble rollout gradient.
``AnalyticDistractorGym`` exposes the same system to the SAC data generator.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from .nets import GaussianEnsemble


class AnalyticDistractorEnv:
    """Differentiable ground-truth Distractor-Gym (control block ``d_c = 2``)."""

    def __init__(
        self,
        d_d: int = 0,
        sigma_dist: float = 0.0,
        seed: int = 0,
        dt: float = 0.1,
        goal: float = 0.0,
        damping: float = 0.5,
        control_cost: float = 0.001,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.d_c = 2
        self.d_d = int(d_d)
        self.dim = self.d_c + self.d_d
        self.sigma_dist = float(sigma_dist)
        self.dt = float(dt)
        self.goal = float(goal)
        self.damping = float(damping)
        self.control_cost = float(control_cost)
        self.dtype = dtype
        if self.d_d > 0:
            gen = torch.Generator().manual_seed(int(seed) + 101)
            self.A = torch.randn(self.d_d, self.d_d, generator=gen, dtype=dtype) / np.sqrt(self.d_d)
            self.B = torch.randn(self.d_d, 1, generator=gen, dtype=dtype)
        else:
            self.A = torch.zeros(0, 0, dtype=dtype)
            self.B = torch.zeros(0, 1, dtype=dtype)

    def reward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Dense control reward; depends only on the control block, shape ``(batch,)``."""
        x = s[..., 0]
        v = s[..., 1]
        act = a[..., 0] if a.dim() > 1 else a
        return torch.exp(-0.5 * ((x - self.goal) ** 2 + v**2)) - self.control_cost * act**2

    def step(self, s: torch.Tensor, a: torch.Tensor, noise: bool = False) -> torch.Tensor:
        """One differentiable step of the true dynamics (noise optional)."""
        x = s[..., 0]
        v = s[..., 1]
        act = a[..., 0] if a.dim() > 1 else a
        x2 = x + self.dt * v
        v2 = v + self.dt * (act - self.damping * v)
        if self.d_d == 0:
            return torch.stack([x2, v2], dim=-1)
        s_d = s[..., self.d_c :]
        drive = s_d @ self.A.T + act.unsqueeze(-1) * self.B.T
        if noise and self.sigma_dist > 0.0:
            drive = drive + self.sigma_dist * torch.randn_like(drive)
        s_d2 = torch.tanh(drive)
        return torch.cat([torch.stack([x2, v2], dim=-1), s_d2], dim=-1)

    def sample_states(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Random initial states: ``x, v ~ U(-2, 2) x U(-1, 1)``, distractors zero."""
        x = torch.rand(n, generator=generator, dtype=self.dtype) * 4.0 - 2.0
        v = torch.rand(n, generator=generator, dtype=self.dtype) * 2.0 - 1.0
        zeros = torch.zeros(n, self.d_d, dtype=self.dtype)
        return torch.cat([torch.stack([x, v], dim=-1), zeros], dim=-1)


def rollout_return(
    env: AnalyticDistractorEnv,
    actor,
    s0: torch.Tensor,
    horizon: int,
    gamma: float,
    model: GaussianEnsemble | None = None,
    x_stats: tuple | None = None,
    y_stats: tuple | None = None,
) -> torch.Tensor:
    """Discounted return of the deterministic policy, differentiated through the rollout.

    ``model=None`` uses the exact analytic dynamics (``g_true``); otherwise the model
    ensemble's differentiable mean is used (``g_model``).
    """
    s = s0
    total = torch.zeros(s0.shape[0], dtype=s0.dtype, device=s0.device)
    for t in range(horizon):
        a = actor.deterministic(s)
        total = total + (gamma**t) * env.reward(s, a)
        if model is None:
            s = env.step(s, a, noise=False)
        else:
            s = _model_mean(model, s, a, x_stats, y_stats)
    return total.mean()


def _model_mean(model, s, a, x_stats, y_stats) -> torch.Tensor:
    x = torch.cat([s, a], dim=-1)
    x_mean, x_std = x_stats
    y_mean, y_std = y_stats
    pred_n, _ = model((x - x_mean) / x_std)
    return pred_n.mean(dim=1) * y_std + y_mean


class AnalyticDistractorGym(gym.Env):
    """Gymnasium view of ``AnalyticDistractorEnv`` for the SAC data generator."""

    metadata = {"render_modes": []}

    def __init__(self, d_d: int = 0, sigma_dist: float = 0.0, seed: int = 0, horizon: int = 100) -> None:
        super().__init__()
        self.env = AnalyticDistractorEnv(d_d=d_d, sigma_dist=sigma_dist, seed=seed)
        self.horizon = int(horizon)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._s = torch.zeros(self.env.dim)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            torch.manual_seed(int(seed))
        gen = torch.Generator().manual_seed(int(seed) if seed is not None else 0)
        self._s = self.env.sample_states(1, generator=gen)[0].detach()
        self._t = 0
        return self._s.numpy().astype(np.float32), {}

    def step(self, action):
        a = torch.as_tensor(np.asarray(action, dtype=np.float32).reshape(1))
        with torch.no_grad():
            self._s = self.env.step(self._s.unsqueeze(0), a.unsqueeze(0), noise=True)[0]
            r = float(self.env.reward(self._s.unsqueeze(0), a.unsqueeze(0))[0])
        self._t += 1
        truncated = self._t >= self.horizon
        return self._s.numpy().astype(np.float32), r, False, truncated, {}
