"""Tabular Distractor-Gym suite.

Ground-truth lab: exact ``g_true``, ``delta_TD``, ``eps_model`` and ``grad V`` are
computable, enabling exact validation of the weight-estimator theory and the
``|delta_TD| ~ ||grad V|| * eps_model`` decomposition (RESEARCH_PLAN.md Sec. 2, 5).

State ``s = (s_c, s_d)``: ``s_c`` is a 1D control position on ``grid_c`` and ``s_d``
is a ``d_d``-dimensional distractor block on ``grid_d``. Distractor dynamics are a
deterministic chaotic (logistic) map per dimension; reward depends only on ``s_c``,
so ``grad_{s_d} V = 0`` by construction.
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
    """Discretized MDP with a 1D control axis and chaotic distractor dims.

    Actions are ``{-1, 0, +1}`` (move left / stay / move right); with
    ``transition_noise`` the control move is perturbed by a quantized random walk.
    The distractor block evolves deterministically and independently of the action.

    Reward semantics (``reward_sparsity``/``goal_radius``):
    - dense: ``r(x) = exp(-(x - goal)^2 / 2)`` (used when ``goal_radius is None``).
    - sparse: ``r(x) = 1`` iff ``|x - goal| <= radius``, else ``0``, with radius from
      ``goal_radius`` or, if unset, ``reward_sparsity * (high - low) / 2``.
    """

    ACTIONS = np.array([-1, 0, 1])

    def __init__(
        self,
        config: RegimeConfig,
        grid_c: Grid | None = None,
        grid_d: Grid | None = None,
        goal: float = 2.0,
    ) -> None:
        self.config = config
        self.grid_c = grid_c if grid_c is not None else Grid(-3.0, 3.0, 15)
        self.grid_d = grid_d if grid_d is not None else Grid(-3.0, 3.0, 5)
        self.goal = goal
        self.n_c = self.grid_c.n
        self.n_d = self.grid_d.n
        self.d_d = config.d_d
        self.n_di = self.n_d**self.d_d
        self.S = self.n_c * self.n_di
        self.n_a = len(self.ACTIONS)
        self.rng = np.random.default_rng(config.seed)
        self.transition = self._build_transition()
        self.rewards = self._build_rewards()
        self.features = self._build_features()

    def _unpack(self, idx: int | np.ndarray) -> tuple:
        i = np.asarray(idx) // self.n_di
        j = np.asarray(idx) % self.n_di
        return i, j

    def _pack(self, i: int, j: int) -> int:
        return int(i) * self.n_di + int(j)

    def control_coord(self, idx: int) -> float:
        i, _ = self._unpack(idx)
        return float(self.grid_c.points[i])

    def distractor_next_index(self, j: int) -> int:
        """Logistic map on each distractor dimension, quantized onto ``grid_d``."""
        if self.d_d == 0:
            return 0
        coords = np.unravel_index(int(j), [self.n_d] * self.d_d)
        nxt = []
        for k in range(self.d_d):
            p = (self.grid_d.points[coords[k]] - self.grid_d.low) / (
                self.grid_d.high - self.grid_d.low
            )
            p = 4.0 * p * (1.0 - p)
            p2 = self.grid_d.low + p * (self.grid_d.high - self.grid_d.low)
            nxt.append(int(np.argmin(np.abs(self.grid_d.points - p2))))
        return int(np.ravel_multi_index(tuple(nxt), [self.n_d] * self.d_d))

    def _control_next(self, i: int, a: int, noise: float) -> dict[int, float]:
        target = int(np.clip(i + a, 0, self.n_c - 1))
        if noise <= 0.0:
            return {target: 1.0}
        p = min(float(noise), 0.49)
        dist = {target: 1.0 - p}
        for step in (-1, 1):
            nb = int(np.clip(target + step, 0, self.n_c - 1))
            dist[nb] = dist.get(nb, 0.0) + p / 2.0
        return dist

    def _build_transition(self) -> np.ndarray:
        P = np.zeros((self.S, self.n_a, self.S))
        noise = self.config.transition_noise
        for i in range(self.n_c):
            for j in range(self.n_di):
                s0 = self._pack(i, j)
                j2 = self.distractor_next_index(j)
                for a_idx, a in enumerate(self.ACTIONS):
                    for i2, pr in self._control_next(i, a, noise).items():
                        P[s0, a_idx, self._pack(i2, j2)] = pr
        return P

    def _build_rewards(self) -> np.ndarray:
        xc = self.grid_c.points
        radius = self.config.goal_radius
        if radius is None and self.config.reward_sparsity >= 1.0:
            r = np.exp(-0.5 * (xc - self.goal) ** 2)
        else:
            if radius is None:
                radius = self.config.reward_sparsity * (self.grid_c.high - self.grid_c.low) / 2.0
            r = (np.abs(xc - self.goal) <= radius).astype(float)
        return np.repeat(r, self.n_di)

    def _build_features(self) -> np.ndarray:
        x = np.repeat(self.grid_c.points, self.n_di)
        return np.column_stack([np.ones(self.S), x, x**2])

    def value_iteration(self, gamma: float = 0.99, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
        """Exact value function under the true transition via value iteration."""
        V = np.zeros(self.S)
        P, r = self.transition, self.rewards
        for _ in range(max_iter):
            Q = r[:, None] + gamma * np.tensordot(P, V, axes=([2], [0]))
            V_new = np.max(Q, axis=1)
            if np.max(np.abs(V_new - V)) < tol:
                return V_new
            V = V_new
        return V

    def evaluate_v(self, P: np.ndarray, policy_probs: np.ndarray, gamma: float = 0.99) -> np.ndarray:
        """Value of ``policy_probs`` (S, A) under transition ``P`` by exact linear solve."""
        P_pi = np.einsum("sa,san->sn", policy_probs, P)
        A = np.eye(self.S) - gamma * P_pi
        return np.linalg.solve(A, self.rewards)

    def value_grad(self, V: np.ndarray) -> np.ndarray:
        """Finite-difference gradient ``grad_s V`` on the grid; zero in distractor dims.

        Returns an array of shape ``(S, 1 + d_d)``: central differences along the
        control axis, one-sided at boundaries, and zeros along the distractor block.
        """
        grad = np.zeros((self.S, 1 + self.d_d))
        Vc = V.reshape(self.n_c, self.n_di).mean(axis=1)
        dx = self.grid_c.dx
        for i in range(self.n_c):
            if 0 < i < self.n_c - 1:
                g = (Vc[i + 1] - Vc[i - 1]) / (2 * dx)
            elif i == 0:
                g = (Vc[1] - Vc[0]) / dx
            else:
                g = (Vc[-1] - Vc[-2]) / dx
            grad[i * self.n_di : (i + 1) * self.n_di, 0] = g
        return grad

    def td_error(self, V: np.ndarray, s_prime: int, s_hat_prime: int) -> float:
        """Model-induced bootstrap-target error ``|V(s') - V(s_hat')|``."""
        return float(abs(V[s_prime] - V[s_hat_prime]))

    def transition_error(self, s_prime: np.ndarray, s_hat_prime: np.ndarray) -> float:
        """One-step model prediction error ``eps_model = ||s_hat' - s'||`` in coordinates."""
        a = self.coordinates(s_prime)
        b = self.coordinates(s_hat_prime)
        return float(np.linalg.norm(a - b))

    def coordinates(self, idx: int) -> np.ndarray:
        """Continuous coordinate vector of a state index: control position then distractor dims."""
        i, j = self._unpack(idx)
        coord = [float(self.grid_c.points[i])]
        if self.d_d > 0:
            coord += [float(self.grid_d.points[k]) for k in np.unravel_index(int(j), [self.n_d] * self.d_d)]
        return np.array(coord)