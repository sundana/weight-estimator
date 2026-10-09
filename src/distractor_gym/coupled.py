"""Non-stationary coupled model/critic lab (Part II, Phase A).

Ground-truth tabular system for the COVAL coupling: a transition model ``theta`` (the
predicted next-state coordinates per ``(s, a)``) and a state-value critic ``phi`` are
updated simultaneously under constant step sizes. The critic supplies the value-aware
weights to the model loss (``w = w(phi)``), and the model supplies the Bellman target to
the critic (``E_{P_theta}[phi_bar]``), forming the weight-estimator feedback loop
``phi -> w -> theta -> P_theta -> phi`` analyzed in ``paper/coval/notes/coval_scope.md``.

The class exposes the exact update map, its finite-difference Jacobian, and the
tracking-error / Lyapunov series used to locate the stability boundary ``kappa`` of the
learning-rate ratio ``alpha_model / alpha_critic`` and to trace how that boundary moves
with the weight-estimator noise ``sigma_w^2`` (Phase A of
``configs/exp2_coupled_lab.yaml``).

The critic is the tabular state-value under a uniform action average, so the Bellman
target is ``r(s) + gamma * E_{P_theta}[phi_bar](s)`` with the semi-gradient TD(0) update;
this deliberately couples ``theta`` into ``phi`` and ``phi`` into the model weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tabular import TabularDistractorEnv

_WEIGHT_KINDS = ("mle", "vaml1", "vagram")
_NORMALIZATIONS = ("none", "min_max", "batch_self")


@dataclass
class CoupledConfig:
    """Knobs of the tabular coupled model/critic system."""

    gamma: float = 0.99
    alpha_model: float = 2e-2
    alpha_critic: float = 2e-2
    polyak_tau: float = 1.0
    weight_kind: str = "mle"
    weight_noise: float = 0.0
    self_norm: bool = False
    normalization: str = "none"
    clip_wmax: float | None = None
    clip_median_ratio: float | None = None
    spectral_norm: bool = False
    lip_target: float | None = None
    anneal_steps: int = 0
    bandwidth: float = 0.5
    n_steps: int = 500
    seed: int = 0


class TabularCoupledSystem:
    """Simultaneous model/critic updates with exact, testable coupling.

    ``theta`` has shape ``(S * n_a, D)`` with ``D = 1 + d_d`` predicted next-state
    coordinates; ``phi`` and the Polyak average ``phi_bar`` have shape ``(S,)``. The
    packed state is ``[theta.ravel(), phi, phi_bar]``.
    """

    def __init__(self, env: TabularDistractorEnv, data: np.ndarray, cfg: CoupledConfig) -> None:
        if cfg.weight_kind not in _WEIGHT_KINDS:
            raise ValueError(f"unknown weight_kind: {cfg.weight_kind}")
        if cfg.normalization not in _NORMALIZATIONS:
            raise ValueError(f"unknown normalization: {cfg.normalization}")
        self.env = env
        self.cfg = cfg
        self.data = np.asarray(data, dtype=np.int64)
        self.n = len(self.data)
        self.S = env.S
        self.A = env.n_a
        self.D = 1 + env.d_d
        self.m = self.S * self.A * self.D
        self.state_dim = self.m + 2 * self.S
        self.x_next = env.coordinates_batch(self.data[:, 2])

        rng = np.random.default_rng(cfg.seed + 7)
        theta = np.zeros((self.S, self.A, self.D))
        counts = np.zeros((self.S, self.A, 1))
        np.add.at(theta, (self.data[:, 0], self.data[:, 1]), self.x_next)
        np.add.at(counts, (self.data[:, 0], self.data[:, 1]), 1.0)
        theta = theta / np.maximum(counts, 1.0)
        self.theta0 = theta.reshape(self.m, self.D)
        self.phi0 = 0.1 * rng.normal(size=self.S)
        self.phi_bar0 = self.phi0.copy()

    def initial_state(self) -> np.ndarray:
        """Packed initial state ``[theta0, phi0, phi_bar0]``."""
        return np.concatenate([self.theta0.reshape(-1), self.phi0, self.phi_bar0])

    def _unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = x[: self.m].reshape(self.m, self.D)
        phi = x[self.m : self.m + self.S]
        phi_bar = x[self.m + self.S :]
        return theta, phi, phi_bar

    def _model_transition(self, theta: np.ndarray) -> np.ndarray:
        """Action-averaged model transition ``P_theta(s' | s)`` from Gaussian bandwidth."""
        mu = theta.reshape(self.S, self.A, self.D)
        coords = self.env.coordinates_batch(np.arange(self.S))
        diff = coords[None, None, :, :] - mu[:, :, None, :]
        dist2 = np.sum(diff**2, axis=-1)
        logits = -dist2 / (2.0 * self.cfg.bandwidth**2)
        logits = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        return probs.mean(axis=1)

    def _weights(self, phi: np.ndarray, noise_vec: np.ndarray | None, lam: float) -> np.ndarray:
        """Per-transition value-aware weights ``w(phi)`` with noise, normalization, clipping."""
        s, sp = self.data[:, 0], self.data[:, 2]
        if self.cfg.weight_kind == "mle":
            w = np.ones(self.n)
        elif self.cfg.weight_kind == "vaml1":
            w = np.abs(phi[sp] - phi[s])
        else:
            grad = self.env.value_grad(phi)
            w = np.linalg.norm(grad[sp], axis=1)
        if noise_vec is not None:
            w = np.maximum(w + noise_vec, 0.0)
        w = self._normalize_weights(w)
        if self.cfg.clip_wmax is not None:
            w = np.minimum(w, self.cfg.clip_wmax)
        elif self.cfg.clip_median_ratio is not None:
            w = np.minimum(w, self.cfg.clip_median_ratio * (np.median(w) + 1e-12))
        if self.cfg.anneal_steps > 0:
            w = (1.0 - lam) + lam * w
        return w

    def _normalize_weights(self, w: np.ndarray) -> np.ndarray:
        """Apply the configured weight normalization.

        ``none`` leaves the weights unchanged; ``batch_self`` divides by the batch mean
        (fixing the effective step size); ``min_max`` rescales into ``[0, 1]`` (bounding
        the dynamic range). ``self_norm=True`` is kept as an alias of ``batch_self``.
        """
        mode = self.cfg.normalization
        if mode == "none" and self.cfg.self_norm:
            mode = "batch_self"
        if mode == "none":
            return w
        if mode == "batch_self":
            return w / (w.mean() + 1e-8)
        lo, hi = float(w.min()), float(w.max())
        return (w - lo) / (hi - lo + 1e-8)

    def critic_lipschitz(self, phi: np.ndarray) -> float:
        """Empirical Lipschitz constant of the critic: max ``||grad_s V_phi||`` over states."""
        grad = self.env.value_grad(phi)
        return float(np.max(np.linalg.norm(grad, axis=1)))

    def _project_lipschitz(self, phi: np.ndarray) -> np.ndarray:
        """Rescale the critic about its mean to cap the Lipschitz constant at ``lip_target``."""
        target = self.cfg.lip_target
        lip = self.critic_lipschitz(phi)
        if target is not None and lip > target > 0.0:
            center = phi.mean()
            return center + (phi - center) * (target / lip)
        return phi

    def _model_grad(self, theta: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Gradient of the weighted regression loss w.r.t. ``theta``."""
        mu = theta.reshape(self.S, self.A, self.D)
        pred = mu[self.data[:, 0], self.data[:, 1]]
        resid = pred - self.x_next
        g = np.zeros((self.S, self.A, self.D))
        np.add.at(g, (self.data[:, 0], self.data[:, 1]), (2.0 / self.n) * w[:, None] * resid)
        return g.reshape(self.m, self.D)

    def _critic_target(self, theta: np.ndarray, phi_bar: np.ndarray) -> np.ndarray:
        """Semi-gradient Bellman target ``r + gamma * E_{P_theta}[phi_bar]``."""
        return self.env.rewards + self.cfg.gamma * (self._model_transition(theta) @ phi_bar)

    def _critic_grad(self, phi: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (2.0 / self.S) * (phi - target)

    def _step_with_info(
        self, x: np.ndarray, noise_vec: np.ndarray | None, lam: float
    ) -> tuple[np.ndarray, dict]:
        theta, phi, phi_bar = self._unpack(x)
        w = self._weights(phi, noise_vec, lam)
        g_theta = self._model_grad(theta, w)
        target = self._critic_target(theta, phi_bar)
        g_phi = self._critic_grad(phi, target)
        theta_next = theta - self.cfg.alpha_model * g_theta
        phi_next = phi - self.cfg.alpha_critic * g_phi
        if self.cfg.spectral_norm:
            phi_next = self._project_lipschitz(phi_next)
        if self.cfg.polyak_tau < 1.0:
            phi_bar_next = self.cfg.polyak_tau * phi_next + (1.0 - self.cfg.polyak_tau) * phi_bar
        else:
            phi_bar_next = phi_next
        x_next = np.concatenate([theta_next.reshape(-1), phi_next, phi_bar_next])
        info = {
            "weight_var": float(np.var(w)),
            "weight_mean": float(np.mean(w)),
            "g_theta_norm": float(np.linalg.norm(g_theta)),
            "g_phi_norm": float(np.linalg.norm(g_phi)),
            "critic_lip": self.critic_lipschitz(phi_next),
        }
        return x_next, info

    def step(self, x: np.ndarray, noise_vec: np.ndarray | None = None, lam: float = 1.0) -> np.ndarray:
        """One simultaneous model/critic update of the packed state."""
        return self._step_with_info(x, noise_vec, lam)[0]

    def rollout(self, n_steps: int | None = None) -> dict:
        """Run the coupled iteration and record the state and diagnostic series."""
        n_steps = self.cfg.n_steps if n_steps is None else n_steps
        rng = np.random.default_rng(self.cfg.seed + 12345)
        x = self.initial_state()
        states = np.empty((n_steps + 1, self.state_dim))
        states[0] = x
        weight_var = np.empty(n_steps)
        weight_mean = np.empty(n_steps)
        g_theta_norm = np.empty(n_steps)
        g_phi_norm = np.empty(n_steps)
        critic_lip = np.empty(n_steps)
        diverged = False
        for t in range(n_steps):
            lam = min(1.0, (t + 1) / self.cfg.anneal_steps) if self.cfg.anneal_steps > 0 else 1.0
            noise = (
                rng.normal(0.0, self.cfg.weight_noise, size=self.n)
                if self.cfg.weight_noise > 0.0
                else None
            )
            x, info = self._step_with_info(x, noise, lam)
            states[t + 1] = x
            weight_var[t] = info["weight_var"]
            weight_mean[t] = info["weight_mean"]
            g_theta_norm[t] = info["g_theta_norm"]
            g_phi_norm[t] = info["g_phi_norm"]
            critic_lip[t] = info["critic_lip"]
            if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e8:
                diverged = True
                states[t + 1 :] = np.nan
                weight_var[t + 1 :] = np.nan
                weight_mean[t + 1 :] = np.nan
                g_theta_norm[t + 1 :] = np.nan
                g_phi_norm[t + 1 :] = np.nan
                critic_lip[t + 1 :] = np.nan
                break
        return {
            "states": states,
            "weight_var": weight_var,
            "weight_mean": weight_mean,
            "g_theta_norm": g_theta_norm,
            "g_phi_norm": g_phi_norm,
            "critic_lip": critic_lip,
            "diverged": diverged,
        }

    def jacobian(self, x: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
        """Central finite-difference Jacobian of the deterministic update map at ``x``."""
        x = self.initial_state() if x is None else np.asarray(x, dtype=float)
        d = len(x)
        J = np.zeros((d, d))
        for j in range(d):
            xp = x.copy()
            xm = x.copy()
            xp[j] += eps
            xm[j] -= eps
            J[:, j] = (self.step(xp) - self.step(xm)) / (2.0 * eps)
        return J

    def spectral_radius(self, x: np.ndarray | None = None, eps: float = 1e-6) -> float:
        """Largest eigenvalue modulus of the update-map Jacobian."""
        J = self.jacobian(x, eps=eps)
        if not np.all(np.isfinite(J)):
            return float("inf")
        return float(np.max(np.abs(np.linalg.eigvals(J))))

    def tracking_error(self, states: np.ndarray, k: int = 1) -> np.ndarray:
        """Parameter distance ``||x_t - x_{t-k}||`` along a recorded trajectory."""
        if k < 1:
            raise ValueError("k must be >= 1")
        return np.linalg.norm(states[k:] - states[:-k], axis=1)

    def lyapunov_series(self, states: np.ndarray, reference: np.ndarray, gamma_lyap: float = 1.0) -> np.ndarray:
        """Lyapunov candidate ``0.5*||theta-theta*||^2 + (gamma/2)*||phi-phi*||^2``."""
        theta = states[:, : self.m]
        phi = states[:, self.m : self.m + self.S]
        theta_ref = reference[: self.m]
        phi_ref = reference[self.m : self.m + self.S]
        return 0.5 * np.sum((theta - theta_ref) ** 2, axis=1) + 0.5 * gamma_lyap * np.sum(
            (phi - phi_ref) ** 2, axis=1
        )

    def reference_state(self, alpha_scale: float = 1e-3, n_steps: int = 4000) -> np.ndarray:
        """Converged fixed point estimated by a slow deterministic run of the same map."""
        scale_model = self.cfg.alpha_model
        scale_critic = self.cfg.alpha_critic
        self.cfg.alpha_model = scale_model * alpha_scale
        self.cfg.alpha_critic = scale_critic * alpha_scale
        try:
            out = self.rollout(n_steps=n_steps)
        finally:
            self.cfg.alpha_model = scale_model
            self.cfg.alpha_critic = scale_critic
        return out["states"][-1]
