"""Exact tabular RL machinery: policies, policy gradients, model fitting, rollouts.

All quantities are computed exactly from transition tensors, making the tabular suite
a ground-truth lab for the weight-estimator theory (RESEARCH_PLAN.md Sec. 2, 5).
"""

from __future__ import annotations

import numpy as np

from .tabular import TabularDistractorEnv


class SoftmaxLinearPolicy:
    """Softmax policy ``pi(a | s) ~ exp(theta_a . phi(s))`` over linear features."""

    def __init__(self, features: np.ndarray, n_actions: int, theta: np.ndarray | None = None) -> None:
        self.features = np.asarray(features, dtype=float)
        self.n_actions = n_actions
        self.d_feat = self.features.shape[1]
        if theta is None:
            theta = np.zeros((n_actions, self.d_feat))
        self.theta = np.asarray(theta, dtype=float)

    def logits(self) -> np.ndarray:
        return self.features @ self.theta.T

    def probs(self) -> np.ndarray:
        z = self.logits()
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def act(self, s: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.n_actions, p=self.probs()[s]))


def centered_policy(env: TabularDistractorEnv, center: float, gain: float = 2.0) -> SoftmaxLinearPolicy:
    """Policy steering toward ``center``: move right when left of it, left when right.

    ``logit(+1) = gain * (center - x)`` and ``logit(-1) = -logit(+1)`` with features
    ``[1, x, x^2]``, giving a non-uniform softmax with a non-trivial gradient.
    """
    theta = np.zeros((env.n_a, env.features.shape[1]))
    theta[env.n_a - 1] = [gain * center, -gain, 0.0]  # action +1
    theta[0] = [-gain * center, gain, 0.0]  # action -1
    return SoftmaxLinearPolicy(env.features, env.n_a, theta)


def goal_policy(env: TabularDistractorEnv, gain: float = 2.0) -> SoftmaxLinearPolicy:
    """Policy steering toward the environment goal."""
    return centered_policy(env, center=env.goal, gain=gain)


def policy_gradient(
    env: TabularDistractorEnv,
    P: np.ndarray,
    policy: SoftmaxLinearPolicy,
    gamma: float = 0.99,
    d0: np.ndarray | None = None,
) -> np.ndarray:
    """Exact policy gradient under transition ``P`` via the policy gradient theorem.

    ``g = sum_s mu(s) sum_a Q^pi(s, a) grad_theta pi(a | s)`` with the discounted
    unnormalized occupancy ``mu = (I - gamma P_pi^T)^-1 d0``.
    """
    probs = policy.probs()
    P_pi = np.einsum("sa,san->sn", probs, P)
    V = env.evaluate_v(P, probs, gamma)
    Q = env.rewards[:, None] + gamma * np.tensordot(P, V, axes=([2], [0]))
    if d0 is None:
        d0 = np.full(env.S, 1.0 / env.S)
    mu = np.linalg.solve(np.eye(env.S) - gamma * P_pi.T, d0)
    g = np.zeros((policy.n_actions, policy.d_feat))
    for a in range(policy.n_actions):
        g[a] += (mu * probs[:, a] * Q[:, a] * (1.0 - probs[:, a])) @ policy.features
        for b in range(policy.n_actions):
            if b != a:
                g[a] -= (mu * probs[:, b] * Q[:, b] * probs[:, a]) @ policy.features
    return g


def fit_empirical_model(
    env: TabularDistractorEnv,
    data: np.ndarray,
    weights: np.ndarray | None = None,
    alpha: float = 0.1,
) -> np.ndarray:
    """Weighted empirical (Laplace-smoothed) transition model from a transition batch.

    ``data`` is an ``(n, 3)`` array of ``(s, a, s')``; ``weights`` the per-sample
    weight (MLE if ``None``). Returns ``P_hat`` of shape ``(S, n_a, S)``.
    """
    s, a, sp = data[:, 0], data[:, 1], data[:, 2]
    if weights is None:
        weights = np.ones(len(s))
    P = np.full((env.S, env.n_a, env.S), alpha, dtype=float)
    np.add.at(P, (s, a, sp), weights)
    P = P / P.sum(axis=2, keepdims=True)
    return P


def mode_prediction(P_hat: np.ndarray) -> np.ndarray:
    """Most likely next state ``s_hat'`` per ``(s, a)``."""
    return np.argmax(P_hat, axis=2)


def collect_transitions(
    env: TabularDistractorEnv,
    n_steps: int,
    behavior_probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Collect an ``(n, 3)`` transition batch by rolling out one behavior trajectory."""
    out = np.empty((n_steps, 3), dtype=np.int64)
    s = int(rng.choice(env.S))
    for t in range(n_steps):
        a = int(rng.choice(env.n_a, p=behavior_probs[s]))
        sp = int(rng.choice(env.S, p=env.transition[s, a]))
        out[t] = (s, a, sp)
        s = sp
    return out


def sample_transitions(
    env: TabularDistractorEnv,
    n: int,
    behavior_probs: np.ndarray,
    rng: np.random.Generator,
    state_probs: np.ndarray | None = None,
) -> np.ndarray:
    """Draw an i.i.d. ``(n, 3)`` batch of ``(s, a, s')`` transitions.

    Unlike ``collect_transitions`` (a single random walk), this samples states
    independently from ``state_probs`` (uniform over states by default), so the
    coverage knob controls the state distribution exactly rather than through the
    mixing of one trajectory.
    """
    if state_probs is None:
        state_probs = np.full(env.S, 1.0 / env.S)
    s = rng.choice(env.S, size=n, p=state_probs)
    a = np.array([rng.choice(env.n_a, p=behavior_probs[si]) for si in s], dtype=np.int64)
    u = rng.random(n)
    cum = np.cumsum(env.transition[s, a], axis=1)
    sp = (cum < u[:, None]).sum(axis=1)
    sp = np.minimum(sp, env.S - 1)
    return np.stack([s, a, sp], axis=1).astype(np.int64)


def fit_feature_model(
    env: TabularDistractorEnv,
    data: np.ndarray,
    weights: np.ndarray | None = None,
    capacity: int = 1,
    model_noise: float = 0.5,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Capacity-limited (reduced-rank) transition model fit by weighted least squares.

    Fits ``mu(s, a) = phi(s, a) C`` with ``rank(C) <= capacity`` to the observed
    next-state coordinates, where ``phi = [s_coords, a, 1]``. The restored transition
    tensor is a fixed-bandwidth Gaussian over the grid coordinates,
    ``P_hat(s' | s, a) ~ exp(-||coord(s') - mu(s, a)||^2 / (2 model_noise^2))``.

    A small ``capacity`` relative to the state dimension forces the model to spend
    its representational budget on the highest-variance state directions (the
    distractors), degrading control-relevant predictions -- the tabular analogue
    of a finite-capacity neural model.
    """
    s, a, sp = data[:, 0], data[:, 1], data[:, 2]
    D = 1 + env.d_d
    phi = np.column_stack([env.coordinates_batch(s), a.astype(float), np.ones(len(s))])
    Y = env.coordinates_batch(sp)
    if weights is None:
        w = np.ones(len(data))
    else:
        w = np.asarray(weights, dtype=float)
    sw = np.sqrt(np.maximum(w, 0.0))[:, None]
    Xw = sw * phi
    Yw = sw * Y
    p = Xw.shape[1]
    G = Xw.T @ Xw + ridge * np.eye(p)
    H = Xw.T @ Yw
    evals, evecs = np.linalg.eigh(G)
    Ginv_sqrt = (evecs / np.sqrt(evals)) @ evecs.T
    M = Ginv_sqrt @ H
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = int(max(1, min(capacity, len(S))))
    C = Ginv_sqrt @ (U[:, :r] * S[:r]) @ Vt[:r]

    idx = np.arange(env.S)
    coords = env.coordinates_batch(idx)
    phi_all = np.zeros((env.S * env.n_a, p))
    act = np.repeat(env.ACTIONS.astype(float), env.S)
    for a_idx in range(env.n_a):
        block = slice(a_idx * env.S, (a_idx + 1) * env.S)
        phi_all[block, :D] = coords
        phi_all[block, D] = env.ACTIONS[a_idx]
        phi_all[block, D + 1] = 1.0
    mu_all = phi_all @ C
    dist2 = ((mu_all[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    logits = -dist2 / (2.0 * model_noise**2)
    logits = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P = P / P.sum(axis=1, keepdims=True)
    return P.reshape(env.S, env.n_a, env.S)


def uniform_behavior(env: TabularDistractorEnv) -> np.ndarray:
    """Uniform random-action behavior policy."""
    return np.full((env.S, env.n_a), 1.0 / env.n_a)


def induced_value_error(
    env: TabularDistractorEnv, P_hat: np.ndarray, V: np.ndarray, data: np.ndarray
) -> float:
    """Value-aware model risk: mean ``|V(s') - E_{P_hat}[V]|`` over transitions.

    Measures whether the model predicts value-relevant outcomes accurately,
    independent of any policy (RESEARCH_PLAN.md Sec. 5 Part B).
    """
    E_V = np.tensordot(P_hat, V, axes=([2], [0]))
    s, a, sp = data[:, 0], data[:, 1], data[:, 2]
    return float(np.mean(np.abs(V[sp] - E_V[s, a])))