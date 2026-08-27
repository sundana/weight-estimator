import numpy as np
import pytest

from distractor_gym.agents import (
    SoftmaxLinearPolicy,
    collect_transitions,
    fit_empirical_model,
    goal_policy,
    induced_value_error,
    policy_gradient,
    uniform_behavior,
)
from distractor_gym.core import RegimeConfig
from distractor_gym.tabular import TabularDistractorEnv


def make_env(**kw):
    return TabularDistractorEnv(RegimeConfig(d_d=kw.pop("d_d", 0), transition_noise=kw.pop("noise", 0.1), **kw))


def test_policy_gradient_matches_finite_difference():
    env = make_env()
    theta = np.array([[1.0, -0.5, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.5, 0.0]])
    pi = SoftmaxLinearPolicy(env.features, env.n_a, theta)
    g = policy_gradient(env, env.transition, pi)

    def J(t):
        p = SoftmaxLinearPolicy(env.features, env.n_a, t)
        V = env.evaluate_v(env.transition, p.probs())
        return float(V @ np.full(env.S, 1 / env.S))

    eps = 1e-4
    num = np.zeros_like(theta)
    for a in range(env.n_a):
        for k in range(env.features.shape[1]):
            tp = theta.copy()
            tm = theta.copy()
            tp[a, k] += eps
            tm[a, k] -= eps
            num[a, k] = (J(tp) - J(tm)) / (2 * eps)
    assert np.abs(g - num).max() < 1e-4


def test_goal_policy_is_nonuniform():
    env = make_env()
    pi = goal_policy(env)
    assert np.allclose(pi.probs().sum(axis=1), 1.0)
    assert np.abs(pi.probs().max(axis=1) - 1 / env.n_a).max() > 0.1


def test_goal_policy_gradient_nonzero():
    env = make_env()
    pi = goal_policy(env)
    g = policy_gradient(env, env.transition, pi)
    assert np.linalg.norm(g) > 0


def test_fit_empirical_model_is_stochastic():
    env = make_env()
    rng = np.random.default_rng(0)
    data = collect_transitions(env, 2000, uniform_behavior(env), rng)
    P = fit_empirical_model(env, data)
    assert np.allclose(P.sum(axis=2), 1.0)
    assert (P >= 0).all()


def test_fit_empirical_model_mle_equals_counts():
    env = make_env(d_d=0, noise=0.0)
    rng = np.random.default_rng(0)
    data = collect_transitions(env, 5000, uniform_behavior(env), rng)
    P = fit_empirical_model(env, data, None, alpha=0.0)
    counts = np.zeros_like(P)
    np.add.at(counts, (data[:, 0], data[:, 1], data[:, 2]), 1)
    visited = counts.sum(axis=2) > 0
    assert np.allclose(P[visited], counts[visited] / counts[visited].sum(axis=1, keepdims=True))


def test_induced_value_error_nonnegative():
    env = make_env()
    rng = np.random.default_rng(0)
    data = collect_transitions(env, 1000, uniform_behavior(env), rng)
    P = fit_empirical_model(env, data)
    V = env.value_iteration()
    err = induced_value_error(env, P, V, data)
    assert err >= 0.0


def test_collect_transitions_shape():
    env = make_env()
    rng = np.random.default_rng(0)
    data = collect_transitions(env, 100, uniform_behavior(env), rng)
    assert data.shape == (100, 3)
    assert (data[:, 0] < env.S).all() and (data[:, 2] < env.S).all()