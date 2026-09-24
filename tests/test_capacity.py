import numpy as np
import pytest

from distractor_gym.agents import (
    fit_empirical_model,
    fit_feature_model,
    mode_prediction,
    sample_transitions,
    uniform_behavior,
)
from distractor_gym.core import DistractorClass, RegimeConfig
from distractor_gym.tabular import TabularDistractorEnv


def make_env(d_d=1, distractor_class="nonlinear", transition_noise=0.1, seed=0):
    return TabularDistractorEnv(
        RegimeConfig(
            d_d=d_d,
            distractor_class=DistractorClass(distractor_class),
            transition_noise=transition_noise,
            seed=seed,
        )
    )


def mean_cos_phi(env, P, data):
    V = env.evaluate_v(env.transition, np.full((env.S, env.n_a), 1.0 / env.n_a))
    grad = env.value_grad(V)
    s_hat = mode_prediction(P)[data[:, 0], data[:, 1]]
    err = env.coordinates_batch(s_hat) - env.coordinates_batch(data[:, 2])
    eps = np.linalg.norm(err, axis=1)
    gn = np.linalg.norm(grad[data[:, 2]], axis=1)
    nz = eps > 1e-9
    cos = np.zeros(len(data))
    cos[nz] = np.einsum("ij,ij->i", grad[data[:, 2]][nz], err[nz]) / (gn[nz] * eps[nz])
    return float(np.abs(cos[nz]).mean())


def test_stochastic_distractor_is_not_deterministic():
    env = make_env(d_d=1, distractor_class="stochastic")
    row = env.transition[0, 0]
    assert row.sum() == pytest.approx(1.0)
    assert (row > 1e-9).sum() > 1


def test_deterministic_distractor_has_single_next_state():
    env = make_env(d_d=1, distractor_class="nonlinear")
    for i in range(env.n_c):
        for j in range(env.n_di):
            assert len(env._distractor_dist(j)) == 1


def test_distractor_class_changes_transition():
    a = make_env(d_d=1, distractor_class="linear").transition
    b = make_env(d_d=1, distractor_class="stochastic").transition
    assert not np.allclose(a, b)


def test_empirical_decomposition_invariant_to_d_d():
    cos = {}
    for d_d in (0, 1, 2):
        env = make_env(d_d=d_d, distractor_class="nonlinear")
        rng = np.random.default_rng(0)
        data = sample_transitions(env, 2000, uniform_behavior(env), rng)
        P = fit_empirical_model(env, data, None, alpha=0.1)
        cos[d_d] = mean_cos_phi(env, P, data)
    assert abs(cos[0] - cos[2]) < 1e-6


def test_capacity_limited_model_is_d_d_sensitive():
    cos = {}
    for d_d in (1, 2):
        env = make_env(d_d=d_d, distractor_class="stochastic")
        rng = np.random.default_rng(0)
        data = sample_transitions(env, 3000, uniform_behavior(env), rng)
        P = fit_feature_model(env, data, None, capacity=1, model_noise=0.5)
        cos[d_d] = mean_cos_phi(env, P, data)
    assert cos[2] < cos[1]
