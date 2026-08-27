import numpy as np
import pytest

from distractor_gym.core import RegimeConfig
from distractor_gym.tabular import Grid, TabularDistractorEnv


def make_env(**kw):
    return TabularDistractorEnv(RegimeConfig(d_d=kw.pop("d_d", 1), transition_noise=kw.pop("noise", 0.1), **kw))


def test_transition_is_stochastic_matrix():
    env = make_env()
    assert env.transition.shape == (env.S, env.n_a, env.S)
    assert np.allclose(env.transition.sum(axis=2), 1.0)
    assert (env.transition >= 0).all()


def test_value_flat_in_distractor_dims():
    env = make_env()
    V = env.value_iteration()
    Vc = V.reshape(env.n_c, env.n_di)
    assert np.allclose(Vc, Vc[:, :1])


def test_reward_flat_in_distractor_dims():
    env = make_env()
    r = env.rewards.reshape(env.n_c, env.n_di)
    assert np.allclose(r, r[:, :1])


def test_value_grad_zero_in_distractor_dims():
    env = make_env()
    V = env.value_iteration()
    grad = env.value_grad(V)
    assert grad.shape == (env.S, 1 + env.d_d)
    assert np.allclose(grad[:, 1:], 0.0)


def test_value_grad_control_nontrivial():
    env = make_env(d_d=0)
    V = env.value_iteration()
    grad = env.value_grad(V)
    assert np.abs(grad[:, 0]).max() > 0


def test_sparse_reward_value_monotone_to_goal():
    env = make_env(d_d=0, goal_radius=0.3)
    V = env.value_iteration()[: env.n_c]
    x = env.grid_c.points
    left = x < env.goal
    right = x >= env.goal
    assert np.all(np.diff(V[left]) >= 0)
    assert np.all(np.diff(V[right]) <= 0)


def test_low_gamma_sparse_reward_flattens_value():
    env = make_env(d_d=0, goal_radius=0.3)
    V = env.value_iteration(gamma=0.2)[: env.n_c]
    x = env.grid_c.points
    far = V[np.abs(x - env.goal) > 1.0]
    assert far.max() < 0.1


def test_transition_error_uses_coordinates():
    env = make_env(d_d=1)
    s1 = env._pack(5, 2)
    s2 = env._pack(7, 2)
    assert env.transition_error(s1, s2) == pytest.approx(2 * env.grid_c.dx)


def test_grid_dx():
    g = Grid(-3.0, 3.0, 15)
    assert g.dx == pytest.approx(6.0 / 14)