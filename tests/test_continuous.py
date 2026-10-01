import numpy as np
import pytest

from distractor_gym.continuous import DistractorGym, make_distractor_gym
from distractor_gym.core import DistractorClass, DistractorDynamics, RegimeConfig


def make_env(d_d=2, base_task="pendulum", distractor_class=DistractorClass.NONLINEAR):
    cfg = RegimeConfig(d_d=d_d, distractor_class=distractor_class, seed=0)
    return make_distractor_gym(cfg, base_task=base_task)


def test_observation_shape_and_dtype():
    env = make_env(d_d=3)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3 + 3,)
    assert obs.dtype == np.float32


def test_step_preserves_shape():
    env = make_env(d_d=2)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
    assert obs.shape == (3 + 2,)
    assert isinstance(reward, float)


def test_reward_flat_in_distractor_initial():
    env = make_env(d_d=0)
    obs0, _ = env.reset(seed=0)
    obs1, _ = env.reset(seed=0)
    assert np.allclose(obs0, obs1)


def test_distractor_block_bounded_nonlinear():
    env = make_env(d_d=4, distractor_class=DistractorClass.NONLINEAR)
    env.reset(seed=0)
    max_abs = 0.0
    for _ in range(100):
        obs, *_ = env.step(np.array([0.0]))
        max_abs = max(max_abs, np.abs(obs[3:]).max())
    assert max_abs <= 1.0 + 1e-6


def test_action_space_from_base():
    env = make_env()
    assert env.action_space.shape == (1,)


def test_nonlinear_dynamics_matches_closed_form():
    cfg = RegimeConfig(d_d=4, distractor_class=DistractorClass.NONLINEAR, seed=0)
    dyn = DistractorDynamics(cfg, action_dim=2)
    rng = np.random.default_rng(0)
    s = rng.normal(size=4)
    a = rng.normal(size=2)
    expected = np.tanh(dyn._matrix @ s + dyn._action_matrix @ a)
    assert np.allclose(dyn.step(s, a), expected)


def test_distractor_action_coupled():
    cfg = RegimeConfig(d_d=4, distractor_class=DistractorClass.NONLINEAR, seed=0)
    dyn = DistractorDynamics(cfg, action_dim=1)
    s = np.zeros(4)
    s_zero = dyn.step(s, np.array([0.0]))
    s_pos = dyn.step(s, np.array([1.0]))
    assert not np.allclose(s_zero, s_pos)


def test_zero_reward_relevance_to_distractor_dims():
    cfgs = [
        RegimeConfig(d_d=0, distractor_class=DistractorClass.NONLINEAR, seed=0),
        RegimeConfig(d_d=8, distractor_class=DistractorClass.NONLINEAR, seed=0),
    ]
    envs = [make_distractor_gym(c, base_task="pendulum") for c in cfgs]
    for env in envs:
        env.reset(seed=3)
    for a in (np.array([0.3]), np.array([-1.1]), np.array([0.0])):
        rewards = [env.step(a)[1] for env in envs]
        assert rewards[0] == rewards[1]


def test_reset_seed_reproducible():
    env = make_env(d_d=4)
    env.reset(seed=7)
    first = [env.step(np.array([0.0]))[0].copy() for _ in range(5)]
    env.reset(seed=7)
    second = [env.step(np.array([0.0]))[0].copy() for _ in range(5)]
    assert all(np.allclose(a, b) for a, b in zip(first, second))


def test_distractor_noise_seed_dependent():
    cfg = RegimeConfig(d_d=3, distractor_class=DistractorClass.NONLINEAR, sigma_dist=1.0, seed=0)
    env = make_distractor_gym(cfg, base_task="pendulum")
    env.reset(seed=1)
    a = env.step(np.array([0.0]))[0][3:].copy()
    env.reset(seed=2)
    b = env.step(np.array([0.0]))[0][3:].copy()
    assert not np.allclose(a, b)


@pytest.mark.parametrize(
    ("base_task", "d_c"),
    [("inverted_pendulum", 4), ("halfcheetah", 17), ("walker2d", 17)],
)
def test_mujoco_base_task_shapes(base_task, d_c):
    pytest.importorskip("mujoco")
    env = make_env(d_d=10, base_task=base_task)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (d_c + 10,)
    assert env.d_c == d_c
    obs2, reward, *_ = env.step(env.action_space.sample())
    assert obs2.shape == (d_c + 10,)
    assert isinstance(reward, float)