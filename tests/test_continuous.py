import numpy as np
import pytest

from distractor_gym.continuous import DistractorGym, make_distractor_gym
from distractor_gym.core import DistractorClass, RegimeConfig


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
    assert max_abs <= 1.5 + 1e-6


def test_action_space_from_base():
    env = make_env()
    assert env.action_space.shape == (1,)