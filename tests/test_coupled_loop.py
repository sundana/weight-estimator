import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from distractor_gym.deep.coupled_loop import CoupledLoopConfig, DeepCoupledLoop
from distractor_gym.deep.mujoco_diff import MujocoDistractorEnv, MujocoDistractorGym


def _make_envs(d_d=0, sigma=0.0):
    gym = MujocoDistractorGym(d_d=d_d, sigma_dist=sigma, seed=0, horizon=50)
    tenv = MujocoDistractorEnv(d_d=d_d, sigma_dist=sigma, seed=0)
    return gym, tenv


def _loop(family, **over):
    cfg = CoupledLoopConfig(
        family=family,
        horizon=3,
        iterations=1,
        model_updates=3,
        agent_updates=3,
        batch_size=64,
        init_steps=150,
        collect_steps=50,
        rollouts_per_iter=1,
        rollout_batch=32,
        eval_every=1,
        eval_episodes=1,
        n_models=2,
        hidden=16,
        **over,
    )
    gym, tenv = _make_envs()
    try:
        history = DeepCoupledLoop(gym, tenv, cfg).run()
    finally:
        gym.close()
    return history


def test_coupled_loop_smoke_mle():
    history = _loop("mle")
    assert len(history) == 1
    assert np.isfinite(history[0]["return"])
    assert history[0]["real_size"] > 0


def test_coupled_loop_coval_records_weight_stats():
    history = _loop("coval", target_critic=True, self_norm=True, clip_wmax=5.0, anneal_steps=10)
    assert history[0]["weight_var"] is not None
    assert np.isfinite(history[0]["weight_var"])
    assert history[0]["ess"] is not None and np.isfinite(history[0]["ess"])


def test_coupled_loop_vagram_smoke():
    history = _loop("vagram", self_norm=False)
    assert np.isfinite(history[0]["return"])
    assert np.isfinite(history[0]["weight_var"])


def test_unknown_family_rejected():
    gym, tenv = _make_envs()
    try:
        with pytest.raises(ValueError):
            DeepCoupledLoop(gym, tenv, CoupledLoopConfig(family="bad"))
    finally:
        gym.close()
