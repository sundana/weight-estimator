import numpy as np
import pytest

from distractor_gym.agents import sample_transitions, uniform_behavior
from distractor_gym.coupled import CoupledConfig, TabularCoupledSystem
from distractor_gym.core import RegimeConfig
from distractor_gym.tabular import TabularDistractorEnv


def make_system(
    weight_kind="mle", alpha_model=1e-2, alpha_critic=1e-2, n_data=1000, seed=0, self_norm=False
):
    env = TabularDistractorEnv(RegimeConfig(d_d=0, transition_noise=0.1, seed=seed))
    rng = np.random.default_rng(seed)
    data = sample_transitions(env, n_data, uniform_behavior(env), rng)
    cfg = CoupledConfig(
        weight_kind=weight_kind,
        alpha_model=alpha_model,
        alpha_critic=alpha_critic,
        self_norm=self_norm,
        n_steps=50,
        seed=seed,
    )
    return TabularCoupledSystem(env, data, cfg)


def test_jacobian_matches_directional_finite_difference():
    system = make_system()
    x = system.initial_state()
    rng = np.random.default_rng(3)
    v = rng.normal(size=len(x))
    v = v / np.linalg.norm(v)
    eps = 1e-6
    J = system.jacobian(x, eps=eps)
    directional = (system.step(x + eps * v) - system.step(x - eps * v)) / (2 * eps)
    assert np.abs(J @ v - directional).max() < 1e-4


def test_mle_weights_are_linearly_stable_at_small_step():
    system = make_system(weight_kind="mle", alpha_model=1e-3, alpha_critic=1e-2)
    assert system.spectral_radius() < 1.0


def test_mle_rollout_tracking_error_decays():
    system = make_system(weight_kind="mle", alpha_model=1e-3, alpha_critic=1e-2, n_data=4000)
    out = system.rollout(n_steps=200)
    error = system.tracking_error(out["states"], k=5)
    assert not out["diverged"]
    assert error[-1] < error[0]


def test_vaml_weight_variance_positive():
    system = make_system(weight_kind="vaml1")
    out = system.rollout(n_steps=10)
    assert np.all(out["weight_var"] > 0.0)


def test_value_aware_coupling_destabilizes_at_large_step():
    stable = make_system(
        weight_kind="vaml1", alpha_model=0.5, alpha_critic=5e-2, self_norm=True
    )
    unstable = make_system(
        weight_kind="vaml1", alpha_model=25.0, alpha_critic=5e-2, self_norm=True
    )
    assert stable.spectral_radius() < 1.0 + 1e-3
    assert unstable.spectral_radius() > 1.0
    assert unstable.rollout(n_steps=100)["diverged"]


def test_self_normalized_value_aware_is_less_stable_than_mle():
    mle = make_system(weight_kind="mle", alpha_model=25.0, alpha_critic=5e-2)
    vaml = make_system(
        weight_kind="vaml1", alpha_model=25.0, alpha_critic=5e-2, self_norm=True
    )
    assert vaml.spectral_radius() > mle.spectral_radius()


def test_tracking_error_shape():
    system = make_system()
    out = system.rollout(n_steps=30)
    error = system.tracking_error(out["states"], k=3)
    assert error.shape == (out["states"].shape[0] - 3,)


def test_normalization_modes():
    system = make_system(weight_kind="vaml1")
    phi = system.phi0
    system.cfg.normalization = "none"
    w_none = system._weights(phi, None, 1.0)
    system.cfg.normalization = "batch_self"
    w_self = system._weights(phi, None, 1.0)
    system.cfg.normalization = "min_max"
    w_mm = system._weights(phi, None, 1.0)
    assert np.isclose(w_self.mean(), 1.0, atol=1e-6)
    assert w_mm.min() >= 0.0 and w_mm.max() <= 1.0 + 1e-9
    assert not np.allclose(w_none, w_self)


def test_clip_median_ratio_bounds_weights():
    system = make_system(weight_kind="vaml1")
    phi = system.phi0
    system.cfg.normalization = "batch_self"
    unclipped = system._weights(phi, None, 1.0)
    system.cfg.clip_median_ratio = 2.0
    clipped = system._weights(phi, None, 1.0)
    assert clipped.max() <= 2.0 * np.median(unclipped) + 1e-9


def test_polyak_target_decouples_from_online_critic():
    system = make_system(weight_kind="vaml1", alpha_model=0.1, alpha_critic=5e-2)
    system.cfg.polyak_tau = 0.01
    out = system.rollout(n_steps=20)
    x = out["states"][-1]
    phi = x[system.m : system.m + system.S]
    phi_bar = x[system.m + system.S :]
    assert not np.allclose(phi, phi_bar)


def test_spectral_norm_caps_critic_lipschitz():
    system = make_system(weight_kind="vaml1", alpha_model=1.0, alpha_critic=5e-2)
    system.cfg.spectral_norm = True
    system.cfg.lip_target = 0.5
    out = system.rollout(n_steps=50)
    finite = out["critic_lip"][np.isfinite(out["critic_lip"])]
    assert finite.size and finite.max() <= 0.5 + 1e-6


def test_unknown_normalization_rejected():
    env = TabularDistractorEnv(RegimeConfig(d_d=0, seed=0))
    data = sample_transitions(env, 100, uniform_behavior(env), np.random.default_rng(0))
    with pytest.raises(ValueError):
        TabularCoupledSystem(env, data, CoupledConfig(normalization="bad"))
