import numpy as np
import pytest

from distractor_gym.diagnostics import (
    decompose_td_error,
    gradient_alignment,
    weight_estimator_stats,
    weight_signal_to_noise,
)
from distractor_gym.losses import LossFamily, weight


def test_gradient_alignment_parallel_vectors():
    g = np.array([1.0, 0.0])
    assert gradient_alignment(g, 2.0 * g) == pytest.approx(1.0)


def test_gradient_alignment_orthogonal_vectors():
    g = np.array([1.0, 0.0])
    assert gradient_alignment(g, np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_mle_weight_is_ones():
    s = np.zeros((4, 2))
    s_prime = np.ones((4, 2))
    V = np.zeros(4)
    grad_V = np.ones((4, 2))
    w = weight(LossFamily.MLE, V_sp=V, grad_V_sp=grad_V)
    assert np.allclose(w, np.ones(4))


def test_decompose_td_error_perfect_fit():
    n = 1000
    rng = np.random.default_rng(0)
    grad_norm = rng.uniform(0.1, 2.0, n)
    eps = rng.uniform(0.0, 1.0, n)
    cos = np.abs(np.cos(rng.uniform(0.0, np.pi, n)))
    pred = grad_norm * eps * cos
    delta = 2.0 * pred
    res = decompose_td_error(delta, grad_norm, eps, cos)
    assert res.r2 == pytest.approx(1.0, abs=1e-9)
    assert res.slope == pytest.approx(2.0, abs=1e-9)


def test_weight_signal_to_noise_degenerate():
    losses = np.full(10, 3.0)
    assert weight_signal_to_noise(losses, np.ones(10)) == float("inf")


def test_weight_signal_to_noise_formula():
    losses = np.array([1.0, 3.0])
    assert weight_signal_to_noise(losses, np.ones(2)) == pytest.approx(4.0, abs=1e-9)


def test_weight_estimator_stats_shape():
    w = np.ones(10)
    w_oracle = np.full(10, 2.0)
    stats = weight_estimator_stats(w, w_oracle, np.full(10, 3.0))
    assert stats.bias == pytest.approx(-1.0)
    assert stats.variance == pytest.approx(0.0)
    assert stats.effective_sample_size == pytest.approx(10.0)