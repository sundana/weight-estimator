"""Monte-Carlo check of the corrected Theorem-1 variance penalty.

The penalty for weighting a ratio estimator is ``sigma_w^2 Var(V) / n``. The earlier
draft used the numerator-only ``sigma_w^2 E[V^2] / n``; the two are far apart when
``E[V]`` is large relative to ``Var(V)``, which this test exploits.
"""

import numpy as np
import pytest

from distractor_gym.diagnostics import weight_signal_to_noise
from distractor_gym.losses import LossFamily, weight


def test_corrected_ratio_variance_matches_monte_carlo():
    rng = np.random.default_rng(0)
    n = 100
    trials = 40000
    shift = 5.0
    mu_mle, mu_w = [], []
    for _ in range(trials):
        y = rng.normal(size=n)
        V = shift + y
        w = np.abs(rng.normal(size=n))
        w = w / w.mean()
        mu_mle.append(V.mean())
        mu_w.append((w * V).sum() / w.sum())
    var_mle = float(np.var(mu_mle))
    var_w = float(np.var(mu_w))

    big = rng.normal(size=1_000_000)
    V = shift + big
    w = np.abs(rng.normal(size=1_000_000))
    w = w / w.mean()
    sigma_w2 = float(np.var(w))
    var_V = float(np.var(V))
    e_V2 = float(np.mean(V**2))

    corrected = var_mle + sigma_w2 * var_V / n
    numerator_only = var_mle + sigma_w2 * e_V2 / n

    assert abs(var_w - corrected) / corrected < 0.03
    assert numerator_only - corrected > 0.5 * corrected


def test_weight_signal_to_noise_uses_weighted_losses():
    losses = np.array([1.0, 3.0])
    assert weight_signal_to_noise(losses, np.ones(2)) == pytest.approx(4.0)


def test_decision_aligned_weight_is_passthrough():
    fn = np.linspace(0.1, 1.0, 10)
    assert np.allclose(weight(LossFamily.DECISION_ALIGNED, weight_fn=fn), fn)
