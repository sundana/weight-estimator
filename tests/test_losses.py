import numpy as np
import pytest

from distractor_gym.losses import (
    LossFamily,
    clip_weights,
    self_normalize,
    weight,
    weighted_loss,
)


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    V_s = rng.normal(size=50)
    V_sp = rng.normal(size=50)
    grad = rng.normal(size=(50, 3))
    return V_s, V_sp, grad


def test_mle_weights_are_ones(arrays):
    V_s, V_sp, grad = arrays
    w = weight(LossFamily.MLE, V_sp=V_sp)
    assert np.allclose(w, 1.0)


def test_vaml1_weights(arrays):
    V_s, V_sp, grad = arrays
    w = weight(LossFamily.VAML1, V_s=V_s, V_sp=V_sp)
    assert np.allclose(w, np.abs(V_sp - V_s))


def test_vagram_weights(arrays):
    V_s, V_sp, grad = arrays
    w = weight(LossFamily.VAGRAM, grad_V_sp=grad)
    assert np.allclose(w, np.linalg.norm(grad, axis=-1))


def test_td_error_weights(arrays):
    V_s, V_sp, grad = arrays
    delta = 0.3 * grad[:, 0] - 0.2 * grad[:, 1]
    w = weight(LossFamily.TD_ERROR, delta_td=delta)
    assert np.allclose(w, np.abs(delta))


def test_td_error_requires_delta_td(arrays):
    with pytest.raises(ValueError):
        weight(LossFamily.TD_ERROR, V_sp=arrays[1])


def test_calibrated_weights_formula(arrays):
    V_s, V_sp, grad = arrays
    eps = np.full(50, 0.5)
    sigma = np.full(50, 0.25)
    w = weight(
        LossFamily.CALIBRATED,
        grad_V_sp=grad,
        eps_model=eps,
        sigma_epistemic=sigma,
        eps_reg=1e-8,
    )
    expected = np.linalg.norm(grad, axis=-1) * eps / (sigma + 1e-8)
    assert np.allclose(w, expected)


def test_calibrated_requires_inputs(arrays):
    with pytest.raises(ValueError):
        weight(LossFamily.CALIBRATED, grad_V_sp=arrays[2], eps_model=np.ones(50))


def test_self_normalize_unit_mean(arrays):
    V_s, V_sp, grad = arrays
    w = np.abs(grad[:, 0]) + 0.1
    wn = self_normalize(w)
    assert wn.mean() == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(wn, w / (w.mean() + 1e-8))


def test_clip_weights_bounds(arrays):
    V_s, V_sp, grad = arrays
    w = np.array([-1.0, 0.5, 3.0, 10.0])
    assert np.allclose(clip_weights(w, w_min=0.0, w_max=5.0), [0.0, 0.5, 3.0, 5.0])
    assert np.allclose(clip_weights(w, w_min=0.0), [0.0, 0.5, 3.0, 10.0])


def test_lambert_weights_nonnegative_and_bounded(arrays):
    V_s, V_sp, grad = arrays
    w = weight(LossFamily.LAMBERT, V_sp=V_sp, tau=0.5)
    assert (w >= 0).all() and (w <= 1.0).all()


def test_decision_aligned_requires_weight_fn(arrays):
    with pytest.raises(ValueError):
        weight(LossFamily.DECISION_ALIGNED, V_sp=arrays[1])


def test_decision_aligned_passthrough(arrays):
    V_s, V_sp, grad = arrays
    fn = np.linspace(0.1, 1.0, 50)
    w = weight(LossFamily.DECISION_ALIGNED, weight_fn=fn)
    assert np.allclose(w, fn)


def test_missing_arrays_raise():
    with pytest.raises(ValueError):
        weight(LossFamily.VAGRAM, V_sp=np.zeros(3))


def test_weighted_loss_scalar():
    model = lambda s, a: np.full((len(s), 5), 0.2)
    batch = np.array([[0, 0, 1], [0, 1, 2]], dtype=int)
    w = np.array([1.0, 1.0])
    assert weighted_loss(model, batch, w) == pytest.approx(-np.log(0.2))