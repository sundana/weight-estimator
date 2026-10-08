import numpy as np
import pytest

torch = pytest.importorskip("torch")

from distractor_gym.deep.losses import model_weights, self_normalize, weighted_mse
from distractor_gym.deep.nets import GaussianEnsemble, StateValue, SquashedGaussianActor, TwinCritic
from distractor_gym.deep.sac import ReplayBuffer, SACAgent, collect_random
from distractor_gym.deep.trainer import fit_dynamics, model_one_step_mse
from distractor_gym.deep.vjp import (
    finite_difference_grad,
    per_sample_grad,
    state_value_grad_norm,
)
from distractor_gym.losses import LossFamily


def test_per_sample_grad_matches_closed_form():
    x = torch.randn(16, 4)
    grad = per_sample_grad(lambda s: (s**2).sum(), x)
    assert torch.allclose(grad, 2.0 * x, atol=1e-5)


def test_state_value_grad_norm_matches_loop_autograd():
    torch.manual_seed(0)
    net = StateValue(obs_dim=4, hidden=16)
    obs = torch.randn(12, 4)
    vjp_norm = state_value_grad_norm(net, obs)
    x = obs.detach().clone().requires_grad_(True)
    grads = torch.autograd.grad(net(x).sum(), x)[0]
    assert torch.allclose(vjp_norm, grads.norm(dim=-1), atol=1e-4)


def test_finite_difference_close_to_vjp():
    x = torch.randn(8, 3)
    ref = per_sample_grad(lambda s: (s**2).sum(), x)
    fd = finite_difference_grad(lambda xb: (xb**2).sum(dim=-1), x, eps=1e-3)
    assert torch.allclose(fd, ref, atol=1e-3)


def test_gaussian_ensemble_shapes_and_epistemic():
    model = GaussianEnsemble(in_dim=5, state_dim=3, n_models=4, hidden=16)
    x = torch.randn(7, 5)
    mean, log_var = model(x)
    assert mean.shape == (7, 4, 3)
    assert log_var.shape == (7, 4, 3)
    pred, std = model.predict(x)
    assert pred.shape == (7, 3) and std.shape == (7, 3)
    assert torch.all(std >= 0)


def test_twin_critic_and_actor_shapes():
    critic = TwinCritic(obs_dim=3, act_dim=2, hidden=16)
    obs, act = torch.randn(6, 3), torch.randn(6, 2)
    q1, q2 = critic(obs, act)
    assert q1.shape == (6,) and q2.shape == (6,)
    actor = SquashedGaussianActor(obs_dim=3, act_dim=2, hidden=16)
    action, log_prob, mean_action = actor.sample(obs)
    assert action.shape == (6, 2) and log_prob.shape == (6, 1)
    assert torch.all(action.abs() <= 1.0 + 1e-5)
    assert torch.all(mean_action.abs() <= 1.0 + 1e-5)


def test_model_weights_families():
    grad = torch.tensor([3.0, 4.0])
    eps = torch.tensor([2.0, 1.0])
    sig = torch.tensor([1.0, 3.0])
    delta = torch.tensor([0.5, -2.0])
    assert torch.allclose(model_weights(LossFamily.MLE, grad_norm=grad), torch.ones(2))
    assert torch.allclose(model_weights(LossFamily.VAGRAM, grad_norm=grad), grad)
    assert torch.allclose(model_weights(LossFamily.TD_ERROR, delta_td=delta), delta.abs())
    cal = model_weights(
        LossFamily.CALIBRATED, grad_norm=grad, eps_model=eps, sigma_epistemic=sig, eps_reg=0.0
    )
    assert torch.allclose(cal, torch.tensor([6.0, 4.0 / 3.0]))


def test_weighted_mse_self_normalized():
    pred = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    target = torch.zeros(2, 2)
    weights = torch.tensor([1.0, 3.0])
    loss = weighted_mse(pred, target, weights, normalize=True)
    per = torch.tensor([0.0, 2.0])
    assert torch.isclose(loss, (self_normalize(weights) * per).mean())


def test_fit_dynamics_smoke_vagram():
    rng = np.random.default_rng(0)
    n, obs_dim, act_dim = 96, 3, 1
    obs = rng.normal(size=(n, obs_dim)).astype(np.float32)
    act = rng.normal(size=(n, act_dim)).astype(np.float32)
    next_obs = (obs + 0.1 * rng.normal(size=(n, obs_dim))).astype(np.float32)
    data = {"obs": obs, "act": act, "next_obs": next_obs}
    fd = fit_dynamics(
        data,
        LossFamily.VAGRAM,
        value_fn=lambda o: o.sum(dim=-1),
        grad_norm_fn=lambda o: torch.ones(len(o)),
        n_models=2,
        hidden=16,
        epochs=2,
        batch_size=32,
    )
    pred = fd.predict_mean(obs[:5], act[:5])
    std = fd.epistemic_std(obs[:5], act[:5])
    assert pred.shape == (5, obs_dim)
    assert np.isfinite(pred).all() and np.isfinite(std).all()


def test_replay_buffer_roundtrip(tmp_path):
    buf = ReplayBuffer(capacity=10, obs_dim=3, act_dim=1)
    rng = np.random.default_rng(0)
    for i in range(5):
        buf.add(
            rng.normal(size=3),
            rng.normal(size=1),
            1.0,
            rng.normal(size=3),
            float(i % 2),
            terminal=float(i % 3 == 0),
        )
    assert len(buf) == 5
    batch = buf.sample(4)
    assert batch["obs"].shape == (4, 3)
    path = tmp_path / "replay.npz"
    buf.save(path)
    loaded = ReplayBuffer.load(path)
    assert len(loaded) == 5
    assert np.allclose(loaded.obs[:5], buf.obs[:5])
    assert np.allclose(loaded.terminal[:5], buf.terminal[:5])


def test_fit_dynamics_all_deep_families():
    rng = np.random.default_rng(0)
    n, obs_dim, act_dim = 64, 3, 1
    obs = rng.normal(size=(n, obs_dim)).astype(np.float32)
    act = rng.normal(size=(n, act_dim)).astype(np.float32)
    next_obs = (obs + 0.1 * rng.normal(size=(n, obs_dim))).astype(np.float32)
    data = {
        "obs": obs,
        "act": act,
        "next_obs": next_obs,
        "rew": rng.normal(size=n).astype(np.float32),
        "done": np.zeros(n, dtype=np.float32),
    }
    torch.manual_seed(0)
    sv = StateValue(obs_dim=obs_dim, hidden=16)
    vfn = lambda o: sv(o)
    gfn = lambda o: state_value_grad_norm(sv, o)
    families = [
        LossFamily.MLE,
        LossFamily.VAML1,
        LossFamily.VAGRAM,
        LossFamily.TD_ERROR,
        LossFamily.CALIBRATED,
        LossFamily.LAMBERT,
    ]
    for fam in families:
        fd = fit_dynamics(
            data, fam, value_fn=vfn, grad_norm_fn=gfn, n_models=2, hidden=16, epochs=1, batch_size=32
        )
        mse = model_one_step_mse(fd, data)
        assert np.isfinite(mse) and mse >= 0.0


def test_replay_buffer_sample_rng_is_deterministic():
    buf = ReplayBuffer(capacity=8, obs_dim=1, act_dim=1)
    for i in range(8):
        buf.add(np.array([i]), np.array([0.0]), 0.0, np.array([i]), 0.0)
    a = buf.sample(4, rng=np.random.default_rng(0))
    b = buf.sample(4, rng=np.random.default_rng(0))
    assert np.allclose(a["obs"].numpy(), b["obs"].numpy())
    c = buf.sample(4, rng=np.random.default_rng(1))
    assert not np.allclose(a["obs"].numpy(), c["obs"].numpy())


def test_sac_agent_update_smoke():
    agent = SACAgent(obs_dim=3, act_dim=1, hidden=16)
    rng = np.random.default_rng(0)
    batch = {
        "obs": torch.as_tensor(rng.normal(size=(8, 3)), dtype=torch.float32),
        "act": torch.as_tensor(rng.normal(size=(8, 1)), dtype=torch.float32),
        "rew": torch.as_tensor(rng.normal(size=8), dtype=torch.float32),
        "next_obs": torch.as_tensor(rng.normal(size=(8, 3)), dtype=torch.float32),
        "done": torch.zeros(8),
    }
    out = agent.update(batch)
    assert all(np.isfinite(list(out.values())))
    action = agent.select_action(np.zeros(3, dtype=np.float32))
    assert action.shape == (1,)


def test_collect_random_smoke():
    gym = pytest.importorskip("gymnasium")
    env = gym.make("Pendulum-v1")
    buf = collect_random(env, n_steps=20, seed=0)
    assert len(buf) == 20
    assert buf.obs.shape == (20, 3)


def test_deep_alignment_gradient_cosine_smoke():
    import pathlib
    import sys

    pytest.importorskip("mujoco")

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from experiments.exp1_deep_alignment import _random_data, gradient_cosine, policy_gradient
    from distractor_gym.deep.mujoco_diff import MujocoDistractorEnv

    env = MujocoDistractorEnv(d_d=3, sigma_dist=0.0, seed=0)
    actor = SACAgent(obs_dim=env.dim, act_dim=1, hidden=16).actor
    s0 = env.sample_states(8)
    g_true = policy_gradient(env, actor, s0, horizon=4, gamma=0.99)
    assert g_true.norm() > 0.0
    data = _random_data(env, 256, seed=0)
    fd = fit_dynamics(data, LossFamily.MLE, n_models=2, hidden=16, epochs=3, batch_size=64)
    g_model = policy_gradient(env, actor, s0, horizon=4, gamma=0.99, fd=fd)
    cos = gradient_cosine(g_true, g_model)
    assert -1.0 <= cos <= 1.0
