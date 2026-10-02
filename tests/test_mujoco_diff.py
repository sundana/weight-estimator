import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from distractor_gym.deep.mujoco_diff import (
    MujocoDistractorEnv,
    MujocoDistractorGym,
    MujocoPhysics,
    idp_reward,
    rollout_return,
)


def _mj_step_state(physics, state, ctrl):
    physics.data.ctrl[:] = ctrl
    physics.set_state(state)
    physics.mujoco.mj_step(physics.model, physics.data)
    return np.concatenate([physics.data.qpos.copy(), physics.data.qvel.copy()])


def _central_fd(physics, state, ctrl, eps=1e-6):
    n = physics.state_dim
    a = np.zeros((n, n))
    b = np.zeros((n, physics.nu))
    for i in range(n):
        step = np.zeros(n)
        step[i] = eps
        a[:, i] = (
            _mj_step_state(physics, state + step, ctrl)
            - _mj_step_state(physics, state - step, ctrl)
        ) / (2 * eps)
    for i in range(physics.nu):
        step = np.zeros(physics.nu)
        step[i] = eps
        b[:, i] = (
            _mj_step_state(physics, state, ctrl + step)
            - _mj_step_state(physics, state, ctrl - step)
        ) / (2 * eps)
    return a, b


def test_transition_jacobian_matches_finite_difference():
    physics = MujocoPhysics()
    rng = np.random.default_rng(0)
    state = np.concatenate([rng.normal(size=physics.nq), rng.normal(size=physics.nv)])
    ctrl = rng.normal(size=physics.nu) * 0.1
    physics.data.ctrl[:] = ctrl
    physics.set_state(state)
    a, b = physics.transition_jacobian()
    a_fd, b_fd = _central_fd(physics, state, ctrl)
    assert np.allclose(a, a_fd, atol=1e-6)
    assert np.allclose(b, b_fd, atol=1e-6)


def test_reward_matches_gym_native():
    physics = MujocoPhysics()
    rng = np.random.default_rng(1)
    qpos = rng.normal(size=physics.nq) * 0.1
    qvel = rng.normal(size=physics.nv) * 0.1
    physics.set_state(np.concatenate([qpos, qvel]))
    x, _, y = physics.data.site_xpos[0]
    dist = 0.01 * x**2 + (y - 2) ** 2
    v1, v2 = physics.data.qvel[1:3]
    native = 10.0 - dist - (1e-3 * v1**2 + 5e-3 * v2**2)
    ours = float(idp_reward(torch.as_tensor(qpos), torch.as_tensor(qvel)))
    assert ours == pytest.approx(native, abs=1e-9)


def test_gym_and_diff_env_agree_on_step():
    env = MujocoDistractorEnv(d_d=0, sigma_dist=0.0, seed=0)
    gym_env = MujocoDistractorGym(d_d=0, sigma_dist=0.0, seed=0)
    obs, _ = gym_env.reset(seed=0)
    s = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    a = torch.tensor([[0.1]])
    with torch.no_grad():
        s2 = env.step(s, a, noise=False)
    obs2, _, _, _, _ = gym_env.step(a.numpy().reshape(-1))
    assert np.allclose(s2.numpy().reshape(-1), obs2, atol=1e-5)


def test_diff_step_gradient_matches_finite_difference():
    from distractor_gym.deep.nets import SquashedGaussianActor

    torch.manual_seed(0)
    env = MujocoDistractorEnv(d_d=0, sigma_dist=0.0, seed=0, dtype=torch.float64)
    actor = SquashedGaussianActor(obs_dim=env.dim, act_dim=1, hidden=8, n_layers=1).double()
    s0 = env.sample_states(4, generator=torch.Generator().manual_seed(3))
    horizon, gamma = 3, 0.99
    params = list(actor.parameters())
    grads = torch.autograd.grad(rollout_return(env, actor, s0, horizon, gamma), params)
    eps = 1e-5
    for param, grad in zip(params, grads):
        flat = param.detach().reshape(-1)
        fd = torch.zeros_like(flat)
        for j in range(flat.numel()):
            orig = flat[j].item()
            with torch.no_grad():
                flat[j] = orig + eps
                plus = float(rollout_return(env, actor, s0, horizon, gamma))
                flat[j] = orig - eps
                minus = float(rollout_return(env, actor, s0, horizon, gamma))
                flat[j] = orig
            fd[j] = (plus - minus) / (2 * eps)
        assert torch.allclose(grad.reshape(-1), fd, atol=1e-4, rtol=1e-3)


def test_reward_flat_in_distractor():
    env = MujocoDistractorEnv(d_d=3)
    s = env.sample_states(1)
    perturbed = s.clone()
    perturbed[0, env.d_c :] = 7.0
    a = torch.zeros(1, 1)
    assert torch.allclose(env.reward(s, a), env.reward(perturbed, a))


def test_data_gym_obs_shape():
    gym_env = MujocoDistractorGym(d_d=4, base_task="inverted_double_pendulum")
    obs, _ = gym_env.reset(seed=0)
    assert obs.shape == (gym_env.d_c + 4,)
    obs2, _, _, _, _ = gym_env.step(np.zeros(1, dtype=np.float32))
    assert obs2.shape == obs.shape
