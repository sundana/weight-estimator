"""Differentiable MuJoCo Distractor-Gym for the WP1 deep gradient diagnostic.

Standard Gymnasium/MuJoCo bindings are not differentiable, so the WP1 Exp 1.1
reference policy gradient ``g_true`` is obtained here by chaining per-step
transition Jacobians from ``mujoco.mjd_transitionFD`` inside a PyTorch
``autograd.Function``. The physics engine is the real MuJoCo one: the model is
switched to the Euler integrator (the only one ``mjd_transitionFD`` supports), and
one ``diff_step`` applies exactly one integrator step. A ``d_d``-dimensional
analytic distractor block with zero reward relevance is appended to the MuJoCo
physics state, mirroring ``core.DistractorDynamics``. ``MujocoDistractorGym``
exposes the same system to the SAC/random data generators.

The transition Jacobian covers models with ``nq == nv`` and no activation states
(e.g. ``InvertedDoublePendulum-v4``); other models raise ``ValueError``. The
distractor couplings are drawn with NumPy so the data-generating Gym view and the
torch rollout share the exact same block.
"""

from __future__ import annotations

import numpy as np
import torch


_BASE_TASKS = {
    "inverted_double_pendulum": "InvertedDoublePendulum-v4",
}

_IDP = "inverted_double_pendulum"
_IDP_LINK = 0.6


def _import_mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "mujoco is required for distractor_gym.deep.mujoco_diff; install "
            "`distractor-gym[mujoco]`."
        ) from exc
    return mujoco


def _import_gym():
    import gymnasium as gym

    return gym


def _couplings(d_d: int, action_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Distractor couplings ``(A, B)`` with the same draw as ``core.DistractorDynamics``."""
    coupling = np.random.default_rng(int(seed) + 101)
    if d_d > 0:
        matrix = coupling.normal(size=(d_d, d_d)) / np.sqrt(d_d)
        action_matrix = (
            coupling.normal(size=(d_d, action_dim)) / np.sqrt(max(action_dim, 1))
            if action_dim > 0
            else np.zeros((d_d, 0))
        )
    else:
        matrix = np.zeros((0, 0))
        action_matrix = np.zeros((0, action_dim))
    return matrix.astype(np.float64), action_matrix.astype(np.float64)


def idp_reward(qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
    """Native ``InvertedDoublePendulum`` reward as a differentiable function of state.

    Reproduces the Gymnasium ``step`` reward ``10 - 0.01 x^2 - (y - 2)^2 - 1e-3 v1^2
    - 5e-3 v2^2`` where ``(x, y)`` is the ``tip`` site position obtained from the
    analytic two-link forward kinematics.
    """
    x_cart = qpos[..., 0]
    th1 = qpos[..., 1]
    th2 = qpos[..., 2]
    tip_x = x_cart + _IDP_LINK * (torch.sin(th1 + th2) + torch.sin(th1))
    tip_z = _IDP_LINK * (torch.cos(th1 + th2) + torch.cos(th1))
    dist_penalty = 0.01 * tip_x**2 + (tip_z - 2.0) ** 2
    v1 = qvel[..., 1]
    v2 = qvel[..., 2]
    vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
    return 10.0 - dist_penalty - vel_penalty


def idp_reward_goal(qpos: torch.Tensor, qvel: torch.Tensor, sigma: float = 0.25) -> torch.Tensor:
    """Narrow goal reward ``exp(-d^2 / (2 sigma^2))`` around the upright tip.

    ``d`` is the analytic tip distance to the goal ``(tip_x, tip_z) = (0, 2)``. A small
    ``sigma`` concentrates reward near the goal, creating the sharp value cliff (large
    ``||grad_s V||`` and high weight variance) that the crossover criterion identifies as
    the value-aware-win regime, while staying differentiable for the policy gradient.
    """
    x_cart = qpos[..., 0]
    th1 = qpos[..., 1]
    th2 = qpos[..., 2]
    tip_x = x_cart + _IDP_LINK * (torch.sin(th1 + th2) + torch.sin(th1))
    tip_z = _IDP_LINK * (torch.cos(th1 + th2) + torch.cos(th1))
    d2 = tip_x**2 + (tip_z - 2.0) ** 2
    return torch.exp(-0.5 * d2 / sigma**2)


def _goal_reward_np(qpos: np.ndarray, qvel: np.ndarray, sigma: float) -> float:
    """NumPy narrow goal reward for the Gymnasium data-generating view."""
    x_cart = qpos[0] + _IDP_LINK * (np.sin(qpos[1] + qpos[2]) + np.sin(qpos[1]))
    tip_z = _IDP_LINK * (np.cos(qpos[1] + qpos[2]) + np.cos(qpos[1]))
    return float(np.exp(-0.5 * (x_cart**2 + (tip_z - 2.0) ** 2) / sigma**2))


def idp_terminated(qpos: torch.Tensor) -> torch.Tensor:
    """Native termination indicator ``tip_z <= 1`` (unused by the fixed-horizon rollout)."""
    th1 = qpos[..., 1]
    th2 = qpos[..., 2]
    tip_z = _IDP_LINK * (torch.cos(th1 + th2) + torch.cos(th1))
    return tip_z <= 1.0


_REWARDS = {_IDP: idp_reward}
_TERMINATED = {_IDP: idp_terminated}


class MujocoPhysics:
    """Real MuJoCo model with an Euler integrator and transition-Jacobian config."""

    def __init__(
        self,
        base_task: str = _IDP,
        *,
        eps: float = 1e-6,
        centered: bool = True,
    ) -> None:
        if base_task not in _BASE_TASKS:
            raise ValueError(f"unknown base task: {base_task}; use {sorted(_BASE_TASKS)}")
        mujoco = _import_mujoco()
        gym = _import_gym()
        env = gym.make(_BASE_TASKS[base_task])
        unwrapped = env.unwrapped
        fullpath = str(unwrapped.fullpath)
        self.frame_skip = int(unwrapped.frame_skip)
        self.action_low = np.asarray(unwrapped.action_space.low, dtype=np.float64)
        self.action_high = np.asarray(unwrapped.action_space.high, dtype=np.float64)
        self.init_qpos = np.asarray(unwrapped.init_qpos, dtype=np.float64).copy()
        self.init_qvel = np.asarray(unwrapped.init_qvel, dtype=np.float64).copy()
        env.close()

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self.data = mujoco.MjData(self.model)
        self.eps = float(eps)
        self.centered = bool(centered)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.nu = int(self.model.nu)
        if self.nq != self.nv or self.model.na != 0:
            raise ValueError("mujoco_diff supports nq == nv and na == 0 models only")

    @property
    def state_dim(self) -> int:
        return self.nq + self.nv

    def set_state(self, state: np.ndarray) -> None:
        """Write a physics state ``[qpos, qvel]`` and run ``mj_forward``."""
        self.data.qpos[:] = state[: self.nq]
        self.data.qvel[:] = state[self.nq : self.nq + self.nv]
        self.mujoco.mj_forward(self.model, self.data)

    def transition_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        """One-step discrete Jacobians ``(A, B)`` about the current ``mj_forward`` state."""
        n = self.state_dim
        a = np.zeros((n, n))
        b = np.zeros((n, self.nu))
        self.mujoco.mjd_transitionFD(
            self.model, self.data, self.eps, self.centered, a, b, None, None
        )
        return a, b


class _DiffMujocoStep(torch.autograd.Function):
    """One MuJoCo integrator step, differentiable through ``mjd_transitionFD``."""

    @staticmethod
    def forward(ctx, state: torch.Tensor, ctrl: torch.Tensor, physics: MujocoPhysics):
        mujoco = physics.mujoco
        model, data = physics.model, physics.data
        nq, nv = physics.nq, physics.nv
        states = state.detach().cpu().numpy().astype(np.float64)
        ctrls = ctrl.detach().cpu().numpy().astype(np.float64)
        out = np.empty_like(states)
        for i in range(states.shape[0]):
            data.qpos[:] = states[i, :nq]
            data.qvel[:] = states[i, nq : nq + nv]
            data.ctrl[:] = ctrls[i]
            mujoco.mj_step(model, data)
            out[i, :nq] = data.qpos
            out[i, nq : nq + nv] = data.qvel
        ctx.save_for_backward(state, ctrl)
        ctx.physics = physics
        return torch.as_tensor(out, dtype=state.dtype, device=state.device)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        physics = ctx.physics
        state, ctrl = ctx.saved_tensors
        nq, nv, nu = physics.nq, physics.nv, physics.nu
        states = state.detach().cpu().numpy().astype(np.float64)
        ctrls = ctrl.detach().cpu().numpy().astype(np.float64)
        grad = grad_out.detach().cpu().numpy().astype(np.float64)
        grad_state = np.zeros_like(states)
        grad_ctrl = np.zeros_like(ctrls)
        for i in range(states.shape[0]):
            physics.data.ctrl[:] = ctrls[i]
            physics.set_state(states[i])
            a, b = physics.transition_jacobian()
            grad_state[i] = grad[i] @ a
            grad_ctrl[i] = grad[i] @ b
        return (
            torch.as_tensor(grad_state, dtype=state.dtype, device=state.device),
            torch.as_tensor(grad_ctrl, dtype=ctrl.dtype, device=ctrl.device),
            None,
        )


class MujocoDistractorEnv:
    """Differentiable ground-truth Distractor-Gym over a real MuJoCo base task."""

    def __init__(
        self,
        d_d: int = 0,
        sigma_dist: float = 0.0,
        seed: int = 0,
        base_task: str = _IDP,
        *,
        eps: float = 1e-6,
        centered: bool = True,
        dtype: torch.dtype = torch.float32,
        reward_mode: str = "dense",
        goal_sigma: float = 0.25,
    ) -> None:
        self.physics = MujocoPhysics(base_task, eps=eps, centered=centered)
        self.base_task = base_task
        self.dtype = dtype
        self.d_c = self.physics.state_dim
        self.d_d = int(d_d)
        self.dim = self.d_c + self.d_d
        self.sigma_dist = float(sigma_dist)
        self.frame_skip = self.physics.frame_skip
        matrix, action_matrix = _couplings(self.d_d, self.physics.nu, seed)
        self.A = torch.as_tensor(matrix, dtype=dtype)
        self.B = torch.as_tensor(action_matrix, dtype=dtype)
        self.reward_fn = _REWARDS[base_task]
        self.terminated_fn = _TERMINATED[base_task]
        self.reward_mode = reward_mode
        self.goal_sigma = float(goal_sigma)

    def reward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Control reward, shape ``(batch,)``; independent of the distractor block."""
        qpos = s[..., : self.physics.nq]
        qvel = s[..., self.physics.nq : self.d_c]
        if self.reward_mode == "goal":
            return idp_reward_goal(qpos, qvel, sigma=self.goal_sigma)
        return self.reward_fn(qpos, qvel)

    def step(self, s: torch.Tensor, a: torch.Tensor, noise: bool = False) -> torch.Tensor:
        """One differentiable step of the true dynamics; distractor noise optional."""
        s = s.to(self.dtype)
        action = a.to(self.dtype)
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        state = s[..., : self.d_c]
        for _ in range(self.frame_skip):
            state = _DiffMujocoStep.apply(state, action, self.physics)
        if self.d_d > 0:
            s_d = s[..., self.d_c :]
            drive = s_d @ self.A.T + action @ self.B.T
            if noise and self.sigma_dist > 0.0:
                drive = drive + self.sigma_dist * torch.randn_like(drive)
            s_d = torch.tanh(drive)
        else:
            s_d = s[..., self.d_c :]
        return torch.cat([state, s_d], dim=-1)

    def sample_states(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Random initial states near the upright configuration; distractors zero."""
        init_qpos = torch.as_tensor(self.physics.init_qpos, dtype=self.dtype)
        qpos = init_qpos + (
            torch.rand(n, self.physics.nq, generator=generator, dtype=self.dtype) - 0.5
        ) * 0.6
        qvel = (
            torch.randn(n, self.physics.nv, generator=generator, dtype=self.dtype) * 0.3
        )
        state = torch.cat([qpos, qvel], dim=-1)
        if self.d_d > 0:
            state = torch.cat([state, torch.zeros(n, self.d_d, dtype=self.dtype)], dim=-1)
        return state


class MujocoDistractorGym:
    """Gymnasium view of the same system for the SAC/random data generators."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        d_d: int = 0,
        sigma_dist: float = 0.0,
        seed: int = 0,
        base_task: str = _IDP,
        horizon: int = 100,
        reward_mode: str = "dense",
        goal_sigma: float = 0.25,
    ) -> None:
        gym = _import_gym()
        mujoco = _import_mujoco()
        self.base = gym.make(_BASE_TASKS[base_task])
        self.base.unwrapped.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self.action_space = self.base.action_space
        nq = int(self.base.unwrapped.model.nq)
        nv = int(self.base.unwrapped.model.nv)
        self.d_c = nq + nv
        self.d_d = int(d_d)
        self.dim = self.d_c + self.d_d
        self.sigma_dist = float(sigma_dist)
        self.horizon = int(horizon)
        self.base_task = base_task
        self.reward_mode = reward_mode
        self.goal_sigma = float(goal_sigma)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.dim,),
            dtype=np.float32,
        )
        self._matrix, self._action_matrix = _couplings(
            self.d_d, int(np.prod(self.action_space.shape) or 1), seed
        )
        self._s_d = np.zeros(self.d_d, dtype=np.float64)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.base.reset(seed=seed, options=options)
        self._s_d = np.zeros(self.d_d, dtype=np.float64)
        self._rng = np.random.default_rng(int(seed) if seed is not None else 0)
        self._t = 0
        return self._obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.base.step(action)
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if self.d_d > 0:
            drive = self._s_d @ self._matrix.T + self._action_matrix @ a
            if self.sigma_dist > 0.0:
                drive = drive + self._rng.normal(0.0, self.sigma_dist, size=self.d_d)
            self._s_d = np.tanh(drive)
        if self.reward_mode == "goal":
            data = self.base.unwrapped.data
            qpos = np.asarray(data.qpos, dtype=np.float64)
            qvel = np.asarray(data.qvel, dtype=np.float64)
            reward = _goal_reward_np(qpos, qvel, self.goal_sigma)
        self._t += 1
        truncated = bool(truncated) or self._t >= self.horizon
        return self._obs(), float(reward), bool(terminated), truncated, info

    def close(self) -> None:
        self.base.close()

    def _obs(self) -> np.ndarray:
        data = self.base.unwrapped.data
        physics = np.concatenate(
            [np.asarray(data.qpos, dtype=np.float64), np.asarray(data.qvel, dtype=np.float64)]
        )
        return np.concatenate([physics, self._s_d]).astype(np.float32)


def rollout_return(
    env: MujocoDistractorEnv,
    actor,
    s0: torch.Tensor,
    horizon: int,
    gamma: float,
    model=None,
    x_stats: tuple | None = None,
    y_stats: tuple | None = None,
    clip_sigma: float = 6.0,
) -> torch.Tensor:
    """Discounted return of the deterministic policy, differentiated through the rollout.

    ``model=None`` uses the true MuJoCo dynamics (``g_true``); otherwise the model
    ensemble's differentiable mean is used (``g_model``). ``clip_sigma`` bounds the
    model prediction to the training band (``<= 0`` disables the clip).
    """
    s = s0
    total = torch.zeros(s0.shape[0], dtype=s0.dtype, device=s0.device)
    for t in range(horizon):
        a = actor.deterministic(s)
        total = total + (gamma**t) * env.reward(s, a)
        if model is None:
            s = env.step(s, a, noise=False)
        else:
            s = _model_mean(model, s, a, x_stats, y_stats, clip_sigma=clip_sigma)
    return total.mean()


def _model_mean(
    model, s: torch.Tensor, a: torch.Tensor, x_stats, y_stats, clip_sigma: float = 6.0
) -> torch.Tensor:
    """Differentiable ensemble-mean next state, clipped to the training data band.

    Long closed-loop model rollouts compound one-step error and can diverge far outside
    the data support, which makes the model-induced policy gradient meaningless. The
    prediction is therefore bounded to ``y_mean +/- clip_sigma * y_std``, the standard
    rollout-stabilization device; ``clip_sigma=-1`` disables it.
    """
    x = torch.cat([s, a], dim=-1)
    x_mean, x_std = x_stats
    y_mean, y_std = y_stats
    pred_n, _ = model((x - x_mean) / x_std)
    raw = pred_n.mean(dim=1) * y_std + y_mean
    if clip_sigma and clip_sigma > 0.0:
        raw = torch.clamp(raw, y_mean - clip_sigma * y_std, y_mean + clip_sigma * y_std)
    return raw
