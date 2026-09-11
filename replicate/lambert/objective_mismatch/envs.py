"""CartPole environments.

Preference order: Gymnasium/SB3 CartPole-v1, MuJoCo-backed cartpole, then a
custom Euler-integration fallback. Every env exposes the same interface:
reset(random_init) -> state, step(action) -> (state, reward, done), and
set_state(state) for injecting arbitrary states.
State: [x, x_dot, theta, theta_dot]; action: normalized force in [-1, 1]."""

import numpy as np

from .config import CP_XML

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:  # pragma: no cover
    mujoco = None
    MUJOCO_AVAILABLE = False

try:
    import gymnasium
    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover
    gymnasium = None
    GYMNASIUM_AVAILABLE = False


class MujocoCartPoleEnv:
    """Continuous cartpole running on the real MuJoCo physics engine.
    State: [x, x_dot, theta, theta_dot]; action: force in [-10, 10] N.
    Matches the paper's cartpole (masscart 1.0, masspole 0.1, length 0.5)."""

    def __init__(self):
        if not MUJOCO_AVAILABLE:
            raise ImportError("mujoco is required for MujocoCartPoleEnv")
        self.model = mujoco.MjModel.from_xml_string(CP_XML)
        self.data = mujoco.MjData(self.model)
        self.slider = self.model.joint("slider").id
        self.hinge = self.model.joint("hinge").id
        self.xq = self.model.jnt_qposadr[self.slider]
        self.tq = self.model.jnt_qposadr[self.hinge]
        self.xv = self.model.jnt_dofadr[self.slider]
        self.tv = self.model.jnt_dofadr[self.hinge]
        self.theta_threshold = 15 * np.pi / 180
        self.x_threshold = 2.4
        self.max_steps = 200
        self.state = None
        self.current_step = 0

    def set_state(self, state):
        self.data.qpos[self.xq] = state[0]
        self.data.qpos[self.tq] = state[2]
        self.data.qvel[self.xv] = state[1]
        self.data.qvel[self.tv] = state[3]
        mujoco.mj_forward(self.model, self.data)
        self.state = state.copy()
        return self.state

    def reset(self, random_init=True):
        mujoco.mj_resetData(self.model, self.data)
        if random_init:
            noise = np.random.uniform(-0.05, 0.05, size=4)
        else:
            noise = np.zeros(4)
        self.set_state(noise)
        self.current_step = 0
        return self.state.copy()

    def _read_state(self):
        return np.array([self.data.qpos[self.xq], self.data.qvel[self.xv],
                         self.data.qpos[self.tq], self.data.qvel[self.tv]],
                        dtype=np.float32)

    def _fallen(self, x, theta):
        return bool(x < -self.x_threshold or x > self.x_threshold
                    or theta < -self.theta_threshold or theta > self.theta_threshold)

    def step(self, action):
        self.data.ctrl[0] = float(np.clip(action, -1.0, 1.0))
        mujoco.mj_step(self.model, self.data)
        self.state = self._read_state()
        self.current_step += 1
        x, theta = self.state[0], self.state[2]
        fallen = self._fallen(x, theta)
        reward = 1.0 if not fallen else 0.0
        done = fallen or self.current_step >= self.max_steps
        return self.state.copy(), reward, done


class ContinuousCartPoleEnv:
    """Custom Euler-integration cartpole, used as a fallback when MuJoCo is absent.
    State: [x, x_dot, theta, theta_dot]; action: force in [-10, 10] N."""

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.max_steps = 200
        self.theta_threshold_radians = 15 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.state = None
        self.current_step = 0

    def set_state(self, state):
        self.state = np.asarray(state, dtype=np.float32).copy()
        return self.state

    def reset(self, random_init=True):
        if random_init:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        else:
            self.state = np.zeros(4)
        self.current_step = 0
        return self.state.copy()

    def _fall(self, x, theta):
        return bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        force = action * self.force_mag
        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.current_step += 1
        reward = 1.0 if not self._fall(x, theta) else 0.0
        done = self._fall(x, theta) or self.current_step >= self.max_steps
        return self.state.copy(), reward, done


class Sb3CartPoleEnv:
    """CartPole-v1 from Gymnasium (the environment backend shipped with
    stable-baselines3). State: [x, x_dot, theta, theta_dot]; the CEM-planned
    normalized force in [-1, 1] is mapped to the discrete push (+-10 N)."""

    def __init__(self, render=False):
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required for Sb3CartPoleEnv")
        self.env = gymnasium.make("CartPole-v1", render_mode="human" if render else None,
                                  disable_env_checker=True)
        self.max_steps = 200
        self.state = None
        self.current_step = 0

    def set_state(self, state):
        self.env.reset()
        state = np.asarray(state, dtype=np.float32)
        self.env.unwrapped.state = state.copy()
        self.state = state.copy()
        return self.state

    def reset(self, random_init=True):
        self.env.reset()
        if random_init:
            noise = np.random.uniform(-0.05, 0.05, size=4)
        else:
            noise = np.zeros(4)
        self.set_state(noise)
        self.current_step = 0
        return self.state.copy()

    def step(self, action):
        force = float(np.clip(action, -1.0, 1.0))
        obs, reward, terminated, truncated, _ = self.env.step(1 if force >= 0 else 0)
        self.state = np.asarray(obs, dtype=np.float32)
        self.current_step += 1
        done = bool(terminated or truncated) or self.current_step >= self.max_steps
        if done:
            self.env.reset()
        return self.state.copy(), float(reward), done

    def render(self):
        return self.env.render()


def make_env(name, render=False):
    """Factory returning a Gymnasium cartpole (SB3 backend), falling back to
    MuJoCo, then to custom Euler-integration physics."""
    if name == "cartpole":
        if GYMNASIUM_AVAILABLE:
            return Sb3CartPoleEnv(render=render)
        if MUJOCO_AVAILABLE:
            return MujocoCartPoleEnv()
        return ContinuousCartPoleEnv()
    raise ValueError(f"Unknown environment: {name}")