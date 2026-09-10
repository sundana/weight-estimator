import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:  # pragma: no cover
    mujoco = None
    MUJOCO_AVAILABLE = False

try:
    from scipy.linalg import solve_continuous_are
except ImportError:  # pragma: no cover
    solve_continuous_are = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eksperimen berjalan pada perangkat: {device}  (mujoco: {MUJOCO_AVAILABLE})")

# ---- PETS hyper-parameters (Lambert et al. 2020, Appendix tab:paramspets) ----
NET_WIDTH = 500
NET_DEPTH = 2
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.9
FULL_EPOCHS = 100

CEM_HORIZON = 25
CEM_SAMPLES = 400
CEM_ELITES = 40
CEM_ITERATIONS = 5
REWARD_TRIALS = 10

# Dataset sizes (Lambert et al. 2020, Appendix tab:dataset)
CP_GRID_SIZE = 16807
CP_EXPERT_SIZE = 2400
CP_ONPOLICY_SIZE = 3780
EXPERT_REWARD_THRESHOLD = 179.0

# ---- MuJoCo models (MJCF) ----
CP_XML = """
<mujoco model="cartpole">
  <compiler angle="radian"/>
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <body name="cart" pos="0 0 0.1">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2.5 2.5"/>
      <geom name="cart" type="box" size="0.1 0.1 0.04" pos="0 0 -0.04" mass="1.0"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0.1"/>
        <geom name="pole" type="capsule" fromto="0 0 0.1 0 0 0.6" size="0.02" mass="0.1"/>
        <inertial pos="0 0 0.35" mass="0.1" diaginertia="0.001 0.001 0.001"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="cart_motor" joint="slider" gear="10" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

HC_XML = """
<mujoco model="half_cheetah">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <geom name="floor" pos="0 0 0" size="40 40 0.25" type="plane" rgba="0.8 0.9 0.8 1"/>
    <light diffuse="0.6 0.6 0.6" pos="0 1 4" dir="0 -1 -3"/>
    <body name="torso" pos="0 0 1.0">
      <joint name="root" type="free"/>
      <geom name="torso_geom" type="capsule" fromto="-.5 0 0 .5 0 0" size="0.1" mass="1"/>
      <body name="bthigh" pos="-0.5 0 0">
        <joint name="bthigh" type="hinge" axis="0 1 0" pos="0 0 -.05"/>
        <geom name="bthigh_geom" type="capsule" fromto="0 0 0 -0.35 0 -0.25" size="0.08" mass="1"/>
        <body name="bshin" pos="-0.35 0 -0.25">
          <joint name="bshin" type="hinge" axis="0 1 0" pos="0 0 0"/>
          <geom name="bshin_geom" type="capsule" fromto="0 0 0 -0.25 0 -0.3" size="0.06" mass="1"/>
          <body name="bfoot" pos="-0.25 0 -0.3">
            <joint name="bfoot" type="hinge" axis="0 1 0" pos="0 0 0"/>
            <geom name="bfoot_geom" type="capsule" fromto="0 0 0 0.05 0 -0.3" size="0.04" mass="1"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos="0.5 0 0">
        <joint name="fthigh" type="hinge" axis="0 1 0" pos="0 0 -.05"/>
        <geom name="fthigh_geom" type="capsule" fromto="0 0 0 0.35 0 -0.25" size="0.08" mass="1"/>
        <body name="fshin" pos="0.35 0 -0.25">
          <joint name="fshin" type="hinge" axis="0 1 0" pos="0 0 0"/>
          <geom name="fshin_geom" type="capsule" fromto="0 0 0 0.25 0 -0.3" size="0.06" mass="1"/>
          <body name="ffoot" pos="0.25 0 -0.3">
            <joint name="ffoot" type="hinge" axis="0 1 0" pos="0 0 0"/>
            <geom name="ffoot_geom" type="capsule" fromto="0 0 0 -0.05 0 -0.3" size="0.04" mass="1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="bthigh" joint="bthigh"/>
    <motor name="bshin" joint="bshin"/>
    <motor name="bfoot" joint="bfoot"/>
    <motor name="fthigh" joint="fthigh"/>
    <motor name="fshin" joint="fshin"/>
    <motor name="ffoot" joint="ffoot"/>
  </actuator>
</mujoco>
"""


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


class MujocoHalfCheetahEnv:
    """Half-cheetah (MuJoCo). Obs is a full concatenation of qpos/qvel
    (approximation; exact PETS 17-dim protocol is defined in the HC experiment)."""

    def __init__(self):
        if not MUJOCO_AVAILABLE:
            raise ImportError("mujoco is required for MujocoHalfCheetahEnv")
        self.model = mujoco.MjModel.from_xml_string(HC_XML)
        self.data = mujoco.MjData(self.model)
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, random_init=True):
        mujoco.mj_resetData(self.model, self.data)
        if random_init:
            for i in range(6):
                self.data.qpos[self.model.nq - 6 + i] = np.random.uniform(-0.05, 0.05)
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        return self.observation()

    def observation(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        reward = float(self.data.qvel[0])  # forward velocity of the torso
        done = self.current_step >= self.max_steps
        return self.observation(), reward, done


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


def make_env(name):
    """Factory returning a MuJoCo-backed env, falling back to custom physics."""
    if name == "cartpole":
        if MUJOCO_AVAILABLE:
            return MujocoCartPoleEnv()
        return ContinuousCartPoleEnv()
    if name == "halfcheetah":
        if not MUJOCO_AVAILABLE:
            raise ImportError("Half-cheetah requires MuJoCo.")
        return MujocoHalfCheetahEnv()
    raise ValueError(f"Unknown environment: {name}")


class ProbabilisticDynamicsModel(nn.Module):
    """PETS probabilistic forward model (P). Predicts delta = s' - s as a
    Gaussian (mean, logvar), with bounded log-variance.
    """

    def __init__(self, state_dim=4, action_dim=1, hidden_dim=NET_WIDTH, depth=NET_DEPTH):
        super(ProbabilisticDynamicsModel, self).__init__()
        layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(hidden_dim, state_dim)
        self.fc_logvar = nn.Linear(hidden_dim, state_dim)
        self.max_logvar = nn.Parameter(torch.ones(1, state_dim) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(1, state_dim) * -10.0)

    def forward(self, state, action):
        x = self.features(torch.cat([state, action], dim=-1))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def nll(self, state, action, next_state):
        """Mean Gaussian negative log-likelihood of delta = next_state - state."""
        target_delta = next_state - state
        mean, logvar = self.forward(state, action)
        inv_var = torch.exp(-logvar)
        mse_loss = torch.sum((target_delta - mean) ** 2 * inv_var, dim=-1)
        var_loss = torch.sum(logvar, dim=-1)
        return 0.5 * torch.mean(mse_loss + var_loss)


def train_model(model, s, a, sn, epochs=FULL_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """Full-batch Adam training of a dynamics model (PETS cartpole: 100 epochs)."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = s.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = model.nll(s[idx], a[idx], sn[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def validation_ll(model, s, a, sn):
    """Mean log-likelihood on a held-out set (used for the LL-Reward scatter)."""
    model.eval()
    with torch.no_grad():
        nll = model.nll(s, a, sn).item()
    model.train()
    return -nll


# ---- Dataset generators (3 cartpole protocols, paper Tab. tab:dataset) ----

STATE_RANGES = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.5, 0.5], [-3.0, 3.0]])
ACTION_RANGE = np.array([-1.0, 1.0])


def make_grid_dataset(env, size=CP_GRID_SIZE):
    """Uniform slicing of the 5-dim state-action space -> 7^5 = 16807 tuples."""
    n_bins = round(size ** (1.0 / 5.0))
    while n_bins**5 < size:
        n_bins += 1
    axes = []
    for lo, hi in STATE_RANGES:
        axes.append(np.linspace(lo, hi, n_bins))
    axes.append(np.linspace(ACTION_RANGE[0], ACTION_RANGE[1], n_bins))
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([m.ravel() for m in mesh], axis=1)[:size]
    states = points[:, :4].astype(np.float32)
    actions = points[:, 4:5].astype(np.float32)
    next_states = np.empty_like(states)
    for i in range(size):
        env.set_state(states[i].copy())
        next_states[i], _, _ = env.step(float(actions[i, 0]))
    return states, actions, next_states


def _lqr_gains():
    """LQR state-feedback gain for the linearized continuous cartpole (balances
    reliably, enabling high-reward 'expert' trajectory collection)."""
    m = 0.1
    M = 1.0
    l = 0.5
    g = 9.8
    Mtot = M + m
    c1 = 1.0 / (l * (4.0 / 3.0 - m / Mtot))
    c2 = g * c1
    c3 = c1 / Mtot
    A = np.array([[0, 1, 0, 0], [0, 0, -m * l * c2 / Mtot, 0],
                  [0, 0, 0, 1], [0, 0, c2, 0]])
    B = np.array([[0], [1 / Mtot - m * l * c3 / Mtot], [0], [-c3]])
    Q = np.diag([10.0, 1.0, 10.0, 1.0])
    R = np.array([[1.0]])
    if solve_continuous_are is not None:
        P = solve_continuous_are(A, B, Q, R)
        return np.linalg.solve(R, B.T @ P).flatten()
    return np.array([-3.16227766, -4.67318987, -38.34455367, -9.84935501])


def _run_controller_rollout(env, seed, filter_threshold=None, max_episodes=None,
                            max_attempts=10000, gains=None):
    """Collect controller rollouts; optionally keep only episodes with
    reward > filter_threshold (expert). Returns (s, a, s'), episodes kept."""
    if gains is None:
        gains = _lqr_gains()
    np.random.seed(seed)
    states, actions, next_states = [], [], []
    kept = 0
    attempts = 0
    while max_episodes is None or kept < max_episodes:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not collect {max_episodes} high-reward episodes "
                f"({kept} found, {max_attempts} attempts). Controller too weak."
            )
        state = env.reset(random_init=True)
        done = False
        ep_s, ep_a, ep_n = [], [], []
        while not done:
            action = float(np.clip(-gains @ state, -1.0, 1.0))
            next_state, _, done = env.step(action)
            ep_s.append(state)
            ep_a.append([action])
            ep_n.append(next_state)
            state = next_state
        reward = len(ep_s)
        if filter_threshold is None or reward > filter_threshold:
            states.extend(ep_s)
            actions.extend(ep_a)
            next_states.extend(ep_n)
            kept += 1
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
    )


def make_expert_dataset(env, seed, size=CP_EXPERT_SIZE):
    """High-reward (r>179) on-policy trajectories, ~2400 points."""
    return _run_controller_rollout(env, seed, filter_threshold=EXPERT_REWARD_THRESHOLD,
                                   max_episodes=int(np.ceil(size / 200.0)))


def make_onpolicy_dataset(env, seed, size=CP_ONPOLICY_SIZE):
    """On-policy working-controller data (no reward filter), ~3780 points."""
    return _run_controller_rollout(env, seed, filter_threshold=None,
                                   max_episodes=int(np.ceil(size / 200.0)))


def split_dataset(s, a, sn, split=TRAIN_SPLIT):
    n = len(s)
    idx = np.random.permutation(n)
    k = int(split * n)
    tr, va = idx[:k], idx[k:]
    return (
        torch.tensor(s[tr], device=device), torch.tensor(a[tr], device=device),
        torch.tensor(sn[tr], device=device),
        torch.tensor(s[va], device=device), torch.tensor(a[va], device=device),
        torch.tensor(sn[va], device=device),
    )


# ---- PETS / CEM controller ----

class CEMPlanner:
    """Cross-Entropy Method planner (PETS params: H=25, N=400, elites=40, iters=5)."""

    def __init__(self, model, horizon=None, num_samples=None,
                 num_elites=None, iterations=None):
        self.model = model
        self.horizon = CEM_HORIZON if horizon is None else horizon
        self.num_samples = CEM_SAMPLES if num_samples is None else num_samples
        self.num_elites = CEM_ELITES if num_elites is None else num_elites
        self.iterations = CEM_ITERATIONS if iterations is None else iterations

    @staticmethod
    def _reward(states):
        x = states[:, 0]
        theta = states[:, 2]
        return torch.exp(-(theta**2 / 0.05 + x**2 / 1.0))

    def plan(self, current_state):
        mean_actions = np.zeros(self.horizon)
        std_actions = np.ones(self.horizon) * 0.5
        cur = torch.tensor(current_state, dtype=torch.float32, device=device)
        for _ in range(self.iterations):
            action_samples = torch.tensor(
                np.clip(np.random.normal(mean_actions, std_actions,
                                         size=(self.num_samples, self.horizon)),
                        -1.0, 1.0),
                dtype=torch.float32, device=device)
            sim_states = cur.expand(self.num_samples, -1).clone()
            total_rewards = torch.zeros(self.num_samples, device=device)
            for t in range(self.horizon):
                delta_mean, _ = self.model(sim_states, action_samples[:, t : t + 1])
                sim_states = sim_states + delta_mean
                total_rewards += self._reward(sim_states)
            elites = action_samples[torch.argsort(total_rewards)[-self.num_elites:]]
            mean_actions = elites.mean(dim=0).cpu().numpy()
            std_actions = elites.std(dim=0).cpu().numpy() + 1e-4
        return mean_actions[0]


def evaluate_reward(env, model, num_trials=None):
    """Mean episode reward over num_trials independent rollouts."""
    if num_trials is None:
        num_trials = REWARD_TRIALS
    model.eval()
    planner = CEMPlanner(model)
    rewards = []
    for _ in range(num_trials):
        state = env.reset(random_init=False)
        total = 0.0
        done = False
        while not done:
            with torch.no_grad():
                action = planner.plan(state)
            next_state, reward, done = env.step(action)
            total += reward
            state = next_state
        rewards.append(total)
    model.train()
    return np.mean(rewards)


# ---- Core experiment (Fig 3: LL vs Reward scatter across M models) ----

def run_scatter_experiment(datasets, M=100, epochs=FULL_EPOCHS, num_trials=REWARD_TRIALS,
                           cem_horizon=CEM_HORIZON, cem_samples=CEM_SAMPLES,
                           cem_elites=CEM_ELITES, cem_iters=CEM_ITERATIONS,
                           save_path="objective_mismatch_scatter.png"):
    """For each dataset, train M fresh P models; record (val LL, mean reward)."""
    global CEM_HORIZON, CEM_SAMPLES, CEM_ELITES, CEM_ITERATIONS, REWARD_TRIALS
    CEM_HORIZON, CEM_SAMPLES, CEM_ELITES, CEM_ITERATIONS = (
        cem_horizon, cem_samples, cem_elites, cem_iters)
    REWARD_TRIALS = num_trials
    env = make_env("cartpole")
    results = {}
    for name, (s, a, sn) in datasets.items():
        tr, av, tn, vr, va, vn = split_dataset(s, a, sn)
        lls, rewards = [], []
        for m in range(M):
            model = ProbabilisticDynamicsModel().to(device)
            train_model(model, tr, av, tn, epochs=epochs)
            lls.append(validation_ll(model, vr, va, vn))
            rewards.append(evaluate_reward(env, model, num_trials=num_trials))
        rho = np.corrcoef(lls, rewards)[0, 1]
        results[name] = {"ll": np.array(lls), "reward": np.array(rewards), "rho": rho}
        print(f"[{name}] M={M} | rho(LL, reward) = {rho:.3f}")
    plot_scatter(results, save_path)
    return results


def plot_scatter(results, save_path):
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5), sharey=True)
    if len(results) == 1:
        axes = [axes]
    colors = {"expert": "tab:green", "on-policy": "tab:orange", "grid": "tab:blue"}
    for ax, (name, res) in zip(axes, results.items()):
        ax.scatter(res["ll"], res["reward"], alpha=0.6, s=18,
                   color=colors.get(name, "tab:gray"))
        ax.set_title(f"{name}  (rho = {res['rho']:.2f})")
        ax.set_xlabel("Validation Log-Likelihood")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Mean Episode Reward (10 trials)")
    fig.suptitle("Objective Mismatch (Lambert et al., 2020) - LL vs Reward", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    print(f"Grafik berhasil disimpan sebagai '{save_path}'.")


# ---- Half-cheetah scaffold (MuJoCo-backed env; PETS protocol is planned work) ----

HC_PARAMS = {
    "net_width": 200, "net_depth": 3, "batch_size": 64,
    "init_epochs": 20, "incr_epochs": 10,
    "cem": {"horizon": 30, "samples": 500, "elites": 50, "iterations": 5},
    "datasets": {"expert": 3000, "on-policy": 90900, "sampled": 200000},
}


def run_halfcheetah_experiment():
    """Half-cheetah replication. The MuJoCo env (MujocoHalfCheetahEnv) is ready,
    but the PETS incremental training + dataset protocols are planned work."""
    raise NotImplementedError(
        "Half-cheetah experiment is planned work: it needs the PETS incremental "
        "training protocol, 17-dim obs mapping, and dataset collection per "
        "Lambert et al. 2020 Appendix. See AGENTS.md."
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Reproduce Lambert et al. 2020 Fig 3: LL vs Reward scatter "
                    "across M dynamics models on 3 cartpole dataset types "
                    "(MuJoCo-backed when available).")
    parser.add_argument("--M", type=int, default=100, help="models per dataset")
    parser.add_argument("--epochs", type=int, default=FULL_EPOCHS)
    parser.add_argument("--trials", type=int, default=REWARD_TRIALS)
    parser.add_argument("--horizon", type=int, default=CEM_HORIZON)
    parser.add_argument("--samples", type=int, default=CEM_SAMPLES)
    parser.add_argument("--elites", type=int, default=CEM_ELITES)
    parser.add_argument("--iters", type=int, default=CEM_ITERATIONS)
    parser.add_argument("--out", default="objective_mismatch_scatter.png")
    args = parser.parse_args()

    env = make_env("cartpole")
    datasets = {
        "grid": make_grid_dataset(env),
        "expert": make_expert_dataset(env, seed=0),
        "on-policy": make_onpolicy_dataset(env, seed=0),
    }
    for name, (s, a, sn) in datasets.items():
        print(f"[{name}] dataset size = {len(s)}")

    run_scatter_experiment(
        datasets, M=args.M, epochs=args.epochs, num_trials=args.trials,
        cem_horizon=args.horizon, cem_samples=args.samples,
        cem_elites=args.elites, cem_iters=args.iters, save_path=args.out)


if __name__ == "__main__":
    main()
