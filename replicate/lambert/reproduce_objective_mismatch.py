import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Menentukan perangkat komputasi (CPU / GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eksperimen berjalan pada perangkat: {device}")

class ContinuousCartPoleEnv:
    """
    Simulasi dinamika fisik Continuous CartPole 
    State: [x (posisi kart), x_dot (kecepatan), theta (sudut pole), theta_dot (kecepatan sudut)]
    Action: continuous force applied to cart [-10.0, 10.0] N
    """
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # Separuh panjang pole
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # Langkah waktu integrasi (dt)
        self.max_steps = 200

        # Ambang batas terminasi
        self.theta_threshold_radians = 15 * 2 * np.pi / 360  # ~15 derajat
        self.x_threshold = 2.4

        self.state = None
        self.steps_beyond_done = None
        self.current_step = 0

    def reset(self, random_init=True):
        if random_init:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        else:
            self.state = np.zeros(4)
        self.current_step = 0
        return self.state.copy()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        force = action * self.force_mag

        x, x_dot, theta, theta_dot = self.state

        # Persamaan gerak diferensial non-linear CartPole
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Integrasi numerik Euler
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.current_step += 1

        # Evaluasi kondisi jatuh / keluar batas
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.current_step >= self.max_steps
        )

        # Reward = 1 jika seimbang, 0 jika jatuh (maksimal 200 reward per episode)
        reward = 1.0 if not (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        ) else 0.0

        return self.state.copy(), reward, done

class ProbabilisticDynamicsModel(nn.Module):
    """
    Probabilistic Neural Network yang memprediksi transisi delta_state = s_{t+1} - s_t.
    Output: mu (rata-rata) dan log_var (logaritma variansi) untuk Gaussian distribution.
    """
    def __init__(self, state_dim=4, action_dim=1, hidden_dim=256):
        super(ProbabilisticDynamicsModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, state_dim)
        self.fc_logvar = nn.Linear(hidden_dim, state_dim)

        self.relu = nn.ReLU()
        # Batas variansi untuk stabilitas numerik
        self.max_logvar = nn.Parameter(torch.ones(1, state_dim) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(1, state_dim) * -10.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        # Menjepit nilai variansi agar tidak meledak atau menuju 0 absolut
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)

        return mean, logvar

    def loss_nll(self, state, action, next_state):
        """
        Menghitung Gaussian Negative Log-Likelihood (NLL).
        Meminimalkan NLL setara dengan memaksimalkan Log-Likelihood.
        """
        target_delta = next_state - state
        mean, logvar = self.forward(state, action)
        inv_var = torch.exp(-logvar)

        # Formula NLL: 0.5 * [ log(2*pi*sigma^2) + (target - mean)^2 / sigma^2 ]
        mse_loss = torch.sum((target_delta - mean)**2 * inv_var, dim=-1)
        var_loss = torch.sum(logvar, dim=-1)
        total_nll = 0.5 * torch.mean(mse_loss + var_loss)
        return total_nll

class CEMPlanner:
    """
    Cross-Entropy Method (CEM) untuk mencari sekuens aksi optimal berdasarkan model dinamika.
    """
    def __init__(self, model, horizon=15, num_samples=150, num_elites=25, iterations=4):
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.iterations = iterations
        self.action_dim = 1

    def compute_reward_batch(self, states):
        """Fungsi reward prediktif di dalam simulasi imajinasi MPC"""
        x = states[:, 0]
        theta = states[:, 2]
        # Reward berkelanjutan berdasarkan kedekatan dengan titik seimbang
        reward = np.exp(- (theta**2 / 0.05 + x**2 / 1.0))
        return reward

    def plan(self, current_state):
        mean_actions = np.zeros(self.horizon)
        std_actions = np.ones(self.horizon) * 0.5

        for _ in range(self.iterations):
            # 1. Sampel kandidat sekuens aksi
            action_samples = np.random.normal(
                mean_actions, std_actions, size=(self.num_samples, self.horizon)
            )
            action_samples = np.clip(action_samples, -1.0, 1.0)

            # 2. Gulirkan lintasan (rollout) imajinasi menggunakan model
            sim_states = np.tile(current_state, (self.num_samples, 1))
            total_rewards = np.zeros(self.num_samples)

            for t in range(self.horizon):
                actions_t = action_samples[:, t:t+1]
                with torch.no_grad():
                    s_tensor = torch.tensor(sim_states, dtype=torch.float32, device=device)
                    a_tensor = torch.tensor(actions_t, dtype=torch.float32, device=device)
                    delta_mean, _ = self.model(s_tensor, a_tensor)
                    next_sim_states = sim_states + delta_mean.cpu().numpy()

                rewards_t = self.compute_reward_batch(sim_states)
                total_rewards += rewards_t
                sim_states = next_sim_states

            # 3. Pilih kandidat elite terbaik
            elite_indices = np.argsort(total_rewards)[-self.num_elites:]
            elites = action_samples[elite_indices]

            # 4. Perbarui distribusi distribusi probabilitas aksi
            mean_actions = np.mean(elites, axis=0)
            std_actions = np.std(elites, axis=0) + 1e-4

        # Mengembalikan aksi pertama dari rangkaian rencana terbaik
        return mean_actions[0]

def collect_dataset(env, num_episodes=20):
    """
    Mengumpulkan kumpulan data transisi (s, a, s') dari lingkungan nyata.
    """
    states, actions, next_states = [], [], []
    for _ in range(num_episodes):
        state = env.reset(random_init=True)
        done = False
        while not done:
            # Campuran aksi acak dan aksi stabilisasi sederhana untuk variasi data
            if np.random.rand() < 0.3:
                action = np.random.uniform(-1.0, 1.0)
            else:
                action = float(np.clip(-2.0 * state[2] - 0.5 * state[3], -1.0, 1.0))

            next_state, _, done = env.step(action)
            states.append(state)
            actions.append([action])
            next_states.append(next_state)
            state = next_state

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(next_states, dtype=np.float32)
    )

def evaluate_controller(env, model, num_eval_episodes=3):
    """
    Menguji model dinamika yang telah dilatih pada kontroler MPC di lingkungan nyata
    untuk mengukur Episode Reward sesungguhnya.
    """
    model.eval()
    planner = CEMPlanner(model, horizon=15, num_samples=100, num_elites=15, iterations=3)
    episode_rewards = []

    for _ in range(num_eval_episodes):
        state = env.reset(random_init=False)
        total_reward = 0
        done = False
        while not done:
            action = planner.plan(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
        episode_rewards.append(total_reward)

    model.train()
    return np.mean(episode_rewards)

def run_objective_mismatch_experiment(seeds=[42, 123, 456, 789, 101]):
    """
    Menjalankan eksperimen pada beberapa random seeds independen (M = len(seeds)),
    kemudian mengagregasi hasilnya untuk menghasilkan kurva halus dan pita simpangan baku.
    """
    env = ContinuousCartPoleEnv()
    num_epochs = 120
    eval_interval = 10
    batch_size = 64
    epochs_logged = [1] + list(range(eval_interval, num_epochs + 1, eval_interval))
    num_eval_pts = len(epochs_logged)

    all_val_nll = np.zeros((len(seeds), num_eval_pts))
    all_rewards = np.zeros((len(seeds), num_eval_pts))

    print(f"=== Memulai Replikasi Multi-Seed (Total: {len(seeds)} Seeds) ===")

    for s_idx, seed in enumerate(seeds):
        print(f"\n--- Menjalankan Seed {s_idx + 1}/{len(seeds)} (Seed ID: {seed}) ---")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 1. Kumpulkan dataset spesifik untuk seed ini
        s, a, s_next = collect_dataset(env, num_episodes=25)
        dataset_size = len(s)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        split = int(0.8 * dataset_size)
        train_idx, val_idx = indices[:split], indices[split:]

        s_train = torch.tensor(s[train_idx], device=device)
        a_train = torch.tensor(a[train_idx], device=device)
        sn_train = torch.tensor(s_next[train_idx], device=device)

        s_val = torch.tensor(s[val_idx], device=device)
        a_val = torch.tensor(a[val_idx], device=device)
        sn_val = torch.tensor(s_next[val_idx], device=device)

        # 2. Inisialisasi model dan optimizer baru per seed
        model = ProbabilisticDynamicsModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        eval_counter = 0

        for epoch in range(1, num_epochs + 1):
            perm = torch.randperm(s_train.size(0))
            for i in range(0, s_train.size(0), batch_size):
                batch_idx = perm[i:i + batch_size]
                loss = model.loss_nll(s_train[batch_idx], a_train[batch_idx], sn_train[batch_idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % eval_interval == 0 or epoch == 1:
                with torch.no_grad():
                    val_nll = model.loss_nll(s_val, a_val, sn_val).item()

                reward = evaluate_controller(env, model, num_eval_episodes=2)

                all_val_nll[s_idx, eval_counter] = val_nll
                all_rewards[s_idx, eval_counter] = reward
                print(f"Seed {seed} | Epoch {epoch:3d}/{num_epochs:3d} | Val NLL: {val_nll:7.3f} | Reward: {reward:5.1f}")
                eval_counter += 1

    mean_val_nll = np.mean(all_val_nll, axis=0)
    std_val_nll = np.std(all_val_nll, axis=0)

    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_blue = 'tab:blue'
    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('Validation Negative Log-Likelihood (NLL)', color=color_blue, fontsize=12)
    line1 = ax1.plot(epochs_logged, mean_val_nll, color=color_blue, linewidth=2.5, marker='o', label='Mean Val NLL')
    ax1.fill_between(
        epochs_logged,
        mean_val_nll - std_val_nll,
        mean_val_nll + std_val_nll,
        color=color_blue,
        alpha=0.25,
        label='NLL ± 1 Std Dev'
    )
    ax1.tick_params(axis='y', labelcolor=color_blue)
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2 = ax1.twinx()
    color_red = 'tab:red'
    ax2.set_ylabel('Episode Reward Lingkungan', color=color_red, fontsize=12)
    line2 = ax2.plot(epochs_logged, mean_rewards, color=color_red, linewidth=2.5, marker='s', label='Mean Episode Reward')
    ax2.fill_between(
        epochs_logged,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color=color_red,
        alpha=0.25,
        label='Reward ± 1 Std Dev'
    )
    ax2.tick_params(axis='y', labelcolor=color_red)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', framealpha=0.9)

    plt.title(
        f"Replikasi Objective Mismatch (Lambert et al., 2020)\nRata-rata & Pita Simpangan Baku dari {len(seeds)} Seeds Independen",
        fontsize=13,
        pad=12
    )
    plt.tight_layout()
    plt.savefig("objective_mismatch_multi_seed.png", dpi=300)
    print("\nGrafik berhasil disimpan sebagai 'objective_mismatch_multi_seed.png'.")
    plt.show()

if __name__ == "__main__":
    # Menjalankan 5 seeds untuk efisiensi waktu komputasi (bisa ditambahkan hingga 10 seeds)
    run_objective_mismatch_experiment(seeds=[42, 101, 202, 303, 404])