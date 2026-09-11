"""CEM planner and model-based episode-reward evaluation."""

import numpy as np
import torch

from .config import CEM_ELITES, CEM_HORIZON, CEM_ITERATIONS, CEM_SAMPLES, REWARD_TRIALS, device


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


def evaluate_reward(env, model, num_trials=None, render=False):
    """Mean episode reward over num_trials independent rollouts."""
    if num_trials is None:
        num_trials = REWARD_TRIALS
    model.eval()
    planner = CEMPlanner(model)
    rewards = []
    for _ in range(num_trials):
        state = env.reset(random_init=True)
        total = 0.0
        done = False
        while not done:
            with torch.no_grad():
                action = planner.plan(state)
            next_state, reward, done = env.step(action)
            if render:
                env.render()
            total += reward
            state = next_state
        rewards.append(total)
    model.train()
    return np.mean(rewards)