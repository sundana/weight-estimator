"""Offline data generation: a replay buffer and a compact SAC trainer.

The WP1 Exp 1.1 protocol needs two offline datasets -- random-policy rollouts and
``medium-replay`` SAC rollouts. ``collect_random`` produces the former; ``train_sac``
fills a ``ReplayBuffer`` that doubles as the latter. Replays can be persisted with
``save``/``load`` so the deep diagnostics are reproducible without re-training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .nets import SquashedGaussianActor, TwinCritic


@dataclass
class ReplayBuffer:
    """Fixed-capacity transition buffer stored as numpy arrays."""

    capacity: int
    obs_dim: int
    act_dim: int

    def __post_init__(self) -> None:
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)
        self.terminal = np.zeros((self.capacity,), dtype=np.float32)
        self._idx = 0
        self._size = 0

    def add(self, obs, act, rew, next_obs, done, terminal=None) -> None:
        i = self._idx
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.terminal[i] = float(done) if terminal is None else float(terminal)
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        rng: np.random.Generator | None = None,
    ) -> dict:
        if rng is None:
            idx = np.random.randint(0, self._size, size=batch_size)
        else:
            idx = rng.integers(0, self._size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "act": torch.as_tensor(self.act[idx], device=device),
            "rew": torch.as_tensor(self.rew[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "done": torch.as_tensor(self.done[idx], device=device),
            "terminal": torch.as_tensor(self.terminal[idx], device=device),
        }

    def arrays(self) -> dict:
        n = self._size
        return {
            "obs": self.obs[:n].copy(),
            "act": self.act[:n].copy(),
            "rew": self.rew[:n].copy(),
            "next_obs": self.next_obs[:n].copy(),
            "done": self.done[:n].copy(),
            "terminal": self.terminal[:n].copy(),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays())
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        data = np.load(path)
        n, obs_dim = data["obs"].shape
        buf = cls(capacity=n, obs_dim=obs_dim, act_dim=data["act"].shape[1])
        for key in ("obs", "act", "rew", "next_obs", "done"):
            getattr(buf, key)[:n] = data[key]
        if "terminal" in data:
            buf.terminal[:n] = data["terminal"]
        else:
            buf.terminal[:n] = buf.done[:n]
        buf._idx = 0
        buf._size = n
        return buf


def collect_random(env, n_steps: int, seed: int = 0, action_low=None, action_high=None) -> ReplayBuffer:
    """Roll out a uniform random policy and fill a replay buffer."""
    obs, _ = env.reset(seed=seed)
    low = env.action_space.low if action_low is None else action_low
    high = env.action_space.high if action_high is None else action_high
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape) or 1)
    buf = ReplayBuffer(capacity=n_steps, obs_dim=obs_dim, act_dim=act_dim)
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        act = rng.uniform(low, high).astype(np.float32).reshape(act_dim)
        next_obs, rew, term, trunc, _ = env.step(act)
        done = term or trunc
        buf.add(obs, act, rew, next_obs, done, terminal=term)
        obs = next_obs
        if done:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    return buf


class SACAgent:
    """Soft actor-critic with twin critics, squashed Gaussian actor, and auto-entropy."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 256,
        n_layers: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        alpha: float | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden=hidden, n_layers=n_layers).to(self.device)
        self.critic = TwinCritic(obs_dim, act_dim, hidden=hidden, n_layers=n_layers).to(self.device)
        self.target_critic = TwinCritic(obs_dim, act_dim, hidden=hidden, n_layers=n_layers).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.auto_alpha = alpha is None
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.fixed_alpha = 0.2 if alpha is None else float(alpha)

    @property
    def alpha(self) -> torch.Tensor:
        if self.auto_alpha:
            return self.log_alpha.exp()
        return torch.tensor(self.fixed_alpha, device=self.device)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            _, _, action = self.actor.sample(obs_t)
        else:
            action, _, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: dict) -> dict:
        obs, act = batch["obs"], batch["act"]
        rew, next_obs, done = batch["rew"], batch["next_obs"], batch["done"]
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_obs)
            q1t, q2t = self.target_critic(next_obs, next_action)
            q_target = torch.minimum(q1t, q2t) - self.alpha * next_logp.squeeze(-1)
            target = rew + self.gamma * (1.0 - done) * q_target
        q1, q2 = self.critic(obs, act)
        critic_loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        action, log_prob, _ = self.actor.sample(obs)
        q = self.critic.q_min(obs, action)
        actor_loss = (self.alpha.detach() * log_prob.squeeze(-1) - q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
        }


def train_sac(
    env,
    n_steps: int,
    seed: int = 0,
    start_steps: int = 1000,
    batch_size: int = 256,
    update_after: int = 1000,
    update_every: int = 50,
    per_update: int = 1,
    hidden: int = 256,
    n_layers: int = 2,
    device: torch.device | str = "cpu",
) -> tuple[SACAgent, ReplayBuffer]:
    """Train SAC on ``env`` and return the agent with its filled replay buffer.

    The buffer collected here is the ``medium-replay`` dataset for the WP1 diagnostics.
    """
    obs, _ = env.reset(seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape) or 1)
    buf = ReplayBuffer(capacity=n_steps, obs_dim=obs_dim, act_dim=act_dim)
    agent = SACAgent(obs_dim, act_dim, hidden=hidden, n_layers=n_layers, device=device)
    rng = np.random.default_rng(seed)
    for step in range(n_steps):
        if step < start_steps:
            act = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            act = act.reshape(act_dim)
        else:
            act = agent.select_action(obs)
        next_obs, rew, term, trunc, _ = env.step(act)
        done = term or trunc
        buf.add(obs, act, rew, next_obs, done, terminal=term)
        obs = next_obs
        if done:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        if step >= update_after and step % update_every == 0 and len(buf) >= batch_size:
            for _ in range(per_update):
                agent.update(buf.sample(batch_size, device=agent.device, rng=rng))
    return agent, buf
