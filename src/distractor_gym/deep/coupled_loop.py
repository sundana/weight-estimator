"""Online model-based RL coupled loop (Part II, Phase C).

Trains a probabilistic dynamics ensemble and a SAC actor/critic jointly on the deep
Distractor-Gym, so the value-aware weight ``w = w(V, grad V)`` is supplied by the critic
that is itself being learned (the weight-estimator feedback of ``paper/coval`` Part II).
Each iteration fits the model on real transitions under a loss family (MLE, VAML-1,
VaGraM, CVAML, or COVAL), rolls the model forward under the current policy to fill an
imagined buffer, updates the agent on real + imagined data, collects fresh real data, and
evaluates the policy on the true environment.

COVAL combines the VaGraM weight with the stabilization primitives: a detached target
critic, batch self-normalization, weight clipping, and curriculum annealing. The loop
records the policy return, the weight-estimator variance ``sigma_w^2``, the effective
sample size, and model-fit diagnostics for the return-level H1.1 test and the COVAL-vs-MLE
comparison.

Torch is imported here (this module lives under ``distractor_gym.deep``); the environment
views are supplied by the caller so the loop stays agnostic to the MuJoCo plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .nets import CriticValue, GaussianEnsemble
from .sac import ReplayBuffer, SACAgent
from .vjp import state_value_grad_norm

_FAMILIES = ("mle", "vaml1", "vagram", "cvaml", "coval")


@dataclass
class CoupledLoopConfig:
    """Knobs of the online coupled MBRL loop."""

    gamma: float = 0.99
    horizon: int = 5
    iterations: int = 10
    model_updates: int = 200
    agent_updates: int = 200
    batch_size: int = 256
    model_lr: float = 1e-3
    agent_lr: float = 3e-4
    n_models: int = 3
    hidden: int = 64
    n_layers: int = 2
    bottleneck: int | None = None
    family: str = "mle"
    tau: float = 0.5
    target_critic: bool = False
    self_norm: bool = True
    clip_wmax: float | None = None
    anneal_steps: int = 0
    weight_every: int = 1
    clip_sigma: float = 6.0
    init_steps: int = 1000
    collect_steps: int = 250
    rollouts_per_iter: int = 2
    rollout_batch: int = 256
    use_imagined: bool = True
    eval_every: int = 2
    eval_episodes: int = 3
    seed: int = 0


def _concat_batches(a: dict, b: dict) -> dict:
    """Concatenate two replay batches along the batch dimension."""
    return {k: torch.cat([a[k], b[k]], dim=0) for k in a}


class DeepCoupledLoop:
    """Online model/critic co-training loop over a deep Distractor-Gym.

    ``gym_env`` is the Gymnasium view used for real interaction and evaluation;
    ``torch_env`` is the differentiable view used only for the reward in imagined
    rollouts (the dynamics come from the learned ensemble).
    """

    def __init__(self, gym_env, torch_env, cfg: CoupledLoopConfig) -> None:
        if cfg.family not in _FAMILIES:
            raise ValueError(f"unknown family: {cfg.family}")
        self.gym_env = gym_env
        self.torch_env = torch_env
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.obs_dim = int(gym_env.dim)
        self.act_dim = 1
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self._model_update_count = 0

        real_capacity = cfg.init_steps + cfg.iterations * cfg.collect_steps + 16
        imag_capacity = max(
            16, cfg.iterations * cfg.rollouts_per_iter * cfg.rollout_batch * cfg.horizon + 16
        )
        self.real_buf = ReplayBuffer(real_capacity, self.obs_dim, self.act_dim)
        self.imagined_buf = ReplayBuffer(imag_capacity, self.obs_dim, self.act_dim)

        self.agent = SACAgent(
            self.obs_dim,
            self.act_dim,
            hidden=cfg.hidden,
            n_layers=cfg.n_layers,
            gamma=cfg.gamma,
            lr=cfg.agent_lr,
            device=self.device,
        )
        self.model = GaussianEnsemble(
            in_dim=self.obs_dim + self.act_dim,
            state_dim=self.obs_dim,
            n_models=cfg.n_models,
            hidden=cfg.hidden,
            n_layers=cfg.n_layers,
            bottleneck=cfg.bottleneck,
        ).to(self.device)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=cfg.model_lr)
        self._scalers_ready = False

    def _fit_scalers(self) -> None:
        arr = self.real_buf.arrays()
        x = np.concatenate([arr["obs"], arr["act"]], axis=-1)
        self.x_mean = torch.as_tensor(x.mean(0, keepdims=True), dtype=torch.float32)
        self.x_std = torch.as_tensor(x.std(0, keepdims=True) + 1e-6, dtype=torch.float32)
        self.y_mean = torch.as_tensor(arr["next_obs"].mean(0, keepdims=True), dtype=torch.float32)
        self.y_std = torch.as_tensor(arr["next_obs"].std(0, keepdims=True) + 1e-6, dtype=torch.float32)
        self._scalers_ready = True

    def _value_fn(self) -> CriticValue:
        critic = self.agent.target_critic if self.cfg.target_critic else self.agent.critic
        return CriticValue(self.agent.actor, critic)

    def _weights(self, batch: dict, eps_model: torch.Tensor, sigma_epi: torch.Tensor) -> torch.Tensor:
        fam = self.cfg.family
        obs, next_obs = batch["obs"], batch["next_obs"]
        if fam == "mle":
            return torch.ones(len(obs), device=obs.device)
        vfn = self._value_fn()
        if fam == "vaml1":
            return (vfn(next_obs) - vfn(obs)).abs()
        grad_norm = state_value_grad_norm(vfn, next_obs)
        if fam in ("vagram", "coval"):
            return grad_norm
        return grad_norm * eps_model.detach() / (sigma_epi.detach() + 1e-8)

    def _stabilize(self, w: torch.Tensor) -> torch.Tensor:
        if self.cfg.self_norm:
            w = w / (w.mean() + 1e-8)
        if self.cfg.clip_wmax is not None:
            w = torch.clamp(w, max=self.cfg.clip_wmax)
        if self.cfg.anneal_steps > 0:
            lam = min(1.0, self._model_update_count / self.cfg.anneal_steps)
            w = (1.0 - lam) + lam * w
        return w

    def _model_mean(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        mean_n, _ = self.model((x - self.x_mean) / self.x_std)
        raw = mean_n.mean(dim=1) * self.y_std + self.y_mean
        if self.cfg.clip_sigma and self.cfg.clip_sigma > 0.0:
            raw = torch.clamp(
                raw,
                self.y_mean - self.cfg.clip_sigma * self.y_std,
                self.y_mean + self.cfg.clip_sigma * self.y_std,
            )
        return raw

    def _model_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        obs, act, next_obs = batch["obs"], batch["act"], batch["next_obs"]
        x = torch.cat([obs, act], dim=-1)
        xn = (x - self.x_mean) / self.x_std
        yn = (next_obs - self.y_mean) / self.y_std
        mean_n, _ = self.model(xn)
        per_member = ((mean_n - yn.unsqueeze(1)) ** 2).sum(dim=-1)
        with torch.no_grad():
            pred_raw = mean_n.mean(dim=1) * self.y_std + self.y_mean
            sigma_epi = (mean_n * self.y_std).std(dim=1).mean(dim=-1)
            eps_model = (pred_raw - next_obs).norm(dim=-1)
            w = self._weights(batch, eps_model, sigma_epi)
            w = self._stabilize(w)
            ess = float(w.sum() ** 2 / (w.pow(2).sum() + 1e-12))
            diagnostics = {
                "weight_var": float(w.var()),
                "weight_mean": float(w.mean()),
                "ess": ess,
                "one_step_mse": float((eps_model**2).mean()),
            }
        loss = (w.unsqueeze(1) * per_member).mean()
        return loss, diagnostics

    def _train_model(self) -> dict:
        last = {}
        n = len(self.real_buf)
        for step in range(self.cfg.model_updates):
            batch = self.real_buf.sample(
                min(self.cfg.batch_size, n), device=self.device, rng=self.rng
            )
            loss, diagnostics = self._model_loss(batch)
            self.model_opt.zero_grad()
            loss.backward()
            self.model_opt.step()
            self._model_update_count += 1
            last = diagnostics
            if not np.isfinite(float(loss.detach())):
                return {**last, "diverged": 1.0, "model_loss": float(loss.detach())}
        return {**last, "diverged": 0.0, "model_loss": float(loss.detach())}

    def _imagine(self) -> None:
        n = len(self.real_buf)
        for _ in range(self.cfg.rollouts_per_iter):
            idx = self.rng.integers(0, n, size=self.cfg.rollout_batch)
            obs = torch.as_tensor(self.real_buf.obs[idx], dtype=torch.float32)
            s = obs
            for _ in range(self.cfg.horizon):
                with torch.no_grad():
                    a, _, _ = self.agent.actor.sample(s)
                    r = self.torch_env.reward(s, a)
                    s_next = self._model_mean(s, a)
                for i in range(len(s)):
                    self.imagined_buf.add(
                        s[i].numpy(),
                        a[i].numpy(),
                        float(r[i]),
                        s_next[i].numpy(),
                        0.0,
                        terminal=0.0,
                    )
                s = s_next

    def _agent_batch(self) -> dict:
        half = self.cfg.batch_size // 2
        if (
            self.cfg.use_imagined
            and len(self.imagined_buf) >= half
            and len(self.real_buf) >= half
        ):
            real = self.real_buf.sample(half, device=self.device, rng=self.rng)
            imag = self.imagined_buf.sample(half, device=self.device, rng=self.rng)
            return _concat_batches(real, imag)
        return self.real_buf.sample(
            min(self.cfg.batch_size, len(self.real_buf)), device=self.device, rng=self.rng
        )

    def _train_agent(self) -> float:
        last = 0.0
        for _ in range(self.cfg.agent_updates):
            out = self.agent.update(self._agent_batch())
            last = out["critic_loss"]
        return last

    def _collect_real(self, steps: int, random_policy: bool) -> None:
        obs, _ = self.gym_env.reset(seed=int(self.rng.integers(0, 2**31)))
        for _ in range(steps):
            if random_policy:
                action = self.rng.uniform(-1.0, 1.0, size=self.act_dim).astype(np.float32)
            else:
                action = self.agent.select_action(obs)
            next_obs, rew, term, trunc, _ = self.gym_env.step(action)
            done = term or trunc
            self.real_buf.add(obs, action, rew, next_obs, done, terminal=term)
            obs = next_obs
            if done:
                obs, _ = self.gym_env.reset(seed=int(self.rng.integers(0, 2**31)))

    def _evaluate(self, episodes: int) -> float:
        returns = []
        for e in range(episodes):
            obs, _ = self.gym_env.reset(seed=self.cfg.seed * 1000 + e)
            done = False
            total = 0.0
            while not done:
                action = self.agent.select_action(obs, deterministic=True)
                obs, rew, term, trunc, _ = self.gym_env.step(action)
                total += float(rew)
                done = term or trunc
            returns.append(total)
        return float(np.mean(returns))

    def run(self) -> list[dict]:
        """Run the loop and return the per-evaluation history."""
        self._collect_real(self.cfg.init_steps, random_policy=True)
        self._fit_scalers()
        history = []
        for it in range(self.cfg.iterations):
            model_diag = self._train_model()
            if self.cfg.use_imagined:
                self._imagine()
            critic_loss = self._train_agent()
            self._collect_real(self.cfg.collect_steps, random_policy=False)
            if it % self.cfg.eval_every == 0 or it == self.cfg.iterations - 1:
                history.append(
                    {
                        "iteration": it,
                        "return": self._evaluate(self.cfg.eval_episodes),
                        "critic_loss": critic_loss,
                        "model_loss": model_diag.get("model_loss"),
                        "weight_var": model_diag.get("weight_var"),
                        "ess": model_diag.get("ess"),
                        "diverged": model_diag.get("diverged", 0.0),
                        "real_size": len(self.real_buf),
                        "imagined_size": len(self.imagined_buf),
                    }
                )
        return history
