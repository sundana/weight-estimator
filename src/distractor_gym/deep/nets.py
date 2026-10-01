"""torch networks for the WP1 deep diagnostics.

Provides the value-aware model (probabilistic Gaussian ensemble), the twin critic, and
the squashed Gaussian actor used by the offline model-learning and SAC machinery.
Torch is a hard dependency of this module; ``distractor_gym.deep`` does not import it
eagerly, so the base package stays torch-free.
"""

from __future__ import annotations

import torch
from torch import nn


def make_mlp(
    in_dim: int,
    out_dim: int,
    hidden: int = 256,
    n_layers: int = 2,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    """Plain MLP trunk ``in_dim -> hidden x n_layers -> out_dim``."""
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(last, hidden), activation()]
        last = hidden
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class GaussianEnsemble(nn.Module):
    """Probabilistic next-state model: an ensemble of diagonal Gaussians.

    Each ensemble member maps ``(s, a)`` to ``(mean, log_var)`` over the next state.
    ``forward`` returns stacked tensors of shape ``(batch, n_models, state_dim)``. The
    spread across members is the epistemic uncertainty used by the calibrated weight.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        n_models: int = 5,
        hidden: int = 256,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_models = n_models
        self.state_dim = state_dim
        self.members = nn.ModuleList(
            make_mlp(in_dim, 2 * state_dim, hidden=hidden, n_layers=n_layers)
            for _ in range(n_models)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outs = torch.stack([m(x) for m in self.members], dim=1)
        mean, log_var = outs.chunk(2, dim=-1)
        return mean, log_var.clamp(-10.0, 5.0)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean prediction and epistemic std across members, shape ``(batch, state_dim)``."""
        mean, _ = self.forward(x)
        return mean.mean(dim=1), mean.std(dim=1)


class TwinCritic(nn.Module):
    """Twin Q-network ``Q(s, a)`` with separate heads for the SAC pessimistic target."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 256,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.q1 = make_mlp(obs_dim + act_dim, 1, hidden=hidden, n_layers=n_layers)
        self.q2 = make_mlp(obs_dim + act_dim, 1, hidden=hidden, n_layers=n_layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q_min(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs, act)
        return torch.minimum(q1, q2)


class StateValue(nn.Module):
    """State-value head ``V(s)`` used for per-sample value gradients (VaGraM weights)."""

    def __init__(self, obs_dim: int, hidden: int = 256, n_layers: int = 2) -> None:
        super().__init__()
        self.net = make_mlp(obs_dim, 1, hidden=hidden, n_layers=n_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class SquashedGaussianActor(nn.Module):
    """SAC actor: diagonal Gaussian squashed by ``tanh`` into the action box.

    ``forward`` returns the pre-squash mean and log-std; ``sample`` returns an action,
    its reparameterized log-probability, and the deterministic ``tanh(mean)`` action.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 256,
        n_layers: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.trunk = make_mlp(obs_dim, 2 * act_dim, hidden=hidden, n_layers=n_layers)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.trunk(obs).chunk(2, dim=-1)
        return mean, log_std.clamp(self.log_std_min, self.log_std_max)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log1p(-action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)
