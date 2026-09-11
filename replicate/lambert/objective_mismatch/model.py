"""Probabilistic dynamics model (P) and its training/validation helpers."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import BATCH_SIZE, FULL_EPOCHS, LEARNING_RATE, NET_DEPTH, NET_WIDTH, device


class ProbabilisticDynamicsModel(nn.Module):
    """PETS probabilistic forward model (P). Predicts delta = s' - s as a
    Gaussian (mean, logvar), with bounded log-variance."""

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
    for _ in tqdm(range(epochs), desc="train", unit="ep", leave=False,
                  dynamic_ncols=True):
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