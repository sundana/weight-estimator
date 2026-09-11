"""Dataset generators for the three cartpole protocols (paper Tab. tab:dataset)."""

import numpy as np
import torch
from tqdm import tqdm

from .config import (ACTION_RANGE, CP_EXPERT_SIZE, CP_GRID_SIZE, CP_ONPOLICY_SIZE,
                     EXPERT_REWARD_THRESHOLD, STATE_RANGES, TRAIN_SPLIT, device)

try:
    from scipy.linalg import solve_continuous_are
except ImportError:  # pragma: no cover
    solve_continuous_are = None


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
    for i in tqdm(range(size), desc=f"grid ({size})", unit="pt",
                  dynamic_ncols=True):
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
                            max_attempts=10000, gains=None, solved_only=False):
    """Collect controller rollouts; optionally keep only episodes with
    reward > filter_threshold (expert) or that fully solve the task
    (on-policy, per paper: data from a trial that solved the task).
    Returns (s, a, s'), episodes kept."""
    if gains is None:
        gains = _lqr_gains()
    np.random.seed(seed)
    states, actions, next_states = [], [], []
    kept = 0
    attempts = 0
    pbar = tqdm(total=max_episodes, desc="rollout", unit="ep",
                dynamic_ncols=True) if max_episodes else None
    while max_episodes is None or kept < max_episodes:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not collect {max_episodes} qualifying episodes "
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
        keep = filter_threshold is not None and reward > filter_threshold
        keep = keep or (solved_only and reward >= env.max_steps)
        if keep:
            states.extend(ep_s)
            actions.extend(ep_a)
            next_states.extend(ep_n)
            kept += 1
            if pbar is not None:
                pbar.set_postfix({"attempts": attempts, "len": reward}, refresh=False)
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
    )


def make_expert_dataset(env, seed, size=CP_EXPERT_SIZE):
    """High-reward (r>179) on-policy trajectories, ~2400 points (LQR proxy)."""
    return _run_controller_rollout(env, seed, filter_threshold=EXPERT_REWARD_THRESHOLD,
                                   max_episodes=int(np.ceil(size / 200.0)))


def make_onpolicy_dataset(env, seed, size=CP_ONPOLICY_SIZE):
    """On-policy data from the end of trials that solved the task (~3780 pts),
    per the paper (not unfiltered rollouts)."""
    return _run_controller_rollout(env, seed, filter_threshold=None,
                                   solved_only=True,
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