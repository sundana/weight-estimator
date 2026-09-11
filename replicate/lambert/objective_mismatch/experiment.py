"""Core experiment: Pearson correlation between validation LL and episode reward."""

import numpy as np
from tqdm import tqdm

from .config import FULL_EPOCHS, REWARD_TRIALS, device
from .datasets import split_dataset
from .envs import make_env
from .model import ProbabilisticDynamicsModel, train_model, validation_ll
from .planner import evaluate_reward


def run_pearson_experiment(datasets, M=100, epochs=FULL_EPOCHS, num_trials=REWARD_TRIALS,
                           render=False):
    """For each dataset train M fresh P models, compute validation LL and mean
    episode reward per model, and report the Pearson correlation rho(LL, reward).
    This is exactly the quantity that quantifies objective mismatch in Fig 3."""
    env = make_env("cartpole", render=render)
    for name, (s, a, sn) in datasets.items():
        tr, av, tn, vr, va, vn = split_dataset(s, a, sn)
        lls, rewards = [], []
        pbar = tqdm(range(M), desc=f"[{name}]", unit="model", dynamic_ncols=True)
        for m in pbar:
            model = ProbabilisticDynamicsModel().to(device)
            train_model(model, tr, av, tn, epochs=epochs)
            lls.append(validation_ll(model, vr, va, vn))
            rewards.append(evaluate_reward(env, model, num_trials=num_trials, render=render))
            pbar.set_postfix(
                {"mean_LL": f"{np.mean(lls):.3f}",
                 "mean_rew": f"{np.mean(rewards):.1f}",
                 "rho": (f"{np.corrcoef(lls, rewards)[0, 1]:+.3f}"
                         if len(lls) >= 2 else "---")},
                refresh=True)
        rho = np.corrcoef(lls, rewards)[0, 1] if len(lls) >= 2 else float("nan")
        print(f"[{name}] M={M} | rho(LL, reward) = {rho:.3f}")
        print(f"    mean LL = {np.mean(lls):.3f} | mean reward = {np.mean(rewards):.1f}")