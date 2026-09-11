"""Endpoint: reproduce Lambert et al. 2020 Fig 3 objective mismatch.

Builds the three cartpole datasets (grid / expert / on-policy), trains M
probabilistic dynamics models per dataset, evaluates the CEM-controlled mean
episode reward, and prints the Pearson correlation rho(LL, reward).

Usage:
    python replicate/lambert/reproduce_objective_mismatch.py [--M N] [--seed S]
        [--epochs N] [--trials N] [--dataset {all,grid,expert,on-policy}] [--render]
"""

import argparse
import random

import numpy as np
import torch

from objective_mismatch.config import FULL_EPOCHS, REWARD_TRIALS, device
from objective_mismatch.datasets import (make_expert_dataset, make_grid_dataset,
                                         make_onpolicy_dataset)
from objective_mismatch.envs import GYMNASIUM_AVAILABLE, MUJOCO_AVAILABLE, make_env
from objective_mismatch.experiment import run_pearson_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Lambert et al. 2020 Fig 3 objective mismatch: train "
                    "M probabilistic dynamics models per cartpole dataset type and "
                    "report the Pearson correlation between validation LL and mean "
                    "episode reward.")
    parser.add_argument("--M", type=int, default=1000, help="models per dataset")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--epochs", type=int, default=FULL_EPOCHS)
    parser.add_argument("--trials", type=int, default=REWARD_TRIALS)
    parser.add_argument("--dataset", choices=["all", "grid", "expert", "on-policy"],
                        default="all", help="which dataset(s) to use (default: all)")
    parser.add_argument("--render", action="store_true",
                        help="show the CartPole animation during reward evaluation "
                             "(rendering is slow; use a small --M and --trials)")
    args = parser.parse_args()

    print(f"Eksperimen berjalan pada perangkat: {device}  "
          f"(mujoco: {MUJOCO_AVAILABLE}, gymnasium: {GYMNASIUM_AVAILABLE})")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env("cartpole")
    all_datasets = {
        "grid": make_grid_dataset(env),
        "expert": make_expert_dataset(env, seed=0),
        "on-policy": make_onpolicy_dataset(env, seed=0),
    }
    datasets = all_datasets if args.dataset == "all" else {
        args.dataset: all_datasets[args.dataset]
    }
    for name, (s, a, sn) in datasets.items():
        print(f"[{name}] dataset size = {len(s)}")

    run_pearson_experiment(datasets, M=args.M, epochs=args.epochs,
                           num_trials=args.trials, render=args.render)


if __name__ == "__main__":
    main()