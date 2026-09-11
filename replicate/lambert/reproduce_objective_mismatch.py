"""Endpoint: reproduce Lambert et al. 2020 Fig 3 objective mismatch.

Builds the three cartpole datasets (grid / expert / on-policy), trains M
probabilistic dynamics models per dataset, evaluates the CEM-controlled mean
episode reward, and reports the Pearson correlation rho(LL, reward). The
per-model points, summary metrics, and Fig-3-style figures are saved under
--output.

Usage:
    python replicate/lambert/reproduce_objective_mismatch.py [--M N] [--seed S]
        [--epochs N] [--trials N] [--dataset {all,grid,expert,on-policy}]
        [--output DIR] [--render]
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from objective_mismatch.config import FULL_EPOCHS, REWARD_TRIALS, device
from objective_mismatch.datasets import (make_expert_dataset, make_grid_dataset,
                                         make_onpolicy_dataset)
from objective_mismatch.envs import GYMNASIUM_AVAILABLE, MUJOCO_AVAILABLE, make_env
from objective_mismatch.experiment import run_pearson_experiment
from objective_mismatch.report import save_results

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"


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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="directory for saved metrics and figures "
                             "(default: replicate/lambert/outputs)")
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

    results = run_pearson_experiment(datasets, M=args.M, epochs=args.epochs,
                                     num_trials=args.trials, render=args.render)

    saved = save_results(results, args.output)
    for path in saved:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()