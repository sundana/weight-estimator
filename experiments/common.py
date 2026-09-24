"""Shared helpers for tabular experiment entry points."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml

from distractor_gym.agents import fit_empirical_model, fit_feature_model
from distractor_gym.core import DistractorClass, RegimeConfig
from distractor_gym.tabular import Grid, TabularDistractorEnv

REGIME_FIELDS = {f.name for f in fields(RegimeConfig)}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(regime: dict, seed: int | None = None) -> TabularDistractorEnv:
    """Build a ``TabularDistractorEnv`` from a regime dict with an optional seed override."""
    regime = dict(regime)
    dc = regime.get("distractor_class")
    if isinstance(dc, str):
        regime["distractor_class"] = DistractorClass(dc)
    cfg = RegimeConfig(**regime)
    if seed is not None:
        cfg.seed = seed
    gc = regime.get("grid_c")
    gd = regime.get("grid_d")
    grid_c = Grid(**gc) if gc else None
    grid_d = Grid(**gd) if gd else None
    return TabularDistractorEnv(cfg, grid_c=grid_c, grid_d=grid_d)


def save_json(obj: dict, out_dir: str, name: str) -> Path:
    path = Path(out_dir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def git_sha() -> str:
    """Best-effort current git commit (with ``-dirty`` suffix) for run manifests."""
    try:
        root = Path(__file__).resolve().parent
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        )
        return f"{sha}-dirty" if dirty.stdout.strip() else sha
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def save_manifest(cfg: dict, out_dir: str, name: str = "manifest") -> Path:
    """Write a reproducibility manifest (git sha, config, versions) next to results."""
    manifest = {
        "git_sha": git_sha(),
        "config": cfg,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    return save_json(manifest, out_dir, name)


def fit_model(
    env: TabularDistractorEnv,
    data: np.ndarray,
    weights: np.ndarray | None,
    cfg: dict,
) -> np.ndarray:
    """Fit the configured transition model (empirical table or feature-budgeted)."""
    if cfg.get("model_kind", "empirical") == "feature":
        return fit_feature_model(
            env,
            data,
            weights,
            capacity=int(cfg.get("capacity", 1)),
            model_noise=float(cfg.get("model_noise", 0.5)),
        )
    return fit_empirical_model(env, data, weights, alpha=float(cfg.get("alpha", 0.1)))


def coverage_state_probs(
    env: TabularDistractorEnv, coverage: str, goal_std: float = 0.75
) -> np.ndarray:
    """State distribution for i.i.d. data collection: uniform or goal-concentrated."""
    if coverage == "uniform":
        return np.full(env.S, 1.0 / env.S)
    if coverage == "goal":
        xc = env.grid_c.points
        p = np.exp(-0.5 * ((xc - env.goal) / goal_std) ** 2)
        p = p / p.sum()
        return np.repeat(p / env.n_di, env.n_di)
    raise ValueError(f"unknown coverage: {coverage}")


def save_fig(fig, out_dir: str, name: str) -> Path:
    path = Path(out_dir) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def coverage_behavior(env: TabularDistractorEnv, coverage: str, gain: float = 1.5) -> np.ndarray:
    """Behavior policy for data collection: uniform or goal-directed coverage."""
    if coverage == "uniform":
        from distractor_gym.agents import uniform_behavior

        return uniform_behavior(env)
    if coverage == "goal":
        from distractor_gym.agents import goal_policy

        return goal_policy(env, gain=gain).probs()
    raise ValueError(f"unknown coverage: {coverage}")


def iter_sweep(cfg: dict) -> list[tuple[dict, dict]]:
    """Cartesian product over ``sweep`` keys, split into (regime, experiment knobs).

    Sweep keys matching ``RegimeConfig`` fields go to the regime dict; others are
    returned as experiment knobs (e.g. ``coverage``).
    """
    base = dict(cfg.get("regime", {}))
    sweep = cfg.get("sweep", {})
    if not sweep:
        return [(dict(base), {})]
    keys = list(sweep)
    out = []
    for combo in np.ndindex(*(len(v) for v in sweep.values())):
        regime, extra = dict(base), {}
        for k, i in zip(keys, combo):
            val = list(sweep[k])[i]
            (regime if k in REGIME_FIELDS else extra)[k] = val
        out.append((regime, extra))
    return out