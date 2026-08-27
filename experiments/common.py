"""Shared helpers for tabular experiment entry points."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml

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