import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn

from distractor_gym.profiling.bench import (
    estimate_mlp_flops,
    finite_diff_scheme,
    measure,
    vjp_scheme,
)
from experiments.exp1_profiling import run


def test_measure_returns_positive_metrics():
    x = torch.randn(16, 4)
    ms, throughput, peak = measure(lambda: (x**2).sum(), 16, iters=5, warmup=1, device="cpu")
    assert ms > 0.0
    assert throughput > 0.0
    assert peak == 0.0


def test_estimate_mlp_flops_positive():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    assert estimate_mlp_flops(net, 10) > 0.0
    assert estimate_mlp_flops(net, 10, backward=True) == pytest.approx(
        2.0 * estimate_mlp_flops(net, 10)
    )


class _LinearScalar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x).squeeze(-1)


def test_finite_diff_matches_exact_on_linear():
    net = _LinearScalar()
    x = torch.randn(6, 5)
    exact = vjp_scheme(net)(x)
    fd = finite_diff_scheme(net, eps=1e-3)(x)
    assert torch.allclose(fd, exact, atol=1e-4)


def test_run_tiny(tmp_path):
    cfg = {
        "seed": 0,
        "device": "cpu",
        "obs_dim": 8,
        "hidden": 8,
        "n_layers": 1,
        "batch_sizes": [4, 8],
        "iters": 3,
        "warmup": 1,
        "fd_iters": 1,
        "fd_eps": 0.001,
        "stale_k": [2],
        "threshold": 2.2,
        "batch_threshold": 4,
    }
    run(cfg, str(tmp_path))
    rows = json.loads((tmp_path / "profiling.json").read_text())
    schemes = {r["scheme"] for r in rows}
    assert {"mle_forward", "vjp_exact", "finite_diff", "stale_K2"} <= schemes
    for r in rows:
        assert np.isfinite(r["ms_per_step"]) and r["ms_per_step"] > 0.0
        assert np.isfinite(r["overhead_vs_forward"])
