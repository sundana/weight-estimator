"""Hardware profiling harness for per-sample value-Jacobian computation (WP1 Exp 1.2).

Provides wall-clock, throughput, and peak-memory measurement of VJP computation
schemes (exact online ``torch.func.vmap``, stale-weight caching, finite differences,
and the forward-only MLE baseline) plus an analytic FLOP estimate for the TFLOPS
column. Torch is a hard dependency of this module; ``distractor_gym.profiling`` does
not import it eagerly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ..deep.vjp import StaleValueGrad, finite_difference_grad, make_value_grad_norm_fn


@dataclass
class BenchResult:
    """One scheme/batch measurement."""

    scheme: str
    batch: int
    ms_per_step: float
    throughput: float
    peak_mem_mb: float


def _use_cuda(device: torch.device | str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def measure(
    fn: Callable[[], object],
    batch_size: int,
    iters: int = 50,
    warmup: int = 10,
    device: torch.device | str = "cpu",
) -> tuple[float, float, float]:
    """Time ``fn`` over ``iters`` calls; return ``(ms_per_step, throughput, peak_mb)``.

    ``throughput`` is transitions per second (``batch_size`` per call). Peak memory is
    the max CUDA allocation during the measured block (0 on CPU).
    """
    for _ in range(warmup):
        fn()
    cuda = _use_cuda(device)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    ms_per_step = elapsed * 1e3
    throughput = batch_size / elapsed if elapsed > 0 else float("inf")
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if cuda else 0.0
    return ms_per_step, throughput, peak_mb


def forward_scheme(net: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Forward-only baseline (standard MLE forward pass)."""
    return lambda x: net(x)


def _value_grad_norm_fn(net: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    def single(s: torch.Tensor) -> torch.Tensor:
        return net(s.unsqueeze(0)).reshape(())

    return make_value_grad_norm_fn(single)


def vjp_scheme(net: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Exact online per-sample VJP via ``torch.func.vmap(grad)`` (transform built once)."""
    return _value_grad_norm_fn(net)


def stale_vjp_scheme(
    net: nn.Module, refresh_every: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Stale-weight caching: recompute the VJP only every ``refresh_every`` calls."""
    grad_norm = _value_grad_norm_fn(net)
    cache = StaleValueGrad(refresh_every)

    def scheme(x: torch.Tensor) -> torch.Tensor:
        return cache.get(lambda: grad_norm(x))

    return scheme


def finite_diff_scheme(net: nn.Module, eps: float = 1e-3) -> Callable[[torch.Tensor], torch.Tensor]:
    """Finite-difference approximation of the per-sample value gradient."""
    return lambda x: finite_difference_grad(lambda xb: net(xb), x, eps=eps).norm(dim=-1)


def estimate_mlp_flops(net: nn.Module, batch_size: int, backward: bool = False) -> float:
    """Estimated multiply-add FLOPs for one forward (optionally with a backward sweep)."""
    flops = 0
    for module in net.modules():
        if isinstance(module, nn.Linear):
            flops += 2 * module.in_features * module.out_features
    flops *= batch_size
    return flops * (2.0 if backward else 1.0)
