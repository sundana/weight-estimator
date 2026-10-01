"""Per-sample value-gradient (VJP) machinery for the WP1 weights.

Computes ``||grad_s V(s)||`` per transition without a Python loop over the batch:
``torch.func.vmap(torch.func.grad(...))``. ``per_sample_value_grad_norm`` is the
convenience wrapper used by the weighted model losses; ``finite_difference_grad`` is
the reference scheme the profiling harness (WP1 Exp 1.2) compares against.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def per_sample_grad(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Jacobian of a scalar per-sample function ``fn`` at each row of ``x``.

    ``fn`` maps a single state vector ``(d,)`` to a scalar; the result has shape
    ``(batch, d)`` and is computed with a single vectorized VJP (no Python loop).
    """
    grad_fn = torch.func.vmap(torch.func.grad(fn))
    return grad_fn(x)


def per_sample_value_grad_norm(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """Per-sample L2 norm ``||grad_s fn(s)||`` of a scalar value function, shape ``(batch,)``."""
    return per_sample_grad(fn, x).norm(dim=-1)


def make_value_grad_norm_fn(fn):
    """Build the ``vmap(grad)`` transform once for a per-sample scalar ``fn``.

    Reusing the transform avoids re-tracing functorch on every call, which otherwise
    dominates the wall-clock for small networks (see WP1 Exp 1.2).
    """
    grad_fn = torch.func.vmap(torch.func.grad(fn))

    def value_grad_norm(x: torch.Tensor) -> torch.Tensor:
        return grad_fn(x).norm(dim=-1)

    return value_grad_norm


def state_value_grad_norm(net, obs: torch.Tensor) -> torch.Tensor:
    """``||grad_s V(s)||`` for a module mapping a batch ``obs`` to a scalar per row."""

    def single(s: torch.Tensor) -> torch.Tensor:
        return net(s.unsqueeze(0)).reshape(())

    return make_value_grad_norm_fn(single)(obs)


def finite_difference_grad(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    """Central finite-difference Jacobian of ``fn`` at ``x`` (shape ``(batch, d)``)."""
    batch, dim = x.shape
    grad = torch.zeros_like(x)
    for j in range(dim):
        plus = x.clone()
        minus = x.clone()
        plus[:, j] += eps
        minus[:, j] -= eps
        grad[:, j] = (fn(plus) - fn(minus)) / (2 * eps)
    return grad


class StaleValueGrad:
    """Cache of per-sample value-grad norms refreshed every ``refresh_every`` updates.

    Emulates the stale-weight caching scheme of WP1 Exp 1.2: weights are recomputed
    from the critic only every ``refresh_every`` calls, and reused in between.
    """

    def __init__(self, refresh_every: int = 1) -> None:
        if refresh_every < 1:
            raise ValueError("refresh_every must be >= 1")
        self.refresh_every = int(refresh_every)
        self._values: torch.Tensor | None = None
        self._counter = 0

    def get(self, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
        if self._values is None or self._counter % self.refresh_every == 0:
            self._values = compute()
        self._counter += 1
        return self._values
