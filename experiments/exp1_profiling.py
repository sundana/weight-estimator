"""Experiment 1.2: hardware profiling of per-sample VJP (WP1).

Sweeps batch size and VJP computation scheme -- exact online ``torch.func.vmap``,
stale-weight caching (refresh every K), central finite differences, and the forward-only
MLE baseline -- and records wall-clock (ms/step), throughput (transitions/s), peak CUDA
memory, and estimated TFLOPS. Tests H1.3: the exact VJP overhead over the forward pass
is <= 2.2x at batch size >= 512.

Hardware note: the plan targets an A100/4090; runs here use the local RTX 5060 Ti, so
absolute numbers differ and the overhead ratio is the reportable quantity.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from distractor_gym.deep.nets import StateValue
from distractor_gym.profiling.bench import (
    estimate_mlp_flops,
    finite_diff_scheme,
    forward_scheme,
    measure,
    stale_vjp_scheme,
    vjp_scheme,
)

from .common import load_config, save_fig, save_json, save_manifest


def _device(cfg: dict) -> str:
    want = cfg.get("device", "auto")
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return want


def _flops(scheme: str, net, batch: int, obs_dim: int, eps: float) -> float:
    if scheme == "mle_forward":
        return estimate_mlp_flops(net, batch, backward=False)
    if scheme.startswith("vjp") or scheme.startswith("stale"):
        return estimate_mlp_flops(net, batch, backward=True)
    if scheme == "finite_diff":
        return estimate_mlp_flops(net, batch, backward=False) * 2.0 * obs_dim
    return float("nan")


def run(cfg: dict, out_dir: str) -> dict:
    device = _device(cfg)
    torch.manual_seed(cfg.get("seed", 0))
    obs_dim = int(cfg.get("obs_dim", 64))
    net = StateValue(
        obs_dim, hidden=cfg.get("hidden", 256), n_layers=cfg.get("n_layers", 2)
    ).to(device)
    net.eval()
    iters = int(cfg.get("iters", 50))
    warmup = int(cfg.get("warmup", 10))
    fd_iters = int(cfg.get("fd_iters", 3))
    stale_ks = list(cfg.get("stale_k", [5, 10, 50]))
    eps = float(cfg.get("fd_eps", 1e-3))

    rows = []
    for batch in cfg.get("batch_sizes", [64, 128, 256, 512, 1024, 2048]):
        x = torch.randn(int(batch), obs_dim, device=device)
        schemes: list[tuple[str, object, int]] = [
            ("mle_forward", forward_scheme(net), iters),
            ("vjp_exact", vjp_scheme(net), iters),
            ("finite_diff", finite_diff_scheme(net, eps=eps), fd_iters),
        ]
        schemes += [(f"stale_K{k}", stale_vjp_scheme(net, k), iters) for k in stale_ks]
        for name, scheme, n_iter in schemes:
            fn = (lambda s=scheme, xb=x: s(xb))
            ms, thr, peak = measure(fn, int(batch), iters=n_iter, warmup=warmup, device=device)
            flops = _flops(name, net, int(batch), obs_dim, eps)
            tflops = flops / (ms * 1e-3) / 1e12 if ms > 0 else float("nan")
            rows.append(
                {
                    "scheme": name,
                    "batch": int(batch),
                    "ms_per_step": ms,
                    "throughput": thr,
                    "peak_mem_mb": peak,
                    "tflops": tflops,
                }
            )
        print(f"[exp1_profiling] batch={batch} done", flush=True)

    base = {r["batch"]: r["ms_per_step"] for r in rows if r["scheme"] == "mle_forward"}
    for r in rows:
        r["overhead_vs_forward"] = r["ms_per_step"] / base[r["batch"]] if base[r["batch"]] else float("nan")
    h13 = _h13(rows, cfg.get("threshold", 2.2), cfg.get("batch_threshold", 512))
    save_manifest(cfg, out_dir, name="manifest")
    save_json(rows, out_dir, "profiling")
    save_json(h13, out_dir, "h13")
    _plot(rows, out_dir, cfg.get("threshold", 2.2))
    return {"rows": rows, "h13": h13, "out_dir": out_dir}


def _h13(rows: list[dict], threshold: float, batch_threshold: int) -> dict:
    checked = [
        r for r in rows if r["scheme"] == "vjp_exact" and r["batch"] >= batch_threshold
    ]
    ratios = [r["overhead_vs_forward"] for r in checked]
    max_ratio = max(ratios) if ratios else float("nan")
    verdict = bool(ratios) and max_ratio <= threshold
    stale10 = [r["overhead_vs_forward"] for r in rows if r["scheme"] == "stale_K10"]
    print(
        f"[exp1_profiling] H1.3 exact-VJP overhead at batch>={batch_threshold}: "
        f"max={max_ratio:.3f} threshold={threshold} -> {'PASS' if verdict else 'FAIL'}"
        + (f"; stale_K10 max={max(stale10):.3f}" if stale10 else ""),
        flush=True,
    )
    return {
        "hypothesis": "H1.3",
        "pass": verdict,
        "threshold": threshold,
        "batch_threshold": batch_threshold,
        "max_overhead": max_ratio,
        "ratios": ratios,
        "stale_K10_max_overhead": max(stale10) if stale10 else float("nan"),
        "fallback": "stale-weight caching (K>=5) or TD-error surrogate per the WP1 plan",
    }


def _plot(rows: list[dict], out_dir: str, threshold: float) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    schemes = sorted({r["scheme"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for scheme in schemes:
        sub = sorted([r for r in rows if r["scheme"] == scheme], key=lambda r: r["batch"])
        x = [r["batch"] for r in sub]
        axes[0].plot(x, [r["ms_per_step"] for r in sub], marker="o", label=scheme)
        axes[1].plot(x, [r["overhead_vs_forward"] for r in sub], marker="o", label=scheme)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("batch size")
    axes[0].set_ylabel("ms / step")
    axes[0].set_title("VJP scheme wall-clock")
    axes[0].legend(fontsize=7)
    axes[1].set_xscale("log", base=2)
    axes[1].axhline(threshold, color="r", ls="--", lw=1.0, label=f"threshold {threshold}x")
    axes[1].set_xlabel("batch size")
    axes[1].set_ylabel("overhead vs forward MLE")
    axes[1].set_title("H1.3 overhead")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, out_dir, "profiling")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="runs/exp1_profiling")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
