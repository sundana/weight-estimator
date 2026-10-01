"""Hardware profiling harness for per-sample value-Jacobian computation (WP1 Exp 1.2).

Benchmarks the vector-Jacobian product ``||grad_s V(s')||`` under exact online VJP
(``torch.func``), stale-weight caching, finite differences, and the forward MLE
baseline, reporting wall-clock, throughput, peak memory, and TFLOPS. Torch is
imported lazily inside the modules so the base package imports without it.
"""
