# Paper folders (WP1-WP4)

Each work package produces a distinct paper with its own publication target. Sources
live in per-WP folders so the documents, notes, and generated tables do not mix.

| WP | Folder | Target venue | Paper |
|---|---|---|---|
| WP1 | `wp1/` | ICML/ICLR | Objective-mismatch diagnostics and per-sample VJP profiling |
| WP2 | `wp2/` | NeurIPS/ICLR | Coupled-stability theory and the stabilized algorithm (COVAL) |
| WP3 | `wp3/` | ICML/TMLR | Scaling, latent dynamics, and goal-conditioned generalization |
| WP4 | `wp4/` | Dissertation / TMLR-JMLR-TPAMI | Synthesis, reproducibility package, dissertation |

Shared assets (bibliography, macros) live in `shared/`. Per-WP tables are generated from
committed `runs/` by `experiments/report.py`; the current WP1 experiment outputs stay in
`runs/exp*` (not grouped by WP).

## Build

```powershell
cd paper/wp1
latexmk -pdf main.tex
```
