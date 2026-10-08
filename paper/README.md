# Paper folders

Each work package produces a paper with its own publication target. Sources live in
per-WP folders so the documents, notes, and generated tables do not mix.

| WP | Folder | Target venue | Paper |
|---|---|---|---|
| WP1+WP2 | `coval/` | ICML/ICLR | **Bundled**: objective-mismatch diagnostics (Part I) + COVAL coupled-stability theory and stabilized algorithm (Part II) |
| WP3 | `wp3/` | ICML/TMLR | Scaling, latent dynamics, and goal-conditioned generalization |
| WP4 | `wp4/` | Dissertation / TMLR-JMLR-TPAMI | Synthesis, reproducibility package, dissertation |

WP1 and WP2 are bundled into `coval/` because the WP1 diagnostic negative result alone
would only motivate an unproven companion method; the bundle makes COVAL an internal
Part II. The bundle is **not submittable until Part II has results** (EXP 2.1-2.5); only
Part I is validated so far. The retired `paper/wp1/` and `paper/wp2/` folder contents were
moved into `coval/` (see `coval/notes/coval_scope.md`).

Shared assets (bibliography, macros) live in `shared/`. Tables are generated from
committed `runs/` by `experiments/report.py` (default `--out paper/coval/tables`); the
current experiment outputs stay in `runs/exp*` (not grouped by WP).

## Build

```powershell
cd paper/coval
latexmk -pdf main.tex
```
