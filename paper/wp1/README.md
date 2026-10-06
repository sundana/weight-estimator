# WP1 — Objective-Mismatch Diagnostics & Per-Sample VJP Profiling

- **Venue target:** ICML/ICLR (analysis + benchmark paper).
- **Main document:** `main.tex` (WP1 revision; supersedes the tabular Draft v2).
- **Notes:** `notes/` — `theory.md` (Lemma 1, Theorems 1-3), `wp1_report.md` (H1.1-H1.3
  results), `outline.md`, `phase0_implementation.md`. Historical drafts: `notes/archive/`.
- **Tables:** `tables/*.tex`, generated from committed `runs/` by
  `python -m experiments.report` (default `--out paper/wp1/tables`).
- **Figures:** `runs/exp1_deep_alignment/deep_alignment.png` (main grid),
  `deep_alignment_horizon.png` (horizon sweep), `deep_alignment_clip.png` (clip
  sensitivity); regenerate from committed JSON with
  `python -m experiments.exp1_deep_alignment --config configs/exp1_deep_alignment.yaml
  --replot`.
- **Experiments:** `runs/exp1*`, `runs/exp2*`, `runs/exp5_phase`,
  `runs/exp1_deep_alignment`, `runs/exp1_profiling`.
- **Build:** `latexmk -pdf main.tex` from this directory.
