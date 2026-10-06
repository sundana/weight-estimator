# WP1 — Objective-Mismatch Diagnostics & Per-Sample VJP Profiling

- **Venue target:** ICML 2026 (analysis + benchmark paper); formatted with the official
  `icml2026.sty`/`icml2026.bst` vendored in this directory (`fancyhdr.sty`,
  `algorithm.sty`, `algorithmic.sty` are its dependencies).
- **Main document:** `main.tex` (WP1 revision; supersedes the tabular Draft v2).
  Anonymous submission mode (`\usepackage{icml2026}`); switch to
  `\usepackage[accepted]{icml2026}` for camera-ready and fill in the real
  author affiliations/emails in the title block. Main body targets the 8-page limit;
  detailed diagnostic tables live in the appendix.
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
