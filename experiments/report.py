"""Generate LaTeX tables from committed run artifacts.

Reads ``runs/*/**.json`` and writes ``paper/tables/*.tex`` so the paper never drifts
from the committed results. Kept dependency-light (json + pathlib); the LaTeX is
ordinary ``table`` environments with ``booktabs``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rad(regime: dict) -> str:
    return "sparse" if regime.get("goal_radius") is not None else "dense"


def exp1_alignment_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Policy-gradient alignment $\cos(g_{\text{true}}, g_{\mathrm{model}})$"
        r" under the capacity-limited model with stochastic distractors (10 seeds).}",
        r"\label{tab:exp1}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Regime & $d_d$ & MLE & VAML-1 & VaGraM & Lambert \\",
        r"\midrule",
    ]
    for r in rows:
        g = r["regime"]
        lines.append(
            f"{_rad(g)}, {r['coverage']} & {g['d_d']} & "
            f"{r['mle']:.2f} & {r['vaml1']:.2f} & {r['vagram']:.2f} & {r['lambert']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def exp2_oracle_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Capacity-limited weight-estimator ablation at $d_d = 2$: estimated"
        r" ($\hat w$) vs oracle ($w$) weights on the policy-free value-aware risk"
        r" $\mathbb{E}\lvert V(s') - \mathbb{E}_{\hat P}[V]\rvert$ and on alignment.}",
        r"\label{tab:exp2b}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Regime & family & $R_{\text{MLE}}$ & $R_{\text{est}}$ & $R_{\text{ora}}$"
        r" & $a_{\text{MLE}}$ & $a_{\text{est}}$ & $a_{\text{ora}}$ \\",
        r"\midrule",
    ]
    for r in rows:
        g = r["regime"]
        if g["d_d"] != 2:
            continue
        for fam in ("vagram", "vaml1"):
            f = r[fam]
            lines.append(
                f"{_rad(g)}, {r['coverage']} & {fam} & {r['risk_mle']:.3f} & "
                f"{f['risk_est']:.3f} & {f['risk_ora']:.3f} & {r['align_mle']:.2f} & "
                f"{f['align_est']:.2f} & {f['align_ora']:.2f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def crossover_scope_table(capacity_rows: list[dict], empirical_rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Theorem~\ref{thm:pred} dominance ($R_b(\hat w) \ge R_b(\text{MLE})$)"
        r" by model class. The result holds for the uncapacitated estimator (Exp~1b) but"
        r" not under a finite-capacity fit.}",
        r"\label{tab:scope}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & family & regimes & $\text{est} \ge \text{MLE}$ \\",
        r"\midrule",
    ]
    for name, rows in (("uncapacitated", empirical_rows), ("capacity-limited", capacity_rows)):
        for fam in ("vagram", "vaml1"):
            hits = sum(r[f"r_b_{fam}_est"] >= r["r_b_mle"] - 1e-9 for r in rows)
            lines.append(f"{name} & {fam} & {len(rows)} & {100 * hits / len(rows):.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="paper/tables")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp1_alignment.tex").write_text(
        exp1_alignment_table(_read(f"{args.runs}/exp1_capacity/results.json")), encoding="utf-8"
    )
    (out / "exp2_oracle.tex").write_text(
        exp2_oracle_table(_read(f"{args.runs}/exp2_capacity/part_b.json")), encoding="utf-8"
    )
    (out / "exp1b_scope.tex").write_text(
        crossover_scope_table(
            _read(f"{args.runs}/exp1b_capacity/crossover.json"),
            _read(f"{args.runs}/exp1b/crossover.json"),
        ),
        encoding="utf-8",
    )
    print(f"wrote 3 tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
