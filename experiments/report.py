"""Generate LaTeX tables from committed run artifacts.

Reads ``runs/*/**.json`` and writes ``paper/coval/tables/*.tex`` so the paper never drifts
from the committed results. Kept dependency-light (json + pathlib); the LaTeX is
ordinary ``table`` environments with ``booktabs``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean = sum(xs) / n
    std = (sum((x - mean) ** 2 for x in xs) / n) ** 0.5
    return mean, std


_H11_FAMILIES = ["mle", "vaml1", "vagram", "td_error", "calibrated", "lambert"]


def _families(rows: list[dict]) -> list[str]:
    present = {r["family"] for r in rows}
    return [f for f in _H11_FAMILIES if f in present]


def _default_arm(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("arm", "default") == "default"]


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _rad(regime: dict) -> str:
    return "sparse" if regime.get("goal_radius") is not None else "dense"


def exp1_alignment_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
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
    lines += [        r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
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


def h12_lemma_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Lemma~1 / H1.2 on the uncapacitated tabular suite: the fit of the"
        r" first-order factorization and the Spearman rank correlation between"
        r" $|\delta_{\text{TD}}|$ and $\|\nabla V\|\,\|\hat s'-s'\|$ ($\rho_{\text{prod}}$).}",
        r"\label{tab:h12}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Regime & $d_d$ & coverage & $R^2$ & $\rho_{\text{prod}}$ & $\overline{|\cos\phi|}$"
        r" & curv. \\",
        r"\midrule",
    ]
    for r in rows:
        g = r["regime"]
        lines.append(
            f"{_rad(g)} & {g['d_d']} & {r['coverage']} & {r['r2']:.3f} & "
            f"{r['spearman_product']:.3f} & {r['mean_cos_phi']:.3f} & "
            f"{r['curvature_residual']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def h11_alignment_table(rows: list[dict]) -> str:
    rows = _default_arm(rows)
    families = _families(rows)
    datasets = sorted({r["dataset"] for r in rows})
    d_ds = sorted({r["d_d"] for r in rows})
    header = " & ".join(fam.replace("_", "-") for fam in families)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Exp~1.1 deep policy-gradient alignment $\cos(g_{\text{true}},"
        r" g_{\text{model}})$ on the MuJoCo Distractor-Gym (differentiated via"
        r" \texttt{mjd\_transitionFD}; mean $\pm$ std over seeds and $\sigma_{\text{dist}}$);"
        r" random and medium-replay SAC datasets.}"
        r"\label{tab:h11}",
        r"\begin{tabular}{ll" + "c" * len(families) + "}",
        r"\toprule",
        f"Dataset & $d_d$ & {header} \\\\",
        r"\midrule",
    ]
    for ds in datasets:
        for d in d_ds:
            vals = []
            for fam in families:
                xs = [r["cos"] for r in rows if r["dataset"] == ds and r["d_d"] == d and r["family"] == fam]
                if xs:
                    mean, std = _mean_std(xs)
                    vals.append(f"${mean:+.3f}\\pm{std:.3f}$")
                else:
                    vals.append("--")
            lines.append(f"{ds} & {d} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def _paired_deltas(rows: list[dict], ds: str, d: int, fam: str) -> list[float]:
    seeds = sorted({r["seed"] for r in rows if r["dataset"] == ds and r["d_d"] == d})
    diffs = []
    for seed in seeds:
        f = [
            r["cos"]
            for r in rows
            if r["dataset"] == ds and r["d_d"] == d and r["family"] == fam and r["seed"] == seed
        ]
        m = [
            r["cos"]
            for r in rows
            if r["dataset"] == ds and r["d_d"] == d and r["family"] == "mle" and r["seed"] == seed
        ]
        if f and m:
            diffs.append(sum(f) / len(f) - sum(m) / len(m))
    return diffs


def h11_paired_table(rows: list[dict]) -> str:
    rows = _default_arm(rows)
    families = [f for f in _families(rows) if f != "mle"]
    datasets = sorted({r["dataset"] for r in rows})
    d_ds = sorted({r["d_d"] for r in rows})
    header = " & ".join(fam.replace("_", "-") for fam in families)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Exp~1.1 paired alignment gain $\Delta = \cos(\text{family}) -"
        r" \cos(\text{MLE})$ within a seed (removes actor/data noise), mean over 10 seeds"
        r" with paired $t$-statistic.}",
        r"\label{tab:h11paired}",
        r"\begin{tabular}{ll" + "c" * len(families) + "}",
        r"\toprule",
        f"Dataset & $d_d$ & {header} \\\\",
        r"\midrule",
    ]
    for ds in datasets:
        for d in d_ds:
            cells = []
            for fam in families:
                diffs = _paired_deltas(rows, ds, d, fam)
                if diffs:
                    mean, std = _mean_std(diffs)
                    if std > 1e-6:
                        t = mean / (std / len(diffs) ** 0.5)
                        cells.append(f"${mean:+.2f}$ ($t={t:+.1f}$)")
                    else:
                        cells.append(f"${mean:+.2f}$")
                else:
                    cells.append("--")
            lines.append(f"{ds} & {d} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def h11_horizon_table(rows: list[dict]) -> str:
    rows = [r for r in rows if abs(r["sigma_dist"]) < 1e-9]
    rows = [r for r in rows if r.get("arm", "horizon") != "default"]
    families = [f for f in _families(rows) if f in ("mle", "vagram", "td_error", "calibrated")]
    datasets = sorted({r["dataset"] for r in rows})
    d_ds = sorted({r["d_d"] for r in rows})
    horizons = sorted({r["horizon"] for r in rows})
    header = " & ".join(str(h) for h in horizons)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Exp~1.1 alignment vs rollout horizon ($\sigma_{\text{dist}}=0$):"
        r" $\cos(g_{\text{true}}, g_{\text{model}})$ mean over seeds. A model that is"
        r" aligned at the shortest horizon but misaligned at horizon~8 indicates"
        r" compounding rollout error rather than one-step underfit.}",
        r"\label{tab:h11horizon}",
        r"\begin{tabular}{lll" + "c" * len(horizons) + "}",
        r"\toprule",
        f"Dataset & $d_d$ & family & {header} \\\\",
        r"\midrule",
    ]
    for ds in datasets:
        for d in d_ds:
            for fam in families:
                cells = []
                for h in horizons:
                    xs = [
                        r["cos"]
                        for r in rows
                        if r["dataset"] == ds
                        and r["d_d"] == d
                        and r["family"] == fam
                        and r["horizon"] == h
                    ]
                    cells.append(f"${sum(xs) / len(xs):+.2f}$" if xs else "--")
                lines.append(
                    f"{ds} & {d} & {fam.replace('_', '-')} & " + " & ".join(cells) + r" \\"
                )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def h11_quality_table(rows: list[dict]) -> str:
    rows = [r for r in _default_arm(rows) if r["family"] == "mle"]
    datasets = sorted({r["dataset"] for r in rows})
    d_ds = sorted({r["d_d"] for r in rows})
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Exp~1.1 diagnostic calibration (MLE model): scale-free one-step error"
        r" $\overline{((\hat s'-s')/\sigma_y)^2}$, median gradient-norm ratio"
        r" $\|\!g_{\text{model}}\!\|/\|\!g_{\text{true}}\!\|$, fraction of rows with a"
        r" near-zero model gradient ($<0.1$), and mean cosine at horizon~8.}",
        r"\label{tab:h11quality}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & $d_d$ & one-step err & $\text{med}\,\|\!g\|\text{-ratio}$"
        r" & frac $\|\!g_{\text{model}}\!\|<0.1$ & mean $\cos$ \\",
        r"\midrule",
    ]
    for ds in datasets:
        for d in d_ds:
            sub = [r for r in rows if r["dataset"] == ds and r["d_d"] == d]
            err = sum(r["one_step_mse"] for r in sub) / len(sub)
            ratio = _median([r["grad_ratio"] for r in sub])
            frac = sum(1 for r in sub if r["g_model_norm"] < 0.1) / len(sub)
            cos = sum(r["cos"] for r in sub) / len(sub)
            lines.append(
                f"{ds} & {d} & {err:.2f} & {ratio:.2f} & {100 * frac:.0f}\\% & {cos:+.2f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def h11_clip_table(rows: list[dict]) -> str:
    rows = [r for r in rows if r["horizon"] == max(x["horizon"] for x in rows)]
    families = _families(rows)
    clips = sorted({r["clip_sigma"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    header = " & ".join(f"{c:g}" for c in clips)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Exp~1.1 model-rollout clip sensitivity at horizon 8, $d_d = 50$:"
        r" mean $\cos(g_{\text{true}}, g_{\text{model}})$ vs the prediction clip"
        r" $\pm\,k\,\sigma$ of the training band.}",
        r"\label{tab:h11clip}",
        r"\begin{tabular}{ll" + "c" * len(clips) + "}",
        r"\toprule",
        f"Dataset & family & {header} \\\\",
        r"\midrule",
    ]
    for ds in datasets:
        for fam in families:
            cells = []
            for c in clips:
                xs = [
                    r["cos"]
                    for r in rows
                    if r["dataset"] == ds and r["family"] == fam and r["clip_sigma"] == c
                ]
                cells.append(f"${sum(xs) / len(xs):+.2f}$" if xs else "--")
            lines.append(f"{ds} & {fam.replace('_', '-')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def h11_control_table(rows: list[dict]) -> str:
    families = _families(rows)
    datasets = sorted({r["dataset"] for r in rows})
    sigmas = sorted({r["sigma_dist"] for r in rows})
    header = " & ".join(fam.replace("_", "-") for fam in families)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Exp~1.1 identical-loss control at $d_d = 0$: every family is fit with"
        r" the MLE objective ($w = 1$) while retaining its label. Identical cells confirm"
        r" the fitting/evaluation pipeline is deterministic, so family spread at"
        r" $d_d > 0$ is attributable to the weights.}",
        r"\label{tab:h11control}",
        r"\begin{tabular}{lll" + "c" * len(families) + "}",
        r"\toprule",
        f"Dataset & $\\sigma_{{\\text{{dist}}}}$ & $d_d$ & {header} \\\\",
        r"\midrule",
    ]
    for ds in datasets:
        for sig in sigmas:
            d = sorted({r["d_d"] for r in rows if r["dataset"] == ds and r["sigma_dist"] == sig})[0]
            vals = []
            for fam in families:
                xs = [
                    r["cos"]
                    for r in rows
                    if r["dataset"] == ds and r["sigma_dist"] == sig and r["family"] == fam
                ]
                vals.append(f"${sum(xs) / len(xs):+.4f}$" if xs else "--")
            lines.append(f"{ds} & {sig} & {d} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def h13_overhead_table(rows: list[dict], h13: dict) -> str:
    batches = sorted({r["batch"] for r in rows})
    schemes = ["mle_forward", "vjp_exact", "stale_K5", "stale_K10", "stale_K50", "finite_diff"]
    verdict = "PASS" if h13.get("pass") else "FAIL"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Exp~1.2 VJP overhead relative to the forward MLE baseline on the local"
        r" RTX 5060 Ti. H1.3 (exact VJP $\le 2.2\times$ at batch $\ge 512$): "
        + f"{verdict} (max ${h13['max_overhead']:.1f}\\times$); stale caching ($K=10$) stays"
        r" below the threshold.}",
        r"\label{tab:h13}",
        r"\begin{tabular}{l" + "r" * len(batches) + "}",
        r"\toprule",
        "Scheme & " + " & ".join(str(b) for b in batches) + r" \\",
        r"\midrule",
    ]
    for scheme in schemes:
        cells = []
        for b in batches:
            match = next((r for r in rows if r["scheme"] == scheme and r["batch"] == b), None)
            cells.append(f"{match['overhead_vs_forward']:.2f}" if match else "--")
        lines.append(f"{scheme.replace('_', '-')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _g(value) -> str:
    return "--" if value is None else f"{value:g}"


def _e(value) -> str:
    return "--" if value is None else f"{value:.2e}"


def coupled_lab_table(lab: dict) -> str:
    kappa_a = lab.get("kappa_a", {})
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Phase~A tabular coupled lab: linear-stability boundary $\kappa$"
        r" ($\alpha_{model}$) of the coupled model/critic system per weight family."
        r" Self-normalized value-aware weights lower the boundary relative to MLE.}",
        r"\label{tab:h20}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"weight family & $\kappa$ ($\alpha_{model}$) \\",
        r"\midrule",
    ]
    for fam in ("mle", "vaml1", "vagram"):
        lines.append(f"{fam} & {_g(kappa_a.get(fam))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def coval_stability_table(stability: dict) -> str:
    e21 = stability["exp21"]
    e22 = stability["exp22"]["rows"]
    e23 = stability["exp23"]["rows"]
    e24 = stability["exp24"]["rows"]
    e25 = stability["exp25"]["rows"]
    best22 = min(e22, key=lambda r: r["tracking_error_final"])
    norm = {r["normalization"]: r["weight_var_mean"] for r in e23}
    clip_none = next(r for r in e24 if r["clip_median_ratio"] is None)
    clip_tight = min(
        (r for r in e24 if r["clip_median_ratio"] is not None),
        key=lambda r: r["clip_median_ratio"],
    )
    lip_off = next(r["critic_lip_final"] for r in e25 if not r["spectral_norm"])
    lip_on = next(r["critic_lip_final"] for r in e25 if r["spectral_norm"])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{EXP~2.1--2.5 stabilization ablations on the tabular coupled lab"
        r" (one factor at a time around the baseline operating point; $5$ seeds).}",
        r"\label{tab:h21}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Factor & Setting & Result \\",
        r"\midrule",
        f"LR ratio & $\\kappa$ & {_g(e21['kappa_ratio'])} \\\\",
        f"Polyak $\\tau$ & best & {_g(best22['polyak_tau'])}"
        f" (te {best22['tracking_error_final']:.3f}) \\\\",
        r"Normalization & $\mathrm{Var}(\bar w)$ & none "
        f"{_e(norm.get('none'))}, min-max {_e(norm.get('min_max'))}, "
        f"self {_e(norm.get('batch_self'))} \\\\",
        r"Clipping & $\|\nabla L\|$ (none $\to$ tightest) & "
        f"{clip_none['g_theta_norm_final']:.2e} $\\to$ "
        f"{clip_tight['g_theta_norm_final']:.2e} \\\\",
        r"Spectral & $\|V_\phi\|_{Lip}$ (off $\to$ on) & "
        f"{lip_off:.2f} $\\to$ {lip_on:.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="paper/coval/tables")
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
    added = 0
    for name, source, builder in [
        ("h12_lemma.tex", f"{args.runs}/exp2/part_a.json", h12_lemma_table),
        ("h11_alignment.tex", f"{args.runs}/exp1_deep_alignment/alignment.json", h11_alignment_table),
        ("h11_paired.tex", f"{args.runs}/exp1_deep_alignment/alignment.json", h11_paired_table),
        ("h11_quality.tex", f"{args.runs}/exp1_deep_alignment/alignment.json", h11_quality_table),
        ("h11_control.tex", f"{args.runs}/exp1_deep_alignment/alignment_control.json", h11_control_table),
        (
            "h11_horizon.tex",
            f"{args.runs}/exp1_deep_alignment/alignment_horizon.json",
            h11_horizon_table,
        ),
        ("h11_clip.tex", f"{args.runs}/exp1_deep_alignment/alignment_clip.json", h11_clip_table),
    ]:
        try:
            (out / name).write_text(builder(_read(source)), encoding="utf-8")
            added += 1
        except FileNotFoundError:
            pass
    try:
        h13 = _read(f"{args.runs}/exp1_profiling/h13.json")
        (out / "h13_overhead.tex").write_text(
            h13_overhead_table(_read(f"{args.runs}/exp1_profiling/profiling.json"), h13),
            encoding="utf-8",
        )
        added += 1
    except FileNotFoundError:
        pass
    for name, source, builder in [
        ("h20_coupled.tex", f"{args.runs}/exp2_coupled_lab/coupled_lab.json", coupled_lab_table),
        ("h21_stability.tex", f"{args.runs}/exp2_coval_stability/stability.json", coval_stability_table),
    ]:
        try:
            (out / name).write_text(builder(_read(source)), encoding="utf-8")
            added += 1
        except FileNotFoundError:
            pass
    print(f"wrote {3 + added} tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
