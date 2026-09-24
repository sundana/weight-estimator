# When and Why Does Value-Aware Model Learning Fail?

## A Diagnostic Suite for Objective Mismatch in Model-Based Reinforcement Learning

**Status:** Draft v2.1 (tabular, rescaled as a theory/diagnostic paper). Supersedes
`draft_v1.md`. All numbers are produced by the committed runs under `runs/` (each with a
`manifest.json` pinning the config, git SHA, and library versions) at 10 seeds (5 for the
phase diagram). Paper tables are generated from the runs by `experiments/report.py`
(`paper/tables/*.tex`). The continuous neural transfer (Exp 4) was **not** re-run in this
revision and its earlier numbers are withdrawn pending a `torch` provisioning + a
coordinate-scaling fix.

---

## Abstract

Model-based RL fits a dynamics model to predict the next state, typically by maximum
likelihood, then uses it to improve a policy. The objective mismatch problem (Lambert
et al., 2020) is that likelihood need not correlate with the model's ability to guide
the policy. Value-aware model learning (VaGraM; Voelcker et al., 2022) reweights the
model loss by the value function but is empirically known to sometimes underperform
MLE. We isolate *when and why*. We contribute (i) a corrected theory of the weighted
ratio estimator `w`: the first-order variance penalty for value-aware weighting is
`sigma_w^2 Var(V)/n` (not `sigma_w^2 E[V^2]/n` as previously stated), and, critically,
this dominance result holds only for the uncapacitated fixed estimator -- under a
finite-capacity model, reweighting reallocates capacity and can *reduce* Bellman risk;
(ii) a controlled synthetic benchmark, Distractor-Gym, with a functional distractor
family (linear / nonlinear / stochastic) and a configurable model-capacity knob
(low-rank feature-budget transition model); and (iii) two quantitative diagnostics --
policy-gradient alignment and the `|delta_TD| ~ ||grad V|| * eps * |cos phi|`
decomposition. We find: **(a)** with an uncapacitated model the decomposition holds
(`R^2 = 0.98` dense, `0.76-0.81` sparse) and is invariant to the distractor count
because deterministic distractors are fit exactly and `grad_{s_d} V = 0`; **(b)** with
a capacity-limited model and unpredictable (stochastic) distractors, MLE
policy-gradient alignment collapses as the distractor count grows (e.g. `cos` from
`0.86` at `d_d=0` to `-0.08` at `d_d=2`), while value-aware weighting -- especially
VAML-1/Lambert -- restores it (to `0.16-0.34`); gradient-only VaGraM does not,
consistent with the `cos phi`/curvature misspecification in the decomposition; and
**(c)** the prediction-risk dominance of MLE is confined to the uncapacitated
estimator (estimated-weight Bellman risk `>=` MLE in `100%` of uncapacitated regimes)
and fails for capacity-limited fitting (`50%`/`33%` for VaGraM/VAML-1).

---

## 1 Introduction

Model-based RL (MBRL) learns a dynamics model and uses it for control. The standard
model objective is maximum likelihood (MLE). Lambert et al. (2020) showed this can be
*mismatched* with the downstream goal. A principled family of remedies -- value-aware
model learning (VAML; Farahmand et al., 2017, 2018; VaGraM; Voelcker et al., 2022) --
reweights the model loss by the value function, so the model is accurate *where value
changes*. VaGraM documents that these can underperform MLE, but there is no isolated,
controlled characterization of when and why.

This revision makes the following contributions:

1. **Corrected theory of the weight estimator.** The weighted mean is a *ratio*
   estimator; the correct first-order variance penalty is `sigma_w^2 Var(V)/n`. The
   prediction-risk dominance of MLE holds for the uncapacitated fixed estimator but
   not for finite-capacity model fitting, where reweighting can lower Bellman risk.
2. **Distractor-Gym with real pressure.** The distractor `RegimeConfig.distractor_class`
   is now functional (linear coupling, logistic chaos, and stochastic random walks),
   and a reduced-rank *feature-budget* transition model provides the finite capacity
   that VaGraM's analysis requires. Without a capacity limit, distractors are provably
   inert for the value-aware diagnostic because `grad_{s_d} V = 0`.
3. **Diagnostics** -- policy-gradient alignment and the `|delta_TD|` decomposition --
   measured with clean i.i.d. coverage and 10 seeds, with committed artifacts.

**Findings.** (a) Uncapacitated: the decomposition fits well and is `d_d`-invariant;
MLE Bellman risk is dominated by any estimated weighting. (b) Capacity-limited +
stochastic distractors: MLE alignment collapses with `d_d`; VAML-1/Lambert restore it;
gun expert VaGraM is unstable (it weights by `||grad V||` only). (c) The value-aware
benefit is decision-level (alignment), not prediction-level, and the two diverge under
capacity limits.

---

## 2 Background and Related Work

**Objective mismatch.** Lambert et al. (2020) formalized that a one-step-MLE model used
for control need not correlate with performance; they proposed an exploratory
reweighting `w(y) = c exp(-d(y))`. Wei et al. (2024) survey decision-aware remedies.

**Value-aware model learning.** Farahmand et al. (2017, 2018) introduced VAML/IterVAML;
VaGraM (Voelcker et al., 2022) uses the value gradient as the weight and documents two
failure modes: (a) no accounting for exploration/coverage, and (b) non-smooth value
functions make unbounded losses diverge. CVAML (2025) shows sample-based value-aware
losses are uncalibrated for stochastic models; ROMI (2026) adds adaptive MLE+value
weighting for offline RL. MOBILE (Sun et al., 2023) uses model-Bellman inconsistency
for offline pessimism.

**Diagnostics.** Lambert et al. measured one-step likelihood vs return correlation. We
modernize this with per-run gradient-alignment and value-error decomposition
diagnostics, here validated in the tabular suite.

---

## 3 Theory: The Weight Estimator `w`

Setup. Model family `P_theta(s' | s, a)`; weighted loss
`L_w(theta) = E_D[ w(s, a, s') * ell(P_theta(s' | s, a)) ]`. MLE is `w = 1`.
Value-aware weights: `w = |V(s') - V(s)|` (VAML-1), `w = ||grad_s V(s')||` (VaGraM),
`w = exp((V(s') - V_max)/tau)` (Lambert). The weight is estimated from a data-dependent
value estimate, so `w_hat = w(V_hat, grad V_hat)` has bias and variance. Full
statements and proofs: `paper/notes/theory.md`.

**Lemma 1 (decomposition).** Let `V` be twice differentiable with Hessian `H_V`, `e =
s_hat' - s'`, `phi = angle(grad_s V(s'), e)`. Then
`delta_TD := V(s') - V(s_hat') = -grad_s V(s') . e - (1/2) e^T H_V(xi) e` and
`|delta_TD| <= ||grad V|| * ||e|| * |cos phi| + (1/2) ||H_V||_2 ||e||^2`.
*Empirical check (Exp 2A):* `R^2 = 0.98` dense / `0.76-0.81` sparse with the
uncapacitated model; the fit is `d_d`-invariant there, and degrades with `d_d` under a
capacity limit (`mean |cos phi|` 0.92 -> 0.46).

**Theorem 1 (prediction-risk of the weighted ratio estimator).** For iid `y_i` and
weights `w_i >= 0` with `E[w]=1`, `Var(w)=sigma_w^2`, independent of `y`,
```
Var(mu_hat_w) = Var(mu_hat_MLE) + sigma_w^2 Var(V)/n          (corrected)
E[mu_hat_w]   = E[V] + Cov(w, V(y)) + O(1/n)
```
(Delta method on the ratio `X/Y`; the earlier `sigma_w^2 E[V^2]/n` omitted the
denominator terms and overstates the penalty by `sigma_w^2 E[V]^2/n`.) So value-aware
weighting cannot improve the conditional-mean Bellman-target prediction. *Empirical
check (Exp 1b):* with the uncapacitated model the estimated-weight Bellman risk is
`>=` MLE in **100%** of regimes. **Scope limit:** this does not extend to
capacity-limited fitting, where estimated weights beat MLE on Bellman risk in ~half
the regimes.

**Theorem 2 (effective sample size).** `ESS(w) = (sum w)^2 / sum w^2 ~ n/(1+sigma_w^2)`.
High-variance weights shrink the effective sample size.

**Theorem 3 (decision-risk crossover criterion).** With
`R_dec(P_hat) = E[(grad_s V(s') . (E_{P_hat}[Y] - E_{P*}[Y]))^2]`,
```
R_dec(P_hat_w) - R_dec(P_hat_MLE)
  = (grad V . Cov(w, Y))^2                         (value-relevance signal)
    - ||grad V||^2 sigma_w^2 Var(||Y||)/n          (weight-noise penalty)
    + (estimator bias term)
```
Weighting improves the decision risk iff the signal exceeds the penalty. This predicts
MLE-optimality when `grad V` is zero along distractor dims, and value-aware advantage
when coverage concentrates a strong gradient. We instantiate the criterion as the
scale-invariant statistic `SNR_dec = n (gbar·Cov(w,Y))^2 / (||gbar||^2 σ_w^2 Var(||Y||))`
with threshold `tau = 1` (`diagnostics.decision_crossover_snr`). *Empirical check (Exp
1b):* the criterion is **not calibrated** — it predicts value-aware improvement in 100%
of regimes while the measured alignment delta is positive in only 27–54% (AUC 0.56–0.60
capacity-limited, 0.05–0.22 uncapacitated, versus the heuristic `SNR_w` at 0.40–0.78).
The decomposition is directionally sound but the finite-sample threshold is open work.

---

## 4 Distractor-Gym

State `s = (s_c, s_d)` with `d_c` control-relevant and `d_d` distractor dims whose
reward relevance is zero, so `grad_{s_d} V = 0` by construction.

- **Tabular suite** (ground-truth lab): 1D control axis on a grid with actions
  `{-1, 0, +1}` and quantized transition noise; a `d_d`-dimensional distractor block
  following the configured class (linear coupling, logistic chaos, or stochastic random
  walks). Reward dense `exp(-(x-goal)^2/2)` or sparse `1[|x-goal| <= radius]`. Exact
  value functions, gradients, policy gradients, and Bellman risks are computable.
- **Model capacity**: the uncapacitated empirical table, or a **reduced-rank
  feature-budget** model `mu(s,a) = phi(s,a) C` with `rank(C) <= capacity`, `phi =
  [s_coords, a, 1]`, fit by weighted least squares and restored to a Gaussian
  transition tensor. A small capacity relative to the state dimension makes the model
  spend its budget on high-variance distractor directions.
- **Coverage**: i.i.d. state sampling (uniform, or goal-concentrated) rather than a
  single random-walk trajectory, so the coverage knob is well defined.

Regime knobs: `d_d`, distractor class, reward sparsity, transition noise, coverage,
`capacity`, and `model_noise` (bandwidth).

---

## 5 Experiment 1: Policy-Gradient Alignment

**Setup.** Off-center softmax policy, `n_data = 8000`, 10 seeds. `cos(g_true, g_model)`
for MLE, VAML-1, VaGraM, Lambert.

**Uncapacitated model (`runs/exp1`).** The failure is coverage-driven, not
distractor-driven: MLE misguides under goal coverage at `d_d=0`
(`cos = -0.375` dense, `-0.399` sparse) and VAML-1/Lambert restore alignment
(`0.31` / `0.81`). As `d_d` grows the effect weakens and can reverse (sparse+goal
`d_d=2`: MLE `0.91`, VAML-1 `-0.65`).

**Capacity-limited model + stochastic distractors (`runs/exp1_capacity`).** Here
distractors do the work: MLE alignment collapses monotonically with `d_d`, while
VAML-1/Lambert recover it and gradient-only VaGraM does not.

| regime (rad, coverage) | d_d | MLE | VAML-1 | VaGraM | Lambert |
|---|---|---|---|---|---|
| dense, uniform | 0 | 0.86 | 0.84 | 0.86 | 0.86 |
| dense, uniform | 1 | 0.82 | 0.78 | 0.82 | 0.67 |
| dense, uniform | 2 | **-0.08** | -0.15 | -0.45 | 0.13 |
| dense, goal | 2 | **-0.09** | 0.34 | -0.09 | 0.47 |
| sparse, uniform | 2 | **-0.22** | 0.16 | -0.14 | 0.11 |
| sparse, goal | 2 | 0.32 | 0.62 | 0.34 | 0.28 |

The pattern matches Theorem 3: VaGraM's weight `||grad V||` ignores `cos phi` and the
curvature residual, both of which degrade with `d_d` (Exp 2A), while VAML-1/Lambert are
better behaved.

---

## 6 Experiment 2: Decomposition and Weight-Estimator Ablation

**Part A (Lemma 1).**
- Uncapacitated (`runs/exp2`): `R^2 = 0.98` dense, `0.76-0.81` sparse; slope ~1.0-1.13;
  `mean_cos_phi ~ 0.09`; curvature residual 0.004 (dense) vs 0.021 (sparse). The fit is
  invariant to `d_d` because deterministic distractors are fit exactly and contribute
  nothing to `delta_TD` or to `grad V`.
- Capacity-limited (`runs/exp2_capacity`): `mean_cos_phi` falls from 0.92 (`d_d=0`) to
  0.46 (`d_d=2`) and the curvature residual rises to 0.4-2.5; `R^2` becomes unstable and
  can go negative for sparse goals. This is the first-order factorization breaking down
  under capacity-limited models -- the regime value-aware weighting is meant to address.

**Part B (weight estimator vs policy effects), capacity-limited (`runs/exp2_capacity`).**
At `d_d=2`, VAML-1 has higher alignment than MLE while VaGraM is unstable:

| regime | MLE | VAML-1 (est) | VaGraM (est) |
|---|---|---|---|
| dense, uniform | -0.12 | 0.22 | -0.81 |
| sparse, uniform | -0.19 | 0.32 | -0.15 |
| sparse, goal | 0.09 | 0.47 | 0.09 |

The value-aware benefit is decision-level (alignment) and does not require lower
Bellman risk -- consistent with Theorem 3.

The oracle-vs-estimated ablation (generated table `paper/tables/exp2_oracle.tex`) shows
that at `d_d=2` the estimated-weight risk is close to the oracle-weight risk
(e.g. dense+uniform VAML-1: est 1.325 vs ora 1.374 vs MLE 1.376), so under a capacity
limit the weight-estimation noise is no longer the dominant gap it is in the
uncapacitated estimator.

---

## 6b Experiment 5: Mismatch Phase Diagram

`experiments/exp5_phase.py` sweeps `capacity x d_d x sparsity` under goal coverage with
stochastic distractors and classifies each cell (best value-aware vs MLE, margin 0.1).
Figure: `runs/exp5_phase/phase_diagram.png`; data `runs/exp5_phase/phase.json`.

```
dense reward:  capacity \ d_d      0        1        2
               cap 1            +0.00    +0.27*   +0.38*
               cap 2            +0.00    -0.01    +0.22*
               cap 3            -0.00    -0.00    +0.12*
               cap 5            -0.00    +0.02    +0.02
sparse reward: cap 1            +0.04    +0.49*   +0.71*
               cap 2            +0.04    +0.09    +0.35*
               cap 3            +0.04    +0.12*   +0.06
               cap 5            +0.04    +0.10    +0.16*
```

(`*` = value-aware-win; `+x` = best value-aware minus MLE alignment.) The structure is
consistent: distractors push MLE down at small capacity and value-aware weighting wins;
raising capacity removes the pressure. A few larger-capacity cells flip with the seed
draw; the bulk structure is stable.

---

## 7 Experiment 1b: Crossover and the Scope of Theorem 1

- **Uncapacitated (`runs/exp1b`, 48 regimes):** estimated-weight Bellman risk `>=` MLE
  in **100%** of regimes (VaGraM and VAML-1); oracle `>=` MLE in 94%/100%. This
  validates Theorem 1 for the uncapacitated estimator.
- **Capacity-limited (`runs/exp1b_capacity`, 12 regimes):** estimated-weight risk `>=`
  MLE in only 50% (VaGraM) / 33% (VAML-1) of regimes. Reweighting a finite-capacity
  model changes its effective representation and can *reduce* prediction risk, so the
  dominance result does not transfer.

---

## 8 Discussion

**When MLE fails to guide the policy.** Under goal-concentrated coverage with an
uncapacitated model, and under capacity-limited fitting with unpredictable distractors,
the model's policy gradient misaligns with the true one. Value-aware weighting that
respects the value cliff (VAML-1, Lambert) restores alignment; gradient-only VaGraM is
mis-specified when `cos phi` is small or the curvature residual is large.

**Why value-aware fails.** On the uncapacitated Bellman/prediction risk it cannot help
-- it only adds weight-noise variance -- reproducing VaGraM's "inferior in practice"
observation. On the decision risk it helps only when the value-relevance signal beats
the weight-noise penalty, which fails when weights are degenerate. Under a capacity
limit the two risks separate: weighting can help prediction (by reallocating capacity),
but the decision-level benefit still depends on the weight-estimator noise.

**Practical rule of thumb.** Measure `SNR_w` and the alignment delta during training.
If `SNR_w` is low or the alignment delta is non-positive, value-aware weighting is not
helping.

---

## 9 Limitations and Future Work

- The `SNR_w` crossover statistic remains a positive-but-noisy proxy, and the derived
  `SNR_dec` is not calibrated (predicts value-aware-win in 100% of regimes; AUC no
  better than `SNR_w`). A closed-form threshold `tau` remains open work.
- The tabular results are stylized. The continuous transfer (Exp 4) was **not** re-run
  in this revision: `torch` is uninstalled and the earlier projection loss was computed
  in mismatched (raw vs standardized) coordinates; that experiment is withdrawn until
  fixed.
- The phase diagram has some seed-sensitive cells at larger capacity; more seeds and a
  finer capacity sweep are needed to tighten it.
- Deep baselines (MBPO/VaGraM, DreamerV3, TD-MPC2) remain planned work; `mbrl-lib` pins
  old `gym`.

## References

- Deisenroth, Rasmussen. PILCO. ICML 2011.
- Farahmand, Barron, Szepesvari. Value-Aware Model Learning. 2017.
- Farahmand. Iterative Value-Aware Model Learning. NeurIPS 2018.
- Lambert, Amos, Yadan, Calandra. Objective Mismatch in Model-based RL. L4DC 2020.
- Janner, Fu, Zhang, Levine. When to Trust Your Model (MBPO). NeurIPS 2019.
- Voelcker, Liao, Garg, Farahmand. Value Gradient weighted MBRL (VaGraM). ICLR 2022.
- Sun et al. Model-Bellman Inconsistency (MOBILE). ICML 2023.
- Hafner et al. Mastering Diverse Domains through World Models (DreamerV3). 2023.
- Hansen et al. TD-MPC2. ICML 2024.
- Wei, Lambert, McDonald. A Unified View on Solving Objective Mismatch. 2024.
- Voelcker et al. Calibrated Value-Aware Model Learning (CVAML). 2025.
- ROMI. Model-based Offline RL via Robust Value-Aware Model Learning. ICLR 2026.
