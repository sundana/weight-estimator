# When and Why Does Value-Aware Model Learning Fail?

## A Diagnostic Suite for Objective Mismatch in Model-Based Reinforcement Learning

**Status:** Draft v1 (Phase 1d). Tabular (Exp 1/2/1b) and continuous neural (Exp 4)
results included. Deep-baseline benchmark (Phase 2) pending provisioning.

---

## Abstract

Model-based reinforcement learning fits a dynamics model to predict the next state,
typically by maximum likelihood, and then uses that model to improve a policy. The
objective mismatch problem (Lambert et al., 2020) is that the model's likelihood need
not correlate with its ability to guide the policy. Value-aware model learning
(VaGraM; Voelcker et al., 2022) reweights the model loss by the value function, but
is empirically known to sometimes underperform MLE. We isolate *when and why* this
happens. We contribute (i) a theory of the value-aware weight estimator $w$ -- a
prediction-risk dominance result showing weighting can never improve the
Bellman-target prediction (it only adds a variance penalty $\sigma_w^2 E[V^2]/n$ and a
bias term), and a decision-risk crossover criterion showing weighting helps exactly
when the value-relevance signal exceeds the weight-noise penalty; (ii) a synthetic
benchmark, Distractor-Gym, that provably produces the failure regimes (spurious
state dimensions and sparse rewards); and (iii) two quantitative diagnostics --
policy-gradient alignment (cosine between true and model-induced gradients) and the
$|\delta_TD| ~ ||grad V|| * eps_model$ decomposition -- validated in a ground-truth
tabular suite and transferred to a continuous neural setting. We find that MLE
misguides the policy exactly when data coverage is concentrated where the value
changes (alignment near zero or negative), that value-aware weighting restores
alignment there, and that value-aware weighting collapses when its weights are
degenerate: sparse rewards (noisy knife-edge value gradient) or spread coverage.
The decomposition holds with R^2 = 0.99 (dense) / 0.83 (sparse reward).

---

## 1 Introduction

Model-based RL (MBRL) is a sample-efficient framework for continuous control
(Deisenroth & Rasmussen, 2011; Janner et al., 2019; Hafner et al., 2023; Hansen et
al., 2024). A dynamics model is fit to predict one-step transitions and then used by
a policy or planner. The standard objective for the model is maximum likelihood
(MLE), which minimizes prediction error uniformly over the state space. Lambert et
al. (2020) showed this objective can be *mismatched* with the downstream goal: the
one-step likelihood is not always correlated with control performance.

A principled family of remedies -- value-aware model learning (VAML; Farahmand et
al., 2017, 2018; VaGraM; Voelcker et al., 2022) -- reweights the model loss by the
value function, so that the model is accurate *where value changes*. VaGraM showed
these losses help under small model capacity and distracting state dimensions, yet
the literature also documents regimes in which value-aware losses *underperform*
MLE (Voelcker et al., 2022, Sec. 1; CVAML, 2025; ROMI, 2026). There is currently no
isolated, controlled characterization of when and why value-aware weighting fails,
and no modern diagnostic suite ported to current baselines.

This paper provides both. We make the following contributions:

1. **Theory of the weight estimator.** The value-aware weight `w = w(V, grad V)` is
   itself estimated from finite data and is a random variable with bias and
   variance. We prove (i) a *prediction-risk dominance* result: weighting can never
   improve the conditional-mean Bellman-target prediction, adding a variance penalty
   `sigma_w^2 E[V^2]/n` and a bias term (Theorem 1); (ii) an *effective sample size*
   reduction `ESS ~ n/(1 + sigma_w^2)` (Theorem 2); and (iii) a *decision-risk
   crossover* criterion: weighting improves the policy-relevant error iff the
   value-relevance signal `S = (grad V . Cov(w, Y))^2` exceeds the weight-noise
   penalty (Theorem 3).
2. **Distractor-Gym.** A synthetic environment with spurious state dimensions
   (complex, reward-irrelevant dynamics) and a sparse-reward variant, whose regime
   knobs provably produce the failure conditions of the theory.
3. **Diagnostics.** Policy-gradient alignment and the `|delta_TD| ~ ||grad V|| *
   eps_model` decomposition, measured in a ground-truth tabular suite and a
   continuous neural setting.

**Findings.** (a) MLE *fails to guide the policy* precisely when data coverage is
concentrated where the value changes: the cosine between the true and the
model-induced policy gradient falls to ~0 or negative (e.g., -0.08 under sparse
reward + goal-directed coverage), while value-aware weighting restores it to ~0.9.
(b) Value-aware weighting *fails* exactly when its weights are degenerate -- sparse
reward (weight sits on a noisy knife-edge value gradient) or spread coverage -- and
in these regimes MLE is strictly better. (c) On the Bellman-target prediction risk,
MLE dominates value-aware weighting in every regime we tested, and the gap is
dominated by weight-estimator noise (oracle weights are much better than estimated
weights) -- reproducing and explaining VaGraM's observation that value-aware losses
"tend to be inferior in practice to MLE."

---

## 2 Background and Related Work

**Objective mismatch.** Lambert et al. (2020) formalized that a model trained by
one-step MLE likelihood and used for control may not correlate with performance;
they proposed an exploratory sample reweighting `w(y) = c exp(-d(y))`. Wei et al.
(2024) survey the decision-aware remedies that followed.

**Value-aware model learning.** Farahmand et al. (2017, 2018) introduced VAML and
IterVAML, replacing the model loss with the induced value error. VaGraM (Voelcker et
al., 2022) uses the value gradient as the weight, and explicitly documents two
failure modes of naive value-aware learning: (a) the loss does not account for
exploration/coverage of the state space, and (b) the value function is often
non-smooth, so unbounded value-aware losses diverge. CVAML (Voelcker et al., 2025)
shows sample-based value-aware losses are uncalibrated for stochastic models; ROMI
(2026) adds adaptive MLE+value weighting for offline RL. MOBILE (Sun et al., 2023)
uses model-Bellman inconsistency for offline pessimism.

**Diagnostics.** Lambert et al. (2020) measured the correlation between one-step
likelihood and return. We modernize this: our diagnostics are per-run, quantitative,
and cover gradient alignment and the value-error decomposition, ported (Phase 2) to
MBPO, DreamerV3, TD-MPC2, and VaGraM.

---

## 3 Theory: The Weight Estimator `w`

Setup. Model family `P_theta(s' | s, a)`; weighted loss
`L_w(theta) = E_D[ w(s, a, s') * ell(P_theta(s' | s, a)) ]`. MLE is `w = 1`.
Value-aware weights: `w = |V(s') - V(s)|` (VAML-1), `w = ||grad_s V(s')||` (VaGraM),
`w = exp((V(s') - V_max)/tau)` (Lambert). The weight is never known; it is estimated
from a data-dependent value estimate, so `w_hat = w(V_hat, grad V_hat)` has bias and
variance. Full statements and proofs in `paper/notes/theory.md`.

**Lemma 1 (decomposition).** Let `V` be twice differentiable with Hessian `H_V`, and
let `e = s_hat' - s'` be the model prediction error. Then, with
`phi = angle(grad_s V(s'), e)`,

```
delta_TD(s,a) := V(s') - V(s_hat')
  = -grad_s V(s') . e - (1/2) e^T H_V(xi) e
|delta_TD| <= ||grad_s V(s')|| * ||e|| * |cos phi| + (1/2) ||H_V||_2 ||e||^2
```

The second term is the **curvature residual** that the first-order
`||grad V|| * eps_model` factorization ignores. This decomposition is exactly what
value-aware weighting targets: VaGraM weights by `||grad V||` alone, so it is
mis-specified wherever the curvature term or the `cos phi` distribution dominates.
*Empirical check (Exp 2 Part A):* R^2 = 0.99 (dense reward) and 0.83 (sparse
step-function value), where the curvature residual grows.

**Theorem 1 (prediction-risk dominance).** Let `y_1..y_n` be iid draws from the true
transition at a fixed `(s,a)`, and `w_i` non-negative weights with `E[w]=1`,
`Var(w)=sigma_w^2`, independent of `y`. Then, to first order in `1/n`:

```
Var(mu_hat_w) = Var(mu_hat_MLE) + sigma_w^2 * E[V^2] / n
E[mu_hat_w]   = E[V(y)] + Cov(w, V(y)) + O(1/n)
E[(mu_hat_w - E[V])^2] = E[(mu_hat_MLE - E[V])^2]
                        + sigma_w^2 E[V^2]/n + Cov(w, V(y))^2 + O(n^{-3/2})
```

Value-aware weighting **cannot improve the conditional-mean (Bellman-target)
prediction**: it adds variance and bias. Both grow when the weight estimator is
noisy (sparse reward -> noisy `V_hat` -> large `sigma_w^2`) and when the value is
flat (distractor dims -> `Cov(w, V) ~ 0`). *Empirical check (Exp 1b):* the
Bellman risk of MLE is at or below that of every weighted model across all regimes;
the estimated-weight model is strictly worse than the oracle-weight model.

**Theorem 2 (effective sample size).** For mean-normalized weights,
`ESS(w) = (sum w)^2 / sum w^2 ~ n / (1 + sigma_w^2) <= n`. High-variance weights
shrink the effective sample size of the weighted loss; this is the same variance
inflation as Theorem 1 expressed as sample loss.

**Theorem 3 (decision-risk crossover criterion).** The *decision* risk is the
value-relevant prediction error linearized along the value gradient,
`R_dec(P_hat) = E[ (grad_s V(s') . (E_{P_hat}[Y] - E_{P*}[Y]))^2 ]`. Applying
Theorem 1 to the vector estimator gives

```
R_dec(P_hat_w) - R_dec(P_hat_MLE)
  = [ (grad V . Cov(w, Y))^2 ]            (value-relevance signal)
    - ||grad V||^2 sigma_w^2 E[||Y||^2]/n  (weight-noise penalty)
    + (weight-estimator bias term)
```

The weighted model improves the decision risk iff the signal exceeds the penalty.
This yields the crossover conditions:

- **Distractor dims**: `grad_{s_d} V = 0` -> signal 0 -> MLE optimal.
- **Sparse reward / noisy `V_hat`**: `sigma_w^2` large (weights on a noisy value
  cliff) -> penalty dominates -> MLE wins.
- **Informative coverage with a strong gradient**: signal dominates -> value-aware
  wins.

The computable statistic `SNR_w = (E[w * ell])^2 / Var(w * ell)` proxies signal over
penalty. *Empirical check (Exp 1b):* the alignment delta
`cos(g_true, g_w) - cos(g_true, g_MLE)` is positive in goal-coverage sparse regimes
and negative precisely where `SNR_w` collapses; it is a positive-but-noisy function
of `SNR_w` (Pearson ~ 0.25-0.34), motivating a sharper diagnostic (open item).

---

## 4 Distractor-Gym

A synthetic environment isolating the failure conditions. State
`s = (s_c, s_d)` with `d_c` control-relevant and `d_d` distractor dimensions whose
dynamics are complex (linear, nonlinear chaotic, or stochastic white noise) and
whose reward relevance is zero, so `grad_{s_d} V = 0` by construction.

- **Tabular suite** (ground-truth lab): 1D control axis `s_c in [-3, 3]` on a grid
  with actions {-1, 0, +1} and quantized transition noise; distractor block evolves
  by a deterministic chaotic logistic map per dimension. Reward dense
  `r = exp(-(x - goal)^2/2)` or sparse `r = 1[|x - goal| <= radius]`. Exact value
  functions, value gradients, policy gradients, and Bellman risks are computable.
- **Continuous suite**: Pendulum-v1 with an appended chaotic or white-noise
  distractor block (Gymnasium wrapper). Reward depends only on the control block.

Regime knobs swept: `d_d` (distractor count), distractor dynamics class, reward
sparsity (via goal radius and discount `gamma`), transition noise, coverage
(uniform vs goal-directed data collection), and model capacity. These produce the
mismatch phase diagram (MLE-win / value-aware-win / both-fail) that the diagnostics
quantify.

---

## 5 Experiment 1: Policy-Gradient Alignment

**Setup.** For a fixed (deliberately off-center) softmax policy `pi`, compute the
exact policy gradient under the true dynamics `g_true` and under a model fitted by
each loss family `g_model`; report `cos(g_true, g_model)` (tabulated in Exp 1 over
`d_d`, reward sparsity, coverage; averaged over 5 seeds).

**Result.** Table 1. MLE *misguides the policy* under sparse reward + goal-directed
coverage: alignment falls to -0.08 (its gradient points against the true
improvement direction), while value-aware VAML-1/Lambert restore alignment to ~0.9.
Under dense reward + uniform coverage, MLE is near-perfectly aligned (1.00) and
value-aware weighting is equal or worse. Value-aware *fails* in two regimes: sparse
reward + uniform coverage (VaGraM alignment 0.03 vs MLE 0.99 -- the knife-edge
weights concentrate the model on a razor-thin band) and dense + uniform (VAML-1
drops to 0.88). These are exactly the Theorem 3 predictions.

| regime              | MLE  | VAML-1 | VaGraM | Lambert |
|---------------------|------|--------|--------|---------|
| dense + uniform     | 1.00 | 0.88   | 1.00   | 0.31    |
| sparse + uniform    | 0.99 | 0.95   | 0.03   | 0.95    |
| sparse + goal       | -0.08| 0.89   | 0.29   | 0.91    |

## 6 Experiment 2: Decomposition and Weight-Estimator Ablation

**Part A (Lemma 1).** Across regimes, the factorization
`|delta_TD| ~ ||grad V|| * eps_model * |cos phi|` holds with slope ~1.06-1.18 and
R^2 = 0.99 (dense) / 0.83 (sparse). The curvature residual (0.006 vs 0.022) is the
gap and grows with reward sparsity -- the regime where gradient-only weighting is
mis-specified.

**Part B (weight estimator vs policy effects).** Oracle weights (true `V`, `grad V`)
vs estimated weights (`V_hat` from data). The estimated-weight model has
consistently higher Bellman risk than the oracle-weight model (e.g., VAML-1 risk
0.239 vs 0.184; VaGraM 0.466 vs 0.168 at `d_d=2`), isolating the weight-estimator
noise term `sigma_w^2 E[V^2]/n`. Yet value-aware alignment improvements are robust
to this noise (align_est and align_ora both well above MLE in sparse regimes). The
value-aware benefit is decision-level (alignment) and does not come from better
Bellman prediction -- consistent with Theorems 1 and 3.

## 7 Experiment 4: Continuous Neural Transfer

Small MLP dynamics model on Pendulum-plus-distractors, MLE-MSE vs the VaGraM
projection loss `(grad r(s') . (s_hat' - s'))^2`.

- **Pure projection (lam = 1) diverges** and fails to train a usable model --
  reproducing VaGraM's documented divergence of naive value-aware losses.
- **Bounded combination (lam = 0.5)** is stable and matches MLE on global MSE, while
  improving the decision-relevant metrics when unpredictable stochastic distractors
  consume capacity: value error 0.214 vs 0.306 (d_d=8) and 0.524 vs 0.674 (d_d=16);
  directional alignment 0.971/0.942 vs 0.952/0.905. The benefit grows with
  distractor count and requires sufficient model capacity.

## 8 Discussion

**When MLE fails to guide the policy.** Under goal-concentrated coverage with sparse
reward, the MLE model's gradient misaligns (cos ~ 0 or negative) because the value
cliff is thin and the model spreads its accuracy where the policy does not visit.
Value-aware weighting reallocates the model's effective mass to the cliff and
restores alignment.

**Why value-aware fails.** Two mechanisms, both predicted by the theory: (1) on the
Bellman/prediction risk it *cannot* help -- it only adds weight-noise variance --
which reproduces VaGraM's "inferior in practice" observation; (2) on the decision
risk it helps only when the value-relevance signal beats the weight-noise penalty,
which fails when weights are degenerate (sparse reward, spread coverage).

**Practical rule of thumb.** Measure `SNR_w` and the gradient alignment during
training. If `SNR_w` is low or the alignment delta is non-positive, value-aware
weighting is not helping; the model is in the MLE-optimal regime.

## 9 Limitations and Future Work

- The `SNR_w` crossover statistic is a positive-but-noisy predictor; a sharper
  decision-aware statistic is open work.
- Tabular results are exact but stylized; the continuous transfer used reward
  gradients as the value proxy, not a learned value function.
- Phase 2: the diagnostics must be ported to MBPO, DreamerV3, TD-MPC2, and VaGraM
  (benchmark provisioning is an open item; `mbrl-lib` pins old `gym`).

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