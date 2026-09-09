# Research Plan: When and Why Does Value-Aware Model Learning Fail?

## Weight-Estimator Theory & a Modern Objective-Mismatch Diagnostic Suite

**Working title:** *Diagnosing Objective Mismatch: When (and Why) MSE/MLE Stops Guiding the Policy, and the Bias-Variance of Value-Aware Weighting*

**Target venue:** ICML/ICLR (analysis + benchmark paper).

**Format:** paper + released diagnostic suite + Distractor-Gym benchmark.

**Goal:** Isolate the conditions in which MSE/MLE fails to guide the policy, and derive the
bias-variance analysis of the weight estimator $w$. Deliver a diagnostic suite that answers
*when and why* value-aware model learning fails, and modernize the objective-mismatch
diagnostic of Lambert et al. (2020) for current baselines (MBPO, DreamerV3, TD-MPC2, VaGraM).

---

## 1. Motivation & Gap

Lambert et al. (2020, L4DC) formalized **objective mismatch**: models trained by one-step MLE
likelihood are used for control, yet likelihood does not correlate with downstream returns.
Their mitigation (re-weighting samples by $w(y) = c \cdot e^{−d(y)}$) was exploratory, and their
diagnostics predate current baselines.

The field's principled answer is the **value-aware model learning (VAML) family** —
Farahmand (2017, 2018), VaGraM (Voelcker et al., 2022), CVAML (2025) — which re-weights the
model loss by the value function. The literature already documents that these *sometimes*
underperform MLE (VaGraM Sec. 1; CVAML stochasticity results; ROMI 2026 adaptive weighting),
but:

- There is **no isolated, controlled characterization** of *when and why* value-aware
  weighting fails.
- There is **no modern diagnostic suite** ported to current baselines (MBPO, DreamerV3,
  TD-MPC2, VaGraM).

**This project's contribution:** (a) a theory of the **weight estimator** `w` (bias/variance
and the crossover condition where MLE strictly beats value-aware weighting), (b) a controlled
synthetic environment (**Distractor-Gym**) that isolates the failure conditions, (c) two
quantitative diagnostics — **gradient alignment** (cosine similarity) and the **`delta_TD`
decomposition** — and (d) a benchmark suite modernizing Lambert's diagnostics.

---

## 2. Theoretical Core: The Weight Estimator $w$

### 2.1 Setup

Model family $p_\theta(s^\prime | s, a)$, dataset $D$ of transitions. A general **weighted
model-learning family**:

$$
L_w(\theta) \;=\; \mathbb{E}_{D}\!\left[\; w(s, a, s') \cdot \ell\!\left(p_\theta(s' \mid s, a)\right)\; \right]
$$

- MLE/MSE: $w = 1$.
- VAML-1: $w = \lvert V(s') - V(s) \rvert$.
- VaGraM: $w = \lVert \nabla_s V(s') \rVert$ (gradient-weighted, with bounded variant).
- Lambert: $w = w(s, a) = c \cdot e^{−d(s, a)}$.
- Decision-aligned (MOBILE-style): $w$ derived from model-Bellman inconsistency.

The weight is never known; it is **estimated** from $\hat{V},\ \nabla\hat{V}$ computed on finite
data, so the *weight estimator* $\hat{w} = w(\hat{V},\ \nabla\hat{V})$ is a random variable with bias
and variance.

### 2.2 Target decomposition (Task 3)

Define the model-induced TD (bootstrap-target) error:

$$
\delta_{\text{TD}}(s, a) \;=\; \bigl\lvert V(s') - V(\hat{s}')\bigr\rvert, \qquad
\hat{s}' = \text{model prediction},\ \ s' = \text{true next state}
$$

First-order Taylor gives the decomposition to be empirically validated:

$$
\lvert \delta_{\text{TD}} \rvert
\;\approx\; \bigl\lvert \nabla_s V(s') \cdot (\hat{s}' - s')\bigr\rvert
\;=\; \lVert \nabla V(s') \rVert \cdot \varepsilon_{\text{model}}(s,a) \cdot \lvert \cos \phi(s,a) \rvert
$$
$$
\varepsilon_{\text{model}} = \lVert \hat{s}' - s' \rVert, \qquad
\phi = \text{angle}\bigl(\nabla V(s'),\; \hat{s}' - s'\bigr)
$$

The `~` hides two terms we will **measure**, not assume: the **curvature residual**
(second-order term) and the **distribution of cos phi**. This is exactly why the decomposition
is a diagnostic: VaGraM-type weights use only `||grad V||`; when `cos phi` is small or the
curvature term is large, gradient-weighted learning is mis-specified even where `||grad V||`
is large.

### 2.3 Bias-variance of `w` (Task 3 theory)

For the weighted loss with estimated weights, we derive a risk decomposition for the *induced*
value error of the learned model:

$$
R(\hat{w}) \;=\; \underbrace{\operatorname{Bias}^2(\hat{w})}_{\text{weight-estimation bias}}
\;+\; \underbrace{\operatorname{Var}\!\bigl(\hat{w} \cdot \ell\bigr)}_{\text{weight-induced variance}}
\;+\; \underbrace{\text{value-relevance term}}_{\text{aligned with the induced value error}}
$$

Planned formal results:

- **Lemma 1 (factorization):** exact Taylor expansion of $\delta_{\text{TD}}$ with an explicit
  second-order (curvature $H_V$) remainder bound.
- **Theorem 1 (crossover):** comparison of MLE-model risk vs value-weighted-model risk. Under
  a Lipschitz value function and a noise model on $\hat{V}$, define the weight signal-to-noise
  ratio $\mathrm{SNR}_w = \dfrac{\bigl(\mathbb{E}[\lVert \nabla V \rVert \cdot \varepsilon \cdot \cos \phi]\bigr)^2}{\operatorname{Var}(\hat{w} \cdot \ell)}$.
  Show there exists a threshold $\tau$ such that for $\mathrm{SNR}_w < \tau$ the MLE model
  ($w = 1$) has strictly lower risk — i.e., the **exact condition where MLE/MSE is optimal**
  and value-aware weighting fails. Predictions:
  - **Sparse reward** $\to$ $\hat{V}$ flat/noisy $\to$ $\operatorname{Bias}^2$ large,
    $\operatorname{Var}$ large $\to$ $\mathrm{SNR}_w$ collapses.
  - **Distractor dims** $\to$ model capacity spent on $s_d$ where $\nabla V = 0$
    $\to$ value-relevance term $\approx 0$ but MLE error concentrated
    $\to$ $\lVert \nabla V \rVert \cdot \varepsilon \cdot \cos \phi$ small $\to$ MLE dominates.
- **Theorem 2 (effective sample size):** high-variance $\hat{w}$ reduces the effective sample
  size of the weighted loss (ESS bound); connects Theorem 1 to a practical, computable
  diagnostic.

These mirror but **extend** VaGraM's qualitative analysis (exploration + function
approximation) into a quantitative, testable crossover.

---

## 3. Distractor-Gym: The Synthetic Environment (Task 1)

State $s = (s_c,\ s_d)$: $d_c$ control-relevant dims, $d_d$ distractor dims with **complex
dynamics and zero reward relevance**. Reward $r(s, a)$ depends only on $s_c$. Value is flat in
$s_d$: $\nabla_{s_d} V = 0$ by construction.

**Distractor dynamics classes** (increasing failure pressure):

1. Linear coupled oscillators.
2. Nonlinear (logistic-map / chaotic).
3. High-dimensional stochastic noise processes (hard to fit but reward-irrelevant).

**Sparse-reward variant:** $r \neq 0$ only inside a small goal region of $s_c$ (density knob
$\rho$). Effect: $V$ flat almost everywhere $\to$ value-aware weights $\approx 0$/noisy almost
everywhere $\to$ value weighting fails *by construction*; this is the "where value weighting
should fail" regime.

**Swept regime knobs:**

1. $d_d$ ($0 \to$ large) — distractor count.
2. Distractor dynamics class.
3. Reward sparsity $\rho$ + goal-region shape.
4. Transition stochasticity $\sigma$.
5. Model capacity (under/over-parameterized).

**Two linked suites (same knobs):**

- **Tabular / known-dynamics suite:** exact computation of $g_{\text{true}}$, $\delta_{\text{TD}}$,
  $\varepsilon_{\text{model}}$, $\nabla V$ $\to$ the ground-truth lab for Theorem 1 and the
  decomposition.
- **Continuous suite (Gymnasium/DMC-style wrapper):** for deep baselines, with the same
  distractor structure appended to a DMC-style reach task (default) or a 1D/2D control task.

**Expected phase diagram** (the paper's headline figure): a 2D regime map over
(distractor complexity x reward sparsity) showing MLE-win / value-aware-win / both-fail
regions, with the Theorem-1 crossover superimposed.

---

## 4. Experiment 1 — Gradient Alignment (Task 2)

Define:

- $g_{\text{true}} = \nabla_{\theta_\pi} J(\pi)$ — policy gradient under **true** dynamics
  (exact in the tabular suite; MC-rollout estimate in continuous).
- $g_{\text{model}} = \nabla_{\theta_\pi} \hat{J}(\pi)$ — policy gradient computed **inside the
  model** (model rollouts, as in MBPO/Dyna).

Measure **$\cos\bigl(g_{\text{true}},\ g_{\text{model}}\bigr)$** over training for each loss family:

- **MLE** ($w = 1$), **value-weighted** (VAML-1, VaGraM), **decision-aligned**
  (model-Bellman / MOBILE-style weighting).

Diagnostic attribution: also project the *model-loss gradient* $\nabla_\theta L_w$ onto the
value-relevant parameter direction, to show *which* weight choice targets the direction that
matters.

**Hypotheses:**

- **H1:** in benign regimes, value-aware weights increase
  $\cos\bigl(g_{\text{true}},\ g_{\text{model}}\bigr)$ (model updates move the policy in the
  true improvement direction).
- **H2:** in distractor/sparse regimes, alignment collapses ($\to 0$ or negative) for *all*
  families — isolating the failure condition.
- **H3:** decision-aligned (Bellman-error) weighting behaves *differently* from gradient-based
  weighting under sparse reward (Bellman targets are themselves noisy when `V_hat` is sparse),
  giving a three-way comparison.

---

## 5. Experiment 2 — Decomposition & Weight-Estimator Ablation (Task 3)

**Part A — Test the decomposition.** Per-transition measurement (tabular suite) of
$\delta_{\text{TD}}$, $\lVert \nabla V \rVert$, $\varepsilon_{\text{model}}$, $\cos \phi$,
curvature residual. Report: $R^2$ and slope of the factorization, per-bin correlation, and the
*residual (curvature) term* as a function of regime. This quantifies *which* approximation in
$\lvert \delta_{\text{TD}} \rvert \approx \lVert \nabla V \rVert \cdot \varepsilon_{\text{model}}$
breaks down, and where.

**Part B — Separate the weight estimator from policy effects (ablation matrix):**

1. **Oracle vs estimated weights:** $\hat{w}(\hat{V},\ \nabla\hat{V})$ vs
   $w(V_{\text{true}},\ \nabla V_{\text{true}})$ — isolates the *weight-estimation bias*
   (Theorem 1's $\operatorname{Bias}^2$).
2. **Fixed model, varying weights:** same model, different $w$ — isolates whether the weight
   itself changes model quality.
3. **Fixed weights, varying policy learner:** same $\hat{w}$, different policy-update schemes —
   isolates policy-side interaction.
4. **Effective sample size / $\mathrm{SNR}_w$:** measure ESS and $\mathrm{SNR}_w$ across
   regimes and check against the Theorem-1 crossover (theory-experiment match plot).

---

## 6. Benchmark Suite — Modernizing Lambert's Diagnostics (Deliverable)

- **Reproduce Lambert et al.'s core diagnostic** (one-step likelihood vs return correlation,
  the "mismatch" curve) on Distractor-Gym.
- **Extend** with the new per-run diagnostics: gradient-alignment curve, $\delta_{\text{TD}}$-
  decomposition fit, weight-estimator bias/var/ESS/$\mathrm{SNR}_w$.
- **Phased baselines:**
  - **Phase 1:** MBPO (MLE model), MBPO+VaGraM, and native VaGraM. VaGraM's own regime (small
    capacity + distractors) is exactly where the crossover should appear.
  - **Phase 2:** DreamerV3, TD-MPC2 (instrumented with the same diagnostics where applicable).
- **Protocol:** standardized configs, >= 10 seeds, rliable-style median/IQR + performance
  profiles (Agarwal et al., 2021); pinned baseline commits.

---

## 7. Paper Structure (Outline)

1. **Introduction** — objective mismatch, value-aware learning, the missing diagnostic.
2. **Background & Related Work** — Lambert; VAML/IterVAML; VaGraM; CVAML; MOBILE;
   decision-aware taxonomy (Wei et al., 2024).
3. **Theory** — Lemma 1, Theorem 1 (crossover), Theorem 2 (ESS); conditions where MLE is
   optimal.
4. **Distractor-Gym** — design, regime knobs, phase diagram.
5. **Gradient-Alignment Analysis** (Exp 1).
6. **delta_TD Decomposition & Weight-Estimator Ablation** (Exp 2).
7. **Benchmark** — Phase 1 (MBPO/VaGraM) + Phase 2 (DreamerV3/TD-MPC2) diagnostics.
8. **Discussion & Conclusion** — when to use value-aware learning; practical rule of thumb from
   `SNR_w`.

---

## 8. Timeline (phased, ~6-8 months)

| Phase | Milestone | Est. |
|---|---|---|
| 0 | Repo scaffold, diagnostics API, theory skeleton | 1-2 wk |
| 1a | Tabular Distractor-Gym + exact `g_true` / `delta_TD` / `eps_model` | 3 wk |
| 1b | Theorem proofs + validation in tabular suite (crossover plot) | 3-4 wk |
| 1c | MBPO + VaGraM continuous runs + full diagnostics | 4-6 wk |
| 1d | Paper draft v1, internal review | 2-3 wk |
| 2 | Instrument DreamerV3 + TD-MPC2 | 4-6 wk |
| 3 | Full benchmark, paper v2, submission prep | 3-4 wk |

---

## 9. Risks & Mitigations

- **Compute cost** -> phased plan; tabular suite is cheap and carries the core scientific
  claims.
- **Negative results** (value-aware never helps in some regime) -> framed as the paper's point
  (crossover theorem), not a bug.
- **Reproducibility of baselines** -> pin official repos/commits; VaGraM and MBPO use
  mbrl-lib, DreamerV3/TD-MPC2 use official code.
- **Theory-practice gap** (exploration, function approximation) -> explicitly inherited from
  VaGraM's analysis; Theorem 1 stated for the tabular setting and *empirically checked*
  against the deep regime via the phase diagram.

---

## 10. Deliverables

1. `RESEARCH_PLAN.md` — this document.
2. Repo scaffold — `src/distractor_gym` package (envs, losses, diagnostics), `configs/`,
   `experiments/`, `paper/`, `pyproject.toml`, and the diagnostics API.

---

## 11. Key References

- Lambert, Amos, Yadan, Calandra. **Objective Mismatch in Model-based Reinforcement Learning.** L4DC 2020. arXiv:2002.04523.
- Farahmand, Barron, Szepesvari. **Value-Aware Model Learning.** 2017.
- Farahmand. **Iterative Value-Aware Model Learning.** NeurIPS 2018.
- Voelcker, Liao, Garg, Farahmand. **Value Gradient weighted Model-Based Reinforcement Learning (VaGraM).** ICLR 2022. arXiv:2204.01464.
- Janner, Fu, Zhang, Levine. **When to Trust Your Model (MBPO).** NeurIPS 2019.
- Hafner, Pasukonis, Ba, Lillicrap. **Mastering Diverse Domains through World Models (DreamerV3).** 2023.
- Hansen, Su, Wang. **TD-MPC2.** ICML 2024.
- Sun, Zhang, Jia, Lin, Ye, Yu. **Model-Bellman Inconsistency for Model-based Offline RL (MOBILE).** ICML 2023.
- Voelcker et al. **Calibrated Value-Aware Model Learning with (Probabilistic/Stochastic) Environment Models (CVAML).** 2025.
- ROMI. **Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting.** ICLR 2026.
- Wei, Lambert, McDonald. **A Unified View on Solving Objective Mismatch in Model-Based Reinforcement Learning.** 2024.