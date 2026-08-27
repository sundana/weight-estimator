# Theory: The Weight Estimator `w`

Notation (RESEARCH_PLAN.md Sec. 2). Model family `P_theta(s' | s, a)`; weighted model
loss `L_w(theta) = E_D[ w(s, a, s') * ell(P_theta(s' | s, a)) ]`. MLE is `w == 1`.
The value-aware weights studied here are `w = |V(s') - V(s)|` (VAML-1),
`w = ||grad_s V(s')||` (VaGraM), and `w = exp((V(s') - V_max)/tau)` (Lambert-style).

The weight is never known and is estimated from a data-dependent value estimate
`w_hat = w(V_hat, grad V_hat)`; `w_hat` is a random variable with bias and variance.
All statements below are validated in the tabular Distractor-Gym suite
(`experiments/exp1b_crossover.py`, `exp2_decomposition.py`).

---

## Lemma 1 (delta_TD decomposition)

Let `V` be twice continuously differentiable with Hessian `H_V`, let `s'` be the true
next state and `s_hat'` the model prediction, and set `e = s_hat' - s'` and
`delta_TD(s, a) = V(s') - V(s_hat')`. Taylor's theorem with Lagrange remainder gives

```
delta_TD = -grad_s V(s')^T e - (1/2) e^T H_V(xi) e      for some xi on [s', s_hat']
```

so that, with `phi = angle(grad_s V(s'), e)`,

```
|delta_TD| <= ||grad_s V(s')|| * ||e|| * |cos phi| + (1/2) ||H_V||_2 ||e||^2
```

The second term is the **curvature residual**: the part of the model-induced value
error unexplained by the first-order (value-gradient) term.

**Empirical check (exp2 Part A).** In the tabular suite the factorization
`|delta_TD| ~ ||grad V|| * eps_model * |cos phi|` holds with `R^2 = 0.99` under the
dense reward and `R^2 = 0.83` under the sparse (step-function) value, where the
curvature residual grows. This decomposition is the object value-aware weighting
targets: VaGraM weights by `||grad V||` alone, so it is mis-specified wherever the
curvature term or the `cos phi` distribution dominates.

---

## Theorem 1 (prediction-risk dominance of MLE)

Let `y_1, ..., y_n` be iid draws from the true transition at a fixed `(s, a)`, with
value scores `V(y_i)`. Let `w_i >= 0` be non-negative weights with `E[w_i] = 1`,
`Var(w_i) = sigma_w^2`, **independent of** `y_i`. Define the ratio estimators

```
mu_hat_MLE = mean(V(y_i)),        mu_hat_w = sum w_i V(y_i) / sum w_i
```

Then, to first order in `1/n`,

```
Var(mu_hat_w) = Var(mu_hat_MLE) + sigma_w^2 * E[V^2] / n
E[mu_hat_w]   = E[V(y)] + Cov(w, V(y)) + O(1/n)
```

so the expected squared value-prediction error satisfies

```
E[(mu_hat_w - E[V])^2] = E[(mu_hat_MLE - E[V])^2]
                         + sigma_w^2 * E[V^2] / n + Cov(w, V(y))^2 + O(n^{-3/2})
```

**Proof sketch.** Delta method for the ratio estimator: with `E[w] = 1`,
`Var(mu_hat_w) ~ (1/n) Var(w V) = (1/n)(Var(V) + sigma_w^2 E[V^2])` by independence;
bias via `E[w V]/E[w] ~ E[w V] = E[V] + Cov(w, V)`. The MLE terms drop out of the
variance by the same argument with `sigma_w = 0`.

**Consequences (validated in exp1b Part Bellman).** Value-aware weighting **cannot
improve the conditional-mean (Bellman-target) value prediction**: it only adds a
variance penalty `sigma_w^2 E[V^2]/n` and a bias term `Cov(w, V)^2`. Both grow when
the weight estimator is noisy (sparse reward -> high-variance `V_hat` -> large
`sigma_w^2`) and when the value is flat (distractor dims -> `Cov(w, V) ~ 0`). This
reproduces the empirical fact that value-aware model learning tends to underperform
MLE on global accuracy metrics (VaGraM 2022, Sec. 1). In the tabular suite, the
Bellman risk of the MLE model is at or below that of every weighted model, and the
estimated-weight model is strictly worse than the oracle-weight model (weight
estimator noise dominates the gap).

---

## Theorem 2 (effective sample size of the weight estimator)

For non-negative weights `w_i`, Kish's effective sample size is
`ESS(w) = (sum w)^2 / sum w^2`. With mean-normalized weights `E[w] = 1`,
`Var(w) = sigma_w^2`:

```
E[ESS] ~ n / (1 + sigma_w^2) <= n
```

**Proof.** `sum w ~ n` and `sum w^2 ~ n(1 + sigma_w^2)`; combine. 

**Consequence.** High-variance weights (from sparse/noisy value estimates) shrink the
effective sample size of the weighted model loss; this is the same variance inflation
as Theorem 1 expressed as a sample loss. `ESS` is a computable diagnostic per regime
(exp1b) and tracks the regimes in which value-aware weighting fails.

---

## Theorem 3 (decision-risk crossover - criterion)

Theorem 1 applies to conditional-mean prediction. The **decision** risk is the
value-relevant prediction error linearized along the value gradient

```
R_dec(P_hat) = E_{(s,a)} [ ( grad_s V(s') * (E_{P_hat}[Y | s, a] - E_{P*}[Y | s, a]) )^2 ]
```

For the weighted estimator at a fixed `(s, a)`, applying Theorem 1 to the vector
estimator `E_{P_hat_w}[Y]` gives, to first order,

```
R_dec(P_hat_w) - R_dec(P_hat_MLE)
    = [ (grad V * Cov(w, Y))^2 ]                       (value-relevance signal)
      - ||grad V||^2 * sigma_w^2 * E[||Y||^2] / n      (weight-noise penalty)
      + (weight-estimator bias term from w_hat = w(V_hat, grad V_hat))
```

The weighted model strictly improves the decision risk iff the signal term exceeds the
penalty. This yields the **crossover conditions**:

- **Distractor dims**: `grad_{s_d} V = 0`, so the signal `(grad V * Cov(w, Y))^2 = 0`
  and MLE is optimal.
- **Sparse reward / noisy `V_hat`**: `sigma_w^2` is large (weights sit on a noisy
  knife-edge value cliff), the penalty and estimator-bias term dominate, and MLE wins.
- **Informative coverage with a strong gradient**: `Cov(w, Y)` aligns with `grad V`,
  the signal dominates, and value-aware weighting wins.

The computable statistic

```
SNR_w = (E[w * ell])^2 / Var(w * ell)
```

proxies signal over penalty: it is large exactly when the weights are informative and
stable. In the tabular suite (exp1b), the policy-gradient alignment delta
`cos(g_true, g_w) - cos(g_true, g_MLE)` is positive in the goal-coverage sparse
regimes and negative precisely where `SNR_w` collapses (uniform coverage, dense
reward), while the Bellman-risk delta is at or below zero across all regimes
(Theorem 1). The alignment delta is a positive (if noisy) function of `SNR_w`
(Spearman/Pearson ~ 0.3-0.5 across the current sweep), motivating a sharper
decision-aware diagnostic as the follow-up.

---

## Notes and caveats

- Theorem 1 assumes weights independent of outcomes; oracle weights
  `w(V, grad V)` are outcome-dependent, so small improvements over MLE on the
  Bellman risk are possible in finite samples (observed in exp1b for
  goal-coverage dense regimes). The independence assumption matches the practical
  estimator `w_hat`, whose noise inflates `sigma_w^2`.
- The tabular suite uses Laplace smoothing `alpha`; the statements above hold for the
  pure empirical estimator, and smoothing shifts the balance slightly.
- Theorem 3 is stated as a criterion (signal vs penalty decomposition), not a full
  characterization; the empirical crossover in exp1b validates the sign and
  monotonicity rather than a closed-form threshold `tau`.