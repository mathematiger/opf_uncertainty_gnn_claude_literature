## 2026-11-03 (Tuesday) — X019 · Track 04: Stochastic, chance-constrained, risk-averse & robust optimization — Hedging against your own empirical distribution: a tutorial on Wasserstein DRO

**Type:** Cross-phase preview
**Main reference:** [Wasserstein Distributionally Robust Optimization: Theory and Applications in Machine Learning — D. Kuhn, P. Mohajerin Esfahani, V. A. Nguyen & S. Shafieezadeh-Abadeh (2019), INFORMS TutORials in Operations Research, pp. 130–166](https://arxiv.org/abs/1908.08729) — read Section 1 (pp. 1–7: the optimizer's curse, Example 2; Wasserstein distance, Definition 1; the Kantorovich–Rubinstein theorem, Theorem 2). Then read Section 2.1 (Theorems 5–7: Lipschitz upper bound, robust lower bound, strong duality), Theorem 10 in Section 2.2, Theorems 18–19 at the start of Section 3 and Section 4.1 up to Proposition 2. Skip the elliptical results in Section 2.3 and the rest of Section 4. The theorem numbers are those of arXiv v1. The arXiv v2 (November 2024) corrects several errors in the published version, so read v2 if you need exact statements. Published version: [DOI 10.1287/educ.2019.0198](https://doi.org/10.1287/educ.2019.0198).
**Further reading:**
- [Distributionally Robust Logistic Regression — S. Shafieezadeh-Abadeh, P. Mohajerin Esfahani & D. Kuhn (2015), NeurIPS 2015](https://arxiv.org/abs/1509.09259) — the first Wasserstein DRO classifier. A short paper that shows the regularisation link from the inside.
- [Distributionally Robust Optimization: A Review — H. Rahimian & S. Mehrotra (2022), Open Journal of Mathematical Optimization 3, article 4](https://arxiv.org/abs/1908.05659) — the map of ambiguity sets (moment-based, φ-divergence, Wasserstein, ...). Use it as a reference, not a cover-to-cover read.

**Why this matters** — A dispatch that is optimal for the historical forecast-error samples is, by construction, tuned to the noise in those samples, so it will disappoint on new days. Distributionally robust optimization (DRO) makes this effect explicit: it optimizes against the worst distribution "near" the data, and with the right radius this gives a certified upper bound on out-of-sample cost. Anyone building learned OPF solvers under forecast uncertainty will meet DRO both as a target formulation and as an explanation of why regularisation works.

**Core ideas**

**1. The optimizer's curse (Section 1, Example 2).** Let `R(P, ℓ) = E_P[ℓ(ξ)]` be the expected cost of a decision whose cost function is `ℓ`. Here `ξ` is the uncertainty, e.g. wind forecast errors at several buses. Using the empirical distribution `P̂_N = (1/N) Σ_i δ_{ξ̂_i}` gives an unbiased risk estimate *for a fixed decision*. But the minimiser adapts to the sample:

```
E[ min_ℓ R(P̂_N, ℓ) ]  ≤  min_ℓ R(P, ℓ)  ≤  R(P, ℓ*)
   in-sample cost          true optimum      out-of-sample cost of ℓ*
```

The gap between the right and left ends is the "post-decision disappointment", and on average it is positive. You know this from ML as training loss versus test loss. The difference here is that the "model" is a decision, such as a dispatch or a reserve level, so overfitting shows up as real money or as violated limits.

**2. The ambiguity set.** DRO replaces `P̂_N` by a ball of distributions around it and optimises the worst case:

```
min_ℓ  sup_{Q : W_p(Q, P̂_N) ≤ ε}  E_Q[ℓ(ξ)]
W_p(Q, Q')^p = inf_{π ∈ Π(Q,Q')} ∫ ‖ξ − ξ'‖^p π(dξ, dξ')
```

`W_p` is the optimal-transport cost of moving the probability mass of `Q` onto `Q'`. Two limits frame the idea. With `ε = 0` you recover sample-average approximation, i.e. empirical risk minimisation. As `ε → ∞` you get classical robust optimization over the support (X010 was a refined version of that end). The Wasserstein ball has an advantage over KL or other φ-divergence balls: it contains distributions with support *off* the training points. A KL ball around `P̂_N` contains only reweightings of the N samples, so it can never imagine a forecast error worse than one already seen.

**3. Three results that make it tractable.**

- *Lipschitz bound (Theorem 5).* By Kantorovich–Rubinstein, `sup_{Q ∈ B_ε} E_Q[ℓ] ≤ E_{P̂_N}[ℓ] + ε · Lip(ℓ)`. In words, the adversary gains at most slope × transport budget.
- *Strong duality (Theorem 7).* The infinite-dimensional sup over distributions equals a one-dimensional convex problem:

```
sup_{Q ∈ B_{ε,p}(P̂_N)} E_Q[ℓ(ξ)]  =  inf_{γ ≥ 0}  γ ε^p + (1/N) Σ_i sup_{z ∈ Ξ} [ ℓ(z) − γ ‖z − ξ̂_i‖^p ]
```

`γ` is the Lagrange multiplier of the transport budget (A13's machinery, with a distribution as the primal variable). Each inner sup is a per-sample adversarial perturbation with a penalty on the distance moved. It looks just like adversarial training, except that the perturbation budget is shared across the dataset rather than fixed per sample. For piecewise-linear or piecewise-concave losses on a convex support, the inner sups can be dualised again, and the whole DRO problem becomes a finite convex program (Theorems 8–9).
- *Exact regularisation (Theorem 10, Proposition 2).* If `p = 1`, `Ξ = R^m` and `ℓ` is convex, the bound of Theorem 5 is tight: worst-case risk = empirical risk + `ε · Lip(ℓ)`. For a linear classifier with logloss or hinge loss, `Lip = ‖w‖_*`, so Wasserstein DRO with labels that cannot be flipped *is* norm-regularised ERM with weight `ε`. The norm that measures transport cost on the inputs fixes which dual norm becomes the regulariser.

**4. Choosing the radius (Section 3, Theorems 18–19).** For light-tailed `P`, measure concentration gives a radius `ε_N(η)` such that `P ∈ B_ε(P̂_N)` with probability at least `1 − η`. On that event, the DRO optimal value upper-bounds the true expected cost of the DRO decision, a finite-sample certificate. The catch: in dimension `m > 2p` the radius shrinks only like `N^(−p/m)`, a curse of dimensionality the tutorial calls essentially unavoidable. In practice `ε` is therefore often tuned by cross-validation, and that weakens the certificate. That trade-off between a provable radius and a useful radius comes back throughout phase B.

**A small example**

One bus and one decision: book upward reserve `r` (MW) at 10 €/MW. Any shortfall `max(ξ − r, 0)` costs 100 €/MW. Four historical forecast errors: `ξ̂ = {−20, 0, 10, 30}` MW, each with mass 1/4. The cost is `ℓ(ξ; r) = 10r + 100 · max(ξ − r, 0)`.

*Empirical optimum.* For `10 ≤ r ≤ 30` the cost is `10r + 25(30 − r) = 750 − 15r`, which decreases in r. So `r_SAA = 30` at 300 € (exactly the largest sample: the curse in action).

*Unbounded support, `ε = 2`, `p = 1`.* `ℓ` is convex in ξ with slope 100, so by Theorem 10 the worst case is `cost + 200` for *every* `r`. The decision does not move. The adversary takes the sample at 30 (mass 1/4) and pushes it 8 MW past `r`. That costs `8/4 = 2` units of transport and adds `100 · 8/4 = 200` €.

*Bounded support `ξ ≤ 40` MW (say, installed wind capacity).* Now the adversary first has to spend budget reaching `r` and cannot go past 40:
- `r = 30`: 300 + 200 = 500
- `r = 35`: reaching 35 costs 1.25, the remaining 0.75 buys 3 MW of shortfall → 350 + 75 = 425
- `r = 38`: reaching 38 costs exactly 2, no budget left → 380
- `r = 40`: no shortfall possible → 400

The DRO choice is `r = 38 = 30 + N·ε`. It adds a margin above the worst sample seen, sized by the transport budget. Conservatism enters through the geometry of the support and the slopes, not through a hand-set safety factor.

**How it connects**

- **Builds on:** X001 (Powell's policy classes: DRO is one way to build a "cost function approximation" policy that hedges), X010 (budgeted robust optimization, the `ε → ∞` end of the spectrum, with a set of parameters instead of a set of distributions), A13 (Theorem 7 is Lagrangian duality with a distribution as the primal variable, and `γ` is a price on transport budget), X005 and X014 (distributions as objects with a metric; MMD and Wasserstein are the two standard choices), X017 (a finite-sample guarantee from data, here for a decision rather than an interval).
- **Differs from:** conformal methods (X008, X017) calibrate a single quantity with distribution-free coverage. Wasserstein DRO certifies a whole expected cost but needs light tails, and its radius degrades with dimension.
- **Next:** in phase B, B01 places DRO in the taxonomy, B04–B05 introduce CVaR and chance constraints (a worst-case CVaR over a Wasserstein ball is a common hybrid), and B12 applies Wasserstein DRO to OPF. The focus days continue with A14 (interior-point methods), the solvers that the convex programs of Theorems 8–10 are eventually handed to.

**Check yourself**

1. Why does a KL-divergence ball around `P̂_N` fail to protect against a forecast error larger than any in the training set, while a Wasserstein ball does not?
2. In the example with unbounded support, Wasserstein DRO did not change the reserve decision. What feature of the loss caused this, and what changed when the support was bounded?
3. Theorem 18 says the certified radius scales like `N^(−p/m)`. For `p = 1` and errors at `m = 20` wind sites, how much more data halves the radius, and what does that suggest about how `ε` is chosen in practice?

<details><summary>Answers</summary>

1. `KL(Q‖P̂_N) < ∞` requires `Q` to be absolutely continuous with respect to `P̂_N`, so `Q` can only reweight the N observed points. The worst case is then bounded by the worst observed sample. A Wasserstein ball can move mass to new points at a cost proportional to distance, so it includes larger, unseen errors.
2. With `Ξ = R` and a convex loss whose slope in ξ is 100 for every `r`, Theorem 10 gives worst case = empirical + `ε · 100`. That is a constant shift that does not depend on `r`, so the minimiser is unchanged. With `ξ ≤ 40` the adversary must first transport mass up to `r` (wasted budget) and is capped at 40, so a larger `r` really does reduce the worst case. The trade-off produced `r = 38`.
3. `ε ∝ N^(−1/20)`, so halving ε needs `2^20 ≈ 10^6` times more data. Certified radii are hopeless at that dimension. Practitioners therefore choose `ε` by cross-validation or exploit structure (low-dimensional factors, elliptical nominals as in Section 2.3), and give up the clean finite-sample certificate.

</details>
