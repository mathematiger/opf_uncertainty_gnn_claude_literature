## 2026-10-31 (Saturday) — X017 · Track 11: Conformal prediction, risk control, selective prediction — Conformal intervals without a held-out split: the jackknife+ and CV+

**Type:** Cross-phase preview
**Main reference:** [Predictive inference with the jackknife+ — R. Foygel Barber, E. J. Candès, A. Ramdas & R. J. Tibshirani (2021), Annals of Statistics 49(1):486–507](https://arxiv.org/abs/1905.02928) — read Sections 1–4: the setup, the jackknife+ interval, Theorem 1 (`1 − 2α`) and Theorem 2 (sharpness), jackknife-minmax, CV+ and the summary table of guarantees and costs. Skim Section 5 for the stability results and look at the d ≈ n plot in Section 7.1. Section 6, the proof, is short and worth one careful read. The link goes to the open arXiv version (v3, 2020). The published version is open access on [Project Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-1/Predictive-inference-with-the-jackknife/10.1214/20-AOS1965.pdf).
**Further reading:**
- [Predictive Inference Is Free with the Jackknife+-after-Bootstrap — B. Kim, C. Xu & R. Foygel Barber (2020), NeurIPS 2020](https://arxiv.org/abs/2002.09025) — if you already train a bootstrap ensemble, you can get jackknife+-style intervals with no extra fitting, again with a worst-case `1 − 2α` guarantee.
- [Cross-conformal predictors — V. Vovk (2012), arXiv:1208.0806](https://arxiv.org/abs/1208.0806) — the original K-fold hybrid of inductive conformal prediction and cross-validation. CV+ is a close relative that comes with a proven guarantee.

**Why this matters** — Split conformal is simple, but it throws away data. Every point you hold out for calibration is a point the model never trains on. Anyone who trains an expensive regression model on a few thousand solved optimization instances, such as an OPF proxy whose labels each cost a solver run, has to choose between a better model and a valid interval. The jackknife+ and CV+ remove most of that tradeoff. They also show precisely what the price is: a factor of 2 on α in the worst case, and none in practice when the learner is stable.

**Core ideas**

**1. Setting.** You have exchangeable pairs `(X_1, Y_1), …, (X_n, Y_n)` and a test point `X_{n+1}`. You want an interval `Ĉ(X_{n+1})` with `P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≥ 1 − α`. This is the regression analogue of the coverage guarantee you know from classification. The only requirement on the fitting algorithm `A` is symmetry: its output does not depend on the order of the training points. Randomised training, such as SGD with random shuffling, can be treated as symmetric in distribution, and Appendix A covers algorithms that are genuinely asymmetric.

**2. Three ways to reuse the data.**
- *Naive:* fit `μ̂` on all `n` points and use the quantile of the training residuals `|Y_i − μ̂(X_i)|`. These residuals are in-sample, so they are too small when the model overfits. There is no guarantee.
- *Split conformal:* fit on `n_1` points and compute residuals on the other `n_2`. This gives an exact `1 − α` guarantee, but the model only sees `n_1` points.
- *Jackknife (leave-one-out):* fit `μ̂_{−i}` without point `i`, take the out-of-sample residual `R_i = |Y_i − μ̂_{−i}(X_i)|`, and centre the interval on the full fit:

```
Ĉ_jack(x) = [ μ̂(x) − q̂⁺{R_i},  μ̂(x) + q̂⁺{R_i} ]
```

Here `q̂⁺` is the `⌈(1−α)(n+1)⌉`-th smallest value. This looks right, but it mixes two different models. The residuals come from the `n` leave-one-out fits, while the centre comes from the full fit `μ̂`, which is a different function. When the algorithm is unstable, `μ̂` can be far from every `μ̂_{−i}`. Theorem 2 constructs cases where the jackknife's coverage is zero. In Section 7.1, least squares with dimension `d` close to `n = 100` is the realistic version, and the jackknife covers only about half the time at a target of 90%.

**3. The jackknife+.** The fix is to also use the leave-one-out models at the test point:

```
Ĉ_jack+(x) = [ q̂⁻{ μ̂_{−i}(x) − R_i },   q̂⁺{ μ̂_{−i}(x) + R_i } ]
```

Here `q̂⁻` is the `⌊α(n+1)⌋`-th smallest value. Each training point votes for an interval `μ̂_{−i}(x) ± R_i` around its own leave-one-out prediction. The jackknife+ takes the lower and upper quantiles of the endpoints. The spread of `μ̂_{−i}(x)` across `i` measures how much the fitted function moves at `x`, which is a crude measure of variance from the model fit. The interval widens when that spread is large.

**4. The guarantee and why it is `1 − 2α`.** Theorem 1 says that for any distribution and any symmetric algorithm,

```
P( Y_{n+1} ∈ Ĉ_jack+(X_{n+1}) ) ≥ 1 − 2α
```

The proof (Section 6) is a tournament argument. Imagine `n+1` points, including the test point, and fit all `(n+1)n/2` leave-two-out models `μ̃_{−(i,j)}`. Point `i` "beats" point `j` if `i`'s residual under the model that left out both is larger. By exchangeability, the test point is equally likely to be any of the `n+1`. A counting lemma shows that at most `2α(n+1)` points can be "strange", meaning they beat at least `(1−α)(n+1)` others. If the test point falls outside `Ĉ_jack+`, it must be strange. Theorem 2 shows the factor of 2 cannot be removed in general. The paper's practical message is different: "In practice, we generally expect to achieve the target level 1−α with either version of the jackknife." Section 5 makes this precise. If the algorithm is out-of-sample stable, with `|μ̂(x) − μ̂_{−i}(x)| ≤ ε` except with probability `ν`, a slightly inflated jackknife+ has coverage `≥ 1 − α − 4√ν`.

A variant, *jackknife-minmax*, uses the min and max of the `μ̂_{−i}(x)`. It has coverage `≥ 1 − α` (Theorem 3), but the paper notes it is generally too conservative.

**5. CV+.** `n` leave-one-out fits are unaffordable for a neural network. With `K` folds, CV+ uses the fold models `μ̂_{−S_k(i)}` in place of `μ̂_{−i}` and keeps the same formula. Theorem 4 gives

```
coverage ≥ 1 − 2α − √(2/n)     for any K
```

so for `K = 5` or `10` you pay `K` fits for almost the jackknife+ guarantee. Section 3.2 relates CV+ to Vovk's cross-conformal predictor. Section 4 tabulates the tradeoffs: full conformal is exact but needs a refit for every candidate `y`, split conformal needs one fit, CV+ needs `K` fits and the jackknife+ needs `n` fits.

**6. What the experiments show.** On simulated least squares and three real regression datasets (Communities and Crime, BlogFeedback, MEPS) with ridge, random forests and neural networks at `n = 200`, the naive method sometimes undercovers and jackknife-minmax overcovers. The jackknife+ and CV+ sit near the 90% target. Split conformal is valid but sometimes clearly wider, because its model was trained on half the data.

**A small example**

Take four training responses `Y = 1, 2, 3, 10` and the simplest symmetric algorithm, "predict the training mean". This is a constant model, so `μ̂_{−i}(x)` does not depend on `x`. Use `α = 0.4`, so `(n+1) = 5`, `q̂⁺` is the 3rd smallest value and `q̂⁻` is the 2nd smallest value.

```
left out   μ̂_{−i}   R_i = |Y_i − μ̂_{−i}|   μ̂_{−i} − R_i   μ̂_{−i} + R_i
   1        5.00        4.00               1.00           9.00
   2        4.67        2.67               2.00           7.33
   3        4.33        1.33               3.00           5.67
  10        2.00        8.00              −6.00          10.00
```

- **Jackknife+:** lower endpoints sorted are `−6, 1, 2, 3`, and the 2nd smallest is `1`. Upper endpoints sorted are `5.67, 7.33, 9, 10`, and the 3rd smallest is `9`. The interval is `[1, 9]`.
- **Jackknife:** the full mean is `4`, and the 3rd smallest residual is `4`. The interval is `[0, 8]`.

The jackknife+ interval is not centred on the full fit. It shifts upward because three of the four leave-one-out models, the ones that kept the outlier `10`, predict about 4.3–5. The leave-one-out predictions disagree, and the interval reflects it. With a stable learner and large `n`, the two intervals would nearly coincide.

**How it connects**

**X008** (conformal decision theory) calibrated a decision online with one scalar. This entry stays in the batch, exchangeable world and asks how to calibrate without wasting training data. **X015** (mixture density networks) noted that conditional densities can yield conformal sets that are unions of intervals. The jackknife+ always returns a single interval around point predictors, and phase F's **F02** (conformalized quantile regression) makes it adapt to the input. The data-cost motivation goes back to the learned OPF proxies in **X003** and **X012**: each training label is an AC-OPF solve of the kind written out in **A10**, so a 20% calibration split is a real loss, while 5-fold CV+ costs five trainings. Within phase F, **F01** develops split conformal for regression (Lei et al.). **F05** drops exchangeability, which matters because grid data come in time order. **F06** shows why none of these marginal guarantees are conditional on `x`. Tomorrow is another cross-phase day. The next focus lesson is **A13** (Lagrangian, KKT conditions and duality, seen through OPF).

**Check yourself**

1. Why can the plain jackknife reach zero coverage even though every residual `R_i` it uses is out-of-sample?
2. You train a GNN regressor on 5,000 solved instances. List the number of model fits and the worst-case coverage guarantee at `α = 0.1` for split conformal, the jackknife+ and CV+ with `K = 5`.
3. In the toy example, what happens to the jackknife+ interval at `α = 0.2`, and why does the worst-case guarantee then say almost nothing?

<details><summary>Answers</summary>

1. The residuals measure the errors of the leave-one-out models `μ̂_{−i}`, but the interval is centred on the full-data model `μ̂`, which none of those residuals describes. If adding one point changes the fit a lot, as least squares does near `d ≈ n`, then `μ̂(X_{n+1})` can be far from where the leave-one-out models would put it. A width calibrated for those models then misses. The jackknife+ avoids this by evaluating the same models `μ̂_{−i}` both on their residuals and at the test point.

2. Split conformal: 1 fit, coverage `≥ 0.9`, exactly, but the model sees only the training part. Jackknife+: 5,000 fits, coverage `≥ 1 − 2α = 0.8` in the worst case and about 0.9 if training is stable. CV+ with `K = 5`: 5 fits, coverage `≥ 1 − 2α − √(2/n) = 0.8 − √(0.0004) = 0.78` in the worst case, and typically about 0.9. (Theorem 4's sharper term `min{2(1−1/K)/(n/K+1), (1−K/n)/(K+1)}` is about 0.0016 here.)

3. With `n + 1 = 5`, `⌈0.8·5⌉ = 4` and `⌊0.2·5⌋ = 1`, so the interval runs from the minimum lower endpoint to the maximum upper endpoint, `[−6, 10]`. The guarantee is `≥ 1 − 2·0.2 = 0.6`. With four points the quantiles are extreme order statistics, so the interval is wide and the bound is loose. Below `α = 1/(n+1)` the upper quantile is not even defined (it is `+∞`). Small `n` and small `α` do not mix in any conformal method.

</details>
