## 2026-11-14 (Saturday) — X024 · Track 09: Aleatoric/epistemic UQ and evaluating uncertainty signals — One forward pass, two uncertainties? Deep evidential regression and why it is a heuristic

**Type:** Cross-phase preview
**Main reference:** [Deep Evidential Regression — A. Amini, W. Schwarting, A. Soleimany & D. Rus (2020), Advances in Neural Information Processing Systems 33 (NeurIPS 2020)](https://arxiv.org/abs/1910.02600). Read §1, §2 (the Gaussian NLL baseline, eq. 2) and all of §3 (eqs. 3–10: the Normal-Inverse-Gamma prior, the three moments, the Student-t evidence loss and the evidence regularizer). Read §4.1 (the cubic toy problem, Fig. 3, and the UCI table). Skim §4.2–4.3 (depth estimation, out-of-distribution and adversarial tests).
**Further reading:**
- [The Unreasonable Effectiveness of Deep Evidential Regression — N. Meinert, J. Gawlikowski & A. Lavin (2023), AAAI 37(8):9134–9142](https://arxiv.org/abs/2205.10060): the critique. It shows that the loss does not identify the four outputs, and that the "epistemic" output mostly records how fast each training point converged. Read §1.2 and §2.
- [Pitfalls of Epistemic Uncertainty Quantification through Loss Minimisation — V. Bengs, E. Hüllermeier & W. Waegeman (2022), NeurIPS 2022](https://arxiv.org/abs/2203.06102): the general version of the argument for classification. A network that outputs a distribution over distributions and is trained by minimising an average of ordinary losses is not pushed to represent epistemic uncertainty faithfully.

**Why this matters** — A learned OPF proxy or forecaster is attractive because it is fast. An uncertainty signal that needs five ensemble members or fifty dropout passes eats into that speed. Deep evidential regression promises both aleatoric and epistemic uncertainty from a single deterministic forward pass, with no out-of-distribution data. It is widely used, and its failure modes teach exactly what phase D will ask of any uncertainty signal: what does it estimate, is that quantity identified by the training loss, and how would you check it?

**Core ideas**

**1. From one Gaussian to a distribution over Gaussians.** A heteroscedastic head outputs `(μ(x), σ²(x))` and minimises the Gaussian NLL `½ log(2πσ²) + (y − μ)²/(2σ²)` (eq. 2). This models noise in `y` (aleatoric), but the model says nothing about whether its own `μ(x)` is trustworthy. Evidential regression goes one level up. It treats `(μ, σ²)` as unknown and has the network output the parameters `m = (γ, ν, α, β)` of their conjugate prior, the Normal-Inverse-Gamma (NIG) distribution (eqs. 3–4):

```
y ~ N(μ, σ²),    μ | σ² ~ N(γ, σ²/ν),    σ² ~ InvGamma(α, β)
constraints: γ ∈ ℝ, ν > 0, α > 1, β > 0   (softplus on ν, α, β; +1 on α)
```

The network therefore has four outputs per target. In conjugate Bayesian statistics, `ν` and `α` count "virtual observations": the mean behaves as if estimated from `ν` samples, and the variance from about `α` samples. The paper calls `Φ = 2ν + α` the total evidence. This is the regression version of evidential classification, where a network outputs Dirichlet concentration parameters instead of softmax probabilities.

**2. The three numbers you read off (eq. 5).**

```
prediction    E[μ]  = γ
aleatoric     E[σ²] = β / (α − 1)
epistemic     Var[μ] = β / (ν (α − 1)) = E[σ²] / ν
```

Integrating `μ` and `σ²` out gives a Student-t predictive density (eq. 7):

```
p(y | m) = St(y ; γ, β(1 + ν)/(ν α), 2α)
```

Its variance is `β(1+ν)/(ν(α−1)) = E[σ²] + Var[μ]`. So the split is again the law of total variance, as in the MDN of X015, but now the conditioning variable is the unknown likelihood parameter, not a mixture label.

**3. The loss.** The fit term is the negative log of that Student-t (eq. 8). On its own it is overconfident. The paper adds an evidence regularizer (eq. 9) and trades the two off with `λ` (eq. 10):

```
L_i = −log St(y_i ; γ, β(1+ν)/(να), 2α)  +  λ · |y_i − γ| · (2ν + α)
```

The regularizer reduces evidence where the prediction is wrong. Small errors permit large evidence; large errors force it down. With `λ = 0` the model is overconfident, and with `λ` too large it inflates uncertainty everywhere. In practice `λ` must be tuned.

**4. What the paper reports.** On `y = x³ + ε`, trained on `[−4, 4]` and tested on `[−6, 6]`, the regularized model's epistemic band widens outside the training range. The unregularized model's band does not (Fig. 3). On nine UCI regression benchmarks it matches dropout and 5-member ensembles in RMSE, has the best NLL in the table, and needs one forward pass. On NYU Depth v2 with a U-Net, the epistemic map tracks the depth errors and rises on out-of-distribution scenes and adversarially perturbed inputs.

**5. The critique (Meinert et al., 2023).** Three points matter.

- *The loss does not identify the parameters.* The Student-t term depends on `m` only through the location `γ`, the width `w_St = √(β(1+ν)/(να))` and the degrees of freedom `2α`. You can move `ν` and compensate with `β` without changing the likelihood. So the data do not determine `ν`, and hence do not determine the aleatoric/epistemic split. Only the regularizer acts on `ν`, and it always pushes `ν` down, more strongly where the residual is large.
- *"Epistemic" becomes a convergence-speed proxy.* Points with large residuals early in training get their `ν` pushed down faster. The final `1/ν` records how hard each region was to fit during optimization, not a posterior variance. The authors' verdict is that DER should be treated as "a heuristic rather than an exact uncertainty quantification."
- *Aleatoric leaks into epistemic.* Because the penalty uses the raw residual `|y − γ|`, noisy regions also get low evidence. They propose normalizing the residual by `w_St`, and reading `w_St` and `1/√ν` as the two proxies.

Bengs, Hüllermeier and Waegeman make the general point. If the training signal is an average of ordinary losses over single labels, nothing forces a second-order output (Dirichlet or NIG) to match epistemic uncertainty. One label per input simply contains no information about how much the model's belief should spread.

**Why it still works, sometimes.** The signal correlates with error, costs nothing at inference, and needs no OOD data. If its only use is to trigger a fallback (call the solver, flag for review), a well-ranked heuristic can be useful. Its quality must then be checked with ranking metrics such as AUROC for large errors and risk–coverage curves (D15), and its numbers must not be read as calibrated variances.

**A small example**

Suppose a proxy predicts the loading of a transmission line, in percent of its rating, and the evidential head outputs `γ = 80`, `ν = 4`, `α = 3`, `β = 200`.

- Aleatoric: `E[σ²] = 200/2 = 100`, so a noise standard deviation of 10 percentage points.
- Epistemic: `Var[μ] = 100/4 = 25`, a standard deviation of 5.
- Predictive: Student-t with `2α = 6` degrees of freedom and squared scale `β(1+ν)/(να) = 200·5/12 = 83.3`. Its variance is `83.3 · 6/4 = 125 = 100 + 25`. ✓

Now change only `ν` to 1 and set `β = 125`. The squared scale is `125·2/3 = 83.3`, the same as before, and `α` is unchanged. So the predictive density, and hence the NLL on every training point, is identical. But now `E[σ²] = 62.5` and `Var[μ] = 62.5`: the "epistemic" share has gone from 20 % to 50 % of the same total variance of 125. The data cannot tell these two models apart. Which one you get is decided by the regularizer, by `λ`, and by the optimization path.

What is identified is the predictive t-distribution. A one-sided 95 % upper bound from it is `80 + 1.943·√83.3 ≈ 97.7 %` of the rating (1.943 is the 95 % quantile of a t with 6 degrees of freedom). Whether that bound actually holds 95 % of the time is an empirical question for regression calibration (D13) or conformal correction (phase F).

**How it connects**

**X006** separated sources of uncertainty conceptually. **X015** showed that the law of total variance can split a predictive variance in ways that do not mean "epistemic". This entry is a third case: a split that the training loss does not even pin down. In phase D, **D07** defines the decomposition, **D08** covers the heteroscedastic Gaussian head (the `m = (μ, σ²)` baseline here) and its pitfalls, **D09–D10** cover ensembles and Bayesian approximations (the sampling-based baselines this method tries to replace), and **D11** asks whether decompositions hold up empirically, which is the question Meinert et al. answer negatively for DER. **D15** supplies the ranking metrics a heuristic signal should be judged by. In phase F, conformal methods turn any such score into intervals with a guarantee. Next focus lesson: **A17** (modelling stacks: PowerModels.jl, MATPOWER, pandapower OPF).

**Check yourself**

1. Why does a heteroscedastic Gaussian head trained with NLL give aleatoric but not epistemic uncertainty, even with unlimited model capacity?
2. In the example, why can two parameter vectors with different `ν` have exactly the same training loss without the regularizer, and what then decides the split?
3. You want to use the evidential `1/ν` to decide when a learned dispatch should be sent back to a solver. Which evaluation would you trust, and which would you not?

<details><summary>Answers</summary>

1. Its loss is a proper scoring rule for `p(y|x)`, so at the optimum `σ²(x)` matches the noise variance of `y` given `x`. Nothing in the loss refers to how uncertain the fitted `μ(x)` is. Far from the data the head extrapolates `σ²(x)` like any other output, often confidently.
2. The Student-t evidence depends on `(γ, ν, α, β)` only through `γ`, `β(1+ν)/(να)` and `α`. Lowering `ν` and lowering `β` so the scale stays fixed leaves the likelihood unchanged. The split is then set by the regularizer `λ|y − γ|(2ν + α)`, by `λ` and by the training dynamics, which is why Meinert et al. call `1/ν` a convergence-speed proxy.
3. Trust ranking-based checks on held-out and shifted data: AUROC for detecting large errors or constraint violations, and risk–coverage curves of "accept only when `1/ν` is below a threshold", with the threshold chosen on calibration data (phase F). Do not trust reading `Var[μ]` as a calibrated posterior variance, or checking only that it grows on a hand-picked OOD set.

</details>
