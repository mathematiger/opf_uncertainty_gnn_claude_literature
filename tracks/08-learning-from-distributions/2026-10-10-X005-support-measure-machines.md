## 2026-10-10 (Saturday) — X005 · Track 08: Uncertainty as input: distributions, sets, embeddings — When the input is a distribution: support measure machines

**Type:** Cross-phase preview
**Main reference:** [Learning from Distributions via Support Measure Machines — K. Muandet, K. Fukumizu, F. Dinuzzo & B. Schölkopf (2012), Advances in Neural Information Processing Systems 25 (NeurIPS 2012)](https://arxiv.org/abs/1202.6504). Read Sections 1–4 (about 4 pages: the mean map, the representer theorem over distributions, expected kernels with Table 1, the risk-deviation bound and the Flex-SVM equivalence). Skim Section 6 (experiments). The link goes to the open arXiv version (v2, 2013).
**Further reading:**
- [Distribution-Free Distribution Regression — B. Póczos, A. Singh, A. Rinaldo & L. Wasserman (2013), AISTATS, PMLR 31:507–515](https://proceedings.mlr.press/v31/poczos13a.html): the regression counterpart. Each input distribution is seen only through samples, and the paper gives risk rates for kernel-smoothing estimators on distributions. Read it for the "two-stage sampling" picture, which comes back in D02.

**Why this matters** — A probabilistic forecast (**X002**) is a distribution, not a number. Anyone building learned OPF solvers under forecast uncertainty has to decide how a network *reads* such an object: feed it the mean, a few quantiles, samples, or something richer. This paper is the cleanest early statement of the principled option: treat each input as a probability measure, map it to a point in a Hilbert space without losing information, and learn there. Phase D, unit 14, builds on this view.

**Core ideas**

**The setting.** Ordinary supervised learning sees pairs `(x_i, y_i)` with `x_i ∈ X`. Here the training set is `{(P_i, y_i)}`, where each `P_i` is a probability distribution on `X`. The goal is a function `h: P ↦ y`. The paper motivates this with noisy, replicated measurements and with large datasets compressed into groups. In a power-system setting the analogue is a forecast ensemble or a predictive density for next-hour renewable output.

**The mean embedding.** Choose a kernel `k` on `X` with RKHS `H`, and map each distribution to its mean feature:

```
μ_P = E_{x~P}[ k(x, ·) ] ∈ H,        E_P[f] = ⟨μ_P, f⟩_H  for all f ∈ H
```

For a *characteristic* kernel (Gaussian RBF, for example) the map `P ↦ μ_P` is injective, so no information about `P` is lost. This is the kernel version of "all moments at once": a linear kernel would keep only the mean, a degree-2 polynomial kernel the first two moments, and an RBF kernel keeps everything. The inner product of two embeddings is the *expected kernel*

```
K(P, Q) = ⟨μ_P, μ_Q⟩_H = E_{x~P, z~Q}[ k(x, z) ]
```

This is a valid p.d. kernel on distributions, so any kernel method (SVM, kernel ridge, GP) now runs on distributions. The paper names `SVM + K` the *support measure machine* (SMM). The distance `‖μ_P − μ_Q‖_H` is the MMD you may know from two-sample testing.

**Three results worth remembering.**

1. *Representer theorem (Thm. 1).* If the loss depends on the training distributions only through `E_{P_i}[f]`, the regularized minimizer is `f = Σ_i α_i μ_{P_i}`. With Dirac inputs `P_i = δ_{x_i}` this reduces to the usual representer theorem, so SVMs are a special case of SMMs.
2. *What objective you are actually optimizing.* An SMM minimizes `ℓ(y, E_P[f(x)])`. That is neither "train on the means" (`ℓ(y, f(E_P[x]))`) nor "train on infinitely many samples" (`E_P[ℓ(y, f(x))]`). It sits between the two. Theorem 3 bounds the gap to the sample-based risk:

```
| E_P[ ℓ(y, f(x)) ] − ℓ(y, E_P[f(x)]) | ≤ 2 C_ℓ C_f σ
```

Here `C_ℓ, C_f` are Lipschitz constants and `σ` is the standard deviation of `P`. When inputs are sharp the two objectives agree; when they are wide, the choice matters. This is a Jensen-type gap, and the same kind of gap separates "the cost of the expected scenario" from "the expected cost" in stochastic optimization (phase B).
3. *Flex-SVM (Lemma 4).* If the distributions differ only by location (`P_i` has density `g(x_i, ·)`), a linear SMM equals an ordinary SVM whose kernel is *different at each data point*. For isotropic Gaussians `N(x_i, σ_i² I)` and a Gaussian RBF kernel, point `x_i` gets an RBF bump that is wider by `2σ_i²`. Uncertain points spread their influence; certain points stay sharp.

**Closed forms.** For Gaussian inputs `P_i = N(m_i, Σ_i)` and an RBF kernel `exp(−γ/2 ‖x − z‖²)`, Table 1 gives

```
K(P_i, P_j) = exp( −½ (m_i − m_j)ᵀ (Σ_i + Σ_j + γ⁻¹ I)⁻¹ (m_i − m_j) ) / |γΣ_i + γΣ_j + I|^(1/2)
```

Otherwise use the empirical estimate `K̂ = (1/(nm)) Σ_a Σ_b k(x_a, z_b)` from samples, with `O(m^(−1/2))` error. On top of `K` you can put a second, nonlinear "level-2" kernel such as `exp(−‖μ_P − μ_Q‖²_H / 2)`, which gives a nonlinear predictor on distributions.

**Findings.** On a synthetic 7-Gaussian problem, an SVM trained on the means alone overweights dense regions. An "augmented SVM" (ASVM) trained on 30 samples per distribution is costlier and sensitive to heavy tails. The SMM uses the covariances directly. On 10-D synthetic Gaussians, the choice of embedding kernel mattered more than the level-2 kernel (best: RBF–RBF, 89.65 ± 1.37 %). On USPS digit pairs with transformation-invariance priors, the SMM mostly beat SVM and ASVM as the number of virtual samples grew, and ran much faster than ASVM.

**A small example**

Two hourly wind forecasts for a site, in units of 10 MW. Both have mean 5 (50 MW). Forecast `P = N(5, 0.5²)` is confident; forecast `Q = N(5, 2²)` is uncertain. Use a 1-D RBF kernel with length-scale `ℓ = 1`. For 1-D Gaussians the expected kernel is

```
K(N(a, s²), N(b, t²)) = ℓ / sqrt(ℓ² + s² + t²) · exp( −(a − b)² / (2(ℓ² + s² + t²)) )
```

With equal means, the exponential factor is 1:

```
K(P,P) = 1/sqrt(1.5)  = 0.816
K(Q,Q) = 1/sqrt(9)    = 0.333
K(P,Q) = 1/sqrt(5.25) = 0.436
MMD²(P,Q) = 0.816 + 0.333 − 2·0.436 = 0.277
```

A model fed only the point forecast sees distance 0 and must output the same decision for both hours. The embedding separates them, so a downstream predictor of, say, the reserve needed or the redispatch cost *can* react to the wider forecast. Theorem 3 says why it should: with `σ = 2` instead of `0.5`, the gap between "loss at the expected feature" and "expected loss" can be four times larger.

**How it connects**

**X002** argued that a wind forecast should be a distribution. This entry answers the next question: how a learner can take that distribution as input. **X001**'s policy classes treat the forecast as part of the *state* a policy conditions on, and a mean embedding is one way to represent that state. The phase B lessons on stochastic and chance-constrained optimization will make the Jensen gap above concrete for dispatch costs. In phase D, **D01** contrasts conditioning on input uncertainty with propagating it. **D02** covers kernel mean embeddings and distribution regression properly, with the two-stage sampling theory from the further-reading paper. **D03–D05** replace the fixed kernel feature `k(x, ·)` with a *learned* one: Deep Sets is essentially `ρ(Σ_a φ(x_a))`, which is a learned empirical mean embedding. **D06** asks when the target itself must depend on the whole input distribution. For GNN readers (**X004**): a per-bus mean embedding is a natural node feature when every load and renewable injection carries its own forecast distribution. Next focus lesson: **A04**.

**Check yourself**

1. Why does the choice of embedding kernel decide whether an SMM can tell `N(0, 1)` from a Laplace distribution with mean 0 and variance 1? Which of linear, degree-2 polynomial and Gaussian RBF can?
2. An SMM minimizes `ℓ(y, E_P[f(x)])`. Write the other two natural objectives and say when all three coincide.
3. In the Flex-SVM view, what happens to a training point's kernel bump as its input variance `σ_i²` grows, and why is that sensible behaviour?

<details><summary>Answers</summary>

1. The embedding keeps only what the kernel's features can see. A linear kernel embeds only the mean, and a degree-2 polynomial kernel only the mean and second moments, so both map these two distributions (equal mean and variance) to the same point. The Gaussian RBF kernel is characteristic, so its embedding is injective and distinguishes them, for example through their different fourth moments.
2. "Train on the mean input": `ℓ(y, f(E_P[x]))`. "Train on the full distribution of inputs": `E_P[ℓ(y, f(x))]`. All three coincide when `P` is a point mass (zero variance). The SMM objective also equals the third when `ℓ` is linear in its second argument, and equals the first when `f` is linear. In general they differ by Jensen-type gaps, bounded by `2 C_ℓ C_f σ` (Theorem 3) for the SMM versus sample-based risk.
3. Convolving Gaussians adds variances: the kernel between points `i` and `j` gets bandwidth `σ² + σ_i² + σ_j²`, which is `σ² + 2σ_i²` when both share the same input variance. The bump gets wider and lower, so an uncertain point influences a larger region but less strongly at any one location. A precise measurement constrains the decision boundary locally, while a vague one should only nudge it broadly.

</details>
