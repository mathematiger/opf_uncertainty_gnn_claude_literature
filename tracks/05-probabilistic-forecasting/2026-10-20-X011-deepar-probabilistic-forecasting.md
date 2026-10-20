## 2026-10-20 (Tuesday) — X011 · Track 05: Probabilistic forecasting, scoring rules, scenario generation — Forecasting by sampling: DeepAR and the global autoregressive likelihood model

**Type:** Cross-phase preview
**Main reference:** [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks — D. Salinas, V. Flunkert & J. Gasthaus (2017 preprint; journal version 2020), International Journal of Forecasting 36(3):1181–1191](https://arxiv.org/abs/1704.04110) — read §1, all of §3 (model, likelihoods, training, scale handling, features) and §4.2 with Figures 4–5. Skim Table 1. The arXiv v3 is open access. Journal version: [DOI 10.1016/j.ijforecast.2019.07.001](https://doi.org/10.1016/j.ijforecast.2019.07.001).
**Further reading:**
- [High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes — D. Salinas, M. Bohlke-Schneider, L. Callot, R. Medico & J. Gasthaus (2019), NeurIPS](https://arxiv.org/abs/1910.03002) — the follow-up that makes the samples *joint across thousands of series*, which is what a grid with many loads and renewable plants needs.
- [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting (TimeGrad) — K. Rasul, C. Seward, I. Schuster & R. Vollgraf (2021), ICML](https://arxiv.org/abs/2101.12072) — keeps DeepAR's autoregressive skeleton but replaces the per-step parametric likelihood with a diffusion model.

**Why this matters** — Stochastic and chance-constrained OPF consume *samples of the future*: whole trajectories of load and renewable output, not one number per hour. DeepAR is the cleanest example of how a deep network turns into a scenario generator. It trains with a plain likelihood and produces joint sample paths by feeding its own samples back in. Anyone building learned OPF pipelines under forecast uncertainty will meet this design, its descendants, and its failure modes.

**Core ideas**

**1. The problem: many related series, one model.** Classical forecasting (ARIMA, exponential smoothing, state-space models) fits one model per series. DeepAR fits a single **global** model across a large set of related series. In the paper these are 500K Amazon product demand series. The public benchmarks include an `electricity` set of hourly consumption for 370 customers and a `traffic` set of lane occupancies. The ML analogy is the move from per-task models to one network with shared weights and a per-entity embedding. Series with short histories ("new products", or a newly connected feeder) borrow strength from the rest.

**2. The model: an autoregressive likelihood.** For series `i`, with conditioning range `1..t₀−1` and prediction range `t₀..T`, DeepAR factorizes

```
Q_Θ(z_{i,t₀:T} | z_{i,1:t₀−1}, x_{i,1:T}) = Π_{t=t₀}^{T} ℓ(z_{i,t} | θ(h_{i,t}, Θ))
h_{i,t} = h(h_{i,t−1}, z_{i,t−1}, x_{i,t}, Θ)            (LSTM)
```

The network sees the last observed value `z_{i,t−1}`, known covariates `x_{i,t}` (time-of-day, day-of-week, an "age" feature and an item embedding) and its own hidden state. It outputs the *parameters* of a fixed distribution `ℓ`. For real-valued data this is a Gaussian, with `μ = wᵀh + b` and `σ = softplus(wᵀh + b)`. For counts it is a negative binomial with mean `μ` and shape `α`, so that `Var[z] = μ + μ²α`. Training maximizes the log-likelihood `L = Σ_i Σ_t log ℓ(z_{i,t} | θ(h_{i,t}))` with teacher forcing. Because `h` is a deterministic function of observed inputs, there is no latent-variable inference, unlike classical state-space models. The authors also report that scheduled sampling did not help, so exposure bias was not a practical problem here.

**3. Prediction by ancestral sampling.** At forecast time, run the LSTM over the history, then for `t = t₀, …, T` draw `z̃_{i,t} ~ ℓ(·|θ(h̃_{i,t}))` and feed `z̃_{i,t}` back as the next input. One pass gives one **sample path**. Repeating it, say, 200 times gives an empirical joint distribution over the whole horizon. Any quantity can then be read off by Monte Carlo: the 90% quantile of energy over the next 6 hours, the probability that the peak exceeds a threshold, and so on. Because the paths are produced jointly, quantiles of *aggregates* are consistent, which you cannot get by stacking per-hour marginal quantiles.

**4. Scale handling.** Series magnitudes follow a roughly power-law distribution across items, which breaks standard normalization and batch norm. DeepAR divides the autoregressive inputs by a per-series scale `ν_i = 1 + (1/t₀) Σ_{t≤t₀} z_{i,t}` and multiplies the scale-dependent output parameters back (for the negative binomial, `μ = ν_i·softplus(o_μ)` and `α = softplus(o_α)/√ν_i`). It also samples training windows with probability proportional to `ν_i`, so the few large series are not underfit. Grid data has the same skew: a substation and a single household differ by orders of magnitude.

**5. What the experiments show.** Scores are quantile ("ρ-risk", i.e. normalized pinball) losses at ρ = 0.5 and 0.9 over different lead-time spans, relative to the strongest published baseline. In the arXiv version, DeepAR's average relative risk is 0.77 on `ec-sub`, 0.85 on `ec` and 0.94 on `parts`. On `electricity` it reaches a normalized deviation of 0.07 against 0.16 for the matrix-factorization baseline. Two ablations matter more than the headline numbers. First, an RNN with a Gaussian likelihood does clearly worse on count data, so **the likelihood family is a modelling decision**. Second, without scaling and weighted sampling the gains disappear on the power-law data sets. The calibration plot (Fig. 5) adds a key finding. If the samples are **shuffled across time steps**, which keeps every marginal but destroys temporal correlation, calibration over a 9-step span gets worse and the 0.9-risk goes up by about 10%. Joint structure is not cosmetic.

**6. Limits to keep in mind.** (a) Each series is sampled independently given its own history, so cross-series dependence (wind at neighbouring sites, load across a city) is only what shared covariates induce. The copula follow-up in Further reading fixes this. (b) Sampling cost grows with horizon × paths, and errors compound along a path. (c) The marginal family is hand-picked. Bounded quantities such as PV output in `[0, P_max]` need a beta, censored or logit-transformed likelihood. The paper mentions a beta option but does not test it. (d) Calibration is shown only visually. Proper scores and PIT checks come later in the curriculum.

**A small example**

Take a toy Gaussian autoregressive model in standardized units: `z_t = 0.8·z_{t−1} + ε_t`, with `ε_t ~ N(0, 1)` and `z_0 = 0`. Treat it as DeepAR with a trivially simple "network". You must reserve for the **two-hour total** `S = z_1 + z_2` at the 90% level.

- Marginals: `z_1 ~ N(0, 1)`, `z_2 = 0.8 z_1 + ε_2 ~ N(0, 1.64)`, so the standard deviations are 1 and 1.281.
- Joint (ancestral samples): `S = 1.8 z_1 + ε_2`, `Var = 3.24 + 1 = 4.24`, `sd = 2.059`, so `q₀.₉(S) = 1.2816 × 2.059 ≈ 2.64`.
- Shuffled samples (correct marginals, independence): `Var = 1 + 1.64 = 2.64`, `sd = 1.625`, so `q₀.₉ ≈ 2.08`, an **under-reserve by about 21%**.
- Summing marginal quantiles: `1.2816 × (1 + 1.281) ≈ 2.92`, which over-reserves by about 11%. This is the answer under perfect positive dependence, the worst case.

The right number sits between "independent" and "comonotone", and only the joint sample paths give it. This is a tiny version of the paper's shuffling experiment, and the same reasoning applies to a ramping or storage constraint that couples hours.

**How it connects**

**X002** argued that wind forecasts must be distributions and, for coupled decisions, *trajectories*. It sketched the classical route: marginal quantiles plus a Gaussian copula. DeepAR is the deep-learning route to the same object. It gives trajectories directly by sampling and learns the temporal dependence instead of estimating a copula covariance. **X005** fed a distribution *into* a model. Here the model *outputs* one, as samples. These samples are exactly what a two-stage or chance-constrained OPF consumes (**X001**, **X010**). In phase B, **B13** treats forecasts as distributions, **B14** covers the pinball loss behind the ρ-risk used here, **B15–B17** cover calibration, proper scores and the energy and variogram scores that would detect the shuffling problem quantitatively, **B19** covers copula scenarios and **B21** covers deep generative scenario models (GANs and flows), the direct successors of this design. The spatio-temporal GNNs of **C19–C21** add the graph to the time axis. Next focus lesson: **A08**, economic dispatch and the Lagrange multiplier as a marginal price.

**Check yourself**

1. Why can DeepAR be trained by plain maximum likelihood with no inference step, unlike a linear-Gaussian state-space model with latent states?
2. A colleague trains DeepAR, then for each hour reports the 5% and 95% quantiles from the sample paths and builds a "90% band" for daily energy by summing the hourly 95% quantiles. What is wrong with this, and how should it be done?
3. You forecast PV output for 1,000 rooftop systems with one DeepAR model and a Gaussian likelihood. Name two problems you would expect and how to fix each.

<details><summary>Answers</summary>

1. The hidden state `h_{i,t}` is a deterministic function of observed inputs (past targets under teacher forcing, and covariates). So every likelihood term `ℓ(z_{i,t}|θ(h_{i,t}))` can be evaluated and differentiated directly, and the objective is a sum of exact log-likelihoods. In a state-space model the state is random and unobserved, so the likelihood requires marginalizing over it (Kalman filtering or approximate inference).
2. Summing marginal quantiles gives the quantile of the sum only under perfect positive dependence (comonotonicity), so it is generally too wide, as the toy example shows (2.92 against 2.64). Building a joint band from per-hour marginals also ignores the dependence structure entirely. The correct procedure is to sum each sample path over the day and take the empirical 5% and 95% quantiles of those sums. This is what the paper does for its span metrics.
3. (a) Scale and heterogeneity: system sizes differ, so normalize by a per-system scale such as `ν_i` (or capacity) and sample training windows non-uniformly. (b) Wrong support: a Gaussian puts mass below 0 and above capacity, and at night the output is exactly 0. Use a bounded likelihood (beta or censored, possibly with a point mass at zero) or a logit transform of the capacity factor. As a bonus, (c) nearby systems share clouds, so per-series independent sampling understates the risk of the aggregate. Use a joint (copula or multivariate) output model.

</details>
