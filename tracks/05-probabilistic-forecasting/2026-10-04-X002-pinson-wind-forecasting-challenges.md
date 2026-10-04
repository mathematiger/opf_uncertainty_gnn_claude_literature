## 2026-10-04 (Sunday) — X002 · Track 05: Probabilistic forecasting, scoring rules, scenario generation — Why a wind forecast should be a distribution: Pinson's forecasting challenges

**Type:** Cross-phase preview
**Main reference:** [Wind Energy: Forecasting Challenges for Its Operational Management — P. Pinson (2013), Statistical Science 28(4):564–585](https://arxiv.org/abs/1312.6471) — read Sections 2–4 (about 15 pages: wind power as a stochastic process, two decision problems, and point → density → trajectory forecasts). Skim Section 5 (open challenges). The [Project Euclid version](https://doi.org/10.1214/13-STS445) is the published one.
**Further reading:**
- [Probabilistic electric load forecasting: A tutorial review — T. Hong & S. Fan (2016), International Journal of Forecasting 32(3):914–938](https://doi.org/10.1016/j.ijforecast.2015.11.011) — the same story for load instead of wind: techniques, evaluation and common misunderstandings.
- [Quantiles as optimal point forecasts — T. Gneiting (2011), International Journal of Forecasting 27(2):197–207](https://doi.org/10.1016/j.ijforecast.2009.12.015) — the general result behind Pinson's market example: under asymmetric piecewise-linear losses the best single number is a quantile, not the mean.

**Why this matters** — Every OPF or dispatch problem under uncertainty starts from a forecast. The form of that forecast (a number, a set of quantiles, a density or a set of joint scenarios) decides which optimization problems you can even write down. Anyone building learned solvers that take forecast uncertainty as input needs a clear picture of what forecasters can actually deliver, what shape real renewable errors have, and which decision needs which forecast type. This paper gives that picture in a form a statistician or ML researcher can read without any power-systems background.

**Core ideas**

**Wind power is a bounded, nonlinear, nonstationary process.** A turbine converts wind speed into power through a *power curve*: zero below the cut-in speed (about 4 m/s for the old turbine in the paper), a steep cubic-like rise up to the rated speed (16 m/s there), flat at nominal power `P_n` up to the cut-off speed (25 m/s), then zero again when the turbine shuts down to protect itself. A farm's empirical curve is a noisy version of this. In the paper's example farm, a measured wind speed of 5 m/s goes with outputs anywhere between 0 and 7 MW of a 21 MW farm. Pushing even a Gaussian wind-speed error through this curve has three consequences:

1. Power normalized by capacity lives in `[0, 1]`. Gaussian predictive densities are wrong by construction, and probability mass piles up at 0 and 1 (Pinson suggests treating it as a discrete–continuous mixture, like precipitation).
2. Uncertainty is largest on the steep middle of the curve and smallest near the flat ends. Spread depends on the predicted level, so errors are heteroskedastic in a structured way.
3. Weather regimes switch predictability on and off, so the process is nonstationary (the paper cites GARCH-type models for this).

**Notation.** Write `Y_{s,t+k}` for normalized power at location `s ∈ {s_1..s_m}` and lead time `k ∈ {1..n}`, issued at time `t`. The full target is the `m × n`-dimensional random vector `Y_{s,t+k}`, and the ideal forecast is its joint predictive CDF `F̂_{s,t+k|t}` given the information `Ω_t` (local measurements plus numerical weather predictions). Since that is hard to produce and to verify, practice uses simpler summaries of it:

- **Point forecast:** the conditional mean `ŷ_{s,t+k|t} = ∫_0^1 y f̂_{s,t+k|t}(y) dy`, which is optimal for squared loss.
- **Quantiles / marginal densities:** one predictive distribution per location and lead time. These are either parametric (censored Gaussian, Beta, generalized logit-normal, with the parameters predicted from `Ω_t`) or nonparametric, as a set of quantile forecasts `{q̂^(α_i)}` produced by quantile regression or by "dressing" a point forecast with past errors observed in similar conditions.
- **Space–time trajectories (scenarios):** samples from the joint distribution, which keep the correlation across hours and sites.

**Which decision needs which forecast.** This is the paper's central argument, made with two decision problems.

*Market offer (the newsvendor).* A wind producer sells `y^c` MWh in the day-ahead market 13–37 hours before delivery and then pays for deviations in the balancing market. Surplus energy is sold back below the day-ahead price, at a unit cost `π↓`. A shortfall has to be bought above it, at a unit cost `π↑`. The expected-revenue-maximizing offer has a closed form (Bremnes, 2004):

```
y*_{t+k} = F̂⁻¹_{t+k|t}( π↓ / (π↓ + π↑) )
```

The optimal offer is a *quantile* of the predictive distribution, and its level changes from hour to hour with the predicted prices. A point forecast cannot serve this problem: you would need a different "point" for every price ratio.

*Reserve sizing (the operator's side).* The system operator must hold upward and downward reserve capacity before it knows the outcome. The total deviation `O_{t+k}` combines load forecast errors, unplanned outages and wind forecast errors. If these are independent, the densities convolve:

```
f̂^O = f̂^{ε_L} ∗ f̂^G ∗ f̂^{ε_Y}
```

The optimal reserve levels are again quantiles, this time of the *combined* density `f̂^O`. A wind quantile alone gives no direct answer. You need the whole wind density to convolve it.

*Anything with time or space coupling.* Ramp limits, storage, network congestion and risk aversion all need the *joint* distribution, so the input becomes scenarios for a stochastic program. Marginals alone cannot tell you whether a low-wind hour at site A comes together with a low-wind hour at site B, or with a low-wind next hour.

**From marginals to scenarios with a Gaussian copula.** Pinson's practical recipe for producing trajectories, which keeps the calibrated marginals and adds dependence:

```
z_{s,t+k} = Φ⁻¹( F̂_{s,t+k|t}(y_{s,t+k}) )              # map observations to a latent Gaussian
z^(j) ~ N(0, Ĉ_t)                                        # sample with a space–time covariance
ŷ^(j)_{s,t+k} = F̂⁻¹_{s,t+k|t}( Φ(z^(j)_{s,t+k}) )       # map back through the marginals
```

The covariance `Ĉ_t` is tracked over time, for example by exponential smoothing of past latent vectors. The recipe only works if the marginals are *probabilistically calibrated*, because otherwise the PIT values `F̂(y)` are not uniform and the latent variables are not Gaussian. An ML reader will recognize this as the probability-integral-transform trick used in normalizing flows and in copula-based multivariate calibration.

**Open challenges (Section 5).** (i) Better models that use the whole sensor network and weather-dependent, nonstationary covariance structures. (ii) Verification of high-dimensional probabilistic forecasts, where even proper scores become noisy with small, autocorrelated test sets. (iii) Closing the gap between forecast *quality* (scores) and forecast *value* (cost of the decisions made with it), following Murphy's distinction. Point (iii) is still open, and later phases of this curriculum come back to it from the ML side.

**A small example**

A 100 MW wind farm. The predictive distribution for tomorrow 14:00 puts output at 20, 40, 60 or 80 MW with probabilities 0.1, 0.3, 0.4 and 0.2. The mean is 54 MW. Balancing costs are `π↓ = 10 €/MWh` for surplus and `π↑ = 30 €/MWh` for shortfall, so the optimal level is `α = 10/40 = 0.25`. The CDF reaches 0.1 at 20 MW and 0.4 at 40 MW, so `q̂^(0.25) = 40 MW`.

Expected balancing cost `B(y) = π↓·E[(Y − y)⁺] + π↑·E[(y − Y)⁺]`:

```
offer 20:  10·34  + 30·0    = 340 €
offer 40:  10·16  + 30·2    = 220 €    ← the 0.25-quantile
offer 54:  10·7.6 + 30·7.6  = 304 €    ← the mean forecast
offer 60:  10·4   + 30·10   = 340 €
```

Offering the mean costs 38 % more than offering the right quantile, even though the mean is the "best" forecast under squared error. Now suppose the forecaster is overconfident and reports all mass on 50–60 MW. The reported 0.25-quantile then sits above 40 MW and the producer is short more often than planned. Poor *calibration* turns directly into lost money, which is why the paper calls calibration a prerequisite.

**How it connects**

**X001** ran the same newsvendor logic from the buyer's side (commit capacity, pay a premium for shortfall) and argued that the forecast belongs in the decision state `S_t`. This entry says what that forecast should look like: a quantile for single-period, piecewise-linear costs, a full density when uncertainties combine, and trajectories when decisions are coupled over time or space. In Powell's language, a quantile offer rule is a PFA that conditions on the forecast distribution. The phase B forecasting unit develops each piece in depth: **B13** (forecasts as distributions, error behaviour), **B14** (quantile regression and pinball loss, the training objective for the quantile forecasts used here), **B15–B16** (calibration, sharpness and proper scores), **B18** (logit-normal and other bounded distributions) and **B19** (the Gaussian-copula scenario method sketched above). The quality-versus-value gap returns in **C07–C08** (decision-focused learning). Next focus lesson: **A01**, why grids are hard to operate.

**Check yourself**

1. Why does a symmetric, Gaussian wind-speed forecast error turn into a non-Gaussian, heteroskedastic power forecast error, and where on the power curve is the power uncertainty largest?
2. In the market problem, how does the optimal offer level `α` change if shortfall becomes much more expensive than surplus? What happens in the limit `π↑ → ∞`?
3. Why does the Gaussian-copula scenario recipe require calibrated marginals, and what goes wrong in the generated scenarios if the marginals are too narrow?

<details><summary>Answers</summary>

1. Power is a bounded, S-shaped function of wind speed. On the steep middle part a small change in wind speed moves power a lot, so the error spreads out. Near cut-in (output close to 0) and above rated speed (output at `P_n`) the curve is flat, so the error is compressed and probability mass collects at the bounds. The spread therefore depends on the predicted level, and the distribution is skewed and truncated near 0 and 1. Uncertainty is largest at intermediate output levels.
2. `α = π↓/(π↓ + π↑)` falls toward 0, so the producer offers a lower quantile and accepts surplus to avoid costly shortfalls. As `π↑ → ∞`, `α → 0` and the offer goes to the lower end of the predictive support (offer only what is almost certain).
3. The recipe maps observations to a latent space with `Φ⁻¹(F̂(y))`. That latent variable is standard normal only if `F̂(y)` is uniform, i.e. the marginals are calibrated, and only then is estimating a Gaussian covariance `Ĉ_t` justified. With marginals that are too narrow, the PIT values pile up near 0 and 1, the latent variables have inflated variance, and the estimated dependence structure is distorted. Mapping back through the narrow marginals then produces scenarios that are too concentrated, so a stochastic program built on them underestimates the risk.

</details>
