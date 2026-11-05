## 2026-11-05 (Thursday) — X020 · Track 05: Probabilistic forecasting, scoring rules, scenario generation — A map of energy forecasting: Hong et al.'s review and outlook

**Type:** Cross-phase preview
**Main reference:** [Energy Forecasting: A Review and Outlook — T. Hong, P. Pinson, Y. Wang, R. Weron, D. Yang & H. Zareipour (2020), IEEE Open Access Journal of Power and Energy 7:376–388](https://doi.org/10.1109/OAJPE.2020.3029979) — read Section III (research frontiers: ML, combination, hierarchical and probabilistic forecasting), Section V-A and V-B3 (common issues; comparative studies) and Sections VI-B and VI-C (closed-loop forecasting; valuation of forecasts). Skim Section II for the list of subdomain reviews and Section IV-B for the six open data sources. The paper is open access (CC BY 4.0).
**Further reading:**
- [Electricity price forecasting: A review of the state-of-the-art with a look into the future — R. Weron (2014), International Journal of Forecasting 30(4):1030–1081](https://doi.org/10.1016/j.ijforecast.2014.08.008) — the standard reference for the price subdomain. It is also where the "same data, same error measures, statistical tests" argument was first made at length.
- [Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization — S. L. Wickramasuriya, G. Athanasopoulos & R. J. Hyndman (2019), JASA 114(526):804–819](https://doi.org/10.1080/01621459.2018.1448825) — the MinT method behind Section III-C: reconciliation as a generalized-least-squares projection with a closed form.

**Why this matters** — Every OPF under uncertainty starts from a forecast: of load, of wind and solar output, sometimes of prices. Before learning the formal tools of phase B, it helps to see what the forecasting community already knows about its own failure modes. That includes leaky evaluation, unsuitable metrics, incoherent forecasts across grid levels, and forecasts that change the outcome they predict. Anyone building learned OPF solvers under forecast uncertainty inherits these problems through the input distribution.

**Core ideas**

**1. Four subdomains, one field.** "Energy forecasting" here means forecasting in the electric power industry, with four subdomains: load (demand), electricity prices, wind power and solar power. Each subdomain has its own reviews (Section II-A is a reading list). Each also has its own physics. Load is driven by temperature, calendar and human behaviour. Wind power is wind speed passed through a nonlinear, bounded "power curve". Solar is irradiance, dominated by clouds, with a deterministic day/night envelope. Prices come out of a market that clears an optimization, so they inherit the jumps of the merit order and of congestion (recall the LMPs of **A11**). The review's thesis is that most methods travel across subdomains, so practitioners should read beyond their own. Its examples: copulas moved from wind to solar, and calendar features moved from load to wind.

**2. Forecast horizons are named after decisions.** The paper insists on precise vocabulary. A *day-ahead* forecast is issued at some time today for the 24 hours of tomorrow. So if it is issued at 12:00, its lead times run from 12 to 36 hours. A *24-hour-ahead* forecast has one fixed lead time. Solar forecasting uses *intra-hour*, *intra-day* and *day-ahead*, and these map to input sources: sky cameras, satellites and numerical weather prediction (NWP). The horizon matters because the error distribution depends strongly on the lead time, and each market or operating decision (day-ahead market, intraday trading, real-time balancing) uses a specific one.

**3. From point to probabilistic.** The paper calls the move from point to probabilistic forecasting "probably the most important step in the recent history of energy forecasting". Wind led it because wind forecasters worked closely with meteorologists. A probabilistic forecast can take several forms:

```
density  f_t+h(y | info_t)      parametric (truncated Gaussian, generalized logit-normal), or kernel-based
quantiles q_τ(t+h), τ ∈ {0.01,…,0.99}   quantile regression, minimizing the pinball loss
intervals [q_α/2, q_1−α/2]
ensembles / scenarios  {y^(s)_t+1:t+H}  joint paths, e.g. via a Gaussian copula on the marginals
```

The pinball loss for level `τ` is `ρ_τ(y, q) = (y − q)(τ − 1{y < q})`. Its population minimizer is the conditional `τ`-quantile, the same order statistic that split conformal reads off its calibration scores. The paper's guiding principle comes from Gneiting et al.: **maximize sharpness subject to calibration**. For a classification-calibration person, think "reliability first, then refinement". Marginal forecasts per site and lead time are not enough for OPF, which needs *joint* behaviour across buses and hours. The review names copulas and GANs as the two main routes to that.

**4. Combination and ensembles.** Averaging forecasts is one of the most reliable improvements in forecasting, and energy is no exception. The examples include combining "sister" regression models for load, and *quantile regression averaging* (QRA): regress the target's quantiles on several point forecasts. QRA was proposed for prices and then found effective for load. Even averaging one model over a few short and a few long training windows helps. A second kind of ensemble comes from weather centres. The 51-member ECMWF ensemble perturbs initial conditions and integrates each member forward. Wind and solar forecasters post-process such ensembles, because raw NWP ensembles are typically under-dispersed. Post-processing (e.g. ensemble model output statistics) recalibrates them, the regression analogue of temperature scaling.

**5. Hierarchical forecasting.** Grids aggregate. The feeder loads under a substation must sum to the substation load (minus small losses), and regional loads must sum to the system load. Base forecasts made independently at each level will not add up. Reconciliation projects them onto the coherent subspace. Let `b` be the bottom-level series, `S` the summing matrix, so that all levels satisfy `y = S b`, and let `ŷ` be the stacked base forecasts:

```
ỹ = S (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹ ŷ
```

`W = I` is OLS reconciliation. MinT takes `W` as the covariance of base-forecast errors (further reading). The review reports two benefits: coherence, and usually better accuracy than the base forecasts. It also reports that contestants largely ignored the hierarchy in two global competitions. It notes that there was then no consensus on what coherence means for *probabilistic* forecasts.

**6. What goes wrong in papers (Section V).** The authors, editors at major journals, list recurring problems. Models are evaluated on one short, unique dataset. Datasets are so heavily reused that future information leaks in. Metrics are poor, e.g. MAPE on quantities near zero, such as PV at dawn or prices that turn negative. Authors report only the metric that favours their model and skip significance tests. They compare only against weak baselines, sometimes not even a naive one. Their remedies are proper scores for probabilistic forecasts, Diebold–Mariano tests when errors are close, comparison against naive and state-of-the-art baselines, and a leakage check whenever results look "too good to be true". This is ML's benchmark-hygiene debate, transplanted.

**7. Two open problems that land on the optimizer.**
- *Closed-loop forecasting (VI-B).* Forecasts trigger actions: demand response, battery charging, bids. Those actions change load and prices, so a forecast that was accurate becomes wrong because people acted on it. The historical data are contaminated too. If demand response shaved past peaks, a model trained on that history underestimates the "organic" peak. ML readers will recognize *performative prediction* and the selection bias of logged-policy data.
- *Valuation (VI-C).* Error metrics need not track economic value. One cited study estimates that a 1% load-forecast error reduction is worth up to $1.6 million a year to a 10,000 MW utility in unit-commitment costs. Yet costs are asymmetric, and one study found a moving-horizon deterministic model beating stochastic and robust ones in volatile periods. The review lists "practical error measures for probabilistic energy forecasting" as the one prediction from 2015 that saw little progress.

**A small example**

Two feeders under one substation. The base forecasts for 18:00 are feeder 1 `ŷ₁ = 60` MW, feeder 2 `ŷ₂ = 30` MW and substation `ŷ_T = 100` MW, so the bottom-up sum disagrees with the top forecast by 10 MW. With `b = (y₁, y₂)` and stacked order `(T, 1, 2)`:

```
S = [[1,1],[1,0],[0,1]]     SᵀS = [[2,1],[1,2]]     (SᵀS)⁻¹ = (1/3)[[2,−1],[−1,2]]
Sᵀŷ = (100+60, 100+30) = (160, 130)
b̃ = (1/3)(2·160 − 130, −160 + 2·130) = (63.33, 33.33)     ỹ_T = 96.67
```

OLS reconciliation splits the 10 MW disagreement equally over the three nodes: each feeder moves up by 3.33 and the substation moves down by 3.33. If the substation forecast had historically been much more accurate (small entry in `W`), MinT would move the feeders almost all the way to 100 instead. Now the metric trap. A PV plant with actual output 0.5 MW and forecast 1.5 MW at 07:00 contributes an absolute percentage error of 200%. The same 1 MW error at noon, with 40 MW actual, contributes 2.5%. MAPE averaged over a day is therefore dominated by dawn and dusk.

**How it connects**

This is the third Track 05 preview. **X002** (Pinson on wind) showed why the wind subdomain went probabilistic first, and this review places that story in the whole field. **X011** (DeepAR) is one concrete instance of Section III-A's deep-learning trend and of the scenario form in item 3. Its independent-per-series sampling is exactly the gap that copulas fill. The valuation problem in VI-C is the forecasting side of **X008** (conformal decision theory) and of **X019** (Wasserstein DRO). Both ask how errors in the input distribution turn into decision cost. The hierarchical constraint `y = S b` has the same flavour as the power-balance constraints of **A08**–**A11**: a linear coherence condition that a learned output must satisfy, here solved by a projection, as in **X016** (HardNet). The prices in Section VI-D are the LMPs of **A11**.

In phase B, **B13** formalizes forecasts as distributions, **B14** develops quantile regression and the pinball loss, **B15**–**B16** cover calibration, sharpness and proper scores (CRPS, interval score), **B17** the multivariate energy and variogram scores, **B19** copula scenarios, and **B24** open data and leak-free time-series splits. The next focus lesson is **A15**, Ipopt in practice.

**Check yourself**

1. A forecast is issued at 10:00 on Monday as a "day-ahead" forecast. What range of lead times does it cover, and why might its errors not be identically distributed across the 24 delivered hours?
2. In the two-feeder example, which `W` reproduces pure bottom-up forecasting (top level = 90), at least in the limit? What does that say about the trust placed in the top-level forecast?
3. Why does closed-loop forecasting make a held-out test set built from recent history an optimistic estimate of forecast error, even with no data leakage in the usual sense?

<details><summary>Answers</summary>

1. Tuesday 00:00–24:00 is 14 to 38 hours ahead of issue (counting each hour's start). Errors grow with lead time, and they also depend on the hour of day: the evening ramp and solar output at noon behave differently from the night. So the hourly errors differ in both scale and shape, i.e. they are heteroskedastic across the horizon. The model should be evaluated per lead time, not as one pooled score.
2. Let the top-level weight `W_TT → ∞`, so the top forecast gets no weight and `b̃ = ŷ_bottom = (60, 30)`, giving 90 at the top. Bottom-up is the special case that ignores the top-level forecast entirely, i.e. treats it as infinitely noisy.
3. The forecasts themselves triggered actions (demand response, storage, bids) that moved the realized outcomes. Test-set outcomes were therefore partly produced by reactions to forecasts made under the old regime. A new model deployed in a world with more responsive participants faces a shifted distribution: a performative, policy-induced shift. Error estimated on yesterday's data does not account for this.

</details>
