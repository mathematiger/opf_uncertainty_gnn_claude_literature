# Reading log

All entries, newest first. See [`curriculum.md`](curriculum.md) for the plan.

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

---

## 2026-10-30 (Friday) — A12 · Track 02: Optimal power flow: formulations, relaxations, solvers — Security-constrained OPF: surviving N-1, preventively or correctively

**Type:** Focus lesson
**Main reference:** [State-of-the-art, challenges, and future trends in security constrained optimal power flow — F. Capitanescu, J. L. Martinez Ramos, P. Panciatici, D. Kirschen, A. Marano Marcolini, L. Platbrood & L. Wehenkel (2011), Electric Power Systems Research 81(8):1731–1741](https://orbi.uliege.be/handle/2268/88635) — open author preprint on ORBi ([DOI](https://doi.org/10.1016/j.epsr.2011.04.003)). Read §1–3.1 (motivation, the formulation (1)–(8), limits on corrective actions) and §4.1 (reducing problem size: contingency filtering, Benders, linearization). Skim §3.2 (stability), §4.3 (discrete variables) and §5 (stochastic and worst-case SCOPF); this lesson is an overview only.
**Further reading:**
- [Recent Developments in Security-Constrained AC Optimal Power Flow: Overview of Challenge 1 in the ARPA-E Grid Optimization Competition — I. Aravena, D. K. Molzahn et al. (2022), arXiv:2206.07843](https://arxiv.org/abs/2206.07843) — a precise modern SC-AC-OPF formulation with generator response to contingencies, plus how the top competition teams solved it at scale; good to revisit after A13–A15.

**Why this matters** — Real operators do not solve "the" OPF of A10: they solve one that must stay feasible after any single equipment failure, and that requirement often costs more than every other constraint combined. It multiplies the problem size by the number of contingencies, which is exactly why learned solvers, contingency screening and warm starts attract interest here. Anyone building learned OPF solvers under forecast uncertainty needs to know what "secure" means, because a proxy that is feasible only in the intact network is not deployable.

**Core ideas**

**1. The N-1 criterion.** A01 introduced the rule: the grid must survive the loss of any single element (a line, transformer or generator) without violating limits and without load loss. Each such outage is a *contingency* `k = 1, …, c`, and `k = 0` is the intact base case. In ML terms, the base-case decision must be robust against a finite, enumerated set of perturbations to the graph itself: deleting an edge changes the admittance matrix (A03) and so the whole flow pattern.

**2. The conventional SCOPF.** Capitanescu et al. write it compactly (their eq. 1–8):

```
min_{x_0..x_c, u_0..u_c}  f_0(x_0, u_0)
s.t.  g_0(x_0, u_0) = 0,      h_0(x_0, u_0) ≤ L^l                  (base case, normal limits)
      g_k^s(x_k^s, u_0) = 0,  h_k^s(x_k^s, u_0) ≤ L^s     k=1..c   (just after outage, emergency limits)
      g_k(x_k, u_k) = 0,      h_k(x_k, u_k) ≤ L^m         k=1..c   (after correction, medium-term limits)
      |u_k − u_0| ≤ Δu_k = T_k · du_k/dt                   k=1..c   (coupling)
```

`u` are controls (generator P and voltage setpoints, transformer taps, shunts, phase shifters, breaker states) and `x` are states (bus voltages and angles). The `g` are A04's AC power-flow equations in the respective topology; the `h` are A07's limits. Note the three limit sets, with `L^l ≤ L^m ≤ L^s`: equipment can tolerate a higher current for a short time (A07's `RATE_A/B/C`). The paper's French example has three line current limits with disconnection delays of 1, 5 and 20 minutes. The operator cannot act within one minute, so the first limit must hold by preventive action alone; one corrective action (typically a predefined topology switch) is expected within 5 minutes; satisfying the third limit may need several, such as redispatch or starting fast units.

Only the base-case cost `f_0` is minimised: post-contingency actions are treated as free because they are rarely used. The `c` copies of the network share only the controls `u_0` through the coupling constraints, so the constraint matrix is block-angular, which is the structure Benders decomposition (and later, parallel solvers) exploit.

**3. Preventive vs corrective.** The **preventive** SCOPF (PSCOPF) forbids non-automatic post-contingency actions: `u_k = u_0` for all `k`, so the base case must already be safe for every outage (only automatic responses such as frequency-control droop and tap changers act). The **corrective** SCOPF lets each contingency have its own recovery controls within the ramp budget `Δu_k`. Corrective is cheaper, since it buys security with an option to act later instead of an up-front redispatch, but riskier: the actions must actually work in the available time, the short-term state before they act must be viable, and the sequence must not make things worse. The paper notes that market pressure pushes operators towards corrective control, and that in some systems N-1 cannot be met at all without it. This is a two-stage decision problem: first-stage `u_0`, recourse `u_k` per scenario, the template B02 makes general.

**4. Why it is hard.** Size: the paper cites a European extra-high-voltage model with about 13,000 buses and about 30,000 N-1 contingencies, and a North American Eastern Interconnection model with about 43,000 buses and about 70,000 contingencies. Stacking all of them gives an AC problem tens of thousands of times larger than one OPF. On top come discrete controls (switching, taps, unit start-ups), which make it a MINLP, and stability, which the static model ignores entirely; the paper's §3.2 surveys how transient and voltage stability are bolted on as extra constraints. Limiting the number of corrective actions per contingency needs binaries: `−s_k Δu_k ≤ u_k − u_0 ≤ s_k Δu_k`, `1ᵀs_k ≤ N_k`, `s_k ∈ {0,1}` (eq. 9–11).

**5. How it is solved in practice.** The key empirical fact: *most contingencies do not constrain the optimum*. So practical methods (§4.1) iterate:

1. Solve the SCOPF with a small subset of contingencies (initially none).
2. Run a cheap security analysis: a post-contingency power flow (A05) for every outage, or DC screening with LODFs (A06).
3. A contingency filter picks the violated or "dominating" ones; add them and repeat.

This is lazy constraint generation, the same idea as cutting planes or active-set methods. Filters use severity indices or Lagrange multipliers of a relaxed problem; the parameter-free "non-dominated contingency" filter compares violations across contingencies. Alternatives shrink each contingency instead: Benders decomposition (exact only for convex problems, so careful with AC), linearizing post-contingency constraints around the base case (with DC models, this is the DC-SCOPF that sets LMPs in markets such as PJM), or network compression that keeps only the electrically nearby area exactly. Solutions from simplified models must be re-checked against the full model.

**6. Beyond N-1.** §5 argues that equal-weight, deterministic contingency lists are outdated: renewable forecast errors act like continuous "injection contingencies". Two directions follow. A **stochastic SCOPF** minimises `f_0 + λ Σ_k p_k f_k(x_k, u_0, u_k)`, weighing contingencies by probability and pricing corrective actions. A **worst-case (robust) SCOPF** is a bi-level max–min: an adversary picks injections in a box `[i_min, i_max]` to maximise the post-contingency violation that the best preventive and corrective actions leave. These are Phase B's stochastic and robust programs (B02, B08) applied to security.

**A small example**

Reuse A11's three-bus triangle (equal reactances, bus 3 reference, load at bus 3; generator 1 at bus 1 costs 10 $/MWh, generator 2 at bus 2 costs 30 $/MWh). Now let all lines be rated 250 MW except line 1–2, rated 150 MW, and lower the load to 240 MW.

- **Ordinary DC-OPF:** `P_1 = 240`, `P_2 = 0`. Base flows `F_13 = 160`, `F_12 = 80`, `F_23 = 80`, all within limits. Cost 2400 $/h.
- **Contingency "line 1–3 out":** the grid becomes the path 1–2–3, so all of bus 1's output crosses line 1–2: `F_12 = P_1 = 240 > 150`. The OPF solution is not N-1 secure. (The other two outages are fine: without 1–2, `F_13 = P_1 = 240 ≤ 250`; without 2–3, `F_13 = 240` and `|F_12| = P_2 = 0`.)
- **Preventive SCOPF:** require `P_1 ≤ 150` in advance, so `P_2 = 90`. Cost 1500 + 2700 = 4200 $/h. The price of security is 1800 $/h, paid every hour although the outage is rare. At this solution the binding constraint is a *post-contingency* flow limit, so it also sets the prices: an extra MW at bus 3 must come from generator 2, giving `LMP_3 = 30`.
- **Corrective SCOPF:** suppose after the outage generator 1 can ramp down and generator 2 up by `Δ = 50` MW within the allowed time, and line 1–2 has a short-term emergency rating of 200 MW. Then the base case only needs `P_1 ≤ 200` (short-term: `F_12 = P_1 ≤ 200`; after correction: `P_1 − 50 ≤ 150`). Take `P_1 = 200`, `P_2 = 40`: cost 2000 + 1200 = 3200 $/h, saving 1000 $/h relative to preventive, provided the 50 MW of corrective redispatch actually happens on time.

**How it connects**

The constraints are those of A10 (AC) or A11 (DC), copied once per topology. Post-contingency flows come from A05's power flow or A06's LODFs; the three limit sets refine A07's ratings, and A01 introduced the N-1 rule. X013 used N-1 line removals as a test of whether GNNs generalise across topology, and X012 mentioned a learned proxy for N-1 security-constrained DC-OPF with tens of millions of variables; this lesson is the problem behind both. The block-angular structure and its decomposition become precise once A13 gives Lagrangian duality and A14–A15 explain interior-point solvers. A30 adds the unit-commitment layer the paper mentions, and B02/B08 generalise the preventive–corrective split to two-stage stochastic programs and affine recourse policies.

**Check yourself**

1. In the conventional SCOPF, what is the role of the coupling constraint `|u_k − u_0| ≤ Δu_k`, and what does the preventive SCOPF set it to?
2. Why do practical SCOPF solvers not simply include all 30,000 contingencies at once, and what property of real systems makes the iterative approach work?
3. In the three-bus example, what corrective ramp `Δ` would make the corrective SCOPF reach the unconstrained cost of 2400 $/h, and which other limit would then bind?

<details><summary>Answers</summary>

1. It limits how far each contingency's recovery controls may move away from the base-case controls, with `Δu_k = T_k · du_k/dt` set by the time available and the ramp rates. It is what makes `u_0` matter for the post-contingency states. The preventive SCOPF sets `Δu_k = 0` (for non-automatic controls), i.e. `u_k = u_0`.
2. Memory and CPU time would be prohibitive (the problem grows roughly by a factor of `c`), and the many redundant constraints can cause numerical trouble. It works because in practice only a small subset of contingencies is binding at the optimum, so constraint generation with a cheap screening step finds the same optimum with a fraction of the constraints.
3. With `P_1 = 240` the condition after correction is `240 − Δ ≤ 150`, so `Δ ≥ 90` MW (and generator 2 must be able to ramp from 0 to 90). But the short-term state, before the correction acts, then has `F_12 = 240`, above the 200 MW emergency rating. So the short-term limit binds first: the corrective SCOPF cannot go beyond `P_1 = 200` unless the emergency rating is at least 240 MW.

</details>

---

## 2026-10-29 (Thursday) — X016 · Track 10: Physics-informed and constrained learning — HardNet: one closed-form layer that makes affine constraints impossible to violate

**Type:** Cross-phase preview
**Main reference:** [HardNet: Hard-Constrained Neural Networks with Universal Approximation Guarantees — Youngjae Min & Navid Azizan (2024; revised v4 2025), arXiv preprint 2410.10807](https://arxiv.org/abs/2410.10807). Read Section 4: the problem setup (eq. 4), Assumption 6, the closed-form layer (eq. 5) and Propositions 7–8. Then read the statements of Theorem 9 and Corollary 10, and Section 5.2 with Table 3 (learning an optimization solver). Skim Section 5.1 and the safe-control example in 5.3. The arXiv page lists no conference or journal venue, so treat it as a preprint.
**Further reading:**
- [DC3: A Learning Method for Optimization with Hard Constraints — Donti, Rolnick & Kolter (2021), ICLR](https://arxiv.org/abs/2104.12225) — the main baseline HardNet compares against. It uses equality completion plus gradient-based inequality correction and is covered properly in lesson C06.
- [OptNet: Differentiable Optimization as a Layer in Neural Networks — Amos & Kolter (2017), ICML](https://arxiv.org/abs/1703.00443) — the machinery behind HardNet's general convex variant: backpropagating through the solution of a QP (lesson C09).

**Why this matters** — A learned dispatch or OPF proxy that violates a power-balance or line-limit constraint cannot be used as is. A penalty term in the loss makes violations rarer on the training distribution, but it guarantees nothing on a new forecast. HardNet is a short, clean answer to a narrower question: when the constraints are affine in the output (their coefficients may depend on the input), can a network satisfy them exactly on every input without losing expressive power? Knowing exactly what this answer covers, and where it stops, is useful preparation for the constraint-handling lessons in phases C and E.

**Core ideas**

**1. The constraint class.** The network maps an input `x` (say a load forecast) to an output `y = f_θ(x)` (say generator setpoints). The constraints are two-sided and affine in `y`, with input-dependent data (eq. 4):

```
b_l(x) ≤ A(x) y ≤ b_u(x),     A(x) ∈ ℝ^{n_c × n_out}
```

Setting `b_l = b_u` gives an equality, and `±∞` gives a one-sided inequality. The "input-dependent" part matters for OPF-like problems: total demand, and therefore the balance constraint, changes from one input to the next. A fixed output layer such as a sigmoid box therefore cannot encode it. Assumption 6 requires (i) that a feasible `y` exists for every `x` and (ii) that `A(x)` has full row rank, which forces `n_c ≤ n_out`. Note that `n_c` counts rows, and a two-sided bound costs only one row.

**2. The layer.** Let `A⁺ = Aᵀ(AAᵀ)⁻¹` be the right pseudo-inverse. HardNet-Aff appends (eq. 5)

```
P(f)(x) = f + A⁺ [ ReLU(b_l − A f) − ReLU(A f − b_u) ]
```

with all quantities evaluated at `x`. Read the bracket row by row. For each constraint row `i`, it holds zero if `a_iᵀ f` already lies in `[b_l,i, b_u,i]`, and otherwise the signed distance to the violated bound. Multiplying by `A⁺` turns that vector of required row changes into the minimum-norm change of `y` that produces it. One can check that `A P(f) = A f + bracket`. Every violated row therefore lands exactly on its bound and every satisfied row keeps its value (Proposition 8). Proposition 7 states this as an ℓ2 projection: among all `y` with those row values, it picks the one closest to `f`. The layer involves no solver and no iterations. It costs one small linear solve per input and is differentiable almost everywhere, since ReLU is piecewise linear. The whole model `P ∘ f_θ` is trained with an ordinary unconstrained optimizer on whatever loss you like, for example the dispatch cost.

**3. Caveat: this is not the Euclidean projection onto the feasible polytope.** Rows that are satisfied are frozen at their current values, even when moving them would bring the output closer. The result is feasible, but it can be farther from `f` than the true projection. The example below shows a case.

**4. Expressivity.** A natural worry is that a hard layer shrinks the function class. Theorem 9 says it does not: on a compact input set, with continuous `A(x)` and Assumption 6, HardNet-Aff networks are dense in the constraint-satisfying target functions whenever the plain networks are dense. Corollary 10 gives a width condition for ReLU networks, `w ≥ max{n_in + 1, n_out}`, for `L^p` targets. Intuition (my paraphrase of the mechanism): `P` leaves feasible points unchanged and moves an output by no more than its violation, scaled by `‖A⁺‖`. So if `f_θ` is close to a feasible target, `P(f_θ)` is at least as close up to a constant. The general variant, HardNet-Cvx (eq. 8), replaces the closed form with an ℓ2 projection onto an arbitrary convex input-dependent set, solved by a differentiable convex-optimization layer. It has a matching approximation theorem (Theorem 11), but the paper reports no experiments with it.

**5. What they found.** All models are 3-layer ReLU MLPs with 200 hidden units. On a learned-solver benchmark (Section 5.2) with 100 variables, 50 equalities and 50 inequalities (exactly `n_c = n_out`), HardNet-Aff reaches an objective of −14.10 with zero violations. The IPOPT solver reaches −14.28 at about 1182 ms per instance, against 6.69 ms for HardNet-Aff. DC3 is also feasible but worse (−12.57, 9.61 ms). Plain and penalty-trained networks violate the constraints. In safe control (Section 5.3), HardNet-Aff has zero violations at 2.71 ms, against 579 ms for a control-barrier-function QP. DC3 and soft penalties reach lower cost there, but with maximum violations near 6.9. These are synthetic, small benchmarks, so read them as a proof of concept and not as power-system evidence.

**A small example**

Two generators serve demand `D = 100` MW. Generator 1 is capped at 60 MW. Output `y = (P_1, P_2)`, with constraints written as rows:

```
row 1:  P_1 + P_2 ∈ [100, 100]     (balance, input-dependent: D comes from x)
row 2:  P_1       ∈ [0, 60]
A = [[1, 1], [1, 0]],   A⁺ = A⁻¹ = [[0, 1], [1, −1]]
```

**Case 1:** raw output `f = (80, 30)`. Then `A f = (110, 80)`. Row 1 is 10 too high and row 2 is 20 too high, so the bracket is `(−10, −20)`. `A⁻¹(−10, −20) = (−20, 10)`, which gives `P(f) = (60, 40)`. The result is balanced and at the cap. This matches the Euclidean projection here.

**Case 2:** `f = (50, 30)`. Then `A f = (80, 50)`: row 1 is 20 short and row 2 is fine, so the bracket is `(20, 0)` and `P(f) = (50, 50)`. Because row 2's value `P_1 = 50` is frozen, the entire shortfall goes to generator 2. The Euclidean projection onto the feasible set would be `(60, 40)`, which splits the shortfall evenly. Both points are feasible, and HardNet's moves the output farther (distance 20 against 14.1). Training compensates: the loss is computed after `P`, so `f_θ` learns where to aim.

**Where it stops.** Add a bound `P_2 ∈ [0, 80]`. That is a third row for two outputs, so `A` can no longer have full row rank and HardNet-Aff no longer applies. A realistic DC-OPF (A11) has one balance row, one row per line flow limit (through PTDFs, A06) and one per generator bound. That is far more rows than generators, so the problem falls to HardNet-Cvx or to other methods.

**How it connects**

This continues **X007**: Lagaris's trial solution `A(x) + F(x, N)` satisfies boundary conditions by construction, and HardNet does the same for a whole family of input-dependent affine constraints. It also complements **X012**, where repair layers such as E2ELR were built for one specific feasible set (economic dispatch with reserves). HardNet trades that problem-specific structure for a generic closed form with a row-count limit. The pseudo-inverse correction is the same minimum-norm idea behind equal incremental cost in **A08**. The limit `n_c ≤ n_out` explains why the line-limited DC-OPF of **A11** needs more machinery. In the curriculum, C06 (DC3) and C09 (differentiable optimization layers) cover the general case, and E04, E05 and E16 compare soft constraints, constraints by construction and projection layers for AC-OPF, where the constraints are no longer affine. Next in the focus phase, A12 adds security constraints (N-1), which multiply the number of constraint rows.

**Check yourself**

1. Why does the layer use `A⁺ = Aᵀ(AAᵀ)⁻¹`, and which part of Assumption 6 makes it well defined?
2. In Case 2, compute the Jacobian of `P` with respect to `f` and say which direction of the upstream gradient it removes.
3. A grid has 5 generators and 8 lines with two-sided flow limits, plus a balance equation and generator bounds. Can HardNet-Aff enforce all constraints? What is the largest set you could enforce with it directly?

<details><summary>Answers</summary>

1. The bracket specifies how much each constraint row `a_iᵀy` must change. `A⁺` gives the smallest change `Δy` with `A Δy` equal to that vector, so the output moves as little as possible while hitting the target row values. It requires `AAᵀ` to be invertible, which is exactly the full-row-rank part of Assumption 6 (and hence `n_c ≤ n_out`).
2. In the violated region of row 1 with row 2 satisfied, `P(f) = f + A⁻¹(100 − (f_1 + f_2), 0)ᵀ = f − (0, f_1 + f_2 − 100)`. So `∂P/∂f = I − A⁻¹ e_1 a_1ᵀ = [[1, 0], [−1, 0]]`. The output depends on `f_1` alone, and the upstream gradient along `f_2` is zeroed: changing `f_2` cannot change a balanced output whose `P_1` is frozen. This can create "dead" gradient directions much like an active clip, and the network learns only through `f_1` in this region.
3. No. There is 1 balance row, 8 line rows and 5 generator rows, so 14 rows for 5 outputs. Full row rank allows at most 5 linearly independent rows: for example the balance equation plus 4 generator bounds, or the balance plus a few critical line limits. Everything else needs HardNet-Cvx (a QP projection), DC3-style correction or a problem-specific repair layer.

</details>

---

## 2026-10-28 (Wednesday) — A11 · Track 02: Optimal power flow: formulations, relaxations, solvers — DC-OPF, congestion, and locational marginal prices as dual variables

**Type:** Focus lesson
**Main reference:** [Fundamentals of Power System Economics — D. S. Kirschen & G. Strbac (2004), John Wiley & Sons](https://doi.org/10.1002/0470020598) — Chapter 6 "Transmission Networks and Electricity Markets", §6.3 "Centralized Trading over a Transmission Network": §6.3.1 (the two-bus example, p. 152 in the first edition), §6.3.2 (the three-bus example, from p. 155), §6.3.4 (mathematical formulation of nodal pricing, from p. 181). Skim §6.3.3 (losses) and §6.3.5 (managing transmission risks). The book is not open access; a second edition (2019) keeps the same chapter.
**Further reading:**
- [Electricity Markets, Lecture 5: Markets with Transmission Congestion — T. Brown & M. Schäfer (2016), FIAS / Goethe University Frankfurt course slides](https://nworbmot.org/courses/electricity_markets/Lecture_5/em-lecture-5.pdf) — open slides that redo Kirschen & Strbac's two-node example and then derive multi-node nodal prices from the KKT conditions; [Lecture 4](https://nworbmot.org/courses/electricity_markets/Lecture_4/em-lecture-4.pdf) covers the DC network constraints and PTDFs used there.

**Why this matters** — The DC-OPF is the optimization problem that clears most wholesale electricity markets with a network, and its dual variables are the prices people actually pay. It is also the simplest OPF where duality, sensitivity and active sets can be computed by hand. Anyone building learned OPF solvers, especially ones that predict prices, active constraints or sensitivities under forecast uncertainty, needs to know what these multipliers mean and why they jump.

**Core ideas**

**1. From AC-OPF to DC-OPF.** Take A10's AC-OPF and apply A06's DC assumptions: flat voltages (`V_i = 1` pu), small angle differences (`sin θ ≈ θ`), lossless lines (`R ≪ X`), and ignore reactive power. What remains is a linear network:

```
min_{P^G, θ}   Σ_g C_g(P_g)
s.t.   Σ_{g at i} P_g − d_i = Σ_{k} (θ_i − θ_k) / x_ik     ∀ i      (λ_i)
       −F_l^max ≤ (θ_i − θ_k) / x_l ≤ F_l^max               ∀ l=(i,k)  (μ_l^−, μ_l^+)
       P_g^min ≤ P_g ≤ P_g^max,   θ_ref = 0
```

With linear (or piecewise-linear) costs this is an LP; with quadratic costs `c1 P + c2 P²` it is a convex QP. Either way it is convex, so the KKT conditions (A13 makes them precise) are necessary and sufficient, and every local optimum is global. That is the big structural difference from the AC-OPF.

**2. The PTDF form.** Eliminating `θ` with the PTDF matrix from A06 gives the form Kirschen & Strbac use: one system-wide balance and line flows as linear functions of net injections `z_i = P_i^G − d_i`:

```
Σ_i z_i = 0                                     (λ)
−F^max ≤ PTDF · z ≤ F^max                       (μ^−, μ^+)
```

The line-flow rows are the only place the network appears. If no line limit binds, the problem collapses to A08's economic dispatch.

**3. Locational marginal prices are duals.** The **LMP** at bus `i` is the multiplier of bus `i`'s balance constraint, which equals the change in optimal total cost when the load at `i` grows by one MW (an envelope-theorem / sensitivity statement):

```
LMP_i = ∂C*/∂d_i = λ − Σ_l μ_l · PTDF_{l,i}         (μ_l = μ_l^+ − μ_l^−)
```

Here `λ` is the price at the reference bus (the energy component). The sum is the **congestion component**: each binding line `l` adds its shadow price `μ_l`, weighted by how much an injection at `i` loads that line. In a lossful model (K&S §6.3.3) a third, **loss component** appears; in the lossless DC model it is zero. Stationarity for a generator that is strictly between its bounds says `C_g'(P_g) = LMP_{bus(g)}`: every unconstrained generator runs where its marginal cost equals its local price, just as A08's equal-incremental-cost rule, now localised.

**4. What congestion does to prices.** Kirschen & Strbac build the intuition in three steps.

- *Two buses, one line.* (§6.3.1, their "Borduria–Syldavia" example.) Bus 1 has cheap generation (`MC_1 = 10 + 0.01 Q_1` $/MWh) and 500 MW of load; bus 2 has expensive generation (`MC_2 = 13 + 0.02 Q_2`) and 1500 MW of load. Without a limit, the market clears at one price: `Q_1 = 1433`, `Q_2 = 567`, price ≈ 24.3 $/MWh, flow 933 MW. With a 400 MW line limit, bus 1 can export only 400 MW: `Q_1 = 900`, `Q_2 = 1100`, and the prices split to `λ_1 = 19` and `λ_2 = 35` $/MWh. The difference, 16 $/MWh, is the shadow price of the line.
- *Three buses, a loop.* (§6.3.2.) With a meshed network, power follows Kirchhoff's laws, not contracts: a transfer from one bus to another spreads across all parallel paths in proportion to the PTDFs. One binding line therefore changes the prices at *every* bus, and some results look paradoxical. In general, a bus can have a price higher than the marginal cost of every generator in the system, and prices can even become negative: adding load at a bus may relieve a constraint and lower total cost.
- *Merchandising surplus.* With congestion, loads pay more in total than generators receive. The difference, `Σ_i LMP_i · d_i − Σ_g LMP_{bus(g)} · P_g`, equals `Σ_l μ_l F_l^max` (by LP duality). This **congestion surplus** is collected by the system operator and is what financial transmission rights redistribute (§6.3.5).

**5. LP degeneracy and the shape of the price map.** As a function of the load vector `d`, the DC-OPF is a *parametric* program. For an LP the load space splits into polyhedral "critical regions", each with a fixed set of marginal generators and binding lines. Inside a region the dispatch is affine in `d` and the LMPs are constant. At region boundaries the prices jump, and *on* a boundary the duals are not unique (any value between the two neighbouring prices is optimal; markets need a tie-breaking rule). With quadratic costs, prices vary continuously within a region but still have kinks where the active set changes. For a learned model this means: the dispatch map is continuous and piecewise affine, but the price map is piecewise constant (LP) or piecewise affine (QP) with kinks and jumps, which is much harder to regress with smooth activations.

**A small example**

Three buses, three identical lines (`x` the same on 1–2, 1–3, 2–3). Generator 1 at bus 1 costs 10 $/MWh, generator 2 at bus 2 costs 30 $/MWh, both with ample capacity. A 300 MW load sits at bus 3. Take bus 3 as reference. With equal reactances, an injection at bus 1 withdrawn at bus 3 sends 2/3 along the direct line 1–3 and 1/3 around the loop 1–2–3. So `PTDF_{13,1} = 2/3` and `PTDF_{13,2} = 1/3`, and the flow on line 1–3 is `F_13 = (2/3) P_1 + (1/3) P_2`.

- **No limit:** bus 1 supplies everything, `P_1 = 300`, `F_13 = 200`. All prices are 10.
- **Limit `F_13 ≤ 160`:** with `P_1 + P_2 = 300`, `F_13 = 100 + P_1/3 ≤ 160` gives `P_1 = 180`, `P_2 = 120`. Cost = 1800 + 3600 = 5400 $/h. Flows: `F_13 = 160`, `F_23 = 140`, `F_12 = 20`.
- **LMPs:** both generators are marginal, so `LMP_1 = 10` and `LMP_2 = 30`. For bus 3, serve one more MW without overloading 1–3: `dP_1 + dP_2 = 1` and `(2/3) dP_1 + (1/3) dP_2 = 0` give `dP_1 = −1`, `dP_2 = +2`. The cost change is −10 + 60, so `LMP_3 = 50` $/MWh, higher than either generator's cost.
- **Line shadow price:** from `LMP_1 = LMP_3 − μ · (2/3)`, `μ = 60` $/MWh. Check `LMP_2 = 50 − 60/3 = 30` ✓. Directly: one more MW on line 1–3 lets `P_1` rise by 3 and `P_2` fall by 3, saving 3 · 20 = 60 ✓.
- **Surplus:** load pays 50 · 300 = 15000; generators receive 1800 + 3600 = 5400; the difference 9600 = `μ · F^max` = 60 · 160 ✓.

Now nudge the load to 301 MW: dispatch moves continuously, but the prices stay (10, 30, 50). Raise the limit above 200 MW and they all drop at once to 10. That jump is a critical-region boundary.

**How it connects**

This lesson combines A06 (DC power flow and PTDFs) with A08 (economic dispatch and the multiplier as a price) and simplifies A10's AC-OPF. A07's thermal limits are the constraints whose multipliers make up the congestion component. The active-set view of section 5 is the structure behind X003 (learning which constraints bind) and X012 (proxies with guarantees): a classifier over critical regions recovers the exact dispatch by one linear solve. X015 noted that averaging solutions is safe for convex problems such as this one: the DC-OPF dispatch map is continuous, so squared-error regression is well posed for dispatch, though not for the jumping prices. Next comes A12 (security-constrained OPF), which adds contingency flows to the same PTDF rows. A13 then formalises the KKT and duality arguments used here, and B07 makes this DC-OPF chance-constrained under forecast errors.

**Check yourself**

1. In the two-bus example with a 400 MW line limit, verify the prices 19 and 35 $/MWh and compute the congestion surplus.
2. Why can the LMP at bus 3 in the three-bus example exceed the marginal cost of every generator, and what would happen to it if generator 2's capacity were capped at 100 MW?
3. Why does a neural network trained to predict LMPs from loads face a harder regression problem than one predicting the dispatch, for an LP-based DC-OPF?

<details><summary>Answers</summary>

1. Demand totals 2000 MW and bus 1 exports 400 MW, so `Q_1 = 500 + 400 = 900` and `Q_2 = 1500 − 400 = 1100`. Then `λ_1 = 10 + 0.01 · 900 = 19` and `λ_2 = 13 + 0.02 · 1100 = 35` $/MWh. The congestion surplus is `(35 − 19) · 400 = 6400` $/h, equal to the line's shadow price times its capacity.
2. Serving one extra MW at bus 3 without increasing flow on the binding line 1–3 requires backing off cheap generator 1 by 1 MW and raising expensive generator 2 by 2 MW, so the marginal cost of that redispatch (−10 + 2 · 30 = 50) exceeds either unit's cost. If generator 2 were capped at 100 MW, the 300 MW load could not be served with `F_13 ≤ 160` (it needs `P_1 ≤ 180`, so `P_2 ≥ 120`): the problem becomes infeasible. In practice it would then need load shedding at a high penalty price, which would set `LMP_3`.
3. For an LP, the dispatch is a continuous, piecewise-affine function of the loads, but the LMPs are piecewise constant: they stay flat inside each critical region and jump at the boundaries, where the duals are not even unique. Smooth networks blur the jumps, and errors concentrate exactly near the boundaries, where the active set changes and prices matter most.

</details>

---

## 2026-10-27 (Tuesday) — X015 · Track 09: Aleatoric/epistemic UQ and evaluating uncertainty signals — When the answer is not unique: Bishop's mixture density networks

**Type:** Cross-phase preview
**Main reference:** [Mixture Density Networks — C. M. Bishop (1994), Technical Report NCRG/94/004, Neural Computing Research Group, Aston University](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf). This is the open PDF from the Aston repository (26 pages; record page [here](https://publications.aston.ac.uk/id/eprint/373)). Read Section 1 and Section 2 (least squares gives the conditional mean). Read Section 3 up to eq. (31) (the model and its loss), then the part of Section 3 from eq. (45) to eq. (49) (mean, variance and mode of the predicted density). Read Section 5 (the toy inverse problem, Figures 4–7) and Section 7 (discussion, choosing the number of kernels). Skim the gradient derivations and Section 6 (robot arm).
**Further reading:**
- [Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction — O. Makansi, E. Ilg, Ö. Çiçek & T. Brox (2019), CVPR 2019](https://arxiv.org/abs/1906.03631): a modern look at how MDNs fail in practice (unstable training, mode collapse). It replaces direct likelihood training with winner-takes-all sampling followed by fitting a mixture. It also discusses how to evaluate a multimodal prediction when each input comes with only one observed outcome.

**Why this matters** — Many maps that a learned OPF or power-flow model must learn are inverse maps: from a desired outcome back to setpoints or states. Inverse maps can be multi-valued, and a network trained with squared error then predicts the average of several valid answers, which may itself be invalid. Bishop's short report is where the standard fix comes from: predict a full conditional density. It also gives the first clean example of a "variance" that is neither noise nor ignorance but ambiguity. Read it before phase D's heteroscedastic heads and ensembles. It is the simplest model in which the uncertainty decompositions used there can be written down exactly.

**Core ideas**

**1. Least squares learns the conditional mean, and only that.** With unlimited data and a flexible network, minimising `E = ½ ∫∫ ‖f(x) − t‖² p(t, x) dt dx` gives `f(x) = ⟨t | x⟩`, the conditional mean (eqs. 4–7). Bishop splits the loss into `½ ∫ ‖f(x) − ⟨t|x⟩‖² p(x) dx` plus `½ ∫ (⟨‖t‖²|x⟩ − ‖⟨t|x⟩‖²) p(x) dx`. Only the first term depends on the weights. The second is the average conditional variance left at the optimum (eqs. 8–9). So a trained MSE network carries two statistics: a mean that depends on `x` and *one global* variance (the residual loss). Equivalently, MSE is Gaussian maximum likelihood with a fixed, input-independent variance. For classification with one-hot targets the conditional mean *is* the vector of class posteriors, which is why cross-entropy and MSE heads can be calibrated at all. For continuous targets the mean is a single summary of `p(t|x)`, and it can be a very poor one.

**2. The failure case: multi-valued inverse problems.** Many physical systems have a single-valued *forward* map (causes → effects) and a multi-valued *inverse* map (desired effect → causes). Bishop's toy example (Section 5) samples `x = t + 0.3 sin(2πt) + ε`, with `ε ~ U(−0.1, 0.1)` and 1,000 points. Regressing `x` on `t` is easy: a 5-hidden-unit MLP fits it well. Regressing `t` on `x` with the *same data* fails. For intermediate `x` there are three valid branches, and the network draws a curve through their average, which lies far from all of the data. "The average of several solutions is not necessarily itself a solution." The fit is also fragile: it is sensitive to architecture and initialisation, and the best result needed careful tuning.

**3. The model.** Replace the single Gaussian by a mixture whose parameters are all outputs of a network:

```
p(t | x) = Σ_{i=1..m} π_i(x) · N(t ; μ_i(x), σ_i(x)² I)          (eqs. 22–23)
π_i = softmax(z^π)_i,   σ_i = exp(z^σ_i),   μ_ik = z^μ_ik         (eqs. 25–27)
E = − Σ_q ln Σ_i π_i(x_q) N(t_q ; μ_i(x_q), σ_i(x_q)² I)           (eq. 29)
```

For a `c`-dimensional target the network has `(c + 2)·m` outputs. The softmax keeps the mixing weights on the simplex. The exponential keeps the widths positive and avoids widths of exactly zero. The loss is a negative log-likelihood, trained by ordinary backpropagation. The gradients have an EM-like form: the posterior responsibility `γ_i = π_i N_i / Σ_j π_j N_j` (eq. 30) of each component for each training point weights its update. Bishop notes that a Gaussian mixture with isotropic components can still approximate any density given enough components. So the model is fully general, even though each component treats the output dimensions as independent. With `m = 1` the MDN reduces to a heteroscedastic Gaussian regressor, and with the width also fixed it reduces to least squares.

**4. Reading the density out.** The predicted mean is `Σ_i π_i μ_i`, which is exactly what least squares would give (eq. 45). The predicted variance (eq. 47, written here for a scalar target) is:

```
s²(x) = Σ_i π_i σ_i²   +   Σ_i π_i (μ_i − Σ_j π_j μ_j)²
        (within-component)  (between-component)
```

This is the law of total variance, with the component label as the conditioning variable. For a single answer, Bishop suggests one of two choices. The first is the centre of the component with the largest peak, `max_i π_i/σ_i^c` (eq. 48), as an approximate mode. The second, which suits mappings with a few distinct branches, is the centre of the component with the largest mass, `max_i π_i` (eq. 49). The second choice is also less sensitive to artificially small `σ_i` in sparse regions.

**5. Results and the open practical issue.** On the toy problem an MDN with 3 kernels, a 20-hidden-unit MLP and 9 outputs recovers the three branches. Figure 7 shows `p(t|x)` with three modes at `x = 0.5`. Using 5 kernels gives almost the same result. Surplus kernels are harmless in practice, because the network can switch them off (`π_i → 0`) or merge them. On the 2-link robot arm, the MDN separates "elbow up" from "elbow down" where both reach the target. Bishop leaves model-order selection (how many kernels) open. Later work, such as the further reading, documents the main practical problems: unstable training, components collapsing onto a few points, and mode collapse.

**What this means for uncertainty.** The two terms in `s²(x)` look like the "aleatoric + epistemic" split that ensembles report (phase D). Here, however, *both* are properties of the data-generating process. The between-component term measures how ambiguous the mapping is, not how ignorant the model is. A single Gaussian head would fold that ambiguity into one large `σ(x)` centred in an empty region. A calibration check on intervals around the mean can then pass even though the point prediction is never correct.

**A small example**

Take a lossless 2-bus line with reactance `X`. The slack bus 1 is held at `V_1 = 1` pu. Bus 2 draws active power `P` and no reactive power. From A04's equations, the active balance gives `P·X = V_2 sin θ` (where `θ` is the angle difference). The reactive balance gives `V_2² = V_2 cos θ`, so `V_2 = cos θ`. Substituting, `P·X = sin θ cos θ = ½ sin 2θ`.

Set `P·X = 0.4` pu. Then `sin 2θ = 0.8`, so `2θ = 53.13°` or `126.87°`. The two roots are:

- `θ = 26.57°`, `V_2 = 0.894` pu (the high-voltage, normal operating point);
- `θ = 63.43°`, `V_2 = 0.447` pu (the low-voltage root, the one a voltage band removes, as in A09).

Suppose a dataset of solved cases contains both roots, for example because Newton–Raphson (A05) was started from different initial points, in a 70 / 30 ratio. A least-squares network then predicts `0.7·0.894 + 0.3·0.447 = 0.760` pu. Check it: `V_2 = 0.760` requires `θ = arccos 0.760 = 40.5°`, and then `½ sin 81° = 0.494 ≠ 0.4`. The averaged prediction violates power balance. A Gaussian-variance head would report a standard deviation of about `√(0.7·0.134² + 0.3·0.313²) = √0.0420 ≈ 0.205` pu around a value that is not a solution. A 2-component MDN with `π = (0.7, 0.3)`, `μ = (0.894, 0.447)`, `σ = 0.01` reports the same mean and almost the same total variance. But nearly all of that variance (0.0420 out of 0.0421) is between-component. Its largest-mass mode (eq. 49) is a valid state. As `P·X → 0.5` (the nose of the curve, maximum loadability), the two components move together and merge, and above 0.5 no solution exists at all.

**How it connects**

**X006** mapped where uncertainty comes from. This entry adds one more source that neither label fits well: a non-unique answer. **A04–A05** gave the power-flow equations and Newton–Raphson. The example above is why NR's starting point matters, and **A09**'s remark that the voltage band removes a spurious root is exactly the "pick a branch" step that an MDN makes explicit. Supervised OPF proxies trained with squared error (**X012**, and phase C's **C01–C04**) predict conditional means. They are safe only where the solution map is single-valued, or where averaging solutions stays feasible, as it does for convex problems such as the DC-OPF of the next focus lesson. Phase D picks this up directly: **D08** (heteroscedastic heads and their pitfalls) is the `m = 1` case. **D09–D11** (ensembles and entropy or variance decompositions) use the same law-of-total-variance split with the ensemble member as the mixture label, and ask whether the between-term really means "epistemic". **D05** (latent-variable models) generalises the discrete mixture label to a continuous latent. In phase F, conditional densities like this one give conformal sets that can be unions of intervals rather than one interval. Next focus lesson: **A11** (DC-OPF, congestion and locational marginal prices).

**Check yourself**

1. A network trained with MSE on one-hot class targets outputs well-calibratable class probabilities. The same loss on a multi-valued continuous target outputs something useless. What single fact explains both?
2. In the 2-bus example, the MDN's between-component variance is large. Would collecting ten times more data, or training an ensemble, make it shrink? What does this say about labelling it "epistemic"?
3. Give two reasons why choosing the output by largest mass `π_i` (eq. 49) can be preferable to choosing it by largest peak `π_i/σ_i` (eq. 48).

<details><summary>Answers</summary>

1. The minimiser of expected squared error is the conditional mean `⟨t|x⟩`. For one-hot targets the conditional mean is the vector of posteriors `P(class | x)`, a complete description of a categorical distribution. For a multi-valued continuous target, the conditional mean of a multimodal `p(t|x)` can lie in a region of zero density, so it is neither a valid answer nor a full description.
2. No. The two roots and their 70 / 30 frequency are properties of how the data were generated, so with more data the MDN only estimates `π` and `μ` more precisely. An ensemble of MDNs would agree on both components, so its between-member spread would be small. The between-*component* term measures ambiguity in the data (aleatoric in the sense of X006, given `x`), not model ignorance. A decomposition that equates "between-mixture-label" with "epistemic" is only right when the label is a *model* index, as in an ensemble or posterior samples.
3. (a) With well-separated branches, `π_i` is the probability mass of a branch, which is what "most probable branch" means. The peak height also rewards narrowness. (b) In sparse regions a component can become artificially narrow, which inflates `π_i/σ_i` without any real support. Bishop notes that eq. (49) is less susceptible to this. (Also: `π_i/σ_i^c` depends on the units of `t`, while the mass does not.)

</details>

---

## 2026-10-26 (Monday) — A10 · Track 02: Optimal power flow: formulations, relaxations, solvers — The AC-OPF written out: variables, objective, constraints, and polar vs rectangular

**Type:** Focus lesson
**Main reference:** [An introduction to optimal power flow: theory, formulation, and examples — S. Frank & S. Rebennack (2016), IIE Transactions 48(12):1172–1197](https://sop.ior.kit.edu/img/An%20introduction%20to%20optimal%20power%20flow%20Theory%20formulation%20and%20examples.pdf) — this time read it as an optimization paper: §2.4 (classic OPF, eqs. 5–11, and the control/state split, p. 1176), §5.2 (branch flow model, eqs. 63–71), §5.4 (PF equations as constraints, incl. Table 2 on polar vs rectangular), §5.5 (side constraints: transformer controls and line limits, eqs. 75–77), §6.2–6.3 (OPF solution methods; degrees vs radians, flat vs warm start) and §7.3 (the fully written-out five-bus OPF and its optimum). Publisher version: [DOI 10.1080/0740817X.2016.1189626](https://doi.org/10.1080/0740817X.2016.1189626).
**Further reading:**
- [MATPOWER User's Manual, Version 8.1 — R. D. Zimmerman & C. E. Murillo-Sánchez (2025)](https://matpower.org/docs/MATPOWER-manual.pdf) — §6.1 "Standard AC OPF" with §6.1.1 (Cartesian vs polar voltage) and §6.1.2 (current vs power balance): the same problem in compact matrix notation, as implemented in a widely used tool.
- [PowerModels.jl documentation, "Mathematical Model" — LANL-ANSI](https://lanl-ansi.github.io/PowerModels.jl/stable/math-model/) — the AC-OPF in complex notation with branch-flow variables, apparent-power and current limits and angle-difference bounds, plus the branch-flow and current-voltage variants side by side.

**Why this matters** — A learned OPF solver is a function approximator for the argmin of a specific nonlinear program. Which variables it outputs, which constraints those outputs must satisfy, and how a violation is measured all depend on how that program is written down. Anyone building or evaluating such a model needs to be able to read an AC-OPF line by line and count its variables, equalities and degrees of freedom.

**Core ideas**

**1. The classic problem.** A09 gave the family tree. Here is the member most papers mean by "AC-OPF" (Frank & Rebennack eqs. 5–11, plus the line limit of §5.5):

```
min    Σ_{i∈G} C_i(P_i^G)                                   C_i quadratic, e.g. c1 P + c2 P²
s.t.   P_i(V, δ) = P_i^G − P_i^L            ∀ i ∈ N         (real power balance)
       Q_i(V, δ) = Q_i^G − Q_i^L            ∀ i ∈ N         (reactive power balance)
       P_i^min ≤ P_i^G ≤ P_i^max,  Q_i^min ≤ Q_i^G ≤ Q_i^max   ∀ i ∈ G
       V_i^min ≤ V_i ≤ V_i^max                ∀ i ∈ N
       P_ik² + Q_ik² ≤ (S_ik^max)²            ∀ (i,k) ∈ L, both ends
       δ_ref = 0
```

Loads `P^L, Q^L` are parameters, the "input" of the problem. Decision variables are generator outputs and the complex voltage at every bus. Everything else (line flows `P_ik, Q_ik`, losses) is a function of the voltages. In ML terms, OPF maps a load vector to an argmin, and a proxy model learns that map.

**2. Counting.** With N buses there are 4N bus quantities (`P, Q, V, δ`) and 2N balance equations, so the system is underdetermined by exactly 2N (§5.1). One angle is fixed as reference. The remaining freedom is what the objective optimizes over, and the box constraints shrink it. Older algorithms split the variables into **controls** `u = (P^G, Q^G)` (plus transformer settings) and **states** `x = (δ_2..δ_N, V_2..V_N)` (§2.4). For fixed `u`, a power-flow solve (A05) returns `x`, so many methods search over `u` and call a power flow at each step. This control/state split is exactly the design choice that learned proxies face: predict only `u` and recover `x` by a power-flow solve, or predict everything and repair violations afterwards.

**3. Three ways to write the balance equations.** §5.1 lists them (eqs. 57–62). With `Y_ik = G_ik + jB_ik`:

```
polar V, rectangular Y:   P_i = V_i Σ_k V_k (G_ik cos δ_ik + B_ik sin δ_ik)
                          Q_i = V_i Σ_k V_k (G_ik sin δ_ik − B_ik cos δ_ik)
rectangular V = E + jF:   P_i = Σ_k G_ik (E_i E_k + F_i F_k) + B_ik (F_i E_k − E_i F_k)
                          Q_i = Σ_k G_ik (F_i E_k − E_i F_k) − B_ik (E_i E_k + F_i F_k)
```

All are exact and give the same feasible set. They differ in what is easy (Table 2):

- **Polar.** Voltage limits are simple bounds `V_min ≤ V_i ≤ V_max`, and a fixed magnitude removes a variable. The price is trigonometric equations whose Hessian changes at every iterate.
- **Rectangular.** The equations are quadratic: no trig, an exact second-order Taylor expansion, a constant Hessian, and the whole OPF becomes a nonconvex QCQP if the side constraints are quadratic too. The price is that voltage limits become functional constraints `V_min² ≤ E_i² + F_i² ≤ V_max²`. The lower one is reverse-convex (A07), and a fixed magnitude becomes a nonconvex equality. Most OPF codes default to polar. The quadratic form is the starting point for the SDP, SOCP and QC relaxations (A20–A22).

**4. Bus injection vs branch flow.** The equations above are the **bus injection model**: compact (4N variables). The **branch flow model** (§5.2, Baran–Wu, Farivar–Low) adds directed flows `P_ik, Q_ik` (and, in Low's version, currents) as explicit variables: 4N + 4L variables and 2N + 4L equations, still with 2N degrees of freedom. It is larger, but the line limits are now constraints on variables rather than on functions, and the form admits relaxations that are exact on radial networks (A21). PowerModels writes even its "bus injection" AC-OPF with branch flow variables `S_ij`, because the thermal limit `|S_ij| ≤ s^u` then reads directly.

**5. Side constraints and the details that bite.** §5.5 adds controllable transformers. A phase shifter `φ_ik` and a tap ratio `T_ik` get bounds (eqs. 75–76), but they also enter `Y`, so the balance equations gain bilinear and trilinear terms. Line limits are naturally on current, `|V_i − V_k| y_ik ≤ I^max` (eq. 77), and are almost always replaced by the apparent-power form above. §6.3 lists practical traps: data files store angles in degrees while solvers use radians (a 180/π error, even in DC models); a local solver from a flat start (`V = 1, δ = 0`) can fail or reach a different local optimum than one warm-started from the current operating point; and decoupled OPF (A09) gives an *inexact* optimum, unlike decoupled power flow. Finally, §2.5 notes that optimal solutions often improve on the starting dispatch by only a few percent. That is a useful scale for judging a proxy's "optimality gap".

**6. The five-bus optimum (§7.3).** Frank & Rebennack write every constraint of a 5-bus, 6-branch case with three generators, one phase shifter `φ_34 ∈ [−30°, 30°]` and one tap changer `T_35 ∈ [0.95, 1.05]`. The optimum has `φ_34 ≈ 12.38°` and `T_35 ≈ 0.95` (at its bound) with cost 0.4016596. Fixing the transformers to their nominal −3.0° and 0.98 raises the cost to 0.4041438, about 0.6% more. The three coordinate forms, solved with SNOPT, MINOS and CONOPT, all reached the same solution, and a global solver verified it as globally optimal. Small, but typical: the controls matter at the sub-percent level, and every form agrees when the problem is well posed.

**A small example**

Two buses joined by a lossless line `y = −j10` (`B_12 = 10`). Bus 1 has a cheap generator (cost `1·P_1`), bus 2 an expensive one (cost `3·P_2`) and a load `1.0 + j0.2` pu. Voltages lie in `[0.95, 1.05]`, `P^G ∈ [0, 2]`, `Q^G ∈ [−1, 1]`, and the line has `S^max = 0.6` at both ends.

- **Variables:** `V_1, V_2, δ_2` (`δ_1 = 0`), `P_1, Q_1, P_2, Q_2`, so 7 variables, 4 balance equations and 3 degrees of freedom.
- **Flows (polar):** `P_12 = 10 V_1 V_2 sin(δ_1 − δ_2)`, `Q_12 = 10 V_1² − 10 V_1 V_2 cos(δ_1 − δ_2)`, `P_21 = −P_12`, `Q_21 = 10 V_2² − 10 V_1 V_2 cos(δ_1 − δ_2)`.
- **Without the line limit** bus 1 serves the whole load at cost 1.0. The voltages are not unique, because nothing in the cost depends on them. A proxy trained on such data sees label noise that is really a flat direction of the optimum.
- **With the limit** (solved numerically with multi-start SLSQP): `V_1 = V_2 = 1.05`, `δ_2 ≈ −3.12°`, `P_12 ≈ 0.5998`, `Q_12 = Q_21 ≈ 0.0163`, `P_2 ≈ 0.4002`, cost ≈ 1.80. Both voltages sit at the *upper* bound. For a given `P_12`, higher voltage means less current, a smaller angle and less reactive power absorbed by the line (`|I|² x = (0.6/1.05)² · 0.1 ≈ 0.0327 = Q_12 + Q_21`), which leaves more of the 0.6 MVA for real power. Relaxing `S^max` by a small `ε` saves about `2ε` (3 − 1, the price gap between the generators). That sensitivity is the dual variable of the line limit, the subject of A11.
- **Rectangular check:** `V_2 = E_2 + jF_2 ≈ 1.0484 − j0.0571`. Then `P_12 = 10(F_1 E_2 − E_1 F_2) = 10 · 1.05 · 0.0571 ≈ 0.600` with no trig, and the voltage band becomes `0.9025 ≤ E_2² + F_2² ≤ 1.1025`, with the upper side active here.

**How it connects**

This lesson writes out the problem named in A09, using A04's equations and A07's limits as its constraint set. A08's economic dispatch is the special case with the network deleted: drop the balance equations per bus, keep one system-wide balance. The control/state split explains the design space of learned proxies (X003, X012): predict the active set, the controls, or the full state, each followed by a different repair step. The bus graph in every sum `Σ_k` is A03's admittance graph, which is why message passing (X004, X013) fits. Next, A11 linearizes this problem to a DC-OPF and reads the multipliers of the balance and line constraints as locational marginal prices. A12 adds contingencies. A13–A15 explain how Ipopt actually solves the program above via its KKT conditions, and A20–A22 relax the rectangular QCQP form.

**Check yourself**

1. In an N-bus bus injection AC-OPF, how many real variables and equality constraints are there before bounds, and why is the branch flow model "underdetermined by the same 2N" despite having 4L more variables?
2. Give two reasons to prefer rectangular voltage coordinates and two to prefer polar coordinates.
3. In the two-bus example, why do both voltages go to their upper bound once the line limit binds, and what would a model predicting only `(P_1, P_2)` need to do to output a complete operating point?

<details><summary>Answers</summary>

1. 4N variables (`P_i, Q_i, V_i, δ_i` at every bus) and 2N balance equations, so 2N degrees of freedom, minus one for the angle reference; loads fix some injections and bounds restrict the rest. The branch flow model adds 4L flow (or current) variables but also 4L defining equations, one per directed real and reactive flow, so the difference stays 2N. It is the same feasible set written with auxiliary variables.
2. Rectangular: the equations are quadratic (no trig, faster evaluation), the Hessian is constant, a second-order Taylor model is exact, and the problem becomes a QCQP that relaxations can attack. Polar: voltage-magnitude limits are simple bounds, and a fixed magnitude eliminates a variable instead of adding a nonconvex equality `E² + F² = V²`. Also, fewer variables in conventional power flow.
3. The binding constraint is apparent power `P² + Q² ≤ 0.36`. Raising the voltages lowers the current and the angle needed for a given `P`, which lowers the reactive power the line absorbs, so more of the rating is available for cheap real power. Nothing in the cost penalizes high voltage, so they hit the upper bound. A model predicting only `(P_1, P_2)` must still choose voltage setpoints (or `Q`) and run a power flow (A05) to recover `V, δ, Q`, then check the line and voltage limits; the predicted dispatch alone does not determine a feasible operating point.

</details>

---

## 2026-10-25 (Sunday) — X014 · Track 08: Uncertainty as input: distributions, sets, embeddings — How wide must a set encoder be? Wagstaff et al. on the limits of sum pooling

**Type:** Cross-phase preview
**Main reference:** [On the Limitations of Representing Functions on Sets — E. Wagstaff, F. B. Fuchs, M. Engelcke, I. Posner & M. A. Osborne (2019), Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97](https://arxiv.org/abs/1901.09006). Read Section 1, Section 2 (Definitions 2.1–2.4 and the two background Theorems 2.8 and 2.9, with the proof sketch of 2.9), Section 3 up to Theorem 3.1, and all of Section 4 (Theorem 4.1, Lemma 4.2 and its proof, Theorems 4.3–4.4, §4.1 Discussion, §4.2 with Figure 3). The rest of Section 3 (Theorems 3.2–3.3) and the appendices are optional. About 8 pages. The link goes to the open arXiv version (v2, October 2019).
**Further reading:**
- [Universal Approximation of Functions on Sets — E. Wagstaff, F. B. Fuchs, M. Engelcke, M. A. Osborne & I. Posner (2022), JMLR 23(151):1–56](https://jmlr.org/papers/v23/21-0730.html): the journal follow-up. It shows that a latent space one dimension too small can make some piecewise-affine functions no better approximable than a constant, and places Deep Sets inside Janossy pooling.

**Why this matters** — A forecast that arrives as M scenarios, an ensemble of M weather runs, or M samples from a generative model is an unordered set. Anyone who wants a network to *read* such a set, not just its mean, has to choose a pooling operation and a latent width. This paper gives the clean answer for sum pooling: to represent every continuous function of a set of M scalars, the space you sum in needs at least M dimensions, and M is also enough.

**Core ideas**

**The model.** The architecture behind Deep Sets and PointNet is

```
f(X) = ρ( Σ_{x ∈ X} φ(x) ),    φ : R → R^N,   ρ : R^N → R
```

Each element is encoded on its own by `φ`, the codes are summed (which makes the result permutation-invariant), and `ρ` decodes the sum. The paper calls `f` *sum-decomposable via R^N*, and *continuously* sum-decomposable when `φ` and `ρ` can be chosen continuous. The latent width `N` is the number we care about. (The paper's theorems treat sets of scalars. The journal follow-up treats the general case.)

**What was known (Section 2).** You already know the countable case from GIN: if the elements come from a countable universe, enumerate it with `c(x) ∈ N` and set `φ(x) = 4^(−c(x))`. Every finite set gets a distinct base-4 expansion, so the sum is injective and a one-dimensional latent space suffices (Theorem 2.8, from Zaheer et al.). For real-valued elements Zaheer et al. used power sums. For a set of size M in `[0,1]`,

```
Φ_q(X) = Σ_{m=1}^{M} x_m^q,    q = 0, …, M
```

is continuous and injective with a continuous inverse, so any continuous symmetric `f` equals `ρ ∘ Φ` with `ρ = f ∘ Φ⁻¹` (Theorem 2.9, latent width M+1).

**Continuity is the whole story (Section 3).** The countable trick hides a problem. Take the rationals as the universe. Theorem 3.1 shows there are functions on finite sets of rationals for which *every* sum-decomposition through R has a `φ` that is discontinuous at every point. A neural network cannot represent that, so Theorem 2.8 says nothing about what a trained network can do. The useful question is about continuous `φ` and `ρ` on real inputs.

**The necessity result (Theorem 4.1).** For `M > N` there are continuous symmetric functions `f : R^M → R` that are not continuously sum-decomposable via `R^N`. The witness is a function you would actually want: `max`. The proof has two steps.

- *Lemma 4.2.* If `max(X) = ρ(Σ_x φ(x))` holds for all finite sets, then the sum map `Φ_M` restricted to sets of size M is injective. The proof is by induction: equal sums give equal maxima, so remove the maximum and recurse.
- *Topology.* Restrict `Φ_M` to sorted vectors `x_1 < … < x_M`. That is an open convex subset of `R^M`, so it is homeomorphic to `R^M`. Now `Φ_M` would be a continuous injection from `R^M` into `R^N` with `N < M`, and no such map exists.

**The sufficiency result (Theorems 4.3–4.4).** The power-sum construction can be tightened to width exactly M, even when set sizes vary up to M. So:

```
continuous symmetric f on sets of ≤ M reals  ⇔  ρ(Σ φ(x)) with φ, ρ continuous and N = M
```

N = M is necessary in the worst case and sufficient in all cases.

**What it does not say (§4.1).** It does not say every function needs width M. The mean needs one dimension, and the sum and count need one each. The theorem is a statement about *arbitrary* targets. Real tasks are often simpler, and the data can be redundant.

**The experiment (§4.2).** The authors train `ρ(Σφ)` with large MLPs to predict the median of sets of size M (15 to 500) drawn from uniform, Gaussian or Gamma distributions, and sweep N. Three findings: the error falls monotonically in N; it plateaus beyond a "critical" N; and the critical N grows roughly linearly with M (Figure 3b). The critical points sit *below* N = M, because the network learns to estimate the median of the training distributions, not an exact algorithm. That gap is a lesson in its own right: a set encoder trained on one family of input distributions may be using a compressed code that fails on another.

**A small example**

Take sets of two numbers, `X = {x_1, x_2}`, and sum in `R^2` with `φ(x) = (x, x²)`:

```
s_1 = x_1 + x_2,   s_2 = x_1² + x_2²
```

For `X = {1, 3}`: `s_1 = 4`, `s_2 = 10`. Recover the set from the elementary symmetric polynomials: `e_1 = s_1 = 4` and `e_2 = (s_1² − s_2)/2 = 3`. The elements are the roots of `t² − 4t + 3 = (t − 1)(t − 3)`. So `ρ` can compute anything, for example the maximum:

```
max = ( s_1 + sqrt(2 s_2 − s_1²) ) / 2 = ( 4 + sqrt(20 − 16) ) / 2 = 3
```

using `2 s_2 − s_1² = (x_1 − x_2)²`. This `ρ` is continuous.

Now try width 1. A continuous `φ : R → R` would make `(x_1, x_2) ↦ φ(x_1) + φ(x_2)`, on the open half-plane `x_1 < x_2`, a continuous injection from a 2-D region into a line. Picture squashing a sheet onto a wire without two points ever landing on the same spot: removing one point from a line disconnects it, but removing one point from a plane does not. So no continuous width-1 sum encoder recovers the max of two numbers. Read "two numbers" as "two scenarios of tomorrow's wind output" and the same holds: an encoder narrower than the number of scenarios cannot, in general, recover a quantity as basic as the worst case.

**How it connects**

The obvious link is **X005**. A kernel mean embedding `μ_P = E_{x∼P}[k(x, ·)]` is exactly a sum (mean) decomposition with an infinite-dimensional `φ(x) = k(x, ·)`, which is why a characteristic kernel can be injective on distributions. As soon as you replace the RKHS with a finite feature map (random features, a learned encoder), this paper's width bound applies. Note one difference: the mean `(1/M) Σ φ(x)` forgets the set size and multiplicities, which a sum keeps. It also connects to **X004** and **X013**: one message-passing step is a sum decomposition over a neighbourhood, so a GNN layer with continuous node features and width below the neighbourhood size cannot realize every continuous aggregation. GIN's injectivity argument lives in the countable case, the one Section 3 warns about. For scenario inputs, it complements **X002** and **X011**: once forecasts are sets of samples, how the model reads them becomes an architecture question.

Phase D picks this up directly. **D02** develops kernel mean embeddings and distribution regression, **D03** the Deep Sets theorem that this paper refines, **D04** attention pooling in the Set Transformer (pairwise interactions *before* pooling, one way to buy expressivity without enormous N), and **D06** asks how to represent a forecast distribution as an input to an OPF learner. The next focus lesson is **A10**, the AC-OPF problem written out in full.

**Check yourself**

1. Why does the countable-universe result `φ(x) = 4^(−c(x))` give no useful guidance for a neural set encoder on real-valued inputs?
2. In the proof of Theorem 4.1, why is it enough to restrict to sorted vectors `x_1 < … < x_M`, and what does `max` contribute to the argument?
3. The median experiment finds critical widths *below* M. Does this contradict Theorem 4.1? What risk does it point to when such an encoder is deployed on input distributions it was not trained on?

<details><summary>Answers</summary>

1. That construction relies on enumerating the universe. On the rationals, Theorem 3.1 shows that some functions force a sum-decomposition via R whose `φ` is discontinuous at every point. Networks (and GPs) represent continuous functions, so a statement relying on such a `φ` says nothing about what a trained encoder can represent. For real inputs, the meaningful question is continuous decomposability, and there the width must grow with M.
2. Sorted vectors give each set of M distinct reals exactly once, and they form an open convex region homeomorphic to `R^M`. `max` forces injectivity: Lemma 4.2 shows that if `ρ(Σφ)` computes the max for all set sizes, then equal sums imply equal sets, by peeling off the maximum and inducting. A continuous injection from an `R^M`-like region into `R^N` with `N < M` cannot exist (invariance of domain), which is the contradiction.
3. No. The theorem is a worst case over all continuous functions and all inputs. The network learned the median only for inputs drawn from the training distributions, where it can use redundancy and need not be injective. The risk is that this compressed code is distribution-specific: on a different input family (another weather regime, another scenario generator) the information discarded below width M may be exactly what is needed, and the error can rise without any warning sign at the encoder.

</details>

---

## 2026-10-24 (Saturday) — X013 · Track 07: GNNs and spatio-temporal GNNs for power grids — Does the grid graph help a learned OPF? A controlled test of locality

**Type:** Cross-phase preview
**Main reference:** [Leveraging Power Grid Topology in Machine Learning Assisted Optimal Power Flow — T. Falconer & L. Mones (2023), IEEE Transactions on Power Systems 38(3):2234–2246, doi:10.1109/TPWRS.2022.3187218](https://arxiv.org/abs/2110.00306). Read Section I-A (the two ways ML is used in OPF), Section II-D (how the grid becomes a graph and which architectures are compared), Section III-B and III-C (fixed vs varying topology) and Section III-D with Fig. 5 (the locality analysis). Skim Section II-B/C (the "estimator of an OPF operator" framing). The link goes to the open arXiv preprint (v3, April 2022).
**Further reading:**
- [Topology-aware Graph Neural Networks for Learning Feasible and Adaptive AC-OPF Solutions — S. Liu, C. Wu & H. Zhu (2023), IEEE TPWRS](https://arxiv.org/abs/2205.10129): the counterpoint. It argues that OPF solutions *do* have exploitable graph structure (through the LMPs) and builds a GNN around it. Read it after this entry and decide which locality claim you believe.

**Why this matters** — The default assumption in learned OPF is "the grid is a graph, therefore use a GNN". That is an inductive-bias claim, and like any inductive-bias claim it can be tested. This paper tests it in a controlled way: same data, matched parameter counts, many grid sizes. The answer is uncomfortable and useful: on a fixed grid, the graph barely helps, because the OPF solution map is not local. Anyone designing a GNN for OPF should know this result before choosing an architecture.

**Core ideas**

**The problem being learned.** Optimal power flow (**A09**) chooses generator outputs and voltage set-points that serve all loads at minimum cost while respecting the power-flow physics (**A04**) and the operating limits (**A07**). For a fixed grid, the only thing that changes from one hour to the next is the load vector `d = (P_d, Q_d)`. So the solver defines a function

```
F : d  ↦  y*(d) = argmin_y  cost(y)   s.t.  power-flow equations(y, d),  limits(y)
```

where `y` collects generator active powers `P_g` and voltage magnitudes `|V_g|`. The paper treats every ML-for-OPF method as an estimator of `F` (or of a related operator). Two tasks are studied:

1. **Regression (end-to-end or warm start):** predict `y*` directly, trained with MSE. The prediction can be used as is, or handed to the interior-point solver as a starting point (**A18** will explain why that is harder than it sounds).
2. **Classification (reduced problem):** predict which inequality constraints are *active* (binding) at the optimum, a multi-label problem trained with binary cross-entropy. Solve a smaller OPF with only those constraints, then check the full problem and add back any violated ones. This is the active-set idea from **X003**.

**The architectures.** All models see the same inputs and are trained on the same data.
- *FCNN* (MLP): the concatenated load vector, `x ∈ ℝ^(2|L|)`, one or three hidden layers.
- *CNN*: the inputs reshaped into a 2-D tensor and convolved in Euclidean space, a deliberately "wrong" inductive bias as a control.
- *GNN*: five convolution types from PyTorch Geometric (ChebConv with `K = 4`, GCNConv, GraphConv, GATConv, SplineConv). In the graph, each generator and each load is its own node, and devices at the same bus are connected. Edges can be weighted by the line admittance, `e_ij = exp(−k log|Z_ij|) = |Z_ij|^(−k)`, with `k = 0` giving a plain adjacency matrix.
- Each GNN comes in a *local* version (graph convolutions only, every output depends only on a few-hop neighbourhood) and a *global* version (convolutions followed by a fully connected *readout* over all nodes).

Channel widths are chosen so that the models have similar parameter counts. Data: 11 PGLib grids (**A24**) from 24 to 2853 buses; 10 000 feasible samples per grid, each load scaled independently by a factor from `U(0.8, 1.2)`; labels from PowerModels.jl + Ipopt; results are means with 95 % confidence intervals over 10 training runs.

**Finding 1: on a fixed grid, the MLP is as good or better.** For set-point regression, the plain FCNN is close to the best GNN on small grids and beats all GNNs on the two largest. On 2853-sdet, the three-layer FCNN reaches a test MSE of 5.99 ± 0.16 (×10⁻³) against 8.49 ± 0.41 for the best GNN. For active-set classification the authors call the single-layer FCNN "even more dominant". The FCNN also trains at least an order of magnitude faster than the global GNNs.

**Finding 2: local GNNs fail; the readout does the work.** Removing the readout makes every GNN much worse. Adding it back restores performance. If the answer were local, a few rounds of message passing would be enough and the readout would add little. The authors read this as direct evidence that each generator's optimal output depends on loads all over the grid.

**Finding 3: the locality probe.** Section III-D measures locality directly. Starting from 100 load configurations, they perturb one load by 1 %, re-solve, and record the relative change in every generator's active power, voltage set-point and every bus's LMP, plotted against the *hop distance* between the perturbed load and the output (Fig. 5). There is "little evidence" that nearby outputs react more than distant ones. In large grids, generators also tend to sit far from loads, so a local neighbourhood often contains no generator at all.

**Why the map is not local: the power balance.** This is the first-principles core, and it comes straight from **A08**. Total generation must match total load. In an uncongested, lossless network, economic dispatch sets every generator's marginal cost equal to one system price `λ`, so the optimal response to an extra megawatt depends on *how much* load there is, not *where* it is. Mathematically, `∂P_g*/∂d_j` is the same for every load bus `j`. That is the opposite of a local function. Congestion (**A11**) breaks the uniformity, but the new sensitivities come from PTDFs (**A06**), which are dense: a line limit couples buses that are many hops apart. In ML terms, the target behaves like a function of a *global sum* of the inputs, and a depth-3 message-passing network has a 3-hop receptive field. It cannot compute a global sum on a 300-bus graph without a pooling/readout layer, and once you add one, you have effectively rebuilt an MLP.

**Finding 4: when the topology changes, GNNs win clearly.** In the second experiment, one random transmission line is removed per sample (an N−1 contingency, **A12**). Now the ChebConv GNN is the best model on 7 of 9 grids for regression and 8 of 9 for classification; for example, on 24-ieee-rts the classification loss is 1.44 ± 0.11 (×10⁻²) for ChebConv versus 3.56 ± 0.36 for the three-layer FCNN. One caveat when you read the tables: as described, the FCNN input is the load vector only, so it is not told which line is out, while the GNN sees the changed graph. Part of the gap is therefore an *information* gap, not purely an inductive-bias gap. The honest summary: the graph matters as an **input encoding of topology**, not as a locality prior.

**What the paper does not measure.** Feasibility and optimality of the predictions are left out (the authors say so), and the warm-start variant is described but not timed. Neither the physics violations nor the solver speed-up, which **F16** and **F19** will insist on, are reported.

**A small example**

Two generators on a chain of buses 1–2–…–10: G1 at bus 1, G2 at bus 10, lossless and uncongested. Costs `C1 = 10 P1 + 0.05 P1²` and `C2 = 12 P2 + 0.05 P2²` (€/h, `P` in MW). Equal incremental cost (**A08**):

```
10 + 0.1 P1 = 12 + 0.1 P2 = λ,     P1 + P2 = D
```

With total load `D = 100 MW`: `P1 = 60`, `P2 = 40`, `λ = 16 €/MWh`.

Now add 1 MW of load. Whether it appears at bus 2 (next to G1) or at bus 9 (next to G2, seven hops from G1), the answer is the same: `P1 = 60.5`, `P2 = 40.5`, `λ = 16.05`. The sensitivity matrix `∂P_g/∂d_j` has identical columns. A 2-layer GNN at G1 never sees bus 9 and therefore cannot respond to it; an MLP that only needs `Σ_j d_j` learns this map with a single hidden unit.

Now give the line 3–4 a 30 MW limit and put most of the load beyond it. Once the line saturates, G1 can no longer push power past it: extra load at buses 4–10 must come from G2 alone, while extra load at bus 2 still comes from G1. The two columns of `∂P_g/∂d_j` now differ, depending on which side of the congested line each load sits. The map becomes piecewise linear with a *topology-dependent* switch. Remove a line, and the switch moves. That second picture is where encoding the graph pays off.

**How it connects**

This entry builds on **A08** (one system price explains why the solution map is global), **A09** (the OPF problem being learned) and **X003** (the active-set classification task, here used as the second benchmark). It is the natural sequel to **X004**: there, a GNN beat an MLP on *power flow* with randomized line parameters, and failed on unseen grids. Here, on *OPF* with fixed parameters, the MLP is enough, and the GNN's advantage appears only when the topology varies. Both papers point to the same lesson: the graph is valuable when the structure changes. Coming up in phase A, **A06**'s PTDFs and **A11**'s congestion pricing make the "dense sensitivity" argument exact, and **A12** introduces the N−1 setting where GNNs did win. In phase C, **C14** (over-squashing and long-range limits) gives the theory behind "local GNNs fail", **C17** returns to GNN OPF proxies and topology transfer (including the Liu, Wu & Zhu counterpoint), and **F20** asks for exactly this kind of ablation. Next focus lesson: **A10**.

**Check yourself**

1. Why does removing the readout layer hurt the GNNs so much, and what does that say about the OPF solution map?
2. In a lossless, uncongested grid, what is `∂P_g*/∂d_j` as a function of the load bus `j`, and why does that make a hop-distance locality prior useless?
3. In the N−1 experiment the GNN beats the MLP by a large margin. Give one reason this is not purely evidence for the graph *inductive bias*, and describe a fairer MLP baseline.

<details><summary>Answers</summary>

1. Without a readout, each generator's prediction is a function of the loads within a few hops only. The optimal set-points depend on loads across the whole grid (through the global power balance and dense congestion sensitivities), so a local model is missing most of the relevant inputs. The readout restores global access, which shows that the map is not local; the useful computation is global, much as in an MLP.
2. It is the same for every `j`: equal incremental cost fixes one system price, and the extra megawatt is shared among generators according to their cost curvature, regardless of where it appears. So the output responds to the *sum* of the loads, and nearby loads matter no more than distant ones. A prior that says "nearby inputs matter more" points at the wrong structure.
3. As described, the MLP receives only the load vector, so it is not told which line was removed. It is missing information that the GNN gets from the graph. A fairer baseline appends a line-status vector (one binary entry per branch) or the flattened admittance entries to the MLP input. Then any remaining gap can be attributed to the architecture (weight sharing, permutation structure) rather than to the input.

</details>

---

## 2026-10-23 (Friday) — A09 · Track 02: Optimal power flow: formulations, relaxations, solvers — Fifty years of OPF: what the problem is, and three ways to write it down

**Type:** Focus lesson
**Main reference:** [History of Optimal Power Flow and Formulations (Optimal Power Flow Paper 1) — M. B. Cain, R. P. O'Neill & A. Castillo (Dec. 2012, rev. Aug. 2013), FERC staff technical paper](https://www.ferc.gov/sites/default/files/2020-05/acopf-1-history-formulation-testing.pdf) — read §1 (Introduction, incl. Table 1), §2 (History, incl. Table 2 of problem types), §5 (ACOPF formulations, incl. the comparison table at the end) and §7 (Conclusions). Skim §3–4 (notation, admittance matrix, constraint types). §6 (literature on current-injection formulations) is optional. About 36 pages, open access.
**Further reading:**
- [The IV Formulation and Linear Approximations of the AC Optimal Power Flow Problem (OPF Paper 2) — R. P. O'Neill, A. Castillo & M. B. Cain (2012), FERC staff paper](https://www.ferc.gov/sites/default/files/2020-05/acopf-2-iv-linearization.pdf) — the sequel. It develops the current-voltage formulation from §5 and its linear approximations in detail.
- [Convex Relaxation of Optimal Power Flow, Part I: Formulations and Equivalence — S. H. Low (2014), IEEE Trans. Control of Network Systems (arXiv:1405.0766)](https://arxiv.org/abs/1405.0766) — a more mathematical tour of the same territory. Its bus-injection versus branch-flow split is the vocabulary that the relaxation lessons A20–A21 use.

**Why this matters** — "OPF" is not one problem. It is a family of problems that share a name, and papers in the area quietly differ in objective, constraint set and variable choice. Anyone who trains or benchmarks a learned OPF solver needs to know which member of the family they are approximating, because the choice of formulation decides which outputs a model predicts, which constraints it can violate, and how comparable its reported numbers are.

**Core ideas**

**1. Three problems that are easy to confuse.** The paper's Table 2 separates three classical problem types. The rest of the taxonomy hangs off them:

- **Power flow (load flow)** (A04–A05). Given injections, solve the network equations for the voltages. It is a square nonlinear system with no objective. A solution satisfies the physics, but it need not respect any limits, and it is not optimal for anything.
- **Economic dispatch** (A08). Find the cheapest generation that meets demand (plus reserves). The network is ignored or simplified.
- **Optimal power flow.** Minimize an objective subject to the full power-flow equations *and* operating limits (A07). This is economic dispatch made to obey physics, or power flow given degrees of freedom and a goal.

Within OPF the variants are named by what they keep. **ACOPF** keeps the exact AC equations and optimizes real and reactive power together. **DCOPF** fixes `|V| = 1` and linearizes angles (A06). **Decoupled OPF** alternates between a `P–θ` subproblem and a `Q–|V|` subproblem. **SCED** (security-constrained economic dispatch), the workhorse of real markets, usually has only thermal limits and linear *proxies* for voltage limits. **SCOPF** adds contingency constraints (A12).

**2. A short history: the problem was fixed early, and the solvers improved slowly.**

- **Before computers.** From 1929, power flow was solved on analog network analyzers. In the 1930s economic dispatch was solved by hand or with special slide rules using equal incremental loading (A08), and a run could take up to 8 hours.
- **1956.** Ward and Hale publish the first automated digital power-flow solution.
- **1960s.** Newton–Raphson becomes standard once Tinney and others develop sparse-matrix techniques for the admittance matrix.
- **1962.** Carpentier states optimality conditions for dispatch with the full AC equations and variable bounds, based on the Kuhn–Tucker conditions. This is generally treated as the first complete OPF formulation. Carpentier assumed "suitable convexity", which the authors call a big assumption (see A16).
- **1987.** Stott et al. warn that OPF is *not* just "power flow plus an objective". Once you add degrees of freedom, formulations that look harmless can be badly posed, with objectives, controls and constraints that conflict. One example is minimizing losses while letting generators' real power vary, which pulls the dispatch away from minimum cost.
- **1991.** Huneault and Galiana survey more than 300 OPF papers and summarize the field as the application of ever stronger optimization tools to a problem well defined since the early 1960s. The methods tried (gradient, LP, QP, penalty) all remain slow and prone to ill-conditioning.

The paper's diagnosis as of 2012 is that 50 years after Carpentier there was still no fast, robust solver for the full ACOPF. Markets therefore solve linearized and decoupled approximations, with operators stepping in for voltage problems the models miss. The authors put a size on the stakes: 2009 US production cost was about USD 112 billion per year at 30 USD/MWh or USD 372 billion at 100 USD/MWh, so a 5% efficiency gain is worth roughly USD 6–19 billion per year. They also list the time budgets that ISO software must meet: weekly problems in 8 hours, daily in 2 hours, hourly in 15 minutes, five-minute dispatch in 1 minute, and post-contingency corrective actions in 30 seconds. These budgets are the reason learned and warm-started solvers attract interest at all.

**3. Objectives.** The literature uses minimum generation cost, maximum market surplus (equivalent to minimum cost when demand is fixed), minimum losses, minimum total generation, and maximum transfer. The authors argue that for economic dispatch the correct objective is minimum cost under a full constraint set. In that setting, lower losses and fewer control actions come out as by-products, and minimizing losses is "inconsistent with economic principles". The usual cost is `c(P) = Σ_n (c1_n p_n + c2_n p_n²)`.

**4. Bus types go away in optimization.** Power flow needs PQ, PV and slack buses (A04) so that the number of equations equals the number of unknowns. An optimization problem has no such requirement. Carpentier already noted in 1962 that with voltage and angle as decision variables, the fixed reference voltage can be dropped. Fixing the slack voltage at 1.0 p.u. when generators may operate anywhere in [0.95, 1.05] can only make the solution worse. In OPF a "PV bus" is just a bus with a generator and some bounds, and `P^min = P^max` encodes a fixed injection. (An angle reference is still needed to remove the rotational symmetry `θ → θ + c`, but it costs nothing.)

**5. The three formulations.** Write `V = V^r + jV^j` (rectangular) or `|V|∠θ` (polar), let `•` be the elementwise product and `Y = G + jB`.

```
Polar PQV          P_n = Σ_m |V_n||V_m| (G_nm cos θ_nm + B_nm sin θ_nm)
                   Q_n = Σ_m |V_n||V_m| (G_nm sin θ_nm − B_nm cos θ_nm)
                   V^min ≤ |V| ≤ V^max,  θ^min_nm ≤ θ_n − θ_m ≤ θ^max_nm    (linear)

Rectangular PQV    P + jQ = V • (Y V)*                 (2N quadratic equalities)
                   (V^min)² ≤ V^r•V^r + V^j•V^j ≤ (V^max)²
                   angle limits: θ^min ≤ atan(v^j_n/v^r_n) − atan(v^j_m/v^r_m) ≤ θ^max

Rectangular IV     I = Y V                             (2N LINEAR equalities)
                   P = V^r•I^r + V^j•I^j,  Q = V^j•I^r − V^r•I^j   (local, bilinear)
                   plus the same voltage and current-magnitude bounds
```

All three describe the same feasible set. They place the nonconvexity in different spots:

| | Polar PQV | Rectangular PQV | Rectangular IV |
|---|---|---|---|
| Network-wide equations | trigonometric, nonlinear | quadratic | **linear** (`I = YV`) |
| Per-bus constraints | linear boxes | quadratic, some nonconvex | bilinear/quadratic, some nonconvex |
| Variables per bus | 4 (`P, Q, |V|, θ`) | 4 | 6 (`P, Q, V^r, V^j, I^r, I^j`) |

The paper's hypothesis is that the IV form may be easier to solve because everything coupling the whole network is linear and the nonlinearity is local to each bus and line. Note the source of nonconvexity that never goes away: the lower voltage bound `(V^min)² ≤ (v^r)² + (v^j)²` is the outside of a disk, an annulus constraint, and no change of variables makes it convex. A modelling note from §4: limits on lines can be stated as current, apparent power, real power or angle difference, and these are *surrogates* of one another with different approximation errors. Two test cases that share a network but state their line limits differently are different problems.

**A small example**

Two buses joined by a lossless line with `x = 0.1` p.u., so `y = −j10` and `Y = [[−j10, j10], [j10, −j10]]`. Bus 1 holds `v_1 = 1 + j0`. Bus 2 withdraws `P_2 = −0.5` p.u. (50 MW on a 100 MVA base) with `Q_2 = 0`. Write `v_2 = v^r + jv^j`.

- **IV form.** `i_2 = j10 v_1 − j10 v_2 = 10v^j + j10(1 − v^r)`. This is linear in the voltage. The injections are bilinear: `P_2 = v^r i^r + v^j i^j = 10 v^j` and `Q_2 = v^j i^r − v^r i^j = 10(v^r² + v^j² − v^r)`.
- **Solve.** `P_2 = −0.5` gives `v^j = −0.05`. `Q_2 = 0` gives `v^r² − v^r + 0.0025 = 0`, so `v^r = (1 ± √0.99)/2`.
  - Root 1: `v^r ≈ 0.9975`, `|V_2| ≈ 0.9987`, `θ_2 ≈ −2.87°`. This is the normal operating point.
  - Root 2: `v^r ≈ 0.0025`, `|V_2| ≈ 0.050`. This is a mathematically valid but useless low-voltage solution of the same equations.
- **Polar check.** `P_2 = 10|V_2| sin θ_2 = 10 · 0.9987 · sin(−2.87°) ≈ −0.50`. ✓
- **DC approximation (A06).** `P_2 = 10 θ_2` gives `θ_2 = −0.05 rad ≈ −2.86°` with `|V_2| = 1`. The angle is almost right, the 0.13% voltage drop is missed, and the low-voltage root does not exist in the DC model.

The voltage band `0.95² ≤ v^r² + v^j²` is exactly what removes root 2. This is why that nonconvex constraint cannot simply be dropped.

**How it connects**

This lesson places the earlier pieces in one picture. A04–A05 are the power-flow row of Table 2. A06 is the DCOPF row. A07 supplies the inequality constraints. A08 is the economic-dispatch row, and its Lagrange multiplier comes back as a nodal price once the network is added. The paper also restates A03's observation that `G` and `B` are weighted graph Laplacians when there are no transformers or shunts. That is why a GNN over the bus graph is a natural fit (X004). The formulation also decides what a learned proxy (X003, X012) outputs. Predicting `(|V|, θ)` and predicting `(V^r, V^j)` give different loss geometry and different ways of violating constraints. Next: A10 writes out the full AC-OPF in polar and rectangular form (Frank & Rebennack), A11 derives LMPs from DCOPF, and A12 adds security constraints. The claims about nonconvexity and multiple solutions are made precise in A13 (KKT) and A16 (local optima, NP-hardness), and A19–A23 turn the rectangular quadratic form into convex relaxations.

**Check yourself**

1. Why does a power flow need a slack bus while an OPF does not? What does the OPF still need to fix, and why?
2. In the rectangular IV formulation, which constraints are nonconvex? Can a cleverer choice of variables remove all of them?
3. Two papers both report results on "ACOPF" for the same network: one uses current-magnitude line limits and minimum losses, the other apparent-power limits and minimum cost. Why are their numbers not comparable, and what does Stott et al.'s warning say about the first setup?

<details><summary>Answers</summary>

1. Power flow is a square system: `2N` equations need exactly `2N` unknowns, so at each bus two of the four quantities are fixed. One bus must absorb the losses, which are not known in advance, and serve as the angle reference. An OPF is an optimization with more variables than equality constraints, so injections and voltages can all move within bounds. Nothing has to be fixed except a single angle reference, which removes the symmetry `θ → θ + c`. That symmetry does not change the cost, but it makes the solution non-unique.
2. The network equations `I = YV` are linear. The injection definitions `P = V^r•I^r + V^j•I^j` and `Q = ...` are bilinear equalities, which are nonconvex. The lower voltage bound `(V^min)² ≤ |V|²` is a reverse-convex (annulus) constraint. The upper voltage and current-magnitude bounds are convex. No change of variables removes the nonconvexity entirely: the same injections can be met by several separated voltage solutions, as the two roots in the example show, and a convex set containing both would also contain the points between them, which violate the physics. Relaxations (A19–A22) give up exactness instead.
3. Different line-limit surrogates define different feasible sets, and different objectives define different optima, so costs, losses and feasibility rates measure different things. On the first setup, Stott et al. would call minimizing losses with real-power generation as a control badly posed for dispatch, because it moves away from the minimum-cost solution. Cain et al. likewise argue that minimum cost is the economically consistent objective. Benchmarks such as PGLib-OPF (A24) exist to fix one convention.

</details>

---

## 2026-10-22 (Thursday) — X012 · Track 06: Learning to optimize, OPF proxies, decision-focused learning — Proxies you can trust by construction: Van Hentenryck's "Optimization Learning"

**Type:** Cross-phase preview
**Main reference:** [Optimization Learning — P. Van Hentenryck (2025), arXiv:2501.03443 [math.OC]](https://arxiv.org/abs/2501.03443) — read Sections 1–2 (the learning task and the two proxy definitions), Section 3 (the economic-dispatch model, Model 1), Section 5.2 (dedicated repair layers, eqs. 5–6) and Section 6 up to Model 2 (dual proxies and the LP completion layer). Then read Section 5.5 (real-time risk assessment) and skim Tables 2–5 and Section 7 (primal-dual learning for security-constrained OPF). It is about 22 pages plus references, written up from the author's EURO-2024 keynote. The link goes to the open arXiv version (v1, January 2025). No journal version is listed.
**Further reading:**
- [Dual Conic Proxies for AC Optimal Power Flow — G. Qiu, M. Tanneau & P. Van Hentenryck (2024), PSCC 2024, arXiv:2310.02969](https://arxiv.org/abs/2310.02969) — the first dual proxy, for the SOCP relaxation of AC-OPF. Its completion layer uses the structure of the dual optimum to fill in the variables the network does not predict.
- [Dual Lagrangian Learning for Conic Optimization — M. Tanneau & P. Van Hentenryck (2024), arXiv:2402.03086](https://arxiv.org/abs/2402.03086) — generalizes the "predict some duals, complete the rest so every output is dual-feasible" idea to general conic programs.

**Why this matters** — A neural network that maps loads to generator setpoints is fast but almost never exactly feasible, and it gives no hint of how far from optimal it is. Anyone building learned OPF solvers has to decide how to fix both problems. This essay gives a compact, opinionated answer: build the network so its output is feasible *by construction* (primal proxy), and learn a second network whose output is a *certified* lower bound (dual proxy). Reading it now gives the vocabulary that phase C makes precise.

**Core ideas**

**The learning task.** Grid operators solve the same parametric problem again and again (every five minutes in US real-time markets) with slightly different inputs `x` (loads, costs, limits):

```
Φ(x) = argmin_y f_x(y)   s.t.   h_x(y) = 0,  g_x(y) ≥ 0
```

The instances come from a distribution `P` estimated from history and forecasts. The paper asks for two learned maps:

- a **primal proxy** `Φ↑` whose output is feasible for every `x ~ P`: `h_x(Φ↑(x)) = 0` and `g_x(Φ↑(x)) ≥ 0`;
- a **dual proxy** `Φ↓` whose output gives a valid lower bound on the optimal cost `f_x(Φ(x))`.

Taken together they give, in milliseconds, a feasible decision and a certificate of how good it is. That is the learned version of a solver's primal–dual pair with its duality gap.

**Why plain regression is not enough (Section 4).** Supervised learning (solve instances offline, regress `y` on `x`) gives a function approximator, not a proxy. A regressor's outputs are only approximately right, so they violate constraints. Adding Lagrangian penalty terms `λ|h_x(ŷ)| + ν max(0, −g_x(ŷ))` and updating `λ, ν` by subgradient steps reduces violations. But the multipliers are shared across all instances, and the paper notes that on unseen instances this "almost always violates constraints".

**Primal proxies = network + repair layer (Section 5).** The proxy is `R ∘ M_θ`. Here `M_θ` is a neural network, and `R` maps its raw output into the feasible set. The generic choice is a projection `R(ŷ) = argmin_y ‖y − ŷ‖` over the feasible set. You differentiate it with the implicit function theorem applied to its KKT conditions (the OptNet / implicit-layer idea, **C09–C10**). That works but is expensive. The paper's main point is that cheap, problem-specific repair layers often exist. For economic dispatch (a single power-balance equation, as in **A08**), a sigmoid enforces generator bounds, and then all generators are scaled *proportionally* toward their upper or lower limits until supply equals demand:

```
if 1ᵀp̂ < 1ᵀd:  p̃ = (1 − ζ↑) p̂ + ζ↑ p̄,   ζ↑ = (1ᵀd − 1ᵀp̂) / (1ᵀp̄ − 1ᵀp̂)
else:          p̃ = (1 − ζ↓) p̂ + ζ↓ p̲,   ζ↓ = (1ᵀp̂ − 1ᵀd) / (1ᵀp̂ − 1ᵀp̲)
```

Each `p̃` is a convex combination of points inside the generator boxes, so it stays inside the bounds. It is differentiable almost everywhere, so the whole pipeline (called E2ELR) trains end to end. A second layer of the same kind repairs the reserve constraint. Power engineers will recognize this as the participation-factor idea from automatic generation control, turned into a neural-network layer.

**Self-supervised training.** Because the optimization problem is known in closed form, you don't need solver labels. You minimize the problem's own objective through the repair layer:

```
min_θ (1/n) Σ_i f_{x_i}(Φ↑_θ(x_i))
```

The loss is the operating cost (plus penalties on soft thermal limits), and feasibility is enforced by the architecture. Supervised and self-supervised training take comparable time. Self-supervised training skips the offline solves (65.9 CPU-hours for the 30,000-bus case) and often reaches smaller gaps.

**Dual proxies = network + completion layer (Section 6).** Consider an LP with bounded variables, `min cᵀy` subject to `Ay = b` and `l ≤ y ≤ u`. Its dual has the variables `(z, z_l, z_u)`, the constraints `zA + z_l − z_u = c` and `z_l, z_u ≥ 0`, and the objective `bᵀz + lᵀz_l − uᵀz_u`. The trick: predict only `ẑ` (one multiplier per equality constraint, i.e. the prices). Then set the bound multipliers from the reduced costs:

```
z_l = max(0, c − ẑA),   z_u = max(0, −(c − ẑA))
```

Any `ẑ` gives a dual-feasible point, so by weak duality its objective is a valid lower bound. Train by *maximizing* that bound. The network can be arbitrarily wrong and the certificate is still valid. Only how tight it is depends on learning.

**Primal-dual learning (Section 7).** Here a second network predicts instance-specific multipliers and the two networks are trained by mimicking the augmented Lagrangian method. This is lesson **C05**. The showcase is security-constrained DC-OPF under N-1 contingencies. The full formulation is a mixed-integer program with up to 64.9 million continuous variables. On the largest case, the state-of-the-art decomposition algorithm needs at least 4,648 seconds per instance. The proxy answers in about 10 ms.

**Findings worth remembering.** On PGLib cases with 300–30,000 buses, E2ELR's mean optimality gaps (which include penalties for violations) are below 2 % in every case and 0.19–0.33 % on the three largest grids with self-supervised training. The plain network never satisfies power balance, and the DeepOPF and DC3 baselines satisfy it on between 1 % and 100 % of instances, depending on the grid. A batch of 256 instances takes about 5–12 ms on a GPU. Gurobi on one CPU needs 12.6 ms to 4.7 s *per instance*. In a Monte-Carlo risk dashboard (288 dispatches per 24-hour scenario), the solver takes about 15 minutes per scenario and the proxy about 5 seconds. Proxies without feasibility guarantees report many *false* constraint violations there, so the risk estimate itself goes wrong. The dual proxy on DC-OPF reaches geometric-mean gaps of 0.23–0.50 %. Read these numbers with phase F in mind: the setting is GPU batch inference against a single-thread CPU solver, and loads are perturbed synthetically (a uniform global scaling in [0.8, 1.2] times 5 % log-normal noise). **C11**, **F16** and **F17** look at both choices critically.

**A small example**

**Repair.** Two generators, `p̲ = (0, 0)`, `p̄ = (100, 50)` MW, demand `d = 120`. The network outputs `p̂ = (80, 20)`, which falls 20 MW short. Then `ζ↑ = (120 − 100)/(150 − 100) = 0.4`, and `p̃ = 0.6·(80, 20) + 0.4·(100, 50) = (88, 32)`. That sums to 120 and is within the bounds. Each generator takes 40 % of its remaining headroom.

**Certificate.** Costs `c = (10, 30)` €/MWh. The true optimum dispatches the cheap unit fully, `(100, 20)`, at a cost of 1,600 €. Suppose the dual network predicts the price `ẑ = 25`. The reduced costs are `c − ẑ·(1, 1) = (−15, 5)`, so `z_u = (15, 0)` and `z_l = (0, 5)`. The bound is `120·25 + 0 − 100·15 = 1,500 €`, which is valid with a 6.25 % gap. With `ẑ = 30` (the true marginal price, set by the expensive unit) the reduced costs are `(−20, 0)` and the bound is `3,600 − 2,000 = 1,600 €`, which is exact. So learning the dual means learning the price from **A08**.

**How it connects**

This entry continues **X003**, which kept an exact solver behind a learned guess (active sets). Here the safeguard is built into the network instead: a repair layer for feasibility and duality for quality. **X009** made the same "learn inside a certified procedure" point for MILPs, and **X007** (Lagaris) built constraints into the architecture for differential equations, the same move as a repair layer. The dual proxy is **A08**'s Lagrange multiplier as a price, now predicted per instance, and the PTDF constraints in the paper's models come from **A06**. Coming up: **A11** (DC-OPF duals as locational prices), **A12** (security-constrained OPF) and **A13** (KKT, duality) give the theory used here. In phase C, **C01** frames amortized optimization, **C04** is the shared-multiplier Lagrangian method criticized in Section 4, **C05** is primal-dual learning, **C06** (DC3) is the iterative-repair alternative, and **C09–C10** are implicit layers. **F16** reads the E2ELR paper itself, with an eye on honest speed-up accounting. Next focus lesson: **A09**, the history and taxonomy of OPF formulations.

**Check yourself**

1. Why does a dual proxy give a *valid* lower bound even when its network is badly trained, while a primal proxy without a repair layer gives no feasibility guarantee at all?
2. In the risk-assessment application, why do proxies that only approximately satisfy power balance produce a *biased* risk estimate rather than just a noisier one?
3. The proportional repair layer guarantees feasibility for economic dispatch. What property of the feasible set makes such a cheap layer possible, and why is the same trick much harder for AC-OPF?

<details><summary>Answers</summary>

1. The completion layer turns any prediction `ẑ` into a point that satisfies every dual constraint exactly (the free bound multipliers absorb the residual `c − ẑA`). By weak duality, the objective of any dual-feasible point is at most the primal optimum. Training affects only how tight the bound is. A raw primal network solves a regression problem whose outputs are only approximately right, and nothing maps them onto the constraint surface, so equalities such as power balance hold only by luck.
2. Every slightly infeasible output is counted as an adverse event (an imbalance or overload) even when the true optimal dispatch has none. These errors all point the same way, toward more violations, so averaging over Monte-Carlo scenarios does not cancel them. The paper's figures show such false positives throughout the day, while the feasible proxy matches the solver's ground truth.
3. The ED feasible set is a box intersected with one hyperplane (plus a reserve constraint). Moving toward a box corner is a convex combination that stays in the box and changes total output monotonically, so a single scalar `ζ` restores balance in closed form. AC power flow adds nonconvex equality constraints (the power-flow equations from **A04**). The feasible set is a curved manifold without such a simple parametrization, so repair needs an iterative solve (Newton–Raphson, **A05**) or an implicit layer. That is why DC3-style and power-flow-based repairs exist (**C06**, **E16**).

</details>

---

## 2026-10-21 (Wednesday) — A08 · Track 02: Optimal power flow: formulations, relaxations, solvers — Economic dispatch: equal incremental cost and the Lagrange multiplier as a price

**Type:** Focus lesson
**Main reference:** [Power Generation, Operation, and Control, 3rd ed. — A. J. Wood, B. F. Wollenberg & G. B. Sheblé (2014), Wiley](https://www.wiley-vch.de/en/areas-interest/engineering/power-generation-operation-and-control-978-0-471-79055-6) — read Chapter 3, "Economic Dispatch of Thermal Units and Methods of Solution": §3.1 (the problem and its Lagrangian), §3.4 (lambda iteration), §3.8 (base point and participation factors) and §3.9 (network losses). Skim §3.2–3.3 (piecewise-linear costs, LP) and §3.10 (the idea of the locational marginal price). Appendix 3A, "Optimization Within Constraints", is a compact review of Lagrange multipliers and inequality constraints. The book is not open access. Library or e-book access is usually available.
**Further reading:**
- [ECEN 615 Lecture 20: Economic Dispatch, Optimal Power Flow — T. Overbye, Texas A&M University (Fall 2019), lecture slides](https://overbye.engr.tamu.edu/wp-content/uploads/sites/146/2019/11/ECEN615_Fall2019_Lect21.pdf) — free slides that follow the book. Slides 3–5 give the Lagrangian with losses and penalty factors, and the later slides move from one "area lambda" to bus-by-bus marginal prices.

**Why this matters** — Economic dispatch is the smallest optimization problem in power systems that already contains the main ideas of OPF: a cost to minimize, a balance constraint, box limits and a dual variable that turns out to be the electricity price. Anyone who trains a model to output dispatch decisions or to predict prices is learning an approximation of this problem's primal or dual solution, so its structure tells you what the model has to get right.

**Core ideas**

**1. The problem.** A set of `N` generating units is already switched on. Choosing which units are on is *unit commitment*, a later and harder problem. The units must jointly supply a demand `P_D`. Unit `i` produces `P_i` MW at a cost rate `C_i(P_i)` in $/h. This comes from its fuel-input curve (heat rate times fuel price) and is usually fitted as a convex quadratic, `C_i(P) = a_i + b_i P + c_i P²`. Ignoring the network (one "copper plate" bus, no losses), the problem is

```
min_P  Σ_i C_i(P_i)
s.t.   Σ_i P_i = P_D                       (power balance, multiplier λ)
       P_i^min ≤ P_i ≤ P_i^max              (A07's generator limits, multipliers μ_i^-, μ_i^+)
```

This is a separable convex problem with one coupling constraint. In ML terms it is a resource-allocation problem with the same structure as water-filling: each unit has its own cost and a single budget constraint ties them together.

**2. The optimality condition: equal incremental cost.** The Lagrangian is `L = Σ_i C_i(P_i) + λ (P_D − Σ_i P_i)`, and Appendix 3A adds the inequality multipliers. Setting `∂L/∂P_i = 0` and adding complementary slackness gives the KKT conditions:

```
dC_i/dP_i = λ      if P_i^min < P_i < P_i^max
dC_i/dP_i ≤ λ      if P_i = P_i^max
dC_i/dP_i ≥ λ      if P_i = P_i^min
```

`dC_i/dP_i` is the unit's *incremental* (marginal) cost in $/MWh. At the optimum, every unit that is not at a limit runs at the same marginal cost `λ`. The intuition is an exchange argument. If unit 1 had marginal cost 15 and unit 2 had 18, moving 1 MW from unit 2 to unit 1 would save about 3 $/h while still meeting the balance. A unit at its upper limit is "cheaper than the price but full". A unit at its lower limit is "dearer than the price but cannot go lower". Because each `C_i` is convex, the KKT conditions are both necessary and sufficient.

**3. λ is a price.** By the envelope theorem, the multiplier of the balance constraint is the sensitivity of the optimal cost to the right-hand side: `λ = d C*(P_D) / d P_D`. It is the cost of serving one more MWh of demand, the *system marginal cost*. In a market, this is the clearing price that every supplier receives. It is not anyone's average cost, and a cheap unit at its upper limit earns `λ − dC_i/dP_i` per MWh above its marginal cost. That margin is exactly its upper-limit multiplier `μ_i^+`. This is the first appearance of an idea that runs through the whole curriculum: **dual variables are prices**, and you can read economic meaning off an optimization solution.

**4. Solving it: lambda iteration.** Fix `λ`, and each unit solves its own one-dimensional problem: `P_i(λ) = clip((dC_i/dP)^{-1}(λ), P_i^min, P_i^max)`. For a quadratic cost this is `P_i(λ) = clip((λ − b_i)/(2 c_i), P_i^min, P_i^max)`. The total output `Σ_i P_i(λ)` is non-decreasing in `λ`, so a bisection on `λ` until `Σ_i P_i(λ) = P_D` solves the problem (§3.4–3.5). This is *dual decomposition*, and it is the same idea that makes the Lagrangian methods in constrained RL work: a single scalar price coordinates many independent agents. §3.2–3.3 cover piecewise-linear costs, which are what real markets use. Each unit then offers a *staircase* of blocks, and dispatch fills them from cheapest to most expensive (the *merit order*). `λ` becomes the price of the marginal block. When demand lands exactly on a block boundary, `λ` is not unique: any value in an interval is a valid dual, so it is a subgradient rather than a gradient. This matters for learning, because the price-as-a-function-of-demand map is a step function.

**5. Base points and participation factors (§3.8).** Demand changes every few seconds, and nobody re-solves the dispatch each time. Differentiate the optimality conditions around a solution (a "base point"). For units off their limits, `ΔP_i = (1 / C_i'') · Δλ`, so a demand change `ΔP_D` is split as

```
ΔP_i = ( (1/C_i'') / Σ_j (1/C_j'') ) · ΔP_D
```

These fractions are the **participation factors**. Units with flatter marginal-cost curves absorb more of the change. This linear recourse rule returns, almost unchanged, as the *affine balancing policy* of chance-constrained OPF in phase B.

**6. Losses (§3.9).** Once network losses `P_L(P)` enter the balance, the stationarity condition becomes `dC_i/dP_i = λ (1 − ∂P_L/∂P_i)`. Define the penalty factor `L_i = 1 / (1 − ∂P_L/∂P_i)`, and the rule becomes `L_i · dC_i/dP_i = λ`. A unit whose extra output raises losses is penalized, so the "price" seen at different locations stops being equal. Add line limits (A07) and the single `λ` breaks into one price per bus. This is the locational marginal price that A11 derives from DC-OPF.

**A small example**

Three units, costs `C_i = b_i P + c_i P²`, so the marginal cost is `b_i + 2c_i P`:

| Unit | b_i ($/MWh) | c_i ($/MW²h) | Marginal cost | Limits (MW) |
|---|---|---|---|---|
| 1 | 10 | 0.010 | 10 + 0.02 P | [50, 400] |
| 2 | 12 | 0.020 | 12 + 0.04 P | [50, 300] |
| 3 | 14 | 0.005 | 14 + 0.01 P | [50, 250] |

Unclipped responses: `P_1 = 50(λ − 10)`, `P_2 = 25(λ − 12)`, `P_3 = 100(λ − 14)`, with a total of `175λ − 2200`.

- **Demand 600 MW.** `175λ − 2200 = 600` gives `λ = 16 $/MWh`, so `P = (300, 100, 200)`, all within limits. All three units run at marginal cost 16. Unit 3 has the highest intercept but the flattest curve, so it takes a large share.
- **Demand 700 MW.** The unclipped solution is `λ = 2900/175 ≈ 16.57`, but then `P_3 ≈ 257 > 250`. Clip `P_3 = 250` and re-solve units 1 and 2 for the remaining 450 MW: `75λ − 800 = 450`, so `λ ≈ 16.67`, with `P_1 ≈ 333.3` and `P_2 ≈ 116.7`. Unit 3's marginal cost at its limit is `16.5 ≤ λ`, so its upper-limit multiplier is `μ_3^+ ≈ 0.17 $/MWh`. One binding limit raised the price for everyone, and the stretch from 600 to 700 MW was served at a rising marginal cost.
- **Participation factors at 600 MW.** `1/C_i'' = 1/(2c_i) = 50, 25, 100`, so a small demand change splits 2/7, 1/7 and 4/7 among the three units. These factors are only valid until the next limit binds. At 700 MW unit 3's factor has dropped to zero, and units 1 and 2 split changes 2/3 and 1/3.

**How it connects**

A07 defined the generator limits that become the inequality constraints here. A06's DC model and A03's network are exactly what this lesson leaves out, and adding them back gives DC-OPF (A11), where `λ` splits into locational prices. Which units sit at a limit is the *active set*, and predicting it is what X003's active-set learning does. Robust and chance-constrained versions of this problem (X010, X001's policy classes) start from this same structure. Next: A09 surveys the history and formulations of OPF, A10 writes out the full AC-OPF, A13 formalizes KKT conditions and duality, and B07–B08 reuse §3.8's participation factors as affine recourse policies.

**Check yourself**

1. In the 700 MW case, demand rises by 1 MW. By how much does the total cost increase, and which units supply the extra megawatt?
2. A learned model predicts dispatch for the three-unit system and reports a 1% average error in `P`. Why might its implied price (the marginal cost of a non-limited unit) still be badly wrong near 690–710 MW?
3. With piecewise-linear (staircase) offers, why is `λ` sometimes not unique, and what does that mean for a model trained to regress price from demand?

<details><summary>Answers</summary>

1. About `λ ≈ 16.67 $` per hour for that extra MW. Unit 3 is at its upper limit, so units 1 and 2 supply the increase in the ratio of their participation factors, `50 : 25`, which is about 0.67 MW and 0.33 MW.
2. The price is decided by *which* units are at their limits. Near 700 MW, unit 3 switches from free to limited, which changes the slope of the price curve (from `1/175` to `1/75` $/MWh per MW). A small error in `P_3` near 250 MW can place the model on the wrong side of this switch, giving the wrong active set and a wrong marginal unit. Small primal error does not imply small dual error.
3. When demand lands exactly at the end of one block, the cost function is not differentiable there, and any price between the last accepted block's offer and the next block's offer satisfies the optimality conditions (the subdifferential is an interval). Price as a function of demand is a step function with jumps, so a regression model with squared loss will smooth across the steps, and solvers may report different but equally valid duals. Training labels then depend on the solver's tie-breaking, and evaluation should allow for it.

</details>

---

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

---

## 2026-10-19 (Monday) — A07 · Track 01: Grid physics & power flow — Operating limits: thermal ratings, voltage bands and generator capability

**Type:** Focus lesson
**Main reference:** [An introduction to optimal power flow: theory, formulation, and examples — S. Frank & S. Rebennack (2016), IIE Transactions 48(12):1172–1197](https://sop.ior.kit.edu/img/An%20introduction%20to%20optimal%20power%20flow%20Theory%20formulation%20and%20examples.pdf) — re-read §2.4 for the classic OPF box constraints (eqs. 8–11, pp. 1175–1176), then read §5.5 "Side constraints" (eqs. 75–77 and the `P² + Q² ≤ S_max²` simplification, p. 1184). Skim the bound constraints of §3.1–3.3 (eqs. 16–22, 28–33) to see that the same limits appear in every OPF variant. Publisher version: [DOI 10.1080/0740817X.2016.1189626](https://doi.org/10.1080/0740817X.2016.1189626).
**Further reading:**
- [MATPOWER User's Manual, v8.1 — R. D. Zimmerman & C. E. Murillo-Sánchez (2025)](https://matpower.org/docs/MATPOWER-manual.pdf) — §6.1 (eqs. 6.9–6.15: branch limits as `S`, `P` or `I`, set by the `RATE_A` column) and §6.4.3–6.4.4 (trapezoidal generator P–Q capability curves, angle-difference limits). This is how the limits look in the test cases most ML papers use.
- [Dynamic Line Rating: Report to Congress — U.S. Department of Energy (June 2019)](https://www.energy.gov/sites/prod/files/2019/08/f66/Congressional_DLR_Report_June2019_final_508_0.pdf) — pp. 10–11, "Thermal Limits of Transmission Lines", "Static Line Ratings" and "Dynamic Line Rating": where a line rating physically comes from, and why it is really a weather-dependent quantity.

**Why this matters** — The power-flow equations (A04) say which states are physically possible. The operating limits say which of them an operator may actually use, and in practice almost every hard decision comes from a limit that binds. Anyone who trains a learned OPF proxy and reports "constraint violation" needs to know what each limit protects, how it is written in the data, and why the same physical limit can be written in ways that disagree.

**Core ideas**

**1. Three families of limits.** Frank & Rebennack's classic OPF (eqs. 5–11) minimizes generation cost subject to the power-flow equations (6)–(7) and box constraints:

```
P_i^G,min ≤ P_i^G ≤ P_i^G,max      (8)    generator active power
Q_i^G,min ≤ Q_i^G ≤ Q_i^G,max      (9)    generator reactive power
V_i^min   ≤ V_i   ≤ V_i^max        (10)   bus voltage magnitude
δ_i^min   ≤ δ_i   ≤ δ_i^max        (11)   bus voltage angle
```

§5.5 adds branch limits and bounds on controllable transformer taps and phase shifts (eqs. 75–76). Group them by what they protect: **branches** (thermal), **buses** (voltage) and **generators** (capability). In ML terms, (8)–(9) bound the *control* variables and are easy to enforce, for example with a sigmoid output layer. Voltage and flow limits constrain *state* variables, which are only defined implicitly through the power-flow equations. That is why a learned model satisfies the first kind easily and violates the second kind.

**2. Thermal limits are limits on current.** A conductor heats up as `I² R`. As it heats, it expands and sags towards the ground and trees, and above some temperature it ages permanently. The DOE report defines the thermal limit as the "maximum current carrying capacity at a given voltage", computed from a heat balance with assumed ambient temperature, wind speed and solar radiation (the IEEE 738 standard). So the physical constraint is on `|I_ik|`. Frank & Rebennack write it exactly as

```
|I_ik| = |V_i − V_k| · |y_ik| ≤ I_ik^max
⇔ (V_i cos δ_i − V_k cos δ_k)² + (V_i sin δ_i − V_k sin δ_k)² ≤ (I_ik^max / |y_ik|)²     (77)
```

(this uses only the series admittance; for shunt charging and off-nominal taps, see §5.5 and MATPOWER eq. 3.9). They then note that most formulations instead bound the apparent power, `P_ik² + Q_ik² ≤ (S_ik^max)²`, and call this simplification "sufficiently accurate" in almost all cases. The two are linked by `|S_ik| = V_i |I_ik|`. An MVA limit equals a current limit only at `V_i = 1` pu. Below nominal voltage, an MVA rating lets through *more* current than the conductor allows, so the error is not on the safe side. MATPOWER (eq. 6.11) lets you choose `S`, `P` or `I` with the same number `RATE_A`, so "the limit" in a benchmark depends on a solver option. Three more details:
- *Both ends.* Losses and line charging make the flow at the "from" end differ from the flow at the "to" end, so the limit is imposed at each end (MATPOWER eqs. 6.9–6.10).
- *Several ratings.* Branch data carries a long-term rating (`RATE_A`) plus short-term and emergency ratings (`RATE_B`, `RATE_C`). After a contingency, operators may temporarily use the higher ones. Frank & Rebennack's SCED section (§3.1) explains this relaxation: the operation is brief, and the system is reconfigured straight away.
- *Ratings are uncertain inputs.* A static rating assumes fixed, conservative weather. Dynamic line rating replaces it with a time-varying, weather-dependent value. On a cool, windy day the true limit is higher, and on a hot, still day it can be lower than the static rating. This makes the right-hand side of the constraint a forecast quantity too.

**3. Voltage bands.** Each bus must stay inside a band, typically `[0.95, 1.05]` pu or `[0.9, 1.1]` pu in test cases. The upper bound protects insulation and customer equipment. The lower bound protects equipment that misbehaves at low voltage. It also keeps the system away from the "nose" of A04's PV curve, where the high-voltage power-flow solution meets the low-voltage one and voltage collapse begins. Low voltage also raises the current for a given power, so voltage and thermal limits interact through `|S| = V|I|`. A side remark from Frank & Rebennack's Table 2 is worth knowing: in polar coordinates `V_min ≤ V_i` is a simple bound, but in rectangular coordinates it becomes `E_i² + F_i² ≥ V_min²`. This is a *reverse-convex* constraint, the region outside a disc, which makes it one of the sources of non-convexity that the convex relaxations of A19–A22 must deal with. Angle bounds (11), and especially angle-*difference* limits across a branch, serve as a rough stand-in for stability limits (MATPOWER §6.4.4).

**4. Generator capability is not a box.** Constraints (8)–(9) treat P and Q as independent. A real synchronous machine has a curved P–Q region:
- *Armature (stator) current limit:* `P² + Q² ≤ S_rated²`, a circle.
- *Field (rotor) current limit:* this caps how much reactive power the machine can produce when over-excited. It is another circle, not centred at the origin.
- *Under-excitation and stability limit:* this caps reactive *absorption*.
- *Prime-mover limit:* `P ≤ P^max` (turbine, or the wind and sun available).

The consequence is that maximum P and maximum Q are not available at the same time. MATPOWER approximates the curve by cutting the box with two sloped lines through the points `(p1, q1min/max)` and `(p2, q2min/max)` (§6.4.3). Inverter-based resources (solar, wind, batteries) have the same tradeoff from the inverter's current rating, `P² + Q² ≤ S_inv²`, with `P` capped by the available wind or sun. When P comes from a forecast, the generator's reactive range also becomes uncertain.

**5. The geometry, for later.** `P² + Q² ≤ S²` is a convex second-order cone in `(P, Q)`. It is not convex in `(V, δ)`, because `P` and `Q` are trigonometric functions of the state. Most of the difficulty in OPF comes from this: convex limits on outputs, composed with a non-convex physics map.

**A small example**

A line has a thermal rating of `I^max = 1.0` pu, and the dataset stores it as `S^max = 1.0` pu (it was converted at nominal voltage). It carries `P = 0.8` and `Q = 0.5` pu, so `|S| = √(0.64 + 0.25) = 0.943` pu.
- At `V = 1.00` pu, `|I| = 0.943`. Both formulations report 94 % loading.
- At `V = 0.92` pu (an under-voltage, stressed operating point), `|I| = 0.943 / 0.92 = 1.025`. The MVA constraint still reports 94 % loading, but the conductor is 2.5 % above its rating.

Now take a generator with `S_rated = 1.0` pu, box limits `P ∈ [0, 1]`, `Q ∈ [−0.4, 0.6]`, and the armature circle. At `P = 0.8` the circle allows at most `Q = √(1 − 0.64) = 0.6`, so the box is tight there. At `P = 0.95`, the circle gives `Q ≤ √(1 − 0.9025) = 0.31`, while the box still allows 0.6. A dispatch that asks for full active output and full voltage support at the same time is box-feasible and physically impossible. MATPOWER's sloped cut with `(p1, q1max) = (0, 0.6)` and `(p2, q2max) = (1, 0.3)` gives `Q ≤ 0.6 − 0.3P`, which is 0.315 at `P = 0.95`. The linear cut approximates the circle well here.

**How it connects**

A02 introduced apparent power and per-unit, and here `|S| = V|I|` is what separates an MVA limit from a current limit. A04's PQ/PV bus types assume a PV generator can supply whatever Q it needs. The `Q` limits here are why real solvers switch a PV bus to PQ once its reactive limit is reached. A05's failed Newton runs and A06's reactive-driven congestion table are examples of these limits binding: A06's last row was a current/MVA overload that the DC model could not see. On the uncertainty side, X008's overload loss and X010's robust constraints are written against exactly these thermal limits, and dynamic line rating turns the limit itself into a forecast. Next, A08 adds costs and gets economic dispatch with the Lagrange multiplier as a price. A10 then writes the full AC-OPF, with the power-flow equations and every limit from this lesson as constraints.

**Check yourself**

1. A benchmark reports that a learned OPF proxy has "zero thermal violations", checked with `P² + Q² ≤ RATE_A²`. Under what operating conditions could the same dispatch still overload the conductor, and why?
2. Why do learned OPF models usually satisfy generator bounds but violate voltage and flow limits? Name one architectural trick that fixes the first kind and explain why it does not carry over to the second.
3. A wind farm's inverter is rated `S = 1.0` pu, and tomorrow's forecast says available power is `P = 0.9 ± 0.1` pu. What reactive range can the operator count on, and what does this imply for voltage-control planning?

<details><summary>Answers</summary>

1. When terminal voltages are below 1 pu. The physical constraint is on current, `|I| = |S|/V`, so at `V < 1` an MVA-feasible flow can exceed the current rating (in the example, 94 % in MVA corresponds to 102.5 % in current at `V = 0.92`). The check is also wrong if the line has a lower dynamic rating than `RATE_A` that day, or if it was evaluated only at one end of the branch.
2. Generator limits bound the model's outputs directly, so a bounded activation (scaled sigmoid, clipping) satisfies them by construction. Voltages and flows are states defined implicitly through the non-linear power-flow equations, so they are not outputs you can clip independently. Bounding them needs a power-flow solve or projection, a penalty, or a correction layer (phase E).
3. The circle gives `|Q| ≤ √(1 − P²)`. At `P = 0.8` this is 0.6, and at `P = 1.0` it is 0, so the guaranteed range over the forecast interval is only `|Q| ≤ 0` unless the operator curtails P. More practically, the operator should plan with the worst case (`P = 1.0`) or reserve inverter headroom by capping P. The reactive range is uncertain because the active output is, so voltage control from inverter-based resources needs either curtailment or a chance-constrained or robust treatment (phase B).

</details>

---

## 2026-10-18 (Sunday) — X010 · Track 04: Stochastic, chance-constrained, risk-averse & robust optimization — Robust optimization with a dial: Bertsimas & Sim's "price of robustness"

**Type:** Cross-phase preview
**Main reference:** [The Price of Robustness — D. Bertsimas & M. Sim (2004), Operations Research 52(1):35–53](https://www.mit.edu/~dbertsim/papers/Robust%20Optimization/The%20price%20of%20Robustness.pdf) — read §1–3 (about 4 pages: the uncertainty model, Soyster's worst-case counterpart (2), the budgeted counterpart (4), Proposition 1 and Theorem 1, which give the linear reformulation (7)). Then read §4 up to Theorem 2 and Table 1, and §6.1 (robust knapsack). Skim the proof of Theorem 3, §4.2 and §5. The link goes to the authors' copy of the published paper. The DOI is [10.1287/opre.1030.0065](https://doi.org/10.1287/opre.1030.0065).
**Further reading:**
- [Theory and Applications of Robust Optimization — D. Bertsimas, D. B. Brown & C. Caramanis (2011), SIAM Review 53(3):464–501](https://arxiv.org/abs/1010.5445) — the standard survey: which uncertainty sets give tractable counterparts, how robust optimization relates to chance constraints, and adaptable (multistage) models. Use it as a map, not to read end to end.
- [Adaptive Robust Optimization for the Security Constrained Unit Commitment Problem — D. Bertsimas, E. Litvinov, X. A. Sun, J. Zhao & T. Zheng (2013), IEEE Transactions on Power Systems 28(1):52–63](https://dspace.mit.edu/handle/1721.1/68971) — budgeted uncertainty sets for nodal net injections in a two-stage unit-commitment model, tested on the ISO New England system. It shows the idea from this entry at grid scale.

**Why this matters** — A probability distribution of forecast errors is not always available, or not always trusted. Robust optimization instead asks for a *set* of possible data and requires the decision to stay feasible for every member of that set. The obvious set (every parameter at its worst at once) is absurdly pessimistic. This paper shows how to pick a smaller set with one interpretable knob, and still keep a linear program and a probabilistic guarantee. Anyone building learned or classical dispatch tools under forecast uncertainty will meet the budgeted uncertainty set again and again: in robust OPF, robust unit commitment, and as the target that learned "robust" solvers are measured against.

**Core ideas**

*Setting.* Start from a nominal LP, `max cᵀx  s.t. Ax ≤ b, l ≤ x ≤ u`. Uncertainty sits in the entries of `A`. A cost vector can be moved into the constraints with an epigraph variable, `max z s.t. z − cᵀx ≤ 0`. For row `i`, let `J_i` be the set of uncertain coefficients. Each uncertain coefficient is a random variable `ã_ij ∈ [a_ij − â_ij, a_ij + â_ij]` with an *unknown but symmetric* distribution. In normalized form, `η_ij = (ã_ij − a_ij)/â_ij ∈ [−1, 1]`. ML analogy: this is a bounded, symmetric perturbation model, the same shape as an ℓ∞ threat model in adversarial robustness. The difference is that here we optimize a *decision* against it, not a classifier.

*Two extremes.* **Soyster (1973)** protects against every coefficient moving to its worst value at once:

```
Σ_j a_ij x_j + Σ_{j∈J_i} â_ij |x_j| ≤ b_i        (Soyster: Γ_i = |J_i|)
```

This is always feasible, and very conservative. Ben-Tal & Nemirovski use ellipsoidal sets, which are much less conservative but turn the LP into a second-order cone program. That is fine for continuous problems, but awkward for MILPs.

*The budget.* Bertsimas & Sim add a per-row parameter `Γ_i ∈ [0, |J_i|]`. The intuition is that nature will not push *all* coefficients to their extremes at once, so the decision is protected against the worst case in which at most `⌊Γ_i⌋` coefficients move fully and one more moves by the fraction `Γ_i − ⌊Γ_i⌋`. The **protection function** is

```
β_i(x, Γ_i) = max_{S_i ∪ {t_i}: S_i ⊆ J_i, |S_i| = ⌊Γ_i⌋, t_i ∈ J_i\S_i}
              Σ_{j∈S_i} â_ij |x_j| + (Γ_i − ⌊Γ_i⌋) â_it_i |x_t_i|
```

The robust constraint is `Σ_j a_ij x_j + β_i(x, Γ_i) ≤ b_i`. `Γ_i = 0` gives back the nominal problem, and `Γ_i = |J_i|` gives back Soyster. The dial in between is what the authors call the "price of robustness".

*The trick that keeps it linear (Proposition 1, Theorem 1).* For a fixed `x`, the inner maximization over subsets is the LP `max Σ_j â_ij |x_j| z_ij  s.t. Σ_j z_ij ≤ Γ_i, 0 ≤ z_ij ≤ 1`. It is a fractional knapsack with an integral optimum. By LP duality its value equals `min Γ_i z_i + Σ_j p_ij  s.t. z_i + p_ij ≥ â_ij |x_j|, p, z ≥ 0`. Because this is a *minimization*, you can paste it into the outer problem, and the whole robust counterpart becomes one LP (eq. 7):

```
max cᵀx
s.t. Σ_j a_ij x_j + Γ_i z_i + Σ_{j∈J_i} p_ij ≤ b_i      ∀i
     z_i + p_ij ≥ â_ij y_j,   −y_j ≤ x_j ≤ y_j,   p, y, z ≥ 0
```

This adds `k + 1` variables and `k + n` constraints, where `k = Σ_i |J_i|`, and it keeps the sparsity of `A`. If `x` has integer components, the counterpart is still a MILP of the same type. This "dualize the inner adversary" move is the most reusable idea in robust optimization. You will see it again for polyhedral uncertainty sets, in adjustable robust optimization (B08), and in Wasserstein DRO (B12).

*The guarantee (Theorem 2).* Suppose the `η_ij` are independent and symmetric on `[−1, 1]`, with nothing else assumed about their distribution. Then the robust solution violates row `i` with probability at most

```
P(violation of row i) ≤ exp(−Γ_i² / (2|J_i|))
```

This is a Hoeffding-type bound. Theorem 3 tightens it with an exact binomial-tail bound, which the authors show is the best possible. They also give a normal approximation, `≈ 1 − Φ((Γ_i − 1)/√|J_i|)`. The consequence is that the protection you need grows like `√|J_i|`, not like `|J_i|`. Table 1 (violation probability below 1%) shows this: with `|J_i| = 200` the tight bound needs `Γ_i ≈ 33.9`, about 17% of the uncertain coefficients. The Hoeffding bound needs 42.9. With `|J_i| = 2000` the tight bound needs about 105. With `|J_i| = 5`, full (Soyster) protection is needed anyway. This is a concentration-of-measure argument, the same mechanism as in a generalization bound. Many small independent errors rarely all line up.

*Findings (§6.1).* Take a random 0–1 knapsack problem with 200 items, capacity 4,000, and weights uncertain by ±10%. The nominal optimum is 5,592. Full Soyster protection lowers it by 5.5%, to 5,283. A guarantee of at most 0.57% violation probability costs only 1.54% of the objective. The robust MILPs solved in 0.05–50 s with CPLEX 6.0. The portfolio experiment (§6.2) shows the dial from another side. At `Γ = 0` the portfolio holds only the stock with the highest expected return. It diversifies as `Γ` grows to 17, stays fully diversified up to 41, and above 41 it jumps abruptly to the single stock that the Soyster solution picks.

*Caveats to keep in mind.* The guarantee needs independence and symmetry. Correlated errors, such as wind farms behind one weather front, break it. §5 handles correlation only through a linear factor model. The bound is also distribution-free, so for well-behaved errors it is conservative. And `Γ` protects each row separately. It says nothing about the probability that *some* row fails, which is the joint-versus-individual chance-constraint question that phase B takes up.

**A small example**

One transmission line has thermal limit `F̄`. From A06, its DC flow is linear in the injections: `f = f₀ + Σ_k PTDF_k w̃_k`. Take 100 wind farms with `PTDF_k = 1`, and suppose each output deviates from its forecast by an independent, symmetric error in `[−1, +1]` MW. How much headroom below `F̄` should the dispatch keep?

```
Soyster (Γ = 100):                         margin 100 MW   (all farms worst at once)
Bertsimas–Sim, Hoeffding bound, ε = 1%:    Γ = √(2·100·ln 100) ≈ 30.3 → margin 30.3 MW
Bertsimas–Sim, tight bound (Table 1):      Γ ≈ 24.3                   → margin 24.3 MW
True 99% quantile if errors are ±1 coin flips: σ = 10   → ≈ 2.33·10  ≈ 23.3 MW
True 99% quantile if errors are uniform:       σ = √(100/3) ≈ 5.8 → ≈ 13.4 MW
```

The tight bound is almost exact in the worst symmetric case (coin flips), and conservative by a factor of about 1.8 when the errors are uniform. That is the price of making no assumption about the shape of the distribution. If you know the shape, a chance constraint (B05, B07) can recover the gap. If the errors are correlated, every number in the table is too optimistic.

The protection function for one row with `â_j|x_j| = (5, 3, 2)` and `Γ = 1.5` is `5 + 0.5·3 = 6.5`. The dual gives the same value: with `z = 3` you get `p = (2, 0, 0)` and `1.5·3 + 2 = 6.5`.

**How it connects**

- **X001** (Powell): robust optimization is a cost-function approximation policy. You solve a deterministic problem whose constraints have been tightened by a tunable parameter (`Γ`), and you tune that parameter by simulation.
- **X002** (Pinson): probabilistic wind forecasts tell you how wide the intervals `[a − â, a + â]` should be. Robust optimization only needs those widths, not the full density.
- **X008** (conformal decision theory): this entry calibrates the *uncertainty set* through a concentration bound. X008 calibrates the *decision* online. Both turn a risk level into a single conservativeness parameter.
- **X009** (B&B): the budgeted counterpart of a MILP is still a MILP, which is why this set is popular for unit commitment (A30).
- **A06**: PTDFs make every line flow linear in the injections, so this row-wise robust counterpart applies directly to DC line limits.
- **Coming up:** B01 places robust, stochastic, chance-constrained and distributionally robust optimization in one taxonomy. B05–B06 give the chance-constrained and scenario-based guarantees that this entry's bound competes with. B08 makes the recourse adjustable, and B12 replaces the box set with a Wasserstein ball around data.

**Check yourself**

1. Why does the inner maximization over "which `⌊Γ⌋` coefficients move" not make the robust counterpart combinatorial?
2. For a row with `|J_i| = 400` uncertain coefficients, the Hoeffding bound is to give at most a 1% violation probability. Roughly what `Γ_i` is needed, and what fraction of full Soyster protection is that?
3. Name two realistic grid situations in which the guarantee of Theorem 2 does not hold, and say which assumption fails in each.

<details><summary>Answers</summary>

1. For a fixed `x`, the subset choice is a fractional knapsack LP whose optimum is integral (pick the `⌊Γ⌋` largest `â_j|x_j|` plus a fraction of the next one). LP duality turns this inner *max* into a *min* over `(z_i, p_ij)` with the same value. A min-constraint of the form "`… + min(…) ≤ b`" can be enforced by adding the dual variables to the outer problem, which gives a single LP (Theorem 1). The combinatorics disappear because the relaxation of the inner problem is exact.
2. Solve `exp(−Γ²/(2·400)) = 0.01`: `Γ = √(800 · ln 100) ≈ √3684 ≈ 60.7`, which is about 15% of `|J_i| = 400`. The tight bound would need a bit less. The `√n` scaling is what makes the fraction shrink as rows contain more uncertain terms.
3. (a) Wind farms in the same region share weather, so their forecast errors are positively correlated and independence fails; the sum's tail is much heavier than the bound assumes. (b) Errors that are skewed or bounded on one side, for example PV output that cannot exceed clear-sky output or wind near cut-out, break symmetry. Other valid answers: a few large uncertain terms dominate the row (`|J_i|` small, so Soyster is needed anyway), or the requirement is a *joint* guarantee across many lines, while `Γ_i` gives only row-wise bounds.

</details>

---

## 2026-10-17 (Saturday) — X009 · Track 12: Hybrid ML–solver pipelines, warm starts, benchmarking — Keep the solver, learn its decisions: GNN branching inside an exact MILP solver

**Type:** Cross-phase preview
**Main reference:** [Exact Combinatorial Optimization with Graph Convolutional Neural Networks — M. Gasse, D. Chételat, N. Ferroni, L. Charlin & A. Lodi (2019), Advances in Neural Information Processing Systems 32 (NeurIPS 2019)](https://arxiv.org/abs/1906.01629) — read Sections 1 and 3–6 (about 6 pages: branch-and-bound as an MDP, imitation of strong branching, the bipartite state, the two half-convolutions, eq. 4, and the prenorm layer). Then read Tables 2 and 3 slowly, with the metric definitions in §5.1 "Evaluation". Skim Section 2. The link goes to the open arXiv version (v3, October 2019). The code is at [github.com/ds4dm/learn2branch](https://github.com/ds4dm/learn2branch).
**Further reading:**
- [Solving Mixed Integer Programs Using Neural Networks — V. Nair, S. Bartunov, F. Gimeno et al. (2020, rev. 2021), arXiv:2012.13349](https://arxiv.org/abs/2012.13349) — scales the same idea up: a learned brancher plus "Neural Diving", which proposes partial integer assignments that the solver then completes. The solver still certifies the result.
- [Ecole: A Gym-like Library for Machine Learning in Combinatorial Optimization Solvers — A. Prouvost, J. Dumouchelle, L. Scavuzzo, M. Gasse, D. Chételat & A. Lodi (2020), NeurIPS Workshop on Learning Meets Combinatorial Algorithms](https://arxiv.org/abs/2011.06069) — the tooling that came out of this line of work: solver decisions exposed as Gym-style environments.

**Why this matters** — There are two ways to bring ML to an optimization problem. You can replace the solver with a network that outputs a solution, or you can keep the solver and let a network make some of its internal decisions. In the second design a bad prediction costs time, not correctness, and anyone building learned solvers for grid operations needs to know both designs and how to measure them fairly. This paper is the standard reference for the second design. It also shows how to report results so that an optimization expert believes them.

**Core ideas**

*The problem.* A mixed-integer linear program (MILP) is

```
min_x  cᵀx   s.t.  A x ≤ b,  l ≤ x ≤ u,  x ∈ ℤ^p × ℝ^(n−p)
```

Power-system decisions such as generator on/off (unit commitment, A30) make problems MILPs. If you drop the integrality requirement you get an LP, called the *relaxation*. It is cheap, and its value is a **lower bound** on the MILP optimum because it optimizes over a larger set.

*Branch-and-bound (B&B).* Start by solving the relaxation. If some integer variable has a fractional value, say `x_i* = 2.4`, split the problem into two children, one with `x_i ≤ 2` and one with `x_i ≥ 3`, and solve their relaxations. Repeat. This builds a search tree. Every integer-feasible solution you find is an **upper bound** (an *incumbent*). The minimum of the relaxation values over the open leaves is a **global lower bound**. A subtree whose relaxation value is already worse than the incumbent cannot contain anything better, so you *prune* it. The algorithm stops when the two bounds meet. If you stop early, you still have a certified *optimality gap*: "the solution is within 2% of optimal", even though you do not know the optimum. This certificate is what "exact" means in the title, and it is why commercial solvers are built on B&B.

*The decision that is learned.* At each node the solver has to pick **which** fractional variable to branch on. The choice can change the tree size by orders of magnitude. The best known rule for small trees is **strong branching**: for every candidate, solve both child LPs and score the candidate by how much the bound improves. It produces tiny trees, but it costs two LPs per candidate per node, so its total runtime is poor. Solvers therefore use cheap proxies instead. SCIP's default, *reliability pseudocost branching* (RPB), uses strong-branching scores early on and then switches to historical estimates.

*MDP view, but no RL.* The solver is the environment, the brancher is the agent, actions come from the candidate set `A(s_t)`, and an episode is one solved instance. RL is slow here because episodes get longer the worse the policy is. The authors use **behavioural cloning** of strong branching instead. They record expert pairs `(s, a*)` and minimise

```
L(θ) = −(1/N) Σ_(s,a*)∈D  log π_θ(a* | s)
```

This is plain classification. Earlier work instead regressed the expert scores or learned rankings.

*The state as a graph.* Each node LP is encoded as a **bipartite graph**. There is one node per constraint (row of `A`) and one node per variable (column), and an edge `(i, j)` wherever `A_ij ≠ 0`. Constraint, variable and edge feature tensors `C ∈ ℝ^(m×c)`, `V ∈ ℝ^(n×d)` and `E ∈ ℝ^(m×n×e)` describe the current LP; the full feature list is in the supplement. If cuts are only added at the root, the topology is the same at every node of the tree. The network does one graph convolution, split into two half-convolutions:

```
c_i ← f_C( c_i , Σ_{j:(i,j)∈E} g_C(c_i, v_j, e_ij) )      (variables → constraints)
v_j ← f_V( v_j , Σ_{i:(i,j)∈E} g_V(c_i, v_j, e_ij) )      (constraints → variables)
```

Here `f` and `g` are 2-layer MLPs. A final MLP on the variable nodes is followed by a softmax that is *masked* to the candidate set. The network runs at every node, so it must be cheap: deeper models made slightly better decisions but were slower overall.

*The GNN detail you will recognise.* They use **sum** aggregation, not mean aggregation, and cite the GIN argument (Xu et al.): a mean aggregator cannot count, for example how many constraints a variable appears in. Unnormalised sums make weight initialisation depend on node degree. They fix this with a **prenorm** layer, `x ← (x − β)/σ`, applied after each sum. Its `β` and `σ` are set from training-set statistics and then frozen. In the ablation (Table 3, set covering) the variants tie on small instances; on Hard, sum + prenorm needs 56,348 nodes against 62,227 for mean.

*Results and how they are reported.* There are four NP-hard families: set covering, combinatorial auctions, capacitated facility location and maximum independent set. Each has Easy, Medium and Hard sizes. Training uses **only the Easy size**: for each family, 100,000 expert samples from 10,000 instances. Every method runs inside SCIP 6.0.1 with presolve and primal heuristics on and a 1-hour limit. Each cell covers 20 instances × 5 seeds. The reported metrics are the 1-shifted geometric mean of solving time, *wins* (the number of instances on which a method was fastest, over the number it solved) and node counts on instances that every method solved. Highlights from Table 2:

- Set covering, Hard (2,000 rows, trained on 500): GCNN takes 1,490 s on average against 1,677 s for RPB. It is fastest on 66 of the 70 instances it solves, while RPB is fastest on 4 of its 65. GCNN needs 29,981 nodes against 47,299 for RPB.
- Full strong branching gives the smallest trees on Easy but solves **no** Hard set-covering instance within the hour.
- Maximum independent set is the honest failure. On Hard, GCNN is the fastest of the learned branchers, but it solves only 29 instances, while RPB solves 42, and its variance is high.

**A small example**

Take `min −5x₁ − 4x₂` subject to `6x₁ + 4x₂ ≤ 24`, `x₁ + 2x₂ ≤ 6`, with `x ≥ 0` and integer.

```
root LP:  x = (3, 1.5),  value −21          → lower bound −21
branch on x₂ (the only fractional variable):
  x₂ ≥ 2:  x = (2, 2),     value −18        integer → incumbent −18
  x₂ ≤ 1:  x = (3.33, 1),  value −20.67     bound beats −18 → keep open
    branch on x₁:
      x₁ ≤ 3: x = (3, 1), value −19         integer → incumbent −19
      x₁ ≥ 4: x = (4, 0), value −20         integer → incumbent −20
no open leaves → optimum −20 at (4, 0), certified
```

Strong branching at the root would score `x₂` by its child bounds, −20.67 and −18, which improve on −21 by 0.33 and 3. SCIP combines the two gains as a product, so the score is ≈ 1. Here there is only one candidate, so the rule makes no choice. In a unit-commitment MILP there are hundreds of fractional on/off variables at every node, and that is where a learned scorer pays off. Note also that if the network had chosen badly, the answer would still be −20. Only the tree would have been bigger.

**How it connects**

- **X003** (active-set learning) has the same structure: ML proposes, and an exact method checks. There the guess is which constraints bind, and the solver confirms or falls back. Here the guess is a branching decision, and correctness never depends on it.
- **X001** (Powell's policy classes): learning a fast approximation of an expensive look-ahead rule (strong branching) by imitation is a policy-function approximation of a look-ahead policy, used *inside* an algorithm.
- **A05** (Newton–Raphson) showed that the starting point decides whether a solver converges. F14 returns to it as learned warm starts, another "help the solver" component.
- The GNN toolkit here (sum aggregation, edge features, bipartite structure) is what phase C (C13–C17) applies to grid graphs.
- **Coming up:** A30 introduces unit commitment as a MILP. Phase F's hybrid unit (F13–F18) then covers AI proposals with classical checks, warm starts, local repair, and honest speed-up accounting (F16). Tables 2 and 3 of this paper are a good model for F16: report the *whole* pipeline's wall-clock time against the solver's own default, not against a slow expert.

**Check yourself**

1. Why is the GCNN brancher "exact" while a network that directly outputs an integer solution is not? What does a bad prediction cost in each case?
2. Full strong branching gives the smallest trees, yet it is the slowest method on Hard instances. What does this say about the metric you should optimise and report for a learned component inside a solver?
3. Why does sum aggregation matter for this task, and what problem does the prenorm layer solve?

<details><summary>Answers</summary>

1. B&B only prunes a subtree when the relaxation bound proves it cannot beat the incumbent, and it stops only when the bounds meet. The branching variable changes the shape of the tree, not the validity of these proofs, so any policy, even a random one, returns a certified optimum. A bad branching choice only makes the tree larger and the solve slower. A network that outputs a solution directly gives no bound. A bad prediction can be suboptimal or infeasible, and you would need a separate check to find out.
2. Fewer nodes is only a proxy. What matters is end-to-end time, including the cost of each decision (feature extraction and inference for a network, two LPs per candidate for strong branching). So you should report wall-clock time, wins and solved counts under a fixed time limit, compared with the solver's tuned default (RPB), and not only accuracy or tree size. The authors kept the model small for exactly this reason.
3. Good branching depends on counts and degrees, for example in how many constraints a variable appears. A mean aggregator cannot represent these (the GIN argument). Unnormalised sums give activations whose scale grows with node degree, which makes initialisation and training unstable and hurts transfer to larger instances with higher degrees. The prenorm layer standardises each sum with frozen training-set statistics. In the ablation, sum + prenorm generalises best to Hard instances.

</details>

---

## 2026-10-16 (Friday) — A06 · Track 01: Grid physics & power flow — DC power flow: the linear grid model behind markets, PTDFs, and where it breaks

**Type:** Focus lesson
**Main reference:** [DC Power Flow Revisited — B. Stott, J. Jardim & O. Alsaç (2009), IEEE Transactions on Power Systems 24(3), pp. 1290–1300](https://doi.org/10.1109/TPWRS.2009.2021235) — read the whole paper (11 pages), and focus on how the different DC model variants are built, the "hot-start" vs "cold-start" distinction, how losses are handled, and the conclusions on accuracy. The paper is not open access. If you cannot get it, the MATPOWER manual below covers the derivation and the PTDF construction (but not the paper's critical assessment).
**Further reading:**
- [MATPOWER User's Manual, v8.1 — R. D. Zimmerman & C. E. Murillo-Sánchez (2025)](https://matpower.org/docs/MATPOWER-manual.pdf) — §3.7 "DC Modeling" (pp. 31–34): the three assumptions and the matrices `B_bus` and `B_f`. §4.5 "Linear Shift Factors" (pp. 46–47): PTDFs, slack distributions and line outage distribution factors (LODFs).
- [Power Flow Review and Shift Factors — PSERC education module (Sloan series, module 1)](https://documents.pserc.wisc.edu/documents/pserc_education/sloan/module1/Shift_Factors_Sloan.pdf) — slides that derive the linearized power flow from the AC line equation and work through shift factors with numbers.

**Why this matters** — Most of the optimization-under-uncertainty literature for power systems (chance-constrained OPF, robust dispatch, market clearing, congestion forecasting) runs on the DC model. It is linear, so uncertainty in the injections maps to uncertainty in the line flows through a single matrix. Anyone who trains, benchmarks or constrains a learned dispatch model needs to know what the DC model keeps, what it throws away, and which failures it cannot see.

**Core ideas**

**1. Four assumptions turn AC into linear algebra.** Start from A04's flow on a line `i→k` with series admittance `g + jb`:

```
P_ik = V_i² g − V_i V_k (g cos θ_ik + b sin θ_ik),     θ_ik = θ_i − θ_k
```

Now assume (a) `r ≪ x`, so `g ≈ 0` and `b ≈ −1/x`; (b) `V_i ≈ V_k ≈ 1` pu; (c) small angle differences, so `sin θ_ik ≈ θ_ik`; and (d) ignore reactive power entirely. What remains is

```
P_ik ≈ (θ_i − θ_k) / x_ik
```

This is Ohm's law for a resistor network: angle plays the role of voltage, active power the role of current, and reactance the role of resistance. (MATPOWER also carries tap ratios and phase-shift angles through, see §3.7. They only add a constant offset.)

**2. Network form: the Laplacian again.** Summing over lines at each bus gives `P = B θ`. Here `B` is the weighted graph Laplacian from A03 with edge weights `1/x_ik`, and `P` is the net injection vector. `B` is singular (constant vectors are in its null space), which mirrors the physics: only angle differences matter. So fix `θ_slack = 0`, delete that row and column, and solve:

```
θ_red = B_red⁻¹ P_red,          f = B_f θ          (B_f = diag(1/x) · A,  A = branch–bus incidence)
```

This takes one sparse linear solve. There is no iteration, no convergence failure and no second root (contrast A05). The solution always exists and is unique, even at injections where the real AC system has no solution.

**3. PTDFs: the whole grid as one matrix.** Combine both steps:

```
f = H P,         H = B_f B_red⁻¹   (with a zero column for the slack)
```

`H` is the **power transfer distribution factor** (PTDF) matrix, also called shift factors. `H_lj` is the change in flow on line `l` when one extra MW is injected at bus `j` and withdrawn at the slack. Three properties do most of the work in later lessons:
- *Superposition.* Flows from many transactions or many forecast errors simply add up. A transfer from bus `a` to bus `b` moves `H_{:,a} − H_{:,b}` per MW.
- *Slack dependence.* Each PTDF column depends on where the MW is withdrawn. With a distributed slack `w` (weights summing to 1), `H_w = H_k (I − w 1ᵀ)` (MATPOWER eq. 4.37). For any *balanced* change (`1ᵀ ΔP = 0`), all versions give the same answer.
- *Graph meaning.* With `L⁺` the Laplacian pseudo-inverse, the flow on line `(i,k)` per MW sent from `a` to `b` is `(1/x_ik)·[(L⁺_ia − L⁺_ib) − (L⁺_ka − L⁺_kb)]`. These are effective-resistance quantities. Power flows along *every* path in parallel, in proportion to how electrically easy each path is, and no operator can route it along a chosen path. This "loop flow" is why one wind farm can congest a line in a neighbouring country.

Line outages follow from the same matrix. If `h̃_ij` is the sensitivity of line `i`'s flow to a transfer between line `j`'s endpoints, the line outage distribution factor is `LODF_ij = h̃_ij / (1 − h̃_jj)`. This makes N−1 screening (A01's security criterion) a sequence of rank-one updates and avoids thousands of new power flows.

**4. Why the paper exists: "the DC model" is many models.** The abstract notes that DC models are "in widespread and even increasing use, particularly in congestion-constrained market applications", but that their MW flows are "by no means assured" to be correct. Every DC model makes modelling choices, and the paper examines them:
- *Cold start vs hot start.* A cold-start model linearizes around the flat profile (`V = 1`, `θ = 0`) and needs nothing but network data. A hot-start model is built around a solved AC base case and predicts *changes* (`f ≈ f_AC,base + H ΔP`). It is far better near that base case, but it is tied to it.
- *Which susceptance.* `1/x` or the exact `−Im(1/(r + jx)) = x/(r² + x²)`. These differ when `r/x` is not small.
- *Losses.* A lossless model makes the slack pick up nothing. Real systems lose a few percent, and how losses are allocated (fictitious loads, loss factors) shifts the flows.

The lesson for ML: when a paper reports "DC-OPF" or uses a "DCPF baseline", ask which variant it is.

**5. Where it breaks.** Each assumption fails in a known regime:
- *High r/x* (distribution feeders, cables): `g` is no longer negligible, losses become large, and A02's P–δ / Q–V decoupling weakens.
- *Stressed systems*: large angle differences break `sin θ ≈ θ`. Depressed voltages break `V ≈ 1`.
- *Voltage- and reactive-driven congestion*, the blind spot. Thermal limits are limits on *current*, `|I| = |S|/V` with `|S| = √(P² + Q²)` (A07 defines them properly). A line carrying large reactive power, or one feeding a low-voltage area, can overload while its active-power flow is moderate. The DC model has no Q and no V, so it can never flag this. A dispatch that is "DC-feasible" can still violate voltage bands and thermal limits in AC.

**A small example**

Take a triangle with `x_12 = x_13 = 0.1` pu and `x_23 = 0.2` pu. Bus 1 is the slack, bus 2 a generator injecting 1 pu, and bus 3 a load of 1 pu. The DC model gives

```
B_red = [[15, −5], [−5, 15]],   B_red⁻¹ = (1/200)[[15, 5], [5, 15]]
θ_2 = +0.05 rad (2.86°),  θ_3 = −0.05 rad
f_12 = −0.5,  f_13 = +0.5,  f_23 = +0.5
PTDF columns (bus 2, bus 3):   line 1–2: (−0.75, −0.25)   line 1–3: (−0.25, −0.75)   line 2–3: (0.25, −0.25)
```

Half the transfer goes directly over line 2–3. The other half detours through the slack bus along the path 2→1→3, because that path has the same total reactance. Now compare with an AC Newton solve (bus 2 held at `V = 1`). Each row adds resistance or reactive load:

| case | P on 1–3 | \|S\| on 1–3 | V_3 | losses |
|---|---|---|---|---|
| DC model | 0.500 | 0.500 | 1 (assumed) | 0 |
| AC, r/x = 0, Q_3 = 0 | 0.501 | 0.502 | 0.998 | 0 |
| AC, r/x = 0.1, Q_3 = 0.3 | 0.510 | 0.573 | 0.970 | 0.011 |
| AC, r/x = 0.5, Q_3 = 0.3 | 0.540 | 0.631 | 0.941 | 0.061 |
| AC, r/x = 0.1, Q_3 = 1.5 | 0.537 | 1.313 | 0.876 | 0.032 |

The active power flow is never off by more than 8 %, which is why DC models look good in MW-error plots. But if line 1–3 is rated at 1.0 pu, the DC model reports 50 % loading in the last row, while the line actually carries 131 % and bus 3 sits far below a 0.95 pu voltage limit. That is reactive-driven congestion, and the linear model has no way to see it.

**How it connects**

A02 derived the P–δ / Q–V decoupling. This lesson keeps only the P–δ half. A03 showed that `B ≈ −`(Laplacian with weights `1/x`), and here that Laplacian is the entire model. A04's equations and A05's Newton solver give the AC reference that the DC model approximates. The DC solution is also what pandapower uses to initialize Newton. X004's DCPF baseline was a cold-start DC model, and it held up out of distribution precisely because nothing in it is learned. X008's rule "flow = schedule + error" is the PTDF relation `f = H(P + ε)`, and with Gaussian injection errors `ε ~ N(0, Σ)` the flow errors are `N(0, H Σ Hᵀ)`. Phase B's chance constraints build on exactly this. Next: A07 makes the operating limits precise (current vs apparent power, voltage bands, generator capability), A08 adds costs to get economic dispatch, and A23 revisits linear approximations that keep voltage and reactive power (LPAC).

**Check yourself**

1. A trader computes flows for a transfer of 100 MW from bus `a` to bus `b` with a PTDF built for slack bus 1. A colleague uses a distributed slack. Do their answers differ? What changes if the trade is a 100 MW injection at `a` with no matching withdrawal?
2. Injection forecast errors are `ε ~ N(0, Σ)`. Under the DC model, what is the distribution of the flow on line `l`, and why is this result unavailable in closed form for AC power flow?
3. A learned dispatch policy is trained with a DC-flow penalty in its loss and achieves zero DC violations on test data. Name two ways its decisions can still be infeasible in AC.

<details><summary>Answers</summary>

1. No. The transfer is balanced (`1ᵀΔP = 0`), and `H_w ΔP = H_k (I − w1ᵀ) ΔP = H_k ΔP`. For an unbalanced injection, the extra 100 MW has to be withdrawn somewhere, and the slack model decides where. The resulting flows then differ.
2. The flow on line `l` is `f_l ~ N(f̄_l, h_lᵀ Σ h_l)`, with `h_l` the `l`-th row of `H`. A chance constraint `P(f_l ≤ f̄^max) ≥ 1 − α` becomes the deterministic constraint `f̄_l + z_{1−α} √(h_lᵀ Σ h_l) ≤ f̄^max`. In AC the map from injections to flows is nonlinear and only implicitly defined (A05), so a Gaussian input does not give a Gaussian output, and the output distribution has no closed form.
3. (a) Thermal overloads from reactive flow or low voltage: `|S|` or `|I|` exceeds the rating while `P` is within it (the last row of the table). (b) Voltage-band violations, which the DC model does not represent. Also: losses change the slack's output and the flows; and at stressed points the AC power flow may have no solution at all (A05), a case the DC model always "solves".

</details>

---

## 2026-10-15 (Thursday) — X008 · Track 11: Conformal prediction, risk control, selective prediction — Calibrate the decision, not the prediction: conformal decision theory

**Type:** Cross-phase preview
**Main reference:** [Conformal Decision Theory: Safe Autonomous Decisions from Imperfect Predictions — J. Lekeufack, A. N. Angelopoulos, A. Bajcsy, M. I. Jordan & J. Malik (2023, rev. 2024), arXiv:2310.05921 [stat.ML]](https://arxiv.org/abs/2310.05921) — read Sections I, III and IV (about 4 pages: decision families indexed by a control variable, the conformal controller update, eq. 5, Definition 1 "eventually safe", Theorem 1 and the batch corollary). Then read Section V-A (robot navigation among pedestrians, where the controller is compared with prediction-set calibration). Skim V-B and V-C. The link goes to the open arXiv version (v3, May 2024).
**Further reading:**
- [Conformal PID Control for Time Series Prediction — A. N. Angelopoulos, E. J. Candès & R. J. Tibshirani (2023), NeurIPS 2023](https://arxiv.org/abs/2307.16895) — the same "online update of one scalar" idea applied to prediction intervals, with integral and derivative terms added. It is the closest relative on the prediction-set side.
- [Adaptive Conformal Inference Under Distribution Shift — I. Gibbs & E. J. Candès (2021), NeurIPS 2021](https://arxiv.org/abs/2106.00170) — the online update that the conformal controller generalizes. Read Section 2 to see the same one-line recursion with a miscoverage loss. (The curriculum covers it in depth in F05.)

**Why this matters** — Anyone who puts a learned model in front of an operational decision (a dispatch, a security margin, an accept/reject call on a learned OPF solution) ends up asking how often the decision goes wrong, not how often the prediction interval misses. Conformal decision theory shows that the distribution-free machinery of conformal prediction can calibrate the decision directly. Its guarantee holds even when the data are not exchangeable and the sequence is adversarial. That makes it a good preview of Phase F's question: what can you promise about a learned system's risk with almost no assumptions?

**Core ideas**

**1. The gap between calibrated predictions and safe decisions.** You know split conformal: score a calibration set, take the `⌈(n+1)(1−α)⌉`-th smallest score, and get sets with marginal coverage `≥ 1 − α`. The usual pipeline for decisions is then "predict a set, then act robustly against the whole set". This has two weaknesses. First, coverage is the wrong currency: a miss on a variable that does not affect the decision costs nothing, while a small miss on a critical variable may be catastrophic. Second, the set must cover everything jointly, so it grows with the number of uncertain quantities, including irrelevant ones. The paper's robot example shows this. Adaptive conformal sets widened after a mispredicted pedestrian far away, and the robot froze. The conformal controller only paid attention to errors that actually produced losses.

**2. Setup.** At each time `t` you observe `x_t`. You have a family of decision rules `D_t^λ : X → U`, indexed by one scalar `λ ∈ ℝ` (the "conformal control variable"). Read `λ` as an aggressiveness knob: larger `λ` means a more efficient but riskier action. You act with `u_t = D_t^{λ_t}(x_t)`. Then the outcome `y_t` is revealed and you incur a bounded loss

```
ℓ_t = L(u_t, y_t) ∈ [0, 1],     R̂_T = (1/T) Σ_{t=1..T} ℓ_t
```

The goal is to keep the realized average risk `R̂_T` at or below a target `ε`, up to a term that vanishes as `T` grows. Nothing is assumed about how `(x_t, y_t)` is generated. The decision rule can wrap any predictor, planner or optimizer.

**3. The conformal controller.** The whole algorithm is one line (eq. 5):

```
λ_{t+1} = λ_t + η (ε − ℓ_t),     η > 0
```

If you incurred less loss than the budget, become a bit more aggressive. If you incurred more, back off. This is online gradient descent on the pinball-style objective behind adaptive conformal inference, with the decision loss in place of a miscoverage indicator. If the loss is "the set missed `y_t`" and `λ` shrinks the set, you recover ACI. So prediction-set calibration is the special case where the decision is the set.

**4. Why it works: a telescoping identity.** Summing the update from `1` to `t` gives `λ_{t+1} = λ_1 + η(tε − Σ ℓ_k)`, hence

```
R̂_t = ε + (λ_1 − λ_{t+1}) / (η t)
```

So the risk exceeds `ε` only if `λ` drifts far below where it started. This needs one assumption about the decision family (Definition 1, "eventually safe"). There is a level `λ^safe` such that if `λ_k ≤ λ^safe` for `K` consecutive steps, the average loss over those steps is at most `ε^safe ≤ ε`. In words, a sufficiently conservative fallback exists. Under this assumption `λ_t` can never fall much below `λ^safe`, and Theorem 1 gives, for all `t ≥ K` (with `λ_1 ≥ λ^safe − η`),

```
R̂_t ≤ ε + ( (λ_1 − λ^safe)/η + K ) / t
```

The guarantee is deterministic, holds for every sequence, and the excess decays as `O(1/t)`. No exchangeability, no model of the world, no calibration set.

**5. The batch version.** With exchangeable data and a loss that is monotone in `λ`, the paper also gives an offline rule (Corollary 2): choose `λ̂ = sup{λ : (1/n) Σ_i L_i(λ) ≤ ε − (1−ε)/n}` on a calibration set, and then `E[L_{n+1}(λ̂)] ≤ ε`. This is conformal risk control (lesson F08), and with a miscoverage loss it reduces to the split-conformal quantile you already know.

**6. Caveats worth taking into Phase F.**
- The guarantee is on the *time-averaged realized* loss. It is not a per-step probability, so it is not a chance constraint (Phase B), and it cannot rule out a burst of failures.
- It needs feedback: `y_t` must be revealed after each decision.
- It needs a safe fallback. If even the most conservative `λ` is unsafe, the theorem says nothing.
- `η` trades reaction speed against oscillation. Large `η` reacts fast but makes the decisions jumpy, while small `η` inflates the constant `(λ_1 − λ^safe)/η`. In the drone experiment the controller was safe for `η ∈ {100, 500, 1000}` and unsafe for `η = 50` over the finite horizon.
- One scalar knob calibrates one aggregate loss. Several constraint classes with separate budgets need several controllers or a combined loss.

**A small example**

One transmission line is rated at 100 MW. A learned dispatch rule schedules a flow of `80 + λ` MW on it: `λ` is how much of the remaining headroom you dare to use. The real-time flow is the schedule plus a forecast error `e_t ∈ [−15, 15]` MW. Let the loss be `ℓ_t = 1` if `80 + λ_t + e_t > 100` (an overload that needs corrective redispatch), and 0 otherwise. Take `ε = 0.05`, `η = 20`, `λ_1 = 0`.

- **Update sizes.** Without an overload, `λ` grows by `20 · 0.05 = 1` MW. After an overload, it drops by `20 · 0.95 = 19` MW.
- **Safe level.** With `λ ≤ 5`, the flow is at most `85 + 15 = 100`, so the loss is always zero. Thus `λ^safe = 5`, `K = 1`, `ε^safe = 0`. Also `λ_1 = 0 ≥ 5 − 20` holds.
- **A run.** `λ` climbs 0, 1, …, 10 over steps 1–11. At step 11, `e_11 = 12 > 20 − 10`, so the line overloads, and `λ_12 = 10 − 19 = −9`. Check the identity: `R̂_11 = 1/11 ≈ 0.091`, and `0.05 + (0 − (−9))/(20 · 11) = 0.05 + 0.041 = 0.091`. ✓
- **The bound.** Theorem 1 gives `R̂_t ≤ 0.05 + ((0 − 5)/20 + 1)/t = 0.05 + 0.75/t`. After 100 steps the realized overload rate is at most 5.75 %, and after 1,000 steps at most 5.075 %. This holds whatever the error sequence does, even if it is adversarial.

Now compare this with the set-based route on a grid with 200 lines. You would build a joint 95 % prediction set for 200 flow errors and dispatch robustly against all of them. The set is driven by the noisiest lines, even those that never come close to their rating. The controller only spends caution when an overload actually happens.

**How it connects**

This is the decision-level counterpart of what X002 (Pinson) argued for forecasts: once you have uncertainty, what matters is the decision built on it. In the language of X001 (Powell), the conformal controller is a policy function approximation whose single parameter is tuned online against realized cost. X006's point that uncertainty has many sources is the reason for a guarantee that assumes nothing about them. On the grid side, the "safe fallback" `λ^safe` is the conservative security margin operators already use (A01), and the overload loss refers to the thermal limits that A07 will define properly. Phase B will contrast this with chance constraints, which promise a per-step probability under a distributional model. Phase F builds the full toolkit: split and adaptive conformal (F01, F05), risk-controlling sets and conformal risk control (F07, F08), and Learn-then-Test (F09). This paper sits at the intersection of those lessons. Tomorrow's focus lesson, A06, is the DC power-flow approximation, which is exactly the kind of linear model behind a rule like "flow = schedule + error".

**Check yourself**

1. Derive `R̂_t = ε + (λ_1 − λ_{t+1})/(ηt)` from the update rule, and explain in one sentence why a lower bound on `λ_t` is all you need for a risk guarantee.
2. In the line example, an adversary picks `e_t = +15` at every step. What happens to `λ_t` and to the realized overload rate over time? Is the theorem's bound still respected?
3. Why does the theorem not give a chance constraint of the form `P(overload at step t) ≤ 0.05`, and which assumption would you need to add to get a statement about the next single step?

<details><summary>Answers</summary>

1. Sum `λ_{k+1} − λ_k = η(ε − ℓ_k)` over `k = 1..t` to get `λ_{t+1} − λ_1 = η(tε − Σℓ_k)`. Divide by `ηt` and rearrange to get the identity. The excess risk `R̂_t − ε` equals `(λ_1 − λ_{t+1})/(ηt)`, so if `λ_{t+1}` is bounded below by a constant, the excess is `O(1/t)`. The "eventually safe" assumption provides exactly that lower bound, because whenever `λ` is low the losses stop and `λ` climbs back.
2. An overload happens when `λ_t > 5`. Starting from 0, `λ` climbs by 1 per step and reaches 6 at step 7. That step overloads, and `λ` drops to `6 − 19 = −13`. It then needs 19 overload-free steps to get back above 5. The pattern repeats with one overload every 20 steps, so the realized rate tends to exactly `1/20 = 5 %`. At `t = 7` the rate is `1/7 ≈ 0.143`, and the bound gives `0.05 + 0.75/7 ≈ 0.157`, so the bound holds and is nearly tight early on.
3. The guarantee is about the average of realized losses along one sequence, and it holds for adversarial sequences, where per-step probabilities are not even defined. A statement about the next step needs a probabilistic model linking past and future, such as exchangeability of the calibration data and the test point. Under exchangeability, the batch rule (Corollary 2, conformal risk control) gives `E[L_{n+1}(λ̂)] ≤ ε`. With a 0/1 loss, that is a marginal probability bound on the next step, averaged over calibration draws. It is still not conditional on the current input.

</details>

---

## 2026-10-14 (Wednesday) — A05 · Track 01: Grid physics & power flow — Solving power flow: Newton–Raphson, the Jacobian, and why solvers fail

**Type:** Focus lesson
**Main reference:** [pandapower — An Open-Source Python Tool for Convenient Modeling, Analysis, and Optimization of Electric Power Systems — L. Thurner, A. Scheidler, F. Schäfer, J.-H. Menke, J. Dollichon, F. Meier, S. Meinecke & M. Braun (2018), IEEE Transactions on Power Systems 33(6)](https://arxiv.org/abs/1709.06743) — read Section IV-A, "Power Flow" (about one page, with Fig. 3: the Newton–Raphson solver, the three initialization options, the speed comparison with MATPOWER and PYPOWER, and reusing the admittance matrix across time steps). Skim Section II-C on how the element-based model is converted to a bus-branch model. The link goes to the open arXiv version. The published version is at [DOI 10.1109/TPWRS.2018.2829021](https://doi.org/10.1109/TPWRS.2018.2829021).
**Further reading:**
- [Power System Analysis — G. Andersson, ETH Zürich lecture notes (2012 version)](https://pees-nov.feit.ukim.edu.mk/predmeti/aees/nastava/literatura/Goran%20Andersson;%20PowerSystemAnalysis.pdf) — ch. 6, "Solution of the Power Flow Problem" (pp. 37–49): Gauss–Seidel, Newton–Raphson in one and many dimensions, quadratic convergence, the power-flow Jacobian (§6.3) and P–δ / Q–V decoupling (§6.4). This is the textbook half of the lesson.
- [pandapower documentation: AC power flow (`runpp`)](https://pandapower.readthedocs.io/en/latest/powerflow/ac.html) — the options discussed below (`algorithm`, `init`, `max_iteration`, `tolerance_mva`, `enforce_q_lims`), with their defaults.

**Why this matters** — Every power-flow dataset used to train a surrogate was produced by a Newton–Raphson solver, and every learned power-flow or OPF model is benchmarked against one. Knowing how the solver works tells you what its labels mean (a tolerance, not exact zeros), which samples are silently missing (the cases it failed on), and why the hardest operating points are also the ones it handles worst.

**Core ideas**

**1. Power flow as root finding.** A04 reduced power flow to a square system. Stack the unknowns `x = (δ for all non-slack buses, V for all PQ buses)` and define the mismatch

```
f(x) = [ P_i(V, δ) − P_i^spec   for every non-slack bus
         Q_i(V, δ) − Q_i^spec   for every PQ bus ]
```

Solving power flow means finding `x*` with `f(x*) = 0`. There is no objective: it is root finding, not optimization. In ML language, `f` is the physics residual of X007, and the solver drives it to zero for one input instead of minimizing it on average over a dataset.

**2. Newton–Raphson.** Linearize at the current iterate and solve the linear model for the step (Andersson §6.2):

```
J(x^ν) Δx = −f(x^ν),     x^{ν+1} = x^ν + Δx,     J = ∂f/∂x
```

Stop when `max |f| < ε`. pandapower's default is `tolerance_mva = 1e-8` MVA, with at most 10 iterations for Newton–Raphson. Near a simple root with a nonsingular `J`, the error roughly squares each iteration (quadratic convergence, §6.2.2): a well-posed case typically converges in 3–5 iterations from a flat start. This is the deterministic, exact-Hessian counterpart of the second-order methods ML people rarely run: one linear solve per step, but very few steps.

**3. The Jacobian has four blocks, and they mirror the graph.** Order the unknowns as angles, then magnitudes:

```
J = [ ∂P/∂δ   ∂P/∂V ]
    [ ∂Q/∂δ   ∂Q/∂V ]
```

Writing `δ_ik = δ_i − δ_k`, the entries are (Andersson §6.3, in his notation `U` for `V` and `θ` for `δ`):

```
off-diagonal (k ≠ i):  ∂P_i/∂δ_k =  V_i V_k (G_ik sin δ_ik − B_ik cos δ_ik)
                       ∂Q_i/∂δ_k = −V_i V_k (G_ik cos δ_ik + B_ik sin δ_ik)
diagonal:              ∂P_i/∂δ_i = −Q_i − B_ii V_i²     ∂P_i/∂V_i = P_i/V_i + G_ii V_i
                       ∂Q_i/∂δ_i =  P_i − G_ii V_i²     ∂Q_i/∂V_i = Q_i/V_i − B_ii V_i
```

Every off-diagonal entry carries a factor `G_ik` or `B_ik`. So entry `(i, k)` is nonzero only if buses `i` and `k` are connected, and each block has the sparsity pattern of the admittance matrix: the graph Laplacian of A03. A grid with 10,000 buses and about 3 lines per bus has a Jacobian with well under 0.1 % nonzeros. Real solvers never invert `J`. They factor it with sparse LU, and the cost depends on fill-in, which depends on the graph's structure and the elimination ordering. That is why topology, not just bus count, sets the runtime.

**4. Cheaper variants trade iterations for cost per iteration.**
- *Dishonest Newton*: keep an old `J` for several steps. More iterations, no refactoring (Andersson §6.2.1).
- *Fast decoupled*: on transmission grids `∂P/∂V` and `∂Q/∂δ` are small (A02's P–δ and Q–V coupling), so drop them and solve two half-size systems. Andersson §6.4 stresses the key point: if the iteration converges, it converges to the *exact* solution, because the residual `f` is unchanged. Only the step is approximate. This is a preconditioning idea, not a model approximation.
- *Backward/forward sweep*: exploits radial (tree) distribution feeders. It is pandapower's `bfsw`.
- *Gauss–Seidel*: cheap, slow, still in textbooks; pandapower's default cap is 1000 iterations against Newton's 10.

**5. What pandapower adds (Section IV-A).** Its solver began as a port of PYPOWER (itself a port of MATPOWER), but it assigns bus types and does the per-unit conversion automatically from an element-based model. Building the Jacobian is JIT-compiled with numba. Three findings worth remembering:
- *Initialization matters.* Options are a flat start (`V = 1`, `δ = 0`), a DC power flow (A06) for the angles, or the previous result. The paper recommends DC initialization in meshed grids, "where large voltage angle differences between the buses might lead to non-convergence in case of a flat start".
- *Overhead dominates small cases.* On MATPOWER test cases from 30 to 9241 buses (Fig. 3), converting to the bus-branch model takes more than half the time on small grids. pandapower is slower than MATPOWER there and faster on medium and large grids.
- *Reuse across time series.* If only injections change between runs, the admittance matrix and model conversion can be reused, and the previous solution makes a good warm start. Generating many training samples on one topology looks exactly like this.

**6. How power flow fails.** "Did not converge" (pandapower raises `LoadflowNotConverged`) has distinct causes, and a dataset generator should not treat them as one:
- *No solution exists.* The load is beyond the nose of the PV curve (A04). `J` becomes singular as you approach the nose, because two solutions merge there. No solver can fix this.
- *Bad start.* A solution exists but the iterate leaves its basin, for example with large angles from a flat start.
- *Wrong root.* The iteration converges, but to the low-voltage solution. The residual is zero and the answer is still useless.
- *Bus-type switching.* With `enforce_q_lims`, generators that hit reactive limits become PQ buses and the power flow runs again (A04). Each switch can push the case closer to infeasibility.

**A small example**

Use A04's two-bus grid: line `j0.1` pu, slack `V_1 = 1∠0`, PQ load at bus 2. The mismatch and Jacobian are

```
f(δ, V) = [ 10 V sin δ − P_2,   10V² − 10 V cos δ − Q_2 ]
J       = [ 10 V cos δ    10 sin δ
            10 V sin δ    20V − 10 cos δ ]
```

**Light load**, `P_2 = −0.5`, `Q_2 = −0.2`, flat start `(δ, V) = (0, 1)`. Here `J = [[10, 0], [0, 10]]` and `f = (0.5, 0.2)`, so the first step is `(−0.05 rad, −0.02)`.

| iter | δ (deg) | V (pu) | max abs mismatch | det J |
|---|---|---|---|---|
| 0 | 0 | 1 | 5.0·10⁻¹ | 100 |
| 1 | −2.865 | 0.9800 | 1.6·10⁻² | 93.8 |
| 2 | −2.930 | 0.97825 | 3.6·10⁻⁵ | 93.3 |
| 3 | −2.930 | 0.97825 | 1.7·10⁻¹⁰ | 93.3 |

The mismatch exponent roughly doubles each step (−1, −2, −5, −10). This is quadratic convergence, and the answer is A04's high-voltage solution.

**Heavy load**, `P_2 = −4.9`, `Q_2 = 0` (98 % of the nose limit `1/(2x) = 5`). From a flat start it takes 7 iterations to reach `V = 0.774`, `δ = −39.3°`, and `det J` falls from 100 to 15.4. The linearization is getting worse and the steps are getting less reliable.

**Beyond the nose**, `P_2 = −5.1`. `det J` drops to about 9.6, changes sign, and comes back near zero at `V ≈ 0.707`, `δ ≈ −45°`, which is exactly the nose point. The next step jumps to `V = 2.03`, `δ = +62°`, and the iteration diverges. No tolerance or iteration count would help.

**Bad start**, light load again but from `V = 0.1`. The iteration wanders through angles beyond −180° and converges to `V = −0.055`, `δ = −245.3°`. That is the same point as `V = 0.055`, `δ = −65.3°`: A04's low-voltage solution, with a residual of `10⁻¹⁶`. A residual check alone would label this as a success.

**How it connects**

A04 wrote down `f`, and this lesson solves `f = 0`. The Jacobian's blocks inherit the Laplacian sparsity of A03, and fast decoupled power flow exploits A02's P–δ / Q–V intuition. X007 previewed residual losses: Newton drives that same residual to machine precision for one case, while a learned model only lowers its average. That is why NR-generated labels (as in X004's line-graph GNN) count as ground truth only up to `tolerance_mva` and only for cases that converged. The singular Jacobian at the nose is also where the implicit map from loads to voltages stops being differentiable, which matters when later lessons differentiate through solvers (A13 on KKT, C-phase decision-focused learning). Next, A06 linearizes everything around a flat profile to get DC power flow: the same DC solution that pandapower uses to initialize Newton.

**Check yourself**

1. Fast decoupled power flow drops two Jacobian blocks. Why does it still converge to the exact AC solution when it converges, and why might it fail more often on a heavily loaded distribution feeder?
2. You generate 50,000 training samples by sampling loads and running Newton–Raphson, and you discard runs that raise `LoadflowNotConverged`. What bias does this introduce in the training set, and which part of the input space will a model trained on it know least about?
3. In the "bad start" example the residual was `10⁻¹⁶`. Give a cheap check that would have flagged the result as unphysical.

<details><summary>Answers</summary>

1. The decoupled method only changes the step `Δx`. The residual `f` is still the full AC mismatch, so any fixed point satisfies the exact equations. On distribution feeders, lines have a high `R/X` ratio (A02), so `∂P/∂V` and `∂Q/∂δ` are not small. With heavy loading, the dropped coupling makes the approximate step a poor direction, and the iteration converges slowly or not at all.
2. The surviving samples are biased away from the stability limit. Points near and beyond the nose, and cases with a bad basin of attraction, are under-represented or missing. The model therefore knows least about heavily stressed operating points, which are exactly where operators most need accurate answers. It will also never output "no solution exists", because that label never appears.
3. Check that voltage magnitudes are positive and inside a plausible band (for example 0.9–1.1 pu), and that angle differences across lines are small (wrap angles to (−180°, 180°] first). Here `V = −0.055` (or 0.055 after rewrapping) fails immediately. Rewritten with `V > 0`, the low-voltage root also has `det J ≈ −5.25`, against `+93.3` at the physical root, and a rerun from a DC or flat start lands on the high-voltage solution.

</details>

---

## 2026-10-13 (Tuesday) — X007 · Track 10: Physics-informed and constrained learning — Before PINNs: residual losses and constraints satisfied by construction (Lagaris et al., 1998)

**Type:** Cross-phase preview
**Main reference:** [Artificial Neural Networks for Solving Ordinary and Partial Differential Equations — I. E. Lagaris, A. Likas & D. I. Fotiadis (1998), IEEE Transactions on Neural Networks 9(5):987–1000](https://arxiv.org/abs/physics/9705023) — read Sections 1–3 (about 6 pages: the collocation loss, eq. 3; the trial-solution split, eq. 4; derivatives of the network with respect to its inputs, eqs. 5–10; and the trial forms for ODEs, coupled ODEs and Dirichlet/Neumann PDEs, eqs. 11–21). Then read Section 4.3 with Table 1 (comparison with finite elements) and Section 5. Skim the model problems in Sections 4.1–4.2. The link goes to the open arXiv preprint (1997). The published version is at [DOI 10.1109/72.712178](https://doi.org/10.1109/72.712178).
**Further reading:**
- [Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems — J. Willard, X. Jia, S. Xu, M. Steinbach & V. Kumar (2022), ACM Computing Surveys 55(4)](https://arxiv.org/abs/2003.04919) — Section 3 sorts the field into four families: physics-guided loss functions, initialization, architecture design and hybrid physics–ML models. It is a useful map for phase E.

**Why this matters** — Anyone training a network to output grid states or dispatch decisions has to choose how the physics enters. One option is a penalty in the loss, which the network may still violate. The other is a parameterization in which some constraints cannot be violated at all. This 1998 paper, written two decades before the term "PINN", sets up both options cleanly on toy differential equations. It is the shortest route to the vocabulary that phase E builds on.

**Core ideas**

**1. The problem as a loss.** Take a differential equation `G(x, Ψ(x), ∇Ψ(x), ∇²Ψ(x)) = 0` on a domain `D`, with boundary conditions (BCs) on its edge. Pick collocation points `x_i ∈ D̂ ⊂ D` and a model `Ψ_t(x, p)` with parameters `p`. The paper solves (eq. 3)

```
min_p  Σ_{x_i ∈ D̂}  G(x_i, Ψ_t(x_i,p), ∇Ψ_t(x_i,p), ∇²Ψ_t(x_i,p))²     subject to the BCs
```

To an ML reader this is self-supervised regression. There are no labels `Ψ(x_i)`, only the requirement that a known operator's residual be zero at sampled inputs. The "dataset" is just a set of points where the equation is checked. Modern PINNs (lesson E01) keep this loss, compute the derivatives with autodiff, and handle the BCs differently.

**2. Constraints by construction.** The key move is eq. 4:

```
Ψ_t(x) = A(x) + F(x, N(x, p))
```

Here `A` has no trainable parameters and satisfies the BCs exactly. `F` contains the network `N` but is shaped so that it vanishes wherever a BC is imposed. Examples from Section 3:

- first-order initial value problem `Ψ(0) = A`: `Ψ_t(x) = A + x·N(x)` (eq. 11);
- two-point Dirichlet problem `Ψ(0) = A`, `Ψ(1) = B`: `Ψ_t(x) = A(1−x) + Bx + x(1−x)·N(x)` (eq. 14);
- Poisson on the unit square: `Ψ_t = A(x,y) + x(1−x)y(1−y)·N(x,y)`, where `A` interpolates the four boundary functions (eqs. 17–18).

The authors state the payoff directly: the problem "has been reduced from the original constrained optimization problem to an unconstrained one". In optimization language, you have parameterized the feasible set, so any `p` gives a feasible point and plain gradient methods (they use BFGS) apply. ML analogies: a softmax output layer makes "probabilities sum to one" impossible to violate, and a residual connection `x + f(x)` builds in a known baseline. Mixed BCs are handled too. For a Neumann condition on one edge, eq. 19 subtracts `N(x,1)` and `∂N/∂y(x,1)` so that the correction's normal derivative is zero there.

**3. Derivatives of a network with respect to its inputs.** For one hidden layer, `N = Σ_i v_i σ(w_i·x + u_i)`, and any mixed input derivative is again a one-hidden-layer network with the same `w_i, u_i`. Its output weights are `v_i Π_k w_ik^{λ_k}`, and the activation is replaced by its `Λ`-th derivative (eqs. 5–7). Section 2.1 then derives the parameter gradients of these "gradient networks" by hand (eqs. 8–10). Today autodiff does this. The structural point still matters: the loss depends on `σ''` (and higher derivatives), so the choice of activation shapes the loss landscape. That observation returns in the PINN training pathologies of E02.

**4. What they found.** All experiments use 10 sigmoid hidden units, which is 30 parameters for ODEs and 40 for PDEs. Training uses 10 equidistant points per dimension (a 10×10 grid for PDEs). The test problems have known analytic solutions: four ODE problems (including a coupled system) and four PDEs (Dirichlet, mixed BCs, and one nonlinear PDE). Table 1 compares the maximum deviation from the exact solution with a finite-element (FEM) solver using an 18×18 biquadratic mesh (1369 unknowns). The FEM is often more accurate at its own nodes. The network's error at 900 unseen interior points, however, matches its training-point error. FEM interpolation degrades by one to two orders of magnitude, for example from `2·10⁻⁸` at the nodes to `1.5·10⁻⁵` in between on problem 5, where the network gives `5·10⁻⁷` at both.

**5. Read it critically.** These are smooth, low-dimensional problems on boxes with a tiny network, and the paper does no ablation over architecture (Section 5 says so). Hand-built `A` and `F` exist only for simple geometries and simple constraint types. Section 5 also anticipates adaptive placement of collocation points where the error is high, which later became standard in PINN practice. What carries over: penalties give approximate feasibility with a tunable weight, while constructions give exact feasibility but require you to find a parameterization of the feasible set.

**A small example**

Solve `Ψ' = −Ψ`, `Ψ(0) = 1` on `[0, 1]` (exact solution `e^{−x}`, `e^{−1} = 0.368`). Collocate at `x ∈ {0, 0.5, 1}` and replace the network with a single parameter so everything can be done by hand.

**Hard constraint (Lagaris).** `Ψ_t = 1 + a·x`, residual `r(x) = Ψ_t' + Ψ_t = (1 + a) + a·x`. The loss is `(1+a)² + (1+1.5a)² + (1+2a)²`. Setting its derivative to zero gives `4.5 + 7.25a = 0`, so `a = −0.621`. Then `Ψ_t(0) = 1` exactly and `Ψ_t(1) = 0.379`.

**Soft constraint (penalty).** `Ψ = c + a·x` with loss `Σ r² + λ(c − 1)²`, where now `r = c + a(1 + x)`. First note that without the penalty, `c = a = 0` gives zero residual: the trivial solution `Ψ ≡ 0` satisfies the ODE perfectly. The boundary condition is what selects the solution. The normal equations give `a = −0.621c` and `(3 + λ)c + 4.5a = λ`.
- `λ = 1`: `c = 0.829`, so the initial condition is violated by 17 % and `Ψ(1) = 0.314`.
- `λ = 100`: `c = 0.998`, a violation of 0.2 %.

Raising `λ` buys feasibility. In a real network it also worsens conditioning, which leads to the loss-balancing problem of E03 and the penalty-versus-augmented-Lagrangian discussion of E04.

**How it connects**

The power-flow equations of A04 are algebraic, not differential, but they fit the same template. A residual loss is `Σ_i (P_i(V, δ) − P_i^G + P_i^L)² + (same for Q)` at every bus. A "boundary condition" is anything pinned by bus type, such as the slack bus's `V_1 = 1, δ_1 = 0` or a PV bus's voltage setpoint. Those can be imposed by construction by simply not predicting them: output only the free coordinates and insert the pinned ones, as in eq. 4. The nonlinear balance equations themselves have no closed-form `A + F` split. That gap is why phase C has completion and feasibility-restoration layers (C06) and phase E has projection layers (E16). The `σ''`-dependence of the loss foreshadows E02. The label-free training signal is the counterpart to X004's line-graph GNN, which regressed Newton–Raphson labels and had nothing enforcing power balance. Next in the focus phase, A05 solves the A04 equations with Newton–Raphson, the classical solver that every residual-loss model is measured against.

**Check yourself**

1. In `Ψ_t(x) = A(1−x) + Bx + x(1−x)N(x)`, why does the factor `x(1−x)` not limit what the model can represent in the interior, and what could go wrong near the boundary?
2. The paper's network matched its training error at unseen points, while FEM did not. Give two reasons this comparison flatters the network.
3. A model outputs `(V_i, δ_i)` for all buses of a grid and is trained with the power-flow residual alone. Which degeneracy, analogous to `Ψ ≡ 0` in the example, does it have, and which A04 fact fixes it?

<details><summary>Answers</summary>

1. Any target `Ψ` that meets the BCs can be written as `A + x(1−x)h(x)` with `h = (Ψ − A)/(x(1−x))`. This `h` is smooth in the interior and has finite limits at the endpoints when `Ψ` is differentiable, so the network only has to approximate `h`. Near the boundary, though, the residual depends on `h` through derivatives of a product that is close to zero. Errors in `h` are damped there and amplified in the derivative terms, which can make the loss poorly conditioned close to the edges.
2. First, the test problems are smooth, low-dimensional and have analytic solutions, a regime that suits a global smooth approximator. Second, the network's "test points" lie inside the same box as its 10×10 training grid, so they test interpolation, not higher dimensions, rough solutions or out-of-distribution inputs. (FEM between its nodes is limited by its piecewise-polynomial basis, and neither method was tuned.)
3. The equations depend only on angle differences (`Y·1 = 0` in A03), so any common shift of all `δ_i` gives the same residual. Without a reference the model has a flat direction and ill-posed targets. Fixing the slack's `δ_1 = 0` (and its `V_1`), and pinning the other bus-type setpoints by construction, removes it. Even then, the low-voltage second solution from A04's example is a residual-zero point that the loss alone does not exclude.

</details>

---

## 2026-10-12 (Monday) — A04 · Track 01: Grid physics & power flow — The AC power-flow equations and bus types: why every bus fixes two of four numbers

**Type:** Focus lesson
**Main reference:** [An introduction to optimal power flow: theory, formulation, and examples — S. Frank & S. Rebennack (2016), IIE Transactions 48(12):1172–1197](https://sop.ior.kit.edu/img/An%20introduction%20to%20optimal%20power%20flow%20Theory%20formulation%20and%20examples.pdf) — §2.2–2.4 (network representation, conventional power flow, Table 1 on bus types, classic OPF eqs. 5–11; pp. 1175–1176) and §5.1 and §5.4 (bus injection model in its three coordinate forms, eqs. 57–62, and polar vs rectangular coordinates as constraints; pp. 1181–1184). Publisher version: [DOI 10.1080/0740817X.2016.1189626](https://doi.org/10.1080/0740817X.2016.1189626).
**Further reading:**
- [Power System Analysis — G. Andersson, ETH Zürich lecture notes (2012 version)](https://pees-nov.feit.ukim.edu.mk/predmeti/aees/nastava/literatura/Goran%20Andersson;%20PowerSystemAnalysis.pdf) — ch. 5, "Basic Power Flow Problem" (pp. 31–35): bus types, the switch from PV to PQ when a reactive limit is hit, and the count of equations against unknowns on a 5-bus example.

**Why this matters** — The AC power-flow equations are the equality constraints of every AC-OPF, the residual behind every "physics-informed" power-flow loss, and the map that any learned power-flow or OPF surrogate approximates. Bus types decide which quantities are inputs and which are outputs of that map. Get them wrong and a dataset silently mixes features with targets, or a model is asked to predict quantities that were in fact fixed.

**Core ideas**

**1. Four numbers per bus, two equations per bus.** A03 gave `I = Y E` and complex power `S_i = E_i I_i*`. Write each bus voltage in polar form `V_i ∠ δ_i` and split `Y = G + jB`. Taking real and imaginary parts of `S = diag(E)(Y E)*` gives the bus injection model (Frank & Rebennack eqs. 57–58):

```
P_i(V, δ) = V_i Σ_k V_k ( G_ik cos(δ_i − δ_k) + B_ik sin(δ_i − δ_k) ) = P_i^G − P_i^L
Q_i(V, δ) = V_i Σ_k V_k ( G_ik sin(δ_i − δ_k) − B_ik cos(δ_i − δ_k) ) = Q_i^G − Q_i^L
```

Each bus has four real quantities: net active injection `P_i`, net reactive injection `Q_i`, voltage magnitude `V_i` and angle `δ_i`. It has only two equations. For N buses that is 2N equations in 4N unknowns, so the system is underdetermined by exactly 2N. In ML terms the equations define a 2N-dimensional manifold in a 4N-dimensional space. To get a point on it you must pin 2N coordinates, two per bus.

**2. Bus types are the choice of which two to pin.** Conventional power flow, a feasibility problem with no objective, uses three types (Frank & Rebennack Table 1):

| Type | Known | Unknown | Typical device |
|---|---|---|---|
| Slack (Vδ, swing) | `V, δ` | `P, Q` | one large generator |
| PQ (load bus) | `P, Q` | `V, δ` | loads, buses with nothing attached |
| PV (voltage-controlled) | `P, V` | `Q, δ` | generators with voltage regulators |

The slack bus has two jobs. First, it fixes the angle reference: A03 showed `Y·1 = 0`, so only angle *differences* are determined, and the slack plays the role of grounding the Laplacian. Second, it closes the active-power balance. Losses `Σ_i P_i` depend on the solution and are unknown in advance, so one bus must have free `P` to absorb the mismatch. A PV bus models a generator whose automatic voltage regulator holds the terminal voltage at a setpoint by adjusting its reactive output. Andersson notes that PQ buses typically make up more than 80 % of all buses.

**3. The count works out.** With M PQ buses and N − M − 1 PV buses, the unknowns are N − 1 angles (all non-slack buses) and M magnitudes (PQ buses only), so N + M − 1 unknowns. You keep the `P` equation at every non-slack bus and the `Q` equation at every PQ bus, so N + M − 1 equations. Once you have the state `(V, δ)`, the remaining `P` and `Q` (slack `P, Q` and the PV buses' `Q`) follow by direct evaluation. A square system does not guarantee a solution. Frank & Rebennack point out that a nonlinear system can have several solutions, but usually only one is physically meaningful (high voltage, small angles).

**4. Limits change the bus type.** A PV generator can deliver only `Q_min ≤ Q ≤ Q_max`. If the solved `Q` breaks a limit, the solver pins `Q` at the limit and frees `V`, turning the bus into PQ. If conditions change, it can switch back (Andersson §5.2). This makes power flow *piecewise*: the input–output map has kinks wherever a generator hits its reactive limit. This is the same active-set phenomenon as in X003, but it already appears in power flow itself.

**5. From power flow to OPF.** OPF frees some of the pinned quantities and chooses them by optimization. In the classic formulation (eqs. 5–11), loads `P^L, Q^L` stay fixed, while generator `P^G, Q^G` and all `V, δ` become decision variables. Minimize `Σ_{i∈G} C_i(P_i^G)` subject to the two equations above at every bus, plus box limits on `P^G`, `Q^G`, `V` and `δ`. The slack bus survives only as the angle reference `δ_1 = 0`. The power-flow equations are what separate OPF from plain economic dispatch (A08).

**6. Coordinates matter for optimization.** The same physics can be written three common ways (eqs. 57–62):
- polar voltage with rectangular admittance (above);
- polar voltage with polar admittance `|Y_ik| ∠ θ_ik`;
- rectangular voltage `E_i + jF_i`, which gives *quadratic* equations, for example `P_i = Σ_k G_ik(E_iE_k + F_iF_k) + B_ik(F_iE_k − E_iF_k)`.

The trade-off (§5.4): polar form makes voltage bounds simple boxes, `V_min ≤ V_i ≤ V_max`, but the equations are trigonometric. Rectangular form makes the equations polynomial, but voltage bounds become `V_min² ≤ E_i² + F_i² ≤ V_max²`, a nonconvex annulus. Rectangular form turns AC-OPF into a nonconvex QCQP, and that is the starting point for the SDP and SOCP relaxations later in phase A. For a neural network the choice is a parameterization choice too: predicting `(V, δ)` makes magnitude bounds easy to enforce, while predicting `(E, F)` makes the physics residual polynomial.

**A small example**

Two buses, a lossless line `z = j0.1` pu, so `y = −j10`, `B_12 = B_21 = 10`, `B_11 = B_22 = −10`, `G = 0`. Bus 1 is the slack, `V_1 = 1 ∠ 0`. Bus 2 is PQ with load `P^L = 0.5`, `Q^L = 0.2` pu (50 MW, 20 Mvar on a 100 MVA base), so `P_2 = −0.5`, `Q_2 = −0.2`. The equations at bus 2 reduce to

```
P_2 = 10 V sin δ = −0.5           →  V sin δ = −0.05
Q_2 = 10 V² − 10 V cos δ = −0.2   →  V cos δ = V² + 0.02
```

Square and add, with `u = V²`: `u² − 0.96 u + 0.0029 = 0`, so `u = (0.96 ± √0.91)/2`.

- **High-voltage solution:** `u = 0.95697`, so `V = 0.9783` pu and `δ = −2.93°`. This is the physical operating point.
- **Low-voltage solution:** `u = 0.00303`, so `V = 0.055` pu. It is mathematically valid and physically meaningless, an example of the extra solutions Frank & Rebennack warn about.

The slack then supplies `P_1 = 10·0.9783·sin(2.93°) = 0.500` (lossless line) and `Q_1 = 10 − 10·0.9783·cos(2.93°) = 0.230`. The extra `0.030` is the reactive loss `|I|²x = (0.5² + 0.2²)/0.957 · 0.1 = 0.030`.

**Loadability.** For general loads, a real solution exists only if `(1 − 2xQ^L)² ≥ 4x²((P^L)² + (Q^L)²)`. With `Q^L = 0` this means `P^L ≤ 1/(2x) = 5` pu. At that point the two solutions merge at `V = 0.707`, `δ = −45°`. This is the "nose" of the PV curve. Beyond it there is no solution, however good the solver.

**PV variant.** Put a generator at bus 2 holding `V_2 = 1.0`. Now `sin δ = −0.05`, and the bus must inject net `Q_2 = 10(1 − cos δ) = 0.0125`, so the generator produces `Q^G = 0.2125`. If its limit is `Q_max = 0.1`, the bus switches to PQ with `Q_2 = 0.1 − 0.2 = −0.1`. Solving `u² − 0.98u + 0.0026 = 0` gives `V_2 = 0.989`: the voltage sags below the setpoint it can no longer defend.

**How it connects**

This lesson turns A03's matrix identity `I = Y E` into the scalar equations that solvers and OPF models actually use. A02's per-unit system and its P–δ / Q–V coupling show up directly in the example: the `P` equation is driven by `sin δ`, the `Q` equation by voltage magnitudes. X004's line-graph GNN and most power-flow GNN datasets encode bus type as a node feature or a mask, because it decides which of `(P, Q, V, δ)` are inputs and which are targets. The PV→PQ switching echoes X003's active sets. Next, A05 solves these equations with Newton–Raphson, where the Jacobian of the system above is the central object and the nose point is where it becomes singular. A06 linearizes them into DC power flow, and A07 adds the operating limits that turn power flow into OPF constraints.

**Check yourself**

1. A 30-bus system has 1 slack, 5 PV and 24 PQ buses. How many unknowns and equations does conventional power flow solve, and which quantities are computed afterwards?
2. Why can't every bus be a PQ bus, even when all injections are known from forecasts?
3. A dataset gives `(P, Q, V, δ)` at every bus from solved power flows, and a model predicts all four everywhere from the loads. Which predicted entries are trivially known, and which mistake does this hide?

<details><summary>Answers</summary>

1. Unknowns: 29 angles (all non-slack buses) and 24 magnitudes (PQ buses), so 53. Equations: 29 `P` balances (PV and PQ) and 24 `Q` balances (PQ), so 53. Afterwards, the slack's `P` and `Q` and the five PV buses' `Q` are evaluated directly from the solved `(V, δ)`.
2. Two reasons. First, angles are only defined up to a common shift (`Y·1 = 0`), so some bus must fix a reference. Second, losses depend on the solution and are unknown in advance, so fixing every `P` would over-determine the active-power balance and generally leave no solution. One bus must have free `P` to absorb the losses.
3. The slack's `V` and `δ`, every PV bus's `P` and `V`, and every PQ bus's `P` and `Q` are inputs (setpoints), not predictions, so including them inflates accuracy. It also hides any switching from PV to PQ: a model that does not know which generators hit their `Q` limits mixes two regimes and can be confidently wrong near the kinks.

</details>

---

## 2026-10-11 (Sunday) — X006 · Track 09: Aleatoric/epistemic UQ and evaluating uncertainty signals — Where uncertainty comes from: a statistician's map beyond "aleatoric vs epistemic"

**Type:** Cross-phase preview
**Main reference:** [Sources of Uncertainty in Supervised Machine Learning – A Statisticians' View — C. Gruber, P. O. Schenk, M. Schierholz, F. Kreuter & G. Kauermann (2023, rev. 2025), arXiv:2305.16703 [stat.ML]](https://arxiv.org/abs/2305.16703). Read Section 1 and Section 3 (definitions, the linear-model example, the bias–variance link, misspecification via KL), then Section 4.2.1–4.2.3 (omitted variables and errors in X). Read Section 5.4.1 (deployment and distribution shift). Skim Sections 5.2–5.3 (total survey error, missing data) and the example in Section 5.4.6. The link goes to the open arXiv version (v3, January 2025). No journal version is listed.
**Further reading:**
- [Aleatory or epistemic? Does it matter? — A. Der Kiureghian & O. Ditlevsen (2009), Structural Safety 31(2):105–112](https://doi.org/10.1016/j.strusafe.2008.06.020): the engineering-reliability classic that the ML split comes from. It defines an uncertainty as epistemic if the modeller sees a way to reduce it, and it warns that epistemic uncertainty creates statistical dependence between events, which goes unnoticed if you model it as independent noise. The publisher version is paywalled; the [DTU record](https://orbit.dtu.dk/en/publications/aleatoric-or-epistemic-does-it-matter/) has the abstract.

**Why this matters** — Anyone who attaches an uncertainty signal to a learned OPF solution, a forecast or a constraint-violation predictor has to say what the signal measures. "Aleatoric vs epistemic" is the usual answer, but in practice the uncertainty comes from missing features, noisy measurements, mislabelled data and shifts between training and deployment. This paper is a short, careful map of those sources. Read it before phase D's methods (ensembles, heteroscedastic heads, decompositions) so that you know what each method can and cannot see.

**Core ideas**

**A precise definition, and what it implies.** Fix the inputs `X` and target `Y`. The true relationship ("nature") is the conditional law `Y | x ~ f_{Y|X}(·|x)`. The paper defines

```
aleatoric uncertainty at x  :=  Var(Y | x)     (entropy of f_{Y|X} for discrete Y)
epistemic uncertainty       :=  everything else
```

and splits the epistemic part into *model* uncertainty (the hypothesis class `P = {p(y|x; θ)}` may not contain `f_{Y|X}`) and *estimation* uncertainty (`θ̂` from finite data differs from the best `θ_0` in `P`). The key consequence: aleatoric uncertainty is defined *relative to the chosen inputs X*. The authors' dice example: with no features, a die roll is purely aleatoric; if you could measure the initial position and every movement, it would be deterministic. So the split is a property of a model and a dataset, not of the world. This echoes Der Kiureghian & Ditlevsen's "reducible *by this modeller*" criterion.

**The bias–variance link.** For squared loss and a mean prediction `ŷ = μ̂(x)` trained on a random dataset `D`:

```
E_{Y|x} E_D[(Y − ŷ)²] = Var(Y|x)  +  Var_D(ŷ | x)  +  bias(ŷ | x)²
                        aleatoric    estimation       model
```

The first term does not depend on the learner at all. The second shrinks with more data. The third shrinks only with a richer class. Even in this simplest case, the authors stress that a *prediction interval* does not add the pieces up. For linear regression, the 1−α interval is `x₀ᵀβ̂ ± t_{n−p,1−α/2} · σ̂ · sqrt(1 + x₀ᵀ(XᵀX)⁻¹x₀)`. The aleatoric term (`1`) and the estimation term (`x₀ᵀ(XᵀX)⁻¹x₀`) sit together under one square root, multiplied by an estimated `σ̂`. "Total = aleatoric + epistemic" is a conceptual statement, not an identity you can read off a model. Section 3.3 adds that under misspecification, maximum likelihood still converges to the KL-closest `θ_0` in `P`, so estimation uncertainty can be reduced even when model uncertainty is large.

**Omitted variables: the paper's central technical point.** Let `Z` be a context variable that affects `Y` but is not recorded. The law of total variance gives

```
Var(Y | x) = E_{Z|x}[ Var(Y | x, Z) ]  +  E_{Z|x}[ bias(x, Z)² ],   bias(x, z) = E[Y|x,z] − E[Y|x]
```

Two lessons follow. (i) Part of what an X-only model calls "irreducible" noise is really an omitted-variable bias, which more *features* (not more samples) would remove. (ii) Appendix B shows that for some `z`, `Var(Y|x,z)` can *exceed* `Var(Y|x)`. So an X-only model can understate uncertainty in some regimes and overstate it in others, even with zero average bias. Prediction intervals then over-cover in one regime and under-cover in another, while looking well calibrated on average. A related twist: a `Z` with no effect on the *mean* of `Y` is never selected by a squared-error loss, yet it can still drive heteroscedasticity (the paper's example is annotators of different reliability).

**This undercuts a common argument.** One argument says model uncertainty can be ignored for universal approximators such as deep nets. The authors reject it: the argument silently assumes that all relevant features are present and measured without error.

**Errors in X, errors in Y, and data production.** If the true feature is `Z` but you observe a noisy `X` (with `Y ⊥ X | Z`), the "error-prone" aleatoric variance becomes `E[Var(Y|Z) | x] + Var(E[Y|Z] | x)`. The second term is inflation caused purely by measurement noise. As the noise grows, the feature behaves like an omitted variable. Section 5 broadens this to *data uncertainty*: how the dataset was produced, framed with the survey-methodology "total survey error" checklist (coverage, sampling, nonresponse, measurement). It also covers *deployment*, where `f^Tr_{Y,X} ≠ f^De_{Y,X}`. The model is *transportable* only if the relevant conditionals are the same in both environments, which is a substantive, domain-knowledge claim rather than something a test split can check.

**A small example**

A wind farm's hourly output `Y` (MW) is predicted from a forecast wind speed `x` alone. There is an unrecorded regime `Z`, say whether the wind comes from a direction where the farm's wake losses are large and erratic, with `P(Z=0|x) = P(Z=1|x) = 0.5`:

```
Z = 0 :  Y | x, Z ~ N(40, 4²)
Z = 1 :  Y | x, Z ~ N(60, 20²)
```

The X-only model sees mean `50` and, by the law of total variance,

```
Var(Y|x) = ½(16 + 400)  +  ½(10² + 10²) = 208 + 100 = 308,   sd ≈ 17.5
```

The 100 MW² from the bias term is "aleatoric" to this model but would vanish if the regime were recorded. Now take the Gaussian 90 % interval the X-only model would report, `50 ± 1.645·17.5 = [21.1, 78.9]`. Its coverage in each regime:

```
Z = 0 :  P(21.1 < N(40, 4²)  < 78.9) ≈ 1.000
Z = 1 :  P(21.1 < N(60, 20²) < 78.9) ≈ 0.801
average                                ≈ 0.901
```

On average the interval looks correct, at 90 %. In the hard regime it covers only 80 %, because `Var(Y|x,Z=1) = 400 > 308`. This is exactly where a downstream dispatch or reserve decision is most likely to fail. A reliability diagram pooled over all hours would not show it.

**How it connects**

**X002** argued that forecasts should be distributions, and **X005** showed how a learner can take a distribution as *input*. This entry asks what a model's *output* uncertainty measures. The omitted-variable term above is the same law-of-total-variance (Jensen-type) gap seen in X005, now viewed as a missing input. The regime example also previews a power-systems fact from **A01–A03**: one unobserved common factor (weather) moves many buses at once. Der Kiureghian & Ditlevsen's warning that epistemic uncertainty induces *dependence* is why errors at different nodes are correlated, which matters for the joint chance constraints in phase B. In phase D, **D07** gives the canonical ML treatment (Hüllermeier & Waegeman), which this paper extends. **D08** covers heteroscedastic heads that try to learn `Var(Y|x)`. **D10–D11** cover entropy-based decompositions and whether they hold up. **D12** covers shift. **D13** and **D17** turn the "fine on average, wrong per regime" problem into regression calibration and subgroup calibration, and **F06** revisits it for conformal coverage. Next focus lesson: **A04**.

**Check yourself**

1. Under the paper's definition, can adding a new input feature *increase* the aleatoric uncertainty at some points? Can it decrease the average?
2. Why is "use a deep network, so model uncertainty is negligible" not enough to make `Var(Y|x)` irreducible in practice?
3. In the example, which single change to the data would make the Z=1 coverage problem visible without changing the model, and which change would fix it?

<details><summary>Answers</summary>

1. Yes to both. Aleatoric uncertainty is `Var(Y | inputs)`, so it changes with the input set. Averaged over `Z`, `E[Var(Y|x,Z)] ≤ Var(Y|x)` by the law of total variance, so the average cannot go up. But for particular values `z`, `Var(Y|x,z)` can be larger than `Var(Y|x)` (in the example, 400 > 308), so the pointwise aleatoric uncertainty rises in that regime.
2. Universal approximation only removes the gap between the class `P` and `f_{Y|X}` for the *observed* `X`. If relevant variables are omitted or mismeasured, then `f_{Y|X}` itself carries omitted-variable bias terms and measurement-noise inflation. These look like irreducible noise to any learner on `X`, however flexible, but more or better features would reduce them.
3. Recording `Z` in the evaluation data (not necessarily in the model's inputs) lets you compute coverage per regime, which shows 80 % vs 100 %. Recording `Z` and feeding it to the model fixes it: the model then predicts `N(40, 4²)` or `N(60, 20²)`, gets regime-specific 90 % intervals, and removes the 100 MW² bias term.

</details>

---

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

---

## 2026-10-09 (Friday) — A03 · Track 01: Grid physics & power flow — The network model: π-lines, transformers, and the admittance matrix as a weighted graph Laplacian

**Type:** Focus lesson
**Main reference:** [Power System Analysis: Power Flow, Fault Analysis, Dynamics and Stability — G. Andersson, ETH Zürich lecture 227-0526-00 (2012 version)](https://pees-nov.feit.ukim.edu.mk/predmeti/aees/nastava/literatura/Goran%20Andersson;%20PowerSystemAnalysis.pdf) — ch. 2, "Network Models" (§2.1 lines and cables, §2.2 transformers incl. §2.2.3 unified branch model, §2.3 shunts; pp. 5–16) and ch. 4, "Nodal Formulation of the Network Equations" (pp. 27–29). (Same copy as in A02, hosted by another university.)
**Further reading:**
- [MATPOWER User's Manual, Version 8.1 — R. D. Zimmerman & C. E. Murillo-Sánchez (2025)](https://matpower.org/docs/MATPOWER-manual.pdf) — §3.2 "Branches" and §3.6 "Network Equations": the branch model and the sparse construction of `Y_bus` exactly as used to generate most OPF datasets (PGLib cases are in MATPOWER format).
- [Electrical Networks and Algebraic Graph Theory: Models, Properties, and Applications — F. Dörfler, J. W. Simpson-Porco & F. Bullo (2018), Proceedings of the IEEE 106(5)](https://www.control.utoronto.ca/~jwsimpson/papers/2017k.pdf) — the graph-theory view of this lesson: Laplacians, incidence matrices, effective resistance and Kron reduction (author copy).

**Why this matters** — The admittance matrix `Y` is the single object that encodes the grid's topology and line physics, and every power-flow equation, OPF constraint and physics-informed loss is built from it. For anyone feeding grids into a GNN it is also the natural weighted adjacency structure. Knowing exactly how it is assembled (and where transformers break its symmetry) prevents a whole class of data bugs and makes clear what a "graph" of a power grid really is.

**Core ideas**

**1. From a distributed line to a π-model.** A physical line is a continuum: every small piece `dx` has series resistance and inductance (`R′dx`, `L′dx`) and shunt capacitance and conductance to ground (`C′dx`, `G′dx`). In sinusoidal steady state (A02), this gives the telegraph equations, an ODE in the position along the line. We only care about the two ends, so Andersson collapses the line into a lumped **π-model**: one series impedance `z_km = r_km + j x_km` between buses k and m, and half of the line's total shunt admittance at each end, `y_km^sh = g^sh + j b^sh` (often `b_c/2`, with `g^sh ≈ 0`). The series admittance is

```
y_km = 1/z_km = g_km + j b_km,   g_km = r/(r² + x²) > 0,   b_km = −x/(r² + x²) < 0
```

The shunt susceptance is positive (capacitive): long lines and cables *produce* reactive power, which links back to A02. Kirchhoff and Ohm then give the two terminal currents as a linear map of the two terminal voltages:

```
[ I_km ]   [ y_km + y_km^sh      −y_km        ] [ E_k ]
[ I_mk ] = [   −y_km         y_km + y_km^sh   ] [ E_m ]
```

A line is a symmetric element: the matrix is symmetric and its diagonals are equal. Andersson's Example 2.1 (a 138 kV section, `z = 0.0062 + j0.0360` pu) gives `g = 4.64`, `b = −27.0` pu and `x/r = 5.8`. A 750 kV line in Example 2.2 has `x/r = 24.3`, so the higher the voltage, the more reactive the line and the better the P–θ/Q–|V| decoupling of A02.

**2. Transformers: an ideal ratio plus a leakage impedance.** A two-winding transformer is modelled as an ideal transformer with complex ratio `t_km = a_km e^{jφ_km}` in series with an impedance `z_km`. The ideal part has no losses, so `E_k I_km* + E_p I_mk* = 0`. For an **in-phase (tap-changing) transformer** (`φ = 0`, ratio `1 : a` on the k side, as in Andersson Fig. 2.4) this gives

```
[ I_km ]   [ a² y    −a y ] [ E_k ]
[ I_mk ] = [ −a y      y  ] [ E_m ]
```

This matrix is still symmetric, but its diagonals now differ. It can be redrawn as an asymmetric π-model with series admittance `A = a·y` and shunts `B = a(a−1)·y` at bus k and `C = (1−a)·y` at bus m. A tap away from 1.0 works like a pair of fictitious shunts that push reactive power, and therefore voltage magnitude, to one side. This is how operators control voltage with tap changers.

For a **phase-shifting transformer** (`φ ≠ 0`) the off-diagonals become `−t*·y` and `−t·y`. They are no longer equal, so **no π-model exists and the matrix is not symmetric**. Phase shifters rotate the angle difference across a branch and so steer *active* power. They are the grid's one direct "routing" control.

**Unified branch model.** Andersson §2.2.3 and MATPOWER §3.2 merge all of this into one branch: a π-line in series with an ideal phase-shifting transformer. MATPOWER puts the tap `τ e^{jθ_shift}` at the from-end with ratio `τ:1`, which gives

```
Y_br = [ (y_s + j b_c/2)/τ²         −y_s /(τ e^{−jθ_shift}) ]
       [ −y_s /(τ e^{+jθ_shift})        y_s + j b_c/2        ]
```

For an in-phase tap at the from-bus, Andersson's `1 : a` is MATPOWER's `τ : 1` with `a = 1/τ`, so `a² y` becomes `y/τ²`. Each tool fixes the tap side and the direction of the ratio its own way, and mixing them up is a classic bug when converting between tools (MATPOWER ↔ pandapower ↔ PowerModels).

**3. Kirchhoff at every bus gives `I = Y E`.** Kirchhoff's current law at bus k says the net injection from generators and loads equals the sum of the branch currents leaving k, plus the current into any bus shunt `y_k^sh` (capacitor banks, reactors). Stack all buses (Andersson eqs. 4.4–4.6):

```
I = Y E,   Y_km = −t*_km t_mk y_km  (k ≠ m, k adjacent to m),   Y_kk = y_k^sh + Σ_{m∈Ω_k} a_km² (y_km + y_km^sh)
```

MATPOWER builds the same matrix with sparse algebra: `Y_bus = C_fᵀ Y_f + C_tᵀ Y_t + diag(Y_sh)`. Here `C_f` and `C_t` are the branch-to-bus incidence matrices of the from- and to-ends. Then the complex power injection is `S_k = E_k I_k*`, and splitting it into real and imaginary parts gives the AC power-flow equations of **A04**:

```
P_k = U_k Σ_m U_m (G_km cos θ_km + B_km sin θ_km),    Q_k = U_k Σ_m U_m (G_km sin θ_km − B_km cos θ_km)
```

with `Y = G + jB`. The network part is linear. All the nonlinearity of power flow comes from the bilinear product `S = diag(E) (Y E)*` and from specifying `P, Q, |V|` instead of currents.

**4. Y is a complex-weighted graph Laplacian.** Take a network of lines only (no transformers or shunts). Let `A ∈ {0, ±1}^{L×N}` be the oriented branch-bus incidence matrix and `D_y = diag(y_1, …, y_L)` the series admittances. Then

```
Y = Aᵀ D_y A + diag(shunts)
```

Without shunts this is exactly the weighted Laplacian `L = Aᵀ W A` from spectral graph theory, with complex edge weights `y_km`. The usual Laplacian facts carry over:
- **Rows sum to zero**, so `Y·1 = 0`: if all buses have the same voltage phasor, no current flows. `Y` is singular, and the all-ones vector reflects the fact that only *differences* matter. This is why power flow needs a reference (slack) bus that fixes the angle, the counterpart of grounding a graph Laplacian.
- **Shunts ground it.** Line charging and bus shunts add to the diagonal ("Laplacian plus diagonal"), which usually makes `Y` invertible. Its inverse `Z_bus` is dense and is used in fault analysis.
- **Sparsity.** Each bus has a handful of neighbours. Andersson notes that a 1000-bus, 1500-branch network has typically more than 99 % zeros in `Y`. Newton–Raphson (A05) exploits this with sparse LU.
- **Kron reduction.** Eliminating buses with zero injection is a Schur complement, and the result is again a Laplacian of a denser graph. Dörfler, Simpson-Porco & Bullo cover this and effective resistance.
- **Not Hermitian.** `Y` is complex *symmetric* (`Yᵀ = Y`) for lines and in-phase transformers, but not Hermitian, so the spectral theorem does not apply directly. It loses symmetry altogether with phase shifters. For mostly inductive lines, `B = Im(Y)` is close to the *negative* of a real Laplacian with weights `1/x`: off-diagonals `B_km ≈ +1/x_km` and diagonals negative. That real Laplacian is the matrix of DC power flow (**A06**).

**ML reading.** `I = Y E` is one linear message-passing layer: edge-weighted aggregation of neighbour states, plus a self-loop term, with complex weights fixed by physics. A GNN on a grid either learns this operator or receives it as edge features (`r, x, b_c, τ, θ_shift` per branch, which is what most OPF datasets provide). The asymmetry of transformers means edge *direction* matters, so a model that symmetrises edges throws information away.

**A small example**

Three buses in a triangle, lossless lines, per-unit reactances `x_12 = 0.1`, `x_13 = 0.2`, `x_23 = 0.25`, no shunts. The series admittances are `y = −j/x = −j10, −j5, −j4`. Then

```
        [ −j15   j10    j5 ]          B = Im(Y) = −[ 15 −10  −5 ]
    Y = [  j10  −j14    j4 ]                       [−10  14  −4 ]
        [   j5    j4   −j9 ]                       [ −5  −4   9 ]   ← −(Laplacian with weights 1/x)
```

Each row sums to zero. Now set `E = (1, 0.98, 1)` (all angles 0). Then `I = Y E = (−j0.20, +j0.28, −j0.08)`, which sums to zero (no shunts, so no current to ground), and `S_k = E_k I_k*`:

```
S_1 = +j0.200,   S_2 = 0.98·(−j0.28) = −j0.2744,   S_3 = +j0.080
```

Buses 1 and 3 inject reactive power, bus 2 (the low-voltage bus) absorbs it, and no active power moves because all angles are equal. That is A02's Q–|V| coupling. The Q injections sum to `0.0056`. Check this against the lines: `|I_12| = 0.02/0.1 = 0.2`, giving `0.2²·0.1 = 0.0040`; `|I_23| = 0.02/0.25 = 0.08`, giving `0.08²·0.25 = 0.0016`. The total is `0.0056` ✓, the reactive losses `I²x`.

Transformer variant (Andersson Ex. 2.3): replace line 1–2 by a transformer with `x = 0.23` and tap `1 : 1.030`. Its π-equivalent has `A = −j4.48`, `B = −j0.13` (inductive shunt at the tap side) and `C = +j0.13` (capacitive shunt at the other side). Rows 1 and 2 of `Y` no longer sum to zero, and `Y_11` and `Y_22` change by different amounts.

**How it connects**

This builds on **A02**: impedance, admittance, per-unit (the `Z_base` conversion is how `x = 0.1` pu arises) and the two-bus formulas, which are the 2×2 special case of `I = Y E`. **X004** turned the grid into a line graph to put physics on nodes; this lesson shows the original graph object it re-encodes. Next, **A04** writes `S = diag(E)(Y E)*` as the bus-wise AC power-flow equations and introduces PQ/PV/slack buses (the slack is exactly the "grounding" of the Laplacian above). **A05** solves them with Newton–Raphson using the sparsity of `Y`, and **A06** linearizes them to `P = B′θ`, with `B′` the real Laplacian seen in the example. In phase C, **C13** covers message passing with edge features, the learning-side view of `Y`.

**Check yourself**

1. Why is the bus admittance matrix of a lossless network without shunts singular, and how does power flow deal with this?
2. A branch has `Y_12 ≠ Y_21`. What device does it contain, and what does the device control?
3. A 2-bus line has `z = j0.1` pu and total charging `b_c = 0.04` pu. Write its `Y` and say whether it is invertible.

<details><summary>Answers</summary>

1. Every row of `Y` sums to zero (Laplacian structure), so `Y·1 = 0`: a uniform voltage phasor drives no current, and only voltage differences are determined. Power flow fixes a reference (slack) bus with known angle (and magnitude), which removes the null direction, like grounding a node of a resistor network.
2. A phase-shifting transformer: with complex ratio `t = a e^{jφ}`, the off-diagonals are `−t*y` and `−t y`, which differ when `φ ≠ 0`. It shifts the angle difference across the branch and thus controls how much active power flows on it, compared with parallel paths.
3. `y = −j10` and each end gets `j0.02`, so `Y = [[−j9.98, j10], [j10, −j9.98]]`. Its determinant is `(−j9.98)² − (j10)² = −99.6004 + 100 = 0.3996 ≠ 0`, so it is invertible. The charging susceptance grounds the Laplacian, but only weakly, so `Y` is badly conditioned.

</details>

---

## 2026-10-08 (Thursday) — X004 · Track 07: GNNs and spatio-temporal GNNs for power grids — Turning the grid inside out: a line-graph GNN for power flow

**Type:** Cross-phase preview
**Main reference:** [Power Flow Balancing with Decentralized Graph Neural Networks — J. B. Hansen, S. N. Anfinsen & F. M. Bianchi (2022), IEEE Transactions on Power Systems, doi:10.1109/TPWRS.2022.3195301](https://arxiv.org/abs/2111.02169). Read Section III (the line-graph representation and the GNN design) and Section V (the three experiments, Tables II–IV and Fig. 7). Skim Section II (background) and Section IV (related work). The link goes to the open arXiv preprint (v2, 2022). An open-access accepted manuscript is also in the [UiT repository](https://munin.uit.no/handle/10037/27777).
**Further reading:**
- [Graph Neural Networks with Convolutional ARMA Filters — F. M. Bianchi, D. Grattarola, L. Livi & C. Alippi (2022), IEEE TPAMI, doi:10.1109/TPAMI.2021.3054830](https://arxiv.org/abs/1901.01343): the layer that makes the 40-hop model trainable. Read it for the rational-filter view of why ARMA layers can produce sharp, high-frequency outputs that GCN stacks smooth away.

**Why this matters** — A power grid is a graph, and the physics is local: what flows through a line depends directly only on the two buses at its ends. That makes GNNs the obvious architecture for learned power-flow and OPF solvers. It also raises design questions that a generic GNN paper does not answer: what should be a node, where should the line parameters go, and how deep must the network be when an effect at one end of the grid moves flows at the other end? This paper is a clear, honest first answer. It also reports where the approach fails (unseen grids), which makes it a good preview of phase C.

**Core ideas**

**The task, from first principles.** *Power flow* (lessons **A04–A05**) is the grid's forward problem. Given what every load consumes and what every generator produces, find the steady state of the network: the complex voltage `V_i = |V_i|∠θ_i` at each bus and the power on every line. The state solves a system of nonlinear equations, one active-power and one reactive-power balance per bus:

```
P_i = Σ_j |V_i||V_j| (G_ij cos θ_ij + B_ij sin θ_ij)
Q_i = Σ_j |V_i||V_j| (G_ij sin θ_ij − B_ij cos θ_ij)
```

Here `θ_ij = θ_i − θ_j`, and `G_ij + jB_ij` is an entry of the admittance matrix, which **A03** introduces as a weighted graph Laplacian. Classical solvers use Newton–Raphson. It is accurate but needs a factorization of a Jacobian at each iteration, and it can fail to converge. A learned surrogate maps grid inputs straight to the solution in one forward pass. The paper trains a GNN by supervised regression on Newton–Raphson solutions (MATPOWER).

**Design choice 1: branches as nodes.** The natural graph has buses as nodes and lines as edges. The authors note two problems with it. First, line parameters (resistance `r`, reactance `x`, charging susceptance `b`, transformer ratio `τ` and phase shift `θ_shift`) are *edge* features, and most cheap GNN layers handle only node features. Edge-conditioned convolutions are costly, and in a very deep network an edge-aware first layer has little influence on the output. Second, some bus outputs are fixed by the bus type. A PV (generator) bus has a set voltage magnitude, so a uniform local update has to be patched for those nodes.

Their fix is the *line graph* `L(G)`. Every branch becomes a node, and two nodes are adjacent when their branches share a bus. Each node then carries everything relevant to its branch in a 21-dimensional vector:

```
x_i = [ r, x, b, τ, θ_shift          (branch, 5)
      | P_d, Q_d, G_s, B_s, P_g, |V_g|, i_slack(2)   (from-bus, 8)
      | same 8 features                (to-bus, 8) ]
```

The from-bus block always comes first, so the branch direction is encoded in the feature order, while the graph itself stays undirected so that information can diffuse everywhere. The *target* changes as well. Instead of bus voltages, the model predicts 8 numbers per branch: the real and imaginary parts of the complex power and the current at both ends. Branch flows are natural line-graph targets. A bus voltage would appear on several line-graph nodes and would need a consistency constraint.

**Design choice 2: very deep, without over-smoothing.** A power-flow solution is *not* smooth on the graph: two neighbouring lines can carry very different flows. A stack of GCN layers is a low-pass filter, so after many layers neighbouring outputs become alike (over-smoothing, lesson **C14**). The model uses ARMA layers. Each has `K` parallel stacks of `T` recursive propagation steps with a skip connection to the input:

```
X̄_k^(t) = σ( Ã X̄_k^(t−1) W_k + X V_k ),   t = 1..T,   X̄ = (1/K) Σ_k X̄_k^(T)
```

Here `Ã = D^(−1/2) A D^(−1/2)`. The skip term `X V_k` re-injects the node's own raw features at every step, which keeps outputs sharp. The final network has 2 dense pre-processing layers, 5 ARMA layers with `T = 8` (a receptive field of 40 hops), 2 dense post-processing layers and a linear 8-unit head, all 64 units wide. It is trained with plain MSE. All quantities are in per unit (**A02**), so different grids have similar value ranges.

**Baselines.** *DCPF* is the linear DC approximation (**A06**), computed without learning. *Local MLP* sees only one branch's 21 features. *Global MLP* concatenates the whole grid into one vector. *GCN* is a 40-layer GCN with the same receptive field. The metric is the RMSE per output normalized by that output's standard deviation, averaged over outputs, so `NRMSE = 1` is about as good as predicting the mean.

**Findings.**
- *Fixed topology* (15 000 samples per grid; loads ±50 %, line parameters ±10 %, transformer settings and generator setpoints resampled). On case118, the NRMSE was ARMA 0.057, Global MLP 0.134, Local MLP 0.447, DCPF 0.674 and GCN 0.887. GCN was worse than the physics-free DC approximation, and Fig. 7 shows why: its outputs are nearly constant across neighbouring lines.
- *Random outages* (5–20 branches removed from case300). ARMA 0.243 versus Global MLP 0.371. The Global MLP holds up because each input slot still maps to a fixed component.
- *Training on six grids, testing on two unseen ones.* On the training grids the ARMA GNN beats the Global MLP on 5 of 6 (e.g. case89pegase 0.071 vs 0.791). On unseen case57 it scores 1.345 against DCPF's 0.584, and on case300 it scores 2.478 ± 1.636. The authors say plainly that DCPF is the better choice there. Size-independence makes it *possible* to run the model on a new grid, not *safe* to.

**A small example**

Four buses with branches a = (1,2), b = (2,3), c = (1,3) and d = (3,4). In the line graph:

```
a–b (share bus 2)   a–c (share bus 1)
b–c, b–d, c–d (all share bus 3)
```

Bus 3 has degree 3, so it becomes a *triangle* {b, c, d} in `L(G)`. A bus of degree `k` becomes a `k`-clique. Each line-graph node knows both of its endpoints' loads and generation, so one message-passing step on `L(G)` already mixes information across two buses and two lines.

Now cut branch c. In `L(G)` this deletes node c and its three edges, leaving a path a–b–d. The GNN's weights are shared across nodes, so it runs unchanged. A Global MLP would instead have to zero out c's input slot and hope it learned what "zero impedance" means. As the paper notes, zero `r` and `x` are hard to tell apart from a very short line.

The NRMSE scale: if the true power flow on line a has standard deviation 0.3 p.u. across samples, an NRMSE of 0.057 means a typical error of roughly 0.017 p.u. (1.7 MW on a 100 MVA base). An NRMSE of 0.887 means the model has barely improved on predicting the average flow.

**How it connects**

This entry builds on **A02** (per-unit values and complex power, the units of every input and output here). It looks ahead to **A03** (the admittance matrix, which the line graph re-encodes), **A04** (the equations the model is learning to solve) and **A05** (the Newton–Raphson labels, and convergence failures, which the dataset simply drops). **A06**'s DC approximation is the baseline that remains competitive out of distribution. Compared with **X003**: active-set learning keeps a solver in the loop and so is always either exact or flagged, while this model regresses the answer and nothing enforces that the predicted flows satisfy power balance at each bus. Closing that gap is phase E's job (**E01**, **E06**). In phase C, **C13** formalizes message passing with edge features (the alternative to the line-graph trick), **C14** explains over-smoothing, **C15** covers typed graphs for heterogeneous grid components, and **C16** returns to GNN power-flow solvers. The failure on unseen grids is a distribution-shift problem, the topic of **D12**. Next focus lesson: **A03**.

**Check yourself**

1. Why does converting to a line graph let the authors use a GNN layer that handles only node features, and what does it cost in graph size for a bus with many connections?
2. Why is a 40-layer GCN worse than the DC approximation, while a 5-layer ARMA network with the same 40-hop receptive field is the best model?
3. The model reaches an NRMSE of 0.057 on case118. Give two reasons why that number alone does not show that its output can be used as a power-flow solution.

<details><summary>Answers</summary>

1. Each branch has exactly two endpoints. So a branch's own parameters and both endpoint buses' features fit into one node feature vector, and no edge features remain. The cost: a bus of degree `k` becomes a `k`-clique, so the number of line-graph edges is `Σ_i k_i(k_i − 1)/2` instead of `Σ_i k_i/2`. That is cheap on sparse grids, but highly meshed substations add many edges.
2. Repeated GCN propagation acts as a low-pass filter. After 40 rounds the node features converge towards a smooth signal, but power flows on neighbouring lines can differ sharply in size and sign, so the GCN predicts something close to an average flow. ARMA layers re-inject the raw input at each recursive step (`X V_k`), which lets them represent higher-frequency graph signals. Using several layers with fewer recursions each also helped in practice.
3. (a) It is an average error under a training-like distribution. The same model scores 1.3–2.5 on unseen grids, and it was never tested on shifted load patterns. (b) Its outputs are trained with MSE only, so they do not have to satisfy the power-flow equations. Power injections at a bus need not balance, and the implied voltages need not agree between branches, so a small regression error can still be a physically inconsistent state. A per-line error can also matter much more near a thermal limit than the average suggests.

</details>

---

## 2026-10-07 (Wednesday) — A02 · Track 01: Grid physics & power flow — AC basics for ML people: phasors, complex power, reactive power, per-unit

**Type:** Focus lesson
**Main reference:** [Electric Power Systems: A Conceptual Introduction — A. von Meier (2006), Wiley-IEEE Press](https://doi.org/10.1002/0470036427) — ch. 3, "AC Power" (alternating current and voltage, phasors, real/reactive/apparent power, power factor); use ch. 2, "Basic Circuit Analysis", as a refresher on impedance if needed.
**Further reading:**
- [Class Notes Chapter 2: AC Power Flow in Linear Networks — J. L. Kirtley Jr., MIT 6.061 (Spring 2011), MIT OpenCourseWare](https://ocw.mit.edu/courses/6-061-introduction-to-electric-power-systems-spring-2011/9cf8de165233601f547b284cee5c2131_MIT6_061S11_ch2.pdf) — free and more mathematical: complex amplitudes, phasors, the decomposition of instantaneous power, and power transfer over an inductive line (the two-bus formulas used below).
- [Power System Analysis: Power Flow, Fault Analysis, Dynamics and Stability — G. Andersson, ETH Zürich lecture 227-0526-00 (2012 version)](https://pees-nov.feit.ukim.edu.mk/predmeti/aees/nastava/literatura/Goran%20Andersson;%20PowerSystemAnalysis.pdf) — ch. 3, "Active and Reactive Power Flows" (§3.1, transmission lines). These notes return as the main reading in **A03**. (This copy is hosted by another university; no ETH-hosted copy could be verified.)

**Why this matters** — Every OPF dataset, every GNN feature vector for a power grid and every constraint in an OPF solver is written in the vocabulary of this lesson: voltage magnitudes and angles, active and reactive injections, all in per-unit. Anyone building learned OPF solvers needs to know why there are two kinds of power, why voltage magnitude and angle behave so differently, and why all numbers sit near 1.0.

**Core ideas**

**1. Sinusoidal steady state and phasors.** In normal operation every voltage and current in the grid is (nearly) a sinusoid at the common frequency ω = 2π·50 Hz (A01). A sinusoid at a known frequency has only two free parameters: amplitude and phase. So we write

```
v(t) = √2 · |V| · cos(ωt + θ)    ⟷    V = |V| · e^{jθ} = |V|∠θ
```

where `|V|` is the RMS value (the √2 converts peak to RMS) and `j = √−1` (engineers reserve `i` for current). The phasor `V` is the complex amplitude: take the rotating vector `√2·V·e^{jωt}`, drop the known rotation, and keep its position at t = 0. The ML analogy is demodulation. You remove a known carrier and keep the 2-D state. Linear circuit elements act on phasors by complex multiplication, which is why steady-state grid analysis is complex linear algebra rather than differential equations.

**2. Impedance and admittance.** For a phasor current `I` through an element, `V = Z·I`, with impedance `Z = R + jX`. A resistor gives `R`; an inductor gives `X = ωL > 0`; a capacitor gives `X = −1/(ωC) < 0`. Transmission lines are mostly inductive: on high-voltage lines `X/R` is typically 5–10 or more. The inverse is the admittance `Y = 1/Z = G + jB`. Lesson **A03** assembles the admittances of all lines into the admittance matrix, which is a complex-weighted graph Laplacian.

**3. Instantaneous power splits into two parts.** Take `v = √2|V|cos(ωt)` and `i = √2|I|cos(ωt − φ)`, so the current lags the voltage by φ (φ > 0 for an inductive load). A trigonometric identity gives

```
p(t) = |V||I| cos φ · (1 + cos 2ωt)  +  |V||I| sin φ · sin 2ωt
       └──── P · (1 + cos 2ωt) ────┘     └──── Q · sin 2ωt ────┘
```

The first term never goes negative, and its average is `P = |V||I| cos φ`, the **active (real) power** in watts. This is the energy that actually does work and that the swing-equation balance of A01 is about. The second term averages to zero. Energy flows back and forth between the source and the inductor's magnetic field (or a capacitor's electric field) twice per cycle. Its amplitude is the **reactive power** `Q = |V||I| sin φ`, measured in var. Q does no net work, but the current that carries it is real current. That current heats the conductors (I²R losses, A01) and uses up the thermal capacity of lines, transformers and generators.

**4. Complex power packs both into one number.**

```
S = V · I*  =  P + jQ,        |S| = √(P² + Q²)   (apparent power, VA),     power factor = P/|S| = cos φ
```

The conjugate makes the phase difference `θ_V − θ_I = φ` appear. By convention, a load with `Q > 0` *absorbs* reactive power (inductive: motors, transformers, lines carrying heavy current). Capacitors and lightly loaded cables *produce* it. Equipment is rated in apparent power: a generator or transformer of rating `S_max` must satisfy `P² + Q² ≤ S_max²`, which is a disc in the (P, Q) plane. This convex constraint comes back with the other operating limits in **A07**. Supplying reactive power consumes rating that could otherwise carry active power.

**5. Angles move P, magnitudes move Q.** For a lossless line with reactance `X` between bus 1 (`V₁∠θ₁`) and bus 2 (`V₂∠θ₂`), Kirtley's notes derive

```
P₁₂ = |V₁||V₂| sin(θ₁ − θ₂) / X
Q₁₂ = ( |V₁|² − |V₁||V₂| cos(θ₁ − θ₂) ) / X
```

Active power flows "downhill" in **angle**. Reactive power flows mainly downhill in voltage **magnitude**. In normal operation angle differences are small (sin δ ≈ δ, cos δ ≈ 1), so `P₁₂ ≈ δ/X` and `Q₁₂ ≈ |V₁|(|V₁| − |V₂|)/X`. This approximate decoupling has two big consequences. First, dropping Q and fixing all magnitudes at 1 gives the linear DC power flow (`P = B·θ`, lesson **A06**), the backbone of most market models. Second, reactive power is a *local* quantity. Because `X ≫ R`, moving Q over long distances causes large magnitude drops and losses (Andersson makes the same point), so voltage must be supported locally by generators, capacitor banks or inverters. In OPF, voltage limits such as `0.95 ≤ |V| ≤ 1.05` are enforced mainly through Q.

**6. Three phases, one equivalent.** Real grids carry three sinusoids shifted by 120°. When the three phases are balanced, which is the standard assumption in transmission OPF, they are copies of each other up to rotation. Then one "per-phase" circuit describes everything, and total power is `S_3φ = 3·V_phase·I*` or `√3·V_line·I*` in magnitude. Distribution feeders are often unbalanced and need three-phase models; that is one reason distribution OPF is a different, harder problem.

**7. The per-unit system: normalization done by engineers.** Pick a system-wide power base `S_base` (by convention often 100 MVA) and a voltage base `V_base` for each voltage level, normally the nominal voltage. All other bases follow:

```
I_base = S_base / (√3 · V_base),    Z_base = V_base² / S_base,    x_pu = x / x_base
```

Why bother? (a) Voltages that are "healthy" are all near 1.0 pu, whether the network runs at 400 kV or 0.4 kV, so limits and features are comparable everywhere. It is feature scaling with a physical reason. (b) If the voltage bases on the two sides of a transformer follow its nominal turns ratio, the ideal transformer disappears from the per-unit circuit. The whole multi-voltage network becomes one circuit. (c) Equipment impedances given on their own rating convert easily: `Z_pu,new = Z_pu,old · (S_base,new/S_base,old) · (V_base,old/V_base,new)²`. Standard OPF test cases (MATPOWER, PGLib) store impedances in pu with `baseMVA = 100`, while loads and generator limits are in MW/Mvar. Mixing the two up is a common data-pipeline bug.

**A small example**

*Two buses, one line.* Use `S_base = 100 MVA`. A 380 kV line has `X = 144.4 Ω`. Then `Z_base = 380²/100 = 1444 Ω`, so `X = 0.1 pu`. Ignore R.

Case 1: `V₁ = 1.0∠0°`, `V₂ = 1.0∠−10°`.

```
P₁₂ = 1·1·sin(10°)/0.1 = 0.1736/0.1 = 1.736 pu = 173.6 MW
Q₁₂ = (1 − cos 10°)/0.1 = (1 − 0.9848)/0.1 = 0.152 pu = 15.2 Mvar
```

By symmetry bus 2 also sends 15.2 Mvar into the line, so the line absorbs about 30.4 Mvar in total. Check: `|I| = |V₁ − V₂|/X = 2·sin(5°)/0.1 = 1.743 pu`, and `|I|²·X = 3.038·0.1 = 0.304 pu`. ✓ Even with equal voltage magnitudes, carrying active power costs reactive power.

Case 2: same angles, but `|V₂| = 0.95`.

```
P₁₂ = 0.95·0.1736/0.1 = 1.650 pu      (−5 %)
Q₁₂ = (1 − 0.95·0.9848)/0.1 = 0.644 pu   (more than 4× case 1)
```

A 5 % change in voltage magnitude barely moves P but quadruples the reactive flow. That is the P–θ / Q–|V| split in numbers.

**How it connects**

**A01** explained why bulk power travels at high voltage (`P_loss ∝ P²/V²`) and why frequency signals the active-power balance. This lesson adds the second half: voltage *magnitude* is governed by reactive power, and it is local, not system-wide. **A03** turns the line formula above into a network-wide object, the π-line model and the admittance matrix `Y = G + jB`. **A04** writes the general AC power-flow equations, which are exactly `S_i = V_i · (Σ_k Y_ik V_k)*` per bus. **A06** shows the DC approximation obtained from the linearization in core idea 5.

**Check yourself**

1. An industrial load draws 80 MW at power factor 0.8 (lagging). What are its Q and |S|, and why does the utility care about the power factor if only P is "useful"?
2. In case 1 of the example, the angle difference doubles to 20°. Roughly what are the new P₁₂ and the reactive power absorbed by the line? What does this say about heavily loaded lines and voltage?
3. A transformer connects 380 kV to 110 kV and has `X = 0.05 pu` on a 100 MVA base. What is X in ohms, seen from each side, and why does the per-unit value not depend on the side?

<details><summary>Answers</summary>

1. `|S| = P / pf = 100 MVA`, and `Q = √(100² − 80²) = 60 Mvar` (absorbed). The current, and thus the I²R losses and the use of line and transformer capacity, scales with |S|, not P. Equipment must be sized for 100 MVA to deliver 80 MW, which is why tariffs often penalize low power factor.
2. `P₁₂ = sin 20°/0.1 ≈ 3.42 pu` (about double). The line absorbs `2·(1 − cos 20°)/0.1 ≈ 1.21 pu`, about four times as much, because reactive losses grow with `|I|²`. Heavily loaded lines absorb a lot of reactive power, which pulls voltages down unless Q is supplied locally. That is the root of voltage-stability problems.
3. On the 380 kV side, `Z_base = 1444 Ω`, so `X = 72.2 Ω`. On the 110 kV side, `Z_base = 110²/100 = 121 Ω`, so `X = 6.05 Ω`. The ratio is `(380/110)² ≈ 11.9`, exactly how an ideal transformer reflects impedance. Since the voltage bases follow the turns ratio, the per-unit value is the same on both sides, and the ideal transformer drops out of the per-unit circuit.

</details>

---

## 2026-10-06 (Tuesday) — X003 · Track 06: Learning to optimize, OPF proxies, decision-focused learning — Learning which constraints bind: active-set learning for repeated OPF

**Type:** Cross-phase preview
**Main reference:** [Learning for Constrained Optimization: Identifying Optimal Active Constraint Sets — S. Misra, L. Roald & Y. Ng (2022), INFORMS Journal on Computing 34(1):463–480](https://arxiv.org/abs/1802.09639) — read Sections 1–3 (the active-set idea, the DiscoverMass algorithm and its theorems, about 9 pages) and Section 5 (DC-OPF experiments, Table 1). Skip the proofs in Section 4. The link goes to the open arXiv preprint (v4, 2019). The [journal version](https://doi.org/10.1287/ijoc.2020.1037) adds case studies in supply-chain planning and shortest-path routing.
**Further reading:**
- [Learning for DC-OPF: Classifying active sets using neural nets — D. Deka & S. Misra (2019), arXiv:1902.05607](https://arxiv.org/abs/1902.05607) — the follow-up the main paper leaves as future work: a neural classifier from loads to active sets, tested on PGLib cases.
- [Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon — Y. Bengio, A. Lodi & A. Prouvost (2021), European Journal of Operational Research 290(2):405–421](https://arxiv.org/abs/1811.06128) — the wider view: should ML replace the solver end to end, or make decisions inside it? Active-set learning is an early example of the second kind.

**Why this matters** — Grid operators solve almost the same optimization problem again and again, every few minutes, with slightly different loads and renewable output. Anyone building a learned OPF solver has to decide *what* the network should predict. Raw setpoints are one option. A discrete summary of the solution that a solver can turn back into an exact, feasible answer is another. This paper makes the case for the second option and attaches a sample-size guarantee to it. That makes it a good first picture of "learning to optimize" before phase C formalizes the field.

**Core ideas**

**Parametric programs and their active sets.** The setting is a family of optimization problems indexed by an uncertain parameter `ω` (for OPF: load deviations from the forecast):

```
x*(ω) ∈ argmin_x f(x, ω)   s.t.   g_j(x, ω) ≤ 0,   j = 1..m
```

A constraint is *active* (binding) at the optimum if it holds with equality there. Examples are a line at its thermal limit or a generator at its maximum output. The *optimal active set* `A*(ω)` is the set of these constraints. The key observation: if you know `A*(ω)`, you can drop every other constraint and solve the much smaller reduced problem over `j ∈ A*`. For a linear program this is little more than a linear solve. The reduced problem is a relaxation of the original. So if you guess an active set and solve its reduced problem, the solution is either optimal for the original problem or violates some dropped constraint, and checking that is cheap. There is no silent middle ground.

**Why this is a good learning target.** For a DC-OPF (an LP; lessons **A06** and **A11** cover it properly), the map `ω → x*(ω)` is piecewise affine. Parameter space splits into polyhedral regions, one per active set, and inside each region the solution is an affine function of `ω`. This is a classical result from multiparametric programming. A regression network has to approximate the kinks between regions and never gets constraints exactly right. A classifier only has to say which region `ω` is in, and the solver then produces the exact answer. For an ML reader: predict a discrete structural latent, then decode it deterministically through the solver, instead of regressing a high-dimensional continuous output.

**The catch: too many classes.** The number of possible active sets grows exponentially with the number of constraints. The empirical bet is that engineered systems are *low-complexity*: a handful of active sets carry almost all the probability mass under realistic operating conditions. The paper defines the *mass* of an active set as `π(A) = P_ω(A*(ω) = A)`. It then asks how many solved samples you need before the active sets you have seen cover at least `1 − α` of the mass.

**DiscoverMass: a streaming stopping rule.** Draw `ω_i` i.i.d. from the operating distribution, solve each problem and record its active set. After `M` samples, look at a window of the next `W_M` samples and compute the *rate of discovery*:

```
R_{M,W} = (1/W) · Σ_{i=1..W} 1[ A_{M+i} ∉ {A_1, …, A_M} ]
```

This is the fraction of fresh samples whose active set is new. It is an unbiased estimate of the mass `π(U_M)` of the still-unobserved active sets, the same "missing mass" question behind Good–Turing estimation. Theorem 1 bounds the deviation `π(U_M) − R_{M,W}` uniformly over all `M`, for windows growing like `log M`. Theorem 2 then shows that stopping once `R_{M,W} < α − ε` gives observed active sets with total mass at least `1 − α`, with probability at least `1 − δ`. It assumes nothing about the problem's structure or the distribution. Theorem 3 bounds the cost: if `K_0` active sets carry `1 − α_0` of the mass, the number of iterations grows roughly like `K_0 / (α − α_0)`, i.e. linearly in the number of important active sets and independent of the grid's size.

**From active sets to decisions.** The paper describes two policies. A *classifier* maps `ω` to one active set (left to future work, picked up in the Deka–Misra follow-up). The *ensemble policy*, used in the experiments, solves the reduced problem for every discovered active set in parallel, discards infeasible solutions and keeps the best feasible one. Failure means returning nothing feasible, and that happens only when `A*(ω)` was never observed. So the failure probability is bounded by the undiscovered mass, about `α`.

**Findings (Section 5, DC-OPF on 15 PGLib cases, 3–1951 buses).** Loads were perturbed with independent zero-mean noise, either Gaussian with `σ = 3 %` of the load or uniform on `±9 %`. The settings were `α = 0.05`, `δ = 0.01`, `ε = 0.04`. Most systems turned out to be low-complexity. With Gaussian noise, the 118-bus case needed 2 active sets, and the 1888-bus and 1951-bus French cases needed 3 and 5. Most systems stopped after fewer than 200 samples. Grid size did not predict complexity. The hard cases were atypical: two network-reduced PSERC cases (the 240-bus one produced 2993 active sets without stopping) and the RTS cases, which have many generators per bus and, for the 73-bus case, three identical areas. Out-of-sample failure rates of the ensemble policy stayed below `α`. The broader uniform noise produced more active sets than the Gaussian.

**A small example**

Two buses, one line. Generator 1 sits at bus 1 and costs 10 €/MWh. Generator 2 sits at bus 2 and costs 30 €/MWh. Both have limits `0 ≤ p ≤ 100` MW. The load `d` sits at bus 2, and the line from bus 1 to bus 2 carries at most 60 MW. Balance requires `p_1 + p_2 = d`, and the line flow equals `p_1`.

- **Region 1** (`d ≤ 60`): the active constraint is `p_2 ≥ 0`. Solution: `p_1 = d`, `p_2 = 0`.
- **Region 2** (`60 < d ≤ 160`): the active constraint is the line limit `p_1 ≤ 60`. Solution: `p_1 = 60`, `p_2 = d − 60`.

The map `d → (p_1, p_2)` is piecewise affine with one kink at 60 MW. Let `d ~ N(50, 10²)`. Then `π(A_1) = P(d ≤ 60) ≈ 0.84` and `π(A_2) ≈ 0.16`. A few dozen samples find both active sets, and higher-load regions such as `p_2` at its maximum have negligible mass. Now take a new load `d = 70`. The reduced problem for `A_1` returns `p_1 = 70`, which violates the dropped line limit, so it is rejected. The reduced problem for `A_2` returns `(60, 10)`, which is feasible and therefore optimal. Note also that the marginal cost at bus 2 jumps from 10 to 30 €/MWh when the active set changes. Active sets also determine prices, which **A11** develops as locational marginal prices.

**How it connects**

**X001** called a learned OPF proxy a PFA. Active-set learning is a hybrid: a learned PFA picks the structure, and an exact solve inside fills in the numbers. **X002** stressed that the operating distribution of forecast errors drives everything downstream. Here, `P_ω` decides which active sets matter, so a forecast distribution that shifts (a new season, more PV) can make an unseen active set likely and void the guarantee. That theme returns in **D12** and **F05**. In phase A, **A06** and **A11** explain why DC-OPF is an LP with these properties, and **A13** (KKT conditions) makes "active constraint" precise. In phase C, **C01** (amortized optimization) gives the general frame, **C03** covers proxies that regress setpoints directly, and **C06** the alternative fix for feasibility (completion and correction). The "guess, then check cheaply, fall back if needed" pattern is the core of **F10–F11** (reject and defer options) and **F13–F16** (hybrid ML–solver pipelines and honest speed-up accounting). Next focus lesson: **A02**, AC basics.

**Check yourself**

1. Why is a solution of the reduced problem (only the constraints in a guessed active set) either optimal for the full problem or infeasible for it?
2. The DiscoverMass guarantee is "with probability `1 − δ`, the unobserved active sets have mass at most `α`". Which assumption behind it is most likely to fail in grid operation, and what happens to the ensemble policy when it does?
3. In the two-bus example, a regression network is trained to predict `p_1` from `d`. Why might it output `p_1 = 61` at `d = 75`, and why can the active-set approach not make this kind of error?

<details><summary>Answers</summary>

1. Dropping constraints enlarges the feasible set, so the reduced optimum costs at most as much as the true optimum. If that point also satisfies all the dropped constraints, it is feasible for the full problem at a cost no higher than the optimum, so it is optimal. Otherwise it violates some dropped constraint, which a cheap check detects. (For degenerate problems the paper uses a consistent tie-breaking rule.)
2. The assumption that test-time `ω` is drawn from the same distribution as the training samples. Under distribution shift (new renewables, a topology change, a different season) an active set that was never observed can carry real mass. The ensemble policy then returns only infeasible candidates more often than `α`. At least the failure is detected, because the feasibility check catches it, so the system can fall back to a full solve.
3. A smooth regressor has to approximate the kink at `d = 60`, and the flat part `p_1 = 60` is approximated with some error. Nothing forces its output to respect the line limit, so 61 MW, an overload, is a plausible output. The active-set approach predicts only which constraints bind and then solves for the numbers exactly. It returns either the exact solution `(60, 15)` or a candidate that is flagged as infeasible, never a slightly-off point that is silently infeasible.

</details>

---

## 2026-10-05 (Monday) — A01 · Track 01: Grid physics & power flow — Why grids are hard to operate: instantaneous balance, transmission vs distribution, who decides what

**Type:** Focus lesson
**Main reference:** [Electric Power Systems: A Conceptual Introduction — A. von Meier (2006), Wiley-IEEE Press](https://doi.org/10.1002/0470036427) — read §1.4 (resistive heating, transmission voltage and resistive losses) and skim ch. 9 (system operation, management and new technology). The curriculum lists ch. 1, but ch. 1 turns out to be "The Physics of Electricity", so this lesson pairs its loss section with the operations chapter. (A second edition from 2024 also exists.)
**Further reading:**
- [Balancing and Frequency Control Reference Document — NERC Resources Subcommittee (2021)](https://www.nerc.com/comm/RSTC_Reliability_Guidelines/Reference_Document_NERC_Balancing_and_Frequency_Control.pdf) — Chapter 1, "Balancing Fundamentals" (about 12 pages), is a clear, free explanation of frequency, inertia, the control continuum and area control error.
- [The Future of the Electric Grid: An Interdisciplinary MIT Study — MIT Energy Initiative (2011)](https://per.mit.edu/wp-content/uploads/2014/05/Electric_Grid_Full_Report.pdf) — Appendix B, "Electric Power System Basics" (pp. 243–259): the structure of the power system, how it is operated, and wholesale markets. It is US-centred, but the structure carries over to other systems.

**Why this matters** — Every OPF formulation in this curriculum is a frozen snapshot of a control problem that never stops. To build a learned OPF solver, you need to know which decision it replaces: who makes it, on what timescale, with what information, and what happens in the seconds and minutes after it is wrong. Without that, "feasibility", "reserves" and "redispatch" are just words in a loss function.

**Core ideas**

**1. Balance must hold at every instant.** A power grid has almost no buffer. Except for batteries and pumped hydro, which are small compared with total demand, electrical energy cannot be stored in the network itself. Generation must equal consumption plus losses at every moment. The only short-term "storage" is the kinetic energy of the large synchronous generators, whose rotors all spin at a speed locked to the grid frequency (50 Hz in Europe, 60 Hz in North America). When load exceeds generation, the missing energy is taken from these rotors, so they slow down and frequency falls. This is the *swing equation*, aggregated over the whole system:

```
(2·H·S / f₀) · df/dt = P_gen − P_load        (H = inertia constant in seconds, S = rated power, f₀ = nominal frequency)
```

Frequency is therefore a global, real-time readout of the supply–demand imbalance, the "water level" in NERC's tank analogy. It is the same everywhere in a synchronous area, such as Continental Europe or the US Eastern Interconnection.

**2. A hierarchy of controllers, separated by timescale.** No single decision keeps the balance. A layered stack does, and each layer is slower and more economical than the one below it:

```
seconds        inertia            physics: rotors release kinetic energy (nobody decides)
~seconds–30 s  primary control    governors raise output ∝ frequency drop (droop); stops the fall
minutes        secondary control  automatic generation control (AGC) restores nominal frequency
                                  and scheduled flows between areas
~15 min–hours  tertiary control   operators reschedule units and replace the reserves used up
hours–day      dispatch / OPF     markets and optimization set schedules from forecasts
days–years     commitment, planning, investment
```

ML readers will recognize a cascade of controllers at different timescales. Fast layers are simple feedback rules with fixed gains. Slow layers solve optimization problems using forecasts. OPF lives in the slow layers. It sets the operating point, and the fast layers absorb the forecast error until the next solve. The capacity held back for the fast layers is called *reserves*, and deciding how much to hold is itself an optimization under uncertainty (X002).

In a multi-area system, each balancing area runs its secondary control on the *area control error*. In its basic NERC form:

```
ACE = (NI_actual − NI_scheduled) − 10·B·(f_actual − f_scheduled)      [MW]
```

Here `NI` is the net interchange over tie-lines (in MW) and `B` is the frequency bias in MW/0.1 Hz (a negative number). An area that is short pulls in unscheduled imports and lowers frequency, so its ACE is negative and its AGC raises generation. A neighbour that is in balance sees the two terms cancel and does nothing. Each area fixes its own mismatch, but everyone shares the frequency.

**3. Power goes where physics sends it, not where contracts say.** In a meshed network you cannot route power the way you route packets. Flows split over all parallel paths according to line impedances, by Kirchhoff's laws. A sale from north to south also loads lines in the east and west (loop flows). This is the main reason the *network* constraints make dispatch hard: a cheap generator may be unusable because some distant line would overload. Lessons **A03–A06** make this precise.

**4. Transmission vs distribution: why voltage levels exist.** Line losses are `P_loss = I²R`, and for a fixed power `P = V·I`, so `P_loss ∝ R·P²/V²`. Doubling the voltage cuts losses by a factor of four. That is why bulk power travels at 220–400 kV (or higher) and is stepped down by transformers to medium voltage (10–30 kV) and finally to low voltage (230/400 V in Europe). This is von Meier's §1.4 argument. The result is two very different networks:

- **Transmission:** meshed, few nodes, heavily monitored, run by transmission system operators (TSOs). This is where classical OPF and markets live.
- **Distribution:** mostly radial (tree-shaped), huge numbers of nodes, sparse measurements, run by distribution system operators (DSOs). Historically it was passive, but rooftop PV, heat pumps and EVs make it active, with reverse flows and voltage problems.

**5. Who decides what.** In a liberalized system, the decisions are split across actors:

- *Generators, retailers and aggregators* bid and schedule in markets (day-ahead, intraday) and are financially responsible for their own imbalances (in Europe, as "balance responsible parties").
- *Market operators or exchanges* clear the auctions. In US-style ISO/RTO markets, the system operator itself runs a security-constrained dispatch, i.e. an OPF.
- *TSOs* operate the transmission grid, buy reserves, run balancing and frequency control, and fix congestion that the market ignored (redispatch, **A28**).
- *DSOs* operate the distribution grid and increasingly manage local congestion and voltage.
- *Regulators* set the rules and the cost recovery.

The key fact for a modeller is that **no single actor solves "the" OPF**. A textbook OPF is an idealization: one central planner, one objective, perfect information. Real operation is a sequence of partial optimizations by different actors with different information. In European zonal markets, for example, the network inside each zone is ignored when the market clears, and the TSO repairs the result afterwards.

**6. Why this adds up to "hard".** Balance must hold instantly, physics couples everything, limits apply to thousands of components, demand and renewables are uncertain, decisions are coupled in time (ramps, storage), failures can cascade, and authority is split across actors. Phase A is about the first five of these; phase B adds uncertainty explicitly.

**A small example**

*Losing a generator.* A synchronous area serves 50 GW with aggregate inertia `H = 5 s` on a 50 GW base. A 1 GW plant trips. Just after the trip, by the swing equation:

```
df/dt = f₀ · ΔP / (2·H·S) = 50 · (−1) / (2 · 5 · 50) = −0.1 Hz/s
```

Without any control, frequency would reach 49.8 Hz after 2 s and keep falling. Assume primary control plus frequency-sensitive load gives a combined response of 10 GW/Hz (a toy number). Frequency then settles at `50 − 1/10 = 49.9 Hz` within seconds. Over the next minutes, secondary control in the affected area ramps up 1 GW of reserve and brings frequency back to 50 Hz. Then operators reschedule so that the reserve is free again for the next event.

Now replace half of the synchronous machines with inverter-based wind and PV, so that `H = 2.5 s`. The initial rate of change doubles to −0.2 Hz/s, and primary control has half the time to act. This is the "low-inertia" problem.

*Why high voltage.* Deliver 100 MW over a line with 10 Ω resistance, using a single-phase simplification. At 20 kV, `I = 5000 A` and `P_loss = 5000² · 10 = 250 MW`, which is more than you deliver, so it is impossible. At 380 kV, `I ≈ 263 A` and `P_loss ≈ 0.69 MW`, i.e. 0.7 %.

**How it connects**

**X002** argued that reserve sizing needs the full forecast density. This lesson shows where those reserves act: in the primary and secondary layers, which absorb the error left by the slow, forecast-driven schedule. **X001**'s policy classes fit here too. Droop and AGC are PFAs (fixed-gain rules), while day-ahead scheduling is a DLA. The next lesson, **A02**, gives the AC vocabulary this one avoided (phasors, active and reactive power, per-unit). **A03–A06** then turn "physics routes the power" into the admittance matrix and the power-flow equations. Timescales come back in **A25–A30** (multi-period operation, redispatch, MPC, unit commitment).

**Check yourself**

1. Why is frequency the same everywhere in a synchronous area, while voltage magnitude differs from bus to bus? What does a falling frequency tell an operator?
2. In the example, what happens to the initial rate of frequency change if the lost generator were 2 GW instead of 1 GW, and if inertia were halved at the same time?
3. Two balancing areas are connected by a tie-line. Area 1 loses 500 MW of generation. Explain qualitatively why only Area 1's AGC should respond in the long run, even though both areas see the frequency drop.

<details><summary>Answers</summary>

1. All synchronous machines in the area are electromechanically locked together, so in steady state they rotate at one common electrical speed, and frequency is a system-wide quantity. Voltage magnitudes depend on local flows and reactive power, so they vary from bus to bus (more in **A02**). A falling frequency means that, system-wide, consumption plus losses exceed generation and kinetic energy is being drawn from the rotors.
2. The rate scales as `ΔP / H`. Doubling `ΔP` and halving `H` gives four times the original rate: −0.4 Hz/s.
3. In Area 1, the unscheduled import (negative `NI` error) and the frequency term add up to a negative ACE, so its AGC raises generation. In Area 2, the extra export it delivers through primary control (positive `NI` error) is cancelled by the bias term `−10·B·Δf` (negative, because `B < 0` and `Δf < 0`), so its ACE is about zero. Area 2 helps in the first seconds through primary control but is not asked to fix the imbalance permanently.

</details>

---

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

---

## 2026-10-03 (Saturday) — X001 · Track 04: Stochastic, chance-constrained, risk-averse & robust optimization — One language for decisions under uncertainty: Powell's four policy classes

**Type:** Cross-phase preview
**Main reference:** [Tutorial on Stochastic Optimization in Energy — Part I: Modeling and Policies — W. B. Powell & S. Meisel (2016), IEEE Transactions on Power Systems 31(2):1459–1467](https://doi.org/10.1109/TPWRS.2015.2424974) — read the whole paper (9 pages). Focus on the canonical model and the four policy classes, and skim the notation discussion. ([Princeton record with abstract](https://collaborate.princeton.edu/en/publications/tutorial-on-stochastic-optimization-in-energy-part-i-modeling-and/). The IEEE version is paywalled. The open paper below presents the same framework.)
**Further reading:**
- [From Reinforcement Learning to Optimal Control: A unified framework for sequential decisions — W. B. Powell (2019/2021), arXiv:1912.03513](https://arxiv.org/abs/1912.03513) — open-access, longer version of the same framework. Sections 4 and 6 hold the five model elements and the four policy classes, written for an RL audience.
- [Tutorial on Stochastic Optimization in Energy — Part II: An Energy Storage Illustration — Powell & Meisel (2016), IEEE TPWRS 31(2):1468–1475](https://doi.org/10.1109/TPWRS.2015.2424980) — five variants of one storage problem, built so that each policy class wins on at least one.

**Why this matters** — Power-system operation under uncertainty is studied by several communities: stochastic programming, robust optimization, model predictive control, dynamic programming and reinforcement learning. Each has its own notation, and each believes its method is the natural one. Anyone who builds learned dispatch or OPF solvers under forecast uncertainty has to place their model in this landscape. Is it a policy, a value function, or a fast surrogate inside a lookahead? Without that map, comparisons between methods are mostly accidents of notation.

**Core ideas**

The tutorial's main claim is methodological: *model first, then solve*. You first write down the sequential decision problem without saying how you will solve it. Only then do you choose a class of policies. The model has five elements:

1. **State** `S_t`: everything needed, from now on, to compute costs, constraints and the next state. Powell splits it into a physical state `R_t` (for example a battery's state of charge or generator outputs), other information `I_t` (prices, weather, the current forecast) and beliefs `B_t` (parameters of distributions over quantities you cannot observe). An ML reader should note that *the forecast itself is part of the state*. A model that conditions on a forecast distribution is conditioning on part of `S_t`.
2. **Decision** `x_t`, chosen by a policy `X^π(S_t)`. In energy problems `x_t` is often a high-dimensional continuous vector with constraints, such as setpoints for hundreds of generators. That is why the RL framing in terms of "action spaces" fits poorly and the optimization framing is the default.
3. **Exogenous information** `W_{t+1}`: what you learn between `t` and `t+1`, for example realized wind, load or prices.
4. **Transition** `S_{t+1} = S^M(S_t, x_t, W_{t+1})`. For a battery: `SoC_{t+1} = SoC_t + η·charge_t − discharge_t/η`. In RL this is the environment step, and in control the plant equation.
5. **Objective**: search over policies, not over decisions:

```
max_π  E[ Σ_{t=0..T} C_t(S_t, X^π_t(S_t)) | S_0 ],   S_{t+1} = S^M(S_t, X^π_t(S_t), W_{t+1})
```

This is the RL objective written in operations-research notation. The difference is emphasis. In OR, `C_t` and `S^M` are usually known and structured (linear constraints, power-flow physics), and decisions are vectors that a solver must produce.

**The four policy classes.** The tutorial argues that every practical method falls into one of four classes, or a hybrid of them:

- **PFA (policy function approximation):** a direct map from state to decision, `x_t = f(S_t | θ)`. Examples are an affine rule (`x_t = θ_0 + θ_1·φ(S_t)`), a "charge if price < θ" threshold, or a neural network. Affine balancing rules in power systems are PFAs, as is a learned OPF proxy evaluated without any solver.
- **CFA (cost function approximation):** solve a *deterministic* optimization whose objective or constraints are tuned by `θ`: `x_t = argmax_{x ∈ X_t(θ)} C̄(S_t, x | θ)`. This is what industry mostly does. Examples are reserve margins, forecast inflation, and tightened line limits. Uncertainty enters only through tuned buffers.
- **VFA (value function approximation):** `x_t = argmax_x [ C(S_t, x) + E V̄_{t+1}(S_{t+1}) ]`. This covers Bellman methods, approximate dynamic programming, Q-learning and stochastic dual dynamic programming.
- **DLA (direct lookahead approximation):** at each step, optimize over an *approximate model of the future*, then implement only the first decision. The lookahead can be deterministic (model predictive control with point forecasts) or stochastic (a scenario tree, as in two-stage stochastic programming), and it can be robust (worst case over an uncertainty set).

PFAs and CFAs are found by **policy search**: tune `θ` by simulating the base model. VFAs and DLAs are built by **approximating the lookahead**. The exact lookahead is the Bellman-optimal decision:

```
X*_t(S_t) = argmax_x ( C(S_t, x) + E[ max_π E[ Σ_{t'>t} C(S_{t'}, X^π_{t'}(S_{t'})) | S_{t+1} ] | S_t, x ] )
```

This exact form is almost never computable. A VFA replaces the whole inner term with `V̄`. A DLA replaces the *model* inside it with something smaller: fewer scenarios, a shorter horizon, or a deterministic forecast.

**Two key messages.** First, no class dominates. Part II builds variants of one storage problem and shows that, depending on the problem's characteristics, each class can be the best. Powell's general rule of thumb: PFAs when the policy's structure is obvious, DLAs when forecasts carry strong time-varying information, VFAs when the state is low-dimensional and the problem is stationary. Second, *a deterministic lookahead model is not a deterministic policy*. A rolling MPC with a point forecast is a legitimate stochastic policy, and its quality must be measured by simulating it against the real uncertainty, i.e. evaluating the base objective, not the lookahead objective it optimizes internally. This gap between in-model and out-of-sample performance comes back many times in this curriculum.

**A small example**

A day-ahead commitment, written as a one-step problem. You commit `x` MWh of generation today at 30 €/MWh. Tomorrow demand `D` turns out to be 80, 100 or 120 MWh, each with probability 1/3. A shortfall is covered in real time at 100 €/MWh, and surplus is wasted. Expected cost:

```
J(x) = 30·x + 100·E[max(D − x, 0)]
J(80)  = 2400 + 100·(0 + 20 + 40)/3 = 4400
J(100) = 3000 + 100·(0 + 0 + 20)/3  ≈ 3667
J(120) = 3600 + 0                   = 3600
```

- **Deterministic lookahead on the mean forecast** (plan for `D = 100`) gives `x = 100` and an expected cost of 3667. Inside its own model it expects to pay only 3000.
- **Stochastic lookahead** (minimize `J` over the scenarios) gives `x = 120` and a cost of 3600. This is the newsvendor critical-ratio rule: keep adding capacity while `P(D > x)·100 > 30`, i.e. while `P(D > x) > 0.3`.
- **CFA**: plan for `mean + θ` and tune `θ` by simulation. `θ = 20` recovers the stochastic optimum with a deterministic solver. That is why reserve margins work in practice, and why the tuning has to be redone when the forecast error distribution changes.
- **PFA**: `x = θ·forecast` with `θ = 1.2` does the same here without solving any optimization problem.

Three policies from three classes reach the same decision. The difference shows up when the uncertainty changes from day to day (heteroskedastic forecasts). A fixed buffer `θ` then over- or under-reserves, while a policy that conditions on the forecast *distribution* can adapt.

**How it connects**

This is the first entry in the log, so there are no earlier entries to build on. It previews the phase B unit on decision-making under uncertainty. **B01** (the taxonomy of stochastic, chance-constrained, robust and distributionally robust optimization) and **B02** (two-stage stochastic programming, recourse, SAA) are mostly about building good **DLAs**. **B08**'s affine balancing rules are **PFAs** embedded inside a lookahead. **A29** (receding-horizon MPC) is the textbook deterministic DLA. Read Powell's frame as the "type system" for all of these. Phase C's learned OPF proxies (**C01–C03**) are, in this language, either PFAs (the network *is* the policy) or fast components inside a DLA. **F18**'s out-of-sample evaluation of stochastic decisions is the base-model simulation Powell insists on. Next in the focus phase: **A01**, why grids are hard to operate.

**Check yourself**

1. Why does Powell put the current forecast (and its uncertainty) inside the state `S_t` rather than treating it as a fixed parameter of the problem?
2. A grid operator solves a deterministic DC-OPF every 15 minutes, using the latest point forecast and line limits tightened by 5 %. Which policy class is this, and what is `θ`?
3. In the example, the mean-forecast plan "expects" to pay 3000 but really pays about 3667 in expectation. Which two objectives are being confused, and how should you evaluate such a policy?

<details><summary>Answers</summary>

1. The state must contain everything needed to make decisions and compute what happens next. The forecast is information available at time `t` that changes over time and affects the best decision, so it belongs in `I_t` (and its spread in `B_t`). If you treat it as fixed, the "policy" cannot respond to a day with tight versus wide uncertainty, which is exactly where uncertainty-aware decisions add value.
2. It is a deterministic **DLA** (rolling horizon on a point forecast) whose constraints are parametrically modified. That makes it a **CFA/DLA hybrid**, with `θ` the 5 % tightening (plus any reserve margins). `θ` should be tuned by simulating the policy against realistic forecast errors.
3. The lookahead model's objective (cost under the assumed deterministic future) is being confused with the base model's objective (expected cost under the true distribution of `W`). A policy must be evaluated on the base model: simulate it on many sampled or historical outcomes not used to build it, and average the realized cost and constraint violations.

</details>

---
