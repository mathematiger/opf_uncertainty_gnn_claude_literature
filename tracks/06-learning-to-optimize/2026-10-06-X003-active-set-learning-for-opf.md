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
