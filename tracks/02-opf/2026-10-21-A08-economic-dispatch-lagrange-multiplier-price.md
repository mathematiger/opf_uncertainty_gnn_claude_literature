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
