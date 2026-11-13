## 2026-11-13 (Friday) — A16 · Track 02: Optimal power flow: formulations, relaxations, solvers — Nonconvexity, local optima and NP-hardness: how bad is AC-OPF, really?

**Type:** Focus lesson
**Main reference:** [Local Solutions of the Optimal Power Flow Problem — W. A. Bukhsh, A. Grothey, K. I. M. McKinnon & P. A. Trodden (2013), IEEE Transactions on Power Systems 28(4), 4780–4788](https://doi.org/10.1109/TPWRS.2013.2274577) ([accepted version, free](https://eprints.whiterose.ac.uk/id/eprint/90482/)) — read §II-C, §III (WB2, the 22-bus loop), §IV (the search method and §IV-E on starting points) and §V; skim §VI on the SDP method, which A20 covers. Companion: [Strong NP-hardness of AC power flows feasibility — D. Bienstock & A. Verma (2019), Operations Research Letters 47(6), 494–501](https://doi.org/10.1016/j.orl.2019.08.009) ([arXiv:1512.07315](https://arxiv.org/abs/1512.07315)) — read §1 and §1.1 for the gadget, and skim the reduction in §1.2.
**Further reading:**
- [AC-Feasibility on Tree Networks is NP-Hard — K. Lehmann, A. Grastien & P. Van Hentenryck (IEEE TPWRS, 2016; arXiv:1410.8253)](https://arxiv.org/abs/1410.8253) — the earlier, weaker result: hardness even on radial networks, by a subset-sum reduction.
- [Local optima test case archive — University of Edinburgh](https://webhomes.maths.ed.ac.uk/OptEnergy/localopt/introduction.html) — data and every known local solution for the cases in the paper. A ready-made stress test for any solver or learned proxy.

**Why this matters** — Every OPF label in a training set comes from a local solver (A14–A15) applied to a nonconvex problem. Anyone who trains on those labels, compares a learned proxy against them or reports an "optimality gap" needs to know when the labels are global optima, when they are not, and how big the difference can get.

**Core ideas**

**1. Where the nonconvexity comes from.** A10's AC-OPF has a convex objective (linear or quadratic cost) and convex box constraints on `|V|`, `p_g` and `q_g`. The nonconvexity sits entirely in the power-flow equalities:

```
p_ij = |V_i|² G_ii + |V_i||V_j| (G_ij cos θ_ij + B_ij sin θ_ij)
```

A nonlinear equality `h(x) = 0` defines a curved surface, and a curved surface is never a convex set. In rectangular coordinates (`V = e + jf`) the flows become quadratics, but the lower voltage bound becomes `e² + f² ≥ V_min²`. That is the outside of a disk, a reverse-convex constraint, so the feasible voltages at each bus form a ring (an annulus). Convex cost plus nonconvex feasible set means the KKT conditions (A13) certify only *local* optimality. ML analogy: a convex loss on a nonconvex parameter manifold.

**2. Two mechanisms that create local optima (Bukhsh et al., §III and §V).**

*(a) Disconnected feasible regions.* The load-flow equations of A04 can have several solutions for the same injections, typically a "high-voltage" and a "low-voltage" branch that meet at the nose of the PV curve (the point of maximum transfer). Once voltage bounds are imposed, the feasible set can split into separate pieces. The best point in each piece is a local optimum, so there are at least as many optima as pieces. In WB2, a 2-bus system with load `350 MW − j350 Mvar` (capacitive) and bounds `[0.95, 1.05]`, the feasible set has two pieces:

| | cost | v₁ | v₂ | θ₂ | p_G (MW) |
|---|---|---|---|---|---|
| S1 (global) | 877.78 | 0.952 | 1.050 | −57.14° | 438.89 |
| S2 (local) | 905.73 | 0.950 | 0.976 | −64.94° | 452.86 |

Both serve the same 350 MW. S2 simply loses about 14 MW more in the line (102.86 against 88.89 MW).

*(b) Loop flows.* On a meshed grid, the voltage angles around a cycle only need to add up to a multiple of `2π`. For a ring of `n` buses with alternating generators and loads, there is a family of solutions

```
θ_k = 2πm (k − 1)/n     (generator buses),       m ∈ ℤ
```

The winding number `m` creates a power flow that circulates round the ring and adds losses without serving any load. For `n = 22`, `z = 0.01 + j0.05` and ±5% voltage bounds, `m = 0` is global and `m = ±1` are local optima 31% more expensive. With ±10% bounds, `m = ±2` also becomes feasible. This is a topological source of multimodality: the local optima are separated by a discrete invariant, not by a small energy barrier.

The paper lists the common ingredients across all its examples: disconnected feasible regions, loop flow, an excess of real or reactive power (for example capacitive loads or lightly loaded cables), and large angle differences across lines.

**3. How often does this happen?** The search method is to start each solver from more than 2000 uniform random points within the variable bounds and accept a point as a local optimum only if first-order conditions hold and Ipopt, KNITRO, SNOPT and MATPOWER/fmincon all return to it from nearby starts. On the *unmodified* IEEE 14-, 24-, 30-, 57-, 118-, 300- and MATPOWER 9- and 39-bus cases, no local optima were found. They appear after modifications: loads scaled to 60% with reactive lower limits raised to −5 Mvar (case9mod: 4 optima, worst 37% above global), loads halved with ±5% voltage bounds (39-bus: local 115% above global), or generator limits scaled up 7× (118-bus: locals 38% and 51% above global).

Starting points matter (§IV-E). Over 649 cost-perturbed instances known to have local optima, Ipopt from a **flat start** (all `|V| = 1`, all `θ = 0`, generators at mid-range) ended at a non-global local optimum in 2.6% of cases and at an infeasible point in 0.3%. From random starts the figures were 23.2% and 1.4%. Hence the folk wisdom "local solutions are usually good enough": on realistic data and with sensible starts, Ipopt's KKT point is very often global. But "usually" is an empirical statement about benchmark distributions, not a theorem.

**4. The worst case: strong NP-hardness (Bienstock & Verma).** They study the simplest AC model: lossless lines (`r = 0`), unlimited reactive power, and only angles as unknowns:

```
N f = b,     sin(θ_i − θ_j) = x_ij f_ij,     |θ_i − θ_j| ≤ θ_ij^max
```

Deciding whether this system is feasible is **strongly** NP-hard. Weak NP-hardness (Lehmann et al., on trees) allows pseudo-polynomial algorithms whose cost grows with the numeric values in the data. Strong hardness rules them out unless P = NP: the problem stays hard even when every number has polynomially many bits. The core is a 5-bus gadget whose flow balance reduces to

```
Δ(θ) = −sin θ + (5/8) sin 2θ ≈ 0,      Δ = 0  ⇔  θ = 0  or  θ = arccos(4/5) ≈ 0.6435
```

so the gadget runs in one of two **modes**: throughput below 0.1592 (Mode I) or between 0.7746 and 0.8867 (Mode II). That is a binary switch built out of trigonometry. Gadgets for variables and clauses, linked by lines with tight limits, encode ONE-IN-THREE 3-SAT, and the question "can the network deliver at least T units to the load?" is equivalent to satisfiability. The lesson for anyone hoping a network will learn OPF "exactly": a worst-case instance family exists where finding even a *feasible* point is as hard as SAT. Practical grids are not that family, which is why the statistics in §3 look so benign.

**A small example**

One lossless line with reactance `x = 0.5` p.u. and `|V₁| = |V₂| = 1` carries `P = sin δ / x = 2 sin δ`, where `δ = θ₁ − θ₂`. To send `P = 1` p.u. we need `sin δ = 0.5`, which has **two** solutions:

| δ | P | reactive power absorbed by the line, `2(1 − cos δ)/x` |
|---|---|---|
| 30° | 1.0 | 2(1 − 0.866)/0.5 = 0.54 p.u. |
| 150° | 1.0 | 2(1 + 0.866)/0.5 = 7.46 p.u. |

Both satisfy the power-flow equations. The second absorbs roughly 14× more reactive power and lies on the far side of the maximum-transfer point `δ = 90°` (`P_max = 2`). An angle-difference limit `θ^max < π/2` removes it, and that is precisely the kind of constraint Bienstock & Verma include, which shows such limits do not rescue worst-case tractability on general networks. As `P → 2` the two roots merge (the nose). Beyond that there is no solution at all, which is the 1-line version of "load too high, OPF infeasible".

**How it connects**

This lesson explains why A13's KKT conditions are only a local certificate and why A15's "Converged to a point of local infeasibility" proves nothing global. WB2's two branches are A05's multiple power-flow solutions with a cost attached. For learning there are three consequences. First, the solution map *load → optimal dispatch* can jump discontinuously: in WB2, tightening `v₂^UB` below 1.035 makes S1 infeasible and the optimum jumps to the other branch. That is the setting where MSE regression averages modes and mixture outputs help (X015). Second, labels depend on the solver and its start, so they carry non-aleatoric "label noise" (A15, check question 3). Third, an "optimality gap" for a learned proxy measured against Ipopt is a gap to a *local* reference. Next, A17 shows the modelling stacks that produce these labels and A18 asks why warm starts help so little. A19–A22 then build convex relaxations that give *lower bounds*, so that a local solution can be certified as near-global. Bukhsh et al.'s §VI already previews A20: in WB2 the SDP method fails exactly when `0.976 < v₂^UB < 1.035`.

**Check yourself**

1. The AC-OPF objective and all its bound constraints are convex. Point to the constraints that make it nonconvex in polar form, and to the one bound that is itself nonconvex in rectangular form.
2. Lehmann et al. proved AC feasibility NP-hard on trees, and Bienstock & Verma proved it *strongly* NP-hard. What does the extra word rule out, and does either result say that Ipopt will struggle on a 300-bus benchmark?
3. A dataset is generated from WB2 by sampling `v₂^UB` uniformly in `[1.00, 1.05]`, and an MLP is trained with MSE to predict `v₂` at the optimum. What will go wrong near `v₂^UB = 1.035`, and what would a better model output?

<details><summary>Answers</summary>

1. In polar form, the nonlinear power-balance and flow equalities (`cos`/`sin` terms in `θ_ij`, products `|V_i||V_j|`): a curved equality surface is never convex. In rectangular form the flows become quadratic equalities (still nonconvex), and in addition the lower voltage bound `e_i² + f_i² ≥ V_min²` is reverse-convex: it excludes a disk, leaving an annulus.
2. "Strong" rules out pseudo-polynomial algorithms, whose running time is polynomial in the size of the network and in the magnitudes of the numbers (unless P = NP). A weakly NP-hard problem can still be easy when the numbers are small. Neither result says anything about typical instances. They are worst-case statements about specially built families. On unmodified IEEE cases, Bukhsh et al. found no local optima at all from more than 2000 random starts.
3. For `v₂^UB ≥ 1.035` the optimum is S1 with `v₂ = v₂^UB` (the bound is active). For `v₂^UB < 1.035` S1 is infeasible and the optimum jumps to the other branch at S2 (`v₂ = 0.976`). The target is discontinuous, so an MSE regressor smooths through the jump and predicts intermediate voltages near the threshold that belong to neither branch and may be infeasible. Better options are to predict a branch indicator plus a per-branch regression (a mixture or classification head, X015), or to output both candidates and let a solver or feasibility check choose.

</details>
