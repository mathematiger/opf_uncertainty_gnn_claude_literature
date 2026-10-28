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
