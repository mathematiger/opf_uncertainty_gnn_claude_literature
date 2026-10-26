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
