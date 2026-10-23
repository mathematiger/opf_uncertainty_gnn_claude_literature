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
