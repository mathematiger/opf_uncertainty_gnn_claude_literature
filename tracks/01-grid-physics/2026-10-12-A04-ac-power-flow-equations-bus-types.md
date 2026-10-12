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
