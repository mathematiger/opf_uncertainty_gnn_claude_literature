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

