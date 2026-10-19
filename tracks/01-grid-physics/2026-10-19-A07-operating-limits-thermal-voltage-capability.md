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
