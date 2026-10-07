# Reading log

All entries, newest first. See [`curriculum.md`](curriculum.md) for the plan.

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
