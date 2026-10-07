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
