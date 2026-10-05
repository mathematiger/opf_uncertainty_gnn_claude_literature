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
