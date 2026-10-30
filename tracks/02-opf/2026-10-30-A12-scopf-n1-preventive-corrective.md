## 2026-10-30 (Friday) — A12 · Track 02: Optimal power flow: formulations, relaxations, solvers — Security-constrained OPF: surviving N-1, preventively or correctively

**Type:** Focus lesson
**Main reference:** [State-of-the-art, challenges, and future trends in security constrained optimal power flow — F. Capitanescu, J. L. Martinez Ramos, P. Panciatici, D. Kirschen, A. Marano Marcolini, L. Platbrood & L. Wehenkel (2011), Electric Power Systems Research 81(8):1731–1741](https://orbi.uliege.be/handle/2268/88635) — open author preprint on ORBi ([DOI](https://doi.org/10.1016/j.epsr.2011.04.003)). Read §1–3.1 (motivation, the formulation (1)–(8), limits on corrective actions) and §4.1 (reducing problem size: contingency filtering, Benders, linearization). Skim §3.2 (stability), §4.3 (discrete variables) and §5 (stochastic and worst-case SCOPF); this lesson is an overview only.
**Further reading:**
- [Recent Developments in Security-Constrained AC Optimal Power Flow: Overview of Challenge 1 in the ARPA-E Grid Optimization Competition — I. Aravena, D. K. Molzahn et al. (2022), arXiv:2206.07843](https://arxiv.org/abs/2206.07843) — a precise modern SC-AC-OPF formulation with generator response to contingencies, plus how the top competition teams solved it at scale; good to revisit after A13–A15.

**Why this matters** — Real operators do not solve "the" OPF of A10: they solve one that must stay feasible after any single equipment failure, and that requirement often costs more than every other constraint combined. It multiplies the problem size by the number of contingencies, which is exactly why learned solvers, contingency screening and warm starts attract interest here. Anyone building learned OPF solvers under forecast uncertainty needs to know what "secure" means, because a proxy that is feasible only in the intact network is not deployable.

**Core ideas**

**1. The N-1 criterion.** A01 introduced the rule: the grid must survive the loss of any single element (a line, transformer or generator) without violating limits and without load loss. Each such outage is a *contingency* `k = 1, …, c`, and `k = 0` is the intact base case. In ML terms, the base-case decision must be robust against a finite, enumerated set of perturbations to the graph itself: deleting an edge changes the admittance matrix (A03) and so the whole flow pattern.

**2. The conventional SCOPF.** Capitanescu et al. write it compactly (their eq. 1–8):

```
min_{x_0..x_c, u_0..u_c}  f_0(x_0, u_0)
s.t.  g_0(x_0, u_0) = 0,      h_0(x_0, u_0) ≤ L^l                  (base case, normal limits)
      g_k^s(x_k^s, u_0) = 0,  h_k^s(x_k^s, u_0) ≤ L^s     k=1..c   (just after outage, emergency limits)
      g_k(x_k, u_k) = 0,      h_k(x_k, u_k) ≤ L^m         k=1..c   (after correction, medium-term limits)
      |u_k − u_0| ≤ Δu_k = T_k · du_k/dt                   k=1..c   (coupling)
```

`u` are controls (generator P and voltage setpoints, transformer taps, shunts, phase shifters, breaker states) and `x` are states (bus voltages and angles). The `g` are A04's AC power-flow equations in the respective topology; the `h` are A07's limits. Note the three limit sets, with `L^l ≤ L^m ≤ L^s`: equipment can tolerate a higher current for a short time (A07's `RATE_A/B/C`). The paper's French example has three line current limits with disconnection delays of 1, 5 and 20 minutes. The operator cannot act within one minute, so the first limit must hold by preventive action alone; one corrective action (typically a predefined topology switch) is expected within 5 minutes; satisfying the third limit may need several, such as redispatch or starting fast units.

Only the base-case cost `f_0` is minimised: post-contingency actions are treated as free because they are rarely used. The `c` copies of the network share only the controls `u_0` through the coupling constraints, so the constraint matrix is block-angular, which is the structure Benders decomposition (and later, parallel solvers) exploit.

**3. Preventive vs corrective.** The **preventive** SCOPF (PSCOPF) forbids non-automatic post-contingency actions: `u_k = u_0` for all `k`, so the base case must already be safe for every outage (only automatic responses such as frequency-control droop and tap changers act). The **corrective** SCOPF lets each contingency have its own recovery controls within the ramp budget `Δu_k`. Corrective is cheaper, since it buys security with an option to act later instead of an up-front redispatch, but riskier: the actions must actually work in the available time, the short-term state before they act must be viable, and the sequence must not make things worse. The paper notes that market pressure pushes operators towards corrective control, and that in some systems N-1 cannot be met at all without it. This is a two-stage decision problem: first-stage `u_0`, recourse `u_k` per scenario, the template B02 makes general.

**4. Why it is hard.** Size: the paper cites a European extra-high-voltage model with about 13,000 buses and about 30,000 N-1 contingencies, and a North American Eastern Interconnection model with about 43,000 buses and about 70,000 contingencies. Stacking all of them gives an AC problem tens of thousands of times larger than one OPF. On top come discrete controls (switching, taps, unit start-ups), which make it a MINLP, and stability, which the static model ignores entirely; the paper's §3.2 surveys how transient and voltage stability are bolted on as extra constraints. Limiting the number of corrective actions per contingency needs binaries: `−s_k Δu_k ≤ u_k − u_0 ≤ s_k Δu_k`, `1ᵀs_k ≤ N_k`, `s_k ∈ {0,1}` (eq. 9–11).

**5. How it is solved in practice.** The key empirical fact: *most contingencies do not constrain the optimum*. So practical methods (§4.1) iterate:

1. Solve the SCOPF with a small subset of contingencies (initially none).
2. Run a cheap security analysis: a post-contingency power flow (A05) for every outage, or DC screening with LODFs (A06).
3. A contingency filter picks the violated or "dominating" ones; add them and repeat.

This is lazy constraint generation, the same idea as cutting planes or active-set methods. Filters use severity indices or Lagrange multipliers of a relaxed problem; the parameter-free "non-dominated contingency" filter compares violations across contingencies. Alternatives shrink each contingency instead: Benders decomposition (exact only for convex problems, so careful with AC), linearizing post-contingency constraints around the base case (with DC models, this is the DC-SCOPF that sets LMPs in markets such as PJM), or network compression that keeps only the electrically nearby area exactly. Solutions from simplified models must be re-checked against the full model.

**6. Beyond N-1.** §5 argues that equal-weight, deterministic contingency lists are outdated: renewable forecast errors act like continuous "injection contingencies". Two directions follow. A **stochastic SCOPF** minimises `f_0 + λ Σ_k p_k f_k(x_k, u_0, u_k)`, weighing contingencies by probability and pricing corrective actions. A **worst-case (robust) SCOPF** is a bi-level max–min: an adversary picks injections in a box `[i_min, i_max]` to maximise the post-contingency violation that the best preventive and corrective actions leave. These are Phase B's stochastic and robust programs (B02, B08) applied to security.

**A small example**

Reuse A11's three-bus triangle (equal reactances, bus 3 reference, load at bus 3; generator 1 at bus 1 costs 10 $/MWh, generator 2 at bus 2 costs 30 $/MWh). Now let all lines be rated 250 MW except line 1–2, rated 150 MW, and lower the load to 240 MW.

- **Ordinary DC-OPF:** `P_1 = 240`, `P_2 = 0`. Base flows `F_13 = 160`, `F_12 = 80`, `F_23 = 80`, all within limits. Cost 2400 $/h.
- **Contingency "line 1–3 out":** the grid becomes the path 1–2–3, so all of bus 1's output crosses line 1–2: `F_12 = P_1 = 240 > 150`. The OPF solution is not N-1 secure. (The other two outages are fine: without 1–2, `F_13 = P_1 = 240 ≤ 250`; without 2–3, `F_13 = 240` and `|F_12| = P_2 = 0`.)
- **Preventive SCOPF:** require `P_1 ≤ 150` in advance, so `P_2 = 90`. Cost 1500 + 2700 = 4200 $/h. The price of security is 1800 $/h, paid every hour although the outage is rare. At this solution the binding constraint is a *post-contingency* flow limit, so it also sets the prices: an extra MW at bus 3 must come from generator 2, giving `LMP_3 = 30`.
- **Corrective SCOPF:** suppose after the outage generator 1 can ramp down and generator 2 up by `Δ = 50` MW within the allowed time, and line 1–2 has a short-term emergency rating of 200 MW. Then the base case only needs `P_1 ≤ 200` (short-term: `F_12 = P_1 ≤ 200`; after correction: `P_1 − 50 ≤ 150`). Take `P_1 = 200`, `P_2 = 40`: cost 2000 + 1200 = 3200 $/h, saving 1000 $/h relative to preventive, provided the 50 MW of corrective redispatch actually happens on time.

**How it connects**

The constraints are those of A10 (AC) or A11 (DC), copied once per topology. Post-contingency flows come from A05's power flow or A06's LODFs; the three limit sets refine A07's ratings, and A01 introduced the N-1 rule. X013 used N-1 line removals as a test of whether GNNs generalise across topology, and X012 mentioned a learned proxy for N-1 security-constrained DC-OPF with tens of millions of variables; this lesson is the problem behind both. The block-angular structure and its decomposition become precise once A13 gives Lagrangian duality and A14–A15 explain interior-point solvers. A30 adds the unit-commitment layer the paper mentions, and B02/B08 generalise the preventive–corrective split to two-stage stochastic programs and affine recourse policies.

**Check yourself**

1. In the conventional SCOPF, what is the role of the coupling constraint `|u_k − u_0| ≤ Δu_k`, and what does the preventive SCOPF set it to?
2. Why do practical SCOPF solvers not simply include all 30,000 contingencies at once, and what property of real systems makes the iterative approach work?
3. In the three-bus example, what corrective ramp `Δ` would make the corrective SCOPF reach the unconstrained cost of 2400 $/h, and which other limit would then bind?

<details><summary>Answers</summary>

1. It limits how far each contingency's recovery controls may move away from the base-case controls, with `Δu_k = T_k · du_k/dt` set by the time available and the ramp rates. It is what makes `u_0` matter for the post-contingency states. The preventive SCOPF sets `Δu_k = 0` (for non-automatic controls), i.e. `u_k = u_0`.
2. Memory and CPU time would be prohibitive (the problem grows roughly by a factor of `c`), and the many redundant constraints can cause numerical trouble. It works because in practice only a small subset of contingencies is binding at the optimum, so constraint generation with a cheap screening step finds the same optimum with a fraction of the constraints.
3. With `P_1 = 240` the condition after correction is `240 − Δ ≤ 150`, so `Δ ≥ 90` MW (and generator 2 must be able to ramp from 0 to 90). But the short-term state, before the correction acts, then has `F_12 = 240`, above the 200 MW emergency rating. So the short-term limit binds first: the corrective SCOPF cannot go beyond `P_1 = 200` unless the emergency rating is at least 240 MW.

</details>
