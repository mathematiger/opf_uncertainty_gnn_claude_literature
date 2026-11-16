## 2026-11-16 (Monday) — A17 · Track 02: Optimal power flow: formulations, relaxations, solvers — Modelling stacks: separating the problem, the formulation and the solver (PowerModels.jl, MATPOWER, pandapower)

**Type:** Focus lesson
**Main reference:** [PowerModels.jl: An Open-Source Framework for Exploring Power Flow Formulations — C. Coffrin, R. Bent, K. Sundar, Y. Ng & M. Lubin (2018), Power Systems Computation Conference (PSCC)](https://arxiv.org/abs/1711.01728) — read all of it (8 pages): §II for the modelling-language idea, §III-A/B for the problem/formulation split and the `V_i V_j*` mappings (eqs. 1–4), §III-D on proving infeasibility, §IV-C on the five-formulation study. The code in the paper uses old APIs (PowerModels v0.5, JuMP v0.18). For current syntax see the [PowerModels quick-start guide](https://lanl-ansi.github.io/PowerModels.jl/stable/quickguide/).
**Further reading:**
- [JuMP 1.0: Recent improvements to a modeling language for mathematical optimization — Lubin, Dowson, Dias Garcia, Huchette, Legat & Vielma (arXiv:2206.03866, 2023)](https://arxiv.org/abs/2206.03866) — the layer underneath PowerModels: how JuMP talks to solvers through MathOptInterface.
- [pandapower OPF documentation](https://pandapower.readthedocs.io/en/latest/opf/formulation.html) — how the same OPF is declared through table columns (`max_p_mw`, `max_loading_percent`, `create_poly_cost`) rather than equations.

**Why this matters** — Every OPF dataset, label and baseline number in the learning-to-optimize literature was produced by one specific software stack, with its own formulation, defaults and solver. To reproduce a dataset, generate one or trust a reported optimality gap, you need to know which part of a result comes from the physics and which from the tool.

**Core ideas**

**1. Three layers, and why they are kept apart.** A mathematical programming language (AMPL, GAMS, Pyomo, CVX, JuMP) lets you write an optimization problem as algebra and hands it to a solver through a standard interface. The paper's §II makes the key point: the *specification* of a problem is independent of the *algorithm* that solves it. Within one problem class (an LP, say), you can swap Clp for Gurobi without touching the model. If a change to the model moves it into another class (LP to SDP), only the solver changes. ML analogy: a model written in PyTorch is independent of the backend (CPU, CUDA) that runs it.

PowerModels adds one more layer on top of JuMP. The stack is:

```
network data  (MATPOWER .m file → Julia Dict)
      │
problem spec  (OPF, PF, OTS, TNEP …)       "what constraints exist"
      │  ×
formulation   (ACP, ACR, DCP, SOC, QC …)   "how each constraint is written"
      │
JuMP model  →  MathOptInterface  →  solver (Ipopt, Gurobi, Mosek …)
```

**2. The central trick: write the problem once over `V_i V_j*`.** Look back at A04 and A10. Every power-flow equation is linear in the products `V_i V_j*` of complex voltages (`*` is the complex conjugate). The flow from bus `i` to `j` is `S_ij = Y_ij* (|V_i|² − V_i V_j*)` (up to the shunt term). PowerModels writes each *problem* (for example `build_opf`) only in terms of these generic products, as a list of per-component calls: a voltage variable per bus, a Kirchhoff current-law constraint per bus, Ohm's law per branch, thermal and angle-difference limits per branch. A *formulation* is then a rule for replacing the generic product with real variables:

```
ACP (polar):        V_i V_j*  ⇒  |V_i||V_j| cos(θ_i−θ_j) + j |V_i||V_j| sin(θ_i−θ_j)
ACR (rectangular):  V_i V_j*  ⇒  v_i^R v_j^R + v_i^I v_j^I + j (v_i^I v_j^R − v_i^R v_j^I)
DCP (DC approx.):   V_i V_j*  ⇒  0 + j (θ_i − θ_j)
SOC / QC:           V_i V_j*  ⇒  W_ij  (a lifted variable) + convex constraints linking the W's
```

Multiplying the problem list by the formulation list gives a whole grid of models. You call it like `solve_opf("case24_ieee_rts.m", SOCWRPowerModel, Ipopt.Optimizer)`. Only the formulation argument changes between ACP, DCP and SOC. You can build the JuMP model without solving it via `instantiate_model(file, ACPPowerModel, build_opf)` and then inspect it. The same pattern extends to other problems. Optimal transmission switching reuses the OPF functions and replaces each branch constraint with an `_on_off` version that includes a binary switching variable.

**3. Why the split pays off (§III-C, §III-D).**
- *Fair comparisons.* With the problem, data and solver held fixed, any difference in results is due to the formulation. That rules out a common bug in papers that compare relaxations implemented in different codebases.
- *Sanity checks for free.* SOC and QC are relaxations of AC (A19–A22 will show why), so their objectives must lie *below* every AC local optimum. If they do not, something is wrong.
- *Infeasibility proofs.* Recall from A15–A16 that when Ipopt reports "local infeasibility" on the nonconvex AC model, you cannot tell whether the problem is truly infeasible or the solver merely got stuck. If a *convex* relaxation is infeasible, the AC problem is provably infeasible too, because the relaxation's feasible set contains the AC feasible set. This is a cheap way to clean broken network data.

**4. What the paper measured (§IV).** The study used the 108 networks of PGLib-OPF v17.08 (A24 covers this library). All PowerModels models were solved with Ipopt (MA27 linear solver, tolerance 10⁻⁶) and compared against MATPOWER 6.0 with its default solver. Every case under 1000 buses solved in under a second in every formulation. On the larger cases the AC-polar objectives matched MATPOWER's exactly wherever MATPOWER found a solution. MATPOWER's own solver was often faster but failed more often. DC objectives differed slightly, because the two tools implement slightly different DC variants. That detail is a warning that even "DC-OPF" is not one model. Across formulations, the three nonconvex AC forms (polar, rectangular, and a "w–θ" form) reached nearly the same objectives, with rectangular usually fastest. QC gave tighter (smaller) gaps than SOC, and SOC ran faster.

**5. The other two stacks you will meet.**
- **MATPOWER** (MATLAB/Octave). Its `.m` case format is the de facto data standard: PGLib, PowerModels and pandapower all read it. `runopf(mpc)` solves AC-OPF with **MIPS**, MATPOWER's own primal–dual interior-point solver (A14), by default. Other solvers are chosen with `mpoption('opf.ac.solver', 'IPOPT')` (also FMINCON, KNITRO and others).
- **pandapower** (Python). You build a network as pandas tables, and OPF constraints are table columns: `min_vm_pu`/`max_vm_pu` on buses, `max_loading_percent` on lines and transformers, `min_p_mw`/`max_p_mw` and a `controllable` flag on generators, static generators, loads and storage. Costs come from `create_poly_cost` or `create_pwl_cost`. `pp.runopp(net)` calls PYPOWER's interior-point OPF, starting by default from the centre of the operational limits. pandapower can also call PowerModels through its PandaModels.jl interface. The docs warn that OPF "will not start if constraints are not defined".

**Practical consequence for dataset builders.** The same nominal problem run in two stacks can differ in: the DC variant, whether angle-difference limits are enforced, whether line limits are on current or apparent power (A07), the starting point (flat, mid-range or a power-flow solution), the solver's tolerance, and therefore *which* local optimum you get (A16). A reproducible dataset records all of these, not just "AC-OPF solved with Ipopt".

**A small example**

The paper's Code Block 1 is a 3-bus DC-OPF written in raw JuMP. Loads are `1, 2, 4` p.u. at buses 1–3. Generator costs are `1, 10, 100` per p.u. Line susceptances are `b₁₂ = 10`, `b₁₃ = 20`, `b₂₃ = 30`, and the flows on lines 1–2 and 1–3 are limited to ±0.5. Set `θ₁ = 0`. The DC power balances (A06) give the generator outputs as affine functions of the two free angles:

```
p₁ = 1 − 10θ₂ − 20θ₃
p₂ = 2 + 40θ₂ − 30θ₃
p₃ = 4 − 30θ₂ + 50θ₃
cost = 421 − 2610 θ₂ + 4680 θ₃,     θ₂ ∈ [−0.05, 0.05],  θ₃ ∈ [−0.025, 0.025]
```

The cost is linear over a box, so the optimum is the corner `θ₂ = 0.05`, `θ₃ = −0.025`. That gives `p = (1, 4.75, 1.25)` with cost `173.5`, and all outputs are non-negative, so the corner is feasible. The surprise: the cheapest generator produces only its own local load. Line 1–2 carries 0.5 *into* bus 1 and line 1–3 carries 0.5 *out*. Bus 1 becomes a transit node for power from bus 2, the bus-2 generator relieves the very expensive one at bus 3, and with every generator strictly inside its limits the LMPs (A11) are simply `1, 10, 100`. The PowerModels version of this model is the single call `solve_opf(data, DCPPowerModel, HiGHS.Optimizer)`. Swapping `DCPPowerModel` for `ACPPowerModel` and the LP solver for Ipopt gives the AC version of the same problem.

**How it connects**

This lesson turns A10's formulation choices (polar vs rectangular) into a software design: one problem, many formulations. It also explains where the DC model of A06 and A11 lives in code. The infeasibility-proof argument completes A15–A16: a local solver cannot prove AC infeasibility, but a relaxation can. Learned OPF proxies like those in X012 and X013 are trained on labels that stacks like these produce, so the labels carry each stack's defaults. Next, A18 asks why warm-starting the interior-point solvers behind these stacks helps less than ML intuition suggests. A19–A23 then open up the SOC, QC and LPAC formulations that appear here as one-line arguments, and A24 covers the PGLib benchmark used in this paper's experiments.

**Check yourself**

1. In PowerModels, what exactly does a "formulation" change, and what stays the same, when you switch from `ACPPowerModel` to `SOCWRPowerModel` in `solve_opf`?
2. Ipopt reports "locally infeasible" on an AC-OPF instance. How can a modelling stack such as PowerModels help you decide whether the data is truly infeasible, and why does the argument go only one way?
3. Two papers report "AC-OPF optimal costs" on the same PGLib case and disagree by 0.3%. Name three stack-level reasons this could happen even if both are bug-free.

<details><summary>Answers</summary>

1. The network data, the problem specification (which variables, which constraints for each bus and branch, the objective) and the solver stay fixed. The formulation changes how the generic voltage products `V_i V_j*` are written in real variables: polar trigonometric expressions for ACP, and lifted variables `W_ij` with second-order-cone constraints for SOC. As a result the problem class changes from a nonconvex NLP to a convex SOCP.
2. Solve a convex relaxation (SOC or QC) of the same instance. Its feasible set contains the AC feasible set, so if the relaxation is infeasible (which a convex solver can certify) the AC problem is infeasible too. The converse fails: a feasible relaxation does not imply a feasible AC problem, because the relaxation may contain points that satisfy no AC power flow. In that case the result is inconclusive.
3. Examples: different local optima reached from different starting points or solvers (A16); different constraint modelling, such as thermal limits on current vs apparent power, or angle-difference limits enforced or not; different solver tolerances or linear solvers; different parsing of the case data (cost models, transformer conventions). The paper itself observed that the DC variants in MATPOWER and PowerModels differ slightly.

</details>
