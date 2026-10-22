## 2026-10-22 (Thursday) — X012 · Track 06: Learning to optimize, OPF proxies, decision-focused learning — Proxies you can trust by construction: Van Hentenryck's "Optimization Learning"

**Type:** Cross-phase preview
**Main reference:** [Optimization Learning — P. Van Hentenryck (2025), arXiv:2501.03443 [math.OC]](https://arxiv.org/abs/2501.03443) — read Sections 1–2 (the learning task and the two proxy definitions), Section 3 (the economic-dispatch model, Model 1), Section 5.2 (dedicated repair layers, eqs. 5–6) and Section 6 up to Model 2 (dual proxies and the LP completion layer). Then read Section 5.5 (real-time risk assessment) and skim Tables 2–5 and Section 7 (primal-dual learning for security-constrained OPF). It is about 22 pages plus references, written up from the author's EURO-2024 keynote. The link goes to the open arXiv version (v1, January 2025). No journal version is listed.
**Further reading:**
- [Dual Conic Proxies for AC Optimal Power Flow — G. Qiu, M. Tanneau & P. Van Hentenryck (2024), PSCC 2024, arXiv:2310.02969](https://arxiv.org/abs/2310.02969) — the first dual proxy, for the SOCP relaxation of AC-OPF. Its completion layer uses the structure of the dual optimum to fill in the variables the network does not predict.
- [Dual Lagrangian Learning for Conic Optimization — M. Tanneau & P. Van Hentenryck (2024), arXiv:2402.03086](https://arxiv.org/abs/2402.03086) — generalizes the "predict some duals, complete the rest so every output is dual-feasible" idea to general conic programs.

**Why this matters** — A neural network that maps loads to generator setpoints is fast but almost never exactly feasible, and it gives no hint of how far from optimal it is. Anyone building learned OPF solvers has to decide how to fix both problems. This essay gives a compact, opinionated answer: build the network so its output is feasible *by construction* (primal proxy), and learn a second network whose output is a *certified* lower bound (dual proxy). Reading it now gives the vocabulary that phase C makes precise.

**Core ideas**

**The learning task.** Grid operators solve the same parametric problem again and again (every five minutes in US real-time markets) with slightly different inputs `x` (loads, costs, limits):

```
Φ(x) = argmin_y f_x(y)   s.t.   h_x(y) = 0,  g_x(y) ≥ 0
```

The instances come from a distribution `P` estimated from history and forecasts. The paper asks for two learned maps:

- a **primal proxy** `Φ↑` whose output is feasible for every `x ~ P`: `h_x(Φ↑(x)) = 0` and `g_x(Φ↑(x)) ≥ 0`;
- a **dual proxy** `Φ↓` whose output gives a valid lower bound on the optimal cost `f_x(Φ(x))`.

Taken together they give, in milliseconds, a feasible decision and a certificate of how good it is. That is the learned version of a solver's primal–dual pair with its duality gap.

**Why plain regression is not enough (Section 4).** Supervised learning (solve instances offline, regress `y` on `x`) gives a function approximator, not a proxy. A regressor's outputs are only approximately right, so they violate constraints. Adding Lagrangian penalty terms `λ|h_x(ŷ)| + ν max(0, −g_x(ŷ))` and updating `λ, ν` by subgradient steps reduces violations. But the multipliers are shared across all instances, and the paper notes that on unseen instances this "almost always violates constraints".

**Primal proxies = network + repair layer (Section 5).** The proxy is `R ∘ M_θ`. Here `M_θ` is a neural network, and `R` maps its raw output into the feasible set. The generic choice is a projection `R(ŷ) = argmin_y ‖y − ŷ‖` over the feasible set. You differentiate it with the implicit function theorem applied to its KKT conditions (the OptNet / implicit-layer idea, **C09–C10**). That works but is expensive. The paper's main point is that cheap, problem-specific repair layers often exist. For economic dispatch (a single power-balance equation, as in **A08**), a sigmoid enforces generator bounds, and then all generators are scaled *proportionally* toward their upper or lower limits until supply equals demand:

```
if 1ᵀp̂ < 1ᵀd:  p̃ = (1 − ζ↑) p̂ + ζ↑ p̄,   ζ↑ = (1ᵀd − 1ᵀp̂) / (1ᵀp̄ − 1ᵀp̂)
else:          p̃ = (1 − ζ↓) p̂ + ζ↓ p̲,   ζ↓ = (1ᵀp̂ − 1ᵀd) / (1ᵀp̂ − 1ᵀp̲)
```

Each `p̃` is a convex combination of points inside the generator boxes, so it stays inside the bounds. It is differentiable almost everywhere, so the whole pipeline (called E2ELR) trains end to end. A second layer of the same kind repairs the reserve constraint. Power engineers will recognize this as the participation-factor idea from automatic generation control, turned into a neural-network layer.

**Self-supervised training.** Because the optimization problem is known in closed form, you don't need solver labels. You minimize the problem's own objective through the repair layer:

```
min_θ (1/n) Σ_i f_{x_i}(Φ↑_θ(x_i))
```

The loss is the operating cost (plus penalties on soft thermal limits), and feasibility is enforced by the architecture. Supervised and self-supervised training take comparable time. Self-supervised training skips the offline solves (65.9 CPU-hours for the 30,000-bus case) and often reaches smaller gaps.

**Dual proxies = network + completion layer (Section 6).** Consider an LP with bounded variables, `min cᵀy` subject to `Ay = b` and `l ≤ y ≤ u`. Its dual has the variables `(z, z_l, z_u)`, the constraints `zA + z_l − z_u = c` and `z_l, z_u ≥ 0`, and the objective `bᵀz + lᵀz_l − uᵀz_u`. The trick: predict only `ẑ` (one multiplier per equality constraint, i.e. the prices). Then set the bound multipliers from the reduced costs:

```
z_l = max(0, c − ẑA),   z_u = max(0, −(c − ẑA))
```

Any `ẑ` gives a dual-feasible point, so by weak duality its objective is a valid lower bound. Train by *maximizing* that bound. The network can be arbitrarily wrong and the certificate is still valid. Only how tight it is depends on learning.

**Primal-dual learning (Section 7).** Here a second network predicts instance-specific multipliers and the two networks are trained by mimicking the augmented Lagrangian method. This is lesson **C05**. The showcase is security-constrained DC-OPF under N-1 contingencies. The full formulation is a mixed-integer program with up to 64.9 million continuous variables. On the largest case, the state-of-the-art decomposition algorithm needs at least 4,648 seconds per instance. The proxy answers in about 10 ms.

**Findings worth remembering.** On PGLib cases with 300–30,000 buses, E2ELR's mean optimality gaps (which include penalties for violations) are below 2 % in every case and 0.19–0.33 % on the three largest grids with self-supervised training. The plain network never satisfies power balance, and the DeepOPF and DC3 baselines satisfy it on between 1 % and 100 % of instances, depending on the grid. A batch of 256 instances takes about 5–12 ms on a GPU. Gurobi on one CPU needs 12.6 ms to 4.7 s *per instance*. In a Monte-Carlo risk dashboard (288 dispatches per 24-hour scenario), the solver takes about 15 minutes per scenario and the proxy about 5 seconds. Proxies without feasibility guarantees report many *false* constraint violations there, so the risk estimate itself goes wrong. The dual proxy on DC-OPF reaches geometric-mean gaps of 0.23–0.50 %. Read these numbers with phase F in mind: the setting is GPU batch inference against a single-thread CPU solver, and loads are perturbed synthetically (a uniform global scaling in [0.8, 1.2] times 5 % log-normal noise). **C11**, **F16** and **F17** look at both choices critically.

**A small example**

**Repair.** Two generators, `p̲ = (0, 0)`, `p̄ = (100, 50)` MW, demand `d = 120`. The network outputs `p̂ = (80, 20)`, which falls 20 MW short. Then `ζ↑ = (120 − 100)/(150 − 100) = 0.4`, and `p̃ = 0.6·(80, 20) + 0.4·(100, 50) = (88, 32)`. That sums to 120 and is within the bounds. Each generator takes 40 % of its remaining headroom.

**Certificate.** Costs `c = (10, 30)` €/MWh. The true optimum dispatches the cheap unit fully, `(100, 20)`, at a cost of 1,600 €. Suppose the dual network predicts the price `ẑ = 25`. The reduced costs are `c − ẑ·(1, 1) = (−15, 5)`, so `z_u = (15, 0)` and `z_l = (0, 5)`. The bound is `120·25 + 0 − 100·15 = 1,500 €`, which is valid with a 6.25 % gap. With `ẑ = 30` (the true marginal price, set by the expensive unit) the reduced costs are `(−20, 0)` and the bound is `3,600 − 2,000 = 1,600 €`, which is exact. So learning the dual means learning the price from **A08**.

**How it connects**

This entry continues **X003**, which kept an exact solver behind a learned guess (active sets). Here the safeguard is built into the network instead: a repair layer for feasibility and duality for quality. **X009** made the same "learn inside a certified procedure" point for MILPs, and **X007** (Lagaris) built constraints into the architecture for differential equations, the same move as a repair layer. The dual proxy is **A08**'s Lagrange multiplier as a price, now predicted per instance, and the PTDF constraints in the paper's models come from **A06**. Coming up: **A11** (DC-OPF duals as locational prices), **A12** (security-constrained OPF) and **A13** (KKT, duality) give the theory used here. In phase C, **C01** frames amortized optimization, **C04** is the shared-multiplier Lagrangian method criticized in Section 4, **C05** is primal-dual learning, **C06** (DC3) is the iterative-repair alternative, and **C09–C10** are implicit layers. **F16** reads the E2ELR paper itself, with an eye on honest speed-up accounting. Next focus lesson: **A09**, the history and taxonomy of OPF formulations.

**Check yourself**

1. Why does a dual proxy give a *valid* lower bound even when its network is badly trained, while a primal proxy without a repair layer gives no feasibility guarantee at all?
2. In the risk-assessment application, why do proxies that only approximately satisfy power balance produce a *biased* risk estimate rather than just a noisier one?
3. The proportional repair layer guarantees feasibility for economic dispatch. What property of the feasible set makes such a cheap layer possible, and why is the same trick much harder for AC-OPF?

<details><summary>Answers</summary>

1. The completion layer turns any prediction `ẑ` into a point that satisfies every dual constraint exactly (the free bound multipliers absorb the residual `c − ẑA`). By weak duality, the objective of any dual-feasible point is at most the primal optimum. Training affects only how tight the bound is. A raw primal network solves a regression problem whose outputs are only approximately right, and nothing maps them onto the constraint surface, so equalities such as power balance hold only by luck.
2. Every slightly infeasible output is counted as an adverse event (an imbalance or overload) even when the true optimal dispatch has none. These errors all point the same way, toward more violations, so averaging over Monte-Carlo scenarios does not cancel them. The paper's figures show such false positives throughout the day, while the feasible proxy matches the solver's ground truth.
3. The ED feasible set is a box intersected with one hyperplane (plus a reserve constraint). Moving toward a box corner is a convex combination that stays in the box and changes total output monotonically, so a single scalar `ζ` restores balance in closed form. AC power flow adds nonconvex equality constraints (the power-flow equations from **A04**). The feasible set is a curved manifold without such a simple parametrization, so repair needs an iterative solve (Newton–Raphson, **A05**) or an implicit layer. That is why DC3-style and power-flow-based repairs exist (**C06**, **E16**).

</details>
