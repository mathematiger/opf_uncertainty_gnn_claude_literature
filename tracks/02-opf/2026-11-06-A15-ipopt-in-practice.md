## 2026-11-06 (Friday) — A15 · Track 02: Optimal power flow: formulations, relaxations, solvers — Ipopt in practice: the filter line search, the restoration phase, scaling and the options that matter

**Type:** Focus lesson
**Main reference:** [On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming — A. Wächter & L. T. Biegler (2006), Mathematical Programming 106(1), 25–57](https://doi.org/10.1007/s10107-004-0559-y) ([free author copy](http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf)) — read §2 (the barrier subproblems, the termination test, the filter line search and the full Algorithm A), §3.1 (inertia correction), §3.3.1 (restoration phase), §3.6 (initialization) and §3.8 (automatic scaling). Skim §4 (numerical results); skip §3.2, §3.4–3.5, §3.7 and §3.9 on a first pass.
**Further reading:**
- [Ipopt options reference — COIN-OR](https://coin-or.github.io/Ipopt/OPTIONS.html) — the official list of every option with its default. Use it as a lookup table next to §6 below.
- [Nonlinear programming without a penalty function — R. Fletcher & S. Leyffer (2002), Mathematical Programming 91(2)](https://doi.org/10.1007/s101070100244) — the paper that introduced filter methods. Read §1–§2 for the idea, before Ipopt adapted it to barrier problems.

**Why this matters** — When a paper says its OPF labels were produced by "PowerModels + Ipopt", the labels depend on Ipopt's tolerances, its scaling, its starting point and its handling of hard instances. Anyone who generates training data with this solver, uses it as a baseline, or tries to warm-start or replace it needs to know what it does when things go well and when they go badly, and how to read its output.

**Core ideas**

**1. What Ipopt actually solves.** Ipopt takes the form `min f(x)` s.t. `c(x) = 0`, `x_L ≤ x ≤ x_U`. A general inequality `d_L ≤ d(x) ≤ d_U`, such as a line-flow limit, becomes `d(x) − s = 0` with a bounded slack `s`. Only the simple bounds get log-barriers, so the barrier subproblem is

```
min_x  φ_μ(x) = f(x) − μ Σ_i ln(x_i − x_L,i) − μ Σ_i ln(x_U,i − x_i)     s.t.  c(x) = 0
```

This is A14's scheme with `x ≥ 0` written as general bounds. Progress is measured by a scaled optimality error, the largest of three residuals:

```
E_μ = max{ ‖∇f + ∇c λ − z‖_∞ / s_d ,  ‖c(x)‖_∞ ,  ‖XZe − μe‖_∞ / s_c }
```

These are dual infeasibility, primal infeasibility and complementarity. The scalings `s_d`, `s_c ≥ 1` take effect only when the average multiplier exceeds `s_max = 100`. Ipopt stops when `E_0 ≤ tol`, with default `tol = 1e-8`. It moves to the next barrier problem once `E_μ ≤ κ_ε μ` (`κ_ε = 10`) and updates

```
μ_{j+1} = max{ tol/10 ,  min{ κ_μ μ_j ,  μ_j^{θ_μ} } },      κ_μ = 0.2,  θ_μ = 1.5,  μ_0 = 0.1
```

So `μ` first shrinks linearly by a factor of 5 per step, then superlinearly with exponent 1.5, and never drops below what the tolerance needs. The fraction-to-the-boundary parameter from A14 is `τ_j = max{0.99, 1 − μ_j}`.

**2. The filter: two losses, no penalty weight.** A Newton step improves either feasibility `θ(x) = ‖c(x)‖` or the barrier objective `φ_μ`, and the two goals often conflict. The classical fix combines them in a merit function `φ_μ + ν θ`, which means tuning the penalty weight `ν`. A filter, following Fletcher & Leyffer, treats the pair as a bi-objective problem instead. A backtracking trial point `x_k + α d_k` is acceptable if it makes sufficient progress in either goal:

```
θ(x_trial) ≤ (1 − γ_θ) θ(x_k)     or     φ_μ(x_trial) ≤ φ_μ(x_k) − γ_φ θ(x_k),       γ_θ = γ_φ = 1e-5
```

It must also not lie in the **filter**, a stored set of forbidden `(θ, φ)` regions, one corner per earlier iterate, that stops the method from cycling back. ML analogy: this resembles keeping a Pareto front of (constraint violation, loss) pairs and accepting any point that no stored point dominates, rather than fixing a Lagrangian weight in advance. There is one twist. When the iterate is nearly feasible (`θ ≤ θ_min`) and the step is a clear descent direction, the **switching condition** `α[−∇φ_μᵀd]^{2.3} > δ θ^{1.1}` holds. The filter rule then gives way to an ordinary Armijo decrease on `φ_μ`. This stops the method from settling at a feasible but non-optimal point. A **second-order correction** (§2.4) adds a cheap extra step when the first trial point raises `θ`. This counters the Maratos effect, in which curvature in `c` makes good Newton steps look worse than they are.

**3. Inertia correction.** For the nonconvex AC-OPF, A14 noted that the KKT matrix needs inertia `(n, m, 0)`. Ipopt reads the inertia from the `LDLᵀ` factorisation. If it is wrong, Ipopt adds `δ_w I` to the Hessian block, and, if the matrix is singular, `−δ_c I` to the constraint block. The trial values follow a fixed heuristic (§3.1): the first `δ_w` is `1e-4`, and further failures multiply it by 100 at first and by 8 later. Successful values are remembered and reduced by a factor of 3 next time, and `δ_c = 1e-8 μ^{1/4}`. Each attempt costs one more factorisation. The `lg(rg)` column of the log shows `log10 δ_w`, so a column full of numbers means you are far from a region where the problem looks locally convex.

**4. The restoration phase.** If backtracking drives `α` below a minimum, or the matrix cannot be factorised even with large `δ_w`, Ipopt stops trying to optimise and switches to minimising infeasibility. It solves a second NLP with the same algorithm:

```
min_{x, p, n}  ρ Σ_i (p_i + n_i) + (ζ/2) ‖D_R (x − x_R)‖²     s.t.  c(x) − p + n = 0,   p, n ≥ 0
```

This is an ℓ1 feasibility problem with a proximal term that keeps the new point near the iterate `x_R` where restoration started. Restoration iterations appear in the log with an `r` after the iteration number. If restoration converges to a point with nonzero violation, Ipopt reports local infeasibility. For OPF this usually means a load profile the network cannot serve under its limits, or contradictory data. It does not prove global infeasibility, since the problem is nonconvex (A16). For anyone building datasets: "failed" samples are not random noise. They cluster near the feasibility boundary, which is exactly where a learned model is most likely to be wrong, so dropping them silently biases the data.

**5. Initialization and scaling.** Ipopt moves the user's starting point into the interior, pushing each bounded variable at least `κ_1 max{1, |x_L|}` away from its bound, with `κ_1 = κ_2 = 1e-2` (`bound_push`, `bound_frac`). It sets bound multipliers to 1 and estimates `λ_0` by least squares. If `‖λ_0‖_∞ > 1e3`, it sets `λ_0 = 0` instead. Scaling is deliberately conservative. With `g_max = 100` it computes

```
d_f = min{1, g_max / ‖∇f(x_0)‖_∞},      d_c^(j) = min{1, g_max / ‖∇c_j(x_0)‖_∞}
```

so it only ever scales functions down, and only when a gradient is large at the start. On the 954 CUTEr problems used in the paper, Ipopt with default options terminated successfully on 895 problems, and on 872 with scaling off. KNITRO 3.1 solved 829 and LOQO 6.06 solved 847. The authors stress that this is not a rigorous comparison: the codes used different tolerances and 2004 versions.

**6. The options worth knowing** (defaults from the official reference):

| Option | Default | What it controls |
|---|---|---|
| `tol` / `acceptable_tol`, `acceptable_iter` | 1e-8 / 1e-6, 15 | stop early if "acceptable" holds for 15 iterations in a row |
| `constr_viol_tol`, `dual_inf_tol`, `compl_inf_tol` | 1e-4, 1, 1e-4 | extra *unscaled* checks that must also hold at termination |
| `max_iter` | 3000 | iteration cap (restoration iterations count) |
| `mu_strategy`, `mu_init` | monotone, 0.1 | rule (7) above; `adaptive` picks `μ` per iteration |
| `nlp_scaling_method` | gradient-based | §3.8 rule, `nlp_scaling_max_gradient = 100` |
| `bound_push`, `bound_frac` | 0.01, 0.01 | how far the start is pushed inside |
| `bound_relax_factor` | 1e-8 | bounds are relaxed by this relative amount |
| `warm_start_init_point` | no | use given multipliers; tune with `warm_start_bound_push` etc. |
| `linear_solver` | ma27 | the sparse `LDLᵀ`; builds without HSL use MUMPS |
| `hessian_approximation` | exact | `limited-memory` = L-BFGS when no Hessian is available |

Two practical points follow from the table. First, "optimal" from Ipopt means a scaled KKT error of 1e-8 with bounds relaxed by about 1e-8. A learned model that hits the labels to 1e-6 is therefore within label noise. Second, the linear solver often changes run time more than any algorithmic option, because factorisation dominates the cost (A14).

**A small example**

**Reading a filter decision.** Suppose the current iterate has `(θ_k, φ_k) = (0.5, 2.0)` and the filter holds one corner from an earlier iterate, `(0.2, 3.0)`. That corner forbids every point with `θ ≥ 0.2` and `φ ≥ 3.0` (ignoring the tiny `γ` margins). Three trial points:

| Trial | (θ, φ) | Progress vs current? | In forbidden region? | Verdict |
|---|---|---|---|---|
| A | (0.30, 2.5) | yes, θ falls | no, φ < 3.0 | accept |
| B | (0.25, 3.2) | yes, θ falls | yes, θ ≥ 0.2 and φ ≥ 3.0 | reject, shrink α |
| C | (0.60, 1.0) | yes, φ falls by more than γ_φ θ_k | no | accept |

B improves on the current point but is dominated by an earlier one, so it would let the method cycle. C makes feasibility worse but is still acceptable, which no monotone merit function with a large `ν` would allow.

**Scaling in per-unit.** An OPF in per-unit (A02) with a 100 MVA base and a cost of 30 €/MWh has an objective gradient of `30 · 100 = 3000` per p.u. of output. Ipopt's rule gives `d_f = min{1, 100/3000} ≈ 0.033`, so the objective is scaled down about 30-fold while the power-balance rows, whose gradients are of order `|V|²|Y|`, are left alone unless they also exceed 100. If you write the cost in €/h with MW variables instead, the gradient changes, and so do the scaling, the path and sometimes the local optimum found.

**How it connects**

A13 gave the KKT system and A14 the primal–dual barrier method. This lesson adds what makes it robust: the filter in place of a merit function, inertia correction for the nonconvexity of A10's AC-OPF, a restoration phase for infeasible starts, and scaling. The restoration problem is an ℓ1 "minimum load-shed"-style feasibility problem, the solver-side counterpart of the feasibility repair layers in X012 and X016. Next, A16 asks how good Ipopt's local KKT points are on nonconvex AC-OPF. A17 shows the modelling stacks (PowerModels.jl, MATPOWER, pandapower) that call Ipopt. A18 returns to `warm_start_init_point` and `bound_push` to explain why handing Ipopt a learned solution so often fails to speed it up.

**Check yourself**

1. Why does the filter need the switching condition? What could go wrong if it accepted any point that reduces `θ` or `φ_μ` and is not filtered?
2. An Ipopt log shows iterations `41r, 42r, …, 77r` and ends with "Converged to a point of local infeasibility". What did the solver do, and what can and cannot you conclude about the OPF instance?
3. You generate 50 000 OPF labels with default Ipopt and discard the 3% that did not report "Optimal Solution Found". Name two ways this choice can distort what a learned model, or its uncertainty estimate, learns.

<details><summary>Answers</summary>

1. Without it, the method could keep accepting steps that only reduce infeasibility and converge to a feasible point that is not optimal, because once `θ ≈ 0` the filter test is easy to pass through small changes in `θ`. When the iterate is nearly feasible and the step is a descent direction, the switching condition demands a real Armijo decrease in `φ_μ`, so the objective must improve. Steps of that kind do not add to the filter.
2. The line search could not make progress, so Ipopt entered the restoration phase (the `r` iterations) and minimised the ℓ1 constraint violation near the current point. That subproblem converged to a stationary point with nonzero violation. You can conclude that the instance is locally infeasible from that region. Because AC-OPF is nonconvex, this does not prove global infeasibility. A different start or a relaxation-based check (A19–A22) is needed to say more.
3. (i) Selection bias: failures concentrate near the feasibility boundary and at stressed loadings, so the training distribution under-represents exactly the hard, high-risk regime, and calibration measured on the cleaned test set looks better than it will be in deployment. (ii) Hidden label inconsistency: "acceptable" stops and relaxed bounds (`bound_relax_factor`) mean the kept labels also carry solver-dependent errors near 1e-6 to 1e-8, and runs that use a different start or linear solver may land on different local optima (A16). That adds label noise the model may interpret as aleatoric uncertainty. Keep the failures as labelled outcomes, not as missing data.

</details>
