## 2026-11-04 (Wednesday) — A14 · Track 02: Optimal power flow: formulations, relaxations, solvers — Interior-point methods: barriers, the central path and primal–dual Newton steps

**Type:** Focus lesson
**Main reference:** [Numerical Optimization, 2nd ed. — J. Nocedal & S. J. Wright (2006), Springer](https://doi.org/10.1007/978-0-387-40065-5) — ch. 19 "Interior-Point Methods for Nonlinear Programming". Read §19.1–19.3 (the two derivations, the basic primal–dual algorithm, solving the Newton system) and §19.4 (barrier parameter updates and the fraction-to-the-boundary rule). Skim the rest (merit functions, line search and trust region variants, nonconvexity handling). If you want the LP warm-up first, ch. 14 covers the linear case.
**Further reading:**
- [Interior Methods for Nonlinear Optimization — A. Forsgren, P. E. Gill & M. H. Wright (2002), SIAM Review 44(4)](https://people.kth.se/~andersf/doc/sirev41494.pdf) — free author PDF. A survey with history, covering why barrier methods were dropped in the 1970s and came back after Karmarkar, and how primal–dual methods handle nonconvex problems. Read §1–§5.
- [Convex Optimization — S. Boyd & L. Vandenberghe (2004), ch. 11](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) — the convex version of the same story: barrier method, central path, the `m/t` duality-gap bound and primal–dual IPMs, with clean proofs.

**Why this matters** — Almost every production AC-OPF solve, whether through Ipopt, KNITRO, MATPOWER's MIPS or PowerModels, is an interior-point method. Its behaviour explains a lot of OPF practice: iteration counts that barely grow with grid size, solve time dominated by one sparse linear factorisation per iteration, and warm starts that rarely help. Anyone who wants to replace, accelerate or warm-start such a solver with a learned model needs to know what the solver actually does.

**Core ideas**

**1. The barrier idea.** Start from the standard form of A13, `min f_0(x)` s.t. `f_i(x) ≤ 0`, `h(x) = 0`. Inequalities make Newton's method awkward because you do not know in advance which ones bind. That is a combinatorial question, and it is what active-set methods (and X003) try to answer. Interior-point methods avoid the question. They replace each inequality by a logarithmic barrier that blows up at the boundary:

```
min_x  f_0(x) − μ Σ_i log(−f_i(x))    s.t.  h(x) = 0,      μ > 0
```

For fixed `μ` this problem has only equality constraints, so Newton's method applies directly. As `μ → 0` the barrier flattens everywhere except near the boundary, and the minimiser `x(μ)` tends to a solution of the original problem. ML analogy: this is like annealing a temperature. A log-barrier is a smooth surrogate for a hard indicator, as softmax is for argmax, and `μ` is turned down gradually rather than set to zero at once.

**2. The central path and perturbed KKT.** Write down the stationarity condition of the barrier problem and define `λ_i = μ / (−f_i(x))`. You get exactly A13's KKT system, with one change:

```
∇f_0(x) + Σ_i λ_i ∇f_i(x) + ∇h(x)ᵀ ν = 0
h(x) = 0,   f(x) + s = 0                      (slacks s_i = −f_i(x) > 0)
s_i λ_i = μ    for all i                      (perturbed complementarity)
s > 0,  λ > 0
```

Complementary slackness `s_i λ_i = 0` has become `s_i λ_i = μ`. The solutions `(x(μ), s(μ), λ(μ), ν(μ))` for `μ > 0` trace the **central path**. For a convex problem every point on it is dual feasible, and its duality gap is exactly

```
f_0(x(μ)) − g(λ(μ), ν(μ)) = m · μ        (m = number of inequalities)
```

(B&V §11.2). So `μ` is a direct measure of suboptimality. This is why IPMs report a certified gap at every outer iteration of a convex problem, and why "tolerance 1e-8" is effectively a target value of `μ`.

**3. Primal–dual Newton steps.** Modern methods do not minimise the barrier function itself. They apply Newton's method to the perturbed KKT system in all variables `(x, s, λ, ν)` together. That is the "primal–dual" in the name. Write `W = ∇²_xx L` for the Hessian of the Lagrangian, `J_f`, `J_h` for the constraint Jacobians, `S = diag(s)` and `Λ = diag(λ)`. Eliminate `Δs` and `Δλ` and what remains is a symmetric indefinite **KKT system**:

```
[ W + J_fᵀ Σ J_f    J_hᵀ ] [ Δx ]     [ r_x ]
[ J_h                0   ] [ Δν ] = − [ r_h ],        Σ = S⁻¹ Λ
```

Two practical facts follow.
- **Cost and sparsity.** In OPF the matrix has the sparsity pattern of the network. Every bus couples only to its neighbours in the admittance matrix (A03), just as the power-flow Jacobian did (A05). One sparse `LDLᵀ` factorisation per iteration (MA27/MA57, MUMPS, Pardiso) is most of the run time.
- **Ill-conditioning by design.** As `μ → 0`, `Σ_ii = λ_i / s_i` goes to `∞` for active constraints and to `0` for inactive ones. The matrix becomes badly conditioned, but in a structured way, and stable factorisations cope with it (N&W §19.3). Seen this way, the IPM discovers the active set gradually through the entries of `Σ` instead of guessing it.

**4. Staying inside.** A full Newton step can drive some `s_i` or `λ_i` negative. The **fraction-to-the-boundary rule** caps the step length at

```
α_max = max{ α ∈ (0, 1] : s + αΔs ≥ (1 − τ) s,  λ + αΔλ ≥ (1 − τ) λ },     τ ≈ 0.99–0.995
```

so each variable can lose at most 99–99.5% of its distance to zero per step. A merit function or a filter then decides whether the trial point is accepted (A15 covers Ipopt's filter).

**5. Driving μ to zero.** In the **monotone** (Fiacco–McCormick) strategy you solve the barrier problem approximately, then shrink `μ`, for example `μ ← 0.2 μ` or superlinearly. **Adaptive** strategies choose `μ` at each iteration from the current average complementarity `sᵀλ / m`. Mehrotra's predictor–corrector, the workhorse for LP, is the best-known example. For LP, short-step path-following methods need `O(√n log(1/ε))` iterations. In practice IPMs take about 20–80 iterations on problems from ten variables to millions, which is the property that made them dominant.

**6. Nonconvexity.** AC-OPF is nonconvex (A13 §3), so `W` may be indefinite and the Newton direction may point uphill. The fix is to check the **inertia** (the numbers of positive, negative and zero eigenvalues) that the `LDLᵀ` factorisation reports. If it is not (n positive, p negative, 0 zero), the solver adds `δI` to the `W` block and refactorises (N&W §19.3; Forsgren et al. §5). The result converges to a KKT point that is usually a local minimum. It is not guaranteed to be the global one (A16).

**A small example**

Take A13's toy problem: `min ½P_1² + P_2²` s.t. `P_1 + P_2 = 3`, `P_1 ≤ 1.5`. The KKT solution is `P_1 = 1.5`, `λ = 1.5`, `ν = 3`. Put a barrier on the cap with slack `s = 1.5 − P_1` and substitute `P_2 = 3 − P_1`:

```
φ_μ(P_1) = ½P_1² + (3 − P_1)² − μ log(1.5 − P_1)
φ'_μ = 0   ⇔   3s² + 1.5s − μ = 0   ⇒   s(μ) = (−1.5 + √(2.25 + 12μ)) / 6
λ(μ) = μ / s,    ν(μ) = 2P_2 = 3 + 2s
```

| μ | s(μ) | P_1(μ) | λ(μ) | ν(μ) | gap = μ |
|---|---|---|---|---|---|
| 1 | 0.379 | 1.121 | 2.637 | 3.758 | 1 |
| 0.1 | 0.0596 | 1.440 | 1.679 | 3.119 | 0.1 |
| 0.01 | 0.00658 | 1.493 | 1.520 | 3.013 | 0.01 |
| → 0 | → 0 | → 1.5 | → 1.5 | → 3 | → 0 |

Three things to notice. Every iterate is strictly feasible (`P_1 < 1.5`). The multiplier comes from the cap's price from above, so for large `μ` the barrier "overprices" a constraint that is not yet binding. And `s` goes to zero roughly like `μ / λ*`: shrinking `μ` tenfold moves the point about ten times closer to the boundary. You can check stationarity in `P_1` at `μ = 1`: `P_1 − ν + λ = 1.121 − 3.758 + 2.637 ≈ 0`.

Now consider warm starting. Suppose a learned model hands you the exact answer `P_1 = 1.5`. The interior method cannot even begin there, because `s = 0` and the barrier is infinite. If you nudge it to `s = 10⁻⁶` but leave `μ` at a default of `0.1`, the iterate sits far off the central path (`s λ ≈ 1.5·10⁻⁶ ≠ 0.1`). The first Newton steps then pull it back toward `s(0.1) ≈ 0.06`, away from the answer. This is the core of why warm-starting IPMs is hard (A18).

**How it connects**

A13 gave the KKT conditions. This lesson turns them into an algorithm by relaxing complementarity to `sλ = μ` and following the central path with Newton steps. The linear solve inside each step is a close relative of the power-flow Newton–Raphson of A05, on a matrix with the same network sparsity (A03). The duals the IPM returns are the LMPs and congestion prices of A11 and A08. A15 opens up Ipopt: its filter line search, the restoration phase it enters when it cannot make progress on feasibility, scaling, and the options that matter. A16 asks how good the local KKT points are on nonconvex AC-OPF. A18 explains the warm-start problem from the example. In phase C, OptNet (C09) differentiates through the solution of a primal–dual IPM, and the Lagrangian-dual training of C04–C05 can be seen as a learned, inexact version of the same primal–dual updates.

**Check yourself**

1. Why do interior-point methods not need to guess the active set, and where in the algorithm does the active set eventually show up?
2. In the example, what is the certified duality gap at `μ = 0.01`, and how does it compare with the actual cost gap `f_0(x(μ)) − p*`?
3. A colleague feeds an exact OPF solution from yesterday into an IPM as the starting point and sees no speed-up, sometimes even a slowdown. Explain why, in terms of the central path.

<details><summary>Answers</summary>

1. The log-barrier replaces each inequality by a smooth term, so every subproblem has only equality constraints and Newton's method applies with no combinatorial choice. The active set shows up in `Σ = S⁻¹Λ`: as `μ → 0`, the entries for binding constraints grow without bound (`s_i → 0`, `λ_i > 0`) and those for slack constraints go to zero (`λ_i → 0`). It also shows up in the final multipliers through complementary slackness.
2. With one inequality, `m = 1`, so the certified gap is `μ = 0.01`. The actual cost gap is smaller. `f_0` at `P_1 = 1.4934, P_2 = 1.5066` is `1.11515 + 2.26978 ≈ 3.38494`, against `p* = 3.375`, so the actual gap is about `0.0099`. Here the two nearly coincide, but in general the certified bound only guarantees `f_0(x(μ)) − p* ≤ m μ`.
3. An exact solution lies on the boundary: some slacks or multipliers are zero, and there the barrier is undefined. After it is pushed inside it is still badly centred, because the products `s_i λ_i` are tiny and very unequal while the solver's `μ` is usually reset to a default. The fraction-to-the-boundary rule then allows only very short steps, or the method first moves back toward the central path, wasting the head start. Warm starts help only if the starting point is near the central path for a suitably small `μ` (A18).

</details>
