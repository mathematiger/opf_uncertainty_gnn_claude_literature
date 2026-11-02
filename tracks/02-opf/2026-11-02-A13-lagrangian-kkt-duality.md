## 2026-11-02 (Monday) — A13 · Track 02: Optimal power flow: formulations, relaxations, solvers — Lagrangian, KKT and duality, seen through OPF

**Type:** Focus lesson
**Main reference:** [Convex Optimization — S. Boyd & L. Vandenberghe (2004), Cambridge University Press](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) — free PDF from the authors. Read ch. 5 "Duality": §5.1–5.2 (Lagrange dual function, dual problem, weak and strong duality, Slater's condition), §5.5 (optimality conditions, complementary slackness, KKT) and §5.6 (perturbation and sensitivity). Skim §5.3–5.4 (geometric and saddle-point views).
**Further reading:**
- [Numerical Optimization, 2nd ed. — J. Nocedal & S. J. Wright (2006), Springer](https://doi.org/10.1007/978-0-387-40065-5) — ch. 12 "Theory of Constrained Optimization": the nonconvex picture, with tangent cones, LICQ/MFCQ, the KKT theorem, second-order conditions and multipliers as sensitivities. This is the version that applies to AC-OPF.

**Why this matters** — Every OPF solver you will meet (interior-point, SQP, decomposition) is a method for finding a point that satisfies the KKT conditions. Every "price" in a power system (A08's λ, A11's LMPs) is a Lagrange multiplier. Anyone training learned OPF solvers needs this vocabulary: primal–dual training losses, duality-gap certificates, warm starts and feasibility checks are all stated in it.

**Core ideas**

**1. The standard form and the Lagrangian.** Write any OPF (A10, A11) as

```
min_x  f_0(x)   s.t.  f_i(x) ≤ 0  (i = 1..m),   h_j(x) = 0  (j = 1..p)
```

Here `x` stacks generator outputs, voltage magnitudes and angles. The `h_j` are the power-balance equations (A04, one or two per bus), and the `f_i` are the limits (A07: generator boxes, voltage bands, line ratings). The **Lagrangian** attaches a price to each constraint:

```
L(x, λ, ν) = f_0(x) + Σ_i λ_i f_i(x) + Σ_j ν_j h_j(x),     λ ≥ 0, ν free
```

ML analogy: this is a penalised loss whose penalty weights are themselves variables. Fixing the weights by hand gives the soft-penalty training used in many OPF proxies. Optimising over them gives duality.

**2. The dual function and weak duality.** Define `g(λ, ν) = inf_x L(x, λ, ν)`. It is a pointwise infimum of functions that are affine in `(λ, ν)`, so it is **always concave**, even when the OPF is nonconvex. For any feasible `x` and any `λ ≥ 0`, `L(x, λ, ν) ≤ f_0(x)`, because the added terms are ≤ 0. Hence

```
d* = max_{λ ≥ 0, ν} g(λ, ν)  ≤  p*       (weak duality)
```

Any dual-feasible point is a certified **lower bound** on the optimal cost. That is what makes duality useful in practice: a learned or heuristic primal solution with cost `f_0(x̂)` and any dual point `(λ̂, ν̂)` give a certified optimality gap `f_0(x̂) − g(λ̂, ν̂)` without solving the problem to optimality.

**3. Strong duality and Slater.** If the problem is convex (`f_0`, `f_i` convex, `h_j` affine) and there is a strictly feasible point (`f_i(x) < 0` for all non-affine `i`: **Slater's condition**), then `d* = p*`. DC-OPF (A11) is convex with affine constraints, so strong duality holds whenever it is feasible. AC-OPF has nonconvex equality constraints (products `|V_i||V_j| cos θ_ij`), so a positive duality gap is possible. When the gap is zero for a convex relaxation of AC-OPF, the relaxation is exact; that is the subject of A19–A22.

**4. KKT conditions.** For differentiable problems, a point `x*` with multipliers `(λ*, ν*)` is a **KKT point** if

```
∇f_0(x*) + Σ_i λ_i* ∇f_i(x*) + Σ_j ν_j* ∇h_j(x*) = 0     (stationarity)
f_i(x*) ≤ 0,  h_j(x*) = 0                                (primal feasibility)
λ* ≥ 0                                                   (dual feasibility)
λ_i* f_i(x*) = 0  for all i                              (complementary slackness)
```

Complementary slackness says a limit can carry a nonzero price only if it binds. You have seen this structure before in the SVM dual: support vectors are the training points whose margin constraints are active, and all other multipliers are zero. In OPF the "support vectors" are the congested lines and the generators at their limits, which is why active-set learning (X003) works: knowing which constraints bind reduces KKT to a square system of equations.

Two different theorems use these conditions:
- **Convex case (B&V §5.5.3):** with strong duality, KKT is *necessary and sufficient*. Any KKT point is globally optimal.
- **Nonconvex case (N&W ch. 12):** KKT is *necessary* at a local minimum only if a **constraint qualification** holds. The most common one is **LICQ**: the gradients of the active constraints are linearly independent. Then the multipliers are also unique. The weaker MFCQ still guarantees that multipliers exist, but they may not be unique. KKT is not sufficient here: you also need second-order conditions (the Hessian of `L` is positive definite on the critical cone), and even then the result is only a local minimum. This is the regime of AC-OPF and of Ipopt (A14–A15). "Solved" means "converged to a KKT point".

**5. Multipliers are sensitivities.** Perturb the constraints to `f_i(x) ≤ u_i`, `h_j(x) = w_j`, and let `p*(u, w)` be the optimal value. Under strong duality and differentiability (B&V §5.6.3),

```
λ_i* = −∂p*/∂u_i,      ν_j* = −∂p*/∂w_j
```

The balance constraint at bus `k` is `(demand_k − injection_k) = 0`, so raising demand there is a perturbation of `h_k`, and `|ν_k|` is the marginal cost of serving one more MW at bus `k`: the **LMP** of A11. A line multiplier `λ_ℓ` is the cost reduction per MW of extra rating, i.e. the value of a transmission upgrade. In the nonconvex case the same formula holds locally (N&W ch. 12) at a nondegenerate local minimum.

**6. Why decomposition and learning care.** The dual is concave and separable whenever the Lagrangian is. Dualising A12's coupling constraints `|u_k − u_0| ≤ Δu_k` splits a security-constrained OPF into independent per-contingency problems coordinated by prices: Lagrangian relaxation, and the reasoning behind Benders cuts. In ML, alternating gradient descent in `x` and ascent in `(λ, ν)` on `L` is the min–max training scheme behind Lagrangian-dual OPF proxies (C04, C05).

**A small example**

Two generators serve demand `D = 3` with costs `C_1 = ½P_1²` and `C_2 = P_2²` (marginal costs `P_1` and `2P_2`), and generator 1 has a cap `P_1 ≤ 1.5`.

Lagrangian: `L = ½P_1² + P_2² + ν(3 − P_1 − P_2) + λ(P_1 − 1.5)`.

- Without the cap, A08's equal-incremental-cost rule gives `P_1 = 2P_2`, so `P_1 = 2`, `P_2 = 1`, with price `ν = 2`. This violates the cap.
- With the cap active, stationarity in `P_2` gives `2P_2 = ν`, and in `P_1` gives `P_1 − ν + λ = 0`. With `P_1 = 1.5` we get `P_2 = 1.5`, `ν = 3`, `λ = ν − P_1 = 1.5 ≥ 0`. The cap binds and has a positive price, so complementary slackness holds and this is the KKT point. The cost is `p* = 1.125 + 2.25 = 3.375`.
- **Dual check:** `inf_{P_1}(½P_1² − (ν−λ)P_1) = −½(ν−λ)²` and `inf_{P_2}(P_2² − νP_2) = −ν²/4`, so `g(ν, λ) = −½(ν−λ)² − ν²/4 + 3ν − 1.5λ`. At `(3, 1.5)` this is `−1.125 − 2.25 + 9 − 2.25 = 3.375 = p*`: zero duality gap, as Slater predicts for this convex problem.
- **Sensitivity:** raise demand to `3 + ε`. Only generator 2 can respond, so the cost rises by `≈ 2·1.5·ε = 3ε`, matching `ν = 3`. Raise the cap to `1.5 + ε`. Then `P_1 = 1.5 + ε`, `P_2 = 1.5 − ε`, and the cost changes by `1.5ε − 3ε = −1.5ε`, matching `λ = 1.5`.
- **When KKT fails:** `min x s.t. x² ≤ 0`. The only feasible point is `x* = 0`, but stationarity `1 + 2λx* = 1 = 0` has no solution. The active constraint's gradient vanishes, so LICQ fails, and no strictly feasible point exists, so Slater fails. Degenerate limits like this, e.g. a voltage band with `V_min = V_max`, are a known cause of numerical trouble in solvers.

**How it connects**

This lesson makes A08's "λ is the system price" and A11's "LMPs are duals of the balance constraints" precise: both are statement 5 above. It also gives the language for A12's decomposition ideas and for X003's active-set learning, which predicts the support of `λ*`. Next, A14 turns the KKT system into an algorithm. Interior-point methods replace complementary slackness `λ_i f_i = 0` by `λ_i f_i = −μ` and follow the central path as `μ → 0` with Newton steps on the perturbed KKT equations. A15 shows how Ipopt does this robustly. A16 explains why AC-OPF's KKT points are usually, but not always, global, and A19–A22 return to the duality gap through convex relaxations. C04–C06 and C09–C10 reuse the KKT system for training and for differentiating through optimisers.

**Check yourself**

1. Why is the dual function `g` concave even for AC-OPF, and what does that buy you?
2. In the example, suppose the cap were `P_1 ≤ 2.5`. What are `λ` and `ν`, and why?
3. A neural network outputs a dispatch with cost 101.0, and a separate network outputs dual variables whose dual function value is 99.5. What can you conclude, and what further check is still needed?

<details><summary>Answers</summary>

1. `g(λ, ν) = inf_x L(x, λ, ν)` is an infimum over `x` of functions that are affine in `(λ, ν)`, and such an infimum is always concave. Maximising the dual is therefore a convex problem, and it yields a valid lower bound on `p*` (weak duality) whether or not the primal is convex. This is the basis of bounding and certifying solutions.
2. The unconstrained optimum `P_1 = 2, P_2 = 1` already satisfies `P_1 ≤ 2.5`. The cap is slack, so complementary slackness forces `λ = 0`, and `ν = 2` is the common marginal cost.
3. If the dual point is dual-feasible (`λ ≥ 0`, and `g` evaluated exactly), weak duality gives `p* ≥ 99.5`. So, if the dispatch is primal feasible, it is within 1.5 (about 1.5%) of optimal. You must still verify primal feasibility, i.e. that the dispatch satisfies power balance and all limits, e.g. with a power flow (A05). Otherwise its cost is not an upper bound on anything. For AC-OPF, also remember that the gap may never close to zero, because of a possible duality gap.

</details>
