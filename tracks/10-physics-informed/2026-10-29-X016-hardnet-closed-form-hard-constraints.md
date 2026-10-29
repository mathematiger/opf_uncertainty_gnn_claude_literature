## 2026-10-29 (Thursday) — X016 · Track 10: Physics-informed and constrained learning — HardNet: one closed-form layer that makes affine constraints impossible to violate

**Type:** Cross-phase preview
**Main reference:** [HardNet: Hard-Constrained Neural Networks with Universal Approximation Guarantees — Youngjae Min & Navid Azizan (2024; revised v4 2025), arXiv preprint 2410.10807](https://arxiv.org/abs/2410.10807). Read Section 4: the problem setup (eq. 4), Assumption 6, the closed-form layer (eq. 5) and Propositions 7–8. Then read the statements of Theorem 9 and Corollary 10, and Section 5.2 with Table 3 (learning an optimization solver). Skim Section 5.1 and the safe-control example in 5.3. The arXiv page lists no conference or journal venue, so treat it as a preprint.
**Further reading:**
- [DC3: A Learning Method for Optimization with Hard Constraints — Donti, Rolnick & Kolter (2021), ICLR](https://arxiv.org/abs/2104.12225) — the main baseline HardNet compares against. It uses equality completion plus gradient-based inequality correction and is covered properly in lesson C06.
- [OptNet: Differentiable Optimization as a Layer in Neural Networks — Amos & Kolter (2017), ICML](https://arxiv.org/abs/1703.00443) — the machinery behind HardNet's general convex variant: backpropagating through the solution of a QP (lesson C09).

**Why this matters** — A learned dispatch or OPF proxy that violates a power-balance or line-limit constraint cannot be used as is. A penalty term in the loss makes violations rarer on the training distribution, but it guarantees nothing on a new forecast. HardNet is a short, clean answer to a narrower question: when the constraints are affine in the output (their coefficients may depend on the input), can a network satisfy them exactly on every input without losing expressive power? Knowing exactly what this answer covers, and where it stops, is useful preparation for the constraint-handling lessons in phases C and E.

**Core ideas**

**1. The constraint class.** The network maps an input `x` (say a load forecast) to an output `y = f_θ(x)` (say generator setpoints). The constraints are two-sided and affine in `y`, with input-dependent data (eq. 4):

```
b_l(x) ≤ A(x) y ≤ b_u(x),     A(x) ∈ ℝ^{n_c × n_out}
```

Setting `b_l = b_u` gives an equality, and `±∞` gives a one-sided inequality. The "input-dependent" part matters for OPF-like problems: total demand, and therefore the balance constraint, changes from one input to the next. A fixed output layer such as a sigmoid box therefore cannot encode it. Assumption 6 requires (i) that a feasible `y` exists for every `x` and (ii) that `A(x)` has full row rank, which forces `n_c ≤ n_out`. Note that `n_c` counts rows, and a two-sided bound costs only one row.

**2. The layer.** Let `A⁺ = Aᵀ(AAᵀ)⁻¹` be the right pseudo-inverse. HardNet-Aff appends (eq. 5)

```
P(f)(x) = f + A⁺ [ ReLU(b_l − A f) − ReLU(A f − b_u) ]
```

with all quantities evaluated at `x`. Read the bracket row by row. For each constraint row `i`, it holds zero if `a_iᵀ f` already lies in `[b_l,i, b_u,i]`, and otherwise the signed distance to the violated bound. Multiplying by `A⁺` turns that vector of required row changes into the minimum-norm change of `y` that produces it. One can check that `A P(f) = A f + bracket`. Every violated row therefore lands exactly on its bound and every satisfied row keeps its value (Proposition 8). Proposition 7 states this as an ℓ2 projection: among all `y` with those row values, it picks the one closest to `f`. The layer involves no solver and no iterations. It costs one small linear solve per input and is differentiable almost everywhere, since ReLU is piecewise linear. The whole model `P ∘ f_θ` is trained with an ordinary unconstrained optimizer on whatever loss you like, for example the dispatch cost.

**3. Caveat: this is not the Euclidean projection onto the feasible polytope.** Rows that are satisfied are frozen at their current values, even when moving them would bring the output closer. The result is feasible, but it can be farther from `f` than the true projection. The example below shows a case.

**4. Expressivity.** A natural worry is that a hard layer shrinks the function class. Theorem 9 says it does not: on a compact input set, with continuous `A(x)` and Assumption 6, HardNet-Aff networks are dense in the constraint-satisfying target functions whenever the plain networks are dense. Corollary 10 gives a width condition for ReLU networks, `w ≥ max{n_in + 1, n_out}`, for `L^p` targets. Intuition (my paraphrase of the mechanism): `P` leaves feasible points unchanged and moves an output by no more than its violation, scaled by `‖A⁺‖`. So if `f_θ` is close to a feasible target, `P(f_θ)` is at least as close up to a constant. The general variant, HardNet-Cvx (eq. 8), replaces the closed form with an ℓ2 projection onto an arbitrary convex input-dependent set, solved by a differentiable convex-optimization layer. It has a matching approximation theorem (Theorem 11), but the paper reports no experiments with it.

**5. What they found.** All models are 3-layer ReLU MLPs with 200 hidden units. On a learned-solver benchmark (Section 5.2) with 100 variables, 50 equalities and 50 inequalities (exactly `n_c = n_out`), HardNet-Aff reaches an objective of −14.10 with zero violations. The IPOPT solver reaches −14.28 at about 1182 ms per instance, against 6.69 ms for HardNet-Aff. DC3 is also feasible but worse (−12.57, 9.61 ms). Plain and penalty-trained networks violate the constraints. In safe control (Section 5.3), HardNet-Aff has zero violations at 2.71 ms, against 579 ms for a control-barrier-function QP. DC3 and soft penalties reach lower cost there, but with maximum violations near 6.9. These are synthetic, small benchmarks, so read them as a proof of concept and not as power-system evidence.

**A small example**

Two generators serve demand `D = 100` MW. Generator 1 is capped at 60 MW. Output `y = (P_1, P_2)`, with constraints written as rows:

```
row 1:  P_1 + P_2 ∈ [100, 100]     (balance, input-dependent: D comes from x)
row 2:  P_1       ∈ [0, 60]
A = [[1, 1], [1, 0]],   A⁺ = A⁻¹ = [[0, 1], [1, −1]]
```

**Case 1:** raw output `f = (80, 30)`. Then `A f = (110, 80)`. Row 1 is 10 too high and row 2 is 20 too high, so the bracket is `(−10, −20)`. `A⁻¹(−10, −20) = (−20, 10)`, which gives `P(f) = (60, 40)`. The result is balanced and at the cap. This matches the Euclidean projection here.

**Case 2:** `f = (50, 30)`. Then `A f = (80, 50)`: row 1 is 20 short and row 2 is fine, so the bracket is `(20, 0)` and `P(f) = (50, 50)`. Because row 2's value `P_1 = 50` is frozen, the entire shortfall goes to generator 2. The Euclidean projection onto the feasible set would be `(60, 40)`, which splits the shortfall evenly. Both points are feasible, and HardNet's moves the output farther (distance 20 against 14.1). Training compensates: the loss is computed after `P`, so `f_θ` learns where to aim.

**Where it stops.** Add a bound `P_2 ∈ [0, 80]`. That is a third row for two outputs, so `A` can no longer have full row rank and HardNet-Aff no longer applies. A realistic DC-OPF (A11) has one balance row, one row per line flow limit (through PTDFs, A06) and one per generator bound. That is far more rows than generators, so the problem falls to HardNet-Cvx or to other methods.

**How it connects**

This continues **X007**: Lagaris's trial solution `A(x) + F(x, N)` satisfies boundary conditions by construction, and HardNet does the same for a whole family of input-dependent affine constraints. It also complements **X012**, where repair layers such as E2ELR were built for one specific feasible set (economic dispatch with reserves). HardNet trades that problem-specific structure for a generic closed form with a row-count limit. The pseudo-inverse correction is the same minimum-norm idea behind equal incremental cost in **A08**. The limit `n_c ≤ n_out` explains why the line-limited DC-OPF of **A11** needs more machinery. In the curriculum, C06 (DC3) and C09 (differentiable optimization layers) cover the general case, and E04, E05 and E16 compare soft constraints, constraints by construction and projection layers for AC-OPF, where the constraints are no longer affine. Next in the focus phase, A12 adds security constraints (N-1), which multiply the number of constraint rows.

**Check yourself**

1. Why does the layer use `A⁺ = Aᵀ(AAᵀ)⁻¹`, and which part of Assumption 6 makes it well defined?
2. In Case 2, compute the Jacobian of `P` with respect to `f` and say which direction of the upstream gradient it removes.
3. A grid has 5 generators and 8 lines with two-sided flow limits, plus a balance equation and generator bounds. Can HardNet-Aff enforce all constraints? What is the largest set you could enforce with it directly?

<details><summary>Answers</summary>

1. The bracket specifies how much each constraint row `a_iᵀy` must change. `A⁺` gives the smallest change `Δy` with `A Δy` equal to that vector, so the output moves as little as possible while hitting the target row values. It requires `AAᵀ` to be invertible, which is exactly the full-row-rank part of Assumption 6 (and hence `n_c ≤ n_out`).
2. In the violated region of row 1 with row 2 satisfied, `P(f) = f + A⁻¹(100 − (f_1 + f_2), 0)ᵀ = f − (0, f_1 + f_2 − 100)`. So `∂P/∂f = I − A⁻¹ e_1 a_1ᵀ = [[1, 0], [−1, 0]]`. The output depends on `f_1` alone, and the upstream gradient along `f_2` is zeroed: changing `f_2` cannot change a balanced output whose `P_1` is frozen. This can create "dead" gradient directions much like an active clip, and the network learns only through `f_1` in this region.
3. No. There is 1 balance row, 8 line rows and 5 generator rows, so 14 rows for 5 outputs. Full row rank allows at most 5 linearly independent rows: for example the balance equation plus 4 generator bounds, or the balance plus a few critical line limits. Everything else needs HardNet-Cvx (a QP projection), DC3-style correction or a problem-specific repair layer.

</details>
