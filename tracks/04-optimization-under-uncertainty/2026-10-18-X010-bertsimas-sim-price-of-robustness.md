## 2026-10-18 (Sunday) — X010 · Track 04: Stochastic, chance-constrained, risk-averse & robust optimization — Robust optimization with a dial: Bertsimas & Sim's "price of robustness"

**Type:** Cross-phase preview
**Main reference:** [The Price of Robustness — D. Bertsimas & M. Sim (2004), Operations Research 52(1):35–53](https://www.mit.edu/~dbertsim/papers/Robust%20Optimization/The%20price%20of%20Robustness.pdf) — read §1–3 (about 4 pages: the uncertainty model, Soyster's worst-case counterpart (2), the budgeted counterpart (4), Proposition 1 and Theorem 1, which give the linear reformulation (7)). Then read §4 up to Theorem 2 and Table 1, and §6.1 (robust knapsack). Skim the proof of Theorem 3, §4.2 and §5. The link goes to the authors' copy of the published paper. The DOI is [10.1287/opre.1030.0065](https://doi.org/10.1287/opre.1030.0065).
**Further reading:**
- [Theory and Applications of Robust Optimization — D. Bertsimas, D. B. Brown & C. Caramanis (2011), SIAM Review 53(3):464–501](https://arxiv.org/abs/1010.5445) — the standard survey: which uncertainty sets give tractable counterparts, how robust optimization relates to chance constraints, and adaptable (multistage) models. Use it as a map, not to read end to end.
- [Adaptive Robust Optimization for the Security Constrained Unit Commitment Problem — D. Bertsimas, E. Litvinov, X. A. Sun, J. Zhao & T. Zheng (2013), IEEE Transactions on Power Systems 28(1):52–63](https://dspace.mit.edu/handle/1721.1/68971) — budgeted uncertainty sets for nodal net injections in a two-stage unit-commitment model, tested on the ISO New England system. It shows the idea from this entry at grid scale.

**Why this matters** — A probability distribution of forecast errors is not always available, or not always trusted. Robust optimization instead asks for a *set* of possible data and requires the decision to stay feasible for every member of that set. The obvious set (every parameter at its worst at once) is absurdly pessimistic. This paper shows how to pick a smaller set with one interpretable knob, and still keep a linear program and a probabilistic guarantee. Anyone building learned or classical dispatch tools under forecast uncertainty will meet the budgeted uncertainty set again and again: in robust OPF, robust unit commitment, and as the target that learned "robust" solvers are measured against.

**Core ideas**

*Setting.* Start from a nominal LP, `max cᵀx  s.t. Ax ≤ b, l ≤ x ≤ u`. Uncertainty sits in the entries of `A`. A cost vector can be moved into the constraints with an epigraph variable, `max z s.t. z − cᵀx ≤ 0`. For row `i`, let `J_i` be the set of uncertain coefficients. Each uncertain coefficient is a random variable `ã_ij ∈ [a_ij − â_ij, a_ij + â_ij]` with an *unknown but symmetric* distribution. In normalized form, `η_ij = (ã_ij − a_ij)/â_ij ∈ [−1, 1]`. ML analogy: this is a bounded, symmetric perturbation model, the same shape as an ℓ∞ threat model in adversarial robustness. The difference is that here we optimize a *decision* against it, not a classifier.

*Two extremes.* **Soyster (1973)** protects against every coefficient moving to its worst value at once:

```
Σ_j a_ij x_j + Σ_{j∈J_i} â_ij |x_j| ≤ b_i        (Soyster: Γ_i = |J_i|)
```

This is always feasible, and very conservative. Ben-Tal & Nemirovski use ellipsoidal sets, which are much less conservative but turn the LP into a second-order cone program. That is fine for continuous problems, but awkward for MILPs.

*The budget.* Bertsimas & Sim add a per-row parameter `Γ_i ∈ [0, |J_i|]`. The intuition is that nature will not push *all* coefficients to their extremes at once, so the decision is protected against the worst case in which at most `⌊Γ_i⌋` coefficients move fully and one more moves by the fraction `Γ_i − ⌊Γ_i⌋`. The **protection function** is

```
β_i(x, Γ_i) = max_{S_i ∪ {t_i}: S_i ⊆ J_i, |S_i| = ⌊Γ_i⌋, t_i ∈ J_i\S_i}
              Σ_{j∈S_i} â_ij |x_j| + (Γ_i − ⌊Γ_i⌋) â_it_i |x_t_i|
```

The robust constraint is `Σ_j a_ij x_j + β_i(x, Γ_i) ≤ b_i`. `Γ_i = 0` gives back the nominal problem, and `Γ_i = |J_i|` gives back Soyster. The dial in between is what the authors call the "price of robustness".

*The trick that keeps it linear (Proposition 1, Theorem 1).* For a fixed `x`, the inner maximization over subsets is the LP `max Σ_j â_ij |x_j| z_ij  s.t. Σ_j z_ij ≤ Γ_i, 0 ≤ z_ij ≤ 1`. It is a fractional knapsack with an integral optimum. By LP duality its value equals `min Γ_i z_i + Σ_j p_ij  s.t. z_i + p_ij ≥ â_ij |x_j|, p, z ≥ 0`. Because this is a *minimization*, you can paste it into the outer problem, and the whole robust counterpart becomes one LP (eq. 7):

```
max cᵀx
s.t. Σ_j a_ij x_j + Γ_i z_i + Σ_{j∈J_i} p_ij ≤ b_i      ∀i
     z_i + p_ij ≥ â_ij y_j,   −y_j ≤ x_j ≤ y_j,   p, y, z ≥ 0
```

This adds `k + 1` variables and `k + n` constraints, where `k = Σ_i |J_i|`, and it keeps the sparsity of `A`. If `x` has integer components, the counterpart is still a MILP of the same type. This "dualize the inner adversary" move is the most reusable idea in robust optimization. You will see it again for polyhedral uncertainty sets, in adjustable robust optimization (B08), and in Wasserstein DRO (B12).

*The guarantee (Theorem 2).* Suppose the `η_ij` are independent and symmetric on `[−1, 1]`, with nothing else assumed about their distribution. Then the robust solution violates row `i` with probability at most

```
P(violation of row i) ≤ exp(−Γ_i² / (2|J_i|))
```

This is a Hoeffding-type bound. Theorem 3 tightens it with an exact binomial-tail bound, which the authors show is the best possible. They also give a normal approximation, `≈ 1 − Φ((Γ_i − 1)/√|J_i|)`. The consequence is that the protection you need grows like `√|J_i|`, not like `|J_i|`. Table 1 (violation probability below 1%) shows this: with `|J_i| = 200` the tight bound needs `Γ_i ≈ 33.9`, about 17% of the uncertain coefficients. The Hoeffding bound needs 42.9. With `|J_i| = 2000` the tight bound needs about 105. With `|J_i| = 5`, full (Soyster) protection is needed anyway. This is a concentration-of-measure argument, the same mechanism as in a generalization bound. Many small independent errors rarely all line up.

*Findings (§6.1).* Take a random 0–1 knapsack problem with 200 items, capacity 4,000, and weights uncertain by ±10%. The nominal optimum is 5,592. Full Soyster protection lowers it by 5.5%, to 5,283. A guarantee of at most 0.57% violation probability costs only 1.54% of the objective. The robust MILPs solved in 0.05–50 s with CPLEX 6.0. The portfolio experiment (§6.2) shows the dial from another side. At `Γ = 0` the portfolio holds only the stock with the highest expected return. It diversifies as `Γ` grows to 17, stays fully diversified up to 41, and above 41 it jumps abruptly to the single stock that the Soyster solution picks.

*Caveats to keep in mind.* The guarantee needs independence and symmetry. Correlated errors, such as wind farms behind one weather front, break it. §5 handles correlation only through a linear factor model. The bound is also distribution-free, so for well-behaved errors it is conservative. And `Γ` protects each row separately. It says nothing about the probability that *some* row fails, which is the joint-versus-individual chance-constraint question that phase B takes up.

**A small example**

One transmission line has thermal limit `F̄`. From A06, its DC flow is linear in the injections: `f = f₀ + Σ_k PTDF_k w̃_k`. Take 100 wind farms with `PTDF_k = 1`, and suppose each output deviates from its forecast by an independent, symmetric error in `[−1, +1]` MW. How much headroom below `F̄` should the dispatch keep?

```
Soyster (Γ = 100):                         margin 100 MW   (all farms worst at once)
Bertsimas–Sim, Hoeffding bound, ε = 1%:    Γ = √(2·100·ln 100) ≈ 30.3 → margin 30.3 MW
Bertsimas–Sim, tight bound (Table 1):      Γ ≈ 24.3                   → margin 24.3 MW
True 99% quantile if errors are ±1 coin flips: σ = 10   → ≈ 2.33·10  ≈ 23.3 MW
True 99% quantile if errors are uniform:       σ = √(100/3) ≈ 5.8 → ≈ 13.4 MW
```

The tight bound is almost exact in the worst symmetric case (coin flips), and conservative by a factor of about 1.8 when the errors are uniform. That is the price of making no assumption about the shape of the distribution. If you know the shape, a chance constraint (B05, B07) can recover the gap. If the errors are correlated, every number in the table is too optimistic.

The protection function for one row with `â_j|x_j| = (5, 3, 2)` and `Γ = 1.5` is `5 + 0.5·3 = 6.5`. The dual gives the same value: with `z = 3` you get `p = (2, 0, 0)` and `1.5·3 + 2 = 6.5`.

**How it connects**

- **X001** (Powell): robust optimization is a cost-function approximation policy. You solve a deterministic problem whose constraints have been tightened by a tunable parameter (`Γ`), and you tune that parameter by simulation.
- **X002** (Pinson): probabilistic wind forecasts tell you how wide the intervals `[a − â, a + â]` should be. Robust optimization only needs those widths, not the full density.
- **X008** (conformal decision theory): this entry calibrates the *uncertainty set* through a concentration bound. X008 calibrates the *decision* online. Both turn a risk level into a single conservativeness parameter.
- **X009** (B&B): the budgeted counterpart of a MILP is still a MILP, which is why this set is popular for unit commitment (A30).
- **A06**: PTDFs make every line flow linear in the injections, so this row-wise robust counterpart applies directly to DC line limits.
- **Coming up:** B01 places robust, stochastic, chance-constrained and distributionally robust optimization in one taxonomy. B05–B06 give the chance-constrained and scenario-based guarantees that this entry's bound competes with. B08 makes the recourse adjustable, and B12 replaces the box set with a Wasserstein ball around data.

**Check yourself**

1. Why does the inner maximization over "which `⌊Γ⌋` coefficients move" not make the robust counterpart combinatorial?
2. For a row with `|J_i| = 400` uncertain coefficients, the Hoeffding bound is to give at most a 1% violation probability. Roughly what `Γ_i` is needed, and what fraction of full Soyster protection is that?
3. Name two realistic grid situations in which the guarantee of Theorem 2 does not hold, and say which assumption fails in each.

<details><summary>Answers</summary>

1. For a fixed `x`, the subset choice is a fractional knapsack LP whose optimum is integral (pick the `⌊Γ⌋` largest `â_j|x_j|` plus a fraction of the next one). LP duality turns this inner *max* into a *min* over `(z_i, p_ij)` with the same value. A min-constraint of the form "`… + min(…) ≤ b`" can be enforced by adding the dual variables to the outer problem, which gives a single LP (Theorem 1). The combinatorics disappear because the relaxation of the inner problem is exact.
2. Solve `exp(−Γ²/(2·400)) = 0.01`: `Γ = √(800 · ln 100) ≈ √3684 ≈ 60.7`, which is about 15% of `|J_i| = 400`. The tight bound would need a bit less. The `√n` scaling is what makes the fraction shrink as rows contain more uncertain terms.
3. (a) Wind farms in the same region share weather, so their forecast errors are positively correlated and independence fails; the sum's tail is much heavier than the bound assumes. (b) Errors that are skewed or bounded on one side, for example PV output that cannot exceed clear-sky output or wind near cut-out, break symmetry. Other valid answers: a few large uncertain terms dominate the row (`|J_i|` small, so Soyster is needed anyway), or the requirement is a *joint* guarantee across many lines, while `Γ_i` gives only row-wise bounds.

</details>
