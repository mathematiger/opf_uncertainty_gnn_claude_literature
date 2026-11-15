## 2026-11-15 (Sunday) — X025 · Track 10: Physics-informed and constrained learning — Conservation laws to machine precision: soft penalties vs a fixed "residual" layer (Beucler et al.)

**Type:** Cross-phase preview
**Main reference:** [Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems — T. Beucler, M. Pritchard, S. Rasp, J. Ott, P. Baldi & P. Gentine (2021), Physical Review Letters 126, 098302](https://arxiv.org/abs/1909.00912). Read all of the main text (5 pages): §II.A–C (writing constraints as a linear system, the two ways to enforce it, and the error decomposition in Eq. 9) and §III.E (results, Fig. 3). Skip the climate-specific variable definitions.
**Further reading:**
- [Imposing Hard Constraints on Deep Networks: Promises and Limitations — P. Márquez-Neila, M. Salzmann & P. Fua (2017), arXiv:1706.02025](https://arxiv.org/abs/1706.02025). An early, frank negative result: hard constraints can be enforced at reasonable cost, but in that study they did not beat soft penalties.
- [Climate-invariant machine learning — T. Beucler et al. (2024), Science Advances 10(6), eadj7250](https://arxiv.org/abs/2112.08440). The same group's follow-up. It argues that satisfying physical constraints does not give you extrapolation, and that physically motivated input and output transformations help more.

**Why this matters** — A learned emulator of a physical system can have a small average error and still break, on single samples, a law that the real system obeys exactly: energy or mass conservation, or the power balance of a grid. Anyone building learned surrogates of a physical system, learned OPF solvers included, has to decide whether a conservation law goes into the loss (soft) or into the architecture (hard). This short paper is the clearest small-scale study of what each choice costs and buys.

**Core ideas**

*The problem.* A network maps inputs `x ∈ R^m` to outputs `y ∈ R^p`. The physics says certain combinations of inputs and outputs must vanish exactly. The paper's first move is to rewrite these as a **linear** system of rank `n`:

```
C · [x ; y] = 0,      C ∈ R^{n × (m+p)},   n < m + p
```

What if the physical constraint is nonlinear? The authors **redefine the features and targets** so it becomes linear in the new variables (their Fig. 1). Their second example is the differential constraint `x0² − x0 · dy0/dt = 0`. Discretize it with forward Euler and take `x := x0,t²` as an input feature and `y := Δt⁻¹ · x0,t · (y0,t − y0,t−1)` as the output. The constraint becomes `x − y = 0`, and `y0,t` is recovered afterwards because `x0,t` and `y0,t−1` are also fed as inputs. This trick only works when you can isolate the nonlinearity in quantities you are allowed to choose as inputs or targets. It is worth remembering when you meet the nonlinear AC power-flow equations (A04): they are not linear in voltages and angles, but some balances (for example the total active power balance) are linear in the injection variables.

*Option 1: soft constraints (LCnet).* Add the mean squared constraint residual to the loss:

```
P(x, ŷ) = (1/n) Σ_i ( Σ_j C_ij x_j + Σ_k C_{i,m+k} ŷ_k )²
L(α)    = α · P(x, ŷ) + (1 − α) · MSE(y, ŷ),      α ∈ [0, 1]
```

*Option 2: hard constraints (ACnet).* A standard MLP predicts only `p − n` "direct" outputs. A **fixed, parameter-free layer** then solves the `n` constraint equations (in row-echelon form, bottom to top) for the remaining `n` "residual" outputs. Concatenated, the output satisfies `C·[x; ŷ] = 0` to machine precision for every input, including inputs far outside the training data. The loss is still the MSE on **all** `p` outputs, so gradients flow through the fixed layer, and the network learns direct outputs whose implied residuals are also accurate. This is the key difference from the naive alternative of training an unconstrained net and then overwriting `n` outputs afterwards. Which outputs you choose as residuals is a design choice (the paper calls it `n` extra hyperparameters).

*The error decomposition (Eq. 9).* Assume the truth satisfies the constraints and write `ŷ = y + e`. Then the violation of constraint `i` depends only on the error:

```
(C·[x; ŷ])_i  =  Σ_k C_{i,m+k} e_k
(C·[x; ŷ])_i² =  Σ_k C_{i,m+k}² e_k²   +   Σ_k Σ_{l≠k} C_{i,m+k} C_{i,m+l} e_k e_l
                 (squared error ≥ 0)      (cross-term)
```

With a hard constraint the left side is zero, so the cross-term must exactly cancel the squared-error term. In other words, errors on the variables inside a constraint are forced to be **anti-correlated**. This explains both of the paper's findings. The total of a constrained group gets better, but individual components need not improve, and the error can pile up in whichever outputs were made "residual".

*Setting and results.* The application is a learned subgrid parameterization of convection and clouds for a climate model. There are 304 inputs, 218 outputs (including diagnostic variables), and a sparse `4 × (304 + 218)` constraint matrix encoding column energy, mass, longwave radiation and shortwave radiation conservation. The data are 42M samples each for training, validation and test, and the network is a 5×512 leaky-ReLU MLP with about 1.3M parameters. The main findings:

1. Raising `α` gives a monotonic trade-off: lower constraint violation, higher MSE. A soft penalty never removes violations. It only buys a point on a Pareto curve.
2. The ACnet comes within about 3% of the best unconstrained MSE and satisfies the constraints to about `10⁻⁹ %`.
3. The residual outputs have systematically larger errors. The authors fix this by up-weighting them in the loss by a factor `β > 1`.
4. Enforcing the constraints, hard or soft, reduces the error of the **aggregate** quantity that appears in the energy constraint, but not of its individual components, and not of hard-to-predict, non-Gaussian outputs such as precipitation.
5. Conserving energy does **not** guarantee generalization to a warmer climate. Satisfying a constraint and extrapolating well are different properties.

**A small example**

Think of a toy dispatch with three generators that must exactly cover a load `D = 100 MW` (lossless power balance, A06/A08): `g1 + g2 + g3 − D = 0`, so the constraint row is `C = [−1 | 1 1 1]` acting on `[D ; g1 g2 g3]`. The true dispatch is `(40, 30, 30)`.

- **Unconstrained net** predicts `(42, 31, 30)`. The error is `e = (2, 1, 0)`, the per-output squared errors sum to `4 + 1 + 0 = 5`, and the balance is violated by `3 MW`. Check Eq. 9: `3² = 9 = 5 + 2·(2·1 + 2·0 + 1·0) = 5 + 4`. Positively correlated errors make the violation worse.
- **ACnet** with `g3` as the residual output predicts the same `(42, 31)` directly and computes `g3 = 100 − 42 − 31 = 27`. Now `e = (2, 1, −3)` and the balance holds exactly. But the squared errors sum to `4 + 1 + 9 = 14`, and the cross-term is `2·(2·1 + 2·(−3) + 1·(−3)) = −14`, cancelling them as Eq. 9 says it must. The whole imbalance has landed on generator 3.

So the hard layer turned an infeasible but small-error answer into a feasible answer whose error is concentrated on one unit. During training, the MSE on `g3` pushes the direct outputs to correct themselves, and `β` controls how hard it pushes. On a real grid, an equality-feasible output can still break generator limits: had `g3` a minimum output of 28 MW, the residual 27 MW would violate it. Equalities and inequalities need different tools.

**How it connects**

- It generalizes **X007** (Lagaris et al.), where a boundary condition was satisfied by construction for one PDE, to any set of linear (or linearizable) equality constraints in a generic regressor.
- It is the simplest member of the family that **X016** (HardNet) extends. HardNet adds a closed-form correction for affine *inequalities* as well, while the ACnet only handles equalities (the paper mentions positive activations for simple bounds on direct outputs).
- The toy example uses the power balance from **A06** and **A08**. The "who absorbs the imbalance" picture is the learned-model analogue of the slack bus from **A04/A05**, which picks up whatever mismatch the scheduled injections leave.
- Up next in the focus track is **A17** (modelling stacks: PowerModels.jl, MATPOWER, pandapower). Phase E then makes this entry's dichotomy precise: **E04** (soft vs hard constraints, penalties, augmented Lagrangian, projection), **E05** (constraints by construction) and **E16** (projection and feasibility-restoration layers for OPF).

**Check yourself**

1. Why is an ACnet better than training an unconstrained network and overwriting `n` of its outputs with the constraint solution afterwards?
2. With a hard equality constraint, Eq. 9 forces the cross-term to equal minus the squared-error term. What does this say about the joint error distribution of constrained outputs, and why can individual components get worse even as their sum gets better?
3. In an LCnet, can any finite `α < 1` drive the constraint violation to zero on test data? What does the `α` sweep give you instead?

<details><summary>Answers</summary>

1. In an ACnet the residual outputs are part of the training loss, so gradients pass through the fixed constraint layer, and the network learns direct outputs that also imply accurate residuals. With post-hoc overwriting, the network never sees the error it will induce in the overwritten outputs, so those errors are uncontrolled and typically larger.
2. The errors on the variables inside a constraint must be anti-correlated (weighted by the constraint coefficients), so they cancel in the constrained combination. Only the weighted sum is pinned down. The network can still distribute large, offsetting errors across components, and the dispatch example shows how one output can absorb the whole imbalance.
3. No. With `α < 1` the optimizer trades violation against MSE, so in general it settles at a nonzero violation, and nothing guarantees anything on unseen inputs. The sweep traces a monotonic trade-off curve (lower violation for higher MSE). Exactness on every input requires an architectural or projection-type mechanism.

</details>
