## 2026-10-13 (Tuesday) — X007 · Track 10: Physics-informed and constrained learning — Before PINNs: residual losses and constraints satisfied by construction (Lagaris et al., 1998)

**Type:** Cross-phase preview
**Main reference:** [Artificial Neural Networks for Solving Ordinary and Partial Differential Equations — I. E. Lagaris, A. Likas & D. I. Fotiadis (1998), IEEE Transactions on Neural Networks 9(5):987–1000](https://arxiv.org/abs/physics/9705023) — read Sections 1–3 (about 6 pages: the collocation loss, eq. 3; the trial-solution split, eq. 4; derivatives of the network with respect to its inputs, eqs. 5–10; and the trial forms for ODEs, coupled ODEs and Dirichlet/Neumann PDEs, eqs. 11–21). Then read Section 4.3 with Table 1 (comparison with finite elements) and Section 5. Skim the model problems in Sections 4.1–4.2. The link goes to the open arXiv preprint (1997). The published version is at [DOI 10.1109/72.712178](https://doi.org/10.1109/72.712178).
**Further reading:**
- [Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems — J. Willard, X. Jia, S. Xu, M. Steinbach & V. Kumar (2022), ACM Computing Surveys 55(4)](https://arxiv.org/abs/2003.04919) — Section 3 sorts the field into four families: physics-guided loss functions, initialization, architecture design and hybrid physics–ML models. It is a useful map for phase E.

**Why this matters** — Anyone training a network to output grid states or dispatch decisions has to choose how the physics enters. One option is a penalty in the loss, which the network may still violate. The other is a parameterization in which some constraints cannot be violated at all. This 1998 paper, written two decades before the term "PINN", sets up both options cleanly on toy differential equations. It is the shortest route to the vocabulary that phase E builds on.

**Core ideas**

**1. The problem as a loss.** Take a differential equation `G(x, Ψ(x), ∇Ψ(x), ∇²Ψ(x)) = 0` on a domain `D`, with boundary conditions (BCs) on its edge. Pick collocation points `x_i ∈ D̂ ⊂ D` and a model `Ψ_t(x, p)` with parameters `p`. The paper solves (eq. 3)

```
min_p  Σ_{x_i ∈ D̂}  G(x_i, Ψ_t(x_i,p), ∇Ψ_t(x_i,p), ∇²Ψ_t(x_i,p))²     subject to the BCs
```

To an ML reader this is self-supervised regression. There are no labels `Ψ(x_i)`, only the requirement that a known operator's residual be zero at sampled inputs. The "dataset" is just a set of points where the equation is checked. Modern PINNs (lesson E01) keep this loss, compute the derivatives with autodiff, and handle the BCs differently.

**2. Constraints by construction.** The key move is eq. 4:

```
Ψ_t(x) = A(x) + F(x, N(x, p))
```

Here `A` has no trainable parameters and satisfies the BCs exactly. `F` contains the network `N` but is shaped so that it vanishes wherever a BC is imposed. Examples from Section 3:

- first-order initial value problem `Ψ(0) = A`: `Ψ_t(x) = A + x·N(x)` (eq. 11);
- two-point Dirichlet problem `Ψ(0) = A`, `Ψ(1) = B`: `Ψ_t(x) = A(1−x) + Bx + x(1−x)·N(x)` (eq. 14);
- Poisson on the unit square: `Ψ_t = A(x,y) + x(1−x)y(1−y)·N(x,y)`, where `A` interpolates the four boundary functions (eqs. 17–18).

The authors state the payoff directly: the problem "has been reduced from the original constrained optimization problem to an unconstrained one". In optimization language, you have parameterized the feasible set, so any `p` gives a feasible point and plain gradient methods (they use BFGS) apply. ML analogies: a softmax output layer makes "probabilities sum to one" impossible to violate, and a residual connection `x + f(x)` builds in a known baseline. Mixed BCs are handled too. For a Neumann condition on one edge, eq. 19 subtracts `N(x,1)` and `∂N/∂y(x,1)` so that the correction's normal derivative is zero there.

**3. Derivatives of a network with respect to its inputs.** For one hidden layer, `N = Σ_i v_i σ(w_i·x + u_i)`, and any mixed input derivative is again a one-hidden-layer network with the same `w_i, u_i`. Its output weights are `v_i Π_k w_ik^{λ_k}`, and the activation is replaced by its `Λ`-th derivative (eqs. 5–7). Section 2.1 then derives the parameter gradients of these "gradient networks" by hand (eqs. 8–10). Today autodiff does this. The structural point still matters: the loss depends on `σ''` (and higher derivatives), so the choice of activation shapes the loss landscape. That observation returns in the PINN training pathologies of E02.

**4. What they found.** All experiments use 10 sigmoid hidden units, which is 30 parameters for ODEs and 40 for PDEs. Training uses 10 equidistant points per dimension (a 10×10 grid for PDEs). The test problems have known analytic solutions: four ODE problems (including a coupled system) and four PDEs (Dirichlet, mixed BCs, and one nonlinear PDE). Table 1 compares the maximum deviation from the exact solution with a finite-element (FEM) solver using an 18×18 biquadratic mesh (1369 unknowns). The FEM is often more accurate at its own nodes. The network's error at 900 unseen interior points, however, matches its training-point error. FEM interpolation degrades by one to two orders of magnitude, for example from `2·10⁻⁸` at the nodes to `1.5·10⁻⁵` in between on problem 5, where the network gives `5·10⁻⁷` at both.

**5. Read it critically.** These are smooth, low-dimensional problems on boxes with a tiny network, and the paper does no ablation over architecture (Section 5 says so). Hand-built `A` and `F` exist only for simple geometries and simple constraint types. Section 5 also anticipates adaptive placement of collocation points where the error is high, which later became standard in PINN practice. What carries over: penalties give approximate feasibility with a tunable weight, while constructions give exact feasibility but require you to find a parameterization of the feasible set.

**A small example**

Solve `Ψ' = −Ψ`, `Ψ(0) = 1` on `[0, 1]` (exact solution `e^{−x}`, `e^{−1} = 0.368`). Collocate at `x ∈ {0, 0.5, 1}` and replace the network with a single parameter so everything can be done by hand.

**Hard constraint (Lagaris).** `Ψ_t = 1 + a·x`, residual `r(x) = Ψ_t' + Ψ_t = (1 + a) + a·x`. The loss is `(1+a)² + (1+1.5a)² + (1+2a)²`. Setting its derivative to zero gives `4.5 + 7.25a = 0`, so `a = −0.621`. Then `Ψ_t(0) = 1` exactly and `Ψ_t(1) = 0.379`.

**Soft constraint (penalty).** `Ψ = c + a·x` with loss `Σ r² + λ(c − 1)²`, where now `r = c + a(1 + x)`. First note that without the penalty, `c = a = 0` gives zero residual: the trivial solution `Ψ ≡ 0` satisfies the ODE perfectly. The boundary condition is what selects the solution. The normal equations give `a = −0.621c` and `(3 + λ)c + 4.5a = λ`.
- `λ = 1`: `c = 0.829`, so the initial condition is violated by 17 % and `Ψ(1) = 0.314`.
- `λ = 100`: `c = 0.998`, a violation of 0.2 %.

Raising `λ` buys feasibility. In a real network it also worsens conditioning, which leads to the loss-balancing problem of E03 and the penalty-versus-augmented-Lagrangian discussion of E04.

**How it connects**

The power-flow equations of A04 are algebraic, not differential, but they fit the same template. A residual loss is `Σ_i (P_i(V, δ) − P_i^G + P_i^L)² + (same for Q)` at every bus. A "boundary condition" is anything pinned by bus type, such as the slack bus's `V_1 = 1, δ_1 = 0` or a PV bus's voltage setpoint. Those can be imposed by construction by simply not predicting them: output only the free coordinates and insert the pinned ones, as in eq. 4. The nonlinear balance equations themselves have no closed-form `A + F` split. That gap is why phase C has completion and feasibility-restoration layers (C06) and phase E has projection layers (E16). The `σ''`-dependence of the loss foreshadows E02. The label-free training signal is the counterpart to X004's line-graph GNN, which regressed Newton–Raphson labels and had nothing enforcing power balance. Next in the focus phase, A05 solves the A04 equations with Newton–Raphson, the classical solver that every residual-loss model is measured against.

**Check yourself**

1. In `Ψ_t(x) = A(1−x) + Bx + x(1−x)N(x)`, why does the factor `x(1−x)` not limit what the model can represent in the interior, and what could go wrong near the boundary?
2. The paper's network matched its training error at unseen points, while FEM did not. Give two reasons this comparison flatters the network.
3. A model outputs `(V_i, δ_i)` for all buses of a grid and is trained with the power-flow residual alone. Which degeneracy, analogous to `Ψ ≡ 0` in the example, does it have, and which A04 fact fixes it?

<details><summary>Answers</summary>

1. Any target `Ψ` that meets the BCs can be written as `A + x(1−x)h(x)` with `h = (Ψ − A)/(x(1−x))`. This `h` is smooth in the interior and has finite limits at the endpoints when `Ψ` is differentiable, so the network only has to approximate `h`. Near the boundary, though, the residual depends on `h` through derivatives of a product that is close to zero. Errors in `h` are damped there and amplified in the derivative terms, which can make the loss poorly conditioned close to the edges.
2. First, the test problems are smooth, low-dimensional and have analytic solutions, a regime that suits a global smooth approximator. Second, the network's "test points" lie inside the same box as its 10×10 training grid, so they test interpolation, not higher dimensions, rough solutions or out-of-distribution inputs. (FEM between its nodes is limited by its piecewise-polynomial basis, and neither method was tuned.)
3. The equations depend only on angle differences (`Y·1 = 0` in A03), so any common shift of all `δ_i` gives the same residual. Without a reference the model has a flat direction and ill-posed targets. Fixing the slack's `δ_1 = 0` (and its `V_1`), and pinning the other bus-type setpoints by construction, removes it. Even then, the low-voltage second solution from A04's example is a residual-zero point that the loss alone does not exclude.

</details>
