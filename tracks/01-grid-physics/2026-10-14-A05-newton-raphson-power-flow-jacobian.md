## 2026-10-14 (Wednesday) — A05 · Track 01: Grid physics & power flow — Solving power flow: Newton–Raphson, the Jacobian, and why solvers fail

**Type:** Focus lesson
**Main reference:** [pandapower — An Open-Source Python Tool for Convenient Modeling, Analysis, and Optimization of Electric Power Systems — L. Thurner, A. Scheidler, F. Schäfer, J.-H. Menke, J. Dollichon, F. Meier, S. Meinecke & M. Braun (2018), IEEE Transactions on Power Systems 33(6)](https://arxiv.org/abs/1709.06743) — read Section IV-A, "Power Flow" (about one page, with Fig. 3: the Newton–Raphson solver, the three initialization options, the speed comparison with MATPOWER and PYPOWER, and reusing the admittance matrix across time steps). Skim Section II-C on how the element-based model is converted to a bus-branch model. The link goes to the open arXiv version. The published version is at [DOI 10.1109/TPWRS.2018.2829021](https://doi.org/10.1109/TPWRS.2018.2829021).
**Further reading:**
- [Power System Analysis — G. Andersson, ETH Zürich lecture notes (2012 version)](https://pees-nov.feit.ukim.edu.mk/predmeti/aees/nastava/literatura/Goran%20Andersson;%20PowerSystemAnalysis.pdf) — ch. 6, "Solution of the Power Flow Problem" (pp. 37–49): Gauss–Seidel, Newton–Raphson in one and many dimensions, quadratic convergence, the power-flow Jacobian (§6.3) and P–δ / Q–V decoupling (§6.4). This is the textbook half of the lesson.
- [pandapower documentation: AC power flow (`runpp`)](https://pandapower.readthedocs.io/en/latest/powerflow/ac.html) — the options discussed below (`algorithm`, `init`, `max_iteration`, `tolerance_mva`, `enforce_q_lims`), with their defaults.

**Why this matters** — Every power-flow dataset used to train a surrogate was produced by a Newton–Raphson solver, and every learned power-flow or OPF model is benchmarked against one. Knowing how the solver works tells you what its labels mean (a tolerance, not exact zeros), which samples are silently missing (the cases it failed on), and why the hardest operating points are also the ones it handles worst.

**Core ideas**

**1. Power flow as root finding.** A04 reduced power flow to a square system. Stack the unknowns `x = (δ for all non-slack buses, V for all PQ buses)` and define the mismatch

```
f(x) = [ P_i(V, δ) − P_i^spec   for every non-slack bus
         Q_i(V, δ) − Q_i^spec   for every PQ bus ]
```

Solving power flow means finding `x*` with `f(x*) = 0`. There is no objective: it is root finding, not optimization. In ML language, `f` is the physics residual of X007, and the solver drives it to zero for one input instead of minimizing it on average over a dataset.

**2. Newton–Raphson.** Linearize at the current iterate and solve the linear model for the step (Andersson §6.2):

```
J(x^ν) Δx = −f(x^ν),     x^{ν+1} = x^ν + Δx,     J = ∂f/∂x
```

Stop when `max |f| < ε`. pandapower's default is `tolerance_mva = 1e-8` MVA, with at most 10 iterations for Newton–Raphson. Near a simple root with a nonsingular `J`, the error roughly squares each iteration (quadratic convergence, §6.2.2): a well-posed case typically converges in 3–5 iterations from a flat start. This is the deterministic, exact-Hessian counterpart of the second-order methods ML people rarely run: one linear solve per step, but very few steps.

**3. The Jacobian has four blocks, and they mirror the graph.** Order the unknowns as angles, then magnitudes:

```
J = [ ∂P/∂δ   ∂P/∂V ]
    [ ∂Q/∂δ   ∂Q/∂V ]
```

Writing `δ_ik = δ_i − δ_k`, the entries are (Andersson §6.3, in his notation `U` for `V` and `θ` for `δ`):

```
off-diagonal (k ≠ i):  ∂P_i/∂δ_k =  V_i V_k (G_ik sin δ_ik − B_ik cos δ_ik)
                       ∂Q_i/∂δ_k = −V_i V_k (G_ik cos δ_ik + B_ik sin δ_ik)
diagonal:              ∂P_i/∂δ_i = −Q_i − B_ii V_i²     ∂P_i/∂V_i = P_i/V_i + G_ii V_i
                       ∂Q_i/∂δ_i =  P_i − G_ii V_i²     ∂Q_i/∂V_i = Q_i/V_i − B_ii V_i
```

Every off-diagonal entry carries a factor `G_ik` or `B_ik`. So entry `(i, k)` is nonzero only if buses `i` and `k` are connected, and each block has the sparsity pattern of the admittance matrix: the graph Laplacian of A03. A grid with 10,000 buses and about 3 lines per bus has a Jacobian with well under 0.1 % nonzeros. Real solvers never invert `J`. They factor it with sparse LU, and the cost depends on fill-in, which depends on the graph's structure and the elimination ordering. That is why topology, not just bus count, sets the runtime.

**4. Cheaper variants trade iterations for cost per iteration.**
- *Dishonest Newton*: keep an old `J` for several steps. More iterations, no refactoring (Andersson §6.2.1).
- *Fast decoupled*: on transmission grids `∂P/∂V` and `∂Q/∂δ` are small (A02's P–δ and Q–V coupling), so drop them and solve two half-size systems. Andersson §6.4 stresses the key point: if the iteration converges, it converges to the *exact* solution, because the residual `f` is unchanged. Only the step is approximate. This is a preconditioning idea, not a model approximation.
- *Backward/forward sweep*: exploits radial (tree) distribution feeders. It is pandapower's `bfsw`.
- *Gauss–Seidel*: cheap, slow, still in textbooks; pandapower's default cap is 1000 iterations against Newton's 10.

**5. What pandapower adds (Section IV-A).** Its solver began as a port of PYPOWER (itself a port of MATPOWER), but it assigns bus types and does the per-unit conversion automatically from an element-based model. Building the Jacobian is JIT-compiled with numba. Three findings worth remembering:
- *Initialization matters.* Options are a flat start (`V = 1`, `δ = 0`), a DC power flow (A06) for the angles, or the previous result. The paper recommends DC initialization in meshed grids, "where large voltage angle differences between the buses might lead to non-convergence in case of a flat start".
- *Overhead dominates small cases.* On MATPOWER test cases from 30 to 9241 buses (Fig. 3), converting to the bus-branch model takes more than half the time on small grids. pandapower is slower than MATPOWER there and faster on medium and large grids.
- *Reuse across time series.* If only injections change between runs, the admittance matrix and model conversion can be reused, and the previous solution makes a good warm start. Generating many training samples on one topology looks exactly like this.

**6. How power flow fails.** "Did not converge" (pandapower raises `LoadflowNotConverged`) has distinct causes, and a dataset generator should not treat them as one:
- *No solution exists.* The load is beyond the nose of the PV curve (A04). `J` becomes singular as you approach the nose, because two solutions merge there. No solver can fix this.
- *Bad start.* A solution exists but the iterate leaves its basin, for example with large angles from a flat start.
- *Wrong root.* The iteration converges, but to the low-voltage solution. The residual is zero and the answer is still useless.
- *Bus-type switching.* With `enforce_q_lims`, generators that hit reactive limits become PQ buses and the power flow runs again (A04). Each switch can push the case closer to infeasibility.

**A small example**

Use A04's two-bus grid: line `j0.1` pu, slack `V_1 = 1∠0`, PQ load at bus 2. The mismatch and Jacobian are

```
f(δ, V) = [ 10 V sin δ − P_2,   10V² − 10 V cos δ − Q_2 ]
J       = [ 10 V cos δ    10 sin δ
            10 V sin δ    20V − 10 cos δ ]
```

**Light load**, `P_2 = −0.5`, `Q_2 = −0.2`, flat start `(δ, V) = (0, 1)`. Here `J = [[10, 0], [0, 10]]` and `f = (0.5, 0.2)`, so the first step is `(−0.05 rad, −0.02)`.

| iter | δ (deg) | V (pu) | max abs mismatch | det J |
|---|---|---|---|---|
| 0 | 0 | 1 | 5.0·10⁻¹ | 100 |
| 1 | −2.865 | 0.9800 | 1.6·10⁻² | 93.8 |
| 2 | −2.930 | 0.97825 | 3.6·10⁻⁵ | 93.3 |
| 3 | −2.930 | 0.97825 | 1.7·10⁻¹⁰ | 93.3 |

The mismatch exponent roughly doubles each step (−1, −2, −5, −10). This is quadratic convergence, and the answer is A04's high-voltage solution.

**Heavy load**, `P_2 = −4.9`, `Q_2 = 0` (98 % of the nose limit `1/(2x) = 5`). From a flat start it takes 7 iterations to reach `V = 0.774`, `δ = −39.3°`, and `det J` falls from 100 to 15.4. The linearization is getting worse and the steps are getting less reliable.

**Beyond the nose**, `P_2 = −5.1`. `det J` drops to about 9.6, changes sign, and comes back near zero at `V ≈ 0.707`, `δ ≈ −45°`, which is exactly the nose point. The next step jumps to `V = 2.03`, `δ = +62°`, and the iteration diverges. No tolerance or iteration count would help.

**Bad start**, light load again but from `V = 0.1`. The iteration wanders through angles beyond −180° and converges to `V = −0.055`, `δ = −245.3°`. That is the same point as `V = 0.055`, `δ = −65.3°`: A04's low-voltage solution, with a residual of `10⁻¹⁶`. A residual check alone would label this as a success.

**How it connects**

A04 wrote down `f`, and this lesson solves `f = 0`. The Jacobian's blocks inherit the Laplacian sparsity of A03, and fast decoupled power flow exploits A02's P–δ / Q–V intuition. X007 previewed residual losses: Newton drives that same residual to machine precision for one case, while a learned model only lowers its average. That is why NR-generated labels (as in X004's line-graph GNN) count as ground truth only up to `tolerance_mva` and only for cases that converged. The singular Jacobian at the nose is also where the implicit map from loads to voltages stops being differentiable, which matters when later lessons differentiate through solvers (A13 on KKT, C-phase decision-focused learning). Next, A06 linearizes everything around a flat profile to get DC power flow: the same DC solution that pandapower uses to initialize Newton.

**Check yourself**

1. Fast decoupled power flow drops two Jacobian blocks. Why does it still converge to the exact AC solution when it converges, and why might it fail more often on a heavily loaded distribution feeder?
2. You generate 50,000 training samples by sampling loads and running Newton–Raphson, and you discard runs that raise `LoadflowNotConverged`. What bias does this introduce in the training set, and which part of the input space will a model trained on it know least about?
3. In the "bad start" example the residual was `10⁻¹⁶`. Give a cheap check that would have flagged the result as unphysical.

<details><summary>Answers</summary>

1. The decoupled method only changes the step `Δx`. The residual `f` is still the full AC mismatch, so any fixed point satisfies the exact equations. On distribution feeders, lines have a high `R/X` ratio (A02), so `∂P/∂V` and `∂Q/∂δ` are not small. With heavy loading, the dropped coupling makes the approximate step a poor direction, and the iteration converges slowly or not at all.
2. The surviving samples are biased away from the stability limit. Points near and beyond the nose, and cases with a bad basin of attraction, are under-represented or missing. The model therefore knows least about heavily stressed operating points, which are exactly where operators most need accurate answers. It will also never output "no solution exists", because that label never appears.
3. Check that voltage magnitudes are positive and inside a plausible band (for example 0.9–1.1 pu), and that angle differences across lines are small (wrap angles to (−180°, 180°] first). Here `V = −0.055` (or 0.055 after rewrapping) fails immediately. Rewritten with `V > 0`, the low-voltage root also has `det J ≈ −5.25`, against `+93.3` at the physical root, and a rerun from a DC or flat start lands on the high-voltage solution.

</details>
