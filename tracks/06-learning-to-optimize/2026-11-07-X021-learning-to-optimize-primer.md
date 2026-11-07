## 2026-11-07 (Saturday) — X021 · Track 06: Learning to optimize, OPF proxies, decision-focused learning — Learning the iterations, not the answer: a primer on learning to optimize

**Type:** Cross-phase preview
**Main reference:** [Learning to Optimize: A Primer and A Benchmark — T. Chen, X. Chen, W. Chen, H. Heaton, J. Liu, Z. Wang & W. Yin (2022), Journal of Machine Learning Research 23(189):1–59](https://www.jmlr.org/papers/v23/21-0308.html) ([arXiv:2103.12828](https://arxiv.org/abs/2103.12828)) — read §1 (motivation and the meta-training objective, Eq. 1), the opening of §2 and §2.1 (LSTM optimizers), the opening of §3, §3.2.3–3.2.4 (objective-based unrolling, safeguarding, LISTA) and §4.4 (take-home messages). Skim the rest of §4; skip §2.3, §3.3 and §3.4 on a first pass.
**Further reading:**
- [Learning Fast Approximations of Sparse Coding — K. Gregor & Y. LeCun (2010), ICML 2010, pp. 399–406](https://mlanthology.org/icml/2010/gregor2010icml-learning) — the LISTA paper: a truncated ISTA turned into a trainable network. The founding example of algorithm unrolling.
- [Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing — V. Monga, Y. Li & Y. C. Eldar (2021), IEEE Signal Processing Magazine 38(2); arXiv:1912.10557](https://arxiv.org/abs/1912.10557) — a tutorial on unrolling, with more on the signal-processing side than the primer gives.

**Why this matters** — There are two basic ways to use learning on a repeated optimization problem. You can learn the map from problem data to solution, or you can learn the algorithm that finds the solution. Anyone building learned OPF solvers will meet both, often in the same paper: "unrolled" ADMM or gradient steps, learned correction steps, learned warm starts. This primer is the clearest map of the second option, and it is honest about where it breaks.

**Core ideas**

**1. Optimizer vs optimizee, and the task distribution.** The paper calls the algorithm the *optimizer* and the problem instance the *optimizee*. A classical optimizer (gradient descent, Newton, the interior-point method of A14–A15) is designed for a *class* of problems defined by mathematical properties, such as "smooth and convex with linear constraints", and comes with a guarantee for every member of that class. Learning to optimize (L2O) trains the optimizer on samples from a *task distribution* `T` of similar optimizees: the same objective with new data each time. The bet is the usual ML bet. If the distribution is narrow, a learned optimizer can "overfit" to it and find short cuts that a worst-case design cannot take. Repeated OPF fits this setting exactly: the same network is solved every few minutes with slightly different loads.

**2. The meta-training objective.** A classical update is `x_{t+1} = x_t − α ∇f(x_t)`. L2O replaces the hand-written rule with a parameterised one that may look at everything seen so far, `z_t = (x_0, …, x_t, ∇f(x_0), …, ∇f(x_t), …)`:

```
min_φ  E_{f ∈ T} [ Σ_{t=1..T} w_t f(x_t) ],     x_{t+1} = x_t − g(z_t, φ)            (Eq. 1)
```

`T` is the *unrolling length*. The loss is backpropagated through all `T` steps, as in backpropagation through time, so memory grows with `T`. The weights `w_t` choose what you reward. `w_T = 1` and all other weights zero means "be good after exactly T steps", which is common in unrolled sparse coding. `w_t = 1` for all `t` means "be good all along", which is the setting of the original LSTM optimizer (Andrychowicz et al., 2016, reference [5] in the paper). ML analogy: this is meta-learning with an inner loop. The outer parameters `φ` are trained on the *trajectory* the inner loop produces, much as MAML trains an initialisation by differentiating through inner SGD steps.

**3. Model-free L2O.** Here `g` is a generic network, usually an LSTM applied coordinate-wise. One set of weights is shared across all coordinates, each coordinate keeps its own hidden state, and gradients go in as `(log|∇|, sign ∇)` to tame their dynamic range. Since a neural network is a universal approximator, it could in principle discover any update rule. In practice the paper lists the costs: no convergence guarantee, many training problems needed, memory limits on `T`, and poor *extrapolation*. A rule trained for 100 steps on small networks often diverges when run for 1000 steps or on a larger network. Basic operations that classical methods take for granted (projection, normalisation, decaying step sizes) have to be relearned from scratch.

**4. Model-based L2O: keep the algorithm, learn its knobs.** Model-based methods start from a known algorithm and make parts of it learnable. There are two main styles.

*Plug-and-play (PnP).* Take a splitting method such as ADMM and replace one proximal step with a separately trained network, for example a denoiser. The network is trained on its own task and then inserted without further training.

*Algorithm unrolling.* Truncate an iterative method at `K` steps, `x_{k+1} = T(x_k; θ_k)`, treat each step as a network layer with its own parameters, and train end to end on a loss of the final iterate `x_K`. The canonical example is LISTA. For LASSO, `min ½‖Ax − d‖² + λ‖x‖₁`, the classical ISTA step is

```
x_{k+1} = η_{λ/L}( x_k − (1/L) Aᵀ(A x_k − d) ),      η_θ(z) = sign(z) · max(0, |z| − θ)
```

where `L` is the largest eigenvalue of `AᵀA`. LISTA rewrites it as `x_{k+1} = η_θ(W_e d + S x_k)` and learns `W_e`, `S` and `θ`. Setting `W_e = Aᵀ/L`, `S = I − AᵀA/L` and `θ = λ/L` gives back ISTA exactly, so the learned network contains the classical algorithm as one parameter setting. Later variants learn far fewer parameters: ALISTA learns only two scalars per layer, a step size and a threshold, and comes with a convergence guarantee.

The primer also separates two *goals* of unrolling. *Objective-based* unrolling aims to minimise `f` faster, and its loss can be `f(x_K)`, a gradient norm, the distance to `x*`, or the fixed-point residual `‖x − T(x)‖`. *Inverse-problem* unrolling aims to recover a true signal, `min_Θ E ‖x_K(Θ, d) − x*_d‖²`. In that case the optimization model is only a surrogate, and the learned method may deliberately *not* converge to its minimiser.

**5. Safeguarding.** To keep a guarantee, propose a learned step and accept it only if it satisfies a descent test, such as a lower objective or a smaller fixed-point residual. If it fails, take the classical step instead. Then the worst case is the classical method plus some overhead, and the typical case is the learned speed. This is the same "trust, but verify" pattern as the filter in A15 and the fallbacks of X018.

**6. What the benchmark found (§4.4).** Open-L2O compares methods on LASSO and sparse recovery, the nonconvex Rastrigin function, and neural-network training (tested on an unseen architecture and dataset). The authors' two take-home messages are these. When suitable structure is available, model-based L2O "clearly outperform[s]" both model-free L2O and analytic optimizers, and does so reliably on in-distribution optimizees. Model-free gains are "not consistent". They come into question on larger optimizees, with more test-time iterations, or out of distribution, and no model-free method won consistently. In the concluding section, the open problems are theory for the L2O training process itself, guarantees outside the task distribution, and scaling.

**A small example**

Take a one-dimensional LASSO, `min ½(a x − d)² + λ|x|`, with `a = 1`, `λ = 1` and data `d = 3`. The solution is `x* = η_1(3) = 2`.

*ISTA with a loose step bound.* Suppose all we know is that `L ≤ 2` and we use step `1/2`. For positive iterates the update is `x_{k+1} = η_{0.5}(0.5 x_k + 1.5) = 0.5 x_k + 1`. Starting at `x_0 = 0` gives `1, 1.5, 1.75, 1.875, …`, and the error `2 · 0.5^k` halves each step. Getting below 0.02 takes 7 iterations.

*LISTA, one layer.* Train `x_1 = η_θ(W_e d + S x_0)` on many values of `d` with `a = 1` fixed. The optimum is `W_e = 1`, `S = 0`, `θ = 1`, which makes `x_1 = η_1(d)`, the exact solution map, in **one** layer for every `d` in the distribution. That is the speed-up the primer's Figure 2 shows at scale: an order of magnitude fewer iterations.

*Leaving the task distribution.* Now a test optimizee has `a = 2` and `d = 3`. The true solution solves `2(2x − 3) + 1 = 0`, so `x* = 1.25`. The trained layer still outputs `η_1(3) = 2`, an error of 0.75. It cannot know that `A` has changed, because `A` was baked into its weights. ISTA, given `a`, still converges. A safeguard that compares the objective of the learned step, `½(4 − 3)² + 2 = 2.5`, with the objective of a classical step would catch the failure here. ISTA's first step from 0, with `L = 4`, gives `η_{0.25}(1.5) = 1.25` and objective `½(2.5 − 3)² + 1.25 = 1.375`. The safeguard rejects the learned step.

**How it connects**

X012 framed "optimization learning" mainly as amortization: learn the map from `x` to `y*(x)` and repair the output. This entry is the other half of the picture, learning the *iteration*. X003 (learning the active set) and X009 (learning branching decisions inside an exact MILP solver) sit between the two: a classical solver whose decisions are learned, which the primer would call close to model-based L2O. The interior-point method of A14 and Ipopt's filter of A15 are exactly the kind of iteration that can be unrolled: the primer cites unrolled proximal interior-point methods with log barriers for constrained image restoration (§3.2.1). The phase A focus continues with A16 on the local optima of nonconvex AC-OPF, which bears on what a learned optimizer is even supposed to converge to. In phase C, C01 (Amos) formalises amortized optimization, which includes unrolled optimizers as a special case. C06 (DC3) unrolls gradient-based correction steps to restore feasibility, and C09 (OptNet) differentiates through the solution instead of through the iterations. F14–F15 return to learned warm starts and local repair as safeguarded hybrids.

**Check yourself**

1. In Eq. 1, what changes in the learned optimizer's behaviour if you set `w_T = 1` and every other `w_t = 0`, instead of `w_t = 1` for all `t`? Why might each choice be preferred?
2. LISTA with `W_e = Aᵀ/L`, `S = I − AᵀA/L` and `θ = λ/L` is ISTA. Why does this nesting matter for both the expressiveness and the trainability of the unrolled model, and what did ALISTA trade away to stabilise training?
3. You unroll 20 iterations of a first-order OPF solver and train it on load profiles from one season. Name two concrete ways it can fail in deployment, and one mechanism that keeps a worst-case guarantee.

<details><summary>Answers</summary>

1. With only `w_T = 1`, the optimizer is rewarded solely for the final iterate, so intermediate iterates can wander, even increase `f`, as long as step `T` is good. This fits a fixed-depth unrolled network used as a fast approximator. With all `w_t = 1`, every iterate counts, which encourages fast, monotone-looking progress and gives a denser training signal through BPTT. It suits an optimizer that might be stopped early or run for a different number of steps, though running beyond the training horizon is still not guaranteed to work.
2. Because ISTA is one parameter setting, the trained network can do at least as well as ISTA at `K` steps in principle, and initialising there gives a sensible starting point. The model only needs to learn a correction to a known good algorithm, which is far more data-efficient than learning an update rule from scratch. Untying all matrices across layers adds capacity but makes training unstable and hard to analyse. ALISTA keeps the matrix structure analytic, computed from `A`, and learns only a step size and a threshold per layer. It gives up most of the free capacity in exchange for stability and a convergence guarantee.
3. Failures: (i) distribution shift: other seasons, unusual renewable output or a topology change alter the effective problem data, as in the `a = 1 → a = 2` example, so the 20-step output may be far from optimal or infeasible. (ii) Horizon mismatch: running more than 20 iterations to "be safe" can make the learned rule diverge, since it was never trained beyond step 20. A guarantee-keeping mechanism is safeguarding: accept each learned step only if a descent or residual test passes, otherwise take the classical step, with a final feasibility check and a fallback to a full solver.

</details>

