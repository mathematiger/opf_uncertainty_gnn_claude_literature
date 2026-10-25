## 2026-10-25 (Sunday) — X014 · Track 08: Uncertainty as input: distributions, sets, embeddings — How wide must a set encoder be? Wagstaff et al. on the limits of sum pooling

**Type:** Cross-phase preview
**Main reference:** [On the Limitations of Representing Functions on Sets — E. Wagstaff, F. B. Fuchs, M. Engelcke, I. Posner & M. A. Osborne (2019), Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97](https://arxiv.org/abs/1901.09006). Read Section 1, Section 2 (Definitions 2.1–2.4 and the two background Theorems 2.8 and 2.9, with the proof sketch of 2.9), Section 3 up to Theorem 3.1, and all of Section 4 (Theorem 4.1, Lemma 4.2 and its proof, Theorems 4.3–4.4, §4.1 Discussion, §4.2 with Figure 3). The rest of Section 3 (Theorems 3.2–3.3) and the appendices are optional. About 8 pages. The link goes to the open arXiv version (v2, October 2019).
**Further reading:**
- [Universal Approximation of Functions on Sets — E. Wagstaff, F. B. Fuchs, M. Engelcke, M. A. Osborne & I. Posner (2022), JMLR 23(151):1–56](https://jmlr.org/papers/v23/21-0730.html): the journal follow-up. It shows that a latent space one dimension too small can make some piecewise-affine functions no better approximable than a constant, and places Deep Sets inside Janossy pooling.

**Why this matters** — A forecast that arrives as M scenarios, an ensemble of M weather runs, or M samples from a generative model is an unordered set. Anyone who wants a network to *read* such a set, not just its mean, has to choose a pooling operation and a latent width. This paper gives the clean answer for sum pooling: to represent every continuous function of a set of M scalars, the space you sum in needs at least M dimensions, and M is also enough.

**Core ideas**

**The model.** The architecture behind Deep Sets and PointNet is

```
f(X) = ρ( Σ_{x ∈ X} φ(x) ),    φ : R → R^N,   ρ : R^N → R
```

Each element is encoded on its own by `φ`, the codes are summed (which makes the result permutation-invariant), and `ρ` decodes the sum. The paper calls `f` *sum-decomposable via R^N*, and *continuously* sum-decomposable when `φ` and `ρ` can be chosen continuous. The latent width `N` is the number we care about. (The paper's theorems treat sets of scalars. The journal follow-up treats the general case.)

**What was known (Section 2).** You already know the countable case from GIN: if the elements come from a countable universe, enumerate it with `c(x) ∈ N` and set `φ(x) = 4^(−c(x))`. Every finite set gets a distinct base-4 expansion, so the sum is injective and a one-dimensional latent space suffices (Theorem 2.8, from Zaheer et al.). For real-valued elements Zaheer et al. used power sums. For a set of size M in `[0,1]`,

```
Φ_q(X) = Σ_{m=1}^{M} x_m^q,    q = 0, …, M
```

is continuous and injective with a continuous inverse, so any continuous symmetric `f` equals `ρ ∘ Φ` with `ρ = f ∘ Φ⁻¹` (Theorem 2.9, latent width M+1).

**Continuity is the whole story (Section 3).** The countable trick hides a problem. Take the rationals as the universe. Theorem 3.1 shows there are functions on finite sets of rationals for which *every* sum-decomposition through R has a `φ` that is discontinuous at every point. A neural network cannot represent that, so Theorem 2.8 says nothing about what a trained network can do. The useful question is about continuous `φ` and `ρ` on real inputs.

**The necessity result (Theorem 4.1).** For `M > N` there are continuous symmetric functions `f : R^M → R` that are not continuously sum-decomposable via `R^N`. The witness is a function you would actually want: `max`. The proof has two steps.

- *Lemma 4.2.* If `max(X) = ρ(Σ_x φ(x))` holds for all finite sets, then the sum map `Φ_M` restricted to sets of size M is injective. The proof is by induction: equal sums give equal maxima, so remove the maximum and recurse.
- *Topology.* Restrict `Φ_M` to sorted vectors `x_1 < … < x_M`. That is an open convex subset of `R^M`, so it is homeomorphic to `R^M`. Now `Φ_M` would be a continuous injection from `R^M` into `R^N` with `N < M`, and no such map exists.

**The sufficiency result (Theorems 4.3–4.4).** The power-sum construction can be tightened to width exactly M, even when set sizes vary up to M. So:

```
continuous symmetric f on sets of ≤ M reals  ⇔  ρ(Σ φ(x)) with φ, ρ continuous and N = M
```

N = M is necessary in the worst case and sufficient in all cases.

**What it does not say (§4.1).** It does not say every function needs width M. The mean needs one dimension, and the sum and count need one each. The theorem is a statement about *arbitrary* targets. Real tasks are often simpler, and the data can be redundant.

**The experiment (§4.2).** The authors train `ρ(Σφ)` with large MLPs to predict the median of sets of size M (15 to 500) drawn from uniform, Gaussian or Gamma distributions, and sweep N. Three findings: the error falls monotonically in N; it plateaus beyond a "critical" N; and the critical N grows roughly linearly with M (Figure 3b). The critical points sit *below* N = M, because the network learns to estimate the median of the training distributions, not an exact algorithm. That gap is a lesson in its own right: a set encoder trained on one family of input distributions may be using a compressed code that fails on another.

**A small example**

Take sets of two numbers, `X = {x_1, x_2}`, and sum in `R^2` with `φ(x) = (x, x²)`:

```
s_1 = x_1 + x_2,   s_2 = x_1² + x_2²
```

For `X = {1, 3}`: `s_1 = 4`, `s_2 = 10`. Recover the set from the elementary symmetric polynomials: `e_1 = s_1 = 4` and `e_2 = (s_1² − s_2)/2 = 3`. The elements are the roots of `t² − 4t + 3 = (t − 1)(t − 3)`. So `ρ` can compute anything, for example the maximum:

```
max = ( s_1 + sqrt(2 s_2 − s_1²) ) / 2 = ( 4 + sqrt(20 − 16) ) / 2 = 3
```

using `2 s_2 − s_1² = (x_1 − x_2)²`. This `ρ` is continuous.

Now try width 1. A continuous `φ : R → R` would make `(x_1, x_2) ↦ φ(x_1) + φ(x_2)`, on the open half-plane `x_1 < x_2`, a continuous injection from a 2-D region into a line. Picture squashing a sheet onto a wire without two points ever landing on the same spot: removing one point from a line disconnects it, but removing one point from a plane does not. So no continuous width-1 sum encoder recovers the max of two numbers. Read "two numbers" as "two scenarios of tomorrow's wind output" and the same holds: an encoder narrower than the number of scenarios cannot, in general, recover a quantity as basic as the worst case.

**How it connects**

The obvious link is **X005**. A kernel mean embedding `μ_P = E_{x∼P}[k(x, ·)]` is exactly a sum (mean) decomposition with an infinite-dimensional `φ(x) = k(x, ·)`, which is why a characteristic kernel can be injective on distributions. As soon as you replace the RKHS with a finite feature map (random features, a learned encoder), this paper's width bound applies. Note one difference: the mean `(1/M) Σ φ(x)` forgets the set size and multiplicities, which a sum keeps. It also connects to **X004** and **X013**: one message-passing step is a sum decomposition over a neighbourhood, so a GNN layer with continuous node features and width below the neighbourhood size cannot realize every continuous aggregation. GIN's injectivity argument lives in the countable case, the one Section 3 warns about. For scenario inputs, it complements **X002** and **X011**: once forecasts are sets of samples, how the model reads them becomes an architecture question.

Phase D picks this up directly. **D02** develops kernel mean embeddings and distribution regression, **D03** the Deep Sets theorem that this paper refines, **D04** attention pooling in the Set Transformer (pairwise interactions *before* pooling, one way to buy expressivity without enormous N), and **D06** asks how to represent a forecast distribution as an input to an OPF learner. The next focus lesson is **A10**, the AC-OPF problem written out in full.

**Check yourself**

1. Why does the countable-universe result `φ(x) = 4^(−c(x))` give no useful guidance for a neural set encoder on real-valued inputs?
2. In the proof of Theorem 4.1, why is it enough to restrict to sorted vectors `x_1 < … < x_M`, and what does `max` contribute to the argument?
3. The median experiment finds critical widths *below* M. Does this contradict Theorem 4.1? What risk does it point to when such an encoder is deployed on input distributions it was not trained on?

<details><summary>Answers</summary>

1. That construction relies on enumerating the universe. On the rationals, Theorem 3.1 shows that some functions force a sum-decomposition via R whose `φ` is discontinuous at every point. Networks (and GPs) represent continuous functions, so a statement relying on such a `φ` says nothing about what a trained encoder can represent. For real inputs, the meaningful question is continuous decomposability, and there the width must grow with M.
2. Sorted vectors give each set of M distinct reals exactly once, and they form an open convex region homeomorphic to `R^M`. `max` forces injectivity: Lemma 4.2 shows that if `ρ(Σφ)` computes the max for all set sizes, then equal sums imply equal sets, by peeling off the maximum and inducting. A continuous injection from an `R^M`-like region into `R^N` with `N < M` cannot exist (invariance of domain), which is the contradiction.
3. No. The theorem is a worst case over all continuous functions and all inputs. The network learned the median only for inputs drawn from the training distributions, where it can use redundancy and need not be injective. The risk is that this compressed code is distribution-specific: on a different input family (another weather regime, another scenario generator) the information discarded below width M may be exactly what is needed, and the error can rise without any warning sign at the encoder.

</details>
