## 2026-10-27 (Tuesday) — X015 · Track 09: Aleatoric/epistemic UQ and evaluating uncertainty signals — When the answer is not unique: Bishop's mixture density networks

**Type:** Cross-phase preview
**Main reference:** [Mixture Density Networks — C. M. Bishop (1994), Technical Report NCRG/94/004, Neural Computing Research Group, Aston University](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf). This is the open PDF from the Aston repository (26 pages; record page [here](https://publications.aston.ac.uk/id/eprint/373)). Read Section 1 and Section 2 (least squares gives the conditional mean). Read Section 3 up to eq. (31) (the model and its loss), then the part of Section 3 from eq. (45) to eq. (49) (mean, variance and mode of the predicted density). Read Section 5 (the toy inverse problem, Figures 4–7) and Section 7 (discussion, choosing the number of kernels). Skim the gradient derivations and Section 6 (robot arm).
**Further reading:**
- [Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction — O. Makansi, E. Ilg, Ö. Çiçek & T. Brox (2019), CVPR 2019](https://arxiv.org/abs/1906.03631): a modern look at how MDNs fail in practice (unstable training, mode collapse). It replaces direct likelihood training with winner-takes-all sampling followed by fitting a mixture. It also discusses how to evaluate a multimodal prediction when each input comes with only one observed outcome.

**Why this matters** — Many maps that a learned OPF or power-flow model must learn are inverse maps: from a desired outcome back to setpoints or states. Inverse maps can be multi-valued, and a network trained with squared error then predicts the average of several valid answers, which may itself be invalid. Bishop's short report is where the standard fix comes from: predict a full conditional density. It also gives the first clean example of a "variance" that is neither noise nor ignorance but ambiguity. Read it before phase D's heteroscedastic heads and ensembles. It is the simplest model in which the uncertainty decompositions used there can be written down exactly.

**Core ideas**

**1. Least squares learns the conditional mean, and only that.** With unlimited data and a flexible network, minimising `E = ½ ∫∫ ‖f(x) − t‖² p(t, x) dt dx` gives `f(x) = ⟨t | x⟩`, the conditional mean (eqs. 4–7). Bishop splits the loss into `½ ∫ ‖f(x) − ⟨t|x⟩‖² p(x) dx` plus `½ ∫ (⟨‖t‖²|x⟩ − ‖⟨t|x⟩‖²) p(x) dx`. Only the first term depends on the weights. The second is the average conditional variance left at the optimum (eqs. 8–9). So a trained MSE network carries two statistics: a mean that depends on `x` and *one global* variance (the residual loss). Equivalently, MSE is Gaussian maximum likelihood with a fixed, input-independent variance. For classification with one-hot targets the conditional mean *is* the vector of class posteriors, which is why cross-entropy and MSE heads can be calibrated at all. For continuous targets the mean is a single summary of `p(t|x)`, and it can be a very poor one.

**2. The failure case: multi-valued inverse problems.** Many physical systems have a single-valued *forward* map (causes → effects) and a multi-valued *inverse* map (desired effect → causes). Bishop's toy example (Section 5) samples `x = t + 0.3 sin(2πt) + ε`, with `ε ~ U(−0.1, 0.1)` and 1,000 points. Regressing `x` on `t` is easy: a 5-hidden-unit MLP fits it well. Regressing `t` on `x` with the *same data* fails. For intermediate `x` there are three valid branches, and the network draws a curve through their average, which lies far from all of the data. "The average of several solutions is not necessarily itself a solution." The fit is also fragile: it is sensitive to architecture and initialisation, and the best result needed careful tuning.

**3. The model.** Replace the single Gaussian by a mixture whose parameters are all outputs of a network:

```
p(t | x) = Σ_{i=1..m} π_i(x) · N(t ; μ_i(x), σ_i(x)² I)          (eqs. 22–23)
π_i = softmax(z^π)_i,   σ_i = exp(z^σ_i),   μ_ik = z^μ_ik         (eqs. 25–27)
E = − Σ_q ln Σ_i π_i(x_q) N(t_q ; μ_i(x_q), σ_i(x_q)² I)           (eq. 29)
```

For a `c`-dimensional target the network has `(c + 2)·m` outputs. The softmax keeps the mixing weights on the simplex. The exponential keeps the widths positive and avoids widths of exactly zero. The loss is a negative log-likelihood, trained by ordinary backpropagation. The gradients have an EM-like form: the posterior responsibility `γ_i = π_i N_i / Σ_j π_j N_j` (eq. 30) of each component for each training point weights its update. Bishop notes that a Gaussian mixture with isotropic components can still approximate any density given enough components. So the model is fully general, even though each component treats the output dimensions as independent. With `m = 1` the MDN reduces to a heteroscedastic Gaussian regressor, and with the width also fixed it reduces to least squares.

**4. Reading the density out.** The predicted mean is `Σ_i π_i μ_i`, which is exactly what least squares would give (eq. 45). The predicted variance (eq. 47, written here for a scalar target) is:

```
s²(x) = Σ_i π_i σ_i²   +   Σ_i π_i (μ_i − Σ_j π_j μ_j)²
        (within-component)  (between-component)
```

This is the law of total variance, with the component label as the conditioning variable. For a single answer, Bishop suggests one of two choices. The first is the centre of the component with the largest peak, `max_i π_i/σ_i^c` (eq. 48), as an approximate mode. The second, which suits mappings with a few distinct branches, is the centre of the component with the largest mass, `max_i π_i` (eq. 49). The second choice is also less sensitive to artificially small `σ_i` in sparse regions.

**5. Results and the open practical issue.** On the toy problem an MDN with 3 kernels, a 20-hidden-unit MLP and 9 outputs recovers the three branches. Figure 7 shows `p(t|x)` with three modes at `x = 0.5`. Using 5 kernels gives almost the same result. Surplus kernels are harmless in practice, because the network can switch them off (`π_i → 0`) or merge them. On the 2-link robot arm, the MDN separates "elbow up" from "elbow down" where both reach the target. Bishop leaves model-order selection (how many kernels) open. Later work, such as the further reading, documents the main practical problems: unstable training, components collapsing onto a few points, and mode collapse.

**What this means for uncertainty.** The two terms in `s²(x)` look like the "aleatoric + epistemic" split that ensembles report (phase D). Here, however, *both* are properties of the data-generating process. The between-component term measures how ambiguous the mapping is, not how ignorant the model is. A single Gaussian head would fold that ambiguity into one large `σ(x)` centred in an empty region. A calibration check on intervals around the mean can then pass even though the point prediction is never correct.

**A small example**

Take a lossless 2-bus line with reactance `X`. The slack bus 1 is held at `V_1 = 1` pu. Bus 2 draws active power `P` and no reactive power. From A04's equations, the active balance gives `P·X = V_2 sin θ` (where `θ` is the angle difference). The reactive balance gives `V_2² = V_2 cos θ`, so `V_2 = cos θ`. Substituting, `P·X = sin θ cos θ = ½ sin 2θ`.

Set `P·X = 0.4` pu. Then `sin 2θ = 0.8`, so `2θ = 53.13°` or `126.87°`. The two roots are:

- `θ = 26.57°`, `V_2 = 0.894` pu (the high-voltage, normal operating point);
- `θ = 63.43°`, `V_2 = 0.447` pu (the low-voltage root, the one a voltage band removes, as in A09).

Suppose a dataset of solved cases contains both roots, for example because Newton–Raphson (A05) was started from different initial points, in a 70 / 30 ratio. A least-squares network then predicts `0.7·0.894 + 0.3·0.447 = 0.760` pu. Check it: `V_2 = 0.760` requires `θ = arccos 0.760 = 40.5°`, and then `½ sin 81° = 0.494 ≠ 0.4`. The averaged prediction violates power balance. A Gaussian-variance head would report a standard deviation of about `√(0.7·0.134² + 0.3·0.313²) = √0.0420 ≈ 0.205` pu around a value that is not a solution. A 2-component MDN with `π = (0.7, 0.3)`, `μ = (0.894, 0.447)`, `σ = 0.01` reports the same mean and almost the same total variance. But nearly all of that variance (0.0420 out of 0.0421) is between-component. Its largest-mass mode (eq. 49) is a valid state. As `P·X → 0.5` (the nose of the curve, maximum loadability), the two components move together and merge, and above 0.5 no solution exists at all.

**How it connects**

**X006** mapped where uncertainty comes from. This entry adds one more source that neither label fits well: a non-unique answer. **A04–A05** gave the power-flow equations and Newton–Raphson. The example above is why NR's starting point matters, and **A09**'s remark that the voltage band removes a spurious root is exactly the "pick a branch" step that an MDN makes explicit. Supervised OPF proxies trained with squared error (**X012**, and phase C's **C01–C04**) predict conditional means. They are safe only where the solution map is single-valued, or where averaging solutions stays feasible, as it does for convex problems such as the DC-OPF of the next focus lesson. Phase D picks this up directly: **D08** (heteroscedastic heads and their pitfalls) is the `m = 1` case. **D09–D11** (ensembles and entropy or variance decompositions) use the same law-of-total-variance split with the ensemble member as the mixture label, and ask whether the between-term really means "epistemic". **D05** (latent-variable models) generalises the discrete mixture label to a continuous latent. In phase F, conditional densities like this one give conformal sets that can be unions of intervals rather than one interval. Next focus lesson: **A11** (DC-OPF, congestion and locational marginal prices).

**Check yourself**

1. A network trained with MSE on one-hot class targets outputs well-calibratable class probabilities. The same loss on a multi-valued continuous target outputs something useless. What single fact explains both?
2. In the 2-bus example, the MDN's between-component variance is large. Would collecting ten times more data, or training an ensemble, make it shrink? What does this say about labelling it "epistemic"?
3. Give two reasons why choosing the output by largest mass `π_i` (eq. 49) can be preferable to choosing it by largest peak `π_i/σ_i` (eq. 48).

<details><summary>Answers</summary>

1. The minimiser of expected squared error is the conditional mean `⟨t|x⟩`. For one-hot targets the conditional mean is the vector of posteriors `P(class | x)`, a complete description of a categorical distribution. For a multi-valued continuous target, the conditional mean of a multimodal `p(t|x)` can lie in a region of zero density, so it is neither a valid answer nor a full description.
2. No. The two roots and their 70 / 30 frequency are properties of how the data were generated, so with more data the MDN only estimates `π` and `μ` more precisely. An ensemble of MDNs would agree on both components, so its between-member spread would be small. The between-*component* term measures ambiguity in the data (aleatoric in the sense of X006, given `x`), not model ignorance. A decomposition that equates "between-mixture-label" with "epistemic" is only right when the label is a *model* index, as in an ensemble or posterior samples.
3. (a) With well-separated branches, `π_i` is the probability mass of a branch, which is what "most probable branch" means. The peak height also rewards narrowness. (b) In sparse regions a component can become artificially narrow, which inflates `π_i/σ_i` without any real support. Bishop notes that eq. (49) is less susceptible to this. (Also: `π_i/σ_i^c` depends on the units of `t`, while the mass does not.)

</details>
