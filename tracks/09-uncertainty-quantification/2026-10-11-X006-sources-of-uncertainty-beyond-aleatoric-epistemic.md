## 2026-10-11 (Sunday) — X006 · Track 09: Aleatoric/epistemic UQ and evaluating uncertainty signals — Where uncertainty comes from: a statistician's map beyond "aleatoric vs epistemic"

**Type:** Cross-phase preview
**Main reference:** [Sources of Uncertainty in Supervised Machine Learning – A Statisticians' View — C. Gruber, P. O. Schenk, M. Schierholz, F. Kreuter & G. Kauermann (2023, rev. 2025), arXiv:2305.16703 [stat.ML]](https://arxiv.org/abs/2305.16703). Read Section 1 and Section 3 (definitions, the linear-model example, the bias–variance link, misspecification via KL), then Section 4.2.1–4.2.3 (omitted variables and errors in X). Read Section 5.4.1 (deployment and distribution shift). Skim Sections 5.2–5.3 (total survey error, missing data) and the example in Section 5.4.6. The link goes to the open arXiv version (v3, January 2025). No journal version is listed.
**Further reading:**
- [Aleatory or epistemic? Does it matter? — A. Der Kiureghian & O. Ditlevsen (2009), Structural Safety 31(2):105–112](https://doi.org/10.1016/j.strusafe.2008.06.020): the engineering-reliability classic that the ML split comes from. It defines an uncertainty as epistemic if the modeller sees a way to reduce it, and it warns that epistemic uncertainty creates statistical dependence between events, which goes unnoticed if you model it as independent noise. The publisher version is paywalled; the [DTU record](https://orbit.dtu.dk/en/publications/aleatoric-or-epistemic-does-it-matter/) has the abstract.

**Why this matters** — Anyone who attaches an uncertainty signal to a learned OPF solution, a forecast or a constraint-violation predictor has to say what the signal measures. "Aleatoric vs epistemic" is the usual answer, but in practice the uncertainty comes from missing features, noisy measurements, mislabelled data and shifts between training and deployment. This paper is a short, careful map of those sources. Read it before phase D's methods (ensembles, heteroscedastic heads, decompositions) so that you know what each method can and cannot see.

**Core ideas**

**A precise definition, and what it implies.** Fix the inputs `X` and target `Y`. The true relationship ("nature") is the conditional law `Y | x ~ f_{Y|X}(·|x)`. The paper defines

```
aleatoric uncertainty at x  :=  Var(Y | x)     (entropy of f_{Y|X} for discrete Y)
epistemic uncertainty       :=  everything else
```

and splits the epistemic part into *model* uncertainty (the hypothesis class `P = {p(y|x; θ)}` may not contain `f_{Y|X}`) and *estimation* uncertainty (`θ̂` from finite data differs from the best `θ_0` in `P`). The key consequence: aleatoric uncertainty is defined *relative to the chosen inputs X*. The authors' dice example: with no features, a die roll is purely aleatoric; if you could measure the initial position and every movement, it would be deterministic. So the split is a property of a model and a dataset, not of the world. This echoes Der Kiureghian & Ditlevsen's "reducible *by this modeller*" criterion.

**The bias–variance link.** For squared loss and a mean prediction `ŷ = μ̂(x)` trained on a random dataset `D`:

```
E_{Y|x} E_D[(Y − ŷ)²] = Var(Y|x)  +  Var_D(ŷ | x)  +  bias(ŷ | x)²
                        aleatoric    estimation       model
```

The first term does not depend on the learner at all. The second shrinks with more data. The third shrinks only with a richer class. Even in this simplest case, the authors stress that a *prediction interval* does not add the pieces up. For linear regression, the 1−α interval is `x₀ᵀβ̂ ± t_{n−p,1−α/2} · σ̂ · sqrt(1 + x₀ᵀ(XᵀX)⁻¹x₀)`. The aleatoric term (`1`) and the estimation term (`x₀ᵀ(XᵀX)⁻¹x₀`) sit together under one square root, multiplied by an estimated `σ̂`. "Total = aleatoric + epistemic" is a conceptual statement, not an identity you can read off a model. Section 3.3 adds that under misspecification, maximum likelihood still converges to the KL-closest `θ_0` in `P`, so estimation uncertainty can be reduced even when model uncertainty is large.

**Omitted variables: the paper's central technical point.** Let `Z` be a context variable that affects `Y` but is not recorded. The law of total variance gives

```
Var(Y | x) = E_{Z|x}[ Var(Y | x, Z) ]  +  E_{Z|x}[ bias(x, Z)² ],   bias(x, z) = E[Y|x,z] − E[Y|x]
```

Two lessons follow. (i) Part of what an X-only model calls "irreducible" noise is really an omitted-variable bias, which more *features* (not more samples) would remove. (ii) Appendix B shows that for some `z`, `Var(Y|x,z)` can *exceed* `Var(Y|x)`. So an X-only model can understate uncertainty in some regimes and overstate it in others, even with zero average bias. Prediction intervals then over-cover in one regime and under-cover in another, while looking well calibrated on average. A related twist: a `Z` with no effect on the *mean* of `Y` is never selected by a squared-error loss, yet it can still drive heteroscedasticity (the paper's example is annotators of different reliability).

**This undercuts a common argument.** One argument says model uncertainty can be ignored for universal approximators such as deep nets. The authors reject it: the argument silently assumes that all relevant features are present and measured without error.

**Errors in X, errors in Y, and data production.** If the true feature is `Z` but you observe a noisy `X` (with `Y ⊥ X | Z`), the "error-prone" aleatoric variance becomes `E[Var(Y|Z) | x] + Var(E[Y|Z] | x)`. The second term is inflation caused purely by measurement noise. As the noise grows, the feature behaves like an omitted variable. Section 5 broadens this to *data uncertainty*: how the dataset was produced, framed with the survey-methodology "total survey error" checklist (coverage, sampling, nonresponse, measurement). It also covers *deployment*, where `f^Tr_{Y,X} ≠ f^De_{Y,X}`. The model is *transportable* only if the relevant conditionals are the same in both environments, which is a substantive, domain-knowledge claim rather than something a test split can check.

**A small example**

A wind farm's hourly output `Y` (MW) is predicted from a forecast wind speed `x` alone. There is an unrecorded regime `Z`, say whether the wind comes from a direction where the farm's wake losses are large and erratic, with `P(Z=0|x) = P(Z=1|x) = 0.5`:

```
Z = 0 :  Y | x, Z ~ N(40, 4²)
Z = 1 :  Y | x, Z ~ N(60, 20²)
```

The X-only model sees mean `50` and, by the law of total variance,

```
Var(Y|x) = ½(16 + 400)  +  ½(10² + 10²) = 208 + 100 = 308,   sd ≈ 17.5
```

The 100 MW² from the bias term is "aleatoric" to this model but would vanish if the regime were recorded. Now take the Gaussian 90 % interval the X-only model would report, `50 ± 1.645·17.5 = [21.1, 78.9]`. Its coverage in each regime:

```
Z = 0 :  P(21.1 < N(40, 4²)  < 78.9) ≈ 1.000
Z = 1 :  P(21.1 < N(60, 20²) < 78.9) ≈ 0.801
average                                ≈ 0.901
```

On average the interval looks correct, at 90 %. In the hard regime it covers only 80 %, because `Var(Y|x,Z=1) = 400 > 308`. This is exactly where a downstream dispatch or reserve decision is most likely to fail. A reliability diagram pooled over all hours would not show it.

**How it connects**

**X002** argued that forecasts should be distributions, and **X005** showed how a learner can take a distribution as *input*. This entry asks what a model's *output* uncertainty measures. The omitted-variable term above is the same law-of-total-variance (Jensen-type) gap seen in X005, now viewed as a missing input. The regime example also previews a power-systems fact from **A01–A03**: one unobserved common factor (weather) moves many buses at once. Der Kiureghian & Ditlevsen's warning that epistemic uncertainty induces *dependence* is why errors at different nodes are correlated, which matters for the joint chance constraints in phase B. In phase D, **D07** gives the canonical ML treatment (Hüllermeier & Waegeman), which this paper extends. **D08** covers heteroscedastic heads that try to learn `Var(Y|x)`. **D10–D11** cover entropy-based decompositions and whether they hold up. **D12** covers shift. **D13** and **D17** turn the "fine on average, wrong per regime" problem into regression calibration and subgroup calibration, and **F06** revisits it for conformal coverage. Next focus lesson: **A04**.

**Check yourself**

1. Under the paper's definition, can adding a new input feature *increase* the aleatoric uncertainty at some points? Can it decrease the average?
2. Why is "use a deep network, so model uncertainty is negligible" not enough to make `Var(Y|x)` irreducible in practice?
3. In the example, which single change to the data would make the Z=1 coverage problem visible without changing the model, and which change would fix it?

<details><summary>Answers</summary>

1. Yes to both. Aleatoric uncertainty is `Var(Y | inputs)`, so it changes with the input set. Averaged over `Z`, `E[Var(Y|x,Z)] ≤ Var(Y|x)` by the law of total variance, so the average cannot go up. But for particular values `z`, `Var(Y|x,z)` can be larger than `Var(Y|x)` (in the example, 400 > 308), so the pointwise aleatoric uncertainty rises in that regime.
2. Universal approximation only removes the gap between the class `P` and `f_{Y|X}` for the *observed* `X`. If relevant variables are omitted or mismeasured, then `f_{Y|X}` itself carries omitted-variable bias terms and measurement-noise inflation. These look like irreducible noise to any learner on `X`, however flexible, but more or better features would reduce them.
3. Recording `Z` in the evaluation data (not necessarily in the model's inputs) lets you compute coverage per regime, which shows 80 % vs 100 %. Recording `Z` and feeding it to the model fixes it: the model then predicts `N(40, 4²)` or `N(60, 20²)`, gets regime-specific 90 % intervals, and removes the 100 MW² bias term.

</details>
