## 2026-10-15 (Thursday) — X008 · Track 11: Conformal prediction, risk control, selective prediction — Calibrate the decision, not the prediction: conformal decision theory

**Type:** Cross-phase preview
**Main reference:** [Conformal Decision Theory: Safe Autonomous Decisions from Imperfect Predictions — J. Lekeufack, A. N. Angelopoulos, A. Bajcsy, M. I. Jordan & J. Malik (2023, rev. 2024), arXiv:2310.05921 [stat.ML]](https://arxiv.org/abs/2310.05921) — read Sections I, III and IV (about 4 pages: decision families indexed by a control variable, the conformal controller update, eq. 5, Definition 1 "eventually safe", Theorem 1 and the batch corollary). Then read Section V-A (robot navigation among pedestrians, where the controller is compared with prediction-set calibration). Skim V-B and V-C. The link goes to the open arXiv version (v3, May 2024).
**Further reading:**
- [Conformal PID Control for Time Series Prediction — A. N. Angelopoulos, E. J. Candès & R. J. Tibshirani (2023), NeurIPS 2023](https://arxiv.org/abs/2307.16895) — the same "online update of one scalar" idea applied to prediction intervals, with integral and derivative terms added. It is the closest relative on the prediction-set side.
- [Adaptive Conformal Inference Under Distribution Shift — I. Gibbs & E. J. Candès (2021), NeurIPS 2021](https://arxiv.org/abs/2106.00170) — the online update that the conformal controller generalizes. Read Section 2 to see the same one-line recursion with a miscoverage loss. (The curriculum covers it in depth in F05.)

**Why this matters** — Anyone who puts a learned model in front of an operational decision (a dispatch, a security margin, an accept/reject call on a learned OPF solution) ends up asking how often the decision goes wrong, not how often the prediction interval misses. Conformal decision theory shows that the distribution-free machinery of conformal prediction can calibrate the decision directly. Its guarantee holds even when the data are not exchangeable and the sequence is adversarial. That makes it a good preview of Phase F's question: what can you promise about a learned system's risk with almost no assumptions?

**Core ideas**

**1. The gap between calibrated predictions and safe decisions.** You know split conformal: score a calibration set, take the `⌈(n+1)(1−α)⌉`-th smallest score, and get sets with marginal coverage `≥ 1 − α`. The usual pipeline for decisions is then "predict a set, then act robustly against the whole set". This has two weaknesses. First, coverage is the wrong currency: a miss on a variable that does not affect the decision costs nothing, while a small miss on a critical variable may be catastrophic. Second, the set must cover everything jointly, so it grows with the number of uncertain quantities, including irrelevant ones. The paper's robot example shows this. Adaptive conformal sets widened after a mispredicted pedestrian far away, and the robot froze. The conformal controller only paid attention to errors that actually produced losses.

**2. Setup.** At each time `t` you observe `x_t`. You have a family of decision rules `D_t^λ : X → U`, indexed by one scalar `λ ∈ ℝ` (the "conformal control variable"). Read `λ` as an aggressiveness knob: larger `λ` means a more efficient but riskier action. You act with `u_t = D_t^{λ_t}(x_t)`. Then the outcome `y_t` is revealed and you incur a bounded loss

```
ℓ_t = L(u_t, y_t) ∈ [0, 1],     R̂_T = (1/T) Σ_{t=1..T} ℓ_t
```

The goal is to keep the realized average risk `R̂_T` at or below a target `ε`, up to a term that vanishes as `T` grows. Nothing is assumed about how `(x_t, y_t)` is generated. The decision rule can wrap any predictor, planner or optimizer.

**3. The conformal controller.** The whole algorithm is one line (eq. 5):

```
λ_{t+1} = λ_t + η (ε − ℓ_t),     η > 0
```

If you incurred less loss than the budget, become a bit more aggressive. If you incurred more, back off. This is online gradient descent on the pinball-style objective behind adaptive conformal inference, with the decision loss in place of a miscoverage indicator. If the loss is "the set missed `y_t`" and `λ` shrinks the set, you recover ACI. So prediction-set calibration is the special case where the decision is the set.

**4. Why it works: a telescoping identity.** Summing the update from `1` to `t` gives `λ_{t+1} = λ_1 + η(tε − Σ ℓ_k)`, hence

```
R̂_t = ε + (λ_1 − λ_{t+1}) / (η t)
```

So the risk exceeds `ε` only if `λ` drifts far below where it started. This needs one assumption about the decision family (Definition 1, "eventually safe"). There is a level `λ^safe` such that if `λ_k ≤ λ^safe` for `K` consecutive steps, the average loss over those steps is at most `ε^safe ≤ ε`. In words, a sufficiently conservative fallback exists. Under this assumption `λ_t` can never fall much below `λ^safe`, and Theorem 1 gives, for all `t ≥ K` (with `λ_1 ≥ λ^safe − η`),

```
R̂_t ≤ ε + ( (λ_1 − λ^safe)/η + K ) / t
```

The guarantee is deterministic, holds for every sequence, and the excess decays as `O(1/t)`. No exchangeability, no model of the world, no calibration set.

**5. The batch version.** With exchangeable data and a loss that is monotone in `λ`, the paper also gives an offline rule (Corollary 2): choose `λ̂ = sup{λ : (1/n) Σ_i L_i(λ) ≤ ε − (1−ε)/n}` on a calibration set, and then `E[L_{n+1}(λ̂)] ≤ ε`. This is conformal risk control (lesson F08), and with a miscoverage loss it reduces to the split-conformal quantile you already know.

**6. Caveats worth taking into Phase F.**
- The guarantee is on the *time-averaged realized* loss. It is not a per-step probability, so it is not a chance constraint (Phase B), and it cannot rule out a burst of failures.
- It needs feedback: `y_t` must be revealed after each decision.
- It needs a safe fallback. If even the most conservative `λ` is unsafe, the theorem says nothing.
- `η` trades reaction speed against oscillation. Large `η` reacts fast but makes the decisions jumpy, while small `η` inflates the constant `(λ_1 − λ^safe)/η`. In the drone experiment the controller was safe for `η ∈ {100, 500, 1000}` and unsafe for `η = 50` over the finite horizon.
- One scalar knob calibrates one aggregate loss. Several constraint classes with separate budgets need several controllers or a combined loss.

**A small example**

One transmission line is rated at 100 MW. A learned dispatch rule schedules a flow of `80 + λ` MW on it: `λ` is how much of the remaining headroom you dare to use. The real-time flow is the schedule plus a forecast error `e_t ∈ [−15, 15]` MW. Let the loss be `ℓ_t = 1` if `80 + λ_t + e_t > 100` (an overload that needs corrective redispatch), and 0 otherwise. Take `ε = 0.05`, `η = 20`, `λ_1 = 0`.

- **Update sizes.** Without an overload, `λ` grows by `20 · 0.05 = 1` MW. After an overload, it drops by `20 · 0.95 = 19` MW.
- **Safe level.** With `λ ≤ 5`, the flow is at most `85 + 15 = 100`, so the loss is always zero. Thus `λ^safe = 5`, `K = 1`, `ε^safe = 0`. Also `λ_1 = 0 ≥ 5 − 20` holds.
- **A run.** `λ` climbs 0, 1, …, 10 over steps 1–11. At step 11, `e_11 = 12 > 20 − 10`, so the line overloads, and `λ_12 = 10 − 19 = −9`. Check the identity: `R̂_11 = 1/11 ≈ 0.091`, and `0.05 + (0 − (−9))/(20 · 11) = 0.05 + 0.041 = 0.091`. ✓
- **The bound.** Theorem 1 gives `R̂_t ≤ 0.05 + ((0 − 5)/20 + 1)/t = 0.05 + 0.75/t`. After 100 steps the realized overload rate is at most 5.75 %, and after 1,000 steps at most 5.075 %. This holds whatever the error sequence does, even if it is adversarial.

Now compare this with the set-based route on a grid with 200 lines. You would build a joint 95 % prediction set for 200 flow errors and dispatch robustly against all of them. The set is driven by the noisiest lines, even those that never come close to their rating. The controller only spends caution when an overload actually happens.

**How it connects**

This is the decision-level counterpart of what X002 (Pinson) argued for forecasts: once you have uncertainty, what matters is the decision built on it. In the language of X001 (Powell), the conformal controller is a policy function approximation whose single parameter is tuned online against realized cost. X006's point that uncertainty has many sources is the reason for a guarantee that assumes nothing about them. On the grid side, the "safe fallback" `λ^safe` is the conservative security margin operators already use (A01), and the overload loss refers to the thermal limits that A07 will define properly. Phase B will contrast this with chance constraints, which promise a per-step probability under a distributional model. Phase F builds the full toolkit: split and adaptive conformal (F01, F05), risk-controlling sets and conformal risk control (F07, F08), and Learn-then-Test (F09). This paper sits at the intersection of those lessons. Tomorrow's focus lesson, A06, is the DC power-flow approximation, which is exactly the kind of linear model behind a rule like "flow = schedule + error".

**Check yourself**

1. Derive `R̂_t = ε + (λ_1 − λ_{t+1})/(ηt)` from the update rule, and explain in one sentence why a lower bound on `λ_t` is all you need for a risk guarantee.
2. In the line example, an adversary picks `e_t = +15` at every step. What happens to `λ_t` and to the realized overload rate over time? Is the theorem's bound still respected?
3. Why does the theorem not give a chance constraint of the form `P(overload at step t) ≤ 0.05`, and which assumption would you need to add to get a statement about the next single step?

<details><summary>Answers</summary>

1. Sum `λ_{k+1} − λ_k = η(ε − ℓ_k)` over `k = 1..t` to get `λ_{t+1} − λ_1 = η(tε − Σℓ_k)`. Divide by `ηt` and rearrange to get the identity. The excess risk `R̂_t − ε` equals `(λ_1 − λ_{t+1})/(ηt)`, so if `λ_{t+1}` is bounded below by a constant, the excess is `O(1/t)`. The "eventually safe" assumption provides exactly that lower bound, because whenever `λ` is low the losses stop and `λ` climbs back.
2. An overload happens when `λ_t > 5`. Starting from 0, `λ` climbs by 1 per step and reaches 6 at step 7. That step overloads, and `λ` drops to `6 − 19 = −13`. It then needs 19 overload-free steps to get back above 5. The pattern repeats with one overload every 20 steps, so the realized rate tends to exactly `1/20 = 5 %`. At `t = 7` the rate is `1/7 ≈ 0.143`, and the bound gives `0.05 + 0.75/7 ≈ 0.157`, so the bound holds and is nearly tight early on.
3. The guarantee is about the average of realized losses along one sequence, and it holds for adversarial sequences, where per-step probabilities are not even defined. A statement about the next step needs a probabilistic model linking past and future, such as exchangeability of the calibration data and the test point. Under exchangeability, the batch rule (Corollary 2, conformal risk control) gives `E[L_{n+1}(λ̂)] ≤ ε`. With a 0/1 loss, that is a marginal probability bound on the next step, averaged over calibration draws. It is still not conditional on the current input.

</details>
