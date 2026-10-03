## 2026-10-03 (Saturday) — X001 · Track 04: Stochastic, chance-constrained, risk-averse & robust optimization — One language for decisions under uncertainty: Powell's four policy classes

**Type:** Cross-phase preview
**Main reference:** [Tutorial on Stochastic Optimization in Energy — Part I: Modeling and Policies — W. B. Powell & S. Meisel (2016), IEEE Transactions on Power Systems 31(2):1459–1467](https://doi.org/10.1109/TPWRS.2015.2424974) — read the whole paper (9 pages). Focus on the canonical model and the four policy classes, and skim the notation discussion. ([Princeton record with abstract](https://collaborate.princeton.edu/en/publications/tutorial-on-stochastic-optimization-in-energy-part-i-modeling-and/). The IEEE version is paywalled. The open paper below presents the same framework.)
**Further reading:**
- [From Reinforcement Learning to Optimal Control: A unified framework for sequential decisions — W. B. Powell (2019/2021), arXiv:1912.03513](https://arxiv.org/abs/1912.03513) — open-access, longer version of the same framework. Sections 4 and 6 hold the five model elements and the four policy classes, written for an RL audience.
- [Tutorial on Stochastic Optimization in Energy — Part II: An Energy Storage Illustration — Powell & Meisel (2016), IEEE TPWRS 31(2):1468–1475](https://doi.org/10.1109/TPWRS.2015.2424980) — five variants of one storage problem, built so that each policy class wins on at least one.

**Why this matters** — Power-system operation under uncertainty is studied by several communities: stochastic programming, robust optimization, model predictive control, dynamic programming and reinforcement learning. Each has its own notation, and each believes its method is the natural one. Anyone who builds learned dispatch or OPF solvers under forecast uncertainty has to place their model in this landscape. Is it a policy, a value function, or a fast surrogate inside a lookahead? Without that map, comparisons between methods are mostly accidents of notation.

**Core ideas**

The tutorial's main claim is methodological: *model first, then solve*. You first write down the sequential decision problem without saying how you will solve it. Only then do you choose a class of policies. The model has five elements:

1. **State** `S_t`: everything needed, from now on, to compute costs, constraints and the next state. Powell splits it into a physical state `R_t` (for example a battery's state of charge or generator outputs), other information `I_t` (prices, weather, the current forecast) and beliefs `B_t` (parameters of distributions over quantities you cannot observe). An ML reader should note that *the forecast itself is part of the state*. A model that conditions on a forecast distribution is conditioning on part of `S_t`.
2. **Decision** `x_t`, chosen by a policy `X^π(S_t)`. In energy problems `x_t` is often a high-dimensional continuous vector with constraints, such as setpoints for hundreds of generators. That is why the RL framing in terms of "action spaces" fits poorly and the optimization framing is the default.
3. **Exogenous information** `W_{t+1}`: what you learn between `t` and `t+1`, for example realized wind, load or prices.
4. **Transition** `S_{t+1} = S^M(S_t, x_t, W_{t+1})`. For a battery: `SoC_{t+1} = SoC_t + η·charge_t − discharge_t/η`. In RL this is the environment step, and in control the plant equation.
5. **Objective**: search over policies, not over decisions:

```
max_π  E[ Σ_{t=0..T} C_t(S_t, X^π_t(S_t)) | S_0 ],   S_{t+1} = S^M(S_t, X^π_t(S_t), W_{t+1})
```

This is the RL objective written in operations-research notation. The difference is emphasis. In OR, `C_t` and `S^M` are usually known and structured (linear constraints, power-flow physics), and decisions are vectors that a solver must produce.

**The four policy classes.** The tutorial argues that every practical method falls into one of four classes, or a hybrid of them:

- **PFA (policy function approximation):** a direct map from state to decision, `x_t = f(S_t | θ)`. Examples are an affine rule (`x_t = θ_0 + θ_1·φ(S_t)`), a "charge if price < θ" threshold, or a neural network. Affine balancing rules in power systems are PFAs, as is a learned OPF proxy evaluated without any solver.
- **CFA (cost function approximation):** solve a *deterministic* optimization whose objective or constraints are tuned by `θ`: `x_t = argmax_{x ∈ X_t(θ)} C̄(S_t, x | θ)`. This is what industry mostly does. Examples are reserve margins, forecast inflation, and tightened line limits. Uncertainty enters only through tuned buffers.
- **VFA (value function approximation):** `x_t = argmax_x [ C(S_t, x) + E V̄_{t+1}(S_{t+1}) ]`. This covers Bellman methods, approximate dynamic programming, Q-learning and stochastic dual dynamic programming.
- **DLA (direct lookahead approximation):** at each step, optimize over an *approximate model of the future*, then implement only the first decision. The lookahead can be deterministic (model predictive control with point forecasts) or stochastic (a scenario tree, as in two-stage stochastic programming), and it can be robust (worst case over an uncertainty set).

PFAs and CFAs are found by **policy search**: tune `θ` by simulating the base model. VFAs and DLAs are built by **approximating the lookahead**. The exact lookahead is the Bellman-optimal decision:

```
X*_t(S_t) = argmax_x ( C(S_t, x) + E[ max_π E[ Σ_{t'>t} C(S_{t'}, X^π_{t'}(S_{t'})) | S_{t+1} ] | S_t, x ] )
```

This exact form is almost never computable. A VFA replaces the whole inner term with `V̄`. A DLA replaces the *model* inside it with something smaller: fewer scenarios, a shorter horizon, or a deterministic forecast.

**Two key messages.** First, no class dominates. Part II builds variants of one storage problem and shows that, depending on the problem's characteristics, each class can be the best. Powell's general rule of thumb: PFAs when the policy's structure is obvious, DLAs when forecasts carry strong time-varying information, VFAs when the state is low-dimensional and the problem is stationary. Second, *a deterministic lookahead model is not a deterministic policy*. A rolling MPC with a point forecast is a legitimate stochastic policy, and its quality must be measured by simulating it against the real uncertainty, i.e. evaluating the base objective, not the lookahead objective it optimizes internally. This gap between in-model and out-of-sample performance comes back many times in this curriculum.

**A small example**

A day-ahead commitment, written as a one-step problem. You commit `x` MWh of generation today at 30 €/MWh. Tomorrow demand `D` turns out to be 80, 100 or 120 MWh, each with probability 1/3. A shortfall is covered in real time at 100 €/MWh, and surplus is wasted. Expected cost:

```
J(x) = 30·x + 100·E[max(D − x, 0)]
J(80)  = 2400 + 100·(0 + 20 + 40)/3 = 4400
J(100) = 3000 + 100·(0 + 0 + 20)/3  ≈ 3667
J(120) = 3600 + 0                   = 3600
```

- **Deterministic lookahead on the mean forecast** (plan for `D = 100`) gives `x = 100` and an expected cost of 3667. Inside its own model it expects to pay only 3000.
- **Stochastic lookahead** (minimize `J` over the scenarios) gives `x = 120` and a cost of 3600. This is the newsvendor critical-ratio rule: keep adding capacity while `P(D > x)·100 > 30`, i.e. while `P(D > x) > 0.3`.
- **CFA**: plan for `mean + θ` and tune `θ` by simulation. `θ = 20` recovers the stochastic optimum with a deterministic solver. That is why reserve margins work in practice, and why the tuning has to be redone when the forecast error distribution changes.
- **PFA**: `x = θ·forecast` with `θ = 1.2` does the same here without solving any optimization problem.

Three policies from three classes reach the same decision. The difference shows up when the uncertainty changes from day to day (heteroskedastic forecasts). A fixed buffer `θ` then over- or under-reserves, while a policy that conditions on the forecast *distribution* can adapt.

**How it connects**

This is the first entry in the log, so there are no earlier entries to build on. It previews the phase B unit on decision-making under uncertainty. **B01** (the taxonomy of stochastic, chance-constrained, robust and distributionally robust optimization) and **B02** (two-stage stochastic programming, recourse, SAA) are mostly about building good **DLAs**. **B08**'s affine balancing rules are **PFAs** embedded inside a lookahead. **A29** (receding-horizon MPC) is the textbook deterministic DLA. Read Powell's frame as the "type system" for all of these. Phase C's learned OPF proxies (**C01–C03**) are, in this language, either PFAs (the network *is* the policy) or fast components inside a DLA. **F18**'s out-of-sample evaluation of stochastic decisions is the base-model simulation Powell insists on. Next in the focus phase: **A01**, why grids are hard to operate.

**Check yourself**

1. Why does Powell put the current forecast (and its uncertainty) inside the state `S_t` rather than treating it as a fixed parameter of the problem?
2. A grid operator solves a deterministic DC-OPF every 15 minutes, using the latest point forecast and line limits tightened by 5 %. Which policy class is this, and what is `θ`?
3. In the example, the mean-forecast plan "expects" to pay 3000 but really pays about 3667 in expectation. Which two objectives are being confused, and how should you evaluate such a policy?

<details><summary>Answers</summary>

1. The state must contain everything needed to make decisions and compute what happens next. The forecast is information available at time `t` that changes over time and affects the best decision, so it belongs in `I_t` (and its spread in `B_t`). If you treat it as fixed, the "policy" cannot respond to a day with tight versus wide uncertainty, which is exactly where uncertainty-aware decisions add value.
2. It is a deterministic **DLA** (rolling horizon on a point forecast) whose constraints are parametrically modified. That makes it a **CFA/DLA hybrid**, with `θ` the 5 % tightening (plus any reserve margins). `θ` should be tuned by simulating the policy against realistic forecast errors.
3. The lookahead model's objective (cost under the assumed deterministic future) is being confused with the base model's objective (expected cost under the true distribution of `W`). A policy must be evaluated on the base model: simulate it on many sampled or historical outcomes not used to build it, and average the realized cost and constraint violations.

</details>
