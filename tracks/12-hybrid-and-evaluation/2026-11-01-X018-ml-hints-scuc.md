## 2026-11-01 (Sunday) — X018 · Track 12: Hybrid ML–solver pipelines, warm starts, benchmarking — Keep the guarantee, reuse yesterday's answers: ML hints for security-constrained unit commitment

**Type:** Cross-phase preview
**Main reference:** [Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems — Á. S. Xavier, F. Qiu & S. Ahmed (2021), INFORMS Journal on Computing 33(2):739–756](https://arxiv.org/abs/1902.01697) — read Sections 1, 2.1 and 3 (the setting, Algorithm 1 and the three predictors, about 8 pages). Then read Section 4 through its tables, mainly Tables 3, 5, 7 and 9, and Section 4.6 (out-of-distribution test). Skip Section 2.2 (kNN/SVM basics) and Appendix A (full formulation). The link goes to the open arXiv version (v2, December 2019). The journal DOI is 10.1287/ijoc.2020.0976.
**Further reading:**
- [Implied Constraint Satisfaction in Power System Optimization: The Impacts of Load Variations — L. Roald & D. K. Molzahn (2019), arXiv:1904.01757](https://arxiv.org/abs/1904.01757) — the optimization-side explanation of why the first predictor works so well: for DC models, most line limits stay redundant over a wide range of loads, and this can be proved by solving small screening problems.
- [MIPLearn — Á. S. Xavier, F. Qiu, X. Gu, B. Becu & S. S. Dey (Argonne National Laboratory), open-source framework, BSD licence](https://github.com/ANL-CEEESA/MIPLearn) — the paper's ideas packaged as a general library: learn from solved MIP instances and pass constraint, warm-start and variable-fixing hints to Gurobi, CPLEX or XPRESS.

**Why this matters** — Grid operators solve almost the same large mixed-integer program every day, under tight deadlines, and the data barely changes between solves. Anyone building ML-accelerated grid optimization needs a clear picture of the "hint" design: the network never outputs the final decision, it only tells an exact solver where to look. This paper is a clean early example, and it measures carefully which hints keep the solver's optimality certificate and which ones give it up.

**Core ideas**

*The problem: security-constrained unit commitment (SCUC).* A12 introduced security-constrained OPF for one snapshot. Unit commitment (the topic of A30) adds a day-long horizon `T = {1..24}` and on/off decisions. Let `x_gt ∈ {0,1}` say whether generator `g` is running in hour `t`, and let `y_gt ≥ 0` be its output. Simplified (the paper's eqs. 1–7):

```
min   Σ_g c_g(x_g•, y_g•)                                   startup + production cost
s.t.  (x_g•, y_g•) ∈ G_g                                    min up/down times, ramps, limits
      Σ_g y_gt = Σ_b d_bt                         ∀ t       power balance
      −F_l^c ≤ Σ_b δ_lb^c (Σ_{g∈G_b} y_gt − d_bt) ≤ F_l^c   ∀ l, c ∈ L∪{0}, t
```

The last line is the DC line-flow limit of A06 and A11. `δ_lb^c` are injection shift factors (the PTDFs of A06), computed for the intact network (`c = 0`) and for each single-line outage `c` (the LODF updates of A06, giving the N−1 criterion of A12). Once every `x_gt` is fixed, what remains is an LP, economic dispatch over time (A08), which is easy. The difficulty is in two places. The binaries make the problem combinatorial. And the flow constraints number roughly `|L|² · |T|`: on a 9,000-line grid that is far too many to write down, and they are dense.

*The baseline: lazy constraint generation.* Practitioners do not write all of them. Algorithm 1 solves the MIP with no flow limits, checks every `(l, c, t)` for violations, adds the worst few (at most 15 per hour), and re-solves until nothing is violated. This is exact, because the final solution satisfies every constraint. But each round is a full MIP solve, and the baseline needs about 5 rounds on average.

*The setting.* Each instance `I(p)` is determined by a parameter vector `p` (cost multipliers, load distribution across buses, peak load, hourly profile). The topology is fixed and known. On the day before, the operator solves a few hundred sampled or historical instances. At clearing time a new `p̃` arrives and an ML predictor maps it to **hints** for a customized solver. The total time reported includes the time to compute the hints. In ML terms this is amortization over a fixed distribution (C01 will make that framing formal), but the output is a set of hints, not a solution.

*Predictor 1: which flow constraints to add up front.* Record the set `L_i` of constraints that Algorithm 1 ended up adding for each training instance. For a test `p̃`, find its `k` nearest training instances and add constraint `(l,c,t)` in the first round if it appears in at least 10% of their sets. The threshold is deliberately low because the two errors cost very different amounts. A false positive makes the LP slightly larger. A false negative costs one more MIP round. Since Algorithm 1 still checks for violations at the end, **a wrong prediction can only cost time, never correctness**. Result (Table 3): with `k = 300`, i.e. just counting how often each constraint was needed, the number of rounds falls from 4.92 to 1.02 and time from 227 s to 80 s (2.8×). That equals an oracle that knows the exact set (80.9 s). The small difference in favour of kNN is ordinary MIP run-to-run variability, as the authors point out. This is a useful warning for any benchmark.

*Predictor 2: partial warm starts.* Take the commitments `x^i` of the `k = 50` nearest neighbours and average them per variable, giving `x̄_gt`. Then set

```
x_gt := 1  if x̄_gt > p,     x_gt := 0  if x̄_gt ≤ 1 − p,     otherwise leave x_gt free
```

and give this partial assignment to CPLEX in "repair" mode, so the solver fills in the free variables. The key result (Table 5) is the role of the consensus threshold. Majority vote (`p = 0.5`) assigns every binary, and *none* of those assignments was feasible or repairable. With `p = 0.9`, a warm start was valid for 100% of instances, the repaired warm start turned out to be optimal in every case, and the speed-up reached 4.3×, close to the oracle's 4.9×. The paper's reading: ML proposes only what it is confident about, and the solver does the rest. This is the same calibration logic as selective prediction (F10), applied per variable.

*Predictor 3: affine subspaces (gives up the guarantee).* For each binary, predict whether the optimum satisfies `x_gt = 0`, `x_gt = 1` or `x_gt = x_g,t+1`, and add those equalities as hard constraints. Each candidate hyperplane has its own linear SVM, trained on a few features (peak load, hourly loads, own and average others' cost). The SVM is used only if it passes a cross-validated filter: recall `≥ α_R`, and

```
precision ≥ max(z̄, 1 − z̄) · (1 − α_P) + α_P
```

where `z̄` is how often the hyperplane held in training. So the classifier must clearly beat the constant predictor. Hyperplanes that hold almost always (`z̄ ≥ z_FIX`) are added unconditionally. Hyperplanes whose labels are too imbalanced to learn from are skipped. The recommended setting fixes about 94% of commitment variables, gives a 10.2× speed-up, and in the in-distribution tests the solution was never infeasible and never worse than the 0.1% target gap. Without the filter (`aff:C`), 76% of test instances became **infeasible**. An oracle that fixes everything correctly would reach 61×, so there is a lot of room left between the two.

*Shift.* Section 4.6 shifts the test distribution (cost multipliers now centred at +5% instead of 0, peak load +3%, Gaussian instead of uniform). The speed-ups drop to 2.5×, 3.6× and 8.6×. Predictors 1–2 keep the guarantee under any shift, and only their speed-up suffers. Predictor 3 can return a wrong or infeasible answer if the shift is large enough. This three-way split (exact, exact-but-maybe-slower, heuristic) is the main framework to take into phase F.

**A small example**

Warm start for one generator at three hours, with `k = 10` neighbours and threshold `p = 0.9`:

| hour | neighbours with `x = 1` | `x̄` | hint |
|---|---|---|---|
| 07 | 10/10 | 1.0 | fix to 1 |
| 08 | 9/10 | 0.9 | free (`0.9 > 0.9` is false) |
| 09 | 0/10 | 0.0 | fix to 0 |

With majority vote, hour 08 would be fixed to 1. Do that across thousands of variables, each with a small chance of being wrong, and the combined assignment usually violates some coupling constraint, such as a minimum up time or a ramp. Independent per-variable votes do not respect constraints that couple many variables. Leaving the uncertain ones free hands exactly those couplings back to the solver.

Filter check: suppose `x_gt = 1` held in `z̄ = 0.6` of training cases, with `α_P = 0.75` (the paper's value for this hyperplane type). The required precision is `0.6 · 0.25 + 0.75 = 0.90`. A classifier with 85% cross-validated precision is rejected and the variable stays free.

**How it connects**

- It sits directly on A12 (N−1 security and constraint generation) and A06/A11 (shift factors, DC flow limits). Once the binaries are fixed, the remaining LP is A08's economic dispatch over 24 hours.
- Predictor 1 is the MIP-and-contingency version of X003 (Misra et al.'s active-set learning for OPF), and the Roald–Molzahn link explains why constraint sets are so stable.
- Like X009 (Gasse et al., GNN branching), it keeps an exact solver in the loop. But where X009 learns a decision *inside* branch-and-bound, this paper changes the problem before the solve (which constraints are present, which incumbent is available, which variables are fixed), using kNN and SVMs and no deep learning.
- Its "never infeasible in our tests" is empirical. X012 (optimization learning) and F12 ask how to turn that into a bound on the probability of failure.
- Next: Monday's focus lesson A13 (KKT conditions and duality) supplies the dual view these screening methods rest on. A30 covers unit-commitment formulations properly. In phase F, F14 (warm starts), F16 (honest speed-up accounting, including hint time and fallbacks) and F19 (metrics) return to exactly the questions raised by Tables 3–10.

**Check yourself**

1. Why does predictor 1 use a 10% threshold instead of majority vote, and why can a bad prediction never change the final answer?
2. Majority-vote warm starts were never feasible, yet each per-variable vote is usually right. Explain the paradox.
3. Which of the three predictors is still guaranteed to return a 0.1%-optimal solution under severe distribution shift, and what is the worst that happens to the others?

<details><summary>Answers</summary>

1. A false positive only adds a redundant row to the LP, while a false negative costs another full MIP round. That asymmetry favours recall, hence the low threshold. Algorithm 1 still checks every flow constraint after each solve and adds any violated ones, so the returned solution is feasible for the full problem whatever the hints were. Hints change only the starting set.
2. Feasibility is a property of the joint assignment. With ~10⁴ binaries, even a 0.1% error rate per variable gives about ten wrong commitments, and one is enough to break a min-up/down or ramp constraint that couples hours. Independent votes ignore these couplings. Leaving low-consensus variables free (`p = 0.9`) lets the solver resolve them.
3. Predictors 1 and 2 (constraint hints and partial warm starts): the solver still proves the 0.1% gap, so shift costs only speed (2.8×→2.5×, 4.3×→3.6×). Predictor 3 adds hard equalities, so a shift can make the restricted problem infeasible or cut off the optimum. It needs a detect-and-fallback step, such as re-solving without the fixings.

</details>
