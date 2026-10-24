## 2026-10-24 (Saturday) — X013 · Track 07: GNNs and spatio-temporal GNNs for power grids — Does the grid graph help a learned OPF? A controlled test of locality

**Type:** Cross-phase preview
**Main reference:** [Leveraging Power Grid Topology in Machine Learning Assisted Optimal Power Flow — T. Falconer & L. Mones (2023), IEEE Transactions on Power Systems 38(3):2234–2246, doi:10.1109/TPWRS.2022.3187218](https://arxiv.org/abs/2110.00306). Read Section I-A (the two ways ML is used in OPF), Section II-D (how the grid becomes a graph and which architectures are compared), Section III-B and III-C (fixed vs varying topology) and Section III-D with Fig. 5 (the locality analysis). Skim Section II-B/C (the "estimator of an OPF operator" framing). The link goes to the open arXiv preprint (v3, April 2022).
**Further reading:**
- [Topology-aware Graph Neural Networks for Learning Feasible and Adaptive AC-OPF Solutions — S. Liu, C. Wu & H. Zhu (2023), IEEE TPWRS](https://arxiv.org/abs/2205.10129): the counterpoint. It argues that OPF solutions *do* have exploitable graph structure (through the LMPs) and builds a GNN around it. Read it after this entry and decide which locality claim you believe.

**Why this matters** — The default assumption in learned OPF is "the grid is a graph, therefore use a GNN". That is an inductive-bias claim, and like any inductive-bias claim it can be tested. This paper tests it in a controlled way: same data, matched parameter counts, many grid sizes. The answer is uncomfortable and useful: on a fixed grid, the graph barely helps, because the OPF solution map is not local. Anyone designing a GNN for OPF should know this result before choosing an architecture.

**Core ideas**

**The problem being learned.** Optimal power flow (**A09**) chooses generator outputs and voltage set-points that serve all loads at minimum cost while respecting the power-flow physics (**A04**) and the operating limits (**A07**). For a fixed grid, the only thing that changes from one hour to the next is the load vector `d = (P_d, Q_d)`. So the solver defines a function

```
F : d  ↦  y*(d) = argmin_y  cost(y)   s.t.  power-flow equations(y, d),  limits(y)
```

where `y` collects generator active powers `P_g` and voltage magnitudes `|V_g|`. The paper treats every ML-for-OPF method as an estimator of `F` (or of a related operator). Two tasks are studied:

1. **Regression (end-to-end or warm start):** predict `y*` directly, trained with MSE. The prediction can be used as is, or handed to the interior-point solver as a starting point (**A18** will explain why that is harder than it sounds).
2. **Classification (reduced problem):** predict which inequality constraints are *active* (binding) at the optimum, a multi-label problem trained with binary cross-entropy. Solve a smaller OPF with only those constraints, then check the full problem and add back any violated ones. This is the active-set idea from **X003**.

**The architectures.** All models see the same inputs and are trained on the same data.
- *FCNN* (MLP): the concatenated load vector, `x ∈ ℝ^(2|L|)`, one or three hidden layers.
- *CNN*: the inputs reshaped into a 2-D tensor and convolved in Euclidean space, a deliberately "wrong" inductive bias as a control.
- *GNN*: five convolution types from PyTorch Geometric (ChebConv with `K = 4`, GCNConv, GraphConv, GATConv, SplineConv). In the graph, each generator and each load is its own node, and devices at the same bus are connected. Edges can be weighted by the line admittance, `e_ij = exp(−k log|Z_ij|) = |Z_ij|^(−k)`, with `k = 0` giving a plain adjacency matrix.
- Each GNN comes in a *local* version (graph convolutions only, every output depends only on a few-hop neighbourhood) and a *global* version (convolutions followed by a fully connected *readout* over all nodes).

Channel widths are chosen so that the models have similar parameter counts. Data: 11 PGLib grids (**A24**) from 24 to 2853 buses; 10 000 feasible samples per grid, each load scaled independently by a factor from `U(0.8, 1.2)`; labels from PowerModels.jl + Ipopt; results are means with 95 % confidence intervals over 10 training runs.

**Finding 1: on a fixed grid, the MLP is as good or better.** For set-point regression, the plain FCNN is close to the best GNN on small grids and beats all GNNs on the two largest. On 2853-sdet, the three-layer FCNN reaches a test MSE of 5.99 ± 0.16 (×10⁻³) against 8.49 ± 0.41 for the best GNN. For active-set classification the authors call the single-layer FCNN "even more dominant". The FCNN also trains at least an order of magnitude faster than the global GNNs.

**Finding 2: local GNNs fail; the readout does the work.** Removing the readout makes every GNN much worse. Adding it back restores performance. If the answer were local, a few rounds of message passing would be enough and the readout would add little. The authors read this as direct evidence that each generator's optimal output depends on loads all over the grid.

**Finding 3: the locality probe.** Section III-D measures locality directly. Starting from 100 load configurations, they perturb one load by 1 %, re-solve, and record the relative change in every generator's active power, voltage set-point and every bus's LMP, plotted against the *hop distance* between the perturbed load and the output (Fig. 5). There is "little evidence" that nearby outputs react more than distant ones. In large grids, generators also tend to sit far from loads, so a local neighbourhood often contains no generator at all.

**Why the map is not local: the power balance.** This is the first-principles core, and it comes straight from **A08**. Total generation must match total load. In an uncongested, lossless network, economic dispatch sets every generator's marginal cost equal to one system price `λ`, so the optimal response to an extra megawatt depends on *how much* load there is, not *where* it is. Mathematically, `∂P_g*/∂d_j` is the same for every load bus `j`. That is the opposite of a local function. Congestion (**A11**) breaks the uniformity, but the new sensitivities come from PTDFs (**A06**), which are dense: a line limit couples buses that are many hops apart. In ML terms, the target behaves like a function of a *global sum* of the inputs, and a depth-3 message-passing network has a 3-hop receptive field. It cannot compute a global sum on a 300-bus graph without a pooling/readout layer, and once you add one, you have effectively rebuilt an MLP.

**Finding 4: when the topology changes, GNNs win clearly.** In the second experiment, one random transmission line is removed per sample (an N−1 contingency, **A12**). Now the ChebConv GNN is the best model on 7 of 9 grids for regression and 8 of 9 for classification; for example, on 24-ieee-rts the classification loss is 1.44 ± 0.11 (×10⁻²) for ChebConv versus 3.56 ± 0.36 for the three-layer FCNN. One caveat when you read the tables: as described, the FCNN input is the load vector only, so it is not told which line is out, while the GNN sees the changed graph. Part of the gap is therefore an *information* gap, not purely an inductive-bias gap. The honest summary: the graph matters as an **input encoding of topology**, not as a locality prior.

**What the paper does not measure.** Feasibility and optimality of the predictions are left out (the authors say so), and the warm-start variant is described but not timed. Neither the physics violations nor the solver speed-up, which **F16** and **F19** will insist on, are reported.

**A small example**

Two generators on a chain of buses 1–2–…–10: G1 at bus 1, G2 at bus 10, lossless and uncongested. Costs `C1 = 10 P1 + 0.05 P1²` and `C2 = 12 P2 + 0.05 P2²` (€/h, `P` in MW). Equal incremental cost (**A08**):

```
10 + 0.1 P1 = 12 + 0.1 P2 = λ,     P1 + P2 = D
```

With total load `D = 100 MW`: `P1 = 60`, `P2 = 40`, `λ = 16 €/MWh`.

Now add 1 MW of load. Whether it appears at bus 2 (next to G1) or at bus 9 (next to G2, seven hops from G1), the answer is the same: `P1 = 60.5`, `P2 = 40.5`, `λ = 16.05`. The sensitivity matrix `∂P_g/∂d_j` has identical columns. A 2-layer GNN at G1 never sees bus 9 and therefore cannot respond to it; an MLP that only needs `Σ_j d_j` learns this map with a single hidden unit.

Now give the line 3–4 a 30 MW limit and put most of the load beyond it. Once the line saturates, G1 can no longer push power past it: extra load at buses 4–10 must come from G2 alone, while extra load at bus 2 still comes from G1. The two columns of `∂P_g/∂d_j` now differ, depending on which side of the congested line each load sits. The map becomes piecewise linear with a *topology-dependent* switch. Remove a line, and the switch moves. That second picture is where encoding the graph pays off.

**How it connects**

This entry builds on **A08** (one system price explains why the solution map is global), **A09** (the OPF problem being learned) and **X003** (the active-set classification task, here used as the second benchmark). It is the natural sequel to **X004**: there, a GNN beat an MLP on *power flow* with randomized line parameters, and failed on unseen grids. Here, on *OPF* with fixed parameters, the MLP is enough, and the GNN's advantage appears only when the topology varies. Both papers point to the same lesson: the graph is valuable when the structure changes. Coming up in phase A, **A06**'s PTDFs and **A11**'s congestion pricing make the "dense sensitivity" argument exact, and **A12** introduces the N−1 setting where GNNs did win. In phase C, **C14** (over-squashing and long-range limits) gives the theory behind "local GNNs fail", **C17** returns to GNN OPF proxies and topology transfer (including the Liu, Wu & Zhu counterpoint), and **F20** asks for exactly this kind of ablation. Next focus lesson: **A10**.

**Check yourself**

1. Why does removing the readout layer hurt the GNNs so much, and what does that say about the OPF solution map?
2. In a lossless, uncongested grid, what is `∂P_g*/∂d_j` as a function of the load bus `j`, and why does that make a hop-distance locality prior useless?
3. In the N−1 experiment the GNN beats the MLP by a large margin. Give one reason this is not purely evidence for the graph *inductive bias*, and describe a fairer MLP baseline.

<details><summary>Answers</summary>

1. Without a readout, each generator's prediction is a function of the loads within a few hops only. The optimal set-points depend on loads across the whole grid (through the global power balance and dense congestion sensitivities), so a local model is missing most of the relevant inputs. The readout restores global access, which shows that the map is not local; the useful computation is global, much as in an MLP.
2. It is the same for every `j`: equal incremental cost fixes one system price, and the extra megawatt is shared among generators according to their cost curvature, regardless of where it appears. So the output responds to the *sum* of the loads, and nearby loads matter no more than distant ones. A prior that says "nearby inputs matter more" points at the wrong structure.
3. As described, the MLP receives only the load vector, so it is not told which line was removed. It is missing information that the GNN gets from the graph. A fairer baseline appends a line-status vector (one binary entry per branch) or the flattened admittance entries to the MLP input. Then any remaining gap can be attributed to the architecture (weight sharing, permutation structure) rather than to the input.

</details>
