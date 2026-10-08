## 2026-10-08 (Thursday) — X004 · Track 07: GNNs and spatio-temporal GNNs for power grids — Turning the grid inside out: a line-graph GNN for power flow

**Type:** Cross-phase preview
**Main reference:** [Power Flow Balancing with Decentralized Graph Neural Networks — J. B. Hansen, S. N. Anfinsen & F. M. Bianchi (2022), IEEE Transactions on Power Systems, doi:10.1109/TPWRS.2022.3195301](https://arxiv.org/abs/2111.02169). Read Section III (the line-graph representation and the GNN design) and Section V (the three experiments, Tables II–IV and Fig. 7). Skim Section II (background) and Section IV (related work). The link goes to the open arXiv preprint (v2, 2022). An open-access accepted manuscript is also in the [UiT repository](https://munin.uit.no/handle/10037/27777).
**Further reading:**
- [Graph Neural Networks with Convolutional ARMA Filters — F. M. Bianchi, D. Grattarola, L. Livi & C. Alippi (2022), IEEE TPAMI, doi:10.1109/TPAMI.2021.3054830](https://arxiv.org/abs/1901.01343): the layer that makes the 40-hop model trainable. Read it for the rational-filter view of why ARMA layers can produce sharp, high-frequency outputs that GCN stacks smooth away.

**Why this matters** — A power grid is a graph, and the physics is local: what flows through a line depends directly only on the two buses at its ends. That makes GNNs the obvious architecture for learned power-flow and OPF solvers. It also raises design questions that a generic GNN paper does not answer: what should be a node, where should the line parameters go, and how deep must the network be when an effect at one end of the grid moves flows at the other end? This paper is a clear, honest first answer. It also reports where the approach fails (unseen grids), which makes it a good preview of phase C.

**Core ideas**

**The task, from first principles.** *Power flow* (lessons **A04–A05**) is the grid's forward problem. Given what every load consumes and what every generator produces, find the steady state of the network: the complex voltage `V_i = |V_i|∠θ_i` at each bus and the power on every line. The state solves a system of nonlinear equations, one active-power and one reactive-power balance per bus:

```
P_i = Σ_j |V_i||V_j| (G_ij cos θ_ij + B_ij sin θ_ij)
Q_i = Σ_j |V_i||V_j| (G_ij sin θ_ij − B_ij cos θ_ij)
```

Here `θ_ij = θ_i − θ_j`, and `G_ij + jB_ij` is an entry of the admittance matrix, which **A03** introduces as a weighted graph Laplacian. Classical solvers use Newton–Raphson. It is accurate but needs a factorization of a Jacobian at each iteration, and it can fail to converge. A learned surrogate maps grid inputs straight to the solution in one forward pass. The paper trains a GNN by supervised regression on Newton–Raphson solutions (MATPOWER).

**Design choice 1: branches as nodes.** The natural graph has buses as nodes and lines as edges. The authors note two problems with it. First, line parameters (resistance `r`, reactance `x`, charging susceptance `b`, transformer ratio `τ` and phase shift `θ_shift`) are *edge* features, and most cheap GNN layers handle only node features. Edge-conditioned convolutions are costly, and in a very deep network an edge-aware first layer has little influence on the output. Second, some bus outputs are fixed by the bus type. A PV (generator) bus has a set voltage magnitude, so a uniform local update has to be patched for those nodes.

Their fix is the *line graph* `L(G)`. Every branch becomes a node, and two nodes are adjacent when their branches share a bus. Each node then carries everything relevant to its branch in a 21-dimensional vector:

```
x_i = [ r, x, b, τ, θ_shift          (branch, 5)
      | P_d, Q_d, G_s, B_s, P_g, |V_g|, i_slack(2)   (from-bus, 8)
      | same 8 features                (to-bus, 8) ]
```

The from-bus block always comes first, so the branch direction is encoded in the feature order, while the graph itself stays undirected so that information can diffuse everywhere. The *target* changes as well. Instead of bus voltages, the model predicts 8 numbers per branch: the real and imaginary parts of the complex power and the current at both ends. Branch flows are natural line-graph targets. A bus voltage would appear on several line-graph nodes and would need a consistency constraint.

**Design choice 2: very deep, without over-smoothing.** A power-flow solution is *not* smooth on the graph: two neighbouring lines can carry very different flows. A stack of GCN layers is a low-pass filter, so after many layers neighbouring outputs become alike (over-smoothing, lesson **C14**). The model uses ARMA layers. Each has `K` parallel stacks of `T` recursive propagation steps with a skip connection to the input:

```
X̄_k^(t) = σ( Ã X̄_k^(t−1) W_k + X V_k ),   t = 1..T,   X̄ = (1/K) Σ_k X̄_k^(T)
```

Here `Ã = D^(−1/2) A D^(−1/2)`. The skip term `X V_k` re-injects the node's own raw features at every step, which keeps outputs sharp. The final network has 2 dense pre-processing layers, 5 ARMA layers with `T = 8` (a receptive field of 40 hops), 2 dense post-processing layers and a linear 8-unit head, all 64 units wide. It is trained with plain MSE. All quantities are in per unit (**A02**), so different grids have similar value ranges.

**Baselines.** *DCPF* is the linear DC approximation (**A06**), computed without learning. *Local MLP* sees only one branch's 21 features. *Global MLP* concatenates the whole grid into one vector. *GCN* is a 40-layer GCN with the same receptive field. The metric is the RMSE per output normalized by that output's standard deviation, averaged over outputs, so `NRMSE = 1` is about as good as predicting the mean.

**Findings.**
- *Fixed topology* (15 000 samples per grid; loads ±50 %, line parameters ±10 %, transformer settings and generator setpoints resampled). On case118, the NRMSE was ARMA 0.057, Global MLP 0.134, Local MLP 0.447, DCPF 0.674 and GCN 0.887. GCN was worse than the physics-free DC approximation, and Fig. 7 shows why: its outputs are nearly constant across neighbouring lines.
- *Random outages* (5–20 branches removed from case300). ARMA 0.243 versus Global MLP 0.371. The Global MLP holds up because each input slot still maps to a fixed component.
- *Training on six grids, testing on two unseen ones.* On the training grids the ARMA GNN beats the Global MLP on 5 of 6 (e.g. case89pegase 0.071 vs 0.791). On unseen case57 it scores 1.345 against DCPF's 0.584, and on case300 it scores 2.478 ± 1.636. The authors say plainly that DCPF is the better choice there. Size-independence makes it *possible* to run the model on a new grid, not *safe* to.

**A small example**

Four buses with branches a = (1,2), b = (2,3), c = (1,3) and d = (3,4). In the line graph:

```
a–b (share bus 2)   a–c (share bus 1)
b–c, b–d, c–d (all share bus 3)
```

Bus 3 has degree 3, so it becomes a *triangle* {b, c, d} in `L(G)`. A bus of degree `k` becomes a `k`-clique. Each line-graph node knows both of its endpoints' loads and generation, so one message-passing step on `L(G)` already mixes information across two buses and two lines.

Now cut branch c. In `L(G)` this deletes node c and its three edges, leaving a path a–b–d. The GNN's weights are shared across nodes, so it runs unchanged. A Global MLP would instead have to zero out c's input slot and hope it learned what "zero impedance" means. As the paper notes, zero `r` and `x` are hard to tell apart from a very short line.

The NRMSE scale: if the true power flow on line a has standard deviation 0.3 p.u. across samples, an NRMSE of 0.057 means a typical error of roughly 0.017 p.u. (1.7 MW on a 100 MVA base). An NRMSE of 0.887 means the model has barely improved on predicting the average flow.

**How it connects**

This entry builds on **A02** (per-unit values and complex power, the units of every input and output here). It looks ahead to **A03** (the admittance matrix, which the line graph re-encodes), **A04** (the equations the model is learning to solve) and **A05** (the Newton–Raphson labels, and convergence failures, which the dataset simply drops). **A06**'s DC approximation is the baseline that remains competitive out of distribution. Compared with **X003**: active-set learning keeps a solver in the loop and so is always either exact or flagged, while this model regresses the answer and nothing enforces that the predicted flows satisfy power balance at each bus. Closing that gap is phase E's job (**E01**, **E06**). In phase C, **C13** formalizes message passing with edge features (the alternative to the line-graph trick), **C14** explains over-smoothing, **C15** covers typed graphs for heterogeneous grid components, and **C16** returns to GNN power-flow solvers. The failure on unseen grids is a distribution-shift problem, the topic of **D12**. Next focus lesson: **A03**.

**Check yourself**

1. Why does converting to a line graph let the authors use a GNN layer that handles only node features, and what does it cost in graph size for a bus with many connections?
2. Why is a 40-layer GCN worse than the DC approximation, while a 5-layer ARMA network with the same 40-hop receptive field is the best model?
3. The model reaches an NRMSE of 0.057 on case118. Give two reasons why that number alone does not show that its output can be used as a power-flow solution.

<details><summary>Answers</summary>

1. Each branch has exactly two endpoints. So a branch's own parameters and both endpoint buses' features fit into one node feature vector, and no edge features remain. The cost: a bus of degree `k` becomes a `k`-clique, so the number of line-graph edges is `Σ_i k_i(k_i − 1)/2` instead of `Σ_i k_i/2`. That is cheap on sparse grids, but highly meshed substations add many edges.
2. Repeated GCN propagation acts as a low-pass filter. After 40 rounds the node features converge towards a smooth signal, but power flows on neighbouring lines can differ sharply in size and sign, so the GCN predicts something close to an average flow. ARMA layers re-inject the raw input at each recursive step (`X V_k`), which lets them represent higher-frequency graph signals. Using several layers with fewer recursions each also helped in practice.
3. (a) It is an average error under a training-like distribution. The same model scores 1.3–2.5 on unseen grids, and it was never tested on shifted load patterns. (b) Its outputs are trained with MSE only, so they do not have to satisfy the power-flow equations. Power injections at a bus need not balance, and the implied voltages need not agree between branches, so a small regression error can still be a physically inconsistent state. A per-line error can also matter much more near a thermal limit than the average suggests.

</details>
