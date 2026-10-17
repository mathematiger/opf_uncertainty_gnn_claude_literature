## 2026-10-17 (Saturday) — X009 · Track 12: Hybrid ML–solver pipelines, warm starts, benchmarking — Keep the solver, learn its decisions: GNN branching inside an exact MILP solver

**Type:** Cross-phase preview
**Main reference:** [Exact Combinatorial Optimization with Graph Convolutional Neural Networks — M. Gasse, D. Chételat, N. Ferroni, L. Charlin & A. Lodi (2019), Advances in Neural Information Processing Systems 32 (NeurIPS 2019)](https://arxiv.org/abs/1906.01629) — read Sections 1 and 3–6 (about 6 pages: branch-and-bound as an MDP, imitation of strong branching, the bipartite state, the two half-convolutions, eq. 4, and the prenorm layer). Then read Tables 2 and 3 slowly, with the metric definitions in §5.1 "Evaluation". Skim Section 2. The link goes to the open arXiv version (v3, October 2019). The code is at [github.com/ds4dm/learn2branch](https://github.com/ds4dm/learn2branch).
**Further reading:**
- [Solving Mixed Integer Programs Using Neural Networks — V. Nair, S. Bartunov, F. Gimeno et al. (2020, rev. 2021), arXiv:2012.13349](https://arxiv.org/abs/2012.13349) — scales the same idea up: a learned brancher plus "Neural Diving", which proposes partial integer assignments that the solver then completes. The solver still certifies the result.
- [Ecole: A Gym-like Library for Machine Learning in Combinatorial Optimization Solvers — A. Prouvost, J. Dumouchelle, L. Scavuzzo, M. Gasse, D. Chételat & A. Lodi (2020), NeurIPS Workshop on Learning Meets Combinatorial Algorithms](https://arxiv.org/abs/2011.06069) — the tooling that came out of this line of work: solver decisions exposed as Gym-style environments.

**Why this matters** — There are two ways to bring ML to an optimization problem. You can replace the solver with a network that outputs a solution, or you can keep the solver and let a network make some of its internal decisions. In the second design a bad prediction costs time, not correctness, and anyone building learned solvers for grid operations needs to know both designs and how to measure them fairly. This paper is the standard reference for the second design. It also shows how to report results so that an optimization expert believes them.

**Core ideas**

*The problem.* A mixed-integer linear program (MILP) is

```
min_x  cᵀx   s.t.  A x ≤ b,  l ≤ x ≤ u,  x ∈ ℤ^p × ℝ^(n−p)
```

Power-system decisions such as generator on/off (unit commitment, A30) make problems MILPs. If you drop the integrality requirement you get an LP, called the *relaxation*. It is cheap, and its value is a **lower bound** on the MILP optimum because it optimizes over a larger set.

*Branch-and-bound (B&B).* Start by solving the relaxation. If some integer variable has a fractional value, say `x_i* = 2.4`, split the problem into two children, one with `x_i ≤ 2` and one with `x_i ≥ 3`, and solve their relaxations. Repeat. This builds a search tree. Every integer-feasible solution you find is an **upper bound** (an *incumbent*). The minimum of the relaxation values over the open leaves is a **global lower bound**. A subtree whose relaxation value is already worse than the incumbent cannot contain anything better, so you *prune* it. The algorithm stops when the two bounds meet. If you stop early, you still have a certified *optimality gap*: "the solution is within 2% of optimal", even though you do not know the optimum. This certificate is what "exact" means in the title, and it is why commercial solvers are built on B&B.

*The decision that is learned.* At each node the solver has to pick **which** fractional variable to branch on. The choice can change the tree size by orders of magnitude. The best known rule for small trees is **strong branching**: for every candidate, solve both child LPs and score the candidate by how much the bound improves. It produces tiny trees, but it costs two LPs per candidate per node, so its total runtime is poor. Solvers therefore use cheap proxies instead. SCIP's default, *reliability pseudocost branching* (RPB), uses strong-branching scores early on and then switches to historical estimates.

*MDP view, but no RL.* The solver is the environment, the brancher is the agent, actions come from the candidate set `A(s_t)`, and an episode is one solved instance. RL is slow here because episodes get longer the worse the policy is. The authors use **behavioural cloning** of strong branching instead. They record expert pairs `(s, a*)` and minimise

```
L(θ) = −(1/N) Σ_(s,a*)∈D  log π_θ(a* | s)
```

This is plain classification. Earlier work instead regressed the expert scores or learned rankings.

*The state as a graph.* Each node LP is encoded as a **bipartite graph**. There is one node per constraint (row of `A`) and one node per variable (column), and an edge `(i, j)` wherever `A_ij ≠ 0`. Constraint, variable and edge feature tensors `C ∈ ℝ^(m×c)`, `V ∈ ℝ^(n×d)` and `E ∈ ℝ^(m×n×e)` describe the current LP; the full feature list is in the supplement. If cuts are only added at the root, the topology is the same at every node of the tree. The network does one graph convolution, split into two half-convolutions:

```
c_i ← f_C( c_i , Σ_{j:(i,j)∈E} g_C(c_i, v_j, e_ij) )      (variables → constraints)
v_j ← f_V( v_j , Σ_{i:(i,j)∈E} g_V(c_i, v_j, e_ij) )      (constraints → variables)
```

Here `f` and `g` are 2-layer MLPs. A final MLP on the variable nodes is followed by a softmax that is *masked* to the candidate set. The network runs at every node, so it must be cheap: deeper models made slightly better decisions but were slower overall.

*The GNN detail you will recognise.* They use **sum** aggregation, not mean aggregation, and cite the GIN argument (Xu et al.): a mean aggregator cannot count, for example how many constraints a variable appears in. Unnormalised sums make weight initialisation depend on node degree. They fix this with a **prenorm** layer, `x ← (x − β)/σ`, applied after each sum. Its `β` and `σ` are set from training-set statistics and then frozen. In the ablation (Table 3, set covering) the variants tie on small instances; on Hard, sum + prenorm needs 56,348 nodes against 62,227 for mean.

*Results and how they are reported.* There are four NP-hard families: set covering, combinatorial auctions, capacitated facility location and maximum independent set. Each has Easy, Medium and Hard sizes. Training uses **only the Easy size**: for each family, 100,000 expert samples from 10,000 instances. Every method runs inside SCIP 6.0.1 with presolve and primal heuristics on and a 1-hour limit. Each cell covers 20 instances × 5 seeds. The reported metrics are the 1-shifted geometric mean of solving time, *wins* (the number of instances on which a method was fastest, over the number it solved) and node counts on instances that every method solved. Highlights from Table 2:

- Set covering, Hard (2,000 rows, trained on 500): GCNN takes 1,490 s on average against 1,677 s for RPB. It is fastest on 66 of the 70 instances it solves, while RPB is fastest on 4 of its 65. GCNN needs 29,981 nodes against 47,299 for RPB.
- Full strong branching gives the smallest trees on Easy but solves **no** Hard set-covering instance within the hour.
- Maximum independent set is the honest failure. On Hard, GCNN is the fastest of the learned branchers, but it solves only 29 instances, while RPB solves 42, and its variance is high.

**A small example**

Take `min −5x₁ − 4x₂` subject to `6x₁ + 4x₂ ≤ 24`, `x₁ + 2x₂ ≤ 6`, with `x ≥ 0` and integer.

```
root LP:  x = (3, 1.5),  value −21          → lower bound −21
branch on x₂ (the only fractional variable):
  x₂ ≥ 2:  x = (2, 2),     value −18        integer → incumbent −18
  x₂ ≤ 1:  x = (3.33, 1),  value −20.67     bound beats −18 → keep open
    branch on x₁:
      x₁ ≤ 3: x = (3, 1), value −19         integer → incumbent −19
      x₁ ≥ 4: x = (4, 0), value −20         integer → incumbent −20
no open leaves → optimum −20 at (4, 0), certified
```

Strong branching at the root would score `x₂` by its child bounds, −20.67 and −18, which improve on −21 by 0.33 and 3. SCIP combines the two gains as a product, so the score is ≈ 1. Here there is only one candidate, so the rule makes no choice. In a unit-commitment MILP there are hundreds of fractional on/off variables at every node, and that is where a learned scorer pays off. Note also that if the network had chosen badly, the answer would still be −20. Only the tree would have been bigger.

**How it connects**

- **X003** (active-set learning) has the same structure: ML proposes, and an exact method checks. There the guess is which constraints bind, and the solver confirms or falls back. Here the guess is a branching decision, and correctness never depends on it.
- **X001** (Powell's policy classes): learning a fast approximation of an expensive look-ahead rule (strong branching) by imitation is a policy-function approximation of a look-ahead policy, used *inside* an algorithm.
- **A05** (Newton–Raphson) showed that the starting point decides whether a solver converges. F14 returns to it as learned warm starts, another "help the solver" component.
- The GNN toolkit here (sum aggregation, edge features, bipartite structure) is what phase C (C13–C17) applies to grid graphs.
- **Coming up:** A30 introduces unit commitment as a MILP. Phase F's hybrid unit (F13–F18) then covers AI proposals with classical checks, warm starts, local repair, and honest speed-up accounting (F16). Tables 2 and 3 of this paper are a good model for F16: report the *whole* pipeline's wall-clock time against the solver's own default, not against a slow expert.

**Check yourself**

1. Why is the GCNN brancher "exact" while a network that directly outputs an integer solution is not? What does a bad prediction cost in each case?
2. Full strong branching gives the smallest trees, yet it is the slowest method on Hard instances. What does this say about the metric you should optimise and report for a learned component inside a solver?
3. Why does sum aggregation matter for this task, and what problem does the prenorm layer solve?

<details><summary>Answers</summary>

1. B&B only prunes a subtree when the relaxation bound proves it cannot beat the incumbent, and it stops only when the bounds meet. The branching variable changes the shape of the tree, not the validity of these proofs, so any policy, even a random one, returns a certified optimum. A bad branching choice only makes the tree larger and the solve slower. A network that outputs a solution directly gives no bound. A bad prediction can be suboptimal or infeasible, and you would need a separate check to find out.
2. Fewer nodes is only a proxy. What matters is end-to-end time, including the cost of each decision (feature extraction and inference for a network, two LPs per candidate for strong branching). So you should report wall-clock time, wins and solved counts under a fixed time limit, compared with the solver's tuned default (RPB), and not only accuracy or tree size. The authors kept the model small for exactly this reason.
3. Good branching depends on counts and degrees, for example in how many constraints a variable appears. A mean aggregator cannot represent these (the GIN argument). Unnormalised sums give activations whose scale grows with node degree, which makes initialisation and training unstable and hurts transfer to larger instances with higher degrees. The prenorm layer standardises each sum with frozen training-set statistics. In the ablation, sum + prenorm generalises best to Hard instances.

</details>
