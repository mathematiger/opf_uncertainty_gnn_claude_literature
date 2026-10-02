# Curriculum: Machine Learning for Power-System Optimization under Uncertainty

A daily self-study curriculum (one 7–10 minute read per day) for an ML researcher entering the field of
learned optimal power flow. It covers four areas:

- the grid physics and the optimization problems underneath;
- optimization and forecasting under uncertainty;
- graph-based, physics-informed and uncertainty-aware learning;
- how to make, check and evaluate a learned decision.

## Weekly rhythm

The curriculum moves through six phases (A–F). Each phase is the **focus** for a stretch of weeks, but every
week also gives the other phases room:

| Day (Berlin time) | Entry type | What it is |
|---|---|---|
| Mon, Wed, Fri | **Focus lesson** | The next unread lesson of the focus phase, in the order listed below (`A01`, `A02`, …). When a phase's list is done, the next phase becomes the focus. |
| Tue, Thu, Sat, Sun | **Cross-phase entry** | One entry from a track that belongs to a *different* phase than the focus (ID `X<nnn>`). For a phase **not yet reached**, it is a *preview*: a landmark paper, tutorial or blog post that motivates the area. For a phase **already completed**, it is a *deepening*: a recent paper, a harder follow-up or a worked example that builds on its lessons. Cross-phase entries never repeat a reference already listed in this file or in `log.md`, so they don't consume or duplicate the focus lessons. |

The cross-phase track is the one used least recently in `log.md` among the tracks outside the focus phase.
Focus lessons are sequential and not tied to dates, so a missed day shifts the plan by one lesson. The last
lesson of each phase is a **synthesis** that explains how the phase's literature fits together. Every entry
also has a "How it connects" section linking it to earlier entries by ID.

### Planned focus windows (if no day is missed)

| Phase | Tracks | Focus lessons | Focus window |
|---|---|---|---|
| A: Grid physics and OPF | 01, 02, 03 | 31 | 2026-10-05 → 2026-12-14 |
| B: Uncertainty in optimization and forecasting | 04, 05 | 25 | 2026-12-16 → 2027-02-10 |
| C: Learning to optimize and graph learning | 06, 07 | 25 | 2027-02-12 → 2027-04-09 |
| D: Uncertainty-aware learning | 08, 09 | 19 | 2027-04-12 → 2027-05-24 |
| E: Physics-informed and constrained learning | 10 | 19 | 2027-05-26 → 2027-07-07 |
| F: Conformal risk control, hybrid pipelines, evaluation | 11, 12 | 25 | 2027-07-09 → 2027-09-03 |

After phase F the focus days switch to **frontier mode** (see the end of this file).

**Where entries go.** Every entry is added at the top of `log.md`, which holds the full chronological record.
The same entry also goes into the folder of its track (`tracks/<NN-slug>/<date>-<lesson-id>-<short-title>.md`).

The ML side assumes prior knowledge of GNN basics, classification calibration (ECE, reliability diagrams)
and conformal prediction for classification. It builds from there instead of starting from zero. The power
systems and optimization side assumes no prior knowledge.

## Tracks

| # | Folder | Track |
|---|--------|-------|
| 00 | `tracks/00-synthesis` | Phase syntheses: how the literature connects |
| 01 | `tracks/01-grid-physics` | Grid physics & power flow |
| 02 | `tracks/02-opf` | Optimal power flow: formulations, relaxations, solvers |
| 03 | `tracks/03-multiperiod-operations` | Multi-period OPF & grid operations (dispatch, redispatch, storage, flexibility) |
| 04 | `tracks/04-optimization-under-uncertainty` | Stochastic, chance-constrained, risk-averse & robust optimization |
| 05 | `tracks/05-probabilistic-forecasting` | Probabilistic forecasting, scoring rules, scenario generation |
| 06 | `tracks/06-learning-to-optimize` | Learning to optimize, OPF proxies, decision-focused learning |
| 07 | `tracks/07-graph-spatiotemporal` | GNNs and spatio-temporal GNNs for power grids |
| 08 | `tracks/08-learning-from-distributions` | Uncertainty as input: distributions, sets, embeddings |
| 09 | `tracks/09-uncertainty-quantification` | Aleatoric/epistemic UQ and evaluating uncertainty signals |
| 10 | `tracks/10-physics-informed` | Physics-informed and constrained learning |
| 11 | `tracks/11-conformal-risk-control` | Conformal prediction, risk control, selective prediction |
| 12 | `tracks/12-hybrid-and-evaluation` | Hybrid ML–solver pipelines, warm starts, benchmarking |

The main references below give author, title and year. The daily job looks up and verifies the exact link
for each one.

---

## Phase A: Grid physics and optimal power flow

### Unit 1 · Power-flow physics
| ID | Track | Topic | Main reference |
|---|---|---|---|
| A01 | 01 | Why grids are hard to operate: instantaneous balance, transmission vs distribution, who decides what | von Meier, *Electric Power Systems: A Conceptual Introduction* (2006), ch. 1 |
| A02 | 01 | AC basics for ML people: phasors, complex power, reactive power intuition, per-unit system | von Meier (2006), ch. 2–3; Andersson, *Modelling and Analysis of Electric Power Systems* (ETH lecture notes) |
| A03 | 01 | Network model: π-line model, transformers, the admittance matrix as a weighted graph Laplacian | Andersson (ETH notes), network modelling chapter |
| A04 | 01 | The AC power-flow equations and bus types (PQ, PV, slack) | Frank & Rebennack, "An introduction to optimal power flow: theory, formulation, and examples" (IIE Trans., 2016) |
| A05 | 01 | Solving power flow: Newton–Raphson, the Jacobian, convergence failure, what pandapower does | Thurner et al., "pandapower" (IEEE TPWRS, 2018); Andersson (ETH notes) |
| A06 | 01 | The DC approximation: assumptions, PTDFs, and when it breaks (voltage/reactive-driven congestion) | Stott, Jardim & Alsaç, "DC Power Flow Revisited" (IEEE TPWRS, 2009) |

### Unit 2 · Operating limits and the birth of OPF
| ID | Track | Topic | Main reference |
|---|---|---|---|
| A07 | 01 | Operating limits: thermal (current vs apparent power), voltage bands, generator P/Q capability | Frank & Rebennack (2016) |
| A08 | 02 | Economic dispatch without a network; the Lagrange multiplier as marginal price | Wood, Wollenberg & Sheblé, *Power Generation, Operation, and Control*, economic dispatch chapter |
| A09 | 02 | History and taxonomy of OPF formulations | Cain, O'Neill & Castillo, "History of Optimal Power Flow and Formulations" (FERC, 2012) |
| A10 | 02 | The AC-OPF problem written out: variables, objective, constraints, polar vs rectangular | Frank & Rebennack (2016) |
| A11 | 02 | DC-OPF as LP/QP, congestion, and locational marginal prices as duals | Kirschen & Strbac, *Fundamentals of Power System Economics*, transmission pricing chapter |
| A12 | 02 | Security: N-1, preventive vs corrective security-constrained OPF (overview only) | Capitanescu et al., "State-of-the-art, challenges, and future trends in SCOPF" (EPSR, 2011) |

### Unit 3 · Nonlinear optimization and solvers
| ID | Track | Topic | Main reference |
|---|---|---|---|
| A13 | 02 | Lagrangian, KKT conditions, constraint qualifications, duality, seen through OPF | Boyd & Vandenberghe, *Convex Optimization*, ch. 5; Nocedal & Wright, *Numerical Optimization*, ch. 12 |
| A14 | 02 | Interior-point methods: barrier, central path, primal–dual Newton steps | Nocedal & Wright, ch. 19 |
| A15 | 02 | Ipopt in practice: filter line search, restoration phase, scaling, key options | Wächter & Biegler, "On the implementation of an interior-point filter line-search algorithm…" (Math. Prog., 2006) |
| A16 | 02 | Nonconvexity, local optima, NP-hardness, and why local solutions are usually good enough | Bienstock & Verma, "Strong NP-hardness of AC power flows feasibility" (2019); Bukhsh et al., "Local solutions of the optimal power flow problem" (IEEE TPWRS, 2013) |
| A17 | 02 | Modelling stacks: JuMP/PowerModels.jl, MATPOWER, pandapower OPF | Coffrin et al., "PowerModels.jl" (PSCC, 2018) |
| A18 | 02 | Why warm-starting interior-point methods is hard, and what helps | Yildirim & Wright, "Warm-start strategies in interior-point methods for linear programming" (SIAM J. Optim., 2002); Baker, "Learning warm-start points for AC OPF" (MLSP, 2019) |

### Unit 4 · Convex relaxations and approximations
| ID | Track | Topic | Main reference |
|---|---|---|---|
| A19 | 02 | Why relax: lower bounds, global optimality, optimality gaps | Molzahn & Hiskens, "A Survey of Relaxations and Approximations of the Power Flow Equations" (FnT, 2019), ch. 1–2 |
| A20 | 02 | SDP relaxation and rank-1 exactness | Lavaei & Low, "Zero duality gap in optimal power flow problem" (IEEE TPWRS, 2012) |
| A21 | 02 | SOCP relaxation / branch-flow model; exactness on radial networks | Jabr (2006); Farivar & Low, "Branch flow model: relaxations and convexification" (IEEE TPWRS, 2013) |
| A22 | 02 | QC relaxation, strengthening, and why relaxed solutions are not AC-feasible on meshed grids | Coffrin, Hijazi & Van Hentenryck, "The QC Relaxation" (IEEE TPWRS, 2016) |
| A23 | 02 | Linear approximations beyond DC (LPAC and friends) | Coffrin & Van Hentenryck, "A Linear-Programming Approximation of AC Power Flows" (INFORMS JoC, 2014) |
| A24 | 02 | Benchmarking OPF solvers: PGLib-OPF and reporting optimality gaps | Babaeinejadsarookolaee et al., "The Power Grid Library for Benchmarking AC OPF Algorithms" (2019) |

### Unit 5 · Multi-period OPF and grid operations
| ID | Track | Topic | Main reference |
|---|---|---|---|
| A25 | 03 | Time coupling: ramps, and why the problem stops decomposing per timestep | Gill, Kockar & Ault, "Dynamic Optimal Power Flow for Active Distribution Networks" (IEEE TPWRS, 2014) |
| A26 | 03 | Storage modelling: state of charge, efficiencies, no simultaneous charge/discharge | Murillo-Sánchez et al., "Secure planning and operations of systems with stochastic sources, energy storage, and active demand" (MATPOWER MOST, IEEE TSG, 2013) |
| A27 | 03 | Flexible and shiftable loads, curtailment as flexibility | Albadi & El-Saadany, "A summary of demand response in electricity markets" (EPSR, 2008) |
| A28 | 03 | Dispatch, redispatch and curtailment in practice (German/European context, Redispatch 2.0) | Bundesnetzagentur / ENTSO-E official explainers |
| A29 | 03 | Planning with forecasts: receding horizon / model predictive control | Rawlings, Mayne & Diehl, *Model Predictive Control*, ch. 1 |
| A30 | 03 | Unit commitment and MINLP: why discrete decisions explode (context) | Knueven, Ostrowski & Watson, "On mixed-integer programming formulations for the unit commitment problem" (INFORMS JoC, 2020) |
| A31 | 00 | Phase synthesis: from grid physics to multi-period decisions | — |

## Phase B: Uncertainty in optimization and forecasting

### Unit 6 · Decision-making under uncertainty
| ID | Track | Topic | Main reference |
|---|---|---|---|
| B01 | 04 | Taxonomy: stochastic, chance-constrained, robust, distributionally robust | Roald et al., "Power systems optimization under uncertainty: a review" (EPSR, 2023) |
| B02 | 04 | Two-stage stochastic programming, recourse, sample average approximation | Shapiro, Dentcheva & Ruszczyński, *Lectures on Stochastic Programming*, ch. 2 & 5 |
| B03 | 04 | EVPI and the value of the stochastic solution: deterministic vs stochastic vs perfect foresight | Birge, "The value of the stochastic solution in stochastic linear programs with fixed recourse" (Math. Prog., 1982) |
| B04 | 04 | Risk measures: VaR, CVaR, coherence, and the Rockafellar–Uryasev reformulation | Rockafellar & Uryasev, "Optimization of Conditional Value-at-Risk" (J. Risk, 2000) |
| B05 | 04 | Chance constraints and CVaR as their convex conservative approximation | Nemirovski & Shapiro, "Convex approximations of chance constrained programs" (SIAM J. Optim., 2006) |
| B06 | 04 | The scenario approach: how many samples buy a guarantee | Calafiore & Campi, "The Scenario Approach to Robust Control Design" (IEEE TAC, 2006); Campi & Garatti (SIAM J. Optim., 2008) |

### Unit 7 · Uncertainty in OPF
| ID | Track | Topic | Main reference |
|---|---|---|---|
| B07 | 04 | Chance-constrained DC-OPF with affine participation factors | Bienstock, Chertkov & Harnett, "Chance-Constrained Optimal Power Flow" (SIAM Rev., 2014) |
| B08 | 04 | Balancing rules as recourse: affine policies and adjustable robust optimization | Warrington et al., "Policy-based reserves for power systems" (IEEE TPWRS, 2013); Ben-Tal et al. (Math. Prog., 2004) |
| B09 | 04 | Chance-constrained AC-OPF reformulations | Roald & Andersson, "Chance-constrained AC optimal power flow: reformulations and efficient algorithms" (IEEE TPWRS, 2018) |
| B10 | 04 | Scenario-based OPF and reserve scheduling | Vrakopoulou et al. (IEEE TPWRS, 2013); Margellos, Goulart & Lygeros (IEEE TAC, 2014) |
| B11 | 04 | CVaR-constrained stochastic OPF | Summers et al., "Stochastic optimal power flow based on CVaR and distributional robustness" (IJEPES, 2015) |
| B12 | 04 | Distributionally robust OPF with Wasserstein balls | Mohajerin Esfahani & Kuhn (Math. Prog., 2018); Duan et al. (IEEE TPWRS, 2018) |

### Unit 8 · Probabilistic forecasting and its evaluation
| ID | Track | Topic | Main reference |
|---|---|---|---|
| B13 | 05 | Forecasts as distributions; how load, wind and PV errors behave (heteroskedastic, horizon-dependent, bounded) | Gneiting & Katzfuss, "Probabilistic Forecasting" (Annu. Rev. Stat., 2014); Hong et al., "GEFCom2014" (IJF, 2016) |
| B14 | 05 | Quantile regression, pinball loss, quantile crossing | Koenker & Bassett, "Regression Quantiles" (Econometrica, 1978) |
| B15 | 05 | Calibration and sharpness of forecasts, PIT histograms | Gneiting, Balabdaoui & Raftery (JRSS-B, 2007) |
| B16 | 05 | Proper scoring rules: CRPS, interval score, log score, Brier | Gneiting & Raftery, "Strictly Proper Scoring Rules…" (JASA, 2007) |
| B17 | 05 | Evaluating multivariate forecasts and scenarios: energy and variogram scores | Scheuerer & Hamill (MWR, 2015); Pinson & Girard (Applied Energy, 2012) |
| B18 | 05 | Non-Gaussian errors: skew-normal/skew-t, logit transforms for bounded quantities | Azzalini & Capitanio (JRSS-B, 2003); Pinson, "Very-short-term probabilistic forecasting of wind power with generalized logit-normal distributions" (JRSS-C, 2012) |

### Unit 9 · Scenarios and synthetic uncertainty
| ID | Track | Topic | Main reference |
|---|---|---|---|
| B19 | 05 | From marginals to scenarios: Gaussian copulas and space–time dependence | Pinson et al., "From probabilistic forecasts to statistical scenarios of short-term wind power production" (Wind Energy, 2009) |
| B20 | 05 | Space–time trajectories for PV | Golestaneh, Gooi & Pinson (Applied Energy, 2016) |
| B21 | 05 | Generative scenario models: GANs and normalizing flows | Chen et al. (IEEE TPWRS, 2018); Dumas et al. (Applied Energy, 2022) |
| B22 | 05 | Scenario reduction: how many scenarios and which ones | Dupačová, Gröwe-Kuska & Römisch (Math. Prog., 2003) |
| B23 | 05 | Simulating forecast errors with controlled properties; true vs reported uncertainty | Hodge et al., "Wind power forecasting error distributions: an international comparison" (2012); Hyndman & Athanasopoulos, *FPP3* (ARIMA chapters) |
| B24 | 05 | Open grid and time-series data (SimBench, chronix2grid) and leak-free splits for time series | Meinecke et al., "SimBench" (Energies, 2020); Bergmeir & Benítez (Inf. Sci., 2012) |
| B25 | 00 | Phase synthesis: the uncertainty pipeline from data to a risk-aware decision | — |

## Phase C: Learning to optimize and graph learning

### Unit 10 · Learning to optimize
| ID | Track | Topic | Main reference |
|---|---|---|---|
| C01 | 06 | Amortized optimization: learning the map from problem parameters to solutions | Amos, "Tutorial on Amortized Optimization" (FnT ML, 2023) |
| C02 | 06 | Survey of the field and of OPF proxies | Kotary, Fioretto & Van Hentenryck, "End-to-End Constrained Optimization Learning: A Survey" (IJCAI, 2021) |
| C03 | 06 | Supervised OPF proxies | Pan et al., "DeepOPF" (IEEE TPWRS, 2021); Zamzam & Baker (SmartGridComm, 2020) |
| C04 | 06 | Lagrangian-dual training for constraint satisfaction | Fioretto, Mak & Van Hentenryck (AAAI, 2020) |
| C05 | 06 | Self-supervised primal–dual learning (no solver labels) | Park & Van Hentenryck, "Self-Supervised Primal-Dual Learning for Constrained Optimization" (AAAI, 2023) |
| C06 | 06 | Hard feasibility: completion/correction and feasibility restoration | Donti, Rolnick & Kolter, "DC3" (ICLR, 2021); Han et al., "FRMNet" (IEEE TPWRS, 2024) |

### Unit 11 · Decision-focused learning, differentiable optimization, datasets
| ID | Track | Topic | Main reference |
|---|---|---|---|
| C07 | 06 | Predict-then-optimize vs decision-focused learning | Elmachtoub & Grigas, "Smart Predict, then Optimize" (Mgmt. Sci., 2022); Mandi et al. (JAIR, 2024) |
| C08 | 06 | Task-based end-to-end learning in stochastic optimization | Donti, Amos & Kolter (NeurIPS, 2017) |
| C09 | 06 | Differentiable optimization layers | Amos & Kolter, "OptNet" (ICML, 2017); Agrawal et al., "Differentiable Convex Optimization Layers" (NeurIPS, 2019) |
| C10 | 06 | Implicit differentiation at a fixed point | Blondel et al., "Efficient and Modular Implicit Differentiation" (NeurIPS, 2022); Bai, Kolter & Koltun, "Deep Equilibrium Models" (NeurIPS, 2019) |
| C11 | 06 | Datasets for learned OPF and how the training distribution is built | Joswig-Jones et al., "OPF-Learn" (2022); Lovett et al., "OPFData" (2024); Varbella et al., "PowerGraph" (NeurIPS D&B, 2024) |
| C12 | 06 | Worst-case guarantees and verification for neural OPF proxies | Venzke et al. (SmartGridComm, 2020); Giraud et al. (2025) |

### Unit 12 · GNNs for power grids
| ID | Track | Topic | Main reference |
|---|---|---|---|
| C13 | 07 | Brief refresher: message passing with edge features; relational inductive bias on grids | Gilmer et al., "Neural Message Passing for Quantum Chemistry" (ICML, 2017); Battaglia et al. (2018) |
| C14 | 07 | Long-range limits: over-smoothing and over-squashing on high-diameter graphs | Topping et al., "Understanding over-squashing and bottlenecks on graphs via curvature" (ICLR, 2022); Alon & Yahav (ICLR, 2021) |
| C15 | 07 | Heterogeneous/typed graphs for grid components | Schlichtkrull et al., "R-GCN" (ESWC, 2018); Lopez-Garcia & Domínguez-Navarro (IEEE TPWRS, 2025) |
| C16 | 07 | GNN power-flow solvers | Donon et al., "Neural networks for power flow: Graph neural solver" (EPSR, 2020); Lin et al., "PowerFlowNet" (IJEPES, 2024) |
| C17 | 07 | GNN OPF proxies and topology transfer | Owerko, Gama & Ribeiro (ICASSP, 2020); Liu, Wu & Zhu (IEEE TPWRS, 2023); Piloto et al., "CANOS" (2024) |
| C18 | 07 | Reviews and lessons from real grids | Liao et al. (JMPSCE, 2022); Ringsquandl et al. (CIKM, 2021) |

### Unit 13 · Spatio-temporal GNNs
| ID | Track | Topic | Main reference |
|---|---|---|---|
| C19 | 07 | Sequence models for a planning horizon: GRU, TCN, attention | Bai, Kolter & Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks" (2018) |
| C20 | 07 | STGCN and graph-convolutional recurrence | Yu, Yin & Zhu, "STGCN" (IJCAI, 2018); Seo et al., "GCRN" (ICONIP, 2018) |
| C21 | 07 | Diffusion convolution and adaptive adjacency | Li et al., "DCRNN" (ICLR, 2018); Wu et al., "Graph WaveNet" (IJCAI, 2019) |
| C22 | 07 | Taxonomy of spatio-temporal graph learning | Cini et al., "Graph Deep Learning for Time Series Forecasting" (ACM CSUR, 2025) |
| C23 | 07 | Spatio-temporal GNNs for multi-period OPF | Rajaei, Arowolo & Cremer (ISGT Europe, 2025); Kim et al. (Electronics, 2026) |
| C24 | 07 | Temporal multi-task learning with storage; physics-informed spatio-temporal GCNs | Dai et al. (IEEE TII, 2025); Rong & Qin (IJEPES, 2026) |
| C25 | 00 | Phase synthesis: from single-step proxies to whole-horizon learned optimizers | — |

## Phase D: Uncertainty-aware learning

### Unit 14 · Uncertainty as an input
| ID | Track | Topic | Main reference |
|---|---|---|---|
| D01 | 08 | Two strategies: conditioning on input uncertainty vs propagating it | Gast & Roth, "Lightweight Probabilistic Deep Networks" (CVPR, 2018) |
| D02 | 08 | Kernel mean embeddings and distribution regression | Muandet et al. (FnT ML, 2017); Szabó et al. (JMLR, 2016) |
| D03 | 08 | Deep Sets and the permutation-invariance theorem | Zaheer et al., "Deep Sets" (NeurIPS, 2017) |
| D04 | 08 | Set Transformer and attention pooling | Lee et al., "Set Transformer" (ICML, 2019) |
| D05 | 08 | Latent set representations: neural processes and VAEs | Garnelo et al., "Neural Processes" (2018); Kingma & Welling (ICLR, 2014) |
| D06 | 08 | Quantiles vs scenarios vs embeddings, and why the target must depend on the input distribution | Donti et al. (2017) revisited; Saffari et al., probabilistic AC-OPF with a generative graph model (IEEE TETCI, 2024) |

### Unit 15 · Aleatoric vs epistemic uncertainty
| ID | Track | Topic | Main reference |
|---|---|---|---|
| D07 | 09 | Concepts and decomposition | Hüllermeier & Waegeman (Mach. Learn., 2021) |
| D08 | 09 | Heteroscedastic regression and its pitfalls | Kendall & Gal (NeurIPS, 2017); Seitzer et al., "On the pitfalls of heteroscedastic uncertainty estimation…" (ICLR, 2022) |
| D09 | 09 | Deep ensembles and why they work | Lakshminarayanan et al. (NeurIPS, 2017); Fort, Hu & Lakshminarayanan (2019) |
| D10 | 09 | Bayesian approximations and information-theoretic decomposition | Gal & Ghahramani (ICML, 2016); Depeweg et al. (ICML, 2018); Daxberger et al., "Laplace Redux" (NeurIPS, 2021) |
| D11 | 09 | Do decompositions hold up? | Mucsányi, Kirchhof & Oh (NeurIPS, 2024); Wimmer et al. (UAI, 2023) |
| D12 | 09 | Uncertainty under distribution shift, and on graphs | Ovadia et al., "Can You Trust Your Model's Uncertainty?" (NeurIPS, 2019); Stadler et al., "Graph Posterior Network" (NeurIPS, 2021) |

### Unit 16 · Evaluating uncertainty signals
| ID | Track | Topic | Main reference |
|---|---|---|---|
| D13 | 09 | Calibration for regression and recalibration | Kuleshov, Fenner & Ermon (ICML, 2018) |
| D14 | 09 | Prediction intervals: coverage, width, interval score | Dewolf, De Baets & Waegeman (AI Review, 2023) |
| D15 | 09 | Detecting large errors: AUROC and risk–coverage curves | Geifman & El-Yaniv (NeurIPS, 2017); Geifman et al. (ICLR, 2019) |
| D16 | 09 | Event probabilities: reliability diagrams, ECE pitfalls, Brier score | Nixon et al., "Measuring Calibration in Deep Learning" (CVPRW, 2019) |
| D17 | 09 | Conditional and subgroup calibration | Hébert-Johnson et al., "Multicalibration" (ICML, 2018) |
| D18 | 09 | Reporting over seeds and statistical comparison | Bouthillier et al. (MLSys, 2021); Agarwal et al. (NeurIPS, 2021) |
| D19 | 00 | Phase synthesis: using and trusting uncertainty in learned models | — |

## Phase E: Physics-informed and constrained learning

### Unit 17 · PINN foundations
| ID | Track | Topic | Main reference |
|---|---|---|---|
| E01 | 10 | The PINN idea: residuals of known equations as losses | Raissi, Perdikaris & Karniadakis (JCP, 2019); Karniadakis et al. (Nat. Rev. Phys., 2021) |
| E02 | 10 | Why PINNs fail to train: NTK view, stiffness | Wang, Yu & Perdikaris (JCP, 2022); Krishnapriyan et al. (NeurIPS, 2021) |
| E03 | 10 | Balancing many loss terms: fixed, uncertainty-based, gradient-based, dual-ascent weights | Wang, Teng & Perdikaris (SIAM J. Sci. Comput., 2021); Chen et al., "GradNorm" (ICML, 2018) |
| E04 | 10 | Soft vs hard constraints: penalties, augmented Lagrangian, projection | Lu et al., "PINNs with hard constraints for inverse design" (SIAM J. Sci. Comput., 2021); Nocedal & Wright, ch. 17 |
| E05 | 10 | Constraints by construction: bounded outputs, integrators for state variables | Kotary et al. (2021) revisited; Lu et al. (2021) |
| E06 | 10 | Physics-informed ML in power systems | Misyris, Venzke & Chatzivasileiadis (PESGM, 2020); Huang & Wang, review (IEEE TPWRS, 2023) |

### Unit 18 · Physics-informed OPF learning
| ID | Track | Topic | Main reference |
|---|---|---|---|
| E07 | 10 | KKT-informed neural AC-OPF | Nellikkath & Chatzivasileiadis (EPSR, 2022) |
| E08 | 10 | Physics-informed geometric deep learning | de Jongh et al. (EPSR, 2022) |
| E09 | 10 | Physics-guided GNNs with Lagrangian duality | Gao et al. (IEEE TPWRS, 2024); Yang et al. (IEEE TII, 2024) |
| E10 | 10 | Typed physics-informed GNNs with hard constraints | Lopez-Garcia & Domínguez-Navarro (2025); Li et al. (SEGAN, 2026) |
| E11 | 10 | AC-OPF as constrained optimization inside a GNN | Varbella et al., "PINCO" (2024) |
| E12 | 10 | Differentiable and GPU-batched power flow | Öz et al., "Differentiable Power-Flow Optimization" (2026); Wang, Wende-von Berg & Braun (SEGAN, 2021) |

### Unit 19 · Time coupling and risk in the loss
| ID | Track | Topic | Main reference |
|---|---|---|---|
| E13 | 10 | Penalties for intertemporal constraints (ramps, state-of-charge bounds, terminal state) and normalization | Rong & Qin (2026) revisited; Gill et al. (2014) |
| E14 | 10 | Differentiating through a Newton solve (implicit layers revisited) | Blondel et al. (2022) |
| E15 | 10 | Out-of-distribution generalization of physics-informed models | Krishnapriyan et al. (2021) |
| E16 | 10 | Projection and feasibility-restoration layers for OPF | Zamzam & Baker (2020); Han et al. (2024) |
| E17 | 10 | Tail-aware training: CVaR of losses over scenarios | Curi et al., "Adaptive Sampling for Stochastic Risk-Averse Learning" (NeurIPS, 2020) |
| E18 | 10 | Physics-informed generative models for probabilistic AC-OPF | Saffari et al. (2024) revisited |
| E19 | 00 | Phase synthesis: what physics-informed learning can and cannot guarantee | — |

## Phase F: Conformal prediction, risk control, hybrid pipelines, evaluation

### Unit 20 · Conformal prediction for regression and structured outputs
| ID | Track | Topic | Main reference |
|---|---|---|---|
| F01 | 11 | Split/inductive conformal for regression | Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (2021); Papadopoulos et al. (ECML, 2002); Lei et al. (JASA, 2018) |
| F02 | 11 | Conformalized quantile regression | Romano, Patterson & Candès (NeurIPS, 2019) |
| F03 | 11 | Multi-output and multi-horizon conformal: joint coverage across nodes and timesteps | Stankevičiūtė, Alaa & van der Schaar, "Conformal Time-Series Forecasting" (NeurIPS, 2021) |
| F04 | 11 | Conformal prediction on graphs | Huang et al., "Uncertainty Quantification over Graph with Conformalized GNNs" (NeurIPS, 2023) |
| F05 | 11 | Beyond exchangeability: time series, weighting, adaptive conformal | Barber et al. (Ann. Stat., 2023); Gibbs & Candès (NeurIPS, 2021) |
| F06 | 11 | Limits of conditional coverage and practical remedies (Mondrian, localized) | Foygel Barber et al. (Inf. Inference, 2021); Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World* |

### Unit 21 · Risk control and selective prediction
| ID | Track | Topic | Main reference |
|---|---|---|---|
| F07 | 11 | Risk-controlling prediction sets | Bates et al. (J. ACM, 2021) |
| F08 | 11 | Conformal risk control | Angelopoulos et al. (ICLR, 2024) |
| F09 | 11 | Learn then Test: threshold selection as multiple testing | Angelopoulos et al. (Ann. Appl. Stat., 2025) |
| F10 | 11 | Selective prediction and the reject option | Chow (IEEE Trans. IT, 1970); El-Yaniv & Wiener (JMLR, 2010) |
| F11 | 11 | Learning to defer to an expert (or a solver) | Madras, Pitassi & Zemel (NeurIPS, 2018); Mozannar & Sontag (ICML, 2020) |
| F12 | 11 | Bounding a violation probability from finite simulation (Clopper–Pearson, Hoeffding, sample sizes) | Clopper & Pearson (Biometrika, 1934); Campi & Garatti (2008) revisited |

### Unit 22 · Hybrid ML–solver pipelines
| ID | Track | Topic | Main reference |
|---|---|---|---|
| F13 | 12 | AI proposals plus classical checks in critical infrastructure | Leyli-Abadi et al. (IEEE SMC, 2025); Marot et al. (JMPSCE, 2022) |
| F14 | 12 | Warm starts from learned predictions | Baker (2019); Diehl, "Warm-Starting AC OPF with Graph Neural Networks" (NeurIPS CCAI workshop, 2019) |
| F15 | 12 | Local repair: gradient correction through differentiable physics and windowed re-optimization | Donti et al. (2021) revisited |
| F16 | 12 | Honest speed-up accounting: checks, repairs and fallbacks included | Chen, Tanneau & Van Hentenryck, "End-to-end feasible optimization proxies for large-scale economic dispatch" (IEEE TPWRS, 2023) |
| F17 | 12 | GPU-accelerated solvers as the competing baseline | Shin, Anitescu & Pacaud (EPSR, 2024) |
| F18 | 12 | Out-of-sample evaluation of stochastic decisions; excess risk over the solver's own risk | Kaut & Wallace, "Evaluation of scenario-generation methods for stochastic programming" (2007); Birge (1982) revisited |

### Unit 23 · Evaluation and reproducible benchmarking
| ID | Track | Topic | Main reference |
|---|---|---|---|
| F19 | 12 | Metrics for learned OPF: optimality gap, violation frequency and magnitude per constraint class, regret | Kotary et al. (2021); PGLib (2019) |
| F20 | 12 | Ablation design: separating the effect of inputs, targets and losses | Sculley et al., "Winner's Curse? On Pace, Progress, and Empirical Rigor" (ICLR-W, 2018) |
| F21 | 12 | Shift test sets and challenge sets | Koh et al., "WILDS" (ICML, 2021) |
| F22 | 12 | Statistical comparison across seeds and datasets | Demšar (JMLR, 2006) |
| F23 | 12 | Scaling studies: grid size, horizon, number of scenarios | Lovett et al. (2024); Piloto et al. (2024) |
| F24 | 12 | Open tooling: pandapower, PowerModels.jl, Grid2Op, OpenRAO | Thurner et al. (2018); Coffrin et al. (2018) |
| F25 | 00 | Capstone synthesis: a map of the whole literature | — |

---

## Frontier mode (after F25)

Focus days (Mon, Wed, Fri) then cover one recent paper each (published within roughly the last 12 months).
Tracks 02–12 rotate, always picking the track used least recently. Every fourth frontier entry is a synthesis
that relates the recent papers to each other and to the curriculum. Frontier IDs are `N<nnn>`. Cross-phase days
continue as before, now drawing on any track.
