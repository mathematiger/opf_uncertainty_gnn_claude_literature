## 2026-11-08 (Sunday) — X022 · Track 07: GNNs and spatio-temporal GNNs for power grids — PV systems as a network of virtual weather stations: spatio-temporal GNNs for solar forecasting

**Type:** Cross-phase preview
**Main reference:** [Spatio-temporal graph neural networks for multi-site PV power forecasting — J. Simeunović, B. Schubnel, P.-J. Alet & R. E. Carrillo (2022), IEEE Transactions on Sustainable Energy 13(2):1210–1220](https://arxiv.org/abs/2107.13875) — read §II (graph convolution and the forecasting problem, eqs. 1–6), §III-A (GCLSTM, eq. 7 and the paragraph on the learned Laplacian), §III-B (GCTrafo, Fig. 2), and §IV-E–F (metrics, Table I, Figs. 4–7 and Table II). Skim §IV-A–D (data, baselines, training) and the Appendix (hyperparameters). About 10 pages. The link goes to the open arXiv version (v2, November 2021). The published version is at [DOI 10.1109/TSTE.2021.3125200](https://doi.org/10.1109/TSTE.2021.3125200).
**Further reading:**
- [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering — M. Defferrard, X. Bresson & P. Vandergheynst (2016), NeurIPS](https://arxiv.org/abs/1606.09375) — the source of the Chebyshev graph filter used in every layer of both models. Read §2 for why a degree-K polynomial in the Laplacian is a K-hop local filter.
- [High-resolution PV forecasting from imperfect data: a graph-based solution — R. E. Carrillo et al. (2020), Energies 13(21):5763](https://doi.org/10.3390/en13215763) — the STAR baseline: a linear spatio-temporal autoregression with group-Lasso neighbour selection. It is open access and a good reminder of how strong a linear graph model can be at short horizons.

**Why this matters** — An OPF solved for the next hours is only as good as the PV and wind forecasts it receives. Those forecast errors are correlated across sites, because the same cloud passes over many systems. Anyone building learned OPF solvers under forecast uncertainty will meet spatio-temporal GNNs twice: as the forecaster that produces the uncertain inputs, and as the architecture template (graph convolution plus a sequence model) for multi-period learned dispatch. This paper is a clean, readable example of the first role, and it introduces the building blocks of phase C's spatio-temporal unit.

**Core ideas**

**The physical intuition.** PV output at a site is roughly `clear-sky output × (1 − cloud attenuation)`. The clear-sky part is a deterministic function of location and time (sun position), so the hard part is the clouds. Clouds are advected by wind and they change shape slowly, so a shadow that hits one PV system now is likely to hit its downwind neighbours in 10–60 minutes. A dense fleet of rooftop systems therefore acts as a *network of virtual weather stations*. Their recent output tells you where the clouds are and where they are heading. Numerical weather prediction (NWP) is the classical alternative. It is good for day-ahead forecasts, but its grid is coarse in space (kilometres) and in time (hourly or 3-hourly), so it is poor at sub-hourly, site-level forecasts.

**The problem.** Let `p(t) ∈ ℝ^N` be the power of all `N` sites at time step `t` (15-minute resolution). The goal is a sequence-to-sequence forecaster

```
p̂(t), …, p̂(t+H−1) = f_β( p(t−M), …, p(t−1) )
```

trained with summed squared error over the horizon, `Σ_t Σ_{τ=t}^{t+H−1} ‖p̂(τ) − p(τ)‖²`. Here `M = 16` past steps (4 h) and `H = 24` future steps (6 h). The output is a point forecast, not a distribution.

**The graph is not the electrical grid.** This is the key difference from X004 and X013. Nodes are PV sites; edges come from a *k-nearest-neighbour* graph on geographic coordinates (`A_ij = 1` if the sites are among each other's nearest neighbours; 15–24 neighbours depending on the model). The graph encodes proximity of weather, not physical power flow. The authors then go further: the non-zero entries of the sparse scaled Laplacian are **learned** by SGD along with the weights. kNN fixes only the sparsity pattern; the edge weights adapt to terrain, orientation and typical wind directions.

**The spatial operator: Chebyshev graph convolution.** With Laplacian `L = D − A`, largest eigenvalue `λ_max` and scaled Laplacian `L̃ = 2L/λ_max − I`, a graph filter is a polynomial

```
h ∗_G x ≈ Σ_{k=0}^{K−1} θ_k T_k(L̃) x,      T_0 = I, T_1 = L̃, T_k = 2 L̃ T_{k−1} − T_{k−2}
```

Because `T_k(L̃)` is a degree-`k` polynomial in a sparse matrix, the output at a node depends only on nodes up to `K−1` hops away, at cost `O(K|E|)`. In ML terms, it is a message-passing layer with `K−1` rounds of neighbour aggregation folded into one linear operator, with separate weights per hop order. With feature channels, `θ_k` becomes a weight matrix `W_k ∈ ℝ^(f_out × f_in)` (eq. 4). The A03 entry showed that the admittance matrix is a weighted Laplacian; here the Laplacian is a spatial-smoothness operator on a sensor graph, but the algebra is the same.

**Model 1: GCLSTM.** An encoder-decoder LSTM in which every matrix multiplication in the gates is replaced by a graph convolution:

```
f(τ) = σ(W_f,h ∗_G h(τ−1) + W_f,x ∗_G x(τ) + b_f)        (input, output gates analogous)
c(τ) = i(τ) ⊙ tanh(W_c,h ∗_G h(τ−1) + W_c,x ∗_G x(τ) + b_c) + f(τ) ⊙ c(τ−1)
h(τ) = o(τ) ⊙ tanh(c(τ))
```

Hidden states live on the nodes, `h(τ) ∈ ℝ^(N × 32)`. Each recurrent step lets information travel a few more hops, so the receptive field grows with time, which is exactly what advection needs. This is the graph-convolutional recurrent network idea (GCRN, see C20).

**Model 2: GCTrafo.** A transformer variant in which attention runs over *time*, separately at each node, while the queries, keys and values are produced by graph convolutions after 1-D temporal convolutions:

```
q(τ) = W_q ∗_G x̃(τ),   k(τ) = W_k ∗_G x̆(τ),   v(τ) = W_v ∗_G x̌(τ)
att_v(τ) = Σ_τ' softmax_τ'( q_v(τ) · k_v(τ') ) v_v(τ')
```

There are 8 heads. Spatial mixing happens in the graph convolutions, temporal mixing in the attention. The decoder uses no graph convolution, because its inputs are deterministic clear-sky values.

**Inputs that matter.** The encoder sees, at each node, the power, the clear-sky global irradiance from PVlib (computed from latitude, longitude, altitude and time), and a rolling mean of the power over `[τ − 72h, τ − 24h]`. The decoder sees future clear-sky global and direct irradiance plus that rolling mean. Power is normalized per site by its maximum in the training year. No weather data at all.

**Setup and results.** Two Swiss datasets with 15-minute resolution: 304 real PV systems, and 1000 synthetic systems simulated from satellite-derived irradiance. Training is on 2016 and testing on 2017; night times are excluded. Baselines are STAR (linear graph autoregression, refit every two weeks) and STCNN (a 2-D CNN on sites reordered by proximity). Errors are reported as NRMSE normalized by the site's peak power. From Table I (real dataset, NRMSE in %):

```
horizon      STAR    GCLSTM   GCTrafo   STCNN
15 min       7.32     7.36      8.76     9.30
1 h         10.40     9.93     10.95    11.31
3 h         14.20    12.40     13.54    14.62
6 h         17.33    15.53     16.07    17.17
```

Three findings are worth remembering. (1) At 15 minutes, the linear STAR model ties or beats the deep models: very short horizons are close to persistence and linear. The GNN advantage grows with the horizon. (2) Against single-site models that *do* use NWP (SVR and an encoder-decoder LSTM), the graph models win from 1 to 4 hours ahead. NWP wins at 5–6 hours. (3) Fig. 5 shows no clear degradation for isolated sites up to about 40 km from their nearest neighbour.

**The failure mode.** Table II and Fig. 7 show a revealing case. At 6 hours ahead in the morning, the graph models see only zeros from the night and fall back to clear sky plus the rolling mean. They overpredict until real daylight data arrives, and then their forecasts drop suddenly. On one cloudy day the 6-hour NRMSE is 20.9 % for GCLSTM versus 7.7 % for the NWP-based LSTM. A forecaster that only knows the recent past cannot see weather that has not arrived yet.

**A small example**

Three sites in a row from west to east, A — B — C, 10 km apart, so the kNN graph is the path A–B–C. Use clear-sky-normalized power. At 12:15 a cloud is over A (`x_A = 0.3`); B and C are still sunny (`x_B = x_C = 1`). The wind blows east at about 30 km/h, so the shadow should reach C around 12:55.

The Laplacian of the path has eigenvalues `0, 1, 3`, so `λ_max = 3` and

```
L̃ = 2L/3 − I = [ −1/3  −2/3    0  ]
               [ −2/3   1/3  −2/3 ]
               [   0   −2/3  −1/3 ]
```

Look at the deviation from a uniformly sunny sky, `δ = x − 1 = (−0.7, 0, 0)`:

```
T_1 δ = L̃ δ             = ( 0.233,  0.467,  0     )
T_2 δ = 2 L̃ T_1 δ − δ   = (−0.078,  0,     −0.622 )
```

With `K = 2` (terms `T_0, T_1`), site C's features are blind to the cloud: its entry in `T_1 δ` is 0. With `K = 3`, C receives a strong signal (`−0.622`) from two hops upwind. In GCLSTM, even a `K = 2` filter reaches C one recurrent step later, because B's hidden state already carries the news. Note also that the polynomial filter is symmetric: B "hears" A and C alike. The direction of the wind is not in the graph; it has to be learned from the time series, and partly from the learned Laplacian weights.

**How it connects**

X002 argued that wind and solar forecasts should be distributions and, for coupled decisions, space-time *trajectories*. This paper builds the spatio-temporal dependence but outputs only a point forecast; the authors list a probabilistic extension as future work. X011 (DeepAR) is the sampling-based temporal model without the graph, and X020's review places this kind of work under "machine learning" and "spatio-temporal" forecasting. Compare X013: there, the graph of the electrical grid did not help learn the OPF map, because that map is global. Here the graph *does* help, because cloud advection is genuinely local in space and time. This is the same lesson from the other side: a graph prior pays off when the target function is local on that graph. The current focus phase continues with A16 (nonconvexity and local optima of AC-OPF). Later, B13–B20 turn such forecasts into calibrated distributions and space-time scenarios (B20 on PV trajectories in particular), C19–C22 cover the sequence models, STGCN/GCRN, DCRNN with adaptive adjacency and the taxonomy of spatio-temporal graph learning, and C23–C24 apply the same template to multi-period OPF itself.

**Check yourself**

1. Why is the kNN graph of PV sites a better match for a GNN inductive bias than the electrical grid graph was for the OPF map in X013?
2. With a Chebyshev filter of `K` terms, how far can information travel in one layer, and how does GCLSTM reach farther without increasing `K`?
3. Why does the linear STAR model match the GNNs at 15 minutes but fall behind at 3–6 hours, and why does NWP win again at 5–6 hours?

<details><summary>Answers</summary>

1. The target, future PV output at a site, depends mostly on clouds that are currently near that site and upwind of it, so the function is local on a proximity graph. In X013 the OPF set-points depended on the total load through a single system price and on dense PTDF couplings, so the function was global on the grid graph, and a local GNN needed a global readout.
2. `K − 1` hops, since `T_k(L̃)` is a degree-`k` polynomial in a sparse matrix. GCLSTM applies a graph convolution to the hidden state at every recurrent step, so information spreads a few more hops at each 15-minute step. Over a 16-step encoder, the receptive field covers far more of the graph than one layer.
3. At 15 minutes, the best predictor is close to persistence with small, mostly linear corrections, which an autoregression captures well. At longer horizons, the clouds that matter come from farther away and change shape, and nonlinear spatio-temporal features help. At 5–6 hours, the clouds that will matter have not yet reached any PV site, so production data cannot see them, while NWP models the weather that has not arrived yet.

</details>
