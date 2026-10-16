## 2026-10-16 (Friday) — A06 · Track 01: Grid physics & power flow — DC power flow: the linear grid model behind markets, PTDFs, and where it breaks

**Type:** Focus lesson
**Main reference:** [DC Power Flow Revisited — B. Stott, J. Jardim & O. Alsaç (2009), IEEE Transactions on Power Systems 24(3), pp. 1290–1300](https://doi.org/10.1109/TPWRS.2009.2021235) — read the whole paper (11 pages), and focus on how the different DC model variants are built, the "hot-start" vs "cold-start" distinction, how losses are handled, and the conclusions on accuracy. The paper is not open access. If you cannot get it, the MATPOWER manual below covers the derivation and the PTDF construction (but not the paper's critical assessment).
**Further reading:**
- [MATPOWER User's Manual, v8.1 — R. D. Zimmerman & C. E. Murillo-Sánchez (2025)](https://matpower.org/docs/MATPOWER-manual.pdf) — §3.7 "DC Modeling" (pp. 31–34): the three assumptions and the matrices `B_bus` and `B_f`. §4.5 "Linear Shift Factors" (pp. 46–47): PTDFs, slack distributions and line outage distribution factors (LODFs).
- [Power Flow Review and Shift Factors — PSERC education module (Sloan series, module 1)](https://documents.pserc.wisc.edu/documents/pserc_education/sloan/module1/Shift_Factors_Sloan.pdf) — slides that derive the linearized power flow from the AC line equation and work through shift factors with numbers.

**Why this matters** — Most of the optimization-under-uncertainty literature for power systems (chance-constrained OPF, robust dispatch, market clearing, congestion forecasting) runs on the DC model. It is linear, so uncertainty in the injections maps to uncertainty in the line flows through a single matrix. Anyone who trains, benchmarks or constrains a learned dispatch model needs to know what the DC model keeps, what it throws away, and which failures it cannot see.

**Core ideas**

**1. Four assumptions turn AC into linear algebra.** Start from A04's flow on a line `i→k` with series admittance `g + jb`:

```
P_ik = V_i² g − V_i V_k (g cos θ_ik + b sin θ_ik),     θ_ik = θ_i − θ_k
```

Now assume (a) `r ≪ x`, so `g ≈ 0` and `b ≈ −1/x`; (b) `V_i ≈ V_k ≈ 1` pu; (c) small angle differences, so `sin θ_ik ≈ θ_ik`; and (d) ignore reactive power entirely. What remains is

```
P_ik ≈ (θ_i − θ_k) / x_ik
```

This is Ohm's law for a resistor network: angle plays the role of voltage, active power the role of current, and reactance the role of resistance. (MATPOWER also carries tap ratios and phase-shift angles through, see §3.7. They only add a constant offset.)

**2. Network form: the Laplacian again.** Summing over lines at each bus gives `P = B θ`. Here `B` is the weighted graph Laplacian from A03 with edge weights `1/x_ik`, and `P` is the net injection vector. `B` is singular (constant vectors are in its null space), which mirrors the physics: only angle differences matter. So fix `θ_slack = 0`, delete that row and column, and solve:

```
θ_red = B_red⁻¹ P_red,          f = B_f θ          (B_f = diag(1/x) · A,  A = branch–bus incidence)
```

This takes one sparse linear solve. There is no iteration, no convergence failure and no second root (contrast A05). The solution always exists and is unique, even at injections where the real AC system has no solution.

**3. PTDFs: the whole grid as one matrix.** Combine both steps:

```
f = H P,         H = B_f B_red⁻¹   (with a zero column for the slack)
```

`H` is the **power transfer distribution factor** (PTDF) matrix, also called shift factors. `H_lj` is the change in flow on line `l` when one extra MW is injected at bus `j` and withdrawn at the slack. Three properties do most of the work in later lessons:
- *Superposition.* Flows from many transactions or many forecast errors simply add up. A transfer from bus `a` to bus `b` moves `H_{:,a} − H_{:,b}` per MW.
- *Slack dependence.* Each PTDF column depends on where the MW is withdrawn. With a distributed slack `w` (weights summing to 1), `H_w = H_k (I − w 1ᵀ)` (MATPOWER eq. 4.37). For any *balanced* change (`1ᵀ ΔP = 0`), all versions give the same answer.
- *Graph meaning.* With `L⁺` the Laplacian pseudo-inverse, the flow on line `(i,k)` per MW sent from `a` to `b` is `(1/x_ik)·[(L⁺_ia − L⁺_ib) − (L⁺_ka − L⁺_kb)]`. These are effective-resistance quantities. Power flows along *every* path in parallel, in proportion to how electrically easy each path is, and no operator can route it along a chosen path. This "loop flow" is why one wind farm can congest a line in a neighbouring country.

Line outages follow from the same matrix. If `h̃_ij` is the sensitivity of line `i`'s flow to a transfer between line `j`'s endpoints, the line outage distribution factor is `LODF_ij = h̃_ij / (1 − h̃_jj)`. This makes N−1 screening (A01's security criterion) a sequence of rank-one updates and avoids thousands of new power flows.

**4. Why the paper exists: "the DC model" is many models.** The abstract notes that DC models are "in widespread and even increasing use, particularly in congestion-constrained market applications", but that their MW flows are "by no means assured" to be correct. Every DC model makes modelling choices, and the paper examines them:
- *Cold start vs hot start.* A cold-start model linearizes around the flat profile (`V = 1`, `θ = 0`) and needs nothing but network data. A hot-start model is built around a solved AC base case and predicts *changes* (`f ≈ f_AC,base + H ΔP`). It is far better near that base case, but it is tied to it.
- *Which susceptance.* `1/x` or the exact `−Im(1/(r + jx)) = x/(r² + x²)`. These differ when `r/x` is not small.
- *Losses.* A lossless model makes the slack pick up nothing. Real systems lose a few percent, and how losses are allocated (fictitious loads, loss factors) shifts the flows.

The lesson for ML: when a paper reports "DC-OPF" or uses a "DCPF baseline", ask which variant it is.

**5. Where it breaks.** Each assumption fails in a known regime:
- *High r/x* (distribution feeders, cables): `g` is no longer negligible, losses become large, and A02's P–δ / Q–V decoupling weakens.
- *Stressed systems*: large angle differences break `sin θ ≈ θ`. Depressed voltages break `V ≈ 1`.
- *Voltage- and reactive-driven congestion*, the blind spot. Thermal limits are limits on *current*, `|I| = |S|/V` with `|S| = √(P² + Q²)` (A07 defines them properly). A line carrying large reactive power, or one feeding a low-voltage area, can overload while its active-power flow is moderate. The DC model has no Q and no V, so it can never flag this. A dispatch that is "DC-feasible" can still violate voltage bands and thermal limits in AC.

**A small example**

Take a triangle with `x_12 = x_13 = 0.1` pu and `x_23 = 0.2` pu. Bus 1 is the slack, bus 2 a generator injecting 1 pu, and bus 3 a load of 1 pu. The DC model gives

```
B_red = [[15, −5], [−5, 15]],   B_red⁻¹ = (1/200)[[15, 5], [5, 15]]
θ_2 = +0.05 rad (2.86°),  θ_3 = −0.05 rad
f_12 = −0.5,  f_13 = +0.5,  f_23 = +0.5
PTDF columns (bus 2, bus 3):   line 1–2: (−0.75, −0.25)   line 1–3: (−0.25, −0.75)   line 2–3: (0.25, −0.25)
```

Half the transfer goes directly over line 2–3. The other half detours through the slack bus along the path 2→1→3, because that path has the same total reactance. Now compare with an AC Newton solve (bus 2 held at `V = 1`). Each row adds resistance or reactive load:

| case | P on 1–3 | \|S\| on 1–3 | V_3 | losses |
|---|---|---|---|---|
| DC model | 0.500 | 0.500 | 1 (assumed) | 0 |
| AC, r/x = 0, Q_3 = 0 | 0.501 | 0.502 | 0.998 | 0 |
| AC, r/x = 0.1, Q_3 = 0.3 | 0.510 | 0.573 | 0.970 | 0.011 |
| AC, r/x = 0.5, Q_3 = 0.3 | 0.540 | 0.631 | 0.941 | 0.061 |
| AC, r/x = 0.1, Q_3 = 1.5 | 0.537 | 1.313 | 0.876 | 0.032 |

The active power flow is never off by more than 8 %, which is why DC models look good in MW-error plots. But if line 1–3 is rated at 1.0 pu, the DC model reports 50 % loading in the last row, while the line actually carries 131 % and bus 3 sits far below a 0.95 pu voltage limit. That is reactive-driven congestion, and the linear model has no way to see it.

**How it connects**

A02 derived the P–δ / Q–V decoupling. This lesson keeps only the P–δ half. A03 showed that `B ≈ −`(Laplacian with weights `1/x`), and here that Laplacian is the entire model. A04's equations and A05's Newton solver give the AC reference that the DC model approximates. The DC solution is also what pandapower uses to initialize Newton. X004's DCPF baseline was a cold-start DC model, and it held up out of distribution precisely because nothing in it is learned. X008's rule "flow = schedule + error" is the PTDF relation `f = H(P + ε)`, and with Gaussian injection errors `ε ~ N(0, Σ)` the flow errors are `N(0, H Σ Hᵀ)`. Phase B's chance constraints build on exactly this. Next: A07 makes the operating limits precise (current vs apparent power, voltage bands, generator capability), A08 adds costs to get economic dispatch, and A23 revisits linear approximations that keep voltage and reactive power (LPAC).

**Check yourself**

1. A trader computes flows for a transfer of 100 MW from bus `a` to bus `b` with a PTDF built for slack bus 1. A colleague uses a distributed slack. Do their answers differ? What changes if the trade is a 100 MW injection at `a` with no matching withdrawal?
2. Injection forecast errors are `ε ~ N(0, Σ)`. Under the DC model, what is the distribution of the flow on line `l`, and why is this result unavailable in closed form for AC power flow?
3. A learned dispatch policy is trained with a DC-flow penalty in its loss and achieves zero DC violations on test data. Name two ways its decisions can still be infeasible in AC.

<details><summary>Answers</summary>

1. No. The transfer is balanced (`1ᵀΔP = 0`), and `H_w ΔP = H_k (I − w1ᵀ) ΔP = H_k ΔP`. For an unbalanced injection, the extra 100 MW has to be withdrawn somewhere, and the slack model decides where. The resulting flows then differ.
2. The flow on line `l` is `f_l ~ N(f̄_l, h_lᵀ Σ h_l)`, with `h_l` the `l`-th row of `H`. A chance constraint `P(f_l ≤ f̄^max) ≥ 1 − α` becomes the deterministic constraint `f̄_l + z_{1−α} √(h_lᵀ Σ h_l) ≤ f̄^max`. In AC the map from injections to flows is nonlinear and only implicitly defined (A05), so a Gaussian input does not give a Gaussian output, and the output distribution has no closed form.
3. (a) Thermal overloads from reactive flow or low voltage: `|S|` or `|I|` exceeds the rating while `P` is within it (the last row of the table). (b) Voltage-band violations, which the DC model does not represent. Also: losses change the slack's output and the flows; and at stressed points the AC power flow may have no solution at all (A05), a case the DC model always "solves".

</details>
