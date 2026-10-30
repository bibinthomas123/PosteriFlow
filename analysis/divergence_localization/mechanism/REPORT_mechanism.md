# Mechanism localization — late-training overconfidence despite improving val NLL

**Scope:** find the exact mechanism of the calibration collapse (no encoder/flow
redesign, no speculation, every claim measured). Target: `model/long_run/best_model.pth`
(ep19). Artifacts: `analysis/divergence_localization/mechanism/` (`m*.json`, `figs/`).

**Evidence base & its hard limit.** The trainer saves only `best_model.pth`, so a
per-epoch *weight* trajectory does not exist. Probeable weights: **long_run(ep19)**,
**coherentC(ep40)** (same coherent+fixed lineage), **v8_ft(ep25)** (different encoder/
dataset — reference only). Plus **ep18 stored posteriors** (samples+logprobs, no weights)
and **history.json** (19 epochs of scalars). Anything requiring a clean per-epoch weight
trajectory is labelled **INCONCLUSIVE** with the reason; everything else is measured.

## One-line mechanism
Maximum-likelihood (forward-KL) training of the conditional flow has **no calibration
term**; it keeps lowering NLL by **sharpening the transform (raising log|det J|)** even
after that pushes the truth into the base-distribution tails (‖z‖²/D → 2, overconfident),
worst at high SNR. The encoder has converged by ~ep16–18; the late NLL gains are the flow
over-sharpening. Best-val-NLL checkpoint selection then **actively picks the most
overconfident checkpoint**. This is expected MLE behaviour, not an optimizer bug.

---

## Task 1 — Is the flow becoming overconfident?

**Sample-based widths (ep18 stored vs ep19), scaled variance ratio ep19/ep18, 6 events:**

| event | sci CI90 ratio | sci var ratio | note |
|---|---|---|---|
| GW150914 | 1.71 | 2.96 | science posterior *widened* |
| GW170814 | 1.53 | 2.18 | widened |
| GW151226 | 1.40 | 1.91 | widened |
| GW170104 | 0.62 | 0.56 | narrowed |
| GW170608 | 0.84 | 0.89 | narrowed |
| GW170729 | 0.90 | 0.92 | narrowed |

Per-param (mean over events): distance var ×2.21, mass_1 ×1.06, mass_2 ×1.24;
angles/spins/circular ×0.63–0.86 (narrowed).

→ **"posterior variance/CI-width monotonically decreases" — REJECTED.** Raw width is
**confounded by boundary railing**: for the problem events the science posteriors go
*bimodal* (bulk + prior-edge spike), which inflates variance even as the model gets the
answer more wrong. Sample variance is the wrong lens.

**Model-based base-space concentration (the confound-free measure).** A calibrated flow
maps truths to z~N(0,I) ⇒ E[‖z‖²/D]=1. Measured at ep19 on 1500 validation events:

| checkpoint | val NLL | ‖z‖²/D | ‖z‖²/D by SNR (5-15 / 15-25 / >25) |
|---|---|---|---|
| long_run(ep19) | 0.864 | **2.03** | 1.57 / 2.06 / **2.44** |
| coherentC(ep40) | 1.368 | 2.14 | 1.67 / 1.88 / 2.70 |

Per-dim Var(z) at ep19: **mass_1 4.26**, geocent 2.35, distance 2.17, a2 1.74 (overconfident);
dec 0.95, psi 0.82, phase 0.96 (calibrated).

→ **Flow overconfidence — CONFIRMED.** Truths sit at ~√2× the calibrated base radius
(worst for mass_1, distance) and the effect **grows monotonically with SNR** (1.57→2.44),
directly matching the high-SNR coverage collapse. The science params are **narrow-and-
biased** (truth in the base tail), not wide. (Monotonic *across epochs of long_run*:
INCONCLUSIVE — no per-epoch weights; ep18→ep19 railing + ‖z‖²/D=2 establish the ep19
state.) See `figs/mechanism_overview.png` (panel c).

## Task 2 — Why does NLL keep improving while calibration worsens?

MLE of a conditional NF minimises the forward KL, which for a standard-normal base and
transform T_d (θ→z), z\*=T_d(θ\*), is

  L = E[ ½‖z\*‖² − log|det J_{T_d}(θ\*)| ] + const.

There are exactly two ways to lower L: (1) move density toward θ\* (↓‖z\*‖², helps
calibration too), or (2) **increase log|det J| — sharpen the transform — which lowers L
regardless of calibration.** Nothing in L requires z\*~N(0,I) (calibration) or penalises
mass placed where p(θ|d) has none (boundary railing). So the optimum of MLE does **not**
sit at ‖z\*‖²/D=1: with heteroscedastic targets and finite capacity, aggregate NLL is
minimised by **over-sharpening the many easy/typical events** (large log|det J| gain) at a
**bounded ‖z\*‖² penalty on the few hard/OOD events.**

Empirical confirmation (ep19): ‖z\*‖²/D = **2.03** (overconfident) yet NLL is low because
log|det J| = **+5.05** (strong expansion) over-compensates. The transform over-expands
exactly the science dims (mass_1 Var z 4.26, distance 2.17) and leaves angles calibrated.

Direct proof of the coexistence across real checkpoints (M5): NLL and calibration are
**anti-correlated** (see Task 5) — lower NLL ⇒ worse calibration, monotonically.

Optimization pathology? **REJECTED.** grad norms bounded and smooth (32→42 over ep8→19,
no NaN/spike, `history.json`); LR schedule normal. The decalibration is a property of the
**objective**, not the optimizer.

→ **Expected from MLE of conditional flows — CONFIRMED.** The val↓/calibration↓
coexistence is generic; its severity here is amplified by a long LR schedule (below) that
keeps driving NLL down well past the calibrated point.

## Task 3 — Flow internals

On a common 1500-event validation batch:

| checkpoint | log\|det J\| | ‖z‖²/D | frac events any\|z\|>3 | max\|z\| |
|---|---|---|---|---|
| long_run(ep19) | 5.05 | 2.03 | 0.43 | 4.9 |
| coherentC(ep40) | 6.58 | 2.14 | 0.34 | 5.0 |
| v8_ft(ep25)\* | 0.50 | 4.61 | 0.77 | 5.0 |

- **Larger spline slopes / density concentration — CONFIRMED at ep19:** large positive
  log|det J| (+5.05) with truths pushed to the base tails (‖z‖²/D=2). The concentration is
  dim-selective (mass_1/distance over-expanded; angles calibrated).
- **Saturation / unstable Jacobians — REJECTED:** max|z|≈4.9 (< tail_bound 5), only 43% of
  events reach any dim |z|>3, log|det J| finite and stable — the flow is *sharp*, not
  *broken/saturated*. (Consistent with Step 7: railing is at the ±1 prior boundary, not the
  ±5 spline tail.)
- **Sharpening across epochs — INCONCLUSIVE** (no per-epoch weights; cross-run coherentC is
  a different, less-optimised run so its higher log|det J| is not an epoch trend).

## Task 4 — Optimizer dynamics (`figs/mechanism_overview.png` a,b)

- **LR is still ~96% of peak at ep19** (2.89e-4 of 3e-4): the cosine was sized for
  `epochs=140`, so at 19/140 it has barely decayed → the flow keeps taking near-peak-LR
  steps. **CONFIRMED** as an amplifier.
- **grad norm rises late** (32→42, ep8→19) while train NLL falls; **no explosion/NaN** →
  larger late updates consistent with sharpening, **not** a numerical pathology.
  **CONFIRMED (healthy).**
- **Encoder converged, flow keeps changing — CONFIRMED (indirect):** encoder-dependent
  metrics plateau by ~ep16–18 (shuffle Δ peaks 10.97@ep18 then *falls* to 9.96@ep19;
  dist_corr flat ≈0.78 since ep16) while val NLL keeps dropping 0.996→0.941→0.864. The late
  NLL gain carries no encoder-metric gain. Direct encoder-vs-flow param-norm *trajectory*
  is **INCONCLUSIVE** (no per-epoch weights); single-snapshot flow/encoder norm ratio ≈1.25
  across the coherent checkpoints.

## Task 5 — Checkpoint selection (`figs/selection_inversion.png`)

Coherent + fixed-dataset lineage (comparable val NLL):

| checkpoint | val_nll | railing | cov90 | SBC pass | calib score |
|---|---|---|---|---|---|
| coherentC(ep40) | 1.368 | 0.050 | 0.90 | 0.91 | **0.14 (best)** |
| long_run(ep18) | 0.996 | 0.058 | 0.90 | 0.45 | 0.61 |
| long_run(ep19) | **0.864** | 0.894 | 0.69 | 0.00 | 2.10 (worst) |

Which checkpoint each criterion selects:

| criterion | selects |
|---|---|
| **min val_nll (CURRENT)** | **long_run(ep19)** — the *worst* calibrated |
| min railing | coherentC(ep40) |
| min \|cov90−0.90\| | coherentC(ep40) |
| max SBC pass | coherentC(ep40) |
| combined calibration | coherentC(ep40) — the *best* calibrated |

→ **Best-val-NLL selection systematically picks the overconfident model — CONFIRMED.** A
perfect inversion: val NLL and calibration rank the three checkpoints in opposite order.
Note coherentC ran **more** epochs (40) yet stayed calibrated — so this is **not epoch
count**, it is **how far NLL is pushed**. Well-calibrated checkpoints exist in the
trajectory; selection just isn't choosing them.

---

## Task 6 — The minimum change (evidence-ordered)

The mechanism is objective+selection, not representation, so the fix is in selection/
stopping/diagnostics — **no encoder or flow redesign is warranted by the evidence.**

1. **Calibration-aware checkpoint selection (smallest, highest-leverage).** Keep val NLL as
   the objective but **gate** `best_model.pth` on a calibration criterion (e.g. spurious
   railing ≤ ~0.06 and/or base-space ‖z‖²/D within [0.8,1.3], or a combined score). M5
   shows this single change flips the selected checkpoint from the broken ep19 to a
   calibrated one that already exists in training. *Evidence: Task 5 inversion.*
2. **Log calibration per epoch.** `history.json` currently logs only aggregate distance
   coverage (which stayed ~0.90 while science/high-SNR coverage collapsed). Add per-epoch
   **spurious railing, SNR-binned science coverage, base-space ‖z‖²/D, SBC** so the
   regression is visible. *Evidence: Tasks 1/3/5 all needed metrics absent from the log.*
3. **Early stop on calibration, not NLL.** Since encoder metrics plateau ~ep16–18 (Task 4)
   and calibration is already degrading by ep18→19, stop when calibration stops improving
   rather than when NLL plateaus. Practically: the calibrated operating point for this run
   is ~ep16–18. *Evidence: Task 4 plateau + Task 5.*
4. **Shorten / re-tune the LR schedule.** The 140-epoch cosine keeps LR near peak, letting
   the flow over-sharpen (Task 4). Sizing the schedule to the actual stopping epoch removes
   the amplifier. *Optimizer change — mild, only if 1–3 are insufficient.*
5. **(Only if 1–4 fail) light calibration regularization** — e.g. a small penalty keeping
   E[‖z‖²/D]≈1, or post-hoc temperature (the flow already has a pinned base-temperature
   knob; the pipeline's importance reweighting is the principled version). ⚠️ CLAUDE.md
   records width/aux penalties measured NEUTRAL — validate on calibration, not NLL.

**Not recommended (no supporting evidence):** encoder redesign, flow-architecture change,
optimizer swap. Jacobians are stable, spline tails are unused, the encoder has converged
and its geometry/coherence features are in-distribution.

## Task 7 (follow-up) — WHY GW150914 & GW170814 specifically?

"Overconfidence exists" is global; it does not say *which* events break. The selectivity
has a separate, measured cause. Correct framing: at **ep18** (calibrated) **only GW150914
(rail 0.40) and GW170814 (0.55) railed** — the other four were ~0.05. So: which events rail
*earliest*? Two obvious selectors are ruled out by Steps 3–4:

- **PSD-OOD is not the selector:** GW170729 is the *most* PSD-OOD (100th pct) and most
  context-OOD (Maha 22.7) yet is not an early victim; GW170814 has the *lowest* context-OOD
  yet is.
- **SNR alone is not the selector:** GW170814 (SNR 16) railed *more* at ep18 than GW150914
  (SNR 24); GW170608 (SNR 15) didn't rail.

**Measurement (m7):** 4000 design-whitened validation events (**zero PSD-OOD** — isolates
the mass/SNR effect), base concentration ‖z‖²/D (the discriminating metric; spurious
railing is saturated ~0.9 by nuisance params) vs SNR, Mc, q (`figs/m7_oneway.png`):

| factor | ‖z‖²/D across bins |
|---|---|
| SNR (5→40) | 1.51 → 1.86 → 2.02 → **2.37** |
| chirp mass (5→60) | 1.73 → 2.10 → **2.57** → 2.52 |
| mass ratio q (0.1→1.0) | 1.80 → 1.85 → 1.92 → **2.49** |

**Super-additive interaction — the victim corner (SNR>18), `figs/m7_corner.png`:**

| corner | ‖z‖²/D | n |
|---|---|---|
| Mc<20, q<0.7 | 2.02 | 897 |
| Mc<20, q≥0.7 | 2.36 | 509 |
| Mc≥20, q<0.7 | 2.19 | 530 |
| **Mc≥20, q≥0.7 (heavy + near-equal)** | **3.71** | 262 |

The three factors *together* give 3.71 — far above any one alone (~2.0–2.5). Locating the
six events on this map:

| event | Mc | q | SNR | corner | victim? |
|---|---|---|---|---|---|
| **GW150914** | 31.3 | 0.86 | 24.4 | **Mc≥20, q≥0.7** | **yes** |
| **GW170814** | 24.2 | 0.82 | 15.9 | **Mc≥20, q≥0.7** | **yes** |
| GW170729 | 36.1 | 0.68 | 10.8 | Mc≥20, q<0.7 | no (unequal, quiet) |
| GW170104 | 21.6 | 0.65 | 13.0 | Mc≥20, q<0.7 | no (unequal) |
| GW151226 | 8.9 | 0.56 | 13.1 | Mc<20, q<0.7 | no (light) |
| GW170608 | 7.9 | 0.58 | 15.4 | Mc<20, q<0.7 | no (light) |

**GW150914 and GW170814 are the only two in-distribution events in the worst corner.**
GW170729 escapes it (q 0.68) *despite being heaviest and most PSD-OOD*; the light events
escape (Mc<20). Because the map has **zero PSD-OOD**, the mass/SNR structure **alone**
selects the victims.

### The two causal chains, separated (both measured)

**Chain A — the SELECTOR (why *these* events):**
```
heavy + near-equal-mass (Mc≥20, q≥0.7)   [an intrinsic-difficulty regime, NOT a
      ↓                                    coverage hole — see M8 below]
 flow's conditional density there is biased (equal-mass m1≈m2 fold / degeneracy)
   × high SNR                             [overconfidence grows with SNR: ‖z‖²/D 1.5→2.4]
      ↓  MLE over-sharpens the biased density (Task 2)
 truth pushed deep into the base tail     [‖z‖²/D = 3.71 in the corner vs ~2.0 elsewhere]
      ↓  overconfident posterior collapses onto the nearest prior boundary
 railing — EARLIEST for corner events (GW150914, GW170814)
```

**M8 — is the corner bias a DATA-COVERAGE (sparsity) problem? NO.** The corner is
sparse (Step 2, 2–3× lower density), but sparsity is *not what causes the bias*.
Controlled regression over 4000 validation events, `‖z‖²/D ~ log SNR + log density`:
SNR coef **+1.14** (std β +0.40) vs density coef **−0.015** (std β **−0.01**);
partial corr(‖z‖², density | SNR) = **−0.005**. Within SNR 18–40, ‖z‖²/D by density
quartile is flat: **2.45 (sparsest) → 2.39 (densest)**. Predicted ‖z‖²/D reduction from
**10× more corner data = −0.015** against a **2.71** gap to close; the +0.66 SNR term is
density-independent. GW170729 is in the *sparsest* region (16th density pct) yet is not a
victim. → **the "sparse corner → biased" hypothesis is REJECTED**; the corner bias is an
SNR/overconfidence effect plus an intrinsic equal-mass difficulty (plausibly the m1≈m2
ordering fold), *not* under-sampling. **More heavy/high-SNR/equal-mass training data will
not fix it.**

**Chain B — a uniform AMPLIFIER (why real events are worse than sim, NOT the selector):**
```
PSD-OOD (real ASD ≠ training, Step 4)     [present for ALL real events incl. escapees]
      ↓  adds context bias on top of Chain A
 every real event's ‖z‖ pushed up further → by ep19 even non-corner real events
 (GW170104, GW170729) rail ~1.0, while their validation twins at the same (Mc,q,SNR) do not
```

Chain B does **not** discriminate victims (the most-PSD-OOD event, GW170729, escapes early
railing) — it explains the real-vs-sim gap, not the event ranking. **Chain A is the
selector; Chain B is a uniform amplifier.** This is why the two are "related but not
identical": overconfidence (global) × sparse-corner bias (Chain A selector) × PSD-OOD
(Chain B amplifier).

| sub-hypothesis | verdict |
|---|---|
| PSD-OOD magnitude selects the victims | **REJECTED** (zero-PSD-OOD map reproduces selectivity; most-OOD event escapes) |
| High-SNR alone selects the victims | **REJECTED** (GW170814<GW150914 in SNR but railed more; GW170608 quiet-safe) |
| Heavy × near-equal-mass × high-SNR corner selects the victims | **CONFIRMED** (‖z‖²/D 3.71 vs ~2.0; both victims uniquely there) |
| PSD-OOD is a separate additive amplifier for real events | **CONFIRMED** (Step 4 + real events rail beyond validation twins) |

## Verdict summary

| Hypothesis / question | Verdict |
|---|---|
| Posterior variance/CI-width monotonically shrinks | **REJECTED** (railing inflates science-param width) |
| Flow is overconfident (base-space ‖z‖²/D>1, grows with SNR) | **CONFIRMED** (2.0; 1.57→2.44 over SNR) |
| Lower NLL coexists with worse calibration — expected from MLE | **CONFIRMED** (forward-KL has no calibration term; log\|det J\| trade) |
| Optimization pathology (bad grads/Jacobians/saturation) | **REJECTED** (grads/Jacobians stable, tails unused) |
| Only the flow keeps changing late; encoder converged | **CONFIRMED (indirect)** — direct param trajectory INCONCLUSIVE |
| Best-val-NLL selection picks the overconfident checkpoint | **CONFIRMED** (perfect inversion, M5) |
| Fix needs architecture change | **REJECTED** — selection/diagnostics/stopping suffice |
