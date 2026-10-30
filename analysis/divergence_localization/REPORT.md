# LeanNPE divergence localization — validation improves while GWTC events worsen

**Model under study:** `model/long_run/best_model.pth` — **epoch 19**, val_nll 0.8636,
`encoder_type=coherent`, `psd_cond=True`, `real_noise_prob=0.9`. Confirmed by the user
to be "the best model so far." All model-based results are on this checkpoint
(snapshotted to scratch; live and snapshot verified byte-identical).

**Bottom line (localized):** The divergence is **not** a data-coverage problem and
**not** primarily an encoder-OOD problem. It is a **confidence-calibration collapse of
the conditional flow in late training** (measurable between the stored epoch-18 state and
the current epoch-19 checkpoint). The flow becomes **overconfident** — posteriors narrow
and dump probability mass at the parameter prior boundaries — **worst at high SNR**, and
this happens **on in-distribution simulated validation just as much as on real events**.
`val_nll` keeps improving because NLL rewards density *at the truth* and is blind to the
boundary mass; the metrics that would catch it (spurious railing, SBC, coverage) are **not
in the per-epoch log** and the CI was last run at epoch 18. GW150914/GW170814 are the
visible victims only because they are the loudest/heaviest, mildly-OOD in-distribution
events — exactly where overconfidence bites hardest.

---

## Caveats on the evidence base (read first)

1. **No per-epoch checkpoints exist.** The trainer saves only `best_model.pth` (best val
   NLL). A clean "epoch-15 → latest" trajectory of one run is therefore **not
   reconstructable**. The within-run signal I *do* have is the stored **epoch-18 CI
   posteriors** (`analysis/ci/long_run_ep18/`) vs the current **epoch-19** checkpoint —
   two points, identical events, identical preprocessing → a clean ep18→ep19 delta.
2. **The `model/` directory churned during the session.** `model/long_finetune/` (present
   at session start and in the git snapshot) was deleted mid-analysis, and
   `long_run/history.json` shrank from 51 to 19 epochs. An external process is rewriting
   `model/`. I snapshotted every artifact before use.
3. **`real` in the per-epoch `history.json` = real O3 *noise* domain**, not the 6 GWTC
   *events*. The GWTC events are only evaluated by `validate_checkpoint`, whose last run
   for this lineage was epoch 18 — so the per-epoch log cannot, by construction, show
   GW150914 worsening.
4. **SNR convention.** Network optimal SNR of a whitened signal = its L2 norm
   (`remix_data.py`). Verified: the per-signal L2-norm distribution matches the stored
   measured `network_snr` to <5% (median 14.7 vs 15.2). GWTC event SNRs are GWTC-1 catalog
   values (annotations only).

---

## STEP 1 — Training SNR distribution audit → **REJECTED**

Per-signal network SNR over all 60,306 training signals (`figs/step1_snr.png`):

| stat | median | 90th | 95th | 99th | max |
|---|---|---|---|---|---|
| per-signal SNR | 14.7 | 52 | 80 | 170 | **934** |
| rank-0 (loudest-per-event) | 19.5 | 68 | 103 | 204 | 934 |

Counts above thresholds (per-signal): >10: 41,350 (69%); >15: 29,533 (49%); >20: 22,323
(37%); >25: 17,513 (29%); >30: 14,200 (24%).

| event | SNR | %ile (per-signal) | %ile (rank-0) | rank-0 nbrs ±2 SNR | signals louder |
|---|---|---|---|---|---|
| GW150914 | 24.4 | 70.2 | 59.3 | 2,384 | 17,999 |
| GW170814 | 15.9 | 53.5 | 41.0 | 4,268 | 28,026 |
| GW170608 | 15.4 | 52.2 | 39.6 | 4,460 | 28,850 |
| GW151226 | 13.1 | 44.6 | 32.2 | 5,331 | 33,383 |
| GW170104 | 13.0 | 44.2 | 31.8 | 5,395 | 33,634 |
| GW170729 | 10.8 | 35.2 | 23.3 | 6,191 | 39,074 |

**Reasoning / verdict.** The two worsening events sit at the **41st–70th percentile** with
**thousands** of rank-0 neighbors within ±2 SNR and **18k–28k louder** training signals.
The genuinely sparse region is SNR > ~80 (95th pct); **none** of the six events are there.
Loud-event SNR sparsity is **REJECTED** as the cause.

---

## STEP 2 — Heavy-BBH coverage audit → **CONTRIBUTING / INCONCLUSIVE (density gradient, not a hole)**

Detector-frame joint distribution (`figs/step2_pairplot.png`, `figs/step2_mc_snr_density.png`).
Marginal percentiles and local joint counts:

| event | Mc | Mtot | q | %Mc | %Mtot | %q | (Mc,SNR) box | (Mtot,q) box |
|---|---|---|---|---|---|---|---|---|
| GW150914 | 31.3 | 72.2 | 0.86 | **90** | **82** | **80** | 811 | 541 |
| GW170814 | 24.2 | 56.0 | 0.82 | **82** | **72** | **76** | 1002 | 580 |
| GW170729 | 36.1 | 84.9 | 0.68 | 93 | 88 | 61 | 324 | 670 |
| GW170104 | 21.6 | 51.1 | 0.65 | 80 | 69 | 59 | 1138 | 767 |
| GW151226 | 8.9 | 21.4 | 0.56 | 50 | 42 | 51 | 1376 | 1405 |
| GW170608 | 7.9 | 19.0 | 0.58 | 45 | 38 | 53 | 1225 | 1492 |

**Reasoning / verdict.** GW150914/GW170814 share the **heavy + near-equal-mass** corner
(Mc 82–90th, q 76–80th), where local density is **2–3× lower** than for the light events
(~800–1000 vs ~1400 neighbors) — but **not a coverage hole** (hundreds of neighbors
remain). Decisively: **GW170729 is heavier (Mc 93rd) and genuinely OOD in distance
(dL 2750 > training max ~2000), yet is not one of the worsening events.** So "heaviest /
most-OOD" does **not** select the worseners — heavy-BBH underrepresentation **modulates
severity** but is not the trigger. **INCONCLUSIVE as root cause; CONTRIBUTING.**

---

## STEP 3 — Real-event trajectory → **degradation is abrupt & flow-driven (encoder OOD is stable)**

**Generation proxy** (`figs/step3_trajectory.png`, `step3_trajectory.json`): across model
generations the **OOD percentile of every event rises monotonically 60→100**, but railing
and point estimates are oscillatory (different arch/data — a proxy, not a clean trajectory).

**Clean within-run ep18→ep19 delta** (identical events & preprocessing):

| event | rail 18→19 | Mahalanobis 18→19 | m1 18→19 (pub) |
|---|---|---|---|
| GW150914 | 0.40 → **0.99** | 18.25 → 17.95 | 50 → 64 (39) |
| GW151226 | 0.05 → **0.72** | 17.74 → 15.95 | 16 → 21 (14) |
| GW170104 | 0.04 → **1.00** | 18.45 → 18.93 | 65 → 85 (31) |
| GW170608 | 0.04 → **0.64** | 17.70 → 16.53 | 22 → 26 (12) |
| GW170729 | 0.05 → **1.00** | 22.71 → 20.21 | 77 → 88 (51) |
| GW170814 | 0.55 → **1.00** | 15.39 → 16.67 | 31 → 36 (31) |

**Reasoning / verdict.** In ~1–2 epochs, railing exploded for **all** events while the
encoder **OOD Mahalanobis stayed flat (even fell for some)**. So the ep-over-ep driver is
**the flow, not the encoder**. GW150914/GW170814 were simply the *earliest* to rail (0.40/
0.55 at ep18 while others were 0.04). Degradation is **abrupt, not a gentle monotonic
drift**. (Monotonic-vs-oscillatory at finer epoch granularity is **unmeasurable** — no
intermediate checkpoints.)

---

## STEP 4 — OOD decomposition by encoder feature group → **PSD/sensitivity dominates; coherence is in-distribution**

Per-group OOD percentile vs validation (RMS-z), `figs/step4_ood_groups.png`:

| event | energy | coherence | arrival_t | ampratio | geom_embed | **psd** | context |
|---|---|---|---|---|---|---|---|
| GW150914 | 47 | 27 | 15 | 61 | 42 | **98.5** | 65 |
| GW170814 | 87 | 13 | 79 | 66 | 33 | **99.9** | 62 |
| GW170729 | 94 | 25 | 73 | 95 | 54 | **100** | 83 |
| GW170104 | 44 | 46 | 23 | 16 | 62 | **88.8** | 76 |
| GW170608 | 80 | 38 | 65 | 89 | 32 | **85.2** | 57 |
| GW151226 | 49 | 10 | 19 | 48 | 45 | **99.1** | 44 |

**Reasoning / verdict.** The **PSD/`asd_bands` sensitivity feature is by far the most
out-of-distribution** (85–100th percentile, RMS-z 1.3–3.1) for all events — real GWOSC
detector ASDs differ from the training noise-bank's recolored ASDs. **Coherence is
in-distribution (10–46th)**; arrival-time/geometry are moderate. **But this OOD is a
chronic, stable predisposing condition** (Step 3: Mahalanobis unchanged ep18→ep19) — it is
**not** what changed to trigger the divergence. **Geometry/coherence OOD REJECTED;
PSD-OOD CONFIRMED as a predisposing (not triggering) factor.**

---

## STEP 5 — High-SNR conditioning audit → **CONFIRMED: overconfidence worsens with SNR, in-distribution, both domains**

ep19 over 4,000 validation events per domain, binned by SNR (`figs/step5_calibration_vs_snr.png`).
`cov*_sci` = coverage of science params (m1,m2,dL); targets 0.50/0.90.

**Gaussian domain**

| SNR | n | NLL | dcorr | mcorr | cov50 | cov90 | rail(spur) | dL rail | SBC p |
|---|---|---|---|---|---|---|---|---|---|
| 5–10 | 815 | 12.4 | 0.24 | 0.45 | 0.65 | **0.96** | 0.90 | 0.56 | 4e-22 |
| 10–15 | 674 | 14.1 | 0.47 | 0.77 | 0.51 | 0.86 | 0.91 | 0.60 | 4e-19 |
| 15–20 | 480 | 15.5 | 0.65 | 0.87 | 0.34 | 0.71 | 0.92 | 0.62 | 8e-53 |
| 20–25 | 383 | 15.9 | 0.72 | 0.90 | 0.22 | 0.61 | 0.91 | 0.58 | 4e-57 |
| 25–30 | 266 | 16.9 | 0.66 | 0.88 | 0.21 | 0.55 | 0.90 | 0.53 | 5e-42 |
| >30 | 1382 | 19.5 | 0.65 | 0.87 | 0.20 | **0.50** | 0.87 | 0.30 | 2e-251 |

**Real-noise domain** is essentially identical (cov90 0.95→0.49; rail 0.90→0.87; SBC ~0).

**Reasoning / verdict.** Point-estimate correlation is healthy and **improves** with SNR
(dcorr 0.24→0.72, mcorr 0.45→0.90) — the population trend is fine. But **science-param
coverage collapses with SNR** (cov90 **0.96 → 0.49**; cov50 0.65 → 0.20): high-SNR
posteriors are **too narrow and biased** (overconfident → under-covered). SBC fails
everywhere, worst at high SNR (p~1e-251). **Gaussian ≈ real** ⇒ not a noise-domain gap.
Crucially this is on **in-distribution simulated** events where training data is abundant
(Step 1) — so the high-SNR failure is **overconfidence, not data sparsity.** High-SNR
conditioning deterioration **CONFIRMED**, as a calibration/overconfidence effect.

---

## STEP 6 — Per-parameter railing localization → **science-param railing appeared ep18→ep19; railing is at the ±1 prior boundary**

Per-parameter **naive** railing % of samples (ep18 → ep19), non-circular params:

| event | epoch | mass_1 | mass_2 | dL | theta_jn | a1 | a2 |
|---|---|---|---|---|---|---|---|
| GW150914 | ep18→ep19 | 1→11 | 1→2 | 1→24 | 15→59 | 35→71 | 18→82 |
| GW170814 | ep18→ep19 | 1→3 | 1→0 | 1→16 | 19→73 | 48→85 | 26→90 |
| GW170104 | ep18→ep19 | 1→24 | 1→4 | 1→88 | 1→11 | 1→38 | 2→61 |
| GW170729 | ep18→ep19 | 1→32 | 1→9 | 1→87 | 1→9 | 1→53 | 2→66 |

**Spurious** (truth-aware, vs published) science-param railing:

| event | m1 18→19 | m2 18→19 | dL 18→19 |
|---|---|---|---|
| GW150914 | 1→11 | 1→2 | 1→**24** |
| GW170104 | 1→24 | 1→4 | 1→**88** |
| GW170729 | 1→32 | 1→9 | 0→1 |
| GW170814 | 1→3 | 0→0 | 1→**16** |

**Reasoning / verdict.** Railing is dominated by the **weakly-constrained parameters**
(spins a1/a2 22–90%, then luminosity_distance, then theta_jn for the equal-mass events);
masses rail far less. A large part of the *naive* spin railing is **legitimate low-spin
pile-up** (training a1 median 0.39, a2 0.25; a=0 maps to scaled −1) — validation shows the
same (Step 5, a2 naive ~0.6, spurious ~0.3). The **decisive, non-benign** change is that
**science-param spurious railing was ~1% at ep18 and jumped to 11–32% (mass) / 16–88%
(distance) at ep19** — i.e. the overconfidence spread from nuisance params into the
science parameters. All railing is at the **±1 ParamScaler / prior boundary** (see Step 7),
i.e. **physical prior limits**, not spline tails or a scaler bug.

---

## STEP 7 — Flow tail audit → **REJECTED**

Actual pipeline: `ParamScaler` (fixed physical↔[−1,1], **hard-clamped**) → NSF with
`tail_bound=5.0` (global) / `FLOW_NORM_BOUND=3.0` (per-param). (The task's
"StandardScaler + tail_bound=3" does not match the code.) Normalized training targets,
pre-clamp:

| stat | value |
|---|---|
| max |z| over ALL params (pre-clamp) | **1.69** (luminosity_distance) |
| every other param, max |z| | ≤ 1.0 |
| fraction of targets > ±3 | **0.0000%** |
| fraction of targets > ±5 | **0.0000%** |
| post-clamp targets (what the flow trains on) | max |z| = 1.0 |

**Reasoning / verdict.** The spline support (±5) is **never approached**: max training
target is |z|=1.69, and **0%** exceed ±3. Railing is at **±1 — one-fifth of the spline
support** — so it is a **prior-boundary** phenomenon, categorically not a spline-tail
limitation. `tail_bound` **REJECTED**. (Secondary: distance clips ~2.3% of training at the
*near* end, dL<40 Mpc — a minor learned near-boundary channel, opposite to the far-side dL
railing seen on real events.)

---

## STEP 8 — Why validation improves while GWTC worsens (quantitative)

The four "population metrics said to improve" and what actually happens at ep19:

| population metric | trend | measured at ep19 | blind to the failure? |
|---|---|---|---|
| val_nll | improves 0.996→0.864 | rewards density **at truth** only | **yes** — ignores boundary mass |
| shuffle ΔNLL | ~10 nats | conditional info intact | yes — not a calibration test |
| distance correlation | 0.777→0.783 | dcorr healthy, improves w/ SNR (Step 5) | yes — a population *trend*, not per-event width |
| "SBC / railing improve" | from ep18 CI (5.81%) | **SBC p~0; spurious railing ~0.90** (Step 5) | these are **not in the per-epoch log**; last CI was ep18 |

**Mechanism (all measured):**
1. Continued training sharpens the conditional flow → higher density at the truth → **val
   NLL falls**. NLL cannot see that the flow also piles mass at the ±1 boundaries.
2. The sharpening is **overconfident**: posteriors become too narrow/biased, so **coverage
   collapses at high SNR** (cov90 0.96→0.49, Step 5) and **spurious railing explodes**
   (ep18 ~6% → ep19 ~90% any-dim in validation; science-param railing 1%→11–88% on real
   events, Steps 5/6). This is **domain-independent** (Gaussian ≈ real).
3. The **encoder is not the driver**: OOD Mahalanobis is flat ep18→ep19 (Step 3). Real
   events are chronically OOD in the **PSD/sensitivity** feature (Step 4), but that is a
   *stable predisposing* condition, equally true at ep18 (when they were fine) and ep19.
4. GW150914 (SNR 24) and GW170814 (SNR 16) are the loudest/heaviest, mildly-OOD
   in-distribution events, so they land where overconfidence is worst (high SNR, Step 5) ×
   where bias is largest (heavy/equal-mass sparse corner, Step 2) → they visibly break
   first and hardest. They are **not qualitatively different** from high-SNR in-distribution
   validation events, which are also under-covered.

**Cause:** **confidence-calibration drift / overconfident flow sharpening in late
training**, selected *for* by a val-NLL-only checkpoint criterion that does not penalize
boundary mass. **Not** SNR imbalance, **not** a density hole, **not** an encoder-OOD
increase, **not** a spline-tail limit.

---

## FINAL SUMMARY — Hypothesis · Evidence · Verdict

| Hypothesis | Evidence | Verdict |
|---|---|---|
| **High-SNR underrepresentation** | Events at 41–70th SNR pct with 18k–28k louder signals (Step 1); high-SNR failure present even on abundant in-distribution data (Step 5) | **REJECTED** |
| **Heavy-BBH underrepresentation** | Worseners at Mc 82–90th pct, local density 2–3× lower but not empty; heaviest+most-OOD event (GW170729) is *not* a worsener (Step 2) | **CONTRIBUTING / INCONCLUSIVE** (modulates severity, not trigger) |
| **Distance-distribution mismatch** | Only GW170729 is OOD in dL (2750>2000); worseners in-distribution (440/600). dL rails as a *symptom* at ep19 (Steps 2,6) | **REJECTED as cause** |
| **Geometry-token / coherence OOD** | Coherence 10–46th pct, geometry moderate; not elevated (Step 4) | **REJECTED** |
| **PSD OOD** | PSD group 85–100th pct OOD, RMS-z 1.3–3.1 — the dominant OOD feature; but stable ep18→ep19 (Steps 3,4) | **CONFIRMED as predisposing, not triggering** |
| **Flow tail limitation (tail_bound)** | max training |z|=1.69; 0% beyond ±3/±5; railing at ±1 (Step 7) | **REJECTED** |
| **Standardization / scaler issue** | ±1 prior boundary is *where* mass dumps; ~2.3% near-dL clip; but scaler is fixed & correct (Steps 6,7) | **PARTIAL (locus, not cause)** |
| **Checkpoint over-specialization** | ep19 (best val NLL) is calibration-broken vs ep18: spurious railing 6%→90%, cov90→0.49, SBC→0 (Steps 3,5,6) | **CONFIRMED** |
| **Confidence-calibration drift (overconfident flow)** | Flow sharpens (val NLL↓) while coverage collapses & railing explodes, in-distribution & domain-independent, encoder OOD flat (Steps 3,5,6,8) | **CONFIRMED — primary mechanism** |

**Localized failure:** an **overconfident, mis-calibrated conditional flow in late
training** (visible ep18→ep19), amplified at high SNR and in the heavy/equal-mass +
PSD-OOD corner, **hidden from the val-NLL-based checkpoint criterion**. Fixes are out of
scope per the brief; the diagnostics that would have caught it (per-epoch spurious railing,
SBC, SNR-binned coverage) are not currently logged during training.

*Artifacts:* `analysis/divergence_localization/` — `step{1..7}_*.json`, `figs/*.png`,
per-step scripts, `events_ep19/` (contexts, samples). Model: `model/long_run/best_model.pth`
(epoch 19), snapshotted.
