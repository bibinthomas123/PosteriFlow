# GW150914 distance-loss localization

**Superseded (2026-07-21):** this directory's investigation and its
recommended fixes (A/B/C below) predate and are superseded by
`analysis/attractor_basin/REPORT.md`, which localizes the GW150914 failure
to a geometry/matched-filter-token attractor rather than context→distance
decoding, and traces its current form to noise-realization sensitivity in
the MF-token branch. Read that report for the live state of the
investigation; treat this directory as historical.

**Model under test:** `model/lean_v8_psdcond_ft/best_model.pth` (epoch 25, val_nll −1.188, psd_cond=True, psd_bands=16). This is the checkpoint whose GW150914 dL median ≈ 735 Mpc matches the "700–750, truth outside CI" symptom. (`lean_v8_psdcond_ft_with_mc_oversample` was an experiment that "did not improve" and has no CI report; not the canonical model.)

**Symptom.** GW150914 (H1+L1, O1; V1 not operational): LeanNPE dL median **735 Mpc**, ci90 [655, 1513]. Dynesty on the SAME prepared strain: **466 ± 114** (ci90 [267, 642]), truth 440 inside. Masses also biased (m1 median 30, railing 61%).

**Rule:** diagnostics only; every hypothesis gets CONFIRMED / REJECTED / INCONCLUSIVE with numbers.

All artifacts in `analysis/gw150914_localization/` (`_common.py` = shared harness; `stepN_*.py` + `.json`).

---

## STEP 0 — prior & pipeline parity  → **REJECTED as failure locus**

Both methods provably solve the same problem, and the full signal reaches the model input.

| parity axis | LeanNPE | dynesty | status |
|---|---|---|---|
| PSD (whitening ↔ likelihood) | measured median ASD (`prepared.psds`) | **same `prepared.psds`** (`benchmark_real_events.py:123`, `dynesty_bridge.py:130`) | identical by construction |
| waveform | IMRPhenomXP (train) | IMRPhenomXP (default) | identical |
| analysis window | 4 s @ 4096 Hz, trigger-centered | same | identical |
| distance prior | ~d² on [40,2200] | UniformSourceFrame(50,2100) | both rise with d; can't create a *high* bias dynesty avoids |
| mass prior | log-uniform | Uniform(5,100) | affects masses, not the distance direction |

**Intermediate-array check — matched-filter SNR on LeanNPE's exact whitened input** (phase/time-maximized MF, template scaled by the same off-source `norm`):

| det | optimal (at guessed extrinsics) | **MF SNR (data·template)** |
|---|---|---|
| H1 | 34.7 | **19.2** |
| L1 | 22.5 | **13.2** |
| network | 41.3 | **23.3** |

MF SNR 23.3 ≈ published GW150914 network SNR ~24 → **the full-SNR chirp is present and intact in LeanNPE's model input.** Preprocessing/whitening/PSD/detector-ordering all REJECTED as the loss locus.

Learned-prior probe (pure-noise input): dL reversion median ≈ **906** (GW150914 asd_bands) / **1003** (design asd_bands). GW150914 posterior (735) sits between this no-signal reversion and dynesty's likelihood value (466): the encoder extracts only ~40% of the available distance pull.

→ **The information is in the strain; the loss is inside the model.**

---

## STEP 4 — semi-real injection  → **failure is NOT GW150914-signal-specific**

Known synthetic IMRPhenomXP signal (GW150914 params, dL=440) injected into **off-source real O1 strain**, whitened by the same `prepare_real` path:

| scenario | SNR | dL median | dL/true | ci90 |
|---|---|---|---|---|
| REAL GW150914 | 23.3 | 735 | 1.67 | [655,1513] |
| semi-real O1, off −400s | 29.5 | 653 | 1.48 | [540,1283] |
| semi-real O1, off −300s | 31.3 | 663 | 1.51 | [534,1345] |
| semi-real O1, off −200s | 29.3 | 628 | 1.43 | [531,1268] |
| semi-real O1, off +200s | 26.7 | 674 | 1.53 | [605,1200] |
| semi-real O1, off +300s | 28.7 | 653 | 1.49 | [572,1328] |

Median semi-real dL/true = **1.49** — the same ~1.5× high bias as the real event. **→ CONFIRMED: rules out calibration error, waveform/template mismatch, and glitch/data-quality as the cause.** A clean synthetic signal in real O1 noise fails the same way.

### STEP 4b — design-domain dL/SNR sweep (disentangle real-noise domain vs SNR regime)

Fixed GW150914 masses; vary dL in **design Gaussian** noise (asd_bands=0, amplitude=distance cleanly):

| true dL | SNR | dL median | ratio |
|---|---|---|---|
| 440 | 98 | 934 | 2.12 |
| 700 | 62 | 653 | 0.93 |
| 1000 | 43 | 695 | 0.70 |
| 1400 | 31 | 728 | 0.52 |
| 1800 | 24 | 752 | 0.42 |
| 2100 | 20 | 822 | 0.39 |

**Predicted dL is pinned ~650–930 for every true dL from 440→2100, at both in-range and OOD SNR.** A 5× amplitude change ⇒ ~no change in inferred distance. This is *design (training) domain* — so the collapse is **not** the real-noise domain; it is that the model carries ~zero distance information for this source class anywhere. Real-O1 (~650) and design (~700) agree.

### STEP 4c — source-class distance response (design noise)

Predicted dL clusters by **chirp mass**, not true distance (response slope ≈ 0 for all classes):

| class | Mc | predicted dL (≈const) | slope d(dL̂)/d(dL) | spearman |
|---|---|---|---|---|
| light q1.0 | ~9 | ~200 | +0.11 | +0.60 |
| light q0.35 | ~8 | ~185 | +0.16 | +1.00 |
| moderate | ~17 | ~500 | +0.05 | +0.40 |
| **heavy (GW150914)** | ~29 | ~680 | **−0.05** | **−0.70** |

The model maps **chirp mass → a near-fixed distance** (heavier ⇒ farther), ~ignoring amplitude. This is the mechanism: predicted dL ≈ P(dL|Mc) from the training prior. GW150914 (Mc~29 → model says ~700) is genuinely nearby (440) ⇒ ~1.5× high.

---

## STEP 2 — asd_bands (psd_cond input) OOD  → **REJECTED as primary driver**

GW150914's asd_bands vs the training bank distribution (reproduced via RemixDataset's recipe):

| det | bands beyond [5,95]pct | NN euclid (vs typical) | Mahalanobis (vs typical) |
|---|---|---|---|
| H1 | 2/16 | 0.16 (typ 0.11) | 5.37 (typ 3.59) |
| L1 | 6/16 | 0.18 (typ 0.12) | 5.27 (typ 3.69) |

Only ~1.5× elevated, with close O1 training neighbors (bank has 55/47 O1 segments for H1/L1). Mildly OOD but well-covered. Consistent with Step 4b (design domain, asd_bands=0, fails too). **REJECTED** as the cause.

---

## STEP 1 + 5 — encoder audit  → **distance is chirp-mass-mediated; amplitude is encoded but unused**

> **⚠ AUDIT CORRECTION (see `AUDIT_mcdl.md`).** The "Mc→dL R²=0.81" below was measured on the **pooled** validation set and is an **event-type-mixing artifact** (BNS/NSBH/BBH have different mass *and* distance ranges). Within BBH the true correlation is 0.05–0.15 (all signals) up to ~0.4 for detected loud events (Malmquist selection) — **not 0.81**. The within-BBH numbers (which are the ones that matter for GW150914) are in the second table.

**Information content — POOLED validation (all event types; the 0.81 is inflated by type mixing):**

| probe | R² | reading |
|---|---|---|
| Mc → log dL | 0.81 | ⚠ **type-mixing artifact**, not a within-BBH prior |
| context → log dL | 0.41 | — |
| context → log SNR | 0.88 | encoder strongly encodes loudness |
| context → theta_jn / ra / dec / psi | 0.25 / 0.26 / 0.24 / 0.29 | modest extrinsic info (not ~0) |

**Information content — WITHIN BBH (the GW150914 class):**

| probe | R² | reading |
|---|---|---|
| Mc → log dL (all BBH signals) | 0.05–0.15 | weak; ~0.4 for detected SNR>12 (Malmquist) |
| context → log dL \| Mc | 0.53 | context carries distance info beyond Mc |
| context → log SNR \| Mc | 0.63 | context carries amplitude beyond Mc |
| **end-to-end: predicted dL vs chirp mass** | Spearman **+0.68** | model's distance tracks Mc… |
| **end-to-end: predicted dL vs TRUE dL / SNR** | Spearman **+0.31 / +0.31** | …far more than true distance/amplitude |

So amplitude **is** in the context (SNR\|Mc=0.65) — yet the posterior ignores it for distance (Step 4b/4c). The loss is in the **context→distance decoding**, not the input.

**Context geometry (Step 5)** — cos-sim / Euclidean to real GW150914's context (population median cos=0.10, euc=2.95):

| event | true dL | cos → GW150914 | euc → GW150914 |
|---|---|---|---|
| REAL GW150914 | 440 | 1.00 | 0.00 |
| matched synthetic in **real O1** noise | 440 | **0.95** | 0.66 |
| matched synthetic in **design** noise (dL1800, SNR24) | 1800 | −0.10 | 2.44 |
| matched synthetic in **design** noise (dL440, SNR98) | 440 | −0.15 | 2.40 |

Real GW150914 ≈ its matched O1 injection (cos 0.95) → the encoder does **not** treat GW150914 as special (Step 5 answer). But the *same physical source* in design vs O1 noise lands in **opposite** context regions (cos ≈ −0.1) → the context is **noise-domain-dominated**.

**Synthesis of the mechanism (audit-corrected).** For BBH the model infers distance disproportionately from **chirp mass** — end-to-end, predicted dL tracks Mc (Spearman 0.68) far more than true dL or amplitude (0.31 each). It over-uses a *modest* ~0.4 Mc–dL relation that exists in detected BBH due to SNR (Malmquist) selection, and under-uses the amplitude information that IS in the context (within-BBH context→SNR|Mc R²=0.63). At fixed masses, a 5× amplitude change moves the distance posterior ~0 (Step 4b). Plausible training incentive: amplitude is a domain-ambiguous distance cue (design SNR≈S vs O1-recolored SNR≈S/3 at identical dL labels) and the weak extrinsic info (θ_jn R²=0.25) can't break the distance–inclination degeneracy, so leaning on chirp mass is NLL-cheap. GW150914 — heavy (Mc-expectation ⇒ far, ~700–900) but genuinely nearby (440) and near face-on (an amplitude dynesty *can* exploit) — is biased ~1.5–1.7× high. Consistent with the frozen-context verdict (other/bigger flows don't fix it): the issue is the learned representation/objective, not flow capacity. *(Earlier draft cited a "strong 0.81 Mc–dL prior"; the audit showed 0.81 is an event-type-mixing artifact — corrected to the within-BBH ~0.4.)*

---

## STEP 3 — noise-era evaluation  → O1 modestly worse, confounded with SNR

Validation design signals recolored into each era's bank noise (150 events/era):

| era | mean SNR | dL ratio (median) | spearman(pred,true) | cov90 |
|---|---|---|---|---|
| O1 | 12.7 | 1.13 | 0.46 | 0.90 |
| O2 | 13.9 | 1.02 | 0.47 | 0.93 |
| O3 | 21.8 | 0.98 | 0.68 | 0.91 |

O1 (GW150914's era) is modestly worse (higher bias, lower rank-corr), but this is **confounded with SNR** — the O1 recolored bank noise is quieter (SNR~13 vs O3~22). Coverage holds across eras. The population-level O1 bias (1.13) is far milder than GW150914's 1.67, i.e. GW150914's bias is not an O1-population effect but its specific heavy-nearby-BBH placement. Not the primary driver.

## STEP 6 — waveform family (XPHM vs XP)  → REJECTED

Paired design-noise injections (150), same noise both arms:

| approximant | dL ratio | \|log ratio\| | spearman(pred,true) | cov90 |
|---|---|---|---|---|
| IMRPhenomXP | 0.834 | 0.409 | 0.117 | 0.940 |
| IMRPhenomXPHM | 0.808 | 0.415 | 0.154 | 0.927 |

XPHM (higher modes + precession) gives **no meaningful improvement** in distance recovery — differences within noise. Waveform mismatch REJECTED (also consistent with Step 4: the semi-real XP injection already reproduced the bias). The low spearman (0.12–0.15) again shows weak distance tracking across a realistic population.

## STEP 7 — external baseline (DINGO)  → **BLOCKED (not installed)**

`dingo-gw` is not installed in the `ahsd` env and no pretrained network is present. Installing DINGO + downloading a pretrained network (~GB, specific dependency pins, GNPE settings) is a heavy, environment-risky action outside the diagnostics-only scope, so it was not attempted unprompted.

What the existing external baseline already establishes: **dynesty (non-amortized, exact likelihood) recovers GW150914** (466 ± 114, truth inside) on the *same prepared strain*. So the distance information is extractable by a non-NPE method — the failure is specific to LeanNPE's amortized inference, not a physics floor. The one question DINGO would add — *is this a limitation of amortized NPE in general, or of LeanNPE specifically?* — remains open. (DINGO papers report well-calibrated GW150914 distance, which would point to LeanNPE-specific, but that is literature, not measured here.)

---

## ENCODER vs FLOW — where exactly is the amplitude→distance signal dropped?

Controlled experiment (`encoder_amplitude_response.py`): fixed heavy BBH (GW150914 intrinsics **and** extrinsics), sweep **only** dL (250→2100 Mpc, amplitude 8×) in design noise, 24 noise realizations per point.

| quantity | value | meaning |
|---|---|---|
| context → log dL, ridge R² (on sweep, CV) | **0.81** | the context **linearly encodes** the distance/amplitude |
| context proj. on amplitude axis vs log dL (noise-avg) | Pearson **−0.96** | encoding is clean and monotonic |
| amplitude-move / within-noise-scatter (raw L2) | 0.99 | on one realization the signal ≈ noise in raw L2, but is linearly separable in 256-D |
| **FLOW posterior-median dL vs true dL** | Spearman ~0, **flat** (759→617 over 250→2100) | the flow **does not use it** |

**→ The amplitude→distance information survives all the way into the 256-D context; the FLOW discards it.** Figure: `encoder_amplitude_response.png`.

Why the flow discards a signal that is present: in the *training* data, amplitude = distance × geometry (inclination/sky), and the encoder under-captures geometry (θ_jn R²=0.25). So the context's amplitude axis is **degenerate** (near+edge-on ≡ far+face-on); the flow cannot attribute it to distance and reverts to the chirp-mass shortcut (amplified by the dataset's event-type→distance coupling, `AUDIT_mcdl.md`). This reconciles the frozen-context verdict (all flows reproduce the bias): with the amplitude degenerate and geometry weak, "ignore amplitude, lean on Mc" is NLL-optimal for **any** flow. The bottleneck is not amplitude capture (fine) or flow capacity — it is (a) weak geometry extraction and (b) the dataset shortcut.

## FINAL verdict table

| # | Hypothesis | Quantitative evidence | Verdict |
|---|---|---|---|
| 0 | Pipeline / prior / PSD / whitening / detector-order mismatch | MF SNR **23.3 ≈ 24** in LeanNPE's own whitened input; dynesty uses the *same* `prepared.psds`; both IMRPhenomXP | ✗ REJECTED |
| 0 | Distance-prior form drives the high bias | both priors rise with d; dynesty (same-ish prior) still reaches 466 | ✗ REJECTED |
| 4 | GW150914-signal-specific: calibration / waveform mismatch / glitch | known synthetic signal in real O1 off-source noise biased **1.49×** (5 epochs), same as real 1.67× | ✗ REJECTED |
| 4b | Real-noise domain per se | **design** domain also pins dL ~650–930 across true dL 440→2100 | ✗ REJECTED |
| 2 | psd_cond input (asd_bands) is OOD | Mahalanobis 5.3 vs typical 3.6; NN 0.16 vs 0.11; O1 neighbors present | ✗ REJECTED (mildly elevated only) |
| — | Amplitude→distance response collapse (fixed masses) | design slope d(dL̂)/d(dL) ≈ **0**; a 5× amplitude change moves posterior ~0 | ✓ **CONFIRMED** |
| 1/5 | Distance is chirp-mass-mediated, not amplitude-based | end-to-end BBH: predicted dL vs Mc Spearman **0.68** vs true-dL/SNR **0.31**; amplitude in context (SNR\|Mc=0.63) but under-used | ✓ **CONFIRMED** |
| — | *(audited)* the exploited Mc–dL correlation is "0.81" | **0.81 is an event-type-mixing artifact**; within-BBH ~0.05–0.4 (Malmquist), not 0.81 | ✗ **corrected** (`AUDIT_mcdl.md`) |
| 1/5 | Encoder treats GW150914 as special | cos(real, matched-O1)=**0.95**; encoder is noise-domain-dominated, not signal-specific | ✗ REJECTED |
| 5 | Encoder has ~0 extrinsic info (degeneracy fully unbreakable) | context→θ_jn/ra/dec R² = 0.25/0.26/0.24 — weak but **not zero** | ? PARTIAL |
| 3 | O1 era systematically worse | O1 dL ratio 1.13 vs O3 0.98, but confounded with SNR; far milder than GW150914's 1.67 | ✗ REJECTED (not primary) |
| 6 | Waveform mismatch (XPHM vs XP) | XPHM ≈ XP (dL ratio 0.81 vs 0.83; spearman 0.15 vs 0.12) | ✗ REJECTED |
| 7 | General amortized-NPE limitation (DINGO) | dynesty recovers → extractable; DINGO not installed | ? BLOCKED |

## Localization (evidence-based)

The GW150914 distance failure is **inside the model's context→distance decoding**: for BBH, **LeanNPE infers luminosity distance disproportionately from chirp mass, not from signal amplitude** (end-to-end predicted dL vs Mc Spearman 0.68 vs true-dL/SNR 0.31; at fixed masses a 5× amplitude change moves the posterior ≈ 0). It over-uses the *modest* ~0.4 Mc–dL relation present in detected BBH (SNR/Malmquist selection), and under-uses the amplitude information that IS in the context (within-BBH SNR|Mc R²=0.63; θ_jn R²=0.25). *(The exploited correlation is ~0.4, NOT the 0.81 an earlier draft cited — 0.81 was an event-type-mixing artifact; see `AUDIT_mcdl.md`.)* GW150914 is a **heavy** BBH (Mc~29 → Mc-expectation ~700–900 Mpc) that is in fact **nearby** (440) and **near face-on** — a configuration dynesty resolves from the full amplitude+phase likelihood but the NPE cannot — hence the ~1.5–1.7× high bias.

Not the cause (measured): preprocessing/whitening/PSD/detector-order (0), prior form (0), the actual GW150914 signal / calibration / glitch (4), the real-noise domain per se (4b), asd_bands OOD (2), waveform family (6), flow capacity (frozen-context history). Plausible training-side root: amplitude is a domain-ambiguous distance cue in training (design SNR≈S vs O1-recolored SNR≈S/3 at identical dL labels) and the weak extrinsic info can't break the distance–inclination degeneracy, so leaning on chirp mass is NLL-cheap.

Figure: `summary_mechanism.png`. All numbers reproducible from `stepN_*.py` + `.json`.

---

## Recommended fixes (evidence-mapped, ordered by leverage)

Each targets a *measured* cause. The verify bar for all: on the fixed-mass amplitude sweep (`encoder_amplitude_response.py`) the **flow** median must track true dL (slope→1, Spearman→+); GW150914 dL→~440; SBC/coverage held; end-to-end BBH predicted-vs-true dL Spearman ↑ from 0.31.

**A. Break the distance–inclination degeneracy — enrich the encoder's geometry input (highest value; this is what dynesty exploits).**
Root: amplitude is in the context (R²=0.81) but unusable because geometry is weak (θ_jn/ra/dec R²≈0.25), so the flow can't deconvolve amplitude→distance. Fix: add a **coalescence-aligned, frequency-domain complex (real/imag) branch** to `LeanStrainEncoder` (tc-rewound, DINGO/GNPE-style) that exposes inter-detector amplitude+phase coherence — the information that pins inclination and thus distance. The v3-state note already measured "aligned-FD re/im gives a real inclination gain." Requires retrain, not regen. *Directly addresses the confirmed encoder→flow gap.*

**B. Remove the dataset's chirp-mass→distance shortcut (high leverage; addresses `AUDIT_mcdl.md`).**
Root: per-event-type distance/mass ranges (BBH [50,2000], BNS [10,300], NSBH [20,800]) manufacture the Mc→distance association the flow leans on. Fix, any of: (i) **unify/overlap the distance ranges across types** so every Mc sees the full distance range; (ii) **condition the flow on component masses / event-type** so Mc cannot proxy distance; (iii) flatten the post-selection distance prior. Requires dataset regen. Pair with A (A gives the model a *real* distance cue; B removes the *fake* one).

**C. Inference-time distance recalibration (no retrain; immediate partial mitigation).**
Build a **bias map** dL_corr = f(Mc, SNR, asd_bands) from validation truth and apply to real events; it will pull GW150914's median toward 440 and can widen the CI to restore coverage. Does not fix the degeneracy (posterior stays broad) but removes the systematic median bias today. Prefer over full importance reweighting (measured to fail on real events — tempering stalls).

**D. Objective nudge (low confidence, cheap).** An auxiliary amplitude↔distance consistency term could push the flow to use the context's amplitude axis — but the earlier aux-head experiment was measured NEUTRAL, so treat as secondary to A+B. Fixing the data (B) lets plain NLL reward amplitude use without extra losses.

**Recommendation:** A + B together (real degeneracy-breaking input **and** removal of the Mc shortcut) target the two confirmed roots; C as the immediate no-retrain stopgap. Not worth pursuing again (measured dead): bigger/other flows (frozen-context), aux-heads, mc_oversample, higher-order-mode waveforms, real-noise-bank changes, PSD-conditioning tweaks — all already tried or shown neutral for this specific failure.
