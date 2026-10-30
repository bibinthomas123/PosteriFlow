# The (30,30) attractor: localized to the coherent encoder's geometry tokens

## RESOLUTION (2026-07-21)

**Deploy candidate: `model/lean_npe_mf_scratch/best_model.pth`.** Best
all-around checkpoint — passes every population/calibration gate AND the
synthetic basin probe (worst row q>0.95 0.16, rail 0.15). Do not deploy
`lean_npe_noiseinv`/`lean_npe_noiseinv_real` in its place: both cost the
same population quality for no real-event benefit (see below).

**Population-level fix: COMPLETE.** MF-tokens + delay-aug + aux-head closed
the {τ≈+7ms, heavy, quiet} basin for every synthetic and semi-synthetic
test: the offline probe (AUC 0.989), the frozen basin-probe bank (all 4 rows
passing), and off-source real-noise twins (q_med 0.44-0.51, rail 2-4%, no
longer pinned). This is real, validated progress over the original
(30,30)-always collapse.

**Residual failure: GW150914/GW170814's OWN exact strain, mechanistically
understood, NOT further pursued via training.** Signal subtraction +
token-grafting (2026-07-20) proved the collapse on these two specific events
is carried by the MF-token branch's response to their PARTICULAR local noise
realization, not their astrophysical content or the real-noise domain in
general (generic off-source O1 noise twins are clean). Two rounds of
noise-consistency fine-tuning (design-noise multiview, then real-noise
same-segment multiview) converged cleanly in training but did not close the
real-event gap (rail 74.0% baseline -> 81.6% design-mode -> 79.3% real-mode
— net negative to flat). Root cause of the non-transfer, confirmed
2026-07-21 (`noise_bank_severity_scan.py`): GW150914's segment is a genuine
severity OUTLIER in matched-filter SNR suppression relative to the entire
O1 slice of the real-noise training bank (0/25 sampled segments come close;
z ≈ -11 on H1) — no amount of consistency training over the existing bank
could have taught robustness to a fluctuation that severe, because nothing
like it was in the training distribution.

**Decision: do not chase this further via bespoke training.** Any fix
narrow enough to target GW150914's specific noise segment (hard-negative
augmentation manufactured to match its exact severity) is overfitting to a
single known test case with no guaranteed transfer to whatever the next
hard event looks like, for a benefit the system already has: the
OOD/confidence gate (`src/ahsd/inference/gating.py`) correctly flags both
events LOW confidence and routes them to dynesty refinement, whose
posterior on GW150914's exact injection was independently verified accurate
(`dynesty_reference.py`: q>0.95 mass 0.067, well-calibrated). The
amortized-NPE + OOD-gated-refinement SYSTEM is correct end-to-end for this
event today; only the NPE's own point estimate on this one case is not, and
that's the tool's designed fallback path, not an open bug.

**If revisited later, the productive direction is general, not
event-specific:** grow the real-noise bank's diversity (more GWOSC segments
across O1-O4) as a population-wide data-quality investment — it reduces the
chance of ANY future event landing in an uncovered severity tail, without
hand-fitting to GW150914's particular noise structure. Not scheduled; no
evidence yet that the current bank size is a practical problem beyond this
one pathological case.

---

**Date:** 2026-07-16
**Models:** `model/lean_npe_mcq` (coherent, mc_q, ep40) and `model/lean_npe`
(coherent, m1m2, ep40); controls `model/lean_npe_era` (ep12, undertrained),
`model/lean_v8_psdcond_ft` (conv).
**Scripts:** `twin_test.py` … `twin_test8.py`, `corner_check.py` (this dir).
Numbers below are medians over 4000 posterior samples; `rail` is the trainer's
spurious-railing metric.

## TL;DR

GW150914/GW170814 collapse to (m1,m2)≈(30,30), q→1 with ~65% railing because
their sky geometry puts the H1–L1 arrival delay at **+7 ms**, and converged
coherent-encoder models have learned a **degenerate geometry-token mapping in
the cell {τ_HL ≈ +6…+9 ms} × {loud} × {heavy}**: every event in that cell gets
(nearly) the same context, so the flow returns the cell-average posterior —
(30,30) — regardless of the true masses, mass ratio, or inclination.
Proven causally by token grafting (round 8): grafting non-basin geometry
tokens onto the real GW150914 strain removes the attractor entirely
(q 1.000→0.48, rail 0.63→0.06); grafting basin tokens onto a clean strain
installs it. Flow parameterization (m1m2 vs mc_q) is exonerated: both flows
collapse identically. The mc_q reparameterization *did* structurally eliminate
ordering violations and softened the basin (partial vs full capture) but did
not remove it.

**Bug found on the way:** `analysis/gw150914_localization/_common.py::raw_projected_signal`
applies antenna patterns but **no per-detector time delay** — every injection
built with it (incl. `inject_into_real_noise`, hence the step4_semireal
results) has τ=0 between detectors, i.e., geometrically impossible signals
that never probe the basin. Past "twins are clean" results built on this
helper are invalid for geometry-conditional questions. The dataset generator
itself is fine (audited: stored-event lags spread ±12 ms, mean |lag| 6.8 ms).

## Round-by-round evidence

**R0 — mcq training run (ep40).** All population gates green (11/11 SBC,
coverage clean both domains, shufΔ 16/14 nats, railing 2%). Corner
(heavy+q>0.8) stable from ep20: q_bias ≈ −0.29, no q≈1 spike, corner rail
1.4% at ep40 — no late-training collapse (`corner_check.py`, epochs 20/30/40).
Real events: mcq ≈ m1m2-old on all six GWTC smoke tests; GW150914/GW170814
collapsed in both; GW170104/GW170729 q-underestimated in both (that second
failure mode matches the in-distribution corner bias and is encoder-shared:
identical across both flows).

**R1 — twins, Δt=0 (buggy geometry), design noise, SNR 85.** Clean in both
models (Mc≈31.5 recovered, q̂≈0.44 low-biased, rail 4–5%). Old "twin
reproduces failure" verdict seemed dead — later explained by the Δt=0 bug.

**R2 — SNR-matched (dL 1564 → SNR 24) and semi-real (±300 s off-source O1)
twins, still Δt=0.** All clean. Amplitude regime and noise domain ruled out.

**R3 — on-source isolation.** asd_bands zeroed: collapse unchanged (pathway
inert). H1-only / L1-only: partial collapse each (rail 0.30–0.53) — both
detectors' signals carry it; full pair completes it. Same-segment twins ±16 s
and SEOBNRv4 twin: clean (segment and waveform family ruled out).

**R4 — perturb the real input.** Collapse survives sign flip, V1 reseed, and
amplitude ×1.0–2.0; breaks at ×0.5 (amplitude threshold ≈ SNR 15–20); breaks
under ±0.5–1 s rolls; **breaks under H1↔L1 swap** (τ → −6.8 ms). Measured
real H1–L1 lag +6.84 ms; the pseudo-truth twin's lag was **0.00 ms** — the
`_common.py` bug discovered.

**R5 — delay-corrected twins + dataset audit.** Training data has proper
delays (±12 ms spread). Delay-corrected twin (+7 ms): **full collapse
reproduced with a known-truth injection**, in off-source real noise AND in
pure Gaussian design noise, at SNR 85 and SNR 24. Mirror control (−6.6 ms):
clean. The failure is in-distribution and geometry-gated.

**R6 — basin map (design noise, fixed seed).** Δt scan: railing lobe at
+5…+9 ms only (old model peak rail 0.56 at +7/+8; mcq attenuated, peak 0.31);
nothing on the negative side. q-truth scan at +7 ms, fixed Mc 31.3: the old
model returns the identical (30.4, 30.7) posterior for every truth q from
0.50 to 1.00 — constant output over a truth cell = localized context
collapse. Edge-on at +7 ms: **fully collapses** (the earlier "edge-on
escapes" observation was an artifact of Δt=0 injections).

**R7 — generations.** era (coherent, ep12, undertrained): no basin (basin
emerges with training). v8-conv: globally broken on these inputs (rail >0.93
everywhere, masses at prior bounds) — uninformative as a control.

**R8 — causality (token grafting).**

| input | condition | mcq q / rail | old q / rail |
|---|---|---|---|
| basin strain (Δt+7) | normal | 0.85 / 0.41 | 0.98 / 0.54 |
| basin strain | graft Δt−7 tokens | 0.45 / 0.016 | 0.41 / 0.016 |
| clean strain (Δt−7) | graft Δt+7 tokens | 0.74 / 0.36 | 0.72 / 0.41 |
| real GW150914 | normal | 1.000 / 0.63 | 0.996 / 0.66 |
| real GW150914 | graft Δt−7 tokens | 0.48 / 0.057 | 0.44 / 0.043 |

Geometry tokens are necessary and sufficient. (gtok=0 rescues the old model
but not mcq — the zero-token point is itself OOD and happens to land near the
basin for mcq; grafts of valid geometries are the reliable causal probe.)

## Root-cause synthesis

The GCC/coherence feature code (`coherent_encoder.py`) is symmetric and
unremarkable — the +τ-only lobe is **learned**, not structural. The enabling
mechanism is a training-diversity hole: **RemixDataset's augmentations never
vary inter-detector delay.** Time-shift is a common-mode roll; noise swap,
distance rescale, and re-coloring leave τ untouched. So the geometry branch
sees exactly one τ per stored event — orders of magnitude less effective
diversity than the amplitude/noise paths — and can memorize degenerate
token mappings in thin joint cells like {heavy, loud, τ≈+7 ms} without any
training signal opposing it. Population metrics can't see it (the cell is a
tiny prior fraction), and the corner diagnostic can't either (it aggregates
over sky, so basin events are diluted ~50:1).

## Recommended fixes (ranked)

1. **Geometry-token dropout during training** (drop/perturb `gtok` with
   p≈0.2–0.3): prevents token dominance, forces the conv path to stay
   self-sufficient, directly targets the proven causal pathway. Cheap.
2. **Differential-delay augmentation**: per-detector rolls of a few ms in
   RemixDataset *with sky labels handled honestly* (either re-solve ra/dec to
   match the new τ, or mask sky params from the loss on augmented samples).
   Restores τ diversity; requires care with ra/dec/geocent labels.
3. **Basin-cell diagnostic in the per-epoch table**: add a corner row that
   fixes sky geometry (τ ∈ [+5,+9] ms) × heavy × SNR>18 via a small bank of
   frozen delay-corrected injections — the existing corner is blind to this.
4. **Fix `_common.raw_projected_signal`** to apply per-detector
   `time_delay_from_geocenter` (see `twin_test5.delayed_signal`), and re-run
   any conclusions that leaned on step4_semireal-style injections.
5. Keep the mc_q flow: it eliminated support violations structurally and
   halves the basin capture; the remaining in-distribution q-underestimate
   (corner q_bias ≈ −0.29, shared by both flows) is an encoder-information
   issue to revisit after the basin fix.

## Addendum (2026-07-17): dropout outcome, density audit, and the feature-level mechanism

**Dropout fine-tunes (gdrop p=0.25 ×5ep, gdrop2 p=0.5 ×10ep from mcq).**
Population metrics reached best-ever (val −1.17, SBC 11/11) and the LOUD basin
rows now pass the gate (snr85: q>0.95 0.42→0.23, rail 0.39→0.20; q.65:
0.39→0.16 / 0.35→0.12), but the quiet rows did not move (snr24: 0.79/0.78;
edgeon: 0.67/0.65) and real GW150914/GW170814 stay collapsed (rail 52%/71%).
Dropout only helps where the conv path can substitute for the tokens.

**Density audit (`tau_density_audit.npz`, exact τ/SNR from stored
components).** The coverage-hole hypothesis is DEAD: τ∈[+5,+9] × heavy ×
SNR15-40 holds 1,849 rank-0 training events (613 with q>0.8). Enrichment
alone cannot fix this.

**V1 test.** With a V1 signal present (3-detector twins), snr85/q.65/edgeon
go fully clean on BOTH models (rail ≤0.02) — but basin snr24 stays collapsed
(q>0.95 ~0.63-0.66) regardless of V1, dropout, or flow. The irreducible core
is {measured τ≈+7 ms} × {heavy} × {quiet} × {H1L1-dominated}.

**The feature-level mechanism (audit anomaly).** Measured τ correlates with
mass ratio in the training components — heavy q>0.8 events pile at τ∈[+5,+9]
at 3× the mirror-bin rate (39.5% vs 13.3%, 61.5% positive), light events are
symmetric (49.5%) — impossible at truth level, and NOT a plain-xcorr artifact
(PHAT weighting reproduces it to 0.24 ms). For narrow-band heavy signals any
correlation-based lag conflates true delay with the inter-detector
polarization phase offset (shift up to a half-cycle at the ~100-150 Hz
carrier ≈ 3.5-7 ms; H1-L1 near-anti-alignment biases the fold to +). So the
measured-τ≈+7 ms heavy cell is a MANY-TO-ONE preimage of true geometries and
masses: given the geometry features, events in the cell are genuinely
ambiguous, and the cell-average answer — (30,30), q→1 — is the encoder's
NLL-rational response. GW150914's true Δt=+6.9 ms lands it in exactly this
cell. V1 pairs or high SNR break the degeneracy; quiet 2-detector heavy
events cannot.

**Sharpened fix.** The τ scalar (GCC argmax) is the misleading feature for
heavy systems. The disambiguating quantity is the per-band coherence phase
SLOPE across frequency (φ(f) = 2πfΔt + Δφ: slope = delay, intercept =
polarization offset — separable by linear fit, unlike any single-lag
statistic). Replace/augment `_tau_features` with a weighted phase-slope delay
estimate (+ report the intercept), or feed the top-k |cc| lags instead of the
argmax; then fine-tune. Dropout (now in `--geom_dropout`) remains useful as a
regularizer but is not the cure.

## Addendum (2026-07-19): oversampling verdict, the tau–q correlation is PHYSICAL, and the delay-aug + aux-head fix

**Basin-cell oversampling (lean_npe_boost, x4 loss weight, 15 ep) FAILED —
and made the quiet cell worse.** basin snr24 q_med stayed 0.997–1.000 in
every epoch and the late low-LR epochs sharpened INTO the collapse (final:
q>0.95 0.79, rail 0.79; edgeon degraded as well; tier 0 throughout). Root
cause of the failure mode: the oversampled cell is itself q-skewed (33%
q>0.8), so extra loss weight pays the model to sharpen onto the cell
average — duplication cannot beat a skewed conditional.

**The tau–q correlation is physical, not a sampling artifact.** Remapping
1,500 heavy training events to fully random isotropic skies (exact delay
re-rolls) leaves the heavy q>0.8 pileup at measured tau in [+5,+9] ms largely
intact (ratio 2.84 -> 2.22 under free re-skying). The training skies were
already isotropic; the pileup is produced by the polarization fold itself for
near-circular (q~1) heavy systems, and will also be present at deployment.
Consequence: **no data augmentation or reweighting can flatten
p(q | tau-cell)** — the geometry pathway can only ever return the skewed
population answer, and the model MUST read the per-detector chirp (conv
path) to disambiguate an individual event. LVC's GW150914 posterior (SNR 24,
2 detectors, q informative) proves the information is in the strain.

**Fix shipped (experiments/delay_aug.py + trainer flags):**

1. `--delay_aug p` — differential-delay augmentation, the honest version of
   recommended fix #2. Per signal, with probability p: a target tau_HL is
   drawn isotropically (one random grid point), the new sky is the best
   antenna-response match ON THAT tau RING (weighted phase-Procrustes over
   64 of 16,384 Fibonacci grid points), per-detector delays are re-rolled
   EXACTLY (FFT phase ramp; the fixed generation epoch GPS_REF makes true
   delays computable from ra/dec labels — verified against stored components
   to <0.06 ms), and ra/dec/psi/distance are relabeled honestly (psi = the
   closed-form optimal polarization rotation; the optimal response SCALE is
   absorbed into an exact distance relabel). Measured: corr(tau_old,
   tau_new) = +0.03 (per-event geometry memorization broken — tau-stratified
   candidate draws are essential: matching over the whole sphere drags the
   new sky back toward the old one, corr +0.81), residual antenna-shape
   label noise 0.25 median (the ring floor; augmented samples only).
   All runtime math is pure numpy (no bilby in workers), verified to machine
   precision.
2. `--aux_mass w` — auxiliary mass regression on the encoder CONTEXT
   (SmoothL1 to the rank-0 normalized (Mc, q); training-scaffold module,
   never saved, inference byte-identical). Directly penalizes the proven
   mechanism — localized context collapse: an 11-dim NLL barely notices one
   flattened coordinate on a thin cell, but a per-event regression through
   the context cannot be satisfied unless contexts separate by mass within
   the cell; near-zero on resolved events, so the gradient self-targets the
   basin.

Rationale for the pairing: delay-aug removes the memorization escape hatch
(fresh geometry every epoch), the aux head supplies the missing incentive to
extract q from the chirp. Neither alone sufficed in prior rounds (features
fixed the loud rows only; incentives-by-duplication reinforced the skew).

## Addendum (2026-07-19, later): dynesty reference, MF-token probe, and the encoder fix

**Dynesty reference (`dynesty_reference.py`, exact basin snr24 injection):
q_med 0.722, q>0.95 mass 0.067, q 90% CI [0.49, 0.96], Mc_err +0.02.** The
0.30 gate is comfortably attainable — the quiet-cell information exists in
the data and the NPE deficit (q>0.95 ~ 0.65-0.79 across pslope2/boost/dda)
is an extraction failure, not physics. (Gate stays as-is.)

**Offline feature probe (`mf_probe.py`, 240 delay-correct cell injections at
SNR~24, 5-fold CV, snr-control clean at 0.42):**

| features                        | q AUC | q R^2 |
|---------------------------------|-------|-------|
| geometry vector (encoder's own) | 0.532 | -0.24 |
| frozen pslope2 context          | 0.821 | +0.46 |
| matched-filter (Mc,q)-grid overlaps | **0.989** | **+0.91** |

Geometry features carry NO q information in-cell (the collapse, measured at
feature level). The frozen context already partially decodes q — part of the
failure is the flow discarding available information (supports the aux
head). MF overlaps make the cell nearly deterministic.

**MF-token branch implemented** (`--mf_tokens`; docs/mf_token_encoder_design.md):
120-template design-whitened (Mc,q) bank (`build_mf_bank.py` ->
`assets/mf_bank.npz`), alias-free band-limited ifft overlaps (exact envelope
at ~1 ms sampling, no noise folding), per-(det,template) peak SNR/phase/time
features, injected ADDITIVELY into the geometry-token slots behind a
zero-init gate (init_from function preserved exactly — verified to 0.0
context diff; a zero concatenated token would still perturb attention).
Torch path reproduces the probe (|z| corr 0.9998, AUC 0.989); 1.8 s per
128-batch on CPU. Template peak-time differences double as the
polarization-immune delay estimator.

**Runs:** `lean_npe_dda` (delay-aug + aux head, no MF) — COMPLETE, negative:
tier 0 all 15 epochs, final basin snr24 q_med 0.999 / q>0.95 0.699 / rail
0.660 (vs dynesty-attainable 0.067). Railing halved early (0.83->~0.45) and
edgeon touched 0.30 once, but the quiet cell never unlocked: with the
shortcut suppressed and the aux gradient on, the conv stem still cannot
extract q at SNR 24 — the encoder lacks the statistic, closing the case for
MF tokens. `lean_npe_mft` (MF tokens + delay-aug + aux head, init pslope2,
15 ep) — the FIRST run ever to unpin the quiet cell: basin snr24 q_med left
the chronic ~1.00 at epochs 4 and 7 (0.89/0.88, q>0.95 down to 0.48, Mc_err
halved to -0.06), and the whole q95 oscillation band sat below dda's
(0.48-0.67 vs 0.63-0.79). But the resolving state alternated with relocked
epochs instead of freezing in — final epoch tier 0 (0.54/0.50). Diagnosis
shift: no longer an information problem (MF pathway demonstrably extracts
in-model), now a BISTABILITY problem — the fine-tune fights the attractor
carved into the pslope2 init. `lean_npe_mft_anneal` (8 ep, lr 3e-5, no warmup,
aux_mass 2.5, init from the banked epoch-4 unlocked checkpoint) — STOPPED
after 3/3 relocked epochs (snr24 q_med 0.999-1.000, q95 0.63-0.69): even at
3e-5, one epoch of NLL re-carves the basin, and further LR decay only
freezes the relocked state harder. VERDICT: no fine-tune within the pslope2
lineage escapes the attractor — the basin is a property of the converged
weights, not of the training incentives that remain.

Final experiment: `lean_npe_mf_scratch` — the full recipe FROM INIT (40 ep,
lr 3e-4, mf_tokens + delay_aug 0.5 + aux_mass 1.0 + geom_dropout 0.25,
mc_q). Rationale: the era ep-12 result (no basin before convergence) shows
the attractor is carved late by NLL sharpening under the OLD incentive
landscape; with the shortcut broken (delay-aug), context separation paid
(aux), and the mass statistic present (MF) from step 1, that carving
pressure never exists. Launched 2026-07-19 ~17:00.

## What this resolves from earlier studies

- "val improves / GWTC worsens" divergence: the GWTC regression on
  near-equal heavies is basin capture sharpening as the encoder converges —
  invisible to val NLL and to sky-aggregated corners (consistent with
  analysis/divergence_localization; the basin is the concrete mechanism).
- The original "in-distribution failure" twin verdict was correct; the
  intermediate "twins are clean → OOD" readings (rounds 1–3 here) were
  artifacts of the Δt=0 injection bug.
- "Edge-on restores sensible q" was also a Δt=0 artifact: at the true
  geometry, edge-on collapses too.

## Addendum (2026-07-22): mf_tokens-ALONE ablation — aux_mass/delay_aug are load-bearing

Follow-up question from the deeper "why does the learned system still fail
to convert MF information into dynesty-level posteriors" investigation
(`analysis/attractor_basin/layer_pool_probe.py`, `mf_probe.py`): does
`mf_tokens` alone, without `aux_mass` and `delay_aug`, resolve the basin on
its own, or were those two extra levers actually load-bearing for
`lean_npe_mf_scratch`'s success (not just incidental co-riders)?

**Run:** `mf_tokens=True` only (no `aux_mass`, no `delay_aug`), trained FROM
SCRATCH, same base recipe otherwise (mc_q, coherent, psd_cond, geom_dropout
0.25, real_noise_prob 0.9), 15 epochs. (Training was interrupted by repeated
external kills of the background process — see below — and stitched back
together via `--init_from` across 4 segments; final segment used
`--workers 0`, which stopped the kills. This means the LR/warmup schedule
was restarted at each stitch point rather than being a single uninterrupted
run — a caveat on exact quantitative comparison to `mf_scratch`'s numbers,
though the qualitative result below is unambiguous.) Final: val_nll 1.523,
spurious_railing 0.048, dist_corr 0.78, sbc_pass_frac 0.545 — healthy
population-level metrics, no sign of the `lean_npe_mft` bistability
(unlock-then-relock) pattern at any point across the 15 epochs.

**Basin-probe result (`probe3_eval.py`, reusing `twin_test6`/`twin_test`
exactly): NEGATIVE.**

- QS scan (q-truth 0.50 -> 1.00 at Δt=+7ms, Mc fixed 31.3): posterior q_med
  is FLAT at ~0.59-0.62 across the ENTIRE truth range — q=0.50 truth gives
  q_med 0.609, q=1.00 truth gives q_med 0.592. The model is not tracking q
  at all in this cell (worse than a biased collapse — no q information is
  being used).
- Real GW150914: m1 33.1, m2 31.8, **q 0.992**, q>0.95 frac 0.580, rail
  0.369 — the SAME (30,30)/q->1 collapse the original attractor diagnosis
  identified, reproduced by a checkpoint that never saw `aux_mass` or
  `delay_aug`.

**Verdict: `aux_mass` and `delay_aug` are NOT incidental in `mf_scratch`'s
success — they are necessary.** `mf_tokens` alone gives the encoder access
to the right statistic (per `mf_probe.py`'s AUC 0.989 on raw MF overlaps)
but does not, by itself, teach the model to USE it: `delay_aug` is what
breaks the Δt-shortcut that lets NLL minimization ignore q in favor of the
spurious geometry correlation, and `aux_mass` is what supplies enough direct
gradient pressure on the mass-ratio statistic for the flow to actually
condition on it. This directly confirms the token-fusion/embedding-loss
hypothesis from the parallel probe-1/2 investigation is necessary but not
sufficient — the representation being extractable (probe 1/2, `mf_probe.py`)
is a different fact from the training incentive actually extracting it
(this ablation).

**Infra note:** the background training process for this run was killed
externally 6 times over ~2.5 hours with no OOM/jetsam signature in the
system log, no correlation to system sleep/wake, and no correlation to
Claude's own follow-up shell commands (confirmed by a fully hands-off
relaunch that was also killed). Relaunching the final segment with
`--workers 0` (no DataLoader subprocess workers) completed cleanly on the
first try — circumstantial evidence the kills were tied to the training
process's subprocess tree rather than the main process alone, though this
was not conclusively isolated. Worth using `--workers 0` by default for any
future long unattended background run in this environment until the actual
mechanism is understood.

## Addendum (2026-07-22): non-amortized control — the collapse is AMORTIZATION, not representation

Decisive test for the "is this fundamentally amortization vs representation"
branch of the investigation. Trained `mf_tokens` + `geom_dropout 0.25` +
`mc_q`/coherent/psd_cond on a NARROW, GW150914-sky-locked, non-amortized
population only (`gw150914_narrow_dataset.py`: Mc in [25,40] near GW150914's
true Mc, q broad [0.3,1.0] — still a real inference target — ra/dec FIXED
at GW150914's true sky location so τ_HL is pinned to the +7ms basin
geometry by construction; distance/spin/angles/phase drawn from normal
broad priors; ~2500 signals, min SNR 8, real-noise-bank augmented, 30
epochs, `model/lean_npe_gw150914_narrow`). This model never has to share
weights across the BNS/NSBH/light-BBH population `mf_scratch` amortizes
over — a much easier, narrower inference problem, but still a genuine one
(q is not a point mass in this set).

(Checkpoint-selection note: `best_model.pth` picked epoch 1 because val_nll
was computed against the FULL dataset's validation split, symlinked in as a
pragmatic stand-in for a proper narrow-distribution val set — a totally
different distribution from the training population, so val_nll is
meaningless here. Used `last_model.pth`, epoch 30, where the trainer's own
in-training diagnostics show real learning: rail 0.97->0.10, dist_corr
0.02->0.40 over the run.)

**Result (`probe4_eval.py`): the real-event-specific collapse DOES NOT
REPRODUCE.**

| condition | m1 | m2 | q_med | q>0.95 | rail |
|---|---|---|---|---|---|
| design twin (truth params, seed 0-2) | 46.2-46.5 | 29.4-29.6 | 0.654-0.656 | 0.07-0.08 | 0.065-0.066 |
| **real GW150914 (held out)** | **46.0** | **29.0** | **0.647** | **0.062** | **0.062** |

Real GW150914 lands statistically indistinguishable from the in-distribution
design twins on every metric — no special penalty, no pinning at q=1 (q>0.95
is only 6%, vs. 58% for the amortized `mf_only_ablation` checkpoint on the
same event in the addendum above, and vs. the original always-(30,30)
collapse in the population-wide models). When this architecture does not
have to amortize across the full population, it treats GW150914's actual
strain exactly like any other synthetic draw from its (narrow) training
distribution.

There IS a real, separate bias here: q_med 0.65 undershoots truth (0.861) by
about the same amount on BOTH real and synthetic data — a general
mass-ratio calibration undershoot of this specific narrow model (plausibly
just a 30-epoch/small-dataset training-budget artifact, not inspected
further), clearly a different failure signature from a collapse-to-q=1
pathology.

**Verdict: the residual GW150914/GW170814 real-event failure that survived
`mf_scratch`'s population-level fix is attributable to amortization
(having to fit one shared function across the full training population),
not to a representational or architectural ceiling.** The architecture,
given a narrow enough population that it doesn't have to trade real-event
accuracy for population-wide coverage, CAN resolve q correctly on the exact
strain that collapses under full amortization. This reframes the earlier
"noise-severity-outlier, not further pursued via training" residual-failure
conclusion in the RESOLUTION section above: the outlier-severity framing
explained why fine-tuning on the EXISTING bank/population couldn't fix it,
but this result shows the underlying capability is there — it's the shared-
weight population fit, not an intrinsic inability to extract q from this
particular strain, that is the limiting factor.

## Addendum (2026-07-22): MF-branch information trace — the additive fusion merge, not the embedding, is the dominant loss

Deferred follow-up from the probe-1/2 investigation (`layer_pool_probe.py`:
raw MF AUC 0.989 -> fusion layer 0 AUC 0.820, loss localized to "before the
fusion transformer" but not further). `mf_embedding_tap.py` taps every stage
inside the MF branch itself (`coherent_encoder.py` forward, lines ~280-286)
on the same checkpoint (`mf_scratch`) and basin-cell injections:

| stage | dim | AUC | q-R² |
|---|---|---|---|
| raw `_mf_features()` | 1440 | 0.983 | 0.897 |
| `mf_mlp` embedding (1440->128) | 128 | 0.897 | 0.609 |
| `mf_to_tokens` projection, pre-gate | 768 | 0.883 | 0.559 |
| × `mf_gate` (learned scale) | 768 | 0.883 | 0.559 |
| **+ geometry tokens (additive merge)** | 768 | **0.738** | **0.197** |
| fusion layer 0 (recovers some) | 256 | 0.820 | — |

**The dominant loss is the additive merge into the shared geometry-token
slot (`gtok = gtok + mf_gate * mftok`), not the `mf_mlp` embedding
compression.** That single operation costs more AUC/R² than the 1440->128
embedding step, and more than any individual transformer layer downstream
(the fusion transformer then partially RECOVERS information, 0.738->0.820,
consistent with the earlier finding that transformer/pooling stages net
improve or only mildly cost information). The geometry branch alone was
independently measured at AUC 0.532 in this same cell (near-uninformative —
the original attractor-basin mechanism) — summing a highly informative
signal (MF, 0.883) additively into the same 4-token slot as a near-noise
one appears to let the geometry signal corrupt it, rather than the two
living side by side. Actionable implication: separating MF and geometry
into distinct token slots (concatenation, not addition) rather than merging
them into one additive channel is the most direct target suggested by this
trace.

**IMPLEMENTED 2026-07-22:** `CoherentEncoder(mf_token_fusion="add"|"concat")`
(`src/ahsd/models/coherent_encoder.py`, threaded through
`LeanNPE(mf_token_fusion=...)` and `--mf_token_fusion` in
`train_lean_npe.py`, and every checkpoint-loading site in the repo per the
project's load-site discipline — parameter SHAPES are identical between the
two modes, so a wrong-mode reconstruction would silently strict-load with
wrong behavior rather than crash, making this more important than the
usual missing-kwarg case). `"concat"`: MF tokens become 4 independent
tokens appended alongside (not summed into) the 4 geometry tokens, so the
fusion transformer sees 8 extra tokens instead of 4 and learns how to
combine the two streams itself, rather than the streams being pre-combined
by a fixed linear (additive) operation before attention ever runs. `"add"`
remains the default so every existing checkpoint is unaffected. Verified:
existing `mf_scratch` checkpoint still loads strict under the new code; a
fresh `concat`-mode model runs `encode`/`sample_posterior` cleanly with 8
tokens. Added a zero-init modality/type embedding
(`enc.token_type_embed`, 2 x d_model, segment-embedding style) so the
transformer is told directly which stream each token came from rather than
inferring it from value content alone — applied before geom_dropout so a
dropped geometry stream still zeroes out completely (no constant "dropped"
leak). Everything else held IDENTICAL to `mf_scratch` (encoder, MF
extraction, optimizer/LR, loss, flow, aux_mass, delay_aug) so any change in
outcome is attributable to the fusion mechanism alone. Training launched
(`model/lean_npe_mf_concat`, `--mf_token_fusion concat --workers 0`,
~11-12h; killed and relaunched once after adding the type embedding, since
the first launch pre-dated it).

**Queued follow-ups (NOT started, ranked, single-variable, only after this
run's result is in):**
2. Increase MF token capacity: 256-d embedding -> 8-12 MF tokens (currently
   128-d -> 4). Not "more is always better" — the concern is that 4 tokens
   may be too restrictive a bottleneck to represent the full (Mc,q)
   matched-filter landscape (120-template bank).
3. Less aggressive `mf_mlp` compression: 1440->512->256->tokens instead of
   the current 1440->256->128->tokens. The embedding step was measured as a
   real but SECONDARY loss (raw 0.983 -> 128-d 0.897) — not the dominant
   bottleneck (that's the additive merge, see above), but worth revisiting
   once the fusion-mechanism fix's ceiling is known.

**RESULT (2026-07-22, `model/lean_npe_mf_concat`, 39 epochs, val_nll
-1.073, rail 0.020, dcorr 0.80 — clean healthy run, no instability):
encoder representation improved exactly as hypothesized; downstream
posterior did NOT improve, and the real-event collapse got WORSE.**

Information trace (`mf_embedding_tap_v2.py`, same basin-cell injections):

| stage | `add` (mf_scratch) | `concat` (new) |
|---|---|---|
| raw MF | 0.983 | 0.991 |
| gated tokens (pre-merge) | 0.883 | 0.861 |
| merge point (extra_tokens) | 0.738 | **0.817** |
| fusion layer 0 | 0.820 (below pre-merge) | **0.910 (ABOVE pre-merge)** |
| final context | 0.848 | **0.889** |

Exceeds the original success criterion: `concat`'s layer 0 doesn't just
land closer to its pre-merge value, it EXCEEDS it (0.910 > 0.861) — the
fusion transformer is now demonstrably adding q-information via
cross-attention between the two streams, not just partially recovering
what an additive merge threw away. The encoder-side fix is real and
confirmed at every representational checkpoint.

**But the actual posterior behavior (`probe4_eval.py` twins/real-event,
`probe3_eval.py`-style DT/QS scan) shows no improvement — if anything, a
regression on the real event:**

| test | `add` (mf_scratch) | `concat` (new) |
|---|---|---|
| QS scan q_med, truth q=0.50->1.00 | flat ~0.27-0.36 (truth-independent) | still flat, ~0.28-0.37 (truth-independent) |
| Real GW150914 | q=1.000, q>0.95=0.742, rail=0.715 | q=1.000, q>0.95=**0.896**, rail=**0.888** |

The QS scan is the most telling data point: q_med barely moves across the
ENTIRE true-q range (0.50 through 1.00) in EITHER model — this is not "a
bias toward low q," it is the flow not conditioning on q information in
this cell at all, matching the signature measured in the `mf_only_ablation`
addendum above. `concat` demonstrably has more q-information available in
its context vector (confirmed above) and still produces this same
flat/uninformative posterior — worse still, real GW150914 railing got
WORSE (0.715->0.888).

**Verdict: information content in the encoder's context vector was not the
(or not the only) practical bottleneck.** The fusion-mechanism fix is real,
correctly diagnosed, and correctly implemented — and it made no difference
to the thing that actually matters (posterior quality), arguably making the
real-event failure worse. This reframes the priority stack from the
2026-07-22 hypothesis-tree discussion: encoder/token-fusion (now tested,
ruled out as the practical driver) should drop below the FLOW and the
CONTEXT-FLOW CONDITIONING INTERFACE, which were on the original candidate
list from the very first investigation and have not yet been directly
tested. Next diagnostic direction: does the flow's conditioning actually
USE the extra q-information now present in the context (e.g. probe whether
perturbing/ablating specific context dimensions known to carry q-signal
changes the flow's output at all in this cell), rather than further
encoder-side changes (queued items #2 token capacity / #3 gentler
compression are now lower priority given this result).

## Addendum (2026-07-22, later): the oracle-context test — the flow's capacity was never the bottleneck

Direct test of the open "is the flow itself incapable, or just never given the
right context" question, using gradient-based optimization instead of a
hand-picked direction (`flow_oracle_context_test.py`, `lean_npe_mf_scratch`,
same basin-cell events). Method: for each event, freeze every model weight
and run Adam directly on the `[context; rank]` vector fed to the flow, for
300 steps, minimizing `compute_psd_aware_nll` (the model's ACTUAL training
loss, not `NSFPosteriorFlow.log_prob()` -- see bug note below) evaluated at
the event's TRUE physical parameters. Then resample from the flow using the
optimized context and check the resulting median q.

**Result: near-exact recovery of the true q in every case tested.**

| true q | q before (real context) | q after (oracle context) |
|---|---|---|
| 0.449 | 0.450 | 0.475 |
| 0.582 | 0.392 | 0.601 |
| 0.950 | 0.317 | 0.947 |
| 0.986 | 0.446 | 0.985 |
| 0.956 | 0.412 | 0.956 |
| 0.981 | 0.465 | 0.978 |

**Verdict: the flow's conditioning MECHANISM is not the bottleneck -- it has
full capacity to express the correct q-dependent posterior.** The entire
failure is upstream, in what context the encoder actually produces. This
corrects/sharpens the previous addendum's framing ("the flow ignores the
q-informative direction"): it's not that the flow categorically can't use
q-information, it's that the flow only responds to context shifts of a
scale/direction the encoder never naturally produces. Measured: the
optimized context required `|delta_context| ~ 10.5-12.4`, LARGER than the
original context's own norm (`~5.1-5.2`, ~2x) -- not a small nudge, a
genuinely different context. This reprioritizes the whole 2026-07-22
investigation back toward the ENCODER (why doesn't it produce
larger-magnitude, better-separated contexts along the q-relevant direction?)
rather than the flow's conditioning architecture -- de-emphasizing the
ranked flow-side fixes (nonlinear conditioner, residual block, FiLM,
cross-attention) from the previous addendum, none of which are motivated
anymore now that the flow side is cleared.

**Bug found and worked around while building this test:**
`NSFPosteriorFlow.log_prob()` is broken whenever the base distribution is
`PSDScaledNormal` (always, in this codebase) -- it calls nflows'
`Flow.log_prob(context=...)`, but `PSDScaledNormal.log_prob()` takes
`log_sigma_psd`, not `context`, and raises `TypeError`. The actual training
path (`LeanNPE.nll()` -> `compute_psd_aware_nll()`) never goes through
`log_prob()`, which is why this has never surfaced before. Not yet
fixed/filed as a standalone issue -- `flow_oracle_context_test.py` uses
`compute_psd_aware_nll()` directly instead, which is what every checkpoint
here was actually trained against anyway.

## Addendum (2026-07-22, later still): why doesn't the encoder produce the oracle context? Resolved — direction mismatch, not magnitude

Follow-up to the oracle-context result: the oracle solution required
`|delta_context|` ~2x the context's own norm, so why doesn't the encoder's
natural output ever reach it?

**Hypothesis 1 (tested, NOT confirmed): basin-cell-specific context
collapse.** Given the oracle test used exactly the {tau_HL~+7ms, heavy,
quiet} cell from the original attractor discovery, the natural guess was
that this specific cell's contexts are compressed/collapsed relative to a
normal geometry regime (`context_spread_basin_vs_control.py`: basin cell vs.
a matched Delta t=-7ms control, same Mc/q range). Result: the basin cell
actually has MORE raw spread along the fitted q-direction (std 2.39) than
the control (std 1.38), and a larger raw context distance between q_max and
q_min (0.659 vs 0.383). The encoder is NOT failing to vary contexts here —
if anything it varies them more than in a "normal" geometry cell. Hypothesis
rejected.

**Hypothesis 2 (tested, CONFIRMED): the oracle direction is orthogonal to
the linear q-direction.** `oracle_vs_ridge_direction.py`: computed the
cosine similarity between the oracle-found context shift and the unit
ridge-regression q-direction (the SAME direction `flow_context_sensitivity.py`
manually amplified with no effect), across 6 basin-cell events spanning
q=0.50-0.95. **Result: cosine similarity -0.093 to -0.008 in every case —
essentially zero, i.e. orthogonal.**

**Verdict: the encoder's natural context variation and the flow's actual
input sensitivity are pointed in almost completely different directions in
context space.** The encoder varies q roughly LINEARLY (which is exactly
what a ridge-regression probe is built to detect and exactly what the
`--aux_mass`/`--aux_q_embed` SmoothL1 losses train FOR), but the flow's
downstream processing — a linear context injection at each of 10 layers,
followed by nonlinear MADE residual blocks — responds to some other,
nonlinear combination of context dimensions that nothing in training
currently rewards the encoder for producing. This is not a magnitude
problem (substantial natural variation exists, more in the hard cell than
outside it) and not a basin-cell-specific collapse (rejected above) — it is
a genuine GEOMETRIC MISALIGNMENT between what the encoder learns to vary
(linearly-probable) and what the flow learns to listen to (a direction
linear probing structurally cannot see).

**Implication for the in-progress `--q_embed_dim`/`--aux_q_embed` experiment
(`model/lean_npe_q_embed`):** `aux_q_embed`'s SmoothL1 loss only trains
`q_embed_head` to be LINEARLY predictive of q (same shortcoming as
`aux_mass`) — it does not guarantee the flow learns to route through that
channel via whatever nonlinear combination its own architecture actually
needs; the flow still has to discover that through ordinary end-to-end NLL
gradient, the same discovery problem that has failed throughout this whole
investigation. Watch specifically for whether the trained flow's dependence
on the dedicated channel (once informative) is linear or requires the same
kind of oracle-search to unlock — this is now a concrete, testable
prediction for when that run finishes.

## Addendum (2026-07-22, later still): geometry of the real-vs-oracle context gap

Direct characterization of how different the real (encoder-produced) and
oracle (optimized) contexts actually are (`oracle_context_geometry.py`,
`lean_npe_mf_scratch`, 8 basin-cell events spanning q=0.45-0.98). Plumbing
was independently verified correct first (a live perturbation test:
encoder->q_embed_head->flow all connected and reactive, no dead-ending) --
this addendum is about representation geometry, not wiring.

| quantity | value |
|---|---|
| cos(real, oracle) | 0.26-0.35 (partial relationship, far from aligned) |
| `\|\|real - oracle\|\|` | 9.3-10.9 |
| `\|\|real\|\|` (scale reference) | 2.1-2.4 |
| oracle's Mahalanobis distance from the basin-cell population mean (10 PCs) | 19.7-26.3 |
| a TYPICAL real context's own Mahalanobis distance from that same population mean | 2.79 |

**Verdict: NOT "close but slightly misaligned." The oracle context sits
7-9x farther from the population center than any real context ever lands,
across all 240 basin-cell events tested -- the encoder doesn't just fail to
reach the oracle point, it never visits anywhere near that region of latent
space for any event.** The movement is moderately concentrated (top-20 of
256 context dims carry ~40-46% of the shift's energy, vs. ~7.8% if
uniformly spread -- roughly a 50-100 dim subspace, not a single axis, not
uniform), with partial but incomplete overlap in WHICH dims dominate across
different events (a handful, e.g. dims 102/168/191/203/225/244, recur, but
each event's top-5 differs substantially).

**Important caveat this raises:** because the oracle context is so far
outside anything the encoder's training ever produces, the flow's behavior
there is effectively EXTRAPOLATION -- its weights were never shaped by
training data anywhere near this region. This does not invalidate the
oracle test's core finding (the flow's parameters do have the raw capacity
to map some input to the correct output) but it weakens the follow-on
implication that ordinary end-to-end encoder training could easily reach a
working solution by "pushing harder in the right direction" -- the working
region isn't a short walk from where the encoder currently lives, and nothing
in standard NLL training pressure has ever pushed the encoder to explore that
far. Reaching it via ordinary joint training (rather than direct,
flow-frozen context optimization) is a substantially harder and less
guaranteed proposition than the raw oracle-recovery result alone suggested.

## Addendum (2026-07-22, later still): manifold-regularized trade-off — good q recovery IS reachable near the manifold

Direct test of whether the oracle solution's extreme distance (previous
addendum) is an optimization artifact (unconstrained search wandering
farther than necessary) or a genuine requirement (correct q only exists far
from the manifold). Method (`oracle_manifold_tradeoff.py`,
`oracle_manifold_tradeoff_diverse.py`, `lean_npe_mf_scratch`): same
flow-frozen context optimization as the oracle test, but now minimizing
`NLL(theta_true|c) + lambda * D(c,M)^2`, where `D(c,M)` is a whitened
20-PC Mahalanobis-style distance to the basin-cell population (fit once,
held fixed), swept over `lambda in [0, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3]`
on 4 pilot events, then confirmed on 12 events spanning the full true-q
range in even deciles.

**Result: good q recovery survives almost all the way down to distances
AT or NEAR a typical real context's own natural distance from the
population (`D≈4.2` mean, 3.6-4.7 range for real contexts) -- degradation
only sets in once `D` is forced BELOW roughly 1-2, i.e. closer to the
manifold center than a typical real point itself sits.**

| event | true q | q recovered at D~3.6-4.9 (near real-context range) | q recovered forced to D<1 |
|---|---|---|---|
| 164 | 0.594 | 0.616 (D=3.61) | 0.381 (D=0.83) |
| 123 | 0.546 | 0.605 (D=3.59) | 0.310 (D=1.52) |
| 16 | 0.964 | 0.964, holds down to D=0.98 | 0.850 (D=0.58) |
| 37 | 0.958 | 0.959, holds down to D=0.85 | 0.840 (D=0.54) |

The unconstrained optimizer (previous addendum, `lambda=0`) drifts to
`D~75-103` simply because nothing penalizes doing so -- it does not NEED
to go that far. A working solution exists much closer in, at or inside the
natural population's own typical spread.

**Verdict: this REVISES the previous addendum's caveat toward the
optimistic branch. Correct q-recovery does NOT require leaving the
manifold entirely -- modest movement near (or within) the natural
population's own distance range is sufficient in every case tested.** This
is evidence FOR "there is still an optimization/representation-learning
route to achieving this end-to-end," not evidence of an irreconcilable
latent-geometry mismatch. Combined with the earlier direction-mismatch
finding (oracle shift orthogonal to the linear ridge/aux_mass/aux_q_embed
direction), the fuller picture is: a nearby, reachable solution exists, but
nothing in the CURRENT training signal (plain NLL, or a linear auxiliary
loss) is shaped correctly to find it. That is a solvable training-signal
problem, not a fundamental capacity or reachability problem.

**Follow-up training objective proposed (not yet implemented): semi-amortized
context refinement + distillation**, directly motivated by this result and
adapted from the amortized-inference literature (Kim et al. 2018,
"Semi-Amortized VAEs"; Marino et al., "Iterative Amortized Inference") --
exactly the tool this session already built, made cheap enough to run
inside training: for a subset of training examples per epoch (prioritize
known-hard/basin cells), run a short (K~20-50 step) flow-frozen context
refinement to a per-example target `c*`, then add
`L_distill = ||encoder(strain) - c*||^2` pulling the amortized encoder
toward it (optionally using `c*` for that example's primary NLL loss too,
the standard semi-amortized-VI recipe). Better-motivated than
`--aux_mass`/`--aux_q_embed` (which only reward LINEAR predictiveness, now
confirmed to be the wrong direction) and than `--basin_oversample`
(confirmed failed -- it reweights the SAME NLL loss without changing what
the encoder is pushed toward; this proposal changes the target itself).

**Confirmed on a diverse, full-range sample** (`oracle_manifold_tradeoff_diverse.py`,
12 events spanning true q in even deciles across the WHOLE range, not just
the pilot's low/high extremes):

| lambda | mean D(c,M) | mean \|err\| | median \|err\| |
|---|---|---|---|
| 0.000 (unconstrained) | 75.20 | 0.015 | 0.006 |
| 0.010 | 6.40 | 0.012 | 0.003 |
| 0.100 | 2.14 | 0.014 | 0.005 |
| 0.300 | 1.75 | 0.018 | 0.015 |
| 1.000 | 1.03 | **0.123** | 0.112 |
| 3.000 | 0.62 | **0.291** | 0.301 |

A sharp, clean, and consistent transition: recovery stays essentially exact
(mean error <=0.018) down to D~1.75-2.14, then degrades rapidly once forced
below D~1 -- holding across the entire true-q range, not just the pilot's
tested extremes. This is a robust confirmation, not a pilot-sample fluke.

## Addendum (2026-07-22): direct test — the flow ignores the q-informative direction it's given

Direct test of "does the flow actually use the context" (`flow_context_sensitivity.py`,
`lean_npe_mf_concat`, same basin-cell events). Method: fit `ridge(context ->
q)` on real strain-derived contexts (unit direction `w_hat`, the
q-informative axis; in-sample R2 0.741, consistent with the 5-fold CV 0.584
reported above), then for 8 individual real events (spanning true q
0.45-0.99) manually perturb THAT event's own context by +-3 sigma along
`w_hat` (sigma = the natural real-event spread along that exact axis, so
+-3 sigma is 3x LARGER than how much real events actually vary along this
direction) and resample from the flow with a fixed seed, tracking median q.
Controls: an equal-magnitude perturbation along a random direction, and a
full context-zeroing ablation.

**Result:**

| perturbation | median-q shift |
|---|---|
| q-direction, +-3 sigma (3x natural spread) | ~0.01-0.03 |
| random direction, same magnitude | ~same size shift — no directional specificity |
| real event-to-event context differences (8 real events) | spread of ~0.22 (0.36-0.58) |
| context fully zeroed | collapses to a FIXED constant (0.421) for every event |

Three things follow: (1) the flow is not ignoring context outright — real
contexts give different outputs from each other and from the zeroed
control; (2) but a deliberate, oversized push along the exact axis a linear
probe confirms carries q information produces the SAME tiny shift as an
equal-sized push along a RANDOM direction — the conditioning mechanism is
not preferentially routing through the q-informative subspace at all; (3)
there is no relationship between true q and predicted median q across real
events (e.g. event 33, true q=0.95, median ~0.46; event 13, true q=0.986,
median ~0.37 — LOWER despite higher truth), consistent with the flat QS
scan above.

**Verdict: the flow's conditioning pathway demonstrably does not use the
q-information that IS present and linearly decodable in the context
(confirmed twice over now — by the encoder probes and directly by this
perturbation test).** This rules out "not enough information reaches the
context" definitively (already suspected after the concat-fusion result)
and instead directly implicates the conditioning mechanism's expressiveness
and/or the training dynamics that shape it — matching the flow/"is the
conditioning mechanism expressive enough" branch of the 2026-07-22
hypothesis-tree discussion. `NSFPosteriorFlow` uses
`context_features=288` (256 ctx + 32 rank) fed as a flat vector into every
`MaskedPiecewiseRationalQuadraticAutoregressiveTransform` layer (10 layers);
`use_masked_context`'s per-parameter block-context conditioner auto-disables
here since 288 does not divide evenly by 11 features — so conditioning is
the SAME flat 288-d vector at every layer via each layer's linear MADE
context-embedding, not a per-parameter or attention-routed mechanism. Next
direction: test whether a richer conditioning mechanism (e.g. an explicit
per-layer conditioning MLP with more capacity, or forcing `use_masked_context`
by choosing context_dim/rank_dim so 288 -> a multiple of 11) lets training
actually discover and exploit the q-informative subspace — not yet
implemented or tested.

**Training-free confirmation while that run is in progress
(`frozen_concat_test.py`):** reused `mf_scratch`'s OWN already-trained
geometry/MF token embeddings (frozen, weights never touched, only ever
optimized for the additive path) and just recombined them via
concatenation instead of addition, post-hoc:

| combination | AUC | q-R² |
|---|---|---|
| gtok alone | 0.553 | −0.238 |
| mftok alone (gated) | 0.853 | +0.577 |
| gtok + mftok (actual, as-trained) | 0.667 | +0.112 |
| [gtok ; mftok] (same frozen weights, concatenated) | **0.819** | **+0.409** |

Merely changing the combination rule — no gradient step taken — recovers
almost all the information the additive merge was destroying (back near
mftok's own standalone 0.853). This is strong training-free evidence that
the loss is mechanical (destructive interference from summing a
near-uninformative signal into a clean one), not something that requires
new learned weights to fix, and de-risks the `lean_npe_mf_concat` run
before it finishes.

**Head-to-head control (`probe4_eval.py` run against `mf_scratch` itself,
identical eval code/twin injections/truth as above):** `mf_scratch` (the
full-dataset, 40-epoch, deploy-candidate model) does NOT reproduce q≈0.86
on clean synthetic design-noise twins built at GW150914's exact true
parameters (GW150914 ra/dec, so real basin-cell geometry) either — it
collapses to q_med **0.27-0.36** (q>0.95 frac only 0.02-0.05), i.e. the
OPPOSITE direction from the (30,30)/q→1 collapse. On the real GW150914
strain the SAME checkpoint flips to q_med **1.000**, q>0.95 **0.742**, rail
**0.715**. So `mf_scratch` gives two different, both-wrong answers for
essentially the same source depending only on whether the noise is clean
synthetic Gaussian or the real measured O1 segment — a stronger, more
unstable failure mode than a single fixed bias. The narrow (probe 4) model
gives the SAME (still biased, q≈0.65) answer regardless of noise
realization. Neither model recovers the published q≈0.86 yet; the narrow
model's remaining gap looks like an ordinary calibration/training-budget
shortfall (same bias in- and out-of-distribution), while `mf_scratch`'s gap
looked noise-realization-dependent from 3 twins — directly tested next.

**Is it noise-induced mode-switching, or just a bigger consistent bias?
(`noise_mode_switch_scan.py`, 100 independent twins, same GW150914 true
params, clean synthetic Gaussian design noise, mf_scratch): CONSISTENT
BIAS, not mode-switching.** q_med across all 100 independent noise draws:
mean 0.318, std **0.028**, range [0.263, 0.397] — 0/100 twins landed above
q=0.5, let alone near the real event's q=1.000. So the correction to the
"unstable failure mode" framing above: under ordinary synthetic-noise
variation `mf_scratch` is NOT chaotic — it gives a tight, repeatable wrong
answer (q≈0.32 vs. true 0.861). GW150914's own measured segment (q=1.000,
71.5% rail) is a discrete, qualitatively different regime that essentially
no ordinary noise realization comes close to reproducing — NOT a tail of a
continuous noise-sensitivity spectrum. This corroborates, rather than
complicates, the earlier severity-outlier finding
(`noise_bank_severity_scan.py`, z≈-11 vs. the training noise bank): the
real segment is doing something qualitatively unusual, not just unlucky
within a normal-ish spread. Plot: `noise_mode_switch_scan.png`.

## Addendum (2026-07-22, later): dedicated q-conditioning embedding — implemented, training launched

Direct response to the flow-ignores-q-direction addendum above.
`LeanNPE(q_embed_dim=N)` adds a small MLP (`q_embed_head`, context_dim->64
->N) whose output is concatenated onto the FLOW's own context input (not
just an encoder-side auxiliary loss like `--aux_mass`), supervised by its
own SmoothL1 regression head (`q_pred_head`, same target as `--aux_mass`).
Rationale: `flow_context_sensitivity.py` showed the flow's per-layer dense
context->hidden map is equally (in)sensitive to the q-informative
direction as to a random direction of the same size -- rather than asking
the flow to discover that subspace unaided among 256 largely-redundant
dims, hand it a small, explicitly-supervised, dedicated channel instead.
`q_embed_dim=0` (default) is byte-identical to prior behavior;
`q_embed_head`/`q_pred_head` ARE saved in the checkpoint (unlike
`--aux_mass`'s head) since the embedding is live at inference too.
Threaded through every checkpoint load site -- this one is a hard
shape-mismatch CRASH if any site is missed (changes flow
`context_features`), unlike `mf_token_fusion`, which only risked silent
wrong behavior.

Training launched (`model/lean_npe_q_embed`, exact `lean_npe_mf_concat`
recipe + `--q_embed_dim 12 --aux_q_embed 1.0`, single-variable change on
top of the current best encoder).

**Success criterion on completion:** repeat
`flow_context_sensitivity.py`'s perturbation test, this time targeting the
dedicated 12-d channel specifically -- does perturbing THAT channel move
median q more than an equal-size perturbation elsewhere in context? -- plus
the QS scan and real GW150914 test. If the flow still ignores the
dedicated channel, the bottleneck is demonstrably inside the flow's
conditioning mechanism itself (rules out "the flow just can't find q amid
256 dims"); if it starts tracking q through this channel, the issue was
purely locational and the fix is this simple.

**DEPRIORITIZED 2026-07-23 (superseded by evidence, not deleted).**
`--aux_q_embed`'s SmoothL1 loss only trains `q_embed_head` to be LINEARLY
predictive of q -- the same flaw as `--aux_mass`. The oracle-vs-ridge-
direction addendum further below directly proved the flow's actual usable
direction is ORTHOGONAL to the linear direction, so this channel almost
certainly gets trained to encode information along the same
flow-irrelevant axis that already doomed `aux_mass`. The in-progress
40-epoch run (`model/lean_npe_q_embed`) stopped on its own after 8
healthy-but-early epochs (val_nll 2.86, rail 0.067, dcorr 0.755 -- too
early to be a real test either way); not restarted, rather than spend the
remaining budget confirming a now-well-predicted failure. Superseded by the
semi-amortized approach (see the manifold-trade-off and semi-amortized
addenda below), which distills the encoder toward whatever direction an
oracle refinement ACTUALLY finds, rather than a hand-picked linear target.
The `--q_embed_dim`/`--aux_q_embed` CODE itself is kept (small, harmless,
fully backward-compatible, `q_embed_dim=0` is byte-identical to prior
behavior) -- the mechanism of "give the flow a dedicated channel" isn't
wrong, only this particular supervision target was; worth revisiting if a
better-targeted signal (e.g. distilling q_embed_head toward oracle-refined
values instead of linear regression) is ever wanted.

## Addendum (2026-07-23): semi-amortized context refinement — implemented, one serious bug found and fixed live

Full implementation of the semi-amortized proposal above:
`experiments/semi_amortized.py` (`ContextRefiner`, `DistillationLoss`,
`SemiAmortizedTrainer`, `ManifoldStats`, `select_hard_examples`), wired into
`train_lean_npe.py` behind `--semi_amortized` (default False, byte-identical
baseline training otherwise) plus a minimal additive `RemixDataset`
change (`return_event_idx`, `basin_cell_mask()`) for basin-cell hard-example
selection. Gradient routing verified exactly per spec on both a
`q_embed_dim=0` and a `q_embed_dim>0` model: `L_flow_refined` touches ONLY
`model.flow.parameters()` (not `q_embed_head`, `rank_embed`, or the
encoder), `L_distill` touches ONLY the encoder — both confirmed via direct
`.backward()` + per-parameter-group grad inspection, not just code review.

**Bug found on the first real 40-epoch run (caught by the user watching the
logs, not by me): the manifold-lambda penalty silently didn't work,
reproducing the SAME collapse as fully unconstrained refinement despite
`--sa_manifold_lambda 0.1` being set.** Root cause: `ContextRefiner`'s
penalty used RAW (unwhitened) squared Euclidean distance in context space,
while the `maha` diagnostic (and the `oracle_manifold_tradeoff.py` sweep
that validated lambda values in the first place) uses a WHITENED PCA
Mahalanobis-style distance -- completely different units, so a coefficient
validated in one metric meant nothing in the other. `maha` stayed at
~450-500 across 3 epochs (rail climbing 0.655->0.717->0.824, gnorm 260-330,
basin probe collapsing toward q_med~0/1 depending on region) -- the exact
runaway feedback loop predicted for the fully-unconstrained case, because
the "constraint" was never actually constraining anything in the metric
that matters. A second, compounding bug: even had the metric been right,
`ManifoldStats.distance()` was `@torch.no_grad()`, so gradient couldn't
have flowed through it into the optimized context anyway.

**Fix:** added `ManifoldStats.distance2_diffable()` (same whitened-PCA
metric, WITHOUT `@torch.no_grad()`), threaded a `manifold_stats` argument
into `ContextRefiner` and `SemiAmortizedTrainer`, and made the refinement
penalty use it directly instead of raw context distance. Verified on the
real `lean_npe_mf_scratch` checkpoint (a fresh untrained model was
tried first and gave confusing, inconsistent results across lambda values
-- an untrained flow's wild NLL landscape isn't a meaningful test bed;
always validate against a converged checkpoint): `lambda=0` -> D~50-67,
`lambda=0.01` -> D~8-26, `lambda=0.1` -> D~2.6-18 -- monotonic, sensible
control restored, consistent with `oracle_manifold_tradeoff.py`'s original
finding. Re-verified all gradient-routing tests still pass after the fix.

**Separately, a real speed problem was found and fixed**: the diagnostic
`q_before`/`q_after` point estimates (logged every batch, never used in the
loss or early-stopping decision unless `--sa_q_error_threshold` is set)
call `model.flow.inverse()`, which for this autoregressive spline flow is a
sequential, per-dimension operation -- much more expensive per sample than
the parallelizable forward direction the actual training loss
(`compute_psd_aware_nll`) uses. Reducing the sample count for this
diagnostic-only computation from 256 to 32 (`--sa_diag_samples`, new
default) gave a measured 4.3x speedup on the semi-amortized step itself
(5.72s->1.33s per refined batch in an isolated timing test) with no change
to training-relevant behavior (q_before/q_after get noisier but are cosmetic).

**Status:** implementation complete and verified; a fresh 40-epoch run with
the corrected `--sa_manifold_lambda` (whitened metric, now actually
constraining) is the next thing to observe -- not yet done as of this
writing.
