# Why LeanNPE's posteriors disagree with Bilby: a structured audit

**Date:** 2026-07-21
**Model under audit:** `model/lean_npe_mf_scratch/best_model.pth` (ep40, `mass_param=mc_q`,
`encoder_type=coherent`, `mf_tokens=True`, `psd_cond=True`, `delay_aug=0.5`, pure-NLL objective)
**Controls:** `lean_npe_mcq` (identical, no MF tokens), `lean_npe_noiseinv_real`
**Scripts (this dir):** `e1_probe_vs_posterior.py`, `e3_mf_features.py`
**Data:** `data/dataset/memmap/validation`, 1200 in-distribution events, design (Gaussian) noise,
rank-0 signal, 512 posterior samples/event.

---

## 0. Executive summary

The headline hypothesis in the audit brief — *"Mc is well constrained, q develops spurious
low-value tails, caused by an inefficient (log Mc, linear q) coordinate system"* — is **half right
about the symptom and wrong about the cause.**

Three measured facts reframe the problem:

1. **The flow is not the bottleneck, and neither is the parameterization.** The flow's own
   posterior median is *as accurate as the best regression probe of its own input context*
   (log Mc: flow R² 0.721 vs probe R² 0.711; q: 0.142 vs 0.111). Posterior width equals the
   context's information content to within 4–21% (`post_std/probe_residual` = 0.79 for log Mc,
   **0.96 for q**). A flow that reproduces its conditioning variable's information content to
   within 4% has no headroom to recover. **Expected improvement from a larger NSF or a different
   mass coordinate: ≈ 0.**

2. **q is not "leaking a small low tail" — it is essentially unmeasured, everywhere,
   in-distribution.** Mean posterior σ_q / prior σ_q = **0.89**; mean 90% CI width 0.83 on a prior
   support of ~0.99. Encoder→q probe R² = 0.11 globally, **0.03 inside the MF bank's own coverage**,
   0.19 even at SNR > 40. The model returns approximately the prior for q. The m1 tail is that
   uninformative q pushed through m1 ∝ q^(−0.6), which is a 8× amplification at q = 0.03.

3. **Mc is not well constrained in absolute terms either — it is only *better* than q.** On
   GW170608 the model gives Mc 90% CI [6.34, 10.04] (±23%); GWTC-1 gives ±0.4%. That is a **~50×
   deficit**. Worse, σ_logMc × SNR *increases* monotonically with SNR (7.06 → 7.80 → 8.76 → 11.81 →
   **19.29** across SNR bins 8.6 → 83). A matched-filter-limited estimator has this product roughly
   constant. **The encoder saturates exactly where information is most abundant.**

**The reason every calibration gate is green while the posteriors are 50× too wide is that SBC,
PP-plots and coverage are invariant to sharpness.** Measured on the deploy checkpoint: 90% coverage
= 0.901 (log Mc) and 0.893 (q), textbook-perfect, while the q posterior is 89% as wide as the prior.
A model that returns the prior passes SBC exactly. **The entire existing validation suite is blind
to the failure being investigated.** This is the single most important methodological finding.

**Primary bottleneck: encoder information capacity — specifically the loss of waveform phase
coherence in the strided conv stem, and an MF branch whose template bank covers 25% of the
training prior and is injected through a scalar gate that trained to 0.026.**

**Recommendation: redesign the encoder front-end (frequency-domain / SVD-compressed
representation, DINGO-style). Do not redesign the flow, the parameterization, or the objective.**

> **CONFIRMED AGAINST GROUND TRUTH (E-H, 12 events, 2026-07-21).** dynesty on identical strain with
> matched priors: the NPE's posteriors are **54.6× (Mc)** and **5.2× (q)** too wide, with a median
> Mc bias of **+7.9%** against dynesty's +0.18%. dynesty's own q width is 0.169, so **both**
> parameters are encoder-limited and the information is present in the data. The Mc ratio worsens
> with SNR (38.5× → 86.1×), confirming saturation. Every alternative was tested and eliminated:
> flow capacity, mass parameterization, SBI formulation, auxiliary losses, curriculum, bank
> coverage, bank Mc resolution, and time-summarization. One workstream: the encoder front-end.

---

## 1. Posterior representation — is the NSF expressive enough?

**Current configuration** (`lean_npe.py:377`, hardcoded defaults, *not* stored in checkpoint args):
`flow_layers=10`, `flow_hidden=256`, `flow_bins=16`, `tail_bound=5.0`, `context_features=288`
(256 context + 32 rank), `StandardNormal`-equivalent base (`PSDScaledNormal` with log_sigma≡0),
temperature pinned at 1.0 (`requires_grad_(False)`).

Note: these are **masked autoregressive** RQS transforms
(`MaskedPiecewiseRationalQuadraticAutoregressiveTransform`) with `ReversePermutation` between
layers — not coupling layers. Autoregressive transforms are strictly more expressive than coupling
at equal depth, so the brief's "number of coupling layers" framing understates current capacity.

**Verdict: PROBABLY IRRELEVANT.** Evidence:

| quantity | log Mc | q |
|---|---|---|
| flow posterior-median R² (vs truth) | 0.721 | 0.142 |
| best probe of context R² (RidgeCV/GBM, 5-fold CV) | 0.711 | 0.111 |
| mean posterior σ / best probe residual | 0.79 | **0.96** |

The flow **outperforms** the strongest probe I could fit to its own conditioning vector. It is
extracting all the information present in the context and reporting a width that matches it. There
is no representational slack.

Corroborating prior evidence: `analysis/encoder_vs_flow/report.json` (distance, older checkpoint)
found encoder probe R² 0.352 vs flow-median R² 0.310 — the same pattern, independently.

**Could the encoder contain the necessary information while the flow cannot represent it?**
Measured answer: **no.** That is precisely what the 0.79/0.96 ratios rule out.

**Estimated improvement from a larger NSF: 0 ± 2% in posterior width.** Not worth running.

*One real but minor implementation inconsistency:* `LeanNPE` constructs the flow with
`tail_bound=5.0`, but `NSFPosteriorFlow.inverse()` hard-clamps its output to
`±FLOW_NORM_BOUND = 3.0` (`flows.py:663`). Harmless today (targets live in [−1,1]) but it is a
latent trap if bounds are ever widened. Also, with data confined to [−1,1] and the spline domain
spanning [−5,5], only ~20% of the initial bin budget covers the data at initialization. RQS bin
widths are learnable so this is a conditioning/trainability nuisance, not a capacity ceiling.

**Note for reproducibility:** `flow_layers/hidden/bins` are absent from checkpoint `args`, so
checkpoints silently depend on the working tree's defaults. This is already documented in
CLAUDE.md; it should be fixed by storing them.

---

## 2. Parameterization — is (log Mc, linear q) a poor coordinate system?

**Verdict: PROBABLY IRRELEVANT as a cause. Relevant only as a *reporting* choice.**

Coordinates redistribute density; they cannot create information. Since the q posterior is already
~the prior (σ ratio 0.89) and the flow reproduces the available information to within 4%, no
reparameterization of q can sharpen it. Three independent lines of evidence:

1. **Direct measurement (this audit):** `post_std/probe` = 0.96 for q. There is nothing for a better
   coordinate to recover.
2. **Existing ablation:** `analysis/attractor_basin/REPORT.md` R8 — `m1m2` and `mc_q` models
   "collapse identically" on the basin events. Parameterization was already explicitly exonerated
   by token-grafting.
3. **Cross-model control (this audit):** `lean_npe_mcq` (no MF tokens) gives q post/prior ratio
   0.92 vs 0.89 for the deploy model — the coordinate system is identical, the tiny difference
   comes from the MF branch, not the coordinates.

**Would log(q) / logit(q) / arcsinh(q) help?** No, for the reason above — and `logit(q)` would
actively *hurt*: it sends q→1 to +∞, and q≈1 is where the measured `q_med = 1.000` railing
already occurs on GW150914/GW170814. The brief's concern about "worsening resolution near q≈1" is
correct and is a reason **not** to move. `mc_eta` is already correctly rejected in the `ParamScaler`
docstring (dη/dq = 0 at q=1 gives an inverse-sqrt singularity at the failure boundary).

**Where the parameterization *does* matter — and this is worth acting on:** m1 = Mc(1+q)^0.2 q^(−0.6)
is a variance amplifier. Reporting a 90% CI on m1 derived from an uninformative q is guaranteed to
produce absurd upper limits regardless of model quality. This is a *presentation* bug on top of the
information bug. See §4 for the prior-mismatch component and §9 for the recommended diagnostic split.

---

## 3. Likelihood approximation — would a different SBI formulation help?

**Verdict: PROBABLY IRRELEVANT (for the observed failure).**

NPE, NLE, NRE, FMPE and score-based posterior estimation differ in *how the density is
represented and trained*, not in *how much information the summary statistic carries*. All of them
consume the same 256-D context here. Since the failure is measurably upstream of the density
estimator (§1), swapping the estimator family moves nothing.

The one genuine exception worth flagging: **DINGO's relevance is not its flow — it is its data
representation.** DINGO conditions on frequency-domain strain compressed by an SVD basis (typically
100–400 coefficients/detector) that *preserves phase*, plus GNPE for time-shift equivariance. That
is an encoder change, and it is exactly the §7/§10 recommendation. Do not read "adopt DINGO" as
"adopt a different SBI objective."

Note the repo already contains `src/ahsd/models/build_svd_basis.py` — **it is imported nowhere.**
The machinery for the right fix is present and unused.

Importance sampling (`src/ahsd/inference/importance.py`) deserves separate mention: it is the
standard DINGO-IS remedy for exactly this situation, and it converts a too-wide proposal into
correct samples *provided the proposal covers the truth*. Coverage is 0.89–0.90 here, so the
proposal does cover. This is the cheapest available accuracy win (see Rank 4).

---

## 4. Training distribution

**Verdict: SECONDARY bottleneck, with one concrete high-impact defect (MF bank coverage, §7).**

Measured from `data/dataset/memmap/train/params.npy` (n = 60,306):

| param | min | p5 | median | p95 | max |
|---|---|---|---|---|---|
| m1 | 1.00 | 1.50 | 19.17 | 84.58 | 99.99 |
| m2 | 1.00 | 1.11 | 6.67 | 36.29 | 99.26 |
| Mc | 0.87 | 1.17 | 8.91 | 41.34 | 86.70 |
| q | 0.010 | 0.046 | 0.550 | 0.983 | 1.000 |
| dL | 10.1 | 73.2 | 643.8 | 1819.3 | 1999.9 |
| a1 | 0.00 | 0.008 | 0.391 | 0.931 | 0.990 |

Findings:

- **The q prior is much broader than GWTC analysis priors.** m1 is log-uniform, m2 uniform on
  [1, m1], giving median q = 0.55 and **25% of training signals with q < 0.25**. GWTC BBH runs
  typically use priors that keep q above ~0.05–0.125 with far less low-q weight. Since the model
  returns ~the prior for q, **it inherits this tail directly**, and m1 ∝ q^(−0.6) amplifies it.
  A meaningful fraction of the GW170608 m1 = 60 M☉ upper limit is prior-inherited, not
  model-pathological. *This is testable in one hour by prior-reweighting (§ Experiments).*
- **Distance boundary clamping — real bug, secondary magnitude.** `ParamScaler.RANGES` sets
  dL ∈ [40, 2200], but the dataset contains dL down to 10.1 Mpc: **2.28% of training targets are
  clamped to the boundary** by `normalize()`, i.e. a 10 Mpc source is taught to the flow as sitting
  at exactly 40 Mpc. Clamped targets create a spurious density spike at the boundary.
  *(Correction, 2026-07-21: an earlier draft of this report claimed `RemixDataset`'s distance
  rescale amplified this to 3.4%/9.9%. It does not — `remix_data.py:409` already guards the
  rescale and reverts out-of-range draws to s=1. The remix path contributes nothing here; the
  2.28% comes entirely from the generator emitting below the scaler's declared floor.)*
- Spins are aligned-magnitude only (a1, a2 ∈ [0, 0.99]); no tilts, no precession. The MF bank is
  **zero-spin**. The Mc–spin degeneracy is therefore unmodelled in the matched-filter branch.
- SNR, PSD selection, sky position, inclination: no defects found. Sky is isotropic
  (`arcsin`/`arccos` sampling, `parameter_sampler.py:111-114`), delays audited in the basin report
  (±12 ms spread), `delay_aug=0.5` active in the deploy model.

**Does the training distribution underrepresent the failure region?** Not for the *events* — it
covers them. It underrepresents them in the **MF bank** (§7), which is the defect that matters.

**Expected benefit of dataset work vs model work:** dataset fixes (bank coverage + dL bounds +
prior alignment) are cheap and address a well-identified 25%-coverage hole; they will *not* fix the
50× Mc deficit, which is architectural. Do both, model second in priority order but larger in
ultimate effect.

---

## 5. Auxiliary supervision

**Verdict: PROBABLY IRRELEVANT (already tested and measured neutral).**

*Correction to the brief:* the deploy model has **no auxiliary heads**. `aux_heads` is absent from
the trainer's argparse entirely, and `--aux_heads` was removed after being measured NEUTRAL
(chirp-mass encoder R² 0.36 → 0.34, all targets within noise; CLAUDE.md, `lean_npe.py:29`). The
brief's premise "current auxiliary heads include mass prediction" and "auxiliary losses improved
metrics" does not describe this checkpoint. The measured improvements came from **MF tokens +
delay-aug**, not aux losses.

On the proposed targets: network SNR, per-detector SNR, arrival delays, amplitude ratios and
coalescence time are all **already recoverable** from existing branches — the energy branch carries
amplitude losslessly by design, and the coherent encoder computes τ_ij explicitly
(`coherent_encoder.py:14`). Supervising them re-derives what is present. The measured pattern
(auxiliary losses move representation *statistics* without changing *information content*) is
consistent with this.

The one target that is **not** redundant and is worth a single experiment:
**whitened template overlap / likelihood-ratio regression against a dense (Mc, q) grid** — i.e.
supervising the encoder to reproduce the matched-filter likelihood surface rather than a scalar.
That is a fundamentally different signal because it is the sufficient statistic. Labelled a
**hypothesis**, expected impact Medium, cost Medium.

---

## 6. Hard-example curriculum

**Verdict: PROBABLY IRRELEVANT for this failure; the machinery exists and is off.**

`--basin_oversample` exists (`train_lean_npe.py:426`) and is **0** in the deploy model. There is no
NLL-ranked or error-ranked resampling anywhere in the trainer.

Curriculum reallocates gradient budget across examples. The failure here is not that a hard subset
is under-fit — it is that q is unmeasured **uniformly across the entire distribution** (probe R² is
0.03–0.19 in every subset tested, including high-SNR and in-bank). There is no "hard region" to
up-weight; the information is absent everywhere. Curriculum on an information-limited target
produces sharper *and wrongly-placed* posteriors — it would degrade the currently-good calibration.

If revisited after the encoder is fixed, the safe formulation is: rank by **held-out NLL on
simulated events binned by (Mc, q, SNR, τ_HL)** and oversample *cells*, never individual events —
cell-level oversampling is what `basin_oversample` already implements, and it is the correct
guard against the event-specific overfitting the brief rightly worries about (§8).

---

## 7. Matched-filter tokens — has this branch become the bottleneck?

**Verdict: SECONDARY bottleneck, but the specific defects found are concrete and cheap.**
This section contains the audit's most actionable findings.

**Defect 7a — the bank covers a quarter of the prior.** `build_mf_bank.py` defaults:
`n_mc=20` log-spaced over `mc_range=(12.0, 45.0)`, `q_grid=(1.0, 0.85, 0.7, 0.55, 0.4, 0.25)`,
zero spin, IMRPhenomXP, 120 templates total. Measured against the training prior:

| coverage | fraction of training signals |
|---|---|
| Mc inside [12, 45] | 0.360 |
| q ≥ 0.25 (bank minimum) | 0.749 |
| **fully covered (both)** | **0.253** |
| Mc **below** bank (< 12) | **0.601** |
| Mc above bank (> 45) | 0.039 |
| q below bank (< 0.25) | 0.251 |

**60% of training signals have a chirp mass below the bank's lower edge**, and the bank has no
template at all below q = 0.25 while the prior extends to q = 0.008.

This is confirmed causally in the feature space (`e3_mf_features.py`): MF-feature → log Mc probe R²
is **0.55–0.64 inside the bank and 0.02 outside it**. The branch is informative exactly where it has
templates and inert elsewhere.

GW170608 (Mc = 7.9) and GW151226 (Mc = 8.9) — two of the six GWTC smoke-test events — **fall below
the bank entirely.** GW151226's posterior median Mc is 11.34 against a truth of 8.9, biased toward
the bank's 12 M☉ lower edge. The brief's flagship failure case is an event the MF branch cannot see.

**Defect 7b — the injection gate never opened.** `coherent_encoder.py:271`:
`gtok = gtok + self.mf_gate * mftok`, with `mf_gate = nn.Parameter(torch.zeros(1))`. Trained values:

| checkpoint | mf_gate |
|---|---|
| `lean_npe_mf_scratch` (deploy) | **0.0265** |
| `lean_npe_mft` | **−0.0007** |
| `lean_npe_noiseinv` / `_real` | 0.0289 |

`mf_to_tokens` and `geom_to_tokens` weights have comparable RMS (0.061 vs 0.055), so the gate is the
dominant scale factor: **the MF branch is injected at ~2.6% amplitude**, and in `lean_npe_mft` it
was switched off entirely. A zero-initialized scalar gate on a branch whose gradient signal is weak
is a well-known optimization pathology.

**Defect 7c — the branch's validation was not representative.** `mf_probe_results.npz` reports
AUC 0.989 for q — but that is a *binary in-cell discrimination* at fixed Mc ≈ 31 in the basin cell
(n = 240). Measured globally in this audit, the same MF features carry **q at R² = −0.001 (linear)
/ −0.044 (GBM), and R² = 0.006 in-bank.** The 0.989 number does not generalize; it justified a
global architecture change on a narrow local test. The branch did what it was built for (closing
the basin) and was then assumed to be a general q-information channel, which it measurably is not.

**Would attention / cross-attention / uncertainty-aware aggregation be justified?** Not yet, and not
as the first move. Aggregation sophistication cannot extract q from features that carry R² ≈ 0 of
q. **Fix coverage and the gate first, re-measure, and only then consider aggregation.** With a dense
bank (see experiments) the argument for cross-attention over templates becomes strong, because the
statistic becomes a genuine 2-D likelihood surface worth attending over.

---

## 8. Difficult-region fine-tuning

**Verdict: PROBABLY IRRELEVANT as currently framed; one variant is safe and useful.**

Targeted sampling of high-mass / low-SNR / unequal-mass / high-spin regions redistributes training
weight. As in §6, the q deficit is uniform across regions, so redistribution does not address it.

The distinction the brief asks for is already settled in this repo, correctly, and should be
preserved: `analysis/attractor_basin/REPORT.md` explicitly declined to hard-negative-augment
GW150914's specific noise segment, on the grounds that a fix narrow enough to hit one known test
case has no guaranteed transfer. That reasoning is sound and generalizes: **anything conditioned on
a named GWTC event is overfitting; anything conditioned on a physical cell (Mc, q, SNR, τ) is
curriculum.** `basin_oversample`'s cell-based design is the right template.

The one clearly-safe and useful variant: **expand real-noise bank diversity across O1–O4**. The
basin report already identified GW150914's segment as a severity outlier (z ≈ −11 on H1, 0/25
sampled segments comparable) — a population-level data-quality investment with no event-specific
fitting. Low risk, Medium cost, Medium impact on robustness (not on the q/Mc width problem).

---

## 9. Posterior sharpness — which of the five candidate causes?

**Verdict: encoder information loss. Not epistemic, not the flow, not the coordinates, not
calibration constraints.**

| candidate | verdict | evidence |
|---|---|---|
| true epistemic uncertainty | **rejected** | Bilby achieves ±0.4% Mc on the same data; the information exists |
| flow expressiveness | **rejected** | flow R² ≥ probe R²; post_std/probe = 0.79 / 0.96 (§1) |
| parameterization artifact | **rejected** | m1m2 and mc_q collapse identically; §2 |
| training objective | **rejected** | pure NLL is proper; it is being minimized against an information-poor context |
| calibration constraint | **rejected** | temperature pinned at 1.0, no width penalty, no calibration term in the loss |
| **encoder information capacity** | **ACCEPTED** | probe R²: log Mc 0.71, q 0.11; σ_logMc×SNR rises 7.1 → 19.3 with SNR |

The **σ_logMc × SNR** diagnostic is the sharpest single number in this audit:

| SNR bin | median SNR | σ_logMc | σ_logMc × SNR | q 90% CI width |
|---|---|---|---|---|
| [0, 12) | 8.6 | 0.820 | 7.06 | 0.919 |
| [12, 20) | 15.3 | 0.509 | 7.80 | 0.875 |
| [20, 35) | 26.0 | 0.337 | 8.76 | 0.815 |
| [35, 60) | 44.2 | 0.267 | 11.81 | 0.779 |
| [60, 200) | 83.2 | 0.232 | **19.29** | 0.698 |

A matched-filter-limited estimator holds this product ~constant. It nearly **triples**. The encoder
extracts a fixed, small amount of information regardless of how much the data contains — the
signature of a representation that has thrown away the phase coherence that Mc precision lives in.

**The physical mechanism (hypothesis, well-supported):** the conv stem
(`lean_npe.py:242-247`) has strides 8·4·4·2 = **256×**, reducing 16384 samples to **61 tokens**
= 15.2 Hz, **65.6 ms per token ≈ 6.6 GW cycles at 100 Hz**. Chirp-mass precision at the 0.4% level
requires tracking inspiral phase to a small fraction of a cycle over hundreds of cycles. Intra-token
phase is not represented. This is precisely the design axis where DINGO differs (phase-preserving
frequency-domain SVD projection), and the repo's unused `build_svd_basis.py` targets it.

**Recommended standing diagnostics** (none currently in the validation suite):
1. `σ_θ × SNR` vs SNR, per parameter — flat = matched-filter-limited; rising = encoder-saturating.
2. **posterior width / prior width** per parameter — the informativeness metric SBC cannot see.
   Gate: q must fall below ~0.5 to claim q is measured.
3. `post_std / probe_residual` — the encoder-vs-flow attribution ratio (§1).
4. Per-event **KL(NPE ‖ dynesty)** on a fixed injection set, not just median offsets.

Add these to `scripts/validate_checkpoint.py`. The current suite would have rated a
return-the-prior model as fully passing.

---

## 10. Code audit — implementation details that may bias the posterior

Verified by reading, not assumed. Ordered by severity.

1. **`ParamScaler` distance range vs. dataset support — CONFIRMED BUG, moderate impact.**
   `RANGES["luminosity_distance"] = (40, 2200)`; dataset min 10.1 Mpc; `normalize()` clamps to
   [−1,1]. 2.28% of raw targets clamp; up to 3.4% (low) / 9.9% (high) after `RemixDataset`'s
   `dist_scale_range=(0.75, 1.33)`. Produces boundary density pile-up in dL. **Fix: widen the
   range.**
2. **`tail_bound` / `FLOW_NORM_BOUND` inconsistency — latent bug, no current impact.**
   `LeanNPE` passes `tail_bound=5.0`; `NSFPosteriorFlow.inverse()` clamps to ±3.0
   (`flows.py:663`); `normalize()` emits [−1,1]. Three different bounds in one path. Harmless now,
   a trap on any future widening.
3. **Flow hyperparameters not persisted in checkpoint `args`** — `flow_layers/hidden/bins` are
   hardcoded defaults, so checkpoint loading silently depends on the working tree. Already
   documented in CLAUDE.md as having caused failures twice. **Fix: store them.**
4. **`scaler.wrap()` clamps non-circular params at sampling** (`lean_npe.py:150`). This is the
   mechanical origin of the `rail` metric and of `q_med = 1.000` exactly on GW150914/GW170814.
   Correct as designed (the support really is bounded), but it means railing must be read as
   "posterior mass pushed past the bound", not "posterior peaked at the bound".
5. **`mf_gate` zero-init** (§7b) — not a correctness bug, an optimization defect with large
   measured consequence.
6. **MF bank / prior coverage mismatch** (§7a) — the highest-impact single defect found.
7. Checked and **found correct**: `_to_internal`/`_from_internal` are exact inverses and
   rank-agnostic (the `...` indexing convention is respected, and the m1≥m2 guarantee holds by
   construction under `mc_q`); circular wrapping is exact for ra/phase/psi; the temperature
   Jacobian in `log_prob` is correctly applied (and temperature is pinned at 1.0 anyway); the
   real-noise re-coloring `irfft(rfft(sig)·ASD_design/ASD_measured)` is exact for diagonal
   whitening; `net_snr` is recomputed post-augmentation rather than read from stale storage;
   sky sampling is isotropic; no per-sample normalization contaminates the amplitude path.

I found no sign-error, no Jacobian omission, and no train/eval transform asymmetry. The
implementation is in good shape; the problem is representational, not a coding error.

---

## 11. Post-mortem: the previous SVD attempt (why it is not a refutation)

An SVD front-end **was built and abandoned** — commit `6bb2ccd`, "remove experimental SVD-only
encoder in favor of conv+geometry fusion". `assets/svd_basis.npz` (256 × 4016 complex, 8 MB) is
still on disk and is **imported by nothing**. The A/B that killed it is
`analysis/encoder_ab_compare/verdict.json`:

| probe R² | lean (conv) | experimental (SVD) | Δ |
|---|---|---|---|
| chirp mass | 0.6029 | 0.6012 | −0.0017 |
| distance | 0.3890 | 0.3874 | −0.0016 |
| network SNR | 0.8885 | 0.8893 | +0.0008 |

`RECOMMEND_FULL_TRAINING: false`. Taken at face value: SVD ≈ conv, no benefit.

**That test could not have detected a benefit, and the basis was mis-specified.** Four defects,
each independently sufficient to null the result:

1. **The A/B ran 4 epochs on 6000 events**, both arms fine-tuned from `lean_v6`, and both were
   flagged `unstable`. Chirp-mass probe R² moved from 0.5973 → 0.6029 (lean) and stayed flat in the
   experimental arm across all four epochs — **both arms were pinned at their `init_from` value**.
   The experiment measured "4 epochs of fine-tuning does not move probe R²", not "SVD ≠ conv". A
   front-end change alters what the *downstream* layers must learn; 4 epochs from a conv-pretrained
   init is far too short for that to express.
2. **Time-shift mismatch of 100×, and this is the decisive one.** `build_svd_basis.py:66` applies
   random arrival-time shifts of only **±0.015 s** when building the basis. The training data spans
   **geocent_time ∈ [−1.5, +1.5] s** (measured, σ = 0.857 s). A time shift τ multiplies the
   frequency-domain waveform by e^(−2πifτ): across [20, 1024] Hz that is ~15 cycles of phase ramp at
   τ = 15 ms but **~1500 cycles at τ = 1.5 s**. A basis whose training set only ever saw ±15 ms of
   ramp cannot represent a 1.5 s-shifted signal in 256 components — the projection coefficients are
   near-meaningless for most events. **This is exactly the problem DINGO solves with GNPE**
   (group-equivariant NPE: iteratively estimate and remove the time shift so the SVD basis only ever
   sees near-aligned waveforms). The previous attempt used DINGO's compression without DINGO's
   alignment, which is the one ingredient that makes the compression work.
3. **The basis was built from 800 waveforms and truncated to 256 components.** The reported
   `captured_power = 0.99999999998` is vacuous: 256 components spanning an 800-waveform set is close
   to interpolation and says nothing about generalization to unseen waveforms. DINGO builds SVD
   bases from 10⁵–10⁶ waveforms.
4. **Coverage gaps, same failure mode as the MF bank (§7a):** masses drawn from m1 ∈ [5, 100],
   m2 ∈ [5, m1] ⇒ Mc ≳ 4.4, missing the low-Mc bulk of the prior; **plus-polarization only**; a
   single H1/L1-averaged ASD rather than per-detector whitening; tilts fixed to zero.

**Revised assessment.** The prior attempt is uninformative about whether a phase-preserving
front-end helps — but it does establish that the *naive* version fails, and it correctly identifies
where the difficulty lies. This **raises the cost and risk of E-E** relative to my §9 framing:
the work is not "project onto an SVD basis", it is "solve time-shift alignment, then project onto a
properly-built basis". Concretely, E-E must be re-scoped as:

- **E-E1 (prerequisite) — EXECUTED 2026-07-21 (`eE1_alignment_gate.py`). CONDITIONAL FAIL,
  for a fixable reason.** The MF branch's per-template peak time (`_mf_features`, `t_rel`, ±1.6 s
  window at ~1 ms resolution) is already an absolute coalescence-time estimator. Measured residual
  against true coalescence time over 1000 in-distribution validation events (light-travel floor is
  ~10 ms, since per-detector arrival differs from geocent_time by up to that much):

  | subset | n | median &#124;residual&#124; | p90 | <15 ms |
  |---|---|---|---|---|
  | **ALL** | 1000 | **186.6 ms** | 1519 ms | 24.8% |
  | Mc in bank [12,45] | 379 | **14.8 ms** | 329 ms | 50.7% |
  | Mc **below** bank (<12) | 573 | **626.1 ms** | 1815 ms | 3.7% |
  | Mc above bank (>45) | 48 | 9.5 ms | 86 ms | 72.9% |
  | in-bank AND SNR>20 | 241 | **13.2 ms** | **38.9 ms** | 54.8% |
  | in-bank AND SNR>20 AND q>0.5 | 95 | **10.6 ms** | 74.5 ms | 60.0% |

  **The estimator sits at the light-travel floor wherever the bank has templates (10–15 ms) and
  fails completely where it does not (626 ms).** The global 186.6 ms is entirely an artifact of the
  bank covering 25% of the prior (§7a) — the same defect behind the information deficit. There is
  also a real SNR dependence (612 ms at SNR<12 vs 20 ms at SNR>60) which is physical: a weak
  signal's merger time is genuinely hard to localize, and those events have broad posteriors anyway.

  *Consequence:* **E-B (dense, prior-matched MF bank) is now the top priority** — it is the shared
  prerequisite for both the q-information problem and the alignment problem, and it is cheap.
  Re-run this gate immediately after E-B. Residual tail (p90 ≈ 39 ms even in-bank at high SNR) is
  what GNPE's *iteration* is for: estimate → align → re-estimate tightens exactly this.
  *Gate on re-run:* median < 15 ms **globally**, not just in-bank.
- **E-E2.** Rebuild the basis properly: ≥10⁵ waveforms, both polarizations, per-detector design
  ASD, mass coverage matching the full training prior (Mc down to ~0.9), and a shift window matching
  the E-E1 residual. Verify **held-out** captured power on waveforms not in the build set — the
  current 0.99999999998 must be replaced with an honest out-of-sample number.
- **E-E3.** Only then re-run the A/B, from scratch (not `init_from` a conv model) and for a
  realistic budget (≥20 epochs, full dataset).

If E-E1's gate fails, the SVD path should be abandoned for real, and the fallback is to reduce the
conv stem's stride (currently 256×, 6.6 GW cycles per token) — a cheaper, less principled, but
directly targeted intervention on the same quantity.

---

## 12. Fixes applied in this audit

Three of the §10 defects were fixed; verified with all 24 current-generation checkpoints still
loading `strict=True` and no new test failures (the suite's 27 pre-existing failures are in
`bias_corrector`, `priority_net`, `transformer_encoder`, and the legacy `TorchParameterScaler`
tests — none in the modules touched here).

1. **`flows.py` — `inverse()` clamped to the wrong bound.** It hard-clamped output to the
   module-level `FLOW_NORM_BOUND = 3.0` while `LeanNPE` builds the flow with `tail_bound = 5.0`,
   silently truncating the outer 40% of the spline's support. Now clamps to the flow's own
   `max(tail_bounds_list)` via `self._out_clamp` (works for the per-parameter dict form too).
   *Impact measured before accepting:* 1.1 × 10⁻⁵ of samples exceed 3.0 and none exceed 5.0, so
   existing checkpoints are numerically unaffected (~4 samples in 360k, all deep in the tail).
   The change matters for circular parameters, where clamp-then-wrap was simply wrong.
2. **`remix_data.py` — hardcoded distance guard.** The rescale guard used literals `45.0/2100.0`
   against a scaler range of `40.0/2200.0`; the two could drift apart silently. Now derived from
   `ParamScaler.RANGES` with a 5% margin (`_dist_guard_window()` → 42.0/2090.0, preserving current
   behavior).
3. **`train_lean_npe.py` — flow geometry not persisted, and silent target clamping.**
   Added `--flow_layers/--flow_hidden/--flow_bins` (defaults matching `LeanNPE`), threaded into the
   model and therefore recorded in `vars(args)`, so a checkpoint no longer depends on the working
   tree's defaults to load — the `strict=True` size-mismatch hazard documented in CLAUDE.md.
   Added `audit_scaler_support()`, which runs at startup and reports any training targets outside
   `ParamScaler.RANGES`. On the shipped dataset it prints exactly one offender:
   `luminosity_distance range=[40, 2200] below=2.28% data=[10.06, 1999.86]`.
   Also corrected the stale `--encoder_type` help text, which advertised `coherent` as
   "FD+SVD+relative-features" — it has used no SVD since `6bb2ccd` (§11).

**Deliberately NOT fixed: widening `ParamScaler.RANGES["luminosity_distance"]`.** This is the
correct fix for defect 1 in §10, but changing `RANGES` changes the physical↔normalized map, which
would **silently invalidate every one of the 24 existing checkpoints** — they would keep loading
without error and return wrong distances. It requires a coordinated change (versioned ranges stored
in checkpoint args, threaded through all ~14 load sites, plus a retrain). Flagged for a deliberate
decision rather than done unilaterally. The new startup audit makes the current cost visible in the
meantime.

---

## Deliverable A — Ranked bottlenecks

| # | Bottleneck | Impact | Cost to investigate | Cost to fix | Regression risk | Evidence |
|---|---|---|---|---|---|---|
| 1 | **Encoder phase-coherence loss** (256× strided conv → 61 tokens, 6.6 cycles/token) caps Mc at ~50× Bilby's width; σ_logMc×SNR rises 7.1→19.3 | **High** | Medium | High | Medium | §9, §1 |
| 2 | **Encoder carries ~no q information** (probe R² 0.03–0.19; posterior/prior width 0.89) — the direct cause of the m1 tail | **High** | Low (done) | High | Medium | §1, §4 |
| 3 | **MF bank covers 25% of the prior** (Mc [12,45] vs prior [0.87,91.4]; q≥0.25 vs prior ≥0.008); GW170608/GW151226 below it. **Now also proven to be what breaks time-shift alignment** (E-E1: 14.8 ms in-bank vs 626 ms below bank) — the shared prerequisite for the encoder redesign. **Do this first.** | **High** | Low (done) | **Low** | Low | §7a, §11 |
| ~~4~~ | ~~**No prior reweighting applied**~~ — **DE-RANKED, tested and refuted (E-A).** Reweighting to a GWTC-style mass prior makes m1 *worse*; the training prior is log-flat and already the more favorable one. None of the tail is prior-inherited. | **None** | Low (done) | — | — | E-A |
| 5 | **`mf_gate` zero-init → trained to 0.026**; branch injected at 2.6% amplitude | Medium | Low (done) | **Low** | Low | §7b |
| 6 | **Validation suite blind to sharpness** — SBC/coverage pass at 0.89–0.90 while posterior ≈ prior | Medium (process) | Low | **Low** | None | §9 |
| 7 | **dL range clamping** (2.3% raw, up to 9.9% after remix rescale) | Medium | Low (done) | **Low** | Low | §10.1 |
| 8 | **Zero-spin, (2,2)-only MF templates** vs a prior with a1,a2 up to 0.99 | Low–Medium | Medium | Medium | Low | §4, §7a |
| 9 | **Real-noise bank diversity** (GW150914 segment a z≈−11 severity outlier) | Low–Medium | Medium | Medium | Low | §8 |
| 10 | **Flow hyperparameters not persisted**; `tail_bound` 5.0/3.0/1.0 inconsistency | Low (latent) | Low | **Low** | Low | §10.2–3 |

Explicitly **de-ranked** (investigated, found not to be bottlenecks): NSF capacity (§1),
mass parameterization (§2), SBI formulation (§3), auxiliary supervision (§5), curriculum (§6),
event-specific fine-tuning (§8).

---

## Deliverable B — Experiments, in recommended order

**E-A. Prior reweighting — EXECUTED 2026-07-21, hypothesis REFUTED.**
(`eA_prior_reweight.py`.) I predicted the m1 tail was substantially inherited training prior and
would shrink to ~25–35 M☉ under a GWTC-style uniform-in-(m1,m2) mass prior. **It does the
opposite.** m1 90% upper limits, as-is → reweighted → GWTC:

| event | as-is | reweighted | GWTC | ESS |
|---|---|---|---|---|
| GW150914 | 78.5 | 88.7 | 40.5 | 1440 |
| GW151226 | 65.1 | 79.6 | 19.4 | **16** |
| GW170104 | 92.8 | 98.4 | 39.0 | 1518 |
| GW170608 | 58.0 | 74.3 | 15.9 | **12** |
| GW170729 | 98.8 | 101.2 | 66.5 | 1583 |
| GW170814 | 74.5 | 90.5 | 35.4 | 1132 |

The reason is straightforward in hindsight: the training prior is **log-flat in both masses**
(`parameter_sampler.py:154-157`, p ∝ 1/(m1·m2)), which puts *more* weight at low mass than a
uniform-in-(m1,m2) prior does. The training prior is therefore the **more favorable** of the two,
and removing it pushes mass upward.

**Consequence: essentially none of the m1 tail is prior-inherited — it is all information deficit.**
This *strengthens* the encoder diagnosis and removes "just reweight the prior" from the list of
cheap wins. GW170608's as-is m1 90% upper is 3.6× the GWTC value with the model's own prior already
in its favor.

*Caveat, stated plainly:* ESS collapses to 12–16 (of ~3000) for the two low-Mc events, so **their
reweighted numbers are noise** and should not be quoted. The four well-sampled events (ESS > 1100)
carry the conclusion, and all four move in the same direction.

**E-B. Dense, prior-matched MF bank — EXECUTED 2026-07-21 (`eB_dense_bank_gates.py`).
GATE 1 improved 5×; GATE 2 FAILED decisively.**
Built `assets/mf_bank_dense.npz`: `--n_mc 80 --mc_range 1.0 95.0`, q grid 1.0→0.07 (10 values),
614 templates after skipping 186 unphysical (Mc, q) cells (a `--m_min/--m_max` filter added to
`build_mf_bank.py` for this — the old coarse grid never needed it). Covers Mc [1.0, 89.7] vs the
old [12, 45]. Deployed bank left untouched.

| gate | old bank (120) | dense bank (614) | target | verdict |
|---|---|---|---|---|
| **1: alignment**, global median &#124;residual&#124; | 186.6 ms | **35.2 ms** | < 15 ms | **MARGINAL** |
| — Mc < 12 (old bank blind) | 626.1 ms | **154.7 ms** | | improved 4× |
| — SNR > 20 | — | 20.1 ms | | p90 still 825 ms |
| — SNR < 12 | 612.4 ms | 430.0 ms | | still hopeless |
| **2: MF → log Mc** probe R² (ALL) | 0.248 | **0.406** | — | clear gain |
| **2: MF → q** probe R² (ALL) | −0.001 | **0.012** | > 0.25 | **FAIL** |
| **2: MF → q** probe R² (SNR>20) | 0.006 | **0.029** | > 0.25 | **FAIL** |

**Interpretation — this is the most consequential result in the audit after §1.**

*The dense bank is a clear net win and should be adopted:* it cut the alignment error 5×, and lifted
Mc information 0.248 → 0.406. Both effects are exactly what the coverage analysis (§7a) predicted.
Cost: 614 vs 120 templates is ~5× more compute in `_mf_features` per forward pass, which will slow
training materially — worth measuring before committing to it in the training loop.

*But q did not move at all.* A 5× denser bank, now covering q down to 0.07, still yields q R² =
0.012. Coverage was **not** the reason the MF branch carries no q information. Two candidate
explanations, and they have very different consequences:

  (a) **The q↔aligned-spin degeneracy.** The bank is **zero-spin** while the training prior draws
      a1, a2 ~ U[0, 0.99]. Mass ratio and effective spin are degenerate in the inspiral phase, so a
      grid spanning only (Mc, q) might not break it.
  (b) **q is substantially unmeasurable at these SNRs regardless.**

> **BOTH SUPERSEDED — see E-H and E-I/J/K below.** (b) is refuted outright: dynesty resolves q to a
> median 90% width of **0.169** on this exact data, so q is measurable and the NPE simply is not
> measuring it. (a) is undercut too — dynesty marginalizes spin to 0.99 and still reaches 0.169.
> The actual explanation is split between a genuine limit of hand-designed peak-|z| summaries and,
> embarrassingly, **an underpowered probe on my part** (E-K): the same features yield q R² ≈ 0.18
> once summarized as a peak-normalized profile instead of ~10⁴ raw features at n = 300.
> **The q deficit is an encoder failure after all, exactly as Rank #2 originally stated.**

**E-I / E-J / E-K. Why does the MF branch miss q? Three tests, two refutations and one
measurement error of mine.** (`eI_mc_resolution.py`, `eJ_common_time.py`.)

Matched filtering *is* the sufficient statistic, so a profile of peak |z| over a dense (Mc, q) grid
should carry q. It does not. Three candidate explanations, tested in order:

| hypothesis | test | q R² | verdict |
|---|---|---|---|
| bank coverage too narrow | dense bank, 614 templates, q→0.07 (E-B) | −0.001 → 0.012 | refuted |
| **Mc grid too coarse** | purpose-built bank at **0.30%** Mc spacing vs 6%, same events, same features (n=303) | −0.012 → 0.029 | **refuted** |
| **per-template time argmax** absorbs mass mismatch (Mc–t_c degeneracy) | read every template at a *common* time | 0.005 → −0.055 | **refuted** |
| **the probe was underpowered** | collapse to the network-SNR profile over q, **normalized by its own peak**: 12 features, not ~10⁴ | 0.005 → **0.178** (0.219 at SNR>15) | **confirmed** |

The Mc-resolution hypothesis was mine and it was wrong: a 20× finer grid moved log Mc only
0.353 → 0.367. Common-time readout did nothing.

**The real finding is a measurement error in this audit.** The earlier claim "the MF features carry
no q information (R² = 0.012)" compared ridge over ~10⁴ raw features at n = 300–600 against a
*context* probe with 256 features at n = 1200. Not comparable. Under a properly-conditioned,
physically-motivated summary the same features give q R² ≈ 0.18–0.22. **The information is present;
it was being measured badly.** The substantive part is the peak-normalization: absolute |z| is
dominated by distance/amplitude, and q lives in the profile's relative *shape*.

**E-K-fix (IMPLEMENTED): `--mf_shape_norm`.** `_mf_features` gains a 5th channel,
`snr / snr.amax(dim=-1)` — the scale-free profile shape. Off by default; all 24 current-generation
checkpoints still load `strict=True` (verified), and the flag is threaded through every one of the
18 load sites CLAUDE.md warns about.

**Measured payoff — narrower than predicted, and this matters.** Through the shipped encoder path on
the **deployed** bank (120 templates, 6 q values, Mc [12,45]; n = 500 in-coverage events):

| target | 4-channel (current) | 5-channel (+shape) |
|---|---|---|
| q | 0.006 | 0.011 |
| log Mc | 0.525 | **0.597** |
| log Mc (SNR>15) | 0.598 | **0.670** |

So on the deployed bank the shape channel is a **chirp-mass** gain (+0.07 R²), *not* a q gain. The
q result (0.178) required the fine bank — 98 Mc points at 0.30% spacing × 12 q values. Recovering
that in production would need ~0.3% Mc spacing across the full prior: of order 1.5×10³ Mc points ×
12 q ≈ 1.8×10⁴ templates, which is ~30× the dense bank and far beyond what is affordable per
forward pass.

**Conclusion: keep `--mf_shape_norm` (cheap, real +0.07 Mc), but it does not rescue the MF route for
q.** Hand-designed peak-|z| summaries at any deployable bank size discard most of the coherent
structure dynesty exploits. This is now a *measured* argument for a learned phase-preserving
front-end rather than an assumed one.

**E-H. The reference measurement — EXECUTED 2026-07-21, 12/12 events
(`eH_dynesty_vs_npe.py`, `eH_dynesty/summary.json`). THE DECISIVE RESULT OF THIS AUDIT.**

dynesty vs LeanNPE on *identical* strain: same injection, same noise realization, matched priors,
zero-spin BBH drawn over an SNR ladder (12/18/25/40), Mc 6–40, q 0.18–0.91.

| quantity | result |
|---|---|
| **Mc 90% width, NPE / dynesty** | **median 54.6×** (range 9.6–171.7) |
| **q 90% width, NPE / dynesty** | **median 5.2×** (range 2.0–31.0) |
| **dynesty's own q 90% width** | **median 0.169** (range 0.028–0.398); **10/12 events below 0.25** |
| NPE's own q 90% width | median 0.874 (prior support ≈ 0.99) |
| **NPE Mc median bias** | **+7.9%** (mean abs 11.8%) |
| dynesty Mc median bias | +0.18% (mean abs 0.43%) |
| NPE q median bias | **−0.238** (systematically low) |

Scaling:

| SNR bin | Mc ratio | q ratio | | true q | Mc ratio | q ratio | dynesty q width |
|---|---|---|---|---|---|---|---|
| 12–15 | 38.5× | 4.8× | | q < 0.5 | 44.7× | **9.3×** | 0.094 |
| 15–30 | 47.5× | 5.5× | | 0.5–0.8 | 85.8× | 5.0× | 0.167 |
| 30–50 | **86.1×** | 5.8× | | q > 0.8 | 53.0× | 3.6× | 0.231 |

**q is comfortably measurable and the NPE is not measuring it.** dynesty resolves q to a median 90%
width of 0.169 — 10 of 12 events under 0.25 — while the NPE returns 0.874, i.e. the prior. This
**refutes the "q is largely physical" conclusion recorded in E-B above**, which rested on a single
archived event (`basin_snr24`, dynesty q width 0.47): heavy, Mc ≈ 31, q_true 0.86 — near-equal-mass,
where q genuinely *is* degenerate. In this sweep the analogous event (ev02, Mc 40.1, q 0.86) gives
the smallest ratio in the set, 2.0×, and dynesty's widest q, 0.398. The one prior data point was
the least representative case available.

Note the q gap is **worst at low q** (9.3× for q < 0.5) — the NPE fails hardest exactly where q is
most informative — and the Mc gap **worsens with SNR** (38.5× → 86.1×), the saturation signature of
§9 now confirmed against ground truth rather than inferred from σ×SNR alone.

**This closes the audit's original question.** The brief asked why medians differ from Bilby and why
posteriors are wider. Measured, in-distribution, against a reference sampler with matched priors:
median Mc bias **+7.9% vs dynesty's +0.18%**, and posteriors **54.6× (Mc) / 5.2× (q)** too wide.
Both parameters are encoder-limited; the information is demonstrably present in the data. There is
one workstream, not two.

**Spin caveat — RESOLVED 2026-07-21 (`eH_spin_check.py`, `eH_spin/`, n = 4 with
a1, a2 ~ U[0, 0.9]).** The sweep above used zero-spin injections while the training prior draws
spins. The re-run also happened to draw lower masses, so the comparison below is **mass-matched**
(Mc < 13) to separate the two effects:

| | Mc ratio | q ratio | dynesty's own q width |
|---|---|---|---|
| zero-spin, Mc < 13 (n=8) | 77.1× | 7.1× | 0.120 |
| spins U[0,0.9], Mc < 13 (n=4) | **117.6×** | 4.6× | **0.292** |

Two conclusions:

1. **The Mc result is not an artifact of zero-spin — it is if anything understated.** With realistic
   spins the gap *widens* (77× → 118×) and dynesty's own Mc width does not loosen. The headline
   54.6× (full 12-event sweep) is the conservative number.
2. **Part of q's difficulty is genuinely physical after all.** dynesty's achievable q width nearly
   triples with realistic spins (0.120 → 0.292) — the q↔χ_eff degeneracy is real, exactly as
   hypothesised in E-B and then over-corrected against when the zero-spin sweep landed. But the
   **NPE/dynesty ratio is stable at ~4.6–7×**, so the NPE is still ~5× short of what is achievable
   *on the actual training distribution*. q is a real encoder deficit; its ceiling is just lower
   than the zero-spin sweep implied.

*Consequence for the gates:* `validate_checkpoint.py`'s `mass_ratio` reference was moved to the
spin-inclusive optimum (0.288, from 0.167) — holding the model to a zero-spin target would demand
better-than-achievable performance on data that has spins. `chirp_mass` keeps the 12-event value,
which is the looser of the two measurements. Both choices are deliberately conservative, and the q
reference rests on only 4 events: **re-measure with ~12 spinning events before treating it as
settled.**

**E-C. Open the gate (½ day, Low cost).** Re-train `lean_npe_mf_scratch` with `mf_gate`
initialized at 1.0 (or removed), holding everything else fixed, against the dense bank from E-B.
*Measures:* whether 7a or 7b dominates. Report encoder→q probe R² and q posterior/prior ratio.

**E-D. Run the experiment that was already designed and never executed (1 day).**
`experiments/frozen_context_heads.py` — heads A (current NSF) / B (larger NSF) / C (MDN) on
identical frozen contexts. It has no outputs (`model/frozen_heads/` and `data/frozen_ctx/` do not
exist). My §1 result predicts **A ≈ B ≈ C**. Running it converts my single-model probe argument
into the repo's own pre-registered control and settles §1 definitively.

**E-E. Phase-preserving front-end (High cost, High expected value — see §11 first).**
Prototype a frequency-domain encoder projecting onto an SVD basis. **This was attempted before and
abandoned; §11 shows why that attempt does not refute the approach, and what must be fixed first
(time-shift alignment is the real blocker, not the basis).**
*Gate:* σ_logMc × SNR must become approximately flat in SNR (currently 7.1 → 19.3), and
σ_logMc at SNR > 60 must fall below 0.05 (currently 0.232).
*This is the only experiment on the list with a plausible path to Bilby-comparable Mc precision.*

**E-F. Distance-range fix (1 hour, trivial).** Widen `RANGES["luminosity_distance"]` to ≈[8, 2700];
confirm the dL boundary spike disappears in the PP plot.

**E-G. Sharpness diagnostics in `validate_checkpoint.py` — IMPLEMENTED 2026-07-21.**
`sharpness_metrics()` / `sharpness_figure()` now run per domain, land in `report.json` under
`<domain>.sharpness`, render a two-panel figure, and drive four new gates. Verified on synthetic
sharp vs prior-like posteriors (gates pass / fail respectively) and then on the deploy checkpoint:

```
PASS  gaussian shuffle ΔNLL > 5 nats                     (14.73)
PASS  gaussian 50%/90% coverage within tol               (violations: none)
PASS  gaussian SBC KS p>1e-3 for ≥9/11 params            (11/11)
PASS  gaussian spurious railing < 5%                     (2.29%)
PASS  distance correlation > 0.5                         (0.798)
FAIL  chirp-mass informativeness < 0.15                  (0.429)
FAIL  mass-ratio informativeness < 0.50                  (0.887)
FAIL  no parameter effectively unmeasured (>0.85)        (dec, theta_jn, psi, phase, mass_ratio)
FAIL  Mc σ×SNR flat in SNR (ratio < 2.0)                 (7.1→8.2→8.8→12.1→21.3, ratio 3.02)
PASS  [diagnostic] Mc post-width / encoder-probe residual (0.77 — ENCODER-limited)
```

**Every pre-existing gate passes; all four sharpness gates fail.** This is the audit's central
methodological claim reproduced by an independent code path, and it is now a standing regression
test rather than a one-off analysis. The σ×SNR ratio (3.02), q informativeness (0.887) and
flow/encoder ratio (0.77) all match the E1 measurements (7.06→19.29, 0.89, 0.79) within noise.

Two implementation notes worth keeping:
- **Circular parameters use a circular statistic, not a linear std.** A sharp but bimodal sky
  posterior (the expected antipodal degeneracy) has a large linear std and was initially
  mislabelled "unmeasured". `ra`, `psi`, `phase` are now scored by `1 − R` (mean resultant length).
  With that fix `ra` correctly drops off the unmeasured list — timing triangulation *is* working —
  while `psi`/`phase` remain flagged, which is physically expected.
- **Non-finite informativeness counts as a failure.** NaN compares False against any threshold, so
  an unguarded `v > 0.85` would have let a degenerate parameter through silently.

*New finding surfaced by the gate:* `dec` (0.975) and `theta_jn` (1.009) are **also** at prior
width. theta_jn is the well-known distance–inclination degeneracy, but `dec` at 0.98 while `ra` is
informative is not obviously physical and deserves a look — the audit had not measured sky
informativeness at all. Not pursued here.

*Also fixed in passing:* `--help` crashed with `TypeError: not enough arguments for format string`
(an unescaped `%` in the `--n_events` help text) — pre-existing, unrelated to the diagnostics.

**These gates are expected to fail on every current checkpoint. Do not relax them to green** — the
failure they describe is the open work item (ranks 1–2), and a suite that reports PASS on a
return-the-prior model is exactly the problem being fixed.

---

## Deliverable C — Final recommendation

| option | verdict |
|---|---|
| fine-tune current model | **No** — cannot add information the context lacks |
| redesign flow | **No** — measured to have no headroom (post_std/probe = 0.79 / 0.96) |
| redesign parameterization | **No** — refuted here and previously by token-grafting |
| redesign objective | **No** — pure NLL is proper; the loss is not the problem |
| redesign dataset | **Partially yes** — MF bank coverage and dL bounds are real, cheap defects |
| keep current architecture | **No** for the encoder front-end |

**Recommendation: redesign the encoder's data representation, preceded by the cheap
dataset/instrumentation fixes (E-A, E-B, E-C, E-D, E-G).**

Sequence the cheap work first: it costs about a week, is low-risk, and E-A/E-B will materially
revise the expected payoff of the encoder work before that larger investment is committed.

**Revised in light of §11:** the encoder redesign must start with **E-E1 (time-shift alignment)**,
not with the SVD projection itself. The previous SVD attempt failed for reasons that are now
understood and are fixable, but the fix is alignment, not a better basis. If the alignment gate
fails, fall back to reducing the conv stem's 256× stride — less principled, but aimed at the same
quantity (6.6 GW cycles per token) and much cheaper. Treat "SVD front-end" as a *hypothesis with a
prerequisite*, not a committed plan.

**Expected outcomes, with explicit uncertainty.** Confidence is high that the flow, the mass
coordinates and the objective are *not* the bottleneck — those rest on direct measurement.
Confidence is high that the encoder is information-limited. Confidence is **moderate** that the
strided conv stem's phase destruction is the specific mechanism: it is a physically well-motivated
hypothesis consistent with every measurement, and DINGO's contrasting design is strong external
evidence, but it has not been tested by ablation here. E-E is the test. If a phase-preserving
front-end does *not* flatten σ_logMc × SNR, the mechanism hypothesis is wrong and the next suspect
is the pooling stage (8 learned queries compressing 183 tokens into a 256-D vector).

**A caution on success criteria.** Bilby-comparable *q* precision may not be reachable even with a
perfect encoder — GWTC's own q posteriors are broad (GW170608 q 90% CI ≈ [0.44, 1.0]). The target
is not a narrow q; it is a q posterior that correctly *excludes* q < 0.4 for this event. Combined
with correct Mc, that alone brings the m1 90% upper limit from ~60 M☉ to ~15 M☉ — which is the
entirety of the discrepancy in the brief.
