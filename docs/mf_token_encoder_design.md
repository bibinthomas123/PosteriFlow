# Encoder v3 design: matched-filter mass tokens (MF-tokens)

**Status (2026-07-21): IMPLEMENTED AND SHIPPED.** `--mf_tokens` landed in
`experiments/train_lean_npe.py` / `src/ahsd/models/coherent_encoder.py` on
2026-07-20 and has been trained through several runs (`lean_npe_mft`,
`lean_npe_mft_anneal`, `lean_npe_mf_scratch`, `lean_npe_noiseinv*`).
`lean_npe_mf_scratch` is the current best all-around checkpoint — passes
every population gate AND the synthetic basin probe, though real GW150914/
GW170814 strain still collapses (traced to MF-branch noise-realization
sensitivity, not this design). The "Sequencing" section below is historical
(both gating experiments it names have long since reported) — see
`analysis/attractor_basin/REPORT.md` for the live state of the
investigation, not this doc.

## Why the current encoder cannot resolve the quiet basin

Diagnosis chain (analysis/attractor_basin/REPORT.md, incl. 2026-07-19
addendum):

1. The measured-τ↔q correlation is **physical** (polarization fold for
   near-circular heavy systems). The geometry branch — ANY geometry branch —
   can only return the skewed population answer in the
   {τ≈+7 ms, heavy, quiet} cell.
2. The disambiguating information is the per-detector chirp phase evolution
   (how LVC measured q ≈ 0.86 for GW150914 at SNR 24 with 2 detectors).
3. The conv stem is structurally far from able to extract it at SNR 24. The
   q-signal at fixed Mc is a ~fraction-of-a-radian cumulative phase-evolution
   difference spread over ~1 s of data with per-sample SNR ≪ 1. The
   SNR-optimal extractor is correlation against coherent full-duration
   templates; a stride-8 conv stem on `asinh(strain)` with 64-sample kernels
   followed by pooling is not that, and no incentive (aux head, delay-aug,
   oversampling) can conjure capacity the architecture lacks. Delay-aug +
   aux-head (the dda run) tests exactly whether the capacity was there and
   merely unpaid; this design is the contingency if the answer is no.

**Requirement:** hand the encoder near-sufficient statistics for (Mc, q)
whose SNR efficiency matches matched filtering, and which are
delay/polarization-decoupled by construction.

## Core idea

A third token family alongside conv tokens and geometry tokens: per-detector
**complex matched-filter overlaps against a fixed (Mc, q) template grid**,
computed inside the model as batched FD ops.

For whitened data d_d and unit-normalized whitened template h_k:

    z_{d,k}(t) = irfft( D_d(f) · conj(H_k(f)) )      (complex analytic overlap)

Per (detector, template): peak |z| in the merger window, peak time, phase at
peak. Then:

- The pattern of |z| across the (Mc, q) grid **is** the profile likelihood
  over masses — the exact statistic LVC infers masses from. Discriminability
  check: mismatch between templates at (30,30) vs (44.8,29.1) (same Mc) is
  1−match ≈ 0.01–0.03, so ΔSNR² ≈ SNR²·(1−match) ≈ 6–17 at SNR 24 —
  informative but not decisive, which is precisely LVC's broad-but-
  informative q posterior. The current encoder gets ~none of this; the flow
  should get all of it.
- Inter-detector **peak-time differences are the polarization-immune delay
  estimator**: the template pins the full frequency evolution, so the
  carrier-cycle ambiguity that biases GCC (and, at low SNR, the phase
  slope) disappears; the polarization offset appears separately and cleanly
  as the inter-detector phase-at-peak difference. This retires the τ-feature
  problem instead of patching it again.

## Relation to the abandoned SVD front-end ("stage 1")

`src/ahsd/models/build_svd_basis.py` runs and writes
`assets/svd_basis.npz` (verified 2026-07-19), but **no code has ever
consumed it** — the stage-2 reduced-order front-end was never built. This
design supersedes that plan: an SVD-coefficient token branch (project
D_d(f) onto the 256 basis rows, feed complex coefficients as tokens) is the
compact-but-opaque variant of MF-tokens. Keep it as the fallback if the MF
grid underperforms; the MF grid is preferred because its output is directly
the mass-likelihood landscape and is offline-falsifiable per template.

## Bank

- Grid: log-spaced Mc 12→45 M⊙ (20 pts) × q {1.0, 0.85, 0.7, 0.55, 0.4,
  0.25} = **120 templates**; zero-spin, IMRPhenomXP (2,2), same family as the
  generator. Optional later: χ_eff ∈ {−0.3, 0, +0.3} planes (×3).
- Stored design-whitened per detector, unit-normalized, band 20–1024 Hz:
  buffer [3, 120, 4016] complex64 ≈ 12 MB. (Real-noise recolored data are
  whitened differently; the asd_bands branch already tells the model the
  per-band deviation — accept the mild mismatch, as everywhere else.)
- Generation reuses the waveform machinery in `build_svd_basis.py`.

## Cost control (the naive version is too slow)

Full-resolution z(t) needs B·3·120 length-16384 irffts per batch — too much.
Fold instead: alias the band-limited product spectrum into 1024 bins →
irfft(1024) gives |z(t)| at ~4 ms resolution over the whole window (exact
envelope decimation for peak FINDING), then refine peak time to sub-ms and
read phase analytically from the local FD phase slope around the coarse
peak. B=128: 46k tiny FFTs, batched — trivial on MPS/CPU workers.

## Integration (backward compatible, like every previous surgery)

- `MFBank` module: registered template buffer + the fold/peak/refine ops;
  features per (d, k): (|z|, t_peak, cosφ, sinφ) → [B, 3·120·4] → 2-layer
  MLP → **4 MF-tokens** [B, 4, d_model], passed through the existing
  `extra_tokens` path in `_compute_feats` (same mechanism as geometry
  tokens — concatenate `[gtok, mftok]`).
- **Zero-init output gate** on the MF-token projection so `--init_from` any
  existing checkpoint preserves the pretrained function exactly (same
  contract as the phase-residual zero-padding).
- Trainer flag `--mf_tokens`; no inference-interface change; checkpoint
  consumers unaffected (new keys are additive; the strict=False load path
  already logs fresh-init keys).

## Offline go/no-go gate — BEFORE any training run

The design is falsifiable for ~an hour of compute and zero training:
compute MF features for the frozen probe bank + ~200 delay-correct cell
injections spanning truth q 0.4–1.0, fit a logistic probe (q > 0.9 vs
q < 0.75) on MF features alone, versus the same probe on the current
geometry features (in-cell AUC expected ≈ 0.5–0.6 — that's the collapse).

- MF AUC ≳ 0.8 in-cell at SNR ~24 → implement and fine-tune.
- MF AUC ≈ geometry AUC → the statistic doesn't separate at this SNR in
  THIS noise; skip the architecture entirely and go straight to gate
  recalibration + explicit dynesty routing for the cell. (This outcome
  would also mean dynesty's own q posterior is near-prior in the cell —
  cross-check against the dynesty reference.)

## Risks

- **Waveform-family circularity:** bank = generator family. Fine for this
  dataset's science; flag for real-data claims (LVC does the same with
  wider banks).
- **Spin mismatch:** training prior a ∈ [0, 0.99] aligned vs zero-spin bank
  degrades |z| by the spin mismatch; the grid pattern stays informative.
  Add χ_eff planes only if the offline probe says spin blurs the q axis.
- **Overlapping signals:** peak window keeps the rank-0 (loudest) signal
  dominant; MF tokens describe it; rank conditioning is unchanged. Known
  limitation, matches how geometry features already behave.
- **Template count creep:** resist enlarging the bank before the offline
  probe demands it; 120 templates is already a 1440-dim raw feature.

## Alternatives considered (and why not first)

- **Complex TF front-end** (constant-Q spectrogram, instantaneous-frequency
  channels, 2D convs): generic, no bank bias, but pays the low-SNR
  extraction problem back to the network instead of solving it — the chirp
  track at SNR 24 is subthreshold per TF pixel. Higher training risk, not
  offline-falsifiable.
- **GNPE-style delay standardization** (DINGO): iteratively shift detectors
  to zero-lag using estimated delays, making the encoder geometry-blind by
  construction. Proven to reach LVC quality, but changes the inference loop
  (Gibbs iterations), breaks the one-shot amortized contract of the current
  pipeline, and is a rewrite, not a fine-tune. Escalation target if
  MF-tokens fail their offline gate but dynesty says the information exists.

## Sequencing

1. Wait for `lean_npe_dda` (running) — if the basin gate clears, shelve this
   doc; the architecture was sufficient.
2. Wait for the dynesty reference (running) — if optimal q>0.95 mass ≳0.30,
   recalibrate `basin_gate` first; re-score dda/pslope2 against the real
   optimum before deciding anything.
3. Only then: offline go/no-go probe (above), then MFBank implementation as
   a fine-tune from the best surviving checkpoint with
   `--delay_aug 0.5 --aux_mass 1.0` kept on.
