# Encoder audit: where the 54× Mc / 5× q deficit actually comes from

**Date:** 2026-07-21
**Model:** `model/lean_npe_mf_scratch/best_model.pth` (coherent encoder, mf_tokens, mc_q, ep40)
**Scripts (this dir):** `s1_stagewise_probe.py`; capacity/aliasing analysis and the PCA-matched
comparison are reproduced inline below.
**Reference:** `analysis/posterior_bias_audit/REPORT.md` (E-H: dynesty vs NPE, 12 events).

---

## 0. Root cause, stated up front

**There is no localized information-loss stage. The encoder does not *lose* the information — it
never *extracts* it.**

Three independent measurements converge on this, and each one contradicts a hypothesis in the brief
(including one I previously argued for myself):

1. **The conv stem compresses by only 1.40×.** Raw input is 49,152 numbers (3 det × 16,384); stem
   output is 35,136. `conv1` *expands* 4× (49,152 → 195,936). Aggressive-downsampling-destroys-
   information is **not** supported on capacity grounds.
2. **Attention pooling is the largest single compression (23.4×) and is measurably NOT lossy.**
   Dimension-matched PCA at 256 dims, n = 2400: post-fusion tokens **0.658**, pooled **0.677**,
   context **0.713** (log Mc). The context has the *highest* linearly-accessible information of the
   three. Nothing is being thrown away at the pool.
3. **Stage-wise probing shows information rising monotonically, with no drop anywhere** (§1).

So the information is present in the strain (dynesty extracts it), the architecture has the capacity
to carry it, and the pooling preserves what reaches it — yet the final context supports only
log Mc R² ≈ 0.71 and q R² ≈ 0.13.

**The gap is a learning / inductive-bias problem, not a compression problem.** Recovering Mc to
0.3% requires *coherent matched filtering*: correlating the data against a precise waveform over
hundreds of cycles, where the likelihood is violently oscillatory in the parameters. A strided CNN
plus transformer trained by NLL on ~50k events has to discover that operation from scratch. It does
not, and there is no reason to expect it to.

**The supporting evidence is that every measured Mc gain in this project came from giving the
network matched-filter output explicitly, never from changing the learned path:**

| change | log Mc probe R² |
|---|---|
| learned conv path alone (context, no MF branch: `lean_npe_mcq`) | 0.670 |
| + MF tokens, 120 templates | 0.711 |
| MF features alone, 120-template bank | 0.525 |
| MF features alone, 614-template dense bank | 0.641 (in-bank) |
| MF features alone, + shape-normalized channel | **0.597 → 0.670** (SNR>15) |
| MF features alone, 0.30%-spaced fine bank | **0.508** at n=303 (probe-limited, not feature-limited) |

**Smallest change with the highest expected impact: replace the hand-built template bank with
SVD-compressed matched filtering, so the network sees the equivalent of 10³–10⁴ templates at the
cost of ~200 basis projections — with time alignment as its prerequisite.** This is *not* "adopt
DINGO's encoder"; it is using `build_svd_basis.py` for what an SVD bank is actually for in GW
search: making a dense matched filter affordable.

---

## 1. Information-loss localization

`s1_stagewise_probe.py`, n = 1600, median SNR 20.2. Every stage reduced to **the same 384 dims** by
a fixed Gaussian random projection and probed with the same RidgeCV/5-fold estimator — matched probe
power, because an earlier step of this audit reached a wrong conclusion by comparing a 10⁴-feature
probe at n = 300 against a 256-feature probe at n = 1200.

| stage | log Mc R² | Δ | q R² | Δ |
|---|---|---|---|---|
| 0 raw strain | −0.002 | | −0.003 | |
| 1 asinh strain | −0.003 | −0.001 | −0.004 | −0.002 |
| 2 stem tokens | 0.247 | **+0.250** | 0.017 | +0.022 |
| 3 tokens + pos/det embed | 0.259 | +0.013 | 0.019 | +0.001 |
| 4 post-fusion | 0.667 | **+0.408** | 0.087 | **+0.068** |
| 5 pooled | 0.690 | +0.023 | 0.104 | +0.017 |
| 6 energy branch (side) | 0.096 | — | −0.008 | — |
| 7 geometry+MF tokens (side) | 0.413 | — | 0.008 | — |
| 8 context | 0.705 | +0.015 | 0.118 | +0.014 |

**Critical caveat, stated because it is easy to misread:** rows 0–1 probe ≈ 0 **not** because raw
strain lacks information — dynesty extracts Mc to 0.3% from exactly this array — but because a
*linear* probe on a *randomly projected* time series cannot express matched filtering. These numbers
measure **linear accessibility**, not information content. The stage-to-stage *deltas* are the
meaningful quantity.

**Verdict per stage:**

| stage | could it explain 54× Mc / 5× q? | evidence |
|---|---|---|
| whitening | **No** | shared with dynesty; `prepare_simulated` verified against basin_probe SNR labels (87.0 vs "snr85") |
| projection / asinh | **No** | monotone, bounded, invertible on the observed range; Δ ≈ 0 |
| early convolutions | **Unlikely** | 4× *expansion*, 1.40× total stem compression; largest single positive Δ after fusion |
| downsampling | **Unlikely** (capacity) / **inconclusive** (aliasing) | see §2 |
| tokenization | **No** | +0.013 for pos/det embedding; not a bottleneck either way |
| transformer | **No** — it is doing the most work | largest Δ (+0.408 Mc, +0.068 q) |
| pooling | **Refuted** | PCA-256 matched test: post-fusion 0.658 ≤ pooled 0.677 ≤ context 0.713 |
| context vector | **No** | highest accessibility of all stages |

---

## 2. Strided convolutions

Measured geometry (`conv1..4`, strides 8·4·4·2 = 256×):

| stage | numbers/detector | total (×3) | cumulative compression |
|---|---|---|---|
| raw whitened strain | 16,384 | 49,152 | 1.00× |
| conv1 k=64 s=8 c=32 | 65,312 | 195,936 | **0.25× (4× expansion)** |
| conv2 k=16 s=4 c=64 | 32,448 | 97,344 | 0.50× |
| conv3 k=8 s=4 c=128 | 16,000 | 48,000 | 1.02× |
| conv4 k=4 s=2 c=192 | 11,712 | 35,136 | **1.40×** |

**"Could the first convolution discard phase evolution?" — not by capacity.** conv1 maps 64 input
samples to 32 channels every 8 samples: 4× oversampled. A multirate filterbank at that redundancy
*can* be information-preserving (this is how QMF/wavelet decompositions work). Whether the *learned*
filters are is a separate question — see the experiment in §9.

Aliasing arithmetic, for the record:

| after | token rate | representable band |
|---|---|---|
| conv1 | 512 Hz | 0–256 Hz |
| conv2 | 128 Hz | 0–64 Hz |
| conv3 | 32 Hz | 0–16 Hz |
| conv4 | **16 Hz** | **0–8 Hz** |

vs an analysis band of [20, 1024] Hz and f_ISCO = 4400/M_total Hz (M=10 M☉ → 440 Hz; M=40 → 110 Hz).
The **token sequence** cannot represent phase evolution faster than 8 Hz — but the 192 channels can
carry within-token structure, so this is a constraint on the transformer's cross-token modelling,
not a proof of information destruction.

**Verdict: INCONCLUSIVE on aliasing, REFUTED on capacity. Expected impact of reducing stride: LOW.**
Halving the stride (256→128) doubles token count 61→122, roughly doubles stem FLOPs and transformer
cost (attention is quadratic in tokens: 183→366 tokens ⇒ ~4× attention FLOPs), for a change that
capacity analysis says was not the constraint. Cost: ~1 day + a full retrain. Regression risk:
moderate (every checkpoint invalidated). **Do not do this first.**

---

## 3. Phase preservation

The current path feeds `asinh(strain)` — a real-valued, amplitude-monotone transform. Phase is
implicitly present but never made explicit.

**Supported by evidence:** the MF branch, which *is* an explicit coherent-phase operation (complex
overlap, `cos φ`/`sin φ` channels), is the only component whose additions have produced measurable
Mc gains (§0 table). That is indirect but consistent evidence that explicit phase helps and learned
phase extraction does not happen on its own.

**Options ranked by evidence, not novelty:**

| option | verdict | rationale |
|---|---|---|
| complex/analytic-signal input, real+imag channels | **worth testing** | costs one Hilbert transform; makes phase explicit without changing topology |
| frequency-domain input | **worth testing, but see §5** | the coherent encoder already ingests FD for its geometry branch |
| instantaneous frequency as an extra channel | **low priority** | derived, noisy at low SNR |
| learned phase embeddings | **not supported** | no evidence any phase representation is being *learned*; adding capacity to an unused pathway |

---

## 4. Multi-scale representation

**Verdict: NOT SUPPORTED as a primary fix.** Gradual-vs-immediate compression is the wrong axis:
§2 shows compression through the stem is 1.40×, i.e. there is barely any compression to make more
gradual. U-Net/FPN/WaveNet/dilated stacks would change *how* features are formed, not how much
capacity survives, and nothing measured points at capacity.

Dilated convolutions have one genuine argument in their favour — a long receptive field without
striding, matching the fact that Mc information is spread over the whole 4 s inspiral. If the §9
reconstruction test shows the stem *is* aliasing, dilation becomes the cheapest remedy. Until then
it is speculative. **Expected impact: LOW–MEDIUM. Cost: MEDIUM. Risk: MEDIUM.**

---

## 5. Frequency-domain representation

The coherent encoder **already** computes an FD representation — `_geometry_rel` operates on
`rfft(x)` over [20, 1024] Hz in 16 log bands — and the stage probe puts the geometry+MF token block
at log Mc R² 0.413 on its own. So "move to the frequency domain" is not an untried idea here; the
question is *at what resolution*.

The current geometry branch uses **16 log-spaced bands** across 4016 frequency bins — a ~250×
reduction that discards exactly the fine phase structure Mc depends on. That is a real, specific
weakness, and it is consistent with everything above: the FD path exists but is summarized far too
coarsely to support matched filtering.

STFT/wavelets/complex spectrograms: **not recommended.** A CBC inspiral is a deterministic chirp
whose optimal statistic is a matched filter, not a time–frequency energy map; spectrogram magnitude
discards precisely the coherent phase that carries Mc. Literature support for the matched-filter
framing over spectrograms in CBC PE is unambiguous (this is why every production search — GstLAL,
PyCBC — is a matched-filter bank, and why DINGO conditions on SVD-projected *complex* FD strain
rather than a spectrogram).

---

## 6. Alignment

Measured (`analysis/posterior_bias_audit/eE1_alignment_gate.py`, `eB_dense_bank_gates.py`):

| bank | global median &#124;residual&#124; | Mc<12 | in-bank, SNR>20 |
|---|---|---|---|
| 120-template | 186.6 ms | 626.1 ms | 13.2 ms (p90 38.9) |
| 614-template dense | **35.2 ms** | 154.7 ms | 20.1 ms (p90 825) |

Light-travel floor ≈ 10 ms.

**Alignment should happen BEFORE encoding, and it should be estimated, not learned end-to-end.**
Reason: alignment is what makes a *fixed* basis (SVD) compact — a τ = 1.5 s shift sweeps ~1500
cycles of phase ramp across [20,1024] Hz, so an unaligned basis needs orders of magnitude more
components. This is precisely why the previous SVD attempt failed (§7) and precisely what GNPE
exists to solve.

| approach | verdict |
|---|---|
| **GNPE-style iterative alignment** | **recommended** — estimate τ, align, re-estimate; the residual p90 tail (39 ms in-bank) is exactly what iteration tightens |
| coarse-to-fine localization | equivalent to the above; fold in |
| MF peak-time delay prediction | **already available and mostly working** — this is the estimator measured above; reuse it |
| cross-correlation alignment | already present as the GCC τ_ij branch, but gives *inter-detector* delay, not absolute t_c — insufficient alone |

---

## 7. SVD — and the reframing that matters

The previous attempt (`analysis/encoder_ab_compare/verdict.json`) was invalid for four independent
reasons already documented: 4 epochs / 6000 events / both arms `init_from` a conv model and flagged
unstable; a basis built with ±15 ms shifts against data spanning ±1.5 s (100× mismatch); 800
waveforms truncated to 256 components (`captured_power = 0.99999999998` is vacuous — it is nearly
interpolating its own build set); and mass coverage Mc ≳ 4.4 with plus-polarization only.

**But the more important point is that the audit has been framing SVD wrongly, including in my own
earlier recommendation.** An SVD basis in GW search is not primarily a *data compression* trick —
it is what makes a **dense template bank affordable**. Overlaps against K templates are computed by
projecting the data onto ~N_b ≪ K basis vectors and recombining. §0 established that explicit
matched filtering is the only thing that has ever moved Mc in this project, and §7-of-the-other-report
established that reaching 0.3% Mc spacing across the prior needs ~1.8×10⁴ templates — unaffordable
directly, entirely affordable through a basis of ~200–400.

**That converts SVD from "a different front-end" into "the mechanism that makes the one demonstrably
effective component (matched filtering) dense enough to matter."** That is a much stronger argument
than the data-compression one I gave earlier, and it is grounded in this project's own measurements
rather than in DINGO's architecture.

**Verdict: WORTH REVISITING, as alignment-aware SVD-compressed matched filtering.** Conventional
convolutional improvements are *not* likely to achieve the same benefit more cheaply, because §1–§2
show the convolutional path is not where the loss is.

---

## 8. Alternative encoder families

Ranked by suitability for CBC PE, not popularity. The ranking criterion is: *does this family make
coherent matched filtering easier to represent or learn?*

| family | verdict | reasoning |
|---|---|---|
| **SVD-projected FD + existing transformer** | **1st** | directly supplies the sufficient statistic; smallest change; reuses the trained downstream stack |
| **Dilated / large-receptive-field CNN** | 2nd | long context without striding; the cheap remedy *if* §9 shows aliasing |
| Conformer | 3rd | conv+attention hybrid suits local waveform structure + global phase coherence, but is a large rewrite for an unquantified gain |
| Mamba / S4 / state-space | 4th | strong on very long sequences; a linear SSM is *in principle* a learned IIR filterbank, which is matched-filter-adjacent — genuinely interesting, entirely unvalidated here |
| ConvNeXt-1D / ResNet | 5th | better-trained CNNs; addresses optimization, not the missing operation |
| U-Net / FPN | 6th | built for dense per-sample output; PE needs a global summary |
| Perceiver IO | 7th | its whole point is aggressive cross-attention compression to a latent array — the one thing measured *not* to be the bottleneck |
| Hyena, Wave Transformer | 8th | no evidence base for this problem |

**None of these should be adopted on architectural grounds alone.** The measured deficit is a missing
*operation* (dense coherent matched filtering), not a missing *architecture*.

---

## 9. Diagnostic experiments, ranked by information per unit effort

| # | experiment | distinguishes | effort | value |
|---|---|---|---|---|
| **D1** | **Stem invertibility / aliasing test.** Train a small decoder to reconstruct the input from `stem` output (or: probe stem output for the MF SNR time series). | Settles the one INCONCLUSIVE verdict in §2 — whether the strided convs alias. Decides whether dilation/stride reduction is worth anything at all. | ~½ day | **Highest** |
| **D2** | **SVD-projected features probe (no training).** Align with the MF peak time, project onto a properly built basis, probe log Mc / q — exactly as done for MF features. | Whether the SVD path can supply what the conv path cannot, *before* committing to a retrain. | ~1 day | **Highest** |
| **D3** | Nonlinear probe (small MLP, trained on 20k+ events) on the frozen context. | Whether the context's information is present but non-linearly encoded — i.e. whether R²=0.71 is a probe artifact. Guards against exactly the error made earlier in this audit. | ~½ day | High |
| **D4** | Frequency-resolution sweep of the geometry branch: 16 → 64 → 256 bands, probe only. | Tests §5's specific claim that 16 log bands is the FD bottleneck. | ~½ day | High |
| **D5** | Reduced-stride stem, probe-only (random-init and trained). | Confirms/refutes §2 directly. | ~1 day + retrain | Medium |
| **D6** | Complex/analytic-signal input channels, probe-only. | §3. | ~½ day | Medium |
| D7 | CCA / MI estimates between stages | More rigorous version of §1; unlikely to change the conclusion | ~1 day | Low |

**D1 and D2 first — both are probe-only, need no retraining, and between them they decide the
architecture.**

---

## 10. Final recommendation

**Root cause:** the encoder has adequate capacity at every stage (stem 1.40× compression, pooling
measurably non-lossy) but never learns the coherent matched-filter operation that Mc precision
requires. The deficit is missing inductive bias, not lost capacity.

**Ranked weaknesses:**

| # | weakness | evidence | impact |
|---|---|---|---|
| 1 | No dense coherent matched filter anywhere in the model | every Mc gain came from explicit MF; learned path plateaus at R² 0.71 | **High** |
| 2 | FD path summarized to 16 log bands (~250× reduction) | §5; geometry tokens alone reach R² 0.413 | **High** |
| 3 | MF bank 30× too sparse in Mc to resolve the Mc–q ridge | 0.30% needed vs 5.9% deployed | **High** |
| 4 | Alignment residual 35 ms global, heavy tail | §6 | Medium (gates #1/#3) |
| 5 | `mf_gate` zero-init trained to 0.026 | posterior_bias_audit §7b | Medium |
| 6 | Token rate 16 Hz limits cross-token phase modelling | §2 | Low–Medium, unconfirmed |
| 7 | Strided conv aliasing | §2, **inconclusive** | Unknown — D1 decides |

**Try first, in order:** D1 and D2 (probe-only, no retrain) → then alignment-aware SVD-compressed
matched filtering as the single architectural change, with the existing transformer/flow retained.

**Do NOT pursue on current evidence:**
- reducing conv stride (capacity refutes it; D1 may revive it)
- U-Net / FPN / multi-scale gradual compression (no compression problem to solve)
- Perceiver IO (compresses at the stage measured not to be lossy)
- spectrograms / STFT / wavelets (discard the coherent phase that carries Mc)
- replacing pooling (PCA-matched test: pooling is not lossy)
- larger flow, different mass coordinates, different SBI objective (settled in the prior audit)

**Expected improvement:** if dense SVD-compressed matched filtering reaches the Mc information the
fine-bank probe suggests is available, log Mc informativeness should move from 0.429 toward the
0.0028 dynesty reference — but **no honest point estimate is possible before D2**, which is exactly
why D2 is probe-only and comes first. Anyone quoting a number before then is guessing.

**Regression risk:** the SVD front-end invalidates every existing checkpoint and requires a full
retrain; the population/calibration gates (SBC, coverage, railing, basin probe) must all be re-run,
and the new sharpness gates in `validate_checkpoint.py` — now anchored to the dynesty optima
(chirp_mass 0.00276, mass_ratio 0.1670) — are the acceptance criteria.
