# PosteriFlow — Neural Posterior Estimation for Overlapping Gravitational-Wave Signals

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Amortized, calibrated posterior estimation for overlapping compact-binary
> coalescences in multi-detector strain — seconds per event instead of
> minutes-to-hours, with per-signal posteriors for up to 5 overlapping sources.

---

## 1. The problem

Ground-based gravitational-wave detectors (LIGO Hanford/Livingston, Virgo)
observe compact-binary mergers as chirping strain signals buried in colored
noise. Parameter estimation — recovering masses, distance, sky position,
spins from the strain — is classically done with stochastic samplers
(dynesty, bilby-MCMC) that evaluate a waveform likelihood hundreds of
thousands of times per event. That costs minutes to hours per event and,
more fundamentally, assumes **one signal per analysis segment**.

As detector sensitivity improves (O4/O5, and especially next-generation
observatories like Einstein Telescope), the in-band event rate rises until
**signals overlap in time**. For overlapping sources the single-signal
likelihood is simply wrong: sampling one signal while another sits in the
data biases both. Joint N-signal sampling exists but scales combinatorially.

**PosteriFlow's answer** is simulation-based inference: train a neural
network on millions of simulated (signal + noise → parameters) examples so
that, at observation time, a *single forward pass* returns the posterior.
The network never needs an explicit likelihood, and overlap handling is
built into the training distribution rather than bolted onto a sampler.

The estimator, **LeanNPE**, answers queries of the form:

> *"Given this 4 s of whitened H1/L1/V1 strain, what is the posterior over
> the 11 parameters of the k-th loudest signal present?"*

Parameters: `mass_1, mass_2, luminosity_distance, ra, dec, theta_jn, psi,
phase, geocent_time, a1, a2` (aligned spins; `mass_1 ≥ mass_2` convention).

A separately trained **PriorityNet** ranks signals so the downstream
extract-and-subtract pipeline (`src/ahsd/core/`) can iterate: infer loudest
→ subtract → re-infer.

---

## 2. How it works

### 2.1 The model (LeanNPE, ~6M parameters)

```
whitened strain [3, 16384] ──► norm-free conv stem (asinh input) ──► 61 tokens/det
                                        │  + detector & positional embeddings
                                        ▼
                        3-layer pre-norm Transformer (cross-detector fusion)
                                        │  attention pooling (8 learned queries)
raw strain ──► per-window log-energy ───┤
                                        ▼
                              context vector (256) ⊕ signal-rank embedding (32)
                                        ▼
                    Neural Spline Flow (8 layers, 16 bins) ──► posterior samples
```

- **Conv stem, no normalization layers.** The strain is already whitened
  (unit noise floor), so absolute amplitude *is* the distance/SNR cue.
  Per-sample normalization (InstanceNorm, per-detector std division,
  LayerNorm stacks) provably destroyed it in a previous architecture — the
  stem instead sees `asinh`-compressed strain (amplitude-monotone, bounded),
  and a parallel branch feeds per-window log-energies of the *raw* strain so
  amplitude survives any internal normalization downstream.
- **Cross-detector Transformer.** Sky position and distance/inclination
  information lives in amplitude ratios and time delays *between* detectors;
  self-attention across all detectors' tokens fuses it.
- **Rank conditioning.** For overlap events, the flow's context includes an
  embedding of the queried signal's loudness rank. Without it, one context
  would be trained to predict *every* overlapping signal's parameters — an
  irreducibly ambiguous (mixture) target. This single embedding turns the
  overlap problem into a well-posed conditional.
- **Neural Spline Flow** (rational-quadratic, autoregressive; built on
  `nflows`) maps a standard normal to the 11-D posterior, conditioned on the
  296-D context. Parameters are mapped to [−1, 1] by a fixed, deterministic
  scaler (log-space for masses and distance) — no fitted statistics that can
  drift between training and inference.
- **Pure NLL objective.** No auxiliary losses, no diversity regularizers, no
  posterior-width penalties. Every such term tried in this project's history
  was measured to optimize the *statistics* of the representation rather
  than its *information content* (see §5).

### 2.2 The data engine: component storage + on-the-fly remixing

The single most important lesson from this project: **for NPE, the dataset
design matters more than the architecture.** Two failure modes were measured
and fixed here:

1. **Fixed noise → memorization.** If each training event stores one frozen
   `signal + noise` sum, the network eventually memorizes the noise wiggles
   (train NLL → −8 while validation → +8, observed). Noise must be *fresh*
   every epoch.
2. **One noise draw per signal → wrong widths.** Posterior width is learned
   from seeing how much the data can vary for fixed parameters. One noise
   realization per event under-teaches exactly the quantity calibration
   depends on.

So the generator (`src/ahsd/data/`, fully bilby-based: IMRPhenomXP /
NRTidalv2 waveforms, design-sensitivity PSDs, matched-filter SNR accounting)
stores every event as **separate whitened components** — the noise and each
signal individually (float16, exact to the summed strain within float16
rounding, verified). The training loader (`experiments/v2_remix_data.py`)
then assembles a *different* example from the same ingredients every epoch:

| Augmentation | Mechanism | Label handling |
|---|---|---|
| Noise swap | any of 40k+ stored noise realizations, or real O3 crops | — |
| Real-noise re-coloring | `irfft(rfft(sig)·ASD_design/ASD_measured)` — exact, since whitening is diagonal in frequency | — |
| Time shift | circular roll ±0.1 s, identical across detectors (preserves inter-detector delays) | `geocent_time += δt` |
| Distance rescale | amplitude × s (exact leading-order GR) | `d_L ÷ s` |
| Overlaps | 2–5 signals summed; per-event loudness re-sort after rescaling | rank labels stay consistent |
| Pre-merger | BNS/NSBH with merger 0.5–3 s *past* the window end (early-warning regime) | `--premerger` widens the time scaler |

37k stored events × fresh noise pairings ≈ effectively unlimited training
data; the memorization channel is closed by construction (verified: val NLL
tracks train NLL ~2.7× longer and ~0.35 nats deeper than with fixed noise).

### 2.3 Real detector noise

A model trained purely on Gaussian design-PSD noise **does not transfer** to
real O3 data — measured here as NLL 0.7 → 27 and 90% coverage 0.93 → 0.35 on
real-noise injections. The gap is spectral (measured-PSD shape,
non-stationarity, glitches), not an amplitude-scale artifact (normalizing
crops recovers almost nothing).

The remedy is built into the same loader — no separate pipeline:

- `scripts/download_gwosc_noise_bank.py` fetches 64 s O3b segments per
  detector from GWOSC, **whitens each manually with its own measured ASD**
  (the saved ASD and the applied whitening filter are the same array, by
  construction), and stores strain + ASD under `data/noise_bank/`.
- `RemixDataset(real_noise_dir=…, real_noise_prob=p)` draws, per event per
  epoch, real crops with probability *p* and Gaussian noise otherwise —
  re-coloring the stored design-whitened signals into each segment's
  whitening exactly. All other augmentations apply unchanged.
- Training logs metrics on **two fixed validation sets each epoch**
  (Gaussian and real-noise), and checkpoint selection uses their mean — so
  robustness on real data cannot silently trade away simulated-data
  performance, or vice versa.

Mixing ratios (0 / 0.25 / 0.5 / 0.75 / 1.0) are pure configuration; no code
changes between experiments.

---

## 3. Measured results

All numbers from this repo's diagnostics on held-out validation data
(`analysis/`), not projections.

**Core performance (v2 checkpoint, Gaussian noise, 1.5–2k events):**

| Metric | Value | Why it matters |
|---|---|---|
| Shuffle ΔNLL | **+10.1 nats** | *The* conditional-inference test: NLL degradation when contexts are shuffled across events. 0.000 = the model ignores the data entirely (the previous architecture scored exactly that). |
| Coverage, all 11 params @ 50/68/90/95% | within ±3–5% of nominal | Posteriors are honest: a 90% interval contains the truth ~90% of the time. |
| Distance correlation (log-median vs truth) | 0.65 | Bounded below 1 by the physical distance–inclination degeneracy. |
| Chirp-mass error vs SNR | 43% → 8.4% (SNR 6 → 100) | Scales with loudness as it should — no plateau. |
| Overlap severity (2–4 signals, min Δt down to <0.3 s) | no degradation vs singles | The rank-conditioned design solves the overlap task rather than degrading gracefully. |
| Inference cost | ~5 s / 3,000 samples | vs ~7 min for dynesty on the same event (≈100×), on a laptop. |

**Independent cross-check:** bilby/dynesty, fed the *unwhitened* strain +
PSDs from this repo's own generator, recovers injected truths within ~1σ —
externally certifying the injection/PSD/SNR bookkeeping chain
(`scripts/dynesty_compare.py`).

**Known limits (measured, not hidden):**

- Distance error plateaus at ~23% above SNR ≈ 16, with mild core
  overconfidence in the loudest bin — the model stops extracting amplitude
  information where an optimal estimator would keep sharpening.
- Timing σ ≈ 65 ms — the encoder's temporal stride — versus dynesty's
  ~0.3 ms phase-coherent timing. Sky localization inherits this (RA/dec far
  wider than dynesty): one bottleneck, two symptoms. Fixing it requires
  phase-preserving features in the encoder; it is the top architectural
  target for a next iteration.
- SBC rank histograms show small systematic slopes for `mass_1` and
  `distance` (slight underestimation) and wrap-around artifacts for circular
  parameters; interval coverage is unaffected, but publication-grade SBC
  would want post-hoc quantile recalibration.

---

## 4. Using it

### Setup

```bash
conda activate ahsd          # environment with torch (MPS/CUDA), bilby, gwpy, nflows
pip install -e . --no-deps
```

### Generate a dataset (component storage on by default)

```bash
python src/ahsd/data/scripts/generate_dataset.py \
    --config configs/data_config.yaml --n-samples 50000 --output-dir data/dataset
```

`configs/data_config.yaml` controls event mix (BBH/BNS/NSBH/noise), overlap
fraction, SNR gate, pre-merger fraction, and component storage. ~2 h for 50k
events on a laptop; ~22 GB.

### Train from scratch (remix, Gaussian noise)

```bash
python -u experiments/train_lean_npe.py --data data/dataset --remix \
    --epochs 40 --batch 128 --outdir model/lean_npe_v2
```

First run builds a memmap cache (`data/dataset/memmap/`) so epochs stream
from disk without holding 22 GB in RAM. ~9 min/epoch on an M-series Mac.

### Download real noise & fine-tune

```bash
python scripts/download_gwosc_noise_bank.py --per_det 120 --outdir data/noise_bank

python -u experiments/train_lean_npe.py --data data/dataset --remix \
    --noise_bank data/noise_bank --real_noise_prob 0.5 \
    --init_from model/lean_npe_v2/best_model.pth \
    --lr 1e-4 --epochs 20 --outdir model/lean_npe_v3
```

### Reading the per-epoch log

```
epoch 12 | train 1.72 | val 2.00 | shufΔ +2.46 | dcorr +0.45 | dcov50/90 0.52/0.95 | gnorm 41 | 534s | REAL val 3.1 dcorr +0.31 dcov 0.44/0.86
```

| Field | Healthy | Failure signature |
|---|---|---|
| `shufΔ` | climbs steadily, several nats | pinned near 0 → marginal fit; stop and diagnose the encoder, don't train longer |
| `dcorr` | lifts off 0 by ~epoch 10–15, → 0.6+ | flat while `shufΔ` grows → amplitude not being extracted |
| `dcov50/90` | ~0.50/0.90 throughout (±0.04 noise) | steady downward drift → overconfidence creeping in |
| `val` vs `train` | fall together | val flat/rising while train falls → memorization; the data, not the model, is the bottleneck |
| `REAL …` block | converges toward the Gaussian block | stays far apart → increase `real_noise_prob` or bank size |

Checkpoints save on best validation NLL (mean of both domains when a noise
bank is configured), so a run left too long never loses its best model.

### Certify a checkpoint

```bash
python scripts/lean_npe_diagnostics.py  --model model/lean_npe_v2/best_model.pth \
    --data data/dataset --outdir analysis/diag        # SBC, PP, coverage; add --noise real for O3 variant
python scripts/lean_npe_extended_eval.py               # error vs SNR bins, overlap severity, calib splits
python scripts/dynesty_compare.py --n_events 30        # head-to-head vs bilby/dynesty
```

### Sample a posterior in code

```python
import torch
from ahsd.models import LeanNPE

ckpt = torch.load("model/lean_npe_v2/best_model.pth", map_location="cpu", weights_only=False)
model = LeanNPE(premerger=ckpt["args"].get("premerger", False))
model.load_state_dict(ckpt["model_state_dict"]); model.eval()

strain = torch.randn(1, 3, 16384)   # whitened H1/L1/V1 @ 4096 Hz, 4 s window
post = model.sample_posterior(strain, rank=0, n_samples=3000)   # [1, 3000, 11] physical units
# rank=0 queries the loudest signal; rank=1 the second-loudest, etc.
```

---

## 5. Why it looks like this: the evidence trail

This architecture is the survivor of a measured post-mortem, kept short here
because the details live in `analysis/` and the git history:

- The predecessor (`OverlapNeuralPE`, 22.7M params: tokenized encoder,
  11 per-parameter attention queries, masked-context flow, 6 auxiliary
  losses) produced a context that was **constant across events** — shuffle
  ΔNLL exactly 0.000, linear-probe R² < 0 for every parameter, posterior
  essentially identical for every input. A pure marginal fit.
- Root causes, each verified: stacked per-sample normalizations erased
  per-event amplitude information; readout attention collapsed to uniform
  over near-identical tokens; the training loop asked one context to predict
  all overlapping signals with no query conditioning (a mixture target); and
  the loss was dominated by regularizers (weights 20 and 15) that sculpted
  statistics while a 6e-6 learning rate starved the flow.
- Every "fix" that penalized statistics (VICReg, variance floors, diversity
  losses, posterior-std targets) moved the statistic and left inference
  unchanged. The fixes that worked changed **what information reaches the
  model** (amplitude-preserving input path, rank conditioning) and **what
  the objective rewards** (pure NLL on non-memorizable data).

The forensic methodology — shuffle tests, linear probes, coverage/SBC before
architecture opinions — is itself the reusable asset; the diagnostics
scripts encode it.

---

## 6. Project structure

```
src/ahsd/
├── models/
│   ├── lean_npe.py            # LeanNPE: encoder + rank embedding + NSF (the model)
│   ├── flows.py               # Neural Spline Flow (nflows-based)
│   ├── parameter_scalers.py   # legacy scalers (flows.py imports FLOW_NORM_BOUND)
│   └── transformer_encoder.py # strain encoder used by PriorityNet
├── core/                      # PriorityNet, adaptive subtractor, end-to-end pipeline
├── data/                      # bilby-based generation: sampler, injector, whitener, PSDs
│   └── scripts/generate_dataset.py
└── evaluation/                # metrics

experiments/
├── train_lean_npe.py          # trainer: remix, real-noise mixing, fine-tune, dual validation
├── v2_remix_data.py           # memmap cache + RemixDataset (all augmentations, both noise sources)
└── train_priority_net.py      # PriorityNet training

scripts/
├── download_gwosc_noise_bank.py  # real O3 bank: exact-ASD whitening, strain + ASD pairs
├── lean_npe_diagnostics.py       # SBC / PP / coverage certification (gaussian | real)
├── lean_npe_extended_eval.py     # error vs SNR, overlap severity, calibration splits
├── dynesty_compare.py            # convention-audited head-to-head vs bilby/dynesty
└── real_noise_test.py            # real-noise robustness probe

analysis/     # measured evidence: diagnostic reports, audits, comparison outputs
configs/      # YAML configs (data_config.yaml is the main one)
data/         # dataset (component storage), memmap caches, noise_bank
model/        # checkpoints (best_model.pth includes weights + args + diagnostics)
```

## 7. Roadmap

1. **Real-noise fine-tune** (infrastructure complete) → then inference on
   real GWTC events with GWTC-catalog comparison.
2. **Encoder temporal resolution** — phase-preserving features to break the
   65 ms timing floor; unlocks sky localization and the high-SNR distance
   plateau.
3. Pre-merger (early-warning) training run (`--premerger`; data already in
   the dataset).
4. Post-hoc quantile recalibration for publication-grade SBC.
5. End-to-end overlap pipeline evaluation: PriorityNet ranking → LeanNPE
   posteriors → adaptive subtraction, on real-noise overlaps.

## References

- Dax et al., *Real-Time Gravitational Wave Science with Neural Posterior
  Estimation* (DINGO): [arXiv:2106.12594](https://arxiv.org/abs/2106.12594)
- Durkan et al., *Neural Spline Flows*: [arXiv:1906.04032](https://arxiv.org/abs/1906.04032)
- Ashton et al., *Bilby*: [arXiv:1811.02042](https://arxiv.org/abs/1811.02042)
- Talts et al., *Simulation-Based Calibration*: [arXiv:1804.06788](https://arxiv.org/abs/1804.06788)
- GWOSC open data: [gwosc.org](https://gwosc.org)

## License & citation

MIT — see [LICENSE](LICENSE).

```bibtex
@software{thomas2026posteriflow,
  title  = {PosteriFlow: Calibrated Neural Posterior Estimation
            for Overlapping Gravitational-Wave Signals},
  author = {Thomas, Bibin},
  year   = {2026},
  url    = {https://github.com/bibinthomas123/PosteriFlow}
}
```

**Author:** Bibin Thomas — bibinthomas951@gmail.com

*Last updated: 2026-07-07*
