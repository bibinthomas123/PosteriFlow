# PosteriFlow — Neural Posterior Estimation for Overlapping Gravitational-Wave Signals

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Amortized, calibration-audited posterior estimation for compact-binary
> coalescences in whitened 3-detector strain — seconds per event instead of
> minutes-to-hours, with per-signal posteriors for time-overlapping sources.

---

## 1. The problem

Ground-based detectors (LIGO Hanford/Livingston, Virgo) observe compact-binary
mergers as chirping strain buried in colored noise. Parameter estimation —
recovering masses, distance, sky position, spins — is classically done with
stochastic samplers (dynesty, bilby-MCMC) that evaluate a waveform likelihood
hundreds of thousands of times per event. That costs minutes to hours per
event and, more fundamentally, assumes **one signal per analysis segment**. As
sensitivity improves (O4/O5, Einstein Telescope), the in-band rate rises until
**signals overlap in time**, where the single-signal likelihood is simply
wrong.

**PosteriFlow's answer** is simulation-based inference: train a neural network
on millions of simulated `(signal + noise → parameters)` examples so that, at
observation time, a *single forward pass* returns the posterior. No explicit
likelihood; overlap handling is built into the training distribution rather
than bolted onto a sampler.

The estimator, **LeanNPE**, answers:

> *"Given this 4 s of whitened H1/L1/V1 strain, what is the posterior over the
> 11 parameters of the k-th loudest signal present?"*

Parameters (fixed order): `mass_1, mass_2, luminosity_distance, ra, dec,
theta_jn, psi, phase, geocent_time, a1, a2` (aligned spins; `mass_1 ≥ mass_2`
convention). Overlaps are handled by **rank conditioning** — query rank 0 for
the loudest signal, rank 1 for the second-loudest, etc.

---

## 2. How it works

### 2.1 The model (`src/ahsd/models/lean_npe.py`, ~6M parameters)

```
whitened strain [3, 16384] ─┬─► norm-free conv stem (asinh input) ─► tokens/det
                            │        + detector & positional embeddings
                            │   ┌──────────────────────────────────────────┐
   geometry branch ─────────┼──►│  power-weighted coherence |γ|, phase;     │
   (CoherentEncoder)        │   │  GCC arrival-time Δt  →  geometry tokens   │
                            │   └──────────────────────────────────────────┘
                            ▼        injected INTO the transformer
              3-layer pre-norm Transformer (cross-detector fusion)
                            │        attention pooling (learned queries)
   raw strain ─► log-energy branch  (amplitude survives any LayerNorm)
   ASD summary ─► psd_cond branch    (whitened amplitude → physical SNR)
                            ▼
              context (256) ⊕ signal-rank embedding (32)  =  288-D
                            ▼
        Neural Spline Flow (10 layers, 16 bins, tail_bound 5) ─► posterior
```

Three design invariants — each reverses a **measured** failure of the
predecessor (`OverlapNeuralPE`); see §5 — and should not be changed casually:

- **No normalization on the amplitude path.** The strain is whitened (unit
  noise floor), so absolute amplitude *is* the distance/SNR cue. The stem sees
  `asinh`-compressed strain; a parallel branch feeds per-window log-energies of
  the *raw* strain so amplitude survives any downstream LayerNorm.
- **Rank conditioning.** The flow's context includes a 32-D embedding of the
  queried signal's loudness rank (ordered by `Mc^(5/6)/d_L`). This turns the
  ill-posed "predict every overlapping signal" mixture target into a well-posed
  conditional.
- **Pure-NLL objective.** No auxiliary/diversity/width-penalty losses — every
  such term tried here was measured to move representation *statistics* without
  changing *information content*.

**Encoder variants.** `encoder_type="conv"` is the norm-free base
(`LeanStrainEncoder`); `encoder_type="coherent"` (`CoherentEncoder`, the
production choice) adds the geometry branch — power-weighted complex coherence
`γ_ij`, its magnitude/phase, and GCC arrival-time delays — injected as
conditioning tokens *into* the transformer, making the convolutional
representation geometry-aware. This is what lifted distance correlation off the
floor (see §5).

**PSD conditioning** (`psd_cond=True`). Whitened amplitude = distance ×
detector-sensitivity; without knowing the sensitivity the model reads a
quiet/near source and a loud/far source identically (and mis-estimates
distance for less-sensitive O1/O2 events). The `psd_cond` branch conditions on
a per-detector log-ASD-vs-design band summary (`asd_bands`) so the model can
convert whitened amplitude → physical SNR → distance. Design-whitened
(simulated) data passes zeros; real events compute it via
`preprocessing.compute_asd_bands`.

**Parameter handling.** `ParamScaler` is a *fixed, deterministic* invertible
map physical ↔ [−1, 1] (log-space for masses/distance, linear otherwise) — no
fitted statistics that drift between train and eval. Circular params (`ra`,
`phase`, `psi`) wrap exactly at sampling.

### 2.2 The data engine — component storage + on-the-fly remixing

**The biggest lesson in this project: for NPE the dataset matters more than the
architecture.** Two measured failure modes are fixed by construction:

1. **Fixed noise → memorization** (train NLL diverges from val).
2. **One noise draw per signal → wrong posterior widths.**

The generator (`src/ahsd/data/`, fully bilby-based) stores every event as
**separate whitened components** — noise and each signal individually (float16).
`RemixDataset` (`experiments/remix_data.py`) then assembles a *different*
example every epoch:

| Augmentation | Mechanism | Label handling |
|---|---|---|
| Noise swap | any stored Gaussian realization, or a real O3 crop | — |
| Real-noise re-coloring | `irfft(rfft(sig)·ASD_design/ASD_measured)` — exact (whitening is diagonal in frequency) | — |
| Time shift | circular roll ±0.1 s, identical across detectors | `geocent_time += δt` |
| Distance rescale | amplitude × s (exact leading-order GR) | `d_L ÷ s` |
| Overlaps | 2–5 signals summed; per-event loudness re-sort after rescaling | rank labels stay consistent |
| Detector dropout | keep a random detector subset (dropped → unit white noise) | matches the inference-time fill for missing detectors |
| Pre-merger | merger up to ~3 s past the window end (early-warning) | `--premerger` widens the time scaler |

`persistent_workers` stays **False** so `set_epoch()` reaches workers for fresh
pairings each epoch. The network SNR of a whitened example is exactly the L2
norm of its summed whitened signal, so it stays correct under every
augmentation with no stored SNR.

### 2.3 Real detector noise

A model trained purely on Gaussian design-PSD noise does not transfer to real
O3 data (the gap is spectral — measured-PSD shape, non-stationarity, glitches).
The remedy is built into the same loader:

- `scripts/download_gwosc_noise_bank.py` fetches O3 segments per detector from
  GWOSC, whitens each with its **own measured ASD**, and stores strain + ASD
  under `data/noise_bank/`.
- `RemixDataset(real_noise_dir=…, real_noise_prob=p)` draws, per event per
  epoch, real crops with probability *p* — re-coloring the design-whitened
  signals into each segment's whitening exactly, and (for `psd_cond`) emitting
  the matching `asd_bands` sensitivity summary.
- Training reports metrics on **two fixed validation sets each epoch**
  (Gaussian and real-noise); checkpoint selection uses their mean so robustness
  on real data cannot silently trade away simulated-data performance.

---

## 3. Calibration-gated checkpoint selection

Selecting the checkpoint on **best validation NLL alone is unsafe** — and this
is measured, not hypothetical. Maximum-likelihood training of a conditional
flow keeps lowering NLL by *over-sharpening* the posterior long after the model
stops improving: NLL rewards density *at the truth* and is blind to probability
mass piled at the parameter prior boundaries. In this project a run that was
well-calibrated at epoch ~18 (spurious railing ~6%, coverage ~0.90) had, two
epochs later, a **lower** val NLL but ~90% spurious railing and high-SNR
coverage collapsed to ~0.49 — the "best val NLL" criterion actively selects the
overconfident checkpoint. Full study: `analysis/divergence_localization/`.

The trainer therefore reports, **every epoch**, the calibration metrics NLL is
blind to, and gates checkpoint selection on them:

- `spurious_railing` — fraction of posterior samples pinned at a prior boundary
  in a dimension whose truth is not near that boundary (the overconfidence
  signature).
- `base_conc` = E‖z‖²/D — base-space concentration of the truth; **1.0 =
  calibrated**, >1 = overconfident (posteriors too narrow for where the truth
  sits).
- `sci_cov90` / `sci_cov90_highsnr` — mass/distance 90% coverage, all events and
  the high-SNR subset (where overconfidence bites first).
- `sbc_pass_frac` — fraction of parameters passing an SBC uniformity test.

`best_model.pth` is the **lowest-val-NLL epoch that still passes the gate**
(`spurious_railing ≤ --max_spurious_railing`, default 0.10; optional
`--min_highsnr_cov90`). Until the first calibrated epoch appears it falls back
to best-NLL so a run never ends with no checkpoint; once a calibrated epoch
claims `best_model.pth`, a later overconfident epoch cannot reclaim it. The run
also writes `last_model.pth` every epoch and `epoch_XXXX.pth` every
`--ckpt_every` epochs, so a good state is never unrecoverable.

---

## 4. Using it

All commands run in the conda env **`ahsd` (Python 3.10)**, which has an
editable install pointing at `src/`. Prefix with `conda run -n ahsd` (or
activate it) — do **not** run under a bare system Python, where a stale
installed `ahsd` can shadow `src/`.

### Setup

```bash
conda activate ahsd          # torch (MPS/CUDA), bilby, gwpy, nflows
pip install -e .             # editable install into the env
```

### Generate a dataset (component storage)

```bash
conda run -n ahsd python src/ahsd/data/scripts/generate_dataset.py \
    --config configs/data_config.yaml --n-samples 50000 --output-dir data/dataset
```

~2 h for 50k events on a laptop; ~22 GB. `configs/data_config.yaml` controls
event mix, overlap fraction, SNR gate, and pre-merger fraction.

### Download the real-noise bank

```bash
conda run -n ahsd python scripts/download_gwosc_noise_bank.py \
    --per_det 120 --outdir data/noise_bank
```

### Train (production config: coherent encoder + PSD conditioning + real noise)

```bash
conda run -n ahsd python -u experiments/train_lean_npe.py \
    --data data/dataset --remix \
    --epochs 40 --batch 128 --lr 3e-4 \
    --encoder_type coherent --psd_cond --psd_bands 16 \
    --noise_bank data/noise_bank --real_noise_prob 0.9 --det_dropout 0.2 \
    --ckpt_every 2 --max_spurious_railing 0.10 \
    --outdir model/lean_npe
```

The first run builds a memmap cache (`data/dataset/memmap/`) so epochs stream
from disk. Fine-tune from a checkpoint with `--init_from <ckpt>` (fresh
optimizer/schedule).

### Reading the per-epoch log

```
epoch 18 | train 1.32 | val 0.94 | shufΔ +10.97 | dcorr +0.78 | dcov50/90 0.47/0.91 |
          rail 0.058 zc 1.05 covHi 0.88 sbc 0.55 | gnorm 40 | 250s | REAL val 0.02 …
  ✓ calibrated (rail 0.058) -> best_model.pth (select 0.94)
```

| Field | Healthy | Failure signature |
|---|---|---|
| `shufΔ` | climbs several nats | pinned near 0 → marginal fit; diagnose the encoder, don't train longer |
| `dcorr` | lifts off 0 by ~epoch 10–15, → 0.7+ | flat while `shufΔ` grows → amplitude not extracted |
| `rail` (spurious railing) | ≤ ~0.06 | climbing → overconfidence; epoch is **rejected** for `best_model` |
| `zc` (‖z‖²/D) | ≈ 1.0 | > ~1.5 → posteriors too narrow (overconfident) |
| `covHi` (high-SNR coverage) | ≈ 0.90 | dropping → high-SNR under-coverage |
| `val` vs `train` | fall together | val flat/rising while train falls → memorization |
| `REAL …` block | converges toward Gaussian | stays far apart → raise `real_noise_prob` / bank size |

### Certify a checkpoint (SBC / coverage / railing + GWTC real-event smoke tests → HTML)

```bash
conda run -n ahsd python scripts/validate_checkpoint.py --model model/lean_npe/best_model.pth
```

Other benchmarks: `scripts/benchmark_real_events.py` (vs dynesty),
`scripts/overlap_benchmark.py`, `scripts/twin_grid.py` (twin-injection bias).

### Inference CLI (real GWOSC event / local strain / fresh injection)

```bash
conda run -n ahsd python infer.py --event GW150914 --samples 10000 --output results/GW150914
```

### Sample a posterior in code

```python
import torch
from ahsd.models.lean_npe import LeanNPE

ckpt = torch.load("model/lean_npe/best_model.pth", map_location="cpu", weights_only=False)
a = ckpt["args"]
model = LeanNPE(premerger=a.get("premerger", False),
                psd_cond=a.get("psd_cond", False) or False,
                psd_bands=a.get("psd_bands", 16),
                encoder_type=a.get("encoder_type", "conv"))
model.load_state_dict(ckpt["model_state_dict"]); model.eval()

strain = torch.randn(1, 3, 16384)   # whitened H1/L1/V1 @ 4096 Hz, 4 s
post = model.sample_posterior(strain, rank=0, n_samples=3000)   # [1, 3000, 11] physical
# rank=0 = loudest signal; rank=1 = second-loudest, etc.
```

Production inference should go through `ahsd.inference.infer` / `infer.py`,
which add preprocessing, OOD/confidence gating, and importance reweighting.

---

## 5. Why it looks like this: the evidence trail

The architecture and training are the survivors of a measured post-mortem;
details live in `analysis/` and the git history. Validate changes that touch
encoder representation or the data pipeline against these diagnostics, **not
just NLL** — NLL and real-event accuracy have been measured to decouple.

- **Context collapse (the predecessor).** `OverlapNeuralPE` (22.7M params)
  produced a context that was *constant across events* — shuffle ΔNLL exactly
  0, linear-probe R² < 0 for every parameter. Root causes: stacked per-sample
  normalizations erased amplitude, readout attention collapsed to uniform, one
  context was asked to predict all overlapping signals (a mixture target), and
  regularizers dominated the loss. LeanNPE reverses each.
- **The dataset, not the model.** Fixed noise memorizes; one noise draw per
  signal under-teaches width. Component storage + remixing closes both.
- **Geometry-aware encoder.** On a corrected dataset, adding the coherence /
  arrival-time geometry branch (`CoherentEncoder`) lifted distance correlation
  from ~0.04 (flat) to ~0.43 (genuine amplitude → distance) — both the encoder
  branch and the dataset fix were required.
- **PSD conditioning** closes the distance bias where quieter (O1/O2) events
  were read as farther.
- **Overconfidence in late training** (the reason for §3). With a pure-NLL
  objective the flow keeps sharpening past the calibrated point; the collapse
  is domain-independent, worst at high SNR, and invisible to val NLL. It is
  fixed by calibration-gated selection, not by more data — controlling for SNR,
  local training density has ~zero effect on the per-event bias
  (`analysis/divergence_localization/`).

---

## 6. Project structure

```
src/ahsd/
├── models/
│   ├── lean_npe.py            # LeanNPE: encoder + rank embedding + NSF (the model) + ParamScaler
│   ├── coherent_encoder.py    # CoherentEncoder: geometry branch (coherence, GCC Δt) → tokens
│   ├── flows.py               # NSFPosteriorFlow (nflows-based)
│   └── parameter_scalers.py   # FLOW_NORM_BOUND and legacy scalers
├── inference/                 # pipeline.infer(), preprocessing, OOD gating, dynesty bridge
├── core/                      # PriorityNet, adaptive subtractor, end-to-end overlap pipeline
├── data/                      # bilby-based generation: sampler, injector, whitener, PSDs, snr_utils
│   └── scripts/generate_dataset.py
└── evaluation/                # metrics

experiments/
├── train_lean_npe.py          # trainer: remix, real-noise, calibration-gated selection, dual validation
├── remix_data.py              # memmap cache + RemixDataset (all augmentations, both noise sources)
└── train_priority_net.py      # PriorityNet training

scripts/
├── validate_checkpoint.py     # CI: SBC / coverage / railing + 6 GWTC real-event smoke tests → HTML
├── benchmark_real_events.py   # real-event benchmark vs bilby/dynesty
├── overlap_benchmark.py       # overlapping-signal benchmark
├── twin_grid.py               # twin-injection bias grid
├── download_gwosc_noise_bank.py
└── dynesty_compare.py

infer.py       # inference CLI (real GWOSC event / local strain / fresh injection)
analysis/      # measured evidence: diagnostic reports, audits, comparison outputs (the paper's evidence)
configs/       # YAML configs (data_config.yaml is the main one)
data/          # dataset (component storage), memmap caches, noise_bank  (git-ignored)
model/         # checkpoints (best_model.pth bundles weights + args + diagnostics)
```

> Note: the model's flow size (`flow_layers`, `flow_hidden`, `flow_bins`) is a
> hardcoded `LeanNPE` default, not stored in `args`. A checkpoint loads
> `strict=True` only if the current defaults match the generation it was trained
> with; reconcile a size mismatch against the checkpoint before assuming a bug.
> Note: `./build/` is a git-ignored setuptools artifact — never on `sys.path`.

## 7. Roadmap

1. Re-run the production config under calibration-gated selection; benchmark the
   selected checkpoint on GWTC events vs dynesty.
2. Investigate the residual near-equal-mass (`m1 ≈ m2`) bias — a mass-pair
   reparameterization (`(chirp_mass, mass_ratio)`) is the smallest candidate.
3. Encoder temporal resolution — phase-preserving features to tighten timing
   and sky localization.
4. Pre-merger (early-warning) training run (`--premerger`).
5. End-to-end overlap pipeline on real-noise overlaps.

## References

- Dax et al., *DINGO — Real-Time GW Science with NPE*: [arXiv:2106.12594](https://arxiv.org/abs/2106.12594)
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
# PosteriFlow — Neural Posterior Estimation for Overlapping Gravitational-Wave Signals

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Amortized, calibration-audited posterior estimation for compact-binary
> coalescences in whitened 3-detector strain — seconds per event instead of
> minutes-to-hours, with per-signal posteriors for time-overlapping sources.

---

## 1. The problem

Ground-based detectors (LIGO Hanford/Livingston, Virgo) observe compact-binary
mergers as chirping strain buried in colored noise. Parameter estimation —
recovering masses, distance, sky position, spins — is classically done with
stochastic samplers (dynesty, bilby-MCMC) that evaluate a waveform likelihood
hundreds of thousands of times per event. That costs minutes to hours per
event and, more fundamentally, assumes **one signal per analysis segment**. As
sensitivity improves (O4/O5, Einstein Telescope), the in-band rate rises until
**signals overlap in time**, where the single-signal likelihood is simply
wrong.

**PosteriFlow's answer** is simulation-based inference: train a neural network
on millions of simulated `(signal + noise → parameters)` examples so that, at
observation time, a *single forward pass* returns the posterior. No explicit
likelihood; overlap handling is built into the training distribution rather
than bolted onto a sampler.

The estimator, **LeanNPE**, answers:

> *"Given this 4 s of whitened H1/L1/V1 strain, what is the posterior over the
> 11 parameters of the k-th loudest signal present?"*

Parameters (fixed order): `mass_1, mass_2, luminosity_distance, ra, dec,
theta_jn, psi, phase, geocent_time, a1, a2` (aligned spins; `mass_1 ≥ mass_2`
convention). Overlaps are handled by **rank conditioning** — query rank 0 for
the loudest signal, rank 1 for the second-loudest, etc.

---

## 2. How it works

### 2.1 The model (`src/ahsd/models/lean_npe.py`, ~6M parameters)

```
whitened strain [3, 16384] ─┬─► norm-free conv stem (asinh input) ─► tokens/det
                            │        + detector & positional embeddings
                            │   ┌──────────────────────────────────────────┐
   geometry branch ─────────┼──►│  power-weighted coherence |γ|, phase;     │
   (CoherentEncoder)        │   │  GCC arrival-time Δt  →  geometry tokens   │
                            │   └──────────────────────────────────────────┘
                            ▼        injected INTO the transformer
              3-layer pre-norm Transformer (cross-detector fusion)
                            │        attention pooling (learned queries)
   raw strain ─► log-energy branch  (amplitude survives any LayerNorm)
   ASD summary ─► psd_cond branch    (whitened amplitude → physical SNR)
                            ▼
              context (256) ⊕ signal-rank embedding (32)  =  288-D
                            ▼
        Neural Spline Flow (10 layers, 16 bins, tail_bound 5) ─► posterior
```

Three design invariants — each reverses a **measured** failure of the
predecessor (`OverlapNeuralPE`); see §5 — and should not be changed casually:

- **No normalization on the amplitude path.** The strain is whitened (unit
  noise floor), so absolute amplitude *is* the distance/SNR cue. The stem sees
  `asinh`-compressed strain; a parallel branch feeds per-window log-energies of
  the *raw* strain so amplitude survives any downstream LayerNorm.
- **Rank conditioning.** The flow's context includes a 32-D embedding of the
  queried signal's loudness rank (ordered by `Mc^(5/6)/d_L`). This turns the
  ill-posed "predict every overlapping signal" mixture target into a well-posed
  conditional.
- **Pure-NLL objective.** No auxiliary/diversity/width-penalty losses — every
  such term tried here was measured to move representation *statistics* without
  changing *information content*.

**Encoder variants.** `encoder_type="conv"` is the norm-free base
(`LeanStrainEncoder`); `encoder_type="coherent"` (`CoherentEncoder`, the
production choice) adds the geometry branch — power-weighted complex coherence
`γ_ij`, its magnitude/phase, and GCC arrival-time delays — injected as
conditioning tokens *into* the transformer, making the convolutional
representation geometry-aware. This is what lifted distance correlation off the
floor (see §5).

**PSD conditioning** (`psd_cond=True`). Whitened amplitude = distance ×
detector-sensitivity; without knowing the sensitivity the model reads a
quiet/near source and a loud/far source identically (and mis-estimates
distance for less-sensitive O1/O2 events). The `psd_cond` branch conditions on
a per-detector log-ASD-vs-design band summary (`asd_bands`) so the model can
convert whitened amplitude → physical SNR → distance. Design-whitened
(simulated) data passes zeros; real events compute it via
`preprocessing.compute_asd_bands`.

**Parameter handling.** `ParamScaler` is a *fixed, deterministic* invertible
map physical ↔ [−1, 1] (log-space for masses/distance, linear otherwise) — no
fitted statistics that drift between train and eval. Circular params (`ra`,
`phase`, `psi`) wrap exactly at sampling.

### 2.2 The data engine — component storage + on-the-fly remixing

**The biggest lesson in this project: for NPE the dataset matters more than the
architecture.** Two measured failure modes are fixed by construction:

1. **Fixed noise → memorization** (train NLL diverges from val).
2. **One noise draw per signal → wrong posterior widths.**

The generator (`src/ahsd/data/`, fully bilby-based) stores every event as
**separate whitened components** — noise and each signal individually (float16).
`RemixDataset` (`experiments/remix_data.py`) then assembles a *different*
example every epoch:

| Augmentation | Mechanism | Label handling |
|---|---|---|
| Noise swap | any stored Gaussian realization, or a real O3 crop | — |
| Real-noise re-coloring | `irfft(rfft(sig)·ASD_design/ASD_measured)` — exact (whitening is diagonal in frequency) | — |
| Time shift | circular roll ±0.1 s, identical across detectors | `geocent_time += δt` |
| Distance rescale | amplitude × s (exact leading-order GR) | `d_L ÷ s` |
| Overlaps | 2–5 signals summed; per-event loudness re-sort after rescaling | rank labels stay consistent |
| Detector dropout | keep a random detector subset (dropped → unit white noise) | matches the inference-time fill for missing detectors |
| Pre-merger | merger up to ~3 s past the window end (early-warning) | `--premerger` widens the time scaler |

`persistent_workers` stays **False** so `set_epoch()` reaches workers for fresh
pairings each epoch. The network SNR of a whitened example is exactly the L2
norm of its summed whitened signal, so it stays correct under every
augmentation with no stored SNR.

### 2.3 Real detector noise

A model trained purely on Gaussian design-PSD noise does not transfer to real
O3 data (the gap is spectral — measured-PSD shape, non-stationarity, glitches).
The remedy is built into the same loader:

- `scripts/download_gwosc_noise_bank.py` fetches O3 segments per detector from
  GWOSC, whitens each with its **own measured ASD**, and stores strain + ASD
  under `data/noise_bank/`.
- `RemixDataset(real_noise_dir=…, real_noise_prob=p)` draws, per event per
  epoch, real crops with probability *p* — re-coloring the design-whitened
  signals into each segment's whitening exactly, and (for `psd_cond`) emitting
  the matching `asd_bands` sensitivity summary.
- Training reports metrics on **two fixed validation sets each epoch**
  (Gaussian and real-noise); checkpoint selection uses their mean so robustness
  on real data cannot silently trade away simulated-data performance.

---

## 3. Calibration-gated checkpoint selection

Selecting the checkpoint on **best validation NLL alone is unsafe** — and this
is measured, not hypothetical. Maximum-likelihood training of a conditional
flow keeps lowering NLL by *over-sharpening* the posterior long after the model
stops improving: NLL rewards density *at the truth* and is blind to probability
mass piled at the parameter prior boundaries. In this project a run that was
well-calibrated at epoch ~18 (spurious railing ~6%, coverage ~0.90) had, two
epochs later, a **lower** val NLL but ~90% spurious railing and high-SNR
coverage collapsed to ~0.49 — the "best val NLL" criterion actively selects the
overconfident checkpoint. Full study: `analysis/divergence_localization/`.

The trainer therefore reports, **every epoch**, the calibration metrics NLL is
blind to, and gates checkpoint selection on them:

- `spurious_railing` — fraction of posterior samples pinned at a prior boundary
  in a dimension whose truth is not near that boundary (the overconfidence
  signature).
- `base_conc` = E‖z‖²/D — base-space concentration of the truth; **1.0 =
  calibrated**, >1 = overconfident (posteriors too narrow for where the truth
  sits).
- `sci_cov90` / `sci_cov90_highsnr` — mass/distance 90% coverage, all events and
  the high-SNR subset (where overconfidence bites first).
- `sbc_pass_frac` — fraction of parameters passing an SBC uniformity test.

`best_model.pth` is the **lowest-val-NLL epoch that still passes the gate**
(`spurious_railing ≤ --max_spurious_railing`, default 0.10; optional
`--min_highsnr_cov90`). Until the first calibrated epoch appears it falls back
to best-NLL so a run never ends with no checkpoint; once a calibrated epoch
claims `best_model.pth`, a later overconfident epoch cannot reclaim it. The run
also writes `last_model.pth` every epoch and `epoch_XXXX.pth` every
`--ckpt_every` epochs, so a good state is never unrecoverable.

---

## 4. Using it

All commands run in the conda env **`ahsd` (Python 3.10)**, which has an
editable install pointing at `src/`. Prefix with `conda run -n ahsd` (or
activate it) — do **not** run under a bare system Python, where a stale
installed `ahsd` can shadow `src/`.

### Setup

```bash
conda activate ahsd          # torch (MPS/CUDA), bilby, gwpy, nflows
pip install -e .             # editable install into the env
```

### Generate a dataset (component storage)

```bash
conda run -n ahsd python src/ahsd/data/scripts/generate_dataset.py \
    --config configs/data_config.yaml --n-samples 50000 --output-dir data/dataset
```

~2 h for 50k events on a laptop; ~22 GB. `configs/data_config.yaml` controls
event mix, overlap fraction, SNR gate, and pre-merger fraction.

### Download the real-noise bank

```bash
conda run -n ahsd python scripts/download_gwosc_noise_bank.py \
    --per_det 120 --outdir data/noise_bank
```

### Train (production config: coherent encoder + PSD conditioning + real noise)

```bash
conda run -n ahsd python -u experiments/train_lean_npe.py \
    --data data/dataset --remix \
    --epochs 40 --batch 128 --lr 3e-4 \
    --encoder_type coherent --psd_cond --psd_bands 16 \
    --noise_bank data/noise_bank --real_noise_prob 0.9 --det_dropout 0.2 \
    --ckpt_every 2 --max_spurious_railing 0.10 \
    --outdir model/lean_npe
```

The first run builds a memmap cache (`data/dataset/memmap/`) so epochs stream
from disk. Fine-tune from a checkpoint with `--init_from <ckpt>` (fresh
optimizer/schedule).

### Reading the per-epoch log

```
epoch 18 | train 1.32 | val 0.94 | shufΔ +10.97 | dcorr +0.78 | dcov50/90 0.47/0.91 |
          rail 0.058 zc 1.05 covHi 0.88 sbc 0.55 | gnorm 40 | 250s | REAL val 0.02 …
  ✓ calibrated (rail 0.058) -> best_model.pth (select 0.94)
```

| Field | Healthy | Failure signature |
|---|---|---|
| `shufΔ` | climbs several nats | pinned near 0 → marginal fit; diagnose the encoder, don't train longer |
| `dcorr` | lifts off 0 by ~epoch 10–15, → 0.7+ | flat while `shufΔ` grows → amplitude not extracted |
| `rail` (spurious railing) | ≤ ~0.06 | climbing → overconfidence; epoch is **rejected** for `best_model` |
| `zc` (‖z‖²/D) | ≈ 1.0 | > ~1.5 → posteriors too narrow (overconfident) |
| `covHi` (high-SNR coverage) | ≈ 0.90 | dropping → high-SNR under-coverage |
| `val` vs `train` | fall together | val flat/rising while train falls → memorization |
| `REAL …` block | converges toward Gaussian | stays far apart → raise `real_noise_prob` / bank size |

### Certify a checkpoint (SBC / coverage / railing + GWTC real-event smoke tests → HTML)

```bash
conda run -n ahsd python scripts/validate_checkpoint.py --model model/lean_npe/best_model.pth
```

Other benchmarks: `scripts/benchmark_real_events.py` (vs dynesty),
`scripts/overlap_benchmark.py`, `scripts/twin_grid.py` (twin-injection bias).

### Inference CLI (real GWOSC event / local strain / fresh injection)

```bash
conda run -n ahsd python infer.py --event GW150914 --samples 10000 --output results/GW150914
```

### Sample a posterior in code

```python
import torch
from ahsd.models.lean_npe import LeanNPE

ckpt = torch.load("model/lean_npe/best_model.pth", map_location="cpu", weights_only=False)
a = ckpt["args"]
model = LeanNPE(premerger=a.get("premerger", False),
                psd_cond=a.get("psd_cond", False) or False,
                psd_bands=a.get("psd_bands", 16),
                encoder_type=a.get("encoder_type", "conv"))
model.load_state_dict(ckpt["model_state_dict"]); model.eval()

strain = torch.randn(1, 3, 16384)   # whitened H1/L1/V1 @ 4096 Hz, 4 s
post = model.sample_posterior(strain, rank=0, n_samples=3000)   # [1, 3000, 11] physical
# rank=0 = loudest signal; rank=1 = second-loudest, etc.
```

Production inference should go through `ahsd.inference.infer` / `infer.py`,
which add preprocessing, OOD/confidence gating, and importance reweighting.

---

## 5. Why it looks like this: the evidence trail

The architecture and training are the survivors of a measured post-mortem;
details live in `analysis/` and the git history. Validate changes that touch
encoder representation or the data pipeline against these diagnostics, **not
just NLL** — NLL and real-event accuracy have been measured to decouple.

- **Context collapse (the predecessor).** `OverlapNeuralPE` (22.7M params)
  produced a context that was *constant across events* — shuffle ΔNLL exactly
  0, linear-probe R² < 0 for every parameter. Root causes: stacked per-sample
  normalizations erased amplitude, readout attention collapsed to uniform, one
  context was asked to predict all overlapping signals (a mixture target), and
  regularizers dominated the loss. LeanNPE reverses each.
- **The dataset, not the model.** Fixed noise memorizes; one noise draw per
  signal under-teaches width. Component storage + remixing closes both.
- **Geometry-aware encoder.** On a corrected dataset, adding the coherence /
  arrival-time geometry branch (`CoherentEncoder`) lifted distance correlation
  from ~0.04 (flat) to ~0.43 (genuine amplitude → distance) — both the encoder
  branch and the dataset fix were required.
- **PSD conditioning** closes the distance bias where quieter (O1/O2) events
  were read as farther.
- **Overconfidence in late training** (the reason for §3). With a pure-NLL
  objective the flow keeps sharpening past the calibrated point; the collapse
  is domain-independent, worst at high SNR, and invisible to val NLL. It is
  fixed by calibration-gated selection, not by more data — controlling for SNR,
  local training density has ~zero effect on the per-event bias
  (`analysis/divergence_localization/`).

---

## 6. Project structure

```
src/ahsd/
├── models/
│   ├── lean_npe.py            # LeanNPE: encoder + rank embedding + NSF (the model) + ParamScaler
│   ├── coherent_encoder.py    # CoherentEncoder: geometry branch (coherence, GCC Δt) → tokens
│   ├── flows.py               # NSFPosteriorFlow (nflows-based)
│   └── parameter_scalers.py   # FLOW_NORM_BOUND and legacy scalers
├── inference/                 # pipeline.infer(), preprocessing, OOD gating, dynesty bridge
├── core/                      # PriorityNet, adaptive subtractor, end-to-end overlap pipeline
├── data/                      # bilby-based generation: sampler, injector, whitener, PSDs, snr_utils
│   └── scripts/generate_dataset.py
└── evaluation/                # metrics

experiments/
├── train_lean_npe.py          # trainer: remix, real-noise, calibration-gated selection, dual validation
├── remix_data.py              # memmap cache + RemixDataset (all augmentations, both noise sources)
└── train_priority_net.py      # PriorityNet training

scripts/
├── validate_checkpoint.py     # CI: SBC / coverage / railing + 6 GWTC real-event smoke tests → HTML
├── benchmark_real_events.py   # real-event benchmark vs bilby/dynesty
├── overlap_benchmark.py       # overlapping-signal benchmark
├── twin_grid.py               # twin-injection bias grid
├── download_gwosc_noise_bank.py
└── dynesty_compare.py

infer.py       # inference CLI (real GWOSC event / local strain / fresh injection)
analysis/      # measured evidence: diagnostic reports, audits, comparison outputs (the paper's evidence)
configs/       # YAML configs (data_config.yaml is the main one)
data/          # dataset (component storage), memmap caches, noise_bank  (git-ignored)
model/         # checkpoints (best_model.pth bundles weights + args + diagnostics)
```

> Note: the model's flow size (`flow_layers`, `flow_hidden`, `flow_bins`) is a
> hardcoded `LeanNPE` default, not stored in `args`. A checkpoint loads
> `strict=True` only if the current defaults match the generation it was trained
> with; reconcile a size mismatch against the checkpoint before assuming a bug.
> Note: `./build/` is a git-ignored setuptools artifact — never on `sys.path`.

## 7. Roadmap

1. Re-run the production config under calibration-gated selection; benchmark the
   selected checkpoint on GWTC events vs dynesty.
2. Investigate the residual near-equal-mass (`m1 ≈ m2`) bias — a mass-pair
   reparameterization (`(chirp_mass, mass_ratio)`) is the smallest candidate.
3. Encoder temporal resolution — phase-preserving features to tighten timing
   and sky localization.
4. Pre-merger (early-warning) training run (`--premerger`).
5. End-to-end overlap pipeline on real-noise overlaps.

## References

- Dax et al., *DINGO — Real-Time GW Science with NPE*: [arXiv:2106.12594](https://arxiv.org/abs/2106.12594)
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
