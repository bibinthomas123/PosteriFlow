# Dataset generation

How LeanNPE training data is made, checked and consumed (current as of
2026-10-31). Older docs described a d²-prior, HDF5-based setup that no
longer exists; this replaces them.

## What one event is

Each event is 4 s at 4096 Hz in H1, L1 and V1, generated with bilby
(IMRPhenomXP for BBH, IMRPhenomNSBH, IMRPhenomD_NRTidalv2 for BNS) and
whitened with the design PSD. Every event is stored as **separate
components** (float16): the whitened noise and each whitened signal. Because
whitening is linear, `RemixDataset` (`experiments/remix_data.py`) can
re-assemble a different example every epoch — new noise (Gaussian or a real
O1–O4 segment, re-coloured), time shifts, distance rescaling (amplitude × s,
d_L ÷ s), and overlap sums — so a fixed dataset behaves like an endless one.

About 30% of events contain 2–4 time-overlapping signals; ~5% are noise
only. Events whose loudest signal has network SNR < `min_snr` (6) are
redrawn, so roughly 40% of parameter draws are rejected.

## Configs

| Config | Use |
|---|---|
| `configs/data_config_fixed.yaml` | **default**: BBH/NSBH/BNS mixture, uniform-in-distance prior |
| `configs/data_config_{bbh,nsbh,bns}_v3.yaml` | single-class variants (same settings, one source class) |
| `configs/data_config.yaml` | legacy reference of every option (d² prior, narrower ranges); not used for training |

Key settings in `data_config_fixed.yaml`:

- `distance_prior: uniform` — flat in d within each class's range. A d² prior
  plus the SNR cut made heavy sources systematically farther (a chirp-mass →
  distance shortcut the model learned). Convert posteriors to the LVC d²
  prior with `PosteriorResult.reweight_distance_prior()`.
- `distance_ranges` — BBH 100–4000, NSBH 20–1200, BNS 10–400 Mpc. The
  model's distance support must cover these divided by the remix rescale
  factor (0.75–1.33): train with `--dist_range 8 6000` (mixed) or the
  per-class ranges, otherwise labels are clamped (the trainer's scaler
  audit warns).

## Generating

```bash
# single process (~20 events/s on a laptop core)
python src/ahsd/data/scripts/generate_dataset.py \
    --config configs/data_config_fixed.yaml --n-samples 50000 \
    --output-dir data/dataset_v3 --random-seed 2026

# N shards in parallel, merged into one dataset (~250-430 events/s on 32 cores)
python src/ahsd/data/scripts/generate_parallel.py \
    --config configs/data_config_fixed.yaml --n-samples 200000 \
    --output-dir data/dataset_v3 --shards 32 --seed 2026 --validate

# on the cluster (builds the memmap cache afterwards)
sbatch --export=ALL,N=200000,OUT=data/dataset_v3 scripts/slurm/generate_parallel.sbatch
```

The same seed reproduces a dataset bit for bit (parameters and noise).
Shards use seeds spaced by 1,000,003, and every detector's noise seed is an
independent 63-bit draw, so shards never share events or noise.

### Output layout

```
data/dataset_v3/
  batches/batch_XXXXX.pkl       100 events each (components + parameters)
  train/ validation/ test/      RELATIVE symlinks into batches/ (80/10/10)
  dataset_config.yaml           resolved config (provenance; the trainer
                                records its distance prior and source classes)
  generation_manifest.json      parallel runs: seeds, shard sizes, splits
  logs/                         per-shard generation logs
  memmap/{train,validation}/    float16 cache built by the trainer on first use
```

## Checking — always audit before training

```bash
python scripts/audit_dataset.py data/dataset_v3 --dist-range 8 6000
```

Reads every event of the memmap cache and fails (non-zero exit) on: waveform
failures in the logs, broken event index, non-finite or out-of-range
parameters, `m1 < m2`, distances outside the configured ranges or the model
support, source classes present/absent contrary to the config, NaN/inf or
all-zero waveforms (= failed generation), float16 saturation, and exact
duplicate parameter rows or noise segments. It also reports the overlap
fraction, source mix, distance shape per class, SNR distribution and
whitened-noise level. `scripts/slurm/audit_dataset.sbatch` runs it as a
SLURM gate (`--dependency=afterok:` training jobs only start on clean data).

## Bugs this pipeline has had (and their checks)

| Bug | Effect | Fixed / caught by |
|---|---|---|
| overlap slots could draw "noise" → fixed placeholder (30, 30, face-on, zero spin) injected as a real BBH | 70% of heavy equal-mass training signals were one event; caused the GW150914/GW170814 (30, 30) attractor | `ParameterSampler.sample_signal_parameters()`; audit duplicate check |
| ParamScaler distance support 40–2200 Mpc | BNS < 40 Mpc and d > 2200 Mpc clamped | `--dist_range`; trainer scaler audit |
| cluster lalsuite 7.26 / lalsimulation 6.2.0 | every BNS/NSBH waveform failed → zeros | pin lalsuite 7.26.15 (pip), bilby 2.8.0; audit zero-waveform check |
| noise seed never reached bilby's RNG | `--random-seed` reproduced parameters, not noise | seed `bilby.core.utils.random` |
| per-event interferometer rebuilt from disk | ~5× slower generation | cached interferometer templates |

## Training-size and per-class variants

```bash
# 50k ablation sharing the parent's validation set (links only)
python scripts/make_dataset_subset.py data/dataset_v3 data/dataset_v3_50k --train-batches 400
```

Per-class datasets come from the `_v3` per-class configs; models trained on
them are selected at inference with `infer.py --event-type {BBH,NSBH,BNS}`.
