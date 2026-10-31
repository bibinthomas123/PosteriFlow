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

## 2. Current status (read this before deploying)

| | |
|---|---|
| **Current models** | `model/lean_npe_v3` (mixed BBH/NSBH/BNS, 200k clean events) and per-class `model/lean_npe_{bbh,nsbh,bns}_v3`, selected with `infer.py --event-type` |
| **Passes** (`lean_npe_v3`) | population calibration/coverage, SBC 10/11, spurious railing 0.7%, shuffle ΔNLL 19.9, synthetic basin probe; GW150914 detector-frame m1/m2 42.6/35.1 vs 42.4/36.5 (railing 4%, was 74%) |
| **Known limitations** | the mixed model confuses source classes (biases per-class distance; GW170608 lands in an NSBH mode) — the motivation for per-class models; real-event distances ~1.2–1.5× high under the LVC d² prior; chirp mass ~150× wider than dynesty |
| **2026-10-31 reset** | every earlier model (incl. `lean_npe_mf_scratch`) was trained on data with a placeholder-event bug that produced the GW150914/GW170814 (30, 30) attractor; those checkpoints were deleted. §5's evidence trail predates the fix — read it as the record of that bug's effects |

If you just want a working checkpoint: use `lean_npe_v3` (or the class model
for a known source class) and trust the OOD gate for anything it flags LOW
confidence. If you're extending
the research, start at §5 — the evidence trail explains *why* each design
choice exists and which recent ones didn't pan out, so you don't repeat a
tested dead end.

---

## 3. How it works

### 3.1 The model (`src/ahsd/models/lean_npe.py`, ~6M parameters)

```
whitened strain [3, 16384] ─┬─► norm-free conv stem (asinh input) ─► tokens/det
                            │        + detector & positional embeddings
                            │   ┌──────────────────────────────────────────┐
   geometry branch ─────────┼──►│  power-weighted coherence |γ|, phase;     │
   (CoherentEncoder)        │   │  GCC arrival-time Δt  →  4 geometry tokens│
                            │   ├──────────────────────────────────────────┤
   MF-token branch ─────────┼──►│  per-detector complex overlaps vs a fixed │
   (--mf_tokens)            │   │  (Mc,q) template grid → 4 MF tokens       │
                            │   └──────────────────────────────────────────┘
                            │        injected INTO the transformer —
                            │        additively ("add") or as 4 independent
                            │        tokens alongside geometry ("concat")
                            ▼
              3-layer pre-norm Transformer (cross-detector + cross-branch fusion)
                            │        attention pooling (learned queries)
   raw strain ─► log-energy branch  (amplitude survives any LayerNorm)
   ASD summary ─► psd_cond branch    (whitened amplitude → physical SNR)
                            ▼
              context (256)
                            ├─(optional, --q_embed_dim)─► q_embed_head ─► q-embedding (8-16)
                            ▼
   context ⊕ [q-embedding] ⊕ signal-rank embedding (32)  =  flow context_features
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
- **Pure-NLL objective.** No auxiliary/diversity/width-penalty losses shape the
  *main* loss — every such term tried here was measured to move representation
  *statistics* without changing *information content*. (Two narrow,
  well-isolated exceptions exist and are additive, not part of the main NLL:
  `--aux_mass`, a training-only regularizer on the encoder context, and
  `--aux_q_embed`, which supervises the dedicated q-embedding described
  below — both are auxiliary SmoothL1 terms added to the loss, not
  replacements for it.)

**Encoder variants.** `encoder_type="conv"` is the norm-free base
(`LeanStrainEncoder`); `encoder_type="coherent"` (`CoherentEncoder`, the
production choice) adds the geometry branch — power-weighted complex coherence
`γ_ij`, its magnitude/phase, and GCC arrival-time delays — injected as
conditioning tokens *into* the transformer, making the convolutional
representation geometry-aware. This is what lifted distance correlation off the
floor (see §5).

**MF-token branch** (`--mf_tokens`, `docs/mf_token_encoder_design.md`). Adds a
second conditioning branch: per-detector complex overlaps against a fixed
120-template `(Mc, q)` grid (`assets/mf_bank.npz`, built once by
`build_mf_bank.py`), extracted via an alias-free band-limited IFFT (envelope
fidelity at ~1 ms sampling, no noise folding). This is the chirp-phase mass
statistic the conv stem cannot extract at low SNR and that geometry features
provably lack (in-cell mass-ratio AUC: MF 0.989 vs. geometry 0.532 —
`analysis/attractor_basin/mf_probe.py`). Two fusion modes, set by
`--mf_token_fusion`:

- `"add"` (default, currently deployed): MF tokens are gated
  (`mf_gate`, zero-init) and summed into the same 4 geometry-token slots.
  **Measured to be the single largest information-loss stage in the whole
  pipeline** — bigger than the MF branch's own 1440→128 embedding
  compression — because a near-uninformative geometry signal (in-cell AUC
  0.532) gets summed into a clean, informative one (0.883), corrupting it
  (`analysis/attractor_basin/mf_embedding_tap_v2.py`).
- `"concat"`: MF tokens become 4 *independent* tokens appended alongside (not
  summed into) the geometry tokens — 8 extra tokens instead of 4, plus a
  zero-init modality/type embedding so the transformer knows which stream
  each token came from. Demonstrably preserves more information at every
  stage (final-context mass-ratio AUC 0.889 vs. 0.848) — **but this improved
  representation did not translate into a better posterior** in the one
  controlled experiment run so far (see §5.3). Not yet the default; the
  finding is real and the mechanism is understood, but swapping the default
  requires resolving why better information didn't help before recommending
  it for production.

**Dedicated q-conditioning embedding** (`--q_embed_dim`, `--aux_q_embed`,
2026-07-22, active research). A direct test of *why* `"concat"`'s better
representation didn't produce a better posterior found that the flow's
per-layer context conditioning is **equally (in)sensitive to the
mass-ratio-informative direction in context space as to a random direction of
the same size** — the information is present and linearly decodable, but the
flow isn't preferentially routing through it
(`analysis/attractor_basin/flow_context_sensitivity.py`). `q_embed_dim=N`
adds a small MLP (`q_embed_head`) producing an N-D embedding that is
concatenated directly onto the **flow's** context input (not just an
encoder-side auxiliary loss), supervised by its own regression head
(`q_pred_head`) via `--aux_q_embed`'s weight — handing the flow a small,
unambiguous channel instead of asking it to discover the subspace amid 256
competing dimensions. `q_embed_dim=0` (default) is byte-identical to prior
behavior. **Result pending** — training in progress as of this writing.

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
`phase`, `psi`) wrap exactly at sampling. `mass_param="mc_q"` (chirp
mass/mass-ratio) is the production choice — it structurally forbids
`mass_1 < mass_2` ordering violations and softens the near-equal-mass
attractor discussed in §5.2, vs. the legacy `"m1m2"` coordinates.

> **Checkpoint-compatibility gotcha:** `mf_token_fusion` and `q_embed_dim` are
> both real architectural switches. `mf_token_fusion` doesn't change any
> tensor *shapes* (same-sized `mf_to_tokens` output, just combined
> differently) — an old load site that doesn't thread it through will
> **silently reconstruct the wrong forward pass** rather than crash, so treat
> a missing kwarg here as seriously as a shape mismatch. `q_embed_dim` DOES
> change tensor shapes (the flow's `context_features` grows), so a missing
> kwarg there is a hard `strict=True` crash. Every load site in this repo
> threads both through (`ckpt["args"].get("mf_token_fusion", "add")`,
> `ckpt["args"].get("q_embed_dim", 0)`) — do the same in any new script that
> loads a checkpoint. See the full list under §6.

### 3.2 The data engine — component storage + on-the-fly remixing

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
| Delay/sky augmentation (`--delay_aug`) | remap sky position to a scored candidate from a grid (`--delay_aug_candidates`), varying inter-detector delay honestly | geometry labels updated consistently |
| Basin oversampling (`--basin_oversample`) | extra loss weight / repeats on a known hard geometry cell | — |
| Pre-merger | merger up to ~3 s past the window end (early-warning) | `--premerger` widens the time scaler |

`persistent_workers` stays **False** so `set_epoch()` reaches workers for fresh
pairings each epoch. The network SNR of a whitened example is exactly the L2
norm of its summed whitened signal, so it stays correct under every
augmentation with no stored SNR.

> **Why `--delay_aug` exists:** without it, every stored event has exactly one
> inter-detector delay (rolls are common-mode), so the geometry branch sees a
> single τ per event and can memorize degenerate geometry→parameter mappings
> in thin τ-cells — the root enabler of the attractor-basin failure mode in
> §5.2. `--delay_aug` breaks that by varying delay honestly across training.

### 3.3 Real detector noise

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
- Noise-bank diversity is finite: a specific real segment can be a genuine
  severity **outlier** relative to the bank (measured z≈-11 in matched-filter
  SNR suppression for GW150914's own segment,
  `analysis/attractor_basin/noise_bank_severity_scan.py`) — no amount of
  consistency training over the *existing* bank teaches robustness to a
  fluctuation that far outside the training distribution. Growing bank
  diversity (more GWOSC segments across O1–O4) is the general remedy; see §7.

---

## 4. Calibration-gated checkpoint selection

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

> **Gotcha (non-standard training populations):** the gate's val-NLL
> comparison is only meaningful when the validation set is drawn from the
> SAME distribution the model is trained on. A narrow/non-amortized training
> run evaluated against the repo's default (broad-population) validation
> split will select an essentially undertrained early epoch as
> `best_model.pth`, because val-NLL is dominated by distribution mismatch,
> not fit quality — use `last_model.pth` (or a matched validation set) for
> any narrow-population experiment. Discovered running the non-amortized
> control in `analysis/attractor_basin/gw150914_narrow_dataset.py`.

---

## 5. Why it looks like this: the evidence trail

The architecture and training are the survivors of a measured post-mortem;
details live in `analysis/` and the git history. Validate changes that touch
encoder representation or the data pipeline against these diagnostics, **not
just NLL** — NLL and real-event accuracy have been measured to decouple.

### 5.1 Foundational fixes

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
- **Overconfidence in late training** (the reason for §4). With a pure-NLL
  objective the flow keeps sharpening past the calibrated point; the collapse
  is domain-independent, worst at high SNR, and invisible to val NLL. It is
  fixed by calibration-gated selection, not by more data — controlling for SNR,
  local training density has ~zero effect on the per-event bias
  (`analysis/divergence_localization/`).

### 5.2 The attractor-basin investigation (2026-07 — the project's largest single study)

Full detail: `analysis/attractor_basin/REPORT.md` (the most thorough
single evidence trail in the repo). Condensed timeline:

1. **Discovery.** GW150914 and GW170814 both collapsed to a near-(30,30),
   mass-ratio-≈1 posterior with 60-70%+ spurious railing, despite passing
   every population-level metric. Root-caused via token-grafting to the
   coherent encoder's **geometry tokens**: converged models map the whole
   `{H1–L1 delay ≈ +6..+9 ms} × {SNR ≳ 20} × {heavy}` cell to a near-constant
   context, so the flow returns the cell average regardless of true mass
   ratio. Root enabler: `RemixDataset` never varied inter-detector delay
   before `--delay_aug` existed, so the geometry branch could memorize a
   degenerate mapping in a thin, under-populated cell.
2. **Five-way elimination of encoder/loss-side fixes** (flow reparameterization,
   geometry-token dropout, phase-slope delay features, phase-residual
   features, cell oversampling) — each measured to help partially or not at
   all; the quiet/heavy cell stayed loss-optimal to collapse under every
   incentive tried within the geometry-only encoder.
3. **The MF-token fix.** Matched-filter overlap features (§3.1) carry the
   mass statistic geometry provably lacks (in-cell AUC 0.989 vs. 0.532).
   `mf_tokens` + `--delay_aug` + `--aux_mass`, trained from scratch, closed
   the basin for every synthetic and semi-synthetic test —
   **`lean_npe_mf_scratch`** (superseded: see §2, the attractor's root cause
   was later found in the training data).
4. **Residual real-event failure, root-caused.** GW150914/GW170814's own
   exact strain still collapsed. Signal-subtraction + token-grafting showed
   the collapse is carried by the MF-token branch's response to their
   PARTICULAR local noise realization, not their astrophysical content.
   `noise_bank_severity_scan.py` confirmed both segments are genuine z≈-11
   matched-filter-SNR-suppression outliers vs. the training noise bank —
   not reachable by more consistency training over the existing bank.
   **Decision: do not chase via bespoke training; rely on the OOD/confidence
   gate**, independently verified to correctly flag both events and route
   them to dynesty (which recovers them correctly).
5. **`aux_mass`/`delay_aug` are necessary, not incidental.** An ablation
   (`mf_tokens` alone, neither lever) reproduced the full collapse — MF
   tokens give the encoder access to the right statistic but do not by
   themselves teach the model to use it.
6. **Amortization vs. representation, resolved.** A non-amortized control —
   the *same* architecture trained only on a narrow, GW150914-sky-locked
   population (never sharing weights across the full training population) —
   recovered a sensible, non-collapsed mass ratio on the real, held-out
   GW150914 strain (statistically indistinguishable from synthetic twins at
   the same parameters). **The residual real-event failure is caused by
   population-wide amortization, not a representational ceiling** — the
   capability exists; sharing one function across the whole population is
   what trades it away.

### 5.3 Encoder fusion vs. flow conditioning (2026-07-22, ongoing)

1. **Information-loss localization.** A layer-by-layer linear-probe trace
   (`mf_embedding_tap_v2.py`) found the additive MF/geometry token merge
   (`"add"` fusion) is the single largest information-destruction stage in
   the whole encoder — bigger than the MF branch's own embedding compression.
2. **The `"concat"` fix — representation improved, posterior did not.**
   Replacing the additive merge with independent tokens (§3.1) demonstrably
   improved every representational checkpoint (final-context mass-ratio AUC
   0.848→0.889), confirmed both by a training-free frozen-weight test and by
   training `lean_npe_mf_concat` to completion. **But the actual sampled
   posterior showed no improvement — and the real GW150914 collapse got
   worse** (railing 0.715→0.888). This is the clearest evidence in the
   project that encoder representation quality and posterior quality can
   decouple.
3. **Direct test: the flow ignores the q-informative direction.**
   `flow_context_sensitivity.py` manually perturbed real events' contexts
   along the exact direction a ridge regression identifies as most
   mass-ratio-predictive (by 3× the natural real-event spread on that axis)
   and found the SAME tiny output shift as an equal-magnitude perturbation
   along a **random** direction — the flow's conditioning is not
   preferentially routing through the informative subspace, even though it
   demonstrably exists. This is a training-effect finding, not a capacity
   one: the flow's per-layer context injection is a full dense linear map
   with structural access to any direction; training simply never
   discovered/rewarded using this one.
4. **Current experiment: a dedicated q-conditioning embedding** (§3.1,
   `--q_embed_dim`/`--aux_q_embed`) — hand the flow a small, explicitly
   supervised channel instead of asking it to find mass-ratio information
   amid 256 competing dimensions. If the flow starts tracking it, the issue
   was locational; if not, the bottleneck is inside the flow's conditioning
   mechanism itself, and the next candidates (ranked, not yet tried) are a
   nonlinear per-layer conditioner, a context residual block, FiLM-style
   modulation throughout the conditioner, and — last resort, most invasive —
   cross-attention from the flow into a set of context tokens instead of one
   flat vector.

---

## 6. Using it

All commands run in the conda env **`ahsd` (Python 3.10)**, which has an
editable install pointing at `src/`. Prefix with `conda run -n ahsd` (or
activate it) — do **not** run under a bare system Python, where a stale
installed `ahsd` can shadow `src/`.

### Setup

```bash
conda activate ahsd          # torch (MPS/CUDA), bilby, gwpy, nflows
pip install -e .             # editable install into the env
```

If you'll train with `--mf_tokens`, build the matched-filter template bank
once (fast, offline):

```bash
conda run -n ahsd python src/ahsd/models/build_mf_bank.py   # -> src/ahsd/models/assets/mf_bank.npz
```

### Generate a dataset (component storage)

```bash
# parallel shards merged into one dataset (~8 min for 200k events on 32 cores)
conda run -n ahsd python src/ahsd/data/scripts/generate_parallel.py \
    --config configs/data_config_fixed.yaml --n-samples 200000 \
    --output-dir data/dataset_v3 --shards 32 --seed 2026
# then audit every event before training
conda run -n ahsd python scripts/audit_dataset.py data/dataset_v3 --dist-range 8 6000
```

`configs/data_config_fixed.yaml` (uniform distance prior) is the default;
`data_config_{bbh,nsbh,bns}_v3.yaml` are single-class variants. See
`docs/DATASET_GENERATION.md` for the layout, the audit, and the generator
bugs fixed on 2026-10-31.

### Download the real-noise bank

```bash
conda run -n ahsd python scripts/download_gwosc_noise_bank.py \
    --per_det 120 --outdir data/noise_bank
```

### Train (current deploy-candidate recipe: coherent encoder + MF tokens + PSD conditioning + real noise)

```bash
conda run -n ahsd python -u experiments/train_lean_npe.py \
    --data data/dataset --remix \
    --epochs 40 --batch 128 --lr 3e-4 \
    --encoder_type coherent --mass_param mc_q --psd_cond --psd_bands 16 \
    --mf_tokens --mf_token_fusion add --delay_aug 0.5 --aux_mass 1.0 --geom_dropout 0.25 \
    --noise_bank data/noise_bank --real_noise_prob 0.9 --det_dropout 0.2 \
    --ckpt_every 2 --max_spurious_railing 0.10 \
    --outdir model/lean_npe
```

The first run builds a memmap cache (`data/dataset/memmap/`) so epochs stream
from disk. Fine-tune from a checkpoint with `--init_from <ckpt>` (fresh
optimizer/schedule; note `--init_from` zero-pads new-but-compatibly-shaped
weights, but cannot bridge a `--q_embed_dim` change, since that reshapes the
flow's context input in the middle of the vector, not at the end — train a
`q_embed_dim` run from scratch, or extend `--init_from`'s padding logic first).

**Research-mode flags** (see §3.1/§5.3 for the evidence behind each):

| Flag | Effect | Status |
|---|---|---|
| `--mf_token_fusion {add,concat}` | additive vs. independent-token MF/geometry fusion | `add` deployed; `concat` representationally better, posterior unchanged |
| `--q_embed_dim N` | dedicated, explicitly-supervised mass-ratio channel fed directly to the flow | experimental, result pending |
| `--aux_q_embed W` | SmoothL1 weight supervising the embedding above (requires `--q_embed_dim>0`) | paired with the above |
| `--delay_aug P` | per-signal probability of remapping to a scored sky candidate | required to avoid the attractor-basin failure mode (§5.2) |
| `--basin_oversample` | extra weight/repeats on the known hard geometry cell | measured insufficient alone (§5.2) |
| `--aux_mass W` | training-only SmoothL1 mass regression on the encoder context | required alongside `--mf_tokens`/`--delay_aug` (§5.2) |
| `--geom_phase_residual` | append residual polarization phase (post delay-removal) to geometry features | measured inert on the hardest cell; harmless elsewhere |
| `--premerger` | widen the time scaler for early-warning (pre-merger) training | separate experimental track |

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
| `val` vs `train` | fall together | val flat/rising while train falls → memorization (or, for a narrow-population run, a validation-set domain mismatch — see §4's gotcha) |
| `REAL …` block | converges toward Gaussian | stays far apart → raise `real_noise_prob` / bank size |

A trainer-internal **basin probe** (a fixed set of hard-cell injections, see
§5.2) also prints every epoch alongside the corner table — this is the
project's own early-warning diagnostic for the attractor failure mode and is
invisible to every metric above it.

### Certify a checkpoint (SBC / coverage / railing + GWTC real-event smoke tests → HTML)

```bash
conda run -n ahsd python scripts/validate_checkpoint.py --model model/lean_npe_v3/best_model.pth
```

Other benchmarks: `scripts/benchmark_real_events.py` (vs dynesty),
`scripts/overlap_benchmark.py`, `scripts/twin_grid.py` (twin-injection bias),
`scripts/dynesty_compare.py`. The attractor-basin-specific diagnostics
(`analysis/attractor_basin/`) are the right tools for anything touching
geometry/MF-token behavior specifically — start with `mf_probe.py` (offline
go/no-go), `mf_embedding_tap_v2.py` (full information trace, works for both
fusion modes), and `flow_context_sensitivity.py` (context-perturbation test).

### Inference CLI (real GWOSC event / local strain / fresh injection)

```bash
conda run -n ahsd python infer.py --event GW150914 --samples 10000 --output results/GW150914
```

### Sample a posterior in code

```python
import torch
from ahsd.models.lean_npe import LeanNPE

ckpt = torch.load("model/lean_npe_v3/best_model.pth", map_location="cpu", weights_only=False)
a = ckpt["args"]
model = LeanNPE(premerger=a.get("premerger", False),
                psd_cond=a.get("psd_cond", False) or False,
                psd_bands=a.get("psd_bands", 16),
                encoder_type=a.get("encoder_type", "conv"),
                mass_param=a.get("mass_param", "m1m2"),
                geom_phase_residual=a.get("geom_phase_residual", False),
                mf_tokens=a.get("mf_tokens", False),
                mf_shape_norm=a.get("mf_shape_norm", False),
                mf_token_fusion=a.get("mf_token_fusion", "add"),
                q_embed_dim=a.get("q_embed_dim", 0))
model.load_state_dict(ckpt["model_state_dict"]); model.eval()

strain = torch.randn(1, 3, 16384)   # whitened H1/L1/V1 @ 4096 Hz, 4 s
post = model.sample_posterior(strain, rank=0, n_samples=3000)   # [1, 3000, 11] physical
# rank=0 = loudest signal; rank=1 = second-loudest, etc.
```

Every `LeanNPE(**kw)` argument above must be threaded through **every**
checkpoint-loading site when a new one is added — this has caused real bugs
twice (`geom_phase_residual`, then `mf_tokens`) and a near-miss a third time
(`mf_token_fusion`). Current load sites: `src/ahsd/inference/pipeline.py`,
`scripts/validate_checkpoint.py`, `scripts/twin_grid.py`,
`scripts/overlap_benchmark.py`, `scripts/lean_npe_diagnostics.py`,
`scripts/lean_npe_extended_eval.py`, `scripts/real_noise_test.py`,
`scripts/dynesty_compare.py`, `scripts/probe_tiers_and_amplitude.py`,
`experiments/frozen_context_heads.py`,
`analysis/gw150914_localization/_common.py`, every script under
`analysis/attractor_basin/` that loads a model, and the `infer.py` path
(via `pipeline.py`).

Production inference should go through `ahsd.inference.infer` / `infer.py`,
which add preprocessing, OOD/confidence gating, and importance reweighting —
this is the layer that correctly catches the residual GW150914/GW170814
failure mode described in §5.2 and routes it to dynesty.

---

## 7. Project structure

```
src/ahsd/
├── models/
│   ├── lean_npe.py            # LeanNPE: encoder + rank embedding + NSF (the model) + ParamScaler
│   ├── coherent_encoder.py    # CoherentEncoder: geometry + MF-token branches → fused tokens
│   ├── build_mf_bank.py       # offline: builds assets/mf_bank.npz (the MF template grid)
│   ├── flows.py               # NSFPosteriorFlow (nflows-based); MaskedContextLinear conditioner
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

analysis/attractor_basin/      # the (30,30) collapse investigation (§5.2/§5.3) — REPORT.md is the
                                # full evidence trail; mf_probe.py / mf_embedding_tap_v2.py /
                                # flow_context_sensitivity.py / probe4_eval.py are the reusable
                                # diagnostic tools, not one-off scripts
analysis/gw150914_localization/  # distance-bias root-cause study; _common.py has shared
                                  # model-loading/real-event-preparation helpers reused throughout
analysis/divergence_localization/  # the "val improves / GWTC worsens" calibration-collapse study

infer.py       # inference CLI (real GWOSC event / local strain / fresh injection)
analysis/      # measured evidence: diagnostic reports, audits, comparison outputs (the paper's evidence)
configs/       # YAML configs (data_config_fixed.yaml is the default; data_config.yaml is a legacy reference)
data/          # dataset (component storage), memmap caches, noise_bank  (git-ignored)
model/         # checkpoints (best_model.pth bundles weights + args + diagnostics). Every pre-v3
               # run was trained on contaminated data and deleted (2026-10-31); lean_npe_v3
               # (mixed) and lean_npe_{bbh,nsbh,bns}_v3 (per class) are the current models
               # research checkpoints, and most others are historical/superseded generations kept
               # for reproducibility of the evidence trail
```

> Note: the model's flow size (`flow_layers`, `flow_hidden`, `flow_bins`) is a
> hardcoded `LeanNPE` default, not stored in `args`. A checkpoint loads
> `strict=True` only if the current defaults match the generation it was trained
> with; reconcile a size mismatch against the checkpoint before assuming a bug.
> Note: `./build/` is a git-ignored setuptools artifact — never on `sys.path`.

---

## 8. Roadmap

1. **Resolve the flow-conditioning bottleneck** (§5.3): evaluate the
   dedicated q-conditioning embedding; if the flow still doesn't track mass
   ratio through it, move to a nonlinear per-layer conditioner, then a
   context residual block / FiLM modulation, reserving cross-attention
   conditioning as the highest-cost fallback.
2. Once resolved, re-evaluate `mf_token_fusion="concat"` as the new default
   — its representational advantage should finally show up in the posterior.
3. Grow real-noise-bank diversity (more GWOSC segments across O1–O4) as a
   population-wide hedge against the severity-outlier failure mode (§3.3),
   rather than chasing any single event.
4. Re-run the full production config under calibration-gated selection with
   whichever architecture wins above; benchmark on GWTC events vs. dynesty.
5. Encoder temporal resolution — phase-preserving features to tighten timing
   and sky localization.
6. Pre-merger (early-warning) training run (`--premerger`) to production
   maturity.
7. End-to-end overlap pipeline on real-noise overlaps.

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
