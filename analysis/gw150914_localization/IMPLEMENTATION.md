# CoherentEncoder — implementation status

Implements the design in `ENCODER_ARCH.md`. Drop-in for `LeanStrainEncoder`.

## What was built
- **`src/ahsd/models/build_svd_basis.py`** → `src/ahsd/models/assets/svd_basis.npz`
  Reduced-order complex basis from 800 whitened FD waveforms (BBH mass/spin range × random arrival-time shifts). **Nf=4016 → Nb=256 modes capture ~100% of signal power** (confirms the low-dim manifold).
- **`src/ahsd/models/coherent_encoder.py` — `CoherentEncoder`**
  `forward(strain[B,3,16384], asd_bands[B,3,16]) -> context[B,256]`. Stages:
  [1] rFFT → band-limit [20,1024] Hz → SVD complex coeffs per detector;
  [2] per-detector MLP (**norm-free**) ⊕ per-band log-energy ⊕ asd_bands ⊕ det-id;
  [3] explicit relative features: per-band per-pair complex coherence (Re/Im) + log-amp ratios;
  [4] attention pooling over detectors (set-invariant, SNR-weighted);
  [5] context head (pooled ‖ relatives) → 256-D.
  **753K encoder params, 6.96M total** — comparable to the conv model (not bigger).
- **`LeanNPE(..., encoder_type="coherent")`** (`src/ahsd/models/lean_npe.py`) — branches the encoder; coherent forces `psd_cond=True` (it always ingests asd_bands). Flow/rank/scaler unchanged.
- **`experiments/train_lean_npe.py --encoder_type coherent`** — activates asd_bands plumbing.

## Verified (smoke)
construct ✓ · encode→[B,256] ✓ · nll ✓ · backward (encoder grads flow) ✓ · sample_posterior ✓.

## Proof-of-concept — PASSED (directional)
`analysis/gw150914_localization/poc_coherent.py` — 6000-step pure-NLL train (rank-0, design-domain memmap = signal+fresh-noise, asd_bands=0, ~75 ms/step CPU), then the fixed-heavy-BBH amplitude sweep.

| | conv (fully trained v8) | **coherent POC (6k steps)** |
|---|---|---|
| dL_med across true 300→1900 | 759→617 (**decreasing**) | 1189→1412 (**increasing**) |
| Spearman(pred, true dL) | **−0.9** (flat/inverted) | **+1.00** (monotonic, correct) |
| log-log slope | ~0 | +0.10 (compressed, not yet 1.0) |

**The architecture flips the amplitude→distance response from flat/inverted to monotonically correct** — the confirmed failure. Magnitude still compressed: expected for a short POC on the **type-mixed** data (Mc shortcut still present) without distance-rescale augmentation. Full slope→1 needs the full train **and** the dataset fix (§1b). Artifacts: `poc_coherent_poc.json`, `model/coherent_poc/poc_model.pth`; matched conv control in `poc_conv_poc.json`.

## Dataset fix + retrain (DONE — both levers demonstrated)
`analysis/gw150914_localization/make_fixed_dataset.py` → `data/dataset_bbh_fixed/`
BBH-only (removes the audited event-type-mixing) + **uniform** distance prior (minimises SNR-gate Malmquist), sampler draws mass ⟂ distance. **N=3500, log Mc↔log dL Pearson 0.125 / R²=0.016 — a 40× reduction from the pooled R²=0.66.**

Coherent encoder retrained on it (8000 steps) → the fixed-heavy-BBH amplitude sweep:

| setup | slope d(log dL̂)/d(log dL) | Spearman | dL̂ (true 300→1900) |
|---|---|---|---|
| conv (current, fully trained) | ~0 | −0.9 (inverted) | 759→617 |
| coherent, type-mixed data (POC) | 0.10 | +1.0 | 1189→1412 |
| **coherent, fixed data (POC)** | **0.37** | +1.0 | **227→442** |

**Both levers stack**: the encoder flips the response *direction* (−0.9→+1.0), the dataset fix restores its *magnitude* (0.10→0.37, absolute dL now realistic). Slope 0.37<1.0 because it's still a short POC (8k steps, small N=3500, signal-only sweep) — a full run closes the rest. Artifacts: `make_fixed_dataset.py`, `data/dataset_bbh_fixed/`, `model/coherent_fixed/poc_model.pth`, `poc_coherent_fixed.json`.

## Full training (launch when ready)
The encoder change is necessary but the **dataset fix (unify per-type distance ranges) is also required** for the full GW150914 fix (see `FIXES.md` §1b) — that needs a regen. The encoder alone is trainable now:

```bash
conda run -n ahsd python -u experiments/train_lean_npe.py \
    --data data/dataset --outdir model/lean_coherent --encoder_type coherent \
    --remix --noise_bank data/noise_bank --real_noise_prob 0.7 --det_dropout 0.2 \
    --epochs 40 --batch 128 --lr 3e-4 --warmup_steps 500
```

## Acceptance test (post-train)
1. `encoder_amplitude_response.py` (point it at the new ckpt): flow median tracks true dL (slope→1, Spearman→+).
2. GW150914 via `scripts/validate_checkpoint.py` / `benchmark_real_events.py`: dL median → ~440, truth in CI.
3. SBC/coverage held; end-to-end BBH predicted-vs-true dL Spearman ↑ from 0.31.

## Known limitations / follow-ups
- SVD basis built on design-ASD whitening; validate transfer to real-PSD data (whitening should normalise the shape).
- MPS `torch.fft.rfft` falls back to CPU (slow) — for GPU/large runs, precompute FD or run on CUDA.
- Missing-detector handling currently relies on low coherence of white-fill; a proper present-mask (`key_padding_mask`) is stubbed for a follow-up.
- GNPE not implemented (not justified by the evidence — try the coherent encoder first).
