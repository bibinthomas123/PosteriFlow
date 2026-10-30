# Audit: is the Mc→dL R²=0.81 real, or an artifact?

**Verdict: the correlation is REAL in the pooled training data (directly measured), but it is an ARTIFACT of EVENT-TYPE MIXING — it does NOT reflect a within-BBH Mc–dL correlation.** My earlier "the model reads distance off a strong Mc–dL prior" was based on the pooled number and is retracted as stated; see the corrected mechanism below.

## 1. How the training population is generated

- **Sampler** (`src/ahsd/data/parameter_sampler.py`): masses and distance are drawn **independently**.
  - masses log-uniform (`_sample_masses`), distance P(d)∝d² (`_sample_distance`), spins/angles isotropic. Docstring: *"SNR is NOT targeted here — measured SNR drives accept/reject downstream."*
  - **Per-event-type ranges** (the key confound), lines 32–39:
    - distance: BBH **[50, 2000]**, BNS **[10, 300]**, NSBH **[20, 800]**
    - mass: BBH [5,100], BNS [1,2.5], NSBH [3,100]+[1,2.5]
- **The only filter** (`src/ahsd/data/dataset_generator.py:356–359`): reject a sample if network matched-filter SNR < `min_snr` (default **6.0**, `_make_sample` → `rejected_snr`). No SNR-distribution rejection, weighting, or curriculum in the accept path (the `snr_distribution` config is diagnostic only).

## 2. Direct measurement (stored data, bypassing the probe)

log Mc vs log dL, computed straight from `data/dataset/memmap/*/params.npy`:

| population | Pearson(log) | R² | Spearman |
|---|---|---|---|
| train, ALL types pooled | +0.803 | 0.644 | 0.740 |
| val, ALL types pooled | +0.812 | 0.660 | 0.750 |
| **val, BBH only (m₁,m₂≥5)** | **+0.056** | **0.003** | +0.033 |
| val, BNS only | +0.061 | 0.004 | +0.097 |
| val, NSBH only | +0.129 | 0.017 | +0.103 |

**Within any single event type the correlation is ~0.** The pooled 0.81 is the block structure of three type clusters (BNS: low-Mc/near; NSBH: mid; BBH: high-Mc/far) — visible as a staircase in `audit_mcdl_scatter.png` panel C.

## 3. Where 0.81 comes from (mechanism decomposition)

| step | Pearson(logMc,logdL) | reading |
|---|---|---|
| (A) raw sampler, BBH, no selection | **0.001** | masses ⟂ distance confirmed |
| (B) after SNR≥6 gate, BBH only | 0.147 | SNR cut induces only a weak within-BBH correlation… |
| (B) after SNR≥8 / ≥12 / ≥20 | 0.22 / 0.34 / 0.53 | …strengthening with threshold, but the generator uses 6 |
| (C) stored pooled, all types | 0.81 | **type mixing**, not within-BBH selection |

So: (i) **event-type mixing** dominates the pooled 0.81; (ii) a **weak within-BBH Malmquist selection** (~0.15 at the min_snr=6 gate) exists but is nearly inactive because BBH in [50,2000] Mpc are loud enough to pass SNR>6 regardless of distance (hence within-BBH R²=0.003).

Where the probe's R²=0.81 came from: it was computed on the **pooled validation set** (BNS+NSBH+BBH), so it inherited the type-mixing correlation. Not leakage, not a subset, not a probe bug — a genuine (but type-mixing-driven) dataset property.

## 4. Within-BBH: the real (much smaller) correlation the model can exploit

The within-BBH correlation depends on which population you measure (all measured via GBM, cross-validated; shuffled-Mc control R²=−0.02, so the probe is honest):

| within-BBH population | Pearson(logMc,logdL) | GBM R² |
|---|---|---|
| all per-signal rows (incl. faint overlap secondaries) | 0.056 | 0.145 |
| detected/loud, **SNR>12** (the GW150914 regime) | **0.377** | 0.42 |
| detected/loud, SNR>20 | ~0.53 | — |

This is **Malmquist selection** at the effective detection threshold (heavier detected BBH can be farther), matching the injector-gate simulation in §3 (SNR≥12→0.34, ≥20→0.53). So the correlation the model can legitimately use for a GW150914-loudness BBH is **~0.4, not 0.81**.

**End-to-end (BBH val events): the model over-relies on chirp mass for distance.**

| predicted-median dL vs … | Spearman |
|---|---|
| chirp mass | **+0.68** |
| TRUE dL | +0.31 |
| TRUE SNR (amplitude) | +0.31 |

Within-BBH the context *does* carry amplitude (context→SNR\|Mc R²=0.63) and distance-beyond-Mc (context→dL\|Mc R²=0.53), but the posterior median tracks Mc (0.68) far more than true distance/amplitude (0.31). At fixed masses a 5× amplitude change moves the posterior ~0 (Step 4b).

## 5. Conclusion

**✓ The dataset genuinely contains the Mc–dL correlation (0.81 measured), but the 0.81 magnitude is an EVENT-TYPE-MIXING artifact.** Within BBH the true correlation is 0.05–0.15 (all signals) up to ~0.4 for detected loud events (Malmquist selection) — **never 0.81**. Not a probe bug, subset trick, or leakage: the probe simply ran on the pooled (all-type) validation set.

**Corrected localization (direction unchanged, magnitude/explanation corrected):** the model infers BBH distance disproportionately from **chirp mass** (predicted∝Mc Spearman 0.68 vs true-dL 0.31), over-using the *modest* ~0.4 selection-induced Mc–dL relation and under-using the amplitude information that is present in the context. The original "strong 0.81 Mc–dL prior" wording is retracted — the exploited correlation is ~0.4, and it is selection-induced within detected BBH, not a raw prior. The pipeline/signal-specific/real-noise-domain rejections (Steps 0/2/4/4b) and the fixed-mass flat amplitude response (4b) are unaffected.
