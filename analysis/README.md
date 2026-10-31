# analysis/ — evidence index

Every folder here is a **measured** diagnostic that justified (or killed) a
design decision. They are grouped by the question they answered, in roughly
chronological order, so the sequence reads as the paper's "what failed and
why" narrative. Each entry: what was tested → verdict → what replaced it.

> **Every model before `lean_npe_v3` (2026-10-31) was trained on contaminated
> data.** The generator injected a fixed placeholder source (m1 = m2 = 30,
> face-on, zero spin) into overlap events; it made up 70% of all heavy
> equal-mass training signals. The reports below are kept as the record of
> what that bug did — most of §4's "attractor" is its fingerprint (railing on
> GW150914/GW170814 fell from ~74% to 0.5–4% on clean data). Their
> checkpoints and NPE results have been deleted; the dynesty/bilby runs in
> `results/` do not involve the models and remain valid ground truth.
>
> The per-model certification reports of those runs (`ci/<run>/report.*`)
> and `figures/` were then removed from the working tree as well; every table
> below that cites them reflects their contents, and the files are recoverable
> from git history (`git show 818340a:analysis/ci/<run>/report.json`). Note
> that `ci/lean_npe_v3/` now holds the 2026-10-31 clean-data model; the July
> run of the same name (the third LeanNPE generation in §6) is in history.

Regenerable bulk (posterior-sample dumps, dynesty resume pickles, `.npy`/`.pt`
caches, per-event CI folders) was moved out on 2026-10-30 to
`~/Developer/PosteriFlow-analysis-archive/` (same layout; `MANIFEST.txt`
lists every item). Scripts in `attractor_basin/` that read cached contexts
need those caches restored or regenerated. Bilby/dynesty `*_result.json`
reference posteriors are **kept** — they are the ground truth for every
NPE-vs-Bilby comparison and cost hours to recompute.

---

## 1. OverlapNeuralPE → LeanNPE: the encoder ignored the data

| Folder | Tested | Verdict |
|---|---|---|
| `lean_npe_v2_diagnostics/` | first LeanNPE: coverage, SBC, PP, distance-vs-SNR | baseline calibration established; distance weakly informed |
| `lean_npe_v2_extended/` | v2 calibration by SNR and overlap count | low-SNR vs high-SNR calibration split; per-event table |
| `encoder_vs_flow/` | is the shrinkage in the encoder or the flow? | encoder-limited: flow reproduces its context's information |
| `encoder_layer_audit/` | probe R² vs depth through the encoder | where information appears along the stack |
| `encoder_geometry/` | PCA/t-SNE of contexts coloured by parameters | which quantities are entangled in the context |
| `encoder_ab_compare/`, `encoder_v2_postmortem/` | experimental encoder v2 vs lean encoder | **FAILED** — v2 did not beat the lean encoder |
| `ablation_tier0/`, `probe_tiers_amplitude/` | linear-probe tiers; amplitude path ablation | amplitude reaches the context (no-normalization invariant kept) |

## 2. Distance bias

| Folder | Tested | Verdict |
|---|---|---|
| `priors/` | train / validation / GWTC distributions (Mc, q, dL, SNR) | training prior vs real-event mismatch quantified |
| `degeneracy/` | mass–distance joint posteriors in several regimes | dL–Mc coupling is a model shortcut, not physics |
| `distance_pipeline/` | PIT/reliability, IS on real events, sensitivity sweep | not PSD-conditioned → amplitude ambiguous; DINGO-IS **fails** on real events |
| `anchors/` | NPE vs dynesty at 4 fixed (Mc, q, θjn) anchors | part of edge-on distance bias is irreducible (dynesty shares it) |
| `twin_dynesty/` | GW150914 twin: NPE vs dynesty on identical data | likelihood recovers the truth → NPE under-extracts |
| `gw150914_localization/` (`REPORT.md`, `AUDIT_mcdl.md`) | step-by-step localization of GW150914's dL error | **Mc-mediated distance; amplitude encoded but unused**; Mc↔dL R²=0.81 was event-type mixing → `configs/data_config_fixed.yaml` (uniform distance prior) |

Follow-up (2026-10-30, `mf_scratch`): BBH distance is calibrated when
conditioned on the prediction; remaining failures were the 40–2200 Mpc
ParamScaler clamp (BNS 2.3× too far, GW170729 unreachable) and an unrecorded
training prior → `--dist_range`, dataset provenance, per-type distance gate
in `scripts/validate_checkpoint.py`.

## 3. "Validation improves while real events get worse"

| Folder | Tested | Verdict |
|---|---|---|
| `divergence_localization/` (`REPORT.md`, `mechanism/`) | SNR sparsity, density holes, PSD-OOD, tail_bound, flow sharpening | **root cause = cosine LR schedule sized for 140 epochs** (flow over-sharpens at ~96% peak LR) + val-NLL-only selection picks the worst-calibrated checkpoint. Fixed by sizing the schedule + calibration-gated selection (`ci/lean_npe`, val −1.50, railing 2.3%) |
| `ci/long_run_ep18/`, `ci/long_run_gated/` | the diverging run before/after | the regression the study explains |

## 4. The (30,30) mass-ratio attractor on GW150914 / GW170814

| Folder | Tested | Verdict |
|---|---|---|
| `gw150914_modes/` | mode structure / railing of the GW150914 posterior | one correlated q→1 + spin + θjn railing mode, not independent railing |
| `attractor_basin/` (`REPORT.md`, 1000+ lines) | twin tests, token grafting, five training-side fixes, MF tokens, amortization test | localized to the coherent encoder's geometry tokens (inter-detector delay aliases with polarization phase); **five fixes failed** (see table below); `mf_scratch` fixed every synthetic test; the residual real-event collapse is **amortization, not representation** (a narrow non-amortized model resolves it) |

Training-side fixes, each certified in `ci/` — GW150914 stayed at ≈(31, 31)
while validation NLL kept improving:

| `ci/` run | Change | val NLL | GW150914 m1/m2 | railing | Outcome |
|---|---|---|---|---|---|
| `lean_npe` | calibration-gated baseline | −1.50 | 30.2 / 30.0 | 69% | attractor present |
| `lean_npe_mcq` | (Mc, q) flow coordinates | −0.92 | 30.7 / 30.6 | 64% | flow parameterization exonerated |
| `lean_npe_mcq_gdrop2` | geometry-token dropout 0.5 | −1.17 | 31.4 / 31.0 | 52% | fixes loud rows only |
| `lean_npe_pslope2` | phase-slope delay feature | −1.40 | 44.0 / 31.1 | 48% | first un-lock of real GW150914 |
| `lean_npe_presid` | + explicit phase residual | −1.60 | 31.2 / 31.0 | 52% | best population numbers, re-locked |
| `lean_npe_boost` | 4× basin-cell oversampling | −1.60 | 30.6 / 30.6 | 63% | incentive is not the lever |
| `lean_npe_mft` | MF tokens (fine-tune) | −1.61 | 43.3 / 31.6 | 48% | partial |
| `lean_npe_noiseinv`, `_noiseinv_real` | noise-invariance views | −1.28 / −1.26 | 31.1 / 31.0 | 82% / 79% | worse on real events |
| `lean_npe_mf_scratch` | MF tokens + delay aug + aux mass, from scratch | −0.97 | 31.0 / 30.9 | 74% | best of the contaminated-data era: all synthetic basin tests pass, real events still collapse |

## 5. Why LeanNPE's posteriors are wider than Bilby's

| Folder | Tested | Verdict |
|---|---|---|
| `encoder_audit/` (`REPORT.md`) | where the 54× Mc / 5× q information deficit arises | no lossy stage — the encoder never **extracts** the information (stem 1.4×, pooling not lossy) |
| `posterior_bias_audit/` (`REPORT.md`, Bilby runs in `eH_*`) | flow expressiveness, parameterization, q tails, Mc width vs Bilby | flow is **not** the bottleneck (matches its context's information to 4–21%); q is essentially **unmeasured** (σ_post/σ_prior 0.89); Mc ~50× wider than GWTC on GW170608 |

## 6. Model generations (`ci/`)

`ci/<run>/report.json` + `report.html` (self-contained, images embedded) is
the certification record of every checkpoint generation; per-event outputs
are kept only for the current best, `ci/lean_npe_mf_scratch/`.

| Run | val NLL | gates | GW150914 m1/m2, dL (pub 38.8/33.4, 440) | railing | Note |
|---|---|---|---|---|---|
| `lean_npe_v2` | 0.69 | 7/10 | – | – | first LeanNPE |
| `lean_npe_v3` | −0.71 | 10/10 | – | – | inference package + CI suite |
| `lean_npe_v4`, `_v5` | −0.53 / −0.51 | 9/10 | 60.5/36.3, 1362 · 53.8/30.7, 1219 | 20% / 31% | early (ep 1–2) checkpoints |
| `lean_v6` | −0.02 | 8/10 | 46.7/29.9, 1079 | 45% | 256-wide flow |
| `lean_v7_aux` | −0.04 | 9/10 | 50.6/30.0, 1144 | 43% | aux heads — **neutral**, removed |
| `lean_v8_psdcond`, `_ft` | 1.01 / −1.19 | 8/10 · 9/10 | 31.1/30.1, 1120 · 30.1/30.0, 737 | 55% / 61% | PSD conditioning: fixes sensitivity, not real-event dL |
| `lean_coherentC` | 1.37 | 9/10 | 30.4/10.9, 407 | 14% | coherent encoder, calibrated |
| `long_run_ep18`, `long_run_gated` | 1.00 / 2.85 | 8/10 · 5/10 | 50.4/29.5, 632 · 30.3/9.6, 466 | 41% / 17% | §3 divergence |
| attractor-fix runs | see §4 | | | | |

## 7. Resolution: the data, not the model (2026-10-31)

| Evidence | Where | Finding |
|---|---|---|
| full-dataset audit | `scripts/audit_dataset.py` | overlap slots injected a fixed placeholder source (m1 = m2 = 30, face-on, zero spin): **70% of heavy equal-mass training signals** in every pre-v3 dataset |
| clean 50k vs contaminated | `ci/lean_npe_v3_50k/` | same size, clean data: GW150914 railing 74% → 0.7% — the (30, 30) attractor is that placeholder |
| clean 200k | `ci/lean_npe_v3/` | val NLL −2.32, SBC 10/11, railing 0.7%; GW150914 detector-frame m1/m2 42.6/35.1 (truth 42.4/36.5), q 0.83 (0.86) |
| type confusion | `scripts/type_confusion_probe.py` | the mixed model puts 18% (BBH) / 48% (NSBH) / 38% (BNS) of posterior mass on the wrong class; restricting to the right class removes the per-class distance bias (BNS ×0.51 → ×0.90) → per-class models |

## Other

- `figures/` — overview figures (training mass plane, distance prior check, overlap separation, noise-bank quality, twin-grid biases).
- `overlap_benchmark.json`, `twin_grid_v3.json`, `context_conditioning_test.json` — outputs of `scripts/overlap_benchmark.py`, `scripts/twin_grid.py` and the context-shuffle conditioning test.

Removed as superseded (in the archive): `enc_finetune_distance_only/`,
`enc_finetune_multitask/` (bare training histories of an encoder fine-tune
later replaced by the aux-head experiment), `logs/` (raw supervised-q logs).
