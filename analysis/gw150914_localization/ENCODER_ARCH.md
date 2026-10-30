# LeanNPE encoder v2 — from-scratch architecture

Complete spec with tensor shapes. Batch = B; detectors D≤3 (H1,L1,V1); 4 s @ 4096 Hz.
Builds on the localization: the bottleneck is the **geometry term Θ(ι,α,δ,ψ)**, not amplitude; the fix is a **phase-coherent, geometry-explicit** front-end, not more capacity and not (on current evidence) GNPE.

---

## 1. Design principle

Per detector the whitened data factorize:

    d_d(f) = [ (Mc^{5/6}/d_L) · 𝒜_d(α,δ,ψ,ι) · e^{2πif τ_d} ] · h₀(f; Mc,q,χ) / (common intrinsic mode)  + n_d(f)
             └──────────── extrinsic prefactor P_d ────────────┘   └── intrinsic, common across detectors ──┘

- **Intrinsic** (Mc,q,χ) live in the *common-mode* waveform h₀(f) — already well captured (Mc R²≈0.7).
- **Extrinsic geometry** Θ lives in the *relative* prefactors P_d/P_d′ = |𝒜_d/𝒜_d′|·e^{i(Δφ)}·e^{2πifΔτ}. This is what the current encoder loses.

The encoder must therefore (a) compress each detector to phase-coherent complex features, (b) expose the **relative** inter-detector quantities explicitly, (c) keep amplitude un-normalized.

---

## 2. Component-by-component (physical quantity · why current fails · why it helps distance · cost · complexity · expected gain · drawbacks)

### [1] Complex FD input + SVD reduced-order projection  — **ADOPT**
- **Physical quantity**: the full complex per-detector signal `d_d(f)` on the signal manifold; magnitude = amplitude (∝Mc^{5/6}/d_L·|𝒜_d|), phase = coalescence phase + `2πfτ_d` + antenna phase.
- **Why current fails**: TD stem applies `asinh` (amplitude–morphology entanglement) + real convs + attention pool → relative phase is implicit and smeared; amplitude survives only via a bolt-on log-energy branch.
- **Why it helps distance**: keeps phase explicit and per-bin, so §3 can form the relative quantities that fix Θ, hence d_L.
- **Cost**: rFFT (cheap); SVD projection is a fixed matmul `V^H d`, `V∈C^{Nf×Nb}` — 3·Nf·Nb ≈ 3·4016·200 ≈ 2.4M MACs/sample (negligible vs a transformer). Compresses input ~20× (4016→200/det).
- **Complexity**: moderate — precompute the SVD basis from a whitened waveform bank once; band-limit [20,1024] Hz.
- **Expected gain**: large — this is the representation dynesty/DINGO effectively use.
- **Drawbacks**: basis is PSD/whitening- and waveform-family-dependent (mitigated: whitening normalizes the signal shape, so a design-whitened basis transfers; refit if the waveform family changes). Precession/HOM break the strict common-mode factorization (secondary for GW150914-class).

### [2] Detector-specific encoder (shared weights, NORM-FREE)  — **ADOPT / keep the no-norm invariant**
- **Physical quantity**: per-detector amplitude+phase summary `f_d` (the extrinsic prefactor P_d entangled with the common h₀).
- **Why current fails**: it *is* per-detector at the stem, but LayerNorm inside the transformer and pooling degrade amplitude/phase; the "no-normalization on the amplitude path" invariant is only partially honored.
- **Why it helps distance**: `‖f_d‖` carries the per-detector SNR (amplitude), preserved for the ratio features.
- **Cost**: one shared MLP 400→512→128, applied 3×. Tiny.
- **Complexity**: low. **Expected gain**: enabling (feeds §3). **Drawbacks**: none material; must keep it **norm-free** (a LayerNorm here would erase the amplitude that is the distance cue — this is the current invariant worth keeping).

### [3] Explicit relative features (amplitude ratios, complex phase diffs, arrival-time deltas)  — **ADOPT — the core fix**
- **Physical quantity**: `|c_i|/|c_j|`, `arg(⟨c_i,c_j⟩)`, `Δτ_ij` for present pairs → the antenna-ratio + sky determinants of Θ. `⟨c_i,c_j⟩=Σ_f conj(c_i)c_j` is the coherent cross-detector overlap (the exact coherence sufficient statistic given h₀).
- **Why current fails**: the model must *re-derive* these from a pooled 256-vector; measured θ_jn/ra/dec R²≈0.25 shows it does so poorly.
- **Why it helps distance**: these cancel (Mc,d_L) and isolate Θ; with Θ known, `d_L = Mc^{5/6}Θ√I/ρ`.
- **Cost**: a handful of dot products over 200-dim coeffs × 3 pairs — negligible.
- **Complexity**: low–moderate (define the features, mask absent pairs). **Expected gain**: the single largest lever for distance. **Drawbacks**: with H1/L1 nearly co-aligned the +/× (inclination) content is physically weak — expect sky/time/relative-amplitude gains, limited ι gain (physics floor, not fixable in software).

### [4] Cross-detector combination — **REPLACE deep attention with: explicit relatives + LIGHT attention *pooling* only**
- **Recommendation & exact justification for keeping attention**: drop the 3-layer cross-detector *transformer* (over-engineered for 3 tokens; an MLP computes any 3-way relation). **Keep attention only as the set-aggregation POOLING** (a few learned queries over present-detector tokens), for two concrete reasons that an MLP-on-concatenation cannot give: (i) **variable/missing detectors** — real events have {H1,L1}, {H1,L1,V1}, or single-IFO; masked attention pooling is permutation-invariant and handles any present-set, whereas fixed-order concat needs the current white-noise-fill hack; (ii) **SNR-weighted aggregation** — the query learns to down-weight the noisy/absent detector. That is the value with 3 detectors; not "learning cross-detector relations."
- **Cost**: few queries × 3 tokens — trivial. **Complexity**: low. **Expected gain**: robustness (no fill hack) + cleaner combination. **Drawbacks**: none material.

### [5] PSD conditioning — **KEEP asd_bands; REJECT a learned PSD embedding**
- **Physical quantity**: per-detector sensitivity S_d(f) (sets ρ_d for a given d_L).
- **Why current is adequate**: Step 2 measured asd_bands only mildly OOD (Maha 5.3 vs 3.6), not the bottleneck. A learned PSD net adds parameters for a non-driver.
- **Cost/where it enters**: 16-band log(design/measured) per detector, concatenated into the per-detector token [2]; whitening already uses the measured PSD upstream (so PSD enters **twice**: whitening sets the scale, asd_bands tells the model the sensitivity to convert amplitude→physical SNR). **Expected gain**: none beyond current. **Drawback of over-building it**: wasted capacity.

### [6] Pooling: CLS vs attention vs mean — **attention pooling; NEVER mean**
- **Mean pooling is disqualified**: averaging complex/temporal features washes out the phase coherence that is the entire signal for Θ. **Attention/CLS pooling** preserves structure and enables SNR-weighting. But the decisive point (measured) is *what you pool*: pool **after** the explicit relative features exist, else any pooling loses coherence. Pooling type is second-order; do not over-invest.

### [7] Physics-aware latent tokens — **ADOPT per-detector complex overlaps; REJECT scalar coherent/network SNR**
- **Per-detector complex overlaps ⟨c_i,c_j⟩ (magnitude+phase): ADOPT** — the coherence sufficient statistic (§3). **Per-detector SNR ‖c_d‖: ADOPT** (feeds ratios). **Scalar network/coherent SNR ρ_net: REJECT** — it is exactly the degenerate amplitude already encoded (context→SNR R²=0.88); it cannot break Θ and would not move the flat response. **Arrival-time deltas: ADOPT** (cheap sky info).

### [8] SVD vs learned feature extractor — **SVD (fixed or lightly fine-tuned)**
- On the signal manifold the SVD basis is provably near-optimal (retains >99.9% signal power in ~200 modes) and **denoises** (projects out the orthogonal noise subspace, SNR gain ~√(Nf/Nb)). A learned extractor has no advantage on the manifold, needs more data, and risks re-learning the Mc shortcut. Use SVD as the compression; allow light end-to-end fine-tuning of a thin adapter after it. **Drawback**: model-dependence (acceptable, like a template bank).

### [9] GNPE / iterative refinement — **NOT justified by the evidence; encoder first**
- The failure was localized to *representation-level geometry*, and the pipeline is already trigger-centered (t_c ±0.2 s). GNPE addresses t_c/sky *alignment*; its benefit here is DINGO-empirical, not implied by any measurement. Build [1]–[5] first; add GNPE only if validation shows residual geometry loss. It costs 2–5× inference + Gibbs-style training complexity — do not adopt on faith.

---

## 3. The three architectural questions

**Q1. Separate intrinsic (masses,spins) from extrinsic (d_L,sky,ι,ψ) before the context?**
**Soft separation, yes; hard split, no.** The physics factorizes (common-mode h₀ = intrinsic; relative prefactors = extrinsic), which *motivates two feature streams*: an **intrinsic stream** from the common-mode coefficients (average/coherent-combined c_d) and an **extrinsic stream** from the relative features (§3). Concatenate both into one 256-D context feeding one flow. **Do not** build two separate contexts/flows: the posterior has strong intrinsic–extrinsic correlations (d_L–ι, and Mc is measured from the phase independent of geometry) that the joint flow must model; a hard split discards them and adds complexity for uncertain gain. Net: structure the *features* by physics, keep the *posterior* joint.

**Q2. Give the flow masses directly, or remove the shortcut via dataset design?**
**Dataset design — and I explicitly correct my earlier note here.** "Conditioning the flow on masses" is not even well-defined for a flow that *predicts* masses (masses are outputs θ, not conditioning), and giving the context *more* mass emphasis would *entrench* the shortcut if the data correlation persists. The shortcut has two parts: (i) the **type-mixing artifact** (BBH/BNS/NSBH different distance ranges) — WRONG, remove by unifying/overlapping the distance priors + importance-weighting so `p_train(d_L|Mc)` matches the true astrophysical relation; (ii) the **within-BBH Malmquist** (~0.4) — physically REAL, the model *should* keep it. The dataset fix removes (i); the geometry encoder ensures the model also uses amplitude so it does not *over*-rely on (ii). No mass-to-flow plumbing.

**Q3. Better coherence representation than the pooled latent?**
**Yes.** Coherence is not something to hope survives pooling — represent it by its **sufficient statistics**: the set of per-detector complex coefficients {c_d} plus the pairwise complex overlaps {⟨c_i,c_j⟩} (magnitude+phase) and time deltas. The coherent likelihood is a function of exactly these; feed them explicitly to the context MLP instead of compressing 3 detectors into one 256-vector and praying the phase relations survive.

---

## 4. End-to-end architecture (block diagram + tensor shapes)

```
 raw whitened strain                       [B, 3, 16384]  real   (V1 slot may be masked-absent)
   │  rFFT → band-limit [20,1024] Hz
   ▼
 complex FD  d_d(f)                         [B, 3, 4016] complex  → [B, 3, 4016, 2] real
   │  [1] SVD projection  c = Vᴴ d  (V:[4016,200] complex, per-det SHARED, fixed/fine-tuned)   ── PSD enters (whitening) ──┐
   ▼                                                                                                                        │
 per-det complex coeffs                     [B, 3, 200] complex   → [B, 3, 400] real                                        │
   │  [2] per-det MLP (SHARED, NORM-FREE)  400→512→128                                                                       │
   ▼                                                                                                                        │
 per-det features  f_d                      [B, 3, 128]                                                                     │
   │  ⊕ asd_bands [B,3,16]  ⊕ det-id emb [B,3,8]  ⊕ present-mask [B,3,1]        ◄── PSD (sensitivity) + DETECTOR GEOMETRY ──┘
   ▼
 per-det tokens                             [B, 3, 153]
   ├────────────────────────────────►  [3] EXPLICIT relative features (present pairs i<j)
   │                                        oᵢⱼ = Σ_f conj(cᵢ)cⱼ → [|oᵢⱼ|, arg oᵢⱼ],  Δτᵢⱼ,  ‖cᵢ‖, ‖cⱼ‖
   │                                        → [B, 3pairs, 5] → masked-flatten → [B, 15]      ◄── DETECTOR GEOMETRY (coherence)
   ▼
 [4] attention POOLING  (n_q=4 learned queries over PRESENT det tokens, masked; SNR-weighted, set-invariant)
     → [B, 4×153] → [B, 612]
   │  ⊕ relative features [B, 15]
   ▼
 [5] context head:  intrinsic stream (coherent-mean of c_d) ⊕ extrinsic stream (pooled + relatives) → MLP → CONTEXT   [B, 256]
   │  ⊕ rank embedding  [B, 32]                                          ◄── conditioning
   ▼
 [6] NSF flow   p(θ | [context ‖ rank])          θ ∈ [B, 11]   (ParamScaler: log masses/dL, circular wrap ra/φ/ψ)
   ▼
 (optional) [7] GNPE loop: estimate (t_c, sky) → re-time-shift FD → re-project [1] → re-infer   (only if [3]+[4] underperform)
```

**Data flow**: raw→FD→SVD coeffs→per-det features→(pooled ‖ relatives)→context→flow→11-D posterior.
**Conditioning flow**: `context ‖ rank_embed` conditions the flow. `asd_bands` + `det-id` + `present-mask` condition the per-detector tokens.
**Where PSD enters**: (a) whitening upstream of the FFT (sets the amplitude scale); (b) `asd_bands` at the per-detector token (lets the model convert whitened amplitude → physical SNR = the distance-relevant quantity).
**Where geometry enters**: (a) `det-id` embedding (associates each detector with its known location/orientation, so measured relatives map to sky); (b) the explicit relative features `oᵢⱼ, Δτᵢⱼ, ‖c‖` (the measured geometry). Optionally feed the fixed detector position/orientation constants for a stronger antenna-pattern prior.
**How the context is formed**: from the coherence sufficient statistics (per-det features + pairwise overlaps), not a blind pooled latent — with a soft intrinsic/extrinsic split of the *features*, one joint flow.

---

## 5. Keep vs replace (relative to current LeanNPE)

**KEEP**
- Rank embedding (32-D) + rank-conditioning — the overlap mechanism works.
- NSF flow (`flows.py`) — capacity is not the problem.
- `ParamScaler` — deterministic invertible map, circular wrapping.
- **The "no normalization on the amplitude path" invariant** — extend it to the SVD-coeff MLP.
- Measured-PSD whitening preprocessing + `asd_bands` conditioning + quality checks.
- Checkpoint/loader contract.

**REPLACE**
- TD `asinh` conv stem → FD + SVD projection [1].
- 3-layer cross-detector transformer → per-det shared MLP + explicit relatives + light attention *pooling* [2–4].
- The separate log-energy amplitude branch → subsumed (amplitude is native in FD coeff magnitudes).
- The single pooled 256-latent as the coherence carrier → explicit coherence sufficient statistics [3].

**ADD (new)**
- Explicit relative/coherence features [3]; soft intrinsic/extrinsic feature split [5]; present-mask handling for missing detectors [4].

---

## 6. Critical caveats (do not oversell)
- With **2 co-aligned detectors (H1/L1)** inclination/polarization is physically under-determined; expect distance recovery via coherent amplitude+phase+sky+volume-prior (as dynesty does on GW150914), **not** sharp ι. Edge-on distance stays hard even for the exact likelihood.
- SVD basis is model/PSD-dependent; validate transfer to real PSDs.
- The dataset fix (Q2) is **required** alongside the encoder — a better encoder without removing the type-mixing shortcut will still be pulled by the spurious Mc→d_L association.
- Acceptance test (build-time): fixed-mass amplitude sweep → flow median tracks true d_L (slope→1); GW150914 d_L→~440 with truth in CI; SBC/coverage held; end-to-end BBH predicted-vs-true d_L Spearman ↑ from 0.31.
```
