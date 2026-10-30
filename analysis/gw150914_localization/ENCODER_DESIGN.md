# Next-generation encoder for LeanNPE — critical design

Design deliverable. First I correct what the investigation actually supports; then evaluate each candidate component; then give the architecture.

## Part 0 — Where the investigation is over-interpreted (read first)

**0.1 "The flow discards usable distance information" — OVER-INTERPRETED.**
The evidence: fixed heavy BBH, fixed geometry, sweep only d_L → context→d_L ridge R²=0.81, flow output flat. But (a) the probe separated only **8 discrete amplitude levels** (24 replicas each) — an easy classification, so R²=0.81 mainly re-confirms "the encoder encodes amplitude," which we already had (context→SNR R²=0.88). (b) More importantly, at fixed geometry amplitude = d_L, but **in training, geometry Θ varies, so amplitude does NOT determine d_L**. A near-flat `E[d_L | ρ, no-Θ]` is close to the *correct* Θ-marginalized Bayesian posterior — not a flow "throwing away" usable information. The fixable deficiency is the **missing geometry Θ in the context** (an encoder problem), not a decoder that discards signal. Corrected statement: *the encoder supplies amplitude but not geometry; the flow behaves ~correctly given that.* (Secondary: the flow's Mc-tracking 0.68 > the data's ~0.4 does show *some* genuine over-reliance on the dataset Mc-shortcut — real but not the dominant term.) This redirects the fix to the encoder, which is the right place — but the "encoder-good/flow-bad" framing was wrong.

**0.2 θ_jn R²=0.25 is partly a PHYSICS floor, not purely an encoder flaw.**
Inclination from a single 4 s, 2-nearly-co-aligned-detector (H1/L1) event is intrinsically weak — even dynesty returns broad θ_jn for most events. Expect the redesign to recover **distance via coherent amplitude+phase+sky**, NOT via resolving ι. Do not expect θ_jn to become sharp.

**0.3 "Dynesty recovers GW150914 ⇒ the NPE must be able to" — true, but GW150914 is a FAVORABLE case.**
Dynesty gets d_L=440 largely because GW150914 is **near face-on** (amplitude near-maximal, least-degenerate inclination) plus the volume prior — not because it resolves inclination. For edge-on sources even dynesty is broad/biased ~2× (the anchors). So the realistic target is *"match dynesty on favorable-geometry events,"* not "solve distance universally."

**0.4 FD vs TD is NOT an information difference.**
TD↔FD is a unitary FFT; neither has more information. The current TD encoder's failure is its **specific design** (asinh compression + attention-pool to one vector + no explicit geometry features + no t_c alignment), not "time domain is lossy." Any claim that FD "recovers information TD cannot" is wrong in principle. FD's value is **inductive bias + efficiency**, argued per-component below.

**0.5 GNPE is NOT justified by this investigation's evidence** (details in §2.9).

With those corrections: the encoder must **expose the geometry-carrying quantities** (relative inter-detector amplitude, phase, arrival-time) so the flow can break the d_L–Θ degeneracy. That is the whole design.

---

## Part 1 — The physics the encoder must preserve

Per detector, whitened: `d_d(f) = (Mc^{5/6}/d_L)·𝒜_d(ra,dec,ψ,ι)·e^{iΨ(f)}·e^{2πif·τ_d}/√S_d(f) + n_d(f)`, with `𝒜_d = F+_d(1+cos²ι)/2 − iF×_d cos ι` and τ_d the arrival time.

- **Overall amplitude** `Mc^{5/6}/d_L·|𝒜_d|` — the degenerate scale. Encoder already captures this (✔).
- **Relative amplitudes** `|d_H1|/|d_L1| = |𝒜_H1|/|𝒜_L1|` — cancels (Mc, d_L), isolates antenna/inclination geometry.
- **Relative phase** `arg(d_H1/d_L1)` and **time delay** `τ_H1−τ_L1` — constrain sky (ra, dec) and hence F+_d, F×_d.
- **+/× content** (needs mis-aligned detectors or higher modes) — constrains ι. For GW150914 (2 co-aligned detectors) this is a physics floor.
- **Intrinsic phase Ψ(f;Mc,q,χ)** — masses/spins (already well captured; Mc R²~0.7).

Distance is `d_L = Mc^{5/6}·Θ·√I/ρ`; recovering it needs **Θ from the relative quantities above**. The encoder must keep them **explicit and phase-coherent**.

---

## Part 2 — Component-by-component evaluation

### 2.1 Complex frequency-domain input vs time-domain — **ADOPT FD (inductive bias, not information)**
- *Physical info*: the whitened Gaussian likelihood is **diagonal in frequency**, the waveform is smooth in FD (few effective DoF), and inter-detector **phase** is an explicit per-bin quantity.
- *Why current can't*: the TD stem applies `asinh` (a nonlinear amplitude compression that entangles amplitude with morphology) and real-valued convs, then pools — relative phase is only implicit and is degraded by the attention pooling.
- *Why it reduces the degeneracy*: explicit complex per-bin values make `arg(d_H1/d_L1)` and `|d_H1|/|d_L1|` directly available → Θ.
- *Evidence*: θ_jn/ra/dec R²≈0.25 (geometry poorly captured) while amplitude R²=0.81 (scale well captured) ⇒ what's missing is the *phase-coherent geometry*, which FD exposes. **Caveat (0.4): this is convenience, not more information.**

### 2.2 Detector-specific encoders vs early fusion — **KEEP per-detector (shared weights), combine late**
- *Physical info*: geometry lives in **relative** per-detector quantities; you must keep per-detector amplitude/phase separable.
- *Why current can't*: the current encoder *already* uses a shared per-detector stem + transformer fusion with detector embeddings — so it is not "too-early fusion." Its problem is that fusion→pool discards the relative phase, not that it fuses early.
- *Why it reduces the degeneracy*: late combination on clean per-detector complex summaries preserves the ratios/phase-diffs.
- *Evidence*: same as 2.1. Verdict: retain the per-detector-then-combine structure; change *what* is combined and *how* (below).

### 2.3 Cross-detector attention — **REJECT (heavy) / keep it trivial**
- With **3 detectors**, "attention over 3 tokens" is over-engineering. The pairwise relations you want (ratios, phase/time differences) are cheaply computed **explicitly** (§2.4). A small MLP on `[f_H1, f_L1, f_V1, pairwise-features]` is sufficient and cheaper. Reserve attention for the *within-detector* sequence/bins, not across the 3 detectors.
- *Evidence*: no measurement suggests the failure is an inability to attend across detectors; it's the absence of the relative features themselves.

### 2.4 Explicit relative phase / arrival-time / polarization — **ADOPT (this is the core fix)**
- *Physical info*: `|d_H1|/|d_L1|`, `arg(d_H1/d_L1)`, `τ_H1−τ_L1` = the sky + antenna-pattern determinants of Θ.
- *Why current can't*: the network must *learn* to compute these from a pooled 256-vector; θ_jn R²=0.25 shows it does so poorly.
- *Why it reduces the degeneracy*: these quantities cancel (Mc, d_L) and expose Θ directly — mathematically the exact information that turns `ρ` into `d_L`.
- *Evidence*: the single strongest lever. **Honest limit**: with H1/L1 nearly co-aligned, the +/× (polarization/inclination) content is physically weak — expect sky/time/relative-amplitude gains, limited inclination gain (0.2).

### 2.5 Learned PSD embedding vs asd_bands — **KEEP asd_bands; REJECT a fancy PSD net**
- *Physical info*: PSD sets the amplitude normalization ρ_d.
- *Why current is fine*: **Step 2 measured asd_bands is only mildly OOD (Maha 5.3 vs 3.6) and NOT the primary driver.** A learned PSD embedding adds parameters/complexity for a non-bottleneck. Keep the 16-band log-ratio conditioning. Priority: Low. (Reject as not addressing the confirmed failure.)

### 2.6 CLS-token vs global-average pooling — **either is fine; the point is pool AFTER relative features**
- *Physical info*: pooling must not average away phase. **Global average pooling is harmful** (it washes out the phase-coherent structure). CLS/attention pooling (current design) is acceptable.
- *Why current can't*: not the pooling *type* — it's that pooling happens **before** the relative geometry features are computed. Fix: compute §2.4 features first, then pool. Pooling choice is second-order.
- *Evidence*: no measurement isolates pooling as the failure; do not over-invest here.

### 2.7 Physics-inspired latent tokens — **ADOPT per-detector complex; REJECT scalar coherent-SNR**
- **Per-detector complex matched-filter / complex SVD coefficients: ADOPT.** They carry amplitude+phase per detector; ratios give geometry. (Explicit MF needs a template bank — prefer SVD coeffs of the *data*, which carry the same info without template management.)
- **Coherent / network SNR scalar: REJECT.** `ρ_net=||d_w||` is *exactly the degenerate amplitude* the context already encodes at R²=0.81. Adding it cannot break Θ and will not change the flat response. (This is the key negative from the fixes analysis.)
- **Arrival-time delays: ADOPT as a cheap token** (feeds sky). **Polarization features: physics-limited** for 2 co-aligned detectors.

### 2.8 Reduced-order / SVD waveform representation — **ADOPT (efficiency + denoising + phase-preserving)**
- *Physical info*: the whitened signal lives on a low-dim manifold (~11 params → ~O(100) SVD coeffs/detector capture it). Projecting onto an SVD basis V (from a waveform bank) compresses 8193 FD bins → ~200 complex coeffs, **denoises** (projects onto the signal subspace), and **preserves amplitude+phase**.
- *Why current can't*: the TD conv stem must learn its own compression from noisy 16384-sample series; SVD gives a physics-optimal basis for free.
- *Why it reduces the degeneracy*: cleaner per-detector complex summaries → cleaner relative features (§2.4).
- *Evidence*: efficiency/robustness argument (DINGO uses exactly this); consistent with "info is present but poorly extracted."
- *Cost note*: introduces waveform-model dependence via the basis (acceptable, like a template bank).

### 2.9 GNPE / iterative refinement — **NOT justified by THIS evidence; try the encoder first**
- *What GNPE does*: iteratively estimate (t_c, sky) → align data → re-infer, using time-translation equivariance to sharpen phase coherence.
- *Is it justified here?* My investigation localized the failure to **missing geometry in the representation**, and the pipeline is already **trigger-centered** (t_c within ±0.2 s), so gross t_c-misalignment is not the demonstrated problem. GNPE's benefit is empirical (DINGO), **not implied by my measurements.**
- *Verdict*: **build the coherent encoder (§2.1/2.4/2.8) first; it is likely sufficient for GW150914-class events. Add GNPE only if the coherent encoder underperforms on validation.** Do not adopt it on faith — it is 2–5× inference cost and a large training complication.

### Rejected as unnecessary complexity
Heavy cross-detector attention (3 detectors), learned PSD embedding (non-bottleneck), scalar SNR tokens (degenerate), explicit matched-filter template bank (SVD suffices), GNPE up-front (unproven here), and any capacity increase (flow capacity ruled out).

---

## Part 3 — The architecture I would build

Per-detector processing with **shared weights**; geometry made **explicit** before pooling; flow unchanged.

```
 raw strain d∈{H1,L1,V1}
     │  whiten by measured PSD (per det);  → complex FD  d_d(f), f∈[20,1024]Hz
     ▼
[1] SVD / reduced-basis projection  (basis V from a waveform bank, shared)
     → ~150–250 COMPLEX coeffs per detector        · compact · denoised · phase-preserving
     ▼
[2] per-detector embedding MLP (SHARED weights) on (Re,Im) coeffs
     ⊕ asd_bands_d (16)  ⊕ detector-id                → f_d  (per-det amplitude+phase summary)
     │
     ├───────────────┬────────────────────────────┐
     ▼               ▼                             ▼
  {f_H1,f_L1,f_V1}   [3] EXPLICIT relative features:            measured Δτ (H1-L1, H1-V1)
                     log‖f_d‖ ratios, arg(f_d/f_d') phase diffs   (arrival-time delays)
     └───────────────┴────────────────────────────┘
     ▼
[4] geometry combiner  (concat → small MLP; NOT heavy attention for 3 dets)
     → resolves sky/antenna ⇒ Θ ⇒ breaks d_L–Θ degeneracy
     ▼
[5] context head → 256-D context   ⊕ rank embedding (keep overlap conditioning)
     ▼
[6] NSF flow   p(θ₁₁ | context, rank)      · condition on component masses so Mc≠distance proxy
     ▼
 (optional) [7] GNPE loop: re-center on estimated (t_c, sky), re-project [1], re-infer
             — ONLY if [3]+[4] underperform on validation
```

**Why each stage exists**
- **[1] SVD FD**: physics-optimal, phase-preserving compression; denoises onto the signal manifold; makes the network's job the *inference*, not the *feature discovery*. (Fixes: TD stem learning compression from noise.)
- **[2] shared per-detector embedding**: keeps per-detector amplitude+phase separable (geometry lives in the *relative* terms). (Fixes: premature loss of per-detector structure.)
- **[3] explicit relative features + [4] combiner**: the crux — hands the network the ratio/phase-diff/time-delay quantities that cancel (Mc, d_L) and pin Θ, instead of hoping a pooled vector re-derives them (θ_jn R²=0.25 shows it doesn't). (Fixes: the confirmed degeneracy-breaking gap.)
- **[5] context + rank**: unchanged interface; keeps the working overlap/rank conditioning.
- **[6] flow + mass conditioning**: capacity is fine; explicit mass conditioning removes the residual Mc→distance shortcut (dataset audit).
- **[7] GNPE**: contingency only — evidence doesn't compel it.

**Efficiency**: [1] shrinks the input ~40× (8193→~200/det); the rest is small MLPs on 3 detectors → cheaper than the current transformer stack, and still one forward pass (unless [7] is enabled).

**What this is expected to buy**: on favorable-geometry events (GW150914, near face-on) — dynesty-like distance (bias 1.6×→~1.0–1.2×, truth in CI), because it exposes exactly the coherent amplitude+phase the exact likelihood uses. It will **not** make inclination sharp for 2-co-aligned-detector events (physics floor, §0.2), and edge-on distance will remain hard (as it is for dynesty).

**Validation bar** (no new experiments proposed here — this is the acceptance test for the build): the fixed-mass amplitude sweep must show the flow median tracking true d_L (slope→1); GW150914 d_L→~440 with truth in CI; SBC/coverage held; end-to-end BBH predicted-vs-true d_L Spearman ↑ from 0.31.
