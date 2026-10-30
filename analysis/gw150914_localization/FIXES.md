# Fix design for the GW150914 distance failure

Design deliverable (no new experiments). Grounded in the measured failure in `REPORT.md` / `AUDIT_mcdl.md` / `encoder_amplitude_response.*`.

## 0. The failure, stated mathematically

Whitened detector strain of the loudest signal (schematically):

    h_d(f) = (Mc^{5/6} / d_L) · 𝒜_d(ra,dec,ψ,ι) · e^{iΨ(f;Mc,q,χ)} / √S_d(f)
    𝒜_d = F+_d(ra,dec,ψ)·(1+cos²ι)/2  −  i·F×_d(ra,dec,ψ)·cos ι

Network optimal SNR:

    ρ² = Σ_d 4∫|h_d|²/S_d df = (Mc^{5/6}/d_L)² · Σ_d|𝒜_d|² · I(Mc,q)

**d_L enters only through the overall scale 1/d_L, degenerate with the geometry factor Θ ≡ (Σ_d|𝒜_d|²)^{1/2}.** From amplitude alone `d_L = Mc^{5/6}·Θ·√I / ρ`, and Θ ranges ~2.5× (face-on |𝒜|~1 vs edge-on |𝒜|~F+/2). **Distance is recoverable only if the model resolves Θ** — i.e. (ι, sky, ψ). What resolves Θ (what dynesty/DINGO use): the *relative* complex per-detector amplitudes h_H1/h_L1 (magnitude **and phase**), the arrival-time delays, and the +/× polarization content.

Mapping the measurements onto this:
- Encoder pins the overall amplitude ρ: context→d_L R²=**0.81** at fixed Θ (`encoder_amplitude_response`). ✔
- Encoder pins Θ's ingredients weakly: θ_jn/ra/dec R²≈**0.25**. ✘ ← bottleneck.
- With Θ unresolved, the NLL-optimal `p(d_L|c)` integrates over Θ's 2.5× range ⇒ `E[d_L|c]` ≈ insensitive to ρ (flat) and regresses to the chirp-mass-conditional mean.
- The dataset supplies a spurious `p_train(d_L|Mc)` slope (pooled 0.81 = type mixing; within-BBH ~0.4 = Malmquist), so NLL reads d_L off Mc.

**Two roots.** (R1, primary) the single 256-D context under-encodes Θ, so amplitude is uninvertible to d_L — this is why the flow is flat even in *single-domain design noise*. (R2) the dataset offers a chirp-mass distance shortcut. Any real fix must **break R1 (supply Θ)** and **remove R2 (kill the Mc crutch)**; R2-removal without R1 only widens the posterior toward the prior.

---

## Tier 1 — Training / data

### 1a. Paired-domain training (same source, K PSD/noise domains, identical labels)
- **Why it helps**: enforces d_L invariance to noise domain, so the model learns d_L from *PSD-normalized physical* SNR, not raw whitened amplitude.
- **Why current fails (math)**: RemixDataset recolors a *different* source each epoch, so no gradient enforces `E[d_L|domainA]=E[d_L|domainB]` for one source; the model only sees identical d_L labels at 3× different amplitudes across the batch ⇒ learns amplitude is a noisy d_L cue.
- **Expected improvement**: fixes the design↔real amplitude offset (helps the real-noise component). **But the flat response is present in single-domain design noise (Step 4b), so this alone will NOT make the flow track amplitude.** Modest on GW150914's median.
- **Cost**: low–moderate (K× recolor + a consistency loss; uses the existing bank). **Complexity**: low. **Risks**: over-weighting the consistency loss washes out legitimate PSD effects; keep asd_bands active.
- **Priority: Medium** (hygiene, not the decisive lever).

### 1b. Remove the Mc→distance shortcut in detected BBH
- **Why it helps**: kills R2 — per-type distance ranges (BBH[50,2000]/BNS[10,300]/NSBH[20,800]) make the pooled 0.81; the min_snr gate adds ~0.4 within BBH.
- **Why current fails (math)**: `∂E_train[d_L|Mc]/∂Mc > 0`, so the NLL-optimal decoder sets `E[d_L|c] ≈ E_train[d_L|Mc(c)]` — Mc-driven, amplitude-blind.
- **How**: (i) unify/overlap distance ranges across types; (ii) importance-weight so `p_train(d_L|Mc) ≈ p_train(d_L)`; (iii) lower/remove the SNR gate to retain the far tail at each Mc; (iv) condition the flow on component masses so Mc carries no spurious distance info.
- **Expected improvement**: removes the Mc systematic and restores coverage — but **alone it widens the posterior toward the prior** (without R1 the model still can't use amplitude). Pair with Tier 2.
- **Cost**: dataset regeneration (~2 h/50k). **Complexity**: low code, full regen+retrain. **Risks**: naive range-unification fills the set with undetectable BNS@2000 Mpc — prefer reweighting to a hard unify.
- **Priority: High** (confirmed spurious driver; must pair with Fix 2).

### 1c. "Other data redesign to force amplitude use"
- **Honest limit**: no pure-data change *forces* amplitude use, because amplitude is physically degenerate with Θ. Data can remove the crutch (1b) and fix domains (1a); the model can only *use* amplitude once it can measure Θ — a **representation** change. Tier 1 is necessary-but-insufficient.

---

## Tier 2 — Representation (the decisive fix)

### 2a. Coalescence-aligned, frequency-domain, complex multi-detector input (break the Θ degeneracy)
- **Why it helps**: Θ is resolved by the *relative* complex amplitudes/phases across detectors + time delays. The current front-end (asinh time-domain conv → single pooled 256-vector) discards inter-detector **phase**. Feeding whitened **FD complex** strain per detector (real/imag), aligned to t_c, preserves h_H1(f)/h_L1(f) magnitude+phase ⇒ the network can infer (ra,dec,ψ,ι) and split amplitude into d_L vs Θ.
- **Why current fails (math)**: `I(c; Θ)` is low (θ_jn R²=0.25); given c that fixes ρ but not Θ, `Var(d_L|c)` spans Θ's range and `∂E[d_L|c]/∂ρ→0`. Exposing coherent FD data raises `I(c;Θ)`, shrinks `Var(d_L|c)`, restores amplitude sensitivity.
- **Expected improvement**: **HIGH** — this is exactly what lets dynesty/DINGO recover GW150914 d_L=440 with truth in CI. Target: bias 1.6×→~1.0–1.2×, CI covers truth, end-to-end BBH predicted-vs-true dL Spearman 0.31→dynesty-like.
- **Cost**: moderate retrain (FD transform cheap, re-whiten existing signals; no regen needed for the representation). **Complexity**: moderate (new encoder front-end + t_c alignment). **Risks**: t_c estimation (chicken-egg → Fix 3); with 2 co-aligned detectors GW150914 inclination is intrinsically limited, but amplitude+phase+volume-prior still succeed (dynesty proves it).
- **Priority: HIGH — the single most important fix.**

### 2b. Physics-derived tokens — which to inject, mathematically
- **Network SNR ρ_net (scalar): DO NOT add.** `ρ_net = ||h_w||` is *exactly* the amplitude the context already encodes at R²=0.81 — the degenerate quantity. Adding it is redundant and cannot break Θ (it would not change the flat response). **This is the key negative: SNR tokens are not the fix.**
- **Per-detector complex SNR (ρ_d, phase_d, t_peak,d): DO add.** Since `ρ_d = (Mc^{5/6}/d_L)·𝒜_d·√I`, cross-detector ratios `|ρ_H1|/|ρ_L1|` cancel (Mc,d_L) and isolate the geometry `|𝒜_H1|/|𝒜_L1|`; phase/time differences give the sky delay. A compact, physics-exact summary of what 2a exposes; directly raises `I(c;Θ)`.
- **Antenna patterns F+,F×: cannot be data** (functions of the unknown ra,dec,ψ) — provide detector-geometry constants and let the net learn the map from ρ_d-ratios→sky.
- **Inclination quantities: cannot be injected** (they are the target) — the representation must *expose* the +/× content (via 2a).
- **PSD embedding**: keep asd_bands (measured mildly-OOD, not primary); optional learned log-PSD embedding, low priority.
- **Priority: High for per-detector complex SNR; explicitly reject scalar network-SNR tokens.**

---

## Tier 3 — Architecture (only what addresses the confirmed failure)

### 3. GNPE / iterative geometry-conditioned inference (DINGO-style)
- **Why**: the degeneracy-breaking phase info is usable only once the data are aligned to the unknown t_c/sky. GNPE iterates estimate-(t_c,sky)→align→re-infer using the exact time-translation symmetry — the proven mechanism that makes amortized NPE match dynesty on GW150914.
- **Why current fails (math)**: one un-aligned pass smears the coherent phase over the t_c uncertainty (~ms × kHz = many radians), destroying the inter-detector phase that resolves Θ.
- **Expected improvement**: HIGH (the DINGO result). **Cost**: high (2–5× passes, Gibbs training). **Complexity**: high. **Risks**: convergence/training complexity; may be overkill if 2a suffices.
- **Priority: Medium** (endgame; do it if 2a underperforms).

**Explicitly not recommended** (measured dead / generic): bigger flow (capacity ruled out), aux prediction heads (measured neutral), mc_oversample (rejected), higher-order-mode waveforms (mode-SNR<1 for GW150914-class), PSD-cond tweaks (asd_bands fine), any "scale it up."

---

## Immediate no-retrain stopgap
Residual **bias-map recalibration** `d_L ← g(d̂_L; Mc, ρ, asd_bands)` fit on validation truth: pulls GW150914's median toward 440 and widens the CI to restore coverage today. Does not fix the degeneracy (posterior stays broad); ship as a safety net while 2a is built. Prefer over full importance reweighting (measured to fail on real events).

---

## If I rebuilt LeanNPE from scratch today (physics-informed, not bigger)

Essentially **DINGO adapted to the rank-conditioned overlap setting**:

1. **Input**: per-detector whitened **complex FD** strain (real/imag), t_c-aligned, [20,1024] Hz; whiten by measured PSD; carry a per-detector log-PSD embedding (subsumes asd_bands).
2. **Physics compression**: per-detector **reduced-order (SVD / reduced-basis)** projection of the FD data — few coefficients, keep amplitude+phase DoF, **no premature pooling** to one vector.
3. **Geometry-aware embedding**: cross-detector attention over the per-detector coefficients + per-detector (ρ, phase, t_peak) tokens, so the *ratios/phase-differences* that encode (ra,dec,ψ,ι) are first-class; keep the rank embedding for overlaps.
4. **Flow**: current neural spline flow capacity is fine; condition explicitly on component masses so Mc carries no spurious distance signal.
5. **GNPE** iterative refinement on (t_c, sky) — the mechanism that breaks the distance–inclination degeneracy in one amortized system.
6. **Data**: unified/overlapping distance priors + importance weighting so `p(d_L|Mc)` is flat (no shortcut); paired multi-domain injections with a domain-consistency loss; physically-motivated but reweighted detection floor. Pure NLL.
7. **Serving**: keep SBC/coverage gates + a residual Mc/ρ bias-map safety net.

**One-line rationale**: the investigation shows amplitude reaches the latent but is uninvertible to distance because the representation discards the coherent inter-detector phase that fixes the geometry, and the dataset offers a chirp-mass shortcut. The redesign restores the physics (FD-coherent, t_c-aligned, GNPE) and removes the shortcut (unified priors + explicit mass conditioning) — exactly the configuration under which the exact likelihood (dynesty) already recovers GW150914.

**What to do first**: (2a) FD-coherent representation → (1b) kill the Mc shortcut → measure; add (3) GNPE only if needed. (1a) and the bias-map are supporting/stopgap.
