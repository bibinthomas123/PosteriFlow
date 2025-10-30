Dataset fixes
Populate edge_type_id with variance

Problem: edge_type_id variance is 0, so the edge-conditioning path is bypassed.​

Fix: in convert_to_priority_scenarios, assign a stable integer per scenario encoding detector-pair and overlap context.

Example scheme:

0–2: single-detector H1/L1/V1

3–5: pairwise overlaps (H1–L1, H1–V1, L1–V1)

6: HLV triple overlap

7–16: reserved for special overlap types (e.g., time proximity bins, morphology class)

Ensure these IDs flow into PriorityNetDataset items as edge_type_id and are tensors in the dataloader.

Ensure SNR is present and scaled consistently

Problem: SNR deltas of +2 yield Δpred≈0, implying weak coupling or normalization clipping.​

Fix:

Always include network_snr in detection dicts and in _detections_to_tensor.

Normalize with a calibrated transform that preserves small deltas:

Suggested: snr_scaled = min(snr, 35) / 35.0, do not z-score unless you trained that way.

Validate with a unit test: +2 SNR should increase priority by ≥ 0.01 on a mid-range sample.

Expand overlap diversity for training

Problem: model excels at rank but calibration is narrow, often a sign of limited headroom in diverse overlaps.​

Fix:

Increase the proportion of 5+ overlaps in training to 25–35%.

Mix event types in overlaps (BBH+BNS, NSBH+BBH) with controlled SNR spreads.

Add “decoy-injection” samples (true + weaker synthetic) to bind the scale.

Preserve real-parameter diversity in eval

Implement the robust GWOSC parser in your evaluator to avoid defaulting params; keep a curated static catalog fallback for CI stability.​

PriorityNet fixes
Remove or relax output squashing

Problem: pred max ≈ 0.40 vs target max ≈ 0.95 (gap 0.55) indicates output compression.​

Fix:

If the priority head ends with sigmoid/tanh, remove it and train on raw targets scaled to .

Alternatively, keep linear output and learn an affine calibration at inference: ŷ = a·z + b, where a,b learned by minimizing L1 on validation.

Strengthen SNR pathway

Add a dedicated SNR embedding branch feeding into the fusion block:

snr_embed = MLP([1→16→32], GELU), concatenated before the final head.

Add a small auxiliary loss: L_snr = MSE(∂ŷ/∂snr, c) around a target sensitivity c≈0.02 for mid SNR range, or contrastive margin where higher SNR should rank higher holding others fixed.

Re-enable edge conditioning

Ensure the edge embedding table is used:

edge_embed = nn.Embedding(n_edge_types, edge_dim)

Fuse edge_embed into node features via addition or cross-attention key/value bias.

Add dropout ≈0.1 to prevent overfitting to a single ID.

Calibrated training objective

Keep your ranking loss for order, add a mild calibration term to widen spread:

L = L_rank + λ1·|mean(ŷ) − mean(y)| + λ2·|max(ŷ) − max(y)| with λ1≈0.05, λ2≈0.05.

Use ReduceLROnPlateau on validation Kendall/Spearman and calibration gap to stabilize late training.

Head initialization

Initialize the final linear layer with small weights and bias near the dataset mean priority to avoid early saturation:

nn.init.normal_(head.weight, 0, 0.01); head.bias.data.fill_(0.2)

Verification checklist after fixes
Edge variance

Log edge_type_id variance > 0.5 on a 500-sample validation slice.​

SNR sensitivity

Unit test: +2 SNR delta yields Δpred ≥ 0.01 for a mid-range sample.​

Correlate prediction with SNR over real events: Spearman(ŷ, SNR) ≥ 0.4 in BBH-only subset.

Calibration

Validation max gap: max(true) − max(pred) ≤ 0.18.​

Reliability curve decile MAE ≤ 0.03.

Overlap-5+ bucket

τ ≥ 0.56, Spearman ≥ 0.72 on 5+ overlaps (target you already approach).​

Real events

On the curated 16 GWTC events, std(pred) ≥ 0.03, range widens with calibration fix; decoys still rank lower.​