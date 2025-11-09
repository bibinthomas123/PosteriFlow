
## 🧩 1. Dataset fixes — get the data right before the network trains

### **A. Edge-type variance**

* **Problem:** All your `edge_type_id` values are identical (variance = 0).
  That disables the *edge-conditioning pathway* in your graph model—because the embedding for edges sees no difference between them.
* **Fix:** In `convert_to_priority_scenarios()`, assign each detector-pair/overlap context a distinct, stable integer ID.
  Example map:

  ```
  0–2 → single-detector (H1, L1, V1)
  3–5 → pairwise (H1-L1, H1-V1, L1-V1)
  6   → triple overlap HLV
  7–16 → reserved for special/time/morphology bins
  ```

  Those IDs must travel through the dataloader as `edge_type_id` tensors so the `nn.Embedding` in PriorityNet can actually learn conditioning.

### **B. SNR inclusion and normalization**

* **Problem:** Many detections miss `network_snr`; when it’s there, normalization clips small differences (ΔSNR = +2 → Δpred≈0).
* **Fix:**

  * Always populate `network_snr` in detection dicts and in `_detections_to_tensor`.
  * Normalize with a simple bounded scale, not z-score:

    ```python
    snr_scaled = min(network_snr, 35) / 35.0
    ```

    That preserves small variations and avoids compressing high values.
  * Write a quick unit test: increasing SNR by +2 should increase predicted priority ≥ 0.01 for mid-range examples.

### **C. Overlap diversity**

* **Problem:** The model ranks correctly but calibration is narrow—means your training overlaps are too simple.
* **Fix:**

  * Raise 5 + overlaps to **25–35 %** of training data (we already added this patch earlier).
  * Mix event types inside overlaps (e.g., BBH + BNS) with realistic SNR spread.
  * Add a few “decoy” composites: real + weak synthetic signals, to teach the scale boundaries.

### **D. Real-parameter diversity in evaluation**

* Use a robust **GWOSC parser** to pull true parameters for evaluation; don’t let defaults fill with zeros.
* Keep a small static catalog fallback (for CI/tests) so evaluation doesn’t break if the API changes.

---

## ⚙️ 2. PriorityNet fixes — repair learning dynamics

### **A. Output compression**

* **Problem:** Predictions saturate at 0.4 while true labels go to 0.95 → output layer is squashing (sigmoid/tanh).
* **Fix:**

  * Remove the final sigmoid/tanh; use a **linear head** trained on raw 0–1 targets.
  * Or keep the head linear and learn an **affine calibration** post-training:
    (\hat y = a·z + b) minimizing L1 gap on validation.

### **B. Strengthen SNR pathway**

* Add a dedicated SNR embedding:

  ```python
  self.snr_embed = MLP([1,16,32], activation=GELU)
  ```

  concatenate it before the fusion block.
* Add an auxiliary sensitivity loss: encourage ∂ŷ/∂SNR ≈ 0.02 (so a +2 SNR shift raises priority by ≈ 0.04).
  Simple version: `L_snr = MSE((ŷ2-ŷ1)/(SNR2-SNR1), 0.02)`.

### **C. Re-enable edge conditioning**

* Add / verify:

  ```python
  self.edge_embed = nn.Embedding(n_edge_types, edge_dim)
  ```

  and fuse it into node features or attention keys.
* Dropout ≈ 0.1 on the edge embedding prevents the net from latching onto one ID.

### **D. Calibrated loss**

* Keep your ranking loss (e.g., pairwise logistic) but add gentle calibration penalties:
  [
  L = L_\text{rank}
  + λ_1·|mean(ŷ) - mean(y)|
  + λ_2·|max(ŷ) - max(y)|
  ]
  with λ₁ ≈ λ₂ ≈ 0.05.
  This widens predicted spread to match target scale.

### **E. Training stability**

* Use `ReduceLROnPlateau` monitoring Spearman/Kendall τ and calibration gap.
* Initialize final head near the dataset mean to avoid early flatlining:

  ```python
  nn.init.normal_(head.weight, 0, 0.01)
  head.bias.data.fill_(0.2)
  ```

---
