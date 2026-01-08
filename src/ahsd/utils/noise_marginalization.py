"""
Noise-Marginalized Loss

Problem: STEP 4 generates K noise realizations per parameter set θ.
But standard training treats each (θ, noise_k) as independent.

This causes:
- Model learns spurious correlations: distance = f(noise_pattern)
- Posterior shifts, over-coverage, SBC rank collapse

Solution: Group samples by base parameter ID and average loss within each group.
This approximates: E_θ [ E_noise [ -log p(θ | strain) ] ] instead of E_{(θ,noise)} [ ... ]

Usage:
    losses = compute_batch_loss(batch)  # Standard loss computation
    marginalized_loss = marginalize_loss_by_theta(losses, batch_sample_ids)
"""

from typing import List, Dict, Tuple
import torch
from collections import defaultdict


def extract_base_id(sample_id: str) -> str:
    """
    Extract base parameter θ identifier from sample_id.
    
    Rules:
    - Single-signal with noise: "000123_noise4" → "000123" (marginalize over noise)
    - Special multi-sample types: "overlap_001234" → "overlap" (group by type)
    - Special multi-sample types: "pre_merger_001234" → "pre_merger" (group by type)
    - Other special types: keep as-is
    
    Examples:
        "000123_noise0" → "000123"
        "000123_noise4" → "000123"
        "001000_noise10" → "001000"
        "overlap_001234" → "overlap"  (multiple overlaps grouped as one type)
        "overlap_999" → "overlap"
        "pre_merger_001234" → "pre_merger"  (multiple pre_merger grouped as one type)
        "pre_merger_999" → "pre_merger"
        
    Args:
        sample_id: Sample identifier string
        
    Returns:
        Base ID for grouping:
        - For _noiseK samples: the numeric ID before _noiseK
        - For special prefixes (overlap, pre_merger, etc.): the prefix only
        - For other samples: the full ID (treated as unique)
    
    Why this matters:
        If we don't strip special prefix suffixes, multiple overlap/pre_merger samples
        would be treated as different θ, breaking noise marginalization.
        This would re-introduce spurious distance bias.
    """
    # ✅ STEP 4 noise marginalization: strip _noiseK
    if "_noise" in sample_id:
        base = sample_id.rsplit("_noise", 1)[0]
        return base
    
    # ✅ Special multi-sample prefixes: return prefix only (strip numeric suffix)
    # These are handled as a group, not as individual θ
    special_prefixes = ("overlap", "pre_merger")
    for prefix in special_prefixes:
        if sample_id.startswith(prefix + "_"):
            # Strip the prefix and suffix, return prefix only
            # e.g., "overlap_001234" → "overlap"
            return prefix
    
    # ✅ Default: treat as unique θ (no grouping)
    # This includes: glitch samples, custom samples, etc.
    return sample_id


def group_batch_by_theta(
    batch_sample_ids: List[str],
) -> Dict[str, List[Tuple[int, str]]]:
    """
    Group batch samples by their base parameter ID.
    
    Args:
        batch_sample_ids: List of sample_id strings from batch
        
    Returns:
        Dict mapping base_id → [(batch_index, original_sample_id), ...]
        
    Example:
        batch = ["000123_noise0", "000456_noise1", "000123_noise2"]
        groups = {
            "000123": [(0, "000123_noise0"), (2, "000123_noise2")],
            "000456": [(1, "000456_noise1")]
        }
    """
    groups = defaultdict(list)
    for batch_idx, sample_id in enumerate(batch_sample_ids):
        base_id = extract_base_id(sample_id)
        groups[base_id].append((batch_idx, sample_id))
    return groups


# ✅ JAN 7: CRITICAL - Only marginalize likelihood terms, not regularizers
MARGINALIZE_KEYS = {
    "total_loss",       # Overall loss (if it's just sum of likelihoods)
    "flow_loss",        # Flow NLL
    "nll_loss",         # NLL (alternative name)
    "posterior_nll",    # Posterior NLL (alternative name)
}

def marginalize_loss_by_theta(
    loss_dict: Dict[str, torch.Tensor], batch_sample_ids: List[str], verbose: bool = False,
    batch_idx: int = 0, log_frequency: int = 50
) -> Dict[str, torch.Tensor]:
    """
    Marginalize loss over noise realizations within each parameter group.
    
    Only marginalizes likelihood-like terms (flow_loss, nll_loss, total_loss).
    Regularizers (KL loss, context variance, Jacobian penalties, distance priors) are NOT marginalized.
    
    This implements: E_θ [ E_noise [ -log p(θ | strain) ] ]
    
    For each unique θ, average the LIKELIHOOD across its K noise realizations.
    Then average the likelihoods across all unique θ.
    
    Regularizers are averaged across the entire batch (standard way).
    
    Args:
        loss_dict: Dictionary with loss components:
            - "total_loss": [batch_size] tensor (marginalized if K-varying)
            - "flow_loss": [batch_size] tensor (marginalized if K-varying)
            - Other regularizers (KL, physics, uncertainty, etc.): [batch_size] (batch-averaged only)
        batch_sample_ids: [batch_size] list of sample_id strings
        verbose: If True, print grouping information
        
    Returns:
        Dictionary with same structure:
        - Likelihood terms: marginalized then averaged
        - Regularizer terms: batch-averaged (standard way)
        
    Example (STEP 4 data with K=5):
        # Before marginalization (WRONG - entangles distance with noise):
        batch = ["000123_noise0", "000123_noise1", "000123_noise2", 
                 "000456_noise0", "000456_noise1"]
        flow_loss = [2.5, 2.3, 2.4, 3.1, 3.0]  # Each (θ, noise) independent
        mean_loss = 2.66 (all samples equally weighted)
        
        # After marginalization (CORRECT - marginalizes over noise):
        groups = {
            "000123": [2.5, 2.3, 2.4] → mean = 2.4  (averaged over 3 noise realizations)
            "000456": [3.1, 3.0]     → mean = 3.05 (averaged over 2 noise realizations)
        }
        mean_loss = (2.4 + 3.05) / 2 = 2.725 (each θ equally weighted)
        
        This correctly implements: E_θ [ E_noise [ -log p(θ | strain) ] ]
    
    Key assumption:
        - All unique θ should have the SAME K (number of noise realizations)
        - Currently enforced by STEP 4 data generation (K=5 always)
        - If K varies, need weighted averaging (not yet implemented)
    """
    groups = group_batch_by_theta(batch_sample_ids)
    
    marginalized_dict = {}
    
    for key, loss_tensor in loss_dict.items():
        if not isinstance(loss_tensor, torch.Tensor):
            # Skip non-tensor values (e.g., learning_rate, epoch)
            marginalized_dict[key] = loss_tensor
            continue
        
        if loss_tensor.numel() == 1:
            # Already scalar, no marginalization needed
            marginalized_dict[key] = loss_tensor
            continue
        
        # Ensure loss_tensor is 1D [batch_size]
        if loss_tensor.dim() > 1:
            # If 2D or higher, take mean across all but batch dim
            loss_tensor = loss_tensor.view(loss_tensor.shape[0], -1).mean(dim=1)
        
        if loss_tensor.shape[0] != len(batch_sample_ids):
            # Size mismatch, skip marginalization
            marginalized_dict[key] = loss_tensor
            continue
        
        # ✅ CRITICAL: Check if this key should be marginalized
        if key not in MARGINALIZE_KEYS:
            # Regularizer: batch-average (standard way)
            marginalized_dict[key] = loss_tensor.mean()
            continue
        
        # Likelihood: marginalize over noise within each θ
        group_losses = []
        for base_id, indices in groups.items():
            batch_indices = [idx for idx, _ in indices]
            # Average loss for this θ across all its noise realizations
            theta_loss = loss_tensor[batch_indices].mean()
            group_losses.append(theta_loss)
        
        # Average across all unique θ
        marginalized_loss = torch.stack(group_losses).mean()
        marginalized_dict[key] = marginalized_loss
    
    # ✅ Log only every N batches to avoid spam
    if verbose and batch_idx % log_frequency == 0:
        n_unique_theta = len(groups)
        n_total_samples = len(batch_sample_ids)
        avg_noise_k = n_total_samples / n_unique_theta
        print(
            f"  ✅ Marginalization (batch {batch_idx}): {n_unique_theta} unique θ, "
            f"{n_total_samples} total samples (K≈{avg_noise_k:.1f})"
        )
        marginalized_keys = [k for k in loss_dict.keys() if k in MARGINALIZE_KEYS]
        regularizer_keys = [k for k in loss_dict.keys() if k not in MARGINALIZE_KEYS and isinstance(loss_dict[k], torch.Tensor)]
        if marginalized_keys:
            print(f"    Marginalized (likelihood): {marginalized_keys}")
        if regularizer_keys:
            print(f"    Batch-averaged (regularizers): {regularizer_keys}")
    
    return marginalized_dict


def should_marginalize(batch_sample_ids: List[str]) -> bool:
    """
    Check if batch contains STEP 4 data (multiple noise realizations per θ).
    
    Returns True only if batch contains noise-variant samples that can be grouped.
    """
    noise_samples = sum(1 for sid in batch_sample_ids if "_noise" in sid)
    # Only marginalize if we have meaningful groups
    return noise_samples > 0 and len(set(extract_base_id(sid) for sid in batch_sample_ids)) < len(
        batch_sample_ids
    )
