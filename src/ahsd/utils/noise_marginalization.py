"""
Noise-Marginalized Loss

The dataset generates K noise realizations per parameter set θ, but standard
training treats each (θ, noise_k) pair as independent. That lets the model learn
spurious correlations (e.g. distance = f(noise_pattern)), causing posterior
shifts, over-coverage, and SBC rank collapse.

This module groups samples by base parameter ID and averages loss within each
group, approximating E_θ [ E_noise [ -log p(θ | strain) ] ] instead of
E_{(θ,noise)} [ ... ].

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
    - Multi-signal types with noise: "overlap_001234_noise1" → "overlap_001234" (group by type+id)
    - Special sample types (eccentric_mergers, strong_glitches, psd_drift, etc.):
      Extract until _noise: "eccentric_mergers_005688_noise1" → "eccentric_mergers_005688"
    
    Examples:
        "000123_noise0" → "000123"
        "000123_noise4" → "000123"
        "001000_noise10" → "001000"
        "overlap_001234_noise1" → "overlap_001234"
        "pre_merger_002517_noise1" → "pre_merger_002517"
        "eccentric_mergers_005688_noise1" → "eccentric_mergers_005688"
        "strong_glitches_005763_noise0" → "strong_glitches_005763"
        "psd_drift_005808_noise2" → "psd_drift_005808"
        
    Args:
        sample_id: Sample identifier string
        
    Returns:
        Base ID for grouping:
        - Everything before _noise suffix (works for all types)
        - If no _noise, return full ID (unique)

    Why this matters:
        Each noisy variant of the same θ has suffix _noise0, _noise1, ..., _noiseK-1.
        Stripping everything from _noise onward groups them back together,
        regardless of sample type (regular, overlap, special).
    """
    if "_noise" in sample_id:
        base = sample_id.rsplit("_noise", 1)[0]
        return base

    # No noise suffix: treat as a unique θ
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


# Only marginalize likelihood terms, not regularizers
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
        
    Example (K=5 noise realizations per θ):
        batch = ["000123_noise0", "000123_noise1", "000123_noise2",
                 "000456_noise0", "000456_noise1"]
        flow_loss = [2.5, 2.3, 2.4, 3.1, 3.0]  # each (θ, noise) pair independent

        groups = {
            "000123": [2.5, 2.3, 2.4] → mean = 2.4  (averaged over 3 noise realizations)
            "000456": [3.1, 3.0]     → mean = 3.05 (averaged over 2 noise realizations)
        }
        mean_loss = (2.4 + 3.05) / 2 = 2.725 (each θ equally weighted)

        This implements: E_θ [ E_noise [ -log p(θ | strain) ] ]

    Key assumption:
        - All unique θ have the same K (number of noise realizations)
        - If K varies across θ, this would need weighted averaging (not implemented)
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
    
    if verbose and batch_idx % log_frequency == 0:
        import logging as _logging
        _log = _logging.getLogger(__name__)
        n_unique_theta = len(groups)
        n_total_samples = len(batch_sample_ids)
        avg_noise_k = n_total_samples / n_unique_theta
        marginalized_keys = [k for k in loss_dict.keys() if k in MARGINALIZE_KEYS]
        regularizer_keys = [k for k in loss_dict.keys() if k not in MARGINALIZE_KEYS and isinstance(loss_dict[k], torch.Tensor)]
        _log.debug(
            f"Marginalization (batch {batch_idx}): {n_unique_theta} unique θ, "
            f"{n_total_samples} total samples (K≈{avg_noise_k:.1f}) | "
            f"marginalized={marginalized_keys} regularizers={regularizer_keys}"
        )
    
    return marginalized_dict


def should_marginalize(batch_sample_ids: List[str]) -> bool:
    """
    Check if batch contains multiple noise realizations per θ that can be grouped.
    """
    noise_samples = sum(1 for sid in batch_sample_ids if "_noise" in sid)
    # Only marginalize if we have meaningful groups
    return noise_samples > 0 and len(set(extract_base_id(sid) for sid in batch_sample_ids)) < len(
        batch_sample_ids
    )
