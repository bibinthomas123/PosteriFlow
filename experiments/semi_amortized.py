"""Context refinement toolkit for LeanNPE (research prototype) -- the
building blocks for OFFLINE teacher-distillation training.

Motivated by analysis/attractor_basin/REPORT.md's 2026-07-22 oracle-context
addenda: the flow can recover accurate q when given a directly-optimized
("oracle") context (flow_oracle_context_test.py), and a manifold-regularized
sweep (oracle_manifold_tradeoff.py) showed good contexts exist close to the
encoder's own output distribution (Mahalanobis distance ~3-5, comparable to
a typical real context's own distance from the population). That combination
means this is primarily an optimization/training-signal problem, not a
representational-capacity one -- exactly the setting Semi-Amortized VAEs
(Kim et al. 2018, https://arxiv.org/abs/1802.02550) and iterative amortized
inference (Marino et al. 2018) were designed for, adapted here from VAEs to
a conditional normalizing flow.

**2026-07-23: the LIVE/online version of this (a `SemiAmortizedTrainer` that
ran ContextRefiner inside every training batch) has been REMOVED.** It
worked mechanically (gradient routing was verified exactly correct) but was
expensive (an extra K-step inner optimization loop every batch) and, more
importantly, unsafe: refinement targets were computed against a flow that
was SIMULTANEOUSLY being updated by the same training step, creating a
feedback loop where each epoch's targets drifted further from anything
reasonable -- a live 40-epoch run diverged into a total collapse (spurious
railing climbing 0.66->0.72->0.82, Mahalanobis distance climbing epoch over
epoch instead of converging) before a first bug (wrong-units manifold
penalty) was even fully fixed. The tools below (`ContextRefiner`,
`ManifoldStats`, `DistillationLoss`, `select_hard_examples`) are unchanged
and still correct -- they're now used OFFLINE instead, by
`experiments/precompute_distill_targets.py`: refine a batch of hard training
examples ONCE against a frozen checkpoint snapshot, cache the results to
disk, then train against that FIXED cache with an ordinary cheap
lookup-and-compare loss (train_lean_npe.py's `--offline_distill_cache`).
Freezing the reference model removes the feedback loop entirely -- nothing
the cached targets depend on can change during the run they're used in --
and doing the expensive refinement once (or periodically, on a fresh
snapshot) instead of every batch removes nearly all of the runtime cost too.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Hard-example selection
# --------------------------------------------------------------------------- #

def select_hard_examples(
    params_phys: torch.Tensor,          # [B, max_n, 11] physical, rank-sorted
    per_example_nll: Optional[torch.Tensor],   # [B] or None
    basin_mask: Optional[torch.Tensor],        # [B] bool or None (True = known basin-cell event)
    fraction: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return a LongTensor of batch indices to refine, sized
    round(fraction * B), chosen by priority: basin failures > low q >
    high posterior error > random fallback. Each example appears at most
    once; earlier tiers are filled first, later tiers only fill remaining
    budget with examples not already selected.

    `basin_mask`/`per_example_nll` are optional: when either is unavailable
    (e.g. a RemixDataset built without return_event_idx, or a call site
    that hasn't computed per-example NLL yet) that tier is simply skipped,
    degrading gracefully to whatever cues ARE available plus random
    fallback -- never an error.
    """
    B = params_phys.shape[0]
    n_select = int(round(fraction * B))
    if n_select <= 0:
        return torch.empty(0, dtype=torch.long, device=params_phys.device)
    if n_select >= B:
        return torch.arange(B, dtype=torch.long, device=params_phys.device)

    device = params_phys.device
    selected: List[int] = []
    selected_set = set()

    def add(idx_tensor: torch.Tensor):
        for idx in idx_tensor.tolist():
            if len(selected) >= n_select:
                return
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)

    # Tier 1: known basin-cell (geometry-collapse) failures.
    if basin_mask is not None:
        basin_idx = torch.nonzero(basin_mask, as_tuple=True)[0]
        add(basin_idx)

    # Tier 2: low true mass ratio (q close to 0, i.e. very unequal masses --
    # exactly the population the (30,30)/q->1 attractor mispredicts).
    if len(selected) < n_select:
        m1 = params_phys[:, 0, 0]
        m2 = params_phys[:, 0, 1]
        q = torch.minimum(m1, m2) / torch.maximum(m1, m2).clamp_min(1e-6)
        order = torch.argsort(q)  # ascending: lowest q (hardest) first
        add(order)

    # Tier 3: highest current per-example NLL (worst-fit under the live model).
    if len(selected) < n_select and per_example_nll is not None:
        order = torch.argsort(per_example_nll, descending=True)
        add(order)

    # Tier 4: random fallback to fill any remaining budget.
    if len(selected) < n_select:
        remaining = [i for i in range(B) if i not in selected_set]
        perm = torch.randperm(len(remaining), generator=generator).tolist()
        for j in perm:
            if len(selected) >= n_select:
                break
            selected.append(remaining[j])
            selected_set.add(remaining[j])

    return torch.tensor(selected, dtype=torch.long, device=device)


# --------------------------------------------------------------------------- #
# Context refinement (the inner loop)
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class RefinementResult:
    context_star: torch.Tensor          # [n, context_dim], detached leaf tensor
    steps_taken: torch.Tensor           # [n] long, actual iterations run per example
    nll_initial: torch.Tensor           # [n]
    nll_final: torch.Tensor             # [n]
    q_before: torch.Tensor              # [n] median-free point estimate (see note in code)
    q_after: torch.Tensor               # [n]
    refine_distance: torch.Tensor       # [n] ||context_star - context0||
    cosine_sim: torch.Tensor            # [n] cos(context0, context_star)
    success: torch.Tensor               # [n] bool, NLL improved and stayed finite
    trajectory: Optional[List[torch.Tensor]] = None  # only when requested


class ContextRefiner:
    """Per-example, flow-frozen context optimization: given the encoder's
    raw context0 [n, context_dim] and each example's TRUE physical
    parameters, runs Adam directly on a context leaf tensor (initialized
    from context0) to maximize the flow's likelihood at the truth, with an
    optional quadratic penalty keeping the result near context0 (the
    "stay near the encoder manifold" regularizer,
    --sa_manifold_lambda). The flow's own conditioning path (q_embed_head,
    if present, and rank_embed) is applied FRESH each step from the current
    refined context, with all of THEIR parameters frozen for the duration
    -- exactly "freeze flow" from the spec: only the context tensor itself
    is a leaf requiring grad.

    Batched: all `n` examples are refined in parallel (independent leaf
    tensors, one shared Adam optimizer over the whole batch), for
    throughput; early stopping is still tracked and reported per-example
    even though only whole-batch step budgets are actually enforced (a
    per-example step cap inside a batched Adam call is not meaningful --
    PATIENCE is evaluated on the BATCH-MEAN nll improvement, and once
    triggered stops refinement for the whole call. `steps_taken` in the
    result records the batch's actual stopping step for every example in
    it, which is what the epoch-level diagnostics in train_lean_npe.py
    average over).
    """

    def __init__(self, model, steps: int = 20, lr: float = 0.05,
                patience: int = 5, tol: float = 1e-3,
                q_error_threshold: Optional[float] = None,
                manifold_lambda: float = 0.0,
                record_trajectory: bool = False,
                diag_samples: int = 32,
                manifold_stats: Optional["ManifoldStats"] = None):
        self.model = model
        self.steps = int(steps)
        self.lr = float(lr)
        self.patience = int(patience)
        self.tol = float(tol)
        self.q_error_threshold = q_error_threshold
        self.manifold_lambda = float(manifold_lambda)
        self.record_trajectory = bool(record_trajectory)
        # The penalty MUST use the same whitened-PCA Mahalanobis metric the
        # oracle_manifold_tradeoff.py sweep validated lambda against -- raw
        # (unwhitened) context distance uses different units entirely and
        # silently fails to constrain anything (see ManifoldStats' docstring
        # for the live-run evidence). Falls back to a raw-L2 penalty ONLY
        # when no manifold_stats is supplied (e.g. too few contexts sampled
        # to fit a PCA) -- that fallback is NOT validated against the
        # oracle sweep's lambda values and should be treated as a different,
        # weaker regularizer if it's ever actually used.
        self.manifold_stats = manifold_stats
        # q_before/q_after are diagnostic-only (never used in the loss or
        # early-stopping decision unless q_error_threshold is set) but each
        # call runs model.flow.inverse(), which for this autoregressive
        # spline flow is a sequential, per-dimension operation -- much more
        # expensive per sample than the parallelizable forward direction
        # compute_psd_aware_nll() uses for the actual training loss. Keeping
        # this small (default 32, vs. the 256+ used for real posterior
        # sampling elsewhere in the repo) avoids paying real-posterior-
        # quality sampling cost for a number that only ever appears in a
        # log line.
        self.diag_samples = int(diag_samples)

    def _flow_nll(self, context: torch.Tensor, rank: torch.Tensor,
                 y_true: torch.Tensor, log_sigma_psd: torch.Tensor) -> torch.Tensor:
        """[n] per-example NLL for `context` (encoder-space, BEFORE
        q_embed/rank) at the given normalized targets `y_true`. Rebuilds the
        flow's full conditioning vector fresh from `context` every call --
        q_embed_head/rank_embed act as frozen, differentiable-w.r.t.-context
        functions here (their PARAMETERS require no grad during refinement,
        set by the caller)."""
        model = self.model
        full = model._full_context(context, rank)
        return model.flow.compute_psd_aware_nll(y_true, full, log_sigma_psd)

    @staticmethod
    def _q_point_estimate(model, context: torch.Tensor, rank: torch.Tensor,
                          n_samp: int = 256, seed: int = 0) -> torch.Tensor:
        """Cheap [n] median-q point estimate for diagnostics only (NOT part
        of the refinement objective, which is likelihood at the truth, not
        a target q value). n_samp<=0 skips sampling entirely (returns NaN
        placeholders) for users who want zero diagnostic overhead."""
        n = context.shape[0]
        if n_samp <= 0:
            return torch.full((n,), float("nan"), device=context.device)
        with torch.no_grad():
            full = model._full_context(context, rank)
            z = torch.randn(n, n_samp, len(model.param_names), device=context.device)
            fcr = full.unsqueeze(1).expand(n, n_samp, full.shape[1]).reshape(n * n_samp, -1)
            y, _ = model.flow.inverse(z.reshape(n * n_samp, -1), fcr)
            s = model.scaler.denormalize(y).reshape(n, n_samp, -1)
            m1, m2 = s[..., 0], s[..., 1]
            q = torch.minimum(m1, m2) / torch.maximum(m1, m2).clamp_min(1e-6)
            return q.median(dim=1).values

    def refine(self, context0: torch.Tensor, rank: torch.Tensor,
              params_true_phys: torch.Tensor) -> RefinementResult:
        """context0: [n, context_dim] (encoder output, will be detached).
        rank: [n] long. params_true_phys: [n, 11] physical, this example's
        rank-0 truth (the parameters the refined context should explain)."""
        model = self.model
        device = context0.device
        n = context0.shape[0]

        context0 = context0.detach()
        y_true = model.scaler.normalize(params_true_phys)
        log_sigma_psd = torch.zeros_like(y_true)

        # Freeze everything except the context leaf: encoder is not even
        # in this call's graph (context0 already detached above); flow +
        # q_embed_head/q_pred_head + rank_embed parameters are frozen for
        # the duration of the inner loop and restored after.
        frozen_modules = [model.flow, model.rank_embed]
        if model.q_embed_dim > 0:
            frozen_modules += [model.q_embed_head, model.q_pred_head]
        prev_requires_grad = []
        for mod in frozen_modules:
            for p in mod.parameters():
                prev_requires_grad.append((p, p.requires_grad))
                p.requires_grad_(False)

        try:
            context = context0.clone().requires_grad_(True)
            opt = torch.optim.Adam([context], lr=self.lr)

            q_before = self._q_point_estimate(model, context0, rank, n_samp=self.diag_samples)
            nll_initial = self._flow_nll(context0, rank, y_true, log_sigma_psd).detach()

            best_nll = nll_initial.clone()
            stall_count = torch.zeros(n, dtype=torch.long, device=device)
            steps_taken = torch.zeros(n, dtype=torch.long, device=device)
            active = torch.ones(n, dtype=torch.bool, device=device)
            trajectory: List[torch.Tensor] = [context0.clone()] if self.record_trajectory else None

            last_nll = nll_initial
            for step in range(self.steps):
                if not active.any():
                    break
                opt.zero_grad()
                nll = self._flow_nll(context, rank, y_true, log_sigma_psd)
                if self.manifold_lambda > 0.0:
                    if self.manifold_stats is not None:
                        penalty = self.manifold_lambda * self.manifold_stats.distance2_diffable(context)
                    else:
                        penalty = self.manifold_lambda * ((context - context0) ** 2).sum(dim=1)
                else:
                    penalty = torch.zeros(n, device=device)
                loss = (nll + penalty)
                # Only backprop through examples still being refined, so
                # already-converged examples don't keep getting nudged by a
                # (possibly noisy) gradient after they've satisfied the
                # early-stopping criterion.
                loss_active = (loss * active.float()).sum()
                loss_active.backward()
                opt.step()

                with torch.no_grad():
                    steps_taken += active.long()
                    improved = (best_nll - nll.detach()) > self.tol
                    best_nll = torch.where(improved, nll.detach(), best_nll)
                    stall_count = torch.where(improved, torch.zeros_like(stall_count),
                                              stall_count + 1)
                    newly_stalled = stall_count >= self.patience
                    if self.q_error_threshold is not None:
                        q_now = self._q_point_estimate(model, context, rank, n_samp=self.diag_samples)
                        # NOTE: q_error_threshold is an ABSOLUTE q value the
                        # point estimate must reach (not an error vs. an
                        # unknown truth, since refinement targets likelihood
                        # at the truth, not a target q directly) -- used
                        # here as "close enough to a sensible mass ratio,
                        # stop early" heuristic per the --sa_tol/--sa_patience
                        # spec; disabled (None) by default.
                        threshold_met = (q_now - q_before).abs() < self.q_error_threshold
                        newly_stalled = newly_stalled | threshold_met
                    active = active & (~newly_stalled)
                    last_nll = nll.detach()
                    if self.record_trajectory:
                        trajectory.append(context.detach().clone())

            context_star = context.detach()
            nll_final = last_nll
            q_after = self._q_point_estimate(model, context_star, rank, n_samp=self.diag_samples)

            refine_distance = (context_star - context0).norm(dim=1)
            cosine_sim = F.cosine_similarity(context0, context_star, dim=1)
            success = torch.isfinite(nll_final) & (nll_final <= nll_initial + 1e-6)

            return RefinementResult(
                context_star=context_star, steps_taken=steps_taken,
                nll_initial=nll_initial, nll_final=nll_final,
                q_before=q_before, q_after=q_after,
                refine_distance=refine_distance, cosine_sim=cosine_sim,
                success=success, trajectory=trajectory,
            )
        finally:
            for p, req in prev_requires_grad:
                p.requires_grad_(req)


# --------------------------------------------------------------------------- #
# Manifold distance (diagnostic only -- see analysis/attractor_basin/
# oracle_manifold_tradeoff.py, the exact methodology this reuses)
# --------------------------------------------------------------------------- #

class ManifoldStats:
    """Whitened-PCA Mahalanobis-style distance to a reference population of
    encoder contexts, fit ONCE (e.g. from a sample of training contexts)
    and held fixed thereafter. Used BOTH as a cheap no-grad diagnostic
    (`.distance()`, what the `maha` log field reports) AND, via
    `.distance2_diffable()`, as the ACTUAL --sa_manifold_lambda penalty
    inside ContextRefiner -- these must be the same metric, or the penalty
    coefficient validated in analysis/attractor_basin/oracle_manifold_tradeoff.py
    (where D(c,M) is exactly this whitened distance) doesn't mean what it
    says it means. An earlier version of ContextRefiner penalized RAW
    (unwhitened) squared context distance instead, which uses completely
    different units -- lambda=0.1 there did not correspond to lambda=0.1
    here at all, and a live training run confirmed the mismatch: `maha`
    stayed at ~450-500 (full collapse) despite --sa_manifold_lambda 0.1
    being set, identical to the fully-unconstrained (lambda=0) run."""

    def __init__(self, contexts: torch.Tensor, n_components: int = 20):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, contexts.shape[0] - 1,
                                   contexts.shape[1])).fit(contexts.cpu().numpy())
        self.mean = torch.from_numpy(pca.mean_.astype("float32"))
        self.components = torch.from_numpy(pca.components_.astype("float32"))
        self.inv_std = torch.from_numpy(
            (1.0 / (pca.explained_variance_ ** 0.5)).astype("float32"))

    def to(self, device):
        self.mean = self.mean.to(device)
        self.components = self.components.to(device)
        self.inv_std = self.inv_std.to(device)
        return self

    def distance2_diffable(self, context: torch.Tensor) -> torch.Tensor:
        """[n] SQUARED whitened Mahalanobis-style distance, WITH gradient
        (no @torch.no_grad()) -- this is the differentiable form used
        directly inside ContextRefiner's penalty term. mean/components/
        inv_std are fixed buffers (never require grad themselves), so
        gradient flows through `context` only, exactly as intended."""
        z = (context - self.mean) @ self.components.T * self.inv_std
        return (z ** 2).sum(dim=1)

    @torch.no_grad()
    def distance(self, context: torch.Tensor) -> torch.Tensor:
        """context: [n, context_dim] -> [n] distance."""
        z = (context - self.mean) @ self.components.T * self.inv_std
        return z.norm(dim=1)


# --------------------------------------------------------------------------- #
# Distillation loss
# --------------------------------------------------------------------------- #

class DistillationLoss(nn.Module):
    """Pulls the encoder's LIVE context (grad-enabled, freshly recomputed
    from strain) toward the (detached) refined target context*. Never
    touches the flow -- this loss is defined purely in context space."""

    def __init__(self, kind: str = "mse"):
        super().__init__()
        assert kind in ("mse", "cosine", "huber")
        self.kind = kind

    def forward(self, context_live: torch.Tensor, context_star: torch.Tensor) -> torch.Tensor:
        target = context_star.detach()
        if self.kind == "mse":
            return F.mse_loss(context_live, target)
        if self.kind == "huber":
            return F.smooth_l1_loss(context_live, target)
        # cosine: 1 - cos_sim, averaged, so 0 = perfectly aligned (a proper
        # minimization target, not raw cosine similarity itself)
        cos = F.cosine_similarity(context_live, target, dim=-1)
        return (1.0 - cos).mean()


# --------------------------------------------------------------------------- #
# NOTE: the live per-batch orchestrator (SemiAmortizedTrainer) that used to
# live here has been REMOVED (2026-07-23) -- see the module docstring for
# why. Use experiments/precompute_distill_targets.py to run ContextRefiner
# OFFLINE against a frozen checkpoint instead, and train_lean_npe.py's
# --offline_distill_cache to train against the resulting fixed cache.
# --------------------------------------------------------------------------- #
