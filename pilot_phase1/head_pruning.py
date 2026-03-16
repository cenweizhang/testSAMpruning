# -*- coding: utf-8 -*-
"""
Head-level structured pruning for MedSAM Image Encoder.

Implements 4 head importance scoring methods:
  1. Random scoring
  2. Magnitude scoring (L2 norm of head's qkv + proj weights)
  3. Pointwise Regression scoring (per-sample gradient alignment)
  4. EWR scoring (Entropy-regularized Wasserstein Regression)

Changes vs original:
  P0.1: apply_head_mask_to_model — hook moved to BEFORE proj (forward_pre_hook
        on attn.proj), so zeroing is semantically correct head removal.
  P0.2: compute_head_gradient_projections_fast — single forward+backward pass
        per sample (all blocks together), eliminating the 12x redundant passes.
  P0.3: Same single-pass produces consistent global gradients; projection sum
        is now computed within one computation graph.
  P1.1: compute_head_gradient_projections_fast accepts freq_weights and applies
        sqrt(ω_freq) weighting to each sample's projection (spec §3).
  P1.2: score_heads_ewr adds L2 regularization term λ1‖θ_j‖² (spec §3 EWR obj).
        compute_head_l2_norms() computes ‖θ̄_j‖ for each head.
  Removed: compute_head_gradient_projections() (batch-averaged, wrong per-sample logic).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def dice_loss(pred, target):
    """Soft Dice loss (sigmoid activation, squared_pred=False, mean reduction)."""
    pred = torch.sigmoid(pred)
    pred   = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    loss = 1.0 - (2.0 * intersection + 1e-5) / (
        pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1e-5
    )
    return loss.mean()


# ==============================================================================
# P0.2 + P0.3 + P1.1 — Gradient Computation (single-pass, all blocks, weighted)
# ==============================================================================

def compute_head_gradient_projections_fast(
    model, dataloader, device,
    num_blocks=12, num_heads=12,
    freq_weights=None,
):
    """
    Compute per-sample gradient projections for each attention head.

    P0.2 fix: ONE forward+backward pass per sample (all attention blocks
    enabled simultaneously), eliminating 12 redundant forward passes.

    P0.3 fix: Because all blocks share the same computation graph, the
    per-head projections g_j^T·θ_j are computed from a single consistent
    gradient, making their sum a valid approximation of the global attention-
    subspace projection.

    P1.1: Applies sqrt(ω_i^freq) weighting:
        per_sample_projections[i, j] = sqrt(ω_i) · g_{i,j}^T · θ_j
    which constructs the frequency-weighted empirical distribution for EWR.

    Args:
        model        : MedSAM model (Sam) on device, eval mode
        dataloader   : calibration DataLoader (batch_size arbitrary)
        device       : torch device
        num_blocks   : number of ViT blocks (12 for ViT-B)
        num_heads    : attention heads per block (12 for ViT-B)
        freq_weights : np.ndarray of shape (n_samples,) with ω_i^freq values.
                       If None, all weights are 1 (no frequency prior).

    Returns:
        head_importance        : np.ndarray (total_heads,) — mean |projection|
        per_sample_projections : np.ndarray (n_samples, total_heads) — weighted
    """
    model.eval()

    n_samples  = len(dataloader.dataset)
    total_heads = num_blocks * num_heads

    if freq_weights is None:
        freq_weights = np.ones(n_samples, dtype=np.float32)

    assert len(freq_weights) == n_samples, (
        f"freq_weights length {len(freq_weights)} != n_samples {n_samples}"
    )

    per_sample_projections = np.zeros((n_samples, total_heads), dtype=np.float32)

    sample_idx = 0

    for batch in tqdm(dataloader, desc="Computing head gradients (single-pass)"):
        images = batch["image"].to(device)    # (B, 3, 1024, 1024)
        masks  = batch["mask_256"].to(device) # (B, 1, 256, 256)
        bboxes = batch["bbox"].numpy()        # (B, 4)
        B = images.shape[0]

        for b_idx in range(B):
            if sample_idx >= n_samples:
                break

            img = images[b_idx:b_idx + 1]
            msk = masks[b_idx:b_idx + 1]
            box = bboxes[b_idx:b_idx + 1]

            # -------------------------------------------------------------- #
            # P0.2: Freeze all params, then unfreeze ALL attention blocks at once
            # -------------------------------------------------------------- #
            for p in model.parameters():
                p.requires_grad = False
            for blk_idx in range(num_blocks):
                for p in model.image_encoder.blocks[blk_idx].attn.parameters():
                    p.requires_grad = True

            model.zero_grad()

            # -------------------------------------------------------------- #
            # Single forward pass (computation graph retained for all blocks)
            # -------------------------------------------------------------- #
            image_embedding = model.image_encoder(img)   # (1, 256, 64, 64)

            with torch.no_grad():
                box_t = torch.as_tensor(box, dtype=torch.float32, device=device)
                if len(box_t.shape) == 2:
                    box_t = box_t[:, None, :]
                sparse_emb, dense_emb = model.prompt_encoder(
                    points=None, boxes=box_t, masks=None
                )

            # Mask decoder is NOT under no_grad: gradients must flow back
            # through it to reach image_embedding → image_encoder parameters.
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )   # (1, 1, 256, 256)

            loss = (
                F.binary_cross_entropy_with_logits(low_res_masks, msk.float())
                + dice_loss(low_res_masks, msk.float())
            )

            # Single backward — computes gradients for all enabled attention blocks
            loss.backward()

            # -------------------------------------------------------------- #
            # P1.1: Frequency weighting — apply sqrt(ω_i) to this sample's proj
            # -------------------------------------------------------------- #
            sqrt_omega = float(np.sqrt(freq_weights[sample_idx]))

            # -------------------------------------------------------------- #
            # P0.3: Extract per-head projections from ALL blocks in one loop
            #        (all gradients from the same computation graph)
            # -------------------------------------------------------------- #
            with torch.no_grad():
                for blk_idx in range(num_blocks):
                    attn      = model.image_encoder.blocks[blk_idx].attn
                    dim       = attn.qkv.weight.shape[1]
                    head_dim  = dim // num_heads

                    qkv_g = attn.qkv.weight.grad   # (3*dim, dim)
                    qkv_w = attn.qkv.weight.data
                    proj_g = attn.proj.weight.grad  # (dim, dim)
                    proj_w = attn.proj.weight.data

                    for h in range(num_heads):
                        head_id = blk_idx * num_heads + h
                        q_s = slice(h * head_dim,            (h + 1) * head_dim)
                        k_s = slice(dim + h * head_dim,      dim + (h + 1) * head_dim)
                        v_s = slice(2 * dim + h * head_dim,  2 * dim + (h + 1) * head_dim)
                        p_s = slice(h * head_dim,            (h + 1) * head_dim)

                        val = 0.0
                        for s in [q_s, k_s, v_s]:
                            val += (qkv_g[s, :] * qkv_w[s, :]).sum().item()
                        val += (proj_g[:, p_s] * proj_w[:, p_s]).sum().item()

                        # Bias terms (included for completeness)
                        if attn.qkv.bias is not None and attn.qkv.bias.grad is not None:
                            qb_g = attn.qkv.bias.grad
                            qb_w = attn.qkv.bias.data
                            for s in [q_s, k_s, v_s]:
                                val += (qb_g[s] * qb_w[s]).sum().item()

                        if attn.proj.bias is not None and attn.proj.bias.grad is not None:
                            pb_g = attn.proj.bias.grad
                            pb_w = attn.proj.bias.data
                            # proj bias is shared; distribute equally across heads
                            val += (pb_g * pb_w).sum().item() / num_heads

                        # P1.1: frequency-weighted projection
                        per_sample_projections[sample_idx, head_id] = val * sqrt_omega

            # Clean up
            model.zero_grad()
            for blk_idx in range(num_blocks):
                for p in model.image_encoder.blocks[blk_idx].attn.parameters():
                    p.requires_grad = False

            sample_idx += 1

    head_importance = np.abs(per_sample_projections).mean(axis=0)
    return head_importance, per_sample_projections


# ==============================================================================
# Scoring Methods
# ==============================================================================

def score_heads_random(num_blocks=12, num_heads=12, seed=42):
    """Random importance scores."""
    rng = np.random.RandomState(seed)
    return rng.rand(num_blocks * num_heads)


def score_heads_magnitude(model, num_blocks=12, num_heads=12):
    """
    Magnitude-based scoring: L2 norm of each head's qkv + proj weights.
    Higher norm → more important.
    """
    scores = np.zeros(num_blocks * num_heads)

    for blk_idx in range(num_blocks):
        attn     = model.image_encoder.blocks[blk_idx].attn
        dim      = attn.qkv.weight.shape[1]
        head_dim = dim // num_heads

        with torch.no_grad():
            qkv_w  = attn.qkv.weight.data
            proj_w = attn.proj.weight.data

            for h in range(num_heads):
                head_id = blk_idx * num_heads + h
                q_s = slice(h * head_dim,           (h + 1) * head_dim)
                k_s = slice(dim + h * head_dim,     dim + (h + 1) * head_dim)
                v_s = slice(2 * dim + h * head_dim, 2 * dim + (h + 1) * head_dim)
                p_s = slice(h * head_dim,           (h + 1) * head_dim)

                norm_sq = 0.0
                for s in [q_s, k_s, v_s]:
                    norm_sq += qkv_w[s, :].norm().item() ** 2
                norm_sq += proj_w[:, p_s].norm().item() ** 2
                scores[head_id] = np.sqrt(norm_sq)

    return scores


def score_heads_pointwise(per_sample_projections):
    """
    Pointwise regression scoring.
    Score = mean |g_i^T · θ_j| across calibration samples.
    (If freq_weights were applied in gradient computation, projections are
    already frequency-weighted; no additional weighting needed here.)
    """
    return np.abs(per_sample_projections).mean(axis=0)


# ==============================================================================
# P1.2 — L2 norm helper for EWR regularisation
# ==============================================================================

def compute_head_l2_norms(model, num_blocks=12, num_heads=12):
    """
    Compute L2 norm of teacher parameters for each attention head.

    Used for the EWR regularisation term:
        λ1 · ‖θ_z − θ̄‖²  ≈  λ1 · Σ_{j pruned} ‖θ̄_j‖²

    Because removing head j zeros out its parameters (θ_j → 0),
    the contribution of head j to the L2 penalty is ‖θ̄_j‖².

    Returns:
        norms: np.ndarray (total_heads,) — ‖θ̄_j‖ for each head j
    """
    norms = np.zeros(num_blocks * num_heads, dtype=np.float64)

    for blk_idx in range(num_blocks):
        attn     = model.image_encoder.blocks[blk_idx].attn
        dim      = attn.qkv.weight.shape[1]
        head_dim = dim // num_heads

        with torch.no_grad():
            qkv_w  = attn.qkv.weight.data
            proj_w = attn.proj.weight.data

            for h in range(num_heads):
                head_id = blk_idx * num_heads + h
                q_s = slice(h * head_dim,           (h + 1) * head_dim)
                k_s = slice(dim + h * head_dim,     dim + (h + 1) * head_dim)
                v_s = slice(2 * dim + h * head_dim, 2 * dim + (h + 1) * head_dim)
                p_s = slice(h * head_dim,           (h + 1) * head_dim)

                norm_sq = 0.0
                for s in [q_s, k_s, v_s]:
                    norm_sq += qkv_w[s, :].norm().item() ** 2
                norm_sq += proj_w[:, p_s].norm().item() ** 2

                if attn.qkv.bias is not None:
                    for s in [q_s, k_s, v_s]:
                        norm_sq += attn.qkv.bias.data[s].norm().item() ** 2

                norms[head_id] = np.sqrt(norm_sq)

    return norms.astype(np.float32)


# ==============================================================================
# Sinkhorn (log-domain)
# ==============================================================================

def _logsumexp(a, axis=None):
    """Numerically stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    result = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return result.squeeze(axis=axis)


def _sinkhorn_distance(x, y, epsilon=0.05, n_iter=200):
    """
    Entropy-regularised 2-Wasserstein distance W_{2,ε}² between two
    1D empirical distributions, computed in log-domain for stability.

    n_iter increased to 200 (from 100) to improve convergence for small ε.

    Args:
        x: (n,) samples from μ (pruned projections)
        y: (m,) samples from ν (teacher projections)
        epsilon: entropic regularisation (smaller → closer to true W₂)
        n_iter: Sinkhorn iterations

    Returns:
        float: approximate W_{2,ε}²
    """
    n = len(x)
    m = len(y)

    C      = (x[:, None] - y[None, :]) ** 2   # (n, m) cost matrix
    log_K  = -C / epsilon

    log_a  = np.full(n, -np.log(n))
    log_b  = np.full(m, -np.log(m))

    log_u  = np.zeros(n)
    log_v  = np.zeros(m)

    for _ in range(n_iter):
        log_u = log_a - _logsumexp(log_K + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp(log_K + log_u[:, None], axis=0)

    log_pi = log_u[:, None] + log_K + log_v[None, :]
    pi     = np.exp(log_pi)

    return float((C * pi).sum())


# ==============================================================================
# P1.2 — EWR scoring with L2 regularisation
# ==============================================================================

def score_heads_ewr(
    per_sample_projections,
    epsilon=0.05,
    model=None,
    lambda1=0.01,
    num_blocks=12,
    num_heads=12,
):
    """
    EWR (Entropy-regularised Wasserstein Regression) head importance scoring.

    For each head j, score = increase in W_{2,ε}² when head j is removed,
    plus an L2 regularisation term:

        score_j = W_{2,ε}²(μ_{without j}, ν_teacher)  +  λ1 · ‖θ̄_j‖²

    P1.2 change: The L2 term λ1·‖θ̄_j‖² is added when model is provided.
    It penalises removing large-norm heads, matching the EWR objective in spec §3.

    If freq_weights were applied during gradient computation (P1.1), the
    projections in per_sample_projections are already ω-weighted, so the
    distributions μ and ν are the correct frequency-weighted ones (spec §3).

    Args:
        per_sample_projections : (n_samples, total_heads) — already ω-weighted
        epsilon                : Sinkhorn regularisation
        model                  : Sam model (needed for L2 norm computation).
                                 Pass None to disable L2 term.
        lambda1                : L2 regularisation coefficient (default 0.01)
        num_blocks, num_heads  : architecture parameters

    Returns:
        scores: (total_heads,) — higher means head is more important (keep it)
    """
    n_samples, total_heads = per_sample_projections.shape
    scores = np.zeros(total_heads, dtype=np.float32)

    # Teacher projection: sum over all heads (approximation of global projection)
    # Because all projections come from the same computation graph (P0.2/P0.3),
    # this sum is a consistent approximation of g_i^T · θ_attn.
    teacher_total = per_sample_projections.sum(axis=1)   # (n_samples,)

    # P1.2: Pre-compute L2 norms of teacher head parameters
    if model is not None and lambda1 > 0.0:
        head_norms   = compute_head_l2_norms(model, num_blocks, num_heads)
        head_l2_sq   = (head_norms ** 2).astype(np.float32)
    else:
        head_l2_sq   = np.zeros(total_heads, dtype=np.float32)

    for j in tqdm(range(total_heads), desc=f"EWR scoring (ε={epsilon}, λ1={lambda1})"):
        # Distribution when head j is removed
        pruned_total = teacher_total - per_sample_projections[:, j]

        # Wasserstein distance between pruned and teacher distributions
        w_dist = _sinkhorn_distance(pruned_total, teacher_total, epsilon=epsilon)

        # P1.2: Add L2 regularisation term
        scores[j] = w_dist + lambda1 * head_l2_sq[j]

    return scores


# ==============================================================================
# P0.1 — Head Mask Generation and Application (hook BEFORE proj)
# ==============================================================================

def generate_head_mask(scores, sparsity, num_blocks=12, num_heads=12):
    """
    Generate a binary head mask: remove the (sparsity * total_heads) least
    important heads (lowest scores).

    Args:
        scores   : (total_heads,) importance scores — higher means more important
        sparsity : fraction of heads to remove (e.g. 0.3)

    Returns:
        head_mask: (total_heads,) binary mask, 1 = keep, 0 = remove
    """
    total_heads = num_blocks * num_heads
    n_remove    = int(total_heads * sparsity)

    sorted_indices = np.argsort(scores)   # ascending: least important first
    mask = np.ones(total_heads, dtype=np.float32)
    mask[sorted_indices[:n_remove]] = 0.0
    return mask


def apply_head_mask_to_model(model, head_mask, num_blocks=12, num_heads=12):
    """
    Apply head mask via a forward_pre_hook on each block's attn.proj layer.

    P0.1 fix: The hook now fires BEFORE the projection layer, intercepting
    the concatenated head outputs (shape: ..., dim) and zeroing the slice
    corresponding to each pruned head. This is the semantically correct point
    to remove a head's contribution, because proj (W_proj) mixes all heads —
    masking after proj does not correspond to head removal.

    The hook targets attn.proj's input, which has shape (..., dim) where
    dim = num_heads * head_dim and head h occupies columns
    [h*head_dim : (h+1)*head_dim].

    Args:
        model     : Sam model
        head_mask : (total_heads,) np array, 1=keep, 0=remove

    Returns:
        hooks: list of hook handles (call .remove() to undo)
    """
    hooks = []

    for blk_idx in range(num_blocks):
        attn_module = model.image_encoder.blocks[blk_idx].attn
        block_mask  = head_mask[blk_idx * num_heads: (blk_idx + 1) * num_heads]

        # Pre-compute the channel-level mask once per block (shape: dim,)
        # so the hook itself has no Python loop over heads.
        head_dim     = attn_module.qkv.weight.shape[1] // num_heads
        channel_mask = np.repeat(block_mask, head_dim).astype(np.float32)

        def make_pre_hook(ch_mask):
            """Closure capturing the channel mask for one block."""
            def hook_fn(module, input):
                # input is a tuple; input[0]: (..., dim)  (before W_proj)
                x = input[0]
                m = torch.tensor(ch_mask, dtype=x.dtype, device=x.device)
                # Broadcast: (..., dim) * (dim,)  →  zero pruned head channels
                return (x * m,)
            return hook_fn

        # register_forward_pre_hook fires before module.forward(), i.e. before proj
        hook = attn_module.proj.register_forward_pre_hook(
            make_pre_hook(channel_mask)
        )
        hooks.append(hook)

    return hooks


def remove_hooks(hooks):
    """Remove all registered hooks to restore original model behaviour."""
    for h in hooks:
        h.remove()


# ==============================================================================
# Convenience: compute all scores
# ==============================================================================

def compute_all_head_scores(
    model, cal_loader, device,
    freq_weights=None,
    epsilon_values=None,
    lambda1=0.01,
):
    """
    Compute head importance scores for all methods.

    Args:
        model          : MedSAM model on device
        cal_loader     : calibration DataLoader
        device         : torch device
        freq_weights   : np.ndarray (n_samples,) — ω_freq per sample (P1.1).
                         Pass None to skip frequency weighting.
        epsilon_values : list of ε for EWR (default [0.01, 0.05, 0.1])
        lambda1        : L2 regularisation coefficient for EWR (P1.2)

    Returns:
        dict  method_name → scores array (total_heads,)
    """
    if epsilon_values is None:
        epsilon_values = [0.01, 0.05, 0.1]

    print("=" * 60)
    print("Computing head importance scores")
    print("=" * 60)

    results = {}

    print("\n[1/4] Random scoring ...")
    results["random"] = score_heads_random()

    print("[2/4] Magnitude scoring ...")
    results["magnitude"] = score_heads_magnitude(model)

    print("[3/4] Computing gradient projections (single-pass per sample) ...")
    head_importance, per_sample_proj = compute_head_gradient_projections_fast(
        model, cal_loader, device, freq_weights=freq_weights
    )

    print("[3/4] Pointwise scoring ...")
    results["pointwise"] = score_heads_pointwise(per_sample_proj)

    print("[4/4] EWR scoring ...")
    for eps in epsilon_values:
        key = f"ewr_eps{eps}"
        results[key] = score_heads_ewr(
            per_sample_proj,
            epsilon=eps,
            model=model,
            lambda1=lambda1,
        )

    return results
