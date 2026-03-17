# -*- coding: utf-8 -*-
"""
Head-level structured pruning for MedSAM Image Encoder.
Version 2: Output-space Taylor proxy + block-wise exhaustive search.

V2 changes (vs v1 P0/P1/P2):
  V2.1: compute_head_gradient_projections_fast — replaced parameter-space proxy
        g_j^T θ_j with output-space Taylor criterion
        c_{i,l,h} = <δ_{i,l,h}, a_{i,l,h}>
        via register_forward_hook + register_full_backward_hook on attn.proj.
        No parameter gradient extraction; freq_weight still applied.

  V2.2: score_heads_pointwise / score_heads_ewr are now per-block:
        T_{i,l} = Σ_h c_{i,l,h}  (block teacher response)
        T_{i,l}(z_l) = Σ_{h: z=1} c_{i,l,h}  (mask-retained response)

  V2.3: generate_head_mask_blockwise — exhaustive C(H,k) search per block:
        argmin_{z_l} Q(z_l)  where Q is Q_pw or Q_ewr.
        Uniform cross-block budget: k_remove = floor(H * sparsity) per block.

  V2.4: Relative epsilon ε_l = α × median_pairwise_dist({T_{i,l}})
        computed per block, replacing hardcoded absolute ε.

  V2.5: diagnose_epsilon_sensitivity — Check 3 diagnostic.
"""

import torch
import torch.nn.functional as F
import numpy as np
from itertools import combinations
from tqdm import tqdm


# ==============================================================================
# Utility
# ==============================================================================

def dice_loss(pred, target):
    """Soft Dice loss (sigmoid, mean reduction)."""
    pred   = torch.sigmoid(pred)
    pred   = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    return (1.0 - (2.0 * intersection + 1e-5) /
            (pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1e-5)).mean()


# ==============================================================================
# V2.1 — Output-space Taylor proxy via hooks
# ==============================================================================

def compute_head_gradient_projections_fast(
    model, dataloader, device,
    num_blocks=12, num_heads=12,
    freq_weights=None,
):
    """
    Compute per-sample output-space Taylor proxy c_{i,l,h} for each head.

    For block l, head h, sample i:
        c_{i,l,h} = <δ_{i,l,h}, a_{i,l,h}>
    where
        a_{i,l,h} ∈ R^{N×d_h}  : pre-proj activation slice for head h
        δ_{i,l,h} = ∂ℓ/∂a_{i,l,h}  : gradient at that activation

    Captured via:
        register_forward_hook on attn.proj  → input[0] = pre-proj tensor (*, dim)
        register_full_backward_hook on attn.proj → grad_input[0] = ∂ℓ/∂(pre-proj)

    Freq weighting (P1.1 kept):
        per_sample_proj[i, l*H+h] = sqrt(ω_i) · c_{i,l,h}

    Returns:
        head_importance        : (total_heads,) mean |c_{i,l,h}|
        per_sample_projections : (n_samples, total_heads) freq-weighted c values
    """
    model.eval()

    n_samples   = len(dataloader.dataset)
    total_heads = num_blocks * num_heads

    if freq_weights is None:
        freq_weights = np.ones(n_samples, dtype=np.float32)
    assert len(freq_weights) == n_samples

    per_sample_projections = np.zeros((n_samples, total_heads), dtype=np.float32)

    # Pre-compute head_dim once
    head_dim = model.image_encoder.blocks[0].attn.qkv.weight.shape[1] // num_heads

    sample_idx = 0

    for batch in tqdm(dataloader, desc="Computing Taylor proxy c_{i,l,h} (hook-based)"):
        images = batch["image"].to(device)
        masks  = batch["mask_256"].to(device)
        bboxes = batch["bbox"].numpy()
        B = images.shape[0]

        for b_idx in range(B):
            if sample_idx >= n_samples:
                break

            img = images[b_idx:b_idx + 1]
            msk = masks[b_idx:b_idx + 1]
            box = bboxes[b_idx:b_idx + 1]

            # -------------------------------------------------------------- #
            # Enable attn params requires_grad to anchor the computation graph
            # so that loss.backward() flows gradients to the image encoder.
            # -------------------------------------------------------------- #
            for p in model.parameters():
                p.requires_grad = False
            for blk_idx in range(num_blocks):
                for p in model.image_encoder.blocks[blk_idx].attn.parameters():
                    p.requires_grad = True

            model.zero_grad()

            # -------------------------------------------------------------- #
            # Register hooks: capture pre-proj activations and their gradients
            # -------------------------------------------------------------- #
            saved_pre  = {}  # blk_idx → (*, dim) pre-proj tensor
            saved_grad = {}  # blk_idx → (*, dim) gradient at pre-proj

            def make_fwd_hook(idx):
                def hook(module, inp, out):
                    # inp[0]: tensor fed into nn.Linear (= pre-proj activation)
                    saved_pre[idx] = inp[0].detach()
                return hook

            def make_bwd_hook(idx):
                def hook(module, grad_inp, grad_out):
                    # grad_inp[0]: ∂loss/∂(pre-proj activation)
                    if grad_inp[0] is not None:
                        saved_grad[idx] = grad_inp[0].detach()
                return hook

            fwd_hooks, bwd_hooks = [], []
            for blk_idx in range(num_blocks):
                proj = model.image_encoder.blocks[blk_idx].attn.proj
                fwd_hooks.append(proj.register_forward_hook(make_fwd_hook(blk_idx)))
                bwd_hooks.append(proj.register_full_backward_hook(make_bwd_hook(blk_idx)))

            # -------------------------------------------------------------- #
            # Single forward + backward
            # -------------------------------------------------------------- #
            image_embedding = model.image_encoder(img)

            with torch.no_grad():
                box_t = torch.as_tensor(box, dtype=torch.float32, device=device)
                if box_t.dim() == 2:
                    box_t = box_t[:, None, :]
                sparse_emb, dense_emb = model.prompt_encoder(
                    points=None, boxes=box_t, masks=None
                )

            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

            loss = (
                F.binary_cross_entropy_with_logits(low_res_masks, msk.float())
                + dice_loss(low_res_masks, msk.float())
            )
            loss.backward()

            # Remove hooks immediately after backward
            for hk in fwd_hooks + bwd_hooks:
                hk.remove()

            # -------------------------------------------------------------- #
            # Extract c_{i,l,h} = <pre[..., s], grad[..., s]>.sum()
            # -------------------------------------------------------------- #
            sqrt_omega = float(np.sqrt(freq_weights[sample_idx]))

            with torch.no_grad():
                for blk_idx in range(num_blocks):
                    if blk_idx not in saved_pre or blk_idx not in saved_grad:
                        continue  # guard: should not happen

                    pre  = saved_pre[blk_idx].float()   # (*, dim)
                    grad = saved_grad[blk_idx].float()  # (*, dim)

                    for h in range(num_heads):
                        s = slice(h * head_dim, (h + 1) * head_dim)
                        c_ilh = (pre[..., s] * grad[..., s]).sum().item()
                        per_sample_projections[sample_idx, blk_idx * num_heads + h] = (
                            c_ilh * sqrt_omega
                        )

            # Clean up
            model.zero_grad()
            for blk_idx in range(num_blocks):
                for p in model.image_encoder.blocks[blk_idx].attn.parameters():
                    p.requires_grad = False

            sample_idx += 1

    head_importance = np.abs(per_sample_projections).mean(axis=0)
    return head_importance, per_sample_projections


# ==============================================================================
# Baseline scoring (random, magnitude) — unchanged, still used for global ranking
# ==============================================================================

def score_heads_random(num_blocks=12, num_heads=12, seed=42):
    """Random importance scores."""
    return np.random.RandomState(seed).rand(num_blocks * num_heads)


def score_heads_magnitude(model, num_blocks=12, num_heads=12):
    """L2 norm of head qkv + proj weights as importance proxy."""
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
                norm_sq = sum(qkv_w[s, :].norm().item() ** 2 for s in [q_s, k_s, v_s])
                norm_sq += proj_w[:, p_s].norm().item() ** 2
                scores[head_id] = np.sqrt(norm_sq)
    return scores


def score_heads_pointwise(per_sample_projections):
    """
    Legacy global-ranking Pointwise: mean |c_{i,l,h}| per head.
    Kept for diagnostics / comparison with blockwise version.
    """
    return np.abs(per_sample_projections).mean(axis=0)


# ==============================================================================
# Sinkhorn (log-domain, unchanged)
# ==============================================================================

def _logsumexp(a, axis=None):
    a_max  = np.max(a, axis=axis, keepdims=True)
    result = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return result.squeeze(axis=axis)


def _sinkhorn_distance(x, y, epsilon=0.05, n_iter=200):
    """
    Entropy-regularised W_{2,ε}² between two 1D empirical distributions (log-domain).
    """
    n, m  = len(x), len(y)
    C     = (x[:, None] - y[None, :]) ** 2
    log_K = -C / epsilon
    log_a = np.full(n, -np.log(n))
    log_b = np.full(m, -np.log(m))
    log_u = np.zeros(n)
    log_v = np.zeros(m)
    for _ in range(n_iter):
        log_u = log_a - _logsumexp(log_K + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp(log_K + log_u[:, None], axis=0)
    log_pi = log_u[:, None] + log_K + log_v[None, :]
    pi     = np.exp(log_pi)
    return float((C * pi).sum())


# ==============================================================================
# V2.2 — Block-level evaluation functions
# ==============================================================================

def _median_pairwise_dist(vals):
    """
    Compute median |T_i - T_j| over all i < j pairs.
    Used for per-block relative epsilon ε_l = α × this value.
    """
    n = len(vals)
    diffs = np.abs(vals[:, None] - vals[None, :])
    idx   = np.triu_indices(n, k=1)
    return float(np.median(diffs[idx]))


def _eval_block_pointwise(c_block, mask_z):
    """
    Q_pw(z) = mean_i (T_{i,l}(z) - T_{i,l})²

    T_{i,l}    = Σ_h c_{i,l,h}       (all heads, teacher)
    T_{i,l}(z) = Σ_{h: z_h=1} c_{i,l,h}  (kept heads)

    Args:
        c_block : (n_samples, num_heads) float32
        mask_z  : (num_heads,) binary float32 — 1=keep, 0=remove
    """
    T_teacher = c_block.sum(axis=1)
    T_pruned  = (c_block * mask_z).sum(axis=1)
    return float(((T_pruned - T_teacher) ** 2).mean())


def _eval_block_ewr(c_block, mask_z, epsilon, n_iter=200):
    """
    Q_ewr(z) = W_{2,ε}²({T_{i,l}(z)}, {T_{i,l}})

    Args:
        c_block : (n_samples, num_heads) float32
        mask_z  : (num_heads,) binary float32
        epsilon : Sinkhorn regularisation (per-block)
    """
    T_teacher = c_block.sum(axis=1)
    T_pruned  = (c_block * mask_z).sum(axis=1)
    return _sinkhorn_distance(T_pruned, T_teacher, epsilon=epsilon, n_iter=n_iter)


# ==============================================================================
# V2.3 — Block-wise exhaustive search
# ==============================================================================

def generate_head_mask_blockwise(
    per_sample_proj,
    sparsity,
    method,
    epsilon=None,
    alpha=None,
    n_iter=200,
    num_blocks=12,
    num_heads=12,
    verbose=True,
):
    """
    Generate head mask via per-block exhaustive search over C(H, k) candidates.

    For each block l:
      1. k_remove = floor(num_heads * sparsity)  [uniform cross-block budget]
      2. Enumerate all C(num_heads, k_remove) candidate "keep" index sets
      3. Evaluate Q(z_l) for each candidate
      4. Select argmin → block mask

    Method 'pointwise':  Q_pw(z_l) = mean_i (T_{i,l}(z_l) - T_{i,l})²
    Method 'ewr':        Q_ewr(z_l) = W_{2,ε_l}²({T_{i,l}(z_l)}, {T_{i,l}})
                         with ε_l = alpha × median_pairwise_dist({T_{i,l}})

    Candidate counts:
        30% sparsity → C(12,4) = 495 per block
        50% sparsity → C(12,6) = 924 per block
        70% sparsity → C(12,8) = 495 per block

    Args:
        per_sample_proj : (n_samples, total_heads) — c_{i,l,h} values
        sparsity        : fraction of heads to remove per block
        method          : 'pointwise' or 'ewr'
        epsilon         : absolute epsilon (overrides alpha if set)
        alpha           : relative epsilon factor; default 0.5 for EWR
        n_iter          : Sinkhorn iterations (EWR only)
        verbose         : show tqdm progress bar

    Returns:
        head_mask : (total_heads,) float32 — 1=keep, 0=remove
    """
    assert method in ('pointwise', 'ewr'), f"Unknown method: {method!r}"
    assert 0.0 < sparsity < 1.0

    if method == 'ewr' and alpha is None and epsilon is None:
        alpha = 0.5  # default relative scale

    k_remove   = int(num_heads * sparsity)
    n_keep     = num_heads - k_remove
    candidates = list(combinations(range(num_heads), n_keep))

    head_mask = np.ones(num_blocks * num_heads, dtype=np.float32)

    block_iter = (
        tqdm(range(num_blocks), desc=f"Blockwise exhaustive ({method})")
        if verbose else range(num_blocks)
    )

    for blk_idx in block_iter:
        c_block   = per_sample_proj[:, blk_idx * num_heads: (blk_idx + 1) * num_heads]
        T_teacher = c_block.sum(axis=1)  # (n_samples,)

        # Per-block epsilon for EWR
        if method == 'ewr':
            if epsilon is not None:
                eps_l = float(epsilon)
            else:
                med = _median_pairwise_dist(T_teacher)
                eps_l = alpha * med
                if eps_l < 1e-10:
                    # Fallback: std-based scale (avoids degenerate zero epsilon)
                    eps_l = max(alpha * float(np.std(T_teacher)), 1e-8)

        best_q    = float('inf')
        best_cand = candidates[0]

        for keep_idx in candidates:
            mask_z       = np.zeros(num_heads, dtype=np.float32)
            mask_z[list(keep_idx)] = 1.0

            if method == 'pointwise':
                q = _eval_block_pointwise(c_block, mask_z)
            else:
                q = _eval_block_ewr(c_block, mask_z, eps_l, n_iter=n_iter)

            if q < best_q:
                best_q    = q
                best_cand = keep_idx

        block_mask               = np.zeros(num_heads, dtype=np.float32)
        block_mask[list(best_cand)] = 1.0
        head_mask[blk_idx * num_heads: (blk_idx + 1) * num_heads] = block_mask

    return head_mask


# ==============================================================================
# V2.5 — Epsilon sensitivity diagnostic (Check 3)
# ==============================================================================

def diagnose_epsilon_sensitivity(
    per_sample_proj,
    sparsity,
    alpha_values=(0.1, 0.5, 1.0),
    num_blocks=12,
    num_heads=12,
    n_iter=200,
):
    """
    Check 3: For different α, compare block-level optimal mask selections.

    Runs per-block exhaustive EWR for each α and computes:
      - Selected mask per block per α
      - Pairwise block-mask overlap across α pairs

    Returns:
        masks_by_alpha  : dict {alpha: head_mask (total_heads,)}
        overlap_matrix  : (len(alpha_values), len(alpha_values)) — fraction of
                          blocks where both alphas select the same mask
    """
    masks_by_alpha = {}
    for alpha in tqdm(alpha_values, desc="Check 3: ε sensitivity"):
        masks_by_alpha[alpha] = generate_head_mask_blockwise(
            per_sample_proj, sparsity, method='ewr', alpha=alpha,
            n_iter=n_iter, num_blocks=num_blocks, num_heads=num_heads,
            verbose=False,
        )

    n = len(alpha_values)
    overlap_matrix = np.zeros((n, n))
    for i, a1 in enumerate(alpha_values):
        for j, a2 in enumerate(alpha_values):
            m1 = masks_by_alpha[a1].reshape(num_blocks, num_heads)
            m2 = masks_by_alpha[a2].reshape(num_blocks, num_heads)
            same = [np.array_equal(m1[b], m2[b]) for b in range(num_blocks)]
            overlap_matrix[i, j] = np.mean(same)

    return masks_by_alpha, overlap_matrix


# ==============================================================================
# Global-ranking mask generation (random / magnitude)
# ==============================================================================

def generate_head_mask(scores, sparsity, num_blocks=12, num_heads=12):
    """
    Remove the (sparsity × total_heads) lowest-scored heads globally.
    Used for random and magnitude methods.
    """
    total_heads = num_blocks * num_heads
    n_remove    = int(total_heads * sparsity)
    sorted_idx  = np.argsort(scores)
    mask        = np.ones(total_heads, dtype=np.float32)
    mask[sorted_idx[:n_remove]] = 0.0
    return mask


# ==============================================================================
# P0.1 — Head mask application (unchanged from v1)
# ==============================================================================

def apply_head_mask_to_model(model, head_mask, num_blocks=12, num_heads=12):
    """
    Apply head mask via forward_pre_hook on attn.proj (BEFORE W_proj).

    Zeros out channel slice [h*head_dim : (h+1)*head_dim] for each pruned head h,
    which is the semantically correct point to disable a head's contribution.
    """
    hooks = []
    for blk_idx in range(num_blocks):
        attn_module  = model.image_encoder.blocks[blk_idx].attn
        block_mask   = head_mask[blk_idx * num_heads: (blk_idx + 1) * num_heads]
        head_dim     = attn_module.qkv.weight.shape[1] // num_heads
        channel_mask = np.repeat(block_mask, head_dim).astype(np.float32)

        def make_pre_hook(ch_mask):
            def hook_fn(module, inp):
                x = inp[0]
                m = torch.tensor(ch_mask, dtype=x.dtype, device=x.device)
                return (x * m,)
            return hook_fn

        hook = attn_module.proj.register_forward_pre_hook(
            make_pre_hook(channel_mask)
        )
        hooks.append(hook)
    return hooks


def remove_hooks(hooks):
    """Remove all registered hook handles."""
    for h in hooks:
        h.remove()
