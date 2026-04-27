# -*- coding: utf-8 -*-
"""
Head mask generation and application for the dual-intervention pilot study.

Provides:
  - Baseline scorers : random, magnitude
  - Mask generator   : generate_head_mask (global greedy ranking)
  - Mask applier     : apply_head_mask_to_model (pre-hook on attn.proj)
  - Hook remover     : remove_hooks
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Baseline scorers
# ---------------------------------------------------------------------------

def score_heads_random(num_blocks=12, num_heads=12, seed=42):
    """Uniform-random importance scores (lower = prune first after inversion)."""
    rng = np.random.RandomState(seed)
    return rng.rand(num_blocks * num_heads).astype(np.float32)


def score_heads_magnitude(model, num_blocks=12, num_heads=12):
    """
    L2-norm of each head's weight tensors as an importance proxy.
    Group = qkv rows + proj cols (same definition as the dual-intervention group).
    """
    scores = np.zeros(num_blocks * num_heads, dtype=np.float32)
    for l in range(num_blocks):
        attn     = model.image_encoder.blocks[l].attn
        dim      = attn.qkv.weight.shape[1]
        head_dim = dim // num_heads
        with torch.no_grad():
            qkv_w  = attn.qkv.weight.data.float()
            proj_w = attn.proj.weight.data.float()
            has_b  = attn.qkv.bias is not None
            if has_b:
                qkv_b = attn.qkv.bias.data.float()
            for h in range(num_heads):
                head_id = l * num_heads + h
                q_rows = slice(h * head_dim,           (h + 1) * head_dim)
                k_rows = slice(dim + h * head_dim,     dim + (h + 1) * head_dim)
                v_rows = slice(2 * dim + h * head_dim, 2 * dim + (h + 1) * head_dim)
                p_cols = slice(h * head_dim,           (h + 1) * head_dim)

                parts = [
                    qkv_w[q_rows, :], qkv_w[k_rows, :], qkv_w[v_rows, :],
                    proj_w[:, p_cols],
                ]
                if has_b:
                    parts += [qkv_b[q_rows], qkv_b[k_rows], qkv_b[v_rows]]
                norm_sq = sum(t.pow(2).sum().item() for t in parts)
                scores[head_id] = float(norm_sq ** 0.5)
    return scores


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def generate_head_mask(scores, sparsity, num_blocks=12, num_heads=12):
    """
    Global greedy ranking: remove the lowest-scored fraction of heads.

    Args:
        scores   : (total_heads,) importance scores — HIGHER means more important.
                   For dual-intervention scores (lower = prune first), pass
                   -scores or use negate=True.
        sparsity : fraction of heads to remove in [0, 1).

    Returns:
        mask : (total_heads,) float32 — 1 = keep, 0 = remove.
    """
    total_heads = num_blocks * num_heads
    n_remove    = int(round(total_heads * sparsity))
    n_remove    = max(0, min(n_remove, total_heads - 1))

    sorted_idx  = np.argsort(scores)        # ascending: smallest score first
    mask        = np.ones(total_heads, dtype=np.float32)
    mask[sorted_idx[:n_remove]] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Mask application via forward pre-hook
# ---------------------------------------------------------------------------

def apply_head_mask_to_model(model, head_mask, num_blocks=12, num_heads=12):
    """
    Zero out pruned-head channels in the input to attn.proj via a pre-hook.

    The hook fires BEFORE W_proj, so the effective contribution of a pruned
    head is zeroed without modifying any weight tensors.

    Returns:
        hooks : list of hook handles (pass to remove_hooks when done)
    """
    hooks = []
    for l in range(num_blocks):
        attn_module = model.image_encoder.blocks[l].attn
        head_dim    = attn_module.qkv.weight.shape[1] // num_heads
        block_mask  = head_mask[l * num_heads: (l + 1) * num_heads]  # (num_heads,)

        # Expand to channel-level mask: each head occupies head_dim channels
        channel_mask = np.repeat(block_mask, head_dim).astype(np.float32)  # (dim,)

        def _make_hook(ch_mask):
            def hook_fn(module, inp):
                x = inp[0]
                m = torch.tensor(ch_mask, dtype=x.dtype, device=x.device)
                return (x * m,)
            return hook_fn

        hook = attn_module.proj.register_forward_pre_hook(_make_hook(channel_mask))
        hooks.append(hook)
    return hooks


def remove_hooks(hooks):
    """Remove all registered forward pre-hooks."""
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# MLP neuron scoring baseline
# ---------------------------------------------------------------------------

def score_mlp_magnitude(model, num_blocks=12):
    """
    L2-norm of each MLP neuron's weight group as an importance proxy.
    Group = lin1 row + lin2 col (same definition as dual-intervention group).
    """
    mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]
    scores  = np.zeros(num_blocks * mlp_dim, dtype=np.float32)
    for l in range(num_blocks):
        mlp = model.image_encoder.blocks[l].mlp
        with torch.no_grad():
            l1w    = mlp.lin1.weight.data.float()   # (mlp_dim, embed_dim)
            l2w    = mlp.lin2.weight.data.float()   # (embed_dim, mlp_dim)
            norm_sq = l1w.pow(2).sum(dim=1) + l2w.pow(2).sum(dim=0)
            if mlp.lin1.bias is not None:
                norm_sq = norm_sq + mlp.lin1.bias.data.float().pow(2)
            scores[l * mlp_dim: (l + 1) * mlp_dim] = norm_sq.sqrt().cpu().numpy()
    return scores


# ---------------------------------------------------------------------------
# MLP neuron mask generation
# ---------------------------------------------------------------------------

def generate_neuron_mask(scores, sparsity, num_blocks=12, mlp_dim=3072):
    """
    Global greedy ranking over MLP neurons: remove lowest-scored fraction.

    Args:
        scores   : (num_blocks * mlp_dim,) importance — HIGHER = more important.
                   For dual-intervention Q_g scores pass directly (lower = prune).
        sparsity : fraction to remove in [0, 1).

    Returns:
        mask : (num_blocks * mlp_dim,) float32 — 1=keep, 0=remove.
    """
    total  = num_blocks * mlp_dim
    n_rem  = max(0, min(int(round(total * sparsity)), total - 1))
    idx    = np.argsort(scores)
    mask   = np.ones(total, dtype=np.float32)
    mask[idx[:n_rem]] = 0.0
    return mask


# ---------------------------------------------------------------------------
# MLP neuron mask application via forward pre-hook
# ---------------------------------------------------------------------------

def apply_mlp_mask_to_model(model, neuron_mask, num_blocks=12):
    """
    Zero out pruned-neuron channels in the input to mlp.lin2 via a pre-hook.

    The hook fires BEFORE lin2, zeroing GELU(lin1(x)) channels that correspond
    to pruned neurons without modifying any weight tensors.

    Args:
        model       : MedSAM model.
        neuron_mask : (num_blocks * mlp_dim,) float32, 1=keep 0=prune.
        num_blocks  : transformer block count.

    Returns:
        hooks : list of hook handles (pass to remove_hooks when done).
    """
    hooks   = []
    mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]

    for l in range(num_blocks):
        mlp_module = model.image_encoder.blocks[l].mlp
        block_mask = neuron_mask[l * mlp_dim: (l + 1) * mlp_dim].astype(np.float32)

        def _make_hook(ch_mask):
            def hook_fn(module, inp):
                x = inp[0]   # (B, seq_len, mlp_dim)
                m = torch.tensor(ch_mask, dtype=x.dtype, device=x.device)
                return (x * m,)
            return hook_fn

        hook = mlp_module.lin2.register_forward_pre_hook(_make_hook(block_mask))
        hooks.append(hook)
    return hooks


# ---------------------------------------------------------------------------
# Cascade (head + MLP) model statistics
# ---------------------------------------------------------------------------

def compute_cascade_stats(model, head_mask, neuron_mask,
                          num_blocks=12, num_heads=12,
                          img_size=1024, patch_size=16):
    """
    Combined param count and FLOPs for a cascade-pruned model.

    Applies both head_mask (attention) and neuron_mask (MLP) simultaneously.

    Returns
    -------
    dict with keys:
        n_params_total / _remaining / _pruned, param_reduction_pct
        flops_total_G  / flops_remaining_G,    flop_reduction_pct
        n_heads_kept / n_heads_total
        n_neurons_kept / n_neurons_total
        attn_param_reduction_pct, mlp_param_reduction_pct
        attn_flop_reduction_pct,  mlp_flop_reduction_pct
    """
    seq_len            = (img_size // patch_size) ** 2   # 4096
    global_attn_blocks = {2, 5, 8, 11}
    window_size        = 14
    seq_window         = window_size * window_size        # 196
    n_windows          = seq_len / seq_window

    embed   = model.image_encoder.blocks[0].attn.qkv.weight.shape[1]   # 768
    head_dim = embed // num_heads                                         # 64
    mlp_dim  = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]   # 3072

    n_params_total = sum(p.numel() for p in model.image_encoder.parameters())

    # ---- Attention params/FLOPs per head ----
    attn_params = np.zeros(num_blocks * num_heads, dtype=np.int64)
    attn_flops  = np.zeros(num_blocks * num_heads, dtype=np.float64)
    for l in range(num_blocks):
        attn     = model.image_encoder.blocks[l].attn
        per_head = 3 * head_dim * embed + embed * head_dim
        if attn.qkv.bias is not None:
            per_head += 3 * head_dim
        qkv_f  = 6 * seq_len * embed * head_dim
        proj_f = 2 * seq_len * head_dim * embed
        attn_f = (4 * seq_len * seq_len * head_dim
                  if l in global_attn_blocks
                  else 4 * n_windows * seq_window * seq_window * head_dim)
        for h in range(num_heads):
            attn_params[l * num_heads + h] = per_head
            attn_flops[l * num_heads + h]  = qkv_f + attn_f + proj_f

    head_pruned          = (head_mask   == 0) if head_mask   is not None else np.zeros(num_blocks * num_heads, dtype=bool)
    n_attn_params_pruned = int(np.sum(attn_params[head_pruned]))
    n_attn_flops_pruned  = float(np.sum(attn_flops[head_pruned]))

    # ---- MLP params/FLOPs per neuron ----
    has_bias     = model.image_encoder.blocks[0].mlp.lin1.bias is not None
    per_neuron_p = 2 * embed + (1 if has_bias else 0)   # 1537
    per_neuron_f = 4 * seq_len * embed                  # 12,582,912

    neuron_pruned        = (neuron_mask == 0) if neuron_mask is not None else np.zeros(num_blocks * mlp_dim, dtype=bool)
    n_mlp_params_pruned  = int(np.sum(neuron_pruned)) * per_neuron_p
    n_mlp_flops_pruned   = float(np.sum(neuron_pruned)) * per_neuron_f

    # ---- Totals ----
    mlp_flops_total  = float(num_blocks * 2 * 2 * seq_len * embed * mlp_dim)
    attn_flops_total = float(np.sum(attn_flops))
    patch_emb_flops  = float(2 * seq_len * embed * (patch_size ** 2 * 3))
    total_flops      = attn_flops_total + mlp_flops_total + patch_emb_flops
    pruned_flops     = n_attn_flops_pruned + n_mlp_flops_pruned

    n_params_pruned    = n_attn_params_pruned + n_mlp_params_pruned
    n_params_remaining = n_params_total - n_params_pruned
    flops_remaining    = total_flops - pruned_flops

    return {
        "n_params_total":           int(n_params_total),
        "n_params_remaining":       int(n_params_remaining),
        "n_params_pruned":          int(n_params_pruned),
        "param_reduction_pct":      round(100.0 * n_params_pruned   / max(n_params_total, 1), 2),
        "flops_total_G":            round(total_flops    / 1e9, 2),
        "flops_remaining_G":        round(flops_remaining / 1e9, 2),
        "flop_reduction_pct":       round(100.0 * pruned_flops / max(total_flops, 1), 2),
        "n_heads_kept":             int(np.sum(~head_pruned)),
        "n_heads_total":            num_blocks * num_heads,
        "n_neurons_kept":           int(np.sum(~neuron_pruned)),
        "n_neurons_total":          num_blocks * mlp_dim,
        "attn_param_reduction_pct": round(100.0 * n_attn_params_pruned / max(n_params_total, 1), 2),
        "mlp_param_reduction_pct":  round(100.0 * n_mlp_params_pruned  / max(n_params_total, 1), 2),
        "attn_flop_reduction_pct":  round(100.0 * n_attn_flops_pruned  / max(total_flops, 1), 2),
        "mlp_flop_reduction_pct":   round(100.0 * n_mlp_flops_pruned   / max(total_flops, 1), 2),
    }


# ---------------------------------------------------------------------------
# Model statistics: parameter count and FLOPs under a given head mask
# ---------------------------------------------------------------------------

def compute_model_stats(model, head_mask, num_blocks=12, num_heads=12,
                        img_size=1024, patch_size=16):
    """
    Compute structural parameter count and estimated FLOPs under a head mask.

    'Structural' means: parameters / FLOPs that would be eliminated if pruned
    heads were physically removed from the architecture.  The current pilot
    uses pre-hooks (no weight removal), so these figures represent the savings
    achievable with a real structured-pruning implementation.

    FLOPs are counted as multiply-adds × 2 (standard convention).

    SAM ViT-B specifics:
        • 12 blocks, 12 heads, dim=768, head_dim=64
        • Global attention at blocks {2, 5, 8, 11}  (full 4096-token sequence)
        • Window attention at all other blocks        (14×14 = 196 tokens/window)
        • 1024×1024 input → 64×64 = 4 096 patch tokens

    Returns
    -------
    dict with keys:
        n_params_total      : total image-encoder param count
        n_params_remaining  : params after structurally removing pruned heads
        n_params_pruned     : params in pruned head groups
        param_reduction_pct : % of encoder params eliminated
        flops_total_G       : estimated full-model encoder FLOPs (GFLOPs)
        flops_remaining_G   : estimated FLOPs after pruning
        flop_reduction_pct  : % FLOPs eliminated
        n_heads_kept        : number of attention heads kept
        n_heads_total       : total attention heads
    """
    seq_len             = (img_size // patch_size) ** 2   # 4 096
    global_attn_blocks  = {2, 5, 8, 11}
    window_size         = 14
    seq_window          = window_size * window_size        # 196
    n_windows           = seq_len / seq_window             # ≈ 20.9

    # ---- Parameter counting ----------------------------------------
    n_params_total = sum(p.numel() for p in model.image_encoder.parameters())

    head_params = np.zeros(num_blocks * num_heads, dtype=np.int64)
    for l in range(num_blocks):
        attn     = model.image_encoder.blocks[l].attn
        dim      = attn.qkv.weight.shape[1]
        head_dim = dim // num_heads
        per_head = (
            3 * head_dim * dim   # qkv.weight rows  (Q + K + V)
            + dim * head_dim     # proj.weight cols
        )
        if attn.qkv.bias is not None:
            per_head += 3 * head_dim   # qkv.bias entries
        for h in range(num_heads):
            head_params[l * num_heads + h] = per_head

    pruned            = (head_mask == 0)
    n_params_pruned   = int(np.sum(head_params[pruned]))
    n_params_remaining = n_params_total - n_params_pruned

    # ---- FLOPs estimation ------------------------------------------
    # Per head per block:
    #   QKV projection (Q+K+V):  6 × seq × dim × head_dim
    #   Attention scores + AV:   4 × N_w × seq_w² × head_dim
    #   Proj attribution:        2 × seq × head_dim × dim
    head_flops = np.zeros(num_blocks * num_heads, dtype=np.float64)
    for l in range(num_blocks):
        attn     = model.image_encoder.blocks[l].attn
        dim      = attn.qkv.weight.shape[1]
        head_dim = dim // num_heads

        qkv_f  = 6 * seq_len * dim * head_dim
        proj_f = 2 * seq_len * head_dim * dim
        if l in global_attn_blocks:
            attn_f = 4 * seq_len * seq_len * head_dim
        else:
            attn_f = 4 * n_windows * seq_window * seq_window * head_dim

        per_head_f = qkv_f + attn_f + proj_f
        for h in range(num_heads):
            head_flops[l * num_heads + h] = per_head_f

    # Non-attention FLOPs (MLP + patch embedding)
    dim0        = model.image_encoder.blocks[0].attn.qkv.weight.shape[1]
    mlp_f       = num_blocks * 2 * 2 * seq_len * dim0 * (4 * dim0)
    patch_emb_f = (
        2 * seq_len * dim0 * (patch_size * patch_size * 3)
    )

    total_attn_flops   = float(head_flops.sum())
    non_attn_flops     = mlp_f + patch_emb_f
    total_flops        = total_attn_flops + non_attn_flops
    pruned_attn_flops  = float(np.sum(head_flops[pruned]))
    remaining_flops    = total_flops - pruned_attn_flops

    return {
        "n_params_total":      int(n_params_total),
        "n_params_remaining":  int(n_params_remaining),
        "n_params_pruned":     int(n_params_pruned),
        "param_reduction_pct": round(100.0 * n_params_pruned / max(n_params_total, 1), 2),
        "flops_total_G":       round(total_flops    / 1e9, 2),
        "flops_remaining_G":   round(remaining_flops / 1e9, 2),
        "flop_reduction_pct":  round(100.0 * pruned_attn_flops / max(total_flops, 1), 2),
        "n_heads_kept":        int(np.sum(~pruned)),
        "n_heads_total":       num_blocks * num_heads,
    }
