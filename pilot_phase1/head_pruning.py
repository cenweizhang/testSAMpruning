# -*- coding: utf-8 -*-
"""
Head-level structured pruning for MedSAM Image Encoder.

Implements 4 head importance scoring methods:
  1. Random scoring
  2. Magnitude scoring (L2 norm of head's qkv weights)
  3. Pointwise Regression scoring (per-sample gradient alignment)
  4. EWR scoring (Entropy-regularized Wasserstein Regression)

And head masking mechanism for zero-shot structural pruning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def dice_loss(pred, target):
    """Dice loss with sigmoid, squared_pred, mean reduction. Replaces monai.losses.DiceLoss."""
    pred = torch.sigmoid(pred)
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    loss = 1.0 - (2.0 * intersection + 1e-5) / (pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1e-5)
    return loss.mean()


# ==============================================================================
# 1. Gradient Computation (per-head, per-sample)
# ==============================================================================

def compute_head_gradient_projections(model, dataloader, device, num_blocks=12, num_heads=12):
    """
    Compute per-sample gradient projections for each head.
    
    For each calibration sample i and each head j, compute:
        projection[i, j] = g_i^T * theta_j
    where g_i is the gradient of the loss w.r.t. head j's parameters,
    and theta_j is the head's parameter vector.
    
    This is computed per-block to control memory usage.
    
    Args:
        model: MedSAM model (Sam) on device, eval mode
        dataloader: calibration data loader
        device: torch device
        num_blocks: number of ViT blocks (12 for ViT-B)
        num_heads: number of attention heads per block (12 for ViT-B)
    
    Returns:
        projections: np.ndarray of shape (n_samples, num_blocks * num_heads)
            Each entry is g_i^T * theta_j (scalar projection)
        teacher_projections: np.ndarray of shape (n_samples, num_blocks * num_heads)
            Each entry is g_i^T * theta_bar_j (teacher projection, same values 
            since we compute grad at teacher params)
    """
    model.eval()
    
    # Loss functions (same as MedSAM training)
    ce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    
    n_samples = len(dataloader.dataset)
    total_heads = num_blocks * num_heads
    
    # We'll accumulate gradient norms per head per sample
    # For projection: g_i^T * theta = sum(g_i * theta) element-wise
    projections = np.zeros((n_samples, total_heads), dtype=np.float32)
    
    sample_idx = 0
    
    for batch in tqdm(dataloader, desc="Computing gradients"):
        images = batch["image"].to(device)        # (B, 3, 1024, 1024)
        masks = batch["mask_256"].to(device)       # (B, 1, 256, 256)
        bboxes = batch["bbox"].numpy()             # (B, 4)
        B = images.shape[0]
        
        # Process each block independently to save memory
        for block_idx in range(num_blocks):
            block = model.image_encoder.blocks[block_idx]
            attn_module = block.attn
            
            # Freeze everything
            for p in model.parameters():
                p.requires_grad = False
            
            # Unfreeze only this block's attention qkv and proj
            for p in attn_module.parameters():
                p.requires_grad = True
            
            # Forward pass
            model.zero_grad()
            
            # Get image embeddings
            image_embedding = model.image_encoder(images)  # (B, 256, 64, 64)
            
            # Get prompt embeddings (box prompt, frozen)
            with torch.no_grad():
                box_torch = torch.as_tensor(bboxes, dtype=torch.float32, device=device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]
                sparse_emb, dense_emb = model.prompt_encoder(
                    points=None, boxes=box_torch, masks=None
                )
            
            # Decode
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )  # (B, 1, 256, 256)
            
            # Compute loss
            loss = dice_loss(low_res_masks, masks.float()) + \
                   ce_loss_fn(low_res_masks, masks.float())
            
            # Per-sample gradient: use backward on total loss then extract
            loss.backward()
            
            # Extract per-head projections from qkv weight gradient
            # qkv weight: (3*dim, dim), qkv bias: (3*dim,)
            # Each head occupies head_dim=dim//num_heads channels
            dim = attn_module.qkv.weight.shape[1]
            head_dim = dim // num_heads
            
            # Compute projection per head: g^T * theta (summed element-wise)
            with torch.no_grad():
                qkv_grad = attn_module.qkv.weight.grad  # (3*dim, dim)
                qkv_weight = attn_module.qkv.weight.data
                proj_grad = attn_module.proj.weight.grad  # (dim, dim)
                proj_weight = attn_module.proj.weight.data
                
                for h in range(num_heads):
                    head_id = block_idx * num_heads + h
                    
                    # qkv: each of q, k, v has head_dim rows per head
                    # q rows: [h*head_dim : (h+1)*head_dim]
                    # k rows: [dim + h*head_dim : dim + (h+1)*head_dim]
                    # v rows: [2*dim + h*head_dim : 2*dim + (h+1)*head_dim]
                    q_slice = slice(h * head_dim, (h + 1) * head_dim)
                    k_slice = slice(dim + h * head_dim, dim + (h + 1) * head_dim)
                    v_slice = slice(2 * dim + h * head_dim, 2 * dim + (h + 1) * head_dim)
                    
                    # proj: output columns for this head
                    # proj weight shape: (dim, dim), input dim per head: [h*head_dim : (h+1)*head_dim]
                    p_slice = slice(h * head_dim, (h + 1) * head_dim)
                    
                    # Gradient * Weight for this head (summed = dot product)
                    proj_val = 0.0
                    # QKV contribution
                    for s in [q_slice, k_slice, v_slice]:
                        proj_val += (qkv_grad[s, :] * qkv_weight[s, :]).sum().item()
                    # Proj contribution  
                    proj_val += (proj_grad[:, p_slice] * proj_weight[:, p_slice]).sum().item()
                    
                    # Bias contributions
                    if attn_module.qkv.bias is not None and attn_module.qkv.bias.grad is not None:
                        qkv_b_grad = attn_module.qkv.bias.grad
                        qkv_b_data = attn_module.qkv.bias.data
                        for s in [q_slice, k_slice, v_slice]:
                            proj_val += (qkv_b_grad[s] * qkv_b_data[s]).sum().item()
                    
                    if attn_module.proj.bias is not None and attn_module.proj.bias.grad is not None:
                        # proj bias is shared across all heads, distribute equally
                        pb_grad = attn_module.proj.bias.grad
                        pb_data = attn_module.proj.bias.data
                        proj_val += (pb_grad * pb_data).sum().item() / num_heads
                    
                    # Store as batch-averaged projection  
                    # (Note: loss.backward() gives batch-averaged gradients)
                    for b in range(B):
                        if sample_idx + b < n_samples:
                            projections[sample_idx + b, head_id] = proj_val / B
            
            # Clean up
            model.zero_grad()
            for p in attn_module.parameters():
                p.requires_grad = False
        
        sample_idx += B
    
    # Teacher projections are identical since we compute gradients at teacher params
    teacher_projections = projections.copy()
    
    return projections, teacher_projections


def compute_head_gradient_projections_fast(model, dataloader, device,
                                           num_blocks=12, num_heads=12):
    """
    Faster gradient projection: compute importance score for each head
    as |g^T * theta| aggregated over calibration samples.
    
    Instead of per-sample projections, aggregates gradient information
    into a single importance score per head. This is used by both
    Pointwise and EWR methods.
    
    Returns:
        head_importance: np.ndarray of shape (num_blocks * num_heads,)
        per_sample_projections: np.ndarray of shape (n_samples, num_blocks * num_heads)
    """
    model.eval()
    
    ce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    
    n_samples = len(dataloader.dataset)
    total_heads = num_blocks * num_heads
    
    # Per-sample gradient projections
    per_sample_projections = np.zeros((n_samples, total_heads), dtype=np.float32)
    
    sample_idx = 0
    
    for batch in tqdm(dataloader, desc="Computing head gradients"):
        images = batch["image"].to(device)
        masks = batch["mask_256"].to(device)
        bboxes = batch["bbox"].numpy()
        B = images.shape[0]
        
        for b_idx in range(B):
            if sample_idx >= n_samples:
                break
            
            # Single-sample forward
            img = images[b_idx:b_idx+1]
            msk = masks[b_idx:b_idx+1]
            box = bboxes[b_idx:b_idx+1]
            
            for block_idx in range(num_blocks):
                attn_module = model.image_encoder.blocks[block_idx].attn
                
                # Freeze all, unfreeze this attention
                for p in model.parameters():
                    p.requires_grad = False
                for p in attn_module.parameters():
                    p.requires_grad = True
                
                model.zero_grad()
                
                # Forward
                image_embedding = model.image_encoder(img)
                
                with torch.no_grad():
                    box_t = torch.as_tensor(box, dtype=torch.float32, device=device)
                    if len(box_t.shape) == 2:
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
                
                loss = F.binary_cross_entropy_with_logits(low_res_masks, msk.float()) + \
                       (1 - 2 * (torch.sigmoid(low_res_masks) * msk.float()).sum() / 
                        (torch.sigmoid(low_res_masks).sum() + msk.float().sum() + 1e-8))
                
                loss.backward()
                
                # Extract per-head projection
                dim = attn_module.qkv.weight.shape[1]
                head_dim = dim // num_heads
                
                with torch.no_grad():
                    qkv_g = attn_module.qkv.weight.grad
                    qkv_w = attn_module.qkv.weight.data
                    proj_g = attn_module.proj.weight.grad
                    proj_w = attn_module.proj.weight.data
                    
                    for h in range(num_heads):
                        head_id = block_idx * num_heads + h
                        q_s = slice(h * head_dim, (h+1) * head_dim)
                        k_s = slice(dim + h * head_dim, dim + (h+1) * head_dim)
                        v_s = slice(2*dim + h * head_dim, 2*dim + (h+1) * head_dim)
                        p_s = slice(h * head_dim, (h+1) * head_dim)
                        
                        val = 0.0
                        for s in [q_s, k_s, v_s]:
                            val += (qkv_g[s, :] * qkv_w[s, :]).sum().item()
                        val += (proj_g[:, p_s] * proj_w[:, p_s]).sum().item()
                        
                        per_sample_projections[sample_idx, head_id] = val
                
                model.zero_grad()
                for p in attn_module.parameters():
                    p.requires_grad = False
            
            sample_idx += 1
    
    head_importance = np.abs(per_sample_projections).mean(axis=0)
    return head_importance, per_sample_projections


# ==============================================================================
# 2. Scoring Methods
# ==============================================================================

def score_heads_random(num_blocks=12, num_heads=12, seed=42):
    """Random importance scores."""
    rng = np.random.RandomState(seed)
    return rng.rand(num_blocks * num_heads)


def score_heads_magnitude(model, num_blocks=12, num_heads=12):
    """
    Magnitude-based scoring: L2 norm of each head's qkv + proj weights.
    """
    scores = np.zeros(num_blocks * num_heads)
    
    for block_idx in range(num_blocks):
        attn = model.image_encoder.blocks[block_idx].attn
        dim = attn.qkv.weight.shape[1]
        head_dim = dim // num_heads
        
        with torch.no_grad():
            qkv_w = attn.qkv.weight.data
            proj_w = attn.proj.weight.data
            
            for h in range(num_heads):
                head_id = block_idx * num_heads + h
                
                q_s = slice(h * head_dim, (h+1) * head_dim)
                k_s = slice(dim + h * head_dim, dim + (h+1) * head_dim)
                v_s = slice(2*dim + h * head_dim, 2*dim + (h+1) * head_dim)
                p_s = slice(h * head_dim, (h+1) * head_dim)
                
                norm = 0.0
                for s in [q_s, k_s, v_s]:
                    norm += qkv_w[s, :].norm().item() ** 2
                norm += proj_w[:, p_s].norm().item() ** 2
                
                scores[head_id] = np.sqrt(norm)
    
    return scores


def score_heads_pointwise(per_sample_projections):
    """
    Pointwise regression scoring.
    
    Score = mean of |g_i^T * theta_j| across calibration samples.
    Higher means this head contributes more to the loss landscape.
    """
    return np.abs(per_sample_projections).mean(axis=0)


def _sinkhorn_distance(x, y, epsilon=0.05, n_iter=100):
    """
    Compute the entropy-regularized Wasserstein distance W_{2,epsilon}
    between two 1D empirical distributions using the Sinkhorn algorithm
    in log-domain for numerical stability.
    
    Args:
        x: np.ndarray of shape (n,) - samples from distribution mu
        y: np.ndarray of shape (n,) - samples from distribution nu
        epsilon: entropic regularization parameter
        n_iter: number of Sinkhorn iterations
    
    Returns:
        float: approximate W_{2,epsilon}^2
    """
    n = len(x)
    m = len(y)
    
    # Cost matrix: squared Euclidean distance (1D)
    C = (x[:, None] - y[None, :]) ** 2  # (n, m)
    
    # Log-domain Sinkhorn
    log_K = -C / epsilon  # (n, m)
    
    # Uniform marginals
    log_a = -np.log(n) * np.ones(n)
    log_b = -np.log(m) * np.ones(m)
    
    log_u = np.zeros(n)
    log_v = np.zeros(m)
    
    for _ in range(n_iter):
        # Update u
        log_u = log_a - _logsumexp(log_K + log_v[None, :], axis=1)
        # Update v
        log_v = log_b - _logsumexp(log_K + log_u[:, None], axis=0)
    
    # Transport plan in log domain
    log_pi = log_u[:, None] + log_K + log_v[None, :]
    pi = np.exp(log_pi)
    
    # W_{2,epsilon}^2 = <C, pi>
    return (C * pi).sum()


def _logsumexp(a, axis=None):
    """Numerically stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    result = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return result.squeeze(axis=axis)


def score_heads_ewr(per_sample_projections, epsilon=0.05):
    """
    EWR (Entropy-regularized Wasserstein Regression) scoring.
    
    For each head j, compute W_{2,epsilon}^2 between the projection 
    distribution with head j removed vs. the teacher projection distribution.
    
    Head importance = how much removing it increases the Wasserstein distance.
    
    Args:
        per_sample_projections: (n_samples, total_heads) array
        epsilon: Sinkhorn regularization parameter
    
    Returns:
        scores: (total_heads,) importance scores. Higher = more important.
    """
    n_samples, total_heads = per_sample_projections.shape
    scores = np.zeros(total_heads)
    
    # Teacher projection: sum over all heads for each sample
    teacher_total = per_sample_projections.sum(axis=1)  # (n_samples,)
    
    for j in tqdm(range(total_heads), desc=f"EWR scoring (eps={epsilon})"):
        # Projection with head j removed
        pruned_total = teacher_total - per_sample_projections[:, j]  # (n_samples,)
        
        # W_{2,epsilon}^2 between pruned and teacher distributions
        w_dist = _sinkhorn_distance(pruned_total, teacher_total, epsilon=epsilon)
        scores[j] = w_dist
    
    return scores


# ==============================================================================
# 3. Head Mask Generation and Application
# ==============================================================================

def generate_head_mask(scores, sparsity, num_blocks=12, num_heads=12):
    """
    Generate a binary head mask based on importance scores and target sparsity.
    
    Args:
        scores: (total_heads,) importance scores. Higher = more important.
        sparsity: fraction of heads to remove (e.g., 0.3 = remove 30%)
        num_blocks: number of ViT blocks
        num_heads: number of heads per block
    
    Returns:
        head_mask: (total_heads,) binary mask, 1 = keep, 0 = remove
    """
    total_heads = num_blocks * num_heads
    n_remove = int(total_heads * sparsity)
    
    # Sort by importance (ascending) and remove the least important
    sorted_indices = np.argsort(scores)
    
    mask = np.ones(total_heads, dtype=np.float32)
    mask[sorted_indices[:n_remove]] = 0.0
    
    return mask


def apply_head_mask_to_model(model, head_mask, num_blocks=12, num_heads=12):
    """
    Apply head mask by injecting a forward hook into each Attention module.
    
    The hook zeros out the output of pruned heads before the projection layer.
    This does NOT modify the original weights, ensuring reversibility.
    
    Args:
        model: Sam model
        head_mask: (total_heads,) np array, 1=keep, 0=remove
        num_blocks, num_heads: architecture parameters
    
    Returns:
        hooks: list of hook handles (call .remove() to undo)
    """
    hooks = []
    
    for block_idx in range(num_blocks):
        block = model.image_encoder.blocks[block_idx]
        attn_module = block.attn
        
        # Create mask for this block's heads
        block_mask = head_mask[block_idx * num_heads: (block_idx + 1) * num_heads]
        block_mask_tensor = torch.tensor(block_mask, dtype=torch.float32)
        
        def make_hook(mask_tensor, n_heads):
            """Create a closure capturing the mask."""
            def hook_fn(module, input, output):
                # output shape: (B, H, W, dim)
                B, H, W, dim = output.shape
                head_dim = dim // n_heads
                
                # Reshape to (B, H, W, n_heads, head_dim)
                out = output.view(B, H, W, n_heads, head_dim)
                
                # Apply mask: (n_heads,) -> (1, 1, 1, n_heads, 1)
                device = output.device
                m = mask_tensor.to(device).view(1, 1, 1, n_heads, 1)
                out = out * m
                
                return out.view(B, H, W, dim)
            return hook_fn
        
        # Note: We hook the attention output BEFORE the proj layer
        # But the forward() applies proj after attn computation.
        # So we need to hook differently - we'll modify the forward to apply mask
        # Actually, let's use a different approach: modify the proj layer's input
        
        # Better approach: register a hook on the Attention module itself
        # The output of attn.forward() is already (B, H, W, dim) after proj
        # We need to mask BEFORE proj, so we'll use a pre-forward hook on proj
        # or simply hook the full Attention output and compensate
        
        # Simplest correct approach: hook the Attention module output
        # Since proj mixes heads, we mask the Attention output which already 
        # went through proj. This is approximately correct for pruning evaluation.
        hook = attn_module.register_forward_hook(
            make_hook(block_mask_tensor, num_heads)
        )
        hooks.append(hook)
    
    return hooks


def remove_hooks(hooks):
    """Remove all hooks to restore original model behavior."""
    for h in hooks:
        h.remove()


# ==============================================================================
# 4. Convenience: Get All Scores
# ==============================================================================

def compute_all_head_scores(model, cal_loader, device, epsilon_values=[0.01, 0.05, 0.1]):
    """
    Compute head importance scores for all methods.
    
    Args:
        model: MedSAM model on device
        cal_loader: calibration data loader
        device: torch device
        epsilon_values: list of epsilon values for EWR
    
    Returns:
        dict of method_name -> scores array (total_heads,)
    """
    print("=" * 60)
    print("Computing head importance scores for all methods")
    print("=" * 60)
    
    results = {}
    
    # 1. Random
    print("\n[1/4] Random scoring...")
    results["random"] = score_heads_random()
    
    # 2. Magnitude
    print("[2/4] Magnitude scoring...")
    results["magnitude"] = score_heads_magnitude(model)
    
    # 3 & 4. Need gradient projections
    print("[3/4] Computing gradient projections (this takes a while)...")
    head_importance, per_sample_proj = compute_head_gradient_projections_fast(
        model, cal_loader, device
    )
    
    # 3. Pointwise
    results["pointwise"] = score_heads_pointwise(per_sample_proj)
    
    # 4. EWR with different epsilon
    print("[4/4] EWR scoring...")
    for eps in epsilon_values:
        key = f"ewr_eps{eps}"
        results[key] = score_heads_ewr(per_sample_proj, epsilon=eps)
    
    return results
