# -*- coding: utf-8 -*-
"""
Phase 1 Experiment Runner: EWR Foundation.

Compares 5 pruning methods at 3 sparsity levels on CVC-ColonDB.

Usage (on server):
    cd /path/to/MedSAM
    python -m pilot_phase1.run_phase1 \
        --data_root assert/CVC-ColonDB \
        --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
        --device cuda:0 \
        --n_calibration 128 \
        --output_dir results/phase1
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np
import torch

# Add parent dir to path for segment_anything imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_anything import sam_model_registry
from pilot_phase1.dataset import build_dataloaders
from pilot_phase1.head_pruning import (
    compute_head_gradient_projections_fast,
    score_heads_random,
    score_heads_magnitude,
    score_heads_pointwise,
    score_heads_ewr,
    generate_head_mask,
)
from pilot_phase1.evaluate import evaluate_pruned_model


def main():
    parser = argparse.ArgumentParser(description="Phase 1: EWR Foundation Experiment")
    parser.add_argument("--data_root", type=str, default="assert/CVC-ColonDB",
                        help="Path to CVC-ColonDB dataset")
    parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth",
                        help="Path to MedSAM checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (cuda:0 or cpu)")
    parser.add_argument("--n_calibration", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for calibration data loading")
    parser.add_argument("--output_dir", type=str, default="results/phase1",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                        help="Sparsity levels to test")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.05, 0.1],
                        help="Epsilon values for EWR Sinkhorn")
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 70)
    print("Phase 1: EWR Foundation Experiment")
    print(f"  Dataset:       {args.data_root}")
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Device:        {device}")
    print(f"  Calibration:   {args.n_calibration} samples")
    print(f"  Sparsities:    {args.sparsities}")
    print(f"  EWR epsilons:  {args.epsilons}")
    print(f"  Output:        {args.output_dir}")
    print("=" * 70)

    # ============================================================
    # 1. Load MedSAM model
    # ============================================================
    print("\n[Step 1] Loading MedSAM model...")
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model = model.to(device)
    model.eval()

    # Freeze prompt encoder and mask decoder (as per pilot study spec)
    for param in model.prompt_encoder.parameters():
        param.requires_grad = False
    for param in model.mask_decoder.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.image_encoder.parameters())
    print(f"  Total params:    {total_params:,}")
    print(f"  Encoder params:  {encoder_params:,}")

    # ============================================================
    # 2. Build data loaders
    # ============================================================
    print("\n[Step 2] Building data loaders...")
    cal_loader, test_loader, _, _ = build_dataloaders(
        args.data_root,
        n_calibration=args.n_calibration,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # ============================================================
    # 3. Evaluate unpruned baseline
    # ============================================================
    print("\n[Step 3] Evaluating unpruned baseline...")
    baseline_results = evaluate_pruned_model(
        model, test_loader, head_mask=None, device=device
    )
    print(f"  Unpruned Dice:   {baseline_results['avg_metrics']['mean_dice']:.4f}")
    print(f"  Unpruned IoU:    {baseline_results['avg_metrics']['mean_iou']:.4f}")
    print(f"  Unpruned BF1:    {baseline_results['avg_metrics']['mean_boundary_f1']:.4f}")
    print(f"  Unpruned HD95:   {baseline_results['avg_metrics']['mean_hd95']:.2f}")

    # ============================================================
    # 4. Compute head importance scores
    # ============================================================
    print("\n[Step 4] Computing head importance scores...")

    # Gradient-based projections (needed for pointwise & EWR)
    t0 = time.time()
    head_importance, per_sample_proj = compute_head_gradient_projections_fast(
        model, cal_loader, device
    )
    grad_time = time.time() - t0
    print(f"  Gradient computation took {grad_time:.1f}s")

    # Compute all scoring methods
    all_scores = {}

    # Random (3 seeds for variance estimation)
    for seed in [42, 123, 456]:
        all_scores[f"random_s{seed}"] = score_heads_random(seed=seed)

    # Magnitude
    all_scores["magnitude"] = score_heads_magnitude(model)

    # Pointwise
    all_scores["pointwise"] = score_heads_pointwise(per_sample_proj)

    # EWR with different epsilon
    for eps in args.epsilons:
        t0 = time.time()
        all_scores[f"ewr_eps{eps}"] = score_heads_ewr(per_sample_proj, epsilon=eps)
        ewr_time = time.time() - t0
        print(f"  EWR (eps={eps}) scoring took {ewr_time:.1f}s")

    # Save scores
    scores_path = os.path.join(args.output_dir, "head_scores.npz")
    np.savez(scores_path, **all_scores)
    print(f"\n  Scores saved to {scores_path}")

    # ============================================================
    # 5. Evaluate all method x sparsity combinations
    # ============================================================
    print("\n[Step 5] Running pruning experiments...")

    all_results = {
        "config": vars(args),
        "baseline": baseline_results["avg_metrics"],
        "experiments": [],
    }

    # Define experiment grid
    methods = ["magnitude", "pointwise"] + [f"ewr_eps{eps}" for eps in args.epsilons]
    # For random, use 3 seeds and average
    random_seeds = [42, 123, 456]

    for sparsity in args.sparsities:
        print(f"\n--- Sparsity: {sparsity*100:.0f}% ---")

        # Random (averaged over 3 seeds)
        random_metrics_list = []
        for seed in random_seeds:
            scores = all_scores[f"random_s{seed}"]
            mask = generate_head_mask(scores, sparsity)
            result = evaluate_pruned_model(model, test_loader, mask, device)
            random_metrics_list.append(result["avg_metrics"])

        # Average random results
        random_avg = {}
        for key in random_metrics_list[0]:
            values = [m[key] for m in random_metrics_list]
            if isinstance(values[0], (int, float, np.integer, np.floating)):
                random_avg[key] = float(np.mean(values))
        random_avg["method"] = "random"
        random_avg["sparsity"] = sparsity
        random_avg["n_seeds"] = len(random_seeds)
        all_results["experiments"].append(random_avg)

        dice_str = f"{random_avg['mean_dice']:.4f}"
        print(f"  random:     Dice={dice_str}  IoU={random_avg['mean_iou']:.4f}  "
              f"BF1={random_avg['mean_boundary_f1']:.4f}  HD95={random_avg['mean_hd95']:.2f}")

        # Other methods
        for method in methods:
            scores = all_scores[method]
            mask = generate_head_mask(scores, sparsity)
            result = evaluate_pruned_model(model, test_loader, mask, device)

            exp_result = result["avg_metrics"].copy()
            exp_result["method"] = method
            exp_result["sparsity"] = sparsity
            all_results["experiments"].append(exp_result)

            m = exp_result
            print(f"  {method:14s}: Dice={m['mean_dice']:.4f}  IoU={m['mean_iou']:.4f}  "
                  f"BF1={m['mean_boundary_f1']:.4f}  HD95={m['mean_hd95']:.2f}")

    # ============================================================
    # 6. Save results
    # ============================================================
    results_path = os.path.join(args.output_dir, "phase1_results.json")

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_results = json.loads(
        json.dumps(all_results, default=convert_to_serializable)
    )
    
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ============================================================
    # 7. Print summary table
    # ============================================================
    print("\n" + "=" * 90)
    print("PHASE 1 RESULTS SUMMARY")
    print("=" * 90)

    # Header
    print(f"{'Method':<16} {'Sparsity':>10} {'Dice':>8} {'IoU':>8} {'BF1':>8} {'HD95':>8}")
    print("-" * 66)

    # Baseline
    b = all_results["baseline"]
    print(f"{'unpruned':<16} {'0%':>10} {b['mean_dice']:>8.4f} {b['mean_iou']:>8.4f} "
          f"{b['mean_boundary_f1']:>8.4f} {b['mean_hd95']:>8.2f}")
    print("-" * 66)

    # Experiments grouped by sparsity
    for sparsity in args.sparsities:
        exps = [e for e in all_results["experiments"] if e["sparsity"] == sparsity]
        for e in exps:
            print(f"{e['method']:<16} {sparsity*100:>9.0f}% {e['mean_dice']:>8.4f} "
                  f"{e['mean_iou']:>8.4f} {e['mean_boundary_f1']:>8.4f} "
                  f"{e['mean_hd95']:>8.2f}")
        print("-" * 66)

    # ============================================================
    # 8. Go/No-Go evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("GO/NO-GO GATE EVALUATION")
    print("=" * 70)

    baseline_dice = all_results["baseline"]["mean_dice"]

    # Find best EWR result at 50% sparsity
    ewr_50 = [e for e in all_results["experiments"]
              if "ewr" in e["method"] and e["sparsity"] == 0.5]
    pw_50 = [e for e in all_results["experiments"]
             if e["method"] == "pointwise" and e["sparsity"] == 0.5]

    if ewr_50 and pw_50:
        best_ewr_dice_50 = max(e["mean_dice"] for e in ewr_50)
        pw_dice_50 = pw_50[0]["mean_dice"]
        gap_50 = best_ewr_dice_50 - pw_dice_50

        print(f"\n  [50% sparsity] EWR best Dice: {best_ewr_dice_50:.4f}, "
              f"Pointwise Dice: {pw_dice_50:.4f}, Gap: {gap_50:.4f}")
        if gap_50 >= 0.015:
            print("  ✅ PASS: EWR Dice >= Pointwise + 1.5%")
        else:
            print("  ❌ FAIL: EWR Dice < Pointwise + 1.5%")
            print("     -> Check Sinkhorn convergence and gradient numerical stability")

    # Check 70% sparsity collapse
    pw_70 = [e for e in all_results["experiments"]
             if e["method"] == "pointwise" and e["sparsity"] == 0.7]
    ewr_70 = [e for e in all_results["experiments"]
              if "ewr" in e["method"] and e["sparsity"] == 0.7]

    if pw_70 and ewr_70:
        pw_dice_70 = pw_70[0]["mean_dice"]
        best_ewr_dice_70 = max(e["mean_dice"] for e in ewr_70)

        pw_drop = baseline_dice - pw_dice_70
        ewr_drop = baseline_dice - best_ewr_dice_70

        print(f"\n  [70% sparsity] Pointwise Dice drop: {pw_drop:.4f}, "
              f"EWR Dice drop: {ewr_drop:.4f}")
        if pw_drop > 0.05:
            print("  ✅ PASS: Pointwise shows collapse (Dice drop > 5%)")
        else:
            print("  ⚠️  Pointwise did NOT collapse at 70% sparsity")

        if ewr_drop <= 0.03:
            print("  ✅ PASS: EWR Dice drop <= 3%")
        else:
            print(f"  ❌ FAIL: EWR Dice drop = {ewr_drop:.4f} > 3%")

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print(f"Full results: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
