# -*- coding: utf-8 -*-
"""
Phase 1 Experiment Runner: EWR Foundation.

Compares 5 pruning methods at 3 sparsity levels.

Changes vs original:
  P0: build_dataloaders now returns 5 values (adds cal_freq_weights).
      freq_weights passed to compute_head_gradient_projections_fast.
  P1: --lambda1 argument; freq_weights and model passed to score_heads_ewr.
      Prints ω_freq distribution stats for discriminativeness check.
  P2.2: --multi_seed flag: re-runs gradient computation + evaluation for
        EWR/Pointwise across 3 calibration seeds to report std (Go/No-Go Gate).
  P2.3: --dataset argument supports CVC-ColonDB, CVC-ClinicDB, Kvasir-SEG.

Usage:
    python -m pilot_phase1.run_phase1 \
        --data_root assert/CVC-ClinicDB \
        --dataset CVC-ClinicDB \
        --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
        --device cuda:0 \
        --n_calibration 128 \
        --lambda1 0.01 \
        --multi_seed
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np
import torch

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


# ==============================================================================
# Helpers
# ==============================================================================

def _scores_for_seed(model, data_root, n_calibration, batch_size, seed, epsilons, lambda1, device):
    """
    Build a calibration loader for the given seed, compute gradient projections,
    and return pointwise + EWR scores.  Used for multi-seed stability evaluation.
    """
    cal_loader, _, _, _, freq_weights = build_dataloaders(
        data_root, n_calibration=n_calibration, batch_size=batch_size, seed=seed
    )
    _, per_sample_proj = compute_head_gradient_projections_fast(
        model, cal_loader, device, freq_weights=freq_weights
    )
    scores = {}
    scores["pointwise"] = score_heads_pointwise(per_sample_proj)
    for eps in epsilons:
        scores[f"ewr_eps{eps}"] = score_heads_ewr(
            per_sample_proj, epsilon=eps, model=model, lambda1=lambda1
        )
    return scores


def _avg_results(results_list):
    """Average a list of avg_metrics dicts."""
    avg = {}
    for key in results_list[0]:
        vals = [r[key] for r in results_list]
        if isinstance(vals[0], (int, float, np.integer, np.floating)):
            avg[key]           = float(np.mean(vals))
            avg[f"std_{key}"]  = float(np.std(vals))
    return avg


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1: EWR Foundation Experiment")
    # P2.3: dataset choice
    parser.add_argument("--dataset", type=str, default="CVC-ClinicDB",
                        choices=["CVC-ColonDB", "CVC-ClinicDB", "Kvasir-SEG"],
                        help="Dataset name (for logging; data_root must point to correct path)")
    parser.add_argument("--data_root", type=str, default="assert/CVC-ClinicDB",
                        help="Path to dataset root (images/ + masks/ subdirs)")
    parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_calibration", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/phase1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    # P1.2: L2 regularisation coefficient
    parser.add_argument("--lambda1", type=float, default=0.01,
                        help="EWR L2 regularisation coefficient (spec §3)")
    # P2.2: multi-seed stability
    parser.add_argument("--multi_seed", action="store_true",
                        help="Re-run EWR/Pointwise with 3 calibration seeds to "
                             "evaluate stability (required for Go/No-Go std gate)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 70)
    print("Phase 1: EWR Foundation Experiment")
    print(f"  Dataset:       {args.dataset}  ({args.data_root})")
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Device:        {device}")
    print(f"  Calibration:   {args.n_calibration} samples")
    print(f"  Sparsities:    {args.sparsities}")
    print(f"  EWR epsilons:  {args.epsilons}")
    print(f"  λ1 (L2 reg):   {args.lambda1}")
    print(f"  Multi-seed:    {args.multi_seed}")
    print(f"  Output:        {args.output_dir}")
    print("=" * 70)

    # ================================================================ #
    # 1. Load MedSAM model
    # ================================================================ #
    print("\n[Step 1] Loading MedSAM model ...")
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model = model.to(device)
    model.eval()

    for param in model.prompt_encoder.parameters():
        param.requires_grad = False
    for param in model.mask_decoder.parameters():
        param.requires_grad = False

    total_params   = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.image_encoder.parameters())
    print(f"  Total params:    {total_params:,}")
    print(f"  Encoder params:  {encoder_params:,}")

    # ================================================================ #
    # 2. Build data loaders
    #    P0: build_dataloaders now returns 5 values including cal_freq_weights
    # ================================================================ #
    print("\n[Step 2] Building data loaders ...")
    cal_loader, test_loader, cal_dataset, test_dataset, cal_freq_weights = build_dataloaders(
        args.data_root,
        n_calibration=args.n_calibration,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # P1: Report frequency weight distribution for discriminativeness check
    cv = cal_freq_weights.std() / max(cal_freq_weights.mean(), 1e-8)
    print(f"\n  ω_freq cv = {cv:.3f}  "
          f"{'✅ discriminative (cv >= 0.3)' if cv >= 0.3 else '⚠️  low discriminativeness (cv < 0.3)'}")

    # ================================================================ #
    # 3. Evaluate unpruned baseline
    # ================================================================ #
    print("\n[Step 3] Evaluating unpruned baseline ...")
    baseline_results = evaluate_pruned_model(
        model, test_loader, head_mask=None, device=device
    )
    b = baseline_results["avg_metrics"]
    print(f"  Dice={b['mean_dice']:.4f}  IoU={b['mean_iou']:.4f}  "
          f"BF1={b['mean_boundary_f1']:.4f}  HD95={b['mean_hd95']:.2f}")

    # ================================================================ #
    # 4. Compute head importance scores (primary seed)
    #    P0: passes freq_weights (single-pass gradient computation)
    #    P1: freq_weights weighted projections + λ1 in EWR
    # ================================================================ #
    print("\n[Step 4] Computing head importance scores ...")

    t0 = time.time()
    _, per_sample_proj = compute_head_gradient_projections_fast(
        model, cal_loader, device, freq_weights=cal_freq_weights
    )
    print(f"  Gradient computation: {time.time() - t0:.1f}s")

    all_scores = {}

    # Random (3 seeds)
    for seed in [42, 123, 456]:
        all_scores[f"random_s{seed}"] = score_heads_random(seed=seed)

    # Magnitude
    all_scores["magnitude"] = score_heads_magnitude(model)

    # Pointwise
    all_scores["pointwise"] = score_heads_pointwise(per_sample_proj)

    # EWR — P1.2: model and lambda1 passed
    for eps in args.epsilons:
        t0 = time.time()
        all_scores[f"ewr_eps{eps}"] = score_heads_ewr(
            per_sample_proj,
            epsilon=eps,
            model=model,
            lambda1=args.lambda1,
        )
        print(f"  EWR (ε={eps}): {time.time() - t0:.1f}s")

    # Save raw scores
    np.savez(os.path.join(args.output_dir, "head_scores.npz"), **all_scores)

    # ================================================================ #
    # P2.2: Multi-seed stability evaluation
    # ================================================================ #
    multi_seed_eval_seeds = [42, 123, 456]
    multi_seed_scores = {}   # method -> list of score arrays (one per seed)

    if args.multi_seed:
        print("\n[Step 4b] Multi-seed stability evaluation ...")
        for s in multi_seed_eval_seeds:
            print(f"  Seed {s} ...")
            seed_scores = _scores_for_seed(
                model, args.data_root, args.n_calibration,
                args.batch_size, s, args.epsilons, args.lambda1, device,
            )
            for method, sc in seed_scores.items():
                multi_seed_scores.setdefault(method, []).append(sc)

    # ================================================================ #
    # 5. Evaluate all method × sparsity combinations
    # ================================================================ #
    print("\n[Step 5] Running pruning experiments ...")

    all_results = {
        "config":      vars(args),
        "baseline":    b,
        "experiments": [],
    }

    methods = ["magnitude", "pointwise"] + [f"ewr_eps{eps}" for eps in args.epsilons]
    random_seeds = [42, 123, 456]

    for sparsity in args.sparsities:
        print(f"\n--- Sparsity: {sparsity * 100:.0f}% ---")

        # Random (averaged over 3 seeds)
        random_metrics_list = []
        for seed in random_seeds:
            scores = all_scores[f"random_s{seed}"]
            mask   = generate_head_mask(scores, sparsity)
            result = evaluate_pruned_model(model, test_loader, mask, device)
            random_metrics_list.append(result["avg_metrics"])

        rand_avg = _avg_results(random_metrics_list)
        rand_avg.update({"method": "random", "sparsity": sparsity,
                         "n_seeds": len(random_seeds)})
        all_results["experiments"].append(rand_avg)
        print(f"  {'random':<16}: Dice={rand_avg['mean_dice']:.4f}  "
              f"IoU={rand_avg['mean_iou']:.4f}  BF1={rand_avg['mean_boundary_f1']:.4f}  "
              f"HD95={rand_avg['mean_hd95']:.2f}")

        # Deterministic methods (primary seed)
        for method in methods:
            scores = all_scores[method]
            mask   = generate_head_mask(scores, sparsity)
            result = evaluate_pruned_model(model, test_loader, mask, device)
            exp    = result["avg_metrics"].copy()
            exp.update({"method": method, "sparsity": sparsity})
            all_results["experiments"].append(exp)
            print(f"  {method:<16}: Dice={exp['mean_dice']:.4f}  "
                  f"IoU={exp['mean_iou']:.4f}  BF1={exp['mean_boundary_f1']:.4f}  "
                  f"HD95={exp['mean_hd95']:.2f}")

        # P2.2: Multi-seed evaluation for EWR / Pointwise
        if args.multi_seed and multi_seed_scores:
            for method in ["pointwise"] + [f"ewr_eps{eps}" for eps in args.epsilons]:
                if method not in multi_seed_scores:
                    continue
                seed_metrics = []
                for sc in multi_seed_scores[method]:
                    mask   = generate_head_mask(sc, sparsity)
                    result = evaluate_pruned_model(model, test_loader, mask, device)
                    seed_metrics.append(result["avg_metrics"])
                ms_avg = _avg_results(seed_metrics)
                ms_avg.update({"method": f"{method}_multiseed",
                               "sparsity": sparsity,
                               "n_seeds": len(multi_seed_eval_seeds)})
                all_results["experiments"].append(ms_avg)
                print(f"  {method+'(ms)':<16}: Dice={ms_avg['mean_dice']:.4f}  "
                      f"std={ms_avg.get('std_mean_dice', float('nan')):.4f}")

    # ================================================================ #
    # 6. Save results
    # ================================================================ #
    results_path = os.path.join(args.output_dir, "phase1_results.json")

    def _to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(
            json.loads(json.dumps(all_results, default=_to_serializable)),
            f, indent=2,
        )
    print(f"\nResults saved to {results_path}")

    # ================================================================ #
    # 7. Summary table
    # ================================================================ #
    print("\n" + "=" * 90)
    print("PHASE 1 RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Method':<20} {'Sparsity':>8} {'Dice':>8} {'IoU':>8} {'BF1':>8} {'HD95':>8}")
    print("-" * 70)
    print(f"{'unpruned':<20} {'0%':>8} {b['mean_dice']:>8.4f} {b['mean_iou']:>8.4f} "
          f"{b['mean_boundary_f1']:>8.4f} {b['mean_hd95']:>8.2f}")
    print("-" * 70)

    for sparsity in args.sparsities:
        exps = [e for e in all_results["experiments"]
                if e["sparsity"] == sparsity and "multiseed" not in e["method"]]
        for e in exps:
            print(f"{e['method']:<20} {sparsity*100:>7.0f}%"
                  f" {e['mean_dice']:>8.4f} {e['mean_iou']:>8.4f}"
                  f" {e['mean_boundary_f1']:>8.4f} {e['mean_hd95']:>8.2f}")
        print("-" * 70)

    # ================================================================ #
    # 8. Go/No-Go Gate Evaluation
    # ================================================================ #
    print("\n" + "=" * 70)
    print("GO/NO-GO GATE EVALUATION")
    print("=" * 70)

    baseline_dice = b["mean_dice"]

    # --- Gate 1: 50% sparsity EWR Dice >= Pointwise + 1.5% ---
    ewr_50 = [e for e in all_results["experiments"]
              if "ewr" in e["method"] and "multiseed" not in e["method"]
              and e["sparsity"] == 0.5]
    pw_50  = [e for e in all_results["experiments"]
              if e["method"] == "pointwise" and e["sparsity"] == 0.5]

    if ewr_50 and pw_50:
        best_ewr_50  = max(ewr_50, key=lambda x: x["mean_dice"])
        pw_dice_50   = pw_50[0]["mean_dice"]
        gap_50       = best_ewr_50["mean_dice"] - pw_dice_50
        print(f"\n  [Gate 1 — 50% sparsity]")
        print(f"    Best EWR ({best_ewr_50['method']}): {best_ewr_50['mean_dice']:.4f}")
        print(f"    Pointwise:                           {pw_dice_50:.4f}")
        print(f"    Gap:                                 {gap_50:+.4f}  "
              f"(need >= +0.0150)")
        if gap_50 >= 0.015:
            print("    ✅ PASS: EWR Dice >= Pointwise + 1.5%")
        else:
            print("    ❌ FAIL: EWR Dice < Pointwise + 1.5%")
            print("       → Check Sinkhorn convergence (increase n_iter) and "
                  "gradient numerical stability.")

    # --- P2.2: Gate 1 std sub-condition (multi-seed required) ---
    if args.multi_seed:
        ewr_ms_50  = [e for e in all_results["experiments"]
                      if "ewr" in e["method"] and "multiseed" in e["method"]
                      and e["sparsity"] == 0.5]
        pw_ms_50   = [e for e in all_results["experiments"]
                      if "pointwise_multiseed" in e["method"] and e["sparsity"] == 0.5]
        if ewr_ms_50 and pw_ms_50:
            best_ewr_ms_std = min(e.get("std_mean_dice", 1.0) for e in ewr_ms_50)
            pw_ms_std       = pw_ms_50[0].get("std_mean_dice", 1.0)
            print(f"\n  [Gate 1 std sub-condition — 3 seeds @ 50%]")
            print(f"    Best EWR std: {best_ewr_ms_std:.4f}   Pointwise std: {pw_ms_std:.4f}")
            if best_ewr_ms_std <= pw_ms_std:
                print("    ✅ PASS: EWR std <= Pointwise std")
            else:
                print("    ❌ FAIL: EWR std > Pointwise std")
    else:
        print("\n  [Gate 1 std sub-condition] — run with --multi_seed to evaluate")

    # --- Gate 2: 70% sparsity collapse check ---
    pw_70  = [e for e in all_results["experiments"]
              if e["method"] == "pointwise" and e["sparsity"] == 0.7]
    ewr_70 = [e for e in all_results["experiments"]
              if "ewr" in e["method"] and "multiseed" not in e["method"]
              and e["sparsity"] == 0.7]

    if pw_70 and ewr_70:
        pw_drop  = baseline_dice - pw_70[0]["mean_dice"]
        best_ewr_70 = max(ewr_70, key=lambda x: x["mean_dice"])
        ewr_drop = baseline_dice - best_ewr_70["mean_dice"]

        print(f"\n  [Gate 2 — 70% sparsity]")
        print(f"    Pointwise Dice drop: {pw_drop:.4f}  (need > 0.05 for collapse)")
        print(f"    EWR Dice drop:       {ewr_drop:.4f}  (need <= 0.03)")

        pw_pass  = pw_drop > 0.05
        ewr_pass = ewr_drop <= 0.03

        print(f"    {'✅' if pw_pass  else '⚠️ '} Pointwise collapse: "
              f"{'YES' if pw_pass else 'NO (collapse not observed)'}")
        print(f"    {'✅' if ewr_pass else '❌'} EWR robustness: "
              f"{'PASS' if ewr_pass else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"Experiment complete.  Full results: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
