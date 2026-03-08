"""Entry point for computing FLOPs and expert activation frequency of trained checkpoints.

Usage:
    python compute_FLOPs/compute_flops.py <ckpt_path> [--num_samples_per_class N] [--seed S] [--guide_scale G]

Example:
    python compute_FLOPs/compute_flops.py outputs/ProMoE_TC_B/004_ProMoE_B/checkpoints/ckpt_step_500000.pth
    python compute_FLOPs/compute_flops.py outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth --num_samples_per_class 5
"""

import os
import sys
import argparse
import time
import inspect
import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from compute_FLOPs.config_utils import (
    resolve_ckpt_info,
    load_config_from_yaml,
    build_model_from_cfg,
    load_ema_weights,
)
from compute_FLOPs.expert_tracker import ExpertActivationTracker
from compute_FLOPs.flops_counter import FLOPsAccumulator
from compute_FLOPs.visualize import plot_expert_frequencies


def get_sampling_sigmas(sampling_steps, shift):
    """Replicate the sigma schedule from sample.py."""
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


def retrieve_timesteps(scheduler, device, sigmas):
    """Replicate retrieve_timesteps from sample.py."""
    accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    if not accept_sigmas:
        raise ValueError(f"Scheduler {scheduler.__class__} does not support custom sigmas.")
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    return scheduler.timesteps, len(scheduler.timesteps)


def main():
    parser = argparse.ArgumentParser(description="Compute FLOPs and expert activation frequency for a trained checkpoint.")
    parser.add_argument("ckpt", type=str, help="Relative path to the checkpoint file (.pth)")
    parser.add_argument("--num_samples_per_class", type=int, default=10,
                        help="Number of samples to generate per class (default: 10)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--guide_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (default: 1.0)")
    args = parser.parse_args()

    # --- 1. Resolve ckpt path to config ---
    print(f"Resolving checkpoint: {args.ckpt}")
    model_name, custom_cfg_name, ckpt_step, config_yaml_path, output_dir = resolve_ckpt_info(args.ckpt)
    print(f"  Model name:      {model_name}")
    print(f"  Config name:     {custom_cfg_name}")
    print(f"  Checkpoint step: {ckpt_step}")
    print(f"  Config YAML:     {config_yaml_path}")

    # --- 2. Load config and build model ---
    cfg = load_config_from_yaml(config_yaml_path)
    num_classes = cfg.num_classes  # 1000
    image_size = cfg.image_size   # 256
    sample_steps = getattr(cfg, "sample_steps", 250)
    sample_shift = getattr(cfg, "sample_shift", 1.0)
    num_train_timesteps = getattr(cfg, "num_train_timesteps", 1000)
    shift = getattr(cfg, "shift", 1.0)

    print(f"\n  num_classes:     {num_classes}")
    print(f"  sample_steps:    {sample_steps}")
    print(f"  sample_shift:    {sample_shift}")
    print(f"  guide_scale:     {args.guide_scale}")
    print(f"  seed:            {args.seed}")
    print(f"  samples/class:   {args.num_samples_per_class}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    print("\nBuilding model...")
    model = build_model_from_cfg(cfg)

    print("Loading EMA weights...")
    model = load_ema_weights(model, args.ckpt)
    model = model.to(device).eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count / 1e6:.2f}M")

    # --- 3. Setup expert activation tracker ---
    tracker = ExpertActivationTracker(model)
    has_moe = tracker.num_moe_blocks > 0
    if has_moe:
        print(f"\n  Found {tracker.num_moe_blocks} MoE blocks, tracking expert activations.")
        tracker.start()
    else:
        print("\n  No MoE blocks found, skipping expert tracking.")

    # --- 4. Setup FLOPs accumulator ---
    flops_acc = FLOPsAccumulator(model)
    flops_acc.start()

    # --- 5. Run sampling loop with FLOPs counting + expert tracking ---
    latent_shape = (4, 1, image_size // 8, image_size // 8)  # (4, 1, 32, 32)
    total_samples = num_classes * args.num_samples_per_class
    guide_scale = args.guide_scale
    forwards_per_step = 2 if guide_scale > 1.0 else 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sampling_sigmas = get_sampling_sigmas(sample_steps, sample_shift)

    cond_forward_passes = 0
    cond_total_flops = 0

    print(f"\nRunning sampling loop: {total_samples} samples x {sample_steps} steps x {forwards_per_step} forwards/step ...")
    start_time = time.time()
    sample_count = 0

    with torch.no_grad():
        for class_idx in range(num_classes):
            for sample_idx in range(args.num_samples_per_class):
                # Re-create scheduler per sample (it has internal state modified by .step())
                scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=num_train_timesteps, shift=shift)
                timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)

                latent = torch.randn(1, *latent_shape, device=device)
                y = torch.tensor([class_idx], device=device)
                y_null = torch.tensor([num_classes], device=device)

                for t in timesteps:
                    timestep = t.unsqueeze(0)

                    # Cond forward pass — count FLOPs
                    noise_pred_cond = model(latent, timestep, context=y)
                    cond_total_flops += flops_acc.collect()
                    cond_forward_passes += 1
                    if isinstance(noise_pred_cond, tuple):
                        noise_pred_cond = noise_pred_cond[0]

                    # Uncond forward pass (CFG) — discard FLOPs (not part of cond routing)
                    if guide_scale > 1.0:
                        noise_pred_uncond = model(latent, timestep, context=y_null)
                        flops_acc.collect()  # collect but don't add to cond_total_flops
                        if isinstance(noise_pred_uncond, tuple):
                            noise_pred_uncond = noise_pred_uncond[0]
                        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

                    if noise_pred.shape[1] != latent.shape[1]:
                        noise_pred, _ = noise_pred.chunk(2, dim=1)

                    latent = scheduler.step(noise_pred.unsqueeze(2), t, latent, return_dict=False)[0]

                sample_count += 1
                if sample_count % 100 == 0 or sample_count == total_samples:
                    elapsed = time.time() - start_time
                    print(f"  Progress: {sample_count}/{total_samples} samples ({elapsed:.1f}s)")

    elapsed_total = time.time() - start_time
    print(f"\nSampling complete in {elapsed_total:.1f}s")

    # --- 6. Compute final FLOPs result ---
    flops_acc.stop()

    # Only use cond forward FLOPs so that CFG=1 and CFG>1 give consistent results.
    # Uncond forward always routes to the dedicated uncond expert (no dynamic routing),
    # so it should not affect the average.
    avg_flops_per_forward = cond_total_flops / cond_forward_passes
    avg_gflops = avg_flops_per_forward / 1e9

    print(f"\n{'='*60}")
    print(f"FLOPs Evaluation Results (cond forward only)")
    print(f"{'='*60}")
    print(f"  Cond total FLOPs:          {cond_total_flops / 1e9:.4f} GFLOPs")
    print(f"  Cond forward passes:       {cond_forward_passes}")
    print(f"  Avg FLOPs/forward:         {avg_gflops:.4f} GFLOPs")
    print(f"{'='*60}")

    # --- 7. Save results ---
    save_dir = os.path.join(output_dir, "sample", f"step{ckpt_step}", "flops_eval")
    os.makedirs(save_dir, exist_ok=True)

    # Save FLOPs result as txt
    txt_path = os.path.join(save_dir, "flops_result.txt")
    with open(txt_path, "w") as f:
        f.write(f"FLOPs Evaluation Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Date:                        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint:                  {os.path.abspath(args.ckpt)}\n")
        f.write(f"Model name:                  {model_name}\n")
        f.write(f"Config:                      {custom_cfg_name}\n")
        f.write(f"Checkpoint step:             {ckpt_step}\n")
        f.write(f"Model parameters:            {param_count / 1e6:.2f}M\n")
        f.write(f"\n")
        f.write(f"--- Computation Settings ---\n")
        f.write(f"Number of classes:           {num_classes}\n")
        f.write(f"Samples per class:           {args.num_samples_per_class}\n")
        f.write(f"Total samples:               {total_samples}\n")
        f.write(f"Sample steps:                {sample_steps}\n")
        f.write(f"Sample shift:                {sample_shift}\n")
        f.write(f"Guide scale (CFG):           {args.guide_scale}\n")
        f.write(f"Forwards per step:           {forwards_per_step}\n")
        f.write(f"Cond forward passes:         {cond_forward_passes}\n")
        f.write(f"Random seed:                 {args.seed}\n")
        f.write(f"\n")
        f.write(f"--- FLOPs Results (cond forward only) ---\n")
        f.write(f"Cond total FLOPs:            {cond_total_flops / 1e9:.4f} GFLOPs\n")
        f.write(f"Cond forward passes:         {cond_forward_passes}\n")
        f.write(f"Avg FLOPs/forward:           {avg_gflops:.4f} GFLOPs\n")
        f.write(f"  = Cond total FLOPs / {cond_forward_passes} cond forward passes\n")
        f.write(f"{'='*60}\n")

    print(f"\nFLOPs result saved to: {txt_path}")

    # --- 8. Plot expert activation frequencies ---
    if has_moe:
        tracker.stop()
        frequencies = tracker.get_frequencies()

        plot_paths = plot_expert_frequencies(
            frequencies,
            save_dir=save_dir,
            model_name=model_name,
        )
        print(f"\nExpert frequency plots saved:")
        for p in plot_paths:
            print(f"  {p}")

        # Also write frequencies to the txt file
        with open(txt_path, "a") as f:
            f.write(f"\n--- Expert Activation Frequencies ---\n")
            block_indices = sorted(frequencies.keys())
            for block_idx in block_indices:
                freq = frequencies[block_idx]
                f.write(f"Block {block_idx}: {[f'{v:.4f}' for v in freq]}\n")
            # Average
            if block_indices:
                num_experts = len(frequencies[block_indices[0]])
                avg_freq = [0.0] * num_experts
                for block_idx in block_indices:
                    for e in range(num_experts):
                        avg_freq[e] += frequencies[block_idx][e]
                avg_freq = [v / len(block_indices) for v in avg_freq]
                f.write(f"Average:  {[f'{v:.4f}' for v in avg_freq]}\n")

    print(f"\nAll results saved to: {save_dir}")


if __name__ == "__main__":
    main()
