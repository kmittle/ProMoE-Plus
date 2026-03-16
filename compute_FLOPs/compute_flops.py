"""Entry point for computing FLOPs and expert activation frequency of trained checkpoints.

Supports multi-GPU execution: GPU IDs are read from the YAML config's `gpu_ids` field.

GPU IDs are automatically read from the YAML config resolved from the checkpoint path.

Usage:
    python compute_FLOPs/compute_flops.py <ckpt_path> [--num_samples_per_class N] [--seed S] [--guide_scale G] [--save_every_steps K]

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
import pickle

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from compute_FLOPs.config_utils import (
    resolve_ckpt_info,
    load_config_from_yaml,
    build_model_from_cfg,
    load_ema_weights,
)
from compute_FLOPs.expert_tracker import ExpertActivationTracker, raw_counts_to_frequencies
from compute_FLOPs.flops_counter import FLOPsAccumulator
from compute_FLOPs.activated_params_tracker import ActivatedParamsTracker
from compute_FLOPs.visualize import plot_expert_frequencies
from utils import find_free_port

M = 1024 ** 2  # Mebi (for parameter counts)
G = 1024 ** 3  # Gibi (for FLOPs)


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


def aggregate_activated_param_stats(stats_list):
    """Merge per-rank activated-parameter stats into one global dict."""
    if not stats_list:
        return {
            "total_params": 0.0,
            "non_moe_params": 0.0,
            "mean_activated_params": 0.0,
            "min_activated_params": 0.0,
            "max_activated_params": 0.0,
            "num_forwards": 0,
            "activation_ratio": 0.0,
        }

    total_params = float(stats_list[0]["total_params"])
    non_moe_params = float(stats_list[0]["non_moe_params"])

    total_forwards = int(sum(int(s.get("num_forwards", 0)) for s in stats_list))
    if total_forwards > 0:
        weighted_sum = sum(
            float(s["mean_activated_params"]) * int(s.get("num_forwards", 0))
            for s in stats_list
        )
        mean_activated = weighted_sum / total_forwards
        min_candidates = [
            float(s["min_activated_params"])
            for s in stats_list
            if int(s.get("num_forwards", 0)) > 0
        ]
        max_candidates = [
            float(s["max_activated_params"])
            for s in stats_list
            if int(s.get("num_forwards", 0)) > 0
        ]
        min_activated = min(min_candidates) if min_candidates else 0.0
        max_activated = max(max_candidates) if max_candidates else 0.0
    else:
        mean_activated = 0.0
        min_activated = 0.0
        max_activated = 0.0

    return {
        "total_params": total_params,
        "non_moe_params": non_moe_params,
        "mean_activated_params": mean_activated,
        "min_activated_params": min_activated,
        "max_activated_params": max_activated,
        "num_forwards": total_forwards,
        "activation_ratio": mean_activated / total_params if total_params > 0 else 0.0,
    }


def compute_saved_step_indices(sample_steps, save_every_steps):
    """Return 1-based denoising step indices to snapshot."""
    if save_every_steps <= 0:
        raise ValueError("--save_every_steps must be positive.")

    saved_step_indices = list(range(save_every_steps, sample_steps + 1, save_every_steps))
    if not saved_step_indices or saved_step_indices[-1] != sample_steps:
        saved_step_indices.append(sample_steps)
    return saved_step_indices


def merge_expert_raw_counts(target_raw_counts, source_raw_counts):
    """Merge source raw expert counts into target in place."""
    for block_idx, (counts_dict, total_tokens) in source_raw_counts.items():
        if block_idx not in target_raw_counts:
            target_raw_counts[block_idx] = ({}, 0)
        merged_counts, merged_total = target_raw_counts[block_idx]
        for expert_idx, count in counts_dict.items():
            merged_counts[expert_idx] = merged_counts.get(expert_idx, 0) + count
        target_raw_counts[block_idx] = (merged_counts, merged_total + total_tokens)
    return target_raw_counts


def merge_stepwise_raw_counts(stepwise_raw_counts_list):
    """Merge per-rank stepwise raw counts."""
    merged = {}
    for stepwise_raw_counts in stepwise_raw_counts_list:
        if not stepwise_raw_counts:
            continue
        for step_idx, raw_counts in stepwise_raw_counts.items():
            if step_idx not in merged:
                merged[step_idx] = {}
            merge_expert_raw_counts(merged[step_idx], raw_counts)
    return merged


def compute_average_frequency(frequencies):
    """Average expert frequencies across all tracked blocks."""
    block_indices = sorted(frequencies.keys())
    if not block_indices:
        return []

    num_experts = len(frequencies[block_indices[0]])
    avg_freq = [0.0] * num_experts
    for block_idx in block_indices:
        for expert_idx in range(num_experts):
            avg_freq[expert_idx] += frequencies[block_idx][expert_idx]
    return [value / len(block_indices) for value in avg_freq]


def serialize_timestep_value(timestep):
    """Convert a scalar timestep tensor/value into a Python int or float."""
    value = float(timestep.item() if hasattr(timestep, "item") else timestep)
    if value.is_integer():
        return int(value)
    return value


def save_stepwise_expert_frequency_reports(
    stepwise_raw_counts,
    num_routed_experts_per_block,
    save_dir,
    saved_step_indices,
    saved_step_timestep_values,
    model_name="",
):
    """Save per-step expert frequency summaries into step-xxx subdirectories."""
    if not saved_step_indices:
        return []

    pad_width = max(3, len(str(max(saved_step_indices))))
    saved_dirs = []
    for step_idx in saved_step_indices:
        step_dir = os.path.join(save_dir, f"step-{step_idx:0{pad_width}d}")
        os.makedirs(step_dir, exist_ok=True)

        frequencies = raw_counts_to_frequencies(
            stepwise_raw_counts.get(step_idx, {}),
            num_routed_experts_per_block,
        )
        plot_expert_frequencies(
            frequencies,
            save_dir=step_dir,
            model_name=model_name,
        )

        txt_path = os.path.join(step_dir, "expert_frequencies.txt")
        with open(txt_path, "w") as f:
            f.write(f"Sampling step index: {step_idx}\n")
            if step_idx in saved_step_timestep_values:
                f.write(f"Scheduler timestep:  {saved_step_timestep_values[step_idx]}\n")
            f.write("\n--- Expert Activation Frequencies ---\n")
            block_indices = sorted(frequencies.keys())
            for block_idx in block_indices:
                freq = frequencies[block_idx]
                f.write(f"Block {block_idx}: {[f'{v:.4f}' for v in freq]}\n")
            avg_freq = compute_average_frequency(frequencies)
            if avg_freq:
                f.write(f"Average:  {[f'{v:.4f}' for v in avg_freq]}\n")

        saved_dirs.append(step_dir)

    return saved_dirs


def worker(rank, world_size, args, model_name, custom_cfg_name, ckpt_step,
           config_yaml_path, output_dir, master_port):
    """Worker function for each GPU."""
    # --- Init distributed ---
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(hours=1)
        )

    is_main = (rank == 0)

    # --- Load config and build model ---
    cfg = load_config_from_yaml(config_yaml_path)
    num_classes = cfg.num_classes
    image_size = cfg.image_size
    sample_steps = getattr(cfg, "sample_steps", 250)
    sample_shift = getattr(cfg, "sample_shift", 1.0)
    num_train_timesteps = getattr(cfg, "num_train_timesteps", 1000)
    shift = getattr(cfg, "shift", 1.0)
    guide_scale = args.guide_scale

    if is_main:
        print(f"\n  Model name:      {model_name}")
        print(f"  Config name:     {custom_cfg_name}")
        print(f"  Checkpoint step: {ckpt_step}")
        print(f"  num_classes:     {num_classes}")
        print(f"  sample_steps:    {sample_steps}")
        print(f"  guide_scale:     {guide_scale}")
        print(f"  seed:            {args.seed}")
        print(f"  samples/class:   {args.num_samples_per_class}")
        print(f"  world_size:      {world_size}")

    model = build_model_from_cfg(cfg)
    model = load_ema_weights(model, args.ckpt)
    model = model.to(device).eval()

    if is_main:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {param_count / M:.2f}M")

    # --- Setup expert activation tracker ---
    tracker = ExpertActivationTracker(model)
    has_moe = tracker.num_moe_blocks > 0
    if has_moe:
        if is_main:
            print(f"\n  Found {tracker.num_moe_blocks} MoE blocks, tracking expert activations.")
        tracker.start()
    else:
        if is_main:
            print("\n  No MoE blocks found, skipping expert tracking.")

    # --- Setup FLOPs accumulator ---
    flops_acc = FLOPsAccumulator(model)
    flops_acc.start()

    # --- Setup activated params tracker ---
    params_tracker = ActivatedParamsTracker(model)
    params_tracker.start()
    if is_main:
        if params_tracker.moe_blocks:
            print(f"  Tracking activated parameters across {len(params_tracker.moe_blocks)} MoE blocks.")
        else:
            print(f"  Dense model: all parameters activated every forward.")

    # --- Partition classes across GPUs ---
    all_classes = list(range(num_classes))
    classes_per_rank = np.array_split(all_classes, world_size)
    my_classes = classes_per_rank[rank]

    latent_shape = (4, 1, image_size // 8, image_size // 8)
    forwards_per_step = 2 if guide_scale > 1.0 else 1
    total_samples_this_rank = len(my_classes) * args.num_samples_per_class

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    sampling_sigmas = get_sampling_sigmas(sample_steps, sample_shift)
    saved_step_indices = compute_saved_step_indices(sample_steps, args.save_every_steps)
    saved_step_index_set = set(saved_step_indices)

    reference_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=num_train_timesteps, shift=shift
    )
    reference_timesteps, _ = retrieve_timesteps(
        reference_scheduler,
        device=device,
        sigmas=sampling_sigmas,
    )
    saved_step_timestep_values = {
        step_idx: serialize_timestep_value(reference_timesteps[step_idx - 1])
        for step_idx in saved_step_indices
    }

    cond_forward_passes = 0
    cond_total_flops = 0
    local_stepwise_raw_counts = (
        {step_idx: {} for step_idx in saved_step_indices} if has_moe else {}
    )

    if is_main:
        total_samples_global = num_classes * args.num_samples_per_class
        print(f"\nRunning sampling loop: {total_samples_global} samples x {sample_steps} steps x {forwards_per_step} forwards/step ...")
        print(f"  This rank handles {len(my_classes)} classes ({total_samples_this_rank} samples)")
        print(f"  Saving per-step expert frequencies every {args.save_every_steps} steps: {saved_step_indices}")

    start_time = time.time()
    sample_count = 0

    with torch.no_grad():
        for class_idx in my_classes:
            for sample_idx in range(args.num_samples_per_class):
                scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=num_train_timesteps, shift=shift)
                timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)

                latent = torch.randn(1, *latent_shape, device=device)
                y = torch.tensor([class_idx], device=device)
                y_null = torch.tensor([num_classes], device=device)

                for step_idx, t in enumerate(timesteps, start=1):
                    timestep = t.unsqueeze(0)

                    if has_moe:
                        tracker.begin_forward()
                    noise_pred_cond = model(latent, timestep, context=y)
                    cond_total_flops += flops_acc.collect()
                    cond_forward_passes += 1
                    params_tracker.record_forward_pass()
                    if has_moe:
                        current_forward_raw_counts = tracker.consume_current_forward_raw_counts()
                        if step_idx in saved_step_index_set:
                            merge_expert_raw_counts(
                                local_stepwise_raw_counts[step_idx],
                                current_forward_raw_counts,
                            )
                    if isinstance(noise_pred_cond, tuple):
                        noise_pred_cond = noise_pred_cond[0]

                    if guide_scale > 1.0:
                        if has_moe:
                            tracker.begin_forward()
                        noise_pred_uncond = model(latent, timestep, context=y_null)
                        flops_acc.collect()  # collect but don't add to cond_total_flops
                        if has_moe:
                            tracker.consume_current_forward_raw_counts()
                        # Don't record uncond forward — activated params should be
                        # cond-only for consistency with FLOPs reporting.  Uncond
                        # forwards route all tokens to the single uncond expert,
                        # which would skew the mean downward.
                        if isinstance(noise_pred_uncond, tuple):
                            noise_pred_uncond = noise_pred_uncond[0]
                        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

                    if noise_pred.shape[1] != latent.shape[1]:
                        noise_pred, _ = noise_pred.chunk(2, dim=1)

                    latent = scheduler.step(noise_pred.unsqueeze(2), t, latent, return_dict=False)[0]

                sample_count += 1
                if is_main and (sample_count % 100 == 0 or sample_count == total_samples_this_rank):
                    elapsed = time.time() - start_time
                    print(f"  [Rank 0] Progress: {sample_count}/{total_samples_this_rank} samples ({elapsed:.1f}s)")

    elapsed_total = time.time() - start_time
    if is_main:
        print(f"\nRank 0 sampling complete in {elapsed_total:.1f}s")

    flops_acc.stop()
    params_tracker.stop()
    local_act_stats = params_tracker.get_stats()
    if not params_tracker.moe_blocks:
        # Dense model: all params are activated every conditional forward.
        local_act_stats["num_forwards"] = cond_forward_passes

    # --- Gather results across GPUs ---
    if world_size > 1:
        # Gather FLOPs via all_reduce
        flops_tensor = torch.tensor([cond_total_flops, float(cond_forward_passes)],
                                    dtype=torch.float64, device=device)
        dist.all_reduce(flops_tensor, op=dist.ReduceOp.SUM)
        cond_total_flops = flops_tensor[0].item()
        cond_forward_passes = int(flops_tensor[1].item())

        # Gather expert counts via pickle + gather
        if has_moe:
            tracker.stop()
            raw_counts = tracker.get_raw_counts()
            raw_bytes = pickle.dumps(raw_counts)
            raw_np = np.frombuffer(raw_bytes, dtype=np.uint8)
            raw_tensor = torch.from_numpy(raw_np.copy()).to(device)
            size_tensor = torch.tensor([len(raw_bytes)], dtype=torch.long, device=device)

            # Gather sizes first
            all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            dist.all_gather(all_sizes, size_tensor)

            max_size = max(s.item() for s in all_sizes)
            padded = torch.zeros(max_size, dtype=torch.uint8, device=device)
            padded[:len(raw_bytes)] = raw_tensor

            all_padded = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
            dist.all_gather(all_padded, padded)

            if is_main:
                for r in range(world_size):
                    if r == rank:
                        continue
                    sz = all_sizes[r].item()
                    other_bytes = all_padded[r][:sz].cpu().numpy().tobytes()
                    other_raw = pickle.loads(other_bytes)
                    tracker.merge_raw_counts(other_raw)

            gathered_stepwise_raw_counts = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_stepwise_raw_counts, local_stepwise_raw_counts)
        else:
            gathered_stepwise_raw_counts = []
    else:
        if has_moe:
            tracker.stop()
            gathered_stepwise_raw_counts = [local_stepwise_raw_counts]
        else:
            gathered_stepwise_raw_counts = []

    # Gather activated-parameter stats across GPUs.
    if world_size > 1:
        gathered_act_stats = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_act_stats, local_act_stats)
    else:
        gathered_act_stats = [local_act_stats]

    # --- Rank 0: compute final results and save ---
    if is_main:
        param_count = sum(p.numel() for p in model.parameters())
        avg_flops_per_forward = cond_total_flops / cond_forward_passes
        avg_gflops = avg_flops_per_forward / G
        total_samples = num_classes * args.num_samples_per_class

        # Activated params stats (global across all ranks)
        act_stats = aggregate_activated_param_stats(gathered_act_stats)
        # Keep this aligned with FLOPs reporting denominator.
        act_stats["num_forwards"] = cond_forward_passes
        mean_activated_per_step = act_stats["mean_activated_params"]
        mean_activated_per_sample = mean_activated_per_step * sample_steps

        print(f"\n{'='*60}")
        print(f"FLOPs Evaluation Results (cond forward only)")
        print(f"{'='*60}")
        print(f"  Cond total FLOPs:          {cond_total_flops / G:.4f} GFLOPs")
        print(f"  Cond forward passes:       {cond_forward_passes}")
        print(f"  Avg FLOPs/forward:         {avg_gflops:.4f} GFLOPs")
        print(f"{'='*60}")
        print(f"\nActivated Parameters (cond forward only)")
        print(f"{'='*60}")
        print(f"  Total model params:        {act_stats['total_params'] / M:.2f}M")
        print(f"  Non-MoE params (always):   {act_stats['non_moe_params'] / M:.2f}M")
        print(f"  Mean activated params:     {act_stats['mean_activated_params'] / M:.2f}M")
        print(f"  Min activated params:      {act_stats['min_activated_params'] / M:.2f}M")
        print(f"  Max activated params:      {act_stats['max_activated_params'] / M:.2f}M")
        print(f"  Mean activated/step:       {mean_activated_per_step / M:.2f}M")
        print(f"  Mean activated/sample:     {mean_activated_per_sample / M:.2f}M")
        print(f"  Activation ratio:          {act_stats['activation_ratio']:.4f}")
        print(f"  Cond forward passes:       {act_stats['num_forwards']}")
        print(f"{'='*60}")

        # Save results
        save_dir = os.path.join(output_dir, "sample", f"step{ckpt_step}", "flops_eval")
        os.makedirs(save_dir, exist_ok=True)

        txt_path = os.path.join(save_dir, "flops_result.txt")
        with open(txt_path, "w") as f:
            f.write(f"FLOPs Evaluation Results\n")
            f.write(f"{'='*60}\n")
            f.write(f"Date:                        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint:                  {os.path.abspath(args.ckpt)}\n")
            f.write(f"Model name:                  {model_name}\n")
            f.write(f"Config:                      {custom_cfg_name}\n")
            f.write(f"Checkpoint step:             {ckpt_step}\n")
            f.write(f"Model parameters:            {param_count / M:.2f}M\n")
            f.write(f"\n")
            f.write(f"--- Computation Settings ---\n")
            f.write(f"Number of classes:           {num_classes}\n")
            f.write(f"Samples per class:           {args.num_samples_per_class}\n")
            f.write(f"Total samples:               {total_samples}\n")
            f.write(f"Sample steps:                {sample_steps}\n")
            f.write(f"Sample shift:                {sample_shift}\n")
            f.write(f"Guide scale (CFG):           {guide_scale}\n")
            f.write(f"Forwards per step:           {forwards_per_step}\n")
            f.write(f"Cond forward passes:         {cond_forward_passes}\n")
            f.write(f"Random seed:                 {args.seed}\n")
            f.write(f"Number of GPUs:              {world_size}\n")
            f.write(f"\n")
            f.write(f"--- FLOPs Results (cond forward only) ---\n")
            f.write(f"Cond total FLOPs:            {cond_total_flops / G:.4f} GFLOPs\n")
            f.write(f"Cond forward passes:         {cond_forward_passes}\n")
            f.write(f"Avg FLOPs/forward:           {avg_gflops:.4f} GFLOPs\n")
            f.write(f"  = Cond total FLOPs / {cond_forward_passes} cond forward passes\n")
            f.write(f"{'='*60}\n")
            f.write(f"\n--- Activated Parameters (cond forward only) ---\n")
            f.write(f"Total model params:          {act_stats['total_params'] / M:.2f}M\n")
            f.write(f"Non-MoE params (always on):  {act_stats['non_moe_params'] / M:.2f}M\n")
            f.write(f"Mean activated params:       {act_stats['mean_activated_params'] / M:.2f}M\n")
            f.write(f"Min activated params:        {act_stats['min_activated_params'] / M:.2f}M\n")
            f.write(f"Max activated params:        {act_stats['max_activated_params'] / M:.2f}M\n")
            f.write(f"Mean activated/step:         {mean_activated_per_step / M:.2f}M\n")
            f.write(f"Mean activated/sample:       {mean_activated_per_sample / M:.2f}M\n")
            f.write(f"Activation ratio:            {act_stats['activation_ratio']:.4f}\n")
            f.write(f"Cond forward passes:         {act_stats['num_forwards']}\n")

        print(f"\nFLOPs result saved to: {txt_path}")

        # Plot expert activation frequencies
        if has_moe:
            frequencies = tracker.get_frequencies()
            num_routed_experts_per_block = tracker.get_num_routed_experts_per_block()
            merged_stepwise_raw_counts = merge_stepwise_raw_counts(gathered_stepwise_raw_counts)

            plot_paths = plot_expert_frequencies(
                frequencies,
                save_dir=save_dir,
                model_name=model_name,
            )
            print(f"\nExpert frequency plots saved:")
            for p in plot_paths:
                print(f"  {p}")

            with open(txt_path, "a") as f:
                f.write(f"\n--- Expert Activation Frequencies ---\n")
                block_indices = sorted(frequencies.keys())
                for block_idx in block_indices:
                    freq = frequencies[block_idx]
                    f.write(f"Block {block_idx}: {[f'{v:.4f}' for v in freq]}\n")
                avg_freq = compute_average_frequency(frequencies)
                if avg_freq:
                    f.write(f"Average:  {[f'{v:.4f}' for v in avg_freq]}\n")

            stepwise_dirs = save_stepwise_expert_frequency_reports(
                stepwise_raw_counts=merged_stepwise_raw_counts,
                num_routed_experts_per_block=num_routed_experts_per_block,
                save_dir=save_dir,
                saved_step_indices=saved_step_indices,
                saved_step_timestep_values=saved_step_timestep_values,
                model_name=model_name,
            )
            if stepwise_dirs:
                print("\nPer-step expert frequency reports saved under:")
                for step_dir in stepwise_dirs:
                    print(f"  {step_dir}")

        print(f"\nAll results saved to: {save_dir}")

    # --- Cleanup ---
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Compute FLOPs and expert activation frequency for a trained checkpoint.")
    parser.add_argument("ckpt", type=str, help="Relative path to the checkpoint file (.pth)")
    parser.add_argument("--num_samples_per_class", type=int, default=5,
                        help="Number of samples to generate per class (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--guide_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (default: 1.0)")
    parser.add_argument("--save_every_steps", type=int, default=50,
                        help="Save per-step expert activation frequencies every N denoising steps (default: 50)")
    args = parser.parse_args()

    # Resolve ckpt path to config YAML
    print(f"Resolving checkpoint: {args.ckpt}")
    model_name, custom_cfg_name, ckpt_step, config_yaml_path, output_dir = resolve_ckpt_info(args.ckpt)
    print(f"  Config YAML:     {config_yaml_path}")

    # Read gpu_ids from the resolved config
    import yaml
    with open(config_yaml_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    gpu_ids = yaml_cfg.get('gpu_ids', [0])
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]

    world_size = len(gpu_ids)

    # Set CUDA_VISIBLE_DEVICES so that rank i maps to gpu_ids[i]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    master_port = find_free_port()

    print(f"  GPU IDs:         {gpu_ids} (world_size={world_size})")

    if world_size == 1:
        worker(0, world_size, args, model_name, custom_cfg_name, ckpt_step,
               config_yaml_path, output_dir, master_port)
    else:
        mp.spawn(
            worker,
            args=(world_size, args, model_name, custom_cfg_name, ckpt_step,
                  config_yaml_path, output_dir, master_port),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
