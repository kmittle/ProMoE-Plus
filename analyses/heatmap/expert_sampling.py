from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from analyses.heatmap.sample_specs import HeatmapSampleSpec
from analyses.heatmap.sampling import (
    _build_latent_noise_for_specs,
    _decode_final_rgb_images,
)
from analyses.t_SNE.sampling import get_sampling_sigmas, retrieve_timesteps


@torch.inference_mode()
def sample_and_collect_token_choice_expert_heatmaps(
    model,
    vae,
    capture,
    runtime_cfg,
    sample_specs: list[HeatmapSampleSpec],
    analysis_steps: list[int],
    seed: int,
    device,
):
    latent_shape = (4, 1, runtime_cfg.image_size // 8, runtime_cfg.image_size // 8)
    latents = _build_latent_noise_for_specs(sample_specs, latent_shape, seed, device)
    class_tensor = torch.tensor(
        [sample_spec.class_id for sample_spec in sample_specs],
        device=device,
        dtype=torch.long,
    )

    sample_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=runtime_cfg.num_train_timesteps,
        shift=runtime_cfg.shift,
    )
    sampling_sigmas = get_sampling_sigmas(runtime_cfg.sample_steps, runtime_cfg.sample_shift)
    timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)

    sample_records = OrderedDict(
        (
            sample_spec.sample_id,
            {
                "sample_id": sample_spec.sample_id,
                "class_id": sample_spec.class_id,
                "class_name": sample_spec.class_name,
                "sample_index": sample_spec.sample_index,
                "expert_maps": OrderedDict(),
                "mean_block_maps": OrderedDict(),
            },
        )
        for sample_spec in sample_specs
    )

    scheduler_timesteps = OrderedDict()
    analysis_step_set = set(analysis_steps)
    autocast_dtype = getattr(runtime_cfg, "val_param_dtype", torch.float32)
    if autocast_dtype in (torch.float16, torch.bfloat16):
        autocast_context_factory = lambda: torch.cuda.amp.autocast(dtype=autocast_dtype)
    else:
        from contextlib import nullcontext

        autocast_context_factory = nullcontext

    token_grid_size = None
    mean_block_sums = OrderedDict()
    mean_step_count = 0

    for timestep_index, timestep_value in enumerate(timesteps):
        denoise_step = timestep_index + 1
        timestep_tensor = torch.full(
            (latents.size(0),),
            float(timestep_value.item()),
            device=device,
            dtype=timesteps.dtype,
        )

        should_store_snapshot = denoise_step in analysis_step_set
        capture.enable(
            denoise_step=denoise_step,
            scheduler_timestep=float(timestep_value.item()),
        )

        with autocast_context_factory():
            noise_pred = model(
                latents,
                timestep_tensor,
                context=class_tensor,
                use_gradient_checkpointing=getattr(runtime_cfg, "use_gradient_checkpointing", False),
            )
            if isinstance(noise_pred, tuple):
                noise_pred = noise_pred[0]

        batch_records = capture.disable_and_collect()
        mean_step_count += 1

        for block_idx, payload in batch_records.items():
            token_grid_size = payload["grid_size"]
            block_maps = payload["expert_index_maps"].numpy().reshape(
                len(sample_specs),
                token_grid_size,
                token_grid_size,
            )
            if block_idx not in mean_block_sums:
                mean_block_sums[block_idx] = block_maps.astype(np.float64, copy=True)
            else:
                mean_block_sums[block_idx] += block_maps

            if should_store_snapshot:
                scheduler_timesteps[denoise_step] = float(timestep_value.item())
                for batch_index, sample_spec in enumerate(sample_specs):
                    sample_records[sample_spec.sample_id]["expert_maps"].setdefault(
                        denoise_step,
                        OrderedDict(),
                    )
                    sample_records[sample_spec.sample_id]["expert_maps"][denoise_step][block_idx] = (
                        block_maps[batch_index].astype(np.float32, copy=False)
                    )

        if noise_pred.shape[1] != latents.shape[1]:
            noise_pred, _ = noise_pred.chunk(2, dim=1)

        latents = sample_scheduler.step(
            noise_pred.unsqueeze(2),
            timestep_value,
            latents,
            return_dict=False,
        )[0]

    if mean_step_count <= 0:
        raise RuntimeError("Failed to capture expert maps during denoising.")

    for block_idx, summed_maps in mean_block_sums.items():
        mean_maps = (summed_maps / mean_step_count).astype(np.float32, copy=False)
        for batch_index, sample_spec in enumerate(sample_specs):
            sample_records[sample_spec.sample_id]["mean_block_maps"][block_idx] = mean_maps[
                batch_index
            ]

    final_images = _decode_final_rgb_images(vae, latents)
    for batch_index, sample_spec in enumerate(sample_specs):
        sample_records[sample_spec.sample_id]["final_image"] = final_images[batch_index]

    return {
        "samples": list(sample_records.values()),
        "scheduler_timesteps": scheduler_timesteps,
        "block_indices": list(capture.block_indices),
        "num_routed_experts_per_block": dict(capture.num_routed_experts),
        "token_grid_size": token_grid_size,
        "mean_step_count": mean_step_count,
        "max_num_routed_experts": max(capture.num_routed_experts.values()),
    }


def save_partial_token_choice_expert_result(partial_path: Path, partial_result: dict) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(partial_result, partial_path)


def load_partial_token_choice_expert_result(partial_path: Path) -> dict:
    return torch.load(partial_path, map_location="cpu", weights_only=False)


def merge_token_choice_expert_partials(partials: list[dict]) -> dict:
    if not partials:
        raise RuntimeError("No partial expert-heatmap results were produced.")

    merged_samples = []
    merged_scheduler_timesteps = OrderedDict()
    block_indices = None
    num_routed_experts_per_block = None
    token_grid_sizes = set()
    mean_step_counts = set()
    max_num_routed_experts = set()

    for partial in partials:
        merged_samples.extend(partial["samples"])
        merged_scheduler_timesteps.update(partial["scheduler_timesteps"])
        token_grid_sizes.add(partial["token_grid_size"])
        mean_step_counts.add(partial["mean_step_count"])
        max_num_routed_experts.add(partial["max_num_routed_experts"])

        if block_indices is None:
            block_indices = list(partial["block_indices"])
        elif block_indices != list(partial["block_indices"]):
            raise RuntimeError(
                f"Inconsistent MoE block indices across partials: {block_indices} vs "
                f"{partial['block_indices']}"
            )

        partial_num_routed_experts = dict(partial["num_routed_experts_per_block"])
        if num_routed_experts_per_block is None:
            num_routed_experts_per_block = partial_num_routed_experts
        elif num_routed_experts_per_block != partial_num_routed_experts:
            raise RuntimeError(
                "Inconsistent num_routed_experts_per_block across partials: "
                f"{num_routed_experts_per_block} vs {partial_num_routed_experts}"
            )

    if len(token_grid_sizes) != 1:
        raise RuntimeError(f"Inconsistent token grid sizes across partials: {token_grid_sizes}")
    if len(mean_step_counts) != 1:
        raise RuntimeError(f"Inconsistent mean step counts across partials: {mean_step_counts}")
    if len(max_num_routed_experts) != 1:
        raise RuntimeError(
            f"Inconsistent max_num_routed_experts across partials: {max_num_routed_experts}"
        )

    merged_samples.sort(key=lambda record: record["sample_id"])
    return {
        "samples": merged_samples,
        "scheduler_timesteps": OrderedDict(sorted(merged_scheduler_timesteps.items())),
        "block_indices": block_indices or [],
        "num_routed_experts_per_block": num_routed_experts_per_block or {},
        "token_grid_size": next(iter(token_grid_sizes)),
        "mean_step_count": next(iter(mean_step_counts)),
        "max_num_routed_experts": next(iter(max_num_routed_experts)),
    }
