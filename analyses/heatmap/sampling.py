from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from analyses.heatmap.sample_specs import HeatmapSampleSpec
from analyses.t_SNE.sampling import get_sampling_sigmas, retrieve_timesteps


def _build_latent_noise_for_specs(
    sample_specs: list[HeatmapSampleSpec],
    latent_shape: tuple[int, ...],
    seed: int,
    device,
) -> torch.Tensor:
    latents = []
    for sample_spec in sample_specs:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + sample_spec.class_id * 1000 + sample_spec.sample_index)
        latents.append(torch.randn(*latent_shape, generator=generator, device=device))
    return torch.stack(latents, dim=0)


def _decode_final_rgb_images(vae, latents: torch.Tensor) -> np.ndarray:
    decoded = vae.decode(latents.squeeze(2).float() / 0.18215).sample
    decoded = torch.clamp(127.5 * decoded + 128.0, 0, 255)
    return decoded.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()


@torch.inference_mode()
def sample_and_collect_repa_heatmaps(
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
                "weight_maps": OrderedDict(),
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
    mean_weight_sum = None
    mean_weight_count = 0
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

        payload = capture.disable_and_collect()
        token_grid_size = payload["grid_size"]
        weight_maps = payload["weights"].numpy().reshape(
            len(sample_specs), token_grid_size, token_grid_size
        )
        if mean_weight_sum is None:
            mean_weight_sum = weight_maps.astype(np.float64, copy=True)
        else:
            mean_weight_sum += weight_maps
        mean_weight_count += 1

        if should_store_snapshot:
            scheduler_timesteps[denoise_step] = float(timestep_value.item())
            for batch_index, sample_spec in enumerate(sample_specs):
                sample_records[sample_spec.sample_id]["weight_maps"][denoise_step] = weight_maps[
                    batch_index
                ].astype(np.float32, copy=False)

        if noise_pred.shape[1] != latents.shape[1]:
            noise_pred, _ = noise_pred.chunk(2, dim=1)

        latents = sample_scheduler.step(
            noise_pred.unsqueeze(2),
            timestep_value,
            latents,
            return_dict=False,
        )[0]

    final_images = _decode_final_rgb_images(vae, latents)
    if mean_weight_sum is None or mean_weight_count <= 0:
        raise RuntimeError("Failed to accumulate token weights across denoising steps.")
    mean_weight_maps = (mean_weight_sum / mean_weight_count).astype(np.float32, copy=False)
    for batch_index, sample_spec in enumerate(sample_specs):
        sample_records[sample_spec.sample_id]["final_image"] = final_images[batch_index]
        sample_records[sample_spec.sample_id]["mean_weight_map"] = mean_weight_maps[batch_index]

    return {
        "samples": list(sample_records.values()),
        "scheduler_timesteps": scheduler_timesteps,
        "capture_block_index": capture.block_index,
        "encoder_depth": capture.encoder_depth,
        "token_grid_size": token_grid_size,
        "mean_weight_count": mean_weight_count,
    }


def save_partial_heatmap_result(partial_path: Path, partial_result: dict) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(partial_result, partial_path)


def load_partial_heatmap_result(partial_path: Path) -> dict:
    return torch.load(partial_path, map_location="cpu", weights_only=False)


def merge_heatmap_partials(partials: list[dict]) -> dict:
    merged_samples = []
    merged_scheduler_timesteps = OrderedDict()
    capture_block_indices = set()
    encoder_depths = set()
    token_grid_sizes = set()
    mean_weight_counts = set()

    for partial in partials:
        merged_samples.extend(partial["samples"])
        merged_scheduler_timesteps.update(partial["scheduler_timesteps"])
        capture_block_indices.add(partial["capture_block_index"])
        encoder_depths.add(partial["encoder_depth"])
        token_grid_sizes.add(partial["token_grid_size"])
        mean_weight_counts.add(partial["mean_weight_count"])

    if len(capture_block_indices) != 1:
        raise RuntimeError(f"Inconsistent capture block indices across partials: {capture_block_indices}")
    if len(encoder_depths) != 1:
        raise RuntimeError(f"Inconsistent encoder depths across partials: {encoder_depths}")
    if len(token_grid_sizes) != 1:
        raise RuntimeError(f"Inconsistent token grid sizes across partials: {token_grid_sizes}")
    if len(mean_weight_counts) != 1:
        raise RuntimeError(f"Inconsistent mean weight counts across partials: {mean_weight_counts}")

    merged_samples.sort(key=lambda record: record["sample_id"])
    return {
        "samples": merged_samples,
        "scheduler_timesteps": OrderedDict(sorted(merged_scheduler_timesteps.items())),
        "capture_block_index": next(iter(capture_block_indices)),
        "encoder_depth": next(iter(encoder_depths)),
        "token_grid_size": next(iter(token_grid_sizes)),
        "mean_weight_count": next(iter(mean_weight_counts)),
    }
