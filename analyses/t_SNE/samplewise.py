from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from analyses.t_SNE.imagenet_utils import build_class_id_signature
from analyses.t_SNE.pooling import pool_block_tokens
from analyses.t_SNE.sampling import (
    get_sampling_sigmas,
    retrieve_timesteps,
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


@dataclass(frozen=True)
class SampleSpec:
    sample_id: int
    class_id: int
    class_name: str
    sample_index: int


def build_sample_specs(
    class_ids: list[int],
    class_names: list[str],
    samples_per_class: int,
) -> list[SampleSpec]:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")

    sample_specs = []
    sample_id = 0
    for class_id in class_ids:
        class_name = class_names[class_id]
        for sample_index in range(samples_per_class):
            sample_specs.append(
                SampleSpec(
                    sample_id=sample_id,
                    class_id=class_id,
                    class_name=class_name,
                    sample_index=sample_index,
                )
            )
            sample_id += 1
    return sample_specs


def serialize_sample_specs(sample_specs: list[SampleSpec]) -> list[dict]:
    return [asdict(sample_spec) for sample_spec in sample_specs]


def deserialize_sample_specs(sample_specs: list[dict]) -> list[SampleSpec]:
    return [SampleSpec(**sample_spec) for sample_spec in sample_specs]


def build_samplewise_output_stem(
    class_ids: list[int],
    pool_type: str,
    seed: int,
    samples_per_class: int,
) -> str:
    class_signature = build_class_id_signature(class_ids)
    return (
        f"samplewise_pool-{pool_type}_seed{seed}"
        f"_cls{len(class_ids)}_{class_signature}_spc{samples_per_class}"
    )


def _build_latent_noise_for_specs(
    sample_specs: list[SampleSpec],
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


@torch.inference_mode()
def sample_and_collect_pooled_features(
    model,
    capture,
    runtime_cfg,
    sample_specs: list[SampleSpec],
    analysis_steps: list[int],
    seed: int,
    device,
    pool_type: str = "mean",
):
    latent_shape = (4, 1, runtime_cfg.image_size // 8, runtime_cfg.image_size // 8)
    latents = _build_latent_noise_for_specs(sample_specs, latent_shape, seed, device)
    class_tensor = torch.tensor(
        [sample_spec.class_id for sample_spec in sample_specs],
        device=device,
        dtype=torch.long,
    )
    sample_ids = np.asarray([sample_spec.sample_id for sample_spec in sample_specs], dtype=np.int64)
    class_ids = np.asarray([sample_spec.class_id for sample_spec in sample_specs], dtype=np.int64)

    sample_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=runtime_cfg.num_train_timesteps,
        shift=runtime_cfg.shift,
    )
    sampling_sigmas = get_sampling_sigmas(runtime_cfg.sample_steps, runtime_cfg.sample_shift)
    timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)

    results = OrderedDict()
    scheduler_timesteps = OrderedDict()
    autocast_dtype = getattr(runtime_cfg, "val_param_dtype", torch.float32)
    analysis_step_set = set(analysis_steps)

    if autocast_dtype in (torch.float16, torch.bfloat16):
        autocast_context_factory = lambda: torch.cuda.amp.autocast(dtype=autocast_dtype)
    else:
        from contextlib import nullcontext

        autocast_context_factory = nullcontext

    for timestep_index, timestep_value in enumerate(timesteps):
        denoise_step = timestep_index + 1
        timestep_tensor = torch.full(
            (latents.size(0),),
            float(timestep_value.item()),
            device=device,
            dtype=timesteps.dtype,
        )

        should_capture = denoise_step in analysis_step_set
        if should_capture:
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

        if should_capture:
            batch_records = capture.disable_and_collect()
            results.setdefault(denoise_step, OrderedDict())
            scheduler_timesteps[denoise_step] = float(timestep_value.item())
            for block_idx, payload in batch_records.items():
                pooled_features = pool_block_tokens(payload["features"], pool_type=pool_type).numpy()
                results[denoise_step][block_idx] = {
                    "vectors": pooled_features,
                    "class_ids": class_ids.copy(),
                    "sample_ids": sample_ids.copy(),
                }

        if noise_pred.shape[1] != latents.shape[1]:
            noise_pred, _ = noise_pred.chunk(2, dim=1)

        latents = sample_scheduler.step(
            noise_pred.unsqueeze(2),
            timestep_value,
            latents,
            return_dict=False,
        )[0]

    return {
        "results": results,
        "scheduler_timesteps": scheduler_timesteps,
        "block_indices": list(capture.block_indices),
    }


def save_partial_samplewise_result(
    partial_path: Path,
    partial_result: dict,
) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(partial_result, partial_path)


def load_partial_samplewise_result(partial_path: Path) -> dict:
    return torch.load(partial_path, map_location="cpu", weights_only=False)


def merge_samplewise_partials(partials: list[dict]) -> dict:
    merged_results = OrderedDict()
    merged_scheduler_timesteps = OrderedDict()
    merged_block_indices = set()

    for partial in partials:
        merged_block_indices.update(partial["block_indices"])
        for denoise_step, timestep_value in partial["scheduler_timesteps"].items():
            merged_scheduler_timesteps[denoise_step] = timestep_value
        for denoise_step, block_records in partial["results"].items():
            merged_results.setdefault(denoise_step, OrderedDict())
            for block_idx, record in block_records.items():
                entry = merged_results[denoise_step].setdefault(
                    block_idx,
                    {
                        "vectors": [],
                        "class_ids": [],
                        "sample_ids": [],
                    },
                )
                entry["vectors"].append(record["vectors"])
                entry["class_ids"].append(record["class_ids"])
                entry["sample_ids"].append(record["sample_ids"])

    for denoise_step, block_records in merged_results.items():
        for block_idx, record in block_records.items():
            vectors = np.concatenate(record["vectors"], axis=0)
            class_ids = np.concatenate(record["class_ids"], axis=0)
            sample_ids = np.concatenate(record["sample_ids"], axis=0)
            sort_order = np.argsort(sample_ids)
            block_records[block_idx] = {
                "vectors": vectors[sort_order],
                "class_ids": class_ids[sort_order],
                "sample_ids": sample_ids[sort_order],
            }

    return {
        "results": merged_results,
        "scheduler_timesteps": OrderedDict(sorted(merged_scheduler_timesteps.items())),
        "block_indices": sorted(merged_block_indices),
    }
