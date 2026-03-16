from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import inspect

import numpy as np
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from analyses.t_SNE.model_registry import model_dict


def get_sampling_sigmas(sampling_steps: int, shift: float) -> np.ndarray:
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"{scheduler.__class__.__name__} does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_sigmas:
            raise ValueError(
                f"{scheduler.__class__.__name__} does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def compute_analysis_steps(sample_steps: int, analysis_every: int) -> list[int]:
    if analysis_every <= 0:
        raise ValueError("analysis_every must be positive.")
    steps = list(range(analysis_every, sample_steps + 1, analysis_every))
    if sample_steps not in steps:
        steps.append(sample_steps)
    if not steps:
        steps = [sample_steps]
    return steps


def chunked(sequence: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [sequence[index:index + chunk_size] for index in range(0, len(sequence), chunk_size)]


def build_model(runtime_cfg, ckpt_path: str, device: torch.device):
    model_class, config_name = model_dict[runtime_cfg.model_name]
    model_cfg = getattr(runtime_cfg, config_name)
    model = model_class(**model_cfg)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["ema_model_state_dict"],
        strict=False,
    )
    model = model.to(device)
    model.eval()
    return model, missing_keys, unexpected_keys


def _build_latent_noise(class_ids: list[int], latent_shape: tuple[int, ...], seed: int, device):
    latents = []
    for class_id in class_ids:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + int(class_id))
        latents.append(torch.randn(*latent_shape, generator=generator, device=device))
    return torch.stack(latents, dim=0)


@torch.inference_mode()
def sample_and_capture_batch(
    model,
    capture,
    runtime_cfg,
    class_ids: list[int],
    analysis_steps: list[int],
    seed: int,
    device,
):
    latent_shape = (4, 1, runtime_cfg.image_size // 8, runtime_cfg.image_size // 8)
    latents = _build_latent_noise(class_ids, latent_shape, seed, device)
    class_tensor = torch.tensor(class_ids, device=device, dtype=torch.long)

    sample_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=runtime_cfg.num_train_timesteps,
        shift=runtime_cfg.shift,
    )
    sampling_sigmas = get_sampling_sigmas(runtime_cfg.sample_steps, runtime_cfg.sample_shift)
    timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)

    results = {
        class_id: OrderedDict()
        for class_id in class_ids
    }

    autocast_dtype = getattr(runtime_cfg, "val_param_dtype", torch.float32)
    analysis_step_set = set(analysis_steps)

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

        if autocast_dtype in (torch.float16, torch.bfloat16):
            autocast_context = torch.cuda.amp.autocast(dtype=autocast_dtype)
        else:
            autocast_context = nullcontext()

        with autocast_context:
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
            for block_idx, payload in batch_records.items():
                features = payload["features"].numpy()
                labels = payload["labels"].numpy()
                scheduler_timestep = payload["scheduler_timestep"]
                for batch_index, class_id in enumerate(class_ids):
                    if denoise_step not in results[class_id]:
                        results[class_id][denoise_step] = OrderedDict()
                    results[class_id][denoise_step][block_idx] = {
                        "features": features[batch_index],
                        "labels": labels[batch_index],
                        "scheduler_timestep": scheduler_timestep,
                    }

        if noise_pred.shape[1] != latents.shape[1]:
            noise_pred, _ = noise_pred.chunk(2, dim=1)

        latents = sample_scheduler.step(
            noise_pred.unsqueeze(2),
            timestep_value,
            latents,
            return_dict=False,
        )[0]

    return results
