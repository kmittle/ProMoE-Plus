"""Causal horizontal-flip audit for top-1 MoE routing."""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyses.denoising_regret.probe import (
    _all_router_weights,
    _configure_torch_threads,
    _extract_prediction,
    _load_checkpoint_model,
    _load_latent,
    _per_sample_mse,
)
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)

from .probe import (
    INTERVENTION_NAMES,
    RouteInputCapture,
    _capture_native_forward,
    _forced_route_matrices,
    _random_matched_routes,
    _relative_mse_changes,
    _route_agreement,
    _route_margin_metrics,
)


def _flip_token_grid(tensor, grid_size):
    if tensor.ndim < 1 or tensor.shape[0] != grid_size * grid_size:
        raise ValueError("Token tensor does not match the square route grid")
    grid_shape = (grid_size, grid_size, *tensor.shape[1:])
    return tensor.reshape(grid_shape).flip(1).reshape(tensor.shape)


def _build_flip_route_references(original_ids, flipped_ids, grid_size):
    if original_ids.ndim != 1 or flipped_ids.ndim != 1:
        raise ValueError("Route IDs must be flat one-dimensional tensors")
    if original_ids.shape != flipped_ids.shape:
        raise ValueError("Original and flipped route maps must align")
    content_follow = _flip_token_grid(original_ids, grid_size)
    position_follow = original_ids.clone()
    valid = torch.ones_like(original_ids, dtype=torch.bool)
    return content_follow, position_follow, valid


def _hidden_flip_metrics(original_hidden, flipped_hidden, grid_size):
    if original_hidden.shape != flipped_hidden.shape:
        raise ValueError("Original and flipped hidden states must align")
    content_reference = _flip_token_grid(original_hidden, grid_size)
    position_reference = original_hidden
    content_cosine = F.cosine_similarity(
        flipped_hidden.float(),
        content_reference.float(),
        dim=-1,
    )
    position_cosine = F.cosine_similarity(
        flipped_hidden.float(),
        position_reference.float(),
        dim=-1,
    )
    content_relative_l2 = (
        (flipped_hidden.float() - content_reference.float()).norm(dim=-1)
        / content_reference.float().norm(dim=-1).clamp_min(1e-12)
    )
    position_relative_l2 = (
        (flipped_hidden.float() - position_reference.float()).norm(dim=-1)
        / position_reference.float().norm(dim=-1).clamp_min(1e-12)
    )
    return {
        "content_follow_cosine_mean": float(content_cosine.mean().item()),
        "position_follow_cosine_mean": float(position_cosine.mean().item()),
        "content_follow_relative_l2_mean": float(
            content_relative_l2.mean().item()
        ),
        "position_follow_relative_l2_mean": float(
            position_relative_l2.mean().item()
        ),
    }


def _full_model_flip_mse(original_prediction, flipped_prediction):
    expected = torch.flip(original_prediction, dims=(-1,))
    return float(
        (flipped_prediction.double() - expected.double())
        .square()
        .mean()
        .item()
    )


def _mean_present(records, path):
    values = []
    for record in records:
        value = record
        for key in path:
            value = value[key]
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else None


def _summarize_flip_records(records):
    if not records:
        raise ValueError("At least one flip record is required")
    summary = {
        "num_cells": len(records),
        "content_follow_route_agreement_mean": float(np.mean([
            record["route_agreement"]["content_follow"]["agreement"]
            for record in records
        ])),
        "position_follow_route_agreement_mean": float(np.mean([
            record["route_agreement"]["position_follow"]["agreement"]
            for record in records
        ])),
        "content_follow_hidden_cosine_mean": float(np.mean([
            record["hidden_flip"]["content_follow_cosine_mean"]
            for record in records
        ])),
        "position_follow_hidden_cosine_mean": float(np.mean([
            record["hidden_flip"]["position_follow_cosine_mean"]
            for record in records
        ])),
        "content_changed_rate_mean": float(np.mean([
            record["route_margin"]["changed_rate"] for record in records
        ])),
        "native_minus_content_score_changed_mean": _mean_present(
            records,
            ("route_margin", "native_minus_content_changed", "mean"),
        ),
        "content_expert_rank_changed_mean": _mean_present(
            records,
            ("route_margin", "content_expert_rank_changed", "mean"),
        ),
        "native_top1_margin_changed_mean": _mean_present(
            records,
            ("route_margin", "native_top1_margin_changed", "mean"),
        ),
        "native_top1_margin_unchanged_mean": _mean_present(
            records,
            ("route_margin", "native_top1_margin_unchanged", "mean"),
        ),
        "max_abs_noop_mse_change": float(max(
            abs(record["mse_change"]["noop_native"])
            for record in records
        )),
        "max_abs_noop_output_change": float(max(
            record["numerical_controls"]["noop_max_abs_output_change"]
            for record in records
        )),
        "max_abs_forced_unforced_output_change": float(max(
            record["numerical_controls"]["forced_unforced_max_abs_output_change"]
            for record in records
        )),
    }
    for name in ("content_follow", "position_follow"):
        changes = np.asarray(
            [record["relative_mse_change"][name] for record in records],
            dtype=np.float64,
        )
        summary[f"{name}_mean_relative_mse_change"] = float(changes.mean())
        summary[f"{name}_median_relative_mse_change"] = float(
            np.median(changes)
        )
        summary[f"{name}_better_rate"] = float((changes < 0).mean())
    random_records = [
        record for record in records if record["random_control_available"]
    ]
    summary["random_control_valid_cells"] = len(random_records)
    random_changes = np.asarray([
        record["relative_mse_change"]["random_matched"]
        for record in random_records
    ], dtype=np.float64)
    summary["random_matched_mean_relative_mse_change"] = (
        float(random_changes.mean()) if random_records else None
    )
    summary["random_matched_median_relative_mse_change"] = (
        float(np.median(random_changes)) if random_records else None
    )
    summary["random_matched_better_rate"] = (
        float((random_changes < 0).mean()) if random_records else None
    )
    summary["content_follow_beats_position_rate"] = float(np.mean([
        record["mse"]["content_follow"] < record["mse"]["position_follow"]
        for record in records
    ]))
    summary["content_follow_beats_random_rate"] = (
        float(np.mean([
            record["mse"]["content_follow"] < record["mse"]["random_matched"]
            for record in random_records
        ]))
        if random_records
        else None
    )
    return summary


def _probe_flip_cell(
    model,
    moe_layer,
    capture,
    clean_latent,
    noise,
    label,
    sigma,
    num_train_timesteps,
    num_routed_experts,
    generator,
):
    sigma_tensor = torch.tensor(
        float(sigma),
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    timestep = torch.full(
        (1,),
        float(sigma) * num_train_timesteps,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    original_noised = (1.0 - sigma_tensor) * clean_latent + sigma_tensor * noise
    original_target = (noise - clean_latent).squeeze(2)
    flipped_clean = torch.flip(clean_latent, dims=(-1,))
    flipped_noise = torch.flip(noise, dims=(-1,))
    flipped_noised = (
        (1.0 - sigma_tensor) * flipped_clean + sigma_tensor * flipped_noise
    )
    flipped_target = (flipped_noise - flipped_clean).squeeze(2)

    (
        original_output,
        original_hidden,
        _,
        original_indices,
    ) = _capture_native_forward(
        model,
        moe_layer,
        capture,
        original_noised,
        timestep,
        label,
    )
    (
        flipped_output,
        flipped_hidden,
        flipped_weights,
        flipped_indices,
    ) = _capture_native_forward(
        model,
        moe_layer,
        capture,
        flipped_noised,
        timestep,
        label,
    )
    original_prediction = _extract_prediction(
        original_output,
        original_target.shape[1],
    )
    flipped_prediction = _extract_prediction(
        flipped_output,
        flipped_target.shape[1],
    )

    original_ids = original_indices[0, :, 0]
    flipped_ids = flipped_indices[0, :, 0]
    num_tokens = int(original_ids.numel())
    grid_size = math.isqrt(num_tokens)
    if grid_size * grid_size != num_tokens:
        raise RuntimeError("Flip probe requires a square token grid")
    content_ids, position_ids, valid_tokens = _build_flip_route_references(
        original_ids,
        flipped_ids,
        grid_size,
    )
    random_ids, content_changed, random_control_available = _random_matched_routes(
        flipped_ids,
        content_ids,
        valid_tokens,
        generator,
    )
    all_router_scores = _all_router_weights(moe_layer, flipped_hidden)[0]
    route_margin = _route_margin_metrics(
        all_router_scores,
        flipped_ids,
        content_ids,
        valid_tokens,
    )

    route_matrix = torch.stack([
        flipped_ids,
        flipped_ids,
        content_ids,
        position_ids,
        random_ids,
    ])
    intervention_count = len(INTERVENTION_NAMES)
    batch_noised = flipped_noised.repeat(
        intervention_count,
        1,
        1,
        1,
        1,
    )
    batch_timestep = timestep.repeat(intervention_count)
    batch_label = label.repeat(intervention_count)
    batch_target = flipped_target.repeat(intervention_count, 1, 1, 1)
    with torch.inference_mode(), _forced_route_matrices(
        moe_layer,
        route_matrix,
    ):
        intervention_output = model(
            batch_noised,
            batch_timestep,
            context=batch_label,
        )
    intervention_prediction = _extract_prediction(
        intervention_output,
        flipped_target.shape[1],
    )
    losses = _per_sample_mse(intervention_prediction, batch_target)
    native_loss = losses[0]
    mse = {
        name: float(losses[index].item())
        for index, name in enumerate(INTERVENTION_NAMES)
    }
    mse_change = {
        name: float((losses[index] - native_loss).item())
        for index, name in enumerate(INTERVENTION_NAMES)
    }
    relative_mse_change = _relative_mse_changes(losses, INTERVENTION_NAMES)

    random_changed = valid_tokens & (random_ids != flipped_ids)
    content_hist = torch.bincount(
        content_ids[content_changed],
        minlength=num_routed_experts,
    )
    random_hist = torch.bincount(
        random_ids[random_changed],
        minlength=num_routed_experts,
    )
    if not torch.equal(content_changed, random_changed):
        raise RuntimeError("Random control must match content disagreement support")
    if not torch.equal(content_hist, random_hist):
        raise RuntimeError("Random control must match replacement-expert counts")

    unforced_flipped_loss = _per_sample_mse(
        flipped_prediction,
        flipped_target,
    )[0]
    original_loss = _per_sample_mse(
        original_prediction,
        original_target,
    )[0]
    return {
        "sigma": float(sigma),
        "timestep": float(timestep.item()),
        "valid_route_tokens": int(valid_tokens.sum().item()),
        "content_changed_tokens": int(content_changed.sum().item()),
        "random_control_available": random_control_available,
        "position_changed_tokens": int(
            (valid_tokens & (position_ids != flipped_ids)).sum().item()
        ),
        "random_differs_from_content_rate": (
            float(
                (random_ids[content_changed] != content_ids[content_changed])
                .float()
                .mean()
                .item()
            )
            if random_control_available
            else None
        ),
        "route_agreement": {
            "content_follow": _route_agreement(
                flipped_ids,
                content_ids,
                valid_tokens,
                num_routed_experts,
            ),
            "position_follow": _route_agreement(
                flipped_ids,
                position_ids,
                valid_tokens,
                num_routed_experts,
            ),
        },
        "route_margin": route_margin,
        "hidden_flip": _hidden_flip_metrics(
            original_hidden[0],
            flipped_hidden[0],
            grid_size,
        ),
        "native_router_weight": {
            "mean": float(flipped_weights[0, :, 0].float().mean().item()),
            "std": float(flipped_weights[0, :, 0].float().std().item()),
            "min": float(flipped_weights[0, :, 0].float().min().item()),
            "max": float(flipped_weights[0, :, 0].float().max().item()),
        },
        "original_native_mse": float(original_loss.item()),
        "flipped_native_unforced_mse": float(unforced_flipped_loss.item()),
        "mse": mse,
        "mse_change": mse_change,
        "relative_mse_change": relative_mse_change,
        "full_model_flip_mse": _full_model_flip_mse(
            original_prediction,
            flipped_prediction,
        ),
        "numerical_controls": {
            "noop_max_abs_output_change": float(
                (intervention_prediction[1] - intervention_prediction[0])
                .abs()
                .max()
                .item()
            ),
            "forced_unforced_max_abs_output_change": float(
                (intervention_prediction[0] - flipped_prediction[0])
                .abs()
                .max()
                .item()
            ),
            "forced_unforced_mse_change": float(
                (native_loss - unforced_flipped_loss).item()
            ),
            "content_random_changed_support_equal": True,
            "content_random_replacement_histogram_equal": True,
            "random_control_available": random_control_available,
        },
    }


def run_routing_flip_probe(
    checkpoint_path,
    latent_path,
    label,
    sigmas=(0.276, 0.5, 0.724),
    block_index=3,
    latent_key="latent",
    seed=0,
    device="cpu",
    num_threads=8,
    weights_checkpoint_path=None,
):
    checkpoint_path = Path(checkpoint_path).resolve()
    weights_checkpoint_path = Path(
        weights_checkpoint_path or checkpoint_path
    ).resolve()
    latent_path = Path(latent_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not weights_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Weights checkpoint does not exist: {weights_checkpoint_path}"
        )
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent does not exist: {latent_path}")
    sigmas = [float(sigma) for sigma in sigmas]
    if (
        not sigmas
        or len(sigmas) != len(set(sigmas))
        or any(not 0 < sigma < 1 for sigma in sigmas)
    ):
        raise ValueError("Sigmas must be unique and strictly between zero and one")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")

    thread_config = _configure_torch_threads(num_threads)
    device = torch.device(device)
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    checkpoint_step = parse_checkpoint_step(checkpoint_path)
    runtime_cfg = load_runtime_cfg(config_path)
    if not 0 <= label < runtime_cfg.num_classes:
        raise ValueError(
            f"label must be in [0, {runtime_cfg.num_classes - 1}], got {label}"
        )
    model, state_name, weights_checkpoint_step, load_seconds = (
        _load_checkpoint_model(runtime_cfg, weights_checkpoint_path, device)
    )
    if weights_checkpoint_step != checkpoint_step:
        raise ValueError(
            f"Loaded checkpoint step {weights_checkpoint_step} does not match "
            f"the canonical checkpoint step {checkpoint_step}"
        )
    if not 0 <= block_index < len(model.blocks):
        raise ValueError(f"block_index {block_index} is outside the model")
    block = model.blocks[block_index]
    if not block.use_moe:
        raise ValueError(f"block {block_index} is not an MoE block")
    moe_layer = block.mlp
    if moe_layer.top_k != 1:
        raise ValueError("Flip routing probe requires top_k == 1")

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    clean_latent = _load_latent(latent_path, latent_key, seed, device)
    torch.manual_seed(seed + 1)
    noise = torch.randn_like(clean_latent)
    label_tensor = torch.tensor([label], device=device, dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 2)

    capture = RouteInputCapture(moe_layer)
    records = []
    probe_start = time.perf_counter()
    try:
        for sigma in sigmas:
            records.append(_probe_flip_cell(
                model=model,
                moe_layer=moe_layer,
                capture=capture,
                clean_latent=clean_latent,
                noise=noise,
                label=label_tensor,
                sigma=sigma,
                num_train_timesteps=runtime_cfg.num_train_timesteps,
                num_routed_experts=moe_layer.num_routed_experts,
                generator=generator,
            ))
    finally:
        capture.close()
    probe_seconds = time.perf_counter() - probe_start

    result = {
        "routing_flip_probe_version": 1,
        "diagnostic_scope": (
            "teacher-forced fixed-compute route intervention; not a direct FID claim"
        ),
        "transform": (
            "exact horizontal flip of the sampled clean latent and paired noise"
        ),
        "training_relevance": (
            "LatentFolder samples original and horizontal-flip latent keys with "
            "equal probability during training"
        ),
        "intervention": (
            "force only top-1 expert identity at one MoE block while preserving "
            "the flipped input's native route weight and expert width"
        ),
        "checkpoint": str(checkpoint_path),
        "weights_checkpoint": str(weights_checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "weights_checkpoint_step": weights_checkpoint_step,
        "checkpoint_state": state_name,
        "config": str(config_path),
        "model_name": runtime_cfg.model_name,
        "latent": str(latent_path),
        "latent_key": latent_key,
        "label": int(label),
        "block_index": int(block_index),
        "sigmas": sigmas,
        "seed": int(seed),
        "device": str(device),
        "num_threads": int(num_threads),
        "thread_config": thread_config,
        "model_load_seconds": load_seconds,
        "probe_seconds": probe_seconds,
        "summary": _summarize_flip_records(records),
        "per_sigma": {
            str(float(sigma)): _summarize_flip_records([
                record
                for record in records
                if record["sigma"] == float(sigma)
            ])
            for sigma in sigmas
        },
        "records": records,
    }
    del model
    gc.collect()
    return result
