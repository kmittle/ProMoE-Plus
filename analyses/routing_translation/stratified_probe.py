"""Margin-stratified causal audit for translation-disrupted MoE routes."""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import numpy as np
import torch

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
    RouteInputCapture,
    _build_route_references,
    _capture_native_forward,
    _forced_route_matrices,
    _hidden_translation_metrics,
    _random_matched_routes,
    _relative_mse_changes,
    _route_agreement,
    _route_margin_metrics,
    _translate_spatial,
    _valid_translation_mse,
    _validate_shifts,
)


STRATUM_NAMES = (
    "low_margin",
    "high_margin",
    "content_top2",
    "content_rank3plus",
)
INTERVENTION_NAMES = (
    "native",
    "noop_native",
    *(name for stratum in STRATUM_NAMES for name in (
        f"{stratum}_content",
        f"{stratum}_random",
    )),
)


def _build_stratum_masks(router_scores, native_ids, content_ids, valid_mask):
    if router_scores.ndim != 2 or router_scores.shape[0] != native_ids.numel():
        raise ValueError("Router scores must align with the flat route IDs")
    if not (
        native_ids.shape == content_ids.shape == valid_mask.shape
        and native_ids.ndim == 1
    ):
        raise ValueError("Route IDs and valid mask must be aligned vectors")
    if router_scores.shape[1] < 2:
        raise ValueError("Stratification requires at least two routed experts")
    if not torch.equal(router_scores.argmax(dim=-1), native_ids):
        raise ValueError("Native IDs must be the router-score argmax")

    changed = valid_mask & (native_ids != content_ids)
    top_two = torch.topk(router_scores, k=2, dim=-1).values
    top1_margin = top_two[:, 0] - top_two[:, 1]
    rows = torch.arange(native_ids.numel(), device=router_scores.device)
    content_scores = router_scores[rows, content_ids]
    content_rank = (
        (router_scores > content_scores.unsqueeze(-1)).sum(dim=-1) + 1
    )

    changed_indices = torch.where(changed)[0]
    low_margin = torch.zeros_like(changed)
    high_margin = torch.zeros_like(changed)
    if changed_indices.numel():
        order = torch.argsort(
            top1_margin[changed_indices],
            stable=True,
        )
        ordered_indices = changed_indices[order]
        low_count = (int(ordered_indices.numel()) + 1) // 2
        low_margin[ordered_indices[:low_count]] = True
        high_margin[ordered_indices[low_count:]] = True

    content_top2 = changed & (content_rank <= 2)
    content_rank3plus = changed & (content_rank >= 3)
    if not torch.equal(low_margin | high_margin, changed):
        raise RuntimeError("Margin strata must cover every changed route")
    if (low_margin & high_margin).any():
        raise RuntimeError("Margin strata must be disjoint")
    if not torch.equal(content_top2 | content_rank3plus, changed):
        raise RuntimeError("Content-rank strata must cover every changed route")
    if (content_top2 & content_rank3plus).any():
        raise RuntimeError("Content-rank strata must be disjoint")
    return {
        "low_margin": low_margin,
        "high_margin": high_margin,
        "content_top2": content_top2,
        "content_rank3plus": content_rank3plus,
    }, {
        "changed": changed,
        "top1_margin": top1_margin,
        "content_rank": content_rank,
        "native_content_deficit": (
            router_scores[rows, native_ids] - content_scores
        ),
    }


def _build_stratum_routes(
    native_ids,
    content_ids,
    valid_mask,
    stratum_masks,
    generator,
    num_routed_experts,
):
    routes = []
    controls = {}
    for name in STRATUM_NAMES:
        mask = stratum_masks[name]
        content_route = native_ids.clone()
        content_route[mask] = content_ids[mask]
        random_route, random_changed, random_control_available = \
            _random_matched_routes(
            native_ids,
            content_route,
            valid_mask,
            generator,
            )
        if not torch.equal(random_changed, mask):
            raise RuntimeError(f"{name} random control changed its support")
        content_hist = torch.bincount(
            content_route[mask],
            minlength=num_routed_experts,
        )
        random_hist = torch.bincount(
            random_route[mask],
            minlength=num_routed_experts,
        )
        if not torch.equal(content_hist, random_hist):
            raise RuntimeError(
                f"{name} random control changed the replacement histogram"
            )
        routes.extend((content_route, random_route))
        controls[name] = {
            "support_equal": True,
            "replacement_histogram_equal": True,
            "random_control_available": random_control_available,
            "random_differs_from_content_rate": (
                float(
                    (random_route[mask] != content_route[mask])
                    .float()
                    .mean()
                    .item()
                )
                if random_control_available
                else None
            ),
        }
    return routes, controls


def _masked_summary(values, mask):
    selected = values[mask].float()
    if selected.numel() == 0:
        return {"mean": None, "median": None}
    return {
        "mean": float(selected.mean().item()),
        "median": float(selected.median().item()),
    }


def _stratum_diagnostics(stratum_masks, router_diagnostics):
    changed_count = int(router_diagnostics["changed"].sum().item())
    diagnostics = {}
    for name, mask in stratum_masks.items():
        count = int(mask.sum().item())
        diagnostics[name] = {
            "tokens": count,
            "fraction_of_changed": (
                float(count / changed_count) if changed_count else None
            ),
            "native_top1_margin": _masked_summary(
                router_diagnostics["top1_margin"],
                mask,
            ),
            "content_expert_rank": _masked_summary(
                router_diagnostics["content_rank"],
                mask,
            ),
            "native_content_deficit": _masked_summary(
                router_diagnostics["native_content_deficit"],
                mask,
            ),
        }
    return diagnostics


def _probe_stratified_cell(
    model,
    moe_layer,
    capture,
    clean_latent,
    noise,
    label,
    sigma,
    num_train_timesteps,
    shift,
    patch_size,
    num_routed_experts,
    generator,
):
    dy, dx = shift
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
    shifted_clean = _translate_spatial(clean_latent, dy, dx)
    shifted_noise = _translate_spatial(noise, dy, dx)
    shifted_noised = (
        (1.0 - sigma_tensor) * shifted_clean + sigma_tensor * shifted_noise
    )
    shifted_target = (shifted_noise - shifted_clean).squeeze(2)

    original_output, original_hidden, _, original_indices = (
        _capture_native_forward(
            model,
            moe_layer,
            capture,
            original_noised,
            timestep,
            label,
        )
    )
    shifted_output, shifted_hidden, shifted_weights, shifted_indices = (
        _capture_native_forward(
            model,
            moe_layer,
            capture,
            shifted_noised,
            timestep,
            label,
        )
    )
    original_prediction = _extract_prediction(
        original_output,
        original_target.shape[1],
    )
    shifted_prediction = _extract_prediction(
        shifted_output,
        shifted_target.shape[1],
    )

    original_ids = original_indices[0, :, 0]
    shifted_ids = shifted_indices[0, :, 0]
    grid_size = math.isqrt(int(original_ids.numel()))
    if grid_size * grid_size != original_ids.numel():
        raise RuntimeError("Stratified probe requires a square token grid")
    token_shift = (dy // patch_size, dx // patch_size)
    content_ids, _, valid_tokens = _build_route_references(
        original_ids,
        shifted_ids,
        grid_size,
        token_shift,
    )
    router_scores = _all_router_weights(
        moe_layer,
        shifted_hidden,
        timestep,
    )[0]
    stratum_masks, router_diagnostics = _build_stratum_masks(
        router_scores,
        shifted_ids,
        content_ids,
        valid_tokens,
    )
    stratum_routes, stratum_controls = _build_stratum_routes(
        shifted_ids,
        content_ids,
        valid_tokens,
        stratum_masks,
        generator,
        num_routed_experts,
    )
    route_matrix = torch.stack([
        shifted_ids,
        shifted_ids,
        *stratum_routes,
    ])
    if route_matrix.shape[0] != len(INTERVENTION_NAMES):
        raise RuntimeError("Intervention names and route matrices diverged")

    intervention_count = len(INTERVENTION_NAMES)
    batch_noised = shifted_noised.repeat(intervention_count, 1, 1, 1, 1)
    batch_timestep = timestep.repeat(intervention_count)
    batch_label = label.repeat(intervention_count)
    batch_target = shifted_target.repeat(intervention_count, 1, 1, 1)
    with torch.inference_mode(), _forced_route_matrices(moe_layer, route_matrix):
        intervention_output = model(
            batch_noised,
            batch_timestep,
            context=batch_label,
        )
    intervention_prediction = _extract_prediction(
        intervention_output,
        shifted_target.shape[1],
    )
    losses = _per_sample_mse(intervention_prediction, batch_target)
    native_loss = losses[0]
    mse = {
        name: float(losses[index].item())
        for index, name in enumerate(INTERVENTION_NAMES)
    }
    relative_mse_change = _relative_mse_changes(losses, INTERVENTION_NAMES)
    mse_change = {
        name: float((losses[index] - native_loss).item())
        for index, name in enumerate(INTERVENTION_NAMES)
    }

    unforced_shifted_loss = _per_sample_mse(
        shifted_prediction,
        shifted_target,
    )[0]
    original_loss = _per_sample_mse(
        original_prediction,
        original_target,
    )[0]
    return {
        "sigma": float(sigma),
        "timestep": float(timestep.item()),
        "shift_latent": [int(dy), int(dx)],
        "shift_tokens": [int(token_shift[0]), int(token_shift[1])],
        "valid_route_tokens": int(valid_tokens.sum().item()),
        "content_changed_tokens": int(
            router_diagnostics["changed"].sum().item()
        ),
        "route_agreement": _route_agreement(
            shifted_ids,
            content_ids,
            valid_tokens,
            num_routed_experts,
        ),
        "route_margin": _route_margin_metrics(
            router_scores,
            shifted_ids,
            content_ids,
            valid_tokens,
        ),
        "strata": _stratum_diagnostics(
            stratum_masks,
            router_diagnostics,
        ),
        "hidden_translation": _hidden_translation_metrics(
            original_hidden[0],
            shifted_hidden[0],
            grid_size,
            token_shift,
        ),
        "native_router_weight": {
            "mean": float(shifted_weights[0, :, 0].float().mean().item()),
            "std": float(shifted_weights[0, :, 0].float().std().item()),
            "min": float(shifted_weights[0, :, 0].float().min().item()),
            "max": float(shifted_weights[0, :, 0].float().max().item()),
        },
        "original_native_mse": float(original_loss.item()),
        "shifted_native_unforced_mse": float(unforced_shifted_loss.item()),
        "mse": mse,
        "mse_change": mse_change,
        "relative_mse_change": relative_mse_change,
        "full_model_translation_mse": _valid_translation_mse(
            original_prediction,
            shifted_prediction,
            dy,
            dx,
        ),
        "numerical_controls": {
            "noop_max_abs_output_change": float(
                (intervention_prediction[1] - intervention_prediction[0])
                .abs()
                .max()
                .item()
            ),
            "forced_unforced_max_abs_output_change": float(
                (intervention_prediction[0] - shifted_prediction[0])
                .abs()
                .max()
                .item()
            ),
            "forced_unforced_mse_change": float(
                (native_loss - unforced_shifted_loss).item()
            ),
            "stratum_random_controls": stratum_controls,
        },
    }


def _summarize_stratified_records(records):
    if not records:
        raise ValueError("At least one stratified record is required")
    summary = {
        "num_cells": len(records),
        "content_route_agreement_mean": float(np.mean([
            record["route_agreement"]["agreement"] for record in records
        ])),
        "content_hidden_cosine_mean": float(np.mean([
            record["hidden_translation"]["content_follow_cosine_mean"]
            for record in records
        ])),
        "max_abs_noop_mse_change": float(max(
            abs(record["mse_change"]["noop_native"])
            for record in records
        )),
        "max_abs_noop_output_change": float(max(
            record["numerical_controls"]["noop_max_abs_output_change"]
            for record in records
        )),
        "max_abs_forced_unforced_output_change": float(max(
            record["numerical_controls"][
                "forced_unforced_max_abs_output_change"
            ]
            for record in records
        )),
        "strata": {},
    }
    for stratum in STRATUM_NAMES:
        nonempty = [
            record for record in records
            if record["strata"][stratum]["tokens"] > 0
        ]
        paired = [
            record for record in nonempty
            if record["numerical_controls"]["stratum_random_controls"]
            [stratum]["random_control_available"]
        ]
        content_key = f"{stratum}_content"
        random_key = f"{stratum}_random"
        contrasts = np.asarray([
            record["relative_mse_change"][content_key]
            - record["relative_mse_change"][random_key]
            for record in paired
        ], dtype=np.float64)
        summary["strata"][stratum] = {
            "nonempty_cells": len(nonempty),
            "valid_random_control_cells": len(paired),
            "mean_tokens": (
                float(np.mean([
                    record["strata"][stratum]["tokens"]
                    for record in nonempty
                ]))
                if nonempty
                else None
            ),
            "content_mean_relative_mse_change": (
                float(np.mean([
                    record["relative_mse_change"][content_key]
                    for record in nonempty
                ]))
                if nonempty
                else None
            ),
            "random_mean_relative_mse_change": (
                float(np.mean([
                    record["relative_mse_change"][random_key]
                    for record in paired
                ]))
                if paired
                else None
            ),
            "content_minus_random_mean": (
                float(contrasts.mean()) if paired else None
            ),
            "content_beats_random_rate": (
                float((contrasts < 0).mean()) if paired else None
            ),
        }
    return summary


def run_routing_translation_stratified_probe(
    checkpoint_path,
    latent_path,
    label,
    sigmas=(0.276, 0.5, 0.724),
    shifts=((0, 2), (0, -2), (2, 0), (-2, 0)),
    block_index=3,
    latent_key="latent",
    seed=0,
    device="cpu",
    num_threads=8,
    weights_checkpoint_path=None,
):
    shifts = _validate_shifts(shifts)
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
        raise ValueError("Stratified routing probe requires top_k == 1")

    patch_size = model.patch_size
    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
            raise ValueError("Stratified probe requires square patches")
        patch_size = patch_size[0]
    patch_size = int(patch_size)
    if patch_size < 1:
        raise ValueError("Model patch size must be positive")
    for shift in shifts:
        if shift[0] % patch_size or shift[1] % patch_size:
            raise ValueError(
                f"Shift {shift} must be divisible by patch size {patch_size}"
            )

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
            for shift in shifts:
                records.append(_probe_stratified_cell(
                    model=model,
                    moe_layer=moe_layer,
                    capture=capture,
                    clean_latent=clean_latent,
                    noise=noise,
                    label=label_tensor,
                    sigma=sigma,
                    num_train_timesteps=runtime_cfg.num_train_timesteps,
                    shift=shift,
                    patch_size=patch_size,
                    num_routed_experts=moe_layer.num_routed_experts,
                    generator=generator,
                ))
    finally:
        capture.close()
    probe_seconds = time.perf_counter() - probe_start

    result = {
        "routing_translation_stratified_probe_version": 1,
        "diagnostic_scope": (
            "teacher-forced fixed-compute stratum interventions; not a FID claim"
        ),
        "stratum_definition": {
            "low_margin": (
                "bottom ceil(n/2) changed tokens per cell by native top1-top2 margin"
            ),
            "high_margin": (
                "remaining top floor(n/2) changed tokens per cell by margin"
            ),
            "content_top2": (
                "changed tokens whose transported expert ranks first or second"
            ),
            "content_rank3plus": (
                "changed tokens whose transported expert ranks third or lower"
            ),
        },
        "intervention": (
            "change only one pre-router stratum at one MoE block; preserve the "
            "shifted input's router weights and top-1 expert compute"
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
        "shifts_latent": [[dy, dx] for dy, dx in shifts],
        "patch_size": patch_size,
        "seed": int(seed),
        "device": str(device),
        "num_threads": int(num_threads),
        "thread_config": thread_config,
        "model_load_seconds": load_seconds,
        "probe_seconds": probe_seconds,
        "intervention_names": list(INTERVENTION_NAMES),
        "summary": _summarize_stratified_records(records),
        "per_sigma": {
            str(float(sigma)): _summarize_stratified_records([
                record for record in records
                if record["sigma"] == float(sigma)
            ])
            for sigma in sigmas
        },
        "records": records,
    }
    del model
    gc.collect()
    return result
