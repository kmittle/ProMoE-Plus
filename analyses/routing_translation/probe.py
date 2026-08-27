"""Causal translation audit for spatial shortcuts in top-1 MoE routing."""

from __future__ import annotations

import gc
import math
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

from analyses.denoising_regret.probe import (
    _all_router_weights,
    _compute_router,
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


INTERVENTION_NAMES = (
    "native",
    "noop_native",
    "content_follow",
    "position_follow",
    "random_matched",
)


def _relative_mse_changes(losses, intervention_names):
    if losses.ndim != 1 or losses.numel() != len(intervention_names):
        raise ValueError("Intervention losses and names must be aligned vectors")
    if not bool(torch.isfinite(losses).all().item()):
        raise RuntimeError("Intervention MSE values must be finite")
    native_loss = losses[0]
    if native_loss.item() <= 0:
        raise RuntimeError("Native MSE must be positive for relative change")
    return {
        name: float(((losses[index] - native_loss) / native_loss).item())
        for index, name in enumerate(intervention_names)
    }


class RouteInputCapture:
    """Capture the exact hidden states and labels seen by one MoE router."""

    def __init__(self, moe_layer):
        self.enabled = False
        self.hidden_states = None
        self.labels = None
        self._handle = moe_layer.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        if not self.enabled:
            return None
        if len(inputs) < 2:
            raise RuntimeError("Expected SparseMoeBlock inputs (hidden_states, labels)")
        if self.hidden_states is not None:
            raise RuntimeError("The target MoE layer ran more than once in one forward")
        self.hidden_states = inputs[0].detach()
        self.labels = inputs[1].detach()
        return None

    def start(self):
        self.enabled = True
        self.hidden_states = None
        self.labels = None

    def stop(self):
        self.enabled = False

    def close(self):
        self._handle.remove()


def _validate_shifts(shifts):
    normalized = []
    seen = set()
    for shift in shifts:
        if not isinstance(shift, (tuple, list)) or len(shift) != 2:
            raise ValueError("Every shift must be a (dy, dx) pair")
        dy, dx = shift
        if (
            isinstance(dy, bool)
            or isinstance(dx, bool)
            or not isinstance(dy, int)
            or not isinstance(dx, int)
        ):
            raise ValueError("Shift components must be integers")
        if dy == 0 and dx == 0:
            raise ValueError("Translation shifts must be nonzero")
        pair = (dy, dx)
        if pair in seen:
            raise ValueError(f"Duplicate translation shift: {pair}")
        seen.add(pair)
        normalized.append(pair)
    if not normalized:
        raise ValueError("At least one translation shift is required")
    return normalized


def _translate_spatial(tensor, dy, dx):
    """Reflect-pad and translate the final two dimensions without wraparound."""

    if tensor.ndim < 2:
        raise ValueError("A spatial tensor must have at least two dimensions")
    height, width = tensor.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError(
            f"Shift {(dy, dx)} must be smaller than spatial size {(height, width)}"
        )
    padding = max(abs(dy), abs(dx))
    if padding == 0:
        return tensor.clone()
    if padding >= min(height, width):
        raise ValueError("Reflection padding must be smaller than both dimensions")

    original_shape = tensor.shape
    flat = tensor.reshape(-1, 1, height, width)
    padded = F.pad(flat, (padding, padding, padding, padding), mode="reflect")
    translated = padded[
        ...,
        padding - dy:padding - dy + height,
        padding - dx:padding - dx + width,
    ]
    return translated.reshape(original_shape)


def _translation_valid_mask(height, width, dy, dx, device=None):
    rows = torch.arange(height, device=device).unsqueeze(1)
    columns = torch.arange(width, device=device).unsqueeze(0)
    source_rows = rows - dy
    source_columns = columns - dx
    return (
        (source_rows >= 0)
        & (source_rows < height)
        & (source_columns >= 0)
        & (source_columns < width)
    )


def _build_route_references(original_ids, shifted_ids, grid_size, token_shift):
    if original_ids.ndim != 1 or shifted_ids.ndim != 1:
        raise ValueError("Route IDs must be flat one-dimensional tensors")
    if original_ids.shape != shifted_ids.shape:
        raise ValueError("Original and shifted route maps must have the same shape")
    if original_ids.numel() != grid_size * grid_size:
        raise ValueError("Route count does not match the square token grid")

    dy, dx = token_shift
    if abs(dy) >= grid_size or abs(dx) >= grid_size:
        raise ValueError("Token shift must be smaller than the route grid")
    valid = _translation_valid_mask(
        grid_size,
        grid_size,
        dy,
        dx,
        device=original_ids.device,
    )
    rows = torch.arange(grid_size, device=original_ids.device).unsqueeze(1)
    columns = torch.arange(grid_size, device=original_ids.device).unsqueeze(0)
    source_rows = (rows - dy).clamp(0, grid_size - 1).expand(-1, grid_size)
    source_columns = (columns - dx).clamp(0, grid_size - 1).expand(grid_size, -1)

    original_grid = original_ids.reshape(grid_size, grid_size)
    shifted_grid = shifted_ids.reshape(grid_size, grid_size)
    content_follow = shifted_grid.clone()
    content_follow[valid] = original_grid[source_rows[valid], source_columns[valid]]
    position_follow = shifted_grid.clone()
    position_follow[valid] = original_grid[valid]
    return content_follow.flatten(), position_follow.flatten(), valid.flatten()


def _random_matched_routes(native_ids, content_ids, valid_mask, generator):
    """Randomize content-follow replacements without changing their support or counts."""

    if native_ids.shape != content_ids.shape or native_ids.shape != valid_mask.shape:
        raise ValueError("Native IDs, content IDs, and valid mask must align")
    changed = valid_mask & (native_ids != content_ids)
    result = native_ids.clone()
    changed_count = int(changed.sum().item())
    if changed_count < 2:
        result[changed] = content_ids[changed]
        return result, changed, False

    native_changed = native_ids[changed]
    randomized = content_ids[changed].clone()
    attempts = max(64, changed_count * 32)
    accepted = 0
    for _ in range(attempts):
        pair = torch.randint(
            changed_count,
            (2,),
            generator=generator,
            device=native_ids.device,
        )
        left = int(pair[0].item())
        right = int(pair[1].item())
        if left == right or randomized[left] == randomized[right]:
            continue
        if (
            randomized[right] == native_changed[left]
            or randomized[left] == native_changed[right]
        ):
            continue
        temporary = randomized[left].clone()
        randomized[left] = randomized[right]
        randomized[right] = temporary
        accepted += 1
    random_control_available = bool(
        accepted > 0
        and (randomized != content_ids[changed]).any().item()
    )
    if not random_control_available:
        randomized = content_ids[changed].clone()

    if not torch.all(randomized != native_changed):
        raise RuntimeError("Random matched routes changed their disagreement support")
    result[changed] = randomized
    return result, changed, random_control_available


def _route_agreement(left, right, valid_mask, num_experts):
    if left.shape != right.shape or left.shape != valid_mask.shape:
        raise ValueError("Route maps and valid mask must align")
    left = left[valid_mask].long()
    right = right[valid_mask].long()
    count = int(left.numel())
    if count == 0:
        raise ValueError("Route agreement needs at least one valid token")
    left_hist = torch.bincount(left, minlength=num_experts).double()
    right_hist = torch.bincount(right, minlength=num_experts).double()
    observed = float((left == right).double().mean().item())
    chance = float(((left_hist / count) * (right_hist / count)).sum().item())
    kappa = None if chance >= 1.0 else float((observed - chance) / (1.0 - chance))
    return {
        "valid_tokens": count,
        "agreement": observed,
        "chance_agreement": chance,
        "chance_corrected_agreement": kappa,
        "left_histogram": [int(value) for value in left_hist.tolist()],
        "right_histogram": [int(value) for value in right_hist.tolist()],
    }


def _route_margin_metrics(router_scores, native_ids, content_ids, valid_mask):
    if router_scores.ndim != 2:
        raise ValueError("Router scores must be shaped [tokens, experts]")
    if not (
        native_ids.shape == content_ids.shape == valid_mask.shape
        and native_ids.ndim == 1
        and router_scores.shape[0] == native_ids.numel()
    ):
        raise ValueError("Router scores, IDs, and valid mask must align")
    if router_scores.shape[1] < 2:
        raise ValueError("Route-margin metrics require at least two experts")
    if native_ids.numel() and (
        native_ids.min() < 0
        or content_ids.min() < 0
        or native_ids.max() >= router_scores.shape[1]
        or content_ids.max() >= router_scores.shape[1]
    ):
        raise ValueError("Route IDs are outside the router score matrix")
    selected_ids = router_scores.argmax(dim=-1)
    if not torch.equal(selected_ids, native_ids):
        raise ValueError("Native IDs must be the router-score argmax")

    changed = valid_mask & (native_ids != content_ids)
    unchanged = valid_mask & ~changed
    token_rows = torch.arange(native_ids.numel(), device=router_scores.device)
    native_scores = router_scores[token_rows, native_ids]
    content_scores = router_scores[token_rows, content_ids]
    top_two = torch.topk(router_scores, k=2, dim=-1).values
    top1_margin = top_two[:, 0] - top_two[:, 1]
    content_rank = (
        (router_scores > content_scores.unsqueeze(-1)).sum(dim=-1) + 1
    ).float()

    def summarize(values, mask):
        values = values[mask].float()
        if values.numel() == 0:
            return {"mean": None, "median": None}
        return {
            "mean": float(values.mean().item()),
            "median": float(values.median().item()),
        }

    return {
        "valid_tokens": int(valid_mask.sum().item()),
        "changed_tokens": int(changed.sum().item()),
        "changed_rate": float(
            changed.sum().double().div(valid_mask.sum().clamp_min(1)).item()
        ),
        "native_minus_content_changed": summarize(
            native_scores - content_scores,
            changed,
        ),
        "content_expert_rank_changed": summarize(content_rank, changed),
        "native_top1_margin_changed": summarize(top1_margin, changed),
        "native_top1_margin_unchanged": summarize(top1_margin, unchanged),
    }


def _hidden_translation_metrics(original_hidden, shifted_hidden, grid_size, token_shift):
    if original_hidden.shape != shifted_hidden.shape:
        raise ValueError("Original and shifted hidden states must align")
    if original_hidden.ndim != 2:
        raise ValueError("Hidden states must be shaped [tokens, channels]")
    if original_hidden.shape[0] != grid_size * grid_size:
        raise ValueError("Hidden-state count does not match the token grid")

    dummy_ids = torch.arange(
        original_hidden.shape[0], device=original_hidden.device
    )
    transported_indices, _, valid = _build_route_references(
        dummy_ids,
        dummy_ids,
        grid_size,
        token_shift,
    )
    content_reference = original_hidden[transported_indices[valid]]
    position_reference = original_hidden[valid]
    shifted_valid = shifted_hidden[valid]

    content_cosine = F.cosine_similarity(
        shifted_valid.float(), content_reference.float(), dim=-1
    )
    position_cosine = F.cosine_similarity(
        shifted_valid.float(), position_reference.float(), dim=-1
    )
    content_relative_l2 = (
        (shifted_valid.float() - content_reference.float()).norm(dim=-1)
        / content_reference.float().norm(dim=-1).clamp_min(1e-12)
    )
    position_relative_l2 = (
        (shifted_valid.float() - position_reference.float()).norm(dim=-1)
        / position_reference.float().norm(dim=-1).clamp_min(1e-12)
    )
    return {
        "content_follow_cosine_mean": float(content_cosine.mean().item()),
        "position_follow_cosine_mean": float(position_cosine.mean().item()),
        "content_follow_relative_l2_mean": float(content_relative_l2.mean().item()),
        "position_follow_relative_l2_mean": float(position_relative_l2.mean().item()),
    }


def _valid_translation_mse(original, shifted, dy, dx):
    translated = _translate_spatial(original, dy, dx)
    if translated.shape != shifted.shape:
        raise ValueError("Translated and shifted tensors must align")
    height, width = shifted.shape[-2:]
    valid = _translation_valid_mask(height, width, dy, dx, shifted.device)
    difference = shifted.double() - translated.double()
    return float(difference[..., valid].square().mean().item())


@contextmanager
def _forced_route_matrices(moe_layer, route_ids):
    """Force top-1 IDs while retaining every native router weight."""

    original_compute_router = moe_layer.compute_router

    def compute_router_with_override(
        this,
        hidden_states,
        labels,
        timestep=None,
    ):
        if timestep is None:
            router_result = original_compute_router(hidden_states, labels)
        else:
            router_result = original_compute_router(
                hidden_states,
                labels,
                timestep,
            )
        weights, indices, auxiliary_loss = router_result
        if weights.shape[-1] != 1:
            raise RuntimeError("Translation route overrides require top_k == 1")
        if route_ids.shape != indices.shape[:2]:
            raise RuntimeError(
                "Forced route matrix must match batch and sequence dimensions"
            )
        if route_ids.device != indices.device:
            forced_ids = route_ids.to(indices.device)
        else:
            forced_ids = route_ids
        if forced_ids.numel() and (
            forced_ids.min() < 0
            or forced_ids.max() >= this.num_routed_experts
        ):
            raise RuntimeError("Forced route IDs must name routed experts")
        conditional_rows = labels != 1000
        indices[conditional_rows, :, 0] = forced_ids[conditional_rows]
        return weights, indices, auxiliary_loss

    if "compute_router" in moe_layer.__dict__:
        raise RuntimeError("MoE layer already has an instance compute_router override")
    moe_layer.compute_router = MethodType(compute_router_with_override, moe_layer)
    try:
        yield
    finally:
        del moe_layer.compute_router


def _capture_native_forward(model, moe_layer, capture, inputs, timestep, label):
    capture.start()
    try:
        with torch.inference_mode():
            output = model(inputs, timestep, context=label)
    finally:
        capture.stop()
    if capture.hidden_states is None or capture.labels is None:
        raise RuntimeError("The target MoE router did not run")
    with torch.inference_mode():
        weights, indices, _ = _compute_router(
            moe_layer,
            capture.hidden_states,
            capture.labels,
            timestep,
        )
    return output, capture.hidden_states, weights, indices


def _summarize_records(records):
    if not records:
        raise ValueError("At least one translation record is required")
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
            record["hidden_translation"]["content_follow_cosine_mean"]
            for record in records
        ])),
        "position_follow_hidden_cosine_mean": float(np.mean([
            record["hidden_translation"]["position_follow_cosine_mean"]
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
        summary[f"{name}_median_relative_mse_change"] = float(np.median(changes))
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


def _probe_cell(
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
        float(sigma), device=clean_latent.device, dtype=clean_latent.dtype
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
    shifted_noised = (1.0 - sigma_tensor) * shifted_clean + sigma_tensor * shifted_noise
    shifted_target = (shifted_noise - shifted_clean).squeeze(2)

    (
        original_output,
        original_hidden,
        original_weights,
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
        shifted_output,
        shifted_hidden,
        shifted_weights,
        shifted_indices,
    ) = _capture_native_forward(
        model,
        moe_layer,
        capture,
        shifted_noised,
        timestep,
        label,
    )
    original_prediction = _extract_prediction(
        original_output, original_target.shape[1]
    )
    shifted_prediction = _extract_prediction(
        shifted_output, shifted_target.shape[1]
    )

    original_ids = original_indices[0, :, 0]
    shifted_ids = shifted_indices[0, :, 0]
    num_tokens = int(original_ids.numel())
    grid_size = math.isqrt(num_tokens)
    if grid_size * grid_size != num_tokens:
        raise RuntimeError("Translation probe requires a square token grid")
    token_shift = (dy // patch_size, dx // patch_size)
    content_ids, position_ids, valid_tokens = _build_route_references(
        original_ids,
        shifted_ids,
        grid_size,
        token_shift,
    )
    random_ids, content_changed, random_control_available = _random_matched_routes(
        shifted_ids,
        content_ids,
        valid_tokens,
        generator,
    )
    all_router_scores = _all_router_weights(
        moe_layer,
        shifted_hidden,
        timestep,
    )[0]
    route_margin = _route_margin_metrics(
        all_router_scores,
        shifted_ids,
        content_ids,
        valid_tokens,
    )

    route_matrix = torch.stack([
        shifted_ids,
        shifted_ids,
        content_ids,
        position_ids,
        random_ids,
    ])
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
    mse_change = {
        name: float((losses[index] - native_loss).item())
        for index, name in enumerate(INTERVENTION_NAMES)
    }
    relative_mse_change = _relative_mse_changes(losses, INTERVENTION_NAMES)

    random_changed = valid_tokens & (random_ids != shifted_ids)
    content_hist = torch.bincount(
        content_ids[content_changed], minlength=num_routed_experts
    )
    random_hist = torch.bincount(
        random_ids[random_changed], minlength=num_routed_experts
    )
    if not torch.equal(content_changed, random_changed):
        raise RuntimeError("Random control must match content-follow disagreement support")
    if not torch.equal(content_hist, random_hist):
        raise RuntimeError("Random control must match replacement-expert counts")

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
        "content_changed_tokens": int(content_changed.sum().item()),
        "random_control_available": random_control_available,
        "position_changed_tokens": int(
            (valid_tokens & (position_ids != shifted_ids)).sum().item()
        ),
        "random_differs_from_content_rate": (
            float((random_ids[content_changed] != content_ids[content_changed]).float().mean().item())
            if random_control_available
            else None
        ),
        "route_agreement": {
            "content_follow": _route_agreement(
                shifted_ids, content_ids, valid_tokens, num_routed_experts
            ),
            "position_follow": _route_agreement(
                shifted_ids, position_ids, valid_tokens, num_routed_experts
            ),
        },
        "route_margin": route_margin,
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
            "content_random_changed_support_equal": True,
            "content_random_replacement_histogram_equal": True,
            "random_control_available": random_control_available,
        },
    }


def run_routing_translation_probe(
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
        raise ValueError("Translation routing probe requires top_k == 1")

    patch_size = model.patch_size
    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
            raise ValueError("Translation probe requires square patches")
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
                records.append(_probe_cell(
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

    per_sigma = {
        str(float(sigma)): _summarize_records([
            record for record in records if record["sigma"] == float(sigma)
        ])
        for sigma in sigmas
    }
    per_shift = {
        f"{dy}:{dx}": _summarize_records([
            record
            for record in records
            if record["shift_latent"] == [dy, dx]
        ])
        for dy, dx in shifts
    }
    result = {
        "routing_translation_probe_version": 2,
        "diagnostic_scope": (
            "teacher-forced fixed-compute route intervention; not a direct FID claim"
        ),
        "intervention": (
            "force only top-1 expert identity at one MoE block while preserving "
            "the shifted input's native route weight and expert width"
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
        "summary": _summarize_records(records),
        "per_sigma": per_sigma,
        "per_shift": per_shift,
        "records": records,
    }
    del model
    gc.collect()
    return result
