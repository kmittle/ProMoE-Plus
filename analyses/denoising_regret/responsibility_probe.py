from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)

from .probe import (
    RoutingProbeCapture,
    _configure_torch_threads,
    _correlation,
    _evaluate_experts,
    _extract_prediction,
    _load_checkpoint_model,
    _load_latent,
    _per_sample_mse,
    _rankdata,
)


def _scale_key(scale):
    return str(float(scale))


def _validate_candidate_scales(candidate_scales):
    scales = [float(scale) for scale in candidate_scales]
    if not scales:
        raise ValueError("At least one candidate scale is required")
    if any(not np.isfinite(scale) for scale in scales):
        raise ValueError("Candidate scales must be finite")
    if len(scales) != len(set(scales)):
        raise ValueError("Candidate scales must be unique")
    return scales


def _summarize_scale_action(predicted, exact):
    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    if predicted.shape != exact.shape or exact.size == 0:
        raise ValueError("Predicted and exact changes must be non-empty and aligned")
    predicted_better = predicted < 0
    exact_better = exact < 0
    true_positive = predicted_better & exact_better
    return {
        "mean_exact_mse_change": float(exact.mean()),
        "median_exact_mse_change": float(np.median(exact)),
        "median_abs_exact_mse_change": float(np.median(np.abs(exact))),
        "exact_better_rate": float(exact_better.mean()),
        "exact_worse_rate": float((exact > 0).mean()),
        "first_order_pearson": _correlation(predicted, exact),
        "first_order_spearman": _correlation(
            _rankdata(predicted),
            _rankdata(exact),
        ),
        "first_order_sign_agreement": float(
            np.mean(np.signbit(predicted) == np.signbit(exact))
        ),
        "predicted_better_precision": (
            float(true_positive.sum() / predicted_better.sum())
            if predicted_better.any()
            else None
        ),
        "predicted_better_recall": (
            float(true_positive.sum() / exact_better.sum())
            if exact_better.any()
            else None
        ),
    }


def summarize_responsibility_records(records, candidate_scales):
    if not records:
        raise ValueError("At least one responsibility record is required")
    candidate_scales = _validate_candidate_scales(candidate_scales)
    scale_keys = [_scale_key(scale) for scale in candidate_scales]
    affinities = np.asarray(
        [record["native_router_weight"] for record in records],
        dtype=np.float64,
    )
    slopes = np.asarray(
        [record["responsibility_slope"] for record in records],
        dtype=np.float64,
    )
    exact_matrix = np.asarray(
        [
            [record["exact_mse_change"][key] for key in scale_keys]
            for record in records
        ],
        dtype=np.float64,
    )
    first_order_matrix = np.asarray(
        [
            [record["first_order_mse_change"][key] for key in scale_keys]
            for record in records
        ],
        dtype=np.float64,
    )

    candidate_indices = exact_matrix.argmin(axis=1)
    best_candidate_scales = np.asarray(
        [candidate_scales[index] for index in candidate_indices],
        dtype=np.float64,
    )
    best_candidate_changes = exact_matrix.min(axis=1)
    native_and_candidates = np.concatenate(
        [np.zeros((len(records), 1), dtype=np.float64), exact_matrix],
        axis=1,
    )
    oracle_changes = native_and_candidates.min(axis=1)
    native_is_best = native_and_candidates.argmin(axis=1) == 0

    per_scale = {}
    for column, key in enumerate(scale_keys):
        action = _summarize_scale_action(
            first_order_matrix[:, column],
            exact_matrix[:, column],
        )
        action["affinity_exact_change_spearman"] = _correlation(
            _rankdata(affinities),
            _rankdata(exact_matrix[:, column]),
        )
        per_scale[key] = action

    return {
        "num_probes": int(len(records)),
        "native_router_weight_mean": float(affinities.mean()),
        "native_router_weight_std": float(affinities.std()),
        "native_router_weight_min": float(affinities.min()),
        "native_router_weight_max": float(affinities.max()),
        "mean_responsibility_slope": float(slopes.mean()),
        "median_abs_responsibility_slope": float(np.median(np.abs(slopes))),
        "increase_weight_recommended_rate": float((slopes < 0).mean()),
        "decrease_weight_recommended_rate": float((slopes > 0).mean()),
        "affinity_slope_spearman": _correlation(
            _rankdata(affinities),
            _rankdata(slopes),
        ),
        "native_best_rate": float(native_is_best.mean()),
        "candidate_oracle_better_rate": float((best_candidate_changes < 0).mean()),
        "candidate_oracle_mean_mse_change": float(best_candidate_changes.mean()),
        "candidate_oracle_median_mse_change": float(
            np.median(best_candidate_changes)
        ),
        "native_inclusive_oracle_mean_mse_change": float(oracle_changes.mean()),
        "native_inclusive_oracle_median_mse_change": float(
            np.median(oracle_changes)
        ),
        "affinity_best_candidate_scale_spearman": _correlation(
            _rankdata(affinities),
            _rankdata(best_candidate_scales),
        ),
        "best_candidate_scale_counts": {
            key: int((candidate_indices == column).sum())
            for column, key in enumerate(scale_keys)
        },
        "per_scale": per_scale,
    }


def summarize_global_records(records, candidate_scales):
    if not records:
        raise ValueError("At least one global responsibility record is required")
    candidate_scales = _validate_candidate_scales(candidate_scales)
    scale_keys = [_scale_key(scale) for scale in candidate_scales]
    exact_matrix = np.asarray(
        [
            [record["exact_mse_change"][key] for key in scale_keys]
            for record in records
        ],
        dtype=np.float64,
    )
    first_order_matrix = np.asarray(
        [
            [record["first_order_mse_change"][key] for key in scale_keys]
            for record in records
        ],
        dtype=np.float64,
    )
    best_indices = exact_matrix.argmin(axis=1)
    best_changes = exact_matrix.min(axis=1)
    native_and_candidates = np.concatenate(
        [np.zeros((len(records), 1), dtype=np.float64), exact_matrix],
        axis=1,
    )
    return {
        "num_cases": int(len(records)),
        "native_best_rate": float(
            (native_and_candidates.argmin(axis=1) == 0).mean()
        ),
        "candidate_oracle_better_rate": float((best_changes < 0).mean()),
        "native_inclusive_oracle_mean_mse_change": float(
            native_and_candidates.min(axis=1).mean()
        ),
        "best_candidate_scale_counts": {
            key: int((best_indices == column).sum())
            for column, key in enumerate(scale_keys)
        },
        "per_scale": {
            key: _summarize_scale_action(
                first_order_matrix[:, column],
                exact_matrix[:, column],
            )
            for column, key in enumerate(scale_keys)
        },
    }


@contextmanager
def _forced_token_route_weights(moe_layer, token_indices, route_weights):
    original_compute_router = moe_layer.compute_router

    def compute_router_with_override(this, hidden_states, labels):
        weights, indices, auxiliary_loss = original_compute_router(
            hidden_states,
            labels,
        )
        if hidden_states.shape[0] != token_indices.numel():
            raise RuntimeError(
                "Forced token-weight count must match the counterfactual batch size"
            )
        if weights.shape[-1] != 1:
            raise RuntimeError("Responsibility overrides require top_k == 1")
        rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        if route_weights is None:
            current_weights = weights[rows, token_indices, 0].clone()
            weights[rows, token_indices, 0] = current_weights
        else:
            weights[rows, token_indices, 0] = route_weights.to(
                device=weights.device,
                dtype=weights.dtype,
            )
        return weights, indices, auxiliary_loss

    if "compute_router" in moe_layer.__dict__:
        raise RuntimeError("MoE layer already has an instance compute_router override")
    moe_layer.compute_router = MethodType(compute_router_with_override, moe_layer)
    try:
        yield
    finally:
        del moe_layer.compute_router


@contextmanager
def _forced_route_weight_matrix(moe_layer, route_weight_matrix):
    original_compute_router = moe_layer.compute_router

    def compute_router_with_override(this, hidden_states, labels):
        weights, indices, auxiliary_loss = original_compute_router(
            hidden_states,
            labels,
        )
        if weights.shape[-1] != 1:
            raise RuntimeError("Responsibility overrides require top_k == 1")
        if route_weight_matrix is not None and weights.shape[:2] != route_weight_matrix.shape:
            raise RuntimeError(
                "Forced route-weight matrix must match batch and sequence dimensions"
            )
        conditional_rows = labels != 1000
        if route_weight_matrix is None:
            current_weights = weights[conditional_rows, :, 0].clone()
            weights[conditional_rows, :, 0] = current_weights
        else:
            weights[conditional_rows, :, 0] = route_weight_matrix[
                conditional_rows
            ].to(device=weights.device, dtype=weights.dtype)
        return weights, indices, auxiliary_loss

    if "compute_router" in moe_layer.__dict__:
        raise RuntimeError("MoE layer already has an instance compute_router override")
    moe_layer.compute_router = MethodType(compute_router_with_override, moe_layer)
    try:
        yield
    finally:
        del moe_layer.compute_router


def _exact_token_weight_changes(
    model,
    moe_layer,
    noised_latent,
    timestep,
    label,
    target,
    token_indices,
    route_weights,
    batch_size,
):
    changes = []
    target_channels = target.shape[1]
    for start in range(0, token_indices.numel(), batch_size):
        stop = min(start + batch_size, token_indices.numel())
        count = stop - start
        batch_latent = noised_latent.repeat(count, 1, 1, 1, 1)
        batch_timestep = timestep.repeat(count)
        batch_label = label.repeat(count)
        batch_target = target.repeat(count, 1, 1, 1)

        with torch.inference_mode():
            base_output = model(batch_latent, batch_timestep, context=batch_label)
            base_prediction = _extract_prediction(base_output, target_channels)
            base_losses = _per_sample_mse(base_prediction, batch_target)
            chunk_route_weights = (
                None
                if route_weights is None
                else route_weights[start:stop]
            )
            with _forced_token_route_weights(
                moe_layer,
                token_indices[start:stop],
                chunk_route_weights,
            ):
                alternative_output = model(
                    batch_latent,
                    batch_timestep,
                    context=batch_label,
                )
            alternative_prediction = _extract_prediction(
                alternative_output,
                target_channels,
            )
            alternative_losses = _per_sample_mse(
                alternative_prediction,
                batch_target,
            )
        changes.append((alternative_losses - base_losses).cpu())
    return torch.cat(changes)


def _exact_global_weight_changes(
    model,
    moe_layer,
    noised_latent,
    timestep,
    label,
    target,
    route_weight_matrix,
    batch_size,
):
    changes = []
    target_channels = target.shape[1]
    num_interventions = 1 if route_weight_matrix is None else route_weight_matrix.shape[0]
    for start in range(0, num_interventions, batch_size):
        stop = min(start + batch_size, num_interventions)
        count = stop - start
        batch_latent = noised_latent.repeat(count, 1, 1, 1, 1)
        batch_timestep = timestep.repeat(count)
        batch_label = label.repeat(count)
        batch_target = target.repeat(count, 1, 1, 1)

        with torch.inference_mode():
            base_output = model(batch_latent, batch_timestep, context=batch_label)
            base_prediction = _extract_prediction(base_output, target_channels)
            base_losses = _per_sample_mse(base_prediction, batch_target)
            chunk_route_weight_matrix = (
                None
                if route_weight_matrix is None
                else route_weight_matrix[start:stop]
            )
            with _forced_route_weight_matrix(
                moe_layer,
                chunk_route_weight_matrix,
            ):
                alternative_output = model(
                    batch_latent,
                    batch_timestep,
                    context=batch_label,
                )
            alternative_prediction = _extract_prediction(
                alternative_output,
                target_channels,
            )
            alternative_losses = _per_sample_mse(
                alternative_prediction,
                batch_target,
            )
        changes.append((alternative_losses - base_losses).cpu())
    return torch.cat(changes)


def _probe_sigma(
    model,
    moe_layer,
    capture,
    clean_latent,
    noise,
    label,
    sigma,
    num_train_timesteps,
    num_token_probes,
    candidate_scales,
    exact_batch_size,
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
    noised_latent = (1.0 - sigma_tensor) * clean_latent + sigma_tensor * noise
    target = (noise - clean_latent).squeeze(2)
    target_channels = target.shape[1]

    capture.start()
    model_output = model(noised_latent, timestep, context=label)
    prediction = _extract_prediction(model_output, target_channels)
    base_loss = _per_sample_mse(prediction, target).mean()
    if capture.moe_output is None:
        raise RuntimeError("The MoE responsibility hook did not capture an output")
    moe_gradient, = torch.autograd.grad(base_loss, capture.moe_output)
    capture.stop()

    hidden_states = capture.hidden_states
    captured_labels = capture.labels
    with torch.no_grad():
        router_weights, expert_indices, _ = moe_layer.compute_router(
            hidden_states,
            captured_labels,
        )
        native_weights = router_weights[0, :, 0].float()
        selected_experts = expert_indices[0, :, 0]
        selected_outputs = _evaluate_experts(
            moe_layer.experts[:moe_layer.num_routed_experts],
            hidden_states[0],
            selected_experts,
        )
        responsibility_slopes = (
            moe_gradient[0].float() * selected_outputs
        ).sum(dim=-1)

    num_tokens = hidden_states.shape[1]
    probe_count = min(num_token_probes, num_tokens)
    token_indices = torch.randperm(
        num_tokens,
        generator=generator,
        device=hidden_states.device,
    )[:probe_count]
    scale_tensor = torch.tensor(
        candidate_scales,
        device=hidden_states.device,
        dtype=native_weights.dtype,
    )
    token_grid = token_indices.repeat_interleave(len(candidate_scales))
    route_weight_grid = scale_tensor.repeat(probe_count)
    exact_grid = _exact_token_weight_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        token_indices=token_grid,
        route_weights=route_weight_grid,
        batch_size=exact_batch_size,
    ).view(probe_count, len(candidate_scales))

    sampled_native = native_weights[token_indices]
    sampled_slopes = responsibility_slopes[token_indices]
    first_order_grid = (
        scale_tensor.unsqueeze(0) - sampled_native.unsqueeze(1)
    ) * sampled_slopes.unsqueeze(1)

    global_weight_matrix = scale_tensor.unsqueeze(1).expand(-1, num_tokens)
    exact_global = _exact_global_weight_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        route_weight_matrix=global_weight_matrix,
        batch_size=exact_batch_size,
    )
    first_order_global = (
        (global_weight_matrix - native_weights.unsqueeze(0))
        * responsibility_slopes.unsqueeze(0)
    ).sum(dim=1)

    noop_count = min(exact_batch_size, probe_count)
    noop_change = _exact_token_weight_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        token_indices=token_indices[:noop_count],
        route_weights=None,
        batch_size=exact_batch_size,
    )
    global_noop_change = _exact_global_weight_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        route_weight_matrix=None,
        batch_size=1,
    )

    scale_keys = [_scale_key(scale) for scale in candidate_scales]
    records = []
    for row in range(probe_count):
        records.append({
            "sigma": float(sigma),
            "timestep": float(timestep.item()),
            "token_index": int(token_indices[row].item()),
            "selected_expert": int(selected_experts[token_indices[row]].item()),
            "native_router_weight": float(sampled_native[row].item()),
            "responsibility_slope": float(sampled_slopes[row].item()),
            "first_order_mse_change": {
                key: float(first_order_grid[row, column].item())
                for column, key in enumerate(scale_keys)
            },
            "exact_mse_change": {
                key: float(exact_grid[row, column].item())
                for column, key in enumerate(scale_keys)
            },
        })

    global_record = {
        "sigma": float(sigma),
        "timestep": float(timestep.item()),
        "first_order_mse_change": {
            key: float(first_order_global[column].item())
            for column, key in enumerate(scale_keys)
        },
        "exact_mse_change": {
            key: float(exact_global[column].item())
            for column, key in enumerate(scale_keys)
        },
    }
    baseline = {
        "mse": float(base_loss.item()),
        "native_router_weight_mean": float(native_weights.mean().item()),
        "native_router_weight_std": float(native_weights.std().item()),
        "native_router_weight_min": float(native_weights.min().item()),
        "native_router_weight_max": float(native_weights.max().item()),
    }
    controls = {
        "noop_num_probes": int(noop_count),
        "noop_token_max_abs_mse_change": float(noop_change.abs().max().item()),
        "noop_global_max_abs_mse_change": float(
            global_noop_change.abs().max().item()
        ),
    }
    return records, global_record, baseline, controls


def run_responsibility_probe(
    checkpoint_path,
    latent_path,
    label,
    sigmas,
    candidate_scales=(0.0, 0.25, 0.5, 0.75, 1.0),
    block_index=3,
    num_token_probes=32,
    exact_batch_size=4,
    latent_key="latent",
    seed=0,
    device="cpu",
    num_threads=8,
    weights_checkpoint_path=None,
):
    candidate_scales = _validate_candidate_scales(candidate_scales)
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
    if not sigmas or any(not 0 < float(sigma) < 1 for sigma in sigmas):
        raise ValueError("Every probe sigma must be strictly between 0 and 1")
    if num_token_probes < 2:
        raise ValueError("num_token_probes must be at least 2")
    if exact_batch_size < 1:
        raise ValueError("exact_batch_size must be positive")
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
        raise ValueError("The responsibility probe requires top_k == 1")
    if moe_layer.router_weight_mode != "identity":
        raise ValueError(
            "The responsibility probe requires router_weight_mode='identity'"
        )
    if not moe_layer.use_shared_expert:
        raise ValueError("The responsibility probe requires a shared expert")

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    clean_latent = _load_latent(latent_path, latent_key, seed, device)
    torch.manual_seed(seed + 1)
    noise = torch.randn_like(clean_latent)
    label_tensor = torch.tensor([label], device=device, dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 2)

    capture = RoutingProbeCapture(moe_layer)
    records = []
    global_records = []
    baseline = {}
    numerical_controls = {}
    probe_start = time.perf_counter()
    try:
        for sigma in sigmas:
            (
                sigma_records,
                sigma_global_record,
                sigma_baseline,
                sigma_controls,
            ) = _probe_sigma(
                model=model,
                moe_layer=moe_layer,
                capture=capture,
                clean_latent=clean_latent,
                noise=noise,
                label=label_tensor,
                sigma=float(sigma),
                num_train_timesteps=runtime_cfg.num_train_timesteps,
                num_token_probes=num_token_probes,
                candidate_scales=candidate_scales,
                exact_batch_size=exact_batch_size,
                generator=generator,
            )
            records.extend(sigma_records)
            global_records.append(sigma_global_record)
            baseline[_scale_key(sigma)] = sigma_baseline
            numerical_controls[_scale_key(sigma)] = sigma_controls
    finally:
        capture.close()
    probe_seconds = time.perf_counter() - probe_start

    per_sigma = {}
    global_per_sigma = {}
    for sigma in sigmas:
        key = _scale_key(sigma)
        sigma_records = [
            record for record in records if record["sigma"] == float(sigma)
        ]
        sigma_global_records = [
            record
            for record in global_records
            if record["sigma"] == float(sigma)
        ]
        per_sigma[key] = summarize_responsibility_records(
            sigma_records,
            candidate_scales,
        )
        global_per_sigma[key] = summarize_global_records(
            sigma_global_records,
            candidate_scales,
        )

    result = {
        "responsibility_probe_version": 1,
        "diagnostic_scope": "teacher-forced oracle; not a direct FID claim",
        "intervention": (
            "fixed dispatch and expert computation; replace only routed top-1 "
            "output scale"
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
        "sigmas": [float(sigma) for sigma in sigmas],
        "candidate_scales": candidate_scales,
        "num_token_probes_requested": int(num_token_probes),
        "exact_batch_size": int(exact_batch_size),
        "seed": int(seed),
        "device": str(device),
        "num_threads": int(num_threads),
        "thread_config": thread_config,
        "model_load_seconds": load_seconds,
        "probe_seconds": probe_seconds,
        "baseline": baseline,
        "numerical_controls": numerical_controls,
        "summary": summarize_responsibility_records(
            records,
            candidate_scales,
        ),
        "global_summary": summarize_global_records(
            global_records,
            candidate_scales,
        ),
        "per_sigma": per_sigma,
        "global_per_sigma": global_per_sigma,
        "global_records": global_records,
        "records": records,
    }
    del model
    gc.collect()
    return result
