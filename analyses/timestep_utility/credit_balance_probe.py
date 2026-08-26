"""Frozen-checkpoint measurements for routed-expert learning credit."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from analyses.denoising_regret.probe import (
    RoutingProbeCapture,
    _all_router_weights,
    _extract_prediction,
    _load_latent,
    _per_sample_mse,
)
from analyses.timestep_utility.probe import _validate_moe_block_contract


PROBE_VERSION = 1
SELECTION_SALT = "promoe-credit-balance-v1"
PERMUTATION_SALT = "promoe-credit-balance-v1-permutation"
BLOCKS = (1, 5, 11)
SIGMAS = (0.2, 0.5, 0.8)
DUPLICATE_BATCH_SIZE = 2
PERMUTATION_RESAMPLES = 4096


def stable_seed(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2 ** 63)


def gini(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Gini values must be a nonempty vector")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Gini values must be finite and nonnegative")
    total = float(values.sum())
    if total <= 0:
        return 0.0
    ordered = np.sort(values)
    indices = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(
        (2.0 * np.dot(indices, ordered) / (ordered.size * total))
        - (ordered.size + 1.0) / ordered.size
    )


def _coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    return float(values.std() / mean) if mean > 0 else 0.0


def permutation_mean_load_credit_tv(
    token_credit,
    native_experts,
    num_experts,
    resamples,
    seed,
    chunk_size=512,
):
    token_credit = np.asarray(token_credit, dtype=np.float64)
    native_experts = np.asarray(native_experts, dtype=np.int64)
    num_experts = int(num_experts)
    resamples = int(resamples)
    if token_credit.ndim != 1 or native_experts.shape != token_credit.shape:
        raise ValueError("Token credit and native experts must be aligned vectors")
    if token_credit.size == 0 or not np.isfinite(token_credit).all():
        raise ValueError("Token credit must be nonempty and finite")
    if np.any(token_credit < 0) or token_credit.sum() <= 0:
        raise ValueError("Token credit must have positive total mass")
    if (
        native_experts.min() < 0
        or native_experts.max() >= num_experts
        or resamples <= 0
    ):
        raise ValueError("Permutation-null dimensions are invalid")

    counts = np.bincount(native_experts, minlength=num_experts).astype(np.float64)
    load_share = counts / counts.sum()
    expert_masks = np.stack(
        [native_experts == expert for expert in range(num_experts)],
        axis=1,
    ).astype(np.float64)
    generator = np.random.default_rng(int(seed))
    total_tv = 0.0
    completed = 0
    broadcast = np.broadcast_to(
        token_credit,
        (min(chunk_size, resamples), token_credit.size),
    )
    while completed < resamples:
        batch = min(chunk_size, resamples - completed)
        source = broadcast if batch == broadcast.shape[0] else np.broadcast_to(
            token_credit,
            (batch, token_credit.size),
        )
        permuted = generator.permuted(source, axis=1)
        expert_credit = permuted @ expert_masks
        credit_share = expert_credit / expert_credit.sum(axis=1, keepdims=True)
        total_tv += float(
            (0.5 * np.abs(credit_share - load_share[None, :]).sum(axis=1)).sum()
        )
        completed += batch
    return total_tv / resamples


def credit_cell_statistics(
    token_credit,
    unit_weight_credit,
    native_experts,
    num_experts,
    permutation_seed,
    permutation_resamples=PERMUTATION_RESAMPLES,
):
    token_credit = np.asarray(token_credit, dtype=np.float64)
    unit_weight_credit = np.asarray(unit_weight_credit, dtype=np.float64)
    native_experts = np.asarray(native_experts, dtype=np.int64)
    if unit_weight_credit.shape != token_credit.shape:
        raise ValueError("Weighted and unit-weight token credit must align")
    if native_experts.shape != token_credit.shape:
        raise ValueError("Native experts and token credit must align")
    if not np.isfinite(unit_weight_credit).all() or np.any(unit_weight_credit < 0):
        raise ValueError("Unit-weight token credit must be finite and nonnegative")

    counts = np.bincount(native_experts, minlength=num_experts).astype(np.int64)
    credit = np.bincount(
        native_experts,
        weights=token_credit,
        minlength=num_experts,
    ).astype(np.float64)
    unit_credit = np.bincount(
        native_experts,
        weights=unit_weight_credit,
        minlength=num_experts,
    ).astype(np.float64)
    if credit.sum() <= 0 or unit_credit.sum() <= 0:
        raise RuntimeError("Expert credit has zero total mass")
    active = counts > 0
    rates = np.zeros(num_experts, dtype=np.float64)
    unit_rates = np.zeros(num_experts, dtype=np.float64)
    rates[active] = credit[active] / counts[active]
    unit_rates[active] = unit_credit[active] / counts[active]
    load_share = counts / counts.sum()
    credit_share = credit / credit.sum()
    unit_credit_share = unit_credit / unit_credit.sum()
    observed_tv = float(0.5 * np.abs(load_share - credit_share).sum())
    null_tv = permutation_mean_load_credit_tv(
        token_credit=token_credit,
        native_experts=native_experts,
        num_experts=num_experts,
        resamples=permutation_resamples,
        seed=permutation_seed,
    )
    return {
        "token_count": counts.tolist(),
        "expert_credit": credit.tolist(),
        "expert_unit_weight_credit": unit_credit.tolist(),
        "expert_credit_rate": rates.tolist(),
        "expert_unit_weight_credit_rate": unit_rates.tolist(),
        "active_experts": int(active.sum()),
        "token_count_cv": _coefficient_of_variation(counts.astype(np.float64)),
        "token_count_gini": gini(counts.astype(np.float64)),
        "credit_rate_gini": gini(rates[active]),
        "unit_weight_credit_rate_gini": gini(unit_rates[active]),
        "load_credit_tv": observed_tv,
        "unit_weight_load_credit_tv": float(
            0.5 * np.abs(load_share - unit_credit_share).sum()
        ),
        "permutation_mean_load_credit_tv": float(null_tv),
        "permutation_excess_tv": float(observed_tv - null_tv),
        "permutation_seed": int(permutation_seed),
        "permutation_resamples": int(permutation_resamples),
    }


def _probe_cell(
    model,
    runtime_cfg,
    moe_layer,
    capture,
    clean_latent,
    noise,
    label,
    case_id,
    block_index,
    sigma,
):
    sigma_tensor = torch.tensor(
        float(sigma),
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    timestep = torch.full(
        (DUPLICATE_BATCH_SIZE,),
        float(sigma) * int(runtime_cfg.num_train_timesteps),
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    noised_latent = (1.0 - sigma_tensor) * clean_latent + sigma_tensor * noise
    target = (noise - clean_latent).squeeze(2)
    repeated_target = target.repeat(DUPLICATE_BATCH_SIZE, 1, 1, 1)

    capture.start()
    try:
        model_output = model(
            noised_latent.repeat(DUPLICATE_BATCH_SIZE, 1, 1, 1, 1),
            timestep,
            context=label.repeat(DUPLICATE_BATCH_SIZE),
        )
        predictions = _extract_prediction(model_output, target.shape[1])
        losses = _per_sample_mse(predictions, repeated_target)
        if capture.moe_output is None:
            raise RuntimeError("Credit probe did not capture the MoE output")
        moe_gradient, = torch.autograd.grad(losses[0], capture.moe_output)
    finally:
        capture.stop()

    hidden_states = capture.hidden_states
    labels = capture.labels
    if hidden_states is None or labels is None:
        raise RuntimeError("Credit probe did not capture router inputs")
    with torch.no_grad():
        route_weights, route_indices, auxiliary_loss = moe_layer.compute_router(
            hidden_states,
            labels,
        )
        router_scores = _all_router_weights(moe_layer, hidden_states)
    if auxiliary_loss is not None:
        raise RuntimeError("Frozen eval router returned an auxiliary loss")
    if not torch.equal(
        route_indices,
        route_indices[0:1].expand_as(route_indices),
    ) or not torch.equal(
        route_weights,
        route_weights[0:1].expand_as(route_weights),
    ):
        raise RuntimeError("Duplicate native rows produced different routes")
    native_experts = route_indices[0, :, 0]
    native_weights = route_weights[0, :, 0]
    route_mismatches = int(
        (router_scores[0].argmax(dim=-1) != native_experts).sum().item()
    )
    if route_mismatches:
        raise RuntimeError("Native routes disagree with all-router scores")

    gradient_energy = moe_gradient[0].double().square().sum(dim=-1)
    token_credit = gradient_energy * native_weights.double().square()
    permutation_seed = stable_seed(
        PERMUTATION_SALT,
        case_id,
        int(block_index),
        f"{float(sigma):.17g}",
    )
    statistics = credit_cell_statistics(
        token_credit=token_credit.detach().cpu().numpy(),
        unit_weight_credit=gradient_energy.detach().cpu().numpy(),
        native_experts=native_experts.detach().cpu().numpy(),
        num_experts=int(moe_layer.num_routed_experts),
        permutation_seed=permutation_seed,
    )
    native_mse = float(losses[0].item())
    return {
        "block_index": int(block_index),
        "sigma": float(sigma),
        "timestep": float(timestep[0].item()),
        "native_mse": native_mse,
        "statistics": statistics,
        "numerical_controls": {
            "max_abs_native_output_drift": float(
                (predictions[1] - predictions[0]).abs().max().item()
            ),
            "native_relative_mse_drift": float(
                abs(losses[1].item() - losses[0].item()) / native_mse
            ),
            "route_mismatches": route_mismatches,
            "nonfinite_token_credits": int(
                (~torch.isfinite(token_credit)).sum().item()
            ),
        },
    }


def run_credit_balance_case(
    model,
    runtime_cfg,
    latent_path,
    label,
    seed,
    case_id,
    block_indices=BLOCKS,
    sigmas=SIGMAS,
    latent_key="latent",
):
    latent_path = Path(latent_path).resolve()
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent does not exist: {latent_path}")
    if not 0 <= int(label) < int(runtime_cfg.num_classes):
        raise ValueError("ImageNet label lies outside the configured class range")
    block_indices = tuple(int(block) for block in block_indices)
    sigmas = tuple(float(sigma) for sigma in sigmas)
    if block_indices != BLOCKS or sigmas != SIGMAS:
        raise ValueError("Credit probe block/sigma contract changed")
    _validate_moe_block_contract(model, block_indices)

    device = next(model.parameters()).device
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    clean_latent = _load_latent(latent_path, latent_key, int(seed), device)
    torch.manual_seed(int(seed) + 1)
    noise = torch.randn_like(clean_latent)
    label_tensor = torch.tensor([int(label)], device=device, dtype=torch.long)

    cells = []
    started = time.perf_counter()
    for block_index in block_indices:
        moe_layer = model.blocks[block_index].mlp
        capture = RoutingProbeCapture(moe_layer)
        try:
            for sigma in sigmas:
                cells.append(_probe_cell(
                    model=model,
                    runtime_cfg=runtime_cfg,
                    moe_layer=moe_layer,
                    capture=capture,
                    clean_latent=clean_latent,
                    noise=noise,
                    label=label_tensor,
                    case_id=case_id,
                    block_index=block_index,
                    sigma=sigma,
                ))
        finally:
            capture.close()
    return {
        "credit_balance_probe_version": PROBE_VERSION,
        "diagnostic_scope": (
            "frozen-checkpoint routed-expert suffix-gradient credit audit; "
            "not a training, generation, FID, routing-utility, or novelty claim"
        ),
        "case_id": str(case_id),
        "label": int(label),
        "latent": str(latent_path),
        "latent_key": latent_key,
        "seed": int(seed),
        "block_indices": list(block_indices),
        "sigmas": list(sigmas),
        "probe_seconds": float(time.perf_counter() - started),
        "cells": cells,
    }
