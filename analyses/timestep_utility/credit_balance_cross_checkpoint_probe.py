"""Loss-Free-compatible output and exact parameter-credit case probe."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from analyses.denoising_regret.probe import (
    RoutingProbeCapture,
    _all_router_weights,
    _compute_router,
    _extract_prediction,
    _load_latent,
    _per_sample_mse,
)
from analyses.timestep_utility.credit_balance_cross_checkpoint import (
    CROSS_CHECKPOINT_VERSION,
    exact_expert_parameter_credit,
    validate_moe_mlp,
)
from analyses.timestep_utility.credit_balance_probe import (
    BLOCKS,
    DUPLICATE_BATCH_SIZE,
    PERMUTATION_RESAMPLES,
    PERMUTATION_SALT,
    PROBE_VERSION,
    SIGMAS,
    credit_cell_statistics,
    stable_seed,
)
from analyses.timestep_utility.probe import _validate_moe_block_contract


MAX_NATIVE_WEIGHT_DRIFT = 5e-7
MEASUREMENT_SCOPES = ("count", "output", "parameter")


def _validate_lossfree_bias_contract(moe_layer, bias_enabled):
    bias = getattr(moe_layer, "expert_bias", None)
    if not bias_enabled:
        return bias
    if not isinstance(bias, torch.Tensor):
        raise TypeError("Enabled Loss-Free routing requires expert_bias")
    registered_bias = dict(moe_layer.named_buffers(recurse=False)).get(
        "expert_bias"
    )
    if registered_bias is not bias:
        raise TypeError("Loss-Free expert_bias must be a registered buffer")
    if "expert_bias" in dict(moe_layer.named_parameters(recurse=False)):
        raise TypeError("Loss-Free expert_bias must not be a Parameter")
    if "expert_bias" in getattr(moe_layer, "_non_persistent_buffers_set", set()):
        raise TypeError("Loss-Free expert_bias must be checkpoint-persistent")
    if bias.requires_grad:
        raise TypeError("Loss-Free expert_bias must not require gradients")
    return bias


def _relative_mse_drift(native_mse, repeated_mse):
    native_mse = float(native_mse)
    repeated_mse = float(repeated_mse)
    if not np.isfinite(native_mse) or native_mse <= 0.0:
        raise RuntimeError("Native MSE must be finite and positive")
    if not np.isfinite(repeated_mse):
        raise RuntimeError("Repeated native MSE must be finite")
    value = abs(repeated_mse - native_mse) / native_mse
    if not np.isfinite(value):
        raise RuntimeError("Native relative MSE drift must be finite")
    return float(value)


def validate_cross_checkpoint_model(
    model,
    block_indices=BLOCKS,
    require_lossfree_bias=None,
):
    """Validate the routed blocks and exact MoeMLP parameter-credit contract."""
    contracts = _validate_moe_block_contract(model, block_indices)
    block_rows = []
    for block_index in block_indices:
        moe_layer = model.blocks[int(block_index)].mlp
        bias_enabled = bool(getattr(moe_layer, "use_lossfree_bias", False))
        if (
            require_lossfree_bias is not None
            and bias_enabled is not bool(require_lossfree_bias)
        ):
            raise ValueError(
                f"block {block_index} Loss-Free bias state is {bias_enabled}; "
                f"expected {bool(require_lossfree_bias)}"
            )
        bias = _validate_lossfree_bias_contract(moe_layer, bias_enabled)
        if bias_enabled:
            if bias.shape != (int(moe_layer.num_routed_experts),):
                raise ValueError("Loss-Free expert_bias width changed")
            if not bool(torch.isfinite(bias).all().item()):
                raise ValueError("Loss-Free expert_bias must be finite")
        expert_contracts = [
            validate_moe_mlp(moe_layer.experts[expert_index])
            for expert_index in range(int(moe_layer.num_routed_experts))
        ]
        if any(row != expert_contracts[0] for row in expert_contracts[1:]):
            raise ValueError("Routed experts no longer share the MoeMLP contract")
        block_rows.append({
            "index": int(block_index),
            "num_routed_experts": int(moe_layer.num_routed_experts),
            "lossfree_bias_enabled": bias_enabled,
            "expert_contract": expert_contracts[0],
        })
    return {**contracts, "parameter_credit_blocks": block_rows}


def _route_controls(moe_layer, hidden_states, labels, timestep=None):
    """Recompute the native route and validate biased selection/unbiased weight."""
    if moe_layer.training:
        raise RuntimeError("Cross-checkpoint routing controls require eval mode")
    with torch.no_grad():
        route_weights, route_indices, auxiliary_loss = _compute_router(
            moe_layer,
            hidden_states,
            labels,
            timestep,
        )
        repeated_weights, repeated_indices, repeated_auxiliary = _compute_router(
            moe_layer,
            hidden_states,
            labels,
            timestep,
        )
        unbiased_scores = _all_router_weights(
            moe_layer,
            hidden_states,
            timestep,
        )
    if auxiliary_loss is not None or repeated_auxiliary is not None:
        raise RuntimeError("Frozen eval router returned an auxiliary loss")
    if route_weights.shape[-1] != 1 or route_indices.shape != route_weights.shape:
        raise RuntimeError("Cross-checkpoint credit requires native top-1 routes")
    for name, values in (
        ("native route weights", route_weights),
        ("repeated route weights", repeated_weights),
        ("unbiased router scores", unbiased_scores),
    ):
        if not bool(torch.isfinite(values).all().item()):
            raise RuntimeError(f"{name} must be finite")
    repeated_route_mismatches = int(
        (route_indices != repeated_indices).sum().item()
    )
    repeated_weight_drift = float(
        (route_weights - repeated_weights).abs().max().item()
    )
    if repeated_route_mismatches or repeated_weight_drift != 0.0:
        raise RuntimeError("Repeated frozen router calls produced different routes")
    if not torch.equal(
        route_indices,
        route_indices[0:1].expand_as(route_indices),
    ) or not torch.equal(
        route_weights,
        route_weights[0:1].expand_as(route_weights),
    ):
        raise RuntimeError("Duplicate native rows produced different routes")

    bias_enabled = bool(getattr(moe_layer, "use_lossfree_bias", False))
    selection_scores = unbiased_scores
    if bias_enabled:
        bias = moe_layer.expert_bias.detach().to(
            device=unbiased_scores.device,
            dtype=unbiased_scores.dtype,
        )
        if bias.shape != (int(moe_layer.num_routed_experts),):
            raise RuntimeError("Loss-Free expert_bias width changed")
        selection_scores = unbiased_scores + bias.view(1, 1, -1)
    if not bool(torch.isfinite(selection_scores).all().item()):
        raise RuntimeError("Loss-Free selection scores must be finite")

    expected_indices = selection_scores.argmax(dim=-1)
    native_indices = route_indices[..., 0]
    route_mismatches = int((expected_indices != native_indices).sum().item())
    unbiased_argmax_mismatches = int(
        (unbiased_scores.argmax(dim=-1) != native_indices).sum().item()
    )
    expected_weights = torch.gather(
        unbiased_scores,
        dim=-1,
        index=route_indices,
    )
    native_weight_drift = float(
        (expected_weights - route_weights).abs().max().item()
    )
    if route_mismatches:
        raise RuntimeError("Native routes disagree with biased selection scores")
    if native_weight_drift > MAX_NATIVE_WEIGHT_DRIFT:
        raise RuntimeError("Native route weights differ from unbiased affinities")
    return route_weights, route_indices, {
        "route_mismatches": route_mismatches,
        "unbiased_argmax_mismatches": unbiased_argmax_mismatches,
        "max_abs_native_weight_drift": native_weight_drift,
        "repeated_route_mismatches": repeated_route_mismatches,
        "max_abs_repeated_weight_drift": repeated_weight_drift,
        "lossfree_bias_enabled": bias_enabled,
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
    measurement_scope,
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
        moe_gradient = None
        if measurement_scope != "count":
            moe_gradient, = torch.autograd.grad(losses[0], capture.moe_output)
    finally:
        capture.stop()

    hidden_states = capture.hidden_states
    labels = capture.labels
    if hidden_states is None or labels is None:
        raise RuntimeError("Credit probe did not capture router inputs")
    route_weights, route_indices, route_controls = _route_controls(
        moe_layer,
        hidden_states,
        labels,
        timestep,
    )
    native_experts = route_indices[0, :, 0]
    native_weights = route_weights[0, :, 0]
    num_experts = int(moe_layer.num_routed_experts)
    counts = torch.bincount(native_experts, minlength=num_experts)
    statistics = {
        "token_count": counts.detach().cpu().tolist(),
        "active_experts": int((counts > 0).sum().item()),
    }
    token_credit = None
    suffix_gradient = None
    if measurement_scope != "count":
        suffix_gradient = moe_gradient[0]
        gradient_energy = suffix_gradient.double().square().sum(dim=-1)
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
            num_experts=num_experts,
            permutation_seed=permutation_seed,
        )
    parameter_statistics = None
    nonfinite_parameter_credits = 0
    if measurement_scope == "parameter":
        parameter_statistics = exact_expert_parameter_credit(
            moe_layer=moe_layer,
            hidden_states=hidden_states[0],
            suffix_gradient=suffix_gradient,
            route_weights=native_weights,
            route_indices=native_experts,
        )
        parameter_arrays = (
            parameter_statistics["expert_parameter_credit"],
            parameter_statistics["expert_parameter_credit_without_bias"],
            parameter_statistics["expert_parameter_credit_rate"],
            parameter_statistics["expert_parameter_credit_rate_without_bias"],
        )
        nonfinite_parameter_credits = int(sum(
            (~np.isfinite(np.asarray(values, dtype=np.float64))).sum()
            for values in parameter_arrays
        ))

    native_mse = float(losses[0].item())
    native_relative_mse_drift = _relative_mse_drift(
        native_mse,
        losses[1].item(),
    )
    native_output_drift = float(
        (predictions[1] - predictions[0]).abs().max().item()
    )
    if not np.isfinite(native_output_drift):
        raise RuntimeError("Native output drift must be finite")
    numerical_controls = {
        "max_abs_native_output_drift": native_output_drift,
        "native_relative_mse_drift": native_relative_mse_drift,
        **route_controls,
    }
    if token_credit is not None:
        numerical_controls["nonfinite_token_credits"] = int(
            (~torch.isfinite(token_credit)).sum().item()
        )
        numerical_controls["nonfinite_parameter_credits"] = (
            nonfinite_parameter_credits
        )
    result = {
        "block_index": int(block_index),
        "sigma": float(sigma),
        "timestep": float(timestep[0].item()),
        "statistics": statistics,
        "numerical_controls": numerical_controls,
    }
    if measurement_scope != "count":
        result["native_mse"] = native_mse
    if parameter_statistics is not None:
        result["parameter_statistics"] = parameter_statistics
    return result


def run_cross_checkpoint_credit_balance_case(
    model,
    runtime_cfg,
    latent_path,
    label,
    seed,
    case_id,
    measurement_scope="output",
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
        raise ValueError("Cross-checkpoint block/sigma contract changed")
    if measurement_scope not in MEASUREMENT_SCOPES:
        raise ValueError(f"Unknown measurement scope: {measurement_scope}")
    validate_cross_checkpoint_model(model, block_indices)

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
                    measurement_scope=measurement_scope,
                ))
        finally:
            capture.close()
    return {
        "cross_checkpoint_probe_version": CROSS_CHECKPOINT_VERSION,
        "credit_balance_probe_version": PROBE_VERSION,
        "diagnostic_scope": (
            "paired frozen-checkpoint routed-expert suffix-gradient and exact "
            "parameter-credit audit; not a training, FID, or novelty claim"
        ),
        "case_id": str(case_id),
        "label": int(label),
        "latent": str(latent_path),
        "latent_key": latent_key,
        "seed": int(seed),
        "block_indices": list(block_indices),
        "sigmas": list(sigmas),
        "measurement_scope": measurement_scope,
        "includes_parameter_credit": measurement_scope == "parameter",
        "probe_seconds": float(time.perf_counter() - started),
        "cells": cells,
    }
