"""Audit stage-conditioned expert utility on untransformed diffusion inputs."""

from __future__ import annotations

import gc
import math
import time
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from analyses.denoising_regret.probe import (
    _all_router_weights,
    _configure_torch_threads,
    _correlation,
    _extract_prediction,
    _load_checkpoint_model,
    _load_latent,
    _per_sample_mse,
    _rankdata,
)
from analyses.routing_translation.probe import (
    RouteInputCapture,
    _capture_native_forward,
)
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)


PROBE_VERSION = 1
DEFAULT_BLOCK_INDICES = (1, 5, 11)
PRIMARY_WEIGHT_MODE = "native"
SENSITIVITY_WEIGHT_MODES = ("candidate", "unit")
WEIGHT_MODES = (PRIMARY_WEIGHT_MODE, *SENSITIVITY_WEIGHT_MODES)


def _validate_weight_modes(weight_modes):
    modes = tuple(weight_modes)
    if not modes or len(modes) != len(set(modes)):
        raise ValueError("Weight modes must be nonempty and unique")
    unknown = set(modes) - set(WEIGHT_MODES)
    if unknown:
        raise ValueError(f"Unknown route-weight modes: {sorted(unknown)}")
    return modes


def _validate_moe_block_contract(model, block_indices):
    """Validate and describe the MoE blocks required by this probe."""

    indices = tuple(int(index) for index in block_indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Block indices must be nonempty and unique")
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("The timestep-utility probe requires model.blocks")
    depth = len(blocks)
    block_contracts = []
    for block_index in indices:
        if not 0 <= block_index < depth:
            raise ValueError(f"block_index {block_index} is outside the model")
        block = blocks[block_index]
        if not getattr(block, "use_moe", False):
            raise ValueError(f"block {block_index} is not an MoE block")
        moe_layer = getattr(block, "mlp", None)
        if moe_layer is None:
            raise ValueError(f"block {block_index} has no MoE layer")
        top_k = getattr(moe_layer, "top_k", None)
        if top_k != 1:
            raise ValueError(
                f"block {block_index} has top_k={top_k}; "
                "the timestep-utility probe requires top_k == 1"
            )
        router_weight_mode = getattr(moe_layer, "router_weight_mode", None)
        if router_weight_mode != "identity":
            raise ValueError(
                f"block {block_index} has router_weight_mode="
                f"{router_weight_mode!r}; the timestep-utility probe requires "
                "identity router weights"
            )
        block_contracts.append({
            "index": block_index,
            "use_moe": True,
            "top_k": top_k,
            "router_weight_mode": router_weight_mode,
        })
    return {
        "depth": depth,
        "blocks": block_contracts,
    }


@contextmanager
def _forced_route_state(moe_layer, route_ids, route_weights):
    """Force full top-1 route IDs and weights for each conditional sample."""

    if route_ids.ndim != 2 or route_weights.shape != route_ids.shape:
        raise ValueError("Forced route IDs and weights must be aligned matrices")
    if route_ids.dtype != torch.long:
        raise ValueError("Forced route IDs must use torch.long")
    if not bool(torch.isfinite(route_weights).all().item()):
        raise ValueError("Forced route weights must be finite")
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
            raise RuntimeError("Timestep-utility overrides require top_k == 1")
        if weights.shape[:2] != route_ids.shape:
            raise RuntimeError(
                "Forced route matrices must match batch and sequence dimensions"
            )
        forced_ids = route_ids.to(device=indices.device)
        forced_weights = route_weights.to(
            device=weights.device,
            dtype=weights.dtype,
        )
        if forced_ids.numel() and (
            forced_ids.min() < 0
            or forced_ids.max() >= this.num_routed_experts
        ):
            raise RuntimeError("Forced route IDs must name routed experts")
        conditional_rows = labels != 1000
        indices[conditional_rows, :, 0] = forced_ids[conditional_rows]
        weights[conditional_rows, :, 0] = forced_weights[conditional_rows]
        return weights, indices, auxiliary_loss

    if "compute_router" in moe_layer.__dict__:
        raise RuntimeError("MoE layer already has a compute_router override")
    moe_layer.compute_router = MethodType(compute_router_with_override, moe_layer)
    try:
        yield
    finally:
        del moe_layer.compute_router


def _validate_route_grid_inputs(
    token_indices,
    native_route_ids,
    native_route_weights,
    router_scores,
    num_experts,
):
    if token_indices.ndim != 1 or token_indices.dtype != torch.long:
        raise ValueError("Token indices must be a torch.long vector")
    if token_indices.numel() == 0:
        raise ValueError("At least one token is required")
    if native_route_ids.ndim != 1 or native_route_ids.dtype != torch.long:
        raise ValueError("Native route IDs must be a torch.long vector")
    if native_route_weights.shape != native_route_ids.shape:
        raise ValueError("Native route IDs and weights must align")
    if token_indices.min() < 0 or token_indices.max() >= native_route_ids.numel():
        raise ValueError("Token indices are outside the route sequence")
    if router_scores.shape != (token_indices.numel(), num_experts):
        raise ValueError("Router-score rows must align with tokens and experts")
    if native_route_ids.min() < 0 or native_route_ids.max() >= num_experts:
        raise ValueError("Native routes must name routed experts")
    if not bool(torch.isfinite(native_route_weights).all().item()):
        raise ValueError("Native route weights must be finite")
    if not bool(torch.isfinite(router_scores).all().item()):
        raise ValueError("Router scores must be finite")


def _exact_route_grid(
    model,
    moe_layer,
    noised_latent,
    timestep,
    label,
    target,
    token_indices,
    native_route_ids,
    native_route_weights,
    router_scores,
    num_experts,
    batch_size,
    weight_modes,
):
    """Evaluate all expert IDs with paired baselines under fixed weight semantics."""

    weight_modes = _validate_weight_modes(weight_modes)
    _validate_route_grid_inputs(
        token_indices,
        native_route_ids,
        native_route_weights,
        router_scores,
        num_experts,
    )
    rows_per_pair = 2 * len(weight_modes)
    if batch_size < rows_per_pair:
        raise ValueError(
            f"batch_size must be at least {rows_per_pair} for {len(weight_modes)} modes"
        )
    pairs_per_forward = batch_size // rows_per_pair
    candidate_ids = torch.arange(
        num_experts,
        device=token_indices.device,
        dtype=torch.long,
    )
    flat_tokens = token_indices.unsqueeze(1).expand(-1, num_experts).reshape(-1)
    flat_candidates = candidate_ids.unsqueeze(0).expand(
        token_indices.numel(),
        -1,
    ).reshape(-1)
    flat_native = native_route_ids[token_indices].unsqueeze(1).expand(
        -1,
        num_experts,
    ).reshape(-1)
    flat_native_weights = native_route_weights[token_indices].unsqueeze(1).expand(
        -1,
        num_experts,
    ).reshape(-1)
    flat_candidate_weights = router_scores.reshape(-1)

    changes = {mode: [] for mode in weight_modes}
    noop_mse = {mode: 0.0 for mode in weight_modes}
    noop_output = {mode: 0.0 for mode in weight_modes}
    target_channels = target.shape[1]
    for start in range(0, flat_tokens.numel(), pairs_per_forward):
        stop = min(start + pairs_per_forward, flat_tokens.numel())
        count = stop - start
        selected_tokens = flat_tokens[start:stop]
        selected_candidates = flat_candidates[start:stop]
        selected_candidate_weights = flat_candidate_weights[start:stop]

        route_id_batches = []
        route_weight_batches = []
        for mode in weight_modes:
            baseline_ids = native_route_ids.unsqueeze(0).expand(count, -1).clone()
            candidate_route_ids = baseline_ids.clone()
            batch_rows = torch.arange(count, device=token_indices.device)
            candidate_route_ids[batch_rows, selected_tokens] = selected_candidates

            baseline_weights = native_route_weights.unsqueeze(0).expand(
                count,
                -1,
            ).clone()
            candidate_route_weights = baseline_weights.clone()
            if mode == "candidate":
                candidate_route_weights[
                    batch_rows,
                    selected_tokens,
                ] = selected_candidate_weights
            elif mode == "unit":
                baseline_weights[batch_rows, selected_tokens] = 1.0
                candidate_route_weights[batch_rows, selected_tokens] = 1.0

            route_id_batches.extend((baseline_ids, candidate_route_ids))
            route_weight_batches.extend((baseline_weights, candidate_route_weights))

        route_id_matrix = torch.cat(route_id_batches, dim=0)
        route_weight_matrix = torch.cat(route_weight_batches, dim=0)
        model_batch = route_id_matrix.shape[0]
        batch_latent = noised_latent.repeat(model_batch, 1, 1, 1, 1)
        batch_timestep = timestep.repeat(model_batch)
        batch_label = label.repeat(model_batch)
        batch_target = target.repeat(count, 1, 1, 1)
        with torch.inference_mode(), _forced_route_state(
            moe_layer,
            route_id_matrix,
            route_weight_matrix,
        ):
            output = model(batch_latent, batch_timestep, context=batch_label)
        prediction = _extract_prediction(output, target_channels)

        offset = 0
        noop = selected_candidates == flat_native[start:stop]
        for mode in weight_modes:
            baseline_prediction = prediction[offset:offset + count]
            candidate_prediction = prediction[offset + count:offset + 2 * count]
            baseline_losses = _per_sample_mse(baseline_prediction, batch_target)
            candidate_losses = _per_sample_mse(candidate_prediction, batch_target)
            delta = candidate_losses - baseline_losses
            changes[mode].append(delta.cpu())
            if noop.any():
                noop_mse[mode] = max(
                    noop_mse[mode],
                    float(delta[noop].abs().max().item()),
                )
                noop_output[mode] = max(
                    noop_output[mode],
                    float((
                        candidate_prediction[noop] - baseline_prediction[noop]
                    ).abs().max().item()),
                )
            offset += 2 * count

    matrices = {
        mode: torch.cat(parts).reshape(token_indices.numel(), num_experts)
        for mode, parts in changes.items()
    }
    controls = {
        mode: {
            "max_abs_noop_mse_change": noop_mse[mode],
            "max_abs_noop_output_change": noop_output[mode],
        }
        for mode in weight_modes
    }
    return matrices, controls


def _solve_capacity_assignment(cost_matrix, capacities):
    """Minimize additive counterfactual cost under integral expert capacities."""

    costs = np.asarray(cost_matrix, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.int64)
    if costs.ndim != 2 or costs.shape[0] == 0 or costs.shape[1] < 2:
        raise ValueError("Cost matrix must be shaped [tokens, experts]")
    if not np.isfinite(costs).all():
        raise ValueError("Assignment costs must be finite")
    if capacities.shape != (costs.shape[1],) or (capacities < 0).any():
        raise ValueError("Capacities must be nonnegative and align with experts")
    if int(capacities.sum()) < costs.shape[0]:
        raise ValueError("Expert capacities cannot cover every token")
    slots = np.repeat(np.arange(costs.shape[1]), capacities)
    row_indices, slot_indices = linear_sum_assignment(costs[:, slots])
    if row_indices.size != costs.shape[0]:
        raise RuntimeError("Capacity assignment did not cover every token")
    assignment = np.empty(costs.shape[0], dtype=np.int64)
    assignment[row_indices] = slots[slot_indices]
    counts = np.bincount(assignment, minlength=costs.shape[1])
    if np.any(counts > capacities):
        raise RuntimeError("Capacity assignment exceeded an expert capacity")
    return assignment


def _load_statistics(assignments, num_experts):
    assignments = np.asarray(assignments, dtype=np.int64)
    if assignments.ndim != 1 or assignments.size == 0:
        raise ValueError("Assignments must be a nonempty vector")
    if assignments.min() < 0 or assignments.max() >= num_experts:
        raise ValueError("Assignments name an invalid expert")
    counts = np.bincount(assignments, minlength=num_experts).astype(np.float64)
    mean = counts.mean()
    sorted_counts = np.sort(counts)
    indices = np.arange(1, num_experts + 1, dtype=np.float64)
    gini = (
        (2.0 * np.sum(indices * sorted_counts))
        / (num_experts * sorted_counts.sum())
        - (num_experts + 1.0) / num_experts
    )
    return {
        "counts": counts.astype(np.int64).tolist(),
        "cv": float(counts.std() / mean),
        "gini": float(gini),
        "max_load": int(counts.max()),
        "active_experts": int((counts > 0).sum()),
    }


def _build_assignments(exact_changes, native_experts, capacity_factor):
    exact_changes = np.asarray(exact_changes, dtype=np.float64)
    native_experts = np.asarray(native_experts, dtype=np.int64)
    if exact_changes.ndim != 2 or native_experts.shape != (exact_changes.shape[0],):
        raise ValueError("Exact changes and native assignments must align")
    num_tokens, num_experts = exact_changes.shape
    native_capacities = np.bincount(native_experts, minlength=num_experts)
    balanced_capacity = int(math.ceil(capacity_factor * num_tokens / num_experts))
    balanced_capacities = np.full(num_experts, balanced_capacity, dtype=np.int64)
    return {
        "native": native_experts.copy(),
        "unconstrained_oracle": exact_changes.argmin(axis=1),
        "native_capacity_oracle": _solve_capacity_assignment(
            exact_changes,
            native_capacities,
        ),
        "balanced_capacity_oracle": _solve_capacity_assignment(
            exact_changes,
            balanced_capacities,
        ),
    }, {
        "native_capacities": native_capacities.tolist(),
        "balanced_capacity_per_expert": balanced_capacity,
        "capacity_factor": float(capacity_factor),
    }


def _exact_assignment_changes(
    model,
    moe_layer,
    noised_latent,
    timestep,
    label,
    target,
    token_indices,
    native_route_ids,
    native_route_weights,
    assignments,
):
    """Validate jointly forced assignments with a paired native row per policy."""

    names = tuple(assignments)
    if not names or names[0] != "native":
        raise ValueError("Assignments must begin with the native policy")
    num_tokens = token_indices.numel()
    route_id_batches = []
    route_weight_batches = []
    for name in names:
        assignment = torch.as_tensor(
            assignments[name],
            device=token_indices.device,
            dtype=torch.long,
        )
        if assignment.shape != (num_tokens,):
            raise ValueError(f"Assignment {name} does not align with sampled tokens")
        baseline_ids = native_route_ids.clone()
        candidate_ids = native_route_ids.clone()
        candidate_ids[token_indices] = assignment
        route_id_batches.extend((baseline_ids, candidate_ids))
        route_weight_batches.extend((native_route_weights, native_route_weights))

    route_id_matrix = torch.stack(route_id_batches)
    route_weight_matrix = torch.stack(route_weight_batches)
    model_batch = route_id_matrix.shape[0]
    with torch.inference_mode(), _forced_route_state(
        moe_layer,
        route_id_matrix,
        route_weight_matrix,
    ):
        output = model(
            noised_latent.repeat(model_batch, 1, 1, 1, 1),
            timestep.repeat(model_batch),
            context=label.repeat(model_batch),
        )
    prediction = _extract_prediction(output, target.shape[1])
    records = {}
    for index, name in enumerate(names):
        baseline_prediction = prediction[2 * index:2 * index + 1]
        candidate_prediction = prediction[2 * index + 1:2 * index + 2]
        baseline_loss = _per_sample_mse(baseline_prediction, target)[0]
        candidate_loss = _per_sample_mse(candidate_prediction, target)[0]
        records[name] = {
            "exact_mse_change": float((candidate_loss - baseline_loss).item()),
            "baseline_mse": float(baseline_loss.item()),
            "candidate_mse": float(candidate_loss.item()),
            "max_abs_output_change": float((
                candidate_prediction - baseline_prediction
            ).abs().max().item()),
        }
    return records


def _forced_native_control(
    model,
    moe_layer,
    noised_latent,
    timestep,
    label,
    target,
    native_route_ids,
    native_route_weights,
    unforced_prediction,
    unforced_loss,
):
    with torch.inference_mode(), _forced_route_state(
        moe_layer,
        native_route_ids.unsqueeze(0),
        native_route_weights.unsqueeze(0),
    ):
        output = model(noised_latent, timestep, context=label)
    prediction = _extract_prediction(output, target.shape[1])
    loss = _per_sample_mse(prediction, target)[0]
    return {
        "max_abs_forced_unforced_output_change": float((
            prediction - unforced_prediction
        ).abs().max().item()),
        "max_abs_forced_unforced_mse_change": float(
            abs(loss.item() - float(unforced_loss))
        ),
    }


def _summarize_token(router_scores, exact_changes, native_expert, base_mse):
    router_scores = np.asarray(router_scores, dtype=np.float64)
    exact_changes = np.asarray(exact_changes, dtype=np.float64)
    if router_scores.shape != exact_changes.shape or router_scores.ndim != 1:
        raise ValueError("Router scores and exact changes must be aligned vectors")
    if not 0 <= native_expert < exact_changes.size:
        raise ValueError("Native expert is outside the candidate set")
    if base_mse <= 0:
        raise ValueError("Base MSE must be positive")
    utility = -exact_changes
    oracle = int(exact_changes.argmin())
    regret = float(exact_changes[native_expert] - exact_changes[oracle])
    return {
        "native_expert": int(native_expert),
        "oracle_expert": oracle,
        "native_is_oracle": bool(native_expert == oracle),
        "native_router_weight": float(router_scores[native_expert]),
        "router_utility_spearman": _correlation(
            _rankdata(router_scores),
            _rankdata(utility),
        ),
        "native_regret": regret,
        "native_regret_relative": float(regret / base_mse),
        "oracle_exact_mse_change": float(exact_changes[oracle]),
        "exact_mse_change_range": float(exact_changes.max() - exact_changes.min()),
        "router_scores": router_scores.tolist(),
        "exact_mse_changes": exact_changes.tolist(),
    }


def _summarize_records(records):
    if not records:
        raise ValueError("At least one token record is required")
    correlations = [
        record["router_utility_spearman"]
        for record in records
        if record["router_utility_spearman"] is not None
    ]
    return {
        "num_tokens": len(records),
        "native_is_oracle_rate": float(np.mean([
            record["native_is_oracle"] for record in records
        ])),
        "mean_native_regret": float(np.mean([
            record["native_regret"] for record in records
        ])),
        "mean_native_regret_relative": float(np.mean([
            record["native_regret_relative"] for record in records
        ])),
        "mean_router_utility_spearman": (
            float(np.mean(correlations)) if correlations else None
        ),
        "positive_router_utility_spearman_rate": (
            float(np.mean(np.asarray(correlations) > 0))
            if correlations else None
        ),
        "mean_exact_mse_change_range": float(np.mean([
            record["exact_mse_change_range"] for record in records
        ])),
        "mean_native_router_weight": float(np.mean([
            record["native_router_weight"] for record in records
        ])),
    }


def _pairwise_order_inversion(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Rank-inversion vectors must align")
    discordant = 0
    comparable = 0
    for first, second in combinations(range(left.size), 2):
        left_delta = left[first] - left[second]
        right_delta = right[first] - right[second]
        if left_delta == 0 or right_delta == 0:
            continue
        comparable += 1
        discordant += int(left_delta * right_delta < 0)
    return float(discordant / comparable) if comparable else None


def _summarize_stage_dynamics(cells, sigmas):
    records = []
    for block_index in sorted({cell["block_index"] for cell in cells}):
        block_cells = {
            float(cell["sigma"]): cell
            for cell in cells
            if cell["block_index"] == block_index
        }
        if set(block_cells) != set(sigmas):
            raise ValueError("Every block must contain the complete sigma grid")
        token_maps = {
            sigma: {token["token_index"]: token for token in cell["tokens"]}
            for sigma, cell in block_cells.items()
        }
        reference_tokens = set(next(iter(token_maps.values())))
        if any(set(token_map) != reference_tokens for token_map in token_maps.values()):
            raise ValueError("Token identities must be fixed across sigma")
        for sigma_left, sigma_right in combinations(sigmas, 2):
            for token_index in sorted(reference_tokens):
                left = token_maps[sigma_left][token_index]
                right = token_maps[sigma_right][token_index]
                left_utility = -np.asarray(left["exact_mse_changes"])
                right_utility = -np.asarray(right["exact_mse_changes"])
                utility_rho = _correlation(
                    _rankdata(left_utility),
                    _rankdata(right_utility),
                )
                router_rho = _correlation(
                    _rankdata(left["router_scores"]),
                    _rankdata(right["router_scores"]),
                )
                records.append({
                    "block_index": int(block_index),
                    "token_index": int(token_index),
                    "sigma_left": float(sigma_left),
                    "sigma_right": float(sigma_right),
                    "utility_rank_spearman": utility_rho,
                    "router_rank_spearman": router_rho,
                    "router_minus_utility_rank_stability": (
                        float(router_rho - utility_rho)
                        if router_rho is not None and utility_rho is not None
                        else None
                    ),
                    "utility_pair_inversion_rate": _pairwise_order_inversion(
                        left_utility,
                        right_utility,
                    ),
                    "oracle_expert_changed": bool(
                        left["oracle_expert"] != right["oracle_expert"]
                    ),
                    "native_expert_changed": bool(
                        left["native_expert"] != right["native_expert"]
                    ),
                })
    valid_utility = [
        row["utility_rank_spearman"]
        for row in records
        if row["utility_rank_spearman"] is not None
    ]
    valid_router = [
        row["router_rank_spearman"]
        for row in records
        if row["router_rank_spearman"] is not None
    ]
    valid_gap = [
        row["router_minus_utility_rank_stability"]
        for row in records
        if row["router_minus_utility_rank_stability"] is not None
    ]
    valid_inversions = [
        row["utility_pair_inversion_rate"]
        for row in records
        if row["utility_pair_inversion_rate"] is not None
    ]
    return {
        "summary": {
            "num_paired_token_comparisons": len(records),
            "mean_utility_rank_spearman": (
                float(np.mean(valid_utility)) if valid_utility else None
            ),
            "mean_router_rank_spearman": (
                float(np.mean(valid_router)) if valid_router else None
            ),
            "mean_router_minus_utility_rank_stability": (
                float(np.mean(valid_gap)) if valid_gap else None
            ),
            "mean_utility_pair_inversion_rate": (
                float(np.mean(valid_inversions)) if valid_inversions else None
            ),
            "oracle_expert_flip_rate": (
                float(np.mean([
                    row["oracle_expert_changed"] for row in records
                ]))
                if records else None
            ),
            "native_expert_flip_rate": (
                float(np.mean([
                    row["native_expert_changed"] for row in records
                ]))
                if records else None
            ),
        },
        "records": records,
    }


def _probe_cell(
    model,
    moe_layer,
    capture,
    clean_latent,
    noise,
    label,
    sigma,
    num_train_timesteps,
    block_index,
    token_indices,
    sensitivity_token_count,
    exact_batch_size,
    capacity_factor,
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
    native_output, hidden_states, native_weights, native_indices = (
        _capture_native_forward(
            model,
            moe_layer,
            capture,
            noised_latent,
            timestep,
            label,
        )
    )
    native_prediction = _extract_prediction(native_output, target.shape[1])
    native_loss = _per_sample_mse(native_prediction, target)[0]
    native_route_ids = native_indices[0, :, 0]
    native_route_weights = native_weights[0, :, 0]
    router_scores = _all_router_weights(
        moe_layer,
        hidden_states,
        timestep,
    )[0, token_indices]
    if not torch.equal(router_scores.argmax(dim=-1), native_route_ids[token_indices]):
        raise RuntimeError("Captured native routes disagree with all-router scores")

    primary_matrices, primary_controls = _exact_route_grid(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        token_indices=token_indices,
        native_route_ids=native_route_ids,
        native_route_weights=native_route_weights,
        router_scores=router_scores,
        num_experts=moe_layer.num_routed_experts,
        batch_size=exact_batch_size,
        weight_modes=(PRIMARY_WEIGHT_MODE,),
    )
    sensitivity_indices = token_indices[:sensitivity_token_count]
    sensitivity_scores = router_scores[:sensitivity_token_count]
    sensitivity_matrices, sensitivity_controls = _exact_route_grid(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        token_indices=sensitivity_indices,
        native_route_ids=native_route_ids,
        native_route_weights=native_route_weights,
        router_scores=sensitivity_scores,
        num_experts=moe_layer.num_routed_experts,
        batch_size=exact_batch_size,
        weight_modes=SENSITIVITY_WEIGHT_MODES,
    )
    forced_control = _forced_native_control(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        native_route_ids=native_route_ids,
        native_route_weights=native_route_weights,
        unforced_prediction=native_prediction,
        unforced_loss=native_loss.item(),
    )

    primary_changes = primary_matrices[PRIMARY_WEIGHT_MODE].numpy()
    sampled_native = native_route_ids[token_indices].cpu().numpy()
    assignments, capacity_spec = _build_assignments(
        primary_changes,
        sampled_native,
        capacity_factor,
    )
    assignment_records = _exact_assignment_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        token_indices=token_indices,
        native_route_ids=native_route_ids,
        native_route_weights=native_route_weights,
        assignments=assignments,
    )
    for name, assignment in assignments.items():
        assignment = np.asarray(assignment, dtype=np.int64)
        predicted_change = float(primary_changes[
            np.arange(token_indices.numel()),
            assignment,
        ].sum())
        assignment_records[name].update({
            "assignment": assignment.tolist(),
            "predicted_additive_mse_change": predicted_change,
            "exact_mse_change_relative": float(
                assignment_records[name]["exact_mse_change"] / native_loss.item()
            ),
            "load": _load_statistics(
                assignment,
                moe_layer.num_routed_experts,
            ),
        })

    tokens = []
    for row, token_index in enumerate(token_indices.tolist()):
        token = _summarize_token(
            router_scores[row].cpu().numpy(),
            primary_changes[row],
            int(sampled_native[row]),
            native_loss.item(),
        )
        token["token_index"] = int(token_index)
        token["sensitivity"] = {}
        if row < sensitivity_token_count:
            for mode in SENSITIVITY_WEIGHT_MODES:
                changes = sensitivity_matrices[mode][row].numpy()
                oracle = int(changes.argmin())
                token["sensitivity"][mode] = {
                    "oracle_expert": oracle,
                    "native_is_oracle": bool(oracle == sampled_native[row]),
                    "native_regret": float(
                        changes[sampled_native[row]] - changes[oracle]
                    ),
                    "exact_mse_changes": changes.tolist(),
                }
        tokens.append(token)

    controls = {
        **forced_control,
        "weight_modes": {
            **primary_controls,
            **sensitivity_controls,
        },
    }
    return {
        "block_index": int(block_index),
        "sigma": float(sigma),
        "timestep": float(timestep.item()),
        "native_mse": float(native_loss.item()),
        "sampled_tokens": int(token_indices.numel()),
        "sensitivity_tokens": int(sensitivity_token_count),
        "summary": _summarize_records(tokens),
        "capacity_spec": capacity_spec,
        "assignments": assignment_records,
        "numerical_controls": controls,
        "tokens": tokens,
    }


def run_timestep_utility_probe(
    checkpoint_path,
    latent_path,
    label,
    sigmas=(0.2, 0.5, 0.8),
    block_indices=DEFAULT_BLOCK_INDICES,
    num_token_probes=8,
    sensitivity_token_count=2,
    exact_batch_size=24,
    capacity_factor=1.25,
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
    for path, description in (
        (checkpoint_path, "checkpoint"),
        (weights_checkpoint_path, "weights checkpoint"),
        (latent_path, "latent"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description.title()} does not exist: {path}")
    sigmas = tuple(float(sigma) for sigma in sigmas)
    if (
        not sigmas
        or len(sigmas) != len(set(sigmas))
        or any(not 0 < sigma < 1 for sigma in sigmas)
    ):
        raise ValueError("Sigmas must be unique and strictly between zero and one")
    block_indices = tuple(int(index) for index in block_indices)
    if num_token_probes < 2:
        raise ValueError("num_token_probes must be at least two")
    if not 1 <= sensitivity_token_count <= num_token_probes:
        raise ValueError("sensitivity_token_count must be within sampled tokens")
    if exact_batch_size < 4:
        raise ValueError("exact_batch_size must be at least four")
    if capacity_factor < 1.0:
        raise ValueError("capacity_factor must be at least one")
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
    model, state_name, weights_step, load_seconds = _load_checkpoint_model(
        runtime_cfg,
        weights_checkpoint_path,
        device,
    )
    if weights_step != checkpoint_step:
        raise ValueError(
            f"Loaded checkpoint step {weights_step} does not match canonical "
            f"checkpoint step {checkpoint_step}"
        )
    _validate_moe_block_contract(model, block_indices)

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    clean_latent = _load_latent(latent_path, latent_key, seed, device)
    torch.manual_seed(seed + 1)
    noise = torch.randn_like(clean_latent)
    label_tensor = torch.tensor([label], device=device, dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 2)
    num_tokens = int(model.x_embedder.num_patches)
    probe_count = min(num_token_probes, num_tokens)
    token_indices = torch.randperm(
        num_tokens,
        generator=generator,
        device=device,
    )[:probe_count]

    cells = []
    probe_start = time.perf_counter()
    for block_index in block_indices:
        moe_layer = model.blocks[block_index].mlp
        capture = RouteInputCapture(moe_layer)
        try:
            for sigma in sigmas:
                cells.append(_probe_cell(
                    model=model,
                    moe_layer=moe_layer,
                    capture=capture,
                    clean_latent=clean_latent,
                    noise=noise,
                    label=label_tensor,
                    sigma=sigma,
                    num_train_timesteps=runtime_cfg.num_train_timesteps,
                    block_index=block_index,
                    token_indices=token_indices,
                    sensitivity_token_count=min(
                        sensitivity_token_count,
                        probe_count,
                    ),
                    exact_batch_size=exact_batch_size,
                    capacity_factor=capacity_factor,
                ))
        finally:
            capture.close()
    probe_seconds = time.perf_counter() - probe_start

    all_tokens = [token for cell in cells for token in cell["tokens"]]
    per_block = {
        str(block_index): _summarize_records([
            token
            for cell in cells
            if cell["block_index"] == block_index
            for token in cell["tokens"]
        ])
        for block_index in block_indices
    }
    per_sigma = {
        str(sigma): _summarize_records([
            token
            for cell in cells
            if cell["sigma"] == sigma
            for token in cell["tokens"]
        ])
        for sigma in sigmas
    }
    stage_dynamics = _summarize_stage_dynamics(cells, sigmas)
    result = {
        "timestep_utility_probe_version": PROBE_VERSION,
        "diagnostic_scope": (
            "frozen-checkpoint natural-input equal-compute routing diagnostic; "
            "not a training or FID claim"
        ),
        "input_domain": (
            "sampled VAE posterior and fixed Gaussian noise with no spatial, "
            "semantic, or teacher-feature transformation"
        ),
        "hypotheses": {
            "routing_accuracy": (
                "native prototype affinity fails to identify the equal-compute "
                "expert with highest exact denoising utility"
            ),
            "stage_structure": (
                "expert-utility ranks change across diffusion noise levels more "
                "than router-affinity ranks track"
            ),
            "capacity_preserving": (
                "route reassignment can improve denoising MSE while preserving "
                "the exact sampled-token expert-count vector"
            ),
        },
        "primary_weight_mode": PRIMARY_WEIGHT_MODE,
        "weight_mode_definitions": {
            "native": "candidate expert keeps the token's native top-1 weight",
            "candidate": "candidate expert uses its own current router affinity",
            "unit": "native and candidate experts both use weight one",
        },
        "checkpoint": str(checkpoint_path),
        "weights_checkpoint": str(weights_checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "weights_checkpoint_step": weights_step,
        "checkpoint_state": state_name,
        "config": str(config_path),
        "model_name": runtime_cfg.model_name,
        "latent": str(latent_path),
        "latent_key": latent_key,
        "label": int(label),
        "sigmas": list(sigmas),
        "block_indices": list(block_indices),
        "token_indices": token_indices.cpu().tolist(),
        "num_token_probes": int(probe_count),
        "sensitivity_token_count": int(min(sensitivity_token_count, probe_count)),
        "exact_batch_size": int(exact_batch_size),
        "capacity_factor": float(capacity_factor),
        "seed": int(seed),
        "device": str(device),
        "num_threads": int(num_threads),
        "thread_config": thread_config,
        "model_load_seconds": load_seconds,
        "probe_seconds": probe_seconds,
        "summary": _summarize_records(all_tokens),
        "per_block": per_block,
        "per_sigma": per_sigma,
        "stage_dynamics": stage_dynamics,
        "cells": cells,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
