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
SPATIAL_CONTROL_CANDIDATES = 256
SPATIAL_CONTROL_SEARCH_STEPS = 256
SPATIAL_CONTROL_MIN_DERANGEMENT = 0.5
SPATIAL_CONTROL_MAX_ADJACENCY_TV = 0.1
INTERVENTION_NAMES = (
    "native",
    "noop_native",
    *(name for stratum in STRATUM_NAMES for name in (
        f"{stratum}_content",
        f"{stratum}_spatial",
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
    grid_size,
    spatial_search_steps=SPATIAL_CONTROL_SEARCH_STEPS,
):
    routes = []
    controls = {}
    control_seeds = torch.randint(
        0,
        torch.iinfo(torch.int64).max,
        (len(STRATUM_NAMES), 2),
        generator=generator,
        device=generator.device,
        dtype=torch.int64,
    ).cpu().tolist()
    for stratum_index, name in enumerate(STRATUM_NAMES):
        random_seed, spatial_seed = control_seeds[stratum_index]
        random_generator = torch.Generator(device=native_ids.device)
        random_generator.manual_seed(random_seed)
        spatial_generator = torch.Generator(device=native_ids.device)
        spatial_generator.manual_seed(spatial_seed)
        mask = stratum_masks[name]
        content_route = native_ids.clone()
        content_route[mask] = content_ids[mask]
        random_route, random_changed, random_control_available = \
            _random_matched_routes(
                native_ids,
                content_route,
                valid_mask,
                random_generator,
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
        spatial_route, spatial_control = _spatially_matched_routes(
            native_ids=native_ids,
            content_ids=content_route,
            changed_mask=mask,
            random_ids=random_route,
            generator=spatial_generator,
            num_routed_experts=num_routed_experts,
            grid_size=grid_size,
            search_steps=spatial_search_steps,
        )
        spatial_changed = spatial_route != native_ids
        spatial_hist = torch.bincount(
            spatial_route[mask],
            minlength=num_routed_experts,
        )
        if not torch.equal(spatial_changed, mask):
            raise RuntimeError(f"{name} spatial control changed its support")
        if not torch.equal(content_hist, spatial_hist):
            raise RuntimeError(
                f"{name} spatial control changed the replacement histogram"
            )
        routes.extend((content_route, spatial_route, random_route))
        controls[name] = {
            "support_equal": True,
            "replacement_histogram_equal": True,
            "random_control_seed": int(random_seed),
            "spatial_control_seed": int(spatial_seed),
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
            **spatial_control,
        }
    return routes, controls


def _four_neighbor_pair_histograms(
    route_ids,
    changed_mask,
    grid_size,
    num_routed_experts,
):
    if route_ids.ndim == 1:
        route_ids = route_ids.unsqueeze(0)
        squeeze = True
    elif route_ids.ndim == 2:
        squeeze = False
    else:
        raise ValueError("Route IDs must be one route or a batch of routes")
    if changed_mask.ndim != 1:
        raise ValueError("Changed mask must be a flat vector")
    if route_ids.shape[1] != changed_mask.numel():
        raise ValueError("Route IDs and changed mask must align")
    if grid_size < 2 or grid_size * grid_size != changed_mask.numel():
        raise ValueError("Four-neighbor statistics require a square grid")
    if num_routed_experts < 1:
        raise ValueError("Number of routed experts must be positive")
    if route_ids.numel() and (
        route_ids.min() < 0 or route_ids.max() >= num_routed_experts
    ):
        raise ValueError("Route IDs are outside the routed expert range")

    token_grid = torch.arange(
        changed_mask.numel(),
        device=changed_mask.device,
    ).reshape(grid_size, grid_size)
    left = torch.cat((token_grid[:, :-1].reshape(-1), token_grid[:-1].reshape(-1)))
    right = torch.cat((token_grid[:, 1:].reshape(-1), token_grid[1:].reshape(-1)))
    active_edges = changed_mask[left] | changed_mask[right]
    left = left[active_edges]
    right = right[active_edges]
    if left.numel() == 0:
        raise ValueError("Spatial control needs at least one incident grid edge")

    left_ids = route_ids[:, left]
    right_ids = route_ids[:, right]
    pair_ids = (
        torch.minimum(left_ids, right_ids) * num_routed_experts
        + torch.maximum(left_ids, right_ids)
    )
    histograms = torch.zeros(
        route_ids.shape[0],
        num_routed_experts * num_routed_experts,
        device=route_ids.device,
        dtype=torch.float64,
    )
    histograms.scatter_add_(
        1,
        pair_ids,
        torch.ones_like(pair_ids, dtype=torch.float64),
    )
    histograms /= float(left.numel())
    return histograms[0] if squeeze else histograms


def _spatially_matched_routes(
    native_ids,
    content_ids,
    changed_mask,
    random_ids,
    generator,
    num_routed_experts,
    grid_size,
    candidate_count=SPATIAL_CONTROL_CANDIDATES,
    search_steps=SPATIAL_CONTROL_SEARCH_STEPS,
    min_derangement=SPATIAL_CONTROL_MIN_DERANGEMENT,
    max_adjacency_tv=SPATIAL_CONTROL_MAX_ADJACENCY_TV,
):
    if not (
        native_ids.ndim == content_ids.ndim == changed_mask.ndim
        == random_ids.ndim == 1
    ):
        raise ValueError("Spatial control inputs must be flat vectors")
    if not (
        native_ids.shape == content_ids.shape == changed_mask.shape
        == random_ids.shape
    ):
        raise ValueError("Spatial control inputs must align")
    if changed_mask.dtype != torch.bool:
        raise ValueError("Changed mask must be boolean")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if search_steps < 1:
        raise ValueError("search_steps must be positive")
    if not 0 < min_derangement <= 1:
        raise ValueError("min_derangement must be in (0, 1]")
    if not 0 <= max_adjacency_tv <= 1:
        raise ValueError("max_adjacency_tv must be in [0, 1]")
    if not torch.equal(changed_mask, native_ids != content_ids):
        raise ValueError("Changed mask must equal the content disagreement support")
    if not torch.equal(changed_mask, native_ids != random_ids):
        raise ValueError("Random control must preserve the disagreement support")

    changed_indices = torch.where(changed_mask)[0]
    changed_count = int(changed_indices.numel())
    minimum_mismatches = (
        next(
            count
            for count in range(changed_count + 1)
            if count / changed_count >= min_derangement
        )
        if changed_count
        else 0
    )
    fallback = content_ids.clone()
    base_diagnostics = {
        "spatial_control_available": False,
        "spatial_candidate_count": int(candidate_count),
        "spatial_search_steps": int(search_steps),
        "spatial_search_strategy": (
            "reach_minimum_derangement_then_nonincreasing_adjacency_tv"
        ),
        "spatial_evaluated_candidates": 0,
        "spatial_unique_candidates": 0,
        "spatial_deranged_candidates": 0,
        "spatial_eligible_candidates": 0,
        "spatial_min_derangement": float(min_derangement),
        "spatial_minimum_mismatches": int(minimum_mismatches),
        "spatial_max_adjacency_tv": float(max_adjacency_tv),
        "spatial_differs_from_content_rate": None,
        "spatial_adjacency_tv": None,
        "random_adjacency_tv": None,
        "spatial_not_worse_than_random": None,
        "spatial_best_deranged_adjacency_tv": None,
        "spatial_best_deranged_differs_from_content_rate": None,
        "spatial_best_deranged_tv_minus_random": None,
        "spatial_rejection_counts": {
            "below_minimum_derangement": 0,
            "above_maximum_adjacency_tv": 0,
            "worse_than_random": 0,
        },
        "spatial_unavailable_reason": None,
    }
    if changed_count < 2:
        base_diagnostics["spatial_unavailable_reason"] = (
            "fewer_than_two_changed_tokens"
        )
        return fallback, base_diagnostics

    content_changed = content_ids[changed_indices]
    native_changed = native_ids[changed_indices]
    reference_histogram = _four_neighbor_pair_histograms(
        content_ids,
        changed_mask,
        grid_size,
        num_routed_experts,
    )

    assignments = content_changed.unsqueeze(0).repeat(candidate_count, 1)
    rows = torch.arange(candidate_count, device=native_ids.device)
    candidate_tv = torch.zeros(
        candidate_count,
        device=native_ids.device,
        dtype=torch.float64,
    )
    mismatch_counts = torch.zeros(
        candidate_count,
        device=native_ids.device,
        dtype=torch.long,
    )
    for _ in range(search_steps):
        pair = torch.randint(
            changed_count,
            (candidate_count, 2),
            generator=generator,
            device=native_ids.device,
        )
        left = pair[:, 0]
        right = pair[:, 1]
        left_values = assignments[rows, left]
        right_values = assignments[rows, right]
        can_swap = (
            (left != right)
            & (left_values != right_values)
            & (right_values != native_changed[left])
            & (left_values != native_changed[right])
        )
        valid_rows = rows[can_swap]
        valid_left = left[can_swap]
        valid_right = right[can_swap]
        proposed_assignments = assignments.clone()
        proposed_assignments[valid_rows, valid_left] = right_values[can_swap]
        proposed_assignments[valid_rows, valid_right] = left_values[can_swap]

        proposed_candidates = native_ids.unsqueeze(0).repeat(
            candidate_count,
            1,
        )
        proposed_candidates[:, changed_indices] = proposed_assignments
        proposed_histograms = _four_neighbor_pair_histograms(
            proposed_candidates,
            changed_mask,
            grid_size,
            num_routed_experts,
        )
        proposed_tv = 0.5 * (
            proposed_histograms - reference_histogram.unsqueeze(0)
        ).abs().sum(dim=1)
        proposed_mismatch_counts = (
            proposed_assignments != content_changed.unsqueeze(0)
        ).sum(dim=1)

        below_minimum = mismatch_counts < minimum_mismatches
        reaches_or_approaches_minimum = (
            proposed_mismatch_counts >= mismatch_counts
        )
        stays_deranged_and_improves_tv = (
            (proposed_mismatch_counts >= minimum_mismatches)
            & (proposed_tv <= candidate_tv + 1e-12)
        )
        accept = can_swap & (
            (below_minimum & reaches_or_approaches_minimum)
            | (~below_minimum & stays_deranged_and_improves_tv)
        )
        assignments[accept] = proposed_assignments[accept]
        candidate_tv[accept] = proposed_tv[accept]
        mismatch_counts[accept] = proposed_mismatch_counts[accept]

    candidates = native_ids.unsqueeze(0).repeat(candidate_count, 1)
    candidates[:, changed_indices] = assignments
    derangement = (assignments != content_changed).double().mean(dim=1)
    random_histogram = _four_neighbor_pair_histograms(
        random_ids,
        changed_mask,
        grid_size,
        num_routed_experts,
    )
    random_tv = float(
        (0.5 * (random_histogram - reference_histogram).abs().sum()).item()
    )
    meets_derangement = derangement >= min_derangement
    meets_max_tv = candidate_tv <= max_adjacency_tv
    not_worse_than_random = candidate_tv <= random_tv + 1e-12
    eligible = meets_derangement & meets_max_tv & not_worse_than_random
    eligible_count = int(eligible.sum().item())
    deranged_indices = torch.where(meets_derangement)[0]
    if deranged_indices.numel():
        best_deranged_local = torch.argmin(candidate_tv[deranged_indices])
        best_deranged_index = deranged_indices[best_deranged_local]
        best_deranged_tv = float(candidate_tv[best_deranged_index].item())
        best_deranged_rate = float(derangement[best_deranged_index].item())
    else:
        best_deranged_tv = None
        best_deranged_rate = None
    diagnostics = {
        **base_diagnostics,
        "spatial_evaluated_candidates": int(candidate_count),
        "spatial_unique_candidates": int(
            torch.unique(assignments, dim=0).shape[0]
        ),
        "spatial_deranged_candidates": int(meets_derangement.sum().item()),
        "spatial_eligible_candidates": eligible_count,
        "random_adjacency_tv": random_tv,
        "spatial_best_deranged_adjacency_tv": best_deranged_tv,
        "spatial_best_deranged_differs_from_content_rate": (
            best_deranged_rate
        ),
        "spatial_best_deranged_tv_minus_random": (
            best_deranged_tv - random_tv
            if best_deranged_tv is not None
            else None
        ),
        "spatial_rejection_counts": {
            "below_minimum_derangement": int(
                (~meets_derangement).sum().item()
            ),
            "above_maximum_adjacency_tv": int(
                (meets_derangement & ~meets_max_tv).sum().item()
            ),
            "worse_than_random": int(
                (
                    meets_derangement
                    & meets_max_tv
                    & ~not_worse_than_random
                ).sum().item()
            ),
        },
    }
    if eligible_count == 0:
        diagnostics["spatial_unavailable_reason"] = (
            "no_candidate_met_all_constraints"
        )
        return fallback, diagnostics

    eligible_indices = torch.where(eligible)[0]
    best_local = torch.argmin(candidate_tv[eligible_indices])
    best_index = eligible_indices[best_local]
    spatial_ids = candidates[best_index].clone()
    spatial_tv = float(candidate_tv[best_index].item())
    spatial_derangement = float(derangement[best_index].item())
    diagnostics.update({
        "spatial_control_available": True,
        "spatial_differs_from_content_rate": spatial_derangement,
        "spatial_adjacency_tv": spatial_tv,
        "spatial_not_worse_than_random": spatial_tv <= random_tv + 1e-12,
        "spatial_unavailable_reason": None,
    })
    return spatial_ids, diagnostics


def _redact_unavailable_spatial_metrics(metrics, stratum_controls):
    redacted = dict(metrics)
    for stratum in STRATUM_NAMES:
        if not stratum_controls[stratum]["spatial_control_available"]:
            redacted[f"{stratum}_spatial"] = None
    return redacted


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
        grid_size,
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
    mse = _redact_unavailable_spatial_metrics(mse, stratum_controls)
    mse_change = _redact_unavailable_spatial_metrics(
        mse_change,
        stratum_controls,
    )
    relative_mse_change = _redact_unavailable_spatial_metrics(
        relative_mse_change,
        stratum_controls,
    )

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
        random_paired = [
            record for record in nonempty
            if record["numerical_controls"]["stratum_random_controls"]
            [stratum]["random_control_available"]
        ]
        spatial_paired = [
            record for record in nonempty
            if record["numerical_controls"]["stratum_random_controls"]
            [stratum]["spatial_control_available"]
        ]
        content_key = f"{stratum}_content"
        spatial_key = f"{stratum}_spatial"
        random_key = f"{stratum}_random"
        random_contrasts = np.asarray([
            record["relative_mse_change"][content_key]
            - record["relative_mse_change"][random_key]
            for record in random_paired
        ], dtype=np.float64)
        spatial_contrasts = np.asarray([
            record["relative_mse_change"][content_key]
            - record["relative_mse_change"][spatial_key]
            for record in spatial_paired
        ], dtype=np.float64)
        summary["strata"][stratum] = {
            "nonempty_cells": len(nonempty),
            "valid_random_control_cells": len(random_paired),
            "valid_spatial_control_cells": len(spatial_paired),
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
                    for record in random_paired
                ]))
                if random_paired
                else None
            ),
            "spatial_mean_relative_mse_change": (
                float(np.mean([
                    record["relative_mse_change"][spatial_key]
                    for record in spatial_paired
                ]))
                if spatial_paired
                else None
            ),
            "content_minus_random_mean": (
                float(random_contrasts.mean()) if random_paired else None
            ),
            "content_beats_random_rate": (
                float((random_contrasts < 0).mean()) if random_paired else None
            ),
            "content_minus_spatial_mean": (
                float(spatial_contrasts.mean()) if spatial_paired else None
            ),
            "content_beats_spatial_rate": (
                float((spatial_contrasts < 0).mean()) if spatial_paired else None
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
        "routing_translation_stratified_probe_version": 5,
        "diagnostic_scope": (
            "teacher-forced fixed-compute stratum interventions with a spatially "
            "matched wrong-correspondence control; not a FID claim"
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
        "spatial_control": {
            "candidate_count": SPATIAL_CONTROL_CANDIDATES,
            "search_steps_per_candidate": SPATIAL_CONTROL_SEARCH_STEPS,
            "search_strategy": (
                "start from the content correspondence, use support-safe "
                "histogram-preserving swaps to reach the minimum derangement, "
                "then accept only nonincreasing adjacency TV"
            ),
            "minimum_derangement": SPATIAL_CONTROL_MIN_DERANGEMENT,
            "maximum_four_neighbor_pair_tv": (
                SPATIAL_CONTROL_MAX_ADJACENCY_TV
            ),
            "edge_scope": (
                "horizontal and vertical token-grid edges incident to the "
                "intervened stratum"
            ),
            "reachability_diagnostics": (
                "report evaluated and unique candidates, candidates meeting "
                "the minimum derangement, the best deranged adjacency TV, and "
                "mutually exclusive rejection counts without relaxing any "
                "acceptance threshold"
            ),
            "random_streams": (
                "draw fixed per-cell, per-stratum seeds for random and spatial "
                "controls before either control consumes random numbers"
            ),
        },
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
