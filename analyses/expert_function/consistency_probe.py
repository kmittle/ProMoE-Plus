"""Test whether transported expert functions predict routing responsibility."""

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
    _correlation,
    _extract_prediction,
    _forced_routes,
    _load_checkpoint_model,
    _load_latent,
    _per_sample_mse,
    _rankdata,
)
from analyses.routing_translation.probe import (
    RouteInputCapture,
    _build_route_references,
    _capture_native_forward,
    _translate_spatial,
    _validate_shifts,
)
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)


PRIMARY_METRIC = "shared_residual_content_advantage"
ROUTER_METRIC = "router_affinity"
PROBE_VERSION = 3
FUNCTION_METRICS = (
    "raw_output_content_advantage",
    "centered_expert_content_advantage",
    "shared_residual_transport_cosine",
    "shared_residual_stability",
    PRIMARY_METRIC,
)
ALL_METRICS = (ROUTER_METRIC, *FUNCTION_METRICS)


def _validate_expert_output_tensors(
    original_outputs,
    shifted_outputs,
    original_shared,
    shifted_shared,
):
    if original_outputs.shape != shifted_outputs.shape:
        raise ValueError("Original and shifted expert outputs must align")
    if original_outputs.ndim != 3:
        raise ValueError("Expert outputs must be shaped [tokens, experts, hidden]")
    expected_shared = (
        original_outputs.shape[0],
        original_outputs.shape[2],
    )
    if original_shared.shape != expected_shared:
        raise ValueError("Original shared output does not match expert outputs")
    if shifted_shared.shape != expected_shared:
        raise ValueError("Shifted shared output does not match expert outputs")


def _validate_token_references(token_indices, content_source_indices, num_tokens):
    if (
        token_indices.ndim != 1
        or content_source_indices.ndim != 1
        or token_indices.shape != content_source_indices.shape
    ):
        raise ValueError("Token and content-source indices must be aligned vectors")
    if token_indices.numel() == 0:
        raise ValueError("At least one token reference is required")
    for name, indices in (
        ("token", token_indices),
        ("content source", content_source_indices),
    ):
        if indices.dtype != torch.long:
            raise ValueError(f"{name} indices must use torch.long")
        if indices.min() < 0 or indices.max() >= num_tokens:
            raise ValueError(f"{name} indices are outside the token grid")


def _transport_scores(shifted, content, position):
    content_cosine = F.cosine_similarity(shifted, content, dim=-1, eps=1e-8)
    position_cosine = F.cosine_similarity(shifted, position, dim=-1, eps=1e-8)
    shifted_norm = shifted.float().norm(dim=-1)
    content_norm = content.float().norm(dim=-1)
    scale_consistency = torch.exp(-torch.abs(torch.log(
        (shifted_norm + 1e-8) / (content_norm + 1e-8)
    )))
    return content_cosine, content_cosine - position_cosine, scale_consistency


def compute_function_scores(
    original_outputs,
    shifted_outputs,
    original_shared,
    shifted_shared,
    token_indices,
    content_source_indices,
):
    """Score each expert by whether its function follows transported content."""

    _validate_expert_output_tensors(
        original_outputs,
        shifted_outputs,
        original_shared,
        shifted_shared,
    )
    _validate_token_references(
        token_indices,
        content_source_indices,
        original_outputs.shape[0],
    )

    raw_original = original_outputs.float()
    raw_shifted = shifted_outputs.float()
    centered_original = raw_original - raw_original.mean(dim=1, keepdim=True)
    centered_shifted = raw_shifted - raw_shifted.mean(dim=1, keepdim=True)
    shared_original = raw_original - original_shared.float().unsqueeze(1)
    shared_shifted = raw_shifted - shifted_shared.float().unsqueeze(1)

    scores = {}
    for name, original, shifted in (
        ("raw_output", raw_original, raw_shifted),
        ("centered_expert", centered_original, centered_shifted),
        ("shared_residual", shared_original, shared_shifted),
    ):
        selected_shifted = shifted[token_indices]
        content_reference = original[content_source_indices]
        position_reference = original[token_indices]
        transport, advantage, scale = _transport_scores(
            selected_shifted,
            content_reference,
            position_reference,
        )
        scores[f"{name}_transport_cosine"] = transport
        scores[f"{name}_content_advantage"] = advantage
        scores[f"{name}_scale_consistency"] = scale
        scores[f"{name}_stability"] = transport * scale

    return {
        metric: scores[metric]
        for metric in FUNCTION_METRICS
    }


def _evaluate_all_experts(experts, hidden_states):
    if hidden_states.ndim != 2:
        raise ValueError("Hidden states must be shaped [tokens, hidden]")
    if not experts:
        raise ValueError("At least one routed expert is required")
    return torch.stack(
        [expert(hidden_states).float() for expert in experts],
        dim=1,
    )


def _exact_route_grid(
    model,
    moe_layer,
    noised_latent,
    timestep,
    label,
    target,
    token_indices,
    native_experts,
    num_experts,
    batch_size,
    unforced_prediction,
    unforced_loss,
):
    """Measure every equal-compute expert identity at each sampled token."""

    if token_indices.shape != native_experts.shape or token_indices.ndim != 1:
        raise ValueError("Token indices and native expert IDs must align")
    if token_indices.numel() == 0:
        raise ValueError("The exact route grid requires at least one token")
    if token_indices.dtype != torch.long or native_experts.dtype != torch.long:
        raise ValueError("Token indices and native expert IDs must use torch.long")
    if token_indices.device != native_experts.device:
        raise ValueError("Token indices and native expert IDs must share a device")
    if token_indices.min() < 0:
        raise ValueError("Token indices must be nonnegative")
    if isinstance(num_experts, bool) or not isinstance(num_experts, int):
        raise ValueError("num_experts must be an integer")
    if num_experts < 3:
        raise ValueError("The exact route grid requires at least three experts")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError("batch_size must be an integer")
    if batch_size < 2:
        raise ValueError("batch_size must be at least two for paired forwards")
    if native_experts.numel() and (
        native_experts.min() < 0 or native_experts.max() >= num_experts
    ):
        raise ValueError("Native expert IDs are outside the candidate grid")
    candidate_ids = torch.arange(
        num_experts,
        device=token_indices.device,
        dtype=torch.long,
    )
    flat_tokens = token_indices.unsqueeze(1).expand(-1, num_experts).reshape(-1)
    flat_native = native_experts.unsqueeze(1).expand(-1, num_experts).reshape(-1)
    flat_candidates = candidate_ids.unsqueeze(0).expand(
        token_indices.numel(), -1
    ).reshape(-1)

    with torch.inference_mode(), _forced_routes(
        moe_layer,
        token_indices.unsqueeze(0),
        native_experts.unsqueeze(0),
    ):
        forced_native_output = model(
            noised_latent,
            timestep,
            context=label,
        )
    forced_native_prediction = _extract_prediction(
        forced_native_output,
        target.shape[1],
    )
    forced_native_loss = _per_sample_mse(
        forced_native_prediction,
        target,
    )[0]

    changes = []
    max_abs_noop_output_change = 0.0
    pair_batch_size = batch_size // 2
    for start in range(0, flat_tokens.numel(), pair_batch_size):
        stop = min(start + pair_batch_size, flat_tokens.numel())
        count = stop - start
        paired_count = 2 * count
        batch_latent = noised_latent.repeat(paired_count, 1, 1, 1, 1)
        batch_timestep = timestep.repeat(paired_count)
        batch_label = label.repeat(paired_count)
        batch_target = target.repeat(count, 1, 1, 1)
        paired_tokens = torch.cat([
            flat_tokens[start:stop],
            flat_tokens[start:stop],
        ])
        paired_experts = torch.cat([
            flat_native[start:stop],
            flat_candidates[start:stop],
        ])
        with torch.inference_mode(), _forced_routes(
            moe_layer,
            paired_tokens,
            paired_experts,
        ):
            paired_output = model(
                batch_latent,
                batch_timestep,
                context=batch_label,
            )
        paired_prediction = _extract_prediction(
            paired_output,
            target.shape[1],
        )
        native_prediction = paired_prediction[:count]
        candidate_prediction = paired_prediction[count:]
        batch_native_losses = _per_sample_mse(native_prediction, batch_target)
        candidate_losses = _per_sample_mse(candidate_prediction, batch_target)

        noop = flat_candidates[start:stop] == flat_native[start:stop]
        if noop.any():
            max_abs_noop_output_change = max(
                max_abs_noop_output_change,
                float((
                    candidate_prediction[noop] - native_prediction[noop]
                ).abs().max().item()),
            )
        changes.append((candidate_losses - batch_native_losses).cpu())

    changes = torch.cat(changes).reshape(token_indices.numel(), num_experts)
    controls = {
        "max_abs_noop_mse_change": float(
            changes[
                torch.arange(token_indices.numel()),
                native_experts.cpu(),
            ].abs().max().item()
        ),
        "max_abs_noop_output_change": max_abs_noop_output_change,
        "max_abs_forced_unforced_output_change": float(
            (forced_native_prediction - unforced_prediction).abs().max().item()
        ),
        "max_abs_forced_unforced_mse_change": float(
            abs(forced_native_loss.item() - float(unforced_loss))
        ),
    }
    return changes, controls


def summarize_token(metric_scores, exact_changes, native_expert):
    exact_changes = np.asarray(exact_changes, dtype=np.float64)
    if exact_changes.ndim != 1 or exact_changes.size < 3:
        raise ValueError("Exact changes must contain at least three experts")
    if not np.isfinite(exact_changes).all():
        raise ValueError("Exact changes must be finite")
    if not 0 <= native_expert < exact_changes.size:
        raise ValueError("Native expert is outside the candidate set")
    for metric in ALL_METRICS:
        if metric not in metric_scores:
            raise KeyError(f"Missing metric: {metric}")
        values = np.asarray(metric_scores[metric], dtype=np.float64)
        if values.shape != exact_changes.shape:
            raise ValueError(f"Metric {metric} does not align with exact changes")
        if not np.isfinite(values).all():
            raise ValueError(f"Metric {metric} must be finite")

    utility = -exact_changes
    oracle_expert = int(exact_changes.argmin())
    router_values = np.asarray(metric_scores[ROUTER_METRIC], dtype=np.float64)
    result = {
        "native_expert": int(native_expert),
        "native_router_weight": float(router_values[native_expert]),
        "native_exact_mse_change": float(exact_changes[native_expert]),
        "oracle_expert": oracle_expert,
        "oracle_exact_mse_change": float(exact_changes[oracle_expert]),
        "exact_mse_change_range": float(exact_changes.max() - exact_changes.min()),
        "metrics": {},
    }
    for metric in ALL_METRICS:
        values = np.asarray(metric_scores[metric], dtype=np.float64)
        selected_expert = int(values.argmax())
        top_three = np.argsort(-values, kind="mergesort")[:3]
        result["metrics"][metric] = {
            "spearman_with_exact_utility": _correlation(
                _rankdata(values),
                _rankdata(utility),
            ),
            "selected_expert": selected_expert,
            "selected_exact_mse_change": float(exact_changes[selected_expert]),
            "selected_beats_native": bool(
                exact_changes[selected_expert] < exact_changes[native_expert]
            ),
            "selected_oracle_regret": float(
                exact_changes[selected_expert] - exact_changes[oracle_expert]
            ),
            "oracle_in_top3": bool(oracle_expert in top_three),
        }
    return result


def summarize_tokens(token_summaries):
    if not token_summaries:
        raise ValueError("At least one token summary is required")
    summary = {
        "num_tokens": len(token_summaries),
        "minimum_native_router_weight": float(min(
            row["native_router_weight"] for row in token_summaries
        )),
        "mean_native_router_weight": float(np.mean([
            row["native_router_weight"] for row in token_summaries
        ])),
        "native_is_oracle_rate": float(np.mean([
            row["native_expert"] == row["oracle_expert"]
            for row in token_summaries
        ])),
        "mean_oracle_exact_mse_change": float(np.mean([
            row["oracle_exact_mse_change"] for row in token_summaries
        ])),
        "mean_exact_mse_change_range": float(np.mean([
            row["exact_mse_change_range"] for row in token_summaries
        ])),
        "metrics": {},
    }
    for metric in ALL_METRICS:
        metric_rows = [row["metrics"][metric] for row in token_summaries]
        correlations = [
            row["spearman_with_exact_utility"]
            for row in metric_rows
            if row["spearman_with_exact_utility"] is not None
        ]
        summary["metrics"][metric] = {
            "valid_correlation_tokens": len(correlations),
            "mean_spearman_with_exact_utility": (
                float(np.mean(correlations)) if correlations else None
            ),
            "median_spearman_with_exact_utility": (
                float(np.median(correlations)) if correlations else None
            ),
            "positive_spearman_rate": (
                float(np.mean(np.asarray(correlations) > 0))
                if correlations else None
            ),
            "selected_beats_native_rate": float(np.mean([
                row["selected_beats_native"] for row in metric_rows
            ])),
            "mean_selected_exact_mse_change": float(np.mean([
                row["selected_exact_mse_change"] for row in metric_rows
            ])),
            "mean_selected_oracle_regret": float(np.mean([
                row["selected_oracle_regret"] for row in metric_rows
            ])),
            "oracle_top3_rate": float(np.mean([
                row["oracle_in_top3"] for row in metric_rows
            ])),
        }

    paired_differences = []
    for row in token_summaries:
        primary = row["metrics"][PRIMARY_METRIC][
            "spearman_with_exact_utility"
        ]
        router = row["metrics"][ROUTER_METRIC][
            "spearman_with_exact_utility"
        ]
        if primary is not None and router is not None:
            paired_differences.append(primary - router)
    summary["primary_metric"] = PRIMARY_METRIC
    summary["primary_minus_router_mean_spearman"] = (
        float(np.mean(paired_differences)) if paired_differences else None
    )
    summary["primary_beats_router_spearman_rate"] = (
        float(np.mean(np.asarray(paired_differences) > 0))
        if paired_differences else None
    )
    return summary


def _sample_token_references(
    num_tokens,
    grid_size,
    token_shift,
    count,
    generator,
    device,
):
    dummy_ids = torch.arange(num_tokens, device=device)
    content_sources, _, valid = _build_route_references(
        dummy_ids,
        dummy_ids,
        grid_size,
        token_shift,
    )
    valid_ids = torch.where(valid)[0]
    sample_count = min(count, valid_ids.numel())
    order = torch.randperm(
        valid_ids.numel(),
        generator=generator,
        device=device,
    )[:sample_count]
    token_indices = valid_ids[order]
    return token_indices, content_sources[token_indices], int(valid_ids.numel())


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
    num_token_probes,
    exact_batch_size,
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
    original_noised = (
        (1.0 - sigma_tensor) * clean_latent + sigma_tensor * noise
    )
    shifted_clean = _translate_spatial(clean_latent, dy, dx)
    shifted_noise = _translate_spatial(noise, dy, dx)
    shifted_noised = (
        (1.0 - sigma_tensor) * shifted_clean + sigma_tensor * shifted_noise
    )
    shifted_target = (shifted_noise - shifted_clean).squeeze(2)

    _, original_hidden, _, _ = _capture_native_forward(
        model,
        moe_layer,
        capture,
        original_noised,
        timestep,
        label,
    )
    shifted_output, shifted_hidden, _, shifted_indices = _capture_native_forward(
        model,
        moe_layer,
        capture,
        shifted_noised,
        timestep,
        label,
    )
    shifted_prediction = _extract_prediction(
        shifted_output,
        shifted_target.shape[1],
    )
    shifted_loss = _per_sample_mse(
        shifted_prediction,
        shifted_target,
    )[0]

    num_tokens = int(original_hidden.shape[1])
    grid_size = math.isqrt(num_tokens)
    if grid_size * grid_size != num_tokens:
        raise RuntimeError("Function transport requires a square token grid")
    token_shift = (dy // patch_size, dx // patch_size)
    token_indices, content_sources, valid_count = _sample_token_references(
        num_tokens=num_tokens,
        grid_size=grid_size,
        token_shift=token_shift,
        count=num_token_probes,
        generator=generator,
        device=original_hidden.device,
    )

    routed_experts = moe_layer.experts[:moe_layer.num_routed_experts]
    with torch.inference_mode():
        original_expert_outputs = _evaluate_all_experts(
            routed_experts,
            original_hidden[0],
        )
        shifted_expert_outputs = _evaluate_all_experts(
            routed_experts,
            shifted_hidden[0],
        )
        original_shared = moe_layer.shared_expert(original_hidden[0]).float()
        shifted_shared = moe_layer.shared_expert(shifted_hidden[0]).float()
        function_scores = compute_function_scores(
            original_expert_outputs,
            shifted_expert_outputs,
            original_shared,
            shifted_shared,
            token_indices,
            content_sources,
        )
        router_scores = _all_router_weights(moe_layer, shifted_hidden)[
            0, token_indices
        ].float()
        native_experts = shifted_indices[0, token_indices, 0]

    exact_changes, controls = _exact_route_grid(
        model=model,
        moe_layer=moe_layer,
        noised_latent=shifted_noised,
        timestep=timestep,
        label=label,
        target=shifted_target,
        token_indices=token_indices,
        native_experts=native_experts,
        num_experts=moe_layer.num_routed_experts,
        batch_size=exact_batch_size,
        unforced_prediction=shifted_prediction,
        unforced_loss=shifted_loss.item(),
    )

    token_summaries = []
    candidate_records = []
    for row in range(token_indices.numel()):
        score_row = {
            ROUTER_METRIC: router_scores[row].cpu().numpy(),
            **{
                metric: values[row].cpu().numpy()
                for metric, values in function_scores.items()
            },
        }
        exact_row = exact_changes[row].numpy()
        token_summary = summarize_token(
            score_row,
            exact_row,
            int(native_experts[row].item()),
        )
        token_summary.update({
            "token_index": int(token_indices[row].item()),
            "content_source_index": int(content_sources[row].item()),
        })
        token_summaries.append(token_summary)
        for expert_id in range(moe_layer.num_routed_experts):
            candidate_records.append({
                "token_index": int(token_indices[row].item()),
                "content_source_index": int(content_sources[row].item()),
                "expert": expert_id,
                "is_native": bool(expert_id == native_experts[row].item()),
                "exact_mse_change": float(exact_row[expert_id]),
                "scores": {
                    metric: float(values[expert_id])
                    for metric, values in score_row.items()
                },
            })

    return {
        "sigma": float(sigma),
        "timestep": float(timestep.item()),
        "shift_latent": [int(dy), int(dx)],
        "shift_tokens": [int(token_shift[0]), int(token_shift[1])],
        "valid_tokens": valid_count,
        "sampled_tokens": int(token_indices.numel()),
        "shifted_native_mse": float(shifted_loss.item()),
        "summary": summarize_tokens(token_summaries),
        "numerical_controls": controls,
        "tokens": token_summaries,
        "candidates": candidate_records,
    }


def _collect_token_summaries(cells):
    return [token for cell in cells for token in cell["tokens"]]


def run_expert_function_consistency_probe(
    checkpoint_path,
    latent_path,
    label,
    sigmas=(0.276, 0.5, 0.724),
    shifts=((0, 2), (0, -2), (2, 0), (-2, 0)),
    block_index=3,
    num_token_probes=8,
    exact_batch_size=24,
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
    if num_token_probes < 2:
        raise ValueError("num_token_probes must be at least two")
    if exact_batch_size < 2:
        raise ValueError("exact_batch_size must be at least two")
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
        raise ValueError("The function consistency probe requires top_k == 1")
    if moe_layer.router_weight_mode != "identity":
        raise ValueError(
            "The function consistency probe requires identity router weights"
        )
    if not moe_layer.use_shared_expert:
        raise ValueError("The function consistency probe requires a shared expert")

    patch_size = model.patch_size
    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
            raise ValueError("The probe requires square patches")
        patch_size = patch_size[0]
    patch_size = int(patch_size)
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
    cells = []
    probe_start = time.perf_counter()
    try:
        for sigma in sigmas:
            for shift in shifts:
                cells.append(_probe_cell(
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
                    num_token_probes=num_token_probes,
                    exact_batch_size=exact_batch_size,
                    generator=generator,
                ))
    finally:
        capture.close()
    probe_seconds = time.perf_counter() - probe_start

    per_sigma = {
        str(sigma): summarize_tokens(_collect_token_summaries([
            cell for cell in cells if cell["sigma"] == sigma
        ]))
        for sigma in sigmas
    }
    per_shift = {
        f"{dy}:{dx}": summarize_tokens(_collect_token_summaries([
            cell for cell in cells if cell["shift_latent"] == [dy, dx]
        ]))
        for dy, dx in shifts
    }
    result = {
        "expert_function_consistency_probe_version": PROBE_VERSION,
        "diagnostic_scope": (
            "frozen-checkpoint equal-compute routing diagnostic; not a FID claim"
        ),
        "hypothesis": (
            "an expert whose residual function follows transported content rather "
            "than fixed position has higher denoising responsibility"
        ),
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_definition": (
            "cos(r_shifted, r_original_content) - "
            "cos(r_shifted, r_original_position), where "
            "r_e(h) = E_e(h) - E_shared(h)"
        ),
        "exact_intervention": (
            "force one token to each routed expert at one block while preserving "
            "the native top-1 router weight and activated compute"
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
        "num_token_probes_per_cell": int(num_token_probes),
        "exact_batch_size": int(exact_batch_size),
        "seed": int(seed),
        "device": str(device),
        "num_threads": int(num_threads),
        "thread_config": thread_config,
        "model_load_seconds": load_seconds,
        "probe_seconds": probe_seconds,
        "summary": summarize_tokens(_collect_token_summaries(cells)),
        "per_sigma": per_sigma,
        "per_shift": per_shift,
        "cells": cells,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
