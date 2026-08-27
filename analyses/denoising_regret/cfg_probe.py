from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch

from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)

from .probe import (
    RoutingProbeCapture,
    _all_router_weights,
    _choose_challengers,
    _configure_torch_threads,
    _correlation,
    _evaluate_experts,
    _extract_prediction,
    _forced_routes,
    _load_checkpoint_model,
    _load_latent,
    _per_sample_mse,
    _rankdata,
)


def _scale_key(scale):
    return str(float(scale))


def _validate_guidance_scales(guidance_scales, analysis_scale):
    if not guidance_scales:
        raise ValueError("At least one guidance scale is required")
    scales = [float(scale) for scale in guidance_scales]
    if any(not np.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("Guidance scales must be finite and positive")
    if len(scales) != len(set(scales)):
        raise ValueError("Guidance scales must be unique")
    analysis_scale = float(analysis_scale)
    if analysis_scale not in scales:
        raise ValueError("analysis_scale must be present in guidance_scales")
    return scales, analysis_scale


def _guidance_metrics(
    conditional_prediction,
    unconditional_prediction,
    target,
    guidance_scales,
):
    if not (
        conditional_prediction.shape
        == unconditional_prediction.shape
        == target.shape
    ):
        raise ValueError("Conditional, unconditional, and target shapes must match")

    conditional = conditional_prediction.double()
    unconditional = unconditional_prediction.double()
    target = target.double()
    residual = (conditional - unconditional).flatten(1)
    correction = (target - unconditional).flatten(1)

    dot = (residual * correction).sum(dim=1)
    residual_square = residual.square().sum(dim=1)
    correction_square = correction.square().sum(dim=1)
    denominator = (residual_square * correction_square).sqrt().clamp_min(1e-24)
    alignment = dot / denominator
    optimal_scale = dot / residual_square.clamp_min(1e-24)
    projection_residual = correction - optimal_scale.unsqueeze(1) * residual
    projection_mse = projection_residual.square().mean(dim=1)

    guided_mse = {}
    for scale in guidance_scales:
        guided_prediction = unconditional + float(scale) * (
            conditional - unconditional
        )
        guided_mse[_scale_key(scale)] = _per_sample_mse(
            guided_prediction,
            target,
        )

    return {
        "guided_mse": guided_mse,
        "alignment": alignment,
        "optimal_scale": optimal_scale,
        "projection_mse": projection_mse,
    }


def _first_order_validation(predicted, exact):
    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    if predicted.shape != exact.shape or predicted.size == 0:
        raise ValueError("First-order and exact changes must be non-empty and aligned")

    predicted_better = predicted < 0
    exact_better = exact < 0
    true_positive = predicted_better & exact_better
    return {
        "pearson": _correlation(predicted, exact),
        "spearman": _correlation(_rankdata(predicted), _rankdata(exact)),
        "sign_agreement": float(
            np.mean(np.signbit(predicted) == np.signbit(exact))
        ),
        "predicted_better_rate": float(predicted_better.mean()),
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


def summarize_cfg_records(records, analysis_scale):
    if not records:
        raise ValueError("At least one CFG probe record is required")
    scale_key = _scale_key(analysis_scale)
    conditional_exact = np.asarray(
        [record["conditional_exact_mse_change"] for record in records],
        dtype=np.float64,
    )
    conditional_first_order = np.asarray(
        [record["conditional_first_order_change"] for record in records],
        dtype=np.float64,
    )
    guided_exact = np.asarray(
        [record["guided_exact_mse_change"][scale_key] for record in records],
        dtype=np.float64,
    )
    guided_first_order = np.asarray(
        [record["guided_first_order_change"][scale_key] for record in records],
        dtype=np.float64,
    )
    alignment_change = np.asarray(
        [record["guidance_alignment_change"] for record in records],
        dtype=np.float64,
    )
    projection_change = np.asarray(
        [record["guidance_projection_mse_change"] for record in records],
        dtype=np.float64,
    )

    conditional_better = conditional_exact < 0
    guided_better = guided_exact < 0
    sign_agreement = np.signbit(conditional_exact) == np.signbit(guided_exact)
    alignment_improved = alignment_change > 0
    projection_improved = projection_change < 0

    return {
        "num_probes": int(len(records)),
        "analysis_scale": float(analysis_scale),
        "conditional_better_rate": float(conditional_better.mean()),
        "guided_better_rate": float(guided_better.mean()),
        "conditional_guided_pearson": _correlation(
            conditional_exact,
            guided_exact,
        ),
        "conditional_guided_spearman": _correlation(
            _rankdata(conditional_exact),
            _rankdata(guided_exact),
        ),
        "conditional_guided_sign_agreement": float(sign_agreement.mean()),
        "route_inversion_rate": float((~sign_agreement).mean()),
        "conditional_better_guided_worse_rate": float(
            (conditional_better & ~guided_better).mean()
        ),
        "conditional_worse_guided_better_rate": float(
            (~conditional_better & guided_better).mean()
        ),
        "guidance_alignment_improved_rate": float(alignment_improved.mean()),
        "guidance_projection_improved_rate": float(projection_improved.mean()),
        "guided_better_and_alignment_improved_rate": float(
            (guided_better & alignment_improved).mean()
        ),
        "guided_better_and_projection_improved_rate": float(
            (guided_better & projection_improved).mean()
        ),
        "median_abs_conditional_exact_mse_change": float(
            np.median(np.abs(conditional_exact))
        ),
        "median_abs_guided_exact_mse_change": float(
            np.median(np.abs(guided_exact))
        ),
        "conditional_first_order_validation": _first_order_validation(
            conditional_first_order,
            conditional_exact,
        ),
        "guided_first_order_validation": _first_order_validation(
            guided_first_order,
            guided_exact,
        ),
    }


def _exact_paired_changes(
    model,
    moe_layer,
    noised_latent,
    timestep,
    conditional_label,
    target,
    unconditional_prediction,
    token_indices,
    expert_ids,
    batch_size,
    guidance_scales,
):
    changes = {
        "conditional_mse": [],
        "guided_mse": {
            _scale_key(scale): [] for scale in guidance_scales
        },
        "alignment": [],
        "optimal_scale": [],
        "projection_mse": [],
    }
    target_channels = target.shape[1]
    for start in range(0, token_indices.numel(), batch_size):
        stop = min(start + batch_size, token_indices.numel())
        count = stop - start
        batch_latent = noised_latent.repeat(count, 1, 1, 1, 1)
        batch_timestep = timestep.repeat(count)
        batch_label = conditional_label.repeat(count)
        batch_target = target.repeat(count, 1, 1, 1)
        batch_unconditional = unconditional_prediction.repeat(count, 1, 1, 1)

        with torch.inference_mode():
            base_output = model(batch_latent, batch_timestep, context=batch_label)
            base_prediction = _extract_prediction(base_output, target_channels)
            with _forced_routes(
                moe_layer,
                token_indices[start:stop],
                expert_ids[start:stop],
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

            base_conditional_mse = _per_sample_mse(
                base_prediction,
                batch_target,
            )
            alternative_conditional_mse = _per_sample_mse(
                alternative_prediction,
                batch_target,
            )
            base_guidance = _guidance_metrics(
                base_prediction,
                batch_unconditional,
                batch_target,
                guidance_scales,
            )
            alternative_guidance = _guidance_metrics(
                alternative_prediction,
                batch_unconditional,
                batch_target,
                guidance_scales,
            )

        changes["conditional_mse"].append(
            (alternative_conditional_mse - base_conditional_mse).cpu()
        )
        for scale in guidance_scales:
            key = _scale_key(scale)
            changes["guided_mse"][key].append(
                (
                    alternative_guidance["guided_mse"][key]
                    - base_guidance["guided_mse"][key]
                ).cpu()
            )
        for name in ("alignment", "optimal_scale", "projection_mse"):
            changes[name].append(
                (alternative_guidance[name] - base_guidance[name]).cpu()
            )

    changes["conditional_mse"] = torch.cat(changes["conditional_mse"])
    for scale in guidance_scales:
        key = _scale_key(scale)
        changes["guided_mse"][key] = torch.cat(changes["guided_mse"][key])
    for name in ("alignment", "optimal_scale", "projection_mse"):
        changes[name] = torch.cat(changes[name])
    return changes


def _probe_sigma(
    model,
    moe_layer,
    capture,
    clean_latent,
    noise,
    conditional_label,
    unconditional_label,
    sigma,
    num_train_timesteps,
    num_token_probes,
    candidate_mode,
    exact_batch_size,
    guidance_scales,
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

    with torch.no_grad():
        unconditional_output = model(
            noised_latent,
            timestep,
            context=unconditional_label,
        )
        unconditional_prediction = _extract_prediction(
            unconditional_output,
            target_channels,
        ).detach()

    capture.start()
    conditional_output = model(
        noised_latent,
        timestep,
        context=conditional_label,
    )
    conditional_prediction = _extract_prediction(
        conditional_output,
        target_channels,
    )
    if capture.moe_output is None:
        raise RuntimeError("The paired-CFG hook did not capture an MoE output")

    conditional_loss = _per_sample_mse(
        conditional_prediction,
        target,
    ).mean()
    guidance = _guidance_metrics(
        conditional_prediction,
        unconditional_prediction,
        target,
        guidance_scales,
    )
    loss_terms = [("conditional", conditional_loss)] + [
        (key, guidance["guided_mse"][key].mean())
        for key in (_scale_key(scale) for scale in guidance_scales)
    ]
    gradients = {}
    for index, (name, loss) in enumerate(loss_terms):
        gradient, = torch.autograd.grad(
            loss,
            capture.moe_output,
            retain_graph=index + 1 < len(loss_terms),
        )
        gradients[name] = gradient
    capture.stop()

    hidden_states = capture.hidden_states
    router_weights = _all_router_weights(moe_layer, hidden_states, timestep)
    current_ids = router_weights.argmax(dim=-1)
    num_tokens = hidden_states.shape[1]
    probe_count = min(num_token_probes, num_tokens)
    token_indices = torch.randperm(
        num_tokens,
        generator=generator,
        device=hidden_states.device,
    )[:probe_count]
    probe_hidden = hidden_states[0, token_indices]
    probe_router_weights = router_weights[0, token_indices]
    probe_current = current_ids[0, token_indices]
    probe_challenger, probe_uses_runner_up = _choose_challengers(
        probe_router_weights,
        probe_current,
        candidate_mode,
        generator,
    )

    with torch.no_grad():
        experts = moe_layer.experts[:moe_layer.num_routed_experts]
        current_outputs = _evaluate_experts(
            experts,
            probe_hidden,
            probe_current,
        )
        challenger_outputs = _evaluate_experts(
            experts,
            probe_hidden,
            probe_challenger,
        )
        probe_slots = torch.arange(probe_count, device=hidden_states.device)
        current_weights = probe_router_weights[
            probe_slots,
            probe_current,
        ].unsqueeze(-1)
        challenger_weights = probe_router_weights[
            probe_slots,
            probe_challenger,
        ].unsqueeze(-1)
        output_delta = current_weights * (
            challenger_outputs - current_outputs
        )
        first_order = {}
        for name, gradient in gradients.items():
            first_order[name] = (
                gradient[0, token_indices].float() * output_delta
            ).sum(dim=-1)
        router_margin = (current_weights - challenger_weights).squeeze(-1)

    exact = _exact_paired_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        conditional_label=conditional_label,
        target=target,
        unconditional_prediction=unconditional_prediction,
        token_indices=token_indices,
        expert_ids=probe_challenger,
        batch_size=exact_batch_size,
        guidance_scales=guidance_scales,
    )
    noop_count = min(exact_batch_size, probe_count)
    noop = _exact_paired_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        conditional_label=conditional_label,
        target=target,
        unconditional_prediction=unconditional_prediction,
        token_indices=token_indices[:noop_count],
        expert_ids=probe_current[:noop_count],
        batch_size=exact_batch_size,
        guidance_scales=guidance_scales,
    )

    records = []
    for index in range(probe_count):
        records.append({
            "sigma": float(sigma),
            "timestep": float(timestep.item()),
            "token_index": int(token_indices[index].item()),
            "current_expert": int(probe_current[index].item()),
            "challenger_expert": int(probe_challenger[index].item()),
            "challenger_source": (
                "runner-up" if probe_uses_runner_up[index].item() else "random"
            ),
            "selected_router_weight": float(current_weights[index].item()),
            "challenger_router_weight": float(
                challenger_weights[index].item()
            ),
            "router_margin": float(router_margin[index].item()),
            "conditional_first_order_change": float(
                first_order["conditional"][index].item()
            ),
            "conditional_exact_mse_change": float(
                exact["conditional_mse"][index].item()
            ),
            "guided_first_order_change": {
                _scale_key(scale): float(
                    first_order[_scale_key(scale)][index].item()
                )
                for scale in guidance_scales
            },
            "guided_exact_mse_change": {
                _scale_key(scale): float(
                    exact["guided_mse"][_scale_key(scale)][index].item()
                )
                for scale in guidance_scales
            },
            "guidance_alignment_change": float(exact["alignment"][index].item()),
            "guidance_projection_mse_change": float(
                exact["projection_mse"][index].item()
            ),
            "optimal_guidance_scale_change": float(
                exact["optimal_scale"][index].item()
            ),
        })

    scale_one_key = _scale_key(1.0)
    controls = {
        "noop_num_probes": int(noop_count),
        "noop_conditional_max_abs_mse_change": float(
            noop["conditional_mse"].abs().max().item()
        ),
        "noop_guided_max_abs_mse_change": {
            _scale_key(scale): float(
                noop["guided_mse"][_scale_key(scale)].abs().max().item()
            )
            for scale in guidance_scales
        },
        "noop_alignment_max_abs_change": float(
            noop["alignment"].abs().max().item()
        ),
        "noop_projection_mse_max_abs_change": float(
            noop["projection_mse"].abs().max().item()
        ),
    }
    if scale_one_key in exact["guided_mse"]:
        controls["scale_one_exact_equivalence_max_abs"] = float(
            (
                exact["guided_mse"][scale_one_key]
                - exact["conditional_mse"]
            ).abs().max().item()
        )
        controls["scale_one_base_equivalence_abs"] = float(
            abs(
                guidance["guided_mse"][scale_one_key].mean().item()
                - conditional_loss.item()
            )
        )

    baseline = {
        "conditional_mse": float(conditional_loss.item()),
        "unconditional_mse": float(
            _per_sample_mse(unconditional_prediction, target).mean().item()
        ),
        "guided_mse": {
            _scale_key(scale): float(
                guidance["guided_mse"][_scale_key(scale)].mean().item()
            )
            for scale in guidance_scales
        },
        "guidance_alignment": float(guidance["alignment"].mean().item()),
        "guidance_projection_mse": float(
            guidance["projection_mse"].mean().item()
        ),
        "optimal_guidance_scale": float(
            guidance["optimal_scale"].mean().item()
        ),
    }
    return records, baseline, controls


def run_cfg_probe(
    checkpoint_path,
    latent_path,
    label,
    sigmas,
    block_index=3,
    num_token_probes=32,
    candidate_mode="mixed",
    exact_batch_size=4,
    guidance_scales=(1.0, 1.5),
    analysis_scale=1.5,
    latent_key="latent",
    seed=0,
    device="cpu",
    num_threads=8,
    weights_checkpoint_path=None,
):
    guidance_scales, analysis_scale = _validate_guidance_scales(
        guidance_scales,
        analysis_scale,
    )
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
        raise ValueError("The paired-CFG probe requires top_k == 1")
    if not moe_layer.use_uncond_expert:
        raise ValueError("The paired-CFG probe requires an unconditional expert")

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    clean_latent = _load_latent(latent_path, latent_key, seed, device)
    torch.manual_seed(seed + 1)
    noise = torch.randn_like(clean_latent)
    conditional_label = torch.tensor([label], device=device, dtype=torch.long)
    unconditional_label = torch.tensor(
        [runtime_cfg.num_classes],
        device=device,
        dtype=torch.long,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 2)

    capture = RoutingProbeCapture(moe_layer)
    records = []
    baseline = {}
    numerical_controls = {}
    probe_start = time.perf_counter()
    try:
        for sigma in sigmas:
            sigma_records, sigma_baseline, sigma_controls = _probe_sigma(
                model=model,
                moe_layer=moe_layer,
                capture=capture,
                clean_latent=clean_latent,
                noise=noise,
                conditional_label=conditional_label,
                unconditional_label=unconditional_label,
                sigma=float(sigma),
                num_train_timesteps=runtime_cfg.num_train_timesteps,
                num_token_probes=num_token_probes,
                candidate_mode=candidate_mode,
                exact_batch_size=exact_batch_size,
                guidance_scales=guidance_scales,
                generator=generator,
            )
            records.extend(sigma_records)
            baseline[_scale_key(sigma)] = sigma_baseline
            numerical_controls[_scale_key(sigma)] = sigma_controls
    finally:
        capture.close()
    probe_seconds = time.perf_counter() - probe_start

    per_sigma = {}
    for sigma in sigmas:
        sigma_records = [
            record for record in records if record["sigma"] == float(sigma)
        ]
        per_sigma[_scale_key(sigma)] = summarize_cfg_records(
            sigma_records,
            analysis_scale,
        )

    result = {
        "cfg_probe_version": 1,
        "diagnostic_scope": "teacher-forced oracle; not a direct FID claim",
        "counterfactual_route_weight": "selected",
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
        "candidate_mode": candidate_mode,
        "sigmas": [float(sigma) for sigma in sigmas],
        "guidance_scales": guidance_scales,
        "analysis_scale": analysis_scale,
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
        "summary": summarize_cfg_records(records, analysis_scale),
        "per_sigma": per_sigma,
        "records": records,
    }
    del model
    gc.collect()
    return result
