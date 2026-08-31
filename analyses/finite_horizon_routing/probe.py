"""Run quota-preserving expert interventions through finite denoising rollouts."""

from __future__ import annotations

import copy
import gc
import hashlib
import os
import stat
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from config import cfg as base_cfg
from utils import deep_update

from analyses.denoising_regret.probe import (
    _all_router_weights,
    _compute_router,
    _configure_torch_threads,
    _extract_prediction,
    _load_checkpoint_model,
    _load_latent,
    _per_sample_mse,
)
from analyses.routing_translation.probe import (
    RouteInputCapture,
    _capture_native_forward,
)
from analyses.t_SNE.checkpoint_utils import (
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)
from analyses.timestep_utility.cycle_probe import (
    _build_short_cycle_arm,
    _stable_seed,
)
from analyses.timestep_utility.probe import (
    _forced_route_state,
    _validate_moe_block_contract,
)

from .protocol import (
    BLOCK_INDICES,
    CANDIDATE_CHUNK_SIZE,
    CANDIDATE_COUNT,
    HORIZONS,
    NUM_TRAIN_TIMESTEPS,
    PROBE_VERSION,
    SAMPLE_SHIFT,
    SAMPLE_STEPS,
    SCHEDULER_SHIFT,
    START_INDICES,
    analytic_flow_state,
    euler_flow_step,
    sampling_sigmas,
    summarize_cell_records,
    validate_count_preserving_candidates,
    validate_schedule_positions,
)


NUMERICAL_EPSILON_RELATIVE = 1e-7
EMA_CHECKPOINT_STATE = "ema_model_state_dict"


def _sha256_handle(handle, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    handle.seek(0)
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _stat_identity(file_stat):
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


@contextmanager
def _open_stable_regular_file(path, description):
    path = Path(path)
    flags = os.O_RDONLY
    for name in ("O_NOFOLLOW", "O_CLOEXEC"):
        value = getattr(os, name, None)
        if value is None:
            raise OSError(f"The platform does not provide {name}")
        flags |= value
    descriptor = os.open(path, flags)
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError(f"{description} must be a regular file")
        with os.fdopen(descriptor, "rb", buffering=0) as handle:
            descriptor = None
            yield handle, opened_stat
            final_stat = os.fstat(handle.fileno())
            if _stat_identity(final_stat) != _stat_identity(opened_stat):
                raise RuntimeError(f"{description} changed while it was open")
            try:
                path_stat = os.stat(path, follow_symlinks=False)
            except FileNotFoundError as error:
                raise RuntimeError(f"{description} path disappeared while it was open") from error
            if _stat_identity(path_stat) != _stat_identity(opened_stat):
                raise RuntimeError(f"{description} path changed while it was open")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_verified_runtime_cfg(config_path, expected_sha256=None):
    """Hash and parse one config through the same stable file handle."""

    config_path = Path(config_path)
    with _open_stable_regular_file(config_path, "Checkpoint config") as (
        handle,
        opened_stat,
    ):
        digest = _sha256_handle(handle)
        if expected_sha256 is not None and digest != str(expected_sha256):
            raise ValueError("Checkpoint config SHA256 differs from the sealed protocol")
        config_bytes = handle.read()
        if _sha256_handle(handle) != digest:
            raise RuntimeError("Checkpoint config changed while it was parsed")
        try:
            payload = yaml.safe_load(config_bytes.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError("Checkpoint config is not valid UTF-8 YAML") from error
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint config must contain a YAML mapping")

    runtime_cfg = copy.deepcopy(base_cfg)
    runtime_payload = copy.deepcopy(payload)
    runtime_payload["custom_cfg_name"] = config_path.stem
    deep_update(runtime_cfg, runtime_payload)
    return runtime_cfg, payload, {
        "size": opened_stat.st_size,
        "sha256": digest,
    }


@contextmanager
def _open_checkpoint_files(checkpoint_path, weights_checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    weights_checkpoint_path = Path(weights_checkpoint_path)
    with _open_stable_regular_file(
        checkpoint_path,
        "Canonical checkpoint",
    ) as (canonical_handle, canonical_stat):
        with _open_stable_regular_file(
            weights_checkpoint_path,
            "Weights checkpoint",
        ) as (candidate_handle, candidate_stat):
            same_file = (
                canonical_stat.st_dev == candidate_stat.st_dev
                and canonical_stat.st_ino == candidate_stat.st_ino
            )
            if same_file:
                yield canonical_handle, canonical_handle, True
            else:
                yield canonical_handle, candidate_handle, False


def _checkpoint_identity_from_handles(
    canonical_handle,
    weights_handle,
    same_file,
    expected_size=None,
    expected_sha256=None,
):
    canonical_size = os.fstat(canonical_handle.fileno()).st_size
    if expected_size is not None and canonical_size != int(expected_size):
        raise ValueError("Canonical checkpoint size differs from the sealed protocol")
    canonical_sha256 = _sha256_handle(canonical_handle)
    if expected_sha256 is not None:
        expected_sha256 = str(expected_sha256)
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("Expected checkpoint SHA256 is malformed")
        if canonical_sha256 != expected_sha256:
            raise ValueError("Canonical checkpoint SHA256 differs from the sealed protocol")

    weights_size = os.fstat(weights_handle.fileno()).st_size
    if weights_size != canonical_size:
        raise ValueError("Weights checkpoint size differs from the canonical checkpoint")
    weights_sha256 = (
        canonical_sha256 if same_file else _sha256_handle(weights_handle)
    )
    if weights_sha256 != canonical_sha256:
        raise ValueError("Weights checkpoint SHA256 differs from the canonical checkpoint")
    return {
        "canonical_size": canonical_size,
        "canonical_sha256": canonical_sha256,
        "weights_size": weights_size,
        "weights_sha256": weights_sha256,
        "same_file": bool(same_file),
    }


def _checkpoint_identity(
    checkpoint_path,
    weights_checkpoint_path,
    expected_size=None,
    expected_sha256=None,
):
    with _open_checkpoint_files(
        checkpoint_path,
        weights_checkpoint_path,
    ) as (canonical_handle, weights_handle, same_file):
        return _checkpoint_identity_from_handles(
            canonical_handle,
            weights_handle,
            same_file,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )


@contextmanager
def _verified_checkpoint_for_loading(
    checkpoint_path,
    weights_checkpoint_path,
    expected_size=None,
    expected_sha256=None,
):
    with _open_checkpoint_files(
        checkpoint_path,
        weights_checkpoint_path,
    ) as (canonical_handle, weights_handle, same_file):
        identity = _checkpoint_identity_from_handles(
            canonical_handle,
            weights_handle,
            same_file,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        yield weights_handle, identity
        observed_after_load = _checkpoint_identity_from_handles(
            canonical_handle,
            weights_handle,
            same_file,
            expected_size=identity["canonical_size"],
            expected_sha256=identity["canonical_sha256"],
        )
        if observed_after_load != identity:
            raise RuntimeError("Checkpoint identity changed while model weights loaded")


def _latent_identity_from_handle(
    handle,
    expected_size=None,
    expected_sha256=None,
):
    size = os.fstat(handle.fileno()).st_size
    if expected_size is not None and size != int(expected_size):
        raise ValueError("Latent size differs from the sealed protocol")
    sha256 = _sha256_handle(handle)
    if expected_sha256 is not None and sha256 != str(expected_sha256):
        raise ValueError("Latent SHA256 differs from the sealed protocol")
    return {"size": size, "sha256": sha256}


@contextmanager
def _verified_latent_for_loading(
    latent_path,
    expected_size=None,
    expected_sha256=None,
):
    with _open_stable_regular_file(latent_path, "Latent") as (handle, _):
        identity = _latent_identity_from_handle(
            handle,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        yield handle, identity
        observed_after_load = _latent_identity_from_handle(
            handle,
            expected_size=identity["size"],
            expected_sha256=identity["sha256"],
        )
        if observed_after_load != identity:
            raise RuntimeError("Latent identity changed while it was loaded")


def _start_captures(captures):
    for capture in captures.values():
        capture.start()


def _stop_captures(captures):
    for capture in captures.values():
        capture.stop()


def _close_captures(captures):
    for capture in captures.values():
        capture.close()


def _forward_with_routes(
    model,
    captures,
    inputs,
    timestep,
    labels,
    target_channels,
    forced_block_index=None,
    forced_route_ids=None,
    forced_route_weights=None,
):
    forced = forced_block_index is not None
    if forced != (forced_route_ids is not None and forced_route_weights is not None):
        raise ValueError("Forced block, route IDs, and route weights must be supplied together")
    if forced and (
        forced_route_ids.shape != forced_route_weights.shape
        or forced_route_ids.shape != (inputs.shape[0], model.x_embedder.num_patches)
    ):
        raise ValueError("Forced route matrices do not match the model batch")

    _start_captures(captures)
    try:
        context = nullcontext()
        if forced:
            context = _forced_route_state(
                model.blocks[forced_block_index].mlp,
                forced_route_ids,
                forced_route_weights,
            )
        with torch.inference_mode(), context:
            output = model(inputs, timestep, context=labels)
    finally:
        _stop_captures(captures)

    prediction = _extract_prediction(output, target_channels)
    routes = {}
    with torch.inference_mode():
        for block_index, capture in captures.items():
            if capture.hidden_states is None or capture.labels is None:
                raise RuntimeError(f"MoE block {block_index} did not run")
            if forced and block_index == forced_block_index:
                routes[block_index] = forced_route_ids.detach().cpu()
                continue
            _, indices, auxiliary_loss = _compute_router(
                model.blocks[block_index].mlp,
                capture.hidden_states,
                capture.labels,
                timestep,
            )
            if auxiliary_loss is not None:
                raise RuntimeError("Frozen eval router unexpectedly returned auxiliary loss")
            routes[block_index] = indices[:, :, 0].detach().cpu()
    return prediction, routes


def _rollout(
    model,
    captures,
    initial_state,
    labels,
    clean_latent,
    noise,
    sigmas,
    start_index,
    num_train_timesteps,
    forced_block_index=None,
    forced_route_ids=None,
    forced_route_weights=None,
):
    batch_size = initial_state.shape[0]
    if labels.shape != (batch_size,):
        raise ValueError("Rollout labels must align with the batch")
    state = initial_state
    target_channels = clean_latent.shape[1]
    first_prediction = None
    states = {}
    routes = {}
    for step_offset in range(max(HORIZONS)):
        sigma = float(sigmas[start_index + step_offset])
        next_sigma = float(sigmas[start_index + step_offset + 1])
        timestep = torch.full(
            (batch_size,),
            sigma * num_train_timesteps,
            device=state.device,
            dtype=state.dtype,
        )
        kwargs = {}
        if step_offset == 0 and forced_block_index is not None:
            kwargs = {
                "forced_block_index": forced_block_index,
                "forced_route_ids": forced_route_ids,
                "forced_route_weights": forced_route_weights,
            }
        prediction, step_routes = _forward_with_routes(
            model=model,
            captures=captures,
            inputs=state,
            timestep=timestep,
            labels=labels,
            target_channels=target_channels,
            **kwargs,
        )
        if step_offset == 0:
            first_prediction = prediction.detach()
        routes[str(step_offset)] = step_routes
        state = euler_flow_step(
            state,
            prediction.unsqueeze(2),
            sigma,
            next_sigma,
        )
        horizon = step_offset + 1
        if horizon in HORIZONS:
            states[str(horizon)] = state.detach()
    if first_prediction is None or set(states) != {str(value) for value in HORIZONS}:
        raise RuntimeError("Rollout did not produce the locked horizon grid")
    return {
        "first_prediction": first_prediction,
        "states": states,
        "routes": routes,
    }


def _rollout_losses(rollout, clean_latent, noise, sigmas, start_index):
    target_velocity = (noise - clean_latent).squeeze(2)
    batch_size = rollout["first_prediction"].shape[0]
    losses = {
        "immediate": _per_sample_mse(
            rollout["first_prediction"],
            target_velocity.expand(batch_size, -1, -1, -1),
        )
    }
    for horizon in HORIZONS:
        target_state = analytic_flow_state(
            clean_latent,
            noise,
            sigmas[start_index + horizon],
        )
        losses[str(horizon)] = _per_sample_mse(
            rollout["states"][str(horizon)],
            target_state.expand(batch_size, -1, -1, -1, -1),
        )
    return losses


def _max_duplicate_drift(rollout, losses):
    controls = {
        "first_prediction": float(
            (rollout["first_prediction"] - rollout["first_prediction"][0:1])
            .abs().max().item()
        ),
        "immediate_mse": float((losses["immediate"] - losses["immediate"][0]).abs().max().item()),
        "horizons": {},
    }
    for horizon in HORIZONS:
        key = str(horizon)
        controls["horizons"][key] = {
            "state": float(
                (rollout["states"][key] - rollout["states"][key][0:1])
                .abs().max().item()
            ),
            "mse": float((losses[key] - losses[key][0]).abs().max().item()),
        }
    return controls


def _max_rollout_difference(left, right, left_losses, right_losses):
    controls = {
        "first_prediction": float(
            (left["first_prediction"] - right["first_prediction"]).abs().max().item()
        ),
        "immediate_mse": float(
            (left_losses["immediate"] - right_losses["immediate"]).abs().max().item()
        ),
        "horizons": {},
    }
    for horizon in HORIZONS:
        key = str(horizon)
        controls["horizons"][key] = {
            "state": float((left["states"][key] - right["states"][key]).abs().max().item()),
            "mse": float((left_losses[key] - right_losses[key]).abs().max().item()),
        }
    return controls


def _route_divergence(routes, pair_index):
    baseline_row = 2 * pair_index
    candidate_row = baseline_row + 1
    divergence = {}
    for step_offset, block_routes in routes.items():
        divergence[step_offset] = {
            str(block_index): float(
                (route_ids[candidate_row] != route_ids[baseline_row])
                .double().mean().item()
            )
            for block_index, route_ids in block_routes.items()
        }
    return divergence


def _candidate_router_margin(candidate, router_scores):
    tokens = torch.as_tensor(
        candidate["tokens"],
        device=router_scores.device,
        dtype=torch.long,
    )
    sources = torch.as_tensor(
        candidate["source_experts"],
        device=router_scores.device,
        dtype=torch.long,
    )
    destinations = torch.as_tensor(
        candidate["destination_experts"],
        device=router_scores.device,
        dtype=torch.long,
    )
    return float(
        (
            router_scores[tokens, sources]
            - router_scores[tokens, destinations]
        ).mean().item()
    )


def _candidate_chunk_routes(native_ids, native_weights, candidates):
    route_rows = []
    for candidate in candidates:
        baseline = native_ids.clone()
        changed = baseline.clone()
        tokens = torch.as_tensor(
            candidate["tokens"],
            device=native_ids.device,
            dtype=torch.long,
        )
        destinations = torch.as_tensor(
            candidate["destination_experts"],
            device=native_ids.device,
            dtype=torch.long,
        )
        changed[tokens] = destinations
        route_rows.extend((baseline, changed))
    route_ids = torch.stack(route_rows)
    route_weights = native_weights.unsqueeze(0).expand(len(route_rows), -1).clone()
    return route_ids, route_weights


def _probe_cell(
    model,
    clean_latent,
    noise,
    label,
    sigmas,
    start_index,
    block_index,
    num_train_timesteps,
    seed,
    candidate_count,
    candidate_chunk_size,
):
    sigma = float(sigmas[start_index])
    initial_state = analytic_flow_state(clean_latent, noise, sigma)
    moe_layer = model.blocks[block_index].mlp
    native_capture = RouteInputCapture(moe_layer)
    timestep = torch.full(
        (1,),
        sigma * num_train_timesteps,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    try:
        _, hidden_states, native_weights, native_indices = _capture_native_forward(
            model,
            moe_layer,
            native_capture,
            initial_state,
            timestep,
            label,
        )
    finally:
        native_capture.close()
    native_ids = native_indices[0, :, 0]
    native_route_weights = native_weights[0, :, 0]
    router_scores = _all_router_weights(moe_layer, hidden_states, timestep)[0]
    if not torch.equal(router_scores.argmax(dim=-1), native_ids):
        raise RuntimeError("Native routes disagree with reconstructed router scores")

    candidates = _build_short_cycle_arm(
        native_experts=native_ids.detach().cpu().numpy(),
        num_experts=int(moe_layer.num_routed_experts),
        candidate_count=candidate_count,
        cycle_tokens=2,
        seed=_stable_seed(seed, "finite_horizon", block_index, start_index),
        arm="finite_horizon_pair_swap",
    )
    candidates = validate_count_preserving_candidates(
        candidates,
        native_ids.detach().cpu().numpy(),
        int(moe_layer.num_routed_experts),
    )
    for candidate in candidates:
        candidate["mean_router_margin"] = _candidate_router_margin(
            candidate,
            router_scores,
        )

    captures = {
        index: RouteInputCapture(model.blocks[index].mlp)
        for index in BLOCK_INDICES
    }
    reference_batch_size = 2 * candidate_chunk_size
    reference_state = initial_state.expand(
        reference_batch_size, -1, -1, -1, -1
    ).clone()
    reference_labels = label.expand(reference_batch_size).clone()
    reference_ids = native_ids.unsqueeze(0).expand(reference_batch_size, -1).clone()
    reference_weights = native_route_weights.unsqueeze(0).expand(
        reference_batch_size, -1
    ).clone()
    try:
        unforced = _rollout(
            model=model,
            captures=captures,
            initial_state=reference_state,
            labels=reference_labels,
            clean_latent=clean_latent,
            noise=noise,
            sigmas=sigmas,
            start_index=start_index,
            num_train_timesteps=num_train_timesteps,
        )
        unforced_losses = _rollout_losses(
            unforced, clean_latent, noise, sigmas, start_index
        )
        forced_native = _rollout(
            model=model,
            captures=captures,
            initial_state=reference_state,
            labels=reference_labels,
            clean_latent=clean_latent,
            noise=noise,
            sigmas=sigmas,
            start_index=start_index,
            num_train_timesteps=num_train_timesteps,
            forced_block_index=block_index,
            forced_route_ids=reference_ids,
            forced_route_weights=reference_weights,
        )
        forced_native_losses = _rollout_losses(
            forced_native, clean_latent, noise, sigmas, start_index
        )

        records = []
        max_paired_baseline = {
            "immediate_mse": 0.0,
            "first_prediction": 0.0,
            "horizons": {
                str(horizon): {"mse": 0.0, "state": 0.0}
                for horizon in HORIZONS
            },
        }
        max_h1_identity_error = 0.0
        for start in range(0, len(candidates), candidate_chunk_size):
            chunk = candidates[start:start + candidate_chunk_size]
            if len(chunk) != candidate_chunk_size:
                raise ValueError("candidate_count must be divisible by candidate_chunk_size")
            route_ids, route_weights = _candidate_chunk_routes(
                native_ids,
                native_route_weights,
                chunk,
            )
            model_batch = route_ids.shape[0]
            rollout = _rollout(
                model=model,
                captures=captures,
                initial_state=initial_state.expand(
                    model_batch, -1, -1, -1, -1
                ).clone(),
                labels=label.expand(model_batch).clone(),
                clean_latent=clean_latent,
                noise=noise,
                sigmas=sigmas,
                start_index=start_index,
                num_train_timesteps=num_train_timesteps,
                forced_block_index=block_index,
                forced_route_ids=route_ids,
                forced_route_weights=route_weights,
            )
            losses = _rollout_losses(
                rollout, clean_latent, noise, sigmas, start_index
            )
            baseline_rows = torch.arange(
                0, model_batch, 2, device=clean_latent.device
            )
            candidate_rows = baseline_rows + 1
            max_paired_baseline["first_prediction"] = max(
                max_paired_baseline["first_prediction"],
                float(
                    (
                        rollout["first_prediction"][baseline_rows]
                        - unforced["first_prediction"][0:1]
                    ).abs().max().item()
                ),
            )
            max_paired_baseline["immediate_mse"] = max(
                max_paired_baseline["immediate_mse"],
                float(
                    (
                        losses["immediate"][baseline_rows]
                        - unforced_losses["immediate"][0]
                    ).abs().max().item()
                ),
            )
            for horizon in HORIZONS:
                key = str(horizon)
                max_paired_baseline["horizons"][key]["state"] = max(
                    max_paired_baseline["horizons"][key]["state"],
                    float(
                        (
                            rollout["states"][key][baseline_rows]
                            - unforced["states"][key][0:1]
                        ).abs().max().item()
                    ),
                )
                max_paired_baseline["horizons"][key]["mse"] = max(
                    max_paired_baseline["horizons"][key]["mse"],
                    float(
                        (
                            losses[key][baseline_rows]
                            - unforced_losses[key][0]
                        ).abs().max().item()
                    ),
                )

            velocity_changes = (
                losses["immediate"][candidate_rows]
                - losses["immediate"][baseline_rows]
            )
            h1_changes = losses["1"][candidate_rows] - losses["1"][baseline_rows]
            step_size = torch.as_tensor(
                sigmas[start_index + 1],
                device=velocity_changes.device,
                dtype=torch.float32,
            ) - torch.as_tensor(
                sigmas[start_index],
                device=velocity_changes.device,
                dtype=torch.float32,
            )
            identity_error = (h1_changes - (step_size ** 2) * velocity_changes).abs()
            max_h1_identity_error = max(
                max_h1_identity_error,
                float(identity_error.max().item()),
            )
            for pair_index, candidate in enumerate(chunk):
                baseline_row = int(baseline_rows[pair_index].item())
                candidate_row = int(candidate_rows[pair_index].item())
                baseline_immediate = float(losses["immediate"][baseline_row].item())
                if baseline_immediate <= 0:
                    raise RuntimeError("Immediate native MSE must be positive")
                record = {
                    **candidate,
                    "immediate_native_mse": baseline_immediate,
                    "immediate_candidate_mse": float(
                        losses["immediate"][candidate_row].item()
                    ),
                    "immediate_gain": float(-velocity_changes[pair_index].item()),
                    "immediate_gain_relative": float(
                        -velocity_changes[pair_index].item() / baseline_immediate
                    ),
                    "route_divergence": _route_divergence(
                        rollout["routes"], pair_index
                    ),
                }
                for horizon in HORIZONS:
                    key = str(horizon)
                    native_loss = float(losses[key][baseline_row].item())
                    candidate_loss = float(losses[key][candidate_row].item())
                    if native_loss <= 0:
                        raise RuntimeError("Native rollout MSE must be positive")
                    record[f"h{horizon}_native_mse"] = native_loss
                    record[f"h{horizon}_candidate_mse"] = candidate_loss
                    record[f"h{horizon}_gain"] = native_loss - candidate_loss
                    record[f"h{horizon}_gain_relative"] = (
                        (native_loss - candidate_loss) / native_loss
                    )
                records.append(record)
    finally:
        _close_captures(captures)

    controls = {
        "reference_duplicate": _max_duplicate_drift(
            unforced, unforced_losses
        ),
        "forced_native_vs_unforced": _max_rollout_difference(
            forced_native,
            unforced,
            forced_native_losses,
            unforced_losses,
        ),
        "paired_native_vs_reference": max_paired_baseline,
        "max_abs_h1_state_velocity_identity_error": max_h1_identity_error,
        "count_mismatches": int(sum(
            not record["full_count_match"] for record in records
        )),
    }
    summary = summarize_cell_records(
        records,
        numerical_epsilon=NUMERICAL_EPSILON_RELATIVE,
    )
    route_divergence_summary = {
        step_offset: {
            str(index): float(np.mean([
                record["route_divergence"][step_offset][str(index)]
                for record in records
            ]))
            for index in BLOCK_INDICES
        }
        for step_offset in records[0]["route_divergence"]
    }
    return {
        "block_index": int(block_index),
        "start_index": int(start_index),
        "start_sigma": sigma,
        "horizon_sigmas": {
            str(horizon): float(sigmas[start_index + horizon])
            for horizon in HORIZONS
        },
        "candidate_count": len(records),
        "candidate_kind": "two-token expert swap",
        "summary": summary,
        "mean_route_divergence": route_divergence_summary,
        "numerical_controls": controls,
        "candidates": records,
    }


def run_finite_horizon_routing_probe(
    checkpoint_path,
    latent_path,
    label,
    block_indices=BLOCK_INDICES,
    start_indices=START_INDICES,
    horizons=HORIZONS,
    sample_steps=SAMPLE_STEPS,
    sample_shift=SAMPLE_SHIFT,
    candidate_count=CANDIDATE_COUNT,
    candidate_chunk_size=CANDIDATE_CHUNK_SIZE,
    latent_key="latent",
    seed=0,
    device="cpu",
    num_threads=8,
    weights_checkpoint_path=None,
    expected_checkpoint_size=None,
    expected_checkpoint_sha256=None,
    expected_config_sha256=None,
    expected_latent_size=None,
    expected_latent_sha256=None,
):
    checkpoint_path = Path(checkpoint_path).resolve()
    weights_checkpoint_path = Path(weights_checkpoint_path or checkpoint_path).resolve()
    latent_path = Path(latent_path).resolve()
    for path, description in (
        (checkpoint_path, "checkpoint"),
        (weights_checkpoint_path, "weights checkpoint"),
        (latent_path, "latent"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description.title()} does not exist: {path}")
    block_indices = tuple(int(index) for index in block_indices)
    if block_indices != BLOCK_INDICES:
        raise ValueError(f"The locked probe requires blocks {BLOCK_INDICES}")
    if tuple(int(value) for value in horizons) != HORIZONS:
        raise ValueError(f"The locked probe requires horizons {HORIZONS}")
    if int(sample_steps) != SAMPLE_STEPS or float(sample_shift) != SAMPLE_SHIFT:
        raise ValueError("The locked probe requires the 250-step, shift-one sampler")
    if int(candidate_count) != CANDIDATE_COUNT:
        raise ValueError(f"The locked probe requires {CANDIDATE_COUNT} candidates")
    if int(candidate_chunk_size) != CANDIDATE_CHUNK_SIZE:
        raise ValueError(
            f"The locked probe requires chunks of {CANDIDATE_CHUNK_SIZE} candidates"
        )
    if candidate_count % candidate_chunk_size:
        raise ValueError("candidate_count must be divisible by candidate_chunk_size")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")

    sigmas = sampling_sigmas(
        sample_steps,
        sample_shift,
        scheduler_shift=SCHEDULER_SHIFT,
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
    )
    start_indices, _ = validate_schedule_positions(
        sigmas,
        start_indices=start_indices,
        horizons=horizons,
    )
    if start_indices != START_INDICES:
        raise ValueError(f"The locked probe requires starts {START_INDICES}")
    thread_config = _configure_torch_threads(num_threads)
    device = torch.device(device)
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    checkpoint_step = parse_checkpoint_step(checkpoint_path)
    runtime_cfg, _, config_identity = _load_verified_runtime_cfg(
        config_path,
        expected_sha256=expected_config_sha256,
    )
    if not 0 <= int(label) < runtime_cfg.num_classes:
        raise ValueError("label lies outside the ImageNet class range")
    if int(runtime_cfg.sample_steps) != SAMPLE_STEPS:
        raise ValueError("Checkpoint config does not use the locked sample_steps")
    if float(runtime_cfg.sample_shift) != SAMPLE_SHIFT:
        raise ValueError("Checkpoint config does not use the locked sample_shift")
    if float(runtime_cfg.shift) != SCHEDULER_SHIFT:
        raise ValueError("Checkpoint config does not use the locked scheduler shift")
    if int(runtime_cfg.num_train_timesteps) != NUM_TRAIN_TIMESTEPS:
        raise ValueError("Checkpoint config changed num_train_timesteps")
    with _verified_checkpoint_for_loading(
        checkpoint_path,
        weights_checkpoint_path,
        expected_size=expected_checkpoint_size,
        expected_sha256=expected_checkpoint_sha256,
    ) as (weights_handle, checkpoint_identity):
        model, state_name, weights_step, load_seconds = _load_checkpoint_model(
            runtime_cfg,
            weights_handle,
            device,
        )
    if weights_step != checkpoint_step:
        raise ValueError("Weights checkpoint step differs from the canonical checkpoint")
    if state_name != EMA_CHECKPOINT_STATE:
        raise ValueError("The locked probe requires EMA checkpoint weights")
    _validate_moe_block_contract(model, block_indices)

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    with _verified_latent_for_loading(
        latent_path,
        expected_size=expected_latent_size,
        expected_sha256=expected_latent_sha256,
    ) as (latent_handle, latent_identity):
        clean_latent = _load_latent(latent_handle, latent_key, seed, device)
    torch.manual_seed(seed + 1)
    noise = torch.randn_like(clean_latent)
    label_tensor = torch.tensor([label], device=device, dtype=torch.long)

    cells = []
    probe_start = time.perf_counter()
    for block_index in block_indices:
        for start_index in start_indices:
            cells.append(_probe_cell(
                model=model,
                clean_latent=clean_latent,
                noise=noise,
                label=label_tensor,
                sigmas=sigmas,
                start_index=start_index,
                block_index=block_index,
                num_train_timesteps=int(runtime_cfg.num_train_timesteps),
                seed=seed,
                candidate_count=candidate_count,
                candidate_chunk_size=candidate_chunk_size,
            ))
    probe_seconds = time.perf_counter() - probe_start
    result = {
        "finite_horizon_routing_probe_version": PROBE_VERSION,
        "diagnostic_scope": (
            "frozen-checkpoint quota-preserving one-step expert-assignment "
            "intervention followed by native-router finite rollout; not a "
            "training, FID, or single-token causal claim"
        ),
        "hypothesis": (
            "The expert assignment that looks best at the intervention step "
            "is often not the assignment that gives the lowest state error "
            "after eight native-routing denoising steps."
        ),
        "falsification_rule": (
            "Reject this research direction when immediate and horizon-eight "
            "assignment rankings remain strongly aligned on the sealed "
            "confirmatory images, layers, and noise levels."
        ),
        "estimand": (
            "Utility of a two-token expert swap that preserves the complete "
            "expert-count vector at the intervention block and step."
        ),
        "checkpoint": str(checkpoint_path),
        "weights_checkpoint": str(weights_checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_state": state_name,
        "checkpoint_identity": checkpoint_identity,
        "config": str(config_path),
        "config_identity": config_identity,
        "model_name": runtime_cfg.model_name,
        "latent": str(latent_path),
        "latent_key": latent_key,
        "latent_identity": latent_identity,
        "label": int(label),
        "seed": int(seed),
        "device": str(device),
        "num_threads": int(num_threads),
        "thread_config": thread_config,
        "model_load_seconds": load_seconds,
        "probe_seconds": probe_seconds,
        "sample_steps": SAMPLE_STEPS,
        "sample_shift": SAMPLE_SHIFT,
        "scheduler_shift": SCHEDULER_SHIFT,
        "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
        "start_indices": list(START_INDICES),
        "start_sigmas": [float(sigmas[index]) for index in START_INDICES],
        "horizons": list(HORIZONS),
        "block_indices": list(BLOCK_INDICES),
        "candidate_count": CANDIDATE_COUNT,
        "candidate_chunk_size": CANDIDATE_CHUNK_SIZE,
        "route_weight_semantics": (
            "Every swapped token keeps its own native top-1 route weight; "
            "only expert identity changes."
        ),
        "numerical_epsilon_relative": NUMERICAL_EPSILON_RELATIVE,
        "cells": cells,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
