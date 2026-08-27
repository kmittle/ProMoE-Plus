"""Forward-only scorer utilities for within-expert FFN-pass exchange."""

from __future__ import annotations

import hashlib
import math
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from analyses.denoising_regret.probe import (
    RoutingProbeCapture,
    _all_router_weights,
    _compute_router,
    _evaluate_experts,
    _extract_prediction,
    _load_latent,
    _per_sample_mse,
)
from analyses.timestep_utility.compute_exchange_probe import (
    EXCHANGE_QUOTA,
    _exact_candidate_changes,
    _exchange_components,
    _pair_concordance,
    _stable_seed,
    _validate_candidate,
    build_same_expert_exchange_candidates,
)


DEPLOYABILITY_VERSION = 1
MOE_BLOCKS = (1, 3, 5, 7, 9, 11)
RETROSPECTIVE_BLOCKS = (1, 5, 11)
SIGMAS = (0.2, 0.5, 0.8)
SCORER_KINDS = ("primary", "router_context", "rolled_correspondence")
HEAD_NAMES = ("donor", "receiver")
PAIRWISE_LOSS_WEIGHT = 0.25
MAX_EPOCHS = 60
MIN_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 8
TRAIN_BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MODEL_SEED = 2026082641
VALIDATION_SALT = "promoe-compute-exchange-deployability-validation-v1"
ROLLED_SALT = "promoe-compute-exchange-deployability-rolled-v1"


def _canonical_seed(*parts):
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) % (2 ** 63)


def array_sha256(array, dtype):
    canonical = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


class ForwardInputCapture:
    """Capture inference-visible MoE inputs without changing the forward pass."""

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


def _validate_source_candidates(source_cell, native_experts, candidate_seed):
    candidates = build_same_expert_exchange_candidates(
        native_experts.detach().cpu().numpy(),
        len(source_cell["records"][0]["native_pass_vector"]),
        int(candidate_seed),
    )
    records = source_cell["records"]
    if len(candidates) != len(records):
        raise RuntimeError("Source candidate count changed")
    keys = (
        "id",
        "donors",
        "receivers",
        "experts",
        "quota_by_expert",
        "transferred_passes",
        "native_pass_vector",
        "candidate_pass_vector",
    )
    for candidate, record in zip(candidates, records):
        for key in keys:
            if candidate[key] != record[key]:
                raise RuntimeError(
                    f"Replayed candidate {candidate['id']} differs at {key}"
                )
    return candidates


def _source_cells_by_key(source_result):
    cells = {}
    for cell in source_result["cells"]:
        key = (int(cell["block_index"]), float(cell["sigma"]))
        if key in cells:
            raise ValueError("Source result contains a duplicate block/sigma cell")
        if "records" not in cell:
            raise ValueError("Deployability replay requires efficacy records")
        cells[key] = cell
    return cells


def _native_route_state(moe_layer, hidden_states, labels, timestep=None):
    with torch.no_grad():
        weights, indices, auxiliary_loss = _compute_router(
            moe_layer,
            hidden_states,
            labels,
            timestep,
        )
        scores = _all_router_weights(moe_layer, hidden_states, timestep)
    if auxiliary_loss is not None:
        raise RuntimeError("Frozen eval router unexpectedly returned an auxiliary loss")
    native_experts = indices[0, :, 0]
    native_weights = weights[0, :, 0]
    if native_experts.max() >= int(moe_layer.num_routed_experts):
        raise RuntimeError("Conditional replay selected the unconditional expert")
    if not torch.equal(scores[0].argmax(dim=-1), native_experts):
        raise RuntimeError("Native routes disagree with all-router scores")
    return native_experts, native_weights, scores[0]


def _noised_state(clean_latent, noise, sigma, num_train_timesteps):
    sigma_tensor = torch.tensor(
        float(sigma),
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    noised_latent = (1.0 - sigma_tensor) * clean_latent + sigma_tensor * noise
    timestep = torch.full(
        (1,),
        float(sigma) * int(num_train_timesteps),
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    return noised_latent, timestep


def _calibration_cell(
    model,
    runtime_cfg,
    moe_layer,
    capture,
    clean_latent,
    noise,
    label,
    block_index,
    sigma,
):
    noised_latent, timestep = _noised_state(
        clean_latent,
        noise,
        sigma,
        runtime_cfg.num_train_timesteps,
    )
    target = (noise - clean_latent).squeeze(2)
    capture.start()
    try:
        model_output = model(noised_latent, timestep, context=label)
        prediction = _extract_prediction(model_output, target.shape[1])
        native_loss = _per_sample_mse(prediction, target).mean()
        if capture.moe_output is None:
            raise RuntimeError("Calibration did not capture the MoE output")
        moe_gradient, = torch.autograd.grad(native_loss, capture.moe_output)
    finally:
        capture.stop()
    hidden_states = capture.hidden_states
    labels = capture.labels
    if hidden_states is None or labels is None:
        raise RuntimeError("Calibration did not capture the router inputs")
    native_experts, native_weights, router_scores = _native_route_state(
        moe_layer,
        hidden_states,
        labels,
        timestep,
    )
    components = _exchange_components(
        moe_layer=moe_layer,
        hidden_states=hidden_states[0],
        moe_gradient=moe_gradient.detach()[0],
        native_experts=native_experts,
        native_weights=native_weights,
    )
    native_mse = float(native_loss.item())
    return {
        "hidden": hidden_states[0].detach(),
        "router_scores": router_scores.detach(),
        "native_experts": native_experts.detach(),
        "native_weights": native_weights.detach(),
        "donor_target": components["donor_change"].detach() / native_mse,
        "receiver_target": components["receiver_change"].detach() / native_mse,
        "native_mse": native_mse,
        "timestep": float(timestep.item()),
        "gradient_enabled": True,
        "target_constructed": True,
        "block_index": int(block_index),
        "sigma": float(sigma),
    }


def _forward_only_cell(
    model,
    runtime_cfg,
    moe_layer,
    capture,
    clean_latent,
    noise,
    label,
    block_index,
    sigma,
):
    noised_latent, timestep = _noised_state(
        clean_latent,
        noise,
        sigma,
        runtime_cfg.num_train_timesteps,
    )
    capture.start()
    try:
        with torch.inference_mode():
            model(noised_latent, timestep, context=label)
    finally:
        capture.stop()
    hidden_states = capture.hidden_states
    labels = capture.labels
    if hidden_states is None or labels is None:
        raise RuntimeError("Forward-only replay did not capture router inputs")
    native_experts, native_weights, router_scores = _native_route_state(
        moe_layer,
        hidden_states,
        labels,
        timestep,
    )
    return {
        "hidden": hidden_states[0].detach(),
        "router_scores": router_scores.detach(),
        "native_experts": native_experts.detach(),
        "native_weights": native_weights.detach(),
        "timestep": float(timestep.item()),
        "gradient_enabled": False,
        "target_constructed": False,
        "block_index": int(block_index),
        "sigma": float(sigma),
    }


def extract_deployability_case(
    model,
    runtime_cfg,
    case,
    latent_root,
    source_result,
    split,
    latent_key="latent",
):
    if split not in {"calibration", "retrospective"}:
        raise ValueError(f"Unknown deployability split: {split}")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Deployability extraction requires a frozen eval model")
    latent_path = Path(latent_root) / case["latent_relative"]
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent does not exist: {latent_path}")
    blocks = MOE_BLOCKS if split == "calibration" else RETROSPECTIVE_BLOCKS
    if split == "calibration":
        if source_result is None:
            raise ValueError("Calibration extraction requires its sealed source result")
        source_cells = _source_cells_by_key(source_result)
        expected_source_keys = {(5, sigma) for sigma in SIGMAS}
        if set(source_cells) != expected_source_keys:
            raise ValueError("Calibration source cells do not match the replay split")
    else:
        if source_result is not None:
            raise ValueError("Retrospective extraction must not receive source results")
        source_cells = {}

    seed = int(case["seed"])
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    clean_latent = _load_latent(latent_path, latent_key, seed, device)
    torch.manual_seed(seed + 1)
    noise = torch.randn_like(clean_latent)
    label = torch.tensor([int(case["label"])], device=device, dtype=torch.long)

    arrays = {
        "hidden": [],
        "router_scores": [],
        "native_experts": [],
        "native_weights": [],
    }
    if split == "calibration":
        arrays["donor_target"] = []
        arrays["receiver_target"] = []
    metadata = []
    for block_index in blocks:
        moe_layer = model.blocks[block_index].mlp
        capture = (
            RoutingProbeCapture(moe_layer)
            if split == "calibration"
            else ForwardInputCapture(moe_layer)
        )
        try:
            for sigma in SIGMAS:
                if split == "calibration":
                    cell = _calibration_cell(
                        model,
                        runtime_cfg,
                        moe_layer,
                        capture,
                        clean_latent,
                        noise,
                        label,
                        block_index,
                        sigma,
                    )
                else:
                    cell = _forward_only_cell(
                        model,
                        runtime_cfg,
                        moe_layer,
                        capture,
                        clean_latent,
                        noise,
                        label,
                        block_index,
                        sigma,
                    )
                candidate_seed = _stable_seed(
                    seed,
                    "compute-exchange",
                    block_index,
                    f"{sigma:.17g}",
                )
                source_cell = source_cells.get((block_index, sigma))
                if source_cell is not None:
                    if int(source_cell["candidate_seed"]) != candidate_seed:
                        raise RuntimeError("Replayed candidate seed differs from source")
                    candidates = _validate_source_candidates(
                        source_cell,
                        cell["native_experts"],
                        candidate_seed,
                    )
                    if split == "calibration":
                        donor = cell["donor_target"].detach().cpu().numpy()
                        receiver = cell["receiver_target"].detach().cpu().numpy()
                        for candidate, record in zip(candidates, source_cell["records"]):
                            replayed = float(
                                donor[np.asarray(candidate["donors"], dtype=np.int64)].sum()
                                + receiver[
                                    np.asarray(candidate["receivers"], dtype=np.int64)
                                ].sum()
                            )
                            recorded = float(record["first_order_change"]) / float(
                                source_cell["native_mse"]
                            )
                            if not math.isclose(replayed, recorded, rel_tol=2e-5, abs_tol=2e-7):
                                raise RuntimeError(
                                    "Replayed first-order label differs from sealed source"
                                )
                for key in arrays:
                    arrays[key].append(cell[key].detach().cpu().numpy())
                cell_metadata = {
                    "block_index": int(block_index),
                    "sigma": float(sigma),
                    "timestep": float(cell["timestep"]),
                    "candidate_seed": int(candidate_seed),
                    "source_cell_available": bool(source_cell is not None),
                    "gradient_enabled": bool(cell["gradient_enabled"]),
                    "target_constructed": bool(cell["target_constructed"]),
                }
                if split == "calibration":
                    cell_metadata["native_mse"] = (
                        float(source_cell["native_mse"])
                        if source_cell is not None else float(cell["native_mse"])
                    )
                metadata.append(cell_metadata)
        finally:
            capture.close()

    output_arrays = {
        "hidden": np.stack(arrays["hidden"]).astype(np.float32),
        "router_scores": np.stack(arrays["router_scores"]).astype(np.float32),
        "native_experts": np.stack(arrays["native_experts"]).astype(np.int16),
        "native_weights": np.stack(arrays["native_weights"]).astype(np.float32),
    }
    if split == "calibration":
        output_arrays.update({
            "donor_target": np.stack(arrays["donor_target"]).astype(np.float32),
            "receiver_target": np.stack(arrays["receiver_target"]).astype(np.float32),
        })
    return output_arrays, {
        "deployability_version": DEPLOYABILITY_VERSION,
        "split": split,
        "case_id": case["id"],
        "label": int(case["label"]),
        "seed": seed,
        "latent_relative": case["latent_relative"],
        "feature_contract": (
            "pre-pass hidden state, native all-expert router scores/ID/weight, "
            "block, sigma, and spatial position only"
        ),
        "privileged_targets_present": split == "calibration",
        "cells": metadata,
    }


def _reveal_action_cell(
    model,
    runtime_cfg,
    moe_layer,
    capture,
    clean_latent,
    noise,
    label,
    action_cell,
):
    block_index = int(action_cell["block_index"])
    sigma = float(action_cell["sigma"])
    noised_latent, timestep = _noised_state(
        clean_latent,
        noise,
        sigma,
        runtime_cfg.num_train_timesteps,
    )
    target = (noise - clean_latent).squeeze(2)
    capture.start()
    try:
        with torch.inference_mode():
            model_output = model(noised_latent, timestep, context=label)
        prediction = _extract_prediction(model_output, target.shape[1])
        native_loss = _per_sample_mse(prediction, target).mean()
    finally:
        capture.stop()
    if capture.hidden_states is None or capture.labels is None:
        raise RuntimeError("Reveal did not capture the native router inputs")
    if not torch.isfinite(native_loss) or float(native_loss.item()) <= 0:
        raise RuntimeError("Reveal native MSE must be finite and positive")
    native_experts, native_weights, _ = _native_route_state(
        moe_layer,
        capture.hidden_states,
        capture.labels,
        timestep,
    )
    route_id_sha256 = array_sha256(
        native_experts.detach().cpu().numpy(),
        np.int64,
    )
    route_weight_sha256 = array_sha256(
        native_weights.detach().float().cpu().numpy(),
        np.float32,
    )
    if route_id_sha256 != action_cell["route_id_sha256"]:
        raise RuntimeError("Reveal route IDs differ from the sealed action state")
    if route_weight_sha256 != action_cell["route_weight_sha256"]:
        raise RuntimeError("Reveal route weights differ from the sealed action state")

    names = tuple(action_cell["actions"])
    candidates = []
    for name in names:
        candidate = dict(action_cell["actions"][name])
        if candidate["id"] != f"exact:{name}":
            raise RuntimeError("Sealed action ID is inconsistent")
        _validate_candidate(
            candidate,
            native_experts.detach().cpu().numpy(),
            int(moe_layer.num_routed_experts),
        )
        candidates.append(candidate)
    exact_records, numerical_controls = _exact_candidate_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        native_route_ids=native_experts,
        native_route_weights=native_weights,
        native_prediction=prediction.detach(),
        native_loss=native_loss.detach(),
        candidates=candidates,
    )
    exact_by_id = {record["id"]: record for record in exact_records}
    if len(exact_by_id) != len(candidates):
        raise RuntimeError("Reveal did not evaluate every sealed action exactly once")
    action_results = {}
    for name in names:
        record = exact_by_id[f"exact:{name}"]
        action_results[name] = {
            "exact_mse_change": float(record["exact_mse_change"]),
            "selected_gain": float(-record["exact_mse_change"] / native_loss.item()),
            "max_abs_output_change": float(record["max_abs_output_change"]),
            "runtime_second_pass_shapes": record["runtime_second_pass_shapes"],
        }
    return {
        "block_index": block_index,
        "sigma": sigma,
        "timestep": float(timestep.item()),
        "native_mse": float(native_loss.item()),
        "route_id_sha256": route_id_sha256,
        "route_weight_sha256": route_weight_sha256,
        "action_results": action_results,
        "numerical_controls": numerical_controls,
    }


def reveal_deployability_case(
    model,
    runtime_cfg,
    case,
    latent_root,
    action_case,
    latent_key="latent",
):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Deployability reveal requires a frozen eval model")
    if action_case["case_id"] != case["id"]:
        raise ValueError("Reveal case differs from its sealed action case")
    latent_path = Path(latent_root) / case["latent_relative"]
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent does not exist: {latent_path}")
    action_cells = {
        (int(cell["block_index"]), float(cell["sigma"])): cell
        for cell in action_case["cells"]
    }
    expected = {
        (block, sigma) for block in RETROSPECTIVE_BLOCKS for sigma in SIGMAS
    }
    if set(action_cells) != expected or len(action_case["cells"]) != len(expected):
        raise ValueError("Sealed action case has an incomplete block/sigma grid")

    seed = int(case["seed"])
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    clean_latent = _load_latent(latent_path, latent_key, seed, device)
    torch.manual_seed(seed + 1)
    noise = torch.randn_like(clean_latent)
    label = torch.tensor([int(case["label"])], device=device, dtype=torch.long)
    cells = []
    for block_index in RETROSPECTIVE_BLOCKS:
        moe_layer = model.blocks[block_index].mlp
        capture = ForwardInputCapture(moe_layer)
        try:
            for sigma in SIGMAS:
                cells.append(_reveal_action_cell(
                    model,
                    runtime_cfg,
                    moe_layer,
                    capture,
                    clean_latent,
                    noise,
                    label,
                    action_cells[(block_index, sigma)],
                ))
        finally:
            capture.close()
    return {
        "deployability_version": DEPLOYABILITY_VERSION,
        "case_id": case["id"],
        "label": int(case["label"]),
        "seed": seed,
        "latent_relative": case["latent_relative"],
        "cells": cells,
    }


def write_npz_atomic(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _spatial_features(sequence_length, device, dtype):
    side = math.isqrt(int(sequence_length))
    if side * side != int(sequence_length):
        raise ValueError("Deployability scorer requires a square token grid")
    axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    x = x.reshape(-1)
    y = y.reshape(-1)
    return torch.stack((x, y, x.square(), y.square(), x * y, x.square() + y.square()), dim=-1)


def build_scorer_features(
    hidden,
    router_scores,
    native_experts,
    sigmas,
    token_indices,
    sequence_length,
    include_hidden,
):
    if hidden.ndim != 2 or router_scores.ndim != 2:
        raise ValueError("Hidden state and router scores must be token matrices")
    if hidden.shape[0] != router_scores.shape[0]:
        raise ValueError("Hidden state and router scores must align by token")
    if native_experts.shape != (hidden.shape[0],) or native_experts.dtype != torch.long:
        raise ValueError("Native expert IDs must align with scorer tokens")
    if sigmas.shape != native_experts.shape or token_indices.shape != native_experts.shape:
        raise ValueError("Sigma and spatial indices must align with scorer tokens")
    if native_experts.min() < 0 or native_experts.max() >= router_scores.shape[1]:
        raise ValueError("Native expert IDs lie outside router scores")
    hidden = hidden.float()
    router_scores = router_scores.float()
    top_two = torch.topk(router_scores, k=2, dim=-1).values
    native_weight = router_scores.gather(1, native_experts[:, None])
    margin = top_two[:, :1] - top_two[:, 1:2]
    probabilities = F.softmax(router_scores, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
        dim=-1,
        keepdim=True,
    ) / math.log(router_scores.shape[1])
    if token_indices.min() < 0 or token_indices.max() >= int(sequence_length):
        raise ValueError("Token index lies outside the spatial grid")
    position = _spatial_features(
        sequence_length,
        hidden.device,
        hidden.dtype,
    )[token_indices]
    sigma_value = sigmas.float().unsqueeze(1)
    sigma_features = torch.cat((
        sigma_value,
        sigma_value.square(),
        torch.log((1.0 - sigma_value).clamp_min(1e-4) / sigma_value.clamp_min(1e-4)),
        torch.sin(math.pi * sigma_value),
        torch.cos(math.pi * sigma_value),
    ), dim=-1)
    parts = [router_scores, native_weight, margin, entropy, position, sigma_features]
    if include_hidden:
        normalized_hidden = F.layer_norm(hidden, (hidden.shape[-1],))
        hidden_rms = hidden.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8).log()
        parts = [normalized_hidden, hidden_rms, *parts]
    features = torch.cat(parts, dim=-1)
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("Scorer features contain non-finite values")
    return features


class DualLinearUtilityScorer(torch.nn.Module):
    """Block/expert-specific forward-only linear donor and receiver heads."""

    def __init__(self, hidden_dim, num_experts, blocks=MOE_BLOCKS, include_hidden=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)
        self.blocks = tuple(int(block) for block in blocks)
        self.include_hidden = bool(include_hidden)
        if not self.blocks or len(self.blocks) != len(set(self.blocks)):
            raise ValueError("Scorer blocks must be nonempty and unique")
        if self.hidden_dim <= 0 or self.num_experts < 2:
            raise ValueError("Scorer dimensions are invalid")
        feature_dim = self.num_experts + 3 + 6 + 5
        if self.include_hidden:
            feature_dim += self.hidden_dim + 1
        self.feature_dim = feature_dim
        self.weight = torch.nn.Parameter(torch.empty(
            len(self.blocks),
            self.num_experts,
            self.feature_dim,
            len(HEAD_NAMES),
        ))
        self.bias = torch.nn.Parameter(torch.zeros(
            len(self.blocks),
            self.num_experts,
            len(HEAD_NAMES),
        ))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.002)
        lookup = torch.full((max(self.blocks) + 1,), -1, dtype=torch.long)
        for slot, block in enumerate(self.blocks):
            lookup[block] = slot
        self.register_buffer("block_lookup", lookup, persistent=True)

    def forward(
        self,
        hidden,
        router_scores,
        native_experts,
        block_indices,
        sigmas,
        token_indices,
        sequence_length,
    ):
        if (
            block_indices.shape != native_experts.shape
            or sigmas.shape != native_experts.shape
            or token_indices.shape != native_experts.shape
        ):
            raise ValueError("Scorer context vectors must align with tokens")
        if block_indices.min() < 0 or block_indices.max() >= self.block_lookup.numel():
            raise ValueError("Scorer received an unknown block index")
        block_slots = self.block_lookup[block_indices]
        if (block_slots < 0).any():
            raise ValueError("Scorer received a non-MoE block")
        features = build_scorer_features(
            hidden,
            router_scores,
            native_experts,
            sigmas,
            token_indices,
            sequence_length,
            self.include_hidden,
        )
        weights = self.weight[block_slots, native_experts]
        biases = self.bias[block_slots, native_experts]
        return torch.einsum("nf,nfh->nh", features, weights) + biases


def normalize_counterfactual_targets(targets, native_experts, cell_ids):
    targets = np.asarray(targets, dtype=np.float64)
    native_experts = np.asarray(native_experts, dtype=np.int64)
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    if targets.ndim != 2 or targets.shape[1] != len(HEAD_NAMES):
        raise ValueError("Counterfactual targets must have donor/receiver columns")
    if native_experts.shape != (len(targets),) or cell_ids.shape != (len(targets),):
        raise ValueError("Target metadata must align with tokens")
    normalized = np.empty_like(targets)
    for cell_id in np.unique(cell_ids):
        cell = cell_ids == cell_id
        centered_parts = []
        for expert in np.unique(native_experts[cell]):
            group = cell & (native_experts == expert)
            centered = targets[group] - targets[group].mean(axis=0, keepdims=True)
            normalized[group] = centered
            centered_parts.append(centered.reshape(-1))
        scale = np.concatenate(centered_parts).std()
        if not np.isfinite(scale) or scale < 1e-10:
            scale = 1.0
        normalized[cell] /= scale
    if not np.isfinite(normalized).all():
        raise RuntimeError("Normalized targets contain non-finite values")
    return normalized.astype(np.float32)


def roll_counterfactual_correspondence(targets, native_experts, cell_ids, seed=MODEL_SEED):
    targets = np.asarray(targets)
    native_experts = np.asarray(native_experts)
    cell_ids = np.asarray(cell_ids)
    rolled = np.empty_like(targets)
    for cell_id in np.unique(cell_ids):
        for expert in np.unique(native_experts[cell_ids == cell_id]):
            positions = np.flatnonzero(
                (cell_ids == cell_id) & (native_experts == expert)
            )
            if positions.size < 2:
                rolled[positions] = targets[positions]
                continue
            for head in range(targets.shape[1]):
                offset = 1 + _canonical_seed(
                    ROLLED_SALT,
                    seed,
                    int(cell_id),
                    int(expert),
                    head,
                ) % (positions.size - 1)
                rolled[positions, head] = targets[np.roll(positions, int(offset)), head]
    return rolled


def pair_indices(native_experts, cell_ids, seed=MODEL_SEED):
    native_experts = np.asarray(native_experts)
    cell_ids = np.asarray(cell_ids)
    partners = np.arange(len(native_experts), dtype=np.int64)
    for cell_id in np.unique(cell_ids):
        for expert in np.unique(native_experts[cell_ids == cell_id]):
            positions = np.flatnonzero(
                (cell_ids == cell_id) & (native_experts == expert)
            )
            if positions.size < 2:
                continue
            offset = 1 + _canonical_seed(
                "pair",
                seed,
                int(cell_id),
                int(expert),
            ) % (positions.size - 1)
            partners[positions] = np.roll(positions, int(offset))
    return partners


def solve_exact_exchange(
    native_experts,
    token_scores,
    quota=EXCHANGE_QUOTA,
    num_experts=None,
):
    native_experts = np.asarray(native_experts, dtype=np.int64)
    token_scores = np.asarray(token_scores, dtype=np.float64)
    if token_scores.shape != (native_experts.size, len(HEAD_NAMES)):
        raise ValueError("Token scores must contain aligned donor/receiver columns")
    if not np.isfinite(token_scores).all():
        raise ValueError("Token scores must be finite")
    num_experts = int(num_experts) if num_experts is not None else int(native_experts.max()) + 1
    if num_experts <= int(native_experts.max()):
        raise ValueError("num_experts does not cover native assignments")
    donors = []
    receivers = []
    experts = []
    quota_by_expert = [0] * num_experts
    for expert in range(num_experts):
        positions = np.flatnonzero(native_experts == expert)
        count = min(int(np.floor(float(quota) * positions.size + 0.5)), positions.size // 2)
        if count == 0:
            continue
        costs = np.zeros((positions.size, positions.size), dtype=np.float64)
        costs[:, :count] = token_scores[positions, 0, None]
        costs[:, count:2 * count] = token_scores[positions, 1, None]
        rows, columns = linear_sum_assignment(costs)
        donor_tokens = positions[rows[columns < count]]
        receiver_tokens = positions[rows[(columns >= count) & (columns < 2 * count)]]
        if donor_tokens.size != count or receiver_tokens.size != count:
            raise RuntimeError("Three-state assignment did not fill its exact quotas")
        donors.extend(int(token) for token in donor_tokens)
        receivers.extend(int(token) for token in receiver_tokens)
        experts.extend([expert] * count)
        quota_by_expert[expert] = count
    native_counts = np.bincount(native_experts, minlength=num_experts).tolist()
    candidate = {
        "id": "learned-exact-exchange",
        "donors": donors,
        "receivers": receivers,
        "experts": experts,
        "quota": float(quota),
        "quota_by_expert": quota_by_expert,
        "transferred_passes": len(donors),
        "native_pass_vector": native_counts,
        "candidate_pass_vector": native_counts,
    }
    _validate_candidate(candidate, native_experts, num_experts)
    return candidate


def candidate_scores(candidates, token_scores):
    token_scores = np.asarray(token_scores, dtype=np.float64)
    values = []
    for candidate in candidates:
        donors = np.asarray(candidate["donors"], dtype=np.int64)
        receivers = np.asarray(candidate["receivers"], dtype=np.int64)
        values.append(float(
            token_scores[donors, 0].sum() + token_scores[receivers, 1].sum()
        ))
    return np.asarray(values, dtype=np.float64)


def candidate_concordance(candidates, predicted_scores, target_scores):
    predicted = candidate_scores(candidates, predicted_scores)
    target = candidate_scores(candidates, target_scores)
    return _pair_concordance(-predicted, -target)


@contextmanager
def scorer_inference(model):
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        model.train(was_training)
