"""Calibration and inference for the forward-only compute-exchange scorer."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyses.timestep_utility.compute_exchange_deployability import (
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    MODEL_SEED,
    MOE_BLOCKS,
    PAIRWISE_LOSS_WEIGHT,
    SCORER_KINDS,
    TRAIN_BATCH_SIZE,
    VALIDATION_SALT,
    WEIGHT_DECAY,
    DualLinearUtilityScorer,
    candidate_concordance,
    normalize_counterfactual_targets,
    pair_indices,
    roll_counterfactual_correspondence,
    build_same_expert_exchange_candidates,
)


@dataclass(frozen=True)
class FeatureDataset:
    hidden: np.ndarray
    router_scores: np.ndarray
    native_experts: np.ndarray
    native_weights: np.ndarray
    block_indices: np.ndarray
    sigmas: np.ndarray
    token_indices: np.ndarray
    case_indices: np.ndarray
    cell_ids: np.ndarray
    case_ids: tuple[str, ...]
    cells: tuple[dict, ...]
    targets: np.ndarray | None
    sequence_length: int
    hidden_dim: int
    num_experts: int

    def indices_for_cases(self, selected_case_ids):
        selected = set(selected_case_ids)
        unknown = selected - set(self.case_ids)
        if unknown:
            raise ValueError(f"Unknown feature cases: {sorted(unknown)}")
        selected_indices = {
            index for index, case_id in enumerate(self.case_ids) if case_id in selected
        }
        return np.flatnonzero(np.isin(
            self.case_indices,
            np.asarray(sorted(selected_indices), dtype=np.int64),
        )).astype(np.int64)


def _load_case(npz_path, metadata_path, require_targets):
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as payload:
        keys = set(payload.files)
        required = {"hidden", "router_scores", "native_experts", "native_weights"}
        target_keys = {"donor_target", "receiver_target"}
        if not required.issubset(keys):
            raise ValueError(f"Feature payload lacks keys: {sorted(required - keys)}")
        if require_targets and not target_keys.issubset(keys):
            raise ValueError("Calibration payload lacks privileged targets")
        if not require_targets and keys & target_keys:
            raise ValueError("Forward-only payload contains privileged targets")
        arrays = {key: np.array(payload[key], copy=True) for key in payload.files}
    hidden = arrays["hidden"]
    router_scores = arrays["router_scores"]
    native_experts = arrays["native_experts"]
    if hidden.ndim != 3 or router_scores.ndim != 3 or native_experts.ndim != 2:
        raise ValueError("Feature payload arrays have invalid ranks")
    if hidden.shape[:2] != router_scores.shape[:2] or hidden.shape[:2] != native_experts.shape:
        raise ValueError("Feature payload arrays do not align")
    if len(metadata["cells"]) != hidden.shape[0]:
        raise ValueError("Feature metadata cell count differs from arrays")
    if metadata["privileged_targets_present"] != bool(require_targets):
        raise ValueError("Feature metadata target contract differs from split")
    if not require_targets:
        forbidden = {"source_result", "source_result_sha256", "native_mse"}
        if forbidden & set(metadata):
            raise ValueError("Forward-only metadata contains privileged source state")
        if any("native_mse" in cell for cell in metadata["cells"]):
            raise ValueError("Forward-only cell metadata contains target-derived MSE")
    if require_targets:
        if arrays["donor_target"].shape != native_experts.shape:
            raise ValueError("Donor targets do not align with tokens")
        if arrays["receiver_target"].shape != native_experts.shape:
            raise ValueError("Receiver targets do not align with tokens")
    return arrays, metadata


def load_feature_dataset(case_files, require_targets):
    if not case_files:
        raise ValueError("At least one feature case is required")
    hidden_parts = []
    router_parts = []
    expert_parts = []
    weight_parts = []
    block_parts = []
    sigma_parts = []
    token_parts = []
    case_parts = []
    cell_parts = []
    target_parts = []
    case_ids = []
    cells = []
    cell_offset = 0
    token_offset = 0
    sequence_length = None
    hidden_dim = None
    num_experts = None
    for case_index, case_file in enumerate(case_files):
        arrays, metadata = _load_case(
            case_file["npz"],
            case_file["metadata"],
            require_targets,
        )
        if metadata["case_id"] != case_file["case_id"]:
            raise ValueError("Feature case ID differs from manifest")
        hidden = arrays["hidden"]
        router = arrays["router_scores"]
        native = arrays["native_experts"].astype(np.int64)
        native_weights = arrays["native_weights"].astype(np.float32)
        num_cells, tokens, dim = hidden.shape
        if sequence_length is None:
            sequence_length = tokens
            hidden_dim = dim
            num_experts = router.shape[-1]
        if (tokens, dim, router.shape[-1]) != (
            sequence_length,
            hidden_dim,
            num_experts,
        ):
            raise ValueError("Feature case dimensions differ")
        hidden_parts.append(hidden.reshape(-1, dim))
        router_parts.append(router.reshape(-1, num_experts))
        expert_parts.append(native.reshape(-1))
        if native_weights.shape != native.shape:
            raise ValueError("Native route weights do not align with tokens")
        weight_parts.append(native_weights.reshape(-1))
        blocks = np.asarray(
            [cell["block_index"] for cell in metadata["cells"]],
            dtype=np.int64,
        )
        sigmas = np.asarray(
            [cell["sigma"] for cell in metadata["cells"]],
            dtype=np.float32,
        )
        block_parts.append(np.repeat(blocks, tokens))
        sigma_parts.append(np.repeat(sigmas, tokens))
        token_parts.append(np.tile(np.arange(tokens, dtype=np.int64), num_cells))
        case_parts.append(np.full(num_cells * tokens, case_index, dtype=np.int64))
        local_cells = np.repeat(np.arange(num_cells, dtype=np.int64), tokens)
        cell_parts.append(local_cells + cell_offset)
        if require_targets:
            target_parts.append(np.stack((
                arrays["donor_target"].reshape(-1),
                arrays["receiver_target"].reshape(-1),
            ), axis=1))
        for local_index, cell in enumerate(metadata["cells"]):
            start = token_offset + local_index * tokens
            cells.append({
                **cell,
                "case_id": metadata["case_id"],
                "case_index": case_index,
                "cell_id": cell_offset + local_index,
                "token_start": start,
                "token_stop": start + tokens,
                "source_result": case_file.get("source_result"),
            })
        token_offset += num_cells * tokens
        cell_offset += num_cells
        case_ids.append(metadata["case_id"])
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Feature dataset contains duplicate case IDs")
    native_experts = np.concatenate(expert_parts)
    cell_ids = np.concatenate(cell_parts)
    raw_targets = np.concatenate(target_parts) if require_targets else None
    targets = (
        normalize_counterfactual_targets(raw_targets, native_experts, cell_ids)
        if require_targets else None
    )
    return FeatureDataset(
        hidden=np.concatenate(hidden_parts),
        router_scores=np.concatenate(router_parts),
        native_experts=native_experts,
        native_weights=np.concatenate(weight_parts),
        block_indices=np.concatenate(block_parts),
        sigmas=np.concatenate(sigma_parts),
        token_indices=np.concatenate(token_parts),
        case_indices=np.concatenate(case_parts),
        cell_ids=cell_ids,
        case_ids=tuple(case_ids),
        cells=tuple(cells),
        targets=targets,
        sequence_length=int(sequence_length),
        hidden_dim=int(hidden_dim),
        num_experts=int(num_experts),
    )


def split_calibration_cases(case_ids, fit_count=18):
    if len(case_ids) != 24 or len(set(case_ids)) != 24:
        raise ValueError("The locked calibration split requires 24 unique cases")
    ranked = sorted(
        case_ids,
        key=lambda case_id: hashlib.sha256(
            f"{VALIDATION_SALT}|{case_id}".encode()
        ).hexdigest(),
    )
    return tuple(ranked[:fit_count]), tuple(ranked[fit_count:])


def _model_batch(model, dataset, indices, device):
    indices = np.asarray(indices, dtype=np.int64)
    return model(
        torch.as_tensor(dataset.hidden[indices], device=device),
        torch.as_tensor(dataset.router_scores[indices], device=device),
        torch.as_tensor(dataset.native_experts[indices], device=device, dtype=torch.long),
        torch.as_tensor(dataset.block_indices[indices], device=device, dtype=torch.long),
        torch.as_tensor(dataset.sigmas[indices], device=device),
        torch.as_tensor(dataset.token_indices[indices], device=device, dtype=torch.long),
        dataset.sequence_length,
    )


def predict_indices(model, dataset, indices, device, batch_size=TRAIN_BATCH_SIZE):
    indices = np.asarray(indices, dtype=np.int64)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            predictions.append(_model_batch(model, dataset, batch, device).float().cpu())
    return torch.cat(predictions).numpy() if predictions else np.empty((0, 2))


def _validation_concordance(model, dataset, case_ids, targets, device):
    selected_cases = set(case_ids)
    values = []
    for cell in dataset.cells:
        if cell["case_id"] not in selected_cases:
            continue
        indices = np.arange(cell["token_start"], cell["token_stop"], dtype=np.int64)
        predictions = predict_indices(model, dataset, indices, device)
        native = dataset.native_experts[indices]
        candidates = build_same_expert_exchange_candidates(
            native,
            dataset.num_experts,
            int(cell["candidate_seed"]),
        )
        values.append(candidate_concordance(candidates, predictions, targets[indices]))
    if not values:
        raise ValueError("Validation split contains no cells")
    return float(np.mean(values)), values


def train_dual_scorer(dataset, fit_case_ids, validation_case_ids, kind, device):
    if kind not in SCORER_KINDS:
        raise ValueError(f"Unknown scorer kind: {kind}")
    if dataset.targets is None:
        raise ValueError("Scorer calibration requires privileged targets")
    if set(fit_case_ids) & set(validation_case_ids):
        raise ValueError("Fit and validation case IDs overlap")
    if set(fit_case_ids) | set(validation_case_ids) != set(dataset.case_ids):
        raise ValueError("Fit and validation cases do not cover calibration")
    device = torch.device(device)
    targets = dataset.targets
    training_targets = (
        roll_counterfactual_correspondence(
            targets,
            dataset.native_experts,
            dataset.cell_ids,
        )
        if kind == "rolled_correspondence" else targets
    )
    fit_indices = dataset.indices_for_cases(fit_case_ids)
    partner = pair_indices(dataset.native_experts, dataset.cell_ids)
    include_hidden = kind != "router_context"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_SEED)
        model = DualLinearUtilityScorer(
            hidden_dim=dataset.hidden_dim,
            num_experts=dataset.num_experts,
            blocks=MOE_BLOCKS,
            include_hidden=include_hidden,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    best_state = None
    best_epoch = None
    best_metric = -np.inf
    stale_epochs = 0
    history = []
    generator = torch.Generator(device="cpu")
    for epoch in range(MAX_EPOCHS):
        generator.manual_seed(MODEL_SEED + epoch)
        order = fit_indices[torch.randperm(len(fit_indices), generator=generator).numpy()]
        model.train()
        losses = []
        for start in range(0, len(order), TRAIN_BATCH_SIZE):
            indices = order[start:start + TRAIN_BATCH_SIZE]
            partner_indices = partner[indices]
            predictions = _model_batch(model, dataset, indices, device)
            partner_predictions = _model_batch(model, dataset, partner_indices, device)
            target = torch.as_tensor(training_targets[indices], device=device)
            partner_target = torch.as_tensor(
                training_targets[partner_indices],
                device=device,
            )
            regression = F.smooth_l1_loss(predictions, target)
            target_delta = target - partner_target
            valid_pairs = target_delta.abs() > 1e-6
            if valid_pairs.any():
                signs = target_delta[valid_pairs].sign()
                predicted_delta = (predictions - partner_predictions)[valid_pairs]
                pairwise = F.softplus(-signs * predicted_delta).mean()
            else:
                pairwise = predictions.sum() * 0.0
            loss = regression + PAIRWISE_LOSS_WEIGHT * pairwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        validation_metric, _ = _validation_concordance(
            model,
            dataset,
            validation_case_ids,
            training_targets,
            device,
        )
        history.append({
            "epoch": epoch + 1,
            "mean_train_loss": float(np.mean(losses)),
            "validation_candidate_concordance": validation_metric,
        })
        if validation_metric > best_metric + 1e-8:
            best_metric = validation_metric
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch + 1 >= MIN_EPOCHS and stale_epochs >= EARLY_STOPPING_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("Scorer calibration did not produce a model state")
    model.load_state_dict(best_state)
    true_metric, true_values = _validation_concordance(
        model,
        dataset,
        validation_case_ids,
        targets,
        device,
    )
    fitted_metric, _ = _validation_concordance(
        model,
        dataset,
        validation_case_ids,
        training_targets,
        device,
    )
    return model.cpu(), {
        "kind": kind,
        "include_hidden": include_hidden,
        "best_epoch": int(best_epoch),
        "best_fitted_target_validation_concordance": float(best_metric),
        "final_fitted_target_validation_concordance": float(fitted_metric),
        "true_target_validation_concordance": float(true_metric),
        "true_target_validation_cell_values": [float(value) for value in true_values],
        "epochs_run": len(history),
        "history": history,
    }


def scorer_bundle(model, kind, fit_summary, calibration_contract):
    return {
        "format_version": 1,
        "kind": kind,
        "model": {
            "hidden_dim": model.hidden_dim,
            "num_experts": model.num_experts,
            "blocks": list(model.blocks),
            "include_hidden": model.include_hidden,
            "state_dict": model.state_dict(),
        },
        "fit_summary": fit_summary,
        "calibration_contract": calibration_contract,
    }


def load_scorer_bundle(path, map_location="cpu"):
    try:
        bundle = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        bundle = torch.load(path, map_location=map_location)
    if bundle.get("format_version") != 1 or bundle.get("kind") not in SCORER_KINDS:
        raise ValueError("Unknown scorer bundle format")
    spec = bundle["model"]
    model = DualLinearUtilityScorer(
        hidden_dim=spec["hidden_dim"],
        num_experts=spec["num_experts"],
        blocks=spec["blocks"],
        include_hidden=spec["include_hidden"],
    )
    model.load_state_dict(spec["state_dict"], strict=True)
    return model.eval(), bundle
