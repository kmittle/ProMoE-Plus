"""Blinded held-out evaluator for the sealed three-arm continuation."""

from __future__ import annotations

import copy
import base64
import gc
import json
import math
import os
import random
import zlib
from itertools import zip_longest
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from analyses.denoising_regret.probe import _build_model
from analyses.t_SNE.checkpoint_utils import load_runtime_cfg

from .controller import (
    BRANCHES,
    CHECKPOINT_STATE_KEY,
    CONTROLLER_STATE_VERSION,
    deterministic_group_sum,
)
from .heldout import canonical_json_sha256
from .protocol import SEALED_GPU_IDS, _sealed_gpu_device_pairs
from .serialization import atomic_write_json, sha256_file, tensor_sha256
from .state_digest import canonical_state_sha256
from .transcript import (
    GLOBAL_RECORD_FIELDS,
    FIELD_ORDER,
    LOCAL_RECORD_FIELDS,
    JsonlLedger,
    build_global_record,
    persisted_identity_field_hashes,
    persisted_record_digest,
    validate_local_transcript_replay,
)


EVALUATOR_VERSION = 1
BLOCK_INDICES = (1, 3, 5, 7, 9, 11)
NUM_EXPERTS = 12
START_STEP = 301001
FINAL_STEP = 321000
CHECKPOINT_STATES = ("ema_model_state_dict", "model_state_dict")
NUMERICAL_COUNTER_NAMES = {
    "nonfinite",
    "rank_disagreement",
    "transcript_mismatch",
    "budget_violation",
    "capture_failure",
    "checkpoint_failure",
}
CONTROLLER_RECORD_FIELDS = {
    "version",
    "step",
    "branch",
    "update_index",
    "global_transcript_digest",
    "rank_consensus_digest",
    "permutation_offset",
    "global_credit",
    "global_count",
    "credit_rate_ema",
    "raw_scales",
    "permuted_scales",
    "selected_budget_factors",
    "applied_scales",
    "pre_gradient_squared_norm",
    "post_gradient_squared_norm",
    "full_pre_gradient_squared_norm",
    "full_post_gradient_squared_norm",
    "block_relative_budget_drift",
    "full_relative_budget_drift",
    "chain_digest",
}
CASE_PUBLIC_FIELDS = {
    "version",
    "branch",
    "checkpoint_state",
    "checkpoint_sha256",
    "protocol_sha256",
    "case_index",
    "label",
    "relative_path",
    "metric_payload_sha256",
}
CASE_METRIC_FIELDS = {
    "version",
    "case_index",
    "label",
    "relative_path",
    "mean_mse",
    "aggregate_credit",
    "aggregate_count",
    "cells",
}
FORMULA_RTOL = 1e-12
GRADIENT_SCALE_RTOL = 1e-6
HELDOUT_CASE_COUNT = 128
CHECKPOINT_TOP_LEVEL_FIELDS = {
    "step",
    "model_state_dict",
    "ema_model_state_dict",
    "optimizer_state_dict",
    "trainer_state",
    CHECKPOINT_STATE_KEY,
}
TRAINER_STATE_FIELDS = {
    "version",
    "augmentation_seed_version",
    "global_seed",
    "sampler_contract",
    "world_size",
    "rank_states",
    "next_step",
    "data_batches_seen",
    "sampler_epoch",
    "sampler_batch_offset",
    "grad_mix",
    "batches_per_epoch",
}
OPTIMIZER_GROUP_FIELDS = {
    "amsgrad",
    "betas",
    "capturable",
    "differentiable",
    "eps",
    "foreach",
    "fused",
    "lr",
    "maximize",
    "params",
    "weight_decay",
}


def _validate_rng_state(state):
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Checkpoint rank RNG state fields differ")
    try:
        random.Random().setstate(state["python"])
        numpy_state = state["numpy"]
        if not isinstance(numpy_state, dict) or set(numpy_state) != {
            "bit_generator",
            "state",
            "position",
            "has_gauss",
            "cached_gaussian",
        }:
            raise ValueError("NumPy RNG state fields differ")
        if not torch.is_tensor(numpy_state["state"]):
            raise TypeError("NumPy RNG state vector is not a tensor")
        vector = numpy_state["state"].detach().cpu()
        if vector.dtype != torch.int64 or vector.ndim != 1 or vector.numel() == 0:
            raise ValueError("NumPy RNG state vector differs")
        vector_np = vector.numpy().astype(np.uint32, copy=True)
        np.random.RandomState().set_state((
            numpy_state["bit_generator"],
            vector_np,
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        ))
        for name in ("torch", "cuda"):
            value = state[name]
            if (
                not torch.is_tensor(value)
                or value.dtype != torch.uint8
                or value.ndim != 1
                or value.numel() == 0
            ):
                raise ValueError(f"Checkpoint {name} RNG state differs")
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError("Checkpoint RNG state is invalid") from error


def _validate_checkpoint_state_mapping(state, name, reference=None):
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Branch checkpoint {name} is missing or empty")
    if reference is not None and set(state) != set(reference):
        raise ValueError(f"Branch checkpoint {name} keys differ from the model")
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError(f"Branch checkpoint {name} contains an invalid entry")
        if value.layout != torch.strided or value.device.type != "cpu":
            raise ValueError(f"Branch checkpoint {name} contains a non-CPU tensor")
        if reference is not None:
            expected = reference[key]
            if (
                tuple(value.shape) != tuple(expected.shape)
                or value.dtype != expected.dtype
                or value.layout != expected.layout
            ):
                raise ValueError(f"Branch checkpoint {name} tensor metadata differs")


def _validate_optimizer_checkpoint(state, model=None):
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("Branch checkpoint optimizer state fields differ")
    groups = state["param_groups"]
    saved_state = state["state"]
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(saved_state, dict):
        raise ValueError("Branch checkpoint optimizer state is incomplete")
    group = groups[0]
    if not isinstance(group, dict) or set(group) != OPTIMIZER_GROUP_FIELDS:
        raise ValueError("Branch checkpoint optimizer parameter-group fields differ")
    params = group["params"]
    if (
        not isinstance(params, list)
        or not params
        or any(isinstance(value, bool) or not isinstance(value, int) for value in params)
        or len(set(params)) != len(params)
        or set(params) != set(range(len(params)))
    ):
        raise ValueError("Branch checkpoint optimizer parameter IDs differ")
    if set(saved_state) != set(params):
        raise ValueError("Branch checkpoint optimizer state coverage differs")
    reference_parameters = (
        [parameter for parameter in model.parameters() if parameter.requires_grad]
        if model is not None
        else None
    )
    if reference_parameters is not None and len(reference_parameters) != len(params):
        raise ValueError("Branch checkpoint optimizer parameter count differs")
    for parameter_id in params:
        value = saved_state[parameter_id]
        if not isinstance(value, dict) or set(value) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise ValueError("Branch checkpoint optimizer moment fields differ")
        step = value["step"]
        if not torch.is_tensor(step) or step.numel() != 1 or step.is_complex():
            raise ValueError("Branch checkpoint optimizer step differs")
        if not math.isfinite(float(step.item())) or float(step.item()) < 0:
            raise ValueError("Branch checkpoint optimizer step is invalid")
        first = value["exp_avg"]
        second = value["exp_avg_sq"]
        if not torch.is_tensor(first) or not torch.is_tensor(second):
            raise ValueError("Branch checkpoint optimizer moments are not tensors")
        if (
            first.layout != torch.strided
            or second.layout != torch.strided
            or first.device.type != "cpu"
            or second.device.type != "cpu"
            or tuple(first.shape) != tuple(second.shape)
            or first.dtype != second.dtype
        ):
            raise ValueError("Branch checkpoint optimizer moment metadata differs")
        if reference_parameters is not None:
            parameter = reference_parameters[parameter_id]
            if tuple(first.shape) != tuple(parameter.shape) or first.dtype != parameter.dtype:
                raise ValueError("Branch checkpoint optimizer moment shape differs")


def _validate_trainer_state(trainer):
    if not isinstance(trainer, dict) or set(trainer) != TRAINER_STATE_FIELDS:
        raise ValueError("Branch checkpoint trainer-v2 fields differ")
    if (
        trainer["version"] != 2
        or trainer["augmentation_seed_version"] != 1
        or trainer["global_seed"] != 0
        or trainer["world_size"] != 4
        or trainer["grad_mix"] != 1
    ):
        raise ValueError("Branch checkpoint trainer-v2 metadata differs")
    integer_fields = (
        "next_step",
        "data_batches_seen",
        "sampler_epoch",
        "sampler_batch_offset",
        "batches_per_epoch",
    )
    if any(
        isinstance(trainer[name], bool)
        or not isinstance(trainer[name], int)
        or trainer[name] < 0
        for name in integer_fields
    ) or trainer["batches_per_epoch"] <= 0:
        raise ValueError("Branch checkpoint trainer-v2 progress differs")
    if (
        trainer["next_step"] != FINAL_STEP + 1
        or trainer["data_batches_seen"] != trainer["next_step"] * trainer["grad_mix"]
        or (trainer["sampler_epoch"], trainer["sampler_batch_offset"])
        != divmod(trainer["data_batches_seen"], trainer["batches_per_epoch"])
    ):
        raise ValueError("Branch checkpoint trainer-v2 sampler position differs")
    sampler = trainer["sampler_contract"]
    if not isinstance(sampler, dict) or set(sampler) != {
        "version",
        "type",
        "global_seed",
        "per_rank_batch_size",
        "drop_last",
        "case1_prob",
        "dataset",
    }:
        raise ValueError("Branch checkpoint sampler contract differs")
    if (
        sampler["version"] != 1
        or sampler["type"] != "distributed"
        or sampler["global_seed"] != 0
        or sampler["per_rank_batch_size"] != 64
        or sampler["drop_last"] is not False
        or sampler["case1_prob"] is not None
    ):
        raise ValueError("Branch checkpoint sampler contract metadata differs")
    dataset = sampler["dataset"]
    if not isinstance(dataset, dict) or set(dataset) != {
        "version",
        "type",
        "num_samples",
        "ordered_samples_sha256",
    }:
        raise ValueError("Branch checkpoint dataset sampler identity differs")
    if (
        dataset["version"] != 1
        or not isinstance(dataset["type"], str)
        or not isinstance(dataset["num_samples"], int)
        or dataset["num_samples"] <= 0
        or not _is_sha256(dataset["ordered_samples_sha256"])
    ):
        raise ValueError("Branch checkpoint dataset sampler identity is invalid")
    rank_states = trainer["rank_states"]
    if not isinstance(rank_states, list) or len(rank_states) != 4:
        raise ValueError("Branch checkpoint rank RNG states are incomplete")
    if any(
        not isinstance(item, dict)
        or set(item) != {"rank", "rng_state"}
        or item["rank"] != rank
        for rank, item in enumerate(rank_states)
    ):
        raise ValueError("Branch checkpoint rank RNG state IDs differ")
    for item in rank_states:
        _validate_rng_state(item["rng_state"])


class EvaluationCapture:
    """Capture native routes and differentiable MoE outputs for one eval cell."""

    def __init__(self, model):
        self.model = model
        self.layers = {}
        self.records = {}
        self._overrides = []
        self._handles = []
        for block_index in BLOCK_INDICES:
            block = model.blocks[block_index]
            layer = block.mlp
            if not bool(getattr(block, "use_moe", False)):
                raise TypeError(f"Block {block_index} is not a routed MoE block")
            if int(getattr(layer, "num_routed_experts", -1)) != NUM_EXPERTS:
                raise ValueError(f"Block {block_index} expert count changed")
            if int(getattr(layer, "top_k", -1)) != 1:
                raise ValueError("Held-out evaluator requires native top-1 routing")
            self.layers[block_index] = layer
            self._install(block_index, layer)
        self.reset()

    def _install(self, block_index, layer):
        if "compute_router" in layer.__dict__:
            raise RuntimeError("MoE layer already has a compute_router override")
        original = layer.compute_router

        def wrapped(this, hidden_states, labels, _original=original, _block=block_index):
            result = _original(hidden_states, labels)
            record = self.records[_block]
            if record["route_weights"] is not None:
                raise RuntimeError(f"Block {_block} routed more than once")
            if not isinstance(result, tuple) or len(result) != 3:
                raise TypeError("Router return contract changed")
            weights, indices, auxiliary = result
            if auxiliary is not None:
                raise RuntimeError("Eval routing unexpectedly produced an auxiliary loss")
            record["route_weights"] = weights.detach()
            record["route_indices"] = indices.detach()
            record["labels"] = labels.detach()
            return result

        layer.compute_router = MethodType(wrapped, layer)
        self._overrides.append(layer)

        def capture_output(module, inputs, output, _block=block_index):
            del module, inputs
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError("Sparse MoE output contract changed")
            if self.records[_block]["output"] is not None:
                raise RuntimeError(f"Block {_block} produced more than one output")
            self.records[_block]["output"] = output[0]
            return None

        self._handles.append(layer.register_forward_hook(capture_output))

    def reset(self):
        self.records = {
            block_index: {
                "route_weights": None,
                "route_indices": None,
                "labels": None,
                "output": None,
            }
            for block_index in BLOCK_INDICES
        }

    def outputs(self):
        outputs = []
        for block_index in BLOCK_INDICES:
            output = self.records[block_index]["output"]
            if output is None:
                raise RuntimeError(f"Block {block_index} output was not captured")
            outputs.append(output)
        return outputs

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for layer in self._overrides:
            del layer.compute_router
        self._overrides.clear()


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_heldout_manifest(path):
    path = Path(path).resolve()
    manifest = _load_json(path)
    digest = canonical_json_sha256(manifest)
    sidecar = path.with_suffix(".sha256")
    if sidecar.read_text(encoding="utf-8") != digest + "\n":
        raise RuntimeError("Held-out manifest hash sidecar differs")
    complete = path.parent / "COMPLETE"
    if complete.read_text(encoding="utf-8") != digest + "\n":
        raise RuntimeError("Held-out manifest COMPLETE marker differs")
    if manifest.get("case_count") != 128 or len(manifest.get("cases", ())) != 128:
        raise ValueError("Held-out manifest must contain exactly 128 cases")
    if manifest.get("noise_draws_per_image") != 8:
        raise ValueError("Held-out manifest must contain eight noise draws per case")
    if manifest.get("sigmas") != [0.2, 0.5, 0.8]:
        raise ValueError("Held-out sigma contract changed")
    labels = [case.get("label") for case in manifest["cases"]]
    if len(set(labels)) != 128:
        raise ValueError("Held-out labels are not class-disjoint")
    return manifest, digest


def _load_tensor(tensor_dir, record):
    path = Path(tensor_dir).resolve() / record["path"]
    if sha256_file(path) != record["file_sha256"]:
        raise RuntimeError(f"Held-out tensor file hash mismatch: {path}")
    array = np.load(path, allow_pickle=False)
    tensor = torch.from_numpy(np.array(array, copy=True)).contiguous()
    expected_dtype = getattr(torch, record["dtype"], None)
    if expected_dtype is None or tensor.dtype != expected_dtype:
        raise TypeError(f"Held-out tensor dtype mismatch: {path}")
    if list(tensor.shape) != record["shape"]:
        raise ValueError(f"Held-out tensor shape mismatch: {path}")
    if tensor_sha256(tensor) != record["tensor_sha256"]:
        raise RuntimeError(f"Held-out tensor content hash mismatch: {path}")
    if tensor.dtype != torch.float32:
        raise TypeError("Held-out evaluation tensors must be float32")
    return tensor


def _prediction_tensor(model_output, target_channels=4):
    if isinstance(model_output, tuple):
        model_output = model_output[0]
    if model_output.ndim != 4:
        raise ValueError("Held-out model output must be four-dimensional")
    if model_output.shape[1] == target_channels * 2:
        model_output = model_output[:, :target_channels]
    if model_output.shape[1] != target_channels:
        raise ValueError("Held-out model output channel count changed")
    return model_output.unsqueeze(2)


def _credit_and_count(record, gradient, expected_label):
    weights = record["route_weights"]
    indices = record["route_indices"]
    labels = record["labels"]
    output = record["output"]
    if any(value is None for value in (weights, indices, labels, output)):
        raise RuntimeError("Held-out routing capture is incomplete")
    if weights.shape != indices.shape or weights.shape[-1] != 1:
        raise ValueError("Held-out route shape differs from native top-1")
    if gradient.shape != output.shape:
        raise ValueError("Held-out suffix gradient shape differs from MoE output")
    if labels.shape != (1,) or int(labels.item()) != int(expected_label):
        raise RuntimeError("Held-out evaluation changed the class label")
    flat_indices = indices.reshape(-1).to(dtype=torch.int64)
    if not bool(torch.all((flat_indices >= 0) & (flat_indices < NUM_EXPERTS))):
        raise RuntimeError("Held-out conditional token used a non-routed expert")
    flat_weights = weights.reshape(-1).to(dtype=torch.float64)
    flat_gradient = gradient.reshape(flat_indices.numel(), -1).to(dtype=torch.float64)
    if not bool(torch.isfinite(flat_weights).all()):
        raise FloatingPointError("Held-out route weights are nonfinite")
    if not bool(torch.isfinite(flat_gradient).all()):
        raise FloatingPointError("Held-out suffix gradient is nonfinite")
    token_credit = flat_weights.square() * flat_gradient.square().sum(dim=1)
    credit, count = deterministic_group_sum(
        token_credit,
        flat_indices,
        NUM_EXPERTS,
    )
    if int(count.sum().item()) != flat_indices.numel():
        raise RuntimeError("Held-out expert count does not cover every token")
    if not bool(torch.isfinite(credit).all()):
        raise FloatingPointError("Held-out expert credit is nonfinite")
    return credit.cpu(), count.cpu()


def evaluate_case(model, capture, case, tensor_dir, sigmas=(0.2, 0.5, 0.8)):
    if model.training:
        raise RuntimeError("Held-out evaluator requires model.eval()")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Held-out evaluator must freeze model parameters")
    device = next(model.parameters()).device
    z = _load_tensor(tensor_dir, case["z"])
    if tuple(z.shape) != (4, 1, 32, 32):
        raise ValueError("Held-out z tensor shape changed")
    noises = case.get("noise")
    if not isinstance(noises, list) or len(noises) != 8:
        raise ValueError("Held-out case must contain eight noise draws")
    label = int(case["label"])
    label_tensor = torch.tensor([label], device=device, dtype=torch.int64)
    cells = []
    aggregate_credit = torch.zeros(6, NUM_EXPERTS, dtype=torch.float64)
    aggregate_count = torch.zeros(6, NUM_EXPERTS, dtype=torch.int64)
    mse_values = []
    for noise_record in noises:
        noise = _load_tensor(tensor_dir, noise_record)
        if tuple(noise.shape) != tuple(z.shape):
            raise ValueError("Held-out z/noise shapes differ")
        z_device = z.to(device=device)
        noise_device = noise.to(device=device)
        for sigma_index, sigma_value in enumerate(sigmas):
            sigma = torch.tensor(sigma_value, device=device, dtype=torch.float32)
            one = torch.tensor(1.0, device=device, dtype=torch.float32)
            thousand = torch.tensor(1000.0, device=device, dtype=torch.float32)
            target = (noise_device - z_device).unsqueeze(0)
            model_input = ((one - sigma) * z_device + sigma * noise_device)
            model_input = model_input.unsqueeze(0).requires_grad_(True)
            timestep = (sigma * thousand).reshape(1)
            capture.reset()
            with torch.autocast(device_type=device.type, enabled=False):
                prediction = _prediction_tensor(
                    model(model_input, timestep, context=label_tensor)
                )
                difference = prediction.to(torch.float64) - target.to(torch.float64)
                mse = difference.square().mean()
            if not bool(torch.isfinite(mse)):
                raise FloatingPointError("Held-out denoising MSE is nonfinite")
            gradients = torch.autograd.grad(
                mse,
                capture.outputs(),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            block_credit = []
            block_count = []
            for row, (block_index, gradient) in enumerate(
                zip(BLOCK_INDICES, gradients)
            ):
                credit, count = _credit_and_count(
                    capture.records[block_index], gradient, label
                )
                aggregate_credit[row] += credit
                aggregate_count[row] += count
                block_credit.append(credit.tolist())
                block_count.append(count.tolist())
            mse_value = float(mse.item())
            mse_values.append(mse_value)
            cells.append({
                "draw": int(noise_record["draw"]),
                "sigma_index": sigma_index,
                "sigma": float(sigma_value),
                "mse": mse_value,
                "credit": block_credit,
                "count": block_count,
            })
            del gradients, mse, prediction, difference, model_input
    if len(cells) != 24:
        raise RuntimeError("Held-out case did not produce exactly 24 cells")
    mean_mse = math.fsum(mse_values) / len(mse_values)
    if not math.isfinite(mean_mse) or mean_mse <= 0:
        raise FloatingPointError("Held-out per-image MSE must be finite and positive")
    return {
        "version": EVALUATOR_VERSION,
        "case_index": int(case["index"]),
        "label": label,
        "relative_path": case["relative_path"],
        "mean_mse": mean_mse,
        "aggregate_credit": aggregate_credit.tolist(),
        "aggregate_count": aggregate_count.tolist(),
        "cells": cells,
    }


def _case_artifact_path(output_root, branch, state_name, case_index):
    return (
        Path(output_root)
        / "raw"
        / branch
        / state_name
        / f"case-{int(case_index):03d}.json"
    )


def _case_metric_artifact_path(output_root, branch, state_name, case_index):
    """Return the delayed numerical artifact path for one held-out case."""
    return (
        Path(output_root)
        / "sealed"
        / branch
        / state_name
        / f"case-{int(case_index):03d}.json"
    )


def _seal_path(path):
    return Path(str(path) + ".seal.json")


def _canonical_json_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _encode_metric_payload(payload, protocol_sha256):
    if set(payload) != CASE_METRIC_FIELDS:
        raise ValueError("Held-out delayed metric fields differ")
    encoded = _canonical_json_bytes(payload)
    return {
        "version": 1,
        "protocol_sha256": protocol_sha256,
        "encoding": "zlib+base64+canonical-json",
        "payload_canonical_sha256": canonical_json_sha256(payload),
        "payload_b64": base64.b64encode(zlib.compress(encoded, level=9)).decode(
            "ascii"
        ),
    }


def _decode_metric_payload(envelope, protocol_sha256):
    if not isinstance(envelope, dict) or set(envelope) != {
        "version",
        "protocol_sha256",
        "encoding",
        "payload_canonical_sha256",
        "payload_b64",
    }:
        raise RuntimeError("Held-out delayed metric envelope fields differ")
    if (
        envelope["version"] != 1
        or envelope["protocol_sha256"] != protocol_sha256
        or envelope["encoding"] != "zlib+base64+canonical-json"
        or not isinstance(envelope["payload_b64"], str)
        or not _is_sha256(envelope["payload_canonical_sha256"])
    ):
        raise RuntimeError("Held-out delayed metric envelope metadata differs")
    try:
        decoded = zlib.decompress(base64.b64decode(envelope["payload_b64"], validate=True))
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, zlib.error, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Held-out delayed metric payload is malformed") from error
    if _canonical_json_bytes(payload) != decoded:
        raise RuntimeError("Held-out delayed metric canonical bytes differ")
    if (
        not isinstance(payload, dict)
        or set(payload) != CASE_METRIC_FIELDS
        or canonical_json_sha256(payload) != envelope["payload_canonical_sha256"]
    ):
        raise RuntimeError("Held-out delayed metric payload commitment differs")
    return payload


def _publish_case(path, payload, protocol_sha256):
    path = Path(path)
    required_metadata = CASE_PUBLIC_FIELDS - {"metric_payload_sha256"}
    if (
        not isinstance(payload, dict)
        or not required_metadata.issubset(payload)
        or not CASE_METRIC_FIELDS.issubset(payload)
    ):
        raise ValueError("Held-out case payload lacks delayed metrics")
    public_payload = {
        key: payload[key]
        for key in CASE_PUBLIC_FIELDS
        if key != "metric_payload_sha256"
    }
    metric_payload = {key: payload[key] for key in CASE_METRIC_FIELDS}
    public_payload["metric_payload_sha256"] = canonical_json_sha256(metric_payload)
    metric_path = _case_metric_artifact_path(
        path.parents[3],
        payload["branch"],
        payload["checkpoint_state"],
        payload["case_index"],
    )
    metric_envelope = _encode_metric_payload(metric_payload, protocol_sha256)
    public_seal = {
        "version": 1,
        "artifact": path.name,
        "artifact_canonical_sha256": canonical_json_sha256(public_payload),
        "protocol_sha256": protocol_sha256,
    }
    metric_seal = {
        "version": 1,
        "artifact": metric_path.name,
        "artifact_canonical_sha256": canonical_json_sha256(metric_envelope),
        "protocol_sha256": protocol_sha256,
    }
    paths = (
        path,
        _seal_path(path),
        metric_path,
        _seal_path(metric_path),
    )
    if any(item.exists() for item in paths):
        if not all(item.exists() for item in paths):
            raise RuntimeError(f"Incomplete held-out artifact set: {path}")
        if (
            _load_json(path) != public_payload
            or _load_json(_seal_path(path)) != public_seal
            or _load_json(metric_path) != metric_envelope
            or _load_json(_seal_path(metric_path)) != metric_seal
        ):
            raise RuntimeError(f"Existing held-out artifact differs: {path}")
        return
    atomic_write_json(metric_path, metric_envelope, mode=0o444)
    atomic_write_json(_seal_path(metric_path), metric_seal, mode=0o444)
    atomic_write_json(path, public_payload, mode=0o444)
    atomic_write_json(_seal_path(path), public_seal, mode=0o444)


def _load_reusable_case(path, expected, protocol_sha256):
    path = Path(path)
    seal_path = _seal_path(path)
    metric_path = _case_metric_artifact_path(
        path.parents[3],
        expected["branch"],
        expected["checkpoint_state"],
        expected["case_index"],
    )
    metric_seal_path = _seal_path(metric_path)
    paths = (path, seal_path, metric_path, metric_seal_path)
    if not any(item.exists() for item in paths):
        return None
    if not all(item.exists() for item in paths):
        raise RuntimeError(f"Incomplete held-out artifact set: {path}")
    public_payload = _load_json(path)
    public_seal = _load_json(seal_path)
    metric_envelope = _load_json(metric_path)
    metric_seal = _load_json(metric_seal_path)
    if set(public_payload) != CASE_PUBLIC_FIELDS:
        raise RuntimeError(f"Held-out public artifact fields differ: {path}")
    expected_public_seal = {
        "version": 1,
        "artifact": path.name,
        "artifact_canonical_sha256": canonical_json_sha256(public_payload),
        "protocol_sha256": protocol_sha256,
    }
    if public_seal != expected_public_seal:
        raise RuntimeError(f"Held-out artifact protocol mismatch: {path}")
    expected_metric_seal = {
        "version": 1,
        "artifact": metric_path.name,
        "artifact_canonical_sha256": canonical_json_sha256(metric_envelope),
        "protocol_sha256": protocol_sha256,
    }
    if metric_seal != expected_metric_seal:
        raise RuntimeError(f"Held-out delayed metric seal mismatch: {metric_path}")
    metric_payload = _decode_metric_payload(metric_envelope, protocol_sha256)
    if public_payload["metric_payload_sha256"] != canonical_json_sha256(
        metric_payload
    ):
        raise RuntimeError(f"Held-out delayed metric commitment mismatch: {path}")
    payload = {**public_payload, **metric_payload}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Held-out artifact metadata mismatch: {path}:{key}")
    return payload


def validate_branch_checkpoint(
    checkpoint_path,
    expected_sha256,
    branch,
    *,
    verify_file_hash=True,
    reference_model=None,
):
    checkpoint_path = Path(checkpoint_path).resolve()
    initial_sha256 = None
    if verify_file_hash:
        initial_sha256 = sha256_file(checkpoint_path)
        if expected_sha256 is not None and initial_sha256 != expected_sha256:
            raise RuntimeError(f"Branch checkpoint hash mismatch: {checkpoint_path}")
    else:
        if expected_sha256 is None:
            raise ValueError("Skipping checkpoint hashing requires an expected hash")
        observed_sha256 = expected_sha256
    load_kwargs = {"map_location": "cpu", "weights_only": True, "mmap": True}
    try:
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("mmap")
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
    if verify_file_hash:
        final_sha256 = sha256_file(checkpoint_path)
        if final_sha256 != initial_sha256:
            raise RuntimeError(
                f"Branch checkpoint changed while loading: {checkpoint_path}"
            )
        if expected_sha256 is not None and final_sha256 != expected_sha256:
            raise RuntimeError(f"Branch checkpoint hash mismatch: {checkpoint_path}")
        observed_sha256 = final_sha256
    if not isinstance(checkpoint, dict) or set(checkpoint) != CHECKPOINT_TOP_LEVEL_FIELDS:
        raise ValueError("Branch checkpoint top-level fields differ")
    if checkpoint.get("step") != FINAL_STEP:
        raise ValueError("Branch checkpoint is not the sealed final step")
    model_state = checkpoint["model_state_dict"]
    ema_state = checkpoint["ema_model_state_dict"]
    reference_state = reference_model.state_dict() if reference_model is not None else None
    _validate_checkpoint_state_mapping(model_state, "model_state_dict", reference_state)
    _validate_checkpoint_state_mapping(
        ema_state,
        "ema_model_state_dict",
        reference_state if reference_state is not None else model_state,
    )
    if set(model_state) != set(ema_state):
        raise ValueError("Branch checkpoint model and EMA keys differ")
    _validate_optimizer_checkpoint(
        checkpoint["optimizer_state_dict"],
        model=reference_model,
    )
    _validate_trainer_state(checkpoint["trainer_state"])
    extension = checkpoint.get(CHECKPOINT_STATE_KEY)
    if not isinstance(extension, dict):
        raise ValueError("Branch checkpoint lacks controller state")
    expected_extension_fields = {
        "version",
        "branch",
        "execution_mode",
        "block_indices",
        "num_experts",
        "start_step",
        "last_step",
        "update_count",
        "normalizer",
        "numerical_counters",
    }
    if set(extension) != expected_extension_fields:
        raise ValueError("Branch checkpoint controller fields differ")
    expected_updates = FINAL_STEP - START_STEP + 1
    controller_metadata = {
        "version": CONTROLLER_STATE_VERSION,
        "branch": branch,
        "execution_mode": "continuation",
        "block_indices": list(BLOCK_INDICES),
        "num_experts": NUM_EXPERTS,
        "start_step": START_STEP,
        "last_step": FINAL_STEP,
        "update_count": expected_updates,
    }
    if any(extension.get(key) != value for key, value in controller_metadata.items()):
        raise ValueError("Branch checkpoint controller metadata differs")
    counters = extension.get("numerical_counters")
    if (
        not isinstance(counters, dict)
        or set(counters) != NUMERICAL_COUNTER_NAMES
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value != 0
            for value in counters.values()
        )
    ):
        raise RuntimeError("Branch checkpoint records a numerical-integrity failure")
    normalizer = extension.get("normalizer")
    if not isinstance(normalizer, dict):
        raise ValueError("Branch checkpoint lacks normalizer state")
    if set(normalizer) != {"ema", "initialized", "ema_decay", "epsilon"}:
        raise ValueError("Branch checkpoint normalizer fields differ")
    if (
        normalizer.get("ema_decay") != 0.99
        or normalizer.get("epsilon") != 1e-30
    ):
        raise ValueError("Branch checkpoint normalizer constants differ")
    ema = normalizer.get("ema")
    initialized = normalizer.get("initialized")
    if (
        not torch.is_tensor(ema)
        or ema.dtype != torch.float64
        or tuple(ema.shape) != (len(BLOCK_INDICES), NUM_EXPERTS)
        or not bool(torch.isfinite(ema).all())
        or not bool(torch.all(ema > 0))
    ):
        raise ValueError("Branch checkpoint normalizer EMA differs")
    if (
        not torch.is_tensor(initialized)
        or initialized.dtype != torch.bool
        or tuple(initialized.shape) != tuple(ema.shape)
        or not bool(torch.all(initialized))
    ):
        raise ValueError("Branch checkpoint normalizer initialization differs")
    return checkpoint, observed_sha256


def _iter_validated_ledger(path, start_step):
    path = Path(path).resolve()
    previous_chain = "0" * 64
    expected_step = int(start_step)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed ledger {path}:{line_number}") from error
            if record.get("step") != expected_step:
                raise ValueError(f"Ledger steps are not contiguous at {path}")
            expected_chain = JsonlLedger._chain(previous_chain, record)
            if record.get("chain_digest") != expected_chain:
                raise ValueError(f"Ledger chain mismatch at {path}:{expected_step}")
            yield record
            previous_chain = expected_chain
            expected_step += 1


def validate_branch_transcripts(
    artifact_root,
    branch,
    start_step=301001,
    *,
    reference_artifact_root=None,
    reference_branch="measure_only_control",
    replay_context=None,
):
    if replay_context is None:
        raise RuntimeError(
            "Transcript tensor fields require deterministic replay; a reference "
            "transcript alone is not a sufficient source of truth"
        )
    if replay_context is not None:
        if not isinstance(replay_context, dict):
            raise TypeError("Transcript replay context must be a mapping")
        validate_local_transcript_replay(
            artifact_root=artifact_root,
            branch=branch,
            start_step=start_step,
            final_step=FINAL_STEP,
            **replay_context,
        )
    root = Path(artifact_root).resolve()
    local_streams = [
        _iter_validated_ledger(
            root / "transcripts" / branch / f"rank-{rank:02d}.jsonl",
            start_step,
        )
        for rank in range(4)
    ]
    global_stream = _iter_validated_ledger(
        root / "transcripts" / branch / "global.jsonl",
        start_step,
    )
    reference_local_streams = None
    reference_global_stream = None
    if reference_artifact_root is not None:
        reference_root = Path(reference_artifact_root).resolve()
        reference_local_streams = [
            _iter_validated_ledger(
                reference_root
                / "transcripts"
                / reference_branch
                / f"rank-{rank:02d}.jsonl",
                start_step,
            )
            for rank in range(4)
        ]
        reference_global_stream = _iter_validated_ledger(
            reference_root
            / "transcripts"
            / reference_branch
            / "global.jsonl",
            start_step,
        )
    expected_steps = FINAL_STEP - start_step + 1
    observed_steps = 0
    final_chain = None
    final_ema = None
    for records in zip_longest(*local_streams, global_stream):
        if any(record is None for record in records):
            raise RuntimeError(f"Branch {branch} transcript lengths differ")
        local_records = records[:4]
        global_record = records[4]
        step = start_step + observed_steps
        for expected_rank, local_record in enumerate(local_records):
            if set(local_record) != LOCAL_RECORD_FIELDS:
                raise RuntimeError(
                    f"Branch {branch} local transcript fields differ at {step}"
                )
            if (
                local_record.get("version") != 1
                or local_record.get("step") != step
                or local_record.get("rank") != expected_rank
                or not isinstance(local_record.get("relative_latent_paths"), list)
                or not isinstance(local_record.get("original_labels"), list)
                or len(local_record["relative_latent_paths"])
                != len(local_record["original_labels"])
            ):
                raise RuntimeError(
                    f"Branch {branch} local transcript identity differs at {step}"
                )
            field_hashes = local_record.get("field_sha256")
            if (
                not isinstance(field_hashes, dict)
                or set(field_hashes) != set(FIELD_ORDER)
                or any(not _is_sha256(value) for value in field_hashes.values())
                or not _is_sha256(local_record.get("step_digest"))
                or any(
                    local_record["field_sha256"].get(name)
                    != persisted_identity_field_hashes(local_record).get(name)
                    for name in ("relative_latent_paths", "original_labels")
                )
                or local_record.get("record_digest")
                != persisted_record_digest(local_record)
            ):
                raise RuntimeError(
                    f"Branch {branch} local transcript digest differs at {step}"
                )
        if [record.get("rank") for record in local_records] != list(range(4)):
            raise RuntimeError(f"Branch {branch} transcript rank IDs differ")
        if any(record.get("step") != step for record in local_records):
            raise RuntimeError(f"Branch {branch} local transcript step differs")
        if global_record.get("step") != step:
            raise RuntimeError(f"Branch {branch} global transcript step differs")
        if set(global_record) != GLOBAL_RECORD_FIELDS:
            raise RuntimeError(f"Branch {branch} global transcript fields differ at {step}")
        expected_global = build_global_record(step, local_records)
        if any(
            global_record.get(key) != expected_global.get(key)
            for key in expected_global
        ):
            raise RuntimeError(f"Branch {branch} local/global transcript mismatch")
        if reference_local_streams is not None:
            reference_records = []
            for reference_stream in reference_local_streams:
                try:
                    reference_records.append(next(reference_stream))
                except StopIteration as error:
                    raise RuntimeError(
                        f"Reference branch {reference_branch} transcript is shorter"
                    ) from error
            try:
                reference_global = next(reference_global_stream)
            except StopIteration as error:
                raise RuntimeError(
                    f"Reference branch {reference_branch} global transcript is shorter"
                ) from error
            if any(
                reference_record != local_record
                for reference_record, local_record in zip(
                    reference_records, local_records
                )
            ) or reference_global != global_record:
                raise RuntimeError(
                    f"Branch {branch} transcript differs from reference {reference_branch}"
                )
        final_chain = global_record["chain_digest"]
        observed_steps += 1
    if observed_steps != expected_steps or final_chain is None:
        raise RuntimeError(f"Branch {branch} has an incomplete transcript")
    if reference_local_streams is not None:
        for reference_stream in (*reference_local_streams, reference_global_stream):
            try:
                next(reference_stream)
            except StopIteration:
                continue
            raise RuntimeError(
                f"Reference branch {reference_branch} transcript is longer"
            )
    return final_chain


def _numeric_array(record, key, shape, dtype, *, positive=False):
    raw = np.asarray(record.get(key))
    if raw.shape != tuple(shape) or raw.dtype.kind not in "iuf":
        raise ValueError(f"Controller {key} numeric shape or type differs")
    value = raw.astype(dtype, copy=False)
    if np.any(~np.isfinite(value)):
        raise FloatingPointError(f"Controller {key} is nonfinite")
    if positive and np.any(value <= 0):
        raise RuntimeError(f"Controller {key} must be positive")
    return value


def _numeric_matrix(record, key, dtype, *, positive=False):
    return _numeric_array(
        record,
        key,
        (len(BLOCK_INDICES), NUM_EXPERTS),
        dtype,
        positive=positive,
    )


def _integer_matrix(record, key, *, positive=False):
    raw = np.asarray(record.get(key))
    expected_shape = (len(BLOCK_INDICES), NUM_EXPERTS)
    if raw.shape != expected_shape or raw.dtype.kind not in "iu":
        raise ValueError(f"Controller {key} integer shape or type differs")
    value = raw.astype(np.int64, copy=False)
    if positive and np.any(value <= 0):
        raise RuntimeError(f"Controller {key} must be positive")
    return value


def _numeric_vector(record, key, *, positive=False):
    return _numeric_array(
        record,
        key,
        (len(BLOCK_INDICES),),
        np.float64,
        positive=positive,
    )


def _numeric_scalar(record, key, *, positive=False):
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Controller {key} scalar type differs")
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"Controller {key} is nonfinite")
    if positive and value <= 0:
        raise RuntimeError(f"Controller {key} must be positive")
    return value


def _require_close(actual, expected, message, *, rtol=FORMULA_RTOL):
    if not np.allclose(actual, expected, rtol=rtol, atol=0.0):
        raise RuntimeError(message)


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_telemetry(artifact_root, branch, expected_gradient_norms=None):
    root = (
        Path(artifact_root).resolve()
        / "controller"
        / branch
        / "adamw_telemetry"
    )
    expected_steps = set(range(START_STEP, START_STEP + 10))
    expected_steps.update(range(302000, FINAL_STEP + 1, 1000))
    expected_names = {f"step-{step:06d}.json" for step in expected_steps}
    observed_names = {
        path.name
        for path in root.iterdir()
        if path.is_file() or path.is_symlink()
    }
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        extra = sorted(observed_names - expected_names)
        raise RuntimeError(
            f"AdamW telemetry inventory differs: missing={missing}, extra={extra}"
        )
    if expected_gradient_norms is not None and set(expected_gradient_norms) != expected_steps:
        raise RuntimeError("Controller ledger telemetry-step inventory differs")
    paths = sorted(root / name for name in expected_names)
    observed_steps = set()
    hashes = {}
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"AdamW telemetry cannot be a symbolic link: {path}")
        payload = _load_json(path)
        step = payload.get("step")
        if not isinstance(step, int) or step not in expected_steps:
            raise RuntimeError(f"Unexpected AdamW telemetry step: {path}")
        expected_fields = {
            "version",
            "step",
            "branch",
            "pre_scale_raw_gradient_squared_norm",
            "applied_raw_gradient_squared_norm",
            "parameter_delta_squared_norm",
            "moments_before",
            "moments_after",
        }
        if set(payload) != expected_fields:
            raise RuntimeError(f"AdamW telemetry fields differ: {path}")
        if (
            payload.get("version") != CONTROLLER_STATE_VERSION
            or payload.get("branch") != branch
            or step in observed_steps
        ):
            raise RuntimeError(f"AdamW telemetry identity differs: {path}")
        observed_steps.add(step)
        pre = _numeric_matrix(
            payload,
            "pre_scale_raw_gradient_squared_norm",
            np.float64,
            positive=True,
        )
        applied = _numeric_matrix(
            payload,
            "applied_raw_gradient_squared_norm",
            np.float64,
            positive=True,
        )
        if expected_gradient_norms is not None:
            expected_pre, expected_applied = expected_gradient_norms[step]
            if not np.array_equal(pre, expected_pre) or not np.array_equal(
                applied, expected_applied
            ):
                raise RuntimeError(f"AdamW telemetry/controller norms differ: {path}")
        deltas = payload.get("parameter_delta_squared_norm")
        moments_before = payload.get("moments_before")
        moments_after = payload.get("moments_after")
        if not all(
            isinstance(value, dict)
            for value in (deltas, moments_before, moments_after)
        ):
            raise ValueError(f"AdamW telemetry mappings differ: {path}")
        expected_block_keys = {str(block_index) for block_index in BLOCK_INDICES}
        if (
            set(deltas) != expected_block_keys
            or set(moments_before) != expected_block_keys
            or set(moments_after) != expected_block_keys
        ):
            raise ValueError(f"AdamW telemetry block inventory differs: {path}")
        for block_index in BLOCK_INDICES:
            block_key = str(block_index)
            delta = np.asarray(deltas.get(block_key), dtype=np.float64)
            if delta.shape != (NUM_EXPERTS,) or np.any(~np.isfinite(delta)):
                raise ValueError(f"AdamW telemetry delta differs: {path}")
            if np.any(delta < 0):
                raise RuntimeError(f"AdamW telemetry delta is negative: {path}")
            for moments in (moments_before, moments_after):
                rows = moments.get(block_key)
                if not isinstance(rows, list) or len(rows) != NUM_EXPERTS:
                    raise ValueError(f"AdamW telemetry moment shape differs: {path}")
                for row in rows:
                    if row.get("exposed") is not True:
                        raise RuntimeError(f"AdamW moment was not exposed: {path}")
                    values = (
                        row.get("first_moment_squared_norm"),
                        row.get("second_moment_squared_norm"),
                    )
                    if any(
                        not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                        for value in values
                    ):
                        raise FloatingPointError(f"AdamW moment is invalid: {path}")
        hashes[path.name] = sha256_file(path)
    if observed_steps != expected_steps:
        missing = sorted(expected_steps - observed_steps)
        extra = sorted(observed_steps - expected_steps)
        raise RuntimeError(f"AdamW telemetry coverage differs: missing={missing}, extra={extra}")
    return hashes


def validate_controller_artifacts(
    artifact_root,
    branch,
    checkpoint,
    start_step=START_STEP,
):
    if branch not in BRANCHES:
        raise ValueError(f"Unknown controller branch: {branch}")
    if checkpoint.get("step") != FINAL_STEP:
        raise ValueError("Controller artifact checkpoint is not the final step")
    extension = checkpoint.get(CHECKPOINT_STATE_KEY)
    if not isinstance(extension, dict):
        raise ValueError("Controller artifact checkpoint lacks controller state")
    expected_steps = FINAL_STEP - int(start_step) + 1
    expected_metadata = {
        "version": CONTROLLER_STATE_VERSION,
        "branch": branch,
        "execution_mode": "continuation",
        "block_indices": list(BLOCK_INDICES),
        "num_experts": NUM_EXPERTS,
        "start_step": int(start_step),
        "last_step": FINAL_STEP,
        "update_count": expected_steps,
    }
    if any(extension.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError(f"Branch {branch} checkpoint controller metadata differs")
    counters = extension.get("numerical_counters")
    if (
        not isinstance(counters, dict)
        or set(counters) != NUMERICAL_COUNTER_NAMES
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value != 0
            for value in counters.values()
        )
    ):
        raise RuntimeError(f"Branch {branch} checkpoint numerical counters differ")
    normalizer = extension.get("normalizer")
    if not isinstance(normalizer, dict) or set(normalizer) != {
        "ema",
        "initialized",
        "ema_decay",
        "epsilon",
    }:
        raise ValueError(f"Branch {branch} checkpoint normalizer fields differ")
    checkpoint_ema_tensor = normalizer.get("ema")
    checkpoint_initialized = normalizer.get("initialized")
    expected_shape = (len(BLOCK_INDICES), NUM_EXPERTS)
    if (
        normalizer.get("ema_decay") != 0.99
        or normalizer.get("epsilon") != 1e-30
        or not torch.is_tensor(checkpoint_ema_tensor)
        or checkpoint_ema_tensor.dtype != torch.float64
        or tuple(checkpoint_ema_tensor.shape) != expected_shape
        or not bool(torch.isfinite(checkpoint_ema_tensor).all())
        or not bool(torch.all(checkpoint_ema_tensor > 0))
        or not torch.is_tensor(checkpoint_initialized)
        or checkpoint_initialized.dtype != torch.bool
        or tuple(checkpoint_initialized.shape) != expected_shape
        or not bool(torch.all(checkpoint_initialized))
    ):
        raise ValueError(f"Branch {branch} checkpoint normalizer state differs")
    ledger_path = (
        Path(artifact_root).resolve()
        / "controller"
        / branch
        / "steps.jsonl"
    )
    transcript_path = (
        Path(artifact_root).resolve()
        / "transcripts"
        / branch
        / "global.jsonl"
    )
    if ledger_path.is_symlink() or transcript_path.is_symlink():
        raise RuntimeError("Controller and transcript ledgers cannot be symbolic links")
    controller_stream = _iter_validated_ledger(ledger_path, start_step)
    transcript_stream = _iter_validated_ledger(transcript_path, start_step)
    observed_steps = 0
    final_chain = None
    final_ema = None
    recomputed_ema = None
    telemetry_norms = {}
    for controller_record, transcript_record in zip_longest(
        controller_stream, transcript_stream
    ):
        if controller_record is None or transcript_record is None:
            raise RuntimeError(f"Branch {branch} controller/transcript lengths differ")
        step = start_step + observed_steps
        update_index = observed_steps
        if set(controller_record) != CONTROLLER_RECORD_FIELDS:
            raise RuntimeError(f"Branch {branch} controller fields differ at {step}")
        if (
            controller_record.get("version") != CONTROLLER_STATE_VERSION
            or controller_record.get("branch") != branch
            or controller_record.get("step") != step
            or controller_record.get("update_index") != update_index
            or controller_record.get("global_transcript_digest")
            != transcript_record.get("global_digest")
            or not _is_sha256(controller_record.get("global_transcript_digest"))
            or not _is_sha256(controller_record.get("rank_consensus_digest"))
        ):
            raise RuntimeError(f"Branch {branch} controller identity differs at {step}")
        expected_offset = 1 + (update_index % 11)
        if controller_record.get("permutation_offset") != expected_offset:
            raise RuntimeError(f"Branch {branch} permutation offset differs at {step}")
        credit = _numeric_matrix(
            controller_record, "global_credit", np.float64, positive=True
        )
        count = _integer_matrix(controller_record, "global_count", positive=True)
        rates = credit / count.astype(np.float64)
        if np.any(~np.isfinite(rates)) or np.any(rates <= 0):
            raise FloatingPointError(f"Branch {branch} credit rate differs at {step}")
        if recomputed_ema is None:
            recomputed_ema = rates.copy()
        else:
            recomputed_ema = 0.99 * recomputed_ema + 0.01 * rates
        ema = _numeric_matrix(
            controller_record, "credit_rate_ema", np.float64, positive=True
        )
        _require_close(
            ema,
            recomputed_ema,
            f"Branch {branch} EMA recurrence differs at {step}",
        )
        reference = np.exp(np.log(recomputed_ema).mean(axis=1))
        expected_raw = np.sqrt(
            reference[:, None] / np.maximum(recomputed_ema, 1e-30)
        )
        expected_raw = np.clip(expected_raw, 0.5, 2.0)
        raw = _numeric_matrix(
            controller_record, "raw_scales", np.float64, positive=True
        )
        _require_close(
            raw,
            expected_raw,
            f"Branch {branch} raw-scale formula differs at {step}",
        )
        permuted = _numeric_matrix(
            controller_record, "permuted_scales", np.float64, positive=True
        )
        factors = _numeric_vector(
            controller_record,
            "selected_budget_factors",
            positive=True,
        )
        applied = _numeric_matrix(
            controller_record, "applied_scales", np.float64, positive=True
        )
        pre = _numeric_matrix(
            controller_record,
            "pre_gradient_squared_norm",
            np.float64,
            positive=True,
        )
        post = _numeric_matrix(
            controller_record,
            "post_gradient_squared_norm",
            np.float64,
            positive=True,
        )
        expected_permuted = np.take(
            raw,
            (np.arange(NUM_EXPERTS) + expected_offset) % NUM_EXPERTS,
            axis=1,
        )
        if not np.array_equal(permuted, expected_permuted):
            raise RuntimeError(f"Branch {branch} permuted scales differ at {step}")
        if branch == "measure_only_control":
            if not np.array_equal(factors, np.ones_like(factors)):
                raise RuntimeError("Measure-only budget factors are not exactly one")
            if not np.array_equal(applied, np.ones_like(applied)):
                raise RuntimeError("Measure-only applied scales are not exactly one")
            if not np.array_equal(pre, post):
                raise RuntimeError("Measure-only gradients changed")
            selected = np.ones_like(raw)
        elif branch == "rotating_permuted_scale_control":
            selected = permuted
        else:
            selected = raw
        expected_factors = np.sqrt(
            pre.sum(axis=1) / (pre * np.square(selected)).sum(axis=1)
        )
        if branch == "measure_only_control":
            if not np.array_equal(expected_factors, np.ones_like(expected_factors)):
                raise RuntimeError("Measure-only recomputed budget factors differ")
        else:
            _require_close(
                factors,
                expected_factors,
                f"Branch {branch} budget-factor formula differs at {step}",
            )
        expected_applied = selected * expected_factors[:, None]
        _require_close(
            applied,
            expected_applied,
            f"Branch {branch} applied-scale formula differs at {step}",
        )
        expected_post = pre * np.square(applied)
        _require_close(
            post,
            expected_post,
            f"Branch {branch} expert gradient scaling differs at {step}",
            rtol=GRADIENT_SCALE_RTOL,
        )
        recorded_drifts = _numeric_vector(
            controller_record,
            "block_relative_budget_drift",
        )
        pre_block_totals = pre.sum(1)
        actual_drifts = np.abs(post.sum(1) - pre_block_totals) / pre_block_totals
        _require_close(
            recorded_drifts,
            actual_drifts,
            f"Branch {branch} recorded block budget drift differs at {step}",
        )
        if (
            np.any(recorded_drifts < 0)
            or np.any(recorded_drifts > 1e-6)
            or np.any(actual_drifts > 1e-6)
        ):
            raise RuntimeError(f"Branch {branch} block budget drift differs at {step}")
        full_before = _numeric_scalar(
            controller_record,
            "full_pre_gradient_squared_norm",
            positive=True,
        )
        full_after = _numeric_scalar(
            controller_record,
            "full_post_gradient_squared_norm",
            positive=True,
        )
        full_drift = _numeric_scalar(
            controller_record,
            "full_relative_budget_drift",
        )
        actual_full_drift = abs(full_after - full_before) / full_before
        _require_close(
            full_drift,
            actual_full_drift,
            f"Branch {branch} recorded full budget drift differs at {step}",
        )
        if full_drift < 0 or full_drift > 1e-6 or actual_full_drift > 1e-6:
            raise RuntimeError(f"Branch {branch} full budget drift differs at {step}")
        if (
            full_before < pre.sum() * (1.0 - FORMULA_RTOL)
            or full_after < post.sum() * (1.0 - FORMULA_RTOL)
        ):
            raise RuntimeError(f"Branch {branch} full gradient budget is incomplete at {step}")
        if branch == "measure_only_control" and full_before != full_after:
            raise RuntimeError("Measure-only full-model gradients changed")
        if step <= START_STEP + 9 or step % 1000 == 0:
            telemetry_norms[step] = (pre.copy(), post.copy())
        final_chain = controller_record["chain_digest"]
        final_ema = ema
        observed_steps += 1
    if observed_steps != expected_steps or final_chain is None:
        raise RuntimeError(f"Branch {branch} controller ledger is incomplete")
    if extension.get("update_count") != observed_steps:
        raise RuntimeError(f"Branch {branch} controller checkpoint/ledger mismatch")
    checkpoint_ema = checkpoint_ema_tensor.detach().cpu().numpy()
    if (
        final_ema is None
        or recomputed_ema is None
        or not np.array_equal(final_ema, checkpoint_ema)
    ):
        raise RuntimeError(f"Branch {branch} ledger/checkpoint normalizer mismatch")
    _require_close(
        checkpoint_ema,
        recomputed_ema,
        f"Branch {branch} recomputed/checkpoint normalizer mismatch",
    )
    return {
        "controller_final_chain_digest": final_chain,
        "controller_ledger_file_sha256": sha256_file(ledger_path),
        "adamw_telemetry_file_sha256": _validate_telemetry(
            artifact_root,
            branch,
            telemetry_norms,
        ),
    }


def load_checkpoint_state_into_model(model, state_dict):
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.eval().requires_grad_(False)


def evaluate_checkpoint_cases(
    config_path,
    checkpoint_path,
    checkpoint_sha256,
    branch,
    cases,
    tensor_dir,
    output_root,
    protocol_sha256,
    device,
):
    if branch not in BRANCHES:
        raise ValueError(f"Unknown held-out branch: {branch}")
    runtime_cfg = load_runtime_cfg(Path(config_path))
    if runtime_cfg.model_name != "ProMoE_TC_B":
        raise ValueError("Held-out evaluator requires the sealed Base model")
    torch.cuda.set_device(device)
    model = _build_model(runtime_cfg).to(device).eval().requires_grad_(False)
    checkpoint, observed_sha256 = validate_branch_checkpoint(
        checkpoint_path,
        checkpoint_sha256,
        branch,
        verify_file_hash=True,
        reference_model=model,
    )
    if observed_sha256 != checkpoint_sha256:
        raise RuntimeError("Worker checkpoint binding changed")
    capture = EvaluationCapture(model)
    completed = []
    try:
        for state_name in CHECKPOINT_STATES:
            load_checkpoint_state_into_model(model, checkpoint[state_name])
            for case in cases:
                expected = {
                    "version": EVALUATOR_VERSION,
                    "branch": branch,
                    "checkpoint_state": state_name,
                    "checkpoint_sha256": checkpoint_sha256,
                    "protocol_sha256": protocol_sha256,
                    "case_index": int(case["index"]),
                    "label": int(case["label"]),
                    "relative_path": case["relative_path"],
                }
                path = _case_artifact_path(
                    output_root, branch, state_name, case["index"]
                )
                reused = _load_reusable_case(path, expected, protocol_sha256)
                if reused is None:
                    result = evaluate_case(model, capture, case, tensor_dir)
                    payload = {**result, **expected}
                    _publish_case(path, payload, protocol_sha256)
                completed.append((state_name, int(case["index"])))
    finally:
        capture.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return completed


def _sealed_replay_device():
    visible_by_physical = dict(_sealed_gpu_device_pairs())
    try:
        visible_index = visible_by_physical[SEALED_GPU_IDS[0]]
    except KeyError as error:
        raise RuntimeError("Sealed replay GPU mapping is incomplete") from error
    return torch.device(f"cuda:{visible_index}")


def validate_protocol_for_evaluation(protocol_path):
    protocol_path = Path(protocol_path).resolve()
    protocol = _load_json(protocol_path)
    protocol_sha256 = canonical_json_sha256(protocol)
    sidecar = protocol_path.with_suffix(".sha256")
    if sidecar.read_text(encoding="utf-8") != protocol_sha256 + "\n":
        raise RuntimeError("Generated protocol sidecar mismatch")
    if protocol.get("status") != "immutable_pre_efficacy":
        raise ValueError("Generated protocol is not sealed for evaluation")
    branches = protocol.get("branches")
    if [entry.get("name") for entry in branches] != list(BRANCHES):
        raise ValueError("Generated protocol branch order differs")
    heldout_path = protocol["heldout"]["manifest_path"]
    manifest, manifest_sha256 = load_heldout_manifest(heldout_path)
    if manifest_sha256 != protocol["heldout"]["manifest_canonical_sha256"]:
        raise RuntimeError("Generated protocol held-out binding differs")
    reference_cfg = load_runtime_cfg(Path(branches[0]["config_path"]))
    reference_model = _build_model(reference_cfg).cpu().eval()
    checkpoint_specs = {}
    transcript_chains = {}
    branch_integrity = {}
    trainer_state_digests = {}
    try:
        for entry in branches:
            checkpoint, checkpoint_sha256 = validate_branch_checkpoint(
                entry["final_checkpoint_path"],
                None,
                entry["name"],
                reference_model=reference_model,
            )
            checkpoint_specs[entry["name"]] = {
                "path": str(Path(entry["final_checkpoint_path"]).resolve()),
                "sha256": checkpoint_sha256,
            }
            trainer_state_digests[entry["name"]] = canonical_state_sha256(
                checkpoint["trainer_state"]
            )
            branch_integrity[entry["name"]] = validate_controller_artifacts(
                entry["artifact_root"], entry["name"], checkpoint
            )
            del checkpoint
            gc.collect()
            # Reconstruct the input stream for every arm.  Comparing a branch
            # to a reference ledger alone would allow a synchronized rewrite
            # of hidden tensor commitments to pass its hash chain.
            replay_context = {
                "initial_checkpoint_path": protocol["frozen_checkpoint"]["path"],
                "expected_checkpoint_sha256": protocol["frozen_checkpoint"][
                    "file_sha256"
                ],
                "runtime_cfg": reference_cfg,
                "dataset_root": protocol["dataset"]["latent_root"],
                "device": _sealed_replay_device(),
            }
            transcript_chains[entry["name"]] = validate_branch_transcripts(
                entry["artifact_root"],
                entry["name"],
                reference_artifact_root=(
                    branches[0]["artifact_root"]
                    if entry["name"] != BRANCHES[0]
                    else None
                ),
                reference_branch=BRANCHES[0],
                replay_context=replay_context,
            )
    finally:
        del reference_model
        gc.collect()
    if len(set(transcript_chains.values())) != 1:
        raise RuntimeError("Three branch input transcripts are not identical")
    if len(set(trainer_state_digests.values())) != 1:
        raise RuntimeError("Three branch trainer states are not identical")
    return (
        protocol,
        protocol_sha256,
        manifest,
        checkpoint_specs,
        transcript_chains,
        branch_integrity,
        trainer_state_digests,
    )


def _validated_case_artifact_inventory(
    output_root,
    protocol_sha256,
    checkpoint_specs,
    *,
    case_count=HELDOUT_CASE_COUNT,
):
    output_root = Path(output_root).resolve()
    raw_root = output_root / "raw"
    metric_root = output_root / "sealed"
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count <= 0:
        raise ValueError("Held-out case count must be a positive integer")
    if set(checkpoint_specs) != set(BRANCHES):
        raise RuntimeError("Held-out checkpoint-spec branch inventory differs")
    for branch, spec in checkpoint_specs.items():
        if (
            not isinstance(spec, dict)
            or set(spec) != {"path", "sha256"}
            or not isinstance(spec["path"], str)
            or not _is_sha256(spec["sha256"])
        ):
            raise RuntimeError(f"Held-out checkpoint spec differs: {branch}")
    if (
        not raw_root.is_dir()
        or raw_root.is_symlink()
        or not metric_root.is_dir()
        or metric_root.is_symlink()
    ):
        raise RuntimeError("Held-out raw/sealed artifact roots are absent or indirect")

    expected_directories = {Path("raw"), Path("sealed")}
    expected_pairs = []
    for branch in BRANCHES:
        branch_dir = raw_root / branch
        expected_directories.add(Path("raw") / branch)
        for state_name in CHECKPOINT_STATES:
            state_dir = branch_dir / state_name
            expected_directories.add(Path("raw") / branch / state_name)
            expected_directories.add(Path("sealed") / branch)
            expected_directories.add(Path("sealed") / branch / state_name)
            for case_index in range(case_count):
                case_path = _case_artifact_path(
                    output_root,
                    branch,
                    state_name,
                    case_index,
                )
                expected_pairs.append(
                    (
                        branch,
                        state_name,
                        case_index,
                        case_path,
                        _seal_path(case_path),
                        _case_metric_artifact_path(
                            output_root, branch, state_name, case_index
                        ),
                        _seal_path(
                            _case_metric_artifact_path(
                                output_root, branch, state_name, case_index
                            )
                        ),
                    )
                )

    expected_files = {
        path.relative_to(output_root)
        for pair in expected_pairs
        for path in pair[-4:]
    }
    observed_files = {
        path.relative_to(output_root)
        for root in (raw_root, metric_root)
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    observed_directories = {Path("raw"), Path("sealed")} | {
        path.relative_to(output_root)
        for root in (raw_root, metric_root)
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if observed_files != expected_files or observed_directories != expected_directories:
        missing = sorted(str(path) for path in expected_files - observed_files)
        extra = sorted(str(path) for path in observed_files - expected_files)
        raise RuntimeError(
            "Held-out case artifact inventory differs: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    case_hashes = {}
    seal_hashes = {}
    metric_hashes = {}
    metric_seal_hashes = {}
    for (
        branch,
        state_name,
        case_index,
        case_path,
        seal_path,
        metric_path,
        metric_seal_path,
    ) in expected_pairs:
        if any(
            item.is_symlink()
            for item in (case_path, seal_path, metric_path, metric_seal_path)
        ):
            raise RuntimeError(f"Held-out artifacts cannot be symbolic links: {case_path}")
        if any(
            item.stat().st_mode & 0o222
            for item in (case_path, seal_path, metric_path, metric_seal_path)
        ):
            raise RuntimeError(f"Held-out artifacts must be read-only: {case_path}")
        expected_metadata = {
            "version": EVALUATOR_VERSION,
            "branch": branch,
            "checkpoint_state": state_name,
            "checkpoint_sha256": checkpoint_specs[branch]["sha256"],
            "protocol_sha256": protocol_sha256,
            "case_index": case_index,
        }
        _load_reusable_case(case_path, expected_metadata, protocol_sha256)
        relative_case = case_path.relative_to(output_root).as_posix()
        relative_seal = seal_path.relative_to(output_root).as_posix()
        relative_metric = metric_path.relative_to(output_root).as_posix()
        relative_metric_seal = metric_seal_path.relative_to(output_root).as_posix()
        case_hashes[relative_case] = sha256_file(case_path)
        seal_hashes[relative_seal] = sha256_file(seal_path)
        metric_hashes[relative_metric] = sha256_file(metric_path)
        metric_seal_hashes[relative_metric_seal] = sha256_file(metric_seal_path)
    return case_hashes, seal_hashes, metric_hashes, metric_seal_hashes


def publish_evaluation_complete(
    output_root,
    protocol_sha256,
    manifest_sha256,
    transcript_chains,
    checkpoint_specs,
    branch_integrity,
    trainer_state_digests,
):
    output_root = Path(output_root).resolve()
    if not _is_sha256(protocol_sha256) or not _is_sha256(manifest_sha256):
        raise RuntimeError("Held-out completion protocol or manifest hash is malformed")
    binding_maps = {
        "transcript chains": transcript_chains,
        "branch integrity": branch_integrity,
        "trainer states": trainer_state_digests,
    }
    for name, value in binding_maps.items():
        if not isinstance(value, dict) or set(value) != set(BRANCHES):
            raise RuntimeError(f"Held-out {name} branch inventory differs")
    if (
        any(not _is_sha256(value) for value in transcript_chains.values())
        or len(set(transcript_chains.values())) != 1
        or any(not _is_sha256(value) for value in trainer_state_digests.values())
        or len(set(trainer_state_digests.values())) != 1
    ):
        raise RuntimeError("Held-out transcript or trainer-state binding differs")
    expected_telemetry_names = {
        f"step-{step:06d}.json"
        for step in (
            list(range(START_STEP, START_STEP + 10))
            + list(range(302000, FINAL_STEP + 1, 1000))
        )
    }
    for branch, integrity in branch_integrity.items():
        if not isinstance(integrity, dict) or set(integrity) != {
            "controller_final_chain_digest",
            "controller_ledger_file_sha256",
            "adamw_telemetry_file_sha256",
        }:
            raise RuntimeError(f"Held-out branch-integrity fields differ: {branch}")
        telemetry = integrity["adamw_telemetry_file_sha256"]
        if (
            not _is_sha256(integrity["controller_final_chain_digest"])
            or not _is_sha256(integrity["controller_ledger_file_sha256"])
            or not isinstance(telemetry, dict)
            or set(telemetry) != expected_telemetry_names
            or any(not _is_sha256(value) for value in telemetry.values())
        ):
            raise RuntimeError(f"Held-out branch-integrity binding differs: {branch}")
    (
        case_hashes,
        seal_hashes,
        metric_hashes,
        metric_seal_hashes,
    ) = _validated_case_artifact_inventory(
        output_root,
        protocol_sha256,
        checkpoint_specs,
    )
    expected_count = len(BRANCHES) * len(CHECKPOINT_STATES) * HELDOUT_CASE_COUNT
    payload = {
        "version": EVALUATOR_VERSION,
        "status": "complete_without_efficacy_aggregation",
        "protocol_sha256": protocol_sha256,
        "heldout_manifest_canonical_sha256": manifest_sha256,
        "transcript_final_chain_digests": copy.deepcopy(transcript_chains),
        "checkpoint_file_sha256": {
            branch: spec["sha256"] for branch, spec in checkpoint_specs.items()
        },
        "trainer_state_sha256": copy.deepcopy(trainer_state_digests),
        "branch_integrity": copy.deepcopy(branch_integrity),
        "case_file_count": expected_count,
        "case_file_sha256": case_hashes,
        "case_seal_file_sha256": seal_hashes,
        "metric_file_sha256": metric_hashes,
        "metric_seal_file_sha256": metric_seal_hashes,
    }
    path = output_root / "evaluation-complete.json"
    seal_path = _seal_path(path)
    seal = {
        "version": 1,
        "artifact": path.name,
        "artifact_canonical_sha256": canonical_json_sha256(payload),
        "protocol_sha256": protocol_sha256,
    }
    if path.exists() or seal_path.exists():
        if not path.exists() or not seal_path.exists():
            raise RuntimeError("Held-out completion artifact pair is incomplete")
        if _load_json(path) != payload or _load_json(seal_path) != seal:
            raise RuntimeError("Existing held-out completion manifest differs")
    else:
        atomic_write_json(path, payload, mode=0o444)
        atomic_write_json(seal_path, seal, mode=0o444)
    os.chmod(path, 0o444)
    os.chmod(seal_path, 0o444)
    return path
