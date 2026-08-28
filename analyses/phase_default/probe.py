"""Test whether diffusion phase is needed for Default-MoE output sketches."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyses.denoising_regret.probe import (
    _compute_router,
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


PROBE_VERSION = 1
MANIFEST_NAME = "phase_default_gate_v1"
DEFAULT_BLOCK_INDICES = (1, 5, 11)
APPROXIMATION_NAMES = ("zero", "global", "phase", "shuffled_phase")
CONTRAST_KEYS = (
    "phase_vs_global_output_error_reduction",
    "phase_minus_global_missing_score_cosine",
    "phase_minus_global_center_cosine",
    "phase_vs_shuffled_output_error_reduction",
    "phase_minus_shuffled_missing_score_cosine",
    "phase_minus_shuffled_center_cosine",
)
PHASE_QUALITY_KEYS = (
    "phase_output_relative_squared_error",
    "phase_missing_score_cosine",
    "phase_center_cosine",
)
SUMMARY_KEYS = (*CONTRAST_KEYS, *PHASE_QUALITY_KEYS)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_finite(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _validate_protocol(protocol):
    required = {
        "model_name",
        "checkpoint_step",
        "checkpoint_state",
        "checkpoint_size",
        "checkpoint_sha256",
        "sigmas",
        "block_indices",
        "tokens_per_cell",
        "phase_shuffle_offset",
        "calibration_cases",
        "confirmatory_cases",
        "bootstrap_resamples",
        "bootstrap_seed",
        "minimum_phase_buffer_coverage",
        "gates",
    }
    if set(protocol) != required:
        raise ValueError(
            "Manifest protocol keys differ from the locked contract: "
            f"expected={sorted(required)}, got={sorted(protocol)}"
        )
    if protocol["model_name"] != "ProMoE_TC_B":
        raise ValueError("The phase-default gate is locked to ProMoE_TC_B")
    _require_int(protocol["checkpoint_step"], "checkpoint_step", 1)
    if protocol["checkpoint_state"] != "ema_model_state_dict":
        raise ValueError("The gate must use EMA checkpoint weights")
    _require_int(protocol["checkpoint_size"], "checkpoint_size", 1)
    checkpoint_hash = protocol["checkpoint_sha256"]
    if (
        not isinstance(checkpoint_hash, str)
        or len(checkpoint_hash) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_hash)
    ):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA256 digest")

    sigmas = tuple(_require_finite(value, "sigma", 0.0, 1.0) for value in protocol["sigmas"])
    if (
        len(sigmas) < 3
        or len(sigmas) != len(set(sigmas))
        or any(value in {0.0, 1.0} for value in sigmas)
        or tuple(sorted(sigmas)) != sigmas
    ):
        raise ValueError("sigmas must be at least three unique, ordered interior values")
    blocks = tuple(_require_int(value, "block_index") for value in protocol["block_indices"])
    if not blocks or len(blocks) != len(set(blocks)):
        raise ValueError("block_indices must be nonempty and unique")
    _require_int(protocol["tokens_per_cell"], "tokens_per_cell", 2)
    offset = _require_int(protocol["phase_shuffle_offset"], "phase_shuffle_offset", 1)
    if offset >= len(sigmas):
        raise ValueError("phase_shuffle_offset must be smaller than the phase count")
    _require_int(protocol["calibration_cases"], "calibration_cases", 2)
    _require_int(protocol["confirmatory_cases"], "confirmatory_cases", 2)
    _require_int(protocol["bootstrap_resamples"], "bootstrap_resamples", 1000)
    _require_int(protocol["bootstrap_seed"], "bootstrap_seed")
    _require_finite(
        protocol["minimum_phase_buffer_coverage"],
        "minimum_phase_buffer_coverage",
        0.0,
        1.0,
    )

    required_gates = {
        "maximum_mean_phase_output_relative_squared_error",
        "maximum_ucb_phase_output_relative_squared_error",
        "minimum_mean_phase_missing_score_cosine",
        "minimum_lcb_phase_missing_score_cosine",
        "minimum_mean_phase_center_cosine",
        "minimum_lcb_phase_center_cosine",
        "minimum_mean_phase_vs_global_output_error_reduction",
        "minimum_lcb_phase_vs_global_output_error_reduction",
        "minimum_mean_phase_minus_global_missing_score_cosine",
        "minimum_lcb_phase_minus_global_missing_score_cosine",
        "minimum_mean_phase_minus_global_center_cosine",
        "minimum_lcb_phase_minus_global_center_cosine",
        "minimum_lcb_phase_vs_shuffled_output_error_reduction",
        "minimum_lcb_phase_minus_shuffled_missing_score_cosine",
        "minimum_lcb_phase_minus_shuffled_center_cosine",
    }
    gates = protocol["gates"]
    if set(gates) != required_gates:
        raise ValueError("Manifest gate keys differ from the locked contract")
    for name, value in gates.items():
        _require_finite(value, name, -1.0, 1.0)
    return sigmas, blocks


def load_manifest(manifest_path, latent_root):
    manifest_path = Path(manifest_path).resolve()
    latent_root = Path(latent_root).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    if not latent_root.is_dir():
        raise NotADirectoryError(f"Latent root does not exist: {latent_root}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(payload) != {"version", "name", "selection", "protocol", "cases"}:
        raise ValueError("Manifest top-level keys differ from the locked schema")
    if payload["version"] != 1 or payload["name"] != MANIFEST_NAME:
        raise ValueError("Unexpected phase-default manifest version or name")
    selection = payload["selection"]
    if not isinstance(selection, dict) or not selection.get(
        "locked_before_any_phase_default_result"
    ):
        raise ValueError("Manifest must declare pre-result locking")
    sigmas, blocks = _validate_protocol(payload["protocol"])

    class_names = sorted(
        path.name
        for path in latent_root.iterdir()
        if path.is_dir()
        and len(path.name) == 9
        and path.name.startswith("n")
        and path.name[1:].isdigit()
    )
    if len(class_names) != 1000:
        raise ValueError(
            f"Expected 1000 ImageNet latent classes, found {len(class_names)}"
        )

    expected_case_keys = {
        "split", "id", "label", "seed", "synset", "latent", "latent_sha256"
    }
    split_counts = defaultdict(int)
    seen_ids = set()
    seen_labels = set()
    seen_latents = set()
    cases = []
    for index, raw_case in enumerate(payload["cases"]):
        if not isinstance(raw_case, dict) or set(raw_case) != expected_case_keys:
            raise ValueError(f"Manifest case {index} has an invalid schema")
        split = raw_case["split"]
        if split not in {"calibration", "confirmatory"}:
            raise ValueError(f"Manifest case {index} has unknown split {split!r}")
        case_id = raw_case["id"]
        if (
            not isinstance(case_id, str)
            or not case_id
            or any(not (char.isalnum() or char in "_-") for char in case_id)
        ):
            raise ValueError(f"Manifest case {index} has an invalid id")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen_ids.add(case_id)
        label = _require_int(raw_case["label"], f"{case_id}.label")
        if label >= len(class_names) or raw_case["synset"] != class_names[label]:
            raise ValueError(f"{case_id} label and synset do not match ImageNet order")
        if label in seen_labels:
            raise ValueError(f"Duplicate case label: {label}")
        seen_labels.add(label)
        _require_int(raw_case["seed"], f"{case_id}.seed")
        relative = Path(raw_case["latent"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{case_id} latent must be a safe relative path")
        latent_path = (latent_root / relative).resolve()
        if latent_root not in latent_path.parents or not latent_path.is_file():
            raise FileNotFoundError(f"{case_id} latent does not exist: {latent_path}")
        if latent_path.parent.name != raw_case["synset"]:
            raise ValueError(f"{case_id} latent parent differs from its synset")
        if latent_path in seen_latents:
            raise ValueError(f"Duplicate latent path: {latent_path}")
        seen_latents.add(latent_path)
        expected_hash = raw_case["latent_sha256"]
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"{case_id} latent_sha256 is invalid")
        actual_hash = sha256_file(latent_path)
        if actual_hash != expected_hash:
            raise ValueError(f"{case_id} latent SHA256 changed")
        split_counts[split] += 1
        cases.append({**raw_case, "latent_path": str(latent_path)})

    protocol = payload["protocol"]
    expected_counts = {
        "calibration": protocol["calibration_cases"],
        "confirmatory": protocol["confirmatory_cases"],
    }
    if dict(split_counts) != expected_counts:
        raise ValueError(
            f"Manifest split counts differ: expected={expected_counts}, "
            f"got={dict(split_counts)}"
        )
    return {
        **payload,
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "sigmas": sigmas,
        "block_indices": blocks,
        "cases": cases,
    }


class MultiMoeCapture:
    """Capture several MoE inputs and their native suffix gradients in one pass."""

    def __init__(self, model, block_indices):
        self.block_indices = tuple(sorted(int(index) for index in block_indices))
        self.first_block = self.block_indices[0]
        self.enabled = False
        self.gradient_mode = False
        self.hidden_states = {}
        self.labels = {}
        self.outputs = {}
        self._handles = []
        for block_index in self.block_indices:
            moe_layer = model.blocks[block_index].mlp
            self._handles.append(moe_layer.register_forward_pre_hook(
                self._pre_hook(block_index)
            ))
            self._handles.append(moe_layer.register_forward_hook(
                self._post_hook(block_index)
            ))

    def _pre_hook(self, block_index):
        def capture(module, inputs):
            if not self.enabled:
                return None
            if block_index in self.hidden_states:
                raise RuntimeError(f"MoE block {block_index} ran twice in one forward")
            if len(inputs) < 2:
                raise RuntimeError("Expected SparseMoeBlock inputs and labels")
            self.hidden_states[block_index] = inputs[0].detach()
            self.labels[block_index] = inputs[1].detach()
            return None

        return capture

    def _post_hook(self, block_index):
        def capture(module, inputs, output):
            if not self.enabled or not self.gradient_mode:
                return None
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("Expected SparseMoeBlock to return (output, loss)")
            moe_output, auxiliary_loss = output
            if auxiliary_loss is not None:
                raise RuntimeError("Frozen eval capture expected no auxiliary loss")
            if block_index == self.first_block:
                moe_output = moe_output.detach().requires_grad_(True)
                self.outputs[block_index] = moe_output
                return moe_output, auxiliary_loss
            if not moe_output.requires_grad:
                raise RuntimeError(
                    f"MoE block {block_index} output is detached from the suffix graph"
                )
            self.outputs[block_index] = moe_output
            return None

        return capture

    def start(self, gradient_mode=False):
        if self.enabled:
            raise RuntimeError("Capture is already active")
        self.enabled = True
        self.gradient_mode = bool(gradient_mode)
        self.hidden_states = {}
        self.labels = {}
        self.outputs = {}

    def stop(self):
        self.enabled = False
        self.gradient_mode = False

    def validate_complete(self):
        expected = set(self.block_indices)
        if set(self.hidden_states) != expected or set(self.labels) != expected:
            raise RuntimeError("Not every requested MoE block captured its inputs")
        if self.gradient_mode and set(self.outputs) != expected:
            raise RuntimeError("Not every requested MoE block captured its output")

    def suffix_gradients(self, loss):
        self.validate_complete()
        ordered_outputs = [self.outputs[index] for index in self.block_indices]
        gradients = torch.autograd.grad(loss, ordered_outputs)
        return {
            block_index: gradient.detach()
            for block_index, gradient in zip(self.block_indices, gradients)
        }

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


class DefaultSketchAccumulator:
    """Accumulate selected-expert output means globally and by phase."""

    def __init__(self, blocks, num_phases, num_experts, hidden_size):
        shape = (num_experts, hidden_size)
        phase_shape = (num_phases, num_experts, hidden_size)
        self.global_sums = {
            block: torch.zeros(shape, dtype=torch.float64) for block in blocks
        }
        self.global_counts = {
            block: torch.zeros(num_experts, dtype=torch.int64) for block in blocks
        }
        self.phase_sums = {
            block: torch.zeros(phase_shape, dtype=torch.float64) for block in blocks
        }
        self.phase_counts = {
            block: torch.zeros((num_phases, num_experts), dtype=torch.int64)
            for block in blocks
        }

    def update(self, block, phase_index, hidden_states, route_ids, experts):
        if hidden_states.ndim != 2 or route_ids.ndim != 1:
            raise ValueError("Sketch updates expect flat token inputs and route IDs")
        if hidden_states.shape[0] != route_ids.numel():
            raise ValueError("Sketch token inputs and route IDs must align")
        if len(experts) != self.global_counts[block].numel():
            raise ValueError("Expert count differs from the sketch contract")
        with torch.no_grad():
            for expert_id, expert in enumerate(experts):
                selected = route_ids == expert_id
                count = int(selected.sum().item())
                if count == 0:
                    continue
                output = expert(hidden_states[selected]).double().sum(dim=0).cpu()
                self.global_sums[block][expert_id] += output
                self.global_counts[block][expert_id] += count
                self.phase_sums[block][phase_index, expert_id] += output
                self.phase_counts[block][phase_index, expert_id] += count

    @staticmethod
    def _means(sums, counts):
        means = torch.zeros_like(sums)
        populated = counts > 0
        means[populated] = sums[populated] / counts[populated].double().unsqueeze(-1)
        return means

    def finalize(self):
        global_defaults = {
            block: self._means(self.global_sums[block], self.global_counts[block])
            for block in self.global_sums
        }
        phase_defaults = {
            block: self._means(self.phase_sums[block], self.phase_counts[block])
            for block in self.phase_sums
        }
        return {
            "global_defaults": global_defaults,
            "phase_defaults": phase_defaults,
            "global_counts": self.global_counts,
            "phase_counts": self.phase_counts,
        }


def _validate_model_contract(model, block_indices):
    if model.training:
        raise ValueError("The phase-default probe requires model.eval()")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The phase-default probe requires frozen model parameters")
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("The phase-default probe requires model.blocks")
    contracts = []
    reference = None
    for block_index in block_indices:
        if not 0 <= block_index < len(blocks):
            raise ValueError(f"block {block_index} is outside the model")
        block = blocks[block_index]
        if not getattr(block, "use_moe", False):
            raise ValueError(f"block {block_index} is not an MoE block")
        moe = block.mlp
        contract = (
            int(moe.num_routed_experts),
            int(moe.hidden_size),
            int(moe.top_k),
            str(moe.router_weight_mode),
        )
        if reference is None:
            reference = contract
        elif contract != reference:
            raise ValueError("Requested MoE blocks do not share one routing contract")
        if contract[2] != 1 or contract[3] != "identity":
            raise ValueError("The gate requires Base top-1 identity routing")
        if getattr(moe, "phase_metric", None) is not None:
            raise ValueError("The gate must diagnose the phase-unaware Base router")
        contracts.append({
            "block_index": int(block_index),
            "num_routed_experts": contract[0],
            "hidden_size": contract[1],
            "top_k": contract[2],
            "router_weight_mode": contract[3],
        })
    return {
        "blocks": contracts,
        "num_experts": reference[0],
        "hidden_size": reference[1],
    }


def _case_inputs(case, device):
    seed = int(case["seed"])
    clean_latent = _load_latent(case["latent_path"], "latent", seed, device)
    torch.manual_seed(seed + 1)
    noise = torch.randn_like(clean_latent)
    label = torch.tensor([case["label"]], device=device, dtype=torch.long)
    return clean_latent, noise, label


def _diffusion_inputs(clean_latent, noise, sigma, num_train_timesteps):
    sigma_tensor = torch.tensor(
        float(sigma), device=clean_latent.device, dtype=clean_latent.dtype
    )
    timestep = torch.full(
        (clean_latent.shape[0],),
        float(sigma) * num_train_timesteps,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    noised = (1.0 - sigma_tensor) * clean_latent + sigma_tensor * noise
    target = (noise - clean_latent).squeeze(2)
    return noised, timestep, target


def _native_routes(moe_layer, hidden_states, labels, timestep):
    with torch.no_grad():
        weights, indices, auxiliary_loss = _compute_router(
            moe_layer, hidden_states, labels, timestep
        )
    if auxiliary_loss is not None:
        raise RuntimeError("Frozen Base routing unexpectedly returned an auxiliary loss")
    if indices.shape[-1] != 1 or weights.shape != indices.shape:
        raise RuntimeError("The phase-default gate requires aligned top-1 routes")
    route_ids = indices[..., 0]
    if (
        route_ids.min().item() < 0
        or route_ids.max().item() >= int(moe_layer.num_routed_experts)
    ):
        raise RuntimeError("Conditional probe routed to a non-routed expert")
    return route_ids, weights[..., 0]


def _token_seed(case_id, sigma, block_index):
    payload = f"phase-default|{case_id}|{float(sigma):.8f}|{int(block_index)}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _sample_tokens(token_count, sample_count, case_id, sigma, block_index, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_token_seed(case_id, sigma, block_index))
    indices = torch.randperm(token_count, generator=generator)[:sample_count]
    return indices.to(device=device)


def _all_expert_outputs(moe_layer, hidden_states):
    outputs = [
        expert(hidden_states).float()
        for expert in moe_layer.experts[:moe_layer.num_routed_experts]
    ]
    return torch.stack(outputs, dim=1)


def _cosine_similarity(left, right):
    left = left.double().reshape(-1)
    right = right.double().reshape(-1)
    left_norm = left.norm()
    right_norm = right.norm()
    if not torch.isfinite(left_norm) or left_norm.item() <= 0:
        raise RuntimeError("Exact comparison vector has zero or non-finite norm")
    if not torch.isfinite(right_norm):
        raise RuntimeError("Approximate comparison vector has non-finite norm")
    if right_norm.item() == 0:
        return 0.0
    return float((left @ right / (left_norm * right_norm)).item())


def center_gradient_from_score_gradient(hidden_states, centers, score_gradient):
    """Map local identity-score gradients through normalized cosine routing."""

    if hidden_states.ndim != 2 or centers.ndim != 2 or score_gradient.ndim != 2:
        raise ValueError("Center-gradient inputs must be matrices")
    if score_gradient.shape != (hidden_states.shape[0], centers.shape[0]):
        raise ValueError("Score gradients must align with tokens and experts")
    hidden = hidden_states.double()
    centers = centers.double()
    scores = score_gradient.double()
    hidden_unit = F.normalize(hidden, p=2, dim=-1)
    center_norm = centers.norm(dim=-1)
    if (center_norm <= 0).any() or not torch.isfinite(center_norm).all():
        raise RuntimeError("Router centers must have finite nonzero norms")
    center_unit = centers / center_norm.unsqueeze(-1)
    cosine = hidden_unit @ center_unit.T
    jacobian = (
        hidden_unit[:, None, :]
        - cosine[:, :, None] * center_unit[None, :, :]
    ) / center_norm[None, :, None]
    return (scores[:, :, None] * jacobian).sum(dim=0)


def approximation_metrics(
    exact_outputs,
    suffix_gradient,
    native_ids,
    hidden_states,
    centers,
    defaults,
):
    """Compare default vectors with exact unselected expert outputs and credit."""

    if exact_outputs.ndim != 3:
        raise ValueError("exact_outputs must be [tokens, experts, hidden]")
    token_count, num_experts, hidden_size = exact_outputs.shape
    if suffix_gradient.shape != (token_count, hidden_size):
        raise ValueError("suffix_gradient shape differs from exact expert outputs")
    if native_ids.shape != (token_count,):
        raise ValueError("native_ids must contain one expert per token")
    if set(defaults) != set(APPROXIMATION_NAMES):
        raise ValueError("Default approximation names differ from the probe contract")
    if any(value.shape != (num_experts, hidden_size) for value in defaults.values()):
        raise ValueError("Every default table must be [experts, hidden]")

    device = exact_outputs.device
    rows = torch.arange(token_count, device=device)
    selected = F.one_hot(native_ids, num_classes=num_experts).bool()
    missing = ~selected
    exact = exact_outputs.float()
    gradient = suffix_gradient.float()
    exact_scores = torch.einsum("th,teh->te", gradient, exact)
    exact_center = center_gradient_from_score_gradient(
        hidden_states, centers, exact_scores
    )
    missing_energy = exact[missing].double().square().sum()
    if not torch.isfinite(missing_energy) or missing_energy.item() <= 0:
        raise RuntimeError("Unselected exact expert outputs have no finite energy")

    metrics = {}
    for name in APPROXIMATION_NAMES:
        table = defaults[name].to(device=device, dtype=exact.dtype)
        approximate = table.unsqueeze(0).expand(token_count, -1, -1).clone()
        approximate[rows, native_ids] = exact[rows, native_ids]
        error = approximate[missing].double() - exact[missing].double()
        relative_output_error = error.square().sum() / missing_energy
        approximate_scores = torch.einsum("th,teh->te", gradient, approximate)
        approximate_center = center_gradient_from_score_gradient(
            hidden_states, centers, approximate_scores
        )
        score_difference = approximate_scores.double() - exact_scores.double()
        center_difference = approximate_center - exact_center
        metrics[name] = {
            "unselected_output_relative_squared_error": float(
                relative_output_error.item()
            ),
            "missing_score_gradient_cosine": _cosine_similarity(
                exact_scores[missing], approximate_scores[missing]
            ),
            "full_score_gradient_cosine": _cosine_similarity(
                exact_scores, approximate_scores
            ),
            "full_score_gradient_relative_l2": float(
                (score_difference.norm() / exact_scores.double().norm()).item()
            ),
            "center_gradient_cosine": _cosine_similarity(
                exact_center, approximate_center
            ),
            "center_gradient_relative_l2": float(
                (center_difference.norm() / exact_center.norm()).item()
            ),
            "center_gradient_norm_ratio": float(
                (approximate_center.norm() / exact_center.norm()).item()
            ),
        }
    return metrics


def _safe_reduction(baseline, candidate, name):
    baseline = float(baseline)
    candidate = float(candidate)
    if not math.isfinite(baseline) or baseline <= 0:
        raise RuntimeError(f"{name} baseline must be finite and positive")
    if not math.isfinite(candidate) or candidate < 0:
        raise RuntimeError(f"{name} candidate must be finite and non-negative")
    return 1.0 - candidate / baseline


def _cell_contrasts(metrics):
    phase = metrics["phase"]
    global_default = metrics["global"]
    shuffled = metrics["shuffled_phase"]
    return {
        "phase_vs_global_output_error_reduction": _safe_reduction(
            global_default["unselected_output_relative_squared_error"],
            phase["unselected_output_relative_squared_error"],
            "phase versus global output error",
        ),
        "phase_minus_global_missing_score_cosine": (
            phase["missing_score_gradient_cosine"]
            - global_default["missing_score_gradient_cosine"]
        ),
        "phase_minus_global_center_cosine": (
            phase["center_gradient_cosine"]
            - global_default["center_gradient_cosine"]
        ),
        "phase_vs_shuffled_output_error_reduction": _safe_reduction(
            shuffled["unselected_output_relative_squared_error"],
            phase["unselected_output_relative_squared_error"],
            "phase versus shuffled output error",
        ),
        "phase_minus_shuffled_missing_score_cosine": (
            phase["missing_score_gradient_cosine"]
            - shuffled["missing_score_gradient_cosine"]
        ),
        "phase_minus_shuffled_center_cosine": (
            phase["center_gradient_cosine"]
            - shuffled["center_gradient_cosine"]
        ),
    }


def _calibrate_sketches(
    model,
    capture,
    cases,
    sigmas,
    block_indices,
    num_train_timesteps,
    accumulator,
    device,
    progress,
):
    for case_index, case in enumerate(cases, start=1):
        clean, noise, label = _case_inputs(case, device)
        for phase_index, sigma in enumerate(sigmas):
            noised, timestep, _ = _diffusion_inputs(
                clean, noise, sigma, num_train_timesteps
            )
            capture.start(gradient_mode=False)
            try:
                with torch.no_grad():
                    model(noised, timestep, context=label)
                capture.validate_complete()
                for block_index in block_indices:
                    moe = model.blocks[block_index].mlp
                    hidden = capture.hidden_states[block_index]
                    labels = capture.labels[block_index]
                    route_ids, _ = _native_routes(moe, hidden, labels, timestep)
                    accumulator.update(
                        block_index,
                        phase_index,
                        hidden[0],
                        route_ids[0],
                        moe.experts[:moe.num_routed_experts],
                    )
            finally:
                capture.stop()
        if progress is not None:
            progress("calibration", case_index, len(cases), case["id"])


def _evaluate_cases(
    model,
    capture,
    cases,
    sigmas,
    block_indices,
    num_train_timesteps,
    tokens_per_cell,
    phase_shuffle_offset,
    sketches,
    device,
    progress,
):
    records = []
    for case_index, case in enumerate(cases, start=1):
        clean, noise, label = _case_inputs(case, device)
        for phase_index, sigma in enumerate(sigmas):
            noised, timestep, target = _diffusion_inputs(
                clean, noise, sigma, num_train_timesteps
            )
            capture.start(gradient_mode=True)
            try:
                model_output = model(noised, timestep, context=label)
                prediction = _extract_prediction(model_output, target.shape[1])
                loss = _per_sample_mse(prediction, target).mean()
                suffix_gradients = capture.suffix_gradients(loss)
                for block_index in block_indices:
                    moe = model.blocks[block_index].mlp
                    hidden = capture.hidden_states[block_index]
                    labels = capture.labels[block_index]
                    route_ids, route_weights = _native_routes(
                        moe, hidden, labels, timestep
                    )
                    token_indices = _sample_tokens(
                        hidden.shape[1],
                        min(tokens_per_cell, hidden.shape[1]),
                        case["id"],
                        sigma,
                        block_index,
                        hidden.device,
                    )
                    sampled_hidden = hidden[0, token_indices]
                    sampled_ids = route_ids[0, token_indices]
                    sampled_gradient = suffix_gradients[block_index][0, token_indices]
                    with torch.no_grad():
                        exact_outputs = _all_expert_outputs(moe, sampled_hidden)
                    shuffled_index = (
                        phase_index + phase_shuffle_offset
                    ) % len(sigmas)
                    phase_defaults = sketches["phase_defaults"][block_index]
                    defaults = {
                        "zero": torch.zeros_like(
                            sketches["global_defaults"][block_index]
                        ),
                        "global": sketches["global_defaults"][block_index],
                        "phase": phase_defaults[phase_index],
                        "shuffled_phase": phase_defaults[shuffled_index],
                    }
                    metrics = approximation_metrics(
                        exact_outputs=exact_outputs,
                        suffix_gradient=sampled_gradient,
                        native_ids=sampled_ids,
                        hidden_states=sampled_hidden,
                        centers=moe.cluster_centers.detach(),
                        defaults=defaults,
                    )
                    records.append({
                        "case_id": case["id"],
                        "label": int(case["label"]),
                        "sigma": float(sigma),
                        "phase_index": int(phase_index),
                        "shuffled_phase_index": int(shuffled_index),
                        "block_index": int(block_index),
                        "native_mse": float(loss.item()),
                        "token_indices": token_indices.cpu().tolist(),
                        "native_route_ids": sampled_ids.cpu().tolist(),
                        "native_route_weights": route_weights[
                            0, token_indices
                        ].float().cpu().tolist(),
                        "global_calibration_counts": sketches["global_counts"][
                            block_index
                        ].tolist(),
                        "phase_calibration_counts": sketches["phase_counts"][
                            block_index
                        ][phase_index].tolist(),
                        "metrics": metrics,
                        "contrasts": _cell_contrasts(metrics),
                    })
            finally:
                capture.stop()
        if progress is not None:
            progress("confirmatory", case_index, len(cases), case["id"])
    return records


def _bootstrap_summary(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Image bootstrap requires a finite vector with two images")
    resamples = _require_int(int(resamples), "bootstrap_resamples", 1000)
    generator = np.random.default_rng(int(seed))
    means = np.empty(resamples, dtype=np.float64)
    chunk_size = 10_000
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = generator.integers(
            0, values.size, size=(stop - start, values.size)
        )
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "one_sided_lcb95": float(np.quantile(means, 0.05)),
        "one_sided_ucb95": float(np.quantile(means, 0.95)),
        "image_values": values.tolist(),
    }


def _image_means(records, expected_cells):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["case_id"]].append(record)
    image_rows = []
    for case_id, cells in sorted(grouped.items()):
        if len(cells) != expected_cells:
            raise RuntimeError(
                f"{case_id} has {len(cells)} cells, expected {expected_cells}"
            )
        row = {"case_id": case_id, "num_cells": len(cells)}
        for key in CONTRAST_KEYS:
            values = np.asarray(
                [cell["contrasts"][key] for cell in cells], dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise RuntimeError(f"{case_id}.{key} contains non-finite values")
            row[key] = float(values.mean())
        phase_metrics = [cell["metrics"]["phase"] for cell in cells]
        row["phase_output_relative_squared_error"] = float(np.mean([
            metric["unselected_output_relative_squared_error"]
            for metric in phase_metrics
        ]))
        row["phase_missing_score_cosine"] = float(np.mean([
            metric["missing_score_gradient_cosine"] for metric in phase_metrics
        ]))
        row["phase_center_cosine"] = float(np.mean([
            metric["center_gradient_cosine"] for metric in phase_metrics
        ]))
        if not all(math.isfinite(row[key]) for key in SUMMARY_KEYS):
            raise RuntimeError(f"{case_id} contains a non-finite image metric")
        image_rows.append(row)
    return image_rows


def _check(observed, required, passed):
    return {
        "observed": float(observed),
        "required": float(required),
        "passed": bool(passed),
    }


def build_gate_summary(records, sketches, protocol):
    sigmas, blocks = _validate_protocol(protocol)
    expected_images = protocol["confirmatory_cases"]
    expected_cells = len(sigmas) * len(blocks)
    image_rows = _image_means(records, expected_cells)
    if len(image_rows) != expected_images:
        raise RuntimeError(
            f"Found {len(image_rows)} confirmatory images, expected {expected_images}"
        )

    bootstrap = {}
    for offset, key in enumerate(SUMMARY_KEYS):
        bootstrap[key] = _bootstrap_summary(
            [row[key] for row in image_rows],
            protocol["bootstrap_resamples"],
            protocol["bootstrap_seed"] + offset,
        )

    phase_counts = torch.cat([
        sketches["phase_counts"][block].reshape(-1) for block in blocks
    ])
    global_counts = torch.cat([
        sketches["global_counts"][block].reshape(-1) for block in blocks
    ])
    phase_coverage = float((phase_counts > 0).double().mean().item())
    global_coverage = float((global_counts > 0).double().mean().item())
    gates = protocol["gates"]
    checks = {
        "phase_buffer_coverage": _check(
            phase_coverage,
            protocol["minimum_phase_buffer_coverage"],
            phase_coverage >= protocol["minimum_phase_buffer_coverage"],
        ),
        "mean_phase_output_relative_squared_error": _check(
            bootstrap["phase_output_relative_squared_error"]["mean"],
            gates["maximum_mean_phase_output_relative_squared_error"],
            bootstrap["phase_output_relative_squared_error"]["mean"]
            <= gates["maximum_mean_phase_output_relative_squared_error"],
        ),
        "ucb_phase_output_relative_squared_error": _check(
            bootstrap["phase_output_relative_squared_error"][
                "one_sided_ucb95"
            ],
            gates["maximum_ucb_phase_output_relative_squared_error"],
            bootstrap["phase_output_relative_squared_error"][
                "one_sided_ucb95"
            ] <= gates["maximum_ucb_phase_output_relative_squared_error"],
        ),
        "mean_phase_missing_score_cosine": _check(
            bootstrap["phase_missing_score_cosine"]["mean"],
            gates["minimum_mean_phase_missing_score_cosine"],
            bootstrap["phase_missing_score_cosine"]["mean"]
            >= gates["minimum_mean_phase_missing_score_cosine"],
        ),
        "lcb_phase_missing_score_cosine": _check(
            bootstrap["phase_missing_score_cosine"]["one_sided_lcb95"],
            gates["minimum_lcb_phase_missing_score_cosine"],
            bootstrap["phase_missing_score_cosine"]["one_sided_lcb95"]
            >= gates["minimum_lcb_phase_missing_score_cosine"],
        ),
        "mean_phase_center_cosine": _check(
            bootstrap["phase_center_cosine"]["mean"],
            gates["minimum_mean_phase_center_cosine"],
            bootstrap["phase_center_cosine"]["mean"]
            >= gates["minimum_mean_phase_center_cosine"],
        ),
        "lcb_phase_center_cosine": _check(
            bootstrap["phase_center_cosine"]["one_sided_lcb95"],
            gates["minimum_lcb_phase_center_cosine"],
            bootstrap["phase_center_cosine"]["one_sided_lcb95"]
            >= gates["minimum_lcb_phase_center_cosine"],
        ),
        "mean_phase_vs_global_output_error_reduction": _check(
            bootstrap["phase_vs_global_output_error_reduction"]["mean"],
            gates["minimum_mean_phase_vs_global_output_error_reduction"],
            bootstrap["phase_vs_global_output_error_reduction"]["mean"]
            >= gates["minimum_mean_phase_vs_global_output_error_reduction"],
        ),
        "lcb_phase_vs_global_output_error_reduction": _check(
            bootstrap["phase_vs_global_output_error_reduction"][
                "one_sided_lcb95"
            ],
            gates["minimum_lcb_phase_vs_global_output_error_reduction"],
            bootstrap["phase_vs_global_output_error_reduction"][
                "one_sided_lcb95"
            ] >= gates["minimum_lcb_phase_vs_global_output_error_reduction"],
        ),
        "mean_phase_minus_global_missing_score_cosine": _check(
            bootstrap["phase_minus_global_missing_score_cosine"]["mean"],
            gates["minimum_mean_phase_minus_global_missing_score_cosine"],
            bootstrap["phase_minus_global_missing_score_cosine"]["mean"]
            >= gates["minimum_mean_phase_minus_global_missing_score_cosine"],
        ),
        "lcb_phase_minus_global_missing_score_cosine": _check(
            bootstrap["phase_minus_global_missing_score_cosine"][
                "one_sided_lcb95"
            ],
            gates["minimum_lcb_phase_minus_global_missing_score_cosine"],
            bootstrap["phase_minus_global_missing_score_cosine"][
                "one_sided_lcb95"
            ] >= gates["minimum_lcb_phase_minus_global_missing_score_cosine"],
        ),
        "mean_phase_minus_global_center_cosine": _check(
            bootstrap["phase_minus_global_center_cosine"]["mean"],
            gates["minimum_mean_phase_minus_global_center_cosine"],
            bootstrap["phase_minus_global_center_cosine"]["mean"]
            >= gates["minimum_mean_phase_minus_global_center_cosine"],
        ),
        "lcb_phase_minus_global_center_cosine": _check(
            bootstrap["phase_minus_global_center_cosine"]["one_sided_lcb95"],
            gates["minimum_lcb_phase_minus_global_center_cosine"],
            bootstrap["phase_minus_global_center_cosine"]["one_sided_lcb95"]
            >= gates["minimum_lcb_phase_minus_global_center_cosine"],
        ),
        "lcb_phase_vs_shuffled_output_error_reduction": _check(
            bootstrap["phase_vs_shuffled_output_error_reduction"][
                "one_sided_lcb95"
            ],
            gates["minimum_lcb_phase_vs_shuffled_output_error_reduction"],
            bootstrap["phase_vs_shuffled_output_error_reduction"][
                "one_sided_lcb95"
            ] >= gates["minimum_lcb_phase_vs_shuffled_output_error_reduction"],
        ),
        "lcb_phase_minus_shuffled_missing_score_cosine": _check(
            bootstrap["phase_minus_shuffled_missing_score_cosine"][
                "one_sided_lcb95"
            ],
            gates["minimum_lcb_phase_minus_shuffled_missing_score_cosine"],
            bootstrap["phase_minus_shuffled_missing_score_cosine"][
                "one_sided_lcb95"
            ] >= gates["minimum_lcb_phase_minus_shuffled_missing_score_cosine"],
        ),
        "lcb_phase_minus_shuffled_center_cosine": _check(
            bootstrap["phase_minus_shuffled_center_cosine"]["one_sided_lcb95"],
            gates["minimum_lcb_phase_minus_shuffled_center_cosine"],
            bootstrap["phase_minus_shuffled_center_cosine"]["one_sided_lcb95"]
            >= gates["minimum_lcb_phase_minus_shuffled_center_cosine"],
        ),
    }

    strata = {"by_block": {}, "by_sigma": {}}
    for block in blocks:
        cells = [record for record in records if record["block_index"] == block]
        strata["by_block"][str(block)] = {
            key: float(np.mean([cell["contrasts"][key] for cell in cells]))
            for key in CONTRAST_KEYS
        }
    for sigma in sigmas:
        cells = [record for record in records if record["sigma"] == sigma]
        strata["by_sigma"][str(sigma)] = {
            key: float(np.mean([cell["contrasts"][key] for cell in cells]))
            for key in CONTRAST_KEYS
        }
    return {
        "passed": bool(all(item["passed"] for item in checks.values())),
        "decision": (
            "authorize_phase_conditioned_default_training"
            if all(item["passed"] for item in checks.values())
            else "reject_or_redesign_before_training"
        ),
        "checks": checks,
        "bootstrap": bootstrap,
        "image_rows": image_rows,
        "strata": strata,
        "coverage": {
            "global_buffer_coverage": global_coverage,
            "phase_buffer_coverage": phase_coverage,
            "global_min_selected_tokens": int(global_counts.min().item()),
            "phase_min_selected_tokens": int(phase_counts.min().item()),
            "global_counts": {
                str(block): sketches["global_counts"][block].tolist()
                for block in blocks
            },
            "phase_counts": {
                str(block): sketches["phase_counts"][block].tolist()
                for block in blocks
            },
        },
    }


def run_phase_default_probe(
    checkpoint_path,
    weights_checkpoint_path,
    manifest,
    device="cpu",
    num_threads=8,
    progress=None,
):
    checkpoint_path = Path(checkpoint_path).resolve()
    weights_checkpoint_path = Path(weights_checkpoint_path).resolve()
    if not checkpoint_path.is_file() or not weights_checkpoint_path.is_file():
        raise FileNotFoundError("Canonical and local checkpoint files are required")
    protocol = manifest["protocol"]
    if parse_checkpoint_step(checkpoint_path) != protocol["checkpoint_step"]:
        raise ValueError("Canonical checkpoint step differs from the manifest")
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    runtime_cfg = load_runtime_cfg(config_path)
    if runtime_cfg.model_name != protocol["model_name"]:
        raise ValueError("Checkpoint config model_name differs from the manifest")

    thread_config = _configure_torch_threads(int(num_threads))
    device = torch.device(device)
    model, state_name, weights_step, load_seconds = _load_checkpoint_model(
        runtime_cfg, weights_checkpoint_path, device
    )
    if state_name != protocol["checkpoint_state"]:
        raise ValueError("Loaded checkpoint state differs from the manifest")
    if weights_step != protocol["checkpoint_step"]:
        raise ValueError("Loaded checkpoint step differs from the manifest")
    model_contract = _validate_model_contract(model, manifest["block_indices"])

    calibration_cases = [
        case for case in manifest["cases"] if case["split"] == "calibration"
    ]
    confirmatory_cases = [
        case for case in manifest["cases"] if case["split"] == "confirmatory"
    ]
    accumulator = DefaultSketchAccumulator(
        blocks=manifest["block_indices"],
        num_phases=len(manifest["sigmas"]),
        num_experts=model_contract["num_experts"],
        hidden_size=model_contract["hidden_size"],
    )
    capture = MultiMoeCapture(model, manifest["block_indices"])
    probe_start = time.perf_counter()
    try:
        _calibrate_sketches(
            model=model,
            capture=capture,
            cases=calibration_cases,
            sigmas=manifest["sigmas"],
            block_indices=manifest["block_indices"],
            num_train_timesteps=runtime_cfg.num_train_timesteps,
            accumulator=accumulator,
            device=device,
            progress=progress,
        )
        sketches = accumulator.finalize()
        records = _evaluate_cases(
            model=model,
            capture=capture,
            cases=confirmatory_cases,
            sigmas=manifest["sigmas"],
            block_indices=manifest["block_indices"],
            num_train_timesteps=runtime_cfg.num_train_timesteps,
            tokens_per_cell=protocol["tokens_per_cell"],
            phase_shuffle_offset=protocol["phase_shuffle_offset"],
            sketches=sketches,
            device=device,
            progress=progress,
        )
    finally:
        capture.close()
    elapsed = time.perf_counter() - probe_start
    gate = build_gate_summary(records, sketches, protocol)
    return {
        "phase_default_probe_version": PROBE_VERSION,
        "diagnostic_scope": (
            "frozen Base checkpoint upper-bound diagnostic; no training or FID claim"
        ),
        "hypothesis": (
            "diffusion-stage-conditioned selected-output means approximate missing "
            "expert outputs and dense router credit better than one global mean"
        ),
        "checkpoint": str(checkpoint_path),
        "weights_checkpoint": str(weights_checkpoint_path),
        "checkpoint_step": int(protocol["checkpoint_step"]),
        "checkpoint_state": state_name,
        "config": str(config_path),
        "model_name": runtime_cfg.model_name,
        "manifest": manifest["path"],
        "manifest_sha256": manifest["sha256"],
        "sigmas": list(manifest["sigmas"]),
        "block_indices": list(manifest["block_indices"]),
        "tokens_per_cell": int(protocol["tokens_per_cell"]),
        "model_contract": model_contract,
        "thread_config": thread_config,
        "device": str(device),
        "load_seconds": float(load_seconds),
        "probe_seconds": float(elapsed),
        "gate": gate,
        "records": records,
    }
