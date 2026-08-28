"""Frozen checkpoint audit for phase-conditioned ProMoE routing."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

from analyses.denoising_regret.probe import (
    _compute_router,
    _configure_torch_threads,
    _exact_counterfactual_changes,
    _extract_prediction,
    _load_checkpoint_model,
    _load_latent,
    _per_sample_mse,
    _rankdata,
)
from analyses.routing_translation.probe import RouteInputCapture
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)


PROBE_VERSION = 1
SPEC_NAME = "phase_metric_50k_gate_v1"
CANONICAL_SPEC_RELATIVE_PATH = Path(
    "analyses/routing_metric/manifests/phase_metric_50k_gate_v1.json"
)
CANONICAL_SPEC_SHA256 = (
    "193f7a2ac437e22c5e2bc547dae4bcc15fb10dba3323d48b488da60d3180fec1"
)
EXPECTED_TRAINING_CONTRACT = {
    "global_seed": 0,
    "total_train_batch_size": 256,
    "lr": 0.0001,
    "candidate_gpu_ids": [0, 1, 2, 3],
    "base_gpu_ids": [4, 5, 6, 7],
}
EXPECTED_ROUTING_CONTRACT = {
    "num_routed_experts": 12,
    "top_k": 1,
    "router_weight_mode": "identity",
    "use_uncond_expert": True,
}
EXPECTED_PHASE_METRIC_CONTRACT = {
    "enabled": True,
    "rank": 8,
    "num_fourier_bands": 4,
    "num_train_timesteps": 1000,
    "scale": 0.25,
    "shuffle_timestep": False,
    "init_seed": 1729,
}
EXPECTED_PROBE_EXECUTION_CONTRACT = {
    "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    "num_threads_per_worker": 4,
}
MODE_DEFINITIONS = {
    "phase_phase": ("phase", "phase", 0),
    "phase_base": ("phase", "base", 0),
    "base_phase": ("base", "phase", 0),
    "base_base": ("base", "base", 0),
    "shuffled_phase_phase": ("phase", "phase", 1),
}
BOOTSTRAP_KEYS = (
    "exact_phase_route_relative_gain",
    "phase_route_win_rate",
    "selection_relative_gain",
    "weight_relative_gain",
    "native_vs_base_base_relative_gain",
    "native_vs_shuffled_relative_gain",
    "candidate_vs_base_relative_gain",
    "route_flip_fraction",
)
DISCOVERY_REQUIREMENTS = {
    "minimum_mean_route_flip_fraction",
    "minimum_exact_probe_count",
    "minimum_mean_exact_phase_route_relative_gain",
    "minimum_mean_phase_route_win_rate",
    "minimum_mean_selection_relative_gain",
    "minimum_mean_native_vs_base_base_relative_gain",
    "minimum_mean_native_vs_shuffled_relative_gain",
    "minimum_mean_candidate_vs_base_relative_gain",
    "maximum_noop_abs_mse_change",
    "maximum_native_override_abs_output_change",
    "maximum_native_override_abs_mse_change",
}
CONFIRMATORY_REQUIREMENTS = DISCOVERY_REQUIREMENTS | {
    "minimum_lcb_exact_phase_route_relative_gain",
    "minimum_lcb_selection_relative_gain",
    "minimum_lcb_native_vs_base_base_relative_gain",
    "minimum_lcb_native_vs_shuffled_relative_gain",
}


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


def _require_finite_tensor(value, name):
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"{name} contains non-finite values")
    return value


def _require_finite_payload(value, name):
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_payload(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_payload(item, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{name} is non-finite")


def load_gate_spec(path, project_root):
    path = Path(path).resolve()
    project_root = Path(project_root).resolve()
    canonical_path = (project_root / CANONICAL_SPEC_RELATIVE_PATH).resolve()
    if path != canonical_path:
        raise ValueError(f"Gate spec must be the canonical manifest: {canonical_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Gate spec does not exist: {path}")
    spec_sha256 = sha256_file(path)
    if spec_sha256 != CANONICAL_SPEC_SHA256:
        raise ValueError("Canonical gate spec SHA256 differs from the locked design")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {
        "version", "name", "locked_before_checkpoint", "protocol"
    }:
        raise ValueError("Gate spec top-level keys differ from the locked schema")
    if (
        payload["version"] != 1
        or payload["name"] != SPEC_NAME
        or payload["locked_before_checkpoint"] is not True
    ):
        raise ValueError("Unexpected phase-metric gate identity")

    protocol = payload["protocol"]
    required = {
        "candidate_config_stem",
        "base_config_stem",
        "model_name",
        "fresh_start_required",
        "training_contract",
        "routing_contract",
        "phase_metric_contract",
        "probe_execution_contract",
        "checkpoint_step",
        "checkpoint_state",
        "case_manifest",
        "case_manifest_sha256",
        "split_counts",
        "sigmas",
        "block_indices",
        "tokens_per_cell",
        "noop_tokens_per_cell",
        "exact_batch_size",
        "phase_shuffle_offset",
        "bootstrap_resamples",
        "bootstrap_seeds",
        "requirements",
    }
    if set(protocol) != required:
        raise ValueError("Gate protocol keys differ from the locked schema")
    if protocol["model_name"] != "ProMoE_TC_B":
        raise ValueError("The gate is locked to ProMoE_TC_B")
    if protocol["fresh_start_required"] is not True:
        raise ValueError("The gate requires independent fresh training")
    if protocol["training_contract"] != EXPECTED_TRAINING_CONTRACT:
        raise ValueError("Training contract differs from the locked design")
    if protocol["routing_contract"] != EXPECTED_ROUTING_CONTRACT:
        raise ValueError("Routing contract differs from the locked design")
    if protocol["phase_metric_contract"] != EXPECTED_PHASE_METRIC_CONTRACT:
        raise ValueError("Phase-Metric contract differs from the locked design")
    if protocol["probe_execution_contract"] != EXPECTED_PROBE_EXECUTION_CONTRACT:
        raise ValueError("Probe execution contract differs from the locked design")
    _require_int(protocol["checkpoint_step"], "checkpoint_step", 1)
    if protocol["checkpoint_state"] != "ema_model_state_dict":
        raise ValueError("The gate requires EMA checkpoint weights")
    sigmas = tuple(
        _require_finite(value, "sigma", 0.0, 1.0)
        for value in protocol["sigmas"]
    )
    if (
        len(sigmas) < 3
        or len(sigmas) != len(set(sigmas))
        or tuple(sorted(sigmas)) != sigmas
        or any(value in {0.0, 1.0} for value in sigmas)
    ):
        raise ValueError("sigmas must be unique ordered interior values")
    blocks = tuple(
        _require_int(value, "block_index") for value in protocol["block_indices"]
    )
    if not blocks or len(blocks) != len(set(blocks)):
        raise ValueError("block_indices must be nonempty and unique")
    _require_int(protocol["tokens_per_cell"], "tokens_per_cell", 1)
    _require_int(protocol["noop_tokens_per_cell"], "noop_tokens_per_cell", 1)
    _require_int(protocol["exact_batch_size"], "exact_batch_size", 1)
    offset = _require_int(
        protocol["phase_shuffle_offset"], "phase_shuffle_offset", 1
    )
    if offset >= len(sigmas):
        raise ValueError("phase_shuffle_offset must be smaller than sigma count")
    _require_int(protocol["bootstrap_resamples"], "bootstrap_resamples", 1000)

    splits = {"discovery", "confirmatory"}
    if set(protocol["split_counts"]) != splits:
        raise ValueError("split_counts must define discovery and confirmatory")
    if set(protocol["bootstrap_seeds"]) != splits:
        raise ValueError("bootstrap_seeds must define both splits")
    if set(protocol["requirements"]) != splits:
        raise ValueError("requirements must define both splits")
    for split in sorted(splits):
        _require_int(protocol["split_counts"][split], f"{split}.count", 2)
        _require_int(protocol["bootstrap_seeds"][split], f"{split}.seed")
        requirements = protocol["requirements"][split]
        expected_requirements = (
            DISCOVERY_REQUIREMENTS
            if split == "discovery"
            else CONFIRMATORY_REQUIREMENTS
        )
        if set(requirements) != expected_requirements:
            raise ValueError(f"{split} requirement keys differ from the locked schema")
        for name, value in requirements.items():
            if name == "minimum_exact_probe_count":
                _require_int(value, f"{split}.{name}", 1)
            else:
                _require_finite(value, f"{split}.{name}", -1.0, 1.0)

    case_manifest = (project_root / protocol["case_manifest"]).resolve()
    try:
        case_manifest.relative_to(project_root)
    except ValueError as error:
        raise ValueError("case_manifest must stay inside the repository") from error
    if not case_manifest.is_file():
        raise FileNotFoundError(f"Case manifest does not exist: {case_manifest}")
    if sha256_file(case_manifest) != protocol["case_manifest_sha256"]:
        raise ValueError("Case manifest SHA256 differs from the gate spec")
    return {
        **payload,
        "path": str(path),
        "sha256": spec_sha256,
        "sigmas": sigmas,
        "block_indices": blocks,
        "case_manifest_path": str(case_manifest),
    }


def _router_score_pair(moe_layer, hidden_states, labels, timestep, phase_shift=0):
    if hidden_states.ndim != 3:
        raise ValueError("Router hidden states must be [batch, tokens, hidden]")
    batch_size, token_count, hidden_size = hidden_states.shape
    if hidden_size != int(moe_layer.hidden_size):
        raise ValueError("Router hidden size differs from the MoE contract")
    if timestep is None:
        raise ValueError("Phase-aware score reconstruction requires timestep")
    phase_metric = getattr(moe_layer, "phase_metric", None)
    if phase_metric is None:
        raise ValueError("Phase-aware score reconstruction requires phase_metric")
    if moe_layer.router_weight_mode != "identity":
        raise ValueError("The factorial gate requires identity router weights")

    flat_input = hidden_states.reshape(-1, hidden_size)
    flat_labels = labels.reshape(batch_size, 1).expand(-1, token_count).reshape(-1)
    conditional = flat_labels != 1000
    conditional_positions = torch.where(conditional)[0]
    num_experts = int(moe_layer.num_routed_experts)
    base_scores = torch.zeros(
        batch_size * token_count,
        num_experts,
        device=hidden_states.device,
        dtype=torch.float32,
    )
    phase_scores = torch.zeros_like(base_scores)
    if conditional_positions.numel() == 0:
        shape = (batch_size, token_count, num_experts)
        return base_scores.view(shape), phase_scores.view(shape), conditional.view(
            batch_size, token_count
        )

    conditional_input = flat_input[conditional_positions]
    input_norm = F.normalize(conditional_input, p=2, dim=-1)
    center_norm = F.normalize(moe_layer.cluster_centers, p=2, dim=-1)
    cosine = input_norm @ center_norm.T
    phase_timesteps = timestep.reshape(-1).to(hidden_states.device)
    if phase_timesteps.numel() != batch_size:
        raise ValueError("timestep must contain one value per batch sample")
    if phase_shift:
        phase_timesteps = torch.roll(
            phase_timesteps, shifts=int(phase_shift), dims=0
        )
    sample_indices = conditional_positions // token_count
    token_timesteps = phase_timesteps[sample_indices]
    residual = phase_metric(input_norm, center_norm, token_timesteps)
    base_scores[conditional_positions] = cosine.to(base_scores.dtype)
    phase_scores[conditional_positions] = (cosine + residual).to(
        phase_scores.dtype
    )
    shape = (batch_size, token_count, num_experts)
    return (
        base_scores.view(shape),
        phase_scores.view(shape),
        conditional.view(batch_size, token_count),
    )


def _top1_margin(scores):
    if scores.shape[-1] < 2:
        return torch.zeros(scores.shape[:-1], device=scores.device)
    values = torch.topk(scores, k=2, dim=-1).values
    return values[..., 0] - values[..., 1]


class FactorialRouterOverride:
    """Select experts and output weights from independent score sources."""

    def __init__(self, model, selection_source, weight_source, phase_shift=0):
        if selection_source not in {"base", "phase"}:
            raise ValueError("Unknown selection score source")
        if weight_source not in {"base", "phase"}:
            raise ValueError("Unknown weight score source")
        self.model = model
        self.selection_source = selection_source
        self.weight_source = weight_source
        self.phase_shift = int(phase_shift)
        self.captures = {}
        self._overridden = []

    def _wrapper(self, block_index, original_compute_router):
        del original_compute_router

        def compute_router_with_factorial(
            module, hidden_states, labels, timestep=None
        ):
            base_scores, phase_scores, conditional = _router_score_pair(
                module,
                hidden_states,
                labels,
                timestep,
                phase_shift=self.phase_shift,
            )
            selection_scores = (
                phase_scores
                if self.selection_source == "phase"
                else base_scores
            )
            weight_scores = (
                phase_scores if self.weight_source == "phase" else base_scores
            )
            selected_ids = torch.topk(
                selection_scores, k=1, dim=-1
            ).indices.squeeze(-1)
            selected_weights = weight_scores.gather(
                -1, selected_ids.unsqueeze(-1)
            ).squeeze(-1)
            batch_size, token_count = selected_ids.shape
            weights = torch.zeros(
                batch_size,
                token_count,
                1,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            indices = torch.zeros(
                batch_size,
                token_count,
                1,
                device=hidden_states.device,
                dtype=torch.long,
            )
            flat_conditional = conditional.reshape(-1)
            flat_weights = weights.reshape(-1)
            flat_indices = indices.reshape(-1)
            flat_weights[flat_conditional] = selected_weights.reshape(-1)[
                flat_conditional
            ].to(weights.dtype)
            flat_indices[flat_conditional] = selected_ids.reshape(-1)[
                flat_conditional
            ]
            unconditional = ~conditional
            if unconditional.any():
                if not module.use_uncond_expert:
                    raise RuntimeError("Unexpected unconditional tokens")
                flat_unconditional = unconditional.reshape(-1)
                flat_weights[flat_unconditional] = 1.0
                flat_indices[flat_unconditional] = int(module.num_experts) - 1

            if block_index in self.captures:
                raise RuntimeError(f"MoE block {block_index} ran more than once")
            base_ids = torch.topk(base_scores, k=1, dim=-1).indices.squeeze(-1)
            phase_ids = torch.topk(phase_scores, k=1, dim=-1).indices.squeeze(-1)
            residual = phase_scores - base_scores
            self.captures[block_index] = {
                "conditional": conditional.detach().cpu(),
                "base_ids": base_ids.detach().cpu(),
                "phase_ids": phase_ids.detach().cpu(),
                "selected_ids": selected_ids.detach().cpu(),
                "selected_weights": selected_weights.detach().float().cpu(),
                "base_margin": _top1_margin(base_scores).detach().float().cpu(),
                "phase_margin": _top1_margin(phase_scores).detach().float().cpu(),
                "residual_rms": residual.float().square().mean(
                    dim=(1, 2)
                ).sqrt().detach().cpu(),
                "residual_abs_max": residual.float().abs().amax(
                    dim=(1, 2)
                ).detach().cpu(),
            }
            return weights, indices, None

        return compute_router_with_factorial

    def __enter__(self):
        for block_index, block in enumerate(self.model.blocks):
            if not getattr(block, "use_moe", False):
                continue
            moe_layer = block.mlp
            if getattr(moe_layer, "phase_metric", None) is None:
                raise ValueError("Every candidate MoE block must have phase_metric")
            if "compute_router" in moe_layer.__dict__:
                raise RuntimeError("MoE layer already has a router override")
            original = moe_layer.compute_router
            moe_layer.compute_router = MethodType(
                self._wrapper(block_index, original), moe_layer
            )
            self._overridden.append(moe_layer)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for moe_layer in self._overridden:
            del moe_layer.compute_router
        self._overridden = []
        return False


def _case_inputs(case, device):
    seed = int(case["seed"])
    clean = _load_latent(case["latent"], case.get("latent_key", "latent"), seed, device)
    torch.manual_seed(seed + 1)
    noise = torch.randn_like(clean)
    label = torch.tensor([case["label"]], device=device, dtype=torch.long)
    return clean, noise, label


def _single_sigma_inputs(clean, noise, label, sigma, num_train_timesteps):
    sigma_tensor = torch.tensor(
        float(sigma), device=clean.device, dtype=clean.dtype
    )
    timestep = torch.full(
        (1,),
        float(sigma) * num_train_timesteps,
        device=clean.device,
        dtype=clean.dtype,
    )
    noised = (1.0 - sigma_tensor) * clean + sigma_tensor * noise
    target = (noise - clean).squeeze(2)
    return noised, timestep, label, target


def _mixed_sigma_inputs(clean, noise, label, sigmas, num_train_timesteps):
    rows = [
        _single_sigma_inputs(
            clean, noise, label, sigma, num_train_timesteps
        )
        for sigma in sigmas
    ]
    return tuple(torch.cat([row[index] for row in rows], dim=0) for index in range(4))


def _model_losses(model, noised, timestep, label, target):
    output = model(noised, timestep, context=label)
    prediction = _extract_prediction(output, target.shape[1])
    _require_finite_tensor(prediction, "model prediction")
    losses = _per_sample_mse(prediction, target)
    _require_finite_tensor(losses, "per-sample MSE")
    return output, losses


def _route_capture_rows(captures, sigmas, selected_blocks):
    rows = []
    for block_index in selected_blocks:
        if block_index not in captures:
            raise RuntimeError(f"Factorial capture missed block {block_index}")
        capture = captures[block_index]
        for row_index, sigma in enumerate(sigmas):
            conditional = capture["conditional"][row_index]
            base_ids = capture["base_ids"][row_index][conditional]
            phase_ids = capture["phase_ids"][row_index][conditional]
            flip_count = int((base_ids != phase_ids).sum())
            token_count = int(conditional.sum())
            if token_count == 0:
                raise RuntimeError("The phase gate requires conditional tokens")
            rows.append({
                "block_index": int(block_index),
                "sigma": float(sigma),
                "token_count": token_count,
                "flip_count": flip_count,
                "flip_fraction": flip_count / token_count,
                "base_margin_mean": float(
                    capture["base_margin"][row_index][conditional].mean()
                ),
                "phase_margin_mean": float(
                    capture["phase_margin"][row_index][conditional].mean()
                ),
                "residual_rms": float(capture["residual_rms"][row_index]),
                "residual_abs_max": float(
                    capture["residual_abs_max"][row_index]
                ),
            })
    return rows


def run_factorial_case(model, case, spec, device):
    protocol = spec["protocol"]
    sigmas = spec["sigmas"]
    clean, noise, label = _case_inputs(case, device)
    noised, timestep, labels, target = _mixed_sigma_inputs(
        clean,
        noise,
        label,
        sigmas,
        num_train_timesteps=model_num_train_timesteps(model),
    )
    del clean, noise
    with torch.inference_mode():
        native_output, native_losses = _model_losses(
            model, noised, timestep, labels, target
        )
        mode_losses = {}
        native_captures = None
        native_override_output = None
        for name, (selection, weight, default_shift) in MODE_DEFINITIONS.items():
            shift = (
                protocol["phase_shuffle_offset"]
                if name == "shuffled_phase_phase"
                else default_shift
            )
            with FactorialRouterOverride(
                model,
                selection_source=selection,
                weight_source=weight,
                phase_shift=shift,
            ) as override:
                output, losses = _model_losses(
                    model, noised, timestep, labels, target
                )
            mode_losses[name] = losses.double().cpu().tolist()
            if name == "phase_phase":
                native_captures = override.captures
                native_override_output = output
        if native_override_output is None or native_captures is None:
            raise RuntimeError("Native factorial mode did not run")
        native_prediction = _extract_prediction(native_output, target.shape[1])
        override_prediction = _extract_prediction(
            native_override_output, target.shape[1]
        )
        output_change = float(
            (native_prediction.double() - override_prediction.double()).abs().max()
        )
        loss_change = float(
            (
                native_losses.double().cpu()
                - torch.tensor(mode_losses["phase_phase"], dtype=torch.float64)
            ).abs().max()
        )
    return {
        "mode_mse": mode_losses,
        "native_unwrapped_mse": native_losses.double().cpu().tolist(),
        "native_override_max_abs_output_change": output_change,
        "native_override_max_abs_mse_change": loss_change,
        "route_rows": _route_capture_rows(
            native_captures, sigmas, spec["block_indices"]
        ),
    }


def model_num_train_timesteps(model):
    phase_metrics = [
        block.mlp.phase_metric
        for block in model.blocks
        if getattr(block, "use_moe", False)
    ]
    if not phase_metrics or any(metric is None for metric in phase_metrics):
        raise ValueError("Candidate model must have phase metrics in every MoE block")
    values = {float(metric.num_train_timesteps) for metric in phase_metrics}
    if len(values) != 1:
        raise ValueError("Candidate phase metrics disagree on timestep count")
    return values.pop()


def _deterministic_token_subset(indices, count, case_id, sigma, block_index):
    candidates = [int(value) for value in indices.detach().cpu().tolist()]
    ranked = sorted(
        candidates,
        key=lambda token: hashlib.sha256(
            (
                f"phase-metric-50k|{case_id}|{float(sigma):.8f}|"
                f"{int(block_index)}|{token}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    selected = ranked[: min(int(count), len(ranked))]
    return torch.tensor(selected, device=indices.device, dtype=torch.long)


def run_exact_dispatch_case(model, case, spec, device):
    protocol = spec["protocol"]
    clean, noise, label = _case_inputs(case, device)
    captures = {
        block_index: RouteInputCapture(model.blocks[block_index].mlp)
        for block_index in spec["block_indices"]
    }
    records = []
    cell_rows = []
    noop_max = 0.0
    try:
        for sigma in spec["sigmas"]:
            noised, timestep, labels, target = _single_sigma_inputs(
                clean,
                noise,
                label,
                sigma,
                model_num_train_timesteps(model),
            )
            for capture in captures.values():
                capture.start()
            try:
                with torch.inference_mode():
                    output, losses = _model_losses(
                        model, noised, timestep, labels, target
                    )
                native_loss = float(losses[0])
                if native_loss <= 0.0:
                    raise RuntimeError("Exact counterfactual requires positive MSE")
            finally:
                for capture in captures.values():
                    capture.stop()

            for block_index, capture in captures.items():
                hidden_states = capture.hidden_states
                captured_labels = capture.labels
                if hidden_states is None or captured_labels is None:
                    raise RuntimeError(f"No router input captured for block {block_index}")
                moe_layer = model.blocks[block_index].mlp
                base_scores, phase_scores, conditional = _router_score_pair(
                    moe_layer,
                    hidden_states,
                    captured_labels,
                    timestep,
                )
                _require_finite_tensor(base_scores, "base router scores")
                _require_finite_tensor(phase_scores, "phase router scores")
                if not conditional.all():
                    raise RuntimeError("Exact phase gate expects conditional inputs")
                native_weights, native_indices, auxiliary_loss = _compute_router(
                    moe_layer, hidden_states, captured_labels, timestep
                )
                _require_finite_tensor(native_weights, "native router weights")
                if auxiliary_loss is not None:
                    raise RuntimeError("Frozen routing returned an auxiliary loss")
                phase_ids = torch.topk(
                    phase_scores, k=1, dim=-1
                ).indices.squeeze(-1)
                base_ids = torch.topk(
                    base_scores, k=1, dim=-1
                ).indices.squeeze(-1)
                if not torch.equal(native_indices[..., 0], phase_ids):
                    raise RuntimeError("Reconstructed phase routes differ from native")
                phase_weights = phase_scores.gather(
                    -1, phase_ids.unsqueeze(-1)
                ).squeeze(-1)
                if not torch.equal(native_weights[..., 0], phase_weights):
                    raise RuntimeError("Reconstructed phase weights differ from native")
                flipped = torch.where(base_ids[0] != phase_ids[0])[0]
                sampled = _deterministic_token_subset(
                    flipped,
                    protocol["tokens_per_cell"],
                    case["id"],
                    sigma,
                    block_index,
                )
                cell_rows.append({
                    "sigma": float(sigma),
                    "block_index": int(block_index),
                    "token_count": int(hidden_states.shape[1]),
                    "flip_count": int(flipped.numel()),
                    "sampled_flip_count": int(sampled.numel()),
                    "native_mse": native_loss,
                })

                noop_candidates = sampled
                if noop_candidates.numel() < protocol["noop_tokens_per_cell"]:
                    noop_candidates = _deterministic_token_subset(
                        torch.arange(
                            hidden_states.shape[1], device=hidden_states.device
                        ),
                        protocol["noop_tokens_per_cell"],
                        case["id"] + "-noop",
                        sigma,
                        block_index,
                    )
                else:
                    noop_candidates = noop_candidates[
                        : protocol["noop_tokens_per_cell"]
                    ]
                noop_changes = _exact_counterfactual_changes(
                    model,
                    moe_layer,
                    noised,
                    timestep,
                    labels,
                    target,
                    noop_candidates,
                    phase_ids[0, noop_candidates],
                    protocol["exact_batch_size"],
                )
                _require_finite_tensor(noop_changes, "noop counterfactual changes")
                noop_max = max(noop_max, float(noop_changes.abs().max()))
                if sampled.numel() == 0:
                    continue
                exact_changes = _exact_counterfactual_changes(
                    model,
                    moe_layer,
                    noised,
                    timestep,
                    labels,
                    target,
                    sampled,
                    base_ids[0, sampled],
                    protocol["exact_batch_size"],
                )
                _require_finite_tensor(exact_changes, "exact counterfactual changes")
                for offset, token_index in enumerate(sampled.tolist()):
                    phase_id = int(phase_ids[0, token_index])
                    base_id = int(base_ids[0, token_index])
                    phase_score_preference = float(
                        phase_scores[0, token_index, phase_id]
                        - phase_scores[0, token_index, base_id]
                    )
                    exact_change = float(exact_changes[offset])
                    records.append({
                        "sigma": float(sigma),
                        "block_index": int(block_index),
                        "token_index": int(token_index),
                        "phase_expert": phase_id,
                        "base_expert": base_id,
                        "native_phase_weight": float(
                            native_weights[0, token_index, 0]
                        ),
                        "base_score_at_phase_expert": float(
                            base_scores[0, token_index, phase_id]
                        ),
                        "base_score_at_base_expert": float(
                            base_scores[0, token_index, base_id]
                        ),
                        "phase_score_at_phase_expert": float(
                            phase_scores[0, token_index, phase_id]
                        ),
                        "phase_score_at_base_expert": float(
                            phase_scores[0, token_index, base_id]
                        ),
                        "phase_score_preference": phase_score_preference,
                        "exact_base_minus_phase_mse": exact_change,
                        "exact_phase_route_relative_gain": exact_change / native_loss,
                    })
    finally:
        for capture in captures.values():
            capture.close()
    return {
        "records": records,
        "cells": cell_rows,
        "noop_max_abs_mse_change": noop_max,
    }


def run_base_case(model, case, sigmas, num_train_timesteps, device):
    clean, noise, label = _case_inputs(case, device)
    noised, timestep, labels, target = _mixed_sigma_inputs(
        clean, noise, label, sigmas, num_train_timesteps
    )
    with torch.inference_mode():
        _, losses = _model_losses(model, noised, timestep, labels, target)
    return losses.double().cpu().tolist()


def _validate_candidate_model(model, spec):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The checkpoint gate requires a frozen eval model")
    contracts = []
    for block_index, block in enumerate(model.blocks):
        if not getattr(block, "use_moe", False):
            continue
        moe = block.mlp
        routing = spec["protocol"]["routing_contract"]
        if (
            int(moe.num_routed_experts) != routing["num_routed_experts"]
            or int(moe.top_k) != routing["top_k"]
            or str(moe.router_weight_mode) != routing["router_weight_mode"]
            or bool(moe.use_uncond_expert) != routing["use_uncond_expert"]
        ):
            raise ValueError("Candidate routing contract differs")
        metric = getattr(moe, "phase_metric", None)
        if metric is None:
            raise ValueError("Candidate checkpoint lacks a phase metric")
        phase = spec["protocol"]["phase_metric_contract"]
        observed_phase = {
            "enabled": True,
            "rank": int(metric.rank),
            "num_fourier_bands": int(metric.num_fourier_bands),
            "num_train_timesteps": int(metric.num_train_timesteps),
            "scale": float(metric.scale),
            "shuffle_timestep": bool(moe.phase_metric_shuffle_timestep),
            "init_seed": int(metric.init_seed),
        }
        if observed_phase != phase:
            raise ValueError("Candidate Phase-Metric contract differs")
        contracts.append({
            "block_index": int(block_index),
            "num_routed_experts": int(moe.num_routed_experts),
            "hidden_size": int(moe.hidden_size),
            "top_k": int(moe.top_k),
            "router_weight_mode": str(moe.router_weight_mode),
            "phase_rank": int(metric.rank),
            "phase_fourier_bands": int(metric.num_fourier_bands),
            "phase_scale": float(metric.scale),
        })
    found = {item["block_index"] for item in contracts}
    if not set(spec["block_indices"]).issubset(found):
        raise ValueError("Not every requested block is a phase-aware MoE block")
    return contracts


def _validate_base_model(model, spec):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The checkpoint gate requires a frozen eval model")
    found = set()
    for block_index, block in enumerate(model.blocks):
        if not getattr(block, "use_moe", False):
            continue
        moe = block.mlp
        routing = spec["protocol"]["routing_contract"]
        if (
            int(moe.num_routed_experts) != routing["num_routed_experts"]
            or int(moe.top_k) != routing["top_k"]
            or str(moe.router_weight_mode) != routing["router_weight_mode"]
            or bool(moe.use_uncond_expert) != routing["use_uncond_expert"]
        ):
            raise ValueError("Base checkpoint routing contract differs")
        if getattr(moe, "phase_metric", None) is not None:
            raise ValueError("Base checkpoint unexpectedly enables phase_metric")
        if getattr(moe, "phase_metric_shuffle_timestep", False):
            raise ValueError("Base checkpoint unexpectedly shuffles phase")
        found.add(block_index)
    if not set(spec["block_indices"]).issubset(found):
        raise ValueError("Not every requested block is a Base MoE block")


def _validate_config_contract(config_path, runtime_cfg, expected_stem, spec):
    if Path(config_path).stem != expected_stem:
        raise ValueError(
            f"Expected config stem {expected_stem}, got {Path(config_path).stem}"
        )
    if runtime_cfg.model_name != spec["protocol"]["model_name"]:
        raise ValueError("Checkpoint config model_name differs from the gate")


def run_probe_split(
    candidate_checkpoint_path,
    candidate_weights_path,
    base_checkpoint_path,
    base_weights_path,
    cases,
    spec,
    device="cpu",
    num_threads=8,
    progress=None,
):
    device = torch.device(device)
    thread_config = _configure_torch_threads(int(num_threads))
    expected_step = spec["protocol"]["checkpoint_step"]
    paths = {
        "candidate": Path(candidate_checkpoint_path).resolve(),
        "candidate_weights": Path(candidate_weights_path).resolve(),
        "base": Path(base_checkpoint_path).resolve(),
        "base_weights": Path(base_weights_path).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint does not exist: {path}")
    if parse_checkpoint_step(paths["candidate"]) != expected_step:
        raise ValueError("Candidate checkpoint filename has the wrong step")
    if parse_checkpoint_step(paths["base"]) != expected_step:
        raise ValueError("Base checkpoint filename has the wrong step")

    candidate_config = resolve_config_from_checkpoint(paths["candidate"])
    candidate_cfg = load_runtime_cfg(candidate_config)
    _validate_config_contract(
        candidate_config,
        candidate_cfg,
        spec["protocol"]["candidate_config_stem"],
        spec,
    )
    start = time.perf_counter()
    candidate_model, candidate_state, candidate_step, candidate_load = (
        _load_checkpoint_model(candidate_cfg, paths["candidate_weights"], device)
    )
    if candidate_state != spec["protocol"]["checkpoint_state"]:
        raise ValueError("Candidate checkpoint state differs from the gate")
    if candidate_step != expected_step:
        raise ValueError("Candidate checkpoint payload has the wrong step")
    candidate_contract = _validate_candidate_model(candidate_model, spec)
    candidate_results = []
    for index, case in enumerate(cases, start=1):
        candidate_results.append({
            "case_id": case["id"],
            "split": case["split"],
            "label": int(case["label"]),
            "seed": int(case["seed"]),
            "latent_sha256": case["latent_sha256"],
            "factorial": run_factorial_case(candidate_model, case, spec, device),
            "exact_dispatch": run_exact_dispatch_case(
                candidate_model, case, spec, device
            ),
        })
        if progress is not None:
            progress("candidate", index, len(cases), case["id"])
    del candidate_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base_config = resolve_config_from_checkpoint(paths["base"])
    base_cfg = load_runtime_cfg(base_config)
    _validate_config_contract(
        base_config,
        base_cfg,
        spec["protocol"]["base_config_stem"],
        spec,
    )
    base_model, base_state, base_step, base_load = _load_checkpoint_model(
        base_cfg, paths["base_weights"], device
    )
    if base_state != spec["protocol"]["checkpoint_state"]:
        raise ValueError("Base checkpoint state differs from the gate")
    if base_step != expected_step:
        raise ValueError("Base checkpoint payload has the wrong step")
    _validate_base_model(base_model, spec)
    base_num_train_timesteps = float(base_cfg.num_train_timesteps)
    for index, (case, candidate_result) in enumerate(
        zip(cases, candidate_results), start=1
    ):
        candidate_result["base_checkpoint_mse"] = run_base_case(
            base_model,
            case,
            spec["sigmas"],
            base_num_train_timesteps,
            device,
        )
        if progress is not None:
            progress("base", index, len(cases), case["id"])
    del base_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "probe_version": PROBE_VERSION,
        "thread_config": thread_config,
        "device": str(device),
        "candidate_config": str(candidate_config),
        "base_config": str(base_config),
        "candidate_contract": candidate_contract,
        "candidate_load_seconds": float(candidate_load),
        "base_load_seconds": float(base_load),
        "total_seconds": float(time.perf_counter() - start),
        "cases": candidate_results,
    }


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(_rankdata(left), _rankdata(right))[0, 1])


def factorial_contrasts(mode_mse, base_checkpoint_mse):
    expected = set(MODE_DEFINITIONS)
    if set(mode_mse) != expected:
        raise ValueError("Factorial modes differ from the locked design")
    means = {
        name: float(np.mean(np.asarray(values, dtype=np.float64)))
        for name, values in mode_mse.items()
    }
    if any(not math.isfinite(value) or value <= 0 for value in means.values()):
        raise ValueError("Factorial MSE values must be finite and positive")
    base_checkpoint_mean = float(
        np.mean(np.asarray(base_checkpoint_mse, dtype=np.float64))
    )
    if not math.isfinite(base_checkpoint_mean) or base_checkpoint_mean <= 0:
        raise ValueError("Base checkpoint MSE must be finite and positive")
    phase_phase = means["phase_phase"]
    phase_base = means["phase_base"]
    base_phase = means["base_phase"]
    base_base = means["base_base"]
    shuffled = means["shuffled_phase_phase"]
    selection_gain = 0.5 * (
        (base_base - phase_base) + (base_phase - phase_phase)
    ) / base_base
    weight_gain = 0.5 * (
        (base_base - base_phase) + (phase_base - phase_phase)
    ) / base_base
    return {
        "selection_relative_gain": selection_gain,
        "weight_relative_gain": weight_gain,
        "selection_weight_interaction": (
            base_base - phase_base - base_phase + phase_phase
        ) / base_base,
        "native_vs_base_base_relative_gain": (
            base_base - phase_phase
        ) / base_base,
        "native_vs_shuffled_relative_gain": (
            shuffled - phase_phase
        ) / shuffled,
        "candidate_vs_base_relative_gain": (
            base_checkpoint_mean - phase_phase
        ) / base_checkpoint_mean,
        "mode_mean_mse": means,
        "base_checkpoint_mean_mse": base_checkpoint_mean,
    }


def _image_row(case_result):
    _require_finite_payload(case_result, case_result.get("case_id", "case"))
    contrasts = factorial_contrasts(
        case_result["factorial"]["mode_mse"],
        case_result["base_checkpoint_mse"],
    )
    route_rows = case_result["factorial"]["route_rows"]
    token_count = sum(row["token_count"] for row in route_rows)
    flip_count = sum(row["flip_count"] for row in route_rows)
    exact_records = case_result["exact_dispatch"]["records"]
    if exact_records:
        exact_gain = float(np.mean([
            record["exact_phase_route_relative_gain"]
            for record in exact_records
        ]))
        win_rate = float(np.mean([
            record["exact_base_minus_phase_mse"] > 0
            for record in exact_records
        ]))
        score_utility_spearman = _correlation(
            [record["phase_score_preference"] for record in exact_records],
            [record["exact_base_minus_phase_mse"] for record in exact_records],
        )
    else:
        exact_gain = 0.0
        win_rate = 0.5
        score_utility_spearman = 0.0
    row = {
        "case_id": case_result["case_id"],
        "exact_phase_route_relative_gain": exact_gain,
        "phase_route_win_rate": win_rate,
        "route_score_utility_spearman": score_utility_spearman,
        "route_flip_fraction": flip_count / token_count,
        "exact_probe_count": len(exact_records),
        "noop_max_abs_mse_change": case_result["exact_dispatch"][
            "noop_max_abs_mse_change"
        ],
        "native_override_max_abs_output_change": case_result["factorial"][
            "native_override_max_abs_output_change"
        ],
        "native_override_max_abs_mse_change": case_result["factorial"][
            "native_override_max_abs_mse_change"
        ],
        **contrasts,
    }
    for key in BOOTSTRAP_KEYS:
        if not math.isfinite(row[key]):
            raise RuntimeError(f"{row['case_id']}.{key} is non-finite")
    return row


def _bootstrap_summary(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Image bootstrap requires a finite vector")
    generator = np.random.default_rng(int(seed))
    means = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), 10_000):
        stop = min(start + 10_000, int(resamples))
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


def _check(observed, required, passed):
    return {
        "observed": float(observed),
        "required": float(required),
        "passed": bool(passed),
    }


def build_gate_summary(case_results, spec, split):
    protocol = spec["protocol"]
    if split not in protocol["split_counts"]:
        raise ValueError(f"Unknown split: {split}")
    expected = protocol["split_counts"][split]
    if len(case_results) != expected:
        raise ValueError(f"Expected {expected} {split} cases")
    image_rows = [_image_row(result) for result in case_results]
    bootstrap = {
        key: _bootstrap_summary(
            [row[key] for row in image_rows],
            protocol["bootstrap_resamples"],
            protocol["bootstrap_seeds"][split] + offset,
        )
        for offset, key in enumerate(BOOTSTRAP_KEYS)
    }
    requirements = protocol["requirements"][split]
    exact_probe_count = sum(row["exact_probe_count"] for row in image_rows)
    noop_max = max(row["noop_max_abs_mse_change"] for row in image_rows)
    override_max = max(
        row["native_override_max_abs_output_change"] for row in image_rows
    )
    override_mse_max = max(
        row["native_override_max_abs_mse_change"] for row in image_rows
    )
    checks = {
        "mean_route_flip_fraction": _check(
            bootstrap["route_flip_fraction"]["mean"],
            requirements["minimum_mean_route_flip_fraction"],
            bootstrap["route_flip_fraction"]["mean"]
            >= requirements["minimum_mean_route_flip_fraction"],
        ),
        "exact_probe_count": _check(
            exact_probe_count,
            requirements["minimum_exact_probe_count"],
            exact_probe_count >= requirements["minimum_exact_probe_count"],
        ),
        "mean_exact_phase_route_relative_gain": _check(
            bootstrap["exact_phase_route_relative_gain"]["mean"],
            requirements["minimum_mean_exact_phase_route_relative_gain"],
            bootstrap["exact_phase_route_relative_gain"]["mean"]
            >= requirements["minimum_mean_exact_phase_route_relative_gain"],
        ),
        "mean_phase_route_win_rate": _check(
            bootstrap["phase_route_win_rate"]["mean"],
            requirements["minimum_mean_phase_route_win_rate"],
            bootstrap["phase_route_win_rate"]["mean"]
            >= requirements["minimum_mean_phase_route_win_rate"],
        ),
        "mean_selection_relative_gain": _check(
            bootstrap["selection_relative_gain"]["mean"],
            requirements["minimum_mean_selection_relative_gain"],
            bootstrap["selection_relative_gain"]["mean"]
            >= requirements["minimum_mean_selection_relative_gain"],
        ),
        "mean_native_vs_base_base_relative_gain": _check(
            bootstrap["native_vs_base_base_relative_gain"]["mean"],
            requirements["minimum_mean_native_vs_base_base_relative_gain"],
            bootstrap["native_vs_base_base_relative_gain"]["mean"]
            >= requirements["minimum_mean_native_vs_base_base_relative_gain"],
        ),
        "mean_native_vs_shuffled_relative_gain": _check(
            bootstrap["native_vs_shuffled_relative_gain"]["mean"],
            requirements["minimum_mean_native_vs_shuffled_relative_gain"],
            bootstrap["native_vs_shuffled_relative_gain"]["mean"]
            >= requirements["minimum_mean_native_vs_shuffled_relative_gain"],
        ),
        "mean_candidate_vs_base_relative_gain": _check(
            bootstrap["candidate_vs_base_relative_gain"]["mean"],
            requirements["minimum_mean_candidate_vs_base_relative_gain"],
            bootstrap["candidate_vs_base_relative_gain"]["mean"]
            >= requirements["minimum_mean_candidate_vs_base_relative_gain"],
        ),
        "noop_abs_mse_change": _check(
            noop_max,
            requirements["maximum_noop_abs_mse_change"],
            noop_max <= requirements["maximum_noop_abs_mse_change"],
        ),
        "native_override_abs_output_change": _check(
            override_max,
            requirements["maximum_native_override_abs_output_change"],
            override_max
            <= requirements["maximum_native_override_abs_output_change"],
        ),
        "native_override_abs_mse_change": _check(
            override_mse_max,
            requirements["maximum_native_override_abs_mse_change"],
            override_mse_max
            <= requirements["maximum_native_override_abs_mse_change"],
        ),
    }
    optional_lcbs = {
        "lcb_exact_phase_route_relative_gain": (
            "exact_phase_route_relative_gain",
            "minimum_lcb_exact_phase_route_relative_gain",
        ),
        "lcb_selection_relative_gain": (
            "selection_relative_gain",
            "minimum_lcb_selection_relative_gain",
        ),
        "lcb_native_vs_base_base_relative_gain": (
            "native_vs_base_base_relative_gain",
            "minimum_lcb_native_vs_base_base_relative_gain",
        ),
        "lcb_native_vs_shuffled_relative_gain": (
            "native_vs_shuffled_relative_gain",
            "minimum_lcb_native_vs_shuffled_relative_gain",
        ),
    }
    for check_name, (metric, requirement_name) in optional_lcbs.items():
        if requirement_name not in requirements:
            continue
        observed = bootstrap[metric]["one_sided_lcb95"]
        required = requirements[requirement_name]
        checks[check_name] = _check(observed, required, observed >= required)

    strata = {"by_block": {}, "by_sigma": {}}
    exact_rows = [
        {**record, "case_id": result["case_id"]}
        for result in case_results
        for record in result["exact_dispatch"]["records"]
    ]
    cell_rows = [
        row
        for result in case_results
        for row in result["exact_dispatch"]["cells"]
    ]
    for block in spec["block_indices"]:
        records = [row for row in exact_rows if row["block_index"] == block]
        cells = [row for row in cell_rows if row["block_index"] == block]
        strata["by_block"][str(block)] = {
            "exact_probe_count": len(records),
            "mean_exact_phase_route_relative_gain": (
                float(np.mean([
                    row["exact_phase_route_relative_gain"] for row in records
                ])) if records else 0.0
            ),
            "route_flip_fraction": (
                sum(row["flip_count"] for row in cells)
                / sum(row["token_count"] for row in cells)
            ),
        }
    for sigma in spec["sigmas"]:
        records = [row for row in exact_rows if row["sigma"] == sigma]
        cells = [row for row in cell_rows if row["sigma"] == sigma]
        strata["by_sigma"][str(sigma)] = {
            "exact_probe_count": len(records),
            "mean_exact_phase_route_relative_gain": (
                float(np.mean([
                    row["exact_phase_route_relative_gain"] for row in records
                ])) if records else 0.0
            ),
            "route_flip_fraction": (
                sum(row["flip_count"] for row in cells)
                / sum(row["token_count"] for row in cells)
            ),
        }
    passed = all(item["passed"] for item in checks.values())
    if passed:
        decision = (
            "authorize_confirmatory"
            if split == "discovery"
            else "authorize_continue_training"
        )
    else:
        decision = "stop_or_redesign_phase_metric"
    return {
        "passed": bool(passed),
        "decision": decision,
        "split": split,
        "checks": checks,
        "requirements": requirements,
        "bootstrap": bootstrap,
        "image_rows": image_rows,
        "strata": strata,
        "exact_probe_count": exact_probe_count,
    }
