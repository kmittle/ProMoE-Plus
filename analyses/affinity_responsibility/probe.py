"""Checkpoint-backed probe for ProMoE RCL and routed responsibility."""

from __future__ import annotations

import gc
import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

from analyses.denoising_regret.probe import (
    RoutingProbeCapture,
    _compute_router,
    _configure_torch_threads,
    _build_model,
    _evaluate_experts,
    _extract_prediction,
    _load_latent,
    _per_sample_mse,
)
from analyses.denoising_regret.responsibility_probe import (
    _exact_global_weight_changes,
    _exact_token_weight_changes,
    _scale_key,
    summarize_global_records,
    summarize_responsibility_records,
)
from analyses.finite_horizon_routing.probe import (
    _verified_checkpoint_for_loading,
    _verified_latent_for_loading,
)
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)

from .protocol import (
    ASSIGNMENT_SHUFFLE_COUNT,
    BLOCK_INDICES,
    CANDIDATE_SCALES,
    CENTER_HALF_STEP_MULTIPLIER,
    CENTER_STEP_RELATIVE_FROBENIUS,
    EXACT_BATCH_SIZE,
    PROBE_VERSION,
    SIGMA_VALUES,
    SUPPORT_BATCH_SIZE,
    SUPPORT_FORWARD_BATCH_SIZE,
    SUPPORT_GROUP_COUNT,
    SUPPORT_SIGMA_POLICY,
    TOKEN_PROBE_COUNT,
    count_preserving_assignment_shuffles,
    norm_preserving_center_step,
    responsibility_center_gradient,
    routing_contrastive_center_gradient,
    summarize_external_rcl_gradient_cell,
)


ONLINE_CHECKPOINT_STATE = "model_state_dict"


def _derived_seed(seed, block_index, sigma, purpose):
    payload = f"{int(seed)}|{int(block_index)}|{float(sigma):.17g}|{purpose}"
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63 - 1)


def _validate_probe_contract(model, runtime_cfg, block_indices):
    if runtime_cfg.model_name != "ProMoE_TC_B":
        raise ValueError("The locked mechanism gate requires ProMoE_TC_B")
    if tuple(block_indices) != BLOCK_INDICES:
        raise ValueError(f"The locked mechanism gate requires blocks {BLOCK_INDICES}")
    if len(model.blocks) <= max(block_indices):
        raise ValueError("The checkpoint model is missing a locked MoE block")
    expected_sigma_contract = {
        "weighting_scheme": SUPPORT_SIGMA_POLICY["distribution"],
        "logit_mean": SUPPORT_SIGMA_POLICY["logit_mean"],
        "logit_std": SUPPORT_SIGMA_POLICY["logit_std"],
        "sigmoid_scale": SUPPORT_SIGMA_POLICY["sigmoid_scale"],
        "shift": SUPPORT_SIGMA_POLICY["shift"],
        "num_train_timesteps": SUPPORT_SIGMA_POLICY["num_train_timesteps"],
    }
    observed_sigma_contract = {
        key: getattr(runtime_cfg, key) for key in expected_sigma_contract
    }
    if observed_sigma_contract != expected_sigma_contract:
        raise ValueError(
            "Fresh training sigma policy differs from the locked support policy: "
            f"{observed_sigma_contract}"
        )
    for block_index in block_indices:
        block = model.blocks[block_index]
        if not block.use_moe:
            raise ValueError(f"Locked block {block_index} is not an MoE block")
        moe_layer = block.mlp
        expected = {
            "top_k": 1,
            "router_weight_mode": "identity",
            "use_shared_expert": True,
            "routing_contrastive_lam": 1,
            "use_top_k_for_routing_contrastive": True,
        }
        observed = {
            key: getattr(moe_layer, key)
            for key in expected
        }
        if observed != expected:
            raise ValueError(
                f"Locked block {block_index} contract changed: {observed}"
            )
        if getattr(moe_layer, "phase_metric", None) is not None:
            raise ValueError("The Fresh Base mechanism gate forbids phase routing")
        temperature = float(moe_layer.routing_contrastive_temperature)
        if temperature != 0.07:
            raise ValueError(
                f"Locked block {block_index} RCL temperature is {temperature}"
            )


def _load_online_checkpoint_model(runtime_cfg, checkpoint_handle, device):
    model = _build_model(runtime_cfg)
    load_start = time.perf_counter()
    load_kwargs = {"map_location": "cpu", "weights_only": True}
    checkpoint_handle.seek(0)
    try:
        checkpoint = torch.load(checkpoint_handle, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only")
        checkpoint_handle.seek(0)
        checkpoint = torch.load(checkpoint_handle, **load_kwargs)
    checkpoint_handle.seek(0)
    if ONLINE_CHECKPOINT_STATE not in checkpoint:
        raise KeyError(f"Checkpoint is missing {ONLINE_CHECKPOINT_STATE}")
    checkpoint_step = checkpoint.get("step")
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int):
        raise ValueError("Checkpoint must contain an integer step")
    state_dict = checkpoint[ONLINE_CHECKPOINT_STATE]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={missing}, unexpected={unexpected}"
        )
    del state_dict
    del checkpoint
    gc.collect()
    model = model.to(device).eval().requires_grad_(False)
    return (
        model,
        ONLINE_CHECKPOINT_STATE,
        checkpoint_step,
        time.perf_counter() - load_start,
    )


class _SupportRoutingCapture:
    def __init__(self, moe_layer):
        self.enabled = False
        self.hidden_states = None
        self.labels = None
        self._handle = moe_layer.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        if not self.enabled:
            return None
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


def _load_support_sample(case, device):
    with _verified_latent_for_loading(
        case["latent"],
        expected_size=case["latent_size"],
        expected_sha256=case["latent_sha256"],
    ) as (latent_handle, latent_identity):
        latent = _load_latent(
            latent_handle,
            case["latent_key"],
            int(case["seed"]),
            device,
        )
    expected_identity = {
        "size": int(case["latent_size"]),
        "sha256": case["latent_sha256"],
    }
    if latent_identity != expected_identity:
        raise RuntimeError("Support latent identity changed after protocol sealing")
    torch.manual_seed(int(case["seed"]) + 1)
    noise = torch.randn_like(latent)
    label = 1000 if case["unconditional"] else int(case["label"])
    return latent, noise, label


def build_rank_local_support_rcl(
    model,
    support_cases,
    support_group_index,
    device,
):
    """Compute one rank's prototype-only RCL gradient from a training-like batch."""

    support_cases = list(support_cases)
    if len(support_cases) != SUPPORT_BATCH_SIZE:
        raise ValueError(
            f"A support group must contain {SUPPORT_BATCH_SIZE} images"
        )
    if {int(case["group_index"]) for case in support_cases} != {
        int(support_group_index)
    }:
        raise ValueError("Support cases do not belong to one locked group")
    if sum(bool(case["unconditional"]) for case in support_cases) != 6:
        raise ValueError("A support group must contain exactly six uncond images")
    captures = {
        block_index: _SupportRoutingCapture(model.blocks[block_index].mlp)
        for block_index in BLOCK_INDICES
    }
    support_results = {}
    try:
        hidden_parts = {block_index: [] for block_index in BLOCK_INDICES}
        assignment_parts = {block_index: [] for block_index in BLOCK_INDICES}
        for start in range(0, len(support_cases), SUPPORT_FORWARD_BATCH_SIZE):
            chunk = support_cases[start:start + SUPPORT_FORWARD_BATCH_SIZE]
            samples = [_load_support_sample(case, device) for case in chunk]
            clean_latent = torch.cat([item[0] for item in samples], dim=0)
            noise = torch.cat([item[1] for item in samples], dim=0)
            labels = torch.tensor(
                [item[2] for item in samples],
                device=device,
                dtype=torch.long,
            )
            sigmas = torch.tensor(
                [float(case["sigma"]) for case in chunk],
                device=device,
                dtype=clean_latent.dtype,
            )
            noised_latent = (
                (1.0 - sigmas).view(-1, 1, 1, 1, 1) * clean_latent
                + sigmas.view(-1, 1, 1, 1, 1) * noise
            )
            timestep = sigmas * 1000.0
            for capture in captures.values():
                capture.start()
            with torch.no_grad():
                model(noised_latent, timestep, context=labels)
            for capture in captures.values():
                capture.stop()
            for block_index, capture in captures.items():
                if capture.hidden_states is None or capture.labels is None:
                    raise RuntimeError(
                        f"Support capture failed at block {block_index}"
                    )
                moe_layer = model.blocks[block_index].mlp
                with torch.no_grad():
                    _, expert_indices, _ = _compute_router(
                        moe_layer,
                        capture.hidden_states,
                        capture.labels,
                        timestep,
                    )
                conditional = capture.labels != 1000
                hidden_parts[block_index].append(
                    capture.hidden_states[conditional].reshape(
                        -1,
                        capture.hidden_states.shape[-1],
                    )
                )
                assignment_parts[block_index].append(
                    expert_indices[conditional, :, 0].reshape(-1)
                )
            del clean_latent, noise, noised_latent, timestep, labels, sigmas

        for block_index in BLOCK_INDICES:
            hidden_states = torch.cat(hidden_parts[block_index], dim=0)
            assignments = torch.cat(assignment_parts[block_index], dim=0)
            expected_tokens = (SUPPORT_BATCH_SIZE - 6) * 256
            if hidden_states.shape[0] != expected_tokens:
                raise ValueError(
                    f"Support block {block_index} has "
                    f"{hidden_states.shape[0]} conditional tokens"
                )
            moe_layer = model.blocks[block_index].mlp
            centers = moe_layer.cluster_centers.detach()
            correct = routing_contrastive_center_gradient(
                hidden_states,
                centers,
                assignments,
                moe_layer.routing_contrastive_temperature,
            )
            shuffle_seed = _derived_seed(
                support_group_index,
                block_index,
                0.0,
                "mixed-sigma-rank-local-support-shuffles",
            )
            shuffled_assignments = count_preserving_assignment_shuffles(
                assignments.detach().cpu().numpy(),
                ASSIGNMENT_SHUFFLE_COUNT,
                shuffle_seed,
            )
            native_counts = np.bincount(
                assignments.detach().cpu().numpy(),
                minlength=centers.shape[0],
            )
            count_mismatches = 0
            shuffled = []
            for shuffled_assignment in shuffled_assignments:
                shuffled_counts = np.bincount(
                    shuffled_assignment,
                    minlength=centers.shape[0],
                )
                count_mismatches += int(
                    not np.array_equal(native_counts, shuffled_counts)
                )
                shuffled.append(routing_contrastive_center_gradient(
                    hidden_states,
                    centers,
                    torch.as_tensor(
                        shuffled_assignment,
                        device=device,
                        dtype=torch.long,
                    ),
                    moe_layer.routing_contrastive_temperature,
                ))

            def cpu_record(record, mismatches=0):
                return {
                    "loss": float(record["loss"]),
                    "gradient": record["gradient"].detach().cpu().numpy(),
                    "valid_experts": [
                        int(value)
                        for value in record["valid_experts"].detach().cpu().tolist()
                    ],
                    "assignment_count_mismatches": int(mismatches),
                    "occupied_expert_count": int(record["valid_experts"].numel()),
                }

            support_results[block_index] = {
                "correct": cpu_record(correct, count_mismatches),
                "shuffled": [cpu_record(item) for item in shuffled],
                "support_group_index": int(support_group_index),
                "support_image_count": SUPPORT_BATCH_SIZE,
                "support_conditional_image_count": SUPPORT_BATCH_SIZE - 6,
                "support_token_count": int(hidden_states.shape[0]),
                "support_sigma_min": min(float(case["sigma"]) for case in support_cases),
                "support_sigma_max": max(float(case["sigma"]) for case in support_cases),
                "support_sigma_mean": float(np.mean([
                    float(case["sigma"]) for case in support_cases
                ])),
                "shuffle_seed": int(shuffle_seed),
            }
            del hidden_states, assignments
    finally:
        for capture in captures.values():
            capture.close()
    return support_results


def aggregate_rank_support_rcl(rank_results):
    """Average four rank-local RCL losses/gradients exactly as DDP does."""

    if not isinstance(rank_results, dict):
        raise TypeError("rank_results must map support-group index to results")
    expected_groups = set(range(SUPPORT_GROUP_COUNT))
    if {int(group) for group in rank_results} != expected_groups:
        raise ValueError("The DDP support aggregate requires all four rank groups")
    ordered = [rank_results[group] for group in range(SUPPORT_GROUP_COUNT)]
    if any(set(result) != set(BLOCK_INDICES) for result in ordered):
        raise ValueError("Every rank must cover every locked MoE block")

    def mean_record(records):
        gradients = [
            np.asarray(record["gradient"], dtype=np.float64)
            for record in records
        ]
        if (
            not gradients
            or len({gradient.shape for gradient in gradients}) != 1
            or not all(np.isfinite(gradient).all() for gradient in gradients)
        ):
            raise ValueError("Rank support gradients must be finite and shape-aligned")
        return {
            "loss": float(np.mean([float(record["loss"]) for record in records])),
            "gradient": np.stack(gradients, axis=0).mean(axis=0),
            "valid_experts": sorted({
                int(expert)
                for record in records
                for expert in record["valid_experts"]
            }),
            "assignment_count_mismatches": int(sum(
                int(record.get("assignment_count_mismatches", 0))
                for record in records
            )),
            "occupied_expert_count": int(min(
                int(record["occupied_expert_count"]) for record in records
            )),
        }

    aggregated = {}
    for block_index in BLOCK_INDICES:
        local = [result[block_index] for result in ordered]
        shuffle_counts = {len(item["shuffled"]) for item in local}
        if shuffle_counts != {ASSIGNMENT_SHUFFLE_COUNT}:
            raise ValueError("Every rank must provide every locked shuffle")
        aggregated[block_index] = {
            "correct": mean_record([item["correct"] for item in local]),
            "shuffled": [
                mean_record([item["shuffled"][index] for item in local])
                for index in range(ASSIGNMENT_SHUFFLE_COUNT)
            ],
            "support_group_indices": list(range(SUPPORT_GROUP_COUNT)),
            "support_gradient_aggregation": "ddp_mean",
            "support_rank_count": SUPPORT_GROUP_COUNT,
            "support_image_count": sum(
                int(item["support_image_count"]) for item in local
            ),
            "support_conditional_image_count": sum(
                int(item["support_conditional_image_count"]) for item in local
            ),
            "support_token_count": sum(
                int(item["support_token_count"]) for item in local
            ),
            "support_sigma_min": min(
                float(item["support_sigma_min"]) for item in local
            ),
            "support_sigma_max": max(
                float(item["support_sigma_max"]) for item in local
            ),
            "support_sigma_mean": float(np.mean([
                float(item["support_sigma_mean"]) for item in local
            ])),
            "shuffle_seeds": [int(item["shuffle_seed"]) for item in local],
        }
    return aggregated


def _token_probe_indices(num_tokens, count, seed, device):
    num_tokens = int(num_tokens)
    count = int(count)
    if num_tokens < count or count < 2:
        raise ValueError("The locked token probe count does not fit the sequence")
    generator = np.random.default_rng(int(seed))
    indices = generator.permutation(num_tokens)[:count].copy()
    return torch.as_tensor(indices, device=device, dtype=torch.long)


def _mechanism_identity_max(mechanism):
    if mechanism.get("valid") is not True:
        return None
    values = [
        mechanism["correct"]["diffusion_gradient_identity_relative_error"],
        mechanism["diffusion_only_control"][
            "diffusion_gradient_identity_relative_error"
        ],
    ]
    values.extend(
        item["diffusion_gradient_identity_relative_error"]
        for item in mechanism["shuffled"]
    )
    return float(max(values))


@contextmanager
def _fixed_dispatch_center_weights(
    moe_layer,
    center_batch,
    expected_assignments,
):
    """Recompute selected cosines from perturbed centers without redispatching."""

    original_compute_router = moe_layer.compute_router
    center_batch = torch.as_tensor(
        center_batch,
        device=moe_layer.cluster_centers.device,
        dtype=moe_layer.cluster_centers.dtype,
    )
    expected_assignments = torch.as_tensor(
        expected_assignments,
        device=moe_layer.cluster_centers.device,
        dtype=torch.long,
    )
    if center_batch.ndim != 3:
        raise ValueError("center_batch must have shape [batch, experts, hidden]")
    if center_batch.shape[1:] != moe_layer.cluster_centers.shape:
        raise ValueError("Perturbed centers do not match the target MoE layer")
    if expected_assignments.ndim != 1:
        raise ValueError("expected_assignments must name one expert per token")
    statistics = {"fixed_dispatch_mismatches": 0, "forward_calls": 0}

    def compute_router_with_perturbed_weights(
        this,
        hidden_states,
        labels,
        timestep=None,
    ):
        if timestep is None:
            weights, indices, auxiliary_loss = original_compute_router(
                hidden_states,
                labels,
            )
        else:
            weights, indices, auxiliary_loss = original_compute_router(
                hidden_states,
                labels,
                timestep,
            )
        if hidden_states.shape[0] != center_batch.shape[0]:
            raise RuntimeError("Center intervention batch size changed")
        if hidden_states.shape[1] != expected_assignments.numel():
            raise RuntimeError("Center intervention token count changed")
        if weights.shape[-1] != 1:
            raise RuntimeError("Center interventions require top_k == 1")
        conditional_rows = labels != 1000
        conditional_count = int(conditional_rows.sum().item())
        if conditional_count:
            conditional_indices = indices[conditional_rows, :, 0]
            expected = expected_assignments.unsqueeze(0).expand(
                conditional_count,
                -1,
            )
            mismatch_count = int((conditional_indices != expected).sum().item())
            statistics["fixed_dispatch_mismatches"] += mismatch_count
            if mismatch_count:
                raise RuntimeError(
                    "Native dispatch changed during a fixed-dispatch intervention"
                )
            conditional_hidden = hidden_states[conditional_rows].float()
            conditional_centers = center_batch[conditional_rows].float()
            center_unit = F.normalize(conditional_centers, p=2, dim=-1)
            gathered_centers = torch.gather(
                center_unit,
                1,
                conditional_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    center_unit.shape[-1],
                ),
            )
            selected_scores = (
                F.normalize(conditional_hidden, p=2, dim=-1)
                * gathered_centers
            ).sum(dim=-1)
            weights[conditional_rows, :, 0] = selected_scores.to(weights.dtype)
        statistics["forward_calls"] += 1
        return weights, indices, auxiliary_loss

    if "compute_router" in moe_layer.__dict__:
        raise RuntimeError("MoE layer already has a compute_router override")
    moe_layer.compute_router = MethodType(
        compute_router_with_perturbed_weights,
        moe_layer,
    )
    try:
        yield statistics
    finally:
        del moe_layer.compute_router


def _exact_center_weight_changes(
    model,
    moe_layer,
    noised_latent,
    timestep,
    label,
    target,
    expected_assignments,
    named_centers,
    batch_size=EXACT_BATCH_SIZE,
):
    """Measure paired MSE changes for fixed-dispatch center perturbations."""

    named_centers = list(named_centers)
    if not named_centers:
        raise ValueError("At least one center intervention is required")
    names = [name for name, _ in named_centers]
    if len(names) != len(set(names)):
        raise ValueError("Center intervention names must be unique")
    changes = {}
    fixed_dispatch_mismatches = 0
    target_channels = target.shape[1]
    for start in range(0, len(named_centers), int(batch_size)):
        chunk = named_centers[start:start + int(batch_size)]
        count = len(chunk)
        batch_latent = noised_latent.repeat(count, 1, 1, 1, 1)
        batch_timestep = timestep.repeat(count)
        batch_label = label.repeat(count)
        batch_target = target.repeat(count, 1, 1, 1)
        center_batch = torch.stack([centers for _, centers in chunk])
        with torch.inference_mode():
            base_output = model(batch_latent, batch_timestep, context=batch_label)
            base_prediction = _extract_prediction(base_output, target_channels)
            base_losses = _per_sample_mse(base_prediction, batch_target)
            with _fixed_dispatch_center_weights(
                moe_layer,
                center_batch,
                expected_assignments,
            ) as statistics:
                alternative_output = model(
                    batch_latent,
                    batch_timestep,
                    context=batch_label,
                )
            alternative_prediction = _extract_prediction(
                alternative_output,
                target_channels,
            )
            alternative_losses = _per_sample_mse(
                alternative_prediction,
                batch_target,
            )
        fixed_dispatch_mismatches += statistics["fixed_dispatch_mismatches"]
        for offset, (name, _) in enumerate(chunk):
            changes[name] = float(
                (alternative_losses[offset] - base_losses[offset]).item()
            )
    return {
        "changes": changes,
        "fixed_dispatch_mismatches": int(fixed_dispatch_mismatches),
    }


def _finite_relative_error(observed, expected, scale):
    denominator = max(abs(float(observed)), abs(float(expected)), float(scale))
    return float(abs(float(observed) - float(expected)) / denominator)


def _add_exact_update_metrics(
    metrics,
    center_step,
    diffusion_gradient,
    exact_mse_change,
    base_mse,
):
    predicted = float(
        (
            diffusion_gradient.to(center_step["displacement"].device)
            * center_step["displacement"]
        ).sum().item()
    )
    numerical_floor = float(base_mse) * 1e-7
    metrics.update({
        "exact_mse_change": float(exact_mse_change),
        "exact_relative_mse_change": float(exact_mse_change / base_mse),
        "finite_step_first_order_mse_change": predicted,
        "finite_step_first_order_relative_error": _finite_relative_error(
            exact_mse_change,
            predicted,
            numerical_floor,
        ),
        "center_step": {
            key: value
            for key, value in center_step.items()
            if key not in {"centers", "displacement"}
        },
    })


def _probe_cell(
    model,
    moe_layer,
    capture,
    clean_latent,
    noise,
    label,
    sigma,
    block_index,
    num_train_timesteps,
    seed,
    support_rcl,
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

    capture.start()
    model_output = model(noised_latent, timestep, context=label)
    prediction = _extract_prediction(model_output, target_channels)
    base_loss = _per_sample_mse(prediction, target).mean()
    if capture.moe_output is None:
        raise RuntimeError("The MoE responsibility hook did not capture an output")
    moe_gradient, = torch.autograd.grad(base_loss, capture.moe_output)
    capture.stop()

    hidden_states = capture.hidden_states
    captured_labels = capture.labels
    with torch.no_grad():
        router_weights, expert_indices, _ = _compute_router(
            moe_layer,
            hidden_states,
            captured_labels,
            timestep,
        )
        native_weights = router_weights[0, :, 0].float()
        selected_experts = expert_indices[0, :, 0]
        selected_outputs = _evaluate_experts(
            moe_layer.experts[:moe_layer.num_routed_experts],
            hidden_states[0],
            selected_experts,
        )
        responsibility_slopes = (
            moe_gradient[0].float() * selected_outputs
        ).sum(dim=-1)

    base_mse = float(base_loss.item())
    if base_mse <= 0:
        raise ValueError("The native denoising MSE must be positive")
    centers = moe_layer.cluster_centers.detach()
    diffusion_gradient, reconstructed_scores, _ = responsibility_center_gradient(
        hidden_states[0],
        centers,
        selected_experts,
        responsibility_slopes,
    )
    rows = torch.arange(selected_experts.numel(), device=selected_experts.device)
    reconstructed_native = reconstructed_scores[rows, selected_experts]
    router_score_reconstruction_error = float(
        (
            reconstructed_native
            - native_weights.to(dtype=torch.float64)
        ).abs().max().item()
    )

    shuffle_seeds = [int(seed_value) for seed_value in support_rcl["shuffle_seeds"]]
    try:
        mechanism = summarize_external_rcl_gradient_cell(
            hidden_states=hidden_states[0],
            centers=centers,
            assignments=selected_experts,
            responsibility_slopes=responsibility_slopes,
            temperature=moe_layer.routing_contrastive_temperature,
            correct_support_rcl=support_rcl["correct"],
            shuffled_support_rcl=support_rcl["shuffled"],
            realized_dtype=moe_layer.cluster_centers.dtype,
        )
        diffusion_step = norm_preserving_center_step(
            centers,
            diffusion_gradient,
            realized_dtype=moe_layer.cluster_centers.dtype,
        )
        correct_step = norm_preserving_center_step(
            centers,
            support_rcl["correct"]["gradient"],
            realized_dtype=moe_layer.cluster_centers.dtype,
        )
        correct_half_step = norm_preserving_center_step(
            centers,
            support_rcl["correct"]["gradient"],
            CENTER_STEP_RELATIVE_FROBENIUS * CENTER_HALF_STEP_MULTIPLIER,
            realized_dtype=moe_layer.cluster_centers.dtype,
        )
        shuffled_steps = [
            norm_preserving_center_step(
                centers,
                item["gradient"],
                realized_dtype=moe_layer.cluster_centers.dtype,
            )
            for item in support_rcl["shuffled"]
        ]
        named_centers = [
            ("noop", centers.double()),
            ("diffusion_only", diffusion_step["centers"]),
            ("correct", correct_step["centers"]),
            ("correct_half", correct_half_step["centers"]),
        ]
        named_centers.extend(
            (f"shuffle_{index:02d}", step["centers"])
            for index, step in enumerate(shuffled_steps)
        )
        exact_center = _exact_center_weight_changes(
            model=model,
            moe_layer=moe_layer,
            noised_latent=noised_latent,
            timestep=timestep,
            label=label,
            target=target,
            expected_assignments=selected_experts,
            named_centers=named_centers,
            batch_size=EXACT_BATCH_SIZE,
        )
        exact_changes = exact_center["changes"]
        _add_exact_update_metrics(
            mechanism["diffusion_only_control"],
            diffusion_step,
            diffusion_gradient,
            exact_changes["diffusion_only"],
            base_mse,
        )
        _add_exact_update_metrics(
            mechanism["correct"],
            correct_step,
            diffusion_gradient,
            exact_changes["correct"],
            base_mse,
        )
        for index, (metrics, step) in enumerate(
            zip(mechanism["shuffled"], shuffled_steps)
        ):
            _add_exact_update_metrics(
                metrics,
                step,
                diffusion_gradient,
                exact_changes[f"shuffle_{index:02d}"],
                base_mse,
            )
        half_predicted = float(
            (
                diffusion_gradient * correct_half_step["displacement"]
            ).sum().item()
        )
        half_exact = exact_changes["correct_half"]
        mechanism["correct_half_step_control"] = {
            "exact_mse_change": float(half_exact),
            "exact_relative_mse_change": float(half_exact / base_mse),
            "finite_step_first_order_mse_change": half_predicted,
            "finite_step_first_order_relative_error": _finite_relative_error(
                half_exact,
                half_predicted,
                base_mse * 1e-7,
            ),
            "full_vs_two_half_secant_relative_error": _finite_relative_error(
                exact_changes["correct"],
                2.0 * half_exact,
                base_mse * 1e-7,
            ),
            "center_step": {
                key: value
                for key, value in correct_half_step.items()
                if key not in {"centers", "displacement"}
            },
        }
        shuffle_exact = np.asarray(
            [item["exact_mse_change"] for item in mechanism["shuffled"]],
            dtype=np.float64,
        )
        mechanism["shuffle_summary"].update({
            "exact_mse_change_mean": float(shuffle_exact.mean()),
            "exact_mse_change_std": float(shuffle_exact.std()),
            "exact_mse_change_min": float(shuffle_exact.min()),
            "exact_mse_change_max": float(shuffle_exact.max()),
            "correct_minus_shuffle_exact_mse_change": float(
                exact_changes["correct"] - shuffle_exact.mean()
            ),
            "correct_exact_harm_shuffle_percentile": float(
                np.mean(shuffle_exact < exact_changes["correct"])
                + 0.5 * np.mean(shuffle_exact == exact_changes["correct"])
            ),
        })
        mechanism.update({
            "support": {
                "group_indices": list(support_rcl["support_group_indices"]),
                "gradient_aggregation": support_rcl[
                    "support_gradient_aggregation"
                ],
                "rank_count": int(support_rcl["support_rank_count"]),
                "image_count": int(support_rcl["support_image_count"]),
                "conditional_image_count": int(
                    support_rcl["support_conditional_image_count"]
                ),
                "token_count": int(support_rcl["support_token_count"]),
                "sigma_min": float(support_rcl["support_sigma_min"]),
                "sigma_max": float(support_rcl["support_sigma_max"]),
                "sigma_mean": float(support_rcl["support_sigma_mean"]),
                "shuffle_seeds": list(support_rcl["shuffle_seeds"]),
            },
            "exact_center_noop_mse_change": float(exact_changes["noop"]),
            "exact_center_noop_relative_mse_change": float(
                abs(exact_changes["noop"]) / base_mse
            ),
            "fixed_dispatch_mismatches": int(
                exact_center["fixed_dispatch_mismatches"]
            ),
            "diffusion_only_exact_descent": bool(
                exact_changes["diffusion_only"] < 0
            ),
            "correct_exact_conflict": bool(exact_changes["correct"] > 0),
        })
        mechanism["valid"] = True
        mechanism["error"] = None
    except ValueError as error:
        mechanism = {
            "valid": False,
            "error": str(error),
            "assignment_count_mismatches": 0,
        }

    token_seed = _derived_seed(seed, block_index, sigma, "token-probes")
    token_indices = _token_probe_indices(
        hidden_states.shape[1],
        TOKEN_PROBE_COUNT,
        token_seed,
        hidden_states.device,
    )
    scale_tensor = torch.tensor(
        CANDIDATE_SCALES,
        device=hidden_states.device,
        dtype=native_weights.dtype,
    )
    scale_dispatch_statistics = {"fixed_dispatch_mismatches": 0}
    token_grid = token_indices.repeat_interleave(len(CANDIDATE_SCALES))
    route_weight_grid = scale_tensor.repeat(TOKEN_PROBE_COUNT)
    exact_grid = _exact_token_weight_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        token_indices=token_grid,
        route_weights=route_weight_grid,
        batch_size=EXACT_BATCH_SIZE,
        expected_expert_indices=selected_experts[token_grid],
        dispatch_statistics=scale_dispatch_statistics,
    ).view(TOKEN_PROBE_COUNT, len(CANDIDATE_SCALES))

    sampled_native = native_weights[token_indices]
    sampled_slopes = responsibility_slopes[token_indices]
    first_order_grid = (
        scale_tensor.unsqueeze(0) - sampled_native.unsqueeze(1)
    ) * sampled_slopes.unsqueeze(1)

    num_tokens = hidden_states.shape[1]
    global_weight_matrix = scale_tensor.unsqueeze(1).expand(-1, num_tokens)
    exact_global = _exact_global_weight_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        route_weight_matrix=global_weight_matrix,
        batch_size=EXACT_BATCH_SIZE,
        expected_expert_indices=selected_experts,
        dispatch_statistics=scale_dispatch_statistics,
    )
    first_order_global = (
        (global_weight_matrix - native_weights.unsqueeze(0))
        * responsibility_slopes.unsqueeze(0)
    ).sum(dim=1)

    noop_token_change = _exact_token_weight_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        token_indices=token_indices,
        route_weights=None,
        batch_size=EXACT_BATCH_SIZE,
        expected_expert_indices=selected_experts[token_indices],
        dispatch_statistics=scale_dispatch_statistics,
    )
    noop_global_change = _exact_global_weight_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        route_weight_matrix=None,
        batch_size=1,
        expected_expert_indices=selected_experts,
        dispatch_statistics=scale_dispatch_statistics,
    )

    mechanism["center_fixed_dispatch_mismatches"] = int(
        mechanism.get("fixed_dispatch_mismatches", 0)
    )
    mechanism["scale_fixed_dispatch_mismatches"] = int(
        scale_dispatch_statistics["fixed_dispatch_mismatches"]
    )
    mechanism["fixed_dispatch_mismatches"] = int(
        mechanism["center_fixed_dispatch_mismatches"]
        + mechanism["scale_fixed_dispatch_mismatches"]
    )

    scale_keys = [_scale_key(scale) for scale in CANDIDATE_SCALES]
    records = []
    for row in range(TOKEN_PROBE_COUNT):
        records.append({
            "token_index": int(token_indices[row].item()),
            "selected_expert": int(selected_experts[token_indices[row]].item()),
            "native_router_weight": float(sampled_native[row].item()),
            "responsibility_slope": float(sampled_slopes[row].item()),
            "first_order_mse_change": {
                key: float(first_order_grid[row, column].item())
                for column, key in enumerate(scale_keys)
            },
            "exact_mse_change": {
                key: float(exact_grid[row, column].item())
                for column, key in enumerate(scale_keys)
            },
        })
    global_record = {
        "first_order_mse_change": {
            key: float(first_order_global[column].item())
            for column, key in enumerate(scale_keys)
        },
        "exact_mse_change": {
            key: float(exact_global[column].item())
            for column, key in enumerate(scale_keys)
        },
    }
    return {
        "block_index": int(block_index),
        "sigma": float(sigma),
        "timestep": float(timestep.item()),
        "token_probe_seed": int(token_seed),
        "assignment_shuffle_seeds": shuffle_seeds,
        "base_mse": base_mse,
        "native_router_weight_mean": float(native_weights.mean().item()),
        "native_router_weight_std": float(native_weights.std().item()),
        "native_router_weight_min": float(native_weights.min().item()),
        "native_router_weight_max": float(native_weights.max().item()),
        "responsibility": summarize_responsibility_records(
            records,
            CANDIDATE_SCALES,
        ),
        "global_responsibility": summarize_global_records(
            [global_record],
            CANDIDATE_SCALES,
        ),
        "mechanism": mechanism,
        "numerical_controls": {
            "noop_token_max_abs_mse_change": float(
                noop_token_change.abs().max().item()
            ),
            "noop_token_max_relative_mse_change": float(
                noop_token_change.abs().max().item() / base_mse
            ),
            "noop_global_max_abs_mse_change": float(
                noop_global_change.abs().max().item()
            ),
            "noop_global_max_relative_mse_change": float(
                noop_global_change.abs().max().item() / base_mse
            ),
            "router_score_reconstruction_error": (
                router_score_reconstruction_error
            ),
            "diffusion_gradient_identity_relative_error": (
                _mechanism_identity_max(mechanism)
            ),
            "exact_center_noop_relative_mse_change": (
                mechanism.get("exact_center_noop_relative_mse_change")
            ),
            "fixed_dispatch_mismatches": mechanism.get(
                "fixed_dispatch_mismatches"
            ),
            "diffusion_only_exact_descent": mechanism.get(
                "diffusion_only_exact_descent"
            ),
            "correct_half_step_first_order_relative_error": (
                mechanism.get("correct_half_step_control", {}).get(
                    "finite_step_first_order_relative_error"
                )
            ),
            "correct_full_vs_two_half_secant_relative_error": (
                mechanism.get("correct_half_step_control", {}).get(
                    "full_vs_two_half_secant_relative_error"
                )
            ),
            "maximum_center_norm_relative_error": (
                max(
                    [
                        mechanism["correct"]["center_step"][
                            "maximum_center_norm_relative_error"
                        ],
                        mechanism["diffusion_only_control"]["center_step"][
                            "maximum_center_norm_relative_error"
                        ],
                        mechanism["correct_half_step_control"]["center_step"][
                            "maximum_center_norm_relative_error"
                        ],
                    ]
                    + [
                        item["center_step"][
                            "maximum_center_norm_relative_error"
                        ]
                        for item in mechanism.get("shuffled", [])
                    ]
                )
                if mechanism.get("valid") is True
                else None
            ),
        },
        "global_record": global_record,
        "records": records,
    }


def load_rcl_responsibility_probe_model(
    checkpoint_path,
    block_indices=BLOCK_INDICES,
    sigmas=SIGMA_VALUES,
    candidate_scales=CANDIDATE_SCALES,
    token_probe_count=TOKEN_PROBE_COUNT,
    exact_batch_size=EXACT_BATCH_SIZE,
    assignment_shuffle_count=ASSIGNMENT_SHUFFLE_COUNT,
    device="cpu",
    num_threads=8,
    weights_checkpoint_path=None,
    expected_checkpoint_size=None,
    expected_checkpoint_sha256=None,
):
    """Load and validate one online checkpoint for several query cases."""

    checkpoint_path = Path(checkpoint_path).resolve()
    weights_checkpoint_path = Path(
        weights_checkpoint_path or checkpoint_path
    ).resolve()
    for path, description in (
        (checkpoint_path, "checkpoint"),
        (weights_checkpoint_path, "weights checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description.title()} does not exist: {path}")
    if tuple(int(value) for value in block_indices) != BLOCK_INDICES:
        raise ValueError(f"The locked probe requires blocks {BLOCK_INDICES}")
    if tuple(float(value) for value in sigmas) != SIGMA_VALUES:
        raise ValueError(f"The locked probe requires sigmas {SIGMA_VALUES}")
    if tuple(float(value) for value in candidate_scales) != CANDIDATE_SCALES:
        raise ValueError(
            f"The locked probe requires candidate scales {CANDIDATE_SCALES}"
        )
    for observed, expected, name in (
        (token_probe_count, TOKEN_PROBE_COUNT, "token_probe_count"),
        (exact_batch_size, EXACT_BATCH_SIZE, "exact_batch_size"),
        (
            assignment_shuffle_count,
            ASSIGNMENT_SHUFFLE_COUNT,
            "assignment_shuffle_count",
        ),
    ):
        if int(observed) != expected:
            raise ValueError(f"The locked probe requires {name}={expected}")
    if int(num_threads) < 1:
        raise ValueError("num_threads must be positive")

    thread_config = _configure_torch_threads(int(num_threads))
    device = torch.device(device)
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    checkpoint_step = parse_checkpoint_step(checkpoint_path)
    runtime_cfg = load_runtime_cfg(config_path)
    if int(runtime_cfg.num_train_timesteps) != 1000:
        raise ValueError("The locked probe requires 1000 training timesteps")

    with _verified_checkpoint_for_loading(
        checkpoint_path,
        weights_checkpoint_path,
        expected_size=expected_checkpoint_size,
        expected_sha256=expected_checkpoint_sha256,
    ) as (weights_handle, checkpoint_identity):
        model, state_name, weights_step, load_seconds = _load_online_checkpoint_model(
            runtime_cfg,
            weights_handle,
            device,
        )
    if weights_step != checkpoint_step:
        raise ValueError("Weights checkpoint step differs from the canonical checkpoint")
    if state_name != ONLINE_CHECKPOINT_STATE:
        raise ValueError("The locked probe requires online checkpoint weights")
    _validate_probe_contract(model, runtime_cfg, BLOCK_INDICES)
    return {
        "model": model,
        "runtime_cfg": runtime_cfg,
        "device": device,
        "checkpoint_path": checkpoint_path,
        "weights_checkpoint_path": weights_checkpoint_path,
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_state": state_name,
        "checkpoint_identity": checkpoint_identity,
        "config_path": config_path,
        "thread_config": thread_config,
        "num_threads": int(num_threads),
        "model_load_seconds": float(load_seconds),
    }


def run_rcl_responsibility_query(
    loaded_probe,
    latent_path,
    label,
    support_results,
    latent_key="latent",
    seed=0,
    expected_latent_size=None,
    expected_latent_sha256=None,
):
    """Run one query with the fixed four-rank DDP-mean support gradients."""

    model = loaded_probe["model"]
    runtime_cfg = loaded_probe["runtime_cfg"]
    device = loaded_probe["device"]
    latent_path = Path(latent_path).resolve()
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent does not exist: {latent_path}")
    if not 0 <= int(label) < runtime_cfg.num_classes:
        raise ValueError("label lies outside the ImageNet class range")
    expected_support_keys = set(BLOCK_INDICES)
    if set(support_results) != expected_support_keys:
        raise ValueError("Support results do not cover every locked MoE block")
    expected_groups = list(range(SUPPORT_GROUP_COUNT))
    for item in support_results.values():
        if (
            item.get("support_group_indices") != expected_groups
            or item.get("support_gradient_aggregation") != "ddp_mean"
            or item.get("support_rank_count") != SUPPORT_GROUP_COUNT
        ):
            raise ValueError("Support results are not the locked DDP aggregate")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    with _verified_latent_for_loading(
        latent_path,
        expected_size=expected_latent_size,
        expected_sha256=expected_latent_sha256,
    ) as (latent_handle, latent_identity):
        clean_latent = _load_latent(latent_handle, latent_key, int(seed), device)
    torch.manual_seed(int(seed) + 1)
    noise = torch.randn_like(clean_latent)
    label_tensor = torch.tensor([label], device=device, dtype=torch.long)

    cells = []
    invalid_gradient_cells = 0
    probe_start = time.perf_counter()
    captures = {}
    try:
        for block_index in BLOCK_INDICES:
            moe_layer = model.blocks[block_index].mlp
            capture = RoutingProbeCapture(moe_layer)
            captures[block_index] = capture
            for sigma in SIGMA_VALUES:
                cell = _probe_cell(
                    model=model,
                    moe_layer=moe_layer,
                    capture=capture,
                    clean_latent=clean_latent,
                    noise=noise,
                    label=label_tensor,
                    sigma=sigma,
                    block_index=block_index,
                    num_train_timesteps=int(runtime_cfg.num_train_timesteps),
                    seed=int(seed),
                    support_rcl=support_results[block_index],
                )
                invalid_gradient_cells += int(
                    cell["mechanism"].get("valid") is not True
                )
                cells.append(cell)
    finally:
        for capture in captures.values():
            capture.close()
    probe_seconds = time.perf_counter() - probe_start

    result = {
        "rcl_responsibility_probe_version": PROBE_VERSION,
        "diagnostic_scope": (
            "Frozen Fresh Base online checkpoint; fixed-dispatch teacher-forced "
            "scale interventions and four-rank prototype-gradient decomposition; "
            "not a training or FID claim."
        ),
        "hypothesis": (
            "The self-assignment RCL prototype gradient often improves current "
            "dispatch geometry while pushing ProMoE's tied routed-output scale "
            "toward higher diffusion loss."
        ),
        "falsification_rule": (
            "Reject the mechanism when Fresh scale mismatch does not replicate, "
            "native RCL is not more diffusion-conflicting than count-preserving "
            "assignment shuffles, or the effect lacks block/sigma coverage."
        ),
        "checkpoint": str(loaded_probe["checkpoint_path"]),
        "weights_checkpoint": str(loaded_probe["weights_checkpoint_path"]),
        "checkpoint_step": int(loaded_probe["checkpoint_step"]),
        "checkpoint_state": loaded_probe["checkpoint_state"],
        "checkpoint_identity": loaded_probe["checkpoint_identity"],
        "config": str(loaded_probe["config_path"]),
        "model_name": runtime_cfg.model_name,
        "latent": str(latent_path),
        "latent_key": latent_key,
        "latent_identity": latent_identity,
        "label": int(label),
        "seed": int(seed),
        "device": str(device),
        "num_threads": int(loaded_probe["num_threads"]),
        "thread_config": loaded_probe["thread_config"],
        "model_load_seconds": float(loaded_probe["model_load_seconds"]),
        "probe_seconds": float(probe_seconds),
        "num_train_timesteps": int(runtime_cfg.num_train_timesteps),
        "block_indices": list(BLOCK_INDICES),
        "sigmas": list(SIGMA_VALUES),
        "candidate_scales": list(CANDIDATE_SCALES),
        "token_probe_count": TOKEN_PROBE_COUNT,
        "exact_batch_size": EXACT_BATCH_SIZE,
        "assignment_shuffle_count": ASSIGNMENT_SHUFFLE_COUNT,
        "support_group_indices": list(range(SUPPORT_GROUP_COUNT)),
        "support_gradient_aggregation": "ddp_mean",
        "support_rank_count": SUPPORT_GROUP_COUNT,
        "support_batch_size_per_rank": SUPPORT_BATCH_SIZE,
        "support_global_batch_size": SUPPORT_BATCH_SIZE * SUPPORT_GROUP_COUNT,
        "support_forward_batch_size": SUPPORT_FORWARD_BATCH_SIZE,
        "center_step_relative_frobenius": CENTER_STEP_RELATIVE_FROBENIUS,
        "center_half_step_multiplier": CENTER_HALF_STEP_MULTIPLIER,
        "gradient_scope": (
            "Direct prototype path only. The separate RCL gradient through the "
            "support hidden states is intentionally outside this diagnosis."
        ),
        "invalid_gradient_cells": int(invalid_gradient_cells),
        "cells": cells,
    }
    return result


def run_rcl_responsibility_probe(
    checkpoint_path,
    latent_path,
    label,
    support_cases,
    block_indices=BLOCK_INDICES,
    sigmas=SIGMA_VALUES,
    candidate_scales=CANDIDATE_SCALES,
    token_probe_count=TOKEN_PROBE_COUNT,
    exact_batch_size=EXACT_BATCH_SIZE,
    assignment_shuffle_count=ASSIGNMENT_SHUFFLE_COUNT,
    latent_key="latent",
    seed=0,
    device="cpu",
    num_threads=8,
    weights_checkpoint_path=None,
    expected_checkpoint_size=None,
    expected_checkpoint_sha256=None,
    expected_latent_size=None,
    expected_latent_sha256=None,
):
    """Convenience wrapper for one standalone support/query probe."""

    loaded_probe = load_rcl_responsibility_probe_model(
        checkpoint_path=checkpoint_path,
        block_indices=block_indices,
        sigmas=sigmas,
        candidate_scales=candidate_scales,
        token_probe_count=token_probe_count,
        exact_batch_size=exact_batch_size,
        assignment_shuffle_count=assignment_shuffle_count,
        device=device,
        num_threads=num_threads,
        weights_checkpoint_path=weights_checkpoint_path,
        expected_checkpoint_size=expected_checkpoint_size,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    try:
        support_cases = list(support_cases)
        rank_results = {}
        for group_index in range(SUPPORT_GROUP_COUNT):
            group_cases = [
                case
                for case in support_cases
                if int(case["group_index"]) == group_index
            ]
            rank_results[group_index] = build_rank_local_support_rcl(
                loaded_probe["model"],
                group_cases,
                group_index,
                loaded_probe["device"],
            )
        support_results = aggregate_rank_support_rcl(rank_results)
        return run_rcl_responsibility_query(
            loaded_probe=loaded_probe,
            latent_path=latent_path,
            label=label,
            support_results=support_results,
            latent_key=latent_key,
            seed=seed,
            expected_latent_size=expected_latent_size,
            expected_latent_sha256=expected_latent_sha256,
        )
    finally:
        del loaded_probe["model"]
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
