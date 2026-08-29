"""Locked mathematics for the RCL-responsibility mechanism gate."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


PROBE_VERSION = 1
BLOCK_INDICES = (1, 3, 5, 7, 9, 11)
SIGMA_VALUES = (0.8, 0.5, 0.2)
CANDIDATE_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
TOKEN_PROBE_COUNT = 16
EXACT_BATCH_SIZE = 16
ASSIGNMENT_SHUFFLE_COUNT = 16
SUPPORT_BATCH_SIZE = 64
SUPPORT_GROUP_COUNT = 4
SUPPORT_SELECTION_SALT = "promoe-rcl-responsibility-support-v1-20260829"
SUPPORT_SIGMA_POLICY = {
    "distribution": "logit_normal",
    "logit_mean": 0.0,
    "logit_std": 1.0,
    "sigmoid_scale": 1.0,
    "shift": 1.0,
    "num_train_timesteps": 1000,
    "seed_salt": "promoe-rcl-support-sigma-v1-20260829",
}
CENTER_STEP_RELATIVE_FROBENIUS = 1e-3
CENTER_HALF_STEP_MULTIPLIER = 0.5
SUPPORT_FORWARD_BATCH_SIZE = 8


def _as_finite_double(tensor, name, dimensions=None):
    tensor = torch.as_tensor(tensor, dtype=torch.float64)
    if dimensions is not None and tensor.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def _validate_router_tensors(hidden_states, centers, assignments):
    hidden_states = _as_finite_double(hidden_states, "hidden_states", 2)
    centers = _as_finite_double(centers, "centers", 2)
    assignments = torch.as_tensor(
        assignments,
        device=hidden_states.device,
        dtype=torch.long,
    )
    if assignments.ndim != 1 or assignments.shape[0] != hidden_states.shape[0]:
        raise ValueError("assignments must name one expert per token")
    if centers.shape[1] != hidden_states.shape[1]:
        raise ValueError("hidden_states and centers must share hidden size")
    if assignments.numel() < 2:
        raise ValueError("At least two token assignments are required")
    if assignments.min() < 0 or assignments.max() >= centers.shape[0]:
        raise ValueError("assignments name an invalid center")
    hidden_norms = hidden_states.norm(dim=1)
    center_norms = centers.norm(dim=1)
    if (hidden_norms == 0).any() or (center_norms == 0).any():
        raise ValueError("Cosine routing requires nonzero tokens and centers")
    return hidden_states, centers, assignments


def cosine_scores_and_selected_center_gradients(
    hidden_states,
    centers,
    assignments,
):
    """Return all cosine scores and d selected-score / d selected-center."""

    hidden_states, centers, assignments = _validate_router_tensors(
        hidden_states,
        centers,
        assignments,
    )
    hidden_unit = F.normalize(hidden_states, p=2, dim=1)
    center_unit = F.normalize(centers, p=2, dim=1)
    scores = hidden_unit @ center_unit.T
    rows = torch.arange(assignments.numel(), device=assignments.device)
    selected_scores = scores[rows, assignments]
    selected_center_unit = center_unit[assignments]
    selected_center_norm = centers[assignments].norm(dim=1)
    selected_gradients = (
        hidden_unit - selected_scores.unsqueeze(1) * selected_center_unit
    ) / selected_center_norm.unsqueeze(1)
    return scores, selected_gradients


def responsibility_center_gradient(
    hidden_states,
    centers,
    assignments,
    responsibility_slopes,
):
    """Reconstruct d diffusion-loss / d centers with fixed top-1 dispatch."""

    hidden_states, centers, assignments = _validate_router_tensors(
        hidden_states,
        centers,
        assignments,
    )
    slopes = _as_finite_double(
        responsibility_slopes,
        "responsibility_slopes",
        1,
    ).to(hidden_states.device)
    if slopes.shape != assignments.shape:
        raise ValueError("responsibility_slopes must align with assignments")
    scores, selected_gradients = cosine_scores_and_selected_center_gradients(
        hidden_states,
        centers,
        assignments,
    )
    gradient = torch.zeros_like(centers)
    gradient.index_add_(
        0,
        assignments,
        slopes.unsqueeze(1) * selected_gradients,
    )
    return gradient, scores, selected_gradients


def cosine_score_jvp(hidden_states, centers, center_direction):
    """Directional derivative of every token-center cosine score."""

    hidden_states = _as_finite_double(hidden_states, "hidden_states", 2)
    centers = _as_finite_double(centers, "centers", 2).to(hidden_states.device)
    direction = _as_finite_double(
        center_direction,
        "center_direction",
        2,
    ).to(hidden_states.device)
    if centers.shape != direction.shape:
        raise ValueError("center_direction must match centers")
    if hidden_states.shape[1] != centers.shape[1]:
        raise ValueError("hidden_states and centers must share hidden size")
    center_norm = centers.norm(dim=1, keepdim=True)
    if (center_norm == 0).any() or (hidden_states.norm(dim=1) == 0).any():
        raise ValueError("Cosine routing requires nonzero tokens and centers")
    center_unit = centers / center_norm
    tangent_direction = (
        direction
        - center_unit * (center_unit * direction).sum(dim=1, keepdim=True)
    ) / center_norm
    hidden_unit = F.normalize(hidden_states, p=2, dim=1)
    return hidden_unit @ tangent_direction.T


def routing_contrastive_loss(
    hidden_states,
    centers,
    assignments,
    temperature,
):
    """Evaluate ProMoE's RCL with fixed token assignments."""

    hidden_states, centers, assignments = _validate_router_tensors(
        hidden_states,
        centers,
        assignments,
    )
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    valid_experts = torch.unique(assignments, sorted=True)
    if valid_experts.numel() < 2:
        raise ValueError("RCL requires at least two occupied experts")
    cluster_means = torch.stack([
        hidden_states[assignments == expert].mean(dim=0)
        for expert in valid_experts
    ])
    logits = (
        F.normalize(centers[valid_experts], p=2, dim=1)
        @ F.normalize(cluster_means, p=2, dim=1).T
    ) / temperature
    labels = torch.arange(valid_experts.numel(), device=hidden_states.device)
    loss = F.cross_entropy(logits, labels)
    return loss, valid_experts


def routing_contrastive_center_gradient(
    hidden_states,
    centers,
    assignments,
    temperature,
):
    """Exact prototype-only gradient of ProMoE's self-assignment RCL."""

    hidden_states, centers, assignments = _validate_router_tensors(
        hidden_states,
        centers,
        assignments,
    )
    working_centers = centers.detach().clone().requires_grad_(True)
    loss, valid_experts = routing_contrastive_loss(
        hidden_states,
        working_centers,
        assignments,
        temperature,
    )
    gradient, = torch.autograd.grad(loss, working_centers)
    return {
        "loss": float(loss.detach().item()),
        "gradient": gradient.detach(),
        "valid_experts": valid_experts.detach(),
    }


def norm_preserving_center_step(
    centers,
    gradient,
    relative_frobenius=CENTER_STEP_RELATIVE_FROBENIUS,
    realized_dtype=None,
):
    """Take one tangent descent step and report the realized model-precision step."""

    centers = _as_finite_double(centers, "centers", 2)
    gradient = _as_finite_double(gradient, "gradient", 2).to(centers.device)
    if gradient.shape != centers.shape:
        raise ValueError("gradient must match centers")
    relative_frobenius = float(relative_frobenius)
    if not math.isfinite(relative_frobenius) or relative_frobenius <= 0:
        raise ValueError("relative_frobenius must be positive")
    center_norms = centers.norm(dim=1, keepdim=True)
    if (center_norms == 0).any():
        raise ValueError("Center updates require nonzero centers")
    center_unit = centers / center_norms
    tangent_gradient = gradient - center_unit * (
        center_unit * gradient
    ).sum(dim=1, keepdim=True)
    tangent_norm = tangent_gradient.norm()
    if tangent_norm <= 0:
        raise ValueError("The tangent center gradient must be nonzero")
    requested_step_norm = centers.norm() * relative_frobenius
    raw_centers = centers - requested_step_norm * tangent_gradient / tangent_norm
    updated_centers_double = F.normalize(raw_centers, p=2, dim=1) * center_norms
    if realized_dtype is None:
        realized_dtype = centers.dtype
    if not isinstance(realized_dtype, torch.dtype) or not realized_dtype.is_floating_point:
        raise TypeError("realized_dtype must be a floating-point torch dtype")
    updated_centers = updated_centers_double.to(dtype=realized_dtype)
    realized_centers = updated_centers.to(dtype=torch.float64)
    displacement = realized_centers - centers
    updated_norms = realized_centers.norm(dim=1, keepdim=True)
    norm_relative_error = (
        (updated_norms - center_norms).abs() / center_norms
    ).max()
    radial_gradient_fraction = (
        (gradient - tangent_gradient).norm()
        / torch.maximum(
            gradient.norm(),
            torch.tensor(1e-30, device=centers.device, dtype=torch.float64),
        )
    )
    return {
        "centers": updated_centers,
        "displacement": displacement,
        "realized_dtype": str(realized_dtype),
        "requested_relative_frobenius": relative_frobenius,
        "requested_step_frobenius": float(requested_step_norm.item()),
        "realized_step_frobenius": float(displacement.norm().item()),
        "realized_relative_frobenius": float(
            (displacement.norm() / centers.norm()).item()
        ),
        "maximum_center_norm_relative_error": float(
            norm_relative_error.item()
        ),
        "gradient_frobenius": float(gradient.norm().item()),
        "tangent_gradient_frobenius": float(tangent_norm.item()),
        "radial_gradient_fraction": float(radial_gradient_fraction.item()),
    }


def count_preserving_assignment_shuffles(assignments, count, seed):
    """Permute token labels while preserving the exact expert-count vector."""

    assignments = np.asarray(assignments, dtype=np.int64)
    count = int(count)
    if assignments.ndim != 1 or assignments.size < 2:
        raise ValueError("assignments must be a one-dimensional token vector")
    if count < 1:
        raise ValueError("count must be positive")
    if np.unique(assignments).size < 2:
        raise ValueError("Cannot shuffle an assignment with one occupied expert")
    generator = np.random.default_rng(int(seed))
    native = tuple(assignments.tolist())
    observed = {native}
    shuffles = []
    maximum_attempts = max(10_000, count * 100)
    for _ in range(maximum_attempts):
        candidate = assignments[generator.permutation(assignments.size)]
        signature = tuple(candidate.tolist())
        if signature in observed:
            continue
        if not np.array_equal(
            np.sort(candidate, kind="stable"),
            np.sort(assignments, kind="stable"),
        ):
            raise RuntimeError("Assignment shuffle changed expert counts")
        observed.add(signature)
        shuffles.append(candidate.copy())
        if len(shuffles) == count:
            return shuffles
    raise RuntimeError("Could not construct enough unique assignment shuffles")


def summarize_center_update_direction(
    hidden_states,
    centers,
    assignments,
    responsibility_slopes,
    diffusion_gradient,
    update_gradient,
):
    """Measure a unit negative-gradient center step without changing dispatch."""

    hidden_states, centers, assignments = _validate_router_tensors(
        hidden_states,
        centers,
        assignments,
    )
    slopes = _as_finite_double(
        responsibility_slopes,
        "responsibility_slopes",
        1,
    ).to(hidden_states.device)
    diffusion_gradient = _as_finite_double(
        diffusion_gradient,
        "diffusion_gradient",
        2,
    ).to(hidden_states.device)
    update_gradient = _as_finite_double(
        update_gradient,
        "update_gradient",
        2,
    ).to(hidden_states.device)
    if slopes.shape != assignments.shape:
        raise ValueError("responsibility_slopes must align with assignments")
    if diffusion_gradient.shape != centers.shape or update_gradient.shape != centers.shape:
        raise ValueError("Both center gradients must match centers")
    diffusion_norm = diffusion_gradient.norm()
    update_norm = update_gradient.norm()
    if diffusion_norm <= 0 or update_norm <= 0:
        raise ValueError("Both center gradients must be nonzero")

    unit_update = -update_gradient / update_norm
    score_delta = cosine_score_jvp(hidden_states, centers, unit_update)
    scores, _ = cosine_scores_and_selected_center_gradients(
        hidden_states,
        centers,
        assignments,
    )
    rows = torch.arange(assignments.numel(), device=assignments.device)
    selected_delta = score_delta[rows, assignments]
    masked_scores = scores.clone()
    masked_scores[rows, assignments] = -torch.inf
    runner_up = masked_scores.argmax(dim=1)
    margin_delta = selected_delta - score_delta[rows, runner_up]
    loss_contributions = slopes * selected_delta
    predicted_change = loss_contributions.sum()
    dot_product = (diffusion_gradient * update_gradient).sum()
    expected_change = -dot_product / update_norm
    identity_scale = torch.maximum(
        torch.maximum(predicted_change.abs(), expected_change.abs()),
        torch.tensor(1e-30, device=hidden_states.device, dtype=torch.float64),
    )
    identity_error = (predicted_change - expected_change).abs() / identity_scale
    gradient_cosine = dot_product / (diffusion_norm * update_norm)

    absolute_work = loss_contributions.abs()
    total_absolute_work = absolute_work.sum()
    if total_absolute_work <= 0:
        raise ValueError("The update has zero responsibility work")
    harmful_work = loss_contributions.clamp_min(0).sum()
    score_epsilon = torch.maximum(
        score_delta.abs().max() * 1e-10,
        torch.tensor(1e-14, device=hidden_states.device, dtype=torch.float64),
    )
    work_epsilon = torch.maximum(
        absolute_work.max() * 1e-10,
        torch.tensor(1e-18, device=hidden_states.device, dtype=torch.float64),
    )
    decisive = absolute_work > work_epsilon
    dispatch_improves = margin_delta > score_epsilon
    responsibility_harmed = loss_contributions > work_epsilon
    joint = decisive & dispatch_improves & responsibility_harmed
    decisive_count = int(decisive.sum().item())

    return {
        "update_gradient_norm": float(update_norm.item()),
        "diffusion_gradient_norm": float(diffusion_norm.item()),
        "gradient_cosine": float(gradient_cosine.item()),
        "gradient_conflict_score": float((-gradient_cosine).item()),
        "diffusion_loss_change_per_unit_step": float(predicted_change.item()),
        "diffusion_loss_change_relative_to_steepest": float(
            (predicted_change / diffusion_norm).item()
        ),
        "diffusion_gradient_identity_relative_error": float(
            identity_error.item()
        ),
        "responsibility_harmful_work_fraction": float(
            (harmful_work / total_absolute_work).item()
        ),
        "dispatch_margin_improve_rate": float(dispatch_improves.double().mean().item()),
        "dispatch_improve_responsibility_harm_rate": (
            float(joint.sum().item() / decisive_count)
            if decisive_count
            else None
        ),
        "dispatch_improve_harmful_work_fraction": float(
            (
                loss_contributions.masked_fill(
                    ~(dispatch_improves & responsibility_harmed),
                    0,
                ).sum()
                / total_absolute_work
            ).item()
        ),
        "selected_affinity_delta_mean": float(selected_delta.mean().item()),
        "selected_affinity_delta_std": float(selected_delta.std().item()),
        "selected_margin_delta_mean": float(margin_delta.mean().item()),
        "decisive_token_count": decisive_count,
        "token_count": int(assignments.numel()),
    }


def summarize_rcl_gradient_cell(
    hidden_states,
    centers,
    assignments,
    responsibility_slopes,
    temperature,
    shuffle_count=ASSIGNMENT_SHUFFLE_COUNT,
    shuffle_seed=0,
):
    """Compare native RCL with count-preserving assignment-shuffle controls."""

    hidden_states, centers, assignments = _validate_router_tensors(
        hidden_states,
        centers,
        assignments,
    )
    diffusion_gradient, scores, _ = responsibility_center_gradient(
        hidden_states,
        centers,
        assignments,
        responsibility_slopes,
    )
    correct_rcl = routing_contrastive_center_gradient(
        hidden_states,
        centers,
        assignments,
        temperature,
    )
    assignment_shuffles = count_preserving_assignment_shuffles(
        assignments.detach().cpu().numpy(),
        shuffle_count,
        shuffle_seed,
    )
    shuffled_rcl = [
        routing_contrastive_center_gradient(
            hidden_states,
            centers,
            torch.as_tensor(
                shuffled_assignments,
                device=hidden_states.device,
                dtype=torch.long,
            ),
            temperature,
        )
        for shuffled_assignments in assignment_shuffles
    ]
    result = summarize_external_rcl_gradient_cell(
        hidden_states=hidden_states,
        centers=centers,
        assignments=assignments,
        responsibility_slopes=responsibility_slopes,
        temperature=temperature,
        correct_support_rcl=correct_rcl,
        shuffled_support_rcl=shuffled_rcl,
    )
    result["assignment_count_mismatches"] = 0
    return result


def summarize_external_rcl_gradient_cell(
    hidden_states,
    centers,
    assignments,
    responsibility_slopes,
    temperature,
    correct_support_rcl,
    shuffled_support_rcl,
    realized_dtype=None,
):
    """Evaluate a DDP-mean support RCL gradient on an independent query."""

    hidden_states, centers, assignments = _validate_router_tensors(
        hidden_states,
        centers,
        assignments,
    )
    diffusion_gradient, scores, _ = responsibility_center_gradient(
        hidden_states,
        centers,
        assignments,
        responsibility_slopes,
    )
    query_rcl = routing_contrastive_center_gradient(
        hidden_states,
        centers,
        assignments,
        temperature,
    )
    query_rcl_gradient = query_rcl["gradient"]
    query_rcl_norm = query_rcl_gradient.norm()
    if query_rcl_norm <= 0:
        raise ValueError("The query RCL gradient must be nonzero")
    if not isinstance(correct_support_rcl, dict):
        raise ValueError("correct_support_rcl must be a mapping")
    if not shuffled_support_rcl:
        raise ValueError("At least one shuffled support RCL gradient is required")
    correct_gradient = _as_finite_double(
        correct_support_rcl.get("gradient"),
        "correct_support_gradient",
        2,
    ).to(hidden_states.device)
    if correct_gradient.shape != centers.shape:
        raise ValueError("correct support gradient must match centers")
    correct = summarize_center_update_direction(
        hidden_states,
        centers,
        assignments,
        responsibility_slopes,
        diffusion_gradient,
        correct_gradient,
    )
    diffusion_control = summarize_center_update_direction(
        hidden_states,
        centers,
        assignments,
        responsibility_slopes,
        diffusion_gradient,
        diffusion_gradient,
    )

    correct_geometry_alignment = float(
        (
            (query_rcl_gradient * correct_gradient).sum()
            / (query_rcl_norm * correct_gradient.norm())
        ).item()
    )
    correct_step = norm_preserving_center_step(
        centers,
        correct_gradient,
        realized_dtype=realized_dtype,
    )
    correct_query_loss, _ = routing_contrastive_loss(
        hidden_states,
        correct_step["centers"],
        assignments,
        temperature,
    )
    correct_query_change = correct_query_loss - query_rcl["loss"]
    correct_query_first_order = (
        query_rcl_gradient * correct_step["displacement"]
    ).sum()
    correct.update({
        "heldout_geometry_alignment": correct_geometry_alignment,
        "heldout_rcl_loss": float(query_rcl["loss"]),
        "heldout_rcl_loss_change": float(correct_query_change.item()),
        "heldout_rcl_geometry_gain": float((-correct_query_change).item()),
        "heldout_rcl_first_order_change": float(
            correct_query_first_order.item()
        ),
        "center_step": {
            key: value
            for key, value in correct_step.items()
            if key not in {"centers", "displacement"}
        },
    })
    shuffled = []
    for shuffled_rcl in shuffled_support_rcl:
        if not isinstance(shuffled_rcl, dict):
            raise ValueError("Every shuffled support RCL result must be a mapping")
        shuffled_gradient = _as_finite_double(
            shuffled_rcl.get("gradient"),
            "shuffled_support_gradient",
            2,
        ).to(hidden_states.device)
        if shuffled_gradient.shape != centers.shape or shuffled_gradient.norm() <= 0:
            raise ValueError("Every shuffled support gradient must be valid")
        metrics = summarize_center_update_direction(
            hidden_states,
            centers,
            assignments,
            responsibility_slopes,
            diffusion_gradient,
            shuffled_gradient,
        )
        metrics["heldout_geometry_alignment"] = float(
            (
                (query_rcl_gradient * shuffled_gradient).sum()
                / (query_rcl_norm * shuffled_gradient.norm())
            ).item()
        )
        shuffled_step = norm_preserving_center_step(
            centers,
            shuffled_gradient,
            realized_dtype=realized_dtype,
        )
        shuffled_query_loss, _ = routing_contrastive_loss(
            hidden_states,
            shuffled_step["centers"],
            assignments,
            temperature,
        )
        shuffled_query_change = shuffled_query_loss - query_rcl["loss"]
        metrics.update({
            "heldout_rcl_loss": float(query_rcl["loss"]),
            "heldout_rcl_loss_change": float(shuffled_query_change.item()),
            "heldout_rcl_geometry_gain": float((-shuffled_query_change).item()),
            "heldout_rcl_first_order_change": float(
                (
                    query_rcl_gradient * shuffled_step["displacement"]
                ).sum().item()
            ),
            "center_step": {
                key: value
                for key, value in shuffled_step.items()
                if key not in {"centers", "displacement"}
            },
        })
        metrics["rcl_loss"] = float(shuffled_rcl["loss"])
        shuffled.append(metrics)

    shuffle_conflicts = np.asarray(
        [item["gradient_conflict_score"] for item in shuffled],
        dtype=np.float64,
    )
    correct_conflict = correct["gradient_conflict_score"]
    shuffle_geometry = np.asarray(
        [item["heldout_geometry_alignment"] for item in shuffled],
        dtype=np.float64,
    )
    shuffle_geometry_gain = np.asarray(
        [item["heldout_rcl_geometry_gain"] for item in shuffled],
        dtype=np.float64,
    )
    strict_below = float(np.mean(shuffle_conflicts < correct_conflict))
    tied = float(np.mean(shuffle_conflicts == correct_conflict))
    return {
        "correct": {**correct, "rcl_loss": float(correct_support_rcl["loss"])},
        "diffusion_only_control": diffusion_control,
        "shuffle_summary": {
            "count": int(shuffle_conflicts.size),
            "gradient_conflict_mean": float(shuffle_conflicts.mean()),
            "gradient_conflict_std": float(shuffle_conflicts.std()),
            "gradient_conflict_min": float(shuffle_conflicts.min()),
            "gradient_conflict_max": float(shuffle_conflicts.max()),
            "correct_minus_shuffle_mean": float(
                correct_conflict - shuffle_conflicts.mean()
            ),
            "correct_shuffle_percentile": strict_below + 0.5 * tied,
            "heldout_geometry_alignment_mean": float(shuffle_geometry.mean()),
            "correct_minus_shuffle_geometry_alignment": float(
                correct_geometry_alignment - shuffle_geometry.mean()
            ),
            "heldout_rcl_geometry_gain_mean": float(
                shuffle_geometry_gain.mean()
            ),
            "correct_minus_shuffle_heldout_rcl_geometry_gain": float(
                correct["heldout_rcl_geometry_gain"]
                - shuffle_geometry_gain.mean()
            ),
        },
        "shuffled": shuffled,
        "assignment_count_mismatches": int(
            correct_support_rcl.get("assignment_count_mismatches", 0)
            + sum(
                item.get("assignment_count_mismatches", 0)
                for item in shuffled_support_rcl
            )
        ),
        "occupied_expert_count": int(
            correct_support_rcl.get("occupied_expert_count", 0)
            or len(correct_support_rcl.get("valid_experts", ()))
        ),
        "native_router_score_min": float(scores.min().item()),
        "native_router_score_max": float(scores.max().item()),
    }
