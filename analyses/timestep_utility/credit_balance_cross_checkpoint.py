"""Cross-checkpoint load and parameter-side credit validation helpers."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from .credit_balance_probe import BLOCKS, SIGMAS, gini


CROSS_CHECKPOINT_VERSION = 1
MAX_BLOCK_COUNT_CV = 0.20
MAX_BLOCK_COUNT_GINI = 0.12
MAX_BLOCK_COUNT_RATIO = 2.0
MIN_BLOCK_FRACTIONAL_REDUCTION = 0.50
MIN_PARAMETER_ACTIVE_EXPERTS = 3
MIN_PARAMETER_MEAN_SPEARMAN = 0.50
MIN_PARAMETER_BOOTSTRAP_LCB = 0.30
PARAMETER_BOOTSTRAP_RESAMPLES = 200_000
PARAMETER_BOOTSTRAP_SEED = 2026082731

_GELU_TANH_COEFFICIENT = math.sqrt(2.0 / math.pi)
_GELU_TANH_CUBIC = 0.044715


def coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("CV values must be a nonempty vector")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("CV values must be finite and nonnegative")
    mean = float(values.mean())
    return float(values.std() / mean) if mean > 0 else 0.0


def _count_ratio(values):
    values = np.asarray(values, dtype=np.float64)
    minimum = float(values.min())
    return float(values.max() / minimum) if minimum > 0 else None


def _fractional_reduction(observed, reference):
    observed = float(observed)
    reference = float(reference)
    if reference <= 0:
        return None
    return float(1.0 - observed / reference)


def _validated_cells(result, split):
    if result.get("batch_case", {}).get("split") != split:
        raise ValueError("Case split differs from the aggregate split")
    cells = result.get("cells", [])
    expected = {(block, sigma) for block in BLOCKS for sigma in SIGMAS}
    observed = {
        (int(cell["block_index"]), float(cell["sigma"])) for cell in cells
    }
    if len(cells) != len(expected) or observed != expected:
        raise ValueError("Case block/sigma cells violate the locked contract")
    return cells


def aggregate_block_load(results, split):
    """Aggregate expert counts per block without mixing expert IDs across blocks."""
    if not results:
        raise ValueError("Block-load aggregation requires case results")
    case_ids = [result.get("batch_case", {}).get("id") for result in results]
    if None in case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Block-load aggregation requires unique case IDs")

    totals = {}
    num_experts = None
    for result in results:
        for cell in _validated_cells(result, split):
            counts = np.asarray(
                cell.get("statistics", {}).get("token_count"),
                dtype=np.float64,
            )
            if (
                counts.ndim != 1
                or counts.size == 0
                or not np.isfinite(counts).all()
                or np.any(counts < 0)
                or not np.equal(counts, np.floor(counts)).all()
            ):
                raise ValueError("Token counts must be finite nonnegative integers")
            if num_experts is None:
                num_experts = int(counts.size)
                totals = {
                    block: np.zeros(num_experts, dtype=np.float64)
                    for block in BLOCKS
                }
            if counts.size != num_experts:
                raise ValueError("Expert-count vector width changed across cells")
            totals[int(cell["block_index"])] += counts

    payload = {}
    for block in BLOCKS:
        counts = totals[block]
        ratio = _count_ratio(counts)
        payload[str(block)] = {
            "token_count": counts.astype(np.int64).tolist(),
            "all_experts_active": bool(np.all(counts > 0)),
            "count_cv": coefficient_of_variation(counts),
            "count_gini": gini(counts),
            "count_ratio": ratio,
        }
    return payload


def evaluate_count_balance(lossfree_results, base_results, split):
    """Evaluate the preregistered paired, per-block count-balance precondition."""
    lossfree_ids = {
        result.get("batch_case", {}).get("id") for result in lossfree_results
    }
    base_ids = {result.get("batch_case", {}).get("id") for result in base_results}
    if None in lossfree_ids or lossfree_ids != base_ids:
        raise ValueError("Loss-Free and Base results must use identical case IDs")
    lossfree = aggregate_block_load(lossfree_results, split)
    base = aggregate_block_load(base_results, split)
    checks = {}
    passed = True
    for block in BLOCKS:
        key = str(block)
        observed = lossfree[key]
        reference = base[key]
        cv_reduction = _fractional_reduction(
            observed["count_cv"],
            reference["count_cv"],
        )
        gini_reduction = _fractional_reduction(
            observed["count_gini"],
            reference["count_gini"],
        )
        block_checks = {
            "all_experts_active": bool(observed["all_experts_active"]),
            "count_cv": observed["count_cv"] <= MAX_BLOCK_COUNT_CV,
            "count_gini": observed["count_gini"] <= MAX_BLOCK_COUNT_GINI,
            "count_ratio": (
                observed["count_ratio"] is not None
                and observed["count_ratio"] <= MAX_BLOCK_COUNT_RATIO
            ),
            "cv_fractional_reduction": (
                cv_reduction is not None
                and cv_reduction >= MIN_BLOCK_FRACTIONAL_REDUCTION
            ),
            "gini_fractional_reduction": (
                gini_reduction is not None
                and gini_reduction >= MIN_BLOCK_FRACTIONAL_REDUCTION
            ),
        }
        block_passed = all(block_checks.values())
        checks[key] = {
            "observed": observed,
            "paired_base": reference,
            "cv_fractional_reduction": cv_reduction,
            "gini_fractional_reduction": gini_reduction,
            "checks": block_checks,
            "passed": bool(block_passed),
        }
        passed = passed and block_passed
    return {
        "split": split,
        "definition": (
            "sum the complete expert count vector across images and sigmas "
            "within each block; never aggregate expert IDs across blocks"
        ),
        "requirements": {
            "maximum_each_block_count_cv": MAX_BLOCK_COUNT_CV,
            "maximum_each_block_count_gini": MAX_BLOCK_COUNT_GINI,
            "maximum_each_block_count_ratio": MAX_BLOCK_COUNT_RATIO,
            "minimum_each_block_fractional_reduction": (
                MIN_BLOCK_FRACTIONAL_REDUCTION
            ),
        },
        "blocks": checks,
        "passed": bool(passed),
    }


def evaluate_count_replay(count_results, credit_results, split):
    """Require count-only and credit passes to reproduce every route count."""
    count_by_id = {
        result.get("batch_case", {}).get("id"): result
        for result in count_results
    }
    credit_by_id = {
        result.get("batch_case", {}).get("id"): result
        for result in credit_results
    }
    if (
        None in count_by_id
        or None in credit_by_id
        or len(count_by_id) != len(count_results)
        or len(credit_by_id) != len(credit_results)
        or set(count_by_id) != set(credit_by_id)
    ):
        raise ValueError("Count replay requires identical unique case IDs")
    mismatches = []
    for case_id in sorted(count_by_id):
        count_cells = {
            (int(cell["block_index"]), float(cell["sigma"])): cell
            for cell in _validated_cells(count_by_id[case_id], split)
        }
        credit_cells = {
            (int(cell["block_index"]), float(cell["sigma"])): cell
            for cell in _validated_cells(credit_by_id[case_id], split)
        }
        for block, sigma in sorted(count_cells):
            count_vector = count_cells[(block, sigma)]["statistics"]["token_count"]
            credit_vector = credit_cells[(block, sigma)]["statistics"]["token_count"]
            if count_vector != credit_vector:
                mismatches.append({
                    "case_id": case_id,
                    "block_index": block,
                    "sigma": sigma,
                    "count_measurement": count_vector,
                    "credit_measurement": credit_vector,
                })
    return {
        "definition": (
            "exact equality of the complete expert count vector for every "
            "image/block/sigma cell across separately executed count and credit passes"
        ),
        "case_count": len(count_by_id),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": len(mismatches) == 0,
    }


def gelu_tanh_derivative(values):
    """Derivative of torch.nn.GELU(approximate='tanh')."""
    inner = _GELU_TANH_COEFFICIENT * (
        values + _GELU_TANH_CUBIC * values.pow(3)
    )
    tanh_inner = torch.tanh(inner)
    inner_derivative = _GELU_TANH_COEFFICIENT * (
        1.0 + 3.0 * _GELU_TANH_CUBIC * values.square()
    )
    return (
        0.5 * (1.0 + tanh_inner)
        + 0.5 * values * (1.0 - tanh_inner.square()) * inner_derivative
    )


def _validate_moe_mlp(expert):
    if not isinstance(getattr(expert, "up_proj", None), nn.Linear):
        raise TypeError("Expert up_proj must be nn.Linear")
    if not isinstance(getattr(expert, "down_proj", None), nn.Linear):
        raise TypeError("Expert down_proj must be nn.Linear")
    activation = getattr(expert, "act_fn", None)
    if not isinstance(activation, nn.GELU) or activation.approximate != "tanh":
        raise TypeError("Expert activation must be GELU(approximate='tanh')")
    if expert.up_proj.bias is None or expert.down_proj.bias is None:
        raise TypeError("Exact parameter credit requires both expert biases")
    if expert.up_proj.out_features != expert.down_proj.in_features:
        raise ValueError("Expert intermediate dimensions are inconsistent")
    if expert.up_proj.in_features != expert.down_proj.out_features:
        raise ValueError("Expert hidden dimensions are inconsistent")
    return {
        "class": f"{type(expert).__module__}.{type(expert).__qualname__}",
        "hidden_size": int(expert.up_proj.in_features),
        "intermediate_size": int(expert.up_proj.out_features),
        "activation": "gelu_tanh",
        "up_bias": True,
        "down_bias": True,
    }


def validate_moe_mlp(expert):
    """Validate and describe the exact parameter-credit architecture."""
    return _validate_moe_mlp(expert)


def _validate_parameter_credit_inputs(expert, inputs, output_grad):
    _validate_moe_mlp(expert)
    if inputs.ndim != 2 or output_grad.ndim != 2:
        raise ValueError("Expert inputs and output gradients must be matrices")
    if inputs.shape[0] != output_grad.shape[0]:
        raise ValueError("Expert inputs and output gradients must align by token")
    if inputs.shape[1] != expert.up_proj.in_features:
        raise ValueError("Expert input width differs from up_proj")
    if output_grad.shape[1] != expert.down_proj.out_features:
        raise ValueError("Expert output-gradient width differs from down_proj")
    if inputs.device != output_grad.device:
        raise ValueError("Expert inputs and output gradients must share a device")
    if not bool(torch.isfinite(inputs).all().item()):
        raise ValueError("Expert inputs must be finite")
    if not bool(torch.isfinite(output_grad).all().item()):
        raise ValueError("Expert output gradients must be finite")


def exact_moe_mlp_token_parameter_credit(expert, inputs, output_grad):
    """Return exact per-token empirical-Fisher trace for a routed MoeMLP."""
    _validate_parameter_credit_inputs(expert, inputs, output_grad)

    with torch.no_grad():
        pre_activation = F.linear(
            inputs,
            expert.up_proj.weight,
            expert.up_proj.bias,
        )
        activation = F.gelu(pre_activation, approximate="tanh")
        pre_activation_grad = (
            output_grad.matmul(expert.down_proj.weight)
            * gelu_tanh_derivative(pre_activation)
        )
        output_energy = output_grad.double().square().sum(dim=-1)
        activation_energy = activation.double().square().sum(dim=-1)
        pre_activation_grad_energy = (
            pre_activation_grad.double().square().sum(dim=-1)
        )
        input_energy = inputs.double().square().sum(dim=-1)
        without_bias = (
            output_energy * activation_energy
            + pre_activation_grad_energy * input_energy
        )
        with_bias = without_bias + output_energy + pre_activation_grad_energy
    return {
        "with_bias": with_bias,
        "without_bias": without_bias,
    }


def autograd_moe_mlp_token_parameter_credit(expert, inputs, output_grad):
    """Slow per-token reference used to validate the closed-form implementation."""
    _validate_parameter_credit_inputs(expert, inputs, output_grad)
    parameters = (
        expert.up_proj.weight,
        expert.up_proj.bias,
        expert.down_proj.weight,
        expert.down_proj.bias,
    )
    if not all(parameter.requires_grad for parameter in parameters):
        raise ValueError("Autograd reference requires trainable expert parameters")
    with_bias = []
    without_bias = []
    for token_index in range(inputs.shape[0]):
        output = expert(inputs[token_index:token_index + 1])
        gradients = torch.autograd.grad(
            output,
            parameters,
            grad_outputs=output_grad[token_index:token_index + 1],
        )
        squared = [gradient.double().square().sum() for gradient in gradients]
        with_bias.append(torch.stack(squared).sum())
        without_bias.append(torch.stack([squared[0], squared[2]]).sum())
    return {
        "with_bias": torch.stack(with_bias),
        "without_bias": torch.stack(without_bias),
    }


def _safe_spearman(left, right, active):
    left = np.asarray(left, dtype=np.float64)[active]
    right = np.asarray(right, dtype=np.float64)[active]
    if left.size < MIN_PARAMETER_ACTIVE_EXPERTS:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def exact_expert_parameter_credit(
    moe_layer,
    hidden_states,
    suffix_gradient,
    route_weights,
    route_indices,
):
    """Aggregate output and exact parameter credit over native top-1 routes."""
    if hidden_states.ndim != 2 or suffix_gradient.shape != hidden_states.shape:
        raise ValueError("Hidden states and suffix gradients must align matrices")
    if getattr(moe_layer, "top_k", 1) != 1:
        raise ValueError("Exact parameter credit requires native top_k == 1")
    if not bool(torch.isfinite(hidden_states).all().item()):
        raise ValueError("Hidden states must be finite")
    if not bool(torch.isfinite(suffix_gradient).all().item()):
        raise ValueError("Suffix gradients must be finite")
    route_weights = route_weights.reshape(-1)
    route_indices = route_indices.reshape(-1).to(torch.long)
    if (
        route_weights.numel() != hidden_states.shape[0]
        or route_indices.numel() != hidden_states.shape[0]
    ):
        raise ValueError("Routes must align with the token dimension")
    if not bool(torch.isfinite(route_weights).all().item()):
        raise ValueError("Native route weights must be finite")
    num_experts = int(moe_layer.num_routed_experts)
    if len(moe_layer.experts) < num_experts:
        raise ValueError("MoE layer exposes fewer experts than its routed count")
    if route_indices.numel() == 0 or torch.any(route_indices < 0):
        raise ValueError("Native routes must be nonempty and nonnegative")
    if torch.any(route_indices >= num_experts):
        raise ValueError("Parameter-credit validation accepts conditional routes only")

    counts = np.zeros(num_experts, dtype=np.int64)
    output_credit = np.zeros(num_experts, dtype=np.float64)
    parameter_credit = np.zeros(num_experts, dtype=np.float64)
    parameter_credit_without_bias = np.zeros(num_experts, dtype=np.float64)
    for expert_index in range(num_experts):
        selected = torch.where(route_indices == expert_index)[0]
        counts[expert_index] = int(selected.numel())
        if selected.numel() == 0:
            continue
        expert_inputs = hidden_states.index_select(0, selected)
        weighted_gradient = suffix_gradient.index_select(0, selected) * (
            route_weights.index_select(0, selected).unsqueeze(-1)
        )
        output_credit[expert_index] = float(
            weighted_gradient.double().square().sum().item()
        )
        token_parameter_credit = exact_moe_mlp_token_parameter_credit(
            moe_layer.experts[expert_index],
            expert_inputs,
            weighted_gradient,
        )
        parameter_credit[expert_index] = float(
            token_parameter_credit["with_bias"].sum().item()
        )
        parameter_credit_without_bias[expert_index] = float(
            token_parameter_credit["without_bias"].sum().item()
        )

    active = counts > 0
    output_rates = np.divide(
        output_credit,
        counts,
        out=np.zeros_like(output_credit),
        where=active,
    )
    parameter_rates = np.divide(
        parameter_credit,
        counts,
        out=np.zeros_like(parameter_credit),
        where=active,
    )
    parameter_rates_without_bias = np.divide(
        parameter_credit_without_bias,
        counts,
        out=np.zeros_like(parameter_credit_without_bias),
        where=active,
    )
    return {
        "token_count": counts.tolist(),
        "expert_output_credit": output_credit.tolist(),
        "expert_parameter_credit": parameter_credit.tolist(),
        "expert_parameter_credit_without_bias": (
            parameter_credit_without_bias.tolist()
        ),
        "expert_output_credit_rate": output_rates.tolist(),
        "expert_parameter_credit_rate": parameter_rates.tolist(),
        "expert_parameter_credit_rate_without_bias": (
            parameter_rates_without_bias.tolist()
        ),
        "active_experts": int(active.sum()),
        "output_parameter_spearman": _safe_spearman(
            output_credit,
            parameter_credit,
            active,
        ),
        "rate_spearman": _safe_spearman(
            output_rates,
            parameter_rates,
            active,
        ),
        "output_parameter_spearman_without_bias": _safe_spearman(
            output_credit,
            parameter_credit_without_bias,
            active,
        ),
    }


def validate_exact_parameter_credit_formula():
    """Numerically cross-check the closed form against per-token autograd."""
    from models.modules import MoeMLP

    cases = ((7, 5, 4, 6), (19, 3, 7, 9), (31, 2, 3, 5))
    rows = []
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    with torch.random.fork_rng(devices=[]):
        for seed, tokens, hidden_size, intermediate_size in cases:
            torch.manual_seed(seed)
            expert = MoeMLP(hidden_size, intermediate_size).double()
            inputs = torch.randn(tokens, hidden_size, dtype=torch.float64)
            output_grad = torch.randn(tokens, hidden_size, dtype=torch.float64)
            exact = exact_moe_mlp_token_parameter_credit(
                expert,
                inputs,
                output_grad,
            )
            reference = autograd_moe_mlp_token_parameter_credit(
                expert,
                inputs,
                output_grad,
            )
            row_errors = {}
            for key in ("with_bias", "without_bias"):
                absolute = (exact[key] - reference[key]).abs()
                relative = absolute / reference[key].abs().clamp_min(1e-12)
                key_absolute = float(absolute.max().item())
                key_relative = float(relative.max().item())
                maximum_absolute_error = max(
                    maximum_absolute_error,
                    key_absolute,
                )
                maximum_relative_error = max(
                    maximum_relative_error,
                    key_relative,
                )
                row_errors[key] = {
                    "maximum_absolute_error": key_absolute,
                    "maximum_relative_error": key_relative,
                }
            rows.append({
                "seed": seed,
                "tokens": tokens,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "errors": row_errors,
            })
    threshold = 1e-5
    return {
        "dtype": "torch.float64",
        "cases": rows,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "maximum_allowed_relative_error": threshold,
        "passed": bool(maximum_relative_error <= threshold),
    }


def _bootstrap_summary(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Parameter bootstrap values must be a finite vector")
    resamples = int(resamples)
    if resamples <= 0:
        raise ValueError("Parameter bootstrap requires positive resamples")
    generator = np.random.default_rng(int(seed))
    means = np.empty(resamples, dtype=np.float64)
    chunk_size = 10_000
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = generator.integers(
            0,
            values.size,
            size=(stop - start, values.size),
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
        "image_values": values.tolist(),
    }


def _parameter_cells(result, checkpoint_role):
    if result.get("cross_checkpoint_probe_version") != CROSS_CHECKPOINT_VERSION:
        raise ValueError("Cross-checkpoint probe version changed")
    if result.get("checkpoint_role") != checkpoint_role:
        raise ValueError("Parameter result checkpoint role changed")
    cells = _validated_cells(result, "discovery")
    for cell in cells:
        if "parameter_statistics" not in cell:
            raise ValueError("Parameter result is missing parameter statistics")
    return sorted(
        cells,
        key=lambda cell: (int(cell["block_index"]), float(cell["sigma"])),
    )


def _parameter_checkpoint_gate(
    results,
    checkpoint_role,
    resamples,
    seed,
):
    case_rows = []
    invalid_cells = []
    control_totals = {
        "route_mismatches": 0,
        "nonfinite_token_credits": 0,
        "nonfinite_parameter_credits": 0,
    }
    for result in results:
        cells = _parameter_cells(result, checkpoint_role)
        primary = []
        rate = []
        without_bias = []
        by_block = {str(block): [] for block in BLOCKS}
        by_sigma = {f"{sigma:.1f}": [] for sigma in SIGMAS}
        active_experts = []
        for cell in cells:
            statistics = cell["parameter_statistics"]
            active = int(statistics["active_experts"])
            active_experts.append(active)
            controls = cell["numerical_controls"]
            for key in control_totals:
                control_totals[key] += int(controls[key])
            value = statistics.get("output_parameter_spearman")
            rate_value = statistics.get("rate_spearman")
            no_bias_value = statistics.get(
                "output_parameter_spearman_without_bias"
            )
            valid = (
                active >= MIN_PARAMETER_ACTIVE_EXPERTS
                and value is not None
                and np.isfinite(value)
            )
            if not valid:
                invalid_cells.append({
                    "case_id": result["batch_case"]["id"],
                    "block_index": int(cell["block_index"]),
                    "sigma": float(cell["sigma"]),
                    "active_experts": active,
                })
            primary_value = float(value) if valid else -1.0
            rate_value = (
                float(rate_value)
                if rate_value is not None and np.isfinite(rate_value)
                else -1.0
            )
            no_bias_value = (
                float(no_bias_value)
                if no_bias_value is not None and np.isfinite(no_bias_value)
                else -1.0
            )
            primary.append(primary_value)
            rate.append(rate_value)
            without_bias.append(no_bias_value)
            by_block[str(int(cell["block_index"]))].append(primary_value)
            by_sigma[f"{float(cell['sigma']):.1f}"].append(primary_value)
        case_rows.append({
            "case_id": result["batch_case"]["id"],
            "primary": float(np.mean(primary)),
            "rate": float(np.mean(rate)),
            "without_bias": float(np.mean(without_bias)),
            "active_experts": float(np.mean(active_experts)),
            "by_block": {
                key: float(np.mean(values)) for key, values in by_block.items()
            },
            "by_sigma": {
                key: float(np.mean(values)) for key, values in by_sigma.items()
            },
        })

    primary_summary = _bootstrap_summary(
        [row["primary"] for row in case_rows],
        resamples,
        seed,
    )
    rate_summary = _bootstrap_summary(
        [row["rate"] for row in case_rows],
        resamples,
        seed + 1,
    )
    without_bias_summary = _bootstrap_summary(
        [row["without_bias"] for row in case_rows],
        resamples,
        seed + 2,
    )
    per_block = {
        key: _bootstrap_summary(
            [row["by_block"][key] for row in case_rows],
            resamples,
            seed + 10 + offset,
        )
        for offset, key in enumerate(str(block) for block in BLOCKS)
    }
    per_sigma = {
        key: _bootstrap_summary(
            [row["by_sigma"][key] for row in case_rows],
            resamples,
            seed + 20 + offset,
        )
        for offset, key in enumerate(f"{sigma:.1f}" for sigma in SIGMAS)
    }
    checks = {
        "all_cells_have_minimum_active_experts": len(invalid_cells) == 0,
        "route_mismatches": control_totals["route_mismatches"] == 0,
        "nonfinite_token_credits": (
            control_totals["nonfinite_token_credits"] == 0
        ),
        "nonfinite_parameter_credits": (
            control_totals["nonfinite_parameter_credits"] == 0
        ),
        "mean_spearman": (
            primary_summary["mean"] >= MIN_PARAMETER_MEAN_SPEARMAN
        ),
        "bootstrap_lcb": (
            primary_summary["one_sided_lcb95"]
            >= MIN_PARAMETER_BOOTSTRAP_LCB
        ),
    }
    return {
        "checkpoint_role": checkpoint_role,
        "case_count": len(results),
        "cell_count": len(results) * len(BLOCKS) * len(SIGMAS),
        "invalid_cells": invalid_cells,
        "numerical_control_totals": control_totals,
        "mean_active_experts": float(np.mean([
            row["active_experts"] for row in case_rows
        ])),
        "primary_output_parameter_spearman": primary_summary,
        "secondary_rate_spearman": rate_summary,
        "secondary_bias_excluded_spearman": without_bias_summary,
        "per_block_primary": per_block,
        "per_sigma_primary": per_sigma,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def aggregate_parameter_credit_validation(
    results_by_checkpoint,
    expected_case_count=16,
    resamples=PARAMETER_BOOTSTRAP_RESAMPLES,
    seed=PARAMETER_BOOTSTRAP_SEED,
):
    """Evaluate exact parameter-side validation for paired Base/Loss-Free cases."""
    if set(results_by_checkpoint) != {"base", "lossfree"}:
        raise ValueError("Parameter validation requires Base and Loss-Free results")
    expected_case_count = int(expected_case_count)
    if expected_case_count < 2:
        raise ValueError("Parameter validation requires at least two images")
    case_orders = {}
    for role, results in results_by_checkpoint.items():
        if len(results) != expected_case_count:
            raise ValueError(
                f"Expected {expected_case_count} {role} cases, got {len(results)}"
            )
        case_orders[role] = [
            result.get("batch_case", {}).get("id") for result in results
        ]
        if None in case_orders[role] or len(set(case_orders[role])) != len(results):
            raise ValueError("Parameter validation case IDs must be unique")
    if case_orders["base"] != case_orders["lossfree"]:
        raise ValueError("Parameter validation requires paired case order")

    checkpoints = {
        role: _parameter_checkpoint_gate(
            results_by_checkpoint[role],
            role,
            int(resamples),
            int(seed) + offset * 100,
        )
        for offset, role in enumerate(("base", "lossfree"))
    }
    return {
        "definition": (
            "image-bootstrap of the within-cell Spearman correlation between "
            "expert suffix-gradient output-credit sums and exact per-token "
            "empirical-Fisher parameter-credit sums"
        ),
        "case_ids": case_orders["base"],
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "requirements": {
            "minimum_active_experts_per_cell": MIN_PARAMETER_ACTIVE_EXPERTS,
            "minimum_mean_spearman_each_checkpoint": (
                MIN_PARAMETER_MEAN_SPEARMAN
            ),
            "minimum_image_bootstrap_lcb_each_checkpoint": (
                MIN_PARAMETER_BOOTSTRAP_LCB
            ),
        },
        "checkpoints": checkpoints,
        "passed": bool(all(row["passed"] for row in checkpoints.values())),
    }
