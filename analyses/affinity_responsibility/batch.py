"""Image-clustered aggregation for the RCL-responsibility mechanism gate."""

from __future__ import annotations

import math

import numpy as np

from .protocol import BLOCK_INDICES, SIGMA_VALUES


BATCH_VERSION = 1
SPLIT_COUNTS = {"plumbing": 4, "discovery": 8, "confirmatory": 24}
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEEDS = {"discovery": 291731, "confirmatory": 581953}

SAFETY_REQUIREMENTS = {
    "maximum_diffusion_gradient_identity_relative_error": 1e-10,
    "maximum_noop_global_relative_mse_drift": 1e-7,
    "maximum_noop_token_relative_mse_drift": 1e-7,
    "maximum_exact_center_noop_relative_mse_drift": 1e-7,
    "maximum_router_score_reconstruction_error": 5e-6,
    "maximum_center_norm_relative_error": 1e-6,
    "maximum_correct_half_step_first_order_relative_error": 0.35,
    "maximum_correct_full_vs_two_half_secant_relative_error": 0.25,
    "required_assignment_count_mismatches": 0,
    "required_fixed_dispatch_mismatches": 0,
    "required_diffusion_exact_descent_failures": 0,
    "required_cells_per_case": 18,
    "required_invalid_gradient_cells": 0,
}

DISCOVERY_REQUIREMENTS = {
    "maximum_mean_native_best_rate": 0.4,
    "maximum_native_best_rate_ucb95": 0.65,
    "minimum_mean_candidate_oracle_better_rate": 0.5,
    "minimum_candidate_oracle_better_rate_lcb95": 0.25,
    "minimum_mean_global_candidate_better_rate": 0.4,
    "minimum_global_candidate_better_rate_lcb95": 0.2,
    "maximum_mean_abs_affinity_best_scale_spearman": 0.5,
    "maximum_abs_affinity_best_scale_spearman_ucb95": 0.75,
    "minimum_mean_gradient_conflict_score": 0.0,
    "minimum_gradient_conflict_score_lcb95": -0.1,
    "minimum_mean_correct_minus_shuffle_conflict": 0.0,
    "minimum_correct_minus_shuffle_conflict_lcb95": -0.1,
    "minimum_mean_correct_shuffle_percentile": 0.5,
    "minimum_correct_shuffle_percentile_lcb95": 0.35,
    "minimum_mean_dispatch_improve_harmful_work": 0.05,
    "minimum_dispatch_improve_harmful_work_lcb95": 0.0,
    "minimum_mean_conflict_cell_rate": 0.4,
    "minimum_conflict_cell_rate_lcb95": 0.2,
    "minimum_mean_correct_minus_shuffle_exact_relative_mse_change": 0.0,
    "minimum_correct_minus_shuffle_exact_relative_mse_change_lcb95": -1e-7,
    "minimum_mean_correct_exact_harm_shuffle_percentile": 0.5,
    "minimum_correct_exact_harm_shuffle_percentile_lcb95": 0.35,
    "minimum_mean_correct_minus_shuffle_geometry_gain": 0.0,
    "minimum_correct_minus_shuffle_geometry_gain_lcb95": -1e-7,
    "minimum_mean_joint_geometry_better_diffusion_worse_cell_rate": 0.4,
    "minimum_joint_geometry_better_diffusion_worse_cell_rate_lcb95": 0.2,
    "minimum_positive_block_strata": 3,
    "minimum_positive_sigma_strata": 2,
}

CONFIRMATORY_REQUIREMENTS = {
    "maximum_mean_native_best_rate": 0.2,
    "maximum_native_best_rate_ucb95": 0.3,
    "minimum_mean_candidate_oracle_better_rate": 0.7,
    "minimum_candidate_oracle_better_rate_lcb95": 0.6,
    "minimum_mean_global_candidate_better_rate": 0.55,
    "minimum_global_candidate_better_rate_lcb95": 0.45,
    "maximum_mean_abs_affinity_best_scale_spearman": 0.3,
    "maximum_abs_affinity_best_scale_spearman_ucb95": 0.4,
    "minimum_mean_gradient_conflict_score": 0.05,
    "minimum_gradient_conflict_score_lcb95": 0.0,
    "minimum_mean_correct_minus_shuffle_conflict": 0.02,
    "minimum_correct_minus_shuffle_conflict_lcb95": 0.0,
    "minimum_mean_correct_shuffle_percentile": 0.6,
    "minimum_correct_shuffle_percentile_lcb95": 0.55,
    "minimum_mean_dispatch_improve_harmful_work": 0.1,
    "minimum_dispatch_improve_harmful_work_lcb95": 0.05,
    "minimum_mean_conflict_cell_rate": 0.55,
    "minimum_conflict_cell_rate_lcb95": 0.5,
    "minimum_mean_correct_minus_shuffle_exact_relative_mse_change": 0.0,
    "minimum_correct_minus_shuffle_exact_relative_mse_change_lcb95": 0.0,
    "minimum_mean_correct_exact_harm_shuffle_percentile": 0.6,
    "minimum_correct_exact_harm_shuffle_percentile_lcb95": 0.55,
    "minimum_mean_correct_minus_shuffle_geometry_gain": 0.0,
    "minimum_correct_minus_shuffle_geometry_gain_lcb95": 0.0,
    "minimum_mean_joint_geometry_better_diffusion_worse_cell_rate": 0.55,
    "minimum_joint_geometry_better_diffusion_worse_cell_rate_lcb95": 0.5,
    "minimum_positive_block_strata": 4,
    "minimum_positive_sigma_strata": 2,
}


def requirements_for_split(split):
    if split == "plumbing":
        return {
            **SAFETY_REQUIREMENTS,
            "expected_case_count": SPLIT_COUNTS[split],
        }
    if split == "discovery":
        return {
            **SAFETY_REQUIREMENTS,
            **DISCOVERY_REQUIREMENTS,
            "expected_case_count": SPLIT_COUNTS[split],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEEDS[split],
        }
    if split == "confirmatory":
        return {
            **SAFETY_REQUIREMENTS,
            **CONFIRMATORY_REQUIREMENTS,
            "expected_case_count": SPLIT_COUNTS[split],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEEDS[split],
        }
    raise ValueError(f"Unknown split: {split}")


def _finite(value, name, default=None):
    if value is None:
        if default is None:
            raise ValueError(f"Missing required metric: {name}")
        return float(default)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Metric {name} is not finite")
    return value


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Metric values must be a finite nonempty vector")
    return float(values.mean())


def _bootstrap_summary(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite image vector")
    generator = np.random.default_rng(int(seed))
    distribution = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), 10_000):
        stop = min(start + 10_000, int(resamples))
        indices = generator.integers(
            0,
            values.size,
            size=(stop - start, values.size),
        )
        distribution[start:stop] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
        "one_sided_lcb95": float(np.quantile(distribution, 0.05)),
        "one_sided_ucb95": float(np.quantile(distribution, 0.95)),
        "image_values": values.tolist(),
    }


def _case_metrics(result):
    cells = result.get("cells", [])
    expected_grid = {
        (int(block), float(sigma))
        for block in BLOCK_INDICES
        for sigma in SIGMA_VALUES
    }
    observed_grid = {
        (int(cell["block_index"]), float(cell["sigma"])) for cell in cells
    }
    if len(cells) != len(expected_grid) or observed_grid != expected_grid:
        raise ValueError("Case cells differ from the locked block/sigma grid")

    cell_metrics = []
    safety = {
        "noop_global_relative_mse_drift": 0.0,
        "noop_token_relative_mse_drift": 0.0,
        "exact_center_noop_relative_mse_drift": 0.0,
        "router_score_reconstruction_error": 0.0,
        "diffusion_gradient_identity_relative_error": 0.0,
        "center_norm_relative_error": 0.0,
        "correct_half_step_first_order_relative_error": 0.0,
        "correct_full_vs_two_half_secant_relative_error": 0.0,
        "assignment_count_mismatches": 0,
        "fixed_dispatch_mismatches": 0,
        "diffusion_exact_descent_failures": 0,
        "invalid_gradient_cells": int(result.get("invalid_gradient_cells", -1)),
        "cell_count": len(cells),
    }
    for cell in cells:
        mechanism = cell.get("mechanism", {})
        if mechanism.get("valid") is not True:
            raise ValueError("A mechanism cell is invalid")
        responsibility = cell["responsibility"]
        global_responsibility = cell["global_responsibility"]
        correct = mechanism["correct"]
        shuffle = mechanism["shuffle_summary"]
        base_mse = _finite(cell["base_mse"], "base_mse")
        exact_difference = _finite(
            shuffle["correct_minus_shuffle_exact_mse_change"],
            "correct_minus_shuffle_exact_mse_change",
        ) / base_mse
        geometry_difference = _finite(
            shuffle["correct_minus_shuffle_heldout_rcl_geometry_gain"],
            "correct_minus_shuffle_heldout_rcl_geometry_gain",
        )
        cell_metrics.append({
            "block": int(cell["block_index"]),
            "sigma": float(cell["sigma"]),
            "native_best_rate": _finite(
                responsibility["native_best_rate"], "native_best_rate"
            ),
            "candidate_oracle_better_rate": _finite(
                responsibility["candidate_oracle_better_rate"],
                "candidate_oracle_better_rate",
            ),
            "global_candidate_better_rate": _finite(
                global_responsibility["candidate_oracle_better_rate"],
                "global_candidate_better_rate",
            ),
            "abs_affinity_best_scale_spearman": abs(_finite(
                responsibility["affinity_best_candidate_scale_spearman"],
                "affinity_best_candidate_scale_spearman",
                default=1.0,
            )),
            "gradient_conflict_score": _finite(
                correct["gradient_conflict_score"], "gradient_conflict_score"
            ),
            "correct_minus_shuffle_conflict": _finite(
                shuffle["correct_minus_shuffle_mean"],
                "correct_minus_shuffle_conflict",
            ),
            "correct_shuffle_percentile": _finite(
                shuffle["correct_shuffle_percentile"],
                "correct_shuffle_percentile",
            ),
            "dispatch_improve_harmful_work": _finite(
                correct["dispatch_improve_harmful_work_fraction"],
                "dispatch_improve_harmful_work",
            ),
            "conflict_cell_rate": float(
                _finite(correct["exact_mse_change"], "exact_mse_change") > 0
            ),
            "correct_minus_shuffle_exact_relative_mse_change": exact_difference,
            "correct_exact_harm_shuffle_percentile": _finite(
                shuffle["correct_exact_harm_shuffle_percentile"],
                "correct_exact_harm_shuffle_percentile",
            ),
            "correct_minus_shuffle_geometry_gain": geometry_difference,
            "joint_geometry_better_diffusion_worse_cell_rate": float(
                exact_difference > 0 and geometry_difference > 0
            ),
        })
        controls = cell["numerical_controls"]
        safety["noop_global_relative_mse_drift"] = max(
            safety["noop_global_relative_mse_drift"],
            _finite(
                controls["noop_global_max_relative_mse_change"],
                "noop_global_max_relative_mse_change",
            ),
        )
        safety["noop_token_relative_mse_drift"] = max(
            safety["noop_token_relative_mse_drift"],
            _finite(
                controls["noop_token_max_relative_mse_change"],
                "noop_token_max_relative_mse_change",
            ),
        )
        safety["exact_center_noop_relative_mse_drift"] = max(
            safety["exact_center_noop_relative_mse_drift"],
            _finite(
                controls["exact_center_noop_relative_mse_change"],
                "exact_center_noop_relative_mse_change",
            ),
        )
        for target, source in (
            ("router_score_reconstruction_error", "router_score_reconstruction_error"),
            (
                "diffusion_gradient_identity_relative_error",
                "diffusion_gradient_identity_relative_error",
            ),
            ("center_norm_relative_error", "maximum_center_norm_relative_error"),
            (
                "correct_half_step_first_order_relative_error",
                "correct_half_step_first_order_relative_error",
            ),
            (
                "correct_full_vs_two_half_secant_relative_error",
                "correct_full_vs_two_half_secant_relative_error",
            ),
        ):
            safety[target] = max(
                safety[target],
                _finite(controls[source], source),
            )
        safety["assignment_count_mismatches"] += int(
            mechanism["assignment_count_mismatches"]
        )
        safety["fixed_dispatch_mismatches"] += int(
            controls["fixed_dispatch_mismatches"]
        )
        safety["diffusion_exact_descent_failures"] += int(
            controls["diffusion_only_exact_descent"] is not True
        )

    metric_names = tuple(
        name for name in cell_metrics[0] if name not in {"block", "sigma"}
    )
    by_block = {
        str(block): {
            name: _mean([
                item[name] for item in cell_metrics if item["block"] == block
            ])
            for name in metric_names
        }
        for block in BLOCK_INDICES
    }
    by_sigma = {
        str(float(sigma)): {
            name: _mean([
                item[name] for item in cell_metrics if item["sigma"] == sigma
            ])
            for name in metric_names
        }
        for sigma in SIGMA_VALUES
    }
    return {
        "case_id": result["batch_case"]["id"],
        **{
            name: _mean([item[name] for item in cell_metrics])
            for name in metric_names
        },
        "by_block": by_block,
        "by_sigma": by_sigma,
        "safety": safety,
    }


def _check(observed, required, passed):
    return {"observed": observed, "required": required, "passed": bool(passed)}


def _safety_gate(metrics, requirements):
    maximum_mapping = {
        "noop_global_relative_mse": (
            "noop_global_relative_mse_drift",
            "maximum_noop_global_relative_mse_drift",
        ),
        "noop_token_relative_mse": (
            "noop_token_relative_mse_drift",
            "maximum_noop_token_relative_mse_drift",
        ),
        "exact_center_noop_relative_mse": (
            "exact_center_noop_relative_mse_drift",
            "maximum_exact_center_noop_relative_mse_drift",
        ),
        "router_score_reconstruction": (
            "router_score_reconstruction_error",
            "maximum_router_score_reconstruction_error",
        ),
        "diffusion_gradient_identity": (
            "diffusion_gradient_identity_relative_error",
            "maximum_diffusion_gradient_identity_relative_error",
        ),
        "center_norm": (
            "center_norm_relative_error",
            "maximum_center_norm_relative_error",
        ),
        "correct_half_step_first_order": (
            "correct_half_step_first_order_relative_error",
            "maximum_correct_half_step_first_order_relative_error",
        ),
        "correct_full_vs_two_half_secant": (
            "correct_full_vs_two_half_secant_relative_error",
            "maximum_correct_full_vs_two_half_secant_relative_error",
        ),
    }
    checks = {}
    for check_name, (metric_name, requirement_name) in maximum_mapping.items():
        observed = max(metric["safety"][metric_name] for metric in metrics)
        required = requirements[requirement_name]
        checks[check_name] = _check(observed, f"<={required}", observed <= required)
    sum_mapping = {
        "assignment_count_mismatches": (
            "assignment_count_mismatches",
            "required_assignment_count_mismatches",
        ),
        "fixed_dispatch_mismatches": (
            "fixed_dispatch_mismatches",
            "required_fixed_dispatch_mismatches",
        ),
        "diffusion_exact_descent_failures": (
            "diffusion_exact_descent_failures",
            "required_diffusion_exact_descent_failures",
        ),
        "invalid_gradient_cells": (
            "invalid_gradient_cells",
            "required_invalid_gradient_cells",
        ),
    }
    for check_name, (metric_name, requirement_name) in sum_mapping.items():
        observed = sum(metric["safety"][metric_name] for metric in metrics)
        required = requirements[requirement_name]
        checks[check_name] = _check(observed, f"=={required}", observed == required)
    observed_cell_counts = sorted({
        metric["safety"]["cell_count"] for metric in metrics
    })
    required_cells = requirements["required_cells_per_case"]
    checks["cells_per_case"] = _check(
        observed_cell_counts,
        f"==[{required_cells}]",
        observed_cell_counts == [required_cells],
    )
    return checks, bool(all(item["passed"] for item in checks.values()))


def _strata(metrics):
    counts = {}
    details = {}
    for kind, values in (("block", BLOCK_INDICES), ("sigma", SIGMA_VALUES)):
        source = f"by_{kind}"
        kind_details = {}
        for value in values:
            key = str(value) if kind == "block" else str(float(value))
            exact = _mean([
                metric[source][key][
                    "correct_minus_shuffle_exact_relative_mse_change"
                ]
                for metric in metrics
            ])
            geometry = _mean([
                metric[source][key]["correct_minus_shuffle_geometry_gain"]
                for metric in metrics
            ])
            joint = _mean([
                metric[source][key][
                    "joint_geometry_better_diffusion_worse_cell_rate"
                ]
                for metric in metrics
            ])
            positive = exact > 0 and geometry > 0 and joint >= 0.5
            kind_details[key] = {
                "mean_exact_relative_mse_disadvantage": exact,
                "mean_heldout_geometry_advantage": geometry,
                "mean_joint_cell_rate": joint,
                "positive": bool(positive),
            }
        details[kind] = kind_details
        counts[kind] = sum(item["positive"] for item in kind_details.values())
    return counts, details


def _efficacy_gate(metrics, requirements):
    names = (
        "native_best_rate",
        "candidate_oracle_better_rate",
        "global_candidate_better_rate",
        "abs_affinity_best_scale_spearman",
        "gradient_conflict_score",
        "correct_minus_shuffle_conflict",
        "correct_shuffle_percentile",
        "dispatch_improve_harmful_work",
        "conflict_cell_rate",
        "correct_minus_shuffle_exact_relative_mse_change",
        "correct_exact_harm_shuffle_percentile",
        "correct_minus_shuffle_geometry_gain",
        "joint_geometry_better_diffusion_worse_cell_rate",
    )
    summaries = {
        name: _bootstrap_summary(
            [metric[name] for metric in metrics],
            requirements["bootstrap_resamples"],
            requirements["bootstrap_seed"] + offset,
        )
        for offset, name in enumerate(names)
    }
    upper = {
        "native_best_rate": (
            "maximum_mean_native_best_rate",
            "maximum_native_best_rate_ucb95",
        ),
        "abs_affinity_best_scale_spearman": (
            "maximum_mean_abs_affinity_best_scale_spearman",
            "maximum_abs_affinity_best_scale_spearman_ucb95",
        ),
    }
    lower = {
        "candidate_oracle_better_rate": (
            "minimum_mean_candidate_oracle_better_rate",
            "minimum_candidate_oracle_better_rate_lcb95",
        ),
        "global_candidate_better_rate": (
            "minimum_mean_global_candidate_better_rate",
            "minimum_global_candidate_better_rate_lcb95",
        ),
        "gradient_conflict_score": (
            "minimum_mean_gradient_conflict_score",
            "minimum_gradient_conflict_score_lcb95",
        ),
        "correct_minus_shuffle_conflict": (
            "minimum_mean_correct_minus_shuffle_conflict",
            "minimum_correct_minus_shuffle_conflict_lcb95",
        ),
        "correct_shuffle_percentile": (
            "minimum_mean_correct_shuffle_percentile",
            "minimum_correct_shuffle_percentile_lcb95",
        ),
        "dispatch_improve_harmful_work": (
            "minimum_mean_dispatch_improve_harmful_work",
            "minimum_dispatch_improve_harmful_work_lcb95",
        ),
        "conflict_cell_rate": (
            "minimum_mean_conflict_cell_rate",
            "minimum_conflict_cell_rate_lcb95",
        ),
        "correct_minus_shuffle_exact_relative_mse_change": (
            "minimum_mean_correct_minus_shuffle_exact_relative_mse_change",
            "minimum_correct_minus_shuffle_exact_relative_mse_change_lcb95",
        ),
        "correct_exact_harm_shuffle_percentile": (
            "minimum_mean_correct_exact_harm_shuffle_percentile",
            "minimum_correct_exact_harm_shuffle_percentile_lcb95",
        ),
        "correct_minus_shuffle_geometry_gain": (
            "minimum_mean_correct_minus_shuffle_geometry_gain",
            "minimum_correct_minus_shuffle_geometry_gain_lcb95",
        ),
        "joint_geometry_better_diffusion_worse_cell_rate": (
            "minimum_mean_joint_geometry_better_diffusion_worse_cell_rate",
            "minimum_joint_geometry_better_diffusion_worse_cell_rate_lcb95",
        ),
    }
    checks = {}
    for name, (mean_requirement, bound_requirement) in upper.items():
        checks[f"{name}_mean"] = _check(
            summaries[name]["mean"],
            f"<={requirements[mean_requirement]}",
            summaries[name]["mean"] <= requirements[mean_requirement],
        )
        checks[f"{name}_ucb95"] = _check(
            summaries[name]["one_sided_ucb95"],
            f"<={requirements[bound_requirement]}",
            summaries[name]["one_sided_ucb95"] <= requirements[bound_requirement],
        )
    for name, (mean_requirement, bound_requirement) in lower.items():
        checks[f"{name}_mean"] = _check(
            summaries[name]["mean"],
            f">={requirements[mean_requirement]}",
            summaries[name]["mean"] >= requirements[mean_requirement],
        )
        checks[f"{name}_lcb95"] = _check(
            summaries[name]["one_sided_lcb95"],
            f">={requirements[bound_requirement]}",
            summaries[name]["one_sided_lcb95"] >= requirements[bound_requirement],
        )
    strata_counts, strata_details = _strata(metrics)
    checks["block_strata"] = _check(
        strata_counts["block"],
        f">={requirements['minimum_positive_block_strata']}",
        strata_counts["block"] >= requirements["minimum_positive_block_strata"],
    )
    checks["sigma_strata"] = _check(
        strata_counts["sigma"],
        f">={requirements['minimum_positive_sigma_strata']}",
        strata_counts["sigma"] >= requirements["minimum_positive_sigma_strata"],
    )
    return {
        "passed": bool(all(item["passed"] for item in checks.values())),
        "checks": checks,
        "summaries": summaries,
        "strata_counts": strata_counts,
        "strata": strata_details,
    }


def aggregate_case_results(
    case_results,
    split,
    requirements=None,
    prerequisite_discovery_passed=None,
):
    expected = requirements_for_split(split)
    requirements = dict(requirements or expected)
    if requirements != expected:
        raise ValueError("Gate requirements differ from the locked protocol")
    if len(case_results) != requirements["expected_case_count"]:
        raise ValueError("Case count does not match the locked split")
    if split == "confirmatory" and prerequisite_discovery_passed is not True:
        raise ValueError("Confirmatory aggregation requires passing discovery")
    if split != "confirmatory" and prerequisite_discovery_passed is not None:
        raise ValueError("Only confirmatory accepts a discovery prerequisite")
    metrics = [_case_metrics(result) for result in case_results]
    if len({metric["case_id"] for metric in metrics}) != len(metrics):
        raise ValueError("Case IDs must be unique")
    safety_checks, safety_passed = _safety_gate(metrics, requirements)
    if split == "plumbing":
        return {
            "batch_version": BATCH_VERSION,
            "split": split,
            "requirements": requirements,
            "safety_checks": safety_checks,
            "safety_passed": safety_passed,
            "efficacy_statistics_withheld": True,
            "passed": safety_passed,
        }
    efficacy = _efficacy_gate(metrics, requirements)
    return {
        "batch_version": BATCH_VERSION,
        "split": split,
        "requirements": requirements,
        "safety_checks": safety_checks,
        "safety_passed": safety_passed,
        "efficacy": efficacy,
        "image_metrics": metrics,
        "passed": bool(safety_passed and efficacy["passed"]),
        "decision": (
            "Passing supports only the diagnosis that ProMoE's direct RCL "
            "prototype update improves held-out grouping more than matched "
            "shuffles while worsening tied routed responsibility. It does not "
            "establish a publishable method or authorize resumed-checkpoint training."
        ),
    }
