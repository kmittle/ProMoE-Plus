"""Image-clustered aggregation for the finite-horizon routing gate."""

from __future__ import annotations

import math

import numpy as np

from analyses.denoising_regret.probe import _rankdata

from .protocol import BLOCK_INDICES, HORIZONS, START_INDICES


BATCH_VERSION = 1
SPLIT_COUNTS = {"plumbing": 4, "discovery": 8, "confirmatory": 24}
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEEDS = {"discovery": 130721, "confirmatory": 421337}
PERMUTATION_RESAMPLES = 10_000
PERMUTATION_SEEDS = {"discovery": 777031, "confirmatory": 888041}
PERMUTATION_POLICY = {
    "role": "diagnostic_only",
    "included_in_pass": False,
    "tail": "greater_association",
    "reason": (
        "Shuffling candidate labels calibrates positive label association; it "
        "does not provide a valid null test for the preregistered misalignment gate."
    ),
}

SAFETY_REQUIREMENTS = {
    "maximum_reference_duplicate_relative_mse_drift": 1e-7,
    "maximum_reference_duplicate_output_drift": 5e-6,
    "maximum_forced_native_relative_mse_drift": 1e-7,
    "maximum_forced_native_output_drift": 5e-6,
    "maximum_paired_native_relative_mse_drift": 1e-7,
    "maximum_paired_native_output_drift": 5e-6,
    # Candidate-minus-native float32 MSE differences lose several ULPs even
    # when the installed scheduler identity is correct.
    "maximum_h1_state_velocity_identity_relative_error": 1e-5,
    "required_count_mismatches": 0,
    "required_h1_rank_mismatches": 0,
}
DISCOVERY_REQUIREMENTS = {
    "maximum_mean_immediate_h8_spearman": 0.65,
    "maximum_immediate_h8_spearman_ucb95": 0.85,
    "maximum_mean_swap_preference_h8_spearman": 0.50,
    "maximum_swap_preference_h8_spearman_ucb95": 0.75,
    "minimum_mean_sign_disagreement": 0.10,
    "minimum_sign_disagreement_lcb95": 0.00,
    "minimum_mean_decisive_candidate_rate": 0.25,
    "minimum_decisive_candidate_rate_lcb95": 0.10,
    "maximum_mean_top_quartile_overlap": 0.75,
    "maximum_top_quartile_overlap_ucb95": 0.95,
    "minimum_mean_regret_fraction": 0.10,
    "minimum_regret_fraction_lcb95": 0.00,
    "minimum_mean_h8_gain_range": 1e-6,
    "minimum_h8_gain_range_lcb95": 0.0,
    "minimum_mean_best_h8_gain_relative": 5e-5,
    "minimum_best_h8_gain_relative_lcb95": 0.0,
    "minimum_mean_h8_beneficial_candidate_rate": 0.10,
    "minimum_h8_beneficial_candidate_rate_lcb95": 0.0,
    "minimum_misaligned_block_strata": 3,
    "minimum_misaligned_sigma_strata": 2,
}
CONFIRMATORY_REQUIREMENTS = {
    "maximum_mean_immediate_h8_spearman": 0.50,
    "maximum_immediate_h8_spearman_ucb95": 0.65,
    "maximum_mean_swap_preference_h8_spearman": 0.25,
    "maximum_swap_preference_h8_spearman_ucb95": 0.50,
    "minimum_mean_sign_disagreement": 0.20,
    "minimum_sign_disagreement_lcb95": 0.10,
    "minimum_mean_decisive_candidate_rate": 0.50,
    "minimum_decisive_candidate_rate_lcb95": 0.35,
    "maximum_mean_top_quartile_overlap": 0.50,
    "maximum_top_quartile_overlap_ucb95": 0.65,
    "minimum_mean_regret_fraction": 0.15,
    "minimum_regret_fraction_lcb95": 0.10,
    "minimum_mean_h8_gain_range": 5e-6,
    "minimum_h8_gain_range_lcb95": 1e-6,
    "minimum_mean_best_h8_gain_relative": 1e-4,
    "minimum_best_h8_gain_relative_lcb95": 5e-5,
    "minimum_mean_h8_beneficial_candidate_rate": 0.20,
    "minimum_h8_beneficial_candidate_rate_lcb95": 0.10,
    "minimum_misaligned_block_strata": 4,
    "minimum_misaligned_sigma_strata": 2,
}

ACTIONABLE_STRATUM_REQUIREMENTS = {
    "maximum_immediate_h8_spearman": 0.65,
    "maximum_swap_preference_h8_spearman": 0.50,
    "minimum_sign_disagreement": 0.10,
    "minimum_decisive_candidate_rate": 0.25,
    "minimum_best_h8_gain_relative": 5e-5,
    "minimum_h8_beneficial_candidate_rate": 0.10,
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
            "permutation_resamples": PERMUTATION_RESAMPLES,
            "permutation_seed": PERMUTATION_SEEDS[split],
        }
    if split == "confirmatory":
        return {
            **SAFETY_REQUIREMENTS,
            **CONFIRMATORY_REQUIREMENTS,
            "expected_case_count": SPLIT_COUNTS[split],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEEDS[split],
            "permutation_resamples": PERMUTATION_RESAMPLES,
            "permutation_seed": PERMUTATION_SEEDS[split],
        }
    raise ValueError(f"Unknown split: {split}")


def _bootstrap_distribution(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite image vector")
    generator = np.random.default_rng(seed)
    means = np.empty(int(resamples), dtype=np.float64)
    chunk_size = 10_000
    for start in range(0, int(resamples), chunk_size):
        stop = min(start + chunk_size, int(resamples))
        indices = generator.integers(
            0,
            values.size,
            size=(stop - start, values.size),
        )
        means[start:stop] = values[indices].mean(axis=1)
    return means


def _bootstrap_summary(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    distribution = _bootstrap_distribution(values, resamples, seed)
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


def _conservative_metric(value, default):
    if value is None:
        return float(default)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("A cell metric is not finite")
    return value


def _cell_h8_metrics(cell):
    horizon = cell["summary"]["per_horizon"]["8"]
    return {
        "rho": _conservative_metric(
            horizon["immediate_future_spearman"],
            1.0,
        ),
        "swap_preference_rho": _conservative_metric(
            horizon["swap_preference_future_spearman"],
            1.0,
        ),
        "sign_disagreement": _conservative_metric(
            horizon["sign_disagreement"]["rate"],
            0.0,
        ),
        "decisive_rate": float(
            horizon["sign_disagreement"]["decisive_candidates"]
            / cell["summary"]["num_candidates"]
        ),
        "top_overlap": _conservative_metric(
            horizon["top_quartile_overlap"],
            1.0,
        ),
        "regret_fraction": _conservative_metric(
            horizon["immediate_best_future_regret_fraction_of_range"],
            0.0,
        ),
        "gain_range": _conservative_metric(horizon["future_gain_range"], 0.0),
        "best_gain": _conservative_metric(
            horizon["best_future_gain_relative"],
            -1.0,
        ),
        "beneficial_rate": _conservative_metric(
            horizon["future_beneficial_rate"],
            0.0,
        ),
    }


def _relative_control_max(cell, control_name):
    control = cell["numerical_controls"][control_name]
    candidates = cell["candidates"]
    immediate_denominator = float(candidates[0]["immediate_native_mse"])
    if immediate_denominator <= 0:
        raise ValueError("Immediate native MSE must be positive")
    relative_mse = float(control["immediate_mse"]) / immediate_denominator
    output = float(control["first_prediction"])
    for horizon in HORIZONS:
        key = str(horizon)
        denominator = float(candidates[0][f"h{horizon}_native_mse"])
        if denominator <= 0:
            raise ValueError("Native rollout MSE must be positive")
        relative_mse = max(
            relative_mse,
            float(control["horizons"][key]["mse"]) / denominator,
        )
        output = max(output, float(control["horizons"][key]["state"]))
    return relative_mse, output


def _case_metrics(result):
    cells = result.get("cells", [])
    expected_cells = len(BLOCK_INDICES) * len(START_INDICES)
    if len(cells) != expected_cells:
        raise ValueError("Case does not contain the complete block/start grid")
    observed = {
        (int(cell["block_index"]), int(cell["start_index"]))
        for cell in cells
    }
    expected = {
        (block, start) for block in BLOCK_INDICES for start in START_INDICES
    }
    if observed != expected:
        raise ValueError("Case block/start cells differ from the locked grid")
    metrics = [_cell_h8_metrics(cell) for cell in cells]

    by_block = {}
    for block in BLOCK_INDICES:
        selected = [
            metric
            for cell, metric in zip(cells, metrics)
            if int(cell["block_index"]) == block
        ]
        by_block[str(block)] = {
            key: float(np.mean([item[key] for item in selected]))
            for key in selected[0]
        }
    by_sigma = {}
    for start in START_INDICES:
        selected = [
            metric
            for cell, metric in zip(cells, metrics)
            if int(cell["start_index"]) == start
        ]
        by_sigma[str(start)] = {
            key: float(np.mean([item[key] for item in selected]))
            for key in selected[0]
        }

    safety = {
        "reference_duplicate_relative_mse_drift": 0.0,
        "reference_duplicate_output_drift": 0.0,
        "forced_native_relative_mse_drift": 0.0,
        "forced_native_output_drift": 0.0,
        "paired_native_relative_mse_drift": 0.0,
        "paired_native_output_drift": 0.0,
        "h1_state_velocity_identity_relative_error": 0.0,
        "count_mismatches": 0,
        "h1_rank_mismatches": 0,
    }
    for cell in cells:
        for source, prefix in (
            ("reference_duplicate", "reference_duplicate"),
            ("forced_native_vs_unforced", "forced_native"),
            ("paired_native_vs_reference", "paired_native"),
        ):
            relative_mse, output = _relative_control_max(cell, source)
            safety[f"{prefix}_relative_mse_drift"] = max(
                safety[f"{prefix}_relative_mse_drift"],
                relative_mse,
            )
            safety[f"{prefix}_output_drift"] = max(
                safety[f"{prefix}_output_drift"],
                output,
            )
        h1_denominator = float(cell["candidates"][0]["h1_native_mse"])
        identity_error = float(
            cell["numerical_controls"][
                "max_abs_h1_state_velocity_identity_error"
            ]
        ) / h1_denominator
        safety["h1_state_velocity_identity_relative_error"] = max(
            safety["h1_state_velocity_identity_relative_error"],
            identity_error,
        )
        safety["count_mismatches"] += int(
            cell["numerical_controls"]["count_mismatches"]
        )
        h1 = cell["summary"]["per_horizon"]["1"]
        h1_rho = h1["immediate_future_spearman"]
        h1_mismatch = (
            h1_rho is not None and float(h1_rho) < 1.0 - 1e-12
        ) or float(h1["top_quartile_overlap"]) != 1.0 or not bool(
            h1["best_candidate_matches"]
        )
        safety["h1_rank_mismatches"] += int(h1_mismatch)

    return {
        "case_id": result["batch_case"]["id"],
        "rho": float(np.mean([metric["rho"] for metric in metrics])),
        "swap_preference_rho": float(np.mean([
            metric["swap_preference_rho"] for metric in metrics
        ])),
        "sign_disagreement": float(np.mean([
            metric["sign_disagreement"] for metric in metrics
        ])),
        "decisive_rate": float(np.mean([
            metric["decisive_rate"] for metric in metrics
        ])),
        "top_overlap": float(np.mean([
            metric["top_overlap"] for metric in metrics
        ])),
        "regret_fraction": float(np.mean([
            metric["regret_fraction"] for metric in metrics
        ])),
        "gain_range": float(np.mean([
            metric["gain_range"] for metric in metrics
        ])),
        "best_gain": float(np.mean([
            metric["best_gain"] for metric in metrics
        ])),
        "beneficial_rate": float(np.mean([
            metric["beneficial_rate"] for metric in metrics
        ])),
        "by_block": by_block,
        "by_sigma": by_sigma,
        "safety": safety,
    }


def _check(observed, required, passed):
    return {"observed": observed, "required": required, "passed": bool(passed)}


def _safety_gate(metrics, requirements):
    maxima = {
        name: max(metric["safety"][name] for metric in metrics)
        for name in (
            "reference_duplicate_relative_mse_drift",
            "reference_duplicate_output_drift",
            "forced_native_relative_mse_drift",
            "forced_native_output_drift",
            "paired_native_relative_mse_drift",
            "paired_native_output_drift",
            "h1_state_velocity_identity_relative_error",
        )
    }
    count_mismatches = sum(
        metric["safety"]["count_mismatches"] for metric in metrics
    )
    h1_rank_mismatches = sum(
        metric["safety"]["h1_rank_mismatches"] for metric in metrics
    )
    mapping = {
        "reference_duplicate_relative_mse": (
            "reference_duplicate_relative_mse_drift",
            "maximum_reference_duplicate_relative_mse_drift",
        ),
        "reference_duplicate_output": (
            "reference_duplicate_output_drift",
            "maximum_reference_duplicate_output_drift",
        ),
        "forced_native_relative_mse": (
            "forced_native_relative_mse_drift",
            "maximum_forced_native_relative_mse_drift",
        ),
        "forced_native_output": (
            "forced_native_output_drift",
            "maximum_forced_native_output_drift",
        ),
        "paired_native_relative_mse": (
            "paired_native_relative_mse_drift",
            "maximum_paired_native_relative_mse_drift",
        ),
        "paired_native_output": (
            "paired_native_output_drift",
            "maximum_paired_native_output_drift",
        ),
        "h1_state_velocity_identity": (
            "h1_state_velocity_identity_relative_error",
            "maximum_h1_state_velocity_identity_relative_error",
        ),
    }
    checks = {}
    for check_name, (metric_name, requirement_name) in mapping.items():
        observed = maxima[metric_name]
        required = requirements[requirement_name]
        checks[check_name] = _check(
            observed,
            f"<={required}",
            observed <= required,
        )
    checks["count_mismatches"] = _check(
        count_mismatches,
        f"=={requirements['required_count_mismatches']}",
        count_mismatches == requirements["required_count_mismatches"],
    )
    checks["h1_rank_mismatches"] = _check(
        h1_rank_mismatches,
        f"=={requirements['required_h1_rank_mismatches']}",
        h1_rank_mismatches == requirements["required_h1_rank_mismatches"],
    )
    return checks, bool(all(check["passed"] for check in checks.values()))


def _rank_rows(values):
    return np.stack([_rankdata(row) for row in values]).astype(np.float64)


def _normalize_rank_rows(values):
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.sqrt((centered ** 2).sum(axis=1, keepdims=True))
    valid = norms[:, 0] > 0
    normalized = np.zeros_like(centered)
    normalized[valid] = centered[valid] / norms[valid]
    return normalized, valid


def _candidate_label_permutation(case_results, resamples, seed):
    immediate_rows = []
    future_rows = []
    for result in case_results:
        for cell in result["cells"]:
            immediate_rows.append([
                candidate["immediate_gain_relative"]
                for candidate in cell["candidates"]
            ])
            future_rows.append([
                candidate["h8_gain_relative"]
                for candidate in cell["candidates"]
            ])
    immediate, immediate_valid = _normalize_rank_rows(
        _rank_rows(np.asarray(immediate_rows))
    )
    future, future_valid = _normalize_rank_rows(_rank_rows(np.asarray(future_rows)))
    valid = immediate_valid & future_valid
    invalid_cells = int((~valid).sum())
    immediate = immediate[valid]
    future = future[valid]
    if immediate.shape[0] == 0:
        return {
            "observed_mean_spearman": None,
            "null_mean": None,
            "null_ci95": [None, None],
            "greater_association_p": None,
            "valid_cells": 0,
            "invalid_constant_cells": invalid_cells,
            "resamples": int(resamples),
            "seed": int(seed),
            "null": "unavailable because every cell has a constant candidate ranking",
        }
    observed_cells = (immediate * future).sum(axis=1)
    observed = float(observed_cells.mean())
    generator = np.random.default_rng(seed)
    null = np.empty(int(resamples), dtype=np.float64)
    cell_count, candidate_count = immediate.shape
    chunk_size = 100
    for start in range(0, int(resamples), chunk_size):
        stop = min(start + chunk_size, int(resamples))
        order = np.argsort(
            generator.random((stop - start, cell_count, candidate_count)),
            axis=2,
        )
        shuffled = np.take_along_axis(future[None, :, :], order, axis=2)
        null[start:stop] = (shuffled * immediate[None, :, :]).sum(axis=2).mean(axis=1)
    return {
        "observed_mean_spearman": observed,
        "null_mean": float(null.mean()),
        "null_ci95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "greater_association_p": float((1 + np.sum(null >= observed)) / (1 + null.size)),
        "valid_cells": int(cell_count),
        "invalid_constant_cells": invalid_cells,
        "resamples": int(resamples),
        "seed": int(seed),
        "null": "shuffle horizon-eight candidate labels independently within each image/block/sigma cell",
    }


def _strata_counts(metrics):
    counts = {}
    details = {}
    for kind, keys in (
        ("block", tuple(str(value) for value in BLOCK_INDICES)),
        ("sigma", tuple(str(value) for value in START_INDICES)),
    ):
        source = f"by_{kind}"
        kind_details = {}
        for key in keys:
            rho = float(np.mean([
                metric[source][key]["rho"] for metric in metrics
            ]))
            swap_preference_rho = float(np.mean([
                metric[source][key]["swap_preference_rho"] for metric in metrics
            ]))
            sign = float(np.mean([
                metric[source][key]["sign_disagreement"] for metric in metrics
            ]))
            decisive = float(np.mean([
                metric[source][key]["decisive_rate"] for metric in metrics
            ]))
            best_gain = float(np.mean([
                metric[source][key]["best_gain"] for metric in metrics
            ]))
            beneficial_rate = float(np.mean([
                metric[source][key]["beneficial_rate"] for metric in metrics
            ]))
            passed = (
                rho
                <= ACTIONABLE_STRATUM_REQUIREMENTS[
                    "maximum_immediate_h8_spearman"
                ]
                and swap_preference_rho
                <= ACTIONABLE_STRATUM_REQUIREMENTS[
                    "maximum_swap_preference_h8_spearman"
                ]
                and sign
                >= ACTIONABLE_STRATUM_REQUIREMENTS[
                    "minimum_sign_disagreement"
                ]
                and decisive
                >= ACTIONABLE_STRATUM_REQUIREMENTS[
                    "minimum_decisive_candidate_rate"
                ]
                and best_gain
                >= ACTIONABLE_STRATUM_REQUIREMENTS[
                    "minimum_best_h8_gain_relative"
                ]
                and beneficial_rate
                >= ACTIONABLE_STRATUM_REQUIREMENTS[
                    "minimum_h8_beneficial_candidate_rate"
                ]
            )
            kind_details[key] = {
                "mean_rho": rho,
                "mean_swap_preference_rho": swap_preference_rho,
                "mean_sign_disagreement": sign,
                "mean_decisive_candidate_rate": decisive,
                "mean_best_h8_gain_relative": best_gain,
                "mean_h8_beneficial_candidate_rate": beneficial_rate,
                "misaligned": bool(passed),
            }
        details[kind] = kind_details
        counts[kind] = sum(item["misaligned"] for item in kind_details.values())
    return counts, details


def _efficacy_gate(metrics, requirements):
    resamples = requirements["bootstrap_resamples"]
    seed = requirements["bootstrap_seed"]
    summaries = {}
    for offset, name in enumerate((
        "rho",
        "swap_preference_rho",
        "sign_disagreement",
        "decisive_rate",
        "top_overlap",
        "regret_fraction",
        "gain_range",
        "best_gain",
        "beneficial_rate",
    )):
        summaries[name] = _bootstrap_summary(
            [metric[name] for metric in metrics],
            resamples,
            seed + offset,
        )
    strata_counts, strata = _strata_counts(metrics)
    checks = {
        "rho_mean": _check(
            summaries["rho"]["mean"],
            f"<={requirements['maximum_mean_immediate_h8_spearman']}",
            summaries["rho"]["mean"]
            <= requirements["maximum_mean_immediate_h8_spearman"],
        ),
        "rho_ucb95": _check(
            summaries["rho"]["one_sided_ucb95"],
            f"<={requirements['maximum_immediate_h8_spearman_ucb95']}",
            summaries["rho"]["one_sided_ucb95"]
            <= requirements["maximum_immediate_h8_spearman_ucb95"],
        ),
        "swap_preference_rho_mean": _check(
            summaries["swap_preference_rho"]["mean"],
            f"<={requirements['maximum_mean_swap_preference_h8_spearman']}",
            summaries["swap_preference_rho"]["mean"]
            <= requirements["maximum_mean_swap_preference_h8_spearman"],
        ),
        "swap_preference_rho_ucb95": _check(
            summaries["swap_preference_rho"]["one_sided_ucb95"],
            f"<={requirements['maximum_swap_preference_h8_spearman_ucb95']}",
            summaries["swap_preference_rho"]["one_sided_ucb95"]
            <= requirements["maximum_swap_preference_h8_spearman_ucb95"],
        ),
        "sign_disagreement_mean": _check(
            summaries["sign_disagreement"]["mean"],
            f">={requirements['minimum_mean_sign_disagreement']}",
            summaries["sign_disagreement"]["mean"]
            >= requirements["minimum_mean_sign_disagreement"],
        ),
        "sign_disagreement_lcb95": _check(
            summaries["sign_disagreement"]["one_sided_lcb95"],
            f">={requirements['minimum_sign_disagreement_lcb95']}",
            summaries["sign_disagreement"]["one_sided_lcb95"]
            >= requirements["minimum_sign_disagreement_lcb95"],
        ),
        "decisive_rate_mean": _check(
            summaries["decisive_rate"]["mean"],
            f">={requirements['minimum_mean_decisive_candidate_rate']}",
            summaries["decisive_rate"]["mean"]
            >= requirements["minimum_mean_decisive_candidate_rate"],
        ),
        "decisive_rate_lcb95": _check(
            summaries["decisive_rate"]["one_sided_lcb95"],
            f">={requirements['minimum_decisive_candidate_rate_lcb95']}",
            summaries["decisive_rate"]["one_sided_lcb95"]
            >= requirements["minimum_decisive_candidate_rate_lcb95"],
        ),
        "top_overlap_mean": _check(
            summaries["top_overlap"]["mean"],
            f"<={requirements['maximum_mean_top_quartile_overlap']}",
            summaries["top_overlap"]["mean"]
            <= requirements["maximum_mean_top_quartile_overlap"],
        ),
        "top_overlap_ucb95": _check(
            summaries["top_overlap"]["one_sided_ucb95"],
            f"<={requirements['maximum_top_quartile_overlap_ucb95']}",
            summaries["top_overlap"]["one_sided_ucb95"]
            <= requirements["maximum_top_quartile_overlap_ucb95"],
        ),
        "regret_fraction_mean": _check(
            summaries["regret_fraction"]["mean"],
            f">={requirements['minimum_mean_regret_fraction']}",
            summaries["regret_fraction"]["mean"]
            >= requirements["minimum_mean_regret_fraction"],
        ),
        "regret_fraction_lcb95": _check(
            summaries["regret_fraction"]["one_sided_lcb95"],
            f">={requirements['minimum_regret_fraction_lcb95']}",
            summaries["regret_fraction"]["one_sided_lcb95"]
            >= requirements["minimum_regret_fraction_lcb95"],
        ),
        "gain_range_mean": _check(
            summaries["gain_range"]["mean"],
            f">={requirements['minimum_mean_h8_gain_range']}",
            summaries["gain_range"]["mean"]
            >= requirements["minimum_mean_h8_gain_range"],
        ),
        "gain_range_lcb95": _check(
            summaries["gain_range"]["one_sided_lcb95"],
            f">={requirements['minimum_h8_gain_range_lcb95']}",
            summaries["gain_range"]["one_sided_lcb95"]
            >= requirements["minimum_h8_gain_range_lcb95"],
        ),
        "best_gain_mean": _check(
            summaries["best_gain"]["mean"],
            f">={requirements['minimum_mean_best_h8_gain_relative']}",
            summaries["best_gain"]["mean"]
            >= requirements["minimum_mean_best_h8_gain_relative"],
        ),
        "best_gain_lcb95": _check(
            summaries["best_gain"]["one_sided_lcb95"],
            f">={requirements['minimum_best_h8_gain_relative_lcb95']}",
            summaries["best_gain"]["one_sided_lcb95"]
            >= requirements["minimum_best_h8_gain_relative_lcb95"],
        ),
        "beneficial_rate_mean": _check(
            summaries["beneficial_rate"]["mean"],
            f">={requirements['minimum_mean_h8_beneficial_candidate_rate']}",
            summaries["beneficial_rate"]["mean"]
            >= requirements["minimum_mean_h8_beneficial_candidate_rate"],
        ),
        "beneficial_rate_lcb95": _check(
            summaries["beneficial_rate"]["one_sided_lcb95"],
            f">={requirements['minimum_h8_beneficial_candidate_rate_lcb95']}",
            summaries["beneficial_rate"]["one_sided_lcb95"]
            >= requirements["minimum_h8_beneficial_candidate_rate_lcb95"],
        ),
        "block_strata": _check(
            strata_counts["block"],
            f">={requirements['minimum_misaligned_block_strata']}",
            strata_counts["block"]
            >= requirements["minimum_misaligned_block_strata"],
        ),
        "sigma_strata": _check(
            strata_counts["sigma"],
            f">={requirements['minimum_misaligned_sigma_strata']}",
            strata_counts["sigma"]
            >= requirements["minimum_misaligned_sigma_strata"],
        ),
    }
    return {
        "passed": bool(all(check["passed"] for check in checks.values())),
        "checks": checks,
        "summaries": summaries,
        "strata_counts": strata_counts,
        "actionable_stratum_requirements": ACTIONABLE_STRATUM_REQUIREMENTS,
        "strata": strata,
    }


def aggregate_case_results(
    case_results,
    split,
    requirements=None,
    prerequisite_discovery_passed=None,
):
    expected_requirements = requirements_for_split(split)
    requirements = dict(requirements or expected_requirements)
    if requirements != expected_requirements:
        raise ValueError("Gate requirements differ from the locked protocol")
    if len(case_results) != requirements["expected_case_count"]:
        raise ValueError("Case count does not match the locked split")
    if split == "confirmatory" and prerequisite_discovery_passed is not True:
        raise ValueError("Confirmatory aggregation requires a passing discovery gate")
    if split != "confirmatory" and prerequisite_discovery_passed is not None:
        raise ValueError("Only confirmatory aggregation accepts a discovery prerequisite")
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
    permutation = _candidate_label_permutation(
        case_results,
        requirements["permutation_resamples"],
        requirements["permutation_seed"],
    )
    passed = bool(safety_passed and efficacy["passed"])
    return {
        "batch_version": BATCH_VERSION,
        "split": split,
        "requirements": requirements,
        "safety_checks": safety_checks,
        "safety_passed": safety_passed,
        "efficacy": efficacy,
        "candidate_label_permutation": {
            **permutation,
            **PERMUTATION_POLICY,
        },
        "pass_components": ["safety", "efficacy"],
        "image_metrics": metrics,
        "passed": passed,
        "decision": (
            "A passing confirmation supports only the diagnosis that the native "
            "router leaves actionable horizon-eight headroom and that immediate, "
            "router swap-preference, and finite-horizon quota-preserving assignment "
            "utility "
            "are systematically misaligned. It does not authorize a long training "
            "run until novelty review and a separate training-method protocol pass."
        ),
    }
