"""Locked case selection and image-level gates for learning-credit balance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

from .credit_balance_probe import BLOCKS, PROBE_VERSION, SELECTION_SALT, SIGMAS


BATCH_VERSION = 1
MANIFEST_NAME = "promoe_learning_credit_balance_gate_base200k_v1"
MODEL_NAME = "ProMoE_TC_B"
CHECKPOINT_STEP = 200000
CHECKPOINT_STATE = "ema_model_state_dict"
EXPECTED_WEIGHTS_SHA256 = (
    "efe2400374c3bf14a80590906a8000189b297dc05738ab4839ed94d8530ed848"
)
EXPECTED_WEIGHTS_SIZE = 4_808_904_390
SPLIT_COUNTS = {"plumbing": 8, "discovery": 32, "confirmatory": 64}
LOCKED_NUM_THREADS = 4
BOOTSTRAP_RESAMPLES = 200_000
BOOTSTRAP_SEED = 2026082721
PREREGISTER_PATH = (
    "/home/dev/promoe-probes/credit-balance-gate-base200k-v1-preregister.json"
)
PREREGISTER_SHA256 = (
    "392be0136b046ebaef8f02dc3f05263925d2b5585fb4f26c2d817ee08abde5b9"
)
SAFETY_REQUIREMENTS = {
    "maximum_native_output_drift": 5e-6,
    "maximum_native_relative_mse_drift": 1e-7,
    "required_route_mismatches": 0,
    "required_nonfinite_token_credits": 0,
}
DISCOVERY_REQUIREMENTS = {
    "minimum_mean_credit_rate_gini": 0.15,
    "minimum_credit_rate_gini_lcb": 0.12,
    "minimum_mean_permutation_excess_tv": 0.03,
    "minimum_permutation_excess_tv_lcb": 0.0,
    "minimum_mean_split_half_rank_spearman": 0.5,
    "minimum_positive_block_strata": 3,
    "minimum_positive_sigma_strata": 3,
}
CONFIRMATORY_REQUIREMENTS = {
    "minimum_mean_credit_rate_gini": 0.15,
    "minimum_credit_rate_gini_lcb": 0.12,
    "minimum_mean_permutation_excess_tv": 0.03,
    "minimum_permutation_excess_tv_lcb": 0.0,
    "minimum_mean_discovery_confirmatory_rank_spearman": 0.5,
    "minimum_rank_spearman_lcb": 0.3,
    "minimum_positive_block_strata": 3,
    "minimum_positive_sigma_strata": 3,
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(*parts):
    return hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def select_cases(latent_root):
    latent_root = Path(latent_root).resolve()
    class_dirs = sorted(path for path in latent_root.iterdir() if path.is_dir())
    if len(class_dirs) != 1000:
        raise ValueError(f"Expected 1000 ImageNet latent classes, found {len(class_dirs)}")
    ranked = sorted(
        (_digest(SELECTION_SALT, label), label, path)
        for label, path in enumerate(class_dirs)
    )
    required = sum(SPLIT_COUNTS.values())
    selected = ranked[:required]
    cases = []
    offset = 0
    for split, count in SPLIT_COUNTS.items():
        for _, label, class_dir in selected[offset:offset + count]:
            latent_paths = sorted(
                class_dir.glob("*.npz"),
                key=lambda path: _digest(
                    SELECTION_SALT,
                    path.relative_to(latent_root).as_posix(),
                ),
            )
            if not latent_paths:
                raise FileNotFoundError(f"No latent files found under {class_dir}")
            latent_path = latent_paths[0]
            relative = latent_path.relative_to(latent_root).as_posix()
            image_name = latent_path.name.removesuffix(".latent.npz")
            seed = int(
                _digest(SELECTION_SALT, "seed", relative)[:16],
                16,
            ) % 2_147_483_647
            cases.append({
                "split": split,
                "id": f"class{label:03d}_{image_name}",
                "label": int(label),
                "seed": int(seed),
                "synset": class_dir.name,
                "latent_relative": relative,
                "latent_sha256": sha256_file(latent_path),
            })
        offset += count
    if len({case["label"] for case in cases}) != required:
        raise RuntimeError("Credit-balance splits are not class disjoint")
    return cases


def case_protocol_view(case):
    return {
        key: case[key]
        for key in (
            "split",
            "id",
            "label",
            "seed",
            "synset",
            "latent_relative",
            "latent_sha256",
        )
    }


def _bootstrap_distribution(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite vector")
    generator = np.random.default_rng(int(seed))
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


def _bootstrap_summary(values, seed):
    values = np.asarray(values, dtype=np.float64)
    distribution = _bootstrap_distribution(values, BOOTSTRAP_RESAMPLES, seed)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
        "one_sided_lcb95": float(np.quantile(distribution, 0.05)),
        "values": values.tolist(),
    }


def _safe_spearman(left, right, active=None):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 3:
        raise ValueError("Spearman inputs must be aligned vectors")
    if active is not None:
        active = np.asarray(active, dtype=bool)
        if active.shape != left.shape:
            raise ValueError("Spearman activity mask must align with expert vectors")
        left = left[active]
        right = right[active]
    if left.size < 3:
        return -1.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else -1.0


def _stratum_key(block, sigma):
    return f"block{int(block)}_sigma{float(sigma):.1f}"


def _validate_result(result, split):
    if result.get("credit_balance_probe_version") != PROBE_VERSION:
        raise ValueError("Credit-balance probe version changed")
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


def _safety_metrics(results, split):
    safety = {
        "native_output_drift": 0.0,
        "native_relative_mse_drift": 0.0,
        "route_mismatches": 0,
        "nonfinite_token_credits": 0,
    }
    for result in results:
        for cell in _validate_result(result, split):
            controls = cell["numerical_controls"]
            safety["native_output_drift"] = max(
                safety["native_output_drift"],
                float(controls["max_abs_native_output_drift"]),
            )
            safety["native_relative_mse_drift"] = max(
                safety["native_relative_mse_drift"],
                float(controls["native_relative_mse_drift"]),
            )
            safety["route_mismatches"] += int(controls["route_mismatches"])
            safety["nonfinite_token_credits"] += int(
                controls["nonfinite_token_credits"]
            )
    checks = {
        "native_output_drift": {
            "observed": safety["native_output_drift"],
            "required": f"<={SAFETY_REQUIREMENTS['maximum_native_output_drift']}",
            "passed": safety["native_output_drift"]
            <= SAFETY_REQUIREMENTS["maximum_native_output_drift"],
        },
        "native_relative_mse_drift": {
            "observed": safety["native_relative_mse_drift"],
            "required": f"<={SAFETY_REQUIREMENTS['maximum_native_relative_mse_drift']}",
            "passed": safety["native_relative_mse_drift"]
            <= SAFETY_REQUIREMENTS["maximum_native_relative_mse_drift"],
        },
        "route_mismatches": {
            "observed": safety["route_mismatches"],
            "required": "==0",
            "passed": safety["route_mismatches"] == 0,
        },
        "nonfinite_token_credits": {
            "observed": safety["nonfinite_token_credits"],
            "required": "==0",
            "passed": safety["nonfinite_token_credits"] == 0,
        },
    }
    return safety, checks, all(row["passed"] for row in checks.values())


def _case_metrics(results, split):
    rows = []
    for result in results:
        cells = _validate_result(result, split)
        rows.append({
            "case_id": result["batch_case"]["id"],
            "credit_rate_gini": float(np.mean([
                cell["statistics"]["credit_rate_gini"] for cell in cells
            ])),
            "unit_weight_credit_rate_gini": float(np.mean([
                cell["statistics"]["unit_weight_credit_rate_gini"] for cell in cells
            ])),
            "permutation_excess_tv": float(np.mean([
                cell["statistics"]["permutation_excess_tv"] for cell in cells
            ])),
            "token_count_cv": float(np.mean([
                cell["statistics"]["token_count_cv"] for cell in cells
            ])),
            "token_count_gini": float(np.mean([
                cell["statistics"]["token_count_gini"] for cell in cells
            ])),
            "per_block_excess": {
                str(block): float(np.mean([
                    cell["statistics"]["permutation_excess_tv"]
                    for cell in cells if int(cell["block_index"]) == block
                ]))
                for block in BLOCKS
            },
            "per_sigma_excess": {
                f"{sigma:.1f}": float(np.mean([
                    cell["statistics"]["permutation_excess_tv"]
                    for cell in cells if float(cell["sigma"]) == sigma
                ]))
                for sigma in SIGMAS
            },
        })
    return rows


def _profile_arrays(results, split):
    case_ids = []
    counts = []
    credits = []
    for result in results:
        cells = sorted(
            _validate_result(result, split),
            key=lambda cell: (int(cell["block_index"]), float(cell["sigma"])),
        )
        case_ids.append(result["batch_case"]["id"])
        counts.append([
            cell["statistics"]["token_count"] for cell in cells
        ])
        credits.append([
            cell["statistics"]["expert_credit"] for cell in cells
        ])
    counts = np.asarray(counts, dtype=np.float64)
    credits = np.asarray(credits, dtype=np.float64)
    expected_shape = (len(results), len(BLOCKS) * len(SIGMAS), credits.shape[-1])
    if counts.shape != expected_shape or credits.shape != expected_shape:
        raise ValueError("Expert profile arrays do not align")
    return tuple(case_ids), counts, credits


def _rates(counts, credits):
    return np.divide(
        credits,
        counts,
        out=np.zeros_like(credits, dtype=np.float64),
        where=counts > 0,
    )


def _profile_payload(counts, credits):
    total_counts = counts.sum(axis=0)
    total_credits = credits.sum(axis=0)
    rates = _rates(total_counts, total_credits)
    payload = {}
    index = 0
    for block in BLOCKS:
        for sigma in SIGMAS:
            payload[_stratum_key(block, sigma)] = {
                "token_count": total_counts[index].tolist(),
                "expert_credit": total_credits[index].tolist(),
                "expert_credit_rate": rates[index].tolist(),
            }
            index += 1
    return payload


def _split_half_stability(case_ids, counts, credits):
    order = sorted(range(len(case_ids)), key=lambda index: _digest(case_ids[index]))
    midpoint = len(order) // 2
    halves = (order[:midpoint], order[midpoint:])
    half_counts = [counts[indices].sum(axis=0) for indices in halves]
    profiles = [
        _rates(half_count, credits[indices].sum(axis=0))
        for half_count, indices in zip(half_counts, halves)
    ]
    values = []
    by_stratum = {}
    index = 0
    for block in BLOCKS:
        for sigma in SIGMAS:
            active = (half_counts[0][index] > 0) & (half_counts[1][index] > 0)
            value = _safe_spearman(
                profiles[0][index],
                profiles[1][index],
                active,
            )
            by_stratum[_stratum_key(block, sigma)] = value
            values.append(value)
            index += 1
    return {"mean": float(np.mean(values)), "values": by_stratum}


def _holm(p_values, alpha=0.05):
    items = sorted(p_values.items(), key=lambda item: item[1])
    rejected = {}
    still_rejecting = True
    total = len(items)
    for rank, (name, p_value) in enumerate(items, start=1):
        threshold = alpha / (total - rank + 1)
        passed = still_rejecting and p_value <= threshold
        rejected[name] = {
            "p_value": float(p_value),
            "holm_threshold": float(threshold),
            "rejected": bool(passed),
        }
        if not passed:
            still_rejecting = False
    return rejected


def _positive_strata(case_rows, field, names, seed):
    summaries = {}
    p_values = {}
    for offset, name in enumerate(names):
        values = np.asarray([row[field][name] for row in case_rows], dtype=np.float64)
        summaries[name] = _bootstrap_summary(values, seed + offset)
        try:
            p_value = float(wilcoxon(values, alternative="greater").pvalue)
        except ValueError:
            p_value = 1.0
        p_values[name] = p_value
    holm = _holm(p_values)
    positive = 0
    for name in names:
        summaries[name]["holm"] = holm[name]
        passed = (
            summaries[name]["one_sided_lcb95"] > 0
            and holm[name]["rejected"]
        )
        summaries[name]["positive"] = bool(passed)
        positive += int(passed)
    return {"positive_count": positive, "strata": summaries}


def _confirmatory_rank_bootstrap(
    discovery_profiles,
    counts,
    credits,
    resamples=BOOTSTRAP_RESAMPLES,
    seed=BOOTSTRAP_SEED,
):
    discovery_counts = np.asarray([
        discovery_profiles[_stratum_key(block, sigma)]["token_count"]
        for block in BLOCKS for sigma in SIGMAS
    ], dtype=np.float64)
    discovery_rates = np.asarray([
        discovery_profiles[_stratum_key(block, sigma)]["expert_credit_rate"]
        for block in BLOCKS for sigma in SIGMAS
    ], dtype=np.float64)
    observed_counts = counts.sum(axis=0)
    observed_rates = _rates(observed_counts, credits.sum(axis=0))
    active_masks = (discovery_counts > 0) & (observed_counts > 0)
    point_values = np.asarray([
        _safe_spearman(
            discovery_rates[index],
            observed_rates[index],
            active_masks[index],
        )
        for index in range(discovery_rates.shape[0])
    ], dtype=np.float64)
    discovery_centered = []
    discovery_norm = []
    for index, active in enumerate(active_masks):
        ranks = rankdata(discovery_rates[index, active])
        centered = ranks - ranks.mean()
        discovery_centered.append(centered)
        discovery_norm.append(
            max(float(np.sqrt(np.square(centered).sum())), 1e-12)
        )

    generator = np.random.default_rng(int(seed))
    means = np.empty(int(resamples), dtype=np.float64)
    chunk_size = 512
    for start in range(0, int(resamples), chunk_size):
        stop = min(start + chunk_size, int(resamples))
        indices = generator.integers(
            0,
            counts.shape[0],
            size=(stop - start, counts.shape[0]),
        )
        sample_counts = counts[indices].sum(axis=1)
        sample_credits = credits[indices].sum(axis=1)
        sample_rates = _rates(sample_counts, sample_credits)
        correlations = []
        for stratum, active in enumerate(active_masks):
            sample_ranks = rankdata(sample_rates[:, stratum, active], axis=-1)
            sample_centered = sample_ranks - sample_ranks.mean(
                axis=-1,
                keepdims=True,
            )
            numerator = sample_centered @ discovery_centered[stratum]
            denominator = (
                np.sqrt(np.square(sample_centered).sum(axis=-1)).clip(min=1e-12)
                * discovery_norm[stratum]
            )
            correlations.append(numerator / denominator)
        means[start:stop] = np.stack(correlations, axis=1).mean(axis=1)
    return {
        "mean": float(point_values.mean()),
        "per_stratum": {
            _stratum_key(block, sigma): float(point_values[index])
            for index, (block, sigma) in enumerate(
                (block, sigma) for block in BLOCKS for sigma in SIGMAS
            )
        },
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "one_sided_lcb95": float(np.quantile(means, 0.05)),
    }


def aggregate_credit_balance(results, split, discovery_summary=None):
    expected_count = SPLIT_COUNTS.get(split)
    if expected_count is None:
        raise ValueError(f"Unknown split: {split}")
    if len(results) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {split} cases, received {len(results)}"
        )
    case_ids = [result.get("batch_case", {}).get("id") for result in results]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Aggregate contains duplicate case IDs")
    safety, safety_checks, safety_passed = _safety_metrics(results, split)
    base = {
        "split": split,
        "requirements": {
            "expected_case_count": expected_count,
            **SAFETY_REQUIREMENTS,
        },
        "case_count": len(results),
        "safety": safety,
        "safety_checks": safety_checks,
        "safety_passed": bool(safety_passed),
    }
    if split == "plumbing":
        return {
            **base,
            "efficacy_hidden": True,
            "passed": bool(safety_passed),
        }

    requirements = (
        DISCOVERY_REQUIREMENTS if split == "discovery"
        else CONFIRMATORY_REQUIREMENTS
    )
    base["requirements"].update(requirements)
    rows = _case_metrics(results, split)
    seed_offset = 0 if split == "discovery" else 1000
    gini_summary = _bootstrap_summary(
        [row["credit_rate_gini"] for row in rows],
        BOOTSTRAP_SEED + seed_offset,
    )
    unit_gini_summary = _bootstrap_summary(
        [row["unit_weight_credit_rate_gini"] for row in rows],
        BOOTSTRAP_SEED + seed_offset + 1,
    )
    excess_summary = _bootstrap_summary(
        [row["permutation_excess_tv"] for row in rows],
        BOOTSTRAP_SEED + seed_offset + 2,
    )
    load_cv_summary = _bootstrap_summary(
        [row["token_count_cv"] for row in rows],
        BOOTSTRAP_SEED + seed_offset + 3,
    )
    load_gini_summary = _bootstrap_summary(
        [row["token_count_gini"] for row in rows],
        BOOTSTRAP_SEED + seed_offset + 4,
    )
    block_strata = _positive_strata(
        rows,
        "per_block_excess",
        [str(block) for block in BLOCKS],
        BOOTSTRAP_SEED + seed_offset + 100,
    )
    sigma_strata = _positive_strata(
        rows,
        "per_sigma_excess",
        [f"{sigma:.1f}" for sigma in SIGMAS],
        BOOTSTRAP_SEED + seed_offset + 200,
    )
    profile_case_ids, counts, credits = _profile_arrays(results, split)
    profiles = _profile_payload(counts, credits)

    checks = {
        "mean_credit_rate_gini": {
            "observed": gini_summary["mean"],
            "required": f">={requirements['minimum_mean_credit_rate_gini']}",
            "passed": gini_summary["mean"]
            >= requirements["minimum_mean_credit_rate_gini"],
        },
        "credit_rate_gini_lcb": {
            "observed": gini_summary["one_sided_lcb95"],
            "required": f">={requirements['minimum_credit_rate_gini_lcb']}",
            "passed": gini_summary["one_sided_lcb95"]
            >= requirements["minimum_credit_rate_gini_lcb"],
        },
        "mean_permutation_excess_tv": {
            "observed": excess_summary["mean"],
            "required": f">={requirements['minimum_mean_permutation_excess_tv']}",
            "passed": excess_summary["mean"]
            >= requirements["minimum_mean_permutation_excess_tv"],
        },
        "permutation_excess_tv_lcb": {
            "observed": excess_summary["one_sided_lcb95"],
            "required": f">{requirements['minimum_permutation_excess_tv_lcb']}",
            "passed": excess_summary["one_sided_lcb95"]
            > requirements["minimum_permutation_excess_tv_lcb"],
        },
        "positive_block_strata": {
            "observed": block_strata["positive_count"],
            "required": f">={requirements['minimum_positive_block_strata']}",
            "passed": block_strata["positive_count"]
            >= requirements["minimum_positive_block_strata"],
        },
        "positive_sigma_strata": {
            "observed": sigma_strata["positive_count"],
            "required": f">={requirements['minimum_positive_sigma_strata']}",
            "passed": sigma_strata["positive_count"]
            >= requirements["minimum_positive_sigma_strata"],
        },
    }
    stability = None
    if split == "discovery":
        stability = _split_half_stability(profile_case_ids, counts, credits)
        checks["split_half_rank_spearman"] = {
            "observed": stability["mean"],
            "required": (
                f">={requirements['minimum_mean_split_half_rank_spearman']}"
            ),
            "passed": stability["mean"]
            >= requirements["minimum_mean_split_half_rank_spearman"],
        }
    else:
        if discovery_summary is None or not discovery_summary.get("passed"):
            raise ValueError("Passing discovery summary is required for confirmation")
        stability = _confirmatory_rank_bootstrap(
            discovery_summary["expert_profiles"],
            counts,
            credits,
        )
        checks["discovery_confirmatory_rank_spearman"] = {
            "observed": stability["mean"],
            "required": (
                f">={requirements['minimum_mean_discovery_confirmatory_rank_spearman']}"
            ),
            "passed": stability["mean"]
            >= requirements["minimum_mean_discovery_confirmatory_rank_spearman"],
        }
        checks["rank_spearman_lcb"] = {
            "observed": stability["one_sided_lcb95"],
            "required": f">={requirements['minimum_rank_spearman_lcb']}",
            "passed": stability["one_sided_lcb95"]
            >= requirements["minimum_rank_spearman_lcb"],
        }
    efficacy_passed = all(row["passed"] for row in checks.values())
    return {
        **base,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "metrics": {
            "credit_rate_gini": gini_summary,
            "unit_weight_credit_rate_gini": unit_gini_summary,
            "permutation_excess_tv": excess_summary,
            "token_count_cv": load_cv_summary,
            "token_count_gini": load_gini_summary,
            "block_strata": block_strata,
            "sigma_strata": sigma_strata,
            "rank_stability": stability,
        },
        "expert_profiles": profiles,
        "checks": checks,
        "efficacy_passed": bool(efficacy_passed),
        "passed": bool(safety_passed and efficacy_passed),
    }
