"""Locked protocol and aggregation for natural-input expert-utility gates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .probe import DEFAULT_BLOCK_INDICES, PROBE_VERSION


BATCH_VERSION = 3
MANIFEST_NAME = "natural_timestep_utility_gate_v1"
SELECTION_SALT = "promoe-natural-timestep-utility-v1-20260826"
MODEL_NAME = "ProMoE_TC_B"
CHECKPOINT_STEP = 100000
CHECKPOINT_STATE = "ema_model_state_dict"
EXPECTED_WEIGHTS_SHA256 = (
    "0da4061e7237924ee65aa65e969bb7c9b4365ed7b6ccc1b1874c0f8ba43e34cc"
)
SIGMAS = (0.2, 0.5, 0.8)
BLOCK_INDICES = DEFAULT_BLOCK_INDICES
NUM_TOKEN_PROBES = 8
SENSITIVITY_TOKEN_COUNT = 2
EXACT_BATCH_SIZE = 24
CAPACITY_FACTOR = 1.25
SPLIT_COUNTS = {"discovery": 8, "confirmatory": 24}
BOOTSTRAP_RESAMPLES = 200000
BOOTSTRAP_SEEDS = {"discovery": 2026082603, "confirmatory": 2026082604}
EXCLUDED_LABELS = (
    0, 36, 69, 78, 96, 100, 106, 113, 120, 131, 142, 144, 150, 156,
    159, 169, 188, 211, 224, 238, 250, 300, 301, 316, 346, 351, 381,
    384, 416, 421, 442, 445, 451, 465, 489, 500, 527, 532, 545, 588,
    589, 594, 599, 604, 615, 620, 656, 658, 683, 710, 724, 725, 735,
    738, 748, 750, 764, 774, 780, 788, 815, 833, 853, 870, 884, 887,
    902, 914, 945, 954, 956, 971, 983, 987, 992, 995, 998, 999,
)
COMMON_REQUIREMENTS = {
    "minimum_mean_native_regret_relative": 5e-5,
    "maximum_mean_native_is_oracle_rate": 0.15,
    "maximum_mean_router_utility_spearman": 0.10,
    "minimum_mean_native_capacity_improvement_relative": 1e-5,
    "require_native_capacity_ci_lower_positive": True,
    "require_every_block_native_capacity_positive": True,
    "require_every_sigma_native_capacity_positive": True,
    "minimum_mean_router_minus_utility_rank_stability": 0.10,
    "require_stage_gap_ci_lower_positive": True,
    "minimum_mean_utility_pair_inversion_rate": 0.30,
    "minimum_oracle_minus_native_flip_rate": 0.10,
    "maximum_noop_abs_mse_change": 0.0,
    "maximum_noop_abs_output_change": 0.0,
    "maximum_forced_unforced_abs_mse_change": 0.0,
    "maximum_forced_unforced_abs_output_change": 0.0,
    "maximum_joint_native_abs_mse_change": 0.0,
    "maximum_joint_native_abs_output_change": 0.0,
}


def requirements_for_split(split):
    if split not in SPLIT_COUNTS:
        raise ValueError(f"Unknown split: {split}")
    requirements = dict(COMMON_REQUIREMENTS)
    requirements.update({
        "expected_case_count": SPLIT_COUNTS[split],
        "minimum_positive_native_capacity_images": (
            5 if split == "discovery" else 16
        ),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEEDS[split],
    })
    return requirements


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_selection():
    return {
        "locked_before_discovery_results": True,
        "salt": SELECTION_SALT,
        "class_rule": (
            "Exclude every label used by earlier routing probes, sort "
            "SHA256(salt|label03|synset), use the first 8 for discovery and "
            "the next 24 for confirmation."
        ),
        "latent_rule": (
            "Sort latent basenames and select "
            "int(SHA256(salt|latent|label03|synset)[0:8],16) modulo class count."
        ),
        "seed_rule": (
            "int(SHA256(salt|latent|label03|synset)[8:16],16) modulo 2147483647."
        ),
        "excluded_labels": list(EXCLUDED_LABELS),
    }


def _case_from_rank(split, label, class_dir):
    latents = sorted(class_dir.glob("*.latent.npz"))
    if not latents:
        raise FileNotFoundError(f"No latent files found under {class_dir}")
    digest = hashlib.sha256(
        f"{SELECTION_SALT}|latent|{label:03d}|{class_dir.name}".encode()
    ).hexdigest()
    latent = latents[int(digest[:8], 16) % len(latents)]
    seed = int(digest[8:16], 16) % 2147483647
    return {
        "split": split,
        "id": f"class{label:03d}_{latent.name.removesuffix('.latent.npz')}",
        "label": label,
        "seed": seed,
        "synset": class_dir.name,
        "latent": f"{class_dir.name}/{latent.name}",
    }


def load_manifest(manifest_path, latent_root):
    manifest_path = Path(manifest_path).resolve()
    latent_root = Path(latent_root).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    if not latent_root.is_dir():
        raise NotADirectoryError(f"Latent root does not exist: {latent_root}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("name") != MANIFEST_NAME:
        raise ValueError("Manifest version or name does not match the locked gate")
    if payload.get("selection") != _canonical_selection():
        raise ValueError("Manifest selection rule is not canonical")

    class_dirs = sorted(
        path
        for path in latent_root.iterdir()
        if path.is_dir()
        and len(path.name) == 9
        and path.name.startswith("n")
        and path.name[1:].isdigit()
    )
    if len(class_dirs) != 1000:
        raise ValueError(
            f"Expected 1000 ImageNet synset directories, found {len(class_dirs)}"
        )
    excluded = set(EXCLUDED_LABELS)
    ranked = sorted(
        (
            hashlib.sha256(
                f"{SELECTION_SALT}|{label:03d}|{path.name}".encode()
            ).hexdigest(),
            label,
            path,
        )
        for label, path in enumerate(class_dirs)
        if label not in excluded
    )
    expected_cases = []
    for split, selected in (
        ("discovery", ranked[:SPLIT_COUNTS["discovery"]]),
        (
            "confirmatory",
            ranked[
                SPLIT_COUNTS["discovery"]:
                SPLIT_COUNTS["discovery"] + SPLIT_COUNTS["confirmatory"]
            ],
        ),
    ):
        expected_cases.extend(
            _case_from_rank(split, label, class_dir)
            for _, label, class_dir in selected
        )
    if payload.get("cases") != expected_cases:
        raise ValueError("Manifest cases do not match the deterministic selection")

    cases = []
    for raw_case in expected_cases:
        latent_path = (latent_root / raw_case["latent"]).resolve()
        try:
            latent_path.relative_to(latent_root)
        except ValueError as error:
            raise ValueError(f"Case {raw_case['id']} escapes the latent root") from error
        if not latent_path.is_file():
            raise FileNotFoundError(
                f"Latent for {raw_case['id']} does not exist: {latent_path}"
            )
        cases.append({
            **raw_case,
            "latent_relative": raw_case["latent"],
            "latent": str(latent_path),
            "latent_key": "latent",
            "latent_sha256": sha256_file(latent_path),
        })
    return {
        "version": 1,
        "name": MANIFEST_NAME,
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "selection": payload["selection"],
        "cases": cases,
    }


def _bootstrap_ci(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite vector with two entries")
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk_size = 10000
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = generator.integers(
            0,
            values.size,
            size=(stop - start, values.size),
        )
        means[start:stop] = values[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _case_metrics(result):
    cells = result["cells"]
    native_capacity = []
    balanced_capacity = []
    unconstrained = []
    per_block = {str(index): [] for index in BLOCK_INDICES}
    per_sigma = {str(sigma): [] for sigma in SIGMAS}
    native_load_cv = []
    balanced_load_cv = []
    capacity_count_matches = []
    safety = {
        "noop_abs_mse_change": 0.0,
        "noop_abs_output_change": 0.0,
        "forced_unforced_abs_mse_change": 0.0,
        "forced_unforced_abs_output_change": 0.0,
        "joint_native_abs_mse_change": 0.0,
        "joint_native_abs_output_change": 0.0,
    }
    sensitivity_agreement = {"candidate": [], "unit": []}
    for cell in cells:
        assignments = cell["assignments"]
        native_improvement = -assignments["native_capacity_oracle"][
            "exact_mse_change_relative"
        ]
        native_capacity.append(native_improvement)
        balanced_capacity.append(-assignments["balanced_capacity_oracle"][
            "exact_mse_change_relative"
        ])
        unconstrained.append(-assignments["unconstrained_oracle"][
            "exact_mse_change_relative"
        ])
        per_block[str(cell["block_index"])].append(native_improvement)
        per_sigma[str(cell["sigma"])].append(native_improvement)
        native_load = assignments["native"]["load"]
        capacity_load = assignments["native_capacity_oracle"]["load"]
        balanced_load = assignments["balanced_capacity_oracle"]["load"]
        native_load_cv.append(native_load["cv"])
        balanced_load_cv.append(balanced_load["cv"])
        capacity_count_matches.append(
            native_load["counts"] == capacity_load["counts"]
        )
        safety["joint_native_abs_mse_change"] = max(
            safety["joint_native_abs_mse_change"],
            abs(assignments["native"]["exact_mse_change"]),
        )
        safety["joint_native_abs_output_change"] = max(
            safety["joint_native_abs_output_change"],
            assignments["native"]["max_abs_output_change"],
        )

        controls = cell["numerical_controls"]
        safety["forced_unforced_abs_mse_change"] = max(
            safety["forced_unforced_abs_mse_change"],
            controls["max_abs_forced_unforced_mse_change"],
        )
        safety["forced_unforced_abs_output_change"] = max(
            safety["forced_unforced_abs_output_change"],
            controls["max_abs_forced_unforced_output_change"],
        )
        for mode_controls in controls["weight_modes"].values():
            safety["noop_abs_mse_change"] = max(
                safety["noop_abs_mse_change"],
                mode_controls["max_abs_noop_mse_change"],
            )
            safety["noop_abs_output_change"] = max(
                safety["noop_abs_output_change"],
                mode_controls["max_abs_noop_output_change"],
            )
        for token in cell["tokens"]:
            for mode in sensitivity_agreement:
                if mode in token["sensitivity"]:
                    sensitivity_agreement[mode].append(
                        token["oracle_expert"]
                        == token["sensitivity"][mode]["oracle_expert"]
                    )

    stage = result["stage_dynamics"]["summary"]
    return {
        "case_id": result["batch_case"]["id"],
        "native_is_oracle_rate": result["summary"]["native_is_oracle_rate"],
        "native_regret_relative": result["summary"]["mean_native_regret_relative"],
        "router_utility_spearman": result["summary"][
            "mean_router_utility_spearman"
        ],
        "native_capacity_improvement_relative": float(np.mean(native_capacity)),
        "balanced_capacity_improvement_relative": float(np.mean(balanced_capacity)),
        "unconstrained_improvement_relative": float(np.mean(unconstrained)),
        "native_capacity_positive": bool(np.mean(native_capacity) > 0),
        "per_block_native_capacity_improvement": {
            key: float(np.mean(values)) for key, values in per_block.items()
        },
        "per_sigma_native_capacity_improvement": {
            key: float(np.mean(values)) for key, values in per_sigma.items()
        },
        "mean_native_load_cv": float(np.mean(native_load_cv)),
        "mean_balanced_load_cv": float(np.mean(balanced_load_cv)),
        "native_capacity_counts_match": bool(all(capacity_count_matches)),
        "candidate_oracle_agreement": float(np.mean(sensitivity_agreement["candidate"])),
        "unit_oracle_agreement": float(np.mean(sensitivity_agreement["unit"])),
        "router_minus_utility_rank_stability": stage[
            "mean_router_minus_utility_rank_stability"
        ],
        "utility_pair_inversion_rate": stage["mean_utility_pair_inversion_rate"],
        "oracle_expert_flip_rate": stage["oracle_expert_flip_rate"],
        "native_expert_flip_rate": stage["native_expert_flip_rate"],
        "oracle_minus_native_flip_rate": (
            stage["oracle_expert_flip_rate"] - stage["native_expert_flip_rate"]
        ),
        "safety": safety,
    }


def _check(observed, required, passed):
    return {
        "observed": observed,
        "required": required,
        "passed": bool(passed),
    }


def aggregate_case_results(case_results, split, requirements=None):
    requirements = dict(requirements or requirements_for_split(split))
    expected = requirements_for_split(split)
    if requirements != expected:
        raise ValueError("Gate requirements differ from the locked split protocol")
    if len(case_results) != requirements["expected_case_count"]:
        raise ValueError("Case-result count does not match the locked split")
    metrics = [_case_metrics(result) for result in case_results]
    if len({row["case_id"] for row in metrics}) != len(metrics):
        raise ValueError("Duplicate case IDs in aggregate")

    def values(name):
        array = np.asarray([row[name] for row in metrics], dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite image-level metric: {name}")
        return array

    capacity = values("native_capacity_improvement_relative")
    stage_gap = values("router_minus_utility_rank_stability")
    capacity_ci = _bootstrap_ci(
        capacity,
        requirements["bootstrap_resamples"],
        requirements["bootstrap_seed"],
    )
    stage_ci = _bootstrap_ci(
        stage_gap,
        requirements["bootstrap_resamples"],
        requirements["bootstrap_seed"] + 1,
    )
    block_means = {
        str(block): float(np.mean([
            row["per_block_native_capacity_improvement"][str(block)]
            for row in metrics
        ]))
        for block in BLOCK_INDICES
    }
    sigma_means = {
        str(sigma): float(np.mean([
            row["per_sigma_native_capacity_improvement"][str(sigma)]
            for row in metrics
        ]))
        for sigma in SIGMAS
    }
    max_safety = {
        name: float(max(row["safety"][name] for row in metrics))
        for name in next(iter(metrics))["safety"]
    }

    safety_checks = {
        "noop_mse": _check(
            max_safety["noop_abs_mse_change"],
            f"<={requirements['maximum_noop_abs_mse_change']}",
            max_safety["noop_abs_mse_change"]
            <= requirements["maximum_noop_abs_mse_change"],
        ),
        "noop_output": _check(
            max_safety["noop_abs_output_change"],
            f"<={requirements['maximum_noop_abs_output_change']}",
            max_safety["noop_abs_output_change"]
            <= requirements["maximum_noop_abs_output_change"],
        ),
        "forced_unforced_mse": _check(
            max_safety["forced_unforced_abs_mse_change"],
            f"<={requirements['maximum_forced_unforced_abs_mse_change']}",
            max_safety["forced_unforced_abs_mse_change"]
            <= requirements["maximum_forced_unforced_abs_mse_change"],
        ),
        "forced_unforced_output": _check(
            max_safety["forced_unforced_abs_output_change"],
            f"<={requirements['maximum_forced_unforced_abs_output_change']}",
            max_safety["forced_unforced_abs_output_change"]
            <= requirements["maximum_forced_unforced_abs_output_change"],
        ),
        "native_capacity_counts": _check(
            all(row["native_capacity_counts_match"] for row in metrics),
            "true",
            all(row["native_capacity_counts_match"] for row in metrics),
        ),
        "joint_native_mse": _check(
            max_safety["joint_native_abs_mse_change"],
            f"<={requirements['maximum_joint_native_abs_mse_change']}",
            max_safety["joint_native_abs_mse_change"]
            <= requirements["maximum_joint_native_abs_mse_change"],
        ),
        "joint_native_output": _check(
            max_safety["joint_native_abs_output_change"],
            f"<={requirements['maximum_joint_native_abs_output_change']}",
            max_safety["joint_native_abs_output_change"]
            <= requirements["maximum_joint_native_abs_output_change"],
        ),
    }
    mean_regret = float(values("native_regret_relative").mean())
    mean_oracle_rate = float(values("native_is_oracle_rate").mean())
    mean_router_rho = float(values("router_utility_spearman").mean())
    mean_capacity = float(capacity.mean())
    positive_images = int(sum(row["native_capacity_positive"] for row in metrics))
    routing_checks = {
        "native_regret_magnitude": _check(
            mean_regret,
            f">={requirements['minimum_mean_native_regret_relative']}",
            mean_regret >= requirements["minimum_mean_native_regret_relative"],
        ),
        "native_oracle_rate": _check(
            mean_oracle_rate,
            f"<={requirements['maximum_mean_native_is_oracle_rate']}",
            mean_oracle_rate <= requirements["maximum_mean_native_is_oracle_rate"],
        ),
        "router_utility_spearman": _check(
            mean_router_rho,
            f"<={requirements['maximum_mean_router_utility_spearman']}",
            mean_router_rho
            <= requirements["maximum_mean_router_utility_spearman"],
        ),
        "native_capacity_improvement": _check(
            mean_capacity,
            (
                ">="
                f"{requirements['minimum_mean_native_capacity_improvement_relative']}"
            ),
            mean_capacity
            >= requirements["minimum_mean_native_capacity_improvement_relative"],
        ),
        "native_capacity_ci_lower": _check(
            capacity_ci[0],
            ">0",
            capacity_ci[0] > 0,
        ),
        "positive_images": _check(
            positive_images,
            f">={requirements['minimum_positive_native_capacity_images']}",
            positive_images
            >= requirements["minimum_positive_native_capacity_images"],
        ),
        "every_block_positive": _check(
            all(value > 0 for value in block_means.values()),
            "true",
            all(value > 0 for value in block_means.values()),
        ),
        "every_sigma_positive": _check(
            all(value > 0 for value in sigma_means.values()),
            "true",
            all(value > 0 for value in sigma_means.values()),
        ),
    }
    mean_stage_gap = float(stage_gap.mean())
    mean_inversion = float(values("utility_pair_inversion_rate").mean())
    mean_flip_gap = float(values("oracle_minus_native_flip_rate").mean())
    stage_checks = {
        "router_minus_utility_rank_stability": _check(
            mean_stage_gap,
            (
                ">="
                f"{requirements['minimum_mean_router_minus_utility_rank_stability']}"
            ),
            mean_stage_gap
            >= requirements["minimum_mean_router_minus_utility_rank_stability"],
        ),
        "stage_gap_ci_lower": _check(stage_ci[0], ">0", stage_ci[0] > 0),
        "utility_pair_inversion": _check(
            mean_inversion,
            f">={requirements['minimum_mean_utility_pair_inversion_rate']}",
            mean_inversion
            >= requirements["minimum_mean_utility_pair_inversion_rate"],
        ),
        "oracle_minus_native_flip": _check(
            mean_flip_gap,
            f">={requirements['minimum_oracle_minus_native_flip_rate']}",
            mean_flip_gap >= requirements["minimum_oracle_minus_native_flip_rate"],
        ),
    }
    safety_passed = all(check["passed"] for check in safety_checks.values())
    routing_passed = all(check["passed"] for check in routing_checks.values())
    stage_passed = all(check["passed"] for check in stage_checks.values())
    return {
        "split": split,
        "inference_unit": "image",
        "num_cases": len(metrics),
        "requirements": requirements,
        "safety_passed": safety_passed,
        "routing_accuracy_gap_passed": routing_passed,
        "stage_structure_passed": stage_passed,
        "passed": bool(safety_passed and routing_passed),
        "safety_checks": safety_checks,
        "routing_accuracy_checks": routing_checks,
        "stage_structure_checks": stage_checks,
        "native_capacity_improvement_bootstrap_ci95": capacity_ci,
        "stage_gap_bootstrap_ci95": stage_ci,
        "per_block_native_capacity_improvement": block_means,
        "per_sigma_native_capacity_improvement": sigma_means,
        "means": {
            "native_is_oracle_rate": mean_oracle_rate,
            "native_regret_relative": mean_regret,
            "router_utility_spearman": mean_router_rho,
            "native_capacity_improvement_relative": mean_capacity,
            "balanced_capacity_improvement_relative": float(values(
                "balanced_capacity_improvement_relative"
            ).mean()),
            "unconstrained_improvement_relative": float(values(
                "unconstrained_improvement_relative"
            ).mean()),
            "router_minus_utility_rank_stability": mean_stage_gap,
            "utility_pair_inversion_rate": mean_inversion,
            "oracle_minus_native_flip_rate": mean_flip_gap,
            "candidate_oracle_agreement": float(values(
                "candidate_oracle_agreement"
            ).mean()),
            "unit_oracle_agreement": float(values("unit_oracle_agreement").mean()),
            "native_load_cv": float(values("mean_native_load_cv").mean()),
            "balanced_load_cv": float(values("mean_balanced_load_cv").mean()),
        },
        "per_image": metrics,
    }
