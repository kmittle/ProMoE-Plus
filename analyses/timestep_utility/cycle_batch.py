"""Locked manifest and image-level gate for count-preserving routing cycles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .cycle_probe import ARM_NAMES, COUNT_PRESERVING_ARMS, PROBE_VERSION


BATCH_VERSION = 2
MANIFEST_NAME = "count_preserving_cycle_gate_v1"
SELECTION_SALT = "promoe-count-preserving-cycle-gate-v1-20260826"
MODEL_NAME = "ProMoE_TC_B"
CHECKPOINT_STEP = 200000
CHECKPOINT_STATE = "ema_model_state_dict"
EXPECTED_WEIGHTS_SHA256 = (
    "efe2400374c3bf14a80590906a8000189b297dc05738ab4839ed94d8530ed848"
)
EXPECTED_WEIGHTS_SIZE = 4_808_904_390
SIGMAS = (0.2, 0.5, 0.8)
BLOCKS_BY_SPLIT = {
    "plumbing": (5,),
    "discovery": (5,),
    "confirmatory": (1, 5, 11),
}
SPLIT_COUNTS = {"plumbing": 8, "discovery": 24, "confirmatory": 48}
EXACT_BATCH_SIZE = 2
LOCKED_NUM_THREADS = 4
BOOTSTRAP_RESAMPLES = 200_000
BOOTSTRAP_SEEDS = {"discovery": 2026082611, "confirmatory": 2026082612}
EXCLUDED_LABELS = (
    0, 36, 50, 69, 78, 81, 95, 96, 100, 106, 113, 115, 120, 131, 142,
    144, 150, 152, 156, 159, 169, 188, 211, 222, 224, 238, 250, 251,
    300, 301, 316, 323, 346, 351, 361, 363, 377, 381, 384, 398, 416,
    421, 432, 441, 442, 445, 451, 465, 489, 500, 527, 531, 532, 545,
    582, 587, 588, 589, 594, 595, 599, 604, 615, 620, 637, 643, 656,
    658, 674, 683, 691, 710, 721, 724, 725, 732, 735, 738, 748, 750,
    764, 774, 778, 780, 788, 802, 815, 833, 853, 864, 870, 883, 884,
    887, 901, 902, 914, 915, 945, 954, 956, 963, 964, 971, 983, 987,
    992, 995, 998, 999,
)
SOURCE_MANIFESTS = (
    (
        "analyses/denoising_regret/manifests/fdrr_gate_v1.json",
        "bf2863839c92272a67bb2b54d489d799a1fb1e7ddf753b8546f67534f3637b04",
    ),
    (
        "analyses/expert_function/manifests/function_transport_gate_v1.json",
        "6e1b1941f17370824e7510495f83577e7e910783970b608d1e202617050736f5",
    ),
    (
        "analyses/timestep_utility/manifests/natural_timestep_utility_gate_v1.json",
        "4afa587649c0ba97fcfc6c2584cd170708e2da7db70e53cde5b86306078c21e4",
    ),
    (
        "/home/dev/promoe-probes/base100k-routing-translation-stratified-"
        "heldout24-block3-v1/manifest.json",
        "199b28b90cf390a2b52181571edffea60976e7f368314d4d82aeaca188f86248",
    ),
    (
        "/home/dev/promoe-probes/base50k-routing-flip-heldout24-v1/manifest.json",
        "cc6d2b3310a7f20f4998e0e4e22720cf0a70c5c5ed88efb42f96cfa909340a5c",
    ),
    (
        "/home/dev/promoe-probes/base50k-routing-translation-heldout24/"
        "manifest.json",
        "a327fdb5d31cb7d290a54556b69dadcefc994e33dd1ecb93f237fb0e4a0cff8a",
    ),
    (
        "/home/dev/promoe-probes/base50k-routing-translation-heldout24-"
        "margin-v2/manifest.json",
        "627942b10d86f40f979bcdda69e73bd656d40d60f7403533d4da1656779b29a8",
    ),
    (
        "/home/dev/promoe-probes/base50k-routing-translation-heldout24-"
        "multiblock/manifest.json",
        "aff161c0580e900b43b005a26c2ce9710e4fd5e0882dc3124e4a32063379b88c",
    ),
    (
        "/home/dev/promoe-probes/base50k-routing-translation-stratified-"
        "heldout24-multiblock-v1/manifest.json",
        "8ecfbd2c9e6b15de4bde7aabe11605fc2970450ee3c66e934ba3ce045e802796",
    ),
    (
        "/home/dev/promoe-probes/base50k-routing-translation-stratified-"
        "heldout24-v1/manifest.json",
        "24686c76afd9b76071c418cf91b898c53a41a5194f29bd82962cb41bdb21fce4",
    ),
    (
        "/home/dev/promoe-probes/fdrr50k-vs-base50k-routing-mechanism-v1/"
        "manifest.json",
        "52d3da52728d2ba1756c1be988015d61672b6e21dd2330999c39fe71c9a154d2",
    ),
)


DISCOVERY_REQUIREMENTS = {
    "minimum_mean_selected_gain": 1e-4,
    "minimum_selected_gain_lcb": 0.0,
    "minimum_positive_images": 16,
    "minimum_mean_pair_concordance": 0.58,
    "minimum_pair_concordance_lcb": 0.52,
    "minimum_mean_selected_positive": 0.60,
    "minimum_selected_positive_lcb": 0.50,
    "minimum_per_flip_random_contrast": 0.0,
    "minimum_per_flip_random_contrast_lcb": 0.0,
    "minimum_single_gain_ratio": 0.25,
    "minimum_single_gain_ratio_lcb": 0.15,
    "minimum_positive_block_strata": 1,
    "minimum_positive_block_strata_lcb": 1,
    "minimum_positive_sigma_strata": 2,
    "minimum_positive_sigma_strata_lcb": 1,
    "minimum_unique_six_rate": 0.03,
    "minimum_unique_six_rate_lcb": 0.0,
    "minimum_images_with_unique_six": 8,
    "minimum_mixed_minus_four": 0.0,
    "minimum_mixed_minus_four_lcb": 0.0,
}
CONFIRMATORY_REQUIREMENTS = {
    "minimum_mean_selected_gain": 1e-4,
    "minimum_selected_gain_lcb": 5e-5,
    "minimum_positive_images": 32,
    "minimum_mean_pair_concordance": 0.60,
    "minimum_pair_concordance_lcb": 0.55,
    "minimum_mean_selected_positive": 0.65,
    "minimum_selected_positive_lcb": 0.55,
    "minimum_per_flip_random_contrast": 1e-5,
    "minimum_per_flip_random_contrast_lcb": 0.0,
    "minimum_single_gain_ratio": 0.30,
    "minimum_single_gain_ratio_lcb": 0.20,
    "minimum_positive_block_strata": 3,
    "minimum_positive_block_strata_lcb": 2,
    "minimum_positive_sigma_strata": 3,
    "minimum_positive_sigma_strata_lcb": 2,
    "minimum_unique_six_rate": 0.05,
    "minimum_unique_six_rate_lcb": 0.01,
    "minimum_images_with_unique_six": 16,
    "minimum_mixed_minus_four": 2.5e-5,
    "minimum_mixed_minus_four_lcb": 0.0,
}
SAFETY_REQUIREMENTS = {
    "maximum_reference_duplicate_relative_mse_drift": 1e-7,
    "maximum_reference_duplicate_output_drift": 5e-6,
    "maximum_noop_relative_mse_change": 1e-7,
    "maximum_noop_output_change": 5e-6,
    "maximum_forced_unforced_relative_mse_change": 1e-7,
    "maximum_forced_unforced_output_change": 5e-6,
    "maximum_paired_native_relative_mse_drift": 1e-7,
    "maximum_paired_native_output_drift": 5e-6,
    "required_count_mismatches": 0,
}


def requirements_for_split(split):
    if split == "discovery":
        return {
            **DISCOVERY_REQUIREMENTS,
            **SAFETY_REQUIREMENTS,
            "expected_case_count": SPLIT_COUNTS[split],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEEDS[split],
        }
    if split == "confirmatory":
        return {
            **CONFIRMATORY_REQUIREMENTS,
            **SAFETY_REQUIREMENTS,
            "expected_case_count": SPLIT_COUNTS[split],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEEDS[split],
        }
    if split == "plumbing":
        return {
            **SAFETY_REQUIREMENTS,
            "expected_case_count": SPLIT_COUNTS[split],
        }
    raise ValueError(f"Unknown split: {split}")


def _authorize_arms(arm_gates, six_retention, prerequisite_authorized_arms=None):
    gated_arms = ("four_cycle", "six_cycle", "mixed_cycle")
    passed_arms = []
    if arm_gates["four_cycle"]["passed"]:
        passed_arms.append("four_cycle")
    for arm in ("six_cycle", "mixed_cycle"):
        if arm_gates[arm]["passed"] and six_retention["passed"]:
            passed_arms.append(arm)

    if prerequisite_authorized_arms is None:
        return passed_arms, passed_arms
    prerequisite_authorized_arms = tuple(prerequisite_authorized_arms)
    if (
        len(prerequisite_authorized_arms)
        != len(set(prerequisite_authorized_arms))
        or any(arm not in gated_arms for arm in prerequisite_authorized_arms)
    ):
        raise ValueError("Prerequisite authorized arms are invalid")
    prerequisite_set = set(prerequisite_authorized_arms)
    authorized_arms = [
        arm for arm in passed_arms if arm in prerequisite_set
    ]
    return passed_arms, authorized_arms


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
        "locked_before_any_cycle_result": True,
        "salt": SELECTION_SALT,
        "class_rule": (
            "Exclude all labels in the 11 declared prior manifests, sort "
            "SHA256(salt|class|label03|synset), assign the first 24 to "
            "discovery and the next 48 to confirmation."
        ),
        "latent_rule": (
            "Sort latent basenames and select int(SHA256(salt|latent|label03|"
            "synset)[0:8],16) modulo the class latent count."
        ),
        "seed_rule": (
            "int(SHA256(salt|latent|label03|synset)[8:16],16) modulo 2147483647."
        ),
        "excluded_labels": list(EXCLUDED_LABELS),
        "source_manifests": [
            {"path": path, "sha256": digest}
            for path, digest in SOURCE_MANIFESTS
        ],
        "plumbing_rule": (
            "Eight previously observed Base-200K utility cases are used only "
            "for runtime and safety validation and never enter efficacy statistics."
        ),
    }


def _selected_case(split, label, class_dir):
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
        "label": int(label),
        "seed": int(seed),
        "synset": class_dir.name,
        "latent": f"{class_dir.name}/{latent.name}",
        "latent_sha256": sha256_file(latent),
    }


def _resolve_source_manifest(path, project_root):
    path = Path(path)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def load_manifest(manifest_path, latent_root, project_root):
    manifest_path = Path(manifest_path).resolve()
    latent_root = Path(latent_root).resolve()
    project_root = Path(project_root).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    if not latent_root.is_dir():
        raise NotADirectoryError(f"Latent root does not exist: {latent_root}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("name") != MANIFEST_NAME:
        raise ValueError("Manifest version or name does not match the locked gate")
    if payload.get("selection") != _canonical_selection():
        raise ValueError("Manifest selection rule is not canonical")
    for path, expected_hash in SOURCE_MANIFESTS:
        source = _resolve_source_manifest(path, project_root)
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"Prior manifest provenance changed: {source}")

    class_dirs = sorted(
        path
        for path in latent_root.iterdir()
        if path.is_dir()
        and len(path.name) == 9
        and path.name.startswith("n")
        and path.name[1:].isdigit()
    )
    if len(class_dirs) != 1000:
        raise ValueError(f"Expected 1000 ImageNet classes, found {len(class_dirs)}")
    ranked = sorted(
        (
            hashlib.sha256(
                f"{SELECTION_SALT}|class|{label:03d}|{path.name}".encode()
            ).hexdigest(),
            label,
            path,
        )
        for label, path in enumerate(class_dirs)
        if label not in set(EXCLUDED_LABELS)
    )
    expected_statistical = []
    for index, (_, label, class_dir) in enumerate(ranked[:72]):
        split = "discovery" if index < 24 else "confirmatory"
        expected_statistical.append(_selected_case(split, label, class_dir))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("Manifest cases must be a list")
    plumbing = [case for case in raw_cases if case.get("split") == "plumbing"]
    statistical = [case for case in raw_cases if case.get("split") != "plumbing"]
    if len(plumbing) != SPLIT_COUNTS["plumbing"]:
        raise ValueError("Manifest plumbing case count is not locked")
    if statistical != expected_statistical:
        raise ValueError("Manifest statistical cases differ from deterministic selection")

    cases = []
    for raw_case in raw_cases:
        if raw_case.get("split") not in SPLIT_COUNTS:
            raise ValueError("Manifest contains an unknown split")
        latent_path = (latent_root / raw_case["latent"]).resolve()
        try:
            latent_path.relative_to(latent_root)
        except ValueError as error:
            raise ValueError(f"Case {raw_case['id']} escapes latent root") from error
        if not latent_path.is_file():
            raise FileNotFoundError(f"Missing latent for {raw_case['id']}")
        actual_hash = sha256_file(latent_path)
        if actual_hash != raw_case.get("latent_sha256"):
            raise ValueError(f"Latent hash changed for {raw_case['id']}")
        cases.append({
            **raw_case,
            "latent_relative": raw_case["latent"],
            "latent": str(latent_path),
        })
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Manifest case IDs must be unique")
    for split, count in SPLIT_COUNTS.items():
        if sum(case["split"] == split for case in cases) != count:
            raise ValueError(f"Manifest case count differs for {split}")
    return {
        "version": 1,
        "name": MANIFEST_NAME,
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "selection": payload["selection"],
        "cases": cases,
    }


def _bootstrap_distribution(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite vector")
    generator = np.random.default_rng(seed)
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
    return means


def _bootstrap_summary(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    distribution = _bootstrap_distribution(values, resamples, seed)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "iqr": [
            float(np.quantile(values, 0.25)),
            float(np.quantile(values, 0.75)),
        ],
        "ci95": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
        "one_sided_lcb95": float(np.quantile(distribution, 0.05)),
        "values": values.tolist(),
    }


def _bootstrap_ratio(numerator, denominator, resamples, seed):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("Bootstrap ratio inputs must be aligned vectors")
    generator = np.random.default_rng(seed)
    ratios = np.empty(resamples, dtype=np.float64)
    chunk_size = 10_000
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = generator.integers(
            0,
            numerator.size,
            size=(stop - start, numerator.size),
        )
        numerator_means = numerator[indices].mean(axis=1)
        denominator_means = denominator[indices].mean(axis=1)
        ratios[start:stop] = np.divide(
            numerator_means,
            denominator_means,
            out=np.full_like(numerator_means, np.nan),
            where=denominator_means > 0,
        )
    valid = ratios[np.isfinite(ratios)]
    if denominator.mean() <= 0 or valid.size < int(0.95 * ratios.size):
        return {
            "ratio": None,
            "ci95": [None, None],
            "one_sided_lcb95": None,
        }
    point = float(numerator.mean() / denominator.mean())
    return {
        "ratio": point,
        "ci95": [float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))],
        "one_sided_lcb95": float(np.quantile(valid, 0.05)),
    }


def _case_metrics(result, split):
    expected_blocks = BLOCKS_BY_SPLIT[split]
    cells = result.get("cells", [])
    expected_cells = len(expected_blocks) * len(SIGMAS)
    if len(cells) != expected_cells:
        raise ValueError("Case cell count does not match the locked split")
    observed_cells = {
        (int(cell["block_index"]), float(cell["sigma"])) for cell in cells
    }
    expected_cell_set = set((block, sigma) for block in expected_blocks for sigma in SIGMAS)
    if observed_cells != expected_cell_set:
        raise ValueError("Case block/sigma cells do not match the locked split")

    arms = {}
    for arm in ARM_NAMES:
        summaries = [cell["arms"][arm]["summary"] for cell in cells]
        arms[arm] = {
            "selected_gain": float(np.mean([
                summary["selected_gain"] for summary in summaries
            ])),
            "selected_per_flip_gain": float(np.mean([
                summary["selected_per_flip_gain"] for summary in summaries
            ])),
            "pair_concordance": float(np.mean([
                summary["pair_concordance"] for summary in summaries
            ])),
            "selected_positive": float(np.mean([
                summary["selected_positive"] for summary in summaries
            ])),
            "selected_harm": float(np.mean([
                summary["selected_harm"] for summary in summaries
            ])),
            "per_block_selected_gain": {
                str(block): float(np.mean([
                    cell["arms"][arm]["summary"]["selected_gain"]
                    for cell in cells if cell["block_index"] == block
                ]))
                for block in expected_blocks
            },
            "per_sigma_selected_gain": {
                str(sigma): float(np.mean([
                    cell["arms"][arm]["summary"]["selected_gain"]
                    for cell in cells if cell["sigma"] == sigma
                ]))
                for sigma in SIGMAS
            },
        }

    safety = {
        "reference_duplicate_relative_mse_drift": 0.0,
        "reference_duplicate_output_drift": 0.0,
        "noop_relative_mse_change": 0.0,
        "noop_output_change": 0.0,
        "forced_unforced_relative_mse_change": 0.0,
        "forced_unforced_output_change": 0.0,
        "paired_native_relative_mse_drift": 0.0,
        "paired_native_output_drift": 0.0,
        "count_mismatches": 0,
    }
    for cell in cells:
        controls = cell["numerical_controls"]
        native_mse = float(cell["native_mse"])
        safety["reference_duplicate_relative_mse_drift"] = max(
            safety["reference_duplicate_relative_mse_drift"],
            controls["max_abs_reference_duplicate_mse_drift"] / native_mse,
        )
        safety["reference_duplicate_output_drift"] = max(
            safety["reference_duplicate_output_drift"],
            controls["max_abs_reference_duplicate_output_drift"],
        )
        safety["noop_relative_mse_change"] = max(
            safety["noop_relative_mse_change"],
            controls["max_abs_noop_mse_change"] / native_mse,
        )
        safety["noop_output_change"] = max(
            safety["noop_output_change"],
            controls["max_abs_noop_output_change"],
        )
        safety["forced_unforced_relative_mse_change"] = max(
            safety["forced_unforced_relative_mse_change"],
            controls["max_abs_forced_unforced_mse_change"] / native_mse,
        )
        safety["forced_unforced_output_change"] = max(
            safety["forced_unforced_output_change"],
            controls["max_abs_forced_unforced_output_change"],
        )
        safety["paired_native_relative_mse_drift"] = max(
            safety["paired_native_relative_mse_drift"],
            controls["max_abs_paired_native_mse_drift"] / native_mse,
        )
        safety["paired_native_output_drift"] = max(
            safety["paired_native_output_drift"],
            controls["max_abs_paired_native_output_drift"],
        )
        safety["count_mismatches"] += int(controls["count_mismatches"])
    six_audits = [cell["six_cycle_audit"] for cell in cells]
    return {
        "case_id": result["batch_case"]["id"],
        "arms": arms,
        "unique_six_rate": float(np.mean([
            audit["unique_six_rate"] for audit in six_audits
        ])),
        "has_unique_six": bool(any(
            audit["has_unique_six"] for audit in six_audits
        )),
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
            "noop_relative_mse_change",
            "noop_output_change",
            "forced_unforced_relative_mse_change",
            "forced_unforced_output_change",
            "paired_native_relative_mse_drift",
            "paired_native_output_drift",
        )
    }
    total_mismatches = sum(
        metric["safety"]["count_mismatches"] for metric in metrics
    )
    checks = {
        "reference_duplicate_relative_mse": _check(
            maxima["reference_duplicate_relative_mse_drift"],
            f"<={requirements['maximum_reference_duplicate_relative_mse_drift']}",
            maxima["reference_duplicate_relative_mse_drift"]
            <= requirements["maximum_reference_duplicate_relative_mse_drift"],
        ),
        "reference_duplicate_output": _check(
            maxima["reference_duplicate_output_drift"],
            f"<={requirements['maximum_reference_duplicate_output_drift']}",
            maxima["reference_duplicate_output_drift"]
            <= requirements["maximum_reference_duplicate_output_drift"],
        ),
        "noop_relative_mse": _check(
            maxima["noop_relative_mse_change"],
            f"<={requirements['maximum_noop_relative_mse_change']}",
            maxima["noop_relative_mse_change"]
            <= requirements["maximum_noop_relative_mse_change"],
        ),
        "noop_output": _check(
            maxima["noop_output_change"],
            f"<={requirements['maximum_noop_output_change']}",
            maxima["noop_output_change"]
            <= requirements["maximum_noop_output_change"],
        ),
        "forced_unforced_relative_mse": _check(
            maxima["forced_unforced_relative_mse_change"],
            f"<={requirements['maximum_forced_unforced_relative_mse_change']}",
            maxima["forced_unforced_relative_mse_change"]
            <= requirements["maximum_forced_unforced_relative_mse_change"],
        ),
        "forced_unforced_output": _check(
            maxima["forced_unforced_output_change"],
            f"<={requirements['maximum_forced_unforced_output_change']}",
            maxima["forced_unforced_output_change"]
            <= requirements["maximum_forced_unforced_output_change"],
        ),
        "paired_native_relative_mse": _check(
            maxima["paired_native_relative_mse_drift"],
            f"<={requirements['maximum_paired_native_relative_mse_drift']}",
            maxima["paired_native_relative_mse_drift"]
            <= requirements["maximum_paired_native_relative_mse_drift"],
        ),
        "paired_native_output": _check(
            maxima["paired_native_output_drift"],
            f"<={requirements['maximum_paired_native_output_drift']}",
            maxima["paired_native_output_drift"]
            <= requirements["maximum_paired_native_output_drift"],
        ),
        "count_mismatches": _check(
            total_mismatches,
            f"=={requirements['required_count_mismatches']}",
            total_mismatches == requirements["required_count_mismatches"],
        ),
    }
    return checks, bool(all(check["passed"] for check in checks.values()))


def _arm_gate(metrics, arm, requirements, resamples, seed):
    selected = np.asarray([
        metric["arms"][arm]["selected_gain"] for metric in metrics
    ], dtype=np.float64)
    concordance = np.asarray([
        metric["arms"][arm]["pair_concordance"] for metric in metrics
    ], dtype=np.float64)
    selected_positive = np.asarray([
        metric["arms"][arm]["selected_positive"] for metric in metrics
    ], dtype=np.float64)
    per_flip = np.asarray([
        metric["arms"][arm]["selected_per_flip_gain"] for metric in metrics
    ], dtype=np.float64)
    random_per_flip = np.asarray([
        metric["arms"]["random_joint"]["selected_per_flip_gain"]
        for metric in metrics
    ], dtype=np.float64)
    single = np.asarray([
        metric["arms"]["single_token"]["selected_gain"] for metric in metrics
    ], dtype=np.float64)
    selected_summary = _bootstrap_summary(selected, resamples, seed)
    concordance_summary = _bootstrap_summary(concordance, resamples, seed + 1)
    positive_summary = _bootstrap_summary(selected_positive, resamples, seed + 2)
    random_contrast = _bootstrap_summary(
        per_flip - random_per_flip,
        resamples,
        seed + 3,
    )
    ratio = _bootstrap_ratio(selected, single, resamples, seed + 4)

    strata = {}
    stratum_values = {"block": [], "sigma": []}
    for kind, metric_name in (
        ("block", "per_block_selected_gain"),
        ("sigma", "per_sigma_selected_gain"),
    ):
        keys = tuple(metrics[0]["arms"][arm][metric_name])
        for key in keys:
            values = np.asarray([
                metric["arms"][arm][metric_name][str(key)] for metric in metrics
            ], dtype=np.float64)
            summary = _bootstrap_summary(
                values,
                resamples,
                seed + 10 + sum(len(items) for items in stratum_values.values()),
            )
            stratum_key = f"{kind}:{key}"
            strata[stratum_key] = summary
            stratum_values[kind].append(summary)
    positive_strata = {
        kind: sum(summary["mean"] > 0 for summary in summaries)
        for kind, summaries in stratum_values.items()
    }
    positive_strata_lcb = {
        kind: sum(
            summary["one_sided_lcb95"] > 0 for summary in summaries
        )
        for kind, summaries in stratum_values.items()
    }
    positive_images = int((selected > 0).sum())
    checks = {
        "selected_gain_mean": _check(
            selected_summary["mean"],
            f">={requirements['minimum_mean_selected_gain']}",
            selected_summary["mean"] >= requirements["minimum_mean_selected_gain"],
        ),
        "selected_gain_lcb": _check(
            selected_summary["one_sided_lcb95"],
            f">{requirements['minimum_selected_gain_lcb']}",
            selected_summary["one_sided_lcb95"]
            > requirements["minimum_selected_gain_lcb"],
        ),
        "positive_images": _check(
            positive_images,
            f">={requirements['minimum_positive_images']}",
            positive_images >= requirements["minimum_positive_images"],
        ),
        "pair_concordance_mean": _check(
            concordance_summary["mean"],
            f">={requirements['minimum_mean_pair_concordance']}",
            concordance_summary["mean"]
            >= requirements["minimum_mean_pair_concordance"],
        ),
        "pair_concordance_lcb": _check(
            concordance_summary["one_sided_lcb95"],
            f">{requirements['minimum_pair_concordance_lcb']}",
            concordance_summary["one_sided_lcb95"]
            > requirements["minimum_pair_concordance_lcb"],
        ),
        "selected_positive_mean": _check(
            positive_summary["mean"],
            f">={requirements['minimum_mean_selected_positive']}",
            positive_summary["mean"]
            >= requirements["minimum_mean_selected_positive"],
        ),
        "selected_positive_lcb": _check(
            positive_summary["one_sided_lcb95"],
            f">{requirements['minimum_selected_positive_lcb']}",
            positive_summary["one_sided_lcb95"]
            > requirements["minimum_selected_positive_lcb"],
        ),
        "per_flip_random_contrast_mean": _check(
            random_contrast["mean"],
            f">={requirements['minimum_per_flip_random_contrast']}",
            random_contrast["mean"]
            >= requirements["minimum_per_flip_random_contrast"],
        ),
        "per_flip_random_contrast_lcb": _check(
            random_contrast["one_sided_lcb95"],
            f">{requirements['minimum_per_flip_random_contrast_lcb']}",
            random_contrast["one_sided_lcb95"]
            > requirements["minimum_per_flip_random_contrast_lcb"],
        ),
        "single_gain_ratio": _check(
            ratio["ratio"],
            f">={requirements['minimum_single_gain_ratio']}",
            ratio["ratio"] is not None
            and ratio["ratio"] >= requirements["minimum_single_gain_ratio"],
        ),
        "single_gain_ratio_lcb": _check(
            ratio["one_sided_lcb95"],
            f">{requirements['minimum_single_gain_ratio_lcb']}",
            ratio["one_sided_lcb95"] is not None
            and ratio["one_sided_lcb95"]
            > requirements["minimum_single_gain_ratio_lcb"],
        ),
        **{
            f"positive_{kind}_strata": _check(
                positive_strata[kind],
                f">={requirements[f'minimum_positive_{kind}_strata']}",
                positive_strata[kind]
                >= requirements[f"minimum_positive_{kind}_strata"],
            )
            for kind in ("block", "sigma")
        },
        **{
            f"positive_{kind}_strata_lcb": _check(
                positive_strata_lcb[kind],
                f">={requirements[f'minimum_positive_{kind}_strata_lcb']}",
                positive_strata_lcb[kind]
                >= requirements[f"minimum_positive_{kind}_strata_lcb"],
            )
            for kind in ("block", "sigma")
        },
    }
    return {
        "passed": bool(all(check["passed"] for check in checks.values())),
        "checks": checks,
        "selected_gain": selected_summary,
        "pair_concordance": concordance_summary,
        "selected_positive": positive_summary,
        "per_flip_random_contrast": random_contrast,
        "single_gain_ratio": ratio,
        "strata": strata,
        "positive_strata": positive_strata,
        "positive_strata_lcb": positive_strata_lcb,
    }


def _six_retention_gate(metrics, requirements, resamples, seed):
    unique_rate = np.asarray([
        metric["unique_six_rate"] for metric in metrics
    ], dtype=np.float64)
    mixed_minus_four = np.asarray([
        metric["arms"]["mixed_cycle"]["selected_gain"]
        - metric["arms"]["four_cycle"]["selected_gain"]
        for metric in metrics
    ], dtype=np.float64)
    unique_summary = _bootstrap_summary(unique_rate, resamples, seed)
    mixed_summary = _bootstrap_summary(mixed_minus_four, resamples, seed + 1)
    images = int(sum(metric["has_unique_six"] for metric in metrics))
    checks = {
        "unique_six_rate": _check(
            unique_summary["mean"],
            f">={requirements['minimum_unique_six_rate']}",
            unique_summary["mean"] >= requirements["minimum_unique_six_rate"],
        ),
        "unique_six_rate_lcb": _check(
            unique_summary["one_sided_lcb95"],
            f">{requirements['minimum_unique_six_rate_lcb']}",
            unique_summary["one_sided_lcb95"]
            > requirements["minimum_unique_six_rate_lcb"],
        ),
        "images_with_unique_six": _check(
            images,
            f">={requirements['minimum_images_with_unique_six']}",
            images >= requirements["minimum_images_with_unique_six"],
        ),
        "mixed_minus_four_mean": _check(
            mixed_summary["mean"],
            f">={requirements['minimum_mixed_minus_four']}",
            mixed_summary["mean"] >= requirements["minimum_mixed_minus_four"],
        ),
        "mixed_minus_four_lcb": _check(
            mixed_summary["one_sided_lcb95"],
            f">{requirements['minimum_mixed_minus_four_lcb']}",
            mixed_summary["one_sided_lcb95"]
            > requirements["minimum_mixed_minus_four_lcb"],
        ),
    }
    return {
        "passed": bool(all(check["passed"] for check in checks.values())),
        "checks": checks,
        "unique_six_rate": unique_summary,
        "mixed_minus_four": mixed_summary,
        "images_with_unique_six": images,
    }


def _bh_fdr(p_values):
    names = list(p_values)
    values = np.asarray([p_values[name] for name in names], dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return {name: float(adjusted[index]) for index, name in enumerate(names)}


def _secondary_stratum_tests(metrics, arm):
    p_values = {}
    for kind, metric_name, keys in (
        (
            "block",
            "per_block_selected_gain",
            metrics[0]["arms"][arm]["per_block_selected_gain"],
        ),
        (
            "sigma",
            "per_sigma_selected_gain",
            metrics[0]["arms"][arm]["per_sigma_selected_gain"],
        ),
    ):
        for key in keys:
            values = np.asarray([
                metric["arms"][arm][metric_name][key] for metric in metrics
            ], dtype=np.float64)
            if np.all(values == 0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(
                    values,
                    alternative="greater",
                    zero_method="zsplit",
                ).pvalue)
            p_values[f"{kind}:{key}"] = p_value
    adjusted = _bh_fdr(p_values)
    return {
        name: {
            "one_sided_wilcoxon_p": p_values[name],
            "bh_fdr_q": adjusted[name],
            "q_le_0p05": bool(adjusted[name] <= 0.05),
        }
        for name in p_values
    }


def aggregate_case_results(
    case_results,
    split,
    requirements=None,
    prerequisite_authorized_arms=None,
):
    expected_requirements = requirements_for_split(split)
    requirements = dict(requirements or expected_requirements)
    if requirements != expected_requirements:
        raise ValueError("Gate requirements differ from the locked protocol")
    if len(case_results) != requirements["expected_case_count"]:
        raise ValueError("Case count does not match the locked split")
    if split == "confirmatory" and prerequisite_authorized_arms is None:
        raise ValueError(
            "Confirmatory aggregation requires discovery-authorized arms"
        )
    if split != "confirmatory" and prerequisite_authorized_arms is not None:
        raise ValueError(
            "Only confirmatory aggregation accepts prerequisite arms"
        )
    metrics = [_case_metrics(result, split) for result in case_results]
    if len({metric["case_id"] for metric in metrics}) != len(metrics):
        raise ValueError("Duplicate case IDs in aggregate")
    safety_checks, safety_passed = _safety_gate(metrics, requirements)
    if split == "plumbing":
        return {
            "split": split,
            "probe_version": PROBE_VERSION,
            "safety_checks": safety_checks,
            "safety_passed": safety_passed,
            "efficacy_statistics_withheld": True,
            "passed": safety_passed,
        }

    resamples = requirements["bootstrap_resamples"]
    seed = requirements["bootstrap_seed"]
    arm_gates = {
        arm: _arm_gate(
            metrics,
            arm,
            requirements,
            resamples,
            seed + 100 * index,
        )
        for index, arm in enumerate((
            "four_cycle",
            "six_cycle",
            "mixed_cycle",
        ))
    }
    six_retention = _six_retention_gate(
        metrics,
        requirements,
        resamples,
        seed + 500,
    )
    passed_arms, authorized = _authorize_arms(
        arm_gates,
        six_retention,
        prerequisite_authorized_arms,
    )
    recommended = (
        "mixed_cycle" if "mixed_cycle" in authorized
        else "four_cycle" if "four_cycle" in authorized
        else "six_cycle" if "six_cycle" in authorized
        else None
    )
    secondary = {
        arm: _secondary_stratum_tests(metrics, arm)
        for arm in ("four_cycle", "six_cycle", "mixed_cycle")
    }
    route_passed = bool(safety_passed and authorized)
    return {
        "split": split,
        "probe_version": PROBE_VERSION,
        "requirements": requirements,
        "safety_checks": safety_checks,
        "safety_passed": safety_passed,
        "arm_gates": arm_gates,
        "six_cycle_retention": six_retention,
        "passed_arms": passed_arms,
        "prerequisite_authorized_arms": (
            list(prerequisite_authorized_arms)
            if prerequisite_authorized_arms is not None
            else None
        ),
        "authorized_arms": authorized,
        "recommended_arm": recommended,
        "secondary_stratum_tests": secondary,
        "image_metrics": metrics,
        "route_passed": route_passed,
        "passed": route_passed,
        "interpretation": (
            "A passing discovery gate authorizes only the locked confirmatory "
            "probe. Confirmatory authorization is the intersection of discovery-"
            "authorized and confirmatory-passing arms. A passing confirmatory gate "
            "authorizes a small router-fitting prototype, not ImageNet long training "
            "or a generation claim; a second checkpoint or seed replication remains "
            "required."
        ),
    }
