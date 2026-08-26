from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .probe import summarize_records


BATCH_VERSION = 4
REQUIRED_PROBE_VERSION = 4
FDRR_GATE_SIGMAS = (0.2, 0.5, 0.8)
FDRR_GATE_BLOCK_INDEX = 3
FDRR_GATE_EXACT_BATCH_SIZE = 4
FDRR_GATE_MIN_CHECKPOINT_STEP = 10000
FDRR_GATE_MODEL_NAME = "ProMoE_TC_REPA_Multi_Align_B"
FDRR_GATE_MANIFEST_NAME = "fdrr_gate_v1"
FDRR_GATE_CASE_SPECS = (
    (
        "class000_n01440764_10027",
        0,
        11,
        "n01440764",
        "n01440764/n01440764_10027.latent.npz",
    ),
    (
        "class100_n01860187_10016",
        100,
        23,
        "n01860187",
        "n01860187/n01860187_10016.latent.npz",
    ),
    (
        "class250_n02110185_10002",
        250,
        37,
        "n02110185",
        "n02110185/n02110185_10002.latent.npz",
    ),
    (
        "class500_n03042490_10010",
        500,
        53,
        "n03042490",
        "n03042490/n03042490_10010.latent.npz",
    ),
    (
        "class750_n04033995_10001",
        750,
        71,
        "n04033995",
        "n04033995/n04033995_10001.latent.npz",
    ),
    (
        "class999_n15075141_10080",
        999,
        89,
        "n15075141",
        "n15075141/n15075141_10080.latent.npz",
    ),
)
FDRR_GATE_REQUIREMENTS = {
    "min_cases": 6,
    "min_spearman": 0.25,
    "min_sign_agreement": 0.60,
    "min_beneficial_challenger_rate": 0.15,
    "max_noop_abs_mse_change": 1e-12,
    "required_token_probes_per_sigma": 256,
}


def _require_int(value, name, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def load_manifest(manifest_path, latent_root):
    manifest_path = Path(manifest_path).resolve()
    latent_root = Path(latent_root).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    if not latent_root.is_dir():
        raise NotADirectoryError(f"Latent root does not exist: {latent_root}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_cases = [
        {
            "id": case_id,
            "label": label,
            "seed": seed,
            "synset": synset,
            "latent": latent,
        }
        for case_id, label, seed, synset, latent in FDRR_GATE_CASE_SPECS
    ]
    expected_payload = {
        "version": 1,
        "name": FDRR_GATE_MANIFEST_NAME,
        "cases": expected_cases,
    }
    if payload != expected_payload:
        raise ValueError(
            "Manifest content does not match the canonical fdrr_gate_v1 cases"
        )
    manifest_name = payload["name"]
    raw_cases = payload["cases"]

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
            f"Expected 1000 ImageNet synset directories, found {len(class_names)}"
        )

    cases = []
    seen_ids = set()
    seen_latents = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Manifest case {index} must be an object")
        case_id = raw_case.get("id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or any(not (char.isalnum() or char in "_-") for char in case_id)
        ):
            raise ValueError(
                f"Manifest case {index} id must use letters, digits, '_' or '-'"
            )
        if case_id in seen_ids:
            raise ValueError(f"Duplicate manifest case id: {case_id}")
        seen_ids.add(case_id)

        label = _require_int(raw_case.get("label"), f"case {case_id} label", 0, 999)
        seed = _require_int(raw_case.get("seed"), f"case {case_id} seed", 0)
        synset = raw_case.get("synset")
        if synset != class_names[label]:
            raise ValueError(
                f"case {case_id} synset {synset!r} does not match label {label} "
                f"({class_names[label]})"
            )

        relative_path = Path(raw_case.get("latent", ""))
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or ".." in relative_path.parts
            or relative_path.parts[0] != synset
        ):
            raise ValueError(f"case {case_id} latent must be relative to its synset")
        latent_path = (latent_root / relative_path).resolve()
        try:
            latent_path.relative_to(latent_root)
        except ValueError as error:
            raise ValueError(f"case {case_id} latent escapes the latent root") from error
        if not latent_path.is_file():
            raise FileNotFoundError(f"case {case_id} latent does not exist: {latent_path}")
        if latent_path in seen_latents:
            raise ValueError(f"Duplicate manifest latent: {latent_path}")
        seen_latents.add(latent_path)

        latent_key = raw_case.get("latent_key", "latent")
        if latent_key not in {"latent", "latent_flip"}:
            raise ValueError(f"case {case_id} has unsupported latent_key {latent_key!r}")
        cases.append({
            "id": case_id,
            "label": label,
            "seed": seed,
            "synset": synset,
            "latent": str(latent_path),
            "latent_relative": relative_path.as_posix(),
            "latent_key": latent_key,
        })

    return {
        "version": 1,
        "name": manifest_name,
        "path": str(manifest_path),
        "latent_root": str(latent_root),
        "cases": cases,
    }


def _sigma_regions(sigmas):
    return {
        "low": any(sigma <= 1.0 / 3.0 for sigma in sigmas),
        "middle": any(1.0 / 3.0 < sigma < 2.0 / 3.0 for sigma in sigmas),
        "high": any(sigma >= 2.0 / 3.0 for sigma in sigmas),
    }


def _metric_passes(value, minimum):
    return value is not None and value >= minimum


def _case_signature(case):
    return (
        case.get("id"),
        case.get("label"),
        case.get("seed"),
        case.get("synset"),
        case.get("latent_relative"),
        case.get("latent_key"),
    )


def _canonical_case_signatures():
    return tuple(
        (case_id, label, seed, synset, latent, "latent")
        for case_id, label, seed, synset, latent in FDRR_GATE_CASE_SPECS
    )


def _validate_requirements(requirements):
    required_keys = set(FDRR_GATE_REQUIREMENTS)
    if set(requirements) != required_keys:
        raise ValueError(
            f"Gate requirements must contain exactly {sorted(required_keys)}"
        )
    _require_int(requirements["min_cases"], "min_cases", 1)
    _require_int(
        requirements["required_token_probes_per_sigma"],
        "required_token_probes_per_sigma",
        2,
    )
    if not -1 <= requirements["min_spearman"] <= 1:
        raise ValueError("min_spearman must be in [-1, 1]")
    for name in (
        "min_sign_agreement",
        "min_beneficial_challenger_rate",
    ):
        if not 0 <= requirements[name] <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if requirements["max_noop_abs_mse_change"] < 0:
        raise ValueError("max_noop_abs_mse_change must be non-negative")


def build_gate_summary(case_results, requirements):
    if not case_results:
        raise ValueError("At least one case result is required")
    _validate_requirements(requirements)
    sigmas = [float(value) for value in case_results[0]["sigmas"]]
    if (
        len(sigmas) != len(set(sigmas))
        or any(not 0 < sigma < 1 for sigma in sigmas)
    ):
        raise ValueError("Probe sigmas must be unique and strictly between 0 and 1")
    all_records = []
    cell_results = []
    contract_failures = []
    reference_contract = {
        key: case_results[0].get(key)
        for key in (
            "checkpoint",
            "weights_checkpoint",
            "checkpoint_step",
            "weights_checkpoint_step",
            "checkpoint_state",
            "config",
            "model_name",
            "block_index",
            "candidate_mode",
            "num_token_probes_requested",
            "exact_batch_size",
            "device",
            "num_threads",
        )
    }
    seen_case_ids = set()
    case_signatures = []

    for result in case_results:
        batch_case = result.get("batch_case")
        if not isinstance(batch_case, dict):
            raise ValueError("Every probe result must contain a batch_case object")
        case_id = batch_case.get("id")
        case_signatures.append(_case_signature(batch_case))
        if case_id in seen_case_ids:
            contract_failures.append(f"{case_id}: duplicate case id")
        seen_case_ids.add(case_id)
        for key, expected_value in reference_contract.items():
            if result.get(key) != expected_value:
                contract_failures.append(f"{case_id}: {key} differs")
        if [float(value) for value in result["sigmas"]] != sigmas:
            contract_failures.append(f"{case_id}: sigma list differs")
        if result.get("probe_version") != REQUIRED_PROBE_VERSION:
            contract_failures.append(
                f"{case_id}: probe_version is not {REQUIRED_PROBE_VERSION}"
            )
        if result.get("counterfactual_route_weight") != "selected":
            contract_failures.append(f"{case_id}: route weight is not fixed-selected")
        if result.get("model_name") != FDRR_GATE_MODEL_NAME:
            contract_failures.append(f"{case_id}: model is not the Multi-Align baseline")
        checkpoint_step = result.get("checkpoint_step")
        if (
            isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step < FDRR_GATE_MIN_CHECKPOINT_STEP
        ):
            contract_failures.append(
                f"{case_id}: checkpoint step is below "
                f"{FDRR_GATE_MIN_CHECKPOINT_STEP}"
            )
        if result.get("weights_checkpoint_step") != checkpoint_step:
            contract_failures.append(f"{case_id}: loaded checkpoint step differs")
        if result.get("block_index") != FDRR_GATE_BLOCK_INDEX:
            contract_failures.append(f"{case_id}: block index is not fixed")
        if result.get("candidate_mode") != "mixed":
            contract_failures.append(f"{case_id}: candidate mode is not mixed")
        if result.get("exact_batch_size") != FDRR_GATE_EXACT_BATCH_SIZE:
            contract_failures.append(f"{case_id}: exact batch size is not fixed")
        if result.get("device") != "cpu":
            contract_failures.append(f"{case_id}: device is not CPU")
        if result.get("label") != batch_case.get("label"):
            contract_failures.append(f"{case_id}: result label differs from manifest")
        if result.get("latent") != batch_case.get("latent"):
            contract_failures.append(f"{case_id}: result latent differs from manifest")
        if result.get("latent_key") != batch_case.get("latent_key"):
            contract_failures.append(
                f"{case_id}: result latent key differs from manifest"
            )
        if result.get("seed") != batch_case.get("seed"):
            contract_failures.append(f"{case_id}: result seed differs from manifest")
        if result.get("num_token_probes_requested") != requirements[
            "required_token_probes_per_sigma"
        ]:
            contract_failures.append(f"{case_id}: token-probe request is not fixed")

        result_records = result["records"]
        all_records.extend(result_records)
        unexpected_sigmas = {
            float(record["sigma"])
            for record in result_records
            if float(record["sigma"]) not in sigmas
        }
        if unexpected_sigmas:
            contract_failures.append(
                f"{case_id}: records contain unexpected sigmas "
                f"{sorted(unexpected_sigmas)}"
            )
        for sigma in sigmas:
            sigma_key = str(sigma)
            diagnostics = result["numerical_controls"][sigma_key]
            records = [
                record
                for record in result_records
                if float(record["sigma"]) == sigma
            ]
            summary = summarize_records(records)
            cached_summary = result["per_sigma"].get(sigma_key)
            cached_summary_matches = cached_summary == summary
            if not cached_summary_matches:
                contract_failures.append(
                    f"{case_id}: cached summary differs at sigma {sigma}"
                )
            source_counts = Counter(
                record.get("challenger_source") for record in records
            )
            expected_sources = {
                "runner-up": (len(records) + 1) // 2,
                "random": len(records) // 2,
            }
            source_split_passed = dict(source_counts) == expected_sources
            metrics_passed = {
                "spearman": _metric_passes(
                    summary["spearman"], requirements["min_spearman"]
                ),
                "sign_agreement": _metric_passes(
                    summary["sign_agreement"],
                    requirements["min_sign_agreement"],
                ),
                "beneficial_challenger_rate": _metric_passes(
                    summary["exact_better_rate"],
                    requirements["min_beneficial_challenger_rate"],
                ),
                "noop_control": (
                    diagnostics["noop_max_abs_mse_change"]
                    <= requirements["max_noop_abs_mse_change"]
                ),
                "token_probe_count": (
                    summary["num_probes"]
                    == requirements["required_token_probes_per_sigma"]
                    == len(records)
                ),
                "mixed_source_split": source_split_passed,
                "cached_summary_matches_records": cached_summary_matches,
            }
            cell_results.append({
                "case_id": case_id,
                "label": int(result["label"]),
                "sigma": sigma,
                "num_probes": summary["num_probes"],
                "spearman": summary["spearman"],
                "sign_agreement": summary["sign_agreement"],
                "beneficial_challenger_rate": summary["exact_better_rate"],
                "median_abs_exact_mse_change": summary[
                    "median_abs_exact_mse_change"
                ],
                "noop_max_abs_mse_change": diagnostics[
                    "noop_max_abs_mse_change"
                ],
                "challenger_source_counts": dict(source_counts),
                "checks": metrics_passed,
                "passed": all(metrics_passed.values()),
            })

    sigma_regions = _sigma_regions(sigmas)
    fixed_case_manifest = tuple(case_signatures) == _canonical_case_signatures()
    top_level_checks = {
        "fixed_case_manifest": fixed_case_manifest,
        "exact_case_count": len(case_results) == requirements["min_cases"],
        "fixed_sigmas": tuple(sigmas) == FDRR_GATE_SIGMAS,
        "probe_contract": not contract_failures,
        "every_case_sigma_cell": all(cell["passed"] for cell in cell_results),
    }
    per_sigma = {}
    for sigma in sigmas:
        per_sigma[str(sigma)] = summarize_records([
            record for record in all_records if float(record["sigma"]) == sigma
        ])

    return {
        "batch_version": BATCH_VERSION,
        "requirements": requirements,
        "num_cases": len(case_results),
        "num_case_sigma_cells": len(cell_results),
        "sigmas": sigmas,
        "sigma_region_coverage": sigma_regions,
        "contract_failures": contract_failures,
        "checks": top_level_checks,
        "passed": all(top_level_checks.values()),
        "summary": summarize_records(all_records),
        "per_sigma": per_sigma,
        "cells": cell_results,
    }
