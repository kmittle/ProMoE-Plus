"""Locked gates for forward-only compute-exchange deployability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from analyses.timestep_utility.compute_exchange_deployability import (
    RETROSPECTIVE_BLOCKS,
    SIGMAS,
    _canonical_seed,
    _pair_concordance,
    array_sha256,
    build_same_expert_exchange_candidates,
    candidate_scores,
    solve_exact_exchange,
)
from analyses.timestep_utility.compute_exchange_deployability_fit import (
    predict_indices,
)
from analyses.timestep_utility.cycle_batch import sha256_file


BATCH_VERSION = 1
BOOTSTRAP_RESAMPLES = 200_000
BOOTSTRAP_SEED = 2026082647
FIT_REQUIREMENTS = {
    "minimum_primary_validation_concordance": 0.58,
    "minimum_primary_minus_router_context": 0.02,
    "minimum_primary_minus_rolled_correspondence": 0.08,
}
RETROSPECTIVE_REQUIREMENTS = {
    "expected_case_count": 48,
    "minimum_mean_gain": 1e-4,
    "minimum_gain_lcb": 5e-5,
    "minimum_positive_images": 32,
    "minimum_pair_concordance": 0.60,
    "minimum_pair_concordance_lcb": 0.55,
    "minimum_oracle_ratio": 0.25,
    "minimum_oracle_ratio_lcb": 0.15,
    "minimum_control_contrast_lcb": 0.0,
    "required_positive_blocks": 3,
    "required_positive_sigmas": 3,
    "required_action_invariance_mismatches": 0,
    "required_numerical_control_failures": 0,
    "maximum_source_reveal_native_mse_relative_drift": 1e-7,
}
ACTION_NAMES = (
    "primary",
    "router_context",
    "rolled_correspondence",
    "random",
    "router_margin",
)
CONTROL_NAMES = (
    "random",
    "router_margin",
    "rolled_utility",
    "router_context",
    "rolled_correspondence",
)
CANDIDATE_CONTRACT_KEYS = (
    "id",
    "donors",
    "receivers",
    "experts",
    "quota_by_expert",
    "transferred_passes",
    "native_pass_vector",
    "candidate_pass_vector",
)


def json_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_json_snapshot(path):
    payload = Path(path).read_bytes()
    return json.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def candidate_bank_sha256(rows):
    contract = [
        {key: row[key] for key in CANDIDATE_CONTRACT_KEYS}
        for row in rows
    ]
    return json_sha256(contract)


def verify_sealed_json(path, protocol_sha256, case_id, return_hashes=False):
    path = Path(path)
    seal_path = path.with_suffix(path.suffix + ".seal.json")
    if not path.is_file() or not seal_path.is_file():
        raise FileNotFoundError(f"Missing sealed payload: {path}")
    payload, payload_file_sha256 = _load_json_snapshot(path)
    seal, seal_file_sha256 = _load_json_snapshot(seal_path)
    if seal != {
        "case_id": case_id,
        "protocol_sha256": protocol_sha256,
        "result_sha256": json_sha256(payload),
        "version": 1,
    }:
        raise ValueError(f"Seal mismatch for {path}")
    if return_hashes:
        return payload, payload_file_sha256, seal_file_sha256
    return payload


def verify_source_gate(source_root, project_root):
    source_root = Path(source_root).resolve()
    protocol_path = source_root / "protocol.json"
    protocol_sha_path = source_root / "protocol.sha256"
    protocol, protocol_file_sha256 = _load_json_snapshot(protocol_path)
    protocol_sha256 = json_sha256(protocol)
    if protocol_sha_path.read_text(encoding="utf-8").strip() != protocol_sha256:
        raise ValueError("Source protocol SHA sidecar mismatch")
    for relative, expected_sha256 in protocol["project_source_sha256"].items():
        path = Path(project_root) / relative
        if sha256_file(path) != expected_sha256:
            raise RuntimeError(f"Sealed source file changed: {relative}")
    summaries = {}
    summary_file_sha256 = {}
    for split in ("discovery", "confirmatory"):
        summary_path = source_root / f"{split}-summary.json"
        summary, summary_sha256, _ = verify_sealed_json(
            summary_path,
            protocol_sha256,
            f"{split}-summary",
            return_hashes=True,
        )
        if not summary["gate"]["safety_passed"] or not summary["gate"]["efficacy_passed"]:
            raise RuntimeError(f"Source {split} gate did not pass")
        summaries[split] = summary
        summary_file_sha256[split] = summary_sha256
    cases = {}
    for split in ("discovery", "confirmatory"):
        assignments = protocol["assignments"][split]
        manifest = {
            case["id"]: case
            for case in protocol["manifest"]["cases"]
            if case["split"] == split
        }
        split_cases = []
        for assignment in assignments:
            case_id = assignment["case_id"]
            case = manifest[case_id]
            result_path = source_root / split / (
                f"{int(assignment['index']):02d}_{case_id}.json"
            )
            _, result_file_sha256, seal_file_sha256 = verify_sealed_json(
                result_path,
                protocol_sha256,
                case_id,
                return_hashes=True,
            )
            split_cases.append({
                **case,
                "source_result": str(result_path),
                "source_result_sha256": result_file_sha256,
                "source_seal": str(result_path.with_suffix(".json.seal.json")),
                "source_seal_sha256": seal_file_sha256,
            })
        cases[split] = split_cases
    return {
        "root": str(source_root),
        "protocol": protocol,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "protocol_file_sha256": protocol_file_sha256,
        "summaries": summaries,
        "summary_file_sha256": summary_file_sha256,
        "cases": cases,
    }


def fit_gate(fit_summaries, requirements=None):
    requirements = dict(requirements or FIT_REQUIREMENTS)
    if requirements != FIT_REQUIREMENTS:
        raise ValueError("Fit requirements differ from the locked gate")
    required_kinds = {"primary", "router_context", "rolled_correspondence"}
    if set(fit_summaries) != required_kinds:
        raise ValueError("Fit summaries do not cover all scorer controls")
    primary = float(fit_summaries["primary"]["true_target_validation_concordance"])
    router = float(
        fit_summaries["router_context"]["true_target_validation_concordance"]
    )
    rolled = float(
        fit_summaries["rolled_correspondence"]["true_target_validation_concordance"]
    )

    def check(observed, required, passed):
        return {"observed": observed, "required": required, "passed": bool(passed)}

    checks = {
        "primary_validation_concordance": check(
            primary,
            f">={requirements['minimum_primary_validation_concordance']}",
            primary >= requirements["minimum_primary_validation_concordance"],
        ),
        "primary_minus_router_context": check(
            primary - router,
            f">={requirements['minimum_primary_minus_router_context']}",
            primary - router >= requirements["minimum_primary_minus_router_context"],
        ),
        "primary_minus_rolled_correspondence": check(
            primary - rolled,
            f">={requirements['minimum_primary_minus_rolled_correspondence']}",
            primary - rolled >= requirements["minimum_primary_minus_rolled_correspondence"],
        ),
    }
    return {
        "requirements": requirements,
        "validation_concordance": {
            "primary": primary,
            "router_context": router,
            "rolled_correspondence": rolled,
        },
        "checks": checks,
        "passed": all(row["passed"] for row in checks.values()),
    }


def _source_cells(source):
    result = source if isinstance(source, dict) else load_json(source)
    return {
        (int(cell["block_index"]), float(cell["sigma"])): cell
        for cell in result["cells"]
    }


def _predict_cell(model, dataset, indices, device, batch_size):
    return predict_indices(model, dataset, indices, device, batch_size=batch_size)


def select_retrospective_actions(dataset, models, device):
    if dataset.targets is not None:
        raise ValueError("Retrospective dataset must not contain privileged targets")
    if set(models) != {"primary", "router_context", "rolled_correspondence"}:
        raise ValueError("Retrospective evaluation requires all scorer controls")
    if any(cell.get("source_result") is not None for cell in dataset.cells):
        raise ValueError("Action selection must not receive source-result paths")
    records = []
    for cell in dataset.cells:
        indices = np.arange(cell["token_start"], cell["token_stop"], dtype=np.int64)
        native = dataset.native_experts[indices]
        native_weights = dataset.native_weights[indices]
        candidates = build_same_expert_exchange_candidates(
            native,
            dataset.num_experts,
            int(cell["candidate_seed"]),
        )
        predictions = {
            name: _predict_cell(model, dataset, indices, device, len(indices))
            for name, model in models.items()
        }
        chunked_primary = _predict_cell(
            models["primary"],
            dataset,
            indices,
            device,
            batch_size=37,
        )
        random_scores = np.random.default_rng(_canonical_seed(
            "retrospective-random",
            cell["case_id"],
            int(cell["block_index"]),
            f"{float(cell['sigma']):.17g}",
        )).standard_normal((len(indices), 2))
        router_scores = dataset.router_scores[indices]
        top_two = np.partition(router_scores, -2, axis=1)[:, -2:]
        margin = top_two[:, 1] - top_two[:, 0]
        score_sets = {
            **predictions,
            "random": random_scores,
            "router_margin": np.column_stack((margin, -margin)),
        }
        actions = {}
        for name in ACTION_NAMES:
            action = solve_exact_exchange(
                native,
                score_sets[name],
                num_experts=dataset.num_experts,
            )
            action["id"] = f"exact:{name}"
            actions[name] = action
        chunked_action = solve_exact_exchange(
            native,
            chunked_primary,
            num_experts=dataset.num_experts,
        )
        action_mismatch = int(
            actions["primary"]["donors"] != chunked_action["donors"]
            or actions["primary"]["receivers"] != chunked_action["receivers"]
        )
        records.append({
            "case_id": cell["case_id"],
            "block_index": int(cell["block_index"]),
            "sigma": float(cell["sigma"]),
            "candidate_seed": int(cell["candidate_seed"]),
            "candidate_ids": [candidate["id"] for candidate in candidates],
            "candidate_bank_sha256": candidate_bank_sha256(candidates),
            "primary_candidate_priority": (
                -candidate_scores(candidates, predictions["primary"])
            ).tolist(),
            "route_id_sha256": array_sha256(native, np.int64),
            "route_weight_sha256": array_sha256(native_weights, np.float32),
            "actions": actions,
            "action_invariance_mismatch": action_mismatch,
            "logical_pass_counts_match": all(
                action["native_pass_vector"] == action["candidate_pass_vector"]
                for action in actions.values()
            ),
        })
    return records


def _reveal_safety_passed(controls, native_mse):
    if not np.isfinite(native_mse) or native_mse <= 0:
        return False
    relative_mse_limits = {
        "max_abs_noop_mse_change": 1e-7,
        "max_abs_hook_mse_change": 1e-12,
        "max_abs_single_vs_paired_native_mse_drift": 1e-7,
        "max_abs_paired_native_mse_drift": 1e-7,
    }
    output_limits = {
        "max_abs_noop_output_change": 5e-6,
        "max_abs_hook_output_change": 0.0,
        "max_abs_single_vs_paired_native_output_drift": 5e-6,
        "max_abs_paired_native_output_drift": 5e-6,
    }
    if any(
        float(controls[key]) / native_mse > limit
        for key, limit in relative_mse_limits.items()
    ):
        return False
    if any(float(controls[key]) > limit for key, limit in output_limits.items()):
        return False
    return all(int(controls[key]) == 0 for key in (
        "logical_count_mismatches",
        "action_contract_mismatches",
        "route_id_mismatches",
        "route_weight_mismatches",
    ))


def combine_retrospective_reveal(action_records, reveal_results, source_results):
    action_by_key = {
        (row["case_id"], int(row["block_index"]), float(row["sigma"])): row
        for row in action_records
    }
    if len(action_by_key) != len(action_records):
        raise ValueError("Sealed actions contain duplicate cells")
    source_cache = {
        case_id: _source_cells(result)
        for case_id, result in source_results.items()
    }
    records = []
    observed_keys = set()
    for result in reveal_results:
        case_id = result["case_id"]
        if case_id not in source_cache:
            raise ValueError(f"Reveal has no sealed source case: {case_id}")
        for cell in result["cells"]:
            key = (case_id, int(cell["block_index"]), float(cell["sigma"]))
            if key in observed_keys or key not in action_by_key:
                raise ValueError("Reveal cell does not uniquely match a sealed action")
            observed_keys.add(key)
            action = action_by_key[key]
            if (
                cell["route_id_sha256"] != action["route_id_sha256"]
                or cell["route_weight_sha256"] != action["route_weight_sha256"]
            ):
                raise RuntimeError("Reveal routes differ from the sealed action routes")
            source_cell = source_cache[case_id][(key[1], key[2])]
            if action["candidate_ids"] != [row["id"] for row in source_cell["records"]]:
                raise RuntimeError("Sealed candidate bank differs from source")
            if action["candidate_bank_sha256"] != candidate_bank_sha256(
                source_cell["records"]
            ):
                raise RuntimeError("Sealed candidate assignments differ from source")
            source_native_mse = float(source_cell["native_mse"])
            reveal_native_mse = float(cell["native_mse"])
            native_mse_scale = max(
                abs(source_native_mse),
                abs(reveal_native_mse),
                np.finfo(np.float64).tiny,
            )
            native_mse_relative_drift = (
                abs(source_native_mse - reveal_native_mse) / native_mse_scale
            )
            native_mse_consistent = bool(
                np.isfinite(source_native_mse)
                and np.isfinite(reveal_native_mse)
                and source_native_mse > 0
                and reveal_native_mse > 0
                and native_mse_relative_drift
                <= RETROSPECTIVE_REQUIREMENTS[
                    "maximum_source_reveal_native_mse_relative_drift"
                ]
            )
            exact_bank_gain = -np.asarray([
                row["exact_mse_change"] for row in source_cell["records"]
            ], dtype=np.float64) / source_native_mse
            selected_gain = {
                name: float(cell["action_results"][name]["selected_gain"])
                for name in ACTION_NAMES
            }
            selected_gain["rolled_utility"] = float(
                source_cell["summary"]["selectors"]["rolled_utility"]["selected_gain"]
            )
            selected_gain["exact_oracle"] = float(
                source_cell["summary"]["selectors"]["exact_oracle"]["selected_gain"]
            )
            controls = cell["numerical_controls"]
            records.append({
                "case_id": case_id,
                "block_index": key[1],
                "sigma": key[2],
                "selected_gain": selected_gain,
                "candidate_concordance": _pair_concordance(
                    np.asarray(action["primary_candidate_priority"], dtype=np.float64),
                    exact_bank_gain,
                ),
                "action_invariance_mismatch": int(
                    action["action_invariance_mismatch"]
                ),
                "logical_pass_counts_match": bool(
                    action["logical_pass_counts_match"]
                    and int(controls["logical_count_mismatches"]) == 0
                ),
                "source_native_mse": source_native_mse,
                "reveal_native_mse": reveal_native_mse,
                "source_reveal_native_mse_relative_drift": float(
                    native_mse_relative_drift
                ),
                "native_mse_consistent": native_mse_consistent,
                "numerical_controls_passed": bool(
                    native_mse_consistent
                    and _reveal_safety_passed(controls, reveal_native_mse)
                ),
                "numerical_controls": controls,
            })
    if observed_keys != set(action_by_key):
        raise ValueError("Reveal results do not cover every sealed action")
    return records


def _bootstrap_means(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap input must be a finite nontrivial vector")
    generator = np.random.default_rng(int(seed))
    means = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), 10_000):
        stop = min(start + 10_000, int(resamples))
        indices = generator.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return means


def _summary(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    boot = _bootstrap_means(values, resamples, seed)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "one_sided_lcb95": float(np.quantile(boot, 0.05)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "values": values.tolist(),
    }


def _ratio_summary(numerator, denominator, resamples, seed):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape or np.any(denominator <= 0):
        raise ValueError("Oracle-ratio vectors are invalid")
    generator = np.random.default_rng(int(seed))
    ratios = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), 10_000):
        stop = min(start + 10_000, int(resamples))
        indices = generator.integers(
            0,
            numerator.size,
            size=(stop - start, numerator.size),
        )
        ratios[start:stop] = (
            numerator[indices].mean(axis=1) / denominator[indices].mean(axis=1)
        )
    return {
        "ratio_of_means": float(numerator.mean() / denominator.mean()),
        "one_sided_lcb95": float(np.quantile(ratios, 0.05)),
        "ci95": [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))],
    }


def _one_sided_p(values):
    values = np.asarray(values, dtype=np.float64)
    if np.all(values == 0):
        return 1.0
    try:
        return float(wilcoxon(values, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def _holm(p_values, alpha=0.05):
    ordered = sorted(p_values, key=p_values.get)
    passed = {}
    active = True
    count = len(ordered)
    for rank, name in enumerate(ordered):
        threshold = alpha / (count - rank)
        active = active and p_values[name] <= threshold
        passed[name] = {
            "p_value": p_values[name],
            "holm_threshold": threshold,
            "passed": bool(active),
        }
    return passed


def aggregate_retrospective(records, requirements=None, resamples=BOOTSTRAP_RESAMPLES):
    requirements = dict(requirements or RETROSPECTIVE_REQUIREMENTS)
    if requirements != RETROSPECTIVE_REQUIREMENTS:
        raise ValueError("Retrospective requirements differ from the locked gate")
    case_ids = sorted({record["case_id"] for record in records})
    if len(case_ids) != requirements["expected_case_count"]:
        raise ValueError("Retrospective case count differs from the locked gate")
    expected_cells = {
        (block, sigma) for block in RETROSPECTIVE_BLOCKS for sigma in SIGMAS
    }
    by_case = {}
    for case_id in case_ids:
        case_records = [record for record in records if record["case_id"] == case_id]
        if {(row["block_index"], row["sigma"]) for row in case_records} != expected_cells:
            raise ValueError(f"Case {case_id} has an incomplete cell grid")
        by_case[case_id] = {
            "primary": float(np.mean([
                row["selected_gain"]["primary"] for row in case_records
            ])),
            "oracle": float(np.mean([
                row["selected_gain"]["exact_oracle"] for row in case_records
            ])),
            "concordance": float(np.mean([
                row["candidate_concordance"] for row in case_records
            ])),
            "controls": {
                name: float(np.mean([
                    row["selected_gain"][name] for row in case_records
                ]))
                for name in CONTROL_NAMES
            },
        }
    primary = np.asarray([by_case[case]["primary"] for case in case_ids])
    oracle = np.asarray([by_case[case]["oracle"] for case in case_ids])
    concordance = np.asarray([by_case[case]["concordance"] for case in case_ids])
    control_values = {
        name: np.asarray([by_case[case]["controls"][name] for case in case_ids])
        for name in CONTROL_NAMES
    }
    gain_summary = _summary(primary, resamples, BOOTSTRAP_SEED)
    concordance_summary = _summary(concordance, resamples, BOOTSTRAP_SEED + 1)
    contrasts = {
        name: _summary(
            primary - values,
            resamples,
            BOOTSTRAP_SEED + 10 + index,
        )
        for index, (name, values) in enumerate(control_values.items())
    }
    ratio = _ratio_summary(primary, oracle, resamples, BOOTSTRAP_SEED + 30)
    block_means = {
        str(block): float(np.mean([
            row["selected_gain"]["primary"]
            for row in records if row["block_index"] == block
        ]))
        for block in RETROSPECTIVE_BLOCKS
    }
    sigma_means = {
        str(sigma): float(np.mean([
            row["selected_gain"]["primary"]
            for row in records if row["sigma"] == sigma
        ]))
        for sigma in SIGMAS
    }
    p_values = {
        "primary_gain": _one_sided_p(primary),
        **{
            f"primary_vs_{name}": _one_sided_p(primary - values)
            for name, values in control_values.items()
        },
    }
    holm = _holm(p_values)
    action_mismatches = sum(row["action_invariance_mismatch"] for row in records)
    logical_mismatches = sum(not row["logical_pass_counts_match"] for row in records)
    numerical_failures = sum(not row["numerical_controls_passed"] for row in records)

    def check(observed, required, passed):
        return {"observed": observed, "required": required, "passed": bool(passed)}

    checks = {
        "mean_gain": check(
            gain_summary["mean"],
            f">={requirements['minimum_mean_gain']}",
            gain_summary["mean"] >= requirements["minimum_mean_gain"],
        ),
        "gain_lcb": check(
            gain_summary["one_sided_lcb95"],
            f">={requirements['minimum_gain_lcb']}",
            gain_summary["one_sided_lcb95"] >= requirements["minimum_gain_lcb"],
        ),
        "positive_images": check(
            int((primary > 0).sum()),
            f">={requirements['minimum_positive_images']}",
            int((primary > 0).sum()) >= requirements["minimum_positive_images"],
        ),
        "candidate_concordance": check(
            concordance_summary["mean"],
            f">={requirements['minimum_pair_concordance']}",
            concordance_summary["mean"] >= requirements["minimum_pair_concordance"],
        ),
        "candidate_concordance_lcb": check(
            concordance_summary["one_sided_lcb95"],
            f">={requirements['minimum_pair_concordance_lcb']}",
            concordance_summary["one_sided_lcb95"]
            >= requirements["minimum_pair_concordance_lcb"],
        ),
        "oracle_ratio": check(
            ratio["ratio_of_means"],
            f">={requirements['minimum_oracle_ratio']}",
            ratio["ratio_of_means"] >= requirements["minimum_oracle_ratio"],
        ),
        "oracle_ratio_lcb": check(
            ratio["one_sided_lcb95"],
            f">={requirements['minimum_oracle_ratio_lcb']}",
            ratio["one_sided_lcb95"] >= requirements["minimum_oracle_ratio_lcb"],
        ),
        "all_control_contrast_lcbs": check(
            {name: summary["one_sided_lcb95"] for name, summary in contrasts.items()},
            f">{requirements['minimum_control_contrast_lcb']}",
            all(
                summary["one_sided_lcb95"]
                > requirements["minimum_control_contrast_lcb"]
                for summary in contrasts.values()
            ),
        ),
        "positive_blocks": check(
            sum(value > 0 for value in block_means.values()),
            f"=={requirements['required_positive_blocks']}",
            sum(value > 0 for value in block_means.values())
            == requirements["required_positive_blocks"],
        ),
        "positive_sigmas": check(
            sum(value > 0 for value in sigma_means.values()),
            f"=={requirements['required_positive_sigmas']}",
            sum(value > 0 for value in sigma_means.values())
            == requirements["required_positive_sigmas"],
        ),
        "action_invariance_mismatches": check(
            action_mismatches,
            f"=={requirements['required_action_invariance_mismatches']}",
            action_mismatches == requirements["required_action_invariance_mismatches"],
        ),
        "logical_pass_count_mismatches": check(
            logical_mismatches,
            "==0",
            logical_mismatches == 0,
        ),
        "numerical_control_failures": check(
            numerical_failures,
            f"=={requirements['required_numerical_control_failures']}",
            numerical_failures == requirements["required_numerical_control_failures"],
        ),
        "holm_family": check(
            {name: value["passed"] for name, value in holm.items()},
            "all true",
            all(value["passed"] for value in holm.values()),
        ),
    }
    return {
        "requirements": requirements,
        "num_cases": len(case_ids),
        "num_cells": len(records),
        "gain": gain_summary,
        "candidate_concordance": concordance_summary,
        "oracle_ratio": ratio,
        "contrasts": contrasts,
        "block_means": block_means,
        "sigma_means": sigma_means,
        "holm": holm,
        "checks": checks,
        "per_case": by_case,
        "safety_passed": bool(
            checks["action_invariance_mismatches"]["passed"]
            and checks["logical_pass_count_mismatches"]["passed"]
            and checks["numerical_control_failures"]["passed"]
        ),
        "efficacy_passed": all(
            row["passed"]
            for name, row in checks.items()
            if name not in {
                "action_invariance_mismatches",
                "logical_pass_count_mismatches",
                "numerical_control_failures",
            }
        ),
        "passed": all(row["passed"] for row in checks.values()),
    }
