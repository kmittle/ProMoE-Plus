"""Locked case selection and image-level gate for compute exchange."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from analyses.timestep_utility.compute_exchange_probe import (
    CANDIDATE_COUNT,
    EXCHANGE_QUOTA,
    NUMERICAL_EPSILON,
    PROBE_VERSION,
    SELECTOR_NAMES,
)
from analyses.timestep_utility.cycle_batch import (
    _bootstrap_distribution,
    _bootstrap_ratio,
    _bootstrap_summary,
    sha256_file,
)


BATCH_VERSION = 1
SELECTION_SALT = "promoe-within-expert-compute-exchange-v1-20260826"
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
LOCKED_NUM_THREADS = 4
BOOTSTRAP_RESAMPLES = 200_000
BOOTSTRAP_SEEDS = {"discovery": 2026082631, "confirmatory": 2026082632}
PRIOR_MANIFEST = "analyses/timestep_utility/manifests/count_preserving_cycle_gate_v1.json"
PRIOR_MANIFEST_SHA256 = (
    "4835412d3f59966fa3231a610d16c8aaaadf28d34d4a176eb66ba55507847dd0"
)


SAFETY_REQUIREMENTS = {
    "maximum_noop_relative_mse_change": 1e-7,
    "maximum_noop_output_change": 5e-6,
    "maximum_hook_relative_mse_change": 1e-12,
    "maximum_hook_output_change": 0.0,
    "maximum_forced_unforced_relative_mse_change": 1e-7,
    "maximum_forced_unforced_output_change": 5e-6,
    "maximum_paired_native_relative_mse_drift": 1e-7,
    "maximum_paired_native_output_drift": 5e-6,
    "required_logical_count_mismatches": 0,
    "required_action_contract_mismatches": 0,
    "required_route_id_mismatches": 0,
    "required_route_weight_mismatches": 0,
}
DISCOVERY_REQUIREMENTS = {
    "minimum_mean_gain": 1e-4,
    "minimum_gain_lcb": 0.0,
    "minimum_positive_images": 16,
    "minimum_selected_positive_rate": 0.60,
    "minimum_selected_positive_lcb": 0.50,
    "minimum_pair_concordance": 0.58,
    "minimum_pair_concordance_lcb": 0.52,
    "minimum_random_contrast": 0.0,
    "minimum_random_contrast_lcb": 0.0,
    "minimum_margin_contrast": 0.0,
    "minimum_margin_contrast_lcb": 0.0,
    "minimum_oracle_positive_images": 18,
    "minimum_positive_block_strata": 1,
    "minimum_positive_sigma_strata": 3,
    "minimum_significant_block_strata": 1,
    "minimum_significant_sigma_strata": 2,
}
CONFIRMATORY_REQUIREMENTS = {
    "minimum_mean_gain": 1e-4,
    "minimum_gain_lcb": 5e-5,
    "minimum_positive_images": 32,
    "minimum_selected_positive_rate": 0.65,
    "minimum_selected_positive_lcb": 0.55,
    "maximum_harm_rate_ucb": 0.35,
    "minimum_pair_concordance": 0.60,
    "minimum_pair_concordance_lcb": 0.55,
    "minimum_random_contrast": 1e-5,
    "minimum_random_contrast_lcb": 0.0,
    "minimum_margin_contrast": 1e-5,
    "minimum_margin_contrast_lcb": 0.0,
    "minimum_rolled_contrast": 1e-5,
    "minimum_rolled_contrast_lcb": 0.0,
    "minimum_oracle_ratio": 0.25,
    "minimum_oracle_ratio_lcb": 0.15,
    "minimum_positive_block_strata": 3,
    "minimum_positive_sigma_strata": 3,
    "minimum_significant_block_strata": 2,
    "minimum_significant_sigma_strata": 2,
}


def requirements_for_split(split):
    if split == "plumbing":
        return {**SAFETY_REQUIREMENTS, "expected_case_count": SPLIT_COUNTS[split]}
    if split == "discovery":
        efficacy = DISCOVERY_REQUIREMENTS
    elif split == "confirmatory":
        efficacy = CONFIRMATORY_REQUIREMENTS
    else:
        raise ValueError(f"Unknown split: {split}")
    return {
        **SAFETY_REQUIREMENTS,
        **efficacy,
        "expected_case_count": SPLIT_COUNTS[split],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEEDS[split],
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
        "latent_relative": f"{class_dir.name}/{latent.name}",
        "latent_sha256": sha256_file(latent),
    }


def select_locked_cases(latent_root, project_root):
    latent_root = Path(latent_root).resolve()
    project_root = Path(project_root).resolve()
    if not latent_root.is_dir():
        raise NotADirectoryError(f"Latent root does not exist: {latent_root}")
    prior_path = (project_root / PRIOR_MANIFEST).resolve()
    if not prior_path.is_file() or sha256_file(prior_path) != PRIOR_MANIFEST_SHA256:
        raise RuntimeError("Prior cycle manifest provenance changed")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    excluded = set(prior["selection"]["excluded_labels"])
    excluded.update(int(case["label"]) for case in prior["cases"])

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
    by_synset = {path.name: path for path in class_dirs}

    plumbing = []
    for raw_case in prior["cases"]:
        if raw_case["split"] != "plumbing":
            continue
        latent = latent_root / raw_case["latent"]
        if not latent.is_file() or sha256_file(latent) != raw_case["latent_sha256"]:
            raise RuntimeError("Locked plumbing latent differs from the prior manifest")
        plumbing.append({
            "split": "plumbing",
            "id": raw_case["id"],
            "label": int(raw_case["label"]),
            "seed": int(raw_case["seed"]),
            "synset": raw_case["synset"],
            "latent_relative": raw_case["latent"],
            "latent_sha256": raw_case["latent_sha256"],
        })
    if len(plumbing) != SPLIT_COUNTS["plumbing"]:
        raise RuntimeError("Prior manifest does not provide eight plumbing cases")

    ranked = sorted(
        (
            hashlib.sha256(
                f"{SELECTION_SALT}|class|{label:03d}|{path.name}".encode()
            ).hexdigest(),
            label,
            path,
        )
        for label, path in enumerate(class_dirs)
        if label not in excluded
    )
    if len(ranked) < SPLIT_COUNTS["discovery"] + SPLIT_COUNTS["confirmatory"]:
        raise RuntimeError("Too few unseen classes remain for the locked gate")
    statistical = []
    for index, (_, label, class_dir) in enumerate(ranked[:72]):
        split = "discovery" if index < 24 else "confirmatory"
        statistical.append(_selected_case(split, label, class_dir))
    cases = plumbing + statistical
    for case in cases:
        if case["synset"] not in by_synset:
            raise RuntimeError("Locked case synset is absent from the latent root")
        case["latent"] = str(latent_root / case["latent_relative"])
    if len({case["id"] for case in cases}) != len(cases):
        raise RuntimeError("Locked compute-exchange case IDs must be unique")
    return {
        "selection": {
            "salt": SELECTION_SALT,
            "prior_manifest": PRIOR_MANIFEST,
            "prior_manifest_sha256": PRIOR_MANIFEST_SHA256,
            "excluded_labels": sorted(excluded),
            "statistical_rule": (
                "Exclude every prior-manifest label, rank remaining classes by "
                "SHA256(salt|class|label|synset), and allocate 24 discovery plus "
                "48 confirmatory classes before any efficacy observation."
            ),
            "plumbing_rule": (
                "Reuse the eight previously observed cycle plumbing images for "
                "safety only; no compute-exchange efficacy is persisted."
            ),
        },
        "latent_root": str(latent_root),
        "cases": cases,
    }


def _case_metrics(result, split):
    cells = result.get("cells", [])
    expected_cells = len(BLOCKS_BY_SPLIT[split]) * len(SIGMAS)
    if len(cells) != expected_cells:
        raise ValueError("Case cell count does not match the locked split")
    observed = {(int(cell["block_index"]), float(cell["sigma"])) for cell in cells}
    expected = {
        (block, sigma)
        for block in BLOCKS_BY_SPLIT[split]
        for sigma in SIGMAS
    }
    if observed != expected:
        raise ValueError("Case block/sigma cells do not match the locked split")

    safety = {
        "noop_relative_mse_change": 0.0,
        "noop_output_change": 0.0,
        "hook_relative_mse_change": 0.0,
        "hook_output_change": 0.0,
        "forced_unforced_relative_mse_change": 0.0,
        "forced_unforced_output_change": 0.0,
        "paired_native_relative_mse_drift": 0.0,
        "paired_native_output_drift": 0.0,
        "logical_count_mismatches": 0,
        "action_contract_mismatches": 0,
        "route_id_mismatches": 0,
        "route_weight_mismatches": 0,
    }
    for cell in cells:
        controls = cell["numerical_controls"]
        native_mse = float(cell["native_mse"])
        for target, source in (
            ("noop_output_change", "max_abs_noop_output_change"),
            ("hook_output_change", "max_abs_hook_output_change"),
            ("forced_unforced_output_change", "max_abs_forced_unforced_output_change"),
            ("paired_native_output_drift", "max_abs_paired_native_output_drift"),
        ):
            safety[target] = max(safety[target], float(controls[source]))
        for target, source in (
            ("noop_relative_mse_change", "max_abs_noop_mse_change"),
            ("hook_relative_mse_change", "max_abs_hook_mse_change"),
            ("forced_unforced_relative_mse_change", "max_abs_forced_unforced_mse_change"),
            ("paired_native_relative_mse_drift", "max_abs_paired_native_mse_drift"),
        ):
            safety[target] = max(
                safety[target],
                float(controls[source]) / native_mse,
            )
        for name in (
            "logical_count_mismatches",
            "action_contract_mismatches",
            "route_id_mismatches",
            "route_weight_mismatches",
        ):
            safety[name] += int(controls[name])

    metric = {
        "case_id": result["batch_case"]["id"],
        "safety": safety,
    }
    if split == "plumbing":
        if not all(cell.get("efficacy_statistics_withheld") for cell in cells):
            raise ValueError("Plumbing result exposed compute-exchange efficacy")
        return metric

    spearman_values = [cell["summary"]["spearman"] for cell in cells]
    spearman_defined = all(
        value is not None and np.isfinite(value)
        for value in spearman_values
    )
    metric.update({
        "selectors": {},
        "pair_concordance": float(np.mean([
            cell["summary"]["pair_concordance"] for cell in cells
        ])),
        "spearman": (
            float(np.mean(spearman_values)) if spearman_defined else None
        ),
        "oracle_positive": bool(np.mean([
            cell["summary"]["oracle_gain"] for cell in cells
        ]) > NUMERICAL_EPSILON),
        "oracle_gain": float(np.mean([
            cell["summary"]["oracle_gain"] for cell in cells
        ])),
        "per_block_first_order_gain": {
            str(block): float(np.mean([
                cell["summary"]["selectors"]["first_order"]["selected_gain"]
                for cell in cells if cell["block_index"] == block
            ]))
            for block in BLOCKS_BY_SPLIT[split]
        },
        "per_sigma_first_order_gain": {
            str(sigma): float(np.mean([
                cell["summary"]["selectors"]["first_order"]["selected_gain"]
                for cell in cells if cell["sigma"] == sigma
            ]))
            for sigma in SIGMAS
        },
    })
    for selector in SELECTOR_NAMES:
        summaries = [cell["summary"]["selectors"][selector] for cell in cells]
        metric["selectors"][selector] = {
            "selected_gain": float(np.mean([
                summary["selected_gain"] for summary in summaries
            ])),
            "selected_per_transferred_pass_gain": float(np.mean([
                summary["selected_per_transferred_pass_gain"]
                for summary in summaries
            ])),
            "selected_positive_rate": float(np.mean([
                summary["selected_positive"] for summary in summaries
            ])),
            "selected_harm_rate": float(np.mean([
                summary["selected_harm"] for summary in summaries
            ])),
        }
    metric["first_order_positive"] = bool(
        metric["selectors"]["first_order"]["selected_gain"] > NUMERICAL_EPSILON
    )
    metric["first_order_oracle_numerator"] = metric[
        "selectors"
    ]["first_order"]["selected_gain"]
    return metric


def _check(observed, required, passed):
    return {"observed": observed, "required": required, "passed": bool(passed)}


def _safety_gate(metrics, requirements):
    threshold_map = {
        "noop_relative_mse_change": "maximum_noop_relative_mse_change",
        "noop_output_change": "maximum_noop_output_change",
        "hook_relative_mse_change": "maximum_hook_relative_mse_change",
        "hook_output_change": "maximum_hook_output_change",
        "forced_unforced_relative_mse_change": "maximum_forced_unforced_relative_mse_change",
        "forced_unforced_output_change": "maximum_forced_unforced_output_change",
        "paired_native_relative_mse_drift": "maximum_paired_native_relative_mse_drift",
        "paired_native_output_drift": "maximum_paired_native_output_drift",
    }
    checks = {}
    for metric_name, requirement_name in threshold_map.items():
        observed = max(metric["safety"][metric_name] for metric in metrics)
        required = requirements[requirement_name]
        checks[metric_name] = _check(observed, f"<={required}", observed <= required)
    for name in (
        "logical_count_mismatches",
        "action_contract_mismatches",
        "route_id_mismatches",
        "route_weight_mismatches",
    ):
        observed = sum(metric["safety"][name] for metric in metrics)
        required = requirements[f"required_{name}"]
        checks[name] = _check(observed, f"=={required}", observed == required)
    return checks, bool(all(check["passed"] for check in checks.values()))


def _bootstrap_upper(values, resamples, seed):
    distribution = _bootstrap_distribution(values, resamples, seed)
    return float(np.quantile(distribution, 0.95))


def _one_sided_p(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("One-sided test values must be a finite vector")
    if np.all(values == 0):
        return 1.0
    return float(wilcoxon(values, alternative="greater").pvalue)


def _holm(p_values, alpha=0.05):
    ordered = sorted(p_values, key=p_values.get)
    rejected = {}
    still_rejecting = True
    for index, name in enumerate(ordered):
        threshold = alpha / (len(ordered) - index)
        passed = bool(still_rejecting and p_values[name] <= threshold)
        rejected[name] = {
            "p_value": float(p_values[name]),
            "threshold": float(threshold),
            "passed": passed,
        }
        if not passed:
            still_rejecting = False
    return {name: rejected[name] for name in p_values}


def _stratum_gate(metrics, key, requirements, resamples, seed):
    strata = tuple(metrics[0][key])
    if any(tuple(metric[key]) != strata for metric in metrics):
        raise ValueError("Image-level stratum keys are inconsistent")
    summaries = {}
    for offset, stratum in enumerate(strata):
        summaries[stratum] = _bootstrap_summary(
            [metric[key][stratum] for metric in metrics],
            resamples,
            seed + offset,
        )
    positive = sum(summary["mean"] > 0 for summary in summaries.values())
    significant = sum(
        summary["one_sided_lcb95"] > 0 for summary in summaries.values()
    )
    prefix = "block" if "block" in key else "sigma"
    checks = {
        f"positive_{prefix}_strata": _check(
            positive,
            f">={requirements[f'minimum_positive_{prefix}_strata']}",
            positive >= requirements[f"minimum_positive_{prefix}_strata"],
        ),
        f"significant_{prefix}_strata": _check(
            significant,
            f">={requirements[f'minimum_significant_{prefix}_strata']}",
            significant >= requirements[f"minimum_significant_{prefix}_strata"],
        ),
    }
    return summaries, checks


def aggregate_case_results(case_results, split, requirements=None):
    if split not in SPLIT_COUNTS:
        raise ValueError(f"Unknown split: {split}")
    requirements = requirements or requirements_for_split(split)
    if len(case_results) != requirements["expected_case_count"]:
        raise ValueError("Completed case count does not match the locked split")
    metrics = [_case_metrics(result, split) for result in case_results]
    if len({metric["case_id"] for metric in metrics}) != len(metrics):
        raise ValueError("Case results contain duplicate IDs")
    safety_checks, safety_passed = _safety_gate(metrics, requirements)
    gate = {
        "split": split,
        "case_count": len(metrics),
        "safety_checks": safety_checks,
        "safety_passed": safety_passed,
    }
    if split == "plumbing":
        gate.update({
            "efficacy_statistics_withheld": True,
            "passed": safety_passed,
        })
        return gate

    resamples = requirements["bootstrap_resamples"]
    seed = requirements["bootstrap_seed"]
    fo = np.asarray([
        metric["selectors"]["first_order"]["selected_gain"]
        for metric in metrics
    ])
    fo_per_pass = np.asarray([
        metric["selectors"]["first_order"]["selected_per_transferred_pass_gain"]
        for metric in metrics
    ])
    random_per_pass = np.asarray([
        metric["selectors"]["random"]["selected_per_transferred_pass_gain"]
        for metric in metrics
    ])
    margin_per_pass = np.asarray([
        metric["selectors"]["router_margin"]["selected_per_transferred_pass_gain"]
        for metric in metrics
    ])
    rolled_per_pass = np.asarray([
        metric["selectors"]["rolled_utility"]["selected_per_transferred_pass_gain"]
        for metric in metrics
    ])
    positive_rate = np.asarray([
        metric["selectors"]["first_order"]["selected_positive_rate"]
        for metric in metrics
    ])
    harm_rate = np.asarray([
        metric["selectors"]["first_order"]["selected_harm_rate"]
        for metric in metrics
    ])
    concordance = np.asarray([metric["pair_concordance"] for metric in metrics])
    summaries = {
        "first_order_gain": _bootstrap_summary(fo, resamples, seed),
        "selected_positive_rate": _bootstrap_summary(positive_rate, resamples, seed + 1),
        "selected_harm_rate": _bootstrap_summary(harm_rate, resamples, seed + 2),
        "pair_concordance": _bootstrap_summary(concordance, resamples, seed + 3),
        "random_per_pass_contrast": _bootstrap_summary(
            fo_per_pass - random_per_pass, resamples, seed + 4
        ),
        "margin_per_pass_contrast": _bootstrap_summary(
            fo_per_pass - margin_per_pass, resamples, seed + 5
        ),
        "rolled_per_pass_contrast": _bootstrap_summary(
            fo_per_pass - rolled_per_pass, resamples, seed + 6
        ),
    }
    positive_images = int(sum(metric["first_order_positive"] for metric in metrics))
    oracle_positive_images = int(sum(metric["oracle_positive"] for metric in metrics))
    spearman_defined_images = int(sum(
        metric["spearman"] is not None for metric in metrics
    ))
    checks = {
        "mean_gain": _check(
            summaries["first_order_gain"]["mean"],
            f">={requirements['minimum_mean_gain']}",
            summaries["first_order_gain"]["mean"] >= requirements["minimum_mean_gain"],
        ),
        "gain_lcb": _check(
            summaries["first_order_gain"]["one_sided_lcb95"],
            f">{requirements['minimum_gain_lcb']}",
            summaries["first_order_gain"]["one_sided_lcb95"]
            > requirements["minimum_gain_lcb"],
        ),
        "positive_images": _check(
            positive_images,
            f">={requirements['minimum_positive_images']}",
            positive_images >= requirements["minimum_positive_images"],
        ),
        "selected_positive_rate": _check(
            summaries["selected_positive_rate"]["mean"],
            f">={requirements['minimum_selected_positive_rate']}",
            summaries["selected_positive_rate"]["mean"]
            >= requirements["minimum_selected_positive_rate"],
        ),
        "selected_positive_lcb": _check(
            summaries["selected_positive_rate"]["one_sided_lcb95"],
            f">{requirements['minimum_selected_positive_lcb']}",
            summaries["selected_positive_rate"]["one_sided_lcb95"]
            > requirements["minimum_selected_positive_lcb"],
        ),
        "pair_concordance": _check(
            summaries["pair_concordance"]["mean"],
            f">={requirements['minimum_pair_concordance']}",
            summaries["pair_concordance"]["mean"]
            >= requirements["minimum_pair_concordance"],
        ),
        "pair_concordance_lcb": _check(
            summaries["pair_concordance"]["one_sided_lcb95"],
            f">{requirements['minimum_pair_concordance_lcb']}",
            summaries["pair_concordance"]["one_sided_lcb95"]
            > requirements["minimum_pair_concordance_lcb"],
        ),
        "spearman_defined": _check(
            spearman_defined_images,
            f"=={len(metrics)}",
            spearman_defined_images == len(metrics),
        ),
        "random_contrast": _check(
            summaries["random_per_pass_contrast"]["mean"],
            f">={requirements['minimum_random_contrast']}",
            summaries["random_per_pass_contrast"]["mean"]
            >= requirements["minimum_random_contrast"],
        ),
        "random_contrast_lcb": _check(
            summaries["random_per_pass_contrast"]["one_sided_lcb95"],
            f">{requirements['minimum_random_contrast_lcb']}",
            summaries["random_per_pass_contrast"]["one_sided_lcb95"]
            > requirements["minimum_random_contrast_lcb"],
        ),
        "margin_contrast": _check(
            summaries["margin_per_pass_contrast"]["mean"],
            f">={requirements['minimum_margin_contrast']}",
            summaries["margin_per_pass_contrast"]["mean"]
            >= requirements["minimum_margin_contrast"],
        ),
        "margin_contrast_lcb": _check(
            summaries["margin_per_pass_contrast"]["one_sided_lcb95"],
            f">{requirements['minimum_margin_contrast_lcb']}",
            summaries["margin_per_pass_contrast"]["one_sided_lcb95"]
            > requirements["minimum_margin_contrast_lcb"],
        ),
    }
    if split == "discovery":
        checks["oracle_positive_images"] = _check(
            oracle_positive_images,
            f">={requirements['minimum_oracle_positive_images']}",
            oracle_positive_images >= requirements["minimum_oracle_positive_images"],
        )
    else:
        summaries["oracle_ratio"] = _bootstrap_ratio(
            np.asarray([
                metric["first_order_oracle_numerator"] for metric in metrics
            ]),
            np.asarray([metric["oracle_gain"] for metric in metrics]),
            resamples,
            seed + 7,
        )
        checks.update({
            "harm_rate_ucb": _check(
                _bootstrap_upper(harm_rate, resamples, seed + 8),
                f"<{requirements['maximum_harm_rate_ucb']}",
                _bootstrap_upper(harm_rate, resamples, seed + 8)
                < requirements["maximum_harm_rate_ucb"],
            ),
            "rolled_contrast": _check(
                summaries["rolled_per_pass_contrast"]["mean"],
                f">={requirements['minimum_rolled_contrast']}",
                summaries["rolled_per_pass_contrast"]["mean"]
                >= requirements["minimum_rolled_contrast"],
            ),
            "rolled_contrast_lcb": _check(
                summaries["rolled_per_pass_contrast"]["one_sided_lcb95"],
                f">{requirements['minimum_rolled_contrast_lcb']}",
                summaries["rolled_per_pass_contrast"]["one_sided_lcb95"]
                > requirements["minimum_rolled_contrast_lcb"],
            ),
            "oracle_ratio": _check(
                summaries["oracle_ratio"]["ratio"],
                f">={requirements['minimum_oracle_ratio']}",
                summaries["oracle_ratio"]["ratio"] is not None
                and summaries["oracle_ratio"]["ratio"]
                >= requirements["minimum_oracle_ratio"],
            ),
            "oracle_ratio_lcb": _check(
                summaries["oracle_ratio"]["one_sided_lcb95"],
                f">{requirements['minimum_oracle_ratio_lcb']}",
                summaries["oracle_ratio"]["one_sided_lcb95"] is not None
                and summaries["oracle_ratio"]["one_sided_lcb95"]
                > requirements["minimum_oracle_ratio_lcb"],
            ),
        })

    block_summaries, block_checks = _stratum_gate(
        metrics,
        "per_block_first_order_gain",
        requirements,
        resamples,
        seed + 100,
    )
    sigma_summaries, sigma_checks = _stratum_gate(
        metrics,
        "per_sigma_first_order_gain",
        requirements,
        resamples,
        seed + 200,
    )
    checks.update(block_checks)
    checks.update(sigma_checks)
    p_values = {
        "first_order_above_zero": _one_sided_p(fo),
        "first_order_above_random": _one_sided_p(
            fo_per_pass - random_per_pass
        ),
        "first_order_above_margin": _one_sided_p(
            fo_per_pass - margin_per_pass
        ),
    }
    if split == "confirmatory":
        p_values["first_order_above_rolled"] = _one_sided_p(
            fo_per_pass - rolled_per_pass
        )
    holm = _holm(p_values)
    checks["holm_primary"] = _check(
        {name: result["p_value"] for name, result in holm.items()},
        "all Holm-Bonferroni one-sided tests pass at alpha=0.05",
        all(result["passed"] for result in holm.values()),
    )
    efficacy_passed = bool(all(check["passed"] for check in checks.values()))
    gate.update({
        "requirements": requirements,
        "summaries": summaries,
        "block_summaries": block_summaries,
        "sigma_summaries": sigma_summaries,
        "positive_images": positive_images,
        "oracle_positive_images": oracle_positive_images,
        "spearman_defined_images": spearman_defined_images,
        "holm_primary": holm,
        "checks": checks,
        "efficacy_passed": efficacy_passed,
        "passed": bool(safety_passed and efficacy_passed),
    })
    return gate
