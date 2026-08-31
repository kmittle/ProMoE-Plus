"""Sealed, resume-safe runner for the RCL-responsibility mechanism gate."""

from __future__ import annotations

import gc
import multiprocessing
import os
import queue
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch

from analyses.finite_horizon_routing.runner import (
    LOCKED_BRANCH,
    LOCKED_DEVICES,
    MODEL_NAME,
    SOURCE_CASE_MANIFEST,
    SOURCE_CASE_MANIFEST_SHA256,
    _analysis_runtime_environment,
    _arm_parent_death_signal,
    _checkpoint_contract as _fresh_checkpoint_contract,
    _exclusive_lock,
    _file_identity,
    _git_contract,
    _json_sha256,
    _main_worktree_root,
    _publish_protocol,
    _read_json,
    _read_sealed_json,
    _sealed_payload,
    _write_sealed_json,
    sha256_file,
)
from analyses.fresh_base_routing.audit import (
    _checkpoint_path,
    _validate_run_dir,
    load_manifest as load_source_manifest,
)
from analyses.t_SNE.checkpoint_utils import load_runtime_cfg

from .batch import (
    BATCH_VERSION,
    CONFIRMATORY_REQUIREMENTS,
    DISCOVERY_REQUIREMENTS,
    SAFETY_REQUIREMENTS,
    SPLIT_COUNTS,
    aggregate_case_results,
    requirements_for_split,
)
from .probe import (
    ONLINE_CHECKPOINT_STATE,
    aggregate_rank_support_rcl,
    build_rank_local_support_rcl,
    load_rcl_responsibility_probe_model,
    run_rcl_responsibility_query,
)
from .protocol import (
    ASSIGNMENT_SHUFFLE_COUNT,
    BLOCK_INDICES,
    CANDIDATE_SCALES,
    CENTER_HALF_STEP_MULTIPLIER,
    CENTER_STEP_RELATIVE_FROBENIUS,
    EXACT_BATCH_SIZE,
    PROBE_VERSION,
    SIGMA_VALUES,
    SUPPORT_BATCH_SIZE,
    SUPPORT_FORWARD_BATCH_SIZE,
    SUPPORT_GROUP_COUNT,
    SUPPORT_SELECTION_SALT,
    SUPPORT_SIGMA_POLICY,
    TOKEN_PROBE_COUNT,
)
from .support import SUPPORT_UNCONDITIONAL_COUNT, select_support_cases


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_STEM = "004_ProMoE_B_fresh_routing_audit_s0"
FRESH_CONFIG_SHA256 = (
    "97fe9376303cc390eada34e2bc82fa903b998b78c82d181486630a25187c0ab6"
)
FRESH_TRAINING_CONFIG_SHA256 = (
    "c11983626dd8e65cf6074be4792c3f37a662acb01561537baca968a7db2ccca9"
)
FRESH_TRAINING_COMMIT = "257d51af287ea93103d7b4cad5ecab9dc1e3b541"
GATE_MANIFEST = (
    PROJECT_ROOT
    / "analyses"
    / "affinity_responsibility"
    / "manifests"
    / "rcl_responsibility_gate_v1.json"
)
CHECKPOINT_STEP = 300_000
PROTOCOL_FILENAME = "protocol.json"
PROTOCOL_SEAL_FILENAME = "protocol.sha256"
RESULT_SHA256_FIELD = "result_sha256"
SUMMARY_SHA256_FIELD = "summary_sha256"
GLOBAL_RUN_LOCK = Path("/tmp/promoe-rcl-responsibility-cuda-0-3.lock")
OUTPUT_RUN_LOCK_FILENAME = ".run-gate.lock"
WORKER_RESULT_POLL_SECONDS = 30
WORKER_SUPPORT_TIMEOUT_SECONDS = 2 * 60 * 60
WORKER_SPLIT_TIMEOUT_SECONDS = 24 * 60 * 60
WORKER_JOIN_TIMEOUT_SECONDS = 30
WORKER_TERMINATE_TIMEOUT_SECONDS = 10
WORKER_KILL_TIMEOUT_SECONDS = 10
SPLIT_ORDER = ("plumbing", "discovery", "confirmatory")
SPLIT_PREREQUISITES = {
    "plumbing": (),
    "discovery": ("plumbing",),
    "confirmatory": ("plumbing", "discovery"),
}
STATIC_SOURCE_PATHS = (
    "requirements.txt",
    "config.py",
    "train.py",
    "models/models_ProMoE_TC.py",
    "models/modules.py",
    "models/phase_metric.py",
    "analyses/denoising_regret/io.py",
    "analyses/denoising_regret/probe.py",
    "analyses/denoising_regret/responsibility_probe.py",
    "analyses/finite_horizon_routing/probe.py",
    "analyses/finite_horizon_routing/runner.py",
    "analyses/fresh_base_routing/audit.py",
    "analyses/t_SNE/checkpoint_utils.py",
    "analyses/affinity_responsibility/__init__.py",
    "analyses/affinity_responsibility/protocol.py",
    "analyses/affinity_responsibility/support.py",
    "analyses/affinity_responsibility/probe.py",
    "analyses/affinity_responsibility/batch.py",
    "analyses/affinity_responsibility/runner.py",
    "analyses/affinity_responsibility/manifests/rcl_responsibility_gate_v1.json",
    "analyses/run_rcl_responsibility_probe_batch.py",
)


def _strict_schema_equal(observed, expected):
    """Compare JSON values without coercing bools, ints, or floats."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        if set(observed) != set(expected):
            return False
        return all(
            _strict_schema_equal(observed[key], expected[key])
            for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _strict_schema_equal(observed_item, expected_item)
            for observed_item, expected_item in zip(observed, expected)
        )
    return observed == expected


def _canonical_gate_manifest():
    payload = _read_json(GATE_MANIFEST)
    expected_protocol = {
        "assignment_shuffle_count": ASSIGNMENT_SHUFFLE_COUNT,
        "block_indices": list(BLOCK_INDICES),
        "candidate_scales": list(CANDIDATE_SCALES),
        "center_half_step_multiplier": CENTER_HALF_STEP_MULTIPLIER,
        "center_step_relative_frobenius": CENTER_STEP_RELATIVE_FROBENIUS,
        "exact_batch_size": EXACT_BATCH_SIZE,
        "sigma_values": list(SIGMA_VALUES),
        "support_batch_size": SUPPORT_BATCH_SIZE,
        "support_forward_batch_size": SUPPORT_FORWARD_BATCH_SIZE,
        "support_global_batch_size": SUPPORT_BATCH_SIZE * SUPPORT_GROUP_COUNT,
        "support_gradient_aggregation": "ddp_mean",
        "support_group_count": SUPPORT_GROUP_COUNT,
        "support_selection_salt": SUPPORT_SELECTION_SALT,
        "support_sigma_policy": SUPPORT_SIGMA_POLICY,
        "support_unconditional_count": SUPPORT_UNCONDITIONAL_COUNT,
        "token_probe_count": TOKEN_PROBE_COUNT,
    }
    expected = {
        "locked_before_any_fresh_rcl_responsibility_result": True,
        "name": "rcl-responsibility-gate-v1",
        "prior_observation": {
            "fresh_result_seen": False,
            "rcl_gradient_result_seen": False,
            "statement": (
                "Dirty 50K scale interventions motivated the responsibility "
                "thresholds, but no Fresh 300K result and no RCL-gradient result "
                "were observed before this revised DDP-faithful manifest was locked."
            ),
        },
        "protocol": expected_protocol,
        "safety_requirements": SAFETY_REQUIREMENTS,
        "source_case_manifest": {
            "path": (
                "analyses/fresh_base_routing/manifests/"
                "fresh_base_routing_audit_v1.json"
            ),
            "sha256": SOURCE_CASE_MANIFEST_SHA256,
        },
        "splits": {
            "confirmatory": {
                "expected_case_count": SPLIT_COUNTS["confirmatory"],
                "requirements": CONFIRMATORY_REQUIREMENTS,
            },
            "discovery": {
                "expected_case_count": SPLIT_COUNTS["discovery"],
                "requirements": DISCOVERY_REQUIREMENTS,
            },
            "plumbing": {
                "efficacy_statistics_withheld": True,
                "expected_case_count": SPLIT_COUNTS["plumbing"],
            },
        },
        "version": 1,
    }
    if not _strict_schema_equal(payload, expected):
        raise ValueError("RCL-responsibility gate manifest differs from canonical schema")
    return payload


def _source_hashes(config_path):
    paths = list(STATIC_SOURCE_PATHS) + [
        str(Path(config_path).resolve().relative_to(PROJECT_ROOT.resolve()))
    ]
    hashes = {}
    for relative in sorted(set(paths)):
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Locked source is missing: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def _default_output_dir(checkpoint_path):
    return (
        Path(checkpoint_path).resolve().parent.parent
        / "sample"
        / f"step{CHECKPOINT_STEP}"
        / "rcl_responsibility_gate_v1"
    )


def _case_record(case):
    return {
        key: case[key]
        for key in (
            "split",
            "id",
            "label",
            "seed",
            "synset",
            "latent_relative",
            "latent",
            "latent_key",
            "latent_size",
            "latent_sha256",
        )
    }


def _support_record(case):
    return {
        key: case[key]
        for key in (
            "id",
            "selection_rank",
            "group_index",
            "label",
            "synset",
            "latent_relative",
            "latent",
            "latent_key",
            "latent_size",
            "latent_sha256",
            "seed",
            "sigma_seed",
            "sigma",
            "unconditional",
        )
    }


def _support_group_digest(cases):
    return _json_sha256([_support_record(case) for case in cases])


def _build_protocol_payload(
    checkpoint_path,
    latent_root,
    devices=LOCKED_DEVICES,
    output_dir=None,
):
    devices = tuple(devices)
    if devices != LOCKED_DEVICES or len(devices) != SUPPORT_GROUP_COUNT:
        raise ValueError(f"Locked devices are {LOCKED_DEVICES}")
    manifest = _canonical_gate_manifest()
    if sha256_file(SOURCE_CASE_MANIFEST) != SOURCE_CASE_MANIFEST_SHA256:
        raise ValueError("Fresh case manifest SHA256 changed")
    git = _git_contract(locked_branch=LOCKED_BRANCH)
    (
        checkpoint_path,
        config_path,
        runtime_cfg,
        model_metadata,
        checkpoint_record,
        fresh_run,
    ) = _fresh_checkpoint_contract(
        checkpoint_path,
        latent_root,
        config_stem=CONFIG_STEM,
        config_sha256=FRESH_CONFIG_SHA256,
        training_config_sha256=FRESH_TRAINING_CONFIG_SHA256,
        training_commit=FRESH_TRAINING_COMMIT,
    )
    checkpoint_record = dict(checkpoint_record)
    checkpoint_record["state"] = ONLINE_CHECKPOINT_STATE
    source_manifest = load_source_manifest(SOURCE_CASE_MANIFEST, latent_root)
    cases = []
    for case in source_manifest["cases"]:
        identity = _file_identity(
            case["latent"],
            f"Query latent {case['id']}",
            expected_sha256=case["latent_sha256"],
        )
        cases.append({
            **{key: case[key] for key in (
                "split",
                "id",
                "label",
                "seed",
                "synset",
                "latent_relative",
                "latent",
                "latent_key",
            )},
            "latent_size": identity["size"],
            "latent_sha256": identity["sha256"],
        })
    support_cases = select_support_cases(
        latent_root,
        excluded_labels={case["label"] for case in cases},
    )
    for index, case in enumerate(support_cases):
        identity = _file_identity(
            case["latent"],
            f"Support latent {index}",
        )
        case.update({
            "id": f"support-{index:03d}",
            "latent_size": identity["size"],
            "latent_sha256": identity["sha256"],
        })
    query_labels = {case["label"] for case in cases}
    if query_labels & {case["label"] for case in support_cases}:
        raise RuntimeError("Support and query labels overlap")
    support_groups = {
        str(group): [
            _support_record(case)
            for case in support_cases
            if int(case["group_index"]) == group
        ]
        for group in range(SUPPORT_GROUP_COUNT)
    }
    for group, group_cases in support_groups.items():
        if len(group_cases) != SUPPORT_BATCH_SIZE:
            raise RuntimeError(f"Support group {group} is not a local batch")
        if sum(item["unconditional"] for item in group_cases) != (
            SUPPORT_UNCONDITIONAL_COUNT
        ):
            raise RuntimeError(f"Support group {group} has wrong uncond count")
    assignments = {}
    for split in SPLIT_COUNTS:
        split_cases = [case for case in cases if case["split"] == split]
        assignments[split] = [
            {
                "case_id": case["id"],
                "device": devices[index % len(devices)],
            }
            for index, case in enumerate(split_cases)
        ]
    output_dir = Path(
        output_dir or _default_output_dir(checkpoint_path)
    ).resolve()
    return {
        "batch_version": BATCH_VERSION,
        "probe_version": PROBE_VERSION,
        "locked_before_any_fresh_rcl_responsibility_result": True,
        "git": git,
        "worktrees": {
            "analysis": str(PROJECT_ROOT.resolve(strict=True)),
            "training": fresh_run["main_worktree_root"],
            "training_outputs": fresh_run["main_output_root"],
        },
        "gate_manifest": {
            "path": str(GATE_MANIFEST),
            "sha256": sha256_file(GATE_MANIFEST),
            "payload": manifest,
        },
        "source_case_manifest": source_manifest,
        "checkpoint": checkpoint_record,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "stem": config_path.stem,
        },
        "model": model_metadata,
        "protocol": manifest["protocol"],
        "requirements": {
            split: requirements_for_split(split) for split in SPLIT_COUNTS
        },
        "devices": list(devices),
        "environment": _analysis_runtime_environment(devices),
        "assignments": assignments,
        "cases": [_case_record(case) for case in cases],
        "support_groups": support_groups,
        "support_group_sha256": {
            group: _support_group_digest(group_cases)
            for group, group_cases in support_groups.items()
        },
        "support_world_sha256": _json_sha256(support_groups),
        "fresh_run": fresh_run,
        "source_hashes": _source_hashes(config_path),
        "output_dir": str(output_dir),
        "scope_boundary": (
            "The gate isolates only RCL's direct prototype gradient from one "
            "training-sized four-rank support batch with DDP-mean gradients. It "
            "does not attribute the "
            "separate RCL gradient through hidden states, and it is not a training, "
            "sampling, FID, or publication claim."
        ),
    }


def prepare_protocol(checkpoint_path, latent_root, devices=LOCKED_DEVICES, output_dir=None):
    protocol = _sealed_payload(
        _build_protocol_payload(
            checkpoint_path=checkpoint_path,
            latent_root=latent_root,
            devices=devices,
            output_dir=output_dir,
        ),
        "protocol_sha256",
    )
    _publish_protocol(Path(protocol["output_dir"]), protocol)
    return protocol


def _rebuild_protocol_payload(output_dir):
    training_root = _main_worktree_root()
    output_root = training_root / "outputs"
    run_dir = _validate_run_dir(
        output_root / MODEL_NAME / CONFIG_STEM,
        output_root=output_root,
        expected_config_stem=CONFIG_STEM,
    )
    checkpoint_path = _checkpoint_path(
        run_dir,
        CHECKPOINT_STEP,
        output_root=output_root,
        expected_config_stem=CONFIG_STEM,
    )
    config_path = PROJECT_ROOT / "configs" / f"{CONFIG_STEM}.yaml"
    runtime_cfg = load_runtime_cfg(config_path)
    latent_root = Path(runtime_cfg.latent_data_path).resolve(strict=True)
    return _build_protocol_payload(
        checkpoint_path=checkpoint_path,
        latent_root=latent_root,
        devices=LOCKED_DEVICES,
        output_dir=output_dir,
    )


def verify_protocol(output_dir):
    output_dir = Path(output_dir).resolve()
    protocol_path = output_dir / PROTOCOL_FILENAME
    seal_path = output_dir / PROTOCOL_SEAL_FILENAME
    if not protocol_path.is_file() or not seal_path.is_file():
        raise FileNotFoundError("Protocol JSON and seal must both exist")
    protocol = _read_json(protocol_path)
    claimed = protocol.pop("protocol_sha256", None)
    observed = _json_sha256(protocol)
    protocol["protocol_sha256"] = claimed
    if claimed != observed or seal_path.read_text(encoding="ascii").strip() != observed:
        raise ValueError("Protocol JSON or seal changed")
    rebuilt = _sealed_payload(
        _rebuild_protocol_payload(output_dir),
        "protocol_sha256",
    )
    if protocol != rebuilt:
        raise ValueError("Protocol differs from canonical current inputs")
    return protocol


def _expected_case_metadata(case):
    return {
        key: case[key]
        for key in (
            "split",
            "id",
            "label",
            "seed",
            "synset",
            "latent_relative",
            "latent_key",
            "latent_size",
            "latent_sha256",
        )
    }


def _result_path(output_dir, split, case_id):
    return Path(output_dir) / "cases" / split / f"{case_id}.json"


def _assignment_maps(protocol, split):
    cases = [case for case in protocol["cases"] if case["split"] == split]
    assignments = protocol["assignments"][split]
    if len(cases) != SPLIT_COUNTS[split] or len(assignments) != len(cases):
        raise ValueError(f"Protocol case count changed for {split}")
    mapping = {}
    for case, assignment in zip(cases, assignments):
        if assignment["case_id"] != case["id"]:
            raise ValueError("Protocol assignment order changed")
        device = assignment["device"]
        if device not in LOCKED_DEVICES:
            raise ValueError("Protocol names an unlocked device")
        if set(assignment) != {"case_id", "device"}:
            raise ValueError("Protocol assignment schema changed")
        mapping[case["id"]] = assignment
    return cases, mapping


def _validate_case_result(result, protocol, case, assignment, path):
    if result.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError(f"Result protocol binding changed: {path}")
    if result.get("batch_case") != _expected_case_metadata(case):
        raise ValueError(f"Result case metadata changed: {path}")
    if result.get("support_world_sha256") != protocol["support_world_sha256"]:
        raise ValueError(f"Result support world changed: {path}")
    checkpoint = protocol["checkpoint"]
    if result.get("checkpoint_identity") != {
        "canonical_size": checkpoint["size"],
        "canonical_sha256": checkpoint["sha256"],
        "weights_size": checkpoint["size"],
        "weights_sha256": checkpoint["sha256"],
        "same_file": True,
    }:
        raise ValueError(f"Result checkpoint identity changed: {path}")
    if result.get("latent_identity") != {
        "size": case["latent_size"],
        "sha256": case["latent_sha256"],
    }:
        raise ValueError(f"Result latent identity changed: {path}")
    exact = {
        "rcl_responsibility_probe_version": PROBE_VERSION,
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_state": ONLINE_CHECKPOINT_STATE,
        "model_name": MODEL_NAME,
        "label": case["label"],
        "seed": case["seed"],
        "device": assignment["device"],
        "block_indices": list(BLOCK_INDICES),
        "sigmas": list(SIGMA_VALUES),
        "candidate_scales": list(CANDIDATE_SCALES),
        "token_probe_count": TOKEN_PROBE_COUNT,
        "exact_batch_size": EXACT_BATCH_SIZE,
        "assignment_shuffle_count": ASSIGNMENT_SHUFFLE_COUNT,
        "support_group_indices": list(range(SUPPORT_GROUP_COUNT)),
        "support_gradient_aggregation": "ddp_mean",
        "support_rank_count": SUPPORT_GROUP_COUNT,
        "support_batch_size_per_rank": SUPPORT_BATCH_SIZE,
        "support_global_batch_size": SUPPORT_BATCH_SIZE * SUPPORT_GROUP_COUNT,
        "support_forward_batch_size": SUPPORT_FORWARD_BATCH_SIZE,
        "center_step_relative_frobenius": CENTER_STEP_RELATIVE_FROBENIUS,
        "center_half_step_multiplier": CENTER_HALF_STEP_MULTIPLIER,
    }
    for key, expected in exact.items():
        if result.get(key) != expected:
            raise ValueError(f"Result field {key!r} changed: {path}")
    for key, expected in (
        ("checkpoint", checkpoint["path"]),
        ("weights_checkpoint", checkpoint["path"]),
        ("config", protocol["config"]["path"]),
        ("latent", case["latent"]),
    ):
        if Path(result.get(key, "")).resolve() != Path(expected).resolve():
            raise ValueError(f"Result path {key!r} changed: {path}")
    return result


def _load_case_result(path, protocol, case, assignment):
    return _validate_case_result(
        _read_sealed_json(path, RESULT_SHA256_FIELD),
        protocol,
        case,
        assignment,
        path,
    )


def _run_device_cases(device, cases, protocol, output_dir, loaded_state):
    assignments = {
        item["case_id"]: item
        for split in SPLIT_ORDER
        for item in protocol["assignments"][split]
        if item["device"] == device
    }
    pending = []
    completed = []
    for case in cases:
        path = _result_path(output_dir, case["split"], case["id"])
        if path.exists():
            _load_case_result(path, protocol, case, assignments[case["id"]])
            completed.append(case["id"])
        else:
            pending.append(case)
    if pending and (
        loaded_state.get("probe") is None or loaded_state.get("support") is None
    ):
        raise RuntimeError("Worker query state was not prepared and DDP-aggregated")
    for case in pending:
        result = run_rcl_responsibility_query(
            loaded_probe=loaded_state["probe"],
            latent_path=case["latent"],
            label=case["label"],
            support_results=loaded_state["support"],
            latent_key=case["latent_key"],
            seed=case["seed"],
            expected_latent_size=case["latent_size"],
            expected_latent_sha256=case["latent_sha256"],
        )
        result.update({
            "batch_case": _expected_case_metadata(case),
            "support_world_sha256": protocol["support_world_sha256"],
            "protocol_sha256": protocol["protocol_sha256"],
        })
        path = _result_path(output_dir, case["split"], case["id"])
        _write_sealed_json(path, result, RESULT_SHA256_FIELD)
        completed.append(case["id"])
    return completed


def _prepare_rank_support(device, protocol, state):
    group_index = LOCKED_DEVICES.index(device)
    if state.get("probe") is None:
        state["probe"] = load_rcl_responsibility_probe_model(
            checkpoint_path=protocol["checkpoint"]["path"],
            device=device,
            num_threads=8,
            expected_checkpoint_size=protocol["checkpoint"]["size"],
            expected_checkpoint_sha256=protocol["checkpoint"]["sha256"],
        )
    support_cases = protocol["support_groups"][str(group_index)]
    return build_rank_local_support_rcl(
        state["probe"]["model"],
        support_cases,
        group_index,
        state["probe"]["device"],
    )


def _persistent_worker(device, protocol, output_dir, command_queue, result_queue, parent_pid):
    _arm_parent_death_signal(parent_pid)
    state = {"probe": None, "support": None}
    try:
        while True:
            command = command_queue.get()
            if command is None:
                break
            phase = command[0]
            try:
                if phase == "prepare_support":
                    local_support = _prepare_rank_support(device, protocol, state)
                    result_queue.put({
                        "device": device,
                        "phase": phase,
                        "support_group_index": LOCKED_DEVICES.index(device),
                        "local_support": local_support,
                        "error": None,
                    })
                elif phase == "install_support":
                    state["support"] = command[1]
                    result_queue.put({
                        "device": device,
                        "phase": phase,
                        "error": None,
                    })
                elif phase == "run_split":
                    split, cases = command[1:]
                    completed = _run_device_cases(
                        device,
                        cases,
                        protocol,
                        output_dir,
                        state,
                    )
                    result_queue.put({
                        "device": device,
                        "phase": phase,
                        "split": split,
                        "completed": completed,
                        "error": None,
                    })
                else:
                    raise ValueError(f"Unknown worker phase: {phase!r}")
            except BaseException:
                result_queue.put({
                    "device": device,
                    "phase": phase,
                    "completed": [],
                    "error": traceback.format_exc(),
                })
                break
    finally:
        if state["probe"] is not None:
            del state["probe"]["model"]
        state.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_split_results(output_dir, protocol, split):
    cases, assignments = _assignment_maps(protocol, split)
    results = [
        _load_case_result(
            _result_path(output_dir, split, case["id"]),
            protocol,
            case,
            assignments[case["id"]],
        )
        for case in cases
    ]
    return cases, results


def _build_summary(protocol, split, cases, results):
    summary = aggregate_case_results(
        results,
        split,
        prerequisite_discovery_passed=(True if split == "confirmatory" else None),
    )
    summary.update({
        "protocol_sha256": protocol["protocol_sha256"],
        "case_ids": [case["id"] for case in cases],
        "case_result_sha256": {
            result["batch_case"]["id"]: result[RESULT_SHA256_FIELD]
            for result in results
        },
    })
    return summary


def _summary_path(output_dir, split):
    return Path(output_dir) / "summaries" / f"{split}.json"


def _write_or_validate_summary(output_dir, protocol, split):
    cases, results = _load_split_results(output_dir, protocol, split)
    expected = _sealed_payload(
        _build_summary(protocol, split, cases, results),
        SUMMARY_SHA256_FIELD,
    )
    path = _summary_path(output_dir, split)
    if path.exists():
        observed = _read_sealed_json(path, SUMMARY_SHA256_FIELD)
        if observed != expected:
            raise RuntimeError(f"{split} summary differs from raw case results")
        return observed
    return _write_sealed_json(
        path,
        _build_summary(protocol, split, cases, results),
        SUMMARY_SHA256_FIELD,
    )


def _verify_prerequisites(output_dir, protocol, split):
    for prerequisite in SPLIT_PREREQUISITES[split]:
        path = _summary_path(output_dir, prerequisite)
        if not path.is_file():
            raise RuntimeError(f"A completed {prerequisite} summary is required")
        summary = _write_or_validate_summary(output_dir, protocol, prerequisite)
        if summary.get("passed") is not True:
            raise RuntimeError(f"A passing {prerequisite} gate is required")


@contextmanager
def _run_locks(output_dir):
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Protocol output directory is missing: {output_dir}")
    with _exclusive_lock(GLOBAL_RUN_LOCK, "GPU 0-3 RCL gate"):
        with _exclusive_lock(
            output_dir / OUTPUT_RUN_LOCK_FILENAME,
            "RCL-responsibility output",
        ):
            yield


def _start_workers(protocol, output_dir):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    workers = {}
    try:
        for device in LOCKED_DEVICES:
            command_queue = context.Queue()
            process = context.Process(
                target=_persistent_worker,
                args=(
                    device,
                    protocol,
                    str(output_dir),
                    command_queue,
                    result_queue,
                    os.getpid(),
                ),
            )
            workers[device] = (process, command_queue)
            process.start()
    except BaseException:
        _stop_workers(workers)
        raise
    return workers, result_queue


def _stop_workers(workers):
    for process, command_queue in workers.values():
        if process.pid is None:
            continue
        if process.is_alive():
            command_queue.put(None)
    for process, _ in workers.values():
        if process.pid is None:
            continue
        process.join(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(timeout=WORKER_TERMINATE_TIMEOUT_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(timeout=WORKER_KILL_TIMEOUT_SECONDS)
    survivors = [
        process.pid
        for process, _ in workers.values()
        if process.pid is not None and process.is_alive()
    ]
    if survivors:
        raise RuntimeError(f"RCL gate workers survived forced shutdown: {survivors}")


def _collect_worker_messages(
    workers,
    result_queue,
    phase,
    timeout_seconds,
    split=None,
):
    deadline = time.monotonic() + float(timeout_seconds)
    messages = {}
    while len(messages) < len(LOCKED_DEVICES):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"RCL gate worker phase {phase!r} exceeded {timeout_seconds}s"
            )
        try:
            message = result_queue.get(
                timeout=min(WORKER_RESULT_POLL_SECONDS, remaining)
            )
        except queue.Empty:
            dead = [
                (device, process.exitcode)
                for device, (process, _) in workers.items()
                if not process.is_alive()
            ]
            if dead:
                raise RuntimeError(
                    f"RCL gate worker exited without a result: {dead}"
                )
            continue
        if not isinstance(message, dict):
            raise RuntimeError("RCL gate worker returned a malformed message")
        device = message.get("device")
        if device not in LOCKED_DEVICES or device in messages:
            raise RuntimeError("RCL gate worker completion set changed")
        if message.get("phase") != phase:
            raise RuntimeError("RCL gate worker phase changed")
        if message.get("error") is not None:
            raise RuntimeError(
                f"RCL gate worker failed on {device}:\n{message['error']}"
            )
        if split is not None and message.get("split") != split:
            raise RuntimeError("RCL gate worker split changed")
        messages[device] = message
    return [messages[device] for device in LOCKED_DEVICES]


def _prepare_global_support(workers, result_queue):
    for _, command_queue in workers.values():
        command_queue.put(("prepare_support",))
    messages = _collect_worker_messages(
        workers,
        result_queue,
        "prepare_support",
        WORKER_SUPPORT_TIMEOUT_SECONDS,
    )
    local_support = {}
    for device, message in zip(LOCKED_DEVICES, messages):
        group_index = int(message["support_group_index"])
        if group_index != LOCKED_DEVICES.index(device):
            raise RuntimeError("Worker returned the wrong support rank")
        local_support[group_index] = message["local_support"]
    global_support = aggregate_rank_support_rcl(local_support)
    for _, command_queue in workers.values():
        command_queue.put(("install_support", global_support))
    _collect_worker_messages(
        workers,
        result_queue,
        "install_support",
        WORKER_SUPPORT_TIMEOUT_SECONDS,
    )
    return global_support


def _dispatch_split(workers, result_queue, protocol, split):
    cases, assignments = _assignment_maps(protocol, split)
    for device, (_, queue) in workers.items():
        queue.put((
            "run_split",
            split,
            [case for case in cases if assignments[case["id"]]["device"] == device],
        ))
    _collect_worker_messages(
        workers,
        result_queue,
        "run_split",
        WORKER_SPLIT_TIMEOUT_SECONDS,
        split=split,
    )


def run_gate(output_dir):
    """Run all allowed splits while keeping one model/support state per GPU."""

    with _run_locks(output_dir):
        protocol = verify_protocol(output_dir)
        output_dir = Path(output_dir).resolve()
        workers, result_queue = _start_workers(protocol, output_dir)
        summaries = {}
        try:
            _prepare_global_support(workers, result_queue)
            for split in SPLIT_ORDER:
                _verify_prerequisites(output_dir, protocol, split)
                summary_path = _summary_path(output_dir, split)
                if summary_path.is_file():
                    summary = _write_or_validate_summary(
                        output_dir,
                        protocol,
                        split,
                    )
                else:
                    _dispatch_split(workers, result_queue, protocol, split)
                    verified = verify_protocol(output_dir)
                    if verified["protocol_sha256"] != protocol["protocol_sha256"]:
                        raise RuntimeError("Protocol changed while workers ran")
                    summary = _write_or_validate_summary(
                        output_dir,
                        protocol,
                        split,
                    )
                summaries[split] = summary
                if summary.get("passed") is not True:
                    break
        finally:
            _stop_workers(workers)
        return {
            "protocol_sha256": protocol["protocol_sha256"],
            "summaries": summaries,
            "completed_splits": list(summaries),
            "passed": bool(
                list(summaries) == list(SPLIT_ORDER)
                and summaries["confirmatory"].get("passed") is True
            ),
        }


def run_split(output_dir, split):
    """Resume one split; use run_gate when model-load efficiency matters."""

    if split not in SPLIT_COUNTS:
        raise ValueError(f"Unknown split: {split}")
    with _run_locks(output_dir):
        protocol = verify_protocol(output_dir)
        output_dir = Path(output_dir).resolve()
        _verify_prerequisites(output_dir, protocol, split)
        if _summary_path(output_dir, split).is_file():
            return _write_or_validate_summary(output_dir, protocol, split)
        workers, result_queue = _start_workers(protocol, output_dir)
        try:
            _prepare_global_support(workers, result_queue)
            _dispatch_split(workers, result_queue, protocol, split)
        finally:
            _stop_workers(workers)
        verified = verify_protocol(output_dir)
        if verified["protocol_sha256"] != protocol["protocol_sha256"]:
            raise RuntimeError("Protocol changed while workers ran")
        return _write_or_validate_summary(output_dir, protocol, split)
