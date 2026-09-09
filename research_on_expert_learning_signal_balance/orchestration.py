"""Preflight, prerequisite, evaluation, and throughput orchestration."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import torch
import yaml

from analyses.t_SNE.checkpoint_utils import load_runtime_cfg

from .controller import BRANCHES
from .evaluator import (
    evaluate_checkpoint_cases,
    publish_evaluation_complete,
    validate_branch_checkpoint,
    validate_controller_artifacts,
    validate_branch_transcripts,
    validate_protocol_for_evaluation,
)
from .heldout import canonical_json_sha256
from .git_provenance import (
    reject_history_overrides,
    repository_state,
    run_git,
    verify_worktree_source_manifest,
)
from .protocol import (
    DEFAULT_OUTPUT_ROOT,
    SEALED_GPU_IDS,
    _sealed_gpu_device_pairs,
    load_and_verify_protocol,
    resolve_archived_artifact_path,
)
from .serialization import atomic_write_json, sha256_file
from .state_digest import checkpoint_state_digests
from .statistics import aggregate_statistics


PREFLIGHT_VERSION = 1
THROUGHPUT_VERSION = 1
START_STEP = 301001
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _publish_sealed(path, payload, protocol_sha256):
    path = Path(path).resolve()
    seal_path = Path(str(path) + ".seal.json")
    seal = {
        "version": 1,
        "artifact": path.name,
        "artifact_canonical_sha256": canonical_json_sha256(payload),
        "protocol_sha256": protocol_sha256,
    }
    if path.exists() or seal_path.exists():
        if not path.exists() or not seal_path.exists():
            raise RuntimeError(f"Sealed artifact pair is incomplete: {path}")
        if _load_json(path) != payload or _load_json(seal_path) != seal:
            raise RuntimeError(f"Existing sealed artifact differs: {path}")
    else:
        atomic_write_json(path, payload, mode=0o444)
        atomic_write_json(seal_path, seal, mode=0o444)
    return path


def _load_sealed(path, protocol_sha256):
    path = Path(path).resolve()
    seal_path = Path(str(path) + ".seal.json")
    if not path.is_file() or not seal_path.is_file():
        raise FileNotFoundError(f"Sealed artifact pair is absent: {path}")
    payload = _load_json(path)
    seal = _load_json(seal_path)
    if (
        seal.get("protocol_sha256") != protocol_sha256
        or seal.get("artifact_canonical_sha256")
        != canonical_json_sha256(payload)
    ):
        raise RuntimeError(f"Sealed artifact verification failed: {path}")
    return payload


def _validate_parent_summary(output_dir, name, protocol_sha256):
    output_dir = Path(output_dir).resolve()
    path = output_dir / f"{name}-summary.json"
    seal_path = Path(str(path) + ".seal.json")
    if not path.is_file() or not seal_path.is_file():
        raise FileNotFoundError(f"Prerequisite summary is absent: {path}")
    payload = _load_json(path)
    seal = _load_json(seal_path)
    if seal.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError(f"Prerequisite protocol mismatch: {path}")
    if seal.get("result_sha256") != canonical_json_sha256(payload):
        raise RuntimeError(f"Prerequisite summary seal mismatch: {path}")
    if payload.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError(f"Prerequisite summary protocol differs: {path}")
    if payload.get("name") not in (None, name):
        raise RuntimeError(f"Prerequisite summary name differs: {path}")
    if payload.get("passed") is not True:
        raise RuntimeError(f"Prerequisite stage did not pass: {path}")
    return seal["result_sha256"]


def _load_protocol_document(path, expected_sha256=None):
    path = Path(path).resolve()
    payload = _load_json(path)
    observed = canonical_json_sha256(payload)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"Prerequisite protocol hash differs: {path}")
    if path.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ) != observed + "\n":
        raise RuntimeError(f"Prerequisite protocol sidecar differs: {path}")
    return payload, observed


def _git_commit_is_ancestor(ancestor, descendant):
    result = run_git(
        PROJECT_ROOT,
        "merge-base",
        "--is-ancestor",
        ancestor,
        descendant,
        text=True,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise RuntimeError(
        "Could not verify cross-checkpoint Git ancestry: "
        + result.stderr.strip()
    )


def _git_blob_sha256(commit, relative):
    result = run_git(
        PROJECT_ROOT,
        "cat-file",
        "blob",
        f"{commit}:{relative}",
        text=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Cross-checkpoint source is absent from its recorded Git commit: "
            f"{relative}"
        )
    return hashlib.sha256(result.stdout).hexdigest()


def _verify_cross_checkpoint_git_provenance(cross_protocol, current_commit):
    cross_git = cross_protocol.get("git", {})
    cross_commit = cross_git.get("commit")
    hexadecimal = frozenset("0123456789abcdef")
    for label, commit in (
        ("cross-checkpoint", cross_commit),
        ("continuation", current_commit),
    ):
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or not set(commit).issubset(hexadecimal)
        ):
            raise RuntimeError(f"{label} Git commit is not a full SHA-1")
    if cross_git.get("origin_repa_divergence") != "0\t0":
        raise RuntimeError("Cross-checkpoint gate was not run from pushed code")
    reject_history_overrides(PROJECT_ROOT)
    if not _git_commit_is_ancestor(cross_commit, current_commit):
        raise RuntimeError(
            "Cross-checkpoint gate commit is not an ancestor of the "
            "continuation implementation"
        )

    source_hashes = cross_protocol.get("project_source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError("Cross-checkpoint protocol has no source binding")
    for relative, expected in source_hashes.items():
        if not isinstance(relative, str):
            raise RuntimeError(
                f"Cross-checkpoint source path is invalid: {relative!r}"
            )
        source_path = PurePosixPath(relative)
        if (
            not relative
            or source_path.is_absolute()
            or ".." in source_path.parts
            or ":" in relative
        ):
            raise RuntimeError(
                f"Cross-checkpoint source path is invalid: {relative!r}"
            )
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or not set(expected).issubset(hexadecimal)
        ):
            raise RuntimeError(
                f"Cross-checkpoint source hash is invalid: {relative}"
            )
        if _git_blob_sha256(cross_commit, relative) != expected:
            raise RuntimeError(
                "Cross-checkpoint Git source binding changed: "
                f"{relative}"
            )


def _verify_continuation_git_provenance(current_commit, source_hashes):
    state = repository_state(PROJECT_ROOT)
    if current_commit != state["commit"]:
        raise RuntimeError("Continuation protocol commit differs from real HEAD")
    if state["commit"] != state["origin_repa"]:
        raise RuntimeError("Continuation implementation is not pushed to origin/repa")
    if state["commit"] != state["authoritative_remote_tip"]:
        raise RuntimeError(
            "Continuation implementation is not pushed to the authoritative "
            "remote repa branch"
        )
    if state["status"]:
        raise RuntimeError("Continuation implementation worktree is not clean")
    verify_worktree_source_manifest(
        PROJECT_ROOT,
        current_commit,
        source_hashes,
    )


def verify_prerequisites(protocol):
    contracts = protocol["prerequisites"]
    base = contracts["base_gate"]
    if sha256_file(base["preregister_path"]) != base["preregister_file_sha256"]:
        raise RuntimeError("Base prerequisite preregistration changed")
    _, base_protocol_sha256 = _load_protocol_document(
        base["protocol_path"], base["protocol_canonical_sha256"]
    )
    base_root = Path(base["output_root"])
    base_stages = tuple(base["required_summaries"])
    base_hashes = {
        name: _validate_parent_summary(base_root, name, base_protocol_sha256)
        for name in base_stages
    }

    cross = contracts["cross_checkpoint_gate"]
    preregistrations = (
        (
            cross["preregister_v1_path"],
            cross["preregister_v1_file_sha256"],
            1,
        ),
        (
            cross["preregister_v2_path"],
            cross["preregister_v2_file_sha256"],
            2,
        ),
    )
    for path, expected, _ in preregistrations:
        if sha256_file(path) != expected:
            raise RuntimeError(f"Cross-checkpoint preregistration changed: {path}")
    cross_protocol, cross_protocol_sha256 = _load_protocol_document(
        cross["protocol_path"]
    )
    sealed_preregistrations = cross_protocol.get("effective_preregistrations")
    if (
        not isinstance(sealed_preregistrations, list)
        or len(sealed_preregistrations) != len(preregistrations)
    ):
        raise RuntimeError("Cross-checkpoint protocol preregistration binding differs")
    for sealed, (expected_path, expected_sha256, expected_version) in zip(
        sealed_preregistrations,
        preregistrations,
    ):
        if (
            not isinstance(sealed, dict)
            or set(sealed) != {"version", "path", "sha256"}
            or sealed.get("version") != expected_version
            or sealed.get("sha256") != expected_sha256
            or not isinstance(sealed.get("path"), str)
        ):
            raise RuntimeError(
                "Cross-checkpoint protocol preregistration binding differs"
            )
        sealed_path = sealed["path"]
        if sealed_path == expected_path:
            relocated_path = Path(expected_path)
        else:
            try:
                relocated_path = resolve_archived_artifact_path(sealed_path)
            except RuntimeError as error:
                raise RuntimeError(
                    "Cross-checkpoint protocol preregistration binding differs"
                ) from error
        if relocated_path.resolve() != Path(expected_path).resolve():
            raise RuntimeError(
                "Cross-checkpoint protocol preregistration binding differs"
            )
    if cross_protocol.get("stage_order") != cross["required_stage_order"]:
        raise RuntimeError("Cross-checkpoint protocol stage order differs")
    if cross_protocol.get("base_protocol", {}).get(
        "canonical_json_sha256"
    ) != base_protocol_sha256:
        raise RuntimeError("Cross-checkpoint protocol used a different Base gate")
    current_commit = protocol["git"]["commit"]
    _verify_continuation_git_provenance(
        current_commit,
        protocol.get("project_source_file_sha256"),
    )
    _verify_cross_checkpoint_git_provenance(
        cross_protocol,
        current_commit,
    )
    for checkpoint in cross_protocol.get("checkpoints", {}).values():
        checkpoint_path = resolve_archived_artifact_path(checkpoint["path"])
        if checkpoint_path.stat().st_size != checkpoint["size"]:
            raise RuntimeError(f"Prerequisite checkpoint size changed: {checkpoint_path}")
        if sha256_file(checkpoint_path) != checkpoint["sha256"]:
            raise RuntimeError(f"Prerequisite checkpoint hash changed: {checkpoint_path}")
    cross_root = Path(cross["output_root"])
    cross_stages = tuple(cross["required_summaries"])
    cross_hashes = {
        name: _validate_parent_summary(cross_root, name, cross_protocol_sha256)
        for name in cross_stages
    }
    return {
        "base_protocol_sha256": base_protocol_sha256,
        "base_stage_result_sha256": base_hashes,
        "cross_checkpoint_protocol_sha256": cross_protocol_sha256,
        "cross_checkpoint_stage_result_sha256": cross_hashes,
    }


def _branch_entry(protocol, branch):
    entries = {entry["name"]: entry for entry in protocol["branches"]}
    if set(entries) != set(BRANCHES):
        raise RuntimeError("Immutable protocol branch set changed")
    return entries[branch]


def _validate_completed_branch(
    entry,
    reference_artifact_root=None,
    replay_context=None,
):
    runtime_cfg = load_runtime_cfg(Path(entry["config_path"]))
    reference_model = _build_model_for_validation(runtime_cfg)
    checkpoint, checkpoint_sha256 = validate_branch_checkpoint(
        entry["final_checkpoint_path"],
        None,
        entry["name"],
        reference_model=reference_model,
    )
    controller_integrity = validate_controller_artifacts(
        entry["artifact_root"], entry["name"], checkpoint
    )
    del checkpoint
    gc.collect()
    transcript_chain = validate_branch_transcripts(
        entry["artifact_root"],
        entry["name"],
        reference_artifact_root=(
            reference_artifact_root
            if entry["name"] != BRANCHES[0]
            else None
        ),
        reference_branch=BRANCHES[0],
        replay_context=replay_context,
    )
    del reference_model
    gc.collect()
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "transcript_final_chain_digest": transcript_chain,
        "controller_integrity": controller_integrity,
    }


def _build_model_for_validation(runtime_cfg):
    """Build a CPU reference model used only for checkpoint-shape validation."""
    from analyses.denoising_regret.probe import _build_model

    return _build_model(runtime_cfg).cpu().eval()


def verify_launch(branch):
    if branch not in BRANCHES:
        raise ValueError(f"Unknown branch: {branch}")
    protocol, protocol_sha256 = load_and_verify_protocol()
    replay_path = Path(protocol["preflight"]["replay_summary_path"])
    if (replay_path.parent / "work").exists():
        raise RuntimeError("Deterministic replay work remains after preflight")
    replay = _load_sealed(replay_path, protocol_sha256)
    if replay.get("passed") is not True:
        raise RuntimeError("Deterministic replay preflight did not pass")
    prerequisites = verify_prerequisites(protocol)
    target_index = BRANCHES.index(branch)
    reference_entry = _branch_entry(protocol, BRANCHES[0])
    reference_replay_context = None
    if target_index > 0 or Path(
        _branch_entry(protocol, branch)["final_checkpoint_path"]
    ).exists():
        reference_runtime_cfg = load_runtime_cfg(
            Path(reference_entry["config_path"])
        )
        reference_replay_context = {
            "initial_checkpoint_path": protocol["frozen_checkpoint"]["path"],
            "expected_checkpoint_sha256": protocol["frozen_checkpoint"][
                "file_sha256"
            ],
            "runtime_cfg": reference_runtime_cfg,
            "dataset_root": protocol["dataset"]["latent_root"],
            # CUDA_VISIBLE_DEVICES remaps the allocated physical slot to
            # logical device 0; replay must use the same GPU RNG algorithm.
            "device": torch.device(
                f"cuda:{dict(_sealed_gpu_device_pairs())[SEALED_GPU_IDS[0]]}"
            ),
        }
    completed = {
        prior: _validate_completed_branch(
            _branch_entry(protocol, prior),
            reference_artifact_root=reference_entry["artifact_root"],
            replay_context=reference_replay_context,
        )
        for prior in BRANCHES[:target_index]
    }
    completed_transcripts = {
        record["transcript_final_chain_digest"] for record in completed.values()
    }
    if len(completed_transcripts) > 1:
        raise RuntimeError("Completed branch input transcripts are not identical")
    target = _branch_entry(protocol, branch)
    if Path(target["final_checkpoint_path"]).exists():
        _validate_completed_branch(
            target,
            reference_artifact_root=reference_entry["artifact_root"],
            replay_context=reference_replay_context,
        )
        raise RuntimeError(f"Branch {branch} is already complete; refusing relaunch")
    return {
        "protocol_sha256": protocol_sha256,
        "branch": branch,
        "prerequisites": prerequisites,
        "completed_branches": completed,
        "target_output_dir": target["output_dir"],
    }


def _write_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run_training(config_path, log_path):
    environment = os.environ.copy()
    # Preserve an allocator-provided visibility/remapping contract.  Only set
    # the physical mask when the caller has not supplied one.
    environment.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    with Path(log_path).open("wb") as log_handle:
        subprocess.run(
            [sys.executable, "train.py", "--config", str(config_path)],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _base_preflight_config(protocol, work_root):
    source_path = Path(protocol["branches"][0]["config_path"])
    with source_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["output_dir"] = str(work_root / "runs")
    config["num_steps"] = START_STEP + 20
    config["save_ckpt_interval"] = 20
    config["gpu_ids"] = [4, 5, 6, 7]
    config["throughput_timer_config"] = {"enabled": False}
    return config


def _preflight_leg_config(protocol, work_root, leg):
    config = _base_preflight_config(protocol, work_root)
    artifact_root = work_root / "artifacts" / leg
    if leg == "baseline":
        config["credit_redistribution_config"] = {"enabled": False}
        config["training_transcript_config"] = {
            "enabled": True,
            "branch": "measure_only_control",
            "execution_mode": "deterministic_replay_baseline",
            "initial_checkpoint_path": protocol["frozen_checkpoint"]["path"],
            "preregister_v3_path": protocol["effective_preregistration"]["v3_path"],
            "preregister_v4_path": protocol["effective_preregistration"]["v4_path"],
            "artifact_root": str(artifact_root),
        }
    else:
        config["training_transcript_config"] = {"enabled": False}
        config["credit_redistribution_config"] = {
            "enabled": True,
            "branch": "measure_only_control",
            "execution_mode": "deterministic_replay",
            "initial_checkpoint_path": protocol["frozen_checkpoint"]["path"],
            "preregister_v3_path": protocol["effective_preregistration"]["v3_path"],
            "preregister_v4_path": protocol["effective_preregistration"]["v4_path"],
            "artifact_root": str(artifact_root),
        }
    return config, artifact_root


def _checkpoint_path(config, config_path):
    return (
        Path(config["output_dir"])
        / config["model_name"]
        / Path(config_path).stem
        / "checkpoints"
        / "ckpt_step_301020.pth"
    )


def _transcript_hashes(artifact_root, branch):
    root = Path(artifact_root) / "transcripts" / branch
    paths = [root / f"rank-{rank:02d}.jsonl" for rank in range(4)]
    paths.append(root / "global.jsonl")
    return {path.name: sha256_file(path) for path in paths}


def run_preflight():
    if not os.environ.get("TMUX"):
        raise RuntimeError("Run deterministic replay inside the attached tmux session")
    protocol, protocol_sha256 = load_and_verify_protocol()
    output_root = DEFAULT_OUTPUT_ROOT / "preflight"
    summary_path = output_root / "replay-summary.json"
    work_root = output_root / "work"
    if summary_path.exists():
        summary = _load_sealed(summary_path, protocol_sha256)
        if summary.get("passed") is not True:
            raise RuntimeError("Existing deterministic replay summary did not pass")
        if work_root.exists():
            raise RuntimeError("Sealed preflight summary claims cleanup, but work remains")
        return summary_path, summary
    if work_root.exists():
        raise RuntimeError("Unexpected preflight work directory already exists")
    legs = ("baseline", "measure-a", "measure-b")
    records = {}
    payload = None
    try:
        for leg in legs:
            config, artifact_root = _preflight_leg_config(
                protocol, work_root, leg
            )
            config_path = work_root / "configs" / f"{leg}.yaml"
            log_path = work_root / "logs" / f"{leg}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _write_yaml(config_path, config)
            _run_training(config_path, log_path)
            checkpoint_path = _checkpoint_path(config, config_path)
            checkpoint_sha256 = sha256_file(checkpoint_path)
            load_kwargs = {
                "map_location": "cpu",
                "weights_only": True,
                "mmap": True,
            }
            try:
                checkpoint = torch.load(checkpoint_path, **load_kwargs)
            except TypeError:
                load_kwargs.pop("mmap")
                checkpoint = torch.load(checkpoint_path, **load_kwargs)
            digests = checkpoint_state_digests(checkpoint)
            del checkpoint
            gc.collect()
            records[leg] = {
                "checkpoint_file_sha256": checkpoint_sha256,
                "checkpoint_state_sha256": digests,
                "transcript_file_sha256": _transcript_hashes(
                    artifact_root, "measure_only_control"
                ),
            }

        measure_equal = (
            records["measure-a"]["checkpoint_state_sha256"]
            == records["measure-b"]["checkpoint_state_sha256"]
            and records["measure-a"]["transcript_file_sha256"]
            == records["measure-b"]["transcript_file_sha256"]
        )
        baseline_sections = (
            "model_state_dict",
            "ema_model_state_dict",
            "optimizer_state_dict",
            "trainer_state",
            "step",
        )
        baseline_equal = all(
            records["baseline"]["checkpoint_state_sha256"][section]
            == records["measure-a"]["checkpoint_state_sha256"][section]
            for section in baseline_sections
        )
        transcript_equal = (
            records["baseline"]["transcript_file_sha256"]
            == records["measure-a"]["transcript_file_sha256"]
        )
        passed = bool(measure_equal and baseline_equal and transcript_equal)
        payload = {
            "version": PREFLIGHT_VERSION,
            "protocol_sha256": protocol_sha256,
            "git_commit": protocol["git"]["commit"],
            "updates_per_leg": 20,
            "legs": list(legs),
            "records": records,
            "checks": {
                "independent_measure_only_replays_identical": measure_equal,
                "measure_only_matches_transcript_only_base_state": baseline_equal,
                "all_input_transcripts_identical": transcript_equal,
            },
            "smoke_artifacts_removed_before_sealing": True,
            "passed": passed,
        }
        if not passed:
            raise RuntimeError("Deterministic replay preflight failed")
    finally:
        if work_root.exists():
            resolved = work_root.resolve()
            expected = (DEFAULT_OUTPUT_ROOT / "preflight" / "work").resolve()
            if resolved != expected:
                raise RuntimeError("Refusing to clean an unexpected preflight path")
            shutil.rmtree(resolved)
    if payload is None:
        raise RuntimeError("Deterministic replay preflight produced no summary")
    _publish_sealed(summary_path, payload, protocol_sha256)
    return summary_path, payload


def _evaluation_worker(payload):
    device = torch.device(payload["device"])
    completed = []
    for entry in payload["branches"]:
        spec = payload["checkpoint_specs"][entry["name"]]
        completed.extend(evaluate_checkpoint_cases(
            config_path=entry["config_path"],
            checkpoint_path=spec["path"],
            checkpoint_sha256=spec["sha256"],
            branch=entry["name"],
            cases=payload["cases"],
            tensor_dir=payload["tensor_dir"],
            output_root=payload["output_root"],
            protocol_sha256=payload["protocol_sha256"],
            device=device,
        ))
    return {"device": str(device), "completed": len(completed)}


def run_evaluation(devices=("cuda:4", "cuda:5", "cuda:6", "cuda:7")):
    if not os.environ.get("TMUX"):
        raise RuntimeError("Run held-out evaluation inside the attached tmux session")
    if tuple(devices) != ("cuda:4", "cuda:5", "cuda:6", "cuda:7"):
        raise ValueError("Held-out evaluation requires the sealed GPU group 4-7")
    visible_devices = tuple(
        f"cuda:{visible_index}"
        for physical_id, visible_index in _sealed_gpu_device_pairs()
        if physical_id in SEALED_GPU_IDS
    )
    if len(visible_devices) != len(SEALED_GPU_IDS):
        raise RuntimeError("Could not resolve all sealed evaluation devices")
    verified, verified_sha256 = load_and_verify_protocol()
    (
        protocol,
        protocol_sha256,
        manifest,
        checkpoint_specs,
        transcript_chains,
        branch_integrity,
        trainer_state_digests,
    ) = validate_protocol_for_evaluation(
        DEFAULT_OUTPUT_ROOT / "protocol.json"
    )
    if protocol != verified or protocol_sha256 != verified_sha256:
        raise RuntimeError("Protocol changed between integrity checks")
    cases = manifest["cases"]
    payloads = []
    for device_index, device in enumerate(visible_devices):
        payloads.append({
            "device": device,
            "branches": protocol["branches"],
            "checkpoint_specs": checkpoint_specs,
            "cases": cases[device_index::len(devices)],
            "tensor_dir": protocol["heldout"]["tensor_directory"],
            "output_root": protocol["heldout_evaluation_output"],
            "protocol_sha256": protocol_sha256,
        })
    context = torch.multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(payloads), mp_context=context
    ) as executor:
        futures = [executor.submit(_evaluation_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            print(f"{result['device']}: {result['completed']} sealed case-state results")
    for entry in protocol["branches"]:
        observed = sha256_file(entry["final_checkpoint_path"])
        if observed != checkpoint_specs[entry["name"]]["sha256"]:
            raise RuntimeError("Branch checkpoint changed during held-out evaluation")
    manifest_sha256 = protocol["heldout"]["manifest_canonical_sha256"]
    completion = publish_evaluation_complete(
        protocol["heldout_evaluation_output"],
        protocol_sha256,
        manifest_sha256,
        transcript_chains,
        checkpoint_specs,
        branch_integrity,
        trainer_state_digests,
    )
    return completion


def _revalidate_before_aggregation(protocol, protocol_sha256):
    completion_path = (
        Path(protocol["heldout_evaluation_output"]) / "evaluation-complete.json"
    )
    completion = _load_sealed(completion_path, protocol_sha256)
    (
        observed_protocol,
        observed_protocol_sha256,
        _,
        checkpoint_specs,
        transcript_chains,
        branch_integrity,
        trainer_state_digests,
    ) = validate_protocol_for_evaluation(DEFAULT_OUTPUT_ROOT / "protocol.json")
    if observed_protocol != protocol or observed_protocol_sha256 != protocol_sha256:
        raise RuntimeError("Protocol changed before efficacy aggregation")
    observed_bindings = {
        "checkpoint_file_sha256": {
            branch: spec["sha256"] for branch, spec in checkpoint_specs.items()
        },
        "transcript_final_chain_digests": transcript_chains,
        "branch_integrity": branch_integrity,
        "trainer_state_sha256": trainer_state_digests,
    }
    for key, value in observed_bindings.items():
        if completion.get(key) != value:
            raise RuntimeError(f"Evaluation completion {key} changed before aggregation")


def run_aggregation():
    protocol, protocol_sha256 = load_and_verify_protocol()
    _revalidate_before_aggregation(protocol, protocol_sha256)
    return aggregate_statistics(
        protocol["heldout_evaluation_output"],
        protocol_sha256,
        protocol["heldout"]["manifest_path"],
    )


def _throughput_leg_config(protocol, work_root, leg, mode, reference_root):
    source_path = Path(protocol["branches"][0]["config_path"])
    with source_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    artifact_root = work_root / "artifacts" / leg
    config.update({
        "output_dir": str(work_root / "runs"),
        "num_steps": START_STEP + 600,
        "save_ckpt_interval": 10_000_000,
        "gpu_ids": [4, 5, 6, 7],
        "log_interval": 10_000_000,
    })
    timer_mode = (
        "transcript_only" if mode == "A" else "matched_redistribution"
    )
    config["throughput_timer_config"] = {
        "enabled": True,
        "leg": leg,
        "mode": timer_mode,
        "start_step": START_STEP,
        "warmup_updates": 100,
        "timed_updates": 500,
        "output_path": str(work_root / "results" / f"{leg}.json"),
    }
    if mode == "A":
        config["credit_redistribution_config"] = {"enabled": False}
        transcript = {
            "enabled": True,
            "branch": "measure_only_control",
            "execution_mode": "throughput_baseline",
            "initial_checkpoint_path": protocol["frozen_checkpoint"]["path"],
            "preregister_v3_path": protocol["effective_preregistration"]["v3_path"],
            "preregister_v4_path": protocol["effective_preregistration"]["v4_path"],
            "artifact_root": str(artifact_root),
        }
        if reference_root is not None:
            transcript["reference_artifact_root"] = str(reference_root)
        config["training_transcript_config"] = transcript
    else:
        config["training_transcript_config"] = {"enabled": False}
        config["credit_redistribution_config"] = {
            "enabled": True,
            "branch": "matched_credit_rate_redistribution",
            "execution_mode": "throughput",
            "initial_checkpoint_path": protocol["frozen_checkpoint"]["path"],
            "preregister_v3_path": protocol["effective_preregistration"]["v3_path"],
            "preregister_v4_path": protocol["effective_preregistration"]["v4_path"],
            "artifact_root": str(artifact_root),
            "reference_artifact_root": str(reference_root),
        }
    return config, artifact_root


def run_throughput():
    if not os.environ.get("TMUX"):
        raise RuntimeError("Run ABBA throughput inside the attached tmux session")
    protocol, protocol_sha256 = load_and_verify_protocol()
    statistics_path = (
        Path(protocol["heldout_evaluation_output"]) / "statistics" / "summary.json"
    )
    statistics = _load_sealed(statistics_path, protocol_sha256)
    if statistics.get("all_required_passed") is not True:
        raise RuntimeError("Efficacy did not authorize throughput/fresh training")
    output_root = DEFAULT_OUTPUT_ROOT / "throughput"
    summary_path = output_root / "summary.json"
    work_root = output_root / "work"
    if summary_path.exists():
        summary = _load_sealed(summary_path, protocol_sha256)
        if summary.get("passed") is not True:
            raise RuntimeError("Existing ABBA throughput summary did not pass")
        if work_root.exists():
            raise RuntimeError("Sealed throughput summary claims cleanup, but work remains")
        return summary_path, summary
    if work_root.exists():
        raise RuntimeError("Unexpected throughput work directory already exists")
    legs = (("A1", "A"), ("B1", "B"), ("B2", "B"), ("A2", "A"))
    reference_root = work_root / "artifacts" / "A1"
    results = {}
    payload = None
    try:
        for leg, mode in legs:
            leg_reference = None if leg == "A1" else reference_root
            config, artifact_root = _throughput_leg_config(
                protocol, work_root, leg, mode, leg_reference
            )
            config_path = work_root / "configs" / f"{leg}.yaml"
            log_path = work_root / "logs" / f"{leg}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _write_yaml(config_path, config)
            _run_training(config_path, log_path)
            result_path = work_root / "results" / f"{leg}.json"
            result = _load_json(result_path)
            expected_mode = (
                "transcript_only" if mode == "A" else "matched_redistribution"
            )
            if (
                result.get("leg") != leg
                or result.get("mode") != expected_mode
                or result.get("world_size") != 4
                or result.get("start_step") != START_STEP
                or result.get("warmup_updates") != 100
                or result.get("timed_updates") != 500
                or not isinstance(result.get("seconds_per_update"), (int, float))
                or not math.isfinite(result["seconds_per_update"])
                or result["seconds_per_update"] <= 0
            ):
                raise RuntimeError(f"Throughput leg result differs: {leg}")
            transcript_branch = (
                "measure_only_control"
                if mode == "A"
                else "matched_credit_rate_redistribution"
            )
            result["transcript_file_sha256"] = _transcript_hashes(
                artifact_root, transcript_branch
            )
            results[leg] = result
        transcript_reference = results["A1"]["transcript_file_sha256"]
        transcripts_identical = all(
            row["transcript_file_sha256"] == transcript_reference
            for row in results.values()
        )
        a_values = sorted(results[leg]["seconds_per_update"] for leg in ("A1", "A2"))
        b_values = sorted(results[leg]["seconds_per_update"] for leg in ("B1", "B2"))
        median_a = (a_values[0] + a_values[1]) / 2.0
        median_b = (b_values[0] + b_values[1]) / 2.0
        slowdown = median_b / median_a - 1.0
        passed = bool(transcripts_identical and slowdown <= 0.10)
        payload = {
            "version": THROUGHPUT_VERSION,
            "protocol_sha256": protocol_sha256,
            "order": [leg for leg, _ in legs],
            "warmup_updates_per_leg": 100,
            "timed_updates_per_leg": 500,
            "legs": results,
            "median_a_seconds_per_update": median_a,
            "median_b_seconds_per_update": median_b,
            "relative_slowdown": slowdown,
            "maximum_slowdown": 0.10,
            "input_transcripts_identical": transcripts_identical,
            "smoke_artifacts_removed_before_sealing": True,
            "passed": passed,
        }
        if not passed:
            raise RuntimeError("ABBA throughput gate failed")
    finally:
        if work_root.exists():
            resolved = work_root.resolve()
            expected = (DEFAULT_OUTPUT_ROOT / "throughput" / "work").resolve()
            if resolved != expected:
                raise RuntimeError("Refusing to clean an unexpected throughput path")
            shutil.rmtree(resolved)
    if payload is None:
        raise RuntimeError("ABBA throughput gate produced no summary")
    _publish_sealed(summary_path, payload, protocol_sha256)
    return summary_path, payload
