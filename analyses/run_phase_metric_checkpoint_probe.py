"""Run the pre-registered Phase-Metric checkpoint mechanism gate."""

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from analyses.denoising_regret.io import write_json_atomic
from analyses.routing_metric.phase_checkpoint_probe import (
    PROBE_VERSION,
    build_gate_summary,
    load_gate_spec,
    run_probe_split,
    sha256_file,
)
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)
from analyses.timestep_utility.batch import load_manifest as load_case_manifest


RUNNER_VERSION = 1
DEFAULT_SPEC = (
    PROJECT_ROOT
    / "analyses"
    / "routing_metric"
    / "manifests"
    / "phase_metric_50k_gate_v1.json"
)
LOCKED_SOURCE_PATHS = (
    "requirements.txt",
    "config.py",
    "utils.py",
    "train.py",
    "models/models_ProMoE_TC.py",
    "models/modules.py",
    "models/phase_metric.py",
    "analyses/denoising_regret/io.py",
    "analyses/denoising_regret/probe.py",
    "analyses/routing_translation/probe.py",
    "analyses/t_SNE/checkpoint_utils.py",
    "analyses/timestep_utility/batch.py",
    "analyses/routing_metric/phase_checkpoint_probe.py",
    "analyses/run_phase_metric_checkpoint_probe.py",
)


def _json_payload_sha256(payload):
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_text_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _orchestration_lock(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".phase-metric-gate.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _verify_file(path, expected_sha256, description):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Locked {description} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"Locked {description} changed: {path}")


def _hash_checkpoint_pair(canonical_path, weights_path, description):
    canonical_hash = sha256_file(canonical_path)
    weights_hash = (
        canonical_hash
        if canonical_path == weights_path
        else sha256_file(weights_path)
    )
    if canonical_hash != weights_hash:
        raise ValueError(
            f"Canonical and local {description} checkpoint bytes differ"
        )
    return canonical_hash, weights_hash


def _checkpoint_contract(path, step, state_name):
    load_kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        checkpoint = torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only")
        checkpoint = torch.load(path, **load_kwargs)
    try:
        if checkpoint.get("step") != step:
            raise ValueError(f"Checkpoint payload step must be {step}")
        if state_name not in checkpoint:
            raise KeyError(f"Checkpoint is missing {state_name}")
    finally:
        del checkpoint
        gc.collect()


def _source_hashes():
    hashes = {}
    for relative in LOCKED_SOURCE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Locked source is missing: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def _runtime_environment(devices):
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "devices": list(devices),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("The locked CUDA probe cannot run without CUDA")
    payload["cuda_devices"] = {}
    for device in devices:
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError("The locked probe requires CUDA devices")
        properties = torch.cuda.get_device_properties(resolved_device)
        payload["cuda_devices"][device] = {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }
    return payload


def _load_cases(spec, latent_root):
    manifest = load_case_manifest(spec["case_manifest_path"], latent_root)
    if manifest["sha256"] != spec["protocol"]["case_manifest_sha256"]:
        raise RuntimeError("Loaded case manifest differs from the gate spec")
    counts = {
        split: sum(case["split"] == split for case in manifest["cases"])
        for split in ("discovery", "confirmatory")
    }
    if counts != spec["protocol"]["split_counts"]:
        raise ValueError("Case split counts differ from the gate spec")
    return manifest


def _config_contract(checkpoint_path, expected_stem, model_name):
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    runtime_cfg = load_runtime_cfg(config_path)
    if config_path.stem != expected_stem:
        raise ValueError(
            f"Expected config stem {expected_stem}, got {config_path.stem}"
        )
    if runtime_cfg.model_name != model_name:
        raise ValueError("Checkpoint config model_name differs from the gate")
    return config_path


def _load_yaml(path):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return payload


def _phase_config(payload, description):
    try:
        phase = payload["DiT_B_config"]["MoE_config"]["phase_metric_config"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{description} lacks phase_metric_config") from error
    if not isinstance(phase, dict):
        raise ValueError(f"{description} phase_metric_config must be a mapping")
    return phase


def _validate_matched_configs(candidate_path, base_path, spec):
    candidate = _load_yaml(candidate_path)
    base = _load_yaml(base_path)
    training = spec["protocol"]["training_contract"]
    for name, payload, expected_gpus in (
        ("candidate", candidate, training["candidate_gpu_ids"]),
        ("Base", base, training["base_gpu_ids"]),
    ):
        for key in ("global_seed", "total_train_batch_size", "lr"):
            if payload.get(key) != training[key]:
                raise ValueError(f"{name} config violates training contract: {key}")
        if payload.get("gpu_ids") != expected_gpus:
            raise ValueError(f"{name} config violates its GPU assignment")

    expected_phase = spec["protocol"]["phase_metric_contract"]
    candidate_phase = _phase_config(candidate, "Candidate config")
    base_phase = _phase_config(base, "Base config")
    if candidate_phase != expected_phase:
        raise ValueError("Candidate config violates the Phase-Metric contract")
    expected_base_phase = {**expected_phase, "enabled": False}
    if base_phase != expected_base_phase:
        raise ValueError("Base config is not the matched Phase-disabled control")

    normalized_candidate = copy.deepcopy(candidate)
    normalized_base = copy.deepcopy(base)
    normalized_candidate.pop("gpu_ids", None)
    normalized_base.pop("gpu_ids", None)
    _phase_config(normalized_candidate, "Candidate config")["enabled"] = False
    if normalized_candidate != normalized_base:
        raise ValueError(
            "Candidate and Base configs differ beyond GPU IDs and Phase enablement"
        )


def _training_log_snapshot(checkpoint_path, expected_stem):
    log_path = checkpoint_path.parent.parent / "training.log"
    if checkpoint_path.parent.parent.name != expected_stem:
        raise ValueError("Checkpoint run directory differs from its config stem")
    if not log_path.is_file():
        raise FileNotFoundError(f"Training log is missing: {log_path}")
    lines = log_path.read_text(encoding="utf-8").splitlines()
    resume_indices = [
        index for index, line in enumerate(lines) if "Resume progress:" in line
    ]
    if not resume_indices:
        raise ValueError("Training log has no resume-state evidence")
    first_resume = resume_indices[0]
    required_resume = (
        "Resume progress: next_step=0, data_batches_seen=0, sampler_epoch=0, "
        "sampler_batch_offset=0"
    )
    if required_resume not in lines[first_resume]:
        raise ValueError("Training did not begin from a fresh sampler state")
    fresh_lines = [
        line for line in lines[:first_resume] if "No checkpoints found in directory:" in line
    ]
    if not fresh_lines:
        raise ValueError("Training log does not show an empty checkpoint directory")
    forbidden_before_fresh = (
        "Loading checkpoint:",
        "Successfully loaded checkpoint",
    )
    if any(
        marker in line
        for line in lines[: first_resume + 1]
        for marker in forbidden_before_fresh
    ):
        raise ValueError("Training loaded a checkpoint before its first update")
    step_zero_indices = [
        index
        for index, line in enumerate(lines[first_resume + 1:], start=first_resume + 1)
        if "epoch 0-step 0 " in line
    ]
    if not step_zero_indices:
        raise ValueError("Training log does not contain the step-0 update")
    step_zero = step_zero_indices[0]
    normalized_prefix = "\n".join(lines[: step_zero + 1]) + "\n"
    return {
        "path": str(log_path.resolve()),
        "fresh_prefix_line_count": int(step_zero + 1),
        "fresh_prefix_sha256": hashlib.sha256(
            normalized_prefix.encode("utf-8")
        ).hexdigest(),
        "fresh_checkpoint_evidence": fresh_lines[-1],
        "fresh_sampler_evidence": lines[first_resume],
        "step_zero_evidence": lines[step_zero],
    }


def _verify_training_log(snapshot, description):
    path = Path(snapshot["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Locked {description} is missing: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    count = snapshot["fresh_prefix_line_count"]
    if len(lines) < count:
        raise RuntimeError(f"Locked {description} lost its fresh-start prefix")
    normalized_prefix = "\n".join(lines[:count]) + "\n"
    observed = hashlib.sha256(normalized_prefix.encode("utf-8")).hexdigest()
    if observed != snapshot["fresh_prefix_sha256"]:
        raise RuntimeError(f"Locked {description} fresh-start prefix changed")


def _checkpoint_snapshot(canonical_path, weights_path, sha256):
    return {
        "canonical_path": str(canonical_path),
        "canonical_sha256": sha256,
        "canonical_size": int(canonical_path.stat().st_size),
        "weights_path": str(weights_path),
        "weights_sha256": sha256,
        "weights_size": int(weights_path.stat().st_size),
    }


def _build_protocol(
    args,
    spec,
    manifest,
    candidate_path,
    candidate_weights,
    base_path,
    base_weights,
):
    protocol_spec = spec["protocol"]
    expected_step = protocol_spec["checkpoint_step"]
    state_name = protocol_spec["checkpoint_state"]
    execution = protocol_spec["probe_execution_contract"]
    if list(args.devices) != execution["devices"]:
        raise ValueError("CLI devices differ from the locked probe devices")
    if int(args.num_threads) != execution["num_threads_per_worker"]:
        raise ValueError("CLI thread count differs from the locked probe contract")
    for path, description in (
        (candidate_path, "candidate"),
        (candidate_weights, "candidate weights"),
        (base_path, "Base"),
        (base_weights, "Base weights"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} checkpoint is missing: {path}")
    if parse_checkpoint_step(candidate_path) != expected_step:
        raise ValueError("Candidate checkpoint filename has the wrong step")
    if parse_checkpoint_step(base_path) != expected_step:
        raise ValueError("Base checkpoint filename has the wrong step")

    candidate_config = _config_contract(
        candidate_path,
        protocol_spec["candidate_config_stem"],
        protocol_spec["model_name"],
    )
    base_config = _config_contract(
        base_path,
        protocol_spec["base_config_stem"],
        protocol_spec["model_name"],
    )
    _validate_matched_configs(candidate_config, base_config, spec)
    candidate_hash, _ = _hash_checkpoint_pair(
        candidate_path, candidate_weights, "candidate"
    )
    base_hash, _ = _hash_checkpoint_pair(base_path, base_weights, "Base")
    _checkpoint_contract(candidate_weights, expected_step, state_name)
    _checkpoint_contract(base_weights, expected_step, state_name)

    return {
        "runner_version": RUNNER_VERSION,
        "probe_version": PROBE_VERSION,
        "locked_before_probe_results": True,
        "hypothesis": (
            "The learned phase-conditioned metric improves expert selection, "
            "rather than merely rescaling routed outputs or exploiting an "
            "unmatched optimization trajectory."
        ),
        "decision_rule": (
            "Discovery must pass before the runner may execute confirmatory "
            "cases; confirmatory must pass before Phase-Metric is authorized "
            "to continue as a formal 500K candidate."
        ),
        "spec": {
            "path": spec["path"],
            "sha256": spec["sha256"],
            "name": spec["name"],
            "version": spec["version"],
        },
        "case_manifest": {
            "path": manifest["path"],
            "sha256": manifest["sha256"],
            "selection": manifest["selection"],
        },
        "cases": [
            {
                "split": case["split"],
                "id": case["id"],
                "label": int(case["label"]),
                "seed": int(case["seed"]),
                "latent": case["latent"],
                "latent_sha256": case["latent_sha256"],
            }
            for case in manifest["cases"]
        ],
        "candidate_checkpoint": _checkpoint_snapshot(
            candidate_path, candidate_weights, candidate_hash
        ),
        "base_checkpoint": _checkpoint_snapshot(
            base_path, base_weights, base_hash
        ),
        "candidate_config": {
            "path": str(candidate_config),
            "sha256": sha256_file(candidate_config),
        },
        "base_config": {
            "path": str(base_config),
            "sha256": sha256_file(base_config),
        },
        "candidate_training_log": _training_log_snapshot(
            candidate_path, protocol_spec["candidate_config_stem"]
        ),
        "base_training_log": _training_log_snapshot(
            base_path, protocol_spec["base_config_stem"]
        ),
        "locked_experiment": protocol_spec,
        "source_sha256": _source_hashes(),
        "runtime": {
            **_runtime_environment(args.devices),
            "num_threads_per_worker": int(args.num_threads),
            "latent_root": str(Path(args.latent_root).resolve()),
        },
        "assignments": {
            split: [
                {
                    "case_id": case["id"],
                    "device": args.devices[index % len(args.devices)],
                }
                for index, case in enumerate(
                    item for item in manifest["cases"] if item["split"] == split
                )
            ]
            for split in ("discovery", "confirmatory")
        },
        "output_dir": str(Path(args.output_dir).resolve()),
    }


def _write_or_validate_protocol(output_dir, protocol):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    hash_path = output_dir / "protocol.sha256"
    result_files = list(output_dir.glob("*-result.json"))
    if not protocol_path.exists() and result_files:
        raise RuntimeError("Refusing result files without a protocol lock")
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError("Existing protocol differs from this invocation")
    else:
        write_json_atomic(protocol_path, protocol)
    protocol_hash = sha256_file(protocol_path)
    expected_line = f"{protocol_hash}  protocol.json\n"
    if hash_path.exists():
        if hash_path.read_text(encoding="utf-8") != expected_line:
            raise ValueError("Protocol checksum file is invalid")
    else:
        _write_text_atomic(hash_path, expected_line)
    return protocol_path, protocol_hash


def _verify_locked_inputs(protocol, protocol_path, protocol_hash):
    _verify_file(protocol_path, protocol_hash, "protocol")
    for key in ("candidate_checkpoint", "base_checkpoint"):
        snapshot = protocol[key]
        checked = set()
        for path_key, hash_key in (
            ("canonical_path", "canonical_sha256"),
            ("weights_path", "weights_sha256"),
        ):
            path = snapshot[path_key]
            if path in checked:
                continue
            _verify_file(path, snapshot[hash_key], key)
            checked.add(path)
    for key in ("candidate_config", "base_config", "spec", "case_manifest"):
        _verify_file(protocol[key]["path"], protocol[key]["sha256"], key)
    for key in ("candidate_training_log", "base_training_log"):
        _verify_training_log(protocol[key], key)
    for relative, expected in protocol["source_sha256"].items():
        _verify_file(PROJECT_ROOT / relative, expected, f"source {relative}")
    for case in protocol["cases"]:
        _verify_file(
            case["latent"], case["latent_sha256"], f"latent {case['id']}"
        )


def _expected_case_identities(protocol, split):
    return [
        {
            "case_id": case["id"],
            "split": case["split"],
            "label": int(case["label"]),
            "seed": int(case["seed"]),
            "latent_sha256": case["latent_sha256"],
        }
        for case in protocol["cases"]
        if case["split"] == split
    ]


def _validate_result(result, split, spec, protocol_hash, protocol):
    if set(result) != {
        "runner_version",
        "probe_version",
        "protocol_sha256",
        "split",
        "probe",
        "gate",
    }:
        raise ValueError("Published result has an unexpected schema")
    if (
        result["runner_version"] != RUNNER_VERSION
        or result["probe_version"] != PROBE_VERSION
        or result["protocol_sha256"] != protocol_hash
        or result["split"] != split
    ):
        raise ValueError("Published result identity differs from the locked run")
    probe = result["probe"]
    if probe.get("probe_version") != PROBE_VERSION:
        raise ValueError("Embedded probe version differs")
    cases = probe.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Published result cases must be a list")
    identity_keys = {
        "case_id", "split", "label", "seed", "latent_sha256"
    }
    observed_identities = []
    for case in cases:
        if not isinstance(case, dict) or not identity_keys.issubset(case):
            raise ValueError("Published result lacks locked case identity fields")
        observed_identities.append({key: case[key] for key in identity_keys})
    expected_identities = _expected_case_identities(protocol, split)
    if observed_identities != expected_identities:
        raise ValueError(
            "Published result case identities or order differ from the protocol"
        )
    if len({case["case_id"] for case in observed_identities}) != len(cases):
        raise ValueError("Published result contains duplicate case identities")
    expected_count = spec["protocol"]["split_counts"][split]
    if len(cases) != expected_count:
        raise ValueError("Published result has the wrong case count")
    recomputed = build_gate_summary(cases, spec, split)
    if recomputed != result["gate"]:
        raise ValueError("Published gate differs from raw-case recomputation")
    return recomputed


def _load_published_result(output_dir, split, spec, protocol_hash, protocol):
    result_path = Path(output_dir) / f"{split}-result.json"
    hash_path = Path(output_dir) / f"{split}-result.sha256"
    if not result_path.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"Incomplete {split} result seal")
    result_hash = sha256_file(result_path)
    expected_line = f"{result_hash}  {result_path.name}\n"
    if hash_path.read_text(encoding="utf-8") != expected_line:
        raise ValueError(f"Invalid {split} result checksum")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    gate = _validate_result(result, split, spec, protocol_hash, protocol)
    return result, gate


def _require_confirmatory_unlock(output_dir, spec, protocol_hash, protocol):
    _, discovery_gate = _load_published_result(
        output_dir, "discovery", spec, protocol_hash, protocol
    )
    if not discovery_gate["passed"]:
        raise RuntimeError("Confirmatory cases remain locked after discovery failure")


def _result_publication_paths(output_dir, split):
    output_dir = Path(output_dir)
    return {
        "result": output_dir / f"{split}-result.json",
        "seal": output_dir / f"{split}-result.sha256",
        "pending_result": output_dir / f"{split}-result.json.pending",
        "pending_seal": output_dir / f"{split}-result.sha256.pending",
    }


def _recover_result_publication(
    output_dir,
    split,
    spec,
    protocol_hash,
    protocol,
):
    paths = _result_publication_paths(output_dir, split)
    present = {name: path.exists() for name, path in paths.items()}
    state = frozenset(name for name, exists in present.items() if exists)
    if not state:
        return "absent"
    if state == {"result", "seal"}:
        _load_published_result(output_dir, split, spec, protocol_hash, protocol)
        return "published"
    recoverable_states = {
        frozenset({"pending_result"}),
        frozenset({"pending_result", "pending_seal"}),
        frozenset({"result", "pending_seal"}),
    }
    if state not in recoverable_states:
        raise RuntimeError(
            f"Unreachable {split} result publication state: {sorted(state)}"
        )

    payload_path = paths["result"] if present["result"] else paths["pending_result"]
    result = json.loads(payload_path.read_text(encoding="utf-8"))
    _validate_result(result, split, spec, protocol_hash, protocol)
    result_hash = sha256_file(payload_path)
    expected_line = f"{result_hash}  {paths['result'].name}\n"
    seal_paths = [
        path for name, path in paths.items()
        if name in {"seal", "pending_seal"} and path.exists()
    ]
    for seal_path in seal_paths:
        if seal_path.read_text(encoding="utf-8") != expected_line:
            raise ValueError(f"Invalid recoverable {split} result checksum")
    if state == {"pending_result"}:
        _write_text_atomic(paths["pending_seal"], expected_line)
        present["pending_seal"] = True

    if not present["result"]:
        os.replace(paths["pending_result"], paths["result"])
    if not present["seal"]:
        os.replace(paths["pending_seal"], paths["seal"])
    _load_published_result(output_dir, split, spec, protocol_hash, protocol)
    return "recovered"


def _publish_result(
    output_dir,
    split,
    result,
    protocol,
    protocol_path,
    protocol_hash,
    spec,
):
    output_dir = Path(output_dir)
    paths = _result_publication_paths(output_dir, split)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Immutable {split} result already exists")
    expected_hash = _json_payload_sha256(result)
    _validate_result(result, split, spec, protocol_hash, protocol)
    _verify_locked_inputs(protocol, protocol_path, protocol_hash)
    write_json_atomic(paths["pending_result"], result)
    _verify_file(
        paths["pending_result"], expected_hash, f"pending {split} result"
    )
    _write_text_atomic(
        paths["pending_seal"],
        f"{expected_hash}  {paths['result'].name}\n",
    )
    _verify_locked_inputs(protocol, protocol_path, protocol_hash)
    os.replace(paths["pending_result"], paths["result"])
    os.replace(paths["pending_seal"], paths["seal"])
    _verify_file(
        paths["result"], expected_hash, f"published {split} result"
    )
    _load_published_result(output_dir, split, spec, protocol_hash, protocol)
    return paths["result"], expected_hash


def _run_device_shard(payload):
    device = payload["device"]
    torch.cuda.set_device(torch.device(device))

    def progress(stage, current, total, case_id):
        print(
            f"[{device}][{stage}] {current}/{total}: {case_id}",
            flush=True,
        )

    result = run_probe_split(
        candidate_checkpoint_path=payload["candidate_checkpoint"],
        candidate_weights_path=payload["candidate_weights"],
        base_checkpoint_path=payload["base_checkpoint"],
        base_weights_path=payload["base_weights"],
        cases=payload["cases"],
        spec=payload["spec"],
        device=device,
        num_threads=payload["num_threads"],
        progress=progress,
    )
    return {"device": device, "result": result}


def _merge_device_shards(shards, cases, devices):
    case_by_id = {}
    worker_metadata = []
    candidate_contract = None
    candidate_config = None
    base_config = None
    for shard in sorted(shards, key=lambda item: devices.index(item["device"])):
        result = shard["result"]
        if candidate_contract is None:
            candidate_contract = result["candidate_contract"]
            candidate_config = result["candidate_config"]
            base_config = result["base_config"]
        elif (
            result["candidate_contract"] != candidate_contract
            or result["candidate_config"] != candidate_config
            or result["base_config"] != base_config
        ):
            raise RuntimeError("Worker model contracts differ")
        for case in result["cases"]:
            if case["case_id"] in case_by_id:
                raise RuntimeError(f"Duplicate worker case: {case['case_id']}")
            case_by_id[case["case_id"]] = case
        worker_metadata.append({
            key: value for key, value in result.items() if key != "cases"
        })
    expected_ids = [case["id"] for case in cases]
    if set(case_by_id) != set(expected_ids):
        raise RuntimeError("Parallel workers did not return every locked case")
    return {
        "candidate_config": candidate_config,
        "base_config": base_config,
        "candidate_contract": candidate_contract,
        "worker_metadata": worker_metadata,
        "cases": [case_by_id[case_id] for case_id in expected_ids],
    }


def _run_parallel_probe(
    candidate_path,
    candidate_weights,
    base_path,
    base_weights,
    cases,
    spec,
    devices,
    num_threads,
):
    payloads = []
    for device_index, device in enumerate(devices):
        shard_cases = [
            case
            for case_index, case in enumerate(cases)
            if case_index % len(devices) == device_index
        ]
        if not shard_cases:
            continue
        payloads.append({
            "device": device,
            "candidate_checkpoint": str(candidate_path),
            "candidate_weights": str(candidate_weights),
            "base_checkpoint": str(base_path),
            "base_weights": str(base_weights),
            "cases": shard_cases,
            "spec": spec,
            "num_threads": int(num_threads),
        })
    start = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    shards = []
    with ProcessPoolExecutor(
        max_workers=len(payloads),
        mp_context=context,
    ) as executor:
        futures = [executor.submit(_run_device_shard, payload) for payload in payloads]
        for future in as_completed(futures):
            shards.append(future.result())

    merged = _merge_device_shards(shards, cases, devices)
    return {
        "probe_version": PROBE_VERSION,
        "devices": list(devices),
        "wall_seconds": float(time.perf_counter() - start),
        **merged,
    }


def _parse_devices(value):
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices or len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError("devices must be nonempty and unique")
    return devices


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked 50K Phase-Metric selection/weight mechanism gate."
        )
    )
    parser.add_argument("--candidate-ckpt", required=True)
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument(
        "--candidate-weights-ckpt",
        help="Optional byte-identical local copy used to load candidate weights",
    )
    parser.add_argument(
        "--base-weights-ckpt",
        help="Optional byte-identical local copy used to load Base weights",
    )
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        choices=("discovery", "confirmatory"),
        default="discovery",
    )
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.num_threads < 1:
        raise ValueError("num_threads must be positive")
    candidate_path = Path(args.candidate_ckpt).resolve()
    base_path = Path(args.base_ckpt).resolve()
    candidate_weights = Path(
        args.candidate_weights_ckpt or candidate_path
    ).resolve()
    base_weights = Path(args.base_weights_ckpt or base_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    with _orchestration_lock(output_dir):
        spec = load_gate_spec(DEFAULT_SPEC, PROJECT_ROOT)
        manifest = _load_cases(spec, args.latent_root)
        protocol = _build_protocol(
            args,
            spec,
            manifest,
            candidate_path,
            candidate_weights,
            base_path,
            base_weights,
        )
        protocol_path, protocol_hash = _write_or_validate_protocol(
            output_dir, protocol
        )
        print(f"Protocol SHA256: {protocol_hash}", flush=True)
        if args.prepare_only:
            return
        if args.split == "confirmatory":
            _recover_result_publication(
                output_dir, "discovery", spec, protocol_hash, protocol
            )
            _require_confirmatory_unlock(
                output_dir, spec, protocol_hash, protocol
            )

        publication_state = _recover_result_publication(
            output_dir, args.split, spec, protocol_hash, protocol
        )
        if publication_state != "absent":
            result, gate = _load_published_result(
                output_dir, args.split, spec, protocol_hash, protocol
            )
            print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
            print(
                f"Immutable {args.split} result is already {publication_state}: "
                f"{output_dir / f'{args.split}-result.json'}",
                flush=True,
            )
            if not gate["passed"]:
                raise SystemExit(1)
            return

        cases = [
            case for case in manifest["cases"] if case["split"] == args.split
        ]
        _verify_locked_inputs(protocol, protocol_path, protocol_hash)
        probe = _run_parallel_probe(
            candidate_path=candidate_path,
            candidate_weights=candidate_weights,
            base_path=base_path,
            base_weights=base_weights,
            cases=cases,
            spec=spec,
            devices=args.devices,
            num_threads=args.num_threads,
        )
        gate = build_gate_summary(probe["cases"], spec, args.split)
        result = {
            "runner_version": RUNNER_VERSION,
            "probe_version": PROBE_VERSION,
            "protocol_sha256": protocol_hash,
            "split": args.split,
            "probe": probe,
            "gate": gate,
        }
        _validate_result(result, args.split, spec, protocol_hash, protocol)
        result_path, result_hash = _publish_result(
            output_dir,
            args.split,
            result,
            protocol,
            protocol_path,
            protocol_hash,
            spec,
        )
        print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
        print(f"Saved: {result_path}", flush=True)
        print(f"Result SHA256: {result_hash}", flush=True)
        if not gate["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
