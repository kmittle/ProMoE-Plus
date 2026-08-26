"""Run the locked Base-200K count-preserving routing-cycle gate."""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.util
import json
import multiprocessing
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy
import torch

from analyses.denoising_regret.io import write_json_atomic
from analyses.denoising_regret.probe import (
    _build_model,
    _configure_torch_threads,
    _load_checkpoint_model,
)
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)
from analyses.timestep_utility.cycle_batch import (
    BATCH_VERSION,
    BLOCKS_BY_SPLIT,
    CHECKPOINT_STATE,
    CHECKPOINT_STEP,
    EXACT_BATCH_SIZE,
    EXPECTED_WEIGHTS_SHA256,
    EXPECTED_WEIGHTS_SIZE,
    LOCKED_NUM_THREADS,
    MANIFEST_NAME,
    MODEL_NAME,
    SIGMAS,
    SPLIT_COUNTS,
    aggregate_case_results,
    load_manifest,
    requirements_for_split,
    sha256_file,
)
from analyses.timestep_utility.cycle_probe import (
    ARM_CANDIDATES,
    ARM_NAMES,
    AUDITED_SIX_CANDIDATES,
    COUNT_PRESERVING_ARMS,
    PROBE_VERSION,
    RANDOM_JOINT_TOKENS,
    run_cycle_probe_case,
)
from analyses.timestep_utility.probe import _validate_moe_block_contract


LOCKED_DEVICES = ("cuda:4", "cuda:5", "cuda:6", "cuda:7")
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "analyses"
    / "timestep_utility"
    / "manifests"
    / "count_preserving_cycle_gate_v1.json"
)
STATIC_SOURCE_PATHS = (
    "requirements.txt",
    "analyses/run_count_preserving_cycle_probe_batch.py",
    "analyses/timestep_utility/cycle_batch.py",
    "analyses/timestep_utility/cycle_probe.py",
)
SEAL_VERSION = 1


def _parse_devices(value):
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if devices != LOCKED_DEVICES:
        raise argparse.ArgumentTypeError(
            "The locked gate requires cuda:4,cuda:5,cuda:6,cuda:7"
        )
    return devices


def _json_sha256(payload):
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _module_source_path(module):
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            return None
    try:
        return path.resolve()
    except OSError:
        return None


def _collect_project_source_hashes(runtime_cfg):
    with torch.random.fork_rng(devices=[]):
        model = _build_model(runtime_cfg)
    contracts = _validate_moe_block_contract(model, (1, 5, 11))
    model_metadata = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "block_contract": contracts,
    }
    del model
    gc.collect()

    project_root = PROJECT_ROOT.resolve()
    relative_paths = set(STATIC_SOURCE_PATHS)
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        source_path = _module_source_path(module)
        if source_path is None or not source_path.is_file():
            continue
        try:
            relative = source_path.relative_to(project_root)
        except ValueError:
            continue
        relative_paths.add(relative.as_posix())
    hashes = {}
    for relative in sorted(relative_paths):
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Locked source is missing: {path}")
        hashes[relative] = sha256_file(path)
    return model_metadata, hashes


def _git_contract():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("Prepare the protocol only from a clean committed tree")
    upstream = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", "origin/repa...HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if upstream != "0\t0":
        raise RuntimeError("Gate commit must already be pushed to origin/repa")
    return {"commit": commit, "origin_repa_divergence": upstream}


def _runtime_environment(devices):
    cuda_devices = {}
    for device in devices:
        properties = torch.cuda.get_device_properties(torch.device(device))
        cuda_devices[device] = {
            "name": properties.name,
            "uuid": (
                str(properties.uuid) if hasattr(properties, "uuid") else None
            ),
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_devices": cuda_devices,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def _checkpoint_contract(weights_checkpoint_path):
    path = Path(weights_checkpoint_path)
    if path.stat().st_size != EXPECTED_WEIGHTS_SIZE:
        raise ValueError("Local Base-200K checkpoint size changed")
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Local Base-200K checkpoint hash changed")
    load_kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        checkpoint = torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only")
        checkpoint = torch.load(path, **load_kwargs)
    if checkpoint.get("step") != CHECKPOINT_STEP:
        raise ValueError("Local checkpoint does not contain Base step 200000")
    if CHECKPOINT_STATE not in checkpoint:
        raise KeyError(f"Checkpoint is missing {CHECKPOINT_STATE}")
    del checkpoint
    gc.collect()
    return actual_hash


def _case_protocol_view(case):
    return {
        "split": case["split"],
        "id": case["id"],
        "label": case["label"],
        "seed": case["seed"],
        "synset": case["synset"],
        "latent_relative": case["latent_relative"],
        "latent_sha256": case["latent_sha256"],
    }


def _build_protocol(
    checkpoint_path,
    weights_checkpoint_path,
    config_path,
    manifest,
    output_dir,
    devices,
    runtime_cfg,
    canonical_sha256,
    weights_sha256,
):
    model_metadata, source_hashes = _collect_project_source_hashes(runtime_cfg)
    git = _git_contract()
    assignments = {}
    for split in SPLIT_COUNTS:
        split_cases = [case for case in manifest["cases"] if case["split"] == split]
        assignments[split] = [
            {
                "index": index,
                "case_id": case["id"],
                "device": devices[(index - 1) % len(devices)],
            }
            for index, case in enumerate(split_cases, start=1)
        ]
    return {
        "batch_version": BATCH_VERSION,
        "probe_version": PROBE_VERSION,
        "locked_before_any_cycle_result": True,
        "hypothesis": (
            "A first-order downstream denoising-utility label can identify "
            "beneficial token-expert assignment cycles while every routed "
            "expert retains its native token count."
        ),
        "decision_rule": (
            "Plumbing exposes no efficacy statistics. Discovery unlocks "
            "confirmation only when a pre-registered fixed-count arm passes. "
            "Confirmation authorizes only a small router-fitting prototype; "
            "a second checkpoint or seed gate is still required before long training."
        ),
        "novelty_boundary": (
            "Alternating assignment cycles and fixed margins are classical. "
            "This gate tests only downstream-utility structured preference "
            "learning over fixed expert-count routing perturbations."
        ),
        "checkpoint": {
            "canonical_path": str(checkpoint_path),
            "canonical_size": checkpoint_path.stat().st_size,
            "canonical_mtime_ns": checkpoint_path.stat().st_mtime_ns,
            "canonical_sha256": canonical_sha256,
            "canonical_expected_sha256": EXPECTED_WEIGHTS_SHA256,
            "weights_path": str(weights_checkpoint_path),
            "weights_size": weights_checkpoint_path.stat().st_size,
            "weights_sha256": weights_sha256,
            "step": CHECKPOINT_STEP,
            "state": CHECKPOINT_STATE,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "model_name": MODEL_NAME,
        },
        "manifest": {
            "name": MANIFEST_NAME,
            "path": manifest["path"],
            "sha256": manifest["sha256"],
            "selection": manifest["selection"],
            "latent_root": str(
                Path(manifest["cases"][0]["latent"]).parents[1]
            ),
            "cases": [_case_protocol_view(case) for case in manifest["cases"]],
        },
        "settings": {
            "sigmas": list(SIGMAS),
            "blocks_by_split": {
                split: list(blocks) for split, blocks in BLOCKS_BY_SPLIT.items()
            },
            "arm_names": list(ARM_NAMES),
            "count_preserving_arms": list(COUNT_PRESERVING_ARMS),
            "candidates_per_arm": ARM_CANDIDATES,
            "random_joint_tokens": RANDOM_JOINT_TOKENS,
            "audited_six_candidates": AUDITED_SIX_CANDIDATES,
            "exact_batch_size": EXACT_BATCH_SIZE,
            "native_reference": (
                "duplicate native rows at the exact batch shape; row 0 alone "
                "defines loss, suffix gradient, router state, and prediction"
            ),
            "route_weight": "native top-1 identity weight",
            "candidate_generation": (
                "locked RNG samples without replacement from the complete legal "
                "eight-token fixed-count random-joint signature space using only "
                "native route IDs; no VJP, exact loss, or teacher feature may "
                "influence candidate banks"
            ),
            "requirements": {
                split: requirements_for_split(split) for split in SPLIT_COUNTS
            },
        },
        "assignments": assignments,
        "model_metadata": model_metadata,
        "project_source_sha256": source_hashes,
        "git": git,
        "environment": _runtime_environment(devices),
        "output_dir": str(output_dir),
    }


def _write_or_validate_protocol(output_dir, protocol):
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    protocol_hash_path = output_dir / "protocol.sha256"
    protocol_sha256 = _json_sha256(protocol)
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("Existing protocol differs from locked inputs")
    else:
        write_json_atomic(protocol_path, protocol)
    expected_hash_text = protocol_sha256 + "\n"
    if protocol_hash_path.exists():
        if protocol_hash_path.read_text(encoding="utf-8") != expected_hash_text:
            raise RuntimeError("Existing protocol hash file is incompatible")
    else:
        temporary = protocol_hash_path.with_suffix(".sha256.tmp")
        temporary.write_text(expected_hash_text, encoding="utf-8")
        os.replace(temporary, protocol_hash_path)
    return protocol_path, protocol_sha256


def _verify_protocol_inputs(protocol):
    checkpoint = protocol["checkpoint"]
    canonical_path = Path(checkpoint["canonical_path"])
    canonical_stat = canonical_path.stat()
    if (
        canonical_stat.st_size != checkpoint["canonical_size"]
        or canonical_stat.st_mtime_ns != checkpoint["canonical_mtime_ns"]
    ):
        raise RuntimeError("Canonical checkpoint changed after protocol lock")
    if (
        checkpoint["canonical_sha256"]
        != checkpoint["canonical_expected_sha256"]
        or checkpoint["canonical_sha256"] != checkpoint["weights_sha256"]
    ):
        raise RuntimeError("Canonical and local checkpoint hashes are inconsistent")
    weights_path = Path(checkpoint["weights_path"])
    if weights_path.stat().st_size != checkpoint["weights_size"]:
        raise RuntimeError("Weights checkpoint size changed after protocol lock")
    if sha256_file(weights_path) != checkpoint["weights_sha256"]:
        raise RuntimeError("Weights checkpoint hash changed after protocol lock")
    if sha256_file(checkpoint["config"]) != checkpoint["config_sha256"]:
        raise RuntimeError("Config changed after protocol lock")
    manifest = protocol["manifest"]
    if sha256_file(manifest["path"]) != manifest["sha256"]:
        raise RuntimeError("Manifest changed after protocol lock")
    for case in manifest["cases"]:
        latent_path = Path(manifest["latent_root"]) / case["latent_relative"]
        if not latent_path.is_file():
            raise RuntimeError(f"Locked latent is missing: {latent_path}")
        if sha256_file(latent_path) != case["latent_sha256"]:
            raise RuntimeError(f"Locked latent changed: {latent_path}")
    for relative, expected_hash in protocol["project_source_sha256"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Locked project source changed: {relative}")
    if _git_contract() != protocol["git"]:
        raise RuntimeError("Git commit or upstream state changed after protocol lock")


def _load_locked_protocol(protocol_path, expected_sha256):
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if _json_sha256(protocol) != expected_sha256:
        raise RuntimeError("On-disk protocol content hash changed")
    hash_path = protocol_path.with_suffix(".sha256")
    if hash_path.read_text(encoding="utf-8") != expected_sha256 + "\n":
        raise RuntimeError("On-disk protocol hash sidecar changed")
    _verify_protocol_inputs(protocol)
    return protocol


def _result_path(output_dir, split, index, case):
    return output_dir / split / f"{index:02d}_{case['id']}.json"


def _seal_path(result_path):
    return result_path.with_suffix(result_path.suffix + ".seal.json")


def _seal_payload(result, protocol_sha256, case_id):
    return {
        "version": SEAL_VERSION,
        "case_id": case_id,
        "protocol_sha256": protocol_sha256,
        "result_sha256": _json_sha256(result),
    }


def _validate_result(result, case, split, protocol_sha256):
    if result.get("cycle_probe_version") != PROBE_VERSION:
        raise RuntimeError("Case result probe version changed")
    if result.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("Case result belongs to another protocol")
    if result.get("batch_case") != _case_protocol_view(case):
        raise RuntimeError("Case result metadata differs from the locked case")
    if result.get("block_indices") != list(BLOCKS_BY_SPLIT[split]):
        raise RuntimeError("Case result block list differs from the locked split")
    if result.get("sigmas") != list(SIGMAS):
        raise RuntimeError("Case result sigma list differs from the locked gate")
    return result


def _load_sealed_result(result_path, case, split, protocol_sha256):
    seal_path = _seal_path(result_path)
    if not result_path.exists() and not seal_path.exists():
        return None
    if result_path.exists() and not seal_path.exists():
        pending_seal = result_path.with_suffix(result_path.suffix + ".pending.seal.json")
        if pending_seal.exists():
            seal = json.loads(pending_seal.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if seal == _seal_payload(result, protocol_sha256, case["id"]):
                os.replace(pending_seal, seal_path)
            else:
                raise RuntimeError(f"Unsealed incompatible result: {result_path}")
        else:
            raise RuntimeError(f"Unsealed result requires inspection: {result_path}")
    if seal_path.exists() and not result_path.exists():
        raise RuntimeError(f"Result seal exists without payload: {seal_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal != _seal_payload(result, protocol_sha256, case["id"]):
        raise RuntimeError(f"Result seal mismatch: {result_path}")
    return _validate_result(result, case, split, protocol_sha256)


def _publish_result(result_path, result, protocol_sha256, case_id):
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = result_path.with_suffix(result_path.suffix + ".pending")
    pending_seal_path = result_path.with_suffix(
        result_path.suffix + ".pending.seal.json"
    )
    write_json_atomic(pending_path, result)
    seal = _seal_payload(result, protocol_sha256, case_id)
    write_json_atomic(pending_seal_path, seal)
    persisted = json.loads(pending_path.read_text(encoding="utf-8"))
    if persisted != result or seal != _seal_payload(persisted, protocol_sha256, case_id):
        raise RuntimeError("Pending result failed its content seal")
    os.replace(pending_path, result_path)
    os.replace(pending_seal_path, _seal_path(result_path))


def _run_device_cases(payload):
    device = torch.device(payload["device"])
    torch.cuda.set_device(device)
    thread_config = _configure_torch_threads(LOCKED_NUM_THREADS)
    protocol = _load_locked_protocol(
        payload["protocol"],
        payload["protocol_sha256"],
    )
    runtime_cfg = load_runtime_cfg(payload["config"])
    model, state_name, weights_step, load_seconds = _load_checkpoint_model(
        runtime_cfg,
        payload["weights_checkpoint"],
        device,
    )
    if state_name != CHECKPOINT_STATE or weights_step != CHECKPOINT_STEP:
        raise RuntimeError("Worker loaded the wrong checkpoint state or step")
    completed = []
    try:
        for job in payload["jobs"]:
            case = job["case"]
            result_path = Path(job["result_path"])
            reused = _load_sealed_result(
                result_path,
                case,
                payload["split"],
                payload["protocol_sha256"],
            )
            if reused is not None:
                completed.append({"case_id": case["id"], "reused": True})
                continue
            torch.cuda.reset_peak_memory_stats(device)
            result = run_cycle_probe_case(
                model=model,
                runtime_cfg=runtime_cfg,
                latent_path=case["latent"],
                label=case["label"],
                seed=case["seed"],
                block_indices=BLOCKS_BY_SPLIT[payload["split"]],
                sigmas=SIGMAS,
                exact_batch_size=EXACT_BATCH_SIZE,
                latent_key="latent",
            )
            result.update({
                "checkpoint": payload["checkpoint"],
                "weights_checkpoint": payload["weights_checkpoint"],
                "checkpoint_step": CHECKPOINT_STEP,
                "checkpoint_state": state_name,
                "config": payload["config"],
                "model_name": MODEL_NAME,
                "device": str(device),
                "num_threads": LOCKED_NUM_THREADS,
                "thread_config": thread_config,
                "model_load_seconds": float(load_seconds),
                "max_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "batch_case": _case_protocol_view(case),
                "protocol": payload["protocol"],
                "protocol_sha256": payload["protocol_sha256"],
            })
            if _load_locked_protocol(
                payload["protocol"],
                payload["protocol_sha256"],
            ) != protocol:
                raise RuntimeError(
                    "On-disk protocol changed before case result publication"
                )
            _validate_result(
                result,
                case,
                payload["split"],
                payload["protocol_sha256"],
            )
            _publish_result(
                result_path,
                result,
                payload["protocol_sha256"],
                case["id"],
            )
            completed.append({"case_id": case["id"], "reused": False})
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if _load_locked_protocol(
        payload["protocol"],
        payload["protocol_sha256"],
    ) != protocol:
        raise RuntimeError("On-disk protocol changed while the worker was running")
    return {"device": str(device), "completed": completed}


def _summary_path(output_dir, split):
    return output_dir / f"{split}-summary.json"


def _load_prior_summary(output_dir, split, protocol_sha256):
    path = _summary_path(output_dir, split)
    seal_path = _seal_path(path)
    if not path.is_file() or not seal_path.is_file():
        raise RuntimeError(f"Required {split} summary is missing")
    summary = json.loads(path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal != _seal_payload(summary, protocol_sha256, f"{split}-summary"):
        raise RuntimeError(f"Required {split} summary seal is invalid")
    if summary.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError(f"Required {split} summary uses another protocol")
    return summary


def _require_split_unlock(output_dir, split, protocol_sha256, manifest):
    if split == "plumbing":
        return
    prerequisite = "plumbing" if split == "discovery" else "discovery"
    summary = _load_prior_summary(output_dir, prerequisite, protocol_sha256)
    prerequisite_cases = [
        case for case in manifest["cases"] if case["split"] == prerequisite
    ]
    expected_case_ids = [case["id"] for case in prerequisite_cases]
    if (
        summary.get("batch_version") != BATCH_VERSION
        or summary.get("probe_version") != PROBE_VERSION
        or summary.get("split") != prerequisite
        or summary.get("case_ids") != expected_case_ids
    ):
        raise RuntimeError(f"Required {prerequisite} summary metadata is invalid")
    results = _load_split_results(
        output_dir,
        prerequisite,
        prerequisite_cases,
        protocol_sha256,
    )
    recomputed_gate = aggregate_case_results(results, prerequisite)
    if summary.get("gate") != recomputed_gate:
        raise RuntimeError(f"Required {prerequisite} summary failed recomputation")
    if not recomputed_gate["passed"]:
        raise RuntimeError(f"{prerequisite} gate did not unlock {split}")


def _load_split_results(
    output_dir,
    split,
    cases,
    protocol_sha256,
):
    results = []
    for index, case in enumerate(cases, start=1):
        path = _result_path(output_dir, split, index, case)
        result = _load_sealed_result(path, case, split, protocol_sha256)
        if result is None:
            raise RuntimeError(f"Missing completed case result: {path}")
        results.append(result)
    return results


def _publish_summary(output_dir, split, cases, gate, protocol_sha256):
    summary = {
        "batch_version": BATCH_VERSION,
        "probe_version": PROBE_VERSION,
        "split": split,
        "protocol": str(output_dir / "protocol.json"),
        "protocol_sha256": protocol_sha256,
        "case_ids": [case["id"] for case in cases],
        "gate": gate,
    }
    path = _summary_path(output_dir, split)
    existing = _load_prior_summary(output_dir, split, protocol_sha256) if (
        path.exists() and _seal_path(path).exists()
    ) else None
    if existing is not None:
        if existing != summary:
            raise RuntimeError(f"Existing {split} summary differs on recomputation")
        return path
    _publish_result(path, summary, protocol_sha256, f"{split}-summary")
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare or run the locked Base-200K routing-cycle gate."
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--weights-ckpt", required=True)
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=LOCKED_DEVICES,
    )
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_COUNTS),
        default="plumbing",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    checkpoint_path = Path(args.ckpt).resolve()
    weights_checkpoint_path = Path(args.weights_ckpt).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not checkpoint_path.is_file() or not weights_checkpoint_path.is_file():
        raise FileNotFoundError("Canonical and local weights checkpoints must exist")
    if checkpoint_path.stat().st_size != EXPECTED_WEIGHTS_SIZE:
        raise ValueError("Canonical Base-200K checkpoint size changed")
    if parse_checkpoint_step(checkpoint_path) != CHECKPOINT_STEP:
        raise ValueError("Canonical gate checkpoint must be step 200000")
    canonical_sha256 = sha256_file(checkpoint_path)
    if canonical_sha256 != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Canonical Base-200K checkpoint hash changed")
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    runtime_cfg = load_runtime_cfg(config_path)
    if runtime_cfg.model_name != MODEL_NAME:
        raise ValueError(f"Gate model must be {MODEL_NAME}")
    manifest = load_manifest(args.manifest, args.latent_root, PROJECT_ROOT)
    weights_sha256 = _checkpoint_contract(weights_checkpoint_path)
    if canonical_sha256 != weights_sha256:
        raise ValueError("Canonical and local Base-200K checkpoints differ")
    protocol = _build_protocol(
        checkpoint_path=checkpoint_path,
        weights_checkpoint_path=weights_checkpoint_path,
        config_path=config_path,
        manifest=manifest,
        output_dir=output_dir,
        devices=args.devices,
        runtime_cfg=runtime_cfg,
        canonical_sha256=canonical_sha256,
        weights_sha256=weights_sha256,
    )
    protocol_path, protocol_sha256 = _write_or_validate_protocol(
        output_dir,
        protocol,
    )
    print(f"Locked protocol: {protocol_path}")
    print(f"Protocol SHA256: {protocol_sha256}")
    if args.prepare_only:
        return

    split_cases = [
        case for case in manifest["cases"] if case["split"] == args.split
    ]
    lock_path = output_dir / ".orchestration.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another cycle-gate orchestrator is running") from error
        _require_split_unlock(
            output_dir,
            args.split,
            protocol_sha256,
            manifest,
        )
        common = {
            "checkpoint": str(checkpoint_path),
            "weights_checkpoint": str(weights_checkpoint_path),
            "config": str(config_path),
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_sha256,
            "split": args.split,
        }
        payloads = []
        for device in args.devices:
            jobs = []
            for index, case in enumerate(split_cases, start=1):
                if args.devices[(index - 1) % len(args.devices)] == device:
                    jobs.append({
                        "case": case,
                        "result_path": str(_result_path(
                            output_dir,
                            args.split,
                            index,
                            case,
                        )),
                    })
            payloads.append({**common, "device": device, "jobs": jobs})

        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(payloads),
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(_run_device_cases, payload) for payload in payloads
            ]
            for future in as_completed(futures):
                print(json.dumps(future.result(), sort_keys=True), flush=True)

        results = _load_split_results(
            output_dir,
            args.split,
            split_cases,
            protocol_sha256,
        )
        gate = aggregate_case_results(results, args.split)
        if _load_locked_protocol(protocol_path, protocol_sha256) != protocol:
            raise RuntimeError("On-disk protocol changed before summary publication")
        summary_path = _publish_summary(
            output_dir,
            args.split,
            split_cases,
            gate,
            protocol_sha256,
        )
        print(json.dumps({
            "split": args.split,
            "safety_passed": gate["safety_passed"],
            "passed": gate["passed"],
            "authorized_arms": gate.get("authorized_arms", []),
            "recommended_arm": gate.get("recommended_arm"),
        }, indent=2, sort_keys=True))
        print(f"Saved: {summary_path}")
        if not gate["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
