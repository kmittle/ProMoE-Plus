#!/usr/bin/env python3
"""Run the locked Base-200K routed-expert learning-credit gate."""

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
from analyses.timestep_utility.credit_balance_batch import (
    BATCH_VERSION,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CHECKPOINT_STATE,
    CHECKPOINT_STEP,
    EXPECTED_WEIGHTS_SHA256,
    EXPECTED_WEIGHTS_SIZE,
    LOCKED_NUM_THREADS,
    MANIFEST_NAME,
    MODEL_NAME,
    PREREGISTER_PATH,
    PREREGISTER_SHA256,
    SPLIT_COUNTS,
    aggregate_credit_balance,
    case_protocol_view,
    select_cases,
    sha256_file,
)
from analyses.timestep_utility.credit_balance_probe import (
    BLOCKS,
    DUPLICATE_BATCH_SIZE,
    PERMUTATION_RESAMPLES,
    PROBE_VERSION,
    SELECTION_SALT,
    SIGMAS,
    run_credit_balance_case,
)
from analyses.timestep_utility.probe import _validate_moe_block_contract


DEFAULT_CHECKPOINT = (
    "outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/"
    "checkpoints/ckpt_step_200000.pth"
)
DEFAULT_WEIGHTS_CHECKPOINT = (
    "/home/dev/promoe-probes/base-seed0-ckpt_step_200000.pth"
)
DEFAULT_LATENT_ROOT = "/home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz"
DEFAULT_OUTPUT_DIR = "/home/dev/promoe-probes/credit-balance-gate-base200k-v1"
LOCKED_DEVICES = ("cuda:4", "cuda:5", "cuda:6", "cuda:7")
STATIC_SOURCE_PATHS = (
    "requirements.txt",
    "analyses/run_learning_credit_balance_probe_batch.py",
    "analyses/timestep_utility/credit_balance_probe.py",
    "analyses/timestep_utility/credit_balance_batch.py",
    "analyses/denoising_regret/probe.py",
    "analyses/timestep_utility/probe.py",
    "models/models_ProMoE_TC.py",
)
SEAL_VERSION = 1
PLUMBING_CELL_KEYS = frozenset({
    "block_index",
    "sigma",
    "numerical_controls",
})


def _parse_devices(value):
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if devices != LOCKED_DEVICES:
        raise argparse.ArgumentTypeError(
            "The locked gate requires cuda:4,cuda:5,cuda:6,cuda:7"
        )
    return devices


def _json_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    contracts = _validate_moe_block_contract(model, BLOCKS)
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
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("Prepare the credit protocol only from a clean tree")
    divergence = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", "origin/repa...HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if divergence != "0\t0":
        raise RuntimeError("Credit-gate commit must already be pushed to origin/repa")
    return {"commit": commit, "origin_repa_divergence": divergence}


def _runtime_environment(devices):
    cuda_devices = {}
    for device in devices:
        properties = torch.cuda.get_device_properties(torch.device(device))
        cuda_devices[device] = {
            "name": properties.name,
            "uuid": str(properties.uuid) if hasattr(properties, "uuid") else None,
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
        raise ValueError("Local checkpoint is not Base step 200000")
    if CHECKPOINT_STATE not in checkpoint:
        raise KeyError(f"Checkpoint is missing {CHECKPOINT_STATE}")
    del checkpoint
    gc.collect()
    return actual_hash


def _build_protocol(
    args,
    checkpoint_path,
    weights_checkpoint_path,
    config_path,
    runtime_cfg,
    cases,
    canonical_sha256,
    weights_sha256,
):
    if sha256_file(PREREGISTER_PATH) != PREREGISTER_SHA256:
        raise RuntimeError("Credit-balance preregistration changed")
    model_metadata, source_hashes = _collect_project_source_hashes(runtime_cfg)
    assignments = {}
    for split in SPLIT_COUNTS:
        split_cases = [case for case in cases if case["split"] == split]
        assignments[split] = [
            {
                "index": index,
                "case_id": case["id"],
                "device": args.devices[(index - 1) % len(args.devices)],
            }
            for index, case in enumerate(split_cases, start=1)
        ]
    return {
        "batch_version": BATCH_VERSION,
        "probe_version": PROBE_VERSION,
        "manifest_name": MANIFEST_NAME,
        "locked_before_any_credit_probe_case_result": True,
        "hypothesis": (
            "Balanced token counts can hide persistent imbalance in the "
            "suffix-gradient energy delivered to routed experts."
        ),
        "claim_boundary": (
            "This frozen checkpoint gate measures count-credit mismatch and "
            "persistence; it does not establish improved optimization or FID."
        ),
        "preregister": {
            "path": PREREGISTER_PATH,
            "sha256": PREREGISTER_SHA256,
        },
        "checkpoint": {
            "canonical_path": str(checkpoint_path),
            "canonical_size": checkpoint_path.stat().st_size,
            "canonical_mtime_ns": checkpoint_path.stat().st_mtime_ns,
            "canonical_sha256": canonical_sha256,
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
            "selection_salt": SELECTION_SALT,
            "selection_rule": (
                "Sort SHA256(salt|integer_label), assign 8/32/64 class-disjoint "
                "labels, then select the first latent after sorting "
                "SHA256(salt|relative_path)."
            ),
            "latent_root": str(Path(args.latent_root).resolve()),
            "cases": [case_protocol_view(case) for case in cases],
        },
        "settings": {
            "blocks_zero_based": list(BLOCKS),
            "sigmas": list(SIGMAS),
            "duplicate_batch_size": DUPLICATE_BATCH_SIZE,
            "permutation_resamples_per_cell": PERMUTATION_RESAMPLES,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "credit": "native_route_weight^2 * squared_suffix_gradient_norm",
            "shared_and_unconditional_scope": "excluded from routed-expert credit",
        },
        "assignments": assignments,
        "model_metadata": model_metadata,
        "project_source_sha256": source_hashes,
        "git": _git_contract(),
        "environment": _runtime_environment(args.devices),
        "output_dir": str(Path(args.output_dir).resolve()),
    }


def _write_or_validate_protocol(output_dir, protocol):
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    hash_path = output_dir / "protocol.sha256"
    protocol_sha256 = _json_sha256(protocol)
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("Existing credit protocol differs from locked inputs")
    else:
        write_json_atomic(protocol_path, protocol)
    expected_text = protocol_sha256 + "\n"
    if hash_path.exists():
        if hash_path.read_text(encoding="utf-8") != expected_text:
            raise RuntimeError("Existing credit protocol hash is incompatible")
    else:
        temporary = hash_path.with_suffix(".sha256.tmp")
        temporary.write_text(expected_text, encoding="utf-8")
        os.replace(temporary, hash_path)
    return protocol_path, protocol_sha256


def _assert_protocol_unchanged(protocol_path, protocol_sha256):
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if _json_sha256(protocol) != protocol_sha256:
        raise RuntimeError("On-disk credit protocol content changed")
    hash_path = Path(protocol_path).with_suffix(".sha256")
    if hash_path.read_text(encoding="utf-8") != protocol_sha256 + "\n":
        raise RuntimeError("On-disk credit protocol hash sidecar changed")
    return protocol


def _verify_source_hashes(protocol):
    for relative, expected in protocol["project_source_sha256"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Locked project source changed: {relative}")


def _verify_protocol_inputs(protocol):
    checkpoint = protocol["checkpoint"]
    canonical_path = Path(checkpoint["canonical_path"])
    canonical_stat = canonical_path.stat()
    if (
        canonical_stat.st_size != checkpoint["canonical_size"]
        or canonical_stat.st_mtime_ns != checkpoint["canonical_mtime_ns"]
    ):
        raise RuntimeError("Canonical checkpoint changed after protocol lock")
    weights_path = Path(checkpoint["weights_path"])
    if weights_path.stat().st_size != checkpoint["weights_size"]:
        raise RuntimeError("Weights checkpoint size changed after protocol lock")
    if sha256_file(weights_path) != checkpoint["weights_sha256"]:
        raise RuntimeError("Weights checkpoint hash changed after protocol lock")
    if checkpoint["canonical_sha256"] != checkpoint["weights_sha256"]:
        raise RuntimeError("Canonical and local checkpoint hashes differ")
    if sha256_file(checkpoint["config"]) != checkpoint["config_sha256"]:
        raise RuntimeError("Checkpoint config changed after protocol lock")
    preregister = protocol["preregister"]
    if sha256_file(preregister["path"]) != preregister["sha256"]:
        raise RuntimeError("Credit preregistration changed after protocol lock")
    latent_root = Path(protocol["manifest"]["latent_root"])
    for case in protocol["manifest"]["cases"]:
        path = latent_root / case["latent_relative"]
        if not path.is_file() or sha256_file(path) != case["latent_sha256"]:
            raise RuntimeError(f"Locked latent changed: {path}")
    _verify_source_hashes(protocol)
    if _git_contract() != protocol["git"]:
        raise RuntimeError("Git commit or upstream state changed after protocol lock")


def _result_path(output_dir, split, index, case_id):
    return Path(output_dir) / split / f"{index:03d}_{case_id}.json"


def _seal_path(result_path):
    return Path(result_path).with_suffix(Path(result_path).suffix + ".seal.json")


def _seal_payload(result, protocol_sha256, case_id):
    return {
        "version": SEAL_VERSION,
        "case_id": case_id,
        "protocol_sha256": protocol_sha256,
        "result_sha256": _json_sha256(result),
    }


def _validate_result(result, case, protocol_sha256):
    if result.get("credit_balance_probe_version") != PROBE_VERSION:
        raise RuntimeError("Case result probe version changed")
    if result.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("Case result belongs to another protocol")
    if result.get("batch_case") != case_protocol_view(case):
        raise RuntimeError("Case result metadata differs from the locked case")
    if result.get("block_indices") != list(BLOCKS):
        raise RuntimeError("Case result block list changed")
    if result.get("sigmas") != list(SIGMAS):
        raise RuntimeError("Case result sigma list changed")
    cells = result.get("cells")
    if not isinstance(cells, list):
        raise RuntimeError("Case result cells are missing")
    expected_cells = {(block, sigma) for block in BLOCKS for sigma in SIGMAS}
    observed_cells = {
        (int(cell["block_index"]), float(cell["sigma"])) for cell in cells
    }
    if len(cells) != len(expected_cells) or observed_cells != expected_cells:
        raise RuntimeError("Case result block/sigma cells changed")
    return result


def _result_for_publish(result, split):
    if split != "plumbing":
        return result
    return {
        **result,
        "efficacy_hidden": True,
        "cells": [
            {
                "block_index": int(cell["block_index"]),
                "sigma": float(cell["sigma"]),
                "numerical_controls": dict(cell["numerical_controls"]),
            }
            for cell in result["cells"]
        ],
    }


def _validate_published_result(result, case, protocol_sha256):
    _validate_result(result, case, protocol_sha256)
    if case["split"] == "plumbing":
        if result.get("efficacy_hidden") is not True:
            raise RuntimeError("Published plumbing result is not efficacy-hidden")
        for cell in result["cells"]:
            if set(cell) != PLUMBING_CELL_KEYS:
                raise RuntimeError("Published plumbing cell leaks efficacy fields")
    else:
        for cell in result["cells"]:
            if "statistics" not in cell or "numerical_controls" not in cell:
                raise RuntimeError("Published efficacy cell is incomplete")
    return result


def _load_sealed_result(result_path, case, protocol_sha256):
    result_path = Path(result_path)
    seal_path = _seal_path(result_path)
    if not result_path.exists() and not seal_path.exists():
        return None
    if result_path.exists() != seal_path.exists():
        raise RuntimeError(f"Partial sealed result requires inspection: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal != _seal_payload(result, protocol_sha256, case["id"]):
        raise RuntimeError(f"Result seal mismatch: {result_path}")
    return _validate_published_result(result, case, protocol_sha256)


def _publish_result(result_path, result, protocol_sha256, case_id):
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = result_path.with_suffix(result_path.suffix + ".pending")
    pending_seal = result_path.with_suffix(result_path.suffix + ".pending.seal.json")
    write_json_atomic(pending_path, result)
    seal = _seal_payload(result, protocol_sha256, case_id)
    write_json_atomic(pending_seal, seal)
    persisted = json.loads(pending_path.read_text(encoding="utf-8"))
    if persisted != result or seal != _seal_payload(persisted, protocol_sha256, case_id):
        raise RuntimeError("Pending credit result failed its content seal")
    os.replace(pending_path, result_path)
    os.replace(pending_seal, _seal_path(result_path))


def _run_device_cases(payload):
    device = torch.device(payload["device"])
    torch.cuda.set_device(device)
    thread_config = _configure_torch_threads(LOCKED_NUM_THREADS)
    protocol = _assert_protocol_unchanged(
        payload["protocol"],
        payload["protocol_sha256"],
    )
    _verify_source_hashes(protocol)
    runtime_cfg = load_runtime_cfg(payload["config"])
    model, state_name, weights_step, load_seconds = _load_checkpoint_model(
        runtime_cfg,
        payload["weights_checkpoint"],
        device,
    )
    if state_name != CHECKPOINT_STATE or weights_step != CHECKPOINT_STEP:
        raise RuntimeError("Worker loaded the wrong checkpoint state or step")
    completed = []
    latent_root = Path(protocol["manifest"]["latent_root"])
    try:
        for job in payload["jobs"]:
            case = job["case"]
            result_path = Path(job["result_path"])
            reused = _load_sealed_result(
                result_path,
                case,
                payload["protocol_sha256"],
            )
            if reused is not None:
                completed.append({"case_id": case["id"], "reused": True})
                continue
            torch.cuda.reset_peak_memory_stats(device)
            result = run_credit_balance_case(
                model=model,
                runtime_cfg=runtime_cfg,
                latent_path=latent_root / case["latent_relative"],
                label=case["label"],
                seed=case["seed"],
                case_id=case["id"],
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
                "batch_case": case_protocol_view(case),
                "protocol": payload["protocol"],
                "protocol_sha256": payload["protocol_sha256"],
            })
            _assert_protocol_unchanged(
                payload["protocol"],
                payload["protocol_sha256"],
            )
            _verify_source_hashes(protocol)
            _validate_result(result, case, payload["protocol_sha256"])
            result = _result_for_publish(result, case["split"])
            _validate_published_result(
                result,
                case,
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
    return {"device": str(device), "completed": completed}


def _summary_path(output_dir, split):
    return Path(output_dir) / f"{split}-summary.json"


def _load_summary(output_dir, split, protocol_sha256):
    path = _summary_path(output_dir, split)
    seal_path = _seal_path(path)
    if not path.is_file() or not seal_path.is_file():
        raise RuntimeError(f"Required {split} summary is missing")
    summary = json.loads(path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal != _seal_payload(summary, protocol_sha256, f"{split}-summary"):
        raise RuntimeError(f"Required {split} summary seal is invalid")
    return summary


def _load_split_results(output_dir, split, cases, protocol_sha256):
    results = []
    for index, case in enumerate(cases, start=1):
        result = _load_sealed_result(
            _result_path(output_dir, split, index, case["id"]),
            case,
            protocol_sha256,
        )
        if result is None:
            raise RuntimeError(f"Missing completed {split} case: {case['id']}")
        results.append(result)
    return results


def _require_split_unlock(output_dir, split, cases, protocol_sha256):
    if split == "plumbing":
        return None
    prerequisite = "plumbing" if split == "discovery" else "discovery"
    prerequisite_cases = [case for case in cases if case["split"] == prerequisite]
    results = _load_split_results(
        output_dir,
        prerequisite,
        prerequisite_cases,
        protocol_sha256,
    )
    discovery_summary = None
    if prerequisite == "discovery":
        discovery_summary = _load_summary(output_dir, "discovery", protocol_sha256)[
            "gate"
        ]
    recomputed = aggregate_credit_balance(
        results,
        prerequisite,
        discovery_summary=discovery_summary,
    )
    published = _load_summary(output_dir, prerequisite, protocol_sha256)
    if published.get("gate") != recomputed:
        raise RuntimeError(f"Required {prerequisite} summary failed recomputation")
    if not recomputed["passed"]:
        raise RuntimeError(f"{prerequisite} gate did not unlock {split}")
    return recomputed


def _publish_summary(output_dir, split, split_cases, gate, protocol_sha256):
    summary = {
        "batch_version": BATCH_VERSION,
        "probe_version": PROBE_VERSION,
        "split": split,
        "protocol": str(Path(output_dir) / "protocol.json"),
        "protocol_sha256": protocol_sha256,
        "case_ids": [case["id"] for case in split_cases],
        "gate": gate,
    }
    path = _summary_path(output_dir, split)
    if path.exists() or _seal_path(path).exists():
        existing = _load_summary(output_dir, split, protocol_sha256)
        if existing != summary:
            raise RuntimeError(f"Existing {split} summary differs on recomputation")
        return path
    _publish_result(path, summary, protocol_sha256, f"{split}-summary")
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the locked Base-200K routed-expert learning-credit gate."
    )
    parser.add_argument("--ckpt", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--weights-ckpt", default=DEFAULT_WEIGHTS_CHECKPOINT)
    parser.add_argument("--latent-root", default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
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
        raise ValueError("Credit gate requires Base step 200000")
    canonical_sha256 = sha256_file(checkpoint_path)
    if canonical_sha256 != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Canonical Base-200K checkpoint hash changed")
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    runtime_cfg = load_runtime_cfg(config_path)
    if runtime_cfg.model_name != MODEL_NAME:
        raise ValueError(f"Credit gate model must be {MODEL_NAME}")
    weights_sha256 = _checkpoint_contract(weights_checkpoint_path)
    if weights_sha256 != canonical_sha256:
        raise ValueError("Canonical and local Base-200K checkpoints differ")
    cases = select_cases(args.latent_root)
    protocol = _build_protocol(
        args=args,
        checkpoint_path=checkpoint_path,
        weights_checkpoint_path=weights_checkpoint_path,
        config_path=config_path,
        runtime_cfg=runtime_cfg,
        cases=cases,
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

    _verify_protocol_inputs(protocol)
    split_cases = [case for case in cases if case["split"] == args.split]
    lock_path = output_dir / ".orchestration.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another credit-gate orchestrator is running") from error
        discovery_gate = _require_split_unlock(
            output_dir,
            args.split,
            cases,
            protocol_sha256,
        )
        common = {
            "checkpoint": str(checkpoint_path),
            "weights_checkpoint": str(weights_checkpoint_path),
            "config": str(config_path),
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_sha256,
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
                            case["id"],
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
        gate = aggregate_credit_balance(
            results,
            args.split,
            discovery_summary=discovery_gate if args.split == "confirmatory" else None,
        )
        _assert_protocol_unchanged(protocol_path, protocol_sha256)
        _verify_source_hashes(protocol)
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
            "efficacy_passed": gate.get("efficacy_passed"),
            "passed": gate["passed"],
        }, indent=2, sort_keys=True))
        print(f"Saved: {summary_path}")
        if not gate["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
