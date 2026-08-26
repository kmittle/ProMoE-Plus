"""Run the locked natural-input timestep-utility gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import multiprocessing
import os
import platform
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
from analyses.denoising_regret.probe import _build_model
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)
from analyses.timestep_utility.batch import (
    BATCH_VERSION,
    BLOCK_INDICES,
    CAPACITY_FACTOR,
    CHECKPOINT_STATE,
    CHECKPOINT_STEP,
    EXACT_BATCH_SIZE,
    EXPECTED_WEIGHTS_SHA256,
    MODEL_NAME,
    NUM_TOKEN_PROBES,
    SENSITIVITY_TOKEN_COUNT,
    SIGMAS,
    SPLIT_COUNTS,
    aggregate_case_results,
    load_manifest,
    requirements_for_split,
    sha256_file,
)
from analyses.timestep_utility.probe import (
    PROBE_VERSION,
    _validate_moe_block_contract,
    run_timestep_utility_probe,
)


LOCKED_DEVICES = ("cuda:4", "cuda:5", "cuda:6", "cuda:7")
LOCKED_NUM_THREADS = 4
PENDING_SEAL_VERSION = 1
RESULT_SEAL_VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "analyses"
    / "timestep_utility"
    / "manifests"
    / "natural_timestep_utility_gate_v1.json"
)
STATIC_SOURCE_PATHS = (
    "requirements.txt",
    "analyses/run_timestep_utility_probe_batch.py",
    "analyses/timestep_utility/batch.py",
    "analyses/timestep_utility/probe.py",
)


def _parse_devices(value):
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if devices != LOCKED_DEVICES:
        raise argparse.ArgumentTypeError(
            "The gate is locked to cuda:4,cuda:5,cuda:6,cuda:7"
        )
    return devices


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
    block_contract = _validate_moe_block_contract(model, BLOCK_INDICES)
    model_metadata = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "block_contract": block_contract,
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
            raise FileNotFoundError(f"Locked project source is missing: {path}")
        hashes[relative] = sha256_file(path)
    return model_metadata, hashes


def _runtime_environment(devices):
    cuda_devices = {}
    for device in devices:
        properties = torch.cuda.get_device_properties(torch.device(device))
        cuda_devices[device] = {
            "name": properties.name,
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
    load_kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        checkpoint = torch.load(weights_checkpoint_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only")
        checkpoint = torch.load(weights_checkpoint_path, **load_kwargs)
    step = checkpoint.get("step")
    if step != CHECKPOINT_STEP:
        raise ValueError(
            f"Loaded checkpoint step must be {CHECKPOINT_STEP}, got {step}"
        )
    if CHECKPOINT_STATE not in checkpoint:
        raise KeyError(f"Checkpoint is missing {CHECKPOINT_STATE}")
    del checkpoint
    gc.collect()


def _build_protocol(
    checkpoint_path,
    weights_checkpoint_path,
    config_path,
    manifest,
    output_dir,
    devices,
    runtime_cfg,
    checkpoint_sha256,
    weights_sha256,
):
    model_metadata, source_hashes = _collect_project_source_hashes(runtime_cfg)
    assignments = {
        split: [
            {
                "index": index,
                "case_id": case["id"],
                "device": devices[(index - 1) % len(devices)],
            }
            for index, case in enumerate(
                [case for case in manifest["cases"] if case["split"] == split],
                start=1,
            )
        ]
        for split in SPLIT_COUNTS
    }
    return {
        "batch_version": BATCH_VERSION,
        "probe_version": PROBE_VERSION,
        "locked_before_discovery_results": True,
        "hypotheses": {
            "routing_accuracy": (
                "native prototype affinity does not identify the equal-compute "
                "expert with highest exact denoising utility"
            ),
            "capacity_preserving": (
                "exact denoising MSE improves after reassigning sampled tokens "
                "while preserving the native per-expert count vector"
            ),
            "stage_structure": (
                "expert-utility ranks vary across sigma more than router-affinity "
                "ranks track"
            ),
        },
        "decision_rule": (
            "routing_accuracy_gap_passed authorizes only utility-aware MoE method "
            "design; stage_structure_passed is additionally required before a "
            "timestep-conditioned routing claim"
        ),
        "checkpoint": {
            "canonical_path": str(checkpoint_path),
            "canonical_sha256": checkpoint_sha256,
            "weights_path": str(weights_checkpoint_path),
            "weights_sha256": weights_sha256,
            "step": CHECKPOINT_STEP,
            "state": CHECKPOINT_STATE,
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "model_name": MODEL_NAME,
        },
        "model": model_metadata,
        "design": {
            "input_domain": (
                "untransformed VAE posterior sample plus fixed Gaussian noise"
            ),
            "primary_route_weight": "native",
            "sensitivity_route_weights": ["candidate", "unit"],
            "sigmas": list(SIGMAS),
            "block_indices": list(BLOCK_INDICES),
            "num_token_probes": NUM_TOKEN_PROBES,
            "sensitivity_token_count": SENSITIVITY_TOKEN_COUNT,
            "exact_batch_size": EXACT_BATCH_SIZE,
            "capacity_factor": CAPACITY_FACTOR,
            "num_threads_per_worker": LOCKED_NUM_THREADS,
        },
        "gate_requirements": {
            split: requirements_for_split(split) for split in SPLIT_COUNTS
        },
        "manifest": {
            "path": manifest["path"],
            "sha256": manifest["sha256"],
            "selection": manifest["selection"],
            "cases": [
                {
                    "split": case["split"],
                    "id": case["id"],
                    "label": case["label"],
                    "seed": case["seed"],
                    "synset": case["synset"],
                    "latent_relative": case["latent_relative"],
                    "latent_sha256": case["latent_sha256"],
                }
                for case in manifest["cases"]
            ],
        },
        "output_dir": str(output_dir),
        "devices": list(devices),
        "assignments": assignments,
        "project_source_sha256": source_hashes,
        "environment": _runtime_environment(devices),
    }


def _write_text_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_or_validate_protocol(output_dir, protocol):
    output_dir = Path(output_dir)
    protocol_path = output_dir / "protocol.json"
    hash_path = output_dir / "protocol.sha256"
    existing_results = list((output_dir / "cases").glob("**/*.json"))
    if not protocol_path.exists() and existing_results:
        raise RuntimeError("Refusing result files without a locked protocol")
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError(f"Existing protocol is incompatible: {protocol_path}")
    else:
        write_json_atomic(protocol_path, protocol)
    protocol_sha256 = sha256_file(protocol_path)
    expected_line = f"{protocol_sha256}  protocol.json\n"
    if hash_path.exists():
        if hash_path.read_text(encoding="utf-8") != expected_line:
            raise ValueError(f"Protocol checksum differs: {hash_path}")
    else:
        _write_text_atomic(hash_path, expected_line)
    return protocol_path, protocol_sha256


def _json_payload_sha256(payload):
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _pending_result_path(result_path, device):
    result_path = Path(result_path)
    return result_path.with_suffix(
        result_path.suffix + f".pending.{device.replace(':', '_')}"
    )


def _pending_seal_path(pending_path):
    return Path(f"{pending_path}.seal")


def _pending_seal(pending_sha256, case, payload):
    return {
        "version": PENDING_SEAL_VERSION,
        "case_id": case["id"],
        "latent_sha256": case["latent_sha256"],
        "device": payload["device"],
        "protocol_sha256": payload["protocol_sha256"],
        "pending_sha256": pending_sha256,
    }


def _result_seal_path(result_path):
    return Path(f"{result_path}.seal")


def _result_seal(result_sha256, case, payload):
    return {
        "version": RESULT_SEAL_VERSION,
        "case_id": case["id"],
        "latent_sha256": case["latent_sha256"],
        "device": payload["device"],
        "protocol_sha256": payload["protocol_sha256"],
        "result_sha256": result_sha256,
    }


def _verify_file(path, expected_sha256, description):
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{description} changed after protocol lock: {Path(path).resolve()}"
        )


def _verify_locked_inputs(payload):
    _verify_file(
        payload["checkpoint"],
        payload["checkpoint_sha256"],
        "Canonical checkpoint",
    )
    _verify_file(
        payload["weights_checkpoint"],
        payload["weights_sha256"],
        "Weights checkpoint",
    )
    _verify_file(payload["config"], payload["config_sha256"], "Config")
    _verify_file(payload["manifest"], payload["manifest_sha256"], "Manifest")
    _verify_file(payload["protocol"], payload["protocol_sha256"], "Protocol")
    for relative, expected in payload["source_sha256"].items():
        _verify_file(PROJECT_ROOT / relative, expected, f"Source {relative}")


def _validate_case_result(result, case, payload):
    expected = {
        "timestep_utility_probe_version": PROBE_VERSION,
        "checkpoint": payload["checkpoint"],
        "weights_checkpoint": payload["weights_checkpoint"],
        "checkpoint_step": CHECKPOINT_STEP,
        "weights_checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_state": CHECKPOINT_STATE,
        "config": payload["config"],
        "model_name": MODEL_NAME,
        "latent": case["latent"],
        "latent_key": "latent",
        "label": case["label"],
        "sigmas": list(SIGMAS),
        "block_indices": list(BLOCK_INDICES),
        "num_token_probes": NUM_TOKEN_PROBES,
        "sensitivity_token_count": SENSITIVITY_TOKEN_COUNT,
        "exact_batch_size": EXACT_BATCH_SIZE,
        "capacity_factor": CAPACITY_FACTOR,
        "seed": case["seed"],
        "device": payload["device"],
        "num_threads": LOCKED_NUM_THREADS,
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "weights_checkpoint_sha256": payload["weights_sha256"],
        "config_sha256": payload["config_sha256"],
        "latent_sha256": case["latent_sha256"],
        "protocol_sha256": payload["protocol_sha256"],
        "batch_case": case,
    }
    mismatches = [
        f"{key}: expected {value!r}, found {result.get(key)!r}"
        for key, value in expected.items()
        if result.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Incompatible result for {case['id']}: " + "; ".join(mismatches)
        )
    if len(result["cells"]) != len(SIGMAS) * len(BLOCK_INDICES):
        raise ValueError(f"{case['id']}: cell count differs from protocol")
    for cell in result["cells"]:
        if len(cell["tokens"]) != NUM_TOKEN_PROBES:
            raise ValueError(f"{case['id']}: token count differs from protocol")


def _case_result_path(output_dir, split, index, case):
    return Path(output_dir) / "cases" / split / f"{index:02d}_{case['id']}.json"


def _load_sealed_pending(pending_path, case, payload):
    seal_path = _pending_seal_path(pending_path)
    result = json.loads(Path(pending_path).read_text(encoding="utf-8"))
    _validate_case_result(result, case, payload)
    pending_sha256 = _json_payload_sha256(result)
    _verify_file(
        pending_path,
        pending_sha256,
        f"Sealed pending result for {case['id']}",
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected_seal = _pending_seal(pending_sha256, case, payload)
    if seal != expected_seal:
        raise ValueError(f"Pending seal is incompatible: {seal_path}")
    return result, pending_sha256


def _load_published_result(result_path, case, payload):
    result_path = Path(result_path)
    seal_path = _result_seal_path(result_path)
    if not seal_path.is_file():
        raise RuntimeError(f"Published result has no seal: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _validate_case_result(result, case, payload)
    result_sha256 = _json_payload_sha256(result)
    _verify_file(
        result_path,
        result_sha256,
        f"Published result for {case['id']}",
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected_seal = _result_seal(result_sha256, case, payload)
    if seal != expected_seal:
        raise ValueError(f"Published result seal is incompatible: {seal_path}")
    _verify_file(
        seal_path,
        _json_payload_sha256(expected_seal),
        f"Published result seal for {case['id']}",
    )
    return result


def _run_device_cases(payload):
    torch.cuda.set_device(torch.device(payload["device"]))
    _verify_locked_inputs(payload)
    pending_results = []
    for job in payload["jobs"]:
        case = job["case"]
        result_path = Path(job["result_path"])
        pending_path = _pending_result_path(result_path, payload["device"])
        seal_path = _pending_seal_path(pending_path)
        result_seal_path = _result_seal_path(result_path)
        _verify_file(
            case["latent"],
            case["latent_sha256"],
            f"Latent for {case['id']}",
        )
        if result_path.exists():
            if result_seal_path.exists():
                result = _load_published_result(result_path, case, payload)
            elif pending_path.exists() and seal_path.exists():
                result, result_sha256 = _load_sealed_pending(
                    pending_path,
                    case,
                    payload,
                )
                published = json.loads(result_path.read_text(encoding="utf-8"))
                _validate_case_result(published, case, payload)
                if published != result:
                    raise RuntimeError(
                        f"Published and pending results differ for {case['id']}"
                    )
                _verify_file(
                    result_path,
                    result_sha256,
                    f"Published result for {case['id']}",
                )
                final_seal = _result_seal(result_sha256, case, payload)
                write_json_atomic(result_seal_path, final_seal)
                _verify_file(
                    result_seal_path,
                    _json_payload_sha256(final_seal),
                    f"Published result seal for {case['id']}",
                )
            else:
                raise RuntimeError(f"Published result has no seal: {result_path}")
            if pending_path.exists() and seal_path.exists():
                pending_result, _ = _load_sealed_pending(
                    pending_path,
                    case,
                    payload,
                )
                if pending_result != result:
                    raise RuntimeError(
                        f"Published and pending results differ for {case['id']}"
                    )
                seal_path.unlink()
                pending_path.unlink()
            elif pending_path.exists():
                pending_path.unlink()
            elif seal_path.exists():
                raise RuntimeError(f"Pending seal has no pending result: {seal_path}")
            print(f"[{payload['device']}] reusing {case['id']}", flush=True)
            continue
        if result_seal_path.exists():
            raise RuntimeError(
                f"Published result seal has no result: {result_seal_path}"
            )
        if pending_path.exists():
            if not seal_path.exists():
                pending_path.unlink()
                print(
                    f"[{payload['device']}] discarding unsealed pending "
                    f"{case['id']}",
                    flush=True,
                )
            else:
                result, pending_sha256 = _load_sealed_pending(
                    pending_path,
                    case,
                    payload,
                )
                pending_results.append((
                    pending_path,
                    seal_path,
                    result_path,
                    case,
                    result,
                    pending_sha256,
                ))
                print(
                    f"[{payload['device']}] reusing sealed pending {case['id']}",
                    flush=True,
                )
                continue
        elif seal_path.exists():
            raise RuntimeError(f"Pending seal has no pending result: {seal_path}")
        print(f"[{payload['device']}] probing {case['id']}", flush=True)
        result = run_timestep_utility_probe(
            checkpoint_path=payload["checkpoint"],
            weights_checkpoint_path=payload["weights_checkpoint"],
            latent_path=case["latent"],
            latent_key="latent",
            label=case["label"],
            sigmas=SIGMAS,
            block_indices=BLOCK_INDICES,
            num_token_probes=NUM_TOKEN_PROBES,
            sensitivity_token_count=SENSITIVITY_TOKEN_COUNT,
            exact_batch_size=EXACT_BATCH_SIZE,
            capacity_factor=CAPACITY_FACTOR,
            seed=case["seed"],
            device=payload["device"],
            num_threads=LOCKED_NUM_THREADS,
        )
        result.update({
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "weights_checkpoint_sha256": payload["weights_sha256"],
            "config_sha256": payload["config_sha256"],
            "latent_sha256": case["latent_sha256"],
            "protocol_sha256": payload["protocol_sha256"],
            "batch_case": case,
        })
        _validate_case_result(result, case, payload)
        _verify_file(
            case["latent"],
            case["latent_sha256"],
            f"Latent for {case['id']}",
        )
        pending_sha256 = _json_payload_sha256(result)
        write_json_atomic(pending_path, result)
        _verify_file(
            pending_path,
            pending_sha256,
            f"Pending result for {case['id']}",
        )
        pending_results.append((
            pending_path,
            seal_path,
            result_path,
            case,
            result,
            pending_sha256,
        ))
        print(f"[{payload['device']}] prepared {result_path}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    _verify_locked_inputs(payload)

    for pending in pending_results:
        pending_path, _, _, case, result, pending_sha256 = pending
        _verify_file(
            case["latent"],
            case["latent_sha256"],
            f"Latent for {case['id']}",
        )
        _verify_file(
            pending_path,
            pending_sha256,
            f"Pending result for {case['id']}",
        )
        reloaded = json.loads(pending_path.read_text(encoding="utf-8"))
        _validate_case_result(reloaded, case, payload)
        if reloaded != result:
            raise RuntimeError(f"Pending result snapshot changed for {case['id']}")

    sealed_results = []
    for pending in pending_results:
        pending_path, seal_path, _, case, _, pending_sha256 = pending
        seal = _pending_seal(pending_sha256, case, payload)
        seal_sha256 = _json_payload_sha256(seal)
        if seal_path.exists():
            existing_seal = json.loads(seal_path.read_text(encoding="utf-8"))
            if existing_seal != seal:
                raise ValueError(f"Pending seal is incompatible: {seal_path}")
        else:
            write_json_atomic(seal_path, seal)
        _verify_file(
            seal_path,
            seal_sha256,
            f"Pending seal for {case['id']}",
        )
        sealed_results.append((*pending, seal_sha256))

    for sealed in sealed_results:
        (
            pending_path,
            seal_path,
            result_path,
            case,
            result,
            pending_sha256,
            seal_sha256,
        ) = sealed
        _verify_file(
            pending_path,
            pending_sha256,
            f"Pending result for {case['id']}",
        )
        _verify_file(
            seal_path,
            seal_sha256,
            f"Pending seal for {case['id']}",
        )
        write_json_atomic(result_path, result)
        _verify_file(
            result_path,
            pending_sha256,
            f"Published result for {case['id']}",
        )
        published = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_case_result(published, case, payload)
        final_seal = _result_seal(pending_sha256, case, payload)
        result_seal_path = _result_seal_path(result_path)
        write_json_atomic(result_seal_path, final_seal)
        _verify_file(
            result_seal_path,
            _json_payload_sha256(final_seal),
            f"Published result seal for {case['id']}",
        )
        seal_path.unlink()
        pending_path.unlink()
        print(f"[{payload['device']}] saved {result_path}", flush=True)
    return payload["device"]


def _load_split_results(output_dir, split, cases, devices, common_payload):
    results = []
    for index, case in enumerate(cases, start=1):
        path = _case_result_path(output_dir, split, index, case)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} result: {path}")
        device = devices[(index - 1) % len(devices)]
        result = _load_published_result(
            path,
            case,
            {**common_payload, "device": device},
        )
        results.append(result)
    return results


def _build_split_summary(
    output_dir,
    split,
    cases,
    protocol_path,
    common_payload,
    gate,
):
    case_results = []
    for index, case in enumerate(cases, start=1):
        path = _case_result_path(output_dir, split, index, case)
        seal_path = _result_seal_path(path)
        case_results.append({
            "case_id": case["id"],
            "path": str(path),
            "sha256": sha256_file(path),
            "seal": str(seal_path),
            "seal_sha256": sha256_file(seal_path),
        })
    return {
        "batch_version": BATCH_VERSION,
        "probe_version": PROBE_VERSION,
        "protocol": str(protocol_path),
        "protocol_sha256": common_payload["protocol_sha256"],
        "split": split,
        "checkpoint": common_payload["checkpoint"],
        "checkpoint_sha256": common_payload["checkpoint_sha256"],
        "weights_checkpoint": common_payload["weights_checkpoint"],
        "weights_checkpoint_sha256": common_payload["weights_sha256"],
        "config": common_payload["config"],
        "config_sha256": common_payload["config_sha256"],
        "manifest": common_payload["manifest"],
        "manifest_sha256": common_payload["manifest_sha256"],
        "case_results": case_results,
        "gate": gate,
    }


def _publish_summary(summary_path, summary, common_payload):
    summary_path = Path(summary_path)
    pending_path = summary_path.with_suffix(summary_path.suffix + ".pending")
    expected_sha256 = _json_payload_sha256(summary)
    _verify_locked_inputs(common_payload)
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise ValueError(
                f"Existing split summary differs from recomputation: {summary_path}"
            )
        _verify_file(summary_path, expected_sha256, "Existing split summary")
        pending_path.unlink(missing_ok=True)
        return
    write_json_atomic(pending_path, summary)
    _verify_locked_inputs(common_payload)
    _verify_file(pending_path, expected_sha256, "Pending split summary")
    os.replace(pending_path, summary_path)
    _verify_file(summary_path, expected_sha256, "Published split summary")


def _require_confirmatory_unlock(
    output_dir,
    protocol_path,
    manifest,
    devices,
    common_payload,
):
    summary_path = Path(output_dir) / "discovery-summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            "Confirmatory split is locked until discovery-summary.json exists"
        )
    _verify_locked_inputs(common_payload)
    discovery_cases = [
        case for case in manifest["cases"] if case["split"] == "discovery"
    ]
    discovery_results = _load_split_results(
        output_dir,
        "discovery",
        discovery_cases,
        devices,
        common_payload,
    )
    recomputed_gate = aggregate_case_results(discovery_results, "discovery")
    expected_summary = _build_split_summary(
        output_dir,
        "discovery",
        discovery_cases,
        protocol_path,
        common_payload,
        recomputed_gate,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary != expected_summary:
        raise ValueError(
            "Discovery summary or case-result hashes differ from recomputation"
        )
    if not (
        recomputed_gate["safety_passed"]
        and recomputed_gate["routing_accuracy_gap_passed"]
        and recomputed_gate["passed"]
    ):
        raise RuntimeError(
            "Confirmatory split requires discovery safety and routing gates to pass"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or run the pre-registered natural-input timestep-utility gate."
        )
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
        default="discovery",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write and verify protocol files without running a case",
    )
    return parser


def main():
    args = build_parser().parse_args()
    checkpoint_path = Path(args.ckpt).resolve()
    weights_checkpoint_path = Path(args.weights_ckpt).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not checkpoint_path.is_file() or not weights_checkpoint_path.is_file():
        raise FileNotFoundError("Canonical and weights checkpoints must exist")
    if parse_checkpoint_step(checkpoint_path) != CHECKPOINT_STEP:
        raise ValueError(f"Gate checkpoint must be step {CHECKPOINT_STEP}")
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    runtime_cfg = load_runtime_cfg(config_path)
    if runtime_cfg.model_name != MODEL_NAME:
        raise ValueError(f"Gate model must be {MODEL_NAME}")
    manifest = load_manifest(args.manifest, args.latent_root)

    checkpoint_sha256 = sha256_file(checkpoint_path)
    weights_sha256 = sha256_file(weights_checkpoint_path)
    if checkpoint_sha256 != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Canonical checkpoint hash is not the locked Base-100K hash")
    if weights_sha256 != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Weights checkpoint hash is not the locked Base-100K hash")
    _checkpoint_contract(weights_checkpoint_path)
    protocol = _build_protocol(
        checkpoint_path=checkpoint_path,
        weights_checkpoint_path=weights_checkpoint_path,
        config_path=config_path,
        manifest=manifest,
        output_dir=output_dir,
        devices=args.devices,
        runtime_cfg=runtime_cfg,
        checkpoint_sha256=checkpoint_sha256,
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
    common_payload = {
        "checkpoint": str(checkpoint_path),
        "weights_checkpoint": str(weights_checkpoint_path),
        "config": str(config_path),
        "manifest": manifest["path"],
        "protocol": str(protocol_path),
        "checkpoint_sha256": checkpoint_sha256,
        "weights_sha256": weights_sha256,
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": manifest["sha256"],
        "source_sha256": protocol["project_source_sha256"],
        "protocol_sha256": protocol_sha256,
    }
    if args.split == "confirmatory":
        _require_confirmatory_unlock(
            output_dir=output_dir,
            protocol_path=protocol_path,
            manifest=manifest,
            devices=args.devices,
            common_payload=common_payload,
        )
    payloads = []
    for device in args.devices:
        jobs = []
        for index, case in enumerate(split_cases, start=1):
            assigned = args.devices[(index - 1) % len(args.devices)]
            if assigned == device:
                jobs.append({
                    "case": case,
                    "result_path": str(_case_result_path(
                        output_dir,
                        args.split,
                        index,
                        case,
                    )),
                })
        payloads.append({**common_payload, "device": device, "jobs": jobs})

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(payloads),
        mp_context=context,
    ) as executor:
        futures = [executor.submit(_run_device_cases, payload) for payload in payloads]
        for future in as_completed(futures):
            print(f"Completed worker: {future.result()}", flush=True)

    _verify_locked_inputs(common_payload)
    results = _load_split_results(
        output_dir,
        args.split,
        split_cases,
        args.devices,
        common_payload,
    )
    gate = aggregate_case_results(results, args.split)
    summary = _build_split_summary(
        output_dir,
        args.split,
        split_cases,
        protocol_path,
        common_payload,
        gate,
    )
    summary_path = output_dir / f"{args.split}-summary.json"
    _publish_summary(summary_path, summary, common_payload)
    print(json.dumps({
        "safety_passed": gate["safety_passed"],
        "routing_accuracy_gap_passed": gate["routing_accuracy_gap_passed"],
        "stage_structure_passed": gate["stage_structure_passed"],
        "passed": gate["passed"],
        "means": gate["means"],
    }, indent=2, sort_keys=True))
    print(f"Saved: {summary_path}")
    if not gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
