"""Run the locked expert-function transport gate over held-out ImageNet cases."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import multiprocessing
import os
import platform
import site
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import (
    PackageNotFoundError,
    distribution as package_distribution,
    distributions as installed_distributions,
    packages_distributions,
    version as package_version,
)
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from analyses.denoising_regret.io import write_json_atomic
from analyses.denoising_regret.probe import _build_model
from analyses.expert_function.batch import (
    BATCH_VERSION,
    BLOCK_INDEX,
    CASE_SPECS,
    CHECKPOINT_STATE,
    CHECKPOINT_STEP,
    EXACT_BATCH_SIZE,
    EXPECTED_WEIGHTS_SHA256,
    GATE_REQUIREMENTS,
    MODEL_NAME,
    NUM_ROUTED_EXPERTS,
    NUM_TOKEN_PROBES,
    SHIFTS,
    SIGMAS,
    aggregate_case_results,
    load_manifest,
    sha256_file,
    validate_case_result,
)
from analyses.expert_function.consistency_probe import (
    PROBE_VERSION,
    run_expert_function_consistency_probe,
)
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)


LOCKED_DEVICES = ("cuda:4", "cuda:5", "cuda:6", "cuda:7")
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "analyses"
    / "expert_function"
    / "manifests"
    / "function_transport_gate_v1.json"
)
LOCKED_STATIC_PATHS = (
    "requirements.txt",
)
PROVENANCE_VERSION = 1
PENDING_SEAL_VERSION = 2


def _default_output_dir(checkpoint_path, manifest_name):
    checkpoint_path = Path(checkpoint_path).resolve()
    step = checkpoint_path.stem.removeprefix("ckpt_step_")
    return (
        checkpoint_path.parent.parent
        / "sample"
        / f"step{step}"
        / "expert_function_consistency_probe_batch"
        / manifest_name
    )


def _parse_devices(value):
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if devices != LOCKED_DEVICES:
        raise argparse.ArgumentTypeError(
            "The confirmatory gate is locked to cuda:4,cuda:5,cuda:6,cuda:7"
        )
    return devices


def _case_result_path(output_dir, index, case):
    return Path(output_dir) / "cases" / f"{index:02d}_{case['id']}.json"


def _runtime_environment(devices, distributions):
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
        "distributions": distributions,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_devices": cuda_devices,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


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


def _loaded_project_source_paths():
    project_root = PROJECT_ROOT.resolve()
    source_paths = set()
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
        source_paths.add(relative.as_posix())
    return tuple(sorted(source_paths))


def _loaded_distribution_versions():
    package_map = packages_distributions()
    distribution_names = set()
    project_root = PROJECT_ROOT.resolve()
    site_roots = {
        Path(path).resolve()
        for path in (*site.getsitepackages(), site.getusersitepackages())
        if path
    }
    unresolved_external = {}
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name:
            continue
        source_path = _module_source_path(module)
        if source_path is None:
            continue
        try:
            source_path.relative_to(project_root)
        except ValueError:
            pass
        else:
            continue
        top_level = module_name.partition(".")[0]
        mapped_distributions = package_map.get(top_level, ())
        if mapped_distributions:
            distribution_names.update(mapped_distributions)
            continue
        try:
            metadata = package_distribution(top_level).metadata
        except PackageNotFoundError:
            if any(
                source_path == root or root in source_path.parents
                for root in site_roots
            ):
                unresolved_external[top_level] = str(source_path)
            continue
        distribution_names.add(metadata.get("Name", top_level))
    if unresolved_external:
        unresolved_paths = {
            Path(path).resolve() for path in unresolved_external.values()
        }
        target_names = {path.name for path in unresolved_paths}
        resolved_owners = {}
        for distribution in installed_distributions():
            distribution_name = distribution.metadata.get("Name")
            if not distribution_name or distribution.files is None:
                continue
            for relative_path in distribution.files:
                if Path(relative_path).name not in target_names:
                    continue
                candidate = Path(distribution.locate_file(relative_path)).resolve()
                if candidate in unresolved_paths:
                    resolved_owners[candidate] = distribution_name
            if len(resolved_owners) == len(unresolved_paths):
                break
        distribution_names.update(resolved_owners.values())
        unresolved_external = {
            module_name: path
            for module_name, path in unresolved_external.items()
            if Path(path).resolve() not in resolved_owners
        }
    if unresolved_external:
        raise RuntimeError(
            "Cannot resolve distributions for loaded external modules: "
            f"{dict(sorted(unresolved_external.items()))}"
        )
    return {
        distribution: package_version(distribution)
        for distribution in sorted(distribution_names, key=str.lower)
    }


def _collect_model_provenance(runtime_cfg):
    with torch.random.fork_rng(devices=[]):
        model = _build_model(runtime_cfg)
    model_metadata = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    del model
    gc.collect()

    source_paths = set(_loaded_project_source_paths())
    source_paths.update(LOCKED_STATIC_PATHS)
    source_hashes = {}
    for relative in sorted(source_paths):
        source_path = PROJECT_ROOT / relative
        if not source_path.is_file():
            raise FileNotFoundError(f"Locked source file is missing: {source_path}")
        source_hashes[relative] = sha256_file(source_path)
    return {
        "version": PROVENANCE_VERSION,
        "scope": (
            "target-model construction import closure plus static protocol files"
        ),
        "model": model_metadata,
        "project_source_sha256": source_hashes,
        "loaded_distributions": _loaded_distribution_versions(),
    }


def _expected_run(
    checkpoint_path,
    weights_checkpoint_path,
    config_path,
    device,
    num_threads,
    checkpoint_sha256,
    weights_sha256,
    protocol_sha256,
):
    return {
        "checkpoint": str(checkpoint_path),
        "weights_checkpoint": str(weights_checkpoint_path),
        "config": str(config_path),
        "device": device,
        "num_threads": int(num_threads),
        "checkpoint_sha256": checkpoint_sha256,
        "weights_sha256": weights_sha256,
        "protocol_sha256": protocol_sha256,
    }


def _write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def _write_or_validate_protocol(output_dir, protocol):
    output_dir = Path(output_dir)
    protocol_path = output_dir / "protocol.json"
    hash_path = output_dir / "protocol.sha256"
    case_paths = list((output_dir / "cases").glob("*.json"))
    summary_path = output_dir / "summary.json"
    if not protocol_path.exists() and (case_paths or summary_path.exists()):
        raise RuntimeError(
            "Refusing result files without a pre-existing locked protocol.json"
        )
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError(
                f"Existing protocol is incompatible with this run: {protocol_path}"
            )
    else:
        write_json_atomic(protocol_path, protocol)

    protocol_hash = sha256_file(protocol_path)
    expected_hash_line = f"{protocol_hash}  protocol.json\n"
    if hash_path.exists():
        if hash_path.read_text(encoding="utf-8") != expected_hash_line:
            raise ValueError(f"Protocol checksum file differs: {hash_path}")
    else:
        _write_text_atomic(hash_path, expected_hash_line)
    return protocol_path, protocol_hash


def _load_compatible_case(path, case, expected_run):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_case_result(result, case, expected_run)
    return result


def _pending_seal_path(pending_path):
    return Path(f"{pending_path}.seal")


def _device_pending_path(result_path, device):
    result_path = Path(result_path)
    return result_path.with_suffix(
        result_path.suffix + f".pending.{device.replace(':', '_')}"
    )


def _json_payload_sha256(payload):
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _pending_seal(pending_sha256, case, expected_run):
    return {
        "version": PENDING_SEAL_VERSION,
        "case_id": case["id"],
        "latent_sha256": case["latent_sha256"],
        "expected_run": expected_run,
        "pending_sha256": pending_sha256,
    }


def _load_sealed_pending(pending_path, case, expected_run):
    seal_path = _pending_seal_path(pending_path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    result = _load_compatible_case(pending_path, case, expected_run)
    pending_sha256 = _json_payload_sha256(result)
    _verify_locked_file(
        pending_path,
        pending_sha256,
        f"Sealed pending result for {case['id']}",
    )
    expected_seal = _pending_seal(pending_sha256, case, expected_run)
    if seal != expected_seal:
        raise ValueError(f"Pending seal is incompatible: {seal_path}")
    return result, pending_sha256


def _cleanup_published_result_artifacts(
    result_path,
    device,
    case,
    expected_run,
    published_result,
):
    pending_path = _device_pending_path(result_path, device)
    seal_path = _pending_seal_path(pending_path)
    if pending_path.exists() and seal_path.exists():
        pending_result, _ = _load_sealed_pending(
            pending_path,
            case,
            expected_run,
        )
        if pending_result != published_result:
            raise RuntimeError(
                f"Published and sealed pending results differ for {case['id']}"
            )
    if seal_path.exists():
        seal_path.unlink()
    if pending_path.exists():
        pending_path.unlink()


def _verify_locked_file(path, expected_sha256, description):
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{description} changed after protocol lock: {Path(path).resolve()}"
        )


def _verify_worker_inputs(payload):
    _verify_locked_file(
        payload["checkpoint"],
        payload["checkpoint_sha256"],
        "Canonical checkpoint",
    )
    _verify_locked_file(
        payload["weights_checkpoint"],
        payload["weights_sha256"],
        "Weights checkpoint",
    )
    _verify_locked_file(
        payload["config"],
        payload["config_sha256"],
        "Config",
    )
    _verify_locked_file(
        payload["manifest"],
        payload["manifest_sha256"],
        "Manifest",
    )
    _verify_locked_file(
        payload["protocol"],
        payload["protocol_sha256"],
        "Protocol",
    )
    for relative, expected_sha256 in payload["source_sha256"].items():
        _verify_locked_file(
            PROJECT_ROOT / relative,
            expected_sha256,
            f"Locked source {relative}",
        )


def _run_device_cases(payload):
    device = payload["device"]
    total_cases = payload["total_cases"]
    _verify_worker_inputs(payload)
    pending_results = []
    for job in payload["jobs"]:
        index = job["index"]
        case = job["case"]
        result_path = Path(job["result_path"])
        pending_path = _device_pending_path(result_path, device)
        seal_path = _pending_seal_path(pending_path)
        _verify_locked_file(
            case["latent"],
            case["latent_sha256"],
            f"Latent for {case['id']}",
        )
        if pending_path.exists():
            if not seal_path.exists():
                pending_path.unlink()
                print(
                    f"[{device}] [{index}/{total_cases}] discarding unsealed "
                    f"pending {case['id']}",
                    flush=True,
                )
            else:
                result, pending_sha256 = _load_sealed_pending(
                    pending_path,
                    case,
                    job["expected_run"],
                )
                pending_results.append(
                    (
                        pending_path,
                        seal_path,
                        result_path,
                        case,
                        job["expected_run"],
                        result,
                        pending_sha256,
                    )
                )
                print(
                    f"[{device}] [{index}/{total_cases}] reusing sealed "
                    f"pending {case['id']}",
                    flush=True,
                )
                continue
        elif seal_path.exists():
            raise RuntimeError(f"Pending seal has no pending result: {seal_path}")
        print(
            f"[{device}] [{index}/{total_cases}] probing {case['id']}",
            flush=True,
        )
        result = run_expert_function_consistency_probe(
            checkpoint_path=payload["checkpoint"],
            weights_checkpoint_path=payload["weights_checkpoint"],
            latent_path=case["latent"],
            latent_key=case["latent_key"],
            label=case["label"],
            sigmas=SIGMAS,
            shifts=SHIFTS,
            block_index=BLOCK_INDEX,
            num_token_probes=NUM_TOKEN_PROBES,
            exact_batch_size=EXACT_BATCH_SIZE,
            seed=case["seed"],
            device=device,
            num_threads=payload["num_threads"],
        )
        result.update({
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "weights_checkpoint_sha256": payload["weights_sha256"],
            "latent_sha256": case["latent_sha256"],
            "protocol_sha256": payload["protocol_sha256"],
        })
        result["batch_case"] = case
        validate_case_result(result, case, job["expected_run"])
        _verify_locked_file(
            case["latent"],
            case["latent_sha256"],
            f"Latent for {case['id']}",
        )
        pending_sha256 = _json_payload_sha256(result)
        write_json_atomic(pending_path, result)
        _verify_locked_file(
            pending_path,
            pending_sha256,
            f"Pending result for {case['id']}",
        )
        pending_results.append(
            (
                pending_path,
                seal_path,
                result_path,
                case,
                job["expected_run"],
                result,
                pending_sha256,
            )
        )
        print(
            f"[{device}] [{index}/{total_cases}] prepared {result_path}",
            flush=True,
        )
    _verify_worker_inputs(payload)
    for pending in pending_results:
        pending_path, _, _, case, expected_run, result, pending_sha256 = pending
        _verify_locked_file(
            case["latent"],
            case["latent_sha256"],
            f"Latent for {case['id']}",
        )
        _verify_locked_file(
            pending_path,
            pending_sha256,
            f"Pending result for {case['id']}",
        )
        if _load_compatible_case(pending_path, case, expected_run) != result:
            raise RuntimeError(f"Pending result snapshot changed for {case['id']}")

    sealed_results = []
    for pending in pending_results:
        pending_path, seal_path, result_path, case, expected_run, result, pending_sha256 = pending
        seal = _pending_seal(pending_sha256, case, expected_run)
        seal_sha256 = _json_payload_sha256(seal)
        write_json_atomic(
            seal_path,
            seal,
        )
        _verify_locked_file(
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
            expected_run,
            result,
            pending_sha256,
            seal_sha256,
        ) = sealed
        _verify_locked_file(
            pending_path,
            pending_sha256,
            f"Pending result for {case['id']}",
        )
        _verify_locked_file(
            seal_path,
            seal_sha256,
            f"Pending seal for {case['id']}",
        )
        write_json_atomic(result_path, result)
        _verify_locked_file(
            result_path,
            pending_sha256,
            f"Published result for {case['id']}",
        )
        _load_compatible_case(result_path, case, expected_run)
        seal_path.unlink()
        pending_path.unlink()
        print(f"[{device}] saved {result_path}", flush=True)
    return device


def _build_protocol(
    checkpoint_path,
    weights_checkpoint_path,
    config_path,
    manifest,
    output_dir,
    devices,
    num_threads,
    checkpoint_sha256,
    weights_sha256,
    runtime_cfg,
):
    provenance = _collect_model_provenance(runtime_cfg)

    assignments = [
        {
            "index": index,
            "case_id": case["id"],
            "device": devices[(index - 1) % len(devices)],
        }
        for index, case in enumerate(manifest["cases"], start=1)
    ]
    return {
        "batch_version": BATCH_VERSION,
        "probe_version": PROBE_VERSION,
        "locked_before_confirmatory_results": True,
        "hypothesis": (
            "shared-expert-relative routed functions that follow transported "
            "content predict exact equal-compute denoising responsibility"
        ),
        "checkpoint": {
            "canonical_path": str(checkpoint_path),
            "canonical_sha256": checkpoint_sha256,
            "weights_path": str(weights_checkpoint_path),
            "step": CHECKPOINT_STEP,
            "state": CHECKPOINT_STATE,
            "weights_sha256": weights_sha256,
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "model_name": MODEL_NAME,
        },
        "manifest": {
            "name": manifest["name"],
            "path": manifest["path"],
            "sha256": manifest["sha256"],
            "selection": manifest["selection"],
            "case_ids": [case["id"] for case in manifest["cases"]],
            "case_inputs": [
                {
                    "id": case["id"],
                    "label": case["label"],
                    "seed": case["seed"],
                    "latent_relative": case["latent_relative"],
                    "latent_sha256": case["latent_sha256"],
                }
                for case in manifest["cases"]
            ],
        },
        "run": {
            "block_index": BLOCK_INDEX,
            "sigmas": list(SIGMAS),
            "shifts_latent": [list(shift) for shift in SHIFTS],
            "num_token_probes_per_cell": NUM_TOKEN_PROBES,
            "num_routed_experts": NUM_ROUTED_EXPERTS,
            "exact_batch_size": EXACT_BATCH_SIZE,
            "devices": list(devices),
            "assignments": assignments,
            "num_threads_per_process": int(num_threads),
            "output_dir": str(output_dir),
        },
        "gate_requirements": dict(GATE_REQUIREMENTS),
        "provenance": provenance,
        "environment": _runtime_environment(
            devices,
            provenance["loaded_distributions"],
        ),
    }


def _publish_summary(summary_path, summary, locked_inputs):
    summary_path = Path(summary_path)
    pending_path = summary_path.with_suffix(summary_path.suffix + ".pending")
    expected_sha256 = _json_payload_sha256(summary)
    _verify_worker_inputs(locked_inputs)
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise ValueError(
                f"Existing summary differs from recomputation: {summary_path}"
            )
        _verify_locked_file(summary_path, expected_sha256, "Existing summary")
        pending_path.unlink(missing_ok=True)
        return
    write_json_atomic(pending_path, summary)
    _verify_worker_inputs(locked_inputs)
    _verify_locked_file(pending_path, expected_sha256, "Pending summary")
    os.replace(pending_path, summary_path)
    _verify_locked_file(summary_path, expected_sha256, "Published summary")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the pre-registered 24-image expert-function transport gate "
            "on the fixed Base EMA checkpoint."
        )
    )
    parser.add_argument("--ckpt", required=True, help="Canonical Base checkpoint")
    parser.add_argument(
        "--weights-ckpt",
        required=True,
        help="Local checkpoint copy whose SHA256 is fixed by the protocol",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Locked function_transport_gate_v1 manifest",
    )
    parser.add_argument("--latent-root", required=True)
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=LOCKED_DEVICES,
        help="Locked device group: cuda:4,cuda:5,cuda:6,cuda:7",
    )
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate inputs and write protocol.json before running any case",
    )
    return parser


def main():
    args = build_parser().parse_args()
    checkpoint_path = Path(args.ckpt).resolve()
    weights_checkpoint_path = Path(args.weights_ckpt).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not weights_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Weights checkpoint does not exist: {weights_checkpoint_path}"
        )
    checkpoint_step = parse_checkpoint_step(checkpoint_path)
    if checkpoint_step != CHECKPOINT_STEP:
        raise ValueError(
            f"Confirmatory gate requires step {CHECKPOINT_STEP}, got {checkpoint_step}"
        )
    checkpoint_hash = sha256_file(checkpoint_path)
    weights_hash = sha256_file(weights_checkpoint_path)
    if checkpoint_hash != EXPECTED_WEIGHTS_SHA256:
        raise ValueError(
            "Canonical checkpoint SHA256 differs from the pre-registered value"
        )
    if weights_hash != EXPECTED_WEIGHTS_SHA256:
        raise ValueError(
            "Weights checkpoint SHA256 differs from the pre-registered value"
        )
    if checkpoint_hash != weights_hash:
        raise ValueError("Canonical and local checkpoint contents differ")
    if args.num_threads < 1:
        raise ValueError("num_threads must be positive")

    devices = tuple(args.devices)
    if devices != LOCKED_DEVICES:
        raise ValueError("Confirmatory devices differ from the locked GPU group")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        raise RuntimeError("The locked four-GPU gate requires visible CUDA devices 4-7")

    config_path = Path(resolve_config_from_checkpoint(checkpoint_path)).resolve()
    runtime_cfg = load_runtime_cfg(config_path)
    if runtime_cfg.model_name != MODEL_NAME:
        raise ValueError(
            f"Confirmatory gate requires {MODEL_NAME}, got {runtime_cfg.model_name}"
        )
    manifest = load_manifest(args.manifest, args.latent_root)
    if len(manifest["cases"]) != len(CASE_SPECS):
        raise ValueError("Manifest case count differs from the locked protocol")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else _default_output_dir(checkpoint_path, manifest["name"])
    )

    protocol = _build_protocol(
        checkpoint_path=checkpoint_path,
        weights_checkpoint_path=weights_checkpoint_path,
        config_path=config_path,
        manifest=manifest,
        output_dir=output_dir,
        devices=devices,
        num_threads=args.num_threads,
        checkpoint_sha256=checkpoint_hash,
        weights_sha256=weights_hash,
        runtime_cfg=runtime_cfg,
    )
    protocol_path, protocol_hash = _write_or_validate_protocol(
        output_dir,
        protocol,
    )
    print(f"Locked protocol: {protocol_path}")
    print(f"Protocol SHA256: {protocol_hash}")
    if args.prepare_only:
        return

    locked_inputs = {
        "checkpoint": str(checkpoint_path),
        "weights_checkpoint": str(weights_checkpoint_path),
        "config": str(config_path),
        "manifest": manifest["path"],
        "protocol": str(protocol_path),
        "checkpoint_sha256": checkpoint_hash,
        "weights_sha256": weights_hash,
        "config_sha256": protocol["config"]["sha256"],
        "manifest_sha256": manifest["sha256"],
        "protocol_sha256": protocol_hash,
        "source_sha256": protocol["provenance"][
            "project_source_sha256"
        ],
    }
    _verify_worker_inputs(locked_inputs)

    expected_runs = {}
    jobs_by_device = {device: [] for device in devices}
    for index, case in enumerate(manifest["cases"], start=1):
        device = devices[(index - 1) % len(devices)]
        expected_run = _expected_run(
            checkpoint_path,
            weights_checkpoint_path,
            config_path,
            device,
            args.num_threads,
            checkpoint_hash,
            weights_hash,
            protocol_hash,
        )
        expected_runs[case["id"]] = expected_run
        result_path = _case_result_path(output_dir, index, case)
        if result_path.exists():
            published_result = _load_compatible_case(
                result_path,
                case,
                expected_run,
            )
            pending_path = _device_pending_path(result_path, device)
            seal_path = _pending_seal_path(pending_path)
            had_stale_artifacts = pending_path.exists() or seal_path.exists()
            _cleanup_published_result_artifacts(
                result_path,
                device,
                case,
                expected_run,
                published_result,
            )
            if had_stale_artifacts:
                print(
                    f"[{index}/{len(manifest['cases'])}] cleaned published "
                    f"artifacts for {case['id']}"
                )
            print(f"[{index}/{len(manifest['cases'])}] reusing {case['id']}")
            continue
        jobs_by_device[device].append({
            "index": index,
            "case": case,
            "result_path": str(result_path),
            "expected_run": expected_run,
        })

    worker_payloads = [
        {
            **locked_inputs,
            "device": device,
            "jobs": jobs,
            "num_threads": int(args.num_threads),
            "total_cases": len(manifest["cases"]),
        }
        for device, jobs in jobs_by_device.items()
        if jobs
    ]
    if worker_payloads:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(worker_payloads),
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(_run_device_cases, payload)
                for payload in worker_payloads
            ]
            for future in as_completed(futures):
                print(f"Completed worker: {future.result()}", flush=True)

    _verify_worker_inputs(locked_inputs)
    case_results = []
    for index, case in enumerate(manifest["cases"], start=1):
        result_path = _case_result_path(output_dir, index, case)
        if not result_path.is_file():
            raise RuntimeError(f"Missing confirmatory case result: {result_path}")
        case_results.append(_load_compatible_case(
            result_path,
            case,
            expected_runs[case["id"]],
        ))

    gate = aggregate_case_results(
        case_results,
        manifest["cases"],
        expected_runs,
    )
    summary = {
        "batch_version": BATCH_VERSION,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "checkpoint": str(checkpoint_path),
        "weights_checkpoint": str(weights_checkpoint_path),
        "manifest": manifest,
        "gate": gate,
    }
    summary_path = output_dir / "summary.json"
    _publish_summary(summary_path, summary, locked_inputs)
    print(json.dumps({
        "passed": gate["passed"],
        "safety_passed": gate["safety_passed"],
        "mechanism_passed": gate["mechanism_passed"],
        "safety_checks": gate["safety_checks"],
        "mechanism_checks": gate["mechanism_checks"],
    }, indent=2, sort_keys=True))
    print(f"Function-transport gate: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"Saved: {summary_path}")
    if not gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
