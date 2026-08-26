#!/usr/bin/env python3
"""Run the locked forward-only compute-exchange deployability screen."""

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import json
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
import yaml

from config import cfg as base_cfg
from analyses.denoising_regret.io import write_json_atomic
from analyses.denoising_regret.probe import (
    _configure_torch_threads,
    _load_checkpoint_model,
)
from analyses.t_SNE.checkpoint_utils import (
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)
from analyses.timestep_utility.compute_exchange_deployability import (
    DEPLOYABILITY_VERSION,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    MODEL_SEED,
    MOE_BLOCKS,
    PAIRWISE_LOSS_WEIGHT,
    RETROSPECTIVE_BLOCKS,
    SCORER_KINDS,
    SIGMAS,
    TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
    extract_deployability_case,
    reveal_deployability_case,
    write_npz_atomic,
)
from analyses.timestep_utility.compute_exchange_deployability_batch import (
    ACTION_NAMES,
    BATCH_VERSION,
    BOOTSTRAP_RESAMPLES,
    FIT_REQUIREMENTS,
    RETROSPECTIVE_REQUIREMENTS,
    aggregate_retrospective,
    combine_retrospective_reveal,
    fit_gate,
    load_json,
    select_retrospective_actions,
    sha256_file,
    verify_source_gate,
)
from analyses.timestep_utility.compute_exchange_deployability_fit import (
    load_feature_dataset,
    load_scorer_bundle,
    scorer_bundle,
    split_calibration_cases,
    train_dual_scorer,
)
from utils import deep_update


DEFAULT_CHECKPOINT = (
    "outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/"
    "checkpoints/ckpt_step_200000.pth"
)
DEFAULT_WEIGHTS_CHECKPOINT = (
    "/home/dev/promoe-probes/base-seed0-ckpt_step_200000.pth"
)
DEFAULT_LATENT_ROOT = "/home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz"
DEFAULT_SOURCE_ROOT = (
    "/home/dev/promoe-probes/within-expert-compute-exchange-base200k-v1"
)
DEFAULT_OUTPUT_DIR = (
    "/home/dev/promoe-probes/compute-exchange-deployability-base200k-v1"
)
LOCKED_DEVICES = ("cuda:4", "cuda:5", "cuda:6", "cuda:7")
LOCKED_FIT_DEVICE = "cuda:4"
LOCKED_NUM_THREADS = 4
STATIC_SOURCE_PATHS = (
    "requirements.txt",
    "analyses/run_compute_exchange_deployability_gate.py",
    "analyses/timestep_utility/compute_exchange_deployability.py",
    "analyses/timestep_utility/compute_exchange_deployability_fit.py",
    "analyses/timestep_utility/compute_exchange_deployability_batch.py",
    "analyses/timestep_utility/compute_exchange_probe.py",
    "analyses/timestep_utility/compute_exchange_batch.py",
    "analyses/denoising_regret/probe.py",
    "models/models_ProMoE_TC.py",
)
SOURCE_RESULT_KEYS = {
    "source_result",
    "source_result_sha256",
    "source_seal",
    "source_seal_sha256",
}


def _sha256_stream(handle):
    handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _read_locked_bytes(path, expected_sha256, label):
    path = Path(path)
    with path.open("rb") as handle:
        payload = handle.read()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(f"Locked {label} hash changed")
    return payload


def _load_locked_json(path, expected_sha256, label):
    payload = _read_locked_bytes(path, expected_sha256, label)
    return json.loads(payload.decode("utf-8"))


def _load_runtime_cfg_locked(config_path, expected_sha256):
    config_path = Path(config_path)
    payload = _read_locked_bytes(config_path, expected_sha256, "config")
    custom_cfg = yaml.safe_load(payload)
    if not isinstance(custom_cfg, dict):
        raise ValueError("Locked config must contain a YAML mapping")
    runtime_cfg = copy.deepcopy(base_cfg)
    custom_cfg["custom_cfg_name"] = config_path.stem
    deep_update(runtime_cfg, custom_cfg)
    return runtime_cfg


def _load_checkpoint_model_locked(runtime_cfg, weights_path, expected_sha256, device):
    weights_path = Path(weights_path)
    with weights_path.open("rb") as handle:
        if _sha256_stream(handle) != expected_sha256:
            raise RuntimeError("Locked weights hash changed before loading")
        loaded = _load_checkpoint_model(runtime_cfg, handle, device)
        if _sha256_stream(handle) != expected_sha256:
            del loaded
            raise RuntimeError("Locked weights changed while loading")
    return loaded


def _load_scorer_bundle_locked(path, expected_sha256):
    path = Path(path)
    with path.open("rb") as handle:
        if _sha256_stream(handle) != expected_sha256:
            raise RuntimeError(f"Locked scorer hash changed before loading: {path}")
        loaded = load_scorer_bundle(handle, map_location="cpu")
        if _sha256_stream(handle) != expected_sha256:
            del loaded
            raise RuntimeError(f"Locked scorer changed while loading: {path}")
    return loaded


def _verify_file_hash(path, expected_sha256, label):
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Locked {label} hash changed")


def _public_source_case(case):
    return {key: value for key, value in case.items() if key not in SOURCE_RESULT_KEYS}


def _parse_devices(value):
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if devices != LOCKED_DEVICES:
        raise argparse.ArgumentTypeError(
            "The locked screen requires cuda:4,cuda:5,cuda:6,cuda:7"
        )
    return devices


def _parse_fit_device(value):
    if value != LOCKED_FIT_DEVICE:
        raise argparse.ArgumentTypeError(
            f"The locked screen requires --fit-device {LOCKED_FIT_DEVICE}"
        )
    return value


def _git_contract(require_clean=True):
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
    if require_clean and status:
        raise RuntimeError("The deployability protocol requires a clean committed tree")
    divergence = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", "origin/repa...HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if require_clean and divergence != "0\t0":
        raise RuntimeError("The deployability commit must be pushed to origin/repa")
    return {
        "commit": commit,
        "origin_repa_divergence": divergence,
        "clean": not bool(status),
    }


def _source_hashes():
    hashes = {}
    for relative in STATIC_SOURCE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Protocol source file is missing: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def _verify_checkpoint(checkpoint_path, weights_path, source):
    expected = source["protocol"]["checkpoint"]
    checkpoint_path = Path(checkpoint_path).resolve()
    weights_path = Path(weights_path).resolve()
    for path, label in ((checkpoint_path, "canonical"), (weights_path, "weights")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint is missing: {path}")
        if path.stat().st_size != int(expected[f"{label}_size"]):
            raise ValueError(f"{label} checkpoint size changed")
        if sha256_file(path) != expected[f"{label}_sha256"]:
            raise ValueError(f"{label} checkpoint hash changed")
    if parse_checkpoint_step(checkpoint_path) != int(expected["step"]):
        raise ValueError("Canonical checkpoint step differs from the source gate")
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    if sha256_file(config_path) != expected["config_sha256"]:
        raise ValueError("Checkpoint config differs from the source gate")
    return checkpoint_path, weights_path, config_path


def _verify_locked_inputs(protocol):
    checkpoint = protocol["checkpoint"]
    canonical_path = Path(checkpoint["canonical_path"]).resolve()
    weights_path = Path(checkpoint["weights_path"]).resolve()
    config_path = Path(checkpoint["config"]).resolve()
    checks = (
        (canonical_path, "canonical", checkpoint["canonical_size"], checkpoint["canonical_sha256"]),
        (weights_path, "weights", checkpoint["weights_size"], checkpoint["weights_sha256"]),
        (config_path, "config", None, checkpoint["config_sha256"]),
    )
    for path, label, expected_size, expected_sha256 in checks:
        if not path.is_file():
            raise FileNotFoundError(f"Locked {label} input is missing: {path}")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise RuntimeError(f"Locked {label} input size changed")
        if sha256_file(path) != expected_sha256:
            raise RuntimeError(f"Locked {label} input hash changed")
    if parse_checkpoint_step(canonical_path) != int(checkpoint["step"]):
        raise RuntimeError("Locked canonical checkpoint step changed")
    if parse_checkpoint_step(weights_path) != int(checkpoint["step"]):
        raise RuntimeError("Locked weights checkpoint step changed")
    if resolve_config_from_checkpoint(canonical_path).resolve() != config_path:
        raise RuntimeError("Locked canonical checkpoint resolves to another config")

    source_contract = protocol["source_gate"]
    _verify_file_hash(
        source_contract["protocol"],
        source_contract["protocol_file_sha256"],
        "source protocol",
    )
    for split in ("discovery", "confirmatory"):
        summary_path = Path(source_contract["root"]) / f"{split}-summary.json"
        expected = source_contract[f"{split}_summary_sha256"]
        _verify_file_hash(summary_path, expected, f"source {split} summary")
    for case in source_contract["cases"]["discovery"]:
        _verify_file_hash(
            case["source_result"],
            case["source_result_sha256"],
            f"calibration source result {case['id']}",
        )
        _verify_file_hash(
            case["source_seal"],
            case["source_seal_sha256"],
            f"calibration source seal {case['id']}",
        )
    for case in source_contract["cases"]["confirmatory"]:
        if SOURCE_RESULT_KEYS & set(case):
            raise RuntimeError("Public protocol exposes confirmatory source results")


def _load_reveal_source(protocol, source_root):
    source = verify_source_gate(source_root, PROJECT_ROOT)
    source_contract = protocol["source_gate"]
    if source["root"] != source_contract["root"]:
        raise RuntimeError("Source gate root changed")
    if source["protocol_path"] != source_contract["protocol"]:
        raise RuntimeError("Source gate protocol path changed")
    if source["protocol_sha256"] != source_contract["protocol_sha256"]:
        raise RuntimeError("Source compute-exchange protocol changed")
    if source["protocol_file_sha256"] != source_contract["protocol_file_sha256"]:
        raise RuntimeError("Source protocol file changed")
    if source["cases"]["discovery"] != source_contract["cases"]["discovery"]:
        raise RuntimeError("Calibration source case commitments changed")
    public_confirmatory = [
        _public_source_case(case) for case in source["cases"]["confirmatory"]
    ]
    if public_confirmatory != source_contract["cases"]["confirmatory"]:
        raise RuntimeError("Confirmatory source case manifest changed")
    for split in ("discovery", "confirmatory"):
        expected = source_contract[f"{split}_summary_sha256"]
        if source["summary_file_sha256"][split] != expected:
            raise RuntimeError(f"Source {split} summary changed")
    return source


def _environment(devices):
    cuda_devices = {}
    for device in devices:
        properties = torch.cuda.get_device_properties(torch.device(device))
        cuda_devices[device] = {
            "name": properties.name,
            "uuid": str(properties.uuid) if hasattr(properties, "uuid") else None,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_devices": cuda_devices,
    }


def _build_protocol(args, source, checkpoint_path, weights_path, config_path):
    calibration_ids = tuple(case["id"] for case in source["cases"]["discovery"])
    fit_ids, validation_ids = split_calibration_cases(calibration_ids)
    assignments = {}
    for split, source_split in (
        ("calibration", "discovery"),
        ("retrospective", "confirmatory"),
    ):
        assignments[split] = [
            {
                "index": index,
                "case_id": case["id"],
                "device": args.devices[(index - 1) % len(args.devices)],
            }
            for index, case in enumerate(source["cases"][source_split], start=1)
        ]
    return {
        "batch_version": BATCH_VERSION,
        "deployability_version": DEPLOYABILITY_VERSION,
        "registration_strength": (
            "Retrospective mechanism screen locked after the source confirmatory "
            "aggregate was known. Its 48 exact candidate banks are never used "
            "for fitting, but this is not a blind confirmatory claim. Passing "
            "only authorizes a new unseen-class second-checkpoint or second-seed gate."
        ),
        "hypothesis": (
            "Inference-visible pre-pass state can separately predict the cost of "
            "removing a routed pass and the value of a recurrent second pass, "
            "allowing beneficial exchange within each image/expert at exact load."
        ),
        "method": {
            "inputs": [
                "pre-pass MoE hidden state",
                "all native router scores plus top-1 weight/margin/entropy",
                "native expert ID",
                "block ID",
                "sigma",
                "spatial position",
            ],
            "forbidden_inputs": [
                "clean latent",
                "noise target",
                "denoising target",
                "target gradient",
                "first routed-expert output",
                "second routed-expert output",
                "exact counterfactual gain",
                "teacher or DINO feature",
            ],
            "supervision": (
                "Calibration only: stop-gradient first-order donor-removal and "
                "receiver-second-pass counterfactual changes."
            ),
            "scorer": (
                "Separate linear donor/receiver heads per MoE block and native "
                "expert; token-local LayerNorm; no batch statistic."
            ),
            "solver": (
                "For every image/expert with n tokens, assign exactly k zero-pass, "
                "k two-pass, and n-2k one-pass states by integral minimum cost, "
                "where k=min(floor(0.1*n+0.5),floor(n/2))."
            ),
            "logical_compute": (
                "Every expert retains its exact native routed-FFN pass count; "
                "shared path, unconditional path, route IDs, and route weights stay fixed."
            ),
        },
        "calibration": {
            "source_split": "source discovery only",
            "blocks": list(MOE_BLOCKS),
            "sigmas": list(SIGMAS),
            "fit_case_ids": list(fit_ids),
            "validation_case_ids": list(validation_ids),
            "target_normalization": (
                "center donor and receiver labels within each cell/expert, then "
                "divide both heads and all experts by one cell-wide scale"
            ),
            "negative_control": (
                "independently roll donor/receiver correspondence within each "
                "calibration cell/expert"
            ),
        },
        "optimization": {
            "seed": MODEL_SEED,
            "fit_device": args.fit_device,
            "max_epochs": MAX_EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "pairwise_loss_weight": PAIRWISE_LOSS_WEIGHT,
            "model_kinds": list(SCORER_KINDS),
            "selection": "highest held-out calibration candidate concordance",
        },
        "retrospective": {
            "source_split": "source confirmatory, never used for fitting",
            "blocks": list(RETROSPECTIVE_BLOCKS),
            "sigmas": list(SIGMAS),
            "selection": (
                "Generate and seal one exact 0/1/2 action per target-free scorer/control. "
                "Only a later invocation may reconstruct denoising targets and reveal "
                "the exact gain of those sealed actions."
            ),
            "controls": [
                "matched random",
                "router margin",
                "source rolled utility",
                "trained router-context-only dual head",
                "trained rolled-correspondence dual head",
            ],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "requirements": {
            "fit": FIT_REQUIREMENTS,
            "retrospective": RETROSPECTIVE_REQUIREMENTS,
        },
        "decision_rule": (
            "A failed fit gate forbids retrospective extraction. Any failed "
            "retrospective core or safety check stops this feature/scorer design. "
            "Passing authorizes only a newly sealed second-checkpoint/seed exact gate, "
            "not ImageNet long training."
        ),
        "checkpoint": {
            "canonical_path": str(checkpoint_path),
            "canonical_sha256": sha256_file(checkpoint_path),
            "canonical_size": checkpoint_path.stat().st_size,
            "weights_path": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
            "weights_size": weights_path.stat().st_size,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "step": parse_checkpoint_step(checkpoint_path),
        },
        "latent_root": str(Path(args.latent_root).resolve()),
        "source_gate": {
            "root": source["root"],
            "protocol": source["protocol_path"],
            "protocol_sha256": source["protocol_sha256"],
            "protocol_file_sha256": source["protocol_file_sha256"],
            "discovery_summary_sha256": source["summary_file_sha256"]["discovery"],
            "confirmatory_summary_sha256": source["summary_file_sha256"]["confirmatory"],
            "cases": {
                "discovery": source["cases"]["discovery"],
                "confirmatory": [
                    _public_source_case(case)
                    for case in source["cases"]["confirmatory"]
                ],
            },
        },
        "assignments": assignments,
        "devices": list(args.devices),
        "num_threads_per_worker": LOCKED_NUM_THREADS,
        "output_dir": str(Path(args.output_dir).resolve()),
        "git": _git_contract(require_clean=True),
        "project_source_sha256": _source_hashes(),
        "environment": _environment(args.devices),
    }


def _protocol_paths(output_dir):
    output_dir = Path(output_dir)
    return output_dir / "protocol.json", output_dir / "protocol.sha256"


def _write_or_validate_protocol(output_dir, protocol):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path, sha_path = _protocol_paths(output_dir)
    if protocol_path.exists():
        if load_json(protocol_path) != protocol:
            raise RuntimeError("Existing deployability protocol is incompatible")
    else:
        write_json_atomic(protocol_path, protocol)
    digest = sha256_file(protocol_path)
    line = f"{digest}  protocol.json\n"
    if sha_path.exists():
        if sha_path.read_text(encoding="utf-8") != line:
            raise RuntimeError("Existing deployability protocol SHA is incompatible")
    else:
        temporary = sha_path.with_suffix(".sha256.tmp")
        temporary.write_text(line, encoding="utf-8")
        temporary.replace(sha_path)
    return digest


def _load_protocol(output_dir):
    protocol_path, sha_path = _protocol_paths(output_dir)
    protocol = load_json(protocol_path)
    digest = sha256_file(protocol_path)
    if sha_path.read_text(encoding="utf-8") != f"{digest}  protocol.json\n":
        raise RuntimeError("Deployability protocol SHA sidecar mismatch")
    git = _git_contract(require_clean=True)
    if git["commit"] != protocol["git"]["commit"]:
        raise RuntimeError("Current commit differs from the deployability protocol")
    for relative, expected in protocol["project_source_sha256"].items():
        if sha256_file(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"Protocol source changed: {relative}")
    _verify_locked_inputs(protocol)
    return protocol, digest


def _case_paths(output_dir, split, index, case_id):
    stem = f"{index:02d}_{case_id}"
    directory = Path(output_dir) / split
    return {
        "npz": directory / f"{stem}.features.npz",
        "metadata": directory / f"{stem}.metadata.json",
        "seal": directory / f"{stem}.seal.json",
    }


def _case_seal(paths, protocol_sha256, case_id):
    return {
        "version": 1,
        "case_id": case_id,
        "protocol_sha256": protocol_sha256,
        "features_sha256": sha256_file(paths["npz"]),
        "metadata_sha256": sha256_file(paths["metadata"]),
    }


def _verify_case_files(paths, protocol_sha256, case_id, split):
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"Incomplete {split} feature case: {case_id}")
    seal = load_json(paths["seal"])
    if seal != _case_seal(paths, protocol_sha256, case_id):
        raise RuntimeError(f"Feature seal mismatch: {case_id}")
    metadata = load_json(paths["metadata"])
    if metadata["case_id"] != case_id or metadata["split"] != split:
        raise RuntimeError(f"Feature metadata mismatch: {case_id}")
    if metadata["protocol_sha256"] != protocol_sha256:
        raise RuntimeError(f"Feature protocol mismatch: {case_id}")
    if split == "retrospective":
        if metadata["privileged_targets_present"]:
            raise RuntimeError("Retrospective metadata exposes privileged targets")
        forbidden_metadata = {"native_mse", "source_result", "source_result_sha256"}
        if forbidden_metadata & set(metadata):
            raise RuntimeError("Retrospective metadata exposes privileged source state")
        if any("native_mse" in cell for cell in metadata["cells"]):
            raise RuntimeError("Retrospective cells expose target-derived native MSE")
        if any(
            cell["gradient_enabled"] or cell["target_constructed"]
            for cell in metadata["cells"]
        ):
            raise RuntimeError("Retrospective extraction violated forward-only mode")
    return {
        "case_id": case_id,
        "npz": str(paths["npz"]),
        "metadata": str(paths["metadata"]),
        "seal": str(paths["seal"]),
        "features_sha256": seal["features_sha256"],
        "metadata_sha256": seal["metadata_sha256"],
    }


def _atomic_torch_save(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _run_worker(payload):
    protocol = payload["protocol"]
    split = payload["split"]
    device = torch.device(payload["device"])
    _configure_torch_threads(protocol["num_threads_per_worker"])
    weights_path = Path(protocol["checkpoint"]["weights_path"])
    config_path = Path(protocol["checkpoint"]["config"])
    runtime_cfg = _load_runtime_cfg_locked(
        config_path,
        protocol["checkpoint"]["config_sha256"],
    )
    model, state_name, checkpoint_step, _ = _load_checkpoint_model_locked(
        runtime_cfg,
        weights_path,
        protocol["checkpoint"]["weights_sha256"],
        device,
    )
    if state_name != "ema_model_state_dict" or checkpoint_step != protocol["checkpoint"]["step"]:
        raise RuntimeError("Worker loaded an unexpected checkpoint state")
    outputs = []
    try:
        for assignment in payload["assignments"]:
            case = assignment["case"]
            paths = {
                key: Path(value) for key, value in assignment["paths"].items()
            }
            if all(path.is_file() for path in paths.values()):
                outputs.append(_verify_case_files(
                    paths,
                    payload["protocol_sha256"],
                    case["id"],
                    split,
                ))
                continue
            if paths["seal"].exists():
                raise RuntimeError(f"Sealed feature output is incomplete for {case['id']}")
            for key in ("npz", "metadata"):
                if paths[key].exists():
                    paths[key].unlink()
            latent_path = Path(protocol["latent_root"]) / case["latent_relative"]
            _verify_file_hash(
                latent_path,
                case["latent_sha256"],
                f"latent {case['id']} before extraction",
            )
            if split == "calibration":
                source_result = _load_locked_json(
                    case["source_result"],
                    case["source_result_sha256"],
                    f"calibration source result {case['id']}",
                )
                _verify_file_hash(
                    case["source_seal"],
                    case["source_seal_sha256"],
                    f"calibration source seal {case['id']}",
                )
            else:
                source_result = None
            arrays, metadata = extract_deployability_case(
                model=model,
                runtime_cfg=runtime_cfg,
                case=case,
                latent_root=protocol["latent_root"],
                source_result=source_result,
                split=split,
            )
            _verify_file_hash(
                latent_path,
                case["latent_sha256"],
                f"latent {case['id']} after extraction",
            )
            metadata.update({
                "protocol_sha256": payload["protocol_sha256"],
                "checkpoint_sha256": protocol["checkpoint"]["weights_sha256"],
                "device": str(device),
            })
            if split == "calibration":
                metadata.update({
                    "source_result": case["source_result"],
                    "source_result_sha256": case["source_result_sha256"],
                })
            write_npz_atomic(paths["npz"], arrays)
            write_json_atomic(paths["metadata"], metadata)
            write_json_atomic(
                paths["seal"],
                _case_seal(paths, payload["protocol_sha256"], case["id"]),
            )
            outputs.append(_verify_case_files(
                paths,
                payload["protocol_sha256"],
                case["id"],
                split,
            ))
    finally:
        del model
        torch.cuda.empty_cache()
    return outputs


def _source_case_map(protocol, split):
    source_split = "discovery" if split == "calibration" else "confirmatory"
    return {
        case["id"]: case
        for case in protocol["source_gate"]["cases"][source_split]
    }


def _feature_manifest(protocol, protocol_sha256, split, require_complete=True):
    rows = []
    for assignment in protocol["assignments"][split]:
        paths = _case_paths(
            protocol["output_dir"],
            split,
            assignment["index"],
            assignment["case_id"],
        )
        if require_complete:
            row = _verify_case_files(
                paths,
                protocol_sha256,
                assignment["case_id"],
                split,
            )
        else:
            row = {
                "case_id": assignment["case_id"],
                **{key: str(value) for key, value in paths.items()},
            }
        rows.append(row)
    return rows


def _verify_feature_inputs(case_files, stage):
    for row in case_files:
        _verify_file_hash(
            row["npz"],
            row["features_sha256"],
            f"feature array {row['case_id']} {stage}",
        )
        _verify_file_hash(
            row["metadata"],
            row["metadata_sha256"],
            f"feature metadata {row['case_id']} {stage}",
        )


def _load_feature_dataset_locked(case_files, require_targets):
    _verify_feature_inputs(case_files, "before loading")
    dataset = load_feature_dataset(case_files, require_targets=require_targets)
    _verify_feature_inputs(case_files, "after loading")
    return dataset


def _run_extract(protocol, protocol_sha256, split):
    if split == "calibration":
        if (Path(protocol["output_dir"]) / "fit-summary.json").exists():
            raise RuntimeError("Calibration cannot change after scorer fitting")
    else:
        _verify_fit_artifacts(protocol, protocol_sha256)
    case_map = _source_case_map(protocol, split)
    by_device = {device: [] for device in protocol["devices"]}
    for assignment in protocol["assignments"][split]:
        case = case_map[assignment["case_id"]]
        paths = _case_paths(
            protocol["output_dir"],
            split,
            assignment["index"],
            assignment["case_id"],
        )
        by_device[assignment["device"]].append({
            "case": case,
            "paths": {key: str(value) for key, value in paths.items()},
        })
    payloads = [
        {
            "protocol": protocol,
            "protocol_sha256": protocol_sha256,
            "split": split,
            "device": device,
            "assignments": assignments,
        }
        for device, assignments in by_device.items()
        if assignments
    ]
    with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
        futures = [executor.submit(_run_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            for row in future.result():
                print(f"Completed {split}: {row['case_id']}", flush=True)
    manifest = _feature_manifest(protocol, protocol_sha256, split)
    summary = {
        "version": 1,
        "split": split,
        "protocol_sha256": protocol_sha256,
        "case_count": len(manifest),
        "cases": manifest,
    }
    summary_path = Path(protocol["output_dir"]) / f"{split}-features-summary.json"
    write_json_atomic(summary_path, summary)
    write_json_atomic(
        summary_path.with_suffix(".json.seal.json"),
        {
            "version": 1,
            "case_id": f"{split}-features-summary",
            "protocol_sha256": protocol_sha256,
            "result_sha256": sha256_file(summary_path),
        },
    )
    print(json.dumps({"split": split, "case_count": len(manifest)}, indent=2))


def _fit_paths(output_dir, kind):
    return Path(output_dir) / f"scorer-{kind}.pth"


def _run_fit(protocol, protocol_sha256, device):
    retrospective_dir = Path(protocol["output_dir"]) / "retrospective"
    if retrospective_dir.exists() and any(retrospective_dir.iterdir()):
        raise RuntimeError("Retrospective features already exist before fitting")
    fit_summary_path = Path(protocol["output_dir"]) / "fit-summary.json"
    scorer_paths = [
        _fit_paths(protocol["output_dir"], kind) for kind in SCORER_KINDS
    ]
    if fit_summary_path.exists() or any(path.exists() for path in scorer_paths):
        raise RuntimeError("Scorer fitting is single-use and already has artifacts")
    case_files = _feature_manifest(protocol, protocol_sha256, "calibration")
    dataset = _load_feature_dataset_locked(case_files, require_targets=True)
    fit_ids = protocol["calibration"]["fit_case_ids"]
    validation_ids = protocol["calibration"]["validation_case_ids"]
    summaries = {}
    scorer_rows = {}
    for kind in SCORER_KINDS:
        print(f"Fitting scorer: {kind}", flush=True)
        model, summary = train_dual_scorer(
            dataset,
            fit_ids,
            validation_ids,
            kind,
            device,
        )
        bundle = scorer_bundle(
            model,
            kind,
            summary,
            {
                "protocol_sha256": protocol_sha256,
                "calibration_feature_sha256": [
                    row["features_sha256"] for row in case_files
                ],
                "fit_case_ids": fit_ids,
                "validation_case_ids": validation_ids,
            },
        )
        path = _fit_paths(protocol["output_dir"], kind)
        _atomic_torch_save(path, bundle)
        summaries[kind] = summary
        scorer_rows[kind] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gate = fit_gate(summaries)
    payload = {
        "version": 1,
        "protocol_sha256": protocol_sha256,
        "device": str(device),
        "scorers": scorer_rows,
        "summaries": summaries,
        "gate": gate,
        "decision": (
            "retrospective_forward_only_screen_authorized"
            if gate["passed"] else "stop_scorer_design_before_retrospective"
        ),
    }
    path = Path(protocol["output_dir"]) / "fit-summary.json"
    write_json_atomic(path, payload)
    write_json_atomic(
        path.with_suffix(".json.seal.json"),
        {
            "version": 1,
            "case_id": "fit-summary",
            "protocol_sha256": protocol_sha256,
            "result_sha256": sha256_file(path),
        },
    )
    print(json.dumps({"gate": gate, "decision": payload["decision"]}, indent=2))
    return bool(gate["passed"])


def _verify_fit_artifacts(protocol, protocol_sha256):
    fit_path = Path(protocol["output_dir"]) / "fit-summary.json"
    fit_seal_path = fit_path.with_suffix(".json.seal.json")
    fit_seal = load_json(fit_seal_path)
    if not isinstance(fit_seal.get("result_sha256"), str):
        raise RuntimeError("Fit summary seal has no content hash")
    expected_seal = {
        "version": 1,
        "case_id": "fit-summary",
        "protocol_sha256": protocol_sha256,
        "result_sha256": fit_seal["result_sha256"],
    }
    if fit_seal != expected_seal:
        raise RuntimeError("Fit summary seal mismatch")
    fit_summary = _load_locked_json(
        fit_path,
        expected_seal["result_sha256"],
        "fit summary",
    )
    if not fit_summary["gate"]["passed"]:
        raise RuntimeError("Fit gate failed; retrospective evaluation is forbidden")
    if set(fit_summary["scorers"]) != set(SCORER_KINDS):
        raise RuntimeError("Fit summary does not contain every locked scorer")
    for kind, row in fit_summary["scorers"].items():
        if sha256_file(row["path"]) != row["sha256"]:
            raise RuntimeError(f"Scorer artifact changed: {kind}")
    return fit_summary


def _load_fitted_models(protocol, protocol_sha256, device):
    fit_summary = _verify_fit_artifacts(protocol, protocol_sha256)
    models = {}
    for kind, row in fit_summary["scorers"].items():
        model, bundle = _load_scorer_bundle_locked(row["path"], row["sha256"])
        if bundle["calibration_contract"]["protocol_sha256"] != protocol_sha256:
            raise RuntimeError(f"Scorer protocol mismatch: {kind}")
        models[kind] = model.to(device).eval()
    return models, fit_summary


def _actions_paths(output_dir):
    path = Path(output_dir) / "retrospective-actions.json"
    return path, path.with_suffix(".json.seal.json")


def _validate_action_names(actions):
    if not isinstance(actions, dict) or set(actions) != set(ACTION_NAMES):
        raise RuntimeError("Sealed action names differ from the protocol")


def _run_select(protocol, protocol_sha256, device):
    action_path, action_seal_path = _actions_paths(protocol["output_dir"])
    if action_path.exists() or action_seal_path.exists():
        raise RuntimeError("Retrospective action selection is single-use")
    reveal_dir = Path(protocol["output_dir"]) / "reveal"
    if reveal_dir.exists() and any(reveal_dir.iterdir()):
        raise RuntimeError("Reveal artifacts exist before action selection")
    case_files = _feature_manifest(protocol, protocol_sha256, "retrospective")
    dataset = _load_feature_dataset_locked(case_files, require_targets=False)
    models, _ = _load_fitted_models(protocol, protocol_sha256, device)
    try:
        records = select_retrospective_actions(dataset, models, torch.device(device))
    finally:
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    records_by_case = {}
    for record in records:
        records_by_case.setdefault(record["case_id"], []).append(record)
    cases = []
    for assignment in protocol["assignments"]["retrospective"]:
        case_id = assignment["case_id"]
        case_records = records_by_case.get(case_id, [])
        if len(case_records) != len(RETROSPECTIVE_BLOCKS) * len(SIGMAS):
            raise RuntimeError(f"Action selection is incomplete for {case_id}")
        cases.append({"case_id": case_id, "cells": case_records})
    action_mismatches = sum(
        cell["action_invariance_mismatch"]
        for case in cases for cell in case["cells"]
    )
    logical_mismatches = sum(
        not cell["logical_pass_counts_match"]
        for case in cases for cell in case["cells"]
    )
    gate = {
        "action_invariance_mismatches": int(action_mismatches),
        "logical_pass_count_mismatches": int(logical_mismatches),
        "passed": action_mismatches == 0 and logical_mismatches == 0,
    }
    payload = {
        "version": 1,
        "protocol_sha256": protocol_sha256,
        "fit_summary_sha256": sha256_file(
            Path(protocol["output_dir"]) / "fit-summary.json"
        ),
        "retrospective_feature_sha256": [
            row["features_sha256"] for row in case_files
        ],
        "target_free_contract": (
            "No confirmatory source-result path or payload, denoising target, "
            "native MSE, target gradient, or exact gain was passed to action "
            "generation."
        ),
        "device": str(device),
        "gate": gate,
        "cases": cases,
    }
    write_json_atomic(action_path, payload)
    write_json_atomic(
        action_seal_path,
        {
            "version": 1,
            "case_id": "retrospective-actions",
            "protocol_sha256": protocol_sha256,
            "result_sha256": sha256_file(action_path),
        },
    )
    print(json.dumps({"gate": gate, "actions": str(action_path)}, indent=2))
    return bool(gate["passed"])


def _verify_action_artifacts(protocol, protocol_sha256):
    action_path, action_seal_path = _actions_paths(protocol["output_dir"])
    action_seal = load_json(action_seal_path)
    if not isinstance(action_seal.get("result_sha256"), str):
        raise RuntimeError("Retrospective action seal has no content hash")
    expected_seal = {
        "version": 1,
        "case_id": "retrospective-actions",
        "protocol_sha256": protocol_sha256,
        "result_sha256": action_seal["result_sha256"],
    }
    if action_seal != expected_seal:
        raise RuntimeError("Retrospective action seal mismatch")
    actions = _load_locked_json(
        action_path,
        expected_seal["result_sha256"],
        "retrospective actions",
    )
    if actions["protocol_sha256"] != protocol_sha256:
        raise RuntimeError("Retrospective actions use another protocol")
    if not actions["gate"]["passed"]:
        raise RuntimeError("Action invariance gate failed; reveal is forbidden")
    feature_manifest = _feature_manifest(
        protocol,
        protocol_sha256,
        "retrospective",
    )
    if actions["retrospective_feature_sha256"] != [
        row["features_sha256"] for row in feature_manifest
    ]:
        raise RuntimeError("Retrospective feature inputs changed after action sealing")
    if actions["fit_summary_sha256"] != sha256_file(
        Path(protocol["output_dir"]) / "fit-summary.json"
    ):
        raise RuntimeError("Fit summary changed after action sealing")
    expected_case_ids = [
        row["case_id"] for row in protocol["assignments"]["retrospective"]
    ]
    if [case["case_id"] for case in actions["cases"]] != expected_case_ids:
        raise RuntimeError("Retrospective action cases differ from the protocol")
    expected_cells = {
        (block, sigma) for block in RETROSPECTIVE_BLOCKS for sigma in SIGMAS
    }
    for case in actions["cases"]:
        cells = {
            (int(cell["block_index"]), float(cell["sigma"])): cell
            for cell in case["cells"]
        }
        if set(cells) != expected_cells or len(case["cells"]) != len(expected_cells):
            raise RuntimeError(f"Sealed action grid is incomplete: {case['case_id']}")
        for cell in case["cells"]:
            _validate_action_names(cell["actions"])
            forbidden = {"native_mse", "selected_gain", "exact_mse_change", "source_result"}
            if forbidden & set(cell):
                raise RuntimeError("Sealed action cell contains revealed target state")
    return actions, action_path, expected_seal["result_sha256"]


def _reveal_case_paths(output_dir, index, case_id):
    stem = f"{int(index):02d}_{case_id}"
    directory = Path(output_dir) / "reveal"
    return {
        "result": directory / f"{stem}.json",
        "seal": directory / f"{stem}.json.seal.json",
    }


def _reveal_case_seal(paths, protocol_sha256, actions_sha256, case_id):
    return {
        "version": 1,
        "case_id": case_id,
        "protocol_sha256": protocol_sha256,
        "actions_sha256": actions_sha256,
        "result_sha256": sha256_file(paths["result"]),
    }


def _verify_reveal_case(
    paths,
    protocol_sha256,
    actions_sha256,
    checkpoint_sha256,
    case_id,
):
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"Incomplete reveal case: {case_id}")
    seal = load_json(paths["seal"])
    if seal != _reveal_case_seal(
        paths,
        protocol_sha256,
        actions_sha256,
        case_id,
    ):
        raise RuntimeError(f"Reveal seal mismatch: {case_id}")
    result = _load_locked_json(
        paths["result"],
        seal["result_sha256"],
        f"reveal result {case_id}",
    )
    if result["case_id"] != case_id:
        raise RuntimeError(f"Reveal result case mismatch: {case_id}")
    if result["protocol_sha256"] != protocol_sha256:
        raise RuntimeError(f"Reveal result protocol mismatch: {case_id}")
    if result["actions_sha256"] != actions_sha256:
        raise RuntimeError(f"Reveal result action mismatch: {case_id}")
    if result["checkpoint_sha256"] != checkpoint_sha256:
        raise RuntimeError(f"Reveal result checkpoint mismatch: {case_id}")
    expected_cells = {
        (block, sigma) for block in RETROSPECTIVE_BLOCKS for sigma in SIGMAS
    }
    observed = {
        (int(cell["block_index"]), float(cell["sigma"]))
        for cell in result["cells"]
    }
    if observed != expected_cells or len(result["cells"]) != len(expected_cells):
        raise RuntimeError(f"Reveal result cell grid is incomplete: {case_id}")
    return {
        "case_id": case_id,
        "result": str(paths["result"]),
        "seal": str(paths["seal"]),
        "result_sha256": sha256_file(paths["result"]),
    }


def _run_reveal_worker(payload):
    protocol = payload["protocol"]
    device = torch.device(payload["device"])
    _configure_torch_threads(protocol["num_threads_per_worker"])
    locked_actions = _load_locked_json(
        payload["actions_path"],
        payload["actions_sha256"],
        "retrospective actions in reveal worker",
    )
    locked_by_case = {case["case_id"]: case for case in locked_actions["cases"]}
    weights_path = Path(protocol["checkpoint"]["weights_path"])
    config_path = Path(protocol["checkpoint"]["config"])
    runtime_cfg = _load_runtime_cfg_locked(
        config_path,
        protocol["checkpoint"]["config_sha256"],
    )
    model, state_name, checkpoint_step, _ = _load_checkpoint_model_locked(
        runtime_cfg,
        weights_path,
        protocol["checkpoint"]["weights_sha256"],
        device,
    )
    if state_name != "ema_model_state_dict" or checkpoint_step != protocol["checkpoint"]["step"]:
        raise RuntimeError("Reveal worker loaded an unexpected checkpoint state")
    outputs = []
    try:
        for assignment in payload["assignments"]:
            case = assignment["case"]
            case_id = case["id"]
            paths = {key: Path(value) for key, value in assignment["paths"].items()}
            if all(path.is_file() for path in paths.values()):
                outputs.append(_verify_reveal_case(
                    paths,
                    payload["protocol_sha256"],
                    payload["actions_sha256"],
                    protocol["checkpoint"]["weights_sha256"],
                    case_id,
                ))
                continue
            if paths["seal"].exists():
                raise RuntimeError(f"Sealed reveal output is incomplete for {case_id}")
            if paths["result"].exists():
                paths["result"].unlink()
            latent_path = Path(protocol["latent_root"]) / case["latent_relative"]
            _verify_file_hash(
                latent_path,
                case["latent_sha256"],
                f"reveal latent {case_id} before loading",
            )
            action_case = assignment["action_case"]
            if locked_by_case.get(case_id) != action_case:
                raise RuntimeError(f"Reveal action case changed for {case_id}")
            result = reveal_deployability_case(
                model=model,
                runtime_cfg=runtime_cfg,
                case=case,
                latent_root=protocol["latent_root"],
                action_case=action_case,
            )
            _verify_file_hash(
                latent_path,
                case["latent_sha256"],
                f"reveal latent {case_id} after loading",
            )
            result.update({
                "protocol_sha256": payload["protocol_sha256"],
                "actions_sha256": payload["actions_sha256"],
                "checkpoint_sha256": protocol["checkpoint"]["weights_sha256"],
                "device": str(device),
            })
            write_json_atomic(paths["result"], result)
            write_json_atomic(
                paths["seal"],
                _reveal_case_seal(
                    paths,
                    payload["protocol_sha256"],
                    payload["actions_sha256"],
                    case_id,
                ),
            )
            outputs.append(_verify_reveal_case(
                paths,
                payload["protocol_sha256"],
                payload["actions_sha256"],
                protocol["checkpoint"]["weights_sha256"],
                case_id,
            ))
    finally:
        del model
        torch.cuda.empty_cache()
    return outputs


def _run_reveal(protocol, protocol_sha256, actions, action_path, actions_sha256):
    case_map = _source_case_map(protocol, "retrospective")
    action_by_case = {case["case_id"]: case for case in actions["cases"]}
    by_device = {device: [] for device in protocol["devices"]}
    for assignment in protocol["assignments"]["retrospective"]:
        case_id = assignment["case_id"]
        paths = _reveal_case_paths(
            protocol["output_dir"],
            assignment["index"],
            case_id,
        )
        by_device[assignment["device"]].append({
            "case": case_map[case_id],
            "action_case": action_by_case[case_id],
            "paths": {key: str(value) for key, value in paths.items()},
        })
    payloads = [
        {
            "protocol": protocol,
            "protocol_sha256": protocol_sha256,
            "actions_path": str(action_path),
            "actions_sha256": actions_sha256,
            "device": device,
            "assignments": assignments,
        }
        for device, assignments in by_device.items()
        if assignments
    ]
    with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
        futures = [executor.submit(_run_reveal_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            for row in future.result():
                print(f"Completed reveal: {row['case_id']}", flush=True)

    manifest = []
    results = []
    for assignment in protocol["assignments"]["retrospective"]:
        paths = _reveal_case_paths(
            protocol["output_dir"],
            assignment["index"],
            assignment["case_id"],
        )
        manifest_row = _verify_reveal_case(
            paths,
            protocol_sha256,
            actions_sha256,
            protocol["checkpoint"]["weights_sha256"],
            assignment["case_id"],
        )
        manifest.append(manifest_row)
        results.append(_load_locked_json(
            paths["result"],
            manifest_row["result_sha256"],
            f"reveal result {assignment['case_id']}",
        ))
    summary = {
        "version": 1,
        "protocol_sha256": protocol_sha256,
        "actions_sha256": actions_sha256,
        "case_count": len(manifest),
        "cases": manifest,
    }
    summary_path = Path(protocol["output_dir"]) / "reveal-summary.json"
    summary_seal_path = summary_path.with_suffix(".json.seal.json")
    if summary_path.exists():
        if load_json(summary_path) != summary:
            raise RuntimeError("Existing reveal summary is incompatible")
    else:
        write_json_atomic(summary_path, summary)
    expected_seal = {
        "version": 1,
        "case_id": "reveal-summary",
        "protocol_sha256": protocol_sha256,
        "actions_sha256": actions_sha256,
        "result_sha256": sha256_file(summary_path),
    }
    if summary_seal_path.exists():
        if load_json(summary_seal_path) != expected_seal:
            raise RuntimeError("Existing reveal summary seal is incompatible")
    else:
        write_json_atomic(summary_seal_path, expected_seal)
    return results, summary_path


def _run_evaluate(protocol, protocol_sha256, source_root):
    output_path = Path(protocol["output_dir"]) / "retrospective-summary.json"
    if output_path.exists():
        raise RuntimeError("Retrospective evaluation is single-use and already exists")
    actions, action_path, actions_sha256 = _verify_action_artifacts(
        protocol,
        protocol_sha256,
    )
    reveal_results, reveal_summary_path = _run_reveal(
        protocol,
        protocol_sha256,
        actions,
        action_path,
        actions_sha256,
    )
    source = _load_reveal_source(protocol, source_root)
    source_results = {}
    for case in source["cases"]["confirmatory"]:
        source_results[case["id"]] = _load_locked_json(
            case["source_result"],
            case["source_result_sha256"],
            f"confirmatory source result {case['id']}",
        )
        _verify_file_hash(
            case["source_seal"],
            case["source_seal_sha256"],
            f"confirmatory source seal {case['id']}",
        )
    action_records = [cell for case in actions["cases"] for cell in case["cells"]]
    records = combine_retrospective_reveal(
        action_records,
        reveal_results,
        source_results,
    )
    gate = aggregate_retrospective(records)
    payload = {
        "version": 1,
        "protocol_sha256": protocol_sha256,
        "actions_sha256": actions_sha256,
        "reveal_summary_sha256": sha256_file(reveal_summary_path),
        "records": records,
        "gate": gate,
        "decision": (
            "new_unseen_second_checkpoint_gate_authorized"
            if gate["passed"] else "stop_forward_only_compute_exchange_direction"
        ),
    }
    write_json_atomic(output_path, payload)
    write_json_atomic(
        output_path.with_suffix(".json.seal.json"),
        {
            "version": 1,
            "case_id": "retrospective-summary",
            "protocol_sha256": protocol_sha256,
            "result_sha256": sha256_file(output_path),
        },
    )
    print(json.dumps({"gate": gate, "decision": payload["decision"]}, indent=2))
    return bool(gate["passed"])


def build_parser():
    parser = argparse.ArgumentParser(
        description="Locked forward-only compute-exchange deployability screen."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument(
        "--extract-split",
        choices=("calibration", "retrospective"),
    )
    action.add_argument("--fit", action="store_true")
    action.add_argument("--select", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    parser.add_argument("--ckpt", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--weights-ckpt", default=DEFAULT_WEIGHTS_CHECKPOINT)
    parser.add_argument("--latent-root", default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=LOCKED_DEVICES,
    )
    parser.add_argument(
        "--fit-device",
        type=_parse_fit_device,
        default=LOCKED_FIT_DEVICE,
    )
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".orchestration.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if args.prepare_only:
            source = verify_source_gate(args.source_root, PROJECT_ROOT)
            checkpoint_path, weights_path, config_path = _verify_checkpoint(
                args.ckpt,
                args.weights_ckpt,
                source,
            )
            latent_root = Path(args.latent_root).resolve()
            if not latent_root.is_dir():
                raise FileNotFoundError(f"Latent root is missing: {latent_root}")
            protocol = _build_protocol(
                args,
                source,
                checkpoint_path,
                weights_path,
                config_path,
            )
            digest = _write_or_validate_protocol(output_dir, protocol)
            print(f"Locked protocol: {output_dir / 'protocol.json'}")
            print(f"Protocol SHA256: {digest}")
            return
        protocol, protocol_sha256 = _load_protocol(output_dir)
        if str(Path(args.source_root).resolve()) != protocol["source_gate"]["root"]:
            raise ValueError("CLI source root differs from locked protocol")
        if args.fit_device != protocol["optimization"]["fit_device"]:
            raise ValueError("CLI fit device differs from the locked protocol")
        if args.extract_split:
            _run_extract(protocol, protocol_sha256, args.extract_split)
        elif args.fit:
            if not _run_fit(protocol, protocol_sha256, args.fit_device):
                raise SystemExit(1)
        elif args.select:
            if not _run_select(protocol, protocol_sha256, args.fit_device):
                raise SystemExit(1)
        else:
            if not _run_evaluate(protocol, protocol_sha256, args.source_root):
                raise SystemExit(1)
    gc.collect()


if __name__ == "__main__":
    main()
