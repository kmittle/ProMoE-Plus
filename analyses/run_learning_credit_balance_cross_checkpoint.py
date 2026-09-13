#!/usr/bin/env python3
"""Run the paired Base/Loss-Free count and exact-credit analysis."""

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
import random
import re
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
    _load_checkpoint_payload,
)
from analyses.t_SNE.checkpoint_utils import load_runtime_cfg, parse_checkpoint_step
from analyses.timestep_utility.credit_balance_batch import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CHECKPOINT_STATE,
    LOCKED_NUM_THREADS,
    MODEL_NAME,
    SAFETY_REQUIREMENTS,
    SPLIT_COUNTS,
    aggregate_credit_balance,
    case_protocol_view,
    select_cases,
    sha256_file,
)
from analyses.timestep_utility.credit_balance_cross_checkpoint import (
    CROSS_CHECKPOINT_VERSION,
    PARAMETER_BOOTSTRAP_RESAMPLES,
    PARAMETER_BOOTSTRAP_SEED,
    aggregate_parameter_credit_validation,
    evaluate_count_balance,
    evaluate_count_replay,
    validate_exact_parameter_credit_formula,
)
from analyses.timestep_utility.credit_balance_cross_checkpoint_probe import (
    MAX_NATIVE_WEIGHT_DRIFT,
    run_cross_checkpoint_credit_balance_case,
    validate_cross_checkpoint_model,
)
from analyses.timestep_utility.credit_balance_probe import (
    BLOCKS,
    DUPLICATE_BATCH_SIZE,
    PERMUTATION_RESAMPLES,
    PROBE_VERSION,
    SELECTION_SALT,
    SIGMAS,
)
from analyses.timestep_utility.repository_output import repository_output_dir


RUNNER_VERSION = 3
SEAL_VERSION = 1
PARAMETER_CASE_COUNT = 16
LOSSFREE_MODEL_NAME = "ProMoE_TC_B_lossfree"
CHECKPOINT_ROLES = ("base", "lossfree")
# Both checkpoints must carry the trainer provenance written by train.py; the
# pair is matched on it instead of on pinned file hashes.
EXPECTED_TRAINER_STATE_VERSION = 2
EXPECTED_AUGMENTATION_SEED_VERSION = 1
EXPECTED_SAMPLER_CONTRACT_VERSION = 1
EXPECTED_DATASET_IDENTITY_VERSION = 1
EXPECTED_DATASET_TYPE = "__mp_main__.LatentFolder"
OPTIONAL_TRAINER_KEYS = frozenset({"run_id", "training_provenance"})
LOCKED_DEVICES = ("cuda:4", "cuda:5", "cuda:6", "cuda:7")
DEFAULT_LATENT_ROOT = "/home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz"
LATENT_PATHS_CACHE = PROJECT_ROOT / "preprocess/latent_paths_cache.txt"
STATIC_SOURCE_PATHS = (
    "requirements.txt",
    "analyses/run_learning_credit_balance_cross_checkpoint.py",
    "analyses/timestep_utility/credit_balance_cross_checkpoint.py",
    "analyses/timestep_utility/credit_balance_cross_checkpoint_probe.py",
    "analyses/timestep_utility/credit_balance_probe.py",
    "analyses/timestep_utility/credit_balance_batch.py",
    "analyses/denoising_regret/probe.py",
    "analyses/timestep_utility/probe.py",
    "analyses/t_SNE/checkpoint_utils.py",
    "models/modules.py",
    "models/models_ProMoE_TC.py",
    "models/models_ProMoE_TC_lossfree.py",
    "train.py",
)
PLUMBING_CELL_KEYS = frozenset({
    "block_index",
    "sigma",
    "numerical_controls",
})
COUNT_CELL_KEYS = frozenset({
    "block_index",
    "sigma",
    "timestep",
    "statistics",
    "numerical_controls",
})
COUNT_STATISTICS_KEYS = frozenset({"token_count", "active_experts"})
STAGE_MEASUREMENT_SCOPE = {
    "plumbing": "output",
    "discovery-count": "count",
    "discovery-credit": "output",
    "parameter": "parameter",
    "confirmatory-count": "count",
    "confirmatory-credit": "output",
}


def _parse_devices(value):
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if devices != LOCKED_DEVICES:
        raise argparse.ArgumentTypeError(
            "The locked cross-checkpoint gate requires cuda:4,cuda:5,cuda:6,cuda:7"
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


def _model_metadata(runtime_cfg, require_lossfree_bias):
    with torch.random.fork_rng(devices=[]):
        model = _build_model(runtime_cfg)
    contract = validate_cross_checkpoint_model(
        model,
        require_lossfree_bias=require_lossfree_bias,
    )
    metadata = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "block_contract": contract,
    }
    del model
    gc.collect()
    return metadata


def _collect_project_source_hashes(base_cfg, lossfree_cfg):
    metadata = {
        "base": _model_metadata(base_cfg, require_lossfree_bias=False),
        "lossfree": _model_metadata(lossfree_cfg, require_lossfree_bias=True),
    }
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
    return metadata, hashes


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
        raise RuntimeError("Prepare the cross-checkpoint protocol from a clean tree")
    divergence = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", "origin/repa...HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if divergence != "0\t0":
        raise RuntimeError("Cross-checkpoint code must already be pushed to origin/repa")
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


def _latent_dataset_identity(latent_root):
    """Recompute the ordered LatentFolder identity with the training algorithm."""
    from train import (
        DATASET_IDENTITY_VERSION,
        _build_latent_class_to_idx,
        _hash_dataset_record,
    )

    latent_root = Path(latent_root).resolve()
    if not latent_root.is_dir():
        raise FileNotFoundError(f"Latent root does not exist: {latent_root}")
    with os.scandir(latent_root) as entries:
        class_entries = [
            entry for entry in entries if entry.is_dir(follow_symlinks=False)
        ]
    disk_paths = []
    for class_entry in class_entries:
        with os.scandir(class_entry.path) as files:
            disk_paths.extend(
                entry.path
                for entry in files
                if entry.is_file(follow_symlinks=False)
                and entry.name.endswith(".latent.npz")
            )
    disk_paths.sort()

    latent_paths = disk_paths
    cache_path = Path(LATENT_PATHS_CACHE)
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        cached_paths = cache_path.read_text(encoding="utf-8").splitlines()
        normalized_root = os.path.normpath(latent_root)
        if cached_paths and os.path.normpath(cached_paths[0]).startswith(
            normalized_root + os.sep
        ):
            latent_paths = cached_paths
            if latent_paths != disk_paths:
                raise RuntimeError(
                    "Training latent cache differs from the complete disk inventory"
                )
    observed_class_names = {
        os.path.basename(os.path.dirname(path)) for path in latent_paths
    }
    class_to_idx = _build_latent_class_to_idx(
        observed_class_names,
        [entry.name for entry in class_entries],
    )

    digest = hashlib.sha256()
    _hash_dataset_record(
        digest,
        DATASET_IDENTITY_VERSION,
        EXPECTED_DATASET_TYPE,
        len(latent_paths),
    )
    normalized_root = os.path.normpath(latent_root)
    for path in latent_paths:
        class_name = os.path.basename(os.path.dirname(path))
        relative = os.path.relpath(os.path.normpath(path), normalized_root)
        _hash_dataset_record(digest, relative, class_to_idx[class_name])
    return {
        "version": DATASET_IDENTITY_VERSION,
        "type": EXPECTED_DATASET_TYPE,
        "num_samples": len(latent_paths),
        "ordered_samples_sha256": digest.hexdigest(),
    }


def _expected_training(runtime_cfg, checkpoint_step, dataset_identity):
    """Derive the trainer provenance a checkpoint of this config must carry."""
    world_size = len(runtime_cfg.gpu_ids)
    global_batch_size = int(runtime_cfg.total_train_batch_size)
    if world_size <= 0 or global_batch_size % world_size != 0:
        raise RuntimeError("Config global batch is not divisible by its world size")
    grad_mix = int(getattr(runtime_cfg, "grad_mix", 1))
    if grad_mix <= 0:
        raise RuntimeError("Config grad_mix must be positive")
    return {
        "global_seed": int(runtime_cfg.global_seed),
        "world_size": world_size,
        "global_batch_size": global_batch_size,
        "per_rank_batch_size": global_batch_size // world_size,
        "grad_mix": grad_mix,
        "checkpoint_step": int(checkpoint_step),
        "dataset_identity": dataset_identity,
    }


def _require_nonnegative_integer(value, field, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"Checkpoint trainer provenance is invalid: {field}")
    if positive and value == 0:
        raise RuntimeError(f"Checkpoint trainer provenance is invalid: {field}")
    return value


def _validate_sha256(value, field):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"Checkpoint trainer provenance is invalid: {field}")
    return value


def _valid_run_id(value):
    return isinstance(value, str) and bool(
        re.fullmatch(r"[A-Za-z0-9_-]{16,128}", value)
    )


def _validate_checkpoint_rng_state(state):
    required_keys = {"python", "numpy", "torch", "cuda"}
    if not isinstance(state, dict) or set(state) != required_keys:
        raise RuntimeError("Checkpoint rank RNG provenance is incomplete")
    try:
        python_rng = random.Random()
        python_rng.setstate(state["python"])

        numpy_state = state["numpy"]
        if not isinstance(numpy_state, dict) or set(numpy_state) != {
            "bit_generator",
            "state",
            "position",
            "has_gauss",
            "cached_gaussian",
        }:
            raise TypeError("NumPy RNG state is incomplete")
        if numpy_state["bit_generator"] != "MT19937":
            raise ValueError("NumPy RNG bit generator changed")
        position = _require_nonnegative_integer(
            numpy_state["position"],
            "rank_rng.numpy.position",
        )
        has_gauss = _require_nonnegative_integer(
            numpy_state["has_gauss"],
            "rank_rng.numpy.has_gauss",
        )
        if has_gauss not in {0, 1}:
            raise ValueError("NumPy RNG has_gauss flag is invalid")
        cached_gaussian = float(numpy_state["cached_gaussian"])
        if not np.isfinite(cached_gaussian):
            raise ValueError("NumPy RNG cached Gaussian is nonfinite")
        state_vector = numpy_state["state"]
        if (
            not torch.is_tensor(state_vector)
            or state_vector.dtype not in {torch.int64, torch.uint32}
            or state_vector.ndim != 1
            or state_vector.numel() == 0
        ):
            raise TypeError("NumPy RNG state vector is invalid")
        state_vector = state_vector.detach().cpu()
        if state_vector.dtype == torch.int64:
            maximum = np.iinfo(np.uint32).max
            if torch.any(state_vector < 0) or torch.any(state_vector > maximum):
                raise ValueError("NumPy RNG state vector is outside uint32 range")
        numpy_rng = np.random.RandomState()
        numpy_rng.set_state((
            numpy_state["bit_generator"],
            state_vector.numpy().astype(np.uint32, copy=True),
            position,
            has_gauss,
            cached_gaussian,
        ))

        torch_state = state["torch"]
        if (
            not torch.is_tensor(torch_state)
            or torch_state.dtype != torch.uint8
            or torch_state.ndim != 1
            or torch_state.numel() == 0
        ):
            raise TypeError("Torch RNG state is invalid")
        torch.Generator(device="cpu").set_state(torch_state.detach().cpu())

        cuda_state = state["cuda"]
        if (
            not torch.is_tensor(cuda_state)
            or cuda_state.dtype != torch.uint8
            or cuda_state.ndim != 1
            or cuda_state.numel() == 0
        ):
            raise TypeError("CUDA RNG state is invalid")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to validate CUDA RNG provenance")
        cuda_generator = torch.Generator(
            device=f"cuda:{torch.cuda.current_device()}"
        )
        cuda_generator.set_state(cuda_state.detach().cpu())
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("Checkpoint rank RNG provenance is invalid") from error


def _checkpoint_training_provenance(checkpoint, expected):
    trainer_state = checkpoint.get("trainer_state")
    if not isinstance(trainer_state, dict):
        raise RuntimeError("Checkpoint lacks trainer provenance")
    expected_fields = {
        "version": EXPECTED_TRAINER_STATE_VERSION,
        "augmentation_seed_version": EXPECTED_AUGMENTATION_SEED_VERSION,
        "global_seed": expected["global_seed"],
        "world_size": expected["world_size"],
        "grad_mix": expected["grad_mix"],
        "next_step": expected["checkpoint_step"] + 1,
        "data_batches_seen": (
            (expected["checkpoint_step"] + 1) * expected["grad_mix"]
        ),
    }
    required_trainer_keys = set(expected_fields) | {
        "batches_per_epoch",
        "sampler_epoch",
        "sampler_batch_offset",
        "sampler_contract",
        "rank_states",
    }
    observed_trainer_keys = set(trainer_state)
    if (
        not required_trainer_keys <= observed_trainer_keys
        or observed_trainer_keys - required_trainer_keys - OPTIONAL_TRAINER_KEYS
    ):
        raise RuntimeError("Checkpoint trainer provenance fields changed")
    run_id = trainer_state.get("run_id")
    if run_id is not None and not _valid_run_id(run_id):
        raise RuntimeError("Checkpoint trainer provenance is invalid: run_id")
    for field, value in expected_fields.items():
        observed = _require_nonnegative_integer(
            trainer_state.get(field),
            field,
            positive=field in {
                "version",
                "augmentation_seed_version",
                "world_size",
                "grad_mix",
                "next_step",
                "data_batches_seen",
            },
        )
        if observed != value:
            raise RuntimeError(f"Checkpoint trainer provenance changed: {field}")
    sampler = trainer_state.get("sampler_contract")
    if not isinstance(sampler, dict):
        raise RuntimeError("Checkpoint lacks sampler provenance")
    expected_sampler_keys = {
        "version",
        "global_seed",
        "per_rank_batch_size",
        "type",
        "drop_last",
        "case1_prob",
        "dataset",
    }
    if set(sampler) != expected_sampler_keys:
        raise RuntimeError("Checkpoint sampler provenance fields changed")
    sampler_fields = {
        "version": EXPECTED_SAMPLER_CONTRACT_VERSION,
        "global_seed": expected["global_seed"],
        "per_rank_batch_size": expected["per_rank_batch_size"],
        "type": "distributed",
        "drop_last": False,
        "case1_prob": None,
    }
    for field, value in sampler_fields.items():
        observed = sampler.get(field)
        if field in {"version", "global_seed", "per_rank_batch_size"}:
            observed = _require_nonnegative_integer(
                observed,
                f"sampler_contract.{field}",
                positive=field in {"version", "per_rank_batch_size"},
            )
        if observed != value:
            raise RuntimeError(f"Checkpoint sampler provenance changed: {field}")
    dataset = sampler.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != {
        "version",
        "type",
        "num_samples",
        "ordered_samples_sha256",
    }:
        raise RuntimeError("Checkpoint sampler dataset provenance is incomplete")
    dataset_version = _require_nonnegative_integer(
        dataset.get("version"),
        "sampler_contract.dataset.version",
        positive=True,
    )
    if dataset_version != EXPECTED_DATASET_IDENTITY_VERSION:
        raise RuntimeError("Checkpoint sampler dataset version changed")
    if dataset.get("type") != EXPECTED_DATASET_TYPE:
        raise RuntimeError("Checkpoint sampler dataset type changed")
    num_samples = _require_nonnegative_integer(
        dataset.get("num_samples"),
        "sampler_contract.dataset.num_samples",
        positive=True,
    )
    _validate_sha256(
        dataset.get("ordered_samples_sha256"),
        "sampler_contract.dataset.ordered_samples_sha256",
    )
    if dataset != expected.get("dataset_identity"):
        raise RuntimeError(
            "Checkpoint sampler dataset differs from the locked latent dataset"
        )
    per_rank_samples = (
        num_samples + expected["world_size"] - 1
    ) // expected["world_size"]
    expected_batches_per_epoch = (
        per_rank_samples + expected["per_rank_batch_size"] - 1
    ) // expected["per_rank_batch_size"]
    batches_per_epoch = _require_nonnegative_integer(
        trainer_state.get("batches_per_epoch"),
        "batches_per_epoch",
        positive=True,
    )
    if batches_per_epoch != expected_batches_per_epoch:
        raise RuntimeError("Checkpoint batches_per_epoch is internally inconsistent")
    sampler_epoch = _require_nonnegative_integer(
        trainer_state.get("sampler_epoch"),
        "sampler_epoch",
    )
    sampler_batch_offset = _require_nonnegative_integer(
        trainer_state.get("sampler_batch_offset"),
        "sampler_batch_offset",
    )
    expected_sampler_position = divmod(
        expected_fields["data_batches_seen"],
        batches_per_epoch,
    )
    if (sampler_epoch, sampler_batch_offset) != expected_sampler_position:
        raise RuntimeError("Checkpoint sampler position is internally inconsistent")
    rank_states = trainer_state.get("rank_states")
    if not isinstance(rank_states, list) or len(rank_states) != expected["world_size"]:
        raise RuntimeError("Checkpoint rank RNG provenance is incomplete")
    rank_ids = []
    for state in rank_states:
        if not isinstance(state, dict) or set(state) != {"rank", "rng_state"}:
            raise RuntimeError("Checkpoint rank RNG provenance is incomplete")
        rank_ids.append(state["rank"])
        _validate_checkpoint_rng_state(state["rng_state"])
    if rank_ids != list(range(expected["world_size"])):
        raise RuntimeError("Checkpoint rank RNG provenance IDs are invalid")
    return {
        "trainer_state_version": trainer_state.get("version"),
        "augmentation_seed_version": trainer_state.get(
            "augmentation_seed_version"
        ),
        **{
            field: value
            for field, value in expected_fields.items()
            if field not in {"version", "augmentation_seed_version"}
        },
        "global_batch_size": expected["global_batch_size"],
        "per_rank_batch_size": expected["per_rank_batch_size"],
        "batches_per_epoch": batches_per_epoch,
        "sampler_epoch": sampler_epoch,
        "sampler_batch_offset": sampler_batch_offset,
        "sampler_contract": sampler,
        "rank_ids": rank_ids,
        "run_id": run_id,
        "strict_training_provenance": (
            trainer_state.get("training_provenance") is not None
        ),
    }


def _checkpoint_contract(checkpoint_path, config_path, model_name, dataset_identity):
    checkpoint_path = Path(checkpoint_path).resolve()
    config_path = Path(config_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    checkpoint_step = parse_checkpoint_step(checkpoint_path)
    runtime_cfg = load_runtime_cfg(config_path)
    if runtime_cfg.model_name != model_name:
        raise ValueError(f"Checkpoint config model must be {model_name}")
    expected_training = _expected_training(
        runtime_cfg,
        checkpoint_step,
        dataset_identity,
    )
    stat = checkpoint_path.stat()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = _load_checkpoint_payload(checkpoint_path)
    if checkpoint.get("step") != checkpoint_step:
        raise ValueError("Checkpoint payload step differs from its file name")
    if CHECKPOINT_STATE not in checkpoint:
        raise KeyError(f"Checkpoint is missing {CHECKPOINT_STATE}")
    training_provenance = _checkpoint_training_provenance(
        checkpoint,
        expected_training,
    )
    del checkpoint
    gc.collect()
    return {
        "path": str(checkpoint_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": checkpoint_sha256,
        "step": checkpoint_step,
        "state": CHECKPOINT_STATE,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model_name": model_name,
        "learning_rate": float(runtime_cfg.lr),
        "training_provenance": training_provenance,
    }


def _validate_paired_training(contracts):
    """Require the pair to share its training setup apart from the Loss-Free bias."""
    base = contracts["base"]
    lossfree = contracts["lossfree"]
    fields = {
        "checkpoint_step": (base["step"], lossfree["step"]),
        "learning_rate": (base["learning_rate"], lossfree["learning_rate"]),
    }
    for field in (
        "global_seed",
        "world_size",
        "global_batch_size",
        "grad_mix",
        "sampler_contract",
    ):
        fields[field] = (
            base["training_provenance"][field],
            lossfree["training_provenance"][field],
        )
    mismatched = sorted(
        field for field, (left, right) in fields.items() if left != right
    )
    if mismatched:
        raise RuntimeError(
            "Base and Loss-Free checkpoints are not a matched pair: "
            + ", ".join(mismatched)
        )
    return {field: left for field, (left, _) in fields.items()}


def _build_assignments(cases, devices):
    def rows(stage_cases):
        return [
            {
                "index": index,
                "case_id": case["id"],
                "checkpoint_roles": list(CHECKPOINT_ROLES),
                "device": devices[(index - 1) % len(devices)],
            }
            for index, case in enumerate(stage_cases, start=1)
        ]

    assignments = {}
    for split in SPLIT_COUNTS:
        split_rows = rows([case for case in cases if case["split"] == split])
        if split == "plumbing":
            assignments["plumbing"] = split_rows
        else:
            assignments[f"{split}-count"] = split_rows
            assignments[f"{split}-credit"] = split_rows
    discovery_cases = [case for case in cases if case["split"] == "discovery"]
    assignments["parameter"] = rows(discovery_cases[:PARAMETER_CASE_COUNT])
    return assignments


def _build_protocol(
    args,
    cases,
    contracts,
    pairing,
    base_cfg,
    lossfree_cfg,
    formula_validation,
):
    if not formula_validation["passed"]:
        raise RuntimeError("Exact parameter-credit formula failed autograd validation")
    model_metadata, source_hashes = _collect_project_source_hashes(
        base_cfg,
        lossfree_cfg,
    )
    return {
        "runner_version": RUNNER_VERSION,
        "cross_checkpoint_version": CROSS_CHECKPOINT_VERSION,
        "credit_balance_probe_version": PROBE_VERSION,
        "scientific_question": (
            "After load is balanced independently in each routed block, does "
            "stable count-adjusted suffix-gradient and parameter-side credit "
            "imbalance remain?"
        ),
        "claim_boundary": (
            "This paired frozen-checkpoint analysis does not establish optimizer "
            "benefit, semantic expert value, FID improvement, or novelty."
        ),
        "checkpoints": dict(contracts),
        "pairing": pairing,
        "manifest": {
            "selection_salt": SELECTION_SALT,
            "latent_root": str(Path(args.latent_root).resolve()),
            "cases": [case_protocol_view(case) for case in cases],
        },
        "settings": {
            "blocks_zero_based": list(BLOCKS),
            "sigmas": list(SIGMAS),
            "duplicate_batch_size": DUPLICATE_BATCH_SIZE,
            "permutation_resamples_per_cell": PERMUTATION_RESAMPLES,
            "output_credit_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "output_credit_bootstrap_seed": BOOTSTRAP_SEED,
            "parameter_cases": PARAMETER_CASE_COUNT,
            "parameter_bootstrap_resamples": PARAMETER_BOOTSTRAP_RESAMPLES,
            "parameter_bootstrap_seed": PARAMETER_BOOTSTRAP_SEED,
            "maximum_native_weight_drift": MAX_NATIVE_WEIGHT_DRIFT,
            "route_selection": (
                "native compute_router selection, including Loss-Free bias"
            ),
            "route_weight": "unbiased native affinity at the selected expert",
            "shared_and_unconditional_scope": "excluded from routed-expert credit",
        },
        "stage_order": [
            "plumbing",
            "discovery_count",
            "parameter_validation",
            "discovery_credit",
            "confirmatory_count",
            "confirmatory_credit",
        ],
        "assignments": _build_assignments(cases, args.devices),
        "formula_validation": formula_validation,
        "model_metadata": model_metadata,
        "project_source_sha256": source_hashes,
        "git": _git_contract(),
        "environment": _runtime_environment(args.devices),
        "output_dir": str(Path(args.output_dir).resolve()),
    }


def _write_or_validate_protocol(output_dir, protocol):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    hash_path = output_dir / "protocol.sha256"
    protocol_sha256 = _json_sha256(protocol)
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("Existing cross-checkpoint protocol differs")
    else:
        write_json_atomic(protocol_path, protocol)
    expected_text = protocol_sha256 + "\n"
    if hash_path.exists():
        if hash_path.read_text(encoding="utf-8") != expected_text:
            raise RuntimeError("Existing cross-checkpoint protocol hash differs")
    else:
        temporary = hash_path.with_suffix(".sha256.tmp")
        temporary.write_text(expected_text, encoding="utf-8")
        os.replace(temporary, hash_path)
    return protocol_path, protocol_sha256


def _assert_protocol_unchanged(protocol_path, protocol_sha256):
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if _json_sha256(protocol) != protocol_sha256:
        raise RuntimeError("On-disk cross-checkpoint protocol content changed")
    hash_path = protocol_path.with_suffix(".sha256")
    if hash_path.read_text(encoding="utf-8") != protocol_sha256 + "\n":
        raise RuntimeError("On-disk cross-checkpoint protocol hash changed")
    return protocol


def _verify_source_hashes(protocol):
    for relative, expected in protocol["project_source_sha256"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Locked project source changed: {relative}")


def _verify_checkpoint_input(contract):
    path = Path(contract["path"])
    if not path.is_file():
        raise RuntimeError(f"Checkpoint disappeared: {path}")
    stat = path.stat()
    if stat.st_size != contract["size"] or stat.st_mtime_ns != contract["mtime_ns"]:
        raise RuntimeError(f"Checkpoint metadata changed: {path}")
    if sha256_file(path) != contract["sha256"]:
        raise RuntimeError(f"Checkpoint content changed: {path}")
    if sha256_file(contract["config"]) != contract["config_sha256"]:
        raise RuntimeError(f"Checkpoint config changed: {contract['config']}")


def _verify_latent_input(protocol, case):
    locked_cases = {
        locked["id"]: locked for locked in protocol["manifest"]["cases"]
    }
    observed = case_protocol_view(case)
    locked = locked_cases.get(observed["id"])
    if locked != observed:
        raise RuntimeError("Latent case differs from the locked manifest")
    path = Path(protocol["manifest"]["latent_root"]) / locked["latent_relative"]
    if not path.is_file() or sha256_file(path) != locked["latent_sha256"]:
        raise RuntimeError(f"Locked latent changed: {path}")
    return path


def _verify_protocol_inputs(protocol, cases):
    for contract in protocol["checkpoints"].values():
        _verify_checkpoint_input(contract)
    if [case_protocol_view(case) for case in cases] != protocol["manifest"]["cases"]:
        raise RuntimeError("Selected cases changed after protocol lock")
    for case in protocol["manifest"]["cases"]:
        _verify_latent_input(protocol, case)
    formula_validation = validate_exact_parameter_credit_formula()
    if formula_validation != protocol["formula_validation"]:
        raise RuntimeError("Exact parameter-credit numerical validation changed")
    _verify_source_hashes(protocol)
    if _git_contract() != protocol["git"]:
        raise RuntimeError("Git commit or upstream state changed after protocol lock")


def _seal_path(result_path):
    result_path = Path(result_path)
    return result_path.with_suffix(result_path.suffix + ".seal.json")


def _seal_payload(result, protocol_sha256, artifact_id):
    return {
        "version": SEAL_VERSION,
        "artifact_id": artifact_id,
        "protocol_sha256": protocol_sha256,
        "result_sha256": _json_sha256(result),
    }


def _publish_result(result_path, result, protocol_sha256, artifact_id):
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = result_path.with_suffix(result_path.suffix + ".pending")
    pending_seal = result_path.with_suffix(result_path.suffix + ".pending.seal.json")
    write_json_atomic(pending_path, result)
    seal = _seal_payload(result, protocol_sha256, artifact_id)
    write_json_atomic(pending_seal, seal)
    persisted = json.loads(pending_path.read_text(encoding="utf-8"))
    if persisted != result or seal != _seal_payload(
        persisted,
        protocol_sha256,
        artifact_id,
    ):
        raise RuntimeError("Pending cross-checkpoint result failed its seal")
    os.replace(pending_path, result_path)
    os.replace(pending_seal, _seal_path(result_path))


def _load_sealed_payload(result_path, protocol_sha256, artifact_id):
    result_path = Path(result_path)
    seal_path = _seal_path(result_path)
    if not result_path.exists() and not seal_path.exists():
        return None
    if result_path.exists() != seal_path.exists():
        raise RuntimeError(f"Partial sealed artifact requires inspection: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal != _seal_payload(result, protocol_sha256, artifact_id):
        raise RuntimeError(f"Artifact seal mismatch: {result_path}")
    return result


def _case_result_path(output_dir, stage, role, index, case_id):
    return (
        Path(output_dir)
        / stage
        / role
        / f"{int(index):03d}_{case_id}.json"
    )


def _case_artifact_id(stage, role, case_id):
    return f"case:{stage}:{role}:{case_id}"


def _validate_case_result(
    result,
    case,
    stage,
    role,
    protocol,
    protocol_sha256,
):
    if result.get("cross_checkpoint_probe_version") != CROSS_CHECKPOINT_VERSION:
        raise RuntimeError("Case result cross-checkpoint version changed")
    if result.get("credit_balance_probe_version") != PROBE_VERSION:
        raise RuntimeError("Case result credit-balance version changed")
    if result.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("Case result belongs to another protocol")
    if result.get("batch_case") != case_protocol_view(case):
        raise RuntimeError("Case result metadata differs from the locked case")
    if result.get("checkpoint_role") != role:
        raise RuntimeError("Case result checkpoint role changed")
    checkpoint = protocol["checkpoints"][role]
    if result.get("checkpoint_sha256") != checkpoint["sha256"]:
        raise RuntimeError("Case result checkpoint SHA256 changed")
    if result.get("block_indices") != list(BLOCKS):
        raise RuntimeError("Case result block list changed")
    if result.get("sigmas") != list(SIGMAS):
        raise RuntimeError("Case result sigma list changed")
    expected_scope = STAGE_MEASUREMENT_SCOPE.get(stage)
    if expected_scope is None:
        raise RuntimeError(f"Unknown sealed result stage: {stage}")
    include_parameter = expected_scope == "parameter"
    if result.get("measurement_scope") != expected_scope:
        raise RuntimeError("Case result measurement scope changed")
    if result.get("includes_parameter_credit") is not include_parameter:
        raise RuntimeError("Case result parameter-credit scope changed")
    cells = result.get("cells")
    if not isinstance(cells, list):
        raise RuntimeError("Case result cells are missing")
    expected_cells = {(block, sigma) for block in BLOCKS for sigma in SIGMAS}
    observed_cells = {
        (int(cell["block_index"]), float(cell["sigma"])) for cell in cells
    }
    if len(cells) != len(expected_cells) or observed_cells != expected_cells:
        raise RuntimeError("Case result block/sigma cells changed")
    if stage == "plumbing":
        if result.get("efficacy_hidden") is not True:
            raise RuntimeError("Published plumbing result is not efficacy-hidden")
        for cell in cells:
            if set(cell) != PLUMBING_CELL_KEYS:
                raise RuntimeError("Published plumbing cell leaks efficacy fields")
    else:
        for cell in cells:
            if "statistics" not in cell or "numerical_controls" not in cell:
                raise RuntimeError("Published efficacy cell is incomplete")
            if expected_scope == "count":
                if set(cell) != COUNT_CELL_KEYS:
                    raise RuntimeError("Published count cell leaks efficacy fields")
                if set(cell["statistics"]) != COUNT_STATISTICS_KEYS:
                    raise RuntimeError("Published count statistics leak efficacy")
                if "nonfinite_token_credits" in cell["numerical_controls"]:
                    raise RuntimeError("Published count controls leak credit fields")
            has_parameter = "parameter_statistics" in cell
            if has_parameter is not include_parameter:
                raise RuntimeError("Published parameter cell scope changed")
    return result


def _result_for_publish(result, stage):
    if stage != "plumbing":
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


def _load_case_result(
    output_dir,
    case,
    stage,
    role,
    index,
    protocol,
    protocol_sha256,
):
    path = _case_result_path(output_dir, stage, role, index, case["id"])
    result = _load_sealed_payload(
        path,
        protocol_sha256,
        _case_artifact_id(stage, role, case["id"]),
    )
    if result is None:
        return None
    return _validate_case_result(
        result,
        case,
        stage,
        role,
        protocol,
        protocol_sha256,
    )


def _load_stage_results(
    output_dir,
    cases,
    stage,
    role,
    protocol,
    protocol_sha256,
):
    results = []
    for index, case in enumerate(cases, start=1):
        result = _load_case_result(
            output_dir,
            case,
            stage,
            role,
            index,
            protocol,
            protocol_sha256,
        )
        if result is None:
            raise RuntimeError(
                f"Missing {stage}/{role} case result: {case['id']}"
            )
        results.append(result)
    return results


def _summary_path(output_dir, name):
    return Path(output_dir) / f"{name}-summary.json"


def _summary_artifact_id(name):
    return f"summary:{name}"


def _load_summary(output_dir, name, protocol_sha256):
    summary = _load_sealed_payload(
        _summary_path(output_dir, name),
        protocol_sha256,
        _summary_artifact_id(name),
    )
    if summary is None:
        raise RuntimeError(f"Required {name} summary is missing")
    if summary.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError(f"{name} summary belongs to another protocol")
    if summary.get("name") != name:
        raise RuntimeError(f"{name} summary name changed")
    return summary


def _publish_summary(output_dir, name, payload, protocol_sha256):
    summary = {
        "runner_version": RUNNER_VERSION,
        "cross_checkpoint_version": CROSS_CHECKPOINT_VERSION,
        "name": name,
        "protocol": str(Path(output_dir) / "protocol.json"),
        "protocol_sha256": protocol_sha256,
        **payload,
    }
    path = _summary_path(output_dir, name)
    if path.exists() or _seal_path(path).exists():
        existing = _load_summary(output_dir, name, protocol_sha256)
        if existing != summary:
            raise RuntimeError(f"Existing {name} summary differs on recomputation")
        return path
    _publish_result(
        path,
        summary,
        protocol_sha256,
        _summary_artifact_id(name),
    )
    return path


def _run_device_cases(payload):
    device = torch.device(payload["device"])
    torch.cuda.set_device(device)
    thread_config = _configure_torch_threads(LOCKED_NUM_THREADS)
    protocol = _assert_protocol_unchanged(
        payload["protocol"],
        payload["protocol_sha256"],
    )
    _verify_source_hashes(protocol)
    completed = []
    for role in payload["roles"]:
        checkpoint = protocol["checkpoints"][role]
        _verify_checkpoint_input(checkpoint)
        runtime_cfg = load_runtime_cfg(checkpoint["config"])
        model, state_name, checkpoint_step, load_seconds = _load_checkpoint_model(
            runtime_cfg,
            checkpoint["path"],
            device,
        )
        _verify_checkpoint_input(checkpoint)
        if state_name != CHECKPOINT_STATE or checkpoint_step != checkpoint["step"]:
            raise RuntimeError("Worker loaded the wrong checkpoint state or step")
        validate_cross_checkpoint_model(
            model,
            require_lossfree_bias=role == "lossfree",
        )
        try:
            for job in payload["jobs"]:
                case = job["case"]
                stage = payload["stage"]
                latent_path = _verify_latent_input(protocol, case)
                result_path = _case_result_path(
                    protocol["output_dir"],
                    stage,
                    role,
                    job["index"],
                    case["id"],
                )
                reused = _load_case_result(
                    protocol["output_dir"],
                    case,
                    stage,
                    role,
                    job["index"],
                    protocol,
                    payload["protocol_sha256"],
                )
                if reused is not None:
                    completed.append({
                        "case_id": case["id"],
                        "checkpoint_role": role,
                        "reused": True,
                    })
                    continue
                torch.cuda.reset_peak_memory_stats(device)
                result = run_cross_checkpoint_credit_balance_case(
                    model=model,
                    runtime_cfg=runtime_cfg,
                    latent_path=latent_path,
                    label=case["label"],
                    seed=case["seed"],
                    case_id=case["id"],
                    measurement_scope=STAGE_MEASUREMENT_SCOPE[stage],
                )
                _verify_latent_input(protocol, case)
                torch.cuda.synchronize(device)
                result.update({
                    "checkpoint_role": role,
                    "checkpoint": checkpoint["path"],
                    "checkpoint_sha256": checkpoint["sha256"],
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_state": state_name,
                    "checkpoint_load_seconds": float(load_seconds),
                    "device": str(device),
                    "thread_configuration": thread_config,
                    "peak_cuda_memory_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "batch_case": case_protocol_view(case),
                    "protocol_sha256": payload["protocol_sha256"],
                })
                result = _result_for_publish(result, stage)
                _validate_case_result(
                    result,
                    case,
                    stage,
                    role,
                    protocol,
                    payload["protocol_sha256"],
                )
                artifact_id = _case_artifact_id(stage, role, case["id"])
                _publish_result(
                    result_path,
                    result,
                    payload["protocol_sha256"],
                    artifact_id,
                )
                completed.append({
                    "case_id": case["id"],
                    "checkpoint_role": role,
                    "reused": False,
                })
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        _verify_checkpoint_input(checkpoint)
    _verify_source_hashes(protocol)
    return {
        "device": str(device),
        "completed": completed,
        "thread_configuration": thread_config,
    }


def _run_stage_cases(
    stage,
    cases,
    roles,
    devices,
    protocol_path,
    protocol_sha256,
):
    payloads = []
    for device in devices:
        jobs = [
            {"index": index, "case": case}
            for index, case in enumerate(cases, start=1)
            if devices[(index - 1) % len(devices)] == device
        ]
        payloads.append({
            "device": device,
            "stage": stage,
            "roles": list(roles),
            "jobs": jobs,
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_sha256,
        })
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


def _finite_control_value(controls, key):
    value = float(controls[key])
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"Numerical control must be finite and nonnegative: {key}")
    return value


def _numerical_safety(results, require_parameter=False, expected_bias=None):
    scopes = {result.get("measurement_scope") for result in results}
    if len(scopes) != 1 or None in scopes:
        raise ValueError("Numerical safety requires one measurement scope")
    measurement_scope = scopes.pop()
    measures_credit = measurement_scope in {"output", "parameter"}
    if require_parameter and measurement_scope != "parameter":
        raise ValueError("Parameter safety requires parameter measurements")
    maxima = {
        "native_output_drift": 0.0,
        "native_relative_mse_drift": 0.0,
        "native_weight_drift": 0.0,
        "repeated_weight_drift": 0.0,
    }
    totals = {
        "route_mismatches": 0,
        "unbiased_argmax_mismatches": 0,
        "repeated_route_mismatches": 0,
        "nonfinite_token_credits": 0,
        "nonfinite_parameter_credits": 0,
        "bias_contract_mismatches": 0,
    }
    for result in results:
        for cell in result["cells"]:
            controls = cell["numerical_controls"]
            control_maxima = {
                "native_output_drift": "max_abs_native_output_drift",
                "native_relative_mse_drift": "native_relative_mse_drift",
                "native_weight_drift": "max_abs_native_weight_drift",
                "repeated_weight_drift": "max_abs_repeated_weight_drift",
            }
            for maximum_key, control_key in control_maxima.items():
                maxima[maximum_key] = max(
                    maxima[maximum_key],
                    _finite_control_value(controls, control_key),
                )
            for key in (
                "route_mismatches",
                "unbiased_argmax_mismatches",
                "repeated_route_mismatches",
            ):
                totals[key] += int(controls[key])
            if measures_credit:
                totals["nonfinite_token_credits"] += int(
                    controls["nonfinite_token_credits"]
                )
                totals["nonfinite_parameter_credits"] += int(
                    controls["nonfinite_parameter_credits"]
                )
            if (
                expected_bias is not None
                and bool(controls["lossfree_bias_enabled"])
                is not bool(expected_bias)
            ):
                totals["bias_contract_mismatches"] += 1
    checks = {
        "native_output_drift": (
            maxima["native_output_drift"]
            <= SAFETY_REQUIREMENTS["maximum_native_output_drift"]
        ),
        "native_relative_mse_drift": (
            maxima["native_relative_mse_drift"]
            <= SAFETY_REQUIREMENTS["maximum_native_relative_mse_drift"]
        ),
        "native_weight_drift": (
            maxima["native_weight_drift"] <= MAX_NATIVE_WEIGHT_DRIFT
        ),
        "repeated_weight_drift": maxima["repeated_weight_drift"] == 0.0,
        "route_mismatches": totals["route_mismatches"] == 0,
        "repeated_route_mismatches": totals["repeated_route_mismatches"] == 0,
        "bias_contract_mismatches": totals["bias_contract_mismatches"] == 0,
    }
    if measures_credit:
        checks["nonfinite_token_credits"] = (
            totals["nonfinite_token_credits"] == 0
        )
    if require_parameter:
        checks["nonfinite_parameter_credits"] = (
            totals["nonfinite_parameter_credits"] == 0
        )
    return {
        "maxima": maxima,
        "totals": totals,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "measurement_scope": measurement_scope,
        "note": (
            "unbiased_argmax_mismatches are reported, not failed, because "
            "Loss-Free intentionally selects with a non-gradient bias"
        ),
    }


def _load_paired_results(output_dir, cases, stage, protocol, protocol_sha256):
    return {
        role: _load_stage_results(
            output_dir,
            cases,
            stage,
            role,
            protocol,
            protocol_sha256,
        )
        for role in CHECKPOINT_ROLES
    }


def _paired_safety(results_by_role, require_parameter=False):
    return {
        role: _numerical_safety(
            results,
            require_parameter=require_parameter,
            expected_bias=role == "lossfree",
        )
        for role, results in results_by_role.items()
    }


def _require_passed_summary(output_dir, name, protocol_sha256):
    summary = _load_summary(output_dir, name, protocol_sha256)
    if not summary.get("passed"):
        raise RuntimeError(f"{name} did not unlock the next stage")
    return summary


def _stage_plumbing(
    output_dir,
    cases,
    devices,
    protocol,
    protocol_path,
    protocol_sha256,
):
    split_cases = [case for case in cases if case["split"] == "plumbing"]
    _run_stage_cases(
        "plumbing",
        split_cases,
        CHECKPOINT_ROLES,
        devices,
        protocol_path,
        protocol_sha256,
    )
    results = _load_paired_results(
        output_dir,
        split_cases,
        "plumbing",
        protocol,
        protocol_sha256,
    )
    probe_safety = _paired_safety(results)
    base_compatible_safety = {
        role: aggregate_credit_balance(role_results, "plumbing")
        for role, role_results in results.items()
    }
    passed = bool(all(
        probe_safety[role]["passed"] and base_compatible_safety[role]["passed"]
        for role in CHECKPOINT_ROLES
    ))
    path = _publish_summary(
        output_dir,
        "plumbing",
        {
            "case_ids": [case["id"] for case in split_cases],
            "efficacy_hidden": True,
            "probe_safety": probe_safety,
            "base_compatible_safety": base_compatible_safety,
            "passed": passed,
        },
        protocol_sha256,
    )
    return path, passed


def _stage_discovery_count(
    output_dir,
    cases,
    devices,
    protocol,
    protocol_path,
    protocol_sha256,
):
    _require_passed_summary(output_dir, "plumbing", protocol_sha256)
    split_cases = [case for case in cases if case["split"] == "discovery"]
    _run_stage_cases(
        "discovery-count",
        split_cases,
        CHECKPOINT_ROLES,
        devices,
        protocol_path,
        protocol_sha256,
    )
    results = _load_paired_results(
        output_dir,
        split_cases,
        "discovery-count",
        protocol,
        protocol_sha256,
    )
    safety = _paired_safety(results)
    count_balance = evaluate_count_balance(
        results["lossfree"],
        results["base"],
        "discovery",
    )
    passed = bool(
        all(row["passed"] for row in safety.values())
        and count_balance["passed"]
    )
    path = _publish_summary(
        output_dir,
        "discovery-count",
        {
            "case_ids": [case["id"] for case in split_cases],
            "probe_safety": safety,
            "count_balance": count_balance,
            "credit_efficacy_deferred": True,
            "passed": passed,
        },
        protocol_sha256,
    )
    return path, passed


def _stage_parameter_validation(
    output_dir,
    cases,
    devices,
    protocol,
    protocol_path,
    protocol_sha256,
):
    discovery_count = _require_passed_summary(
        output_dir,
        "discovery-count",
        protocol_sha256,
    )
    discovery_cases = [case for case in cases if case["split"] == "discovery"]
    parameter_cases = discovery_cases[:PARAMETER_CASE_COUNT]
    _run_stage_cases(
        "parameter",
        parameter_cases,
        CHECKPOINT_ROLES,
        devices,
        protocol_path,
        protocol_sha256,
    )
    parameter_results = _load_paired_results(
        output_dir,
        parameter_cases,
        "parameter",
        protocol,
        protocol_sha256,
    )
    parameter_gate = aggregate_parameter_credit_validation(parameter_results)
    parameter_safety = _paired_safety(parameter_results, require_parameter=True)
    parameter_passed = bool(
        parameter_gate["passed"]
        and all(row["passed"] for row in parameter_safety.values())
    )
    parameter_path = _publish_summary(
        output_dir,
        "parameter",
        {
            "case_ids": [case["id"] for case in parameter_cases],
            "formula_validation": protocol["formula_validation"],
            "probe_safety": parameter_safety,
            "gate": parameter_gate,
            "passed": parameter_passed,
        },
        protocol_sha256,
    )
    if not parameter_passed:
        return parameter_path, False

    count_results = _load_paired_results(
        output_dir,
        discovery_cases,
        "discovery-count",
        protocol,
        protocol_sha256,
    )
    recomputed_count = evaluate_count_balance(
        count_results["lossfree"],
        count_results["base"],
        "discovery",
    )
    if recomputed_count != discovery_count["count_balance"]:
        raise RuntimeError("Discovery count gate changed before credit aggregation")
    _run_stage_cases(
        "discovery-credit",
        discovery_cases,
        CHECKPOINT_ROLES,
        devices,
        protocol_path,
        protocol_sha256,
    )
    credit_results = _load_paired_results(
        output_dir,
        discovery_cases,
        "discovery-credit",
        protocol,
        protocol_sha256,
    )
    count_replay = {
        role: evaluate_count_replay(
            count_results[role],
            credit_results[role],
            "discovery",
        )
        for role in CHECKPOINT_ROLES
    }
    credit_safety = _paired_safety(credit_results)
    lossfree_credit = aggregate_credit_balance(
        credit_results["lossfree"],
        "discovery",
    )
    base_credit = aggregate_credit_balance(credit_results["base"], "discovery")
    discovery_passed = bool(
        recomputed_count["passed"]
        and all(row["passed"] for row in count_replay.values())
        and all(row["passed"] for row in credit_safety.values())
        and lossfree_credit["passed"]
        and base_credit["passed"]
    )
    discovery_path = _publish_summary(
        output_dir,
        "discovery-credit",
        {
            "case_ids": [case["id"] for case in discovery_cases],
            "count_balance": recomputed_count,
            "count_replay": count_replay,
            "probe_safety": credit_safety,
            "lossfree_credit_gate": lossfree_credit,
            "paired_base_credit_gate": base_credit,
            "passed": discovery_passed,
        },
        protocol_sha256,
    )
    print(f"Saved: {discovery_path}")
    return discovery_path, discovery_passed


def _stage_confirmatory(
    output_dir,
    cases,
    devices,
    protocol,
    protocol_path,
    protocol_sha256,
):
    _require_passed_summary(output_dir, "parameter", protocol_sha256)
    discovery_credit = _require_passed_summary(
        output_dir,
        "discovery-credit",
        protocol_sha256,
    )
    split_cases = [case for case in cases if case["split"] == "confirmatory"]
    _run_stage_cases(
        "confirmatory-count",
        split_cases,
        CHECKPOINT_ROLES,
        devices,
        protocol_path,
        protocol_sha256,
    )
    count_results = _load_paired_results(
        output_dir,
        split_cases,
        "confirmatory-count",
        protocol,
        protocol_sha256,
    )
    count_safety = _paired_safety(count_results)
    count_balance = evaluate_count_balance(
        count_results["lossfree"],
        count_results["base"],
        "confirmatory",
    )
    count_passed = bool(
        all(row["passed"] for row in count_safety.values())
        and count_balance["passed"]
    )
    count_path = _publish_summary(
        output_dir,
        "confirmatory-count",
        {
            "case_ids": [case["id"] for case in split_cases],
            "probe_safety": count_safety,
            "count_balance": count_balance,
            "credit_efficacy_deferred": True,
            "passed": count_passed,
        },
        protocol_sha256,
    )
    if not count_passed:
        return count_path, False

    _run_stage_cases(
        "confirmatory-credit",
        split_cases,
        CHECKPOINT_ROLES,
        devices,
        protocol_path,
        protocol_sha256,
    )
    credit_results = _load_paired_results(
        output_dir,
        split_cases,
        "confirmatory-credit",
        protocol,
        protocol_sha256,
    )
    count_replay = {
        role: evaluate_count_replay(
            count_results[role],
            credit_results[role],
            "confirmatory",
        )
        for role in CHECKPOINT_ROLES
    }
    credit_safety = _paired_safety(credit_results)
    lossfree_credit = aggregate_credit_balance(
        credit_results["lossfree"],
        "confirmatory",
        discovery_summary=discovery_credit["lossfree_credit_gate"],
    )
    base_credit = aggregate_credit_balance(
        credit_results["base"],
        "confirmatory",
        discovery_summary=discovery_credit["paired_base_credit_gate"],
    )
    passed = bool(
        all(row["passed"] for row in count_replay.values())
        and all(row["passed"] for row in credit_safety.values())
        and lossfree_credit["passed"]
        and base_credit["passed"]
    )
    path = _publish_summary(
        output_dir,
        "confirmatory",
        {
            "case_ids": [case["id"] for case in split_cases],
            "count_summary": str(count_path),
            "count_balance": count_balance,
            "count_replay": count_replay,
            "probe_safety": credit_safety,
            "lossfree_credit_gate": lossfree_credit,
            "paired_base_credit_gate": base_credit,
            "passed": passed,
        },
        protocol_sha256,
    )
    return path, passed


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a matched Base/Loss-Free checkpoint pair: per-block load "
            "balance, count-adjusted suffix-gradient credit and exact "
            "parameter-side empirical Fisher."
        )
    )
    parser.add_argument(
        "--base-ckpt",
        required=True,
        help="ProMoE_TC_B checkpoint written by the current train.py",
    )
    parser.add_argument(
        "--base-config",
        required=True,
        help="config the Base checkpoint was trained with",
    )
    parser.add_argument(
        "--lossfree-ckpt",
        required=True,
        help="ProMoE_TC_B_lossfree checkpoint from a matched training run",
    )
    parser.add_argument(
        "--lossfree-config",
        required=True,
        help="config the Loss-Free checkpoint was trained with",
    )
    parser.add_argument("--latent-root", default=DEFAULT_LATENT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=repository_output_dir,
        required=True,
        help="git-ignored directory inside this repository",
    )
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=LOCKED_DEVICES,
    )
    parser.add_argument(
        "--stage",
        choices=("plumbing", "discovery", "parameter", "confirmatory"),
        default="plumbing",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    checkpoints = {
        "base": Path(args.base_ckpt).resolve(),
        "lossfree": Path(args.lossfree_ckpt).resolve(),
    }
    configs = {
        "base": Path(args.base_config).resolve(),
        "lossfree": Path(args.lossfree_config).resolve(),
    }
    model_names = {"base": MODEL_NAME, "lossfree": LOSSFREE_MODEL_NAME}
    cases = select_cases(args.latent_root)
    base_cfg = load_runtime_cfg(configs["base"])
    lossfree_cfg = load_runtime_cfg(configs["lossfree"])
    formula_validation = validate_exact_parameter_credit_formula()
    if not formula_validation["passed"]:
        raise RuntimeError("Exact parameter-credit formula failed autograd validation")
    dataset_identity = _latent_dataset_identity(args.latent_root)
    contracts = {
        role: _checkpoint_contract(
            checkpoints[role],
            configs[role],
            model_names[role],
            dataset_identity,
        )
        for role in CHECKPOINT_ROLES
    }
    pairing = _validate_paired_training(contracts)
    protocol = _build_protocol(
        args=args,
        cases=cases,
        contracts=contracts,
        pairing=pairing,
        base_cfg=base_cfg,
        lossfree_cfg=lossfree_cfg,
        formula_validation=formula_validation,
    )
    protocol_path, protocol_sha256 = _write_or_validate_protocol(
        output_dir,
        protocol,
    )
    print(f"Locked protocol: {protocol_path}")
    print(f"Protocol SHA256: {protocol_sha256}")
    print(f"Paired checkpoint step: {pairing['checkpoint_step']}")
    if args.prepare_only:
        return

    _verify_protocol_inputs(protocol, cases)
    lock_path = output_dir / ".orchestration.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another cross-checkpoint orchestrator is running"
            ) from error
        run_stage = {
            "plumbing": _stage_plumbing,
            "discovery": _stage_discovery_count,
            "parameter": _stage_parameter_validation,
            "confirmatory": _stage_confirmatory,
        }[args.stage]
        summary_path, passed = run_stage(
            output_dir,
            cases,
            args.devices,
            protocol,
            protocol_path,
            protocol_sha256,
        )
        _assert_protocol_unchanged(protocol_path, protocol_sha256)
        _verify_protocol_inputs(protocol, cases)
        print(json.dumps({
            "stage": args.stage,
            "passed": passed,
            "summary": str(summary_path),
        }, indent=2, sort_keys=True))
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
