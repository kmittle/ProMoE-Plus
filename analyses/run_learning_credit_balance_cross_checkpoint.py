#!/usr/bin/env python3
"""Run the sealed Base/Loss-Free count and exact-credit gate."""

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
from analyses.t_SNE.checkpoint_utils import load_runtime_cfg, parse_checkpoint_step
from analyses.timestep_utility.credit_balance_batch import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CHECKPOINT_STATE,
    CHECKPOINT_STEP,
    CONFIRMATORY_REQUIREMENTS,
    DISCOVERY_REQUIREMENTS,
    EXPECTED_WEIGHTS_SHA256,
    EXPECTED_WEIGHTS_SIZE,
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
    MAX_BLOCK_COUNT_CV,
    MAX_BLOCK_COUNT_GINI,
    MAX_BLOCK_COUNT_RATIO,
    MIN_BLOCK_FRACTIONAL_REDUCTION,
    MIN_PARAMETER_ACTIVE_EXPERTS,
    MIN_PARAMETER_BOOTSTRAP_LCB,
    MIN_PARAMETER_MEAN_SPEARMAN,
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


RUNNER_VERSION = 1
SEAL_VERSION = 1
PARAMETER_CASE_COUNT = 16
LOSSFREE_MODEL_NAME = "ProMoE_TC_B_lossfree"
LOSSFREE_GLOBAL_SEED = 0
LOSSFREE_WORLD_SIZE = 4
LOSSFREE_TRAINER_STATE_VERSION = 2
LOSSFREE_AUGMENTATION_SEED_VERSION = 1
LOSSFREE_SAMPLER_CONTRACT_VERSION = 1
LOSSFREE_DATASET_IDENTITY_VERSION = 1
LOSSFREE_DATASET_TYPE = "__mp_main__.LatentFolder"
LOCKED_DEVICES = ("cuda:4", "cuda:5", "cuda:6", "cuda:7")
PREREGISTRATIONS = (
    {
        "version": 1,
        "path": (
            "/home/dev/promoe-probes/"
            "credit-balance-lossfree-s0-200k-v1-preregister.json"
        ),
        "sha256": (
            "59ce95f39220511c510b589b78e69b0139c961aaa1d3e4e3f013c16312565a43"
        ),
    },
    {
        "version": 2,
        "path": (
            "/home/dev/promoe-probes/"
            "credit-balance-lossfree-s0-200k-v2-preregister.json"
        ),
        "sha256": (
            "04ced5b1cebf371153c33c4f7b9cf703b58d430ee504d8d52c083a186f254b57"
        ),
    },
)
BASE_PROTOCOL_SHA256 = (
    "9c25bd0144228e921be1a5491dafa32299356f5af00e0a5cc15d857a1eeef096"
)
LOSSFREE_CONFIG_SHA256 = (
    "ce7ce84ad50800ddc66689d8855530ffcb58365297c3ad52845a8bff4d1bcfcc"
)
DEFAULT_BASE_PROTOCOL = (
    "/home/dev/promoe-probes/credit-balance-gate-base200k-v1/protocol.json"
)
DEFAULT_BASE_RESULTS_DIR = (
    "/home/dev/promoe-probes/credit-balance-gate-base200k-v1"
)
DEFAULT_BASE_WEIGHTS = "/home/dev/promoe-probes/base-seed0-ckpt_step_200000.pth"
DEFAULT_BASE_CONFIG = "configs/004_ProMoE_B_seed0_control.yaml"
DEFAULT_LOSSFREE_CHECKPOINT = (
    "/home/dev/promoe-runs/ProMoE_TC_B_lossfree/"
    "004_ProMoE_B_lossfree_u1e2_credit_control_s0_200k/"
    "checkpoints/ckpt_step_200000.pth"
)
DEFAULT_LOSSFREE_CONFIG = (
    "configs/004_ProMoE_B_lossfree_u1e2_credit_control_s0_200k.yaml"
)
DEFAULT_LATENT_ROOT = "/home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz"
DEFAULT_OUTPUT_DIR = (
    "/home/dev/promoe-probes/credit-balance-lossfree-s0-200k-v2"
)
LATENT_PATHS_CACHE = PROJECT_ROOT / "preprocess/latent_paths_cache.txt"
STATIC_SOURCE_PATHS = (
    "requirements.txt",
    "analyses/run_learning_credit_balance_cross_checkpoint.py",
    "analyses/run_learning_credit_balance_probe_batch.py",
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
        LOSSFREE_DATASET_TYPE,
        len(latent_paths),
    )
    normalized_root = os.path.normpath(latent_root)
    for path in latent_paths:
        class_name = os.path.basename(os.path.dirname(path))
        relative = os.path.relpath(os.path.normpath(path), normalized_root)
        _hash_dataset_record(digest, relative, class_to_idx[class_name])
    return {
        "version": DATASET_IDENTITY_VERSION,
        "type": LOSSFREE_DATASET_TYPE,
        "num_samples": len(latent_paths),
        "ordered_samples_sha256": digest.hexdigest(),
    }


def _validate_preregistered_run_inputs(args, base_cfg, lossfree_cfg, documents):
    training = documents[1]["lossfree_training_contract"]
    data = documents[1]["data_contract"]
    if Path(args.lossfree_config).resolve() != Path(training["config"]).resolve():
        raise RuntimeError("Loss-Free config path differs from preregistration")
    if Path(args.lossfree_ckpt).resolve() != Path(
        training["planned_checkpoint"]
    ).resolve():
        raise RuntimeError("Loss-Free checkpoint path differs from preregistration")
    if Path(args.latent_root).resolve() != Path(data["latent_root"]).resolve():
        raise RuntimeError("Latent root differs from preregistration")
    paired_base_seed = documents[1]["paired_base_contract"]["global_seed"]
    if int(base_cfg.global_seed) != int(paired_base_seed):
        raise RuntimeError("Base global seed differs from preregistration")

    expected_config = {
        "model_name": training["model_name"],
        "global_seed": training["global_seed"],
        "total_train_batch_size": training["global_batch_size"],
        "lr": training["learning_rate"],
    }
    for field, expected in expected_config.items():
        _require_preregistered_equal(
            getattr(lossfree_cfg, field),
            expected,
            f"lossfree_runtime_config.{field}",
        )
    _require_preregistered_equal(
        list(lossfree_cfg.gpu_ids),
        [4, 5, 6, 7],
        "lossfree_runtime_config.gpu_ids",
    )
    _require_preregistered_equal(
        Path(lossfree_cfg.latent_data_path).resolve(),
        Path(data["latent_root"]).resolve(),
        "lossfree_runtime_config.latent_data_path",
    )
    if not bool(lossfree_cfg.use_encoded_latents):
        raise RuntimeError("Loss-Free training must use the preregistered latent set")
    moe_cfg = lossfree_cfg.DiT_B_config.MoE_config
    _require_preregistered_equal(
        bool(moe_cfg.use_lossfree_bias),
        True,
        "lossfree_runtime_config.use_lossfree_bias",
    )
    _require_preregistered_equal(
        float(moe_cfg.bias_update_rate),
        float(training["bias_update_rate"]),
        "lossfree_runtime_config.bias_update_rate",
    )
    if int(lossfree_cfg.num_steps) <= CHECKPOINT_STEP:
        raise RuntimeError("Loss-Free config does not train through step 200000")
    if CHECKPOINT_STEP % int(lossfree_cfg.save_ckpt_interval) != 0:
        raise RuntimeError("Loss-Free config does not save the planned checkpoint")
    dataset_identity = _latent_dataset_identity(args.latent_root)

    world_size = len(lossfree_cfg.gpu_ids)
    _require_preregistered_equal(
        world_size,
        LOSSFREE_WORLD_SIZE,
        "lossfree_runtime_config.world_size",
    )
    global_batch_size = int(lossfree_cfg.total_train_batch_size)
    if global_batch_size % world_size != 0:
        raise RuntimeError("Loss-Free global batch is not divisible by world size")
    grad_mix = int(getattr(lossfree_cfg, "grad_mix", 1))
    if grad_mix <= 0:
        raise RuntimeError("Loss-Free grad_mix must be positive")
    return {
        "global_seed": int(training["global_seed"]),
        "world_size": world_size,
        "global_batch_size": global_batch_size,
        "per_rank_batch_size": global_batch_size // world_size,
        "grad_mix": grad_mix,
        "checkpoint_step": CHECKPOINT_STEP,
        "dataset_identity": dataset_identity,
    }


def _require_nonnegative_integer(value, field, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"Loss-Free trainer provenance is invalid: {field}")
    if positive and value == 0:
        raise RuntimeError(f"Loss-Free trainer provenance is invalid: {field}")
    return value


def _validate_sha256(value, field):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"Loss-Free trainer provenance is invalid: {field}")
    return value


def _validate_checkpoint_rng_state(state):
    required_keys = {"python", "numpy", "torch", "cuda"}
    if not isinstance(state, dict) or set(state) != required_keys:
        raise RuntimeError("Loss-Free rank RNG provenance is incomplete")
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
        raise RuntimeError("Loss-Free rank RNG provenance is invalid") from error


def _checkpoint_training_provenance(checkpoint, expected):
    trainer_state = checkpoint.get("trainer_state")
    if not isinstance(trainer_state, dict):
        raise RuntimeError("Loss-Free checkpoint lacks trainer provenance")
    expected_fields = {
        "version": LOSSFREE_TRAINER_STATE_VERSION,
        "augmentation_seed_version": LOSSFREE_AUGMENTATION_SEED_VERSION,
        "global_seed": expected["global_seed"],
        "world_size": expected["world_size"],
        "grad_mix": expected["grad_mix"],
        "next_step": expected["checkpoint_step"] + 1,
        "data_batches_seen": (
            (expected["checkpoint_step"] + 1) * expected["grad_mix"]
        ),
    }
    expected_trainer_keys = set(expected_fields) | {
        "batches_per_epoch",
        "sampler_epoch",
        "sampler_batch_offset",
        "sampler_contract",
        "rank_states",
    }
    if set(trainer_state) != expected_trainer_keys:
        raise RuntimeError("Loss-Free trainer provenance fields changed")
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
            raise RuntimeError(f"Loss-Free trainer provenance changed: {field}")
    sampler = trainer_state.get("sampler_contract")
    if not isinstance(sampler, dict):
        raise RuntimeError("Loss-Free checkpoint lacks sampler provenance")
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
        raise RuntimeError("Loss-Free sampler provenance fields changed")
    sampler_fields = {
        "version": LOSSFREE_SAMPLER_CONTRACT_VERSION,
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
            raise RuntimeError(f"Loss-Free sampler provenance changed: {field}")
    dataset = sampler.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != {
        "version",
        "type",
        "num_samples",
        "ordered_samples_sha256",
    }:
        raise RuntimeError("Loss-Free sampler dataset provenance is incomplete")
    dataset_version = _require_nonnegative_integer(
        dataset.get("version"),
        "sampler_contract.dataset.version",
        positive=True,
    )
    if dataset_version != LOSSFREE_DATASET_IDENTITY_VERSION:
        raise RuntimeError("Loss-Free sampler dataset version changed")
    if dataset.get("type") != LOSSFREE_DATASET_TYPE:
        raise RuntimeError("Loss-Free sampler dataset type changed")
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
            "Loss-Free sampler dataset differs from the locked latent dataset"
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
        raise RuntimeError("Loss-Free batches_per_epoch is internally inconsistent")
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
        raise RuntimeError("Loss-Free sampler position is internally inconsistent")
    rank_states = trainer_state.get("rank_states")
    if not isinstance(rank_states, list) or len(rank_states) != expected["world_size"]:
        raise RuntimeError("Loss-Free rank RNG provenance is incomplete")
    rank_ids = []
    for state in rank_states:
        if not isinstance(state, dict) or set(state) != {"rank", "rng_state"}:
            raise RuntimeError("Loss-Free rank RNG provenance is incomplete")
        rank_ids.append(state["rank"])
        _validate_checkpoint_rng_state(state["rng_state"])
    if rank_ids != list(range(expected["world_size"])):
        raise RuntimeError("Loss-Free rank RNG provenance IDs are invalid")
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
    }


def _checkpoint_contract(
    checkpoint_path,
    config_path,
    model_name,
    expected_sha256=None,
    expected_size=None,
    expected_training=None,
):
    checkpoint_path = Path(checkpoint_path).resolve()
    config_path = Path(config_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if parse_checkpoint_step(checkpoint_path) != CHECKPOINT_STEP:
        raise ValueError("Cross-checkpoint gate requires step 200000")
    stat = checkpoint_path.stat()
    if expected_size is not None and stat.st_size != int(expected_size):
        raise ValueError("Checkpoint size changed from the locked contract")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
        raise ValueError("Checkpoint SHA256 changed from the locked contract")
    runtime_cfg = load_runtime_cfg(config_path)
    if runtime_cfg.model_name != model_name:
        raise ValueError(f"Checkpoint config model must be {model_name}")
    load_kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only")
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
    if checkpoint.get("step") != CHECKPOINT_STEP:
        raise ValueError("Checkpoint payload is not step 200000")
    if CHECKPOINT_STATE not in checkpoint:
        raise KeyError(f"Checkpoint is missing {CHECKPOINT_STATE}")
    training_provenance = None
    if expected_training is not None:
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
        "step": CHECKPOINT_STEP,
        "state": CHECKPOINT_STATE,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model_name": model_name,
        "training_provenance": training_provenance,
    }


def _verify_preregistrations():
    documents = {}
    for row in PREREGISTRATIONS:
        path = Path(row["path"]).resolve()
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(
                f"Loss-Free preregistration v{row['version']} changed"
            )
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("version") != row["version"]:
            raise RuntimeError("Loss-Free preregistration version changed")
        documents[row["version"]] = document
    _validate_preregistered_constants(documents)
    return documents


def _require_preregistered_equal(observed, expected, field):
    if observed != expected:
        raise RuntimeError(f"Preregistered contract changed: {field}")


def _validate_preregistered_constants(documents):
    if set(documents) != {1, 2}:
        raise RuntimeError("Both Loss-Free preregistrations are required")
    v1 = documents[1]
    v2 = documents[2]
    supersedes = v2.get("supersedes", {})
    _require_preregistered_equal(
        supersedes.get("sha256"),
        PREREGISTRATIONS[0]["sha256"],
        "v2.supersedes.sha256",
    )
    _require_preregistered_equal(
        Path(supersedes.get("path", "")).resolve(),
        Path(PREREGISTRATIONS[0]["path"]).resolve(),
        "v2.supersedes.path",
    )

    training = v1.get("lossfree_training_contract", {})
    _require_preregistered_equal(
        training.get("config_sha256"),
        LOSSFREE_CONFIG_SHA256,
        "lossfree_training_contract.config_sha256",
    )
    _require_preregistered_equal(
        training.get("model_name"),
        LOSSFREE_MODEL_NAME,
        "lossfree_training_contract.model_name",
    )
    _require_preregistered_equal(
        training.get("global_seed"),
        LOSSFREE_GLOBAL_SEED,
        "lossfree_training_contract.global_seed",
    )
    _require_preregistered_equal(
        training.get("planned_step"),
        CHECKPOINT_STEP,
        "lossfree_training_contract.planned_step",
    )
    _require_preregistered_equal(
        training.get("planned_state"),
        CHECKPOINT_STATE,
        "lossfree_training_contract.planned_state",
    )
    _require_preregistered_equal(
        Path(training.get("planned_checkpoint", "")).resolve(),
        Path(DEFAULT_LOSSFREE_CHECKPOINT).resolve(),
        "lossfree_training_contract.planned_checkpoint",
    )

    data = v1.get("data_contract", {})
    _require_preregistered_equal(
        data.get("reuse_exact_manifest_from_protocol_sha256"),
        BASE_PROTOCOL_SHA256,
        "data_contract.reuse_exact_manifest_from_protocol_sha256",
    )
    _require_preregistered_equal(
        data.get("split_case_counts"),
        SPLIT_COUNTS,
        "data_contract.split_case_counts",
    )
    _require_preregistered_equal(
        data.get("blocks_zero_based"),
        list(BLOCKS),
        "data_contract.blocks_zero_based",
    )
    _require_preregistered_equal(
        data.get("sigmas"),
        list(SIGMAS),
        "data_contract.sigmas",
    )
    _require_preregistered_equal(
        data.get("bootstrap_resamples"),
        BOOTSTRAP_RESAMPLES,
        "data_contract.bootstrap_resamples",
    )
    _require_preregistered_equal(
        data.get("permutation_resamples_per_cell"),
        PERMUTATION_RESAMPLES,
        "data_contract.permutation_resamples_per_cell",
    )
    _require_preregistered_equal(
        data.get("paired_checkpoint_inputs"),
        True,
        "data_contract.paired_checkpoint_inputs",
    )

    safety = v1.get("stage_1_numerical_safety", {})
    _require_preregistered_equal(
        safety.get("required_complete_cases"),
        SPLIT_COUNTS["plumbing"],
        "stage_1_numerical_safety.required_complete_cases",
    )
    _require_preregistered_equal(
        safety.get("required_finite_cells"),
        SPLIT_COUNTS["plumbing"] * len(BLOCKS) * len(SIGMAS),
        "stage_1_numerical_safety.required_finite_cells",
    )
    _require_preregistered_equal(
        safety.get("required_route_mismatches"),
        SAFETY_REQUIREMENTS["required_route_mismatches"],
        "stage_1_numerical_safety.required_route_mismatches",
    )
    _require_preregistered_equal(
        safety.get("maximum_native_output_drift"),
        SAFETY_REQUIREMENTS["maximum_native_output_drift"],
        "stage_1_numerical_safety.maximum_native_output_drift",
    )
    _require_preregistered_equal(
        safety.get("maximum_native_relative_mse_drift"),
        SAFETY_REQUIREMENTS["maximum_native_relative_mse_drift"],
        "stage_1_numerical_safety.maximum_native_relative_mse_drift",
    )

    count_gate = v1.get("stage_2_count_balance_precondition", {})
    count_requirements = {
        "maximum_each_block_aggregate_count_cv": MAX_BLOCK_COUNT_CV,
        "maximum_each_block_aggregate_count_gini": MAX_BLOCK_COUNT_GINI,
        "maximum_each_block_count_ratio": MAX_BLOCK_COUNT_RATIO,
        "minimum_fractional_reduction_vs_paired_base_for_each_block_cv": (
            MIN_BLOCK_FRACTIONAL_REDUCTION
        ),
        "minimum_fractional_reduction_vs_paired_base_for_each_block_gini": (
            MIN_BLOCK_FRACTIONAL_REDUCTION
        ),
        "required_all_experts_active": True,
    }
    for field, expected in count_requirements.items():
        _require_preregistered_equal(
            count_gate.get(field),
            expected,
            f"stage_2_count_balance_precondition.{field}",
        )

    credit_gate = v1.get("stage_4_count_adjusted_credit_gate", {})
    for split, expected_requirements in (
        ("discovery", DISCOVERY_REQUIREMENTS),
        ("confirmatory", CONFIRMATORY_REQUIREMENTS),
    ):
        observed = credit_gate.get(split, {})
        for field, expected in expected_requirements.items():
            _require_preregistered_equal(
                observed.get(field),
                expected,
                f"stage_4_count_adjusted_credit_gate.{split}.{field}",
            )
    _require_preregistered_equal(
        credit_gate.get("confirmatory", {}).get("multiple_comparison_correction"),
        "Holm",
        "stage_4_count_adjusted_credit_gate.confirmatory.correction",
    )

    parameter_gate = v2.get("replacement_stage_3", {})
    parameter_requirements = {
        "cells_per_case": len(BLOCKS) * len(SIGMAS),
        "minimum_active_experts_for_cell": MIN_PARAMETER_ACTIVE_EXPERTS,
        "minimum_mean_spearman_each_checkpoint": MIN_PARAMETER_MEAN_SPEARMAN,
        "minimum_image_bootstrap_lcb_each_checkpoint": MIN_PARAMETER_BOOTSTRAP_LCB,
    }
    for field, expected in parameter_requirements.items():
        _require_preregistered_equal(
            parameter_gate.get(field),
            expected,
            f"replacement_stage_3.{field}",
        )
    _require_preregistered_equal(
        parameter_gate.get("numerical_validation", {}).get(
            "maximum_relative_error"
        ),
        1e-5,
        "replacement_stage_3.numerical_validation.maximum_relative_error",
    )


def _load_base_protocol(protocol_path, cases):
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if _json_sha256(protocol) != BASE_PROTOCOL_SHA256:
        raise RuntimeError("Base protocol content hash changed")
    hash_path = protocol_path.with_suffix(".sha256")
    if hash_path.read_text(encoding="utf-8") != BASE_PROTOCOL_SHA256 + "\n":
        raise RuntimeError("Base protocol SHA256 sidecar changed")
    expected_cases = [case_protocol_view(case) for case in cases]
    if protocol.get("manifest", {}).get("cases") != expected_cases:
        raise RuntimeError("Base protocol manifest differs from selected cases")
    for relative, expected in protocol["project_source_sha256"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Base protocol source changed: {relative}")
    preregister = protocol["preregister"]
    if sha256_file(preregister["path"]) != preregister["sha256"]:
        raise RuntimeError("Base preregistration changed")
    return protocol


def _build_assignments(cases, devices):
    assignments = {}
    for split in SPLIT_COUNTS:
        split_cases = [case for case in cases if case["split"] == split]
        rows = [
            {
                "index": index,
                "case_id": case["id"],
                "checkpoint_role": "lossfree",
                "device": devices[(index - 1) % len(devices)],
            }
            for index, case in enumerate(split_cases, start=1)
        ]
        if split == "plumbing":
            assignments["plumbing"] = rows
        else:
            assignments[f"{split}-count"] = rows
            assignments[f"{split}-credit"] = rows
    discovery_cases = [case for case in cases if case["split"] == "discovery"]
    assignments["parameter"] = [
        {
            "index": index,
            "case_id": case["id"],
            "checkpoint_roles": ["base", "lossfree"],
            "device": devices[(index - 1) % len(devices)],
        }
        for index, case in enumerate(
            discovery_cases[:PARAMETER_CASE_COUNT],
            start=1,
        )
    ]
    return assignments


def _build_protocol(
    args,
    cases,
    base_protocol,
    base_contract,
    lossfree_contract,
    base_cfg,
    lossfree_cfg,
    formula_validation,
):
    _verify_preregistrations()
    if lossfree_contract["config_sha256"] != LOSSFREE_CONFIG_SHA256:
        raise RuntimeError("Loss-Free training config changed after preregistration")
    if not formula_validation["passed"]:
        raise RuntimeError("Exact parameter-credit formula failed autograd validation")
    model_metadata, source_hashes = _collect_project_source_hashes(
        base_cfg,
        lossfree_cfg,
    )
    base_protocol_path = Path(args.base_protocol).resolve()
    return {
        "runner_version": RUNNER_VERSION,
        "cross_checkpoint_version": CROSS_CHECKPOINT_VERSION,
        "credit_balance_probe_version": PROBE_VERSION,
        "locked_after_lossfree_step_200000_checkpoint_exists": True,
        "scientific_question": (
            "After load is balanced independently in each routed block, does "
            "stable count-adjusted suffix-gradient and parameter-side credit "
            "imbalance remain?"
        ),
        "claim_boundary": (
            "This paired frozen-checkpoint gate does not establish optimizer "
            "benefit, semantic expert value, FID improvement, or novelty."
        ),
        "effective_preregistrations": [dict(row) for row in PREREGISTRATIONS],
        "base_protocol": {
            "path": str(base_protocol_path),
            "canonical_json_sha256": BASE_PROTOCOL_SHA256,
            "file_sha256": sha256_file(base_protocol_path),
            "hash_sidecar": str(base_protocol_path.with_suffix(".sha256")),
            "base_git": base_protocol["git"],
        },
        "checkpoints": {
            "base": base_contract,
            "lossfree": lossfree_contract,
        },
        "manifest": {
            "reuse_base_protocol_sha256": BASE_PROTOCOL_SHA256,
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
        "base_results_dir": str(Path(args.base_results_dir).resolve()),
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
    _verify_preregistrations()
    base_protocol = _load_base_protocol(protocol["base_protocol"]["path"], cases)
    if base_protocol["git"] != protocol["base_protocol"]["base_git"]:
        raise RuntimeError("Base protocol Git contract changed")
    base_protocol_path = Path(protocol["base_protocol"]["path"])
    if sha256_file(base_protocol_path) != protocol["base_protocol"]["file_sha256"]:
        raise RuntimeError("Base protocol bytes changed")
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


def _base_seal_payload(result, case_id):
    return {
        "version": 1,
        "case_id": case_id,
        "protocol_sha256": BASE_PROTOCOL_SHA256,
        "result_sha256": _json_sha256(result),
    }


def _load_base_results(base_results_dir, split, cases):
    results = []
    for index, case in enumerate(cases, start=1):
        path = (
            Path(base_results_dir)
            / split
            / f"{index:03d}_{case['id']}.json"
        )
        seal_path = path.with_suffix(path.suffix + ".seal.json")
        if not path.is_file() or not seal_path.is_file():
            raise RuntimeError(f"Sealed Base result is missing: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if seal != _base_seal_payload(result, case["id"]):
            raise RuntimeError(f"Base result seal mismatch: {path}")
        if result.get("protocol_sha256") != BASE_PROTOCOL_SHA256:
            raise RuntimeError("Base result belongs to another protocol")
        if result.get("batch_case") != case_protocol_view(case):
            raise RuntimeError("Base result case metadata changed")
        if result.get("credit_balance_probe_version") != PROBE_VERSION:
            raise RuntimeError("Base result probe version changed")
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
        if state_name != CHECKPOINT_STATE or checkpoint_step != CHECKPOINT_STEP:
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
        ("lossfree",),
        devices,
        protocol_path,
        protocol_sha256,
    )
    results = _load_stage_results(
        output_dir,
        split_cases,
        "plumbing",
        "lossfree",
        protocol,
        protocol_sha256,
    )
    legacy_gate = aggregate_credit_balance(results, "plumbing")
    safety = _numerical_safety(results, expected_bias=True)
    passed = bool(legacy_gate["passed"] and safety["passed"])
    path = _publish_summary(
        output_dir,
        "plumbing",
        {
            "case_ids": [case["id"] for case in split_cases],
            "efficacy_hidden": True,
            "probe_safety": safety,
            "base_compatible_safety": legacy_gate,
            "passed": passed,
        },
        protocol_sha256,
    )
    return path, passed


def _stage_discovery_count(
    output_dir,
    cases,
    base_results_dir,
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
        ("lossfree",),
        devices,
        protocol_path,
        protocol_sha256,
    )
    lossfree_results = _load_stage_results(
        output_dir,
        split_cases,
        "discovery-count",
        "lossfree",
        protocol,
        protocol_sha256,
    )
    base_results = _load_base_results(
        base_results_dir,
        "discovery",
        split_cases,
    )
    safety = _numerical_safety(lossfree_results, expected_bias=True)
    count_balance = evaluate_count_balance(
        lossfree_results,
        base_results,
        "discovery",
    )
    passed = bool(safety["passed"] and count_balance["passed"])
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
    base_results_dir,
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
        ("base", "lossfree"),
        devices,
        protocol_path,
        protocol_sha256,
    )
    parameter_results = {
        role: _load_stage_results(
            output_dir,
            parameter_cases,
            "parameter",
            role,
            protocol,
            protocol_sha256,
        )
        for role in ("base", "lossfree")
    }
    parameter_gate = aggregate_parameter_credit_validation(parameter_results)
    parameter_safety = {
        role: _numerical_safety(
            results,
            require_parameter=True,
            expected_bias=role == "lossfree",
        )
        for role, results in parameter_results.items()
    }
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

    count_results = _load_stage_results(
        output_dir,
        discovery_cases,
        "discovery-count",
        "lossfree",
        protocol,
        protocol_sha256,
    )
    base_results = _load_base_results(
        base_results_dir,
        "discovery",
        discovery_cases,
    )
    recomputed_count = evaluate_count_balance(
        count_results,
        base_results,
        "discovery",
    )
    if recomputed_count != discovery_count["count_balance"]:
        raise RuntimeError("Discovery count gate changed before credit aggregation")
    _run_stage_cases(
        "discovery-credit",
        discovery_cases,
        ("lossfree",),
        devices,
        protocol_path,
        protocol_sha256,
    )
    lossfree_results = _load_stage_results(
        output_dir,
        discovery_cases,
        "discovery-credit",
        "lossfree",
        protocol,
        protocol_sha256,
    )
    count_replay = evaluate_count_replay(
        count_results,
        lossfree_results,
        "discovery",
    )
    credit_safety = _numerical_safety(
        lossfree_results,
        expected_bias=True,
    )
    lossfree_credit = aggregate_credit_balance(lossfree_results, "discovery")
    base_credit = aggregate_credit_balance(base_results, "discovery")
    discovery_passed = bool(
        recomputed_count["passed"]
        and count_replay["passed"]
        and credit_safety["passed"]
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
    base_results_dir,
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
        ("lossfree",),
        devices,
        protocol_path,
        protocol_sha256,
    )
    count_results = _load_stage_results(
        output_dir,
        split_cases,
        "confirmatory-count",
        "lossfree",
        protocol,
        protocol_sha256,
    )
    base_results = _load_base_results(
        base_results_dir,
        "confirmatory",
        split_cases,
    )
    count_safety = _numerical_safety(count_results, expected_bias=True)
    count_balance = evaluate_count_balance(
        count_results,
        base_results,
        "confirmatory",
    )
    count_passed = bool(count_safety["passed"] and count_balance["passed"])
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
        ("lossfree",),
        devices,
        protocol_path,
        protocol_sha256,
    )
    lossfree_results = _load_stage_results(
        output_dir,
        split_cases,
        "confirmatory-credit",
        "lossfree",
        protocol,
        protocol_sha256,
    )
    count_replay = evaluate_count_replay(
        count_results,
        lossfree_results,
        "confirmatory",
    )
    credit_safety = _numerical_safety(
        lossfree_results,
        expected_bias=True,
    )
    lossfree_credit = aggregate_credit_balance(
        lossfree_results,
        "confirmatory",
        discovery_summary=discovery_credit["lossfree_credit_gate"],
    )
    discovery_cases = [case for case in cases if case["split"] == "discovery"]
    base_discovery_results = _load_base_results(
        base_results_dir,
        "discovery",
        discovery_cases,
    )
    base_discovery_credit = aggregate_credit_balance(
        base_discovery_results,
        "discovery",
    )
    base_credit = aggregate_credit_balance(
        base_results,
        "confirmatory",
        discovery_summary=base_discovery_credit,
    )
    passed = bool(
        count_replay["passed"]
        and credit_safety["passed"]
        and lossfree_credit["passed"]
        and base_credit["passed"]
    )
    payload = {
        "case_ids": [case["id"] for case in split_cases],
        "count_summary": str(count_path),
        "count_balance": count_balance,
        "count_replay": count_replay,
        "probe_safety": credit_safety,
        "lossfree_credit_gate": lossfree_credit,
        "paired_base_credit_gate": base_credit,
        "passed": passed,
    }
    path = _publish_summary(
        output_dir,
        "confirmatory",
        payload,
        protocol_sha256,
    )
    return path, bool(payload["passed"])


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the sealed Base/Loss-Free step-200K load and exact-credit gate."
        )
    )
    parser.add_argument("--base-protocol", default=DEFAULT_BASE_PROTOCOL)
    parser.add_argument("--base-results-dir", default=DEFAULT_BASE_RESULTS_DIR)
    parser.add_argument("--base-weights-ckpt", default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--lossfree-ckpt", default=DEFAULT_LOSSFREE_CHECKPOINT)
    parser.add_argument("--lossfree-config", default=DEFAULT_LOSSFREE_CONFIG)
    parser.add_argument("--latent-root", default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
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
    base_results_dir = Path(args.base_results_dir).resolve()
    base_config = Path(args.base_config).resolve()
    lossfree_config = Path(args.lossfree_config).resolve()
    cases = select_cases(args.latent_root)
    base_protocol = _load_base_protocol(args.base_protocol, cases)
    base_cfg = load_runtime_cfg(base_config)
    lossfree_cfg = load_runtime_cfg(lossfree_config)
    preregistrations = _verify_preregistrations()
    expected_training = _validate_preregistered_run_inputs(
        args,
        base_cfg,
        lossfree_cfg,
        preregistrations,
    )
    formula_validation = validate_exact_parameter_credit_formula()
    if not formula_validation["passed"]:
        raise RuntimeError("Exact parameter-credit formula failed autograd validation")
    base_contract = _checkpoint_contract(
        args.base_weights_ckpt,
        base_config,
        MODEL_NAME,
        expected_sha256=EXPECTED_WEIGHTS_SHA256,
        expected_size=EXPECTED_WEIGHTS_SIZE,
    )
    locked_base = base_protocol["checkpoint"]
    if (
        base_contract["sha256"] != locked_base["weights_sha256"]
        or base_contract["size"] != locked_base["weights_size"]
        or base_contract["config_sha256"] != locked_base["config_sha256"]
    ):
        raise RuntimeError("Base checkpoint contract differs from its sealed protocol")
    lossfree_contract = _checkpoint_contract(
        args.lossfree_ckpt,
        lossfree_config,
        LOSSFREE_MODEL_NAME,
        expected_training=expected_training,
    )
    protocol = _build_protocol(
        args=args,
        cases=cases,
        base_protocol=base_protocol,
        base_contract=base_contract,
        lossfree_contract=lossfree_contract,
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
    print(f"Loss-Free checkpoint SHA256: {lossfree_contract['sha256']}")
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
        if args.stage == "plumbing":
            summary_path, passed = _stage_plumbing(
                output_dir,
                cases,
                args.devices,
                protocol,
                protocol_path,
                protocol_sha256,
            )
        elif args.stage == "discovery":
            summary_path, passed = _stage_discovery_count(
                output_dir,
                cases,
                base_results_dir,
                args.devices,
                protocol,
                protocol_path,
                protocol_sha256,
            )
        elif args.stage == "parameter":
            summary_path, passed = _stage_parameter_validation(
                output_dir,
                cases,
                base_results_dir,
                args.devices,
                protocol,
                protocol_path,
                protocol_sha256,
            )
        else:
            summary_path, passed = _stage_confirmatory(
                output_dir,
                cases,
                base_results_dir,
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
