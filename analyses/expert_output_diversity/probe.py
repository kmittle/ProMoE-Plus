"""Paired checkpoint gate for expert-output repulsion."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import multiprocessing
import re
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from credit_redistribution.git_provenance import run_git
from analyses.denoising_regret.probe import (
    _build_model,
    _compute_router,
    _configure_torch_threads,
    _extract_prediction,
    _load_latent,
    _load_checkpoint_model,
    _per_sample_mse,
)
from analyses.expert_function.consistency_probe import _evaluate_all_experts
from analyses.routing_translation.probe import RouteInputCapture
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)
from models.models_ProMoE_TC import DiT as BaseProMoE
from models.models_ProMoE_TC_expert_contra import DiT as ExpertContraProMoE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_VERSION = 1
DEFAULT_SIGMAS = (0.2, 0.5, 0.8)
DEFAULT_BLOCK_INDICES = (1, 3, 5, 7, 9, 11)
FORMAL_CHECKPOINT_STEP = 50000
FORMAL_NUM_CASES = 8
FORMAL_NUM_ANCHOR_TOKENS = 32
FORMAL_DEVICES = ("cuda:4", "cuda:5", "cuda:6", "cuda:7")
FORMAL_NUM_THREADS = 4
FORMAL_BOOTSTRAP_RESAMPLES = 20000
FORMAL_BOOTSTRAP_SEED = 0
BASE_CONFIG_STEM = "004_ProMoE_B_fresh_routing_audit_s0_v2"
VARIANT_CONFIG_STEM = (
    "004_ProMoE_B_expert_contra_output_tau5_fresh_s0_v2"
)
BASE_CONFIG_SHA256 = (
    "97fe9376303cc390eada34e2bc82fa903b998b78c82d181486630a25187c0ab6"
)
VARIANT_CONFIG_SHA256 = (
    "01f0d3522b1b9a181e1c50286f022d1960a604c722957f290abbd88205e1e099"
)
CANONICAL_MANIFEST_SHA256 = (
    "41affd3a92f7c407fba33f894a10ee2392fc0cd25d105750c6dc095ea22a4824"
)
BASE_MODEL_NAME = "ProMoE_TC_B"
VARIANT_MODEL_NAME = "ProMoE_TC_B_expert_contra"
TRAINER_STATE_VERSION = 2
AUGMENTATION_SEED_VERSION = 1
EXPECTED_GLOBAL_SEED = 0
EXPECTED_WORLD_SIZE = 4
EXPECTED_GLOBAL_BATCH_SIZE = 256
EXPECTED_LEARNING_RATE = 1e-4
EXPECTED_IMG_NUM_WORKERS = 16
EXPECTED_GRAD_MIX = 1
REQUIRED_COMMON_SOURCE_PATHS = frozenset({
    "requirements.txt",
    "config.py",
    "utils.py",
    "train.py",
    "models/modules.py",
    "credit_redistribution/git_provenance.py",
})
PAIRED_IDENTICAL_SOURCE_PATHS = REQUIRED_COMMON_SOURCE_PATHS - {"train.py"}
BASE_TRAIN_SOURCE_SHA256 = (
    "e1f2f88413b7dd240a7178392fdc5ca8b6b83fc8f7da564176c8e808da345041"
)
VARIANT_TRAIN_SOURCE_SHA256 = (
    "ade2f3e582271607d8889ef961e5f18df110bec433753a4644ff949d1b75a0e1"
)
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{15,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "fresh_base_routing"
    / "manifests"
    / "fresh_base_routing_audit_v1.json"
)
FUNCTION_METRICS = (
    "output_rms",
    "expert_rms_cv",
    "pairwise_l2_rms",
    "normalized_pairwise_l2",
    "pairwise_cosine",
    "relative_expert_residual_rms",
    "normalized_effective_rank",
)
GATE_REQUIREMENTS = {
    "pooled_pairwise_l2_relative_gain": 0.10,
    "maximum_pooled_repulsion_tau5_mean_delta": 0.0,
    "common_normalized_pairwise_l2_relative_gain": 0.03,
    "minimum_positive_effective_rank_cases": 6,
    "maximum_common_output_rms_relative_gain": 0.15,
    "minimum_route_entropy_delta": -0.02,
    "maximum_route_share_delta": 0.03,
    "maximum_denoising_mse_relative_gain": 0.03,
    "minimum_active_experts_per_cell": 2,
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_file_metadata(canonical_path, weights_path, label):
    canonical_path = Path(canonical_path).resolve()
    weights_path = Path(weights_path).resolve()
    canonical_size = canonical_path.stat().st_size
    canonical_sha256 = sha256_file(canonical_path)
    if weights_path == canonical_path:
        weights_size = canonical_size
        weights_sha256 = canonical_sha256
    else:
        weights_size = weights_path.stat().st_size
        if weights_size != canonical_size:
            raise RuntimeError(
                f"{label} local weights size does not match the canonical checkpoint"
            )
        weights_sha256 = sha256_file(weights_path)
    if weights_sha256 != canonical_sha256:
        raise RuntimeError(
            f"{label} local weights hash does not match the canonical checkpoint"
        )
    return {
        "canonical_path": str(canonical_path),
        "canonical_size": canonical_size,
        "canonical_sha256": canonical_sha256,
        "weights_path": str(weights_path),
        "weights_size": weights_size,
        "weights_sha256": weights_sha256,
    }


def _json_sha256(payload):
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _load_yaml(path):
    with Path(path).open("r") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return payload


def _training_config_payload_sha256(payload):
    normalized = dict(payload)
    num_steps = normalized.get("num_steps")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int):
        raise ValueError("Training config num_steps must be an integer")
    normalized["num_steps"] = "<runtime-stop-boundary>"
    return _json_sha256(normalized)


def _verify_committed_source_manifest(commit, source_sha256):
    ancestry = run_git(
        PROJECT_ROOT,
        "merge-base",
        "--is-ancestor",
        commit,
        "refs/remotes/origin/repa",
        text=True,
    )
    if ancestry.returncode != 0:
        raise ValueError(
            "Checkpoint training commit is not an ancestor of origin/repa"
        )
    for relative, expected_sha256 in sorted(source_sha256.items()):
        blob = run_git(
            PROJECT_ROOT,
            "cat-file",
            "blob",
            f"{commit}:{relative}",
            text=False,
        )
        if blob.returncode != 0:
            raise ValueError(
                f"Checkpoint training source is absent from Git: {relative}"
            )
        if hashlib.sha256(blob.stdout).hexdigest() != expected_sha256:
            raise ValueError(
                f"Checkpoint training source hash differs from Git: {relative}"
            )


def _verify_current_runtime_sources(source_sha256, config_stem):
    runtime_sources = {
        "config.py",
        "utils.py",
        "models/modules.py",
    }
    if config_stem == BASE_CONFIG_STEM:
        runtime_sources |= {
            "models/models_ProMoE_TC.py",
            "models/phase_metric.py",
        }
    else:
        runtime_sources.add("models/models_ProMoE_TC_expert_contra.py")
    for relative in sorted(runtime_sources):
        if sha256_file(PROJECT_ROOT / relative) != source_sha256[relative]:
            raise ValueError(
                f"Current analysis runtime differs from training source: {relative}"
            )


def _verify_training_source_contract(commit, source_sha256, config_stem):
    _verify_committed_source_manifest(commit, source_sha256)
    _verify_current_runtime_sources(source_sha256, config_stem)


def _validate_formal_configs(base_config, variant_config, latent_root):
    base_config = Path(base_config).resolve()
    variant_config = Path(variant_config).resolve()
    expected_paths = {
        "base": (
            PROJECT_ROOT / "configs" / f"{BASE_CONFIG_STEM}.yaml"
        ).resolve(),
        "variant": (
            PROJECT_ROOT / "configs" / f"{VARIANT_CONFIG_STEM}.yaml"
        ).resolve(),
    }
    if base_config != expected_paths["base"]:
        raise ValueError(f"Formal Base config must be {expected_paths['base']}")
    if variant_config != expected_paths["variant"]:
        raise ValueError(
            f"Formal variant config must be {expected_paths['variant']}"
        )
    if sha256_file(base_config) != BASE_CONFIG_SHA256:
        raise ValueError("Formal Base config SHA256 changed")
    if sha256_file(variant_config) != VARIANT_CONFIG_SHA256:
        raise ValueError("Formal tau=5 config SHA256 changed")

    base = _load_yaml(base_config)
    variant = _load_yaml(variant_config)
    if base.get("model_name") != BASE_MODEL_NAME:
        raise ValueError("Formal Base model_name changed")
    if variant.get("model_name") != VARIANT_MODEL_NAME:
        raise ValueError("Formal tau=5 model_name changed")
    shared_training_fields = (
        "image_size",
        "total_train_batch_size",
        "lr",
        "weight_decay",
        "global_seed",
        "img_num_workers",
        "prefetch_factor",
        "use_pre_latents",
        "use_encoded_latents",
        "latent_data_path",
        "log_interval",
        "num_steps",
        "save_ckpt_interval",
    )
    for key in shared_training_fields:
        if base.get(key) != variant.get(key):
            raise ValueError(f"Base and tau=5 training configs differ at {key}")
    expected_values = {
        "total_train_batch_size": EXPECTED_GLOBAL_BATCH_SIZE,
        "lr": EXPECTED_LEARNING_RATE,
        "weight_decay": 0,
        "global_seed": EXPECTED_GLOBAL_SEED,
        "img_num_workers": EXPECTED_IMG_NUM_WORKERS,
        "use_pre_latents": True,
        "use_encoded_latents": True,
        "save_ckpt_interval": FORMAL_CHECKPOINT_STEP,
    }
    for key, expected in expected_values.items():
        if base.get(key) != expected:
            raise ValueError(f"Formal training config requires {key}={expected!r}")
    if base.get("gpu_ids") != [0, 1, 2, 3]:
        raise ValueError("Formal Base must use GPUs [0,1,2,3]")
    if variant.get("gpu_ids") != [4, 5, 6, 7]:
        raise ValueError("Formal tau=5 must use GPUs [4,5,6,7]")
    if Path(base["latent_data_path"]).resolve() != Path(latent_root).resolve():
        raise ValueError("Formal latent root differs from the training configs")

    base_model = base.get("DiT_B_config", {})
    variant_model = variant.get("DiT_B_config", {})
    if base_model.get("qk_norm") is not False:
        raise ValueError("Formal Base requires qk_norm=False")
    if variant_model.get("qk_norm") is not False:
        raise ValueError("Formal tau=5 requires qk_norm=False")
    variant_moe = variant_model.get("MoE_config", {})
    expected_treatment = {
        "expert_contrastive_lam": 0.5,
        "expert_contrastive_temperature": 5.0,
        "expert_contrastive_mode": "output",
        "expert_contrastive_blocks": list(DEFAULT_BLOCK_INDICES),
    }
    for key, expected in expected_treatment.items():
        if variant_moe.get(key) != expected:
            raise ValueError(f"Formal tau=5 treatment requires {key}={expected!r}")
    return {
        "base": {
            "path": str(base_config),
            "sha256": BASE_CONFIG_SHA256,
            "training_payload_sha256": _training_config_payload_sha256(base),
        },
        "variant": {
            "path": str(variant_config),
            "sha256": VARIANT_CONFIG_SHA256,
            "training_payload_sha256": _training_config_payload_sha256(variant),
        },
    }


def _validate_training_provenance(provenance, config_stem, payload_sha256):
    if not isinstance(provenance, dict) or set(provenance) != {
        "version",
        "strict",
        "git",
        "config",
        "source_sha256",
        "environment",
    }:
        raise ValueError("Checkpoint strict training provenance is malformed")
    if provenance["version"] != 1 or provenance["strict"] is not True:
        raise ValueError("Checkpoint was not produced with strict provenance")
    git_contract = provenance["git"]
    if not isinstance(git_contract, dict):
        raise ValueError("Checkpoint Git provenance is missing")
    commit = git_contract.get("commit")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None
        or git_contract.get("origin_repa_commit") != commit
        or git_contract.get("status_clean") is not True
        or git_contract.get("origin_repa_divergence") != "0\t0"
    ):
        raise ValueError("Checkpoint was not launched from a clean pushed commit")
    config = provenance["config"]
    if (
        not isinstance(config, dict)
        or config.get("version") != 1
        or config.get("basename") != f"{config_stem}.yaml"
        or config.get("payload_sha256") != payload_sha256
    ):
        raise ValueError("Checkpoint config provenance differs from the formal config")
    source_sha256 = provenance["source_sha256"]
    if not isinstance(source_sha256, dict) or not source_sha256:
        raise ValueError("Checkpoint source provenance is empty")
    for relative, digest in source_sha256.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("Checkpoint source provenance is malformed")
    expected_model_source = (
        "models/models_ProMoE_TC.py"
        if config_stem == BASE_CONFIG_STEM
        else "models/models_ProMoE_TC_expert_contra.py"
    )
    required_sources = REQUIRED_COMMON_SOURCE_PATHS | {expected_model_source}
    if config_stem == BASE_CONFIG_STEM:
        required_sources = required_sources | {"models/phase_metric.py"}
    if set(source_sha256) != required_sources:
        raise ValueError(
            "Checkpoint provenance training source set is not canonical"
        )
    _verify_training_source_contract(commit, source_sha256, config_stem)
    environment = provenance["environment"]
    expected_visible_devices = (
        ["0", "1", "2", "3"]
        if config_stem == BASE_CONFIG_STEM
        else ["4", "5", "6", "7"]
    )
    if (
        not isinstance(environment, dict)
        or not isinstance(environment.get("cuda_devices"), dict)
        or len(environment["cuda_devices"]) != EXPECTED_WORLD_SIZE
        or environment.get("cuda_visible_devices") != expected_visible_devices
    ):
        raise ValueError("Checkpoint training environment is malformed")
    return provenance


def _checkpoint_trainer_contract(
    trainer_state,
    checkpoint_step,
    config_stem,
    payload_sha256,
):
    if not isinstance(trainer_state, dict):
        raise ValueError("Checkpoint trainer_state is missing")
    expected_scalars = {
        "version": TRAINER_STATE_VERSION,
        "augmentation_seed_version": AUGMENTATION_SEED_VERSION,
        "global_seed": EXPECTED_GLOBAL_SEED,
        "world_size": EXPECTED_WORLD_SIZE,
        "grad_mix": EXPECTED_GRAD_MIX,
        "next_step": checkpoint_step + 1,
        "data_batches_seen": (checkpoint_step + 1) * EXPECTED_GRAD_MIX,
    }
    for key, expected in expected_scalars.items():
        if trainer_state.get(key) != expected:
            raise ValueError(f"Checkpoint trainer_state requires {key}={expected!r}")
    run_id = trainer_state.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("Checkpoint run_id is missing or malformed")
    batches_per_epoch = trainer_state.get("batches_per_epoch")
    if (
        isinstance(batches_per_epoch, bool)
        or not isinstance(batches_per_epoch, int)
        or batches_per_epoch < 1
    ):
        raise ValueError("Checkpoint batches_per_epoch is invalid")
    expected_position = divmod(
        trainer_state["data_batches_seen"],
        batches_per_epoch,
    )
    if expected_position != (
        trainer_state.get("sampler_epoch"),
        trainer_state.get("sampler_batch_offset"),
    ):
        raise ValueError("Checkpoint sampler position is inconsistent")
    sampler = trainer_state.get("sampler_contract")
    per_rank_batch_size = (
        sampler.get("per_rank_batch_size")
        if isinstance(sampler, dict)
        else None
    )
    if (
        not isinstance(sampler, dict)
        or sampler.get("version") != 1
        or sampler.get("type") != "distributed"
        or sampler.get("global_seed") != EXPECTED_GLOBAL_SEED
        or isinstance(per_rank_batch_size, bool)
        or not isinstance(per_rank_batch_size, int)
        or per_rank_batch_size < 1
        or per_rank_batch_size * EXPECTED_WORLD_SIZE
        != EXPECTED_GLOBAL_BATCH_SIZE
        or sampler.get("drop_last") is not False
        or sampler.get("case1_prob") is not None
    ):
        raise ValueError("Checkpoint sampler contract is not the formal setup")
    dataset = sampler.get("dataset")
    if (
        not isinstance(dataset, dict)
        or dataset.get("version") != 1
        or not isinstance(dataset.get("num_samples"), int)
        or dataset["num_samples"] < 1
        or not isinstance(dataset.get("ordered_samples_sha256"), str)
        or _SHA256_PATTERN.fullmatch(dataset["ordered_samples_sha256"]) is None
    ):
        raise ValueError("Checkpoint dataset identity is malformed")
    rank_states = trainer_state.get("rank_states")
    if not isinstance(rank_states, list) or len(rank_states) != EXPECTED_WORLD_SIZE:
        raise ValueError("Checkpoint rank RNG states are incomplete")
    ranks = {
        state.get("rank")
        for state in rank_states
        if isinstance(state, dict) and isinstance(state.get("rng_state"), dict)
    }
    if ranks != set(range(EXPECTED_WORLD_SIZE)):
        raise ValueError("Checkpoint rank RNG states are malformed")
    provenance = _validate_training_provenance(
        trainer_state.get("training_provenance"),
        config_stem,
        payload_sha256,
    )
    return {
        "run_id": run_id,
        "trajectory": {
            "version": trainer_state["version"],
            "augmentation_seed_version": trainer_state[
                "augmentation_seed_version"
            ],
            "global_seed": trainer_state["global_seed"],
            "world_size": trainer_state["world_size"],
            "grad_mix": trainer_state["grad_mix"],
            "batches_per_epoch": batches_per_epoch,
            "sampler_contract": sampler,
        },
        "progress": {
            "next_step": trainer_state["next_step"],
            "data_batches_seen": trainer_state["data_batches_seen"],
            "sampler_epoch": trainer_state["sampler_epoch"],
            "sampler_batch_offset": trainer_state["sampler_batch_offset"],
        },
        "training_provenance": provenance,
        "training_provenance_sha256": _json_sha256(provenance),
    }


@contextmanager
def _checkpoint_safe_globals():
    """Allow only the metadata classes stored by ProMoE checkpoints."""

    safe_globals = getattr(
        getattr(torch, "serialization", None),
        "safe_globals",
        None,
    )
    if safe_globals is None:
        yield
        return

    try:
        from easydict import EasyDict
        from torch.torch_version import TorchVersion
    except ImportError as error:
        raise RuntimeError(
            "The restricted checkpoint loader cannot import its metadata types"
        ) from error

    with safe_globals([EasyDict, TorchVersion]):
        yield


def _load_checkpoint_payload(checkpoint_path):
    load_kwargs = {"map_location": "cpu"}
    try:
        supports_weights_only = (
            "weights_only" in inspect.signature(torch.load).parameters
        )
    except (TypeError, ValueError):
        supports_weights_only = True

    if supports_weights_only:
        load_kwargs["weights_only"] = True
        with _checkpoint_safe_globals():
            return torch.load(checkpoint_path, **load_kwargs)
    return torch.load(checkpoint_path, **load_kwargs)


def _load_checkpoint_model_and_contract(
    runtime_cfg,
    checkpoint_path,
    device,
    config_stem,
    payload_sha256,
):
    if config_stem == BASE_CONFIG_STEM:
        if runtime_cfg.model_name != BASE_MODEL_NAME:
            raise ValueError("Formal Base runtime model_name changed")
        model = BaseProMoE(**runtime_cfg.DiT_B_config)
    elif config_stem == VARIANT_CONFIG_STEM:
        if runtime_cfg.model_name != VARIANT_MODEL_NAME:
            raise ValueError("Formal tau=5 runtime model_name changed")
        model = ExpertContraProMoE(**runtime_cfg.DiT_B_config)
    else:
        raise ValueError(f"Unsupported formal config stem: {config_stem}")
    load_start = time.perf_counter()
    checkpoint = _load_checkpoint_payload(checkpoint_path)
    checkpoint_step = checkpoint.get("step")
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int):
        raise ValueError("Checkpoint must contain an integer step")
    if "ema_model_state_dict" not in checkpoint:
        raise KeyError("Formal gate requires ema_model_state_dict")
    trainer_contract = _checkpoint_trainer_contract(
        checkpoint.get("trainer_state"),
        checkpoint_step,
        config_stem,
        payload_sha256,
    )
    missing, unexpected = model.load_state_dict(
        checkpoint["ema_model_state_dict"],
        strict=False,
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={missing}, unexpected={unexpected}"
        )
    del checkpoint
    gc.collect()
    model = model.to(device).eval().requires_grad_(False)
    return (
        model,
        "ema_model_state_dict",
        checkpoint_step,
        time.perf_counter() - load_start,
        trainer_contract,
    )


def _validate_paired_trainer_contracts(base_contract, variant_contract):
    if base_contract["run_id"] == variant_contract["run_id"]:
        raise ValueError("Base and tau=5 runs must have different run_id values")
    for key in ("trajectory", "progress"):
        if base_contract[key] != variant_contract[key]:
            raise ValueError(f"Base and tau=5 checkpoint {key} contracts differ")
    base_provenance = base_contract["training_provenance"]
    variant_provenance = variant_contract["training_provenance"]
    base_sources = base_provenance["source_sha256"]
    variant_sources = variant_provenance["source_sha256"]
    for relative in sorted(PAIRED_IDENTICAL_SOURCE_PATHS):
        if base_sources[relative] != variant_sources[relative]:
            raise ValueError(
                f"Base and tau=5 common training source differs: {relative}"
            )
    if base_sources["train.py"] != BASE_TRAIN_SOURCE_SHA256:
        raise ValueError("Base train.py is not the locked formal source")
    if variant_sources["train.py"] != VARIANT_TRAIN_SOURCE_SHA256:
        raise ValueError("Tau=5 train.py is not the locked formal source")
    if _normalized_training_environment(base_provenance["environment"]) != (
        _normalized_training_environment(variant_provenance["environment"])
    ):
        raise ValueError(
            "Base and tau=5 normalized training environments differ"
        )
    return {
        "base": base_contract,
        "variant": variant_contract,
        "same_trajectory": True,
        "distinct_run_ids": True,
        "same_common_training_sources": True,
        "locked_train_entrypoints": True,
        "same_normalized_environment": True,
    }


def _normalized_training_environment(environment):
    scalar_fields = (
        "python",
        "python_executable",
        "torch",
        "numpy",
        "cuda_runtime",
    )
    devices = environment.get("cuda_devices")
    if not isinstance(devices, dict) or not devices:
        raise ValueError("Checkpoint CUDA environment is malformed")
    normalized_devices = []
    for device in devices.values():
        if not isinstance(device, dict):
            raise ValueError("Checkpoint CUDA device contract is malformed")
        normalized_devices.append({
            key: value for key, value in device.items() if key != "uuid"
        })
    normalized_devices.sort(key=_json_sha256)
    return {
        **{key: environment.get(key) for key in scalar_fields},
        "cuda_devices_without_uuid": normalized_devices,
    }


def _finite_float(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite")
    return value


def _upper_triangle(matrix):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Pairwise matrix must be square")
    if matrix.shape[0] < 2:
        raise ValueError("At least two experts are required")
    mask = torch.triu(
        torch.ones_like(matrix, dtype=torch.bool),
        diagonal=1,
    )
    return matrix[mask]


def compute_function_metrics(expert_outputs):
    """Measure expert functions on identical token inputs.

    ``expert_outputs`` is shaped ``[tokens, experts, hidden]``. Distances are
    computed after concatenating the same token set for every expert, so input
    assignment differences cannot masquerade as expert-function differences.
    """

    if expert_outputs.ndim != 3:
        raise ValueError("Expert outputs must be [tokens, experts, hidden]")
    num_tokens, num_experts, hidden_size = expert_outputs.shape
    if num_tokens < 1 or num_experts < 2 or hidden_size < 1:
        raise ValueError("Expert outputs have an invalid empty dimension")
    outputs = expert_outputs.float()
    if not bool(torch.isfinite(outputs).all().item()):
        raise RuntimeError("Expert outputs must be finite")

    functions = outputs.permute(1, 0, 2).reshape(num_experts, -1)
    normalized = F.normalize(functions, p=2, dim=1, eps=1e-12)
    cosine = normalized @ normalized.T
    normalized_l2 = torch.cdist(normalized, normalized, p=2)
    raw_l2_rms = torch.cdist(functions, functions, p=2) / math.sqrt(
        functions.shape[1]
    )

    output_rms = outputs.square().mean().sqrt()
    expert_rms = functions.square().mean(dim=1).sqrt()
    centered_outputs = outputs - outputs.mean(dim=1, keepdim=True)
    residual_rms = centered_outputs.square().mean().sqrt()

    centered_functions = functions - functions.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered_functions)
    energy = singular_values.square()
    energy_sum = energy.sum()
    if energy_sum.item() <= 1e-24:
        normalized_effective_rank = torch.zeros((), device=outputs.device)
    else:
        probabilities = energy / energy_sum
        entropy = -(probabilities * probabilities.clamp_min(1e-24).log()).sum()
        normalized_effective_rank = entropy.exp() / (num_experts - 1)

    metrics = {
        "output_rms": output_rms,
        "expert_rms_cv": expert_rms.std(unbiased=False)
        / expert_rms.mean().clamp_min(1e-12),
        "pairwise_l2_rms": _upper_triangle(raw_l2_rms).mean(),
        "normalized_pairwise_l2": _upper_triangle(normalized_l2).mean(),
        "pairwise_cosine": _upper_triangle(cosine).mean(),
        "relative_expert_residual_rms": residual_rms
        / output_rms.clamp_min(1e-12),
        "normalized_effective_rank": normalized_effective_rank,
    }
    return {
        name: _finite_float(value.item(), name)
        for name, value in metrics.items()
    }


def _gini(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Gini values must be a nonempty vector")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("Gini values must be finite and nonnegative")
    total = values.sum()
    if total == 0:
        return 0.0
    sorted_values = np.sort(values)
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    return float(
        (2 * np.dot(ranks, sorted_values) / total - values.size - 1)
        / values.size
    )


def compute_route_metrics(expert_ids, num_experts):
    expert_ids = expert_ids.reshape(-1).long()
    if expert_ids.numel() == 0:
        raise ValueError("Route metrics require at least one token")
    if expert_ids.min() < 0 or expert_ids.max() >= num_experts:
        raise ValueError("Route IDs are outside the routed expert set")
    counts = torch.bincount(expert_ids, minlength=num_experts).double()
    probabilities = counts / counts.sum()
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    normalized_entropy = entropy / math.log(num_experts)
    return {
        "token_counts": [int(value) for value in counts.tolist()],
        "normalized_entropy": float(normalized_entropy.item()),
        "maximum_share": float(probabilities.max().item()),
        "count_gini": _gini(counts.cpu().numpy()),
    }


def compute_native_pool_metrics(moe_layer, hidden_states, expert_ids):
    if hidden_states.ndim != 2:
        raise ValueError("Native pooled inputs must be [tokens, hidden]")
    expert_ids = expert_ids.reshape(-1).long()
    if hidden_states.shape[0] != expert_ids.numel():
        raise ValueError("Native pooled inputs and route IDs must align")

    pooled_outputs = []
    valid_ids = []
    token_counts = []
    with torch.inference_mode():
        for expert_id in range(moe_layer.num_routed_experts):
            selected = expert_ids == expert_id
            token_count = int(selected.sum().item())
            token_counts.append(token_count)
            if token_count == 0:
                continue
            output = moe_layer.experts[expert_id](hidden_states[selected]).float()
            pooled_outputs.append(output.mean(dim=0))
            valid_ids.append(expert_id)
    pooled = torch.stack(pooled_outputs)
    if len(pooled_outputs) == 1:
        return {
            "valid_expert_ids": valid_ids,
            "num_active_experts": 1,
            "token_counts": token_counts,
            "pooled_output_rms": float(pooled.square().mean().sqrt().item()),
            "pooled_pairwise_l2": 0.0,
            "pooled_pairwise_l2_rms": 0.0,
            "pooled_pairwise_cosine": 1.0,
            "pooled_repulsion_tau5": 0.0,
        }
    raw_l2 = torch.pdist(pooled, p=2)
    normalized = F.normalize(pooled, p=2, dim=1, eps=1e-12)
    cosine = _upper_triangle(normalized @ normalized.T)
    return {
        "valid_expert_ids": valid_ids,
        "num_active_experts": len(valid_ids),
        "token_counts": token_counts,
        "pooled_output_rms": float(pooled.square().mean().sqrt().item()),
        "pooled_pairwise_l2": float(raw_l2.mean().item()),
        "pooled_pairwise_l2_rms": float(
            (raw_l2 / math.sqrt(pooled.shape[1])).mean().item()
        ),
        "pooled_pairwise_cosine": float(cosine.mean().item()),
        "pooled_repulsion_tau5": float(torch.exp(-raw_l2 / 5.0).mean().item()),
    }


def _validate_model_pair(base_model, variant_model, block_indices):
    if len(base_model.blocks) != len(variant_model.blocks):
        raise ValueError("Base and variant must have the same block count")
    for block_index in block_indices:
        if not 0 <= block_index < len(base_model.blocks):
            raise ValueError(f"block {block_index} is outside the model")
        base_block = base_model.blocks[block_index]
        variant_block = variant_model.blocks[block_index]
        if not base_block.use_moe or not variant_block.use_moe:
            raise ValueError(f"block {block_index} must be MoE in both models")
        base_moe = base_block.mlp
        variant_moe = variant_block.mlp
        contract = (
            base_moe.num_routed_experts,
            base_moe.hidden_size,
            base_moe.top_k,
            base_moe.router_weight_mode,
        )
        variant_contract = (
            variant_moe.num_routed_experts,
            variant_moe.hidden_size,
            variant_moe.top_k,
            variant_moe.router_weight_mode,
        )
        if contract != variant_contract:
            raise ValueError(
                f"block {block_index} MoE contracts differ: "
                f"{contract} != {variant_contract}"
            )
        if contract[2:] != (1, "identity"):
            raise ValueError("The diversity gate requires identity top-1 routing")


def _capture_forward(model, captures, noised_latent, timestep, label, target):
    for capture in captures.values():
        capture.start()
    try:
        with torch.inference_mode():
            output = model(noised_latent, timestep, context=label)
    finally:
        for capture in captures.values():
            capture.stop()
    prediction = _extract_prediction(output, target.shape[1])
    mse = _per_sample_mse(prediction, target)[0]
    hidden = {}
    for block_index, capture in captures.items():
        if capture.hidden_states is None or capture.labels is None:
            raise RuntimeError(f"block {block_index} capture is incomplete")
        hidden[block_index] = capture.hidden_states
    return hidden, float(mse.item())


def _anchor_indices(num_tokens, count, seed, block_index, device):
    count = min(int(count), int(num_tokens))
    if count < 2:
        raise ValueError("At least two anchor tokens are required")
    generator = np.random.default_rng(int(seed) + 1009 * int(block_index))
    indices = np.sort(generator.choice(num_tokens, size=count, replace=False))
    return torch.as_tensor(indices, device=device, dtype=torch.long)


def _function_pair(base_experts, variant_experts, hidden_states):
    with torch.inference_mode():
        base_outputs = _evaluate_all_experts(base_experts, hidden_states)
        variant_outputs = _evaluate_all_experts(variant_experts, hidden_states)
    return {
        "base": compute_function_metrics(base_outputs),
        "variant": compute_function_metrics(variant_outputs),
    }


def _probe_case(
    base_model,
    variant_model,
    case,
    latent_root,
    sigmas,
    block_indices,
    num_anchor_tokens,
    num_train_timesteps,
    device,
):
    latent_path = Path(latent_root) / case["latent"]
    clean_latent = _load_latent(
        latent_path,
        "latent",
        int(case["seed"]),
        device,
    )
    torch.manual_seed(int(case["seed"]) + 1)
    noise = torch.randn_like(clean_latent)
    label = torch.tensor([int(case["label"])], device=device, dtype=torch.long)
    base_captures = {
        index: RouteInputCapture(base_model.blocks[index].mlp)
        for index in block_indices
    }
    variant_captures = {
        index: RouteInputCapture(variant_model.blocks[index].mlp)
        for index in block_indices
    }
    cells = []
    try:
        for sigma in sigmas:
            sigma_tensor = torch.tensor(
                float(sigma),
                device=device,
                dtype=clean_latent.dtype,
            )
            timestep = torch.full(
                (1,),
                float(sigma) * num_train_timesteps,
                device=device,
                dtype=clean_latent.dtype,
            )
            noised_latent = (
                (1.0 - sigma_tensor) * clean_latent + sigma_tensor * noise
            )
            target = (noise - clean_latent).squeeze(2)
            base_hidden, base_mse = _capture_forward(
                base_model,
                base_captures,
                noised_latent,
                timestep,
                label,
                target,
            )
            variant_hidden, variant_mse = _capture_forward(
                variant_model,
                variant_captures,
                noised_latent,
                timestep,
                label,
                target,
            )

            for block_index in block_indices:
                base_moe = base_model.blocks[block_index].mlp
                variant_moe = variant_model.blocks[block_index].mlp
                base_input = base_hidden[block_index]
                variant_input = variant_hidden[block_index]
                if base_input.shape != variant_input.shape:
                    raise RuntimeError("Paired block inputs must have the same shape")
                anchors = _anchor_indices(
                    base_input.shape[1],
                    num_anchor_tokens,
                    int(case["seed"]),
                    block_index,
                    device,
                )
                _, base_indices, _ = _compute_router(
                    base_moe,
                    base_input,
                    label,
                    timestep,
                )
                _, variant_indices, _ = _compute_router(
                    variant_moe,
                    variant_input,
                    label,
                    timestep,
                )
                base_ids = base_indices[0, :, 0]
                variant_ids = variant_indices[0, :, 0]
                base_experts = base_moe.experts[:base_moe.num_routed_experts]
                variant_experts = (
                    variant_moe.experts[:variant_moe.num_routed_experts]
                )
                cells.append({
                    "case_id": case["id"],
                    "label": int(case["label"]),
                    "sigma": float(sigma),
                    "block_index": int(block_index),
                    "num_anchor_tokens": int(anchors.numel()),
                    "denoising_mse": {
                        "base": base_mse,
                        "variant": variant_mse,
                    },
                    "base_hidden_functions": _function_pair(
                        base_experts,
                        variant_experts,
                        base_input[0, anchors],
                    ),
                    "variant_hidden_functions": _function_pair(
                        base_experts,
                        variant_experts,
                        variant_input[0, anchors],
                    ),
                    "native_pool": {
                        "base": compute_native_pool_metrics(
                            base_moe,
                            base_input[0],
                            base_ids,
                        ),
                        "variant": compute_native_pool_metrics(
                            variant_moe,
                            variant_input[0],
                            variant_ids,
                        ),
                    },
                    "route": {
                        "base": compute_route_metrics(
                            base_ids,
                            base_moe.num_routed_experts,
                        ),
                        "variant": compute_route_metrics(
                            variant_ids,
                            variant_moe.num_routed_experts,
                        ),
                    },
                })
    finally:
        for capture in (*base_captures.values(), *variant_captures.values()):
            capture.close()
    return {
        "case": dict(case),
        "cells": cells,
    }


def _worker(payload):
    torch.set_grad_enabled(False)
    thread_config = _configure_torch_threads(payload["num_threads"])
    device = torch.device(payload["device"])
    base_cfg = load_runtime_cfg(Path(payload["base_config"]))
    variant_cfg = load_runtime_cfg(Path(payload["variant_config"]))
    if payload["formal"]:
        (
            base_model,
            base_state,
            base_step,
            base_load_seconds,
            base_trainer_contract,
        ) = _load_checkpoint_model_and_contract(
            base_cfg,
            payload["base_weights_checkpoint"],
            device,
            payload["base_config_stem"],
            payload["base_training_payload_sha256"],
        )
        (
            variant_model,
            variant_state,
            variant_step,
            variant_load_seconds,
            variant_trainer_contract,
        ) = _load_checkpoint_model_and_contract(
            variant_cfg,
            payload["variant_weights_checkpoint"],
            device,
            payload["variant_config_stem"],
            payload["variant_training_payload_sha256"],
        )
        paired_trainer_contract = _validate_paired_trainer_contracts(
            base_trainer_contract,
            variant_trainer_contract,
        )
    else:
        base_model, base_state, base_step, base_load_seconds = (
            _load_checkpoint_model(
                base_cfg,
                payload["base_weights_checkpoint"],
                device,
            )
        )
        variant_model, variant_state, variant_step, variant_load_seconds = (
            _load_checkpoint_model(
                variant_cfg,
                payload["variant_weights_checkpoint"],
                device,
            )
        )
        paired_trainer_contract = None
    if base_step != payload["checkpoint_step"]:
        raise ValueError("Base weights do not match the canonical checkpoint step")
    if variant_step != payload["checkpoint_step"]:
        raise ValueError("Variant weights do not match the canonical checkpoint step")
    if base_cfg.num_train_timesteps != variant_cfg.num_train_timesteps:
        raise ValueError("Base and variant diffusion timestep counts differ")
    _validate_model_pair(base_model, variant_model, payload["block_indices"])

    started = time.perf_counter()
    results = [
        _probe_case(
            base_model=base_model,
            variant_model=variant_model,
            case=case,
            latent_root=payload["latent_root"],
            sigmas=payload["sigmas"],
            block_indices=payload["block_indices"],
            num_anchor_tokens=payload["num_anchor_tokens"],
            num_train_timesteps=base_cfg.num_train_timesteps,
            device=device,
        )
        for case in payload["cases"]
    ]
    return {
        "device": str(device),
        "thread_config": thread_config,
        "base_state": base_state,
        "variant_state": variant_state,
        "base_load_seconds": base_load_seconds,
        "variant_load_seconds": variant_load_seconds,
        "paired_trainer_contract": paired_trainer_contract,
        "probe_seconds": time.perf_counter() - started,
        "cases": results,
    }


def _case_metric_rows(case_records):
    rows = []
    for case_record in case_records:
        case_id = case_record["case"]["id"]
        for cell in case_record["cells"]:
            common = {}
            for metric in FUNCTION_METRICS:
                base_on_base = cell["base_hidden_functions"]["base"][metric]
                variant_on_base = cell["base_hidden_functions"]["variant"][metric]
                base_on_variant = cell["variant_hidden_functions"]["base"][metric]
                variant_on_variant = cell["variant_hidden_functions"]["variant"][metric]
                common[f"common_{metric}"] = (
                    0.5 * (base_on_base + base_on_variant),
                    0.5 * (variant_on_base + variant_on_variant),
                )
            pairs = {
                **common,
                "pooled_pairwise_l2": (
                    cell["native_pool"]["base"]["pooled_pairwise_l2"],
                    cell["native_pool"]["variant"]["pooled_pairwise_l2"],
                ),
                "pooled_pairwise_l2_rms": (
                    cell["native_pool"]["base"]["pooled_pairwise_l2_rms"],
                    cell["native_pool"]["variant"]["pooled_pairwise_l2_rms"],
                ),
                "pooled_repulsion_tau5": (
                    cell["native_pool"]["base"]["pooled_repulsion_tau5"],
                    cell["native_pool"]["variant"]["pooled_repulsion_tau5"],
                ),
                "active_experts": (
                    cell["native_pool"]["base"]["num_active_experts"],
                    cell["native_pool"]["variant"]["num_active_experts"],
                ),
                "route_normalized_entropy": (
                    cell["route"]["base"]["normalized_entropy"],
                    cell["route"]["variant"]["normalized_entropy"],
                ),
                "route_maximum_share": (
                    cell["route"]["base"]["maximum_share"],
                    cell["route"]["variant"]["maximum_share"],
                ),
                "route_count_gini": (
                    cell["route"]["base"]["count_gini"],
                    cell["route"]["variant"]["count_gini"],
                ),
                "denoising_mse": (
                    cell["denoising_mse"]["base"],
                    cell["denoising_mse"]["variant"],
                ),
            }
            rows.append({
                "case_id": case_id,
                "sigma": cell["sigma"],
                "block_index": cell["block_index"],
                "pairs": pairs,
            })
    return rows


def _bootstrap_ci(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Bootstrap requires at least two case values")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        values.size,
        size=(int(resamples), values.size),
    )
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def compare_case_records(
    case_records,
    bootstrap_resamples=20000,
    bootstrap_seed=0,
    formal=True,
):
    rows = _case_metric_rows(case_records)
    if not rows:
        raise ValueError("At least one paired case record is required")
    case_ids = sorted({row["case_id"] for row in rows})
    metric_names = sorted(rows[0]["pairs"])
    if any(set(row["pairs"]) != set(metric_names) for row in rows):
        raise ValueError("Paired cells do not share the same metric set")

    metrics = {}
    for metric_index, metric in enumerate(metric_names):
        per_case = []
        for case_id in case_ids:
            case_pairs = np.asarray([
                row["pairs"][metric]
                for row in rows
                if row["case_id"] == case_id
            ], dtype=np.float64)
            if case_pairs.ndim != 2 or case_pairs.shape[1] != 2:
                raise ValueError(f"Metric {metric} has malformed pairs")
            base_mean, variant_mean = case_pairs.mean(axis=0)
            per_case.append((base_mean, variant_mean))
        per_case = np.asarray(per_case, dtype=np.float64)
        deltas = per_case[:, 1] - per_case[:, 0]
        base_mean = float(per_case[:, 0].mean())
        variant_mean = float(per_case[:, 1].mean())
        relative_delta = float(
            (variant_mean - base_mean) / max(abs(base_mean), 1e-12)
        )
        metrics[metric] = {
            "base_mean": base_mean,
            "variant_mean": variant_mean,
            "mean_delta": float(deltas.mean()),
            "relative_delta": relative_delta,
            "positive_cases": int((deltas > 0).sum()),
            "num_cases": len(case_ids),
            "case_delta_ci95": _bootstrap_ci(
                deltas,
                bootstrap_resamples,
                bootstrap_seed + metric_index,
            ),
            "per_case": {
                case_id: {
                    "base": float(pair[0]),
                    "variant": float(pair[1]),
                    "delta": float(pair[1] - pair[0]),
                }
                for case_id, pair in zip(case_ids, per_case)
            },
        }

    pooled = metrics["pooled_pairwise_l2"]
    pooled_repulsion = metrics["pooled_repulsion_tau5"]
    common_distance = metrics["common_normalized_pairwise_l2"]
    effective_rank = metrics["common_normalized_effective_rank"]
    output_rms = metrics["common_output_rms"]
    route_entropy = metrics["route_normalized_entropy"]
    route_share = metrics["route_maximum_share"]
    denoising_mse = metrics["denoising_mse"]
    minimum_variant_active_experts = min(
        int(row["pairs"]["active_experts"][1]) for row in rows
    )
    collapsed_variant_cells = [
        {
            "case_id": row["case_id"],
            "sigma": row["sigma"],
            "block_index": row["block_index"],
            "num_active_experts": int(row["pairs"]["active_experts"][1]),
        }
        for row in rows
        if row["pairs"]["active_experts"][1]
        < GATE_REQUIREMENTS["minimum_active_experts_per_cell"]
    ]
    checks = {
        "active_expert_safety": {
            "passed": not collapsed_variant_cells,
            "observed": {
                "minimum_variant_active_experts": (
                    minimum_variant_active_experts
                ),
                "collapsed_variant_cells": collapsed_variant_cells,
            },
            "required": (
                "every tau=5 cell has active experts >="
                f"{GATE_REQUIREMENTS['minimum_active_experts_per_cell']}"
            ),
        },
        "pooled_distance_effect": {
            "passed": bool(
                pooled["relative_delta"]
                >= GATE_REQUIREMENTS["pooled_pairwise_l2_relative_gain"]
                and pooled["case_delta_ci95"][0] > 0
            ),
            "observed": pooled["relative_delta"],
            "required": (
                f">={GATE_REQUIREMENTS['pooled_pairwise_l2_relative_gain']} "
                "and paired CI lower >0"
            ),
        },
        "pooled_repulsion_objective": {
            "passed": bool(
                pooled_repulsion["mean_delta"]
                <= GATE_REQUIREMENTS[
                    "maximum_pooled_repulsion_tau5_mean_delta"
                ]
                and pooled_repulsion["case_delta_ci95"][1] < 0
            ),
            "observed": {
                "mean_delta": pooled_repulsion["mean_delta"],
                "paired_ci95": pooled_repulsion["case_delta_ci95"],
            },
            "required": (
                "mean delta <="
                f"{GATE_REQUIREMENTS['maximum_pooled_repulsion_tau5_mean_delta']} "
                "and paired CI upper <0"
            ),
        },
        "same_input_scale_free_effect": {
            "passed": bool(
                common_distance["relative_delta"]
                >= GATE_REQUIREMENTS[
                    "common_normalized_pairwise_l2_relative_gain"
                ]
                and common_distance["case_delta_ci95"][0] > 0
            ),
            "observed": common_distance["relative_delta"],
            "required": (
                ">="
                f"{GATE_REQUIREMENTS['common_normalized_pairwise_l2_relative_gain']} "
                "and paired CI lower >0"
            ),
        },
        "effective_rank_effect": {
            "passed": bool(
                effective_rank["mean_delta"] > 0
                and effective_rank["positive_cases"]
                >= GATE_REQUIREMENTS["minimum_positive_effective_rank_cases"]
            ),
            "observed": {
                "mean_delta": effective_rank["mean_delta"],
                "positive_cases": effective_rank["positive_cases"],
            },
            "required": (
                "mean delta >0 and positive cases >="
                f"{GATE_REQUIREMENTS['minimum_positive_effective_rank_cases']}"
            ),
        },
        "output_scale_safety": {
            "passed": bool(
                output_rms["relative_delta"]
                <= GATE_REQUIREMENTS["maximum_common_output_rms_relative_gain"]
            ),
            "observed": output_rms["relative_delta"],
            "required": (
                "<="
                f"{GATE_REQUIREMENTS['maximum_common_output_rms_relative_gain']}"
            ),
        },
        "route_entropy_safety": {
            "passed": bool(
                route_entropy["mean_delta"]
                >= GATE_REQUIREMENTS["minimum_route_entropy_delta"]
            ),
            "observed": route_entropy["mean_delta"],
            "required": f">={GATE_REQUIREMENTS['minimum_route_entropy_delta']}",
        },
        "route_share_safety": {
            "passed": bool(
                route_share["mean_delta"]
                <= GATE_REQUIREMENTS["maximum_route_share_delta"]
            ),
            "observed": route_share["mean_delta"],
            "required": f"<={GATE_REQUIREMENTS['maximum_route_share_delta']}",
        },
        "denoising_mse_safety": {
            "passed": bool(
                denoising_mse["relative_delta"]
                <= GATE_REQUIREMENTS["maximum_denoising_mse_relative_gain"]
            ),
            "observed": denoising_mse["relative_delta"],
            "required": (
                "<="
                f"{GATE_REQUIREMENTS['maximum_denoising_mse_relative_gain']}"
            ),
        },
    }
    formal_passed = bool(all(check["passed"] for check in checks.values()))
    return {
        "num_cases": len(case_ids),
        "num_cells": len(rows),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "requirements": dict(GATE_REQUIREMENTS),
        "checks": checks,
        "decision_mode": "formal" if formal else "exploratory",
        "passed": formal_passed if formal else None,
        "metrics": metrics,
    }


def _load_cases(manifest_path, latent_root, split):
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("r") as handle:
        manifest = json.load(handle)
    cases = [
        case for case in manifest.get("cases", [])
        if split is None or case.get("split") == split
    ]
    if len(cases) < 2:
        raise ValueError(f"Manifest split {split!r} has fewer than two cases")
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Manifest case IDs must be unique and nonempty")
    for case in cases:
        for key in ("label", "latent", "seed"):
            if key not in case:
                raise KeyError(f"Manifest case {case['id']} is missing {key}")
        path = Path(latent_root) / case["latent"]
        if not path.is_file():
            raise FileNotFoundError(f"Manifest latent does not exist: {path}")
    return manifest_path, manifest, cases


def _validate_formal_protocol(
    paths,
    checkpoint_step,
    base_config,
    variant_config,
    latent_root,
    manifest_path,
    manifest_split,
    cases,
    sigmas,
    block_indices,
    devices,
    num_anchor_tokens,
    num_threads,
    bootstrap_resamples,
    bootstrap_seed,
):
    expected_checkpoints = {
        "base_checkpoint": (
            PROJECT_ROOT
            / "outputs"
            / BASE_MODEL_NAME
            / BASE_CONFIG_STEM
            / "checkpoints"
            / f"ckpt_step_{FORMAL_CHECKPOINT_STEP}.pth"
        ).resolve(),
        "variant_checkpoint": (
            PROJECT_ROOT
            / "outputs"
            / VARIANT_MODEL_NAME
            / VARIANT_CONFIG_STEM
            / "checkpoints"
            / f"ckpt_step_{FORMAL_CHECKPOINT_STEP}.pth"
        ).resolve(),
    }
    for name, expected in expected_checkpoints.items():
        if paths[name] != expected:
            raise ValueError(f"Formal {name} must be {expected}")
    if checkpoint_step != FORMAL_CHECKPOINT_STEP:
        raise ValueError(
            f"Formal gate requires checkpoint step {FORMAL_CHECKPOINT_STEP}"
        )

    expected_manifest = DEFAULT_MANIFEST.resolve()
    if manifest_path != expected_manifest:
        raise ValueError(f"Formal manifest must be {expected_manifest}")
    if sha256_file(manifest_path) != CANONICAL_MANIFEST_SHA256:
        raise ValueError("Formal manifest SHA256 changed")
    if manifest_split != "discovery":
        raise ValueError("Formal gate requires the discovery manifest split")
    if len(cases) != FORMAL_NUM_CASES or any(
        case.get("split") != "discovery" for case in cases
    ):
        raise ValueError(
            f"Formal gate requires exactly {FORMAL_NUM_CASES} discovery cases"
        )
    locked_values = {
        "sigmas": (tuple(sigmas), DEFAULT_SIGMAS),
        "block_indices": (tuple(block_indices), DEFAULT_BLOCK_INDICES),
        "devices": (tuple(devices), FORMAL_DEVICES),
        "num_anchor_tokens": (num_anchor_tokens, FORMAL_NUM_ANCHOR_TOKENS),
        "num_threads": (num_threads, FORMAL_NUM_THREADS),
        "bootstrap_resamples": (
            bootstrap_resamples,
            FORMAL_BOOTSTRAP_RESAMPLES,
        ),
        "bootstrap_seed": (bootstrap_seed, FORMAL_BOOTSTRAP_SEED),
    }
    for name, (observed, expected) in locked_values.items():
        if observed != expected:
            raise ValueError(f"Formal gate requires {name}={expected!r}")
    configs = _validate_formal_configs(
        base_config,
        variant_config,
        latent_root,
    )
    return {
        "checkpoint_step": FORMAL_CHECKPOINT_STEP,
        "manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "manifest_split": "discovery",
        "num_cases": FORMAL_NUM_CASES,
        "sigmas": list(DEFAULT_SIGMAS),
        "block_indices": list(DEFAULT_BLOCK_INDICES),
        "devices": list(FORMAL_DEVICES),
        "num_anchor_tokens": FORMAL_NUM_ANCHOR_TOKENS,
        "num_threads": FORMAL_NUM_THREADS,
        "bootstrap_resamples": FORMAL_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
        "configs": configs,
    }


def _validate_formal_case_records(case_records, expected_cases):
    expected_by_id = {case["id"]: case for case in expected_cases}
    observed_ids = [record.get("case", {}).get("id") for record in case_records]
    if (
        len(case_records) != FORMAL_NUM_CASES
        or len(observed_ids) != len(set(observed_ids))
        or set(observed_ids) != set(expected_by_id)
    ):
        raise RuntimeError("Formal worker results changed the locked case set")
    expected_cells = {
        (float(sigma), int(block_index))
        for sigma in DEFAULT_SIGMAS
        for block_index in DEFAULT_BLOCK_INDICES
    }
    for record in case_records:
        case_id = record["case"]["id"]
        if record["case"] != expected_by_id[case_id]:
            raise RuntimeError(f"Formal case metadata changed for {case_id}")
        cells = record.get("cells")
        observed_cells = {
            (float(cell["sigma"]), int(cell["block_index"]))
            for cell in cells
        } if isinstance(cells, list) else set()
        if len(cells or ()) != len(expected_cells) or observed_cells != expected_cells:
            raise RuntimeError(f"Formal case {case_id} has an incomplete cell grid")
        if any(
            cell.get("num_anchor_tokens") != FORMAL_NUM_ANCHOR_TOKENS
            for cell in cells
        ):
            raise RuntimeError(f"Formal case {case_id} changed its token count")


def _validate_inputs(
    base_checkpoint,
    variant_checkpoint,
    base_weights_checkpoint,
    variant_weights_checkpoint,
    sigmas,
    block_indices,
    devices,
    num_anchor_tokens,
    num_threads,
    bootstrap_resamples,
    bootstrap_seed,
):
    paths = {
        "base_checkpoint": Path(base_checkpoint).resolve(),
        "variant_checkpoint": Path(variant_checkpoint).resolve(),
        "base_weights_checkpoint": Path(
            base_weights_checkpoint or base_checkpoint
        ).resolve(),
        "variant_weights_checkpoint": Path(
            variant_weights_checkpoint or variant_checkpoint
        ).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    base_step = parse_checkpoint_step(paths["base_checkpoint"])
    variant_step = parse_checkpoint_step(paths["variant_checkpoint"])
    if base_step != variant_step:
        raise ValueError("Base and variant checkpoint steps must match")
    sigmas = tuple(float(value) for value in sigmas)
    if (
        not sigmas
        or len(sigmas) != len(set(sigmas))
        or any(not 0 < value < 1 for value in sigmas)
    ):
        raise ValueError("Sigmas must be unique and strictly between zero and one")
    block_indices = tuple(int(value) for value in block_indices)
    if not block_indices or len(block_indices) != len(set(block_indices)):
        raise ValueError("Block indices must be unique and nonempty")
    devices = tuple(str(value) for value in devices)
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("Devices must be unique and nonempty")
    if (
        isinstance(num_anchor_tokens, bool)
        or not isinstance(num_anchor_tokens, int)
        or num_anchor_tokens < 2
    ):
        raise ValueError("num_anchor_tokens must be at least two")
    if (
        isinstance(num_threads, bool)
        or not isinstance(num_threads, int)
        or num_threads < 1
    ):
        raise ValueError("num_threads must be a positive integer")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 1000
    ):
        raise ValueError("bootstrap_resamples must be at least 1000")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a non-negative integer")
    checkpoint_files = {
        "base": _checkpoint_file_metadata(
            paths["base_checkpoint"],
            paths["base_weights_checkpoint"],
            "Base",
        ),
        "variant": _checkpoint_file_metadata(
            paths["variant_checkpoint"],
            paths["variant_weights_checkpoint"],
            "Variant",
        ),
    }
    return paths, checkpoint_files, base_step, sigmas, block_indices, devices


def run_expert_output_diversity_gate(
    base_checkpoint,
    variant_checkpoint,
    latent_root,
    base_weights_checkpoint=None,
    variant_weights_checkpoint=None,
    manifest_path=DEFAULT_MANIFEST,
    manifest_split="discovery",
    sigmas=DEFAULT_SIGMAS,
    block_indices=DEFAULT_BLOCK_INDICES,
    num_anchor_tokens=32,
    devices=("cpu",),
    num_threads=4,
    bootstrap_resamples=20000,
    bootstrap_seed=0,
    formal=True,
):
    (
        paths,
        checkpoint_files,
        checkpoint_step,
        sigmas,
        block_indices,
        devices,
    ) = _validate_inputs(
        base_checkpoint,
        variant_checkpoint,
        base_weights_checkpoint,
        variant_weights_checkpoint,
        sigmas,
        block_indices,
        devices,
        num_anchor_tokens,
        num_threads,
        bootstrap_resamples,
        bootstrap_seed,
    )
    latent_root = Path(latent_root).resolve()
    if not latent_root.is_dir():
        raise NotADirectoryError(f"Latent root does not exist: {latent_root}")
    manifest_path, manifest, cases = _load_cases(
        manifest_path,
        latent_root,
        manifest_split,
    )
    base_config = resolve_config_from_checkpoint(paths["base_checkpoint"])
    variant_config = resolve_config_from_checkpoint(paths["variant_checkpoint"])
    if formal:
        formal_protocol = _validate_formal_protocol(
            paths=paths,
            checkpoint_step=checkpoint_step,
            base_config=base_config,
            variant_config=variant_config,
            latent_root=latent_root,
            manifest_path=manifest_path,
            manifest_split=manifest_split,
            cases=cases,
            sigmas=sigmas,
            block_indices=block_indices,
            devices=devices,
            num_anchor_tokens=num_anchor_tokens,
            num_threads=num_threads,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        config_contracts = formal_protocol["configs"]
    else:
        formal_protocol = None
        config_contracts = {
            "base": {
                "path": str(base_config),
                "sha256": sha256_file(base_config),
                "training_payload_sha256": _training_config_payload_sha256(
                    _load_yaml(base_config)
                ),
            },
            "variant": {
                "path": str(variant_config),
                "sha256": sha256_file(variant_config),
                "training_payload_sha256": _training_config_payload_sha256(
                    _load_yaml(variant_config)
                ),
            },
        }

    assignments = [[] for _ in devices]
    for index, case in enumerate(cases):
        assignments[index % len(devices)].append(case)
    payloads = []
    for device, assigned_cases in zip(devices, assignments):
        if not assigned_cases:
            continue
        payloads.append({
            "device": device,
            "num_threads": int(num_threads),
            "base_config": str(base_config),
            "variant_config": str(variant_config),
            "base_config_stem": base_config.stem,
            "variant_config_stem": variant_config.stem,
            "base_training_payload_sha256": config_contracts["base"][
                "training_payload_sha256"
            ],
            "variant_training_payload_sha256": config_contracts["variant"][
                "training_payload_sha256"
            ],
            "base_weights_checkpoint": str(paths["base_weights_checkpoint"]),
            "variant_weights_checkpoint": str(paths["variant_weights_checkpoint"]),
            "checkpoint_step": checkpoint_step,
            "formal": bool(formal),
            "latent_root": str(latent_root),
            "sigmas": sigmas,
            "block_indices": block_indices,
            "num_anchor_tokens": int(num_anchor_tokens),
            "cases": assigned_cases,
        })

    started = time.perf_counter()
    if len(payloads) == 1:
        worker_results = [_worker(payloads[0])]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(payloads),
            mp_context=context,
        ) as executor:
            worker_results = list(executor.map(_worker, payloads))
    case_records = sorted(
        [case for worker in worker_results for case in worker["cases"]],
        key=lambda value: value["case"]["id"],
    )
    if len(case_records) != len(cases):
        raise RuntimeError("Worker results do not cover every manifest case")
    paired_trainer_contract = None
    if formal:
        _validate_formal_case_records(case_records, cases)
        paired_trainer_contract = worker_results[0]["paired_trainer_contract"]
        if paired_trainer_contract is None or any(
            worker["paired_trainer_contract"] != paired_trainer_contract
            for worker in worker_results[1:]
        ):
            raise RuntimeError(
                "Workers did not load one identical paired trainer contract"
            )
    comparison = compare_case_records(
        case_records,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        formal=formal,
    )
    return {
        "expert_output_diversity_probe_version": PROBE_VERSION,
        "decision_mode": "formal" if formal else "exploratory",
        "formal_protocol": formal_protocol,
        "checkpoint_step": checkpoint_step,
        "base_checkpoint": str(paths["base_checkpoint"]),
        "variant_checkpoint": str(paths["variant_checkpoint"]),
        "base_weights_checkpoint": str(paths["base_weights_checkpoint"]),
        "variant_weights_checkpoint": str(paths["variant_weights_checkpoint"]),
        "checkpoint_files": checkpoint_files,
        "base_config": config_contracts["base"],
        "variant_config": config_contracts["variant"],
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "name": manifest.get("name"),
            "version": manifest.get("version"),
            "split": manifest_split,
        },
        "latent_root": str(latent_root),
        "sigmas": list(sigmas),
        "block_indices": list(block_indices),
        "num_anchor_tokens": int(num_anchor_tokens),
        "devices": list(devices),
        "num_threads": int(num_threads),
        "paired_trainer_contract": paired_trainer_contract,
        "workers": [
            {
                key: value
                for key, value in worker.items()
                if key not in {"cases", "paired_trainer_contract"}
            }
            for worker in worker_results
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "comparison": comparison,
        "cases": case_records,
    }


def release_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
