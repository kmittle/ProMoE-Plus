"""Build and verify the immutable post-implementation experiment protocol."""

from __future__ import annotations

import copy
import json
import os
import platform
import subprocess
from pathlib import Path

import diffusers
import numpy as np
import timm
import torch
import yaml

from analyses.run_learning_credit_balance_cross_checkpoint import (
    _latent_dataset_identity,
)
from analyses.t_SNE.checkpoint_utils import load_runtime_cfg

from .controller import BRANCHES
from .heldout import canonical_json_sha256
from .protocol_lock import V3_SHA256, V4_SHA256, load_effective_protocol
from .serialization import atomic_write_json, sha256_file


PROTOCOL_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(
    "/home/dev/promoe-probes/credit-normalized-expert-gradient-ab-v4"
)
DEFAULT_PROTOCOL_PATH = DEFAULT_OUTPUT_ROOT / "protocol.json"
FROZEN_CHECKPOINT = Path("/home/dev/promoe-probes/base-seed0-ckpt_step_301000.pth")
LATENT_ROOT = Path("/home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz")
HELDOUT_MANIFEST = DEFAULT_OUTPUT_ROOT / "heldout" / "manifest.json"
RUN_OUTPUT_ROOT = Path("/home/dev/promoe-runs/credit-normalized-expert-gradient-ab-v4")
BASE_CONFIG = PROJECT_ROOT / "configs/004_ProMoE_B_seed0_control.yaml"
PARENT_PROTOCOL = Path(
    "/home/dev/promoe-probes/credit-balance-gate-base200k-v1/protocol.json"
)
CROSS_CHECKPOINT_ROOT = Path(
    "/home/dev/promoe-probes/credit-balance-lossfree-s0-200k-v2"
)
V3_PATH = Path(
    "/home/dev/promoe-probes/credit-normalized-expert-gradient-ab-v3-preregister.json"
)
V4_PATH = Path(
    "/home/dev/promoe-probes/credit-normalized-expert-gradient-ab-v4-preregister.json"
)
SEALED_GPU_IDS = (4, 5, 6, 7)
SEALED_GPU_MAPPING_ENV = "PROMOE_SEALED_PHYSICAL_GPU_IDS"

BRANCH_DEFINITIONS = (
    {
        "name": "measure_only_control",
        "config": "configs/004_ProMoE_B_credit_rate_measure_only_s0_301k_20k.yaml",
        "launcher": "scripts/credit_redistribution/run_B_credit_rate_measure_only_s0_301k_20k.sh",
    },
    {
        "name": "rotating_permuted_scale_control",
        "config": "configs/004_ProMoE_B_credit_rate_permuted_s0_301k_20k.yaml",
        "launcher": "scripts/credit_redistribution/run_B_credit_rate_permuted_s0_301k_20k.sh",
    },
    {
        "name": "matched_credit_rate_redistribution",
        "config": "configs/004_ProMoE_B_credit_rate_matched_s0_301k_20k.yaml",
        "launcher": "scripts/credit_redistribution/run_B_credit_rate_matched_s0_301k_20k.sh",
    },
)

SOURCE_PATHS = (
    "config.py",
    "train.py",
    "utils.py",
    "models/models_ProMoE_TC.py",
    "models/phase_metric.py",
    "models/modules.py",
    "analyses/denoising_regret/probe.py",
    "analyses/run_learning_credit_balance_cross_checkpoint.py",
    "analyses/run_credit_redistribution_gate.py",
    "analyses/t_SNE/checkpoint_utils.py",
    "credit_redistribution/__init__.py",
    "credit_redistribution/benchmark.py",
    "credit_redistribution/controller.py",
    "credit_redistribution/evaluator.py",
    "credit_redistribution/heldout.py",
    "credit_redistribution/orchestration.py",
    "credit_redistribution/protocol.py",
    "credit_redistribution/protocol_lock.py",
    "credit_redistribution/serialization.py",
    "credit_redistribution/state_digest.py",
    "credit_redistribution/statistics.py",
    "credit_redistribution/transcript.py",
    "credit_redistribution/tests/test_controller.py",
    "credit_redistribution/tests/test_evaluator.py",
    "credit_redistribution/tests/test_heldout.py",
    "credit_redistribution/tests/test_orchestration.py",
    "credit_redistribution/tests/test_protocol.py",
    "credit_redistribution/tests/test_state_digest.py",
    "credit_redistribution/tests/test_statistics.py",
    "credit_redistribution/tests/test_transcript.py",
    "credit_redistribution/tests/test_benchmark.py",
)


def _git_output(*arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_contract(require_clean=True, require_origin=True):
    commit = _git_output("rev-parse", "HEAD")
    origin = _git_output("rev-parse", "origin/repa")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if require_origin and commit != origin:
        raise RuntimeError("Generated protocol requires HEAD == origin/repa")
    if require_clean and status:
        raise RuntimeError("Generated protocol requires a clean worktree")
    return {
        "branch": _git_output("branch", "--show-current"),
        "commit": commit,
        "origin_repa": origin,
        "worktree_clean": not bool(status),
    }


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Experiment config is not a mapping: {path}")
    return value


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON document is not a mapping: {path}")
    return value


def _validate_branch_configs():
    rows = []
    shared = None
    base_config = _load_yaml(BASE_CONFIG)
    base_model_config = base_config.get("DiT_B_config")
    ignored_differences = {
        "credit_redistribution_config",
    }
    for definition in BRANCH_DEFINITIONS:
        config_path = (PROJECT_ROOT / definition["config"]).resolve()
        launcher_path = (PROJECT_ROOT / definition["launcher"]).resolve()
        if not config_path.is_file() or not launcher_path.is_file():
            raise FileNotFoundError(f"Branch definition is incomplete: {definition}")
        config = _load_yaml(config_path)
        controller = config.get("credit_redistribution_config")
        if not isinstance(controller, dict):
            raise ValueError(f"Branch controller config is absent: {config_path}")
        if controller.get("branch") != definition["name"]:
            raise ValueError(f"Branch/config mismatch: {config_path}")
        exact = {
            "model_name": "ProMoE_TC_B",
            "output_dir": str(RUN_OUTPUT_ROOT),
            "gpu_ids": [4, 5, 6, 7],
            "total_train_batch_size": 256,
            "lr": 0.0001,
            "weight_decay": 0,
            "global_seed": 0,
            "img_num_workers": 16,
            "prefetch_factor": 2,
            "use_pre_latents": True,
            "use_encoded_latents": True,
            "latent_data_path": str(LATENT_ROOT),
            "resume_checkpoint": True,
            "num_steps": 321001,
            "save_ckpt_interval": 1000,
        }
        for key, expected in exact.items():
            if config.get(key) != expected:
                raise ValueError(f"{config_path}:{key} differs from {expected!r}")
        if config.get("DiT_B_config") != base_model_config:
            raise ValueError(f"{config_path}:DiT_B_config differs from frozen Base")
        if "resume_checkpoint_step" in config:
            raise ValueError(f"Branch config must not pin a local resume step: {config_path}")
        if controller.get("execution_mode") != "continuation":
            raise ValueError(f"Branch execution mode differs: {config_path}")
        if controller.get("initial_checkpoint_path") != str(FROZEN_CHECKPOINT):
            raise ValueError(f"Branch frozen checkpoint differs: {config_path}")
        expected_artifact_root = (
            DEFAULT_OUTPUT_ROOT / "branches" / definition["name"]
        ).resolve()
        if Path(controller.get("artifact_root", "")).resolve() != expected_artifact_root:
            raise ValueError(f"Branch artifact root differs: {config_path}")
        expected_reference = (
            None
            if definition["name"] == "measure_only_control"
            else str(
                (DEFAULT_OUTPUT_ROOT / "branches" / "measure_only_control").resolve()
            )
        )
        if controller.get("reference_artifact_root") != expected_reference:
            raise ValueError(f"Branch reference artifact root differs: {config_path}")
        expected_controller_keys = {
            "enabled",
            "branch",
            "execution_mode",
            "initial_checkpoint_path",
            "preregister_v3_path",
            "preregister_v4_path",
            "artifact_root",
        }
        if expected_reference is not None:
            expected_controller_keys.add("reference_artifact_root")
        if set(controller) != expected_controller_keys:
            raise ValueError(f"Branch controller fields differ: {config_path}")
        if controller.get("preregister_v3_path") != str(V3_PATH):
            raise ValueError(f"Branch v3 preregistration path differs: {config_path}")
        if controller.get("preregister_v4_path") != str(V4_PATH):
            raise ValueError(f"Branch v4 preregistration path differs: {config_path}")

        if not bool(launcher_path.stat().st_mode & 0o111):
            raise PermissionError(f"Branch launcher is not executable: {launcher_path}")
        launcher = launcher_path.read_text(encoding="utf-8")
        required_launcher_fragments = (
            f'CONFIG="{definition["config"]}"',
            'PYTHON="/mnt/workspace/yujie/.conda/envs/promoe/bin/python"',
            f'--branch {definition["name"]}',
            '"$PYTHON" train.py --config "$CONFIG"',
        )
        if any(fragment not in launcher for fragment in required_launcher_fragments):
            raise ValueError(f"Branch launcher semantics differ: {launcher_path}")
        comparable = {
            key: copy.deepcopy(value)
            for key, value in config.items()
            if key not in ignored_differences
        }
        if shared is None:
            shared = comparable
        elif comparable != shared:
            raise ValueError("Three branch configs differ outside controller settings")
        stem = config_path.stem
        output_dir = (
            Path(config["output_dir"]) / config["model_name"] / stem
        ).resolve()
        artifact_root = Path(controller["artifact_root"]).resolve()
        rows.append({
            "name": definition["name"],
            "config_path": str(config_path),
            "config_file_sha256": sha256_file(config_path),
            "launcher_path": str(launcher_path),
            "launcher_file_sha256": sha256_file(launcher_path),
            "output_dir": str(output_dir),
            "final_checkpoint_path": str(
                output_dir / "checkpoints" / "ckpt_step_321000.pth"
            ),
            "artifact_root": str(artifact_root),
            "reference_artifact_root": controller.get(
                "reference_artifact_root"
            ),
            "gpu_ids": [4, 5, 6, 7],
        })
    if tuple(row["name"] for row in rows) != BRANCHES:
        raise RuntimeError("Branch execution order changed")
    return rows


def _prerequisite_contract(effective):
    base = effective["prerequisites"]["base_gate"]
    cross = effective["prerequisites"]["cross_checkpoint_gate"]
    external_files = (
        (base["preregister_path"], base["preregister_sha256"]),
        (cross["preregister_v1_path"], cross["preregister_v1_sha256"]),
        (cross["preregister_v2_path"], cross["preregister_v2_sha256"]),
    )
    for path, expected in external_files:
        if sha256_file(path) != expected:
            raise RuntimeError(f"Prerequisite preregistration hash differs: {path}")
    parent_digest = canonical_json_sha256(_load_json(PARENT_PROTOCOL))
    if parent_digest != base["sealed_case_protocol_sha256"]:
        raise RuntimeError("Parent Base protocol canonical hash differs")
    if PARENT_PROTOCOL.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ) != parent_digest + "\n":
        raise RuntimeError("Parent Base protocol sidecar differs")
    return {
        "base_gate": {
            "name": base["name"],
            "preregister_path": base["preregister_path"],
            "preregister_file_sha256": base["preregister_sha256"],
            "protocol_path": str(PARENT_PROTOCOL),
            "protocol_canonical_sha256": parent_digest,
            "output_root": str(PARENT_PROTOCOL.parent),
            "required_summaries": ["plumbing", "discovery", "confirmatory"],
        },
        "cross_checkpoint_gate": {
            "name": cross["name"],
            "preregister_v1_path": cross["preregister_v1_path"],
            "preregister_v1_file_sha256": cross["preregister_v1_sha256"],
            "preregister_v2_path": cross["preregister_v2_path"],
            "preregister_v2_file_sha256": cross["preregister_v2_sha256"],
            "protocol_path": str(CROSS_CHECKPOINT_ROOT / "protocol.json"),
            "output_root": str(CROSS_CHECKPOINT_ROOT),
            "required_stage_order": [
                "plumbing",
                "discovery_count",
                "parameter_validation",
                "discovery_credit",
                "confirmatory_count",
                "confirmatory_credit",
            ],
            "required_summaries": [
                "plumbing",
                "discovery-count",
                "parameter",
                "discovery-credit",
                "confirmatory-count",
                "confirmatory",
            ],
        },
    }


def _source_hashes():
    paths = list(SOURCE_PATHS)
    paths.extend(definition["config"] for definition in BRANCH_DEFINITIONS)
    paths.extend(definition["launcher"] for definition in BRANCH_DEFINITIONS)
    hashes = {}
    for relative in sorted(set(paths)):
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Protocol source is absent: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def _sealed_gpu_device_pairs():
    """Return (physical slot, visible CUDA index) pairs for the sealed GPU group."""
    device_count = int(torch.cuda.device_count())
    if device_count < len(SEALED_GPU_IDS):
        raise RuntimeError(
            "The sealed environment requires four visible CUDA devices; "
            f"found {device_count}"
        )

    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [] if raw_visible is None else [
        token.strip() for token in raw_visible.split(",") if token.strip()
    ]
    declared_mapping = os.environ.get(SEALED_GPU_MAPPING_ENV)
    declared_physical_ids = None
    if declared_mapping is not None:
        declared_tokens = [
            token.strip()
            for token in declared_mapping.split(",")
            if token.strip()
        ]
        try:
            declared_physical_ids = [int(token) for token in declared_tokens]
        except ValueError as error:
            raise RuntimeError(
                f"{SEALED_GPU_MAPPING_ENV} must contain numeric GPU IDs"
            ) from error
        if declared_physical_ids != list(SEALED_GPU_IDS):
            raise RuntimeError(
                f"{SEALED_GPU_MAPPING_ENV} must explicitly declare 4,5,6,7"
            )
    numeric_tokens = None
    if tokens:
        try:
            numeric_tokens = [int(token) for token in tokens]
        except ValueError:
            # UUID/MIG visibility cannot expose physical indices.  The sealed
            # runner always allocates exactly the four target devices, so the
            # visible order is the only stable mapping available here.
            numeric_tokens = None
        else:
            if len(set(numeric_tokens)) != len(numeric_tokens):
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES contains duplicate physical GPU IDs"
                )
            if len(numeric_tokens) != device_count:
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES count differs from visible CUDA devices"
                )

    if declared_physical_ids is not None:
        if device_count != len(SEALED_GPU_IDS):
            raise RuntimeError(
                f"{SEALED_GPU_MAPPING_ENV} requires exactly four visible devices; "
                f"found {device_count}"
            )
        if numeric_tokens is not None:
            # A numeric mask is already an authoritative physical mapping;
            # never let a stale declaration reinterpret 0..3 as 4..7.
            if numeric_tokens != list(SEALED_GPU_IDS):
                raise RuntimeError(
                    f"{SEALED_GPU_MAPPING_ENV} contradicts numeric "
                    f"CUDA_VISIBLE_DEVICES={raw_visible!r}"
                )
            visible_order = numeric_tokens
        else:
            # UUID/MIG visibility cannot expose physical indices.  The
            # scheduler declaration is the explicit order in that case.
            visible_order = list(SEALED_GPU_IDS)
    elif not tokens and device_count >= 8:
        # With no visibility mask, CUDA indices retain their physical order.
        # The sealed group therefore maps directly to indices 4-7.
        visible_order = list(range(device_count))
    elif numeric_tokens is not None and all(
        physical_id in numeric_tokens for physical_id in SEALED_GPU_IDS
    ):
        visible_order = numeric_tokens
    elif numeric_tokens is None:
        raise RuntimeError(
            "Cannot prove that UUID/MIG visibility denotes the sealed physical "
            f"GPU group; set {SEALED_GPU_MAPPING_ENV}=4,5,6,7 explicitly"
        )
    else:
        raise RuntimeError(
            "Cannot map the sealed physical GPU group 4-7 to the visible "
            f"CUDA devices (CUDA_VISIBLE_DEVICES={raw_visible!r}, "
            f"device_count={device_count})"
        )

    return [
        (physical_id, visible_order.index(physical_id))
        for physical_id in SEALED_GPU_IDS
    ]


def _environment():
    devices = {}
    for physical_id, visible_index in _sealed_gpu_device_pairs():
        properties = torch.cuda.get_device_properties(visible_index)
        devices[f"cuda:{physical_id}"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "uuid": str(properties.uuid) if hasattr(properties, "uuid") else None,
        }
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "diffusers": diffusers.__version__,
        "timm": timm.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_devices": devices,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def _heldout_binding():
    if not HELDOUT_MANIFEST.is_file():
        raise FileNotFoundError("Held-out tensors must be materialized before protocol")
    with HELDOUT_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    digest = canonical_json_sha256(manifest)
    if HELDOUT_MANIFEST.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ) != digest + "\n":
        raise RuntimeError("Held-out manifest sidecar differs")
    return {
        "manifest_path": str(HELDOUT_MANIFEST),
        "manifest_file_sha256": sha256_file(HELDOUT_MANIFEST),
        "manifest_canonical_sha256": digest,
        "tensor_directory": str(HELDOUT_MANIFEST.parent / "tensors"),
    }


def build_protocol(require_clean=True, require_origin=True):
    effective = load_effective_protocol(V3_PATH, V4_PATH)
    git = git_contract(
        require_clean=require_clean,
        require_origin=require_origin,
    )
    branches = _validate_branch_configs()
    dataset_identity = _latent_dataset_identity(LATENT_ROOT)
    expected_identity = effective["checkpoint"]["provenance"]["dataset_identity"]
    if dataset_identity != expected_identity:
        raise RuntimeError("Current latent dataset identity differs from frozen checkpoint")
    if sha256_file(FROZEN_CHECKPOINT) != effective["checkpoint"]["sha256"]:
        raise RuntimeError("Frozen continuation checkpoint hash differs")
    prerequisites = _prerequisite_contract(effective)
    runtime_cfg = load_runtime_cfg(Path(branches[0]["config_path"]))
    model = _build_model_for_protocol(runtime_cfg)
    model_contract = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    del model
    return {
        "version": PROTOCOL_VERSION,
        "name": "promoe_credit_rate_budget_redistribution_three_arm_v4",
        "status": "immutable_pre_efficacy",
        "scientific_question": effective["scientific_question"],
        "claim_boundary": effective["claim_boundary"],
        "effective_preregistration": {
            "v3_path": str(V3_PATH),
            "v3_file_sha256": V3_SHA256,
            "v4_path": str(V4_PATH),
            "v4_file_sha256": V4_SHA256,
            "effective_version": effective["effective_version"],
            "bootstrap_resamples": effective["statistics"][
                "bootstrap_resamples"
            ],
        },
        "git": git,
        "frozen_checkpoint": {
            "path": str(FROZEN_CHECKPOINT),
            "file_sha256": effective["checkpoint"]["sha256"],
            "size_bytes": FROZEN_CHECKPOINT.stat().st_size,
            "step": effective["checkpoint"]["step"],
        },
        "dataset": {
            "latent_root": str(LATENT_ROOT),
            "identity": dataset_identity,
        },
        "heldout": _heldout_binding(),
        "branches": branches,
        "model_contract": model_contract,
        "project_source_file_sha256": _source_hashes(),
        "evaluator_source_file_sha256": sha256_file(
            PROJECT_ROOT / "credit_redistribution/evaluator.py"
        ),
        "environment": _environment(),
        "prerequisites": prerequisites,
        "preflight": {
            "replay_summary_path": str(DEFAULT_OUTPUT_ROOT / "preflight" / "replay-summary.json"),
            "throughput_summary_path": str(
                DEFAULT_OUTPUT_ROOT / "throughput" / "summary.json"
            ),
        },
        "heldout_evaluation_output": str(DEFAULT_OUTPUT_ROOT / "evaluation"),
    }


def _build_model_for_protocol(runtime_cfg):
    from analyses.denoising_regret.probe import _build_model

    with torch.random.fork_rng(devices=[]):
        return _build_model(runtime_cfg)


def write_protocol(path=DEFAULT_PROTOCOL_PATH):
    path = Path(path).resolve()
    protocol = build_protocol()
    digest = canonical_json_sha256(protocol)
    sidecar = path.with_suffix(".sha256")
    if path.exists() or sidecar.exists():
        if not path.exists() or not sidecar.exists():
            raise RuntimeError("Generated protocol and sidecar must exist together")
        with path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != protocol:
                raise RuntimeError("Existing immutable protocol differs")
        if sidecar.read_text(encoding="utf-8") != digest + "\n":
            raise RuntimeError("Existing immutable protocol sidecar differs")
    else:
        atomic_write_json(path, protocol, mode=0o444)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(digest + "\n", encoding="utf-8")
        os.chmod(sidecar, 0o444)
    os.chmod(path, 0o444)
    return path, digest


def load_and_verify_protocol(path=DEFAULT_PROTOCOL_PATH, require_git=True):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    digest = canonical_json_sha256(protocol)
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != digest + "\n":
        raise RuntimeError("Immutable protocol sidecar differs")
    if protocol.get("status") != "immutable_pre_efficacy":
        raise ValueError("Immutable protocol status differs")
    # Rebuild every protocol field from the current repository and external
    # prerequisites.  Checking only the JSON sidecar is insufficient because
    # an attacker could edit both the document and its sidecar together.
    expected = build_protocol(
        require_clean=bool(require_git),
        require_origin=bool(require_git),
    )
    if require_git:
        if protocol != expected:
            raise RuntimeError(
                "Immutable protocol differs from the reconstructed contract"
            )
    else:
        observed = copy.deepcopy(protocol)
        observed.pop("git", None)
        reconstructed = copy.deepcopy(expected)
        reconstructed.pop("git", None)
        if observed != reconstructed:
            raise RuntimeError(
                "Immutable protocol differs from the reconstructed contract"
            )
    return protocol, digest
