"""Sealed, resume-safe multi-GPU runner for finite-horizon routing."""

from __future__ import annotations

import ctypes
import fcntl
import gc
import hashlib
import inspect
import json
import multiprocessing
import os
import shutil
import signal
import stat
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import torch
import diffusers
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from credit_redistribution.git_provenance import (
    authoritative_remote_tip,
    git_output,
    reject_history_overrides,
    repository_state,
    run_git,
)
from analyses.denoising_regret.io import write_json_atomic
from analyses.denoising_regret.probe import _build_model
from analyses.fresh_base_routing.audit import (
    _checkpoint_path,
    _dataset_identity_from_latent_root,
    _fresh_training_log_snapshot,
    _runtime_environment,
    _trainer_state_contract,
    _validate_config_payload,
    _validate_run_dir,
    _validate_training_provenance_contract,
    _verify_training_log,
    load_manifest as load_source_manifest,
)
from analyses.t_SNE.checkpoint_utils import (
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)

from .batch import (
    BATCH_VERSION,
    CONFIRMATORY_REQUIREMENTS,
    DISCOVERY_REQUIREMENTS,
    PERMUTATION_POLICY,
    PERMUTATION_RESAMPLES,
    PERMUTATION_SEEDS,
    SAFETY_REQUIREMENTS,
    SPLIT_COUNTS,
    aggregate_case_results,
    requirements_for_split,
)
from .probe import _load_verified_runtime_cfg, run_finite_horizon_routing_probe
from .protocol import (
    BLOCK_INDICES,
    CANDIDATE_CHUNK_SIZE,
    CANDIDATE_COUNT,
    HORIZONS,
    NUM_TRAIN_TIMESTEPS,
    PROBE_VERSION,
    SAMPLE_SHIFT,
    SAMPLE_STEPS,
    SCHEDULER_SHIFT,
    START_INDICES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GATE_MANIFEST = (
    PROJECT_ROOT
    / "analyses"
    / "finite_horizon_routing"
    / "manifests"
    / "finite_horizon_routing_gate_v1.json"
)
SOURCE_CASE_MANIFEST = (
    PROJECT_ROOT
    / "analyses"
    / "fresh_base_routing"
    / "manifests"
    / "fresh_base_routing_audit_v1.json"
)
SOURCE_CASE_MANIFEST_SHA256 = (
    "41affd3a92f7c407fba33f894a10ee2392fc0cd25d105750c6dc095ea22a4824"
)
LOCKED_BRANCH = "analysis/long-horizon-routing-v2"
LOCKED_DEVICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
CHECKPOINT_STEP = 300_000
FRESH_CHECKPOINT_STEPS = (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)
CHECKPOINT_STATE = "ema_model_state_dict"
MODEL_NAME = "ProMoE_TC_B"
CONFIG_STEM = "004_ProMoE_B_fresh_routing_audit_s0_v2"
FRESH_CONFIG_SHA256 = (
    "97fe9376303cc390eada34e2bc82fa903b998b78c82d181486630a25187c0ab6"
)
FRESH_TRAINING_CONFIG_SHA256 = (
    "c11983626dd8e65cf6074be4792c3f37a662acb01561537baca968a7db2ccca9"
)
FRESH_TRAINING_COMMIT = "3465ec3cd166c74a066970422b9e2a7134e1f9cb"
BASE_MODEL_CLASS = "models.models_ProMoE_TC.DiT"
BASE_PARAMETER_COUNT = 300_607_520
PROTOCOL_FILENAME = "protocol.json"
PROTOCOL_SEAL_FILENAME = "protocol.sha256"
RESULT_SHA256_FIELD = "result_sha256"
SUMMARY_SHA256_FIELD = "summary_sha256"
GLOBAL_RUN_LOCK = Path("/tmp/promoe-finite-horizon-routing-cuda-0-3.lock")
OUTPUT_RUN_LOCK_FILENAME = ".run-split.lock"
PR_SET_PDEATHSIG = 1
TRAINING_ENVIRONMENT_PROVENANCE_LIMITATIONS = (
    "cuda_driver_version",
    "cudnn_runtime_version",
)
SPLIT_PREREQUISITES = {
    "plumbing": (),
    "discovery": ("plumbing",),
    "confirmatory": ("plumbing", "discovery"),
}
STATIC_SOURCE_PATHS = (
    "requirements.txt",
    "config.py",
    "sample.py",
    "train.py",
    "models/models_ProMoE_TC.py",
    "analyses/denoising_regret/probe.py",
    "analyses/routing_translation/probe.py",
    "analyses/fresh_base_routing/audit.py",
    "analyses/t_SNE/checkpoint_utils.py",
    "analyses/timestep_utility/probe.py",
    "analyses/timestep_utility/cycle_probe.py",
    "analyses/finite_horizon_routing/__init__.py",
    "analyses/finite_horizon_routing/protocol.py",
    "analyses/finite_horizon_routing/probe.py",
    "analyses/finite_horizon_routing/batch.py",
    "analyses/finite_horizon_routing/runner.py",
    "analyses/run_finite_horizon_routing_probe.py",
    "analyses/run_finite_horizon_routing_probe_batch.py",
    "analyses/finite_horizon_routing/manifests/finite_horizon_routing_gate_v1.json",
    "analyses/fresh_base_routing/manifests/fresh_base_routing_audit_v1.json",
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_handle(handle, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    handle.seek(0)
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _stat_identity(file_stat):
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


@contextmanager
def _open_stable_regular_file(path, description):
    """Keep one regular-file object stable across validation and loading."""

    path = Path(path)
    flags = os.O_RDONLY
    for name in ("O_NOFOLLOW", "O_CLOEXEC"):
        value = getattr(os, name, None)
        if value is None:
            raise OSError(f"The platform does not provide {name}")
        flags |= value
    descriptor = os.open(path, flags)
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError(f"{description} must be a regular file")
        with os.fdopen(descriptor, "rb", buffering=0) as handle:
            descriptor = None
            yield handle, opened_stat
            final_stat = os.fstat(handle.fileno())
            if _stat_identity(final_stat) != _stat_identity(opened_stat):
                raise RuntimeError(f"{description} changed while it was open")
            try:
                path_stat = os.stat(path, follow_symlinks=False)
            except FileNotFoundError as error:
                raise RuntimeError(f"{description} path disappeared while it was open") from error
            if _stat_identity(path_stat) != _stat_identity(opened_stat):
                raise RuntimeError(f"{description} path changed while it was open")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _file_identity(path, description, expected_sha256=None):
    with _open_stable_regular_file(path, description) as (handle, opened_stat):
        first_sha256 = _sha256_handle(handle)
        second_sha256 = _sha256_handle(handle)
        if first_sha256 != second_sha256:
            raise RuntimeError(f"{description} changed while it was hashed")
        if expected_sha256 is not None and first_sha256 != expected_sha256:
            raise ValueError(f"{description} SHA256 differs from its locked identity")
        return {
            "size": opened_stat.st_size,
            "sha256": first_sha256,
        }


@contextmanager
def _checkpoint_safe_globals():
    """Allow only metadata classes used by this project's trusted checkpoints.

    PyTorch 2.6 defaults to ``weights_only=True``.  ProMoE checkpoints contain
    an ``EasyDict`` configuration and a ``TorchVersion`` metadata value in
    addition to tensors, so the default restricted unpickler needs these two
    explicitly scoped types.  The caller has already opened and hashed a
    stable regular file before entering this context.
    """

    safe_globals = getattr(getattr(torch, "serialization", None), "safe_globals", None)
    if safe_globals is None:
        yield
        return

    try:
        from easydict import EasyDict
        from torch.torch_version import TorchVersion
    except ImportError as error:
        raise RuntimeError(
            "The restricted checkpoint loader cannot import its locked "
            "metadata types"
        ) from error

    with safe_globals([EasyDict, TorchVersion]):
        yield


def _torch_load_handle(handle):
    load_kwargs = {"map_location": "cpu", "weights_only": True}
    handle.seek(0)
    try:
        with _checkpoint_safe_globals():
            checkpoint = torch.load(handle, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only")
        handle.seek(0)
        checkpoint = torch.load(handle, **load_kwargs)
    handle.seek(0)
    return checkpoint


def _main_worktree_root(project_root=PROJECT_ROOT):
    """Resolve the primary worktree from this linked worktree's common Git dir."""

    project_root = Path(project_root).resolve(strict=True)
    common_dir = Path(git_output(project_root, "rev-parse", "--git-common-dir"))
    if not common_dir.is_absolute():
        common_dir = project_root / common_dir
    common_dir = common_dir.resolve(strict=True)
    if common_dir.name != ".git" or not common_dir.is_dir():
        raise RuntimeError(f"Unexpected Git common directory: {common_dir}")
    main_root = common_dir.parent
    if (main_root / ".git").resolve(strict=True) != common_dir:
        raise RuntimeError("Git common directory is not rooted in the main worktree")
    return main_root


def _json_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return payload


def _canonical_gate_manifest():
    payload = _read_json(GATE_MANIFEST)
    if (
        payload.get("version") != 1
        or payload.get("name") != "finite-horizon-routing-gate-v1"
        or payload.get("locked_before_any_fresh_finite_horizon_result") is not True
    ):
        raise ValueError("Finite-horizon gate manifest is not locked")
    source = payload.get("source_case_manifest", {})
    if source != {
        "path": "analyses/fresh_base_routing/manifests/fresh_base_routing_audit_v1.json",
        "sha256": SOURCE_CASE_MANIFEST_SHA256,
    }:
        raise ValueError("Gate manifest source cases are not canonical")
    expected_permutation = {
        **PERMUTATION_POLICY,
        "resamples": PERMUTATION_RESAMPLES,
        "seeds": PERMUTATION_SEEDS,
    }
    if payload.get("candidate_label_permutation") != expected_permutation:
        raise ValueError("Candidate-label permutation policy changed")
    if payload.get("safety_requirements") != SAFETY_REQUIREMENTS:
        raise ValueError("Safety requirements differ from the gate manifest")
    splits = payload.get("splits", {})
    for split, count in SPLIT_COUNTS.items():
        if splits.get(split, {}).get("expected_case_count") != count:
            raise ValueError(f"Gate manifest case count changed for {split}")
    if splits["discovery"].get("requirements") != DISCOVERY_REQUIREMENTS:
        raise ValueError("Discovery requirements differ from the gate manifest")
    if splits["confirmatory"].get("requirements") != CONFIRMATORY_REQUIREMENTS:
        raise ValueError("Confirmatory requirements differ from the gate manifest")
    if splits["plumbing"].get("efficacy_statistics_withheld") is not True:
        raise ValueError("Plumbing efficacy must remain withheld")
    return payload


def _git_contract(locked_branch=LOCKED_BRANCH):
    remote_ref = f"refs/heads/{locked_branch}"
    state = repository_state(
        PROJECT_ROOT,
        authoritative_remote_ref=remote_ref,
    )
    branch = state["branch"]
    commit = state["commit"]
    if branch != locked_branch:
        raise RuntimeError(f"Gate must run from branch {locked_branch}, got {branch}")
    if state["status"]:
        raise RuntimeError("Gate requires a clean committed worktree")
    if state["authoritative_remote_tip"] != commit:
        raise RuntimeError(
            "Gate commit must already be pushed to the authoritative analysis branch"
        )
    return {
        "branch": branch,
        "commit": commit,
        "authoritative_remote_url": state["authoritative_remote_url"],
        "authoritative_remote_ref": state["authoritative_remote_ref"],
        "authoritative_remote_tip": state["authoritative_remote_tip"],
        "status_clean": True,
    }


def _authoritative_repa_commit():
    return authoritative_remote_tip()


def _validate_training_commit(
    training_commit,
    logged_commit,
    expected_training_commit,
):
    if training_commit != expected_training_commit or logged_commit != training_commit:
        raise ValueError("Fresh training commit differs from the locked run identity")

    reject_history_overrides(PROJECT_ROOT)
    authoritative_tip = _authoritative_repa_commit()
    local_tip = git_output(
        PROJECT_ROOT,
        "rev-parse",
        "--verify",
        "refs/remotes/origin/repa^{commit}",
    )
    if local_tip != authoritative_tip:
        raise RuntimeError(
            "Local origin/repa is stale relative to the authoritative remote"
        )
    ancestry = run_git(
        PROJECT_ROOT,
        "merge-base",
        "--is-ancestor",
        expected_training_commit,
        authoritative_tip,
        text=True,
    )
    if ancestry.returncode == 1:
        raise ValueError(
            "Locked Fresh training commit is no longer in authoritative repa history"
        )
    if ancestry.returncode != 0:
        raise RuntimeError("Could not verify Fresh training commit ancestry")
    return authoritative_tip


def _analysis_runtime_environment(devices):
    environment = _runtime_environment(devices)
    scheduler_source = inspect.getsource(FlowMatchEulerDiscreteScheduler)
    driver_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    driver_versions = sorted({
        line.strip()
        for line in driver_query.stdout.splitlines()
        if line.strip()
    })
    if len(driver_versions) != 1:
        raise RuntimeError(
            f"Expected one NVIDIA driver version, found {driver_versions}"
        )
    cudnn_version = torch.backends.cudnn.version()
    if (
        isinstance(cudnn_version, bool)
        or not isinstance(cudnn_version, int)
        or cudnn_version <= 0
    ):
        raise RuntimeError(f"cuDNN runtime version is unavailable: {cudnn_version}")
    environment.update({
        "diffusers": diffusers.__version__,
        "cuda_driver_version": driver_versions[0],
        "cudnn_runtime_version": cudnn_version,
        "flow_match_scheduler_source_sha256": hashlib.sha256(
            scheduler_source.encode("utf-8")
        ).hexdigest(),
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    })
    return environment


def _arm_parent_death_signal(expected_parent_pid):
    """Kill a spawned GPU worker if the coordinating process disappears."""

    expected_parent_pid = int(expected_parent_pid)
    if expected_parent_pid < 1:
        raise ValueError("Expected parent PID must be positive")
    if os.name != "posix" or os.uname().sysname != "Linux":
        raise RuntimeError("The locked GPU gate requires Linux PR_SET_PDEATHSIG")
    library = ctypes.CDLL(None, use_errno=True)
    prctl = library.prctl
    prctl.restype = ctypes.c_int
    if prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
        raise RuntimeError("GPU worker parent exited before the death signal was armed")


@contextmanager
def _exclusive_lock(path, description):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process holds the {description} lock: {path}") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _run_split_locks(output_dir):
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Protocol output directory is missing: {output_dir}")
    with _exclusive_lock(GLOBAL_RUN_LOCK, "GPU 0-3"):
        with _exclusive_lock(
            output_dir / OUTPUT_RUN_LOCK_FILENAME,
            "finite-horizon output",
        ):
            yield


def _publish_protocol(output_dir, protocol):
    """Publish a complete protocol directory once, under a sibling lock."""

    output_dir = Path(output_dir).resolve()
    lock_path = output_dir.parent / f".{output_dir.name}.prepare.lock"
    with _exclusive_lock(lock_path, "finite-horizon protocol preparation"):
        if os.path.lexists(output_dir):
            raise FileExistsError(
                f"Protocol output already exists; never overwrite a sealed gate: {output_dir}"
            )
        staging_dir = Path(tempfile.mkdtemp(
            prefix=f".{output_dir.name}.prepare-",
            dir=output_dir.parent,
        ))
        try:
            write_json_atomic(staging_dir / PROTOCOL_FILENAME, protocol)
            (staging_dir / PROTOCOL_SEAL_FILENAME).write_text(
                protocol["protocol_sha256"] + "\n",
                encoding="ascii",
            )
            os.rename(staging_dir, output_dir)
            staging_dir = None
        finally:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)


def _sealed_payload(payload, hash_field):
    sealed = dict(payload)
    sealed.pop(hash_field, None)
    sealed[hash_field] = _json_sha256(sealed)
    return sealed


def _read_sealed_json(path, hash_field):
    payload = _read_json(path)
    claimed = payload.pop(hash_field, None)
    observed = _json_sha256(payload)
    payload[hash_field] = claimed
    if claimed != observed:
        raise ValueError(f"Sealed JSON content changed: {path}")
    return payload


def _write_sealed_json(path, payload, hash_field):
    path = Path(path)
    sealed = _sealed_payload(payload, hash_field)
    write_json_atomic(path, sealed)
    return sealed


def _source_hashes(config_path, expected_config_sha256=None):
    paths = list(STATIC_SOURCE_PATHS) + [
        str(Path(config_path).resolve().relative_to(PROJECT_ROOT.resolve()))
    ]
    hashes = {}
    for relative in sorted(set(paths)):
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Locked source is missing: {path}")
        hashes[relative] = sha256_file(path)
    config_relative = str(Path(config_path).resolve().relative_to(PROJECT_ROOT.resolve()))
    if (
        expected_config_sha256 is not None
        and hashes[config_relative] != expected_config_sha256
    ):
        raise ValueError("Fresh Base config changed while the protocol was built")
    return hashes


def _read_checkpoint_record(
    checkpoint_path,
    runtime_cfg,
    fresh_training_log,
    *,
    expected_config_stem=CONFIG_STEM,
    expected_training_config_sha256=FRESH_TRAINING_CONFIG_SHA256,
    expected_training_commit=FRESH_TRAINING_COMMIT,
):
    """Load and identify one checkpoint through the same open file handle."""

    checkpoint_path = Path(checkpoint_path)
    if parse_checkpoint_step(checkpoint_path) != CHECKPOINT_STEP:
        raise ValueError(f"Gate requires checkpoint step {CHECKPOINT_STEP}")
    with _open_stable_regular_file(
        checkpoint_path,
        "Fresh 300K checkpoint",
    ) as (checkpoint_handle, opened_stat):
        first_sha256 = _sha256_handle(checkpoint_handle)
        checkpoint = _torch_load_handle(checkpoint_handle)
        try:
            if checkpoint.get("step") != CHECKPOINT_STEP:
                raise ValueError("Checkpoint payload step differs from its filename")
            if "model_state_dict" not in checkpoint:
                raise KeyError("Checkpoint is missing model_state_dict")
            if "ema_model_state_dict" not in checkpoint:
                raise KeyError("Checkpoint is missing ema_model_state_dict")
            trainer_contract = _trainer_state_contract(
                checkpoint.get("trainer_state"),
                expected_step=CHECKPOINT_STEP,
                expected_world_size=len(runtime_cfg.gpu_ids),
                expected_global_seed=int(runtime_cfg.global_seed),
                expected_total_batch_size=int(runtime_cfg.total_train_batch_size),
                expected_run_id=fresh_training_log["run_id"],
                expected_training_provenance_sha256=fresh_training_log[
                    "training_provenance_sha256"
                ],
                expected_training_config_stem=expected_config_stem,
                expected_training_config_sha256=(
                    expected_training_config_sha256
                ),
                training_git_contract={
                    "commit": expected_training_commit,
                    "origin_repa_divergence": "0\t0",
                },
                training_source_project_root=PROJECT_ROOT,
            )
        finally:
            del checkpoint
            gc.collect()
        second_sha256 = _sha256_handle(checkpoint_handle)
        if second_sha256 != first_sha256:
            raise RuntimeError("Fresh 300K checkpoint changed while it was loaded")

    return {
        "path": str(checkpoint_path),
        "resolved_path": str(checkpoint_path.resolve(strict=True)),
        "step": CHECKPOINT_STEP,
        "size": opened_stat.st_size,
        "sha256": first_sha256,
        "state": CHECKPOINT_STATE,
        "run_id": trainer_contract["run_id"],
        "trainer_contract": trainer_contract,
    }


def _checkpoint_contract(
    checkpoint_path,
    latent_root,
    *,
    config_stem=CONFIG_STEM,
    config_sha256=FRESH_CONFIG_SHA256,
    training_config_sha256=FRESH_TRAINING_CONFIG_SHA256,
    training_commit=FRESH_TRAINING_COMMIT,
):
    training_project_root = _main_worktree_root()
    training_output_root = training_project_root / "outputs"
    expected_run_dir = training_output_root / MODEL_NAME / config_stem
    run_dir = _validate_run_dir(
        expected_run_dir,
        output_root=training_output_root,
        expected_config_stem=config_stem,
    )
    if run_dir != expected_run_dir:
        raise ValueError("Fresh Base run is not under the main worktree output root")
    canonical_checkpoint_path = _checkpoint_path(
        run_dir,
        CHECKPOINT_STEP,
        output_root=training_output_root,
        expected_config_stem=config_stem,
    )
    supplied_checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    if supplied_checkpoint_path != canonical_checkpoint_path.resolve(strict=True):
        raise ValueError("Gate checkpoint is not the canonical Fresh Base run checkpoint")
    checkpoint_path = canonical_checkpoint_path
    if parse_checkpoint_step(checkpoint_path) != CHECKPOINT_STEP:
        raise ValueError(f"Gate requires checkpoint step {CHECKPOINT_STEP}")
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    if config_path.stem != config_stem:
        raise ValueError(f"Gate requires config {config_stem}")
    runtime_cfg, config_payload, config_identity = _load_verified_runtime_cfg(
        config_path,
        expected_sha256=config_sha256,
    )
    _validate_config_payload(config_payload, latent_root=latent_root)
    if runtime_cfg.model_name != MODEL_NAME:
        raise ValueError(f"Gate requires model_name {MODEL_NAME}")
    for name, expected in (
        ("sample_steps", SAMPLE_STEPS),
        ("sample_shift", SAMPLE_SHIFT),
        ("shift", SCHEDULER_SHIFT),
        ("num_train_timesteps", NUM_TRAIN_TIMESTEPS),
    ):
        if getattr(runtime_cfg, name) != expected:
            raise ValueError(f"Gate requires {name}={expected}")

    fresh_training_log = _fresh_training_log_snapshot(
        run_dir,
        checkpoint_steps=FRESH_CHECKPOINT_STEPS,
        project_root=training_project_root,
        expected_config_stem=config_stem,
        expected_training_config_sha256=training_config_sha256,
    )
    training_environment = _runtime_environment(LOCKED_DEVICES)
    analysis_environment = _analysis_runtime_environment(LOCKED_DEVICES)
    model = _build_model(runtime_cfg)
    model_metadata = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    del model
    gc.collect()
    if model_metadata != {
        "class": BASE_MODEL_CLASS,
        "parameter_count": BASE_PARAMETER_COUNT,
    }:
        raise ValueError("Checkpoint config does not build the locked Base ProMoE")
    checkpoint_record = _read_checkpoint_record(
        checkpoint_path,
        runtime_cfg,
        fresh_training_log,
        expected_config_stem=config_stem,
        expected_training_config_sha256=training_config_sha256,
        expected_training_commit=training_commit,
    )
    trainer_contract = checkpoint_record["trainer_contract"]

    checkpoint_marker = fresh_training_log["checkpoint_markers"][
        str(CHECKPOINT_STEP)
    ]
    if (
        checkpoint_marker["size"] != checkpoint_record["size"]
        or checkpoint_marker["sha256"] != checkpoint_record["sha256"]
    ):
        raise ValueError("Fresh 300K checkpoint differs from its training-log marker")
    _verify_training_log(
        fresh_training_log,
        checkpoints={str(CHECKPOINT_STEP): checkpoint_record},
        checkpoint_steps_to_bind=(CHECKPOINT_STEP,),
        run_dir=run_dir,
        output_root=training_output_root,
        project_root=training_project_root,
        expected_config_stem=config_stem,
        expected_training_config_sha256=training_config_sha256,
    )

    training_provenance = trainer_contract["training_provenance"]
    _validate_training_provenance_contract(
        training_provenance,
        expected_sha256=fresh_training_log["training_provenance_sha256"],
        git_contract={
            "commit": training_commit,
            "origin_repa_divergence": "0\t0",
        },
        environment=training_environment,
        expected_config_stem=config_stem,
        expected_config_payload_sha256=training_config_sha256,
        source_project_root=PROJECT_ROOT,
    )
    observed_training_commit = training_provenance["git"]["commit"]
    origin_repa_commit = _validate_training_commit(
        observed_training_commit,
        fresh_training_log["training_git_commit"],
        training_commit,
    )

    dataset_identity = trainer_contract["trajectory"]["sampler_contract"]["dataset"]
    observed_dataset_identity = _dataset_identity_from_latent_root(
        latent_root,
        dataset_identity["type"],
    )
    if observed_dataset_identity != dataset_identity:
        raise ValueError("Current latent dataset differs from the Fresh training dataset")

    return (
        checkpoint_path,
        config_path,
        runtime_cfg,
        model_metadata,
        checkpoint_record,
        {
            "main_worktree_root": str(training_project_root),
            "main_output_root": str(training_output_root),
            "run_dir": str(run_dir),
            "resolved_run_dir": str(run_dir.resolve(strict=True)),
            "config_identity": config_identity,
            "training_log": fresh_training_log,
            "training_commit": observed_training_commit,
            "origin_repa_commit": origin_repa_commit,
            "dataset_identity": dataset_identity,
            "observed_dataset_identity": observed_dataset_identity,
            "training_environment": training_environment,
            "analysis_environment": analysis_environment,
            "training_environment_provenance_limitations": list(
                TRAINING_ENVIRONMENT_PROVENANCE_LIMITATIONS
            ),
        },
    )


def _default_output_dir(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    return (
        checkpoint_path.parent.parent
        / "sample"
        / f"step{CHECKPOINT_STEP}"
        / "finite_horizon_routing_gate_v1"
    )


def _build_protocol_payload(
    checkpoint_path,
    latent_root,
    devices=LOCKED_DEVICES,
    output_dir=None,
):
    """Rebuild the complete gate contract from authoritative current inputs."""

    devices = tuple(devices)
    if devices != LOCKED_DEVICES:
        raise ValueError(f"Locked devices are {LOCKED_DEVICES}")
    gate_manifest = _canonical_gate_manifest()
    if sha256_file(SOURCE_CASE_MANIFEST) != SOURCE_CASE_MANIFEST_SHA256:
        raise ValueError("Fresh case manifest SHA256 changed")
    git = _git_contract()
    (
        checkpoint_path,
        config_path,
        runtime_cfg,
        model_metadata,
        checkpoint_record,
        fresh_run,
    ) = _checkpoint_contract(checkpoint_path, latent_root)
    if Path(runtime_cfg.latent_data_path).resolve() != Path(latent_root).resolve():
        raise ValueError("Runtime latent root differs from the gate input")
    output_dir = Path(output_dir or _default_output_dir(checkpoint_path)).resolve()
    source_manifest = load_source_manifest(SOURCE_CASE_MANIFEST, latent_root)
    cases = []
    for case in source_manifest["cases"]:
        latent_identity = _file_identity(
            case["latent"],
            f"Latent {case['id']}",
            expected_sha256=case["latent_sha256"],
        )
        cases.append({
            "split": case["split"],
            "id": case["id"],
            "label": case["label"],
            "seed": case["seed"],
            "synset": case["synset"],
            "latent_relative": case["latent_relative"],
            "latent": case["latent"],
            "latent_key": case["latent_key"],
            "latent_size": latent_identity["size"],
            "latent_sha256": latent_identity["sha256"],
        })
    assignments = {
        split: [
            {
                "case_id": case["id"],
                "device": devices[index % len(devices)],
            }
            for index, case in enumerate(
                [item for item in cases if item["split"] == split]
            )
        ]
        for split in SPLIT_COUNTS
    }
    protocol = {
        "batch_version": BATCH_VERSION,
        "probe_version": PROBE_VERSION,
        "locked_before_any_fresh_finite_horizon_result": True,
        "git": git,
        "worktrees": {
            "analysis": str(PROJECT_ROOT.resolve(strict=True)),
            "training": fresh_run["main_worktree_root"],
            "training_outputs": fresh_run["main_output_root"],
        },
        "gate_manifest": {
            "path": str(GATE_MANIFEST),
            "sha256": sha256_file(GATE_MANIFEST),
            "payload": gate_manifest,
        },
        "source_case_manifest": source_manifest,
        "checkpoint": checkpoint_record,
        "config": {
            "path": str(config_path),
            "size": fresh_run["config_identity"]["size"],
            "sha256": fresh_run["config_identity"]["sha256"],
            "stem": config_path.stem,
        },
        "model": model_metadata,
        "protocol": {
            "sample_steps": SAMPLE_STEPS,
            "sample_shift": SAMPLE_SHIFT,
            "scheduler_shift": SCHEDULER_SHIFT,
            "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
            "start_indices": list(START_INDICES),
            "horizons": list(HORIZONS),
            "block_indices": list(BLOCK_INDICES),
            "candidate_count": CANDIDATE_COUNT,
            "candidate_chunk_size": CANDIDATE_CHUNK_SIZE,
            "cfg_scale": 1.0,
            "intervention": "one two-token expert swap at one block and one step",
            "load_control": "complete expert-count vector fixed at intervention",
            "future_routing": "native router after the intervention forward",
        },
        "requirements": {
            split: requirements_for_split(split) for split in SPLIT_COUNTS
        },
        "devices": list(devices),
        "environment": fresh_run["analysis_environment"],
        "assignments": assignments,
        "cases": cases,
        "fresh_run": fresh_run,
        "source_hashes": _source_hashes(
            config_path,
            expected_config_sha256=FRESH_CONFIG_SHA256,
        ),
        "output_dir": str(output_dir),
        "novelty_boundary": (
            "This protocol tests a diffusion-MoE-specific diagnosis under strict "
            "quota control. It does not claim the first trajectory-aware router, "
            "the first rollout credit method, or the first balanced assignment."
        ),
    }
    return protocol


def prepare_protocol(checkpoint_path, latent_root, devices=LOCKED_DEVICES, output_dir=None):
    protocol = _sealed_payload(
        _build_protocol_payload(
            checkpoint_path=checkpoint_path,
            latent_root=latent_root,
            devices=devices,
            output_dir=output_dir,
        ),
        "protocol_sha256",
    )
    output_dir = Path(protocol["output_dir"])
    _publish_protocol(output_dir, protocol)
    return protocol


def _rebuild_protocol_payload(output_dir):
    """Derive protocol inputs without trusting any field in ``protocol.json``."""

    training_project_root = _main_worktree_root()
    training_output_root = training_project_root / "outputs"
    run_dir = _validate_run_dir(
        training_output_root / MODEL_NAME / CONFIG_STEM,
        output_root=training_output_root,
        expected_config_stem=CONFIG_STEM,
    )
    checkpoint_path = _checkpoint_path(
        run_dir,
        CHECKPOINT_STEP,
        output_root=training_output_root,
        expected_config_stem=CONFIG_STEM,
    )
    config_path = PROJECT_ROOT / "configs" / f"{CONFIG_STEM}.yaml"
    if config_path.is_symlink() or not config_path.is_file():
        raise FileNotFoundError(f"Canonical Fresh Base config is missing: {config_path}")
    runtime_cfg, _, _ = _load_verified_runtime_cfg(
        config_path,
        expected_sha256=FRESH_CONFIG_SHA256,
    )
    latent_root = Path(runtime_cfg.latent_data_path).resolve(strict=True)
    return _build_protocol_payload(
        checkpoint_path=checkpoint_path,
        latent_root=latent_root,
        devices=LOCKED_DEVICES,
        output_dir=output_dir,
    )


def verify_protocol(output_dir):
    output_dir = Path(output_dir).resolve()
    protocol_path = output_dir / PROTOCOL_FILENAME
    seal_path = output_dir / PROTOCOL_SEAL_FILENAME
    if not protocol_path.is_file() or not seal_path.is_file():
        raise FileNotFoundError("Protocol and seal must both exist")
    protocol = _read_json(protocol_path)
    claimed = protocol.pop("protocol_sha256", None)
    observed = _json_sha256(protocol)
    protocol["protocol_sha256"] = claimed
    sealed = seal_path.read_text(encoding="ascii").strip()
    if claimed != observed or sealed != observed:
        raise ValueError("Protocol JSON or seal changed")
    rebuilt_protocol = _sealed_payload(
        _rebuild_protocol_payload(output_dir),
        "protocol_sha256",
    )
    if protocol != rebuilt_protocol:
        raise ValueError(
            "Protocol differs from the canonical current run/config/source contract"
        )
    return protocol


def _result_path(output_dir, split, case_id):
    return Path(output_dir) / "cases" / split / f"{case_id}.json"


def _expected_case_metadata(case):
    return {
        key: case[key]
        for key in (
            "split",
            "id",
            "label",
            "seed",
            "synset",
            "latent_relative",
            "latent_key",
            "latent_size",
            "latent_sha256",
        )
    }


def _validate_case_result(result, protocol, case, device, path):
    if result.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError(f"Result protocol binding changed: {path}")
    if result.get("batch_case") != _expected_case_metadata(case):
        raise ValueError(f"Result case metadata changed: {path}")
    checkpoint = protocol["checkpoint"]
    identity = result.get("checkpoint_identity", {})
    expected_contract = {
        "canonical_size": checkpoint["size"],
        "canonical_sha256": checkpoint["sha256"],
        "weights_size": checkpoint["size"],
        "weights_sha256": checkpoint["sha256"],
        "same_file": True,
    }
    if identity != expected_contract:
        raise ValueError(f"Result checkpoint identity changed: {path}")
    expected_latent_identity = {
        "size": case["latent_size"],
        "sha256": case["latent_sha256"],
    }
    if result.get("latent_identity") != expected_latent_identity:
        raise ValueError(f"Result latent identity changed: {path}")
    expected_config_identity = {
        "size": protocol["config"]["size"],
        "sha256": protocol["config"]["sha256"],
    }
    if result.get("config_identity") != expected_config_identity:
        raise ValueError(f"Result config identity changed: {path}")
    exact_values = {
        "finite_horizon_routing_probe_version": PROBE_VERSION,
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_state": CHECKPOINT_STATE,
        "model_name": MODEL_NAME,
        "label": case["label"],
        "seed": case["seed"],
        "device": device,
        "sample_steps": SAMPLE_STEPS,
        "sample_shift": SAMPLE_SHIFT,
        "scheduler_shift": SCHEDULER_SHIFT,
        "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
        "start_indices": list(START_INDICES),
        "horizons": list(HORIZONS),
        "block_indices": list(BLOCK_INDICES),
        "candidate_count": CANDIDATE_COUNT,
        "candidate_chunk_size": CANDIDATE_CHUNK_SIZE,
    }
    for key, expected in exact_values.items():
        if result.get(key) != expected:
            raise ValueError(f"Result field {key!r} changed: {path}")
    path_values = {
        "checkpoint": checkpoint["path"],
        "weights_checkpoint": checkpoint["path"],
        "config": protocol["config"]["path"],
        "latent": case["latent"],
    }
    for key, expected in path_values.items():
        if Path(result.get(key, "")).resolve() != Path(expected).resolve():
            raise ValueError(f"Result path {key!r} changed: {path}")
    return result


def _load_case_result(path, protocol, case, device):
    result = _read_sealed_json(path, RESULT_SHA256_FIELD)
    return _validate_case_result(result, protocol, case, device, path)


def _run_device_cases(device, cases, protocol, output_dir):
    completed = []
    for case in cases:
        path = _result_path(output_dir, case["split"], case["id"])
        if path.exists():
            _load_case_result(path, protocol, case, device)
            completed.append(case["id"])
            continue
        result = run_finite_horizon_routing_probe(
            checkpoint_path=protocol["checkpoint"]["path"],
            latent_path=case["latent"],
            label=case["label"],
            latent_key=case["latent_key"],
            seed=case["seed"],
            device=device,
            num_threads=8,
            expected_checkpoint_size=protocol["checkpoint"]["size"],
            expected_checkpoint_sha256=protocol["checkpoint"]["sha256"],
            expected_config_sha256=protocol["config"]["sha256"],
            expected_latent_size=case["latent_size"],
            expected_latent_sha256=case["latent_sha256"],
        )
        result["batch_case"] = _expected_case_metadata(case)
        result["protocol_sha256"] = protocol["protocol_sha256"]
        _write_sealed_json(path, result, RESULT_SHA256_FIELD)
        completed.append(case["id"])
    return {"device": device, "completed": completed}


def _split_cases_and_devices(protocol, split):
    cases = [case for case in protocol["cases"] if case["split"] == split]
    assignments = protocol["assignments"][split]
    if len(cases) != SPLIT_COUNTS[split] or len(assignments) != len(cases):
        raise ValueError(f"Protocol case count changed for {split}")
    case_ids = [case["id"] for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"Protocol case IDs are duplicated for {split}")
    devices = {}
    for case, assignment in zip(cases, assignments):
        if assignment["case_id"] != case["id"]:
            raise ValueError("Protocol case/device assignment order changed")
        if assignment["device"] not in LOCKED_DEVICES:
            raise ValueError("Protocol assignment names an unlocked device")
        devices[case["id"]] = assignment["device"]
    return cases, devices


def _load_split_results(output_dir, protocol, split):
    cases, devices = _split_cases_and_devices(protocol, split)
    results = [
        _load_case_result(
            _result_path(output_dir, split, case["id"]),
            protocol,
            case,
            devices[case["id"]],
        )
        for case in cases
    ]
    return cases, results


def _build_summary(protocol, split, cases, results):
    summary = aggregate_case_results(
        results,
        split,
        prerequisite_discovery_passed=(True if split == "confirmatory" else None),
    )
    summary.update({
        "protocol_sha256": protocol["protocol_sha256"],
        "case_ids": [case["id"] for case in cases],
        "case_result_sha256": {
            result["batch_case"]["id"]: result[RESULT_SHA256_FIELD]
            for result in results
        },
    })
    return summary


def _verify_completed_split(output_dir, protocol, split):
    summary_path = Path(output_dir) / "summaries" / f"{split}.json"
    if not summary_path.is_file():
        raise RuntimeError(f"A completed {split} summary is required")
    cases, results = _load_split_results(output_dir, protocol, split)
    expected = _sealed_payload(
        _build_summary(protocol, split, cases, results),
        SUMMARY_SHA256_FIELD,
    )
    observed = _read_sealed_json(summary_path, SUMMARY_SHA256_FIELD)
    if observed != expected:
        raise RuntimeError(f"{split} summary differs from recomputed case results")
    return observed


def _run_split_locked(output_dir, split):
    protocol = verify_protocol(output_dir)
    output_dir = Path(output_dir).resolve()
    for prerequisite in SPLIT_PREREQUISITES[split]:
        summary = _verify_completed_split(output_dir, protocol, prerequisite)
        if summary.get("passed") is not True:
            raise RuntimeError(f"A passing {prerequisite} gate is required before {split}")

    cases, devices = _split_cases_and_devices(protocol, split)
    by_device = {
        device: [
            case
            for case in cases
            if devices[case["id"]] == device
        ]
        for device in LOCKED_DEVICES
    }
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(LOCKED_DEVICES),
        mp_context=context,
        initializer=_arm_parent_death_signal,
        initargs=(os.getpid(),),
    ) as executor:
        futures = {
            executor.submit(
                _run_device_cases,
                device,
                device_cases,
                protocol,
                str(output_dir),
            ): device
            for device, device_cases in by_device.items()
            if device_cases
        }
        for future in as_completed(futures):
            future.result()
    verified_protocol = verify_protocol(output_dir)
    if verified_protocol["protocol_sha256"] != protocol["protocol_sha256"]:
        raise RuntimeError("Protocol changed while the split was running")
    cases, results = _load_split_results(output_dir, protocol, split)
    summary = _build_summary(protocol, split, cases, results)
    summary_path = output_dir / "summaries" / f"{split}.json"
    if summary_path.exists():
        existing = _read_sealed_json(summary_path, SUMMARY_SHA256_FIELD)
        if existing != _sealed_payload(summary, SUMMARY_SHA256_FIELD):
            raise FileExistsError(f"Existing summary differs: {summary_path}")
        summary = existing
    else:
        summary = _write_sealed_json(
            summary_path,
            summary,
            SUMMARY_SHA256_FIELD,
        )
    return summary


def run_split(output_dir, split):
    if split not in SPLIT_COUNTS:
        raise ValueError(f"Unknown split: {split}")
    with _run_split_locks(output_dir):
        return _run_split_locked(output_dir, split)
