"""Sealed multi-checkpoint audit of Base ProMoE routing utility."""

from __future__ import annotations

import argparse
import copy
import errno
import fcntl
import gc
import hashlib
import json
import multiprocessing
import os
import platform
import re
import stat
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from analyses.denoising_regret.probe import _build_model
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)
from analyses.timestep_utility.batch import (
    BLOCK_INDICES,
    SIGMAS,
    _case_metrics,
    aggregate_case_results,
    sha256_file,
)
from analyses.timestep_utility.probe import (
    PROBE_VERSION,
    _validate_moe_block_contract,
    run_timestep_utility_probe,
)
from credit_redistribution.git_provenance import (
    repository_state,
    verify_worktree_source_manifest,
)


AUDIT_VERSION = 1
MANIFEST_NAME = "fresh_base_routing_audit_v1"
CANONICAL_MANIFEST_SHA256 = "41affd3a92f7c407fba33f894a10ee2392fc0cd25d105750c6dc095ea22a4824"
CANONICAL_CONFIG_SHA256 = "cf0e51098f1d6a09f6cfd45388aed99716bf020e7f11046d2cc0ae47b390e893"
CANONICAL_TRAINING_CONFIG_SHA256 = "c11983626dd8e65cf6074be4792c3f37a662acb01561537baca968a7db2ccca9"
MODEL_NAME = "ProMoE_TC_B"
CONFIG_STEM = "004_ProMoE_B_fresh_routing_audit_s0"
CHECKPOINT_STEPS = (50000, 100000, 150000, 200000)
PRIMARY_CHECKPOINT_STEP = 200000
CHECKPOINT_STATE = "ema_model_state_dict"
SPLIT_COUNTS = {"plumbing": 4, "discovery": 8, "confirmatory": 24}
LOCKED_DEVICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
LOCKED_VISIBLE_GPU_IDS = ("0", "1", "2", "3")
LOCKED_NUM_THREADS = 4
NUM_TOKEN_PROBES = 8
SENSITIVITY_TOKEN_COUNT = 2
EXACT_BATCH_SIZE = 24
CAPACITY_FACTOR = 1.25
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "analyses"
    / "fresh_base_routing"
    / "manifests"
    / "fresh_base_routing_audit_v1.json"
)
ARCHIVE_ROOT = PROJECT_ROOT / "analyses" / "archvied_analyses"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
# The experiment server keeps large run artifacts on a local staging disk and
# exposes them through a symlink below the repository's outputs/ directory.
# Keep this root explicit so an arbitrary symlink cannot silently enter the
# provenance record.
ALLOWED_EXTERNAL_OUTPUT_ROOTS = (Path("/home/dev/promoe-runs"),)
TRAINER_STATE_VERSION = 2
AUGMENTATION_SEED_VERSION = 1
DATASET_IDENTITY_VERSION = 1
EXPECTED_GLOBAL_SEED = 0
EXPECTED_GRAD_MIX = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{15,127}$")
_RUN_MARKER_PATTERN = re.compile(
    r"Fresh run marker: run_id=(?P<run_id>[A-Za-z0-9][A-Za-z0-9_-]{15,127}) "
    r"fresh=(?P<fresh>True|False) config=(?P<config>\S+) "
    r"output_dir=(?P<output_dir>\S+) global_seed=(?P<global_seed>[0-9]+) "
    r"world_size=(?P<world_size>[0-9]+) "
    r"launch_sha256=(?P<launch_sha256>[0-9a-f]{64})"
)
_TRAINING_PROVENANCE_PATTERN = re.compile(
    r"Training provenance: "
    r"run_id=(?P<run_id>[A-Za-z0-9][A-Za-z0-9_-]{15,127}) "
    r"launch_sha256=(?P<launch_sha256>[0-9a-f]{64}) "
    r"git_commit=(?P<git_commit>[0-9a-f]{40,64}) "
    r"config_sha256=(?P<config_sha256>[0-9a-f]{64})"
)
_CHECKPOINT_MARKER_PATTERN = re.compile(
    r"Checkpoint saved at (?P<path>\S+) "
    r"run_id=(?P<run_id>[A-Za-z0-9][A-Za-z0-9_-]{15,127}) "
    r"step=(?P<step>[0-9]+) size=(?P<size>[0-9]+) "
    r"sha256=(?P<sha256>[0-9a-f]{64}) "
    r"launch_sha256=(?P<launch_sha256>[0-9a-f]{64})"
)
_NO_CHECKPOINTS_PATTERN = re.compile(
    r"No checkpoints found in directory:\s*(?P<path>\S+)"
)
_TRAINING_SEED_PATTERN = re.compile(
    r"Training RNG seed:\s*(?P<seed>[0-9]+)\s+"
    r"\(global_seed=(?P<global_seed>[0-9]+), rank=(?P<rank>[0-9]+), "
    r"world_size=(?P<world_size>[0-9]+)\)"
)
_LATENT_DATASET_TYPES = {
    "__mp_main__.LatentFolder",
    "train.LatentFolder",
}
LOCKED_SOURCE_PATHS = (
    "requirements.txt",
    "config.py",
    "utils.py",
    "train.py",
    "models/models_ProMoE_TC.py",
    "models/modules.py",
    "models/phase_metric.py",
    "credit_redistribution/git_provenance.py",
    "analyses/denoising_regret/io.py",
    "analyses/denoising_regret/probe.py",
    "analyses/routing_translation/probe.py",
    "analyses/t_SNE/checkpoint_utils.py",
    "analyses/timestep_utility/batch.py",
    "analyses/timestep_utility/probe.py",
    "analyses/fresh_base_routing/audit.py",
    "analyses/run_fresh_base_routing_audit.py",
    "scripts/fresh_routing/run_B_fresh_routing_audit_s0_train_sample_eval.sh",
)
LOCKED_TRAINING_SOURCE_PATHS = (
    "requirements.txt",
    "config.py",
    "utils.py",
    "train.py",
    "models/models_ProMoE_TC.py",
    "models/modules.py",
    "models/phase_metric.py",
    "credit_redistribution/git_provenance.py",
)


def _parse_devices(value):
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if devices != LOCKED_DEVICES:
        raise argparse.ArgumentTypeError(
            "The fresh audit requires cuda:0,cuda:1,cuda:2,cuda:3"
        )
    return devices


def _json_sha256(payload):
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _absolute_path(path):
    """Normalize a path lexically without following its final symlink."""

    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path, root, *, allow_missing=True):
    """Reject symlink components between an archive root and a target path."""

    path = _absolute_path(path)
    root = _absolute_path(root)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes permitted root {root}: {path}") from error
    if root.is_symlink():
        raise ValueError(f"Permitted root must not be a symlink: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Permitted root is not a directory: {root}")

    current = root
    for component in relative.parts:
        current = current / component
        if not os.path.lexists(current):
            if allow_missing:
                break
            raise FileNotFoundError(f"Missing path component: {current}")
        if current.is_symlink():
            raise ValueError(f"Symlink path component is not allowed: {current}")
        if not current.is_dir() and current != path:
            raise NotADirectoryError(f"Path component is not a directory: {current}")
    return path


def _directory_open_flags():
    """Return flags that open a real directory without following a link."""

    flags = os.O_RDONLY
    for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC"):
        value = getattr(os, name, None)
        if value is None:
            raise OSError(f"The platform does not provide {name}")
        flags |= value
    return flags


def _open_secure_directory(path, root, *, create=True):
    """Open/create ``path`` below ``root`` without path-component races.

    Every component is created and then opened relative to the already-open
    parent descriptor.  Consequently, replacing a component with a symlink
    between the existence check and the open cannot redirect the traversal.
    The caller owns the returned descriptor.
    """

    path = _assert_no_symlink_components(path, root, allow_missing=create)
    root = _absolute_path(root)
    relative = path.relative_to(root)
    descriptor = None
    try:
        descriptor = os.open(root, _directory_open_flags())
        for index, component in enumerate(relative.parts):
            component_path = root.joinpath(*relative.parts[: index + 1])
            if create:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    # A concurrent worker may have created this component.  The
                    # O_NOFOLLOW|O_DIRECTORY open below is the authoritative check.
                    pass
            try:
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except OSError as error:
                if error.errno in (errno.ELOOP, errno.ENOTDIR):
                    try:
                        component_stat = os.stat(
                            component,
                            dir_fd=descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        component_stat = None
                    if component_stat is not None and stat.S_ISLNK(
                        component_stat.st_mode
                    ):
                        raise ValueError(
                            f"Symlink directory is not allowed: {component_path}"
                        ) from error
                if error.errno == errno.ENOENT:
                    raise FileNotFoundError(
                        f"Archive directory is missing: {component_path}"
                    ) from error
                if error.errno == errno.ENOTDIR:
                    raise NotADirectoryError(
                        f"Archive component is not a directory: {component_path}"
                    ) from error
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _mkdir_secure(path, root):
    """Create an archive directory one component at a time, rejecting links."""

    path = _absolute_path(path)
    descriptor = _open_secure_directory(path, root)
    os.close(descriptor)
    return path


def _archive_path(path, output_dir, *, allow_missing=True):
    """Validate a path beneath one already-selected audit output directory."""

    output_dir = _absolute_path(output_dir)
    path = _absolute_path(path)
    _assert_no_symlink_components(path, output_dir, allow_missing=allow_missing)
    return path


def _open_regular_file(name, parent_descriptor, description):
    """Open one regular file relative to a stable parent directory."""

    flags = os.O_RDONLY
    for flag_name in ("O_NOFOLLOW", "O_CLOEXEC"):
        flag = getattr(os, flag_name, None)
        if flag is None:
            raise OSError(f"The platform does not provide {flag_name}")
        flags |= flag
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"{description} must not be a symlink") from error
        raise
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"{description} must be a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            return handle.read()
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_archive_bytes(path, output_dir, description):
    """Read an archive file without following replaceable path components."""

    path = _archive_path(path, output_dir, allow_missing=False)
    parent_descriptor = _open_secure_directory(
        path.parent,
        _absolute_path(ARCHIVE_ROOT),
        create=False,
    )
    try:
        return _open_regular_file(path.name, parent_descriptor, description)
    finally:
        os.close(parent_descriptor)


def _read_archive_pair_bytes(payload_path, sidecar_path, output_dir, description):
    """Read two sibling files through one stable directory descriptor."""

    payload_path = _archive_path(payload_path, output_dir, allow_missing=False)
    sidecar_path = _archive_path(sidecar_path, output_dir, allow_missing=False)
    if payload_path.parent != sidecar_path.parent:
        raise ValueError(f"{description} files must be siblings")
    parent_descriptor = _open_secure_directory(
        payload_path.parent,
        _absolute_path(ARCHIVE_ROOT),
        create=False,
    )
    try:
        payload_bytes = _open_regular_file(
            payload_path.name,
            parent_descriptor,
            f"{description} primary file",
        )
        sidecar_bytes = _open_regular_file(
            sidecar_path.name,
            parent_descriptor,
            f"{description} sidecar",
        )
    finally:
        os.close(parent_descriptor)
    return payload_bytes, sidecar_bytes


def _read_archive_json_pair(payload_path, seal_path, output_dir, description):
    """Read a JSON payload and seal from one stable directory descriptor."""

    payload_bytes, seal_bytes = _read_archive_pair_bytes(
        payload_path,
        seal_path,
        output_dir,
        description,
    )
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
        seal = json.loads(seal_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict) or not isinstance(seal, dict):
        raise ValueError(f"{description} payload and seal must be JSON mappings")
    return payload, seal, hashlib.sha256(payload_bytes).hexdigest()


def _atomic_write(path, content, output_dir):
    """Write bytes atomically without following a pre-existing symlink."""

    path = _archive_path(path, output_dir, allow_missing=True)
    parent_descriptor = _open_secure_directory(
        path.parent,
        _absolute_path(ARCHIVE_ROOT),
    )
    temporary_name = None
    descriptor = None
    try:
        try:
            existing = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise ValueError(f"Destination must not be a symlink: {path}")

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        for _ in range(100):
            candidate = f".{path.name}.{uuid.uuid4().hex}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor is None or temporary_name is None:
            raise FileExistsError(f"Could not allocate temporary file for {path}")
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _write_json_atomic(path, payload, output_dir):
    content = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(path, content, output_dir)


def _write_text_atomic(path, content):
    path = Path(path)
    # The caller has already selected the protocol output directory.
    _atomic_write(path, content.encode("utf-8"), path.parent)


def _verify_file(path, expected_sha256, description):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{description} must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"{description} changed: {path}")


def _canonical_manifest_path(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    canonical_path = DEFAULT_MANIFEST.resolve()
    if manifest_path != canonical_path:
        raise ValueError(
            f"Fresh audit only accepts the canonical manifest: {canonical_path}"
        )
    return manifest_path


def _summary_payload(summary):
    """Return the stage-specific payload that is safe to print."""

    if "gate" in summary:
        return summary["gate"]
    if "decision" in summary:
        return summary["decision"]
    raise ValueError("Audit summary has neither a gate nor a decision")


def _load_yaml(path):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return payload


def _normalized_config_payload(payload):
    """Hide only machine-local paths and the permitted four-GPU half."""

    normalized = copy.deepcopy(payload)
    gpu_ids = normalized.get("gpu_ids")
    if gpu_ids not in ([0, 1, 2, 3], [4, 5, 6, 7]):
        raise ValueError(
            "Fresh Base config gpu_ids must be [0,1,2,3] or [4,5,6,7]"
        )
    latent_root = normalized.get("latent_data_path")
    if not isinstance(latent_root, str) or not latent_root:
        raise ValueError("Fresh Base config must specify latent_data_path")
    normalized["gpu_ids"] = ["<four-gpu-half>"]
    normalized["latent_data_path"] = "<latent-root>"
    return normalized


def _normalized_config_sha256(payload):
    return _json_sha256(_normalized_config_payload(payload))


def _training_config_payload_sha256(payload):
    """Match train.py's launch hash while ignoring only the stop boundary."""

    normalized = copy.deepcopy(payload)
    num_steps = normalized.get("num_steps")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("Fresh Base config num_steps must be a positive integer")
    normalized["num_steps"] = "<runtime-stop-boundary>"
    return _json_sha256(normalized)


def _dataset_identity_from_latent_root(latent_root, dataset_type):
    """Rebuild the sampler identity from the filesystem, never from its cache."""

    if dataset_type not in _LATENT_DATASET_TYPES:
        raise ValueError(f"Unsupported LatentFolder dataset type: {dataset_type!r}")
    root = Path(latent_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise NotADirectoryError(f"Latent root is not a real directory: {root}")

    class_entries = sorted(
        (
            entry
            for entry in os.scandir(root)
            if entry.is_dir(follow_symlinks=False)
        ),
        key=lambda entry: entry.name,
    )
    root_class_names = [entry.name for entry in class_entries]
    samples = []
    observed_class_names = set()
    for class_entry in class_entries:
        latent_entries = sorted(
            (
                entry
                for entry in os.scandir(class_entry.path)
                if entry.is_file(follow_symlinks=False)
                and entry.name.endswith(".latent.npz")
            ),
            key=lambda entry: entry.name,
        )
        if latent_entries:
            observed_class_names.add(class_entry.name)
            samples.extend(
                (Path(entry.path), class_entry.name) for entry in latent_entries
            )
    if not samples:
        raise ValueError(f"No latent samples found under {root}")

    # Keep the exact class-index semantics used by train.py without creating a
    # LatentFolder (which could consult the mutable process-local cache).
    from train import _build_latent_class_to_idx, _hash_dataset_record

    class_to_idx = _build_latent_class_to_idx(
        observed_class_names,
        root_class_names,
    )
    digest = hashlib.sha256()
    _hash_dataset_record(
        digest,
        DATASET_IDENTITY_VERSION,
        dataset_type,
        len(samples),
    )
    for path, class_name in samples:
        relative = os.path.relpath(
            os.path.normpath(os.fspath(path)),
            os.path.normpath(os.fspath(root)),
        )
        _hash_dataset_record(digest, relative, class_to_idx[class_name])
    return {
        "version": DATASET_IDENTITY_VERSION,
        "type": dataset_type,
        "num_samples": len(samples),
        "ordered_samples_sha256": digest.hexdigest(),
    }


def _expected_manifest_cases(selection, class_dirs):
    excluded = set(int(label) for label in selection["excluded_labels"])
    salt = selection["salt"]
    ranked = sorted(
        (
            hashlib.sha256(
                f"{salt}|class|{label:03d}|{class_dir.name}".encode()
            ).hexdigest(),
            label,
            class_dir,
        )
        for label, class_dir in enumerate(class_dirs)
        if label not in excluded
    )
    expected = []
    offset = 0
    for split in ("plumbing", "discovery", "confirmatory"):
        count = SPLIT_COUNTS[split]
        for _, label, class_dir in ranked[offset:offset + count]:
            latents = sorted(class_dir.glob("*.latent.npz"))
            if not latents:
                raise FileNotFoundError(f"No latents under {class_dir}")
            digest = hashlib.sha256(
                f"{salt}|latent|{label:03d}|{class_dir.name}".encode()
            ).hexdigest()
            latent = latents[int(digest[:8], 16) % len(latents)]
            seed = int(digest[8:16], 16) % 2147483647
            expected.append({
                "split": split,
                "id": (
                    f"class{label:03d}_"
                    f"{latent.name.removesuffix('.latent.npz')}"
                ),
                "label": label,
                "synset": class_dir.name,
                "latent": f"{class_dir.name}/{latent.name}",
                "seed": seed,
            })
        offset += count
    return expected


def load_manifest(manifest_path, latent_root):
    manifest_path = _canonical_manifest_path(manifest_path)
    latent_root = Path(latent_root).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest is missing: {manifest_path}")
    if not latent_root.is_dir():
        raise NotADirectoryError(f"Latent root is missing: {latent_root}")
    observed_manifest_sha256 = sha256_file(manifest_path)
    if observed_manifest_sha256 != CANONICAL_MANIFEST_SHA256:
        raise ValueError("Canonical fresh-audit manifest has an unexpected SHA256")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("name") != MANIFEST_NAME:
        raise ValueError("Manifest name or version is not canonical")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Manifest selection metadata is missing")
    if selection.get("locked_before_fresh_results") is not True:
        raise ValueError("Manifest was not locked before fresh results")
    if selection.get("split_counts") != SPLIT_COUNTS:
        raise ValueError("Manifest split counts differ from the audit")
    excluded_labels = selection.get("excluded_labels")
    if not isinstance(excluded_labels, list):
        raise ValueError("Manifest excluded_labels must be a list")
    if any(
        isinstance(label, bool)
        or not isinstance(label, int)
        or not 0 <= label < 1000
        for label in excluded_labels
    ):
        raise ValueError("Manifest excluded_labels must contain ImageNet labels")
    if excluded_labels != sorted(set(excluded_labels)):
        raise ValueError("Manifest excluded_labels must be sorted and unique")
    class_dirs = sorted(
        path
        for path in latent_root.iterdir()
        if path.is_dir()
        and len(path.name) == 9
        and path.name.startswith("n")
        and path.name[1:].isdigit()
    )
    if len(class_dirs) != 1000:
        raise ValueError(
            f"Expected 1000 ImageNet class directories, found {len(class_dirs)}"
        )
    expected = _expected_manifest_cases(selection, class_dirs)
    if payload.get("cases") != expected:
        raise ValueError("Manifest cases differ from deterministic selection")
    cases = []
    for case in expected:
        latent = (latent_root / case["latent"]).resolve()
        try:
            latent.relative_to(latent_root)
        except ValueError as error:
            raise ValueError(f"Case {case['id']} escapes latent root") from error
        if not latent.is_file():
            raise FileNotFoundError(f"Latent is missing for {case['id']}: {latent}")
        cases.append({
            **case,
            "latent_relative": case["latent"],
            "latent": str(latent),
            "latent_key": "latent",
            "latent_sha256": sha256_file(latent),
        })
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "latent_root": str(latent_root),
        "selection": selection,
        "cases": cases,
    }


def _verify_canonical_manifest_binding(protocol_manifest):
    """Re-load the repository manifest instead of trusting protocol metadata."""

    if not isinstance(protocol_manifest, dict):
        raise ValueError("Protocol manifest must be a mapping")
    canonical_path = _canonical_manifest_path(DEFAULT_MANIFEST)
    _verify_file(
        canonical_path,
        CANONICAL_MANIFEST_SHA256,
        "Canonical audit manifest",
    )
    protocol_path = protocol_manifest.get("path")
    if not isinstance(protocol_path, str) or Path(protocol_path).resolve() != canonical_path:
        raise ValueError("Protocol is not bound to the canonical manifest path")
    if protocol_manifest.get("sha256") != CANONICAL_MANIFEST_SHA256:
        raise ValueError("Protocol is not bound to the canonical manifest SHA256")
    latent_root = protocol_manifest.get("latent_root")
    if not isinstance(latent_root, str):
        raise ValueError("Protocol manifest latent_root is missing")
    canonical = load_manifest(canonical_path, latent_root)
    for key in ("path", "sha256", "latent_root", "selection", "cases"):
        if protocol_manifest.get(key) != canonical.get(key):
            raise ValueError(f"Protocol manifest differs from canonical data at {key}")
    return canonical


def _git_contract():
    state = repository_state(PROJECT_ROOT)
    commit = state["commit"]
    if state["status"]:
        raise RuntimeError("Prepare the audit only from a clean committed tree")
    divergence = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", "origin/repa...HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        divergence != "0\t0"
        or state["origin_repa"] != commit
        or state["authoritative_remote_tip"] != commit
    ):
        raise RuntimeError("Audit commit must already be pushed to origin/repa")
    return {"commit": commit, "origin_repa_divergence": divergence}


def _validate_training_provenance_contract(
    contract,
    *,
    expected_sha256=None,
    git_contract=None,
    environment=None,
):
    """Independently verify the launch evidence embedded by train.py."""

    if not isinstance(contract, dict) or set(contract) != {
        "version",
        "strict",
        "git",
        "config",
        "source_sha256",
        "environment",
    }:
        raise ValueError("Checkpoint training provenance fields are malformed")
    if contract["version"] != 1 or contract["strict"] is not True:
        raise ValueError("Checkpoint training provenance is not strict version 1")
    observed_sha256 = _json_sha256(contract)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("Checkpoint training provenance hash differs from the log")

    launch_git = contract["git"]
    if not isinstance(launch_git, dict) or set(launch_git) != {
        "commit",
        "origin_repa_commit",
        "status_clean",
        "origin_repa_divergence",
    }:
        raise ValueError("Checkpoint launch Git contract is malformed")
    commit = launch_git["commit"]
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None
        or launch_git["origin_repa_commit"] != commit
        or launch_git["status_clean"] is not True
        or launch_git["origin_repa_divergence"] != "0\t0"
    ):
        raise ValueError("Checkpoint launch Git contract was not clean and pushed")
    if git_contract is not None and (
        git_contract.get("commit") != commit
        or git_contract.get("origin_repa_divergence") != "0\t0"
    ):
        raise ValueError("Checkpoint launch commit differs from the audit commit")

    config_contract = contract["config"]
    if not isinstance(config_contract, dict) or set(config_contract) != {
        "version",
        "basename",
        "payload_sha256",
    }:
        raise ValueError("Checkpoint launch config contract is malformed")
    if (
        config_contract["version"] != 1
        or config_contract["basename"] != f"{CONFIG_STEM}.yaml"
        or config_contract["payload_sha256"]
        != CANONICAL_TRAINING_CONFIG_SHA256
    ):
        raise ValueError("Checkpoint launch config is not the canonical Base config")

    source_sha256 = contract["source_sha256"]
    if not isinstance(source_sha256, dict) or set(source_sha256) != set(
        LOCKED_TRAINING_SOURCE_PATHS
    ):
        raise ValueError("Checkpoint launch source hash set is not canonical")
    for relative in LOCKED_TRAINING_SOURCE_PATHS:
        digest = source_sha256[relative]
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError("Checkpoint launch source hash is malformed")
        _verify_file(
            PROJECT_ROOT / relative,
            digest,
            f"Training launch source {relative}",
        )
    if git_contract is not None:
        verify_worktree_source_manifest(
            PROJECT_ROOT,
            commit,
            source_sha256,
        )

    launch_environment = contract["environment"]
    if not isinstance(launch_environment, dict):
        raise ValueError("Checkpoint launch environment is malformed")
    if environment is not None and launch_environment != environment:
        raise ValueError(
            "Checkpoint launch environment differs from the audit environment"
        )
    return contract, observed_sha256


def _resolve_logged_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _fresh_training_log_snapshot(run_dir):
    run_dir = _absolute_path(run_dir)
    expected_run_dir = run_dir.resolve(strict=True)
    log_path = run_dir / "training.log"
    if log_path.is_symlink():
        raise ValueError(f"Fresh training log must not be a symlink: {log_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"Training log is missing: {log_path}")
    if log_path.resolve(strict=True) != expected_run_dir / "training.log":
        raise ValueError("Fresh training log is not bound to the current run directory")
    lines = log_path.read_text(encoding="utf-8").splitlines()
    expected_checkpoint_dir = expected_run_dir / "checkpoints"
    resume_marker = (
        "Resume progress: next_step=0, data_batches_seen=0, "
        "sampler_epoch=0, sampler_batch_offset=0"
    )

    # Every fresh marker is evidence about a distinct invocation.  We must not
    # silently select one complete marker while ignoring a trailing failed one.
    marker_records = []
    for index, line in enumerate(lines):
        if "Fresh run marker:" not in line:
            continue
        marker_match = _RUN_MARKER_PATTERN.search(line)
        if marker_match is None:
            raise ValueError(f"Malformed fresh run marker at line {index}")
        if marker_match["fresh"] == "True":
            marker_records.append((index, marker_match))
    if len(marker_records) != 1:
        raise ValueError(
            "Training log must contain exactly one fresh run marker; found "
            f"{len(marker_records)}"
        )
    marker_index, marker_match = marker_records[0]
    run_id = marker_match["run_id"]
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("Fresh run marker has a malformed run_id")
    launch_sha256 = marker_match["launch_sha256"]
    provenance_records = []
    for index in range(marker_index):
        if "Training provenance:" not in lines[index]:
            continue
        provenance_match = _TRAINING_PROVENANCE_PATTERN.search(lines[index])
        if provenance_match is None:
            raise ValueError(f"Malformed training provenance at line {index}")
        provenance_records.append((index, provenance_match))
    if len(provenance_records) != 1:
        raise ValueError(
            "Fresh run marker must have exactly one preceding training "
            f"provenance marker; found {len(provenance_records)}"
        )
    provenance_index, provenance_match = provenance_records[0]
    if (
        provenance_match["run_id"] != run_id
        or provenance_match["launch_sha256"] != launch_sha256
        or provenance_match["config_sha256"]
        != CANONICAL_TRAINING_CONFIG_SHA256
    ):
        raise ValueError("Training provenance marker differs from the fresh run")
    if marker_match["config"] != CONFIG_STEM:
        raise ValueError("Fresh run marker config does not match the audit")
    if marker_match["global_seed"] != "0" or marker_match["world_size"] != "4":
        raise ValueError("Fresh run marker seed/world size does not match the audit")
    if _resolve_logged_path(marker_match["output_dir"]) != expected_run_dir:
        raise ValueError("Fresh run marker output_dir does not match the run directory")

    seed_candidates = []
    for index in range(marker_index):
        seed_match = _TRAINING_SEED_PATTERN.search(lines[index])
        if seed_match is None:
            continue
        if (
            seed_match["seed"] == "0"
            and seed_match["global_seed"] == "0"
            and seed_match["rank"] == "0"
            and seed_match["world_size"] == "4"
        ):
            seed_candidates.append(index)
    if len(seed_candidates) != 1:
        raise ValueError(
            "Fresh training log must contain exactly one rank-0 seed marker "
            f"before the fresh marker; found {len(seed_candidates)}"
        )
    seed_index = seed_candidates[0]

    resume_indices = [
        index
        for index in range(marker_index + 1, len(lines))
        if resume_marker in lines[index]
    ]
    if len(resume_indices) != 1:
        raise ValueError(
            "Fresh run marker must be followed by exactly one step-0 resume "
            f"marker; found {len(resume_indices)}"
        )
    resume_index = resume_indices[0]
    between = lines[marker_index:resume_index + 1]
    if any(
        token in line
        for line in between
        for token in ("Loading checkpoint:", "Successfully loaded checkpoint")
    ):
        raise ValueError("Fresh run segment loads a checkpoint before step 0")

    empty_candidates = []
    for index in range(marker_index, resume_index + 1):
        empty_match = _NO_CHECKPOINTS_PATTERN.search(lines[index])
        if empty_match is None:
            continue
        if _resolve_logged_path(empty_match["path"]) != expected_checkpoint_dir:
            raise ValueError(
                "Fresh run log names a checkpoint directory other than the current run"
            )
        empty_candidates.append(index)
    if len(empty_candidates) != 1:
        raise ValueError(
            "Fresh run segment must contain exactly one empty checkpoint marker; "
            f"found {len(empty_candidates)}"
        )
    empty_index = empty_candidates[0]

    step_zero_candidates = [
        index
        for index in range(resume_index + 1, len(lines))
        if "epoch 0-step 0 " in lines[index]
    ]
    if len(step_zero_candidates) != 1:
        raise ValueError(
            "Fresh run segment must contain exactly one step-0 update; "
            f"found {len(step_zero_candidates)}"
        )
    step_zero_index = step_zero_candidates[0]
    pre_update = lines[marker_index:step_zero_index + 1]
    if any("Checkpoint saved at" in line for line in pre_update):
        raise ValueError("Fresh run saved a checkpoint before step 0")

    checkpoint_candidates = _checkpoint_marker_candidates(
        lines,
        run_id,
        expected_checkpoint_dir,
    )
    expected_marker_keys = {str(step) for step in CHECKPOINT_STEPS}
    if set(checkpoint_candidates) != expected_marker_keys:
        raise ValueError("Training log checkpoint marker set is not canonical")
    if any(len(matches) != 1 for matches in checkpoint_candidates.values()):
        raise ValueError(
            "Training log must contain exactly one fresh save marker for each "
            "locked checkpoint"
        )

    checkpoint_markers = {}
    previous_index = step_zero_index
    for step in CHECKPOINT_STEPS:
        expected_path = expected_checkpoint_dir / f"ckpt_step_{step}.pth"
        match_index, match = checkpoint_candidates[str(step)][0]
        if match_index <= previous_index:
            raise ValueError("Fresh checkpoint markers are not strictly ordered")
        if expected_path.is_symlink() or not expected_path.is_file():
            raise FileNotFoundError(f"Fresh checkpoint is missing: {expected_path}")
        observed_size = expected_path.stat().st_size
        logged_size = int(match["size"])
        if logged_size != observed_size:
            raise ValueError(
                f"Fresh checkpoint size disagrees with its log marker at step {step}"
            )
        observed_sha256 = sha256_file(expected_path)
        if match["sha256"] != observed_sha256:
            raise ValueError(
                f"Fresh checkpoint hash disagrees with its log marker at step {step}"
            )
        if match["launch_sha256"] != launch_sha256:
            raise ValueError(
                "Fresh checkpoint launch provenance differs from its run marker "
                f"at step {step}"
            )
        checkpoint_markers[str(step)] = {
            "line_index": match_index,
            "line": lines[match_index],
            "path": str(expected_path),
            "size": logged_size,
            "sha256": match["sha256"],
            "launch_sha256": match["launch_sha256"],
        }
        previous_index = match_index

    segment = "\n".join(lines[marker_index:step_zero_index + 1]) + "\n"
    return {
        "lexical_path": str(log_path),
        "path": str(log_path.resolve()),
        "run_dir": str(run_dir),
        "resolved_run_dir": str(expected_run_dir),
        "run_id": run_id,
        "training_provenance_sha256": launch_sha256,
        "training_provenance_line_index": provenance_index,
        "training_provenance_line": lines[provenance_index],
        "training_git_commit": provenance_match["git_commit"],
        "training_config_sha256": provenance_match["config_sha256"],
        "marker_line_index": marker_index,
        "marker_line": lines[marker_index],
        "resume_line_index": resume_index,
        "resume_line": lines[resume_index],
        "empty_directory_line_index": empty_index,
        "empty_directory_line": lines[empty_index],
        "seed_line_index": seed_index,
        "seed_line": lines[seed_index],
        "step_zero_line_index": step_zero_index,
        "step_zero_line": lines[step_zero_index],
        "segment_line_count": step_zero_index - marker_index + 1,
        "segment_sha256": hashlib.sha256(segment.encode("utf-8")).hexdigest(),
        "checkpoint_markers": checkpoint_markers,
    }


def _checkpoint_marker_candidates(lines, run_id, checkpoint_dir):
    """Collect every locked-step marker for one run, preserving line order."""

    checkpoint_dir = Path(checkpoint_dir)
    expected = {
        str(step): checkpoint_dir / f"ckpt_step_{step}.pth"
        for step in CHECKPOINT_STEPS
    }
    candidates = {key: [] for key in expected}
    for index, line in enumerate(lines):
        if "Checkpoint saved at" not in line:
            continue
        parsed = _CHECKPOINT_MARKER_PATTERN.search(line)
        if parsed is None or parsed["run_id"] != run_id:
            continue
        step_key = str(int(parsed["step"]))
        if step_key not in expected:
            continue
        if _resolve_logged_path(parsed["path"]) != expected[step_key].resolve():
            raise ValueError(
                f"Training log checkpoint path is not canonical at line {index}"
            )
        candidates[step_key].append((index, parsed))
    return candidates


def _verify_training_log(snapshot, checkpoints=None, run_dir=None):
    path = Path(snapshot["path"])
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Training log is missing: {path}")
    if run_dir is not None:
        expected_run_dir = _validate_run_dir(run_dir)
        if snapshot.get("run_dir") != str(expected_run_dir):
            raise RuntimeError("Training log snapshot is bound to another run path")
        if snapshot.get("resolved_run_dir") != str(expected_run_dir.resolve(strict=True)):
            raise RuntimeError("Training log snapshot resolved run path changed")
        expected_log = expected_run_dir / "training.log"
        if snapshot.get("lexical_path") != str(expected_log):
            raise RuntimeError("Training log snapshot lexical path changed")
        if path.resolve(strict=True) != expected_log.resolve(strict=True):
            raise RuntimeError("Training log is not bound to the current run")
    lines = path.read_text(encoding="utf-8").splitlines()
    marker_index = snapshot["marker_line_index"]
    step_zero_index = snapshot["step_zero_line_index"]
    if not 0 <= marker_index <= step_zero_index < len(lines):
        raise RuntimeError("Training log fresh marker indices are invalid")
    fresh_markers = []
    for index, line in enumerate(lines):
        if "Fresh run marker:" not in line:
            continue
        parsed_marker = _RUN_MARKER_PATTERN.search(line)
        if parsed_marker is None:
            raise RuntimeError(f"Malformed fresh run marker at line {index}")
        if parsed_marker["fresh"] == "True":
            fresh_markers.append(index)
    if fresh_markers != [marker_index]:
        raise RuntimeError("Training log contains an additional fresh run marker")
    segment = "\n".join(lines[marker_index:step_zero_index + 1]) + "\n"
    observed = hashlib.sha256(segment.encode("utf-8")).hexdigest()
    if observed != snapshot["segment_sha256"]:
        raise RuntimeError("Training log fresh-start segment changed")
    if lines[marker_index] != snapshot["marker_line"]:
        raise RuntimeError("Training log fresh run marker changed")
    marker_match = _RUN_MARKER_PATTERN.search(lines[marker_index])
    if marker_match is None or marker_match["fresh"] != "True":
        raise RuntimeError("Training log fresh run marker is invalid")
    if marker_match["run_id"] != snapshot["run_id"]:
        raise RuntimeError("Training log run_id changed")
    if marker_match["launch_sha256"] != snapshot.get(
        "training_provenance_sha256"
    ):
        raise RuntimeError("Training log launch provenance changed")
    if (
        marker_match["config"] != CONFIG_STEM
        or marker_match["global_seed"] != "0"
        or marker_match["world_size"] != "4"
        or _resolve_logged_path(marker_match["output_dir"])
        != Path(snapshot["resolved_run_dir"]).resolve()
    ):
        raise RuntimeError("Training log fresh marker is bound to another run")
    provenance_index = snapshot.get("training_provenance_line_index")
    provenance_indices = [
        index
        for index in range(marker_index)
        if "Training provenance:" in lines[index]
    ]
    if (
        isinstance(provenance_index, bool)
        or not isinstance(provenance_index, int)
        or not 0 <= provenance_index < marker_index
        or provenance_indices != [provenance_index]
        or lines[provenance_index] != snapshot.get("training_provenance_line")
    ):
        raise RuntimeError("Training log provenance marker changed")
    provenance_match = _TRAINING_PROVENANCE_PATTERN.search(lines[provenance_index])
    if provenance_match is None or (
        provenance_match["run_id"] != snapshot["run_id"]
        or provenance_match["launch_sha256"]
        != snapshot["training_provenance_sha256"]
        or provenance_match["git_commit"] != snapshot["training_git_commit"]
        or provenance_match["config_sha256"]
        != snapshot["training_config_sha256"]
    ):
        raise RuntimeError("Training log provenance marker is invalid")
    for key in ("resume_line_index", "empty_directory_line_index"):
        index = snapshot[key]
        if not marker_index <= index <= step_zero_index:
            raise RuntimeError(f"Training log index is out of range: {key}")
    seed_index = snapshot["seed_line_index"]
    if not 0 <= seed_index < marker_index:
        raise RuntimeError("Training log seed marker is out of range")
    if lines[snapshot["resume_line_index"]] != snapshot["resume_line"]:
        raise RuntimeError("Training log fresh sampler marker changed")
    if lines[snapshot["empty_directory_line_index"]] != snapshot["empty_directory_line"]:
        raise RuntimeError("Training log empty-directory evidence changed")
    if lines[snapshot["seed_line_index"]] != snapshot["seed_line"]:
        raise RuntimeError("Training log seed evidence changed")
    seed_match = _TRAINING_SEED_PATTERN.search(lines[seed_index])
    if seed_match is None or (
        seed_match["seed"],
        seed_match["global_seed"],
        seed_match["rank"],
        seed_match["world_size"],
    ) != ("0", "0", "0", "4"):
        raise RuntimeError("Training log seed evidence is invalid")
    empty_match = _NO_CHECKPOINTS_PATTERN.search(
        lines[snapshot["empty_directory_line_index"]]
    )
    if empty_match is None:
        raise RuntimeError("Training log empty-directory evidence is malformed")
    expected_checkpoint_dir = Path(snapshot["resolved_run_dir"]).resolve() / "checkpoints"
    if _resolve_logged_path(empty_match["path"]) != expected_checkpoint_dir:
        raise RuntimeError("Training log empty-directory path changed")
    if lines[step_zero_index] != snapshot["step_zero_line"]:
        raise RuntimeError("Training log step-0 evidence changed")

    marker_records = snapshot.get("checkpoint_markers")
    expected_marker_keys = {str(step) for step in CHECKPOINT_STEPS}
    if not isinstance(marker_records, dict) or set(marker_records) != expected_marker_keys:
        raise RuntimeError("Training log checkpoint marker set is not canonical")
    try:
        candidates = _checkpoint_marker_candidates(
            lines,
            snapshot["run_id"],
            expected_checkpoint_dir,
        )
    except (KeyError, ValueError) as error:
        raise RuntimeError(str(error)) from error
    if set(candidates) != expected_marker_keys or any(
        len(matches) != 1 for matches in candidates.values()
    ):
        raise RuntimeError(
            "Training log must contain exactly one fresh save marker for each "
            "locked checkpoint"
        )

    previous_index = step_zero_index
    for step in CHECKPOINT_STEPS:
        marker = marker_records[str(step)]
        candidate_index, candidate = candidates[str(step)][0]
        if candidate_index != marker.get("line_index"):
            raise RuntimeError(
                f"Training log checkpoint marker index changed for step {step}"
            )
        index = marker["line_index"]
        if index <= previous_index or index >= len(lines):
            raise RuntimeError(f"Training log checkpoint order changed for step {step}")
        if lines[index] != marker["line"]:
            raise RuntimeError(
                f"Training log checkpoint marker changed for step {step}"
            )
        parsed = _CHECKPOINT_MARKER_PATTERN.search(lines[index])
        if parsed is None or parsed["run_id"] != snapshot["run_id"]:
            raise RuntimeError(
                f"Training log checkpoint marker is invalid for step {step}"
            )
        if int(parsed["step"]) != int(step):
            raise RuntimeError(
                f"Training log checkpoint step changed for step {step}"
            )
        if _resolve_logged_path(parsed["path"]) != Path(marker["path"]).resolve():
            raise RuntimeError(
                f"Training log checkpoint path changed for step {step}"
            )
        if int(parsed["size"]) != marker["size"] or parsed["sha256"] != marker["sha256"]:
            raise RuntimeError(
                f"Training log checkpoint digest changed for step {step}"
            )
        if (
            parsed["launch_sha256"] != marker.get("launch_sha256")
            or parsed["launch_sha256"]
            != snapshot["training_provenance_sha256"]
        ):
            raise RuntimeError(
                f"Training log checkpoint provenance changed for step {step}"
            )
        if checkpoints is not None:
            checkpoint = checkpoints.get(str(step))
            if not isinstance(checkpoint, dict):
                raise RuntimeError(f"Protocol lacks checkpoint step {step}")
            if checkpoint.get("size") != marker["size"]:
                raise RuntimeError(f"Checkpoint size is not bound to log at step {step}")
            if checkpoint.get("sha256") != marker["sha256"]:
                raise RuntimeError(f"Checkpoint hash is not bound to log at step {step}")
            if checkpoint.get("run_id") != snapshot["run_id"]:
                raise RuntimeError(f"Checkpoint run_id is not bound at step {step}")
            trainer_contract = checkpoint.get("trainer_contract")
            if (
                not isinstance(trainer_contract, dict)
                or trainer_contract.get("training_provenance_sha256")
                != snapshot["training_provenance_sha256"]
            ):
                raise RuntimeError(
                    f"Checkpoint launch provenance is not bound at step {step}"
                )
        previous_index = index


def _trainer_state_contract(
    trainer_state,
    expected_step,
    expected_world_size,
    expected_global_seed,
    expected_total_batch_size,
    expected_run_id=None,
    expected_training_provenance_sha256=None,
):
    if not isinstance(trainer_state, dict):
        raise ValueError("Checkpoint trainer_state must be a mapping")
    if trainer_state.get("version") != TRAINER_STATE_VERSION:
        raise ValueError(
            f"Checkpoint trainer_state must use version {TRAINER_STATE_VERSION}"
        )
    if trainer_state.get("augmentation_seed_version") != AUGMENTATION_SEED_VERSION:
        raise ValueError("Checkpoint augmentation seed version is incompatible")
    if trainer_state.get("global_seed") != expected_global_seed:
        raise ValueError("Checkpoint global_seed differs from the fresh Base config")
    if trainer_state.get("world_size") != expected_world_size:
        raise ValueError("Checkpoint world size differs from the fresh Base config")
    if trainer_state.get("grad_mix") != EXPECTED_GRAD_MIX:
        raise ValueError("Checkpoint grad_mix differs from the fresh Base contract")
    run_id = trainer_state.get("run_id")
    if expected_run_id is not None:
        if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("Checkpoint run_id is missing or malformed")
        if run_id != expected_run_id:
            raise ValueError("Checkpoint run_id differs from the fresh log marker")
    training_provenance, training_provenance_sha256 = (
        _validate_training_provenance_contract(
            trainer_state.get("training_provenance"),
            expected_sha256=expected_training_provenance_sha256,
        )
    )

    next_step = trainer_state.get("next_step")
    data_batches_seen = trainer_state.get("data_batches_seen")
    sampler_epoch = trainer_state.get("sampler_epoch")
    sampler_batch_offset = trainer_state.get("sampler_batch_offset")
    integer_fields = (next_step, data_batches_seen, sampler_epoch, sampler_batch_offset)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_fields
    ):
        raise ValueError("Checkpoint trainer progress must be non-negative integers")
    if next_step != expected_step + 1:
        raise ValueError("Checkpoint next_step is inconsistent with its saved step")
    if data_batches_seen != next_step * EXPECTED_GRAD_MIX:
        raise ValueError("Checkpoint data_batches_seen is inconsistent with next_step")
    batches_per_epoch = trainer_state.get("batches_per_epoch")
    if (
        isinstance(batches_per_epoch, bool)
        or not isinstance(batches_per_epoch, int)
        or batches_per_epoch < 1
    ):
        raise ValueError("Checkpoint batches_per_epoch must be a positive integer")
    expected_epoch, expected_offset = divmod(data_batches_seen, batches_per_epoch)
    if (sampler_epoch, sampler_batch_offset) != (expected_epoch, expected_offset):
        raise ValueError("Checkpoint sampler position is internally inconsistent")

    sampler_contract = trainer_state.get("sampler_contract")
    if not isinstance(sampler_contract, dict):
        raise ValueError("Checkpoint sampler_contract is missing")
    if sampler_contract.get("version") != 1:
        raise ValueError("Checkpoint sampler_contract version is unsupported")
    if sampler_contract.get("type") != "distributed":
        raise ValueError("Fresh Base requires the distributed sampler contract")
    if sampler_contract.get("global_seed") != expected_global_seed:
        raise ValueError("Sampler contract global_seed is incompatible")
    per_rank_batch_size = sampler_contract.get("per_rank_batch_size")
    if (
        isinstance(per_rank_batch_size, bool)
        or not isinstance(per_rank_batch_size, int)
        or per_rank_batch_size < 1
        or per_rank_batch_size * expected_world_size != expected_total_batch_size
    ):
        raise ValueError("Sampler per-rank batch size is incompatible with the config")
    if sampler_contract.get("drop_last") is not False:
        raise ValueError("Fresh Base requires drop_last=False")
    if sampler_contract.get("case1_prob") is not None:
        raise ValueError("Fresh Base must use the ordinary distributed sampler")
    dataset = sampler_contract.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Sampler dataset contract is missing")
    if dataset.get("version") != DATASET_IDENTITY_VERSION:
        raise ValueError("Sampler dataset identity version is incompatible")
    if dataset.get("type") not in _LATENT_DATASET_TYPES:
        raise ValueError("Fresh Base requires the canonical LatentFolder dataset")
    if (
        isinstance(dataset.get("num_samples"), bool)
        or not isinstance(dataset.get("num_samples"), int)
        or dataset["num_samples"] < 1
    ):
        raise ValueError("Sampler dataset size is invalid")
    ordered_hash = dataset.get("ordered_samples_sha256")
    if not isinstance(ordered_hash, str) or not _SHA256_PATTERN.fullmatch(ordered_hash):
        raise ValueError("Sampler dataset order hash is invalid")

    rank_states = trainer_state.get("rank_states")
    if not isinstance(rank_states, list) or len(rank_states) != expected_world_size:
        raise ValueError("Checkpoint rank RNG states are incomplete")
    states_by_rank = {
        state.get("rank"): state
        for state in rank_states
        if isinstance(state, dict)
    }
    if set(states_by_rank) != set(range(expected_world_size)):
        raise ValueError("Checkpoint rank RNG state IDs are invalid")
    try:
        from train import _validate_rng_state

        for state in states_by_rank.values():
            rng_state = state.get("rng_state")
            if not isinstance(rng_state, dict) or set(rng_state) != {
                "python",
                "numpy",
                "torch",
                "cuda",
            }:
                raise ValueError("Checkpoint rank RNG states are incomplete")
            _validate_rng_state(rng_state)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Checkpoint rank RNG state is not recoverable") from error

    return {
        "trajectory": {
            "version": trainer_state["version"],
            "augmentation_seed_version": trainer_state["augmentation_seed_version"],
            "global_seed": trainer_state["global_seed"],
            "world_size": trainer_state["world_size"],
            "grad_mix": trainer_state["grad_mix"],
            "batches_per_epoch": batches_per_epoch,
            "sampler_contract": sampler_contract,
        },
        "run_id": run_id,
        "training_provenance": training_provenance,
        "training_provenance_sha256": training_provenance_sha256,
        "progress": {
            "next_step": next_step,
            "data_batches_seen": data_batches_seen,
            "sampler_epoch": sampler_epoch,
            "sampler_batch_offset": sampler_batch_offset,
        },
    }


def _optimizer_parameter_specs(runtime_cfg):
    """Build the locked model on ``meta`` and record trainable parameter contracts."""

    with torch.device("meta"):
        model = _build_model(runtime_cfg)
    try:
        specs = tuple(
            {
                "shape": tuple(parameter.shape),
                "dtype": parameter.dtype,
                "layout": parameter.layout,
            }
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    finally:
        del model
    if not specs:
        raise ValueError("Locked model has no trainable parameters")
    return specs


def _tensor_all_finite(value, chunk_size=1_048_576):
    """Check a contiguous tensor without allocating a tensor-sized mask."""

    if not value.is_contiguous():
        return False
    flattened = value.view(-1)
    return all(
        bool(torch.isfinite(flattened[start : start + chunk_size]).all().item())
        for start in range(0, flattened.numel(), chunk_size)
    )


def _optimizer_state_contract(optimizer_state, expected_step, parameter_specs):
    """Validate enough AdamW state to prove that a checkpoint is resumable."""

    if not isinstance(optimizer_state, dict):
        raise ValueError("optimizer_state_dict must be a mapping")
    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("optimizer_state_dict has no parameter groups")
    parameter_ids = []
    amsgrad_by_parameter = {}
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise ValueError("Optimizer parameter groups are malformed")
        amsgrad = group.get("amsgrad")
        if type(amsgrad) is not bool:
            raise ValueError("Optimizer amsgrad options are malformed")
        for parameter_id in group["params"]:
            if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
                raise ValueError("Optimizer parameter IDs are malformed")
            if parameter_id in amsgrad_by_parameter:
                raise ValueError("Optimizer parameter IDs are malformed")
            amsgrad_by_parameter[parameter_id] = amsgrad
            parameter_ids.append(parameter_id)
    if not parameter_ids or len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError("Optimizer parameter IDs are malformed")
    if len(parameter_ids) != len(parameter_specs):
        raise ValueError(
            "Optimizer parameter count differs from the locked model: "
            f"{len(parameter_ids)} != {len(parameter_specs)}"
        )

    state = optimizer_state.get("state")
    if not isinstance(state, dict) or set(state) != set(parameter_ids):
        raise ValueError("Optimizer state does not cover every parameter")
    expected_optimizer_step = expected_step + 1
    specs_by_parameter = dict(zip(parameter_ids, parameter_specs))
    for parameter_id, parameter_state in state.items():
        if not isinstance(parameter_state, dict):
            raise ValueError(
                f"Optimizer state for parameter {parameter_id} is malformed"
            )
        required = {"step", "exp_avg", "exp_avg_sq"}
        if amsgrad_by_parameter[parameter_id]:
            required.add("max_exp_avg_sq")
        if set(parameter_state) != required:
            raise ValueError(
                f"Optimizer state keys for parameter {parameter_id} are incomplete"
            )
        step_value = parameter_state["step"]
        if torch.is_tensor(step_value):
            if step_value.numel() != 1 or step_value.is_complex():
                raise ValueError("Optimizer step must be a real scalar")
            step_value = step_value.item()
        if (
            isinstance(step_value, bool)
            or not isinstance(step_value, (int, float))
            or not np.isfinite(float(step_value))
            or int(step_value) != step_value
            or int(step_value) != expected_optimizer_step
        ):
            raise ValueError(
                f"Optimizer step for parameter {parameter_id} is not "
                f"{expected_optimizer_step}"
            )
        parameter_spec = specs_by_parameter[parameter_id]
        for name in required - {"step"}:
            value = parameter_state[name]
            if (
                not torch.is_tensor(value)
                or value.numel() == 0
                or value.device.type != "cpu"
                or value.layout != parameter_spec["layout"]
                or tuple(value.shape) != parameter_spec["shape"]
                or value.dtype != parameter_spec["dtype"]
            ):
                raise ValueError(
                    f"Optimizer {name} for parameter {parameter_id} differs "
                    "from the locked model"
                )
            if not _tensor_all_finite(value):
                raise ValueError(
                    f"Optimizer {name} for parameter {parameter_id} is non-finite "
                    "or non-contiguous"
                )
    return {
        "num_parameter_groups": len(groups),
        "num_parameters": len(parameter_ids),
        "optimizer_step": expected_optimizer_step,
    }


def _checkpoint_contract(
    path,
    expected_step,
    expected_world_size,
    expected_global_seed,
    expected_total_batch_size,
    optimizer_parameter_specs,
    expected_run_id=None,
    expected_training_provenance_sha256=None,
):
    if parse_checkpoint_step(path) != expected_step:
        raise ValueError(f"Checkpoint filename does not encode step {expected_step}")
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        checkpoint = torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError):
        try:
            checkpoint = torch.load(path, **kwargs)
        except TypeError:
            kwargs.pop("weights_only")
            checkpoint = torch.load(path, **kwargs)
    try:
        if checkpoint.get("step") != expected_step:
            raise ValueError(f"Checkpoint payload step is not {expected_step}")
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint is missing model_state_dict")
        if CHECKPOINT_STATE not in checkpoint:
            raise KeyError(f"Checkpoint is missing {CHECKPOINT_STATE}")
        if "optimizer_state_dict" not in checkpoint:
            raise KeyError("Checkpoint is missing optimizer_state_dict")
        optimizer_contract = _optimizer_state_contract(
            checkpoint["optimizer_state_dict"],
            expected_step,
            optimizer_parameter_specs,
        )
        trainer_contract = _trainer_state_contract(
            checkpoint.get("trainer_state"),
            expected_step,
            expected_world_size,
            expected_global_seed,
            expected_total_batch_size,
            expected_run_id,
            expected_training_provenance_sha256,
        )
        return {
            **trainer_contract,
            "optimizer": optimizer_contract,
        }
    finally:
        del checkpoint
        gc.collect()


def _validate_config(config_path, latent_root=None):
    payload = _load_yaml(config_path)
    if _normalized_config_sha256(payload) != CANONICAL_CONFIG_SHA256:
        raise ValueError("Fresh Base config differs from the locked canonical config")
    if (
        _training_config_payload_sha256(payload)
        != CANONICAL_TRAINING_CONFIG_SHA256
    ):
        raise ValueError(
            "Fresh Base training config differs from the launch-time contract"
        )
    expected = {
        "model_name": MODEL_NAME,
        "total_train_batch_size": 256,
        "lr": 0.0001,
        "global_seed": 0,
        "img_num_workers": 16,
        "use_pre_latents": True,
        "use_encoded_latents": True,
        "resume_checkpoint": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Fresh Base config violates {key}: {payload.get(key)!r}")
    if payload.get("gpu_ids") not in ([0, 1, 2, 3], [4, 5, 6, 7]):
        raise ValueError(
            "Fresh Base config gpu_ids must be [0,1,2,3] or [4,5,6,7]"
        )
    configured_latent_root = payload["latent_data_path"]
    if latent_root is not None and Path(configured_latent_root).resolve() != Path(
        latent_root
    ).resolve():
        raise ValueError(
            "Fresh Base config latent_data_path differs from the audit latent root"
        )
    try:
        moe = payload["DiT_B_config"]["MoE_config"]
        phase = moe["phase_metric_config"]
    except (KeyError, TypeError) as error:
        raise ValueError("Fresh Base config lacks the required MoE settings") from error
    if phase.get("enabled") is not False:
        raise ValueError("Fresh Base unexpectedly enables Phase-Metric")
    if moe.get("top_k") != 1 or moe.get("router_weight_mode") != "identity":
        raise ValueError("Fresh Base router contract changed")
    if payload.get("num_steps", 0) < PRIMARY_CHECKPOINT_STEP + 1:
        raise ValueError("Fresh Base config ends before the primary checkpoint")
    return payload


def _source_contract(runtime_cfg):
    source_hashes = {}
    for relative in LOCKED_SOURCE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Locked source is missing: {path}")
        source_hashes[relative] = sha256_file(path)
    with torch.random.fork_rng(devices=[]):
        model = _build_model(runtime_cfg)
    try:
        block_contract = _validate_moe_block_contract(model, BLOCK_INDICES)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
    finally:
        del model
        gc.collect()
    return {
        "source_sha256": source_hashes,
        "model": {
            "parameter_count": parameter_count,
            "block_contract": block_contract,
        },
    }


def _runtime_environment(devices):
    if not torch.cuda.is_available():
        raise RuntimeError("The fresh audit requires CUDA")
    visible = tuple(
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    )
    if visible != LOCKED_VISIBLE_GPU_IDS:
        raise RuntimeError(
            "The fresh audit requires physical CUDA_VISIBLE_DEVICES=0,1,2,3"
        )
    if torch.cuda.device_count() != len(devices):
        raise RuntimeError(
            f"Expected {len(devices)} visible CUDA devices, "
            f"found {torch.cuda.device_count()}"
        )
    cuda_devices = {}
    for device in devices:
        properties = torch.cuda.get_device_properties(torch.device(device))
        uuid = getattr(properties, "uuid", None)
        if uuid is None:
            raise RuntimeError(f"CUDA device UUID is unavailable for {device}")
        cuda_devices[device] = {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "uuid": str(uuid),
        }
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "devices": list(devices),
        "cuda_visible_devices": list(visible),
        "cuda_devices": cuda_devices,
    }


def _verify_runtime_contract(protocol):
    observed_git = _git_contract()
    if observed_git != protocol["git"]:
        raise RuntimeError("Current Git contract differs from the prepared protocol")
    devices = tuple(protocol["settings"]["devices"])
    observed_environment = _runtime_environment(devices)
    if observed_environment != protocol["environment"]:
        raise RuntimeError(
            "Current Python/Torch/CUDA/GPU environment differs from the "
            "prepared protocol"
        )


def _output_dir(path):
    path = _absolute_path(path)
    archive_root = _absolute_path(ARCHIVE_ROOT)
    try:
        relative = path.relative_to(archive_root)
    except ValueError as error:
        raise ValueError(
            f"Audit output directory must stay under {ARCHIVE_ROOT}"
        ) from error
    if not relative.parts:
        raise ValueError("Audit output must be a child of the analysis archive")
    _assert_no_symlink_components(path, archive_root, allow_missing=True)
    return path


def _validate_run_dir(run_dir):
    run_dir = _absolute_path(run_dir)
    # Keep the lexical repository path in the protocol.  The path may be a
    # symlink, so resolving it before this check would reject the normal
    # runtime layout on the experiment server.
    try:
        run_dir.relative_to(_absolute_path(OUTPUT_ROOT))
    except ValueError as error:
        raise ValueError(
            f"Fresh Base run path must stay lexically under {OUTPUT_ROOT}"
        ) from error
    if run_dir.name != CONFIG_STEM or run_dir.parent.name != MODEL_NAME:
        raise ValueError(
            f"Expected run directory outputs/{MODEL_NAME}/{CONFIG_STEM}, got {run_dir}"
        )
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Fresh Base run directory is missing: {run_dir}")
    resolved = run_dir.resolve(strict=True)
    if resolved.name != CONFIG_STEM or resolved.parent.name != MODEL_NAME:
        raise ValueError(
            "Fresh Base run symlink target does not preserve the canonical run name"
        )
    lexical_output_root = _absolute_path(OUTPUT_ROOT)
    allowed_roots = []
    # If outputs/ itself is a link, its target must be registered explicitly.
    # Otherwise resolving it here would silently turn any target into an
    # allowlisted root and defeat ALLOWED_EXTERNAL_OUTPUT_ROOTS.
    if not lexical_output_root.is_symlink():
        if not lexical_output_root.is_dir():
            raise NotADirectoryError(
                f"Fresh Base output root is missing: {lexical_output_root}"
            )
        allowed_roots.append(lexical_output_root.resolve(strict=True))
    for root in ALLOWED_EXTERNAL_OUTPUT_ROOTS:
        root = _absolute_path(root)
        if root.is_symlink():
            raise ValueError(f"Registered output root must not be a symlink: {root}")
        if root.is_dir():
            allowed_roots.append(root.resolve(strict=True))
    if not any(
        resolved == root or root in resolved.parents
        for root in allowed_roots
    ):
        raise ValueError(
            "Fresh Base run symlink target is outside the registered output roots: "
            f"{resolved}"
        )
    return run_dir


def _checkpoint_path(run_dir, step):
    """Return a real checkpoint directly inside this run's checkpoint dir."""

    run_dir = _validate_run_dir(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise NotADirectoryError(
            f"Fresh checkpoint directory is not a real directory: {checkpoint_dir}"
        )
    path = checkpoint_dir / f"ckpt_step_{int(step)}.pth"
    if path.is_symlink():
        raise ValueError(f"Fresh checkpoint must not be a symlink: {path}")
    if path.resolve(strict=False).parent != checkpoint_dir.resolve(strict=True):
        raise ValueError(f"Fresh checkpoint escapes its checkpoint directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Fresh checkpoint is missing: {path}")
    return path


def _verify_output_dir_contract(output_dir, protocol):
    expected = _output_dir(protocol.get("output_dir", ""))
    observed = _output_dir(output_dir)
    if observed != expected:
        raise ValueError("CLI output directory differs from the sealed protocol")
    expected_resolved = expected.resolve(strict=False)
    if protocol.get("resolved_output_dir") != str(expected_resolved):
        raise ValueError("Sealed output directory resolution is inconsistent")


def _checkpoint_records(
    run_dir,
    config_payload,
    expected_run_id=None,
    expected_training_provenance_sha256=None,
):
    run_dir = _validate_run_dir(run_dir)
    records = {}
    config_path = None
    trajectory_contract = None
    expected_world_size = len(config_payload.get("gpu_ids", ()))
    expected_global_seed = config_payload.get("global_seed")
    expected_total_batch_size = config_payload.get("total_train_batch_size")
    if expected_world_size != 4:
        raise ValueError("Fresh Base config must reserve exactly four training GPUs")
    expected_config_path = (
        PROJECT_ROOT / "configs" / f"{CONFIG_STEM}.yaml"
    ).resolve()
    optimizer_parameter_specs = _optimizer_parameter_specs(
        load_runtime_cfg(expected_config_path)
    )
    for step in CHECKPOINT_STEPS:
        path = _checkpoint_path(run_dir, step)
        trainer_contract = _checkpoint_contract(
            path,
            step,
            expected_world_size,
            expected_global_seed,
            expected_total_batch_size,
            optimizer_parameter_specs,
            expected_run_id,
            expected_training_provenance_sha256,
        )
        if trajectory_contract is None:
            trajectory_contract = trainer_contract["trajectory"]
        elif trainer_contract["trajectory"] != trajectory_contract:
            raise ValueError(
                "Fresh checkpoints do not share one trainer/sampler trajectory contract"
            )
        resolved_config = resolve_config_from_checkpoint(path)
        if config_path is None:
            config_path = resolved_config
        elif resolved_config != config_path:
            raise ValueError("Fresh checkpoints resolve to different configs")
        records[str(step)] = {
            "path": str(path.resolve(strict=True)),
            "lexical_path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
            "state": CHECKPOINT_STATE,
            "run_id": trainer_contract["run_id"],
            "trainer_contract": trainer_contract,
        }
    if config_path.stem != CONFIG_STEM or config_path.resolve() != expected_config_path:
        raise ValueError(f"Expected config {CONFIG_STEM}, got {config_path.stem}")
    return records, config_path, trajectory_contract


def _build_protocol(args, manifest, output_dir):
    run_dir = _validate_run_dir(args.run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Fresh Base run directory is missing: {run_dir}")
    first_checkpoint = _checkpoint_path(run_dir, CHECKPOINT_STEPS[0])
    config_path = resolve_config_from_checkpoint(first_checkpoint)
    if config_path.is_symlink():
        raise ValueError(f"Fresh Base config must not be a symlink: {config_path}")
    config_payload = _validate_config(config_path, args.latent_root)
    fresh_training_log = _fresh_training_log_snapshot(run_dir)
    checkpoints, config_path, trajectory_contract = _checkpoint_records(
        run_dir,
        config_payload,
        expected_run_id=fresh_training_log["run_id"],
        expected_training_provenance_sha256=fresh_training_log[
            "training_provenance_sha256"
        ],
    )
    _verify_training_log(fresh_training_log, checkpoints)
    runtime_cfg = load_runtime_cfg(config_path)
    dataset_identity = trajectory_contract["sampler_contract"]["dataset"]
    observed_dataset_identity = _dataset_identity_from_latent_root(
        args.latent_root,
        dataset_identity["type"],
    )
    if observed_dataset_identity != dataset_identity:
        raise ValueError(
            "Current latent dataset identity differs from the fresh checkpoint"
        )
    source = _source_contract(runtime_cfg)
    git_contract = _git_contract()
    environment = _runtime_environment(args.devices)
    training_provenance = checkpoints[str(CHECKPOINT_STEPS[0])][
        "trainer_contract"
    ]["training_provenance"]
    _validate_training_provenance_contract(
        training_provenance,
        expected_sha256=fresh_training_log["training_provenance_sha256"],
        git_contract=git_contract,
        environment=environment,
    )
    if fresh_training_log["training_git_commit"] != git_contract["commit"]:
        raise ValueError("Training log launch commit differs from the audit commit")
    provenance_path = run_dir / "RUN_PROVENANCE.md"
    provenance = None
    if provenance_path.is_file():
        provenance = {
            "path": str(provenance_path.resolve()),
            "sha256": sha256_file(provenance_path),
        }
    return {
        "audit_version": AUDIT_VERSION,
        "probe_version": PROBE_VERSION,
        "locked_before_fresh_results": True,
        "hypothesis": (
            "Prototype affinity can disagree with downstream denoising utility, "
            "and exact expert-count-preserving reassignment can improve loss "
            "without changing routed-expert load or inference FLOPs."
        ),
        "scope": (
            "Problem-discovery audit only. Passing permits method design, not a "
            "training, FID, or publication claim."
        ),
        "run": {
            "path": str(run_dir),
            "resolved_path": str(run_dir.resolve(strict=True)),
            "fresh_training_log": fresh_training_log,
            "training_provenance": training_provenance,
            "provenance": provenance,
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
            "normalized_sha256": _normalized_config_sha256(config_payload),
            "training_payload_sha256": _training_config_payload_sha256(
                config_payload
            ),
            "stem": config_path.stem,
            "model_name": runtime_cfg.model_name,
        },
        "checkpoints": checkpoints,
        "trainer_trajectory_contract": trajectory_contract,
        "dataset_identity": observed_dataset_identity,
        "manifest": manifest,
        "settings": {
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "primary_checkpoint_step": PRIMARY_CHECKPOINT_STEP,
            "sigmas": list(SIGMAS),
            "block_indices": list(BLOCK_INDICES),
            "num_token_probes": NUM_TOKEN_PROBES,
            "sensitivity_token_count": SENSITIVITY_TOKEN_COUNT,
            "exact_batch_size": EXACT_BATCH_SIZE,
            "capacity_factor": CAPACITY_FACTOR,
            "split_counts": SPLIT_COUNTS,
            "devices": list(args.devices),
            "num_threads_per_worker": LOCKED_NUM_THREADS,
            "selection_weight_rule": (
                "Every candidate expert keeps the token's native top-1 route "
                "weight; expert identity and output scale are not conflated."
            ),
            "load_rule": (
                "native_capacity_oracle must preserve the complete sampled-token "
                "expert-count vector exactly."
            ),
        },
        "source": source,
        "git": git_contract,
        "environment": environment,
        "output_dir": str(output_dir),
        "resolved_output_dir": str(output_dir.resolve(strict=False)),
    }


def _protocol_paths(output_dir):
    output_dir = _absolute_path(output_dir)
    return output_dir / "protocol.json", output_dir / "protocol.sha256"


def _read_protocol_pair(output_dir):
    """Read and decode the protocol pair without path replacement races."""

    protocol_path, sha_path = _protocol_paths(output_dir)
    protocol_bytes, sha_bytes = _read_archive_pair_bytes(
        protocol_path,
        sha_path,
        output_dir,
        "Audit protocol",
    )
    try:
        protocol = json.loads(protocol_bytes.decode("utf-8"))
        recorded_sha = sha_bytes.decode("ascii").strip()
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Audit protocol pair is not valid JSON/ASCII") from error
    if not isinstance(protocol, dict):
        raise ValueError("Audit protocol must be a JSON mapping")
    if _SHA256_PATTERN.fullmatch(recorded_sha) is None:
        raise ValueError("Audit protocol SHA256 sidecar is malformed")
    return protocol, recorded_sha


def _write_or_validate_protocol(output_dir, protocol):
    output_dir = _output_dir(output_dir)
    _mkdir_secure(output_dir, _absolute_path(ARCHIVE_ROOT))
    protocol_path, sha_path = _protocol_paths(output_dir)
    expected_sha = _json_sha256(protocol)
    if os.path.lexists(protocol_path) or os.path.lexists(sha_path):
        try:
            existing, recorded_sha = _read_protocol_pair(output_dir)
        except FileNotFoundError as error:
            raise RuntimeError(
                "Protocol JSON and SHA256 must exist together"
            ) from error

        if existing != protocol or recorded_sha != expected_sha:
            raise RuntimeError("Existing audit protocol is incompatible")
    else:
        _write_json_atomic(protocol_path, protocol, output_dir)
        _write_text_atomic(sha_path, expected_sha + "\n")
    observed, recorded_sha = _read_protocol_pair(output_dir)
    if observed != protocol or recorded_sha != expected_sha:
        raise RuntimeError("Protocol SHA256 sidecar changed")
    return protocol_path, expected_sha


def _load_protocol(output_dir):
    output_dir = _output_dir(output_dir)
    try:
        protocol, recorded_sha = _read_protocol_pair(output_dir)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Prepare the audit protocol before running a stage"
        ) from error

    expected_sha = _json_sha256(protocol)
    if recorded_sha != expected_sha:
        raise RuntimeError("Protocol SHA256 does not match protocol JSON")
    if protocol.get("audit_version") != AUDIT_VERSION:
        raise ValueError("Protocol audit version differs from current source")
    return protocol, expected_sha


def _verify_checkpoint_contracts(
    run_dir,
    config_payload,
    expected_run_id,
    expected_training_provenance_sha256,
    protocol_checkpoints,
    protocol_trajectory,
):
    """Rebuild checkpoint contracts from the files, then compare the protocol."""

    observed, resolved_config, observed_trajectory = _checkpoint_records(
        run_dir,
        config_payload,
        expected_run_id=expected_run_id,
        expected_training_provenance_sha256=(
            expected_training_provenance_sha256
        ),
    )
    if observed_trajectory != protocol_trajectory:
        raise RuntimeError(
            "Fresh checkpoint trajectory differs from the sealed protocol"
        )
    expected_steps = {str(step) for step in CHECKPOINT_STEPS}
    if set(observed) != expected_steps or set(protocol_checkpoints) != expected_steps:
        raise RuntimeError("Fresh checkpoint contract set is not canonical")
    fields = (
        "path",
        "lexical_path",
        "size",
        "mtime_ns",
        "sha256",
        "state",
        "run_id",
        "trainer_contract",
    )
    for step in CHECKPOINT_STEPS:
        observed_record = observed[str(step)]
        protocol_record = protocol_checkpoints[str(step)]
        if not isinstance(protocol_record, dict):
            raise RuntimeError(f"Protocol checkpoint {step} is malformed")
        if set(protocol_record) != set(fields):
            raise RuntimeError(f"Protocol checkpoint {step} fields are not canonical")
        if any(protocol_record[field] != observed_record[field] for field in fields):
            raise RuntimeError(
                f"Protocol checkpoint trainer contract differs at step {step}"
            )
    return observed, resolved_config, observed_trajectory


def _verify_rebuilt_protocol_contract(protocol, canonical_manifest, output_dir, run_dir):
    """Reject a protocol whose complete contract no longer matches the run."""

    rebuild_args = argparse.Namespace(
        run_dir=run_dir,
        latent_root=canonical_manifest["latent_root"],
        devices=tuple(protocol["settings"]["devices"]),
    )
    rebuilt_protocol = _build_protocol(rebuild_args, canonical_manifest, output_dir)
    if rebuilt_protocol != protocol:
        raise RuntimeError(
            "Prepared protocol differs from the current run/config/source contract"
        )
    return rebuilt_protocol


def _verify_protocol_inputs(protocol):
    _verify_runtime_contract(protocol)
    run_meta = protocol.get("run")
    if not isinstance(run_meta, dict):
        raise RuntimeError("Prepared protocol lacks the run contract")
    run_path = run_meta.get("path")
    if not isinstance(run_path, str):
        raise RuntimeError("Prepared protocol lacks the lexical run path")
    run_dir = _validate_run_dir(run_path)
    if str(run_dir) != run_path:
        raise RuntimeError("Prepared protocol run path is not canonical")
    if run_meta.get("resolved_path") != str(run_dir.resolve(strict=True)):
        raise RuntimeError("Prepared protocol resolved run path changed")

    output_dir = _output_dir(protocol.get("output_dir", ""))
    if protocol.get("resolved_output_dir") != str(output_dir.resolve(strict=False)):
        raise RuntimeError("Prepared protocol resolved output path changed")

    canonical_manifest = _verify_canonical_manifest_binding(
        protocol.get("manifest")
    )
    expected_steps = {str(step) for step in CHECKPOINT_STEPS}
    checkpoints = protocol.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != expected_steps:
        raise RuntimeError("Prepared protocol checkpoint set is not canonical")
    trajectory = protocol.get("trainer_trajectory_contract")
    if not isinstance(trajectory, dict):
        raise RuntimeError("Prepared protocol lacks the trainer trajectory contract")
    fresh_training_log = run_meta.get("fresh_training_log")
    if not isinstance(fresh_training_log, dict):
        raise RuntimeError("Prepared protocol lacks the fresh training log contract")
    training_provenance = run_meta.get("training_provenance")
    _validate_training_provenance_contract(
        training_provenance,
        expected_sha256=fresh_training_log.get("training_provenance_sha256"),
        git_contract=protocol.get("git"),
        environment=protocol.get("environment"),
    )
    if fresh_training_log.get("training_git_commit") != protocol["git"]["commit"]:
        raise RuntimeError("Training log launch commit differs from the protocol")
    for step in CHECKPOINT_STEPS:
        record = checkpoints.get(str(step))
        if not isinstance(record, dict):
            raise RuntimeError(f"Prepared protocol lacks checkpoint step {step}")
        trainer_contract = record.get("trainer_contract")
        if (
            not isinstance(trainer_contract, dict)
            or trainer_contract.get("trajectory") != trajectory
            or trainer_contract.get("run_id") != record.get("run_id")
            or trainer_contract.get("training_provenance")
            != training_provenance
            or trainer_contract.get("training_provenance_sha256")
            != fresh_training_log.get("training_provenance_sha256")
        ):
            raise RuntimeError(
                f"Checkpoint step {step} disagrees with the trainer trajectory contract"
            )
        if not _RUN_ID_PATTERN.fullmatch(record.get("run_id", "")):
            raise RuntimeError(f"Checkpoint step {step} has an invalid run_id")
    _verify_training_log(fresh_training_log, checkpoints, run_dir=run_dir)
    provenance = run_meta.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise RuntimeError("Run provenance contract is malformed")
        provenance_path = Path(provenance.get("path", ""))
        expected_provenance = run_dir.resolve(strict=True) / "RUN_PROVENANCE.md"
        if (
            provenance_path.is_symlink()
            or provenance_path.resolve(strict=False) != expected_provenance
        ):
            raise RuntimeError("Run provenance is not bound to the current run")
        _verify_file(
            provenance["path"], provenance["sha256"], "Run provenance"
        )
    config_meta = protocol.get("config")
    if not isinstance(config_meta, dict):
        raise RuntimeError("Prepared protocol lacks the config contract")
    config_path = Path(config_meta.get("path", ""))
    expected_config_path = (PROJECT_ROOT / "configs" / f"{CONFIG_STEM}.yaml").resolve()
    if config_path.is_symlink() or config_path.resolve(strict=False) != expected_config_path:
        raise RuntimeError("Prepared protocol config is not the canonical Base config")
    if config_meta.get("normalized_sha256") != CANONICAL_CONFIG_SHA256:
        raise RuntimeError("Prepared protocol normalized config hash changed")
    if (
        config_meta.get("training_payload_sha256")
        != CANONICAL_TRAINING_CONFIG_SHA256
    ):
        raise RuntimeError("Prepared protocol training config hash changed")
    _verify_file(
        config_meta["path"],
        config_meta["sha256"],
        "Fresh Base config",
    )
    config_payload = _validate_config(
        config_meta["path"],
        canonical_manifest["latent_root"],
    )
    if _normalized_config_sha256(config_payload) != config_meta["normalized_sha256"]:
        raise RuntimeError("Current Base config normalized hash differs")
    if (
        _training_config_payload_sha256(config_payload)
        != config_meta["training_payload_sha256"]
    ):
        raise RuntimeError("Current Base training config hash differs")
    observed_checkpoints, resolved_config, observed_trajectory = (
        _verify_checkpoint_contracts(
            run_dir,
            config_payload,
            fresh_training_log["run_id"],
            fresh_training_log["training_provenance_sha256"],
            checkpoints,
            trajectory,
        )
    )
    if resolved_config.resolve() != config_path.resolve():
        raise RuntimeError("Fresh checkpoints resolve to a different config")
    if observed_trajectory != trajectory:
        raise RuntimeError("Fresh checkpoint trajectory changed")
    if canonical_manifest != protocol["manifest"]:
        raise RuntimeError("Protocol manifest is not the current canonical manifest")
    source = protocol.get("source")
    source_hashes = source.get("source_sha256") if isinstance(source, dict) else None
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(LOCKED_SOURCE_PATHS):
        raise RuntimeError("Prepared protocol source hash set is not canonical")
    for relative, expected in source_hashes.items():
        if Path(relative).is_absolute() or _absolute_path(relative) != PROJECT_ROOT / relative:
            raise RuntimeError(f"Locked source path is not repository-relative: {relative}")
        _verify_file(PROJECT_ROOT / relative, expected, f"Locked source {relative}")
    trajectory_dataset = trajectory.get("sampler_contract", {}).get("dataset")
    dataset_identity = protocol.get("dataset_identity")
    if not isinstance(dataset_identity, dict) or dataset_identity != trajectory_dataset:
        raise RuntimeError("Prepared protocol dataset identity is inconsistent")
    observed_dataset = _dataset_identity_from_latent_root(
        canonical_manifest["latent_root"],
        dataset_identity.get("type"),
    )
    if observed_dataset != dataset_identity:
        raise RuntimeError("Current latent dataset identity differs from the protocol")
    for step, checkpoint in checkpoints.items():
        expected_path = _checkpoint_path(run_dir, int(step))
        path = Path(checkpoint.get("path", ""))
        if (
            checkpoint.get("lexical_path") != str(expected_path)
            or path.is_symlink()
            or path.resolve(strict=False) != expected_path.resolve(strict=True)
        ):
            raise RuntimeError(f"Checkpoint step {step} is not bound to the run")
        if resolve_config_from_checkpoint(path).resolve() != config_path:
            raise RuntimeError(f"Checkpoint step {step} resolves to another config")
        if path.stat().st_size != checkpoint["size"]:
            raise RuntimeError(f"Checkpoint size changed at step {step}")
        _verify_file(path, checkpoint["sha256"], f"Checkpoint step {step}")
    for case in protocol["manifest"]["cases"]:
        _verify_file(
            case["latent"], case["latent_sha256"], f"Latent {case['id']}"
        )

    # The JSON and its sidecar are both writable files.  Checking only their
    # mutual hash would accept an attacker who rewrites both together.  Rebuild
    # the complete contract from the current run and compare every field so a
    # stage can proceed only with the exact protocol that was prepared.
    _verify_rebuilt_protocol_contract(
        protocol,
        canonical_manifest,
        output_dir,
        run_dir,
    )


def _result_path(output_dir, stage, step, index, case):
    output_dir = _output_dir(output_dir)
    path = (
        output_dir
        / "cases"
        / stage
        / f"step{step}"
        / f"{index:02d}_{case['id']}.json"
    )
    return _archive_path(path, output_dir, allow_missing=True)


def _seal_path(payload_path):
    return payload_path.with_suffix(payload_path.suffix + ".seal.json")


def _result_seal(result_sha256, protocol_sha256, step, case):
    return {
        "version": 1,
        "payload_sha256": result_sha256,
        "protocol_sha256": protocol_sha256,
        "checkpoint_step": int(step),
        "case_id": case["id"],
        "latent_sha256": case["latent_sha256"],
    }


def _validate_case_result(result, protocol, protocol_sha256, step, case):
    checkpoint = protocol["checkpoints"][str(step)]
    expected = {
        "checkpoint": checkpoint["path"],
        "weights_checkpoint": checkpoint["path"],
        "checkpoint_step": int(step),
        "weights_checkpoint_step": int(step),
        "checkpoint_state": CHECKPOINT_STATE,
        "config": protocol["config"]["path"],
        "model_name": MODEL_NAME,
        "latent": case["latent"],
        "label": case["label"],
        "seed": case["seed"],
        "sigmas": list(SIGMAS),
        "block_indices": list(BLOCK_INDICES),
        "num_token_probes": NUM_TOKEN_PROBES,
        "sensitivity_token_count": SENSITIVITY_TOKEN_COUNT,
        "exact_batch_size": EXACT_BATCH_SIZE,
        "capacity_factor": CAPACITY_FACTOR,
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint["sha256"],
        "weights_checkpoint_sha256": checkpoint["sha256"],
        "config_sha256": protocol["config"]["sha256"],
        "latent_sha256": case["latent_sha256"],
        "batch_case": case,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"Case {case['id']} step {step} differs at {key}: "
                f"{result.get(key)!r} != {value!r}"
            )


def _load_sealed_result(path, protocol, protocol_sha256, step, case):
    seal_path = _seal_path(path)
    output_dir = _output_dir(protocol["output_dir"])
    result, seal, result_sha = _read_archive_json_pair(
        path,
        seal_path,
        output_dir,
        f"Published result {case['id']} step {step}",
    )
    _validate_case_result(result, protocol, protocol_sha256, step, case)
    expected = _result_seal(result_sha, protocol_sha256, step, case)
    if seal != expected:
        raise RuntimeError(f"Result seal is incompatible: {seal_path}")
    return result


def _worker(payload):
    device = payload["device"]
    torch.cuda.set_device(torch.device(device))
    completed = []
    for job in payload["jobs"]:
        step = job["step"]
        case = job["case"]
        result_path = Path(job["result_path"])
        seal_path = _seal_path(result_path)
        if result_path.is_symlink() or seal_path.is_symlink():
            raise RuntimeError("Published result paths must not be symlinks")
        if result_path.exists() and seal_path.exists():
            _load_sealed_result(
                result_path,
                payload["protocol"],
                payload["protocol_sha256"],
                step,
                case,
            )
            completed.append({"step": step, "case_id": case["id"], "reused": True})
            continue
        if os.path.lexists(seal_path) and not os.path.lexists(result_path):
            raise RuntimeError(f"Seal has no result: {seal_path}")
        if os.path.lexists(result_path):
            if result_path.is_symlink():
                raise RuntimeError("Published result path must not be a symlink")
            result_path.unlink()
        print(f"[{device}] step {step}: {case['id']}", flush=True)
        checkpoint = payload["protocol"]["checkpoints"][str(step)]
        result = run_timestep_utility_probe(
            checkpoint_path=checkpoint["path"],
            weights_checkpoint_path=checkpoint["path"],
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
            device=device,
            num_threads=LOCKED_NUM_THREADS,
        )
        result.update({
            "protocol_sha256": payload["protocol_sha256"],
            "checkpoint_sha256": checkpoint["sha256"],
            "weights_checkpoint_sha256": checkpoint["sha256"],
            "config_sha256": payload["protocol"]["config"]["sha256"],
            "latent_sha256": case["latent_sha256"],
            "batch_case": case,
        })
        _validate_case_result(
            result,
            payload["protocol"],
            payload["protocol_sha256"],
            step,
            case,
        )
        output_dir = _output_dir(payload["protocol"]["output_dir"])
        _write_json_atomic(result_path, result, output_dir)
        result_sha = hashlib.sha256(
            _read_archive_bytes(
                result_path,
                output_dir,
                f"Published result {case['id']} step {step}",
            )
        ).hexdigest()
        seal = _result_seal(
            result_sha,
            payload["protocol_sha256"],
            step,
            case,
        )
        _write_json_atomic(seal_path, seal, output_dir)
        _load_sealed_result(
            result_path,
            payload["protocol"],
            payload["protocol_sha256"],
            step,
            case,
        )
        completed.append({"step": step, "case_id": case["id"], "reused": False})
        gc.collect()
        torch.cuda.empty_cache()
    return {"device": device, "completed": completed}


def _plumbing_summary(results_by_step):
    if not isinstance(results_by_step, dict):
        raise ValueError("Plumbing results must be a mapping by checkpoint")
    normalized = {}
    for step, results in results_by_step.items():
        key = str(step)
        if key in normalized:
            raise ValueError(f"Plumbing checkpoint keys collide at {key}")
        normalized[key] = results
    expected_steps = {str(step) for step in CHECKPOINT_STEPS}
    if set(normalized) != expected_steps:
        raise ValueError("Plumbing results must cover exactly all locked checkpoints")

    by_step = {}
    all_passed = True
    for step in CHECKPOINT_STEPS:
        results = normalized[str(step)]
        if not results:
            raise ValueError(f"Plumbing checkpoint {step} has no case results")
        rows = [_case_metrics(result) for result in results]
        first_safety = rows[0].get("safety")
        safety_names = tuple(first_safety) if isinstance(first_safety, dict) else ()
        if not safety_names:
            raise ValueError(f"Plumbing checkpoint {step} has no safety metrics")
        for row in rows:
            safety = row.get("safety")
            if not isinstance(safety, dict) or tuple(safety) != safety_names:
                raise ValueError(
                    f"Plumbing checkpoint {step} has inconsistent safety metrics"
                )
            for name, value in safety.items():
                if isinstance(value, bool) or not np.isscalar(value):
                    raise ValueError(
                        f"Non-numeric plumbing safety metric: {name}"
                    )
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError(
                        f"Non-finite plumbing safety metric: {name}"
                    )
        max_safety = {
            name: max(float(row["safety"][name]) for row in rows)
            for name in safety_names
        }
        counts_match = all(
            row.get("native_capacity_counts_match") is True for row in rows
        )
        passed = bool(counts_match and all(value == 0.0 for value in max_safety.values()))
        by_step[str(step)] = {
            "num_cases": len(rows),
            "max_numerical_error": max_safety,
            "native_capacity_counts_match": counts_match,
            "passed": passed,
        }
        all_passed = all_passed and passed
    return {
        "passed": all_passed,
        "interpretation": (
            "Only route-override plumbing and exact no-op controls are exposed; "
            "plumbing does not authorize an efficacy claim."
        ),
        "by_checkpoint": by_step,
    }


def longitudinal_decision(gates):
    if not isinstance(gates, dict):
        raise ValueError("Longitudinal gates must be a mapping")
    normalized = {}
    required_boolean_fields = {
        "routing_accuracy_gap_passed",
        "stage_structure_passed",
        "safety_passed",
    }
    for step, gate in gates.items():
        key = str(step)
        if key in normalized:
            raise ValueError(f"Longitudinal checkpoint keys collide at {key}")
        if not isinstance(gate, dict):
            raise ValueError(f"Longitudinal gate {key} must be a mapping")
        missing = required_boolean_fields - set(gate)
        if missing:
            raise ValueError(
                f"Longitudinal gate {key} lacks fields: {sorted(missing)}"
            )
        invalid = sorted(
            field
            for field in required_boolean_fields
            if type(gate[field]) is not bool
        )
        if invalid:
            raise ValueError(
                f"Longitudinal gate {key} fields must be exact booleans: {invalid}"
            )
        normalized[key] = gate
    required = {str(step) for step in CHECKPOINT_STEPS}
    if set(normalized) != required:
        raise ValueError("Longitudinal gates do not cover all locked checkpoints")
    primary = normalized[str(PRIMARY_CHECKPOINT_STEP)]
    earlier = [
        normalized[str(step)]
        for step in CHECKPOINT_STEPS
        if step != PRIMARY_CHECKPOINT_STEP
    ]
    earlier_routing_passes = sum(
        gate["routing_accuracy_gap_passed"] for gate in earlier
    )
    earlier_stage_passes = sum(gate["stage_structure_passed"] for gate in earlier)
    safety_passed = all(gate["safety_passed"] for gate in normalized.values())
    routing_supported = bool(
        safety_passed
        and primary["routing_accuracy_gap_passed"]
        and earlier_routing_passes >= 2
    )
    stage_supported = bool(
        safety_passed
        and primary["stage_structure_passed"]
        and earlier_stage_passes >= 2
    )
    return {
        "safety_passed": safety_passed,
        "primary_checkpoint_step": PRIMARY_CHECKPOINT_STEP,
        "primary_routing_passed": primary["routing_accuracy_gap_passed"],
        "earlier_routing_passes": earlier_routing_passes,
        "routing_gap_supported": routing_supported,
        "primary_stage_passed": primary["stage_structure_passed"],
        "earlier_stage_passes": earlier_stage_passes,
        "phase_structure_supported": stage_supported,
        "authorize_next_stage": routing_supported,
        "interpretation": (
            "A pass establishes a reproducible problem on one fresh training "
            "trajectory. It authorizes method design only; a second seed and "
            "fresh 0-to-500K generation experiment remain mandatory."
        ),
    }


def _summary_path(output_dir, stage):
    return output_dir / f"{stage}-summary.json"


def _stage_cases(protocol, stage):
    if stage not in SPLIT_COUNTS:
        raise ValueError(f"Unsupported audit stage: {stage}")
    manifest = protocol.get("manifest")
    cases_payload = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(cases_payload, list):
        raise ValueError("Protocol manifest cases are missing")
    if any(not isinstance(case, dict) for case in cases_payload):
        raise ValueError("Protocol manifest cases must all be mappings")
    cases = [case for case in cases_payload if case.get("split") == stage]
    if len(cases) != SPLIT_COUNTS[stage]:
        raise ValueError(f"Protocol has the wrong {stage} case count")
    return cases


def _load_stage_results(output_dir, stage, protocol, protocol_sha256, cases):
    return {
        step: [
            _load_sealed_result(
                _result_path(output_dir, stage, step, index, case),
                protocol,
                protocol_sha256,
                step,
                case,
            )
            for index, case in enumerate(cases, start=1)
        ]
        for step in CHECKPOINT_STEPS
    }


def _build_stage_summary(stage, cases, results_by_step, protocol_sha256):
    common = {
        "audit_version": AUDIT_VERSION,
        "protocol_sha256": protocol_sha256,
        "stage": stage,
        "case_ids": [case["id"] for case in cases],
    }
    if stage == "plumbing":
        return {
            **common,
            "gate": _plumbing_summary(results_by_step),
        }
    gates = {
        str(step): aggregate_case_results(results_by_step[step], stage)
        for step in CHECKPOINT_STEPS
    }
    return {
        **common,
        "gates": gates,
        "decision": longitudinal_decision(gates),
    }


def _write_sealed_summary(output_dir, stage, summary, protocol_sha256):
    output_dir = _output_dir(output_dir)
    _mkdir_secure(output_dir, _absolute_path(ARCHIVE_ROOT))
    path = _summary_path(output_dir, stage)
    seal_path = _seal_path(path)
    _archive_path(path, output_dir, allow_missing=True)
    _archive_path(seal_path, output_dir, allow_missing=True)
    if os.path.lexists(path) or os.path.lexists(seal_path):
        try:
            existing, _, _ = _read_archive_json_pair(
                path,
                seal_path,
                output_dir,
                f"{stage} summary",
            )
        except (FileNotFoundError, ValueError) as error:
            raise RuntimeError(f"Summary pair is incomplete for {stage}") from error
        if existing != summary:
            raise RuntimeError(f"Existing {stage} summary differs")
    else:
        _write_json_atomic(path, summary, output_dir)
        seal = {
            "version": 1,
            "payload_sha256": hashlib.sha256(
                _read_archive_bytes(
                    path,
                    output_dir,
                    f"{stage} summary payload",
                )
            ).hexdigest(),
            "protocol_sha256": protocol_sha256,
            "stage": stage,
        }
        _write_json_atomic(seal_path, seal, output_dir)
    _, seal, payload_sha256 = _read_archive_json_pair(
        path,
        seal_path,
        output_dir,
        f"{stage} summary",
    )
    expected = {
        "version": 1,
        "payload_sha256": payload_sha256,
        "protocol_sha256": protocol_sha256,
        "stage": stage,
    }
    if seal != expected:
        raise RuntimeError(f"Summary seal differs for {stage}")


def _load_sealed_summary(output_dir, stage, protocol_sha256):
    output_dir = _output_dir(output_dir)
    path = _summary_path(output_dir, stage)
    seal_path = _seal_path(path)
    summary, seal, payload_sha256 = _read_archive_json_pair(
        path,
        seal_path,
        output_dir,
        f"Required {stage} summary",
    )
    expected = {
        "version": 1,
        "payload_sha256": payload_sha256,
        "protocol_sha256": protocol_sha256,
        "stage": stage,
    }
    if seal != expected:
        raise RuntimeError(f"Required {stage} summary seal differs")
    return summary


def _load_recomputed_stage_summary(
    output_dir,
    stage,
    protocol,
    protocol_sha256,
):
    """Recompute a prior-stage gate from every sealed case result."""

    published = _load_sealed_summary(output_dir, stage, protocol_sha256)
    cases = _stage_cases(protocol, stage)
    results_by_step = _load_stage_results(
        output_dir,
        stage,
        protocol,
        protocol_sha256,
        cases,
    )
    recomputed = _build_stage_summary(
        stage,
        cases,
        results_by_step,
        protocol_sha256,
    )
    if published != recomputed:
        raise RuntimeError(
            f"Published {stage} summary differs from sealed case-result recomputation"
        )
    return recomputed


@contextmanager
def _audit_lock(output_dir):
    output_dir = _output_dir(output_dir)
    _mkdir_secure(output_dir, _absolute_path(ARCHIVE_ROOT))
    lock_path = output_dir / ".fresh-base-routing-audit.lock"
    _archive_path(lock_path, output_dir, allow_missing=True)
    flags = os.O_RDWR | os.O_CREAT
    for flag_name in ("O_NOFOLLOW", "O_CLOEXEC"):
        flag = getattr(os, flag_name, None)
        if flag is None:
            raise OSError(f"The platform does not provide {flag_name}")
        flags |= flag
    output_descriptor = _open_secure_directory(
        output_dir,
        _absolute_path(ARCHIVE_ROOT),
        create=False,
    )
    descriptor = None
    try:
        try:
            descriptor = os.open(
                lock_path.name,
                flags,
                0o664,
                dir_fd=output_descriptor,
            )
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError(f"Audit lock must not be a symlink: {lock_path}") from error
            raise
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Audit lock must be a regular file: {lock_path}")
        with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
            descriptor = None
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(output_descriptor)


def _run_stage(args, protocol, protocol_sha256, output_dir):
    stage = args.stage
    if stage == "discovery":
        plumbing = _load_recomputed_stage_summary(
            output_dir,
            "plumbing",
            protocol,
            protocol_sha256,
        )
        if plumbing["gate"]["passed"] is not True:
            raise RuntimeError("Plumbing failed; discovery is forbidden")
    elif stage == "confirmatory":
        discovery = _load_recomputed_stage_summary(
            output_dir,
            "discovery",
            protocol,
            protocol_sha256,
        )
        if discovery["decision"]["authorize_next_stage"] is not True:
            raise RuntimeError("Discovery did not authorize confirmation")

    cases = _stage_cases(protocol, stage)
    jobs_by_device = {device: [] for device in args.devices}
    job_index = 0
    for step in CHECKPOINT_STEPS:
        for index, case in enumerate(cases, start=1):
            device = args.devices[job_index % len(args.devices)]
            jobs_by_device[device].append({
                "step": step,
                "case": case,
                "result_path": str(
                    _result_path(output_dir, stage, step, index, case)
                ),
            })
            job_index += 1
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(args.devices), mp_context=context
    ) as executor:
        futures = [
            executor.submit(_worker, {
                "device": device,
                "jobs": jobs,
                "protocol": protocol,
                "protocol_sha256": protocol_sha256,
            })
            for device, jobs in jobs_by_device.items()
        ]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)

    _verify_protocol_inputs(protocol)
    results_by_step = _load_stage_results(
        output_dir,
        stage,
        protocol,
        protocol_sha256,
        cases,
    )
    summary = _build_stage_summary(
        stage,
        cases,
        results_by_step,
        protocol_sha256,
    )
    _write_sealed_summary(output_dir, stage, summary, protocol_sha256)
    print(json.dumps(_summary_payload(summary), indent=2))
    print(f"Saved: {_summary_path(output_dir, stage)}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the sealed fresh-Base 50K/100K/150K/200K routing audit."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=LOCKED_DEVICES,
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "plumbing", "discovery", "confirmatory"),
        required=True,
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = _output_dir(args.output_dir)
    with _audit_lock(output_dir):
        if args.stage == "prepare":
            manifest = load_manifest(args.manifest, args.latent_root)
            protocol = _build_protocol(args, manifest, output_dir)
            protocol_path, protocol_sha256 = _write_or_validate_protocol(
                output_dir, protocol
            )
            print(f"Protocol: {protocol_path}")
            print(f"Protocol SHA256: {protocol_sha256}")
            return
        protocol, protocol_sha256 = _load_protocol(output_dir)
        _verify_output_dir_contract(output_dir, protocol)
        if list(args.devices) != protocol["settings"]["devices"]:
            raise ValueError("CLI devices differ from the sealed protocol")
        cli_run_dir = _validate_run_dir(args.run_dir)
        if str(cli_run_dir) != protocol["run"]["path"]:
            raise ValueError("CLI run directory lexical path differs from the sealed protocol")
        if str(cli_run_dir.resolve(strict=True)) != protocol["run"]["resolved_path"]:
            raise ValueError("CLI run directory differs from the sealed protocol")
        if Path(args.latent_root).resolve() != Path(
            protocol["manifest"]["latent_root"]
        ):
            raise ValueError("CLI latent root differs from the sealed protocol")
        canonical_cli_manifest = _canonical_manifest_path(args.manifest)
        if canonical_cli_manifest != Path(protocol["manifest"]["path"]).resolve():
            raise ValueError("CLI manifest differs from the sealed protocol")
        _verify_protocol_inputs(protocol)
        _run_stage(args, protocol, protocol_sha256, output_dir)


if __name__ == "__main__":
    main()
