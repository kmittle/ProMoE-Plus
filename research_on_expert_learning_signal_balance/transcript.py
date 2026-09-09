"""Restart-safe, per-rank input transcripts for sealed continuations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .protocol_lock import load_effective_protocol
from .serialization import sha256_file
from .serialization import (
    field_frame,
    int64_tensor,
    string_list_payload,
    tensor_payload,
)


TRANSCRIPT_VERSION = 1
REPLAY_VERSION = 1
REPLAY_LATENT_SCALE = 0.18215
REPLAY_NUM_CLASSES = 1000
FIELD_ORDER = (
    "relative_latent_paths",
    "original_labels",
    "latent_parameters",
    "realized_z",
    "sampled_u",
    "timestep",
    "sigma",
    "diffusion_noise",
    "noised_model_input",
    "denoising_target",
    "effective_labels",
)
LOCAL_RECORD_FIELDS = {
    "version",
    "step",
    "rank",
    "relative_latent_paths",
    "original_labels",
    "field_sha256",
    "step_digest",
    "record_digest",
    "chain_digest",
}
GLOBAL_RECORD_FIELDS = {
    "version",
    "step",
    "rank_step_digests",
    "rank_record_digests",
    "global_digest",
    "chain_digest",
}


def _dist_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _dist_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def _distributed_error(local_error, phase):
    errors = [None] * _dist_world_size()
    if dist.is_initialized():
        dist.all_gather_object(errors, local_error)
    else:
        errors[0] = local_error
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
    if failures:
        raise RuntimeError(f"{phase} failed; " + "; ".join(failures))


def normalize_relative_paths(paths, root):
    root = Path(root).resolve()
    normalized = []
    for path in paths:
        resolved = Path(path).resolve()
        if not resolved.is_file() or resolved.suffixes[-2:] != [".latent", ".npz"]:
            raise ValueError(f"Latent path is not an encoded latent file: {resolved}")
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Latent path is outside dataset root: {resolved}") from error
        normalized.append(relative.as_posix())
    return normalized


def _sha256_framed_field(name, payload):
    return hashlib.sha256(field_frame(name, payload)).hexdigest()


def persisted_identity_field_hashes(record):
    """Recompute the two raw fields retained in a persisted local record."""
    paths = record.get("relative_latent_paths")
    labels = record.get("original_labels")
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError("Persisted transcript paths are malformed")
    if not isinstance(labels, list) or any(
        isinstance(label, bool) or not isinstance(label, int) for label in labels
    ):
        raise ValueError("Persisted transcript labels are malformed")
    labels_tensor = int64_tensor(labels).reshape(-1)
    return {
        "relative_latent_paths": _sha256_framed_field(
            "relative_latent_paths", string_list_payload(paths)
        ),
        "original_labels": _sha256_framed_field(
            "original_labels", tensor_payload(labels_tensor)
        ),
    }


def build_step_record(step, rank, relative_paths, original_labels, tensors):
    if set(tensors) != set(FIELD_ORDER[2:]):
        missing = sorted(set(FIELD_ORDER[2:]) - set(tensors))
        extra = sorted(set(tensors) - set(FIELD_ORDER[2:]))
        raise ValueError(f"Transcript fields differ: missing={missing}, extra={extra}")
    if len(relative_paths) != int(original_labels.numel()):
        raise ValueError("Transcript paths and labels have different batch sizes")

    relative_paths = [str(path) for path in relative_paths]
    labels = int64_tensor(original_labels)
    if labels.ndim != 1:
        raise ValueError("Transcript labels must be a one-dimensional tensor")
    if len(relative_paths) != int(labels.numel()):
        raise ValueError("Transcript paths and labels have different batch sizes")
    payloads = {
        "relative_latent_paths": string_list_payload(relative_paths),
        "original_labels": tensor_payload(labels),
    }
    payloads.update({name: tensor_payload(tensors[name]) for name in FIELD_ORDER[2:]})
    digest = hashlib.sha256()
    digest.update(field_frame("transcript_version", str(TRANSCRIPT_VERSION).encode("ascii")))
    digest.update(field_frame("step", str(int(step)).encode("ascii")))
    digest.update(field_frame("rank", str(int(rank)).encode("ascii")))
    field_hashes = {}
    for name in FIELD_ORDER:
        framed = field_frame(name, payloads[name])
        digest.update(framed)
        field_hashes[name] = hashlib.sha256(framed).hexdigest()
    record = {
        "version": TRANSCRIPT_VERSION,
        "step": int(step),
        "rank": int(rank),
        "relative_latent_paths": list(relative_paths),
        "original_labels": labels.reshape(-1).tolist(),
        "field_sha256": field_hashes,
        "step_digest": digest.hexdigest(),
    }
    record["record_digest"] = persisted_record_digest(record)
    return record


def persisted_record_digest(record):
    """Digest the complete record content that remains in the JSONL ledger."""
    content = dict(record)
    content.pop("chain_digest", None)
    content.pop("record_digest", None)
    encoded = json.dumps(
        content, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _replay_cfg_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _replay_hash_dataset_samples(paths, labels, dataset_type, dataset_root):
    if len(paths) != len(labels):
        raise ValueError("Replay dataset paths and labels have different lengths")
    digest = hashlib.sha256()

    def update(value):
        encoded = str(value).encode("utf-8", errors="surrogateescape")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)

    update(1)
    update(dataset_type)
    update(len(paths))
    root = Path(dataset_root).resolve()
    for path, label in zip(paths, labels):
        normalized = os.path.normpath(path)
        if root is not None:
            normalized = os.path.relpath(normalized, root)
        update(normalized)
        update(label)
    return digest.hexdigest()


def _replay_load_latent_paths(dataset_root):
    """Read the existing cache without mutating it, or scan deterministically."""

    root = Path(dataset_root).resolve()
    cache_path = Path("preprocess/latent_paths_cache.txt").resolve()
    candidates = None
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        cached = cache_path.read_text(encoding="utf-8").splitlines()
        resolved = []
        seen = set()
        try:
            for item in cached:
                path = Path(item).resolve()
                path.relative_to(root)
                if (
                    path in seen
                    or path.suffixes[-2:] != [".latent", ".npz"]
                    or not path.is_file()
                ):
                    raise ValueError
                seen.add(path)
                resolved.append(str(path))
        except (OSError, ValueError):
            resolved = []
        if resolved and resolved == sorted(resolved):
            candidates = resolved

    if candidates is None:
        candidates = []
        for entry in os.scandir(root):
            if not entry.is_dir(follow_symlinks=False):
                continue
            with os.scandir(entry.path) as children:
                candidates.extend(
                    child.path
                    for child in children
                    if child.is_file(follow_symlinks=False)
                    and child.name.endswith(".latent.npz")
                )
        candidates.sort()
    if not candidates:
        raise RuntimeError(f"Replay dataset contains no latent files: {root}")
    return candidates


class _ReplayLatentDataset(torch.utils.data.Dataset):
    """Read latent files with the same keyed flip rule as LatentFolder."""

    def __init__(self, latent_root, paths, class_to_idx):
        self.latent_dir = str(Path(latent_root).resolve())
        self.latent_paths = list(paths)
        self.class_to_idx = dict(class_to_idx)

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 2:
            raise ValueError("Replay dataset requires seeded sampler indices")
        sample_index, augmentation_seed = index
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index < 0
            or isinstance(augmentation_seed, bool)
            or not isinstance(augmentation_seed, int)
            or augmentation_seed < 0
        ):
            raise ValueError("Replay dataset index is malformed")
        path = self.latent_paths[sample_index]
        class_name = os.path.basename(os.path.dirname(path))
        label = self.class_to_idx[class_name]
        key = "latent_flip" if bool(augmentation_seed & 1) else "latent"
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise KeyError(f"Latent key {key!r} is absent from {path}")
            value = np.array(archive[key], copy=True)
        return path, label, torch.from_numpy(value)


def _replay_class_mapping(paths, runtime_cfg):
    from train import _build_latent_class_to_idx

    observed = {
        os.path.basename(os.path.dirname(path))
        for path in paths
    }
    root = Path(_replay_cfg_value(runtime_cfg, "latent_data_path")).resolve()
    root_classes = [
        entry.name
        for entry in os.scandir(root)
        if entry.is_dir(follow_symlinks=False)
    ]
    return _build_latent_class_to_idx(observed, root_classes)


def _replay_dataset_and_loader(runtime_cfg, dataset_root, rank, epoch, start_batch):
    from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
    from train import ResumableBatchSampler

    paths = _replay_load_latent_paths(dataset_root)
    class_to_idx = _replay_class_mapping(paths, runtime_cfg)
    dataset = _ReplayLatentDataset(dataset_root, paths, class_to_idx)
    world_size = int(_replay_cfg_value(runtime_cfg, "world_size", 4))
    global_seed = int(_replay_cfg_value(runtime_cfg, "global_seed", 0))
    total_batch = int(_replay_cfg_value(runtime_cfg, "total_train_batch_size", 256))
    if world_size < 1 or total_batch < 1 or total_batch % world_size:
        raise ValueError("Replay training geometry is not divisible")
    per_rank_batch = total_batch // world_size
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=global_seed,
    )
    batch_sampler = BatchSampler(
        sampler,
        batch_size=per_rank_batch,
        drop_last=False,
    )
    resumable = ResumableBatchSampler(
        batch_sampler,
        seed=global_seed,
        rank=rank,
    )
    resumable.set_epoch(epoch)
    resumable.set_start_batch(start_batch)
    num_workers = int(_replay_cfg_value(runtime_cfg, "img_num_workers", 0))
    if num_workers < 0:
        raise ValueError("Replay worker count must be non-negative")
    loader_seed = global_seed * world_size + rank + 0x5EEDDA7A
    loader_generator = torch.Generator().manual_seed(loader_seed)
    kwargs = {
        "batch_sampler": resumable,
        "num_workers": num_workers,
        "pin_memory": True,
        "generator": loader_generator,
    }
    if num_workers:
        kwargs["prefetch_factor"] = int(
            _replay_cfg_value(runtime_cfg, "prefetch_factor", 2)
        )
        kwargs["persistent_workers"] = True
    return dataset, resumable, DataLoader(dataset, **kwargs)


def _replay_rng_state(checkpoint, rank, device):
    trainer = checkpoint.get("trainer_state")
    if not isinstance(trainer, dict):
        raise ValueError("Replay checkpoint lacks trainer_state")
    rank_states = trainer.get("rank_states")
    if not isinstance(rank_states, list) or len(rank_states) <= rank:
        raise ValueError("Replay checkpoint lacks the requested rank RNG state")
    state = rank_states[rank]
    if not isinstance(state, dict) or state.get("rank") != rank:
        raise ValueError("Replay checkpoint rank IDs are not contiguous")
    rng = state.get("rng_state")
    if not isinstance(rng, dict):
        raise ValueError("Replay checkpoint RNG state is malformed")
    if not {"python", "numpy", "torch"}.issubset(rng):
        raise ValueError("Replay checkpoint RNG state is incomplete")
    if not torch.is_tensor(rng.get("torch")):
        raise ValueError("Replay checkpoint lacks torch RNG state")
    if device.type == "cuda" and not torch.is_tensor(rng.get("cuda")):
        raise ValueError("Replay checkpoint lacks cuda RNG state")
    return rng


def _replay_set_rng_state(state, device):
    if not isinstance(state, dict):
        raise ValueError("Replay RNG state must be a mapping")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, dict):
        raise ValueError("Replay NumPy RNG state must be a mapping")
    vector = numpy_state["state"]
    if not torch.is_tensor(vector):
        raise ValueError("Replay NumPy RNG state vector must be a tensor")
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            vector.detach().cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"].detach().cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].detach().cpu(), device=device)


def _replay_get_rng_state(device):
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state().clone(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).clone()
    return state


def _replay_tensor_fields(batch, runtime_cfg, device):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    paths, labels, latent_parameters = batch
    labels = labels.to(device, non_blocking=True)
    latent_parameters = latent_parameters.to(device, non_blocking=True)
    weighting_scheme = _replay_cfg_value(runtime_cfg, "weighting_scheme", "logit_normal")
    u = torch.empty(0, device=device)
    if weighting_scheme == "logit_normal":
        u = torch.normal(
            mean=float(_replay_cfg_value(runtime_cfg, "logit_mean", 0.0)),
            std=float(_replay_cfg_value(runtime_cfg, "logit_std", 1.0)),
            size=(labels.shape[0],),
            device=device,
        )
        u = u * float(_replay_cfg_value(runtime_cfg, "sigmoid_scale", 1.0))
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(labels.shape[0],), device=device)
        mode_scale = float(_replay_cfg_value(runtime_cfg, "mode_scale", 1.29))
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    elif weighting_scheme == "uniform":
        u = torch.rand(size=(labels.shape[0],), device=device)
    else:
        raise ValueError(f"Unsupported replay weighting scheme: {weighting_scheme!r}")

    shift = float(_replay_cfg_value(runtime_cfg, "shift", 1.0))
    num_train_timesteps = int(
        _replay_cfg_value(runtime_cfg, "num_train_timesteps", 1000)
    )
    sigma = (shift * u / (1 + (shift - 1) * u)).to(dtype=torch.float32)
    timestep = (sigma * num_train_timesteps).to(dtype=torch.float32)
    sigma_for_model = sigma
    while sigma_for_model.ndim < 4:
        sigma_for_model = sigma_for_model.unsqueeze(-1)

    posterior = DiagonalGaussianDistribution(latent_parameters)
    realized_z = posterior.sample().mul_(REPLAY_LATENT_SCALE)
    realized_z = realized_z.unsqueeze(2)
    noise = torch.randn_like(realized_z)
    target = noise - realized_z
    sigma_scalar = sigma_for_model.squeeze()
    noised = (
        (1.0 - sigma_scalar).view(realized_z.shape[0], 1, 1, 1, 1) * realized_z
        + sigma_scalar.view(realized_z.shape[0], 1, 1, 1, 1) * noise
    )

    model_cfg = _replay_cfg_value(runtime_cfg, "DiT_B_config", {})
    dropout_prob = float(_replay_cfg_value(model_cfg, "class_dropout_prob", 0.1))
    if not math.isfinite(dropout_prob) or not 0.0 <= dropout_prob <= 1.0:
        raise ValueError("Replay class-dropout probability is invalid")
    if dropout_prob:
        drop_ids = torch.rand(labels.shape[0], device=device) < dropout_prob
        effective_labels = torch.where(
            drop_ids,
            torch.full_like(labels, REPLAY_NUM_CLASSES),
            labels,
        )
    else:
        effective_labels = labels

    return {
        "latent_parameters": latent_parameters,
        "realized_z": realized_z,
        "sampled_u": u,
        "timestep": timestep,
        "sigma": sigma_for_model,
        "diffusion_noise": noise,
        "noised_model_input": noised,
        "denoising_target": target,
        "effective_labels": effective_labels,
    }


def iter_replayed_local_records(
    checkpoint,
    runtime_cfg,
    dataset_root,
    start_step,
    final_step,
    rank,
    device,
):
    """Yield records reconstructed from the sealed checkpoint and input rules."""

    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Replay device must be CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for replaying a CUDA checkpoint")
    start_step = int(start_step)
    final_step = int(final_step)
    if final_step < start_step:
        raise ValueError("Replay step interval is empty")
    trainer = checkpoint.get("trainer_state")
    if not isinstance(trainer, dict):
        raise ValueError("Replay checkpoint trainer state is missing")
    if checkpoint.get("step") != start_step - 1:
        raise ValueError("Replay checkpoint step does not precede the transcript")
    if int(trainer.get("next_step", -1)) != start_step:
        raise ValueError("Replay checkpoint next_step differs from transcript start")
    if int(trainer.get("world_size", -1)) != int(
        _replay_cfg_value(runtime_cfg, "world_size", 4)
    ):
        raise ValueError("Replay checkpoint world size differs")
    if int(trainer.get("grad_mix", -1)) != 1:
        raise ValueError("Replay only supports grad_mix=1")
    sampler_contract = trainer.get("sampler_contract")
    if not isinstance(sampler_contract, dict):
        raise ValueError("Replay checkpoint sampler contract is missing")
    if set(sampler_contract) != {
        "version",
        "type",
        "global_seed",
        "per_rank_batch_size",
        "drop_last",
        "case1_prob",
        "dataset",
    }:
        raise ValueError("Replay checkpoint sampler contract fields differ")
    if sampler_contract.get("type") != "distributed":
        raise ValueError("Replay only supports the distributed sampler contract")
    if sampler_contract.get("global_seed") != int(
        _replay_cfg_value(runtime_cfg, "global_seed", 0)
    ):
        raise ValueError("Replay sampler seed differs from checkpoint")
    expected_batch_size = int(
        _replay_cfg_value(runtime_cfg, "total_train_batch_size", 256)
    ) // int(_replay_cfg_value(runtime_cfg, "world_size", 4))
    if sampler_contract.get("per_rank_batch_size") != expected_batch_size:
        raise ValueError("Replay per-rank batch size differs from checkpoint")
    if sampler_contract.get("drop_last") is not False:
        raise ValueError("Replay requires drop_last=False")
    dataset_contract = sampler_contract.get("dataset")
    if not isinstance(dataset_contract, dict) or set(dataset_contract) != {
        "version",
        "type",
        "num_samples",
        "ordered_samples_sha256",
    }:
        raise ValueError("Replay checkpoint dataset contract fields differ")
    configured_root = Path(
        _replay_cfg_value(runtime_cfg, "latent_data_path")
    ).resolve()
    if configured_root != Path(dataset_root).resolve():
        raise ValueError("Replay dataset roots differ")
    epoch = int(trainer.get("sampler_epoch", -1))
    start_batch = int(trainer.get("sampler_batch_offset", -1))
    if epoch < 0 or start_batch < 0:
        raise ValueError("Replay checkpoint sampler position is invalid")

    saved_rng_state = _replay_get_rng_state(device)
    _replay_set_rng_state(_replay_rng_state(checkpoint, rank, device), device)
    dataset = resumable = loader = None
    try:
        dataset, resumable, loader = _replay_dataset_and_loader(
            runtime_cfg,
            dataset_root,
            rank,
            epoch,
            start_batch,
        )
        expected_type = str(dataset_contract.get("type", ""))
        if not expected_type.endswith(".LatentFolder"):
            raise ValueError("Replay checkpoint dataset type is not LatentFolder")
        labels = [
            dataset.class_to_idx[os.path.basename(os.path.dirname(path))]
            for path in dataset.latent_paths
        ]
        observed_hash = _replay_hash_dataset_samples(
            dataset.latent_paths,
            labels,
            expected_type,
            dataset_root,
        )
        expected_hash = dataset_contract.get("ordered_samples_sha256")
        if observed_hash != expected_hash:
            raise RuntimeError("Replay dataset identity differs from checkpoint")

        iterator = iter(loader)
        step = start_step
        while step <= final_step:
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                start_batch = 0
                resumable.set_epoch(epoch)
                resumable.set_start_batch(0)
                iterator = iter(loader)
                batch = next(iterator)
            paths, labels, _ = batch
            tensors = _replay_tensor_fields(batch, runtime_cfg, device)
            yield build_step_record(
                step=step,
                rank=rank,
                relative_paths=normalize_relative_paths(paths, dataset_root),
                original_labels=labels,
                tensors=tensors,
            )
            step += 1
    finally:
        iterator = locals().get("iterator")
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        del loader, resumable, dataset
        _replay_set_rng_state(saved_rng_state, device)


def validate_replayed_local_records(actual_records, expected_records, branch, rank):
    """Compare every persisted field, including tensor-derived commitments."""

    previous_chain = "0" * 64
    actual_iterator = iter(actual_records)
    expected_iterator = iter(expected_records)
    index = 0
    while True:
        try:
            actual = next(actual_iterator)
        except StopIteration:
            actual = None
        try:
            expected = next(expected_iterator)
        except StopIteration:
            expected = None
        if actual is None or expected is None:
            if actual is not None or expected is not None:
                raise RuntimeError(
                    f"Replay transcript length differs for {branch} rank {rank}"
                )
            return index
        if set(actual) != LOCAL_RECORD_FIELDS:
            raise RuntimeError(
                f"Replay transcript fields differ for {branch} rank {rank}"
            )
        expected_chain = JsonlLedger._chain(previous_chain, expected)
        actual_without_chain = dict(actual)
        actual_without_chain.pop("chain_digest", None)
        if actual_without_chain != expected:
            raise RuntimeError(
                f"Replay tensor content differs for {branch} rank {rank} "
                f"at step {expected.get('step')}"
            )
        if actual.get("chain_digest") != expected_chain:
            raise RuntimeError(
                f"Replay chain differs for {branch} rank {rank} "
                f"at step {expected.get('step')}"
            )
        previous_chain = expected_chain
        index += 1


def validate_local_transcript_replay(
    artifact_root,
    branch,
    initial_checkpoint_path,
    runtime_cfg,
    dataset_root,
    start_step,
    final_step,
    device,
    expected_checkpoint_sha256=None,
):
    """Verify persisted local transcripts against deterministic input replay."""

    from .serialization import sha256_file

    configured_checkpoint_path = Path(initial_checkpoint_path)
    if configured_checkpoint_path.is_symlink():
        raise RuntimeError("Replay checkpoint is absent or indirect")
    checkpoint_path = configured_checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError("Replay checkpoint is absent or indirect")
    if expected_checkpoint_sha256 is not None:
        observed = sha256_file(checkpoint_path)
        if observed != expected_checkpoint_sha256:
            raise RuntimeError("Replay checkpoint hash differs from protocol")
    load_kwargs = {
        "map_location": "cpu",
        "weights_only": False,
        "mmap": True,
    }
    checkpoint = None
    try:
        try:
            checkpoint = torch.load(checkpoint_path, **load_kwargs)
        except TypeError:
            load_kwargs.pop("mmap")
            checkpoint = torch.load(checkpoint_path, **load_kwargs)
    except Exception:
        raise
    try:
        root = Path(artifact_root).resolve()
        for rank in range(int(_replay_cfg_value(runtime_cfg, "world_size", 4))):
            path = root / "transcripts" / branch / f"rank-{rank:02d}.jsonl"
            if path.is_symlink():
                raise RuntimeError(f"Replay transcript cannot be a symlink: {path}")
            actual = _iter_transcript_records(path, start_step)
            expected = iter_replayed_local_records(
                checkpoint,
                runtime_cfg,
                dataset_root,
                start_step,
                final_step,
                rank,
                device,
            )
            validate_replayed_local_records(actual, expected, branch, rank)
    finally:
        del checkpoint


def _iter_transcript_records(path, start_step):
    """Yield and validate a local JSONL stream without importing evaluator code."""

    path = Path(path).resolve()
    previous_chain = "0" * 64
    expected_step = int(start_step)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed transcript {path}:{line_number}"
                ) from error
            if record.get("step") != expected_step:
                raise ValueError(f"Transcript steps are not contiguous at {path}")
            expected_chain = JsonlLedger._chain(previous_chain, record)
            if record.get("chain_digest") != expected_chain:
                raise ValueError(f"Transcript chain mismatch at {path}:{expected_step}")
            yield record
            previous_chain = expected_chain
            expected_step += 1


def build_global_record(step, rank_records):
    ordered = sorted(rank_records, key=lambda record: record["rank"])
    if [record["rank"] for record in ordered] != list(range(len(ordered))):
        raise ValueError("Global transcript rank IDs are incomplete")
    digest = hashlib.sha256()
    digest.update(field_frame("transcript_version", str(TRANSCRIPT_VERSION).encode("ascii")))
    digest.update(field_frame("step", str(int(step)).encode("ascii")))
    for record in ordered:
        digest.update(field_frame("rank", str(record["rank"]).encode("ascii")))
        digest.update(field_frame("step_digest", bytes.fromhex(record["step_digest"])))
        digest.update(
            field_frame("record_digest", bytes.fromhex(record["record_digest"]))
        )
    return {
        "version": TRANSCRIPT_VERSION,
        "step": int(step),
        "rank_step_digests": [record["step_digest"] for record in ordered],
        "rank_record_digests": [record["record_digest"] for record in ordered],
        "global_digest": digest.hexdigest(),
    }


class JsonlLedger:
    def __init__(self, path, start_step, read_only=False):
        self.path = Path(path)
        self.start_step = int(start_step)
        self.read_only = bool(read_only)
        self.records = {}
        if self.path.exists():
            self._load()
        elif self.read_only:
            raise FileNotFoundError(f"Reference transcript is absent: {self.path}")

    def _load(self):
        previous_chain = "0" * 64
        expected_step = self.start_step
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed transcript {self.path}:{line_number}"
                    ) from error
                step = record.get("step")
                if step != expected_step or step in self.records:
                    raise ValueError(f"Transcript steps are not contiguous at {self.path}")
                expected_chain = self._chain(previous_chain, record)
                if record.get("chain_digest") != expected_chain:
                    raise ValueError(f"Transcript chain mismatch at {self.path}:{step}")
                self.records[step] = record
                previous_chain = expected_chain
                expected_step += 1

    @staticmethod
    def _chain(previous_chain, record):
        content = dict(record)
        content.pop("chain_digest", None)
        encoded = json.dumps(
            content, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(bytes.fromhex(previous_chain) + encoded).hexdigest()

    def get(self, step):
        return self.records.get(int(step))

    def append_or_verify(self, record):
        if self.read_only:
            raise RuntimeError("Cannot append to a read-only transcript")
        step = int(record["step"])
        previous = self.records.get(step - 1)
        if step == self.start_step:
            previous_chain = "0" * 64
        elif previous is None:
            raise ValueError(f"Transcript cannot skip directly to step {step}")
        else:
            previous_chain = previous["chain_digest"]
        candidate = dict(record)
        candidate["chain_digest"] = self._chain(previous_chain, candidate)
        existing = self.records.get(step)
        if existing is not None:
            if existing != candidate:
                raise RuntimeError(f"Replayed transcript differs at step {step}")
            return existing
        if self.records and step != max(self.records) + 1:
            raise ValueError("New transcript records must append contiguously")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            candidate, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n"
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            raise
        self.records[step] = candidate
        return candidate


def _comparable_local(record):
    return {
        key: record[key]
        for key in (
            "version",
            "step",
            "rank",
            "relative_latent_paths",
            "original_labels",
            "field_sha256",
            "step_digest",
            "record_digest",
        )
    }


def _comparable_global(record):
    return {
        key: record[key]
        for key in (
            "version",
            "step",
            "rank_step_digests",
            "rank_record_digests",
            "global_digest",
        )
    }


class TrainingInputTranscript:
    def __init__(
        self,
        artifact_root,
        branch,
        start_step,
        dataset_root,
        reference_artifact_root=None,
        reference_branch="measure_only_control",
    ):
        self.rank = _dist_rank()
        self.world_size = _dist_world_size()
        self.branch = str(branch)
        self.start_step = int(start_step)
        self.dataset_root = Path(dataset_root).resolve()
        base = Path(artifact_root).resolve() / "transcripts" / self.branch
        self.local = JsonlLedger(
            base / f"rank-{self.rank:02d}.jsonl", self.start_step
        )
        self.global_ledger = (
            JsonlLedger(base / "global.jsonl", self.start_step)
            if self.rank == 0
            else None
        )
        self.reference_local = None
        self.reference_global = None
        if reference_artifact_root is not None:
            reference = (
                Path(reference_artifact_root).resolve()
                / "transcripts"
                / str(reference_branch)
            )
            self.reference_local = JsonlLedger(
                reference / f"rank-{self.rank:02d}.jsonl",
                self.start_step,
                read_only=True,
            )
            if self.rank == 0:
                self.reference_global = JsonlLedger(
                    reference / "global.jsonl",
                    self.start_step,
                    read_only=True,
                )

    def record(self, step, paths, original_labels, tensors):
        local_record = None
        local_error = None
        try:
            relative_paths = normalize_relative_paths(paths, self.dataset_root)
            local_record = build_step_record(
                step=step,
                rank=self.rank,
                relative_paths=relative_paths,
                original_labels=original_labels,
                tensors=tensors,
            )
            if self.reference_local is not None:
                reference = self.reference_local.get(step)
                if reference is None:
                    raise RuntimeError(f"Reference transcript lacks step {step}")
                if _comparable_local(reference) != _comparable_local(local_record):
                    raise RuntimeError(f"Reference transcript mismatch at step {step}")
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _distributed_error(local_error, "local transcript construction")

        gathered = [None] * self.world_size
        if dist.is_initialized():
            dist.all_gather_object(gathered, local_record)
        else:
            gathered[0] = local_record
        global_record = build_global_record(step, gathered)
        global_error = None
        if self.rank == 0 and self.reference_global is not None:
            reference = self.reference_global.get(step)
            if reference is None:
                global_error = f"Reference global transcript lacks step {step}"
            elif _comparable_global(reference) != _comparable_global(global_record):
                global_error = f"Reference global transcript mismatch at step {step}"
        errors = [global_error]
        if dist.is_initialized():
            dist.broadcast_object_list(errors, src=0)
        if errors[0]:
            raise RuntimeError(errors[0])

        write_error = None
        try:
            self.local.append_or_verify(local_record)
            if self.rank == 0:
                self.global_ledger.append_or_verify(global_record)
        except Exception as error:
            write_error = f"{type(error).__name__}: {error}"
        _distributed_error(write_error, "transcript persistence")
        if dist.is_initialized():
            dist.barrier()
        return global_record["global_digest"]


class TranscriptOnlyRecorder:
    """Record sealed inputs without installing any MoE or gradient hook."""

    def __init__(self, model, runtime_cfg, recorder_cfg):
        if not isinstance(recorder_cfg, dict):
            recorder_cfg = dict(recorder_cfg)
        if not bool(recorder_cfg.get("enabled", False)):
            raise ValueError("Transcript-only recorder cannot start while disabled")
        self.model = model
        self.runtime_cfg = runtime_cfg
        self.cfg = dict(recorder_cfg)
        self.protocol = load_effective_protocol(
            self.cfg["preregister_v3_path"],
            self.cfg["preregister_v4_path"],
        )
        self.start_step = int(self.protocol["branches"]["start_step"])
        self.execution_mode = str(self.cfg.get("execution_mode"))
        expected_steps = {
            "deterministic_replay_baseline": 20,
            "throughput_baseline": 600,
        }
        update_total = expected_steps.get(self.execution_mode)
        if update_total is None:
            raise ValueError(
                f"Unknown transcript-only execution mode: {self.execution_mode}"
            )
        self.expected_update_total = update_total
        if _dist_world_size() != 4:
            raise ValueError("Transcript-only sealed runs require four ranks")
        if int(runtime_cfg.num_steps) != self.start_step + update_total:
            raise ValueError("Transcript-only run length differs from its sealed mode")
        source = self.protocol["source_anchor"]["training_facts"]
        exact = {
            "model_name": (str(runtime_cfg.model_name), "ProMoE_TC_B"),
            "global_batch_size": (
                int(runtime_cfg.total_train_batch_size),
                int(source["global_batch_size"]),
            ),
            "global_seed": (int(runtime_cfg.global_seed), 0),
            "grad_mix": (int(runtime_cfg.grad_mix), 1),
        }
        for name, (actual, expected) in exact.items():
            if actual != expected:
                raise ValueError(f"{name}={actual!r}, expected {expected!r}")
        if bool(runtime_cfg.use_gradient_checkpointing):
            raise ValueError("Transcript-only sealed runs forbid checkpointing")
        branch = str(self.cfg.get("branch", "measure_only_control"))
        if branch != "measure_only_control":
            raise ValueError("Transcript-only recorder uses the measure-only branch name")
        self.transcript = TrainingInputTranscript(
            artifact_root=self.cfg["artifact_root"],
            branch=branch,
            start_step=self.start_step,
            dataset_root=runtime_cfg.latent_data_path,
            reference_artifact_root=self.cfg.get("reference_artifact_root"),
        )
        self.current_step = None
        self.update_count = 0
        self._pending_optimizer_step = False
        self.effective_labels = None
        self._handle = model.y_embedder.register_forward_hook(self._capture_labels)

    @property
    def initial_checkpoint_path(self):
        configured = Path(self.cfg["initial_checkpoint_path"]).resolve()
        sealed = Path(self.protocol["checkpoint"]["frozen_path"]).resolve()
        if configured != sealed:
            raise ValueError("Transcript-only initial checkpoint path differs")
        return configured

    def verify_initial_checkpoint(self):
        error = None
        if _dist_rank() == 0:
            try:
                observed = sha256_file(self.initial_checkpoint_path)
                expected = self.protocol["checkpoint"]["sha256"]
                if observed != expected:
                    raise RuntimeError("Transcript-only frozen checkpoint hash differs")
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
        errors = [error]
        if dist.is_initialized():
            dist.broadcast_object_list(errors, src=0)
        if errors[0]:
            raise RuntimeError(errors[0])
        if dist.is_initialized():
            dist.barrier()

    def _capture_labels(self, module, inputs, output):
        del module, inputs
        if self.current_step is None:
            raise RuntimeError("Label embedding occurred outside a transcript step")
        if self.effective_labels is not None:
            raise RuntimeError("Label embedding ran more than once in one step")
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("Label embedder no longer returns labels")
        self.effective_labels = output[1].detach()
        return None

    def begin_step(self, step):
        step = int(step)
        expected = self.start_step + self.update_count
        if self.current_step is not None or self._pending_optimizer_step:
            raise RuntimeError("Previous transcript-only step is incomplete")
        if step != expected:
            raise ValueError(f"Transcript-only step {step} does not equal {expected}")
        self.current_step = step
        self.effective_labels = None

    def record_before_optimizer(self, transcript_inputs):
        if self.current_step is None or self.effective_labels is None:
            raise RuntimeError("Transcript-only step capture is incomplete")
        if self._pending_optimizer_step:
            raise RuntimeError("Transcript-only step was already recorded")
        preparation_error = None
        tensors = None
        try:
            if not isinstance(transcript_inputs, dict):
                raise TypeError("transcript_inputs must be a mapping")
            tensors = dict(transcript_inputs["tensors"])
            tensors["effective_labels"] = self.effective_labels
        except Exception as error:
            preparation_error = f"{type(error).__name__}: {error}"
        _distributed_error(preparation_error, "transcript-only preparation")

        record_error = None
        digest = None
        try:
            digest = self.transcript.record(
                step=self.current_step,
                paths=transcript_inputs["paths"],
                original_labels=transcript_inputs["original_labels"],
                tensors=tensors,
            )
        except Exception as error:
            record_error = f"{type(error).__name__}: {error}"
        _distributed_error(record_error, "transcript-only recording")
        self._pending_optimizer_step = True
        return digest

    def after_optimizer_step(self):
        if self.current_step is None or not self._pending_optimizer_step:
            raise RuntimeError("Transcript-only recorder has no pending step")
        self.update_count += 1
        self.current_step = None
        self.effective_labels = None
        self._pending_optimizer_step = False

    def close(self):
        if self.current_step is not None or self._pending_optimizer_step:
            raise RuntimeError("Cannot close an incomplete transcript-only step")
        if self.update_count != self.expected_update_total:
            raise RuntimeError("Transcript-only run did not complete its locked updates")
        self._handle.remove()
