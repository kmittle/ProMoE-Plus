"""Measure whether route IDs agree with independent patch semantics.

DINOv2 is used only as a frozen measuring instrument. No DINO feature enters
the diffusion model, router, or training loss in this diagnostic.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from credit_redistribution.git_provenance import (
    AUTHORITATIVE_REMOTE_URL,
    authoritative_remote_tip,
    fresh_worktree_status,
    git_output as provenance_git_output,
    reject_history_overrides,
    reject_index_overrides,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "manifests"
    / "dirty_route_capture_semantics_v2.json"
)
ROUTE_KEY_PATTERN = re.compile(
    r"^(?P<prefix>.+)_sigma(?P<sigma>[0-9]+(?:\.[0-9]+)?)_block(?P<block>[0-9]+)$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DINO_SOURCE_IGNORED_DIRECTORIES = frozenset({".git", "__pycache__"})
DINO_SOURCE_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})
DINO_ATTENTION_BACKEND = "torch_no_xformers"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_float32_array(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype("<f4"))
    identity = json.dumps(
        {"dtype": "<f4", "shape": list(canonical.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(identity)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _is_ignored_dino_source_path(relative_path: Path) -> bool:
    return bool(
        DINO_SOURCE_IGNORED_DIRECTORIES.intersection(relative_path.parts)
        or relative_path.suffix in DINO_SOURCE_IGNORED_SUFFIXES
    )


def _dino_source_files(source_path: str | Path) -> list[tuple[str, Path]]:
    source_path = Path(source_path).resolve()
    if source_path.is_symlink() or not source_path.is_dir():
        raise ValueError("DINO source path must be a real directory")
    files = []
    for path in source_path.rglob("*"):
        relative_path = path.relative_to(source_path)
        if _is_ignored_dino_source_path(relative_path):
            continue
        if path.is_symlink():
            raise ValueError(f"DINO source contains a symlink: {relative_path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"DINO source contains a special file: {relative_path}")
        files.append((relative_path.as_posix(), path))
    if not files:
        raise ValueError("DINO source tree contains no locked files")
    return sorted(files)


def sha256_source_tree(source_path: str | Path) -> str:
    digest = hashlib.sha256(b"promoe-dino-source-tree-v1\0")
    for relative_path, path in _dino_source_files(source_path):
        encoded_path = relative_path.encode("utf-8")
        size = path.stat().st_size
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(size.to_bytes(8, "big"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _ignore_dino_source_copy(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in DINO_SOURCE_IGNORED_DIRECTORIES
        or Path(name).suffix in DINO_SOURCE_IGNORED_SUFFIXES
    }


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _require_exact_keys(payload: dict, expected: set[str], description: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        observed = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(
            f"{description} keys differ from the locked protocol: {observed}"
        )


def _require_integer(value, description: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{description} must be an integer >= {minimum}")
    return value


def _require_finite_number(value, description: str, *, lower_bound=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{description} must be a finite number")
    if lower_bound is not None and value < lower_bound:
        raise ValueError(f"{description} must be >= {lower_bound}")
    return value


def _validate_manifest(manifest: dict) -> None:
    _require_exact_keys(
        manifest,
        {
            "protocol_version",
            "protocol_name",
            "scope",
            "question",
            "expected",
            "feature_extractor",
            "statistics",
            "gates",
        },
        "manifest",
    )
    if manifest["protocol_version"] != 2:
        raise ValueError("unsupported semantic-audit protocol version")
    for key in ("protocol_name", "scope", "question"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"manifest {key} must be a nonempty string")

    expected = manifest["expected"]
    _require_exact_keys(
        expected,
        {
            "route_ids_sha256",
            "capture_summary_sha256",
            "dino_state_dict_sha256",
            "dino_source_tree_sha256",
            "vae_config_sha256",
            "vae_state_dict_sha256",
            "route_prefix",
            "sample_count",
            "token_grid_size",
            "num_routed_experts",
            "blocks",
            "sigmas",
            "sample_latents",
        },
        "expected",
    )
    for key in (
        "route_ids_sha256",
        "capture_summary_sha256",
        "dino_state_dict_sha256",
        "dino_source_tree_sha256",
        "vae_config_sha256",
        "vae_state_dict_sha256",
    ):
        if not isinstance(expected[key], str) or SHA256_PATTERN.fullmatch(expected[key]) is None:
            raise ValueError(f"expected.{key} must be a lowercase SHA256 digest")
    if not isinstance(expected["route_prefix"], str) or not expected["route_prefix"]:
        raise ValueError("expected.route_prefix must be a nonempty string")
    sample_count = _require_integer(expected["sample_count"], "sample_count", minimum=2)
    del sample_count
    grid_size = _require_integer(expected["token_grid_size"], "token_grid_size")
    _require_integer(expected["num_routed_experts"], "num_routed_experts", minimum=2)
    blocks = expected["blocks"]
    if (
        not isinstance(blocks, list)
        or not blocks
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in blocks)
        or blocks != sorted(set(blocks))
    ):
        raise ValueError("expected.blocks must be sorted unique nonnegative integers")
    sigmas = expected["sigmas"]
    if not isinstance(sigmas, list) or not sigmas:
        raise ValueError("expected.sigmas must be a nonempty list")
    normalized_sigmas = [
        _require_finite_number(item, "sigma", lower_bound=0.0) for item in sigmas
    ]
    if normalized_sigmas != sorted(set(normalized_sigmas)):
        raise ValueError("expected.sigmas must be sorted and unique")
    sample_latents = expected["sample_latents"]
    if not isinstance(sample_latents, list) or len(sample_latents) != expected["sample_count"]:
        raise ValueError("expected.sample_latents must match sample_count")
    seen_latent_paths = set()
    for index, record in enumerate(sample_latents):
        _require_exact_keys(record, {"relative_path", "sha256"}, f"sample_latents[{index}]")
        relative_path = record["relative_path"]
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"sample_latents[{index}].relative_path must be nonempty")
        parsed_path = Path(relative_path)
        if (
            parsed_path.is_absolute()
            or len(parsed_path.parts) != 2
            or any(part in {"", ".", ".."} for part in parsed_path.parts)
            or relative_path in seen_latent_paths
        ):
            raise ValueError(f"sample_latents[{index}] has an invalid relative path")
        if not isinstance(record["sha256"], str) or SHA256_PATTERN.fullmatch(record["sha256"]) is None:
            raise ValueError(f"sample_latents[{index}].sha256 must be a lowercase SHA256")
        seen_latent_paths.add(relative_path)

    extractor = manifest["feature_extractor"]
    _require_exact_keys(
        extractor,
        {
            "encoder",
            "attention_backend",
            "dino_source_revision",
            "encoder_input_size",
            "latent_decode",
            "latent_key",
            "latent_shape",
            "device",
            "batch_size",
            "software_versions",
            "cross_image_token_stride",
            "route_separation_pair_count",
        },
        "feature_extractor",
    )
    if extractor["encoder"] != "dinov2-vit-b":
        raise ValueError("this protocol requires dinov2-vit-b")
    if extractor["attention_backend"] != DINO_ATTENTION_BACKEND:
        raise ValueError(
            f"this protocol requires {DINO_ATTENTION_BACKEND!r} attention"
        )
    if (
        not isinstance(extractor["dino_source_revision"], str)
        or GIT_COMMIT_PATTERN.fullmatch(extractor["dino_source_revision"]) is None
    ):
        raise ValueError("dino_source_revision must be a full lowercase Git commit")
    if extractor["encoder_input_size"] != 224:
        raise ValueError("this protocol requires a 224-pixel DINOv2 input")
    if extractor["latent_decode"] != "posterior_mode":
        raise ValueError("this protocol requires posterior_mode decoding")
    if not isinstance(extractor["latent_key"], str) or not extractor["latent_key"]:
        raise ValueError("feature_extractor.latent_key must be nonempty")
    if extractor["latent_shape"] != [8, 32, 32]:
        raise ValueError("this protocol requires [8, 32, 32] posterior parameters")
    if extractor["device"] != "cpu":
        raise ValueError("this protocol requires CPU feature extraction")
    _require_integer(extractor["batch_size"], "feature_extractor.batch_size")
    software_versions = extractor["software_versions"]
    _require_exact_keys(
        software_versions,
        {"python", "numpy", "torch", "torchvision", "diffusers", "timm"},
        "feature_extractor.software_versions",
    )
    if any(not isinstance(value, str) or not value for value in software_versions.values()):
        raise ValueError("all locked software versions must be nonempty strings")
    stride = _require_integer(
        extractor["cross_image_token_stride"],
        "cross_image_token_stride",
    )
    if stride > grid_size:
        raise ValueError("cross_image_token_stride exceeds the token grid")
    pair_count = _require_integer(
        extractor["route_separation_pair_count"],
        "route_separation_pair_count",
    )
    maximum_pairs = grid_size * grid_size * (grid_size * grid_size - 1) // 2
    if pair_count > maximum_pairs:
        raise ValueError("route_separation_pair_count exceeds all token pairs")

    statistics = manifest["statistics"]
    _require_exact_keys(
        statistics,
        {
            "control_resamples",
            "bootstrap_resamples",
            "seed",
            "image_is_the_independent_unit",
            "cross_image_bootstrap",
            "multiple_testing",
        },
        "statistics",
    )
    control_resamples = _require_integer(
        statistics["control_resamples"], "control_resamples", minimum=19
    )
    _require_integer(
        statistics["bootstrap_resamples"], "bootstrap_resamples", minimum=100
    )
    _require_integer(statistics["seed"], "statistics.seed", minimum=0)
    if statistics["image_is_the_independent_unit"] is not True:
        raise ValueError("image_is_the_independent_unit must be exactly true")
    if statistics["cross_image_bootstrap"] != (
        "two_way_query_gallery_image_cluster_with_control_draw"
    ):
        raise ValueError("unsupported cross-image bootstrap")
    if statistics["multiple_testing"] != (
        "holm_bonferroni_across_all_54_cell_metrics"
    ):
        raise ValueError("unsupported multiple-testing correction")

    gates = manifest["gates"]
    _require_exact_keys(
        gates,
        {
            "minimum_within_image_knn_delta",
            "minimum_cross_image_knn_delta",
            "minimum_route_separation_delta",
            "maximum_one_sided_control_p",
            "minimum_passing_cells",
            "minimum_passing_blocks",
            "minimum_passing_sigmas",
        },
        "gates",
    )
    for key in (
        "minimum_within_image_knn_delta",
        "minimum_cross_image_knn_delta",
        "minimum_route_separation_delta",
    ):
        _require_finite_number(gates[key], key, lower_bound=0.0)
    maximum_p = _require_finite_number(
        gates["maximum_one_sided_control_p"],
        "maximum_one_sided_control_p",
        lower_bound=0.0,
    )
    if not 0 < maximum_p <= 1:
        raise ValueError("maximum_one_sided_control_p must be in (0, 1]")
    if 1 / (control_resamples + 1) > maximum_p:
        raise ValueError("too few controls to reach maximum_one_sided_control_p")
    cell_count = len(blocks) * len(sigmas)
    family_test_count = cell_count * 3
    if 1 / (control_resamples + 1) > maximum_p / family_test_count:
        raise ValueError("too few controls for the locked Holm family")
    passing_cells = _require_integer(
        gates["minimum_passing_cells"], "minimum_passing_cells"
    )
    passing_blocks = _require_integer(
        gates["minimum_passing_blocks"], "minimum_passing_blocks"
    )
    passing_sigmas = _require_integer(
        gates["minimum_passing_sigmas"], "minimum_passing_sigmas"
    )
    if passing_cells > cell_count:
        raise ValueError("minimum_passing_cells exceeds the cell count")
    if passing_blocks > len(blocks):
        raise ValueError("minimum_passing_blocks exceeds the block count")
    if passing_sigmas > len(sigmas):
        raise ValueError("minimum_passing_sigmas exceeds the sigma count")


def load_manifest(path: str | Path) -> dict:
    manifest = _load_json(path)
    _validate_manifest(manifest)
    return manifest


def validate_locked_inputs(
    manifest: dict,
    *,
    route_ids_path: str | Path,
    capture_summary_path: str | Path,
    dino_path: str | Path,
    dino_source_path: str | Path,
    vae_path: str | Path,
) -> dict:
    expected = manifest["expected"]
    vae_path = Path(vae_path)
    records = {
        "route_ids": {
            "path": str(Path(route_ids_path).resolve()),
            "sha256": sha256_file(route_ids_path),
        },
        "capture_summary": {
            "path": str(Path(capture_summary_path).resolve()),
            "sha256": sha256_file(capture_summary_path),
        },
        "dino_state_dict": {
            "path": str(Path(dino_path).resolve()),
            "sha256": sha256_file(dino_path),
        },
        "dino_source_tree": {
            "path": str(Path(dino_source_path).resolve()),
            "sha256": sha256_source_tree(dino_source_path),
        },
        "vae_config": {
            "path": str((vae_path / "config.json").resolve()),
            "sha256": sha256_file(vae_path / "config.json"),
        },
        "vae_state_dict": {
            "path": str(
                (vae_path / "diffusion_pytorch_model.safetensors").resolve()
            ),
            "sha256": sha256_file(
                vae_path / "diffusion_pytorch_model.safetensors"
            ),
        },
    }
    bindings = {
        "route_ids": "route_ids_sha256",
        "capture_summary": "capture_summary_sha256",
        "dino_state_dict": "dino_state_dict_sha256",
        "dino_source_tree": "dino_source_tree_sha256",
        "vae_config": "vae_config_sha256",
        "vae_state_dict": "vae_state_dict_sha256",
    }
    for record_name, expected_name in bindings.items():
        observed = records[record_name]["sha256"]
        if observed != expected[expected_name]:
            raise ValueError(
                f"{record_name} SHA256 mismatch: {observed} != "
                f"{expected[expected_name]}"
            )
    return records


def load_route_cells(
    path: str | Path,
    *,
    prefix: str,
    sample_count: int,
    token_grid_size: int,
    expected_expert_count: int,
    expected_blocks: list[int] | tuple[int, ...] | None = None,
    expected_sigmas: list[float] | tuple[float, ...] | None = None,
) -> dict[tuple[float, int], np.ndarray]:
    cells: dict[tuple[float, int], np.ndarray] = {}
    expected_shape = (sample_count, token_grid_size * token_grid_size)
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            match = ROUTE_KEY_PATTERN.fullmatch(key)
            if match is None or match.group("prefix") != prefix:
                continue
            sigma = float(match.group("sigma"))
            block = int(match.group("block"))
            routes = np.asarray(archive[key])
            if routes.shape != expected_shape:
                raise ValueError(
                    f"{key} has shape {routes.shape}; expected {expected_shape}"
                )
            if not np.issubdtype(routes.dtype, np.integer):
                raise ValueError(f"{key} must contain integer route IDs")
            if routes.size == 0 or int(routes.min()) < 0:
                raise ValueError(f"{key} contains invalid route IDs")
            if int(routes.max()) >= expected_expert_count:
                raise ValueError(
                    f"{key} contains a route ID outside {expected_expert_count} experts"
                )
            cell = (sigma, block)
            if cell in cells:
                raise ValueError(f"duplicate route cell {cell}")
            cells[cell] = routes.astype(np.int64, copy=True)

    expected_cells = None
    if expected_blocks is not None and expected_sigmas is not None:
        expected_cells = {
            (float(sigma), int(block))
            for sigma in expected_sigmas
            for block in expected_blocks
        }
        if set(cells) != expected_cells:
            missing = sorted(expected_cells - set(cells))
            extra = sorted(set(cells) - expected_cells)
            raise ValueError(f"route-cell mismatch; missing={missing}, extra={extra}")
    if not cells:
        raise ValueError(f"no route cells found for prefix {prefix!r}")
    return cells


def load_capture_samples(
    capture_summary_path: str | Path,
    *,
    latent_root: str | Path,
    expected_count: int,
    expected_latents: list[dict],
) -> list[dict]:
    payload = _load_json(capture_summary_path)
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_count:
        raise ValueError(
            f"capture summary must contain exactly {expected_count} samples"
        )
    latent_root = Path(latent_root).resolve()
    if len(expected_latents) != expected_count:
        raise ValueError("locked latent list does not match expected_count")
    resolved = []
    seen_paths = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} is not an object")
        required = {"label", "class_dir", "latent_path"}
        if not required.issubset(sample):
            raise ValueError(f"sample {index} is missing {sorted(required - set(sample))}")
        if (
            isinstance(sample["label"], bool)
            or not isinstance(sample["label"], int)
            or sample["label"] < 0
        ):
            raise ValueError(f"sample {index} has an invalid label")
        if (
            not isinstance(sample["class_dir"], str)
            or not sample["class_dir"]
            or Path(sample["class_dir"]).name != sample["class_dir"]
        ):
            raise ValueError(f"sample {index} has an invalid class_dir")
        if not isinstance(sample["latent_path"], str) or not sample["latent_path"]:
            raise ValueError(f"sample {index} has an invalid latent_path")
        expected_latent = expected_latents[index]
        relative_path = Path(expected_latent["relative_path"])
        captured_name = Path(sample["latent_path"]).name
        if relative_path.parent.name != sample["class_dir"] or relative_path.name != captured_name:
            raise ValueError(f"sample {index} does not match its locked latent path")
        candidate = latent_root / relative_path
        candidate = candidate.resolve()
        try:
            candidate.relative_to(latent_root)
        except ValueError as error:
            raise ValueError(f"sample {index} is outside latent root") from error
        if not candidate.is_file() or candidate in seen_paths:
            raise ValueError(f"sample {index} has a missing or duplicate latent")
        if candidate.parent.name != sample["class_dir"]:
            raise ValueError(f"sample {index} class_dir does not match its latent")
        latent_sha256 = sha256_file(candidate)
        if latent_sha256 != expected_latent["sha256"]:
            raise ValueError(
                f"sample {index} latent SHA256 mismatch: {latent_sha256} != "
                f"{expected_latent['sha256']}"
            )
        seen_paths.add(candidate)
        resolved.append(
            {
                "index": index,
                "label": int(sample["label"]),
                "class_dir": str(sample["class_dir"]),
                "latent_relative_path": relative_path.as_posix(),
                "latent_path": str(candidate),
                "latent_sha256": latent_sha256,
            }
        )
    return resolved


def _normalize_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("features must have shape [images, tokens, dimensions]")
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("features contain a zero-norm token")
    return features / norms


def _load_locked_dino_encoder(
    dino_path: str | Path,
    *,
    dino_source_path: str | Path,
    dino_source_tree_sha256: str,
    source_revision: str,
    device,
):
    os.environ["XFORMERS_DISABLED"] = "1"

    import torch
    import torch.nn as nn
    import timm.layers.pos_embed

    observed_source_sha256 = sha256_source_tree(dino_source_path)
    if observed_source_sha256 != dino_source_tree_sha256:
        raise ValueError(
            "DINO source tree SHA256 mismatch: "
            f"{observed_source_sha256} != {dino_source_tree_sha256}"
        )
    loaded_dino_modules = sorted(
        name for name in sys.modules if name == "dinov2" or name.startswith("dinov2.")
    )
    if loaded_dino_modules:
        raise RuntimeError(
            "DINO modules were imported before source verification: "
            + ", ".join(loaded_dino_modules)
        )
    with tempfile.TemporaryDirectory(prefix="promoe-locked-dinov2-") as temporary_dir:
        isolated_source = Path(temporary_dir) / f"dinov2-{source_revision}"
        shutil.copytree(
            Path(dino_source_path).resolve(),
            isolated_source,
            ignore=_ignore_dino_source_copy,
        )
        isolated_sha256 = sha256_source_tree(isolated_source)
        if isolated_sha256 != dino_source_tree_sha256:
            raise RuntimeError("isolated DINO source copy differs from the locked tree")
        encoder = torch.hub.load(
            str(isolated_source),
            "dinov2_vitb14",
            source="local",
            pretrained=False,
        )
    attention_module = sys.modules.get("dinov2.layers.attention")
    if (
        attention_module is None
        or getattr(attention_module, "XFORMERS_ENABLED", None) is not False
        or getattr(attention_module, "XFORMERS_AVAILABLE", None) is not False
    ):
        raise RuntimeError("locked DINO source did not disable xFormers attention")
    state_dict = torch.load(
        Path(dino_path).resolve(),
        map_location="cpu",
        weights_only=True,
    )
    encoder.load_state_dict(state_dict, strict=True)
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        encoder.pos_embed.data,
        [16, 16],
    )
    del encoder.head
    encoder.head = nn.Identity()
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


def extract_dino_features(
    samples: list[dict],
    *,
    vae_path: str | Path,
    dino_path: str | Path,
    dino_source_path: str | Path,
    device: str,
    batch_size: int,
    latent_key: str,
    latent_shape: tuple[int, ...],
    dino_source_revision: str,
    dino_source_tree_sha256: str,
) -> np.ndarray:
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    from repa.encoder import extract_teacher_features
    from utils import load_vae

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch_device = torch.device(device)
    vae = load_vae(
        "stabilityai/sd-vae-ft-mse",
        vae_path=str(Path(vae_path).resolve()),
    ).to(torch_device)
    vae.eval()
    encoder = _load_locked_dino_encoder(
        dino_path,
        dino_source_path=dino_source_path,
        dino_source_tree_sha256=dino_source_tree_sha256,
        source_revision=dino_source_revision,
        device=torch_device,
    )

    features = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            parameters = []
            for sample in batch:
                with np.load(sample["latent_path"], allow_pickle=False) as archive:
                    if latent_key not in archive.files:
                        raise ValueError(
                            f"{sample['latent_path']} does not contain {latent_key!r}"
                        )
                    parameters_array = np.asarray(
                        archive[latent_key], dtype=np.float32
                    )
                if parameters_array.shape != latent_shape:
                    raise ValueError(
                        f"{sample['latent_path']} has latent shape "
                        f"{parameters_array.shape}; expected {latent_shape}"
                    )
                if not np.isfinite(parameters_array).all():
                    raise ValueError(
                        f"{sample['latent_path']} contains non-finite latents"
                    )
                parameters.append(parameters_array)
            parameters_tensor = torch.from_numpy(np.stack(parameters)).to(torch_device)
            posterior = DiagonalGaussianDistribution(parameters_tensor)
            clean_latents = posterior.mode()
            decoded = vae.decode(clean_latents).sample
            images = (decoded.float().clamp(-1, 1) + 1.0) / 2.0
            patch_features = extract_teacher_features(
                encoder,
                images,
                "dinov2-vit-b",
            )
            features.append(patch_features.float().cpu().numpy())
    del encoder
    del vae
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return _normalize_features(np.concatenate(features, axis=0))


def load_or_extract_features(
    cache_path: str | Path,
    samples: list[dict],
    *,
    vae_path: str | Path,
    dino_path: str | Path,
    dino_source_path: str | Path,
    device: str,
    batch_size: int,
    latent_key: str,
    latent_shape: tuple[int, ...],
    locked_inputs: dict,
    protocol_sha256: str,
    repository_commit: str,
    dino_source_revision: str,
    dino_source_tree_sha256: str,
    runtime: dict,
) -> tuple[np.ndarray, dict]:
    cache_path = Path(cache_path)
    identity = {
        "sample_latent_sha256": [sample["latent_sha256"] for sample in samples],
        "vae_path": str(Path(vae_path).resolve()),
        "dino_path": str(Path(dino_path).resolve()),
        "dino_source_path": str(Path(dino_source_path).resolve()),
        "decode": "posterior_mode",
        "encoder": "dinov2-vit-b",
        "attention_backend": DINO_ATTENTION_BACKEND,
        "dino_source_revision": dino_source_revision,
        "dino_source_tree_sha256": dino_source_tree_sha256,
        "latent_key": latent_key,
        "latent_shape": list(latent_shape),
        "device": device,
        "batch_size": batch_size,
        "software_versions": runtime["software_versions"],
        "protocol_sha256": protocol_sha256,
        "repository_commit": repository_commit,
        "locked_input_sha256": {
            key: value["sha256"] for key, value in sorted(locked_inputs.items())
        },
    }
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            required_keys = {
                "features",
                "identity_json",
                "generation_runtime_json",
                "feature_sha256",
            }
            if set(archive.files) != required_keys:
                raise ValueError("feature cache metadata is incomplete")
            cached_identity = json.loads(str(archive["identity_json"].item()))
            if cached_identity != identity:
                raise ValueError("feature cache identity does not match locked inputs")
            stored_features = np.asarray(archive["features"], dtype=np.float32)
            expected_feature_sha256 = str(archive["feature_sha256"].item())
            observed_feature_sha256 = sha256_float32_array(stored_features)
            if observed_feature_sha256 != expected_feature_sha256:
                raise ValueError("feature cache content SHA256 mismatch")
            generation_runtime = json.loads(
                str(archive["generation_runtime_json"].item())
            )
            _normalize_features(stored_features)
        return stored_features, {
            "identity": identity,
            "generation_runtime": generation_runtime,
            "feature_sha256": observed_feature_sha256,
            "cache_hit": True,
        }

    features = extract_dino_features(
        samples,
        vae_path=vae_path,
        dino_path=dino_path,
        dino_source_path=dino_source_path,
        device=device,
        batch_size=batch_size,
        latent_key=latent_key,
        latent_shape=latent_shape,
        dino_source_revision=dino_source_revision,
        dino_source_tree_sha256=dino_source_tree_sha256,
    )
    features = _normalize_features(features)
    feature_sha256 = sha256_float32_array(features)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            dir=cache_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                features=features.astype(np.float32),
                identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
                generation_runtime_json=np.asarray(
                    json.dumps(runtime, sort_keys=True)
                ),
                feature_sha256=np.asarray(feature_sha256),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return features, {
        "identity": identity,
        "generation_runtime": runtime,
        "feature_sha256": feature_sha256,
        "cache_hit": False,
    }


def _within_image_neighbors(features: np.ndarray) -> np.ndarray:
    image_count, token_count, _ = features.shape
    if token_count < 2:
        raise ValueError("within-image neighbors require at least two tokens")
    neighbors = np.empty((image_count, token_count), dtype=np.int64)
    for image_index in range(image_count):
        similarities = features[image_index] @ features[image_index].T
        np.fill_diagonal(similarities, -np.inf)
        neighbors[image_index] = np.argmax(similarities, axis=1)
    return neighbors


def _cross_image_neighbors(
    features: np.ndarray,
    *,
    grid_size: int,
    token_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if token_stride <= 0:
        raise ValueError("cross-image token stride must be positive")
    image_count, token_count, dimension = features.shape
    if image_count < 2:
        raise ValueError("cross-image neighbors require at least two images")
    if token_count != grid_size * grid_size:
        raise ValueError("feature token count does not match route grid")
    positions = np.arange(token_count, dtype=np.int64).reshape(grid_size, grid_size)
    positions = positions[::token_stride, ::token_stride].reshape(-1)
    sampled = features[:, positions].reshape(-1, dimension)
    owner = np.repeat(np.arange(image_count, dtype=np.int64), len(positions))
    best_scores = np.empty((len(sampled), image_count), dtype=np.float32)
    best_positions = np.empty((len(sampled), image_count), dtype=np.int64)
    chunk_size = 256
    for start in range(0, len(sampled), chunk_size):
        stop = min(start + chunk_size, len(sampled))
        similarities = sampled[start:stop] @ sampled.T
        same_image = owner[None, :] == owner[start:stop, None]
        similarities[same_image] = -np.inf
        by_donor_image = similarities.reshape(
            stop - start,
            image_count,
            len(positions),
        )
        donor_position_indices = np.argmax(by_donor_image, axis=2)
        best_positions[start:stop] = positions[donor_position_indices]
        best_scores[start:stop] = np.take_along_axis(
            by_donor_image,
            donor_position_indices[:, :, None],
            axis=2,
        )[:, :, 0]
    neighbor_images = np.argmax(best_scores, axis=1)
    neighbor_positions = np.take_along_axis(
        best_positions,
        neighbor_images[:, None],
        axis=1,
    )[:, 0]
    query_images = owner
    query_positions = np.tile(positions, image_count)
    return (
        np.stack([query_images, query_positions], axis=1),
        np.stack([neighbor_images, neighbor_positions], axis=1),
        positions,
        best_scores.reshape(image_count, len(positions), image_count),
        best_positions.reshape(image_count, len(positions), image_count),
    )


def _cross_bootstrap_design(
    candidate_scores: np.ndarray,
    candidate_positions: np.ndarray,
    query_positions: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict:
    image_count, query_count, donor_count = candidate_scores.shape
    if donor_count != image_count or candidate_positions.shape != candidate_scores.shape:
        raise ValueError("cross-image candidate tables have incompatible shapes")
    if query_positions.shape != (query_count,):
        raise ValueError("cross-image query positions have the wrong shape")

    query_draws = []
    neighbor_images = []
    neighbor_positions = []
    chunk_size = 100
    donor_ids = np.arange(image_count, dtype=np.int64)
    for start in range(0, resamples, chunk_size):
        count = min(chunk_size, resamples - start)
        drawn_queries = rng.integers(0, image_count, size=(count, image_count))
        drawn_references = rng.integers(0, image_count, size=(count, image_count))
        for row in range(count):
            while len(np.unique(drawn_references[row])) < 2:
                drawn_references[row] = rng.integers(0, image_count, size=image_count)

        reference_present = np.zeros((count, image_count), dtype=bool)
        reference_present[
            np.repeat(np.arange(count), image_count),
            drawn_references.reshape(-1),
        ] = True
        scores = candidate_scores[drawn_queries]
        valid = reference_present[:, None, None, :]
        valid = valid & (
            donor_ids[None, None, None, :]
            != drawn_queries[:, :, None, None]
        )
        if not np.all(valid.any(axis=3)):
            raise AssertionError("two-way bootstrap drew an empty cross-image gallery")
        selected_images = np.argmax(np.where(valid, scores, -np.inf), axis=3)
        positions = np.take_along_axis(
            candidate_positions[drawn_queries],
            selected_images[:, :, :, None],
            axis=3,
        )[:, :, :, 0]
        query_draws.append(drawn_queries.astype(np.int16))
        neighbor_images.append(selected_images.astype(np.int16))
        neighbor_positions.append(positions.astype(np.int16))
    return {
        "query_images": np.concatenate(query_draws, axis=0),
        "neighbor_images": np.concatenate(neighbor_images, axis=0),
        "neighbor_positions": np.concatenate(neighbor_positions, axis=0),
        "query_positions": query_positions.astype(np.int16, copy=True),
    }


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or np.any(values < 0):
        raise ValueError("Gini values must be a nonnegative vector")
    total = float(values.sum())
    if total == 0:
        return 0.0
    sorted_values = np.sort(values)
    count = len(values)
    coefficients = 2 * np.arange(1, count + 1) - count - 1
    return float(np.dot(coefficients, sorted_values) / (count * total))


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite vector of length >= 2")
    chunk_size = 1000
    means = []
    for start in range(0, resamples, chunk_size):
        count = min(chunk_size, resamples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[indices].mean(axis=1))
    bootstrap = np.concatenate(means)
    return {
        "mean": float(values.mean()),
        "lcb95": float(np.quantile(bootstrap, 0.025)),
        "ucb95": float(np.quantile(bootstrap, 0.975)),
        "positive_images": int(np.sum(values > 0)),
        "image_count": int(len(values)),
    }


def _cross_cluster_bootstrap_interval(
    point_values: np.ndarray,
    routes: np.ndarray,
    design: dict,
    derangements: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict:
    point_values = np.asarray(point_values, dtype=np.float64)
    if point_values.ndim != 1 or len(point_values) != routes.shape[0]:
        raise ValueError("cross bootstrap point values must contain one value per image")
    query_images = design["query_images"]
    neighbor_images = design["neighbor_images"]
    neighbor_positions = design["neighbor_positions"]
    query_positions = design["query_positions"]
    expected_shape = (resamples, routes.shape[0], len(query_positions))
    if query_images.shape != expected_shape[:2]:
        raise ValueError("cross bootstrap query design has the wrong shape")
    if neighbor_images.shape != expected_shape or neighbor_positions.shape != expected_shape:
        raise ValueError("cross bootstrap neighbor design has the wrong shape")
    if derangements.ndim != 2 or derangements.shape[1] != routes.shape[0]:
        raise ValueError("cross bootstrap controls have the wrong shape")

    bootstrap_delta = np.empty(resamples, dtype=np.float64)
    chunk_size = 100
    positions = query_positions[None, None, :]
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        queries = query_images[start:stop].astype(np.int64, copy=False)
        neighbors = neighbor_images[start:stop].astype(np.int64, copy=False)
        neighbor_tokens = neighbor_positions[start:stop].astype(np.int64, copy=False)
        observed_matches = (
            routes[queries[:, :, None], positions]
            == routes[neighbors, neighbor_tokens]
        )

        selected_controls = derangements[
            rng.integers(0, len(derangements), size=stop - start)
        ]
        control_query_images = np.take_along_axis(
            selected_controls,
            queries,
            axis=1,
        )
        control_neighbor_images = selected_controls[
            np.arange(stop - start)[:, None, None],
            neighbors,
        ]
        control_matches = (
            routes[control_query_images[:, :, None], positions]
            == routes[control_neighbor_images, neighbor_tokens]
        )
        bootstrap_delta[start:stop] = (
            observed_matches.mean(axis=(1, 2))
            - control_matches.mean(axis=(1, 2))
        )
    return {
        "mean": float(point_values.mean()),
        "lcb95": float(np.quantile(bootstrap_delta, 0.025)),
        "ucb95": float(np.quantile(bootstrap_delta, 0.975)),
        "bootstrap_mean": float(bootstrap_delta.mean()),
        "positive_images": int(np.sum(point_values > 0)),
        "image_count": int(len(point_values)),
        "method": "two_way_query_gallery_image_cluster_with_control_draw",
    }


def _random_nonzero_shifts(
    rng: np.random.Generator,
    *,
    image_count: int,
    grid_size: int,
) -> np.ndarray:
    flat = rng.integers(1, grid_size * grid_size, size=image_count)
    return np.stack([flat // grid_size, flat % grid_size], axis=1)


def _shift_routes_batch(
    routes: np.ndarray,
    shifts: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    routes = np.asarray(routes)
    shifts = np.asarray(shifts, dtype=np.int64)
    if routes.ndim != 2 or routes.shape[1] != grid_size * grid_size:
        raise ValueError("routes do not match the requested grid")
    if shifts.ndim != 3 or shifts.shape[1:] != (routes.shape[0], 2):
        raise ValueError("shifts must have shape [controls, images, 2]")
    positions = np.arange(grid_size * grid_size, dtype=np.int64)
    y = positions // grid_size
    x = positions % grid_size
    source_y = (y[None, None, :] - shifts[:, :, 0, None]) % grid_size
    source_x = (x[None, None, :] - shifts[:, :, 1, None]) % grid_size
    source = source_y * grid_size + source_x
    broadcast_routes = np.broadcast_to(routes, source.shape)
    return np.take_along_axis(broadcast_routes, source, axis=2)


def _shift_routes(routes: np.ndarray, shifts: np.ndarray, grid_size: int) -> np.ndarray:
    return _shift_routes_batch(routes, np.asarray(shifts)[None], grid_size)[0]


def _derangement(rng: np.random.Generator, count: int) -> np.ndarray:
    identity = np.arange(count)
    for _ in range(1000):
        candidate = rng.permutation(count)
        if np.all(candidate != identity):
            return candidate
    raise RuntimeError("failed to draw an image derangement")


def _within_rates(routes: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    routes = np.asarray(routes)
    return _within_rates_batch(routes[None], neighbors)[0]


def _within_rates_batch(routes: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    routes = np.asarray(routes)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if routes.ndim != 3 or neighbors.shape != routes.shape[1:]:
        raise ValueError("within-image route/neighbor shape mismatch")
    targets = np.take_along_axis(
        routes,
        np.broadcast_to(neighbors, routes.shape),
        axis=2,
    )
    return (routes == targets).mean(axis=2)


def _cross_rates(
    routes: np.ndarray,
    queries: np.ndarray,
    neighbors: np.ndarray,
) -> np.ndarray:
    image_count = routes.shape[0]
    matches = (
        routes[queries[:, 0], queries[:, 1]]
        == routes[neighbors[:, 0], neighbors[:, 1]]
    )
    result = np.empty(image_count, dtype=np.float64)
    for image_index in range(image_count):
        selected = queries[:, 0] == image_index
        if not np.any(selected):
            raise AssertionError("cross-image query owner has no tokens")
        result[image_index] = float(matches[selected].mean())
    return result


def _sample_route_pairs(
    token_count: int,
    *,
    pair_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    upper = np.triu_indices(token_count, k=1)
    available = len(upper[0])
    if pair_count <= 0 or pair_count > available:
        raise ValueError("route-separation pair count is outside the valid range")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(available, size=pair_count, replace=False))
    return upper[0][selected], upper[1][selected]


def _route_pair_similarities(
    features: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
) -> np.ndarray:
    if pair_left.shape != pair_right.shape or pair_left.ndim != 1:
        raise ValueError("route-separation pair indices must be equal vectors")
    return np.einsum(
        "ipd,ipd->ip",
        features[:, pair_left],
        features[:, pair_right],
        optimize=True,
        dtype=np.float64,
    )


def _route_separation_batch(
    routes: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    pair_similarities: np.ndarray,
) -> np.ndarray:
    routes = np.asarray(routes)
    if routes.ndim != 3:
        raise ValueError("batched routes must have shape [controls, images, tokens]")
    image_count = routes.shape[1]
    pair_count = len(pair_left)
    if pair_similarities.shape != (image_count, pair_count):
        raise ValueError("route-separation similarities have the wrong shape")
    same_route = routes[:, :, pair_left] == routes[:, :, pair_right]
    same_count = same_route.sum(axis=2, dtype=np.int64)
    different_count = pair_count - same_count
    if np.any(same_count == 0) or np.any(different_count == 0):
        raise ValueError(
            "route-separation metric requires sampled same- and different-route pairs"
        )
    same_sum = np.einsum(
        "cip,ip->ci",
        same_route,
        pair_similarities,
        optimize=True,
        dtype=np.float64,
    )
    total_sum = pair_similarities.sum(axis=1, dtype=np.float64)[None, :]
    return same_sum / same_count - (total_sum - same_sum) / different_count


def _route_separation(
    routes: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    pair_similarities: np.ndarray,
) -> np.ndarray:
    return _route_separation_batch(
        np.asarray(routes)[None],
        pair_left,
        pair_right,
        pair_similarities,
    )[0]


def _cell_seed(base_seed: int, sigma: float, block: int) -> int:
    material = f"{base_seed}|{sigma:.9f}|{block}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _cell_metrics(
    routes: np.ndarray,
    *,
    sigma: float,
    block: int,
    grid_size: int,
    within_neighbors: np.ndarray,
    cross_queries: np.ndarray,
    cross_neighbors: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    pair_similarities: np.ndarray,
    cross_bootstrap_design: dict,
    expected_expert_count: int,
    control_resamples: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(_cell_seed(seed, sigma, block))
    observed_within = _within_rates(routes, within_neighbors)
    observed_cross = _cross_rates(routes, cross_queries, cross_neighbors)
    observed_separation = _route_separation(
        routes,
        pair_left,
        pair_right,
        pair_similarities,
    )

    null_within = np.empty((control_resamples, len(routes)), dtype=np.float64)
    null_separation = np.empty_like(null_within)
    shifts = _random_nonzero_shifts(
        rng,
        image_count=control_resamples * len(routes),
        grid_size=grid_size,
    ).reshape(control_resamples, len(routes), 2)
    control_chunk_size = 50
    for start in range(0, control_resamples, control_chunk_size):
        stop = min(start + control_chunk_size, control_resamples)
        shifted = _shift_routes_batch(
            routes,
            shifts[start:stop],
            grid_size,
        )
        null_within[start:stop] = _within_rates_batch(shifted, within_neighbors)
        null_separation[start:stop] = _route_separation_batch(
            shifted,
            pair_left,
            pair_right,
            pair_similarities,
        )
    null_within_global = null_within.mean(axis=1)
    null_separation_global = null_separation.mean(axis=1)

    null_cross = np.empty((control_resamples, len(routes)), dtype=np.float64)
    null_cross_global = np.empty(control_resamples, dtype=np.float64)
    derangements = np.empty(
        (control_resamples, len(routes)),
        dtype=np.int64,
    )
    for index in range(control_resamples):
        derangements[index] = _derangement(rng, len(routes))
        shuffled = routes[derangements[index]]
        null_cross[index] = _cross_rates(shuffled, cross_queries, cross_neighbors)
        null_cross_global[index] = null_cross[index].mean()

    within_delta = observed_within - null_within.mean(axis=0)
    cross_delta = observed_cross - null_cross.mean(axis=0)
    separation_delta = observed_separation - null_separation.mean(axis=0)
    within_stats = _bootstrap_mean_interval(
        within_delta,
        rng=rng,
        resamples=bootstrap_resamples,
    )
    cross_stats = _cross_cluster_bootstrap_interval(
        cross_delta,
        routes,
        cross_bootstrap_design,
        derangements,
        rng=rng,
        resamples=bootstrap_resamples,
    )
    separation_stats = _bootstrap_mean_interval(
        separation_delta,
        rng=rng,
        resamples=bootstrap_resamples,
    )
    within_p = float(
        (1 + np.sum(null_within_global >= observed_within.mean()))
        / (control_resamples + 1)
    )
    cross_p = float(
        (1 + np.sum(null_cross_global >= observed_cross.mean()))
        / (control_resamples + 1)
    )
    separation_p = float(
        (1 + np.sum(null_separation_global >= observed_separation.mean()))
        / (control_resamples + 1)
    )

    counts = np.bincount(routes.reshape(-1), minlength=expected_expert_count)
    probabilities = counts / counts.sum()
    nonzero = probabilities[probabilities > 0]
    normalized_entropy = (
        float(
            -(nonzero * np.log(nonzero)).sum()
            / math.log(expected_expert_count)
        )
        if expected_expert_count > 1
        else 1.0
    )
    return {
        "sigma": float(sigma),
        "block": int(block),
        "sample_count": int(len(routes)),
        "token_count_per_sample": int(routes.shape[1]),
        "expert_count": expected_expert_count,
        "route_counts": counts.tolist(),
        "load": {
            "normalized_entropy": normalized_entropy,
            "gini": _gini(counts),
            "maximum_fraction": float(probabilities.max()),
            "minimum_fraction": float(probabilities.min()),
        },
        "within_image_dino_knn": {
            "observed_mean": float(observed_within.mean()),
            "spatial_shift_control_mean": float(null_within.mean()),
            "correct_minus_control": within_stats,
            "raw_one_sided_control_p": within_p,
        },
        "cross_image_dino_knn": {
            "observed_mean": float(observed_cross.mean()),
            "image_mismatch_control_mean": float(null_cross.mean()),
            "correct_minus_control": cross_stats,
            "raw_one_sided_control_p": cross_p,
        },
        "dino_route_separation": {
            "observed_mean": float(observed_separation.mean()),
            "spatial_shift_control_mean": float(null_separation.mean()),
            "correct_minus_control": separation_stats,
            "raw_one_sided_control_p": separation_p,
        },
    }


def _holm_adjusted_p_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Holm correction requires a finite nonempty p-value vector")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("Holm correction received a p-value outside [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    scaled = (len(values) - np.arange(len(values))) * ranked
    adjusted_ranked = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def audit_route_cells(
    features: np.ndarray,
    route_cells: dict[tuple[float, int], np.ndarray],
    *,
    grid_size: int,
    expected_expert_count: int,
    cross_image_token_stride: int,
    route_separation_pair_count: int,
    control_resamples: int,
    bootstrap_resamples: int,
    seed: int,
    gates: dict,
) -> dict:
    features = _normalize_features(features)
    image_count, token_count, _ = features.shape
    if token_count != grid_size * grid_size:
        raise ValueError("DINO feature grid does not match route-token grid")
    for cell, routes in route_cells.items():
        if routes.shape != (image_count, token_count):
            raise ValueError(f"route cell {cell} does not match DINO features")

    within_neighbors = _within_image_neighbors(features)
    (
        cross_queries,
        cross_neighbors,
        cross_positions,
        cross_candidate_scores,
        cross_candidate_positions,
    ) = _cross_image_neighbors(
        features,
        grid_size=grid_size,
        token_stride=cross_image_token_stride,
    )
    cross_bootstrap_design = _cross_bootstrap_design(
        cross_candidate_scores,
        cross_candidate_positions,
        cross_positions,
        rng=np.random.default_rng(_cell_seed(seed, 0.0, -2)),
        resamples=bootstrap_resamples,
    )
    pair_seed = _cell_seed(seed, 0.0, -1)
    pair_left, pair_right = _sample_route_pairs(
        token_count,
        pair_count=route_separation_pair_count,
        seed=pair_seed,
    )
    pair_similarities = _route_pair_similarities(
        features,
        pair_left,
        pair_right,
    )
    pair_indices_sha256 = hashlib.sha256(
        np.stack([pair_left, pair_right], axis=1).astype("<i8").tobytes()
    ).hexdigest()
    cells = []
    for (sigma, block), routes in sorted(route_cells.items()):
        cells.append(
            _cell_metrics(
                routes,
                sigma=sigma,
                block=block,
                grid_size=grid_size,
                within_neighbors=within_neighbors,
                cross_queries=cross_queries,
                cross_neighbors=cross_neighbors,
                pair_left=pair_left,
                pair_right=pair_right,
                pair_similarities=pair_similarities,
                cross_bootstrap_design=cross_bootstrap_design,
                expected_expert_count=expected_expert_count,
                control_resamples=control_resamples,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            )
        )

    metric_specs = (
        (
            "within_image",
            "within_image_dino_knn",
            "minimum_within_image_knn_delta",
        ),
        (
            "cross_image",
            "cross_image_dino_knn",
            "minimum_cross_image_knn_delta",
        ),
        (
            "route_separation",
            "dino_route_separation",
            "minimum_route_separation_delta",
        ),
    )
    metric_locations = []
    raw_p_values = []
    for cell in cells:
        for requirement_name, metric_name, delta_gate_name in metric_specs:
            metric_locations.append(
                (cell, requirement_name, metric_name, delta_gate_name)
            )
            raw_p_values.append(metric_locations[-1][0][metric_name]["raw_one_sided_control_p"])
    adjusted_p_values = _holm_adjusted_p_values(np.asarray(raw_p_values))
    for location, adjusted_p in zip(metric_locations, adjusted_p_values):
        cell, requirement_name, metric_name, delta_gate_name = location
        metric = cell[metric_name]
        metric["holm_adjusted_one_sided_control_p"] = float(adjusted_p)
        cell.setdefault("requirements", {})[requirement_name] = bool(
            metric["correct_minus_control"]["mean"] >= gates[delta_gate_name]
            and metric["correct_minus_control"]["lcb95"] > 0
            and adjusted_p <= gates["maximum_one_sided_control_p"]
        )
    for cell in cells:
        cell["passed"] = bool(all(cell["requirements"].values()))

    passing_cells = [cell for cell in cells if cell["passed"]]
    blocks = sorted({cell["block"] for cell in cells})
    sigmas = sorted({cell["sigma"] for cell in cells})
    passing_blocks = [
        block
        for block in blocks
        if any(cell["passed"] and cell["block"] == block for cell in cells)
    ]
    passing_sigmas = [
        sigma
        for sigma in sigmas
        if any(cell["passed"] and cell["sigma"] == sigma for cell in cells)
    ]
    decision_checks = {
        "passing_cells": {
            "observed": len(passing_cells),
            "required": gates["minimum_passing_cells"],
            "passed": len(passing_cells) >= gates["minimum_passing_cells"],
        },
        "passing_blocks": {
            "observed": passing_blocks,
            "required_count": gates["minimum_passing_blocks"],
            "passed": len(passing_blocks) >= gates["minimum_passing_blocks"],
        },
        "passing_sigmas": {
            "observed": passing_sigmas,
            "required_count": gates["minimum_passing_sigmas"],
            "passed": len(passing_sigmas) >= gates["minimum_passing_sigmas"],
        },
    }
    supported = bool(all(check["passed"] for check in decision_checks.values()))
    return {
        "cell_count": len(cells),
        "cross_image_sampled_token_positions": cross_positions.tolist(),
        "route_separation_pair_count": len(pair_left),
        "route_separation_pair_indices_sha256": pair_indices_sha256,
        "multiple_testing": {
            "method": "holm_bonferroni",
            "family_size": len(raw_p_values),
            "maximum_adjusted_p": gates["maximum_one_sided_control_p"],
        },
        "cells": cells,
        "decision_checks": decision_checks,
        "independent_semantic_structure_supported": supported,
        "interpretation": (
            "Independent DINO semantics agree with these historical ProMoE routes "
            "beyond both locked controls. This is enough to justify confirmation "
            "on a fresh checkpoint, but it does not establish that RCL caused the "
            "agreement or that the selected expert is better for denoising."
            if supported
            else
            "Independent semantic structure was not established by the locked "
            "dirty discovery gate. Stop this semantic claim instead of spending a "
            "fresh training run on it, unless new independent evidence changes the "
            "question."
        ),
    }


def write_json_atomic(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def render_summary_markdown(summary: dict) -> str:
    decision = summary["audit"]["independent_semantic_structure_supported"]
    lines = [
        "# 路由分组是否真的带有独立语义",
        "",
        f"结论：**{'通过' if decision else '未通过'}**。",
        "",
        "这份检查只把 DINOv2 当作尺子。DINO 特征没有送进 ProMoE，也没有参与训练。",
        "当前输入只包含 step 50K checkpoint 的 32 张历史路由抓取，而且没有完整训练封存。因此结果只能帮助筛选假设，不能写成论文证据。",
        "",
        "## 检查方法",
        "",
        "- 图内检查：DINO 认为最相近的两个 patch，是否更常交给同一专家。对照会整体平移路由图，保留每位专家的 token 数和空间连贯性。",
        "- 跨图检查：不同图片中 DINO 认为最相近的 patch，是否更常交给同一专家。对照会把完整路由图换到另一张图片，保留每张路由图本身。",
        "- 分离度检查：同一专家 token 的 DINO 相似度是否高于不同专家，并超过整体平移对照。",
        "- 跨图置信区间同时重采样查询图片和候选图库；54 次 cell/指标检验统一做 Holm 校正。",
        "",
        "## 汇总",
        "",
        "| sigma | block | 图内差值 | 跨图差值 | 分离度差值 | 通过 |",
        "| ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for cell in summary["audit"]["cells"]:
        within = cell["within_image_dino_knn"]["correct_minus_control"]["mean"]
        cross = cell["cross_image_dino_knn"]["correct_minus_control"]["mean"]
        separation = cell["dino_route_separation"]["correct_minus_control"]["mean"]
        lines.append(
            f"| {cell['sigma']:.1f} | {cell['block']} | {within:.4f} | "
            f"{cross:.4f} | {separation:.4f} | "
            f"{'是' if cell['passed'] else '否'} |"
        )
    checks = summary["audit"]["decision_checks"]
    lines.extend(
        [
            "",
            "## 最终门槛",
            "",
            f"- 通过 cell：{checks['passing_cells']['observed']} / "
            f"至少 {checks['passing_cells']['required']}。",
            f"- 覆盖 block：{len(checks['passing_blocks']['observed'])} / "
            f"至少 {checks['passing_blocks']['required_count']}。",
            f"- 覆盖噪声位置：{len(checks['passing_sigmas']['observed'])} / "
            f"至少 {checks['passing_sigmas']['required_count']}。",
            "",
            "## 怎样解释",
            "",
            summary["audit"]["interpretation"],
            "",
            "这里最多能说明这批 ProMoE 路由与独立语义有联系，不能说明联系一定由 RCL 造成。要判断 RCL 的作用，仍需从零训练的无 RCL 对照。",
            "",
        ]
    )
    return "\n".join(lines)


def write_text_atomic(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _git_output(*args: str) -> str:
    return provenance_git_output(PROJECT_ROOT, *args)


def locked_repository_state() -> dict:
    reject_history_overrides(PROJECT_ROOT)
    reject_index_overrides(PROJECT_ROOT)
    status = fresh_worktree_status(PROJECT_ROOT)
    if status:
        raise RuntimeError(
            "semantic audit requires a clean committed worktree; current changes:\n"
            f"{status}"
        )
    commit = _git_output("rev-parse", "HEAD")
    branch = _git_output("branch", "--show-current")
    if not branch:
        raise RuntimeError("semantic audit refuses a detached HEAD")
    try:
        upstream = _git_output(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
        )
        upstream_commit = _git_output("rev-parse", "@{upstream}")
    except subprocess.CalledProcessError as error:
        raise RuntimeError("semantic audit requires a configured pushed upstream") from error
    if upstream_commit != commit:
        raise RuntimeError(
            f"HEAD {commit} is not the pushed upstream commit {upstream_commit}"
        )
    configured_origin_urls = _git_output(
        "config", "--local", "--get-all", "remote.origin.url"
    ).splitlines()
    if configured_origin_urls != [AUTHORITATIVE_REMOTE_URL]:
        raise RuntimeError(
            "semantic audit origin differs from the authoritative repository"
        )
    remote = _git_output("config", "--local", f"branch.{branch}.remote")
    remote_ref = _git_output("config", "--local", f"branch.{branch}.merge")
    if remote != "origin" or remote_ref != f"refs/heads/{branch}":
        raise RuntimeError("semantic audit requires its same-name origin branch upstream")
    remote_commit = authoritative_remote_tip(
        AUTHORITATIVE_REMOTE_URL,
        remote_ref,
    )
    if remote_commit != commit:
        raise RuntimeError(
            f"HEAD {commit} is not the live remote commit {remote_commit}"
        )
    return {
        "branch": branch,
        "commit": commit,
        "upstream": upstream,
        "upstream_commit": upstream_commit,
        "remote": remote,
        "remote_url": AUTHORITATIVE_REMOTE_URL,
        "remote_ref": remote_ref,
        "remote_commit": remote_commit,
        "status_clean": True,
        "fresh_blob_status": status,
        "history_overrides_rejected": True,
        "special_index_entries": [],
    }


def runtime_environment(device: str) -> dict:
    import diffusers
    import timm
    import torch
    import torchvision

    return {
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "diffusers": diffusers.__version__,
            "timm": timm.__version__,
        },
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "dino_attention_backend": DINO_ATTENTION_BACKEND,
        "platform": platform.platform(),
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_directory_parent(path: Path, description: str) -> None:
    ancestor = path.parent
    while not ancestor.exists():
        parent = ancestor.parent
        if parent == ancestor:
            break
        ancestor = parent
    if not ancestor.is_dir():
        raise ValueError(
            f"{description} has a non-directory existing ancestor: {ancestor}"
        )


def _resolve_audit_output_paths(
    output_dir: str | Path,
    feature_cache_path: str | Path | None,
    *,
    protected_paths: list[str | Path],
    protected_roots: list[str | Path],
) -> dict[str, Path]:
    declared_output_dir = Path(output_dir)
    if declared_output_dir.is_symlink():
        raise ValueError("output directory may not be a symlink")
    output_dir = declared_output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output directory path is not a directory")

    declared_paths = {
        "summary_json": output_dir / "summary.json",
        "summary_markdown": output_dir / "summary.md",
        "feature_cache": (
            Path(feature_cache_path)
            if feature_cache_path is not None
            else output_dir / "dino_features.npz"
        ),
    }
    for name, path in declared_paths.items():
        if path.is_symlink():
            raise ValueError(f"{name} path may not be a symlink")
    resolved_paths = {name: path.resolve() for name, path in declared_paths.items()}
    if len(set(resolved_paths.values())) != len(resolved_paths):
        raise ValueError("audit output paths must be distinct")
    resolved_items = list(resolved_paths.items())
    for index, (left_name, left_path) in enumerate(resolved_items):
        for right_name, right_path in resolved_items[index + 1 :]:
            if _is_within(left_path, right_path) or _is_within(
                right_path, left_path
            ):
                raise ValueError(
                    "audit output paths must not be ancestors or descendants: "
                    f"{left_name}={left_path}, {right_name}={right_path}"
                )

    _require_directory_parent(output_dir, "output directory")
    for name, path in resolved_paths.items():
        _require_directory_parent(path, name)

    protected = {Path(path).resolve() for path in protected_paths}
    roots = [Path(path).resolve() for path in protected_roots]
    for name, path in resolved_paths.items():
        if path in protected:
            raise ValueError(f"{name} collides with a locked input")
        if any(_is_within(path, root) for root in roots):
            raise ValueError(f"{name} is inside a protected input/source tree")
        if path.exists() and not path.is_file():
            raise ValueError(f"{name} path exists but is not a file")
    return {"output_dir": output_dir, **resolved_paths}


def run_locked_semantic_audit(
    *,
    route_ids_path: str | Path,
    capture_summary_path: str | Path,
    latent_root: str | Path,
    dino_path: str | Path,
    dino_source_path: str | Path,
    vae_path: str | Path,
    output_dir: str | Path,
    feature_cache_path: str | Path | None,
    device: str,
    batch_size: int,
) -> dict:
    repository = locked_repository_state()
    manifest_path = DEFAULT_MANIFEST.resolve()
    manifest = load_manifest(manifest_path)
    protocol_sha256 = sha256_file(manifest_path)
    locked_inputs = validate_locked_inputs(
        manifest,
        route_ids_path=route_ids_path,
        capture_summary_path=capture_summary_path,
        dino_path=dino_path,
        dino_source_path=dino_source_path,
        vae_path=vae_path,
    )
    expected = manifest["expected"]
    extractor = manifest["feature_extractor"]
    statistics = manifest["statistics"]
    if device != extractor["device"]:
        raise ValueError(
            f"device {device!r} differs from locked device {extractor['device']!r}"
        )
    if batch_size != extractor["batch_size"]:
        raise ValueError(
            f"batch size {batch_size} differs from locked batch size "
            f"{extractor['batch_size']}"
        )
    runtime = runtime_environment(device)
    if runtime["software_versions"] != extractor["software_versions"]:
        raise ValueError(
            "runtime software versions differ from the locked feature extractor"
        )
    if runtime["dino_attention_backend"] != extractor["attention_backend"]:
        raise ValueError(
            "runtime DINO attention backend differs from the locked feature extractor"
        )
    route_cells = load_route_cells(
        route_ids_path,
        prefix=expected["route_prefix"],
        sample_count=expected["sample_count"],
        token_grid_size=expected["token_grid_size"],
        expected_expert_count=expected["num_routed_experts"],
        expected_blocks=expected["blocks"],
        expected_sigmas=expected["sigmas"],
    )
    samples = load_capture_samples(
        capture_summary_path,
        latent_root=latent_root,
        expected_count=expected["sample_count"],
        expected_latents=expected["sample_latents"],
    )

    output_paths = _resolve_audit_output_paths(
        output_dir,
        feature_cache_path,
        protected_paths=[
            manifest_path,
            *(record["path"] for record in locked_inputs.values()),
            *(sample["latent_path"] for sample in samples),
        ],
        protected_roots=[PROJECT_ROOT, latent_root, vae_path, dino_source_path],
    )
    output_dir = output_paths["output_dir"]
    feature_cache_path = output_paths["feature_cache"]
    features, feature_cache = load_or_extract_features(
        feature_cache_path,
        samples,
        vae_path=vae_path,
        dino_path=dino_path,
        dino_source_path=dino_source_path,
        device=device,
        batch_size=batch_size,
        latent_key=extractor["latent_key"],
        latent_shape=tuple(extractor["latent_shape"]),
        locked_inputs=locked_inputs,
        protocol_sha256=protocol_sha256,
        repository_commit=repository["commit"],
        dino_source_revision=extractor["dino_source_revision"],
        dino_source_tree_sha256=expected["dino_source_tree_sha256"],
        runtime=runtime,
    )
    audit = audit_route_cells(
        features,
        route_cells,
        grid_size=expected["token_grid_size"],
        expected_expert_count=expected["num_routed_experts"],
        cross_image_token_stride=extractor["cross_image_token_stride"],
        route_separation_pair_count=extractor["route_separation_pair_count"],
        control_resamples=statistics["control_resamples"],
        bootstrap_resamples=statistics["bootstrap_resamples"],
        seed=statistics["seed"],
        gates=manifest["gates"],
    )

    summary_json = output_paths["summary_json"]
    summary_markdown = output_paths["summary_markdown"]
    summary = {
        "version": 2,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": manifest["scope"],
        "repository": repository,
        "runtime": runtime,
        "protocol": {
            "path": str(manifest_path),
            "sha256": protocol_sha256,
            "payload": manifest,
        },
        "inputs": locked_inputs,
        "samples": samples,
        "feature_cache": {
            "path": str(feature_cache_path),
            **feature_cache,
            "feature_shape": list(features.shape),
        },
        "audit": audit,
        "outputs": {
            "summary_json": str(summary_json),
            "summary_markdown": str(summary_markdown),
        },
        "limitations": [
            "This is dirty discovery on 32 images from a step-50000 checkpoint, not paper evidence.",
            "DINOv2 measures decoded clean-image patch similarity and never enters ProMoE training.",
            "A route-semantic association cannot establish that RCL caused it or that routing lowers denoising error.",
        ],
    }
    write_json_atomic(summary_json, summary)
    write_text_atomic(summary_markdown, render_summary_markdown(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether locked historical ProMoE routes agree with "
            "independent DINOv2 patch semantics."
        )
    )
    parser.add_argument("--route-ids", type=Path, required=True)
    parser.add_argument("--capture-summary", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--dino-path", type=Path, required=True)
    parser.add_argument("--dino-source-path", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    summary = run_locked_semantic_audit(
        route_ids_path=args.route_ids,
        capture_summary_path=args.capture_summary,
        latent_root=args.latent_root,
        dino_path=args.dino_path,
        dino_source_path=args.dino_source_path,
        vae_path=args.vae_path,
        output_dir=args.output_dir,
        feature_cache_path=args.feature_cache,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "independent_semantic_structure_supported": summary["audit"][
                    "independent_semantic_structure_supported"
                ],
                "summary_json": summary["outputs"]["summary_json"],
                "summary_markdown": summary["outputs"]["summary_markdown"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
