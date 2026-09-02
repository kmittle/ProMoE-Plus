"""Test whether DINO patch neighborhoods predict counterfactual expert utility."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from preprocess.build_dino_route_table import (
    LATENT_SCALE,
    _decode_model_latents,
)
from repa.encoder import extract_teacher_features, load_teacher_encoder
from utils import load_vae


PROBE_VERSION = 1
SUPPORTED_SOURCE_PROBE_VERSION = 1
SUPPORTED_SOURCE_MANIFEST_VERSION = 1
EXPECTED_PATCH_TOKENS = 256
METHODS = (
    "dino_correct",
    "dino_wrong_image",
    "dino_spatial_shift",
    "router_scores",
    "random",
)
PRIMARY_METHOD = "dino_correct"
SEMANTIC_CONTROLS = ("dino_wrong_image", "dino_spatial_shift")
METRICS = (
    "utility_spearman",
    "oracle_top1_rate",
    "capacity_additive_gain_relative",
)
DEFAULT_REQUIREMENTS = {
    "minimum_mean_dino_spearman": 0.10,
    "minimum_dino_minus_control_spearman": 0.05,
    "minimum_positive_images": 16,
    "minimum_positive_cells_per_control": 7,
    "minimum_capacity_additive_gain_relative": 1e-5,
    "require_dino_spearman_ci_lower_positive": True,
    "require_control_delta_ci_lower_positive": True,
    "require_capacity_gain_ci_lower_positive": True,
}


@dataclass(frozen=True)
class UtilityCell:
    block_index: int
    sigma: float
    native_mse: float
    exact_changes: np.ndarray
    native_experts: np.ndarray
    router_scores: np.ndarray


@dataclass(frozen=True)
class UtilityCase:
    case_id: str
    source_path: Path
    source_sha256: str
    source_aggregate_path: Path
    source_aggregate_sha256: str
    source_manifest_path: Path
    source_manifest_sha256: str
    latent_path: Path
    latent_sha256: str
    latent_key: str
    label: int
    seed: int
    model_name: str
    config: str
    checkpoint: str
    checkpoint_step: int
    checkpoint_state: str
    weights_checkpoint: str
    weights_checkpoint_step: int
    source_device: str
    source_probe_version: int
    token_indices: np.ndarray
    cells: tuple[UtilityCell, ...]


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{name} must be {qualifier}finite")
    return value


def _finite_array(value: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}D array")
    return array


def _validate_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hex digits")
    return value


def _load_case(path: Path) -> UtilityCase:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read utility result {path}: {error}") from error

    token_indices = np.asarray(payload.get("token_indices"), dtype=np.int64)
    if (
        token_indices.ndim != 1
        or token_indices.size == 0
        or len(set(token_indices.tolist())) != token_indices.size
        or int(token_indices.min()) < 0
    ):
        raise ValueError(f"{path}: token_indices must be unique nonnegative integers")

    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError(f"{path}: cells must be a nonempty list")
    cells = []
    seen_cells = set()
    num_experts = None
    for cell_index, raw_cell in enumerate(raw_cells):
        block_index = _require_int(
            raw_cell.get("block_index"),
            f"{path}: cells[{cell_index}].block_index",
        )
        sigma = _finite_float(
            raw_cell.get("sigma"),
            f"{path}: cells[{cell_index}].sigma",
            positive=True,
        )
        key = (block_index, sigma)
        if key in seen_cells:
            raise ValueError(f"{path}: duplicate cell {key}")
        seen_cells.add(key)
        native_mse = _finite_float(
            raw_cell.get("native_mse"),
            f"{path}: cells[{cell_index}].native_mse",
            positive=True,
        )
        raw_tokens = raw_cell.get("tokens")
        if not isinstance(raw_tokens, list):
            raise ValueError(f"{path}: cells[{cell_index}].tokens must be a list")
        token_map = {}
        for raw_token in raw_tokens:
            token_index = _require_int(
                raw_token.get("token_index"),
                f"{path}: token_index",
            )
            if token_index in token_map:
                raise ValueError(f"{path}: duplicate token {token_index} in cell {key}")
            token_map[token_index] = raw_token
        if set(token_map) != set(token_indices.tolist()):
            raise ValueError(f"{path}: cell {key} does not match top-level tokens")

        exact_rows = []
        router_rows = []
        native_rows = []
        for token_index in token_indices.tolist():
            raw_token = token_map[token_index]
            exact = _finite_array(
                raw_token.get("exact_mse_changes"),
                f"{path}: cell {key} token {token_index} exact_mse_changes",
                1,
            )
            router = _finite_array(
                raw_token.get("router_scores"),
                f"{path}: cell {key} token {token_index} router_scores",
                1,
            )
            if exact.size < 2 or router.shape != exact.shape:
                raise ValueError(f"{path}: malformed expert vectors in cell {key}")
            if num_experts is None:
                num_experts = int(exact.size)
            elif exact.size != num_experts:
                raise ValueError(f"{path}: inconsistent expert count")
            native = _require_int(
                raw_token.get("native_expert"),
                f"{path}: cell {key} token {token_index} native_expert",
            )
            if native >= exact.size:
                raise ValueError(f"{path}: native expert is outside the candidate set")
            if abs(float(exact[native])) > 1e-12:
                raise ValueError(f"{path}: native route must have zero exact change")
            if int(np.argmax(router)) != native:
                raise ValueError(f"{path}: router argmax disagrees with native expert")
            exact_rows.append(exact)
            router_rows.append(router)
            native_rows.append(native)
        cells.append(UtilityCell(
            block_index=block_index,
            sigma=sigma,
            native_mse=native_mse,
            exact_changes=np.stack(exact_rows),
            native_experts=np.asarray(native_rows, dtype=np.int64),
            router_scores=np.stack(router_rows),
        ))

    latent_path = Path(payload.get("latent", "")).expanduser().resolve()
    if not latent_path.is_file():
        raise FileNotFoundError(f"{path}: latent does not exist: {latent_path}")
    checkpoint_step = _require_int(
        payload.get("checkpoint_step"),
        f"{path}: checkpoint_step",
        minimum=1,
    )
    weights_checkpoint_step = _require_int(
        payload.get("weights_checkpoint_step"),
        f"{path}: weights_checkpoint_step",
        minimum=1,
    )
    if weights_checkpoint_step != checkpoint_step:
        raise ValueError(f"{path}: canonical and loaded checkpoint steps differ")
    source_probe_version = _require_int(
        payload.get("timestep_utility_probe_version"),
        f"{path}: timestep_utility_probe_version",
        minimum=1,
    )
    if source_probe_version != SUPPORTED_SOURCE_PROBE_VERSION:
        raise ValueError(
            f"{path}: unsupported timestep_utility_probe_version "
            f"{source_probe_version}; expected "
            f"{SUPPORTED_SOURCE_PROBE_VERSION}"
        )
    seed = _require_int(payload.get("seed"), f"{path}: seed")
    label = _require_int(payload.get("label"), f"{path}: label")
    latent_key = payload.get("latent_key")
    model_name = payload.get("model_name")
    config = payload.get("config")
    checkpoint = payload.get("checkpoint")
    checkpoint_state = payload.get("checkpoint_state")
    weights_checkpoint = payload.get("weights_checkpoint")
    source_device = payload.get("device")
    if not all(isinstance(value, str) and value for value in (
        latent_key,
        model_name,
        config,
        checkpoint,
        checkpoint_state,
        weights_checkpoint,
        source_device,
    )):
        raise ValueError(f"{path}: source provenance metadata is incomplete")
    if source_device != "cpu":
        raise ValueError(
            f"{path}: exact latent replay currently requires source device "
            f"'cpu', found {source_device!r}"
        )

    return UtilityCase(
        case_id=path.stem,
        source_path=path.resolve(),
        source_sha256=_sha256_file(path),
        source_aggregate_path=Path(),
        source_aggregate_sha256="",
        source_manifest_path=Path(),
        source_manifest_sha256="",
        latent_path=latent_path,
        latent_sha256="",
        latent_key=latent_key,
        label=label,
        seed=seed,
        model_name=model_name,
        config=str(Path(config).expanduser().resolve()),
        checkpoint=str(Path(checkpoint).expanduser().resolve()),
        checkpoint_step=checkpoint_step,
        checkpoint_state=checkpoint_state,
        weights_checkpoint=str(
            Path(weights_checkpoint).expanduser().resolve()
        ),
        weights_checkpoint_step=weights_checkpoint_step,
        source_device=source_device,
        source_probe_version=source_probe_version,
        token_indices=token_indices,
        cells=tuple(sorted(cells, key=lambda cell: (cell.block_index, cell.sigma))),
    )


def _load_source_manifest(
    result_dir: Path,
    cases: tuple[UtilityCase, ...],
) -> tuple[UtilityCase, ...]:
    aggregate_path = result_dir / "aggregate.json"
    try:
        with aggregate_path.open("r", encoding="utf-8") as handle:
            aggregate = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read locked source aggregate {aggregate_path}: {error}"
        ) from error
    if not isinstance(aggregate, dict):
        raise ValueError("Source aggregate must be a JSON object")

    manifest_value = aggregate.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ValueError("Source aggregate does not name its locked manifest")
    manifest_path = Path(manifest_value).expanduser().resolve(strict=True)
    expected_manifest_sha256 = _validate_sha256(
        aggregate.get("manifest_sha256"),
        "source aggregate manifest_sha256",
    )
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "Source manifest SHA-256 mismatch: expected "
            f"{expected_manifest_sha256}, found {actual_manifest_sha256}"
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read locked source manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("Source manifest must be a JSON object")
    manifest_version = manifest.get("version")
    if (
        isinstance(manifest_version, bool)
        or not isinstance(manifest_version, int)
        or manifest_version != SUPPORTED_SOURCE_MANIFEST_VERSION
    ):
        raise ValueError(
            "Unsupported source manifest version: expected "
            f"{SUPPORTED_SOURCE_MANIFEST_VERSION}, found "
            f"{manifest_version!r}"
        )
    raw_records = manifest.get("cases")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Source manifest cases must be a nonempty list")
    records = {}
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise ValueError(f"Source manifest case {index} must be an object")
        case_id = record.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in records:
            raise ValueError("Source manifest case IDs must be unique strings")
        records[case_id] = record

    aggregate_case_count = aggregate.get("num_cases")
    aggregate_rows = aggregate.get("per_image")
    if (
        isinstance(aggregate_case_count, bool)
        or aggregate_case_count != len(cases)
        or not isinstance(aggregate_rows, list)
    ):
        raise ValueError("Source aggregate case count does not match source results")
    aggregate_case_ids = []
    for row in aggregate_rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise ValueError("Source aggregate per_image rows are malformed")
        aggregate_case_ids.append(row["case_id"])
    loaded_case_ids = [case.case_id for case in cases]
    if (
        len(aggregate_case_ids) != len(cases)
        or len(set(aggregate_case_ids)) != len(aggregate_case_ids)
        or set(aggregate_case_ids) != set(loaded_case_ids)
    ):
        raise ValueError("Source aggregate image IDs do not match source results")

    aggregate_sha256 = _sha256_file(aggregate_path)
    validated = []
    for case in cases:
        record = records.get(case.case_id)
        if record is None:
            raise ValueError(
                f"Source manifest does not contain case {case.case_id}"
            )
        if record.get("label") != case.label or record.get("seed") != case.seed:
            raise ValueError(
                f"Source manifest label/seed mismatch for {case.case_id}"
            )
        manifest_latent_value = record.get("latent")
        if not isinstance(manifest_latent_value, str) or not manifest_latent_value:
            raise ValueError(
                f"Source manifest latent is missing for {case.case_id}"
            )
        manifest_latent = Path(manifest_latent_value)
        if manifest_latent.is_absolute():
            path_matches = manifest_latent.resolve() == case.latent_path
        else:
            path_matches = case.latent_path.as_posix().endswith(
                "/" + manifest_latent.as_posix()
            )
        if not path_matches:
            raise ValueError(
                f"Source manifest latent path mismatch for {case.case_id}"
            )
        expected_latent_sha256 = _validate_sha256(
            record.get("latent_sha256"),
            f"source manifest latent_sha256 for {case.case_id}",
        )
        actual_latent_sha256 = _sha256_file(case.latent_path)
        if actual_latent_sha256 != expected_latent_sha256:
            raise ValueError(
                f"Latent SHA-256 mismatch for {case.case_id}: expected "
                f"{expected_latent_sha256}, found {actual_latent_sha256}"
            )
        validated.append(replace(
            case,
            source_aggregate_path=aggregate_path.resolve(),
            source_aggregate_sha256=aggregate_sha256,
            source_manifest_path=manifest_path,
            source_manifest_sha256=actual_manifest_sha256,
            latent_sha256=actual_latent_sha256,
        ))
    return tuple(validated)


def load_utility_cases(
    result_dir: str | Path,
    *,
    expected_cases: int | None = None,
) -> tuple[UtilityCase, ...]:
    result_dir = Path(result_dir).expanduser().resolve(strict=True)
    if not result_dir.is_dir():
        raise ValueError(f"result_dir is not a directory: {result_dir}")
    paths = sorted(result_dir.glob("class*.json"))
    if expected_cases is not None and len(paths) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} class results, found {len(paths)}"
        )
    if len(paths) < 3:
        raise ValueError("At least three independent image cases are required")
    cases = tuple(_load_case(path) for path in paths)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Case identifiers must be unique")
    if len({str(case.latent_path) for case in cases}) != len(cases):
        raise ValueError("Each case must use a distinct latent")
    if len({case.seed for case in cases}) != len(cases):
        raise ValueError("Each case must use a distinct seed")
    reference_tokens = cases[0].token_indices.size
    reference_cells = tuple(
        (cell.block_index, cell.sigma) for cell in cases[0].cells
    )
    reference_experts = cases[0].cells[0].exact_changes.shape[1]
    for case in cases[1:]:
        if case.token_indices.size != reference_tokens:
            raise ValueError("Every case must use the same token count")
        if tuple((cell.block_index, cell.sigma) for cell in case.cells) != reference_cells:
            raise ValueError("Every case must use the same block/sigma grid")
        if any(
            cell.exact_changes.shape[1] != reference_experts
            for cell in case.cells
        ):
            raise ValueError("Every case must use the same expert count")
    identity_fields = (
        "model_name",
        "config",
        "checkpoint",
        "checkpoint_step",
        "checkpoint_state",
        "weights_checkpoint",
        "weights_checkpoint_step",
        "source_device",
        "source_probe_version",
    )
    for field in identity_fields:
        if len({getattr(case, field) for case in cases}) != 1:
            raise ValueError(
                f"Every case must have the same source provenance field {field}"
            )
    return _load_source_manifest(result_dir, cases)


def _source_results_sha256(cases: tuple[UtilityCase, ...]) -> str:
    """Seal result contents using sorted case ID and per-file SHA-256 pairs."""
    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda item: item.case_id):
        digest.update(case.case_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(case.source_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_source_seals(
    cases: tuple[UtilityCase, ...],
    *,
    expected_aggregate_sha256: str,
    expected_results_sha256: str,
) -> str:
    expected_aggregate_sha256 = _validate_sha256(
        expected_aggregate_sha256,
        "Expected source aggregate SHA-256",
    )
    expected_results_sha256 = _validate_sha256(
        expected_results_sha256,
        "Expected source results SHA-256",
    )
    actual_aggregate_sha256 = cases[0].source_aggregate_sha256
    if actual_aggregate_sha256 != expected_aggregate_sha256:
        raise ValueError(
            "Source aggregate SHA-256 mismatch: expected "
            f"{expected_aggregate_sha256}, found {actual_aggregate_sha256}"
        )
    actual_results_sha256 = _source_results_sha256(cases)
    if actual_results_sha256 != expected_results_sha256:
        raise ValueError(
            "Source utility-results SHA-256 mismatch: expected "
            f"{expected_results_sha256}, found {actual_results_sha256}"
        )
    return actual_results_sha256


def _sample_model_latent(case: UtilityCase) -> torch.Tensor:
    if case.source_device != "cpu":
        raise ValueError(
            "Exact replay is only supported for utility results generated on CPU"
        )
    actual_latent_sha256 = _sha256_file(case.latent_path)
    if actual_latent_sha256 != case.latent_sha256:
        raise ValueError(
            f"Latent SHA-256 changed before replay for {case.case_id}: "
            f"expected {case.latent_sha256}, found {actual_latent_sha256}"
        )
    with np.load(case.latent_path, allow_pickle=False) as archive:
        if case.latent_key not in archive.files:
            raise KeyError(
                f"{case.latent_path} lacks latent key {case.latent_key!r}"
            )
        parameters = np.asarray(archive[case.latent_key], dtype=np.float32)
    if parameters.ndim != 3 or parameters.shape[0] % 2 != 0:
        raise ValueError(
            f"Expected VAE parameters shaped [2C,H,W], got {parameters.shape}"
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(case.seed)
        posterior = DiagonalGaussianDistribution(
            torch.from_numpy(parameters).unsqueeze(0)
        )
        return posterior.sample().mul_(LATENT_SCALE)


def extract_dino_feature_maps(
    cases: tuple[UtilityCase, ...],
    *,
    dino_path: str | Path,
    vae_path: str | Path,
    device: str = "cuda:0",
    batch_size: int = 4,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dino_path = Path(dino_path).expanduser().resolve(strict=True)
    vae_path = Path(vae_path).expanduser().resolve(strict=True)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    vae = load_vae(
        "stabilityai/sd-vae-ft-mse",
        vae_path=str(vae_path),
    ).to(torch_device).eval().requires_grad_(False)
    encoder, _ = load_teacher_encoder(
        "dinov2-vit-b",
        resolution=256,
        local_root=str(dino_path.parent.parent),
        enc_path=str(dino_path),
    )
    encoder = encoder.to(torch_device).eval().requires_grad_(False)

    if any(
        int(case.token_indices.max()) >= EXPECTED_PATCH_TOKENS for case in cases
    ):
        raise ValueError(
            f"Utility token indices must be below {EXPECTED_PATCH_TOKENS}"
        )
    feature_parts = []
    image_statistics = []
    with torch.inference_mode():
        for start in range(0, len(cases), batch_size):
            batch_cases = cases[start : start + batch_size]
            model_latents = torch.cat([
                _sample_model_latent(case) for case in batch_cases
            ]).to(torch_device)
            decoded = _decode_model_latents(vae, model_latents).float()
            images = (decoded.clamp(-1, 1) + 1.0) / 2.0
            features = extract_teacher_features(
                encoder, images, "dinov2-vit-b"
            ).float()
            if features.ndim != 3 or features.shape[0] != len(batch_cases):
                raise RuntimeError(
                    f"Unexpected DINO feature shape: {tuple(features.shape)}"
                )
            if features.shape[1] != EXPECTED_PATCH_TOKENS:
                raise RuntimeError(
                    f"DINO returned {features.shape[1]} patches; expected "
                    f"{EXPECTED_PATCH_TOKENS} to match the ProMoE token grid"
                )
            features = F.normalize(features, p=2, dim=-1)
            feature_parts.append(features.cpu().numpy().astype(np.float32))
            for case, image in zip(batch_cases, images):
                image_statistics.append({
                    "case_id": case.case_id,
                    "rgb_mean": float(image.mean().item()),
                    "rgb_std": float(image.std().item()),
                })
            print(
                f"extracted DINO patches for {start + len(batch_cases)}/{len(cases)} cases",
                flush=True,
            )

    feature_maps = np.concatenate(feature_parts, axis=0)
    del encoder, vae
    gc.collect()
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return feature_maps, image_statistics


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = _finite_array(values, "features", 2)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Feature rows must have nonzero norm")
    return values / norms


def _utility_rank_profiles(exact_changes: np.ndarray) -> np.ndarray:
    exact_changes = _finite_array(exact_changes, "exact_changes", 2)
    utility = -exact_changes
    profiles = np.stack([
        rankdata(row, method="average") for row in utility
    ])
    profiles -= profiles.mean(axis=1, keepdims=True)
    return _normalize_rows(profiles)


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = _finite_array(left, "left", 1)
    right = _finite_array(right, "right", 1)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("Spearman vectors must have the same nontrivial shape")
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    if denominator <= 1e-12:
        return None
    return float(np.dot(left_rank, right_rank) / denominator)


def leave_one_image_knn_predict(
    features: np.ndarray,
    utility_profiles: np.ndarray,
    case_indices: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = _normalize_rows(features)
    utility_profiles = _finite_array(
        utility_profiles, "utility_profiles", 2
    )
    case_indices = np.asarray(case_indices, dtype=np.int64)
    if (
        utility_profiles.shape[0] != features.shape[0]
        or case_indices.shape != (features.shape[0],)
    ):
        raise ValueError("Features, profiles, and case indices must align")
    if k < 1:
        raise ValueError("k must be positive")
    similarities = features @ features.T
    predictions = np.empty_like(utility_profiles)
    neighbors = np.empty((features.shape[0], k), dtype=np.int64)
    indices = np.arange(features.shape[0])
    for row in range(features.shape[0]):
        eligible = indices[case_indices != case_indices[row]]
        if eligible.size < k:
            raise ValueError(
                f"Only {eligible.size} cross-image neighbors are available for k={k}"
            )
        order = np.lexsort((eligible, -similarities[row, eligible]))
        selected = eligible[order[:k]]
        neighbors[row] = selected
        predictions[row] = utility_profiles[selected].mean(axis=0)
    return predictions, neighbors


def _deranged_donors(num_cases: int, seed: int) -> np.ndarray:
    if num_cases < 2:
        raise ValueError("A wrong-image control needs at least two cases")
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_cases)
    donors = np.empty(num_cases, dtype=np.int64)
    donors[order] = np.roll(order, -1)
    if np.any(donors == np.arange(num_cases)):
        raise RuntimeError("The wrong-image mapping must be a derangement")
    return donors


def _shift_indices(
    token_indices: np.ndarray,
    num_patches: int,
    shift_y: int,
    shift_x: int,
) -> np.ndarray:
    side = math.isqrt(num_patches)
    if side * side != num_patches:
        raise ValueError("DINO patch count must form a square grid")
    if shift_y % side == 0 and shift_x % side == 0:
        raise ValueError("The spatial control shift must be nonzero modulo the grid")
    token_indices = np.asarray(token_indices, dtype=np.int64)
    if (
        token_indices.ndim != 1
        or token_indices.size == 0
        or token_indices.min() < 0
        or token_indices.max() >= num_patches
    ):
        raise ValueError("token_indices are outside the DINO patch grid")
    rows = token_indices // side
    columns = token_indices % side
    return ((rows + shift_y) % side) * side + (columns + shift_x) % side


def _capacity_assignment(
    predicted_profiles: np.ndarray,
    native_experts: np.ndarray,
) -> np.ndarray:
    predicted_profiles = _finite_array(
        predicted_profiles, "predicted_profiles", 2
    )
    native_experts = np.asarray(native_experts, dtype=np.int64)
    num_tokens, num_experts = predicted_profiles.shape
    if native_experts.shape != (num_tokens,):
        raise ValueError("Native experts must align with predicted profiles")
    if native_experts.min() < 0 or native_experts.max() >= num_experts:
        raise ValueError("Native experts are outside the candidate set")
    capacities = np.bincount(native_experts, minlength=num_experts)
    slots = np.repeat(np.arange(num_experts), capacities)
    rows, slot_indices = linear_sum_assignment(-predicted_profiles[:, slots])
    if rows.size != num_tokens:
        raise RuntimeError("Capacity assignment did not cover every token")
    assignment = np.empty(num_tokens, dtype=np.int64)
    assignment[rows] = slots[slot_indices]
    if not np.array_equal(
        np.bincount(assignment, minlength=num_experts), capacities
    ):
        raise RuntimeError("Capacity assignment changed expert token counts")
    return assignment


def _bootstrap_ci(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    values = _finite_array(values, "bootstrap values", 1)
    if values.size < 2 or resamples < 1000:
        raise ValueError("Bootstrap requires at least two values and 1000 resamples")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk_size = 10000
    for start in range(0, resamples, chunk_size):
        count = min(chunk_size, resamples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _group_payload(
    cases: tuple[UtilityCase, ...],
    cell_position: int,
    feature_maps: np.ndarray,
    wrong_maps: np.ndarray,
    shifted_maps: np.ndarray,
    random_maps: np.ndarray,
) -> dict[str, np.ndarray]:
    correct_features = []
    wrong_features = []
    shifted_features = []
    random_features = []
    router_scores = []
    exact_changes = []
    native_experts = []
    case_indices = []
    token_indices = []
    for case_index, case in enumerate(cases):
        cell = case.cells[cell_position]
        tokens = case.token_indices
        correct_features.append(feature_maps[case_index, tokens])
        wrong_features.append(wrong_maps[case_index, tokens])
        shifted_features.append(shifted_maps[case_index, tokens])
        random_features.append(random_maps[case_index, tokens])
        router_scores.append(cell.router_scores)
        exact_changes.append(cell.exact_changes / cell.native_mse)
        native_experts.append(cell.native_experts)
        case_indices.append(np.full(tokens.size, case_index, dtype=np.int64))
        token_indices.append(tokens)
    return {
        "dino_correct": np.concatenate(correct_features),
        "dino_wrong_image": np.concatenate(wrong_features),
        "dino_spatial_shift": np.concatenate(shifted_features),
        "router_scores": np.concatenate(router_scores),
        "random": np.concatenate(random_features),
        "exact_changes_relative": np.concatenate(exact_changes),
        "native_experts": np.concatenate(native_experts),
        "case_indices": np.concatenate(case_indices),
        "token_indices": np.concatenate(token_indices),
    }


def _evaluate_group(
    group: dict[str, np.ndarray],
    *,
    method: str,
    k: int,
    num_cases: int,
) -> dict[str, Any]:
    exact_changes = group["exact_changes_relative"]
    profiles = _utility_rank_profiles(exact_changes)
    predictions, neighbors = leave_one_image_knn_predict(
        group[method], profiles, group["case_indices"], k=k
    )
    correlations = np.asarray([
        np.nan if (value := _spearman(prediction, -actual)) is None else value
        for prediction, actual in zip(predictions, exact_changes)
    ])
    predicted_top1 = predictions.argmax(axis=1)
    actual_top1 = exact_changes.argmin(axis=1)
    top1 = predicted_top1 == actual_top1

    per_case = []
    for case_index in range(num_cases):
        selection = np.flatnonzero(group["case_indices"] == case_index)
        if selection.size == 0:
            raise RuntimeError(f"Missing records for case {case_index}")
        assignment = _capacity_assignment(
            predictions[selection], group["native_experts"][selection]
        )
        rows = np.arange(selection.size)
        selected_changes = exact_changes[selection][rows, assignment]
        native_changes = exact_changes[selection][
            rows, group["native_experts"][selection]
        ]
        valid_correlations = correlations[selection][
            np.isfinite(correlations[selection])
        ]
        if valid_correlations.size == 0:
            raise RuntimeError("A case has no valid utility correlations")
        per_case.append({
            "utility_spearman": float(valid_correlations.mean()),
            "oracle_top1_rate": float(top1[selection].mean()),
            "capacity_additive_gain_relative": float(
                -(selected_changes - native_changes).sum()
            ),
        })
    if np.any(
        group["case_indices"][neighbors]
        == group["case_indices"][:, None]
    ):
        raise RuntimeError("A KNN prediction leaked a same-image neighbor")
    return {
        "per_case": per_case,
        "record_utility_spearman": correlations,
        "record_top1": top1.astype(np.float64),
        "neighbor_case_indices": group["case_indices"][neighbors],
    }


def _summarize_method(
    case_ids: list[str],
    cell_keys: list[tuple[int, float]],
    group_results: list[dict[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    method_index: int,
) -> dict[str, Any]:
    per_case = []
    for case_index, case_id in enumerate(case_ids):
        row = {"case_id": case_id}
        for metric in METRICS:
            row[metric] = float(np.mean([
                result["per_case"][case_index][metric]
                for result in group_results
            ]))
        per_case.append(row)

    summary = {}
    for metric_index, metric in enumerate(METRICS):
        values = np.asarray([row[metric] for row in per_case])
        summary[metric] = {
            "mean_over_images": float(values.mean()),
            "bootstrap_ci95": _bootstrap_ci(
                values,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + method_index * 100 + metric_index,
            ),
        }
    per_cell = []
    for key, result in zip(cell_keys, group_results):
        case_rows = result["per_case"]
        per_cell.append({
            "block_index": key[0],
            "sigma": key[1],
            **{
                metric: float(np.mean([row[metric] for row in case_rows]))
                for metric in METRICS
            },
        })
    return {
        "summary": summary,
        "per_case": per_case,
        "per_cell": per_cell,
    }


def _paired_comparisons(
    method_results: dict[str, dict[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    primary_rows = method_results[PRIMARY_METHOD]["per_case"]
    comparisons = {}
    for control_index, control in enumerate(METHODS[1:]):
        control_rows = method_results[control]["per_case"]
        metric_payload = {}
        for metric_index, metric in enumerate(METRICS):
            differences = np.asarray([
                primary[metric] - baseline[metric]
                for primary, baseline in zip(primary_rows, control_rows)
            ])
            metric_payload[metric] = {
                "mean_paired_difference": float(differences.mean()),
                "bootstrap_ci95": _bootstrap_ci(
                    differences,
                    resamples=bootstrap_resamples,
                    seed=(
                        bootstrap_seed + 1000 + control_index * 100
                        + metric_index
                    ),
                ),
                "positive_images": int((differences > 0).sum()),
            }
        primary_cells = method_results[PRIMARY_METHOD]["per_cell"]
        control_cells = method_results[control]["per_cell"]
        cell_differences = []
        for primary, baseline in zip(primary_cells, control_cells):
            if (primary["block_index"], primary["sigma"]) != (
                baseline["block_index"], baseline["sigma"]
            ):
                raise RuntimeError("Per-cell summaries are misaligned")
            cell_differences.append({
                "block_index": primary["block_index"],
                "sigma": primary["sigma"],
                "utility_spearman_difference": float(
                    primary["utility_spearman"]
                    - baseline["utility_spearman"]
                ),
            })
        comparisons[control] = {
            "metrics": metric_payload,
            "per_cell": cell_differences,
            "positive_spearman_cells": int(sum(
                row["utility_spearman_difference"] > 0
                for row in cell_differences
            )),
        }
    return comparisons


def _gate_decision(
    method_results: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
    requirements: dict[str, Any],
) -> dict[str, Any]:
    dino_summary = method_results[PRIMARY_METHOD]["summary"]
    checks = {}

    def record(name: str, observed: Any, threshold: Any, passed: bool) -> None:
        checks[name] = {
            "observed": observed,
            "threshold": threshold,
            "passed": bool(passed),
        }

    dino_spearman = dino_summary["utility_spearman"]["mean_over_images"]
    dino_spearman_lower = dino_summary["utility_spearman"]["bootstrap_ci95"][0]
    capacity_gain = dino_summary[
        "capacity_additive_gain_relative"
    ]["mean_over_images"]
    capacity_lower = dino_summary[
        "capacity_additive_gain_relative"
    ]["bootstrap_ci95"][0]
    record(
        "minimum_mean_dino_spearman",
        dino_spearman,
        requirements["minimum_mean_dino_spearman"],
        dino_spearman >= requirements["minimum_mean_dino_spearman"],
    )
    record(
        "dino_spearman_ci_lower_positive",
        dino_spearman_lower,
        "> 0",
        (
            not requirements["require_dino_spearman_ci_lower_positive"]
            or dino_spearman_lower > 0
        ),
    )
    record(
        "minimum_capacity_additive_gain_relative",
        capacity_gain,
        requirements["minimum_capacity_additive_gain_relative"],
        capacity_gain
        >= requirements["minimum_capacity_additive_gain_relative"],
    )
    record(
        "capacity_gain_ci_lower_positive",
        capacity_lower,
        "> 0",
        (
            not requirements["require_capacity_gain_ci_lower_positive"]
            or capacity_lower > 0
        ),
    )
    primary_rows = method_results[PRIMARY_METHOD]["per_case"]
    semantic_control_rows = [
        method_results[control]["per_case"] for control in SEMANTIC_CONTROLS
    ]
    robust_image_differences = np.asarray([
        primary["utility_spearman"] - max(
            control_rows[index]["utility_spearman"]
            for control_rows in semantic_control_rows
        )
        for index, primary in enumerate(primary_rows)
    ])
    record(
        "minimum_positive_images_against_both_semantic_controls",
        int((robust_image_differences > 0).sum()),
        requirements["minimum_positive_images"],
        int((robust_image_differences > 0).sum())
        >= requirements["minimum_positive_images"],
    )
    for control in SEMANTIC_CONTROLS:
        comparison = comparisons[control]
        delta = comparison["metrics"]["utility_spearman"]
        record(
            f"minimum_spearman_advantage_over_{control}",
            delta["mean_paired_difference"],
            requirements["minimum_dino_minus_control_spearman"],
            delta["mean_paired_difference"]
            >= requirements["minimum_dino_minus_control_spearman"],
        )
        record(
            f"spearman_advantage_ci_lower_over_{control}",
            delta["bootstrap_ci95"][0],
            "> 0",
            (
                not requirements["require_control_delta_ci_lower_positive"]
                or delta["bootstrap_ci95"][0] > 0
            ),
        )
        record(
            f"positive_spearman_cells_over_{control}",
            comparison["positive_spearman_cells"],
            requirements["minimum_positive_cells_per_control"],
            comparison["positive_spearman_cells"]
            >= requirements["minimum_positive_cells_per_control"],
        )
    return {
        "requirements": requirements,
        "checks": checks,
        "passed": bool(all(check["passed"] for check in checks.values())),
    }


def evaluate_neighborhoods(
    cases: tuple[UtilityCase, ...],
    feature_maps: np.ndarray,
    *,
    k: int = 8,
    wrong_image_seed: int = 2026090301,
    random_seed: int = 2026090302,
    shift_y: int = 5,
    shift_x: int = 7,
    bootstrap_resamples: int = 200000,
    bootstrap_seed: int = 2026090303,
    requirements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if feature_maps.ndim != 3 or feature_maps.shape[0] != len(cases):
        raise ValueError("feature_maps must have shape [cases, patches, channels]")
    if not np.isfinite(feature_maps).all():
        raise ValueError("feature_maps contain non-finite values")
    if bootstrap_resamples < 1000:
        raise ValueError("bootstrap_resamples must be at least 1000")
    requirements = dict(DEFAULT_REQUIREMENTS if requirements is None else requirements)
    if set(requirements) != set(DEFAULT_REQUIREMENTS):
        raise ValueError("requirements must use the complete locked key set")

    donors = _deranged_donors(len(cases), wrong_image_seed)
    wrong_maps = feature_maps[donors]
    shifted_maps = np.empty_like(feature_maps)
    all_indices = np.arange(feature_maps.shape[1])
    shifted_indices = _shift_indices(
        all_indices, feature_maps.shape[1], shift_y, shift_x
    )
    shifted_maps[:, all_indices] = feature_maps[:, shifted_indices]
    rng = np.random.default_rng(random_seed)
    random_maps = rng.standard_normal(feature_maps.shape).astype(np.float32)

    cell_keys = [
        (cell.block_index, cell.sigma) for cell in cases[0].cells
    ]
    per_method_groups = {method: [] for method in METHODS}
    for cell_position, _ in enumerate(cell_keys):
        group = _group_payload(
            cases,
            cell_position,
            feature_maps,
            wrong_maps,
            shifted_maps,
            random_maps,
        )
        for method in METHODS:
            per_method_groups[method].append(_evaluate_group(
                group,
                method=method,
                k=k,
                num_cases=len(cases),
            ))

    case_ids = [case.case_id for case in cases]
    method_results = {
        method: _summarize_method(
            case_ids,
            cell_keys,
            group_results,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            method_index=method_index,
        )
        for method_index, (method, group_results) in enumerate(
            per_method_groups.items()
        )
    }
    comparisons = _paired_comparisons(
        method_results,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    gate = _gate_decision(method_results, comparisons, requirements)
    return {
        "method_results": method_results,
        "paired_dino_comparisons": comparisons,
        "gate": gate,
        "wrong_image_mapping": {
            cases[index].case_id: cases[int(donor)].case_id
            for index, donor in enumerate(donors)
        },
        "cell_keys": [
            {"block_index": block, "sigma": sigma}
            for block, sigma in cell_keys
        ],
    }


def _prepare_new_output_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = Path(os.path.abspath(path))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite locked result: {path}")
    return path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = _prepare_new_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def analyze_dino_utility_neighborhood(
    result_dir: str | Path,
    output: str | Path,
    *,
    dino_path: str | Path,
    vae_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
    expected_source_aggregate_sha256: str,
    expected_source_results_sha256: str,
    expected_dino_sha256: str,
    expected_vae_config_sha256: str,
    expected_vae_weights_sha256: str,
    device: str = "cuda:0",
    batch_size: int = 4,
    expected_cases: int = 24,
    k: int = 8,
    wrong_image_seed: int = 2026090301,
    random_seed: int = 2026090302,
    shift_y: int = 5,
    shift_x: int = 7,
    bootstrap_resamples: int = 200000,
    bootstrap_seed: int = 2026090303,
) -> dict[str, Any]:
    result_dir = Path(result_dir).expanduser().resolve(strict=True)
    output = _prepare_new_output_path(output)
    dino_path = Path(dino_path).expanduser().resolve(strict=True)
    vae_path = Path(vae_path).expanduser().resolve(strict=True)
    vae_config_path = (vae_path / "config.json").resolve(strict=True)
    vae_weights_path = (
        vae_path / "diffusion_pytorch_model.safetensors"
    ).resolve(strict=True)
    cases = load_utility_cases(result_dir, expected_cases=expected_cases)
    source_results_sha256 = _validate_source_seals(
        cases,
        expected_aggregate_sha256=expected_source_aggregate_sha256,
        expected_results_sha256=expected_source_results_sha256,
    )
    checkpoint_path = Path(cases[0].checkpoint).resolve(strict=True)
    weights_checkpoint_path = Path(
        cases[0].weights_checkpoint
    ).resolve(strict=True)
    config_path = Path(cases[0].config).resolve(strict=True)
    expected_hashes = {
        "checkpoint": expected_checkpoint_sha256,
        "weights_checkpoint": expected_checkpoint_sha256,
        "config": expected_config_sha256,
        "dino_weights": expected_dino_sha256,
        "vae_config": expected_vae_config_sha256,
        "vae_weights": expected_vae_weights_sha256,
    }
    for name, expected in expected_hashes.items():
        _validate_sha256(expected, f"Expected {name} SHA-256")
    actual_hashes = {
        "checkpoint": _sha256_file(checkpoint_path),
        "weights_checkpoint": _sha256_file(weights_checkpoint_path),
        "config": _sha256_file(config_path),
        "dino_weights": _sha256_file(dino_path),
        "vae_config": _sha256_file(vae_config_path),
        "vae_weights": _sha256_file(vae_weights_path),
    }
    for name, expected in expected_hashes.items():
        if actual_hashes[name] != expected:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {expected}, "
                f"found {actual_hashes[name]}"
            )
    feature_maps, image_statistics = extract_dino_feature_maps(
        cases,
        dino_path=dino_path,
        vae_path=vae_path,
        device=device,
        batch_size=batch_size,
    )
    evaluation = evaluate_neighborhoods(
        cases,
        feature_maps,
        k=k,
        wrong_image_seed=wrong_image_seed,
        random_seed=random_seed,
        shift_y=shift_y,
        shift_x=shift_x,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    payload = {
        "probe_version": PROBE_VERSION,
        "diagnostic_scope": (
            "frozen-checkpoint, leave-one-image-out semantic-neighborhood "
            "diagnostic; not a training, generation, or FID claim"
        ),
        "hypothesis": (
            "DINO-near patches from different images share expert-utility "
            "rankings at the same ProMoE block and noise level"
        ),
        "result_dir": str(result_dir),
        "output": str(output),
        "case_count": len(cases),
        "token_count_per_case": int(cases[0].token_indices.size),
        "expert_count": int(cases[0].cells[0].exact_changes.shape[1]),
        "checkpoint_step": cases[0].checkpoint_step,
        "checkpoint_state": cases[0].checkpoint_state,
        "source_probe_version": cases[0].source_probe_version,
        "model_name": cases[0].model_name,
        "config": str(config_path),
        "config_sha256": actual_hashes["config"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_hashes["checkpoint"],
        "weights_checkpoint": str(weights_checkpoint_path),
        "weights_checkpoint_sha256": actual_hashes["weights_checkpoint"],
        "source_device": cases[0].source_device,
        "source_aggregate": str(cases[0].source_aggregate_path),
        "source_aggregate_sha256": cases[0].source_aggregate_sha256,
        "source_results_sha256": source_results_sha256,
        "source_manifest": str(cases[0].source_manifest_path),
        "source_manifest_sha256": cases[0].source_manifest_sha256,
        "case_sources": [
            {
                "case_id": case.case_id,
                "result": str(case.source_path),
                "result_sha256": case.source_sha256,
                "latent": str(case.latent_path),
                "latent_sha256": case.latent_sha256,
                "label": case.label,
                "seed": case.seed,
            }
            for case in cases
        ],
        "dino_weights": str(dino_path),
        "dino_weights_sha256": actual_hashes["dino_weights"],
        "vae_path": str(vae_path),
        "vae_config": str(vae_config_path),
        "vae_config_sha256": actual_hashes["vae_config"],
        "vae_weights": str(vae_weights_path),
        "vae_weights_sha256": actual_hashes["vae_weights"],
        "latent_scale_for_model": LATENT_SCALE,
        "decode_rule": "vae.decode(model_latent / latent_scale)",
        "feature_shape": list(feature_maps.shape),
        "image_statistics": image_statistics,
        "protocol": {
            "primary_method": PRIMARY_METHOD,
            "semantic_controls": list(SEMANTIC_CONTROLS),
            "all_methods": list(METHODS),
            "leave_one_image_out": True,
            "k": k,
            "wrong_image_seed": wrong_image_seed,
            "random_seed": random_seed,
            "spatial_shift": [shift_y, shift_x],
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
            "capacity_metric": (
                "sum of held-out single-token exact MSE changes after a "
                "predicted assignment that preserves native expert counts"
            ),
        },
        **evaluation,
    }
    _atomic_write_json(output, payload)
    print(json.dumps({
        "output": str(output),
        "gate_passed": payload["gate"]["passed"],
        "dino_summary": payload["method_results"][PRIMARY_METHOD]["summary"],
    }, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dino-path",
        type=Path,
        default=Path("pretrained_ckpt/encoder/dinov2_vitb14/state_dict.pth"),
    )
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=Path("pretrained_ckpt/vae/stabilityai--sd-vae-ft-mse"),
    )
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-source-aggregate-sha256", required=True)
    parser.add_argument("--expected-source-results-sha256", required=True)
    parser.add_argument("--expected-dino-sha256", required=True)
    parser.add_argument("--expected-vae-config-sha256", required=True)
    parser.add_argument("--expected-vae-weights-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--expected-cases", type=int, default=24)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--wrong-image-seed", type=int, default=2026090301)
    parser.add_argument("--random-seed", type=int, default=2026090302)
    parser.add_argument("--shift-y", type=int, default=5)
    parser.add_argument("--shift-x", type=int, default=7)
    parser.add_argument("--bootstrap-resamples", type=int, default=200000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026090303)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    analyze_dino_utility_neighborhood(
        args.result_dir,
        args.output,
        dino_path=args.dino_path,
        vae_path=args.vae_path,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_config_sha256=args.expected_config_sha256,
        expected_source_aggregate_sha256=(
            args.expected_source_aggregate_sha256
        ),
        expected_source_results_sha256=args.expected_source_results_sha256,
        expected_dino_sha256=args.expected_dino_sha256,
        expected_vae_config_sha256=args.expected_vae_config_sha256,
        expected_vae_weights_sha256=args.expected_vae_weights_sha256,
        device=args.device,
        batch_size=args.batch_size,
        expected_cases=args.expected_cases,
        k=args.k,
        wrong_image_seed=args.wrong_image_seed,
        random_seed=args.random_seed,
        shift_y=args.shift_y,
        shift_x=args.shift_x,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
