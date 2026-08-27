"""Locked paired-bootstrap statistics for credit redistribution."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from .controller import BRANCHES
from .evaluator import (
    CHECKPOINT_STATES,
    EVALUATOR_VERSION,
    _load_reusable_case,
    _seal_path,
    load_heldout_manifest,
)
from .heldout import canonical_json_sha256
from .serialization import atomic_write_json, sha256_file


STATISTICS_VERSION = 1
BOOTSTRAP_RESAMPLES = 200_000
BOOTSTRAP_SEED = 2026082723
BOOTSTRAP_CHUNK_SIZE = 10_000
METRIC_CHUNK_SIZE = 256
CASE_COUNT = 128
BLOCK_COUNT = 6
EXPERT_COUNT = 12
TOKENS_PER_CASE = 24 * 256


def gini(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != EXPERT_COUNT:
        raise ValueError("Gini requires a 12-expert vector")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Gini inputs must be finite and nonnegative")
    ordered = np.sort(values, axis=-1)
    total = ordered.sum(axis=-1)
    weights = np.arange(1, EXPERT_COUNT + 1, dtype=np.float64)
    numerator = 2.0 * np.sum(ordered * weights, axis=-1)
    result = np.zeros_like(total, dtype=np.float64)
    positive = total > 0
    result[positive] = (
        numerator[positive] / (EXPERT_COUNT * total[positive])
        - (EXPERT_COUNT + 1.0) / EXPERT_COUNT
    )
    return result


def _metrics_from_aggregates(mse, credit, count):
    mse = np.asarray(mse, dtype=np.float64)
    credit = np.asarray(credit, dtype=np.float64)
    count = np.asarray(count, dtype=np.int64)
    if mse.shape[-1] != CASE_COUNT:
        raise ValueError("Metric aggregation requires 128 image units")
    if credit.shape[-3:] != (CASE_COUNT, BLOCK_COUNT, EXPERT_COUNT):
        raise ValueError("Credit matrix shape changed")
    if count.shape != credit.shape:
        raise ValueError("Credit/count matrix shapes differ")
    if np.any(~np.isfinite(mse)) or np.any(mse <= 0):
        raise ValueError("Per-image MSE must be finite and positive")
    if np.any(~np.isfinite(credit)) or np.any(credit < 0):
        raise ValueError("Per-image credit must be finite and nonnegative")
    if np.any(count < 0):
        raise ValueError("Per-image count must be nonnegative")

    mean_mse = mse.mean(axis=-1)
    total_credit = credit.sum(axis=-3)
    total_count = count.sum(axis=-3)
    if np.any(total_count.sum(axis=-1) <= 0):
        raise RuntimeError("A bootstrap aggregate has zero total count in a block")
    rates = np.divide(
        total_credit,
        total_count,
        out=np.zeros_like(total_credit),
        where=total_count > 0,
    )
    mean_gini = gini(rates).mean(axis=-1)
    count_mean = total_count.mean(axis=-1, dtype=np.float64)
    if np.any(count_mean <= 0):
        raise RuntimeError("A bootstrap aggregate has nonpositive mean token load")
    count_std = total_count.std(axis=-1, ddof=0, dtype=np.float64)
    mean_cv = (count_std / count_mean).mean(axis=-1)
    for name, value in (
        ("mean MSE", mean_mse),
        ("mean credit-rate Gini", mean_gini),
        ("mean token-load CV", mean_cv),
    ):
        if np.any(~np.isfinite(value)):
            raise FloatingPointError(f"{name} is nonfinite")
    return {
        "mse": mean_mse,
        "gini": mean_gini,
        "cv": mean_cv,
    }


def point_metrics(data):
    expanded = _metrics_from_aggregates(
        data["mse"].reshape(1, CASE_COUNT),
        data["credit"].reshape(1, CASE_COUNT, BLOCK_COUNT, EXPERT_COUNT),
        data["count"].reshape(1, CASE_COUNT, BLOCK_COUNT, EXPERT_COUNT),
    )
    return {name: float(value[0]) for name, value in expanded.items()}


def materialize_bootstrap_indices(
    path,
    *,
    resamples=BOOTSTRAP_RESAMPLES,
    case_count=CASE_COUNT,
    chunk_size=BOOTSTRAP_CHUNK_SIZE,
    seed=BOOTSTRAP_SEED,
):
    path = Path(path).resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        if not path.exists() or not sidecar.exists():
            raise RuntimeError("Bootstrap matrix and sidecar must exist together")
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if matrix.shape != (resamples, case_count) or matrix.dtype != np.int64:
            raise RuntimeError("Existing bootstrap matrix metadata differs")
        digest = sha256_file(path)
        if sidecar.read_text(encoding="utf-8") != digest + "\n":
            raise RuntimeError("Bootstrap matrix sidecar differs")
        # The sidecar authenticates only the current bytes.  Recreate the
        # locked PCG64 stream as well, so replacing both the matrix and its
        # sidecar cannot silently change the confidence intervals.
        generator = np.random.Generator(np.random.PCG64(seed))
        for start in range(0, resamples, chunk_size):
            end = min(start + chunk_size, resamples)
            expected = generator.integers(
                0,
                case_count,
                size=(end - start, case_count),
                dtype=np.int64,
                endpoint=False,
            )
            if not np.array_equal(matrix[start:end], expected):
                raise RuntimeError("Existing bootstrap matrix differs from locked PCG64 stream")
        return path, digest

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npy", dir=path.parent
    )
    os.close(descriptor)
    os.unlink(temporary_name)
    temporary = Path(temporary_name)
    try:
        matrix = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.int64,
            shape=(resamples, case_count),
        )
        generator = np.random.Generator(np.random.PCG64(seed))
        for start in range(0, resamples, chunk_size):
            end = min(start + chunk_size, resamples)
            matrix[start:end] = generator.integers(
                0,
                case_count,
                size=(end - start, case_count),
                dtype=np.int64,
                endpoint=False,
            )
        matrix.flush()
        del matrix
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    digest = sha256_file(path)
    sidecar.write_text(digest + "\n", encoding="utf-8")
    os.chmod(sidecar, 0o444)
    return path, digest


def bootstrap_metrics(data, index_matrix, chunk_size=METRIC_CHUNK_SIZE):
    index_matrix = np.asarray(index_matrix)
    if index_matrix.ndim != 2 or index_matrix.shape[1] != CASE_COUNT:
        raise ValueError("Bootstrap index matrix shape changed")
    if index_matrix.dtype != np.int64:
        raise TypeError("Bootstrap index matrix must use int64")
    result = {
        name: np.empty(index_matrix.shape[0], dtype=np.float64)
        for name in ("mse", "gini", "cv")
    }
    for start in range(0, index_matrix.shape[0], chunk_size):
        end = min(start + chunk_size, index_matrix.shape[0])
        indices = index_matrix[start:end]
        metrics = _metrics_from_aggregates(
            data["mse"][indices],
            data["credit"][indices],
            data["count"][indices],
        )
        for name in result:
            result[name][start:end] = metrics[name]
    return result


def _relative_improvement(reference, matched, name):
    reference = np.asarray(reference, dtype=np.float64)
    matched = np.asarray(matched, dtype=np.float64)
    if np.any(~np.isfinite(reference)) or np.any(reference <= 0):
        raise RuntimeError(f"{name} reference denominator is nonpositive")
    value = (reference - matched) / reference
    if np.any(~np.isfinite(value)):
        raise FloatingPointError(f"Relative {name} improvement is nonfinite")
    return value


def _relative_cv_increase(reference, matched):
    reference = np.asarray(reference, dtype=np.float64)
    matched = np.asarray(matched, dtype=np.float64)
    if np.any(~np.isfinite(reference)) or np.any(~np.isfinite(matched)):
        raise FloatingPointError("Token-load CV comparison is nonfinite")
    result = np.empty_like(reference)
    regular = reference > 1e-12
    result[regular] = (matched[regular] - reference[regular]) / reference[regular]
    both_zero = (~regular) & (matched <= 1e-12)
    result[both_zero] = 0.0
    result[(~regular) & (~both_zero)] = np.inf
    return result


def _interval(distribution):
    distribution = np.asarray(distribution, dtype=np.float64)
    return {
        "lower_one_sided_95": float(
            np.quantile(distribution, 0.05, method="linear")
        ),
        "upper_one_sided_95": float(
            np.quantile(distribution, 0.95, method="linear")
        ),
        "two_sided_95": [
            float(np.quantile(distribution, 0.025, method="linear")),
            float(np.quantile(distribution, 0.975, method="linear")),
        ],
    }


def compare_to_matched(point, distributions, reference_branch):
    matched_branch = "matched_credit_rate_redistribution"
    point_reference = point[reference_branch]
    point_matched = point[matched_branch]
    distribution_reference = distributions[reference_branch]
    distribution_matched = distributions[matched_branch]
    point_values = {
        "relative_mse_improvement": float(
            _relative_improvement(
                [point_reference["mse"]], [point_matched["mse"]], "MSE"
            )[0]
        ),
        "relative_gini_reduction": float(
            _relative_improvement(
                [point_reference["gini"]],
                [point_matched["gini"]],
                "credit-rate Gini",
            )[0]
        ),
        "relative_cv_increase": float(
            _relative_cv_increase(
                np.asarray([point_reference["cv"]]),
                np.asarray([point_matched["cv"]]),
            )[0]
        ),
    }
    distributions_values = {
        "relative_mse_improvement": _relative_improvement(
            distribution_reference["mse"], distribution_matched["mse"], "MSE"
        ),
        "relative_gini_reduction": _relative_improvement(
            distribution_reference["gini"],
            distribution_matched["gini"],
            "credit-rate Gini",
        ),
        "relative_cv_increase": _relative_cv_increase(
            distribution_reference["cv"], distribution_matched["cv"]
        ),
    }
    return {
        name: {
            "point": point_values[name],
            **_interval(distribution),
        }
        for name, distribution in distributions_values.items()
    }


def _load_case(
    path,
    protocol_sha256,
    branch,
    state_name,
    case_index,
    checkpoint_sha256,
    expected_case,
):
    path = Path(path)
    expected = {
        "version": EVALUATOR_VERSION,
        "branch": branch,
        "checkpoint_state": state_name,
        "protocol_sha256": protocol_sha256,
        "case_index": case_index,
        "checkpoint_sha256": checkpoint_sha256,
        "label": int(expected_case["label"]),
        "relative_path": expected_case["relative_path"],
    }
    return _load_reusable_case(path, expected, protocol_sha256)


def load_branch_data(
    output_root,
    protocol_sha256,
    branch,
    state_name,
    checkpoint_sha256,
    expected_cases,
):
    mse = np.empty(CASE_COUNT, dtype=np.float64)
    credit = np.empty((CASE_COUNT, BLOCK_COUNT, EXPERT_COUNT), dtype=np.float64)
    count = np.empty((CASE_COUNT, BLOCK_COUNT, EXPERT_COUNT), dtype=np.int64)
    labels = []
    for case_index in range(CASE_COUNT):
        path = (
            Path(output_root)
            / "raw"
            / branch
            / state_name
            / f"case-{case_index:03d}.json"
        )
        payload = _load_case(
            path,
            protocol_sha256,
            branch,
            state_name,
            case_index,
            checkpoint_sha256,
            expected_cases[case_index],
        )
        mse[case_index] = payload["mean_mse"]
        credit[case_index] = np.asarray(
            payload["aggregate_credit"], dtype=np.float64
        )
        raw_count = np.asarray(payload["aggregate_count"])
        if raw_count.shape != (BLOCK_COUNT, EXPERT_COUNT):
            raise ValueError(f"Held-out count shape changed: {path}")
        if np.any(raw_count != raw_count.astype(np.int64)):
            raise ValueError(f"Held-out count is not integral: {path}")
        count[case_index] = raw_count.astype(np.int64)
        if np.any(count[case_index].sum(axis=1) != TOKENS_PER_CASE):
            raise RuntimeError(f"Held-out token count is incomplete: {path}")
        labels.append(payload["label"])
    if len(set(labels)) != CASE_COUNT:
        raise RuntimeError("Held-out statistical units are not unique classes")
    data = {"mse": mse, "credit": credit, "count": count}
    point_metrics(data)
    return data


def _verify_completion_files(output_root, completion):
    expected_case_paths = {
        (
            Path("raw")
            / branch
            / state_name
            / f"case-{case_index:03d}.json"
        ).as_posix()
        for branch in BRANCHES
        for state_name in CHECKPOINT_STATES
        for case_index in range(CASE_COUNT)
    }
    case_hashes = completion.get("case_file_sha256")
    seal_hashes = completion.get("case_seal_file_sha256")
    metric_hashes = completion.get("metric_file_sha256")
    metric_seal_hashes = completion.get("metric_seal_file_sha256")
    if not isinstance(case_hashes, dict) or set(case_hashes) != expected_case_paths:
        raise RuntimeError("Held-out completion case-file inventory differs")
    expected_seal_paths = {
        f"{relative}.seal.json" for relative in expected_case_paths
    }
    if not isinstance(seal_hashes, dict) or set(seal_hashes) != expected_seal_paths:
        raise RuntimeError("Held-out completion seal-file inventory differs")
    expected_metric_paths = {
        (
            Path("sealed")
            / branch
            / state_name
            / f"case-{case_index:03d}.json"
        ).as_posix()
        for branch in BRANCHES
        for state_name in CHECKPOINT_STATES
        for case_index in range(CASE_COUNT)
    }
    expected_metric_seal_paths = {
        f"{relative}.seal.json" for relative in expected_metric_paths
    }
    if (
        not isinstance(metric_hashes, dict)
        or set(metric_hashes) != expected_metric_paths
    ):
        raise RuntimeError("Held-out completion metric-file inventory differs")
    if (
        not isinstance(metric_seal_hashes, dict)
        or set(metric_seal_hashes) != expected_metric_seal_paths
    ):
        raise RuntimeError("Held-out completion metric-seal inventory differs")
    expected_files = (
        expected_case_paths
        | expected_seal_paths
        | expected_metric_paths
        | expected_metric_seal_paths
    )
    raw_root = output_root / "raw"
    metric_root = output_root / "sealed"
    if (
        not raw_root.is_dir()
        or raw_root.is_symlink()
        or not metric_root.is_dir()
        or metric_root.is_symlink()
    ):
        raise RuntimeError("Held-out raw/sealed artifact roots are absent or indirect")
    observed_files = {
        path.relative_to(output_root).as_posix()
        for root in (raw_root, metric_root)
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected_directories = {
        "raw",
        "sealed",
        *(f"raw/{branch}" for branch in BRANCHES),
        *(
            f"raw/{branch}/{state_name}"
            for branch in BRANCHES
            for state_name in CHECKPOINT_STATES
        ),
        *(f"sealed/{branch}" for branch in BRANCHES),
        *(
            f"sealed/{branch}/{state_name}"
            for branch in BRANCHES
            for state_name in CHECKPOINT_STATES
        ),
    }
    observed_directories = {"raw", "sealed"} | {
        path.relative_to(output_root).as_posix()
        for root in (raw_root, metric_root)
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if observed_files != expected_files or observed_directories != expected_directories:
        raise RuntimeError("Held-out raw artifact inventory changed after evaluation")
    for relative, expected in case_hashes.items():
        path = output_root / relative
        if path.is_symlink() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"Held-out case mutability differs: {relative}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"Held-out case file changed after evaluation: {relative}")
    for relative, expected in seal_hashes.items():
        path = output_root / relative
        if path.is_symlink() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"Held-out case-seal mutability differs: {relative}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"Held-out case seal changed after evaluation: {relative}")
    for relative, expected in metric_hashes.items():
        path = output_root / relative
        if path.is_symlink() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"Held-out metric mutability differs: {relative}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"Held-out metric changed after evaluation: {relative}")
    for relative, expected in metric_seal_hashes.items():
        path = output_root / relative
        if path.is_symlink() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"Held-out metric-seal mutability differs: {relative}")
        if sha256_file(path) != expected:
            raise RuntimeError(
                f"Held-out metric seal changed after evaluation: {relative}"
            )


def _success_gates(comparisons):
    measure = comparisons["measure_only_control"]
    permuted = comparisons["rotating_permuted_scale_control"]
    gates = {
        "ema_mse_vs_measure_only": (
            measure["relative_mse_improvement"]["point"] >= 0.0025
            and measure["relative_mse_improvement"]["lower_one_sided_95"] > 0
        ),
        "ema_mse_vs_permuted": (
            permuted["relative_mse_improvement"]["point"] > 0
            and permuted["relative_mse_improvement"]["lower_one_sided_95"] > 0
        ),
        "ema_credit_gini_vs_measure_only": (
            measure["relative_gini_reduction"]["point"] >= 0.20
            and measure["relative_gini_reduction"]["lower_one_sided_95"] > 0
        ),
        "ema_credit_gini_vs_permuted": (
            permuted["relative_gini_reduction"]["point"] > 0
            and permuted["relative_gini_reduction"]["lower_one_sided_95"] > 0
        ),
        "ema_token_load_vs_measure_only": (
            measure["relative_cv_increase"]["point"] <= 0.05
            and measure["relative_cv_increase"]["upper_one_sided_95"] <= 0.05
        ),
        "ema_token_load_vs_permuted": (
            permuted["relative_cv_increase"]["point"] <= 0.05
            and permuted["relative_cv_increase"]["upper_one_sided_95"] <= 0.05
        ),
    }
    return {name: bool(value) for name, value in gates.items()}


def _json_safe(value):
    if isinstance(value, float):
        if math.isinf(value):
            return "+infinity" if value > 0 else "-infinity"
        if math.isnan(value):
            raise FloatingPointError("Statistical summary contains NaN")
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def aggregate_statistics(
    output_root,
    protocol_sha256,
    heldout_manifest_path,
    *,
    bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    bootstrap_seed=BOOTSTRAP_SEED,
    bootstrap_chunk_size=BOOTSTRAP_CHUNK_SIZE,
):
    if (
        bootstrap_resamples != BOOTSTRAP_RESAMPLES
        or bootstrap_seed != BOOTSTRAP_SEED
        or bootstrap_chunk_size != BOOTSTRAP_CHUNK_SIZE
    ):
        raise ValueError("Bootstrap constants differ from the sealed protocol")
    output_root = Path(output_root).resolve()
    completion_path = output_root / "evaluation-complete.json"
    if not completion_path.is_file():
        raise RuntimeError("Held-out evaluation is not complete")
    with completion_path.open("r", encoding="utf-8") as handle:
        completion = json.load(handle)
    if completion.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("Held-out completion protocol mismatch")
    completion_seal_path = _seal_path(completion_path)
    if not completion_seal_path.is_file():
        raise RuntimeError("Held-out completion seal is absent")
    with completion_seal_path.open("r", encoding="utf-8") as handle:
        completion_seal = json.load(handle)
    if (
        completion_seal.get("protocol_sha256") != protocol_sha256
        or completion_seal.get("artifact_canonical_sha256")
        != canonical_json_sha256(completion)
    ):
        raise RuntimeError("Held-out completion seal mismatch")
    expected_case_file_count = len(BRANCHES) * len(CHECKPOINT_STATES) * CASE_COUNT
    if (
        completion.get("status") != "complete_without_efficacy_aggregation"
        or completion.get("case_file_count") != expected_case_file_count
    ):
        raise RuntimeError("Held-out completion metadata differs")
    checkpoint_hashes = completion.get("checkpoint_file_sha256")
    if not isinstance(checkpoint_hashes, dict) or set(checkpoint_hashes) != set(BRANCHES):
        raise RuntimeError("Held-out completion checkpoint binding differs")
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in checkpoint_hashes.values()
    ):
        raise RuntimeError("Held-out completion checkpoint hash is malformed")
    transcript_chains = completion.get("transcript_final_chain_digests")
    trainer_digests = completion.get("trainer_state_sha256")
    if (
        not isinstance(transcript_chains, dict)
        or set(transcript_chains) != set(BRANCHES)
        or len(set(transcript_chains.values())) != 1
        or not isinstance(trainer_digests, dict)
        or set(trainer_digests) != set(BRANCHES)
        or len(set(trainer_digests.values())) != 1
    ):
        raise RuntimeError("Held-out completion branch replay binding differs")
    branch_integrity = completion.get("branch_integrity")
    if not isinstance(branch_integrity, dict) or set(branch_integrity) != set(BRANCHES):
        raise RuntimeError("Held-out completion controller binding differs")
    _verify_completion_files(output_root, completion)

    manifest, manifest_sha256 = load_heldout_manifest(heldout_manifest_path)
    if (
        manifest_sha256 != completion.get("heldout_manifest_canonical_sha256")
        or len(manifest["cases"]) != CASE_COUNT
    ):
        raise RuntimeError("Held-out completion manifest binding differs")
    expected_cases = manifest["cases"]

    index_path, index_sha256 = materialize_bootstrap_indices(
        output_root / "statistics" / "bootstrap-indices.npy"
    )
    index_matrix = np.load(index_path, mmap_mode="r", allow_pickle=False)
    state_summaries = {}
    for state_name in CHECKPOINT_STATES:
        data = {
            branch: load_branch_data(
                output_root,
                protocol_sha256,
                branch,
                state_name,
                checkpoint_hashes[branch],
                expected_cases,
            )
            for branch in BRANCHES
        }
        points = {branch: point_metrics(data[branch]) for branch in BRANCHES}
        distributions = {
            branch: bootstrap_metrics(
                data[branch], index_matrix, chunk_size=METRIC_CHUNK_SIZE
            )
            for branch in BRANCHES
        }
        comparisons = {
            reference: compare_to_matched(points, distributions, reference)
            for reference in BRANCHES[:2]
        }
        state_summaries[state_name] = {
            "point_metrics": points,
            "comparisons": comparisons,
        }

    ema = state_summaries["ema_model_state_dict"]["comparisons"]
    gates = _success_gates(ema)
    online = state_summaries["model_state_dict"]["comparisons"]
    online_gate = all(
        online[reference]["relative_mse_improvement"]["point"] >= 0
        for reference in BRANCHES[:2]
    )
    gates["online_weight_sensitivity"] = bool(online_gate)
    gates["numerical_and_integrity"] = True
    payload = {
        "version": STATISTICS_VERSION,
        "protocol_sha256": protocol_sha256,
        "evaluation_completion_canonical_sha256": canonical_json_sha256(completion),
        "heldout_manifest_canonical_sha256": manifest_sha256,
        "checkpoint_file_sha256": checkpoint_hashes,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "chunk_size": BOOTSTRAP_CHUNK_SIZE,
            "metric_chunk_size": METRIC_CHUNK_SIZE,
            "case_count": CASE_COUNT,
            "index_matrix_path": str(index_path),
            "index_matrix_file_sha256": index_sha256,
            "generator": "np.random.Generator(np.random.PCG64(seed))",
            "quantile_method": "linear",
        },
        "checkpoint_states": _json_safe(state_summaries),
        "success_gates": gates,
        "all_required_passed": bool(all(gates.values())),
    }
    summary_path = output_root / "statistics" / "summary.json"
    summary_digest = canonical_json_sha256(payload)
    seal = {
        "version": 1,
        "artifact": summary_path.name,
        "artifact_canonical_sha256": summary_digest,
        "protocol_sha256": protocol_sha256,
    }
    seal_path = _seal_path(summary_path)
    if summary_path.exists() or seal_path.exists():
        if not summary_path.exists() or not seal_path.exists():
            raise RuntimeError("Statistical summary and seal must exist together")
        with summary_path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != payload:
                raise RuntimeError("Existing statistical summary differs")
        with seal_path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != seal:
                raise RuntimeError("Existing statistical summary seal differs")
    else:
        atomic_write_json(summary_path, payload, mode=0o444)
        atomic_write_json(seal_path, seal, mode=0o444)
    return summary_path, payload
