"""Measure routed-expert parameter motion across training checkpoints.

The primary measurement is the parameter displacement observed between two
checkpoints.  AdamW moments at the later checkpoint provide a separate local
update-field measurement; they are not treated as integrated training motion.
"""

from __future__ import annotations

import csv
import copy
import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from easydict import EasyDict
from scipy.stats import rankdata

try:
    from torch.torch_version import TorchVersion
except ImportError:  # pragma: no cover - older supported Torch releases
    TorchVersion = None

from analyses.denoising_regret.probe import _build_model
from analyses.t_SNE.checkpoint_utils import load_runtime_cfg, parse_checkpoint_step
from credit_redistribution.git_provenance import (
    repository_state,
    verify_worktree_source_manifest,
)


AUDIT_VERSION = 1
TRAINING_PROVENANCE_VERSION = 1
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
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPERT_PARAMETER_PATTERN = re.compile(
    r"^blocks\.(?P<block>[0-9]+)\.mlp\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<suffix>.+)$"
)
CLUSTER_CENTER_PATTERN = re.compile(
    r"^blocks\.(?P<block>[0-9]+)\.mlp\.cluster_centers$"
)
DEFAULT_CHUNK_SIZE = 1_048_576
OUTPUT_FILENAMES = (
    "audit.json",
    "summary.json",
    "expert_metrics.csv",
    "summary.md",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_unchanged_file(path: Path, expected_sha256: str, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is no longer a regular file: {path}")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"{label} changed while the audit was running: {path}")


def _json_sha256(payload) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _training_config_payload_sha256(config: Mapping) -> str:
    normalized = copy.deepcopy(dict(config))
    num_steps = normalized.get("num_steps")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("Config num_steps must be a positive integer")
    normalized["num_steps"] = "<runtime-stop-boundary>"
    return _json_sha256(normalized)


def coefficient_of_variation(values: Sequence[float]) -> float:
    array = _nonnegative_vector(values, "CV")
    mean = float(array.mean())
    return float(array.std() / mean) if mean > 0 else 0.0


def gini(values: Sequence[float]) -> float:
    array = _nonnegative_vector(values, "Gini")
    total = float(array.sum())
    if total == 0:
        return 0.0
    sorted_values = np.sort(array)
    ranks = np.arange(1, sorted_values.size + 1, dtype=np.float64)
    return float(
        (2.0 * np.dot(ranks, sorted_values) / (sorted_values.size * total))
        - (sorted_values.size + 1.0) / sorted_values.size
    )


def _nonnegative_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} values must be a nonempty vector")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} values must be finite and nonnegative")
    return array


def spearman_correlation(left: Sequence[float], right: Sequence[float]):
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        left_array.ndim != 1
        or right_array.ndim != 1
        or left_array.size != right_array.size
        or left_array.size < 2
        or not np.isfinite(left_array).all()
        or not np.isfinite(right_array).all()
    ):
        raise ValueError("Spearman inputs must be aligned finite vectors")
    if np.ptp(left_array) == 0 or np.ptp(right_array) == 0:
        return None
    left_rank = rankdata(left_array, method="average")
    right_rank = rankdata(right_array, method="average")
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return value if np.isfinite(value) else None


def metric_distribution(values: Sequence[float]) -> dict:
    array = _nonnegative_vector(values, "metric")
    minimum = float(array.min())
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std()),
        "coefficient_of_variation": coefficient_of_variation(array),
        "gini": gini(array),
        "minimum": minimum,
        "maximum": float(array.max()),
        "maximum_to_minimum": (
            float(array.max() / minimum) if minimum > 0 else None
        ),
    }


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be a mapping: {path}")
    return payload


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"YAML payload must be a mapping: {path}")
    return payload


def _checkpoint_safe_globals():
    values = [EasyDict]
    if TorchVersion is not None:
        values.append(TorchVersion)
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    return safe_globals(values) if safe_globals is not None else nullcontext()


def load_checkpoint(path: Path):
    kwargs = {"map_location": "cpu", "weights_only": True}
    with _checkpoint_safe_globals():
        try:
            return torch.load(path, mmap=True, **kwargs)
        except (TypeError, RuntimeError):
            try:
                return torch.load(path, **kwargs)
            except TypeError:
                kwargs.pop("weights_only")
                return torch.load(path, **kwargs)


def build_named_parameter_specs(runtime_cfg) -> list[dict]:
    with torch.device("meta"):
        model = _build_model(runtime_cfg)
    try:
        specs = [
            {
                "name": name,
                "shape": tuple(parameter.shape),
                "dtype": parameter.dtype,
                "numel": int(parameter.numel()),
            }
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    finally:
        del model
    if not specs:
        raise ValueError("The configured model has no trainable parameters")
    if len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("The configured model has duplicate parameter names")
    return specs


def _optimizer_step(value) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1 or value.is_complex():
            raise ValueError("Optimizer step must be a real scalar")
        value = value.item()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or int(value) != value
        or int(value) < 1
    ):
        raise ValueError("Optimizer step must be a positive integer")
    return int(value)


def bind_optimizer_parameters(
    optimizer_state: Mapping,
    named_parameter_specs: Sequence[Mapping],
    expected_optimizer_step: int,
) -> dict[str, dict]:
    if not isinstance(optimizer_state, Mapping):
        raise TypeError("optimizer_state_dict must be a mapping")
    groups = optimizer_state.get("param_groups")
    state = optimizer_state.get("state")
    if not isinstance(groups, list) or not groups:
        raise ValueError("optimizer_state_dict has no parameter groups")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("optimizer_state_dict has no parameter state")

    flattened = []
    group_by_id = {}
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("Optimizer parameter groups are malformed")
        for required in ("lr", "betas", "eps", "weight_decay", "amsgrad"):
            if required not in group:
                raise ValueError(f"Optimizer group is missing {required}")
        betas = group["betas"]
        if (
            not isinstance(betas, (tuple, list))
            or len(betas) != 2
            or not all(isinstance(value, (int, float)) for value in betas)
            or not all(0 <= float(value) < 1 for value in betas)
        ):
            raise ValueError("Optimizer betas are malformed")
        if type(group["amsgrad"]) is not bool:
            raise ValueError("Optimizer amsgrad option is malformed")
        if bool(group.get("maximize", False)):
            raise ValueError("Maximizing optimizers are not supported")
        for parameter_id in group["params"]:
            if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
                raise ValueError("Optimizer parameter IDs must be integers")
            if parameter_id in group_by_id:
                raise ValueError("Optimizer parameter IDs must be unique")
            flattened.append(parameter_id)
            group_by_id[parameter_id] = (group_index, group)

    if len(flattened) != len(named_parameter_specs):
        raise ValueError(
            "Optimizer parameter count differs from the configured model: "
            f"{len(flattened)} != {len(named_parameter_specs)}"
        )
    if set(state) != set(flattened):
        raise ValueError("Optimizer state does not cover every trainable parameter")

    bound = {}
    for parameter_id, spec in zip(flattened, named_parameter_specs):
        parameter_state = state[parameter_id]
        if not isinstance(parameter_state, Mapping):
            raise ValueError(f"Optimizer state {parameter_id} is malformed")
        required_state = {"step", "exp_avg", "exp_avg_sq"}
        _, group = group_by_id[parameter_id]
        if group["amsgrad"]:
            required_state.add("max_exp_avg_sq")
        if set(parameter_state) != required_state:
            raise ValueError(
                f"Optimizer state keys for parameter {parameter_id} are incomplete"
            )
        observed_step = _optimizer_step(parameter_state["step"])
        if observed_step != expected_optimizer_step:
            raise ValueError(
                f"Optimizer step for {spec['name']} is {observed_step}, "
                f"expected {expected_optimizer_step}"
            )
        for state_name in required_state - {"step"}:
            tensor = parameter_state[state_name]
            if (
                not torch.is_tensor(tensor)
                or tensor.device.type != "cpu"
                or tuple(tensor.shape) != tuple(spec["shape"])
                or tensor.dtype != spec["dtype"]
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"Optimizer {state_name} for {spec['name']} differs from "
                    "the configured parameter contract"
                )
        group_index, group = group_by_id[parameter_id]
        bound[spec["name"]] = {
            "parameter_id": parameter_id,
            "group_index": group_index,
            "group": group,
            "state": parameter_state,
            "spec": spec,
        }
    return bound


def routed_expert_parameter_groups(
    named_parameter_specs: Sequence[Mapping],
    model_state: Mapping,
    expected_blocks: Sequence[int],
    expected_experts_per_block: int,
) -> dict[tuple[int, int], list[Mapping]]:
    expected_blocks = tuple(int(value) for value in expected_blocks)
    if len(set(expected_blocks)) != len(expected_blocks):
        raise ValueError("Expected MoE block IDs must be unique")
    if expected_experts_per_block < 2:
        raise ValueError("At least two routed experts are required")

    observed_blocks = set()
    for name, value in model_state.items():
        match = CLUSTER_CENTER_PATTERN.fullmatch(name)
        if match is None:
            continue
        block = int(match.group("block"))
        observed_blocks.add(block)
        if block in expected_blocks and (
            not torch.is_tensor(value)
            or value.ndim != 2
            or value.shape[0] != expected_experts_per_block
        ):
            raise ValueError(
                f"Block {block} cluster centers do not encode "
                f"{expected_experts_per_block} routed experts"
            )
    if observed_blocks != set(expected_blocks):
        raise ValueError(
            f"Configured MoE blocks differ from the protocol: "
            f"{sorted(observed_blocks)} != {sorted(expected_blocks)}"
        )

    groups = defaultdict(list)
    suffixes = defaultdict(set)
    for spec in named_parameter_specs:
        match = EXPERT_PARAMETER_PATTERN.fullmatch(spec["name"])
        if match is None:
            continue
        block = int(match.group("block"))
        expert = int(match.group("expert"))
        if block not in expected_blocks or expert >= expected_experts_per_block:
            continue
        groups[(block, expert)].append(spec)
        suffixes[(block, expert)].add(match.group("suffix"))

    expected_keys = {
        (block, expert)
        for block in expected_blocks
        for expert in range(expected_experts_per_block)
    }
    if set(groups) != expected_keys:
        missing = sorted(expected_keys - set(groups))
        extra = sorted(set(groups) - expected_keys)
        raise ValueError(f"Routed-expert parameter groups differ: missing={missing}, extra={extra}")
    reference_suffixes = suffixes[min(expected_keys)]
    for key in sorted(expected_keys):
        if suffixes[key] != reference_suffixes:
            raise ValueError(f"Expert {key} has a different parameter structure")
    return {key: groups[key] for key in sorted(groups)}


def _finite_tensor(tensor: torch.Tensor, name: str, chunk_size: int) -> None:
    flattened = tensor.reshape(-1)
    for start in range(0, flattened.numel(), chunk_size):
        if not bool(torch.isfinite(flattened[start : start + chunk_size]).all()):
            raise ValueError(f"{name} contains non-finite values")


def parameter_pair_sums(
    previous: torch.Tensor,
    current: torch.Tensor,
    optimizer_binding: Mapping,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict:
    spec = optimizer_binding["spec"]
    if (
        not torch.is_tensor(previous)
        or not torch.is_tensor(current)
        or previous.device.type != "cpu"
        or current.device.type != "cpu"
        or tuple(previous.shape) != tuple(spec["shape"])
        or tuple(current.shape) != tuple(spec["shape"])
        or previous.dtype != spec["dtype"]
        or current.dtype != spec["dtype"]
    ):
        raise ValueError(f"Model tensors for {spec['name']} violate the parameter contract")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    state = optimizer_binding["state"]
    group = optimizer_binding["group"]
    beta1, beta2 = (float(value) for value in group["betas"])
    optimizer_step = _optimizer_step(state["step"])
    bias_correction1 = 1.0 - beta1 ** optimizer_step
    bias_correction2 = 1.0 - beta2 ** optimizer_step
    if bias_correction1 <= 0 or bias_correction2 <= 0:
        raise ValueError("AdamW bias correction is not positive")
    learning_rate = float(group["lr"])
    epsilon = float(group["eps"])
    weight_decay = float(group["weight_decay"])
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(epsilon)
        or epsilon <= 0
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("AdamW scalar options are invalid")

    second_moment = (
        state["max_exp_avg_sq"] if group["amsgrad"] else state["exp_avg_sq"]
    )
    tensors = {
        "previous": previous.reshape(-1),
        "current": current.reshape(-1),
        "exp_avg": state["exp_avg"].reshape(-1),
        "second_moment": second_moment.reshape(-1),
    }
    for name, tensor in tensors.items():
        _finite_tensor(tensor, f"{spec['name']} {name}", chunk_size)

    sums = {
        "parameter_count": int(previous.numel()),
        "previous_parameter_square_sum": 0.0,
        "current_parameter_square_sum": 0.0,
        "displacement_square_sum": 0.0,
        "debiased_first_moment_square_sum": 0.0,
        "debiased_second_moment_sum": 0.0,
        "preconditioned_moment_square_sum": 0.0,
        "adamw_update_field_square_sum": 0.0,
    }
    for start in range(0, previous.numel(), chunk_size):
        stop = min(start + chunk_size, previous.numel())
        previous_chunk = tensors["previous"][start:stop].double()
        current_chunk = tensors["current"][start:stop].double()
        first_moment = tensors["exp_avg"][start:stop].double() / bias_correction1
        second = tensors["second_moment"][start:stop].double() / bias_correction2
        if bool((second < 0).any()):
            raise ValueError(f"AdamW second moment for {spec['name']} is negative")
        displacement = current_chunk - previous_chunk
        preconditioned = first_moment / (second.sqrt() + epsilon)
        # This is the endpoint AdamW vector field.  It is a local diagnostic,
        # not a reconstruction of all updates between the two checkpoints.
        update_field = learning_rate * (
            preconditioned + weight_decay * current_chunk
        )
        sums["previous_parameter_square_sum"] += float(previous_chunk.square().sum())
        sums["current_parameter_square_sum"] += float(current_chunk.square().sum())
        sums["displacement_square_sum"] += float(displacement.square().sum())
        sums["debiased_first_moment_square_sum"] += float(first_moment.square().sum())
        sums["debiased_second_moment_sum"] += float(second.sum())
        sums["preconditioned_moment_square_sum"] += float(preconditioned.square().sum())
        sums["adamw_update_field_square_sum"] += float(update_field.square().sum())
    sums.update({
        "optimizer_step": optimizer_step,
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "weight_decay": weight_decay,
    })
    return sums


def _sum_parameter_statistics(parts: Sequence[Mapping], interval_steps: int) -> dict:
    if not parts or interval_steps < 1:
        raise ValueError("Expert aggregation requires measurements and a positive interval")
    sum_names = (
        "parameter_count",
        "previous_parameter_square_sum",
        "current_parameter_square_sum",
        "displacement_square_sum",
        "debiased_first_moment_square_sum",
        "debiased_second_moment_sum",
        "preconditioned_moment_square_sum",
        "adamw_update_field_square_sum",
    )
    totals = {name: sum(part[name] for part in parts) for name in sum_names}
    parameter_count = int(totals["parameter_count"])
    if parameter_count < 1:
        raise ValueError("Expert parameter count must be positive")
    options = {
        (
            part["optimizer_step"],
            part["learning_rate"],
            part["beta1"],
            part["beta2"],
            part["epsilon"],
            part["weight_decay"],
        )
        for part in parts
    }
    if len(options) != 1:
        raise ValueError("One expert spans incompatible optimizer options")
    optimizer_step, learning_rate, beta1, beta2, epsilon, weight_decay = options.pop()

    previous_square = totals["previous_parameter_square_sum"]
    current_square = totals["current_parameter_square_sum"]
    displacement_square = totals["displacement_square_sum"]
    update_square = totals["adamw_update_field_square_sum"]
    if previous_square <= 0 or current_square <= 0:
        raise ValueError("Expert parameter norm must be positive")
    return {
        "parameter_count": parameter_count,
        "parameter_rms_before": math.sqrt(previous_square / parameter_count),
        "parameter_rms_after": math.sqrt(current_square / parameter_count),
        "displacement_l2": math.sqrt(displacement_square),
        "displacement_rms": math.sqrt(displacement_square / parameter_count),
        "relative_displacement": math.sqrt(displacement_square / previous_square),
        "net_displacement_per_step_rms": (
            math.sqrt(displacement_square / parameter_count) / interval_steps
        ),
        "debiased_first_moment_rms": math.sqrt(
            totals["debiased_first_moment_square_sum"] / parameter_count
        ),
        "debiased_second_moment_rms": math.sqrt(
            totals["debiased_second_moment_sum"] / parameter_count
        ),
        "preconditioned_moment_rms": math.sqrt(
            totals["preconditioned_moment_square_sum"] / parameter_count
        ),
        "adamw_update_field_rms": math.sqrt(update_square / parameter_count),
        "relative_adamw_update_field": math.sqrt(update_square / current_square),
        "optimizer_step": optimizer_step,
        "learning_rate": learning_rate,
        "betas": [beta1, beta2],
        "epsilon": epsilon,
        "weight_decay": weight_decay,
    }


def _model_state(checkpoint: Mapping) -> Mapping:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint model_state_dict is absent or empty")
    return state


def analyze_checkpoint_interval(
    previous_checkpoint: Mapping,
    current_checkpoint: Mapping,
    named_parameter_specs: Sequence[Mapping],
    expected_blocks: Sequence[int],
    expected_experts_per_block: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict:
    previous_step = int(previous_checkpoint["step"])
    current_step = int(current_checkpoint["step"])
    interval_steps = current_step - previous_step
    if interval_steps < 1:
        raise ValueError("Checkpoint steps must be strictly increasing")
    previous_state = _model_state(previous_checkpoint)
    current_state = _model_state(current_checkpoint)
    if set(previous_state) != set(current_state):
        raise ValueError("Model state keys changed across checkpoints")

    groups = routed_expert_parameter_groups(
        named_parameter_specs,
        current_state,
        expected_blocks,
        expected_experts_per_block,
    )
    bound = bind_optimizer_parameters(
        current_checkpoint.get("optimizer_state_dict"),
        named_parameter_specs,
        expected_optimizer_step=current_step + 1,
    )
    expert_metrics = {}
    for (block, expert), specs in groups.items():
        parts = []
        for spec in specs:
            name = spec["name"]
            parts.append(parameter_pair_sums(
                previous_state[name],
                current_state[name],
                bound[name],
                chunk_size=chunk_size,
            ))
        metrics = _sum_parameter_statistics(parts, interval_steps)
        metrics.update({"block_index": block, "expert_index": expert})
        expert_metrics[(block, expert)] = metrics

    blocks = {}
    metric_names = (
        "displacement_rms",
        "relative_displacement",
        "adamw_update_field_rms",
        "relative_adamw_update_field",
        "debiased_second_moment_rms",
    )
    for block in expected_blocks:
        experts = [expert_metrics[(int(block), expert)] for expert in range(expected_experts_per_block)]
        distributions = {
            metric_name: metric_distribution(
                [expert[metric_name] for expert in experts]
            )
            for metric_name in metric_names
        }
        blocks[str(block)] = {
            "block_index": int(block),
            "experts": experts,
            "distributions": distributions,
            "correlations": {
                "displacement_vs_adamw_update_field": spearman_correlation(
                    [expert["displacement_rms"] for expert in experts],
                    [expert["adamw_update_field_rms"] for expert in experts],
                ),
                "relative_displacement_vs_relative_adamw_update_field": (
                    spearman_correlation(
                        [expert["relative_displacement"] for expert in experts],
                        [expert["relative_adamw_update_field"] for expert in experts],
                    )
                ),
                "displacement_vs_second_moment": spearman_correlation(
                    [expert["displacement_rms"] for expert in experts],
                    [expert["debiased_second_moment_rms"] for expert in experts],
                ),
            },
        }
    return {
        "from_step": previous_step,
        "to_step": current_step,
        "interval_steps": interval_steps,
        "blocks": blocks,
    }


def summarize_rank_persistence(intervals: Sequence[Mapping], expected_blocks: Sequence[int]) -> dict:
    if len(intervals) < 2:
        raise ValueError("Rank persistence requires at least two checkpoint intervals")
    expected_pair_count = len(intervals) - 1
    output = {}
    for block in expected_blocks:
        block_key = str(block)
        displacement_correlations = []
        update_correlations = []
        for left, right in zip(intervals[:-1], intervals[1:]):
            left_experts = left["blocks"][block_key]["experts"]
            right_experts = right["blocks"][block_key]["experts"]
            if [row["expert_index"] for row in left_experts] != [
                row["expert_index"] for row in right_experts
            ]:
                raise ValueError("Expert order changed across intervals")
            displacement_correlations.append(spearman_correlation(
                [row["relative_displacement"] for row in left_experts],
                [row["relative_displacement"] for row in right_experts],
            ))
            update_correlations.append(spearman_correlation(
                [row["relative_adamw_update_field"] for row in left_experts],
                [row["relative_adamw_update_field"] for row in right_experts],
            ))
        valid_displacement = [value for value in displacement_correlations if value is not None]
        valid_update = [value for value in update_correlations if value is not None]
        displacement_complete = len(valid_displacement) == expected_pair_count
        update_complete = len(valid_update) == expected_pair_count
        output[block_key] = {
            "block_index": int(block),
            "expected_adjacent_interval_pairs": expected_pair_count,
            "adjacent_interval_relative_displacement_spearman": displacement_correlations,
            "valid_adjacent_interval_relative_displacement_pairs": len(
                valid_displacement
            ),
            "median_adjacent_interval_relative_displacement_spearman": (
                float(np.median(valid_displacement)) if displacement_complete else None
            ),
            "adjacent_interval_relative_adamw_update_spearman": update_correlations,
            "valid_adjacent_interval_relative_adamw_update_pairs": len(valid_update),
            "median_adjacent_interval_relative_adamw_update_spearman": (
                float(np.median(valid_update)) if update_complete else None
            ),
        }
    return output


def evaluate_gate(
    intervals: Sequence[Mapping],
    persistence: Mapping,
    thresholds: Mapping,
    expected_blocks: Sequence[int],
) -> dict:
    cells = [
        interval["blocks"][str(block)]
        for interval in intervals
        for block in expected_blocks
    ]
    displacement_ginis = [
        cell["distributions"]["relative_displacement"]["gini"] for cell in cells
    ]
    update_ginis = [
        cell["distributions"]["relative_adamw_update_field"]["gini"]
        for cell in cells
    ]
    within_correlations = [
        cell["correlations"]["relative_displacement_vs_relative_adamw_update_field"]
        for cell in cells
    ]
    valid_within = [value for value in within_correlations if value is not None]
    block_persistence = [
        persistence[str(block)][
            "median_adjacent_interval_relative_displacement_spearman"
        ]
        for block in expected_blocks
    ]
    valid_persistence = [value for value in block_persistence if value is not None]

    observed = {
        "interval_block_cells": len(cells),
        "median_relative_displacement_gini": float(np.median(displacement_ginis)),
        "fraction_cells_relative_displacement_gini_at_least_effect_size": float(
            np.mean(
                np.asarray(displacement_ginis)
                >= float(thresholds["minimum_cell_relative_displacement_gini"])
            )
        ),
        "median_relative_adamw_update_gini": float(np.median(update_ginis)),
        "valid_within_cell_correlation_fraction": len(valid_within) / len(cells),
        "median_within_cell_displacement_update_spearman": (
            float(np.median(valid_within)) if valid_within else None
        ),
        "valid_block_persistence_fraction": (
            len(valid_persistence) / len(expected_blocks)
        ),
        "median_adjacent_interval_displacement_spearman": (
            float(np.median(valid_persistence)) if valid_persistence else None
        ),
        "fraction_blocks_with_positive_median_adjacent_spearman": (
            float(np.mean(np.asarray(valid_persistence) > 0))
            if valid_persistence
            else 0.0
        ),
    }

    def at_least(name, threshold_name):
        value = observed[name]
        threshold = float(thresholds[threshold_name])
        return {
            "observed": value,
            "required": f">={threshold}",
            "passed": value is not None and value >= threshold,
        }

    checks = {
        "interval_block_cells": at_least(
            "interval_block_cells", "minimum_interval_block_cells"
        ),
        "relative_displacement_gini": at_least(
            "median_relative_displacement_gini",
            "minimum_median_relative_displacement_gini",
        ),
        "widespread_displacement_imbalance": at_least(
            "fraction_cells_relative_displacement_gini_at_least_effect_size",
            "minimum_fraction_cells_above_displacement_effect_size",
        ),
        "relative_adamw_update_gini": at_least(
            "median_relative_adamw_update_gini",
            "minimum_median_relative_adamw_update_gini",
        ),
        "valid_within_cell_correlations": at_least(
            "valid_within_cell_correlation_fraction",
            "minimum_valid_within_cell_correlation_fraction",
        ),
        "displacement_update_agreement": at_least(
            "median_within_cell_displacement_update_spearman",
            "minimum_median_displacement_update_spearman",
        ),
        "valid_block_persistence": at_least(
            "valid_block_persistence_fraction",
            "minimum_valid_block_persistence_fraction",
        ),
        "rank_persistence": at_least(
            "median_adjacent_interval_displacement_spearman",
            "minimum_median_adjacent_interval_displacement_spearman",
        ),
        "widespread_rank_persistence": at_least(
            "fraction_blocks_with_positive_median_adjacent_spearman",
            "minimum_fraction_blocks_with_positive_median_adjacent_spearman",
        ),
    }
    return {
        "decision_scope": (
            "passing only permits a paired frozen route-count and learning-credit "
            "comparison; it does not permit a training method or performance claim"
        ),
        "thresholds": dict(thresholds),
        "observed": observed,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_manifest(manifest: Mapping) -> tuple[dict, dict]:
    if manifest.get("protocol_version") != 1:
        raise ValueError("Unsupported expert update-budget protocol version")
    expected = manifest.get("expected")
    thresholds = manifest.get("gate_thresholds")
    if not isinstance(expected, dict) or not isinstance(thresholds, dict):
        raise ValueError("Protocol expected fields and gate thresholds are required")
    required_expected = {
        "model_name",
        "config_basename",
        "config_payload_sha256",
        "training_git_commit",
        "global_seed",
        "world_size",
        "total_train_batch_size",
        "learning_rate",
        "weight_decay",
        "gpu_ids",
        "checkpoint_steps",
        "moe_blocks",
        "conditional_experts_per_block",
    }
    if set(expected) != required_expected:
        raise ValueError(
            "Protocol expected fields differ from the implementation contract: "
            f"{sorted(set(expected) ^ required_expected)}"
        )
    required_thresholds = {
        "minimum_interval_block_cells",
        "minimum_cell_relative_displacement_gini",
        "minimum_median_relative_displacement_gini",
        "minimum_fraction_cells_above_displacement_effect_size",
        "minimum_median_relative_adamw_update_gini",
        "minimum_valid_within_cell_correlation_fraction",
        "minimum_median_displacement_update_spearman",
        "minimum_valid_block_persistence_fraction",
        "minimum_median_adjacent_interval_displacement_spearman",
        "minimum_fraction_blocks_with_positive_median_adjacent_spearman",
    }
    if set(thresholds) != required_thresholds:
        raise ValueError(
            "Protocol gate thresholds differ from the implementation contract: "
            f"{sorted(set(thresholds) ^ required_thresholds)}"
        )
    steps = expected["checkpoint_steps"]
    if (
        not isinstance(steps, list)
        or len(steps) < 3
        or any(isinstance(step, bool) or not isinstance(step, int) for step in steps)
        or steps != sorted(set(steps))
    ):
        raise ValueError("Protocol checkpoint steps must be unique and increasing")
    return expected, thresholds


def _validate_config(config_path: Path, expected: Mapping):
    if config_path.name != expected["config_basename"]:
        raise ValueError(
            f"Config basename differs from the protocol: {config_path.name}"
        )
    config = _load_yaml(config_path)
    if _training_config_payload_sha256(config) != expected["config_payload_sha256"]:
        raise ValueError("Config payload hash differs from the training protocol")
    exact = {
        "model_name": expected["model_name"],
        "global_seed": expected["global_seed"],
        "total_train_batch_size": expected["total_train_batch_size"],
        "lr": expected["learning_rate"],
        "weight_decay": expected["weight_decay"],
        "gpu_ids": expected["gpu_ids"],
    }
    for name, value in exact.items():
        if config.get(name) != value:
            raise ValueError(
                f"Config violates protocol field {name}: {config.get(name)!r} != {value!r}"
            )
    model_config = config.get("DiT_B_config", {}).get("MoE_config", {})
    if model_config.get("num_routed_experts") != expected["conditional_experts_per_block"]:
        raise ValueError("Config routed-expert count differs from the protocol")
    return config


def _validate_training_provenance(provenance: Mapping, expected: Mapping) -> dict:
    required_fields = {
        "version",
        "strict",
        "git",
        "config",
        "source_sha256",
        "environment",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != required_fields:
        raise ValueError("Checkpoint training provenance fields are malformed")
    if (
        provenance["version"] != TRAINING_PROVENANCE_VERSION
        or provenance["strict"] is not True
    ):
        raise ValueError("Checkpoint training provenance version is invalid")

    git_provenance = provenance["git"]
    if not isinstance(git_provenance, Mapping) or set(git_provenance) != {
        "commit",
        "origin_repa_commit",
        "status_clean",
        "origin_repa_divergence",
    }:
        raise ValueError("Checkpoint training Git provenance is malformed")
    if (
        git_provenance["commit"] != expected["training_git_commit"]
        or git_provenance["origin_repa_commit"] != expected["training_git_commit"]
        or git_provenance["status_clean"] is not True
        or git_provenance["origin_repa_divergence"] != "0\t0"
    ):
        raise ValueError("Checkpoint was not launched from the locked clean commit")

    config_provenance = provenance["config"]
    if not isinstance(config_provenance, Mapping) or set(config_provenance) != {
        "version",
        "basename",
        "payload_sha256",
    }:
        raise ValueError("Checkpoint training config provenance is malformed")
    if (
        config_provenance["version"] != TRAINING_PROVENANCE_VERSION
        or config_provenance["basename"] != expected["config_basename"]
        or config_provenance["payload_sha256"] != expected["config_payload_sha256"]
    ):
        raise ValueError("Checkpoint config provenance differs from the protocol")

    source_hashes = provenance["source_sha256"]
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(
        LOCKED_TRAINING_SOURCE_PATHS
    ):
        raise ValueError("Checkpoint training source hash set is not canonical")
    for relative in LOCKED_TRAINING_SOURCE_PATHS:
        if not isinstance(source_hashes[relative], str) or not SHA256_PATTERN.fullmatch(
            source_hashes[relative]
        ):
            raise ValueError(f"Checkpoint training source hash is invalid: {relative}")

    environment = provenance["environment"]
    environment_fields = {
        "python",
        "python_executable",
        "torch",
        "numpy",
        "cuda_runtime",
        "devices",
        "cuda_visible_devices",
        "cuda_devices",
    }
    if not isinstance(environment, Mapping) or set(environment) != environment_fields:
        raise ValueError("Checkpoint training environment provenance is malformed")
    cuda_devices = environment["cuda_devices"]
    expected_cuda_devices = [f"cuda:{index}" for index in range(expected["world_size"])]
    if (
        environment["devices"] != expected_cuda_devices
        or environment["cuda_visible_devices"]
        != [str(index) for index in expected["gpu_ids"]]
        or not isinstance(cuda_devices, Mapping)
        or set(cuda_devices) != set(expected_cuda_devices)
    ):
        raise ValueError("Checkpoint training CUDA provenance is malformed")
    for device_name in expected_cuda_devices:
        device = cuda_devices[device_name]
        if not isinstance(device, Mapping) or set(device) != {
            "name",
            "compute_capability",
            "total_memory_bytes",
            "uuid",
        }:
            raise ValueError("Checkpoint training CUDA device provenance is malformed")
    return dict(source_hashes)


def _validate_checkpoint_metadata(
    checkpoint: Mapping,
    path: Path,
    expected: Mapping,
    named_parameter_specs: Sequence[Mapping],
) -> dict:
    step = parse_checkpoint_step(path)
    if checkpoint.get("step") != step:
        raise ValueError(f"Checkpoint payload step differs from {path.name}")
    if step not in expected["checkpoint_steps"]:
        raise ValueError(f"Checkpoint step {step} is not in the protocol")
    trainer_state = checkpoint.get("trainer_state")
    if not isinstance(trainer_state, Mapping):
        raise ValueError("Checkpoint trainer_state is absent")
    exact_trainer = {
        "next_step": step + 1,
        "global_seed": expected["global_seed"],
        "world_size": expected["world_size"],
    }
    for name, value in exact_trainer.items():
        if trainer_state.get(name) != value:
            raise ValueError(
                f"Checkpoint {path.name} trainer field {name} is invalid"
            )
    run_id = trainer_state.get("run_id")
    if not isinstance(run_id, str) or len(run_id) < 16:
        raise ValueError("Checkpoint run_id is absent or malformed")
    provenance = trainer_state.get("training_provenance")
    source_hashes = _validate_training_provenance(provenance, expected)

    _model_state(checkpoint)
    bind_optimizer_parameters(
        checkpoint.get("optimizer_state_dict"),
        named_parameter_specs,
        expected_optimizer_step=step + 1,
    )
    return {
        "step": step,
        "run_id": run_id,
        "training_provenance": _plain(provenance),
        "source_sha256": dict(source_hashes),
    }


def _repository_contract() -> dict:
    state = repository_state(PROJECT_ROOT)
    if state["status"]:
        raise RuntimeError("Expert update-budget audit requires a clean worktree")
    if not (
        state["commit"]
        == state["origin_repa"]
        == state["authoritative_remote_tip"]
    ):
        raise RuntimeError(
            "Expert update-budget audit requires HEAD == origin/repa == "
            "the authoritative remote tip"
        )
    return state


def _checkpoint_paths(checkpoint_dir: Path, steps: Sequence[int]) -> list[Path]:
    paths = [checkpoint_dir / f"ckpt_step_{step}.pth" for step in steps]
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint must be a regular file (the run directory may be a symlink): {path}"
            )
    return paths


def _runtime_metadata() -> dict:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload) -> None:
    _atomic_text(
        path,
        json.dumps(_plain(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _write_csv(path: Path, intervals: Sequence[Mapping]) -> None:
    fieldnames = [
        "from_step",
        "to_step",
        "block_index",
        "expert_index",
        "parameter_count",
        "parameter_rms_before",
        "parameter_rms_after",
        "displacement_rms",
        "relative_displacement",
        "net_displacement_per_step_rms",
        "debiased_first_moment_rms",
        "debiased_second_moment_rms",
        "preconditioned_moment_rms",
        "adamw_update_field_rms",
        "relative_adamw_update_field",
    ]
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for interval in intervals:
            for block in interval["blocks"].values():
                for expert in block["experts"]:
                    row = {
                        name: expert[name]
                        for name in fieldnames
                        if name not in {"from_step", "to_step"}
                    }
                    row.update({
                        "from_step": interval["from_step"],
                        "to_step": interval["to_step"],
                    })
                    writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _format_number(value) -> str:
    if value is None:
        return "不可计算"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def render_summary_markdown(payload: Mapping) -> str:
    gate = payload["gate"]
    observed = gate["observed"]
    status = "通过" if gate["passed"] else "未通过"
    lines = [
        "# 专家更新预算审计",
        "",
        f"结论：**{status}**。",
        "",
        "这份报告只看从零训练过程中保存的模型权重和 AdamW 状态。"
        "它检查同一个 MoE block 里的 12 个条件专家，长期参数变化是否明显不同。"
        "它不修改模型，也没有使用 REPA、DINO 或 teacher。",
        "",
        "## 主要数字",
        "",
        "| 检查项 | 观测值 |",
        "| --- | ---: |",
        "| 相对参数位移的中位 Gini | "
        + _format_number(observed["median_relative_displacement_gini"])
        + " |",
        "| 达到预设位移差异的 block/阶段比例 | "
        + _format_number(
            observed["fraction_cells_relative_displacement_gini_at_least_effect_size"]
        )
        + " |",
        "| AdamW 局部更新量的中位 Gini | "
        + _format_number(observed["median_relative_adamw_update_gini"])
        + " |",
        "| 长期位移与局部更新量的中位排序相关 | "
        + _format_number(observed["median_within_cell_displacement_update_spearman"])
        + " |",
        "| 相邻训练阶段的专家位移排序相关 | "
        + _format_number(observed["median_adjacent_interval_displacement_spearman"])
        + " |",
        "",
        "## 每项门槛",
        "",
        "| 项目 | 要求 | 观测值 | 是否通过 |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, check in gate["checks"].items():
        lines.append(
            f"| `{name}` | {check['required']} | "
            f"{_format_number(check['observed'])} | "
            f"{'是' if check['passed'] else '否'} |"
        )
    lines.extend([
        "",
        "## 这能说明什么",
        "",
        "只有全部门槛通过，才值得继续做配对的冻结路由统计："
        "在相同图像、噪声阶段和 block 内，把 token 数、输出侧学习信用和参数更新量放在一起比较。",
        "",
        "即使通过，也不能说明更新较少的专家一定学得差，更不能说明把梯度简单归一化会改善 FID。"
        "下一步仍需证明这种差异不能由 token 数、参数大小或专家本身的功能差异解释，"
        "并用专家对应关系打乱对照检验因果性。",
        "",
        "## 数据来源",
        "",
        f"- 训练 run ID：`{payload['lineage']['run_id']}`",
        f"- 训练代码提交：`{payload['lineage']['training_git_commit']}`",
        f"- 分析代码提交：`{payload['repository']['commit']}`",
        "- checkpoint："
        + ", ".join(str(item["step"]) for item in payload["checkpoints"]),
        "",
    ])
    return "\n".join(lines)


def _prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    archive_root = (PROJECT_ROOT / "analyses" / "archvied_analyses").resolve()
    resolved_parent = output_dir.parent.resolve()
    try:
        resolved_parent.relative_to(archive_root)
    except ValueError as error:
        raise ValueError(
            f"Analysis output must stay under {archive_root}: {output_dir}"
        ) from error
    if os.path.lexists(output_dir) and output_dir.is_symlink():
        raise ValueError(f"Analysis output directory must not be a symlink: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Analysis output path must be a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = list(output_dir.iterdir())
    unknown = [entry for entry in entries if entry.name not in OUTPUT_FILENAMES]
    if unknown:
        raise FileExistsError(
            f"Analysis output directory contains an unknown entry: {unknown[0]}"
        )
    invalid = [entry for entry in entries if entry.is_symlink() or not entry.is_file()]
    if invalid:
        raise FileExistsError(
            f"Analysis output entry must be a regular file: {invalid[0]}"
        )
    if entries and not overwrite:
        raise FileExistsError(
            f"Analysis output already exists; use --overwrite: {entries[0]}"
        )


def run_audit(
    *,
    manifest_path: Path,
    config_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict:
    manifest_path = Path(os.path.abspath(os.fspath(manifest_path)))
    config_path = Path(os.path.abspath(os.fspath(config_path)))
    checkpoint_dir = Path(checkpoint_dir).resolve()
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest must be a regular file: {manifest_path}")
    if config_path.is_symlink() or not config_path.is_file():
        raise FileNotFoundError(f"Config must be a regular file: {config_path}")
    manifest_root = (
        PROJECT_ROOT / "analyses" / "expert_update_budget" / "manifests"
    ).resolve()
    config_root = (PROJECT_ROOT / "configs").resolve()
    if manifest_path.parent != manifest_root:
        raise ValueError(f"Manifest must be stored directly under {manifest_root}")
    if config_path.parent != config_root:
        raise ValueError(f"Config must be stored directly under {config_root}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    manifest_sha256 = sha256_file(manifest_path)
    config_sha256 = sha256_file(config_path)
    manifest = _load_json(manifest_path)
    expected, thresholds = _validate_manifest(manifest)
    _validate_config(config_path, expected)
    repository = _repository_contract()
    checkpoint_paths = _checkpoint_paths(
        checkpoint_dir,
        expected["checkpoint_steps"],
    )
    checkpoint_snapshots = {
        path: {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in checkpoint_paths
    }
    runtime_cfg = load_runtime_cfg(config_path)
    if runtime_cfg.model_name != expected["model_name"]:
        raise ValueError("Runtime model name differs from the protocol")
    named_parameter_specs = build_named_parameter_specs(runtime_cfg)

    checkpoints = []
    intervals = []
    previous_checkpoint = None
    lineage = None
    for path in checkpoint_paths:
        checkpoint = load_checkpoint(path)
        metadata = _validate_checkpoint_metadata(
            checkpoint,
            path,
            expected,
            named_parameter_specs,
        )
        if lineage is None:
            verify_worktree_source_manifest(
                PROJECT_ROOT,
                expected["training_git_commit"],
                metadata["source_sha256"],
            )
            lineage = {
                "run_id": metadata["run_id"],
                "training_git_commit": expected["training_git_commit"],
                "training_provenance": metadata["training_provenance"],
            }
        elif (
            metadata["run_id"] != lineage["run_id"]
            or metadata["training_provenance"] != lineage["training_provenance"]
        ):
            raise ValueError("Checkpoint lineage changed within the trajectory")

        checkpoints.append({
            "step": metadata["step"],
            "path": str(path),
            "size_bytes": checkpoint_snapshots[path]["size_bytes"],
            "sha256": checkpoint_snapshots[path]["sha256"],
        })
        if previous_checkpoint is not None:
            intervals.append(analyze_checkpoint_interval(
                previous_checkpoint,
                checkpoint,
                named_parameter_specs,
                expected["moe_blocks"],
                expected["conditional_experts_per_block"],
                chunk_size=chunk_size,
            ))
            del previous_checkpoint
            gc.collect()
        previous_checkpoint = checkpoint
    del previous_checkpoint
    gc.collect()

    persistence = summarize_rank_persistence(intervals, expected["moe_blocks"])
    gate = evaluate_gate(
        intervals,
        persistence,
        thresholds,
        expected["moe_blocks"],
    )
    verify_unchanged_file(manifest_path, manifest_sha256, "Protocol manifest")
    verify_unchanged_file(config_path, config_sha256, "Training config")
    for path in checkpoint_paths:
        verify_unchanged_file(
            path,
            checkpoint_snapshots[path]["sha256"],
            "Checkpoint",
        )
    if _repository_contract() != repository:
        raise RuntimeError("Repository state changed while the audit was running")

    payload = {
        "audit_version": AUDIT_VERSION,
        "protocol": manifest,
        "protocol_path": str(manifest_path),
        "protocol_sha256": manifest_sha256,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "checkpoint_directory": str(checkpoint_dir),
        "repository": repository,
        "runtime": _runtime_metadata(),
        "lineage": lineage,
        "checkpoints": checkpoints,
        "intervals": intervals,
        "rank_persistence": persistence,
        "gate": gate,
    }
    payload["content_sha256"] = _json_sha256(payload)

    _prepare_output_directory(output_dir, overwrite)
    _atomic_json(output_dir / "audit.json", payload)
    summary = {
        "audit_version": AUDIT_VERSION,
        "protocol_name": manifest["protocol_name"],
        "protocol_sha256": payload["protocol_sha256"],
        "audit_content_sha256": payload["content_sha256"],
        "checkpoint_sha256": {
            str(item["step"]): item["sha256"] for item in checkpoints
        },
        "rank_persistence": persistence,
        "gate": gate,
    }
    _atomic_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "expert_metrics.csv", intervals)
    _atomic_text(output_dir / "summary.md", render_summary_markdown(payload))
    return payload
