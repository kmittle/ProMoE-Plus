"""Sealed training-time controller for expert credit-rate redistribution."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist

from .protocol_lock import load_effective_protocol
from .serialization import atomic_write_json, content_sha256, sha256_file
from .transcript import JsonlLedger, TrainingInputTranscript


CHECKPOINT_STATE_KEY = "credit_redistribution_state"
CONTROLLER_STATE_VERSION = 1
BRANCHES = (
    "measure_only_control",
    "rotating_permuted_scale_control",
    "matched_credit_rate_redistribution",
)


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def _all_reduce_sum(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def _distributed_error(local_error, phase):
    errors = [None] * _world_size()
    if dist.is_initialized():
        dist.all_gather_object(errors, local_error)
    else:
        errors[0] = local_error
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
    if failures:
        raise RuntimeError(f"{phase} failed; " + "; ".join(failures))


def _distributed_guard(phase, function):
    """Run fallible rank-local work and make every rank fail together."""
    result = None
    local_error = None
    try:
        result = function()
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _distributed_error(local_error, phase)
    return result


def _require_finite(tensor, name):
    if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} must be a finite tensor")


def _squared_norm(tensor):
    return tensor.detach().to(dtype=torch.float64).square().sum()


def _parameter_gradient_squared_norm(parameter, name):
    if parameter.grad is None:
        raise RuntimeError(f"Routed expert parameter has no gradient: {name}")
    _require_finite(parameter.grad, f"gradient {name}")
    return _squared_norm(parameter.grad)


def _expert_gradient_squared_norm(expert, prefix):
    total = None
    parameter_count = 0
    for name, parameter in expert.named_parameters():
        value = _parameter_gradient_squared_norm(parameter, f"{prefix}.{name}")
        total = value if total is None else total + value
        parameter_count += parameter.numel()
    if total is None or parameter_count == 0:
        raise RuntimeError(f"Routed expert has no parameters: {prefix}")
    return total


def _model_gradient_squared_norm(model):
    total = None
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        _require_finite(parameter.grad, f"gradient {name}")
        value = _squared_norm(parameter.grad)
        total = value if total is None else total + value
    if total is None or not bool(total > 0):
        raise RuntimeError("Full-model raw gradient budget is nonpositive")
    return total


def compute_budget_factor(expert_norms, scales):
    expert_norms = expert_norms.to(dtype=torch.float64)
    scales = scales.to(device=expert_norms.device, dtype=torch.float64)
    _require_finite(expert_norms, "expert raw-gradient norms")
    _require_finite(scales, "expert raw scales")
    numerator = expert_norms.sum()
    denominator = (expert_norms * scales.square()).sum()
    if not bool(numerator > 0) or not bool(denominator > 0):
        raise RuntimeError("Raw-gradient budget numerator and denominator must be positive")
    factor = torch.sqrt(numerator / denominator)
    _require_finite(factor, "raw-gradient budget factor")
    return factor


def rotating_permuted_scales(raw_scales, update_index):
    raw_scales = raw_scales.to(dtype=torch.float64)
    num_experts = raw_scales.shape[-1]
    if num_experts != 12:
        raise ValueError("The sealed rotating permutation requires 12 experts")
    offset = 1 + (int(update_index) % 11)
    indices = (
        torch.arange(num_experts, device=raw_scales.device) + offset
    ) % num_experts
    return raw_scales.index_select(-1, indices), offset


def deterministic_group_sum(values, indices, num_groups):
    """Sum float64 values by group without CUDA atomic accumulation."""
    values = values.reshape(-1).to(dtype=torch.float64)
    indices = indices.reshape(-1).to(device=values.device, dtype=torch.int64)
    num_groups = int(num_groups)
    if values.numel() != indices.numel():
        raise ValueError("Grouped values and indices must have equal length")
    if num_groups <= 0:
        raise ValueError("Grouped reduction requires at least one group")
    if indices.numel() and not bool(
        torch.all((indices >= 0) & (indices < num_groups))
    ):
        raise ValueError("Grouped reduction index is out of range")
    _require_finite(values, "grouped reduction values")
    group_ids = torch.arange(num_groups, device=values.device, dtype=torch.int64)
    membership = indices[:, None] == group_ids[None, :]
    grouped = torch.where(
        membership,
        values[:, None],
        torch.zeros((), device=values.device, dtype=torch.float64),
    ).sum(dim=0)
    counts = membership.sum(dim=0, dtype=torch.int64)
    _require_finite(grouped, "grouped reduction sums")
    return grouped, counts


class CreditRateNormalizer:
    def __init__(self, shape, device, ema_decay=0.99, epsilon=1e-30):
        self.ema_decay = float(ema_decay)
        self.epsilon = float(epsilon)
        if self.ema_decay != 0.99 or self.epsilon != 1e-30:
            raise ValueError("Credit normalizer constants differ from the sealed protocol")
        self.ema = torch.zeros(shape, device=device, dtype=torch.float64)
        self.initialized = torch.zeros(shape, device=device, dtype=torch.bool)

    def update(self, rates):
        rates = rates.to(device=self.ema.device, dtype=torch.float64)
        _require_finite(rates, "global expert credit rates")
        if not bool(torch.all(rates > 0)):
            raise RuntimeError("Every routed expert credit rate must be positive")
        initialized_count = int(self.initialized.sum().item())
        if initialized_count == 0:
            self.ema.copy_(rates)
            self.initialized.fill_(True)
        elif initialized_count == self.initialized.numel():
            self.ema.mul_(self.ema_decay).add_(
                rates, alpha=1.0 - self.ema_decay
            )
        else:
            raise RuntimeError("Partial credit-normalizer initialization is forbidden")
        _require_finite(self.ema, "credit-rate EMA")
        if not bool(torch.all(self.ema > 0)):
            raise RuntimeError("Every credit-rate EMA must remain positive")
        reference = self.ema.log().mean(dim=1).exp()
        raw = torch.sqrt(reference[:, None] / self.ema.clamp_min(self.epsilon))
        raw = raw.clamp(min=0.5, max=2.0)
        _require_finite(reference, "geometric-mean credit reference")
        _require_finite(raw, "raw expert scales")
        return reference, raw

    def state_dict(self):
        return {
            "ema": self.ema.detach().cpu().clone(),
            "initialized": self.initialized.detach().cpu().clone(),
            "ema_decay": self.ema_decay,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise TypeError("Normalizer checkpoint state must be a mapping")
        if state.get("ema_decay") != self.ema_decay:
            raise ValueError("Normalizer EMA decay changed across resume")
        if state.get("epsilon") != self.epsilon:
            raise ValueError("Normalizer epsilon changed across resume")
        ema = state.get("ema")
        initialized = state.get("initialized")
        if not torch.is_tensor(ema) or tuple(ema.shape) != tuple(self.ema.shape):
            raise ValueError("Normalizer EMA checkpoint shape differs")
        if ema.dtype != torch.float64:
            raise ValueError("Normalizer EMA checkpoint must use float64")
        if not torch.is_tensor(initialized) or initialized.dtype != torch.bool:
            raise ValueError("Normalizer initialization mask must be bool")
        if tuple(initialized.shape) != tuple(self.initialized.shape):
            raise ValueError("Normalizer initialization-mask shape differs")
        self.ema.copy_(ema.to(device=self.ema.device))
        self.initialized.copy_(initialized.to(device=self.initialized.device))
        _require_finite(self.ema, "restored credit-rate EMA")


@dataclass
class BlockCapture:
    route_weights: torch.Tensor | None = None
    route_indices: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    suffix_gradient: torch.Tensor | None = None


class CreditRedistributionController:
    """Measure suffix-gradient credit and optionally redistribute expert gradients."""

    def __init__(self, model, runtime_cfg, controller_cfg):
        self.checkpoint_state_key = CHECKPOINT_STATE_KEY
        if not isinstance(controller_cfg, dict):
            controller_cfg = dict(controller_cfg)
        if not bool(controller_cfg.get("enabled", False)):
            raise ValueError("Credit controller cannot be constructed while disabled")
        self.model = model
        self.runtime_cfg = runtime_cfg
        self.cfg = copy.deepcopy(controller_cfg)
        self.rank = _rank()
        self.world_size = _world_size()
        self.protocol = load_effective_protocol(
            self.cfg["preregister_v3_path"],
            self.cfg["preregister_v4_path"],
        )
        self.branch = str(self.cfg.get("branch"))
        self.execution_mode = str(
            self.cfg.get("execution_mode", "continuation")
        )
        if self.branch not in BRANCHES:
            raise ValueError(f"Unknown sealed branch: {self.branch}")
        if self.protocol["branches"]["names"] != list(BRANCHES):
            raise ValueError("Sealed branch order differs from the implementation")

        measurement = self.protocol["training_measurement"]
        source = self.protocol["source_anchor"]["training_facts"]
        self.block_indices = tuple(source["routed_blocks_zero_based"])
        self.num_experts = int(source["routed_experts_per_block"])
        self.start_step = int(self.protocol["branches"]["start_step"])
        self.last_step = int(self.protocol["branches"]["last_step"])
        self.relative_tolerance = float(
            self.protocol["raw_gradient_budget"]["relative_tolerance"]
        )
        if self.block_indices != (1, 3, 5, 7, 9, 11):
            raise ValueError("Sealed routed block set changed")
        if self.num_experts != 12 or measurement.get("scope") is None:
            raise ValueError("Sealed routed-expert contract changed")
        self._validate_runtime_contract(source)
        self.layers = self._resolve_layers()
        device = next(model.parameters()).device
        self.normalizer = CreditRateNormalizer(
            (len(self.block_indices), self.num_experts),
            device=device,
            ema_decay=float(self.protocol["normalizer"]["ema_decay"]),
            epsilon=float(self.protocol["normalizer"]["epsilon"]),
        )
        self.update_count = 0
        self.numerical_counters = {
            "nonfinite": 0,
            "rank_disagreement": 0,
            "transcript_mismatch": 0,
            "budget_violation": 0,
            "capture_failure": 0,
            "checkpoint_failure": 0,
        }
        self.current_step = None
        self._pending_optimizer_step = False
        self._captures = {block: BlockCapture() for block in self.block_indices}
        self._method_overrides = []
        self._forward_handles = []
        self._telemetry_snapshot = None

        self.artifact_root = Path(self.cfg["artifact_root"]).resolve()
        reference_root = self.cfg.get("reference_artifact_root")
        if self.branch == "measure_only_control":
            if reference_root is not None:
                raise ValueError("Measure-only branch cannot use a reference transcript")
        elif reference_root is None:
            raise ValueError("Feedback branches require the measure-only transcript root")
        self.transcript = TrainingInputTranscript(
            artifact_root=self.artifact_root,
            branch=self.branch,
            start_step=self.start_step,
            dataset_root=runtime_cfg.latent_data_path,
            reference_artifact_root=reference_root,
        )
        self.step_ledger = (
            JsonlLedger(
                self.artifact_root / "controller" / self.branch / "steps.jsonl",
                self.start_step,
            )
            if self.rank == 0
            else None
        )
        self._install_captures()

    @property
    def initial_checkpoint_path(self):
        configured = Path(self.cfg["initial_checkpoint_path"]).resolve()
        sealed = Path(self.protocol["checkpoint"]["frozen_path"]).resolve()
        if configured != sealed:
            raise ValueError("Configured initial checkpoint differs from the sealed path")
        return configured

    def verify_initial_checkpoint(self):
        error = None
        if self.rank == 0:
            try:
                path = self.initial_checkpoint_path
                if not path.is_file():
                    raise FileNotFoundError(f"Frozen checkpoint is absent: {path}")
                observed = sha256_file(path)
                expected = self.protocol["checkpoint"]["sha256"]
                if observed != expected:
                    raise RuntimeError(
                        f"Frozen checkpoint hash mismatch: {observed} != {expected}"
                    )
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
        errors = [error]
        if dist.is_initialized():
            dist.broadcast_object_list(errors, src=0)
        if errors[0]:
            raise RuntimeError(errors[0])
        if dist.is_initialized():
            dist.barrier()

    def _validate_runtime_contract(self, source):
        exact = {
            "world_size": (self.world_size, 4),
            "total_train_batch_size": (
                int(self.runtime_cfg.total_train_batch_size),
                int(source["global_batch_size"]),
            ),
            "grad_mix": (int(self.runtime_cfg.grad_mix), 1),
            "global_seed": (int(self.runtime_cfg.global_seed), 0),
        }
        for name, (actual, expected) in exact.items():
            if actual != expected:
                raise ValueError(f"{name}={actual!r}, expected sealed value {expected!r}")
        floats = {
            "lr": (float(self.runtime_cfg.lr), float(source["learning_rate"])),
            "weight_decay": (
                float(self.runtime_cfg.weight_decay),
                float(source["weight_decay"]),
            ),
            "max_grad_norm": (
                float(self.runtime_cfg.max_grad_norm),
                float(source["max_grad_norm"]),
            ),
        }
        for name, (actual, expected) in floats.items():
            if actual != expected:
                raise ValueError(f"{name}={actual!r}, expected sealed value {expected!r}")
        if bool(self.runtime_cfg.use_gradient_checkpointing):
            raise ValueError("Sealed controller forbids gradient checkpoint recomputation")
        if str(self.runtime_cfg.model_name) != self.protocol["checkpoint"]["model_name"]:
            raise ValueError("Credit continuation requires the sealed Base model")
        execution_contracts = {
            "continuation": {
                "num_steps": self.last_step + 1,
                "save_ckpt_interval": int(
                    self.protocol["branches"]["save_checkpoint_interval"]
                ),
                "branches": BRANCHES,
            },
            "deterministic_replay": {
                "num_steps": self.start_step + 20,
                "save_ckpt_interval": 20,
                "branches": ("measure_only_control",),
            },
            "throughput": {
                "num_steps": self.start_step + 600,
                "save_ckpt_interval": None,
                "branches": ("matched_credit_rate_redistribution",),
            },
        }
        contract = execution_contracts.get(self.execution_mode)
        if contract is None:
            raise ValueError(f"Unknown credit execution mode: {self.execution_mode}")
        if self.branch not in contract["branches"]:
            raise ValueError(
                f"Branch {self.branch} is invalid for {self.execution_mode}"
            )
        if int(self.runtime_cfg.num_steps) != contract["num_steps"]:
            raise ValueError(
                f"num_steps differs for credit mode {self.execution_mode}"
            )
        if (
            contract["save_ckpt_interval"] is not None
            and int(self.runtime_cfg.save_ckpt_interval)
            != contract["save_ckpt_interval"]
        ):
            raise ValueError(
                f"Checkpoint interval differs for credit mode {self.execution_mode}"
            )
        if bool(getattr(self.runtime_cfg, "structured_batch_sampling", False)):
            raise ValueError("Credit continuation requires the standard sampler")
        self.expected_update_total = contract["num_steps"] - self.start_step

    def _resolve_layers(self):
        if not hasattr(self.model, "blocks"):
            raise TypeError("Credit controller requires model.blocks")
        layers = {}
        for block_index in self.block_indices:
            block = self.model.blocks[block_index]
            if not bool(getattr(block, "use_moe", False)):
                raise TypeError(f"Block {block_index} is not a routed MoE block")
            layer = getattr(block, "mlp", None)
            if int(getattr(layer, "num_routed_experts", -1)) != self.num_experts:
                raise ValueError(f"Block {block_index} routed-expert count differs")
            if int(getattr(layer, "top_k", -1)) != 1:
                raise ValueError("Sealed controller requires top_k=1")
            if getattr(layer, "router_weight_mode", None) != "identity":
                raise ValueError("Sealed controller requires identity cosine weights")
            if len(getattr(layer, "experts", ())) < self.num_experts:
                raise ValueError(f"Block {block_index} has too few expert modules")
            layers[block_index] = layer
        return layers

    def _install_captures(self):
        for block_index, layer in self.layers.items():
            if "compute_router" in layer.__dict__:
                raise RuntimeError("MoE layer already has a compute_router override")
            original = layer.compute_router

            def wrapped(this, hidden_states, labels, _original=original, _block=block_index):
                result = _original(hidden_states, labels)
                self._capture_router(_block, labels, result)
                return result

            layer.compute_router = MethodType(wrapped, layer)
            self._method_overrides.append(layer)

            def capture_output(module, inputs, output, _block=block_index):
                del module, inputs
                if self.current_step is None:
                    raise RuntimeError("MoE output occurred outside a controller step")
                if not isinstance(output, tuple) or len(output) != 2:
                    raise TypeError("Sparse MoE output contract changed")
                tensor = output[0]
                if self._captures[_block].suffix_gradient is not None:
                    raise RuntimeError(f"Block {_block} produced multiple outputs in one step")

                def capture_gradient(gradient):
                    capture = self._captures[_block]
                    if capture.suffix_gradient is not None:
                        raise RuntimeError(
                            f"Block {_block} produced multiple suffix gradients"
                        )
                    capture.suffix_gradient = gradient.detach()
                    return gradient

                tensor.register_hook(capture_gradient)
                return None

            self._forward_handles.append(layer.register_forward_hook(capture_output))

    def _capture_router(self, block_index, labels, result):
        if self.current_step is None:
            raise RuntimeError("Router execution occurred outside a controller step")
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("Router return contract changed")
        capture = self._captures[block_index]
        if capture.route_weights is not None:
            raise RuntimeError(f"Block {block_index} routed more than once in one step")
        weights, indices, _ = result
        capture.route_weights = weights.detach()
        capture.route_indices = indices.detach()
        capture.labels = labels.detach()

    def begin_step(self, step):
        step = int(step)
        expected = self.start_step + self.update_count
        if self.current_step is not None or self._pending_optimizer_step:
            raise RuntimeError("Previous credit-controller step is incomplete")
        if step != expected:
            raise ValueError(f"Credit-controller step {step} does not equal {expected}")
        if not self.start_step <= step <= self.last_step:
            raise ValueError("Training step lies outside the sealed continuation window")
        self.current_step = step
        self._captures = {block: BlockCapture() for block in self.block_indices}

    def effective_labels(self):
        labels = [self._captures[block].labels for block in self.block_indices]
        if any(value is None for value in labels):
            raise RuntimeError("Effective class-dropout labels were not captured")
        first = labels[0]
        if any(not torch.equal(first, value) for value in labels[1:]):
            raise RuntimeError("Effective labels differ across routed blocks")
        return first

    def _local_credit_and_count(self):
        device = next(self.model.parameters()).device
        credit = torch.zeros(
            len(self.block_indices), self.num_experts,
            device=device, dtype=torch.float64,
        )
        count = torch.zeros(
            len(self.block_indices), self.num_experts,
            device=device, dtype=torch.int64,
        )
        for row, block_index in enumerate(self.block_indices):
            capture = self._captures[block_index]
            if any(
                value is None
                for value in (
                    capture.route_weights,
                    capture.route_indices,
                    capture.labels,
                    capture.suffix_gradient,
                )
            ):
                raise RuntimeError(f"Block {block_index} capture is incomplete")
            weights = capture.route_weights
            indices = capture.route_indices
            gradient = capture.suffix_gradient
            if weights.shape != indices.shape or weights.shape[-1] != 1:
                raise ValueError("Route weights and IDs violate top-1 shape contract")
            if gradient.shape != weights.shape[:-1] + (gradient.shape[-1],):
                raise ValueError("Suffix gradient and route tensors are misaligned")
            flat_weights = weights.reshape(-1)
            flat_indices = indices.reshape(-1)
            flat_gradient = gradient.reshape(flat_indices.numel(), -1)
            labels = capture.labels.reshape(-1, 1).expand(
                -1, weights.shape[1]
            ).reshape(-1)
            conditional = labels != 1000
            if not torch.equal(conditional, flat_indices < self.num_experts):
                raise RuntimeError("Conditional labels and routed expert IDs disagree")
            selected_indices = flat_indices[conditional].to(dtype=torch.int64)
            selected_weights = flat_weights[conditional]
            selected_gradient = flat_gradient[conditional]
            if selected_indices.numel() == 0:
                raise RuntimeError("A routed block has no conditional tokens")
            chunk_size = 8192
            for start in range(0, selected_indices.numel(), chunk_size):
                end = min(start + chunk_size, selected_indices.numel())
                chunk_indices = selected_indices[start:end]
                chunk_gradient = selected_gradient[start:end].to(torch.float64)
                chunk_weights = selected_weights[start:end].to(torch.float64)
                _require_finite(chunk_gradient, "conditional suffix gradient")
                _require_finite(chunk_weights, "conditional route weight")
                token_credit = chunk_weights.square() * chunk_gradient.square().sum(1)
                _require_finite(token_credit, "conditional token credit")
                chunk_credit, chunk_count = deterministic_group_sum(
                    token_credit,
                    chunk_indices,
                    self.num_experts,
                )
                credit[row].add_(chunk_credit)
                count[row].add_(chunk_count)
        return credit, count

    def _gradient_norm_matrix(self):
        device = next(self.model.parameters()).device
        norms = torch.zeros(
            len(self.block_indices), self.num_experts,
            device=device, dtype=torch.float64,
        )
        for row, block_index in enumerate(self.block_indices):
            layer = self.layers[block_index]
            for expert_index in range(self.num_experts):
                norms[row, expert_index] = _expert_gradient_squared_norm(
                    layer.experts[expert_index],
                    f"blocks.{block_index}.mlp.experts.{expert_index}",
                )
        _require_finite(norms, "routed-expert raw-gradient norms")
        if not bool(torch.all(norms > 0)):
            raise RuntimeError(
                "Every active routed expert must have a positive raw-gradient budget"
            )
        return norms

    @staticmethod
    def _relative_drift(before, after):
        if not bool(before > 0):
            raise RuntimeError("Raw-gradient budget denominator is nonpositive")
        return torch.abs(after - before) / before

    def _rank_consensus(self, payload):
        digest = _distributed_guard(
            "rank-consensus serialization",
            lambda: content_sha256(payload),
        )
        digests = [None] * self.world_size
        if dist.is_initialized():
            dist.all_gather_object(digests, digest)
        else:
            digests[0] = digest
        if len(set(digests)) != 1:
            self.numerical_counters["rank_disagreement"] += 1
            raise RuntimeError(f"Credit-controller rank disagreement: {digests}")
        return digest

    def _apply_scales(self, applied_scales):
        if self.branch == "measure_only_control":
            if not torch.equal(applied_scales, torch.ones_like(applied_scales)):
                raise RuntimeError("Measure-only branch must apply exact unit scales")
            return
        for row, block_index in enumerate(self.block_indices):
            layer = self.layers[block_index]
            for expert_index in range(self.num_experts):
                scale = float(applied_scales[row, expert_index].item())
                if not math.isfinite(scale) or scale <= 0:
                    raise FloatingPointError("Applied expert scale must be finite and positive")
                for parameter in layer.experts[expert_index].parameters():
                    parameter.grad.mul_(scale)

    def _should_record_adamw(self, step):
        return self.start_step <= step <= self.start_step + 9 or step % 1000 == 0

    def _optimizer_moment_norms(self, optimizer):
        result = {}
        for block_index in self.block_indices:
            layer = self.layers[block_index]
            block_result = []
            for expert_index in range(self.num_experts):
                first = torch.zeros((), dtype=torch.float64)
                second = torch.zeros((), dtype=torch.float64)
                exposed = True
                for parameter in layer.experts[expert_index].parameters():
                    state = optimizer.state.get(parameter, {})
                    if "exp_avg" not in state or "exp_avg_sq" not in state:
                        exposed = False
                        break
                    _require_finite(state["exp_avg"], "AdamW first moment")
                    _require_finite(state["exp_avg_sq"], "AdamW second moment")
                    first += _squared_norm(state["exp_avg"]).cpu()
                    second += _squared_norm(state["exp_avg_sq"]).cpu()
                block_result.append({
                    "exposed": exposed,
                    "first_moment_squared_norm": float(first.item()) if exposed else None,
                    "second_moment_squared_norm": float(second.item()) if exposed else None,
                })
            result[str(block_index)] = block_result
        return result

    def _prepare_adamw_snapshot(self, optimizer, pre_norms, post_norms):
        if self.rank != 0 or not self._should_record_adamw(self.current_step):
            self._telemetry_snapshot = None
            return
        parameters = {}
        for block_index in self.block_indices:
            layer = self.layers[block_index]
            for expert_index in range(self.num_experts):
                parameters[(block_index, expert_index)] = [
                    parameter.detach().clone()
                    for parameter in layer.experts[expert_index].parameters()
                ]
        self._telemetry_snapshot = {
            "step": self.current_step,
            "parameters": parameters,
            "pre_scale_raw_gradient_squared_norm": pre_norms.detach().cpu(),
            "applied_raw_gradient_squared_norm": post_norms.detach().cpu(),
            "moments_before": self._optimizer_moment_norms(optimizer),
        }

    def after_backward(self, optimizer, transcript_inputs, scaler_enabled=False):
        if self.current_step is None:
            raise RuntimeError("Credit controller has no active step")
        if scaler_enabled:
            raise RuntimeError("Sealed credit measurement forbids an enabled GradScaler")
        if self._pending_optimizer_step:
            raise RuntimeError("Credit controller already modified this step")

        preparation_error = None
        effective_labels = None
        tensors = None
        try:
            effective_labels = self.effective_labels()
            if not isinstance(transcript_inputs, dict):
                raise TypeError("transcript_inputs must be a mapping")
            tensors = dict(transcript_inputs["tensors"])
            tensors["effective_labels"] = effective_labels
        except Exception as error:
            self.numerical_counters["transcript_mismatch"] += 1
            preparation_error = f"{type(error).__name__}: {error}"
        _distributed_error(preparation_error, "credit transcript preparation")

        transcript_error = None
        global_transcript_digest = None
        try:
            global_transcript_digest = self.transcript.record(
                step=self.current_step,
                paths=transcript_inputs["paths"],
                original_labels=transcript_inputs["original_labels"],
                tensors=tensors,
            )
        except Exception as error:
            self.numerical_counters["transcript_mismatch"] += 1
            transcript_error = f"{type(error).__name__}: {error}"
        _distributed_error(transcript_error, "credit transcript recording")

        local_error = None
        local_credit = local_count = None
        try:
            local_credit, local_count = self._local_credit_and_count()
        except Exception as error:
            self.numerical_counters["capture_failure"] += 1
            local_error = f"{type(error).__name__}: {error}"
        _distributed_error(local_error, "local credit measurement")

        global_credit = local_credit.clone()
        global_count = local_count.clone()
        _all_reduce_sum(global_credit)
        _all_reduce_sum(global_count)
        def build_policy():
            _require_finite(global_credit, "global expert credit")
            if not bool(torch.all(global_count > 0)):
                raise RuntimeError(
                    "Every routed expert must be globally active on every update"
                )
            if not bool(torch.all(global_credit > 0)):
                raise RuntimeError("Every routed expert must have positive global credit")
            rates = global_credit / global_count.to(dtype=torch.float64)
            reference, raw_scales = self.normalizer.update(rates)
            permuted_scales, permutation_offset = rotating_permuted_scales(
                raw_scales, self.update_count
            )
            return (
                rates,
                reference,
                raw_scales,
                permuted_scales,
                permutation_offset,
            )

        (
            rates,
            reference,
            raw_scales,
            permuted_scales,
            permutation_offset,
        ) = _distributed_guard("credit policy construction", build_policy)

        local_error = None
        pre_norms = full_before = None
        try:
            pre_norms = self._gradient_norm_matrix()
            full_before = _model_gradient_squared_norm(self.model)
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _distributed_error(local_error, "raw-gradient budget measurement")

        def build_budget():
            matched_factors = torch.stack([
                compute_budget_factor(pre_norms[row], raw_scales[row])
                for row in range(len(self.block_indices))
            ])
            permuted_factors = torch.stack([
                compute_budget_factor(pre_norms[row], permuted_scales[row])
                for row in range(len(self.block_indices))
            ])
            if self.branch == "measure_only_control":
                selected_raw = torch.ones_like(raw_scales)
                selected_factors = torch.ones_like(matched_factors)
            elif self.branch == "rotating_permuted_scale_control":
                selected_raw = permuted_scales
                selected_factors = permuted_factors
            else:
                selected_raw = raw_scales
                selected_factors = matched_factors
            applied_scales = selected_raw * selected_factors[:, None]
            _require_finite(applied_scales, "applied expert scales")
            return (
                matched_factors,
                permuted_factors,
                selected_factors,
                applied_scales,
            )

        (
            matched_factors,
            permuted_factors,
            selected_factors,
            applied_scales,
        ) = _distributed_guard("raw-gradient budget construction", build_budget)

        consensus_digest = self._rank_consensus({
            "step": self.current_step,
            "update_count": self.update_count,
            "global_credit": global_credit,
            "global_count": global_count,
            "ema": self.normalizer.ema,
            "initialized": self.normalizer.initialized,
            "reference": reference,
            "raw_scales": raw_scales,
            "permuted_scales": permuted_scales,
            "matched_factors": matched_factors,
            "permuted_factors": permuted_factors,
            "selected_factors": selected_factors,
            "applied_scales": applied_scales,
            "permutation_offset": permutation_offset,
            "pre_gradient_norms": pre_norms,
            "full_pre_gradient_squared_norm": full_before,
        })

        def apply_and_verify_budget():
            self._apply_scales(applied_scales)
            post_norms = self._gradient_norm_matrix()
            full_after = _model_gradient_squared_norm(self.model)
            block_drifts = torch.stack([
                self._relative_drift(pre_norms[row].sum(), post_norms[row].sum())
                for row in range(len(self.block_indices))
            ])
            full_drift = self._relative_drift(full_before, full_after)
            _require_finite(block_drifts, "per-block raw-gradient budget drift")
            _require_finite(full_drift, "full-model raw-gradient budget drift")
            if bool(torch.any(block_drifts > self.relative_tolerance)) or bool(
                full_drift > self.relative_tolerance
            ):
                self.numerical_counters["budget_violation"] += 1
                raise RuntimeError(
                    "Raw-gradient budget drift exceeded the sealed 1e-6 tolerance"
                )
            return post_norms, block_drifts, full_after, full_drift

        post_norms, block_drifts, full_after, full_drift = _distributed_guard(
            "raw-gradient budget application",
            apply_and_verify_budget,
        )

        self._prepare_adamw_snapshot(optimizer, pre_norms, post_norms)
        def persist_step():
            if self.rank != 0:
                return
            self.step_ledger.append_or_verify({
                "version": CONTROLLER_STATE_VERSION,
                "step": self.current_step,
                "branch": self.branch,
                "update_index": self.update_count,
                "global_transcript_digest": global_transcript_digest,
                "rank_consensus_digest": consensus_digest,
                "permutation_offset": permutation_offset,
                "global_credit": global_credit.detach().cpu().tolist(),
                "global_count": global_count.detach().cpu().tolist(),
                "credit_rate_ema": self.normalizer.ema.detach().cpu().tolist(),
                "raw_scales": raw_scales.detach().cpu().tolist(),
                "permuted_scales": permuted_scales.detach().cpu().tolist(),
                "selected_budget_factors": selected_factors.detach().cpu().tolist(),
                "applied_scales": applied_scales.detach().cpu().tolist(),
                "pre_gradient_squared_norm": pre_norms.detach().cpu().tolist(),
                "post_gradient_squared_norm": post_norms.detach().cpu().tolist(),
                "full_pre_gradient_squared_norm": float(full_before.item()),
                "full_post_gradient_squared_norm": float(full_after.item()),
                "block_relative_budget_drift": block_drifts.detach().cpu().tolist(),
                "full_relative_budget_drift": float(full_drift.item()),
            })

        _distributed_guard("controller-step persistence", persist_step)
        self._pending_optimizer_step = True
        return {
            "credit_rate_min": float(rates.min().item()),
            "credit_rate_max": float(rates.max().item()),
            "raw_scale_min": float(raw_scales.min().item()),
            "raw_scale_max": float(raw_scales.max().item()),
            "full_budget_drift": float(full_drift.item()),
        }

    def after_optimizer_step(self, optimizer):
        if not self._pending_optimizer_step or self.current_step is None:
            raise RuntimeError("No pending credit-controller optimizer step")
        _distributed_guard(
            "AdamW telemetry persistence",
            lambda: self._persist_adamw_telemetry(optimizer),
        )
        self._telemetry_snapshot = None
        self.update_count += 1
        self._pending_optimizer_step = False
        self.current_step = None

    def _persist_adamw_telemetry(self, optimizer):
        if self.rank == 0 and self._telemetry_snapshot is not None:
            snapshot = self._telemetry_snapshot
            parameter_delta = {}
            for block_index in self.block_indices:
                layer = self.layers[block_index]
                values = []
                for expert_index in range(self.num_experts):
                    before = snapshot["parameters"][(block_index, expert_index)]
                    total = torch.zeros((), dtype=torch.float64)
                    for previous, parameter in zip(
                        before, layer.experts[expert_index].parameters()
                    ):
                        _require_finite(parameter, "post-AdamW routed-expert parameter")
                        delta = parameter.detach() - previous
                        _require_finite(delta, "post-AdamW routed-expert parameter delta")
                        total += _squared_norm(delta).cpu()
                    values.append(float(total.item()))
                parameter_delta[str(block_index)] = values
            payload = {
                "version": CONTROLLER_STATE_VERSION,
                "step": self.current_step,
                "branch": self.branch,
                "pre_scale_raw_gradient_squared_norm": snapshot[
                    "pre_scale_raw_gradient_squared_norm"
                ].tolist(),
                "applied_raw_gradient_squared_norm": snapshot[
                    "applied_raw_gradient_squared_norm"
                ].tolist(),
                "parameter_delta_squared_norm": parameter_delta,
                "moments_before": snapshot["moments_before"],
                "moments_after": self._optimizer_moment_norms(optimizer),
            }
            path = (
                self.artifact_root
                / "controller"
                / self.branch
                / "adamw_telemetry"
                / f"step-{self.current_step:06d}.json"
            )
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    if json.load(handle) != payload:
                        raise RuntimeError("Replayed AdamW telemetry differs")
            else:
                atomic_write_json(path, payload)

    def checkpoint_state_dict(self):
        if self.current_step is not None or self._pending_optimizer_step:
            raise RuntimeError("Cannot checkpoint an incomplete controller update")
        return {
            "version": CONTROLLER_STATE_VERSION,
            "branch": self.branch,
            "execution_mode": self.execution_mode,
            "block_indices": list(self.block_indices),
            "num_experts": self.num_experts,
            "start_step": self.start_step,
            "last_step": self.last_step,
            "update_count": self.update_count,
            "normalizer": self.normalizer.state_dict(),
            "numerical_counters": copy.deepcopy(self.numerical_counters),
        }

    def assert_checkpoint_state_consistent(self, state):
        self._rank_consensus({"checkpoint_state": state})

    def prepare_checkpoint_state(self, checkpoint, is_initial):
        checkpoint_step = checkpoint.get("step")
        state = checkpoint.get(CHECKPOINT_STATE_KEY)
        if is_initial:
            if checkpoint_step != self.start_step - 1:
                raise ValueError("Initial checkpoint step differs from sealed continuation")
            if state is not None:
                raise ValueError("Frozen initial checkpoint unexpectedly has controller state")
            return None
        if not isinstance(state, dict):
            raise ValueError("Branch checkpoint lacks credit-controller state")
        if state.get("version") != CONTROLLER_STATE_VERSION:
            raise ValueError("Unsupported credit-controller checkpoint version")
        if state.get("branch") != self.branch:
            raise ValueError("Credit-controller branch differs across resume")
        if state.get("execution_mode") != self.execution_mode:
            raise ValueError("Credit-controller execution mode differs across resume")
        if state.get("block_indices") != list(self.block_indices):
            raise ValueError("Credit-controller block set differs across resume")
        if state.get("num_experts") != self.num_experts:
            raise ValueError("Credit-controller expert count differs across resume")
        if state.get("start_step") != self.start_step or state.get("last_step") != self.last_step:
            raise ValueError("Credit-controller continuation window differs across resume")
        expected_count = int(checkpoint_step) - self.start_step + 1
        if state.get("update_count") != expected_count:
            raise ValueError("Credit-controller update count differs from checkpoint step")
        counters = state.get("numerical_counters")
        if counters != {key: 0 for key in self.numerical_counters}:
            raise ValueError("Cannot resume a branch with numerical-integrity failures")
        probe = CreditRateNormalizer(
            self.normalizer.ema.shape,
            device=self.normalizer.ema.device,
            ema_decay=self.normalizer.ema_decay,
            epsilon=self.normalizer.epsilon,
        )
        probe.load_state_dict(state.get("normalizer"))
        if expected_count > 0 and not bool(torch.all(probe.initialized)):
            raise ValueError("Resumed normalizer is not fully initialized")
        return copy.deepcopy(state)

    def commit_checkpoint_state(self, prepared_state):
        if prepared_state is None:
            self.update_count = 0
            return
        self.normalizer.load_state_dict(prepared_state["normalizer"])
        self.update_count = int(prepared_state["update_count"])
        self.numerical_counters = copy.deepcopy(
            prepared_state["numerical_counters"]
        )

    def close(self):
        if self.current_step is not None or self._pending_optimizer_step:
            raise RuntimeError("Cannot close an incomplete credit-controller step")
        if self.update_count != self.expected_update_total:
            raise RuntimeError("Credit-controller run did not complete its locked updates")
        for handle in self._forward_handles:
            handle.remove()
        self._forward_handles.clear()
        for layer in self._method_overrides:
            del layer.compute_router
        self._method_overrides.clear()
