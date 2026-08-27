"""Frozen-checkpoint probe for exact within-expert FFN-pass exchange."""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from analyses.denoising_regret.probe import (
    RoutingProbeCapture,
    _all_router_weights,
    _compute_router,
    _configure_torch_threads,
    _evaluate_experts,
    _extract_prediction,
    _load_checkpoint_model,
    _load_latent,
    _per_sample_mse,
    _rankdata,
)
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)
from analyses.timestep_utility.probe import (
    _forced_native_control,
    _forced_route_state,
    _validate_moe_block_contract,
)


PROBE_VERSION = 1
CANDIDATE_COUNT = 64
EXCHANGE_QUOTA = 0.10
NUMERICAL_EPSILON = 1e-7
MAX_CANDIDATE_ATTEMPTS = 4096
SELECTOR_NAMES = (
    "first_order",
    "random",
    "router_margin",
    "rolled_utility",
)


def _stable_seed(seed, *parts):
    payload = "|".join((str(int(seed)), *(str(part) for part in parts)))
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) % (2 ** 63)


def _quota_for_count(count, quota):
    count = int(count)
    quota = float(quota)
    if count < 0 or not 0.0 < quota <= 0.5:
        raise ValueError("Expert count and exchange quota are invalid")
    return min(int(np.floor(quota * count + 0.5)), count // 2)


def _logical_pass_counts(native_experts, donors, receivers, num_experts):
    native_experts = np.asarray(native_experts, dtype=np.int64)
    donors = np.asarray(donors, dtype=np.int64)
    receivers = np.asarray(receivers, dtype=np.int64)
    if native_experts.ndim != 1:
        raise ValueError("Native expert IDs must be a vector")
    if donors.ndim != 1 or receivers.ndim != 1:
        raise ValueError("Donors and receivers must be vectors")
    if donors.size != receivers.size:
        raise ValueError("Every skipped pass must have one receiving pass")
    if donors.size and (
        donors.min() < 0
        or receivers.min() < 0
        or donors.max() >= native_experts.size
        or receivers.max() >= native_experts.size
    ):
        raise ValueError("Exchange token indices lie outside the sequence")
    if set(donors.tolist()) & set(receivers.tolist()):
        raise ValueError("Donors and receivers must be disjoint")
    if donors.size and not np.array_equal(
        native_experts[donors],
        native_experts[receivers],
    ):
        raise ValueError("Every donor and receiver pair must share an expert")

    native = np.bincount(native_experts, minlength=int(num_experts))
    candidate = native.copy()
    if donors.size:
        candidate -= np.bincount(
            native_experts[donors],
            minlength=int(num_experts),
        )
        candidate += np.bincount(
            native_experts[receivers],
            minlength=int(num_experts),
        )
    return native, candidate


def _validate_candidate(candidate, native_experts, num_experts):
    donors = np.asarray(candidate["donors"], dtype=np.int64)
    receivers = np.asarray(candidate["receivers"], dtype=np.int64)
    experts = np.asarray(candidate["experts"], dtype=np.int64)
    if donors.size == 0:
        raise ValueError("A compute-exchange candidate cannot be empty")
    if experts.shape != donors.shape:
        raise ValueError("Candidate expert IDs must align with token pairs")
    if not np.array_equal(native_experts[donors], experts):
        raise ValueError("Donor expert metadata differs from native routing")
    if not np.array_equal(native_experts[receivers], experts):
        raise ValueError("Receiver expert metadata differs from native routing")
    native_counts, candidate_counts = _logical_pass_counts(
        native_experts,
        donors,
        receivers,
        num_experts,
    )
    if not np.array_equal(native_counts, candidate_counts):
        raise RuntimeError("Per-expert logical pass counts are not preserved")
    if candidate["native_pass_vector"] != native_counts.tolist():
        raise ValueError("Candidate native pass-vector metadata is stale")
    if candidate["candidate_pass_vector"] != candidate_counts.tolist():
        raise ValueError("Candidate pass-vector metadata is stale")
    if candidate["transferred_passes"] != donors.size:
        raise ValueError("Candidate transfer-count metadata is stale")


def build_same_expert_exchange_candidates(
    native_experts,
    num_experts,
    seed,
    quota=EXCHANGE_QUOTA,
    candidate_count=CANDIDATE_COUNT,
):
    native_experts = np.asarray(native_experts, dtype=np.int64)
    num_experts = int(num_experts)
    candidate_count = int(candidate_count)
    if native_experts.ndim != 1 or native_experts.size == 0:
        raise ValueError("Native expert IDs must be a nonempty vector")
    if num_experts <= 0 or native_experts.min() < 0 or native_experts.max() >= num_experts:
        raise ValueError("Native expert IDs lie outside the routed-expert range")
    if candidate_count <= 0:
        raise ValueError("Candidate count must be positive")

    pools = {
        expert: np.flatnonzero(native_experts == expert).astype(np.int64)
        for expert in range(num_experts)
    }
    quotas = {
        expert: _quota_for_count(pool.size, quota)
        for expert, pool in pools.items()
    }
    eligible = tuple(expert for expert, count in quotas.items() if count > 0)
    if not eligible:
        raise RuntimeError("No routed expert has enough tokens for compute exchange")

    generator = np.random.default_rng(int(seed))
    native_counts = np.bincount(native_experts, minlength=num_experts)
    signatures = set()
    candidates = []
    attempts = 0
    while len(candidates) < candidate_count and attempts < MAX_CANDIDATE_ATTEMPTS:
        attempts += 1
        donors = []
        receivers = []
        experts = []
        quota_by_expert = [0] * num_experts
        for expert in eligible:
            count = quotas[expert]
            order = generator.permutation(pools[expert])
            donor_tokens = order[:count]
            receiver_tokens = order[count:2 * count]
            donors.extend(int(token) for token in donor_tokens)
            receivers.extend(int(token) for token in receiver_tokens)
            experts.extend([int(expert)] * count)
            quota_by_expert[expert] = int(count)
        signature = tuple(sorted(zip(donors, receivers, experts)))
        if signature in signatures:
            continue
        signatures.add(signature)
        candidate = {
            "id": f"exchange:{len(candidates):03d}",
            "donors": donors,
            "receivers": receivers,
            "experts": experts,
            "quota": float(quota),
            "quota_by_expert": quota_by_expert,
            "transferred_passes": len(donors),
            "native_pass_vector": native_counts.tolist(),
            "candidate_pass_vector": native_counts.tolist(),
        }
        _validate_candidate(candidate, native_experts, num_experts)
        candidates.append(candidate)
    if len(candidates) != candidate_count:
        raise RuntimeError("Could not generate the locked number of unique candidates")
    return candidates


def _validate_compute_exchange_contract(model, block_indices):
    contract = _validate_moe_block_contract(model, block_indices)
    if model.training:
        raise ValueError("The compute-exchange probe requires model.eval()")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError("The compute-exchange probe requires frozen parameters")
    for block in contract["blocks"]:
        moe_layer = model.blocks[block["index"]].mlp
        if int(moe_layer.num_experts) <= int(moe_layer.num_routed_experts):
            raise ValueError("The probe requires a separate unconditional expert")
        if not getattr(moe_layer, "use_shared_expert", False):
            raise ValueError("The probe requires the Base shared-expert path")
    return contract


class _ComputeExchangeInjector:
    """Patch one MoE output while retaining the exact native dispatch."""

    def __init__(self, moe_layer, route_ids, route_weights, actions):
        if route_ids.ndim != 2 or route_weights.shape != route_ids.shape:
            raise ValueError("Forced route IDs and weights must be aligned matrices")
        if route_ids.dtype != torch.long:
            raise ValueError("Forced route IDs must use torch.long")
        if len(actions) != route_ids.shape[0]:
            raise ValueError("One action is required for every batch row")
        if not bool(torch.isfinite(route_weights).all().item()):
            raise ValueError("Forced route weights must be finite")
        if hasattr(moe_layer, "_compute_exchange_probe_active"):
            raise RuntimeError("Compute-exchange overrides cannot be nested")

        self.moe_layer = moe_layer
        self.route_ids = route_ids
        self.route_weights = route_weights
        self.actions = tuple(actions)
        self.first_outputs = {}
        self.second_pass_shapes = []
        self.in_second_pass = False
        self._handles = []
        moe_layer._compute_exchange_probe_active = True
        self._handles.append(moe_layer.register_forward_pre_hook(self._before_moe))
        for expert_id, expert in enumerate(moe_layer.experts):
            self._handles.append(expert.register_forward_hook(
                self._capture_expert(expert_id)
            ))
        self._handles.append(moe_layer.register_forward_hook(self._patch_moe))

    def _before_moe(self, module, inputs):
        if self.in_second_pass:
            return None
        hidden_states = inputs[0]
        if hidden_states.shape[:2] != self.route_ids.shape:
            raise RuntimeError("Exchange route matrices do not match the MoE input")
        self.first_outputs = {}
        self.second_pass_shapes = []
        return None

    def _capture_expert(self, expert_id):
        def capture(module, inputs, output):
            if self.in_second_pass:
                return None
            if expert_id in self.first_outputs:
                raise RuntimeError("A routed expert ran more than once in the native pass")
            self.first_outputs[expert_id] = output
            return None

        return capture

    def _patch_moe(self, module, inputs, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Expected SparseMoeBlock to return (output, aux_loss)")
        if output[1] is not None:
            raise RuntimeError("Frozen eval probe expected no auxiliary MoE loss")
        if not any(action is not None for action in self.actions):
            return None

        hidden_states = inputs[0]
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_dim)
        flat_ids = self.route_ids.to(device=hidden_states.device).reshape(-1)
        flat_weights = self.route_weights.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).reshape(-1)
        native_outputs = torch.zeros_like(flat_hidden)
        native_shapes = []
        for expert_id in range(int(module.num_experts)):
            positions = torch.where(flat_ids == expert_id)[0]
            captured = self.first_outputs.get(expert_id)
            if positions.numel() == 0:
                if captured is None or captured.shape[0] != 1:
                    raise RuntimeError("Empty experts must execute one native dummy row")
                native_shapes.append([expert_id, 0, list(captured.shape)])
                continue
            if captured is None or captured.shape != (positions.numel(), hidden_dim):
                raise RuntimeError("Captured routed-expert output shape is inconsistent")
            native_outputs[positions] = captured.to(native_outputs.dtype)
            native_shapes.append([expert_id, int(positions.numel()), list(captured.shape)])

        patched = output[0].clone().reshape(-1, hidden_dim)
        for row, action in enumerate(self.actions):
            if action is None:
                continue
            _validate_candidate(
                action,
                self.route_ids[row].detach().cpu().numpy(),
                int(module.num_routed_experts),
            )
            donors = torch.as_tensor(
                action["donors"], device=hidden_states.device, dtype=torch.long
            )
            receivers = torch.as_tensor(
                action["receivers"], device=hidden_states.device, dtype=torch.long
            )
            experts = torch.as_tensor(
                action["experts"], device=hidden_states.device, dtype=torch.long
            )
            donor_positions = row * seq_len + donors
            receiver_positions = row * seq_len + receivers
            donor_delta = (
                flat_weights[donor_positions, None]
                * native_outputs[donor_positions]
            )
            patched.index_add_(0, donor_positions, -donor_delta.to(patched.dtype))

            self.in_second_pass = True
            try:
                for expert_id in experts.unique(sorted=True).tolist():
                    pair_indices = torch.where(experts == expert_id)[0]
                    positions = receiver_positions[pair_indices]
                    weights = flat_weights[positions, None]
                    updated = flat_hidden[positions] + weights * native_outputs[positions]
                    second_output = module.experts[expert_id](updated)
                    patched.index_add_(
                        0,
                        positions,
                        (weights * second_output).to(patched.dtype),
                    )
                    self.second_pass_shapes.append([
                        int(row),
                        int(expert_id),
                        list(updated.shape),
                        list(second_output.shape),
                    ])
            finally:
                self.in_second_pass = False
        self.native_dispatch_shapes = native_shapes
        return patched.reshape(batch_size, seq_len, hidden_dim), output[1]

    def close(self):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles = []
        if hasattr(self.moe_layer, "_compute_exchange_probe_active"):
            del self.moe_layer._compute_exchange_probe_active


@contextmanager
def _forced_compute_exchange_state(
    moe_layer,
    route_ids,
    route_weights,
    actions,
):
    injector = _ComputeExchangeInjector(
        moe_layer,
        route_ids,
        route_weights,
        actions,
    )
    try:
        with _forced_route_state(moe_layer, route_ids, route_weights):
            yield injector
    finally:
        injector.close()


def _exchange_components(
    moe_layer,
    hidden_states,
    moe_gradient,
    native_experts,
    native_weights,
):
    if hidden_states.ndim != 2 or moe_gradient.shape != hidden_states.shape:
        raise ValueError("Hidden states and MoE gradients must align")
    if native_experts.shape != native_weights.shape:
        raise ValueError("Native route IDs and weights must align")
    if native_experts.shape != (hidden_states.shape[0],):
        raise ValueError("Native routes must align with hidden-state tokens")
    with torch.no_grad():
        first_output = _evaluate_experts(
            moe_layer.experts[: int(moe_layer.num_routed_experts)],
            hidden_states,
            native_experts,
        )
        weighted_first = native_weights[:, None].float() * first_output
        second_input = hidden_states + weighted_first.to(hidden_states.dtype)
        second_output = _evaluate_experts(
            moe_layer.experts[: int(moe_layer.num_routed_experts)],
            second_input,
            native_experts,
        )
        donor_delta = -weighted_first
        receiver_delta = native_weights[:, None].float() * second_output
        gradient = moe_gradient.float()
        donor_change = (gradient * donor_delta).sum(dim=-1)
        receiver_change = (gradient * receiver_delta).sum(dim=-1)
    return {
        "first_output": first_output,
        "second_output": second_output,
        "donor_delta": donor_delta,
        "receiver_delta": receiver_delta,
        "donor_change": donor_change,
        "receiver_change": receiver_change,
        "gradient_sq_norm": gradient.square().sum(dim=-1),
    }


def _rolled_components(donor_change, receiver_change, native_experts, seed):
    donor = donor_change.detach().cpu().numpy().astype(np.float64, copy=True)
    receiver = receiver_change.detach().cpu().numpy().astype(np.float64, copy=True)
    native = native_experts.detach().cpu().numpy()
    generator = np.random.default_rng(int(seed))
    for expert in np.unique(native):
        positions = np.flatnonzero(native == expert)
        donor[positions] = donor[generator.permutation(positions)]
        receiver[positions] = receiver[generator.permutation(positions)]
    return donor, receiver


def _score_candidates(
    candidates,
    components,
    native_experts,
    native_weights,
    router_scores,
    rolled_seed,
):
    donor_change = components["donor_change"]
    receiver_change = components["receiver_change"]
    rolled_donor, rolled_receiver = _rolled_components(
        donor_change,
        receiver_change,
        native_experts,
        rolled_seed,
    )
    top_two = torch.topk(router_scores, k=2, dim=-1).values
    router_margin = top_two[:, 0] - top_two[:, 1]
    records = []
    for candidate in candidates:
        donors = torch.as_tensor(
            candidate["donors"], device=donor_change.device, dtype=torch.long
        )
        receivers = torch.as_tensor(
            candidate["receivers"], device=donor_change.device, dtype=torch.long
        )
        first_order_change = donor_change[donors].sum() + receiver_change[receivers].sum()
        gradient_sq = (
            components["gradient_sq_norm"][donors].sum()
            + components["gradient_sq_norm"][receivers].sum()
        )
        delta_sq = (
            components["donor_delta"][donors].square().sum()
            + components["receiver_delta"][receivers].square().sum()
        )
        denominator = torch.sqrt(gradient_sq * delta_sq).clamp_min(1e-12)
        donor_np = donors.detach().cpu().numpy()
        receiver_np = receivers.detach().cpu().numpy()
        records.append({
            **candidate,
            "first_order_change": float(first_order_change.item()),
            "normalized_first_order_change": float(
                (first_order_change / denominator).item()
            ),
            "rolled_first_order_change": float(
                rolled_donor[donor_np].sum() + rolled_receiver[receiver_np].sum()
            ),
            "router_margin_priority": float(
                (
                    router_margin[receivers].sum()
                    - router_margin[donors].sum()
                ).item()
            ),
            "donor_delta_l2": float(
                torch.sqrt(components["donor_delta"][donors].square().sum()).item()
            ),
            "receiver_delta_l2": float(
                torch.sqrt(components["receiver_delta"][receivers].square().sum()).item()
            ),
            "mean_native_weight": float(torch.cat((
                native_weights[donors],
                native_weights[receivers],
            )).float().mean().item()),
        })
    return records


def _selector_record(name, index, records, native_mse, epsilon):
    if index is None:
        gain = 0.0
        transferred = 0
    else:
        gain = -float(records[index]["exact_mse_change"]) / float(native_mse)
        transferred = int(records[index]["transferred_passes"])
    return {
        "name": name,
        "selected_candidate_index": index,
        "selected_non_native": index is not None,
        "selected_gain": float(gain),
        "selected_per_transferred_pass_gain": (
            float(gain / transferred) if transferred else 0.0
        ),
        "selected_positive": bool(gain > epsilon),
        "selected_harm": bool(gain < -epsilon),
        "transferred_passes": transferred,
    }


def _pair_concordance(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("Concordance inputs must be aligned nontrivial vectors")
    concordant = 0.0
    comparisons = 0
    for first in range(left.size):
        for second in range(first + 1, left.size):
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            comparisons += 1
            if left_delta == 0 or right_delta == 0:
                concordant += 0.5
            elif np.signbit(left_delta) == np.signbit(right_delta):
                concordant += 1.0
    return float(concordant / comparisons)


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize_selectors(records, native_mse, random_seed, epsilon=NUMERICAL_EPSILON):
    if len(records) != CANDIDATE_COUNT:
        raise ValueError("The locked compute-exchange bank must have 64 records")
    if native_mse <= 0 or epsilon <= 0:
        raise ValueError("native_mse and epsilon must be positive")
    predicted_gain = -np.asarray([
        record["first_order_change"] for record in records
    ], dtype=np.float64) / float(native_mse)
    exact_gain = -np.asarray([
        record["exact_mse_change"] for record in records
    ], dtype=np.float64) / float(native_mse)
    first_order_index = int(predicted_gain.argmax())
    random_index = int(np.random.default_rng(int(random_seed)).integers(len(records)))
    margin_index = int(np.argmax([
        record["router_margin_priority"] for record in records
    ]))
    rolled_index = int(np.argmin([
        record["rolled_first_order_change"] for record in records
    ]))
    oracle_index = int(exact_gain.argmax())
    if exact_gain[oracle_index] <= 0:
        oracle_index = None
    selectors = {
        "first_order": _selector_record(
            "first_order", first_order_index, records, native_mse, epsilon
        ),
        "random": _selector_record(
            "random", random_index, records, native_mse, epsilon
        ),
        "router_margin": _selector_record(
            "router_margin", margin_index, records, native_mse, epsilon
        ),
        "rolled_utility": _selector_record(
            "rolled_utility", rolled_index, records, native_mse, epsilon
        ),
        "exact_oracle": _selector_record(
            "exact_oracle", oracle_index, records, native_mse, epsilon
        ),
    }
    return {
        "selectors": selectors,
        "pair_concordance": _pair_concordance(predicted_gain, exact_gain),
        "spearman": _correlation(
            _rankdata(predicted_gain),
            _rankdata(exact_gain),
        ),
        "sign_agreement": float(np.mean(
            (predicted_gain > 0) == (exact_gain > epsilon)
        )),
        "oracle_positive": bool(oracle_index is not None),
        "oracle_gain": float(max(0.0, exact_gain.max())),
        "first_order_oracle_ratio": (
            float(selectors["first_order"]["selected_gain"] / exact_gain.max())
            if first_order_index is not None and exact_gain.max() > 0 else 0.0
        ),
        "epsilon": float(epsilon),
    }


def _expected_second_pass_shapes(candidate, hidden_dim):
    expected = []
    quotas = candidate["quota_by_expert"]
    for expert_id, count in enumerate(quotas):
        if count:
            expected.append([1, int(expert_id), [int(count), int(hidden_dim)], [int(count), int(hidden_dim)]])
    return expected


def _exact_candidate_changes(
    model,
    moe_layer,
    noised_latent,
    timestep,
    label,
    target,
    native_route_ids,
    native_route_weights,
    native_prediction,
    native_loss,
    candidates,
):
    target_channels = target.shape[1]
    route_ids = native_route_ids.unsqueeze(0).expand(2, -1).clone()
    route_weights = native_route_weights.unsqueeze(0).expand(2, -1).clone()
    if not torch.equal(route_ids[0], route_ids[1]):
        raise RuntimeError("Paired exact rows do not share route IDs")
    if not torch.equal(route_weights[0], route_weights[1]):
        raise RuntimeError("Paired exact rows do not share route weights")

    with torch.inference_mode(), _forced_compute_exchange_state(
        moe_layer,
        route_ids,
        route_weights,
        (None, None),
    ):
        paired_native_output = model(
            noised_latent.repeat(2, 1, 1, 1, 1),
            timestep.repeat(2),
            context=label.repeat(2),
        )
    paired_native_prediction = _extract_prediction(
        paired_native_output,
        target_channels,
    )
    paired_native_losses = _per_sample_mse(
        paired_native_prediction,
        target.repeat(2, 1, 1, 1),
    )

    records = []
    max_native_mse_drift = 0.0
    max_native_output_drift = 0.0
    logical_count_mismatches = 0
    action_contract_mismatches = 0
    native_dispatch_shapes = None
    for candidate in candidates:
        native_counts, action_counts = _logical_pass_counts(
            native_route_ids.detach().cpu().numpy(),
            candidate["donors"],
            candidate["receivers"],
            int(moe_layer.num_routed_experts),
        )
        if not np.array_equal(native_counts, action_counts):
            logical_count_mismatches += 1
        with torch.inference_mode(), _forced_compute_exchange_state(
            moe_layer,
            route_ids,
            route_weights,
            (None, candidate),
        ) as injector:
            output = model(
                noised_latent.repeat(2, 1, 1, 1, 1),
                timestep.repeat(2),
                context=label.repeat(2),
            )
        prediction = _extract_prediction(output, target_channels)
        losses = _per_sample_mse(
            prediction,
            target.repeat(2, 1, 1, 1),
        )
        max_native_mse_drift = max(
            max_native_mse_drift,
            float(abs(losses[0].item() - paired_native_losses[0].item())),
        )
        max_native_output_drift = max(
            max_native_output_drift,
            float((
                prediction[0] - paired_native_prediction[0]
            ).abs().max().item()),
        )
        expected_shapes = _expected_second_pass_shapes(
            candidate,
            hidden_dim=int(moe_layer.hidden_size),
        )
        if injector.second_pass_shapes != expected_shapes:
            action_contract_mismatches += 1
        if native_dispatch_shapes is None:
            native_dispatch_shapes = injector.native_dispatch_shapes
        elif native_dispatch_shapes != injector.native_dispatch_shapes:
            action_contract_mismatches += 1
        records.append({
            **candidate,
            "exact_mse_change": float((losses[1] - losses[0]).item()),
            "exact_mse_change_relative": float(
                (losses[1] - losses[0]).item() / native_loss.item()
            ),
            "max_abs_output_change": float(
                (prediction[1] - prediction[0]).abs().max().item()
            ),
            "runtime_second_pass_shapes": injector.second_pass_shapes,
        })

    with torch.inference_mode(), _forced_route_state(
        moe_layer,
        route_ids,
        route_weights,
    ):
        plain_noop_output = model(
            noised_latent.repeat(2, 1, 1, 1, 1),
            timestep.repeat(2),
            context=label.repeat(2),
        )
    plain_noop_prediction = _extract_prediction(plain_noop_output, target_channels)
    plain_noop_losses = _per_sample_mse(
        plain_noop_prediction,
        target.repeat(2, 1, 1, 1),
    )
    controls = {
        "max_abs_paired_native_mse_drift": max_native_mse_drift,
        "max_abs_paired_native_output_drift": max_native_output_drift,
        "max_abs_single_vs_paired_native_mse_drift": float(
            abs(paired_native_losses[0].item() - native_loss.item())
        ),
        "max_abs_single_vs_paired_native_output_drift": float((
            paired_native_prediction[0] - native_prediction[0]
        ).abs().max().item()),
        "max_abs_noop_mse_change": float(
            abs(paired_native_losses[1].item() - paired_native_losses[0].item())
        ),
        "max_abs_noop_output_change": float((
            paired_native_prediction[1] - paired_native_prediction[0]
        ).abs().max().item()),
        "max_abs_hook_mse_change": float(
            (paired_native_losses - plain_noop_losses).abs().max().item()
        ),
        "max_abs_hook_output_change": float((
            paired_native_prediction - plain_noop_prediction
        ).abs().max().item()),
        "logical_count_mismatches": int(logical_count_mismatches),
        "action_contract_mismatches": int(action_contract_mismatches),
        "route_id_mismatches": int(not torch.equal(route_ids[0], route_ids[1])),
        "route_weight_mismatches": int(not torch.equal(
            route_weights[0], route_weights[1]
        )),
        "native_dispatch_shapes": native_dispatch_shapes,
    }
    return records, controls


def _probe_cell(
    model,
    runtime_cfg,
    moe_layer,
    capture,
    clean_latent,
    noise,
    label,
    sigma,
    block_index,
    candidate_seed,
    safety_only,
):
    sigma_tensor = torch.tensor(
        float(sigma),
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    timestep = torch.full(
        (1,),
        float(sigma) * runtime_cfg.num_train_timesteps,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    noised_latent = (1.0 - sigma_tensor) * clean_latent + sigma_tensor * noise
    target = (noise - clean_latent).squeeze(2)

    capture.start()
    try:
        model_output = model(noised_latent, timestep, context=label)
        native_prediction = _extract_prediction(model_output, target.shape[1])
        native_loss = _per_sample_mse(native_prediction, target).mean()
        if capture.moe_output is None:
            raise RuntimeError("The compute-exchange probe did not capture the MoE output")
        moe_gradient, = torch.autograd.grad(native_loss, capture.moe_output)
    finally:
        capture.stop()

    hidden_states = capture.hidden_states
    labels = capture.labels
    if hidden_states is None or labels is None:
        raise RuntimeError("The compute-exchange probe did not capture router inputs")
    with torch.no_grad():
        native_weights, native_indices, auxiliary_loss = _compute_router(
            moe_layer,
            hidden_states,
            labels,
            timestep,
        )
        router_scores = _all_router_weights(moe_layer, hidden_states, timestep)
    if auxiliary_loss is not None:
        raise RuntimeError("Frozen eval router unexpectedly returned an auxiliary loss")
    native_route_ids = native_indices[0, :, 0]
    native_route_weights = native_weights[0, :, 0]
    if not torch.equal(router_scores[0].argmax(dim=-1), native_route_ids):
        raise RuntimeError("Native routes disagree with all-router scores")
    if native_route_ids.max() >= int(moe_layer.num_routed_experts):
        raise RuntimeError("Conditional probe selected the unconditional expert")

    candidates = build_same_expert_exchange_candidates(
        native_route_ids.detach().cpu().numpy(),
        int(moe_layer.num_routed_experts),
        candidate_seed,
    )
    components = _exchange_components(
        moe_layer=moe_layer,
        hidden_states=hidden_states[0],
        moe_gradient=moe_gradient.detach()[0],
        native_experts=native_route_ids,
        native_weights=native_route_weights,
    )
    scored = _score_candidates(
        candidates=candidates,
        components=components,
        native_experts=native_route_ids,
        native_weights=native_route_weights,
        router_scores=router_scores[0],
        rolled_seed=_stable_seed(candidate_seed, "rolled"),
    )
    evaluated = scored[:1] if safety_only else scored
    exact_records, exact_controls = _exact_candidate_changes(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        native_route_ids=native_route_ids,
        native_route_weights=native_route_weights,
        native_prediction=native_prediction.detach(),
        native_loss=native_loss.detach(),
        candidates=evaluated,
    )
    forced_control = _forced_native_control(
        model=model,
        moe_layer=moe_layer,
        noised_latent=noised_latent,
        timestep=timestep,
        label=label,
        target=target,
        native_route_ids=native_route_ids,
        native_route_weights=native_route_weights,
        unforced_prediction=native_prediction.detach(),
        unforced_loss=native_loss.item(),
    )
    logical_mismatches = sum(
        candidate["native_pass_vector"] != candidate["candidate_pass_vector"]
        for candidate in candidates
    )
    numerical_controls = {
        **exact_controls,
        **forced_control,
        "logical_count_mismatches": int(
            exact_controls["logical_count_mismatches"] + logical_mismatches
        ),
    }
    cell = {
        "block_index": int(block_index),
        "sigma": float(sigma),
        "timestep": float(timestep.item()),
        "native_mse": float(native_loss.item()),
        "candidate_seed": int(candidate_seed),
        "candidate_count": CANDIDATE_COUNT,
        "exchange_quota": EXCHANGE_QUOTA,
        "activation_rms": float(hidden_states[0].float().square().mean().sqrt().item()),
        "numerical_controls": numerical_controls,
    }
    if safety_only:
        cell["efficacy_statistics_withheld"] = True
        return cell

    exact_by_id = {record["id"]: record for record in exact_records}
    if len(exact_by_id) != CANDIDATE_COUNT:
        raise RuntimeError("Every locked exchange candidate must be evaluated exactly once")
    final_records = [exact_by_id[candidate["id"]] for candidate in scored]
    cell.update({
        "records": final_records,
        "summary": summarize_selectors(
            final_records,
            native_mse=native_loss.item(),
            random_seed=_stable_seed(candidate_seed, "matched-random"),
        ),
    })
    return cell


def run_compute_exchange_probe_case(
    model,
    runtime_cfg,
    latent_path,
    label,
    seed,
    block_indices,
    sigmas,
    safety_only=False,
    latent_key="latent",
):
    latent_path = Path(latent_path).resolve()
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent does not exist: {latent_path}")
    if not 0 <= label < runtime_cfg.num_classes:
        raise ValueError("ImageNet label lies outside the configured class range")
    sigmas = tuple(float(sigma) for sigma in sigmas)
    if (
        not sigmas
        or len(sigmas) != len(set(sigmas))
        or any(not 0 < sigma < 1 for sigma in sigmas)
    ):
        raise ValueError("Sigmas must be unique and strictly between zero and one")
    block_indices = tuple(int(index) for index in block_indices)
    _validate_compute_exchange_contract(model, block_indices)

    device = next(model.parameters()).device
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    clean_latent = _load_latent(latent_path, latent_key, seed, device)
    torch.manual_seed(seed + 1)
    noise = torch.randn_like(clean_latent)
    label_tensor = torch.tensor([label], device=device, dtype=torch.long)

    cells = []
    probe_start = time.perf_counter()
    for block_index in block_indices:
        moe_layer = model.blocks[block_index].mlp
        capture = RoutingProbeCapture(moe_layer)
        try:
            for sigma in sigmas:
                cells.append(_probe_cell(
                    model=model,
                    runtime_cfg=runtime_cfg,
                    moe_layer=moe_layer,
                    capture=capture,
                    clean_latent=clean_latent,
                    noise=noise,
                    label=label_tensor,
                    sigma=sigma,
                    block_index=block_index,
                    candidate_seed=_stable_seed(
                        seed,
                        "compute-exchange",
                        block_index,
                        f"{sigma:.17g}",
                    ),
                    safety_only=bool(safety_only),
                ))
        finally:
            capture.close()
    return {
        "compute_exchange_probe_version": PROBE_VERSION,
        "diagnostic_scope": (
            "frozen-checkpoint downstream-utility gate for exact within-expert "
            "FFN-pass exchange; not a training, sampling, FID, or novelty claim"
        ),
        "label": int(label),
        "latent": str(latent_path),
        "latent_key": latent_key,
        "seed": int(seed),
        "block_indices": list(block_indices),
        "sigmas": list(sigmas),
        "safety_only": bool(safety_only),
        "probe_seconds": float(time.perf_counter() - probe_start),
        "cells": cells,
    }


def run_compute_exchange_probe(
    checkpoint_path,
    weights_checkpoint_path,
    latent_path,
    label,
    seed,
    block_indices,
    sigmas,
    safety_only=False,
    latent_key="latent",
    device="cpu",
    num_threads=8,
):
    checkpoint_path = Path(checkpoint_path).resolve()
    weights_checkpoint_path = Path(weights_checkpoint_path).resolve()
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    checkpoint_step = parse_checkpoint_step(checkpoint_path)
    runtime_cfg = load_runtime_cfg(config_path)
    thread_config = _configure_torch_threads(num_threads)
    model, state_name, weights_step, load_seconds = _load_checkpoint_model(
        runtime_cfg,
        weights_checkpoint_path,
        torch.device(device),
    )
    if weights_step != checkpoint_step:
        raise ValueError("Canonical and loaded checkpoint steps differ")
    try:
        result = run_compute_exchange_probe_case(
            model=model,
            runtime_cfg=runtime_cfg,
            latent_path=latent_path,
            label=label,
            seed=seed,
            block_indices=block_indices,
            sigmas=sigmas,
            safety_only=safety_only,
            latent_key=latent_key,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result.update({
        "checkpoint": str(checkpoint_path),
        "weights_checkpoint": str(weights_checkpoint_path),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_state": state_name,
        "config": str(config_path),
        "device": str(device),
        "num_threads": int(num_threads),
        "thread_config": thread_config,
        "model_load_seconds": float(load_seconds),
    })
    return result
