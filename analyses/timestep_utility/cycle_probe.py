"""Exact fixed-count routing-cycle fidelity probe for ProMoE."""

from __future__ import annotations

import gc
import hashlib
import time
from bisect import bisect_right
from functools import lru_cache
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import torch

from analyses.denoising_regret.probe import (
    RoutingProbeCapture,
    _all_router_weights,
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


PROBE_VERSION = 4
ARM_NAMES = (
    "four_cycle",
    "six_cycle",
    "mixed_cycle",
    "single_token",
    "random_joint",
)
COUNT_PRESERVING_ARMS = (
    "four_cycle",
    "six_cycle",
    "mixed_cycle",
    "random_joint",
)
ARM_CANDIDATES = 64
MIXED_FOUR_CANDIDATES = 32
MIXED_SIX_CANDIDATES = 32
RANDOM_JOINT_TOKENS = 8
AUDITED_SIX_CANDIDATES = 16
MAX_CANDIDATE_ATTEMPTS = 1024


def _stable_seed(seed, *parts):
    payload = "|".join((str(int(seed)), *(str(part) for part in parts)))
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) % (2 ** 63)


def _candidate_signature(tokens, destinations):
    return tuple(sorted(zip(
        (int(token) for token in tokens),
        (int(expert) for expert in destinations),
    )))


def _count_vector(experts, num_experts):
    return np.bincount(
        np.asarray(experts, dtype=np.int64),
        minlength=int(num_experts),
    )


def _random_below(generator, upper):
    upper = int(upper)
    if upper <= 0:
        raise ValueError("Random upper bound must be positive")
    if upper == 1:
        return 0
    bit_count = (upper - 1).bit_length()
    word_count = (bit_count + 63) // 64
    mask = (1 << bit_count) - 1
    while True:
        value = 0
        for word_index in range(word_count):
            value |= (
                int(generator.bit_generator.random_raw()) << (64 * word_index)
            )
        value &= mask
        if value < upper:
            return value


def _sample_unique_ranks(generator, population_size, sample_size):
    population_size = int(population_size)
    sample_size = int(sample_size)
    if sample_size < 0 or sample_size > population_size:
        raise ValueError("Cannot sample the requested number of unique ranks")
    selected = set()
    ranks = []
    for upper in range(population_size - sample_size, population_size):
        rank = _random_below(generator, upper + 1)
        if rank in selected:
            rank = upper
        selected.add(rank)
        ranks.append(rank)
    generator.shuffle(ranks)
    return ranks


@lru_cache(maxsize=None)
def _count_deranged_pattern(count_pattern):
    sources = tuple(
        expert
        for expert, count in enumerate(count_pattern)
        for _ in range(count)
    )

    @lru_cache(maxsize=None)
    def count_from(position, remaining):
        if position == len(sources):
            return int(not any(remaining))
        total = 0
        for destination, count in enumerate(remaining):
            if count == 0 or destination == sources[position]:
                continue
            next_remaining = list(remaining)
            next_remaining[destination] -= 1
            total += count_from(position + 1, tuple(next_remaining))
        return total

    return count_from(0, tuple(count_pattern))


def _count_deranged_assignments(counts):
    count_pattern = tuple(sorted(
        (int(count) for count in counts if count),
        reverse=True,
    ))
    return _count_deranged_pattern(count_pattern)


def _unrank_combination(pool, selected_count, rank):
    pool = np.asarray(pool, dtype=np.int64)
    selected_count = int(selected_count)
    rank = int(rank)
    total = comb(pool.size, selected_count)
    if rank < 0 or rank >= total:
        raise ValueError("Combination rank lies outside its population")
    selected = []
    next_index = 0
    for remaining_slots in range(selected_count, 0, -1):
        for index in range(next_index, pool.size):
            block_size = comb(pool.size - index - 1, remaining_slots - 1)
            if rank < block_size:
                selected.append(int(pool[index]))
                next_index = index + 1
                break
            rank -= block_size
    if len(selected) != selected_count or rank != 0:
        raise RuntimeError("Combination unranking failed")
    return selected


def _unrank_token_subset(token_pools, counts, rank):
    radices = [
        comb(pool.size, int(count))
        for pool, count in zip(token_pools, counts)
    ]
    suffix_products = [1] * (len(radices) + 1)
    for index in range(len(radices) - 1, -1, -1):
        suffix_products[index] = suffix_products[index + 1] * radices[index]
    rank = int(rank)
    if rank < 0 or rank >= suffix_products[0]:
        raise ValueError("Token-subset rank lies outside its population")

    tokens = []
    for index, (pool, count) in enumerate(zip(token_pools, counts)):
        tail_size = suffix_products[index + 1]
        combination_rank, rank = divmod(rank, tail_size)
        tokens.extend(_unrank_combination(pool, count, combination_rank))
    if rank != 0:
        raise RuntimeError("Token-subset unranking did not consume its rank")
    return np.asarray(tokens, dtype=np.int64)


def _unrank_multiset_derangement(sources, counts, rank):
    sources = tuple(int(source) for source in sources)
    counts = tuple(int(count) for count in counts)

    @lru_cache(maxsize=None)
    def count_from(position, remaining):
        if position == len(sources):
            return int(not any(remaining))
        total = 0
        for destination, count in enumerate(remaining):
            if count == 0 or destination == sources[position]:
                continue
            next_remaining = list(remaining)
            next_remaining[destination] -= 1
            total += count_from(position + 1, tuple(next_remaining))
        return total

    rank = int(rank)
    total = count_from(0, counts)
    if rank < 0 or rank >= total:
        raise ValueError("Derangement rank lies outside its population")

    remaining = counts
    destinations = []
    for position, source in enumerate(sources):
        for destination, count in enumerate(remaining):
            if count == 0 or destination == source:
                continue
            next_remaining = list(remaining)
            next_remaining[destination] -= 1
            next_remaining = tuple(next_remaining)
            block_size = count_from(position + 1, next_remaining)
            if rank < block_size:
                destinations.append(destination)
                remaining = next_remaining
                break
            rank -= block_size
        else:
            raise RuntimeError("Derangement unranking reached an empty branch")
    if rank != 0 or any(remaining):
        raise RuntimeError("Derangement unranking did not consume its rank")
    return np.asarray(destinations, dtype=np.int64)


def _random_joint_signature_space(native_experts, num_experts):
    capacities = _count_vector(native_experts, num_experts)
    limits = np.minimum(capacities, RANDOM_JOINT_TOKENS // 2).astype(np.int64)
    if int(limits.sum()) < RANDOM_JOINT_TOKENS:
        raise RuntimeError(
            "Native routes do not admit an eight-token fixed-count derangement"
        )
    token_pools = tuple(
        np.flatnonzero(native_experts == expert).astype(np.int64)
        for expert in range(num_experts)
    )
    suffix_limits = np.zeros(num_experts + 1, dtype=np.int64)
    for index in range(num_experts - 1, -1, -1):
        suffix_limits[index] = suffix_limits[index + 1] + limits[index]

    blocks = []
    cumulative_ends = []
    counts = [0] * num_experts
    total_signatures = 0

    def visit(expert_index, remaining_tokens, subset_count):
        nonlocal total_signatures
        if expert_index == num_experts:
            if remaining_tokens != 0:
                return
            count_tuple = tuple(counts)
            derangement_count = _count_deranged_assignments(count_tuple)
            if derangement_count == 0:
                return
            block_size = int(subset_count) * derangement_count
            total_signatures += block_size
            blocks.append((count_tuple, int(subset_count), derangement_count))
            cumulative_ends.append(total_signatures)
            return

        minimum = max(
            0,
            int(remaining_tokens - suffix_limits[expert_index + 1]),
        )
        maximum = min(int(limits[expert_index]), int(remaining_tokens))
        for selected_count in range(minimum, maximum + 1):
            counts[expert_index] = selected_count
            visit(
                expert_index + 1,
                remaining_tokens - selected_count,
                subset_count * comb(
                    int(capacities[expert_index]),
                    selected_count,
                ),
            )
        counts[expert_index] = 0

    visit(0, RANDOM_JOINT_TOKENS, 1)
    return token_pools, blocks, cumulative_ends, total_signatures


def _validate_candidate(candidate, native_experts, num_experts):
    tokens = np.asarray(candidate["tokens"], dtype=np.int64)
    sources = np.asarray(candidate["source_experts"], dtype=np.int64)
    destinations = np.asarray(candidate["destination_experts"], dtype=np.int64)
    if tokens.ndim != 1 or tokens.size == 0:
        raise ValueError("Candidate tokens must be a nonempty vector")
    if sources.shape != tokens.shape or destinations.shape != tokens.shape:
        raise ValueError("Candidate tokens, sources, and destinations must align")
    if np.unique(tokens).size != tokens.size:
        raise ValueError("A routing candidate cannot reuse a token")
    if tokens.min() < 0 or tokens.max() >= native_experts.size:
        raise ValueError("Candidate token lies outside the route sequence")
    if sources.min() < 0 or destinations.min() < 0:
        raise ValueError("Candidate experts must be nonnegative")
    if sources.max() >= num_experts or destinations.max() >= num_experts:
        raise ValueError("Candidate expert lies outside the routed expert set")
    if not np.array_equal(native_experts[tokens], sources):
        raise ValueError("Candidate source experts disagree with native routing")
    if np.any(sources == destinations):
        raise ValueError("Every candidate slot must change expert identity")

    counts_match = bool(np.array_equal(
        _count_vector(sources, num_experts),
        _count_vector(destinations, num_experts),
    ))
    expected_count_preserving = bool(candidate["count_preserving"])
    if counts_match != expected_count_preserving:
        raise ValueError(
            "Candidate count-preserving declaration disagrees with its experts"
        )
    candidate["source_count_vector"] = _count_vector(
        sources, num_experts
    ).tolist()
    candidate["destination_count_vector"] = _count_vector(
        destinations, num_experts
    ).tolist()
    candidate["changed_tokens"] = int(tokens.size)
    return candidate


def _sample_distinct_source_tokens(native_experts, count, generator):
    num_tokens = native_experts.size
    for _ in range(MAX_CANDIDATE_ATTEMPTS):
        tokens = generator.choice(num_tokens, size=count, replace=False)
        sources = native_experts[tokens]
        if np.unique(sources).size == count:
            return tokens.astype(np.int64), sources.astype(np.int64)
    raise RuntimeError(
        f"Could not sample {count} tokens with distinct native experts"
    )


def _build_short_cycle_arm(
    native_experts,
    num_experts,
    candidate_count,
    cycle_tokens,
    seed,
    arm,
):
    generator = np.random.default_rng(seed)
    candidates = []
    seen = set()
    for candidate_index in range(candidate_count):
        for _ in range(MAX_CANDIDATE_ATTEMPTS):
            tokens, sources = _sample_distinct_source_tokens(
                native_experts,
                cycle_tokens,
                generator,
            )
            if cycle_tokens == 2:
                destinations = sources[::-1].copy()
                kind = "four_cycle"
            elif cycle_tokens == 3:
                direction = 1 if int(generator.integers(2)) == 0 else -1
                destinations = np.roll(sources, direction)
                kind = "six_cycle"
            else:
                raise ValueError("Short routing cycles must use two or three tokens")
            signature = _candidate_signature(tokens, destinations)
            if signature in seen:
                continue
            seen.add(signature)
            candidate = {
                "id": f"{arm}:{candidate_index:03d}",
                "arm": arm,
                "kind": kind,
                "tokens": tokens.tolist(),
                "source_experts": sources.tolist(),
                "destination_experts": destinations.tolist(),
                "count_preserving": True,
            }
            candidates.append(_validate_candidate(
                candidate,
                native_experts,
                num_experts,
            ))
            break
        else:
            raise RuntimeError(
                f"Could not construct unique candidate {candidate_index} for {arm}"
            )
    return candidates


def _build_mixed_arm(native_experts, num_experts, seed):
    four = _build_short_cycle_arm(
        native_experts=native_experts,
        num_experts=num_experts,
        candidate_count=MIXED_FOUR_CANDIDATES,
        cycle_tokens=2,
        seed=_stable_seed(seed, "mixed", "four"),
        arm="mixed_cycle",
    )
    six = _build_short_cycle_arm(
        native_experts=native_experts,
        num_experts=num_experts,
        candidate_count=MIXED_SIX_CANDIDATES,
        cycle_tokens=3,
        seed=_stable_seed(seed, "mixed", "six"),
        arm="mixed_cycle",
    )
    candidates = four + six
    for index, candidate in enumerate(candidates):
        candidate["id"] = f"mixed_cycle:{index:03d}"
    return candidates


def _build_single_token_arm(native_experts, num_experts, seed):
    generator = np.random.default_rng(_stable_seed(seed, "single", "edge"))
    candidates = []
    seen = set()
    for candidate_index in range(ARM_CANDIDATES):
        for attempt in range(MAX_CANDIDATE_ATTEMPTS):
            component_seed = _stable_seed(
                seed,
                "single",
                "component",
                candidate_index,
                attempt,
            )
            cycle_tokens = 2 if candidate_index < 32 else 3
            component = _build_short_cycle_arm(
                native_experts,
                num_experts,
                candidate_count=1,
                cycle_tokens=cycle_tokens,
                seed=component_seed,
                arm="single_component",
            )[0]
            slots = generator.permutation(component["changed_tokens"])
            selected = None
            for slot in slots:
                token = int(component["tokens"][slot])
                source = int(component["source_experts"][slot])
                destination = int(component["destination_experts"][slot])
                signature = _candidate_signature([token], [destination])
                if signature not in seen:
                    selected = (token, source, destination, signature)
                    break
            if selected is None:
                continue
            token, source, destination, signature = selected
            seen.add(signature)
            candidate = {
                "id": f"single_token:{candidate_index:03d}",
                "arm": "single_token",
                "kind": "single_token",
                "tokens": [token],
                "source_experts": [source],
                "destination_experts": [destination],
                "count_preserving": False,
                "component_kind": component["kind"],
            }
            candidates.append(_validate_candidate(
                candidate,
                native_experts,
                num_experts,
            ))
            break
        else:
            raise RuntimeError(
                f"Could not construct unique single-token candidate {candidate_index}"
            )
    if len(candidates) != ARM_CANDIDATES:
        raise RuntimeError("Single-token arm does not contain 64 candidates")
    return candidates


def _build_random_joint_arm(native_experts, num_experts, seed):
    generator = np.random.default_rng(seed)
    token_pools, blocks, cumulative_ends, total_signatures = (
        _random_joint_signature_space(native_experts, num_experts)
    )
    if total_signatures < ARM_CANDIDATES:
        raise RuntimeError(
            "Random-joint signature space contains "
            f"{total_signatures} candidates; {ARM_CANDIDATES} are required"
        )
    signature_ranks = _sample_unique_ranks(
        generator,
        total_signatures,
        ARM_CANDIDATES,
    )
    candidates = []
    seen = set()
    for candidate_index, signature_rank in enumerate(signature_ranks):
        block_index = bisect_right(cumulative_ends, signature_rank)
        previous_end = 0 if block_index == 0 else cumulative_ends[block_index - 1]
        counts, subset_count, derangement_count = blocks[block_index]
        local_rank = signature_rank - previous_end
        subset_rank, destination_rank = divmod(
            local_rank,
            derangement_count,
        )
        if subset_rank >= subset_count:
            raise RuntimeError("Random-joint signature rank selected the wrong block")
        tokens = _unrank_token_subset(token_pools, counts, subset_rank)
        sources = native_experts[tokens].astype(np.int64)
        destinations = _unrank_multiset_derangement(
            sources,
            counts,
            destination_rank,
        )
        signature = _candidate_signature(tokens, destinations)
        if signature in seen:
            raise RuntimeError("Random-joint rank unranking produced a collision")
        seen.add(signature)
        candidate = {
            "id": f"random_joint:{candidate_index:03d}",
            "arm": "random_joint",
            "kind": "random_joint",
            "tokens": tokens.tolist(),
            "source_experts": sources.tolist(),
            "destination_experts": destinations.tolist(),
            "count_preserving": True,
            "joint_signature_rank": int(signature_rank),
        }
        candidates.append(_validate_candidate(
            candidate,
            native_experts,
            num_experts,
        ))
    return candidates


def _build_audit_pairs(six_candidates, native_experts, num_experts):
    audits = []
    for six_index, six_candidate in enumerate(
        six_candidates[:AUDITED_SIX_CANDIDATES]
    ):
        tokens = six_candidate["tokens"]
        sources = six_candidate["source_experts"]
        for pair_index, (left, right) in enumerate(combinations(range(3), 2)):
            pair_tokens = [tokens[left], tokens[right]]
            pair_sources = [sources[left], sources[right]]
            candidate = {
                "id": f"audit:{six_index:03d}:{pair_index}",
                "arm": "six_cycle_audit",
                "kind": "direct_four_cycle",
                "tokens": pair_tokens,
                "source_experts": pair_sources,
                "destination_experts": pair_sources[::-1],
                "count_preserving": True,
                "parent_six_id": six_candidate["id"],
            }
            audits.append(_validate_candidate(
                candidate,
                native_experts,
                num_experts,
            ))
    return audits


def build_candidate_banks(native_experts, num_experts, seed):
    native_experts = np.asarray(native_experts, dtype=np.int64)
    if native_experts.ndim != 1 or native_experts.size < RANDOM_JOINT_TOKENS:
        raise ValueError("Native routes must contain at least eight tokens")
    if native_experts.min() < 0 or native_experts.max() >= num_experts:
        raise ValueError("Native routes name an invalid routed expert")
    if np.unique(native_experts).size < 3:
        raise RuntimeError("Cycle gate requires at least three active experts")

    banks = {
        "four_cycle": _build_short_cycle_arm(
            native_experts,
            num_experts,
            ARM_CANDIDATES,
            2,
            _stable_seed(seed, "four_cycle"),
            "four_cycle",
        ),
        "six_cycle": _build_short_cycle_arm(
            native_experts,
            num_experts,
            ARM_CANDIDATES,
            3,
            _stable_seed(seed, "six_cycle"),
            "six_cycle",
        ),
        "mixed_cycle": _build_mixed_arm(
            native_experts,
            num_experts,
            _stable_seed(seed, "mixed_cycle"),
        ),
        "single_token": _build_single_token_arm(
            native_experts,
            num_experts,
            _stable_seed(seed, "single_token"),
        ),
        "random_joint": _build_random_joint_arm(
            native_experts,
            num_experts,
            _stable_seed(seed, "random_joint"),
        ),
    }
    if set(banks) != set(ARM_NAMES):
        raise RuntimeError("Cycle candidate banks do not match the locked arms")
    if any(len(candidates) != ARM_CANDIDATES for candidates in banks.values()):
        raise RuntimeError("Every cycle gate arm must contain 64 candidates")
    audits = _build_audit_pairs(
        banks["six_cycle"],
        native_experts,
        num_experts,
    )
    if len(audits) != AUDITED_SIX_CANDIDATES * 3:
        raise RuntimeError("Six-cycle audit must contain three pair swaps per cycle")
    return banks, audits


def _expert_output_grid(moe_layer, hidden_states):
    if hidden_states.ndim != 2:
        raise ValueError("Hidden states must be shaped [tokens, channels]")
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = int(moe_layer.num_routed_experts)
    expanded_hidden = hidden_states[:, None, :].expand(
        num_tokens,
        num_experts,
        hidden_dim,
    ).reshape(num_tokens * num_experts, hidden_dim)
    expert_ids = torch.arange(
        num_experts,
        device=hidden_states.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(num_tokens, -1).reshape(-1)
    outputs = _evaluate_experts(
        moe_layer.experts[:num_experts],
        expanded_hidden,
        expert_ids,
    )
    return outputs.reshape(num_tokens, num_experts, hidden_dim)


def _edge_first_order_grid(
    moe_layer,
    hidden_states,
    moe_gradient,
    native_experts,
    native_weights,
):
    if hidden_states.shape != moe_gradient.shape or hidden_states.ndim != 2:
        raise ValueError("Hidden states and MoE gradients must align")
    if native_experts.shape != native_weights.shape:
        raise ValueError("Native expert IDs and weights must align")
    num_tokens = hidden_states.shape[0]
    if native_experts.shape != (num_tokens,):
        raise ValueError("Native routes must align with hidden-state tokens")

    with torch.no_grad():
        output_grid = _expert_output_grid(moe_layer, hidden_states)
        rows = torch.arange(num_tokens, device=hidden_states.device)
        native_outputs = output_grid[rows, native_experts]
        output_delta = native_weights[:, None, None].float() * (
            output_grid - native_outputs[:, None, :]
        )
        gradient = moe_gradient.float()
        first_order = torch.einsum("td,ted->te", gradient, output_delta)
        gradient_sq_norm = gradient.square().sum(dim=-1)
        delta_sq_norm = output_delta.square().sum(dim=-1)
    return first_order, gradient_sq_norm, delta_sq_norm


def _score_candidates(
    candidates,
    first_order_grid,
    gradient_sq_norm,
    delta_sq_norm,
    router_scores,
):
    records = []
    for candidate in candidates:
        tokens = torch.as_tensor(
            candidate["tokens"],
            device=first_order_grid.device,
            dtype=torch.long,
        )
        sources = torch.as_tensor(
            candidate["source_experts"],
            device=first_order_grid.device,
            dtype=torch.long,
        )
        destinations = torch.as_tensor(
            candidate["destination_experts"],
            device=first_order_grid.device,
            dtype=torch.long,
        )
        first_order_change = first_order_grid[tokens, destinations].sum()
        denominator = torch.sqrt(
            gradient_sq_norm[tokens].sum()
            * delta_sq_norm[tokens, destinations].sum()
        ).clamp_min(1e-12)
        router_margin = (
            router_scores[tokens, sources] - router_scores[tokens, destinations]
        ).mean()
        records.append({
            **candidate,
            "first_order_change": float(first_order_change.item()),
            "normalized_first_order_change": float(
                (first_order_change / denominator).item()
            ),
            "mean_router_margin": float(router_margin.item()),
        })
    return records


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
    exact_batch_size,
):
    if exact_batch_size < 2 or exact_batch_size % 2:
        raise ValueError("exact_batch_size must be a positive even number")
    candidates_per_forward = exact_batch_size // 2
    target_channels = target.shape[1]
    records = []
    max_native_mse_drift = 0.0
    max_native_output_drift = 0.0
    for start in range(0, len(candidates), candidates_per_forward):
        chunk = candidates[start:start + candidates_per_forward]
        route_rows = []
        candidate_count_vectors = []
        native_count_vector = torch.bincount(
            native_route_ids,
            minlength=int(moe_layer.num_routed_experts),
        )
        for candidate in chunk:
            baseline_ids = native_route_ids.clone()
            candidate_ids = native_route_ids.clone()
            tokens = torch.as_tensor(
                candidate["tokens"],
                device=native_route_ids.device,
                dtype=torch.long,
            )
            destinations = torch.as_tensor(
                candidate["destination_experts"],
                device=native_route_ids.device,
                dtype=torch.long,
            )
            candidate_ids[tokens] = destinations
            candidate_count_vector = torch.bincount(
                candidate_ids,
                minlength=int(moe_layer.num_routed_experts),
            )
            full_count_match = torch.equal(
                native_count_vector,
                candidate_count_vector,
            )
            if full_count_match != bool(candidate["count_preserving"]):
                raise RuntimeError(
                    f"Full route-count contract failed for {candidate['id']}"
                )
            candidate_count_vectors.append(candidate_count_vector.cpu().tolist())
            route_rows.extend((baseline_ids, candidate_ids))
        route_id_matrix = torch.stack(route_rows)
        route_weight_matrix = native_route_weights.unsqueeze(0).expand(
            len(route_rows), -1
        ).clone()
        model_batch = len(route_rows)
        with torch.inference_mode(), _forced_route_state(
            moe_layer,
            route_id_matrix,
            route_weight_matrix,
        ):
            output = model(
                noised_latent.repeat(model_batch, 1, 1, 1, 1),
                timestep.repeat(model_batch),
                context=label.repeat(model_batch),
            )
        prediction = _extract_prediction(output, target_channels)
        losses = _per_sample_mse(
            prediction,
            target.repeat(model_batch, 1, 1, 1),
        ).reshape(len(chunk), 2)
        prediction_pairs = prediction.reshape(
            len(chunk),
            2,
            *prediction.shape[1:],
        )
        native_rows = prediction_pairs[:, 0]
        max_native_mse_drift = max(
            max_native_mse_drift,
            float((losses[:, 0] - native_loss).abs().max().item()),
        )
        max_native_output_drift = max(
            max_native_output_drift,
            float((native_rows - native_prediction).abs().max().item()),
        )
        exact_changes = losses[:, 1] - losses[:, 0]
        output_changes = (
            prediction_pairs[:, 1] - prediction_pairs[:, 0]
        ).abs().flatten(1).max(dim=1).values
        for index, candidate in enumerate(chunk):
            records.append({
                **candidate,
                "exact_mse_change": float(exact_changes[index].item()),
                "exact_mse_change_relative": float(
                    exact_changes[index].item() / native_loss.item()
                ),
                "max_abs_output_change": float(output_changes[index].item()),
                "full_native_count_vector": native_count_vector.cpu().tolist(),
                "full_candidate_count_vector": candidate_count_vectors[index],
                "full_count_match": bool(candidate["count_preserving"]),
            })

    no_op_ids = native_route_ids.unsqueeze(0).expand(2, -1).clone()
    no_op_weights = native_route_weights.unsqueeze(0).expand(2, -1).clone()
    with torch.inference_mode(), _forced_route_state(
        moe_layer,
        no_op_ids,
        no_op_weights,
    ):
        no_op_output = model(
            noised_latent.repeat(2, 1, 1, 1, 1),
            timestep.repeat(2),
            context=label.repeat(2),
        )
    no_op_prediction = _extract_prediction(no_op_output, target_channels)
    no_op_losses = _per_sample_mse(
        no_op_prediction,
        target.repeat(2, 1, 1, 1),
    )
    controls = {
        "max_abs_paired_native_mse_drift": max_native_mse_drift,
        "max_abs_paired_native_output_drift": max_native_output_drift,
        "max_abs_noop_mse_change": float(
            abs(no_op_losses[1].item() - no_op_losses[0].item())
        ),
        "max_abs_noop_output_change": float((
            no_op_prediction[1] - no_op_prediction[0]
        ).abs().max().item()),
    }
    return records, controls


def _pair_concordance(predicted, exact):
    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    if predicted.shape != exact.shape or predicted.ndim != 1:
        raise ValueError("Concordance inputs must be aligned vectors")
    if predicted.size < 2:
        return None
    concordant = 0.0
    comparisons = 0
    for left, right in combinations(range(predicted.size), 2):
        predicted_difference = predicted[left] - predicted[right]
        exact_difference = exact[left] - exact[right]
        comparisons += 1
        if predicted_difference == 0 or exact_difference == 0:
            concordant += 0.5
        elif np.signbit(predicted_difference) == np.signbit(exact_difference):
            concordant += 1.0
    return float(concordant / comparisons)


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize_arm(records, native_mse, epsilon_num):
    if len(records) != ARM_CANDIDATES:
        raise ValueError("A locked cycle arm must contain exactly 64 records")
    if native_mse <= 0 or epsilon_num <= 0:
        raise ValueError("native_mse and epsilon_num must be positive")
    predicted_gain = np.asarray([
        -record["first_order_change"] / native_mse for record in records
    ], dtype=np.float64)
    exact_gain = np.asarray([
        -record["exact_mse_change"] / native_mse for record in records
    ], dtype=np.float64)
    selected_index = int(predicted_gain.argmax())
    selected_non_native = bool(predicted_gain[selected_index] > 0)
    if selected_non_native:
        selected_gain = float(exact_gain[selected_index])
        selected_flips = int(records[selected_index]["changed_tokens"])
    else:
        selected_index = None
        selected_gain = 0.0
        selected_flips = 0
    oracle_index = int(exact_gain.argmax())
    oracle_gain = max(0.0, float(exact_gain[oracle_index]))
    predicted_positive = predicted_gain > 0
    exact_positive = exact_gain > epsilon_num
    true_positive = predicted_positive & exact_positive
    sign_agreement = np.mean(predicted_positive == exact_positive)
    return {
        "num_candidates": len(records),
        "selected_candidate_index": selected_index,
        "selected_non_native": selected_non_native,
        "selected_gain": selected_gain,
        "selected_changed_tokens": selected_flips,
        "selected_per_flip_gain": (
            float(selected_gain / selected_flips) if selected_flips else 0.0
        ),
        "selected_positive": bool(selected_gain > epsilon_num),
        "selected_harm": bool(selected_gain < -epsilon_num),
        "oracle_candidate_index": oracle_index,
        "oracle_gain": oracle_gain,
        "oracle_regret": float(oracle_gain - selected_gain),
        "pair_concordance": _pair_concordance(predicted_gain, exact_gain),
        "spearman": _correlation(
            _rankdata(predicted_gain),
            _rankdata(exact_gain),
        ),
        "sign_agreement": float(sign_agreement),
        "exact_beneficial_rate": float(exact_positive.mean()),
        "predicted_beneficial_rate": float(predicted_positive.mean()),
        "predicted_beneficial_precision": (
            float(true_positive.sum() / predicted_positive.sum())
            if predicted_positive.any() else None
        ),
        "predicted_beneficial_recall": (
            float(true_positive.sum() / exact_positive.sum())
            if exact_positive.any() else None
        ),
        "mean_exact_gain": float(exact_gain.mean()),
        "mean_predicted_gain": float(predicted_gain.mean()),
        "epsilon_num": float(epsilon_num),
    }


def _summarize_six_audit(six_records, audit_records, epsilon_num):
    six_by_id = {record["id"]: record for record in six_records}
    pairs_by_parent = {}
    for record in audit_records:
        pairs_by_parent.setdefault(record["parent_six_id"], []).append(record)
    audits = []
    for six_record in six_records[:AUDITED_SIX_CANDIDATES]:
        pairs = pairs_by_parent.get(six_record["id"], [])
        if len(pairs) != 3:
            raise RuntimeError("Every audited six-cycle must have three pair swaps")
        six_gain = -float(six_record["exact_mse_change_relative"])
        pair_gains = [
            -float(pair["exact_mse_change_relative"]) for pair in pairs
        ]
        audits.append({
            "six_cycle_id": six_record["id"],
            "six_gain": six_gain,
            "direct_four_gains": pair_gains,
            "unique_six": bool(
                six_gain > epsilon_num
                and all(gain <= epsilon_num for gain in pair_gains)
            ),
            "six_minus_best_four_gain": float(six_gain - max(pair_gains)),
        })
    if len(six_by_id) != len(six_records):
        raise RuntimeError("Duplicate six-cycle IDs")
    return {
        "num_audited_six_cycles": len(audits),
        "unique_six_rate": float(np.mean([
            record["unique_six"] for record in audits
        ])),
        "has_unique_six": bool(any(record["unique_six"] for record in audits)),
        "mean_six_minus_best_four_gain": float(np.mean([
            record["six_minus_best_four_gain"] for record in audits
        ])),
        "records": audits,
    }


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
    exact_batch_size,
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
            raise RuntimeError("The cycle probe did not capture the MoE output")
        moe_gradient, = torch.autograd.grad(native_loss, capture.moe_output)
    finally:
        capture.stop()

    hidden_states = capture.hidden_states
    labels = capture.labels
    if hidden_states is None or labels is None:
        raise RuntimeError("The cycle probe did not capture router inputs")
    with torch.no_grad():
        native_weights, native_indices, _ = moe_layer.compute_router(
            hidden_states,
            labels,
        )
        router_scores = _all_router_weights(moe_layer, hidden_states)
    native_route_ids = native_indices[0, :, 0]
    native_route_weights = native_weights[0, :, 0]
    if not torch.equal(router_scores[0].argmax(dim=-1), native_route_ids):
        raise RuntimeError("Native routes disagree with all-router scores")

    banks, audits = build_candidate_banks(
        native_route_ids.cpu().numpy(),
        int(moe_layer.num_routed_experts),
        candidate_seed,
    )
    first_order_grid, gradient_sq_norm, delta_sq_norm = _edge_first_order_grid(
        moe_layer=moe_layer,
        hidden_states=hidden_states[0],
        moe_gradient=moe_gradient.detach()[0],
        native_experts=native_route_ids,
        native_weights=native_route_weights,
    )
    scored_banks = {
        arm: _score_candidates(
            candidates,
            first_order_grid,
            gradient_sq_norm,
            delta_sq_norm,
            router_scores[0],
        )
        for arm, candidates in banks.items()
    }
    scored_audits = _score_candidates(
        audits,
        first_order_grid,
        gradient_sq_norm,
        delta_sq_norm,
        router_scores[0],
    )
    flat_scored = [
        record for arm in ARM_NAMES for record in scored_banks[arm]
    ] + scored_audits
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
        candidates=flat_scored,
        exact_batch_size=exact_batch_size,
    )
    exact_by_id = {record["id"]: record for record in exact_records}
    if len(exact_by_id) != len(exact_records):
        raise RuntimeError("Candidate IDs must be unique within a probe cell")
    final_banks = {
        arm: [exact_by_id[record["id"]] for record in scored_banks[arm]]
        for arm in ARM_NAMES
    }
    final_audits = [exact_by_id[record["id"]] for record in scored_audits]

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
    no_op_relative = (
        exact_controls["max_abs_noop_mse_change"] / native_loss.item()
    )
    epsilon_num = max(1e-8, 10.0 * no_op_relative)
    summaries = {
        arm: summarize_arm(records, native_loss.item(), epsilon_num)
        for arm, records in final_banks.items()
    }
    count_mismatches = 0
    for arm in COUNT_PRESERVING_ARMS:
        for record in final_banks[arm]:
            if (
                not record["full_count_match"]
                or record["full_native_count_vector"]
                != record["full_candidate_count_vector"]
            ):
                count_mismatches += 1
    for record in final_audits:
        if (
            not record["full_count_match"]
            or record["full_native_count_vector"]
            != record["full_candidate_count_vector"]
        ):
            count_mismatches += 1
    return {
        "block_index": int(block_index),
        "sigma": float(sigma),
        "timestep": float(timestep.item()),
        "native_mse": float(native_loss.item()),
        "candidate_seed": int(candidate_seed),
        "epsilon_num": float(epsilon_num),
        "arms": {
            arm: {
                "summary": summaries[arm],
                "records": final_banks[arm],
            }
            for arm in ARM_NAMES
        },
        "six_cycle_audit": _summarize_six_audit(
            final_banks["six_cycle"],
            final_audits,
            epsilon_num,
        ),
        "numerical_controls": {
            **exact_controls,
            **forced_control,
            "count_mismatches": int(count_mismatches),
        },
    }


def run_cycle_probe_case(
    model,
    runtime_cfg,
    latent_path,
    label,
    seed,
    block_indices,
    sigmas,
    exact_batch_size=4,
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
    _validate_moe_block_contract(model, block_indices)
    if exact_batch_size < 2 or exact_batch_size % 2:
        raise ValueError("exact_batch_size must be a positive even number")

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
                        "candidate",
                        block_index,
                        f"{sigma:.17g}",
                    ),
                    exact_batch_size=exact_batch_size,
                ))
        finally:
            capture.close()
    result = {
        "cycle_probe_version": PROBE_VERSION,
        "diagnostic_scope": (
            "frozen-checkpoint exact denoising-utility and first-order fidelity "
            "gate; not a training, sampling, FID, or novelty claim"
        ),
        "label": int(label),
        "latent": str(latent_path),
        "latent_key": latent_key,
        "seed": int(seed),
        "block_indices": list(block_indices),
        "sigmas": list(sigmas),
        "exact_batch_size": int(exact_batch_size),
        "probe_seconds": float(time.perf_counter() - probe_start),
        "cells": cells,
    }
    return result


def run_cycle_probe(
    checkpoint_path,
    weights_checkpoint_path,
    latent_path,
    label,
    seed,
    block_indices,
    sigmas,
    exact_batch_size=4,
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
        result = run_cycle_probe_case(
            model=model,
            runtime_cfg=runtime_cfg,
            latent_path=latent_path,
            label=label,
            seed=seed,
            block_indices=block_indices,
            sigmas=sigmas,
            exact_batch_size=exact_batch_size,
            latent_key=latent_key,
        )
    finally:
        del model
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    result.update({
        "checkpoint": str(checkpoint_path),
        "weights_checkpoint": str(weights_checkpoint_path),
        "checkpoint_step": int(checkpoint_step),
        "weights_checkpoint_step": int(weights_step),
        "checkpoint_state": state_name,
        "config": str(config_path),
        "model_name": runtime_cfg.model_name,
        "device": str(device),
        "num_threads": int(num_threads),
        "thread_config": thread_config,
        "model_load_seconds": float(load_seconds),
    })
    return result
