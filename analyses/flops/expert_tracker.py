"""Hook-based expert activation tracker for MoE models."""

from collections import defaultdict

from analyses.flops.tracking_utils import find_moe_blocks


def raw_counts_to_frequencies(raw_counts, num_routed_experts_per_block):
    """Convert raw expert counts into normalized per-block frequencies.

    Args:
        raw_counts: dict, block_idx -> (counts_dict, total_tokens)
        num_routed_experts_per_block: dict, block_idx -> num_routed_experts

    Returns:
        dict: block_idx -> list[float], where each value is the normalized
        activation frequency of that routed expert within the specified block.
    """
    frequencies = {}
    for block_idx in sorted(num_routed_experts_per_block.keys()):
        num_routed = num_routed_experts_per_block[block_idx]
        counts_dict, total = raw_counts.get(block_idx, ({}, 0))
        if total == 0:
            frequencies[block_idx] = [0.0] * num_routed
        else:
            frequencies[block_idx] = [
                counts_dict.get(expert_idx, 0) / total for expert_idx in range(num_routed)
            ]
    return frequencies


class ExpertActivationTracker:
    """Tracks which experts are activated in each MoE block during inference.

    Usage:
        tracker = ExpertActivationTracker(model)
        tracker.start()
        # ... run forward passes ...
        tracker.stop()
        freq = tracker.get_frequencies()  # dict: block_idx -> {expert_id: count}
    """

    def __init__(self, model):
        self.model = model
        self.moe_blocks = find_moe_blocks(model)
        self._counts = {}  # block_idx -> defaultdict(int)
        self._total_tokens = {}  # block_idx -> int (total cond token assignments)
        self._num_routed_experts_per_block = {}
        self._current_forward_counts = {}  # block_idx -> defaultdict(int)
        self._current_forward_total_tokens = {}  # block_idx -> int

    def start(self):
        """Monkey-patch compute_router on all MoE blocks to track expert assignments."""
        self._counts.clear()
        self._total_tokens.clear()
        self._num_routed_experts_per_block.clear()
        self.begin_forward()

        for block_idx, moe_module in self.moe_blocks:
            self._counts[block_idx] = defaultdict(int)
            self._total_tokens[block_idx] = 0
            self._num_routed_experts_per_block[block_idx] = moe_module.num_routed_experts

            self._patch_compute_router(block_idx, moe_module)

    def _patch_compute_router(self, block_idx, moe_module):
        """Replace compute_router with a wrapper that records routing decisions."""
        tracker = self
        use_uncond_expert = getattr(moe_module, "use_uncond_expert", True)

        original_compute_router = moe_module.compute_router
        moe_module._original_compute_router = original_compute_router

        def hooked_compute_router(
            hidden_states,
            labels,
            *router_args,
            **router_kwargs,
        ):
            result = original_compute_router(
                hidden_states,
                labels,
                *router_args,
                **router_kwargs,
            )
            expert_indices = result[1]
            # expert_indices: (batch_size, seq_len, top_k)
            batch_size, seq_len, top_k = expert_indices.shape

            if use_uncond_expert:
                # Only track conditional tokens — uncond tokens (label=1000)
                # are always routed to the dedicated uncond expert.
                flat_labels = labels.view(batch_size, 1).expand(-1, seq_len).reshape(-1)
                cond_mask = (flat_labels != 1000)
            else:
                # No uncond expert — all tokens participate in prototypical routing.
                cond_mask = None

            if cond_mask is None:
                # Track all tokens
                all_indices = expert_indices.view(-1, top_k).detach().cpu()
                flat_indices = [int(idx) for idx in all_indices.view(-1).tolist()]
                current_counts = tracker._current_forward_counts.setdefault(
                    block_idx, defaultdict(int)
                )
                for idx in flat_indices:
                    tracker._counts[block_idx][idx] += 1
                    current_counts[idx] += 1
                total_assignments = len(flat_indices)
                tracker._total_tokens[block_idx] += total_assignments
                tracker._current_forward_total_tokens[block_idx] = (
                    tracker._current_forward_total_tokens.get(block_idx, 0) + total_assignments
                )
            elif cond_mask.any():
                # (num_cond_tokens, top_k)
                cond_indices = expert_indices.view(-1, top_k)[cond_mask].detach().cpu()
                flat_indices = [int(idx) for idx in cond_indices.view(-1).tolist()]
                current_counts = tracker._current_forward_counts.setdefault(
                    block_idx, defaultdict(int)
                )
                for idx in flat_indices:
                    tracker._counts[block_idx][idx] += 1
                    current_counts[idx] += 1
                total_assignments = len(flat_indices)
                tracker._total_tokens[block_idx] += total_assignments
                tracker._current_forward_total_tokens[block_idx] = (
                    tracker._current_forward_total_tokens.get(block_idx, 0) + total_assignments
                )

            return result

        moe_module.compute_router = hooked_compute_router

    def begin_forward(self):
        """Reset per-forward raw-count buffers before a model forward."""
        self._current_forward_counts = {}
        self._current_forward_total_tokens = {}

    def consume_current_forward_raw_counts(self):
        """Return raw counts from the latest forward pass and clear the buffer."""
        raw = {}
        for block_idx in sorted(self._current_forward_counts.keys()):
            raw[block_idx] = (
                dict(self._current_forward_counts[block_idx]),
                self._current_forward_total_tokens.get(block_idx, 0),
            )
        self.begin_forward()
        return raw

    def stop(self):
        """Restore original compute_router on all MoE blocks."""
        for _, moe_module in self.moe_blocks:
            if hasattr(moe_module, "_original_compute_router"):
                moe_module.compute_router = moe_module._original_compute_router
                del moe_module._original_compute_router

    def get_frequencies(self):
        """Return expert activation frequencies per block (conditional routed experts only).

        Returns:
            dict: block_idx -> list of frequencies (length = num_routed_experts).
                  Each frequency is (count_for_expert / total_cond_token_assignments).
                  Unconditional expert and shared expert are excluded.
        """
        return raw_counts_to_frequencies(
            self.get_raw_counts(),
            self._num_routed_experts_per_block,
        )

    def get_raw_counts(self):
        """Return raw counts and total tokens for cross-GPU aggregation.

        Returns:
            dict: block_idx -> (counts_dict, total_tokens)
        """
        raw = {}
        for block_idx in sorted(self._counts.keys()):
            raw[block_idx] = (dict(self._counts[block_idx]), self._total_tokens[block_idx])
        return raw

    def get_num_routed_experts_per_block(self):
        """Return the number of routed experts for each tracked MoE block."""
        return dict(self._num_routed_experts_per_block)

    def merge_raw_counts(self, other_raw):
        """Merge raw counts from another tracker (or gathered dict) into this one.

        Args:
            other_raw: dict from get_raw_counts(), block_idx -> (counts_dict, total_tokens)
        """
        for block_idx, (counts, total) in other_raw.items():
            if block_idx not in self._counts:
                self._counts[block_idx] = defaultdict(int)
                self._total_tokens[block_idx] = 0
                # try to infer num_routed_experts from existing blocks
                if block_idx not in self._num_routed_experts_per_block and self._num_routed_experts_per_block:
                    ref = next(iter(self._num_routed_experts_per_block.values()))
                    self._num_routed_experts_per_block[block_idx] = ref
            for expert_id, cnt in counts.items():
                self._counts[block_idx][expert_id] += cnt
            self._total_tokens[block_idx] += total

    @property
    def num_moe_blocks(self):
        return len(self.moe_blocks)
