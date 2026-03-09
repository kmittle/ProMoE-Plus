"""Hook-based expert activation tracker for MoE models."""

import torch
import torch.nn as nn
import inspect
from collections import defaultdict


def _is_tc_compute_router(moe_module):
    """Check if the module's compute_router follows TC (Token-Choice) signature.

    TC models: compute_router(self, hidden_states, labels) -> returns tuple with
    (router_weights, expert_indices, ...) where expert_indices has shape (B, S, top_k).

    EC models: compute_router(self, cond_hidden_states) -> incompatible signature/returns.
    """
    if not hasattr(moe_module, "compute_router"):
        return False
    sig = inspect.signature(moe_module.compute_router)
    # TC: (hidden_states, labels); EC: (cond_hidden_states)
    params = [p for p in sig.parameters if p != "self"]
    return len(params) == 2


def _find_moe_blocks(model):
    """Find all MoE SparseMoeBlock modules and return (block_index, module) pairs.

    block_index is the position within model.blocks (DiTBlock index).
    Only DiTBlocks with use_moe=True contain a SparseMoeBlock as their .mlp attribute.
    Skips Expert-Choice (EC) blocks whose compute_router has an incompatible signature.
    """
    moe_blocks = []
    if not hasattr(model, "blocks"):
        return moe_blocks
    for i, block in enumerate(model.blocks):
        if hasattr(block, "use_moe") and block.use_moe:
            moe_module = block.mlp  # SparseMoeBlock
            if _is_tc_compute_router(moe_module):
                moe_blocks.append((i, moe_module))
    return moe_blocks


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
        self.moe_blocks = _find_moe_blocks(model)
        self._counts = {}  # block_idx -> defaultdict(int)
        self._total_tokens = {}  # block_idx -> int (total cond token assignments)
        self._num_routed_experts_per_block = {}

    def start(self):
        """Monkey-patch compute_router on all MoE blocks to track expert assignments."""
        self._counts.clear()
        self._total_tokens.clear()
        self._num_routed_experts_per_block.clear()

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

        def hooked_compute_router(hidden_states, labels):
            result = original_compute_router(hidden_states, labels)
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
                for idx in all_indices.view(-1).tolist():
                    tracker._counts[block_idx][int(idx)] += 1
                tracker._total_tokens[block_idx] += all_indices.numel()
            elif cond_mask.any():
                # (num_cond_tokens, top_k)
                cond_indices = expert_indices.view(-1, top_k)[cond_mask].detach().cpu()
                for idx in cond_indices.view(-1).tolist():
                    tracker._counts[block_idx][int(idx)] += 1
                tracker._total_tokens[block_idx] += cond_indices.numel()

            return result

        moe_module.compute_router = hooked_compute_router

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
        frequencies = {}
        for block_idx in sorted(self._counts.keys()):
            num_routed = self._num_routed_experts_per_block[block_idx]
            total = self._total_tokens[block_idx]
            if total == 0:
                frequencies[block_idx] = [0.0] * num_routed
            else:
                freq_list = []
                for e in range(num_routed):
                    freq_list.append(self._counts[block_idx][e] / total)
                frequencies[block_idx] = freq_list
        return frequencies

    def get_raw_counts(self):
        """Return raw counts and total tokens for cross-GPU aggregation.

        Returns:
            dict: block_idx -> (counts_dict, total_tokens)
        """
        raw = {}
        for block_idx in sorted(self._counts.keys()):
            raw[block_idx] = (dict(self._counts[block_idx]), self._total_tokens[block_idx])
        return raw

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
