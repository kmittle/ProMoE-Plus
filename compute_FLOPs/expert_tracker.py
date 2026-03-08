"""Hook-based expert activation tracker for MoE models."""

import torch
import torch.nn as nn
from collections import defaultdict


def _find_moe_blocks(model):
    """Find all MoE SparseMoeBlock modules and return (block_index, module) pairs.

    block_index is the position within model.blocks (DiTBlock index).
    Only DiTBlocks with use_moe=True contain a SparseMoeBlock as their .mlp attribute.
    """
    moe_blocks = []
    if not hasattr(model, "blocks"):
        return moe_blocks
    for i, block in enumerate(model.blocks):
        if hasattr(block, "use_moe") and block.use_moe:
            moe_module = block.mlp  # SparseMoeBlock
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
        self._hooks = []
        self._counts = {}  # block_idx -> defaultdict(int)
        self._total_tokens = {}  # block_idx -> int (total cond token assignments)
        self._num_routed_experts_per_block = {}

    def start(self):
        """Register forward hooks on all MoE blocks."""
        self._counts.clear()
        self._total_tokens.clear()
        self._num_routed_experts_per_block.clear()
        self._hooks.clear()

        for block_idx, moe_module in self.moe_blocks:
            self._counts[block_idx] = defaultdict(int)
            self._total_tokens[block_idx] = 0
            self._num_routed_experts_per_block[block_idx] = moe_module.num_routed_experts

            hook = self._make_hook(block_idx, moe_module)
            handle = moe_module.register_forward_hook(hook)
            self._hooks.append(handle)

    def _make_hook(self, block_idx, moe_module):
        """Create a forward hook that intercepts the routing decision."""
        tracker = self

        original_compute_router = moe_module.compute_router

        def hooked_compute_router(hidden_states, labels):
            router_weights, expert_indices, load_balance_loss = original_compute_router(hidden_states, labels)
            # expert_indices: (batch_size, seq_len, top_k)
            # Only track conditional tokens (label != num_classes, i.e. != 1000)
            # Uncond tokens are always routed to the dedicated uncond expert,
            # and shared expert has no routing — neither should be counted.
            batch_size, seq_len, top_k = expert_indices.shape
            flat_labels = labels.view(batch_size, 1).expand(-1, seq_len).reshape(-1)
            cond_mask = (flat_labels != 1000)  # uncond class = 1000

            if cond_mask.any():
                # (num_cond_tokens, top_k)
                cond_indices = expert_indices.view(-1, top_k)[cond_mask].detach().cpu()
                for idx in cond_indices.view(-1).tolist():
                    tracker._counts[block_idx][int(idx)] += 1
                tracker._total_tokens[block_idx] += cond_indices.numel()

            return router_weights, expert_indices, load_balance_loss

        moe_module.compute_router = hooked_compute_router

        def hook_fn(module, input, output):
            pass  # actual tracking is done in hooked_compute_router

        # Store reference to restore later
        moe_module._original_compute_router = original_compute_router

        return hook_fn

    def stop(self):
        """Remove all hooks and restore original compute_router."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
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

    @property
    def num_moe_blocks(self):
        return len(self.moe_blocks)
