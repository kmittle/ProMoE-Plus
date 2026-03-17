"""Track the mean number of activated parameters per forward pass for MoE models.

For dense models (no MoE), all parameters are activated every forward pass.
For MoE models, only a subset of experts is activated per forward pass:
- Shared expert: always activated
- Routed experts: only experts that receive actual tokens are activated
  (the remaining experts receive a dummy zero-input forward whose output
  is discarded; those are NOT counted as activated)

Activation detection works by intercepting compute_router to capture
expert_indices — the set of unique expert IDs in the routing result tells
us exactly which experts received real tokens, with no heuristics.
"""

from analyses.flops.tracking_utils import find_moe_blocks


class ActivatedParamsTracker:
    """Track the mean activated parameters per single forward pass.

    Usage:
        tracker = ActivatedParamsTracker(model)
        tracker.start()
        for ...:
            model(x, t, context=y)
            tracker.record_forward_pass()
        tracker.stop()
        mean_params = tracker.get_mean_activated_params()
    """

    def __init__(self, model):
        self.model = model
        self.moe_blocks = find_moe_blocks(model)
        # Per-MoE-block: param count of each expert (indexed by expert id)
        self._expert_params = {}  # block_idx -> list[int]
        # Non-MoE parameter count (always activated, includes shared experts)
        self._non_moe_params = 0
        # Per-forward records: which experts were activated in each MoE block
        self._forward_records = []  # list of dict: block_idx -> set of activated expert ids
        self._current_forward = {}  # block_idx -> set (populated during a model forward)
        self._started = False

    def start(self):
        """Patch MoE block forwards to track expert activation."""
        if self._started:
            return
        self._started = True
        self._forward_records.clear()
        self._current_forward.clear()

        self._compute_param_breakdown()

        for block_idx, moe_module in self.moe_blocks:
            self._patch_forward(block_idx, moe_module)

    def _compute_param_breakdown(self):
        """Separate total params into non-MoE (always-on) and per-expert params."""
        for block_idx, moe_module in self.moe_blocks:
            expert_params_list = []
            for expert in moe_module.experts:
                p_count = sum(p.numel() for p in expert.parameters())
                expert_params_list.append(p_count)
            self._expert_params[block_idx] = expert_params_list

        # non_moe_params = total - routed expert params
        # (shared expert is always activated, so it stays in the "always-on" base)
        total_params = sum(p.numel() for p in self.model.parameters())
        routed_expert_params = sum(
            sum(ep) for ep in self._expert_params.values()
        )
        self._non_moe_params = total_params - routed_expert_params

    def _patch_forward(self, block_idx, moe_module):
        """Patch SparseMoeBlock.forward to capture routing decisions."""
        tracker = self
        original_forward = moe_module.forward
        moe_module._original_forward_for_params = original_forward

        def patched_forward(hidden_states, labels):
            # Temporarily wrap compute_router to capture expert_indices.
            # This composes correctly with ExpertActivationTracker which may
            # have already patched compute_router — we wrap whatever is
            # currently installed and restore it afterward.
            prev_compute_router = moe_module.compute_router
            captured = {}

            def capturing_compute_router(hs, lbs):
                result = prev_compute_router(hs, lbs)
                captured['expert_indices'] = result[1]  # expert_indices
                return result

            moe_module.compute_router = capturing_compute_router
            result = original_forward(hidden_states, labels)
            moe_module.compute_router = prev_compute_router

            # Record which experts received real tokens
            if 'expert_indices' in captured:
                activated = set(captured['expert_indices'].unique().tolist())
                tracker._current_forward[block_idx] = activated

            return result

        moe_module.forward = patched_forward

    def record_forward_pass(self):
        """Call after each model forward pass to snapshot the current activation record.

        Must be called after each model.forward() and before the next one.
        """
        if self._current_forward:
            self._forward_records.append(dict(self._current_forward))
        self._current_forward = {}

    def stop(self):
        """Restore original forwards on all MoE blocks."""
        for _, moe_module in self.moe_blocks:
            if hasattr(moe_module, "_original_forward_for_params"):
                moe_module.forward = moe_module._original_forward_for_params
                del moe_module._original_forward_for_params
        self._started = False

    def _compute_activated_list(self):
        """Compute per-forward activated param counts from recorded forwards."""
        activated_list = []
        for record in self._forward_records:
            activated = self._non_moe_params
            for block_idx, expert_params_list in self._expert_params.items():
                activated_experts = record.get(block_idx, set())
                for eid in activated_experts:
                    if eid < len(expert_params_list):
                        activated += expert_params_list[eid]
            activated_list.append(activated)
        return activated_list

    def get_mean_activated_params(self):
        """Compute the mean number of activated parameters per forward pass.

        Returns:
            float: Mean activated parameters across all recorded forward passes.
                   For dense models, this equals the total parameter count.
        """
        if not self.moe_blocks:
            return float(sum(p.numel() for p in self.model.parameters()))

        if not self._forward_records:
            return 0.0

        activated_list = self._compute_activated_list()
        return sum(activated_list) / len(activated_list)

    def get_stats(self):
        """Return detailed statistics about activated parameters.

        Returns:
            dict with keys:
                - total_params: Total model parameters
                - non_moe_params: Always-activated parameters (includes shared experts)
                - mean_activated_params: Mean activated params per forward
                - min_activated_params: Min activated params across forwards
                - max_activated_params: Max activated params across forwards
                - num_forwards: Number of recorded forward passes
                - activation_ratio: mean_activated / total
        """
        total_params = sum(p.numel() for p in self.model.parameters())

        if not self.moe_blocks:
            return {
                "total_params": total_params,
                "non_moe_params": total_params,
                "mean_activated_params": float(total_params),
                "min_activated_params": float(total_params),
                "max_activated_params": float(total_params),
                "num_forwards": max(len(self._forward_records), 1),
                "activation_ratio": 1.0,
            }

        if not self._forward_records:
            return {
                "total_params": total_params,
                "non_moe_params": self._non_moe_params,
                "mean_activated_params": 0.0,
                "min_activated_params": 0.0,
                "max_activated_params": 0.0,
                "num_forwards": 0,
                "activation_ratio": 0.0,
            }

        activated_list = self._compute_activated_list()
        mean_act = sum(activated_list) / len(activated_list)
        return {
            "total_params": total_params,
            "non_moe_params": self._non_moe_params,
            "mean_activated_params": mean_act,
            "min_activated_params": min(activated_list),
            "max_activated_params": max(activated_list),
            "num_forwards": len(activated_list),
            "activation_ratio": mean_act / total_params if total_params > 0 else 0.0,
        }
