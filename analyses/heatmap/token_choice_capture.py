from __future__ import annotations

import math
from collections import OrderedDict

from analyses.t_SNE.routing_capture import find_token_choice_moe_blocks


class TokenChoiceExpertIndexCapture:
    """Capture per-token routed expert indices for every token-choice MoE block."""

    def __init__(self, model):
        self.model = model
        self.moe_blocks = find_token_choice_moe_blocks(model)
        if not self.moe_blocks:
            raise RuntimeError(
                "No token-choice MoE blocks with compute_router(hidden_states, labels) "
                "were found in this model."
            )

        self.block_indices = [block_idx for block_idx, _, _ in self.moe_blocks]
        self.num_routed_experts = {
            block_idx: int(moe_module.num_routed_experts)
            for block_idx, _, moe_module in self.moe_blocks
        }

        self._enabled = False
        self._current_denoise_step = None
        self._current_scheduler_timestep = None
        self._captures = OrderedDict()
        self._original_compute_router = {}
        self._patch_compute_router()

    def _patch_compute_router(self) -> None:
        for block_idx, _, moe_module in self.moe_blocks:
            original_compute_router = moe_module.compute_router
            self._original_compute_router[block_idx] = original_compute_router

            def hooked_compute_router(hidden_states, labels, *,
                                      _block_idx=block_idx,
                                      _original=original_compute_router):
                result = _original(hidden_states, labels)
                if not self._enabled:
                    return result

                expert_indices = result[1]
                if expert_indices.ndim != 3:
                    raise RuntimeError(
                        f"Expected expert_indices to have 3 dims, got {expert_indices.shape}."
                    )

                _, token_count, _ = expert_indices.shape
                grid_size = int(round(math.sqrt(token_count)))
                if grid_size * grid_size != token_count:
                    raise RuntimeError(
                        f"Token count {token_count} is not a perfect square, "
                        "cannot form an expert heatmap grid."
                    )

                mean_expert_indices = expert_indices.detach().float().mean(dim=-1).cpu()
                self._captures[_block_idx] = OrderedDict(
                    expert_index_maps=mean_expert_indices,
                    denoise_step=self._current_denoise_step,
                    scheduler_timestep=self._current_scheduler_timestep,
                    grid_size=grid_size,
                )
                return result

            moe_module.compute_router = hooked_compute_router

    def enable(self, denoise_step: int, scheduler_timestep: float) -> None:
        self._enabled = True
        self._current_denoise_step = denoise_step
        self._current_scheduler_timestep = scheduler_timestep
        self._captures = OrderedDict()

    def disable_and_collect(self) -> OrderedDict:
        self._enabled = False
        missing_blocks = [
            block_idx for block_idx in self.block_indices if block_idx not in self._captures
        ]
        if missing_blocks:
            raise RuntimeError(
                f"Missing expert-index captures for MoE blocks: {missing_blocks}."
            )
        captures = self._captures
        self._captures = OrderedDict()
        return captures

    def close(self) -> None:
        for block_idx, _, moe_module in self.moe_blocks:
            if block_idx in self._original_compute_router:
                moe_module.compute_router = self._original_compute_router[block_idx]
        self._original_compute_router.clear()
