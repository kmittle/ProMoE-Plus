import inspect
from collections import OrderedDict

import torch


def _supports_token_choice_router(moe_module) -> bool:
    if not hasattr(moe_module, "compute_router"):
        return False
    signature = inspect.signature(moe_module.compute_router)
    params = [name for name in signature.parameters if name != "self"]
    return len(params) == 2


def find_token_choice_moe_blocks(model) -> list[tuple[int, object, object]]:
    moe_blocks = []
    if not hasattr(model, "blocks"):
        return moe_blocks

    for block_idx, block in enumerate(model.blocks):
        if not getattr(block, "use_moe", False):
            continue
        moe_module = getattr(block, "mlp", None)
        if moe_module is None or not _supports_token_choice_router(moe_module):
            continue
        moe_blocks.append((block_idx, block, moe_module))
    return moe_blocks


class TokenRoutingCapture:
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
            block_idx: moe_module.num_routed_experts
            for block_idx, _, moe_module in self.moe_blocks
        }

        self._enabled = False
        self._current_denoise_step = None
        self._current_scheduler_timestep = None
        self._pending_labels = {}
        self._captures = OrderedDict()
        self._handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for block_idx, block, moe_module in self.moe_blocks:
            self._handles.append(moe_module.register_forward_hook(self._make_mlp_hook(block_idx)))
            self._handles.append(block.register_forward_hook(self._make_block_hook(block_idx)))

    def _make_mlp_hook(self, block_idx: int):
        def hook(module, inputs, output):
            if not self._enabled:
                return
            if len(inputs) < 2:
                return

            hidden_states = inputs[0]
            labels = inputs[1]
            with torch.no_grad():
                router_result = module.compute_router(hidden_states, labels)
            expert_indices = router_result[1]
            self._pending_labels[block_idx] = expert_indices[..., 0].detach().cpu()

        return hook

    def _make_block_hook(self, block_idx: int):
        def hook(module, inputs, output):
            if not self._enabled:
                return

            if isinstance(output, tuple):
                block_output = output[0]
            else:
                block_output = output

            expert_labels = self._pending_labels.pop(block_idx, None)
            if expert_labels is None:
                return

            self._captures[block_idx] = {
                "features": block_output.detach().float().cpu(),
                "labels": expert_labels,
                "denoise_step": self._current_denoise_step,
                "scheduler_timestep": self._current_scheduler_timestep,
            }

        return hook

    def enable(self, denoise_step: int, scheduler_timestep: float) -> None:
        self._enabled = True
        self._current_denoise_step = denoise_step
        self._current_scheduler_timestep = scheduler_timestep
        self._pending_labels.clear()
        self._captures = OrderedDict()

    def disable_and_collect(self) -> OrderedDict:
        self._enabled = False
        missing_blocks = [
            block_idx for block_idx in self.block_indices if block_idx not in self._captures
        ]
        if missing_blocks:
            raise RuntimeError(
                f"Missing captured routing outputs for blocks: {missing_blocks}"
            )
        captures = self._captures
        self._captures = OrderedDict()
        self._pending_labels.clear()
        return captures

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
