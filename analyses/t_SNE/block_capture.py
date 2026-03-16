from collections import OrderedDict

import torch


def find_transformer_blocks(model) -> list[tuple[int, object]]:
    if not hasattr(model, "blocks"):
        return []
    return [(block_idx, block) for block_idx, block in enumerate(model.blocks)]


class BlockOutputCapture:
    def __init__(self, model, block_indices: list[int] | None = None):
        self.model = model
        all_blocks = find_transformer_blocks(model)
        if not all_blocks:
            raise RuntimeError("No transformer blocks were found on this model.")

        if block_indices is None:
            self.blocks = all_blocks
        else:
            allowed = set(block_indices)
            self.blocks = [
                (block_idx, block)
                for block_idx, block in all_blocks
                if block_idx in allowed
            ]
        if not self.blocks:
            raise RuntimeError("No blocks matched the requested block indices.")

        self.block_indices = [block_idx for block_idx, _ in self.blocks]
        self._enabled = False
        self._current_denoise_step = None
        self._current_scheduler_timestep = None
        self._captures = OrderedDict()
        self._handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for block_idx, block in self.blocks:
            self._handles.append(block.register_forward_hook(self._make_block_hook(block_idx)))

    def _make_block_hook(self, block_idx: int):
        def hook(module, inputs, output):
            if not self._enabled:
                return

            if isinstance(output, tuple):
                block_output = output[0]
            else:
                block_output = output

            self._captures[block_idx] = {
                "features": block_output.detach().float().cpu(),
                "denoise_step": self._current_denoise_step,
                "scheduler_timestep": self._current_scheduler_timestep,
            }

        return hook

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
                f"Missing captured block outputs for blocks: {missing_blocks}"
            )
        captures = self._captures
        self._captures = OrderedDict()
        return captures

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
