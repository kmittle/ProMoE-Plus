from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn


def _resolve_repa_weight_components(model) -> tuple[int, nn.Module]:
    if not hasattr(model, "blocks"):
        raise RuntimeError("The model does not expose transformer blocks.")
    if not hasattr(model, "encoder_depth") or model.encoder_depth is None:
        raise RuntimeError("The model does not expose encoder_depth for dynamic REPA.")
    if not hasattr(model, "repa_token_weighter") or model.repa_token_weighter is None:
        raise RuntimeError("The model does not expose repa_token_weighter.")

    block_index = int(model.encoder_depth) - 1
    if block_index < 0 or block_index >= len(model.blocks):
        raise RuntimeError(
            f"encoder_depth={model.encoder_depth} resolves to invalid block index {block_index}."
        )

    token_weighter = model.repa_token_weighter
    if isinstance(token_weighter, nn.Sequential) and len(token_weighter) > 0:
        if not isinstance(token_weighter[-1], nn.Sigmoid):
            raise RuntimeError(
                "repa_token_weighter does not end with Sigmoid; "
                "this script is for sigmoid-based dynamic REPA models."
            )

    return block_index, token_weighter


class DynamicRepaWeightCapture:
    """Capture raw MLP+Sigmoid token weights at the dynamic REPA encoder block."""

    def __init__(self, model):
        self.model = model
        self.block_index, self.token_weighter = _resolve_repa_weight_components(model)
        self.encoder_depth = int(model.encoder_depth)

        self._enabled = False
        self._current_denoise_step = None
        self._current_scheduler_timestep = None
        self._capture = OrderedDict()
        self._handle = model.blocks[self.block_index].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self._enabled:
            return

        if isinstance(output, tuple):
            block_output = output[0]
        else:
            block_output = output

        batch_size, token_count, hidden_dim = block_output.shape
        grid_size = int(round(math.sqrt(token_count)))
        if grid_size * grid_size != token_count:
            raise RuntimeError(
                f"Token count {token_count} is not a perfect square, cannot form a heatmap grid."
            )

        with torch.no_grad():
            flat_tokens = block_output.reshape(batch_size * token_count, hidden_dim)
            raw_weights = self.token_weighter(flat_tokens).reshape(batch_size, token_count)

        self._capture = OrderedDict(
            weights=raw_weights.detach().float().cpu(),
            denoise_step=self._current_denoise_step,
            scheduler_timestep=self._current_scheduler_timestep,
            grid_size=grid_size,
        )

    def enable(self, denoise_step: int, scheduler_timestep: float) -> None:
        self._enabled = True
        self._current_denoise_step = denoise_step
        self._current_scheduler_timestep = scheduler_timestep
        self._capture = OrderedDict()

    def disable_and_collect(self) -> OrderedDict:
        self._enabled = False
        if not self._capture:
            raise RuntimeError(
                f"Missing captured dynamic REPA weights for block {self.block_index}."
            )
        capture = self._capture
        self._capture = OrderedDict()
        return capture

    def close(self) -> None:
        self._handle.remove()
