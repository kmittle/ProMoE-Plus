from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


@dataclass
class BlockRoutingStats:
    """Per-generation-block routing statistics."""
    block_idx: int
    mean_weights: np.ndarray       # (m,) average routing weight per teacher block
    top1_freq: np.ndarray          # (m,) frequency of being top-1 selected
    topk_freq: np.ndarray          # (m,) frequency of being in top-k selected
    entropy: float                 # mean routing entropy over all tokens
    token_variance: float          # mean variance of weights across tokens (T dim) per sample
    num_tokens: int                # total tokens accumulated


class _BlockAccumulator:
    """Online accumulator for a single generation block."""

    def __init__(self, num_teacher_blocks: int, top_k: int):
        self.m = num_teacher_blocks
        self.top_k = top_k
        self._weight_sum = np.zeros(self.m, dtype=np.float64)
        self._top1_count = np.zeros(self.m, dtype=np.float64)
        self._topk_count = np.zeros(self.m, dtype=np.float64)
        self._entropy_sum = 0.0
        self._token_var_sum = 0.0
        self._num_tokens = 0
        self._num_samples = 0  # number of (N,) images for variance

    def update(self, weights: torch.Tensor):
        """
        Update with routing weights for one batch.

        Args:
            weights: (N, T, m) softmax-normalized routing weights
        """
        N, T, m = weights.shape
        assert m == self.m

        w = weights.numpy() if isinstance(weights, torch.Tensor) else weights
        flat = w.reshape(-1, m)  # (N*T, m)
        num_tokens = flat.shape[0]

        # Mean weights
        self._weight_sum += flat.sum(axis=0)

        # Top-1 frequency (vectorized)
        top1 = flat.argmax(axis=1)  # (N*T,)
        self._top1_count += np.bincount(top1, minlength=m).astype(np.float64)

        # Top-k frequency (vectorized)
        k = min(self.top_k, m)
        if k >= m:
            # All teacher blocks are in top-k; every token contributes to every block
            self._topk_count += num_tokens
        else:
            topk_indices = np.argpartition(-flat, k, axis=1)[:, :k]  # (N*T, k)
            self._topk_count += np.bincount(topk_indices.ravel(), minlength=m).astype(np.float64)

        # Entropy: -sum(p * log(p))
        eps = 1e-10
        log_w = np.log(flat + eps)
        token_entropy = -(flat * log_w).sum(axis=1)  # (N*T,)
        self._entropy_sum += token_entropy.sum()

        # Token variance: per sample, compute variance across T, then average
        for n in range(N):
            sample_w = w[n]  # (T, m)
            var_across_tokens = sample_w.var(axis=0).mean()  # scalar
            self._token_var_sum += var_across_tokens
        self._num_samples += N

        self._num_tokens += num_tokens

    def finalize(self, block_idx: int) -> BlockRoutingStats:
        n = max(self._num_tokens, 1)
        s = max(self._num_samples, 1)
        return BlockRoutingStats(
            block_idx=block_idx,
            mean_weights=self._weight_sum / n,
            top1_freq=self._top1_count / n,
            topk_freq=self._topk_count / n,
            entropy=self._entropy_sum / n,
            token_variance=self._token_var_sum / s,
            num_tokens=self._num_tokens,
        )


class OnlineRoutingAggregator:
    """
    Accumulates routing statistics online across all timesteps.
    Produces per-block and all-blocks aggregated stats.
    """

    def __init__(self, align_blocks: List[int], num_teacher_blocks: int, top_k: int = 2):
        self.align_blocks = align_blocks
        self.m = num_teacher_blocks
        self.top_k = top_k
        self._accumulators: Dict[int, _BlockAccumulator] = {
            b: _BlockAccumulator(num_teacher_blocks, top_k) for b in align_blocks
        }
        # All-blocks accumulator
        self._all_blocks_acc = _BlockAccumulator(num_teacher_blocks, top_k)

    def update(self, routing_data: Dict[int, torch.Tensor]):
        """
        Accumulate one batch of routing data.

        Args:
            routing_data: {block_idx: (N, T, m)} from MoSRoutingCapture
        """
        for block_idx, weights in routing_data.items():
            if block_idx in self._accumulators:
                self._accumulators[block_idx].update(weights)
                self._all_blocks_acc.update(weights)

    def finalize(self) -> Dict[int, BlockRoutingStats]:
        """Return {block_idx: stats} for per-block + -1 for all-blocks aggregated."""
        result = {}
        for block_idx, acc in self._accumulators.items():
            result[block_idx] = acc.finalize(block_idx)
        result[-1] = self._all_blocks_acc.finalize(-1)
        return result


class PerTimestepRoutingAggregator:
    """
    Accumulates routing statistics keyed by denoising step index.
    Each step maintains its own set of per-block accumulators.
    """

    def __init__(
        self,
        align_blocks: List[int],
        num_teacher_blocks: int,
        analysis_steps: List[int],
        top_k: int = 2,
    ):
        self.align_blocks = align_blocks
        self.m = num_teacher_blocks
        self.top_k = top_k
        self.analysis_steps = analysis_steps
        self._step_aggregators: Dict[int, OnlineRoutingAggregator] = {
            step: OnlineRoutingAggregator(align_blocks, num_teacher_blocks, top_k)
            for step in analysis_steps
        }

    def update(self, routing_data: Dict[int, torch.Tensor], step_idx: int):
        """Accumulate one batch for a specific denoising step."""
        if step_idx in self._step_aggregators:
            self._step_aggregators[step_idx].update(routing_data)

    def finalize(self) -> Dict[int, Dict[int, BlockRoutingStats]]:
        """Return {step_idx: {block_idx: stats}}."""
        return {
            step: agg.finalize()
            for step, agg in self._step_aggregators.items()
        }
