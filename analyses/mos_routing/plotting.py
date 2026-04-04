from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyses.mos_routing.aggregate import BlockRoutingStats


def _save_fig(fig, path: Path):
    """Save as SVG + PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight")
    png_path = path.with_suffix(".png")
    fig.savefig(str(png_path), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_per_block_histograms(
    stats: Dict[int, BlockRoutingStats],
    output_dir: Path,
    align_blocks: List[int],
) -> List[Path]:
    """
    Plot per-block teacher block selection histograms (Chart 1).
    One subplot per aligned block, all in a single figure.
    Returns list of saved file paths.
    """
    block_stats = [stats[b] for b in align_blocks if b in stats]
    if not block_stats:
        return []

    num_blocks = len(block_stats)
    m = len(block_stats[0].top1_freq)
    x = np.arange(m)

    fig, axes = plt.subplots(1, num_blocks, figsize=(5 * num_blocks, 4), squeeze=False)
    for i, bs in enumerate(block_stats):
        ax = axes[0, i]
        ax.bar(x, bs.top1_freq, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Teacher Block Index")
        ax.set_ylabel("Top-1 Selection Frequency")
        ax.set_title(f"Gen Block {bs.block_idx}")
        ax.set_xticks(x)
        ax.set_xlim(-0.5, m - 0.5)
        ax.set_ylim(0, None)

    fig.suptitle("Per-Block Teacher Block Selection Frequency", fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = output_dir / "per_block_hist.svg"
    _save_fig(fig, out_path)
    return [out_path]


def plot_all_blocks_histogram(
    stats: Dict[int, BlockRoutingStats],
    output_dir: Path,
) -> List[Path]:
    """
    Plot all-blocks aggregated teacher block selection histogram (Chart 2).
    Uses the aggregated stats stored under key -1.
    """
    agg = stats.get(-1)
    if agg is None:
        return []

    m = len(agg.top1_freq)
    x = np.arange(m)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, agg.top1_freq, color="coral", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Teacher Block Index")
    ax.set_ylabel("Top-1 Selection Frequency")
    ax.set_title("All Blocks Aggregated — Teacher Block Selection Frequency")
    ax.set_xticks(x)
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(0, None)

    fig.tight_layout()
    out_path = output_dir / "all_blocks_hist.svg"
    _save_fig(fig, out_path)
    return [out_path]


def plot_per_block_hist_by_timestep(
    timestep_stats: Dict[int, Dict[int, BlockRoutingStats]],
    output_dir: Path,
    align_blocks: List[int],
    analysis_steps: List[int],
) -> List[Path]:
    """
    Plot per-block histogram x timestep small multiples (Chart 3).
    Rows = aligned generation blocks, Columns = denoising timestep points.
    """
    if not timestep_stats or not align_blocks or not analysis_steps:
        return []

    num_rows = len(align_blocks)
    num_cols = len(analysis_steps)

    # Determine m from first available stats
    first_step = next(iter(timestep_stats))
    first_block = next(b for b in align_blocks if b in timestep_stats[first_step])
    m = len(timestep_stats[first_step][first_block].top1_freq)
    x = np.arange(m)

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(3 * num_cols, 3 * num_rows),
        squeeze=False,
        sharey=True,
    )

    for row, block_idx in enumerate(align_blocks):
        for col, step in enumerate(analysis_steps):
            ax = axes[row, col]
            step_data = timestep_stats.get(step, {})
            bs = step_data.get(block_idx)
            if bs is not None:
                ax.bar(x, bs.top1_freq, color="steelblue", edgecolor="white", linewidth=0.3)
            ax.set_xlim(-0.5, m - 0.5)
            ax.set_xticks(x)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(f"Step {step}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Block {block_idx}", fontsize=9)
            if row == num_rows - 1:
                ax.set_xlabel("Teacher Block", fontsize=8)

    fig.suptitle("Teacher Block Selection Frequency by Denoising Step", fontsize=13, y=1.02)
    fig.tight_layout()

    out_path = output_dir / "per_block_hist_by_timestep.svg"
    _save_fig(fig, out_path)
    return [out_path]


def plot_token_variance(
    stats: Dict[int, BlockRoutingStats],
    output_dir: Path,
    align_blocks: List[int],
) -> List[Path]:
    """Plot token variance bar chart (Chart 5)."""
    block_stats = [stats[b] for b in align_blocks if b in stats]
    if not block_stats:
        return []

    labels = [f"Block {bs.block_idx}" for bs in block_stats]
    variances = [bs.token_variance for bs in block_stats]

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.5), 4))
    ax.bar(labels, variances, color="mediumpurple", edgecolor="white")
    ax.set_ylabel("Mean Token Variance")
    ax.set_title("Token-wise Routing Weight Variance per Block")
    fig.tight_layout()

    out_path = output_dir / "token_variance.svg"
    _save_fig(fig, out_path)
    return [out_path]


def plot_routing_entropy(
    stats: Dict[int, BlockRoutingStats],
    output_dir: Path,
    align_blocks: List[int],
) -> List[Path]:
    """Plot routing entropy bar chart (Chart 6)."""
    block_stats = [stats[b] for b in align_blocks if b in stats]
    if not block_stats:
        return []

    labels = [f"Block {bs.block_idx}" for bs in block_stats]
    entropies = [bs.entropy for bs in block_stats]

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.5), 4))
    ax.bar(labels, entropies, color="seagreen", edgecolor="white")
    ax.set_ylabel("Mean Routing Entropy (nats)")
    ax.set_title("Routing Entropy per Block")
    fig.tight_layout()

    out_path = output_dir / "routing_entropy.svg"
    _save_fig(fig, out_path)
    return [out_path]
