"""Visualization utilities for expert activation frequency bar charts."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_expert_frequencies(frequencies, save_dir, model_name=""):
    """Plot expert activation frequency bar charts.

    Args:
        frequencies: dict from ExpertActivationTracker.get_frequencies().
            block_idx -> list of frequencies (length = num_experts).
        save_dir: Directory to save the plots.
        model_name: Model name for plot titles.

    Returns:
        list of saved file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    block_indices = sorted(frequencies.keys())
    if not block_indices:
        print("[Warning] No MoE blocks found, skipping frequency plots.")
        return saved_paths

    num_experts = len(frequencies[block_indices[0]])

    # Plot per-block bar charts
    for block_idx in block_indices:
        freq = frequencies[block_idx]
        fig_path = _plot_single_bar(
            freq,
            title=f"Expert Activation Frequency - Block {block_idx}",
            save_path=os.path.join(save_dir, f"expert_freq_block_{block_idx}.png"),
        )
        saved_paths.append(fig_path)

    # Compute and plot averaged frequency across all blocks
    avg_freq = [0.0] * num_experts
    for block_idx in block_indices:
        for e in range(num_experts):
            avg_freq[e] += frequencies[block_idx][e]
    avg_freq = [f / len(block_indices) for f in avg_freq]

    fig_path = _plot_single_bar(
        avg_freq,
        title=f"Expert Activation Frequency - Average Across All Blocks",
        save_path=os.path.join(save_dir, "expert_freq_average.png"),
    )
    saved_paths.append(fig_path)

    return saved_paths


def _plot_single_bar(freq_list, title, save_path):
    """Plot a single bar chart with frequency values on top of each bar.

    Args:
        freq_list: List of frequencies, one per expert.
        title: Plot title.
        save_path: Path to save the figure.

    Returns:
        save_path.
    """
    num_experts = len(freq_list)
    x = np.arange(num_experts)

    fig, ax = plt.subplots(figsize=(max(8, num_experts * 0.6), 5))
    bars = ax.bar(x, freq_list, color="steelblue", edgecolor="black", linewidth=0.5)

    # Add frequency values on top of bars
    for bar, freq in zip(bars, freq_list):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{freq:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(num_experts)])
    ax.set_ylim(0, max(freq_list) * 1.15 if max(freq_list) > 0 else 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path
