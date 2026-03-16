from __future__ import annotations

import numpy as np

from analyses.t_SNE.imagenet_utils import format_class_display_name


def _require_plot_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for token-wise t-SNE plotting. "
            "Please install it first, for example with `pip install matplotlib`."
        ) from exc

    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for token-wise t-SNE plotting. "
            "Please install it first, for example with `pip install scikit-learn`."
        ) from exc

    return plt, Line2D, TSNE


def build_categorical_palette(num_categories: int):
    plt, _, _ = _require_plot_dependencies()
    if num_categories <= 10:
        cmap = plt.get_cmap("tab10", num_categories)
    elif num_categories <= 20:
        cmap = plt.get_cmap("tab20", num_categories)
    else:
        cmap = plt.get_cmap("gist_ncar", num_categories)
    return [cmap(category_id) for category_id in range(num_categories)]


def build_expert_palette(num_experts: int):
    return build_categorical_palette(num_experts)


def run_tsne(features: np.ndarray, seed: int, perplexity: float | None = None) -> np.ndarray:
    _, _, TSNE = _require_plot_dependencies()

    features = features.astype(np.float32, copy=False)
    features = features - features.mean(axis=0, keepdims=True)
    feature_std = features.std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0
    features = features / feature_std

    num_tokens = features.shape[0]
    if num_tokens < 2:
        return np.zeros((num_tokens, 2), dtype=np.float32)

    if perplexity is None:
        perplexity = min(30.0, max(5.0, (num_tokens - 1) / 3.0))
    perplexity = min(perplexity, float(num_tokens - 1))

    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    embedding = tsne.fit_transform(features)
    return embedding.astype(np.float32, copy=False)


def save_tokenwise_tsne_svg(
    output_path,
    class_id: int,
    class_name: str,
    class_records,
    block_indices: list[int],
    analysis_steps: list[int],
    num_experts: int,
    seed: int,
    perplexity: float | None = None,
):
    plt, Line2D, _ = _require_plot_dependencies()

    n_rows = len(block_indices)
    n_cols = len(analysis_steps)
    figsize = (
        max(3.0 * n_cols, 8.0),
        max(2.6 * n_rows, 6.0),
    )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    palette = build_expert_palette(num_experts)
    present_experts = set()

    for row, block_idx in enumerate(block_indices):
        for col, denoise_step in enumerate(analysis_steps):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            block_record = class_records.get(denoise_step, {}).get(block_idx)
            if block_record is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
                continue

            embedding = run_tsne(
                block_record["features"],
                seed=seed + class_id + denoise_step + block_idx,
                perplexity=perplexity,
            )
            labels = block_record["labels"].astype(int, copy=False)
            present_experts.update(np.unique(labels).tolist())
            colors = [palette[label % len(palette)] for label in labels]

            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                s=15,
                c=colors,
                alpha=0.9,
                linewidths=0.0,
            )
            ax.margins(0.03)

            if row == 0:
                scheduler_timestep = block_record["scheduler_timestep"]
                ax.set_title(
                    f"Denoise {denoise_step}\n(t={scheduler_timestep:.1f})",
                    fontsize=10,
                    pad=5,
                )
            if col == 0:
                ax.set_ylabel(f"Block {block_idx}", fontsize=11)

    title = format_class_display_name(class_name, class_id)
    fig.suptitle(title, fontsize=16, y=0.995)

    if present_experts:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=palette[expert_id % len(palette)],
                label=f"Exp {expert_id}",
                markersize=5,
            )
            for expert_id in sorted(present_experts)
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=min(6, len(legend_handles)),
            frameon=False,
            columnspacing=0.9,
            handletextpad=0.4,
            fontsize=9,
        )

    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        top=0.93,
        bottom=0.03,
        wspace=0.06,
        hspace=0.10,
    )
    fig.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_samplewise_tsne_svg(
    output_path,
    merged_records: dict,
    block_indices: list[int],
    analysis_steps: list[int],
    scheduler_timesteps: dict,
    selected_class_ids: list[int],
    class_names: list[str],
    seed: int,
    samples_per_class: int,
    pool_type: str,
    perplexity: float | None = None,
):
    plt, Line2D, _ = _require_plot_dependencies()

    n_rows = len(block_indices)
    n_cols = len(analysis_steps)
    figsize = (
        max(3.0 * n_cols, 8.5),
        max(2.7 * n_rows, 6.5),
    )
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    sorted_class_ids = list(selected_class_ids)
    palette = build_categorical_palette(len(sorted_class_ids))
    class_to_color = {
        class_id: palette[index]
        for index, class_id in enumerate(sorted_class_ids)
    }
    class_to_name = {
        class_id: format_class_display_name(class_names[class_id], class_id)
        for class_id in sorted_class_ids
    }

    for row, block_idx in enumerate(block_indices):
        for col, denoise_step in enumerate(analysis_steps):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            block_record = merged_records.get(denoise_step, {}).get(block_idx)
            if block_record is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
                continue

            embedding = run_tsne(
                block_record["vectors"],
                seed=seed + denoise_step + block_idx,
                perplexity=perplexity,
            )
            labels = block_record["class_ids"].astype(int, copy=False)
            colors = [class_to_color[class_id] for class_id in labels]

            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                s=18,
                c=colors,
                alpha=0.9,
                linewidths=0.0,
            )
            ax.margins(0.03)

            if row == 0:
                ax.set_title(
                    f"Denoise {denoise_step}\n(t={scheduler_timesteps[denoise_step]:.1f})",
                    fontsize=10,
                    pad=5,
                )
            if col == 0:
                ax.set_ylabel(f"Block {block_idx}", fontsize=11)

    fig.suptitle(
        f"Sample-wise t-SNE | pool={pool_type} | {len(sorted_class_ids)} classes x {samples_per_class} samples",
        fontsize=15,
        y=0.995,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=class_to_color[class_id],
            label=class_to_name[class_id],
            markersize=5,
        )
        for class_id in sorted_class_ids
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(3, len(legend_handles)),
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        top=0.92,
        bottom=0.03,
        wspace=0.06,
        hspace=0.10,
    )
    fig.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_imagewise_tsne_svg(
    output_path,
    merged_record: dict,
    selected_class_ids: list[int],
    class_names: list[str],
    seed: int,
    images_per_class: int,
    encoder_name: str,
    perplexity: float | None = None,
):
    plt, Line2D, _ = _require_plot_dependencies()

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 7.4))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    sorted_class_ids = list(selected_class_ids)
    palette = build_categorical_palette(len(sorted_class_ids))
    class_to_color = {
        class_id: palette[index]
        for index, class_id in enumerate(sorted_class_ids)
    }
    class_to_name = {
        class_id: format_class_display_name(class_names[class_id], class_id)
        for class_id in sorted_class_ids
    }

    embedding = run_tsne(
        merged_record["vectors"],
        seed=seed,
        perplexity=perplexity,
    )
    labels = merged_record["class_ids"].astype(int, copy=False)
    colors = [class_to_color[class_id] for class_id in labels]

    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=18,
        c=colors,
        alpha=0.9,
        linewidths=0.0,
    )
    ax.margins(0.03)

    fig.suptitle(
        f"Image-wise t-SNE | encoder={encoder_name} | {len(sorted_class_ids)} classes x {images_per_class} images",
        fontsize=15,
        y=0.985,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=class_to_color[class_id],
            label=class_to_name[class_id],
            markersize=5,
        )
        for class_id in sorted_class_ids
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=min(3, len(legend_handles)),
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.995,
        top=0.88,
        bottom=0.03,
    )
    fig.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
