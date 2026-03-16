from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from analyses.t_SNE.imagenet_utils import format_class_display_name


def _upsample_weight_map(weight_map: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    weight_tensor = torch.as_tensor(weight_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        weight_tensor,
        size=image_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def save_class_repa_heatmap_svg(
    output_path: Path,
    class_id: int,
    class_name: str,
    sample_records: list[dict],
    analysis_steps: list[int],
    mean_column_title: str,
    alpha: float = 0.48,
) -> None:
    if not sample_records:
        raise ValueError(f"No sample records were provided for class {class_id}.")

    ordered_records = sorted(sample_records, key=lambda record: record["sample_index"])
    num_rows = len(ordered_records)
    num_cols = len(analysis_steps) + 1
    column_specs = [(denoise_step, f"Step {denoise_step}") for denoise_step in analysis_steps]
    column_specs.append(("mean", mean_column_title))
    fig_width = max(3.2, num_cols * 2.7 + 0.8)
    fig_height = max(3.2, num_rows * 2.7 + 0.4)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        dpi=180,
    )

    mappable = None
    for row_index, record in enumerate(ordered_records):
        rgb_image = np.asarray(record["final_image"], dtype=np.uint8)
        for col_index, (column_key, column_title) in enumerate(column_specs):
            axis = axes[row_index, col_index]
            axis.imshow(rgb_image)

            if column_key == "mean":
                weight_map = record["mean_weight_map"]
            else:
                weight_map = record["weight_maps"][column_key]
            heatmap = _upsample_weight_map(weight_map, rgb_image.shape[:2])
            mappable = axis.imshow(
                heatmap,
                cmap="coolwarm",
                vmin=0.0,
                vmax=1.0,
                alpha=alpha,
                interpolation="bilinear",
            )

            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

            if row_index == 0:
                axis.set_title(column_title, fontsize=11, pad=4)
            if col_index == 0:
                axis.set_ylabel(
                    f"Sample {record['sample_index'] + 1}",
                    fontsize=11,
                    rotation=90,
                    labelpad=10,
                )

        if num_cols == 1:
            axes[row_index, 0].set_anchor("W")

    fig.suptitle(format_class_display_name(class_name, class_id), fontsize=15, y=0.992)
    fig.subplots_adjust(left=0.05, right=0.90, top=0.93, bottom=0.04, wspace=0.02, hspace=0.02)

    colorbar_axis = fig.add_axes([0.915, 0.12, 0.015, 0.72])
    colorbar = fig.colorbar(mappable, cax=colorbar_axis)
    colorbar.set_label("Sigmoid token weight", fontsize=10)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
