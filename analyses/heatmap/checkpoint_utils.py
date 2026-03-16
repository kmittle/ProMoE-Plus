from __future__ import annotations

from pathlib import Path

from analyses.t_SNE.checkpoint_utils import (
    resolve_run_root_from_image_dir,
    resolve_step_from_image_dir,
)
from analyses.t_SNE.imagenet_utils import slugify_class_name


def resolve_heatmap_output_dir_from_image_dir(
    image_dir: Path,
    analysis_name: str = "repa_dyna",
) -> Path:
    run_root = resolve_run_root_from_image_dir(image_dir)
    step = resolve_step_from_image_dir(image_dir)
    return run_root / "sample" / f"step{step}" / "heatmap" / analysis_name


def build_class_heatmap_output_paths(
    output_dir: Path,
    output_stem: str,
    class_ids: list[int],
    class_names: list[str],
) -> dict[int, Path]:
    output_paths = {}
    for class_id in class_ids:
        class_slug = slugify_class_name(class_names[class_id])
        output_paths[class_id] = output_dir / f"{output_stem}_class{class_id:03d}_{class_slug}.svg"
    return output_paths
