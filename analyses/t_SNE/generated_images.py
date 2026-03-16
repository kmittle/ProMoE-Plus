from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import re

from analyses.t_SNE.imagenet_utils import build_class_id_signature


_CLASS_PATTERN = re.compile(r"_class(\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class GeneratedImageRecord:
    image_id: int
    class_id: int
    class_name: str
    image_path: str


def normalize_generated_image_dir(image_dir: Path) -> Path:
    image_dir = image_dir.resolve()
    if image_dir.is_dir() and image_dir.name == "images":
        return image_dir
    candidate = image_dir / "images"
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find generated image directory from {image_dir}. "
        "Expected either the images directory itself or a folder containing an images/ child."
    )


def parse_class_id_from_image_path(image_path: Path) -> int:
    match = _CLASS_PATTERN.search(image_path.name)
    if match is None:
        raise ValueError(
            f"Cannot parse ImageNet class id from generated image filename: {image_path.name}"
        )
    return int(match.group(1))


def index_generated_images_by_class(image_dir: Path) -> dict[int, list[Path]]:
    image_dir = normalize_generated_image_dir(image_dir)
    class_to_paths = defaultdict(list)
    for image_path in sorted(image_dir.glob("*.png")):
        class_id = parse_class_id_from_image_path(image_path)
        class_to_paths[class_id].append(image_path)
    return dict(class_to_paths)


def sample_eligible_class_ids(
    image_index: dict[int, list[Path]],
    num_classes: int,
    samples_per_class: int,
    seed: int,
) -> list[int]:
    eligible_class_ids = sorted(
        class_id
        for class_id, paths in image_index.items()
        if len(paths) >= samples_per_class
    )
    if len(eligible_class_ids) < num_classes:
        raise ValueError(
            f"Only {len(eligible_class_ids)} classes have at least {samples_per_class} generated images, "
            f"but {num_classes} classes were requested."
        )
    rng = random.Random(seed)
    return sorted(rng.sample(eligible_class_ids, num_classes))


def build_image_records(
    image_index: dict[int, list[Path]],
    class_ids: list[int],
    class_names: list[str],
    images_per_class: int,
    seed: int,
) -> list[GeneratedImageRecord]:
    records = []
    image_id = 0
    for class_id in class_ids:
        paths = list(image_index.get(class_id, []))
        if len(paths) < images_per_class:
            raise ValueError(
                f"Class {class_id} only has {len(paths)} generated images, "
                f"but {images_per_class} were requested."
            )
        if len(paths) == images_per_class:
            selected_paths = paths
        else:
            rng = random.Random(seed + class_id)
            selected_paths = sorted(rng.sample(paths, images_per_class))
        class_name = class_names[class_id]
        for image_path in selected_paths:
            records.append(
                GeneratedImageRecord(
                    image_id=image_id,
                    class_id=class_id,
                    class_name=class_name,
                    image_path=str(image_path),
                )
            )
            image_id += 1
    return records


def serialize_generated_image_records(
    records: list[GeneratedImageRecord],
) -> list[dict]:
    return [asdict(record) for record in records]


def deserialize_generated_image_records(
    records: list[dict],
) -> list[GeneratedImageRecord]:
    return [GeneratedImageRecord(**record) for record in records]


def build_imagewise_output_stem(
    class_ids: list[int],
    encoder_name: str,
    seed: int,
    images_per_class: int,
) -> str:
    class_signature = build_class_id_signature(class_ids)
    encoder_slug = re.sub(r"[^a-zA-Z0-9]+", "_", encoder_name.strip().lower()).strip("_")
    return (
        f"imagewise_encoder-{encoder_slug}_seed{seed}"
        f"_cls{len(class_ids)}_{class_signature}_ipc{images_per_class}"
    )
