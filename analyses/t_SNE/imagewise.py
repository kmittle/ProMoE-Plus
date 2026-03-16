from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def save_partial_imagewise_result(partial_path: Path, partial_result: dict) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(partial_result, partial_path)


def load_partial_imagewise_result(partial_path: Path) -> dict:
    return torch.load(partial_path, map_location="cpu", weights_only=False)


@torch.inference_mode()
def encode_image_records(
    encoder_bundle,
    image_records,
    batch_size: int,
    device,
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    features = []
    class_ids = []
    image_ids = []

    for start in range(0, len(image_records), batch_size):
        batch_records = image_records[start:start + batch_size]
        batch_tensors = []
        for record in batch_records:
            with Image.open(record.image_path) as image:
                image = image.convert("RGB")
                batch_tensors.append(encoder_bundle.preprocess(image))
        batch = torch.stack(batch_tensors, dim=0).to(device)
        batch_features = encoder_bundle.model(batch).detach().float().cpu().numpy()
        features.append(batch_features)
        class_ids.append(np.asarray([record.class_id for record in batch_records], dtype=np.int64))
        image_ids.append(np.asarray([record.image_id for record in batch_records], dtype=np.int64))

    return {
        "vectors": np.concatenate(features, axis=0),
        "class_ids": np.concatenate(class_ids, axis=0),
        "image_ids": np.concatenate(image_ids, axis=0),
    }


def merge_imagewise_partials(partials: list[dict]) -> dict:
    if not partials:
        raise ValueError("No partial image-wise results were provided.")

    vectors = np.concatenate([partial["vectors"] for partial in partials], axis=0)
    class_ids = np.concatenate([partial["class_ids"] for partial in partials], axis=0)
    image_ids = np.concatenate([partial["image_ids"] for partial in partials], axis=0)

    sort_order = np.argsort(image_ids)
    return {
        "vectors": vectors[sort_order],
        "class_ids": class_ids[sort_order],
        "image_ids": image_ids[sort_order],
    }
