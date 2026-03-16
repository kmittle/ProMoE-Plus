from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class ImageEncoderBundle:
    name: str
    model: nn.Module
    preprocess: object
    feature_dim: int


def _get_resnet152_weights():
    try:
        return models.ResNet152_Weights.IMAGENET1K_V2
    except AttributeError:
        return models.ResNet152_Weights.IMAGENET1K_V1


def ensure_image_encoder_ready(encoder_name: str = "resnet152") -> None:
    encoder_name = encoder_name.lower()
    if encoder_name != "resnet152":
        raise ValueError(f"Unsupported encoder_name='{encoder_name}'.")
    weights = _get_resnet152_weights()
    weights.get_state_dict(progress=True)


def load_image_encoder(
    encoder_name: str = "resnet152",
    device: torch.device | None = None,
) -> ImageEncoderBundle:
    encoder_name = encoder_name.lower()
    if encoder_name != "resnet152":
        raise ValueError(f"Unsupported encoder_name='{encoder_name}'.")

    weights = _get_resnet152_weights()
    model = models.resnet152(weights=weights)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    model.eval()
    if device is not None:
        model = model.to(device)

    preprocess = weights.transforms()
    return ImageEncoderBundle(
        name=encoder_name,
        model=model,
        preprocess=preprocess,
        feature_dim=feature_dim,
    )
