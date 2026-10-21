"""Benchmark-only dispatch controls; experiment model defaults are unchanged."""

from types import MethodType

import torch
import torch.nn.functional as F

from models.models_ProMoE_TC_image_shared import ImageWiseSharedExpertPool


def _balanced_router(pool, image_embeddings):
    """Keep cosine-score gradients while fixing each expert's number of images."""
    score_dtype = (torch.float32 if image_embeddings.dtype in (torch.float16, torch.bfloat16)
                   else image_embeddings.dtype)
    with torch.autocast(device_type=image_embeddings.device.type, enabled=False):
        images = F.normalize(image_embeddings.to(score_dtype), dim=-1)
        centers = F.normalize(pool.cluster_centers.to(score_dtype), dim=-1)
        scores = images @ centers.T
        indices = torch.arange(len(images), device=images.device) % pool.num_image_experts
        selected = scores.gather(1, indices[:, None]).squeeze(1)
    return selected, indices


def set_benchmark_routing(model, mode):
    if mode == "native":
        return
    if mode != "balanced":
        raise ValueError("benchmark routing must be native or balanced")
    for module in model.modules():
        if isinstance(module, ImageWiseSharedExpertPool):
            module.compute_router = MethodType(_balanced_router, module)
