import random
import re


def get_imagenet_class_names() -> list[str]:
    try:
        from torchvision.models import ResNet50_Weights

        return list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
    except Exception:
        try:
            from torchvision.models import ResNet50_Weights

            return list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])
        except Exception:
            return [f"class_{class_id:03d}" for class_id in range(1000)]


def sample_class_ids(num_classes: int, seed: int, total_classes: int = 1000) -> list[int]:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if num_classes > total_classes:
        raise ValueError(
            f"Cannot sample {num_classes} classes from {total_classes} ImageNet classes."
        )
    rng = random.Random(seed)
    return sorted(rng.sample(range(total_classes), num_classes))


def slugify_class_name(class_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", class_name.strip().lower()).strip("_")
    return slug or "unnamed_class"
