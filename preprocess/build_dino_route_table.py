"""Build a class-level DINO semantic-confidence table for route calibration.

The table is deliberately smaller than a feature cache: training only needs one
detached uncertainty value per ImageNet class.  DINO features never enter the
DiT representation or a representation-alignment loss.

This builder emits the corrected version-2 contract and refuses to reuse an
existing NPZ or sidecar path.  A consumer config must declare both
``table_version: 2`` and the emitted ``method``; undeclared historical configs
remain restricted to the legacy version-1 contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

# Direct execution puts ``preprocess/`` on sys.path, so add the repository
# root explicitly before importing project packages.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repa.encoder import extract_teacher_features, load_teacher_encoder
from preprocess.dino_route_table_contract import (
    CORRECTED_TABLE_METHOD,
    CORRECTED_TABLE_VERSION,
)
from utils import load_vae


NUM_CLASSES = 1000
LATENT_SHAPE = (8, 32, 32)
LATENT_KEY = "latent"
IMAGE_SIZE = 256
LATENT_SCALE = 0.18215
TABLE_VERSION = CORRECTED_TABLE_VERSION
TABLE_METHOD = CORRECTED_TABLE_METHOD


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _require_new_output_pair(output_path: Path) -> Path:
    if output_path.suffix != ".npz":
        raise ValueError("output path must end with .npz")
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    existing = [
        path for path in (output_path, metadata_path) if _path_exists(path)
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing DINO route-table artifact: "
            + ", ".join(str(path) for path in existing)
        )
    return metadata_path


def _prepare_new_output_pair(output_path: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_path).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    # abspath normalizes the spelling without dereferencing the final path.
    # Path.resolve() would hide a dangling symlink and redirect publication.
    output_path = Path(os.path.abspath(output_path))
    return output_path, _require_new_output_pair(output_path)


def _publish_output_pair_no_replace(
    npz_tmp: Path,
    json_tmp: Path,
    output_path: Path,
    metadata_path: Path,
) -> None:
    """Publish two same-filesystem temporary files without replacing targets."""
    published: list[tuple[Path, Path]] = []
    try:
        for temporary, destination in (
            (npz_tmp, output_path),
            (json_tmp, metadata_path),
        ):
            os.link(temporary, destination)
            published.append((temporary, destination))
    except OSError:
        for temporary, destination in reversed(published):
            try:
                if destination.exists() and os.path.samefile(
                    temporary, destination
                ):
                    destination.unlink()
            except OSError:
                pass
        raise


def _class_files(
    latent_root: Path,
    samples_per_class: int,
) -> tuple[list[str], list[int], list[Path]]:
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2")
    if latent_root.is_symlink() or not latent_root.is_dir():
        raise ValueError(f"latent root is not a real directory: {latent_root}")
    class_dirs = sorted(
        entry for entry in latent_root.iterdir()
        if entry.is_dir() and not entry.is_symlink()
    )
    if len(class_dirs) != NUM_CLASSES:
        raise ValueError(
            f"expected {NUM_CLASSES} class directories, found {len(class_dirs)}"
        )
    class_names: list[str] = []
    class_ids: list[int] = []
    selected: list[Path] = []
    numeric_layout = all(entry.name.isdigit() for entry in class_dirs)
    if numeric_layout:
        class_ids = [int(entry.name) for entry in class_dirs]
        if (
            len(set(class_ids)) != len(class_ids)
            or any(class_id < 0 or class_id >= NUM_CLASSES for class_id in class_ids)
        ):
            raise ValueError("numeric class directories must map into [0, 999]")
    elif all(
        len(entry.name) == 9
        and entry.name.startswith("n")
        and entry.name[1:].isdigit()
        for entry in class_dirs
    ):
        # ImageFolder uses the lexicographic synset order as the class index.
        class_ids = list(range(len(class_dirs)))
    else:
        raise ValueError(
            "class directories must be numeric labels or ImageNet synsets"
        )
    for class_dir, class_id in zip(class_dirs, class_ids):
        files = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and not path.is_symlink() and path.name.endswith(".latent.npz")
        )
        if len(files) < samples_per_class:
            raise ValueError(
                f"{class_dir} has {len(files)} latents; "
                f"need {samples_per_class}"
            )
        class_names.append(class_dir.name)
        selected.extend(files[:samples_per_class])
    return class_names, class_ids, selected


def _load_latent_batch(paths: list[Path], device: torch.device) -> torch.Tensor:
    parameters = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if LATENT_KEY not in archive.files:
                raise ValueError(f"{path} does not contain {LATENT_KEY!r}")
            value = np.asarray(archive[LATENT_KEY], dtype=np.float32)
        if value.shape != LATENT_SHAPE:
            raise ValueError(f"{path} has shape {value.shape}; expected {LATENT_SHAPE}")
        if not np.isfinite(value).all():
            raise ValueError(f"{path} contains non-finite values")
        parameters.append(value)
    return torch.from_numpy(np.stack(parameters)).to(device)


def _minmax(values: torch.Tensor) -> torch.Tensor:
    lo = values.min()
    hi = values.max()
    span = hi - lo
    if float(span) <= 1e-12:
        return torch.zeros_like(values)
    return (values - lo) / span


def _decode_model_latents(vae, model_latents: torch.Tensor) -> torch.Tensor:
    """Decode latents expressed in the diffusion model's scaled space."""
    if model_latents.ndim != 4:
        raise ValueError(
            "model_latents must have shape [batch, channels, height, width]"
        )
    return vae.decode(model_latents / LATENT_SCALE).sample


def build_table(
    latent_root: str | Path,
    output_path: str | Path,
    *,
    dino_path: str | Path,
    vae_path: str | Path,
    samples_per_class: int = 8,
    batch_size: int = 16,
    device: str = "cuda:0",
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    latent_root = Path(latent_root).expanduser()
    if latent_root.is_symlink():
        raise ValueError(f"latent root must not be a symlink: {latent_root}")
    latent_root = latent_root.resolve(strict=True)
    output_path, metadata_path = _prepare_new_output_pair(output_path)
    dino_path = Path(dino_path).resolve(strict=True)
    vae_path = Path(vae_path).resolve(strict=True)
    vae_config_path = (vae_path / "config.json").resolve(strict=True)
    vae_weights_path = (
        vae_path / "diffusion_pytorch_model.safetensors"
    ).resolve(strict=True)
    class_names, class_id_list, paths = _class_files(
        latent_root, samples_per_class
    )
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DINO route-table extraction")

    vae = load_vae(
        "stabilityai/sd-vae-ft-mse",
        vae_path=str(vae_path),
    ).to(torch_device).eval()
    encoder, embed_dim = load_teacher_encoder(
        "dinov2-vit-b",
        resolution=IMAGE_SIZE,
        local_root=str(dino_path.parent.parent),
        enc_path=str(dino_path),
    )
    encoder = encoder.to(torch_device).eval()
    for module in (vae, encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    per_image_features = []
    per_image_classes = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            parameters = _load_latent_batch(batch_paths, torch_device)
            # Match the training/sampling VAE convention before decoding.  The
            # latent files store unscaled distribution parameters.
            model_latents = (
                DiagonalGaussianDistribution(parameters).mode() * LATENT_SCALE
            )
            # The diffusion model consumes scaled latents.  AutoencoderKL.decode
            # consumes the original VAE latent space, as in sample.py.
            decoded = _decode_model_latents(vae, model_latents)
            images = (decoded.float().clamp(-1, 1) + 1.0) / 2.0
            patch_features = extract_teacher_features(
                encoder, images, "dinov2-vit-b"
            )
            image_features = F.normalize(patch_features.float().mean(dim=1), dim=-1)
            per_image_features.append(image_features.cpu())
            per_image_classes.extend(
                class_id_list[index // samples_per_class]
                for index in range(start, start + len(batch_paths))
            )
            if (start // batch_size) % 25 == 0:
                print(f"processed {start + len(batch_paths)}/{len(paths)} latents", flush=True)

    image_features = torch.cat(per_image_features, dim=0)
    class_ids_tensor = torch.tensor(per_image_classes, dtype=torch.long)
    class_embeddings = []
    intra_variance = []
    for class_id in range(NUM_CLASSES):
        members = image_features[class_ids_tensor == class_id]
        centroid = F.normalize(members.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        class_embeddings.append(centroid)
        intra_variance.append(1.0 - (members @ centroid).mean())
    class_embeddings = torch.stack(class_embeddings)
    intra_variance = torch.stack(intra_variance)

    similarities = class_embeddings @ class_embeddings.T
    similarities.fill_diagonal_(-torch.inf)
    nearest = torch.topk(similarities, k=2, dim=1).values
    nearest_margin = nearest[:, 0] - nearest[:, 1]
    uncertainty = 0.5 * (1.0 - _minmax(nearest_margin)) + 0.5 * _minmax(intra_variance)
    uncertainty = uncertainty.clamp(0.0, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": TABLE_VERSION,
        "method": TABLE_METHOD,
        "num_classes": NUM_CLASSES,
        "class_names": class_names,
        # Keep metadata JSON-serializable; the tensor above is only for
        # indexing the per-image feature matrix.
        "class_ids": class_id_list,
        "samples_per_class": samples_per_class,
        "latent_root": str(latent_root),
        "latent_key": LATENT_KEY,
        "latent_shape": list(LATENT_SHAPE),
        "latent_scale_for_model": LATENT_SCALE,
        "vae_decode_rule": "vae.decode(model_latents / latent_scale_for_model)",
        "dino_weights": str(dino_path),
        "dino_weights_sha256": _sha256_file(dino_path),
        "vae_path": str(vae_path),
        "vae_config": str(vae_config_path),
        "vae_config_sha256": _sha256_file(vae_config_path),
        "vae_weights": str(vae_weights_path),
        "vae_weights_sha256": _sha256_file(vae_weights_path),
        "embed_dim": int(embed_dim),
        "device": str(torch_device),
    }
    # np.savez_compressed appends .npz to path strings.  Keep that suffix in
    # the temporary name so the no-replace publisher sees the written file.
    npz_tmp = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp.npz"
    )
    json_tmp = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp.json"
    )
    try:
        np.savez_compressed(
            npz_tmp,
            class_embeddings=class_embeddings.numpy().astype(np.float32),
            uncertainty=uncertainty.numpy().astype(np.float32),
            nearest_margin=nearest_margin.numpy().astype(np.float32),
            intra_variance=intra_variance.numpy().astype(np.float32),
        )
        metadata["table_sha256"] = _sha256_file(npz_tmp)
        with json_tmp.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _publish_output_pair_no_replace(
            npz_tmp,
            json_tmp,
            output_path,
            metadata_path,
        )
    finally:
        npz_tmp.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                **metadata,
                "output": str(output_path),
                "uncertainty_min": float(uncertainty.min()),
                "uncertainty_max": float(uncertainty.max()),
                "uncertainty_mean": float(uncertainty.mean()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dino-path",
        type=Path,
        default=Path("pretrained_ckpt/encoder/dinov2_vitb14/state_dict.pth"),
    )
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=Path("pretrained_ckpt/vae/stabilityai--sd-vae-ft-mse"),
    )
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    build_table(
        args.latent_root,
        args.output,
        dino_path=args.dino_path,
        vae_path=args.vae_path,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
        device=args.device,
    )
