from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_analysis_output_dir_from_image_dir,
    resolve_checkpoint_from_image_dir,
    resolve_config_from_checkpoint,
    resolve_config_from_image_dir,
    resolve_repo_root,
    resolve_visible_gpu_ids,
    sanitize_config_for_yaml,
)
from analyses.t_SNE.generated_images import (
    build_image_records,
    build_imagewise_output_stem,
    deserialize_generated_image_records,
    index_generated_images_by_class,
    normalize_generated_image_dir,
    sample_eligible_class_ids,
    serialize_generated_image_records,
)
from analyses.t_SNE.image_encoders import ensure_image_encoder_ready, load_image_encoder
from analyses.t_SNE.imagenet_utils import get_imagenet_class_names
from analyses.t_SNE.imagewise import (
    encode_image_records,
    load_partial_imagewise_result,
    merge_imagewise_partials,
    save_partial_imagewise_result,
)
from analyses.t_SNE.plotting import save_imagewise_tsne_svg
from utils import str_to_int_list


def setup_logging(rank: int) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(asctime)s-%(levelname)s-rank%(rank)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record

    logging.setLogRecordFactory(record_factory)


def _resolve_ckpt_and_config(args, image_dir: Path, repo_root: Path):
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
        config_path = resolve_config_from_checkpoint(ckpt_path, repo_root=repo_root)
        inferred_ckpt_path = resolve_checkpoint_from_image_dir(image_dir)
        if inferred_ckpt_path != ckpt_path:
            raise ValueError(
                f"--ckpt ({ckpt_path}) is inconsistent with --image-dir ({image_dir}). "
                f"The image directory implies checkpoint {inferred_ckpt_path}."
            )
    else:
        ckpt_path = resolve_checkpoint_from_image_dir(image_dir)
        config_path = resolve_config_from_image_dir(image_dir, repo_root=repo_root)
    return ckpt_path, config_path


def _prepare_runtime_arguments(args):
    repo_root = resolve_repo_root()
    image_dir = normalize_generated_image_dir(Path(args.image_dir))
    ckpt_path, config_path = _resolve_ckpt_and_config(args, image_dir, repo_root)
    runtime_cfg = load_runtime_cfg(config_path)
    output_dir = resolve_analysis_output_dir_from_image_dir(image_dir, analysis_name="image-wise")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = get_imagenet_class_names()
    image_index = index_generated_images_by_class(image_dir)
    if args.class_ids is not None:
        selected_class_ids = sorted(set(args.class_ids))
        for class_id in selected_class_ids:
            if class_id < 0 or class_id >= runtime_cfg.num_classes:
                raise ValueError(
                    f"Class ID {class_id} is out of range [0, {runtime_cfg.num_classes - 1}]."
                )
    else:
        selected_class_ids = sample_eligible_class_ids(
            image_index=image_index,
            num_classes=args.num_classes,
            samples_per_class=args.images_per_class,
            seed=args.seed,
        )

    image_records = build_image_records(
        image_index=image_index,
        class_ids=selected_class_ids,
        class_names=class_names,
        images_per_class=args.images_per_class,
        seed=args.seed,
    )

    visible_gpu_ids = resolve_visible_gpu_ids(runtime_cfg)
    if not visible_gpu_ids:
        raise RuntimeError("No GPU IDs were resolved from YAML or environment.")

    output_stem = build_imagewise_output_stem(
        class_ids=selected_class_ids,
        encoder_name=args.encoder_name,
        seed=args.seed,
        images_per_class=args.images_per_class,
    )
    output_svg_path = output_dir / f"{output_stem}.svg"
    partial_dir = output_dir / f".partials_{output_stem}"

    return {
        "image_dir": str(image_dir),
        "ckpt_path": str(ckpt_path),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "output_svg_path": str(output_svg_path),
        "partial_dir": str(partial_dir),
        "class_names": class_names,
        "selected_class_ids": selected_class_ids,
        "visible_gpu_ids": visible_gpu_ids,
        "runtime_cfg": runtime_cfg,
        "image_records": image_records,
        "output_stem": output_stem,
        "encoder_name": args.encoder_name,
    }


def _write_metadata(runtime_args, args) -> None:
    metadata_path = Path(runtime_args["output_dir"]) / f"{runtime_args['output_stem']}_metadata.yaml"
    ckpt_path = Path(runtime_args["ckpt_path"])
    metadata = {
        "source_image_dir": runtime_args["image_dir"],
        "checkpoint_path": str(ckpt_path),
        "checkpoint_exists": ckpt_path.exists(),
        "checkpoint_step": parse_checkpoint_step(ckpt_path),
        "config_path": runtime_args["config_path"],
        "analysis_output_dir": runtime_args["output_dir"],
        "analysis_svg_path": runtime_args["output_svg_path"],
        "seed": args.seed,
        "encoder_name": args.encoder_name,
        "num_selected_classes": len(runtime_args["selected_class_ids"]),
        "selected_class_ids": runtime_args["selected_class_ids"],
        "images_per_class": args.images_per_class,
        "total_images": len(runtime_args["image_records"]),
        "visible_gpu_ids": runtime_args["visible_gpu_ids"],
        "image_batch_size": args.image_batch_size,
        "runtime_cfg": sanitize_config_for_yaml(runtime_args["runtime_cfg"]),
    }
    with open(metadata_path, "w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False, allow_unicode=True)


def _worker(local_rank: int, worker_args: dict) -> None:
    setup_logging(local_rank)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this analysis script.")

    physical_gpu_id = worker_args["visible_gpu_ids"][local_rank]
    torch.cuda.set_device(physical_gpu_id)
    device = torch.device(f"cuda:{physical_gpu_id}")
    torch.manual_seed(worker_args["seed"])
    torch.cuda.manual_seed_all(worker_args["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    image_records = deserialize_generated_image_records(worker_args["image_records"])
    assigned_image_records = image_records[local_rank::worker_args["num_workers"]]
    if not assigned_image_records:
        logging.info("No images assigned to this worker.")
        return

    encoder_bundle = load_image_encoder(
        encoder_name=worker_args["encoder_name"],
        device=device,
    )
    logging.info(
        "Worker %d will encode %d images with %s.",
        local_rank,
        len(assigned_image_records),
        worker_args["encoder_name"],
    )

    partial_result = encode_image_records(
        encoder_bundle=encoder_bundle,
        image_records=assigned_image_records,
        batch_size=worker_args["image_batch_size"],
        device=device,
    )
    partial_path = Path(worker_args["partial_dir"]) / f"worker_{local_rank:02d}.pt"
    save_partial_imagewise_result(partial_path, partial_result)


def main():
    parser = argparse.ArgumentParser(
        description="Image-wise t-SNE using a pretrained image encoder."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Path to the generated image folder, or its parent sample folder containing images/.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, it is inferred from --image-dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for class and image selection.",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="resnet152",
        help="Pretrained image encoder used to extract vectors before the classification head.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=20,
        help="Number of ImageNet classes to sample when --class-ids is not provided.",
    )
    parser.add_argument(
        "--class-ids",
        type=str_to_int_list,
        default=None,
        help="Optional comma-separated ImageNet class IDs to analyze.",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=50,
        help="Number of generated images to encode for each selected class.",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=64,
        help="Number of images encoded together per worker.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=None,
        help="Optional manual t-SNE perplexity override.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate outputs even if the target SVG already exists.",
    )
    args = parser.parse_args()

    if args.images_per_class <= 0:
        raise ValueError("--images-per-class must be positive.")
    if args.image_batch_size <= 0:
        raise ValueError("--image-batch-size must be positive.")

    runtime_args = _prepare_runtime_arguments(args)
    output_svg_path = Path(runtime_args["output_svg_path"])
    if output_svg_path.exists() and not args.overwrite:
        print(f"Skip because output already exists: {output_svg_path}")
        return

    ensure_image_encoder_ready(args.encoder_name)

    partial_dir = Path(runtime_args["partial_dir"])
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(runtime_args, args)

    num_workers = len(runtime_args["visible_gpu_ids"])
    worker_args = {
        "visible_gpu_ids": runtime_args["visible_gpu_ids"],
        "num_workers": num_workers,
        "image_records": serialize_generated_image_records(runtime_args["image_records"]),
        "seed": args.seed,
        "encoder_name": args.encoder_name,
        "image_batch_size": args.image_batch_size,
        "partial_dir": runtime_args["partial_dir"],
    }

    if num_workers == 1:
        _worker(0, worker_args)
    else:
        mp.spawn(_worker, nprocs=num_workers, args=(worker_args,))

    partial_paths = sorted(partial_dir.glob("worker_*.pt"))
    partials = [load_partial_imagewise_result(partial_path) for partial_path in partial_paths]
    merged = merge_imagewise_partials(partials)
    save_imagewise_tsne_svg(
        output_path=output_svg_path,
        merged_record=merged,
        selected_class_ids=runtime_args["selected_class_ids"],
        class_names=runtime_args["class_names"],
        seed=args.seed,
        images_per_class=args.images_per_class,
        encoder_name=args.encoder_name,
        perplexity=args.perplexity,
    )
    shutil.rmtree(partial_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
