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

from analyses.t_SNE.block_capture import BlockOutputCapture
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_analysis_output_dir,
    resolve_config_from_checkpoint,
    resolve_repo_root,
    resolve_visible_gpu_ids,
    sanitize_config_for_yaml,
)
from analyses.t_SNE.imagenet_utils import get_imagenet_class_names, sample_class_ids
from analyses.t_SNE.plotting import save_samplewise_tsne_svg
from analyses.t_SNE.pooling import SUPPORTED_POOL_TYPES
from analyses.t_SNE.samplewise import (
    build_sample_specs,
    build_samplewise_output_stem,
    deserialize_sample_specs,
    load_partial_samplewise_result,
    merge_samplewise_partials,
    sample_and_collect_pooled_features,
    save_partial_samplewise_result,
    serialize_sample_specs,
)
from analyses.t_SNE.sampling import build_model, chunked, compute_analysis_steps
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


def _prepare_runtime_arguments(args):
    repo_root = resolve_repo_root()
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    config_path = resolve_config_from_checkpoint(ckpt_path, repo_root=repo_root)
    runtime_cfg = load_runtime_cfg(config_path)
    output_dir = resolve_analysis_output_dir(ckpt_path, analysis_name="sample-wise")
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_steps = compute_analysis_steps(runtime_cfg.sample_steps, args.analysis_every)
    class_names = get_imagenet_class_names()
    if args.class_ids is not None:
        selected_class_ids = sorted(set(args.class_ids))
    else:
        selected_class_ids = sample_class_ids(args.num_classes, args.seed, runtime_cfg.num_classes)

    for class_id in selected_class_ids:
        if class_id < 0 or class_id >= runtime_cfg.num_classes:
            raise ValueError(
                f"Class ID {class_id} is out of range [0, {runtime_cfg.num_classes - 1}]."
            )

    visible_gpu_ids = resolve_visible_gpu_ids(runtime_cfg)
    if not visible_gpu_ids:
        raise RuntimeError("No GPU IDs were resolved from YAML or environment.")

    sample_specs = build_sample_specs(
        class_ids=selected_class_ids,
        class_names=class_names,
        samples_per_class=args.samples_per_class,
    )
    output_stem = build_samplewise_output_stem(
        class_ids=selected_class_ids,
        pool_type=args.pool_type,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
    )
    output_svg_path = output_dir / f"{output_stem}.svg"
    partial_dir = output_dir / f".partials_{output_stem}"

    return {
        "ckpt_path": str(ckpt_path),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "output_svg_path": str(output_svg_path),
        "partial_dir": str(partial_dir),
        "analysis_steps": analysis_steps,
        "class_names": class_names,
        "selected_class_ids": selected_class_ids,
        "visible_gpu_ids": visible_gpu_ids,
        "runtime_cfg": runtime_cfg,
        "sample_specs": sample_specs,
        "output_stem": output_stem,
    }


def _write_metadata(runtime_args, args) -> None:
    metadata_path = Path(runtime_args["output_dir"]) / f"{runtime_args['output_stem']}_metadata.yaml"
    metadata = {
        "checkpoint_path": runtime_args["ckpt_path"],
        "checkpoint_step": parse_checkpoint_step(Path(runtime_args["ckpt_path"])),
        "config_path": runtime_args["config_path"],
        "analysis_output_dir": runtime_args["output_dir"],
        "analysis_svg_path": runtime_args["output_svg_path"],
        "seed": args.seed,
        "cfg_scale": 1.0,
        "analysis_every": args.analysis_every,
        "analysis_steps": runtime_args["analysis_steps"],
        "pool_type": args.pool_type,
        "num_selected_classes": len(runtime_args["selected_class_ids"]),
        "selected_class_ids": runtime_args["selected_class_ids"],
        "samples_per_class": args.samples_per_class,
        "total_samples": len(runtime_args["sample_specs"]),
        "visible_gpu_ids": runtime_args["visible_gpu_ids"],
        "sample_batch_size": args.sample_batch_size,
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

    sample_specs = deserialize_sample_specs(worker_args["sample_specs"])
    assigned_sample_specs = sample_specs[local_rank::worker_args["num_workers"]]
    if not assigned_sample_specs:
        logging.info("No samples assigned to this worker.")
        return

    model, missing_keys, unexpected_keys = build_model(
        runtime_cfg=worker_args["runtime_cfg"],
        ckpt_path=worker_args["ckpt_path"],
        device=device,
    )
    if local_rank == 0:
        logging.info("Model missing keys: %s", list(missing_keys))
        logging.info("Model unexpected keys: %s", list(unexpected_keys))

    capture = BlockOutputCapture(model)
    logging.info(
        "Worker %d will analyze %d samples across %d blocks.",
        local_rank,
        len(assigned_sample_specs),
        len(capture.block_indices),
    )

    partial_results = []
    try:
        for sample_batch in chunked(assigned_sample_specs, worker_args["sample_batch_size"]):
            logging.info(
                "Analyzing sample batch ids: %s",
                [sample_spec.sample_id for sample_spec in sample_batch],
            )
            partial_results.append(
                sample_and_collect_pooled_features(
                    model=model,
                    capture=capture,
                    runtime_cfg=worker_args["runtime_cfg"],
                    sample_specs=sample_batch,
                    analysis_steps=worker_args["analysis_steps"],
                    seed=worker_args["seed"],
                    device=device,
                    pool_type=worker_args["pool_type"],
                )
            )
    finally:
        capture.close()

    merged_partial = merge_samplewise_partials(partial_results)
    partial_path = Path(worker_args["partial_dir"]) / f"worker_{local_rank:02d}.pt"
    save_partial_samplewise_result(partial_path, merged_partial)


def main():
    parser = argparse.ArgumentParser(
        description="Sample-wise t-SNE using pooled block token features."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for class sampling and latent sampling.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=5,
        help="Number of ImageNet classes to sample when --class-ids is not provided.",
    )
    parser.add_argument(
        "--class-ids",
        type=str_to_int_list,
        default=None,
        help="Optional comma-separated ImageNet class IDs to analyze.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="How many samples to generate for each selected ImageNet class.",
    )
    parser.add_argument(
        "--analysis-every",
        type=int,
        default=50,
        help="Capture pooled block features every N denoising steps.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=8,
        help="How many samples each worker processes together in one sampling batch.",
    )
    parser.add_argument(
        "--pool-type",
        type=str,
        default="mean",
        choices=SUPPORTED_POOL_TYPES,
        help="How to pool token features into a per-sample vector.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=None,
        help="Optional t-SNE perplexity override. Defaults to an adaptive value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate outputs even if the target SVG already exists.",
    )
    args = parser.parse_args()

    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive.")
    if args.sample_batch_size <= 0:
        raise ValueError("--sample-batch-size must be positive.")

    runtime_args = _prepare_runtime_arguments(args)
    output_svg_path = Path(runtime_args["output_svg_path"])
    if output_svg_path.exists() and not args.overwrite:
        print(f"Skip because output already exists: {output_svg_path}")
        return

    partial_dir = Path(runtime_args["partial_dir"])
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(runtime_args, args)

    num_workers = len(runtime_args["visible_gpu_ids"])
    worker_args = {
        "ckpt_path": runtime_args["ckpt_path"],
        "runtime_cfg": runtime_args["runtime_cfg"],
        "visible_gpu_ids": runtime_args["visible_gpu_ids"],
        "num_workers": num_workers,
        "analysis_steps": runtime_args["analysis_steps"],
        "sample_specs": serialize_sample_specs(runtime_args["sample_specs"]),
        "seed": args.seed,
        "sample_batch_size": args.sample_batch_size,
        "pool_type": args.pool_type,
        "partial_dir": runtime_args["partial_dir"],
    }

    if num_workers == 1:
        _worker(0, worker_args)
    else:
        mp.spawn(_worker, nprocs=num_workers, args=(worker_args,))

    partial_paths = sorted(partial_dir.glob("worker_*.pt"))
    partials = [load_partial_samplewise_result(partial_path) for partial_path in partial_paths]
    merged = merge_samplewise_partials(partials)
    save_samplewise_tsne_svg(
        output_path=output_svg_path,
        merged_records=merged["results"],
        block_indices=merged["block_indices"],
        analysis_steps=runtime_args["analysis_steps"],
        scheduler_timesteps=merged["scheduler_timesteps"],
        selected_class_ids=runtime_args["selected_class_ids"],
        class_names=runtime_args["class_names"],
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        pool_type=args.pool_type,
        perplexity=args.perplexity,
    )
    shutil.rmtree(partial_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
