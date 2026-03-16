from __future__ import annotations

import argparse
import logging
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
    resolve_analysis_output_dir,
    resolve_config_from_checkpoint,
    resolve_repo_root,
    resolve_visible_gpu_ids,
    sanitize_config_for_yaml,
)
from analyses.t_SNE.imagenet_utils import (
    get_imagenet_class_names,
    sample_class_ids,
    slugify_class_name,
)
from analyses.t_SNE.plotting import save_tokenwise_tsne_svg
from analyses.t_SNE.routing_capture import TokenRoutingCapture
from analyses.t_SNE.sampling import (
    build_model,
    chunked,
    compute_analysis_steps,
    sample_and_capture_batch,
)
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
    output_dir = resolve_analysis_output_dir(ckpt_path)
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

    return {
        "ckpt_path": str(ckpt_path),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "analysis_steps": analysis_steps,
        "class_names": class_names,
        "selected_class_ids": selected_class_ids,
        "visible_gpu_ids": visible_gpu_ids,
        "runtime_cfg": runtime_cfg,
    }


def _write_metadata(runtime_args, args) -> None:
    metadata_path = Path(runtime_args["output_dir"]) / "analysis_metadata.yaml"
    metadata = {
        "checkpoint_path": runtime_args["ckpt_path"],
        "checkpoint_step": parse_checkpoint_step(Path(runtime_args["ckpt_path"])),
        "config_path": runtime_args["config_path"],
        "analysis_output_dir": runtime_args["output_dir"],
        "seed": args.seed,
        "cfg_scale": 1.0,
        "analysis_every": args.analysis_every,
        "analysis_steps": runtime_args["analysis_steps"],
        "num_selected_classes": len(runtime_args["selected_class_ids"]),
        "selected_class_ids": runtime_args["selected_class_ids"],
        "visible_gpu_ids": runtime_args["visible_gpu_ids"],
        "analysis_batch_size": args.analysis_batch_size,
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

    runtime_cfg = worker_args["runtime_cfg"]
    assigned_class_ids = worker_args["selected_class_ids"][local_rank::worker_args["num_workers"]]
    if not assigned_class_ids:
        logging.info("No classes assigned to this worker.")
        return

    class_names = worker_args["class_names"]
    output_dir = Path(worker_args["output_dir"])

    if not worker_args["overwrite"]:
        remaining_class_ids = []
        for class_id in assigned_class_ids:
            class_name = class_names[class_id]
            output_path = output_dir / f"class{class_id:03d}_{slugify_class_name(class_name)}.svg"
            if output_path.exists():
                logging.info("Skipping class %d because %s already exists.", class_id, output_path)
                continue
            remaining_class_ids.append(class_id)
        assigned_class_ids = remaining_class_ids

    if not assigned_class_ids:
        logging.info("All assigned classes already have SVG outputs.")
        return

    model, missing_keys, unexpected_keys = build_model(
        runtime_cfg=runtime_cfg,
        ckpt_path=worker_args["ckpt_path"],
        device=device,
    )
    if local_rank == 0:
        logging.info("Model missing keys: %s", list(missing_keys))
        logging.info("Model unexpected keys: %s", list(unexpected_keys))

    capture = TokenRoutingCapture(model)
    block_indices = capture.block_indices
    num_experts = max(capture.num_routed_experts.values())

    logging.info(
        "Worker %d will analyze %d classes across %d routed blocks.",
        local_rank,
        len(assigned_class_ids),
        len(block_indices),
    )

    for class_batch in chunked(assigned_class_ids, worker_args["analysis_batch_size"]):
        logging.info("Analyzing class batch: %s", class_batch)
        batch_results = sample_and_capture_batch(
            model=model,
            capture=capture,
            runtime_cfg=runtime_cfg,
            class_ids=class_batch,
            analysis_steps=worker_args["analysis_steps"],
            seed=worker_args["seed"],
            device=device,
        )

        for class_id in class_batch:
            class_name = class_names[class_id]
            output_path = output_dir / f"class{class_id:03d}_{slugify_class_name(class_name)}.svg"
            save_tokenwise_tsne_svg(
                output_path=output_path,
                class_id=class_id,
                class_name=class_name,
                class_records=batch_results[class_id],
                block_indices=block_indices,
                analysis_steps=worker_args["analysis_steps"],
                num_experts=num_experts,
                seed=worker_args["seed"],
                perplexity=worker_args["perplexity"],
            )

    capture.close()


def main():
    parser = argparse.ArgumentParser(
        description="Token-wise t-SNE analysis for routing behavior during sampling."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for ImageNet class sampling and latent noise.",
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
        "--analysis-every",
        type=int,
        default=50,
        help="Capture token representations every N denoising steps.",
    )
    parser.add_argument(
        "--analysis-batch-size",
        type=int,
        default=4,
        help="How many classes each worker analyzes together in one forward sampling batch.",
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
        help="Re-generate existing SVG outputs instead of skipping them.",
    )
    args = parser.parse_args()
    if args.analysis_batch_size <= 0:
        raise ValueError("--analysis-batch-size must be positive.")

    runtime_args = _prepare_runtime_arguments(args)
    _write_metadata(runtime_args, args)

    num_workers = len(runtime_args["visible_gpu_ids"])
    worker_args = {
        "ckpt_path": runtime_args["ckpt_path"],
        "output_dir": runtime_args["output_dir"],
        "analysis_steps": runtime_args["analysis_steps"],
        "class_names": runtime_args["class_names"],
        "selected_class_ids": runtime_args["selected_class_ids"],
        "runtime_cfg": runtime_args["runtime_cfg"],
        "num_workers": num_workers,
        "visible_gpu_ids": runtime_args["visible_gpu_ids"],
        "seed": args.seed,
        "analysis_batch_size": args.analysis_batch_size,
        "perplexity": args.perplexity,
        "overwrite": args.overwrite,
    }

    if num_workers == 1:
        _worker(0, worker_args)
    else:
        mp.spawn(_worker, nprocs=num_workers, args=(worker_args,))


if __name__ == "__main__":
    main()
