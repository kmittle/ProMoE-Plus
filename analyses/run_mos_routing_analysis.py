from __future__ import annotations

import argparse
import logging
import sys
import yaml
from contextlib import nullcontext
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
    resolve_repo_root,
)
from analyses.t_SNE.imagenet_utils import sample_class_ids
from analyses.t_SNE.sampling import (
    build_model,
    compute_analysis_steps,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from analyses.mos_routing.extract import (
    MoSRoutingCapture,
    build_dummy_teacher_all_z,
    read_mos_repa_params,
    resolve_mos_routing_output_dir,
)
from analyses.mos_routing.aggregate import (
    OnlineRoutingAggregator,
    PerTimestepRoutingAggregator,
)
from analyses.mos_routing.plotting import (
    plot_per_block_histograms,
    plot_all_blocks_histogram,
    plot_per_block_hist_by_timestep,
    plot_token_variance,
    plot_routing_entropy,
)
from utils import str_to_int_list
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


ALL_PLOTS = {"per_block_hist", "all_blocks_hist", "timestep", "spatial", "variance", "entropy"}


def _parse_plots(plots_str: str) -> set:
    if plots_str.lower() == "all":
        return ALL_PLOTS.copy()
    requested = {s.strip() for s in plots_str.split(",")}
    unknown = requested - ALL_PLOTS
    if unknown:
        raise ValueError(f"Unknown plot types: {unknown}. Valid: {sorted(ALL_PLOTS)}")
    return requested


def _prepare_runtime_arguments(args):
    repo_root = resolve_repo_root()
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    config_path = resolve_config_from_checkpoint(ckpt_path, repo_root=repo_root)
    runtime_cfg = load_runtime_cfg(config_path)
    output_dir = resolve_mos_routing_output_dir(ckpt_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_steps = compute_analysis_steps(runtime_cfg.sample_steps, args.analysis_every)
    if args.class_ids is not None:
        selected_class_ids = args.class_ids
    else:
        selected_class_ids = sample_class_ids(args.num_classes, args.seed, runtime_cfg.num_classes)

    for class_id in selected_class_ids:
        if class_id < 0 or class_id >= runtime_cfg.num_classes:
            raise ValueError(f"Class ID {class_id} is out of range [0, {runtime_cfg.num_classes - 1}].")

    return {
        "ckpt_path": str(ckpt_path),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "runtime_cfg": runtime_cfg,
        "analysis_steps": analysis_steps,
        "selected_class_ids": selected_class_ids,
    }


def _write_metadata(runtime_args, args):
    metadata_path = Path(runtime_args["output_dir"]) / "metadata.yaml"
    ckpt_path = Path(runtime_args["ckpt_path"])
    metadata = {
        "analysis_type": "mos_routing",
        "checkpoint": runtime_args["ckpt_path"],
        "checkpoint_step": parse_checkpoint_step(ckpt_path),
        "config_path": runtime_args["config_path"],
        "output_dir": runtime_args["output_dir"],
        "seed": args.seed,
        "num_classes": len(runtime_args["selected_class_ids"]),
        "selected_class_ids": runtime_args["selected_class_ids"],
        "samples_per_class": args.samples_per_class,
        "analysis_every": args.analysis_every,
        "analysis_steps": runtime_args["analysis_steps"],
        "plots": args.plots,
    }
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Metadata saved to {metadata_path}")


@torch.no_grad()
def run_analysis(args):
    runtime_args = _prepare_runtime_arguments(args)
    runtime_cfg = runtime_args["runtime_cfg"]
    output_dir = Path(runtime_args["output_dir"])
    analysis_steps = runtime_args["analysis_steps"]
    selected_class_ids = runtime_args["selected_class_ids"]
    ckpt_path = runtime_args["ckpt_path"]

    requested_plots = _parse_plots(args.plots)

    # Check overwrite
    metadata_path = output_dir / "metadata.yaml"
    if metadata_path.exists() and not args.overwrite:
        logger.info(f"Output already exists at {output_dir}. Use --overwrite to re-generate.")
        return

    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Config: {runtime_args['config_path']}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Classes: {len(selected_class_ids)}, samples/class: {args.samples_per_class}")
    logger.info(f"Analysis steps: {analysis_steps}")
    logger.info(f"Plots: {sorted(requested_plots)}")

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, missing, unexpected = build_model(runtime_cfg, ckpt_path, device)
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # Set train mode to activate routing code path
    model.train()

    # Setup capture
    capture = MoSRoutingCapture(model)
    align_blocks = capture.align_blocks
    mos_params = read_mos_repa_params(model)
    num_teacher_blocks = mos_params["num_teacher_blocks"]
    z_dim = mos_params["z_dim"]

    logger.info(f"Model type: {capture.model_type}, align_blocks: {align_blocks}")
    logger.info(f"Teacher blocks: {num_teacher_blocks}, z_dim: {z_dim}")

    # Setup aggregators
    top_k = getattr(model, 'mos_top_k', 2)
    aggregator = OnlineRoutingAggregator(align_blocks, num_teacher_blocks, top_k)
    timestep_aggregator = PerTimestepRoutingAggregator(
        align_blocks, num_teacher_blocks, analysis_steps, top_k
    )

    # Sampling parameters
    latent_shape = (4, 1, runtime_cfg.image_size // 8, runtime_cfg.image_size // 8)
    latent_size = runtime_cfg.image_size // 8  # 32 for 256px
    grid_size = latent_size // model.patch_size  # 16 for patch_size=2
    num_patches = grid_size * grid_size  # 256

    autocast_dtype = getattr(runtime_cfg, "val_param_dtype", torch.float32)
    analysis_step_set = set(analysis_steps)

    total_images = len(selected_class_ids) * args.samples_per_class
    sampling_sigmas = get_sampling_sigmas(runtime_cfg.sample_steps, runtime_cfg.sample_shift)
    dummy_teacher = build_dummy_teacher_all_z(
        num_teacher_blocks, 1, num_patches, z_dim, device
    )
    logger.info(f"Total images to process: {total_images}")

    # Main sampling loop
    processed = 0
    for class_id in selected_class_ids:
        for sample_idx in range(args.samples_per_class):
            seed_offset = args.seed + class_id * 1000 + sample_idx
            generator = torch.Generator(device=device)
            generator.manual_seed(seed_offset)
            latents = torch.randn(*latent_shape, generator=generator, device=device).unsqueeze(0)
            class_tensor = torch.tensor([class_id], device=device, dtype=torch.long)

            # Setup scheduler (fresh per sample, maintains internal step state)
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=runtime_cfg.num_train_timesteps,
                shift=runtime_cfg.shift,
            )
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)

            # Denoising loop
            for timestep_index, timestep_value in enumerate(timesteps):
                denoise_step = timestep_index + 1
                timestep_tensor = torch.full(
                    (1,), float(timestep_value.item()), device=device, dtype=timesteps.dtype
                )

                should_capture = denoise_step in analysis_step_set

                if should_capture:
                    capture.enable()

                if autocast_dtype in (torch.float16, torch.bfloat16):
                    autocast_context = torch.cuda.amp.autocast(dtype=autocast_dtype)
                else:
                    autocast_context = nullcontext()

                with autocast_context:
                    model_output = model(
                        latents,
                        timestep_tensor,
                        context=class_tensor,
                        teacher_all_z=dummy_teacher,
                    )
                    if isinstance(model_output, tuple):
                        noise_pred = model_output[0]
                    else:
                        noise_pred = model_output

                if should_capture:
                    capture.disable()
                    routing_data = capture.get_routing_data()
                    if routing_data:
                        aggregator.update(routing_data)
                        timestep_aggregator.update(routing_data, denoise_step)

                # Continue denoising
                if noise_pred.shape[1] != latents.shape[1]:
                    noise_pred, _ = noise_pred.chunk(2, dim=1)

                latents = sample_scheduler.step(
                    noise_pred.unsqueeze(2),
                    timestep_value,
                    latents,
                    return_dict=False,
                )[0]

            processed += 1
            if processed % 10 == 0 or processed == total_images:
                logger.info(f"Processed {processed}/{total_images} images")

    # Finalize stats
    stats = aggregator.finalize()
    timestep_stats = timestep_aggregator.finalize()

    # Write metadata
    _write_metadata(runtime_args, args)

    # Generate plots
    saved_paths = []

    if "per_block_hist" in requested_plots:
        paths = plot_per_block_histograms(stats, output_dir, align_blocks)
        saved_paths.extend(paths)
        logger.info(f"Saved per-block histograms: {[str(p) for p in paths]}")

    if "all_blocks_hist" in requested_plots:
        paths = plot_all_blocks_histogram(stats, output_dir)
        saved_paths.extend(paths)
        logger.info(f"Saved all-blocks histogram: {[str(p) for p in paths]}")

    if "timestep" in requested_plots:
        paths = plot_per_block_hist_by_timestep(
            timestep_stats, output_dir, align_blocks, analysis_steps
        )
        saved_paths.extend(paths)
        logger.info(f"Saved timestep histograms: {[str(p) for p in paths]}")

    if "variance" in requested_plots:
        paths = plot_token_variance(stats, output_dir, align_blocks)
        saved_paths.extend(paths)
        logger.info(f"Saved token variance: {[str(p) for p in paths]}")

    if "entropy" in requested_plots:
        paths = plot_routing_entropy(stats, output_dir, align_blocks)
        saved_paths.extend(paths)
        logger.info(f"Saved routing entropy: {[str(p) for p in paths]}")

    if "spatial" in requested_plots:
        logger.info("Spatial routing map: not yet implemented (Phase 6)")

    logger.info(f"Analysis complete. {len(saved_paths)} files saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MoS router teacher block selection patterns."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file (.pth).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--num-classes", type=int, default=20,
        help="Number of ImageNet classes to sample when --class-ids is not provided.",
    )
    parser.add_argument(
        "--class-ids", type=str_to_int_list, default=None,
        help="Comma-separated ImageNet class IDs to analyze.",
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=5,
        help="Number of samples to generate per class.",
    )
    parser.add_argument(
        "--analysis-every", type=int, default=50,
        help="Capture routing weights every N denoising steps.",
    )
    parser.add_argument(
        "--plots", type=str, default="all",
        help="Comma-separated plot types: per_block_hist,all_blocks_hist,timestep,spatial,variance,entropy (or 'all').",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate outputs even if they already exist.",
    )

    args = parser.parse_args()

    if args.num_classes <= 0:
        raise ValueError("--num-classes must be positive.")
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive.")
    if args.analysis_every <= 0:
        raise ValueError("--analysis-every must be positive.")

    run_analysis(args)


if __name__ == "__main__":
    main()
