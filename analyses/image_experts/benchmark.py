"""Paired DDP training-step benchmarks on deterministic synthetic latent batches."""

import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml

from config import cfg as defaults
from models.models_ProMoE_TC_image_shared import DiT as ImageSharedDiT
from models.models_ProMoE_TC_shared_plus_cond_image import DiT as SharedPlusCondImageDiT
from analyses.image_experts.routing import set_benchmark_routing


MODELS = {
    "ProMoE_TC_B_image_shared": ImageSharedDiT,
    "ProMoE_TC_B_shared_plus_cond_image": SharedPlusCondImageDiT,
}


def load_config(path):
    config = copy.deepcopy(defaults)

    def merge(destination, source):
        for key, value in source.items():
            if isinstance(value, dict) and key in destination:
                merge(destination[key], value)
            else:
                destination[key] = value

    merge(config, yaml.safe_load(Path(path).read_text()))
    if config.model_name not in MODELS:
        raise ValueError(f"unsupported benchmark model: {config.model_name}")
    return config


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    # Match train.py's per-parameter EMA update, including its launch overhead.
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].mul_(decay).add_(parameter, alpha=1 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer)


def make_batches(config, count, batch_size, device, rank, seed):
    generator = torch.Generator(device=device).manual_seed(seed + rank + 701)
    shape = (batch_size, 4, 32, 32)
    batches = []
    for _ in range(count):
        latent = torch.randn(shape, device=device, generator=generator)
        noise = torch.randn(shape, device=device, generator=generator)
        sigma = torch.randn(batch_size, device=device, generator=generator).sigmoid()
        labels = torch.randint(config.num_classes, (batch_size,), device=device, generator=generator)
        noisy = latent * (1 - sigma[:, None, None, None]) + noise * sigma[:, None, None, None]
        batches.append((noisy, sigma * 1000, labels, noise - latent))
    return batches


def training_step(model, ema, optimizer, batch, config):
    noisy, timestep, labels, target = batch
    with torch.autocast("cuda", dtype=config.param_dtype, enabled=config.param_dtype == torch.bfloat16):
        prediction = model(noisy, timestep, context=labels)
    prediction = prediction[:, :target.shape[1]]
    mse = (prediction - target).square()
    loss = torch.stack([image.mean() for image in mse]).mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    update_ema(ema, model.module)
    return loss.detach(), grad_norm.detach()


def benchmark_one(config, backend, batches, warmup, steps, device, rank, seed,
                  routing="native", freeze_weights=False):
    torch.manual_seed(seed)
    model = MODELS[config.model_name](**config.DiT_B_config).to(device).train()
    model.set_image_expert_backend(backend)
    set_benchmark_routing(model, routing)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.0 if freeze_weights else config.lr,
        betas=config.betas, weight_decay=config.weight_decay, fused=True,
    )
    model = DistributedDataParallel(model, device_ids=[device.index])
    torch.manual_seed(seed + rank + 1701)
    optimizer.zero_grad(set_to_none=True)
    for index in range(warmup):
        training_step(model, ema, optimizer, batches[index], config)
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(steps)]
    started = time.perf_counter()
    for index in range(steps):
        events[index][0].record()
        loss, grad_norm = training_step(model, ema, optimizer, batches[warmup + index], config)
        events[index][1].record()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    last_loss, last_norm = loss.item(), grad_norm.item()
    if not math.isfinite(last_loss) or not math.isfinite(last_norm):
        raise RuntimeError(f"nonfinite {backend} loss/gradient: {last_loss}, {last_norm}")
    routes = []
    for block in model.module.blocks:
        if block.use_moe:
            pool = getattr(block.mlp, "image_experts", None)
            if pool is None:
                pool = block.mlp.shared_expert
            routes.append(torch.bincount(pool.last_expert_indices.reshape(-1),
                                         minlength=pool.num_image_experts).cpu().tolist())
    local_result = {
        "rank": rank, "elapsed_seconds": elapsed,
        "step_ms": [start.elapsed_time(end) for start, end in events],
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2 ** 30,
        "final_mse": last_loss, "final_grad_norm": last_norm,
        "last_step_image_expert_counts": routes,
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_result)
    del model, ema, optimizer, loss, grad_norm
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return gathered


def run_benchmark(args, project_root):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != args.expected_gpus or args.global_batch_size % world:
            raise ValueError("GPU count or global-batch divisibility does not match the request")
        config = load_config(args.config)
        expert_counts = args.expert_counts or [config.DiT_B_config.MoE_config.num_image_experts]
        sources = [args.config, "analyses/run_image_expert_benchmark.py",
                   "analyses/image_experts/benchmark.py", "analyses/image_experts/routing.py",
                   "models/models_ProMoE_TC.py", "models/models_ProMoE_TC_image_shared.py",
                   "models/models_ProMoE_TC_shared_plus_cond_image.py", "models/modules.py"]
        source_hashes = {name: hashlib.sha256((project_root / name).read_bytes()).hexdigest()
                         for name in sources}
        output = Path(args.output).resolve()
        if not output.is_relative_to(project_root / "outputs"):
            raise ValueError("benchmark results must live under repository outputs/")
        if output.exists():
            raise FileExistsError(output)
        batch_size = args.global_batch_size // world
        batches = make_batches(config, args.warmup + args.steps, batch_size,
                               device, rank, args.seed)
        records = []
        for expert_count in expert_counts:
            config.DiT_B_config.MoE_config.num_image_experts = expert_count
            for repeat in range(args.repeats):
                order = ("bmm", "loop") if repeat % 2 == 0 else ("loop", "bmm")
                for backend in order:
                    if rank == 0:
                        print(f"experts={expert_count} routing={args.routing} repeat={repeat + 1} "
                              f"backend={backend}: {args.warmup} warmup + "
                              f"{args.steps} measured full training steps", flush=True)
                    ranks = benchmark_one(config, backend, batches, args.warmup, args.steps,
                                          device, rank, args.seed, args.routing, args.freeze_weights)
                    elapsed = max(item["elapsed_seconds"] for item in ranks)
                    step_ms = [max(item["step_ms"][index] for item in ranks)
                               for index in range(args.steps)]
                    record = {
                        "num_image_experts": expert_count,
                        "repeat": repeat + 1, "backend": backend,
                        "mean_step_ms": elapsed * 1000 / args.steps,
                        "median_step_ms": statistics.median(step_ms),
                        "images_per_second": args.global_batch_size * args.steps / elapsed,
                        "peak_allocated_gib": max(item["peak_allocated_gib"] for item in ranks),
                        "peak_reserved_gib": max(item["peak_reserved_gib"] for item in ranks),
                        "ranks": ranks,
                    }
                    records.append(record)
                    if rank == 0:
                        print(json.dumps({key: value for key, value in record.items()
                                          if key != "ranks"}), flush=True)
        if rank == 0:
            summary = []
            for expert_count in expert_counts:
                row = {"num_image_experts": expert_count}
                for backend in ("bmm", "loop"):
                    selected = [record for record in records
                                if record["num_image_experts"] == expert_count
                                and record["backend"] == backend]
                    mean_ms = statistics.mean(record["mean_step_ms"] for record in selected)
                    row[f"{backend}_mean_step_ms"] = mean_ms
                    row[f"{backend}_images_per_second"] = args.global_batch_size * 1000 / mean_ms
                    row[f"{backend}_peak_allocated_gib"] = max(
                        record["peak_allocated_gib"] for record in selected)
                row["bmm_speedup"] = row["loop_mean_step_ms"] / row["bmm_mean_step_ms"]
                summary.append(row)
            report = {
                "model": config.model_name, "config": args.config,
                "global_batch_size": args.global_batch_size, "per_gpu_batch_size": batch_size,
                "world_size": world, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
                "cuda": torch.version.cuda, "dtype": str(config.param_dtype),
                "warmup_steps": args.warmup, "measured_steps": args.steps, "seed": args.seed,
                "expert_counts": expert_counts, "routing": args.routing,
                "freeze_weights": args.freeze_weights,
                "optimizer_lr": 0.0 if args.freeze_weights else config.lr,
                "scope": "synthetic preloaded latents; forward, MSE+aux backward, DDP, clip, fused AdamW, EMA; no data I/O",
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip(),
                "source_sha256": source_hashes,
                "records": records, "summary": summary,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            with output.with_name(output.stem + "_summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            print(f"RESULT: {output}", flush=True)
    finally:
        dist.destroy_process_group()
