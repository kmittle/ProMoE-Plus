"""Distributed ABBA throughput timer for sealed credit runs."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.distributed as dist

from .serialization import atomic_write_json


class DistributedThroughputTimer:
    def __init__(self, runtime_cfg, timer_cfg):
        if not isinstance(timer_cfg, dict):
            timer_cfg = dict(timer_cfg)
        if not bool(timer_cfg.get("enabled", False)):
            raise ValueError("Throughput timer cannot start while disabled")
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.start_step = int(timer_cfg.get("start_step", 301001))
        self.warmup_updates = int(timer_cfg.get("warmup_updates", 100))
        self.timed_updates = int(timer_cfg.get("timed_updates", 500))
        self.end_step = self.start_step + self.warmup_updates + self.timed_updates
        self.leg = str(timer_cfg["leg"])
        self.mode = str(timer_cfg["mode"])
        self.output_path = Path(timer_cfg["output_path"]).resolve()
        if self.world_size != 4:
            raise ValueError("Sealed throughput timing requires four ranks")
        if self.warmup_updates != 100 or self.timed_updates != 500:
            raise ValueError("Throughput warmup/timed update counts changed")
        if int(runtime_cfg.num_steps) != self.end_step:
            raise ValueError("Throughput num_steps does not cover exactly 600 updates")
        if int(runtime_cfg.save_ckpt_interval) <= self.end_step:
            raise ValueError("Throughput legs must disable checkpoint saves")
        if self.mode not in {"transcript_only", "matched_redistribution"}:
            raise ValueError(f"Unknown throughput leg mode: {self.mode}")
        self.started_at = None
        self.completed = False

    def before_batch(self, step):
        if int(step) == self.start_step + self.warmup_updates:
            torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()
            self.started_at = time.perf_counter()

    def after_update(self, step):
        if int(step) != self.end_step - 1:
            return
        if self.started_at is None:
            raise RuntimeError("Throughput timed segment never started")
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        elapsed = time.perf_counter() - self.started_at
        local_error = None
        if self.rank == 0:
            try:
                payload = {
                    "version": 1,
                    "leg": self.leg,
                    "mode": self.mode,
                    "world_size": self.world_size,
                    "start_step": self.start_step,
                    "warmup_updates": self.warmup_updates,
                    "timed_updates": self.timed_updates,
                    "elapsed_seconds": elapsed,
                    "seconds_per_update": elapsed / self.timed_updates,
                }
                if self.output_path.exists():
                    import json

                    with self.output_path.open("r", encoding="utf-8") as handle:
                        if json.load(handle) != payload:
                            raise RuntimeError("Existing throughput leg result differs")
                else:
                    atomic_write_json(self.output_path, payload, mode=0o444)
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
        errors = [None] * self.world_size
        if dist.is_initialized():
            dist.all_gather_object(errors, local_error)
        else:
            errors[0] = local_error
        failures = [error for error in errors if error]
        if failures:
            raise RuntimeError("Throughput result persistence failed: " + "; ".join(failures))
        self.completed = True

    def finalize(self, next_step):
        if int(next_step) != self.end_step or not self.completed:
            raise RuntimeError("Throughput leg did not complete its locked timed segment")
