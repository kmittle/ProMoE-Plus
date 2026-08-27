from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from credit_redistribution.benchmark import DistributedThroughputTimer


class BenchmarkTest(unittest.TestCase):
    def test_timer_covers_exact_locked_segment_and_persists_rank_zero_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "leg.json"
            runtime_cfg = SimpleNamespace(
                num_steps=301601,
                save_ckpt_interval=10_000_000,
            )
            timer_cfg = {
                "enabled": True,
                "leg": "A1",
                "mode": "transcript_only",
                "start_step": 301001,
                "warmup_updates": 100,
                "timed_updates": 500,
                "output_path": str(output_path),
            }
            with mock.patch(
                "credit_redistribution.benchmark.dist.is_initialized",
                return_value=True,
            ), mock.patch(
                "credit_redistribution.benchmark.dist.get_rank", return_value=0
            ), mock.patch(
                "credit_redistribution.benchmark.dist.get_world_size", return_value=4
            ), mock.patch(
                "credit_redistribution.benchmark.dist.all_gather_object"
            ), mock.patch(
                "credit_redistribution.benchmark.dist.barrier"
            ) as barrier, mock.patch(
                "credit_redistribution.benchmark.torch.cuda.synchronize"
            ) as synchronize, mock.patch(
                "credit_redistribution.benchmark.time.perf_counter"
            ) as perf_counter:
                calls = []
                synchronize.side_effect = lambda: calls.append("synchronize")
                barrier.side_effect = lambda: calls.append("barrier")
                perf_counter.side_effect = lambda: (
                    calls.append("timer") or (10.0 if calls.count("timer") == 1 else 15.0)
                )
                timer = DistributedThroughputTimer(runtime_cfg, timer_cfg)
                timer.before_batch(301101)
                self.assertEqual(calls, ["synchronize", "barrier", "timer"])
                timer.after_update(301600)
                timer.finalize(301601)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["elapsed_seconds"], 5.0)
            self.assertEqual(payload["seconds_per_update"], 0.01)


if __name__ == "__main__":
    unittest.main()
