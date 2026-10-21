"""Compare image-expert bmm/loop backends with full DDP training steps."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.image_experts.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert-counts", type=int, nargs="+",
                        help="benchmark-only overrides; leave the experiment YAML unchanged")
    parser.add_argument("--routing", choices=("native", "balanced"), default="native")
    parser.add_argument("--freeze-weights", action="store_true",
                        help="run AdamW at lr=0 to isolate execution speed from learned routing drift")
    args = parser.parse_args()
    if min(args.global_batch_size, args.expected_gpus, args.warmup, args.steps, args.repeats) < 1:
        parser.error("batch, GPU, warmup, step and repeat counts must be positive")
    if args.expert_counts and (min(args.expert_counts) < 1
                              or len(set(args.expert_counts)) != len(args.expert_counts)):
        parser.error("expert counts must be distinct positive integers")
    run_benchmark(args, PROJECT_ROOT)


if __name__ == "__main__":
    main()
