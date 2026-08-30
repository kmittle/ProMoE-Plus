"""Compare Base and output-repulsion expert functions at one checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.denoising_regret.io import write_json_atomic
from analyses.expert_output_diversity import (
    DEFAULT_BLOCK_INDICES,
    DEFAULT_SIGMAS,
    run_expert_output_diversity_gate,
)
from analyses.expert_output_diversity.probe import DEFAULT_MANIFEST
from analyses.expert_output_diversity.probe import FORMAL_DEVICES


def _parse_float_list(value):
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one float is required")
    return result


def _parse_int_list(value):
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _parse_devices(value):
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices:
        raise argparse.ArgumentTypeError("at least one device is required")
    return devices


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether expert-output repulsion creates scale-independent "
            "expert-function diversity without harming routing or denoising."
        )
    )
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--variant-ckpt", required=True)
    parser.add_argument("--base-weights-ckpt")
    parser.add_argument("--variant-weights-ckpt")
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--manifest-split", default="discovery")
    parser.add_argument(
        "--sigmas",
        type=_parse_float_list,
        default=DEFAULT_SIGMAS,
    )
    parser.add_argument(
        "--block-indices",
        type=_parse_int_list,
        default=DEFAULT_BLOCK_INDICES,
    )
    parser.add_argument("--num-anchor-tokens", type=int, default=32)
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=FORMAL_DEVICES,
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help=(
            "allow unlocked checkpoints/settings; comparison.passed is null "
            "and cannot be used as the formal 50K decision"
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Result already exists: {output_path}; use --overwrite to replace it"
        )
    result = run_expert_output_diversity_gate(
        base_checkpoint=args.base_ckpt,
        variant_checkpoint=args.variant_ckpt,
        base_weights_checkpoint=args.base_weights_ckpt,
        variant_weights_checkpoint=args.variant_weights_ckpt,
        latent_root=args.latent_root,
        manifest_path=args.manifest,
        manifest_split=args.manifest_split,
        sigmas=args.sigmas,
        block_indices=args.block_indices,
        num_anchor_tokens=args.num_anchor_tokens,
        devices=args.devices,
        num_threads=args.num_threads,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        formal=not args.exploratory,
    )
    write_json_atomic(output_path, result)
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
