"""Validate first-order denoising regret against exact route substitutions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.denoising_regret import run_probe
from analyses.denoising_regret.io import write_json_atomic


def _parse_sigmas(value):
    try:
        sigmas = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("sigmas must be comma-separated floats") from error
    if not sigmas:
        raise argparse.ArgumentTypeError("at least one sigma is required")
    return sigmas


def _default_output_path(checkpoint_path, seed):
    checkpoint_path = Path(checkpoint_path).resolve()
    step = checkpoint_path.stem.removeprefix("ckpt_step_")
    return (
        checkpoint_path.parent.parent
        / "sample"
        / f"step{step}"
        / "denoising_regret_probe"
        / f"probe_seed{seed}.json"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare first-order denoising-regret estimates with exact, "
            "equal-compute single-token expert substitutions."
        )
    )
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument(
        "--weights-ckpt",
        help=(
            "Optional local copy used only for loading weights. Config and output "
            "paths are still resolved from --ckpt."
        ),
    )
    parser.add_argument("--latent", required=True, help="VAE latent .npz path")
    parser.add_argument("--label", type=int, required=True, help="ImageNet class ID")
    parser.add_argument(
        "--sigmas", type=_parse_sigmas, default=_parse_sigmas("0.2,0.5,0.8")
    )
    parser.add_argument("--block-index", type=int, default=3)
    parser.add_argument("--num-token-probes", type=int, default=32)
    parser.add_argument(
        "--candidate-mode",
        choices=("runner-up", "random", "mixed"),
        default="runner-up",
    )
    parser.add_argument("--exact-batch-size", type=int, default=4)
    parser.add_argument(
        "--latent-key", choices=("latent", "latent_flip"), default="latent"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device. CPU is the default so the probe never claims a GPU implicitly.",
    )
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument(
        "--no-save", action="store_true", help="Print results without writing JSON"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing JSON result"
    )
    return parser


def main():
    args = build_parser().parse_args()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else _default_output_path(args.ckpt, args.seed)
    )
    if output_path.exists() and not args.overwrite and not args.no_save:
        raise FileExistsError(
            f"Result already exists: {output_path}; use --overwrite to replace it"
        )

    result = run_probe(
        checkpoint_path=args.ckpt,
        latent_path=args.latent,
        label=args.label,
        sigmas=args.sigmas,
        block_index=args.block_index,
        num_token_probes=args.num_token_probes,
        candidate_mode=args.candidate_mode,
        exact_batch_size=args.exact_batch_size,
        latent_key=args.latent_key,
        seed=args.seed,
        device=args.device,
        num_threads=args.num_threads,
        weights_checkpoint_path=args.weights_ckpt,
    )

    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps(result["per_sigma"], indent=2, sort_keys=True))
    if not args.no_save:
        write_json_atomic(output_path, result)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
