"""Run one finite-horizon quota-preserving routing probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.finite_horizon_routing import run_finite_horizon_routing_probe


def _default_output_path(checkpoint_path, seed):
    checkpoint_path = Path(checkpoint_path).resolve()
    step = checkpoint_path.stem.removeprefix("ckpt_step_")
    return (
        checkpoint_path.parent.parent
        / "sample"
        / f"step{step}"
        / "finite_horizon_routing_probe"
        / f"probe_seed{seed}.json"
    )


def _publish_result(path, payload, overwrite=False):
    """Atomically publish one complete result, with no implicit overwrite."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"Output already exists: {path}; use --overwrite explicitly"
                ) from error
        temporary_path.unlink(missing_ok=True)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "At one denoising step, swap two tokens' expert identities without "
            "changing any expert's token count, then compare immediate utility "
            "with state error after 1, 2, 4, and 8 native-routing Euler steps."
        )
    )
    parser.add_argument("--ckpt", required=True, help="Canonical checkpoint path")
    parser.add_argument(
        "--weights-ckpt",
        help="Optional local copy used only for loading model weights",
    )
    parser.add_argument("--latent", required=True, help="VAE latent .npz path")
    parser.add_argument("--label", type=int, required=True, help="ImageNet class ID")
    parser.add_argument(
        "--latent-key",
        choices=("latent", "latent_flip"),
        default="latent",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Explicit torch device; GPU use is never implicit",
    )
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_path = (
        Path(args.output).resolve()
        if args.output
        else _default_output_path(args.ckpt, args.seed)
    )
    if not args.no_save and output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; use --overwrite explicitly"
        )
    result = run_finite_horizon_routing_probe(
        checkpoint_path=args.ckpt,
        weights_checkpoint_path=args.weights_ckpt,
        latent_path=args.latent,
        label=args.label,
        latent_key=args.latent_key,
        seed=args.seed,
        device=args.device,
        num_threads=args.num_threads,
    )
    if not args.no_save:
        _publish_result(output_path, result, overwrite=args.overwrite)
        result["output"] = str(output_path)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
