"""Prepare and execute the sealed RCL-responsibility mechanism gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.affinity_responsibility.runner import (
    LOCKED_DEVICES,
    prepare_protocol,
    run_gate,
    run_split,
    verify_protocol,
)


def _parse_devices(value):
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if devices != LOCKED_DEVICES:
        raise argparse.ArgumentTypeError(
            "The locked gate requires cuda:0,cuda:1,cuda:2,cuda:3"
        )
    return devices


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare, verify, or run the sealed RCL mechanism gate."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--ckpt", required=True)
    prepare.add_argument("--latent-root", required=True)
    prepare.add_argument("--output-dir")
    prepare.add_argument(
        "--devices",
        type=_parse_devices,
        default=LOCKED_DEVICES,
    )

    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-dir", required=True)

    run_all = subparsers.add_parser("run-gate")
    run_all.add_argument("--output-dir", required=True)

    run = subparsers.add_parser("run-split")
    run.add_argument("--output-dir", required=True)
    run.add_argument(
        "--split",
        choices=("plumbing", "discovery", "confirmatory"),
        required=True,
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_protocol(
            checkpoint_path=args.ckpt,
            latent_root=args.latent_root,
            devices=args.devices,
            output_dir=args.output_dir,
        )
    elif args.command == "verify":
        result = verify_protocol(args.output_dir)
    elif args.command == "run-gate":
        result = run_gate(args.output_dir)
    else:
        result = run_split(args.output_dir, args.split)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def _result_exit_code(result):
    """Return a failing process status when a gate result is negative."""

    return int(isinstance(result, dict) and result.get("passed") is False)


if __name__ == "__main__":
    raise SystemExit(_result_exit_code(main()))
