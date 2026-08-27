"""Operate the sealed credit-rate redistribution continuation gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_redistribution.controller import BRANCHES
from credit_redistribution.heldout import materialize_heldout
from credit_redistribution.orchestration import (
    run_aggregation,
    run_evaluation,
    run_preflight,
    run_throughput,
    verify_launch,
)
from credit_redistribution.protocol import (
    DEFAULT_OUTPUT_ROOT,
    LATENT_ROOT,
    PARENT_PROTOCOL,
    V3_PATH,
    V4_PATH,
    write_protocol,
)


def _print_json(payload):
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "materialize-heldout",
        help="materialize the locked 128-case held-out tensors",
    )
    subparsers.add_parser(
        "write-protocol",
        help="write the post-push immutable implementation protocol",
    )
    subparsers.add_parser(
        "preflight",
        help="run transcript-only plus two 20-update measure-only replays",
    )
    launch = subparsers.add_parser(
        "verify-launch",
        help="verify all prerequisites immediately before one branch launch",
    )
    launch.add_argument("--branch", required=True, choices=BRANCHES)
    subparsers.add_parser(
        "evaluate",
        help="evaluate all completed branches without aggregating efficacy",
    )
    subparsers.add_parser(
        "aggregate",
        help="perform the single paired-bootstrap efficacy reveal",
    )
    subparsers.add_parser(
        "throughput",
        help="run the post-efficacy ABBA throughput gate",
    )
    args = parser.parse_args()

    if args.command == "materialize-heldout":
        path, digest = materialize_heldout(
            latent_root=LATENT_ROOT,
            output_dir=DEFAULT_OUTPUT_ROOT / "heldout",
            parent_protocol_path=PARENT_PROTOCOL,
            preregister_v3_path=V3_PATH,
            preregister_v4_path=V4_PATH,
        )
        _print_json({"manifest_path": str(path), "canonical_sha256": digest})
    elif args.command == "write-protocol":
        path, digest = write_protocol()
        _print_json({"protocol_path": str(path), "canonical_sha256": digest})
    elif args.command == "preflight":
        path, payload = run_preflight()
        _print_json({"summary_path": str(path), "passed": payload["passed"]})
    elif args.command == "verify-launch":
        _print_json(verify_launch(args.branch))
    elif args.command == "evaluate":
        print(run_evaluation())
    elif args.command == "aggregate":
        path, payload = run_aggregation()
        _print_json({
            "summary_path": str(path),
            "all_required_passed": payload["all_required_passed"],
        })
    elif args.command == "throughput":
        path, payload = run_throughput()
        _print_json({
            "summary_path": str(path),
            "passed": payload["passed"],
            "relative_slowdown": payload["relative_slowdown"],
        })
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
