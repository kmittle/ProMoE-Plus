"""Run the frozen checkpoint expert update-budget audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyses.expert_update_budget.audit import DEFAULT_CHUNK_SIZE, run_audit


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure conditional routed-expert parameter motion and endpoint "
            "AdamW update fields across a locked checkpoint trajectory."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    payload = run_audit(
        manifest_path=args.manifest,
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        chunk_size=args.chunk_size,
    )
    gate = payload["gate"]
    print(f"Expert update-budget gate passed: {gate['passed']}")
    for name, check in gate["checks"].items():
        print(
            f"{name}: observed={check['observed']} "
            f"required={check['required']} passed={check['passed']}"
        )
    if not gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
