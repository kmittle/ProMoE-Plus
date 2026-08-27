from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from credit_redistribution.controller import BRANCHES
from credit_redistribution.heldout import canonical_json_sha256
from credit_redistribution.orchestration import (
    _revalidate_before_aggregation,
    verify_prerequisites,
)
from credit_redistribution.serialization import sha256_file


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_protocol(path, payload):
    _write_json(path, payload)
    digest = canonical_json_sha256(payload)
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def _write_summary(root, name, protocol_sha256):
    payload = {
        "name": name,
        "protocol_sha256": protocol_sha256,
        "passed": True,
    }
    path = root / f"{name}-summary.json"
    _write_json(path, payload)
    _write_json(Path(str(path) + ".seal.json"), {
        "protocol_sha256": protocol_sha256,
        "result_sha256": canonical_json_sha256(payload),
    })


class OrchestrationTest(unittest.TestCase):
    def test_preaggregation_revalidates_every_completion_binding(self):
        protocol_sha256 = "a" * 64
        protocol = {"heldout_evaluation_output": "/sealed/evaluation"}
        checkpoint_specs = {
            branch: {"path": f"/{branch}.pth", "sha256": "b" * 64}
            for branch in BRANCHES
        }
        transcripts = {branch: "c" * 64 for branch in BRANCHES}
        integrity = {branch: {"ledger": "d" * 64} for branch in BRANCHES}
        trainer = {branch: "e" * 64 for branch in BRANCHES}
        completion = {
            "checkpoint_file_sha256": {
                branch: spec["sha256"] for branch, spec in checkpoint_specs.items()
            },
            "transcript_final_chain_digests": transcripts,
            "branch_integrity": integrity,
            "trainer_state_sha256": trainer,
        }
        validated = (
            protocol,
            protocol_sha256,
            {},
            checkpoint_specs,
            transcripts,
            integrity,
            trainer,
        )
        with mock.patch(
            "credit_redistribution.orchestration._load_sealed",
            return_value=completion,
        ), mock.patch(
            "credit_redistribution.orchestration.validate_protocol_for_evaluation",
            return_value=validated,
        ):
            _revalidate_before_aggregation(protocol, protocol_sha256)
            changed = copy.deepcopy(completion)
            changed["branch_integrity"][BRANCHES[0]] = {"ledger": "f" * 64}
            with mock.patch(
                "credit_redistribution.orchestration._load_sealed",
                return_value=changed,
            ), self.assertRaisesRegex(RuntimeError, "branch_integrity"):
                _revalidate_before_aggregation(protocol, protocol_sha256)

    def test_prerequisite_chain_binds_both_preregistrations_and_stage_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_root = root / "base"
            cross_root = root / "cross"
            base_protocol_path = base_root / "protocol.json"
            base_protocol_sha256 = _write_protocol(
                base_protocol_path, {"name": "base-protocol"}
            )
            _write_summary(base_root, "plumbing", base_protocol_sha256)

            v1 = root / "v1.json"
            v2 = root / "v2.json"
            base_preregister = root / "base-preregister.json"
            for path, version in ((v1, 1), (v2, 2), (base_preregister, 1)):
                _write_json(path, {"version": version})

            stage_order = ["plumbing", "confirmatory_credit"]
            cross_protocol = {
                "effective_preregistrations": [
                    {"version": 1, "path": str(v1), "sha256": sha256_file(v1)},
                    {"version": 2, "path": str(v2), "sha256": sha256_file(v2)},
                ],
                "stage_order": stage_order,
                "base_protocol": {
                    "canonical_json_sha256": base_protocol_sha256,
                },
                "git": {"commit": "abc", "origin_repa_divergence": "0\t0"},
                "project_source_sha256": {},
                "checkpoints": {},
            }
            cross_protocol_path = cross_root / "protocol.json"
            cross_protocol_sha256 = _write_protocol(
                cross_protocol_path, cross_protocol
            )
            _write_summary(cross_root, "confirmatory", cross_protocol_sha256)

            protocol = {
                "git": {"commit": "abc"},
                "prerequisites": {
                    "base_gate": {
                        "preregister_path": str(base_preregister),
                        "preregister_file_sha256": sha256_file(base_preregister),
                        "protocol_path": str(base_protocol_path),
                        "protocol_canonical_sha256": base_protocol_sha256,
                        "output_root": str(base_root),
                        "required_summaries": ["plumbing"],
                    },
                    "cross_checkpoint_gate": {
                        "preregister_v1_path": str(v1),
                        "preregister_v1_file_sha256": sha256_file(v1),
                        "preregister_v2_path": str(v2),
                        "preregister_v2_file_sha256": sha256_file(v2),
                        "protocol_path": str(cross_protocol_path),
                        "output_root": str(cross_root),
                        "required_stage_order": stage_order,
                        "required_summaries": ["confirmatory"],
                    },
                },
            }
            result = verify_prerequisites(protocol)
            self.assertEqual(result["base_protocol_sha256"], base_protocol_sha256)
            self.assertEqual(
                result["cross_checkpoint_protocol_sha256"],
                cross_protocol_sha256,
            )

            v2.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "preregistration changed"):
                verify_prerequisites(protocol)


if __name__ == "__main__":
    unittest.main()
