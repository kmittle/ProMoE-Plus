import json
import tempfile
import unittest
from pathlib import Path

from analyses.run_count_preserving_cycle_probe_batch import (
    _load_locked_protocol,
    _load_sealed_result,
    _parse_devices,
    _publish_result,
    _require_split_unlock,
    _seal_path,
    _write_or_validate_protocol,
)
from analyses.timestep_utility.cycle_batch import BATCH_VERSION, SIGMAS
from analyses.timestep_utility.cycle_probe import (
    CANDIDATE_SAMPLER_VERSION,
    PROBE_VERSION,
)


def _case():
    return {
        "split": "plumbing",
        "id": "class001_demo",
        "label": 1,
        "seed": 7,
        "synset": "n00000001",
        "latent_relative": "n00000001/demo.latent.npz",
        "latent_sha256": "a" * 64,
        "latent": "/tmp/demo.latent.npz",
    }


def _result(case, protocol_sha256):
    return {
        "cycle_probe_version": PROBE_VERSION,
        "candidate_sampler_version": CANDIDATE_SAMPLER_VERSION,
        "protocol_sha256": protocol_sha256,
        "batch_case": {
            key: case[key]
            for key in (
                "split",
                "id",
                "label",
                "seed",
                "synset",
                "latent_relative",
                "latent_sha256",
            )
        },
        "block_indices": [5],
        "sigmas": list(SIGMAS),
    }


class CycleBatchRunnerTests(unittest.TestCase):
    def test_devices_are_exactly_locked(self):
        self.assertEqual(
            _parse_devices("cuda:4,cuda:5,cuda:6,cuda:7"),
            ("cuda:4", "cuda:5", "cuda:6", "cuda:7"),
        )
        with self.assertRaises(Exception):
            _parse_devices("cuda:0,cuda:1,cuda:2,cuda:3")

    def test_protocol_is_idempotent_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            protocol = {"version": 1, "locked": True}
            path, digest = _write_or_validate_protocol(output_dir, protocol)
            second_path, second_digest = _write_or_validate_protocol(
                output_dir,
                protocol,
            )
            self.assertEqual(path, second_path)
            self.assertEqual(digest, second_digest)
            with self.assertRaises(RuntimeError):
                _write_or_validate_protocol(
                    output_dir,
                    {"version": 2, "locked": True},
                )

    def test_locked_protocol_rejects_on_disk_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            protocol = {"version": 1, "locked": True}
            path, digest = _write_or_validate_protocol(output_dir, protocol)
            path.write_text(
                json.dumps({"version": 2, "locked": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "protocol content hash changed",
            ):
                _load_locked_protocol(path, digest)

    def test_atomic_result_and_seal_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            case = _case()
            protocol_sha256 = "b" * 64
            result = _result(case, protocol_sha256)
            path = Path(directory) / "plumbing" / "01_case.json"
            _publish_result(path, result, protocol_sha256, case["id"])
            loaded = _load_sealed_result(
                path,
                case,
                "plumbing",
                protocol_sha256,
            )
            self.assertEqual(loaded, result)
            self.assertTrue(_seal_path(path).is_file())

    def test_seal_detects_result_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            case = _case()
            protocol_sha256 = "c" * 64
            result = _result(case, protocol_sha256)
            path = Path(directory) / "plumbing" / "01_case.json"
            _publish_result(path, result, protocol_sha256, case["id"])
            mutated = dict(result)
            mutated["block_indices"] = [1]
            path.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _load_sealed_result(
                    path,
                    case,
                    "plumbing",
                    protocol_sha256,
                )

    def test_split_unlock_reloads_prerequisite_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            protocol_sha256 = "d" * 64
            cases = [
                {
                    "split": "plumbing",
                    "id": f"case-{index}",
                    "label": index,
                    "seed": index,
                    "synset": f"n{index:08d}",
                    "latent_relative": f"n{index:08d}/demo.latent.npz",
                    "latent_sha256": "a" * 64,
                }
                for index in range(8)
            ]
            summary = {
                "batch_version": BATCH_VERSION,
                "probe_version": PROBE_VERSION,
                "split": "plumbing",
                "protocol": str(output_dir / "protocol.json"),
                "protocol_sha256": protocol_sha256,
                "case_ids": [case["id"] for case in cases],
                "gate": {"passed": True},
            }
            _publish_result(
                output_dir / "plumbing-summary.json",
                summary,
                protocol_sha256,
                "plumbing-summary",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "Missing completed case result",
            ):
                _require_split_unlock(
                    output_dir,
                    "discovery",
                    protocol_sha256,
                    {"cases": cases},
                )


if __name__ == "__main__":
    unittest.main()
