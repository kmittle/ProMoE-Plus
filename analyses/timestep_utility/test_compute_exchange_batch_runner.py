import json
import tempfile
import unittest
from pathlib import Path

from analyses.run_compute_exchange_probe_batch import (
    _case_protocol_view,
    _load_locked_protocol,
    _load_sealed_result,
    _parse_devices,
    _publish_result,
    _validate_result,
    _write_or_validate_protocol,
)
from analyses.timestep_utility.compute_exchange_batch import (
    BLOCKS_BY_SPLIT,
    SIGMAS,
)
from analyses.timestep_utility.compute_exchange_probe import (
    CANDIDATE_COUNT,
    PROBE_VERSION,
)


def _case(split="plumbing"):
    return {
        "split": split,
        "id": "case-000",
        "label": 1,
        "seed": 2,
        "synset": "n00000001",
        "latent_relative": "n00000001/image.latent.npz",
        "latent_sha256": "abc",
        "latent": "/tmp/image.latent.npz",
    }


def _result(case, protocol_sha256="protocol"):
    split = case["split"]
    cells = []
    for block in BLOCKS_BY_SPLIT[split]:
        for sigma in SIGMAS:
            cell = {
                "block_index": block,
                "sigma": sigma,
                "candidate_count": CANDIDATE_COUNT,
            }
            if split == "plumbing":
                cell["efficacy_statistics_withheld"] = True
            else:
                cell.update({"records": [], "summary": {}})
            cells.append(cell)
    return {
        "compute_exchange_probe_version": PROBE_VERSION,
        "protocol_sha256": protocol_sha256,
        "batch_case": _case_protocol_view(case),
        "block_indices": list(BLOCKS_BY_SPLIT[split]),
        "sigmas": list(SIGMAS),
        "safety_only": split == "plumbing",
        "cells": cells,
    }


class ComputeExchangeBatchRunnerTests(unittest.TestCase):
    def test_device_parser_accepts_only_locked_four_gpu_group(self):
        self.assertEqual(
            _parse_devices("cuda:4,cuda:5,cuda:6,cuda:7"),
            ("cuda:4", "cuda:5", "cuda:6", "cuda:7"),
        )
        with self.assertRaisesRegex(Exception, "requires"):
            _parse_devices("cuda:0,cuda:1,cuda:2,cuda:3")

    def test_result_validation_withholds_plumbing_efficacy(self):
        case = _case("plumbing")
        result = _result(case)
        self.assertIs(_validate_result(result, case, "plumbing", "protocol"), result)
        result["cells"][0]["records"] = []
        with self.assertRaisesRegex(RuntimeError, "forbidden efficacy"):
            _validate_result(result, case, "plumbing", "protocol")

    def test_protocol_hash_sidecar_detects_tampering_without_runtime_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            protocol = {"locked": True, "value": 7}
            path, digest = _write_or_validate_protocol(output_dir, protocol)
            self.assertEqual(
                _load_locked_protocol(path, digest, verify_inputs=False),
                protocol,
            )
            path.write_text(json.dumps({"locked": False}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "content hash"):
                _load_locked_protocol(path, digest, verify_inputs=False)

    def test_sealed_result_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            case = _case("discovery")
            result = _result(case)
            result_path = Path(directory) / "result.json"
            _publish_result(result_path, result, "protocol", case["id"])
            loaded = _load_sealed_result(
                result_path,
                case,
                "discovery",
                "protocol",
            )
            self.assertEqual(loaded, result)


if __name__ == "__main__":
    unittest.main()
