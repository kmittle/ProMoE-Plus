import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyses.dino_utility_neighborhood.probe import (
    _atomic_write_json,
    _capacity_assignment,
    _deranged_donors,
    _prepare_new_output_path,
    _shift_indices,
    _source_results_sha256,
    _validate_source_seals,
    leave_one_image_knn_predict,
    load_utility_cases,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_source_contract(root):
    records = []
    aggregate_rows = []
    for result_path in sorted(root.glob("class*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        latent = Path(payload["latent"])
        records.append({
            "id": result_path.stem,
            "label": payload["label"],
            "latent": str(latent),
            "latent_sha256": _sha256(latent),
            "seed": payload["seed"],
            "split": "confirmatory",
        })
        aggregate_rows.append({"case_id": result_path.stem})
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"version": 1, "cases": records}),
        encoding="utf-8",
    )
    aggregate = root / "aggregate.json"
    aggregate.write_text(json.dumps({
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "num_cases": len(records),
        "per_image": aggregate_rows,
    }), encoding="utf-8")


def _write_case_result(root, case_index, **updates):
    latent = root / f"latent{case_index}.npz"
    np.savez(latent, latent=np.zeros((8, 2, 2), dtype=np.float32))
    payload = {
        "token_indices": [3, 7],
        "cells": [{
            "block_index": 1,
            "sigma": 0.5,
            "native_mse": 1.0,
            "tokens": [
                {
                    "token_index": 7,
                    "native_expert": 1,
                    "exact_mse_changes": [-0.1, 0.0],
                    "router_scores": [0.1, 0.2],
                },
                {
                    "token_index": 3,
                    "native_expert": 0,
                    "exact_mse_changes": [0.0, -0.1],
                    "router_scores": [0.2, 0.1],
                },
            ],
        }],
        "latent": str(latent),
        "latent_key": "latent",
        "label": case_index,
        "seed": case_index,
        "model_name": "ProMoE_TC_B",
        "config": str(root / "config.yaml"),
        "checkpoint": str(root / "checkpoint.pth"),
        "checkpoint_step": 300000,
        "checkpoint_state": "ema_model_state_dict",
        "weights_checkpoint": str(root / "weights.pth"),
        "weights_checkpoint_step": 300000,
        "device": "cpu",
        "timestep_utility_probe_version": 1,
    }
    payload.update(updates)
    path = root / f"class{case_index:03d}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    _refresh_source_contract(root)
    return path


class DinoUtilityNeighborhoodTests(unittest.TestCase):
    def test_knn_excludes_every_record_from_the_query_image(self):
        features = np.asarray([
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
        ])
        profiles = np.asarray([
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ])
        case_indices = np.asarray([0, 0, 1, 2])

        predictions, neighbors = leave_one_image_knn_predict(
            features, profiles, case_indices, k=1
        )

        self.assertEqual(int(neighbors[0, 0]), 2)
        self.assertTrue(np.array_equal(predictions[0], profiles[2]))
        self.assertFalse(np.any(
            case_indices[neighbors] == case_indices[:, None]
        ))

    def test_capacity_assignment_preserves_native_expert_counts(self):
        predictions = np.asarray([
            [3.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 2.0, 1.0],
        ])
        native = np.asarray([2, 2, 1, 0])

        assignment = _capacity_assignment(predictions, native)

        self.assertTrue(np.array_equal(
            np.bincount(assignment, minlength=3),
            np.bincount(native, minlength=3),
        ))

    def test_wrong_image_mapping_is_always_a_derangement(self):
        donors = _deranged_donors(24, 2026090301)

        self.assertEqual(len(set(donors.tolist())), 24)
        self.assertFalse(np.any(donors == np.arange(24)))

    def test_spatial_shift_wraps_on_square_patch_grid(self):
        shifted = _shift_indices(
            np.asarray([0, 15, 240, 255]),
            num_patches=256,
            shift_y=1,
            shift_x=2,
        )

        self.assertTrue(np.array_equal(
            shifted,
            np.asarray([18, 17, 2, 1]),
        ))

    def test_loader_aligns_tokens_by_token_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case_index in range(3):
                _write_case_result(root, case_index)

            cases = load_utility_cases(root, expected_cases=3)

        self.assertEqual(cases[0].token_indices.tolist(), [3, 7])
        self.assertEqual(cases[0].cells[0].native_experts.tolist(), [0, 1])

    def test_loader_rejects_mixed_model_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_case_result(root, 0)
            _write_case_result(root, 1, model_name="OtherModel")
            _write_case_result(root, 2)

            with self.assertRaisesRegex(ValueError, "model_name"):
                load_utility_cases(root, expected_cases=3)

    def test_loader_rejects_cuda_source_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case_index in range(3):
                _write_case_result(root, case_index, device="cuda:0")

            with self.assertRaisesRegex(ValueError, "source device 'cpu'"):
                load_utility_cases(root, expected_cases=3)

    def test_loader_rejects_unknown_source_probe_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case_index in range(3):
                _write_case_result(
                    root,
                    case_index,
                    timestep_utility_probe_version=2,
                )

            with self.assertRaisesRegex(ValueError, "unsupported.*version"):
                load_utility_cases(root, expected_cases=3)

    def test_loader_rejects_latent_changed_after_manifest_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case_index in range(3):
                _write_case_result(root, case_index)
            np.savez(
                root / "latent0.npz",
                latent=np.ones((8, 2, 2), dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "Latent SHA-256 mismatch"):
                load_utility_cases(root, expected_cases=3)

    def test_external_seal_rejects_changed_utility_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case_index in range(3):
                _write_case_result(root, case_index)
            original = load_utility_cases(root, expected_cases=3)
            expected_results = _source_results_sha256(original)
            expected_aggregate = original[0].source_aggregate_sha256

            changed_path = root / "class000.json"
            changed = json.loads(changed_path.read_text(encoding="utf-8"))
            changed["diagnostic_note"] = "changed after protocol lock"
            changed_path.write_text(json.dumps(changed), encoding="utf-8")
            changed_cases = load_utility_cases(root, expected_cases=3)

            with self.assertRaisesRegex(ValueError, "utility-results"):
                _validate_source_seals(
                    changed_cases,
                    expected_aggregate_sha256=expected_aggregate,
                    expected_results_sha256=expected_results,
                )

    def test_atomic_writer_refuses_to_replace_a_locked_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            output.write_text("locked\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                _atomic_write_json(output, {"replacement": True})
            self.assertEqual(output.read_text(encoding="utf-8"), "locked\n")

    def test_output_rejects_a_dangling_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_target = root / "missing.json"
            output_link = root / "result.json"
            output_link.symlink_to(missing_target)

            with self.assertRaises(FileExistsError):
                _prepare_new_output_path(output_link)
            self.assertTrue(output_link.is_symlink())
            self.assertFalse(missing_target.exists())


if __name__ == "__main__":
    unittest.main()
