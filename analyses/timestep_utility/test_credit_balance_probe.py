import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyses import run_learning_credit_balance_probe_batch as runner
from analyses.timestep_utility import credit_balance_batch as batch
from analyses.timestep_utility.credit_balance_probe import (
    BLOCKS,
    SIGMAS,
    credit_cell_statistics,
    gini,
    permutation_mean_load_credit_tv,
)


class CreditBalanceProbeTests(unittest.TestCase):
    def test_gini_handles_equal_and_concentrated_credit(self):
        self.assertEqual(gini([2.0, 2.0, 2.0]), 0.0)
        self.assertAlmostEqual(gini([0.0, 0.0, 3.0]), 2.0 / 3.0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            gini([1.0, -1.0])

    def test_permutation_null_is_deterministic_and_count_preserving(self):
        credit = np.asarray([1.0, 2.0, 10.0, 20.0])
        experts = np.asarray([0, 0, 1, 1])
        first = permutation_mean_load_credit_tv(
            credit,
            experts,
            num_experts=2,
            resamples=256,
            seed=17,
            chunk_size=31,
        )
        second = permutation_mean_load_credit_tv(
            credit,
            experts,
            num_experts=2,
            resamples=256,
            seed=17,
            chunk_size=64,
        )
        self.assertAlmostEqual(first, second)
        self.assertGreater(first, 0.0)
        self.assertLess(first, 0.5)

    def test_credit_statistics_separate_load_and_credit(self):
        experts = np.asarray([0, 0, 1, 1])
        equal = credit_cell_statistics(
            token_credit=np.ones(4),
            unit_weight_credit=np.ones(4),
            native_experts=experts,
            num_experts=3,
            permutation_seed=5,
            permutation_resamples=64,
        )
        self.assertEqual(equal["token_count"], [2, 2, 0])
        self.assertEqual(equal["credit_rate_gini"], 0.0)
        self.assertEqual(equal["load_credit_tv"], 0.0)
        self.assertEqual(equal["permutation_excess_tv"], 0.0)

        imbalanced = credit_cell_statistics(
            token_credit=np.asarray([1.0, 1.0, 9.0, 9.0]),
            unit_weight_credit=np.asarray([1.0, 1.0, 9.0, 9.0]),
            native_experts=experts,
            num_experts=2,
            permutation_seed=7,
            permutation_resamples=256,
        )
        self.assertEqual(imbalanced["token_count"], [2, 2])
        self.assertAlmostEqual(imbalanced["load_credit_tv"], 0.4)
        self.assertAlmostEqual(imbalanced["credit_rate_gini"], 0.4)

    def test_case_selection_is_deterministic_and_class_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in range(1000):
                class_dir = root / f"n{label:08d}"
                class_dir.mkdir()
                (class_dir / f"n{label:08d}_1.latent.npz").write_bytes(
                    f"latent-{label}".encode("ascii")
                )
            first = batch.select_cases(root)
            second = batch.select_cases(root)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 104)
        self.assertEqual(len({case["label"] for case in first}), 104)
        self.assertEqual(
            [sum(case["split"] == split for case in first) for split in batch.SPLIT_COUNTS],
            [8, 32, 64],
        )

    def test_plumbing_gate_hides_efficacy(self):
        results = [
            runner._result_for_publish(self._result("plumbing", index), "plumbing")
            for index in range(8)
        ]
        gate = batch.aggregate_credit_balance(results, "plumbing")
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["efficacy_hidden"])
        self.assertNotIn("metrics", gate)
        self.assertNotIn("expert_profiles", gate)

    def test_plumbing_case_artifact_contains_only_safety_cells(self):
        case = {
            "split": "plumbing",
            "id": "case-000",
            "label": 0,
            "seed": 17,
            "synset": "n00000000",
            "latent_relative": "n00000000/example.latent.npz",
            "latent_sha256": "latent-sha",
        }
        computed = self._result("plumbing", 0)
        computed.update({
            "protocol_sha256": "protocol-sha",
            "batch_case": batch.case_protocol_view(case),
            "block_indices": list(BLOCKS),
            "sigmas": list(SIGMAS),
        })
        with self.assertRaisesRegex(RuntimeError, "not efficacy-hidden"):
            runner._validate_published_result(computed, case, "protocol-sha")
        leaking = {**computed, "efficacy_hidden": True}
        with self.assertRaisesRegex(RuntimeError, "leaks efficacy"):
            runner._validate_published_result(leaking, case, "protocol-sha")
        result = runner._result_for_publish(computed, "plumbing")
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "case.json"
            runner._publish_result(result_path, result, "protocol-sha", "case-000")
            persisted = json.loads(result_path.read_text(encoding="utf-8"))
            loaded = runner._load_sealed_result(
                result_path,
                case,
                "protocol-sha",
            )
        self.assertEqual(loaded, persisted)
        self.assertTrue(persisted["efficacy_hidden"])
        for cell in persisted["cells"]:
            self.assertEqual(set(cell), runner.PLUMBING_CELL_KEYS)
            self.assertNotIn("statistics", cell)
            self.assertNotIn("native_mse", cell)
            self.assertNotIn("timestep", cell)

    def test_discovery_gate_uses_image_level_stability_and_strata(self):
        results = [self._result("discovery", index) for index in range(32)]
        with mock.patch.object(batch, "BOOTSTRAP_RESAMPLES", 500):
            gate = batch.aggregate_credit_balance(results, "discovery")
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["metrics"]["block_strata"]["positive_count"], 3)
        self.assertEqual(gate["metrics"]["sigma_strata"]["positive_count"], 3)
        self.assertAlmostEqual(
            gate["metrics"]["rank_stability"]["mean"],
            1.0,
        )
        self.assertIn("token_count_cv", gate["metrics"])
        self.assertIn("token_count_gini", gate["metrics"])
        self.assertEqual(gate["metrics"]["token_count_gini"]["mean"], 0.0)

    @staticmethod
    def _result(split, index):
        expert_credit = (np.arange(1, 13, dtype=np.float64) * (1.0 + index / 100)).tolist()
        cells = []
        for block in BLOCKS:
            for sigma in SIGMAS:
                cells.append({
                    "block_index": block,
                    "sigma": sigma,
                    "statistics": {
                        "token_count": [10] * 12,
                        "expert_credit": expert_credit,
                        "credit_rate_gini": 0.3,
                        "unit_weight_credit_rate_gini": 0.25,
                        "permutation_excess_tv": 0.1,
                        "token_count_cv": 0.0,
                        "token_count_gini": 0.0,
                    },
                    "numerical_controls": {
                        "max_abs_native_output_drift": 0.0,
                        "native_relative_mse_drift": 0.0,
                        "route_mismatches": 0,
                        "nonfinite_token_credits": 0,
                    },
                })
        return {
            "credit_balance_probe_version": 1,
            "batch_case": {
                "id": f"case-{index:03d}",
                "split": split,
            },
            "cells": cells,
        }


if __name__ == "__main__":
    unittest.main()
