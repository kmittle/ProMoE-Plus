import json
import unittest
from pathlib import Path

import numpy as np

from analyses.timestep_utility.cycle_batch import (
    CONFIRMATORY_REQUIREMENTS,
    MANIFEST_NAME,
    SPLIT_COUNTS,
    _arm_gate,
    _bh_fdr,
    _bootstrap_ratio,
    _bootstrap_summary,
    _canonical_selection,
    aggregate_case_results,
    requirements_for_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "manifests"
    / "count_preserving_cycle_gate_v1.json"
)


def _arm_summary(selected_gain=0.0):
    return {
        "selected_gain": float(selected_gain),
        "selected_per_flip_gain": float(selected_gain / 2),
        "pair_concordance": 0.7,
        "selected_positive": bool(selected_gain > 0),
        "selected_harm": bool(selected_gain < 0),
    }


def _plumbing_result(case_id, forced_drift=0.0):
    arms = {
        arm: {"summary": _arm_summary(1e-4), "records": []}
        for arm in (
            "four_cycle",
            "six_cycle",
            "mixed_cycle",
            "single_token",
            "random_joint",
        )
    }
    return {
        "batch_case": {"id": case_id},
        "cells": [{
            "block_index": 5,
            "sigma": sigma,
            "native_mse": 1.0,
            "arms": arms,
            "six_cycle_audit": {
                "unique_six_rate": 0.1,
                "has_unique_six": True,
            },
            "numerical_controls": {
                "max_abs_noop_mse_change": 0.0,
                "max_abs_noop_output_change": 0.0,
                "max_abs_forced_unforced_mse_change": forced_drift,
                "max_abs_forced_unforced_output_change": 0.0,
                "max_abs_paired_native_mse_drift": 0.0,
                "max_abs_paired_native_output_drift": 0.0,
                "count_mismatches": 0,
            },
        } for sigma in (0.2, 0.5, 0.8)],
    }


class CycleBatchTests(unittest.TestCase):
    def test_checked_in_manifest_locks_all_splits(self):
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["name"], MANIFEST_NAME)
        self.assertEqual(payload["selection"], _canonical_selection())
        for split, count in SPLIT_COUNTS.items():
            self.assertEqual(
                sum(case["split"] == split for case in payload["cases"]),
                count,
            )
        statistical_labels = [
            case["label"] for case in payload["cases"]
            if case["split"] != "plumbing"
        ]
        self.assertEqual(len(statistical_labels), len(set(statistical_labels)))

    def test_bootstrap_uses_paired_image_vectors(self):
        summary = _bootstrap_summary([1.0, 2.0, 3.0], 1000, 17)
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(len(summary["values"]), 3)
        ratio = _bootstrap_ratio(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
            1000,
            19,
        )
        self.assertAlmostEqual(ratio["ratio"], 0.5)
        self.assertAlmostEqual(ratio["one_sided_lcb95"], 0.5)

    def test_bh_fdr_is_monotone_in_sorted_order(self):
        adjusted = _bh_fdr({"a": 0.001, "b": 0.02, "c": 0.2})
        self.assertLessEqual(adjusted["a"], adjusted["b"])
        self.assertLessEqual(adjusted["b"], adjusted["c"])

    def test_plumbing_withholds_efficacy_statistics(self):
        results = [
            _plumbing_result(f"case-{index}")
            for index in range(SPLIT_COUNTS["plumbing"])
        ]
        gate = aggregate_case_results(results, "plumbing")
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["efficacy_statistics_withheld"])
        self.assertNotIn("arm_gates", gate)

    def test_plumbing_fails_closed_on_numerical_drift(self):
        results = [
            _plumbing_result(
                f"case-{index}",
                forced_drift=2e-7 if index == 0 else 0.0,
            )
            for index in range(SPLIT_COUNTS["plumbing"])
        ]
        gate = aggregate_case_results(results, "plumbing")
        self.assertFalse(gate["passed"])
        self.assertFalse(
            gate["safety_checks"]["forced_unforced_relative_mse"]["passed"]
        )

    def test_requirements_are_locked_per_split(self):
        self.assertEqual(
            requirements_for_split("discovery")["expected_case_count"],
            24,
        )
        self.assertEqual(
            requirements_for_split("confirmatory")["expected_case_count"],
            48,
        )
        with self.assertRaises(ValueError):
            requirements_for_split("unknown")

    def test_confirmatory_requires_block_and_sigma_lcbs_separately(self):
        metrics = []
        for index in range(SPLIT_COUNTS["confirmatory"]):
            unstable_positive_mean = 4e-4 if index < 24 else -3e-4
            target_arm = {
                "selected_gain": 2e-4,
                "selected_per_flip_gain": 1e-4,
                "pair_concordance": 0.7,
                "selected_positive": 1.0,
                "selected_harm": 0.0,
                "per_block_selected_gain": {
                    "1": 1e-4,
                    "5": 1e-4,
                    "11": 1e-4,
                },
                "per_sigma_selected_gain": {
                    "0.2": 1e-4,
                    "0.5": unstable_positive_mean,
                    "0.8": unstable_positive_mean,
                },
            }
            metrics.append({
                "arms": {
                    "mixed_cycle": target_arm,
                    "random_joint": {
                        **target_arm,
                        "selected_per_flip_gain": 0.0,
                    },
                    "single_token": {
                        **target_arm,
                        "selected_gain": 2e-4,
                    },
                },
            })
        gate = _arm_gate(
            metrics,
            "mixed_cycle",
            CONFIRMATORY_REQUIREMENTS,
            resamples=10_000,
            seed=20260826,
        )
        self.assertEqual(gate["positive_strata_lcb"]["block"], 3)
        self.assertEqual(gate["positive_strata_lcb"]["sigma"], 1)
        self.assertFalse(
            gate["checks"]["positive_sigma_strata_lcb"]["passed"]
        )
        self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
