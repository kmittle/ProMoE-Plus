import copy
import unittest

from analyses.timestep_utility.batch import (
    BLOCK_INDICES,
    SIGMAS,
    aggregate_case_results,
)


def _synthetic_result(case_id):
    cells = []
    for block_index in BLOCK_INDICES:
        for sigma in SIGMAS:
            native_load = {
                "counts": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                "cv": 0.7071067811865476,
            }
            cells.append({
                "block_index": block_index,
                "sigma": sigma,
                "native_mse": 1.0,
                "assignments": {
                    "native": {
                        "exact_mse_change": 0.0,
                        "exact_mse_change_relative": 0.0,
                        "max_abs_output_change": 0.0,
                        "load": native_load,
                    },
                    "native_capacity_oracle": {
                        "exact_mse_change_relative": -2e-5,
                        "load": copy.deepcopy(native_load),
                    },
                    "balanced_capacity_oracle": {
                        "exact_mse_change_relative": -3e-5,
                        "load": {
                            "counts": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                            "cv": 0.7071067811865476,
                        },
                    },
                    "unconstrained_oracle": {
                        "exact_mse_change_relative": -4e-5,
                        "load": native_load,
                    },
                },
                "numerical_controls": {
                    "max_abs_forced_unforced_mse_change": 0.0,
                    "max_abs_forced_unforced_output_change": 0.0,
                    "weight_modes": {
                        mode: {
                            "max_abs_noop_mse_change": 0.0,
                            "max_abs_noop_output_change": 0.0,
                        }
                        for mode in ("native", "candidate", "unit")
                    },
                },
                "tokens": [
                    {
                        "oracle_expert": 1,
                        "sensitivity": {
                            "candidate": {"oracle_expert": 1},
                            "unit": {"oracle_expert": 1},
                        },
                    }
                    for _ in range(8)
                ],
            })
    return {
        "batch_case": {"id": case_id},
        "summary": {
            "native_is_oracle_rate": 0.05,
            "mean_native_regret_relative": 1e-4,
            "mean_router_utility_spearman": 0.0,
        },
        "stage_dynamics": {
            "summary": {
                "mean_router_minus_utility_rank_stability": 0.2,
                "mean_utility_pair_inversion_rate": 0.4,
                "oracle_expert_flip_rate": 0.7,
                "native_expert_flip_rate": 0.3,
            },
        },
        "cells": cells,
    }


class TimestepUtilityBatchTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            _synthetic_result(f"case-{index}") for index in range(8)
        ]

    def test_discovery_gate_passes_locked_positive_fixture(self):
        gate = aggregate_case_results(self.results, "discovery")
        self.assertTrue(gate["safety_passed"])
        self.assertTrue(gate["routing_accuracy_gap_passed"])
        self.assertTrue(gate["stage_structure_passed"])
        self.assertTrue(gate["passed"])

    def test_stage_failure_does_not_erase_routing_gap(self):
        for result in self.results:
            result["stage_dynamics"]["summary"].update({
                "mean_router_minus_utility_rank_stability": 0.0,
                "mean_utility_pair_inversion_rate": 0.1,
                "oracle_expert_flip_rate": 0.3,
                "native_expert_flip_rate": 0.3,
            })
        gate = aggregate_case_results(self.results, "discovery")
        self.assertTrue(gate["routing_accuracy_gap_passed"])
        self.assertFalse(gate["stage_structure_passed"])
        self.assertTrue(gate["passed"])

    def test_safety_failure_blocks_overall_gate(self):
        self.results[0]["cells"][0]["numerical_controls"][
            "max_abs_forced_unforced_mse_change"
        ] = 1e-12
        gate = aggregate_case_results(self.results, "discovery")
        self.assertFalse(gate["safety_passed"])
        self.assertTrue(gate["routing_accuracy_gap_passed"])
        self.assertFalse(gate["passed"])

    def test_joint_native_noop_failure_blocks_overall_gate(self):
        self.results[0]["cells"][0]["assignments"]["native"][
            "max_abs_output_change"
        ] = 1e-12
        gate = aggregate_case_results(self.results, "discovery")
        self.assertFalse(gate["safety_passed"])
        self.assertFalse(gate["safety_checks"]["joint_native_output"]["passed"])
        self.assertFalse(gate["passed"])

    def test_nonfinite_safety_metric_is_rejected_before_maximum(self):
        self.results[0]["cells"][0]["assignments"]["native"][
            "max_abs_output_change"
        ] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            aggregate_case_results(self.results, "discovery")

    def test_negative_absolute_or_cv_metric_is_rejected(self):
        cases = (
            (
                self.results[0]["cells"][0]["assignments"]["native"],
                "max_abs_output_change",
                -1.0,
            ),
            (
                self.results[0]["cells"][0]["assignments"]["native"]["load"],
                "cv",
                -1.0,
            ),
        )
        for target, key, value in cases:
            with self.subTest(key=key):
                target[key] = value
                with self.assertRaisesRegex(ValueError, "allowed minimum"):
                    aggregate_case_results(self.results, "discovery")
                target[key] = 0.0 if key == "max_abs_output_change" else 0.7071067811865476

    def test_rate_metrics_are_bounded(self):
        fields = (
            ("native_is_oracle_rate", 1.01),
            ("native_is_oracle_rate", -0.01),
            ("mean_utility_pair_inversion_rate", 1.01),
            ("oracle_expert_flip_rate", -0.01),
            ("mean_router_utility_spearman", 1.01),
        )
        for field, value in fields:
            with self.subTest(field=field, value=value):
                if field in self.results[0]["summary"]:
                    self.results[0]["summary"][field] = value
                else:
                    self.results[0]["stage_dynamics"]["summary"][field] = value
                with self.assertRaisesRegex(ValueError, "allowed"):
                    aggregate_case_results(self.results, "discovery")
                if field == "native_is_oracle_rate":
                    self.results[0]["summary"][field] = 0.05
                elif field == "mean_router_utility_spearman":
                    self.results[0]["summary"][field] = 0.0
                elif field == "mean_utility_pair_inversion_rate":
                    self.results[0]["stage_dynamics"]["summary"][field] = 0.4
                else:
                    self.results[0]["stage_dynamics"]["summary"][field] = 0.7


if __name__ == "__main__":
    unittest.main()
