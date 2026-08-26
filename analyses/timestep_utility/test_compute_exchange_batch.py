import unittest

from analyses.timestep_utility.compute_exchange_batch import (
    BLOCKS_BY_SPLIT,
    CONFIRMATORY_REQUIREMENTS,
    DISCOVERY_REQUIREMENTS,
    SIGMAS,
    SPLIT_COUNTS,
    _holm,
    _case_metrics,
    aggregate_case_results,
    requirements_for_split,
)


def _controls(**overrides):
    values = {
        "max_abs_noop_mse_change": 0.0,
        "max_abs_noop_output_change": 0.0,
        "max_abs_hook_mse_change": 0.0,
        "max_abs_hook_output_change": 0.0,
        "max_abs_forced_unforced_mse_change": 0.0,
        "max_abs_forced_unforced_output_change": 0.0,
        "max_abs_paired_native_mse_drift": 0.0,
        "max_abs_paired_native_output_drift": 0.0,
        "logical_count_mismatches": 0,
        "action_contract_mismatches": 0,
        "route_id_mismatches": 0,
        "route_weight_mismatches": 0,
    }
    values.update(overrides)
    return values


def _selector(gain, per_pass=None, positive=True, harm=False):
    return {
        "selected_gain": float(gain),
        "selected_per_transferred_pass_gain": float(
            gain if per_pass is None else per_pass
        ),
        "selected_positive": bool(positive),
        "selected_harm": bool(harm),
    }


def _case_result(split, case_id, fo_gain=4e-4, rolled_gain=5e-5):
    cells = []
    for block in BLOCKS_BY_SPLIT[split]:
        for sigma in SIGMAS:
            cell = {
                "block_index": block,
                "sigma": sigma,
                "native_mse": 1.0,
                "numerical_controls": _controls(),
            }
            if split == "plumbing":
                cell["efficacy_statistics_withheld"] = True
            else:
                cell["summary"] = {
                    "pair_concordance": 0.8,
                    "spearman": 0.7,
                    "oracle_gain": 8e-4,
                    "selectors": {
                        "first_order": _selector(fo_gain, 4e-5),
                        "random": _selector(5e-5, 5e-6),
                        "router_margin": _selector(4e-5, 4e-6),
                        "rolled_utility": _selector(rolled_gain, 5e-6),
                    },
                }
            cells.append(cell)
    return {"batch_case": {"id": case_id}, "cells": cells}


class ComputeExchangeBatchTests(unittest.TestCase):
    def test_requirements_are_locked_per_split(self):
        self.assertEqual(
            requirements_for_split("plumbing")["expected_case_count"],
            SPLIT_COUNTS["plumbing"],
        )
        self.assertEqual(
            requirements_for_split("discovery")["minimum_mean_gain"],
            DISCOVERY_REQUIREMENTS["minimum_mean_gain"],
        )
        self.assertEqual(
            requirements_for_split("confirmatory")["minimum_oracle_ratio"],
            CONFIRMATORY_REQUIREMENTS["minimum_oracle_ratio"],
        )
        with self.assertRaises(ValueError):
            requirements_for_split("unknown")

    def test_plumbing_withholds_efficacy_and_fails_closed(self):
        results = [
            _case_result("plumbing", f"case-{index}")
            for index in range(SPLIT_COUNTS["plumbing"])
        ]
        gate = aggregate_case_results(results, "plumbing")
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["efficacy_statistics_withheld"])
        results[0]["cells"][0]["numerical_controls"][
            "logical_count_mismatches"
        ] = 1
        failed = aggregate_case_results(results, "plumbing")
        self.assertFalse(failed["passed"])
        self.assertFalse(
            failed["safety_checks"]["logical_count_mismatches"]["passed"]
        )

    def test_holm_stops_after_first_nonrejection(self):
        result = _holm({"a": 0.001, "b": 0.06, "c": 0.0015})
        self.assertTrue(result["a"]["passed"])
        self.assertTrue(result["c"]["passed"])
        self.assertFalse(result["b"]["passed"])

    def test_strong_discovery_signal_passes_all_preregistered_checks(self):
        results = [
            _case_result("discovery", f"case-{index}")
            for index in range(SPLIT_COUNTS["discovery"])
        ]
        requirements = requirements_for_split("discovery")
        requirements["bootstrap_resamples"] = 2000
        gate = aggregate_case_results(results, "discovery", requirements)
        self.assertTrue(gate["safety_passed"])
        self.assertTrue(gate["efficacy_passed"])
        self.assertTrue(gate["passed"])

    def test_confirmatory_requires_rolled_utility_superiority(self):
        results = [
            _case_result(
                "confirmatory",
                f"case-{index}",
                fo_gain=4e-4,
                rolled_gain=4e-4,
            )
            for index in range(SPLIT_COUNTS["confirmatory"])
        ]
        for result in results:
            for cell in result["cells"]:
                cell["summary"]["selectors"]["rolled_utility"][
                    "selected_per_transferred_pass_gain"
                ] = 4e-5
        requirements = requirements_for_split("confirmatory")
        requirements["bootstrap_resamples"] = 2000
        gate = aggregate_case_results(results, "confirmatory", requirements)
        self.assertFalse(gate["checks"]["rolled_contrast"]["passed"])
        self.assertFalse(gate["checks"]["holm_primary"]["passed"])
        self.assertFalse(gate["passed"])

    def test_holm_control_contrasts_use_per_pass_gain(self):
        results = [
            _case_result("discovery", f"case-{index}")
            for index in range(SPLIT_COUNTS["discovery"])
        ]
        for result in results:
            for cell in result["cells"]:
                selectors = cell["summary"]["selectors"]
                selectors["first_order"]["selected_per_transferred_pass_gain"] = 1e-6
                selectors["random"]["selected_per_transferred_pass_gain"] = 2e-6
        requirements = requirements_for_split("discovery")
        requirements["bootstrap_resamples"] = 2000
        gate = aggregate_case_results(results, "discovery", requirements)
        self.assertFalse(
            gate["holm_primary"]["first_order_above_random"]["passed"]
        )
        self.assertFalse(gate["checks"]["holm_primary"]["passed"])

    def test_undefined_spearman_fails_closed_without_crashing(self):
        results = [
            _case_result("discovery", f"case-{index}")
            for index in range(SPLIT_COUNTS["discovery"])
        ]
        results[0]["cells"][0]["summary"]["spearman"] = None
        requirements = requirements_for_split("discovery")
        requirements["bootstrap_resamples"] = 2000
        gate = aggregate_case_results(results, "discovery", requirements)
        self.assertFalse(gate["checks"]["spearman_defined"]["passed"])
        self.assertFalse(gate["passed"])

    def test_oracle_recovery_keeps_negative_first_order_gain(self):
        result = _case_result(
            "confirmatory",
            "negative-first-order",
            fo_gain=-4e-4,
        )
        metric = _case_metrics(result, "confirmatory")
        self.assertAlmostEqual(metric["first_order_oracle_numerator"], -4e-4)


if __name__ == "__main__":
    unittest.main()
