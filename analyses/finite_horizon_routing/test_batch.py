"""Tests for image-clustered finite-horizon routing decisions."""

import unittest

import numpy as np

from analyses.finite_horizon_routing.batch import aggregate_case_results
from analyses.finite_horizon_routing.protocol import (
    BLOCK_INDICES,
    HORIZONS,
    START_INDICES,
    summarize_cell_records,
)


def _candidate_records(
    aligned,
    all_future_harmful=False,
    swap_preference_tracks_future=False,
):
    values = np.asarray([
        -0.008,
        -0.007,
        -0.006,
        -0.005,
        -0.004,
        -0.003,
        -0.002,
        -0.001,
        0.001,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
    ])
    future = values if aligned else -values
    if all_future_harmful:
        future = future - 0.02
    swap_preference = future if swap_preference_tracks_future else values
    records = []
    for index, immediate in enumerate(values):
        record = {
            "id": f"candidate-{index:02d}",
            "mean_router_margin": float(-swap_preference[index]),
            "immediate_gain_relative": float(immediate),
            "immediate_native_mse": 1.0,
        }
        for horizon in HORIZONS:
            gain = immediate if horizon < 8 else future[index]
            record[f"h{horizon}_gain_relative"] = float(gain)
            record[f"h{horizon}_native_mse"] = 1.0
        records.append(record)
    return records


def _zero_control():
    return {
        "first_prediction": 0.0,
        "immediate_mse": 0.0,
        "horizons": {
            str(horizon): {"state": 0.0, "mse": 0.0}
            for horizon in HORIZONS
        },
    }


def _case(
    case_id,
    aligned=False,
    all_future_harmful=False,
    swap_preference_tracks_future=False,
):
    cells = []
    for block in BLOCK_INDICES:
        for start in START_INDICES:
            candidates = _candidate_records(
                aligned,
                all_future_harmful,
                swap_preference_tracks_future,
            )
            cells.append({
                "block_index": block,
                "start_index": start,
                "summary": summarize_cell_records(candidates, 1e-7),
                "candidates": candidates,
                "numerical_controls": {
                    "reference_duplicate": _zero_control(),
                    "forced_native_vs_unforced": _zero_control(),
                    "paired_native_vs_reference": _zero_control(),
                    "max_abs_h1_state_velocity_identity_error": 0.0,
                    "count_mismatches": 0,
                },
            })
    return {"batch_case": {"id": case_id}, "cells": cells}


class AggregateTest(unittest.TestCase):
    def test_plumbing_withholds_efficacy(self):
        cases = [_case(f"plumbing-{index}") for index in range(4)]
        summary = aggregate_case_results(cases, "plumbing")
        self.assertTrue(summary["passed"])
        self.assertTrue(summary["efficacy_statistics_withheld"])
        self.assertNotIn("efficacy", summary)

    def test_confirmatory_reversed_ranking_passes(self):
        cases = [_case(f"confirm-{index}") for index in range(24)]
        summary = aggregate_case_results(
            cases,
            "confirmatory",
            prerequisite_discovery_passed=True,
        )
        self.assertTrue(summary["passed"])
        self.assertEqual(
            summary["efficacy"]["summaries"]["rho"]["mean"],
            -1.0,
        )
        self.assertEqual(
            summary["efficacy"]["summaries"]["sign_disagreement"]["mean"],
            1.0,
        )
        self.assertEqual(
            summary["candidate_label_permutation"]["role"],
            "diagnostic_only",
        )
        self.assertFalse(
            summary["candidate_label_permutation"]["included_in_pass"]
        )
        self.assertEqual(summary["pass_components"], ["safety", "efficacy"])

    def test_aligned_discovery_is_rejected(self):
        cases = [_case(f"discovery-{index}", aligned=True) for index in range(8)]
        summary = aggregate_case_results(cases, "discovery")
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["efficacy"]["checks"]["rho_mean"]["passed"])
        self.assertFalse(
            summary["efficacy"]["checks"]["sign_disagreement_mean"]["passed"]
        )

    def test_reversed_ranking_without_native_headroom_is_rejected(self):
        cases = [
            _case(f"confirm-{index}", all_future_harmful=True)
            for index in range(24)
        ]
        summary = aggregate_case_results(
            cases,
            "confirmatory",
            prerequisite_discovery_passed=True,
        )
        self.assertFalse(summary["passed"])
        self.assertTrue(summary["efficacy"]["checks"]["rho_mean"]["passed"])
        self.assertFalse(
            summary["efficacy"]["checks"]["best_gain_mean"]["passed"]
        )
        self.assertFalse(
            summary["efficacy"]["checks"]["beneficial_rate_mean"]["passed"]
        )

    def test_predictive_swap_preference_is_not_called_a_router_failure(self):
        cases = [
            _case(
                f"confirm-{index}",
                swap_preference_tracks_future=True,
            )
            for index in range(24)
        ]
        summary = aggregate_case_results(
            cases,
            "confirmatory",
            prerequisite_discovery_passed=True,
        )
        self.assertFalse(summary["passed"])
        self.assertTrue(summary["efficacy"]["checks"]["rho_mean"]["passed"])
        self.assertFalse(
            summary["efficacy"]["checks"][
                "swap_preference_rho_mean"
            ]["passed"]
        )

    def test_confirmation_cannot_bypass_discovery(self):
        cases = [_case(f"confirm-{index}") for index in range(24)]
        with self.assertRaisesRegex(ValueError, "passing discovery"):
            aggregate_case_results(cases, "confirmatory")


if __name__ == "__main__":
    unittest.main()
