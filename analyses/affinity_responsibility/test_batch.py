"""Tests for image-level RCL gate aggregation."""

import copy
import unittest

from analyses.affinity_responsibility.batch import aggregate_case_results
from analyses.affinity_responsibility.protocol import BLOCK_INDICES, SIGMA_VALUES


def _cell(block, sigma):
    return {
        "block_index": block,
        "sigma": sigma,
        "base_mse": 1.0,
        "responsibility": {
            "native_best_rate": 0.0,
            "candidate_oracle_better_rate": 1.0,
            "affinity_best_candidate_scale_spearman": 0.0,
        },
        "global_responsibility": {"candidate_oracle_better_rate": 1.0},
        "mechanism": {
            "valid": True,
            "assignment_count_mismatches": 0,
            "correct": {
                "gradient_conflict_score": 0.2,
                "dispatch_improve_harmful_work_fraction": 0.2,
                "exact_mse_change": 2e-5,
            },
            "shuffle_summary": {
                "correct_minus_shuffle_mean": 0.1,
                "correct_shuffle_percentile": 0.8,
                "correct_minus_shuffle_exact_mse_change": 1e-5,
                "correct_exact_harm_shuffle_percentile": 0.8,
                "correct_minus_shuffle_heldout_rcl_geometry_gain": 1e-4,
            },
        },
        "numerical_controls": {
            "noop_global_max_relative_mse_change": 0.0,
            "noop_token_max_relative_mse_change": 0.0,
            "exact_center_noop_relative_mse_change": 0.0,
            "router_score_reconstruction_error": 0.0,
            "diffusion_gradient_identity_relative_error": 0.0,
            "maximum_center_norm_relative_error": 0.0,
            "correct_half_step_first_order_relative_error": 0.0,
            "correct_full_vs_two_half_secant_relative_error": 0.0,
            "fixed_dispatch_mismatches": 0,
            "diffusion_only_exact_descent": True,
        },
    }


def _case(case_id):
    return {
        "batch_case": {"id": case_id},
        "invalid_gradient_cells": 0,
        "cells": [
            _cell(block, sigma)
            for block in BLOCK_INDICES
            for sigma in SIGMA_VALUES
        ],
    }


class BatchAggregationTest(unittest.TestCase):
    def test_plumbing_withholds_efficacy_and_passes_controls(self):
        summary = aggregate_case_results(
            [_case(f"case-{index}") for index in range(4)],
            "plumbing",
        )
        self.assertTrue(summary["passed"])
        self.assertTrue(summary["efficacy_statistics_withheld"])
        self.assertNotIn("image_metrics", summary)

    def test_one_bad_noop_blocks_plumbing(self):
        cases = [_case(f"case-{index}") for index in range(4)]
        cases[2] = copy.deepcopy(cases[2])
        cases[2]["cells"][0]["numerical_controls"][
            "exact_center_noop_relative_mse_change"
        ] = 1e-4
        summary = aggregate_case_results(cases, "plumbing")
        self.assertFalse(summary["passed"])
        self.assertFalse(
            summary["safety_checks"]["exact_center_noop_relative_mse"]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
