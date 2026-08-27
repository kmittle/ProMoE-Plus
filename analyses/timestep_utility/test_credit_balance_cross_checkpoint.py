import unittest

import numpy as np
import torch
import torch.nn as nn

from models.modules import MoeMLP
from analyses.timestep_utility.credit_balance_cross_checkpoint import (
    CROSS_CHECKPOINT_VERSION,
    aggregate_block_load,
    aggregate_parameter_credit_validation,
    autograd_moe_mlp_token_parameter_credit,
    evaluate_count_balance,
    evaluate_count_replay,
    exact_expert_parameter_credit,
    exact_moe_mlp_token_parameter_credit,
    validate_exact_parameter_credit_formula,
)
from analyses.timestep_utility.credit_balance_probe import BLOCKS, SIGMAS


class _ToyMoeLayer(nn.Module):
    def __init__(self, num_experts=3, hidden_size=4, intermediate_size=5):
        super().__init__()
        self.num_routed_experts = num_experts
        self.experts = nn.ModuleList([
            MoeMLP(hidden_size, intermediate_size) for _ in range(num_experts)
        ])


class CrossCheckpointCreditTests(unittest.TestCase):
    def test_closed_form_parameter_credit_matches_autograd(self):
        torch.manual_seed(7)
        expert = MoeMLP(4, 6).double()
        inputs = torch.randn(5, 4, dtype=torch.float64)
        output_grad = torch.randn(5, 4, dtype=torch.float64)
        exact = exact_moe_mlp_token_parameter_credit(
            expert,
            inputs,
            output_grad,
        )
        reference = autograd_moe_mlp_token_parameter_credit(
            expert,
            inputs,
            output_grad,
        )
        for key in ("with_bias", "without_bias"):
            torch.testing.assert_close(
                exact[key],
                reference[key],
                rtol=1e-10,
                atol=1e-10,
            )

    def test_closed_form_rejects_an_unregistered_activation_contract(self):
        expert = MoeMLP(4, 6)
        expert.act_fn = nn.GELU(approximate="none")
        with self.assertRaisesRegex(TypeError, "approximate='tanh'"):
            exact_moe_mlp_token_parameter_credit(
                expert,
                torch.randn(2, 4),
                torch.randn(2, 4),
            )

    def test_deterministic_formula_validation_passes(self):
        validation = validate_exact_parameter_credit_formula()
        self.assertTrue(validation["passed"])
        self.assertLessEqual(
            validation["maximum_relative_error"],
            validation["maximum_allowed_relative_error"],
        )

    def test_exact_expert_credit_uses_native_weighted_routes(self):
        torch.manual_seed(11)
        layer = _ToyMoeLayer()
        hidden = torch.randn(6, 4)
        suffix_gradient = torch.randn(6, 4)
        weights = torch.tensor([0.5, 1.0, 0.25, 0.75, 1.25, 0.4])
        indices = torch.tensor([0, 1, 2, 0, 1, 2])
        result = exact_expert_parameter_credit(
            layer,
            hidden,
            suffix_gradient,
            weights,
            indices,
        )
        self.assertEqual(result["token_count"], [2, 2, 2])
        expected = []
        for expert_index in range(3):
            selected = indices == expert_index
            weighted = suffix_gradient[selected] * weights[selected, None]
            expected.append(float(weighted.double().square().sum()))
        np.testing.assert_allclose(result["expert_output_credit"], expected)
        self.assertEqual(result["active_experts"], 3)
        self.assertIsNotNone(result["output_parameter_spearman"])

    def test_exact_expert_credit_rejects_nonfinite_native_weights(self):
        layer = _ToyMoeLayer()
        with self.assertRaisesRegex(ValueError, "route weights must be finite"):
            exact_expert_parameter_credit(
                layer,
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.tensor([1.0, float("nan"), 1.0]),
                torch.tensor([0, 1, 2]),
            )

    def test_block_load_gate_never_cancels_imbalance_across_blocks(self):
        base = [self._result("confirmatory", 0, balanced=False)]
        lossfree = [self._result("confirmatory", 0, balanced=True)]
        aggregated = aggregate_block_load(lossfree, "confirmatory")
        self.assertEqual(set(aggregated), {"1", "5", "11"})
        self.assertEqual(aggregated["1"]["count_cv"], 0.0)
        gate = evaluate_count_balance(lossfree, base, "confirmatory")
        self.assertTrue(gate["passed"])

        cross_block_cancelling = [
            self._result("confirmatory", 0, balanced=False, rotate_blocks=True)
        ]
        failed = evaluate_count_balance(
            cross_block_cancelling,
            base,
            "confirmatory",
        )
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["blocks"]["1"]["passed"])

    def test_zero_variance_base_cannot_claim_fractional_reduction(self):
        base = [self._result("confirmatory", 0, balanced=True)]
        gate = evaluate_count_balance(base, base, "confirmatory")
        self.assertFalse(gate["passed"])
        self.assertIsNone(gate["blocks"]["1"]["cv_fractional_reduction"])

    def test_count_and_credit_passes_must_replay_every_count_vector(self):
        count = [self._result("discovery", 0, balanced=True)]
        credit = [self._result("discovery", 0, balanced=True)]
        self.assertTrue(
            evaluate_count_replay(count, credit, "discovery")["passed"]
        )
        credit[0]["cells"][0]["statistics"]["token_count"][0] += 1
        replay = evaluate_count_replay(count, credit, "discovery")
        self.assertFalse(replay["passed"])
        self.assertEqual(replay["mismatch_count"], 1)

    def test_parameter_gate_uses_paired_image_level_bootstrap(self):
        results = {
            role: [
                self._parameter_result(role, index, spearman=0.8)
                for index in range(16)
            ]
            for role in ("base", "lossfree")
        }
        gate = aggregate_parameter_credit_validation(
            results,
            resamples=500,
            seed=13,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["bootstrap_resamples"], 500)
        self.assertEqual(len(gate["case_ids"]), 16)
        self.assertAlmostEqual(
            gate["checkpoints"]["base"]
            ["primary_output_parameter_spearman"]["mean"],
            0.8,
        )

        results["lossfree"] = [
            self._parameter_result("lossfree", index, spearman=0.1)
            for index in range(16)
        ]
        failed = aggregate_parameter_credit_validation(
            results,
            resamples=500,
            seed=13,
        )
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checkpoints"]["lossfree"]["passed"])

    def test_parameter_gate_rejects_unpaired_case_order(self):
        results = {
            role: [
                self._parameter_result(role, index, spearman=0.8)
                for index in range(16)
            ]
            for role in ("base", "lossfree")
        }
        results["lossfree"].reverse()
        with self.assertRaisesRegex(ValueError, "paired case order"):
            aggregate_parameter_credit_validation(
                results,
                resamples=10,
            )

    @staticmethod
    def _result(split, index, balanced, rotate_blocks=False):
        cells = []
        for block_offset, block in enumerate(BLOCKS):
            for sigma in SIGMAS:
                if balanced:
                    counts = [10] * 12
                else:
                    counts = [2] * 12
                    dominant = block_offset if rotate_blocks else 0
                    counts[dominant] = 98
                cells.append({
                    "block_index": block,
                    "sigma": sigma,
                    "statistics": {"token_count": counts},
                })
        return {
            "batch_case": {"id": f"case-{index:03d}", "split": split},
            "cells": cells,
        }

    @staticmethod
    def _parameter_result(role, index, spearman):
        cells = []
        for block in BLOCKS:
            for sigma in SIGMAS:
                cells.append({
                    "block_index": block,
                    "sigma": sigma,
                    "statistics": {"token_count": [10] * 12},
                    "parameter_statistics": {
                        "active_experts": 12,
                        "output_parameter_spearman": spearman,
                        "rate_spearman": spearman - 0.05,
                        "output_parameter_spearman_without_bias": (
                            spearman - 0.02
                        ),
                    },
                    "numerical_controls": {
                        "route_mismatches": 0,
                        "nonfinite_token_credits": 0,
                        "nonfinite_parameter_credits": 0,
                    },
                })
        return {
            "cross_checkpoint_probe_version": CROSS_CHECKPOINT_VERSION,
            "checkpoint_role": role,
            "batch_case": {
                "id": f"case-{index:03d}",
                "split": "discovery",
            },
            "cells": cells,
        }


if __name__ == "__main__":
    unittest.main()
