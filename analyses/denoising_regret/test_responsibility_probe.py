import unittest

import torch

from analyses.denoising_regret.responsibility_probe import (
    _forced_route_weight_matrix,
    _forced_token_route_weights,
    _validate_candidate_scales,
    summarize_responsibility_records,
)


class _FakeMoe:
    def compute_router(self, hidden_states, labels):
        batch_size, seq_len, _ = hidden_states.shape
        weights = torch.full((batch_size, seq_len, 1), 0.4)
        indices = torch.zeros(batch_size, seq_len, 1, dtype=torch.long)
        return weights, indices, None


class ResponsibilityProbeTests(unittest.TestCase):
    def test_candidate_scale_contract(self):
        self.assertEqual(_validate_candidate_scales([0, 0.5, 1]), [0.0, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "At least one"):
            _validate_candidate_scales([])
        with self.assertRaisesRegex(ValueError, "unique"):
            _validate_candidate_scales([0.5, 0.5])
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_candidate_scales([float("inf")])

    def test_summary_detects_affinity_responsibility_inversion(self):
        records = [
            {
                "native_router_weight": 0.2,
                "responsibility_slope": -1.0,
                "first_order_mse_change": {"0.0": 0.2, "1.0": -0.8},
                "exact_mse_change": {"0.0": 1.0, "1.0": -2.0},
            },
            {
                "native_router_weight": 0.8,
                "responsibility_slope": 1.0,
                "first_order_mse_change": {"0.0": -0.8, "1.0": 0.2},
                "exact_mse_change": {"0.0": -3.0, "1.0": 2.0},
            },
        ]

        summary = summarize_responsibility_records(records, [0.0, 1.0])

        self.assertEqual(summary["candidate_oracle_better_rate"], 1.0)
        self.assertEqual(summary["native_best_rate"], 0.0)
        self.assertAlmostEqual(
            summary["affinity_best_candidate_scale_spearman"],
            -1.0,
        )
        self.assertEqual(summary["increase_weight_recommended_rate"], 0.5)
        self.assertEqual(summary["best_candidate_scale_counts"]["0.0"], 1)
        self.assertEqual(summary["best_candidate_scale_counts"]["1.0"], 1)

    def test_weight_overrides_are_scoped_and_preserve_dispatch(self):
        moe = _FakeMoe()
        hidden_states = torch.zeros(2, 3, 4)
        labels = torch.tensor([1, 2])
        token_indices = torch.tensor([0, 2])

        with _forced_token_route_weights(
            moe,
            token_indices,
            torch.tensor([0.1, 0.9]),
        ):
            weights, indices, _ = moe.compute_router(hidden_states, labels)
            self.assertTrue(torch.equal(indices, torch.zeros_like(indices)))
            self.assertAlmostEqual(weights[0, 0, 0].item(), 0.1)
            self.assertAlmostEqual(weights[1, 2, 0].item(), 0.9)
            self.assertAlmostEqual(weights[0, 1, 0].item(), 0.4)

        restored, _, _ = moe.compute_router(hidden_states, labels)
        self.assertTrue(torch.allclose(restored, torch.full_like(restored, 0.4)))

        matrix = torch.tensor([[0.2, 0.3, 0.4], [0.7, 0.8, 0.9]])
        with _forced_route_weight_matrix(moe, matrix):
            weights, _, _ = moe.compute_router(hidden_states, labels)
            self.assertTrue(torch.allclose(weights[..., 0], matrix))

        with _forced_token_route_weights(moe, token_indices, None):
            weights, _, _ = moe.compute_router(hidden_states, labels)
            self.assertTrue(torch.equal(weights, torch.full_like(weights, 0.4)))

        with _forced_route_weight_matrix(moe, None):
            weights, _, _ = moe.compute_router(hidden_states, labels)
            self.assertTrue(torch.equal(weights, torch.full_like(weights, 0.4)))


if __name__ == "__main__":
    unittest.main()
