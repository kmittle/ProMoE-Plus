import unittest

import torch

from analyses.routing_translation.flip_probe import (
    _build_flip_route_references,
    _flip_token_grid,
    _full_model_flip_mse,
    _hidden_flip_metrics,
)
from analyses.routing_translation.probe import _route_margin_metrics


class RoutingFlipProbeTests(unittest.TestCase):
    def test_flip_token_grid_handles_ids_and_hidden_states(self):
        ids = torch.arange(9)
        expected_ids = torch.tensor([2, 1, 0, 5, 4, 3, 8, 7, 6])
        self.assertTrue(torch.equal(_flip_token_grid(ids, 3), expected_ids))

        hidden = torch.arange(18).reshape(9, 2)
        expected_hidden = hidden.reshape(3, 3, 2).flip(1).reshape(9, 2)
        self.assertTrue(
            torch.equal(_flip_token_grid(hidden, 3), expected_hidden)
        )

    def test_flip_route_references_separate_content_and_position(self):
        original = torch.arange(9)
        flipped_native = torch.full((9,), 9)
        content, position, valid = _build_flip_route_references(
            original,
            flipped_native,
            grid_size=3,
        )
        self.assertTrue(torch.equal(
            content,
            torch.tensor([2, 1, 0, 5, 4, 3, 8, 7, 6]),
        ))
        self.assertTrue(torch.equal(position, original))
        self.assertTrue(valid.all())

    def test_exact_hidden_and_output_flip_controls(self):
        original_hidden = torch.arange(18).reshape(9, 2).float()
        flipped_hidden = _flip_token_grid(original_hidden, 3)
        metrics = _hidden_flip_metrics(
            original_hidden,
            flipped_hidden,
            grid_size=3,
        )
        self.assertAlmostEqual(metrics["content_follow_cosine_mean"], 1.0)
        self.assertEqual(metrics["content_follow_relative_l2_mean"], 0.0)

        original_prediction = torch.arange(16).reshape(1, 1, 4, 4).float()
        flipped_prediction = torch.flip(original_prediction, dims=(-1,))
        self.assertEqual(
            _full_model_flip_mse(original_prediction, flipped_prediction),
            0.0,
        )

    def test_route_margin_separates_changed_and_unchanged_tokens(self):
        scores = torch.tensor([
            [0.8, 0.5, 0.1],
            [0.2, 0.7, 0.4],
            [0.6, 0.1, 0.5],
            [0.9, 0.2, 0.3],
        ])
        native = torch.tensor([0, 1, 0, 0])
        content = torch.tensor([1, 1, 2, 0])
        valid = torch.ones(4, dtype=torch.bool)
        metrics = _route_margin_metrics(
            scores,
            native,
            content,
            valid,
        )
        self.assertEqual(metrics["changed_tokens"], 2)
        self.assertAlmostEqual(metrics["changed_rate"], 0.5)
        self.assertAlmostEqual(
            metrics["native_minus_content_changed"]["mean"],
            0.2,
        )
        self.assertAlmostEqual(
            metrics["content_expert_rank_changed"]["mean"],
            2.0,
        )
        self.assertAlmostEqual(
            metrics["native_top1_margin_changed"]["mean"],
            0.2,
        )
        self.assertAlmostEqual(
            metrics["native_top1_margin_unchanged"]["mean"],
            0.45,
        )


if __name__ == "__main__":
    unittest.main()
