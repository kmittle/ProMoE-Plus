import unittest

import torch

from analyses.routing_translation.probe import (
    _build_route_references,
    _forced_route_matrices,
    _random_matched_routes,
    _relative_mse_changes,
    _route_agreement,
    _translate_spatial,
    _validate_shifts,
)


class _FakeMoe:
    top_k = 1
    num_routed_experts = 4

    def compute_router(self, hidden_states, labels):
        batch_size, seq_len, _ = hidden_states.shape
        weights = torch.arange(
            1,
            batch_size * seq_len + 1,
            dtype=torch.float32,
        ).reshape(batch_size, seq_len, 1)
        indices = torch.zeros(batch_size, seq_len, 1, dtype=torch.long)
        return weights, indices, None


class RoutingTranslationProbeTests(unittest.TestCase):
    def test_shift_contract(self):
        self.assertEqual(_validate_shifts([(0, 2), (-2, 0)]), [(0, 2), (-2, 0)])
        with self.assertRaisesRegex(ValueError, "nonzero"):
            _validate_shifts([(0, 0)])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _validate_shifts([(0, 2), (0, 2)])
        with self.assertRaisesRegex(ValueError, "integers"):
            _validate_shifts([(0, 1.5)])

    def test_reflect_translation_has_no_wraparound(self):
        source = torch.arange(1, 17).reshape(1, 1, 4, 4).float()
        shifted = _translate_spatial(source, 0, 1)
        expected = torch.tensor([
            [2, 1, 2, 3],
            [6, 5, 6, 7],
            [10, 9, 10, 11],
            [14, 13, 14, 15],
        ]).reshape(1, 1, 4, 4).float()
        torch.testing.assert_close(shifted, expected)

    def test_content_and_position_route_references(self):
        original = torch.arange(9)
        shifted_native = torch.full((9,), 9)
        content, position, valid = _build_route_references(
            original,
            shifted_native,
            grid_size=3,
            token_shift=(0, 1),
        )
        self.assertTrue(torch.equal(valid.reshape(3, 3)[:, 0], torch.zeros(3, dtype=torch.bool)))
        self.assertTrue(torch.equal(
            content.reshape(3, 3),
            torch.tensor([[9, 0, 1], [9, 3, 4], [9, 6, 7]]),
        ))
        self.assertTrue(torch.equal(
            position.reshape(3, 3),
            torch.tensor([[9, 1, 2], [9, 4, 5], [9, 7, 8]]),
        ))

    def test_random_control_matches_support_and_histogram(self):
        native = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        content = torch.tensor([1, 2, 2, 3, 3, 0, 0, 1])
        valid = torch.ones_like(native, dtype=torch.bool)
        generator = torch.Generator().manual_seed(17)
        randomized, changed, available = _random_matched_routes(
            native,
            content,
            valid,
            generator,
        )
        self.assertTrue(torch.equal(changed, randomized != native))
        self.assertTrue(torch.equal(
            torch.bincount(content[changed], minlength=4),
            torch.bincount(randomized[changed], minlength=4),
        ))
        self.assertGreater((randomized != content).sum().item(), 0)
        self.assertTrue(available)

    def test_random_control_marks_unrandomizable_support_invalid(self):
        native = torch.tensor([0, 0, 0])
        content = torch.tensor([1, 0, 0])
        valid = torch.ones_like(native, dtype=torch.bool)
        randomized, changed, available = _random_matched_routes(
            native,
            content,
            valid,
            torch.Generator().manual_seed(5),
        )
        self.assertFalse(available)
        self.assertTrue(torch.equal(randomized, content))
        self.assertEqual(changed.sum().item(), 1)

    def test_forced_routes_preserve_native_weights_and_restore_method(self):
        moe = _FakeMoe()
        hidden = torch.zeros(2, 3, 4)
        labels = torch.tensor([1, 2])
        forced = torch.tensor([[1, 2, 3], [3, 2, 1]])
        native_weights, _, _ = moe.compute_router(hidden, labels)
        with _forced_route_matrices(moe, forced):
            weights, indices, _ = moe.compute_router(hidden, labels)
            torch.testing.assert_close(weights, native_weights)
            self.assertTrue(torch.equal(indices[..., 0], forced))
        _, restored, _ = moe.compute_router(hidden, labels)
        self.assertTrue(torch.equal(restored, torch.zeros_like(restored)))

    def test_route_agreement_reports_chance_correction(self):
        left = torch.tensor([0, 0, 1, 1])
        right = torch.tensor([0, 1, 1, 0])
        valid = torch.ones(4, dtype=torch.bool)
        metrics = _route_agreement(left, right, valid, num_experts=2)
        self.assertEqual(metrics["agreement"], 0.5)
        self.assertEqual(metrics["chance_agreement"], 0.5)
        self.assertEqual(metrics["chance_corrected_agreement"], 0.0)

    def test_relative_mse_change_requires_positive_finite_native_loss(self):
        names = ("native", "candidate")
        self.assertEqual(
            _relative_mse_changes(torch.tensor([2.0, 3.0]), names),
            {"native": 0.0, "candidate": 0.5},
        )
        with self.assertRaisesRegex(RuntimeError, "must be positive"):
            _relative_mse_changes(torch.tensor([0.0, 1.0]), names)
        with self.assertRaisesRegex(RuntimeError, "must be finite"):
            _relative_mse_changes(torch.tensor([1.0, float("nan")]), names)


if __name__ == "__main__":
    unittest.main()
