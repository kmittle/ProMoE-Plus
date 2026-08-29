"""Tests for the locked RCL-responsibility mathematics."""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from analyses.affinity_responsibility.protocol import (
    cosine_score_jvp,
    cosine_scores_and_selected_center_gradients,
    count_preserving_assignment_shuffles,
    norm_preserving_center_step,
    responsibility_center_gradient,
    routing_contrastive_center_gradient,
    summarize_external_rcl_gradient_cell,
    summarize_rcl_gradient_cell,
)


class CosineDerivativeTest(unittest.TestCase):
    def setUp(self):
        self.hidden = torch.tensor(
            [[1.0, 2.0, -1.0], [0.5, -1.5, 2.0], [2.0, 0.25, 1.0]],
            dtype=torch.float64,
        )
        self.centers = torch.tensor(
            [[0.75, -0.5, 1.5], [-1.0, 2.0, 0.5]],
            dtype=torch.float64,
        )
        self.assignments = torch.tensor([0, 1, 0])

    def test_all_score_jvp_matches_autograd(self):
        direction = torch.tensor(
            [[0.2, -0.3, 0.4], [-0.5, 0.1, 0.25]],
            dtype=torch.float64,
        )
        centers = self.centers.clone().requires_grad_(True)
        tangent = torch.autograd.functional.jvp(
            lambda value: F.normalize(self.hidden, dim=1)
            @ F.normalize(value, dim=1).T,
            centers,
            direction,
        )[1]
        observed = cosine_score_jvp(self.hidden, self.centers, direction)
        torch.testing.assert_close(observed, tangent, rtol=1e-12, atol=1e-12)

    def test_selected_center_gradient_matches_autograd(self):
        centers = self.centers.clone().requires_grad_(True)
        scores = F.normalize(self.hidden, dim=1) @ F.normalize(centers, dim=1).T
        rows = torch.arange(self.assignments.numel())
        expected, = torch.autograd.grad(
            scores[rows, self.assignments].sum(),
            centers,
        )
        _, per_token = cosine_scores_and_selected_center_gradients(
            self.hidden,
            self.centers,
            self.assignments,
        )
        observed = torch.zeros_like(self.centers)
        observed.index_add_(0, self.assignments, per_token)
        torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)

    def test_responsibility_gradient_matches_autograd(self):
        slopes = torch.tensor([0.25, -0.75, 1.5], dtype=torch.float64)
        centers = self.centers.clone().requires_grad_(True)
        scores = F.normalize(self.hidden, dim=1) @ F.normalize(centers, dim=1).T
        rows = torch.arange(self.assignments.numel())
        expected, = torch.autograd.grad(
            (scores[rows, self.assignments] * slopes).sum(),
            centers,
        )
        observed, _, _ = responsibility_center_gradient(
            self.hidden,
            self.centers,
            self.assignments,
            slopes,
        )
        torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


class RclGradientTest(unittest.TestCase):
    def test_center_gradient_matches_direct_definition(self):
        hidden = torch.tensor(
            [
                [1.0, 0.2, -0.3],
                [0.8, -0.1, 0.4],
                [-0.4, 1.0, 0.2],
                [0.1, 0.7, -0.8],
            ],
            dtype=torch.float64,
        )
        centers = torch.tensor(
            [[1.0, -0.5, 0.25], [-0.2, 0.75, 1.2]],
            dtype=torch.float64,
        )
        assignments = torch.tensor([0, 0, 1, 1])
        temperature = 0.07

        observed = routing_contrastive_center_gradient(
            hidden,
            centers,
            assignments,
            temperature,
        )
        working = centers.clone().requires_grad_(True)
        means = torch.stack([
            hidden[assignments == expert].mean(dim=0)
            for expert in range(2)
        ])
        logits = F.normalize(working, dim=1) @ F.normalize(means, dim=1).T
        loss = F.cross_entropy(
            logits / temperature,
            torch.arange(2),
        )
        expected, = torch.autograd.grad(loss, working)

        self.assertAlmostEqual(observed["loss"], loss.item(), places=12)
        torch.testing.assert_close(
            observed["gradient"],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_assignment_shuffles_preserve_counts_and_are_unique(self):
        assignments = np.asarray([0, 0, 0, 1, 1, 2, 2, 2])
        shuffles = count_preserving_assignment_shuffles(assignments, 16, 123)
        signatures = {tuple(item.tolist()) for item in shuffles}
        self.assertEqual(len(shuffles), 16)
        self.assertEqual(len(signatures), 16)
        for item in shuffles:
            np.testing.assert_array_equal(
                np.bincount(item, minlength=3),
                np.bincount(assignments, minlength=3),
            )
            self.assertFalse(np.array_equal(item, assignments))

    def test_diffusion_only_control_is_the_steepest_descent_identity(self):
        generator = torch.Generator().manual_seed(19)
        hidden = torch.randn(12, 5, generator=generator, dtype=torch.float64)
        centers = torch.randn(3, 5, generator=generator, dtype=torch.float64)
        assignments = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        slopes = torch.randn(12, generator=generator, dtype=torch.float64)
        result = summarize_rcl_gradient_cell(
            hidden,
            centers,
            assignments,
            slopes,
            temperature=0.07,
            shuffle_count=4,
            shuffle_seed=31,
        )
        control = result["diffusion_only_control"]
        self.assertAlmostEqual(control["gradient_cosine"], 1.0, places=12)
        self.assertAlmostEqual(
            control["diffusion_loss_change_relative_to_steepest"],
            -1.0,
            places=12,
        )
        self.assertLess(
            control["diffusion_gradient_identity_relative_error"],
            1e-12,
        )
        self.assertEqual(result["assignment_count_mismatches"], 0)

    def test_norm_preserving_step_has_equal_scale_and_descends(self):
        generator = torch.Generator().manual_seed(113)
        centers = torch.randn(4, 7, generator=generator, dtype=torch.float64)
        gradient = torch.randn(4, 7, generator=generator, dtype=torch.float64)
        result = norm_preserving_center_step(
            centers,
            gradient,
            relative_frobenius=1e-3,
        )
        torch.testing.assert_close(
            result["centers"].norm(dim=1),
            centers.norm(dim=1),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertLess((gradient * result["displacement"]).sum().item(), 0)
        self.assertAlmostEqual(
            result["realized_relative_frobenius"],
            1e-3,
            delta=2e-6,
        )

    def test_model_precision_step_reports_post_cast_norm_error(self):
        generator = torch.Generator().manual_seed(117)
        centers = torch.randn(12, 37, generator=generator, dtype=torch.float32)
        gradient = torch.randn(12, 37, generator=generator, dtype=torch.float64)
        result = norm_preserving_center_step(
            centers,
            gradient,
            realized_dtype=torch.float32,
        )
        self.assertEqual(result["centers"].dtype, torch.float32)
        reference_norms = centers.double().norm(dim=1)
        realized_norms = result["centers"].double().norm(dim=1)
        expected_error = float(
            ((realized_norms - reference_norms).abs() / reference_norms).max().item()
        )
        self.assertEqual(
            result["maximum_center_norm_relative_error"],
            expected_error,
        )
        self.assertGreater(expected_error, 0.0)
        self.assertLess(expected_error, 1e-6)

    def test_independent_matching_gradient_improves_heldout_geometry(self):
        generator = torch.Generator().manual_seed(211)
        hidden = torch.randn(24, 6, generator=generator, dtype=torch.float64)
        centers = torch.randn(3, 6, generator=generator, dtype=torch.float64)
        assignments = torch.tensor([0, 1, 2] * 8)
        slopes = torch.randn(24, generator=generator, dtype=torch.float64)
        query_rcl = routing_contrastive_center_gradient(
            hidden,
            centers,
            assignments,
            0.07,
        )
        shuffled = []
        for shuffled_assignment in count_preserving_assignment_shuffles(
            assignments.numpy(),
            4,
            991,
        ):
            shuffled.append(routing_contrastive_center_gradient(
                hidden,
                centers,
                torch.from_numpy(shuffled_assignment),
                0.07,
            ))
        result = summarize_external_rcl_gradient_cell(
            hidden_states=hidden,
            centers=centers,
            assignments=assignments,
            responsibility_slopes=slopes,
            temperature=0.07,
            correct_support_rcl=query_rcl,
            shuffled_support_rcl=shuffled,
        )
        self.assertAlmostEqual(
            result["correct"]["heldout_geometry_alignment"],
            1.0,
            places=12,
        )
        self.assertGreater(
            result["correct"]["heldout_rcl_geometry_gain"],
            result["shuffle_summary"]["heldout_rcl_geometry_gain_mean"],
        )


if __name__ == "__main__":
    unittest.main()
