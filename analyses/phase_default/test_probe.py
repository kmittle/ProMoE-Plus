import unittest

import numpy as np
import torch

from analyses.phase_default.probe import (
    APPROXIMATION_NAMES,
    DefaultSketchAccumulator,
    MultiMoeCapture,
    _bootstrap_summary,
    _cell_contrasts,
    approximation_metrics,
    center_gradient_from_score_gradient,
)


class _ToyMoe(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, hidden_states, labels):
        return hidden_states * self.scale, None


class _ToyBlock(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.use_moe = True
        self.mlp = _ToyMoe(scale)


class _ToyCaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_ToyBlock(2.0), _ToyBlock(3.0)])

    def forward(self, hidden_states, labels):
        for block in self.blocks:
            hidden_states = block.mlp(hidden_states, labels)[0]
        return hidden_states


class _BiasExpert(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = float(bias)

    def forward(self, hidden_states):
        return hidden_states + self.bias


class PhaseDefaultProbeTests(unittest.TestCase):
    def test_center_gradient_matches_autograd(self):
        torch.manual_seed(3)
        hidden = torch.randn(5, 4, dtype=torch.float64)
        centers = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        score_gradient = torch.randn(5, 3, dtype=torch.float64)
        scores = torch.nn.functional.normalize(hidden, dim=-1) @ (
            torch.nn.functional.normalize(centers, dim=-1).T
        )
        expected, = torch.autograd.grad((scores * score_gradient).sum(), centers)
        observed = center_gradient_from_score_gradient(
            hidden, centers.detach(), score_gradient
        )
        self.assertTrue(torch.allclose(observed, expected, atol=1e-10, rtol=1e-10))

    def test_multi_capture_preserves_values_and_gets_both_suffix_gradients(self):
        model = _ToyCaptureModel()
        hidden = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
        labels = torch.tensor([1])
        expected = model(hidden, labels)
        capture = MultiMoeCapture(model, (0, 1))
        try:
            capture.start(gradient_mode=True)
            observed = model(hidden, labels)
            loss = observed.square().mean()
            gradients = capture.suffix_gradients(loss)
            capture.stop()
        finally:
            capture.close()
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(set(gradients), {0, 1})
        self.assertTrue(torch.isfinite(gradients[0]).all())
        self.assertTrue(torch.isfinite(gradients[1]).all())
        self.assertGreater(float(gradients[0].abs().sum()), 0.0)
        self.assertGreater(float(gradients[1].abs().sum()), 0.0)

    def test_sketch_uses_only_native_expert_rows_and_keeps_empty_zero(self):
        accumulator = DefaultSketchAccumulator(
            blocks=(1,), num_phases=2, num_experts=3, hidden_size=2
        )
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        routes = torch.tensor([0, 0, 1])
        experts = [_BiasExpert(0), _BiasExpert(10), _BiasExpert(20)]
        accumulator.update(1, 0, hidden, routes, experts)
        sketches = accumulator.finalize()
        self.assertEqual(sketches["global_counts"][1].tolist(), [2, 1, 0])
        self.assertEqual(sketches["phase_counts"][1][0].tolist(), [2, 1, 0])
        self.assertTrue(torch.equal(
            sketches["global_defaults"][1][0], torch.tensor([2.0, 3.0], dtype=torch.float64)
        ))
        self.assertTrue(torch.equal(
            sketches["global_defaults"][1][1], torch.tensor([15.0, 16.0], dtype=torch.float64)
        ))
        self.assertTrue(torch.equal(
            sketches["global_defaults"][1][2], torch.zeros(2, dtype=torch.float64)
        ))

    def test_phase_defaults_improve_output_and_gradient_approximations(self):
        torch.manual_seed(7)
        tokens, experts, hidden_size = 6, 3, 4
        hidden = torch.randn(tokens, hidden_size)
        centers = torch.randn(experts, hidden_size)
        phase_table = torch.randn(experts, hidden_size)
        exact = phase_table.unsqueeze(0) + 0.01 * torch.randn(
            tokens, experts, hidden_size
        )
        suffix = torch.ones(tokens, hidden_size)
        native = torch.tensor([0, 1, 2, 0, 1, 2])
        defaults = {
            "zero": torch.zeros_like(phase_table),
            "global": -phase_table,
            "phase": phase_table,
            "shuffled_phase": 2.0 * phase_table,
        }
        self.assertEqual(set(defaults), set(APPROXIMATION_NAMES))
        metrics = approximation_metrics(
            exact, suffix, native, hidden, centers, defaults
        )
        self.assertLess(
            metrics["phase"]["unselected_output_relative_squared_error"],
            metrics["global"]["unselected_output_relative_squared_error"],
        )
        self.assertGreater(
            metrics["phase"]["missing_score_gradient_cosine"],
            metrics["global"]["missing_score_gradient_cosine"],
        )
        self.assertGreater(
            metrics["phase"]["center_gradient_cosine"],
            metrics["global"]["center_gradient_cosine"],
        )
        contrasts = _cell_contrasts(metrics)
        self.assertGreater(contrasts["phase_vs_global_output_error_reduction"], 0)

    def test_bootstrap_is_image_level_and_deterministic(self):
        values = [1.0, 2.0, 4.0, 8.0]
        left = _bootstrap_summary(values, 2000, 17)
        right = _bootstrap_summary(values, 2000, 17)
        self.assertEqual(left, right)
        self.assertAlmostEqual(left["mean"], np.mean(values))
        self.assertEqual(left["image_values"], values)


if __name__ == "__main__":
    unittest.main()
