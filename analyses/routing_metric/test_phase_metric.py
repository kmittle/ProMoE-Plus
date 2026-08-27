import unittest

import torch
import torch.nn.functional as F

from models.phase_metric import PhaseConditionedRoutingMetric


class PhaseMetricTest(unittest.TestCase):
    @staticmethod
    def _inputs():
        generator = torch.Generator().manual_seed(41)
        tokens = F.normalize(
            torch.randn(7, 16, generator=generator), dim=-1
        )
        prototypes = F.normalize(
            torch.randn(4, 16, generator=generator), dim=-1
        )
        timesteps = torch.tensor([50, 150, 300, 500, 700, 850, 950])
        return tokens, prototypes, timesteps

    def test_zero_phase_projection_is_an_exact_zero_residual(self):
        metric = PhaseConditionedRoutingMetric(
            hidden_size=16,
            num_experts=4,
            rank=4,
            num_fourier_bands=2,
        )
        tokens, prototypes, timesteps = self._inputs()
        residual = metric(tokens, prototypes, timesteps)
        torch.testing.assert_close(
            residual,
            torch.zeros_like(residual),
            rtol=0,
            atol=0,
        )

    def test_phase_projection_receives_task_gradient_at_initialization(self):
        metric = PhaseConditionedRoutingMetric(
            hidden_size=16,
            num_experts=4,
            rank=4,
            num_fourier_bands=2,
        )
        tokens, prototypes, timesteps = self._inputs()
        residual = metric(tokens, prototypes, timesteps)
        residual[0, 0].backward()
        self.assertIsNotNone(metric.phase_to_rank.grad)
        self.assertGreater(metric.phase_to_rank.grad.norm().item(), 0.0)
        self.assertEqual(metric.token_basis.grad.norm().item(), 0.0)
        self.assertEqual(metric.prototype_basis.grad.norm().item(), 0.0)

    def test_learned_metric_can_distinguish_diffusion_phases(self):
        metric = PhaseConditionedRoutingMetric(
            hidden_size=16,
            num_experts=4,
            rank=4,
            num_fourier_bands=2,
        )
        tokens, prototypes, _ = self._inputs()
        with torch.no_grad():
            metric.phase_to_rank[0, 0] = 1.0
            metric.phase_to_rank[1, 1] = -0.7
        early = metric(
            tokens[:1].expand(2, -1),
            prototypes,
            torch.tensor([100, 900]),
        )
        self.assertGreater((early[0] - early[1]).abs().max().item(), 1e-6)

    def test_private_initialization_preserves_global_rng(self):
        torch.manual_seed(1234)
        expected = torch.rand(5)
        torch.manual_seed(1234)
        PhaseConditionedRoutingMetric(
            hidden_size=16,
            num_experts=4,
            rank=4,
            num_fourier_bands=2,
        )
        actual = torch.rand(5)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_autocast_keeps_residual_in_float32(self):
        metric = PhaseConditionedRoutingMetric(
            hidden_size=16,
            num_experts=4,
            rank=4,
            num_fourier_bands=2,
        )
        tokens, prototypes, timesteps = self._inputs()
        with torch.no_grad():
            metric.phase_to_rank[0, 0] = 1.0
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            residual = metric(tokens, prototypes, timesteps)
        self.assertEqual(residual.dtype, torch.float32)
        self.assertGreater(residual.abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
