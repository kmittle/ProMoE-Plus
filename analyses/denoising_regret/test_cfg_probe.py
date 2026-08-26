import unittest

import torch

from analyses.denoising_regret.cfg_probe import (
    _guidance_metrics,
    _validate_guidance_scales,
    summarize_cfg_records,
)


class GuidanceMetricTests(unittest.TestCase):
    def test_scale_one_matches_conditional_and_projection_is_exact(self):
        conditional = torch.tensor([[[[2.0, 0.0]]]])
        unconditional = torch.tensor([[[[0.0, 0.0]]]])
        target = torch.tensor([[[[1.0, 0.0]]]])

        metrics = _guidance_metrics(
            conditional,
            unconditional,
            target,
            [1.0, 1.5],
        )

        self.assertEqual(
            metrics["guided_mse"]["1.0"].item(),
            0.5,
        )
        self.assertAlmostEqual(metrics["alignment"].item(), 1.0)
        self.assertAlmostEqual(metrics["optimal_scale"].item(), 0.5)
        self.assertAlmostEqual(metrics["projection_mse"].item(), 0.0)

    def test_guidance_scale_contract(self):
        scales, analysis_scale = _validate_guidance_scales([1, 1.5], 1.5)
        self.assertEqual(scales, [1.0, 1.5])
        self.assertEqual(analysis_scale, 1.5)
        with self.assertRaisesRegex(ValueError, "unique"):
            _validate_guidance_scales([1.0, 1.0], 1.0)
        with self.assertRaisesRegex(ValueError, "present"):
            _validate_guidance_scales([1.0], 1.5)

    def test_summary_counts_both_inversion_directions(self):
        records = [
            {
                "conditional_exact_mse_change": -2.0,
                "conditional_first_order_change": -1.0,
                "guided_exact_mse_change": {"1.5": 3.0},
                "guided_first_order_change": {"1.5": 2.0},
                "guidance_alignment_change": -0.1,
                "guidance_projection_mse_change": 0.2,
            },
            {
                "conditional_exact_mse_change": 2.0,
                "conditional_first_order_change": 1.0,
                "guided_exact_mse_change": {"1.5": -3.0},
                "guided_first_order_change": {"1.5": -2.0},
                "guidance_alignment_change": 0.1,
                "guidance_projection_mse_change": -0.2,
            },
        ]

        summary = summarize_cfg_records(records, 1.5)

        self.assertEqual(summary["route_inversion_rate"], 1.0)
        self.assertEqual(
            summary["conditional_better_guided_worse_rate"],
            0.5,
        )
        self.assertEqual(
            summary["conditional_worse_guided_better_rate"],
            0.5,
        )
        self.assertEqual(summary["guidance_alignment_improved_rate"], 0.5)
        self.assertEqual(summary["guidance_projection_improved_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
