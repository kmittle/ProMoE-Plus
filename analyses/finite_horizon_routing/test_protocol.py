"""Tests for the finite-horizon routing protocol."""

import unittest

import numpy as np
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from sample import get_sampling_sigmas

from analyses.finite_horizon_routing.protocol import (
    HORIZONS,
    analytic_flow_state,
    euler_flow_step,
    sampling_sigmas,
    summarize_cell_records,
    validate_count_preserving_candidates,
    validate_schedule_positions,
)


class ScheduleTest(unittest.TestCase):
    def test_locked_grid_contains_expected_positions_and_terminal_zero(self):
        sigmas = sampling_sigmas()
        self.assertEqual(sigmas.shape, (251,))
        self.assertEqual(sigmas.dtype, np.float32)
        self.assertEqual(sigmas[0], 1.0)
        self.assertEqual(sigmas[-1], 0.0)
        np.testing.assert_allclose(sigmas[[50, 125, 200]], [0.8, 0.5, 0.2])
        self.assertEqual(validate_schedule_positions(sigmas)[1], HORIZONS)

    def test_locked_grid_matches_the_installed_sampling_scheduler(self):
        sigmas = sampling_sigmas()
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
        )
        scheduler.set_timesteps(
            sigmas=get_sampling_sigmas(250, 1.0),
            device="cpu",
        )
        np.testing.assert_array_equal(
            scheduler.sigmas.cpu().numpy(),
            sigmas,
        )
        np.testing.assert_array_equal(
            scheduler.timesteps.cpu().numpy(),
            sigmas[:-1] * 1000,
        )

    def test_euler_step_is_bitwise_equal_to_installed_scheduler(self):
        sigmas = sampling_sigmas()
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
        )
        scheduler.set_timesteps(
            sigmas=get_sampling_sigmas(250, 1.0),
            device="cpu",
        )
        scheduler.set_begin_index(50)
        state = torch.tensor([[1.25, -0.75]], dtype=torch.float32)
        velocity = torch.tensor([[0.125, 2.5]], dtype=torch.float32)
        expected = scheduler.step(
            velocity,
            scheduler.timesteps[50],
            state,
            return_dict=False,
        )[0]
        observed = euler_flow_step(
            state,
            velocity,
            sigmas[50],
            sigmas[51],
        )
        self.assertTrue(torch.equal(observed, expected))

    def test_exact_velocity_stays_on_analytic_path(self):
        clean = torch.tensor([[[1.0, -2.0]]])
        noise = torch.tensor([[[3.0, 4.0]]])
        velocity = noise - clean
        state = analytic_flow_state(clean, noise, 0.8)
        actual = euler_flow_step(state, velocity, 0.8, 0.796)
        expected = analytic_flow_state(clean, noise, 0.796)
        torch.testing.assert_close(actual, expected)

    def test_one_step_state_error_is_step_squared_velocity_error(self):
        state = torch.zeros(2, 3)
        truth = torch.ones(2, 3)
        prediction = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]])
        step = -0.004
        target_state = state + step * truth
        predicted_state = euler_flow_step(state, prediction, 0.8, 0.796)
        state_mse = (predicted_state - target_state).square().mean(dim=1)
        velocity_mse = (prediction - truth).square().mean(dim=1)
        torch.testing.assert_close(state_mse, (step ** 2) * velocity_mse)


class CandidateTest(unittest.TestCase):
    def test_pair_swap_preserves_full_counts(self):
        native = np.asarray([0, 0, 1, 2], dtype=np.int64)
        candidates = [{
            "id": "swap",
            "tokens": [1, 2],
            "source_experts": [0, 1],
            "destination_experts": [1, 0],
        }]
        result = validate_count_preserving_candidates(candidates, native, 3)
        self.assertTrue(result[0]["full_count_match"])
        self.assertEqual(result[0]["full_native_count_vector"], [2, 1, 1])
        self.assertEqual(result[0]["full_candidate_count_vector"], [2, 1, 1])

    def test_single_token_change_is_rejected(self):
        native = np.asarray([0, 0, 1, 2], dtype=np.int64)
        candidates = [{
            "id": "invalid",
            "tokens": [0, 1],
            "source_experts": [0, 0],
            "destination_experts": [1, 0],
        }]
        with self.assertRaisesRegex(ValueError, "different valid expert"):
            validate_count_preserving_candidates(candidates, native, 3)


class SummaryTest(unittest.TestCase):
    def test_summary_detects_reversed_future_ranking(self):
        records = []
        for index in range(8):
            immediate = float(index + 1) / 100.0
            record = {
                "id": f"candidate-{index}",
                "mean_router_margin": -immediate,
                "immediate_gain_relative": immediate,
            }
            for horizon in HORIZONS:
                record[f"h{horizon}_gain_relative"] = (
                    immediate if horizon < 8 else -immediate
                )
            records.append(record)
        summary = summarize_cell_records(records, numerical_epsilon=1e-8)
        self.assertAlmostEqual(
            summary["per_horizon"]["1"]["immediate_future_spearman"],
            1.0,
        )
        self.assertAlmostEqual(
            summary["per_horizon"]["8"]["immediate_future_spearman"],
            -1.0,
        )
        self.assertEqual(
            summary["per_horizon"]["8"]["sign_disagreement"]["rate"],
            1.0,
        )
        self.assertFalse(summary["per_horizon"]["8"]["best_candidate_matches"])


if __name__ == "__main__":
    unittest.main()
