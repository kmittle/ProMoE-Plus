import unittest

import numpy as np
import torch

from analyses.expert_function.consistency_probe import (
    ALL_METRICS,
    PRIMARY_METRIC,
    _exact_route_grid,
    compute_function_scores,
    summarize_token,
    summarize_tokens,
)


class _FakeMoeLayer:
    def compute_router(self, hidden_states, labels):
        batch_size = hidden_states.shape[0]
        weights = torch.ones(batch_size, 2, 1)
        indices = torch.tensor([0, 1], dtype=torch.long).reshape(1, 2, 1)
        return weights, indices.repeat(batch_size, 1, 1), None


class _FakeRouteModel:
    def __init__(self, moe_layer):
        self.moe_layer = moe_layer
        self.batch_sizes = []

    def __call__(self, latent, timestep, context):
        self.batch_sizes.append(latent.shape[0])
        hidden_states = torch.zeros(latent.shape[0], 2, 1)
        _, indices, _ = self.moe_layer.compute_router(hidden_states, context)
        return indices[..., 0].sum(dim=1).float().reshape(-1, 1, 1, 1)


class ExpertFunctionConsistencyTests(unittest.TestCase):
    def test_exact_route_grid_forces_every_equal_compute_candidate(self):
        moe_layer = _FakeMoeLayer()
        model = _FakeRouteModel(moe_layer)
        changes, controls = _exact_route_grid(
            model=model,
            moe_layer=moe_layer,
            noised_latent=torch.zeros(1, 1, 1, 1, 1),
            timestep=torch.zeros(1),
            label=torch.zeros(1, dtype=torch.long),
            target=torch.zeros(1, 1, 1, 1),
            token_indices=torch.tensor([0, 1], dtype=torch.long),
            native_experts=torch.tensor([0, 1], dtype=torch.long),
            num_experts=3,
            batch_size=4,
            unforced_prediction=torch.ones(1, 1, 1, 1),
            unforced_loss=1.0,
        )
        np.testing.assert_allclose(
            changes.numpy(),
            np.asarray([[0.0, 3.0, 8.0], [-1.0, 0.0, 3.0]]),
        )
        self.assertEqual(controls["max_abs_noop_mse_change"], 0.0)
        self.assertEqual(controls["max_abs_noop_output_change"], 0.0)
        self.assertEqual(
            controls["max_abs_forced_unforced_output_change"],
            0.0,
        )
        self.assertEqual(
            controls["max_abs_forced_unforced_mse_change"],
            0.0,
        )
        self.assertEqual(model.batch_sizes, [1, 4, 4, 4])

    def test_exact_route_grid_requires_room_for_a_pair(self):
        moe_layer = _FakeMoeLayer()
        with self.assertRaisesRegex(ValueError, "paired forwards"):
            _exact_route_grid(
                model=_FakeRouteModel(moe_layer),
                moe_layer=moe_layer,
                noised_latent=torch.zeros(1, 1, 1, 1, 1),
                timestep=torch.zeros(1),
                label=torch.zeros(1, dtype=torch.long),
                target=torch.zeros(1, 1, 1, 1),
                token_indices=torch.tensor([0], dtype=torch.long),
                native_experts=torch.tensor([0], dtype=torch.long),
                num_experts=3,
                batch_size=1,
                unforced_prediction=torch.ones(1, 1, 1, 1),
                unforced_loss=1.0,
            )

    def test_shared_residual_content_advantage_tracks_content(self):
        original = torch.zeros(3, 2, 2)
        shifted = torch.zeros(3, 2, 2)
        original_shared = torch.zeros(3, 2)
        shifted_shared = torch.zeros(3, 2)

        original[0, 0] = torch.tensor([1.0, 0.0])
        original[1, 0] = torch.tensor([0.0, 1.0])
        shifted[1, 0] = torch.tensor([1.0, 0.0])
        original[0, 1] = torch.tensor([0.0, 1.0])
        original[1, 1] = torch.tensor([1.0, 0.0])
        shifted[1, 1] = torch.tensor([0.0, 1.0])

        scores = compute_function_scores(
            original,
            shifted,
            original_shared,
            shifted_shared,
            torch.tensor([1], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        )
        primary = scores[PRIMARY_METRIC][0]
        self.assertGreater(primary[0].item(), 0.9)
        self.assertGreater(primary[1].item(), 0.9)

    def test_summarize_token_uses_higher_score_as_higher_utility(self):
        exact = np.array([0.0, -2.0, 1.0])
        scores = {
            metric: np.array([0.0, 2.0, -1.0])
            for metric in ALL_METRICS
        }
        summary = summarize_token(scores, exact, native_expert=0)
        primary = summary["metrics"][PRIMARY_METRIC]
        self.assertEqual(primary["selected_expert"], 1)
        self.assertTrue(primary["selected_beats_native"])
        self.assertEqual(primary["selected_oracle_regret"], 0.0)
        self.assertEqual(primary["spearman_with_exact_utility"], 1.0)

    def test_summarize_tokens_reports_paired_router_difference(self):
        exact = np.array([0.0, -2.0, 1.0])
        scores = {
            metric: np.array([0.0, 2.0, -1.0])
            for metric in ALL_METRICS
        }
        scores["router_affinity"] = np.array([2.0, 0.0, 1.0])
        token = summarize_token(scores, exact, native_expert=0)
        aggregate = summarize_tokens([token, token])
        self.assertEqual(aggregate["num_tokens"], 2)
        self.assertGreater(
            aggregate["primary_minus_router_mean_spearman"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
