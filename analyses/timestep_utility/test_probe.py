import unittest

import numpy as np
import torch

from analyses.timestep_utility.probe import (
    _build_assignments,
    _exact_route_grid,
    _forced_route_state,
    _load_statistics,
    _pairwise_order_inversion,
    _solve_capacity_assignment,
    _summarize_stage_dynamics,
    _summarize_token,
)


class _ToyMoe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_routed_experts = 3
        self.top_k = 1

    def compute_router(self, hidden_states, labels):
        batch, tokens, _ = hidden_states.shape
        weights = torch.full((batch, tokens, 1), 0.5, device=hidden_states.device)
        indices = torch.zeros((batch, tokens, 1), dtype=torch.long, device=hidden_states.device)
        return weights, indices, None

    def forward(self, hidden_states, labels):
        weights, indices, _ = self.compute_router(hidden_states, labels)
        factors = indices[..., 0].to(hidden_states.dtype) + 1.0
        return hidden_states * weights[..., 0:1] * factors.unsqueeze(-1)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = _ToyMoe()

    def forward(self, latent, timestep, context):
        hidden = latent[:, 0, 0, 0, :].unsqueeze(-1)
        output = self.moe(hidden, context)
        return output.squeeze(-1).unsqueeze(1).unsqueeze(2)


class TimestepUtilityProbeTests(unittest.TestCase):
    def test_forced_route_state_replaces_ids_and_weights_then_restores(self):
        moe = _ToyMoe()
        hidden = torch.ones(2, 2, 1)
        labels = torch.tensor([1, 2])
        original = moe.compute_router.__func__
        route_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
        route_weights = torch.tensor([[0.2, 0.3], [0.4, 0.6]])
        with _forced_route_state(moe, route_ids, route_weights):
            weights, indices, _ = moe.compute_router(hidden, labels)
            self.assertTrue(torch.equal(indices[..., 0], route_ids))
            self.assertTrue(torch.equal(weights[..., 0], route_weights))
        self.assertIs(moe.compute_router.__func__, original)

    def test_exact_route_grid_pairs_all_weight_semantics(self):
        model = _ToyModel()
        latent = torch.tensor([[[[[2.0, 3.0]]]]])
        timestep = torch.tensor([0.5])
        label = torch.tensor([0])
        target = torch.zeros(1, 1, 1, 2)
        token_indices = torch.tensor([0], dtype=torch.long)
        native_ids = torch.tensor([0, 0], dtype=torch.long)
        native_weights = torch.tensor([0.5, 0.5])
        router_scores = torch.tensor([[0.5, 0.25, 1.0]])
        matrices, controls = _exact_route_grid(
            model=model,
            moe_layer=model.moe,
            noised_latent=latent,
            timestep=timestep,
            label=label,
            target=target,
            token_indices=token_indices,
            native_route_ids=native_ids,
            native_route_weights=native_weights,
            router_scores=router_scores,
            num_experts=3,
            batch_size=6,
            weight_modes=("native", "candidate", "unit"),
        )
        for mode in ("native", "candidate", "unit"):
            self.assertEqual(tuple(matrices[mode].shape), (1, 3))
            self.assertEqual(float(matrices[mode][0, 0]), 0.0)
            self.assertEqual(controls[mode]["max_abs_noop_mse_change"], 0.0)
            self.assertEqual(controls[mode]["max_abs_noop_output_change"], 0.0)
        self.assertGreater(float(matrices["native"][0, 2]), 0.0)
        self.assertGreater(float(matrices["candidate"][0, 2]), float(matrices["native"][0, 2]))

    def test_capacity_assignment_obeys_exact_counts(self):
        costs = np.array([
            [0.0, 3.0, 2.0],
            [2.0, 0.0, 3.0],
            [3.0, 2.0, 0.0],
            [0.5, 0.6, 0.7],
        ])
        assignment = _solve_capacity_assignment(costs, [2, 1, 1])
        self.assertEqual(np.bincount(assignment, minlength=3).tolist(), [2, 1, 1])
        self.assertEqual(assignment.tolist(), [0, 1, 2, 0])

    def test_build_assignments_preserves_native_capacity(self):
        changes = np.array([
            [0.0, -2.0, -1.0],
            [-2.0, 0.0, -1.0],
            [-1.0, -2.0, 0.0],
            [0.0, -1.0, -2.0],
        ])
        native = np.array([0, 0, 1, 2])
        assignments, spec = _build_assignments(changes, native, 1.25)
        self.assertEqual(
            np.bincount(assignments["native_capacity_oracle"], minlength=3).tolist(),
            [2, 1, 1],
        )
        self.assertGreaterEqual(spec["balanced_capacity_per_expert"], 2)

    def test_load_statistics_include_zero_load_experts(self):
        stats = _load_statistics([0, 0, 1, 1], 4)
        self.assertEqual(stats["counts"], [2, 2, 0, 0])
        self.assertEqual(stats["active_experts"], 2)
        self.assertAlmostEqual(stats["cv"], 1.0)
        self.assertAlmostEqual(stats["gini"], 0.5)

    def test_pairwise_order_inversion(self):
        self.assertEqual(_pairwise_order_inversion([3, 2, 1], [1, 2, 3]), 1.0)
        self.assertEqual(_pairwise_order_inversion([3, 2, 1], [3, 2, 1]), 0.0)

    def test_token_summary_uses_lower_mse_as_higher_utility(self):
        summary = _summarize_token(
            [0.5, 0.9, 0.1],
            [0.0, -0.2, 0.3],
            native_expert=0,
            base_mse=2.0,
        )
        self.assertEqual(summary["oracle_expert"], 1)
        self.assertAlmostEqual(summary["native_regret"], 0.2)
        self.assertAlmostEqual(summary["native_regret_relative"], 0.1)
        self.assertEqual(summary["router_utility_spearman"], 1.0)

    def test_stage_summary_pairs_fixed_tokens(self):
        cells = []
        for sigma, exact, router, native, oracle in (
            (0.2, [0.0, 1.0, 2.0], [3.0, 2.0, 1.0], 0, 0),
            (0.8, [2.0, 1.0, 0.0], [3.0, 2.0, 1.0], 0, 2),
        ):
            cells.append({
                "block_index": 3,
                "sigma": sigma,
                "tokens": [{
                    "token_index": 7,
                    "exact_mse_changes": exact,
                    "router_scores": router,
                    "native_expert": native,
                    "oracle_expert": oracle,
                }],
            })
        result = _summarize_stage_dynamics(cells, (0.2, 0.8))
        summary = result["summary"]
        self.assertEqual(summary["oracle_expert_flip_rate"], 1.0)
        self.assertEqual(summary["native_expert_flip_rate"], 0.0)
        self.assertEqual(summary["mean_utility_pair_inversion_rate"], 1.0)
        self.assertGreater(summary["mean_router_minus_utility_rank_stability"], 0)

    def test_single_sigma_stage_summary_uses_null_rates(self):
        cells = [{
            "block_index": 3,
            "sigma": 0.5,
            "tokens": [{
                "token_index": 7,
                "exact_mse_changes": [0.0, 1.0, 2.0],
                "router_scores": [3.0, 2.0, 1.0],
                "native_expert": 0,
                "oracle_expert": 0,
            }],
        }]
        summary = _summarize_stage_dynamics(cells, (0.5,))["summary"]
        self.assertEqual(summary["num_paired_token_comparisons"], 0)
        self.assertIsNone(summary["oracle_expert_flip_rate"])
        self.assertIsNone(summary["native_expert_flip_rate"])


if __name__ == "__main__":
    unittest.main()
