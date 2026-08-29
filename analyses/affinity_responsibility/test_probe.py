"""Tests for fixed-dispatch prototype interventions."""

import unittest

import torch
import torch.nn.functional as F

from analyses.affinity_responsibility.probe import (
    _exact_center_weight_changes,
    _exact_global_weight_changes,
    _exact_token_weight_changes,
    _fixed_dispatch_center_weights,
    aggregate_rank_support_rcl,
)
from analyses.affinity_responsibility.protocol import (
    ASSIGNMENT_SHUFFLE_COUNT,
    BLOCK_INDICES,
    SUPPORT_GROUP_COUNT,
)


class _FakeMoe(torch.nn.Module):
    def __init__(self, centers):
        super().__init__()
        self.cluster_centers = torch.nn.Parameter(centers.clone())

    def compute_router(self, hidden_states, labels):
        scores = F.normalize(hidden_states, dim=-1) @ F.normalize(
            self.cluster_centers,
            dim=-1,
        ).T
        weights, indices = scores.topk(1, dim=-1)
        return weights, indices, None


class _FakeModel(torch.nn.Module):
    def __init__(self, moe, hidden):
        super().__init__()
        self.moe = moe
        self.register_buffer("hidden", hidden[:1].clone())

    def forward(self, latent, timestep, context):
        hidden = self.hidden.expand(latent.shape[0], -1, -1)
        weights, _, _ = self.moe.compute_router(hidden, context)
        prediction = weights[:, :, 0].sum(dim=1).view(-1, 1, 1, 1)
        return prediction


class FixedDispatchCenterWeightTest(unittest.TestCase):
    def setUp(self):
        self.centers = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.hidden = torch.tensor(
            [
                [[0.9, 0.1, 0.0], [0.2, 0.8, 0.0]],
                [[0.9, 0.1, 0.0], [0.2, 0.8, 0.0]],
            ],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([1, 1])
        self.expected = torch.tensor([0, 1])

    def test_selected_cosines_change_without_changing_expert_ids(self):
        moe = _FakeMoe(self.centers)
        center_batch = self.centers.unsqueeze(0).repeat(2, 1, 1)
        center_batch[1, 0] = torch.tensor([0.8, 0.6, 0.0])
        with _fixed_dispatch_center_weights(
            moe,
            center_batch,
            self.expected,
        ) as statistics:
            weights, indices, _ = moe.compute_router(self.hidden, self.labels)
        torch.testing.assert_close(
            indices[:, :, 0],
            self.expected.unsqueeze(0).expand(2, -1),
        )
        self.assertEqual(statistics["fixed_dispatch_mismatches"], 0)
        native = F.cosine_similarity(self.hidden[0, 0], self.centers[0], dim=0)
        changed = F.cosine_similarity(self.hidden[1, 0], center_batch[1, 0], dim=0)
        self.assertAlmostEqual(weights[0, 0, 0].item(), native.item(), places=6)
        self.assertAlmostEqual(weights[1, 0, 0].item(), changed.item(), places=6)
        self.assertNotAlmostEqual(native.item(), changed.item(), places=4)

    def test_wrong_native_assignment_is_rejected(self):
        moe = _FakeMoe(self.centers)
        with self.assertRaisesRegex(RuntimeError, "Native dispatch changed"):
            with _fixed_dispatch_center_weights(
                moe,
                self.centers.unsqueeze(0).repeat(2, 1, 1),
                torch.tensor([1, 0]),
            ):
                moe.compute_router(self.hidden, self.labels)

    def test_exact_helper_pairs_noop_and_center_intervention(self):
        moe = _FakeMoe(self.centers)
        model = _FakeModel(moe, self.hidden)
        changed_centers = self.centers.clone()
        changed_centers[0] = torch.tensor([0.8, 0.6, 0.0])
        result = _exact_center_weight_changes(
            model=model,
            moe_layer=moe,
            noised_latent=torch.zeros(1, 1, 1, 1, 1),
            timestep=torch.zeros(1),
            label=torch.ones(1, dtype=torch.long),
            target=torch.zeros(1, 1, 1, 1),
            expected_assignments=self.expected,
            named_centers=(
                ("noop", self.centers),
                ("changed", changed_centers),
            ),
            batch_size=2,
        )
        self.assertAlmostEqual(result["changes"]["noop"], 0.0, places=12)
        self.assertNotAlmostEqual(result["changes"]["changed"], 0.0, places=8)
        self.assertEqual(result["fixed_dispatch_mismatches"], 0)

    def test_token_scale_intervention_rejects_changed_expert_ids(self):
        moe = _FakeMoe(self.centers)
        model = _FakeModel(moe, self.hidden)
        with self.assertRaisesRegex(RuntimeError, "changed the native expert IDs"):
            _exact_token_weight_changes(
                model=model,
                moe_layer=moe,
                noised_latent=torch.zeros(1, 1, 1, 1, 1),
                timestep=torch.zeros(1),
                label=torch.ones(1, dtype=torch.long),
                target=torch.zeros(1, 1, 1, 1),
                token_indices=torch.tensor([0, 1]),
                route_weights=torch.tensor([0.25, 0.25]),
                batch_size=2,
                expected_expert_indices=torch.tensor([1, 0]),
                dispatch_statistics={"fixed_dispatch_mismatches": 0},
            )

    def test_global_scale_intervention_records_fixed_dispatch(self):
        moe = _FakeMoe(self.centers)
        model = _FakeModel(moe, self.hidden)
        statistics = {"fixed_dispatch_mismatches": 0}
        changes = _exact_global_weight_changes(
            model=model,
            moe_layer=moe,
            noised_latent=torch.zeros(1, 1, 1, 1, 1),
            timestep=torch.zeros(1),
            label=torch.ones(1, dtype=torch.long),
            target=torch.zeros(1, 1, 1, 1),
            route_weight_matrix=torch.tensor([[0.25, 0.25]]),
            batch_size=1,
            expected_expert_indices=self.expected,
            dispatch_statistics=statistics,
        )
        self.assertEqual(tuple(changes.shape), (1,))
        self.assertEqual(statistics["fixed_dispatch_mismatches"], 0)


class DdpSupportAggregationTest(unittest.TestCase):
    def _rank_result(self, group_index):
        def record(offset):
            return {
                "loss": float(group_index + offset),
                "gradient": torch.full((2, 3), float(group_index + offset)).numpy(),
                "valid_experts": [0, 1],
                "assignment_count_mismatches": 0,
                "occupied_expert_count": 2,
            }

        return {
            block_index: {
                "correct": record(0),
                "shuffled": [
                    record(index + 1) for index in range(ASSIGNMENT_SHUFFLE_COUNT)
                ],
                "support_group_index": group_index,
                "support_image_count": 64,
                "support_conditional_image_count": 58,
                "support_token_count": 58 * 256,
                "support_sigma_min": 0.1 + group_index * 0.01,
                "support_sigma_max": 0.9 - group_index * 0.01,
                "support_sigma_mean": 0.5,
                "shuffle_seed": 100 + group_index,
            }
            for block_index in BLOCK_INDICES
        }

    def test_four_rank_gradients_are_ddp_averaged(self):
        result = aggregate_rank_support_rcl({
            group: self._rank_result(group)
            for group in range(SUPPORT_GROUP_COUNT)
        })
        first = result[BLOCK_INDICES[0]]
        self.assertEqual(first["support_gradient_aggregation"], "ddp_mean")
        self.assertEqual(first["support_image_count"], 256)
        self.assertEqual(first["support_group_indices"], [0, 1, 2, 3])
        torch.testing.assert_close(
            torch.from_numpy(first["correct"]["gradient"]),
            torch.full((2, 3), 1.5, dtype=torch.float64),
        )

    def test_missing_rank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "all four"):
            aggregate_rank_support_rcl({
                group: self._rank_result(group)
                for group in range(SUPPORT_GROUP_COUNT - 1)
            })


if __name__ == "__main__":
    unittest.main()
