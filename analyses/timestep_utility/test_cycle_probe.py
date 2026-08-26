import unittest

import numpy as np
import torch

from analyses.timestep_utility.cycle_probe import (
    ARM_CANDIDATES,
    ARM_NAMES,
    AUDITED_SIX_CANDIDATES,
    _edge_first_order_grid,
    _exact_candidate_changes,
    _score_candidates,
    _summarize_six_audit,
    build_candidate_banks,
    summarize_arm,
)


class _ScaleExpert(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, hidden_states):
        return hidden_states * self.scale


class _ToyMoe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_routed_experts = 3
        self.top_k = 1
        self.experts = torch.nn.ModuleList([
            _ScaleExpert(1.0),
            _ScaleExpert(2.0),
            _ScaleExpert(3.0),
        ])

    def compute_router(self, hidden_states, labels):
        batch, tokens, _ = hidden_states.shape
        if tokens != 2:
            raise RuntimeError("Toy route template expects two tokens")
        indices = torch.tensor([0, 1], device=hidden_states.device).view(1, 2, 1)
        indices = indices.expand(batch, -1, -1).clone()
        weights = torch.tensor([0.5, 0.25], device=hidden_states.device)
        weights = weights.view(1, 2, 1).expand(batch, -1, -1).clone()
        return weights, indices, None

    def forward(self, hidden_states, labels):
        weights, indices, _ = self.compute_router(hidden_states, labels)
        output = torch.zeros_like(hidden_states)
        for expert_index, expert in enumerate(self.experts):
            selected = indices[..., 0] == expert_index
            if selected.any():
                output[selected] = (
                    expert(hidden_states[selected])
                    * weights[..., 0][selected].unsqueeze(-1)
                )
        return output


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = _ToyMoe()

    def forward(self, latent, timestep, context):
        hidden = latent[:, 0, 0, 0, :].unsqueeze(-1)
        output = self.moe(hidden, context)
        return output.squeeze(-1).unsqueeze(1).unsqueeze(2)


class CycleProbeTests(unittest.TestCase):
    def setUp(self):
        self.native = np.resize(np.arange(12, dtype=np.int64), 256)

    def test_candidate_banks_are_deterministic_and_obey_counts(self):
        first_banks, first_audits = build_candidate_banks(self.native, 12, 123)
        second_banks, second_audits = build_candidate_banks(self.native, 12, 123)
        self.assertEqual(first_banks, second_banks)
        self.assertEqual(first_audits, second_audits)
        self.assertEqual(set(first_banks), set(ARM_NAMES))
        for arm, candidates in first_banks.items():
            self.assertEqual(len(candidates), ARM_CANDIDATES)
            for candidate in candidates:
                self.assertEqual(
                    candidate["source_count_vector"]
                    == candidate["destination_count_vector"],
                    arm != "single_token",
                )
                self.assertEqual(
                    len(set(candidate["tokens"])),
                    candidate["changed_tokens"],
                )
        self.assertEqual(
            len(first_audits),
            AUDITED_SIX_CANDIDATES * 3,
        )

    def test_cycle_sizes_match_bipartite_graph_names(self):
        banks, _ = build_candidate_banks(self.native, 12, 456)
        self.assertTrue(all(
            candidate["changed_tokens"] == 2
            for candidate in banks["four_cycle"]
        ))
        self.assertTrue(all(
            candidate["changed_tokens"] == 3
            for candidate in banks["six_cycle"]
        ))
        self.assertEqual(
            sum(candidate["kind"] == "four_cycle" for candidate in banks["mixed_cycle"]),
            32,
        )
        self.assertEqual(
            sum(candidate["kind"] == "six_cycle" for candidate in banks["mixed_cycle"]),
            32,
        )

    def test_edge_first_order_grid_uses_native_weight(self):
        moe = _ToyMoe()
        hidden = torch.tensor([[2.0], [3.0]])
        gradient = torch.tensor([[5.0], [-2.0]])
        native_experts = torch.tensor([0, 1])
        native_weights = torch.tensor([0.5, 0.25])
        changes, gradient_norm, delta_norm = _edge_first_order_grid(
            moe,
            hidden,
            gradient,
            native_experts,
            native_weights,
        )
        self.assertAlmostEqual(float(changes[0, 1]), 5.0)
        self.assertAlmostEqual(float(changes[1, 0]), 1.5)
        self.assertAlmostEqual(float(gradient_norm[0]), 25.0)
        self.assertAlmostEqual(float(delta_norm[0, 1]), 1.0)

    def test_joint_score_sums_edges_once(self):
        candidate = {
            "id": "four_cycle:000",
            "arm": "four_cycle",
            "kind": "four_cycle",
            "tokens": [0, 1],
            "source_experts": [0, 1],
            "destination_experts": [1, 0],
            "count_preserving": True,
            "changed_tokens": 2,
            "source_count_vector": [1, 1],
            "destination_count_vector": [1, 1],
        }
        first_order = torch.tensor([[0.0, -2.0], [3.0, 0.0]])
        gradient_norm = torch.tensor([1.0, 4.0])
        delta_norm = torch.tensor([[0.0, 9.0], [16.0, 0.0]])
        router_scores = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        record = _score_candidates(
            [candidate],
            first_order,
            gradient_norm,
            delta_norm,
            router_scores,
        )[0]
        self.assertEqual(record["first_order_change"], 1.0)
        self.assertAlmostEqual(
            record["normalized_first_order_change"],
            1.0 / np.sqrt(125.0),
        )
        self.assertAlmostEqual(record["mean_router_margin"], 0.5)

    def test_exact_evaluator_changes_two_routes_in_one_forward(self):
        model = _ToyModel()
        latent = torch.tensor([[[[[2.0, 3.0]]]]])
        timestep = torch.tensor([0.5])
        label = torch.tensor([1])
        target = torch.zeros(1, 1, 1, 2)
        native_ids = torch.tensor([0, 1], dtype=torch.long)
        native_weights = torch.tensor([0.5, 0.25])
        with torch.inference_mode():
            native_prediction = model(latent, timestep, context=label)
        native_loss = (
            native_prediction.double() - target.double()
        ).square().flatten(1).mean(dim=1)[0]
        candidate = {
            "id": "four_cycle:000",
            "arm": "four_cycle",
            "kind": "four_cycle",
            "tokens": [0, 1],
            "source_experts": [0, 1],
            "destination_experts": [1, 0],
            "count_preserving": True,
            "changed_tokens": 2,
            "source_count_vector": [1, 1, 0],
            "destination_count_vector": [1, 1, 0],
            "first_order_change": 0.0,
            "normalized_first_order_change": 0.0,
            "mean_router_margin": 0.0,
        }
        records, controls = _exact_candidate_changes(
            model=model,
            moe_layer=model.moe,
            noised_latent=latent,
            timestep=timestep,
            label=label,
            target=target,
            native_route_ids=native_ids,
            native_route_weights=native_weights,
            native_prediction=native_prediction,
            native_loss=native_loss,
            candidates=[candidate],
            exact_batch_size=2,
        )
        self.assertEqual(len(records), 1)
        self.assertNotEqual(records[0]["exact_mse_change"], 0.0)
        self.assertEqual(controls["max_abs_noop_mse_change"], 0.0)
        self.assertEqual(controls["max_abs_noop_output_change"], 0.0)

    def test_arm_summary_keeps_native_when_prediction_has_no_gain(self):
        records = []
        for index in range(ARM_CANDIDATES):
            records.append({
                "first_order_change": 1.0 + index,
                "exact_mse_change": -0.1,
                "changed_tokens": 2,
            })
        summary = summarize_arm(records, native_mse=2.0, epsilon_num=1e-8)
        self.assertIsNone(summary["selected_candidate_index"])
        self.assertEqual(summary["selected_gain"], 0.0)
        self.assertEqual(summary["selected_changed_tokens"], 0)

    def test_unique_six_requires_all_direct_pairs_nonbeneficial(self):
        six_records = []
        audit_records = []
        for index in range(ARM_CANDIDATES):
            six_id = f"six_cycle:{index:03d}"
            six_records.append({
                "id": six_id,
                "exact_mse_change_relative": -0.2 if index == 0 else 0.1,
            })
            if index < AUDITED_SIX_CANDIDATES:
                for pair in range(3):
                    audit_records.append({
                        "parent_six_id": six_id,
                        "exact_mse_change_relative": 0.1,
                    })
        summary = _summarize_six_audit(
            six_records,
            audit_records,
            epsilon_num=1e-8,
        )
        self.assertEqual(summary["unique_six_rate"], 1 / 16)
        self.assertTrue(summary["has_unique_six"])


if __name__ == "__main__":
    unittest.main()
