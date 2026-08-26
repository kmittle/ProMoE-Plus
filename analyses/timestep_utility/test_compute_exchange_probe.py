import unittest

import numpy as np
import torch

from analyses.timestep_utility.compute_exchange_probe import (
    CANDIDATE_COUNT,
    _ComputeExchangeInjector,
    _exact_candidate_changes,
    _exchange_components,
    _forced_compute_exchange_state,
    _logical_pass_counts,
    _score_candidates,
    _validate_candidate,
    build_same_expert_exchange_candidates,
    summarize_selectors,
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
        self.hidden_size = 1
        self.num_routed_experts = 2
        self.num_experts = 3
        self.top_k = 1
        self.router_weight_mode = "identity"
        self.use_shared_expert = True
        self.experts = torch.nn.ModuleList([
            _ScaleExpert(2.0),
            _ScaleExpert(3.0),
            _ScaleExpert(4.0),
        ])
        self.shared_expert = _ScaleExpert(0.1)

    def compute_router(self, hidden_states, labels):
        batch, tokens, _ = hidden_states.shape
        if tokens != 4:
            raise RuntimeError("Toy router expects four tokens")
        ids = torch.tensor([0, 0, 1, 1], device=hidden_states.device)
        ids = ids.view(1, 4, 1).expand(batch, -1, -1).clone()
        weights = torch.tensor(
            [-0.5, 0.25, 0.4, 0.2],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        weights = weights.view(1, 4, 1).expand(batch, -1, -1).clone()
        return weights, ids, None

    def forward(self, hidden_states, labels):
        weights, indices, auxiliary_loss = self.compute_router(hidden_states, labels)
        flat_hidden = hidden_states.reshape(-1, 1)
        flat_weights = weights.reshape(-1)
        flat_ids = indices.reshape(-1)
        output = torch.zeros_like(flat_hidden)
        for expert_id, expert in enumerate(self.experts):
            selected = flat_ids == expert_id
            positions = torch.where(selected)[0]
            if positions.numel():
                values = expert(flat_hidden[positions]) * flat_weights[positions, None]
                output.index_add_(0, positions, values)
            else:
                output[0] += expert(torch.zeros_like(flat_hidden[:1]))[0] * 0
        output = output.reshape_as(hidden_states)
        output += self.shared_expert(hidden_states)
        return output, auxiliary_loss


class _BatchSensitiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = _ToyMoe()

    def forward(self, latent, timestep, context):
        batch = latent.shape[0]
        hidden = torch.tensor(
            [2.0, 3.0, 4.0, 5.0],
            device=latent.device,
            dtype=latent.dtype,
        ).view(1, 4, 1).expand(batch, -1, -1)
        output, _ = self.moe(hidden, context)
        prediction = output.mean(dim=1).reshape(batch, 1, 1, 1)
        return prediction + (0.01 if batch == 1 else 0.0)


def _toy_candidate():
    return {
        "id": "exchange:000",
        "donors": [0, 2],
        "receivers": [1, 3],
        "experts": [0, 1],
        "quota": 0.5,
        "quota_by_expert": [1, 1],
        "transferred_passes": 2,
        "native_pass_vector": [2, 2],
        "candidate_pass_vector": [2, 2],
    }


class ComputeExchangeProbeTests(unittest.TestCase):
    def test_candidate_bank_is_deterministic_and_count_preserving(self):
        native = np.resize(np.arange(12, dtype=np.int64), 256)
        first = build_same_expert_exchange_candidates(native, 12, 123)
        second = build_same_expert_exchange_candidates(native, 12, 123)
        other = build_same_expert_exchange_candidates(native, 12, 124)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), CANDIDATE_COUNT)
        self.assertEqual(len({
            tuple(zip(item["donors"], item["receivers"], item["experts"]))
            for item in first
        }), CANDIDATE_COUNT)
        for candidate in first:
            _validate_candidate(candidate, native, 12)
            self.assertEqual(
                candidate["native_pass_vector"],
                candidate["candidate_pass_vector"],
            )

    def test_low_load_experts_are_ineligible_without_breaking_counts(self):
        native = np.repeat(np.arange(3), [1, 127, 128])
        candidates = build_same_expert_exchange_candidates(native, 3, 4)
        for candidate in candidates:
            self.assertEqual(candidate["quota_by_expert"][0], 0)
            self.assertNotIn(0, candidate["experts"])
            baseline, action = _logical_pass_counts(
                native,
                candidate["donors"],
                candidate["receivers"],
                3,
            )
            np.testing.assert_array_equal(baseline, action)

    def test_candidate_rejects_overlap_and_cross_expert_pairs(self):
        native = np.array([0, 0, 1, 1])
        overlap = {**_toy_candidate(), "receivers": [0, 3]}
        with self.assertRaisesRegex(ValueError, "disjoint"):
            _validate_candidate(overlap, native, 2)
        cross = {**_toy_candidate(), "receivers": [3, 1]}
        with self.assertRaisesRegex(ValueError, "expert metadata"):
            _validate_candidate(cross, native, 2)

    def test_exact_hook_skips_and_repeats_same_expert_with_negative_weight(self):
        moe = _ToyMoe().eval().requires_grad_(False)
        hidden = torch.tensor([[[2.0], [3.0], [4.0], [5.0]]]).repeat(2, 1, 1)
        labels = torch.tensor([1, 1])
        route_weights, route_ids, _ = moe.compute_router(hidden, labels)
        candidate = _toy_candidate()
        with torch.inference_mode():
            native, _ = moe(hidden, labels)
            with _forced_compute_exchange_state(
                moe,
                route_ids[..., 0],
                route_weights[..., 0],
                (None, candidate),
            ) as injector:
                patched, _ = moe(hidden, labels)

        torch.testing.assert_close(patched[0], native[0], rtol=0, atol=0)
        expected = native[1].clone()
        expected[0] -= -0.5 * (2.0 * 2.0)
        expected[2] -= 0.4 * (3.0 * 4.0)
        expected[1] += 0.25 * (2.0 * (3.0 + 0.25 * 2.0 * 3.0))
        expected[3] += 0.2 * (3.0 * (5.0 + 0.2 * 3.0 * 5.0))
        torch.testing.assert_close(patched[1], expected, rtol=0, atol=1e-6)
        self.assertEqual(
            injector.second_pass_shapes,
            [[1, 0, [1, 1], [1, 1]], [1, 1, [1, 1], [1, 1]]],
        )
        self.assertFalse(hasattr(moe, "_compute_exchange_probe_active"))

    def test_noop_hook_is_exact_and_nested_override_is_rejected(self):
        moe = _ToyMoe().eval().requires_grad_(False)
        hidden = torch.tensor([[[2.0], [3.0], [4.0], [5.0]]]).repeat(2, 1, 1)
        labels = torch.tensor([1, 1])
        route_weights, route_ids, _ = moe.compute_router(hidden, labels)
        with torch.inference_mode():
            native, _ = moe(hidden, labels)
            with _forced_compute_exchange_state(
                moe,
                route_ids[..., 0],
                route_weights[..., 0],
                (None, None),
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot be nested"):
                    _ComputeExchangeInjector(
                        moe,
                        route_ids[..., 0],
                        route_weights[..., 0],
                        (None, None),
                    )
                hooked, _ = moe(hidden, labels)
        torch.testing.assert_close(hooked, native, rtol=0, atol=0)
        self.assertFalse(hasattr(moe, "_compute_exchange_probe_active"))

    def test_component_vjp_matches_manual_deltas(self):
        moe = _ToyMoe().eval().requires_grad_(False)
        hidden = torch.tensor([[2.0], [3.0], [4.0], [5.0]])
        gradient = torch.tensor([[1.0], [-2.0], [3.0], [-4.0]])
        experts = torch.tensor([0, 0, 1, 1])
        weights = torch.tensor([-0.5, 0.25, 0.4, 0.2])
        components = _exchange_components(
            moe,
            hidden,
            gradient,
            experts,
            weights,
        )
        self.assertAlmostEqual(float(components["donor_change"][0]), 2.0)
        self.assertAlmostEqual(float(components["receiver_change"][1]), -4.5)
        self.assertAlmostEqual(float(components["donor_change"][2]), -14.4, places=5)
        self.assertAlmostEqual(float(components["receiver_change"][3]), -19.2, places=5)

    def test_exact_control_uses_a_same_shape_native_reference(self):
        model = _BatchSensitiveModel().eval().requires_grad_(False)
        latent = torch.zeros(1, 1, 1, 1, 1)
        timestep = torch.zeros(1)
        label = torch.ones(1, dtype=torch.long)
        target = torch.zeros(1, 1, 1, 1)
        route_weights, route_ids, _ = model.moe.compute_router(
            torch.zeros(1, 4, 1),
            label,
        )
        with torch.inference_mode():
            native_prediction = model(latent, timestep, context=label)
        native_loss = native_prediction.double().square().mean()
        _, controls = _exact_candidate_changes(
            model=model,
            moe_layer=model.moe,
            noised_latent=latent,
            timestep=timestep,
            label=label,
            target=target,
            native_route_ids=route_ids[0, :, 0],
            native_route_weights=route_weights[0, :, 0],
            native_prediction=native_prediction,
            native_loss=native_loss,
            candidates=[_toy_candidate()],
        )
        self.assertEqual(controls["max_abs_paired_native_output_drift"], 0.0)
        self.assertGreater(
            controls["max_abs_single_vs_paired_native_output_drift"],
            0.0,
        )

    def test_scoring_and_summary_use_matched_candidate_bank(self):
        native = np.resize(np.arange(2, dtype=np.int64), 256)
        candidates = build_same_expert_exchange_candidates(native, 2, 9)
        donor_change = torch.arange(256, dtype=torch.float32)
        receiver_change = -torch.arange(256, dtype=torch.float32)
        components = {
            "donor_change": donor_change,
            "receiver_change": receiver_change,
            "donor_delta": torch.ones(256, 2),
            "receiver_delta": torch.ones(256, 2),
            "gradient_sq_norm": torch.ones(256),
        }
        records = _score_candidates(
            candidates,
            components,
            torch.from_numpy(native),
            torch.ones(256),
            torch.stack((
                torch.linspace(1.0, 0.0, 256),
                torch.linspace(0.0, 1.0, 256),
            ), dim=1),
            rolled_seed=11,
        )
        for index, record in enumerate(records):
            record["exact_mse_change"] = float(index - 10)
        summary = summarize_selectors(records, native_mse=2.0, random_seed=13)
        self.assertEqual(set(summary["selectors"]), {
            "first_order",
            "random",
            "router_margin",
            "rolled_utility",
            "exact_oracle",
        })
        self.assertEqual(summary["selectors"]["exact_oracle"]["selected_candidate_index"], 0)

    def test_selectors_do_not_share_first_order_abstention(self):
        records = []
        for index in range(CANDIDATE_COUNT):
            records.append({
                "first_order_change": float(index + 1),
                "exact_mse_change": float(index - 10),
                "router_margin_priority": float(index),
                "rolled_first_order_change": float(CANDIDATE_COUNT - index),
                "transferred_passes": 1,
            })
        summary = summarize_selectors(records, native_mse=1.0, random_seed=3)
        for name in ("first_order", "random", "router_margin", "rolled_utility"):
            self.assertTrue(summary["selectors"][name]["selected_non_native"])
        self.assertEqual(
            summary["selectors"]["first_order"]["selected_candidate_index"],
            0,
        )
        self.assertEqual(
            summary["selectors"]["router_margin"]["selected_candidate_index"],
            CANDIDATE_COUNT - 1,
        )


if __name__ == "__main__":
    unittest.main()
