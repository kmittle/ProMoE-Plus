import unittest
from types import MethodType

import torch

from analyses.denoising_regret.probe import (
    RoutingProbeCapture,
    _all_router_weights,
    _compute_router,
    _forced_routes,
)
from analyses.denoising_regret.responsibility_probe import (
    _forced_route_weight_matrix,
    _forced_token_route_weights,
    _probe_sigma,
)
from analyses.flops.activated_params_tracker import ActivatedParamsTracker
from analyses.flops.expert_tracker import ExpertActivationTracker
from analyses.heatmap.token_choice_capture import TokenChoiceExpertIndexCapture
from analyses.routing_translation.probe import (
    RouteInputCapture,
    _capture_native_forward,
    _forced_route_matrices,
)
from analyses.t_SNE.routing_capture import TokenRoutingCapture
from analyses.timestep_utility.probe import _forced_route_state
from models.models_ProMoE_TC import DiT


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


class _LegacyTwoArgumentRouter:
    phase_metric = None

    def compute_router(self, hidden_states, labels):
        batch_size, token_count, _ = hidden_states.shape
        weights = torch.ones(batch_size, token_count, 1)
        indices = torch.zeros(batch_size, token_count, 1, dtype=torch.long)
        return weights, indices, None


def _model_kwargs(metric_enabled):
    return {
        'input_size': 8,
        'patch_size': 2,
        'in_channels': 4,
        'hidden_size': 32,
        'depth': 2,
        'num_heads': 4,
        'mlp_ratio': 2,
        'class_dropout_prob': 0.1,
        'num_classes': 1000,
        'learn_sigma': False,
        'MoE_config': _AttrDict(
            num_routed_experts=4,
            moe_intermediate_size=48,
            shared_expert_intermediate_size=48,
            load_balance_loss_coef=0,
            norm_topk_prob=False,
            seq_aux=False,
            use_shared_expert=True,
            interleave=True,
            init_MoeMLP=False,
            top_k=1,
            router_weight_mode='identity',
            routing_contrastive_lam=1.0,
            use_top_k_for_routing_contrastive=True,
            routing_contrastive_temperature=0.07,
            phase_metric_config={
                'enabled': metric_enabled,
                'rank': 4,
                'num_fourier_bands': 2,
                'num_train_timesteps': 1000,
                'scale': 0.25,
                'shuffle_timestep': False,
                'init_seed': 1729,
            },
        ),
    }


class PhaseMetricModelTest(unittest.TestCase):
    @staticmethod
    def _build_pair():
        torch.manual_seed(17)
        base = DiT(**_model_kwargs(metric_enabled=False)).eval()
        torch.manual_seed(17)
        candidate = DiT(**_model_kwargs(metric_enabled=True)).eval()
        return base, candidate

    def test_common_parameters_keep_the_base_initialization(self):
        base, candidate = self._build_pair()
        base_state = base.state_dict()
        candidate_state = candidate.state_dict()
        self.assertEqual(
            set(candidate_state) - set(base_state),
            {
                'blocks.1.mlp.phase_metric.token_basis',
                'blocks.1.mlp.phase_metric.prototype_basis',
                'blocks.1.mlp.phase_metric.phase_to_rank',
                'blocks.1.mlp.phase_metric.expert_gain',
                'blocks.1.mlp.phase_metric.fourier_frequencies',
            },
        )
        for name, value in base_state.items():
            torch.testing.assert_close(
                candidate_state[name], value, rtol=0, atol=0
            )

    def test_step_zero_router_and_eval_output_match_base(self):
        base, candidate = self._build_pair()
        generator = torch.Generator().manual_seed(29)
        hidden_states = torch.randn(3, 16, 32, generator=generator)
        labels = torch.tensor([2, 7, 1000])
        timesteps = torch.tensor([100.0, 500.0, 900.0])

        base_router = base.blocks[1].mlp.compute_router(
            hidden_states, labels
        )
        candidate_router = candidate.blocks[1].mlp.compute_router(
            hidden_states, labels, timestep=timesteps
        )
        for base_value, candidate_value in zip(base_router, candidate_router):
            if base_value is None:
                self.assertIsNone(candidate_value)
            else:
                torch.testing.assert_close(
                    candidate_value, base_value, rtol=0, atol=0
                )

        inputs = torch.randn(3, 4, 8, 8, generator=generator)
        with torch.no_grad():
            base_output = base(inputs, timesteps, labels)
            candidate_output = candidate(inputs, timesteps, labels)
        torch.testing.assert_close(
            candidate_output, base_output, rtol=0, atol=0
        )

    def test_step_zero_matches_base_under_bfloat16_autocast(self):
        base, candidate = self._build_pair()
        generator = torch.Generator().manual_seed(43)
        hidden_states = torch.randn(3, 16, 32, generator=generator)
        labels = torch.tensor([2, 7, 1000])
        timesteps = torch.tensor([100.0, 500.0, 900.0])
        inputs = torch.randn(3, 4, 8, 8, generator=generator)

        with torch.no_grad(), torch.autocast(
            device_type='cpu', dtype=torch.bfloat16
        ):
            base_router = base.blocks[1].mlp.compute_router(
                hidden_states, labels
            )
            candidate_router = candidate.blocks[1].mlp.compute_router(
                hidden_states, labels, timesteps
            )
            base_output = base(inputs, timesteps, labels)
            candidate_output = candidate(inputs, timesteps, labels)

        for base_value, candidate_value in zip(base_router, candidate_router):
            if base_value is None:
                self.assertIsNone(candidate_value)
            else:
                torch.testing.assert_close(
                    candidate_value, base_value, rtol=0, atol=0
                )
        torch.testing.assert_close(
            candidate_output, base_output, rtol=0, atol=0
        )

    def test_base_moe_forward_contract_stays_two_argument(self):
        base, candidate = self._build_pair()
        base_inputs = []
        candidate_inputs = []
        base_handle = base.blocks[1].mlp.register_forward_pre_hook(
            lambda module, inputs: base_inputs.append(inputs)
        )
        candidate_handle = candidate.blocks[1].mlp.register_forward_pre_hook(
            lambda module, inputs: candidate_inputs.append(inputs)
        )
        generator = torch.Generator().manual_seed(31)
        inputs = torch.randn(2, 4, 8, 8, generator=generator)
        timesteps = torch.tensor([100.0, 900.0])
        labels = torch.tensor([2, 7])
        try:
            with torch.no_grad():
                base(inputs, timesteps, labels)
                candidate(inputs, timesteps, labels)
        finally:
            base_handle.remove()
            candidate_handle.remove()

        self.assertEqual(len(base_inputs), 1)
        self.assertEqual(len(base_inputs[0]), 2)
        self.assertEqual(len(candidate_inputs), 1)
        self.assertEqual(len(candidate_inputs[0]), 3)
        torch.testing.assert_close(candidate_inputs[0][2], timesteps)

    def test_base_accepts_existing_two_argument_router_wrappers(self):
        base, _ = self._build_pair()
        moe_layer = base.blocks[1].mlp
        original_router = moe_layer.compute_router
        call_count = 0

        def wrapped(this, hidden_states, labels):
            nonlocal call_count
            del this
            call_count += 1
            return original_router(hidden_states, labels)

        moe_layer.compute_router = MethodType(wrapped, moe_layer)
        generator = torch.Generator().manual_seed(33)
        inputs = torch.randn(2, 4, 8, 8, generator=generator)
        timesteps = torch.tensor([100.0, 900.0])
        labels = torch.tensor([2, 7])
        try:
            with torch.no_grad():
                base(inputs, timesteps, labels)
        finally:
            del moe_layer.compute_router
        self.assertEqual(call_count, 1)

    def test_phase_model_works_with_routing_analysis_trackers(self):
        _, candidate = self._build_pair()
        generator = torch.Generator().manual_seed(37)
        inputs = torch.randn(2, 4, 8, 8, generator=generator)
        timesteps = torch.tensor([100.0, 900.0])
        labels = torch.tensor([2, 7])

        expert_tracker = ExpertActivationTracker(candidate)
        expert_tracker.start()
        params_tracker = ActivatedParamsTracker(candidate)
        self.assertEqual(len(params_tracker.moe_blocks), 1)
        params_tracker.start()
        try:
            with torch.no_grad():
                candidate(inputs, timesteps, labels)
            raw_counts = expert_tracker.consume_current_forward_raw_counts()
            params_tracker.record_forward_pass()
            self.assertIn(1, raw_counts)
            self.assertGreater(raw_counts[1][1], 0)
            self.assertEqual(params_tracker.get_stats()['num_forwards'], 1)
        finally:
            params_tracker.stop()
            expert_tracker.stop()

        routing_capture = TokenRoutingCapture(candidate)
        heatmap_capture = TokenChoiceExpertIndexCapture(candidate)
        routing_capture.enable(denoise_step=0, scheduler_timestep=900.0)
        heatmap_capture.enable(denoise_step=0, scheduler_timestep=900.0)
        try:
            with torch.no_grad():
                candidate(inputs, timesteps, labels)
            routing = routing_capture.disable_and_collect()
            heatmaps = heatmap_capture.disable_and_collect()
            self.assertIn(1, routing)
            self.assertIn(1, heatmaps)
        finally:
            routing_capture.close()
            heatmap_capture.close()

    def test_phase_model_works_with_forced_route_analysis_overrides(self):
        _, candidate = self._build_pair()
        moe_layer = candidate.blocks[1].mlp
        original_router = moe_layer.compute_router.__func__
        generator = torch.Generator().manual_seed(47)
        hidden_states = torch.randn(2, 16, 32, generator=generator)
        labels = torch.tensor([2, 7])
        timesteps = torch.tensor([100.0, 900.0])

        native_weights, native_indices, _ = moe_layer.compute_router(
            hidden_states,
            labels,
            timesteps,
        )
        token_indices = torch.tensor([0, 1])
        expert_indices = torch.tensor([1, 2])
        rows = torch.arange(labels.numel())
        with _forced_routes(moe_layer, token_indices, expert_indices):
            _, indices, _ = moe_layer.compute_router(
                hidden_states,
                labels,
                timesteps,
            )
            torch.testing.assert_close(
                indices[rows, token_indices, 0],
                expert_indices,
            )

        forced_token_weights = torch.tensor([0.2, 0.7])
        with _forced_token_route_weights(
            moe_layer,
            token_indices,
            forced_token_weights,
        ):
            weights, _, _ = moe_layer.compute_router(
                hidden_states,
                labels,
                timesteps,
            )
            torch.testing.assert_close(
                weights[rows, token_indices, 0],
                forced_token_weights,
            )

        forced_weight_matrix = torch.full_like(native_weights[..., 0], 0.3)
        with _forced_route_weight_matrix(moe_layer, forced_weight_matrix):
            weights, _, _ = moe_layer.compute_router(
                hidden_states,
                labels,
                timesteps,
            )
            torch.testing.assert_close(weights[..., 0], forced_weight_matrix)

        forced_route_ids = (
            native_indices[..., 0] + 1
        ) % moe_layer.num_routed_experts
        with _forced_route_matrices(moe_layer, forced_route_ids):
            _, indices, _ = moe_layer.compute_router(
                hidden_states,
                labels,
                timesteps,
            )
            torch.testing.assert_close(indices[..., 0], forced_route_ids)

        forced_route_weights = torch.full_like(native_weights[..., 0], 0.6)
        with _forced_route_state(
            moe_layer,
            forced_route_ids,
            forced_route_weights,
        ):
            weights, indices, _ = moe_layer.compute_router(
                hidden_states,
                labels,
                timesteps,
            )
            torch.testing.assert_close(indices[..., 0], forced_route_ids)
            torch.testing.assert_close(weights[..., 0], forced_route_weights)

        self.assertIs(moe_layer.compute_router.__func__, original_router)

    def test_phase_model_works_with_complete_responsibility_probe(self):
        _, candidate = self._build_pair()
        moe_layer = candidate.blocks[1].mlp
        capture = RoutingProbeCapture(moe_layer)
        generator = torch.Generator().manual_seed(53)
        clean_latent = torch.randn(1, 4, 1, 8, 8, generator=generator)
        noise = torch.randn(1, 4, 1, 8, 8, generator=generator)
        label = torch.tensor([7])
        try:
            records, global_record, baseline, controls = _probe_sigma(
                model=candidate,
                moe_layer=moe_layer,
                capture=capture,
                clean_latent=clean_latent,
                noise=noise,
                label=label,
                sigma=0.5,
                num_train_timesteps=1000,
                num_token_probes=2,
                candidate_scales=(0.4, 0.6),
                exact_batch_size=2,
                generator=torch.Generator().manual_seed(59),
            )
        finally:
            capture.close()
        self.assertEqual(len(records), 2)
        self.assertEqual(global_record['timestep'], 500.0)
        self.assertGreaterEqual(baseline['native_router_weight_max'], 0.0)
        self.assertEqual(controls['noop_num_probes'], 2)

    def test_phase_model_works_with_native_translation_capture(self):
        _, candidate = self._build_pair()
        moe_layer = candidate.blocks[1].mlp
        capture = RouteInputCapture(moe_layer)
        generator = torch.Generator().manual_seed(61)
        inputs = torch.randn(1, 4, 1, 8, 8, generator=generator)
        timestep = torch.tensor([700.0])
        label = torch.tensor([11])
        try:
            output, hidden_states, weights, indices = _capture_native_forward(
                candidate,
                moe_layer,
                capture,
                inputs,
                timestep,
                label,
            )
        finally:
            capture.close()
        self.assertEqual(tuple(output.shape), (1, 4, 8, 8))
        self.assertEqual(tuple(hidden_states.shape), (1, 16, 32))
        self.assertEqual(tuple(weights.shape), (1, 16, 1))
        self.assertEqual(tuple(indices.shape), (1, 16, 1))

    def test_all_router_scores_include_the_learned_phase_residual(self):
        _, candidate = self._build_pair()
        moe_layer = candidate.blocks[1].mlp
        with torch.no_grad():
            moe_layer.phase_metric.phase_to_rank[0, 0] = 1.0
            moe_layer.phase_metric.phase_to_rank[1, 1] = -0.5
        generator = torch.Generator().manual_seed(67)
        hidden_states = torch.randn(3, 16, 32, generator=generator)
        labels = torch.tensor([2, 7, 11])
        timesteps = torch.tensor([100.0, 500.0, 900.0])

        weights, indices, _ = moe_layer.compute_router(
            hidden_states,
            labels,
            timesteps,
        )
        scores = _all_router_weights(moe_layer, hidden_states, timesteps)
        torch.testing.assert_close(scores.argmax(dim=-1), indices[..., 0])
        torch.testing.assert_close(
            scores.gather(-1, indices).squeeze(-1),
            weights[..., 0],
        )
        with self.assertRaisesRegex(ValueError, 'requires timestep'):
            _all_router_weights(moe_layer, hidden_states)

    def test_router_adapter_preserves_legacy_two_argument_overrides(self):
        router = _LegacyTwoArgumentRouter()
        hidden_states = torch.zeros(2, 3, 4)
        labels = torch.tensor([1, 2])
        weights, indices, auxiliary_loss = _compute_router(
            router,
            hidden_states,
            labels,
            torch.tensor([100.0, 900.0]),
        )
        self.assertEqual(tuple(weights.shape), (2, 3, 1))
        self.assertEqual(tuple(indices.shape), (2, 3, 1))
        self.assertIsNone(auxiliary_loss)


if __name__ == '__main__':
    unittest.main()
