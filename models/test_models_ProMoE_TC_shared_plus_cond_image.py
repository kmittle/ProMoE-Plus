"""CPU checks for the retained-shared plus conditional-image expert variant."""

import copy
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel

from analyses.flops.activated_params_tracker import ActivatedParamsTracker
from analyses.flops.flops_counter import FLOPsAccumulator
from models.models_ProMoE_TC import DiT as BaseDiT, SparseMoeBlock as BaseSparseMoeBlock
from models.models_ProMoE_TC_shared_plus_cond_image import DiT, SparseMoeBlock
from models.modules import MoeMLP


class SharedPlusConditionalImageTests(unittest.TestCase):
    def config(self):
        return EasyDict(
            num_routed_experts=2, moe_intermediate_size=16,
            shared_expert_intermediate_size=32, uncond_expert_intermediate_size=32,
            image_expert_intermediate_size=16, top_k=1, router_weight_mode="identity",
            routing_contrastive_lam=1.0, use_top_k_for_routing_contrastive=True,
            routing_contrastive_temperature=0.07, interleave=True, init_MoeMLP=False,
            use_shared_expert=True, use_uncond_expert=True,
            num_image_experts=3, image_expert_backend="bmm",
        )

    def build(self, model_class=DiT):
        torch.manual_seed(531)
        return model_class(
            input_size=8, patch_size=2, in_channels=4, hidden_size=16,
            depth=2, num_heads=2, mlp_ratio=4, class_dropout_prob=0.1,
            MoE_config=self.config(),
        )

    def setUp(self):
        torch.manual_seed(51)
        self.hidden = torch.randn(4, 16, 16)
        self.labels = torch.tensor([1, 1000, 2, 3])

    def test_branch_widths_share_the_dense_ffn_budget(self):
        model = self.build()
        mlp = model.blocks[1].mlp
        dense_width = model.blocks[0].mlp.fc1.out_features
        self.assertIsInstance(mlp.shared_expert, MoeMLP)
        self.assertEqual(mlp.shared_expert.intermediate_size, dense_width // 2)
        self.assertEqual(mlp.experts[-1].intermediate_size, dense_width // 2)
        self.assertTrue(all(expert.intermediate_size == dense_width // 4
                            for expert in mlp.experts[:-1]))
        self.assertTrue(all(expert.intermediate_size == dense_width // 4
                            for expert in mlp.image_experts.experts))
        self.assertEqual(mlp.shared_expert.intermediate_size
                         + mlp.experts[0].intermediate_size
                         + mlp.image_experts.experts[0].intermediate_size, dense_width)
        self.assertEqual(mlp.shared_expert.intermediate_size
                         + mlp.experts[-1].intermediate_size, dense_width)
        self.assertIs(mlp.compute_router.__func__, BaseSparseMoeBlock.compute_router)
        base_state = self.build(BaseDiT).state_dict()
        model_state = model.state_dict()
        for name, value in base_state.items():
            if name in model_state and model_state[name].shape == value.shape:
                torch.testing.assert_close(value, model_state[name], rtol=0, atol=0)

    def test_shared_sees_all_images_and_pool_sees_only_conditional_images(self):
        mlp = self.build().blocks[1].mlp.eval()
        seen_shared, seen_pool = [], []
        handles = [
            mlp.shared_expert.register_forward_pre_hook(
                lambda module, inputs: seen_shared.append(inputs[0].detach().clone())),
            mlp.image_experts.pooling.register_forward_pre_hook(
                lambda module, inputs: seen_pool.append(inputs[0].detach().clone())),
        ]
        try:
            with torch.no_grad():
                output, _ = mlp(self.hidden, self.labels)
                expected_uncond = (mlp.shared_expert(self.hidden[1:2])
                                   + mlp.experts[-1](self.hidden[1:2]))
            torch.testing.assert_close(seen_shared[0], self.hidden)
            self.assertEqual(len(seen_pool), 1)
            torch.testing.assert_close(seen_pool[0], self.hidden[[0, 2, 3]])
            torch.testing.assert_close(output[1:2], expected_uncond)
            self.assertEqual(mlp.image_experts.last_expert_indices.numel(), 3)
        finally:
            for handle in handles:
                handle.remove()

    def test_unconditional_only_skips_pool_resets_state_and_has_zero_image_gradients(self):
        mlp = self.build().blocks[1].mlp.train()
        mlp(self.hidden, self.labels)
        self.assertEqual(mlp.image_experts.last_expert_indices.numel(), 3)
        calls = []
        handle = mlp.image_experts.register_forward_hook(lambda *args: calls.append(1))
        try:
            mlp.zero_grad(set_to_none=True)
            output, _ = mlp(self.hidden, torch.full_like(self.labels, 1000))
            output.square().sum().backward()
            self.assertEqual(calls, [])
            self.assertEqual(mlp.image_experts.last_expert_indices.numel(), 0)
            self.assertIsNone(mlp.image_experts.last_routing_contrastive_loss)
            for parameter in mlp.image_experts.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(torch.count_nonzero(parameter.grad), 0)
            self.assertIsNotNone(mlp.cluster_centers.grad)
            self.assertEqual(torch.count_nonzero(mlp.cluster_centers.grad), 0)
        finally:
            handle.remove()

    def test_image_auxiliary_gradient_does_not_reach_unconditional_images(self):
        mlp = self.build().blocks[1].mlp.train()
        hidden = self.hidden.clone().requires_grad_()
        with torch.no_grad():
            mlp.image_experts.cluster_centers.copy_(
                mlp.image_experts.pooling(hidden[[0, 2, 3]]))
        output, _ = mlp(hidden, self.labels)
        self.assertGreaterEqual(mlp.image_experts.last_expert_indices.unique().numel(), 2)
        # The task gradient is zero. The image branch injects only its own auxiliary
        # loss here; the token auxiliary is returned separately by SparseMoeBlock.
        (output.sum() * 0).backward()
        self.assertGreater(mlp.image_experts.pooling.query.grad.abs().sum().item(), 0)
        self.assertGreater(hidden.grad[[0, 2, 3]].abs().sum().item(), 0)
        self.assertEqual(torch.count_nonzero(hidden.grad[1]), 0)

    def test_training_uses_labels_after_condition_dropout(self):
        model = self.build().train()
        model.y_embedder.dropout_prob = 1.0
        calls = []
        handle = model.blocks[1].mlp.image_experts.register_forward_hook(
            lambda *args: calls.append(1))
        try:
            output = model(torch.randn(4, 4, 8, 8), torch.arange(4).float(),
                           torch.tensor([1, 2, 3, 4]))
            self.assertEqual(tuple(output.shape), (4, 8, 8, 8))
            self.assertEqual(calls, [])
            self.assertEqual(model.blocks[1].mlp.image_experts.last_expert_indices.numel(), 0)
        finally:
            handle.remove()

    def test_bmm_and_loop_match_mixed_batch_outputs_and_gradients(self):
        first = self.build().blocks[1].mlp.train()
        second = copy.deepcopy(first)
        second.image_experts.set_expert_backend("loop")
        results = []
        probe = torch.randn_like(self.hidden)
        for mlp in (first, second):
            hidden = self.hidden.clone().requires_grad_()
            output, loss = mlp(hidden, self.labels)
            objective = (output * probe).sum()
            if loss is not None:
                objective = objective + loss
            objective.backward()
            results.append((output, hidden.grad,
                            {name: parameter.grad for name, parameter in mlp.named_parameters()}))
        for left, right in zip(results[0][:2], results[1][:2]):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=2e-5)
        for name, gradient in results[0][2].items():
            self.assertIsNotNone(gradient, name)
            torch.testing.assert_close(gradient, results[1][2][name], rtol=1e-5, atol=2e-5)

    def test_flops_count_only_conditional_pooling_and_restore_backend(self):
        mlp = self.build().blocks[1].mlp.eval()
        pool = mlp.image_experts
        counter = FLOPsAccumulator(mlp)
        counter.start()
        try:
            self.assertEqual(pool.expert_backend, "loop")
            with torch.no_grad():
                mlp(self.hidden, self.labels)
            # Six cond-only K/V, attention, projection/router/FFN components.
            cond, tokens, dim, width, experts = 3, 16, 16, 16, 3
            expected = (2 * cond * tokens * dim * dim + cond * dim * dim
                        + 2 * cond * tokens * dim + cond * dim * experts
                        + 2 * cond * tokens * dim * width)
            observed = sum(module.total_ops.item() for module in pool.modules()
                           if hasattr(module, "total_ops"))
            self.assertEqual(observed, expected)
            counter.collect()
            with torch.no_grad():
                mlp(self.hidden, torch.full_like(self.labels, 1000))
            self.assertEqual(sum(module.total_ops.item() for module in pool.modules()
                                 if hasattr(module, "total_ops")), 0)
        finally:
            counter.stop()
        self.assertEqual(pool.expert_backend, "bmm")

    def test_activated_parameters_exclude_whole_pool_for_unconditional_input(self):
        model = self.build().eval()
        mlp = model.blocks[1].mlp
        pool = mlp.image_experts

        def token_router(hidden_states, labels):
            batch, tokens, _ = hidden_states.shape
            ids = torch.where(labels == 1000, 2, 0)[:, None, None].expand(batch, tokens, 1).contiguous()
            return hidden_states.new_ones(batch, tokens, 1), ids, None

        mlp.compute_router = token_router
        pool.compute_router = lambda embedding: (
            embedding.new_ones(len(embedding)), torch.zeros(len(embedding), dtype=torch.long))
        total = sum(parameter.numel() for parameter in model.parameters())
        tracker = ActivatedParamsTracker(model)
        tracker.start()
        try:
            with torch.no_grad():
                model(torch.randn(4, 4, 8, 8), torch.arange(4).float(), self.labels)
            tracker.record_forward_pass()
            inactive = sum(p.numel() for p in mlp.experts[1].parameters())
            inactive += sum(p.numel() for expert in pool.experts[1:] for p in expert.parameters())
            self.assertEqual(tracker.get_mean_activated_params(), total - inactive)
            with torch.no_grad():
                model(torch.randn(4, 4, 8, 8), torch.arange(4).float(),
                      torch.full_like(self.labels, 1000))
            tracker.record_forward_pass()
            all_cond_params = sum(p.numel() for expert in mlp.experts[:-1] for p in expert.parameters())
            all_image_params = sum(p.numel() for p in pool.parameters())
            self.assertEqual(tracker.get_stats()['min_activated_params'],
                             total - all_cond_params - all_image_params)
        finally:
            tracker.stop()

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU DDP unavailable")
    def test_ddp_can_follow_unconditional_batch_with_mixed_batch(self):
        with tempfile.TemporaryDirectory(prefix="promoe_image_ddp_") as directory:
            dist.init_process_group("gloo", init_method=Path(directory, "rendezvous").as_uri(),
                                    rank=0, world_size=1)
            try:
                mlp = DistributedDataParallel(self.build().blocks[1].mlp.train())
                for labels in (torch.full_like(self.labels, 1000), self.labels, self.labels):
                    mlp.zero_grad(set_to_none=True)
                    output, loss = mlp(self.hidden, labels)
                    objective = output.square().mean()
                    if loss is not None:
                        objective = objective + loss
                    objective.backward()
            finally:
                dist.destroy_process_group()

    def test_invalid_width_and_missing_branches_fail_clearly(self):
        for changes in ({"image_expert_intermediate_size": 0},
                        {"uncond_expert_intermediate_size": True},
                        {"use_shared_expert": False}, {"use_uncond_expert": False},
                        {"top_k": 0}, {"top_k": 3},
                        {"top_k": True}, {"top_k": 1.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config = dict(self.config())
                config.update(changes)
                SparseMoeBlock(hidden_size=16, **config)


if __name__ == "__main__":
    unittest.main()
