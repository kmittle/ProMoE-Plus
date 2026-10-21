"""CPU regression tests for image-wise shared experts and their FLOPs backend."""

import copy
import unittest
from unittest import mock

import torch
from easydict import EasyDict

from analyses.flops.activated_params_tracker import ActivatedParamsTracker
from analyses.flops.flops_counter import FLOPsAccumulator
from models.models_ProMoE_TC import DiT as BaseDiT, SparseMoeBlock
from models.models_ProMoE_TC_image_shared import DiT, ImageWiseSharedExpertPool


class ImageSharedExpertTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.pool = ImageWiseSharedExpertPool(8, 12, pool_num_heads=2).double()
        self.hidden = torch.randn(6, 7, 8, dtype=torch.float64)

    def test_backends_match_outputs_input_and_expert_gradients(self):
        ids = torch.tensor([2, 0, 2, 0, 2, 0])  # expert 1 has no images
        results = []
        probe = torch.randn_like(self.hidden)
        for backend in ("bmm", "loop"):
            pool = copy.deepcopy(self.pool).train()
            # Exercise nonzero biases, not just their zero initialization.
            with torch.no_grad():
                for expert in pool.experts:
                    expert.up_proj.bias.fill_(0.3)
                    expert.down_proj.bias.fill_(-0.2)
            hidden = self.hidden.clone().requires_grad_()
            output = getattr(pool, "_forward_" + backend)(hidden, ids)
            (output * probe).sum().backward()
            grads = [parameter.grad for expert in pool.experts for parameter in expert.parameters()]
            self.assertTrue(all(gradient is not None for gradient in grads))
            self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in pool.experts[1].parameters()))
            results.append((output.detach(), hidden.grad, grads))
        for left, right in zip(results[0][:2], results[1][:2]):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)
        for left, right in zip(results[0][2], results[1][2]):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)

    def test_full_pool_backends_match_router_and_auxiliary_gradients(self):
        with torch.no_grad():
            self.pool.cluster_centers.copy_(self.pool.pooling(self.hidden)[:3])
        probe = torch.randn_like(self.hidden)
        results = []
        for backend in ("bmm", "loop"):
            pool = copy.deepcopy(self.pool).train()
            pool.set_expert_backend(backend)
            hidden = self.hidden.clone().requires_grad_()
            output = pool(hidden)
            (output * probe).sum().backward()
            self.assertGreaterEqual(pool.last_expert_indices.unique().numel(), 2)
            self.assertGreater(pool.last_routing_contrastive_loss.item(), 0)
            self.assertTrue(all(p.grad is not None for p in pool.parameters()))
            self.assertGreater(pool.pooling.query.grad.abs().sum().item(), 0)
            results.append((output.detach(), hidden.grad, {
                name: parameter.grad for name, parameter in pool.named_parameters()
            }))
        for left, right in zip(results[0][:2], results[1][:2]):
            torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-9)
        for name, gradient in results[0][2].items():
            torch.testing.assert_close(gradient, results[1][2][name], rtol=1e-9, atol=1e-9)

    def test_image_contrastive_loss_is_the_base_objective_and_injects_gradients(self):
        self.pool.train()
        with torch.no_grad():
            self.pool.cluster_centers.copy_(self.pool.pooling(self.hidden)[:3])
        embeddings = self.pool.pooling(self.hidden)
        _, ids = self.pool.compute_router(embeddings)
        reference_loss = SparseMoeBlock.compute_routing_contrastive_loss(
            self.pool, embeddings, ids, use_top_k=False
        )
        reference_loss.backward()
        expected = self.pool.cluster_centers.grad.clone()
        self.pool.zero_grad(set_to_none=True)
        # With a zero task gradient, only AddAuxiliaryLoss supplies gradients.
        (self.pool(self.hidden).sum() * 0).backward()
        torch.testing.assert_close(self.pool.cluster_centers.grad, expected)
        self.assertGreater(expected.abs().sum().item(), 0)
        self.assertGreater(self.pool.pooling.query.grad.abs().sum().item(), 0)

    def test_images_route_independently_and_pooling_does_not_modify_tokens(self):
        self.pool.eval()
        original = self.hidden.clone()
        with torch.no_grad():
            together = self.pool(self.hidden)
            ids = self.pool.last_expert_indices.clone()
            individual = torch.cat([self.pool(image[None]) for image in self.hidden])
            permutation = torch.tensor([5, 2, 0, 4, 1, 3])
            permuted = self.pool(self.hidden[permutation])
            torch.testing.assert_close(self.pool.last_expert_indices, ids[permutation])
            torch.testing.assert_close(permuted, together[permutation])
            torch.testing.assert_close(together, individual)
            token_permutation = torch.randperm(self.hidden.shape[1])
            torch.testing.assert_close(
                self.pool.pooling(self.hidden[:, token_permutation]), self.pool.pooling(self.hidden)
            )
        torch.testing.assert_close(self.hidden, original, rtol=0, atol=0)
        self.assertIsNone(self.pool.last_routing_contrastive_loss)

    def test_unused_experts_have_zero_gradients_in_both_training_backends(self):
        hidden = self.hidden[:1].expand(4, -1, -1).contiguous()
        for backend in ("bmm", "loop"):
            with self.subTest(backend=backend):
                pool = copy.deepcopy(self.pool).train()
                pool.set_expert_backend(backend)
                pool(hidden).square().sum().backward()
                selected = pool.last_expert_indices.unique().tolist()
                self.assertEqual(len(selected), 1)
                self.assertTrue(all(p.grad is not None for p in pool.parameters()))
                for index, expert in enumerate(pool.experts):
                    if index not in selected:
                        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in expert.parameters()))

    def test_cpu_bfloat16_autocast_backends_agree(self):
        results = []
        for backend in ("bmm", "loop"):
            pool = copy.deepcopy(self.pool).float().train()
            pool.set_expert_backend(backend)
            hidden = self.hidden.float().clone().requires_grad_()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = pool(hidden)
                loss = output.square().mean()
            loss.backward()
            self.assertEqual(output.dtype, hidden.dtype)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in pool.parameters()))
            results.append((output.detach(), hidden.grad))
        for left, right in zip(results[0], results[1]):
            torch.testing.assert_close(left, right, rtol=0.05, atol=0.005)

    def test_backend_switch_preserves_parameters_and_checkpoint_keys(self):
        self.assertEqual(self.pool.expert_backend, "bmm")
        self.assertEqual(self.pool.num_image_experts, 3)
        before = copy.deepcopy(self.pool.state_dict())
        self.pool.set_expert_backend("loop")
        self.pool.load_state_dict(before, strict=True)
        self.assertEqual(self.pool.state_dict().keys(), before.keys())
        for name, value in self.pool.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.pool.set_expert_backend("invalid")
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ImageWiseSharedExpertPool(8, 12, num_image_experts=invalid)
        with self.assertRaises(ValueError):
            ImageWiseSharedExpertPool(8, 12, pool_num_heads=3)

    def test_flops_uses_loop_counts_real_work_and_restores_backend(self):
        pool = self.pool.eval()
        # Every image selects one expert; unselected experts must not be executed.
        hidden = self.hidden[:1].expand_as(self.hidden).contiguous()
        with torch.no_grad():
            embedding = pool.pooling(hidden)[0]
            pool.cluster_centers[0].copy_(embedding)
            pool.cluster_centers[1:].copy_(-embedding)
        unused_hooks = []
        handles = [expert.register_forward_hook(lambda *args: unused_hooks.append(True))
                   for expert in pool.experts[1:]]
        counter = FLOPsAccumulator(pool)
        keys = set(pool.state_dict())
        try:
            with mock.patch.object(pool, "_forward_bmm", side_effect=AssertionError("bmm used in FLOPs")):
                counter.start()
                self.assertEqual(pool.expert_backend, "loop")
                with torch.no_grad():
                    pool(hidden)
                measured = counter.collect()
                batch, tokens, dim = self.hidden.shape
                expected = (
                    batch * tokens * 2 * dim * 12  # exactly one FFN per image
                    + batch * tokens * 2 * dim * dim  # K/V projections
                    + batch * dim * dim  # pooled output projection
                    + 2 * batch * tokens * dim  # one-query attention products
                    + batch * dim * 3  # image-prototype scores
                )
                self.assertEqual(measured, expected)
                self.assertEqual(unused_hooks, [])
        finally:
            counter.stop()
            for handle in handles:
                handle.remove()
        self.assertEqual(pool.expert_backend, "bmm")
        self.assertEqual(set(pool.state_dict()), keys)
        pool.set_expert_backend("loop")
        counter.start()
        counter.stop()
        self.assertEqual(pool.expert_backend, "loop")

    def test_flops_keeps_ordinary_linear_counting_unchanged(self):
        linear = torch.nn.Linear(8, 12)
        counter = FLOPsAccumulator(linear)
        counter.start()
        try:
            linear(torch.randn(3, 7, 8))
            self.assertEqual(counter.collect(), 3 * 7 * 8 * 12)
        finally:
            counter.stop()


class ImageSharedModelTests(unittest.TestCase):
    def build(self, model_class=DiT):
        torch.manual_seed(113)
        cfg = EasyDict(
            num_routed_experts=2, moe_intermediate_size=24,
            shared_expert_intermediate_size=24, top_k=1, router_weight_mode="identity",
            routing_contrastive_lam=1.0, use_top_k_for_routing_contrastive=True,
            routing_contrastive_temperature=0.07, interleave=True, init_MoeMLP=False,
            use_shared_expert=True, num_image_experts=3, image_expert_backend="bmm",
        )
        return model_class(input_size=8, patch_size=2, in_channels=4, hidden_size=16,
                           depth=2, num_heads=2, class_dropout_prob=0.1, MoE_config=cfg)

    def test_only_shared_branch_changes_and_token_router_is_inherited(self):
        base = self.build(BaseDiT)
        model = self.build()
        base_state = base.state_dict()
        model_state = model.state_dict()
        for name, value in base_state.items():
            if ".shared_expert." not in name:
                torch.testing.assert_close(value, model_state[name], rtol=0, atol=0)
        base_mlp, new_mlp = base.blocks[1].mlp, model.blocks[1].mlp
        self.assertIs(new_mlp.compute_router.__func__, base_mlp.compute_router.__func__)
        self.assertIs(new_mlp.forward.__func__, base_mlp.forward.__func__)
        self.assertEqual(len(new_mlp.shared_expert.experts), 3)
        hidden, labels = torch.randn(3, 16, 16), torch.tensor([1, 1000, 2])
        base_weights, base_ids, _ = base_mlp.compute_router(hidden, labels)
        weights, ids, _ = new_mlp.compute_router(hidden, labels)
        torch.testing.assert_close(weights, base_weights, rtol=0, atol=0)
        torch.testing.assert_close(ids, base_ids)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(3, 4, 8, 8), torch.tensor([10., 500., 900.]), labels)
        self.assertEqual(tuple(output.shape), (3, 8, 8, 8))

    def test_activated_parameters_exclude_unused_image_experts(self):
        model = self.build().eval()
        mlp = model.blocks[1].mlp
        pool = mlp.shared_expert

        def token_router(hidden_states, labels):
            batch, tokens, _ = hidden_states.shape
            ids = torch.where(labels == 1000, 2, 0)[:, None, None].expand(batch, tokens, 1).contiguous()
            return hidden_states.new_ones(batch, tokens, 1), ids, None

        mlp.compute_router = token_router
        pool.compute_router = lambda embedding: (
            embedding.new_ones(len(embedding)), torch.zeros(len(embedding), dtype=torch.long)
        )
        total = sum(p.numel() for p in model.parameters())
        inactive = sum(p.numel() for p in mlp.experts[1].parameters())
        inactive += sum(p.numel() for expert in pool.experts[1:] for p in expert.parameters())
        tracker = ActivatedParamsTracker(model)
        tracker.start()
        try:
            with torch.no_grad():
                model(torch.randn(2, 4, 8, 8), torch.tensor([10., 900.]), torch.tensor([1, 1000]))
            tracker.record_forward_pass()
            self.assertEqual(tracker.get_mean_activated_params(), total - inactive)
        finally:
            tracker.stop()

    def test_base_activated_parameter_tracking_is_unchanged(self):
        model = self.build(BaseDiT).eval()
        mlp = model.blocks[1].mlp

        def token_router(hidden_states, labels):
            batch, tokens, _ = hidden_states.shape
            ids = torch.where(labels == 1000, 2, 0)[:, None, None].expand(batch, tokens, 1).contiguous()
            return hidden_states.new_ones(batch, tokens, 1), ids, None

        mlp.compute_router = token_router
        expected = sum(p.numel() for p in model.parameters())
        expected -= sum(p.numel() for p in mlp.experts[1].parameters())
        tracker = ActivatedParamsTracker(model)
        tracker.start()
        try:
            with torch.no_grad():
                model(torch.randn(2, 4, 8, 8), torch.tensor([10., 900.]), torch.tensor([1, 1000]))
            tracker.record_forward_pass()
            self.assertEqual(tracker.get_mean_activated_params(), expected)
        finally:
            tracker.stop()


if __name__ == "__main__":
    unittest.main()
