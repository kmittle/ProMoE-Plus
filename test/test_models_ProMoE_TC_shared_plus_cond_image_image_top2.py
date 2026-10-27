"""Image Top-2 + token Top-1: routing, gradients and matched FFN budget."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from analyses.flops.activated_params_tracker import ActivatedParamsTracker
from analyses.flops.flops_counter import FLOPsAccumulator
from models.models_ProMoE_TC import SparseMoeBlock as BaseSparseMoeBlock
from models.models_ProMoE_TC_image_shared import ImageWiseSharedExpertPool
from models.models_ProMoE_TC_shared_plus_cond_image import DiT, SparseMoeBlock
from models.modules import MoeMLP
from test.test_models_ProMoE_TC_shared_plus_cond_image_top2 import moe_config


class ImageTop2TokenTop1Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        self.hidden = torch.randn(4, 16, 16)
        self.labels = torch.tensor([1, 1000, 2, 3])

    def config(self, **overrides):
        settings = dict(top_k=1, image_top_k=2)
        settings.update(overrides)
        return moe_config(**settings)

    def build(self, **overrides):
        torch.manual_seed(531)
        return SparseMoeBlock(hidden_size=16, **self.config(**overrides))

    def model(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return DiT(input_size=8, patch_size=2, in_channels=4, hidden_size=16,
                       depth=2, num_heads=2, mlp_ratio=4, MoE_config=self.config(), **kwargs)

    def test_router_selects_two_distinct_experts_with_raw_cosine_weights(self):
        pool = ImageWiseSharedExpertPool(4, 8, num_image_experts=4, top_k=2)
        with torch.no_grad():
            pool.cluster_centers.copy_(torch.eye(4))
        embeddings = torch.tensor([[2., 1., -1., -2.], [-2., -1., -3., -4.]])
        weights, indices = pool.compute_router(embeddings)
        torch.testing.assert_close(indices, torch.tensor([[0, 1], [1, 0]]))
        expected = torch.tensor([[2., 1.], [-1., -2.]]) / embeddings.norm(dim=1, keepdim=True)
        torch.testing.assert_close(weights, expected)

    def test_weighted_sum_contains_two_image_experts_one_token_and_shared(self):
        block = self.build(image_expert_backend="loop").eval()
        uncond_calls, image_calls = [], []
        handles = [block.experts[-1].register_forward_pre_hook(
            lambda module, args: uncond_calls.append(args[0].shape[0]))]
        for expert in block.image_experts.experts:
            handles.append(expert.register_forward_pre_hook(
                lambda module, args: image_calls.append(args[0].numel() // 16)))
        try:
            with torch.no_grad():
                output, _ = block(self.hidden, self.labels)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(uncond_calls, [16])
        self.assertEqual(sum(image_calls), 3 * 16 * 2)
        pool = block.image_experts
        self.assertEqual(tuple(pool.last_expert_indices.shape), (3, 2))
        with torch.no_grad():
            weights, indices, _ = block.compute_router(self.hidden, self.labels)
            self.assertEqual(tuple(indices.shape), (4, 16, 1))
            image_weights, image_ids = pool.compute_router(pool.pooling(self.hidden[[0, 2, 3]]))
            self.assertTrue((image_ids[:, 0] != image_ids[:, 1]).all())
            expected = block.shared_expert(self.hidden)
            expected[1] += block.experts[-1](self.hidden[1])
            for local, image in enumerate((0, 2, 3)):
                for token in range(16):
                    expected[image, token] += weights[image, token, 0] * block.experts[
                        int(indices[image, token, 0])](self.hidden[image, token])
                for slot in range(2):
                    expected[image] += image_weights[local, slot] * pool.experts[
                        int(image_ids[local, slot])](self.hidden[image])
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=2e-6)

    def test_image_contrastive_loss_uses_both_assignments_and_injects_gradients(self):
        pool = self.build().image_experts.double().train()
        hidden = self.hidden[[0, 2, 3]].double()
        with torch.no_grad():
            pool.cluster_centers[:3].copy_(pool.pooling(hidden))
        embeddings = pool.pooling(hidden)
        _, indices = pool.compute_router(embeddings)
        expected = BaseSparseMoeBlock.compute_routing_contrastive_loss(
            pool, embeddings, indices, use_top_k=True) * pool.routing_contrastive_lam
        expected.backward()
        gradients = {name: p.grad.clone() for name, p in pool.named_parameters() if p.grad is not None}
        pool.zero_grad(set_to_none=True)
        (pool(hidden).sum() * 0).backward()
        torch.testing.assert_close(pool.last_routing_contrastive_loss, expected.detach())
        for name, parameter in pool.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-9, atol=1e-9)
        self.assertGreater(pool.pooling.query.grad.abs().sum().item(), 0)
        self.assertGreater(pool.cluster_centers.grad.abs().sum().item(), 0)

    def test_bmm_and_loop_outputs_and_all_gradients_agree(self):
        block = self.build().train()
        results = []
        probe = torch.randn_like(self.hidden)
        for backend in ("bmm", "loop"):
            current = copy.deepcopy(block)
            current.image_experts.set_expert_backend(backend)
            hidden = self.hidden.clone().requires_grad_()
            output, loss = current(hidden, self.labels)
            ((output * probe).sum() + loss).backward()
            grads = {name: p.grad for name, p in current.named_parameters()}
            self.assertTrue(all(gradient is not None for gradient in grads.values()))
            results.append((output, hidden.grad, grads))
        for a, b in zip(results[0][:2], results[1][:2]):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-5)
        for name, gradient in results[0][2].items():
            torch.testing.assert_close(gradient, results[1][2][name], rtol=1e-5, atol=2e-5)

    def test_bfloat16_forward_and_backward_are_finite_in_both_backends(self):
        for backend in ("bmm", "loop"):
            with self.subTest(backend=backend):
                block = self.build(image_expert_backend=backend).train()
                hidden = self.hidden.clone().requires_grad_()
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output, loss = block(hidden, self.labels)
                    objective = output.square().mean() + loss
                objective.backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(hidden.grad).all())
                for name, parameter in block.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_unconditional_images_skip_image_pooling_and_keep_zero_gradients(self):
        block = self.build().train()
        calls = []
        hook = block.image_experts.register_forward_hook(lambda *args: calls.append(1))
        try:
            output, _ = block(self.hidden, torch.full_like(self.labels, 1000))
            torch.testing.assert_close(output, block.shared_expert(self.hidden) + block.experts[-1](self.hidden))
            output.square().mean().backward()
        finally:
            hook.remove()
        self.assertEqual(calls, [])
        self.assertEqual(tuple(block.image_experts.last_expert_indices.shape), (0, 2))
        for name, parameter in block.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            if name.startswith("image_experts.") or name == "cluster_centers":
                self.assertEqual(torch.count_nonzero(parameter.grad), 0, name)
        model = self.model(class_dropout_prob=1.0).train()
        model(torch.randn(4, 4, 8, 8), torch.arange(4).float(), context=torch.tensor([1, 2, 3, 4]))
        self.assertEqual(tuple(model.blocks[1].mlp.image_experts.last_expert_indices.shape), (0, 2))

    def test_image_choices_are_independent_of_batch_composition(self):
        pool = self.build().image_experts.double().eval()
        hidden = self.hidden.double()
        with torch.no_grad():
            together = pool(hidden)
            indices = pool.last_expert_indices.clone()
            separately = torch.cat([pool(image[None]) for image in hidden])
            permutation = torch.tensor([3, 0, 2, 1])
            permuted = pool(hidden[permutation])
        torch.testing.assert_close(separately, together)
        torch.testing.assert_close(permuted, together[permutation])
        torch.testing.assert_close(pool.last_expert_indices, indices[permutation])

    def test_parameter_state_matches_token_top2_and_ffn_work_matches_both_arms_and_base_top3(self):
        current = self.build().eval()
        sibling = self.build(top_k=2, image_top_k=1).eval()
        self.assertEqual(set(current.state_dict()), set(sibling.state_dict()))
        for name, value in current.state_dict().items():
            torch.testing.assert_close(value, sibling.state_dict()[name], rtol=0, atol=0)
        base = BaseSparseMoeBlock(hidden_size=16, **self.config(top_k=3)).eval()
        for block in (base, sibling, current):
            top_k = block.top_k
            def router(hidden, labels, k=top_k):
                batch, tokens, _ = hidden.shape
                ids = ((torch.arange(batch * tokens).view(batch, tokens, 1) + torch.arange(k)) % 4).long()
                weights = hidden.new_ones(batch, tokens, k)
                ids[labels == 1000] = 4
                weights[labels == 1000] = 0
                weights[labels == 1000, :, 0] = 1
                return weights, ids, None
            block.compute_router = router
            counter = FLOPsAccumulator(block)
            counter.start()
            try:
                with torch.no_grad():
                    block(self.hidden, self.labels)
                macs = sum(expert.up_proj.total_ops.item() + expert.down_proj.total_ops.item()
                           for expert in block.modules() if isinstance(expert, MoeMLP))
                self.assertEqual(macs, (3 * 16 * 4 + 16 * 2) * (2 * 16 * 32))
            finally:
                counter.stop()
        self.assertEqual(current.image_experts.expert_backend, "bmm")

    def test_activated_parameters_count_both_selected_image_experts_only(self):
        model = self.model().eval()
        block = model.blocks[1].mlp
        def token_router(hidden_states, labels):
            indices = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], 1, dtype=torch.long)
            indices[labels == 1000] = 4
            return hidden_states.new_ones(indices.shape), indices, None
        block.compute_router = token_router
        block.image_experts.compute_router = lambda embedding: (
            embedding.new_ones(embedding.shape[0], 2), torch.tensor([[0, 2]]).expand(embedding.shape[0], -1))
        tracker = ActivatedParamsTracker(model)
        self.assertEqual(len(tracker.moe_blocks), 1)
        tracker.start()
        try:
            with torch.no_grad():
                model(torch.randn(4, 4, 8, 8), torch.arange(4).float(), context=self.labels)
            tracker.record_forward_pass()
            unused = [block.experts[i] for i in (1, 2, 3)] + [block.image_experts.experts[i] for i in (1, 3)]
            expected = sum(p.numel() for p in model.parameters()) - sum(p.numel() for m in unused for p in m.parameters())
            self.assertEqual(tracker.get_mean_activated_params(), expected)
        finally:
            tracker.stop()

    def test_invalid_image_topk_is_rejected_and_top1_default_is_unchanged(self):
        for invalid in (0, 5, True, 1.5):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "image_top_k"):
                self.build(image_top_k=invalid)
        pool = ImageWiseSharedExpertPool(4, 8, num_image_experts=4)
        scores, indices = pool.compute_router(torch.zeros(3, 4))
        self.assertEqual(pool.top_k, 1)
        self.assertEqual(tuple(indices.shape), (3,))
        self.assertEqual(indices.tolist(), [0, 0, 0])
        self.assertTrue(torch.equal(scores, torch.zeros_like(scores)))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU DDP unavailable")
    def test_ddp_unconditional_then_mixed_batches_in_both_backends(self):
        with tempfile.TemporaryDirectory(prefix=".image_top2_ddp_", dir=Path(__file__).parent) as directory:
            dist.init_process_group("gloo", init_method=Path(directory, "init").resolve().as_uri(), rank=0, world_size=1)
            try:
                for backend in ("bmm", "loop"):
                    model = DistributedDataParallel(self.build(image_expert_backend=backend).train())
                    for labels in (torch.full_like(self.labels, 1000), self.labels, self.labels):
                        model.zero_grad(set_to_none=True)
                        output, loss = model(self.hidden, labels)
                        (output.square().mean() + (loss if loss is not None else 0)).backward()
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
