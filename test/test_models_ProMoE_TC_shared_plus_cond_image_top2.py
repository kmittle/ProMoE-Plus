"""Behavior and FFN-budget checks for token Top-2 plus image Top-1."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel

from analyses.flops.flops_counter import FLOPsAccumulator
from models.models_ProMoE_TC import SparseMoeBlock as BaseSparseMoeBlock
from models.models_ProMoE_TC_shared_plus_cond_image import DiT, SparseMoeBlock
from models.modules import MoeMLP


def moe_config(**overrides):
    config = EasyDict(
        num_routed_experts=4, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, uncond_expert_intermediate_size=32,
        image_expert_intermediate_size=32, top_k=2, router_weight_mode="identity",
        routing_contrastive_lam=1.0, use_top_k_for_routing_contrastive=True,
        routing_contrastive_temperature=0.07, interleave=True, init_MoeMLP=False,
        use_shared_expert=True, use_uncond_expert=True,
        num_image_experts=4, image_expert_backend="bmm",
        image_routing_contrastive_lam=1.0, image_routing_contrastive_temperature=0.07,
    )
    config.update(overrides)
    return config


class TokenTop2ImageTop1Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        self.hidden = torch.randn(4, 16, 16)
        self.labels = torch.tensor([1, 1000, 2, 3])

    def build(self, **overrides):
        torch.manual_seed(531)
        return SparseMoeBlock(hidden_size=16, **moe_config(**overrides))

    def test_weighted_sum_contains_two_token_experts_one_image_and_shared(self):
        block = self.build().eval()
        uncond_calls = []
        hook = block.experts[-1].register_forward_pre_hook(
            lambda module, inputs: uncond_calls.append(inputs[0].shape[0]))
        try:
            with torch.no_grad():
                output, _ = block(self.hidden, self.labels)
        finally:
            hook.remove()
        self.assertEqual(uncond_calls, [16])
        self.assertEqual(block.image_experts.last_expert_indices.numel(), 3)
        with torch.no_grad():
            weights, indices, _ = block.compute_router(self.hidden, self.labels)
            self.assertEqual(tuple(indices.shape), (4, 16, 2))
            self.assertTrue((indices[[0, 2, 3], :, 0] != indices[[0, 2, 3], :, 1]).all())
            self.assertTrue((weights[1, :, 0] == 1).all())
            self.assertTrue((weights[1, :, 1] == 0).all())
            expected = block.shared_expert(self.hidden)
            expected[1] += block.experts[-1](self.hidden[1])
            cond_ids = [0, 2, 3]
            pool = block.image_experts
            image_weights, image_ids = pool.compute_router(pool.pooling(self.hidden[cond_ids]))
            for local, batch_index in enumerate(cond_ids):
                for token_index in range(self.hidden.shape[1]):
                    token = self.hidden[batch_index, token_index]
                    for slot in range(2):
                        expert = block.experts[int(indices[batch_index, token_index, slot])]
                        expected[batch_index, token_index] += (
                            weights[batch_index, token_index, slot] * expert(token))
                expected[batch_index] += (
                    image_weights[local] * pool.experts[int(image_ids[local])](self.hidden[batch_index]))
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=2e-6)

    def test_token_contrastive_loss_uses_both_assignments(self):
        block = self.build().train()
        captured = []
        original = block.compute_routing_contrastive_loss

        def record(embeddings, assignments, use_top_k=False):
            captured.append((embeddings, assignments.detach().clone(), use_top_k))
            return original(embeddings, assignments, use_top_k=use_top_k)

        block.compute_routing_contrastive_loss = record
        _, loss = block(self.hidden, self.labels)
        _, indices, _ = block.compute_router(self.hidden, self.labels)
        self.assertEqual(len(captured), 1)
        embeddings, assignments, use_top_k = captured[0]
        self.assertTrue(use_top_k)
        self.assertEqual(tuple(assignments.shape), (48, 2))
        torch.testing.assert_close(assignments, indices[[0, 2, 3]].reshape(-1, 2))
        expected = original(embeddings, assignments, use_top_k=True)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(block.cluster_centers.grad.abs().sum().item(), 0)

    def test_unconditional_batch_skips_image_branch_and_keeps_zero_gradients(self):
        block = self.build().train()
        calls = []
        hook = block.image_experts.register_forward_hook(lambda *args: calls.append(1))
        try:
            output, _ = block(self.hidden, torch.full_like(self.labels, 1000))
            expected = block.shared_expert(self.hidden) + block.experts[-1](self.hidden)
            torch.testing.assert_close(output, expected)
            output.square().mean().backward()
        finally:
            hook.remove()
        self.assertEqual(calls, [])
        self.assertEqual(block.image_experts.last_expert_indices.numel(), 0)
        for name, parameter in block.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            if name.startswith("image_experts.") or name == "cluster_centers":
                self.assertEqual(torch.count_nonzero(parameter.grad), 0, name)

    def test_bmm_and_loop_outputs_and_gradients_agree_at_token_top2(self):
        original = self.build().train()
        probe = torch.randn_like(self.hidden)
        results = []
        for backend in ("bmm", "loop"):
            block = copy.deepcopy(original)
            block.image_experts.set_expert_backend(backend)
            hidden = self.hidden.clone().requires_grad_()
            output, loss = block(hidden, self.labels)
            ((output * probe).sum() + loss).backward()
            gradients = {name: p.grad for name, p in block.named_parameters()}
            self.assertTrue(all(g is not None for g in gradients.values()))
            results.append((output, hidden.grad, gradients))
        for left, right in zip(results[0][:2], results[1][:2]):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=2e-5)
        for name, gradient in results[0][2].items():
            torch.testing.assert_close(gradient, results[1][2][name], rtol=1e-5, atol=2e-5)

    def test_bfloat16_mixed_batch_has_finite_forward_and_backward(self):
        for backend in ("bmm", "loop"):
            with self.subTest(backend=backend):
                block = self.build(image_expert_backend=backend).train()
                hidden = self.hidden.clone().requires_grad_()
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    output, loss = block(hidden, self.labels)
                    objective = output.square().mean() + loss
                objective.backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(hidden.grad).all())
                for name, parameter in block.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_label_dropout_bypasses_image_branch_in_full_model(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = DiT(input_size=8, patch_size=2, in_channels=4, hidden_size=16,
                        depth=2, num_heads=2, mlp_ratio=4, class_dropout_prob=1.0,
                        MoE_config=moe_config()).train()
        calls = []
        hook = model.blocks[1].mlp.image_experts.register_forward_hook(lambda *args: calls.append(1))
        try:
            output = model(torch.randn(4, 4, 8, 8), torch.arange(4).float(),
                           context=torch.tensor([1, 2, 3, 4]))
            self.assertEqual(tuple(output.shape), (4, 8, 8, 8))
            self.assertEqual(calls, [])
        finally:
            hook.remove()

    def test_ffn_matrix_work_matches_base_top3(self):
        # Exercise all token experts, avoiding the inherited dummy calls for unused
        # token experts. Image experts are measured via the FLOPs loop backend.
        def router(top_k):
            def compute(hidden, labels):
                batch, tokens, _ = hidden.shape
                indices = ((torch.arange(batch * tokens).view(batch, tokens, 1)
                            + torch.arange(top_k)) % 4).long()
                weights = hidden.new_ones(batch, tokens, top_k)
                indices[labels == 1000] = 4
                weights[labels == 1000] = 0
                weights[labels == 1000, :, 0] = 1
                return weights, indices, None
            return compute

        augmented = self.build().eval()
        base = BaseSparseMoeBlock(hidden_size=16, **moe_config(top_k=3)).eval()
        measurements = []
        for block, top_k in ((base, 3), (augmented, 2)):
            block.compute_router = router(top_k)
            counter = FLOPsAccumulator(block)
            counter.start()
            try:
                with torch.no_grad():
                    block(self.hidden, self.labels)
                ffn_macs = sum(
                    expert.up_proj.total_ops.item() + expert.down_proj.total_ops.item()
                    for expert in block.modules() if isinstance(expert, MoeMLP))
                measurements.append((ffn_macs, counter.collect()))
            finally:
                counter.stop()
        # 3 conditional images: 4 FFNs/token; 1 unconditional image: 2 FFNs/token.
        expected = (3 * 16 * 4 + 1 * 16 * 2) * (2 * 16 * 32)
        self.assertEqual(measurements[0][0], expected)
        self.assertEqual(measurements[1][0], expected)
        self.assertGreater(measurements[1][1], measurements[0][1])
        self.assertEqual(augmented.image_experts.expert_backend, "bmm")

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU DDP unavailable")
    def test_ddp_handles_unconditional_then_mixed_top2_batches(self):
        with tempfile.TemporaryDirectory(prefix=".top2_ddp_", dir=Path(__file__).parent) as directory:
            dist.init_process_group("gloo", init_method=Path(directory, "rendezvous").resolve().as_uri(),
                                    rank=0, world_size=1)
            try:
                block = DistributedDataParallel(self.build().train())
                for labels in (torch.full_like(self.labels, 1000), self.labels, self.labels):
                    block.zero_grad(set_to_none=True)
                    output, loss = block(self.hidden, labels)
                    objective = output.square().mean()
                    if loss is not None:
                        objective = objective + loss
                    objective.backward()
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
