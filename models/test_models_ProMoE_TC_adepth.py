"""Regression tests for adaptive-depth inference and scorer-gradient ablations."""

import copy
import unittest

import torch

from models.models_ProMoE_TC_adepth import SparseMoeBlock


class AdaptiveDepthInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.block = SparseMoeBlock(
            num_routed_experts=2,
            hidden_size=8,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            top_k=1,
            router_weight_mode="identity",
            routing_contrastive_lam=0,
            alloc_mode="fixed_q",
            depth_q=0.2,
            depth_warmup=100,
        )
        with torch.no_grad():
            # All conditional tokens choose expert 0. Their depth scores differ.
            self.block.cluster_centers.zero_()
            self.block.cluster_centers[0, 0] = 1
            self.block.cluster_centers[1, 0] = -1
            self.block.depth_gate.weight.zero_()
            self.block.depth_gate.weight[0, 1] = 1
            self.block.depth_gate.bias.zero_()
            self.block.deepen_gain.fill_(0.7)
        self.hidden = torch.randn(2, 10, 8)
        self.hidden[:, :, 0] = 2
        self.hidden[0, :, 1] = torch.linspace(-1, 1, 10)
        self.labels = torch.tensor([1, 1000])

    def forward_with_counts(self, block):
        calls = {0: [], 2: []}
        handles = []
        for expert_id in calls:
            def record(module, inputs, output, expert_id=expert_id):
                calls[expert_id].append(inputs[0].shape[0])
            handles.append(block.experts[expert_id].register_forward_hook(record))
        try:
            with torch.no_grad():
                output, _ = block(self.hidden, self.labels)
        finally:
            for handle in handles:
                handle.remove()
        return output, calls

    def test_trained_checkpoints_keep_skip_and_second_pass_in_eval(self):
        for q, skipped in ((0.1, 1), (0.2, 2), (0.3, 3)):
            with self.subTest(q=q):
                self.block.depth_q = q
                self.block._depth_step.fill_(300001)
                trained = copy.deepcopy(self.block).train()
                restored = copy.deepcopy(self.block).eval()
                train_output, train_calls = self.forward_with_counts(trained)
                eval_output, eval_calls = self.forward_with_counts(restored)
                self.assertEqual(eval_calls[0], [10 - skipped, skipped])
                self.assertEqual(eval_calls[2], [10])  # unconditional: one pass
                self.assertEqual(train_calls, eval_calls)
                torch.testing.assert_close(train_output, eval_output, rtol=0, atol=0)
                self.assertEqual(restored._depth_step.item(), 300001)
                self.assertEqual(trained._depth_step.item(), 300002)

    def test_eval_respects_partial_warmup_without_advancing_it(self):
        self.block._depth_step.fill_(50)  # effective q=0.1, not the final 0.2
        trained = copy.deepcopy(self.block).train()
        restored = copy.deepcopy(self.block).eval()
        train_output, _ = self.forward_with_counts(trained)
        output, calls = self.forward_with_counts(restored)
        self.assertEqual(calls[0], [9, 1])
        torch.testing.assert_close(output, train_output, rtol=0, atol=0)
        repeated, repeated_calls = self.forward_with_counts(restored)
        torch.testing.assert_close(output, repeated, rtol=0, atol=0)
        self.assertEqual(calls, repeated_calls)
        self.assertEqual(restored._depth_step.item(), 50)

    def test_fresh_checkpoint_keeps_single_pass(self):
        self.block.eval()
        self.assertEqual(self.block._depth_step.item(), 0)
        output, calls = self.forward_with_counts(self.block)
        self.assertEqual(calls[0], [10])
        self.block.alloc_mode = "none"
        baseline, _ = self.forward_with_counts(self.block)
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)

    def test_eval_loads_depth_schedule_from_state_dict(self):
        self.block._depth_step.fill_(300001)
        restored = copy.deepcopy(self.block)
        restored._depth_step.zero_()
        restored.load_state_dict(self.block.state_dict(), strict=True)
        restored.eval()
        _, calls = self.forward_with_counts(restored)
        self.assertEqual(calls[0], [8, 2])
        self.assertEqual(restored._depth_step.item(), 300001)

    def backward_with_score_grads(self, block):
        scores = []

        def capture_scores(module, inputs, output):
            # Capture the ten conditional scores, excluding the DDP dummy touch.
            if inputs[0].shape[0] == self.hidden.shape[1]:
                output.retain_grad()
                scores.append(output)

        hidden = self.hidden.detach().clone().requires_grad_(True)
        block.zero_grad(set_to_none=True)
        handle = block.depth_gate.register_forward_hook(capture_scores)
        try:
            output, _ = block(hidden, self.labels)
            output.square().sum().backward()
        finally:
            handle.remove()
        self.assertEqual(len(scores), 1)
        score_grad = scores[0].grad
        if score_grad is None:
            score_grad = torch.zeros_like(scores[0])
        return output.detach(), score_grad.detach().flatten(), hidden.grad

    def test_p_only_preserves_forward_but_removes_normal_score_gradients(self):
        self.block._depth_step.fill_(300001)
        keep_ste = copy.deepcopy(self.block).train()
        p_only = copy.deepcopy(self.block).train()
        p_only.depth_keep_ste = False
        ste_output, ste_grads, _ = self.backward_with_score_grads(keep_ste)
        p_output, p_grads, input_grads = self.backward_with_score_grads(p_only)
        torch.testing.assert_close(ste_output, p_output, rtol=0, atol=0)
        # Increasing depth scores: tokens 0:2 skip, 2:8 run once, 8:10 loop.
        self.assertEqual(torch.count_nonzero(ste_grads[:2]).item(), 0)
        self.assertEqual(torch.count_nonzero(ste_grads[2:8]).item(), 6)
        self.assertEqual(torch.count_nonzero(p_grads[:8]).item(), 0)
        self.assertEqual(torch.count_nonzero(p_grads[8:]).item(), 2)
        self.assertGreater(p_only.depth_gate.weight.grad.abs().sum().item(), 0)
        self.assertGreater(input_grads[0, 2:8].abs().sum().item(), 0)
        expert_grads = [p.grad for p in p_only.experts[0].parameters()]
        self.assertTrue(any(g is not None and g.abs().sum().item() > 0 for g in expert_grads))

    def test_p_only_zero_gain_still_trains_gain_and_experts(self):
        self.block.depth_keep_ste = False
        self.block._depth_step.fill_(300001)
        with torch.no_grad():
            self.block.deepen_gain.zero_()
        _, score_grads, _ = self.backward_with_score_grads(self.block)
        self.assertEqual(torch.count_nonzero(score_grads).item(), 0)
        self.assertEqual(torch.count_nonzero(self.block.depth_gate.weight.grad).item(), 0)
        self.assertGreater(self.block.deepen_gain.grad.abs().sum().item(), 0)
        expert_grads = [p.grad for p in self.block.experts[0].parameters()]
        self.assertTrue(any(g is not None and g.abs().sum().item() > 0 for g in expert_grads))

    def test_p_only_eval_keeps_loop_and_does_not_advance_schedule(self):
        self.block.depth_keep_ste = False
        self.block._depth_step.fill_(300001)
        trained = copy.deepcopy(self.block).train()
        restored = copy.deepcopy(self.block).eval()
        train_output, train_calls = self.forward_with_counts(trained)
        eval_output, eval_calls = self.forward_with_counts(restored)
        torch.testing.assert_close(train_output, eval_output, rtol=0, atol=0)
        self.assertEqual(train_calls, eval_calls)
        self.assertEqual(eval_calls[0], [8, 2])
        self.assertEqual(eval_calls[2], [10])
        self.assertEqual(restored._depth_step.item(), 300001)

    def test_ste_flag_preserves_initial_weights_and_rng(self):
        def construct(**kwargs):
            torch.manual_seed(2718)
            block = SparseMoeBlock(
                num_routed_experts=2, hidden_size=8, moe_intermediate_size=16,
                shared_expert_intermediate_size=16, top_k=1,
                alloc_mode="fixed_q", **kwargs,
            )
            return block, torch.get_rng_state()

        default, default_rng = construct()
        explicit, explicit_rng = construct(depth_keep_ste=True)
        p_only, p_only_rng = construct(depth_keep_ste=False)
        self.assertTrue(default.depth_keep_ste)
        self.assertFalse(p_only.depth_keep_ste)
        for other, rng in ((explicit, explicit_rng), (p_only, p_only_rng)):
            torch.testing.assert_close(default_rng, rng, rtol=0, atol=0)
            self.assertEqual(default.state_dict().keys(), other.state_dict().keys())
            for name, value in default.state_dict().items():
                torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
