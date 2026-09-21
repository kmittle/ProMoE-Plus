import contextlib
import copy
import io
import unittest

import torch
import torch.nn.functional as F
from easydict import EasyDict

from models import models_ProMoE_TC as conference
from models import models_ProMoE_TC_expert_cos as expert_cos


MODEL_KWARGS = dict(input_size=8, patch_size=2, in_channels=4, hidden_size=48, depth=2, num_heads=4)


def moe_config(**overrides):
    cfg = EasyDict(
        num_routed_experts=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        load_balance_loss_coef=0,
        norm_topk_prob=False,
        seq_aux=False,
        use_shared_expert=True,
        interleave=True,
        init_MoeMLP=False,
        top_k=1,
        router_weight_mode="identity",
        routing_contrastive_lam=1,
        use_top_k_for_routing_contrastive=True,
        routing_contrastive_temperature=0.07,
    )
    cfg.update(overrides)
    return cfg


def build(module, **overrides):
    torch.manual_seed(0)
    # The conference constructor prints its MoE layout.
    with contextlib.redirect_stdout(io.StringIO()):
        return module.DiT(MoE_config=copy.deepcopy(moe_config(**overrides)), **MODEL_KWARGS)


def perturb(model):
    """Move every weight off its zero-initialised gates and output layer."""
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02 * torch.randn(parameter.shape, generator=generator))
    return model


def dit_inputs():
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(6, 4, 8, 8, generator=generator)
    t = torch.rand(6, generator=generator)
    y = torch.tensor([3, 1000, 7, 7, 1000, 42])
    return x, t, y


def train_step(model):
    """One training forward + backward; returns the loss and the gradients."""
    model.train()
    model.zero_grad(set_to_none=True)
    x, t, y = dit_inputs()
    torch.manual_seed(99)  # class-label dropout draws from the global RNG
    out = model(x, t, y)
    loss = out.float().square().mean()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads


class ExpertCosineEquivalenceTest(unittest.TestCase):
    def test_lam_zero_is_bit_identical_to_conference(self):
        """With the term off the model must reproduce ProMoE_TC exactly."""
        reference = perturb(build(conference))
        candidate = perturb(build(expert_cos, expert_cos_lam=0.0))

        for (name_a, pa), (name_b, pb) in zip(
            reference.named_parameters(), candidate.named_parameters()
        ):
            self.assertEqual(name_a, name_b)
            self.assertTrue(torch.equal(pa, pb), f"initial weights differ at {name_a}")

        loss_a, grads_a = train_step(reference)
        loss_b, grads_b = train_step(candidate)
        self.assertTrue(torch.equal(loss_a, loss_b))
        self.assertEqual(set(grads_a), set(grads_b))
        for name in grads_a:
            self.assertTrue(torch.equal(grads_a[name], grads_b[name]), f"gradient differs at {name}")

    def test_enabling_the_term_changes_the_gradients(self):
        """A live regularizer must move the expert weights, unlike the dead ones."""
        off = perturb(build(expert_cos, expert_cos_lam=0.0))
        on = perturb(build(expert_cos, expert_cos_lam=1.0))
        _, grads_off = train_step(off)
        _, grads_on = train_step(on)

        expert_grads = [
            name for name in grads_on
            if ".mlp.experts." in name and grads_on[name].abs().sum() > 0
        ]
        self.assertTrue(expert_grads, "no routed-expert parameter received a gradient")
        changed = [n for n in expert_grads if not torch.equal(grads_off[n], grads_on[n])]
        self.assertTrue(changed, "the cosine term produced no gradient on any expert")

    def test_diagnostics_are_recorded_and_finite(self):
        model = perturb(build(expert_cos, expert_cos_lam=1.0))
        train_step(model)
        blocks = model.moe_blocks()
        self.assertTrue(blocks)
        for block in blocks:
            self.assertIsNotNone(block.last_expert_cos_loss)
            self.assertIsNotNone(block.last_expert_cos_raw_mean)
            self.assertIsNotNone(block.last_expert_cos_centered_mean)
            for value in (
                block.last_expert_cos_loss,
                block.last_expert_cos_raw_mean,
                block.last_expert_cos_centered_mean,
                block.last_expert_cos_max,
            ):
                self.assertTrue(torch.isfinite(value).all())
            self.assertGreaterEqual(block.last_expert_cos_active, 2)
            self.assertLessEqual(block.last_expert_cos_max.item(), 1.0 + 1e-5)


class ExpertCosineWarmupTest(unittest.TestCase):
    def test_ramp_scales_the_weight_and_reaches_full_strength(self):
        model = perturb(build(expert_cos, expert_cos_lam=2.0, expert_cos_warmup_steps=100))
        model.train()
        for step, want in ((0, 2.0 * 1 / 100), (49, 2.0 * 50 / 100), (99, 2.0), (500, 2.0)):
            model._set_expert_cos_state(step)
            self.assertAlmostEqual(model.expert_cos_last_state["lam"], want, places=6)
            for block in model.moe_blocks():
                self.assertAlmostEqual(block.expert_cos_effective_lam, want, places=6)

    def test_no_warmup_keeps_full_weight_without_a_step(self):
        model = perturb(build(expert_cos, expert_cos_lam=2.0, expert_cos_warmup_steps=0))
        model.train()
        model._set_expert_cos_state(None)  # must not raise
        self.assertEqual(model.expert_cos_last_state["lam"], 2.0)

    def test_warmup_without_a_step_is_an_error(self):
        model = perturb(build(expert_cos, expert_cos_lam=2.0, expert_cos_warmup_steps=100))
        model.train()
        with self.assertRaises(ValueError):
            model(*dit_inputs()[:1], dit_inputs()[1], dit_inputs()[2])

    def test_eval_mode_ignores_the_ramp(self):
        model = perturb(build(expert_cos, expert_cos_lam=2.0, expert_cos_warmup_steps=100))
        model.eval()
        model._set_expert_cos_state(None)
        self.assertEqual(model.expert_cos_last_state["lam"], 2.0)


class ExpertCosineLossMathTest(unittest.TestCase):
    """The loss on hand-built pooled vectors, against an explicit reference."""

    def _block(self, **overrides):
        torch.manual_seed(0)
        kwargs = dict(
            num_routed_experts=4,
            hidden_size=6,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            top_k=1,
            router_weight_mode="identity",
            routing_contrastive_lam=1,
            use_top_k_for_routing_contrastive=True,
            routing_contrastive_temperature=0.07,
        )
        kwargs.update(overrides)
        return expert_cos.SparseMoeBlock(**kwargs)

    @staticmethod
    def _reference(vectors, center, margin):
        stacked = torch.stack(vectors).float()
        num = stacked.size(0)
        if center:
            stacked = stacked - stacked.mean(dim=0, keepdim=True)
            target = -1.0 / (num - 1) + margin
        else:
            target = margin
        normed = F.normalize(stacked, p=2, dim=1)
        sim = (normed @ normed.T).clamp(-1.0, 1.0)
        mask = torch.triu(torch.ones(num, num, dtype=torch.bool), diagonal=1)
        return (sim[mask] - target).square().mean()

    def test_matches_explicit_reference_centered_and_raw(self):
        generator = torch.Generator().manual_seed(11)
        vectors = [torch.randn(6, generator=generator) for _ in range(4)]
        for center in (True, False):
            block = self._block(expert_cos_lam=1.0, expert_cos_center=center)
            got = block.compute_expert_cosine_loss(vectors)
            want = self._reference(vectors, center=center, margin=0.0)
            self.assertTrue(torch.allclose(got, want, atol=1e-6),
                            f"center={center}: {got} vs {want}")

    def test_margin_shifts_the_target(self):
        """A margin equal to the gap makes an otherwise-penalised set free."""
        base = torch.randn(6, generator=torch.Generator().manual_seed(3))
        vectors = [base, base + 1e-3, torch.randn(6), torch.randn(6)]
        strict = self._block(expert_cos_lam=1.0, expert_cos_center=False, expert_cos_margin=0.0)
        tolerant = self._block(expert_cos_lam=1.0, expert_cos_center=False, expert_cos_margin=1.0)
        # Target 0 penalises the near-duplicate pair (cos ~ 1); target 1 does not.
        self.assertGreater(strict.compute_expert_cosine_loss(vectors[:2]).item(),
                           tolerant.compute_expert_cosine_loss(vectors[:2]).item())

    def test_simplex_is_the_minimum_and_is_reachable(self):
        """Evenly spread directions hit the floor; a collapsed pair does not."""
        block = self._block(expert_cos_lam=1.0, expert_cos_center=True)
        # A regular simplex in 3-D: 4 vertices of a tetrahedron, every pair at -1/3.
        simplex = [torch.tensor(v, dtype=torch.float32) for v in
                   ([1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.])]
        self.assertAlmostEqual(block.compute_expert_cosine_loss(simplex).item(), 0.0, places=5)
        collapsed = [simplex[0], simplex[0].clone(), simplex[1], simplex[2]]
        self.assertGreater(block.compute_expert_cosine_loss(collapsed).item(), 0.05)

    def test_centered_term_is_alive_in_high_dimensions(self):
        """The configuration that killed relu(cos): K=12 pooled vectors in 768-D."""
        generator = torch.Generator().manual_seed(5)
        shared = 3.0 * torch.ones(768)
        vectors = [shared + torch.randn(768, generator=generator) for _ in range(12)]
        block = expert_cos.SparseMoeBlock(
            num_routed_experts=12, hidden_size=768, moe_intermediate_size=8,
            shared_expert_intermediate_size=8, top_k=1, router_weight_mode="identity",
            routing_contrastive_lam=1, use_top_k_for_routing_contrastive=True,
            routing_contrastive_temperature=0.07, expert_cos_lam=1.0, expert_cos_center=True,
        )
        leaves = [v.clone().requires_grad_(True) for v in vectors]
        loss = block.compute_expert_cosine_loss(leaves)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertGreater(sum(v.grad.abs().sum() for v in leaves).item(), 1e-6)
        # And the centered mean is the -1/(K-1) identity, as the docstring claims.
        self.assertAlmostEqual(block.last_expert_cos_centered_mean.item(), -1.0 / 11, places=4)

    def test_fewer_than_two_experts_is_zero_and_flagged(self):
        block = self._block(expert_cos_lam=1.0)
        loss = block.compute_expert_cosine_loss([torch.randn(6)])
        self.assertEqual(loss.item(), 0.0)
        self.assertIsNone(block.last_expert_cos_loss)
        self.assertEqual(block.last_expert_cos_active, 1.0)

    def test_loss_has_a_nonzero_gradient(self):
        """The whole point: unlike exp(-L2/tau), this term must not be inert."""
        generator = torch.Generator().manual_seed(13)
        vectors = [torch.randn(6, generator=generator, requires_grad=True) for _ in range(4)]
        block = self._block(expert_cos_lam=1.0)
        block.compute_expert_cosine_loss(vectors).backward()
        total = sum(v.grad.abs().sum() for v in vectors)
        self.assertGreater(total.item(), 1e-3)

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            self._block(expert_cos_lam=-1.0)
        with self.assertRaises(ValueError):
            self._block(expert_cos_margin=2.0)


if __name__ == "__main__":
    unittest.main()
