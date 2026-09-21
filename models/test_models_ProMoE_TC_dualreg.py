import contextlib
import copy
import io
import unittest

import torch
import torch.nn.functional as F
from easydict import EasyDict

from models import models_ProMoE_TC as conference
from models import models_ProMoE_TC_dualreg as dualreg
from models import models_ProMoE_TC_expert_cos as expert_cos
from models import models_ProMoE_TC_lsreg as lsreg


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


def train_step(model, **forward_kwargs):
    model.train()
    model.zero_grad(set_to_none=True)
    x, t, y = dit_inputs()
    torch.manual_seed(99)  # class-label dropout draws from the global RNG
    out = model(x, t, y, **forward_kwargs)
    out.float().square().mean().backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return out.detach(), grads


def assert_identical(case, reference, candidate, **candidate_kwargs):
    for (name_a, pa), (name_b, pb) in zip(reference.named_parameters(), candidate.named_parameters()):
        case.assertEqual(name_a, name_b)
        case.assertTrue(torch.equal(pa, pb), f"initial weights differ at {name_a}")
    out_a, grads_a = train_step(reference)
    out_b, grads_b = train_step(candidate, **candidate_kwargs)
    case.assertTrue(torch.equal(out_a, out_b), "forward output differs")
    case.assertEqual(set(grads_a), set(grads_b))
    for name in grads_a:
        case.assertTrue(torch.equal(grads_a[name], grads_b[name]), f"gradient differs at {name}")


class DualRegEquivalenceTest(unittest.TestCase):
    """The three exact equivalences the combination study depends on."""

    def test_both_terms_off_is_the_conference_model(self):
        assert_identical(self, perturb(build(conference)),
                         perturb(build(dualreg, expert_reg_lam=0.0, ls_diag_strength=0.0)))

    def test_only_expert_regularizer_matches_expert_cos(self):
        """ls off: must reproduce the standalone cosine model exactly."""
        reference = perturb(build(expert_cos, expert_cos_lam=0.7, expert_cos_center=True))
        candidate = perturb(build(dualreg, expert_reg_form="cosine", expert_reg_lam=0.7,
                                  expert_reg_center=True, ls_diag_strength=0.0))
        assert_identical(self, reference, candidate)

    def test_only_lsreg_matches_the_lsreg_model(self):
        """expert term off: must reproduce lsreg's diag mode exactly."""
        reference = perturb(build(lsreg, ls_apply="diag", ls_diag_sign=1.0,
                                  ls_diag_strength=0.05, ls_ema_beta=0.9))
        candidate = perturb(build(dualreg, expert_reg_lam=0.0, ls_diag_sign=1.0,
                                  ls_diag_strength=0.05, ls_ema_beta=0.9))
        assert_identical(self, reference, candidate)

    def test_both_terms_on_differs_from_each_alone(self):
        both = perturb(build(dualreg, expert_reg_lam=0.7, ls_diag_strength=0.05))
        only_ls = perturb(build(dualreg, expert_reg_lam=0.0, ls_diag_strength=0.05))
        only_reg = perturb(build(dualreg, expert_reg_lam=0.7, ls_diag_strength=0.0))
        _, g_both = train_step(both)
        _, g_ls = train_step(only_ls)
        _, g_reg = train_step(only_reg)
        expert_names = [n for n in g_both if ".mlp.experts." in n]
        self.assertTrue(expert_names)
        # The expert term reaches the expert parameters directly.
        self.assertTrue(any(not torch.equal(g_both[n], g_ls[n]) for n in expert_names))
        # LS-Reg does NOT: its gradient goes to cluster_centers and the token
        # representations, i.e. to layers *before* this block, so with a single
        # MoE block the expert gradients are untouched.  Compare all parameters.
        self.assertTrue(any(not torch.equal(g_both[n], g_reg[n]) for n in g_both))
        centers = [n for n in g_both if "cluster_centers" in n]
        self.assertTrue(centers)
        self.assertTrue(any(not torch.equal(g_both[n], g_reg[n]) for n in centers))


class DualRegExpertTermTest(unittest.TestCase):
    def _block(self, **overrides):
        torch.manual_seed(0)
        kwargs = dict(
            num_routed_experts=4, hidden_size=6, moe_intermediate_size=8,
            shared_expert_intermediate_size=8, top_k=1, router_weight_mode="identity",
            routing_contrastive_lam=1, use_top_k_for_routing_contrastive=True,
            routing_contrastive_temperature=0.07,
        )
        kwargs.update(overrides)
        return dualreg.SparseMoeBlock(**kwargs)

    def test_cosine_matches_an_explicit_reference(self):
        generator = torch.Generator().manual_seed(11)
        vectors = [torch.randn(6, generator=generator) for _ in range(4)]
        for center in (True, False):
            block = self._block(expert_reg_lam=1.0, expert_reg_center=center)
            stacked = torch.stack(vectors).float()
            target = -1.0 / 3 if center else 0.0
            if center:
                stacked = stacked - stacked.mean(dim=0, keepdim=True)
            normed = F.normalize(stacked, p=2, dim=1)
            sim = (normed @ normed.T).clamp(-1.0, 1.0)
            mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
            want = (sim[mask] - target).square().mean()
            self.assertTrue(torch.allclose(block.compute_expert_reg_loss(vectors), want, atol=1e-6))

    def test_l2_form_matches_an_explicit_reference(self):
        generator = torch.Generator().manual_seed(12)
        vectors = [torch.randn(6, generator=generator) for _ in range(4)]
        block = self._block(expert_reg_form="l2", expert_reg_lam=1.0, expert_reg_temperature=2.0)
        stacked = torch.stack(vectors).float()
        diff = stacked.unsqueeze(0) - stacked.unsqueeze(1)
        dist = diff.square().sum(-1).clamp_min(1e-12).sqrt()
        mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
        want = torch.exp(-dist[mask] / 2.0).mean()
        self.assertTrue(torch.allclose(block.compute_expert_reg_loss(vectors), want, atol=1e-6))

    def test_l2_form_dies_as_experts_spread_but_cosine_does_not(self):
        """The measured failure: pooled distances grow ~20 -> ~92 during training."""
        generator = torch.Generator().manual_seed(5)
        base = [torch.randn(768, generator=generator) for _ in range(12)]
        l2 = self._block(expert_reg_form="l2", expert_reg_lam=1.0, expert_reg_temperature=10.0)
        cos = self._block(expert_reg_form="cosine", expert_reg_lam=1.0, expert_reg_center=True)
        unit = (base[0] - base[1]).norm()                                # typical pair distance
        near = [v * (20.0 / unit) for v in base]                         # pooled spread ~20
        far = [v * (92.0 / unit) for v in base]                          # pooled spread ~92
        l2_near, l2_far = l2.compute_expert_reg_loss(near), l2.compute_expert_reg_loss(far)
        cos_near, cos_far = cos.compute_expert_reg_loss(near), cos.compute_expert_reg_loss(far)
        self.assertGreater(l2_near.item() / max(l2_far.item(), 1e-30), 100.0)
        self.assertTrue(torch.allclose(cos_near, cos_far, atol=1e-5),
                        "the cosine form must be scale-free")

    def test_simplex_is_the_reachable_minimum(self):
        block = self._block(expert_reg_lam=1.0, expert_reg_center=True)
        simplex = [torch.tensor(v, dtype=torch.float32) for v in
                   ([1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.])]
        self.assertAlmostEqual(block.compute_expert_reg_loss(simplex).item(), 0.0, places=5)
        collapsed = [simplex[0], simplex[0].clone(), simplex[1], simplex[2]]
        self.assertGreater(block.compute_expert_reg_loss(collapsed).item(), 0.05)

    def test_gradient_is_not_inert(self):
        generator = torch.Generator().manual_seed(13)
        vectors = [torch.randn(6, generator=generator, requires_grad=True) for _ in range(4)]
        self._block(expert_reg_lam=1.0).compute_expert_reg_loss(vectors).backward()
        self.assertGreater(sum(v.grad.abs().sum() for v in vectors).item(), 1e-3)

    def test_fewer_than_two_experts_is_zero_and_flagged(self):
        block = self._block(expert_reg_lam=1.0)
        self.assertEqual(block.compute_expert_reg_loss([torch.randn(6)]).item(), 0.0)
        self.assertIsNone(block.last_expert_reg_loss)
        self.assertEqual(block.last_expert_reg_active, 1.0)

    def test_rejects_invalid_configuration(self):
        for bad in (dict(expert_reg_form="nope"), dict(expert_reg_lam=-1.0),
                    dict(expert_reg_margin=2.0), dict(expert_reg_temperature=0.0),
                    dict(ls_diag_strength=-0.1), dict(ls_diag_sign=0.5)):
            with self.assertRaises(ValueError):
                self._block(**bad)


class DualRegWarmupAndDiagnosticsTest(unittest.TestCase):
    def test_ramp_scales_the_weight(self):
        model = perturb(build(dualreg, expert_reg_lam=2.0, expert_reg_warmup_steps=100))
        model.train()
        for step, want in ((0, 0.02), (49, 1.0), (99, 2.0), (500, 2.0)):
            model._set_expert_reg_state(step)
            self.assertAlmostEqual(model.dualreg_last_state["lam"], want, places=6)
            for block in model.moe_blocks():
                self.assertAlmostEqual(block.expert_reg_effective_lam, want, places=6)

    def test_warmup_without_a_step_is_an_error(self):
        model = perturb(build(dualreg, expert_reg_lam=2.0, expert_reg_warmup_steps=100))
        model.train()
        x, t, y = dit_inputs()
        with self.assertRaises(ValueError):
            model(x, t, y)

    def test_diagnostics_cover_both_terms(self):
        model = perturb(build(dualreg, expert_reg_lam=1.0, ls_diag_strength=0.05))
        train_step(model)
        for block in model.moe_blocks():
            for value in (block.last_expert_reg_loss, block.last_expert_reg_raw_mean,
                          block.last_expert_reg_centered_mean, block.last_expert_reg_max,
                          block.last_expert_reg_dist_min, block.last_mean_eps):
                self.assertIsNotNone(value)
                self.assertTrue(torch.isfinite(value).all())
            self.assertGreaterEqual(block.last_expert_reg_active, 2)


if __name__ == "__main__":
    unittest.main()
