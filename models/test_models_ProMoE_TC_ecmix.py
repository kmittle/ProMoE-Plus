import contextlib
import copy
import io
import math
import unittest

import torch
import torch.nn.functional as F
from easydict import EasyDict

from models import models_ProMoE_EC_batch_choice as ec_batch_choice
from models import models_ProMoE_TC as conference
from models import models_ProMoE_TC_ecmix as ecmix


MODEL_KWARGS = dict(input_size=8, patch_size=2, in_channels=4, hidden_size=48, depth=2, num_heads=4)
BLOCK_KWARGS = dict(
    num_routed_experts=4,
    hidden_size=16,
    moe_intermediate_size=8,
    shared_expert_intermediate_size=8,
    top_k=1,
    router_weight_mode="identity",
    routing_contrastive_lam=1,
    use_top_k_for_routing_contrastive=True,
    routing_contrastive_temperature=0.07,
)


def moe_config(ecmix_config=None):
    config = EasyDict(
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
    if ecmix_config is not None:
        config.ecmix_config = EasyDict(ecmix_config)
    return config


def build(module, config):
    torch.manual_seed(0)
    # The conference constructor prints its MoE layout.
    with contextlib.redirect_stdout(io.StringIO()):
        return module.DiT(MoE_config=copy.deepcopy(config), **MODEL_KWARGS)


def perturb(model):
    """Move every weight off its initial value.

    DiT zero-initialises its adaLN gates and output layer, so a freshly built
    model predicts zeros whatever its MoE path does.  The same seed and the
    same parameter order give both compared models the same perturbation.
    """
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


def train_step(model, **kwargs):
    model.train()
    model.zero_grad(set_to_none=True)
    x, t, y = dit_inputs()
    torch.manual_seed(99)  # class-label dropout draws from the global RNG
    output = model(x, t, y, **kwargs)
    output.float().square().mean().backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), grads


def eval_output(model, **kwargs):
    model.eval()
    x, t, y = dit_inputs()
    with torch.no_grad():
        return model(x, t, y, **kwargs)


def moe_blocks(model):
    return [block.mlp for block in model.blocks if block.use_moe]


class ScheduleTests(unittest.TestCase):
    def test_constant_share_spaces_ec_steps_evenly(self):
        for ratio, period in ((0.05, 20), (0.10, 10), (0.25, 4), (0.50, 2)):
            with self.subTest(ratio=ratio):
                config = ecmix.validate_ecmix_config({"mode": "step", "ratio": ratio})
                ec_steps = [step for step in range(400) if ecmix.is_ec_step(step, config)]
                self.assertEqual(ec_steps, list(range(period - 1, 400, period)))

    def test_zero_and_full_shares(self):
        off = ecmix.validate_ecmix_config({"mode": "step", "ratio": 0.0})
        full = ecmix.validate_ecmix_config({"mode": "step", "ratio": 1.0})
        self.assertFalse(any(ecmix.is_ec_step(step, off) for step in range(100)))
        self.assertTrue(all(ecmix.is_ec_step(step, full) for step in range(100)))

    def test_cosine_ratio_reaches_zero_at_the_anneal_end(self):
        config = ecmix.validate_ecmix_config(
            {"mode": "loss", "schedule": "cosine", "ratio": 0.5, "anneal_steps": 500000}
        )
        self.assertEqual(ecmix.ecmix_ratio_at(0, config), 0.5)
        self.assertAlmostEqual(ecmix.ecmix_ratio_at(250000, config), 0.25)
        self.assertAlmostEqual(
            ecmix.ecmix_ratio_at(300000, config), 0.25 * (1 + math.cos(0.6 * math.pi))
        )
        self.assertEqual(ecmix.ecmix_ratio_at(500000, config), 0.0)
        self.assertEqual(ecmix.ecmix_ratio_at(501000, config), 0.0)

    def test_cosine_ec_steps_are_monotone_and_stop_at_the_anneal_end(self):
        config = ecmix.validate_ecmix_config(
            {"mode": "step", "schedule": "cosine", "ratio": 0.5, "anneal_steps": 500000}
        )
        previous = ecmix.ec_steps_before(0, config)
        for step in range(1, 501001):
            current = ecmix.ec_steps_before(step, config)
            if current < previous or current - previous > 1:
                self.fail(f"EC step count jumps from {previous} to {current} at step {step}")
            previous = current
        self.assertEqual(ecmix.ec_steps_before(500000, config), 125000)
        self.assertEqual(previous, 125000)
        early = sum(ecmix.is_ec_step(step, config) for step in range(1000))
        self.assertIn(early, range(495, 501))

    def test_invalid_configs_are_rejected(self):
        for config in (
            {"mode": "route"},
            {"schedule": "linear"},
            {"ratio": 1.5},
            {"ratio": -0.1},
            {"schedule": "cosine", "ratio": 0.5},
            {"schedule": "constant", "ratio": 0.5, "anneal_steps": 100},
            {"ratio": 0.5, "warmup": 10},
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    ecmix.validate_ecmix_config(config)


class ConferenceEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference = build(conference, moe_config())
        cls.reference_state = {name: value.clone() for name, value in reference.state_dict().items()}
        perturb(reference)
        cls.reference_train = train_step(reference)
        cls.reference_eval = eval_output(reference)

    def assert_train_matches(self, output, grads):
        reference_output, reference_grads = self.reference_train
        self.assertTrue(torch.equal(output, reference_output))
        self.assertEqual(set(grads), set(reference_grads))
        for name, grad in reference_grads.items():
            self.assertTrue(torch.equal(grads[name], grad), name)

    def test_zero_ratio_is_bit_identical_to_the_conference_model(self):
        for ecmix_config in (None, {"mode": "loss", "ratio": 0.0}, {"mode": "step", "ratio": 0.0}):
            with self.subTest(ecmix_config=ecmix_config):
                model = build(ecmix, moe_config(ecmix_config))
                state = model.state_dict()
                self.assertEqual(list(state), list(self.reference_state))
                for name, value in self.reference_state.items():
                    self.assertTrue(torch.equal(state[name], value), name)
                perturb(model)
                self.assert_train_matches(*train_step(model, training_step=7))
                self.assertTrue(torch.equal(eval_output(model), self.reference_eval))

    def test_step_mode_switches_whole_steps(self):
        model = perturb(build(ecmix, moe_config({"mode": "step", "ratio": 0.5})))
        self.assert_train_matches(*train_step(model, training_step=0))
        self.assertFalse(model.ecmix_last_state["ec_step"])

        output, _ = train_step(model, training_step=1)
        self.assertTrue(model.ecmix_last_state["ec_step"])
        self.assertFalse(torch.equal(output, self.reference_train[0]))
        for block in moe_blocks(model):
            self.assertIsNotNone(block.last_ec_contrastive_loss)
            self.assertIsNone(block.last_tc_contrastive_loss)
            self.assertGreater(int(block.last_tc_load_hist.sum()), 0)
            self.assertLessEqual(int(block.last_tc_load_hist.sum()), 6 * 16)

    def test_loss_mode_changes_gradients_but_not_the_forward(self):
        model = perturb(build(ecmix, moe_config({"mode": "loss", "ratio": 0.25})))
        output, grads = train_step(model, training_step=3)
        self.assertTrue(torch.equal(output, self.reference_train[0]))
        changed = [
            name for name, grad in self.reference_train[1].items() if not torch.equal(grads[name], grad)
        ]
        self.assertIn("blocks.1.mlp.cluster_centers", changed)
        block = moe_blocks(model)[0]
        self.assertIsNotNone(block.last_tc_contrastive_loss)
        self.assertIsNotNone(block.last_ec_contrastive_loss)

    def test_evaluation_is_token_choice_even_on_an_ec_step(self):
        model = perturb(build(ecmix, moe_config({"mode": "step", "ratio": 0.5})))
        self.assertTrue(torch.equal(eval_output(model, training_step=1), self.reference_eval))

    def test_mixing_requires_the_training_step(self):
        model = build(ecmix, moe_config({"mode": "loss", "ratio": 0.25}))
        model.train()
        x, t, y = dit_inputs()
        with self.assertRaisesRegex(ValueError, "training_step"):
            model(x, t, y)
        eval_output(model)


class BlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.block = ecmix.SparseMoeBlock(**BLOCK_KWARGS)
        self.ec_block = ec_batch_choice.SparseMoeBlock(**BLOCK_KWARGS)
        self.ec_block.load_state_dict(self.block.state_dict())

    def test_loss_mode_blends_tc_and_ec_contrastive_losses(self):
        generator = torch.Generator().manual_seed(11)
        tokens = torch.randn(40, 16, generator=generator)
        cos_sim = F.normalize(tokens, dim=1) @ F.normalize(self.block.cluster_centers, dim=1).T
        assignments = cos_sim.topk(1, dim=1).indices
        tc_loss = conference.SparseMoeBlock.compute_routing_contrastive_loss(
            self.block, tokens, assignments, use_top_k=True
        )
        _, _, expert_inputs = self.ec_block.compute_router(tokens.reshape(1, 40, 16))
        ec_loss = self.ec_block.compute_routing_contrastive_loss(expert_inputs.mean(dim=1))
        for weight, expected in (
            (0.0, tc_loss),
            (1.0, ec_loss),
            (0.25, tc_loss * 0.75 + ec_loss * 0.25),
        ):
            with self.subTest(weight=weight):
                self.block.ecmix_ec_loss_weight = weight
                loss = self.block.compute_routing_contrastive_loss(tokens, assignments, use_top_k=True)
                torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        self.assertEqual(int(self.block.last_tc_load_hist.sum()), 40)

    def test_ec_step_matches_the_ec_batch_choice_block(self):
        generator = torch.Generator().manual_seed(21)
        hidden = torch.randn(5, 6, 16, generator=generator)
        labels = torch.tensor([1, 1000, 2, 3, 1000])
        self.block.train()
        self.ec_block.train()
        self.block.ecmix_ec_step = True
        hidden_a = hidden.clone().requires_grad_(True)
        hidden_b = hidden.clone().requires_grad_(True)
        output_a, loss_a = self.block(hidden_a, labels)
        output_b, loss_b = self.ec_block(hidden_b, labels)
        torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)
        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
        (output_a.square().mean() + loss_a).backward()
        (output_b.square().mean() + loss_b).backward()
        torch.testing.assert_close(hidden_a.grad, hidden_b.grad, rtol=0, atol=0)
        for (name, parameter_a), (_, parameter_b) in zip(
            self.block.named_parameters(), self.ec_block.named_parameters()
        ):
            torch.testing.assert_close(parameter_a.grad, parameter_b.grad, rtol=0, atol=0, msg=name)
        self.assertEqual(int(self.block.last_tc_load_hist.sum()), 3 * 6)


class TrainStatsTests(unittest.TestCase):
    def test_train_collects_fixed_ecmix_statistics(self):
        import train

        model = build(ecmix, moe_config({"mode": "step", "ratio": 0.5}))
        for step in (0, 1):
            with self.subTest(step=step):
                train_step(model, training_step=step)
                stats = train._collect_ecmix_stats(model)
                self.assertEqual(set(stats), set(train.ECMIX_STAT_NAMES))
                for name, value in stats.items():
                    self.assertTrue(bool(torch.isfinite(value)), name)
                self.assertEqual(float(stats["ecmix_ec_step"]), float(step == 1))
                self.assertGreater(float(stats["ecmix_tc_active_experts"]), 0.0)
        self.assertEqual(train._collect_ecmix_stats(build(conference, moe_config())), {})


if __name__ == "__main__":
    unittest.main()
