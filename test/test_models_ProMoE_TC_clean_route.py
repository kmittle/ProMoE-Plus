import contextlib
import io
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel

from models import models_ProMoE_TC as conference
from models import models_ProMoE_TC_clean_route as clean_route


MODEL_KWARGS = dict(input_size=8, patch_size=2, in_channels=4, hidden_size=48, depth=4, num_heads=4)
BATCH = 6
SEQ_LEN = 16  # (8 / 2) ** 2 patches
LABELS = torch.tensor([3, 1000, 7, 7, 1000, 42])
DROPOUT_SEED = 99


def moe_config(routing_contrastive_lam=1):
    return EasyDict(
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
        routing_contrastive_lam=routing_contrastive_lam,
        use_top_k_for_routing_contrastive=True,
        routing_contrastive_temperature=0.07,
    )


def route_config(**overrides):
    config = dict(contrastive_source="student", teacher_sigma_ratio=0.0, align_lam=0.0, align_temperature=0.07)
    config.update(overrides)
    return config


def build_conference(routing_contrastive_lam=1):
    torch.manual_seed(0)
    # The conference constructor prints its MoE layout.
    with contextlib.redirect_stdout(io.StringIO()):
        return conference.DiT(MoE_config=moe_config(routing_contrastive_lam), **MODEL_KWARGS)


def build_clean_route(routing_contrastive_lam=1, **overrides):
    torch.manual_seed(0)
    return clean_route.DiT(
        MoE_config=moe_config(routing_contrastive_lam),
        clean_route_config=route_config(**overrides),
        **MODEL_KWARGS,
    )


def perturb(model):
    """Move every weight off its zero-initialised gates and output layer."""
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
    return model


def batch(zero_sigma=False, seed=1234):
    """Latents laid out as train.py builds them: (B, C, 1, H, W)."""
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(BATCH, 4, 1, 8, 8, generator=generator)
    noise = torch.randn(BATCH, 4, 1, 8, 8, generator=generator)
    sigma = torch.rand(BATCH, generator=generator)
    if zero_sigma:
        sigma = torch.zeros(BATCH)
    sigma = sigma.view(BATCH, 1, 1, 1)
    x = (1.0 - sigma.squeeze()).view(BATCH, 1, 1, 1, 1) * z + sigma.squeeze().view(BATCH, 1, 1, 1, 1) * noise
    t = sigma.reshape(-1) * 1000
    return x, t, z, noise, sigma


def dropped_labels():
    """The labels the training forward sees after class dropout drawn at DROPOUT_SEED."""
    torch.manual_seed(DROPOUT_SEED)
    drop = torch.rand(BATCH) < 0.1
    return torch.where(drop, torch.full_like(LABELS, 1000), LABELS)


def train_step(model, use_clean_route, zero_sigma=False, loss_scale=1.0, autocast=False):
    """One step as train.py runs it: the forward under autocast, the backward outside."""
    model.train()
    model.zero_grad(set_to_none=True)
    x, t, z, noise, sigma = batch(zero_sigma)
    torch.manual_seed(DROPOUT_SEED)  # class-label dropout draws from the global RNG
    region = torch.autocast(device_type="cpu", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    with region:
        if use_clean_route:
            output = model(x, t, context=LABELS, clean_route_z=z, clean_route_noise=noise, clean_route_sigma=sigma)
        else:
            output = model(x, t, context=LABELS)
    (loss_scale * output.float().square().mean()).backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), grads


class EquivalenceTests(unittest.TestCase):
    def test_parameters_and_initialisation_match_the_conference_model(self):
        reference_state = build_conference().state_dict()
        state = build_clean_route().state_dict()
        self.assertEqual(list(state), list(reference_state))
        for name, value in reference_state.items():
            self.assertTrue(torch.equal(state[name], value), name)

    def test_eval_forward_matches_the_conference_model(self):
        reference = perturb(build_conference()).eval()
        model = perturb(build_clean_route(align_lam=1.0, contrastive_source="teacher")).eval()
        x, t, _, _, _ = batch()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x, t, context=LABELS), reference(x, t, context=LABELS)))

    def test_training_step_with_alignment_off_is_bit_identical(self):
        reference_output, reference_grads = train_step(perturb(build_conference()), use_clean_route=False)
        for ratio in (0.0, 0.5):
            with self.subTest(teacher_sigma_ratio=ratio):
                output, grads = train_step(
                    perturb(build_clean_route(teacher_sigma_ratio=ratio)), use_clean_route=True
                )
                self.assertTrue(torch.equal(output, reference_output))
                self.assertEqual(set(grads), set(reference_grads))
                for name, grad in reference_grads.items():
                    self.assertTrue(torch.equal(grads[name], grad), name)

    def test_bf16_autocast_step_is_bit_identical_and_trains_every_weight(self):
        # train.py runs both passes in one bf16 autocast region; a teacher cast
        # left in the autocast cache would cut the student's weights off.
        reference_output, reference_grads = train_step(
            perturb(build_conference()), use_clean_route=False, autocast=True
        )
        for ratio in (0.0, 0.5):
            with self.subTest(teacher_sigma_ratio=ratio):
                model = perturb(build_clean_route(teacher_sigma_ratio=ratio))
                output, grads = train_step(model, use_clean_route=True, autocast=True)
                trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                self.assertEqual(set(grads), trainable)
                self.assertTrue(torch.equal(output, reference_output))
                for name, grad in reference_grads.items():
                    self.assertTrue(torch.equal(grads[name], grad), name)

    def test_the_teacher_pass_draws_no_random_numbers(self):
        train_step(perturb(build_conference()), use_clean_route=False)
        reference_state = torch.get_rng_state()
        train_step(perturb(build_clean_route(align_lam=1.0, teacher_sigma_ratio=0.5)), use_clean_route=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), reference_state))


class LossValueTests(unittest.TestCase):
    def test_alignment_is_kl_of_teacher_to_student_at_the_temperature(self):
        tau = 0.2
        model = perturb(build_clean_route(align_lam=1.0, teacher_sigma_ratio=0.5, align_temperature=tau))
        for block in model.moe_blocks():
            block.keep_cosines = True
        train_step(model, use_clean_route=True)
        for block in model.moe_blocks():
            log_q = F.log_softmax(block.last_teacher_cosines / tau, dim=1)
            log_p = F.log_softmax(block.last_student_cosines / tau, dim=1)
            expected = (log_q.exp() * (log_q - log_p)).sum(dim=1).mean()
            torch.testing.assert_close(block.last_kl, expected)
            reversed_kl = (log_p.exp() * (log_p - log_q)).sum(dim=1).mean()
            self.assertGreater(float((expected - reversed_kl).abs()), 1e-6)

    def test_teacher_contrastive_uses_the_teacher_tokens_and_assignments(self):
        model = perturb(build_clean_route(contrastive_source="teacher", teacher_sigma_ratio=0.5))
        recorded = {}
        for index, block in enumerate(model.moe_blocks()):
            original = block._record_teacher

            def record(tokens, indices, cosines, seq_len, _original=original, _index=index):
                recorded[_index] = (tokens.detach().clone(), indices.detach().clone())
                return _original(tokens, indices, cosines, seq_len)

            block._record_teacher = record
        train_step(model, use_clean_route=True)
        for index, block in enumerate(model.moe_blocks()):
            tokens, indices = recorded[index]
            expected = conference.SparseMoeBlock.compute_routing_contrastive_loss(
                block, tokens, indices, use_top_k=True
            ) * block.routing_contrastive_lam
            torch.testing.assert_close(block.last_contrastive_loss, expected.detach())


class TeacherPassTests(unittest.TestCase):
    def capture_embedder_inputs(self, model):
        captured = {"x": [], "t": []}
        model.x_embedder.register_forward_pre_hook(lambda module, args: captured["x"].append(args[0].detach().clone()))
        model.t_embedder.register_forward_pre_hook(lambda module, args: captured["t"].append(args[0].detach().clone()))
        return captured

    def test_teacher_input_uses_the_scaled_sigma_and_the_student_noise(self):
        labels = dropped_labels()
        routed = labels != 1000
        self.assertTrue(bool(routed.any()) and not bool(routed.all()))
        x, t, z, noise, sigma = batch()
        for ratio in (0.0, 0.5):
            with self.subTest(teacher_sigma_ratio=ratio):
                model = perturb(build_clean_route(teacher_sigma_ratio=ratio))
                captured = self.capture_embedder_inputs(model)
                train_step(model, use_clean_route=True)
                # The teacher pass runs first and only on the prototype-routed images.
                teacher_x, student_x = captured["x"]
                teacher_t, student_t = captured["t"]
                teacher_sigma = ratio * sigma.reshape(-1)[routed]
                expected_x = (
                    (1.0 - teacher_sigma).view(-1, 1, 1, 1) * z[routed].squeeze(2)
                    + teacher_sigma.view(-1, 1, 1, 1) * noise[routed].squeeze(2)
                )
                torch.testing.assert_close(teacher_x, expected_x)
                torch.testing.assert_close(teacher_t, ratio * t[routed])
                self.assertTrue(torch.equal(student_x, x.squeeze(2)))
                self.assertTrue(torch.equal(student_t, t))
                if ratio == 0.0:
                    self.assertTrue(torch.equal(teacher_x, z[routed].squeeze(2)))
                    self.assertTrue(torch.equal(teacher_t, torch.zeros_like(teacher_t)))

    def test_a_clean_teacher_of_a_clean_student_routes_identically(self):
        model = perturb(build_clean_route(align_lam=1.0))
        train_step(model, use_clean_route=True, zero_sigma=True)
        for block in model.moe_blocks():
            self.assertLess(float(block.last_kl), 1e-6)
            self.assertEqual(float(block.last_agreement), 1.0)

    def test_the_teacher_target_carries_no_gradient(self):
        model = perturb(build_clean_route(align_lam=1.0, teacher_sigma_ratio=0.5))
        for block in model.moe_blocks():
            block.keep_cosines = True
        train_step(model, use_clean_route=True)
        for block in model.moe_blocks():
            self.assertFalse(block.last_teacher_cosines.requires_grad)
            self.assertEqual(block.last_teacher_cosines.shape, block.last_student_cosines.shape)

    def test_the_phases_are_cleared_after_a_step(self):
        model = perturb(build_clean_route(align_lam=1.0))
        train_step(model, use_clean_route=True)
        for block in model.moe_blocks():
            self.assertIsNone(block._phase)
            self.assertIsNone(block._teacher)
            self.assertIsNone(block._sample_sigma)


class GradientRoutingTests(unittest.TestCase):
    def aux_only_grads(self, model):
        """Gradients of the auxiliary losses alone (the main loss is scaled to 0)."""
        _, grads = train_step(model, use_clean_route=True, loss_scale=0.0)
        return grads

    def test_teacher_contrastive_moves_only_the_prototypes(self):
        grads = self.aux_only_grads(perturb(build_clean_route(contrastive_source="teacher")))
        moved = {name for name, grad in grads.items() if grad.abs().max() > 0}
        self.assertTrue(moved)
        self.assertTrue(all(name.endswith("cluster_centers") for name in moved), sorted(moved))

    def test_student_contrastive_moves_the_backbone_too(self):
        grads = self.aux_only_grads(perturb(build_clean_route(contrastive_source="student")))
        moved = {name for name, grad in grads.items() if grad.abs().max() > 0}
        self.assertTrue(any(name.endswith("cluster_centers") for name in moved))
        self.assertTrue(any(not name.endswith("cluster_centers") for name in moved))

    def test_alignment_moves_the_prototypes_and_the_backbone(self):
        model = perturb(build_clean_route(routing_contrastive_lam=0, align_lam=1.0, teacher_sigma_ratio=0.5))
        grads = self.aux_only_grads(model)
        moved = {name for name, grad in grads.items() if grad.abs().max() > 0}
        self.assertTrue(any(name.endswith("cluster_centers") for name in moved))
        self.assertTrue(any(name.startswith("x_embedder") for name in moved))

    def test_no_auxiliary_gradient_without_alignment_and_contrastive(self):
        model = perturb(build_clean_route(routing_contrastive_lam=0, align_lam=0.0))
        grads = self.aux_only_grads(model)
        self.assertTrue(all(grad.abs().max() == 0 for grad in grads.values()))


class DiagnosticsTests(unittest.TestCase):
    def test_statistics_cover_every_prototype_routed_token(self):
        labels = dropped_labels()
        num_routed_tokens = int((labels != 1000).sum()) * SEQ_LEN
        model = perturb(build_clean_route(align_lam=1.0, contrastive_source="teacher", teacher_sigma_ratio=0.5))
        train_step(model, use_clean_route=True)
        for block in model.moe_blocks():
            self.assertEqual(float(block.last_student_load.sum()), num_routed_tokens)
            self.assertEqual(float(block.last_teacher_load.sum()), num_routed_tokens)
            self.assertEqual(tuple(block.last_sigma_bins.shape), (3, clean_route.NUM_SIGMA_BINS))
            self.assertEqual(float(block.last_sigma_bins[2].sum()), num_routed_tokens)
            self.assertTrue(0.0 <= float(block.last_agreement) <= 1.0)
            self.assertGreaterEqual(float(block.last_kl), 0.0)
            torch.testing.assert_close(block.last_align_loss, block.last_kl)  # align_lam = 1
            self.assertGreater(float(block.last_contrastive_loss), 0.0)

    def test_every_image_dropped_skips_the_teacher_pass(self):
        model = perturb(build_clean_route(align_lam=1.0))
        model.train()
        x, t, z, noise, sigma = batch()
        all_dropped = torch.full_like(LABELS, 1000)
        output = model(x, t, context=all_dropped, clean_route_z=z, clean_route_noise=noise, clean_route_sigma=sigma)
        output.float().square().mean().backward()
        for block in model.moe_blocks():
            self.assertEqual(float(block.last_kl), 0.0)
            self.assertEqual(float(block.last_sigma_bins[2].sum()), 0.0)


def _ddp_worker(rank, world_size, init_file, output_dir):
    """Two DDP steps as train.py runs them: bf16 forward, the stat collectives, backward, update."""
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        import train  # the stat collector and reducer under test

        model = perturb(build_clean_route(align_lam=1.0, contrastive_source="teacher", teacher_sigma_ratio=0.5))
        ddp_model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-2)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        steps = []
        for step in range(2):
            ddp_model.train()
            x, t, z, noise, sigma = batch(seed=100 * rank + step)
            torch.manual_seed(1000 * rank + step)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = ddp_model(
                    x, t, context=LABELS, clean_route_z=z, clean_route_noise=noise, clean_route_sigma=sigma
                )
            stats = train._collect_clean_route_stats(ddp_model)
            output.float().square().mean().backward()
            with_grad = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
            train._reduce_clean_route_stats(stats)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps.append({
                "missing_grads": sorted(trainable - with_grad),
                "stats": {name: float(value) for name, value in stats.items()},
            })
        torch.save(steps, os.path.join(output_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "needs torch.distributed with gloo")
class DistributedTests(unittest.TestCase):
    def test_two_ddp_steps_train_every_weight_and_agree_on_the_stats(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as output_dir:
            mp.spawn(
                _ddp_worker,
                args=(world_size, os.path.join(output_dir, "init"), output_dir),
                nprocs=world_size,
                join=True,
            )
            results = [torch.load(os.path.join(output_dir, f"rank{rank}.pt")) for rank in range(world_size)]

        for rank, steps in enumerate(results):
            self.assertEqual(len(steps), 2)
            for step, record in enumerate(steps):
                with self.subTest(rank=rank, step=step):
                    self.assertEqual(record["missing_grads"], [])
        # After the reduction every rank logs the same values.
        for step in range(2):
            reference = results[0][step]["stats"]
            self.assertEqual(set(reference), set(results[1][step]["stats"]))
            for name, value in reference.items():
                self.assertAlmostEqual(results[1][step]["stats"][name], value, places=5, msg=name)


class ConfigTests(unittest.TestCase):
    def test_training_needs_the_clean_route_inputs(self):
        model = build_clean_route().train()
        x, t, _, _, _ = batch()
        with self.assertRaisesRegex(ValueError, "clean_route_z"):
            model(x, t, context=LABELS)

    def test_invalid_settings_are_refused(self):
        cases = {
            "missing key": {key: value for key, value in route_config().items() if key != "align_lam"},
            "unknown key": dict(route_config(), teacher_sigma=0.5),
            "bad source": route_config(contrastive_source="both"),
            "ratio of one": route_config(teacher_sigma_ratio=1.0),
            "negative ratio": route_config(teacher_sigma_ratio=-0.1),
            "negative lam": route_config(align_lam=-1.0),
            "zero temperature": route_config(align_temperature=0.0),
        }
        for name, config in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                clean_route.parse_clean_route_config(config)
        with self.assertRaises(ValueError):
            clean_route.parse_clean_route_config(None)

    def test_load_balance_loss_is_refused(self):
        config = moe_config()
        config.load_balance_loss_coef = 0.1
        with self.assertRaisesRegex(ValueError, "load_balance_loss_coef"):
            clean_route.DiT(MoE_config=config, clean_route_config=route_config(), **MODEL_KWARGS)


if __name__ == "__main__":
    unittest.main()
