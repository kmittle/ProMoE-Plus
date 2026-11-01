"""CPU checks of the six teacher/corrector mechanisms, including DDP edge cases."""
import contextlib
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from models import models_ProMoE_TC_clean_route_dual_teacher as dual
from models import models_ProMoE_TC_clean_route_projector as projector
from models import models_ProMoE_TC_clean_route_token_corrector as local
from test import test_models_ProMoE_TC_corr_router as h


KINDS = {1: "clean_corr", 2: "clean_route", 3: "clean_route", 4: "corr", 5: "clean_corr", 6: "clean_corr"}


def build(arm, **route_overrides):
    torch.manual_seed(0)
    kwargs = dict(h.MODEL_KWARGS, MoE_config=h.moe_config())
    route = h.route_config(align_lam=5e-4)
    route.update(route_overrides)
    if arm == 1:
        route.update(align_lam=2.5e-4, estimated_align_lam=2.5e-4)
        return dual.DiT(**kwargs, clean_route_config=route, reference_config={"skip_intervals": h.INTERVALS})
    if arm == 2:
        return projector.DiT(**kwargs, clean_route_config=route, alignment_projector_config={"hidden_dim": 16})
    if arm == 3:
        return local.DiT(**kwargs, clean_route_config=route, token_corrector_config={"hidden_dim": 16})
    if arm == 4:
        return h.corr_router.DiT(**kwargs, corr_router_config=h.corr_config(reference_mode="current"))
    route.update({"teacher_input": "estimated"} if arm == 5 else {"align_detach_base": True})
    return h.clean_corr.DiT(**kwargs, clean_route_config=route, corr_router_config=h.corr_config())


@contextlib.contextmanager
def record_runs(model):
    original, calls = model._run, []

    def run(x, timestep, y, labels, *args, **kwargs):
        result = original(x, timestep, y, labels, *args, **kwargs)
        reference = args[0] if args else kwargs.get("reference")
        calls.append(dict(x=x.detach().clone(), t=timestep.detach().clone(), labels=labels.clone(),
                          reference=reference, final=kwargs.get("final_layer", True),
                          grad=torch.is_grad_enabled(), result=None if result is None else result.detach().clone()))
        return result

    with mock.patch.object(model, "_run", side_effect=run):
        yield calls


class MechanismTests(unittest.TestCase):
    def test_projector_initialization_never_reseeds_cuda(self):
        # No GPU required: manual_seed calls CUDA's seeding API even before
        # CUDA initialization. Only the test builder's initial seed is allowed.
        with mock.patch("torch.cuda.manual_seed_all") as seed_cuda:
            build(2)
        self.assertEqual(seed_cuda.call_args_list, [mock.call(0)])

    def test_six_models_have_finite_forward_backward_in_fp32_and_bf16(self):
        for arm in KINDS:
            for autocast in (False, True):
                with self.subTest(arm=arm, autocast=autocast):
                    model = h.perturb_correctors(h.perturb(build(arm)))
                    output, grads = h.train_step(model, KINDS[arm], autocast=autocast)
                    self.assertTrue(bool(torch.isfinite(output).all()))
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad:
                            self.assertIn(name, grads)
                            self.assertTrue(bool(torch.isfinite(grads[name]).all()), name)
                    for block in model.moe_blocks():
                        self.assertIsNone(getattr(block, "_teacher", None))
                        self.assertIsNone(getattr(block, "_corr_context", None))

    def test_new_heads_preserve_backbone_initialization_and_global_rng(self):
        parent = h.build_clean_route(align_lam=5e-4)
        expected_rng = torch.get_rng_state()
        for arm in (1, 2, 3):
            with self.subTest(arm=arm):
                model = build(arm)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                for name, parameter in parent.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name], parameter, rtol=0, atol=0, msg=name)

    def test_dual_teacher_inputs_and_two_kl_terms(self):
        model = h.perturb(build(1))
        block = model.moe_blocks()[0]
        snapshots = []
        original = block._alignment_kl

        def capture(log_p, teacher):
            snapshots.append((log_p.detach(), teacher["log_q"].detach(), teacher["estimated_log_q"].detach()))
            return original(log_p, teacher)

        with record_runs(model) as calls, mock.patch.object(block, "_alignment_kl", side_effect=capture):
            h.train_step(model, KINDS[1])
        self.assertEqual(len(calls), 4)  # two estimates, a batched dual teacher, student
        labels = h.dropped_labels()
        routed = labels != 1000
        count = int(routed.sum())
        self.assertEqual([len(c["x"]) for c in calls], [count, count, 2 * count, h.BATCH])
        self.assertEqual([c["grad"] for c in calls], [False, False, False, True])
        self.assertEqual(int(torch.count_nonzero(calls[2]["t"])), 0)
        x, t, z, noise, sigma = h.batch()
        torch.testing.assert_close(calls[2]["x"][:count], z[routed])
        previous = calls[1]
        expected_estimate = previous["x"] - (previous["t"] / 1000).view(-1, 1, 1, 1, 1) * previous["result"][:, :4].unsqueeze(2)
        torch.testing.assert_close(calls[2]["x"][count:], expected_estimate)
        log_p, real, estimated = snapshots[0]
        expected = 2.5e-4 * ((real.exp() * (real - log_p)).sum(1).mean()
                              + (estimated.exp() * (estimated - log_p)).sum(1).mean())
        torch.testing.assert_close(block.last_align_loss, expected)
        self.assertFalse(any("corrector" in name for name, _ in model.named_parameters()))
        import train
        stats = train._collect_clean_route_stats(model)
        self.assertIn("clean_route_real_kl", stats)
        self.assertIn("clean_route_estimated_kl", stats)

    def test_dual_teacher_equal_targets_recover_single_kl_gradient(self):
        block = build(1).moe_blocks()[0]
        raw = torch.randn(7, 4, requires_grad=True)
        target = F.log_softmax(torch.randn(7, 4), dim=1)
        log_p = F.log_softmax(raw / 0.07, dim=1)
        dual_loss = block.align_lam * block._alignment_kl(log_p, {"log_q": target, "estimated_log_q": target}).mean()
        single_loss = 5e-4 * (target.exp() * (target - log_p)).sum(1).mean()
        torch.testing.assert_close(torch.autograd.grad(dual_loss, raw, retain_graph=True)[0],
                                   torch.autograd.grad(single_loss, raw)[0])

    def test_projector_only_affects_alignment_and_normalizes_with_existing_softmax(self):
        model = h.perturb(build(2))
        parent = h.build_clean_route(align_lam=5e-4)
        parent.load_state_dict({k: v for k, v in model.state_dict().items() if "alignment_projector" not in k})
        x, t, *_ = h.batch()
        with torch.no_grad():
            torch.testing.assert_close(model.eval()(x, t, h.LABELS), parent.eval()(x, t, h.LABELS), rtol=0, atol=0)
        block = model.moe_blocks()[0]
        block.keep_cosines = True
        with h.captured_outputs(block.alignment_projector) as projected:
            h.train_step(model, KINDS[2])
        self.assertEqual(len(projected), 1)
        self.assertTrue(bool(((projected[0] > 0) & (projected[0] < 1)).all()))
        torch.testing.assert_close(block.last_student_cosines, projected[0])
        probabilities = F.softmax(projected[0].float() / block.align_temperature, dim=1)
        torch.testing.assert_close(probabilities.sum(1), torch.ones(len(probabilities)))
        self.assertGreater(float(block.alignment_projector[0].weight.grad.abs().sum()), 0)

    def test_local_corrector_inputs_are_detached_and_teacher_is_uncorrected(self):
        model = h.perturb_correctors(h.perturb(build(3)))
        block = model.moe_blocks()[0]
        flags, phases = [], []

        def prehook(module, args):
            flags.append(tuple(value.requires_grad for value in args))
            phases.append(block._phase)

        handle = block.corrector.register_forward_pre_hook(prehook)
        try:
            with record_runs(model) as calls:
                h.train_step(model, KINDS[3])
            self.assertEqual(len(calls), 2)
            self.assertEqual(flags, [(False, False)])
            self.assertEqual(phases, ["student"])
            model.eval()
            x, t, *_ = h.batch()
            with torch.no_grad():
                model(x, t, h.LABELS)
            self.assertEqual(phases[-1], None)  # correction also runs in eval
        finally:
            handle.remove()

    def test_kl_gradient_for_local_corrector_and_symmetric_base_detach(self):
        for arm in (3, 6):
            with self.subTest(arm=arm):
                block = h.perturb_correctors(build(arm)).moe_blocks()[0]
                raw = torch.randn(8, 4, requires_grad=True)
                token = torch.randn(8, 48, requires_grad=True)
                if arm == 3:
                    delta = block.corrector(token.detach(), raw.detach())
                else:
                    delta = block.corrector(token.detach(), raw.detach(), torch.randn(8, 16))
                delta.retain_grad()
                block.set_phase("student", sample_sigma=torch.tensor([0.3, 0.7]))
                block._corr_delta = delta
                log_q = F.log_softmax(torch.randn(8, 4), dim=1)
                block._teacher = {"log_q": log_q, "indices": log_q.argmax(1, keepdim=True)}
                selected = (raw + delta).argmax(1, keepdim=True)
                loss = block._student_losses(torch.tensor(0.0), selected, torch.ones(8, 1), raw,
                                             torch.ones(8, dtype=torch.bool), 4)
                loss.backward()
                self.assertIsNone(token.grad)
                self.assertGreater(float(delta.grad.abs().sum()), 0)
                self.assertGreater(float(block.corrector.out.weight.grad.abs().sum()), 0)
                if arm == 3:
                    self.assertGreater(float(raw.grad.abs().sum()), 0)
                else:
                    self.assertIsNone(raw.grad)

    def test_symmetric_detach_preserves_fm_gradient_to_raw_router(self):
        model = h.perturb_correctors(h.perturb(build(6))).eval()
        x, t, z, noise, sigma = h.batch()
        output = model(x, t, h.LABELS, corr_reference_x0=z, corr_reference_sigma=sigma)
        output.square().mean().backward()
        block = model.moe_blocks()[-1]
        self.assertGreater(float(block.cluster_centers.grad.abs().sum()), 0)
        self.assertGreater(float(block.corrector.out.weight.grad.abs().sum()), 0)

    def test_estimated_teacher_sees_all_valid_estimates_while_student_keeps_dropout(self):
        model = h.perturb_correctors(h.perturb(build(5)))
        with record_runs(model) as calls:
            h.train_step(model, KINDS[5])
        self.assertEqual(len(calls), 4)
        labels = h.dropped_labels()
        routed = labels != 1000
        x, t, z, noise, sigma = h.batch()
        use, reference_sigma = h.expected_draws(0.5, sigma, labels)
        self.assertTrue(bool((routed & ~use).any()))
        self.assertEqual([len(c["x"]) for c in calls[:3]], [int(routed.sum())] * 3)
        previous = calls[1]
        estimate = previous["x"] - reference_sigma[routed].view(-1, 1, 1, 1, 1) * previous["result"][:, :4].unsqueeze(2)
        teacher = calls[2]
        torch.testing.assert_close(teacher["x"], estimate)
        self.assertFalse(teacher["grad"])
        self.assertEqual(int(torch.count_nonzero(teacher["t"])), 0)
        self.assertIsNone(teacher["reference"])
        student_ref, student_sigma, student_mask, _ = calls[3]["reference"]
        torch.testing.assert_close(student_mask, use)
        torch.testing.assert_close(student_ref[use], estimate[use[routed]])
        self.assertEqual(int(torch.count_nonzero(student_ref[~use])), 0)
        torch.testing.assert_close(student_sigma[~use], torch.ones_like(student_sigma[~use]))
        self.assertTrue(bool((teacher["x"][~use[routed]].flatten(1).abs().sum(1) > 0).all()))

    def test_estimated_teacher_does_not_read_ground_truth_latents(self):
        model = h.perturb_correctors(h.perturb(build(5)))
        first, gradients = h.train_step(model, KINDS[5])
        _, _, z, noise, _ = h.batch()
        second, changed_gradients = h.train_step(model, KINDS[5], z=z + 9, noise=noise - 7)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for name in gradients:
            torch.testing.assert_close(gradients[name], changed_gradients[name], rtol=0, atol=0, msg=name)

    def test_current_reference_uses_one_auxiliary_forward_at_current_time(self):
        model = h.perturb_correctors(h.perturb(build(4)))
        with record_runs(model) as calls:
            h.train_step(model, KINDS[4])
        self.assertEqual(len(calls), 2)
        x, t, z, noise, sigma = h.batch()
        use, _ = h.expected_draws(0.5, sigma, h.dropped_labels())
        torch.testing.assert_close(calls[0]["x"], x[use])
        torch.testing.assert_close(calls[0]["t"], t[use])
        expected = x[use] - sigma.reshape(-1)[use].view(-1, 1, 1, 1, 1) * calls[0]["result"][:, :4].unsqueeze(2)
        ref, ref_sigma, mask, _ = calls[1]["reference"]
        torch.testing.assert_close(ref[use], expected)
        torch.testing.assert_close(ref_sigma[use], sigma.reshape(-1)[use])
        self.assertEqual(float(model.last_corr_reference[2]), 0)

    def test_conflicting_gradient_and_reference_flags_fail(self):
        with self.assertRaisesRegex(ValueError, "both"):
            h.build_clean_corr(align_detach_base=True, align_detach_correction=True)
        with self.assertRaisesRegex(ValueError, "align_detach_base"):
            h.build_clean_corr(align_detach_base="true")
        with self.assertRaisesRegex(ValueError, "teacher_input"):
            h.build_clean_corr(teacher_input="zero")
        with self.assertRaisesRegex(ValueError, "current"):
            h.build_corr(reference_mode="current", previous_state_mode="same_noise")
        with self.assertRaisesRegex(ValueError, "reference_mode"):
            h.build_corr(reference_mode="invalid")


class CurrentReferenceSamplingTests(unittest.TestCase):
    def test_current_reference_is_cfg_fused_and_recomputed_at_every_step(self):
        for guide in (1.0, 1.5):
            with self.subTest(guide=guide):
                model = h.perturb_correctors(h.perturb(build(4))).eval()
                recorder = h._RecordingModel(model)
                recorder.module = model  # same unwrapping contract as DDP
                helper = h.SamplerLoopTests()
                helper.run_loop(recorder, True, guide_scale=guide, reference_velocity="cfg")
                width = 2 if guide == 1 else 3
                self.assertEqual(len(recorder.calls), width * helper.STEPS)
                for step in range(helper.STEPS):
                    group = recorder.calls[width * step:width * (step + 1)]
                    preliminary, corrected = group[0], group[-1]
                    self.assertFalse(preliminary["reference"])
                    torch.testing.assert_close(preliminary["x"], corrected["x"], rtol=0, atol=0)
                    torch.testing.assert_close(preliminary["timestep"], corrected["timestep"], rtol=0, atol=0)
                    velocity = preliminary["output"][:, :4].unsqueeze(2)
                    if guide > 1:
                        uncond = group[1]
                        self.assertTrue(bool((uncond["context"] == 1000).all()))
                        self.assertFalse(uncond["reference"])
                        null = uncond["output"][:, :4].unsqueeze(2)
                        velocity = null + guide * (velocity - null)
                    sigma = preliminary["timestep"] / 1000
                    expected = preliminary["x"] - sigma.view(-1, 1, 1, 1, 1) * velocity
                    torch.testing.assert_close(corrected["reference"]["corr_reference_x0"], expected)
                    torch.testing.assert_close(corrected["reference"]["corr_reference_sigma"], sigma)


class ConfigIntegrationTests(unittest.TestCase):
    def test_six_yaml_configs_merge_construct_and_reach_sampling_registry(self):
        import yaml
        import train
        import sample
        from config import cfg
        from utils import deep_update
        from test.test_training_schedule_scripts import ARMS

        root = Path(__file__).resolve().parents[1]
        for arm, (_, tag) in enumerate(ARMS[-6:], 1):
            with self.subTest(arm=arm):
                data = yaml.safe_load((root / "configs" / f"004_ProMoE_B_{tag}.yaml").read_text())
                merged = copy.deepcopy(cfg)
                deep_update(merged, data)
                cls, key = train.model_dict[data["model_name"]]
                self.assertEqual(sample.model_dict[data["model_name"]], (cls, key))
                kwargs = dict(merged[key])
                kwargs.update(h.MODEL_KWARGS)
                kwargs["MoE_config"].moe_intermediate_size = 32
                kwargs["MoE_config"].shared_expert_intermediate_size = 32
                model = cls(**kwargs)
                h.train_step(model, KINDS[arm])
                self.assertEqual(data["gpu_ids"], [0, 1, 2, 3])
                self.assertEqual(data["total_train_batch_size"], 256)
                self.assertEqual(data["step_list_for_sample"], [300000, 500000])
                self.assertEqual(data["guide_scale_list"], [1.0, 1.5])
                self.assertEqual(data["model_name"] in train.CORR_REFERENCE_MODELS, arm in (1, 4, 5, 6))
                self.assertEqual(bool(getattr(model, "is_corr_router_model", False)), arm in (4, 5, 6))
                if arm in (4, 5, 6):
                    self.assertEqual(data["corr_reference_velocity"], "cfg")


def _distributed_worker(rank, size, init_file):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=size)
    try:
        import train
        for arm in KINDS:
            model = h.perturb_correctors(h.perturb(build(arm)))
            wrapped = DistributedDataParallel(model)
            optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-3)
            labels = h.LABELS if rank == 0 else torch.full_like(h.LABELS, 1000)
            for step in range(2):
                x, t, z, noise, sigma = h.batch(seed=1234 + step)
                inputs = h.model_inputs(KINDS[arm], labels, z, noise, sigma, generator_seed=rank + step)
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(h.DROPOUT_SEED)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = wrapped(x, t, **inputs)
                output.float().square().mean().backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        assert parameter.grad is not None, (arm, rank, name)
                        assert torch.isfinite(parameter.grad).all(), (arm, rank, name)
                if getattr(model, "is_clean_route_model", False):
                    stats = train._collect_clean_route_stats(wrapped)
                    train._reduce_clean_route_stats(stats)
                    assert all(torch.isfinite(v).all() for v in stats.values())
                if getattr(model, "is_corr_router_model", False):
                    stats = train._collect_corr_router_stats(wrapped)
                    assert all(torch.isfinite(v).all() for v in stats.values())
                optimizer.step()
            del wrapped, optimizer, model
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "needs CPU gloo")
    def test_two_ddp_steps_for_all_six_arms_with_an_all_unconditional_rank(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix=".six_route_ddp_") as directory:
            mp.spawn(_distributed_worker, args=(2, str(Path(directory) / "init")), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
