"""Behavior and gradient checks for the complete-method mechanism ablations."""
import contextlib
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from test import test_models_ProMoE_TC_corr_router as h


class InputAblationTests(unittest.TestCase):
    def test_explicit_defaults_preserve_initialization_forward_and_gradients(self):
        for autocast in (False, True):
            with self.subTest(autocast=autocast):
                original = h.perturb_correctors(h.perturb(h.build_clean_corr(align_lam=5e-4)))
                explicit = h.perturb_correctors(h.perturb(h.build_clean_corr(
                    corr_overrides=dict(use_reference_embedding=True, use_score_input=True),
                    align_lam=5e-4, align_detach_correction=False,
                )))
                explicit.load_state_dict(original.state_dict(), strict=True)
                out, grads = h.train_step(original, "clean_corr", autocast=autocast)
                expected, expected_grads = h.train_step(explicit, "clean_corr", autocast=autocast)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                self.assertEqual(set(grads), set(expected_grads))
                for name in grads:
                    torch.testing.assert_close(grads[name], expected_grads[name], rtol=0, atol=0, msg=name)

    def test_masks_reach_every_training_pass_and_keep_zero_gradients_for_ddp(self):
        for flag, projection in (("use_reference_embedding", "reference_proj"), ("use_score_input", "score_proj")):
            for autocast in (False, True):
                with self.subTest(flag=flag, autocast=autocast):
                    model = h.perturb_correctors(h.perturb(h.build_clean_corr(
                        corr_overrides={flag: False}, align_lam=5e-4,
                    )))
                    with contextlib.ExitStack() as stack:
                        records = [stack.enter_context(h.captured_inputs(getattr(b.corrector, projection)))
                                   for b in model.moe_blocks()]
                        h.train_step(model, "clean_corr", autocast=autocast)
                    for block, inputs in zip(model.moe_blocks(), records):
                        self.assertGreater(len(inputs), 1)  # reference passes plus the student
                        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in inputs))
                        grad = getattr(block.corrector, projection).weight.grad
                        self.assertIsNotNone(grad)
                        self.assertEqual(int(torch.count_nonzero(grad)), 0)
                    self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))

    def test_reference_mask_removes_reference_content_dependence_in_sampling(self):
        x, t, z, noise, sigma = h.batch()
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model = h.perturb_correctors(h.perturb(h.build_clean_corr(
                    corr_overrides={"use_reference_embedding": enabled}, align_lam=5e-4,
                ))).eval()
                with torch.no_grad():
                    first = model(x, t, h.LABELS, corr_reference_x0=z, corr_reference_sigma=sigma)
                    second = model(x, t, h.LABELS, corr_reference_x0=noise, corr_reference_sigma=sigma)
                if enabled:
                    self.assertFalse(torch.equal(first, second))
                else:
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
                torch.testing.assert_close(first[h.LABELS == 1000], second[h.LABELS == 1000], rtol=0, atol=0)

    def test_score_mask_keeps_raw_scores_in_actual_routing(self):
        model = h.perturb_correctors(h.perturb(h.build_clean_corr(
            corr_overrides={"use_score_input": False}, align_lam=5e-4,
        ))).eval()
        block = model.moe_blocks()[0]
        block.set_corr_context({
            "reference_embed": torch.randn(h.BATCH, h.SEQ_LEN, 48),
            "condition": torch.randn(h.BATCH, 48),
            "reference_sigma_embed": torch.randn(h.BATCH, 48),
        })
        try:
            with h.captured_outputs(block.corrector) as corrections:
                for _ in range(2):
                    hidden = torch.randn(h.BATCH, h.SEQ_LEN, 48)
                    weights, indices, _ = block.compute_router(hidden, h.LABELS)
                    raw = h.routed_cosines(block, hidden, h.LABELS)
                    expected = (raw + corrections[-1]).topk(block.top_k, dim=1)
                    routed = h.LABELS != 1000
                    torch.testing.assert_close(weights[routed].reshape_as(expected.values), expected.values)
                    torch.testing.assert_close(indices[routed].reshape_as(expected.indices), expected.indices)
            torch.testing.assert_close(corrections[0], corrections[1], rtol=0, atol=0)
        finally:
            block.set_corr_context(None)

    def test_new_flags_require_booleans_and_kl_flag_is_combo_only(self):
        for value in (0, 1, None, "false"):
            for flag in ("use_reference_embedding", "use_score_input"):
                with self.subTest(flag=flag, value=value), self.assertRaisesRegex(ValueError, flag):
                    h.build_clean_corr(corr_overrides={flag: value})
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "align_detach_correction"):
                h.build_clean_corr(align_detach_correction=value)
        with self.assertRaisesRegex(ValueError, "unknown"):
            h.build_clean_route(align_detach_correction=True)


class KLGradientAblationTests(unittest.TestCase):
    def test_kl_value_and_raw_score_gradient_stay_equal_but_correction_gradient_stops(self):
        torch.manual_seed(321)
        raw = torch.randn(8, 4)
        reference = torch.randn(8, 48)
        sample_hidden = torch.randn(8, 16)
        log_q = F.log_softmax(torch.randn(8, 4), dim=1)
        values, raw_grads = [], []
        for detached in (False, True):
            model = h.perturb_correctors(h.build_clean_corr(align_lam=1.0, align_detach_correction=detached))
            block = model.moe_blocks()[0]
            block.set_phase("student", sample_sigma=torch.tensor([0.3, 0.7]))
            scores = raw.clone().requires_grad_()
            delta = block.corrector(reference, scores.detach(), sample_hidden)
            delta.retain_grad()
            block._corr_delta = delta
            block._teacher = {"log_q": log_q, "indices": log_q.argmax(dim=1, keepdim=True)}
            selected = (scores + delta).argmax(dim=1, keepdim=True)
            loss = block._student_losses(torch.tensor(0.0), selected, torch.ones(8, 1), scores,
                                         torch.ones(8, dtype=torch.bool), 4)
            loss.backward()
            values.append(loss.detach())
            raw_grads.append(scores.grad)
            if detached:
                self.assertIsNone(delta.grad)
                self.assertTrue(all(p.grad is None for p in block.corrector.parameters()))
            else:
                self.assertGreater(float(delta.grad.abs().sum()), 0.0)
                self.assertGreater(float(block.corrector.out.weight.grad.abs().sum()), 0.0)
        torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        torch.testing.assert_close(raw_grads[0], raw_grads[1], rtol=0, atol=0)
        self.assertGreater(float(raw_grads[1].abs().sum()), 0.0)

    def test_fm_still_trains_the_selected_correction_with_kl_detached(self):
        models = [h.bias_toward(h.perturb(h.build_clean_corr(
            align_lam=1.0, align_detach_correction=detached,
        )), expert=2, value=5.0) for detached in (False, True)]
        first, _ = h.train_step(models[0], "clean_corr")
        second, _ = h.train_step(models[1], "clean_corr")
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        # Use the last MoE block: no later layer KL can reach this corrector.
        normal_grad = models[0].moe_blocks()[-1].corrector.out.bias.grad
        detached_grad = models[1].moe_blocks()[-1].corrector.out.bias.grad
        self.assertGreater(float(detached_grad[2].abs()), 0.0)
        torch.testing.assert_close(detached_grad[[0, 1, 3]], torch.zeros(3), rtol=0, atol=0)
        self.assertTrue(bool((normal_grad[[0, 1, 3]].abs() > 0).all()))


class ScaleAndDistributedTests(unittest.TestCase):
    def test_l_and_xl_depths_log_the_last_moe_block(self):
        import train

        for depth in (24, 28):
            with self.subTest(depth=depth):
                # Native scale depth with small widths to keep this a CPU check.
                kwargs = dict(h.MODEL_KWARGS, depth=depth)
                model = h.perturb_correctors(h.perturb(h.clean_corr.DiT(
                    MoE_config=h.moe_config(), corr_router_config=h.corr_config(),
                    clean_route_config=h.route_config(align_lam=5e-4), **kwargs,
                )))
                h.train_step(model, "clean_corr")
                blocks = model.moe_blocks()
                self.assertEqual(len(blocks), depth // 2)
                stats = train._collect_clean_route_stats(model)
                stats.update(train._collect_corr_router_stats(model))
                torch.testing.assert_close(stats[f"clean_route_kl_b{len(blocks)-1}"], blocks[-1].last_kl)
                self.assertGreater(float(stats[f"corr_router_delta_abs_b{len(blocks)-1}"]), 0.0)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "needs gloo")
    def test_two_ddp_steps_with_masked_inputs_and_detached_kl(self):
        root = Path(__file__).resolve().parent
        for flag, detach in (("use_reference_embedding", False), ("use_score_input", False), (None, True)):
            with self.subTest(flag=flag, detach=detach), tempfile.TemporaryDirectory(
                dir=root, prefix=".corr_ablation_ddp_"
            ) as output:
                mp.spawn(h._ddp_worker, args=(2, str(Path(output) / "init"), output,
                                             "clean_corr", "model_velocity", flag, detach),
                         nprocs=2, join=True)
                results = [torch.load(Path(output) / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
                for steps in results:
                    self.assertEqual(len(steps), 2)
                    self.assertTrue(all(not step["missing_grads"] for step in steps))
                for step in range(2):
                    for name, value in results[0][step]["stats"].items():
                        self.assertAlmostEqual(value, results[1][step]["stats"][name], places=5, msg=name)


if __name__ == "__main__":
    unittest.main()
