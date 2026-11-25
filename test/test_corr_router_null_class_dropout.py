"""CPU checks of reference-side null labels, independent of ordinary class dropout."""
import copy
from pathlib import Path
import unittest
from unittest import mock

import torch
import yaml

from test import test_models_ProMoE_TC_corr_router as h
from test.test_clean_route_mechanisms import record_runs


PROB = 0.85
SHAPE = (-1, 1, 1, 1, 1)
ROOT = Path(__file__).resolve().parents[1]


def _velocity_by_label(x, labels):
    """Different, deterministic speeds for the real and null labels."""
    return (labels.to(x).view(SHAPE) / 1000 + 0.125).expand_as(x)


class NullClassReferenceTests(unittest.TestCase):
    def setUp(self):
        _, _, self.z, self.noise, _ = h.batch()
        self.t = torch.tensor([1000.0, 810.0, 999.9999, 996.0, 500.0, 350.0])
        self.sigma = self.t / 1000
        scale = self.sigma.view(SHAPE)
        self.x = (1 - scale) * self.z + scale * self.noise
        self.labels = h.LABELS.clone()
        self.cond = self.labels != h.corr_router.UNCOND_LABEL
        self.keep, self.previous_sigma = h.expected_draws(PROB, self.sigma, self.labels, intervals=[50])
        self.active = self.cond & (self.t != 1000)
        self.dropped = self.active & ~self.keep
        self.assertEqual(self.dropped.tolist(), [False, False, False, True, False, True])
        self.assertFalse(bool(self.keep[0]))  # Randomly selected, but the endpoint rule wins.

    def check_equal(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def run_reference(self, mode, *, estimate_all, previous_state_mode="model_velocity", prob=PROB):
        overrides = {} if mode is None else {"reference_dropout_mode": mode}
        model = h.build_corr(
            no_reference_prob=prob, zero_reference_at_t1000=True, skip_intervals=[50],
            previous_state_mode=previous_state_mode, **overrides,
        )
        y, labels = model.y_embedder(self.labels, False)
        calls = []

        def predict(x, t, embeddings, labels):
            self.assertFalse(torch.is_grad_enabled())
            calls.append((x.clone(), t.clone(), embeddings.clone(), labels.clone()))
            return _velocity_by_label(x, labels)

        global_rng = torch.get_rng_state()
        generator = torch.Generator().manual_seed(h.GENERATOR_SEED)
        with mock.patch.object(model, "_reference_velocity", side_effect=predict):
            ref, full = model._training_reference_with_estimate(
                self.x, self.t, self.sigma, y, labels, generator,
                z=self.z, noise=self.noise, estimate_all=estimate_all,
            )
        self.assertFalse(ref[0].requires_grad or full.requires_grad)
        self.check_equal(torch.get_rng_state(), global_rng)
        expected_rng = torch.Generator().manual_seed(h.GENERATOR_SEED)
        torch.rand(h.BATCH, generator=expected_rng)
        torch.randint(1, (h.BATCH,), generator=expected_rng)
        self.check_equal(generator.get_state(), expected_rng.get_state())
        # In-place replacements must only touch the local copies for pass 2.
        self.check_equal(labels, h.LABELS)
        self.check_equal(y, model.y_embedder(h.LABELS, False)[0])
        return ref, full, model, calls

    def conditional_previous(self):
        return self.x + (self.previous_sigma - self.sigma).view(SHAPE) * _velocity_by_label(self.x, self.labels)

    def test_omitted_and_explicit_zero_mode_match_legacy_references_exactly(self):
        previous = self.conditional_previous()
        cond_estimate = previous - self.previous_sigma.view(SHAPE) * _velocity_by_label(previous, self.labels)
        keep = self.active & self.keep
        for mode in (None, "zero"):
            for estimate_all in (False, True):
                with self.subTest(mode=mode, estimate_all=estimate_all):
                    reference, full, model, calls = self.run_reference(mode, estimate_all=estimate_all)
                    estimated = self.cond if estimate_all else keep
                    self.check_equal(full, torch.where(estimated.view(SHAPE), cond_estimate, 0))
                    self.check_equal(reference[0], torch.where(keep.view(SHAPE), cond_estimate, 0))
                    self.check_equal(reference[1], torch.where(keep, self.previous_sigma, 1))
                    self.check_equal(reference[2], keep)
                    self.check_equal(calls[0][3], self.labels[estimated])
                    self.check_equal(calls[1][0], previous[estimated])
                    self.assertEqual(model.last_corr_null_ref_count.item(), 0)

    def test_estimated_teacher_uses_only_conditional_half_of_the_shared_forward(self):
        old_ref, old_estimate, _, old_calls = self.run_reference("zero", estimate_all=True)
        reference, estimate, model, calls = self.run_reference("null_class", estimate_all=True)
        self.assertEqual(len(calls), 2)
        for actual, expected in zip(calls[0], old_calls[0]):
            self.check_equal(actual, expected)
        previous, t_ref, y, labels = calls[1]
        nc = int(self.cond.sum())
        self.check_equal(previous[:nc], old_calls[1][0])
        self.check_equal(previous[nc:], previous[:nc][self.dropped[self.cond]])
        self.check_equal(t_ref, torch.cat((old_calls[1][1], self.previous_sigma[self.dropped] * 1000)))
        null_labels = torch.full_like(self.labels[self.dropped], h.corr_router.UNCOND_LABEL)
        self.check_equal(labels, torch.cat((self.labels[self.cond], null_labels)))
        self.check_equal(y[:nc], old_calls[1][2])
        self.check_equal(y[nc:], model.y_embedder(null_labels, False)[0])
        self.check_equal(estimate, old_estimate)

        null_estimate = previous[nc:] - self.previous_sigma[self.dropped].view(SHAPE) * _velocity_by_label(
            previous[nc:], null_labels
        )
        self.check_equal(reference[0][self.dropped], null_estimate)
        self.assertTrue(torch.all(reference[0][self.dropped] != estimate[self.dropped]))
        self.check_equal(reference[0][~self.dropped], old_ref[0][~self.dropped])
        self.check_equal(reference[2], self.active)
        self.check_equal(reference[1], torch.where(self.active, self.previous_sigma, 1))
        self.assertNotEqual(reference[1][-1].item(), 1)
        self.assertEqual(model.last_corr_null_ref_count.item(), int(self.dropped.sum()))

    def test_without_estimated_teacher_switches_labels_in_place_in_pass_two(self):
        for previous_mode in ("model_velocity", "same_noise"):
            with self.subTest(previous_mode=previous_mode):
                reference, _, model, calls = self.run_reference(
                    "null_class", estimate_all=False, previous_state_mode=previous_mode,
                )
                if previous_mode == "model_velocity":
                    self.assertEqual(len(calls), 2)
                    self.check_equal(calls[0][0], self.x[self.cond])
                    self.check_equal(calls[0][3], self.labels[self.cond])
                    previous = self.conditional_previous()
                else:
                    self.assertEqual(len(calls), 1)
                    scale = self.previous_sigma.view(SHAPE)
                    previous = (1 - scale) * self.z + scale * self.noise
                labels = self.labels.masked_fill(self.dropped, h.corr_router.UNCOND_LABEL)
                self.check_equal(calls[-1][0], previous[self.cond])
                self.check_equal(calls[-1][3], labels[self.cond])
                self.check_equal(calls[-1][2], model.y_embedder(labels[self.cond], False)[0])
                result = previous - self.previous_sigma.view(SHAPE) * _velocity_by_label(previous, labels)
                self.check_equal(reference[0], torch.where(self.active.view(SHAPE), result, 0))
                self.check_equal(reference[1], torch.where(self.active, self.previous_sigma, 1))
                self.check_equal(reference[2], self.active)

    def test_zero_probability_does_not_append_any_samples(self):
        zero, old_estimate, _, old_calls = self.run_reference("zero", estimate_all=True, prob=0)
        null, estimate, model, calls = self.run_reference("null_class", estimate_all=True, prob=0)
        self.assertEqual(len(calls), len(old_calls))
        self.assertEqual(calls[-1][0].shape[0], int(self.cond.sum()))
        self.check_equal(estimate, old_estimate)
        for actual, expected in zip(null, zero):
            self.check_equal(actual, expected)
        self.assertEqual(model.last_corr_null_ref_count.item(), 0)

    def test_invalid_mode_and_unsupported_reference_modes_raise(self):
        for mode in (None, False, 1, [], "conditional", "NULL_CLASS"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "reference_dropout_mode"):
                h.build_corr(reference_dropout_mode=mode)
        for mode in h.corr_router.CURRENT_REFERENCE_MODES:
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "requires reference_mode: previous"):
                h.build_corr(reference_dropout_mode="null_class", reference_mode=mode)
        with self.assertRaisesRegex(ValueError, "requires the null class embedding"):
            h.corr_router.DiT(
                MoE_config=h.moe_config(), corr_router_config=h.corr_config(reference_dropout_mode="null_class"),
                class_dropout_prob=0, **h.MODEL_KWARGS,
            )


class NullClassIntegrationTests(unittest.TestCase):
    def test_real_forward_preserves_teacher_input_and_formal_student_labels(self):
        records = []
        for mode in ("zero", "null_class"):
            model = h.perturb_correctors(h.perturb(h.build_clean_corr(
                corr_overrides=dict(reference_dropout_mode=mode, no_reference_prob=PROB),
                teacher_input="estimated", align_detach_base=True, align_lam=5e-4,
            )))
            with record_runs(model) as calls:
                output, grads = h.train_step(model, "clean_corr")
            self.assertTrue(bool(torch.isfinite(output).all()))
            self.assertEqual({n for n, p in model.named_parameters() if p.requires_grad}, set(grads))
            self.assertEqual([c["grad"] for c in calls], [False, False, False, True])
            self.assertIsNone(calls[2]["reference"])
            self.assertFalse(bool(calls[2]["t"].any()))
            records.append(calls)
        zero, null = records
        torch.testing.assert_close(null[0]["x"], zero[0]["x"], rtol=0, atol=0)
        torch.testing.assert_close(null[1]["x"][:len(zero[1]["x"])], zero[1]["x"], rtol=0, atol=0)
        # Mathematically the conditional estimates are identical. A different
        # pass-2 batch size may select different floating point kernels.
        torch.testing.assert_close(null[2]["x"], zero[2]["x"])
        torch.testing.assert_close(null[2]["labels"], zero[2]["labels"])
        torch.testing.assert_close(null[3]["labels"], zero[3]["labels"])
        self.assertGreater(len(null[1]["labels"]), len(zero[1]["labels"]))
        self.assertGreater(int(null[3]["reference"][2].sum()), int(zero[3]["reference"][2].sum()))

    def test_three_configs_are_their_matched_controls_plus_one_flag(self):
        import sample
        import train
        from config import cfg
        from utils import deep_update

        prefix = "004_ProMoE_B_estimated_route_student_corr_router_kl_detach_base_ref_"
        for rate in ("0p05", "0p1", "0p25"):
            with self.subTest(rate=rate):
                data = yaml.safe_load((ROOT / "configs" / f"{prefix}null_drop{rate}.yaml").read_text())
                expected = yaml.safe_load((ROOT / "configs" / f"{prefix}drop{rate}.yaml").read_text())
                expected["DiT_B_config"]["corr_router_config"]["reference_dropout_mode"] = "null_class"
                self.assertEqual(data, expected)
                cls, key = train.model_dict[data["model_name"]]
                self.assertEqual(sample.model_dict[data["model_name"]], (cls, key))
                merged = copy.deepcopy(cfg)
                deep_update(merged, data)
                kwargs = dict(merged[key], **h.MODEL_KWARGS)
                kwargs["MoE_config"].moe_intermediate_size = 32
                kwargs["MoE_config"].shared_expert_intermediate_size = 32
                model = cls(**kwargs)
                self.assertEqual(model.reference_dropout_mode, "null_class")
                output, _ = h.train_step(model, "clean_corr", autocast=True)
                self.assertTrue(bool(torch.isfinite(output).all()))
                stats = train._collect_corr_router_stats(model)
                self.assertEqual(set(stats), set(train.CORR_ROUTER_STAT_NAMES))
                self.assertTrue(all(bool(torch.isfinite(value)) for value in stats.values()))


if __name__ == "__main__":
    unittest.main()
