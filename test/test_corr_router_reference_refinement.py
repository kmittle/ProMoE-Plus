"""CPU checks for current-reference attribution experiments; no real jobs."""
import copy
from pathlib import Path
import unittest

import torch
import yaml

from test import test_models_ProMoE_TC_corr_router as h
from test.test_clean_route_mechanisms import record_runs


MODES = ("current_history", "current_refined")
EXPERIMENTS = {
    "corr_router_current_history": ("current_history", h.INTERVALS),
    "corr_router_current_refined": ("current_refined", h.INTERVALS),
    "corr_router_previous_gap4": ("previous", [4]),
}


def clean_estimate(call, output_key="result"):
    x = call["x"]
    time = call["t"] if "t" in call else call["timestep"]
    sigma = (time / 1000).view((-1,) + (1,) * (x.dim() - 1))
    velocity = call[output_key][:, :4].float().reshape(x.shape)
    return x.float() - sigma * velocity


class TrainingReferenceRefinementTests(unittest.TestCase):
    def test_reference_sources_pass_counts_dropout_and_gradients(self):
        for mode in MODES:
            for autocast in (False, True):
                with self.subTest(mode=mode, autocast=autocast):
                    model = h.perturb_correctors(h.perturb(h.build_corr(reference_mode=mode)))
                    with record_runs(model) as calls:
                        output, grads = h.train_step(model, "corr", autocast=autocast)
                    auxiliary_count = 3 if mode == "current_history" else 2
                    self.assertEqual(len(calls), auxiliary_count + 1)
                    self.assertEqual([c["grad"] for c in calls], [False] * auxiliary_count + [True])
                    x, t, _, _, sigma = h.batch()
                    use, previous_sigma = h.expected_draws(0.5, sigma, h.dropped_labels())
                    self.assertTrue(bool(use.any()))
                    seed_call = calls[-3]
                    refinement = calls[-2]
                    seed, seed_sigma, seed_mask, _ = refinement["reference"]
                    torch.testing.assert_close(seed, clean_estimate(seed_call))
                    expected_sigma = previous_sigma[use] if mode == "current_history" else sigma.reshape(-1)[use]
                    torch.testing.assert_close(seed_sigma, expected_sigma)
                    self.assertTrue(bool(seed_mask.all()))
                    torch.testing.assert_close(refinement["x"], x[use])
                    torch.testing.assert_close(refinement["t"], t[use])
                    if mode == "current_history":
                        delta = (previous_sigma[use] - sigma.reshape(-1)[use]).view(-1, 1, 1, 1, 1)
                        expected_previous = x[use] + delta * calls[0]["result"][:, :4].float().reshape(x[use].shape)
                        torch.testing.assert_close(seed_call["x"], expected_previous)
                    final_ref, final_sigma, final_mask, _ = calls[-1]["reference"]
                    self.assertFalse(final_ref.requires_grad)
                    self.assertFalse(seed.requires_grad)
                    torch.testing.assert_close(final_mask, use)
                    torch.testing.assert_close(final_ref[use], clean_estimate(refinement))
                    self.assertEqual(int(torch.count_nonzero(final_ref[~use])), 0)
                    torch.testing.assert_close(final_sigma[use], sigma.reshape(-1)[use])
                    torch.testing.assert_close(final_sigma[~use], torch.ones_like(final_sigma[~use]))
                    self.assertTrue(bool(torch.isfinite(output).all()))
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad:
                            self.assertIn(name, grads)
                            self.assertTrue(bool(torch.isfinite(grads[name]).all()), name)

    def test_no_auxiliary_work_for_unconditional_or_dropped_references(self):
        for mode in MODES:
            for kwargs in (dict(labels=torch.full_like(h.LABELS, 1000)), {}):
                with self.subTest(mode=mode, kwargs=kwargs):
                    model = h.perturb(h.build_corr(reference_mode=mode, no_reference_prob=0.999999))
                    with record_runs(model) as calls:
                        h.train_step(model, "corr", **kwargs)
                    self.assertEqual(len(calls), 1)
                    self.assertFalse(bool(calls[0]["reference"][2].any()))

    def test_no_clean_target_leak_and_no_parameter_or_rng_changes_at_initialization(self):
        original = h.build_corr(reference_mode="current")
        expected_rng = torch.get_rng_state().clone()
        for mode in MODES:
            with self.subTest(mode=mode):
                model = h.build_corr(reference_mode=mode)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                self.assertEqual(set(model.state_dict()), set(original.state_dict()))
                for name, value in original.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                model = h.perturb_correctors(h.perturb(model))
                first, grads = h.train_step(model, "corr")
                _, _, z, noise, _ = h.batch()
                second, changed = h.train_step(model, "corr", z=z + 9, noise=noise - 7)
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                for name in grads:
                    torch.testing.assert_close(grads[name], changed[name], rtol=0, atol=0, msg=name)

    def test_modes_require_model_velocity(self):
        for mode in MODES:
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "model_velocity"):
                h.build_corr(reference_mode=mode, previous_state_mode="same_noise")


class SamplingReferenceRefinementTests(unittest.TestCase):
    def test_history_and_two_rounds_use_cfg_and_only_final_velocity_advances_latents(self):
        for mode in MODES:
            for guide in (1.0, 1.5):
                with self.subTest(mode=mode, guide=guide):
                    model = h.perturb_correctors(h.perturb(h.build_corr(reference_mode=mode))).eval()
                    recorder = h._RecordingModel(model)
                    recorder.module = model
                    helper = h.SamplerLoopTests()
                    result = helper.run_loop(recorder, True, guide, "cfg")[-1]
                    cond_count = 2 if mode == "current_history" else 3
                    width = cond_count + int(guide > 1)
                    self.assertEqual(len(recorder.calls), width * helper.STEPS)
                    last_reference = None
                    for step in range(helper.STEPS):
                        group = recorder.calls[step * width:(step + 1) * width]
                        cond = [c for c in group if bool((c["context"] != 1000).all())]
                        uncond = [c for c in group if bool((c["context"] == 1000).all())]
                        self.assertEqual(len(cond), cond_count)
                        self.assertEqual(len(uncond), int(guide > 1))
                        self.assertTrue(all(not c["reference"] for c in uncond))
                        if mode == "current_history" and step:
                            for key, value in last_reference.items():
                                torch.testing.assert_close(cond[0]["reference"][key], value)
                        else:
                            self.assertFalse(cond[0]["reference"])
                        for i, call in enumerate(cond):
                            torch.testing.assert_close(call["x"], cond[0]["x"], rtol=0, atol=0)
                            torch.testing.assert_close(call["timestep"], cond[0]["timestep"], rtol=0, atol=0)
                            fused = call["output"]
                            if uncond:
                                fused = uncond[0]["output"] + guide * (fused - uncond[0]["output"])
                            sigma = 1 - step / helper.STEPS
                            reference = h.corr_router.sampling_reference(call["x"], fused, sigma)
                            if i + 1 < len(cond):
                                for key, value in reference.items():
                                    torch.testing.assert_close(cond[i + 1]["reference"][key], value)
                            else:
                                last_reference = reference
                                velocity = fused[:, :4].float().reshape(call["x"].shape)
                                expected_next = (call["x"].float() - velocity / helper.STEPS).to(fused.dtype)
                                actual_next = recorder.calls[(step + 1) * width]["x"] if step + 1 < helper.STEPS else result
                                torch.testing.assert_close(actual_next, expected_next)

    def test_new_modes_reject_non_cfg_references(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                model = h.build_corr(reference_mode=mode).eval()
                with self.assertRaisesRegex(ValueError, "requires corr_reference_velocity=cfg"):
                    h.SamplerLoopTests().run_loop(model, True, reference_velocity="conditional")


class ConfigReferenceRefinementTests(unittest.TestCase):
    def test_new_configs_merge_construct_and_share_sampling_registry(self):
        import train
        import sample
        from config import cfg
        from utils import deep_update

        root = Path(__file__).resolve().parents[1]
        for name, (mode, intervals) in EXPERIMENTS.items():
            with self.subTest(name=name):
                data = yaml.safe_load((root / "configs" / ("004_ProMoE_B_" + name + ".yaml")).read_text())
                merged = copy.deepcopy(cfg)
                deep_update(merged, data)
                cls, key = train.model_dict[data["model_name"]]
                self.assertEqual(sample.model_dict[data["model_name"]], (cls, key))
                kwargs = dict(merged[key])
                kwargs.update(h.MODEL_KWARGS)
                kwargs["MoE_config"].moe_intermediate_size = 32
                kwargs["MoE_config"].shared_expert_intermediate_size = 32
                model = cls(**kwargs)
                self.assertEqual(model.reference_mode, mode)
                self.assertEqual(model.skip_intervals, tuple(intervals))
                self.assertEqual(model.no_reference_prob, 0.5)
                self.assertFalse(model.zero_reference_at_t1000)
                self.assertFalse(getattr(model, "is_clean_route_model", False))
                self.assertEqual(data["gpu_ids"], [0, 1, 2, 3])
                self.assertEqual(data["latent_data_path"], "/mnt/data2/yujie/code/Dataset/ImageNet/sd-vae-ft-mse_Latents_256img_npz")
                h.train_step(model, "corr")


if __name__ == "__main__":
    unittest.main()
