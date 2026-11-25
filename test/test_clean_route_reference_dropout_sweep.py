"""Verify the real-clean zero-reference sweep without launching any jobs."""
import copy
import unittest

import torch

from test import test_models_ProMoE_TC_corr_router as h
from test.test_clean_route_mechanisms import record_runs
from test.test_clean_route_no_ref_drop import ARM as NO_DROP, ESTIMATED, PARENT, load_config
from test.test_corr_router_sampling_scripts import DEPLOYMENT_LATENTS, fixture


RATES = (("0p01", 0.01), ("0p05", 0.05), ("0p1", 0.1), ("0p25", 0.25))


class RealCleanReferenceDropoutSweepTests(unittest.TestCase):
    def test_configs_and_cpu_forward_match_the_zero_reference_controls(self):
        import sample
        import train
        from config import cfg
        from utils import deep_update

        # Draw one dropout even at 1% and retain a near-endpoint image at every rate.
        for seed in range(10000):
            coins = torch.rand(h.BATCH, generator=torch.Generator().manual_seed(seed))
            if coins[5] < 0.01 and coins[2] >= 0.25:
                break
        else:
            self.fail("Could not construct a batch that exercises all dropout rates")
        _, _, z, noise, _ = h.batch()
        t = torch.tensor([1000.0, 810.0, 999.9999, 996.0, 500.0, 350.0])
        sigma = t / 1000
        scale = sigma.view(-1, 1, 1, 1, 1)
        x = (1 - scale) * z + scale * noise
        labels = h.dropped_labels()
        cond = labels != 1000
        self.assertTrue(cond[5].item())

        for tag, prob in RATES:
            arm = f"{PARENT}_ref_drop{tag}"
            with self.subTest(arm=arm):
                data = load_config(arm)
                control = load_config(NO_DROP)
                control["DiT_B_config"]["corr_router_config"]["no_reference_prob"] = prob
                self.assertEqual(data, control)
                if prob != 0.01:
                    estimated = load_config(f"{ESTIMATED.removesuffix('_no_ref_drop')}_ref_drop{tag}")
                    teacher = estimated["DiT_B_config"]["clean_route_config"]
                    self.assertEqual(teacher.pop("teacher_input"), "estimated")
                    self.assertFalse(teacher.pop("align_detach_correction"))
                    self.assertEqual(data, estimated)

                cls, key = train.model_dict[data["model_name"]]
                self.assertEqual(sample.model_dict[data["model_name"]], (cls, key))
                merged = copy.deepcopy(cfg)
                deep_update(merged, data)
                kwargs = dict(merged[key], **h.MODEL_KWARGS)
                kwargs["MoE_config"].moe_intermediate_size = 32
                kwargs["MoE_config"].shared_expert_intermediate_size = 32
                torch.manual_seed(0)
                model = cls(**kwargs)
                self.assertEqual(model.teacher_input, "clean")
                self.assertEqual(model.reference_dropout_mode, "zero")
                self.assertEqual(model.no_reference_prob, prob)
                torch.manual_seed(h.DROPOUT_SEED)
                inputs = h.model_inputs("clean_corr", h.LABELS, z, noise, sigma, generator_seed=seed)
                with record_runs(model) as calls, torch.autocast("cpu", dtype=torch.bfloat16):
                    output = model(x, t, **inputs)
                self.assertTrue(bool(torch.isfinite(output).all()))
                self.assertEqual([call["grad"] for call in calls], [False, False, False, True])
                torch.testing.assert_close(calls[2]["x"], z[cond], rtol=0, atol=0)
                torch.testing.assert_close(calls[-1]["labels"], labels)
                reference, reference_sigma, mask, _ = calls[-1]["reference"]
                keep, previous_sigma = h.expected_draws(prob, sigma, labels, generator_seed=seed)
                expected = keep & (t != 1000)
                torch.testing.assert_close(mask, expected)
                torch.testing.assert_close(reference_sigma, torch.where(expected, previous_sigma, 1))
                self.assertFalse(bool(reference[~expected].any()))
                self.assertFalse(mask[5].item())  # Randomly dropped, not the first timestep.
                self.assertTrue(mask[2].item())   # Ref sigma clamps to 1, current t is below 1000.
                self.assertEqual(reference_sigma[2].item(), 1)

    def test_all_four_launchers_pass_the_correct_rate_and_resume_their_own_run(self):
        for tag, prob in RATES:
            arm = f"{PARENT}_ref_drop{tag}"
            with self.subTest(arm=arm), fixture(arm, reference_dropout=True) as (_, output, run, events):
                started = run(PROMOE_LATENT_PATH="")
                self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
                rows = events()
                self.assertEqual([row["kind"] for row in rows], ["train", "sample", "eval", "eval"] * 2)
                self.assertNotIn("resume_checkpoint_step", rows[0]["config"])
                self.assertEqual(rows[4]["config"]["resume_checkpoint_step"], 300000)
                for step, training, sampling in ((300000, rows[0], rows[1]), (500000, rows[4], rows[5])):
                    settings = training["config"]
                    self.assertEqual(settings["num_steps"], step + 1)
                    self.assertEqual(settings["latent_data_path"], DEPLOYMENT_LATENTS)
                    reference = settings["DiT_B_config"]["corr_router_config"]
                    self.assertEqual(reference["no_reference_prob"], prob)
                    self.assertEqual(reference.get("reference_dropout_mode", "zero"), "zero")
                    self.assertEqual((sampling["mode"], sampling["scales"]), ("cfg", [1.0, 1.5]))
                    self.assertEqual(sampling["source"], str(output / f"checkpoints/ckpt_step_{step}.pth"))
                resumed = run(PROMOE_RESUME="1", PROMOE_LATENT_PATH="")
                self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                self.assertEqual(events(), rows)


if __name__ == "__main__":
    unittest.main()
