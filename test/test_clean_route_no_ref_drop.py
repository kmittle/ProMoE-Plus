"""Check the real-clean no-drop control on CPU and with stub pipeline commands."""
import copy
from pathlib import Path
import unittest

import torch
import yaml

from test import test_models_ProMoE_TC_corr_router as h
from test.test_clean_route_mechanisms import record_runs
from test.test_corr_router_sampling_scripts import DEPLOYMENT_LATENTS, fixture


PARENT = "clean_route_student_ratio0_corr_router_kl_detach_base"
ARM = f"{PARENT}_no_ref_drop"
ESTIMATED = "estimated_route_student_corr_router_kl_detach_base_no_ref_drop"
ROOT = Path(__file__).resolve().parents[1]


def load_config(arm):
    return yaml.safe_load((ROOT / "configs" / f"004_ProMoE_B_{arm}.yaml").read_text())


class CleanRouteNoReferenceDropoutTests(unittest.TestCase):
    def test_config_matches_both_controls_and_forward_uses_real_clean_teacher(self):
        import sample
        import train
        from config import cfg
        from utils import deep_update

        data = load_config(ARM)
        expected = load_config(PARENT)
        expected["DiT_B_config"]["corr_router_config"].update(
            no_reference_prob=0.0, zero_reference_at_t1000=True,
        )
        self.assertEqual(data, expected)
        estimated = load_config(ESTIMATED)
        teacher = estimated["DiT_B_config"]["clean_route_config"]
        self.assertEqual(teacher.pop("teacher_input"), "estimated")
        self.assertFalse(teacher.pop("align_detach_correction"))
        self.assertEqual(data, estimated)
        self.assertEqual(data["latent_data_path"], DEPLOYMENT_LATENTS)

        name = data["model_name"]
        cls, key = train.model_dict[name]
        self.assertEqual(sample.model_dict[name], (cls, key))
        merged = copy.deepcopy(cfg)
        deep_update(merged, data)
        kwargs = dict(merged[key], **h.MODEL_KWARGS)
        kwargs["MoE_config"].moe_intermediate_size = 32
        kwargs["MoE_config"].shared_expert_intermediate_size = 32
        torch.manual_seed(0)
        model = h.perturb(h.perturb_correctors(cls(**kwargs)))
        self.assertEqual(model.teacher_input, "clean")
        self.assertEqual(model.reference_dropout_mode, "zero")
        self.assertTrue(all(b.align_detach_base and not b.align_detach_correction for b in model.moe_blocks()))

        _, _, z, noise, _ = h.batch()
        t = torch.tensor([1000.0, 810.0, 999.9999, 996.0, 500.0, 350.0])
        sigma = t / 1000
        scale = sigma.view(-1, 1, 1, 1, 1)
        x = (1 - scale) * z + scale * noise
        labels = h.dropped_labels()
        cond = labels != 1000
        torch.manual_seed(h.DROPOUT_SEED)
        with record_runs(model) as calls, torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(x, t, **h.model_inputs("clean_corr", h.LABELS, z, noise, sigma))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertEqual([call["grad"] for call in calls], [False, False, False, True])
        teacher_call, student = calls[2:]
        self.assertIsNone(teacher_call["reference"])
        torch.testing.assert_close(teacher_call["x"], z[cond], rtol=0, atol=0)
        torch.testing.assert_close(teacher_call["t"], torch.zeros_like(t[cond]), rtol=0, atol=0)
        torch.testing.assert_close(teacher_call["labels"], labels[cond])
        torch.testing.assert_close(student["labels"], labels)
        ref_x0, ref_sigma, has_ref, _ = student["reference"]
        _, previous_sigma = h.expected_draws(0.0, sigma, labels)
        expected_ref = cond & (t != 1000)
        torch.testing.assert_close(has_ref, expected_ref)
        torch.testing.assert_close(ref_sigma, torch.where(expected_ref, previous_sigma, 1))
        self.assertFalse(bool(ref_x0[~expected_ref].any()))
        self.assertTrue(bool(has_ref[2]))
        self.assertEqual(ref_sigma[2].item(), 1.0)

    def test_pipeline_uses_its_own_checkpoint_and_rechecks_both_strict_thresholds(self):
        with fixture(ARM, reference_dropout=True) as (_, output, run, events):
            result = run(PROMOE_LATENT_PATH="")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = events()
            self.assertEqual([row["kind"] for row in rows], ["train", "sample", "eval", "eval"] * 2)
            self.assertEqual(rows[0]["config"]["num_steps"], 300001)
            self.assertNotIn("resume_checkpoint_step", rows[0]["config"])
            self.assertEqual(rows[4]["config"]["num_steps"], 500001)
            self.assertEqual(rows[4]["config"]["resume_checkpoint_step"], 300000)
            for row in (rows[0], rows[4]):
                self.assertEqual(row["config"]["latent_data_path"], DEPLOYMENT_LATENTS)
                self.assertEqual(row["config"]["gpu_ids"], [4, 5, 6, 7])
            for step, row in ((300000, rows[1]), (500000, rows[5])):
                self.assertEqual(row["scales"], [1.0, 1.5])
                self.assertEqual(row["mode"], "cfg")
                self.assertEqual(row["source"], str(output / f"checkpoints/ckpt_step_{step}.pth"))
            self.assertTrue(all(row["count"] == 50000 for row in rows if row["kind"] == "eval"))

            resume = {"PROMOE_RESUME": "1", "PROMOE_LATENT_PATH": ""}
            result = run(**resume)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(events(), rows)
            for evaluator, threshold in ((rows[2], "30.86"), (rows[3], "9.73")):
                report = Path(evaluator["images"] + "_eval_openai.txt")
                original = report.read_text()
                report.write_text(original.replace("FID: 9.0\n", f"FID: {threshold}\n"))
                stopped = run(**resume)
                self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
                self.assertIn("300K gate STOP", stopped.stdout)
                self.assertEqual(events(), rows)
                report.write_text(original)


if __name__ == "__main__":
    unittest.main()
