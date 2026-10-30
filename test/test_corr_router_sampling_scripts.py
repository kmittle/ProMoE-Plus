"""Exercise orchestration with CPU stubs; never train, sample or claim real GPUs."""
import ast
import contextlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml


REPO = Path(__file__).resolve().parents[1]
BASES = ("corr_router", "clean_route_student_ratio0_corr_router")


@contextlib.contextmanager
def fixture(base, inference=False):
    with tempfile.TemporaryDirectory(dir=REPO / "test", prefix=".corr_shell_") as directory:
        root = Path(directory)
        tag = base + ("_cfg_ref" if inference else "_same_noise")
        script = f"scripts/corr_router/run_B_{tag}_{'sample_eval' if inference else 'train_sample_eval'}.sh"
        paths = [script, "scripts/_eval_metric_helpers.sh", "scripts/corr_router/_dual_reference_eval.sh",
                 f"configs/004_ProMoE_B_{tag}.yaml"]
        if not inference:
            paths += [f"configs/004_ProMoE_B_{tag}_{suffix}.yaml" for suffix in ("cond_ref", "cfg_ref")]
        for path in paths:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / path, target)
        (root / "scripts/_python_env.sh").write_text(
            f"PROMOE_TRAIN_PYTHON={shlex.quote(sys.executable)}\nPROMOE_EVAL_PYTHON={shlex.quote(sys.executable)}\n"
        )
        (root / "bin").mkdir()
        (root / "bin/git").write_text("#!/bin/sh\nexit 1\n")
        (root / "bin/git").chmod(0o755)
        (root / "evaluation").mkdir()
        (root / "stub_io.py").write_text('''import json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def record(kind, **fields):
    with (ROOT / "events.jsonl").open("a") as f:
        f.write(json.dumps(dict(kind=kind, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), **fields)) + "\\n")
''')
        (root / "train.py").write_text('''import argparse, os
from pathlib import Path
import yaml
from stub_io import record
p = argparse.ArgumentParser(); p.add_argument("--config"); a = p.parse_args()
c = yaml.safe_load(Path(a.config).read_text())
record("train", config=c)
if not os.getenv("SKIP_CHECKPOINT"):
    target = Path("outputs") / c["model_name"] / Path(a.config).stem / "checkpoints" / "ckpt_step_300000.pth"
    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b"same checkpoint")
''')
        # Use the actual naming function so the shell must agree with sample.py.
        module = ast.parse((REPO / "sample.py").read_text())
        naming = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                      and node.name == "sampling_folder_name")
        (root / "sample.py").write_text(ast.unparse(naming) + '\n' + '''import argparse, os
from pathlib import Path
from types import SimpleNamespace
import yaml
from stub_io import record
p = argparse.ArgumentParser()
for key in ("config", "step_list_for_sample", "guide_scale_list", "num_fid_samples", "corr-reference-velocity"):
    p.add_argument("--" + key)
a = p.parse_args(); c = yaml.safe_load(Path(a.config).read_text())
mode = a.corr_reference_velocity
scales = [float(x) for x in a.guide_scale_list.split(",")]
steps = [int(x) for x in a.step_list_for_sample.split(",")]
assert len(steps) == 1
step = steps[0]
output = Path(c.get("output_dir", "outputs")) / c["model_name"] / Path(a.config).stem
source = Path(c["sample_checkpoint_dir"]) / f"ckpt_step_{step}.pth"
assert source.is_file() and source.stat().st_size
record("sample", mode=mode, step=step, scales=scales, source=str(source.resolve()), output=str(output.resolve()), seed=c["global_seed"])
if os.getenv("FAIL_SAMPLE_MODE") == mode:
    raise SystemExit(17)
c["corr_reference_velocity"] = mode
for scale in scales:
    folder = output / "sample" / f"step{step}" / sampling_folder_name(SimpleNamespace(**c), scale) / "images"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "stub.png").write_bytes(b"stub")
''')
        (root / "evaluation/run_eval.py").write_text('''import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stub_io import record
p = argparse.ArgumentParser(); p.add_argument("images"); p.add_argument("--count"); a = p.parse_args()
record("eval", images=a.images, count=int(a.count))
metric = "nan" if os.getenv("BAD_METRICS") else "9.0"
Path(a.images + "_eval_openai.txt").write_text("FID: " + metric + "\\nInception Score: 100.0\\n")
''')
        cfg = yaml.safe_load((root / f"configs/004_ProMoE_B_{tag}.yaml").read_text())
        output = root / "outputs" / cfg["model_name"] / f"004_ProMoE_B_{tag}"
        if inference:
            for step in (300000, 500000):
                checkpoint = root / cfg["sample_checkpoint_dir"] / f"ckpt_step_{step}.pth"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(f"original checkpoint {step}".encode())
        env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                   PROMOE_GPU_IDS_OVERRIDE="4,5,6,7", PROMOE_LATENT_PATH="/input/latents", PROMOE_RESUME="0",
                   PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""), TMPDIR=str(root))
        if inference:
            env.pop("PROMOE_RESUME", None)  # Test the fresh-reevaluation default.

        def run(**overrides):
            return subprocess.run(["bash", script], cwd=root, env=dict(env, **overrides),
                                  text=True, capture_output=True, timeout=30)

        def events():
            path = root / "events.jsonl"
            return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

        yield root, output, run, events


class SamplingScriptTests(unittest.TestCase):
    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_train_once_and_evaluate_two_independent_experiments_then_resume(self):
        for base in BASES:
            with self.subTest(base=base), fixture(base) as (root, output, run, events):
                self.assert_success(run())
                log = events()
                self.assertEqual([row["kind"] for row in log], ["train", "sample", "eval", "eval", "sample", "eval", "eval"])
                training = log[0]
                self.assertEqual(training["config"]["gpu_ids"], [4, 5, 6, 7])
                self.assertEqual(training["config"]["latent_data_path"], "/input/latents")
                self.assertEqual(training["config"]["num_steps"], 300001)
                samples = [row for row in log if row["kind"] == "sample"]
                self.assertEqual([row["mode"] for row in samples], ["conditional", "cfg"])
                self.assertEqual(samples[0]["source"], samples[1]["source"])
                self.assertTrue(all(row["scales"] == [1.0, 1.5] and row["gpu"] == "4,5,6,7" for row in samples))
                self.assertTrue(all(row["gpu"] == "4" and row["count"] == 50000 for row in log if row["kind"] == "eval"))
                layouts = []
                for suffix in ("cond_ref", "cfg_ref"):
                    directory = Path(str(output) + "_" + suffix)
                    layouts.append(sorted(str(p.relative_to(directory)) for p in directory.rglob("*")))
                self.assertEqual(*layouts)
                self.assertFalse(any("_corr_ref_cfg" in p for p in layouts[0]))
                self.assertFalse((output / "sample").exists())
                self.assert_success(run(PROMOE_RESUME="1"))
                self.assertEqual(events(), log)

    def test_resume_only_the_unfinished_reference_experiment(self):
        with fixture(BASES[0]) as (root, output, run, events):
            self.assertNotEqual(run(FAIL_SAMPLE_MODE="cfg").returncode, 0)
            before = events()
            self.assert_success(run(PROMOE_RESUME="1"))
            added = events()[len(before):]
            self.assertEqual([row["kind"] for row in added], ["sample", "eval", "eval"])
            self.assertEqual(added[0]["mode"], "cfg")
            metric = next(Path(str(output) + "_cfg_ref").rglob("images_eval_openai.txt"))
            metric.unlink()
            before = events()
            self.assert_success(run(PROMOE_RESUME="1"))
            self.assertEqual([row["kind"] for row in events()[len(before):]], ["sample", "eval"])

    def test_bad_metrics_and_missing_checkpoint_fail(self):
        for overrides in ({"BAD_METRICS": "1"}, {"SKIP_CHECKPOINT": "1"}):
            with self.subTest(overrides=overrides), fixture(BASES[0]) as (root, output, run, events):
                self.assertNotEqual(run(**overrides).returncode, 0)
                self.assertFalse(any(row.get("mode") == "cfg" for row in events()))

    def test_invalid_gpu_override_and_occupied_eval_output_stop_before_training(self):
        with fixture(BASES[0]) as (root, output, run, events):
            self.assertNotEqual(run(PROMOE_GPU_IDS_OVERRIDE="0,1").returncode, 0)
            self.assertEqual(events(), [])
            occupied = Path(str(output) + "_cfg_ref")
            occupied.mkdir(parents=True)
            (occupied / "keep.txt").write_text("existing result")
            self.assertNotEqual(run().returncode, 0)
            self.assertEqual(events(), [])

    def complete_case(self, output, step, scale, legacy=False):
        folder = output / f"sample/step{step}/img256_cfg{scale}_seed0_FID50K_bs128_ema"
        if legacy:
            folder = Path(str(folder) + "_corr_ref_cfg")
        (folder / "images").mkdir(parents=True)
        (folder / "images/keep.png").write_bytes(b"existing image")
        (folder / "images_eval_openai.txt").write_text("FID: 8.17\nInception Score: 130.39\n")
        return folder

    def test_unified_inference_launchers_cover_grid_and_require_explicit_resume(self):
        for base in BASES:
            with self.subTest(base=base), fixture(base, inference=True) as (root, output, run, events):
                self.assert_success(run())
                log = events()
                self.assertEqual([row["kind"] for row in log], ["sample", "eval"] * 4)
                samples = log[::2]
                self.assertEqual([(row["step"], row["scales"]) for row in samples],
                                 [(300000, [1.0]), (300000, [1.5]), (500000, [1.0]), (500000, [1.5])])
                for row in samples:
                    self.assertEqual(Path(row["source"]).read_bytes(), f"original checkpoint {row['step']}".encode())
                    self.assertEqual(row["mode"], "cfg")
                    self.assertEqual(row["gpu"], "4,5,6,7")
                    self.assertEqual(row["seed"], 0)
                    self.assertEqual(Path(row["output"]), output)
                for row in log[1::2]:
                    self.assertNotIn("_corr_ref_cfg", row["images"])
                    self.assertEqual(row["count"], 50000)
                    self.assertEqual(row["gpu"], "4")
                result = run()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("output bucket is not empty", result.stderr)
                self.assertEqual(events(), log)
                self.assert_success(run(PROMOE_RESUME="1"))
                self.assertEqual(events(), log)

    def test_resume_only_missing_cases_in_this_reevaluation(self):
        for base in BASES:
            with self.subTest(base=base), fixture(base, inference=True) as (root, output, run, events):
                completed = self.complete_case(output, 300000, 1.5)
                metric = completed / "images_eval_openai.txt"
                original_metric = metric.read_bytes()
                training_output = output.with_name(output.name.removesuffix("_cfg_ref"))
                conditional = self.complete_case(training_output, 300000, 1.0)
                self.assert_success(run(PROMOE_RESUME="1"))
                samples = [row for row in events() if row["kind"] == "sample"]
                self.assertEqual([(row["step"], row["scales"]) for row in samples],
                                 [(300000, [1.0]), (500000, [1.0]), (500000, [1.5])])
                self.assertEqual(metric.read_bytes(), original_metric)
                self.assertEqual((completed / "images/keep.png").read_bytes(), b"existing image")
                self.assertEqual((conditional / "images_eval_openai.txt").read_bytes(), original_metric)
                self.assertTrue(all(row["mode"] == "cfg" and Path(row["output"]) == output for row in samples))
                # An incomplete/invalid case is retried without touching its completed siblings.
                invalid = output / "sample/step500000/img256_cfg1.0_seed0_FID50K_bs128_ema/images_eval_openai.txt"
                invalid.write_text("FID: nan\nInception Score: 100.0\n")
                before = len(events())
                self.assert_success(run(PROMOE_RESUME="1"))
                added = events()[before:]
                self.assertEqual([row["kind"] for row in added], ["sample", "eval"])
                self.assertEqual((added[0]["step"], added[0]["scales"]), (500000, [1.0]))
                self.assertEqual(metric.read_bytes(), original_metric)

    def test_all_pending_checkpoints_are_checked_before_sampling(self):
        for base in BASES:
            with self.subTest(base=base), fixture(base, inference=True) as (root, output, run, events):
                source = output.with_name(output.name.removesuffix("_cfg_ref")) / "checkpoints"
                missing = source / "ckpt_step_500000.pth"
                missing.unlink()
                result = run()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(str(missing), result.stderr)
                self.assertEqual(events(), [])
                missing.touch()
                self.assertNotEqual(run().returncode, 0)
                self.assertEqual(events(), [])
                # Completed cases do not need their source checkpoint to remain on disk.
                missing.write_bytes(b"original checkpoint 500000")
                (source / "ckpt_step_300000.pth").unlink()
                for scale in (1.0, 1.5):
                    self.complete_case(output, 300000, scale)
                self.assert_success(run(PROMOE_RESUME="1"))
                samples = [row for row in events() if row["kind"] == "sample"]
                self.assertEqual([row["step"] for row in samples], [500000, 500000])

    def test_all_cases_are_resampled_even_when_both_historical_modes_are_complete(self):
        for base in BASES:
            with self.subTest(base=base), fixture(base, inference=True) as (root, output, run, events):
                training_output = output.with_name(output.name.removesuffix("_cfg_ref"))
                old_cfg_output = output.with_name(output.name + "_300k")
                for step in (300000, 500000):
                    for scale in (1.0, 1.5):
                        self.complete_case(training_output, step, scale)
                        self.complete_case(old_cfg_output, step, scale, legacy=(step == 300000))
                old_files = {p: p.read_bytes() for directory in (training_output, old_cfg_output)
                             for p in directory.rglob("*") if p.is_file()}
                self.assert_success(run())
                samples = [row for row in events() if row["kind"] == "sample"]
                self.assertEqual([(row["step"], row["scales"]) for row in samples],
                                 [(300000, [1.0]), (300000, [1.5]), (500000, [1.0]), (500000, [1.5])])
                self.assertTrue(all(row["mode"] == "cfg" and Path(row["output"]) == output for row in samples))
                self.assertEqual({p: p.read_bytes() for directory in (training_output, old_cfg_output)
                                  for p in directory.rglob("*") if p.is_file()}, old_files)

    def test_inference_failures_are_reported(self):
        for overrides in ({"BAD_METRICS": "1"}, {"PROMOE_GPU_IDS_OVERRIDE": "0,1"}, {"PROMOE_RESUME": "2"}):
            with self.subTest(overrides=overrides), fixture(BASES[0], inference=True) as (root, output, run, events):
                self.assertNotEqual(run(**overrides).returncode, 0)
                self.assertLessEqual(len(events()), 2)


if __name__ == "__main__":
    unittest.main()
