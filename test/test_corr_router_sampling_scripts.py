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
def fixture(base, legacy=False):
    with tempfile.TemporaryDirectory(dir=REPO / "test", prefix=".corr_shell_") as directory:
        root = Path(directory)
        tag = base + ("_cfg_ref_300k" if legacy else "_same_noise")
        script = f"scripts/corr_router/run_B_{tag}_{'sample_eval' if legacy else 'train_sample_eval'}.sh"
        paths = [script, "scripts/_eval_metric_helpers.sh", "scripts/corr_router/_dual_reference_eval.sh",
                 f"configs/004_ProMoE_B_{tag}.yaml"]
        if not legacy:
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
output = Path(c.get("output_dir", "outputs")) / c["model_name"] / Path(a.config).stem
source = Path(c["sample_checkpoint_dir"]) / "ckpt_step_300000.pth"
assert source.is_file() and source.stat().st_size
record("sample", mode=mode, scales=scales, source=str(source.resolve()), output=str(output.resolve()), seed=c["global_seed"])
if os.getenv("FAIL_SAMPLE_MODE") == mode:
    raise SystemExit(17)
c["corr_reference_velocity"] = mode
for scale in scales:
    folder = output / "sample" / "step300000" / sampling_folder_name(SimpleNamespace(**c), scale) / "images"
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
        if legacy:
            checkpoint = root / cfg["sample_checkpoint_dir"] / "ckpt_step_300000.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"original checkpoint")
        env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                   PROMOE_GPU_IDS_OVERRIDE="4,5,6,7", PROMOE_LATENT_PATH="/input/latents", PROMOE_RESUME="0",
                   PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""), TMPDIR=str(root))

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

    def test_existing_inference_launchers_use_standard_layout_and_original_weights(self):
        for base in BASES:
            with self.subTest(base=base), fixture(base, legacy=True) as (root, output, run, events):
                self.assert_success(run())
                log = events()
                self.assertEqual([row["kind"] for row in log], ["sample", "eval"])
                self.assertEqual(log[0]["scales"], [1.5])
                self.assertEqual(Path(log[0]["source"]).read_bytes(), b"original checkpoint")
                self.assertNotIn("_corr_ref_cfg", log[1]["images"])
                self.assert_success(run(PROMOE_RESUME="1"))
                self.assertEqual(events(), log)

    def test_legacy_layout_is_moved_intact_on_resume_without_resampling(self):
        for base in BASES:
            with self.subTest(base=base), fixture(base, legacy=True) as (root, output, run, events):
                parent = output / "sample/step300000"
                standard = parent / "img256_cfg1.5_seed0_FID50K_bs128_ema"
                old = Path(str(standard) + "_corr_ref_cfg")
                (old / "images").mkdir(parents=True)
                (old / "images/keep.png").write_bytes(b"existing image")
                (old / "images_eval_openai.txt").write_text("FID: 9.0\nInception Score: 100.0\n")
                self.assert_success(run(PROMOE_RESUME="1"))
                self.assertEqual(events(), [])
                self.assertFalse(old.exists())
                self.assertEqual((standard / "images/keep.png").read_bytes(), b"existing image")


if __name__ == "__main__":
    unittest.main()
