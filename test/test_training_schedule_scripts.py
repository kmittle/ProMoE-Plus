"""Check phase ordering and failed-evaluation recovery with CPU-only stubs."""
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


REPO = Path(__file__).resolve().parents[1]
ARMS = (
    ("base", "top3"),
    ("clean_route", "clean_route_student_ratio0_align_delay100k"),
    ("clean_route", "clean_route_student_ratio0_align_warmup200k"),
)

STUB = '''import argparse, json, os
from pathlib import Path
import yaml
root = Path(os.environ["PIPELINE_TEST_ROOT"])
kind = Path(__file__).stem
parser = argparse.ArgumentParser()
if kind == "run_eval":
    parser.add_argument("images"); parser.add_argument("--count", type=int)
else:
    parser.add_argument("--config"); parser.add_argument("--step_list_for_sample", type=int)
args = parser.parse_args()
if kind == "run_eval":
    images = Path(args.images)
    step = int(images.parent.parent.name.removeprefix("step"))
    scale = float(images.parent.name.split("_cfg")[1].split("_")[0])
    event = dict(kind="eval", step=step, scale=scale, count=args.count)
    fid = "nan" if os.getenv("BAD_METRICS") == "1" and scale == 1.5 else "100.0"
    Path(str(images) + "_eval_openai.txt").write_text("FID: " + fid + "\\nInception Score: 100.0\\n")
else:
    cfg = yaml.safe_load(Path(args.config).read_text())
    output = root / "outputs" / cfg["model_name"] / Path(args.config).stem
    ckpts = output / "checkpoints"
    if kind == "train":
        step = cfg["num_steps"] - 1
        previous = sorted(int(p.stem.removeprefix("ckpt_step_")) for p in ckpts.glob("*.pth"))
        assert cfg["resume_checkpoint"] is True
        assert cfg["gpu_ids"] == [4, 5, 6, 7]
        event = dict(kind="train", step=step, previous=previous)
        ckpts.mkdir(parents=True, exist_ok=True)
        (ckpts / f"ckpt_step_{step}.pth").write_bytes(b"stub checkpoint")
    else:
        step = args.step_list_for_sample
        assert (ckpts / f"ckpt_step_{step}.pth").is_file()
        event = dict(kind="sample", step=step)
        for scale in cfg["guide_scale_list"]:
            if os.getenv("MISSING_IMAGES") == "1" and scale == 1.5:
                continue
            folder = f"img256_cfg{scale}_seed0_FID50K_bs128_ema"
            images = output / "sample" / f"step{step}" / folder / "images"
            images.mkdir(parents=True, exist_ok=True)
            (images / "stub.png").write_bytes(b"stub image")
with (root / "events.jsonl").open("a") as file:
    file.write(json.dumps(event) + "\\n")
'''


@contextlib.contextmanager
def pipeline(arm):
    family, tag = arm
    with tempfile.TemporaryDirectory(dir=REPO / "test", prefix=".train_schedule_") as directory:
        root = Path(directory)
        script = f"scripts/{family}/run_B_{tag}_train_sample_eval.sh"
        for path in (script, f"configs/004_ProMoE_B_{tag}.yaml", "scripts/_eval_metric_helpers.sh"):
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
        for path in ("train.py", "sample.py", "evaluation/run_eval.py"):
            (root / path).write_text(STUB)
        env = dict(os.environ, PIPELINE_TEST_ROOT=str(root), PROMOE_RESUME="0",
                   PROMOE_GPU_IDS_OVERRIDE="4,5,6,7", TMPDIR=str(root),
                   BAD_METRICS="0", MISSING_IMAGES="0",
                   PATH=str(root / "bin") + os.pathsep + os.environ["PATH"])

        def run(**overrides):
            return subprocess.run(["bash", script], cwd=root, env=dict(env, **overrides),
                                  text=True, capture_output=True, timeout=30)

        def events():
            path = root / "events.jsonl"
            return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

        yield run, events


class TrainingScheduleScriptTests(unittest.TestCase):
    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_each_arm_evaluates_before_resuming_its_own_300k_checkpoint(self):
        for arm in ARMS:
            with self.subTest(arm=arm), pipeline(arm) as (run, events):
                self.assert_success(run())
                log = events()
                self.assertEqual([(e["kind"], e["step"]) for e in log],
                                 [(kind, step) for step in (300000, 500000)
                                  for kind in ("train", "sample", "eval", "eval")])
                self.assertEqual(log[0]["previous"], [])
                self.assertEqual(log[4]["previous"], [300000])
                self.assertEqual([(e["scale"], e["count"]) for e in log if e["kind"] == "eval"],
                                 [(1.0, 50000), (1.5, 50000)] * 2)
                # FID 100 is valid: these experiments have no FID threshold gate.
                self.assert_success(run(PROMOE_RESUME="1"))
                self.assertEqual(events(), log)

    def test_missing_or_invalid_300k_result_stops_then_resume_finishes_evaluation_first(self):
        for flag in ("BAD_METRICS", "MISSING_IMAGES"):
            with self.subTest(flag=flag), pipeline(ARMS[0]) as (run, events):
                result = run(**{flag: "1"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("missing or invalid FID/IS", result.stderr)
                self.assertTrue(all(e["step"] == 300000 for e in events()))
                before = len(events())
                self.assert_success(run(PROMOE_RESUME="1"))
                added = events()[before:]
                self.assertEqual([(e["kind"], e["step"]) for e in added],
                                 [("sample", 300000), ("eval", 300000), ("eval", 300000),
                                  ("train", 500000), ("sample", 500000),
                                  ("eval", 500000), ("eval", 500000)])
                self.assertEqual(added[3]["previous"], [300000])


if __name__ == "__main__":
    unittest.main()
