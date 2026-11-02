"""Canonical dual-CFG FID boundaries, using only temporary metric files."""
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/_eval_metric_helpers.sh"


class BaselineGateTests(unittest.TestCase):
    def write_result(self, base, scale, fid, seed=0):
        folder = base / "step300000" / f"img256_cfg{scale}_seed{seed}_FID50K_bs128_ema_corr_ref_cfg"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "images_eval_openai.txt").write_text(f"FID: {fid}\nInception Score: 100.0\n")

    def run_gate(self, base):
        return subprocess.run(
            ["bash", "-c", 'source "$1"; promoe_300k_fid_gate "$2"', "bash", str(HELPER), str(base)],
            text=True, capture_output=True, timeout=10,
        )

    def test_both_fids_must_be_strictly_lower_than_the_baseline(self):
        cases = (("30.85", "9.72", 0), ("0", "0", 0),
                 ("30.86", "9.0", 1), ("30.0", "9.73", 1),
                 ("31.0", "9.0", 1), ("30.0", "10.0", 1), ("100", "100", 1))
        for fid1, fid15, code in cases:
            with self.subTest(fid1=fid1, fid15=fid15), tempfile.TemporaryDirectory(
                dir=ROOT / "test", prefix=".fid_gate_"
            ) as directory:
                base = Path(directory)
                self.write_result(base, "1.0", fid1)
                self.write_result(base, "1.5", fid15)
                result = self.run_gate(base)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertIn("PASS" if code == 0 else "STOP", result.stdout)
                self.assertIn("CFG=1.0", result.stdout)
                self.assertIn("CFG=1.5", result.stdout)

    def test_missing_invalid_or_ambiguous_results_are_errors(self):
        for case in ("missing", "nan", "-1", "1e309", "duplicate"):
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                dir=ROOT / "test", prefix=".fid_gate_"
            ) as directory:
                base = Path(directory)
                self.write_result(base, "1.0", "30.0")
                if case != "missing":
                    self.write_result(base, "1.5", "9.0" if case == "duplicate" else case)
                if case == "duplicate":
                    self.write_result(base, "1.5", "8.0", seed=1)
                result = self.run_gate(base)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("ERROR", result.stderr)
                self.assertNotIn("gate PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
