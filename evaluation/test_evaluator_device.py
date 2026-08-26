import unittest

from evaluation.evaluator import (
    _parse_compute_capability,
    _select_evaluation_device,
)


class EvaluatorDeviceTest(unittest.TestCase):
    def test_compute_capability_parser(self):
        self.assertEqual(_parse_compute_capability("sm_90"), (9, 0))
        self.assertEqual(_parse_compute_capability("compute_100"), (10, 0))
        self.assertIsNone(_parse_compute_capability("cpu"))

    def test_auto_falls_back_for_newer_gpu(self):
        selected, reason = _select_evaluation_device(
            "auto",
            [(9, 0)],
            ["sm_75", "compute_80"],
        )
        self.assertEqual(selected, "cpu")
        self.assertIn("absent", reason)

    def test_auto_falls_back_for_unlisted_lower_gpu(self):
        selected, reason = _select_evaluation_device(
            "auto",
            [(6, 1)],
            ["sm_60", "compute_80"],
        )
        self.assertEqual(selected, "cpu")
        self.assertIn("absent", reason)

    def test_auto_uses_covered_gpu(self):
        selected, _ = _select_evaluation_device(
            "auto",
            [(9, 0)],
            ["sm_89", "compute_90"],
        )
        self.assertEqual(selected, "gpu")

    def test_auto_falls_back_when_build_capability_is_unknown(self):
        selected, reason = _select_evaluation_device("auto", [(9, 0)], [])
        self.assertEqual(selected, "cpu")
        self.assertIn("build capability is unknown", reason)

    def test_auto_falls_back_when_gpu_capability_is_unknown(self):
        selected, reason = _select_evaluation_device(
            "auto",
            [None],
            ["compute_90"],
        )
        self.assertEqual(selected, "cpu")
        self.assertIn("GPU compute capability is unknown", reason)

    def test_explicit_gpu_requires_visible_device(self):
        with self.assertRaisesRegex(RuntimeError, "no GPU is visible"):
            _select_evaluation_device("gpu", [], ["compute_90"])


if __name__ == "__main__":
    unittest.main()
