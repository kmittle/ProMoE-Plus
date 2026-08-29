"""Tests for deterministic rank-local support selection."""

import tempfile
import unittest
from pathlib import Path

from analyses.affinity_responsibility.support import select_support_cases


class SupportSelectionTest(unittest.TestCase):
    def _make_dataset(self, root, classes=12, latents=3):
        for label in range(classes):
            class_dir = root / f"n{label:08d}"
            class_dir.mkdir()
            for index in range(latents):
                (class_dir / f"sample_{index}.latent.npz").touch()

    def test_selection_is_balanced_deterministic_and_query_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._make_dataset(root)
            first = select_support_cases(
                root,
                excluded_labels={1, 5},
                expected_class_count=12,
                group_count=2,
                batch_size=4,
                salt="unit-test",
            )
            second = select_support_cases(
                root,
                excluded_labels={1, 5},
                expected_class_count=12,
                group_count=2,
                batch_size=4,
                salt="unit-test",
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first), 8)
            self.assertFalse({case["label"] for case in first} & {1, 5})
            self.assertEqual(len({case["label"] for case in first}), 8)
            self.assertEqual(
                [sum(case["group_index"] == group for case in first) for group in range(2)],
                [4, 4],
            )
            self.assertEqual(
                sum(case["unconditional"] for case in first),
                8,
            )
            self.assertTrue(all(0.0 < case["sigma"] < 1.0 for case in first))
            self.assertEqual(len({case["sigma_seed"] for case in first}), len(first))

    def test_wrong_class_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._make_dataset(root, classes=3)
            with self.assertRaisesRegex(ValueError, "Expected 4"):
                select_support_cases(
                    root,
                    excluded_labels=(),
                    expected_class_count=4,
                    group_count=1,
                    batch_size=2,
                )


if __name__ == "__main__":
    unittest.main()
