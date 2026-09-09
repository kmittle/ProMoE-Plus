from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from credit_redistribution.statistics import (
    BLOCK_COUNT,
    CASE_COUNT,
    EXPERT_COUNT,
    _json_safe,
    _relative_cv_increase,
    bootstrap_metrics,
    gini,
    materialize_bootstrap_indices,
    point_metrics,
)


def _data(rate=None):
    if rate is None:
        rate = np.ones(EXPERT_COUNT, dtype=np.float64)
    count = np.full(
        (CASE_COUNT, BLOCK_COUNT, EXPERT_COUNT), 3, dtype=np.int64
    )
    credit = count.astype(np.float64) * np.asarray(rate)[None, None, :]
    mse = np.linspace(1.0, 2.0, CASE_COUNT, dtype=np.float64)
    return {"mse": mse, "credit": credit, "count": count}


class StatisticsTest(unittest.TestCase):
    def test_locked_gini_formula(self):
        self.assertEqual(gini(np.ones(EXPERT_COUNT)), 0.0)
        concentrated = np.zeros(EXPERT_COUNT, dtype=np.float64)
        concentrated[-1] = 1.0
        self.assertAlmostEqual(gini(concentrated), 11.0 / 12.0)

    def test_point_and_bootstrap_recompute_aggregate_rates(self):
        rates = np.arange(1, EXPERT_COUNT + 1, dtype=np.float64)
        data = _data(rates)
        point = point_metrics(data)
        self.assertAlmostEqual(point["mse"], data["mse"].mean())
        self.assertAlmostEqual(point["gini"], gini(rates))
        self.assertEqual(point["cv"], 0.0)
        indices = np.tile(
            np.arange(CASE_COUNT, dtype=np.int64), (3, 1)
        )
        distributions = bootstrap_metrics(data, indices, chunk_size=2)
        for name, expected in point.items():
            self.assertTrue(np.array_equal(
                distributions[name], np.full(3, expected)
            ))

    def test_bootstrap_indices_follow_locked_chunked_pcg64_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "indices.npy"
            materialize_bootstrap_indices(
                path,
                resamples=13,
                case_count=5,
                chunk_size=4,
                seed=29,
            )
            observed = np.load(path, allow_pickle=False)
            generator = np.random.Generator(np.random.PCG64(29))
            expected_chunks = [
                generator.integers(
                    0, 5, size=(size, 5), dtype=np.int64, endpoint=False
                )
                for size in (4, 4, 4, 1)
            ]
            self.assertTrue(np.array_equal(observed, np.concatenate(expected_chunks)))

    def test_existing_bootstrap_indices_are_recomputed_not_only_sidecar_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "indices.npy"
            materialize_bootstrap_indices(
                path,
                resamples=13,
                case_count=5,
                chunk_size=4,
                seed=29,
            )
            matrix = np.load(path, allow_pickle=False)
            matrix = np.array(matrix, copy=True)
            matrix[0, 0] = (int(matrix[0, 0]) + 1) % 5
            # Materialization seals the matrix read-only; make the explicit
            # tampering simulation writable before replacing its bytes.
            path.chmod(0o644)
            np.save(path, matrix, allow_pickle=False)
            from credit_redistribution.serialization import sha256_file

            path.with_suffix(path.suffix + ".sha256").chmod(0o644)
            path.with_suffix(path.suffix + ".sha256").write_text(
                sha256_file(path) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "locked PCG64 stream"):
                materialize_bootstrap_indices(
                    path,
                    resamples=13,
                    case_count=5,
                    chunk_size=4,
                    seed=29,
                )

    def test_zero_reference_cv_rule_is_explicitly_serializable(self):
        result = _relative_cv_increase(
            np.asarray([0.0, 0.0]), np.asarray([0.0, 0.1])
        )
        self.assertEqual(result[0], 0.0)
        self.assertTrue(np.isposinf(result[1]))
        self.assertEqual(_json_safe(float(result[1])), "+infinity")


if __name__ == "__main__":
    unittest.main()
