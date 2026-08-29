import itertools
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from analyses.routing_translation.spatial_control import (
    exact_spatial_control_diagnostic,
)


def _two_by_two_pair_counts(route_ids, changed_mask, num_routed_experts):
    counts = np.zeros(num_routed_experts * num_routed_experts, dtype=np.int64)
    edges = ((0, 1), (2, 3), (0, 2), (1, 3))
    active_edges = [
        (left, right)
        for left, right in edges
        if changed_mask[left] or changed_mask[right]
    ]
    for left, right in active_edges:
        lower = min(int(route_ids[left]), int(route_ids[right]))
        upper = max(int(route_ids[left]), int(route_ids[right]))
        counts[lower * num_routed_experts + upper] += 1
    return counts, active_edges


def _brute_force_optimum(
    native_ids,
    content_ids,
    changed_mask,
    grid_size,
    num_routed_experts,
    minimum_mismatches,
):
    changed_indices = np.flatnonzero(changed_mask)
    content_changed = content_ids[changed_indices]
    if grid_size != 2:
        raise ValueError("The independent brute-force oracle is fixed to 2x2")
    reference_counts, edges = _two_by_two_pair_counts(
        content_ids,
        changed_mask,
        num_routed_experts,
    )
    maximum_mismatches = -1
    minimum_tv = None
    for assignment in set(itertools.permutations(content_changed.tolist())):
        assignment = np.asarray(assignment, dtype=np.int64)
        if np.any(assignment == native_ids[changed_indices]):
            continue
        mismatches = int((assignment != content_changed).sum())
        maximum_mismatches = max(maximum_mismatches, mismatches)
        if mismatches < minimum_mismatches:
            continue
        candidate = native_ids.copy()
        candidate[changed_indices] = assignment
        candidate_counts, _ = _two_by_two_pair_counts(
            candidate,
            changed_mask,
            num_routed_experts,
        )
        tv = 0.5 * np.abs(candidate_counts - reference_counts).sum() / len(edges)
        minimum_tv = tv if minimum_tv is None else min(minimum_tv, tv)
    return maximum_mismatches, minimum_tv


class ExactSpatialControlTests(unittest.TestCase):
    def _assert_matches_brute_force(
        self,
        native_ids,
        content_ids,
        minimum_mismatches,
        num_routed_experts=3,
    ):
        native_ids = np.asarray(native_ids, dtype=np.int64)
        content_ids = np.asarray(content_ids, dtype=np.int64)
        changed_mask = native_ids != content_ids
        brute_maximum, brute_minimum_tv = _brute_force_optimum(
            native_ids,
            content_ids,
            changed_mask,
            grid_size=2,
            num_routed_experts=num_routed_experts,
            minimum_mismatches=minimum_mismatches,
        )
        candidate, diagnostics = exact_spatial_control_diagnostic(
            native_ids=native_ids,
            content_ids=content_ids,
            changed_mask=changed_mask,
            grid_size=2,
            num_routed_experts=num_routed_experts,
            minimum_mismatches=minimum_mismatches,
            max_adjacency_tv=1.0,
            random_adjacency_tv=1.0,
        )
        self.assertEqual(diagnostics["maximum_mismatches"], brute_maximum)
        self.assertEqual(
            diagnostics["derangement_feasible"],
            brute_maximum >= minimum_mismatches,
        )
        if brute_minimum_tv is None:
            self.assertIsNone(candidate)
            self.assertIsNone(diagnostics["minimum_adjacency_tv"])
            self.assertIsNone(diagnostics["milp"])
            return

        self.assertIsNotNone(candidate)
        self.assertTrue(diagnostics["milp"]["proven_optimal"])
        self.assertAlmostEqual(
            diagnostics["minimum_adjacency_tv"],
            brute_minimum_tv,
        )
        self.assertTrue(np.array_equal(candidate != native_ids, changed_mask))
        self.assertTrue(np.array_equal(
            np.bincount(candidate[changed_mask], minlength=num_routed_experts),
            np.bincount(content_ids[changed_mask], minlength=num_routed_experts),
        ))
        self.assertGreaterEqual(
            int((candidate[changed_mask] != content_ids[changed_mask]).sum()),
            minimum_mismatches,
        )

    def test_milp_matches_exhaustive_two_by_two_optimum(self):
        self._assert_matches_brute_force(
            native_ids=[2, 2, 2, 2],
            content_ids=[0, 0, 1, 1],
            minimum_mismatches=2,
        )
        self._assert_matches_brute_force(
            native_ids=[2, 2, 0, 1],
            content_ids=[0, 1, 2, 2],
            minimum_mismatches=2,
        )
        native = np.asarray([0, 0, 0, 0], dtype=np.int64)
        content = np.asarray([0, 1, 1, 2], dtype=np.int64)
        brute_maximum, brute_minimum_tv = _brute_force_optimum(
            native,
            content,
            native != content,
            grid_size=2,
            num_routed_experts=3,
            minimum_mismatches=2,
        )
        self.assertEqual(brute_maximum, 2)
        self.assertEqual(brute_minimum_tv, 0.5)
        self._assert_matches_brute_force(
            native_ids=native,
            content_ids=content,
            minimum_mismatches=2,
        )

    def test_assignment_proves_derangement_is_structurally_impossible(self):
        self._assert_matches_brute_force(
            native_ids=[0, 0, 1, 1],
            content_ids=[1, 1, 0, 0],
            minimum_mismatches=2,
            num_routed_experts=2,
        )

    @patch("analyses.routing_translation.spatial_control.milp")
    def test_nonoptimal_milp_is_reported_as_unknown(self, mock_milp):
        mock_milp.return_value = OptimizeResult(
            status=1,
            success=False,
            message="Time limit reached",
        )
        native = np.asarray([2, 2, 2, 2], dtype=np.int64)
        content = np.asarray([0, 0, 1, 1], dtype=np.int64)
        candidate, diagnostics = exact_spatial_control_diagnostic(
            native_ids=native,
            content_ids=content,
            changed_mask=native != content,
            grid_size=2,
            num_routed_experts=3,
            minimum_mismatches=2,
            max_adjacency_tv=0.1,
            random_adjacency_tv=0.5,
            time_limit_seconds=0.25,
        )
        self.assertIsNone(candidate)
        self.assertEqual(diagnostics["milp"]["status"], "limit_reached")
        self.assertFalse(diagnostics["milp"]["proven_optimal"])
        self.assertIsNone(diagnostics["minimum_adjacency_tv"])
        self.assertIsNone(diagnostics["meets_maximum_adjacency_tv"])
        self.assertIsNone(diagnostics["not_worse_than_random"])
        self.assertIsNone(
            diagnostics["all_acceptance_constraints_feasible"]
        )
        self.assertEqual(
            mock_milp.call_args.kwargs["options"]["time_limit"],
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
