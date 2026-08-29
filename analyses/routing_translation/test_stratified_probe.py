import unittest

import torch

from analyses.routing_translation.stratified_probe import (
    STRATUM_NAMES,
    _build_stratum_masks,
    _build_stratum_routes,
    _four_neighbor_pair_histograms,
    _redact_unavailable_spatial_metrics,
    _spatially_matched_routes,
)
from analyses.routing_translation.probe import _random_matched_routes


class RoutingTranslationStratifiedProbeTests(unittest.TestCase):
    def setUp(self):
        self.scores = torch.tensor([
            [0.90, 0.80, 0.10, 0.00],
            [0.70, 0.69, 0.20, 0.10],
            [0.60, 0.20, 0.55, 0.10],
            [0.80, 0.30, 0.20, 0.10],
            [0.51, 0.50, 0.40, 0.20],
            [0.75, 0.10, 0.60, 0.20],
            [0.62, 0.61, 0.40, 0.20],
            [0.90, 0.30, 0.20, 0.80],
            [0.80, 0.70, 0.60, 0.10],
        ])
        self.native = torch.zeros(9, dtype=torch.long)
        self.content = torch.tensor([1, 2, 2, 0, 1, 3, 1, 3, 2])
        self.valid = torch.ones(9, dtype=torch.bool)

    def test_margin_and_rank_strata_are_complete_partitions(self):
        masks, diagnostics = _build_stratum_masks(
            self.scores,
            self.native,
            self.content,
            self.valid,
        )
        changed = self.native != self.content
        self.assertTrue(torch.equal(
            masks["low_margin"] | masks["high_margin"],
            changed,
        ))
        self.assertFalse(
            (masks["low_margin"] & masks["high_margin"]).any()
        )
        self.assertEqual(masks["low_margin"].sum().item(), 4)
        self.assertEqual(masks["high_margin"].sum().item(), 4)
        low_margins = diagnostics["top1_margin"][masks["low_margin"]]
        high_margins = diagnostics["top1_margin"][masks["high_margin"]]
        self.assertLessEqual(low_margins.max().item(), high_margins.min().item())
        self.assertTrue(torch.equal(
            masks["content_top2"] | masks["content_rank3plus"],
            changed,
        ))
        self.assertFalse(
            (masks["content_top2"] & masks["content_rank3plus"]).any()
        )

    def test_each_random_control_matches_its_stratum(self):
        masks, _ = _build_stratum_masks(
            self.scores,
            self.native,
            self.content,
            self.valid,
        )
        routes, controls = _build_stratum_routes(
            self.native,
            self.content,
            self.valid,
            masks,
            torch.Generator().manual_seed(19),
            num_routed_experts=4,
            grid_size=3,
        )
        self.assertEqual(len(routes), 3 * len(STRATUM_NAMES))
        for index, name in enumerate(STRATUM_NAMES):
            content_route = routes[3 * index]
            spatial_route = routes[3 * index + 1]
            random_route = routes[3 * index + 2]
            mask = masks[name]
            self.assertTrue(torch.equal(content_route != self.native, mask))
            self.assertTrue(torch.equal(spatial_route != self.native, mask))
            self.assertTrue(torch.equal(random_route != self.native, mask))
            self.assertTrue(torch.equal(
                torch.bincount(content_route[mask], minlength=4),
                torch.bincount(random_route[mask], minlength=4),
            ))
            self.assertTrue(torch.equal(
                torch.bincount(content_route[mask], minlength=4),
                torch.bincount(spatial_route[mask], minlength=4),
            ))
            self.assertTrue(controls[name]["support_equal"])
            self.assertTrue(controls[name]["replacement_histogram_equal"])
            self.assertEqual(
                controls[name]["random_control_available"],
                bool((random_route[mask] != content_route[mask]).any()),
            )

    def test_four_neighbor_histogram_uses_incident_edges(self):
        route = torch.tensor([0, 0, 1, 1])
        mask = torch.ones(4, dtype=torch.bool)
        histogram = _four_neighbor_pair_histograms(
            route,
            mask,
            grid_size=2,
            num_routed_experts=2,
        )
        expected = torch.tensor([0.25, 0.50, 0.0, 0.25], dtype=torch.float64)
        torch.testing.assert_close(histogram, expected)

    def test_default_spatial_control_is_deterministic_and_matches_structure(self):
        native = torch.full((16,), 2, dtype=torch.long)
        content = torch.tensor([[0, 0, 1, 1]] * 4).flatten()
        mask = torch.ones(16, dtype=torch.bool)
        random_route, random_changed, available = _random_matched_routes(
            native,
            content,
            mask,
            torch.Generator().manual_seed(1002),
        )
        self.assertTrue(available)
        self.assertTrue(torch.equal(random_changed, mask))
        spatial_route, diagnostics = _spatially_matched_routes(
            native_ids=native,
            content_ids=content,
            changed_mask=mask,
            random_ids=random_route,
            generator=torch.Generator().manual_seed(2002),
            num_routed_experts=3,
            grid_size=4,
        )
        repeated_route, repeated_diagnostics = _spatially_matched_routes(
            native_ids=native,
            content_ids=content,
            changed_mask=mask,
            random_ids=random_route,
            generator=torch.Generator().manual_seed(2002),
            num_routed_experts=3,
            grid_size=4,
        )
        self.assertTrue(diagnostics["spatial_control_available"])
        self.assertTrue(torch.equal(spatial_route, repeated_route))
        self.assertEqual(diagnostics, repeated_diagnostics)
        self.assertTrue(torch.equal(spatial_route != native, mask))
        self.assertTrue(torch.equal(
            torch.bincount(spatial_route, minlength=3),
            torch.bincount(content, minlength=3),
        ))
        self.assertGreaterEqual(
            diagnostics["spatial_differs_from_content_rate"],
            0.5,
        )
        self.assertLessEqual(diagnostics["spatial_adjacency_tv"], 0.1)
        self.assertLessEqual(
            diagnostics["spatial_adjacency_tv"],
            diagnostics["random_adjacency_tv"] + 1e-12,
        )
        self.assertEqual(diagnostics["spatial_evaluated_candidates"], 256)
        self.assertLessEqual(diagnostics["spatial_unique_candidates"], 256)
        self.assertGreater(diagnostics["spatial_deranged_candidates"], 0)
        self.assertIsNotNone(
            diagnostics["spatial_best_deranged_adjacency_tv"]
        )
        self.assertEqual(
            sum(diagnostics["spatial_rejection_counts"].values())
            + diagnostics["spatial_eligible_candidates"],
            diagnostics["spatial_evaluated_candidates"],
        )
        self.assertIsNone(diagnostics["spatial_unavailable_reason"])

    def test_spatial_control_marks_single_changed_token_unavailable(self):
        native = torch.zeros(4, dtype=torch.long)
        content = torch.tensor([1, 0, 0, 0])
        changed = native != content
        spatial_route, diagnostics = _spatially_matched_routes(
            native_ids=native,
            content_ids=content,
            changed_mask=changed,
            random_ids=content,
            generator=torch.Generator().manual_seed(3),
            num_routed_experts=2,
            grid_size=2,
        )
        self.assertFalse(diagnostics["spatial_control_available"])
        self.assertTrue(torch.equal(spatial_route, content))
        self.assertEqual(diagnostics["spatial_evaluated_candidates"], 0)
        self.assertEqual(
            diagnostics["spatial_unavailable_reason"],
            "fewer_than_two_changed_tokens",
        )

    def test_unavailable_spatial_control_never_exposes_random_metrics(self):
        native = torch.tensor([0, 0, 1, 2])
        content = torch.tensor([1, 2, 1, 2])
        random_route = torch.tensor([2, 1, 1, 2])
        changed = native != content
        self.assertFalse(torch.equal(random_route, content))
        spatial_route, diagnostics = _spatially_matched_routes(
            native_ids=native,
            content_ids=content,
            changed_mask=changed,
            random_ids=random_route,
            generator=torch.Generator().manual_seed(13),
            num_routed_experts=3,
            grid_size=2,
            candidate_count=1,
            min_derangement=1.0,
            max_adjacency_tv=0.0,
        )
        self.assertFalse(diagnostics["spatial_control_available"])
        self.assertTrue(torch.equal(spatial_route, content))
        self.assertEqual(diagnostics["spatial_evaluated_candidates"], 1)
        self.assertEqual(
            sum(diagnostics["spatial_rejection_counts"].values())
            + diagnostics["spatial_eligible_candidates"],
            1,
        )
        self.assertEqual(
            diagnostics["spatial_unavailable_reason"],
            "no_candidate_met_all_constraints",
        )
        if diagnostics["spatial_deranged_candidates"]:
            self.assertIsNotNone(
                diagnostics["spatial_best_deranged_adjacency_tv"]
            )
            self.assertAlmostEqual(
                diagnostics["spatial_best_deranged_tv_minus_random"],
                diagnostics["spatial_best_deranged_adjacency_tv"]
                - diagnostics["random_adjacency_tv"],
            )

        controls = {
            name: {"spatial_control_available": name != "high_margin"}
            for name in STRATUM_NAMES
        }
        metrics = {
            f"{name}_spatial": float(index)
            for index, name in enumerate(STRATUM_NAMES)
        }
        redacted = _redact_unavailable_spatial_metrics(metrics, controls)
        self.assertIsNone(redacted["high_margin_spatial"])
        self.assertEqual(redacted["low_margin_spatial"], 0.0)
        self.assertIsNot(redacted, metrics)

    def test_native_ids_must_match_router_argmax(self):
        bad_native = self.native.clone()
        bad_native[0] = 1
        with self.assertRaisesRegex(ValueError, "argmax"):
            _build_stratum_masks(
                self.scores,
                bad_native,
                self.content,
                self.valid,
            )


if __name__ == "__main__":
    unittest.main()
