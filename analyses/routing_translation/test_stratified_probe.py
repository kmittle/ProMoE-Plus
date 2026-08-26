import unittest

import torch

from analyses.routing_translation.stratified_probe import (
    STRATUM_NAMES,
    _build_stratum_masks,
    _build_stratum_routes,
)


class RoutingTranslationStratifiedProbeTests(unittest.TestCase):
    def setUp(self):
        self.scores = torch.tensor([
            [0.90, 0.80, 0.10, 0.00],
            [0.70, 0.69, 0.20, 0.10],
            [0.60, 0.20, 0.55, 0.10],
            [0.80, 0.30, 0.20, 0.10],
            [0.51, 0.50, 0.40, 0.20],
            [0.75, 0.10, 0.60, 0.20],
        ])
        self.native = torch.zeros(6, dtype=torch.long)
        self.content = torch.tensor([1, 2, 2, 0, 1, 3])
        self.valid = torch.ones(6, dtype=torch.bool)

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
        self.assertEqual(masks["low_margin"].sum().item(), 3)
        self.assertEqual(masks["high_margin"].sum().item(), 2)
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
        )
        self.assertEqual(len(routes), 2 * len(STRATUM_NAMES))
        for index, name in enumerate(STRATUM_NAMES):
            content_route = routes[2 * index]
            random_route = routes[2 * index + 1]
            mask = masks[name]
            self.assertTrue(torch.equal(content_route != self.native, mask))
            self.assertTrue(torch.equal(random_route != self.native, mask))
            self.assertTrue(torch.equal(
                torch.bincount(content_route[mask], minlength=4),
                torch.bincount(random_route[mask], minlength=4),
            ))
            self.assertTrue(controls[name]["support_equal"])
            self.assertTrue(controls[name]["replacement_histogram_equal"])
            self.assertEqual(
                controls[name]["random_control_available"],
                bool((random_route[mask] != content_route[mask]).any()),
            )

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
