import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from analyses.timestep_utility.credit_balance_cross_checkpoint_probe import (
    _relative_mse_drift,
    _route_controls,
    _validate_lossfree_bias_contract,
)


class _ToyRouter(nn.Module):
    def __init__(self, use_lossfree_bias):
        super().__init__()
        self.hidden_size = 3
        self.num_routed_experts = 3
        self.top_k = 1
        self.router_weight_mode = "identity"
        self.use_lossfree_bias = use_lossfree_bias
        self.cluster_centers = nn.Parameter(torch.eye(3))
        self.register_buffer(
            "expert_bias",
            torch.tensor([-1.0, 1.0, 0.0]),
        )

    def compute_router(self, hidden_states, labels):
        del labels
        scores = (
            F.normalize(hidden_states, dim=-1)
            @ F.normalize(self.cluster_centers, dim=-1).T
        )
        selection = scores
        if self.use_lossfree_bias:
            selection = selection + self.expert_bias.view(1, 1, -1)
        indices = selection.argmax(dim=-1, keepdim=True)
        weights = torch.gather(scores, dim=-1, index=indices)
        return weights, indices, None


class CrossCheckpointProbeTests(unittest.TestCase):
    def setUp(self):
        row = torch.tensor([[1.0, 0.9, 0.0], [1.0, 0.9, 0.0]])
        self.hidden = row.unsqueeze(0).repeat(2, 1, 1)
        self.labels = torch.tensor([1, 1])

    def test_lossfree_selection_bias_is_not_a_route_error(self):
        router = _ToyRouter(use_lossfree_bias=True).eval()
        weights, indices, controls = _route_controls(
            router,
            self.hidden,
            self.labels,
        )
        self.assertTrue(torch.equal(indices, torch.ones_like(indices)))
        self.assertEqual(controls["route_mismatches"], 0)
        self.assertEqual(controls["unbiased_argmax_mismatches"], 4)
        self.assertTrue(controls["lossfree_bias_enabled"])
        expected = F.normalize(self.hidden, dim=-1)[..., 1:2]
        torch.testing.assert_close(weights, expected)

    def test_base_selection_matches_unbiased_argmax(self):
        router = _ToyRouter(use_lossfree_bias=False).eval()
        _, indices, controls = _route_controls(
            router,
            self.hidden,
            self.labels,
        )
        self.assertTrue(torch.equal(indices, torch.zeros_like(indices)))
        self.assertEqual(controls["route_mismatches"], 0)
        self.assertEqual(controls["unbiased_argmax_mismatches"], 0)
        self.assertFalse(controls["lossfree_bias_enabled"])

    def test_route_controls_require_eval_mode(self):
        router = _ToyRouter(use_lossfree_bias=True).train()
        with self.assertRaisesRegex(RuntimeError, "eval mode"):
            _route_controls(router, self.hidden, self.labels)

    def test_lossfree_bias_must_be_a_nontrainable_buffer(self):
        router = _ToyRouter(use_lossfree_bias=True)
        self.assertIs(
            _validate_lossfree_bias_contract(router, True),
            router.expert_bias,
        )
        router.expert_bias = nn.Parameter(router.expert_bias.clone())
        with self.assertRaisesRegex(TypeError, "registered buffer"):
            _validate_lossfree_bias_contract(router, True)

    def test_relative_mse_drift_rejects_zero_and_nonfinite_inputs(self):
        self.assertAlmostEqual(_relative_mse_drift(2.0, 2.5), 0.25)
        for native, repeated in (
            (0.0, 0.0),
            (float("nan"), 1.0),
            (1.0, float("inf")),
        ):
            with self.subTest(native=native, repeated=repeated):
                with self.assertRaisesRegex(RuntimeError, "MSE"):
                    _relative_mse_drift(native, repeated)

    def test_route_controls_reject_nonfinite_scores(self):
        router = _ToyRouter(use_lossfree_bias=True).eval()
        router.cluster_centers.data[0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "finite"):
            _route_controls(router, self.hidden, self.labels)


if __name__ == "__main__":
    unittest.main()
