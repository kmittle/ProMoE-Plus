from __future__ import annotations

import unittest

import torch

from credit_redistribution.state_digest import (
    canonical_state_sha256,
    checkpoint_state_digests,
)


class StateDigestTest(unittest.TestCase):
    def test_mapping_insertion_order_does_not_change_digest(self):
        first = {"a": torch.tensor([1.0, 2.0]), "b": [3, "x"]}
        second = {"b": [3, "x"], "a": torch.tensor([1.0, 2.0])}
        self.assertEqual(
            canonical_state_sha256(first),
            canonical_state_sha256(second),
        )

    def test_tensor_content_change_changes_digest(self):
        first = torch.tensor([1.0, 2.0], dtype=torch.float32)
        second = first.clone()
        second[1] = torch.nextafter(second[1], torch.tensor(float("inf")))
        self.assertNotEqual(
            canonical_state_sha256(first),
            canonical_state_sha256(second),
        )

    def test_checkpoint_digest_covers_every_replay_section(self):
        checkpoint = {
            "step": 7,
            "model_state_dict": {"w": torch.tensor([1.0])},
            "ema_model_state_dict": {"w": torch.tensor([2.0])},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "trainer_state": {"next_step": 8},
        }
        digests = checkpoint_state_digests(checkpoint)
        self.assertEqual(
            set(digests),
            {
                "model_state_dict",
                "ema_model_state_dict",
                "optimizer_state_dict",
                "credit_redistribution_state",
                "trainer_state",
                "step",
                "combined",
            },
        )
        changed = dict(checkpoint)
        changed["trainer_state"] = {"next_step": 9}
        self.assertNotEqual(
            digests["combined"],
            checkpoint_state_digests(changed)["combined"],
        )


if __name__ == "__main__":
    unittest.main()
