from io import BytesIO
import unittest
from unittest.mock import patch

from easydict import EasyDict
import torch
from torch.torch_version import TorchVersion

from analyses.denoising_regret.probe import _load_checkpoint_payload


class CheckpointLoadingTests(unittest.TestCase):
    def test_restricted_loader_allows_project_metadata_types(self):
        payload = {
            "cfg": EasyDict({"model_name": "ProMoE_TC_B"}),
            "torch_version": TorchVersion(str(torch.__version__)),
            "tensor": torch.ones(2),
        }
        buffer = BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)

        loaded = _load_checkpoint_payload(buffer)

        self.assertIsInstance(loaded["cfg"], dict)
        self.assertEqual(loaded["cfg"]["model_name"], "ProMoE_TC_B")
        self.assertEqual(str(loaded["torch_version"]), str(torch.__version__))
        torch.testing.assert_close(loaded["tensor"], payload["tensor"])

    def test_restricted_loader_does_not_retry_an_internal_type_error(self):
        calls = []

        def failing_load(path, map_location=None, weights_only=None):
            calls.append((path, map_location, weights_only))
            raise TypeError("checkpoint payload failure")

        with patch(
            "analyses.denoising_regret.probe.torch.load",
            failing_load,
        ):
            with self.assertRaisesRegex(TypeError, "checkpoint payload failure"):
                _load_checkpoint_payload("checkpoint.pth")

        self.assertEqual(calls, [("checkpoint.pth", "cpu", True)])

    def test_restricted_loader_rejects_an_explicit_legacy_signature(self):
        calls = []

        def legacy_load(path, map_location=None):
            calls.append((path, map_location))
            return {"step": 1}

        with patch(
            "analyses.denoising_regret.probe.torch.load",
            legacy_load,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "requires a PyTorch version with weights_only loading",
            ):
                _load_checkpoint_payload("checkpoint.pth")

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
