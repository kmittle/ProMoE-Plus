from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from credit_redistribution.heldout import (
    _load_npy_tensor,
    _materialize_case,
    stable_seed_mod,
)


class HeldoutMaterializationTest(unittest.TestCase):
    def test_unbatched_parameters_use_the_vae_channel_dimension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            latent_root = root / "latents"
            tensor_root = root / "tensors"
            source = latent_root / "0000" / "sample.latent.npz"
            source.parent.mkdir(parents=True)
            parameters = np.linspace(
                -0.25, 0.25, 8 * 32 * 32, dtype=np.float32
            ).reshape(8, 32, 32)
            np.savez(source, latent=parameters)
            case = {
                "index": 0,
                "label": 0,
                "relative_path": "0000/sample.latent.npz",
            }
            salt = "heldout-unit-test"
            record = _materialize_case(
                case, latent_root, tensor_root, salt, noise_draws=2
            )
            observed = _load_npy_tensor(tensor_root / record["z"]["path"])
            self.assertEqual(tuple(observed.shape), (4, 1, 32, 32))
            self.assertEqual(observed.dtype, torch.float32)

            seed = stable_seed_mod(
                2147483647, salt, case["relative_path"], "posterior"
            )
            generator = torch.Generator(device="cpu").manual_seed(seed)
            distribution = DiagonalGaussianDistribution(
                torch.from_numpy(parameters).unsqueeze(0)
            )
            expected = distribution.sample(generator=generator).squeeze(0)
            expected = expected.float().mul(torch.tensor(0.18215)).unsqueeze(1)
            self.assertTrue(torch.equal(observed, expected))
            self.assertEqual(len(record["noise"]), 2)


if __name__ == "__main__":
    unittest.main()
