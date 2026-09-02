import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from preprocess.build_dino_route_table import (
    LATENT_SCALE,
    TABLE_METHOD,
    TABLE_VERSION,
    _decode_model_latents,
    _prepare_new_output_pair,
    _publish_output_pair_no_replace,
    _require_new_output_pair,
)


class _RecordingVAE:
    def __init__(self):
        self.decode_input = None

    def decode(self, latent):
        self.decode_input = latent.clone()
        return SimpleNamespace(sample=latent + 1.0)


class BuildDinoRouteTableTests(unittest.TestCase):
    def test_decode_converts_model_latents_back_to_vae_space(self):
        vae = _RecordingVAE()
        model_latents = torch.full((2, 4, 3, 3), LATENT_SCALE)

        decoded = _decode_model_latents(vae, model_latents)

        self.assertTrue(torch.equal(vae.decode_input, torch.ones_like(model_latents)))
        self.assertTrue(torch.equal(decoded, torch.full_like(model_latents, 2.0)))

    def test_decode_rejects_non_image_latents(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            _decode_model_latents(_RecordingVAE(), torch.ones(4, 3, 3))

    def test_table_metadata_distinguishes_corrected_decode(self):
        self.assertEqual(TABLE_VERSION, 2)
        self.assertIn("correct_vae_decode", TABLE_METHOD)

    def test_output_pair_refuses_either_existing_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "table.npz"
            metadata = output.with_suffix(".npz.json")
            metadata.write_text("locked\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                _require_new_output_pair(output)
            self.assertEqual(metadata.read_text(encoding="utf-8"), "locked\n")

    def test_output_pair_rejects_a_dangling_output_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_target = root / "missing-table.npz"
            output_link = root / "table.npz"
            output_link.symlink_to(missing_target)

            with self.assertRaises(FileExistsError):
                _prepare_new_output_pair(output_link)
            self.assertTrue(output_link.is_symlink())
            self.assertFalse(missing_target.exists())

    def test_output_pair_publish_never_replaces_a_racing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz_tmp = root / "table.tmp.npz"
            json_tmp = root / "table.tmp.json"
            output = root / "table.npz"
            metadata = root / "table.npz.json"
            npz_tmp.write_bytes(b"new table")
            json_tmp.write_text("new metadata\n", encoding="utf-8")
            output.write_bytes(b"locked table")

            with self.assertRaises(FileExistsError):
                _publish_output_pair_no_replace(
                    npz_tmp, json_tmp, output, metadata
                )
            self.assertEqual(output.read_bytes(), b"locked table")
            self.assertFalse(metadata.exists())

    def test_output_pair_rolls_back_if_metadata_target_wins_race(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz_tmp = root / "table.tmp.npz"
            json_tmp = root / "table.tmp.json"
            output = root / "table.npz"
            metadata = root / "table.npz.json"
            npz_tmp.write_bytes(b"new table")
            json_tmp.write_text("new metadata\n", encoding="utf-8")
            metadata.write_text("locked metadata\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                _publish_output_pair_no_replace(
                    npz_tmp, json_tmp, output, metadata
                )
            self.assertFalse(output.exists())
            self.assertEqual(
                metadata.read_text(encoding="utf-8"),
                "locked metadata\n",
            )


if __name__ == "__main__":
    unittest.main()
