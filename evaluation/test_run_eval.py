import os
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from evaluation import run_eval


class RunEvaluatorTest(unittest.TestCase):
    def test_subprocess_failure_is_logged_and_raised(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = os.path.join(temp_dir, "ref.npz")
            generated_path = os.path.join(temp_dir, "generated.npz")
            np.savez(ref_path, arr_0=np.zeros((1, 1, 1, 3), dtype=np.uint8))
            np.savez(generated_path, arr_0=np.zeros((1, 1, 1, 3), dtype=np.uint8))
            error = subprocess.CalledProcessError(
                7,
                ["evaluator.py"],
                output="partial stdout",
                stderr="activation input is not finite",
            )
            with mock.patch.object(run_eval.subprocess, "run", side_effect=error):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_eval.run_evaluator(ref_path, generated_path)

            log_path = os.path.splitext(generated_path)[0] + "_eval_openai.txt"
            with open(log_path, "r", encoding="utf-8") as file:
                log_text = file.read()
            self.assertIn("exit code 7", log_text)
            self.assertIn("activation input is not finite", log_text)

    def test_missing_image_folder_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            run_eval._validated_png_files("/definitely/not/present", 1)

    def test_image_count_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Image.new("RGB", (2, 2)).save(
                os.path.join(temp_dir, "sample_000000_class0.png")
            )
            with self.assertRaisesRegex(ValueError, "Expected at least 2"):
                run_eval._validated_png_files(temp_dir, 2)

    def test_filename_must_contain_valid_imagenet_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Image.new("RGB", (2, 2)).save(os.path.join(temp_dir, "sample.png"))
            with self.assertRaisesRegex(ValueError, "sample index or label"):
                run_eval._validated_png_files(temp_dir, 1)

    def test_oversampled_tail_is_excluded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for index in range(3):
                Image.new("RGB", (2, 2)).save(
                    os.path.join(temp_dir, f"img{index:06d}_class{index}.png")
                )
            selected = run_eval._validated_png_files(temp_dir, 2)
            self.assertEqual(
                selected,
                [
                    ("img000000_class0.png", 0),
                    ("img000001_class1.png", 1),
                ],
            )

    def test_valid_images_produce_exact_npz(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = os.path.join(temp_dir, "images")
            os.mkdir(image_dir)
            Image.new("RGB", (3, 4), color=(1, 2, 3)).save(
                os.path.join(image_dir, "img000000_class9.png")
            )
            output_path = image_dir + ".npz"
            run_eval.create_npz_from_images(
                image_dir,
                output_path,
                expected_count=1,
                img_size=(2, 2),
                run_eval=False,
                ref_npz_path=None,
            )
            with np.load(output_path) as data:
                self.assertEqual(data["arr_0"].shape, (1, 2, 2, 3))
                self.assertEqual(data["arr_0"].dtype, np.uint8)
                self.assertEqual(data["arr_1"].tolist(), [9])


if __name__ == "__main__":
    unittest.main()
