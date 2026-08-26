import argparse
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from .download_ref_batches import ensure_ref_batches
except ImportError:
    from download_ref_batches import ensure_ref_batches


EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGENET_NUM_CLASSES = 1000


def _format_evaluator_log(command, return_code, stdout, stderr):
    status = "successfully" if return_code == 0 else f"with exit code {return_code}"
    return (
        f"The command finished {status}.\n"
        f"Command: {' '.join(command)}\n"
        "-------------------- STDOUT --------------------\n"
        f"{stdout or ''}\n"
        "-------------------- STDERR --------------------\n"
        f"{stderr or ''}"
    )


def _write_evaluator_log(log_file_path, output_log):
    with open(log_file_path, "w", encoding="utf-8") as file:
        file.write(output_log)
    print(f"The evaluation log has been saved to: {log_file_path}")


def run_evaluator(ref_npz_path, generated_npz_path, device="auto"):
    evaluator_script = os.path.join(EVALUATION_DIR, "evaluator.py")
    if not os.path.isfile(evaluator_script):
        raise FileNotFoundError(f"Evaluation script not found: {evaluator_script}")
    if not os.path.isfile(ref_npz_path):
        raise FileNotFoundError(f"Reference NPZ not found: {ref_npz_path}")
    if not os.path.isfile(generated_npz_path):
        raise FileNotFoundError(f"Generated NPZ not found: {generated_npz_path}")

    print("\n--- Start running the evaluation script ---")
    command = [
        sys.executable,
        evaluator_script,
        ref_npz_path,
        generated_npz_path,
        "--device",
        device,
    ]
    print(f"Executing command: {' '.join(command)}")

    log_file_path = os.path.splitext(generated_npz_path)[0] + "_eval_openai.txt"
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as error:
        output_log = _format_evaluator_log(
            command,
            error.returncode,
            error.stdout,
            error.stderr,
        )
        _write_evaluator_log(log_file_path, output_log)
        print("\nEvaluation failed. Check log for details.", file=sys.stderr)
        raise

    output_log = _format_evaluator_log(
        command,
        result.returncode,
        result.stdout,
        result.stderr,
    )
    _write_evaluator_log(log_file_path, output_log)
    print("\nThe evaluation completed successfully.")


def _validated_png_files(image_folder, expected_count):
    if expected_count <= 0:
        raise ValueError("Expected image count must be positive")
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

    all_files = sorted(
        file_name
        for file_name in os.listdir(image_folder)
        if file_name.lower().endswith(".png")
    )
    if len(all_files) < expected_count:
        raise ValueError(
            f"Expected at least {expected_count} PNG files in {image_folder}, "
            f"found {len(all_files)}"
        )

    label_pattern = re.compile(r"^img(\d+)_class(\d+)\.png$")
    labeled_files = []
    for file_name in all_files:
        match = label_pattern.search(file_name)
        if match is None:
            raise ValueError(f"PNG filename has an invalid sample index or label: {file_name}")
        sample_index = int(match.group(1))
        label = int(match.group(2))
        if not 0 <= label < IMAGENET_NUM_CLASSES:
            raise ValueError(f"ImageNet class label is out of range: {file_name}")
        labeled_files.append((sample_index, file_name, label))

    labeled_files.sort(key=lambda item: item[0])
    selected_files = labeled_files[:expected_count]
    selected_indices = [sample_index for sample_index, _, _ in selected_files]
    if selected_indices != list(range(expected_count)):
        raise ValueError(
            "PNG sample indices must be unique and contiguous from zero through "
            f"{expected_count - 1}"
        )
    return [(file_name, label) for _, file_name, label in selected_files]


def create_npz_from_images(
    image_folder,
    output_path,
    expected_count,
    img_size,
    run_eval,
    ref_npz_path,
    eval_device="auto",
):
    if len(img_size) != 2 or any(value <= 0 for value in img_size):
        raise ValueError("Image size must contain two positive integers")

    print("--- Configuration ---")
    print(f"Source Folder: {image_folder}")
    print(f"Output File:   {output_path}")
    print(f"Expected Count: {expected_count}")
    print(f"Image Size:    {img_size}")
    print(f"Run Evaluation: {'Yes' if run_eval else 'No'}")
    if run_eval:
        print(f"Reference NPZ: {ref_npz_path}")
    print("---------------------\n")

    labeled_files = _validated_png_files(image_folder, expected_count)
    print(f"Found {len(labeled_files)} valid PNG files.")
    images_array = np.zeros(
        (expected_count, img_size[1], img_size[0], 3),
        dtype=np.uint8,
    )
    labels_array = np.zeros((expected_count,), dtype=np.int64)
    for index, (file_name, label) in enumerate(
        tqdm(labeled_files, desc="Processing images")
    ):
        image_path = os.path.join(image_folder, file_name)
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize(img_size, Image.LANCZOS)
            images_array[index] = np.asarray(image)
        labels_array[index] = label

    print(f"\nSaving {expected_count} images to: {output_path}")
    np.savez_compressed(output_path, arr_0=images_array, arr_1=labels_array)
    print("NPZ file created successfully!")

    if run_eval:
        if not ref_npz_path:
            raise ValueError("Evaluation requested without a reference NPZ")
        run_evaluator(ref_npz_path, output_path, device=eval_device)


def main():
    parser = argparse.ArgumentParser(
        description="Create an NPZ dataset from an image folder and optionally run evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("image_folder", type=str, help="Path to the folder containing source images.")
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=50000,
        help="Required number of PNG images",
    )
    parser.add_argument("--size", nargs=2, type=int, default=[256, 256], metavar=('WIDTH', 'HEIGHT'), help="Target image size (WIDTH HEIGHT).")
    parser.add_argument(
        "--ref-npz",
        type=str,
        default=os.path.join(EVALUATION_DIR, "VIRTUAL_imagenet256_labeled.npz"),
        help="Path to the reference NPZ file for evaluation.",
    )
    parser.add_argument("--no-eval", action="store_true", help="If specified, skip the subsequent evaluation script.")
    parser.add_argument(
        "--eval-device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="TensorFlow activation device passed to evaluator.py",
    )

    args = parser.parse_args()

    if not args.no_eval:
        ensure_ref_batches()

    args.output = args.image_folder + ".npz"
    create_npz_from_images(
        image_folder=args.image_folder,
        output_path=args.output,
        expected_count=args.count,
        img_size=tuple(args.size),
        run_eval=not args.no_eval,
        ref_npz_path=args.ref_npz,
        eval_device=args.eval_device,
    )

if __name__ == '__main__':
    main()
