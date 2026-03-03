"""
Download reference statistics NPZ files required for evaluation.

This script ensures that VIRTUAL_imagenet256_labeled.npz and
VIRTUAL_imagenet512.npz exist in the evaluation directory before
running evaluation. It uses a file lock to guarantee that only one
process downloads each file at a time, making it safe to call from
multiple distributed processes simultaneously.

Usage:
    python evaluation/download_ref_stats.py
"""

import os
import sys
import fcntl
import urllib.request
import argparse

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "VIRTUAL_imagenet256_labeled.npz": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
        "ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
    ),
    "VIRTUAL_imagenet512.npz": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
        "ref_batches/imagenet/512/VIRTUAL_imagenet512.npz"
    ),
}


def download_file(url, dest_path):
    """Download a file from url to dest_path with progress reporting."""
    tmp_path = dest_path + ".tmp"
    print(f"Downloading {url} ...")

    def _report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct:.1f}%)")
        else:
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f} MB downloaded")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_report)
        print()  # newline after progress
        os.rename(tmp_path, dest_path)
        print(f"Saved to {dest_path}")
    except Exception:
        # Clean up partial download
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def ensure_file(filename, url):
    """Ensure a reference file exists, downloading if necessary.

    Uses a file lock so that when multiple processes call this
    concurrently, only one actually downloads the file.
    """
    dest_path = os.path.join(EVAL_DIR, filename)

    # Fast path: file already exists
    if os.path.exists(dest_path):
        print(f"[OK] {filename} already exists.")
        return

    lock_path = dest_path + ".lock"
    lock_fd = open(lock_path, "w")
    try:
        # Acquire an exclusive lock (blocks until available)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # Re-check after acquiring lock (another process may have finished)
        if os.path.exists(dest_path):
            print(f"[OK] {filename} already exists (downloaded by another process).")
            return

        download_file(url, dest_path)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        # Best-effort cleanup of lock file
        try:
            os.remove(lock_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Download reference NPZ files for evaluation."
    )
    parser.add_argument(
        "--only-256", action="store_true",
        help="Only download the 256x256 reference file."
    )
    parser.add_argument(
        "--only-512", action="store_true",
        help="Only download the 512x512 reference file."
    )
    args = parser.parse_args()

    if args.only_256:
        targets = {"VIRTUAL_imagenet256_labeled.npz": FILES["VIRTUAL_imagenet256_labeled.npz"]}
    elif args.only_512:
        targets = {"VIRTUAL_imagenet512.npz": FILES["VIRTUAL_imagenet512.npz"]}
    else:
        targets = FILES

    for filename, url in targets.items():
        ensure_file(filename, url)

    print("\nAll reference files are ready.")


if __name__ == "__main__":
    main()
