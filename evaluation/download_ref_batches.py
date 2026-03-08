"""Auto-download OpenAI reference batches for ImageNet FID evaluation."""

import os
import fcntl
import urllib.request

BASE_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet"

REF_FILES = {
    "VIRTUAL_imagenet256_labeled.npz": f"{BASE_URL}/256/VIRTUAL_imagenet256_labeled.npz",
    "VIRTUAL_imagenet512.npz": f"{BASE_URL}/512/VIRTUAL_imagenet512.npz",
}


def _missing_ref_files(eval_dir):
    """Return a list of (filename, url, local_path) entries that are missing."""
    missing = []
    for filename, url in REF_FILES.items():
        local_path = os.path.join(eval_dir, filename)
        if not os.path.isfile(local_path):
            missing.append((filename, url, local_path))
    return missing


def _cleanup_legacy_per_file_locks(eval_dir):
    """Remove old per-file lock artifacts from earlier implementations."""
    for filename in REF_FILES:
        legacy_lock = os.path.join(eval_dir, filename + ".lock")
        if os.path.isfile(legacy_lock):
            try:
                os.remove(legacy_lock)
            except OSError:
                # Best-effort cleanup only.
                pass


def ensure_ref_batches(eval_dir=None):
    """Download reference npz files if they don't already exist.

    Uses a directory-level file lock to prevent multiple processes from
    downloading the same files simultaneously, which can corrupt output.

    Args:
        eval_dir: Directory to store/check files. Defaults to the directory
                  containing this script (i.e. evaluation/).
    """
    if eval_dir is None:
        eval_dir = os.path.dirname(os.path.abspath(__file__))

    # Fast path: if all files already exist, skip locking and cleanup legacy lock files.
    missing = _missing_ref_files(eval_dir)
    if not missing:
        _cleanup_legacy_per_file_locks(eval_dir)
        return

    # Acquire an exclusive lock on the evaluation directory itself.
    dir_fd = os.open(eval_dir, os.O_RDONLY)
    try:
        fcntl.flock(dir_fd, fcntl.LOCK_EX)
        # Re-check under lock in case another process finished downloading.
        missing = _missing_ref_files(eval_dir)
        for filename, url, local_path in missing:
            print(f"Reference file not found: {local_path}")
            print(f"Downloading from {url} ...")
            tmp_path = local_path + ".tmp"
            try:
                urllib.request.urlretrieve(url, tmp_path)
                os.rename(tmp_path, local_path)
                print(f"Saved to {local_path}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
                raise
        _cleanup_legacy_per_file_locks(eval_dir)
    finally:
        fcntl.flock(dir_fd, fcntl.LOCK_UN)
        os.close(dir_fd)
