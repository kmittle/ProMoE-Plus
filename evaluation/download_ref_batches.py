"""Auto-download OpenAI reference batches for ImageNet FID evaluation."""

import os
import fcntl
import urllib.request

BASE_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet"

REF_FILES = {
    "VIRTUAL_imagenet256_labeled.npz": f"{BASE_URL}/256/VIRTUAL_imagenet256_labeled.npz",
    "VIRTUAL_imagenet512.npz": f"{BASE_URL}/512/VIRTUAL_imagenet512.npz",
}


def ensure_ref_batches(eval_dir=None):
    """Download reference npz files if they don't already exist.

    Uses a file lock to prevent multiple processes from downloading the
    same file simultaneously, which can corrupt the output.

    Args:
        eval_dir: Directory to store/check files. Defaults to the directory
                  containing this script (i.e. evaluation/).
    """
    if eval_dir is None:
        eval_dir = os.path.dirname(os.path.abspath(__file__))

    for filename, url in REF_FILES.items():
        local_path = os.path.join(eval_dir, filename)
        lock_path = local_path + ".lock"

        # Acquire an exclusive lock so only one process downloads at a time
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                # Re-check after acquiring lock — another process may have finished
                if os.path.isfile(local_path):
                    continue
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
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
