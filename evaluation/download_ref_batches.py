"""Auto-download OpenAI reference batches for ImageNet FID evaluation."""

import os
import urllib.request

BASE_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet"

REF_FILES = {
    "VIRTUAL_imagenet256_labeled.npz": f"{BASE_URL}/256/VIRTUAL_imagenet256_labeled.npz",
    "VIRTUAL_imagenet512.npz": f"{BASE_URL}/512/VIRTUAL_imagenet512.npz",
}


def ensure_ref_batches(eval_dir=None):
    """Download reference npz files if they don't already exist.

    Args:
        eval_dir: Directory to store/check files. Defaults to the directory
                  containing this script (i.e. evaluation/).
    """
    if eval_dir is None:
        eval_dir = os.path.dirname(os.path.abspath(__file__))

    for filename, url in REF_FILES.items():
        local_path = os.path.join(eval_dir, filename)
        if os.path.isfile(local_path):
            continue
        print(f"Reference file not found: {local_path}")
        print(f"Downloading from {url} ...")
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"Saved to {local_path}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            if os.path.isfile(local_path):
                os.remove(local_path)
            raise
