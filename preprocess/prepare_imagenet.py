#!/usr/bin/env python3
"""
Prepare full-resolution ImageNet-1K for ProMoE training on a fresh server.

Idempotent, resume-safe pipeline (run from the repo root; the companion
``prepare_imagenet.sh`` handles that + env):

  1. check       -- if the latents are already complete, exit 0 (no download,
                    no preprocess) so training can start immediately.
  2. download    -- fetch the ILSVRC/imagenet-1k *full-resolution* train parquet
                    shards (HuggingFace first, ModelScope fallback) and
                    materialise them to  <root>/train/<label:04d>/<name>.JPEG
                    (one class dir per label, per-shard skip via sentinels).
  3. vae         -- ensure the SD-VAE (stabilityai/sd-vae-ft-mse) diffusers files
                    are cached at pretrained_ckpt/vae/ (HF first, ModelScope fallback).
  4. preprocess  -- run preprocess/preprocess_vae.py to encode every image into a
                    .latent.npz under <root>/sd-vae-ft-mse_Latents_256img_npz/
                    (per-file skip; multi-GPU).
  5. verify      -- assert EVERY image has a matching latent (a missing latent is
                    silently zero-filled by the training loader, which would
                    corrupt training) and only then write the completion sentinel.

On-disk layout produced (relative to the repo root):
    datasets/imagenet/train/<label:04d>/<name>.JPEG
    datasets/imagenet/sd-vae-ft-mse_Latents_256img_npz/<label:04d>/<name>.latent.npz

Why <label:04d> class dirs: train.py's CustomImageFolder assigns the class index
by the *sorted* order of the class-dir names (train.py:123-126,160-161). HF
imagenet-1k labels are the canonical ImageNet order (0..999), so zero-padded
label dirs make the folder label == the dataset label == canonical class. And
because train.py derives the latent path with str.replace('train', ...)
(train.py:163), the whole path must contain 'train' exactly once -- guaranteed
here by the relative layout (datasets/imagenet/train/<4 digits>/<name>).

Env / args:
    PROMOE_HF_DATASET   HF dataset id     (default ILSVRC/imagenet-1k)
    PROMOE_MS_DATASET   ModelScope id     (default ILSVRC/imagenet-1k)
    HF_TOKEN            HF token, only needed for the gated HF path
    --gpus              comma list for the VAE preprocess step (default: all visible)
    --source {auto,hf,modelscope}   download source (default auto = HF then MS)
"""

import argparse
import io
import logging
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import time
import urllib.parse

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - prepare_imagenet - %(levelname)s - %(message)s")
log = logging.getLogger("prepare_imagenet")

# ----------------------------------------------------------------------------- constants
REPO_ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), os.pardir))
DATA_ROOT_REL = osp.join("datasets", "imagenet")                       # relative to repo root
TRAIN_REL = osp.join(DATA_ROOT_REL, "train")
LATENT_DIR_NAME = "sd-vae-ft-mse_Latents_256img_npz"                   # MUST match train.py:104
LATENT_REL = osp.join(DATA_ROOT_REL, LATENT_DIR_NAME)
STATE_REL = osp.join(DATA_ROOT_REL, ".state")
DOWNLOAD_CACHE_REL = osp.join(DATA_ROOT_REL, "_parquet")               # transient parquet cache
VAE_HF_ID = "stabilityai/sd-vae-ft-mse"
VAE_CACHE_REL = osp.join("pretrained_ckpt", "vae", VAE_HF_ID.replace("/", "--"))
IMAGE_CACHE_FILE = osp.join("preprocess", "image_paths_cache.txt")     # MUST match train.py:101
EXPECTED_TRAIN_IMAGES = 1_281_167                                       # canonical ImageNet-1k train size
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

MS_BASE = "https://www.modelscope.cn/api/v1"
HF_BASE = "https://huggingface.co"


# ----------------------------------------------------------------------------- small helpers
def _require(mod, pip_name=None):
    try:
        return __import__(mod)
    except ImportError:
        sys.exit(f"[prepare_imagenet] missing dependency '{mod}'. "
                 f"Install it in the training env, e.g.  pip install {pip_name or mod}")


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def sentinel_path(name):
    return osp.join(STATE_REL, name)


def touch(path):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))


def exists_sentinel(name):
    return osp.exists(sentinel_path(name))


# ----------------------------------------------------------------------------- HTTP download
def _session():
    requests = _require("requests")
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
        retry = Retry(total=6, backoff_factor=1.5,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET"]))
        adapter = HTTPAdapter(max_retries=retry)
    except Exception:
        adapter = HTTPAdapter(max_retries=6)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _download_stream(session, url, dst, headers=None, expected_size=None):
    """Stream a URL to dst atomically (.part + rename). Returns bytes written."""
    os.makedirs(osp.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".part"
    written = 0
    with session.get(url, headers=headers or {}, stream=True, timeout=(30, 600)) as r:
        ctype = r.headers.get("Content-Type", "")
        if r.status_code != 200:
            body = ""
            try:
                body = r.content[:200].decode("utf-8", "replace")
            except Exception:
                pass
            raise RuntimeError(f"HTTP {r.status_code} for {url}\n  {body}")
        if "application/json" in ctype:  # ModelScope returns JSON on error with HTTP 200 sometimes
            raise RuntimeError(f"unexpected JSON (error) response for {url}\n  {r.content[:200]!r}")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
    if expected_size is not None and written != expected_size:
        os.remove(tmp)
        raise RuntimeError(f"size mismatch for {url}: got {written}, expected {expected_size}")
    os.replace(tmp, dst)
    return written


# ----------------------------------------------------------------------------- shard listing
def ms_list_train_shards(session, ms_id):
    """List data/train-*.parquet on ModelScope (paginated) -> (sorted paths, {path: size})."""
    url = (f"{MS_BASE}/datasets/{ms_id}/repo/tree"
           f"?Revision=master&Root=data&Recursive=false&PageSize=5000&PageNumber=1")
    r = session.get(url, timeout=(30, 120))
    r.raise_for_status()
    data = (r.json() or {}).get("Data") or {}
    files = data.get("Files") or []
    shards, sizes = [], {}
    for f in files:
        p = f.get("Path", "")
        name = p.rsplit("/", 1)[-1]
        if name.startswith("train-") and name.endswith(".parquet"):
            shards.append(p)
            sizes[p] = f.get("Size") or 0
    return sorted(shards), sizes


def hf_list_train_shards(session, hf_id, token):
    """List train parquet shards on HuggingFace via the datasets 'tree' API."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{HF_BASE}/api/datasets/{hf_id}/tree/main/data?recursive=false"
    r = session.get(url, headers=headers, timeout=(30, 120))
    if r.status_code in (401, 403):
        raise PermissionError(f"HuggingFace access denied for {hf_id} "
                              f"(gated dataset needs HF_TOKEN with accepted licence): HTTP {r.status_code}")
    r.raise_for_status()
    shards, sizes = [], {}
    for f in r.json():
        p = f.get("path", "")
        name = p.rsplit("/", 1)[-1]
        if name.startswith("train-") and name.endswith(".parquet"):
            shards.append(p)
            sizes[p] = f.get("size") or 0
    return sorted(shards), sizes


def ms_file_url(ms_id, filepath):
    return (f"{MS_BASE}/datasets/{ms_id}/repo"
            f"?Revision=master&FilePath={urllib.parse.quote(filepath)}&View=false")


def hf_file_url(hf_id, filepath):
    return f"{HF_BASE}/datasets/{hf_id}/resolve/main/{urllib.parse.quote(filepath)}"


# ----------------------------------------------------------------------------- materialisation
def materialise_shard(parquet_path, train_dir):
    """Read one parquet shard and write every image to train/<label:04d>/<name>.
    Overwrites (shard-level resume via sentinels), returns number of images written."""
    _require("pyarrow", "pyarrow")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    cols = set(pf.schema_arrow.names)
    if "image" not in cols or "label" not in cols:
        raise RuntimeError(f"{parquet_path}: unexpected columns {sorted(cols)} (need image,label)")
    stem = osp.splitext(osp.basename(parquet_path))[0]
    written = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["image", "label"])
        images = tbl.column("image").to_pylist()
        labels = tbl.column("label").to_pylist()
        for i, (img, label) in enumerate(zip(images, labels)):
            label = int(label)
            if not (0 <= label < 1000):
                raise RuntimeError(f"{parquet_path}: label {label} out of [0,1000)")
            data, orig = _img_bytes_and_name(img)
            base = _safe_name(orig, f"{stem}_{rg}_{i}")
            cls_dir = osp.join(train_dir, f"{label:04d}")
            os.makedirs(cls_dir, exist_ok=True)
            with open(osp.join(cls_dir, base), "wb") as f:
                f.write(data)
            written += 1
    return written


def _img_bytes_and_name(img):
    """Extract raw image bytes + original name from a parquet 'image' cell."""
    if isinstance(img, dict):
        data = img.get("bytes")
        name = img.get("path")
        if data is None and name and osp.exists(name):
            with open(name, "rb") as f:
                data = f.read()
        if data is None:
            raise RuntimeError("parquet image cell has no bytes")
        return data, name
    if isinstance(img, (bytes, bytearray)):
        return bytes(img), None
    # PIL / other -> re-encode to JPEG as a last resort
    from PIL import Image  # noqa
    buf = io.BytesIO()
    Image.open(io.BytesIO(img)).convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue(), None


def _safe_name(orig, fallback):
    """Return a filesystem-safe basename that never contains the substring 'train'
    (which would break train.py's replace('train', ...) latent derivation, silently
    zero-filling the image at train time).

    The contract is enforced on the ACTUAL returned value -- including the fallback,
    which is derived from the 'train-NNNNN-of-00294' parquet stem and therefore itself
    carries 'train'. We scrub any residual 'train' (case-insensitively) after choosing
    the name, so both the primary and fallback branches are safe."""
    name = osp.basename(orig) if orig else ""
    if not name or "train" in name.lower() or "/" in name or "\\" in name:
        name = fallback
    root, ext = osp.splitext(name)
    if ext.lower() not in IMG_EXTS:
        ext = ".JPEG"
    root = re.sub("train", "shard", root, flags=re.IGNORECASE)  # honor the no-'train' contract
    if not root:
        root = "img"
    return root + ext


# ----------------------------------------------------------------------------- stages
def stage_download(source, hf_id, ms_id, token, keep_parquet, limit_shards):
    if exists_sentinel("download.done"):
        log.info("download.done sentinel present -- skipping download/materialise")
        return
    os.makedirs(TRAIN_REL, exist_ok=True)
    os.makedirs(DOWNLOAD_CACHE_REL, exist_ok=True)
    session = _session()

    order = {"auto": ["hf", "modelscope"], "hf": ["hf"], "modelscope": ["modelscope"]}[source]
    shards = sizes = None
    picked = None
    for src in order:
        try:
            if src == "hf":
                shards, sizes = hf_list_train_shards(session, hf_id, token)
            else:
                shards, sizes = ms_list_train_shards(session, ms_id)
            if shards:
                picked = src
                log.info("listed %d train shards from %s (%s)",
                         len(shards), src, hf_id if src == "hf" else ms_id)
                break
        except Exception as e:
            log.warning("listing via %s failed: %s", src, e)
    if not picked:
        sys.exit("[prepare_imagenet] could not list train shards from any source. "
                 "For the gated HF dataset set HF_TOKEN; otherwise ModelScope should work.")

    if limit_shards:
        shards = shards[:limit_shards]
        log.warning("--limit-shards=%d : only materialising the first %d shards (DEBUG)",
                    limit_shards, len(shards))

    for n, sp in enumerate(shards, 1):
        shard_name = sp.rsplit("/", 1)[-1]
        done = sentinel_path(osp.join("shards", shard_name + ".done"))
        if osp.exists(done):
            log.info("[%d/%d] %s already materialised -- skip", n, len(shards), shard_name)
            continue
        local_parquet = osp.join(DOWNLOAD_CACHE_REL, shard_name)
        url = hf_file_url(hf_id, sp) if picked == "hf" else ms_file_url(ms_id, sp)
        headers = {"Authorization": f"Bearer {token}"} if (picked == "hf" and token) else None
        log.info("[%d/%d] downloading %s (%s)", n, len(shards), shard_name,
                 human(sizes.get(sp, 0)))
        _download_stream(session, url, local_parquet, headers=headers,
                         expected_size=sizes.get(sp) or None)
        log.info("[%d/%d] materialising %s", n, len(shards), shard_name)
        cnt = materialise_shard(local_parquet, TRAIN_REL)
        touch(done)
        if not keep_parquet:
            os.remove(local_parquet)
        log.info("[%d/%d] %s -> %d images", n, len(shards), shard_name, cnt)

    total = _count_images(TRAIN_REL)
    log.info("materialised train images: %d (expected ~%d)", total, EXPECTED_TRAIN_IMAGES)
    if limit_shards:
        log.warning("--limit-shards set: partial/debug run -- NOT writing download.done sentinel")
        return
    if total < EXPECTED_TRAIN_IMAGES:
        sys.exit(f"[prepare_imagenet] only {total} train images materialised, "
                 f"expected {EXPECTED_TRAIN_IMAGES}. Not writing download.done; re-run to resume.")
    if not keep_parquet and osp.isdir(DOWNLOAD_CACHE_REL):
        shutil.rmtree(DOWNLOAD_CACHE_REL, ignore_errors=True)
    touch(sentinel_path("download.done"))


def stage_vae(source, token):
    if osp.isdir(VAE_CACHE_REL) and os.listdir(VAE_CACHE_REL):
        log.info("VAE already cached at %s -- skip", VAE_CACHE_REL)
        return
    # HF path first, via the repo's own loader (downloads + save_pretrained to the cache)
    if source in ("auto", "hf"):
        try:
            sys.path.insert(0, REPO_ROOT)
            from utils import load_vae
            load_vae(VAE_HF_ID)  # writes into VAE_CACHE_REL
            if osp.isdir(VAE_CACHE_REL) and os.listdir(VAE_CACHE_REL):
                log.info("VAE provisioned from HuggingFace")
                return
        except Exception as e:
            log.warning("HF VAE download failed (%s); trying ModelScope", e)
    # ModelScope fallback: pull the diffusers files into the cache dir
    session = _session()
    os.makedirs(VAE_CACHE_REL, exist_ok=True)
    files = ["config.json", "diffusion_pytorch_model.safetensors"]
    for fn in files:
        url = (f"{MS_BASE}/models/{VAE_HF_ID}/repo"
               f"?Revision=master&FilePath={urllib.parse.quote(fn)}&View=false")
        log.info("downloading VAE file from ModelScope: %s", fn)
        _download_stream(session, url, osp.join(VAE_CACHE_REL, fn))
    log.info("VAE provisioned from ModelScope into %s", VAE_CACHE_REL)


def stage_preprocess(gpus, python_exe):
    if exists_sentinel("latents.done"):
        log.info("latents.done sentinel present -- skipping preprocess")
        return
    _rebuild_image_cache()
    env = dict(os.environ)
    env["PROMOE_DATA_PATH"] = TRAIN_REL  # relative -> keeps train.py's replace('train',..) safe
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    cmd = [python_exe, osp.join("preprocess", "preprocess_vae.py"),
           "--latent_save_root", LATENT_REL, "--skip-existing"]
    log.info("running: CUDA_VISIBLE_DEVICES=%s %s", env.get("CUDA_VISIBLE_DEVICES", "(all)"),
             " ".join(cmd))
    rc = subprocess.call(cmd, cwd=REPO_ROOT, env=env)
    if rc != 0:
        sys.exit(f"[prepare_imagenet] preprocess_vae.py exited with code {rc}")


def stage_verify():
    """Every image must have a matching latent, else training silently zero-fills it."""
    latent_set = set()
    for dirpath, _dirs, files in os.walk(LATENT_REL):
        rel = osp.relpath(dirpath, LATENT_REL)
        for fn in files:
            if fn.endswith(".latent.npz"):
                latent_set.add(osp.normpath(osp.join(rel, fn)))
    missing = 0
    total = 0
    first_missing = []
    for dirpath, _dirs, files in os.walk(TRAIN_REL):
        for fn in files:
            if not fn.lower().endswith(IMG_EXTS):
                continue
            total += 1
            img_path = osp.join(dirpath, fn)
            # Mirror train.py's EXACT latent derivation (replace('train', ...), NOT
            # preprocess's relpath) so a 'train'-in-filename divergence is caught here
            # loudly instead of silently zero-filling that image during training.
            latent_full = osp.splitext(img_path.replace("train", LATENT_DIR_NAME))[0] + ".latent.npz"
            latent_rel = osp.normpath(osp.relpath(latent_full, LATENT_REL))
            if latent_rel not in latent_set:
                missing += 1
                if len(first_missing) < 5:
                    first_missing.append(img_path)
    log.info("verify: %d images, %d latents, %d missing", total, len(latent_set), missing)
    if missing:
        for p in first_missing:
            log.error("  missing latent for %s", p)
        sys.exit(f"[prepare_imagenet] {missing} images have no latent. "
                 f"Re-run to encode the remainder (per-file skip makes this cheap).")
    touch(sentinel_path("latents.done"))
    log.info("ImageNet is fully prepared -- %d images, %d latents.", total, len(latent_set))


# ----------------------------------------------------------------------------- misc
def _count_images(root):
    n = 0
    for _dp, _dirs, files in os.walk(root):
        n += sum(1 for f in files if f.lower().endswith(IMG_EXTS))
    return n


def _rebuild_image_cache():
    """train.py / preprocess_vae.py share one global preprocess/image_paths_cache.txt.
    If it is missing, each spawned DDP rank regenerates it concurrently with
    nondeterministic ordering (ThreadPoolExecutor + as_completed) -> ranks partition
    DIFFERENT images (some never encoded) and the concurrent writes can garble the file.
    Pre-build it here, once, SORTED and ATOMICALLY, so every preprocess/train rank simply
    reads one consistent cache (no per-rank regeneration, no race)."""
    paths = []
    for dirpath, _dirs, files in os.walk(TRAIN_REL):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(osp.join(dirpath, fn))
    paths.sort()
    os.makedirs(osp.dirname(IMAGE_CACHE_FILE), exist_ok=True)
    tmp = IMAGE_CACHE_FILE + ".part"
    with open(tmp, "w") as f:
        f.write("\n".join(paths))
    os.replace(tmp, IMAGE_CACHE_FILE)
    log.info("rebuilt %s with %d image paths (sorted)", IMAGE_CACHE_FILE, len(paths))


def _check_one_derivation(sample_img):
    """Assert the two independent latent-path derivations agree for one image path."""
    pre = osp.normpath(osp.join(LATENT_REL, osp.splitext(osp.relpath(sample_img, TRAIN_REL))[0]
                                + ".latent.npz"))
    trn = osp.normpath(osp.splitext(sample_img.replace("train", LATENT_DIR_NAME))[0] + ".latent.npz")
    if sample_img.count("train") != 1:
        sys.exit(f"[prepare_imagenet] SELF-TEST FAILED: sample path {sample_img!r} contains "
                 f"'train' {sample_img.count('train')} times; the layout is unsafe for train.py's "
                 f"replace('train', ...). Keep the dataset under a 'train'-free path.")
    if pre != trn:
        sys.exit(f"[prepare_imagenet] SELF-TEST FAILED: latent-path derivations disagree.\n"
                 f"  preprocess -> {pre}\n  train      -> {trn}")
    return pre


def self_test_derivation():
    """Guard the two independent latent-path derivations against drift.
    preprocess_vae.save_latent : join(latent_root, relpath(img, data_path)).latent.npz
    train.CustomImageFolder     : img.replace('train', LATENT_DIR_NAME) + .latent.npz
    They must agree for our layout, else training would zero-fill silently."""
    # (1) a clean original-style ImageNet filename
    _check_one_derivation(osp.join(TRAIN_REL, "0000", "n01440764_10026.JPEG"))
    # (2) a fallback-style name: the parquet stem carries 'train', so _safe_name MUST
    #     scrub it -- otherwise the produced filename re-introduces 'train' and the two
    #     derivations diverge (the exact silent-zero-fill bug this guards against).
    fb = _safe_name(None, "train-00000-of-00294_0_5")
    if "train" in fb.lower():
        sys.exit(f"[prepare_imagenet] SELF-TEST FAILED: _safe_name leaked 'train' in {fb!r}")
    _check_one_derivation(osp.join(TRAIN_REL, "0000", fb))
    log.info("self-test OK: latent derivations agree (clean + fallback names, no 'train' leak)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Prepare full-res ImageNet-1K for ProMoE.")
    ap.add_argument("--gpus", default="", help="comma GPU list for the VAE preprocess step")
    ap.add_argument("--python", default=sys.executable, help="python used for preprocess_vae.py")
    ap.add_argument("--source", choices=["auto", "hf", "modelscope"], default="auto")
    ap.add_argument("--hf-dataset", default=os.environ.get("PROMOE_HF_DATASET", "ILSVRC/imagenet-1k"))
    ap.add_argument("--ms-dataset", default=os.environ.get("PROMOE_MS_DATASET", "ILSVRC/imagenet-1k"))
    ap.add_argument("--keep-parquet", action="store_true", help="keep parquet shards after extraction")
    ap.add_argument("--limit-shards", type=int, default=0, help="DEBUG: only first N shards")
    ap.add_argument("--force", action="store_true", help="ignore completion sentinels")
    ap.add_argument("--skip-preprocess", action="store_true", help="download/materialise only")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)  # everything below is relative to the repo root
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if args.force:
        for s in ("download.done", "latents.done"):
            p = sentinel_path(s)
            if osp.exists(p):
                os.remove(p)

    self_test_derivation()

    # fast path: already fully prepared
    if exists_sentinel("latents.done") and not args.force:
        log.info("ImageNet already prepared (latents.done). Nothing to do.")
        return

    stage_download(args.source, args.hf_dataset, args.ms_dataset, token,
                   args.keep_parquet, args.limit_shards)
    if args.skip_preprocess:
        log.info("--skip-preprocess set; stopping after materialisation.")
        return
    stage_vae(args.source, token)
    stage_preprocess(args.gpus, args.python)
    stage_verify()


if __name__ == "__main__":
    main()
