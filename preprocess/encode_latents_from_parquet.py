#!/usr/bin/env python3
"""
Encode VAE latents DIRECTLY from already-downloaded ImageNet-1K parquet shards,
skipping the 140GB intermediate JPEG ImageFolder entirely.

Input : a directory of HF `ILSVRC/imagenet-1k` train parquet shards
        (train-NNNNN-of-00294.parquet), e.g. /lustre01/qianyuan/data/ILSVRC/imagenet-1k/data
Output: <latent_save_root>/<label:04d>/<shard>_<rg>_<i>.latent.npz, one per image,
        each holding 'latent' and 'latent_flip' == vae.encode(x).latent_dist.parameters
        (the EXACT same 8-channel format preprocess_vae.py produces, so the training
        loop's DiagonalGaussianDistribution(z).sample() consumes it unchanged).

The class dir is the zero-padded parquet label (`{label:04d}`), so train.py's
LatentFolder recovers the canonical ImageNet class index directly as int(dirname)
-- no image folder and no fragile str.replace('train', ...) derivation.

Multi-GPU: torch.multiprocessing.spawn; shard-parallel (rank r takes shards[r::world]),
so each rank reads whole parquet shards sequentially (parquet is columnar/row-group
based, so sequential per-shard reads are far cheaper than random row access).
Resume-safe: per-image skip when the target .latent.npz already exists.
"""
import argparse
import datetime
import glob
import io
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from torchvision import transforms
from PIL import Image
from utils import load_vae, find_free_port
from config import cfg
from preprocess.preprocess_vae import center_crop_arr  # reuse the EXACT ADM crop

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

IMG_LABEL_COLS = ["image", "label"]


def _img_bytes(cell):
    """Extract raw image bytes from a parquet 'image' cell (dict{bytes,path} | bytes | PIL-encoded)."""
    if isinstance(cell, dict):
        data = cell.get("bytes")
        if data is None:
            p = cell.get("path")
            if p and osp.exists(p):
                with open(p, "rb") as f:
                    data = f.read()
        if data is None:
            raise RuntimeError("parquet image cell has no bytes")
        return data
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise RuntimeError(f"unexpected parquet image cell type: {type(cell)}")


def _latent_path(latent_save_root, label, shard_stem, rg, i):
    return osp.join(latent_save_root, f"{int(label):04d}", f"{shard_stem}_{rg}_{i}.latent.npz")


def save_latent_atomic(latent_np, latent_flip_np, out_path):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    tmp = out_path + ".part"
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, latent=latent_np, latent_flip=latent_flip_np)
        os.replace(tmp, out_path)
    except BaseException:
        if osp.exists(tmp):
            os.remove(tmp)
        raise


def list_train_shards(parquet_dir):
    shards = sorted(glob.glob(osp.join(parquet_dir, "train-*.parquet")))
    if not shards:
        sys.exit(f"[encode_latents_from_parquet] no train-*.parquet shards under {parquet_dir}")
    return shards


def main(**kwargs):
    cfg.update(**kwargs)
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0))
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    cfg.gpus_per_machine = torch.cuda.device_count()
    cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    if cfg.world_size < 1:
        sys.exit("[encode_latents_from_parquet] no CUDA devices visible")
    print(f'Used GPU Number: {cfg.world_size}')
    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg,))
    return cfg


@torch.no_grad()
def worker(gpu, cfg):
    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', rank=cfg.rank, world_size=cfg.world_size,
                            timeout=datetime.timedelta(hours=5))

    import pyarrow.parquet as pq

    transform = transforms.Compose([
        transforms.Lambda(lambda im: center_crop_arr(im, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    vae = load_vae(cfg.sd_vae_ft_mse_vae_path)
    vae = vae.eval().to(gpu)

    latent_save_root = cfg.latent_save_root
    skip_existing = cfg.get('skip_existing', True)
    batch_size = cfg.preprocess_batch_size

    shards = list_train_shards(cfg.parquet_dir)
    limit = cfg.get('limit_shards', 0)
    if limit:
        shards = shards[:limit]  # DEBUG: only first N shards (matches prepare_imagenet download path)
    my_shards = shards[cfg.rank::cfg.world_size]
    if cfg.rank == 0:
        logging.info(f"{len(shards)} train shards total; rank 0 handles {len(my_shards)} "
                     f"(world_size={cfg.world_size})")

    def flush(batch_tensors, batch_out):
        if not batch_tensors:
            return 0
        x = torch.stack(batch_tensors).to(gpu, non_blocking=True)
        z = vae.encode(x).latent_dist.parameters
        z_flip = vae.encode(x.flip(dims=[3])).latent_dist.parameters
        z = z.detach().cpu().numpy()
        z_flip = z_flip.detach().cpu().numpy()
        for k in range(len(batch_out)):
            save_latent_atomic(z[k], z_flip[k], batch_out[k])
        return len(batch_out)

    written = skipped = 0
    for s_idx, shard in enumerate(my_shards):
        stem = osp.splitext(osp.basename(shard))[0]
        pf = pq.ParquetFile(shard)
        cols = set(pf.schema_arrow.names)
        if "image" not in cols or "label" not in cols:
            sys.exit(f"[encode_latents_from_parquet] {shard}: columns {sorted(cols)} (need image,label)")
        batch_tensors, batch_out = [], []
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=IMG_LABEL_COLS)
            images = tbl.column("image").to_pylist()
            labels = tbl.column("label").to_pylist()
            for i, (cell, label) in enumerate(zip(images, labels)):
                label = int(label)
                if not (0 <= label < 1000):
                    sys.exit(f"[encode_latents_from_parquet] {shard}: label {label} out of [0,1000)")
                out = _latent_path(latent_save_root, label, stem, rg, i)
                if skip_existing and osp.exists(out):
                    skipped += 1
                    continue
                img = Image.open(io.BytesIO(_img_bytes(cell))).convert("RGB")
                batch_tensors.append(transform(img))
                batch_out.append(out)
                if len(batch_tensors) >= batch_size:
                    written += flush(batch_tensors, batch_out)
                    batch_tensors, batch_out = [], []
        written += flush(batch_tensors, batch_out)
        batch_tensors, batch_out = [], []
        if cfg.rank == 0:
            logging.info(f"[rank0] shard {s_idx + 1}/{len(my_shards)} ({stem}) done; "
                         f"written={written} skipped={skipped}")

    logging.info(f"rank {cfg.rank} finished: {written} encoded, {skipped} skipped")
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encode VAE latents directly from ImageNet parquet shards.")
    parser.add_argument('--parquet-dir', dest='parquet_dir', type=str, required=True,
                        help="directory of train-*.parquet shards (already downloaded)")
    parser.add_argument('--latent-save-root', dest='latent_save_root', type=str, required=True,
                        help="root to write <label:04d>/<name>.latent.npz")
    parser.add_argument('--skip-existing', dest='skip_existing', action=argparse.BooleanOptionalAction,
                        default=True, help="skip images whose .latent.npz already exists (resume-safe)")
    parser.add_argument('--limit-shards', dest='limit_shards', type=int, default=0,
                        help="DEBUG: only encode the first N train shards")
    args = parser.parse_args()
    main(parquet_dir=args.parquet_dir, latent_save_root=args.latent_save_root,
         skip_existing=args.skip_existing, limit_shards=args.limit_shards)
