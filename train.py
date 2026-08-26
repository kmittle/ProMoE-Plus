import os
import os.path as osp
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.cuda.amp as amp
import torch.optim as optim
import numpy as np
import logging
import datetime
import copy
import torch.nn as nn
from torch.utils.data import BatchSampler, Dataset, DataLoader, DistributedSampler
import math
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from einops import rearrange
from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import colorlog
import glob
import yaml
import argparse
import random
import operator
import hashlib
import numbers
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
from utils import deep_update, find_free_port, load_vae
from torch.nn.utils import clip_grad_norm_

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

from config import cfg
from models.models_DiT import DiT as DiT
from models.models_TCDiT import DiT as TCDiT
from models.models_ECDiT import DiT as ECDiT
from models.models_DiffMoE import DiT as DiffMoE
from models.models_ProMoE_TC import DiT as ProMoE_TC
from models.models_ProMoE_TC_symmetric import DiT as ProMoE_TC_symmetric
from models.models_ProMoE_TC_sigmoid import DiT as ProMoE_TC_sigmoid
from models.models_ProMoE_TC_hierar import DiT as ProMoE_TC_hierar
from models.models_ProMoE_TC_hetero_expert import DiT as ProMoE_TC_hetero_expert
from models.models_ProMoE_EC import DiT as ProMoE_EC
from models.models_ProMoE_EC_batch_choice import DiT as ProMoE_EC_BC
from models.models_ProMoE_TC_proto_t import DiT as ProMoE_TC_proto_t
from models.models_ProMoE_EC_batch_choice_proto_t import DiT as ProMoE_EC_BC_proto_t
from models.models_ProMoE_EC_batch_choice_hetero import DiT as ProMoE_EC_BC_hetero
from models.models_ProMoE_TC_anchor import DiT as ProMoE_TC_anchor
from models.models_ProMoE_TC_proto_choice import DiT as ProMoE_TC_proto_choice
from models.models_ProMoE_TC_noise_expert import DiT as ProMoE_TC_noise_expert
from models.models_ProMoE_TC_noise_expert_proj import DiT as ProMoE_TC_noise_expert_proj
from models.models_ProMoE_TC_noise_expert_ema import DiT as ProMoE_TC_noise_expert_ema
from models.models_ProMoE_TC_expert_contra import DiT as ProMoE_TC_expert_contra
from models.models_ProMoE_TC_dagfuse import DiT as ProMoE_TC_dagfuse
from models.models_ProMoE_TC_lbcontra import DiT as ProMoE_TC_lbcontra
from models.models_ProMoE_TC_adepth import DiT as ProMoE_TC_adepth
from models.models_ProMoE_TC_lossfree import DiT as ProMoE_TC_lossfree
from models.models_ProMoE_TC_lsreg import DiT as ProMoE_TC_lsreg
from models.models_ProMoE_TC_dagfuse_dense import DiT as ProMoE_TC_dagfuse_dense
from models.models_ProMoE_TC_dagfuse_densenet import DiT as ProMoE_TC_dagfuse_densenet
from models.models_ProMoE_TC_dagfuse_sharedroute import DiT as ProMoE_TC_dagfuse_sharedroute
from models.models_ProMoE_TC_dagfuse_region import DiT as ProMoE_TC_dagfuse_region
from models.models_ProMoE_TC_denoising_regret import DiT as ProMoE_TC_denoising_regret

model_dict = {
    "DiT_B": (DiT, "DiT_B_config"),
    "DiT_L": (DiT, "DiT_L_config"),
    "DiT_XL": (DiT, "DiT_XL_config"),
    "TCDiT_L_E8": (TCDiT, "TCDiT_L_E8_config"),
    "ECDiT_L_E8": (ECDiT, "ECDiT_L_E8_config"),
    "DiffMoE_B_E8": (DiffMoE, "DiffMoE_DiT_B_E8_config"),
    "DiffMoE_L_E8": (DiffMoE, "DiffMoE_DiT_L_E8_config"),
    "DiffMoE_XL_E8": (DiffMoE, "DiffMoE_DiT_XL_E8_config"),
    "ProMoE_TC_S": (ProMoE_TC, "DiT_S_config"),
    "ProMoE_TC_S_symmetric": (ProMoE_TC_symmetric, "DiT_S_config"),
    "ProMoE_TC_S_sigmoid": (ProMoE_TC_sigmoid, "DiT_S_config"),
    "ProMoE_TC_S_hierar": (ProMoE_TC_hierar, "DiT_S_config"),
    "ProMoE_TC_S_hetero_expert": (ProMoE_TC_hetero_expert, "DiT_S_config"),
    "ProMoE_TC_B": (ProMoE_TC, "DiT_B_config"),
    "ProMoE_TC_B_hierar": (ProMoE_TC_hierar, "DiT_B_config"),
    "ProMoE_TC_B_hetero_expert": (ProMoE_TC_hetero_expert, "DiT_B_config"),
    "ProMoE_TC_L": (ProMoE_TC, "DiT_L_config"),
    "ProMoE_TC_XL": (ProMoE_TC, "DiT_XL_config"),
    "ProMoE_EC_L": (ProMoE_EC, "DiT_L_config"),
    "ProMoE_EC_BC_B": (ProMoE_EC_BC, "DiT_B_config"),
    "ProMoE_TC_B_proto_t": (ProMoE_TC_proto_t, "DiT_B_config"),
    "ProMoE_EC_BC_B_proto_t": (ProMoE_EC_BC_proto_t, "DiT_B_config"),
    "ProMoE_EC_BC_hetero_B": (ProMoE_EC_BC_hetero, "DiT_B_config"),
    "ProMoE_TC_B_anchor": (ProMoE_TC_anchor, "DiT_B_config"),
    "ProMoE_TC_B_proto_choice": (ProMoE_TC_proto_choice, "DiT_B_config"),
    "ProMoE_TC_B_noise_expert": (ProMoE_TC_noise_expert, "DiT_B_config"),
    "ProMoE_TC_L_noise_expert": (ProMoE_TC_noise_expert, "DiT_L_config"),
    "ProMoE_TC_B_noise_expert_proj": (ProMoE_TC_noise_expert_proj, "DiT_B_config"),
    "ProMoE_TC_L_noise_expert_proj": (ProMoE_TC_noise_expert_proj, "DiT_L_config"),
    "ProMoE_TC_B_noise_expert_ema": (ProMoE_TC_noise_expert_ema, "DiT_B_config"),
    "ProMoE_TC_L_noise_expert_ema": (ProMoE_TC_noise_expert_ema, "DiT_L_config"),
    "ProMoE_TC_B_expert_contra": (ProMoE_TC_expert_contra, "DiT_B_config"),
    "ProMoE_TC_L_expert_contra": (ProMoE_TC_expert_contra, "DiT_L_config"),
    "ProMoE_TC_B_dagfuse": (ProMoE_TC_dagfuse, "DiT_B_config"),
    "ProMoE_TC_B_lbcontra": (ProMoE_TC_lbcontra, "DiT_B_config"),
    "ProMoE_TC_B_adepth": (ProMoE_TC_adepth, "DiT_B_config"),
    "ProMoE_TC_B_lossfree": (ProMoE_TC_lossfree, "DiT_B_config"),
    "ProMoE_TC_B_lsreg": (ProMoE_TC_lsreg, "DiT_B_config"),
    "ProMoE_TC_B_dagfuse_dense": (ProMoE_TC_dagfuse_dense, "DiT_B_config"),
    "ProMoE_TC_B_dagfuse_densenet": (ProMoE_TC_dagfuse_densenet, "DiT_B_config"),
    "ProMoE_TC_B_dagfuse_sharedroute": (ProMoE_TC_dagfuse_sharedroute, "DiT_B_config"),
    "ProMoE_TC_B_dagfuse_region": (ProMoE_TC_dagfuse_region, "DiT_B_config"),
    "ProMoE_TC_B_FDRR": (ProMoE_TC_denoising_regret, "DiT_B_config"),
}

DENOISING_REGRET_MODELS = {"ProMoE_TC_B_FDRR"}
TRAINER_STATE_VERSION = 2
LEGACY_TRAINER_STATE_VERSION = 1
AUGMENTATION_SEED_VERSION = 1
SAMPLER_CONTRACT_VERSION = 1
DATASET_IDENTITY_VERSION = 1
IMAGENET_NUM_CLASSES = 1000
_UINT64_MASK = (1 << 64) - 1


def _build_latent_class_to_idx(observed_class_names, root_class_names):
    observed = sorted(set(observed_class_names))
    if not observed:
        raise ValueError("No latent class directories contain latent files")

    if all(name.isdigit() for name in observed):
        labels = [int(name) for name in observed]
        if len(labels) != len(set(labels)) or any(
            not 0 <= label < IMAGENET_NUM_CLASSES for label in labels
        ):
            raise ValueError(
                "Numeric latent class directories must map uniquely into [0, 999]"
            )
        return dict(zip(observed, labels))

    is_synset_layout = all(
        len(name) == 9 and name.startswith('n') and name[1:].isdigit()
        for name in observed
    )
    if not is_synset_layout:
        raise ValueError(
            "Latent class directories must be numeric labels or ImageNet "
            f"synsets, got examples: {observed[:3]}"
        )

    root_classes = sorted(set(root_class_names))
    if (
        len(root_classes) != IMAGENET_NUM_CLASSES
        or not all(
            len(name) == 9 and name.startswith('n') and name[1:].isdigit()
            for name in root_classes
        )
    ):
        raise ValueError(
            "Synset latent layout must expose exactly 1000 ImageNet class "
            f"directories, found {len(root_classes)}"
        )
    unknown = sorted(set(observed) - set(root_classes))
    if unknown:
        raise ValueError(f"Latent files reference unknown synsets: {unknown[:3]}")
    return {name: index for index, name in enumerate(root_classes)}


def _require_nonnegative_index(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _mix_uint64(value):
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _augmentation_seed(
    base_seed,
    rank,
    epoch,
    batch_index,
    sample_position,
    sample_index,
):
    """Derive augmentation randomness from the sampler position, not a worker."""

    components = (
        base_seed,
        rank,
        epoch,
        batch_index,
        sample_position,
        sample_index,
        AUGMENTATION_SEED_VERSION,
    )
    mixed = 0
    for name, component in zip(
        (
            'base_seed',
            'rank',
            'epoch',
            'batch_index',
            'sample_position',
            'sample_index',
            'augmentation_seed_version',
        ),
        components,
    ):
        component = _require_nonnegative_index(component, name)
        mixed = _mix_uint64(mixed ^ _mix_uint64(component))
    return mixed & ((1 << 63) - 1)


def _split_dataset_index(index):
    if not isinstance(index, tuple):
        return _require_nonnegative_index(index, 'sample_index'), None
    if len(index) != 2:
        raise ValueError("Seeded dataset index must be (sample_index, augmentation_seed)")
    sample_index, augmentation_seed = index
    return (
        _require_nonnegative_index(sample_index, 'sample_index'),
        _require_nonnegative_index(augmentation_seed, 'augmentation_seed'),
    )


def _hash_dataset_record(digest, *values):
    for value in values:
        encoded = str(value).encode('utf-8', errors='surrogateescape')
        digest.update(len(encoded).to_bytes(8, byteorder='little'))
        digest.update(encoded)


def _dataset_sampler_identity(dataset):
    """Fingerprint the ordered samples and labels that drive sampler indices."""

    if hasattr(dataset, 'latent_paths') and hasattr(dataset, 'class_to_idx'):
        root = getattr(dataset, 'latent_dir', None)
        samples = (
            (
                path,
                dataset.class_to_idx[os.path.basename(os.path.dirname(path))],
            )
            for path in dataset.latent_paths
        )
    elif hasattr(dataset, 'image_paths') and hasattr(dataset, 'class_to_idx'):
        root = getattr(dataset, 'root_dir', None)
        samples = (
            (
                path,
                dataset.class_to_idx[os.path.basename(os.path.dirname(path))],
            )
            for path in dataset.image_paths
        )
    elif hasattr(dataset, 'samples'):
        root = getattr(dataset, 'root', None)
        samples = iter(dataset.samples)
    else:
        raise TypeError(
            "Dataset must expose ordered paths and labels for resumable training"
        )

    normalized_root = os.path.normpath(root) if root is not None else None
    digest = hashlib.sha256()
    dataset_type = f"{type(dataset).__module__}.{type(dataset).__qualname__}"
    _hash_dataset_record(
        digest,
        DATASET_IDENTITY_VERSION,
        dataset_type,
        len(dataset),
    )
    sample_count = 0
    for path, label in samples:
        normalized_path = os.path.normpath(path)
        if normalized_root is not None:
            normalized_path = os.path.relpath(normalized_path, normalized_root)
        _hash_dataset_record(digest, normalized_path, operator.index(label))
        sample_count += 1
    if sample_count != len(dataset):
        raise ValueError(
            "Dataset sample metadata length differs from the dataset length"
        )
    return {
        'version': DATASET_IDENTITY_VERSION,
        'type': dataset_type,
        'num_samples': sample_count,
        'ordered_samples_sha256': digest.hexdigest(),
    }


def _build_sampler_contract(
    dataset,
    sampler_type,
    global_seed,
    per_rank_batch_size,
    case1_prob=None,
):
    if sampler_type not in {'distributed', 'structured_distributed'}:
        raise ValueError(f"Unsupported sampler type: {sampler_type!r}")
    global_seed = _require_nonnegative_index(global_seed, 'global_seed')
    per_rank_batch_size = _require_nonnegative_index(
        per_rank_batch_size,
        'per_rank_batch_size',
    )
    if per_rank_batch_size < 1:
        raise ValueError("per_rank_batch_size must be positive")
    if sampler_type == 'structured_distributed':
        case1_prob = float(case1_prob)
        if not math.isfinite(case1_prob) or not 0.0 <= case1_prob <= 1.0:
            raise ValueError("structured_batch_case1_prob must be in [0, 1]")
    elif case1_prob is not None:
        raise ValueError("case1_prob only applies to structured sampling")
    return {
        'version': SAMPLER_CONTRACT_VERSION,
        'type': sampler_type,
        'global_seed': global_seed,
        'per_rank_batch_size': per_rank_batch_size,
        'drop_last': False,
        'case1_prob': case1_prob,
        'dataset': _dataset_sampler_identity(dataset),
    }


def _validate_sampler_contract(saved_contract, current_contract):
    if not isinstance(saved_contract, dict):
        raise ValueError("Checkpoint sampler contract is missing or invalid")
    if not isinstance(current_contract, dict):
        raise ValueError("Current sampler contract is missing or invalid")
    if saved_contract.get('version') != SAMPLER_CONTRACT_VERSION:
        raise ValueError("Checkpoint sampler contract version is incompatible")
    if current_contract.get('version') != SAMPLER_CONTRACT_VERSION:
        raise ValueError("Current sampler contract version is incompatible")
    if saved_contract != current_contract:
        changed_fields = sorted(
            key
            for key in set(saved_contract) | set(current_contract)
            if saved_contract.get(key) != current_contract.get(key)
        )
        raise ValueError(
            "Checkpoint sampler contract differs from the current data pipeline: "
            f"{changed_fields}"
        )


class ResumableBatchSampler:
    """Skip consumed batches and attach restart-stable augmentation seeds."""

    def __init__(self, batch_sampler, seed=0, rank=0):
        self.batch_sampler = batch_sampler
        self.seed = _require_nonnegative_index(seed, 'seed')
        self.rank = _require_nonnegative_index(rank, 'rank')
        self.epoch = 0
        self.start_batch = 0

    @property
    def full_length(self):
        return len(self.batch_sampler)

    def set_epoch(self, epoch):
        epoch = _require_nonnegative_index(epoch, 'epoch')
        self.epoch = epoch
        target = self.batch_sampler
        if hasattr(target, 'set_epoch'):
            target.set_epoch(epoch)
            return
        sampler = getattr(target, 'sampler', None)
        if sampler is None or not hasattr(sampler, 'set_epoch'):
            raise TypeError("Wrapped batch sampler does not support set_epoch")
        sampler.set_epoch(epoch)

    def set_start_batch(self, start_batch):
        if (
            isinstance(start_batch, bool)
            or not isinstance(start_batch, int)
            or not 0 <= start_batch <= self.full_length
        ):
            raise ValueError(
                f"start_batch must be in [0, {self.full_length}], "
                f"got {start_batch!r}"
            )
        self.start_batch = start_batch

    def __iter__(self):
        for batch_index, batch in enumerate(self.batch_sampler):
            if batch_index >= self.start_batch:
                yield [
                    (
                        _require_nonnegative_index(index, 'sample_index'),
                        _augmentation_seed(
                            self.seed,
                            self.rank,
                            self.epoch,
                            batch_index,
                            sample_position,
                            index,
                        ),
                    )
                    for sample_position, index in enumerate(batch)
                ]

    def __len__(self):
        return self.full_length - self.start_batch

class CustomImageFolder(Dataset):
    def __init__(self, root_dir, cfg=None):
        self.root_dir = root_dir
        self.CACHE_FILE = 'preprocess/image_paths_cache.txt'
        self.image_paths = self._load_or_generate_image_paths()
        self.class_to_idx = self._get_class_to_idx()
        self.latent_dir_name = 'sd-vae-ft-mse_Latents_256img_npz'
        self.latent_shape = (4, 1, cfg.image_size // 8, cfg.image_size // 8)

    def _load_or_generate_image_paths(self):
        if os.path.exists(self.CACHE_FILE) and os.path.getsize(self.CACHE_FILE) > 0:
            with open(self.CACHE_FILE, 'r') as f:
                image_paths = f.read().splitlines()
            logging.info(f"****************Loaded image paths from cache: {self.CACHE_FILE}")
            return image_paths

        image_paths = self._get_image_paths(self.root_dir)
        os.makedirs(osp.dirname(self.CACHE_FILE), exist_ok=True)
        # Save to cache for future use
        with open(self.CACHE_FILE, 'w') as f:
            f.write('\n'.join(image_paths))
        
        logging.info(f"****************Generated cache for image paths: {self.CACHE_FILE}")
        return image_paths

    def _get_class_to_idx(self):
        # Deduce classes from directory names in the root directory
        classes = sorted({os.path.basename(os.path.dirname(path)) for path in self.image_paths})
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

    def _get_image_paths(self, root_dir):
        image_paths = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for entry in os.scandir(root_dir):
                if entry.is_dir(follow_symlinks=False):
                    futures.append(executor.submit(self._get_image_paths_from_dir, entry.path))
                elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(entry.path)

            for future in as_completed(futures):
                image_paths.extend(future.result())

        return image_paths

    def _get_image_paths_from_dir(self, dir_path):
        image_paths = []
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(entry.path)
                elif entry.is_dir(follow_symlinks=False):
                    image_paths.extend(self._get_image_paths_from_dir(entry.path))
        return image_paths

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        idx, augmentation_seed = _split_dataset_index(idx)
        img_path = self.image_paths[idx]

        # Deduce class label from parent directory name
        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[class_name]

        latent_path = img_path.replace('train', self.latent_dir_name)
        latent_path = os.path.splitext(latent_path)[0] + '.latent.npz'

        if osp.exists(latent_path):
            use_flip = (
                bool(augmentation_seed & 1)
                if augmentation_seed is not None
                else bool(torch.rand(1) >= 0.5)
            )
            latent_key = 'latent_flip' if use_flip else 'latent'
            with np.load(latent_path) as npz_data:
                latent_z_data = np.array(npz_data[latent_key], copy=True)
            latent_z = torch.from_numpy(latent_z_data)
        else:
            latent_z = torch.zeros(self.latent_shape)
            logging.info(f"{latent_path} is not exists!!!!")

        return img_path, label, latent_z


class LatentFolder(Dataset):
    """Pre-encoded-latents dataset that does not require the image folder.

    Parquet-direct preprocessing uses zero-padded numeric class directories. The
    original ImageNet preprocessing path uses synset directories such as
    ``n01440764``; sorting those names matches ImageFolder's class-index rule.
    Both layouts contain the same 8-channel VAE distribution parameters.
    """

    def __init__(self, latent_dir, cfg=None):
        self.latent_dir = latent_dir
        self.CACHE_FILE = 'preprocess/latent_paths_cache.txt'
        self.latent_paths = self._load_or_generate_latent_paths()
        self.class_to_idx = self._get_class_to_idx()

    def _get_class_to_idx(self):
        observed_class_names = {
            os.path.basename(os.path.dirname(path))
            for path in self.latent_paths
        }
        root_class_names = [
            entry.name
            for entry in os.scandir(self.latent_dir)
            if entry.is_dir(follow_symlinks=False)
        ]
        return _build_latent_class_to_idx(
            observed_class_names,
            root_class_names,
        )

    def _load_or_generate_latent_paths(self):
        if os.path.exists(self.CACHE_FILE) and os.path.getsize(self.CACHE_FILE) > 0:
            with open(self.CACHE_FILE, 'r') as f:
                latent_paths = f.read().splitlines()
            # Guard against a STALE cache pointing at a different latent_data_path (e.g. after
            # changing PROMOE_LATENT_PATH): only trust it if its entries are under latent_dir.
            root = os.path.normpath(self.latent_dir)
            if latent_paths and os.path.normpath(latent_paths[0]).startswith(root + os.sep):
                logging.info(f"****************Loaded latent paths from cache: {self.CACHE_FILE}")
                return latent_paths
            logging.info(f"****************Stale latent cache (not under {self.latent_dir}); regenerating")

        latent_paths = self._get_latent_paths(self.latent_dir)
        os.makedirs(osp.dirname(self.CACHE_FILE), exist_ok=True)
        # Atomic write via a unique temp: DDP ranks may regenerate this concurrently (if prepare
        # didn't pre-build it). The list is sorted -> identical across ranks, so whichever
        # os.replace wins yields a complete, correct file and readers never see a partial cache.
        tmp = f"{self.CACHE_FILE}.{os.getpid()}.part"
        with open(tmp, 'w') as f:
            f.write('\n'.join(latent_paths))
        os.replace(tmp, self.CACHE_FILE)
        logging.info(f"****************Generated cache for latent paths: {self.CACHE_FILE}")
        return latent_paths

    def _get_latent_paths(self, root_dir):
        latent_paths = []
        for entry in os.scandir(root_dir):
            if entry.is_dir(follow_symlinks=False):
                with os.scandir(entry.path) as sub:
                    for e in sub:
                        if e.is_file(follow_symlinks=False) and e.name.endswith('.latent.npz'):
                            latent_paths.append(e.path)
        return sorted(latent_paths)  # deterministic across DDP ranks

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx):
        idx, augmentation_seed = _split_dataset_index(idx)
        latent_path = self.latent_paths[idx]
        class_name = os.path.basename(os.path.dirname(latent_path))
        label = self.class_to_idx[class_name]
        use_flip = (
            bool(augmentation_seed & 1)
            if augmentation_seed is not None
            else bool(torch.rand(1) >= 0.5)
        )
        latent_key = 'latent_flip' if use_flip else 'latent'
        # Missing/corrupt latents fail loudly; no silent zero-fill by design.
        with np.load(latent_path) as npz_data:
            latent_z_data = np.array(npz_data[latent_key], copy=True)
        latent_z = torch.from_numpy(latent_z_data)
        return latent_path, label, latent_z


class SeededImageFolder(ImageFolder):
    """Make the existing stochastic image transform stable across restarts."""

    def __getitem__(self, index):
        sample_index, augmentation_seed = _split_dataset_index(index)
        if augmentation_seed is None:
            return super().__getitem__(sample_index)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(augmentation_seed)
            return super().__getitem__(sample_index)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())
    for name, buffer in model_buffers.items():
        ema_buffers[name].copy_(buffer)


class Tee:
    def __init__(self, original_stream, file_stream):
        self.original_stream = original_stream
        self.file_stream = file_stream

    def write(self, message):
        self.original_stream.write(message)
        self.file_stream.write(message)
        self.flush()

    def flush(self):
        self.original_stream.flush()
        self.file_stream.flush()

    def fileno(self):
        return self.original_stream.fileno()

def setup_logging(output_dir, rank):
    os.makedirs(output_dir, exist_ok=True)
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s-%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if rank == 0:
        file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"), mode='a')
        plain_formatter = logging.Formatter('[%(asctime)s-%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)


def format_loss_log(epoch, step, loss_dict):
    parts = [f"epoch {epoch}-step {step}"]
    for name, value in loss_dict.items():
        if name in {"loss", "total_loss"} or not torch.is_tensor(value):
            continue
        parts.append(f"{name}: {value.item():.4f}")
    if "total_loss" in loss_dict and torch.is_tensor(loss_dict["total_loss"]):
        parts.append(f"total_loss: {loss_dict['total_loss'].item():.4f}")
    return " ".join(parts)


def write_loss_dict_to_tensorboard(writer, loss_dict, step):
    if "total_loss" in loss_dict and torch.is_tensor(loss_dict["total_loss"]):
        writer.add_scalar('Loss/train', loss_dict["total_loss"].item(), step)
    for name, value in loss_dict.items():
        if name == "loss" or not torch.is_tensor(value):
            continue
        writer.add_scalar(f'Loss/{name}', value.item(), step)


def accumulate_loss_dict(running_loss_dict, loss_dict):
    if running_loss_dict is None:
        running_loss_dict = {}
    for name, value in loss_dict.items():
        detached_value = value.detach() if torch.is_tensor(value) else value
        if name in running_loss_dict:
            running_loss_dict[name] = running_loss_dict[name] + detached_value
        else:
            running_loss_dict[name] = detached_value
    return running_loss_dict


def average_loss_dict(loss_dict, divisor):
    averaged = {}
    for name, value in loss_dict.items():
        averaged[name] = value / divisor
    return averaged


def _seed_training_rng(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        'python': random.getstate(),
        'numpy': {
            'bit_generator': numpy_state[0],
            'state': torch.from_numpy(
                numpy_state[1].astype(np.int64, copy=True)
            ),
            'position': int(numpy_state[2]),
            'has_gauss': int(numpy_state[3]),
            'cached_gaussian': float(numpy_state[4]),
        },
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state()
    return state


def _numpy_rng_state_array(state):
    if not torch.is_tensor(state):
        raise TypeError("NumPy RNG state vector must be a tensor")
    if state.dtype not in {torch.int64, torch.uint32}:
        raise TypeError("NumPy RNG state vector must use int64 or uint32")
    if state.ndim != 1 or state.numel() == 0:
        raise TypeError("NumPy RNG state must be a nonempty vector")
    state = state.detach().cpu()
    if state.dtype == torch.int64:
        max_uint32 = np.iinfo(np.uint32).max
        if torch.any(state < 0) or torch.any(state > max_uint32):
            raise ValueError("NumPy RNG state vector is outside uint32 range")
    return state.numpy().astype(np.uint32, copy=True)


def _restore_rng_state(state):
    required = {'python', 'numpy', 'torch'}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Checkpoint RNG state is incomplete")
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((
        numpy_state['bit_generator'],
        _numpy_rng_state_array(numpy_state['state']),
        int(numpy_state['position']),
        int(numpy_state['has_gauss']),
        float(numpy_state['cached_gaussian']),
    ))
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state(state['cuda'].cpu())


def _validate_rng_state(state):
    required = {'python', 'numpy', 'torch'}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Checkpoint RNG state is incomplete")
    try:
        python_rng = random.Random()
        python_rng.setstate(state['python'])

        numpy_state = state['numpy']
        if not isinstance(numpy_state, dict):
            raise TypeError("NumPy RNG state must be a mapping")
        numpy_rng = np.random.RandomState()
        numpy_rng.set_state((
            numpy_state['bit_generator'],
            _numpy_rng_state_array(numpy_state['state']),
            int(numpy_state['position']),
            int(numpy_state['has_gauss']),
            float(numpy_state['cached_gaussian']),
        ))

        torch_state = state['torch']
        if (
            not torch.is_tensor(torch_state)
            or torch_state.dtype != torch.uint8
            or torch_state.ndim != 1
            or torch_state.numel() == 0
        ):
            raise TypeError("Torch RNG state must be a nonempty uint8 vector")
        torch.Generator(device='cpu').set_state(torch_state.cpu())

        if torch.cuda.is_available():
            cuda_state = state.get('cuda')
            if (
                not torch.is_tensor(cuda_state)
                or cuda_state.dtype != torch.uint8
                or cuda_state.ndim != 1
                or cuda_state.numel() == 0
            ):
                raise TypeError("CUDA RNG state must be a nonempty uint8 vector")
            cuda_generator = torch.Generator(
                device=f'cuda:{torch.cuda.current_device()}'
            )
            cuda_generator.set_state(cuda_state.cpu())
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Checkpoint RNG state is invalid") from error


def _sampler_position(data_batches_seen, batches_per_epoch):
    if (
        isinstance(data_batches_seen, bool)
        or not isinstance(data_batches_seen, int)
        or data_batches_seen < 0
    ):
        raise ValueError("data_batches_seen must be a non-negative integer")
    if (
        isinstance(batches_per_epoch, bool)
        or not isinstance(batches_per_epoch, int)
        or batches_per_epoch < 1
    ):
        raise ValueError("batches_per_epoch must be a positive integer")
    return divmod(data_batches_seen, batches_per_epoch)


def _legacy_resume_state(
    checkpoint_step,
    grad_mix,
    batches_per_epoch,
    fallback_seed,
):
    next_step = checkpoint_step + 1
    data_batches_seen = next_step * grad_mix
    sampler_epoch, sampler_batch_offset = _sampler_position(
        data_batches_seen,
        batches_per_epoch,
    )
    return {
        'next_step': next_step,
        'data_batches_seen': data_batches_seen,
        'sampler_epoch': sampler_epoch,
        'sampler_batch_offset': sampler_batch_offset,
        'rng_state': None,
        'fallback_seed': int(fallback_seed),
        'legacy_checkpoint': True,
        'legacy_augmentation_state': True,
        'legacy_sampler_seed_state': True,
        'trainer_state_version': None,
    }


def _resume_state_from_checkpoint(
    checkpoint,
    rank,
    world_size,
    grad_mix,
    batches_per_epoch,
    fallback_seed,
    global_seed=0,
    sampler_contract=None,
):
    checkpoint_step = checkpoint.get('step')
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        raise ValueError("Checkpoint step must be a non-negative integer")

    trainer_state = checkpoint.get('trainer_state')
    if trainer_state is None:
        return _legacy_resume_state(
            checkpoint_step,
            grad_mix,
            batches_per_epoch,
            fallback_seed,
        )
    trainer_state_version = trainer_state.get('version')
    if trainer_state_version not in {
        LEGACY_TRAINER_STATE_VERSION,
        TRAINER_STATE_VERSION,
    }:
        raise ValueError("Unsupported checkpoint trainer-state version")
    if (
        trainer_state_version == TRAINER_STATE_VERSION
        and trainer_state.get('augmentation_seed_version')
        != AUGMENTATION_SEED_VERSION
    ):
        raise ValueError("Checkpoint augmentation seed version is incompatible")
    global_seed = _require_nonnegative_index(global_seed, 'global_seed')
    if (
        trainer_state_version == TRAINER_STATE_VERSION
        and trainer_state.get('global_seed') != global_seed
    ):
        raise ValueError("Checkpoint global_seed differs from the current configuration")
    if trainer_state_version == TRAINER_STATE_VERSION:
        _validate_sampler_contract(
            trainer_state.get('sampler_contract'),
            sampler_contract,
        )
    if trainer_state.get('world_size') != world_size:
        raise ValueError(
            "Checkpoint world size differs from the current training world size"
        )
    if trainer_state.get('grad_mix') != grad_mix:
        raise ValueError("Checkpoint grad_mix differs from the current configuration")
    if trainer_state.get('batches_per_epoch') != batches_per_epoch:
        raise ValueError(
            "Checkpoint batches_per_epoch differs from the current dataset"
        )

    next_step = trainer_state.get('next_step')
    data_batches_seen = trainer_state.get('data_batches_seen')
    sampler_epoch = trainer_state.get('sampler_epoch')
    sampler_batch_offset = trainer_state.get('sampler_batch_offset')
    integer_fields = {
        'next_step': next_step,
        'data_batches_seen': data_batches_seen,
        'sampler_epoch': sampler_epoch,
        'sampler_batch_offset': sampler_batch_offset,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_fields.values()
    ):
        raise ValueError("Checkpoint trainer progress must use non-negative integers")
    if next_step != checkpoint_step + 1:
        raise ValueError("Checkpoint next_step is inconsistent with its saved step")
    if data_batches_seen != next_step * grad_mix:
        raise ValueError(
            "Checkpoint data_batches_seen is inconsistent with next_step and grad_mix"
        )
    expected_epoch, expected_offset = _sampler_position(
        data_batches_seen,
        batches_per_epoch,
    )
    if (sampler_epoch, sampler_batch_offset) != (
        expected_epoch,
        expected_offset,
    ):
        raise ValueError("Checkpoint sampler position is internally inconsistent")

    rank_states = trainer_state.get('rank_states')
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise ValueError("Checkpoint rank RNG states are incomplete")
    states_by_rank = {
        state.get('rank'): state
        for state in rank_states
        if isinstance(state, dict)
    }
    if set(states_by_rank) != set(range(world_size)):
        raise ValueError("Checkpoint rank RNG state IDs are invalid")
    for state_rank in range(world_size):
        _validate_rng_state(states_by_rank[state_rank].get('rng_state'))
    return {
        'next_step': next_step,
        'data_batches_seen': data_batches_seen,
        'sampler_epoch': sampler_epoch,
        'sampler_batch_offset': sampler_batch_offset,
        'rng_state': states_by_rank[rank].get('rng_state'),
        'fallback_seed': None,
        'legacy_checkpoint': False,
        'legacy_augmentation_state': (
            trainer_state_version == LEGACY_TRAINER_STATE_VERSION
        ),
        'legacy_sampler_seed_state': (
            trainer_state_version == LEGACY_TRAINER_STATE_VERSION
        ),
        'trainer_state_version': trainer_state_version,
    }


def _fresh_resume_state():
    return {
        'next_step': 0,
        'data_batches_seen': 0,
        'sampler_epoch': 0,
        'sampler_batch_offset': 0,
        'rng_state': None,
        'fallback_seed': None,
        'legacy_checkpoint': False,
        'legacy_augmentation_state': False,
        'legacy_sampler_seed_state': False,
        'trainer_state_version': None,
    }


def _validate_module_state_dict(module, state_dict, state_name):
    if not isinstance(state_dict, dict):
        raise TypeError(f"{state_name} must be a state-dict mapping")
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    type_mismatches = []
    shape_mismatches = []
    dtype_mismatches = []
    layout_mismatches = []
    device_mismatches = []
    for key in set(expected) & set(state_dict):
        expected_value = expected[key]
        saved_value = state_dict[key]
        if torch.is_tensor(expected_value):
            if not torch.is_tensor(saved_value):
                type_mismatches.append(
                    f"{key}: {type(saved_value).__name__} is not Tensor"
                )
                continue
            if tuple(expected_value.shape) != tuple(saved_value.shape):
                shape_mismatches.append(
                    f"{key}: {tuple(saved_value.shape)} != {tuple(expected_value.shape)}"
                )
            if expected_value.dtype != saved_value.dtype:
                dtype_mismatches.append(
                    f"{key}: {saved_value.dtype} != {expected_value.dtype}"
                )
            if expected_value.layout != saved_value.layout:
                layout_mismatches.append(
                    f"{key}: {saved_value.layout} != {expected_value.layout}"
                )
            if saved_value.device.type != 'cpu':
                device_mismatches.append(
                    f"{key}: {saved_value.device} is not a materialized CPU tensor"
                )
        elif type(expected_value) is not type(saved_value):
            type_mismatches.append(
                f"{key}: {type(saved_value).__name__} != "
                f"{type(expected_value).__name__}"
            )
    if (
        missing
        or type_mismatches
        or shape_mismatches
        or dtype_mismatches
        or layout_mismatches
        or device_mismatches
    ):
        raise ValueError(
            f"{state_name} is incompatible: missing={missing[:5]}, "
            f"type_mismatches={type_mismatches[:5]}, "
            f"shape_mismatches={shape_mismatches[:5]}, "
            f"dtype_mismatches={dtype_mismatches[:5]}, "
            f"layout_mismatches={layout_mismatches[:5]}, "
            f"device_mismatches={device_mismatches[:5]}"
        )

    return sorted(set(state_dict) - set(expected))


def _optimizer_group_value_matches(saved_value, current_value):
    if torch.is_tensor(saved_value) or torch.is_tensor(current_value):
        return (
            torch.is_tensor(saved_value)
            and torch.is_tensor(current_value)
            and saved_value.shape == current_value.shape
            and saved_value.dtype == current_value.dtype
            and torch.equal(saved_value.cpu(), current_value.cpu())
        )
    if type(saved_value) is not type(current_value):
        return False
    if isinstance(saved_value, dict):
        return (
            set(saved_value) == set(current_value)
            and all(
                _optimizer_group_value_matches(
                    saved_value[key],
                    current_value[key],
                )
                for key in saved_value
            )
        )
    if isinstance(saved_value, (list, tuple)):
        return (
            len(saved_value) == len(current_value)
            and all(
                _optimizer_group_value_matches(saved, current)
                for saved, current in zip(saved_value, current_value)
            )
        )
    return saved_value == current_value


def _validate_optimizer_state_dict(optimizer, state_dict):
    if not isinstance(state_dict, dict):
        raise TypeError("optimizer_state_dict must be a mapping")
    saved_groups = state_dict.get('param_groups')
    if not isinstance(saved_groups, list):
        raise ValueError("optimizer_state_dict has no param_groups list")
    current_groups = optimizer.state_dict()['param_groups']
    if len(saved_groups) != len(current_groups):
        raise ValueError("Optimizer parameter-group count differs")
    for group_index, (saved_group, current_group) in enumerate(
        zip(saved_groups, current_groups)
    ):
        if not isinstance(saved_group, dict) or not isinstance(current_group, dict):
            raise ValueError("Optimizer parameter groups must be mappings")
        if len(saved_group.get('params', [])) != len(current_group.get('params', [])):
            raise ValueError("Optimizer parameter-group sizes differ")
        saved_options = {
            key: value for key, value in saved_group.items() if key != 'params'
        }
        current_options = {
            key: value for key, value in current_group.items() if key != 'params'
        }
        if not _optimizer_group_value_matches(saved_options, current_options):
            raise ValueError(
                f"Optimizer parameter-group {group_index} options differ"
            )

    saved_state = state_dict.get('state')
    if not isinstance(saved_state, dict):
        raise ValueError("optimizer_state_dict has no state mapping")
    if not saved_state:
        raise ValueError("optimizer_state_dict state mapping is empty")
    parameter_by_saved_id = {}
    amsgrad_by_saved_id = {}
    for saved_group, live_group in zip(saved_groups, optimizer.param_groups):
        saved_ids = saved_group.get('params')
        live_parameters = live_group.get('params')
        if not isinstance(saved_ids, list) or not isinstance(live_parameters, list):
            raise ValueError("Optimizer parameter groups are malformed")
        for saved_id, parameter in zip(saved_ids, live_parameters):
            if isinstance(saved_id, bool) or not isinstance(saved_id, int):
                raise ValueError("Optimizer parameter IDs must be integers")
            if saved_id in parameter_by_saved_id:
                raise ValueError("Optimizer parameter IDs must be unique")
            parameter_by_saved_id[saved_id] = parameter
            amsgrad_by_saved_id[saved_id] = bool(saved_group['amsgrad'])

    invalid_state_ids = [
        state_id
        for state_id in saved_state
        if isinstance(state_id, bool) or not isinstance(state_id, int)
    ]
    if invalid_state_ids:
        raise ValueError("Optimizer state parameter IDs must be integers")
    missing_state_ids = sorted(set(parameter_by_saved_id) - set(saved_state))
    unknown_state_ids = sorted(set(saved_state) - set(parameter_by_saved_id))
    if missing_state_ids or unknown_state_ids:
        raise ValueError(
            "Optimizer state parameter coverage differs: "
            f"missing={missing_state_ids[:5]}, extra={unknown_state_ids[:5]}"
        )
    base_state_keys = {'step', 'exp_avg', 'exp_avg_sq'}
    for saved_id, parameter_state in saved_state.items():
        if not isinstance(parameter_state, dict):
            raise ValueError(
                f"Optimizer state for parameter {saved_id} must be a mapping"
            )
        expected_state_keys = set(base_state_keys)
        if amsgrad_by_saved_id[saved_id]:
            expected_state_keys.add('max_exp_avg_sq')
        if set(parameter_state) != expected_state_keys:
            raise ValueError(
                f"Optimizer state keys for parameter {saved_id} differ: "
                f"saved={sorted(parameter_state)}, "
                f"expected={sorted(expected_state_keys)}"
            )
        if not parameter_state:
            raise ValueError(
                f"Optimizer state for parameter {saved_id} is empty"
            )
        step_value = parameter_state['step']
        if torch.is_tensor(step_value):
            if (
                step_value.layout != torch.strided
                or step_value.numel() != 1
                or step_value.dtype == torch.bool
                or step_value.is_complex()
            ):
                raise ValueError(
                    f"Optimizer step for parameter {saved_id} must be a real scalar"
                )
            numeric_step = step_value.item()
        elif isinstance(step_value, numbers.Real) and not isinstance(step_value, bool):
            numeric_step = step_value
        else:
            raise ValueError(
                f"Optimizer step for parameter {saved_id} has invalid type"
            )
        if (
            not math.isfinite(float(numeric_step))
            or numeric_step < 0
            or int(numeric_step) != numeric_step
        ):
            raise ValueError(
                f"Optimizer step for parameter {saved_id} must be non-negative "
                "and integral"
            )

        parameter = parameter_by_saved_id[saved_id]
        for state_name in expected_state_keys - {'step'}:
            if state_name not in parameter_state:
                continue
            state_tensor = parameter_state[state_name]
            if not torch.is_tensor(state_tensor):
                raise ValueError(
                    f"Optimizer {state_name} for parameter {saved_id} is not Tensor"
                )
            if (
                state_tensor.layout != torch.strided
                or tuple(state_tensor.shape) != tuple(parameter.shape)
            ):
                raise ValueError(
                    f"Optimizer {state_name} shape for parameter {saved_id} "
                    f"is {tuple(state_tensor.shape)}, expected {tuple(parameter.shape)}"
                )
            if state_tensor.dtype != parameter.dtype:
                raise ValueError(
                    f"Optimizer {state_name} dtype for parameter {saved_id} "
                    f"is {state_tensor.dtype}, expected {parameter.dtype}"
                )
            if state_tensor.device.type not in {'cpu', parameter.device.type}:
                raise ValueError(
                    f"Optimizer {state_name} device for parameter {saved_id} "
                    f"is incompatible: {state_tensor.device}"
                )
def load_latest_checkpoint(
    model,
    ema_model,
    optimizer,
    checkpoint_dir='checkpoints',
    resume_checkpoint_step=None,
    *,
    rank=0,
    world_size=1,
    grad_mix=1,
    batches_per_epoch=1,
    fallback_seed=0,
    global_seed=0,
    sampler_contract=None,
):
    if resume_checkpoint_step is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_step_{resume_checkpoint_step}.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Specified checkpoint not found: {checkpoint_path}"
            )
        checkpoints_to_try = [checkpoint_path]
    else:
        checkpoints_to_try = sorted(
            glob.glob(os.path.join(checkpoint_dir, 'ckpt_step_*.pth')), 
            key=os.path.getmtime, 
            reverse=True
        )
        if not checkpoints_to_try:
            logging.error(f"No checkpoints found in directory: {checkpoint_dir}")
            return _fresh_resume_state()
    
    prepared = None
    for i, checkpoint_path in enumerate(checkpoints_to_try):
        try:
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=False,
            )
            
            resume_state = _resume_state_from_checkpoint(
                checkpoint,
                rank,
                world_size,
                grad_mix,
                batches_per_epoch,
                fallback_seed,
                global_seed,
                sampler_contract,
            )
            model_state = checkpoint.get('model_state_dict')
            ema_state = checkpoint.get('ema_model_state_dict')
            optimizer_state = checkpoint.get('optimizer_state_dict')
            missing_states = [
                name
                for name, state in (
                    ('model_state_dict', model_state),
                    ('ema_model_state_dict', ema_state),
                    ('optimizer_state_dict', optimizer_state),
                )
                if state is None
            ]
            if missing_states:
                raise ValueError(
                    f"Checkpoint is missing required training states: {missing_states}"
                )
            unexpected_keys = _validate_module_state_dict(
                model,
                model_state,
                'model_state_dict',
            )
            unexpected_ema = _validate_module_state_dict(
                ema_model,
                ema_state,
                'ema_model_state_dict',
            )
            _validate_optimizer_state_dict(optimizer, optimizer_state)
            prepared = {
                'checkpoint': checkpoint,
                'resume_state': resume_state,
                'model_state': model_state,
                'ema_state': ema_state,
                'optimizer_state': optimizer_state,
                'unexpected_keys': unexpected_keys,
                'unexpected_ema': unexpected_ema,
            }
            break
        
        except Exception as e:
            error_msg = f"Failed to load checkpoint {checkpoint_path}: {str(e)}"
            if len(checkpoints_to_try) > 1:
                error_msg += f" (attempt {i+1}/{len(checkpoints_to_try)})"
            logging.error(error_msg)
            
            import traceback
            logging.debug(traceback.format_exc())
            
            if resume_checkpoint_step is not None:
                raise RuntimeError(error_msg) from e

    if prepared is None:
        raise RuntimeError("Could not load any checkpoint without mutating training state")

    # All fallible candidate checks happened on copies. From here on, any
    # unexpected commit failure terminates instead of falling back with a
    # partially mutated model.
    missing_keys, committed_unexpected = model.load_state_dict(
        prepared['model_state'],
        strict=False,
    )
    missing_ema, committed_unexpected_ema = ema_model.load_state_dict(
        prepared['ema_state'],
        strict=False,
    )
    if missing_keys or missing_ema:
        raise RuntimeError("Validated checkpoint changed during state commit")
    if committed_unexpected != prepared['unexpected_keys']:
        raise RuntimeError("Model unexpected-key set changed during state commit")
    if committed_unexpected_ema != prepared['unexpected_ema']:
        raise RuntimeError("EMA unexpected-key set changed during state commit")
    if committed_unexpected:
        logging.warning(
            f"Ignoring unexpected model checkpoint keys: {committed_unexpected[:5]}"
        )
    if committed_unexpected_ema:
        logging.warning(
            "Ignoring unexpected EMA checkpoint keys: "
            f"{committed_unexpected_ema[:5]}"
        )
    optimizer.load_state_dict(prepared['optimizer_state'])
    logging.info("EMA model loaded")
    logging.info("Optimizer loaded")
    checkpoint = prepared['checkpoint']
    resume_state = prepared['resume_state']
    logging.info(
        f"✓ Successfully loaded checkpoint from step "
        f"{checkpoint['step']}; next step is {resume_state['next_step']}"
    )
    return resume_state


def save_checkpoint(
    model,
    ema_model,
    optimizer,
    step,
    trainer_progress,
    checkpoint_dir='checkpoints',
    *,
    global_seed=0,
    sampler_contract=None,
):
    global_seed = _require_nonnegative_index(global_seed, 'global_seed')
    if not isinstance(sampler_contract, dict):
        raise ValueError("sampler_contract must be provided when saving a checkpoint")
    if sampler_contract.get('global_seed') != global_seed:
        raise ValueError("sampler_contract global_seed is inconsistent")
    local_state = {
        'rank': dist.get_rank() if dist.is_initialized() else 0,
        'rng_state': _capture_rng_state(),
    }
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank_states = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(rank_states, local_state)
    else:
        rank_states[0] = local_state

    if local_state['rank'] != 0:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_step_{step}.pth')
    temporary_path = checkpoint_path + '.tmp'
    trainer_state = {
        **trainer_progress,
        'version': TRAINER_STATE_VERSION,
        'augmentation_seed_version': AUGMENTATION_SEED_VERSION,
        'global_seed': global_seed,
        'sampler_contract': copy.deepcopy(sampler_contract),
        'world_size': world_size,
        'rank_states': sorted(rank_states, key=lambda state: state['rank']),
    }
    torch.save({
        'step': step,
        'model_state_dict': model.module.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'trainer_state': trainer_state,
    }, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    logging.info(f'********************* Checkpoint saved at {checkpoint_path}')


def center_crop_lambda(pil_image):
    return center_crop_arr(pil_image, cfg.image_size)

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def get_sigmas_timesteps(u, shift, num_train_timesteps, n_dim=4, dtype=torch.float32):
    sigma = (shift * u / (1 + (shift - 1) * u)).to(dtype=dtype)
    # timesteps
    timesteps = (sigma * num_train_timesteps).to(dtype=dtype)
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    
    return timesteps, sigma

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = 0.0, logit_std: float = 1.0, sigmoid_scale: float = 1.0, mode_scale: float = 1.29, generator=None, device='cpu'
):
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), generator=generator, device=device)
        u = u * sigmoid_scale
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), generator=generator, device=device)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), generator=generator, device=device)
    return u


def main(**kwargs):
    deep_update(cfg, kwargs)
    
    if 'gpu_ids' in kwargs and kwargs['gpu_ids'] is not None:
        gpu_ids = ','.join(map(str, kwargs['gpu_ids']))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"Set CUDA_VISIBLE_DEVICES to {gpu_ids}")
    
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()

    cfg.pmi_rank = int(os.getenv('RANK', 0))
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    print(f"WORLD_SIZE: {cfg.pmi_world_size}")

    if 'gpu_ids' in kwargs and kwargs['gpu_ids'] is not None:
        cfg.gpus_per_machine = len(kwargs['gpu_ids'])
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
    cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, ))
    return cfg


def worker(gpu, cfg):
    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    print(f"Rank {cfg.rank} is working on GPU {gpu}")

    # init distributed processes
    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        rank=cfg.rank,
        world_size=cfg.world_size,
        timeout=datetime.timedelta(hours=5)
    )

    cfg.output_dir = osp.join(cfg.output_dir, cfg.model_name, cfg.custom_cfg_name)
    setup_logging(cfg.output_dir, cfg.rank)

    global_seed = int(getattr(cfg, 'global_seed', 0))
    if global_seed < 0:
        raise ValueError("global_seed must be non-negative")
    cfg.seed = global_seed * cfg.world_size + cfg.rank
    _seed_training_rng(cfg.seed)
    logging.info(
        f"Training RNG seed: {cfg.seed} "
        f"(global_seed={global_seed}, rank={cfg.rank}, world_size={cfg.world_size})"
    )

    denoising_regret_coeff = float(
        getattr(cfg, 'denoising_regret_coeff', 0.0)
    )
    if denoising_regret_coeff < 0:
        raise ValueError("denoising_regret_coeff must be non-negative")
    if (
        denoising_regret_coeff > 0
        and cfg.model_name not in DENOISING_REGRET_MODELS
    ):
        raise ValueError(
            f"denoising_regret_coeff is not supported by {cfg.model_name}"
        )

    if cfg.param_dtype == torch.bfloat16:
        use_amp = True
        logging.info("Training with bfloat16 mixed precision.")
    else:
        use_amp = False

    if cfg.rank == 0:
        writer = SummaryWriter(log_dir=osp.join(cfg.output_dir, "tensorboard"))
    
    cfg.train_img_num = getattr(cfg, 'train_img_num', None)
    cfg.grad_mix = max(int(getattr(cfg, 'grad_mix', 1)), 1)
    
    data_path = cfg.data_path
    if cfg.use_pre_latents:
        if getattr(cfg, 'use_encoded_latents', False):
            # read pre-encoded latents directly (encode_latents_from_parquet.py output);
            # no image folder, no str.replace('train', ...) derivation.
            img_dataset = LatentFolder(cfg.latent_data_path, cfg=cfg)
        else:
            img_dataset = CustomImageFolder(data_path, cfg=cfg)
    else:
        transform = transforms.Compose([
            transforms.Lambda(center_crop_lambda),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        img_dataset = SeededImageFolder(data_path, transform=transform)

    cfg.total_train_batch_size = getattr(cfg, 'total_train_batch_size', 256)
    cfg.train_batch_size = cfg.total_train_batch_size // cfg.world_size
    use_structured = bool(getattr(cfg, 'structured_batch_sampling', False))
    if use_structured:
        from utils import StructuredDistributedBatchSampler
        struct_batch_sampler = StructuredDistributedBatchSampler(
            dataset=img_dataset,
            batch_size=cfg.train_batch_size,
            num_replicas=cfg.world_size,
            rank=cfg.rank,
            case1_prob=float(getattr(cfg, 'structured_batch_case1_prob', 0.5)),
            seed=global_seed,
        )
        sampler_contract = _build_sampler_contract(
            img_dataset,
            sampler_type='structured_distributed',
            global_seed=global_seed,
            per_rank_batch_size=cfg.train_batch_size,
            case1_prob=struct_batch_sampler.case1_prob,
        )
        logging.info(
            f"Structured batch sampling ENABLED (case1_prob="
            f"{struct_batch_sampler.case1_prob}); per-rank batches per epoch="
            f"{len(struct_batch_sampler)}"
        )
        resumable_batch_sampler = ResumableBatchSampler(
            struct_batch_sampler,
            seed=global_seed,
            rank=cfg.rank,
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(cfg.seed + 0x5EEDDA7A)
        image_dataloader = DataLoader(
            img_dataset,
            batch_sampler=resumable_batch_sampler,
            num_workers=cfg.img_num_workers,
            pin_memory=True,
            prefetch_factor=cfg.prefetch_factor,
            persistent_workers=True,
            generator=loader_generator,
        )
    else:
        distributed_sampler = DistributedSampler(
            img_dataset,
            num_replicas=cfg.world_size,
            rank=cfg.rank,
            seed=global_seed
        )
        sampler_contract = _build_sampler_contract(
            img_dataset,
            sampler_type='distributed',
            global_seed=global_seed,
            per_rank_batch_size=cfg.train_batch_size,
        )
        standard_batch_sampler = BatchSampler(
            distributed_sampler,
            batch_size=cfg.train_batch_size,
            drop_last=False,
        )
        resumable_batch_sampler = ResumableBatchSampler(
            standard_batch_sampler,
            seed=global_seed,
            rank=cfg.rank,
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(cfg.seed + 0x5EEDDA7A)
        image_dataloader = DataLoader(
            img_dataset,
            batch_sampler=resumable_batch_sampler,
            num_workers=cfg.img_num_workers,
            pin_memory=True,
            prefetch_factor=cfg.prefetch_factor,
            persistent_workers=True,
            generator=loader_generator,
        )
    batches_per_epoch = resumable_batch_sampler.full_length

    total_images = len(img_dataset)
    batch_size = cfg.train_batch_size
    steps_per_epoch = total_images // batch_size
    if total_images % batch_size != 0:
        steps_per_epoch += 1
    logging.info(f"----------------------Image Num {total_images} , Total number of steps per epoch: {steps_per_epoch // cfg.world_size}")
    
    logging.info('Initializing VAE')
    vae_path = getattr(cfg, 'vae_path', None)
    if not cfg.use_pre_latents:
        if cfg.rank == 0:
            load_vae(cfg.sd_vae_ft_mse_vae_path, vae_path=vae_path)
        dist.barrier()
        vae = load_vae(cfg.sd_vae_ft_mse_vae_path, vae_path=vae_path)
        vae = vae.eval().to(gpu)

        for param in vae.parameters():
            param.requires_grad = False

    logging.info('Initializing transformer models (non-ema and ema)')
    model_class, config_name = model_dict[cfg.model_name]
    model_cfg = getattr(cfg, config_name)
    logging.info(f'model_cfg: {model_cfg}')
    model = model_class(**model_cfg)
    model = model.to(gpu)
    model_ema = copy.deepcopy(model).eval().requires_grad_(False)

    # [model] mark model size
    model_size = sum([p.numel() for p in model.parameters()]) / (1024 ** 2)
    logging.info(f'Created models with {model_size:.3f} M parameters')

    # [optim] optimizer (only include parameters that require gradients;
    # e.g. noise_expert_ema model freezes noise_expert params)
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params=train_params,
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
        fused=True
    )
    scaler = amp.GradScaler(enabled=False)

    for para_id, (name, param) in enumerate(model.named_parameters()):
        logging.info(f"Train parameter {para_id}: {name} (requires_grad={param.requires_grad})")

    cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoints')
    if cfg.resume_checkpoint:
        cfg.resume_checkpoint_step = getattr(cfg, 'resume_checkpoint_step', None)
        resume_state = load_latest_checkpoint(
            model,
            model_ema,
            optimizer,
            os.path.join(cfg.checkpoint_dir),
            cfg.resume_checkpoint_step,
            rank=cfg.rank,
            world_size=cfg.world_size,
            grad_mix=cfg.grad_mix,
            batches_per_epoch=batches_per_epoch,
            fallback_seed=(
                cfg.seed + (int(cfg.resume_checkpoint_step or 0) + 1) * 1000003
            ),
            global_seed=global_seed,
            sampler_contract=sampler_contract,
        )
    else:
        resume_state = _fresh_resume_state()

    step = resume_state['next_step']
    epoch = resume_state['sampler_epoch']
    data_batches_seen = resume_state['data_batches_seen']
    sampler_batch_offset = resume_state['sampler_batch_offset']
    resumable_batch_sampler.set_epoch(epoch)
    resumable_batch_sampler.set_start_batch(sampler_batch_offset)
    image_rank_iter = iter(image_dataloader)

    model = DistributedDataParallel(model, device_ids=[gpu])

    # Restore only after every restart-only initialization. DataLoader has its
    # own generator, while sample augmentations derive from sampler positions.
    if resume_state['rng_state'] is not None:
        _restore_rng_state(resume_state['rng_state'])
        logging.info("Restored per-rank training RNG state from checkpoint")
    elif resume_state['legacy_checkpoint']:
        fallback_seed = (
            cfg.seed + resume_state['next_step'] * 1000003
        )
        _seed_training_rng(fallback_seed)
        logging.warning(
            "Legacy checkpoint has no trainer RNG state; using deterministic "
            f"non-replay fallback seed {fallback_seed}"
        )
    if resume_state['legacy_augmentation_state']:
        logging.warning(
            "Trainer-state v1 used worker-local augmentation RNG; resumed "
            "samples now use deterministic sampler-position seeds"
        )
    if resume_state['legacy_sampler_seed_state']:
        logging.warning(
            "Legacy checkpoint did not record the sampler seed/mode/data "
            "contract; exact sample-order replay cannot be guaranteed"
        )
    logging.info(
        f"Resume progress: next_step={step}, data_batches_seen="
        f"{data_batches_seen}, sampler_epoch={epoch}, "
        f"sampler_batch_offset={sampler_batch_offset}"
    )

    model.train()
    model_ema.eval()
    optimizer.zero_grad(set_to_none=True)
    
    logging.info('Start the training loop')

    accum_steps = 0
    accum_loss_dict = None
    while step < cfg.num_steps:
        # read batch
        try:
            img_batch = next(image_rank_iter)
        except StopIteration:
            epoch += 1
            logging.info("!!!!!!!!!!!!! reload image_dataloader")
            sampler_batch_offset = 0
            resumable_batch_sampler.set_epoch(epoch)
            resumable_batch_sampler.set_start_batch(0)
            image_rank_iter = iter(image_dataloader)
            img_batch = next(image_rank_iter)
        sampler_batch_offset += 1
        data_batches_seen += 1

        if cfg.use_pre_latents:
            rank_img_paths, rank_img_y, rank_img_z = img_batch
            rank_img_y, rank_img_z = rank_img_y.to(gpu, non_blocking=True), rank_img_z.to(gpu, non_blocking=True)
            rank_img_z_is_all_zero = torch.all(rank_img_z == 0).item()
            assert not rank_img_z_is_all_zero, "error: rank_img_z is all zero"
        else:
            rank_images, rank_img_y = img_batch
            rank_images, rank_img_y = rank_images.to(gpu, non_blocking=True), rank_img_y.to(gpu, non_blocking=True)
            rank_images = rearrange(rank_images, "B C H W -> B C 1 H W")

        rank_img_u = compute_density_for_timestep_sampling(
            weighting_scheme=cfg.weighting_scheme,
            batch_size=len(rank_img_y),
            logit_mean=cfg.logit_mean,
            logit_std=cfg.logit_std,
            sigmoid_scale=cfg.sigmoid_scale,
            mode_scale=cfg.mode_scale,
            generator=None,
            device=gpu
        )

        rank_img_t, rank_img_sigma = get_sigmas_timesteps(rank_img_u, cfg.shift, cfg.num_train_timesteps, n_dim=4)

        # Structured batch sampling: in case 1 (random batch — labels differ),
        # broadcast a single timestep across the batch so it varies in class
        # but is fixed in t. Case 2 (all labels equal) keeps per-sample t.
        if use_structured and not torch.all(rank_img_y == rank_img_y[0]):
            rank_img_t = rank_img_t[:1].expand_as(rank_img_t).clone()
            rank_img_sigma = rank_img_sigma[:1].expand_as(rank_img_sigma).clone()

        ################################# VAE preprocess
        if cfg.use_pre_latents:
            posterior = DiagonalGaussianDistribution(rank_img_z)
            rank_img_z = posterior.sample().mul_(0.18215)
            rank_img_z = rearrange(rank_img_z, "B C H W -> B C 1 H W") # [B, 4, 1, 32, 32] img 256x256
        else:
            rank_images = rearrange(rank_images, "B C 1 H W -> B C H W")
            with torch.no_grad():
                rank_img_z = vae.encode(rank_images).latent_dist.sample().mul_(0.18215)
            rank_img_z = rearrange(rank_img_z, "B C H W -> B C 1 H W") # [B, 4, 1, 32, 32] img 256x256
        ################################# VAE preprocess
        context = rank_img_y
        t, sigmas, z = rank_img_t, rank_img_sigma, rank_img_z
        
        arg_c = {'context': context, 'use_gradient_checkpointing': cfg.use_gradient_checkpointing}

        noise = torch.randn_like(z)
        target = noise - z
        noised_z_in = (1.0 - sigmas.squeeze()).view(z.shape[0], 1, 1, 1, 1) * z + sigmas.squeeze().view(z.shape[0], 1, 1, 1, 1) * noise

        if cfg.model_name in DENOISING_REGRET_MODELS:
            arg_c['denoising_target'] = target
            arg_c['training_step'] = step

        with amp.autocast(dtype=cfg.param_dtype, enabled=use_amp):
            model_output = model(noised_z_in, t, **arg_c)
        
        loss_dict = {}
        loss_dict["loss"] = 0
        if cfg.model_name in DENOISING_REGRET_MODELS:
            if not isinstance(model_output, tuple) or len(model_output) != 2:
                raise ValueError(
                    f"{cfg.model_name} must return (prediction, regret_loss) "
                    "during training"
                )
            model_pred, denoising_regret_loss = model_output
            if not torch.is_tensor(denoising_regret_loss):
                raise TypeError("denoising_regret_loss must be a tensor")

            if model_pred.shape[1] != noised_z_in.shape[1]:
                model_pred, _ = model_pred.chunk(2, dim=1)
            model_pred = model_pred.unsqueeze(2)

            loss_dict["denoising_regret_loss"] = denoising_regret_loss
            denoising_regret_loss_weighted = (
                denoising_regret_loss * denoising_regret_coeff
            )
            loss_dict["denoising_regret_loss_weighted"] = \
                denoising_regret_loss_weighted
            loss_dict["loss"] += denoising_regret_loss_weighted
            for stat_name, stat_value in getattr(
                model.module, 'denoising_regret_stats', {}
            ).items():
                loss_dict[f"denoising_regret_{stat_name}"] = stat_value
        elif isinstance(model_output, tuple):
            loss_dict["cp_loss"] = 0
            ########## DiffMoE loss
            loss_stratgy_name = model_output[1]
            if loss_stratgy_name == "Capacity_Pred":
                layer_idx_list, ones_list, pred_c_list, CapacityPred_loss_weight = model_output[2:]
                for layer_idx, ones, pred_c in zip(layer_idx_list, ones_list, pred_c_list):
                    loss_dict[f"Capacity_Pred_loss_{layer_idx}"] = nn.BCEWithLogitsLoss()(pred_c, ones)
                    loss_dict["loss"] += loss_dict[f"Capacity_Pred_loss_{layer_idx}"]  * CapacityPred_loss_weight
                    loss_dict["cp_loss"] += loss_dict[f"Capacity_Pred_loss_{layer_idx}"]  * CapacityPred_loss_weight
            else:
                raise Exception("not defined training loss")

            model_pred = model_output[0]
            if model_pred.shape[1] != noised_z_in.shape[1]:
                model_pred, _ = model_pred.chunk(2, dim=1)

            model_pred = model_pred.unsqueeze(2)
        elif model_output.shape[1] != noised_z_in.shape[1]:
            ########## DiT loss
            model_pred, _ = model_output.chunk(2, dim=1)
            model_pred = model_pred.unsqueeze(2)
        else:
            model_pred = model_output

        mse_loss = (model_pred - target) ** 2
        mse_loss = torch.stack([u.mean() for u in mse_loss])
        mse_loss = sum(mse_loss) / len(mse_loss)

        loss_dict["mse_loss"] = mse_loss
        loss_dict["loss"] += mse_loss

        loss = loss_dict["loss"].mean()
        loss_dict["total_loss"] = loss
        accum_loss_dict = accumulate_loss_dict(accum_loss_dict, loss_dict)
        accum_steps += 1

        scaler.scale(loss / cfg.grad_mix).backward()
        if accum_steps < cfg.grad_mix:
            continue

        logged_loss_dict = average_loss_dict(accum_loss_dict, accum_steps)
        if step % cfg.log_interval == 0:
            logging.info(format_loss_log(epoch, step, logged_loss_dict))
        if cfg.rank == 0:
            write_loss_dict_to_tensorboard(writer, logged_loss_dict, step)
            if step % cfg.log_interval == 0:
                # lsreg: log realized mean label-smoothing epsilon (for fixed-vs-dynamic deconfounding);
                # no-op for models whose blocks don't expose last_mean_eps.
                _ls_eps = [m.last_mean_eps for m in model.module.modules()
                           if getattr(m, "last_mean_eps", None) is not None]
                if _ls_eps:
                    writer.add_scalar('lsreg/mean_eps', float(torch.stack(_ls_eps).mean()), step)

        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        # Update noise expert as EMA of shared expert (if model supports it)
        if hasattr(model.module, 'update_noise_expert_ema'):
            model.module.update_noise_expert_ema()
        optimizer.zero_grad(set_to_none=True)
        update_ema(model_ema, model.module)

        if step != 0 and step % cfg.save_ckpt_interval == 0:
            checkpoint_epoch, checkpoint_offset = _sampler_position(
                data_batches_seen,
                batches_per_epoch,
            )
            save_checkpoint(
                model,
                model_ema,
                optimizer,
                step,
                {
                    'next_step': step + 1,
                    'data_batches_seen': data_batches_seen,
                    'sampler_epoch': checkpoint_epoch,
                    'sampler_batch_offset': checkpoint_offset,
                    'grad_mix': cfg.grad_mix,
                    'batches_per_epoch': batches_per_epoch,
                },
                cfg.checkpoint_dir,
                global_seed=global_seed,
                sampler_contract=sampler_contract,
            )

        accum_steps = 0
        accum_loss_dict = None
        step += 1

    if cfg.rank == 0:
        logging.info('Congratulations! The training is completed!')
        writer.close()
    
    # barrier to ensure all ranks are completed
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoE')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--vae-path', type=str, default=None,
                        help='Local path to a pretrained VAE directory (skip auto-download)')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        custom_cfg = yaml.safe_load(file)

    custom_cfg['custom_cfg_name'] = osp.splitext(osp.basename(args.config))[0]
    if args.vae_path is not None:
        custom_cfg['vae_path'] = args.vae_path
    main(**custom_cfg)
