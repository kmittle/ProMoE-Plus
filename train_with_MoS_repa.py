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
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
from utils import deep_update, find_free_port, load_vae, TrainingMonitor
from torch.nn.utils import clip_grad_norm_

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

from config import cfg
from models.models_DiT import DiT as DiT
from models.models_ProMoE_TC_repa_MoS import DiT as ProMoE_TC_REPA_MoS_DiT
from models.models_ProMoE_TC_repa_MoS_naive import DiT as ProMoE_TC_REPA_MoS_Naive_DiT
from models.models_ProMoE_TC_repa_MoS_naive_choice import DiT as ProMoE_TC_REPA_MoS_Naive_Choice_DiT
from models.models_ProMoE_TC_repa_MoS_naive_choice_ import DiT as ProMoE_TC_REPA_MoS_Naive_Choice_Sep_DiT
from models.models_ProMoE_TC_repa_multi_align import DiT as ProMoE_TC_REPA_Multi_Align_DiT
from models.models_ProMoE_TC_repa_multi_align_affinity import DiT as ProMoE_TC_REPA_Multi_Align_Affinity_DiT
from models.models_ProMoE_TC_repa_MoS_choice_per_block import DiT as ProMoE_TC_REPA_MoS_Choice_PerBlock_DiT
from models.models_ProMoE_TC_repa_MoS_naive_choice_blockwise import DiT as ProMoE_TC_REPA_MoS_Naive_Choice_Blockwise_DiT
from models.models_ProMoE_TC_repa_cross_global_pre import DiT as ProMoE_TC_REPA_CrossGlobalPre_DiT
from models.models_ProMoE_TC_repa_cross_global_block import DiT as ProMoE_TC_REPA_CrossGlobalBlock_DiT
from models.models_ProMoE_TC_repa_cross_expert_local import DiT as ProMoE_TC_REPA_CrossExpertLocal_DiT
from models.models_ProMoE_TC_repa_cross_proto import DiT as ProMoE_TC_REPA_CrossProto_DiT
from models.models_ProMoE_TC_repa_MoS_naive_choice_cross_global_pre import DiT as ProMoE_TC_REPA_MoS_CrossGlobalPre_DiT
from models.models_ProMoE_TC_repa_MoS_naive_choice_cross_global_block import DiT as ProMoE_TC_REPA_MoS_CrossGlobalBlock_DiT
from models.models_ProMoE_TC_repa_MoS_naive_choice_cross_expert_local import DiT as ProMoE_TC_REPA_MoS_CrossExpertLocal_DiT
from models.models_ProMoE_TC_repa_MoS_naive_choice_cross_proto import DiT as ProMoE_TC_REPA_MoS_CrossProto_DiT
from models.models_ProMoE_TC_repa_MoS_naive_choice_fused import DiT as ProMoE_TC_REPA_MoS_Naive_Choice_Fused_DiT
from repa.encoder import load_teacher_encoder, extract_all_teacher_block_features, get_num_teacher_blocks
from repa.loss import compute_repa_loss

model_dict = {
    "ProMoE_TC_REPA_MoS_B": (ProMoE_TC_REPA_MoS_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_MoS_L": (ProMoE_TC_REPA_MoS_DiT, "DiT_L_config"),
    "ProMoE_TC_REPA_MoS_Naive_B": (ProMoE_TC_REPA_MoS_Naive_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_MoS_Naive_L": (ProMoE_TC_REPA_MoS_Naive_DiT, "DiT_L_config"),
    "ProMoE_TC_REPA_MoS_Naive_Choice_B": (ProMoE_TC_REPA_MoS_Naive_Choice_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_MoS_Naive_Choice_L": (ProMoE_TC_REPA_MoS_Naive_Choice_DiT, "DiT_L_config"),
    "ProMoE_TC_REPA_MoS_Naive_Choice_XL": (ProMoE_TC_REPA_MoS_Naive_Choice_DiT, "DiT_XL_config"),
    "ProMoE_TC_REPA_MoS_Naive_Choice_Sep_B": (ProMoE_TC_REPA_MoS_Naive_Choice_Sep_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_Multi_Align_B": (ProMoE_TC_REPA_Multi_Align_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_Multi_Align_Affinity_B": (ProMoE_TC_REPA_Multi_Align_Affinity_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_MoS_Choice_PerBlock_B": (ProMoE_TC_REPA_MoS_Choice_PerBlock_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_MoS_Naive_Choice_Blockwise_B": (ProMoE_TC_REPA_MoS_Naive_Choice_Blockwise_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_CROSS_GLOBAL_PRE_B": (ProMoE_TC_REPA_CrossGlobalPre_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_CROSS_GLOBAL_BLOCK_B": (ProMoE_TC_REPA_CrossGlobalBlock_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_CROSS_EXPERT_LOCAL_B": (ProMoE_TC_REPA_CrossExpertLocal_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_CROSS_PROTO_B": (ProMoE_TC_REPA_CrossProto_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_MoS_CROSS_GLOBAL_PRE_B": (ProMoE_TC_REPA_MoS_CrossGlobalPre_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_MoS_CROSS_GLOBAL_BLOCK_B": (ProMoE_TC_REPA_MoS_CrossGlobalBlock_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_MoS_CROSS_EXPERT_LOCAL_B": (ProMoE_TC_REPA_MoS_CrossExpertLocal_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_MoS_CROSS_PROTO_B": (ProMoE_TC_REPA_MoS_CrossProto_DiT, "DiT_B_config"),
    "ProMoE_TC_REPA_MoS_Naive_Choice_Fused_B": (ProMoE_TC_REPA_MoS_Naive_Choice_Fused_DiT, "DiT_B_config"),
}


class CustomImageFolder(Dataset):
    """Dataset that returns (img_path, label, latent_z, raw_image) when load_raw_image=True."""
    def __init__(self, root_dir, cfg=None, load_raw_image=False):
        self.root_dir = root_dir
        self.CACHE_FILE = 'preprocess/image_paths_cache.txt'
        self.image_paths = self._load_or_generate_image_paths()
        self.class_to_idx = self._get_class_to_idx()
        self.latent_dir_name = 'sd-vae-ft-mse_Latents_256img_npz'
        self.latent_shape = (4, 1, cfg.image_size // 8, cfg.image_size // 8)
        self.load_raw_image = load_raw_image
        self.image_size = cfg.image_size

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
        img_path = self.image_paths[idx]

        # Deduce class label from parent directory name
        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[class_name]

        latent_path = img_path.replace('train', self.latent_dir_name)
        latent_path = os.path.splitext(latent_path)[0] + '.latent.npz'

        # Determine flip for consistency between latent and raw image
        do_flip = torch.rand(1) < 0.5

        if osp.exists(latent_path):
            npz_data = np.load(latent_path)
            if do_flip:
                latent_z_data = npz_data['latent_flip']
            else:
                latent_z_data = npz_data['latent']
            latent_z = torch.from_numpy(latent_z_data)
        else:
            latent_z = torch.zeros(self.latent_shape)
            logging.info(f"{latent_path} is not exists!!!!")

        if self.load_raw_image:
            # Load raw image as tensor in [0, 1] for teacher encoder
            raw_image = Image.open(img_path).convert('RGB')
            raw_image = center_crop_arr(raw_image, self.image_size)
            if do_flip:
                raw_image = raw_image.transpose(Image.FLIP_LEFT_RIGHT)
            raw_image = torch.from_numpy(np.array(raw_image)).permute(2, 0, 1).float() / 255.0  # [C, H, W] in [0, 1]
            return img_path, label, latent_z, raw_image

        return img_path, label, latent_z


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


def load_latest_checkpoint(model, ema_model, optimizer, checkpoint_dir='checkpoints', resume_checkpoint_step=None):
    if resume_checkpoint_step is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_step_{resume_checkpoint_step}.pth')
        if not os.path.exists(checkpoint_path):
            logging.error(f"Specified checkpoint not found: {checkpoint_path}")
            return 0
        checkpoints_to_try = [checkpoint_path]
    else:
        checkpoints_to_try = sorted(
            glob.glob(os.path.join(checkpoint_dir, 'ckpt_step_*.pth')),
            key=os.path.getmtime,
            reverse=True
        )
        if not checkpoints_to_try:
            logging.error(f"No checkpoints found in directory: {checkpoint_dir}")
            return 0

    for i, checkpoint_path in enumerate(checkpoints_to_try):
        try:
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['model_state_dict'],
                strict=False
            )
            assert len(missing_keys) == 0, f"Missing keys: {len(missing_keys)} keys"

            if 'ema_model_state_dict' in checkpoint:
                ema_model.load_state_dict(checkpoint['ema_model_state_dict'], strict=False)
                logging.info("EMA model loaded")

            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logging.info("Optimizer loaded")
                except Exception as e:
                    logging.error(f"Failed to load optimizer state: {str(e)}")

            step = checkpoint.get('step', 0)
            logging.info(f'✓ Successfully loaded checkpoint from step {step}')
            return step

        except Exception as e:
            error_msg = f"Failed to load checkpoint {checkpoint_path}: {str(e)}"
            if len(checkpoints_to_try) > 1:
                error_msg += f" (attempt {i+1}/{len(checkpoints_to_try)})"
            logging.error(error_msg)

            import traceback
            logging.debug(traceback.format_exc())

            if resume_checkpoint_step is not None:
                return 0

    logging.error("Could not load any checkpoint. Training from scratch.")
    return 0


def save_checkpoint(model, ema_model, optimizer, step, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_step_{step}.pth')
    torch.save({
        'step': step,
        'model_state_dict': model.module.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
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

    if cfg.param_dtype == torch.bfloat16:
        use_amp = True
        logging.info("Training with bfloat16 mixed precision.")
    else:
        use_amp = False

    writer = None
    if cfg.rank == 0:
        writer = SummaryWriter(log_dir=osp.join(cfg.output_dir, "tensorboard"))

    cfg.train_img_num = getattr(cfg, 'train_img_num', None)
    cfg.grad_mix = max(int(getattr(cfg, 'grad_mix', 1)), 1)

    # REPA config
    repa_config = getattr(cfg, 'repa_config', None)
    use_repa = repa_config is not None

    data_path = cfg.data_path
    if cfg.use_pre_latents:
        img_dataset = CustomImageFolder(data_path, cfg=cfg, load_raw_image=use_repa)
    else:
        transform = transforms.Compose([
            transforms.Lambda(center_crop_lambda),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        img_dataset = ImageFolder(data_path, transform=transform)

    distributed_sampler = DistributedSampler(
        img_dataset,
        num_replicas=cfg.world_size,
        rank=cfg.rank
    )
    cfg.total_train_batch_size = getattr(cfg, 'total_train_batch_size', 256)
    cfg.train_batch_size = cfg.total_train_batch_size // cfg.world_size
    image_dataloader = DataLoader(
        img_dataset,
        batch_size=cfg.train_batch_size,
        sampler=distributed_sampler,
        shuffle=False,
        num_workers=cfg.img_num_workers,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=True
    )
    image_rank_iter = iter(image_dataloader)

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

    # Initialize teacher encoder for REPA
    # Only rank 0 downloads; others wait, then all load from local cache.
    repa_enc_path = getattr(cfg, 'repa_enc_path', None)
    if use_repa:
        enc_type = repa_config.get('enc_type', 'dinov2-vit-b')
        proj_coeff = repa_config.get('proj_coeff', 0.5)
        teacher_affinity_coeff = repa_config.get('teacher_affinity_coeff', 0.0)
        if cfg.rank == 0:
            logging.info(f'Rank 0: downloading/caching REPA teacher encoder: {enc_type}')
            load_teacher_encoder(enc_type, resolution=cfg.image_size, enc_path=repa_enc_path)
        dist.barrier()  # wait for rank 0 to finish caching
        logging.info(
            f'Initializing REPA teacher encoder: {enc_type}, proj_coeff={proj_coeff}, '
            f'teacher_affinity_coeff={teacher_affinity_coeff}'
        )
        teacher_encoder, teacher_embed_dim = load_teacher_encoder(
            enc_type, resolution=cfg.image_size, enc_path=repa_enc_path
        )
        teacher_encoder = teacher_encoder.to(gpu)
        logging.info(f'Teacher encoder embed_dim: {teacher_embed_dim}')

    logging.info('Initializing transformer models (non-ema and ema)')
    model_class, config_name = model_dict[cfg.model_name]
    model_cfg = getattr(cfg, config_name)

    # Validate z_dims and num_teacher_blocks match teacher encoder
    if use_repa:
        model_repa_cfg = model_cfg.get('repa_config', {})
        for z_dim in model_repa_cfg.get('z_dims', []):
            assert z_dim == teacher_embed_dim, \
                f"repa_config.z_dims entry ({z_dim}) must match teacher embed_dim ({teacher_embed_dim})"
        # Validate enc_type consistency between top-level and model-level repa_config
        model_enc_type = model_repa_cfg.get('enc_type', None)
        if model_enc_type is not None:
            assert model_enc_type == enc_type, \
                f"Model repa_config.enc_type ({model_enc_type}) must match " \
                f"top-level repa_config.enc_type ({enc_type})"
        # Auto-inject num_teacher_blocks from actual encoder if not specified
        actual_num_blocks = get_num_teacher_blocks(enc_type)
        cfg_num_blocks = model_repa_cfg.get('num_teacher_blocks', None)
        if cfg_num_blocks is None:
            model_repa_cfg['num_teacher_blocks'] = actual_num_blocks
            logging.info(f"Auto-set repa_config.num_teacher_blocks = {actual_num_blocks} from {enc_type}")
        else:
            assert cfg_num_blocks == actual_num_blocks, \
                f"repa_config.num_teacher_blocks ({cfg_num_blocks}) must match " \
                f"actual teacher encoder blocks ({actual_num_blocks}) for {enc_type}"
    logging.info(f'model_cfg: {model_cfg}')
    model = model_class(**model_cfg)
    model = model.to(gpu)
    model_ema = copy.deepcopy(model).eval().requires_grad_(False)

    # [model] mark model size
    model_size = sum([p.numel() for p in model.parameters()]) / (1024 ** 2)
    logging.info(f'Created models with {model_size:.3f} M parameters')

    # [optim] optimizer
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
        fused=True
    )
    scaler = amp.GradScaler(enabled=False)

    for para_id, (name, param) in enumerate(model.named_parameters()):
        logging.info(f"Train parameter {para_id}: {name}")

    cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoints')
    if cfg.resume_checkpoint:
        cfg.resume_checkpoint_step = getattr(cfg, 'resume_checkpoint_step', None)
        step = load_latest_checkpoint(model, model_ema, optimizer, os.path.join(cfg.checkpoint_dir), cfg.resume_checkpoint_step)
    else:
        step = 0

    model = DistributedDataParallel(model, device_ids=[gpu])

    model.train()
    model_ema.eval()
    optimizer.zero_grad(set_to_none=True)

    monitor = TrainingMonitor(
        model, logger=logging.getLogger(),
        log_every=cfg.log_interval,
        enabled=(cfg.rank == 0),
        writer=writer,
    )

    logging.info('Start the training loop')

    epoch = 0
    accum_steps = 0
    accum_loss_dict = None
    while step < cfg.num_steps:
        # read batch
        try:
            img_batch = next(image_rank_iter)
        except StopIteration:
            epoch += 1
            logging.info("!!!!!!!!!!!!! reload image_dataloader")
            image_rank_iter = iter(image_dataloader)
            img_batch = next(image_rank_iter)

        if cfg.use_pre_latents:
            if use_repa:
                rank_img_paths, rank_img_y, rank_img_z, rank_raw_images = img_batch
                rank_raw_images = rank_raw_images.to(gpu, non_blocking=True)  # [B, C, H, W] in [0, 1]
            else:
                rank_img_paths, rank_img_y, rank_img_z = img_batch
            rank_img_y, rank_img_z = rank_img_y.to(gpu, non_blocking=True), rank_img_z.to(gpu, non_blocking=True)
            rank_img_z_is_all_zero = torch.all(rank_img_z == 0).item()
            assert not rank_img_z_is_all_zero, "error: rank_img_z is all zero"
        else:
            rank_images, rank_img_y = img_batch
            rank_images, rank_img_y = rank_images.to(gpu, non_blocking=True), rank_img_y.to(gpu, non_blocking=True)
            if use_repa:
                # Convert from [-1, 1] (VAE normalized) to [0, 1] for teacher encoder
                rank_raw_images = (rank_images + 1.0) / 2.0  # [B, C, H, W] in [0, 1]
            rank_images = rearrange(rank_images, "B C H W -> B C 1 H W")

        # Extract ALL teacher block features for MoS REPA
        if use_repa:
            with torch.no_grad():
                with amp.autocast(dtype=cfg.param_dtype, enabled=use_amp):
                    teacher_all_z = extract_all_teacher_block_features(
                        teacher_encoder, rank_raw_images, repa_config['enc_type']
                    )  # (num_teacher_blocks, B, num_patches, teacher_embed_dim)

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
        if use_repa:
            arg_c['teacher_all_z'] = teacher_all_z

        noise = torch.randn_like(z)
        noised_z_in = (1.0 - sigmas.squeeze()).view(z.shape[0], 1, 1, 1, 1) * z + sigmas.squeeze().view(z.shape[0], 1, 1, 1, 1) * noise

        with amp.autocast(dtype=cfg.param_dtype, enabled=use_amp):
            model_output = model(noised_z_in, t, **arg_c)

        loss_dict = {}
        loss_dict["loss"] = 0

        # Handle model output: MoS REPA returns (pred, mos_repa_loss)
        if isinstance(model_output, tuple):
            model_pred, mos_repa_loss = model_output[0], model_output[1]
            teacher_affinity_loss = model_output[2] if len(model_output) == 3 else None

            if model_pred.shape[1] != noised_z_in.shape[1]:
                model_pred, _ = model_pred.chunk(2, dim=1)
            model_pred = model_pred.unsqueeze(2)

            # MoS REPA loss (scalar returned by model, weighted by proj_coeff)
            if use_repa and torch.is_tensor(mos_repa_loss):
                loss_dict["mos_repa_loss"] = mos_repa_loss
                mos_repa_loss_weighted = mos_repa_loss * proj_coeff
                loss_dict["mos_repa_loss_weighted"] = mos_repa_loss_weighted
                loss_dict["loss"] += mos_repa_loss_weighted

            if use_repa and torch.is_tensor(teacher_affinity_loss):
                loss_dict["teacher_affinity_loss"] = teacher_affinity_loss
                teacher_affinity_loss_weighted = (
                    teacher_affinity_loss * teacher_affinity_coeff
                )
                loss_dict["teacher_affinity_loss_weighted"] = teacher_affinity_loss_weighted
                loss_dict["loss"] += teacher_affinity_loss_weighted

        elif model_output.shape[1] != noised_z_in.shape[1]:
            ########## DiT loss
            model_pred, _ = model_output.chunk(2, dim=1)
            model_pred = model_pred.unsqueeze(2)
        else:
            model_pred = model_output

        target = noise - z

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

        scaler.unscale_(optimizer)
        monitor.on_step(step, losses=logged_loss_dict)
        grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        update_ema(model_ema, model.module)

        if cfg.rank == 0 and step != 0 and step % cfg.save_ckpt_interval == 0:
            save_checkpoint(model, model_ema, optimizer, step, cfg.checkpoint_dir)

        accum_steps = 0
        accum_loss_dict = None
        step += 1

    monitor.close()

    if cfg.rank == 0:
        logging.info('Congratulations! The training is completed!')
        writer.close()

    # barrier to ensure all ranks are completed
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoE with REPA')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--vae-path', type=str, default=None,
                        help='Local path to a pretrained VAE directory (skip auto-download)')
    parser.add_argument('--repa-enc-path', type=str, default=None,
                        help='Local path to a REPA teacher encoder state_dict file (skip auto-download)')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        custom_cfg = yaml.safe_load(file)

    custom_cfg['custom_cfg_name'] = osp.splitext(osp.basename(args.config))[0]
    if args.vae_path is not None:
        custom_cfg['vae_path'] = args.vae_path
    if args.repa_enc_path is not None:
        custom_cfg['repa_enc_path'] = args.repa_enc_path
    main(**custom_cfg)
