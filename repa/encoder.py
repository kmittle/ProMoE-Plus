import os
import logging
import torch
import torch.nn as nn
import timm.layers.pos_embed


def load_teacher_encoder(enc_type, resolution=256, local_root="pretrained_ckpt/encoder"):
    """
    Load a frozen teacher encoder for REPA alignment.

    Supported enc_type format: "{family}-{arch}-{size}"
      e.g. "dinov2-vit-b", "dinov2-vit-l", "dinov2-vit-g", "dinov2reg-vit-b"

    First run:  downloads via torch.hub and saves the full model to local_root.
    Second run: loads directly from local_root, no network needed.

    Returns:
        encoder: frozen teacher model
        embed_dim: teacher embedding dimension
    """
    parts = enc_type.split('-')
    assert len(parts) == 3, f"enc_type must be 'family-arch-size', got '{enc_type}'"
    encoder_type, architecture, model_config = parts

    if 'dinov2' in encoder_type:
        encoder, embed_dim = _load_dinov2(encoder_type, model_config, resolution, local_root)
    else:
        raise NotImplementedError(f"Encoder type '{encoder_type}' not supported. Use 'dinov2' or 'dinov2reg'.")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder, embed_dim


def _load_dinov2(encoder_type, model_config, resolution, local_root):
    """Load DINOv2 model, using local cache if available."""
    if 'reg' in encoder_type:
        hub_name = f'dinov2_vit{model_config}14_reg'
    else:
        hub_name = f'dinov2_vit{model_config}14'

    local_path = os.path.join(local_root, hub_name)
    weights_path = os.path.join(local_path, 'state_dict.pth')

    if os.path.isfile(weights_path):
        # Local weights exist: build model structure via torch.hub (repo cached after first download),
        # then load our cached state_dict (avoids re-downloading weights from Meta CDN).
        logging.info(f"Loading DINOv2 weights from local cache: {weights_path}")
        encoder = torch.hub.load('facebookresearch/dinov2', hub_name, pretrained=False)
        state_dict = torch.load(weights_path, map_location='cpu')
        encoder.load_state_dict(state_dict)
    else:
        # First run: download via torch.hub, then cache state_dict locally
        logging.info(f"Local DINOv2 not found at {local_path}, downloading via torch.hub: {hub_name}")
        encoder = torch.hub.load('facebookresearch/dinov2', hub_name)
        os.makedirs(local_path, exist_ok=True)
        torch.save(encoder.state_dict(), weights_path)
        logging.info(f"DINOv2 state_dict cached to {weights_path}")

    embed_dim = encoder.embed_dim

    # Resample positional embeddings for the target resolution
    patch_resolution = 16 * (resolution // 256)
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        encoder.pos_embed.data, [patch_resolution, patch_resolution],
    )

    del encoder.head
    encoder.head = nn.Identity()

    return encoder, embed_dim


@torch.no_grad()
def extract_teacher_features(encoder, raw_images, enc_type):
    """
    Extract patch-level features from the teacher encoder.

    Args:
        encoder: frozen teacher model
        raw_images: (B, C, H, W) in [0, 1] range
        enc_type: encoder type string, e.g. "dinov2-vit-b"

    Returns:
        z: (B, num_patches, embed_dim) teacher features
    """
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from torchvision.transforms import Normalize

    encoder_type = enc_type.split('-')[0]
    resolution = raw_images.shape[-1]

    x = raw_images.clone()
    if 'dinov2' in encoder_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    z = encoder.forward_features(x)
    if 'dinov2' in encoder_type:
        z = z['x_norm_patchtokens']

    return z
