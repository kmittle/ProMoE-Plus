"""Utilities for resolving checkpoint paths, configs, and model builders."""

import os
import os.path as osp
import re
import sys
import inspect
import yaml

# Add project root to path
PROJECT_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import cfg
from utils import deep_update


def resolve_ckpt_info(ckpt_path):
    """Resolve a checkpoint relative path to (model_name, custom_cfg_name, ckpt_step, config_yaml_path, output_dir).

    Expected ckpt_path pattern:
        outputs/{model_name}/{custom_cfg_name}/checkpoints/ckpt_step_{step}.pth
    """
    abs_path = osp.abspath(ckpt_path)

    # Extract step from filename
    basename = osp.basename(abs_path)
    match = re.match(r"ckpt_step_(\d+)\.pth", basename)
    if not match:
        raise ValueError(f"Cannot parse step from checkpoint filename: {basename}")
    ckpt_step = int(match.group(1))

    # Walk up: checkpoints/ -> {custom_cfg_name}/ -> {model_name}/ -> outputs/
    checkpoints_dir = osp.dirname(abs_path)
    cfg_name_dir = osp.dirname(checkpoints_dir)
    model_name_dir = osp.dirname(cfg_name_dir)

    custom_cfg_name = osp.basename(cfg_name_dir)
    model_name = osp.basename(model_name_dir)
    output_dir = cfg_name_dir  # outputs/{model_name}/{custom_cfg_name}

    # Try to find the corresponding YAML config
    config_yaml_path = osp.join(PROJECT_ROOT, "configs", f"{custom_cfg_name}.yaml")
    if not osp.isfile(config_yaml_path):
        raise FileNotFoundError(
            f"Cannot find config YAML: {config_yaml_path}\n"
            f"Derived custom_cfg_name='{custom_cfg_name}' from ckpt path."
        )

    return model_name, custom_cfg_name, ckpt_step, config_yaml_path, output_dir


def load_config_from_yaml(config_yaml_path):
    """Load YAML config and merge into the global cfg object. Returns the updated cfg."""
    with open(config_yaml_path, "r") as f:
        custom_cfg = yaml.safe_load(f)
    custom_cfg["custom_cfg_name"] = osp.splitext(osp.basename(config_yaml_path))[0]
    deep_update(cfg, custom_cfg)
    return cfg


def build_model_from_cfg(cfg):
    """Instantiate the model class based on cfg.model_name and return the model."""
    # Import model dicts (same logic as sample.py).
    # Temporarily suppress TORCH_DISTRIBUTED_DEBUG set by train.py top-level code.
    _prev_debug = os.environ.get("TORCH_DISTRIBUTED_DEBUG")
    from train import model_dict as _base_model_dict
    from train_with_repa import model_dict as _repa_model_dict
    from train_with_MoS_repa import model_dict as _mos_repa_model_dict
    if _prev_debug is None:
        os.environ.pop("TORCH_DISTRIBUTED_DEBUG", None)
    else:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = _prev_debug
    model_dict = {**_base_model_dict, **_repa_model_dict, **_mos_repa_model_dict}

    model_name = cfg.model_name
    if model_name not in model_dict:
        raise KeyError(f"model_name '{model_name}' not found in model_dict. "
                       f"Available: {list(model_dict.keys())}")

    model_class, config_key = model_dict[model_name]
    model_cfg = getattr(cfg, config_key)
    model = model_class(**model_cfg)
    return model


def load_ema_weights(model, ckpt_path):
    """Load EMA weights from checkpoint into model."""
    import torch

    # Keep the restricted unpickler enabled while allowing only metadata types
    # used by this project's checkpoints.  Analysis paths are caller-supplied,
    # so falling back to unrestricted pickle loading would be unsafe.
    try:
        supports_weights_only = "weights_only" in inspect.signature(
            torch.load
        ).parameters
    except (TypeError, ValueError):
        supports_weights_only = False
    if not supports_weights_only:
        raise RuntimeError(
            "Secure checkpoint analysis requires a PyTorch version with "
            "weights_only loading"
        )
    serialization = getattr(torch, "serialization", None)
    safe_globals = getattr(serialization, "safe_globals", None)
    if safe_globals is not None:
        try:
            from easydict import EasyDict
            from torch.torch_version import TorchVersion
        except (ImportError, AttributeError):
            pass
        else:
            with safe_globals([EasyDict, TorchVersion]):
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=True
                )
    else:
        # Keep the restricted loader even on versions that expose only the
        # boolean flag; an EasyDict metadata error is safer than pickle code
        # execution and gives the caller an actionable upgrade signal.
        checkpoint = torch.load(
            ckpt_path, map_location="cpu", weights_only=True
        )
    if "ema_model_state_dict" in checkpoint:
        state_dict = checkpoint["ema_model_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        raise KeyError(
            "Checkpoint has no 'ema_model_state_dict' or 'model_state_dict'. "
            f"Keys: {list(checkpoint.keys())}"
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading checkpoint: {unexpected}")
    return model
