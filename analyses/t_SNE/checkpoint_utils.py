import copy
import os
import os.path as osp
import re
from pathlib import Path

import yaml

from config import cfg as base_cfg
from utils import deep_update


_CKPT_STEP_PATTERN = re.compile(r"ckpt_step_(\d+)\.pth$")


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_checkpoint_step(ckpt_path: Path) -> int:
    match = _CKPT_STEP_PATTERN.search(ckpt_path.name)
    if match is None:
        raise ValueError(f"Cannot parse checkpoint step from path: {ckpt_path}")
    return int(match.group(1))


def resolve_config_from_checkpoint(ckpt_path: Path, repo_root: Path | None = None) -> Path:
    repo_root = repo_root or resolve_repo_root()
    config_stem = ckpt_path.parent.parent.name
    direct_match = repo_root / "configs" / f"{config_stem}.yaml"
    if direct_match.exists():
        return direct_match

    candidates = sorted((repo_root / "configs").glob(f"**/{config_stem}.yaml"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find YAML config for checkpoint {ckpt_path}. "
            f"Expected {direct_match} or a unique match under configs/."
        )
    raise RuntimeError(
        f"Found multiple YAML configs for stem '{config_stem}': "
        f"{[str(path) for path in candidates]}"
    )


def load_runtime_cfg(config_path: Path):
    runtime_cfg = copy.deepcopy(base_cfg)
    with open(config_path, "r") as file:
        custom_cfg = yaml.safe_load(file)
    custom_cfg["custom_cfg_name"] = osp.splitext(osp.basename(config_path))[0]
    deep_update(runtime_cfg, custom_cfg)
    return runtime_cfg


def resolve_analysis_output_dir(ckpt_path: Path) -> Path:
    step = parse_checkpoint_step(ckpt_path)
    run_root = ckpt_path.parent.parent
    return run_root / "sample" / f"step{step}" / "t-sne" / "token-wise"


def resolve_visible_gpu_ids(runtime_cfg) -> list[int]:
    if "sample_gpu_ids" in runtime_cfg and runtime_cfg.sample_gpu_ids is not None:
        return list(runtime_cfg.sample_gpu_ids)
    if "gpu_ids" in runtime_cfg and runtime_cfg.gpu_ids is not None:
        return list(runtime_cfg.gpu_ids)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [int(gpu_id.strip()) for gpu_id in visible.split(",") if gpu_id.strip()]
    return list(range(base_cfg.gpus_per_machine))


def apply_visible_gpu_ids(gpu_ids: list[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)


def sanitize_config_for_yaml(runtime_cfg) -> dict:
    serialized = {}
    for key, value in runtime_cfg.items():
        if key.endswith("_dtype"):
            serialized[key] = str(value)
        elif isinstance(value, dict):
            serialized[key] = sanitize_config_for_yaml(value)
        else:
            serialized[key] = value
    return serialized
