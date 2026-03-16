import copy
import os
import os.path as osp
import re
from pathlib import Path

import yaml

from config import cfg as base_cfg
from utils import deep_update


_CKPT_STEP_PATTERN = re.compile(r"ckpt_step_(\d+)\.pth$")
_STEP_DIR_PATTERN = re.compile(r"step(\d+)$")


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_checkpoint_step(ckpt_path: Path) -> int:
    match = _CKPT_STEP_PATTERN.search(ckpt_path.name)
    if match is None:
        raise ValueError(f"Cannot parse checkpoint step from path: {ckpt_path}")
    return int(match.group(1))


def parse_step_dir_name(step_dir_name: str) -> int:
    match = _STEP_DIR_PATTERN.search(step_dir_name)
    if match is None:
        raise ValueError(f"Cannot parse sampling step from directory name: {step_dir_name}")
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


def resolve_analysis_output_dir(ckpt_path: Path, analysis_name: str = "token-wise") -> Path:
    step = parse_checkpoint_step(ckpt_path)
    run_root = ckpt_path.parent.parent
    return run_root / "sample" / f"step{step}" / "t-sne" / analysis_name


def resolve_run_root_from_image_dir(image_dir: Path) -> Path:
    image_dir = image_dir.resolve()
    for parent in image_dir.parents:
        if parent.name == "sample":
            return parent.parent
    raise ValueError(
        f"Cannot resolve run root from image directory: {image_dir}. "
        "Expected it to live under outputs/<model>/<config>/sample/..."
    )


def resolve_step_from_image_dir(image_dir: Path) -> int:
    image_dir = image_dir.resolve()
    for parent in image_dir.parents:
        if _STEP_DIR_PATTERN.search(parent.name):
            return parse_step_dir_name(parent.name)
    raise ValueError(
        f"Cannot resolve sampling step from image directory: {image_dir}. "
        "Expected a parent directory like step500000."
    )


def resolve_checkpoint_from_image_dir(image_dir: Path) -> Path:
    run_root = resolve_run_root_from_image_dir(image_dir)
    step = resolve_step_from_image_dir(image_dir)
    return run_root / "checkpoints" / f"ckpt_step_{step}.pth"


def resolve_config_from_image_dir(image_dir: Path, repo_root: Path | None = None) -> Path:
    ckpt_path = resolve_checkpoint_from_image_dir(image_dir)
    return resolve_config_from_checkpoint(ckpt_path, repo_root=repo_root)


def resolve_analysis_output_dir_from_image_dir(
    image_dir: Path,
    analysis_name: str = "image-wise",
) -> Path:
    run_root = resolve_run_root_from_image_dir(image_dir)
    step = resolve_step_from_image_dir(image_dir)
    return run_root / "sample" / f"step{step}" / "t-sne" / analysis_name


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
