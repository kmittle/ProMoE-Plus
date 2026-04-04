from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from analyses.t_SNE.checkpoint_utils import parse_checkpoint_step


def resolve_mos_routing_output_dir(ckpt_path: Path) -> Path:
    step = parse_checkpoint_step(ckpt_path)
    run_root = ckpt_path.parent.parent
    return run_root / "sample" / f"step{step}" / "mos_routing"


def build_dummy_teacher_all_z(
    num_teacher_blocks: int,
    batch_size: int,
    num_patches: int,
    teacher_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a dummy teacher_all_z (zeros) to trigger routing code path."""
    return torch.zeros(
        num_teacher_blocks, batch_size, num_patches, teacher_dim,
        device=device,
    )


def read_mos_repa_params(model) -> dict:
    """Read num_teacher_blocks and z_dims from model attributes."""
    num_teacher_blocks = getattr(model, 'num_teacher_blocks', None)
    if num_teacher_blocks is None:
        num_teacher_blocks = 12

    # Try to read z_dim from projectors
    z_dim = 768  # default for dinov2-vit-b
    projectors = getattr(model, 'mos_projectors', None) or getattr(model, 'projectors', None)
    if projectors is not None and len(projectors) > 0:
        last_layer = projectors[0][-1] if hasattr(projectors[0], '__getitem__') else None
        if last_layer is not None and hasattr(last_layer, 'out_features'):
            z_dim = last_layer.out_features

    return {"num_teacher_blocks": num_teacher_blocks, "z_dim": z_dim}


class MoSRoutingCapture:
    """
    Model-agnostic MoS routing weight capture via forward hooks.

    Supports all MoS model variants:
    - 'global': BlockRouter (naive, naive_choice, naive_choice_sep)
    - 'blockwise': BlockRouter with token-pooled routing
    - 'per_block': PerBlockRouter (per-block single-layer router)
    - 'mos': AdaLNRouter (per-block, returns logits)

    Usage:
        capture = MoSRoutingCapture(model)
        capture.enable()
        model(x, t, y, teacher_all_z=dummy)
        capture.disable()
        data = capture.get_routing_data()  # {block_idx: (N, T, m)}
    """

    def __init__(self, model):
        self.model = model
        self.model_type = self._detect_model_type()

        # MoS model aligns ALL blocks (no align_blocks attr); others have explicit align_blocks
        if hasattr(model, 'align_blocks') and model.align_blocks:
            self.align_blocks = list(model.align_blocks)
            self.align_block_to_idx = dict(model.align_block_to_idx)
        elif self.model_type == 'mos':
            depth = len(model.blocks)
            self.align_blocks = list(range(depth))
            self.align_block_to_idx = {i: i for i in range(depth)}
        else:
            self.align_blocks = []
            self.align_block_to_idx = {}

        self._enabled = False
        self._captured: Dict[int, torch.Tensor] = {}
        self._handles: list = []
        self._register_hooks()

    def _detect_model_type(self) -> str:
        """Auto-detect from model attributes."""
        model = self.model
        if hasattr(model, 'per_block_routers') and model.per_block_routers is not None:
            return 'per_block'
        if hasattr(model, 'mos_routers') and model.mos_routers is not None:
            return 'mos'
        if hasattr(model, 'block_router') and model.block_router is not None:
            module_name = type(model).__module__ or ''
            if 'blockwise' in module_name:
                return 'blockwise'
            return 'global'
        raise RuntimeError(f"Cannot detect MoS model type from {type(model).__name__}")

    def _register_hooks(self):
        if self.model_type in ('global', 'blockwise'):
            self._handles.append(
                self.model.block_router.register_forward_hook(self._global_router_hook)
            )
        elif self.model_type == 'per_block':
            for align_idx, router in enumerate(self.model.per_block_routers):
                block_idx = self.align_blocks[align_idx]
                self._handles.append(
                    router.register_forward_hook(self._make_per_block_hook(block_idx))
                )
        elif self.model_type == 'mos':
            for block_idx, router in enumerate(self.model.mos_routers):
                self._handles.append(
                    router.register_forward_hook(self._make_adaln_hook(block_idx))
                )

    def _global_router_hook(self, module, inputs, output):
        if not self._enabled:
            return
        # output: (N, T, m, n) — already softmax-normalized
        routing_weights = output.detach()
        for block_idx, align_idx in self.align_block_to_idx.items():
            self._captured[block_idx] = routing_weights[:, :, :, align_idx].cpu()

    def _make_per_block_hook(self, block_idx: int):
        def hook(module, inputs, output):
            if not self._enabled:
                return
            # output: (N, T, m) — already softmax-normalized
            self._captured[block_idx] = output.detach().cpu()
        return hook

    def _make_adaln_hook(self, block_idx: int):
        def hook(module, inputs, output):
            if not self._enabled:
                return
            # AdaLNRouter returns logits (N, T, K); apply softmax
            self._captured[block_idx] = F.softmax(output.detach(), dim=-1).cpu()
        return hook

    def enable(self):
        self._enabled = True
        self._captured = {}

    def disable(self):
        self._enabled = False

    def get_routing_data(self) -> Dict[int, torch.Tensor]:
        """Return captured routing data: {block_idx: (N, T, m)}."""
        return dict(self._captured)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
