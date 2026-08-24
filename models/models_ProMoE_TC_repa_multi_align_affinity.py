import math

import torch
import torch.nn.functional as F

from .models_ProMoE_TC_repa_multi_align import AddAuxiliaryLoss, DiT as MultiAlignDiT
from .modules import modulate


class DiT(MultiAlignDiT):
    """Multi-align ProMoE with teacher-affinity supervision on one expert router."""

    def __init__(self, *args, repa_config=None, **kwargs):
        super().__init__(*args, repa_config=repa_config, **kwargs)

        repa_config = repa_config or {}
        self.teacher_affinity_block = repa_config.get('teacher_affinity_block', 3)
        self.teacher_affinity_grid_size = repa_config.get('teacher_affinity_grid_size', 8)
        self.teacher_affinity_router_temperature = repa_config.get(
            'teacher_affinity_router_temperature', 0.1
        )
        self.teacher_affinity_relation_temperature = repa_config.get(
            'teacher_affinity_relation_temperature', 0.5
        )
        self.teacher_affinity_eps = repa_config.get('teacher_affinity_eps', 1e-6)

        if not 0 <= self.teacher_affinity_block < len(self.blocks):
            raise ValueError(
                f"teacher_affinity_block must be in [0, {len(self.blocks) - 1}], "
                f"got {self.teacher_affinity_block}"
            )
        if not self.blocks[self.teacher_affinity_block].use_moe:
            raise ValueError(
                f"teacher_affinity_block {self.teacher_affinity_block} must be a MoE block"
            )
        if self.teacher_affinity_grid_size < 2:
            raise ValueError("teacher_affinity_grid_size must be at least 2")
        if self.teacher_affinity_router_temperature <= 0:
            raise ValueError("teacher_affinity_router_temperature must be positive")
        if self.teacher_affinity_relation_temperature <= 0:
            raise ValueError("teacher_affinity_relation_temperature must be positive")
        if self.teacher_affinity_eps <= 0:
            raise ValueError("teacher_affinity_eps must be positive")

    @staticmethod
    def _spatial_pool_tokens(tokens, output_size):
        batch_size, num_tokens, hidden_size = tokens.shape
        side = math.isqrt(num_tokens)
        if side * side != num_tokens:
            raise ValueError(
                f"teacher-affinity routing requires a square token grid, got {num_tokens} tokens"
            )
        if output_size > side:
            raise ValueError(
                f"teacher_affinity_grid_size ({output_size}) cannot exceed token grid size ({side})"
            )

        spatial = tokens.transpose(1, 2).reshape(batch_size, hidden_size, side, side)
        pooled = F.adaptive_avg_pool2d(spatial, (output_size, output_size))
        return pooled.flatten(2).transpose(1, 2)

    def compute_teacher_affinity_loss(self, router_input, labels, teacher_z, moe_layer):
        """Match DINO patch relations to differentiable router co-assignment relations."""
        if teacher_z.shape[:2] != router_input.shape[:2]:
            raise ValueError(
                f"teacher/router token shapes must match, got {teacher_z.shape[:2]} "
                f"and {router_input.shape[:2]}"
            )

        cond_mask = labels != 1000
        if not cond_mask.any():
            return router_input.float().sum() * 0.0

        with torch.autocast(device_type=router_input.device.type, enabled=False):
            teacher = teacher_z[cond_mask].detach().float()
            teacher_mean = teacher.mean(dim=1, keepdim=True)
            teacher_var = teacher.var(dim=1, keepdim=True, unbiased=False)
            teacher = (teacher - teacher_mean) * torch.rsqrt(
                teacher_var + self.teacher_affinity_eps
            )
            teacher = self._spatial_pool_tokens(
                teacher, self.teacher_affinity_grid_size
            )
            teacher = F.normalize(teacher, p=2, dim=-1)

            router_tokens = router_input[cond_mask].float()
            router_tokens = F.normalize(router_tokens, p=2, dim=-1)
            centers = F.normalize(moe_layer.cluster_centers.float(), p=2, dim=-1)
            router_scores = router_tokens @ centers.T
            router_prob = F.softmax(
                router_scores / self.teacher_affinity_router_temperature, dim=-1
            )
            router_prob = self._spatial_pool_tokens(
                router_prob, self.teacher_affinity_grid_size
            )
            router_prob = F.normalize(router_prob, p=2, dim=-1)

            teacher_affinity = teacher @ teacher.transpose(1, 2)
            router_affinity = router_prob @ router_prob.transpose(1, 2)

            num_regions = teacher_affinity.shape[-1]
            diagonal = torch.eye(
                num_regions, dtype=torch.bool, device=teacher_affinity.device
            ).unsqueeze(0)
            mask_value = torch.finfo(teacher_affinity.dtype).min
            teacher_logits = (
                teacher_affinity / self.teacher_affinity_relation_temperature
            ).masked_fill(diagonal, mask_value)
            router_logits = (
                router_affinity / self.teacher_affinity_relation_temperature
            ).masked_fill(diagonal, mask_value)

            teacher_prob = F.softmax(teacher_logits, dim=-1).detach()
            router_log_prob = F.log_softmax(router_logits, dim=-1)
            pairwise_kl = F.kl_div(
                router_log_prob, teacher_prob, reduction='none'
            ).masked_fill(diagonal, 0.0)
            return pairwise_kl.sum(dim=-1).mean()

    def _forward_affinity_block(self, block, x, c, labels, teacher_z):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            block.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * block.attn(
            modulate(block.norm1(x), shift_msa, scale_msa)
        )
        router_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x_mlp, aux_loss = block.mlp(router_input, labels)
        if aux_loss is not None:
            x_mlp = AddAuxiliaryLoss.apply(x_mlp, aux_loss)
        x = x + gate_mlp.unsqueeze(1) * x_mlp

        teacher_affinity_loss = self.compute_teacher_affinity_loss(
            router_input, labels, teacher_z, block.mlp
        )
        return x, teacher_affinity_loss

    def forward(self, x, timestep, context, teacher_all_z=None, **kwargs):
        """Return diffusion output, multi-align loss, and TAR loss during training."""
        if not self.training:
            return super().forward(
                x, timestep, context, teacher_all_z=teacher_all_z, **kwargs
            )

        y = context
        if len(x.shape) != 4:
            x = x.squeeze(2)

        x = self.x_embedder(x) + self.pos_embed
        batch_size, num_tokens, hidden_size = x.shape
        t = self.t_embedder(timestep)
        y, labels = self.y_embedder(y, self.training)
        c = t + y

        align_coeffs = None
        teacher_z = None
        if self.training and self.projectors is not None and teacher_all_z is not None:
            if self.align_coeff_predictor is not None:
                align_coeffs = self.align_coeff_predictor(x, c)
            teacher_z = teacher_all_z[-1]

        repa_loss = torch.tensor(0.0, device=x.device)
        teacher_affinity_loss = x.float().sum() * 0.0
        for block_idx, block in enumerate(self.blocks):
            if block_idx == self.teacher_affinity_block and teacher_z is not None:
                x, teacher_affinity_loss = self._forward_affinity_block(
                    block, x, c, labels, teacher_z
                )
            else:
                x = block(x, c, labels)

            if (
                self.training
                and teacher_z is not None
                and block_idx in self.align_block_to_idx
            ):
                align_idx = self.align_block_to_idx[block_idx]
                block_loss = self.compute_multi_align_loss(
                    x,
                    align_idx,
                    align_coeffs,
                    teacher_z,
                    batch_size,
                    num_tokens,
                    hidden_size,
                )
                repa_loss = repa_loss + block_loss

        if self.training and teacher_z is not None and len(self.align_blocks) > 0:
            repa_loss = repa_loss / len(self.align_blocks)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        return x, repa_loss, teacher_affinity_loss
