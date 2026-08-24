import math

import torch
import torch.nn.functional as F

from .models_ProMoE_TC_repa_multi_align import AddAuxiliaryLoss, DiT as MultiAlignDiT
from .modules import modulate


class DiT(MultiAlignDiT):
    """Multi-Align with training-only shared/routed spectral responsibility."""

    def __init__(self, *args, repa_config=None, **kwargs):
        super().__init__(*args, repa_config=repa_config, **kwargs)

        repa_config = repa_config or {}
        self.spectral_responsibility_block = repa_config.get(
            'spectral_responsibility_block', 3
        )
        self.spectral_responsibility_reverse = repa_config.get(
            'spectral_responsibility_reverse', False
        )
        self.spectral_residual_min_ratio = repa_config.get(
            'spectral_residual_min_ratio', 0.1
        )
        self.spectral_responsibility_eps = repa_config.get(
            'spectral_responsibility_eps', 1e-6
        )

        block_idx = self.spectral_responsibility_block
        if (
            isinstance(block_idx, bool)
            or not isinstance(block_idx, int)
            or not 0 <= block_idx < len(self.blocks)
        ):
            raise ValueError(
                f"spectral_responsibility_block must be an integer in "
                f"[0, {len(self.blocks) - 1}], got {block_idx!r}"
            )
        if block_idx not in self.align_block_to_idx:
            raise ValueError(
                "spectral_responsibility_block must also be present in align_blocks, "
                f"got block {block_idx} and align_blocks={self.align_blocks}"
            )
        if not self.blocks[block_idx].use_moe:
            raise ValueError(
                f"spectral_responsibility_block {block_idx} must be a MoE block"
            )
        if not self.blocks[block_idx].mlp.use_shared_expert:
            raise ValueError("spectral responsibility requires a shared expert")
        if self.spectral_residual_min_ratio < 0:
            raise ValueError("spectral_residual_min_ratio must be non-negative")
        if self.spectral_responsibility_eps <= 0:
            raise ValueError("spectral_responsibility_eps must be positive")

        direction = "shared->high, routed->low" if self.spectral_responsibility_reverse \
            else "shared->low, routed->high"
        print(
            f"Spectral responsibility at block {block_idx}: {direction}, "
            f"residual_min_ratio={self.spectral_residual_min_ratio}"
        )

    @staticmethod
    def _spatial_low_pass(tokens):
        batch_size, num_tokens, hidden_size = tokens.shape
        side = math.isqrt(num_tokens)
        if side * side != num_tokens:
            raise ValueError(
                f"spectral responsibility requires a square token grid, got {num_tokens} tokens"
            )
        spatial = tokens.transpose(1, 2).reshape(
            batch_size, hidden_size, side, side
        )
        spatial = F.pad(spatial, (1, 1, 1, 1), mode='reflect')
        low_pass = F.avg_pool2d(spatial, kernel_size=3, stride=1)
        return low_pass.flatten(2).transpose(1, 2)

    def _branch_alignment_loss(
        self,
        student,
        target,
        projector,
        align_coeff,
        target_mask=None,
    ):
        batch_size, num_tokens, hidden_size = student.shape
        projected = projector(student.reshape(-1, hidden_size)).reshape(
            batch_size, num_tokens, -1
        )
        projected = F.normalize(projected.float(), dim=-1)
        target = F.normalize(target.detach().float(), dim=-1)
        neg_cosine = -(projected * target).sum(dim=-1)

        weights = torch.ones_like(neg_cosine)
        if align_coeff is not None:
            weights = weights * align_coeff.float()
        normalizer = torch.tensor(
            neg_cosine.numel(), dtype=neg_cosine.dtype, device=neg_cosine.device
        )
        if target_mask is not None:
            target_mask = target_mask.detach().to(neg_cosine.dtype)
            weights = weights * target_mask
            normalizer = target_mask.sum().clamp_min(1.0)
        return (neg_cosine * weights).sum() / normalizer

    def compute_spectral_responsibility_loss(
        self,
        shared_output,
        routed_output,
        labels,
        teacher_z,
        align_coeffs,
    ):
        if teacher_z.shape[:2] != shared_output.shape[:2]:
            raise ValueError(
                f"teacher/branch token shapes must match, got {teacher_z.shape[:2]} "
                f"and {shared_output.shape[:2]}"
            )

        cond_mask = labels != 1000
        if not cond_mask.any():
            return (shared_output.float().sum() + routed_output.float().sum()) * 0.0

        teacher = teacher_z[cond_mask].detach().float()
        teacher_mean = teacher.mean(dim=1, keepdim=True)
        teacher_var = teacher.var(dim=1, keepdim=True, unbiased=False)
        teacher = (teacher - teacher_mean) * torch.rsqrt(
            teacher_var + self.spectral_responsibility_eps
        )
        teacher_low = self._spatial_low_pass(teacher)
        teacher_high = teacher - teacher_low

        residual_norm = torch.linalg.vector_norm(teacher_high, dim=-1)
        residual_reference = residual_norm.mean(dim=1, keepdim=True)
        high_mask = (
            residual_norm > self.spectral_responsibility_eps
        ) & (
            residual_norm >= self.spectral_residual_min_ratio * residual_reference
        )

        align_idx = self.align_block_to_idx[self.spectral_responsibility_block]
        projector = self.projectors[align_idx]
        align_coeff = None
        if align_coeffs is not None:
            align_coeff = align_coeffs[cond_mask, :, align_idx]

        shared_output = shared_output[cond_mask]
        routed_output = routed_output[cond_mask]
        if self.spectral_responsibility_reverse:
            shared_target, shared_mask = teacher_high, high_mask
            routed_target, routed_mask = teacher_low, None
        else:
            shared_target, shared_mask = teacher_low, None
            routed_target, routed_mask = teacher_high, high_mask

        shared_loss = self._branch_alignment_loss(
            shared_output,
            shared_target,
            projector,
            align_coeff,
            target_mask=shared_mask,
        )
        routed_loss = self._branch_alignment_loss(
            routed_output,
            routed_target,
            projector,
            align_coeff,
            target_mask=routed_mask,
        )
        return 0.5 * (shared_loss + routed_loss)

    @staticmethod
    def _forward_responsibility_block(block, x, c, labels):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            block.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * block.attn(
            modulate(block.norm1(x), shift_msa, scale_msa)
        )
        router_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x_mlp, aux_loss, routed_output, shared_output = block.mlp(
            router_input, labels, return_branches=True
        )
        if aux_loss is not None:
            x_mlp = AddAuxiliaryLoss.apply(x_mlp, aux_loss)
        x = x + gate_mlp.unsqueeze(1) * x_mlp
        return x, shared_output, routed_output

    def forward(self, x, timestep, context, teacher_all_z=None, **kwargs):
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
        if self.projectors is not None and teacher_all_z is not None:
            if self.align_coeff_predictor is not None:
                align_coeffs = self.align_coeff_predictor(x, c)
            teacher_z = teacher_all_z[-1]

        repa_loss = torch.tensor(0.0, device=x.device)
        spectral_responsibility_loss = x.float().sum() * 0.0
        for block_idx, block in enumerate(self.blocks):
            if block_idx == self.spectral_responsibility_block and teacher_z is not None:
                x, shared_output, routed_output = self._forward_responsibility_block(
                    block, x, c, labels
                )
                spectral_responsibility_loss = \
                    self.compute_spectral_responsibility_loss(
                        shared_output,
                        routed_output,
                        labels,
                        teacher_z,
                        align_coeffs,
                    )
            else:
                x = block(x, c, labels)

            if teacher_z is not None and block_idx in self.align_block_to_idx:
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

        if teacher_z is not None and len(self.align_blocks) > 0:
            repa_loss = repa_loss / len(self.align_blocks)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x, repa_loss, spectral_responsibility_loss
