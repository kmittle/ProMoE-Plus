import math

import torch
import torch.nn.functional as F

from .models_ProMoE_TC_repa_multi_align import AddAuxiliaryLoss, DiT as MultiAlignDiT
from .modules import modulate


class DiT(MultiAlignDiT):
    """Multi-Align with teacher-conditioned routed-expert geometry."""

    def __init__(self, *args, repa_config=None, **kwargs):
        super().__init__(*args, repa_config=repa_config, **kwargs)

        repa_config = repa_config or {}
        self.expert_geometry_block = repa_config.get('expert_geometry_block', 3)
        self.expert_geometry_min_tokens = repa_config.get(
            'expert_geometry_min_tokens', 2
        )
        self.expert_geometry_min_experts = repa_config.get(
            'expert_geometry_min_experts', 3
        )
        self.expert_geometry_teacher_roll = repa_config.get(
            'expert_geometry_teacher_roll', [0, 0]
        )
        self.expert_geometry_eps = repa_config.get('expert_geometry_eps', 1e-6)

        block_idx = self.expert_geometry_block
        if (
            isinstance(block_idx, bool)
            or not isinstance(block_idx, int)
            or not 0 <= block_idx < len(self.blocks)
        ):
            raise ValueError(
                f"expert_geometry_block must be an integer in "
                f"[0, {len(self.blocks) - 1}], got {block_idx!r}"
            )
        if block_idx not in self.align_block_to_idx:
            raise ValueError(
                "expert_geometry_block must also be present in align_blocks, "
                f"got block {block_idx} and align_blocks={self.align_blocks}"
            )
        if not self.blocks[block_idx].use_moe:
            raise ValueError(f"expert_geometry_block {block_idx} must be a MoE block")

        moe_block = self.blocks[block_idx].mlp
        if moe_block.top_k != 1:
            raise ValueError("expert geometry requires top_k == 1")
        if (
            isinstance(self.expert_geometry_min_tokens, bool)
            or not isinstance(self.expert_geometry_min_tokens, int)
            or self.expert_geometry_min_tokens < 1
        ):
            raise ValueError("expert_geometry_min_tokens must be a positive integer")
        if (
            isinstance(self.expert_geometry_min_experts, bool)
            or not isinstance(self.expert_geometry_min_experts, int)
            or not 3 <= self.expert_geometry_min_experts <= moe_block.num_routed_experts
        ):
            raise ValueError(
                "expert_geometry_min_experts must be an integer between 3 and "
                f"{moe_block.num_routed_experts}"
            )
        teacher_roll = self.expert_geometry_teacher_roll
        if (
            not isinstance(teacher_roll, (list, tuple))
            or len(teacher_roll) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in teacher_roll)
        ):
            raise ValueError(
                "expert_geometry_teacher_roll must contain two integer shifts"
            )
        self.expert_geometry_teacher_roll = tuple(teacher_roll)
        if self.expert_geometry_eps <= 0:
            raise ValueError("expert_geometry_eps must be positive")

        print(
            f"Teacher-conditioned expert geometry at block {block_idx}: "
            f"min_tokens={self.expert_geometry_min_tokens}, "
            f"min_experts={self.expert_geometry_min_experts}, "
            f"teacher_roll={self.expert_geometry_teacher_roll}"
        )

    def _roll_teacher_tokens(self, teacher):
        shift_y, shift_x = self.expert_geometry_teacher_roll
        if shift_y == 0 and shift_x == 0:
            return teacher

        batch_size, num_tokens, hidden_size = teacher.shape
        side = math.isqrt(num_tokens)
        if side * side != num_tokens:
            raise ValueError(
                f"expert geometry requires a square token grid, got {num_tokens} tokens"
            )
        spatial = teacher.reshape(batch_size, side, side, hidden_size)
        spatial = torch.roll(spatial, shifts=(shift_y, shift_x), dims=(1, 2))
        return spatial.reshape(batch_size, num_tokens, hidden_size)

    def compute_expert_geometry_loss(
        self,
        expert_outputs,
        expert_assignments,
        labels,
        teacher_z,
    ):
        if teacher_z.shape[:2] != expert_outputs.shape[:2]:
            raise ValueError(
                f"teacher/expert token shapes must match, got {teacher_z.shape[:2]} "
                f"and {expert_outputs.shape[:2]}"
            )
        if expert_assignments.shape != expert_outputs.shape[:2]:
            raise ValueError(
                f"assignment shape must match expert tokens, got "
                f"{expert_assignments.shape} and {expert_outputs.shape[:2]}"
            )

        zero = expert_outputs.float().sum() * 0.0
        cond_indices = torch.where(labels != 1000)[0]
        if cond_indices.numel() == 0:
            return zero

        teacher = self._roll_teacher_tokens(teacher_z.detach().float())
        num_experts = self.blocks[
            self.expert_geometry_block
        ].mlp.num_routed_experts
        image_losses = []

        for image_idx in cond_indices.tolist():
            assignments = expert_assignments[image_idx].detach()
            if ((assignments < 0) | (assignments >= num_experts)).any():
                raise ValueError("conditional expert assignment is out of range")

            counts = torch.bincount(assignments, minlength=num_experts)
            valid = counts >= self.expert_geometry_min_tokens
            if valid.sum().item() < self.expert_geometry_min_experts:
                continue

            student_sums = torch.zeros(
                num_experts,
                expert_outputs.shape[-1],
                dtype=torch.float32,
                device=expert_outputs.device,
            )
            teacher_sums = torch.zeros(
                num_experts,
                teacher.shape[-1],
                dtype=torch.float32,
                device=teacher.device,
            )
            student_sums.index_add_(
                0, assignments, expert_outputs[image_idx].float()
            )
            teacher_sums.index_add_(0, assignments, teacher[image_idx])

            denominators = counts[valid].float().unsqueeze(1)
            student_centroids = student_sums[valid] / denominators
            teacher_centroids = teacher_sums[valid] / denominators
            student_centroids = student_centroids - student_centroids.mean(
                dim=0, keepdim=True
            )
            teacher_centroids = teacher_centroids - teacher_centroids.mean(
                dim=0, keepdim=True
            )

            informative = torch.linalg.vector_norm(
                teacher_centroids, dim=-1
            ) > self.expert_geometry_eps
            if informative.sum().item() < self.expert_geometry_min_experts:
                continue
            student_centroids = student_centroids[informative]
            teacher_centroids = teacher_centroids[informative]

            student_centroids = F.normalize(
                student_centroids, dim=-1, eps=self.expert_geometry_eps
            )
            teacher_centroids = F.normalize(
                teacher_centroids, dim=-1, eps=self.expert_geometry_eps
            )
            student_gram = student_centroids @ student_centroids.T
            teacher_gram = teacher_centroids @ teacher_centroids.T
            pair_mask = torch.triu(
                torch.ones_like(student_gram, dtype=torch.bool), diagonal=1
            )
            image_losses.append(
                F.smooth_l1_loss(
                    student_gram[pair_mask],
                    teacher_gram[pair_mask],
                    reduction='mean',
                )
            )

        if not image_losses:
            return zero
        return torch.stack(image_losses).mean()

    @staticmethod
    def _forward_geometry_block(block, x, c, labels):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            block.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * block.attn(
            modulate(block.norm1(x), shift_msa, scale_msa)
        )
        router_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x_mlp, aux_loss, expert_outputs, expert_assignments = block.mlp(
            router_input, labels, return_expert_trace=True
        )
        if aux_loss is not None:
            x_mlp = AddAuxiliaryLoss.apply(x_mlp, aux_loss)
        x = x + gate_mlp.unsqueeze(1) * x_mlp
        return x, expert_outputs, expert_assignments

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
        expert_geometry_loss = x.float().sum() * 0.0
        for block_idx, block in enumerate(self.blocks):
            if block_idx == self.expert_geometry_block and teacher_z is not None:
                x, expert_outputs, expert_assignments = \
                    self._forward_geometry_block(block, x, c, labels)
                expert_geometry_loss = self.compute_expert_geometry_loss(
                    expert_outputs,
                    expert_assignments,
                    labels,
                    teacher_z,
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
        return x, repa_loss, expert_geometry_loss
