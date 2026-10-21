"""ProMoE-TC with an independently routed, image-wise shared-expert pool.

Only the shared branch is replaced. The base token router, conditional/unconditional
experts, token routing-contrastive loss, attention, and block residuals are inherited.
Each MoE layer pools each image with one learned cross-attention query and a Linear,
then selects one of its own image experts by cosine similarity to learned prototypes.
The selected cosine score weights that image's expert output, as in the base TC router.

Both execution backends use the same parameters and routing decisions: ``bmm`` gathers
expert weights per image, while ``loop`` groups images by expert and calls its Linear
modules. FLOPs instrumentation temporarily selects ``loop`` so those modules are seen.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_ProMoE_TC import AddAuxiliaryLoss, DiT as BaseDiT, SparseMoeBlock
from .modules import MoeMLP


def _init_linear(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ImageAttentionPooling(nn.Module):
    """Read one image embedding without updating its input token sequence."""

    def __init__(self, hidden_size, num_heads=1):
        super().__init__()
        if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
            raise ValueError("image_pool_num_heads must be a positive integer")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by image_pool_num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.empty(1, 1, hidden_size))
        # The query is learned directly in attention space; K/V come from image tokens.
        self.key_value = nn.Linear(hidden_size, 2 * hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.apply(_init_linear)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, hidden_states):
        batch, tokens, channels = hidden_states.shape
        if tokens == 0:
            raise ValueError("image attention pooling requires at least one token")
        head_dim = channels // self.num_heads
        kv = self.key_value(hidden_states).reshape(
            batch, tokens, 2, self.num_heads, head_dim
        ).permute(2, 0, 3, 1, 4)
        key, value = kv.unbind(0)
        query = self.query.expand(batch, -1, -1).reshape(
            batch, 1, self.num_heads, head_dim
        ).transpose(1, 2).to(key.dtype)
        pooled = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        return self.projection(pooled.transpose(1, 2).reshape(batch, channels))


class ImageWiseSharedExpertPool(nn.Module):
    """One top-1 image expert per image, shared by all tokens of that image."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_image_experts=3,
        expert_backend="bmm",
        pool_num_heads=1,
        routing_contrastive_lam=1.0,
        routing_contrastive_temperature=0.07,
        first_expert=None,
    ):
        super().__init__()
        if (isinstance(num_image_experts, bool) or not isinstance(num_image_experts, int)
                or num_image_experts < 1):
            raise ValueError("num_image_experts must be a positive integer")
        if not math.isfinite(routing_contrastive_temperature) or routing_contrastive_temperature <= 0:
            raise ValueError("image routing contrastive temperature must be finite and positive")
        if not math.isfinite(routing_contrastive_lam) or routing_contrastive_lam < 0:
            raise ValueError("image routing contrastive coefficient must be finite and nonnegative")
        self.hidden_size = hidden_size
        self.num_image_experts = num_image_experts
        self.routing_contrastive_lam = routing_contrastive_lam
        self.routing_contrastive_temperature = routing_contrastive_temperature
        self.pooling = ImageAttentionPooling(hidden_size, pool_num_heads)
        self.cluster_centers = nn.Parameter(torch.empty(num_image_experts, hidden_size))
        nn.init.normal_(self.cluster_centers, std=0.02)
        experts = []
        if first_expert is not None:
            if (first_expert.hidden_size != hidden_size
                    or first_expert.intermediate_size != intermediate_size):
                raise ValueError("the inherited shared expert has incompatible dimensions")
            experts.append(first_expert)
        for _ in range(num_image_experts - len(experts)):
            expert = MoeMLP(hidden_size, intermediate_size)
            expert.apply(_init_linear)
            experts.append(expert)
        self.experts = nn.ModuleList(experts)
        self.set_expert_backend(expert_backend)
        # Detached inspection data only; not parameters, buffers, or checkpoint state.
        self.last_expert_indices = None
        self.last_routing_contrastive_loss = None

    def set_expert_backend(self, backend):
        if backend not in ("bmm", "loop"):
            raise ValueError("image_expert_backend must be 'bmm' or 'loop'")
        self.expert_backend = backend

    def compute_router(self, image_embeddings):
        # Score in FP32 under mixed precision, retaining FP64 for numerical checks.
        score_dtype = (torch.float32 if image_embeddings.dtype in (torch.float16, torch.bfloat16)
                       else image_embeddings.dtype)
        with torch.autocast(device_type=image_embeddings.device.type, enabled=False):
            images = F.normalize(image_embeddings.to(score_dtype), dim=-1)
            centers = F.normalize(self.cluster_centers.to(score_dtype), dim=-1)
            scores = images @ centers.T
            selected_scores, expert_indices = scores.max(dim=-1)
        return selected_scores, expert_indices

    def _bmm_linear(self, hidden_states, expert_indices, projection):
        linears = [getattr(expert, projection) for expert in self.experts]
        weights = torch.stack([linear.weight for linear in linears])
        biases = torch.stack([linear.bias for linear in linears])
        # Cast the small expert bank before gathering the per-image copies under AMP.
        # This avoids materializing a full FP32 copy before bmm casts it to BF16/FP16.
        if torch.is_autocast_enabled(hidden_states.device.type) and weights.dtype != torch.float64:
            dtype = torch.get_autocast_dtype(hidden_states.device.type)
            weights = weights.to(dtype)
            biases = biases.to(dtype)
        image_weights = weights.index_select(0, expert_indices)
        image_biases = biases.index_select(0, expert_indices)
        output = torch.bmm(hidden_states, image_weights.transpose(1, 2))
        return output + image_biases[:, None, :]

    def _forward_bmm(self, hidden_states, expert_indices):
        hidden = self._bmm_linear(hidden_states, expert_indices, "up_proj")
        hidden = F.gelu(hidden, approximate="tanh")
        return self._bmm_linear(hidden, expert_indices, "down_proj").to(hidden_states.dtype)

    def _forward_loop(self, hidden_states, expert_indices):
        output = torch.zeros_like(hidden_states)
        for expert_id, expert in enumerate(self.experts):
            image_ids = torch.where(expert_indices == expert_id)[0]
            if image_ids.numel() > 0:
                expert_output = expert(hidden_states.index_select(0, image_ids))
                output = output.index_copy(0, image_ids, expert_output.to(output.dtype))
            elif self.training and torch.is_grad_enabled():
                # Keep unused parameters in DDP's graph without fake Linear calls
                # that would be mistaken for real expert computation by FLOPs hooks.
                zero = sum(parameter.reshape(-1)[0] * 0.0 for parameter in expert.parameters())
                output = output + zero.to(output.dtype)
        return output

    def forward(self, hidden_states):
        image_embeddings = self.pooling(hidden_states)
        selected_scores, expert_indices = self.compute_router(image_embeddings)
        self.last_expert_indices = expert_indices.detach()
        if self.expert_backend == "bmm":
            output = self._forward_bmm(hidden_states, expert_indices)
        else:
            output = self._forward_loop(hidden_states, expert_indices)
        output = output * selected_scores.to(output.dtype)[:, None, None]
        self.last_routing_contrastive_loss = None
        if self.training and self.routing_contrastive_lam > 0:
            # Reuse the base objective exactly, substituting image embeddings for
            # token embeddings. Cond and uncond images both use this shared branch.
            loss = SparseMoeBlock.compute_routing_contrastive_loss(
                self, image_embeddings, expert_indices, use_top_k=False
            ) * self.routing_contrastive_lam
            self.last_routing_contrastive_loss = loss.detach()
            output = AddAuxiliaryLoss.apply(output, loss)
        return output


class DiT(BaseDiT):
    """Independent ablation of the base TC model's shared branch only."""

    def __init__(self, *args, MoE_config=None, **kwargs):
        if MoE_config is None or not MoE_config.get("use_shared_expert", True):
            raise ValueError("image-wise shared experts require use_shared_expert=True")
        super().__init__(*args, MoE_config=MoE_config, **kwargs)
        for block in self.blocks:
            if not block.use_moe:
                continue
            shared = block.mlp.shared_expert
            block.mlp.shared_expert = ImageWiseSharedExpertPool(
                hidden_size=shared.hidden_size,
                intermediate_size=shared.intermediate_size,
                num_image_experts=MoE_config.get("num_image_experts", 3),
                expert_backend=MoE_config.get("image_expert_backend", "bmm"),
                pool_num_heads=MoE_config.get("image_pool_num_heads", 1),
                routing_contrastive_lam=MoE_config.get("image_routing_contrastive_lam", 1.0),
                routing_contrastive_temperature=MoE_config.get(
                    "image_routing_contrastive_temperature", 0.07
                ),
                first_expert=shared,
            )

    def set_image_expert_backend(self, backend):
        for block in self.blocks:
            if block.use_moe:
                block.mlp.shared_expert.set_expert_backend(backend)
