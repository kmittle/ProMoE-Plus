"""ProMoE-TC retaining shared experts and adding conditional image-wise experts.

The fixed shared expert still processes every image. Conditional images additionally
use top-k token-routed experts per token and image-routed experts for the whole image;
unconditional images use only the shared and dedicated unconditional experts.
Widths are independent configuration values so the FFN budget can be apportioned
without changing the pooling, prototype router, or contrastive objective.
"""

import torch

from .models_ProMoE_TC import DiT as BaseDiT, SparseMoeBlock as BaseSparseMoeBlock
from .models_ProMoE_TC_image_shared import ImageWiseSharedExpertPool, _init_linear
from .modules import MoeMLP


def _positive_width(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class SparseMoeBlock(BaseSparseMoeBlock):
    """Keep the base shared/token branches and add a conditional image branch."""

    def __init__(
        self,
        *args,
        image_expert_intermediate_size=None,
        uncond_expert_intermediate_size=None,
        num_image_experts=3,
        image_top_k=1,
        image_expert_backend="bmm",
        image_pool_num_heads=1,
        image_routing_contrastive_lam=1.0,
        image_routing_contrastive_temperature=0.07,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not self.use_shared_expert or not self.use_uncond_expert:
            raise ValueError("conditional image experts require shared and unconditional experts")
        if (isinstance(self.top_k, bool) or not isinstance(self.top_k, int)
                or not 1 <= self.top_k <= self.num_routed_experts):
            raise ValueError("token top_k must be an integer in [1, num_routed_experts]")
        token_width = self.experts[0].intermediate_size
        image_width = _positive_width(
            token_width if image_expert_intermediate_size is None else image_expert_intermediate_size,
            "image_expert_intermediate_size",
        )
        uncond_width = _positive_width(
            self.shared_expert.intermediate_size
            if uncond_expert_intermediate_size is None else uncond_expert_intermediate_size,
            "uncond_expert_intermediate_size",
        )
        if self.experts[-1].intermediate_size != uncond_width:
            self.experts[-1] = MoeMLP(self.hidden_size, uncond_width)
            self.experts[-1].apply(_init_linear)
        self.image_experts = ImageWiseSharedExpertPool(
            hidden_size=self.hidden_size,
            intermediate_size=image_width,
            num_image_experts=num_image_experts,
            top_k=image_top_k,
            expert_backend=image_expert_backend,
            pool_num_heads=image_pool_num_heads,
            routing_contrastive_lam=image_routing_contrastive_lam,
            routing_contrastive_temperature=image_routing_contrastive_temperature,
        )

    def forward(self, hidden_states, labels, timestep=None):
        output, loss = super().forward(hidden_states, labels, timestep)
        # These are the post-label-dropout labels supplied by the inherited DiTBlock.
        # Filter before pooling, routing, expert execution, and image contrastive loss.
        cond_ids = torch.where(labels.reshape(-1) != 1000)[0]
        if cond_ids.numel() > 0:
            image_output = self.image_experts(hidden_states.index_select(0, cond_ids))
            output = output.index_add(0, cond_ids, image_output.to(output.dtype))
        else:
            # Clear previous-forward diagnostics, and keep every image parameter in
            # DDP's graph without running a dummy pooling or expert forward.
            self.image_experts.last_expert_indices = (
                cond_ids.detach() if self.image_experts.top_k == 1
                else cond_ids.new_empty((0, self.image_experts.top_k))
            )
            self.image_experts.last_routing_contrastive_loss = None
            if self.training and torch.is_grad_enabled():
                zero = sum(parameter.reshape(-1)[0] * 0.0
                           for parameter in self.image_experts.parameters())
                # With no conditional tokens the inherited token prototypes are
                # unused too; retain their zero gradient for DDP's next iteration.
                zero = zero + self.cluster_centers.reshape(-1)[0] * 0.0
                output = output + zero.to(output.dtype)
        return output, loss


class DiT(BaseDiT):
    """Independent additive-image-expert ablation; the fixed shared expert stays."""

    def __init__(self, *args, MoE_config=None, **kwargs):
        if MoE_config is None:
            raise ValueError("conditional image experts require MoE_config")
        super().__init__(*args, MoE_config=MoE_config, **kwargs)
        for block in self.blocks:
            if not block.use_moe:
                continue
            original = block.mlp
            replacement = SparseMoeBlock(hidden_size=original.hidden_size, **MoE_config)
            # Preserve the initialized base branches and router. Only a resized
            # unconditional expert and the new image branch need new parameters.
            replacement_state = replacement.state_dict()
            inherited = {
                name: value for name, value in original.state_dict().items()
                if name in replacement_state and value.shape == replacement_state[name].shape
            }
            missing, unexpected = replacement.load_state_dict(inherited, strict=False)
            uncond_prefix = f"experts.{replacement.num_experts - 1}."
            if unexpected or any(
                not name.startswith(("image_experts.", uncond_prefix)) for name in missing
            ):
                raise RuntimeError("unexpected mismatch while inheriting base MoE parameters")
            block.mlp = replacement

    def set_image_expert_backend(self, backend):
        for block in self.blocks:
            if block.use_moe:
                block.mlp.image_experts.set_expert_backend(backend)
