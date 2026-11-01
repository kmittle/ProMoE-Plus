"""Clean-teacher distillation with a local-token routing corrector.

d = MLP(stop_grad(current_token), stop_grad(raw_scores)). Both actual routing
and student KL use s+d without detaching either residual operand. The teacher
uses clean images and raw routing. No estimated reference, dropout, or extra
reference-generation pass is involved, in training or sampling.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_ProMoE_TC_clean_route import DiT as CleanRouteDiT
from .models_ProMoE_TC_clean_route_corr_router import SparseMoeBlock as CleanCorrBlock
from .models_ProMoE_TC_corr_router import CORRECTOR_INIT_SEED


class TokenRoutingCorrector(nn.Module):
    def __init__(self, hidden_size, num_experts, hidden_dim, generator):
        super().__init__()
        self.token_proj = nn.Linear(hidden_size, hidden_dim)
        self.score_proj = nn.Linear(num_experts, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, num_experts)
        with torch.no_grad():
            for layer in (self.token_proj, self.score_proj):
                fan_out, fan_in = layer.weight.shape
                bound = math.sqrt(6.0 / (fan_in + fan_out))
                layer.weight.copy_(torch.rand(layer.weight.shape, generator=generator) * (2 * bound) - bound)
                if layer.bias is not None:
                    layer.bias.zero_()
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, tokens, scores):
        return self.out(F.silu(self.token_proj(tokens) + self.score_proj(scores)))


class SparseMoeBlock(CleanCorrBlock):
    def init_token_corrector(self, hidden_dim, generator):
        if self.router_weight_mode != "identity" or not self.use_uncond_expert:
            raise ValueError("token corrector requires identity router weights and the uncond expert")
        with torch.random.fork_rng(devices=[]):
            self.corrector = TokenRoutingCorrector(
                self.hidden_size, self.num_routed_experts, hidden_dim, generator
            )

    def forward(self, hidden_states, labels, timestep=None):
        if self._phase == "teacher":
            return super().forward(hidden_states, labels, timestep)
        self.set_corr_context({"tokens": hidden_states.detach()})
        try:
            return super().forward(hidden_states, labels, timestep)
        finally:
            self.set_corr_context(None)

    def _correction(self, cos_sim, cond_positions, seq_len, context):
        tokens = context["tokens"].reshape(-1, self.hidden_size)[cond_positions]
        return self.corrector(tokens, cos_sim.detach())


class DiT(CleanRouteDiT):
    moe_block_class = SparseMoeBlock

    def __init__(self, *args, token_corrector_config=None, **kwargs):
        config = {} if token_corrector_config is None else dict(token_corrector_config)
        if set(config) != {"hidden_dim"} or type(config["hidden_dim"]) is not int or config["hidden_dim"] < 1:
            raise ValueError("token_corrector_config requires a positive integer hidden_dim")
        super().__init__(*args, **kwargs)
        if self.teacher_sigma_ratio != 0 or any(b.contrastive_source != "student" for b in self.moe_blocks()):
            raise ValueError("token corrector requires teacher_sigma_ratio: 0 and student contrastive")
        generator = torch.Generator().manual_seed(CORRECTOR_INIT_SEED)
        for block in self.moe_blocks():
            block.init_token_corrector(config["hidden_dim"], generator)


__all__ = ["DiT", "SparseMoeBlock"]
