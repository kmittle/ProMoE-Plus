"""Training-only MLP + Sigmoid projection before clean-teacher routing KL.

The existing temperature softmax normalizes the projected student scores.
Actual expert selection, gates, and student routing contrastive use raw scores.
"""
import torch
import torch.nn as nn

from .models_ProMoE_TC_clean_route import DiT as CleanRouteDiT
from .models_ProMoE_TC_clean_route import SparseMoeBlock as CleanRouteBlock


class SparseMoeBlock(CleanRouteBlock):
    def init_alignment_projector(self, hidden_dim, seed):
        # Append after backbone initialization without changing its RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.alignment_projector = nn.Sequential(
                nn.Linear(self.num_routed_experts, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, self.num_routed_experts), nn.Sigmoid(),
            )
            # torch.manual_seed would also reset every CUDA generator, which
            # fork_rng(devices=[]) does not restore. Keep rank-specific noise
            # and sampling streams intact with a private CPU generator.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            for layer in self.alignment_projector:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, generator=generator)
                    nn.init.zeros_(layer.bias)

    def _student_losses(self, conference_loss, indices, weights, cosines, routed, seq_len):
        with torch.autocast(device_type=cosines.device.type, enabled=False):
            projected = self.alignment_projector(cosines.float())
        loss = super()._student_losses(conference_loss, indices, weights, projected, routed, seq_len)
        # Every DDP rank participates, including an all-unconditional batch or
        # a zero-weight KL schedule. This does not add a nonzero loss.
        anchor = projected.sum() * 0.0
        return anchor if loss is None else loss + anchor


class DiT(CleanRouteDiT):
    moe_block_class = SparseMoeBlock

    def __init__(self, *args, alignment_projector_config=None, **kwargs):
        config = {} if alignment_projector_config is None else dict(alignment_projector_config)
        if set(config) != {"hidden_dim"} or type(config["hidden_dim"]) is not int or config["hidden_dim"] < 1:
            raise ValueError("alignment_projector_config requires a positive integer hidden_dim")
        super().__init__(*args, **kwargs)
        if self.teacher_sigma_ratio != 0 or any(b.contrastive_source != "student" for b in self.moe_blocks()):
            raise ValueError("alignment projector requires teacher_sigma_ratio: 0 and student contrastive")
        for index, block in enumerate(self.moe_blocks()):
            block.init_alignment_projector(config["hidden_dim"], seed=20261102 + index)


__all__ = ["DiT", "SparseMoeBlock"]
