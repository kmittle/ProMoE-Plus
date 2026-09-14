"""ProMoE-TC whose routing-contrastive class centers use the global batch.

The conference routing-contrastive loss averages, separately on every GPU, the
cond tokens routed to each expert and contrasts the prototypes with these
per-GPU class centers, so the centers are noisier and depend on the GPU count.
This model computes each expert's class center over the whole global batch:

* the forward all-reduces each expert's token sum and token count, so every
  rank builds the same global class centers and computes the same loss;
* the other ranks' sums enter as plain values -- no gradient crosses
  processes;
* the gradient through this rank's own tokens is multiplied by the world
  size.  Every rank computes the identical loss, but a token receives
  gradient only on its own rank, and DDP divides all gradients by the world
  size.  The factor restores the token gradient, so after DDP the update
  equals one loss computed on the whole global batch.  The prototype gradient
  is already identical on every rank and needs no factor.

Only the MoE block inherits from the conference model: ``SparseMoeBlock``
extends ``models_ProMoE_TC.SparseMoeBlock`` and replaces nothing but the
contrastive loss.  Token routing, the forward pass, evaluation and sampling
are unchanged, and a single process is bit-identical to the conference model.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from .models_ProMoE_TC import SparseMoeBlock as ConferenceSparseMoeBlock
from .modules import get_2d_sincos_pos_embed, Attention, modulate, TimestepEmbedder, LabelEmbedder, FinalLayer, MoeMLP, Mlp


#################################################################################
#                                 ProMoE Layer                                  #
#################################################################################
class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class SparseMoeBlock(ConferenceSparseMoeBlock):
    """Conference TC block whose contrastive class centers use the global batch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.phase_metric is not None:
            raise ValueError("global_center does not support phase_metric_config.enabled")

    def compute_routing_contrastive_loss(self, token_embeddings, cluster_assignments, use_top_k=False):
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            # A single process already sees the whole batch.
            return super().compute_routing_contrastive_loss(
                token_embeddings, cluster_assignments, use_top_k=use_top_k
            )
        local_sums, local_counts = self._cluster_sums(token_embeddings, cluster_assignments, use_top_k)
        # Every rank reaches this call once per MoE block, in the same order,
        # even when it holds no cond token, so the collective always matches.
        stats = torch.cat([local_sums.detach(), local_counts.unsqueeze(1)], dim=1)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return self._global_center_contrastive_loss(
            local_sums, stats[:, :-1], stats[:, -1], dist.get_world_size()
        )

    def _cluster_sums(self, token_embeddings, cluster_assignments, use_top_k):
        """Per-expert token sums (with gradient) and token counts on this rank."""
        sums = []
        counts = []
        for cluster_id in range(self.cluster_centers.size(0)):
            if use_top_k:
                mask = (cluster_assignments == cluster_id).any(dim=1)
            else:
                mask = cluster_assignments == cluster_id
            sums.append(token_embeddings[mask].float().sum(dim=0))
            counts.append(mask.sum())
        return torch.stack(sums), torch.stack(counts).float()

    def _global_center_contrastive_loss(self, local_sums, global_sums, global_counts, world_size):
        """Contrast prototypes with global class centers.

        ``global_sums`` and ``global_counts`` are the all-reduced values and
        carry no gradient.  The centers take exactly their values; only this
        rank's ``local_sums`` carry gradient, multiplied by ``world_size``.
        """
        valid_clusters = torch.nonzero(global_counts > 0, as_tuple=False).flatten()
        if valid_clusters.numel() < 2:
            return torch.tensor(0.0, device=self.cluster_centers.device)
        sums = global_sums + world_size * (local_sums - local_sums.detach())
        cluster_means = sums[valid_clusters] / global_counts[valid_clusters].unsqueeze(1)
        centers_norm = F.normalize(self.cluster_centers[valid_clusters], p=2, dim=1)
        means_norm = F.normalize(cluster_means, p=2, dim=1)
        sim_matrix = centers_norm @ means_norm.T
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        return F.cross_entropy(sim_matrix / self.routing_contrastive_temperature, labels)


#################################################################################
#                                 Core ProMoE Model                             #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, head_dim=None, mlp_ratio=4.0,
                 use_swiglu=False, MoE_config=None,
                 use_moe=False,
                 **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.use_moe = use_moe
        if use_moe:
            self.mlp = SparseMoeBlock(hidden_size=hidden_size, **MoE_config)
        elif use_swiglu:
            self.mlp = MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim)
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, label):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_moe:
            x_mlp, aux_loss = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), label)
            if aux_loss is not None:
                x_mlp = AddAuxiliaryLoss.apply(x_mlp, aux_loss)
            return x + gate_mlp.unsqueeze(1) * x_mlp
        return x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone and global contrastive class centers.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        qk_norm=False,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        use_swiglu=False,
        MoE_config=None,
        head_dim=None,
    ):
        super().__init__()
        if MoE_config.init_MoeMLP:
            raise ValueError("init_MoeMLP is not supported: MoeMLP has no gate_proj")
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.MoE_config = MoE_config
        use_moe_flag = [True] * depth
        if self.MoE_config.interleave:
            use_moe_flag = [i % 2 == 1 for i in range(depth)]

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob, return_labels=True)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm,
                     use_swiglu=use_swiglu, MoE_config=MoE_config, use_moe=use_moe_flag[i]) for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, timestep, context, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        timestep: (N,) tensor of diffusion timesteps
        context: (N,) tensor of class labels
        """
        y = context
        if len(x.shape) != 4:
            x = x.squeeze(2)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)                   # (N, D)
        y, labels = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c, labels)                     # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        if isinstance(model_out, tuple):
            model_out = model_out[0]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


__all__ = ["DiT", "DiTBlock", "SparseMoeBlock"]
