"""ProMoE-TC with a cosine regularizer on the pooled expert representations.

Each MoE block pools every routed expert's output over the tokens that expert
actually received (plain mean, no router weighting), giving one vector per
expert.  The regularizer penalises how similar those vectors are to each other,
measured by **cosine similarity** rather than the historical
``mean(exp(-L2/tau))`` repulsion.  Cosine is scale-free, so it cannot be
satisfied by inflating the outputs and it cannot silently die when tau does not
match the representation scale -- the failure mode that made the historical
expert-output and expert-parameter regularizers inert (their loss was 1e-6 and
exactly 0 respectively).

The penalty is ``mean((cos - target)^2)`` over the unique expert pairs:

* ``expert_cos_center`` (default ``True``) subtracts the mean over the experts
  before measuring, so the loss sees only how the experts differ from each
  other and not the direction they all share.  Centering makes the pairwise
  cosines average exactly ``-1/(K-1)``, so that value becomes the target: the
  loss pulls the K directions towards a regular simplex, the most evenly
  spread configuration, and its minimum of 0 is attainable.  A one-sided
  ``relu(cos - 0)`` would instead be identically zero here -- with K=12 in 768
  dimensions every centered pair already sits near -0.0909 -- i.e. exactly the
  inert-loss failure this model exists to avoid.
* Without centering the target is 0, so the term is ``mean(cos^2)`` and
  penalises correlation of either sign.
* ``expert_cos_margin`` (default ``0``) shifts the target: positive tolerates
  more similarity, negative demands more spread.  Left at 0 it is a no-op.

The gradient reaches the expert parameters (and, through the token
representations, the earlier layers); it does not reach the router directly,
because the token assignment is discrete and the pooled vectors carry no router
weight.  So this term shapes the experts, not the routing.

Only the MoE block inherits from the conference model: ``SparseMoeBlock``
extends ``models_ProMoE_TC.SparseMoeBlock``, reusing its construction, routing
and routing-contrastive loss.  The forward is the conference forward plus the
pooling and the extra loss, so with ``expert_cos_lam = 0`` this model is
bit-identical to ``ProMoE_TC_B``.  Evaluation and sampling are unchanged.

Every block records its own diagnostics (raw and centered mean cosine, max
cosine, the loss, and how many experts took part) so a dead term is visible in
the training log from step 0 instead of being discovered months later.
"""

import torch
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
    which includes the gradient of the aux loss in backward.
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
    """Conference MoE block plus a cosine penalty on the pooled expert outputs."""

    def __init__(self, *args, expert_cos_lam=0.0, expert_cos_center=True,
                 expert_cos_margin=0.0, expert_cos_warmup_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_cos_lam = float(expert_cos_lam)
        self.expert_cos_center = bool(expert_cos_center)
        self.expert_cos_margin = float(expert_cos_margin)
        self.expert_cos_warmup_steps = int(expert_cos_warmup_steps)
        # The weight actually used this step; the DiT rewrites it every forward
        # while warming up, so a block on its own already behaves correctly.
        self.expert_cos_effective_lam = self.expert_cos_lam

        if self.expert_cos_lam < 0:
            raise ValueError(
                f"expert_cos_lam must be >= 0, got {self.expert_cos_lam}"
            )
        if not -1.0 <= self.expert_cos_margin <= 1.0:
            raise ValueError(
                f"expert_cos_margin must lie in [-1, 1], got {self.expert_cos_margin}"
            )
        if self.expert_cos_warmup_steps < 0:
            raise ValueError(
                f"expert_cos_warmup_steps must be >= 0, got {self.expert_cos_warmup_steps}"
            )

        # Diagnostics, kept as detached tensors so reading them costs no
        # device synchronisation; train.py converts them when it logs.
        self.last_expert_cos_loss = None
        self.last_expert_cos_raw_mean = None
        self.last_expert_cos_centered_mean = None
        self.last_expert_cos_max = None
        self.last_expert_cos_active = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        timestep: torch.Tensor = None,
    ):
        ### token assignment
        if self.phase_metric is None:
            router_weights, expert_indices, load_balance_loss = (
                self.compute_router(hidden_states, labels)
            )
        else:
            router_weights, expert_indices, load_balance_loss = (
                self.compute_router(hidden_states, labels, timestep)
            )
        batch_size, seq_len, hidden_dim = hidden_states.shape

        flat_input = hidden_states.view(-1, hidden_dim)
        flat_weights = router_weights.view(-1, self.top_k)
        flat_indices = expert_indices.view(-1, self.top_k)
        total_tokens = batch_size * seq_len

        final_output = torch.zeros(total_tokens, hidden_dim, device=hidden_states.device)

        # Pool the routed experts' outputs only when the term is actually on.
        # The effective weight follows the warm-up ramp set by the DiT.
        need_pools = self.training and self.expert_cos_effective_lam > 0
        expert_pools = []

        ### process routed experts and unconditional expert
        for expert_id in range(self.num_experts):
            expert_mask = (flat_indices == expert_id).any(dim=1)
            token_ids = torch.where(expert_mask)[0]
            if token_ids.numel() > 0:
                expert_input = flat_input[token_ids]
                expert_weight_mask = (flat_indices[token_ids] == expert_id)
                expert_weights = flat_weights[token_ids] * expert_weight_mask.float()
                combined_weights = expert_weights.sum(dim=1)
                expert_output = self.experts[expert_id](expert_input)
                weighted_output = expert_output * combined_weights.unsqueeze(1)
                final_output.index_add_(0, token_ids, weighted_output)

                # Routed experts only: the shared and unconditional experts see
                # a different token population, so they are not comparable.
                if need_pools and expert_id < self.num_routed_experts:
                    expert_pools.append(expert_output.mean(dim=0))
            else:
                dummy_input = torch.zeros(1, hidden_dim, device=hidden_states.device)
                dummy_output = self.experts[expert_id](dummy_input).float()
                final_output[0] += dummy_output[0] * 0

        final_output = final_output.view(batch_size, seq_len, hidden_dim)

        ### process shared experts
        if self.use_shared_expert:
            shared_output = self.shared_expert(hidden_states)
            final_output += shared_output

        loss = load_balance_loss  # None
        ### routing contrastive loss
        if self.training and self.routing_contrastive_lam > 0:
            flat_labels = labels.view(batch_size, 1).expand(-1, seq_len).reshape(-1)
            if self.use_uncond_expert:
                uncond_mask = (flat_labels == 1000)
                cond_mask = ~uncond_mask
            else:
                cond_mask = torch.ones(batch_size * seq_len, dtype=torch.bool, device=hidden_states.device)

            cond_token_embeddings = flat_input[cond_mask]  # [num_cond_tokens, hidden_dim]

            if self.use_top_k_for_routing_contrastive:
                # top-k
                topk_expert_indices = expert_indices.view(batch_size * seq_len, self.top_k)[cond_mask]  # [num_cond_tokens, top_k]
                cond_cluster_assignments = topk_expert_indices
            else:
                # top-1
                top1_expert_indices = expert_indices.view(batch_size * seq_len, self.top_k)[:, 0]  # [batch_size * seq_len]
                cond_cluster_assignments = top1_expert_indices[cond_mask]  # [num_cond_tokens]

            routing_contrastive_loss = self.compute_routing_contrastive_loss(
                cond_token_embeddings,
                cond_cluster_assignments,
                use_top_k=self.use_top_k_for_routing_contrastive
            )

            routing_contrastive_loss = routing_contrastive_loss * self.routing_contrastive_lam
            if loss is not None:
                loss += routing_contrastive_loss
            else:
                loss = routing_contrastive_loss

        ### expert representation cosine loss
        if need_pools:
            expert_cos_loss = self.compute_expert_cosine_loss(expert_pools)
            expert_cos_loss = expert_cos_loss * self.expert_cos_effective_lam
            if loss is not None:
                loss = loss + expert_cos_loss
            else:
                loss = expert_cos_loss

        return final_output, loss

    def compute_expert_cosine_loss(self, expert_pools):
        """Penalise pairwise cosine similarity between pooled expert outputs.

        expert_pools: list of [hidden_dim] tensors, one per routed expert that
        received at least one token.  Experts with no tokens are skipped; over
        the unique pairs (i < j) of the remaining ones the loss is
        ``mean((cos - target)^2)`` with the target from ``_cosine_target``.

        Note the -1/(K-1) identity behind that target holds exactly only when
        the pools share one norm: ``sum(c_i) = 0`` fixes the sum of the inner
        products, while each cosine also divides by its own two norms.
        """
        device = self.cluster_centers.device
        if len(expert_pools) < 2:
            self._record_cosine_stats(None, None, len(expert_pools))
            return torch.zeros((), device=device)

        # fp32: under bf16 autocast, normalize + matmul drift enough to matter
        # for a quantity that is compared against a margin.
        pooled = torch.stack(expert_pools).float()  # [K, hidden_dim]
        centered = pooled - pooled.mean(dim=0, keepdim=True)

        raw_sim = self._pairwise_cosine(pooled)
        centered_sim = self._pairwise_cosine(centered)
        sim = centered_sim if self.expert_cos_center else raw_sim

        target = self._cosine_target(len(expert_pools))
        loss = (sim - target).square().mean()
        self._record_cosine_stats(raw_sim, centered_sim, len(expert_pools), loss)
        return loss.to(device)

    def _cosine_target(self, num_experts):
        """The cosine value the loss pulls every pair towards.

        Centering forces the pairwise cosines to average exactly -1/(K-1): with
        ``sum(c_i) = 0``, ``|sum(c_i)|^2 = 0`` gives
        ``sum_{i<j} <c_i, c_j> = -0.5 * sum |c_i|^2``.  That value is therefore
        the floor, not something to push below, and it is reached exactly when
        the K directions form a regular simplex -- the most evenly spread
        configuration available.  Pulling the cosines towards it penalises any
        departure from even spread in either direction, and unlike
        ``relu(cos - 0)`` it cannot be trivially satisfied: with K=12 in 768
        dimensions every centered pair already sits near -0.0909, so a relu at 0
        would clip the whole term to zero and the regularizer would be inert.

        Without centering there is no such constraint, so the target is 0 and
        the term becomes ``mean(cos^2)``, penalising correlation of any sign.
        ``expert_cos_margin`` shifts the target: positive tolerates more
        similarity, negative demands more spread.
        """
        base = -1.0 / (num_experts - 1) if self.expert_cos_center else 0.0
        return base + self.expert_cos_margin

    @staticmethod
    def _pairwise_cosine(vecs):
        """Cosine similarity of every unique pair (i < j) of rows.

        ``F.normalize`` maps an all-zero row to an all-zero row, so a degenerate
        expert yields similarity 0 rather than NaN.
        """
        normed = F.normalize(vecs, p=2, dim=1)
        sim = (normed @ normed.T).clamp(-1.0, 1.0)
        num = vecs.size(0)
        mask = torch.triu(
            torch.ones(num, num, device=vecs.device, dtype=torch.bool), diagonal=1
        )
        return sim[mask]

    def _record_cosine_stats(self, raw_sim, centered_sim, num_active, loss=None):
        """Store this step's diagnostics as detached tensors (no .item() here)."""
        if raw_sim is None or raw_sim.numel() == 0:
            self.last_expert_cos_loss = None
            self.last_expert_cos_raw_mean = None
            self.last_expert_cos_centered_mean = None
            self.last_expert_cos_max = None
            self.last_expert_cos_active = float(num_active)
            return
        self.last_expert_cos_loss = loss.detach() if loss is not None else None
        self.last_expert_cos_raw_mean = raw_sim.detach().mean()
        self.last_expert_cos_centered_mean = centered_sim.detach().mean()
        self.last_expert_cos_max = raw_sim.detach().max()
        self.last_expert_cos_active = float(num_active)


#################################################################################
#                                 Core DiT Model                                #
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
    Diffusion model with a Transformer backbone and a cosine regularizer on the
    pooled expert representations.
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
        # Every MoE block carries the term; the model exposes them for train.py.
        self.is_expert_cos_model = True
        self.expert_cos_lam = float(MoE_config.get("expert_cos_lam", 0.0))
        self.expert_cos_warmup_steps = int(MoE_config.get("expert_cos_warmup_steps", 0))
        self.expert_cos_last_state = {"lam": self.expert_cos_lam, "scale": 1.0}

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

    def _set_expert_cos_state(self, training_step):
        """Ramp the regularizer weight over the first ``warmup`` steps.

        The term is relatively strongest exactly when the experts are still
        forming: measured on this model, at step 0 it puts 1.9x the MSE
        gradient on the routed experts against 1.05x at 300K.  Ramping in
        keeps it from steering that early phase, and because every arm uses
        the same ramp it is not an extra factor between arms.
        """
        scale = 1.0
        if self.training and self.expert_cos_lam > 0 and self.expert_cos_warmup_steps > 0:
            if training_step is None:
                raise ValueError(
                    "expert_cos warm-up needs training_step; train.py passes it "
                    "for EXPERT_COS_MODELS"
                )
            scale = min(1.0, (int(training_step) + 1) / self.expert_cos_warmup_steps)
        effective = self.expert_cos_lam * scale
        self.expert_cos_last_state = {"lam": effective, "scale": scale}
        for block in self.blocks:
            if block.use_moe:
                block.mlp.expert_cos_effective_lam = effective

    def moe_blocks(self):
        """The MoE blocks, in depth order -- used to collect the diagnostics."""
        return [block.mlp for block in self.blocks if block.use_moe]

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
        training_step (kwarg): global step driving the warm-up ramp
        """
        self._set_expert_cos_state(kwargs.get("training_step"))
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
