import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as F
from .modules import get_2d_sincos_pos_embed, Attention, modulate, TimestepEmbedder, LabelEmbedder, FinalLayer, MoeMLP, Mlp, PrototypeMLP


#################################################################################
#                         ProMoE_EC_BC proto_t Layer                             #
# Batch-Choice EC + timestep-conditioned prototype routing. Each cond sample's   #
# prototypes are produced by PrototypeMLP using its own timestep embedding, so   #
# cosine similarities are computed within a noise-level-aligned feature space    #
# even though the EC topk selects across the whole batch-flattened cond pool.    #
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

class SparseMoeBlock(nn.Module):
    def __init__(
        self, 
        num_routed_experts, 
        hidden_size, 
        moe_intermediate_size, 
        shared_expert_intermediate_size,
        top_k=1,
        load_balance_loss_coef=0,
        norm_topk_prob=False,
        seq_aux=False,
        use_shared_expert=True,
        use_uncond_expert=True,
        router_weight_mode="softmax",
        routing_contrastive_lam=0,
        use_top_k_for_routing_contrastive=False,
        routing_contrastive_temperature=0.1,
        proto_t_update_mode="residual",
        **kwargs,
    ):
        super().__init__()
        if use_uncond_expert:
            self.num_experts = num_routed_experts + 1
        else:
            self.num_experts = num_routed_experts
        self.num_routed_experts = num_routed_experts
        self.seq_aux = seq_aux
        self.hidden_size = hidden_size
        self.top_k = top_k

        self.cluster_centers = nn.Parameter(torch.randn(num_routed_experts, hidden_size))

        # Timestep-conditioned prototype generator. Step 0 behavior is bit-identical
        # to base ProMoE-EC-BC (residual: zero delta; direct: identity-init proto_proj).
        self.prototype_mlp = PrototypeMLP(
            prototype_dim=hidden_size,
            time_dim=hidden_size,
            output_dim=hidden_size,
            intermediate_dim=2 * hidden_size,
            update_mode=proto_t_update_mode,
        )
        
        self.alpha = load_balance_loss_coef
        self.use_shared_expert = use_shared_expert
        self.use_uncond_expert = use_uncond_expert
        self.router_weight_mode = router_weight_mode

        self.routing_contrastive_lam = routing_contrastive_lam
        self.use_top_k_for_routing_contrastive = use_top_k_for_routing_contrastive
        self.routing_contrastive_temperature = routing_contrastive_temperature
        
        self.experts = nn.ModuleList(
            [MoeMLP(hidden_size=hidden_size, intermediate_size=moe_intermediate_size) 
             for _ in range(self.num_experts)]
        )
        
        if use_shared_expert:
            self.shared_expert = MoeMLP(
                hidden_size=hidden_size, 
                intermediate_size=shared_expert_intermediate_size
            )
        
        self._init_weights()

    def compute_router(self, cond_hidden_states, prototype_t):
        """
        Batch-Choice routing with per-sample timestep-conditioned prototypes.
        Cosine similarity for token (i, s) uses sample i's own prototype_t[i];
        the topk dispatch over (B_cond * S) is still batch-wide.

        Args:
            cond_hidden_states: (B_cond, S, D)
            prototype_t:        (B_cond, E_cond, D) per-sample prototypes
        Returns:
            flat_indices:  (E_cond, k) — flat token indices in [0, B_cond*S)
            gating_scores: (E_cond, k)
            expert_inputs: (E_cond, k, D)
        """
        B_cond, S, D = cond_hidden_states.shape
        num_cond_experts = self.num_routed_experts
        T = B_cond * S  # total cond tokens across the batch

        input_norm = F.normalize(cond_hidden_states, p=2, dim=-1)      # (B_cond, S, D)
        proto_norm = F.normalize(prototype_t, p=2, dim=-1)             # (B_cond, E, D)
        # cos_sim[b, s, e] = <input_norm[b, s], proto_norm[b, e]>
        cos_sim = torch.einsum('bsd,bed->bse', input_norm, proto_norm) # (B_cond, S, E)

        flat_input = cond_hidden_states.reshape(T, D)  # (T, D), still needed for expert_inputs gather
        # Permute to (E, B_cond*S) so topk over T matches the batch-flatten dispatch.
        cos_sim_expert_view = cos_sim.permute(2, 0, 1).reshape(num_cond_experts, T)

        if self.router_weight_mode == "softmax":
            cond_weights = F.softmax(cos_sim_expert_view, dim=-1)
        elif self.router_weight_mode == "sigmoid":
            sigmoid_scale = 1.0
            cond_weights = torch.sigmoid(cos_sim_expert_view * sigmoid_scale)
        elif self.router_weight_mode == "identity":
            cond_weights = cos_sim_expert_view
        else:
            raise ValueError(f"Unsupported router_weight_mode: {self.router_weight_mode}")

        # Capacity scales with the full batch-flattened pool (T = B_cond * S).
        k = max(1, min(int((T / num_cond_experts) * self.top_k), T))
        # router_weights: (E_cond, k), flat_indices: (E_cond, k), indices in [0, T)
        router_weights, flat_indices = torch.topk(cond_weights, k=k, dim=-1, sorted=False)

        # Gather each expert's selected tokens: advanced indexing produces (E_cond, k, D).
        expert_inputs = flat_input[flat_indices]

        return flat_indices, router_weights, expert_inputs

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor, t_emb: torch.Tensor):
        identity = hidden_states
        B, S, D = hidden_states.shape
        final_output = torch.zeros_like(hidden_states)
        loss = None

        cond_batch_mask = (labels.view(-1) != 1000) if self.use_uncond_expert else torch.ones(B, dtype=torch.bool, device=hidden_states.device)
        uncond_batch_mask = ~cond_batch_mask

        ### process routed experts
        cond_experts = self.experts[:-1] if self.use_uncond_expert else self.experts
        if cond_batch_mask.any():
            cond_hidden_states = hidden_states[cond_batch_mask]  # (B_cond, S, D)
            cond_t_emb = t_emb[cond_batch_mask]                  # (B_cond, D_t)
            B_cond = cond_hidden_states.shape[0]
            T = B_cond * S

            ### timestep-conditioned prototypes for the cond sub-batch
            prototype_t = self.prototype_mlp(self.cluster_centers, cond_t_emb)  # (B_cond, E, D)

            ### token assignment (batch-flatten, per-sample prototype)
            flat_indices, gating_scores, expert_inputs = self.compute_router(cond_hidden_states, prototype_t)
            # flat_indices: (E_cond, k), gating_scores: (E_cond, k), expert_inputs: (E_cond, k, D)

            num_cond_experts = len(cond_experts)
            # expert_outputs: (E_cond, k, D)
            expert_outputs = torch.stack(
                [cond_experts[e](expert_inputs[e]) for e in range(num_cond_experts)], dim=0
            )

            # Weight each expert's output by its gating score, then scatter-add back to
            # the flat token grid. index_add_ naturally sums contributions when a token
            # is picked by multiple experts — same semantics as the dispatch_mask einsum
            # in the per-image version, just memory-friendly for batch-flatten capacity.
            weighted = expert_outputs * gating_scores.unsqueeze(-1)  # (E_cond, k, D)
            flat_cond_output = torch.zeros(
                T, D, device=hidden_states.device, dtype=expert_outputs.dtype
            )
            flat_cond_output.index_add_(
                0, flat_indices.reshape(-1), weighted.reshape(-1, D)
            )
            cond_output = flat_cond_output.reshape(B_cond, S, D).to(hidden_states.dtype)
            final_output[cond_batch_mask] = cond_output

            ### routing contrastive loss
            if self.training and self.routing_contrastive_lam > 0 and num_cond_experts > 1:
                # expert_inputs: (E_cond, k, D) -> per-expert mean over its k tokens -> (E_cond, D)
                expert_token_means = expert_inputs.mean(dim=1)
                routing_contrastive_loss = self.compute_routing_contrastive_loss(expert_token_means) * self.routing_contrastive_lam
                if loss is not None:
                    loss += routing_contrastive_loss
                else:
                    loss = routing_contrastive_loss
        else:
            dummy_input = torch.zeros(1, 1, D, device=hidden_states.device, dtype=hidden_states.dtype)
            for expert in cond_experts:
                final_output = final_output + expert(dummy_input).sum() * 0
        
        ### process unconditional experts
        if self.use_uncond_expert:
            if uncond_batch_mask.any():
                uncond_hidden_states = hidden_states[uncond_batch_mask]
                uncond_output = self.experts[-1](uncond_hidden_states)
                final_output[uncond_batch_mask] = uncond_output.to(final_output.dtype)
            else:
                dummy_input = torch.zeros(1, 1, D, device=hidden_states.device, dtype=hidden_states.dtype)
                final_output = final_output + self.experts[-1](dummy_input).sum() * 0

        ### process shared experts
        if self.use_shared_expert:
            shared_output = self.shared_expert(identity).to(hidden_states.dtype)
            final_output += shared_output

        return final_output, loss

    def compute_routing_contrastive_loss(self, expert_token_means):
        """
        Batch-Choice variant: each expert has a single mean across all of its
        selected tokens (flattened across the batch), so there is no batch
        dimension on the similarity matrix anymore.

        Args:
            expert_token_means (torch.Tensor): (num_cond_experts, hidden_size)
        """
        num_cond_experts, D = expert_token_means.shape

        if num_cond_experts < 2:
            return torch.tensor(0.0, device=expert_token_means.device)

        cluster_centers = self.cluster_centers  # (E, D)

        centers_norm = F.normalize(cluster_centers, p=2, dim=1)       # (E, D)
        means_norm = F.normalize(expert_token_means, p=2, dim=1)      # (E, D)

        # sim_matrix[i, j] = sim(center[i], mean[j])
        sim_matrix = centers_norm @ means_norm.T  # (E, E)

        logits = sim_matrix / self.routing_contrastive_temperature

        labels = torch.arange(num_cond_experts, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return loss
    
    def _init_weights(self):
        nn.init.normal_(self.cluster_centers, mean=0.0, std=0.02)

    def reset_expert_tracking(self):
        self.last_expert_choices = None
        self.last_expert_choices_layer = self.layer_index

#################################################################################
#                              Core ProMoE_EC_BC Model                           #
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
            if use_swiglu==False:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = SparseMoeBlock(hidden_size=hidden_size, **MoE_config)
            else:
                self.mlp = SparseMoeBlock(hidden_size=hidden_size, **MoE_config)
        else:
            if use_swiglu==False:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
            else:
                self.mlp = MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim, )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, label, t_emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_moe:
            x_mlp, aux_loss = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), label, t_emb)
            if aux_loss is not None:
                x_mlp = AddAuxiliaryLoss.apply(x_mlp, aux_loss)
            x = x + gate_mlp.unsqueeze(1) * x_mlp
            return x
        else:
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
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
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.MoE_config = MoE_config
        use_moe_flag = [True] * depth
        if self.MoE_config.interleave:
            use_moe_flag = [i%2==1 for i in range(depth)]
        print(use_moe_flag)

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob, return_labels=True)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads // (1 if use_moe_flag[i] else 1), head_dim=head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm, 
                     use_swiglu=use_swiglu, MoE_config=MoE_config, use_moe=use_moe_flag[i]) for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.init_MoeMLP= MoE_config.init_MoeMLP
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

        # new init 
        def init_MoeMLP(module, std=0.006):
            nn.init.normal_(module.gate_proj.weight, std=std)
            nn.init.normal_(module.up_proj.weight, std=std)
            nn.init.normal_(module.down_proj.weight, std=std)
        if self.init_MoeMLP:
            for block in self.blocks:
                for expert in block.mlp.experts:
                    init_MoeMLP(expert)
            print("init MoE related module with std 0.006 like DeepSeek-MoE")

        # Re-assert PrototypeMLP init AFTER the blanket self.apply(_basic_init) above.
        # _basic_init xavier-overwrites every nn.Linear, which would otherwise clobber
        # each PrototypeMLP's zero-init fc2 (and identity-init proto_proj in "direct"
        # mode) and break the step-0 == base-ProMoE prototype invariant for BOTH update
        # modes (residual: prototype_t == cluster_centers; direct: proto_proj(cc) == cc
        # with delta == 0). Both modes keep a cc-skip, so this only restores the intended
        # init; it never zeroes the whole prototype.
        for block in self.blocks:
            if getattr(block, "use_moe", False):
                block.mlp.prototype_mlp._init_weights()

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
            x = block(x, c, labels, t)                   # (N, T, D); t plumbed for prototype conditioning
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x 

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        model_out = model_out[0]
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
