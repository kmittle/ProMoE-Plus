import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as F
from .modules import get_2d_sincos_pos_embed, Attention, modulate, TimestepEmbedder, LabelEmbedder, FinalLayer, MoeMLP, Mlp


#################################################################################
#                             Projector for REPA                               #
#################################################################################
def build_repa_projector(hidden_size, projector_dim, z_dim):
    """3-layer MLP projector following REPA (SiT) design."""
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


#################################################################################
#                     Transformer-based Block Router (MoS)                     #
#################################################################################
class RouterTransformerBlock(nn.Module):
    """
    A simple pre-norm transformer block used by BlockRouter.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

    def forward(self, x):
        # Pre-norm style: norm -> attn/mlp -> residual
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BlockRouter(nn.Module):
    """
    Transformer-based block router for MoS REPA (ref: arxiv 2511.12207).

    Takes the patchified embedding and conditioning as input, and produces
    per-token routing weights that determine which teacher encoder block
    each DiT block should align with.

    Inputs:
        x_embed: (N, T, D) patchified noised latent + positional embedding
        c: (N, D) conditioning embedding (timestep + class)

    Outputs:
        routing_weights: (N, T, m, n) softmax-normalized weights
            m = num_teacher_blocks (understanding tower depth)
            n = depth (generation tower / DiT depth)
            softmax is applied over m for each generation block n
    """
    def __init__(
        self,
        hidden_size,
        cond_size,
        num_teacher_blocks,
        depth,
        router_hidden_dim=None,
        num_router_blocks=2,
        num_heads=8,
        mlp_ratio=4.0,
        qk_norm=False,
    ):
        super().__init__()
        self.num_teacher_blocks = num_teacher_blocks
        self.depth = depth
        if router_hidden_dim is None:
            router_hidden_dim = hidden_size
        self.router_hidden_dim = router_hidden_dim

        # Input projections
        self.x_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.x_proj = nn.Linear(hidden_size, router_hidden_dim, bias=False)

        # Conditioning projection: timestep + class -> extra token
        self.c_norm = nn.LayerNorm(cond_size, eps=1e-6)
        self.c_proj = nn.Linear(cond_size, router_hidden_dim, bias=False)

        # Router transformer blocks (2 blocks, bidirectional self-attention)
        self.blocks = nn.ModuleList([
            RouterTransformerBlock(
                router_hidden_dim, num_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm
            )
            for _ in range(num_router_blocks)
        ])

        # Output head: each patch token predicts an (m x n) routing matrix
        self.routing_head = nn.Linear(
            router_hidden_dim, num_teacher_blocks * depth, bias=False
        )

    def forward(self, x_embed, c):
        """
        Args:
            x_embed: (N, T, D) patchified embedding
            c: (N, D) conditioning (timestep + class embedding)

        Returns:
            routing_weights: (N, T, m, n) with softmax over m (teacher blocks)
        """
        N, T, _ = x_embed.shape

        # Project inputs to router hidden dim
        x_proj = self.x_proj(self.x_norm(x_embed))     # (N, T, router_dim)
        c_proj = self.c_proj(self.c_norm(c)).unsqueeze(1)  # (N, 1, router_dim)

        # Concatenate conditioning token with patch tokens
        h = torch.cat([c_proj, x_proj], dim=1)  # (N, T+1, router_dim)

        # Process through router transformer blocks
        for block in self.blocks:
            h = block(h)

        # Extract patch token outputs (skip the conditioning token)
        h = h[:, 1:, :]  # (N, T, router_dim)

        # Project to routing logits and reshape
        logits = self.routing_head(h)  # (N, T, m * n)
        logits = logits.reshape(N, T, self.num_teacher_blocks, self.depth)

        # Softmax over teacher blocks (m) for each generation block (n)
        routing_weights = F.softmax(logits, dim=2)  # (N, T, m, n)

        return routing_weights


#################################################################################
#                                ProMoE Layer                                  #
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
        top_k=2,
        load_balance_loss_coef=0,
        norm_topk_prob=False,
        seq_aux=False,
        use_shared_expert=True,
        use_uncond_expert=True,
        router_weight_mode="softmax",
        routing_contrastive_lam=0,
        use_top_k_for_routing_contrastive=False,
        routing_contrastive_temperature=0.1,
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

        self._expert_indices = None
        self._proto_sim = None

        self._init_weights()

    def compute_router(self, hidden_states, labels):
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        flat_input = hidden_states.view(-1, self.hidden_size)
        flat_labels = labels.view(batch_size, 1).expand(-1, seq_len).reshape(-1)

        if self.use_uncond_expert and flat_labels is not None:
            uncond_mask = (flat_labels == 1000)
            cond_mask = ~uncond_mask
        else:
            uncond_mask = None
            cond_mask = torch.ones_like(flat_labels, dtype=torch.bool)

        router_weights = torch.zeros(
            batch_size * seq_len, self.top_k, device=device,
        )
        expert_indices = torch.zeros(
            batch_size * seq_len, self.top_k, device=device, dtype=torch.long
        )

        if uncond_mask is not None and uncond_mask.any():
            uncond_positions = torch.where(uncond_mask)[0]
            router_weights[uncond_positions, 0] = 1.0
            expert_indices[uncond_positions] = self.num_experts - 1

        if cond_mask.any():
            cond_positions = torch.where(cond_mask)[0]
            cond_input = flat_input[cond_positions]

            input_norm = F.normalize(cond_input, p=2, dim=1)
            cluster_norm = F.normalize(self.cluster_centers, p=2, dim=1)

            cos_sim = input_norm @ cluster_norm.T

            if self.router_weight_mode == "softmax":
                cond_weights = F.softmax(cos_sim, dim=1)
            elif self.router_weight_mode == "sigmoid":
                sigmoid_scale = 1.0
                cond_weights = torch.sigmoid(cos_sim * sigmoid_scale)
            elif self.router_weight_mode == "identity":
                cond_weights = cos_sim
            else:
                raise ValueError(f"Unsupported router_weight_mode: {self.router_weight_mode}")

            topk_scores, topk_idx = torch.topk(cond_weights, k=self.top_k, dim=1)

            router_weights[cond_positions] = topk_scores.to(router_weights.dtype)
            expert_indices[cond_positions] = topk_idx

        router_weights = router_weights.view(batch_size, seq_len, self.top_k)
        expert_indices = expert_indices.view(batch_size, seq_len, self.top_k)

        # Cache expert indices for cross-alignment
        self._expert_indices = expert_indices

        # Cache each token's raw cosine similarity with its top-1 prototype
        proto_sim = torch.zeros(batch_size * seq_len, device=device)
        if cond_mask.any():
            top1_idx = topk_idx[:, 0]  # (num_cond,)
            top1_sim = cos_sim[torch.arange(len(top1_idx), device=device), top1_idx]  # (num_cond,)
            proto_sim[cond_positions] = top1_sim.to(proto_sim.dtype)
        self._proto_sim = proto_sim.view(batch_size, seq_len)  # (N, T)

        ### load balancing loss (not used in ProMoE)
        if self.training and self.alpha > 0.0:
            cond_batch_size = (labels != 1000).sum()
            if self.router_weight_mode != "softmax":
                scores_for_aux = F.softmax(cond_weights, dim=1)
            else:
                scores_for_aux = cond_weights
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(cond_batch_size, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(cond_batch_size, seq_len, -1)
                ce = torch.zeros(cond_batch_size, self.num_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(cond_batch_size, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.num_routed_experts)
                load_balance_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.num_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.num_routed_experts
                load_balance_loss = (Pi * fi).sum() * self.alpha
        else:
            load_balance_loss = None

        return router_weights, expert_indices, load_balance_loss

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor):
        ### token assignment
        router_weights, expert_indices, load_balance_loss = self.compute_router(hidden_states, labels)
        batch_size, seq_len, hidden_dim = hidden_states.shape

        flat_input = hidden_states.view(-1, hidden_dim)
        flat_weights = router_weights.view(-1, self.top_k)
        flat_indices = expert_indices.view(-1, self.top_k)
        total_tokens = batch_size * seq_len

        final_output = torch.zeros(total_tokens, hidden_dim, device=hidden_states.device)

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

        return final_output, loss

    def compute_routing_contrastive_loss(self, token_embeddings, cluster_assignments, use_top_k=False):
        """
        cluster_centers: [num_clusters, hidden_size]
        token_embeddings: [num_tokens, hidden_size]
        cluster_assignments:
            - use_top_k=False: [num_tokens]
            - use_top_k=True: [num_tokens, top_k]
        """
        cluster_centers = self.cluster_centers
        num_clusters = cluster_centers.size(0)
        device = cluster_centers.device

        cluster_means = []
        valid_clusters = []

        for cluster_id in range(num_clusters):
            if use_top_k:
                mask = (cluster_assignments == cluster_id).any(dim=1)
            else:
                mask = (cluster_assignments == cluster_id)

            if mask.sum() > 0:
                cluster_mean = token_embeddings[mask].mean(dim=0, keepdim=True)
                cluster_means.append(cluster_mean)
                valid_clusters.append(cluster_id)

        if len(valid_clusters) < 2:
            return torch.tensor(0.0, device=device)

        cluster_means = torch.cat(cluster_means, dim=0)  # [num_valid_clusters, hidden_size]
        valid_centers = cluster_centers[valid_clusters]  # [num_valid_clusters, hidden_size]

        centers_norm = F.normalize(valid_centers, p=2, dim=1)
        means_norm = F.normalize(cluster_means, p=2, dim=1)

        sim_matrix = centers_norm @ means_norm.T

        temperature = self.routing_contrastive_temperature
        labels = torch.arange(sim_matrix.size(0), device=device)
        logits = sim_matrix / temperature

        loss = F.cross_entropy(logits, labels)

        return loss

    def _init_weights(self):
        nn.init.normal_(self.cluster_centers, mean=0.0, std=0.02)


#################################################################################
#                                 Core ProMoE Model                            #
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

    def forward(self, x, c, label):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_moe:
            x_mlp, aux_loss = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), label)
            if aux_loss is not None:
                x_mlp = AddAuxiliaryLoss.apply(x_mlp, aux_loss)
            x = x + gate_mlp.unsqueeze(1) * x_mlp
            return x
        else:
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone + MoS REPA + Prototype similarity cross-alignment.
    Combines BlockRouter (selects teacher block depth) with prototype cosine similarity
    cross-alignment weights (no new learnable modules for cross-alignment).
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
        repa_config=None,
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

        # MoS REPA + prototype similarity cross-alignment (no new learnable modules)
        if repa_config is not None:
            num_teacher_blocks = repa_config.get('num_teacher_blocks', 12)
            z_dim = repa_config.get('z_dims', [768])[0]
            projector_dim = repa_config.get('projector_dim', 2048)
            router_hidden_dim = repa_config.get('router_hidden_dim', hidden_size)
            num_router_blocks = repa_config.get('num_router_blocks', 2)
            router_num_heads = repa_config.get('router_num_heads', num_heads)

            self.num_teacher_blocks = num_teacher_blocks
            self.mos_top_k = repa_config.get('mos_top_k', 2)
            self.mos_random_prob = repa_config.get('mos_random_prob', 0.05)

            # Selective block alignment: only align specified DiT blocks
            align_blocks = repa_config.get('align_blocks', None)
            if align_blocks is None:
                align_blocks = list(range(depth))
            self.align_blocks = align_blocks
            self.align_block_to_idx = {b: i for i, b in enumerate(align_blocks)}
            num_align_blocks = len(align_blocks)
            print(f"MoS REPA align_blocks ({num_align_blocks}/{depth}): {align_blocks}")

            # Global transformer-based block router
            self.block_router = BlockRouter(
                hidden_size=hidden_size,
                cond_size=hidden_size,
                num_teacher_blocks=num_teacher_blocks,
                depth=num_align_blocks,
                router_hidden_dim=router_hidden_dim,
                num_router_blocks=num_router_blocks,
                num_heads=router_num_heads,
                qk_norm=qk_norm,
            )

            # Projectors only for selected blocks
            self.mos_projectors = nn.ModuleList([
                build_repa_projector(hidden_size, projector_dim, z_dim) for _ in range(num_align_blocks)
            ])

            # No additional learnable modules for prototype cross-alignment
        else:
            self.num_teacher_blocks = None
            self.mos_top_k = 2
            self.mos_random_prob = 0.05
            self.align_blocks = []
            self.align_block_to_idx = {}
            self.block_router = None
            self.mos_projectors = None

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
                if block.use_moe:
                    for expert in block.mlp.experts:
                        init_MoeMLP(expert)
            print("init MoE related module with std 0.006 like DeepSeek-MoE")

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

    def _build_proto_cross_weights(self, expert_indices, proto_sim, labels, N, T):
        """
        Build cross-alignment weight matrix from prototype cosine similarities.

        Rules:
        - W[n, i, i] = 1 (self-alignment weight fixed to 1 for cond tokens)
        - W[n, i, j] = proto_sim[n, i] * proto_sim[n, j]
          (only when i, j share the same expert and both are conditional)
        - All other entries are 0

        Args:
            expert_indices: (N, T, top_k)
            proto_sim: (N, T) each token's raw cosine similarity with its top-1 prototype
            labels: (N,) class labels
            N, T: batch size, num tokens
        Returns:
            W: (N, T, T)
        """
        # Same-expert mask
        top1_experts = expert_indices[:, :, 0]  # (N, T)
        expert_match = (top1_experts.unsqueeze(2) == top1_experts.unsqueeze(1))  # (N, T, T)

        # Conditional mask
        cond_mask = (labels != 1000).unsqueeze(1).expand(-1, T)  # (N, T)
        pair_cond = cond_mask.unsqueeze(2) & cond_mask.unsqueeze(1)  # (N, T, T)

        # Clamp proto_sim to non-negative before computing cross-weights.
        # Negative proto_sim (token anti-correlated with all prototypes) would
        # produce negative off-diagonal entries, which can make the row_sum
        # negative.  After clamp(min=1e-8) in row normalisation the resulting
        # division by ~0 blows W up to ~1e8, causing the loss to spike.
        # Clamping to 0 makes those tokens fall back to self-alignment only.
        proto_sim = proto_sim.clamp(min=0)

        # Outer product: sim_i * sim_j  (non-negative after clamp)
        outer_sim = proto_sim.unsqueeze(2) * proto_sim.unsqueeze(1)  # (N, T, T)

        # Combine: same expert + conditional + similarity outer product
        W = outer_sim * expert_match.float() * pair_cond.float()  # (N, T, T)

        # Override diagonal to 1 for conditional tokens
        diag_mask = torch.eye(T, device=W.device).unsqueeze(0).expand(N, -1, -1)
        cond_diag = diag_mask * cond_mask.unsqueeze(2).float()  # (N, T, T)
        W = W * (1 - diag_mask) + cond_diag  # off-diagonal: outer product; diagonal: 1 for cond

        return W

    def compute_cross_mos_repa_loss(self, x, align_idx, routing_weights, teacher_all_z,
                                     expert_indices, labels, cross_weights, N, T, D):
        """
        MoS REPA loss with cross-alignment.
        Combines BlockRouter (selects teacher block) with cross-alignment weights
        (determines teacher token positions).

        Args:
            x: (N, T, D) current block output
            align_idx: projector/router column index
            routing_weights: (N, T, m, n_align) BlockRouter output
            teacher_all_z: (m, N, T, D_z) all teacher block features
            expert_indices: (N, T, top_k) routing assignment (None for dense blocks)
            labels: (N,) class labels
            cross_weights: (N, T, T) cross-alignment weight matrix
            N, T, D: batch/token/dim sizes
        Returns:
            block_loss: scalar
        """
        # 1. Project student features
        z_proj = self.mos_projectors[align_idx](x.reshape(-1, D)).reshape(N, T, -1)

        # 2. BlockRouter weights -> select teacher block (same as base MoS)
        block_weights = routing_weights[:, :, :, align_idx]  # (N, T, m)
        m = teacher_all_z.shape[0]
        top_k = min(self.mos_top_k, m)

        if self.training and torch.rand(1).item() < self.mos_random_prob:
            rand_indices = torch.randperm(m, device=x.device)[:top_k]
            select_idx = rand_indices.view(1, 1, top_k).expand(N, T, top_k)
        else:
            _, select_idx = torch.topk(block_weights, k=top_k, dim=-1)

        # 3. Build cross-alignment weight matrix W
        # For prototype cross-alignment, expert_mask is already baked into cross_weights
        # via _build_proto_cross_weights (for MoE blocks) or identity (for dense blocks).
        # We still apply expert_mask for MoE blocks for consistency with experiments 5-7.
        if expert_indices is not None:
            # MoE block: apply expert grouping mask (redundant but consistent)
            top1_experts = expert_indices[:, :, 0]
            expert_mask = (top1_experts.unsqueeze(2) == top1_experts.unsqueeze(1))
            cond_mask = (labels != 1000).unsqueeze(1).expand(-1, T)
            pair_cond = cond_mask.unsqueeze(2) & cond_mask.unsqueeze(1)
            W = cross_weights * expert_mask.float() * pair_cond.float()
        else:
            # Dense block: identity matrix passed as cross_weights
            cond_mask = (labels != 1000).unsqueeze(1).expand(-1, T)
            pair_cond = cond_mask.unsqueeze(2) & cond_mask.unsqueeze(1)
            W = cross_weights * pair_cond.float()

        # 4. Cross-alignment cos_sim for each selected teacher block
        z_proj_norm = F.normalize(z_proj, dim=-1)  # (N, T, D_z)
        total_loss = torch.tensor(0.0, device=x.device)

        for k_idx in range(top_k):
            # BlockRouter weight for the k-th selected teacher block
            selected_block_weight = torch.gather(
                block_weights, dim=-1,
                index=select_idx[:, :, k_idx:k_idx + 1]
            ).squeeze(-1)  # (N, T)

            unique_blocks = select_idx[:, :, k_idx].unique()
            k_loss = torch.tensor(0.0, device=x.device)

            for tb in unique_blocks:
                token_mask = (select_idx[:, :, k_idx] == tb)  # (N, T)
                teacher_z = teacher_all_z[tb]  # (N, T, D_z)
                teacher_norm = F.normalize(teacher_z, dim=-1)

                # Cross cos_sim: (N, T, T)
                # Clamp to [-1, 1] to enforce the mathematical range of cosine similarity;
                # under bf16 autocast, F.normalize + bmm can slip outside this range due
                # to rsqrt/matmul precision, which previously triggered loss spikes.
                cos_sim_matrix = torch.bmm(z_proj_norm, teacher_norm.transpose(1, 2)).clamp(-1.0, 1.0)

                # Weighted cross-alignment similarity per token, normalized by
                # W row sum so cross_sim is a proper weighted average (bounded [-1,1]).
                cross_sim = (W * cos_sim_matrix).sum(dim=-1)  # (N, T)
                cross_sim = cross_sim / W.sum(dim=-1).clamp(min=1)  # (N, T)

                k_loss = k_loss + (-(cross_sim * selected_block_weight) * token_mask.float()).sum()

            total_loss = total_loss + k_loss

        num_cond = (labels != 1000).sum() * T
        block_loss = total_loss / num_cond.clamp(min=1)
        return block_loss

    def forward(self, x, timestep, context, teacher_all_z=None, **kwargs):
        """
        Forward pass of DiT with MoS REPA + prototype similarity cross-alignment.

        Args:
            x: (N, C, H, W) tensor of spatial inputs
            timestep: (N,) tensor of diffusion timesteps
            context: (N,) tensor of class labels
            teacher_all_z: (m, N, T, D_teacher) features from all teacher encoder blocks
                           (only needed during training)

        Returns:
            - inference: x (N, out_channels, H, W)
            - training: (x, mos_repa_loss) where mos_repa_loss is a scalar
        """
        y = context
        if len(x.shape) != 4:
            x = x.squeeze(2)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape
        t = self.t_embedder(timestep)                   # (N, D)
        y, labels = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)

        # Pre-compute routing weights for all blocks via the block router
        routing_weights = None
        if self.training and self.block_router is not None and teacher_all_z is not None:
            routing_weights = self.block_router(x, c)  # (N, T, m, n_align)

        # No pre-block computation — prototype weights built from each MoE block's routing

        mos_repa_loss = torch.tensor(0.0, device=x.device)
        for i, block in enumerate(self.blocks):
            x = block(x, c, labels)                      # (N, T, D)
            # Compute cross MoS REPA loss only for selected blocks
            if self.training and routing_weights is not None and i in self.align_block_to_idx:
                align_idx = self.align_block_to_idx[i]

                if block.use_moe:
                    expert_indices = block.mlp._expert_indices  # (N, T, top_k)
                    proto_sim = block.mlp._proto_sim             # (N, T)
                    # Build prototype similarity weight matrix
                    cross_weights = self._build_proto_cross_weights(
                        expert_indices, proto_sim, labels, N, T
                    )
                else:
                    # Dense block: no routing info, use identity matrix (1-to-1 alignment)
                    expert_indices = None
                    cross_weights = torch.eye(T, device=x.device).unsqueeze(0).expand(N, -1, -1)

                block_loss = self.compute_cross_mos_repa_loss(
                    x, align_idx, routing_weights, teacher_all_z,
                    expert_indices, labels, cross_weights, N, T, D
                )
                mos_repa_loss = mos_repa_loss + block_loss

        # Average over aligned blocks
        if self.training and routing_weights is not None and len(self.align_blocks) > 0:
            mos_repa_loss = mos_repa_loss / len(self.align_blocks)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        if not self.training:
            return x
        return x, mos_repa_loss

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
