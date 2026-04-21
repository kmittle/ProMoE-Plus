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
#                     RouterTransformerBlock for Attention                      #
#################################################################################
class RouterTransformerBlock(nn.Module):
    """
    A simple pre-norm transformer block for computing attention maps.
    Uses bidirectional self-attention.
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
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


#################################################################################
#                      ExpertLocalAttention Module                             #
#################################################################################
class ExpertLocalAttention(nn.Module):
    """
    2-layer transformer + expert-local masked QK attention map.
    Attention is computed only between tokens assigned to the same conditional
    expert within the same image. Non-group positions are masked to -inf before
    softmax, so attention weights are strictly local to each expert group.
    """
    def __init__(self, hidden_size, num_heads=8, num_blocks=2, mlp_ratio=4.0, qk_norm=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            RouterTransformerBlock(hidden_size, num_heads, mlp_ratio, qk_norm)
            for _ in range(num_blocks)
        ])
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size ** -0.5

    def forward(self, x, expert_local_mask):
        """
        Args:
            x: (N, T, D) block output features
            expert_local_mask: (N, T, T) bool, True where two tokens belong to
                the same image and same conditional expert
        Returns:
            attn_map: (N, T, T) expert-local softmax-normalized attention weights
                (positions not in the same group are 0)
        """
        for block in self.blocks:
            x = block(x)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (N, T, T)
        # Mask non-group positions to -inf before softmax
        attn_logits = attn_logits.masked_fill(~expert_local_mask, float('-inf'))
        attn_map = F.softmax(attn_logits, dim=-1)  # (N, T, T)
        # Handle rows that are all -inf (unconditional tokens not in any group):
        # softmax produces nan, replace with 0 so they don't contribute to loss
        attn_map = attn_map.nan_to_num(0.0)
        return attn_map


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

        ### load balancing loss (not used in ProMoE)
        if self.training and self.alpha > 0.0:
            cond_batch_size = (labels != 1000).sum()
            if self.router_weight_mode != "softmax":
                scores_for_aux = F.softmax(cond_weights, dim=1)
            else:
                scores_for_aux = cond_weights
            aux_topk = self.top_k
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

            cond_token_embeddings = flat_input[cond_mask]

            if self.use_top_k_for_routing_contrastive:
                topk_expert_indices = expert_indices.view(batch_size * seq_len, self.top_k)[cond_mask]
                cond_cluster_assignments = topk_expert_indices
            else:
                top1_expert_indices = expert_indices.view(batch_size * seq_len, self.top_k)[:, 0]
                cond_cluster_assignments = top1_expert_indices[cond_mask]

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

        cluster_means = torch.cat(cluster_means, dim=0)
        valid_centers = cluster_centers[valid_clusters]

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
    Diffusion model with a Transformer backbone + cross-alignment via ExpertLocalAttention.
    Attention is masked to only consider tokens within the same expert group.
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
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads // (1 if use_moe_flag[i] else 1), head_dim=head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm,
                     use_swiglu=use_swiglu, MoE_config=MoE_config, use_moe=use_moe_flag[i]) for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.init_MoeMLP= MoE_config.init_MoeMLP

        # REPA: projector heads for representation alignment
        if repa_config is not None:
            self.encoder_depth = repa_config.get('encoder_depth', 4)
            assert self.encoder_depth <= depth, \
                f"repa_config.encoder_depth ({self.encoder_depth}) must be <= model depth ({depth})"
            assert use_moe_flag[self.encoder_depth - 1], \
                f"repa_config.encoder_depth ({self.encoder_depth}) must point to a MoE block " \
                f"(0-indexed block {self.encoder_depth - 1} has use_moe={use_moe_flag[self.encoder_depth - 1]})"
            z_dims = repa_config.get('z_dims', [768])
            projector_dim = repa_config.get('projector_dim', 2048)
            self.projectors = nn.ModuleList([
                build_repa_projector(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])

            # ExpertLocalAttention for cross-alignment (masked to same-expert tokens)
            self.expert_local_attn = ExpertLocalAttention(
                hidden_size=hidden_size,
                num_heads=repa_config.get('align_attn_num_heads', 8),
                num_blocks=2,   # single block alignment → 2 layers
                mlp_ratio=repa_config.get('align_attn_mlp_ratio', 4.0),
                qk_norm=qk_norm,
            )
        else:
            self.encoder_depth = None
            self.projectors = None
            self.expert_local_attn = None

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        def init_MoeMLP(module, std=0.006):
            nn.init.normal_(module.gate_proj.weight, std=std)
            nn.init.normal_(module.up_proj.weight, std=std)
            nn.init.normal_(module.down_proj.weight, std=std)
        if self.init_MoeMLP:
            for block in self.blocks:
                for expert in block.mlp.experts:
                    init_MoeMLP(expert)
            print("init MoE related module with std 0.006 like DeepSeek-MoE")

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _build_expert_local_mask(self, expert_indices, labels, N, T):
        """
        Build expert-local mask: True only for token pairs that belong to the
        same image AND the same conditional expert.

        Each sample in the batch is one image, so the (N, T, T) tensor naturally
        separates different images — no cross-image token pairs exist.

        Args:
            expert_indices: (N, T, top_k)
            labels: (N,)
            N, T: batch size, num tokens
        Returns:
            mask: (N, T, T) bool
        """
        top1_experts = expert_indices[:, :, 0]  # (N, T)
        # Same expert: (N, T, T)
        expert_match = (top1_experts.unsqueeze(2) == top1_experts.unsqueeze(1))

        # Conditional mask: exclude all tokens from unconditional samples
        cond_mask = (labels != 1000)  # (N,)
        token_cond = cond_mask.unsqueeze(1).expand(-1, T)  # (N, T)
        pair_cond = token_cond.unsqueeze(2) & token_cond.unsqueeze(1)  # (N, T, T)

        return expert_match & pair_cond

    def compute_cross_align_loss(self, z_proj, teacher_z, expert_indices, labels, cross_weights):
        """
        Cross-alignment loss: weighted negative cosine similarity.
        The expert_mask here is redundant with the mask already applied in
        ExpertLocalAttention, but retained for interface consistency with
        experiments 1-2.

        Args:
            z_proj: (N, T, D_z) projected student features
            teacher_z: (N, T, D_z) teacher features
            expert_indices: (N, T, top_k) expert assignments
            labels: (N,) class labels
            cross_weights: (N, T, T) cross-alignment weights from attention map
        Returns:
            loss: scalar
        """
        N, T, D_z = z_proj.shape

        z_proj_norm = F.normalize(z_proj, dim=-1)
        teacher_norm = F.normalize(teacher_z, dim=-1)

        # Clamp to [-1, 1] to enforce the mathematical range of cosine similarity;
        # under bf16 autocast, F.normalize + bmm can slip outside this range due
        # to rsqrt/matmul precision, which previously triggered loss spikes.
        cos_sim_raw = torch.bmm(z_proj_norm, teacher_norm.transpose(1, 2))
        cos_sim = cos_sim_raw.clamp(-1.0, 1.0)

        top1_experts = expert_indices[:, :, 0]
        expert_mask = (top1_experts.unsqueeze(2) == top1_experts.unsqueeze(1))

        cond_mask = (labels != 1000)
        token_cond = cond_mask.unsqueeze(1).expand(-1, T)
        pair_cond = token_cond.unsqueeze(2) & token_cond.unsqueeze(1)

        W = cross_weights * expert_mask.float() * pair_cond.float()

        row_sum = W.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        W = W / row_sum

        num_cond_tokens = token_cond.sum()
        loss = -(W * cos_sim).sum() / num_cond_tokens.clamp(min=1)

        with torch.no_grad():
            _r = cos_sim_raw.detach()
            self._cross_align_stats = {
                'cos_sim_absmax': float(_r.abs().max().item()),
                'cos_sim_clamp_count': int((_r.abs() > 1.0).sum().item()),
                'cos_sim_numel': _r.numel(),
            }

        return loss

    def forward(self, x, timestep, context, teacher_all_z=None, **kwargs):
        """
        Forward pass of DiT with cross-alignment via ExpertLocalAttention.

        Args:
            x: (N, C, H, W) spatial inputs
            timestep: (N,) diffusion timesteps
            context: (N,) class labels
            teacher_all_z: (num_teacher_blocks, N, T, D_z) all teacher block features

        Returns (training):
            x: (N, out_channels, H, W) model prediction
            cross_align_loss: scalar cross-alignment loss
        Returns (inference):
            x: (N, out_channels, H, W) model prediction
        """
        y = context
        if len(x.shape) != 4:
            x = x.squeeze(2)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        N, T, D = x.shape
        t = self.t_embedder(timestep)
        y, labels = self.y_embedder(y, self.training)
        c = t + y

        # Pass through DiT blocks
        cross_align_loss = torch.tensor(0.0, device=x.device)
        for i, block in enumerate(self.blocks):
            x = block(x, c, labels)
            # Compute cross-alignment loss at encoder_depth
            if self.training and self.projectors is not None and (i + 1) == self.encoder_depth:
                if teacher_all_z is not None:
                    expert_indices = block.mlp._expert_indices  # (N, T, top_k)

                    # Build expert-local mask: (N, T, T)
                    expert_local_mask = self._build_expert_local_mask(
                        expert_indices, labels, N, T
                    )

                    # Compute expert-local attention map
                    # Detach x so the attention path does not leak gradient
                    # back into block-(encoder_depth); block-(encoder_depth)
                    # only receives REPA gradient via the projection path,
                    # matching the plan 01 (global_pre) decoupling that
                    # trains stably. See crash_diagnosis_report.md.
                    cross_attn_map = self.expert_local_attn(x.detach(), expert_local_mask)

                    teacher_z = teacher_all_z[-1]  # last teacher block
                    z_proj = self.projectors[0](x.reshape(-1, D)).reshape(N, T, -1)
                    cross_align_loss = self.compute_cross_align_loss(
                        z_proj, teacher_z, expert_indices, labels, cross_attn_map
                    )

        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        if not self.training:
            return x
        return x, cross_align_loss

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
