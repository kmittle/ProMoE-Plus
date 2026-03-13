import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
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
#                      K-Means and Hungarian Matching                          #
#################################################################################
def _sq_dists(a, b):
    """Squared L2 distances between rows of a and rows of b via matmul.

    Args:
        a: (N, D), b: (K, D)
    Returns:
        (N, K) squared distances
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a@b^T
    a_sq = (a * a).sum(dim=1, keepdim=True)  # [N, 1]
    b_sq = (b * b).sum(dim=1, keepdim=True)  # [K, 1]
    return a_sq + b_sq.T - 2.0 * (a @ b.T)  # [N, K]


@torch.no_grad()
def kmeans_plus_plus_init(data, n_clusters):
    """K-Means++ initialization on GPU.

    Incrementally maintains min-distance-squared to avoid redundant
    distance computations.

    Args:
        data: (N, D) tensor
        n_clusters: number of clusters

    Returns:
        centers: (n_clusters, D) tensor
    """
    n_samples = data.shape[0]
    device = data.device

    # First center: random
    idx = torch.randint(0, n_samples, (1,), device=device).item()
    centers = [data[idx : idx + 1]]  # list of [1, D]

    # min squared distance from each point to the nearest chosen center
    min_sq_dists = torch.full((n_samples,), float('inf'), device=device)

    for _ in range(1, n_clusters):
        # Update min distances with the latest center only
        new_center = centers[-1]  # [1, D]
        sq_dist_to_new = _sq_dists(data, new_center).squeeze(1)  # [N]
        min_sq_dists = torch.minimum(min_sq_dists, sq_dist_to_new)

        # Sample proportional to min squared distance
        probs = min_sq_dists / min_sq_dists.sum()
        idx = torch.multinomial(probs, 1).item()
        centers.append(data[idx : idx + 1])

    return torch.cat(centers, dim=0)  # [n_clusters, D]


@torch.no_grad()
def kmeans(data, n_clusters, n_iters=10):
    """K-Means clustering with K-Means++ initialization.

    Uses vectorized scatter_add_ for center updates and matmul-based
    squared L2 distances for assignment.

    Args:
        data: (N, D) tensor
        n_clusters: number of clusters
        n_iters: number of iterations

    Returns:
        centers: (n_clusters, D) tensor of cluster centers
    """
    N, D = data.shape
    centers = kmeans_plus_plus_init(data, n_clusters)

    for _ in range(n_iters):
        # Assign each point to nearest center (squared L2)
        sq_dists = _sq_dists(data, centers)  # [N, K]
        assignments = sq_dists.argmin(dim=1)  # [N]

        # Vectorized center update via scatter_add_
        new_centers = torch.zeros_like(centers)  # [K, D]
        counts = torch.zeros(n_clusters, 1, device=data.device, dtype=data.dtype)
        expand_idx = assignments.unsqueeze(1).expand(-1, D)  # [N, D]
        new_centers.scatter_add_(0, expand_idx, data)
        counts.scatter_add_(0, assignments.unsqueeze(1),
                            torch.ones(N, 1, device=data.device, dtype=data.dtype))

        # Handle empty clusters: keep old center
        empty_mask = (counts.squeeze(1) == 0)
        counts = counts.clamp(min=1)
        new_centers = new_centers / counts
        if empty_mask.any():
            new_centers[empty_mask] = centers[empty_mask]
        centers = new_centers

    return centers  # [K, D]


def hungarian_match(cost_matrix):
    """Optimal assignment minimizing total cost using the Hungarian algorithm.

    Args:
        cost_matrix: (n, n) tensor, where cost_matrix[i, j] is the cost of
                     assigning prototype i to teacher cluster j.

    Returns:
        row_ind, col_ind: matched indices (as torch tensors on the same device)
    """
    cost_np = cost_matrix.detach().cpu().float().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    device = cost_matrix.device
    return torch.tensor(row_ind, device=device), torch.tensor(col_ind, device=device)


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
    Diffusion model with a Transformer backbone + REPA projectors
    + REPA-guided router alignment.
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

        # Collect MoE block indices for router alignment
        self.moe_block_indices = [i for i, flag in enumerate(use_moe_flag) if flag]
        self.num_moe_blocks = len(self.moe_block_indices)

        # REPA: projector heads for representation alignment (naive REPA)
        if repa_config is not None:
            self.encoder_depth = repa_config.get('encoder_depth', 4)
            assert self.encoder_depth <= depth, \
                f"repa_config.encoder_depth ({self.encoder_depth}) must be <= model depth ({depth})"
            z_dims = repa_config.get('z_dims', [768])
            projector_dim = repa_config.get('projector_dim', 2048)
            self.projectors = nn.ModuleList([
                build_repa_projector(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])

            # Router REPA: projectors that map teacher cluster centers → prototype space
            teacher_dim = z_dims[0]  # teacher embedding dimension
            router_projector_dim = repa_config.get('router_projector_dim', projector_dim)
            self.router_projectors = nn.ModuleList([
                build_repa_projector(teacher_dim, router_projector_dim, hidden_size)
                for _ in range(self.num_moe_blocks)
            ])
            self.router_repa_coeff = repa_config.get('router_repa_coeff', 1)
            self.kmeans_n_iters = repa_config.get('kmeans_n_iters', 10)
        else:
            self.encoder_depth = None
            self.projectors = None
            self.router_projectors = None
            self.router_repa_coeff = 0.0
            self.kmeans_n_iters = 10

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

    def compute_router_repa_loss(self, teacher_z):
        """Compute REPA-guided router alignment loss across all MoE blocks.

        1. Flatten teacher_z and run K-Means to get n cluster centers in teacher space.
        2. For each MoE block, project teacher centers → prototype space via router_projector.
        3. Hungarian match projected centers with block's cluster_centers (prototypes).
        4. Compute negative cosine similarity loss on matched pairs.

        Args:
            teacher_z: (B, T, D_teacher) frozen teacher encoder features.

        Returns:
            router_align_loss: scalar tensor (averaged across MoE blocks).
        """
        n_clusters = self.MoE_config.num_routed_experts
        device = teacher_z.device

        # Flatten to [B*T, D_teacher] for K-Means
        flat_teacher = teacher_z.reshape(-1, teacher_z.shape[-1]).float()
        teacher_centers = kmeans(flat_teacher, n_clusters, n_iters=self.kmeans_n_iters)
        # teacher_centers: [n_clusters, D_teacher]

        total_loss = torch.tensor(0.0, device=device)
        moe_idx = 0
        for block_idx in self.moe_block_indices:
            block = self.blocks[block_idx]
            prototypes = block.mlp.cluster_centers  # [n_clusters, hidden_size]

            # Project teacher centers to prototype space
            proj_centers = self.router_projectors[moe_idx](teacher_centers)  # [n_clusters, hidden_size]

            # Compute cosine similarity matrix for Hungarian matching
            proto_norm = F.normalize(prototypes, p=2, dim=1)      # [n, hidden_size]
            proj_norm = F.normalize(proj_centers, p=2, dim=1)     # [n, hidden_size]
            cos_sim = proto_norm @ proj_norm.T  # [n, n]

            # Hungarian matching: minimize negative cosine similarity (maximize similarity)
            cost = -cos_sim  # [n, n]
            row_ind, col_ind = hungarian_match(cost)

            # Compute negative cosine similarity loss on matched pairs
            matched_proto = proto_norm[row_ind]    # [n, hidden_size]
            matched_proj = proj_norm[col_ind]      # [n, hidden_size]
            pair_loss = -(matched_proto * matched_proj).sum(dim=-1).mean()

            total_loss = total_loss + pair_loss
            moe_idx += 1

        # Average across MoE blocks
        router_align_loss = total_loss / self.num_moe_blocks
        return router_align_loss

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
        Forward pass of DiT with naive REPA + REPA-guided router alignment.

        Args:
            x: (N, C, H, W) tensor of spatial inputs
            timestep: (N,) tensor of diffusion timesteps
            context: (N,) tensor of class labels
            teacher_z: (N, T, D_teacher) optional teacher encoder features for
                       router alignment (passed via kwargs)

        Returns (training):
            x: (N, out_channels, H, W) model prediction
            zs_proj: list of (N, T, z_dim) projected features for naive REPA
        Returns (inference):
            x: (N, out_channels, H, W) model prediction
        """
        teacher_z = kwargs.get('teacher_z', None)

        y = context
        if len(x.shape) != 4:
            x = x.squeeze(2)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape
        t = self.t_embedder(timestep)                   # (N, D)
        y, labels = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)

        zs_proj = None
        for i, block in enumerate(self.blocks):
            x = block(x, c, labels)                      # (N, T, D)
            # Extract projected features at encoder_depth for naive REPA alignment (training only)
            if self.training and self.projectors is not None and (i + 1) == self.encoder_depth:
                zs_proj = [proj(x.reshape(-1, D)).reshape(N, T, -1) for proj in self.projectors]

        # REPA-guided router alignment loss (training only)
        if self.training and self.router_projectors is not None and teacher_z is not None:
            router_align_loss = self.compute_router_repa_loss(teacher_z)
            router_align_loss = router_align_loss * self.router_repa_coeff
            x = AddAuxiliaryLoss.apply(x, router_align_loss)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        if not self.training:
            return x
        return x, zs_proj

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
