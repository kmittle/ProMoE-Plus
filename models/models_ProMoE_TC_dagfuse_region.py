import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as F
from .modules import get_2d_sincos_pos_embed, Attention, modulate, TimestepEmbedder, LabelEmbedder, FinalLayer, MoeMLP, Mlp


# ============================================================================
# ProMoE-TC Shared-Augment (idea 4: dagfuse_region) — region / Block-AttnRes.
# Following "Attention Residuals" (Kimi Team, arXiv 2603.15031): split the blocks
# into fixed-size contiguous REGIONS; a region's representation is its LAST block's
# output; the current position aggregates over PREVIOUS regions' reps, added back
# as a zero-init residual.
#
# region_size = 3 (default). region rep = last block output of the region.
# Two ATTACH modes (config `region_attach`), each with mech `fuse_mech` in {dag,
# softmax} (SharedAugmentModule):
#   (a) region_attach='shared'  — at each region's FIRST MoE block, augment its
#       SHARED-expert output from all previous regions' reps (cond/all via
#       fuse_apply). Routing byte-identical to base (touches only the shared
#       output). Region 0 has no previous region => that block is plain base.
#   (b) region_attach='resid'   — at each region BOUNDARY (before a new region's
#       first block), add a zero-init attn-residual over previous region reps to
#       the MAIN residual stream x (all tokens; cond/all N/A). This module lives at
#       DiT level, not in the MoE block.
#
# NOTE (Codex review, plan v2 §5/§7): mode (b) is NOT shared-expert augmentation —
# it rewrites x. It is step-0 identical (zero-init delta = 0), but AFTER training it
# changes the routing INPUTS of downstream blocks, so "does not touch routing" holds
# only STRUCTURALLY, not behaviorally. It is kept as an explicit ablation arm.
#
# Both mechanisms read the SAME source set (previous region reps + a self-edge inside
# SharedAugmentModule) for a clean dag-vs-softmax comparison. Zero-init output proj
# => step-0 bit-identical to base ProMoE_TC (non-strict loadable). Config
# (MoE_config): region_size (default 3), region_attach {shared, resid}, fuse_apply
# {none, cond, all}, fuse_mech {dag, softmax}, fuse_dim (d_g, default 64).
# ============================================================================


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


class FusedRMSNorm(nn.Module):
    """RMSNorm with an fp32 variance reduction (bf16-safe), à la LlamaRMSNorm."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(in_dtype)


class SharedAugmentModule(nn.Module):
    """Zero-init, QUERY-ONLY augmentation of a target token set from K source token
    sets (plan v2 §1.1). See models_ProMoE_TC_dagfuse_dense for the full contract.
    Only the target is updated; a self-edge is always included; the output
    projection is zero-initialized (step-0 identity)."""
    def __init__(self, hidden_size, dag_dim=64, mech="dag"):
        super().__init__()
        assert mech in ("dag", "softmax"), mech
        self.mech = mech
        self.hidden_size = hidden_size
        self.dag_dim = dag_dim
        self.norm = FusedRMSNorm(hidden_size)
        if mech == "dag":
            self.down_proj = nn.Linear(hidden_size, dag_dim, bias=False)
            self.combined_proj = nn.Linear(dag_dim, 4 * dag_dim, bias=False)
            self.up_proj = nn.Linear(dag_dim, hidden_size, bias=False)  # zero-init
            self.act = nn.GELU(approximate="tanh")
        else:
            self.q_proj = nn.Linear(hidden_size, dag_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, dag_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # zero-init
        self.zero_out()

    def zero_out(self):
        if self.mech == "dag":
            nn.init.zeros_(self.up_proj.weight)
        else:
            nn.init.zeros_(self.out_proj.weight)

    def forward(self, target, sources):
        # target: [N, d]; sources: [N, K, d] (K >= 1). Returns [N, d] = target + zero-init delta.
        if sources is None or sources.shape[1] == 0:
            return target
        keys = torch.cat([target.unsqueeze(1), sources], dim=1)  # [N, 1+K, d]; key 0 = self-edge
        xn = self.norm(keys)
        if self.mech == "dag":
            xd = self.down_proj(xn)
            comb = self.combined_proj(xd)
            g_src, g_tgt, v_src, v_tgt = comb.chunk(4, dim=-1)
            g_query = g_src[:, 0, :]
            v_query = v_src[:, 0, :]
            gate = self.act(g_query.unsqueeze(1) + g_tgt)
            value = v_query.unsqueeze(1) + v_tgt
            agg = (gate * value).sum(dim=1)
            delta = self.up_proj(agg)
        else:
            q = self.q_proj(xn[:, 0, :]).float()
            k = self.k_proj(xn).float()
            v = self.v_proj(xn)
            attn = (q.unsqueeze(1) * k).sum(-1) / (self.dag_dim ** 0.5)
            attn = F.softmax(attn, dim=-1).to(v.dtype)
            ctx = (attn.unsqueeze(-1) * v).sum(dim=1)
            delta = self.out_proj(ctx)
        return target + delta


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
        fuse_apply="none",
        fuse_mech="dag",
        fuse_dim=64,
        region_num_prev=0,
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

        # ---- (a) shared-attach augmentation: only on region-first MoE blocks with
        #      at least one previous region (region_num_prev > 0) ----
        if fuse_apply not in ("none", "cond", "all"):
            raise ValueError(f"Unsupported fuse_apply: {fuse_apply!r} (expected 'none'/'cond'/'all')")
        self.fuse_apply = fuse_apply
        self.region_num_prev = region_num_prev
        self.has_augment = (fuse_apply != "none" and region_num_prev > 0)
        if self.has_augment:
            assert use_shared_expert, "fuse_apply requires a shared expert (augment target)"
            self.aug_mod = nn.Linear(hidden_size, hidden_size)         # per-block source modulation
            self.shared_augment = SharedAugmentModule(hidden_size, fuse_dim, fuse_mech)

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

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor, aug_sources: torch.Tensor = None):
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

        ### process shared experts (+ optional region-rep shared augmentation, mode (a))
        if self.use_shared_expert:
            shared_output = self.shared_expert(hidden_states)
            if self.has_augment:
                if aug_sources is not None and aug_sources.shape[2] > 0:
                    shared_output = self._augment_shared(shared_output, aug_sources, labels)
                shared_output = self._fake_touch(shared_output)  # keep augment params in the DDP graph
            final_output = final_output + shared_output

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

    def _augment_shared(self, shared_output, sources, labels):
        """Add a zero-init gated delta to `shared_output` from K per-token region-rep
        sources. sources: [B, S, K, d] (previous regions' reps; modulated by the
        per-block aug_mod). fuse_apply='cond' augments only cond tokens; 'all' every
        token. Routing is untouched (rewrites only the shared output)."""
        B, S, d = shared_output.shape
        K = sources.shape[2]
        if K == 0:
            return shared_output
        tgt = shared_output.reshape(B * S, d)
        src = self.aug_mod(sources.reshape(B * S, K, d))  # per-block modulation
        if self.fuse_apply == "cond" and self.use_uncond_expert:
            flat_labels = labels.view(B, 1).expand(-1, S).reshape(-1)
            pos = torch.where(flat_labels != 1000)[0]
            if pos.numel() == 0:
                return shared_output
            aug = self.shared_augment(tgt[pos], src[pos])
            out = tgt.index_copy(0, pos, aug.to(tgt.dtype))
        else:
            out = self.shared_augment(tgt, src)
        return out.view(B, S, d)

    def _fake_touch(self, x):
        """Add 0 * (sum of augment params) so the augment submodule always receives
        gradient — DDP unused-parameter safety on the rare all-uncond batch."""
        s = x.new_zeros(())
        for p in self.aug_mod.parameters():
            s = s + p.sum()
        for p in self.shared_augment.parameters():
            s = s + p.sum()
        return x + 0.0 * s

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
                 use_moe=False, region_num_prev=0,
                 **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.use_moe = use_moe
        # region shared-attach count, also stored on the DiTBlock so DiT.forward can
        # read it (block.region_num_prev) to decide when to feed previous region reps.
        self.region_num_prev = region_num_prev if use_moe else 0
        if use_moe:
            if use_swiglu==False:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = SparseMoeBlock(hidden_size=hidden_size, region_num_prev=region_num_prev, **MoE_config)
            else:
                self.mlp = SparseMoeBlock(hidden_size=hidden_size, region_num_prev=region_num_prev, **MoE_config)
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

    def forward(self, x, c, label, aug_sources=None):
        # aug_sources: [B, S, K, d] previous region reps (mode (a) region-first MoE blocks); else None.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_moe:
            x_mlp, aux_loss = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), label, aug_sources=aug_sources)
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

        # ---- region / Block-AttnRes configuration ----
        self.region_size = int(MoE_config.get("region_size", 3))
        self.region_attach = MoE_config.get("region_attach", "shared")  # 'shared' | 'resid'
        self.fuse_apply = MoE_config.get("fuse_apply", "none")
        self.fuse_mech = MoE_config.get("fuse_mech", "dag")
        fuse_dim = int(MoE_config.get("fuse_dim", 64))
        if self.region_attach not in ("shared", "resid"):
            raise ValueError(f"Unsupported region_attach: {self.region_attach!r} (expected 'shared'/'resid')")
        num_regions = math.ceil(depth / self.region_size)
        # last block of each region (its representative); marks where reps are cached
        self.region_last_block = {min((r + 1) * self.region_size, depth) - 1 for r in range(num_regions)}
        self.region_last_block.add(depth - 1)

        # (a) shared attach: first MoE block of each region r>0 gets region_num_prev = r
        region_attach_block = {}
        if self.region_attach == "shared" and self.fuse_apply != "none":
            for r in range(num_regions):
                for i in range(r * self.region_size, min((r + 1) * self.region_size, depth)):
                    if use_moe_flag[i]:
                        if r > 0:
                            region_attach_block[i] = r
                        break

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob, return_labels=True)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        blocks = []
        for i in range(depth):
            blocks.append(DiTBlock(hidden_size, num_heads // (1 if use_moe_flag[i] else 1), head_dim=head_dim,
                                   mlp_ratio=mlp_ratio, qk_norm=qk_norm, use_swiglu=use_swiglu,
                                   MoE_config=MoE_config, use_moe=use_moe_flag[i],
                                   region_num_prev=region_attach_block.get(i, 0)))
        self.blocks = nn.ModuleList(blocks)

        # (b) resid attach: DiT-level module per region boundary (before region r>0's first block)
        self.resid_attach = nn.ModuleDict()
        if self.region_attach == "resid" and self.fuse_apply != "none":
            for r in range(1, num_regions):
                b = r * self.region_size
                if b < depth:
                    self.resid_attach[str(b)] = nn.ModuleDict({
                        "aug_mod": nn.Linear(hidden_size, hidden_size),
                        "augment": SharedAugmentModule(hidden_size, fuse_dim, self.fuse_mech),
                    })

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

        # Re-assert SharedAugmentModule output-projector zero-init AFTER the blanket
        # self.apply(_basic_init) above => augment delta 0 at init => step-0 identical.
        # (a) shared-attach modules live in MoE blocks; (b) resid-attach at DiT level.
        for block in self.blocks:
            if getattr(block, "use_moe", False) and getattr(block.mlp, "has_augment", False):
                block.mlp.shared_augment.zero_out()
        for key in self.resid_attach:
            self.resid_attach[key]["augment"].zero_out()

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

    def _resid_attach_fn(self, x, region_reps, key):
        """(b) Add a zero-init attn-residual over previous region reps to the main
        stream x (all tokens). key => self.resid_attach[key]. Step-0: delta == 0."""
        mod = self.resid_attach[key]
        B, S, d = x.shape
        sources = torch.stack(region_reps, dim=2)             # [B, S, r, d]
        tgt = x.reshape(B * S, d)
        src = mod["aug_mod"](sources.reshape(B * S, len(region_reps), d))
        out = mod["augment"](tgt, src)                        # [B*S, d] = tgt + zero-init delta
        return out.view(B, S, d)

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

        region_reps = []  # forward-local: completed regions' reps (last block output; kept in graph)
        for i, block in enumerate(self.blocks):
            # (b) resid: augment the main stream x at a region boundary BEFORE this block
            if self.region_attach == "resid" and str(i) in self.resid_attach and len(region_reps) > 0:
                x = self._resid_attach_fn(x, region_reps, str(i))
            # (a) shared: feed previous region reps to the region-first MoE block
            nprev = getattr(block, "region_num_prev", 0)
            if self.region_attach == "shared" and nprev > 0 and len(region_reps) >= nprev:
                sources = torch.stack(region_reps[:nprev], dim=2)  # [B, S, nprev, d]
                x = block(x, c, labels, aug_sources=sources)
            else:
                x = block(x, c, labels, aug_sources=None)
            # cache region rep when this block is a region's last block
            if i in self.region_last_block:
                region_reps.append(x)

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
