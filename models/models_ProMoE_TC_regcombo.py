"""ProMoE-TC carrying the routing and expert regularizers of the 2026-10 combination study.

Three independent knobs, each a copy of an existing single-module model, so
that an arm of the combination study runs exactly the code its single-module
run did.  With every knob off the model is bit-identical to ``ProMoE_TC_B``.

* **Global class centers** (``global_center``) -- the routing-contrastive class
  centers come from all-reduced per-expert token sums and counts, and the
  gradient through this rank's tokens is multiplied by the world size
  (``models_ProMoE_TC_global_center``).
* **LS-Reg diagonal offset** (``ls_diag_strength``, ``ls_diag_sign``,
  ``ls_ema_beta``) -- idea-1 of ``models_ProMoE_TC_lsreg`` (its ``diag``
  branch): a detached, load-dependent offset on the diagonal of the
  routing-contrastive similarity matrix.  ``ls_count_scope`` picks the counts:
  ``local`` is the source behaviour (this rank's counts; under DDP the EMA
  buffer is broadcast from rank 0 before every forward), ``global``
  all-reduces the counts before the EMA so every rank uses the same offsets.
* **Expert repulsion** (``expert_contrastive_mode`` ``param`` or ``output``,
  ``expert_contrastive_lam``, ``expert_contrastive_temperature``,
  ``expert_contrastive_include_bias``, ``expert_contrastive_blocks``) --
  ``mean(exp(-L2/tau))`` over expert pairs, exactly as
  ``models_ProMoE_TC_expert_contra``.  ``expert_output_scope`` picks the pooled
  outputs: ``local`` is the source behaviour (each routed expert's mean output
  over the tokens it got on this rank), ``global`` all-reduces each expert's
  output sum and token count and multiplies the gradient through this rank's
  tokens by the world size, the same correction as the global class centers.

Combining global class centers with a local LS-Reg offset is approximate: the
offsets differ slightly between ranks, while the world-size factor assumes every
rank computes the same loss.  The study accepts this.

Every collective is issued by every rank in every training forward of a block
whose knob is on, before any early return, so ranks cannot fall out of step.
Each block records detached diagnostics that ``train.py`` logs.
"""

import inspect

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from .models_ProMoE_TC import AddAuxiliaryLoss
from .models_ProMoE_TC import SparseMoeBlock as ConferenceSparseMoeBlock
from .modules import get_2d_sincos_pos_embed, Attention, modulate, TimestepEmbedder, LabelEmbedder, FinalLayer, MoeMLP, Mlp


SCOPES = ("local", "global")
EXPERT_CONTRASTIVE_MODES = ("param", "output")
_CONFERENCE_BLOCK_ARGS = frozenset(
    inspect.signature(ConferenceSparseMoeBlock.__init__).parameters
) - {"self", "kwargs"}
# MoE_config keys the DiT consumes; they also reach every block through **MoE_config.
_DIT_LEVEL_KEYS = frozenset({"interleave", "init_MoeMLP", "expert_contrastive_blocks"})


def _distributed_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class SparseMoeBlock(ConferenceSparseMoeBlock):
    """Conference MoE block plus global centers, LS-Reg and expert repulsion."""

    def __init__(self, *args,
                 global_center=False,
                 ls_diag_strength=0.0, ls_diag_sign=1.0, ls_ema_beta=0.9, ls_count_scope="local",
                 expert_contrastive_mode="output", expert_contrastive_lam=0.0,
                 expert_contrastive_temperature=0.5, expert_contrastive_include_bias=True,
                 expert_output_scope="local",
                 **kwargs):
        # The conference block swallows unknown keys, so a misspelt knob would
        # silently switch its term off.  Refuse anything it does not know.
        unknown = sorted(set(kwargs) - _CONFERENCE_BLOCK_ARGS - _DIT_LEVEL_KEYS)
        if unknown:
            raise ValueError(f"unknown MoE_config keys for the regcombo model: {unknown}")
        super().__init__(*args, **kwargs)
        if self.phase_metric is not None:
            raise ValueError("regcombo does not support phase_metric_config.enabled")

        self.global_center = bool(global_center)
        self.ls_diag_strength = float(ls_diag_strength)
        self.ls_diag_sign = float(ls_diag_sign)
        self.ls_ema_beta = float(ls_ema_beta)
        self.ls_count_scope = str(ls_count_scope)
        self.expert_contrastive_mode = str(expert_contrastive_mode)
        self.expert_contrastive_lam = float(expert_contrastive_lam)
        self.expert_contrastive_temperature = float(expert_contrastive_temperature)
        self.expert_contrastive_include_bias = bool(expert_contrastive_include_bias)
        self.expert_output_scope = str(expert_output_scope)
        # Whether this block computes the expert repulsion (set by DiT.__init__).
        self.compute_expert_contrastive = False

        if self.ls_diag_strength < 0:
            raise ValueError(f"ls_diag_strength must be >= 0, got {self.ls_diag_strength}")
        if self.ls_diag_sign not in (1.0, -1.0):
            raise ValueError(f"ls_diag_sign must be +1 or -1, got {self.ls_diag_sign}")
        if not 0.0 <= self.ls_ema_beta < 1.0:
            raise ValueError(f"ls_ema_beta must lie in [0, 1), got {self.ls_ema_beta}")
        if self.ls_count_scope not in SCOPES:
            raise ValueError(f"ls_count_scope must be one of {SCOPES}, got {self.ls_count_scope!r}")
        if self.ls_count_scope == "global" and self.ls_diag_strength == 0.0:
            raise ValueError("ls_count_scope 'global' needs ls_diag_strength > 0")
        if self.expert_contrastive_mode not in EXPERT_CONTRASTIVE_MODES:
            raise ValueError(
                f"expert_contrastive_mode must be one of {EXPERT_CONTRASTIVE_MODES}, "
                f"got {self.expert_contrastive_mode!r}"
            )
        if self.expert_contrastive_lam < 0:
            raise ValueError(f"expert_contrastive_lam must be >= 0, got {self.expert_contrastive_lam}")
        if self.expert_contrastive_temperature <= 0:
            raise ValueError(
                f"expert_contrastive_temperature must be > 0, got {self.expert_contrastive_temperature}"
            )
        if self.expert_output_scope not in SCOPES:
            raise ValueError(f"expert_output_scope must be one of {SCOPES}, got {self.expert_output_scope!r}")
        if self.expert_output_scope == "global" and self.expert_contrastive_mode != "output":
            raise ValueError("expert_output_scope 'global' needs expert_contrastive_mode 'output'")

        # Same buffers as models_ProMoE_TC_lsreg (non-persistent: not in state_dict).
        self.register_buffer("ls_load_ema", torch.zeros(self.num_routed_experts), persistent=False)
        self.register_buffer("_ls_step", torch.zeros(1), persistent=False)

        # Diagnostics, detached tensors -- reading them costs no device sync.
        self.last_mean_eps = None
        self.last_load_hist = None
        self.last_expert_loss = None
        self.last_expert_dist_min = None
        self.last_expert_dist_mean = None
        self.last_expert_norm_mean = None
        self.last_expert_active = None

    # ------------------------------------------------------------------ forward
    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor, timestep: torch.Tensor = None):
        ### token assignment
        router_weights, expert_indices, load_balance_loss = self.compute_router(hidden_states, labels)
        batch_size, seq_len, hidden_dim = hidden_states.shape

        flat_input = hidden_states.view(-1, hidden_dim)
        flat_weights = router_weights.view(-1, self.top_k)
        flat_indices = expert_indices.view(-1, self.top_k)
        total_tokens = batch_size * seq_len

        final_output = torch.zeros(total_tokens, hidden_dim, device=hidden_states.device)

        need_expert_loss = (
            self.training
            and self.compute_expert_contrastive
            and self.expert_contrastive_lam > 0
        )
        need_expert_outputs = need_expert_loss and self.expert_contrastive_mode == "output"
        expert_output_pools = {}  # expert_id -> pooled output (local) or output sum (global)
        expert_token_counts = {}  # expert_id -> token count on this rank (global scope)

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

                # Routed experts only, as in expert_contra.
                if need_expert_outputs and expert_id < self.num_routed_experts:
                    if self.expert_output_scope == "local":
                        expert_output_pools[expert_id] = expert_output.mean(dim=0)  # [hidden_dim]
                    else:
                        expert_output_pools[expert_id] = expert_output.float().sum(dim=0)
                        expert_token_counts[expert_id] = token_ids.numel()
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

        ### expert repulsion
        if need_expert_loss:
            if self.expert_contrastive_mode == "output":
                if self.expert_output_scope == "local":
                    expert_contra_loss = self._expert_contrastive_output(expert_output_pools)
                else:
                    expert_contra_loss = self._expert_contrastive_output_global(
                        expert_output_pools, expert_token_counts, hidden_dim, hidden_states.device
                    )
            else:
                expert_contra_loss = self._expert_contrastive_param()

            expert_contra_loss = expert_contra_loss * self.expert_contrastive_lam
            if loss is not None:
                loss = loss + expert_contra_loss
            else:
                loss = expert_contra_loss

        return final_output, loss

    # ------------------------------------------------ routing-contrastive loss
    def compute_routing_contrastive_loss(self, token_embeddings, cluster_assignments, use_top_k=False):
        distributed = _distributed_world_size() > 1
        if self.ls_diag_strength == 0.0:
            if self.global_center and distributed:
                return self._global_center_loss(token_embeddings, cluster_assignments, use_top_k)
            # A single process already sees the whole batch.
            return super().compute_routing_contrastive_loss(
                token_embeddings, cluster_assignments, use_top_k=use_top_k
            )
        if self.global_center and distributed:
            return self._global_center_ls_loss(token_embeddings, cluster_assignments, use_top_k)
        return self._ls_diag_loss(token_embeddings, cluster_assignments, use_top_k, distributed)

    def _ls_diag_loss(self, token_embeddings, cluster_assignments, use_top_k, distributed):
        """The ``diag`` branch of models_ProMoE_TC_lsreg, line for line.

        With ``ls_count_scope == "global"`` the per-prototype counts are
        all-reduced before the EMA; the class means stay on this rank.
        """
        cluster_centers = self.cluster_centers
        num_clusters = cluster_centers.size(0)
        device = cluster_centers.device

        cluster_means = []
        valid_clusters = []
        full_counts = torch.zeros(num_clusters, device=device)

        for cluster_id in range(num_clusters):
            if use_top_k:
                mask = (cluster_assignments == cluster_id).any(dim=1)
            else:
                mask = (cluster_assignments == cluster_id)

            full_counts[cluster_id] = mask.sum().float()
            if mask.sum() > 0:
                cluster_mean = token_embeddings[mask].mean(dim=0, keepdim=True)
                cluster_means.append(cluster_mean)
                valid_clusters.append(cluster_id)

        if self.ls_count_scope == "global" and distributed:
            # Before the early return below, so every rank joins the collective.
            dist.all_reduce(full_counts, op=dist.ReduceOp.SUM)

        if len(valid_clusters) < 2:
            return torch.tensor(0.0, device=device)

        cluster_means = torch.cat(cluster_means, dim=0)
        valid_centers = cluster_centers[valid_clusters]

        valid_counts = self._update_load_ema(full_counts)[valid_clusters]

        num_valid = valid_centers.size(0)
        centers_norm = F.normalize(valid_centers, p=2, dim=1)
        means_norm = F.normalize(cluster_means, p=2, dim=1)
        sim_matrix = (centers_norm @ means_norm.T).clamp(-1.0, 1.0)  # bf16-safe
        temperature = self.routing_contrastive_temperature
        labels = torch.arange(num_valid, device=device)

        sim_matrix = sim_matrix + torch.diag(self._diag_offset(valid_counts).to(sim_matrix.dtype))
        logits = sim_matrix / temperature
        return F.cross_entropy(logits, labels)

    def _global_center_loss(self, token_embeddings, cluster_assignments, use_top_k):
        """models_ProMoE_TC_global_center's loss (distributed path)."""
        local_sums, local_counts = self._cluster_sums(token_embeddings, cluster_assignments, use_top_k)
        global_sums, global_counts, world_size = self._all_reduce_cluster_stats(local_sums, local_counts)
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

    def _global_center_ls_loss(self, token_embeddings, cluster_assignments, use_top_k):
        """Global class centers with the LS-Reg diagonal offset.

        The similarity matrix is the global-center one over the prototypes
        that hold a token anywhere; the offset comes from this rank's counts
        (``local``) or the all-reduced counts (``global``), smoothed by the
        same EMA as the lsreg model, and the matrix is clamped as lsreg does.
        """
        local_sums, local_counts = self._cluster_sums(token_embeddings, cluster_assignments, use_top_k)
        global_sums, global_counts, world_size = self._all_reduce_cluster_stats(local_sums, local_counts)
        device = self.cluster_centers.device
        valid_clusters = torch.nonzero(global_counts > 0, as_tuple=False).flatten()
        if valid_clusters.numel() < 2:
            return torch.tensor(0.0, device=device)

        counts = global_counts if self.ls_count_scope == "global" else local_counts
        valid_counts = self._update_load_ema(counts)[valid_clusters]

        sums = global_sums + world_size * (local_sums - local_sums.detach())
        cluster_means = sums[valid_clusters] / global_counts[valid_clusters].unsqueeze(1)
        centers_norm = F.normalize(self.cluster_centers[valid_clusters], p=2, dim=1)
        means_norm = F.normalize(cluster_means, p=2, dim=1)
        sim_matrix = (centers_norm @ means_norm.T).clamp(-1.0, 1.0)
        sim_matrix = sim_matrix + torch.diag(self._diag_offset(valid_counts).to(sim_matrix.dtype))
        labels = torch.arange(sim_matrix.size(0), device=device)
        return F.cross_entropy(sim_matrix / self.routing_contrastive_temperature, labels)

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

    @staticmethod
    def _all_reduce_cluster_stats(local_sums, local_counts):
        # Every rank reaches this once per MoE block, in the same order, even
        # when it holds no cond token, so the collective always matches.
        stats = torch.cat([local_sums.detach(), local_counts.unsqueeze(1)], dim=1)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats[:, :-1], stats[:, -1], dist.get_world_size()

    def _update_load_ema(self, full_counts):
        """The lsreg EMA of per-prototype counts; returns the smoothed counts."""
        with torch.no_grad():
            beta = self.ls_ema_beta
            if beta and beta > 0.0:
                self.ls_load_ema.mul_(beta).add_(full_counts, alpha=(1.0 - beta))
                ema = self.ls_load_ema
            else:
                ema = full_counts
            if self.training:
                self._ls_step += 1
            self.last_load_hist = full_counts.detach()
        return ema

    def _diag_offset(self, valid_counts):
        """idea-1: a detached, signed, load-dependent offset on the similarity
        diagonal; sign=+1 weakens an overloaded prototype's own target."""
        with torch.no_grad():
            counts = valid_counts.float()
            mean_count = counts.mean()
            rel = ((counts - mean_count) / (mean_count + 1e-6)).clamp(-1.0, 1.0)
            delta = (self.ls_diag_sign * self.ls_diag_strength * rel).detach()
            self.last_mean_eps = delta.abs().mean().detach()
        return delta

    # ------------------------------------------------------- expert repulsion
    def _expert_contrastive_output(self, expert_output_pools):
        """expert_contra's output loss: each routed expert's mean output on this rank."""
        device = self.cluster_centers.device
        valid_ids = sorted(expert_output_pools.keys())
        if len(valid_ids) < 2:
            self._record_expert_stats(None, len(valid_ids))
            return torch.tensor(0.0, device=device)

        pooled = torch.stack([expert_output_pools[eid] for eid in valid_ids])  # [K, hidden_dim]
        return self._pairwise_repulsion_loss(pooled)

    def _expert_contrastive_output_global(self, output_sums, token_counts, hidden_dim, device):
        """Output repulsion on each expert's mean output over the global batch.

        Each routed expert's output sum (float32) and token count are
        all-reduced.  The pooled vectors take exactly the global values; only
        this rank's sums carry gradient, multiplied by the world size, because
        every rank computes the same loss while a token is differentiated only
        on its own rank and DDP divides by the world size.
        """
        num_routed = self.num_routed_experts
        local_sums = torch.stack([
            output_sums[eid] if eid in output_sums
            else torch.zeros(hidden_dim, device=device, dtype=torch.float32)
            for eid in range(num_routed)
        ])
        local_counts = torch.tensor(
            [float(token_counts.get(eid, 0)) for eid in range(num_routed)], device=device
        )
        world_size = _distributed_world_size()
        if world_size > 1:
            # Every rank reaches this once per selected block in every training forward.
            stats = torch.cat([local_sums.detach(), local_counts.unsqueeze(1)], dim=1)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_sums, global_counts = stats[:, :-1], stats[:, -1]
        else:
            global_sums, global_counts = local_sums.detach(), local_counts
        valid_ids = torch.nonzero(global_counts > 0, as_tuple=False).flatten()
        if valid_ids.numel() < 2:
            self._record_expert_stats(None, valid_ids.numel())
            return torch.tensor(0.0, device=device)
        sums = global_sums + world_size * (local_sums - local_sums.detach())
        pooled = sums[valid_ids] / global_counts[valid_ids].unsqueeze(1)
        return self._pairwise_repulsion_loss(pooled)

    def _flatten_expert(self, expert):
        """One expert's parameters as a 1-D vector; biases dropped unless included."""
        parts = []
        for name, p in expert.named_parameters():
            if not self.expert_contrastive_include_bias and name.endswith("bias"):
                continue
            parts.append(p.flatten())
        return torch.cat(parts)

    def _expert_contrastive_param(self):
        """expert_contra's parameter loss over the routed experts."""
        param_vecs = [self._flatten_expert(self.experts[eid]) for eid in range(self.num_routed_experts)]
        param_vecs = torch.stack(param_vecs)  # [K, total_params]
        return self._pairwise_repulsion_loss(param_vecs)

    def _pairwise_repulsion_loss(self, vecs):
        """expert_contra's mean(exp(-||v_i - v_j||_2 / temperature)) over unique pairs."""
        K = vecs.size(0)
        if K < 2:
            self._record_expert_stats(None, K)
            return torch.tensor(0.0, device=vecs.device)

        diffs = vecs.unsqueeze(0) - vecs.unsqueeze(1)  # [K, K, D]
        l2_dists = diffs.norm(p=2, dim=-1)  # [K, K]

        mask = torch.triu(torch.ones(K, K, device=vecs.device, dtype=torch.bool), diagonal=1)
        pairwise_dists = l2_dists[mask]  # [K*(K-1)/2]

        loss = torch.exp(-pairwise_dists / self.expert_contrastive_temperature).mean()
        self._record_expert_stats(vecs, K, pairwise_dists, loss)
        return loss

    def _record_expert_stats(self, vecs, num_active, pairwise_dists=None, loss=None):
        """Store this step's expert-repulsion diagnostics as detached tensors.

        The loss itself (unweighted) shows whether the term is alive at this
        temperature; the pair distances show what it acts on; the mean vector
        norm shows a term being met by inflating weights or outputs rather than
        by moving them apart.
        """
        self.last_expert_active = float(num_active)
        if vecs is None:
            self.last_expert_loss = None
            self.last_expert_dist_min = None
            self.last_expert_dist_mean = None
            self.last_expert_norm_mean = None
            return
        with torch.no_grad():
            self.last_expert_loss = loss.detach().float()
            self.last_expert_dist_min = pairwise_dists.detach().float().min()
            self.last_expert_dist_mean = pairwise_dists.detach().float().mean()
            self.last_expert_norm_mean = vecs.detach().float().norm(p=2, dim=1).mean()


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
    Diffusion model with a Transformer backbone carrying the combination-study regularizers.
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

        # Expert repulsion: set the flag on the selected (0-based) blocks.  A
        # weight without blocks, or blocks without a weight, is a config error
        # that would otherwise leave the term silently off.
        expert_contrastive_blocks = list(MoE_config.get("expert_contrastive_blocks", []) or [])
        expert_contrastive_lam = float(MoE_config.get("expert_contrastive_lam", 0.0))
        if expert_contrastive_lam > 0 and not expert_contrastive_blocks:
            raise ValueError("expert_contrastive_lam > 0 needs a non-empty expert_contrastive_blocks")
        if expert_contrastive_blocks and expert_contrastive_lam <= 0:
            raise ValueError("expert_contrastive_blocks is set but expert_contrastive_lam is 0")
        for block_idx in expert_contrastive_blocks:
            if not 0 <= block_idx < depth or not self.blocks[block_idx].use_moe:
                raise ValueError(f"expert_contrastive_blocks contains block {block_idx}, which is not a MoE block")
            self.blocks[block_idx].mlp.compute_expert_contrastive = True

        self.initialize_weights()
        self.is_regcombo_model = True

    def moe_blocks(self):
        """The MoE blocks, in depth order -- used to collect the diagnostics."""
        return [block.mlp for block in self.blocks if block.use_moe]

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


__all__ = ["DiT", "DiTBlock", "SparseMoeBlock", "SCOPES", "EXPERT_CONTRASTIVE_MODES"]
