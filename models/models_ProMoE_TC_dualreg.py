"""ProMoE-TC carrying both regularizers of the 2026-09 combination study.

Two independent terms, each switchable off:

* **LS-Reg diagonal offset** (``ls_diag_strength``) -- the historical idea-1:
  a detached, load-dependent offset added to the diagonal of the
  routing-contrastive similarity matrix, so an overloaded prototype gets its
  own target weakened.  Copied line for line from
  ``models_ProMoE_TC_lsreg.py``'s ``diag`` branch, EMA buffer included, so
  ``ls_diag_strength=0.05`` reproduces ``B_lsreg_diag_idea1_s0p05``.  It
  changes only the routing loss; no new parameters.
* **Expert representation regularizer** (``expert_reg_lam``) -- every routed
  expert's output is pooled over the tokens it received (plain mean, no router
  weight), and the pooled vectors are pushed apart.  ``expert_reg_form``
  selects how:

  - ``cosine`` (default): centre the pooled vectors, then
    ``mean((cos - target)^2)`` with ``target = -1/(K-1)``.  Centring forces the
    pairwise cosines to average exactly that value, so the target is the
    regular simplex -- the most evenly spread configuration -- and 0 is
    attainable.  Cosine is scale-free, which is what this term needs: measured
    on a real run the pooled distances grow from ~20 at step 0 to ~92 by step
    1200, and a scale-dependent form cannot survive that.
  - ``l2``: the historical ``mean(exp(-L2/tau))`` repulsion, kept only so the
    old arms can be reproduced.  **It does not work here** -- a 2K-step probe
    at the best available tau (10) showed the term decaying 1278x within 2000
    steps (0.767 -> 0.0006) as the experts spread out, i.e. it switches itself
    off in the first ~30 steps of a 500K run.  At the historical tau=0.5 it is
    already ~1e-18 at step 0.  Do not use it for new work.

The two terms act on different objects -- the first reshapes the routing loss,
the second shapes the experts (its gradient reaches the expert parameters and,
through the token representations, the earlier layers, but never the router,
because the assignment is discrete and the pooled vectors carry no router
weight).

Only the MoE block inherits from the conference model.  Three exact
equivalences hold and are covered by tests:

===========================================  ==================================
``expert_reg_lam=0``, ``ls_diag_strength=0``  ``ProMoE_TC_B`` (conference)
``ls_diag_strength=0``                        the expert regularizer alone
``expert_reg_lam=0``                          ``ProMoE_TC_B_lsreg`` diag mode
===========================================  ==================================

Every block records its own diagnostics so a dead term shows up in the log from
step 0 -- the failure that cost this project a whole batch of experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from .models_ProMoE_TC import SparseMoeBlock as ConferenceSparseMoeBlock
from .modules import get_2d_sincos_pos_embed, Attention, modulate, TimestepEmbedder, LabelEmbedder, FinalLayer, MoeMLP, Mlp


EXPERT_REG_FORMS = ("cosine", "l2")


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
    """Conference MoE block plus the LS-Reg diagonal offset and the expert regularizer."""

    def __init__(self, *args,
                 expert_reg_form="cosine", expert_reg_lam=0.0, expert_reg_center=True,
                 expert_reg_margin=0.0, expert_reg_temperature=10.0,
                 expert_reg_warmup_steps=0,
                 ls_diag_sign=1.0, ls_diag_strength=0.0, ls_ema_beta=0.9,
                 num_routed_experts=None, **kwargs):
        super().__init__(*args, num_routed_experts=num_routed_experts, **kwargs)

        self.expert_reg_form = str(expert_reg_form)
        self.expert_reg_lam = float(expert_reg_lam)
        self.expert_reg_center = bool(expert_reg_center)
        self.expert_reg_margin = float(expert_reg_margin)
        self.expert_reg_temperature = float(expert_reg_temperature)
        self.expert_reg_warmup_steps = int(expert_reg_warmup_steps)
        # Weight used this step; the DiT rewrites it every forward while ramping.
        self.expert_reg_effective_lam = self.expert_reg_lam

        self.ls_diag_sign = float(ls_diag_sign)
        self.ls_diag_strength = float(ls_diag_strength)
        self.ls_ema_beta = float(ls_ema_beta)

        if self.expert_reg_form not in EXPERT_REG_FORMS:
            raise ValueError(
                f"expert_reg_form must be one of {EXPERT_REG_FORMS}, got {self.expert_reg_form!r}"
            )
        if self.expert_reg_lam < 0:
            raise ValueError(f"expert_reg_lam must be >= 0, got {self.expert_reg_lam}")
        if not -1.0 <= self.expert_reg_margin <= 1.0:
            raise ValueError(f"expert_reg_margin must lie in [-1, 1], got {self.expert_reg_margin}")
        if self.expert_reg_temperature <= 0:
            raise ValueError(f"expert_reg_temperature must be > 0, got {self.expert_reg_temperature}")
        if self.expert_reg_warmup_steps < 0:
            raise ValueError(f"expert_reg_warmup_steps must be >= 0, got {self.expert_reg_warmup_steps}")
        if self.ls_diag_strength < 0:
            raise ValueError(f"ls_diag_strength must be >= 0, got {self.ls_diag_strength}")
        if self.ls_diag_sign not in (1.0, -1.0):
            raise ValueError(f"ls_diag_sign must be +1 or -1, got {self.ls_diag_sign}")

        # Same buffers as models_ProMoE_TC_lsreg (non-persistent: not in state_dict).
        self.register_buffer("ls_load_ema", torch.zeros(self.num_routed_experts), persistent=False)
        self.register_buffer("_ls_step", torch.zeros(1), persistent=False)

        # Diagnostics, detached tensors -- reading them costs no device sync.
        self.last_expert_reg_loss = None
        self.last_expert_reg_raw_mean = None
        self.last_expert_reg_centered_mean = None
        self.last_expert_reg_max = None
        self.last_expert_reg_dist_min = None
        self.last_expert_reg_active = None
        self.last_mean_eps = None
        self.last_load_hist = None

    # ------------------------------------------------------------------ forward
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

        need_pools = self.training and self.expert_reg_effective_lam > 0
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

                # Routed experts only: the shared and unconditional experts see a
                # different token population, so they are not comparable.
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

        ### expert representation regularizer
        if need_pools:
            expert_reg_loss = self.compute_expert_reg_loss(expert_pools)
            expert_reg_loss = expert_reg_loss * self.expert_reg_effective_lam
            if loss is not None:
                loss = loss + expert_reg_loss
            else:
                loss = expert_reg_loss

        return final_output, loss

    # ------------------------------------------------- LS-Reg diagonal offset
    def compute_routing_contrastive_loss(self, token_embeddings, cluster_assignments, use_top_k=False):
        """Conference InfoNCE, optionally with idea-1's load-dependent diagonal offset.

        With ``ls_diag_strength == 0`` this defers to the conference
        implementation, which keeps the whole model bit-identical to
        ``ProMoE_TC_B``.  Otherwise it reproduces the ``diag`` branch of
        ``models_ProMoE_TC_lsreg.py`` exactly, including the clamp that branch
        applies and the EMA smoothing of the per-prototype counts.
        """
        if self.ls_diag_strength == 0.0:
            return super().compute_routing_contrastive_loss(
                token_embeddings, cluster_assignments, use_top_k=use_top_k
            )

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

        if len(valid_clusters) < 2:
            return torch.tensor(0.0, device=device)

        cluster_means = torch.cat(cluster_means, dim=0)
        valid_centers = cluster_centers[valid_clusters]

        with torch.no_grad():
            beta = self.ls_ema_beta
            if beta and beta > 0.0:
                self.ls_load_ema.mul_(beta).add_(full_counts, alpha=(1.0 - beta))
                ema = self.ls_load_ema
            else:
                ema = full_counts
            if self.training:
                self._ls_step += 1
            valid_counts = ema[valid_clusters]
            self.last_load_hist = full_counts.detach()

        num_valid = valid_centers.size(0)
        centers_norm = F.normalize(valid_centers, p=2, dim=1)
        means_norm = F.normalize(cluster_means, p=2, dim=1)
        sim_matrix = (centers_norm @ means_norm.T).clamp(-1.0, 1.0)  # bf16-safe
        temperature = self.routing_contrastive_temperature
        labels = torch.arange(num_valid, device=device)

        # idea-1: a detached, signed, load-dependent offset on the similarity
        # diagonal; sign=+1 weakens an overloaded prototype's own target.
        with torch.no_grad():
            counts = valid_counts.float()
            mean_count = counts.mean()
            rel = ((counts - mean_count) / (mean_count + 1e-6)).clamp(-1.0, 1.0)
            delta = (self.ls_diag_sign * self.ls_diag_strength * rel).detach()
            self.last_mean_eps = delta.abs().mean().detach()
        sim_matrix = sim_matrix + torch.diag(delta.to(sim_matrix.dtype))
        logits = sim_matrix / temperature
        return F.cross_entropy(logits, labels)

    # ------------------------------------------- expert representation term
    def compute_expert_reg_loss(self, expert_pools):
        """Push the pooled expert representations apart.

        ``expert_pools``: one [hidden_dim] tensor per routed expert that received
        at least one token; experts with no tokens are skipped.
        """
        device = self.cluster_centers.device
        if len(expert_pools) < 2:
            self._record_stats(None, None, None, len(expert_pools))
            return torch.zeros((), device=device)

        # fp32: under bf16 autocast, normalize + matmul drift enough to matter
        # for a quantity compared against a target.
        pooled = torch.stack(expert_pools).float()
        centered = pooled - pooled.mean(dim=0, keepdim=True)

        raw_sim = self._pairwise_cosine(pooled)
        centered_sim = self._pairwise_cosine(centered)
        dists = self._pairwise_distance(pooled)

        if self.expert_reg_form == "l2":
            # Historical form, kept for reproduction only -- see the module
            # docstring for why it switches itself off during training.
            loss = torch.exp(-dists / self.expert_reg_temperature).mean()
        else:
            sim = centered_sim if self.expert_reg_center else raw_sim
            loss = (sim - self._cosine_target(len(expert_pools))).square().mean()

        self._record_stats(raw_sim, centered_sim, dists, len(expert_pools), loss)
        return loss.to(device)

    def _cosine_target(self, num_experts):
        """The cosine value the loss pulls every pair towards.

        Centring forces the pairwise cosines to average exactly -1/(K-1):
        ``sum(c_i) = 0`` gives ``sum_{i<j} <c_i, c_j> = -0.5 * sum |c_i|^2``
        (exactly so when the pools share one norm, since each cosine also
        divides by its own two norms).  That is the floor, reached by a regular
        simplex, so it is the target rather than something to push below.  A
        one-sided ``relu(cos - 0)`` would instead clip to zero here: with K=12
        in 768 dimensions every centred pair already sits near -0.0909.

        Without centring there is no such constraint, so the target is 0 and the
        term becomes ``mean(cos^2)``.  ``expert_reg_margin`` shifts the target.
        """
        base = -1.0 / (num_experts - 1) if self.expert_reg_center else 0.0
        return base + self.expert_reg_margin

    @staticmethod
    def _pairwise_cosine(vecs):
        """Cosine of every unique pair (i < j).  A zero row yields 0, not NaN."""
        normed = F.normalize(vecs, p=2, dim=1)
        sim = (normed @ normed.T).clamp(-1.0, 1.0)
        num = vecs.size(0)
        mask = torch.triu(torch.ones(num, num, device=vecs.device, dtype=torch.bool), diagonal=1)
        return sim[mask]

    @staticmethod
    def _pairwise_distance(vecs):
        """Euclidean distance of every unique pair (i < j)."""
        num = vecs.size(0)
        diff = vecs.unsqueeze(0) - vecs.unsqueeze(1)
        dist = diff.square().sum(dim=-1).clamp_min(1e-12).sqrt()
        mask = torch.triu(torch.ones(num, num, device=vecs.device, dtype=torch.bool), diagonal=1)
        return dist[mask]

    def _record_stats(self, raw_sim, centered_sim, dists, num_active, loss=None):
        """Store this step's diagnostics as detached tensors (no .item() here).

        ``last_expert_reg_raw_mean`` is the one to watch: the loss constrains
        only the centred directions, and an expert can move its pooled mean with
        its output bias alone, so a run whose loss falls while the raw cosine
        stands still is satisfying the term without changing what the experts
        compute.  ``last_expert_reg_dist_min`` tracks the pooled spread that
        makes the l2 form die.
        """
        if raw_sim is None or raw_sim.numel() == 0:
            self.last_expert_reg_loss = None
            self.last_expert_reg_raw_mean = None
            self.last_expert_reg_centered_mean = None
            self.last_expert_reg_max = None
            self.last_expert_reg_dist_min = None
            self.last_expert_reg_active = float(num_active)
            return
        self.last_expert_reg_loss = loss.detach() if loss is not None else None
        self.last_expert_reg_raw_mean = raw_sim.detach().mean()
        self.last_expert_reg_centered_mean = centered_sim.detach().mean()
        self.last_expert_reg_max = raw_sim.detach().max()
        self.last_expert_reg_dist_min = dists.detach().min()
        self.last_expert_reg_active = float(num_active)


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
    Diffusion model with a Transformer backbone carrying both regularizers.
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

        self.is_dualreg_model = True
        self.expert_reg_lam = float(MoE_config.get("expert_reg_lam", 0.0))
        self.expert_reg_warmup_steps = int(MoE_config.get("expert_reg_warmup_steps", 0))
        self.dualreg_last_state = {"lam": self.expert_reg_lam, "scale": 1.0}

    def _set_expert_reg_state(self, training_step):
        """Ramp the expert-regularizer weight over the first ``warmup`` steps.

        The term is relatively strongest exactly when the experts are still
        forming: measured here, at step 0 it puts 1.9x the MSE gradient on the
        routed experts against 1.05x at 300K, and a probe showed the routing
        load settling within the first ~100 steps.  Ramping keeps it from
        steering that phase; every arm shares the ramp, so it is not an extra
        factor between arms.
        """
        scale = 1.0
        if self.training and self.expert_reg_lam > 0 and self.expert_reg_warmup_steps > 0:
            if training_step is None:
                raise ValueError(
                    "expert_reg warm-up needs training_step; train.py passes it "
                    "for DUALREG_MODELS"
                )
            scale = min(1.0, (int(training_step) + 1) / self.expert_reg_warmup_steps)
        effective = self.expert_reg_lam * scale
        self.dualreg_last_state = {"lam": effective, "scale": scale}
        for block in self.blocks:
            if block.use_moe:
                block.mlp.expert_reg_effective_lam = effective

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
        training_step (kwarg): global step driving the warm-up ramp
        """
        self._set_expert_reg_state(kwargs.get("training_step"))
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


__all__ = ["DiT", "DiTBlock", "SparseMoeBlock", "EXPERT_REG_FORMS"]
