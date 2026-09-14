"""Capacity-aware ProMoE-TC model with training-only H/R/O/P factors.

This module keeps the original expert-contrastive implementation available for
historical runs and adds a separate model class whose MoE blocks can switch on
four factors: heterogeneous routed expert widths (H), a token-count LS-Reg
diagonal offset (R), expert-output repulsion (O) and width-invariant
expert-parameter repulsion (P).  No factor changes the inference computation:
the extra terms are training-only losses, while heterogeneous routed experts
keep the original average intermediate width.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from . import models_ProMoE_TC_expert_contra as _base
from .modules import (
    Attention,
    FinalLayer,
    LabelEmbedder,
    Mlp,
    MoeMLP,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    modulate,
)


class CapacityAwareSparseMoeBlock(_base.SparseMoeBlock):
    """ProMoE block with heterogeneous widths, LS-Reg and expert repulsion."""

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
        expert_contrastive_lam=0,
        expert_contrastive_temperature=0.5,
        expert_contrastive_mode="output",
        expert_contrastive_include_bias=True,
        expert_contrastive_include_shared=False,
        expert_contrastive_include_uncond=False,
        expert_output_blocks=None,
        expert_param_blocks=None,
        hetero_expert=False,
        hetero_min_ratio=1.0,
        hetero_max_ratio=3.0,
        ls_balance_mode=None,
        ls_diag_sign=1.0,
        ls_diag_strength=0.05,
        ls_ema_beta=0.9,
        expert_contrastive_signature_bins=32,
        expert_contrastive_output_temperature=None,
        expert_contrastive_param_temperature=None,
        expert_contrastive_output_lam=None,
        expert_contrastive_param_lam=None,
        **kwargs,
    ):
        requested_mode = expert_contrastive_mode
        valid_modes = {"output", "param_signature", "dual_additive"}
        if requested_mode not in valid_modes:
            raise ValueError(
                f"Unsupported expert_contrastive_mode: {requested_mode}"
            )
        # A dual-view block needs independent coefficients: otherwise the
        # parent scalar ``expert_contrastive_lam`` would multiply the output
        # loss in every output-only block while the parameter loss exists in
        # only one block.  Keep the old scalar behavior when neither new key
        # is present, and use a unit parent scale for the explicit form.
        explicit_view_lams = (
            expert_contrastive_output_lam is not None
            or expert_contrastive_param_lam is not None
        )
        parent_expert_lam = (
            1.0 if explicit_view_lams else expert_contrastive_lam
        )
        # Parent forward collects pooled outputs only for mode == "output".
        # Map the dual-view mode to that path and keep the requested mode
        # locally.
        parent_mode = (
            "output" if requested_mode == "dual_additive" else requested_mode
        )
        # Build the block directly instead of calling the historical parent
        # constructor first.  The parent creates equal-width experts before a
        # subclass can replace them, which consumes random numbers and makes a
        # same-seed heterogeneous model follow a different initialization
        # trajectory from the established hetero-expert implementation.
        # Keeping the original construction order here (cluster centers,
        # routed experts, unconditional expert, shared expert) preserves that
        # initialization while retaining the parent loss methods.
        nn.Module.__init__(self)
        self.num_experts = num_routed_experts + (1 if use_uncond_expert else 0)
        self.num_routed_experts = num_routed_experts
        self.seq_aux = seq_aux
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.cluster_centers = nn.Parameter(
            torch.randn(num_routed_experts, hidden_size)
        )
        self.alpha = load_balance_loss_coef
        self.use_shared_expert = use_shared_expert
        self.use_uncond_expert = use_uncond_expert
        self.router_weight_mode = router_weight_mode
        self.routing_contrastive_lam = routing_contrastive_lam
        self.use_top_k_for_routing_contrastive = use_top_k_for_routing_contrastive
        self.routing_contrastive_temperature = routing_contrastive_temperature
        self.expert_contrastive_lam = parent_expert_lam
        self.expert_contrastive_temperature = expert_contrastive_temperature
        self.expert_contrastive_mode = parent_mode
        self.expert_contrastive_include_bias = expert_contrastive_include_bias
        self.expert_contrastive_include_shared = expert_contrastive_include_shared
        self.expert_contrastive_include_uncond = expert_contrastive_include_uncond
        self.compute_expert_contrastive = False
        self.is_capacity_combo_block = True

        self.requested_expert_contrastive_mode = requested_mode
        self.hetero_expert = bool(hetero_expert)
        # YAML 1.1 parsers (including PyYAML's safe loader) interpret an
        # unquoted ``off`` as boolean False.  Accept that spelling so a config
        # typo cannot turn a model without LS-Reg into a startup failure.
        if ls_balance_mode is None or ls_balance_mode is False:
            ls_balance_mode = "off"
        if ls_balance_mode not in {"off", "token"}:
            raise ValueError("ls_balance_mode must be one of: off, token")
        # ``token`` is the historical, already-tested diagonal LS-Reg.
        self.ls_balance_mode = ls_balance_mode
        self.ls_diag_sign = float(ls_diag_sign)
        self.ls_diag_strength = float(ls_diag_strength)
        self.ls_ema_beta = float(ls_ema_beta)
        self.expert_contrastive_signature_bins = int(
            expert_contrastive_signature_bins
        )
        # Keep the historical output and parameter temperatures independent.
        # The historical arms used 0.5 for output repulsion and 0.7 for
        # parameter repulsion; the old shared temperature remains the fallback.
        self.expert_contrastive_output_temperature = float(
            expert_contrastive_temperature
            if expert_contrastive_output_temperature is None
            else expert_contrastive_output_temperature
        )
        self.expert_contrastive_param_temperature = float(
            expert_contrastive_temperature
            if expert_contrastive_param_temperature is None
            else expert_contrastive_param_temperature
        )
        self._explicit_view_lams = explicit_view_lams
        if explicit_view_lams:
            self.expert_contrastive_output_lam = float(
                0.0
                if expert_contrastive_output_lam is None
                else expert_contrastive_output_lam
            )
            self.expert_contrastive_param_lam = float(
                0.0
                if expert_contrastive_param_lam is None
                else expert_contrastive_param_lam
            )
            if (
                self.expert_contrastive_output_lam < 0.0
                or self.expert_contrastive_param_lam < 0.0
            ):
                raise ValueError(
                    "explicit expert contrastive view coefficients must be non-negative"
                )
        else:
            # These attributes are still useful in diagnostics, while the
            # parent scalar remains authoritative without explicit view keys.
            self.expert_contrastive_output_lam = float(
                expert_contrastive_lam
            )
            self.expert_contrastive_param_lam = float(
                expert_contrastive_lam
            )
        # The model-level block lists are consumed by ``DiT``.  Keeping them
        # explicit here prevents them from being forwarded through ``kwargs``.
        self.expert_output_blocks = tuple(
            int(index) for index in (expert_output_blocks or [])
        )
        self.expert_param_blocks = tuple(
            int(index) for index in (expert_param_blocks or [])
        )
        if self.expert_contrastive_signature_bins < 1:
            raise ValueError("expert_contrastive_signature_bins must be positive")
        if (
            self.expert_contrastive_output_temperature <= 0.0
            or self.expert_contrastive_param_temperature <= 0.0
        ):
            raise ValueError(
                "expert contrastive temperatures must be positive"
            )
        if requested_mode == "dual_additive" and (
            expert_contrastive_include_shared or expert_contrastive_include_uncond
        ):
            raise ValueError(
                "dual_additive compares routed experts only; use the "
                "param_signature mode to include shared/unconditional experts"
            )

        if self.hetero_expert:
            sizes = self._make_heterogeneous_sizes(
                num_routed_experts=num_routed_experts,
                hidden_size=hidden_size,
                target_intermediate_size=moe_intermediate_size,
                min_ratio=float(hetero_min_ratio),
                max_ratio=float(hetero_max_ratio),
            )
            routed = [
                MoeMLP(hidden_size=hidden_size, intermediate_size=size)
                for size in sizes
            ]
            if use_uncond_expert:
                routed.append(
                    MoeMLP(
                        hidden_size=hidden_size,
                        intermediate_size=moe_intermediate_size,
                    )
                )
            self.experts = nn.ModuleList(routed)
        else:
            sizes = [int(moe_intermediate_size)] * num_routed_experts
            routed = [
                MoeMLP(hidden_size=hidden_size, intermediate_size=size)
                for size in sizes
            ]
            if use_uncond_expert:
                routed.append(
                    MoeMLP(
                        hidden_size=hidden_size,
                        intermediate_size=moe_intermediate_size,
                    )
                )
            self.experts = nn.ModuleList(routed)

        if use_shared_expert:
            self.shared_expert = MoeMLP(
                hidden_size=hidden_size,
                intermediate_size=shared_expert_intermediate_size,
            )

        # Match the parent block's cluster-center initialization.  The model
        # level ``initialize_weights`` call below then applies the same Xavier
        # pass to all Linear modules as the base and historical hetero models.
        self._init_weights()

        self.expert_intermediate_sizes = tuple(sizes)
        # Runs stop at 300K for evaluation and may then resume to 500K.
        # Persist this EMA so the second phase follows the same routing state
        # instead of silently restarting the smoothing statistics.
        self.register_buffer(
            "ls_load_ema",
            torch.zeros(num_routed_experts, dtype=torch.float32),
            persistent=True,
        )
        self.last_load_hist = None
        self.last_mean_eps = None
        self.last_expert_output_loss = None
        self.last_expert_param_loss = None

    @staticmethod
    def _make_heterogeneous_sizes(
        num_routed_experts,
        hidden_size,
        target_intermediate_size,
        min_ratio,
        max_ratio,
    ):
        if num_routed_experts < 1:
            raise ValueError("num_routed_experts must be positive")
        if min_ratio <= 0.0 or max_ratio < min_ratio:
            raise ValueError("invalid heterogeneous expert ratio range")
        # Use the same floor-based width rule as the established
        # ``models_ProMoE_TC_hetero_expert.py`` implementation.  For the B
        # setting (target width = 2 * hidden size) this differs from an exact
        # equal-sum rounding by at most a few hidden units (<0.03% FLOPs), but
        # it keeps the widths directly comparable to the historical result.
        ratios = torch.linspace(
            min_ratio,
            max_ratio,
            num_routed_experts,
            dtype=torch.float64,
        )
        sizes = [
            max(1, int(float(hidden_size) * ratio.item()))
            for ratio in ratios
        ]
        return sizes

    def compute_routing_contrastive_loss(
        self,
        token_embeddings,
        cluster_assignments,
        use_top_k=False,
    ):
        cluster_means = []
        local_valid_clusters = []
        counts = torch.zeros(
            self.num_routed_experts,
            device=token_embeddings.device,
            dtype=torch.float32,
        )
        for cluster_id in range(self.num_routed_experts):
            if use_top_k:
                mask = (cluster_assignments == cluster_id).any(dim=1)
            else:
                mask = cluster_assignments == cluster_id
            count = mask.sum()
            counts[cluster_id] = count.float()
            if count > 0:
                cluster_means.append(token_embeddings[mask].mean(dim=0, keepdim=True))
                local_valid_clusters.append(cluster_id)

        # Keep the histogram local here.  This is intentional: the historical
        # LS-Reg implementation uses each rank's batch counts and lets DDP
        # average the resulting parameter gradients.  A separate diagnostic
        # reduction in train.py reports the cross-rank mean.
        self.last_load_hist = counts.detach()

        if self.ls_balance_mode == "off":
            self.last_mean_eps = None
            return super().compute_routing_contrastive_loss(
                token_embeddings,
                cluster_assignments,
                use_top_k=use_top_k,
            )

        if len(local_valid_clusters) < 2:
            # There is no valid InfoNCE matrix on this rank.  Report a zero
            # offset so diagnostics never retain a stale value from a
            # previous step.
            self.last_mean_eps = counts.new_zeros(())
            return torch.tensor(0.0, device=token_embeddings.device)

        with torch.no_grad():
            # This is the exact historical diagonal correction: smooth the
            # per-prototype token counts, compare valid rows to their mean,
            # and add a detached signed offset to the similarity diagonal.
            beta = self.ls_ema_beta
            if beta > 0.0:
                self.ls_load_ema.mul_(beta).add_(
                    counts, alpha=1.0 - beta
                )
                smoothed = self.ls_load_ema
            else:
                smoothed = counts
            valid_load = smoothed[local_valid_clusters]
            reference = valid_load.mean()
            relative = (
                (valid_load - reference)
                / (reference + 1e-6)
            ).clamp(-1.0, 1.0)
            delta = (
                self.ls_diag_sign * self.ls_diag_strength * relative
            ).detach()
            self.last_mean_eps = delta.abs().mean().detach()

        valid_clusters = local_valid_clusters
        valid_centers = self.cluster_centers[valid_clusters]
        cluster_means = torch.cat(cluster_means, dim=0)
        centers_norm = F.normalize(valid_centers, p=2, dim=1)
        means_norm = F.normalize(cluster_means, p=2, dim=1)
        sim_matrix = (centers_norm @ means_norm.T).clamp(-1.0, 1.0)

        sim_matrix = sim_matrix + torch.diag(delta.to(sim_matrix.dtype))
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        logits = sim_matrix / self.routing_contrastive_temperature
        return F.cross_entropy(logits, labels)

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor):
        # The parent forward can legitimately skip an output loss when a rank
        # sees fewer than two experts.  Clear per-step hooks first so a stale
        # value from the preceding batch is never logged or reduced.
        self.last_load_hist = None
        self.last_mean_eps = None
        self.last_expert_output_loss = None
        self.last_expert_param_loss = None
        return super().forward(hidden_states, labels)

    def _expert_contrastive_output(self, expert_output_pools):
        valid_ids = sorted(expert_output_pools.keys())
        device = self.cluster_centers.device
        if len(valid_ids) < 2:
            output_loss = torch.zeros((), device=device)
            if self.requested_expert_contrastive_mode == "dual_additive":
                # Parameter geometry does not depend on the current token
                # assignments, so it remains a valid view even when this rank
                # has too few active output pools.
                param_loss = self._parameter_signature_loss()
                self.last_expert_output_loss = output_loss.detach()
                self.last_expert_param_loss = param_loss.detach()
                return self._weighted_output_loss(output_loss) + self._weighted_param_loss(
                    param_loss
                )
            self.last_expert_output_loss = output_loss.detach()
            self.last_expert_param_loss = None
            return self._weighted_output_loss(output_loss)

        pooled = torch.stack([expert_output_pools[eid] for eid in valid_ids])
        output_loss = self._pairwise_repulsion_loss_with_temperature(
            pooled, self.expert_contrastive_output_temperature
        )
        if self.requested_expert_contrastive_mode != "dual_additive":
            self.last_expert_output_loss = output_loss.detach()
            self.last_expert_param_loss = None
            return self._weighted_output_loss(output_loss)

        # Add the two views after computing them independently.  This avoids
        # hiding a change in one view inside a concatenated vector and keeps
        # each contribution separately measurable.  Explicit view coefficients
        # are applied inside the two helpers, so output-only blocks and a
        # dual-view block use the same calibrated strengths.
        param_loss = self._parameter_signature_loss()
        combined = self._weighted_output_loss(output_loss) + self._weighted_param_loss(
            param_loss
        )
        self.last_expert_output_loss = output_loss.detach()
        self.last_expert_param_loss = param_loss.detach()
        return combined

    @staticmethod
    def _pool_and_normalize(profile, bins=None):
        """Pool a one-dimensional summary and remove its arbitrary scale."""
        if bins is not None:
            profile = F.adaptive_avg_pool1d(
                profile.reshape(1, 1, -1), bins
            ).reshape(-1)
        # Clamp the squared norm before sqrt so all-zero bias summaries have
        # a finite, zero gradient rather than NaN from d(sqrt(x))/dx at x=0.
        scale = profile.float().square().mean().clamp_min(1e-12).sqrt()
        return profile / scale.to(profile.dtype)

    @classmethod
    def _parameter_profile(cls, values, reduce_dim, bins=None):
        """Summarize a parameter matrix without depending on its width.

        Positive energy curves alone can make two experts look identical even
        when their signed directions differ, and they encourage artificial
        sparse profiles.  Mean, spread, and RMS retain both signed and
        magnitude information; pooling the intermediate-axis summaries gives
        every expert the same signature length even when its hidden width is
        different.
        """
        values = values.float()
        mean = values.mean(dim=reduce_dim)
        centered = values - values.mean(dim=reduce_dim, keepdim=True)
        # The singleton reduction used for bias vectors has exactly zero
        # spread.  Clamp before sqrt to keep the signature differentiable and
        # finite at initialization.
        spread = centered.square().mean(dim=reduce_dim).clamp_min(1e-12).sqrt()
        rms = values.square().mean(dim=reduce_dim).clamp_min(1e-12).sqrt()
        return torch.cat(
            [
                cls._pool_and_normalize(mean, bins=bins),
                cls._pool_and_normalize(spread, bins=bins),
                cls._pool_and_normalize(rms, bins=bins),
            ]
        )

    def _width_invariant_signature(self, expert):
        """Return a fixed-length parameter geometry vector for any width."""
        up = expert.up_proj.weight
        down = expert.down_proj.weight
        bins = self.expert_contrastive_signature_bins
        parts = [
            # Input/output-axis summaries retain the common hidden dimension.
            self._parameter_profile(up, reduce_dim=0),
            self._parameter_profile(down, reduce_dim=1),
            # Intermediate-axis summaries are pooled to a fixed length.
            self._parameter_profile(up, reduce_dim=1, bins=bins),
            self._parameter_profile(down, reduce_dim=0, bins=bins),
        ]
        if self.expert_contrastive_include_bias:
            if expert.up_proj.bias is None or expert.down_proj.bias is None:
                raise ValueError(
                    "width-invariant parameter signatures require linear biases "
                    "when expert_contrastive_include_bias=True"
                )
            parts.extend(
                [
                    self._parameter_profile(
                        expert.up_proj.bias.reshape(-1, 1), reduce_dim=1, bins=bins
                    ),
                    self._parameter_profile(
                        expert.down_proj.bias.reshape(-1, 1), reduce_dim=1
                    ),
                ]
            )
        return F.normalize(torch.cat(parts), p=2, dim=0)

    def _expert_contrastive_param(self):
        # The parent forward calls this only for parameter-only blocks, whose
        # mode is ``param_signature``.
        param_loss = self._parameter_signature_loss()
        self.last_expert_output_loss = None
        self.last_expert_param_loss = param_loss.detach()
        return self._weighted_param_loss(param_loss)

    def _weighted_output_loss(self, loss):
        """Apply the explicit output-view coefficient, if configured."""
        if self._explicit_view_lams:
            return loss * self.expert_contrastive_output_lam
        return loss

    def _weighted_param_loss(self, loss):
        """Apply the explicit parameter-view coefficient, if configured."""
        if self._explicit_view_lams:
            return loss * self.expert_contrastive_param_lam
        return loss

    def _parameter_signature_loss(self):
        """Compare fixed-length parameter geometry for routed experts."""

        experts = [
            self.experts[expert_id]
            for expert_id in range(self.num_routed_experts)
        ]
        if self.expert_contrastive_include_shared and self.use_shared_expert:
            experts.append(self.shared_expert)
        if self.expert_contrastive_include_uncond and self.use_uncond_expert:
            # The unconditional expert is stored in ``self.experts`` after
            # the routed experts; it is distinct from ``shared_expert`` even
            # though both options use the same logical index in some configs.
            experts.append(self.experts[self.num_experts - 1])
        signatures = []
        for expert in experts:
            signatures.append(self._width_invariant_signature(expert))
        if len(signatures) < 2:
            return torch.tensor(0.0, device=self.cluster_centers.device)
        signatures = torch.stack(signatures)
        # Use the same exponential L2 objective as the historical equal-width
        # ``param`` experiment.  The fixed-length signatures make that
        # objective valid for heterogeneous experts without changing its
        # functional form.
        return self._pairwise_repulsion_loss_with_temperature(
            signatures, self.expert_contrastive_param_temperature
        )

    @staticmethod
    def _pairwise_repulsion_loss_with_temperature(vectors, temperature):
        """Historical exp(-L2/temperature) repulsion for fixed-size views."""
        if vectors.size(0) < 2:
            return torch.zeros((), device=vectors.device, dtype=vectors.dtype)
        diffs = vectors.unsqueeze(0) - vectors.unsqueeze(1)
        distances = diffs.float().norm(p=2, dim=-1)
        mask = torch.triu(
            torch.ones(
                distances.size(0),
                distances.size(1),
                device=distances.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        return torch.exp(-distances[mask] / float(temperature)).mean()


class CapacityAwareDiTBlock(nn.Module):
    """DiT block using CapacityAwareSparseMoeBlock for its MoE path."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim=None,
        mlp_ratio=4.0,
        use_swiglu=False,
        MoE_config=None,
        use_moe=False,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            qkv_bias=True,
            **block_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.use_moe = use_moe
        if use_moe:
            self.mlp = CapacityAwareSparseMoeBlock(
                hidden_size=hidden_size,
                **MoE_config,
            )
        elif use_swiglu:
            self.mlp = MoeMLP(
                hidden_size=hidden_size,
                intermediate_size=mlp_hidden_dim,
            )
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, label):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        if self.use_moe:
            x_mlp, aux_loss = self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp), label
            )
            if aux_loss is not None:
                x_mlp = _base.AddAuxiliaryLoss.apply(x_mlp, aux_loss)
            return x + gate_mlp.unsqueeze(1) * x_mlp
        return x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )


class DiT(_base.DiT):
    """ProMoE-TC model whose MoE blocks can enable the H/R/O/P factors."""

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
        nn.Module.__init__(self)
        self.is_capacity_combo_model = True
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.MoE_config = MoE_config

        use_moe_flag = [True] * depth
        if self.MoE_config.interleave:
            use_moe_flag = [i % 2 == 1 for i in range(depth)]

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob, return_labels=True
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                CapacityAwareDiTBlock(
                    hidden_size,
                    num_heads,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    use_swiglu=use_swiglu,
                    MoE_config=MoE_config,
                    use_moe=use_moe_flag[i],
                )
                for i in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.init_MoeMLP = MoE_config.init_MoeMLP

        # Output and parameter repulsion use independent block sets: output
        # regularization historically used all six MoE blocks, whereas the
        # parameter regularization used a single block.  A missing or empty
        # list disables that view.
        output_blocks = tuple(
            int(index) for index in MoE_config.get("expert_output_blocks", [])
        )
        param_blocks = tuple(
            int(index) for index in MoE_config.get("expert_param_blocks", [])
        )
        valid_indices = set(output_blocks) | set(param_blocks)
        for block_idx in valid_indices:
            if block_idx < 0 or block_idx >= len(self.blocks):
                raise ValueError(
                    f"expert contrastive block index out of range: {block_idx}"
                )
            if not self.blocks[block_idx].use_moe:
                raise ValueError(
                    f"expert contrastive blocks contains non-MoE block {block_idx}"
                )

        for block_idx, block in enumerate(self.blocks):
            if not block.use_moe:
                continue
            output_enabled = block_idx in output_blocks
            param_enabled = block_idx in param_blocks
            block.mlp.compute_expert_contrastive = output_enabled or param_enabled
            if not block.mlp.compute_expert_contrastive:
                continue
            if output_enabled and param_enabled:
                # Add the two calibrated views separately so each contribution
                # can be reported and ablated independently.
                block_mode = "dual_additive"
                parent_mode = "output"
            elif output_enabled:
                block_mode = "output"
                parent_mode = "output"
            else:
                block_mode = "param_signature"
                parent_mode = "param_signature"
            block.mlp.requested_expert_contrastive_mode = block_mode
            block.mlp.expert_contrastive_mode = parent_mode

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the capacity-aware model without assuming gate_proj.

        ``MoeMLP`` in this repository is a two-matrix GELU MLP (up/down),
        whereas the inherited helper was written for a gated three-matrix
        MLP.  Keeping the initialization here makes the new registered model
        valid for both values of ``init_MoeMLP``.
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5),
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

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

        if self.init_MoeMLP:
            for block in self.blocks:
                if not block.use_moe:
                    continue
                experts = list(block.mlp.experts)
                if block.mlp.use_shared_expert:
                    experts.append(block.mlp.shared_expert)
                for expert in experts:
                    nn.init.normal_(expert.up_proj.weight, std=0.006)
                    nn.init.normal_(expert.down_proj.weight, std=0.006)
                    if expert.up_proj.bias is not None:
                        nn.init.constant_(expert.up_proj.bias, 0)
                    if expert.down_proj.bias is not None:
                        nn.init.constant_(expert.down_proj.bias, 0)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """Classifier-free guidance helper with the plain-tensor contract."""
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


__all__ = ["CapacityAwareSparseMoeBlock", "CapacityAwareDiTBlock", "DiT"]
