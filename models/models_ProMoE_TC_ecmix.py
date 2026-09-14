"""ProMoE-TC with Expert-Choice (EC) mixed into training.

Only the MoE block is inherited from the conference model: ``SparseMoeBlock``
extends ``models_ProMoE_TC.SparseMoeBlock``, so its construction,
initialisation, token-choice (TC) routing, TC forward and TC
routing-contrastive loss are the conference implementation unchanged.  The EC
additions and the DiT around the block are written out in this file, as in the
other variant files.  Evaluation and sampling always run TC.

``MoE_config.ecmix_config`` selects one of two ablations:

* ``mode: loss`` -- every step still routes with TC; the routing-contrastive
  loss becomes ``(1 - ratio) * TC loss + ratio * EC loss``.
* ``mode: step`` -- a ``ratio`` share of training steps are complete EC steps
  (EC routing and the EC contrastive loss); the remaining steps are TC steps.

EC follows ``models_ProMoE_EC_batch_choice.py``: on each GPU, every routed
expert takes the ``k = T / E * top_k`` conditional tokens most similar to its
prototype from the rank-local pool of ``T`` tokens, and the EC contrastive loss
contrasts each prototype with the means of all experts' selections.

``schedule: constant`` keeps ``ratio`` fixed; ``schedule: cosine`` anneals it
from ``ratio`` to 0 at ``anneal_steps``.  Whether a step is an EC step is a
pure function of the global training step, so all ranks and resumed runs
agree.  With ``ratio: 0`` the model is bit-identical to the conference model.
"""

import math
from fractions import Fraction

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from .models_ProMoE_TC import SparseMoeBlock as ConferenceSparseMoeBlock
from .modules import get_2d_sincos_pos_embed, Attention, modulate, TimestepEmbedder, LabelEmbedder, FinalLayer, MoeMLP, Mlp


#################################################################################
#                               EC mixing schedule                              #
#################################################################################
ECMIX_CONFIG_KEYS = frozenset({"mode", "schedule", "ratio", "anneal_steps"})
ECMIX_MODES = ("loss", "step")
ECMIX_SCHEDULES = ("constant", "cosine")
# The cosine schedule flattens out next to ``anneal_steps``, where its integral
# approaches an integer more slowly than float round-off; the offset keeps the
# floor below monotone there.
_EC_STEP_COUNT_EPSILON = 1e-9


def validate_ecmix_config(config):
    """Return a normalised copy of ``ecmix_config``; a missing config is off."""
    config = dict(config or {})
    unknown = sorted(set(config) - ECMIX_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown ecmix_config keys: {unknown}")
    mode = config.get("mode", "loss")
    if mode not in ECMIX_MODES:
        raise ValueError(f"ecmix_config.mode must be one of {ECMIX_MODES}, got {mode!r}")
    schedule = config.get("schedule", "constant")
    if schedule not in ECMIX_SCHEDULES:
        raise ValueError(
            f"ecmix_config.schedule must be one of {ECMIX_SCHEDULES}, got {schedule!r}"
        )
    ratio = float(config.get("ratio", 0.0))
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ecmix_config.ratio must be in [0, 1], got {ratio}")
    anneal_steps = config.get("anneal_steps")
    if schedule == "cosine":
        if anneal_steps is None or int(anneal_steps) <= 0:
            raise ValueError("the cosine schedule needs a positive ecmix_config.anneal_steps")
        anneal_steps = int(anneal_steps)
    elif anneal_steps is not None:
        raise ValueError("ecmix_config.anneal_steps only applies to the cosine schedule")
    return {
        "mode": mode,
        "schedule": schedule,
        "ratio": ratio,
        "anneal_steps": anneal_steps,
    }


def ecmix_ratio_at(step, config):
    """EC loss weight (``mode: loss``) or EC-step share (``mode: step``) at ``step``."""
    ratio = config["ratio"]
    if config["schedule"] == "constant":
        return ratio
    anneal_steps = config["anneal_steps"]
    progress = min(max(int(step), 0), anneal_steps) / anneal_steps
    return ratio * 0.5 * (1.0 + math.cos(math.pi * progress))


def ec_steps_before(step, config):
    """Number of EC steps among the training steps ``0 .. step - 1``."""
    step = max(int(step), 0)
    ratio = config["ratio"]
    if config["schedule"] == "constant":
        # Exact rational arithmetic: a 5% share is one EC step in every 20.
        share = Fraction(str(ratio))
        return (step * share.numerator) // share.denominator
    anneal_steps = config["anneal_steps"]
    capped = min(step, anneal_steps)
    # Closed-form integral of the cosine schedule over [0, capped).
    integral = ratio * 0.5 * (
        capped + anneal_steps / math.pi * math.sin(math.pi * capped / anneal_steps)
    )
    return math.floor(integral + _EC_STEP_COUNT_EPSILON)


def is_ec_step(step, config):
    """Whether training step ``step`` is a complete EC step."""
    return ec_steps_before(step + 1, config) > ec_steps_before(step, config)


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
    """Conference TC block that can also run EC during training.

    Inherited unchanged: construction, initialisation, TC routing
    (``compute_router``), the TC forward and the TC routing-contrastive loss.
    Added here: EC selection, the EC contrastive loss and the complete EC step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.phase_metric is not None:
            raise ValueError("ecmix does not support phase_metric_config.enabled")
        # DiT sets both before every forward; the defaults are the conference block.
        self.ecmix_ec_step = False
        self.ecmix_ec_loss_weight = 0.0
        self._reset_ecmix_stats()

    def _reset_ecmix_stats(self):
        self.last_tc_load_hist = None
        self.last_tc_contrastive_loss = None
        self.last_ec_contrastive_loss = None

    def forward(self, hidden_states, labels, timestep=None):
        self._reset_ecmix_stats()
        if self.training and self.ecmix_ec_step:
            return self._forward_ec_step(hidden_states, labels)
        return super().forward(hidden_states, labels, timestep)

    def compute_routing_contrastive_loss(self, token_embeddings, cluster_assignments, use_top_k=False):
        with torch.no_grad():
            self.last_tc_load_hist = self._load_histogram(cluster_assignments)
        weight = self.ecmix_ec_loss_weight
        if weight <= 0.0:
            tc_loss = super().compute_routing_contrastive_loss(
                token_embeddings, cluster_assignments, use_top_k=use_top_k
            )
            self.last_tc_contrastive_loss = tc_loss.detach()
            return tc_loss
        ec_loss = self._ec_contrastive_loss(token_embeddings)
        self.last_ec_contrastive_loss = ec_loss.detach()
        if weight >= 1.0:
            return ec_loss
        tc_loss = super().compute_routing_contrastive_loss(
            token_embeddings, cluster_assignments, use_top_k=use_top_k
        )
        self.last_tc_contrastive_loss = tc_loss.detach()
        return tc_loss * (1.0 - weight) + ec_loss * weight

    def _load_histogram(self, expert_indices):
        counts = F.one_hot(expert_indices.reshape(-1), num_classes=self.num_experts).sum(dim=0)
        return counts[: self.num_routed_experts].float()

    def _ec_select(self, flat_cond_input):
        """EC-BC selection over a rank-local pool of cond tokens.

        Returns ``(gating_scores, flat_indices)``, both shaped ``(E, k)``.
        """
        num_tokens = flat_cond_input.shape[0]
        input_norm = F.normalize(flat_cond_input, p=2, dim=-1)
        cluster_norm = F.normalize(self.cluster_centers, p=2, dim=-1)
        cos_sim_expert_view = (input_norm @ cluster_norm.T).transpose(0, 1)
        if self.router_weight_mode == "softmax":
            weights = F.softmax(cos_sim_expert_view, dim=-1)
        elif self.router_weight_mode == "sigmoid":
            weights = torch.sigmoid(cos_sim_expert_view * 1.0)
        elif self.router_weight_mode == "identity":
            weights = cos_sim_expert_view
        else:
            raise ValueError(f"Unsupported router_weight_mode: {self.router_weight_mode}")
        capacity = max(1, min(int((num_tokens / self.num_routed_experts) * self.top_k), num_tokens))
        return torch.topk(weights, k=capacity, dim=-1, sorted=False)

    def _ec_contrastive_from_means(self, expert_token_means):
        num_experts = expert_token_means.shape[0]
        if num_experts < 2:
            return torch.tensor(0.0, device=expert_token_means.device)
        centers_norm = F.normalize(self.cluster_centers, p=2, dim=1)
        means_norm = F.normalize(expert_token_means, p=2, dim=1)
        logits = (centers_norm @ means_norm.T) / self.routing_contrastive_temperature
        labels = torch.arange(num_experts, device=logits.device)
        return F.cross_entropy(logits, labels)

    def _ec_contrastive_loss(self, token_embeddings):
        """EC contrastive loss for the tokens a TC step routed (loss mode)."""
        if token_embeddings.shape[0] == 0:
            return torch.tensor(0.0, device=token_embeddings.device)
        with torch.no_grad():
            _, flat_indices = self._ec_select(token_embeddings)
        return self._ec_contrastive_from_means(token_embeddings[flat_indices].mean(dim=1))

    def _forward_ec_step(self, hidden_states, labels):
        """One complete EC step, as ``models_ProMoE_EC_batch_choice.SparseMoeBlock``."""
        identity = hidden_states
        batch_size, seq_len, hidden_dim = hidden_states.shape
        final_output = torch.zeros_like(hidden_states)
        loss = None

        if self.use_uncond_expert:
            cond_batch_mask = labels.view(-1) != 1000
        else:
            cond_batch_mask = torch.ones(batch_size, dtype=torch.bool, device=hidden_states.device)
        uncond_batch_mask = ~cond_batch_mask

        cond_experts = self.experts[:-1] if self.use_uncond_expert else self.experts
        if cond_batch_mask.any():
            cond_hidden_states = hidden_states[cond_batch_mask]
            num_tokens = cond_hidden_states.shape[0] * seq_len
            flat_input = cond_hidden_states.reshape(num_tokens, hidden_dim)

            gating_scores, flat_indices = self._ec_select(flat_input)
            expert_inputs = flat_input[flat_indices]
            expert_outputs = torch.stack(
                [cond_experts[e](expert_inputs[e]) for e in range(len(cond_experts))], dim=0
            )
            # A token picked by several experts receives the sum of their
            # outputs; a token picked by none gets only the shared expert.
            weighted = expert_outputs * gating_scores.unsqueeze(-1)
            flat_cond_output = torch.zeros(
                num_tokens, hidden_dim, device=hidden_states.device, dtype=expert_outputs.dtype
            )
            flat_cond_output.index_add_(0, flat_indices.reshape(-1), weighted.reshape(-1, hidden_dim))
            final_output[cond_batch_mask] = flat_cond_output.reshape(-1, seq_len, hidden_dim).to(
                hidden_states.dtype
            )

            with torch.no_grad():
                # Diagnostic only: the experts token choice would have picked.
                cos_sim = F.normalize(flat_input, p=2, dim=-1) @ F.normalize(
                    self.cluster_centers, p=2, dim=-1
                ).T
                self.last_tc_load_hist = self._load_histogram(cos_sim.argmax(dim=1))

            if self.training and self.routing_contrastive_lam > 0 and len(cond_experts) > 1:
                ec_loss = self._ec_contrastive_from_means(expert_inputs.mean(dim=1))
                self.last_ec_contrastive_loss = ec_loss.detach()
                loss = ec_loss * self.routing_contrastive_lam
        else:
            dummy_input = torch.zeros(1, 1, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            for expert in cond_experts:
                final_output = final_output + expert(dummy_input).sum() * 0

        if self.use_uncond_expert:
            if uncond_batch_mask.any():
                uncond_output = self.experts[-1](hidden_states[uncond_batch_mask])
                final_output[uncond_batch_mask] = uncond_output.to(final_output.dtype)
            else:
                dummy_input = torch.zeros(1, 1, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
                final_output = final_output + self.experts[-1](dummy_input).sum() * 0

        if self.use_shared_expert:
            final_output += self.shared_expert(identity).to(hidden_states.dtype)
        return final_output, loss


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
    Diffusion model with a Transformer backbone and EC mixed into training.
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
        self.is_ecmix_model = True
        self.ecmix_config = validate_ecmix_config(MoE_config.get("ecmix_config"))
        self.ecmix_last_state = {"ratio": 0.0, "ec_step": False}

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

    def _set_ecmix_state(self, training_step):
        config = self.ecmix_config
        ratio = 0.0
        ec_step = False
        if self.training and config["ratio"] > 0.0:
            if training_step is None:
                raise ValueError(
                    "ecmix training needs training_step; train.py passes it for ECMIX_MODELS"
                )
            ratio = ecmix_ratio_at(training_step, config)
            if config["mode"] == "step":
                ec_step = is_ec_step(training_step, config)
        ec_loss_weight = ratio if config["mode"] == "loss" else 0.0
        self.ecmix_last_state = {"ratio": ratio, "ec_step": ec_step}
        for block in self.blocks:
            if block.use_moe:
                block.mlp.ecmix_ec_step = ec_step
                block.mlp.ecmix_ec_loss_weight = ec_loss_weight

    def forward(self, x, timestep, context, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        timestep: (N,) tensor of diffusion timesteps
        context: (N,) tensor of class labels
        training_step (kwarg): global step that decides this step's EC mixing
        """
        self._set_ecmix_state(kwargs.get("training_step"))
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


__all__ = [
    "DiT",
    "DiTBlock",
    "SparseMoeBlock",
    "ec_steps_before",
    "ecmix_ratio_at",
    "is_ec_step",
    "validate_ecmix_config",
]
