"""ProMoE-TC whose prototype routing gets an additive correction computed from
the model's own estimate of the clean latent ("correction router").

In every MoE block a small MLP, the corrector, turns four inputs into a
correction ``delta`` of the routed-expert scores:

* ``e``: the patch embedding (``x_embedder`` + ``pos_embed``) of a reference
  estimate of the clean latent, ``x0_hat``, at the token's position;
* ``s``: the block's raw token-prototype cosines;
* ``c``: the sample's conditioning ``t_emb + y_emb`` (the adaLN input);
* ``r``: the timestep embedding of the reference's noise level ``sigma_ref``.

All four inputs are detached, so the corrector trains only its own weights.
Each input is projected to ``hidden_dim``, the projections are summed, then
SiLU and a linear map to the expert scores; that output layer starts at zero.
Expert selection takes the top-k of ``s + delta``, and the selected expert's
output is weighted by ``s_sel + delta_sel``.  The routing-contrastive loss
groups its clusters by that corrected selection; its InfoNCE matrix
(prototypes against cluster means) is the conference one.

The reference.  By default, at sampling step i it is the previous step's conditional
prediction, ``x0_hat = x_{i-1} - sigma_{i-1} * v_{i-1}`` with
``sigma_ref = sigma_{i-1}`` (sample.py carries it over); the first step has
none (``x0_hat = 0``, ``sigma_ref = 1``). The sampling-only switch
``corr_reference_velocity=cfg`` instead extrapolates with the CFG-fused velocity.
A training step has no previous
step, so the model builds one for each prototype-routed image.  With
probability ``no_reference_prob`` the image gets no reference.  Otherwise a gap
``k`` is drawn uniformly from ``skip_intervals``, in thousandths of sigma (one
Euler step of a ``1000 / k``-step sampler), and ``sigma_ref = min(sigma +
k / 1000, 1)``.  By default (``previous_state_mode: model_velocity``), two
no-grad passes without reference then step back along the model's own velocity:

    v = model(x_t, sigma);   x_prev = x_t + (sigma_ref - sigma) * v
    x0_hat = x_prev - sigma_ref * model(x_prev, sigma_ref)

``x_prev`` is a function of ``x_t`` alone, as the previous point of a sampling
trajectory is a function of the current one.  Building it from the clean
latent with the same noise instead would put both points on the straight
conditional path, which two points determine together with its clean end, and
the reference would leak the target.

The training-only ablation ``previous_state_mode: same_noise`` deliberately
builds that leaking previous state, from the exact clean latent ``z`` and noise
used for ``x_t``:

    x_prev = (1 - sigma_ref) * z + sigma_ref * noise
    x0_hat = x_prev - sigma_ref * model(x_prev, sigma_ref)

This uses one no-grad pass instead of two. Only the previous-state construction
changes: the gap and no-reference draws, the reference formula
``x0_hat = x_prev - sigma_ref * v(x_prev)``, and the sampling path stay the same.

The coin and the gaps come from a generator train.py seeds from the run's
seed and the micro-batch's position (step, micro-step, rank) alone, so the
global RNG stream (data, noise, sigma and label dropout) is the conference
model's, and a resumed run draws what an uninterrupted one does.  Dropped
labels must go to the uncond expert (``use_uncond_expert``), which never
routes with a reference, in training as in sampling.
The corrector is built and initialised without touching the global RNG and
outputs exactly zero at step 0, so the first training step of a fresh model
routes, predicts and updates the conference weights exactly as
``ProMoE_TC_B``'s does; from then on the corrector learns.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from .models_ProMoE_TC import AddAuxiliaryLoss
from .models_ProMoE_TC import SparseMoeBlock as ConferenceSparseMoeBlock
from .modules import get_2d_sincos_pos_embed, Attention, modulate, TimestepEmbedder, LabelEmbedder, FinalLayer, MoeMLP, Mlp


CORR_ROUTER_CONFIG_KEYS = ("hidden_dim", "skip_intervals", "no_reference_prob")
# The conference block sends the dropped class label to the unconditional expert.
UNCOND_LABEL = 1000
# train.py and sample.py both use t = 1000 * sigma.
NUM_TRAIN_TIMESTEPS = 1000
# Diagnostics bin the image's sigma into this many equal-width bins on [0, 1].
NUM_SIGMA_BINS = 5
# Rows of a block's per-sigma-bin diagnostic sums (``last_corr_bins``), over its
# prototype-routed tokens: |delta| averaged over the experts, token count,
# switched-expert count and token count for images with / without a reference,
# summed gate, and the count of negative gates.  A token "switched" when the
# top-1 of s + delta differs from the top-1 of s.
CORR_BIN_ROWS = (
    "delta_abs",
    "tokens",
    "switch_ref",
    "ref_tokens",
    "switch_noref",
    "noref_tokens",
    "gate",
    "gate_negative",
)
# Seed of the private generator that initialises the correctors.
CORRECTOR_INIT_SEED = 20261022


def sampling_reference(latents, prediction, sigma):
    """The reference the next sampling step routes with, as ``forward`` keyword arguments.

    It is this step's estimate of the clean latent,
    ``x0_hat = x_t - sigma_t * v_t``, at noise level ``sigma_t``.
    latents: (N, C, ...) the step's input x_t; prediction: the model's
    conditional or CFG-fused output, whose first C channels are the velocity; sigma: the
    step's noise level (a scalar).
    """
    sigma = torch.as_tensor(sigma, device=latents.device, dtype=torch.float32).reshape(())
    velocity = prediction[:, : latents.shape[1]].float().reshape(latents.shape)
    return {
        "corr_reference_x0": latents.float() - sigma * velocity,
        "corr_reference_sigma": sigma.expand(latents.shape[0]),
    }


def _require_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"corr_router_config.{name} must be an integer, got {value!r}")
    return value


def parse_corr_router_config(corr_router_config):
    """Validate three required settings and the optional previous-state mode."""
    if corr_router_config is None:
        raise ValueError(
            "corr_router models need DiT_*_config.corr_router_config with keys "
            + ", ".join(CORR_ROUTER_CONFIG_KEYS)
        )
    config = dict(corr_router_config)
    missing = [key for key in CORR_ROUTER_CONFIG_KEYS if key not in config]
    unknown = sorted(set(config) - set(CORR_ROUTER_CONFIG_KEYS) - {"previous_state_mode"})
    if missing or unknown:
        raise ValueError(f"corr_router_config: missing keys {missing}, unknown keys {unknown}")

    hidden_dim = _require_int(config["hidden_dim"], "hidden_dim")
    if hidden_dim < 1:
        raise ValueError("corr_router_config.hidden_dim must be >= 1")
    skip_intervals = config["skip_intervals"]
    if isinstance(skip_intervals, (str, bytes)) or not hasattr(skip_intervals, "__iter__"):
        raise ValueError("corr_router_config.skip_intervals must be a list of integers")
    skip_intervals = tuple(_require_int(value, "skip_intervals") for value in skip_intervals)
    if not skip_intervals:
        raise ValueError("corr_router_config.skip_intervals must not be empty")
    if any(not 1 <= value < NUM_TRAIN_TIMESTEPS for value in skip_intervals):
        raise ValueError(f"corr_router_config.skip_intervals must lie in [1, {NUM_TRAIN_TIMESTEPS})")
    if len(set(skip_intervals)) != len(skip_intervals):
        raise ValueError("corr_router_config.skip_intervals must not repeat a value")
    no_reference_prob = float(config["no_reference_prob"])
    if not (math.isfinite(no_reference_prob) and 0.0 <= no_reference_prob < 1.0):
        raise ValueError("corr_router_config.no_reference_prob must lie in [0, 1)")
    previous_state_mode = config.get("previous_state_mode", "model_velocity")
    if not isinstance(previous_state_mode, str) or previous_state_mode not in ("model_velocity", "same_noise"):
        raise ValueError("corr_router_config.previous_state_mode must be 'model_velocity' or 'same_noise'")
    return {
        "hidden_dim": hidden_dim,
        "skip_intervals": skip_intervals,
        "no_reference_prob": no_reference_prob,
        "previous_state_mode": previous_state_mode,
    }


#################################################################################
#                                 ProMoE Layer                                  #
#################################################################################
class RoutingCorrector(nn.Module):
    """Two-layer MLP from the corrector inputs to a correction of the expert scores.

    The per-token inputs (the reference embedding and the raw cosines) and the
    per-sample inputs (the conditioning and the reference noise-level
    embedding) are each projected to ``hidden_dim`` and summed; SiLU; then a
    linear map to the ``num_routed_experts`` scores.  ``reset_parameters``
    zeroes that output layer, so a fresh corrector returns exactly zero.
    """

    def __init__(self, hidden_size, num_routed_experts, hidden_dim):
        super().__init__()
        self.reference_proj = nn.Linear(hidden_size, hidden_dim)
        self.score_proj = nn.Linear(num_routed_experts, hidden_dim, bias=False)
        self.condition_proj = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.reference_sigma_proj = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, num_routed_experts)

    @torch.no_grad()
    def reset_parameters(self, generator):
        """Xavier-uniform inputs (DiT's Linear init) drawn from ``generator``, zero output."""
        for linear in (self.reference_proj, self.score_proj, self.condition_proj, self.reference_sigma_proj):
            fan_out, fan_in = linear.weight.shape
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            weight = torch.rand(linear.weight.shape, generator=generator) * (2.0 * bound) - bound
            linear.weight.copy_(weight)
            if linear.bias is not None:
                linear.bias.zero_()
        self.out.weight.zero_()
        self.out.bias.zero_()

    def sample_hidden(self, condition, reference_sigma_embed):
        """The per-sample part of the hidden layer, (N, hidden_dim)."""
        return self.condition_proj(condition) + self.reference_sigma_proj(reference_sigma_embed)

    def forward(self, reference_embed, scores, sample_hidden):
        hidden = self.reference_proj(reference_embed) + self.score_proj(scores) + sample_hidden
        return self.out(F.silu(hidden))


class CorrRouterBlockMixin:
    """Routing correction for a token-choice SparseMoeBlock with identity weights.

    Mixed in before a conference-compatible block class.  The model hands the
    block its corrector inputs with ``set_corr_context`` around each pass; a
    block without a context routes exactly as its base class.  ``last_corr_bins``
    holds the per-sigma-bin diagnostic sums (rows ``CORR_BIN_ROWS``) of the
    latest recorded pass.
    """

    _corr_context = None
    _corr_record = False
    _corr_delta = None
    last_corr_bins = None

    def init_corr_router(self, hidden_dim, generator):
        if self.router_weight_mode != "identity":
            raise ValueError("corr_router needs router_weight_mode: identity (the gate is the corrected score)")
        if self.phase_metric is not None:
            raise ValueError("corr_router does not support phase_metric_config.enabled")
        if self.alpha > 0:
            raise ValueError("corr_router does not support load_balance_loss_coef > 0")
        if not self.use_uncond_expert:
            # sample.py gives the uncond pass no reference; without the uncond
            # expert its tokens would be prototype-routed and miss one.
            raise ValueError("corr_router needs use_uncond_expert: True (dropped labels go to the uncond expert)")
        # Built inside fork_rng and initialised from ``generator``: adding the
        # corrector leaves the global RNG, and with it every later draw,
        # exactly as the conference model leaves it.
        with torch.random.fork_rng(devices=[]):
            self.corrector = RoutingCorrector(self.hidden_size, self.num_routed_experts, hidden_dim)
        self.corrector.reset_parameters(generator)

    def set_corr_context(self, context, record=False):
        """Enter a pass with corrector inputs ``context`` (a dict built by the model), or leave it (``None``)."""
        self._corr_context = context
        self._corr_record = bool(record) and context is not None
        self._corr_delta = None

    def compute_router(self, hidden_states, labels, timestep=None):
        context = self._corr_context
        if context is None:
            return super().compute_router(hidden_states, labels, timestep)

        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        flat_input = hidden_states.view(-1, self.hidden_size)
        flat_labels = labels.view(batch_size, 1).expand(-1, seq_len).reshape(-1)
        # use_uncond_expert is required (init_corr_router).
        uncond_mask = flat_labels == UNCOND_LABEL
        cond_mask = ~uncond_mask

        router_weights = torch.zeros(batch_size * seq_len, self.top_k, device=device)
        expert_indices = torch.zeros(batch_size * seq_len, self.top_k, device=device, dtype=torch.long)
        if uncond_mask.any():
            uncond_positions = torch.where(uncond_mask)[0]
            router_weights[uncond_positions, 0] = 1.0
            expert_indices[uncond_positions] = self.num_experts - 1

        delta = None
        if cond_mask.any():
            cond_positions = torch.where(cond_mask)[0]
            # The conference cosines, operation for operation.
            input_norm = F.normalize(flat_input[cond_positions], p=2, dim=1)
            cluster_norm = F.normalize(self.cluster_centers, p=2, dim=1)
            cos_sim = input_norm @ cluster_norm.T
            delta = self._correction(cos_sim, cond_positions, seq_len, context)
            topk_scores, topk_idx = torch.topk(cos_sim + delta.to(cos_sim.dtype), k=self.top_k, dim=1)
            router_weights[cond_positions] = topk_scores.to(router_weights.dtype)
            expert_indices[cond_positions] = topk_idx
            if self._corr_record:
                self._record_correction(cos_sim, delta, topk_scores, topk_idx, cond_positions, seq_len, context)
        else:
            if torch.is_grad_enabled():
                # No prototype-routed token here: keep the prototypes and the
                # corrector in the graph with zero weight, since DDP expects a
                # gradient for every parameter on every rank.
                router_weights = router_weights + 0.0 * (
                    self.cluster_centers.sum()
                    + sum(parameter.sum() for parameter in self.corrector.parameters())
                )
            if self._corr_record:
                self.last_corr_bins = torch.zeros(len(CORR_BIN_ROWS), NUM_SIGMA_BINS, device=device)

        router_weights = router_weights.view(batch_size, seq_len, self.top_k)
        expert_indices = expert_indices.view(batch_size, seq_len, self.top_k)
        router_output = (router_weights, expert_indices, None)
        # The corrections of the prototype-routed tokens, in flat order, for a
        # subclass loss on the corrected scores (the clean-route alignment).
        self._corr_delta = delta
        if getattr(self, "_phase", None) is not None:
            # A clean-route pass reads its exact routing back (see models_ProMoE_TC_clean_route).
            self._router_output = router_output
        return router_output

    def _correction(self, cos_sim, cond_positions, seq_len, context):
        sample_index = torch.div(cond_positions, seq_len, rounding_mode="floor")
        reference_embed = context["reference_embed"].reshape(-1, self.hidden_size)[cond_positions]
        sample_hidden = self.corrector.sample_hidden(context["condition"], context["reference_sigma_embed"])
        return self.corrector(reference_embed, cos_sim.detach(), sample_hidden[sample_index])

    @torch.no_grad()
    def _record_correction(self, cos_sim, delta, topk_scores, topk_idx, cond_positions, seq_len, context):
        device = cos_sim.device
        sample_index = torch.div(cond_positions, seq_len, rounding_mode="floor")
        sigma = context["sample_sigma"].to(device=device, dtype=torch.float32)[sample_index]
        referenced = context["has_reference"].to(device=device)[sample_index].float()
        bins = (sigma * NUM_SIGMA_BINS).long().clamp_(0, NUM_SIGMA_BINS - 1)
        # The same selection call on both sides, so a tie cannot read as a switch.
        uncorrected_top1 = torch.topk(cos_sim, k=self.top_k, dim=1).indices[:, 0]
        switched = (topk_idx[:, 0] != uncorrected_top1).float()
        gate = topk_scores[:, 0].float()
        rows = torch.stack([
            delta.float().abs().mean(dim=1),
            torch.ones_like(gate),
            switched * referenced,
            referenced,
            switched * (1.0 - referenced),
            1.0 - referenced,
            gate,
            (gate < 0).float(),
        ])
        table = torch.zeros(len(CORR_BIN_ROWS), NUM_SIGMA_BINS, device=device)
        table.index_add_(1, bins, rows)
        self.last_corr_bins = table


class SparseMoeBlock(CorrRouterBlockMixin, ConferenceSparseMoeBlock):
    """The conference TC block with the routing correction."""


#################################################################################
#                                 Core ProMoE Model                             #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning; its MoE
    layer is an instance of ``moe_block_class``.
    """
    def __init__(self, hidden_size, num_heads, moe_block_class, head_dim=None, mlp_ratio=4.0,
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
            self.mlp = moe_block_class(hidden_size=hidden_size, **MoE_config)
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
    Diffusion model with a Transformer backbone and the routing correction.
    """
    moe_block_class = SparseMoeBlock

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
        corr_router_config=None,
    ):
        super().__init__()
        if MoE_config.init_MoeMLP:
            raise ValueError("init_MoeMLP is not supported: MoeMLP has no gate_proj")
        if num_classes != UNCOND_LABEL:
            raise ValueError(f"corr_router assumes {UNCOND_LABEL} classes: the dropped label is the uncond label")
        settings = parse_corr_router_config(corr_router_config)
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
            DiTBlock(hidden_size, num_heads, self.moe_block_class, head_dim=head_dim, mlp_ratio=mlp_ratio,
                     qk_norm=qk_norm, use_swiglu=use_swiglu, MoE_config=MoE_config, use_moe=use_moe_flag[i])
            for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        if not self.moe_blocks():
            raise ValueError("corr_router needs at least one MoE block")
        # The correctors come after the conference initialisation, so that one
        # draws exactly what it draws in ProMoE_TC_B.
        generator = torch.Generator().manual_seed(CORRECTOR_INIT_SEED)
        for block in self.moe_blocks():
            block.init_corr_router(settings["hidden_dim"], generator)
        self.skip_intervals = settings["skip_intervals"]
        self.no_reference_prob = settings["no_reference_prob"]
        self.previous_state_mode = settings["previous_state_mode"]
        # [reference images, prototype-routed images, summed sigma_ref - sigma
        # of the reference images] of the latest training step, for train.py.
        self.last_corr_reference = None
        self.is_corr_router_model = True

    def moe_blocks(self):
        """The MoE blocks, in depth order."""
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

    def _no_reference(self, x, sample_sigma=None):
        """The reference of an image without one: x0_hat = 0, sigma_ref = 1."""
        batch = x.shape[0]
        return (
            torch.zeros_like(x),
            torch.ones(batch, device=x.device),
            torch.zeros(batch, dtype=torch.bool, device=x.device),
            sample_sigma,
        )

    def _velocity(self, output, like):
        """The predicted velocity (the first ``in_channels`` output channels), shaped like ``like``."""
        return output[:, : self.in_channels].reshape(like.shape)

    def _corr_context(self, reference, c):
        reference_x0, reference_sigma, has_reference, sample_sigma = reference
        if reference_x0.dim() != 4:
            reference_x0 = reference_x0.squeeze(2)
        # Embedded with gradient and then detached: autocast caches a weight's
        # low-precision copy whatever the grad mode, and a copy cast under
        # no_grad would cut the weight's gradient in every later use.
        reference_embed = (self.x_embedder(reference_x0) + self.pos_embed).detach()
        reference_sigma_embed = self.t_embedder(reference_sigma * NUM_TRAIN_TIMESTEPS).detach()
        return {
            "reference_embed": reference_embed,
            "condition": c.detach(),
            "reference_sigma_embed": reference_sigma_embed,
            "has_reference": has_reference,
            "sample_sigma": sample_sigma,
        }

    def _run(self, x, timestep, y, labels, reference=None, record=False, final_layer=True):
        """The conference forward after the label embedding.

        ``reference`` = (x0_hat, sigma_ref, has_reference, sample_sigma) feeds
        every corrector; ``None`` routes without correction.  ``record`` keeps
        the correction diagnostics of this pass.
        """
        if len(x.shape) != 4:
            x = x.squeeze(2)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)                   # (N, D)
        c = t + y                                # (N, D)
        blocks = self.moe_blocks()
        if reference is not None:
            context = self._corr_context(reference, c)
            for block in blocks:
                block.set_corr_context(context, record=record)
        try:
            for block in self.blocks:
                x = block(x, c, labels)                 # (N, T, D)
        finally:
            if reference is not None:
                for block in blocks:
                    block.set_corr_context(None)
        if not final_layer:
            return None
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def _training_reference(self, x, timestep, sigma, y, labels, generator, *, z=None, noise=None):
        """The reference of every image of a training batch (see the module docstring)."""
        batch = x.shape[0]
        device = x.device
        if not torch.allclose(timestep.float(), sigma * NUM_TRAIN_TIMESTEPS, rtol=1e-4, atol=1e-3):
            raise ValueError(f"corr_router assumes timestep = {NUM_TRAIN_TIMESTEPS} * sigma")
        if self.previous_state_mode == "same_noise":
            for name, value in (("clean latent", z), ("noise", noise)):
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"corr_router same_noise training requires the original {name} tensor")
                if value.shape != x.shape or value.device != x.device:
                    raise ValueError(f"corr_router same_noise {name} must match x's shape and device")
        routed = labels != UNCOND_LABEL
        # One coin and one gap per image, routed or not, so the draws do not
        # depend on the label dropout.
        coin = torch.rand(batch, generator=generator, device=generator.device).to(device)
        gap_index = torch.randint(
            len(self.skip_intervals), (batch,), generator=generator, device=generator.device
        ).to(device)
        gaps = torch.tensor(self.skip_intervals, dtype=torch.float32, device=device)[gap_index] / NUM_TRAIN_TIMESTEPS
        use_reference = routed & (coin >= self.no_reference_prob)

        reference_x0 = torch.zeros_like(x)
        reference_sigma = torch.ones(batch, device=device)
        if bool(use_reference.any()):
            index = torch.where(use_reference)[0]
            current_sigma = sigma[index]
            previous_sigma = (current_sigma + gaps[index]).clamp(max=1.0)
            shape = (-1,) + (1,) * (x.dim() - 1)
            with torch.no_grad():
                if self.previous_state_mode == "same_noise":
                    previous = (
                        (1.0 - previous_sigma).view(shape) * z[index]
                        + previous_sigma.view(shape) * noise[index]
                    )
                else:
                    current = x[index]
                    velocity = self._velocity(
                        self._run(current, timestep[index], y[index], labels[index], self._no_reference(current)),
                        current,
                    )
                    previous = current + (previous_sigma - current_sigma).view(shape) * velocity
                previous_velocity = self._velocity(
                    self._run(
                        previous,
                        previous_sigma * NUM_TRAIN_TIMESTEPS,
                        y[index],
                        labels[index],
                        self._no_reference(previous),
                    ),
                    previous,
                )
                reference_x0[index] = (previous - previous_sigma.view(shape) * previous_velocity).to(reference_x0.dtype)
            reference_sigma[index] = previous_sigma
        self.last_corr_reference = torch.stack([
            use_reference.sum().float(),
            routed.sum().float(),
            (reference_sigma - sigma)[use_reference].sum(),
        ]).detach()
        return reference_x0, reference_sigma, use_reference, sigma

    def _eval_reference(self, x, reference_x0, reference_sigma):
        """Sampling: the previous step's estimate from sample.py, or none at the first step."""
        if reference_x0 is None and reference_sigma is None:
            return self._no_reference(x)
        if reference_x0 is None or reference_sigma is None:
            raise ValueError("corr_reference_x0 and corr_reference_sigma go together")
        batch = x.shape[0]
        reference_sigma = reference_sigma.reshape(-1).to(device=x.device, dtype=torch.float32)
        if reference_sigma.numel() != batch or reference_x0.reshape(batch, -1).shape != x.reshape(batch, -1).shape:
            raise ValueError("corr_reference_x0 / corr_reference_sigma do not match the batch")
        return reference_x0, reference_sigma, torch.ones(batch, dtype=torch.bool, device=x.device), None

    def forward(self, x, timestep, context, corr_router_sigma=None, corr_router_generator=None,
                corr_reference_x0=None, corr_reference_sigma=None, corr_router_z=None, corr_router_noise=None,
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        timestep: (N,) tensor of diffusion timesteps
        context: (N,) tensor of class labels
        corr_router_sigma / corr_router_generator: (N, ...) noise level of ``x`` and the
            generator of the reference draws; required in training (train.py passes them).
        corr_router_z / corr_router_noise: the original clean latent and noise used to
            construct ``x``; required in training only for ``previous_state_mode: same_noise``.
        corr_reference_x0 / corr_reference_sigma: eval only, the previous sampling step's
            estimate of the clean latent and its noise level; omitted at the first step.
        """
        if not self.training:
            y, labels = self.y_embedder(context, False)
            reference = self._eval_reference(x, corr_reference_x0, corr_reference_sigma)
            return self._run(x, timestep, y, labels, reference)
        if corr_router_sigma is None or corr_router_generator is None:
            raise ValueError(
                "corr_router training needs corr_router_sigma and corr_router_generator "
                "(train.py passes them for CORR_ROUTER_MODELS)"
            )
        if corr_reference_x0 is not None or corr_reference_sigma is not None:
            raise ValueError("corr_router builds its own training reference; corr_reference_* is for sampling")

        y, labels = self.y_embedder(context, True)
        sigma = corr_router_sigma.reshape(-1).float()
        reference = self._training_reference(
            x, timestep, sigma, y, labels, corr_router_generator, z=corr_router_z, noise=corr_router_noise
        )
        # The no-grad passes above left history-free weight casts in the
        # autocast cache; drop them so this pass casts afresh (see _corr_context).
        torch.clear_autocast_cache()
        return self._run(x, timestep, y, labels, reference, record=True)

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
    "CORR_BIN_ROWS",
    "CorrRouterBlockMixin",
    "DiT",
    "DiTBlock",
    "RoutingCorrector",
    "SparseMoeBlock",
    "parse_corr_router_config",
    "sampling_reference",
]
