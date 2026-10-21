"""ProMoE-TC whose noisy-input routing is distilled from the routing of the
same images at lower noise ("clean route").

Every training step runs the network twice on the same batch:

1. a teacher pass, without gradient, on the same latents at noise
   ``sigma_a = teacher_sigma_ratio * sigma`` with the student's own noise
   sample -- ratio 0 feeds the clean latents at t = 0;
2. the normal student pass at the sampled ``sigma``.

In every MoE block the student's routing distribution over the prototypes,
``p = softmax(cos(h_S, P) / tau)``, is pulled toward the teacher's
``q = softmax(cos(h_T, P) / tau)`` by ``align_lam * KL(q || p)`` (soft labels;
``q`` is a fixed target).  ``P`` is the block's ``cluster_centers``: both passes
use the same ``P`` and it is never detached, so it receives the alignment
gradient through the student's cosines.  ``contrastive_source`` chooses the
tokens of the conference routing-contrastive loss: ``student`` keeps it on the
student's tokens and assignments (conference behaviour); ``teacher`` computes
it on the teacher's tokens and assignments instead.  The teacher's tokens carry
no gradient, so that loss updates only ``P``.

Class-label dropout is drawn once and shared by both passes, and the teacher
pass runs only on the images whose tokens are prototype-routed, so the i-th
prototype-routed token of the two passes is the same position of the same
image.

Evaluation and sampling (eval mode) run the conference forward unchanged.
With ``align_lam: 0`` and ``contrastive_source: student`` a training step is
bit-identical to the conference model, also under bf16 autocast: the teacher
pass draws no random numbers, creates no gradient, and its weight casts are
dropped from the autocast cache before the student pass.  Only the MoE block inherits from the
conference model; the parameters, and so the checkpoints, are the conference
model's.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from .models_ProMoE_TC import AddAuxiliaryLoss
from .models_ProMoE_TC import SparseMoeBlock as ConferenceSparseMoeBlock
from .modules import get_2d_sincos_pos_embed, Attention, modulate, TimestepEmbedder, LabelEmbedder, FinalLayer, MoeMLP, Mlp


CLEAN_ROUTE_CONFIG_KEYS = (
    "contrastive_source",
    "teacher_sigma_ratio",
    "align_lam",
    "align_temperature",
)
CONTRASTIVE_SOURCES = ("student", "teacher")
# The conference block sends the dropped class label to the unconditional expert.
UNCOND_LABEL = 1000
# Diagnostics bin the student's sigma into this many equal-width bins on [0, 1].
NUM_SIGMA_BINS = 5


def parse_clean_route_config(clean_route_config):
    """Validate the four clean-route settings; every key is required."""
    if clean_route_config is None:
        raise ValueError(
            "ProMoE_TC_B_clean_route needs DiT_*_config.clean_route_config with keys "
            + ", ".join(CLEAN_ROUTE_CONFIG_KEYS)
        )
    config = dict(clean_route_config)
    missing = [key for key in CLEAN_ROUTE_CONFIG_KEYS if key not in config]
    unknown = sorted(set(config) - set(CLEAN_ROUTE_CONFIG_KEYS))
    if missing or unknown:
        raise ValueError(f"clean_route_config: missing keys {missing}, unknown keys {unknown}")

    contrastive_source = str(config["contrastive_source"])
    if contrastive_source not in CONTRASTIVE_SOURCES:
        raise ValueError(f"clean_route_config.contrastive_source must be one of {CONTRASTIVE_SOURCES}")
    teacher_sigma_ratio = float(config["teacher_sigma_ratio"])
    if not 0.0 <= teacher_sigma_ratio < 1.0:
        raise ValueError("clean_route_config.teacher_sigma_ratio must lie in [0, 1)")
    align_lam = float(config["align_lam"])
    if not (math.isfinite(align_lam) and align_lam >= 0.0):
        raise ValueError("clean_route_config.align_lam must be a finite value >= 0")
    align_temperature = float(config["align_temperature"])
    if not (math.isfinite(align_temperature) and align_temperature > 0.0):
        raise ValueError("clean_route_config.align_temperature must be a finite value > 0")
    return {
        "contrastive_source": contrastive_source,
        "teacher_sigma_ratio": teacher_sigma_ratio,
        "align_lam": align_lam,
        "align_temperature": align_temperature,
    }


#################################################################################
#                                 ProMoE Layer                                  #
#################################################################################
class SparseMoeBlock(ConferenceSparseMoeBlock):
    """Conference TC block with a teacher phase and a student phase.

    The parent model sets ``_phase`` around its two passes.  ``None`` (eval,
    sampling) is the conference block.  In the teacher pass the block records
    the teacher's routing of its prototype-routed tokens and returns no loss.
    In the student pass it adds the alignment loss and, for
    ``contrastive_source: teacher``, moves the routing-contrastive loss onto the
    recorded teacher tokens.  ``last_*`` attributes hold detached diagnostics of
    the latest student pass for ``train.py``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.phase_metric is not None:
            raise ValueError("clean_route does not support phase_metric_config.enabled")
        if self.alpha > 0:
            raise ValueError("clean_route does not support load_balance_loss_coef > 0")
        self.contrastive_source = "student"
        self.align_lam = 0.0
        self.align_temperature = 1.0
        # Analysis only: keep the float32 cosines of both passes.
        self.keep_cosines = False
        self._phase = None
        self._sample_sigma = None
        self._teacher = None
        self._router_output = None
        self.reset_clean_route_stats()

    def configure_clean_route(self, contrastive_source, align_lam, align_temperature):
        self.contrastive_source = contrastive_source
        self.align_lam = float(align_lam)
        self.align_temperature = float(align_temperature)

    def set_phase(self, phase, sample_sigma=None):
        """Enter ``teacher``/``student``, or return to the conference block (``None``)."""
        if phase not in (None, "teacher", "student"):
            raise ValueError(f"unknown clean_route phase {phase!r}")
        self._phase = phase
        self._sample_sigma = sample_sigma
        self._router_output = None
        if phase != "student":
            # A student pass consumes the record of the teacher pass before it;
            # any other transition starts without one.
            self._teacher = None

    def reset_clean_route_stats(self):
        self.last_kl = None
        self.last_align_loss = None
        self.last_contrastive_loss = None
        self.last_agreement = None
        self.last_student_entropy = None
        self.last_teacher_entropy = None
        self.last_gate_mean = None
        self.last_student_load = None
        self.last_teacher_load = None
        self.last_student_image_dominance = None
        self.last_teacher_image_dominance = None
        self.last_sigma_bins = None
        self.last_student_cosines = None
        self.last_teacher_cosines = None
        self.last_token_sigma = None

    def compute_router(self, hidden_states, labels, timestep=None):
        router_output = super().compute_router(hidden_states, labels, timestep)
        if self._phase is not None:
            # The conference forward does not return its routing; keep it so
            # the diagnostics and the teacher assignments use the exact routing.
            self._router_output = router_output
        return router_output

    def compute_routing_contrastive_loss(self, token_embeddings, cluster_assignments, use_top_k=False):
        # The conference forward calls this on the current pass's own tokens.
        # The teacher pass needs no loss, and contrastive_source "teacher"
        # replaces it by the loss on the teacher's tokens (_student_losses).
        if self._phase == "teacher" or (self._phase == "student" and self.contrastive_source == "teacher"):
            return torch.zeros((), device=self.cluster_centers.device)
        return super().compute_routing_contrastive_loss(token_embeddings, cluster_assignments, use_top_k=use_top_k)

    def forward(self, hidden_states, labels, timestep=None):
        if self._phase is None:
            return super().forward(hidden_states, labels, timestep)

        final_output, conference_loss = super().forward(hidden_states, labels, timestep)
        router_weights, expert_indices, _ = self._router_output
        self._router_output = None

        batch_size, seq_len, _ = hidden_states.shape
        routed = self._prototype_routed_tokens(labels, batch_size, seq_len)
        tokens = hidden_states.reshape(-1, self.hidden_size)[routed]
        indices = expert_indices.reshape(-1, self.top_k)[routed]
        weights = router_weights.reshape(-1, self.top_k)[routed]
        cosines = self._alignment_cosines(tokens)

        if self._phase == "teacher":
            self._record_teacher(tokens, indices, cosines, seq_len)
            return final_output, None
        loss = self._student_losses(conference_loss, indices, weights, cosines, routed, seq_len)
        return final_output, loss

    def _prototype_routed_tokens(self, labels, batch_size, seq_len):
        """Flat mask of the tokens the prototypes route (the conference cond mask)."""
        flat_labels = labels.view(batch_size, 1).expand(-1, seq_len).reshape(-1)
        if self.use_uncond_expert:
            return flat_labels != UNCOND_LABEL
        return torch.ones_like(flat_labels, dtype=torch.bool)

    def _alignment_cosines(self, tokens):
        """Token-prototype cosines in float32, outside autocast."""
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            token_norm = F.normalize(tokens.float(), p=2, dim=1)
            center_norm = F.normalize(self.cluster_centers.float(), p=2, dim=1)
            return token_norm @ center_norm.T

    def _image_dominance(self, top1, seq_len):
        """Mean share of an image's tokens that its most used expert receives."""
        if top1.numel() == 0:
            return torch.zeros((), device=top1.device)
        counts = F.one_hot(top1, num_classes=self.num_routed_experts).view(-1, seq_len, self.num_routed_experts)
        return (counts.sum(dim=1).max(dim=1).values.float() / seq_len).mean()

    def _load(self, top1):
        return torch.bincount(top1, minlength=self.num_routed_experts).float()

    def _record_teacher(self, tokens, indices, cosines, seq_len):
        log_q = F.log_softmax(cosines / self.align_temperature, dim=1)
        self._teacher = {
            "log_q": log_q,
            "indices": indices,
            # Only contrastive_source "teacher" needs the tokens themselves.
            "tokens": tokens if self.contrastive_source == "teacher" else None,
            "cosines": cosines if self.keep_cosines else None,
        }
        top1 = indices[:, 0]
        self.last_teacher_entropy = (
            -(log_q.exp() * log_q).sum(dim=1).mean() if top1.numel() > 0 else torch.zeros((), device=top1.device)
        )
        self.last_teacher_load = self._load(top1)
        self.last_teacher_image_dominance = self._image_dominance(top1, seq_len)

    def _student_losses(self, conference_loss, indices, weights, cosines, routed, seq_len):
        teacher = self._teacher
        self._teacher = None
        device = cosines.device
        num_tokens = cosines.shape[0]
        zero = torch.zeros((), device=device)

        if num_tokens == 0:
            # No prototype-routed token (every image dropped): nothing to align.
            self._record_empty_student(zero, conference_loss)
            return conference_loss
        if teacher is None:
            raise RuntimeError("clean_route student pass ran without a teacher pass")
        log_q = teacher["log_q"]
        if log_q.shape != cosines.shape:
            raise RuntimeError(
                f"clean_route teacher records {tuple(log_q.shape)} routing rows but the student has "
                f"{tuple(cosines.shape)}; the passes are not aligned"
            )

        log_p = F.log_softmax(cosines / self.align_temperature, dim=1)
        kl_per_token = (log_q.exp() * (log_q - log_p)).sum(dim=1)
        kl = kl_per_token.mean()
        align_loss = self.align_lam * kl

        if self.contrastive_source == "teacher":
            contrastive_loss = zero
            if self.training and self.routing_contrastive_lam > 0:
                teacher_indices = teacher["indices"]
                assignments = teacher_indices if self.use_top_k_for_routing_contrastive else teacher_indices[:, 0]
                contrastive_loss = ConferenceSparseMoeBlock.compute_routing_contrastive_loss(
                    self,
                    teacher["tokens"],
                    assignments,
                    use_top_k=self.use_top_k_for_routing_contrastive,
                ) * self.routing_contrastive_lam
            loss = contrastive_loss
        else:
            contrastive_loss = conference_loss
            loss = conference_loss
        if self.align_lam > 0:
            loss = align_loss if loss is None else loss + align_loss

        with torch.no_grad():
            top1 = indices[:, 0]
            teacher_top1 = teacher["indices"][:, 0]
            agreement = (top1 == teacher_top1).float()
            self.last_kl = kl.detach()
            self.last_align_loss = align_loss.detach()
            self.last_contrastive_loss = (
                contrastive_loss.detach().float().reshape(()) if torch.is_tensor(contrastive_loss) else zero
            )
            self.last_agreement = agreement.mean()
            self.last_student_entropy = -(log_p.exp() * log_p).sum(dim=1).mean()
            self.last_gate_mean = weights[:, 0].float().mean()
            self.last_student_load = self._load(top1)
            self.last_student_image_dominance = self._image_dominance(top1, seq_len)
            token_sigma = self._token_sigma(routed, seq_len, device)
            bins = (token_sigma * NUM_SIGMA_BINS).long().clamp_(0, NUM_SIGMA_BINS - 1)
            sigma_bins = torch.zeros(3, NUM_SIGMA_BINS, device=device)
            sigma_bins[0].index_add_(0, bins, kl_per_token.detach())
            sigma_bins[1].index_add_(0, bins, agreement)
            sigma_bins[2].index_add_(0, bins, torch.ones_like(agreement))
            self.last_sigma_bins = sigma_bins
            if self.keep_cosines:
                self.last_student_cosines = cosines.detach()
                self.last_teacher_cosines = teacher["cosines"]
                self.last_token_sigma = token_sigma
        return loss

    def _token_sigma(self, routed, seq_len, device):
        if self._sample_sigma is None:
            raise RuntimeError("clean_route student pass needs the per-sample sigma")
        return self._sample_sigma.to(device=device, dtype=torch.float32).repeat_interleave(seq_len)[routed]

    def _record_empty_student(self, zero, conference_loss):
        self.last_kl = zero
        self.last_align_loss = zero
        self.last_contrastive_loss = (
            conference_loss.detach().float().reshape(()) if torch.is_tensor(conference_loss) else zero
        )
        self.last_agreement = zero
        self.last_student_entropy = zero
        self.last_teacher_entropy = zero
        self.last_gate_mean = zero
        self.last_student_load = zero.new_zeros(self.num_routed_experts)
        self.last_teacher_load = zero.new_zeros(self.num_routed_experts)
        self.last_student_image_dominance = zero
        self.last_teacher_image_dominance = zero
        self.last_sigma_bins = zero.new_zeros(3, NUM_SIGMA_BINS)


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
    Diffusion model with a Transformer backbone and clean-route routing distillation.
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
        clean_route_config=None,
    ):
        super().__init__()
        if MoE_config.init_MoeMLP:
            raise ValueError("init_MoeMLP is not supported: MoeMLP has no gate_proj")
        if num_classes != UNCOND_LABEL:
            raise ValueError(f"clean_route assumes {UNCOND_LABEL} classes: the dropped label is the uncond label")
        settings = parse_clean_route_config(clean_route_config)
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

        if not self.moe_blocks():
            raise ValueError("clean_route needs at least one MoE block")
        self.teacher_sigma_ratio = settings["teacher_sigma_ratio"]
        for block in self.moe_blocks():
            block.configure_clean_route(
                settings["contrastive_source"], settings["align_lam"], settings["align_temperature"]
            )
        self.is_clean_route_model = True

    def moe_blocks(self):
        """The MoE blocks, in depth order -- used to set phases and collect diagnostics."""
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

    def _set_phase(self, phase, sample_sigma=None):
        for block in self.moe_blocks():
            block.set_phase(phase, sample_sigma=sample_sigma)

    def _run(self, x, timestep, y, labels, final_layer=True):
        """The conference forward after the label embedding."""
        if len(x.shape) != 4:
            x = x.squeeze(2)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)                   # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c, labels)                     # (N, T, D)
        if not final_layer:
            return None
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward(self, x, timestep, context, clean_route_z=None, clean_route_noise=None,
                clean_route_sigma=None, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        timestep: (N,) tensor of diffusion timesteps
        context: (N,) tensor of class labels
        clean_route_z / clean_route_noise: (N, C, [1,] H, W) clean latents and the noise
            sample that built ``x``; clean_route_sigma: (N, ...) its noise level.
            Required in training, ignored in eval mode.
        """
        if not self.training:
            y, labels = self.y_embedder(context, False)
            return self._run(x, timestep, y, labels)
        if clean_route_z is None or clean_route_noise is None or clean_route_sigma is None:
            raise ValueError(
                "clean_route training needs clean_route_z, clean_route_noise and clean_route_sigma "
                "(train.py passes them for CLEAN_ROUTE_MODELS)"
            )

        # Draw the class-label dropout once: both passes see the same labels.
        y, labels = self.y_embedder(context, True)
        sigma = clean_route_sigma.reshape(-1).float()
        if self.moe_blocks()[0].use_uncond_expert:
            routed_images = labels != UNCOND_LABEL
        else:
            routed_images = torch.ones_like(labels, dtype=torch.bool)
        try:
            if bool(routed_images.any()):
                ratio = self.teacher_sigma_ratio
                teacher_sigma = (ratio * sigma)[routed_images]
                shape = (-1,) + (1,) * (clean_route_z.dim() - 1)
                teacher_x = (
                    (1.0 - teacher_sigma).view(shape) * clean_route_z[routed_images]
                    + teacher_sigma.view(shape) * clean_route_noise[routed_images]
                )
                teacher_t = timestep[routed_images] * ratio
                with torch.no_grad():
                    self._set_phase("teacher")
                    self._run(teacher_x, teacher_t, y[routed_images], labels[routed_images], final_layer=False)
                # Autocast caches the low-precision copy of a weight the first time
                # the weight is cast in a region, without checking grad mode.  The
                # copies cast in the no-grad teacher pass have no autograd history;
                # a student pass reusing them would leave every cast weight without
                # gradient (and DDP with unfinished reductions).  Drop them, so the
                # student casts afresh, exactly like the conference forward.
                torch.clear_autocast_cache()
            self._set_phase("student", sample_sigma=sigma)
            return self._run(x, timestep, y, labels)
        finally:
            self._set_phase(None)

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


__all__ = ["DiT", "DiTBlock", "SparseMoeBlock", "parse_clean_route_config"]
