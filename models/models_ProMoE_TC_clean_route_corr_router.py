"""Clean-route routing distillation with the routing correction of corr_router.

The model is the clean_route model (models_ProMoE_TC_clean_route.py) whose MoE
blocks also carry the corr_router corrector (models_ProMoE_TC_corr_router.py):

* the teacher pass routes without the correction, so the target
  ``q = softmax(cos(h_T, P) / tau)`` is the clean-route one.  With the
  corrector on both sides of the KL, sending every token to one expert would
  make the two sides agree trivially, and the routing-contrastive loss (zero
  below two non-empty clusters) could not stop that;
* the student routes, weights its experts and groups its contrastive clusters
  by the corrected scores ``s + delta``, and its alignment term uses
  ``p = softmax((cos(h_S, P) + delta) / tau)``, so the KL trains the corrector
  directly, for every expert;
* a training step runs the corr_router reference passes, then the teacher
  pass, then the student pass. With ``previous_state_mode: same_noise``, the
  reference reuses ``clean_route_z`` and ``clean_route_noise`` and needs one
  auxiliary pass; the teacher pass is unchanged.

The teacher is the clean latents at t = 0 (``teacher_sigma_ratio: 0``): how a
noisy teacher should use the correction is left open (user decision,
2026-10-22), so other ratios are refused.  Eval and sampling route with the
correction, fed the reference that sample.py carries over from the previous
step.  The corrector outputs zero at step 0, so the first training step of a
fresh model matches clean_route with the same ``clean_route_config`` bit for
bit (and so ``ProMoE_TC_B`` when ``align_lam`` is 0 and ``contrastive_source``
is ``student``); from then on the corrector learns.
"""

import torch

from .models_ProMoE_TC_clean_route import SparseMoeBlock as CleanRouteSparseMoeBlock
from .models_ProMoE_TC_clean_route import UNCOND_LABEL, clean_route_align_scale, parse_clean_route_config
from .models_ProMoE_TC_corr_router import CorrRouterBlockMixin
from .models_ProMoE_TC_corr_router import DiT as CorrRouterDiT


#################################################################################
#                                 ProMoE Layer                                  #
#################################################################################
class SparseMoeBlock(CorrRouterBlockMixin, CleanRouteSparseMoeBlock):
    """The clean-route block with the routing correction.

    The teacher pass runs without a corrector context, so it routes and
    records exactly as the clean-route block does.  The student pass routes by
    ``s + delta`` (CorrRouterBlockMixin) and aligns the corrected scores.
    """

    def _student_losses(self, conference_loss, indices, weights, cosines, routed, seq_len):
        delta = self._corr_delta
        self._corr_delta = None
        if cosines.shape[0] > 0:
            if delta is None or tuple(delta.shape) != tuple(cosines.shape):
                raise RuntimeError("clean_route student pass ran without the routing correction")
            # The float32 alignment cosines plus the correction: the student's
            # distribution is that of the scores it actually routes by.
            cosines = cosines + delta.float()
        return super()._student_losses(conference_loss, indices, weights, cosines, routed, seq_len)


#################################################################################
#                                 Core ProMoE Model                             #
#################################################################################
class DiT(CorrRouterDiT):
    """
    Diffusion model with a Transformer backbone, clean-route routing distillation
    and the routing correction.
    """
    moe_block_class = SparseMoeBlock

    def __init__(self, *args, clean_route_config=None, **kwargs):
        settings = parse_clean_route_config(clean_route_config)
        if settings["teacher_sigma_ratio"] != 0.0:
            raise ValueError(
                "clean_route_corr_router supports only teacher_sigma_ratio 0: how a noisy teacher "
                "should use the routing correction is undecided"
            )
        super().__init__(*args, **kwargs)
        self.teacher_sigma_ratio = settings["teacher_sigma_ratio"]
        self.align_start_step = settings["align_start_step"]
        self.align_warmup_steps = settings["align_warmup_steps"]
        for block in self.moe_blocks():
            block.configure_clean_route(
                settings["contrastive_source"], settings["align_lam"], settings["align_temperature"]
            )
        self.is_clean_route_model = True

    def _set_phase(self, phase, sample_sigma=None):
        for block in self.moe_blocks():
            block.set_phase(phase, sample_sigma=sample_sigma)

    def forward(self, x, timestep, context, clean_route_z=None, clean_route_noise=None,
                clean_route_sigma=None, corr_router_sigma=None, corr_router_generator=None,
                corr_reference_x0=None, corr_reference_sigma=None, training_step=None, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        timestep: (N,) tensor of diffusion timesteps
        context: (N,) tensor of class labels
        clean_route_z / clean_route_noise / clean_route_sigma: the clean-route teacher inputs,
            and corr_router_sigma / corr_router_generator the reference inputs (the same
            sigma); all required in training (train.py passes them).
            The same clean latent and noise also construct the previous reference state
            when ``corr_router_config.previous_state_mode`` is ``same_noise``.
        corr_reference_x0 / corr_reference_sigma: eval only, as in models_ProMoE_TC_corr_router.
        training_step: absolute optimizer step for the optional clean-route KL schedule.
        """
        if not self.training:
            return super().forward(
                x, timestep, context,
                corr_reference_x0=corr_reference_x0, corr_reference_sigma=corr_reference_sigma,
            )
        required = (clean_route_z, clean_route_noise, clean_route_sigma, corr_router_sigma, corr_router_generator)
        if any(value is None for value in required):
            raise ValueError(
                "clean_route_corr_router training needs clean_route_z, clean_route_noise, clean_route_sigma, "
                "corr_router_sigma and corr_router_generator (train.py passes them)"
            )
        if corr_reference_x0 is not None or corr_reference_sigma is not None:
            raise ValueError("corr_router builds its own training reference; corr_reference_* is for sampling")
        sigma = clean_route_sigma.reshape(-1).float()
        if not torch.equal(corr_router_sigma.reshape(-1).float(), sigma):
            raise ValueError("clean_route_sigma and corr_router_sigma must be the same noise levels")

        align_scale = clean_route_align_scale(training_step, self.align_start_step, self.align_warmup_steps)
        for block in self.moe_blocks():
            block.align_scale = align_scale

        # Draw the class-label dropout once: every pass sees the same labels.
        y, labels = self.y_embedder(context, True)
        routed_images = labels != UNCOND_LABEL  # use_uncond_expert is required
        try:
            # The reference passes run with no phase set: a teacher record set
            # before them would be dropped by their phase changes.
            reference = self._training_reference(
                x, timestep, sigma, y, labels, corr_router_generator, z=clean_route_z, noise=clean_route_noise
            )
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
                    # No reference: the teacher routes by the uncorrected cosines.
                    self._run(teacher_x, teacher_t, y[routed_images], labels[routed_images], final_layer=False)
            # The no-grad passes above left history-free weight casts in the
            # autocast cache; drop them so the student casts afresh.
            torch.clear_autocast_cache()
            self._set_phase("student", sample_sigma=sigma)
            return self._run(x, timestep, y, labels, reference, record=True)
        finally:
            self._set_phase(None)


__all__ = ["DiT", "SparseMoeBlock"]
