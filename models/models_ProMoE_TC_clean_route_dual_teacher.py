"""Raw-router KL from real-clean and model-estimated-clean teachers.

There is no corrector and no reference dropout. All conditional images get a
detached estimate using the parent's two model-velocity reference predictions.
The real and estimated teachers share model weights, run at t=0 without
gradients, and supervise the same student's raw router. Sampling is ordinary TC.
"""
import math

import torch
import torch.nn.functional as F

from .models_ProMoE_TC_clean_route import DiT as CleanRouteDiT
from .models_ProMoE_TC_clean_route import SparseMoeBlock as CleanRouteBlock
from .models_ProMoE_TC_clean_route import clean_route_align_scale, UNCOND_LABEL
from .models_ProMoE_TC_corr_router import draw_reference_plan, estimate_clean_latent, NUM_TRAIN_TIMESTEPS


class SparseMoeBlock(CleanRouteBlock):
    estimated_fraction = 0.5
    last_real_kl = None
    last_estimated_kl = None

    def _student_losses(self, conference_loss, indices, weights, cosines, routed, seq_len):
        loss = super()._student_losses(conference_loss, indices, weights, cosines, routed, seq_len)
        if cosines.shape[0] == 0:
            # Even a rank with no conditional image must connect its router
            # parameters to the distributed backward (with zero local loss).
            anchor = cosines.sum() * 0.0
            return anchor if loss is None else loss + anchor
        return loss

    def _record_teacher(self, tokens, indices, cosines, seq_len):
        # One batched teacher pass: real images first, estimated images second.
        if cosines.shape[0] % (2 * seq_len):
            raise RuntimeError("dual teacher pass needs equal real/estimated image batches")
        half = cosines.shape[0] // 2
        super()._record_teacher(tokens[:half], indices[:half], cosines[:half], seq_len)
        self._teacher["estimated_log_q"] = F.log_softmax(cosines[half:] / self.align_temperature, dim=1)

    def _alignment_kl(self, log_p, teacher):
        real_kl = super()._alignment_kl(log_p, teacher)
        estimated_log_q = teacher["estimated_log_q"]
        if estimated_log_q.shape != log_p.shape:
            raise RuntimeError("estimated teacher and student routing rows are not aligned")
        estimated_kl = (estimated_log_q.exp() * (estimated_log_q - log_p)).sum(dim=1)
        self.last_real_kl = real_kl.detach().mean()
        self.last_estimated_kl = estimated_kl.detach().mean()
        return (1.0 - self.estimated_fraction) * real_kl + self.estimated_fraction * estimated_kl

    def _record_empty_student(self, zero, conference_loss):
        super()._record_empty_student(zero, conference_loss)
        self.last_real_kl = zero
        self.last_estimated_kl = zero


class DiT(CleanRouteDiT):
    moe_block_class = SparseMoeBlock

    def __init__(self, *args, clean_route_config=None, reference_config=None, **kwargs):
        route = {} if clean_route_config is None else dict(clean_route_config)
        estimated_lam = route.pop("estimated_align_lam", None)
        if estimated_lam is None or not math.isfinite(float(estimated_lam)) or float(estimated_lam) < 0:
            raise ValueError("clean_route_config.estimated_align_lam must be finite and >= 0")
        if reference_config is None or set(reference_config) != {"skip_intervals"}:
            raise ValueError("dual teacher reference_config requires only skip_intervals")
        intervals = reference_config["skip_intervals"]
        if (not isinstance(intervals, (list, tuple)) or not intervals
                or any(type(k) is not int or not 1 <= k < NUM_TRAIN_TIMESTEPS for k in intervals)
                or len(set(intervals)) != len(intervals)):
            raise ValueError("reference_config.skip_intervals must be distinct integers in [1, 1000)")
        super().__init__(*args, clean_route_config=route, **kwargs)
        if self.teacher_sigma_ratio != 0 or any(
            b.contrastive_source != "student" or not b.use_uncond_expert for b in self.moe_blocks()
        ):
            raise ValueError("dual teacher requires clean t=0 teachers, student contrastive and uncond expert")
        self.skip_intervals = tuple(intervals)
        for block in self.moe_blocks():
            total = block.align_lam + float(estimated_lam)
            block.estimated_fraction = float(estimated_lam) / total if total else 0.0
            block.align_lam = total
        self.is_dual_route_model = True

    def _reference_velocity(self, x, timestep, y, labels):
        prediction = self._run(x, timestep, y, labels)
        return prediction[:, :self.in_channels].reshape(x.shape)

    def forward(self, x, timestep, context, clean_route_z=None, clean_route_noise=None,
                clean_route_sigma=None, corr_router_generator=None, training_step=None, **kwargs):
        if not self.training:
            return super().forward(x, timestep, context)
        if any(v is None for v in (clean_route_z, clean_route_noise, clean_route_sigma, corr_router_generator)):
            raise ValueError("dual teacher training requires clean-route inputs and corr_router_generator")
        sigma = clean_route_sigma.reshape(-1).float()
        if not torch.allclose(timestep.float(), sigma * NUM_TRAIN_TIMESTEPS, rtol=1e-4, atol=1e-3):
            raise ValueError("dual teacher assumes timestep = 1000 * sigma")
        scale = clean_route_align_scale(training_step, self.align_start_step, self.align_warmup_steps)
        for block in self.moe_blocks():
            block.align_scale = scale
        y, labels = self.y_embedder(context, True)
        routed, reference_sigma = draw_reference_plan(sigma, labels, self.skip_intervals, 0.0, corr_router_generator)
        try:
            if bool(routed.any()):
                estimate = estimate_clean_latent(
                    self._reference_velocity, x[routed], timestep[routed], sigma[routed], reference_sigma[routed],
                    y[routed], labels[routed],
                ).to(x.dtype)
                with torch.no_grad():
                    self._set_phase("teacher")
                    self._run(
                        torch.cat((clean_route_z[routed], estimate)),
                        torch.zeros_like(timestep[routed]).repeat(2),
                        y[routed].repeat(2, 1), labels[routed].repeat(2), final_layer=False,
                    )
            torch.clear_autocast_cache()
            self._set_phase("student", sample_sigma=sigma)
            return self._run(x, timestep, y, labels)
        finally:
            self._set_phase(None)


__all__ = ["DiT", "SparseMoeBlock"]
