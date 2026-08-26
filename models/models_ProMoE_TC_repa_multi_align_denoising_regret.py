import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .models_ProMoE_TC_repa_multi_align import (
    DiT as MultiAlignDiT,
    suppress_auxiliary_loss_backward,
)


class DiT(MultiAlignDiT):
    """Multi-Align with sparse first-order denoising-regret routing labels."""

    def __init__(self, *args, repa_config=None, **kwargs):
        super().__init__(*args, repa_config=repa_config, **kwargs)

        repa_config = repa_config or {}
        self.denoising_regret_block = repa_config.get(
            'denoising_regret_block', 3
        )
        self.denoising_regret_probe_interval = repa_config.get(
            'denoising_regret_probe_interval', 16
        )
        self.denoising_regret_token_ratio = repa_config.get(
            'denoising_regret_token_ratio', 0.0625
        )
        self.denoising_regret_candidate_mode = repa_config.get(
            'denoising_regret_candidate_mode', 'mixed'
        )
        self.denoising_regret_confidence_quantile = repa_config.get(
            'denoising_regret_confidence_quantile', 0.5
        )
        self.denoising_regret_temperature = repa_config.get(
            'denoising_regret_temperature', 0.1
        )
        self.denoising_regret_warmup_steps = repa_config.get(
            'denoising_regret_warmup_steps', 10000
        )
        self.denoising_regret_ramp_steps = repa_config.get(
            'denoising_regret_ramp_steps', 10000
        )
        self.denoising_regret_label_roll = repa_config.get(
            'denoising_regret_label_roll', 0
        )
        self.denoising_regret_seed = repa_config.get(
            'denoising_regret_seed', 271828
        )
        self.denoising_regret_eps = repa_config.get(
            'denoising_regret_eps', 1e-6
        )

        block_idx = self.denoising_regret_block
        if (
            isinstance(block_idx, bool)
            or not isinstance(block_idx, int)
            or not 0 <= block_idx < len(self.blocks)
        ):
            raise ValueError(
                f"denoising_regret_block must be an integer in "
                f"[0, {len(self.blocks) - 1}], got {block_idx!r}"
            )
        if not self.blocks[block_idx].use_moe:
            raise ValueError(f"denoising_regret_block {block_idx} must be a MoE block")

        moe_layer = self.blocks[block_idx].mlp
        if moe_layer.num_routed_experts < 2:
            raise ValueError("denoising regret requires at least two routed experts")
        if moe_layer.top_k != 1:
            raise ValueError("denoising regret requires top_k == 1")
        if moe_layer.router_weight_mode != 'identity':
            raise ValueError(
                "denoising regret currently requires router_weight_mode='identity'"
            )
        self._validate_positive_int(
            'denoising_regret_probe_interval',
            self.denoising_regret_probe_interval,
        )
        self._validate_nonnegative_int(
            'denoising_regret_warmup_steps',
            self.denoising_regret_warmup_steps,
        )
        self._validate_nonnegative_int(
            'denoising_regret_ramp_steps',
            self.denoising_regret_ramp_steps,
        )
        self._validate_nonnegative_int(
            'denoising_regret_label_roll',
            self.denoising_regret_label_roll,
        )
        self._validate_nonnegative_int(
            'denoising_regret_seed',
            self.denoising_regret_seed,
        )
        if not 0 < self.denoising_regret_token_ratio <= 1:
            raise ValueError("denoising_regret_token_ratio must be in (0, 1]")
        if self.denoising_regret_candidate_mode not in {
            'runner-up', 'random', 'mixed'
        }:
            raise ValueError(
                "denoising_regret_candidate_mode must be runner-up, random, or mixed"
            )
        if not 0 <= self.denoising_regret_confidence_quantile < 1:
            raise ValueError(
                "denoising_regret_confidence_quantile must be in [0, 1)"
            )
        if self.denoising_regret_temperature <= 0:
            raise ValueError("denoising_regret_temperature must be positive")
        if self.denoising_regret_eps <= 0:
            raise ValueError("denoising_regret_eps must be positive")

        self.denoising_regret_stats = {}
        print(
            f"First-order denoising regret at block {block_idx}: "
            f"interval={self.denoising_regret_probe_interval}, "
            f"token_ratio={self.denoising_regret_token_ratio}, "
            f"candidate={self.denoising_regret_candidate_mode}, "
            f"label_roll={self.denoising_regret_label_roll}"
        )

    @staticmethod
    def _validate_positive_int(name, value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _validate_nonnegative_int(name, value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    def _schedule_scale(self, training_step):
        if training_step < self.denoising_regret_warmup_steps:
            return 0.0
        if self.denoising_regret_ramp_steps == 0:
            return 1.0
        completed = training_step - self.denoising_regret_warmup_steps + 1
        return min(completed / self.denoising_regret_ramp_steps, 1.0)

    def _should_probe(self, training_step):
        return (
            self._schedule_scale(training_step) > 0
            and training_step % self.denoising_regret_probe_interval == 0
        )

    def _make_generator(self, device, training_step):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(
            self.denoising_regret_seed + training_step * world_size + rank
        )
        return generator

    @staticmethod
    def _evaluate_experts(moe_layer, hidden_states, expert_ids):
        outputs = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            for expert_id, expert in enumerate(
                moe_layer.experts[:moe_layer.num_routed_experts]
            ):
                selected = expert_ids == expert_id
                if selected.any():
                    outputs[selected] = expert(hidden_states[selected]).float()
        return outputs

    def _choose_challengers(
        self,
        router_scores,
        current_ids,
        generator,
    ):
        current_mask = F.one_hot(
            current_ids, num_classes=router_scores.shape[-1]
        ).bool()
        runner_up = router_scores.masked_fill(current_mask, -torch.inf).argmax(
            dim=-1
        )

        random_ids = torch.randint(
            router_scores.shape[-1] - 1,
            current_ids.shape,
            generator=generator,
            device=current_ids.device,
        )
        random_ids = random_ids + (random_ids >= current_ids).long()
        if self.denoising_regret_candidate_mode == 'runner-up':
            return runner_up
        if self.denoising_regret_candidate_mode == 'random':
            return random_ids

        probe_slots = torch.arange(
            current_ids.shape[1], device=current_ids.device
        )[None]
        return torch.where(probe_slots % 2 == 0, runner_up, random_ids)

    def _distributed_preference_loss(
        self,
        local_loss_sum,
        local_selected_count,
        local_stats,
    ):
        stats = torch.stack([
            local_selected_count.detach(),
            local_loss_sum.detach(),
            *[value.detach() for value in local_stats],
        ])
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()

        global_selected = stats[0]
        denominator = global_selected.clamp_min(1.0)
        differentiable_mean = local_loss_sum * world_size / denominator
        global_mean = stats[1] / denominator
        loss = differentiable_mean + (
            global_mean - differentiable_mean.detach()
        )
        return loss, stats

    def compute_denoising_regret_loss(
        self,
        prediction,
        denoising_target,
        capture,
        training_step,
    ):
        hidden_states = capture['hidden_states']
        labels = capture['labels']
        moe_output = capture['moe_output']
        moe_layer = self.blocks[self.denoising_regret_block].mlp
        batch_size, num_tokens, hidden_size = hidden_states.shape

        if denoising_target.ndim == 5 and denoising_target.shape[2] == 1:
            denoising_target = denoising_target.squeeze(2)
        model_prediction = prediction[:, :self.in_channels]
        if model_prediction.shape != denoising_target.shape:
            raise ValueError(
                f"denoising target shape {denoising_target.shape} does not match "
                f"prediction shape {model_prediction.shape}"
            )
        diffusion_mse = F.mse_loss(
            model_prediction.float(), denoising_target.detach().float()
        )
        with suppress_auxiliary_loss_backward():
            moe_gradient, = torch.autograd.grad(
                diffusion_mse, moe_output, retain_graph=True
            )

        conditional_rows = torch.where(labels != 1000)[0]
        probe_tokens_per_image = max(
            1, math.ceil(num_tokens * self.denoising_regret_token_ratio)
        )
        if self.denoising_regret_label_roll and probe_tokens_per_image < 2:
            raise ValueError(
                "denoising_regret_label_roll requires at least two probes per image"
            )

        local_zero = moe_layer.cluster_centers.float().sum() * 0.0
        if conditional_rows.numel() == 0:
            local_stats = [
                local_zero.detach(),
                local_zero.detach(),
                local_zero.detach(),
                local_zero.detach(),
                local_zero.detach(),
            ]
            raw_loss, stats = self._distributed_preference_loss(
                local_zero,
                local_zero.detach(),
                local_stats,
            )
            schedule_scale = self._schedule_scale(training_step)
            self._set_regret_stats(
                stats, raw_loss, schedule_scale, probe_tokens_per_image
            )
            return raw_loss * schedule_scale

        generator = self._make_generator(hidden_states.device, training_step)
        random_scores = torch.rand(
            conditional_rows.numel(),
            num_tokens,
            generator=generator,
            device=hidden_states.device,
        )
        token_indices = torch.topk(
            random_scores,
            k=probe_tokens_per_image,
            dim=1,
            sorted=False,
        ).indices
        image_indices = conditional_rows[:, None].expand_as(token_indices)

        with torch.no_grad():
            route_weights, route_indices, _ = moe_layer.compute_router(
                hidden_states.detach(), labels
            )
        current_ids = route_indices[image_indices, token_indices, 0]
        selected_route_weights = route_weights[
            image_indices, token_indices, 0
        ].detach().float()

        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            normalized_hidden = F.normalize(
                hidden_states.detach().float(), p=2, dim=-1
            )
            normalized_centers = F.normalize(
                moe_layer.cluster_centers.float(), p=2, dim=-1
            )
            all_router_scores = normalized_hidden @ normalized_centers.T
            probe_router_scores = all_router_scores[
                image_indices, token_indices
            ]

        challenger_ids = self._choose_challengers(
            probe_router_scores,
            current_ids,
            generator,
        )
        probe_hidden = hidden_states[
            image_indices, token_indices
        ].reshape(-1, hidden_size)
        flat_current_ids = current_ids.reshape(-1)
        flat_challenger_ids = challenger_ids.reshape(-1)
        current_outputs = self._evaluate_experts(
            moe_layer, probe_hidden, flat_current_ids
        ).view_as(probe_hidden).reshape(
            conditional_rows.numel(), probe_tokens_per_image, hidden_size
        )
        challenger_outputs = self._evaluate_experts(
            moe_layer, probe_hidden, flat_challenger_ids
        ).view_as(probe_hidden).reshape_as(current_outputs)

        output_delta = selected_route_weights.unsqueeze(-1) * (
            challenger_outputs - current_outputs
        )
        probe_gradient = moe_gradient[
            image_indices, token_indices
        ].detach().float()
        first_order_change = (probe_gradient * output_delta).sum(dim=-1)
        normalized_change = F.cosine_similarity(
            probe_gradient,
            output_delta,
            dim=-1,
            eps=self.denoising_regret_eps,
        )
        preference_labels = normalized_change.detach()
        if self.denoising_regret_label_roll:
            preference_labels = torch.roll(
                preference_labels,
                shifts=self.denoising_regret_label_roll,
                dims=1,
            )

        absolute_labels = preference_labels.abs()
        confidence_threshold = torch.quantile(
            absolute_labels.flatten(),
            self.denoising_regret_confidence_quantile,
        )
        selected = (
            (absolute_labels >= confidence_threshold)
            & (absolute_labels > self.denoising_regret_eps)
        )
        rows = torch.arange(
            conditional_rows.numel(), device=hidden_states.device
        )[:, None]
        slots = torch.arange(
            probe_tokens_per_image, device=hidden_states.device
        )[None]
        current_scores = probe_router_scores[rows, slots, current_ids]
        challenger_scores = probe_router_scores[rows, slots, challenger_ids]
        preference_margin = (
            current_scores - challenger_scores
        ) / self.denoising_regret_temperature
        current_is_better = (preference_labels > 0).float()
        pair_losses = F.binary_cross_entropy_with_logits(
            preference_margin,
            current_is_better,
            reduction='none',
        )
        local_loss_sum = (pair_losses * selected).sum()
        local_selected_count = selected.sum().float()
        local_probed_count = torch.tensor(
            preference_labels.numel(),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        local_stats = [
            local_probed_count,
            (preference_labels < 0).sum().float(),
            absolute_labels.sum(),
            (current_scores - challenger_scores).detach().sum(),
            first_order_change.detach().abs().sum(),
        ]
        raw_loss, stats = self._distributed_preference_loss(
            local_loss_sum,
            local_selected_count,
            local_stats,
        )
        schedule_scale = self._schedule_scale(training_step)
        self._set_regret_stats(
            stats, raw_loss, schedule_scale, probe_tokens_per_image
        )
        return raw_loss * schedule_scale

    def _set_regret_stats(
        self,
        stats,
        raw_loss,
        schedule_scale,
        probe_tokens_per_image,
    ):
        selected_count = stats[0]
        probed_count = stats[2]
        denominator = probed_count.clamp_min(1.0)
        self.denoising_regret_stats = {
            'active': torch.ones_like(raw_loss.detach()),
            'schedule_scale': torch.as_tensor(
                schedule_scale,
                dtype=raw_loss.dtype,
                device=raw_loss.device,
            ),
            'raw_loss': raw_loss.detach(),
            'selected_fraction': selected_count / denominator,
            'beneficial_challenger_rate': stats[3] / denominator,
            'mean_abs_normalized_change': stats[4] / denominator,
            'mean_router_margin': stats[5] / denominator,
            'mean_abs_first_order_change': stats[6] / denominator,
            'probes_per_conditional_image': torch.as_tensor(
                probe_tokens_per_image,
                dtype=raw_loss.dtype,
                device=raw_loss.device,
            ),
        }

    def _inactive_regret_loss(self, prediction, training_step):
        zero = prediction.float().sum() * 0.0
        schedule_scale = (
            self._schedule_scale(training_step)
            if training_step is not None
            else 0.0
        )
        self.denoising_regret_stats = {
            'active': zero.detach(),
            'schedule_scale': torch.as_tensor(
                schedule_scale,
                dtype=zero.dtype,
                device=zero.device,
            ),
        }
        return zero

    def forward(
        self,
        x,
        timestep,
        context,
        teacher_all_z=None,
        denoising_target=None,
        training_step=None,
        **kwargs,
    ):
        if not self.training:
            return super().forward(
                x, timestep, context, teacher_all_z=teacher_all_z, **kwargs
            )
        if training_step is not None and (
            isinstance(training_step, bool) or not isinstance(training_step, int)
        ):
            raise ValueError("training_step must be an integer")

        probe_active = (
            torch.is_grad_enabled()
            and denoising_target is not None
            and training_step is not None
            and self._should_probe(training_step)
        )
        capture = {}
        hook = None
        if probe_active:
            moe_layer = self.blocks[self.denoising_regret_block].mlp

            def capture_moe_output(module, inputs, output):
                capture['hidden_states'] = inputs[0]
                capture['labels'] = inputs[1]
                capture['moe_output'] = output[0]

            hook = moe_layer.register_forward_hook(capture_moe_output)

        try:
            prediction, repa_loss = super().forward(
                x,
                timestep,
                context,
                teacher_all_z=teacher_all_z,
                **kwargs,
            )
        finally:
            if hook is not None:
                hook.remove()

        if not probe_active:
            regret_loss = self._inactive_regret_loss(
                prediction, training_step
            )
        else:
            required = {'hidden_states', 'labels', 'moe_output'}
            if capture.keys() != required:
                raise RuntimeError(
                    f"denoising regret capture is incomplete: {capture.keys()}"
                )
            regret_loss = self.compute_denoising_regret_loss(
                prediction,
                denoising_target,
                capture,
                training_step,
            )
        return prediction, repa_loss, regret_loss
