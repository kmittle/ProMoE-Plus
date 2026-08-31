"""ProMoE-TC with semantic-uncertainty-aware load-aware routing.

DINOv2 is used only offline to estimate one uncertainty value for each
ImageNet class.  The value is not aligned with a DiT feature and no teacher
representation enters the backbone.  During training, uncertain classes get
a small, detached preference for experts whose recent conditional load is
low.  Expert outputs keep the original cosine routing weight, so the added
signal changes assignment and specialization rather than the representation
scale.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .models_ProMoE_TC import (
    DiT as BaseDiT,
    SparseMoeBlock as BaseSparseMoeBlock,
)


class DinoRouteSparseMoeBlock(BaseSparseMoeBlock):
    """Base token-choice block with a detached DINO/load route preference."""

    def __init__(self, *args, dino_route_config=None, **kwargs):
        super().__init__(*args, **kwargs)

        config = dict(dino_route_config or {})
        self.dino_route_enabled = bool(config.get("enabled", True))
        self.dino_route_mapping = str(config.get("mapping", "correct"))
        self.dino_route_permutation_seed = int(
            config.get("permutation_seed", 20260831)
        )
        self.dino_route_strength = float(config.get("strength", 0.08))
        self.dino_route_ema_decay = float(config.get("ema_decay", 0.99))
        self.dino_route_warmup_updates = int(
            config.get("warmup_updates", 2000)
        )
        self.dino_route_bias_cap = float(config.get("bias_cap", 0.20))
        self.dino_route_num_classes = int(config.get("num_classes", 1000))

        if self.phase_metric is not None:
            raise ValueError(
                "DINO route experiments must disable phase_metric_config; "
                "the two routing interventions are not combined"
            )
        if self.dino_route_mapping not in {"correct", "shuffled"}:
            raise ValueError(
                "dino route mapping must be 'correct' or 'shuffled'"
            )
        if self.dino_route_strength < 0:
            raise ValueError("dino route strength must be non-negative")
        if not 0.0 <= self.dino_route_ema_decay < 1.0:
            raise ValueError("dino route ema_decay must be in [0, 1)")
        if self.dino_route_warmup_updates < 0:
            raise ValueError("dino route warmup_updates must be non-negative")
        if self.dino_route_bias_cap < 0:
            raise ValueError("dino route bias_cap must be non-negative")
        if self.dino_route_num_classes <= 0:
            raise ValueError("dino route num_classes must be positive")

        uncertainty = self._load_uncertainty_table(config)
        permutation = torch.arange(self.dino_route_num_classes, dtype=torch.long)
        if self.dino_route_mapping == "shuffled":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.dino_route_permutation_seed)
            permutation = torch.randperm(
                self.dino_route_num_classes,
                generator=generator,
            )
            # A fixed-point permutation would make the control accidentally
            # identical to the correct mapping.
            if torch.equal(
                permutation,
                torch.arange(permutation.numel(), dtype=torch.long),
            ):
                permutation = torch.roll(permutation, shifts=1, dims=0)
            uncertainty = uncertainty[permutation]

        self.register_buffer(
            "dino_class_uncertainty", uncertainty.contiguous(), persistent=True
        )
        self.register_buffer(
            "dino_class_permutation", permutation.contiguous(), persistent=True
        )
        # These buffers are part of the checkpoint because inference must use
        # the load estimate reached by the trained router.
        self.register_buffer(
            "dino_route_load_ema",
            torch.zeros(self.num_routed_experts, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "dino_route_update_count",
            torch.zeros(1, dtype=torch.long),
            persistent=True,
        )
        self.last_dino_route_stats = {}

        table_path = config.get("table_path")
        print(
            "DINO route calibration: "
            f"mapping={self.dino_route_mapping}, "
            f"strength={self.dino_route_strength}, "
            f"ema_decay={self.dino_route_ema_decay}, "
            f"warmup_updates={self.dino_route_warmup_updates}, "
            f"table={table_path}"
        )

    def _load_uncertainty_table(self, config):
        if not self.dino_route_enabled:
            return torch.zeros(self.dino_route_num_classes, dtype=torch.float32)

        table_path_value = config.get("table_path")
        if not table_path_value:
            raise ValueError(
                "dino route table_path is required when the route is enabled"
            )
        table_path = Path(os.path.expanduser(str(table_path_value)))
        if table_path.suffix != ".npz" or not table_path.is_file():
            raise FileNotFoundError(
                f"DINO route table must be an existing .npz file: {table_path}"
            )
        with np.load(table_path, allow_pickle=False) as archive:
            if "uncertainty" not in archive.files:
                raise ValueError(
                    f"DINO route table lacks uncertainty: {table_path}"
                )
            values = np.asarray(archive["uncertainty"], dtype=np.float32)
        metadata_path = table_path.with_suffix(table_path.suffix + ".json")
        if metadata_path.is_file():
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Cannot read DINO route metadata: {metadata_path}"
                ) from error
            if int(metadata.get("num_classes", -1)) != self.dino_route_num_classes:
                raise ValueError(
                    "DINO route metadata num_classes does not match the model"
                )
        if values.shape != (self.dino_route_num_classes,):
            raise ValueError(
                "DINO uncertainty table shape must be "
                f"({self.dino_route_num_classes},), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("DINO uncertainty table contains non-finite values")
        values = np.clip(values, 0.0, 1.0)
        return torch.from_numpy(values)

    def _load_preference(self, device, dtype):
        """Return positive values for underloaded routed experts."""

        load = self.dino_route_load_ema.to(device=device, dtype=torch.float32)
        mean_load = load.mean()
        has_history = mean_load > 1e-6
        preference = (mean_load - load) / (mean_load + 1e-6)
        preference = preference.clamp(-1.0, 1.0)
        if self.dino_route_warmup_updates > 0:
            warmup = (
                self.dino_route_update_count.to(device=device, dtype=torch.float32)
                / float(self.dino_route_warmup_updates)
            ).clamp(0.0, 1.0)
            preference = preference * warmup
        preference = preference.masked_fill(~has_history, 0.0)
        preference = preference * self.dino_route_strength
        if self.dino_route_bias_cap > 0:
            preference = preference.clamp(
                -self.dino_route_bias_cap, self.dino_route_bias_cap
            )
        return preference.to(dtype=dtype)

    @torch.no_grad()
    def _update_load_ema(self, selected_indices, device):
        """Update a globally synchronized load estimate without gradients."""

        counts = torch.zeros(
            self.num_routed_experts, device=device, dtype=torch.float32
        )
        if selected_indices.numel() > 0:
            counts.scatter_add_(
                0,
                selected_indices.reshape(-1).to(dtype=torch.long),
                torch.ones(
                    selected_indices.numel(), device=device, dtype=torch.float32
                ),
            )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        self.dino_route_load_ema.mul_(self.dino_route_ema_decay).add_(
            counts, alpha=1.0 - self.dino_route_ema_decay
        )
        self.dino_route_update_count.add_(1)
        return counts

    def compute_router(self, hidden_states, labels, timestep=None):
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        flat_input = hidden_states.view(-1, self.hidden_size)
        flat_labels = labels.view(batch_size, 1).expand(-1, seq_len).reshape(-1)

        if self.use_uncond_expert and flat_labels is not None:
            uncond_mask = flat_labels == 1000
            cond_mask = ~uncond_mask
        else:
            uncond_mask = None
            cond_mask = torch.ones_like(flat_labels, dtype=torch.bool)

        router_weights = torch.zeros(
            batch_size * seq_len, self.top_k, device=device
        )
        expert_indices = torch.zeros(
            batch_size * seq_len,
            self.top_k,
            device=device,
            dtype=torch.long,
        )
        cond_weights = None
        topk_idx = None
        bias = torch.zeros(
            0, self.num_routed_experts, device=device, dtype=hidden_states.dtype
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

            # The preference is detached by construction: both the class table
            # and EMA load state are buffers, and only selection sees the bias.
            class_ids = flat_labels[cond_positions].to(dtype=torch.long)
            if (
                class_ids.numel() > 0
                and (
                    class_ids.min() < 0
                    or class_ids.max() >= self.dino_route_num_classes
                )
            ):
                raise ValueError("Conditional labels fall outside the DINO table")
            uncertainty = self.dino_class_uncertainty[class_ids].to(
                device=device, dtype=torch.float32
            )
            preference = self._load_preference(device, torch.float32)
            bias = uncertainty.unsqueeze(1) * preference.unsqueeze(0)
            selection_cos = cos_sim.float() + bias

            if self.router_weight_mode == "softmax":
                cond_weights = F.softmax(cos_sim, dim=1)
                selection_weights = F.softmax(selection_cos, dim=1)
            elif self.router_weight_mode == "sigmoid":
                sigmoid_scale = 1.0
                cond_weights = torch.sigmoid(cos_sim * sigmoid_scale)
                selection_weights = torch.sigmoid(selection_cos * sigmoid_scale)
            elif self.router_weight_mode == "identity":
                cond_weights = cos_sim
                selection_weights = selection_cos
            else:
                raise ValueError(
                    f"Unsupported router_weight_mode: {self.router_weight_mode}"
                )

            _, topk_idx = torch.topk(selection_weights, k=self.top_k, dim=1)
            topk_scores = torch.gather(cond_weights, 1, topk_idx)
            router_weights[cond_positions] = topk_scores.to(router_weights.dtype)
            expert_indices[cond_positions] = topk_idx

            uncertainty_mean = uncertainty.mean()
            bias_abs_mean = bias.abs().mean()
        else:
            uncertainty_mean = torch.zeros((), device=device)
            bias_abs_mean = torch.zeros((), device=device)

        if self.dino_route_enabled and self.training:
            # Every rank executes this path, including an all-unconditional
            # batch, so the optional DDP collective cannot deadlock.
            selected = (
                topk_idx
                if topk_idx is not None
                else torch.empty(0, dtype=torch.long, device=device)
            )
            global_counts = self._update_load_ema(selected, device)
        else:
            global_counts = torch.zeros(
                self.num_routed_experts, device=device, dtype=torch.float32
            )

        self.last_dino_route_stats = {
            "uncertainty_mean": uncertainty_mean.detach(),
            "bias_abs_mean": bias_abs_mean.detach(),
            "batch_load_cv": (
                global_counts.std(unbiased=False)
                / (global_counts.mean() + 1e-6)
            ).detach(),
        }

        # Keep the original auxiliary-loss contract.  The experiment uses
        # alpha=0, but this branch remains compatible with existing callers.
        if self.training and self.alpha > 0.0 and topk_idx is not None:
            cond_batch_size = (labels != 1000).sum()
            if self.router_weight_mode != "softmax":
                scores_for_aux = F.softmax(cond_weights, dim=1)
            else:
                scores_for_aux = cond_weights
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(
                    cond_batch_size, seq_len, -1
                )
                ce = torch.zeros(
                    cond_batch_size,
                    self.num_routed_experts,
                    device=device,
                )
                ce.scatter_add_(
                    1,
                    topk_idx.view(cond_batch_size, -1),
                    torch.ones(
                        cond_batch_size,
                        seq_len * self.top_k,
                        device=device,
                    ),
                ).div_(seq_len * self.top_k / self.num_routed_experts)
                load_balance_loss = (
                    ce * scores_for_seq_aux.mean(dim=1)
                ).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx.view(-1), num_classes=self.num_routed_experts
                )
                ce = mask_ce.float().mean(0)
                pi = scores_for_aux.mean(0)
                fi = ce * self.num_routed_experts
                load_balance_loss = (pi * fi).sum() * self.alpha
        else:
            load_balance_loss = None

        return (
            router_weights.view(batch_size, seq_len, self.top_k),
            expert_indices.view(batch_size, seq_len, self.top_k),
            load_balance_loss,
        )


class DiT(BaseDiT):
    """Instantiate the base DiT and replace only its routed MoE blocks."""

    def __init__(self, *args, **kwargs):
        moe_config = kwargs.get("MoE_config")
        if moe_config is None:
            raise ValueError("DINO route model requires MoE_config")
        dino_config = dict(moe_config.get("dino_route_config", {}))
        base_moe_kwargs = dict(moe_config)
        base_moe_kwargs.pop("dino_route_config", None)

        super().__init__(*args, **kwargs)

        replaced = 0
        for block in self.blocks:
            if not getattr(block, "use_moe", False):
                continue
            old_moe = block.mlp
            # The base constructor initializes every expert and prototype with
            # torch RNG calls.  Those temporary values are immediately replaced
            # by ``old_state`` below, so they must not perturb the training RNG
            # stream used by the otherwise identical baseline.
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_states = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            )
            try:
                new_moe = DinoRouteSparseMoeBlock(
                    hidden_size=old_moe.hidden_size,
                    dino_route_config=dino_config,
                    **base_moe_kwargs,
                )
            finally:
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
            old_state = old_moe.state_dict()
            new_state_keys = set(new_moe.state_dict())
            missing_source = sorted(set(old_state) - new_state_keys)
            if missing_source:
                raise RuntimeError(
                    "DINO route replacement cannot copy base MoE state: "
                    f"{missing_source}"
                )
            missing, unexpected = new_moe.load_state_dict(
                old_state, strict=False
            )
            expected_missing = {
                "dino_class_uncertainty",
                "dino_class_permutation",
                "dino_route_load_ema",
                "dino_route_update_count",
            }
            if set(missing) != expected_missing:
                raise RuntimeError(
                    "Unexpected missing keys while copying base MoE state: "
                    f"{missing}"
                )
            if unexpected:
                raise RuntimeError(
                    f"Unexpected keys while copying base MoE state: {unexpected}"
                )
            block.mlp = new_moe
            replaced += 1

        if replaced == 0:
            raise ValueError("DINO route model found no MoE blocks")
