"""Low-rank diffusion-phase metric used by the ProMoE router.

The metric is deliberately a residual over the existing cosine affinity.  Its
phase projection is zero-initialized, so enabling the module does not alter the
step-0 routing function or the number of active experts.  The learned factors
form a bounded, expert-specific trilinear interaction between a token, a
prototype, and the scalar diffusion phase.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseConditionedRoutingMetric(nn.Module):
    """Factorized phase-conditioned residual for token/prototype affinity."""

    def __init__(
        self,
        hidden_size,
        num_experts,
        rank=8,
        num_fourier_bands=4,
        num_train_timesteps=1000,
        scale=0.25,
        init_seed=1729,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.rank = int(rank)
        self.num_fourier_bands = int(num_fourier_bands)
        self.num_train_timesteps = float(num_train_timesteps)
        self.scale = float(scale)
        self.init_seed = int(init_seed)

        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.num_experts < 1:
            raise ValueError("num_experts must be positive")
        if self.rank < 1 or self.rank > self.hidden_size:
            raise ValueError(
                f"rank must be in [1, hidden_size], got {self.rank}"
            )
        if self.num_fourier_bands < 1:
            raise ValueError("num_fourier_bands must be positive")
        if self.num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if self.scale <= 0:
            raise ValueError("scale must be positive")

        feature_dim = 1 + 2 * self.num_fourier_bands
        frequencies = torch.arange(
            1, self.num_fourier_bands + 1, dtype=torch.float32
        )
        self.register_buffer("fourier_frequencies", frequencies)

        # U and V define shared low-rank token/prototype coordinates.  The
        # phase projection is zero at initialization; expert_gain starts at
        # zero and is mapped to a bounded gain around one in forward().
        self.token_basis = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.prototype_basis = nn.Parameter(
            torch.empty(self.rank, self.hidden_size)
        )
        self.phase_to_rank = nn.Parameter(torch.zeros(feature_dim, self.rank))
        self.expert_gain = nn.Parameter(torch.zeros(self.num_experts, self.rank))
        self._init_weights()

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        # Use a private generator so adding this module cannot shift the
        # global RNG stream used to initialize the original ProMoE weights.
        generator = torch.Generator(device=self.token_basis.device)
        generator.manual_seed(self.init_seed)
        with torch.no_grad():
            self.token_basis.copy_(
                torch.randn(
                    self.token_basis.shape,
                    dtype=self.token_basis.dtype,
                    device=self.token_basis.device,
                    generator=generator,
                ) * std
            )
            self.prototype_basis.copy_(
                torch.randn(
                    self.prototype_basis.shape,
                    dtype=self.prototype_basis.dtype,
                    device=self.prototype_basis.device,
                    generator=generator,
                ) * std
            )

    def _phase_features(self, timesteps):
        timesteps = timesteps.reshape(-1).float()
        tau = (timesteps / self.num_train_timesteps).clamp(0.0, 1.0)
        angles = 2.0 * math.pi * tau[:, None] * self.fourier_frequencies[None]
        features = torch.cat(
            [tau[:, None], torch.sin(angles), torch.cos(angles)], dim=-1
        )
        return features / math.sqrt(features.shape[-1])

    def forward(self, token_norm, prototype_norm, timesteps):
        """Return a residual affinity matrix with shape ``(N, E)``.

        ``token_norm`` and ``prototype_norm`` are expected to be unit-normalized
        in the caller.  Calculations are promoted to float32 for stable routing
        under bf16 autocast; gradients still flow to all metric parameters.
        """
        if token_norm.ndim != 2 or token_norm.shape[-1] != self.hidden_size:
            raise ValueError(
                "token_norm must have shape (N, hidden_size), got "
                f"{tuple(token_norm.shape)}"
            )
        if prototype_norm.ndim != 2 or prototype_norm.shape[-1] != self.hidden_size:
            raise ValueError(
                "prototype_norm must have shape (E, hidden_size), got "
                f"{tuple(prototype_norm.shape)}"
            )
        if prototype_norm.shape[0] != self.num_experts:
            raise ValueError(
                f"expected {self.num_experts} prototypes, "
                f"got {prototype_norm.shape[0]}"
            )
        if timesteps.reshape(-1).shape[0] != token_norm.shape[0]:
            raise ValueError(
                "timesteps must contain one value per token; got "
                f"{timesteps.reshape(-1).shape[0]} for {token_norm.shape[0]} tokens"
            )

        with torch.autocast(device_type=token_norm.device.type, enabled=False):
            token_norm = token_norm.float()
            prototype_norm = prototype_norm.float()
            token_basis = F.normalize(self.token_basis.float(), dim=-1)
            prototype_basis = F.normalize(self.prototype_basis.float(), dim=-1)

            token_coordinates = F.linear(token_norm, token_basis)
            prototype_coordinates = F.linear(prototype_norm, prototype_basis)
            phase_features = self._phase_features(timesteps).to(token_norm.device)
            phase_coordinates = torch.tanh(
                phase_features @ self.phase_to_rank.float()
            )
            expert_gain = 1.0 + 0.5 * torch.tanh(self.expert_gain.float())

            # Each term couples the same rank coordinate from the token, phase,
            # and prototype. Dividing by sqrt(rank) keeps the residual scale
            # stable as the rank changes in controlled ablations.
            residual = torch.einsum(
                "nr,nr,er->ne",
                token_coordinates,
                phase_coordinates,
                prototype_coordinates * expert_gain,
            )
            residual = residual / math.sqrt(self.rank)
            return residual * self.scale
