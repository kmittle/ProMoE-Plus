"""Model-free integration tests for finite-horizon routing rollouts."""

import copy
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from analyses.finite_horizon_routing.batch import (
    SAFETY_REQUIREMENTS,
    aggregate_case_results,
)
from analyses.finite_horizon_routing.probe import (
    _close_captures,
    _checkpoint_identity,
    _probe_cell,
    _rollout,
    _rollout_losses,
    _verified_checkpoint_for_loading,
    _verified_latent_for_loading,
)
from analyses.finite_horizon_routing.protocol import (
    BLOCK_INDICES,
    START_INDICES,
    analytic_flow_state,
    sampling_sigmas,
)
from analyses.routing_translation.probe import RouteInputCapture


class FakeMoe(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_routed_experts = 2
        self.num_experts = 2
        self.top_k = 1
        self.router_weight_mode = "identity"
        self.phase_metric = None
        self.cluster_centers = nn.Parameter(
            torch.tensor([[-1.0], [1.0]]),
            requires_grad=False,
        )

    def compute_router(self, hidden_states, labels, timestep=None):
        del labels, timestep
        indices = (hidden_states[..., 0] >= 0).long().unsqueeze(-1)
        weights = torch.ones_like(indices, dtype=hidden_states.dtype)
        return weights, indices, None

    def forward(self, hidden_states, labels, timestep=None):
        weights, indices, auxiliary = self.compute_router(
            hidden_states,
            labels,
            timestep,
        )
        signed_expert = indices[..., 0].to(hidden_states.dtype) * 2.0 - 1.0
        return hidden_states + 0.05 * weights[..., 0:1] * signed_expert.unsqueeze(-1), auxiliary


class FakeModel(nn.Module):
    def __init__(self, num_patches=2):
        super().__init__()
        self.x_embedder = SimpleNamespace(num_patches=num_patches)
        blocks = []
        for index in range(12):
            block = nn.Module()
            block.use_moe = index in BLOCK_INDICES
            block.mlp = FakeMoe() if block.use_moe else nn.Identity()
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs, timestep, context):
        height, width = inputs.shape[-2:]
        hidden = inputs[:, 0, 0].reshape(inputs.shape[0], -1, 1)
        token_bias = torch.linspace(
            -0.75,
            0.75,
            steps=height * width,
            device=hidden.device,
            dtype=hidden.dtype,
        ).view(1, height * width, 1)
        hidden = hidden + token_bias
        for index in BLOCK_INDICES:
            hidden, _ = self.blocks[index].mlp(hidden, context, timestep)
        return hidden.squeeze(-1).reshape(inputs.shape[0], 1, height, width)


class RolloutTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel().eval()
        self.captures = {
            index: RouteInputCapture(self.model.blocks[index].mlp)
            for index in BLOCK_INDICES
        }
        self.sigmas = sampling_sigmas()
        self.clean = torch.tensor([[[[[-1.0, 1.0]]]]])
        self.noise = torch.tensor([[[[[1.0, -1.0]]]]])
        self.start_index = 50
        initial = analytic_flow_state(
            self.clean,
            self.noise,
            self.sigmas[self.start_index],
        )
        self.initial = initial.expand(2, -1, -1, -1, -1).clone()
        self.labels = torch.tensor([3, 3], dtype=torch.long)

    def tearDown(self):
        _close_captures(self.captures)

    def _run(self, **kwargs):
        return _rollout(
            model=self.model,
            captures=self.captures,
            initial_state=self.initial,
            labels=self.labels,
            clean_latent=self.clean,
            noise=self.noise,
            sigmas=self.sigmas,
            start_index=self.start_index,
            num_train_timesteps=1000,
            **kwargs,
        )

    def test_forced_native_is_identical_and_override_is_removed(self):
        native = self._run()
        route_ids = native["routes"]["0"][1]
        route_weights = torch.ones_like(route_ids, dtype=self.initial.dtype)
        forced = self._run(
            forced_block_index=1,
            forced_route_ids=route_ids,
            forced_route_weights=route_weights,
        )
        torch.testing.assert_close(
            forced["first_prediction"],
            native["first_prediction"],
            rtol=0,
            atol=0,
        )
        for horizon in (1, 2, 4, 8):
            torch.testing.assert_close(
                forced["states"][str(horizon)],
                native["states"][str(horizon)],
                rtol=0,
                atol=0,
            )
        self.assertNotIn("compute_router", self.model.blocks[1].mlp.__dict__)

    def test_swap_is_forced_only_at_the_intervention_step(self):
        native = self._run()
        route_ids = native["routes"]["0"][1].clone()
        swapped = route_ids.clone()
        swapped[:, [0, 1]] = swapped[:, [1, 0]]
        self.assertTrue(torch.equal(
            torch.bincount(route_ids[0], minlength=2),
            torch.bincount(swapped[0], minlength=2),
        ))
        forced = self._run(
            forced_block_index=1,
            forced_route_ids=swapped,
            forced_route_weights=torch.ones_like(swapped, dtype=self.initial.dtype),
        )
        self.assertTrue(torch.equal(forced["routes"]["0"][1], swapped))
        self.assertFalse(torch.equal(
            forced["first_prediction"],
            native["first_prediction"],
        ))
        self.assertNotIn("compute_router", self.model.blocks[1].mlp.__dict__)

    def test_horizon_one_matches_immediate_velocity_order(self):
        native = self._run()
        route_ids = native["routes"]["0"][1].clone()
        swapped = route_ids.clone()
        swapped[:, [0, 1]] = swapped[:, [1, 0]]
        forced = self._run(
            forced_block_index=1,
            forced_route_ids=swapped,
            forced_route_weights=torch.ones_like(swapped, dtype=self.initial.dtype),
        )
        native_losses = _rollout_losses(
            native,
            self.clean,
            self.noise,
            self.sigmas,
            self.start_index,
        )
        forced_losses = _rollout_losses(
            forced,
            self.clean,
            self.noise,
            self.sigmas,
            self.start_index,
        )
        velocity_change = forced_losses["immediate"] - native_losses["immediate"]
        state_change = forced_losses["1"] - native_losses["1"]
        step = self.sigmas[self.start_index + 1] - self.sigmas[self.start_index]
        torch.testing.assert_close(
            state_change,
            (step ** 2) * velocity_change,
            rtol=2e-4,
            atol=1e-9,
        )


class CheckpointIdentityTest(unittest.TestCase):
    def test_local_weight_copy_must_match_canonical_sha256(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            canonical = root / "ckpt_step_300000.pth"
            local_copy = root / "local" / "ckpt_step_300000.pth"
            local_copy.parent.mkdir()
            content = b"canonical-ema-checkpoint"
            canonical.write_bytes(content)
            local_copy.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            identity = _checkpoint_identity(
                canonical,
                local_copy,
                expected_size=len(content),
                expected_sha256=digest,
            )
            self.assertFalse(identity["same_file"])
            self.assertEqual(identity["weights_sha256"], digest)

            local_copy.write_bytes(b"Canonical-ema-checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                _checkpoint_identity(
                    canonical,
                    local_copy,
                    expected_size=len(content),
                    expected_sha256=digest,
                )

    def test_same_file_is_hashed_instead_of_trusting_expected_sha256(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "ckpt_step_300000.pth"
            checkpoint.write_bytes(b"actual-checkpoint")
            wrong_digest = hashlib.sha256(b"wrong--checkpoint").hexdigest()
            with self.assertRaisesRegex(ValueError, "Canonical checkpoint SHA256"):
                _checkpoint_identity(
                    checkpoint,
                    checkpoint,
                    expected_size=checkpoint.stat().st_size,
                    expected_sha256=wrong_digest,
                )

    def test_loaded_handle_is_rehashed_after_loading(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "ckpt_step_300000.pth"
            original = b"canonical-checkpoint"
            checkpoint.write_bytes(original)
            digest = hashlib.sha256(original).hexdigest()
            with self.assertRaisesRegex(ValueError, "Canonical checkpoint SHA256"):
                with _verified_checkpoint_for_loading(
                    checkpoint,
                    checkpoint,
                    expected_size=len(original),
                    expected_sha256=digest,
                ) as (handle, identity):
                    self.assertEqual(handle.read(), original)
                    self.assertEqual(identity["weights_sha256"], digest)
                    checkpoint.write_bytes(b"tampered--checkpoint")


class LatentIdentityTest(unittest.TestCase):
    def test_atomic_replacement_does_not_change_the_open_latent(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            latent = root / "case.latent.npz"
            replacement = root / "replacement.latent.npz"
            expected = np.zeros((8, 2, 2), dtype=np.float32)
            np.savez(latent, latent=expected)
            np.savez(replacement, latent=np.ones_like(expected))
            content = latent.read_bytes()
            with self.assertRaisesRegex(
                RuntimeError,
                r"Latent(?: path)? changed while it was open",
            ):
                with _verified_latent_for_loading(
                    latent,
                    expected_size=len(content),
                    expected_sha256=hashlib.sha256(content).hexdigest(),
                ) as (handle, identity):
                    os.replace(replacement, latent)
                    with np.load(handle) as latent_file:
                        np.testing.assert_array_equal(latent_file["latent"], expected)
                    self.assertEqual(identity["size"], len(content))

    def test_in_place_latent_change_is_rejected_after_loading(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            latent = Path(temporary_dir) / "case.latent.npz"
            np.savez(latent, latent=np.zeros((8, 2, 2), dtype=np.float32))
            content = latent.read_bytes()
            with self.assertRaisesRegex(ValueError, "Latent SHA256"):
                with _verified_latent_for_loading(
                    latent,
                    expected_size=len(content),
                    expected_sha256=hashlib.sha256(content).hexdigest(),
                ) as (handle, _):
                    handle.read(1)
                    latent.write_bytes(b"x" * len(content))


class ProbeCellTest(unittest.TestCase):
    def test_complete_cell_preserves_counts_and_tracks_propagation(self):
        model = FakeModel(num_patches=16).eval()
        clean = torch.linspace(-1.0, 1.0, steps=16).reshape(1, 1, 1, 4, 4)
        noise = torch.linspace(1.0, -1.0, steps=16).reshape(1, 1, 1, 4, 4)
        sigmas = sampling_sigmas()
        cell = _probe_cell(
            model=model,
            clean_latent=clean,
            noise=noise,
            label=torch.tensor([7], dtype=torch.long),
            sigmas=sigmas,
            start_index=50,
            block_index=1,
            num_train_timesteps=1000,
            seed=123,
            candidate_count=16,
            candidate_chunk_size=8,
        )
        self.assertEqual(cell["candidate_count"], 16)
        self.assertEqual(cell["numerical_controls"]["count_mismatches"], 0)
        self.assertTrue(all(
            candidate["full_count_match"] for candidate in cell["candidates"]
        ))
        self.assertAlmostEqual(
            cell["mean_route_divergence"]["0"]["1"],
            2 / 16,
        )
        h1 = cell["summary"]["per_horizon"]["1"]
        self.assertAlmostEqual(h1["immediate_future_spearman"], 1.0)
        self.assertTrue(h1["best_candidate_matches"])
        h1_identity_relative_error = (
            cell["numerical_controls"][
                "max_abs_h1_state_velocity_identity_error"
            ]
            / cell["candidates"][0]["h1_native_mse"]
        )
        self.assertLessEqual(
            h1_identity_relative_error,
            SAFETY_REQUIREMENTS[
                "maximum_h1_state_velocity_identity_relative_error"
            ],
        )
        plumbing_cases = []
        for case_index in range(4):
            cells = []
            for block_index in BLOCK_INDICES:
                for start_index in START_INDICES:
                    copied = copy.deepcopy(cell)
                    copied["block_index"] = block_index
                    copied["start_index"] = start_index
                    cells.append(copied)
            plumbing_cases.append({
                "batch_case": {"id": f"float32-plumbing-{case_index}"},
                "cells": cells,
            })
        self.assertTrue(
            aggregate_case_results(plumbing_cases, "plumbing")["passed"]
        )


if __name__ == "__main__":
    unittest.main()
