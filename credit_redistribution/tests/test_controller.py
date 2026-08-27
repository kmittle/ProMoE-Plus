from __future__ import annotations

import copy
import datetime
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from credit_redistribution.controller import (
    CreditRateNormalizer,
    CreditRedistributionController,
    _distributed_guard,
    compute_budget_factor,
    deterministic_group_sum,
    rotating_permuted_scales,
)


BLOCKS = (1, 3, 5, 7, 9, 11)
START_STEP = 301001
LAST_STEP = 321000


def _protocol():
    return {
        "branches": {
            "names": [
                "measure_only_control",
                "rotating_permuted_scale_control",
                "matched_credit_rate_redistribution",
            ],
            "start_step": START_STEP,
            "last_step": LAST_STEP,
            "save_checkpoint_interval": 1000,
        },
        "checkpoint": {
            "model_name": "ProMoE_TC_B",
            "frozen_path": "/tmp/frozen-credit-test.pth",
            "sha256": "0" * 64,
        },
        "source_anchor": {
            "training_facts": {
                "learning_rate": 1e-4,
                "weight_decay": 0,
                "global_batch_size": 256,
                "max_grad_norm": 0.5,
                "routed_experts_per_block": 12,
                "routed_blocks_zero_based": list(BLOCKS),
            },
        },
        "training_measurement": {"scope": "test"},
        "normalizer": {"ema_decay": 0.99, "epsilon": 1e-30},
        "raw_gradient_budget": {"relative_tolerance": 1e-6},
    }


class _TinyExpert(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return torch.tanh(self.proj(hidden_states))


class _TinyMoe(nn.Module):
    num_routed_experts = 12
    top_k = 1
    router_weight_mode = "identity"

    def __init__(self, hidden_size):
        super().__init__()
        self.experts = nn.ModuleList([
            _TinyExpert(hidden_size) for _ in range(self.num_routed_experts)
        ])

    def compute_router(self, hidden_states, labels):
        del labels
        batch, tokens, _ = hidden_states.shape
        indices = torch.arange(tokens, device=hidden_states.device)
        indices = indices.remainder(self.num_routed_experts)
        indices = indices.view(1, tokens, 1).expand(batch, -1, -1)
        weights = 0.75 + indices.to(hidden_states.dtype) / 100.0
        return weights, indices, None

    def forward(self, hidden_states, labels):
        weights, indices, _ = self.compute_router(hidden_states, labels)
        output = torch.zeros_like(hidden_states)
        for expert_index, expert in enumerate(self.experts):
            mask = indices[..., 0] == expert_index
            expert_output = expert(hidden_states[mask])
            selected_weights = weights[..., 0][mask].unsqueeze(1)
            output[mask] = expert_output * selected_weights
        return output, None


class _TinyBlock(nn.Module):
    def __init__(self, hidden_size, use_moe):
        super().__init__()
        self.use_moe = use_moe
        self.mlp = _TinyMoe(hidden_size) if use_moe else nn.Linear(
            hidden_size, hidden_size
        )

    def forward(self, hidden_states, labels):
        if self.use_moe:
            output, _ = self.mlp(hidden_states, labels)
            return hidden_states + output
        return hidden_states + torch.tanh(self.mlp(hidden_states))


class _TinyModel(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            _TinyBlock(hidden_size, index in BLOCKS) for index in range(12)
        ])
        self.head = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, labels):
        for block in self.blocks:
            hidden_states = block(hidden_states, labels)
        return self.head(hidden_states)


def _runtime_cfg(dataset_root):
    return SimpleNamespace(
        total_train_batch_size=256,
        grad_mix=1,
        global_seed=0,
        lr=1e-4,
        weight_decay=0,
        max_grad_norm=0.5,
        use_gradient_checkpointing=False,
        model_name="ProMoE_TC_B",
        num_steps=LAST_STEP + 1,
        save_ckpt_interval=1000,
        structured_batch_sampling=False,
        latent_data_path=str(dataset_root),
    )


def _controller_cfg(artifact_root):
    return {
        "enabled": True,
        "branch": "measure_only_control",
        "execution_mode": "continuation",
        "artifact_root": str(artifact_root),
        "initial_checkpoint_path": "/tmp/frozen-credit-test.pth",
        "preregister_v3_path": "/tmp/v3.json",
        "preregister_v4_path": "/tmp/v4.json",
    }


def _transcript_inputs(path, hidden_states, labels):
    tensors = {
        "latent_parameters": hidden_states.detach().clone(),
        "realized_z": hidden_states.detach().clone(),
        "sampled_u": torch.tensor([0.25], dtype=torch.float32),
        "timestep": torch.tensor([250.0], dtype=torch.float32),
        "sigma": torch.tensor([0.25], dtype=torch.float32),
        "diffusion_noise": hidden_states.detach().clone(),
        "noised_model_input": hidden_states.detach().clone(),
        "denoising_target": hidden_states.detach().clone(),
    }
    return {
        "paths": [str(path)],
        "original_labels": labels,
        "tensors": tensors,
    }


def _distributed_failure_worker(rank, world_size, init_file, queue):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=10),
    )
    try:
        def operation():
            if rank == 2:
                raise ValueError("intentional rank-local failure")
            return rank

        try:
            _distributed_guard("unit-test phase", operation)
        except RuntimeError as error:
            queue.put((rank, "intentional rank-local failure" in str(error)))
        else:
            queue.put((rank, False))
    finally:
        dist.destroy_process_group()


class ControllerFormulaTest(unittest.TestCase):
    def test_deterministic_group_sum_matches_explicit_float64_reduction(self):
        values = torch.tensor(
            [0.5, 1.25, 2.0, 4.5, 8.0], dtype=torch.float64
        )
        indices = torch.tensor([2, 0, 2, 1, 0], dtype=torch.int64)
        first_sum, first_count = deterministic_group_sum(values, indices, 4)
        second_sum, second_count = deterministic_group_sum(values, indices, 4)
        self.assertTrue(torch.equal(first_sum, second_sum))
        self.assertTrue(torch.equal(first_count, second_count))
        self.assertEqual(first_sum.tolist(), [9.25, 4.5, 2.5, 0.0])
        self.assertEqual(first_count.tolist(), [2, 1, 2, 0])

    def test_budget_factor_preserves_float64_squared_norm(self):
        norms = torch.tensor([1.0, 2.0, 7.0], dtype=torch.float64)
        scales = torch.tensor([0.5, 1.25, 2.0], dtype=torch.float64)
        factor = compute_budget_factor(norms, scales)
        self.assertEqual(factor.dtype, torch.float64)
        before = norms.sum()
        after = (norms * (factor * scales).square()).sum()
        self.assertLessEqual(float(torch.abs(before - after) / before), 1e-15)

    def test_rotating_permutation_has_no_identity_and_covers_all_offsets(self):
        scales = torch.arange(12, dtype=torch.float64).view(1, 12)
        offsets = []
        for update in range(11):
            permuted, offset = rotating_permuted_scales(scales, update)
            offsets.append(offset)
            expected = torch.roll(scales, shifts=-offset, dims=-1)
            self.assertTrue(torch.equal(permuted, expected))
            self.assertFalse(torch.equal(permuted, scales))
        self.assertEqual(offsets, list(range(1, 12)))

    def test_normalizer_checkpoint_round_trip_is_exact(self):
        normalizer = CreditRateNormalizer((2, 12), device="cpu")
        rates = torch.linspace(0.1, 2.4, 24, dtype=torch.float64).reshape(2, 12)
        normalizer.update(rates)
        state = normalizer.state_dict()
        restored = CreditRateNormalizer((2, 12), device="cpu")
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(normalizer.ema, restored.ema))
        self.assertTrue(torch.equal(normalizer.initialized, restored.initialized))


class ControllerIntegrationTest(unittest.TestCase):
    def _make_controller(self, model, root, dataset_root):
        with mock.patch(
            "credit_redistribution.controller.load_effective_protocol",
            return_value=_protocol(),
        ), mock.patch(
            "credit_redistribution.controller._world_size",
            return_value=4,
        ):
            controller = CreditRedistributionController(
                model=model,
                runtime_cfg=_runtime_cfg(dataset_root),
                controller_cfg=_controller_cfg(root),
            )
        controller.world_size = 1
        return controller

    def test_measure_only_keeps_gradients_and_adamw_update_bit_identical(self):
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "latents"
            dataset_root.mkdir()
            latent_path = dataset_root / "sample.latent.npz"
            latent_path.touch()

            original = _TinyModel()
            baseline = copy.deepcopy(original)
            controlled = copy.deepcopy(original)
            baseline_optimizer = torch.optim.AdamW(
                baseline.parameters(), lr=1e-4, weight_decay=0
            )
            controlled_optimizer = torch.optim.AdamW(
                controlled.parameters(), lr=1e-4, weight_decay=0
            )
            controller = self._make_controller(
                controlled, root / "artifacts", dataset_root
            )
            hidden_states = torch.linspace(
                -1.0, 1.0, 48, dtype=torch.float32
            ).reshape(1, 12, 4)
            labels = torch.tensor([7], dtype=torch.int64)
            target = torch.full_like(hidden_states, 0.125)

            baseline_loss = (baseline(hidden_states, labels) - target).square().mean()
            baseline_loss.backward()

            controller.begin_step(START_STEP)
            controlled_loss = (
                controlled(hidden_states, labels) - target
            ).square().mean()
            controlled_loss.backward()
            controller.after_backward(
                controlled_optimizer,
                _transcript_inputs(latent_path, hidden_states, labels),
            )

            for baseline_parameter, controlled_parameter in zip(
                baseline.parameters(), controlled.parameters()
            ):
                self.assertTrue(
                    torch.equal(baseline_parameter.grad, controlled_parameter.grad)
                )

            baseline_optimizer.step()
            controlled_optimizer.step()
            controller.after_optimizer_step(controlled_optimizer)
            for baseline_parameter, controlled_parameter in zip(
                baseline.parameters(), controlled.parameters()
            ):
                self.assertTrue(torch.equal(baseline_parameter, controlled_parameter))

            state = controller.checkpoint_state_dict()
            self.assertEqual(state["update_count"], 1)
            self.assertEqual(state["execution_mode"], "continuation")
            checkpoint = {
                "step": START_STEP,
                controller.checkpoint_state_key: state,
            }
            second = self._make_controller(
                copy.deepcopy(original), root / "resume", dataset_root
            )
            prepared = second.prepare_checkpoint_state(checkpoint, is_initial=False)
            second.commit_checkpoint_state(prepared)
            self.assertEqual(second.update_count, 1)
            self.assertTrue(torch.equal(second.normalizer.ema, controller.normalizer.ema))
            with self.assertRaisesRegex(RuntimeError, "locked updates"):
                controller.close()
            controller.expected_update_total = 1
            second.expected_update_total = 1
            controller.close()
            second.close()

    def test_one_rank_failure_reaches_every_rank_without_hanging(self):
        context = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as temporary:
            init_file = str(Path(temporary) / "gloo-init")
            queue = context.Queue()
            processes = [
                context.Process(
                    target=_distributed_failure_worker,
                    args=(rank, 4, init_file, queue),
                )
                for rank in range(4)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=20)
            alive = [process for process in processes if process.is_alive()]
            for process in alive:
                process.terminate()
                process.join(timeout=5)
            self.assertEqual(alive, [])
            results = sorted(queue.get(timeout=2) for _ in range(4))
            self.assertEqual(results, [(rank, True) for rank in range(4)])
            self.assertTrue(all(process.exitcode == 0 for process in processes))


if __name__ == "__main__":
    unittest.main()
