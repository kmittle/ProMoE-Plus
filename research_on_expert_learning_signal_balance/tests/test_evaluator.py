from __future__ import annotations

import copy
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from credit_redistribution.controller import (
    BRANCHES,
    CHECKPOINT_STATE_KEY,
    CONTROLLER_STATE_VERSION,
)
from credit_redistribution.evaluator import (
    BLOCK_INDICES,
    CHECKPOINT_STATES,
    FINAL_STEP,
    NUM_EXPERTS,
    _credit_and_count,
    _publish_case,
    _prediction_tensor,
    _validated_case_artifact_inventory,
    validate_branch_checkpoint,
    validate_branch_transcripts,
    validate_controller_artifacts,
)
from credit_redistribution.transcript import (
    FIELD_ORDER,
    JsonlLedger,
    build_global_record,
    build_step_record,
)


def _transcript_tensors():
    value = torch.ones(1, dtype=torch.float32)
    return {name: value for name in FIELD_ORDER[2:-1]} | {
        "effective_labels": torch.tensor([4], dtype=torch.int64),
    }


def _valid_rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(
                numpy_state[1].astype(np.int64, copy=True)
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        # The sealed checkpoint contract records one CUDA state per rank even
        # when this CPU-only unit test is running without a CUDA device.
        "cuda": torch.zeros(1, dtype=torch.uint8),
    }


def _case_payload(branch, state_name, checkpoint_sha256, case_index=0):
    return {
        "version": 1,
        "branch": branch,
        "checkpoint_state": state_name,
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": "c" * 64,
        "case_index": case_index,
        "label": 4,
        "relative_path": "0004/sample.latent.npz",
        "mean_mse": 1.0,
        "aggregate_credit": [[1.0] * NUM_EXPERTS for _ in BLOCK_INDICES],
        "aggregate_count": [[512] * NUM_EXPERTS for _ in BLOCK_INDICES],
        "cells": [],
    }


def _controller_record(branch="matched_credit_rate_redistribution"):
    count = np.full((len(BLOCK_INDICES), NUM_EXPERTS), 2, dtype=np.int64)
    rates = np.linspace(
        0.25,
        2.0,
        len(BLOCK_INDICES) * NUM_EXPERTS,
        dtype=np.float64,
    ).reshape(len(BLOCK_INDICES), NUM_EXPERTS)
    credit = rates * count
    reference = np.exp(np.log(rates).mean(axis=1))
    raw = np.sqrt(reference[:, None] / rates)
    raw = np.clip(raw, 0.5, 2.0)
    offset = 1
    permuted = np.take(
        raw,
        (np.arange(NUM_EXPERTS) + offset) % NUM_EXPERTS,
        axis=1,
    )
    pre = np.linspace(
        0.5,
        3.0,
        len(BLOCK_INDICES) * NUM_EXPERTS,
        dtype=np.float64,
    ).reshape(len(BLOCK_INDICES), NUM_EXPERTS)
    if branch == "measure_only_control":
        selected = np.ones_like(raw)
    elif branch == "rotating_permuted_scale_control":
        selected = permuted
    else:
        selected = raw
    factors = np.sqrt(
        pre.sum(axis=1) / (pre * np.square(selected)).sum(axis=1)
    )
    applied = selected * factors[:, None]
    post = pre * np.square(applied)
    block_drifts = np.abs(post.sum(axis=1) - pre.sum(axis=1)) / pre.sum(axis=1)
    full_norm = float(pre.sum() + 10.0)
    return {
        "version": CONTROLLER_STATE_VERSION,
        "step": FINAL_STEP,
        "branch": branch,
        "update_index": 0,
        "global_transcript_digest": "a" * 64,
        "rank_consensus_digest": "b" * 64,
        "permutation_offset": offset,
        "global_credit": credit.tolist(),
        "global_count": count.tolist(),
        "credit_rate_ema": rates.tolist(),
        "raw_scales": raw.tolist(),
        "permuted_scales": permuted.tolist(),
        "selected_budget_factors": factors.tolist(),
        "applied_scales": applied.tolist(),
        "pre_gradient_squared_norm": pre.tolist(),
        "post_gradient_squared_norm": post.tolist(),
        "full_pre_gradient_squared_norm": full_norm,
        "full_post_gradient_squared_norm": full_norm,
        "block_relative_budget_drift": block_drifts.tolist(),
        "full_relative_budget_drift": 0.0,
    }


def _controller_checkpoint(branch, ema):
    counters = {
        "nonfinite": 0,
        "rank_disagreement": 0,
        "transcript_mismatch": 0,
        "budget_violation": 0,
        "capture_failure": 0,
        "checkpoint_failure": 0,
    }
    return {
        "step": FINAL_STEP,
        CHECKPOINT_STATE_KEY: {
            "version": CONTROLLER_STATE_VERSION,
            "branch": branch,
            "execution_mode": "continuation",
            "block_indices": list(BLOCK_INDICES),
            "num_experts": NUM_EXPERTS,
            "start_step": FINAL_STEP,
            "last_step": FINAL_STEP,
            "update_count": 1,
            "normalizer": {
                "ema": torch.as_tensor(ema, dtype=torch.float64),
                "initialized": torch.ones(
                    len(BLOCK_INDICES), NUM_EXPERTS, dtype=torch.bool
                ),
                "ema_decay": 0.99,
                "epsilon": 1e-30,
            },
            "numerical_counters": counters,
        },
    }


def _write_controller_artifacts(root, branch, record):
    transcript_path = root / "transcripts" / branch / "global.jsonl"
    JsonlLedger(transcript_path, FINAL_STEP).append_or_verify({
        "step": FINAL_STEP,
        "global_digest": "a" * 64,
    })
    ledger_path = root / "controller" / branch / "steps.jsonl"
    JsonlLedger(ledger_path, FINAL_STEP).append_or_verify(record)


class EvaluatorTest(unittest.TestCase):
    def test_prediction_contract_restores_time_dimension(self):
        output = torch.arange(8 * 2 * 2, dtype=torch.float32).reshape(1, 8, 2, 2)
        prediction = _prediction_tensor(output)
        self.assertEqual(tuple(prediction.shape), (1, 4, 1, 2, 2))
        self.assertTrue(torch.equal(prediction[:, :, 0], output[:, :4]))

    def test_credit_and_count_use_float64_top1_formula(self):
        weights = torch.tensor([[[0.5], [1.0], [0.25], [2.0]]])
        indices = torch.tensor([[[0], [1], [0], [2]]])
        gradient = torch.tensor([[[1.0, 2.0], [2.0, 0.0], [3.0, 4.0], [1.0, 1.0]]])
        record = {
            "route_weights": weights,
            "route_indices": indices,
            "labels": torch.tensor([7]),
            "output": torch.zeros_like(gradient),
        }
        credit, count = _credit_and_count(record, gradient, 7)
        self.assertEqual(credit.dtype, torch.float64)
        self.assertEqual(count.tolist()[:3], [2, 1, 1])
        self.assertAlmostEqual(credit[0].item(), 0.25 * 5.0 + 0.0625 * 25.0)
        self.assertAlmostEqual(credit[1].item(), 4.0)
        self.assertAlmostEqual(credit[2].item(), 8.0)

    def test_controller_validator_recomputes_policy_and_checkpoint_ema(self):
        branch = "matched_credit_rate_redistribution"
        record = _controller_record(branch)
        checkpoint = _controller_checkpoint(branch, record["credit_rate_ema"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_controller_artifacts(root, branch, record)
            with mock.patch(
                "credit_redistribution.evaluator._validate_telemetry",
                return_value={},
            ):
                result = validate_controller_artifacts(
                    root,
                    branch,
                    checkpoint,
                    start_step=FINAL_STEP,
                )
                self.assertEqual(len(result["controller_final_chain_digest"]), 64)

                mismatched = copy.deepcopy(checkpoint)
                mismatched[CHECKPOINT_STATE_KEY]["normalizer"]["ema"][0, 0] *= 1.01
                with self.assertRaisesRegex(RuntimeError, "ledger/checkpoint"):
                    validate_controller_artifacts(
                        root,
                        branch,
                        mismatched,
                        start_step=FINAL_STEP,
                    )

    def test_controller_validator_rejects_formula_tampering(self):
        branch = "matched_credit_rate_redistribution"
        base = _controller_record(branch)
        mutations = {
            "EMA recurrence": lambda record: record["credit_rate_ema"][0].__setitem__(
                0, record["credit_rate_ema"][0][0] * 1.01
            ),
            "raw-scale formula": lambda record: record["raw_scales"][0].__setitem__(
                0, record["raw_scales"][0][0] * 1.01
            ),
            "budget-factor formula": lambda record: record[
                "selected_budget_factors"
            ].__setitem__(0, record["selected_budget_factors"][0] * 1.01),
            "expert gradient scaling": lambda record: record[
                "post_gradient_squared_norm"
            ][0].__setitem__(
                0, record["post_gradient_squared_norm"][0][0] * 1.01
            ),
            "recorded full budget drift": lambda record: record.__setitem__(
                "full_relative_budget_drift", 1e-7
            ),
        }
        for expected_message, mutate in mutations.items():
            with self.subTest(expected_message=expected_message):
                record = copy.deepcopy(base)
                mutate(record)
                checkpoint = _controller_checkpoint(
                    branch,
                    record["credit_rate_ema"],
                )
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    _write_controller_artifacts(root, branch, record)
                    with mock.patch(
                        "credit_redistribution.evaluator._validate_telemetry",
                        return_value={},
                    ), self.assertRaisesRegex(RuntimeError, expected_message):
                        validate_controller_artifacts(
                            root,
                            branch,
                            checkpoint,
                            start_step=FINAL_STEP,
                        )

    def test_case_inventory_binds_exact_paths_and_metadata(self):
        protocol_sha256 = "c" * 64
        checkpoint_specs = {
            branch: {"path": f"/{branch}.pth", "sha256": f"{index + 1:x}" * 64}
            for index, branch in enumerate(BRANCHES)
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for branch in BRANCHES:
                for state_name in CHECKPOINT_STATES:
                    path = (
                        root
                        / "raw"
                        / branch
                        / state_name
                        / "case-000.json"
                    )
                    payload = _case_payload(
                        branch,
                        state_name,
                        checkpoint_specs[branch]["sha256"],
                    )
                    payload["protocol_sha256"] = protocol_sha256
                    _publish_case(path, payload, protocol_sha256)
            (
                case_hashes,
                seal_hashes,
                metric_hashes,
                metric_seal_hashes,
            ) = _validated_case_artifact_inventory(
                root,
                protocol_sha256,
                checkpoint_specs,
                case_count=1,
            )
            self.assertEqual(len(case_hashes), len(BRANCHES) * len(CHECKPOINT_STATES))
            self.assertEqual(len(seal_hashes), len(case_hashes))
            self.assertEqual(len(metric_hashes), len(case_hashes))
            self.assertEqual(len(metric_seal_hashes), len(case_hashes))

            unexpected = root / "raw" / "replacement.json"
            unexpected.write_text("{}\n", encoding="utf-8")
            os.chmod(unexpected, 0o444)
            with self.assertRaisesRegex(RuntimeError, "inventory differs"):
                _validated_case_artifact_inventory(
                    root,
                    protocol_sha256,
                    checkpoint_specs,
                    case_count=1,
                )

    def test_final_checkpoint_requires_complete_positive_controller_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pth"
            model = torch.nn.Linear(2, 1)
            ema_model = copy.deepcopy(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            optimizer.zero_grad()
            model(torch.ones(1, 2)).square().sum().backward()
            optimizer.step()
            counters = {
                "nonfinite": 0,
                "rank_disagreement": 0,
                "transcript_mismatch": 0,
                "budget_violation": 0,
                "capture_failure": 0,
                "checkpoint_failure": 0,
            }
            checkpoint = {
                "step": FINAL_STEP,
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "ema_model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in ema_model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "trainer_state": {
                    "version": 2,
                    "augmentation_seed_version": 1,
                    "global_seed": 0,
                    "sampler_contract": {
                        "version": 1,
                        "type": "distributed",
                        "global_seed": 0,
                        "per_rank_batch_size": 64,
                        "drop_last": False,
                        "case1_prob": None,
                        "dataset": {
                            "version": 1,
                            "type": "latent",
                            "num_samples": 1,
                            "ordered_samples_sha256": "a" * 64,
                        },
                    },
                    "world_size": 4,
                    "rank_states": [
                        {"rank": rank, "rng_state": _valid_rng_state()}
                        for rank in range(4)
                    ],
                    "next_step": FINAL_STEP + 1,
                    "data_batches_seen": FINAL_STEP + 1,
                    "sampler_epoch": FINAL_STEP + 1,
                    "sampler_batch_offset": 0,
                    "grad_mix": 1,
                    "batches_per_epoch": 1,
                },
                CHECKPOINT_STATE_KEY: {
                    "version": CONTROLLER_STATE_VERSION,
                    "branch": "measure_only_control",
                    "execution_mode": "continuation",
                    "block_indices": list(BLOCK_INDICES),
                    "num_experts": NUM_EXPERTS,
                    "start_step": 301001,
                    "last_step": FINAL_STEP,
                    "update_count": 20_000,
                    "numerical_counters": counters,
                    "normalizer": {
                        "ema": torch.ones(6, NUM_EXPERTS, dtype=torch.float64),
                        "initialized": torch.ones(6, NUM_EXPERTS, dtype=torch.bool),
                        "ema_decay": 0.99,
                        "epsilon": 1e-30,
                    },
                },
            }
            torch.save(checkpoint, path)
            loaded, digest = validate_branch_checkpoint(
                path, None, "measure_only_control", reference_model=model
            )
            self.assertEqual(loaded["step"], FINAL_STEP)
            self.assertEqual(len(digest), 64)

            incomplete = copy.deepcopy(checkpoint)
            incomplete["model_state_dict"] = {}
            torch.save(incomplete, path)
            with self.assertRaisesRegex(ValueError, "missing or empty"):
                validate_branch_checkpoint(
                    path, None, "measure_only_control", reference_model=model
                )

    def test_streaming_transcript_cross_checks_local_and_global_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            branch = "measure_only_control"
            transcript_root = root / "transcripts" / branch
            records = []
            for rank in range(4):
                record = build_step_record(
                    FINAL_STEP,
                    rank,
                    [f"0004/sample-{rank}.latent.npz"],
                    torch.tensor([4]),
                    _transcript_tensors(),
                )
                JsonlLedger(
                    transcript_root / f"rank-{rank:02d}.jsonl", FINAL_STEP
                ).append_or_verify(record)
                records.append(record)
            global_record = build_global_record(FINAL_STEP, records)
            global_ledger = JsonlLedger(
                transcript_root / "global.jsonl", FINAL_STEP
            )
            persisted = global_ledger.append_or_verify(global_record)
            with self.assertRaisesRegex(RuntimeError, "deterministic replay"):
                validate_branch_transcripts(root, branch, start_step=FINAL_STEP)


if __name__ == "__main__":
    unittest.main()
