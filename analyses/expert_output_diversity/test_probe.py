import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from analyses.expert_output_diversity.probe import (
    BASE_TRAIN_SOURCE_SHA256,
    BASE_CONFIG_STEM,
    DEFAULT_BLOCK_INDICES,
    DEFAULT_SIGMAS,
    EXPECTED_WORLD_SIZE,
    FORMAL_NUM_ANCHOR_TOKENS,
    REQUIRED_COMMON_SOURCE_PATHS,
    VARIANT_TRAIN_SOURCE_SHA256,
    VARIANT_CONFIG_STEM,
    _checkpoint_file_metadata,
    _checkpoint_trainer_contract,
    _validate_formal_case_records,
    _validate_inputs,
    _validate_paired_trainer_contracts,
    compare_case_records,
    compute_function_metrics,
    compute_native_pool_metrics,
    compute_route_metrics,
)


def _cell(
    case_id,
    base_scale,
    variant_scale,
    route_shift=0.0,
    mse_shift=0.0,
    variant_repulsion=0.15,
    variant_active_experts=12,
):
    function_names = (
        "output_rms",
        "expert_rms_cv",
        "pairwise_l2_rms",
        "normalized_pairwise_l2",
        "pairwise_cosine",
        "relative_expert_residual_rms",
        "normalized_effective_rank",
    )
    base_functions = {name: 1.0 for name in function_names}
    variant_functions = {name: 1.0 + variant_scale for name in function_names}
    base_functions["output_rms"] = 1.0
    variant_functions["output_rms"] = 1.0 + base_scale
    return {
        "case_id": case_id,
        "sigma": 0.5,
        "block_index": 3,
        "num_anchor_tokens": 4,
        "denoising_mse": {"base": 1.0, "variant": 1.0 + mse_shift},
        "base_hidden_functions": {
            "base": dict(base_functions),
            "variant": dict(variant_functions),
        },
        "variant_hidden_functions": {
            "base": dict(base_functions),
            "variant": dict(variant_functions),
        },
        "native_pool": {
            "base": {
                "num_active_experts": 12,
                "pooled_pairwise_l2": 10.0,
                "pooled_pairwise_l2_rms": 1.0,
                "pooled_repulsion_tau5": 0.2,
            },
            "variant": {
                "num_active_experts": variant_active_experts,
                "pooled_pairwise_l2": 12.0,
                "pooled_pairwise_l2_rms": 1.2,
                "pooled_repulsion_tau5": variant_repulsion,
            },
        },
        "route": {
            "base": {
                "normalized_entropy": 0.9,
                "maximum_share": 0.15,
                "count_gini": 0.1,
            },
            "variant": {
                "normalized_entropy": 0.9 + route_shift,
                "maximum_share": 0.15 - route_shift,
                "count_gini": 0.1 - route_shift,
            },
        },
    }


def _trainer_state(config_stem, run_id, payload_sha256):
    model_source = (
        "models/models_ProMoE_TC.py"
        if config_stem == BASE_CONFIG_STEM
        else "models/models_ProMoE_TC_expert_contra.py"
    )
    source_paths = REQUIRED_COMMON_SOURCE_PATHS | {model_source}
    if config_stem == BASE_CONFIG_STEM:
        source_paths = source_paths | {"models/phase_metric.py"}
    batches_seen = 50001
    batches_per_epoch = 5004
    sampler_epoch, sampler_batch_offset = divmod(
        batches_seen,
        batches_per_epoch,
    )
    visible_devices = (
        ["0", "1", "2", "3"]
        if config_stem == BASE_CONFIG_STEM
        else ["4", "5", "6", "7"]
    )
    return {
        "version": 2,
        "augmentation_seed_version": 1,
        "global_seed": 0,
        "world_size": EXPECTED_WORLD_SIZE,
        "grad_mix": 1,
        "next_step": 50001,
        "data_batches_seen": batches_seen,
        "run_id": run_id,
        "batches_per_epoch": batches_per_epoch,
        "sampler_epoch": sampler_epoch,
        "sampler_batch_offset": sampler_batch_offset,
        "sampler_contract": {
            "version": 1,
            "type": "distributed",
            "global_seed": 0,
            "per_rank_batch_size": 64,
            "drop_last": False,
            "case1_prob": None,
            "dataset": {
                "version": 1,
                "num_samples": 1281167,
                "ordered_samples_sha256": "a" * 64,
            },
        },
        "rank_states": [
            {"rank": rank, "rng_state": {}}
            for rank in range(EXPECTED_WORLD_SIZE)
        ],
        "training_provenance": {
            "version": 1,
            "strict": True,
            "git": {
                "commit": "b" * 40,
                "origin_repa_commit": "b" * 40,
                "status_clean": True,
                "origin_repa_divergence": "0\t0",
            },
            "config": {
                "version": 1,
                "basename": f"{config_stem}.yaml",
                "payload_sha256": payload_sha256,
            },
            "source_sha256": {
                path: (
                    BASE_TRAIN_SOURCE_SHA256
                    if path == "train.py" and config_stem == BASE_CONFIG_STEM
                    else VARIANT_TRAIN_SOURCE_SHA256
                    if path == "train.py"
                    else "c" * 64
                )
                for path in source_paths
            },
            "environment": {
                "python": "3.10.0",
                "python_executable": "/env/bin/python",
                "torch": "2.0.0",
                "numpy": "1.0.0",
                "cuda_runtime": "12.0",
                "devices": [f"cuda:{index}" for index in range(EXPECTED_WORLD_SIZE)],
                "cuda_visible_devices": visible_devices,
                "cuda_devices": {
                    f"cuda:{index}": {
                        "name": "Test GPU",
                        "compute_capability": [9, 0],
                        "total_memory_bytes": 1024,
                        "uuid": f"GPU-{visible_devices[index]}",
                    }
                    for index in range(EXPECTED_WORLD_SIZE)
                },
            },
        },
    }


class ExpertOutputDiversityTests(unittest.TestCase):
    def test_checkpoint_copy_must_match_canonical_file(self):
        with TemporaryDirectory() as directory:
            canonical = Path(directory) / "canonical.pth"
            local = Path(directory) / "local.pth"
            canonical.write_bytes(b"checkpoint-bytes")
            local.write_bytes(b"checkpoint-bytes")
            metadata = _checkpoint_file_metadata(canonical, local, "Test")
            self.assertEqual(
                metadata["canonical_sha256"],
                metadata["weights_sha256"],
            )

            local.write_bytes(b"wrong-checkpoint")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                _checkpoint_file_metadata(canonical, local, "Test")

    def test_native_pool_reports_single_expert_collapse(self):
        class CollapsedMoe:
            num_routed_experts = 3
            experts = torch.nn.ModuleList([
                torch.nn.Identity(),
                torch.nn.Identity(),
                torch.nn.Identity(),
            ])

        metrics = compute_native_pool_metrics(
            CollapsedMoe(),
            torch.randn(8, 4),
            torch.zeros(8, dtype=torch.long),
        )
        self.assertEqual(metrics["num_active_experts"], 1)
        self.assertEqual(metrics["pooled_pairwise_l2"], 0.0)
        self.assertEqual(metrics["pooled_repulsion_tau5"], 0.0)

    @patch(
        "analyses.expert_output_diversity.probe._verify_training_source_contract"
    )
    def test_checkpoint_contract_rejects_noninteger_batch_size_cleanly(
        self,
        _verify_manifest,
    ):
        payload_sha256 = "d" * 64
        trainer_state = _trainer_state(
            BASE_CONFIG_STEM,
            "base_run_identifier_0001",
            payload_sha256,
        )
        trainer_state["sampler_contract"]["per_rank_batch_size"] = None
        with self.assertRaisesRegex(ValueError, "sampler contract"):
            _checkpoint_trainer_contract(
                trainer_state,
                50000,
                BASE_CONFIG_STEM,
                payload_sha256,
            )

    @patch(
        "analyses.expert_output_diversity.probe._verify_training_source_contract"
    )
    def test_checkpoint_contract_requires_complete_source_manifest(
        self,
        _verify_manifest,
    ):
        payload_sha256 = "d" * 64
        trainer_state = _trainer_state(
            VARIANT_CONFIG_STEM,
            "variant_run_identifier_01",
            payload_sha256,
        )
        del trainer_state["training_provenance"]["source_sha256"]["train.py"]
        with self.assertRaisesRegex(ValueError, "source set"):
            _checkpoint_trainer_contract(
                trainer_state,
                50000,
                VARIANT_CONFIG_STEM,
                payload_sha256,
            )

    @patch(
        "analyses.expert_output_diversity.probe._verify_training_source_contract"
    )
    def test_paired_checkpoint_contracts_require_same_trajectory(
        self,
        _verify_manifest,
    ):
        payload_sha256 = "d" * 64
        base = _checkpoint_trainer_contract(
            _trainer_state(
                BASE_CONFIG_STEM,
                "base_run_identifier_0001",
                payload_sha256,
            ),
            50000,
            BASE_CONFIG_STEM,
            payload_sha256,
        )
        variant = _checkpoint_trainer_contract(
            _trainer_state(
                VARIANT_CONFIG_STEM,
                "variant_run_identifier_01",
                payload_sha256,
            ),
            50000,
            VARIANT_CONFIG_STEM,
            payload_sha256,
        )
        variant["trajectory"]["sampler_contract"]["dataset"][
            "ordered_samples_sha256"
        ] = "e" * 64
        with self.assertRaisesRegex(ValueError, "trajectory"):
            _validate_paired_trainer_contracts(base, variant)

    def test_input_validation_rejects_nonpositive_thread_count(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            base = directory / "base" / "ckpt_step_50000.pth"
            variant = directory / "variant" / "ckpt_step_50000.pth"
            base.parent.mkdir()
            variant.parent.mkdir()
            base.write_bytes(b"base")
            variant.write_bytes(b"variant")
            with self.assertRaisesRegex(ValueError, "num_threads"):
                _validate_inputs(
                    base,
                    variant,
                    None,
                    None,
                    DEFAULT_SIGMAS,
                    DEFAULT_BLOCK_INDICES,
                    ("cpu",),
                    FORMAL_NUM_ANCHOR_TOKENS,
                    0,
                    20000,
                    0,
                )

    def test_function_metrics_detect_scale_free_specialization(self):
        base = torch.tensor([
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ])
        specialized = torch.tensor([
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]],
        ])
        base_metrics = compute_function_metrics(base)
        specialized_metrics = compute_function_metrics(specialized)
        self.assertEqual(base_metrics["normalized_effective_rank"], 0.0)
        self.assertGreater(
            specialized_metrics["normalized_pairwise_l2"],
            base_metrics["normalized_pairwise_l2"],
        )
        self.assertGreater(
            specialized_metrics["relative_expert_residual_rms"],
            base_metrics["relative_expert_residual_rms"],
        )
        self.assertGreater(
            specialized_metrics["normalized_effective_rank"],
            base_metrics["normalized_effective_rank"],
        )

    def test_function_metrics_separate_norm_growth_from_direction_change(self):
        outputs = torch.randn(5, 4, 7)
        original = compute_function_metrics(outputs)
        scaled = compute_function_metrics(outputs * 3.0)
        self.assertAlmostEqual(
            original["normalized_pairwise_l2"],
            scaled["normalized_pairwise_l2"],
            places=6,
        )
        self.assertAlmostEqual(
            original["normalized_effective_rank"],
            scaled["normalized_effective_rank"],
            places=6,
        )
        self.assertAlmostEqual(
            scaled["output_rms"] / original["output_rms"],
            3.0,
            places=6,
        )

    def test_route_metrics_report_balance(self):
        balanced = compute_route_metrics(torch.arange(12).repeat(2), 12)
        collapsed = compute_route_metrics(torch.zeros(24, dtype=torch.long), 12)
        self.assertAlmostEqual(balanced["normalized_entropy"], 1.0)
        self.assertAlmostEqual(balanced["maximum_share"], 1 / 12)
        self.assertEqual(balanced["count_gini"], 0.0)
        self.assertEqual(collapsed["normalized_entropy"], 0.0)
        self.assertEqual(collapsed["maximum_share"], 1.0)
        self.assertGreater(collapsed["count_gini"], 0.9)

    def test_comparison_gate_passes_real_diversity_without_scale_cheat(self):
        cases = [
            {
                "case": {"id": f"case{index}"},
                "cells": [_cell(f"case{index}", 0.05, 0.10)],
            }
            for index in range(8)
        ]
        result = compare_case_records(
            cases,
            bootstrap_resamples=1000,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(all(
            check["passed"] for check in result["checks"].values()
        ))

    def test_comparison_gate_rejects_output_norm_inflation(self):
        cases = [
            {
                "case": {"id": f"case{index}"},
                "cells": [_cell(f"case{index}", 0.30, 0.10)],
            }
            for index in range(8)
        ]
        result = compare_case_records(
            cases,
            bootstrap_resamples=1000,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["output_scale_safety"]["passed"])

    def test_comparison_gate_requires_actual_repulsion_loss_to_fall(self):
        cases = [
            {
                "case": {"id": f"case{index}"},
                "cells": [
                    _cell(
                        f"case{index}",
                        0.05,
                        0.10,
                        variant_repulsion=0.25,
                    )
                ],
            }
            for index in range(8)
        ]
        result = compare_case_records(cases, bootstrap_resamples=1000)
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["checks"]["pooled_repulsion_objective"]["passed"]
        )

    def test_comparison_gate_rejects_single_active_variant_expert(self):
        cases = [
            {
                "case": {"id": f"case{index}"},
                "cells": [
                    _cell(
                        f"case{index}",
                        0.05,
                        0.10,
                        variant_active_experts=(1 if index == 0 else 12),
                    )
                ],
            }
            for index in range(8)
        ]
        result = compare_case_records(cases, bootstrap_resamples=1000)
        self.assertFalse(result["passed"])
        active_check = result["checks"]["active_expert_safety"]
        self.assertFalse(active_check["passed"])
        self.assertEqual(
            active_check["observed"]["minimum_variant_active_experts"],
            1,
        )

    @patch(
        "analyses.expert_output_diversity.probe._verify_training_source_contract"
    )
    def test_paired_checkpoint_contracts_reject_environment_drift(
        self,
        _verify_sources,
    ):
        payload_sha256 = "d" * 64
        base = _checkpoint_trainer_contract(
            _trainer_state(
                BASE_CONFIG_STEM,
                "base_run_identifier_0001",
                payload_sha256,
            ),
            50000,
            BASE_CONFIG_STEM,
            payload_sha256,
        )
        variant_state = _trainer_state(
            VARIANT_CONFIG_STEM,
            "variant_run_identifier_01",
            payload_sha256,
        )
        variant_state["training_provenance"]["environment"]["torch"] = "9.9.9"
        variant = _checkpoint_trainer_contract(
            variant_state,
            50000,
            VARIANT_CONFIG_STEM,
            payload_sha256,
        )
        with self.assertRaisesRegex(ValueError, "environments differ"):
            _validate_paired_trainer_contracts(base, variant)

    def test_exploratory_comparison_cannot_emit_formal_pass(self):
        cases = [
            {
                "case": {"id": f"case{index}"},
                "cells": [_cell(f"case{index}", 0.05, 0.10)],
            }
            for index in range(8)
        ]
        result = compare_case_records(
            cases,
            bootstrap_resamples=1000,
            formal=False,
        )
        self.assertEqual(result["decision_mode"], "exploratory")
        self.assertIsNone(result["passed"])

    def test_formal_case_grid_rejects_missing_cell(self):
        expected_cases = [
            {"id": f"case{index}", "split": "discovery"}
            for index in range(8)
        ]
        records = []
        for case in expected_cases:
            cells = [
                {
                    "sigma": sigma,
                    "block_index": block_index,
                    "num_anchor_tokens": FORMAL_NUM_ANCHOR_TOKENS,
                }
                for sigma in DEFAULT_SIGMAS
                for block_index in DEFAULT_BLOCK_INDICES
            ]
            records.append({"case": dict(case), "cells": cells})
        _validate_formal_case_records(records, expected_cases)
        records[0]["cells"].pop()
        with self.assertRaisesRegex(RuntimeError, "incomplete cell grid"):
            _validate_formal_case_records(records, expected_cases)


if __name__ == "__main__":
    unittest.main()
