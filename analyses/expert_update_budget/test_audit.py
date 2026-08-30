import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from analyses.expert_update_budget import audit as audit_module
from analyses.expert_update_budget.audit import (
    LOCKED_TRAINING_SOURCE_PATHS,
    _prepare_output_directory,
    _validate_training_provenance,
    analyze_checkpoint_interval,
    bind_optimizer_parameters,
    coefficient_of_variation,
    evaluate_gate,
    gini,
    parameter_pair_sums,
    routed_expert_parameter_groups,
    sha256_file,
    spearman_correlation,
    summarize_rank_persistence,
    verify_unchanged_file,
    _validate_manifest,
)
from analyses import run_expert_update_budget_audit as audit_entrypoint


class ExpertUpdateBudgetAuditTests(unittest.TestCase):
    def test_distribution_helpers_handle_equal_and_concentrated_values(self):
        self.assertEqual(coefficient_of_variation([3.0, 3.0, 3.0]), 0.0)
        self.assertEqual(gini([3.0, 3.0, 3.0]), 0.0)
        self.assertAlmostEqual(gini([0.0, 0.0, 3.0]), 2.0 / 3.0)
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(spearman_correlation([1, 1, 1], [1, 2, 3]))

    def test_optimizer_binding_uses_model_parameter_order_and_shapes(self):
        specs = self._specs(num_experts=2, include_unconditional=False)
        optimizer = self._optimizer(specs, step=11)
        bound = bind_optimizer_parameters(optimizer, specs, expected_optimizer_step=11)
        self.assertEqual(list(bound), [spec["name"] for spec in specs])
        self.assertEqual(bound[specs[1]["name"]]["parameter_id"], 1)

        optimizer["state"][1]["exp_avg"] = torch.ones(3, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "parameter contract"):
            bind_optimizer_parameters(optimizer, specs, expected_optimizer_step=11)

    def test_parameter_pair_reports_actual_motion_and_separate_adam_field(self):
        spec = {
            "name": "blocks.1.mlp.experts.0.up_proj.weight",
            "shape": (2,),
            "dtype": torch.float64,
            "numel": 2,
        }
        state = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.tensor([2.0, 4.0], dtype=torch.float64),
            "exp_avg_sq": torch.tensor([4.0, 16.0], dtype=torch.float64),
        }
        binding = {
            "spec": spec,
            "state": state,
            "group": {
                "lr": 0.1,
                "betas": (0.0, 0.0),
                "eps": 1e-12,
                "weight_decay": 0.0,
                "amsgrad": False,
            },
        }
        sums = parameter_pair_sums(
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([2.0, 4.0], dtype=torch.float64),
            binding,
            chunk_size=1,
        )
        self.assertAlmostEqual(sums["displacement_square_sum"], 5.0)
        self.assertAlmostEqual(sums["preconditioned_moment_square_sum"], 2.0)
        self.assertAlmostEqual(sums["adamw_update_field_square_sum"], 0.02)

    def test_routed_groups_exclude_the_unconditional_expert(self):
        specs = self._specs(num_experts=3, include_unconditional=False)
        model_state = {
            "blocks.1.mlp.cluster_centers": torch.ones(2, 4, dtype=torch.float64),
        }
        groups = routed_expert_parameter_groups(specs, model_state, [1], 2)
        self.assertEqual(set(groups), {(1, 0), (1, 1)})

    def test_interval_analysis_keeps_blocks_and_experts_separate(self):
        specs = self._specs(num_experts=3, include_unconditional=False)
        previous = self._checkpoint(specs, step=5, offsets=[0.0, 0.0, 0.0])
        current = self._checkpoint(specs, step=10, offsets=[1.0, 0.1, 7.0])
        result = analyze_checkpoint_interval(
            previous,
            current,
            specs,
            expected_blocks=[1],
            expected_experts_per_block=2,
            chunk_size=1,
        )
        experts = result["blocks"]["1"]["experts"]
        self.assertEqual([row["expert_index"] for row in experts], [0, 1])
        self.assertGreater(
            experts[0]["relative_displacement"],
            experts[1]["relative_displacement"],
        )

    def test_rank_persistence_tracks_expert_identity_within_each_block(self):
        first = self._interval([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        second = self._interval([2.0, 4.0, 6.0], [0.2, 0.4, 0.6])
        persistence = summarize_rank_persistence([first, second], [1])
        self.assertAlmostEqual(
            persistence["1"][
                "median_adjacent_interval_relative_displacement_spearman"
            ],
            1.0,
        )

        reversed_second = self._interval([6.0, 4.0, 2.0], [0.6, 0.4, 0.2])
        reversed_result = summarize_rank_persistence(
            [first, reversed_second], [1]
        )
        self.assertAlmostEqual(
            reversed_result["1"][
                "median_adjacent_interval_relative_displacement_spearman"
            ],
            -1.0,
        )

    def test_rank_persistence_rejects_a_partially_undefined_block(self):
        constant = self._interval([1.0, 1.0, 1.0], [0.1, 0.1, 0.1])
        ranked = self._interval([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        scaled = self._interval([2.0, 4.0, 6.0], [0.2, 0.4, 0.6])
        persistence = summarize_rank_persistence(
            [constant, ranked, scaled],
            [1],
        )["1"]
        self.assertEqual(persistence["expected_adjacent_interval_pairs"], 2)
        self.assertEqual(
            persistence["valid_adjacent_interval_relative_displacement_pairs"],
            1,
        )
        self.assertIsNone(
            persistence["median_adjacent_interval_relative_displacement_spearman"]
        )

    def test_gate_requires_effect_size_agreement_and_persistence(self):
        interval = self._gate_interval(
            displacement_gini=0.2,
            update_gini=0.2,
            within_correlation=0.8,
        )
        persistence = {
            "1": {
                "median_adjacent_interval_relative_displacement_spearman": 0.8,
            }
        }
        thresholds = {
            "minimum_interval_block_cells": 1,
            "minimum_cell_relative_displacement_gini": 0.08,
            "minimum_median_relative_displacement_gini": 0.08,
            "minimum_fraction_cells_above_displacement_effect_size": 0.67,
            "minimum_median_relative_adamw_update_gini": 0.08,
            "minimum_valid_within_cell_correlation_fraction": 0.9,
            "minimum_median_displacement_update_spearman": 0.3,
            "minimum_valid_block_persistence_fraction": 1.0,
            "minimum_median_adjacent_interval_displacement_spearman": 0.3,
            "minimum_fraction_blocks_with_positive_median_adjacent_spearman": 0.67,
        }
        self.assertTrue(
            evaluate_gate([interval], persistence, thresholds, [1])["passed"]
        )
        interval["blocks"]["1"]["correlations"][
            "relative_displacement_vs_relative_adamw_update_field"
        ] = -0.5
        failed = evaluate_gate([interval], persistence, thresholds, [1])
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["displacement_update_agreement"]["passed"])

    def test_seed_protocols_share_thresholds_before_seed1_finishes(self):
        manifest_dir = Path(__file__).parent / "manifests"
        with open(
            manifest_dir / "expert_update_budget_seed0_v1.json",
            encoding="utf-8",
        ) as handle:
            seed0 = json.load(handle)
        with open(
            manifest_dir / "expert_update_budget_seed1_v1.json",
            encoding="utf-8",
        ) as handle:
            seed1 = json.load(handle)
        _validate_manifest(seed0)
        _validate_manifest(seed1)
        self.assertEqual(seed0["gate_thresholds"], seed1["gate_thresholds"])
        self.assertEqual(seed0["expected"]["global_seed"], 0)
        self.assertEqual(seed1["expected"]["global_seed"], 1)

    def test_training_provenance_requires_the_complete_source_set(self):
        expected = {
            "training_git_commit": "a" * 40,
            "config_basename": "locked.yaml",
            "config_payload_sha256": "b" * 64,
            "world_size": 4,
            "gpu_ids": [0, 1, 2, 3],
        }
        cuda_devices = {
            f"cuda:{index}": {
                "name": "GPU",
                "compute_capability": [9, 0],
                "total_memory_bytes": 1,
                "uuid": f"GPU-{index}",
            }
            for index in range(4)
        }
        provenance = {
            "version": 1,
            "strict": True,
            "git": {
                "commit": expected["training_git_commit"],
                "origin_repa_commit": expected["training_git_commit"],
                "status_clean": True,
                "origin_repa_divergence": "0\t0",
            },
            "config": {
                "version": 1,
                "basename": expected["config_basename"],
                "payload_sha256": expected["config_payload_sha256"],
            },
            "source_sha256": {
                relative: "c" * 64 for relative in LOCKED_TRAINING_SOURCE_PATHS
            },
            "environment": {
                "python": "3.10",
                "python_executable": "/python",
                "torch": "2.0",
                "numpy": "1.0",
                "cuda_runtime": "12.0",
                "devices": [f"cuda:{index}" for index in range(4)],
                "cuda_visible_devices": ["0", "1", "2", "3"],
                "cuda_devices": cuda_devices,
            },
        }
        self.assertEqual(
            set(_validate_training_provenance(provenance, expected)),
            set(LOCKED_TRAINING_SOURCE_PATHS),
        )
        provenance["source_sha256"].pop(LOCKED_TRAINING_SOURCE_PATHS[-1])
        with self.assertRaisesRegex(ValueError, "source hash set"):
            _validate_training_provenance(provenance, expected)

    def test_file_hash_recheck_detects_mid_audit_change(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "input.bin"
            path.write_bytes(b"before")
            expected_sha256 = sha256_file(path)
            verify_unchanged_file(path, expected_sha256, "Input")
            path.write_bytes(b"after")
            with self.assertRaisesRegex(RuntimeError, "changed while"):
                verify_unchanged_file(path, expected_sha256, "Input")

    def test_output_directory_rejects_unknown_files_even_with_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            output_dir = (
                project_root
                / "analyses"
                / "archvied_analyses"
                / "audit"
            )
            output_dir.mkdir(parents=True)
            (output_dir / "unrelated.txt").write_text("keep", encoding="utf-8")
            with mock.patch.object(audit_module, "PROJECT_ROOT", project_root):
                with self.assertRaisesRegex(FileExistsError, "unknown entry"):
                    _prepare_output_directory(output_dir, overwrite=True)

    def test_cli_returns_nonzero_when_the_gate_fails(self):
        arguments = [
            "run_expert_update_budget_audit.py",
            "--manifest",
            "manifest.json",
            "--config",
            "config.yaml",
            "--checkpoint-dir",
            "checkpoints",
            "--output-dir",
            "output",
        ]
        with mock.patch.object(sys, "argv", arguments), mock.patch.object(
            audit_entrypoint,
            "run_audit",
            return_value={"gate": {"passed": False, "checks": {}}},
        ):
            with self.assertRaisesRegex(SystemExit, "1"):
                audit_entrypoint.main()

    @staticmethod
    def _specs(num_experts, include_unconditional=False):
        del include_unconditional
        return [
            {
                "name": f"blocks.1.mlp.experts.{expert}.up_proj.weight",
                "shape": (2,),
                "dtype": torch.float64,
                "numel": 2,
            }
            for expert in range(num_experts)
        ]

    @staticmethod
    def _optimizer(specs, step):
        return {
            "param_groups": [{
                "params": list(range(len(specs))),
                "lr": 0.1,
                "betas": (0.0, 0.0),
                "eps": 1e-12,
                "weight_decay": 0.0,
                "amsgrad": False,
                "maximize": False,
            }],
            "state": {
                index: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.full(spec["shape"], index + 1.0, dtype=spec["dtype"]),
                    "exp_avg_sq": torch.full(
                        spec["shape"], (index + 1.0) ** 2, dtype=spec["dtype"]
                    ),
                }
                for index, spec in enumerate(specs)
            },
        }

    @classmethod
    def _checkpoint(cls, specs, step, offsets):
        model_state = {
            "blocks.1.mlp.cluster_centers": torch.ones(2, 4, dtype=torch.float64),
        }
        for index, spec in enumerate(specs):
            model_state[spec["name"]] = torch.tensor(
                [1.0 + offsets[index], 2.0 + offsets[index]],
                dtype=spec["dtype"],
            )
        return {
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": cls._optimizer(specs, step=step + 1),
        }

    @staticmethod
    def _interval(displacements, updates):
        experts = [
            {
                "expert_index": index,
                "relative_displacement": displacement,
                "relative_adamw_update_field": update,
            }
            for index, (displacement, update) in enumerate(
                zip(displacements, updates)
            )
        ]
        return {"blocks": {"1": {"experts": experts}}}

    @staticmethod
    def _gate_interval(displacement_gini, update_gini, within_correlation):
        return {
            "blocks": {
                "1": {
                    "distributions": {
                        "relative_displacement": {"gini": displacement_gini},
                        "relative_adamw_update_field": {"gini": update_gini},
                    },
                    "correlations": {
                        "relative_displacement_vs_relative_adamw_update_field": (
                            within_correlation
                        ),
                    },
                }
            }
        }


if __name__ == "__main__":
    unittest.main()
