import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from analyses import run_learning_credit_balance_cross_checkpoint as runner
from analyses.timestep_utility.credit_balance_probe import BLOCKS, SIGMAS


class CrossCheckpointRunnerTests(unittest.TestCase):
    def test_devices_are_exactly_locked(self):
        self.assertEqual(
            runner._parse_devices("cuda:4,cuda:5,cuda:6,cuda:7"),
            runner.LOCKED_DEVICES,
        )
        with self.assertRaises(Exception):
            runner._parse_devices("cuda:0,cuda:1,cuda:2,cuda:3")

    def test_protocol_is_idempotent_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            protocol = {"version": 1, "locked": True}
            path, digest = runner._write_or_validate_protocol(
                output_dir,
                protocol,
            )
            second_path, second_digest = runner._write_or_validate_protocol(
                output_dir,
                protocol,
            )
            self.assertEqual((path, digest), (second_path, second_digest))
            with self.assertRaises(RuntimeError):
                runner._write_or_validate_protocol(
                    output_dir,
                    {"version": 2, "locked": True},
                )

    def test_result_seal_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            result = {"value": 1}
            runner._publish_result(path, result, "a" * 64, "case:demo")
            self.assertEqual(
                runner._load_sealed_payload(
                    path,
                    "a" * 64,
                    "case:demo",
                ),
                result,
            )
            path.write_text(json.dumps({"value": 2}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "seal mismatch"):
                runner._load_sealed_payload(
                    path,
                    "a" * 64,
                    "case:demo",
                )

    def test_plumbing_publish_removes_all_efficacy_fields(self):
        result = {
            "includes_parameter_credit": False,
            "cells": [{
                "block_index": 1,
                "sigma": 0.2,
                "timestep": 200.0,
                "native_mse": 1.0,
                "statistics": {"secret": 1.0},
                "numerical_controls": self._controls(),
            }],
        }
        published = runner._result_for_publish(result, "plumbing")
        self.assertTrue(published["efficacy_hidden"])
        self.assertEqual(
            set(published["cells"][0]),
            runner.PLUMBING_CELL_KEYS,
        )

    def test_unbiased_argmax_difference_is_report_only(self):
        result = {
            "measurement_scope": "output",
            "cells": [{
                "numerical_controls": self._controls(
                    unbiased_argmax_mismatches=17,
                ),
            }],
        }
        safety = runner._numerical_safety(
            [result],
            expected_bias=True,
        )
        self.assertTrue(safety["passed"])
        self.assertEqual(safety["totals"]["unbiased_argmax_mismatches"], 17)

    def test_numerical_safety_rejects_nonfinite_controls(self):
        controls = self._controls()
        controls["max_abs_native_output_drift"] = float("nan")
        result = {
            "measurement_scope": "output",
            "cells": [{"numerical_controls": controls}],
        }
        with self.assertRaisesRegex(ValueError, "must be finite"):
            runner._numerical_safety([result], expected_bias=True)

    def test_count_stage_rejects_credit_field_leakage(self):
        case = {
            "split": "discovery",
            "id": "case-001",
            "label": 1,
            "seed": 7,
            "synset": "n00000001",
            "latent_relative": "n00000001/demo.latent.npz",
            "latent_sha256": "a" * 64,
        }
        controls = self._controls()
        controls.pop("nonfinite_token_credits")
        controls.pop("nonfinite_parameter_credits")
        result = {
            "cross_checkpoint_probe_version": runner.CROSS_CHECKPOINT_VERSION,
            "credit_balance_probe_version": runner.PROBE_VERSION,
            "protocol_sha256": "c" * 64,
            "batch_case": case,
            "checkpoint_role": "lossfree",
            "checkpoint_sha256": "d" * 64,
            "block_indices": list(BLOCKS),
            "sigmas": list(SIGMAS),
            "measurement_scope": "count",
            "includes_parameter_credit": False,
            "cells": [
                {
                    "block_index": block,
                    "sigma": sigma,
                    "timestep": sigma * 1000,
                    "statistics": {
                        "token_count": [10] * 12,
                        "active_experts": 12,
                    },
                    "numerical_controls": dict(controls),
                }
                for block in BLOCKS for sigma in SIGMAS
            ],
        }
        protocol = {
            "checkpoints": {"lossfree": {"sha256": "d" * 64}},
        }
        runner._validate_case_result(
            result,
            case,
            "discovery-count",
            "lossfree",
            protocol,
            "c" * 64,
        )
        result["cells"][0]["native_mse"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "leaks efficacy"):
            runner._validate_case_result(
                result,
                case,
                "discovery-count",
                "lossfree",
                protocol,
                "c" * 64,
            )

    def test_failed_summary_cannot_unlock_next_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            runner._publish_summary(
                output_dir,
                "plumbing",
                {"passed": False},
                "b" * 64,
            )
            with self.assertRaisesRegex(RuntimeError, "did not unlock"):
                runner._require_passed_summary(
                    output_dir,
                    "plumbing",
                    "b" * 64,
                )

    def test_latent_input_is_hashed_at_use_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "0001" / "sample.npz"
            path.parent.mkdir()
            path.write_bytes(b"locked")
            case = {
                "split": "discovery",
                "id": "case-001",
                "label": 1,
                "seed": 7,
                "synset": "n00000001",
                "latent_relative": "0001/sample.npz",
                "latent_sha256": runner.sha256_file(path),
            }
            protocol = {
                "manifest": {
                    "latent_root": str(root),
                    "cases": [dict(case)],
                },
            }
            self.assertEqual(
                runner._verify_latent_input(protocol, case),
                path,
            )
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "Locked latent changed"):
                runner._verify_latent_input(protocol, case)

    def test_lossfree_checkpoint_provenance_is_bound(self):
        expected = {
            "global_seed": 0,
            "world_size": 4,
            "global_batch_size": 256,
            "per_rank_batch_size": 64,
            "grad_mix": 1,
            "checkpoint_step": 200000,
            "dataset_identity": {
                "version": 1,
                "type": "__mp_main__.LatentFolder",
                "num_samples": 1281167,
                "ordered_samples_sha256": "a" * 64,
            },
        }
        trainer_state = {
            "version": 2,
            "augmentation_seed_version": 1,
            "global_seed": 0,
            "world_size": 4,
            "grad_mix": 1,
            "next_step": 200001,
            "data_batches_seen": 200001,
            "sampler_contract": {
                "version": 1,
                "global_seed": 0,
                "per_rank_batch_size": 64,
                "type": "distributed",
                "drop_last": False,
                "case1_prob": None,
                "dataset": {
                    "version": 1,
                    "type": "__mp_main__.LatentFolder",
                    "num_samples": 1281167,
                    "ordered_samples_sha256": "a" * 64,
                },
            },
            "batches_per_epoch": 5005,
            "sampler_epoch": 39,
            "sampler_batch_offset": 4806,
            "rank_states": [
                {"rank": rank, "rng_state": self._rng_state()}
                for rank in range(4)
            ],
        }
        provenance = runner._checkpoint_training_provenance(
            {"trainer_state": trainer_state},
            expected,
        )
        self.assertEqual(provenance["world_size"], 4)
        trainer_state["world_size"] = 8
        with self.assertRaisesRegex(RuntimeError, "world_size"):
            runner._checkpoint_training_provenance(
                {"trainer_state": trainer_state},
                expected,
            )

    def test_lossfree_checkpoint_rejects_inconsistent_sampler_position(self):
        checkpoint, expected = self._checkpoint_provenance_fixture()
        checkpoint["trainer_state"]["sampler_batch_offset"] += 1
        with self.assertRaisesRegex(RuntimeError, "sampler position"):
            runner._checkpoint_training_provenance(checkpoint, expected)

    def test_lossfree_checkpoint_rejects_malformed_dataset_identity(self):
        checkpoint, expected = self._checkpoint_provenance_fixture()
        checkpoint["trainer_state"]["sampler_contract"]["dataset"][
            "ordered_samples_sha256"
        ] = "A" * 64
        with self.assertRaisesRegex(RuntimeError, "ordered_samples_sha256"):
            runner._checkpoint_training_provenance(checkpoint, expected)

    def test_lossfree_checkpoint_rejects_different_dataset_identity(self):
        checkpoint, expected = self._checkpoint_provenance_fixture()
        checkpoint["trainer_state"]["sampler_contract"]["dataset"][
            "ordered_samples_sha256"
        ] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "locked latent dataset"):
            runner._checkpoint_training_provenance(checkpoint, expected)

    def test_lossfree_checkpoint_rejects_incomplete_rng_state(self):
        checkpoint, expected = self._checkpoint_provenance_fixture()
        del checkpoint["trainer_state"]["rank_states"][0]["rng_state"]["cuda"]
        with self.assertRaisesRegex(RuntimeError, "RNG provenance"):
            runner._checkpoint_training_provenance(checkpoint, expected)

    def test_lossfree_checkpoint_rejects_malformed_cuda_rng_state(self):
        checkpoint, expected = self._checkpoint_provenance_fixture()
        checkpoint["trainer_state"]["rank_states"][0]["rng_state"][
            "cuda"
        ] = torch.zeros(1, dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "RNG provenance"):
            runner._checkpoint_training_provenance(checkpoint, expected)

    def test_latent_dataset_identity_matches_training_producer(self):
        from train import _dataset_sampler_identity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for class_name, filenames in (
                ("0000", ("b.latent.npz", "a.latent.npz")),
                ("0007", ("c.latent.npz",)),
            ):
                class_dir = root / class_name
                class_dir.mkdir()
                for filename in filenames:
                    path = class_dir / filename
                    path.touch()
                    paths.append(str(path))

            dataset_type = type(
                "LatentFolder",
                (),
                {"__len__": lambda self: len(self.latent_paths)},
            )
            dataset_type.__module__ = "__mp_main__"
            dataset = dataset_type()
            dataset.latent_dir = str(root)
            dataset.latent_paths = sorted(paths)
            dataset.class_to_idx = {"0000": 0, "0007": 7}

            expected = _dataset_sampler_identity(dataset)
            self.assertEqual(runner._latent_dataset_identity(root), expected)

    def test_latent_dataset_identity_rejects_in_root_cache_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "latents"
            class_dir = root / "0000"
            class_dir.mkdir(parents=True)
            paths = []
            for filename in ("a.latent.npz", "b.latent.npz"):
                path = class_dir / filename
                path.touch()
                paths.append(str(path))
            cache_path = Path(directory) / "latent_paths_cache.txt"

            for cached_paths in (list(reversed(paths)), paths[:1]):
                with self.subTest(cached_paths=cached_paths):
                    cache_path.write_text(
                        "\n".join(cached_paths),
                        encoding="utf-8",
                    )
                    with (
                        mock.patch.object(
                            runner,
                            "LATENT_PATHS_CACHE",
                            cache_path,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError,
                            "cache differs from the complete disk inventory",
                        ),
                    ):
                        runner._latent_dataset_identity(root)

    def test_main_validates_formula_before_loading_checkpoints(self):
        events = []

        def validate_formula():
            events.append("formula")
            return {"passed": True}

        def load_checkpoint(*_args, **_kwargs):
            events.append("checkpoint")
            raise RuntimeError("stop after ordering check")

        with (
            mock.patch.object(sys, "argv", ["credit-gate", "--prepare-only"]),
            mock.patch.object(runner, "select_cases", return_value=[]),
            mock.patch.object(runner, "_load_base_protocol", return_value={}),
            mock.patch.object(runner, "load_runtime_cfg", return_value=object()),
            mock.patch.object(runner, "_verify_preregistrations", return_value={}),
            mock.patch.object(
                runner,
                "_validate_preregistered_run_inputs",
                return_value={},
            ),
            mock.patch.object(
                runner,
                "validate_exact_parameter_credit_formula",
                side_effect=validate_formula,
            ),
            mock.patch.object(
                runner,
                "_checkpoint_contract",
                side_effect=load_checkpoint,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "ordering check"):
                runner.main()
        self.assertEqual(events, ["formula", "checkpoint"])

    def test_parameter_stage_returns_discovery_credit_summary(self):
        cases = [
            {"split": "discovery", "id": f"case-{index:03d}"}
            for index in range(runner.PARAMETER_CASE_COUNT)
        ]
        count_balance = {"passed": True}

        def publish_summary(_output_dir, name, _payload, _protocol_sha256):
            return Path(f"{name}-summary.json")

        with (
            mock.patch.object(
                runner,
                "_require_passed_summary",
                return_value={"count_balance": count_balance},
            ),
            mock.patch.object(runner, "_run_stage_cases"),
            mock.patch.object(
                runner,
                "_load_stage_results",
                side_effect=lambda *args: [{"stage": args[2], "role": args[3]}],
            ),
            mock.patch.object(
                runner,
                "_load_base_results",
                return_value=[{"base": True}],
            ),
            mock.patch.object(
                runner,
                "aggregate_parameter_credit_validation",
                return_value={"passed": True},
            ),
            mock.patch.object(
                runner,
                "_numerical_safety",
                return_value={"passed": True},
            ),
            mock.patch.object(
                runner,
                "_publish_summary",
                side_effect=publish_summary,
            ),
            mock.patch.object(
                runner,
                "evaluate_count_balance",
                return_value=count_balance,
            ),
            mock.patch.object(
                runner,
                "evaluate_count_replay",
                return_value={"passed": True},
            ),
            mock.patch.object(
                runner,
                "aggregate_credit_balance",
                return_value={"passed": True},
            ),
            mock.patch("builtins.print"),
        ):
            summary_path, passed = runner._stage_parameter_validation(
                Path("/unused"),
                cases,
                Path("/unused-base"),
                runner.LOCKED_DEVICES,
                {"formula_validation": {"passed": True}},
                Path("/unused-protocol.json"),
                "a" * 64,
            )
        self.assertTrue(passed)
        self.assertEqual(summary_path, Path("discovery-credit-summary.json"))

    @staticmethod
    def _controls(unbiased_argmax_mismatches=0):
        return {
            "max_abs_native_output_drift": 0.0,
            "native_relative_mse_drift": 0.0,
            "route_mismatches": 0,
            "unbiased_argmax_mismatches": unbiased_argmax_mismatches,
            "max_abs_native_weight_drift": 0.0,
            "repeated_route_mismatches": 0,
            "max_abs_repeated_weight_drift": 0.0,
            "lossfree_bias_enabled": True,
            "nonfinite_token_credits": 0,
            "nonfinite_parameter_credits": 0,
        }

    @staticmethod
    def _rng_state():
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA RNG provenance requires CUDA")
        numpy_state = np.random.get_state()
        cuda_generator = torch.Generator(
            device=f"cuda:{torch.cuda.current_device()}"
        )
        cuda_generator.manual_seed(0)
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
            "cuda": cuda_generator.get_state(),
        }

    @classmethod
    def _checkpoint_provenance_fixture(cls):
        expected = {
            "global_seed": 0,
            "world_size": 4,
            "global_batch_size": 256,
            "per_rank_batch_size": 64,
            "grad_mix": 1,
            "checkpoint_step": 200000,
            "dataset_identity": {
                "version": 1,
                "type": "__mp_main__.LatentFolder",
                "num_samples": 1281167,
                "ordered_samples_sha256": "a" * 64,
            },
        }
        trainer_state = {
            "version": 2,
            "augmentation_seed_version": 1,
            "global_seed": 0,
            "world_size": 4,
            "grad_mix": 1,
            "next_step": 200001,
            "data_batches_seen": 200001,
            "batches_per_epoch": 5005,
            "sampler_epoch": 39,
            "sampler_batch_offset": 4806,
            "sampler_contract": {
                "version": 1,
                "global_seed": 0,
                "per_rank_batch_size": 64,
                "type": "distributed",
                "drop_last": False,
                "case1_prob": None,
                "dataset": {
                    "version": 1,
                    "type": "__mp_main__.LatentFolder",
                    "num_samples": 1281167,
                    "ordered_samples_sha256": "a" * 64,
                },
            },
            "rank_states": [
                {"rank": rank, "rng_state": cls._rng_state()}
                for rank in range(4)
            ],
        }
        return {"trainer_state": trainer_state}, expected


if __name__ == "__main__":
    unittest.main()
