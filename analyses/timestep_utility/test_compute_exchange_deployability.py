import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from analyses import run_compute_exchange_deployability_gate as deployability_gate
from analyses.denoising_regret.io import write_json_atomic
from analyses.timestep_utility.compute_exchange_deployability import (
    DualLinearUtilityScorer,
    build_same_expert_exchange_candidates,
    build_scorer_features,
    normalize_counterfactual_targets,
    roll_counterfactual_correspondence,
    solve_exact_exchange,
    write_npz_atomic,
)
from analyses.timestep_utility.compute_exchange_deployability_batch import (
    ACTION_NAMES,
    RETROSPECTIVE_BLOCKS,
    SIGMAS,
    aggregate_retrospective,
    candidate_bank_sha256,
    combine_retrospective_reveal,
    fit_gate,
    select_retrospective_actions,
)
from analyses.timestep_utility.compute_exchange_deployability_fit import (
    FeatureDataset,
    load_feature_dataset,
    split_calibration_cases,
    train_dual_scorer,
)


class ComputeExchangeDeployabilityTests(unittest.TestCase):
    def test_locked_pre_reveal_inputs_do_not_parse_confirmatory_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir()
            canonical = root / "canonical" / "ckpt_step_1.pth"
            weights = root / "weights" / "ckpt_step_1.pth"
            config = root / "config.yaml"
            canonical.parent.mkdir()
            weights.parent.mkdir()
            canonical.write_bytes(b"canonical")
            weights.write_bytes(b"weights")
            config.write_text("model_name: ProMoE_TC_B\n", encoding="utf-8")
            source_protocol = source_root / "protocol.json"
            discovery_summary = source_root / "discovery-summary.json"
            confirmatory_summary = source_root / "confirmatory-summary.json"
            discovery_result = source_root / "discovery-result.json"
            discovery_seal = source_root / "discovery-result.json.seal.json"
            for path, payload in (
                (source_protocol, "{}"),
                (discovery_summary, "{}"),
                (confirmatory_summary, "{}"),
                (discovery_result, "{}"),
                (discovery_seal, "{}"),
            ):
                path.write_text(payload, encoding="utf-8")

            digest = deployability_gate.sha256_file
            protocol = {
                "checkpoint": {
                    "canonical_path": str(canonical),
                    "canonical_size": canonical.stat().st_size,
                    "canonical_sha256": digest(canonical),
                    "weights_path": str(weights),
                    "weights_size": weights.stat().st_size,
                    "weights_sha256": digest(weights),
                    "config": str(config),
                    "config_sha256": digest(config),
                    "step": 1,
                },
                "source_gate": {
                    "root": str(source_root),
                    "protocol": str(source_protocol),
                    "protocol_file_sha256": digest(source_protocol),
                    "discovery_summary_sha256": digest(discovery_summary),
                    "confirmatory_summary_sha256": digest(confirmatory_summary),
                    "cases": {
                        "discovery": [{
                            "id": "discovery-0",
                            "source_result": str(discovery_result),
                            "source_result_sha256": digest(discovery_result),
                            "source_seal": str(discovery_seal),
                            "source_seal_sha256": digest(discovery_seal),
                        }],
                        "confirmatory": [{"id": "confirmatory-0"}],
                    },
                },
            }
            with (
                mock.patch.object(
                    deployability_gate,
                    "resolve_config_from_checkpoint",
                    return_value=config,
                ),
                mock.patch.object(
                    deployability_gate,
                    "verify_source_gate",
                    side_effect=AssertionError("pre-reveal parsed source result"),
                ) as reveal_loader,
            ):
                deployability_gate._verify_locked_inputs(protocol)
            reveal_loader.assert_not_called()

    def test_sorted_action_round_trip_preserves_key_set_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actions.json"
            write_json_atomic(
                path,
                {"actions": {name: {} for name in ACTION_NAMES}},
            )
            reloaded = deployability_gate.load_json(path)["actions"]
            self.assertNotEqual(tuple(reloaded), ACTION_NAMES)
            deployability_gate._validate_action_names(reloaded)
            with self.assertRaisesRegex(RuntimeError, "action names"):
                deployability_gate._validate_action_names({"primary": {}})

    def test_evaluate_cannot_load_source_before_action_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = {"output_dir": directory}
            with (
                mock.patch.object(
                    deployability_gate,
                    "_verify_action_artifacts",
                    side_effect=RuntimeError("invalid actions"),
                ),
                mock.patch.object(
                    deployability_gate,
                    "_load_reveal_source",
                ) as reveal_loader,
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid actions"):
                    deployability_gate._run_evaluate(protocol, "digest", "/source")
            reveal_loader.assert_not_called()

            action_path = Path(directory) / "actions.json"
            with (
                mock.patch.object(
                    deployability_gate,
                    "_verify_action_artifacts",
                    return_value=({"cases": []}, action_path, "actions-digest"),
                ),
                mock.patch.object(
                    deployability_gate,
                    "_run_reveal",
                    side_effect=RuntimeError("reveal failed"),
                ),
                mock.patch.object(
                    deployability_gate,
                    "_load_reveal_source",
                ) as reveal_loader,
            ):
                with self.assertRaisesRegex(RuntimeError, "reveal failed"):
                    deployability_gate._run_evaluate(protocol, "digest", "/source")
            reveal_loader.assert_not_called()

    def test_forward_features_and_scorer_are_token_batch_invariant(self):
        torch.manual_seed(3)
        hidden = torch.randn(16, 8)
        router = torch.randn(16, 3)
        experts = router.argmax(dim=-1)
        sigmas = torch.linspace(0.2, 0.8, 16)
        tokens = torch.arange(16)
        features = build_scorer_features(
            hidden,
            router,
            experts,
            sigmas,
            tokens,
            sequence_length=16,
            include_hidden=True,
        )
        self.assertEqual(features.shape, (16, 8 + 1 + 3 + 3 + 6 + 5))
        self.assertTrue(torch.isfinite(features).all())

        model = DualLinearUtilityScorer(8, 3).eval()
        blocks = torch.full((16,), 1, dtype=torch.long)
        with torch.inference_mode():
            whole = model(hidden, router, experts, blocks, sigmas, tokens, 16)
            order = torch.tensor([7, 2, 14, 0, 11, 5, 9, 1, 15, 3, 6, 10, 4, 13, 8, 12])
            permuted = model(
                hidden[order],
                router[order],
                experts[order],
                blocks[order],
                sigmas[order],
                tokens[order],
                16,
            )
        torch.testing.assert_close(permuted, whole[order], rtol=0, atol=1e-7)

    def test_exact_three_state_solver_preserves_each_expert_count(self):
        native = np.repeat(np.arange(3), [4, 4, 1])
        scores = np.full((9, 2), 10.0)
        scores[[0, 4], 0] = -5.0
        scores[[1, 5], 1] = -4.0
        action = solve_exact_exchange(
            native,
            scores,
            quota=0.25,
            num_experts=4,
        )
        self.assertEqual(action["donors"], [0, 4])
        self.assertEqual(action["receivers"], [1, 5])
        self.assertEqual(action["experts"], [0, 1])
        self.assertEqual(action["native_pass_vector"], [4, 4, 1, 0])
        self.assertEqual(
            action["native_pass_vector"],
            action["candidate_pass_vector"],
        )

    def test_target_normalization_and_roll_stay_within_cell_expert(self):
        targets = np.arange(32, dtype=np.float64).reshape(16, 2)
        experts = np.tile(np.repeat([0, 1], 4), 2)
        cells = np.repeat([0, 1], 8)
        normalized = normalize_counterfactual_targets(targets, experts, cells)
        for cell in (0, 1):
            for expert in (0, 1):
                group = (cells == cell) & (experts == expert)
                np.testing.assert_allclose(normalized[group].mean(axis=0), 0, atol=1e-6)
        rolled = roll_counterfactual_correspondence(normalized, experts, cells)
        self.assertFalse(np.array_equal(rolled, normalized))
        for cell in (0, 1):
            for expert in (0, 1):
                group = (cells == cell) & (experts == expert)
                for head in (0, 1):
                    np.testing.assert_allclose(
                        np.sort(rolled[group, head]),
                        np.sort(normalized[group, head]),
                    )

    def test_feature_loader_rejects_targets_in_forward_only_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {
                "hidden": np.zeros((1, 4, 3), dtype=np.float16),
                "router_scores": np.zeros((1, 4, 2), dtype=np.float16),
                "native_experts": np.zeros((1, 4), dtype=np.int16),
                "native_weights": np.zeros((1, 4), dtype=np.float32),
                "donor_target": np.zeros((1, 4), dtype=np.float32),
                "receiver_target": np.zeros((1, 4), dtype=np.float32),
            }
            metadata = {
                "case_id": "case0",
                "split": "retrospective",
                "privileged_targets_present": False,
                "cells": [{"block_index": 1, "sigma": 0.2, "candidate_seed": 1}],
            }
            npz = root / "case.features.npz"
            meta = root / "case.metadata.json"
            write_npz_atomic(npz, arrays)
            write_json_atomic(meta, metadata)
            with self.assertRaisesRegex(ValueError, "privileged targets"):
                load_feature_dataset([
                    {"case_id": "case0", "npz": str(npz), "metadata": str(meta)}
                ], require_targets=False)

            del arrays["donor_target"]
            del arrays["receiver_target"]
            metadata["cells"][0]["native_mse"] = 0.5
            write_npz_atomic(npz, arrays)
            write_json_atomic(meta, metadata)
            with self.assertRaisesRegex(ValueError, "target-derived MSE"):
                load_feature_dataset([
                    {"case_id": "case0", "npz": str(npz), "metadata": str(meta)}
                ], require_targets=False)

    def test_retrospective_selection_is_target_free_and_exact_counted(self):
        torch.manual_seed(5)
        tokens = 16
        hidden_dim = 4
        num_experts = 2
        hidden = np.random.default_rng(5).normal(
            size=(tokens, hidden_dim)
        ).astype(np.float32)
        router = np.tile(np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32), (8, 1))
        experts = router.argmax(axis=1).astype(np.int64)
        dataset = FeatureDataset(
            hidden=hidden,
            router_scores=router,
            native_experts=experts,
            native_weights=router[np.arange(tokens), experts],
            block_indices=np.ones(tokens, dtype=np.int64),
            sigmas=np.full(tokens, 0.5, dtype=np.float32),
            token_indices=np.arange(tokens, dtype=np.int64),
            case_indices=np.zeros(tokens, dtype=np.int64),
            cell_ids=np.zeros(tokens, dtype=np.int64),
            case_ids=("case0",),
            cells=({
                "case_id": "case0",
                "block_index": 1,
                "sigma": 0.5,
                "candidate_seed": 17,
                "token_start": 0,
                "token_stop": tokens,
                "source_result": None,
            },),
            targets=None,
            sequence_length=tokens,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
        )
        models = {
            name: DualLinearUtilityScorer(hidden_dim, num_experts).eval()
            for name in ("primary", "router_context", "rolled_correspondence")
        }
        models["router_context"] = DualLinearUtilityScorer(
            hidden_dim,
            num_experts,
            include_hidden=False,
        ).eval()
        records = select_retrospective_actions(dataset, models, torch.device("cpu"))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(tuple(record["actions"]), ACTION_NAMES)
        self.assertNotIn("native_mse", record)
        self.assertNotIn("selected_gain", record)
        self.assertEqual(record["action_invariance_mismatch"], 0)
        for action in record["actions"].values():
            self.assertEqual(action["native_pass_vector"], action["candidate_pass_vector"])

    def test_reveal_combiner_uses_exact_action_gain_not_bank_selection(self):
        native = np.tile([0, 1], 8)
        candidates = build_same_expert_exchange_candidates(native, 2, 23)
        action = {
            "case_id": "case0",
            "block_index": 1,
            "sigma": 0.5,
            "candidate_ids": [row["id"] for row in candidates],
            "candidate_bank_sha256": candidate_bank_sha256(candidates),
            "primary_candidate_priority": np.linspace(0, 1, len(candidates)).tolist(),
            "route_id_sha256": "routes",
            "route_weight_sha256": "weights",
            "action_invariance_mismatch": 0,
            "logical_pass_counts_match": True,
        }
        source_records = [
            {**candidate, "exact_mse_change": -0.01 * (index + 1)}
            for index, candidate in enumerate(candidates)
        ]
        source = {
            "cells": [{
                "block_index": 1,
                "sigma": 0.5,
                "native_mse": 1.0,
                "records": source_records,
                "summary": {"selectors": {
                    "rolled_utility": {"selected_gain": -0.2},
                    "exact_oracle": {"selected_gain": 0.32},
                }},
            }],
        }
        controls = {
            "max_abs_noop_mse_change": 0.0,
            "max_abs_hook_mse_change": 0.0,
            "max_abs_single_vs_paired_native_mse_drift": 0.0,
            "max_abs_paired_native_mse_drift": 0.0,
            "max_abs_noop_output_change": 0.0,
            "max_abs_hook_output_change": 0.0,
            "max_abs_single_vs_paired_native_output_drift": 0.0,
            "max_abs_paired_native_output_drift": 0.0,
            "logical_count_mismatches": 0,
            "action_contract_mismatches": 0,
            "route_id_mismatches": 0,
            "route_weight_mismatches": 0,
        }
        reveal = [{
            "case_id": "case0",
            "cells": [{
                "block_index": 1,
                "sigma": 0.5,
                "native_mse": 1.0,
                "route_id_sha256": "routes",
                "route_weight_sha256": "weights",
                "action_results": {
                    name: {"selected_gain": 0.123 if name == "primary" else 0.01}
                    for name in ACTION_NAMES
                },
                "numerical_controls": controls,
            }],
        }]
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            write_json_atomic(source_path, source)
            records = combine_retrospective_reveal(
                [action],
                reveal,
                {"case0": str(source_path)},
            )
        self.assertEqual(records[0]["selected_gain"]["primary"], 0.123)
        self.assertTrue(records[0]["numerical_controls_passed"])
        self.assertTrue(records[0]["native_mse_consistent"])

        source["cells"][0]["native_mse"] = 2.0
        mismatched = combine_retrospective_reveal(
            [action],
            reveal,
            {"case0": source},
        )
        self.assertFalse(mismatched[0]["native_mse_consistent"])
        self.assertFalse(mismatched[0]["numerical_controls_passed"])

    def test_short_synthetic_fit_runs_with_image_holdout(self):
        generator = np.random.default_rng(9)
        case_ids = tuple(f"case{index:02d}" for index in range(24))
        tokens_per_case = 256
        total = len(case_ids) * tokens_per_case
        hidden = generator.normal(size=(total, 4)).astype(np.float16)
        router = generator.normal(size=(total, 2)).astype(np.float16)
        experts = np.tile(np.arange(tokens_per_case) % 2, len(case_ids)).astype(np.int64)
        cells = np.repeat(np.arange(len(case_ids)), tokens_per_case)
        targets = np.stack((
            hidden[:, 0].astype(np.float32),
            hidden[:, 1].astype(np.float32),
        ), axis=1)
        dataset = FeatureDataset(
            hidden=hidden,
            router_scores=router,
            native_experts=experts,
            native_weights=np.ones(total, dtype=np.float32),
            block_indices=np.ones(total, dtype=np.int64),
            sigmas=np.full(total, 0.2, dtype=np.float32),
            token_indices=np.tile(np.arange(tokens_per_case), len(case_ids)),
            case_indices=np.repeat(np.arange(len(case_ids)), tokens_per_case),
            cell_ids=cells,
            case_ids=case_ids,
            cells=tuple({
                "case_id": case_id,
                "candidate_seed": 100 + index,
                "token_start": index * tokens_per_case,
                "token_stop": (index + 1) * tokens_per_case,
            } for index, case_id in enumerate(case_ids)),
            targets=targets,
            sequence_length=tokens_per_case,
            hidden_dim=4,
            num_experts=2,
        )
        fit_ids, validation_ids = split_calibration_cases(case_ids)
        with (
            mock.patch(
                "analyses.timestep_utility.compute_exchange_deployability_fit.MAX_EPOCHS",
                2,
            ),
            mock.patch(
                "analyses.timestep_utility.compute_exchange_deployability_fit.MIN_EPOCHS",
                1,
            ),
            mock.patch(
                "analyses.timestep_utility.compute_exchange_deployability_fit.EARLY_STOPPING_PATIENCE",
                1,
            ),
            mock.patch(
                "analyses.timestep_utility.compute_exchange_deployability_fit.TRAIN_BATCH_SIZE",
                1024,
            ),
        ):
            model, summary = train_dual_scorer(
                dataset,
                fit_ids,
                validation_ids,
                "primary",
                "cpu",
            )
        self.assertIsInstance(model, DualLinearUtilityScorer)
        self.assertIn(summary["best_epoch"], (1, 2))
        self.assertGreater(summary["true_target_validation_concordance"], 0.5)

    def test_fit_gate_requires_hidden_and_correct_correspondence(self):
        summaries = {
            "primary": {"true_target_validation_concordance": 0.70},
            "router_context": {"true_target_validation_concordance": 0.60},
            "rolled_correspondence": {"true_target_validation_concordance": 0.50},
        }
        self.assertTrue(fit_gate(summaries)["passed"])
        summaries["primary"]["true_target_validation_concordance"] = 0.57
        self.assertFalse(fit_gate(summaries)["passed"])

    def test_retrospective_gate_uses_image_level_effects_and_safety(self):
        records = []
        for case_index in range(48):
            for block in RETROSPECTIVE_BLOCKS:
                for sigma in SIGMAS:
                    records.append({
                        "case_id": f"case{case_index:02d}",
                        "block_index": block,
                        "sigma": sigma,
                        "selected_gain": {
                            "primary": 0.001,
                            "exact_oracle": 0.002,
                            "random": 0.0,
                            "router_margin": 0.0,
                            "rolled_utility": 0.0,
                            "router_context": 0.0,
                            "rolled_correspondence": 0.0,
                        },
                        "candidate_concordance": 0.8,
                        "action_invariance_mismatch": 0,
                        "logical_pass_counts_match": True,
                        "numerical_controls_passed": True,
                    })
        gate = aggregate_retrospective(records, resamples=1000)
        self.assertTrue(gate["passed"])
        records[0]["action_invariance_mismatch"] = 1
        failed = aggregate_retrospective(records, resamples=1000)
        self.assertFalse(failed["safety_passed"])
        self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
