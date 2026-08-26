import copy
import unittest
from unittest.mock import patch

import numpy as np

from analyses.expert_function.batch import (
    BLOCK_INDEX,
    CASE_SPECS,
    CHECKPOINT_STATE,
    CHECKPOINT_STEP,
    EXACT_BATCH_SIZE,
    GATE_REQUIREMENTS,
    MODEL_NAME,
    NUM_ROUTED_EXPERTS,
    NUM_TOKEN_PROBES,
    SHIFTS,
    SIGMAS,
    _recompute_cell,
    _validate_requirements,
    build_gate_summary,
    validate_case_result,
)
from analyses.expert_function.consistency_probe import (
    ALL_METRICS,
    PRIMARY_METRIC,
    PROBE_VERSION,
    ROUTER_METRIC,
    summarize_token,
    summarize_tokens,
)


def _requirements(**updates):
    requirements = dict(GATE_REQUIREMENTS)
    requirements["bootstrap_resamples"] = 1000
    requirements.update(updates)
    return requirements


def _gate_rows(primary=-0.2, router=0.0, router_weight=-0.1):
    observations = []
    controls = []
    for case_index in range(len(CASE_SPECS)):
        case_id = f"case{case_index:02d}"
        primary_value = (
            float(primary(case_index)) if callable(primary) else float(primary)
        )
        for sigma in SIGMAS:
            for shift in SHIFTS:
                controls.append({
                    "case_id": case_id,
                    "max_abs_noop_mse_change": 0.0,
                    "max_abs_noop_output_change": 0.0,
                    "max_abs_forced_unforced_mse_change": 0.0,
                    "max_abs_forced_unforced_output_change": 0.0,
                })
                for token_index in range(NUM_TOKEN_PROBES):
                    observations.append({
                        "case_id": case_id,
                        "sigma": float(sigma),
                        "shift": list(shift),
                        "token_index": token_index,
                        "primary_spearman": primary_value,
                        "router_spearman": float(router),
                        "primary_minus_router_spearman": (
                            primary_value - float(router)
                        ),
                        "native_router_weight": float(router_weight),
                        "exact_mse_change_range": 1.0,
                        "primary_selected_beats_native": False,
                        "primary_oracle_top3": False,
                    })
    return observations, controls


def _case(case_id="case00"):
    return {
        "id": case_id,
        "label": 1,
        "seed": 7,
        "synset": "n00000001",
        "latent": "/latents/n00000001/example.latent.npz",
        "latent_relative": "n00000001/example.latent.npz",
        "latent_key": "latent",
        "latent_sha256": "1" * 64,
    }


def _cell(sigma, shift):
    tokens = []
    candidates = []
    exact = np.arange(NUM_ROUTED_EXPERTS - 1, -1, -1, dtype=np.float64)
    metric_scores = {
        metric: np.arange(1, NUM_ROUTED_EXPERTS + 1, dtype=np.float64)
        for metric in ALL_METRICS
    }
    native_expert = NUM_ROUTED_EXPERTS - 1
    for token_index in range(NUM_TOKEN_PROBES):
        content_source = token_index + 1
        token = summarize_token(metric_scores, exact, native_expert)
        token.update({
            "token_index": token_index,
            "content_source_index": content_source,
        })
        tokens.append(token)
        for expert in range(NUM_ROUTED_EXPERTS):
            candidates.append({
                "token_index": token_index,
                "content_source_index": content_source,
                "expert": expert,
                "is_native": expert == native_expert,
                "exact_mse_change": float(exact[expert]),
                "scores": {
                    metric: float(values[expert])
                    for metric, values in metric_scores.items()
                },
            })
    return {
        "sigma": float(sigma),
        "timestep": float(sigma * 1000),
        "shift_latent": list(shift),
        "shift_tokens": [shift[0] // 2, shift[1] // 2],
        "valid_tokens": 200,
        "sampled_tokens": NUM_TOKEN_PROBES,
        "shifted_native_mse": 1.0,
        "summary": summarize_tokens(tokens),
        "numerical_controls": {
            "max_abs_noop_mse_change": 0.0,
            "max_abs_noop_output_change": 0.0,
            "max_abs_forced_unforced_output_change": 0.0,
            "max_abs_forced_unforced_mse_change": 0.0,
        },
        "tokens": tokens,
        "candidates": candidates,
    }


def _case_result():
    case = _case()
    cells = [_cell(sigma, shift) for sigma in SIGMAS for shift in SHIFTS]
    all_tokens = [token for cell in cells for token in cell["tokens"]]
    checkpoint = "/outputs/checkpoints/ckpt_step_50000.pth"
    weights_checkpoint = "/local/base-seed0-ckpt_step_50000.pth"
    config = "/configs/004_ProMoE_B_seed0_control.yaml"
    checkpoint_sha256 = "2" * 64
    weights_sha256 = "3" * 64
    protocol_sha256 = "4" * 64
    result = {
        "expert_function_consistency_probe_version": PROBE_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "checkpoint": checkpoint,
        "weights_checkpoint": weights_checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "weights_checkpoint_sha256": weights_sha256,
        "checkpoint_step": CHECKPOINT_STEP,
        "weights_checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_state": CHECKPOINT_STATE,
        "config": config,
        "model_name": MODEL_NAME,
        "latent": case["latent"],
        "latent_key": case["latent_key"],
        "latent_sha256": case["latent_sha256"],
        "label": case["label"],
        "block_index": BLOCK_INDEX,
        "sigmas": list(SIGMAS),
        "shifts_latent": [list(shift) for shift in SHIFTS],
        "patch_size": 2,
        "num_token_probes_per_cell": NUM_TOKEN_PROBES,
        "exact_batch_size": EXACT_BATCH_SIZE,
        "seed": case["seed"],
        "device": "cuda:4",
        "num_threads": 8,
        "protocol_sha256": protocol_sha256,
        "batch_case": case,
        "cells": cells,
        "summary": summarize_tokens(all_tokens),
        "per_sigma": {
            str(float(sigma)): summarize_tokens([
                token
                for cell in cells
                if cell["sigma"] == float(sigma)
                for token in cell["tokens"]
            ])
            for sigma in SIGMAS
        },
        "per_shift": {
            f"{dy}:{dx}": summarize_tokens([
                token
                for cell in cells
                if cell["shift_latent"] == [dy, dx]
                for token in cell["tokens"]
            ])
            for dy, dx in SHIFTS
        },
    }
    expected_run = {
        "checkpoint": checkpoint,
        "weights_checkpoint": weights_checkpoint,
        "config": config,
        "device": "cuda:4",
        "num_threads": 8,
        "checkpoint_sha256": checkpoint_sha256,
        "weights_sha256": weights_sha256,
        "protocol_sha256": protocol_sha256,
    }
    return result, case, expected_run


class ExpertFunctionBatchTests(unittest.TestCase):
    def test_boolean_gate_switches_are_enforced(self):
        observations, controls = _gate_rows()
        disabled = _requirements(
            minimum_mean_primary_spearman=-1.0,
            minimum_positive_images=0,
            minimum_mean_primary_minus_router_spearman=-2.0,
            require_every_sigma_positive=False,
            require_primary_ci_lower_positive=False,
            require_delta_ci_lower_positive=False,
            require_positive_native_router_weight=False,
        )
        disabled_gate = build_gate_summary(observations, controls, disabled)
        self.assertTrue(disabled_gate["passed"])

        enabled = dict(disabled)
        enabled.update({
            "require_every_sigma_positive": True,
            "require_primary_ci_lower_positive": True,
            "require_delta_ci_lower_positive": True,
            "require_positive_native_router_weight": True,
        })
        enabled_gate = build_gate_summary(observations, controls, enabled)
        self.assertFalse(enabled_gate["passed"])
        self.assertFalse(enabled_gate["safety_checks"][
            "positive_native_router_weight"
        ]["passed"])
        self.assertFalse(enabled_gate["mechanism_checks"][
            "every_sigma_positive"
        ]["passed"])

    def test_gate_rejects_non_boolean_switch(self):
        requirements = _requirements(require_every_sigma_positive=1)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            _validate_requirements(requirements)

    def test_bootstrap_receives_one_cluster_mean_per_image(self):
        observations, controls = _gate_rows(
            primary=lambda case_index: case_index / 100.0,
            router=0.0,
            router_weight=0.5,
        )
        requirements = _requirements(
            minimum_mean_primary_spearman=-1.0,
            minimum_positive_images=0,
            minimum_mean_primary_minus_router_spearman=-2.0,
            require_every_sigma_positive=False,
            require_primary_ci_lower_positive=False,
            require_delta_ci_lower_positive=False,
        )
        captured = []

        def fake_bootstrap(values, resamples, seed):
            captured.append(np.asarray(values, dtype=np.float64))
            return [-1.0, 1.0]

        with patch(
            "analyses.expert_function.batch._bootstrap_ci",
            side_effect=fake_bootstrap,
        ):
            gate = build_gate_summary(observations, controls, requirements)
        self.assertEqual(gate["inference_unit"], "image")
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0].shape, (len(CASE_SPECS),))
        np.testing.assert_allclose(
            captured[0],
            np.arange(len(CASE_SPECS), dtype=np.float64) / 100.0,
        )

    def test_case_contract_is_recomputed_from_candidate_grid(self):
        result, case, expected_run = _case_result()
        observations, controls = validate_case_result(
            result,
            case,
            expected_run,
        )
        self.assertEqual(
            len(observations),
            len(SIGMAS) * len(SHIFTS) * NUM_TOKEN_PROBES,
        )
        self.assertEqual(len(controls), len(SIGMAS) * len(SHIFTS))

        tampered = copy.deepcopy(result)
        tampered["cells"][0]["tokens"][0]["native_expert"] = 0
        with self.assertRaisesRegex(ValueError, "token"):
            validate_case_result(tampered, case, expected_run)

        tampered = copy.deepcopy(result)
        tampered["protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "protocol_sha256"):
            validate_case_result(tampered, case, expected_run)

    def test_noop_control_must_match_native_candidate_change(self):
        cell = _cell(SIGMAS[0], SHIFTS[0])
        native = cell["tokens"][0]["native_expert"]
        candidate = next(
            row
            for row in cell["candidates"]
            if row["token_index"] == 0 and row["expert"] == native
        )
        candidate["exact_mse_change"] = 0.5
        token_candidates = [
            row for row in cell["candidates"] if row["token_index"] == 0
        ]
        token_candidates.sort(key=lambda row: row["expert"])
        exact = np.asarray([
            row["exact_mse_change"] for row in token_candidates
        ])
        scores = {
            metric: np.asarray([
                row["scores"][metric] for row in token_candidates
            ])
            for metric in ALL_METRICS
        }
        updated = summarize_token(scores, exact, native)
        updated.update({"token_index": 0, "content_source_index": 1})
        cell["tokens"][0] = updated
        cell["summary"] = summarize_tokens(cell["tokens"])
        with self.assertRaisesRegex(ValueError, "no-op MSE control"):
            _recompute_cell(cell, "case00")


if __name__ == "__main__":
    unittest.main()
